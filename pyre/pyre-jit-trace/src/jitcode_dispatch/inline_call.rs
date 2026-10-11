//! Inline-call tracing: resolving a user-defined callee and walking its
//! body inline instead of recording a residual call.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s
//! `opimpl_inline_call_*` / `opimpl_recursive_call` and the `MetaInterp`
//! inline-callee machinery. PyPy keeps these inside `pyjitpl.py`; the split
//! into this file is pyre-local navigability, not a PyPy file boundary.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Covers callee
//! recognition and inlinability checks, active-box / call-stack
//! reconstruction, callee register-bank allocation, the sub-jitcode walk
//! driver, the self-recursive `CALL_ASSEMBLER` fold, the inline user-call /
//! binop / compareop / exception-string specializers, and the
//! `dispatch_inline_call_*` per-shape dispatchers. The `inline_call_*`
//! opname arms stay in `handle` (mod.rs) and call into these.

use majit_jitcode::codewriter::jitcode::{DESCENT_ENTRY_LEN_SLOTS, DescentBlockerSummary};
use pyre_interpreter::locals_w;
use rustpython_wtf8::Wtf8;

use super::*;

#[derive(Clone, Copy)]
struct BoundMethodInline {
    method_op: OpRef,
    function: pyre_object::PyObjectRef,
    receiver: pyre_object::PyObjectRef,
}

/// The receiver and attribute name an inlined `__getattribute__` /
/// `__getattr__` call was resolved for, carried into the callee walk so an
/// `AttributeError` the hook raises can be given the `name` / `obj` pair
/// `enrich_attribute_error` (`error.rs`) fills in.
#[derive(Clone, Copy)]
struct AttributeErrorInlineContext {
    obj: OpRef,
    obj_concrete: pyre_object::PyObjectRef,
    name: OpRef,
    name_concrete: pyre_object::PyObjectRef,
}

/// Whether this concrete receiver takes PyPy's installed `__len__` shortcut
/// straight to a builtin layout's length body.
///
/// `typedef.py use_special_method_shortcut('__len__')` replaces the generic
/// lookup in `descroperation.py _len` for builtin implementations.  The
/// generated walk is execution-driven and therefore follows just that body,
/// while [`descent_blocker_summary`] deliberately joins the generic error and
/// override arms too.  Those arms contain formatting helpers after apparent
/// effects and make the static pre-scan reject a clean exact-builtin walk.
///
/// This predicate emits no IR and proves no later value.  It only admits the
/// generated body; that body still records the payload-class and promoted
/// `w_class` guards which keep the recording-time shortcut valid at runtime.
unsafe fn exact_builtin_len_shortcut_receiver(obj: pyre_object::PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let ob_type = unsafe { (*obj).ob_type };
    let exact_w_class = if std::ptr::eq(ob_type, &pyre_object::pyobject::LIST_TYPE) {
        // The admitted set is the strategies whose length read the generated
        // body already lowers.  `IntOrFloat` is `Integer`'s own read —
        // `live_len` sends both to `ll_list_int_length` — so it is admitted on
        // the same proof.  Bytes/Ascii and the range strategies read a
        // differently shaped nested storage, and exact `list` alone does not
        // prove that lowering.
        if !(unsafe { pyre_object::w_list_uses_int_storage(obj) }
            || unsafe { pyre_object::w_list_uses_int_or_float_storage(obj) }
            || unsafe { pyre_object::w_list_uses_float_storage(obj) }
            || unsafe { pyre_object::w_list_uses_object_storage(obj) }
            || unsafe { pyre_object::w_list_uses_empty_storage(obj) })
        {
            return false;
        }
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE)
    } else if std::ptr::eq(ob_type, &pyre_object::pyobject::TUPLE_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::pyobject::STR_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::bytesobject::BYTES_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::bytearrayobject::BYTEARRAY_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::setobject::SET_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::setobject::FROZENSET_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::functional::RANGE_TYPE)
        // `len_slot` forwards an exact dict and its views to `w_dict_len`.
        // That leaf stays `dont_look_inside`, so the walk records the call
        // and does not read strategy storage itself.
        || std::ptr::eq(ob_type, &pyre_object::pyobject::DICT_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::dictmultiobject::DICT_KEYS_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::dictmultiobject::DICT_VALUES_TYPE)
        || std::ptr::eq(ob_type, &pyre_object::dictmultiobject::DICT_ITEMS_TYPE)
    {
        pyre_object::pyobject::get_instantiate(unsafe { &*ob_type })
    } else if specialised_pair_kind(ob_type).is_some() {
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE)
    } else {
        return false;
    };
    std::ptr::eq(unsafe { (*obj).w_class }, exact_w_class)
}

/// Where an element of a `defs_w` tuple lives, and therefore what the trace
/// emits to read one.  `w_tuple_new` routes EVERY arity-2 tuple through
/// `makespecialisedtuple2` (`specialisedtupleobject.py`), so a callee
/// with exactly two defaulted parameters — an extremely ordinary signature —
/// never has an array-backed `defs_w` at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefaultsRepr {
    /// Array-backed `W_TupleObject`: the elements live in the `wrappeditems`
    /// block, the shape upstream's `defs_w?[*]` lowers to.
    ItemsBlock,
    /// `Cls_ii`: two inline machine ints.  `wraps[i]` is `wrapint`
    /// (`specialisedtupleobject.py`), which is what
    /// `_flat_pycall_defaults` already runs per call through
    /// `w_tuple_getitem`, so the emitted box is the same fresh box the
    /// interpreter would have made.
    PairInt,
    /// `Cls_oo`: two inline object slots, for which `wraps[i]` is the identity
    /// (`specialisedtupleobject.py`) — the field read IS the element.
    PairObject,
}

struct PositionalDefaultsInline {
    tuple: pyre_object::PyObjectRef,
    /// Shadow slot of `tuple` on the caller's long-lived argument roots.
    /// Element pins allocate and can move a young defaults tuple; the emit
    /// reloads this slot instead of the address captured before those pins.
    tuple_root: usize,
    repr: DefaultsRepr,
    /// `len(defs_w)` at resolve time.  A live re-read recomputes each
    /// element's index as `len - (nparams - param_index)` rather than
    /// requiring this value to hold (`function.py _flat_pycall_defaults`).
    len: usize,
    /// `(parameter index, tuple index, concrete value)`.
    values: Vec<(usize, usize, pyre_object::PyObjectRef)>,
}

/// The `ob_type` [`positional_defaults_for_inline`] matched to choose `repr`,
/// and therefore what a `GuardClass` has to re-prove before the reads that
/// `repr` selects run against a later tuple.
fn defaults_repr_class_addr(repr: DefaultsRepr) -> i64 {
    let ty: *const pyre_object::pyobject::PyType = match repr {
        DefaultsRepr::ItemsBlock => &pyre_object::pyobject::TUPLE_TYPE,
        DefaultsRepr::PairInt => &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE,
        DefaultsRepr::PairObject => &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
    };
    ty as i64
}

/// Read element `tuple_index` of a pair tuple whose class is already guarded
/// to `repr`.  For `Cls_oo` the `valueN` field is the element; for `Cls_ii`
/// the element is `wrapint` of the field (`specialisedtupleobject.py`
/// `wraps`), a fresh box the interpreter's `w_tuple_getitem` makes as well.
/// The second value is the concrete a boxed int was stamped with, which the
/// caller's argument concrete has to follow.
fn emit_pair_tuple_item<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    tuple_op: OpRef,
    repr: DefaultsRepr,
    tuple_index: usize,
    value: pyre_object::PyObjectRef,
) -> Result<(OpRef, Option<ConcreteValue>), DispatchError> {
    if repr == DefaultsRepr::PairObject {
        let descr = if tuple_index == 0 {
            crate::descr::specialised_tuple_oo_value0_descr()
        } else {
            crate::descr::specialised_tuple_oo_value1_descr()
        };
        let elem = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, tuple_op, descr);
        return Ok((elem, None));
    }
    debug_assert!(repr == DefaultsRepr::PairInt);
    let descr = if tuple_index == 0 {
        crate::descr::specialised_tuple_ii_value0_descr()
    } else {
        crate::descr::specialised_tuple_ii_value1_descr()
    };
    let raw = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, tuple_op, descr);
    let elem = unsafe { pyre_object::w_int_get_value(value) };
    let boxed = walker_box_int(ctx, op_pc, raw, elem)?;
    // `wrapint` emits a heap `NewWithVtable`, so the stamped concrete has to
    // be a heap pointer too — the walk-time `w_tuple_getitem` box may be a
    // tagged immediate.  The caller re-homes the argument's concrete onto
    // whatever `box_int_concrete` picked, or its seeding loop would re-stamp
    // the op with the other one.
    let concrete = box_int_concrete(elem, value as i64);
    ctx.trace_ctx.set_opref_concrete(boxed, concrete);
    let rehomed = match concrete {
        majit_ir::Value::Ref(gcref) => Some(ConcreteValue::Ref(
            gcref.as_usize() as pyre_object::PyObjectRef
        )),
        _ => None,
    };
    Ok((boxed, rehomed))
}

/// Pin every ref already stored in `concretes`, run `body`, then copy the
/// collector's forwarded slots back. `body` may allocate (`wrapint` does),
/// and it reads the forwarded pointers: the copy it was called with can
/// already have moved inside `pin_roots`.
fn with_arg_refs_pinned<T>(
    concretes: &mut [ConcreteValue],
    body: impl FnOnce(&mut [ConcreteValue]) -> T,
) -> T {
    let mut live = Vec::new();
    let mut slots = Vec::new();
    for (i, concrete) in concretes.iter().copied().enumerate() {
        if let ConcreteValue::Ref(obj) = concrete
            && !obj.is_null()
        {
            slots.push((i, live.len()));
            live.push(obj);
        }
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&live);
    let refresh = |concretes: &mut [ConcreteValue]| {
        for &(i, slot) in &slots {
            if let ConcreteValue::Ref(obj) = &mut concretes[i] {
                *obj = pyre_object::gc_roots::shadow_stack_get(base + slot);
            }
        }
    };
    refresh(concretes);
    let result = body(concretes);
    refresh(concretes);
    result
}

fn refresh_rooted_refs(concretes: &mut [ConcreteValue], slots: &[Option<usize>]) {
    for (concrete, slot) in concretes.iter_mut().zip(slots) {
        if let (ConcreteValue::Ref(obj), Some(slot)) = (concrete, *slot) {
            *obj = pyre_object::gc_roots::shadow_stack_get(slot);
        }
    }
}

fn refresh_varkw_concretes(extras: &mut [VarkwExtra], slots: &[Option<usize>]) {
    for (extra, slot) in extras.iter_mut().zip(slots) {
        if let (ConcreteValue::Ref(obj), Some(slot)) = (&mut extra.concrete, *slot) {
            *obj = pyre_object::gc_roots::shadow_stack_get(slot);
        }
    }
}

/// Publish `obj` on `roots` and point `slots[index]` at that word. A later
/// pin can move every earlier word, so both slices are re-read afterwards.
fn adopt_rooted_ref(
    roots: &pyre_object::gc_roots::RootScope,
    concretes: &mut [ConcreteValue],
    slots: &mut [Option<usize>],
    varkw: &mut [VarkwExtra],
    varkw_slots: &[Option<usize>],
    index: usize,
    obj: pyre_object::PyObjectRef,
) {
    let _ = roots.pin_root(obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if let Some(entry) = slots.get_mut(index) {
        *entry = Some(slot);
    }
    // A missing parameter is `Null`. Refresh only rewrites an existing
    // `Ref`, so the hole has to become one before the slots are re-read.
    if let Some(concrete) = concretes.get_mut(index) {
        *concrete = ConcreteValue::Ref(pyre_object::gc_roots::shadow_stack_get(slot));
    }
    refresh_rooted_refs(concretes, slots);
    refresh_varkw_concretes(varkw, varkw_slots);
}

/// The [`DefaultsRepr`] a tuple of this exact `ob_type` stores its elements
/// in, or `None` for a layout the walker has no element read for.
fn defaults_repr_of(ob_type: *const pyre_object::pyobject::PyType) -> Option<DefaultsRepr> {
    if std::ptr::eq(ob_type, &pyre_object::pyobject::TUPLE_TYPE) {
        Some(DefaultsRepr::ItemsBlock)
    } else if std::ptr::eq(
        ob_type,
        &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE,
    ) {
        Some(DefaultsRepr::PairInt)
    } else if std::ptr::eq(
        ob_type,
        &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
    ) {
        Some(DefaultsRepr::PairObject)
    } else {
        None
    }
}

/// `function.py:188-193,217-231` — which `defs_w` element fills each parameter
/// the call left unbound.  `defs_w` covers the LAST `len(defs_w)` parameters,
/// so parameter `p` takes `defs_w[p - (nparams - ndefaults)]`; a `missing`
/// parameter below that floor has no default and the call would raise, so the
/// whole inline declines.
///
/// `Cls_ff` is left out: its slots are inline `f64` and the walker has no
/// float-field read to pair with `wrapfloat` here, so a two-float defaults
/// tuple safely stays on the residual call path.
unsafe fn positional_defaults_for_inline(
    roots: &pyre_object::gc_roots::RootScope,
    callable: pyre_object::PyObjectRef,
    missing: &[usize],
    nparams: usize,
) -> Option<PositionalDefaultsInline> {
    if missing.is_empty() {
        return None;
    }
    let tuple = unsafe { pyre_interpreter::function_get_defaults(callable) };
    if tuple.is_null() {
        return None;
    }
    // The caller's argument roots outlive every later element pin. The local
    // scope below ends before those pins, so the tuple has to stay on `roots`.
    let _ = roots.pin_root(tuple);
    let tuple_root = pyre_object::gc_roots::shadow_stack_len() - 1;
    let tuple = pyre_object::gc_roots::shadow_stack_get(tuple_root);
    // The layout is what `ob_type` names.  The type test here only decides
    // which read to emit; keeping that decision valid for a later tuple is the
    // emitting half's job, through the shape guards it records.
    let repr = defaults_repr_of(unsafe { (*tuple).ob_type })?;
    let ndefaults = unsafe { pyre_object::w_tuple_len(tuple) };
    let first_defaulted = nparams.checked_sub(ndefaults)?;
    let item_roots = pyre_object::gc_roots::push_roots();
    let tuple_slot = item_roots.base();
    let _ = item_roots.pin_root(tuple);
    // The tuple pin occupies `tuple_slot`; boxed elements follow it.
    let values_base = tuple_slot + 1;
    let mut values = Vec::with_capacity(missing.len());
    for &param_index in missing {
        let tuple_index = param_index.checked_sub(first_defaulted)?;
        // `Cls_ii` / `Cls_ff` box a fresh int or float. Re-read the tuple and
        // pin each box before the next `w_tuple_getitem`.
        let tuple = item_roots.get(tuple_slot);
        let value = unsafe { pyre_object::w_tuple_getitem(tuple, tuple_index as i64) }?;
        let _ = item_roots.pin_root(value);
        values.push((param_index, tuple_index, value));
    }
    for (offset, entry) in values.iter_mut().enumerate() {
        entry.2 = item_roots.get(values_base + offset);
    }
    Some(PositionalDefaultsInline {
        tuple: pyre_object::gc_roots::shadow_stack_get(tuple_root),
        tuple_root,
        repr,
        len: ndefaults,
        values,
    })
}

/// One keyword-only parameter's binding: the namespace slot's stored
/// value-or-cell, and the value that unwraps to.
struct KwonlyDefaultInline {
    /// Where the entry sits in the mapping, kept so the emit can read it
    /// again once the version marker is installed.
    slot: usize,
    stored: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
}

/// What the record-time resolve proved about `Function.w_kw_defs`, carried to
/// the emit so it re-establishes the same shape instead of re-deriving it.
struct KwonlyDefaultsInline {
    mapping: pyre_object::PyObjectRef,
    strategy: *mut pyre_object::celldict::ModuleDictStrategy,
    values: Vec<KwonlyDefaultInline>,
}

/// The `w_kw_defs` entry `_match_signature` fills each keyword-only
/// parameter, in slot order.  A parameter the mapping holds no entry for has
/// no default at all and the call would raise `TypeError`, so the whole inline
/// declines rather than reproduce a raise.
///
/// A definition installs these through `function.py init_kwdefaults_dict`,
/// which builds the namespace mapping precisely so each entry is normally
/// constant.  That is what is read here: the slot's stored cell, baked, with
/// the mapping's version pinned.  A rebind, a `del`, a new key or a storage
/// switch all bump that version and revoke the constant, while an in-place
/// `f.__kwdefaults__[name] = ...` is absorbed by the cell and stays visible
/// because the fold reads the cell's field live.
///
/// An ordinary dict here is an assigned `f.__kwdefaults__` — the setter leaves
/// what it is given unconverted, matching `fset_func_kwdefaults` — and has no
/// version to pin, so it declines.
///
/// # Safety
/// `callable` must be the live `Function` and `w_code` its code object.
unsafe fn kwonly_defaults_for_inline(
    callable: pyre_object::PyObjectRef,
    w_code: *const (),
    arg_count: usize,
    kwonly_count: usize,
) -> Option<KwonlyDefaultsInline> {
    let dict = unsafe { pyre_interpreter::function_get_kwdefaults(callable) };
    if dict.is_null() {
        return None;
    }
    // The version lives on the strategy box, so that is what the pin names and
    // what has to exist.  An ordinary dict has none.
    let strategy = unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict) };
    // The emit records `record_quasiimmut_field` against this strategy box.
    // `QuasiImmutDescr.struct_ptr` is a GCREF `walk_const_ptr_refs` forwards,
    // the same way `quasiimmut.py self.struct` is.  Movability does not
    // decide the pin.
    if strategy.is_null() {
        return None;
    }
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    let varnames = unsafe { &(*raw).varnames };
    if varnames.len() < arg_count + kwonly_count {
        return None;
    }
    let mut out = Vec::with_capacity(kwonly_count);
    for slot in arg_count..arg_count + kwonly_count {
        let Some(cell_slot) =
            crate::state::module_dict_cell_slot_direct(dict, varnames[slot].as_str())
        else {
            return None;
        };
        let Some(stored) = crate::state::module_dict_cell_value_direct(dict, cell_slot) else {
            return None;
        };
        if stored.is_null() {
            return None;
        }
        let value = unsafe { pyre_object::celldict::unwrap_cell(stored) };
        if value.is_null() {
            return None;
        }
        out.push(KwonlyDefaultInline {
            slot: cell_slot,
            stored,
            value,
        });
    }
    Some(KwonlyDefaultsInline {
        mapping: dict,
        strategy,
        values: out,
    })
}

/// Path-1 (#68): resolve a scalar `getfield_vable_r` read off an inlined
/// callee's OWN (unseeded) portal frame to the callee's compile-time
/// constant. Only the Ref-typed `pycode` (field 1) remains a concrete
/// virtualizable field; globals is derived through PyFrame.get_w_globals.
/// Int frame state (`last_instr`, `valuestackdepth`)
/// does not.  Returns `None` when not an inline sub-walk, the field is not
/// resolvable, or the layout is absent — callers fall through to the
/// `VableBoxNotSeeded` error (such callees are declined up-front by
/// [`callee_fast_path_inlinable`], so reaching here unresolved is genuine).
pub(crate) fn try_resolve_inline_callee_static_field<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.inline_callee_consts.is_none() {
        return Ok(None);
    }
    let descr = read_descr(code, op, 1, ctx)?;
    let field_idx = {
        let Some(info) = ctx.trace_ctx.virtualizable_info() else {
            return Ok(None);
        };
        match info.static_field_by_descr(&descr) {
            Some(idx) => idx,
            None => return Ok(None),
        }
    };
    let const_ptr = match field_idx {
        VABLE_CODE_FIELD_IDX => ctx.inline_w_code(),
        _ => return Ok(None),
    };
    let result = ctx.trace_ctx.const_ref(const_ptr as i64);
    let dst = code[op.pc + 4] as usize;
    write_vable_field_ref_reg(
        ctx,
        op.pc,
        dst,
        result,
        ConcreteValue::Ref(const_ptr as pyre_object::PyObjectRef),
    )?;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

#[allow(non_snake_case)]
/// #62 call-inlining recognition probe (env-gated `PYRE_DIAG_INLINE_RECOG`,
/// ZERO behavior change — emits only diagnostics).  A user-function `CALL`
/// lowers to a `call_fn` residual whose `funcptr` is the generic call helper,
/// not the callee; the actual callable is a runtime Ref arg.  This scans the
/// Ref args' concrete values for a user Python function (`FUNCTION_TYPE`,
/// non-builtin) and reports whether its per-`CodeObject` JitCode is installed
/// (`jitcode_lookup`).  It confirms the runtime callable -> `CodeObject` ->
/// JitCode recognition seam fires before any inline sub-walk wiring lands.
pub(crate) fn diagnose_inline_recognition(arg_concretes: &[ConcreteValue], op_pc: usize) {
    let function_type_addr = &pyre_interpreter::FUNCTION_TYPE as *const _ as usize;
    // Single-letter kind tag per arg, so the call_fn arg layout (which slot
    // holds the callable vs the positional args) can be read off empirically
    // without touching CodeObject internals.
    let shape: String = arg_concretes
        .iter()
        .map(|cv| match cv {
            ConcreteValue::Int(_) => 'i',
            ConcreteValue::Float(_) => 'f',
            ConcreteValue::Bool(_) => 'b',
            ConcreteValue::Ref(_) => 'r',
            ConcreteValue::Null => '_',
        })
        .collect();
    for (i, cv) in arg_concretes.iter().enumerate() {
        let ConcreteValue::Ref(obj) = *cv else {
            continue;
        };
        if obj.is_null() {
            continue;
        }
        unsafe {
            if !pyre_interpreter::is_function(obj) {
                continue;
            }
            // Only the pure-Python `function` type is a sub-walk candidate;
            // builtins share `is_function` but have no per-fn JitCode.
            if (*obj).ob_type as *const () as usize != function_type_addr {
                continue;
            }
            let code = pyre_interpreter::function_get_code(obj);
            // Exercise the slice-(3) obtain step: build (if needed) + view the
            // callee per-fn JitCode as a SubJitCodeBody.  Reports the callee
            // register-bank shape the sub-walk will allocate.
            match crate::state::sub_jitcode_body_for_code(code) {
                Some(body) => eprintln!(
                    "[inline-recog] pc={op_pc} nargs={} shape=[{shape}] callable@{i} \
                     code={code:?} body-OK regs_r={} regs_i={} regs_f={} code_len={}",
                    arg_concretes.len(),
                    body.num_regs_r,
                    body.num_regs_i,
                    body.num_regs_f,
                    body.code.len()
                ),
                None => eprintln!(
                    "[inline-recog] pc={op_pc} nargs={} shape=[{shape}] callable@{i} \
                     code={code:?} body-NONE",
                    arg_concretes.len()
                ),
            }
        }
    }
}

/// The FBW fast-path inline convention (`try_walker_inline_user_call`) first
/// checks whether the callee body can be followed, before the full inline path
/// seeds the callee's positional arguments and its own frame/ec reds. A callee
/// whose frame shape cannot be materialized must lower to an ordinary residual
/// call (the orthodox non-inlinable path, `should_inline` = False →
/// `do_residual_call`, `pyjitpl.py`) instead of entering with an empty frame red.
///
/// The predicate still distinguishes straight-line leaves from branch-bearing
/// bodies because the latter enter the multi-frame branch walk. Both successful
/// paths now snapshot the guard at the callee's coordinate with a paused caller
/// frame; neither collapses resume to the caller's CALL boundary.
pub(crate) fn callee_fast_path_inlinable<Sym: WalkSym>(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    ctx: &WalkContext<'_, '_, Sym>,
    callee_frame_reg: u16,
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            // Undecodable tail — be conservative and decline the fast path.
            return false;
        };
        if d.opname.starts_with("goto_if_not") || d.opname.starts_with("switch") {
            if fbw_strict_diag_enabled() {
                eprintln!("[strict-reject] pc={} op={} (branch)", d.pc, d.opname);
            }
            return false;
        }
        // A vable op is inlinable on the strict straight-line path when it is
        // either a static const-field read (`pycode` / `w_globals`, resolved
        // frame-free) OR a read/write off the callee's OWN portal frame
        // register — the latter is the `fresh_virtualizable` case, folded
        // register-to-register through the per-slot OpRef shadow by the two
        // `*_vable_via_metainterp` short-circuits (no GC op emitted).  Under
        // the unconditional Portal input shape, LOAD_FAST / STORE_FAST lower to
        // `getarrayitem_vable_r` / `setarrayitem_vable_r(frame, slot)`, so a
        // branchless leaf's locals prologue must not decline the fast path.
        // `callee_frame_reg == u16::MAX` for shapeless skeletons makes
        // `inline_resolvable_seeded_frame_op` return false, so this is inert
        // outside drained per-code jitcodes.
        if d.opname.contains("vable")
            && !inline_resolvable_static_vable_read(body_code, &d, callee_descr_refs, ctx)
            && !inline_resolvable_seeded_frame_op(body_code, &d, callee_frame_reg)
        {
            if fbw_strict_diag_enabled() {
                eprintln!(
                    "[strict-reject] pc={} op={} (non-static vable)",
                    d.pc, d.opname
                );
            }
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// True iff `d` is a scalar `getfield_vable_r` whose field is a Ref-typed
/// compile-time constant (`pycode` / `w_globals`) — the only vable op
/// [`try_resolve_inline_callee_static_field`] can satisfy without a seeded
/// frame box.  `setfield_vable`, array vable ops, and `getfield_vable_i/f`
/// (mutable Int/Float frame state) all return false → decline the inline.
pub(crate) fn inline_resolvable_static_vable_read<Sym: WalkSym>(
    body_code: &[u8],
    d: &DecodedOp,
    callee_descr_refs: &[DescrRef],
    ctx: &WalkContext<'_, '_, Sym>,
) -> bool {
    if !d.opname.starts_with("getfield_vable_r") {
        return false;
    }
    let Some(info) = ctx.trace_ctx.virtualizable_info() else {
        return false;
    };
    // `rd>r` layout: 1B reg + 2B descr-pool index + 1B dst; descr at `pc + 2`.
    if d.pc + 3 >= body_code.len() {
        return false;
    }
    let descr_index = body_code[d.pc + 2] as usize | ((body_code[d.pc + 3] as usize) << 8);
    let Some(descr) = callee_descr_refs.get(descr_index) else {
        return false;
    };
    matches!(
        info.static_field_by_descr(descr),
        Some(VABLE_CODE_FIELD_IDX)
    )
}

/// Relaxed variant of [`callee_fast_path_inlinable`] for the multi-frame
/// inline path (#68): a FORWARD `goto_if_not`
/// (branch target ahead of the branch op) is now inlinable because its
/// in-callee guard resumes through a multi-frame snapshot
/// ([`walker_capture_snapshot_for_last_guard_impl`]'s parent-frame branch).
/// A BACKWARD `goto_if_not` (a loop back-edge) and any `switch` still decline:
/// a loop in the callee needs a `jit_merge_point` the inline snapshot does
/// not model, and a multi-target switch is not yet handled.
///
/// Vable reads are accepted in two cases: (a) a scalar static-field read
/// (`pycode` / `w_globals`) resolvable without a seeded frame
/// (`inline_resolvable_static_vable_read`), or (b) a frame-LOCAL read
/// (`getfield_vable_r` / `getarrayitem_vable_r`) whose base register equals the
/// seeded callee frame reg `callee_frame_reg` — the multiframe path seeds that
/// frame as a virtual object graph (`emit_new_pyframe_inline_with_params`), so
/// the optimizer folds the read to the seeded param value rather than aborting
/// `VableBoxNotSeeded`.  Every `setfield_vable_*` / `setarrayitem_vable_*`
/// (a write into a vable, which would escape the virtual frame) and any vable
/// op against a base reg OTHER than the seeded frame still decline.
pub(crate) fn callee_fast_path_inlinable_allowing_forward_branch<Sym: WalkSym>(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    ctx: &WalkContext<'_, '_, Sym>,
    callee_frame_reg: u16,
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            if fbw_strict_diag_enabled() {
                eprintln!("[strict-reject-mf] pc={pc} (undecodable)");
            }
            return false;
        };
        if d.opname.starts_with("switch") {
            if fbw_strict_diag_enabled() {
                eprintln!("[strict-reject-mf] pc={} op={} (switch)", d.pc, d.opname);
            }
            return false;
        }
        if d.opname.starts_with("goto_if_not") {
            // Label is the last operand (`branch.rs` `goto.argcodes.len() - 1`,
            // `label_operand_offset`).  A fused compare (`goto_if_not_int_lt/iiL`)
            // puts two Int regs before `L`; reading offset 1 takes the second
            // register as the label and treats a forward `if k < 0` as a
            // back-edge, so `try_multiframe` refuses the callee.
            // `perform_call` (`pyjitpl.py`) has no such screen.
            let Some(off) = label_operand_offset(d.argcodes) else {
                return false;
            };
            let target = read_label(body_code, &d, off);
            if target <= d.pc {
                if fbw_strict_diag_enabled() {
                    eprintln!(
                        "[strict-reject-mf] pc={} op={} target={target} (backward branch)",
                        d.pc, d.opname
                    );
                }
                return false;
            }
        }
        if d.opname.contains("vable")
            && !inline_resolvable_static_vable_read(body_code, &d, callee_descr_refs, ctx)
            && !inline_resolvable_seeded_frame_op(body_code, &d, callee_frame_reg)
        {
            if fbw_strict_diag_enabled() {
                eprintln!(
                    "[strict-reject-mf] pc={} op={} base_reg={:?} frame_reg={callee_frame_reg} \
                     (non-static, foreign vable)",
                    d.pc,
                    d.opname,
                    body_code.get(d.pc + 1).copied()
                );
            }
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// True iff `d` is a frame-LOCAL vable op — a `getfield_vable` /
/// `getarrayitem_vable` read OR a `setfield_vable` / `setarrayitem_vable` write
/// — whose base register byte (`body_code[d.pc + 1]`, the first operand for
/// every `rX...` / `riX...` vable layout) equals the seeded callee frame
/// register.  The multiframe inline seeds this frame as a VIRTUAL `PyFrame`
/// (`emit_new_pyframe_inline_with_params`) whose locals array holds the param
/// boxes; the post-trace optimizer keeps the frame + its array virtual, folding
/// reads to the seeded/stored value (`optimize_getfield_gc` /
/// `optimize_getarrayitem_gc`) and recording writes into `vinfo.items` /
/// `vinfo.fields` WITHOUT forcing (`optimize_setarrayitem_gc` /
/// `optimize_setfield_gc`).  A store INTO the callee's own virtual frame
/// (param-init `STORE_FAST` prologue, intermediate local writes) is therefore
/// foldable too — only a vable op against a DIFFERENT base reg (a genuinely
/// foreign vable, e.g. the caller's frame the seed does not own) escapes the
/// fold and must decline. A malformed non-Ref-base vable op is never treated
/// as frame-local; the assembler rejects that shape because every upstream
/// `BlackholeInterpreter.bhimpl_*_vable_*` base argument is `r`.
pub(crate) fn inline_resolvable_seeded_frame_op(
    body_code: &[u8],
    d: &DecodedOp,
    callee_frame_reg: u16,
) -> bool {
    if callee_frame_reg == u16::MAX || callee_frame_reg > u8::MAX as u16 {
        return false;
    }
    // Only ref-base vable ops (`getfield_vable_r/rd>r`,
    // `getarrayitem_vable_r/ridd>r`, `setfield_vable_*/rXd`,
    // `setarrayitem_vable_*/riXdd`) carry the frame ref in operand 0.  The
    // Any malformed non-Ref-base vable bytecode is not the seeded frame and is
    // rejected here as a second line of defence behind the assembler.
    let is_frame_vable = d.opname.starts_with("getfield_vable_r")
        || d.opname.starts_with("getarrayitem_vable_r")
        || (d.opname.starts_with("setfield_vable") && d.argcodes.starts_with('r'))
        || (d.opname.starts_with("setarrayitem_vable") && d.argcodes.starts_with('r'));
    if !is_frame_vable {
        return false;
    }
    match body_code.get(d.pc + 1).copied() {
        Some(base) => base as u16 == callee_frame_reg,
        None => false,
    }
}

/// True iff `body_code` contains a `raise` op.  A raise from a callee inlined
/// BELOW another inlined frame has to cross the suspended intermediate
/// frame(s) through the cross-frame exception-unwind bridge (gh#343 / gh#467),
/// so the depth is bounded rather than unbounded: the call site caps a raising
/// callee at THREE distinct multiframe levels, not at the top level — see
/// `fbw_effective_multiframe_depth`.  A recursive / duplicate raising chain is
/// bounded by `max_unroll_recursion` alone.
/// Straight value-returning chains never raise, so they still inline to the
/// full `fbw_max_multiframe_depth`.
pub(crate) fn callee_body_contains_raise(body_code: &[u8]) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname == "raise" {
            return true;
        }
        pc = d.next_pc;
    }
    false
}

/// True iff `body_code` carries a `jit_merge_point`, i.e. the callee owns a
/// loop header of its own.
///
/// Reaching one during an inline sub-walk surfaces
/// `SubLoopCalleeCallAssembler`, the one arm of
/// [`try_walker_inline_resolved_user_call`] that consumes the residual's
/// `funcptr` / `r_args` / `call_descr` — and it consumes them as a CALL's
/// `[callable, null_or_self, args…]` operand list.  A route that enters with a
/// callee it resolved itself (rather than off a CALL) carries a differently
/// shaped operand list, so it declines such a body up front.
///
/// A loop header is also where a deferred `load_method_self`'s two
/// operand-stack entries can be crossed by a failing guard.
pub(crate) fn callee_body_owns_loop_header(body_code: &[u8]) -> bool {
    crate::jitcode_runtime::decoded_ops(body_code).any(|op| op.opname == "jit_merge_point")
}

/// Whether running a straight-line callee body before deciding to keep it can
/// have no app-visible effect: no branch, no live-heap write, and no residual
/// outside `pure_helpers`.  Portal-frame vable traffic and constant/int boxing
/// are local, so both callers pass those.
fn flags_field_descr(callee_descr_refs: &[DescrRef], descr_index: usize) -> bool {
    callee_descr_refs
        .get(descr_index)
        .and_then(|descr| descr.as_field_descr())
        .is_some_and(|field| field.offset() == crate::frame_layout::PYFRAME_FLAGS_OFFSET)
}

/// `RETURN_VALUE` stores `frame_finished_execution` as
/// `getfield flags; int_or FLAG_FRAME_FINISHED; setfield flags`
/// immediately before `ref_return`. The bit is sticky, so sampling the
/// body and then replaying the real call ORs the same bit.
fn frame_finished_return_store(
    body_code: &[u8],
    op: &crate::jitcode_runtime::DecodedOp,
    callee_descr_refs: &[DescrRef],
    vable_reg: Option<u8>,
    finished_or: Option<(u8, u8)>,
) -> bool {
    let Some((frame, value_reg)) = finished_or else {
        return false;
    };
    if !op.argcodes.starts_with('r') {
        return false;
    }
    let Some(&target_reg) = body_code.get(op.pc + 1) else {
        return false;
    };
    let descr = super::decode_descr_index(body_code, op, 2);
    vable_reg == Some(frame)
        && target_reg == frame
        && body_code.get(op.pc + 2).copied() == Some(value_reg)
        && flags_field_descr(callee_descr_refs, descr)
        && crate::jitcode_runtime::decode_op_at(body_code, op.next_pc)
            .is_some_and(|next| next.key == "ref_return/r")
}

fn note_frame_finished_ops(
    body_code: &[u8],
    op: &crate::jitcode_runtime::DecodedOp,
    callee_descr_refs: &[DescrRef],
    num_regs_i: usize,
    constants_i: &[i64],
    flags_get: Option<(u8, u8)>,
) -> (Option<(u8, u8)>, Option<(u8, u8)>) {
    if op.key == "getfield_gc_i/rd>i" {
        if let (Some(&frame), Some(&dst)) = (
            body_code.get(op.pc + 1),
            body_code.get(op.next_pc.wrapping_sub(1)),
        ) {
            let descr = super::decode_descr_index(body_code, op, 1);
            if flags_field_descr(callee_descr_refs, descr) {
                return (Some((frame, dst)), None);
            }
        }
    } else if op.key == "int_or/ii>i" {
        if let Some((frame, src)) = flags_get {
            let a = body_code.get(op.pc + 1).copied();
            let b = body_code.get(op.pc + 2).copied();
            let dst = body_code.get(op.next_pc.wrapping_sub(1)).copied();
            if let (Some(a), Some(b), Some(dst)) = (a, b, dst) {
                let finished = i64::from(pyre_interpreter::PyFrame::FLAG_FRAME_FINISHED);
                let a_const = body_int_operand_constant(a, num_regs_i, constants_i);
                let b_const = body_int_operand_constant(b, num_regs_i, constants_i);
                let ors_finished = (a == src && a_const.is_none() && b_const == Some(finished))
                    || (b == src && b_const.is_none() && a_const == Some(finished));
                if ors_finished {
                    return (None, Some((frame, dst)));
                }
            }
        }
    }
    (None, None)
}

fn body_sample_safe_with(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    pure_helpers: &[majit_ir::RuntimeHelperKind],
    num_regs_i: usize,
    constants_i: &[i64],
) -> bool {
    let mut flags_get: Option<(u8, u8)> = None;
    let mut finished_or: Option<(u8, u8)> = None;
    // First ref-slot accessor names the callee frame. A later `>r` that
    // overwrites that register drops the identity.
    let mut vable_reg: Option<u8> = None;
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname.starts_with("goto_if_not") || d.opname.starts_with("switch") {
            return false;
        }
        let ref_slot_access = d.opname.starts_with("getarrayitem_vable_r")
            || d.key.starts_with("setarrayitem_vable_r");
        if ref_slot_access && d.argcodes.starts_with("ri") && vable_reg.is_none() {
            vable_reg = body_code.get(d.pc + 1).copied();
        }
        if d.opname.starts_with("residual_call") {
            let kind = residual_call_helper_kind_in_body(body_code, &d, callee_descr_refs);
            if !kind.is_some_and(|kind| pure_helpers.contains(&kind)) {
                return false;
            }
        } else if d.opname.starts_with("setfield_gc") {
            if !frame_finished_return_store(
                body_code,
                &d,
                callee_descr_refs,
                vable_reg,
                finished_or,
            ) {
                return false;
            }
        } else if d.opname.starts_with("setarrayitem_gc")
            || d.opname.starts_with("setinteriorfield_gc")
            || d.opname.starts_with("raw_store")
            || d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            return false;
        }
        (flags_get, finished_or) = note_frame_finished_ops(
            body_code,
            &d,
            callee_descr_refs,
            num_regs_i,
            constants_i,
            flags_get,
        );
        if d.argcodes.ends_with(">r")
            && let Some(&dst) = body_code.get(d.next_pc.saturating_sub(1))
            && vable_reg == Some(dst)
        {
            vable_reg = None;
        }
        pc = d.next_pc;
    }
    true
}

/// Whether sampling an exception string override before recording can have no
/// app-visible effect. Portal-frame vable traffic and constant/int boxing are
/// local; branches, other calls, and live-heap writes decline the sample.
pub(crate) fn exception_string_override_sample_safe(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    num_regs_i: usize,
    constants_i: &[i64],
) -> bool {
    body_sample_safe_with(
        body_code,
        callee_descr_refs,
        &[
            majit_ir::RuntimeHelperKind::LoadConst,
            majit_ir::RuntimeHelperKind::BoxInt,
        ],
        num_regs_i,
        constants_i,
    )
}

/// The same question for the `_index` callee ([`prepare_walker_inline_index`]),
/// which additionally reads the value it returns off its own instance.
///
/// `LOAD_ATTR` is admitted where the exception-override route refuses it.  The
/// residual covers both a plain instance-attribute read, which runs no app-level
/// code, and a descriptor whose getter does, and the bytecode alone cannot tell
/// them apart — but the ones that matter here are separable at run time by the
/// executed-effect odometer, which is what [`try_walker_inline_index`] consults
/// once the body has run.  `STORE_ATTR` is a distinct residual
/// ([`majit_ir::RuntimeHelperKind::StoreAttr`]) and stays rejected here, so the
/// ordinary writing `__index__` never runs at all: it has to be refused before
/// execution, because by the time the odometer could report it the write has
/// already happened once.
pub(crate) fn index_inline_sample_safe(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    num_regs_i: usize,
    constants_i: &[i64],
) -> bool {
    body_sample_safe_with(
        body_code,
        callee_descr_refs,
        &[
            majit_ir::RuntimeHelperKind::LoadConst,
            majit_ir::RuntimeHelperKind::BoxInt,
            majit_ir::RuntimeHelperKind::LoadAttr,
        ],
        num_regs_i,
        constants_i,
    )
}

/// The bounded builtin-dispatch route only admits a straight-line app-level
/// override. A control-flow-bearing method stays on the original residual
/// dispatch path, where the interpreter owns its frame and branch semantics.
pub(crate) fn exception_string_override_straight_line(body_code: &[u8]) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname.starts_with("goto_if_not") || d.opname.starts_with("switch") {
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// The un-lowered helper a descent into this jitcode body can reach on a path
/// that has already applied an effect, if any.
///
/// [`try_execute_residual_call_via_executor`] refuses to record a residual call
/// whose funcbox is an un-lowered helper's symbolic hash — the hash is not a
/// code address, so a compiled trace would branch to it — and raises
/// `OrthodoxSubWalkTraceUnsupported` at that call.  The caller answers that
/// abort in two ways, and which one it can use is the whole question here.  A
/// descent that executed nothing is cut back to `pre_fold_pos` and re-run as an
/// ordinary residual call, which applies the effect exactly once.  A descent
/// that already executed one cannot be rewound: the enclosing frame resumes at
/// its own `CALL`, the Python call runs a second time, and the first result is
/// discarded — `random.random()` in a loop advances the Mersenne Twister once
/// per aborted descent without producing a value for it (`gen.random()` drew
/// 4003 times for 4000 appends).
///
/// So the property to decline on is not "this body holds such a call" but
/// "this body can reach one with an effect behind it".  The walk is
/// execution-driven: at a `goto_if_not` it evaluates the condition, records a
/// guard, and steps into the taken arm only — the other arm is a guard-failure
/// resume, never walked.  A blocker sitting on an arm the call at hand does not
/// take is therefore never reached, and a whole-body scan that declines on it
/// declines a body the walk would have recorded cleanly.  That is not a corner:
/// the generated `__majit_wrap_*` gateways all begin by checking their argument
/// count and calling `gateway::method_{noarg,arity,min_arity}_failure` on the
/// reject arm, and those three helpers are `#[dont_look_inside]` precisely so
/// the formatting stays out of the wrapper's hot JitCode.  Declining on them
/// refuses the wrapper for the shape of its error path.
///
/// [`summarize_descent_blockers`] therefore walks the body's control-flow graph
/// carrying whether an effect has executed plus the Int/array-length facts it
/// can prove.  A known condition follows one successor exactly; a red
/// condition conservatively joins both.  It reports the two kinds of blocker
/// separately, and only `blocker_after_effect` is a decline.
///
/// The other decline is the scan admitting it did not read the whole body.
/// Everything the walk concludes rests on having seen every path, so a body it
/// could not decode in full is refused for that alone, without a blocker to
/// name.
///
/// The scan follows `inline_call_*` into the callee bodies the descent would
/// enter, because the abort propagates from any depth.  A body already on the
/// scan stack is a cycle, and answers "executes an effect, names no blocker":
/// the blockers behind the recursion belong to the occurrence that opened it,
/// but reporting no effect there would let a blocker past the recursion read
/// as effect-free, which is the one direction this scan may not err in.
///
/// The answer names the blocker rather than merely reporting one, because a
/// declined descent is a wall to be removed and the removal work is per
/// blocker, not per declining builtin.  The hash resolves to a description
/// through the `symbolic_fnaddr_paths` registry the pipeline snapshots, which
/// `jit_metadata.json` carries.
///
/// The answer is memoized on the jitcode itself, so the scan runs once per
/// (body, entry argument-array length) rather than once per call site.  Only
/// this entry point memoizes: the summary a cycle produces belongs to the
/// occurrence that opened it, not to the body, so
/// [`summarize_descent_blockers`] caches nothing, and a body entered with more
/// than [`DESCENT_ENTRY_LEN_SLOTS`] arguments recomputes.
fn descent_decline(
    jitcode_index: usize,
    entry_array_lengths: &[(usize, usize)],
) -> Option<DescentDecline> {
    if !descent_unlowered_helper_scan_enabled() {
        return None;
    }
    let summary = match entry_array_lengths {
        [] => descent_blocker_summary(jitcode_index),
        &[(0, entry_len)] => descent_blocker_summary_for_entry_len(jitcode_index, entry_len),
        other => summarize_descent_blockers_with_entry(jitcode_index, &mut Vec::new(), other),
    };
    if summary.body_not_walked {
        return Some(DescentDecline::BodyNotWalked);
    }
    summary.blocker_after_effect.map(DescentDecline::Helper)
}

/// Why a descent into a body is refused.
enum DescentDecline {
    /// An un-lowered helper the descent reaches with an effect already
    /// executed, named by the symbolic hash standing in for its funcbox.
    Helper(i64),
    /// Part of the body could not be decoded, so what was walked is a subgraph
    /// and the blockers it reports are only the ones that happened to be in
    /// it.  Nothing is named because the unread region is what is objected to.
    BodyNotWalked,
}

impl std::fmt::Display for DescentDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Helper(fnaddr) => write!(f, "un-lowered helper call in body blocker={fnaddr:#x}"),
            Self::BodyNotWalked => f.write_str("body holds bytes the scan cannot decode"),
        }
    }
}

/// Kill switch for the scan above: `PYRE_FBW_DESCENT_SCAN_OFF=1` lets the
/// descent run into the symbolic call and abort there instead of declining
/// before it starts.
///
/// The same shape and the same reason as `PYRE_WALKABORT_OFF` in
/// `trace.rs`: the scan decides, for every builtin the walker might descend
/// into, whether the descent happens at all, and its cost is invisible in
/// output — a declined descent is a correct answer that is merely slower.
/// One binary and one env var is the only way to weigh what the
/// conservatism buys against what it costs.
///
/// Graded over the 561 `__pyre_wrap_*` gateway wrappers, 448 of the 487 the
/// scan declines block on a call no complete path through the descent
/// executes, so what this switch measures is mostly the price of a static
/// answer to a dynamic question. It is a diagnostic, not a tuning knob:
/// the scan is on unless it is explicitly turned off, because the rewind it
/// prevents is a wrong answer and not a slow one. The attempted rollback
/// compiles `test_pickle` writes to an empty payload and changes
/// `test_hashlib`'s bytes argument into a string.
fn descent_unlowered_helper_scan_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_DESCENT_SCAN_OFF").is_none())
}

/// Print every un-lowered helper the descent into `jitcode_index` could reach,
/// once per body, under `PYRE_FBW_INLINE_DIAG`.
///
/// This is deliberately separate from the memoized decision above: production
/// needs only the effect-aware verdict, while the diagnostic pays to enumerate
/// the whole body and every callee named by `inline_call_*`.
///
/// The blockers print as bare symbolic funcbox hashes, which name nothing on
/// their own. The hash-to-path map is not in the binary — it is snapshotted by
/// the translation pipeline (`symbolic_fnaddr_paths_snapshot`) and written to
/// the codegen cache, so resolve a hash by reading `symbolic_fnaddr_paths` out
/// of the newest `build/pyre-jit-trace-cache/*/*/jit_metadata.json` (newest by
/// mtime — older generations hold different jitcode indices for the same path).
/// A hash whose high bit is set is stored there as a negative `i64`.
fn log_descent_unlowered_helper_blockers(jitcode_index: usize) {
    if !fbw_inline_diag_enabled() {
        return;
    }
    static LOGGED: parking_lot::Mutex<Option<std::collections::HashSet<usize>>> =
        parking_lot::Mutex::new(None);
    let mut guard = LOGGED.lock();
    if !guard
        .get_or_insert_with(Default::default)
        .insert(jitcode_index)
    {
        return;
    }
    drop(guard);
    let blockers = collect_descent_unlowered_helper_blockers(jitcode_index, &mut Vec::new());
    let named = blockers
        .iter()
        .map(|blocker| format!("{blocker:#x}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[builtin-inline-blockers] jitcode={jitcode_index} n={} {named}",
        blockers.len()
    );

    let mut reachable = Vec::new();
    collect_descent_effect_aware_blockers(
        jitcode_index,
        false,
        &mut std::collections::HashSet::new(),
        &mut reachable,
    );
    let after_effect = reachable.iter().filter(|&&(_, effect)| effect).count();
    let named = reachable
        .iter()
        .map(|&(blocker, effect)| format!("{blocker:#x}{}", if effect { "*" } else { "" }))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[builtin-inline-reachable] jitcode={jitcode_index} n={} after_effect={after_effect} {named}",
        reachable.len()
    );
}

/// Every un-lowered helper an effect-aware walk of `jitcode_index` reaches,
/// each paired with whether an effect had already been applied on some path to
/// it — the `*` suffix in the `[builtin-inline-reachable]` line.
///
/// [`summarize_descent_blockers`] answers the production question, "does this
/// descent abort unrewindably", and so stops at the first blocker on each path
/// and names one.  Removing a wall asks the other question: what stands behind
/// that first blocker, because clearing it only moves the decline to the next.
/// This walk therefore records a blocker and keeps going, which turns the
/// decline into a work list — and, read against
/// [`collect_descent_unlowered_helper_blockers`]'s whole-body census, separates
/// the helpers a descent can actually reach from the ones sitting on arms it
/// never takes.
///
/// The two lists are not one walk with a flag because they answer to different
/// costs: the census enumerates bytes, while this reruns the dataflow.
///
/// Blockers found in an `inline_call` callee are appended after the caller's
/// own rather than interleaved at the call, so this is walk order per body and
/// not strictly byte order across the descent.  The first blocker is already
/// named by the decline line; what this adds is the set.
///
/// A body is walked at most once per entry-effect state rather than once per
/// path reaching it.  Both are sound for a set, but re-walking runs this
/// body-wide fixpoint once per path through the callee DAG, which does not
/// terminate in practice: the whole-body census next door survives the same
/// shape only because its per-body cost is one linear decode.
fn collect_descent_effect_aware_blockers(
    jitcode_index: usize,
    entry_effect: bool,
    visited: &mut std::collections::HashSet<(usize, bool)>,
    out: &mut Vec<(i64, bool)>,
) {
    if !visited.insert((jitcode_index, entry_effect)) {
        return;
    }
    let Some(jitcode) = crate::jitcode_runtime::get_jitcode_ref_by_index(jitcode_index) else {
        return;
    };
    let descrs = crate::jitcode_runtime::descr_ref_table();
    // Collected rather than recorded in place: both hooks would otherwise hold
    // `out` borrowed at once.
    let mut here: Vec<(i64, bool)> = Vec::new();
    let mut callees: Vec<(usize, bool)> = Vec::new();
    summarize_body_blockers_with(
        jitcode.code.as_slice(),
        jitcode.num_regs_i(),
        jitcode.constants_i.as_slice(),
        |descr_index, caller_effect| {
            let callee = descrs
                .at(descr_index)
                .and_then(|descr| descr.as_jitcode_descr().map(|jc| jc.jitcode_index()))?;
            callees.push((callee, caller_effect));
            // The memoizing entry point, so a cycle among callees terminates
            // inside it rather than through this walk's own `seen`.
            Some(descent_blocker_summary(callee))
        },
        &mut |blocker, effect| {
            here.push((blocker, effect));
            false
        },
        &mut residual_call_is_effect_free,
        &mut switch_descr_targets,
        &[],
    );
    for (blocker, effect) in here {
        record_reachable_blocker(out, blocker, entry_effect || effect);
    }
    // Per-jitcode summaries (and the decode feeding them) are debug-abort
    // output; inline-diag alone keeps the recursive walk quiet.
    if fbw_debug_abort_enabled() {
        let memo = descent_blocker_summary(jitcode_index);
        eprintln!(
            "[builtin-inline-summary] jitcode={jitcode_index} may_effect={} free={:?} after={:?} not_walked={}",
            memo.may_execute_effect,
            memo.blocker_effect_free.map(|b| format!("{b:#x}")),
            memo.blocker_after_effect.map(|b| format!("{b:#x}")),
            memo.body_not_walked
        );
        if let Some(pc) = memo.first_effect_pc {
            let (opname, descr) = crate::jitcode_runtime::decode_op_at(jitcode.code.as_slice(), pc)
                .map(|d| {
                    let descr = descr_operand_index(jitcode.code.as_slice(), &d);
                    (d.opname, descr)
                })
                .unwrap_or(("?", None));
            eprintln!(
                "[builtin-inline-first-effect] jitcode={jitcode_index} pc={pc} op={opname} descr={descr:?}"
            );
        }
    }
    for (callee, caller_effect) in callees {
        collect_descent_effect_aware_blockers(callee, entry_effect || caller_effect, visited, out);
    }
}

/// Add one blocker to the reachable list, keeping the first sighting's position
/// and widening its effect flag: reaching the same helper once with an effect
/// behind it is what makes it a decline, however many effect-free paths also
/// reach it.
fn record_reachable_blocker(out: &mut Vec<(i64, bool)>, blocker: i64, effect: bool) {
    match out.iter_mut().find(|(known, _)| *known == blocker) {
        Some((_, known_effect)) => *known_effect |= effect,
        None => out.push((blocker, effect)),
    }
}

/// Collect the distinct symbolic funcboxes in a body's decoded byte stream and
/// in the bodies named by its `inline_call_*` instructions, in byte order.
fn collect_descent_unlowered_helper_blockers(
    jitcode_index: usize,
    seen: &mut Vec<usize>,
) -> Vec<i64> {
    if seen.contains(&jitcode_index) {
        return Vec::new();
    }
    let Some(body) = crate::jitcode_dispatch::sub_jitcode_body_by_index(jitcode_index) else {
        return Vec::new();
    };
    seen.push(jitcode_index);
    let descrs = crate::jitcode_runtime::descr_ref_table();
    let mut known_i = vec![None; body.num_regs_i + body.constants_i.len()];
    for (slot, &value) in body.constants_i.iter().enumerate() {
        known_i[body.num_regs_i + slot] = Some(value);
    }
    let mut pc = 0usize;
    let mut blockers = Vec::new();
    while pc < body.code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body.code, pc) else {
            break;
        };
        if d.opname.starts_with("residual_call") {
            let funcbox = body.code.get(d.pc + 1).copied().unwrap_or(0) as usize;
            if let Some(Some(fnaddr)) = known_i.get(funcbox)
                && majit_jitcode::codewriter::call::is_symbolic_fnaddr(*fnaddr)
                && !blockers.contains(fnaddr)
            {
                blockers.push(*fnaddr);
            }
        } else if d.opname.starts_with("inline_call") {
            let descr_index = body.code.get(d.pc + 1).copied().unwrap_or(0) as usize
                | ((body.code.get(d.pc + 2).copied().unwrap_or(0) as usize) << 8);
            if let Some(callee) = descrs
                .at(descr_index)
                .and_then(|descr| descr.as_jitcode_descr().map(|jc| jc.jitcode_index()))
            {
                for blocker in collect_descent_unlowered_helper_blockers(callee, seen) {
                    if !blockers.contains(&blocker) {
                        blockers.push(blocker);
                    }
                }
            }
        }
        if d.argcodes
            .split_once('>')
            .is_some_and(|(_, dst)| dst == "i")
            && let Some(&dst) = body.code.get(d.next_pc.wrapping_sub(1))
        {
            let carried = (d.key == "int_copy/i>i")
                .then(|| body.code.get(d.pc + 1))
                .flatten()
                .and_then(|&src| known_i.get(src as usize).copied())
                .flatten();
            if let Some(slot) = known_i.get_mut(dst as usize) {
                *slot = carried;
            }
        }
        pc = d.next_pc;
    }
    seen.pop();
    blockers
}

/// The answer for a region of a descent the scan did not walk.
///
/// `may_execute_effect` because an unwalked region can have done anything, and
/// `body_not_walked` because that is the condition itself; `descent_decline`
/// reads the second alone as a refusal, so this is the fail-closed answer to
/// "what does this body hold".
fn body_not_walked() -> DescentBlockerSummary {
    DescentBlockerSummary {
        may_execute_effect: true,
        body_not_walked: true,
        ..DescentBlockerSummary::default()
    }
}

/// Memoizing entry point for [`summarize_descent_blockers`] with the entry
/// argument-array length a call site knows.
fn descent_blocker_summary_for_entry_len(
    jitcode_index: usize,
    entry_len: usize,
) -> DescentBlockerSummary {
    let Some(jitcode) = crate::jitcode_runtime::get_jitcode_ref_by_index(jitcode_index) else {
        return body_not_walked();
    };
    jitcode.descent_blocker_summary_for_entry_len(entry_len, || {
        summarize_descent_blockers_with_entry(jitcode_index, &mut Vec::new(), &[(0, entry_len)])
    })
}

/// Memoizing entry point for [`summarize_descent_blockers`].
fn descent_blocker_summary(jitcode_index: usize) -> DescentBlockerSummary {
    let Some(jitcode) = crate::jitcode_runtime::get_jitcode_ref_by_index(jitcode_index) else {
        // The same condition the descent answers `body_not_walked` for, one
        // level up and with nothing walked at all.  Unreachable from the gate,
        // which holds the `JitCode` this index came off; see the note in
        // `summarize_descent_blockers`.
        return body_not_walked();
    };
    jitcode.descent_blocker_summary(|| summarize_descent_blockers(jitcode_index, &mut Vec::new()))
}

/// One reachable point of [`summarize_descent_blockers`]'s dataflow.
///
/// The carried facts only lose information when two paths meet — `effect` goes
/// false to true, disagreeing known-value slots go to `None`, and freshness
/// goes true to false — so the worklist converges.
#[derive(Clone)]
struct DescentPoint {
    /// Whether some path from the body entry to here executed an effect.
    effect: bool,
    /// What each Int-bank slot is known to hold, in the same encoding
    /// `allocate_callee_register_banks` uses: the slots at and above
    /// `num_regs_i` are pre-filled from `constants_i`.
    known_i: Vec<Option<i64>>,
    /// Known lengths of Ref-bank slots that hold GC arrays.  Generated
    /// `__majit_wrap_*` gateways receive their concrete argument-array length
    /// from the call site; copies preserve it and any other Ref producer drops
    /// it back to unknown.
    known_array_len_r: Vec<Option<usize>>,
    /// Which Ref-bank slots hold an object this body itself allocated (a
    /// `new*` result) on every path to here.  A write into such an object is
    /// not an effect: a rewind discards the allocation with it.
    fresh_r: Vec<bool>,
}

/// True when a funcbox constant names a residual
/// `do_residual_call`'s `provably_side_effect_free` already spares by
/// address: a re-runnable bookkeeping helper or a root-bracket one.
///
/// The scan asks this of a constant where the walk asks it of an executed
/// call.  They have to agree — a body the scan calls effectful and the walk
/// does not is a descent refused for an effect that never happens.
fn effect_free_residual_fnaddr(fnaddr: i64) -> bool {
    pyre_interpreter::is_rewindable_root_bracket_residual_i64(fnaddr)
        || pyre_interpreter::is_rerunnable_bookkeeping_residual(fnaddr as usize)
}

/// Whether stepping `opname` applies an effect the walk would have to undo.
///
/// The walk's own odometer (`fbw_bump_executed_effect`) counts an executed
/// residual call that writes live heap or enters a frame, plus the journaled
/// list and cell stores.  Deciding those conditions needs the concrete values a
/// static scan does not have, so read every executed residual call as effectful
/// and add the direct heap writes, which the odometer reaches only through a
/// journal.  Both directions are conservative: the scan may call a body
/// effectful where the walk would not, never the reverse.
///
/// Two of those the caller spares, because the odometer spares them too: a
/// residual the walk would not count ([`effect_free_residual_fnaddr`], or
/// [`residual_call_is_effect_free`]) and a store into an object this same body
/// allocated.  Calling either effectful here refuses a descent for an effect
/// that never happens.
fn descent_op_applies_effect(opname: &str) -> bool {
    opname.starts_with("residual_call")
        || opname.starts_with("setfield_gc")
        || opname.starts_with("setfield_raw")
        || opname.starts_with("setarrayitem_gc")
        || opname.starts_with("setarrayitem_raw")
        || opname.starts_with("setinteriorfield_gc")
        || opname.starts_with("raw_store")
        || opname.starts_with("strsetitem")
        || opname.starts_with("unicodesetitem")
        || opname.starts_with("copystrcontent")
        || opname.starts_with("copyunicodecontent")
}

/// Byte offset of an op's first `L` operand, counted from the byte after the
/// opcode, or `None` when it carries none.  Same operand widths
/// `decode_op_at` advances by.
fn label_operand_offset(argcodes: &str) -> Option<usize> {
    let mut offset = 0usize;
    let mut chars = argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'L' => return Some(offset),
            'i' | 'c' | 'r' | 'f' => offset += 1,
            'd' | 'j' => offset += 2,
            // A varlist's width depends on the code bytes, which this shape
            // question does not read; no opname puts one before its label.
            'I' | 'R' | 'F' | 'P' => return None,
            '>' => {
                chars.next()?;
                offset += 1;
            }
            _ => return None,
        }
    }
    None
}

/// Register operands are one byte, so this many Ref-bank slots cover any body.
const FRESH_SLOTS: usize = 256;

/// Whether `op` is a heap write whose object operand -- the leading `r`
/// register of every `setfield_gc_*` / `setarrayitem_gc_*` /
/// `setinteriorfield_gc_*` -- holds an object allocated by this body.  The
/// vable spellings write the frame, which is never fresh, and are not asked.
fn heap_write_into_fresh_object(
    code: &[u8],
    op: &crate::jitcode_runtime::DecodedOp,
    fresh_r: &[bool],
) -> bool {
    (op.opname.starts_with("setfield_gc")
        || op.opname.starts_with("setarrayitem_gc")
        || op.opname.starts_with("setinteriorfield_gc"))
        && op.argcodes.starts_with('r')
        && code
            .get(op.pc + 1)
            .is_some_and(|&obj| fresh_r.get(obj as usize).copied().unwrap_or(false))
}

/// The descr-pool index an op's first `d` operand names, read past the
/// register and varlist operands before it the way `decode_op_at` advances
/// over them: the funcbox and varlists of a `residual_call_*`, the key
/// register of a `switch/id`.
fn descr_operand_index(code: &[u8], op: &crate::jitcode_runtime::DecodedOp) -> Option<usize> {
    let mut cursor = op.pc + 1;
    let mut chars = op.argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' | 'c' | 'r' | 'f' => cursor += 1,
            'j' => cursor += 2,
            'I' | 'R' | 'F' => cursor += 1 + *code.get(cursor)? as usize,
            'd' => {
                return Some(
                    *code.get(cursor)? as usize | ((*code.get(cursor + 1)? as usize) << 8),
                );
            }
            '>' => {
                chars.next()?;
                cursor += 1;
            }
            _ => return None,
        }
    }
    None
}

/// Whether the residual call behind `descr_index` applies no effect the walk
/// would have to undo, by the call's own effectinfo.
///
/// An elidable or loop-invariant callee writes no live heap.  A
/// `not_in_trace` callee is exempt for a different reason: what has to hold
/// is not that the call left nothing behind, but that calling it again is
/// harmless, because a rolled-back walk is another trace attempt and the
/// region will be walked again.  `rlib/jit.py not_in_trace` states the
/// contract that makes that true — the call is still made "by the jit tracing
/// and blackholing, but not by the final assembler", so a callee already owes
/// the same answer once per attempt whether or not any attempt is undone, and
/// the tracer alone decides how many attempts there are.  pyre's one such
/// callee, `ensure_object_subclass_ranges_initialized`, is a `OnceLock`
/// initializer, which is that property in its strongest form.
///
/// Anything else keeps the scan's conservative reading of an executed
/// residual call as effectful.
fn residual_call_is_effect_free(descr_index: usize) -> bool {
    let descrs = crate::jitcode_runtime::descr_ref_table();
    let Some(descr) = descrs.at(descr_index) else {
        return false;
    };
    let Some(call) = descr.as_call_descr() else {
        return false;
    };
    let info = call.get_extra_info();
    info.oopspecindex == majit_ir::descr::OopSpecIndex::NotInTrace
        || matches!(
            info.extraeffect,
            majit_ir::ExtraEffect::ElidableCannotRaise
                | majit_ir::ExtraEffect::ElidableCanRaise
                | majit_ir::ExtraEffect::ElidableOrMemoryError
                | majit_ir::ExtraEffect::LoopInvariant
        )
}

/// The arm targets of the `switch/id` whose descr is `descr_index`, in key
/// order, or `None` when the descr is not a switch table or a listed key
/// does not resolve to a target.
fn switch_descr_targets(descr_index: usize) -> Option<Vec<usize>> {
    let descrs = crate::jitcode_runtime::descr_ref_table();
    let descr = descrs.at(descr_index)?;
    let switch = descr.as_switch_descr()?;
    // Every listed key must resolve: dropping one would narrow the successor
    // set, and a region this scan does not traverse is a region it wrongly
    // reports clean.  An unresolved key therefore widens the whole answer to
    // `None`.
    switch
        .const_keys_in_order()
        .iter()
        .map(|&key| switch.lookup(key))
        .collect()
}

/// One-key lookup for a known `switch/id` discriminant.
///
/// `bhimpl_switch`: `switchdict.dict[switchvalue]`, or the fallthrough pc
/// on `KeyError`. `None` means this descr is not a switch table, so the
/// scan keeps the conservative all-arm successor set.
fn switch_descr_lookup(descr_index: usize, value: i64) -> Option<Option<usize>> {
    let descrs = crate::jitcode_runtime::descr_ref_table();
    let descr = descrs.at(descr_index)?;
    let switch = descr.as_switch_descr()?;
    Some(switch.lookup(value))
}

/// Evaluate the Int result of the small, side-effect-free opcode family the
/// blocker scan needs to decide a following conditional edge.
///
/// This is the static counterpart of the corresponding `opimpl_*` methods in
/// `pyjitpl.py`: it does not invent a value for a red operand.  It only folds
/// when every input is already known: an `i` operand names a slot whose value
/// the scan carries (a register it folded, or a constant one of the pool slots
/// above `num_regs_i` was seeded with), and a `c` operand is the value itself.
/// Unknown and overflow-sensitive operations stay unknown, so the caller keeps
/// both successors.
fn known_int_result(
    code: &[u8],
    op: &crate::jitcode_runtime::DecodedOp,
    known_i: &[Option<i64>],
    known_array_len_r: &[Option<usize>],
) -> Option<i64> {
    let int = |offset: usize| {
        code.get(op.pc + 1 + offset)
            .and_then(|&slot| known_i.get(slot as usize))
            .copied()
            .flatten()
    };
    // A `c` argcode is `assembler.py emit_const(allow_short=True)`: the small
    // ConstInt is written inline as one signed byte, so the byte is the source
    // value and there is no pool slot to index.
    let immediate = |offset: usize| code.get(op.pc + 1 + offset).map(|&byte| byte as i8 as i64);
    let bool_word = |value: bool| i64::from(value);

    match op.key {
        "arraylen_gc/rd>i" => code
            .get(op.pc + 1)
            .and_then(|&slot| known_array_len_r.get(slot as usize))
            .copied()
            .flatten()
            .and_then(|value| i64::try_from(value).ok()),
        "int_copy/i>i" => int(0),
        "int_copy/c>i" => immediate(0),
        "int_add/ii>i" => Some(int(0)?.wrapping_add(int(1)?)),
        "int_sub/ii>i" => Some(int(0)?.wrapping_sub(int(1)?)),
        "int_mul/ii>i" => Some(int(0)?.wrapping_mul(int(1)?)),
        "int_and/ii>i" => Some(int(0)? & int(1)?),
        // `executor.execute_binary_int_const_row` accepts `IntSignext` only for
        // a byte count in `1..=8`; `support.int_signext` asserts `numbytes > 0`.
        "int_signext/ii>i" => {
            let value = int(0)?;
            let numbytes = int(1)?;
            (1..=8)
                .contains(&numbytes)
                .then(|| majit_metainterp::support::int_signext(value, numbytes))
        }
        "int_or/ii>i" => Some(int(0)? | int(1)?),
        "int_xor/ii>i" => Some(int(0)? ^ int(1)?),
        "int_neg/i>i" => Some(int(0)?.wrapping_neg()),
        "int_invert/i>i" => Some(!int(0)?),
        "int_is_true/i>i" => Some(bool_word(int(0)? != 0)),
        "int_lt/ii>i" => Some(bool_word(int(0)? < int(1)?)),
        "int_le/ii>i" => Some(bool_word(int(0)? <= int(1)?)),
        "int_eq/ii>i" => Some(bool_word(int(0)? == int(1)?)),
        "int_ne/ii>i" => Some(bool_word(int(0)? != int(1)?)),
        "int_gt/ii>i" => Some(bool_word(int(0)? > int(1)?)),
        "int_ge/ii>i" => Some(bool_word(int(0)? >= int(1)?)),
        "uint_lt/ii>i" => Some(bool_word((int(0)? as u64) < (int(1)? as u64))),
        "uint_le/ii>i" => Some(bool_word((int(0)? as u64) <= (int(1)? as u64))),
        "uint_gt/ii>i" => Some(bool_word((int(0)? as u64) > (int(1)? as u64))),
        "uint_ge/ii>i" => Some(bool_word((int(0)? as u64) >= (int(1)? as u64))),
        _ => None,
    }
}

/// Whether a known Int condition makes this `goto_if_not*` take its label.
/// `None` is the deliberately conservative answer for a red condition or for
/// the pointer/float forms, whose value domains this scan does not carry.
fn known_goto_if_taken(
    code: &[u8],
    op: &crate::jitcode_runtime::DecodedOp,
    known_i: &[Option<i64>],
) -> Option<bool> {
    let int = |offset: usize| {
        code.get(op.pc + 1 + offset)
            .and_then(|&slot| known_i.get(slot as usize))
            .copied()
            .flatten()
    };
    let predicate = match op.key {
        "goto_if_not/iL" | "goto_if_not_int_is_true/iL" => int(0)? != 0,
        "goto_if_not_int_is_zero/iL" => int(0)? == 0,
        "goto_if_not_int_lt/iiL" => int(0)? < int(1)?,
        "goto_if_not_int_le/iiL" => int(0)? <= int(1)?,
        "goto_if_not_int_eq/iiL" => int(0)? == int(1)?,
        "goto_if_not_int_ne/iiL" => int(0)? != int(1)?,
        "goto_if_not_int_gt/iiL" => int(0)? > int(1)?,
        "goto_if_not_int_ge/iiL" => int(0)? >= int(1)?,
        _ => return None,
    };
    Some(!predicate)
}

/// Recursive worker of [`descent_blocker_summary`].  `seen` is the stack of
/// jitcode indices currently being scanned.
fn summarize_descent_blockers(
    jitcode_index: usize,
    seen: &mut Vec<usize>,
) -> DescentBlockerSummary {
    let mut cycle_hits = 0;
    summarize_descent_blockers_with_entry_inner(jitcode_index, seen, &[], &mut cycle_hits)
}

/// [`summarize_descent_blockers`] for a callee, memoized on the callee's body
/// when the answer belongs to that body.
///
/// A callee is scanned with no entry facts, so what it computes is the same
/// fact-free summary [`descent_blocker_summary`] holds — except under a cycle,
/// where a body already on the stack answers "executes an effect, names no
/// blocker" for the occurrence that opened the cycle rather than for the body.
/// A subtree that took that arm nowhere did not read the stack at all and its
/// answer is a property of the body alone.  Without the memo a callee shared by
/// several paths is re-walked once per path, and the walk is transitive, so the
/// scan a single gateway pays grows with the shape of the graph beneath it.
fn summarize_descent_blockers_cached(
    jitcode_index: usize,
    seen: &mut Vec<usize>,
    cycle_hits: &mut u64,
) -> DescentBlockerSummary {
    if seen.contains(&jitcode_index) {
        return summarize_descent_blockers_with_entry_inner(jitcode_index, seen, &[], cycle_hits);
    }
    let Some(jitcode) = crate::jitcode_runtime::get_jitcode_ref_by_index(jitcode_index) else {
        return summarize_descent_blockers_with_entry_inner(jitcode_index, seen, &[], cycle_hits);
    };
    if let Some(cached) = jitcode.descent_blocker_summary_if_computed() {
        return cached;
    }
    let cycles_before = *cycle_hits;
    let summary = summarize_descent_blockers_with_entry_inner(jitcode_index, seen, &[], cycle_hits);
    if *cycle_hits == cycles_before {
        jitcode.descent_blocker_summary(|| summary);
    }
    summary
}

/// [`summarize_descent_blockers`] with concrete facts known at the entry of
/// the top body.  Callee facts are not guessed: without mapping an
/// `inline_call_*` varlist into the callee banks, recursive bodies use their
/// ordinary conservative summaries.
fn summarize_descent_blockers_with_entry(
    jitcode_index: usize,
    seen: &mut Vec<usize>,
    entry_array_lengths: &[(usize, usize)],
) -> DescentBlockerSummary {
    // This memoization fence belongs to this analysis stack, not to the
    // executing thread. Every recursive callee shares it; an independent
    // analysis (including a reentrant one) must not change its value.
    let mut cycle_hits = 0;
    summarize_descent_blockers_with_entry_inner(
        jitcode_index,
        seen,
        entry_array_lengths,
        &mut cycle_hits,
    )
}

fn summarize_descent_blockers_with_entry_inner(
    jitcode_index: usize,
    seen: &mut Vec<usize>,
    entry_array_lengths: &[(usize, usize)],
    cycle_hits: &mut u64,
) -> DescentBlockerSummary {
    if seen.contains(&jitcode_index) {
        *cycle_hits += 1;
        return DescentBlockerSummary {
            may_execute_effect: true,
            ..DescentBlockerSummary::default()
        };
    }
    let Some(jitcode) = crate::jitcode_runtime::get_jitcode_ref_by_index(jitcode_index) else {
        // A body the scan cannot obtain is a callee it holds nothing about, so
        // it answers the same way as an `inline_call` whose descr names no
        // JitCode.  Returning the default instead would say "executes no
        // effect, holds no blocker" — the most permissive answer there is,
        // reached by not looking — and would also let the blocker behind such a
        // call read as effect-free, which asserts a rewind covers it.
        //
        // Unreachable in a consistent build, and not for the reason the entry
        // point's own lookup suggests: `descent_blocker_summary` resolves the
        // index before calling here, so at the top level this repeats a lookup
        // that just succeeded and only the `inline_call` descent can arrive
        // with an index of its own.  `get_jitcode_ref_by_index` answers `None`
        // only past the end of the table — inside it, the cell is loaded on
        // demand — and every `JitCode` descr names an index inside it: 2335 of
        // 2335 in this build, over a table of 2905 whose indices are dense.
        return body_not_walked();
    };
    seen.push(jitcode_index);
    let descrs = crate::jitcode_runtime::descr_ref_table();
    let summary = summarize_body_blockers_with(
        jitcode.code.as_slice(),
        jitcode.num_regs_i(),
        jitcode.constants_i.as_slice(),
        |descr_index, _caller_effect| {
            descrs
                .at(descr_index)
                .and_then(|descr| descr.as_jitcode_descr().map(|jc| jc.jitcode_index()))
                .map(|callee| summarize_descent_blockers_cached(callee, seen, cycle_hits))
        },
        &mut |_blocker, _effect| true,
        &mut residual_call_is_effect_free,
        &mut switch_descr_targets,
        entry_array_lengths,
    );
    seen.pop();
    summary
}

/// The dataflow of [`summarize_descent_blockers`] over one body's bytes.
///
/// `callee_summary` answers an `inline_call`'s descr operand with the callee's
/// own summary, or `None` when that operand names no JitCode.  Splitting it out
/// keeps the analysis testable on a hand-built body, which is the only way to
/// state the two-kind split without an installed jitcode table.
///
/// Test-only: production reaches the same dataflow through
/// [`summarize_body_blockers_with`], because every one of the hooks defaulted
/// here has a real production answer.
#[cfg(test)]
pub(crate) fn summarize_body_blockers(
    code: &[u8],
    num_regs_i: usize,
    constants_i: &[i64],
    mut callee_summary: impl FnMut(usize) -> Option<DescentBlockerSummary>,
) -> DescentBlockerSummary {
    summarize_body_blockers_with(
        code,
        num_regs_i,
        constants_i,
        |descr_index, _caller_effect| callee_summary(descr_index),
        &mut |_blocker, _effect| true,
        &mut |_descr_index| false,
        &mut |_descr_index| None,
        &[],
    )
}

/// [`summarize_body_blockers`] with the two hooks the diagnostic needs.
///
/// `on_blocker` is told each blocker and whether an effect had been applied
/// before it, and answers whether the walk stops there.  The production
/// summary stops, because that is where the descent aborts; a walk that wants
/// what stands *behind* the first blocker keeps going, treating the call as
/// having returned an unknown value — which is what the destination-blanking
/// below already does for any non-`int_copy` write.
///
/// `callee_summary` is additionally told the caller's effect state at the
/// `inline_call`, which the summary itself does not need — it folds the two
/// cases through `blocker_after_effect.or(blocker_effect_free)` — but which a
/// walk descending into the callee needs to classify what it finds there.
///
/// `call_effect_free` answers a `residual_call_*` op's descr index with
/// whether that call is exempt from the effectful reading
/// `descent_op_applies_effect` gives every residual call; the production
/// answer is [`residual_call_is_effect_free`].
///
/// `switch_targets` answers a `switch/id` op's descr index with the arm
/// targets its table holds ([`switch_descr_targets`]); `None` widens the
/// successors to every instruction start.
///
/// `entry_array_lengths` seeds concrete GC-array lengths known at this call
/// site.  Production uses it for the generated builtin wrapper's sole `r0`
/// argument; tests and memoized whole-body summaries pass an empty slice.
pub(crate) fn summarize_body_blockers_with(
    code: &[u8],
    num_regs_i: usize,
    constants_i: &[i64],
    mut callee_summary: impl FnMut(usize, bool) -> Option<DescentBlockerSummary>,
    on_blocker: &mut dyn FnMut(i64, bool) -> bool,
    call_effect_free: &mut dyn FnMut(usize) -> bool,
    switch_targets: &mut dyn FnMut(usize) -> Option<Vec<usize>>,
    entry_array_lengths: &[(usize, usize)],
) -> DescentBlockerSummary {
    let mut summary = DescentBlockerSummary::default();

    // Instruction starts, so a decoded branch target that lands mid-op (or
    // past the end) is dropped rather than decoded as an operand.  The
    // assembler lays the body out linearly, so one forward decode names them
    // all, ascending.  `switch` enumerates this as its successor set, so the
    // order has to be the body's own rather than a hash order: which of
    // several blockers the answer names would otherwise change per process.
    let mut starts = Vec::new();
    let mut pc = 0usize;
    while let Some(d) = crate::jitcode_runtime::decode_op_at(code, pc) {
        starts.push(pc);
        pc = d.next_pc;
        if pc >= code.len() {
            break;
        }
    }
    if pc < code.len() {
        // The decode stopped inside the body, so `starts` names a prefix of it
        // and the `push!` gate below would drop every target past that point
        // as "not a start".  A dropped successor removes a region from the
        // graph, and a region the scan never enters reports no blocker, so
        // walking a graph already known to be partial can only produce an
        // answer that admits for the wrong reason.  Say what is actually
        // known instead.
        return body_not_walked();
    }

    let mut entry_known = vec![None; num_regs_i + constants_i.len()];
    for (slot, &value) in constants_i.iter().enumerate() {
        entry_known[num_regs_i + slot] = Some(value);
    }
    // A handler runs with the join of every state in the region it protects, so
    // its registers hold no known value -- but a constant slot is never a `>i`
    // destination, so it holds the same value at every one of those points and
    // survives the join exactly.  Blanking those too would drop the operand a
    // `residual_call` in a handler reads its funcbox from, and a funcbox that
    // reads unknown is a blocker the scan does not report.
    let handler_known = entry_known.clone();
    let mut entry_array_len_r = vec![None; FRESH_SLOTS];
    for &(slot, len) in entry_array_lengths {
        if let Some(value) = entry_array_len_r.get_mut(slot) {
            *value = Some(len);
        }
    }
    let mut points: std::collections::HashMap<usize, DescentPoint> =
        std::collections::HashMap::new();
    points.insert(
        0,
        DescentPoint {
            effect: false,
            known_i: entry_known,
            known_array_len_r: entry_array_len_r,
            fresh_r: vec![false; FRESH_SLOTS],
        },
    );
    let mut work = std::collections::VecDeque::from([0usize]);

    // `push` is the dataflow's only join: it merges `state` into the point at
    // `target` and re-queues it when that widened anything.
    macro_rules! push {
        ($points:expr, $work:expr, $target:expr, $state:expr) => {{
            let target: usize = $target;
            if starts.binary_search(&target).is_ok() {
                let state: DescentPoint = $state;
                match $points.get_mut(&target) {
                    None => {
                        $points.insert(target, state);
                        $work.push_back(target);
                    }
                    Some(existing) => {
                        let mut widened = false;
                        if state.effect && !existing.effect {
                            existing.effect = true;
                            widened = true;
                        }
                        for (slot, incoming) in existing.known_i.iter_mut().zip(&state.known_i) {
                            if *slot != *incoming && slot.is_some() {
                                *slot = None;
                                widened = true;
                            }
                        }
                        for (slot, incoming) in existing
                            .known_array_len_r
                            .iter_mut()
                            .zip(&state.known_array_len_r)
                        {
                            if *slot != *incoming && slot.is_some() {
                                *slot = None;
                                widened = true;
                            }
                        }
                        for (slot, incoming) in existing.fresh_r.iter_mut().zip(&state.fresh_r) {
                            if *slot && !*incoming {
                                *slot = false;
                                widened = true;
                            }
                        }
                        if widened {
                            $work.push_back(target);
                        }
                    }
                }
            }
        }};
    }

    while let Some(pc) = work.pop_front() {
        let Some(point) = points.get(&pc).cloned() else {
            continue;
        };
        let Some(d) = crate::jitcode_runtime::decode_op_at(code, pc) else {
            // Every queued point is one of `starts`, and the guard above
            // established that those cover the body, so this does not happen.
            // It is still the same missing-region shape if it ever does.
            summary.body_not_walked = true;
            continue;
        };
        let mut effect = point.effect;
        let mut known_i = point.known_i;
        let mut known_array_len_r = point.known_array_len_r;
        let mut fresh_r = point.fresh_r;

        if d.opname.starts_with("residual_call") {
            // Every `residual_call_*` argcode string opens with the `i` funcbox
            // operand, so it is the byte right after the opcode.
            let funcbox = code.get(d.pc + 1).copied().unwrap_or(0) as usize;
            if let Some(Some(fnaddr)) = known_i.get(funcbox)
                && majit_jitcode::codewriter::call::is_symbolic_fnaddr(*fnaddr)
            {
                let slot = if effect {
                    &mut summary.blocker_after_effect
                } else {
                    &mut summary.blocker_effect_free
                };
                slot.get_or_insert(*fnaddr);
                if on_blocker(*fnaddr, effect) {
                    // The walk stops here, so this op has no successors.
                    continue;
                }
            }
        } else if d.opname.starts_with("inline_call") {
            // `inline_call_*` opens with the `d` descr operand, a two-byte
            // index into the descriptor pool, naming the callee JitCode.
            let descr_index = code.get(d.pc + 1).copied().unwrap_or(0) as usize
                | ((code.get(d.pc + 2).copied().unwrap_or(0) as usize) << 8);
            if let Some(callee) = callee_summary(descr_index, effect) {
                // Entering with an effect behind us turns every blocker the
                // callee holds into one that cannot be rewound.
                let after = if effect {
                    callee.blocker_after_effect.or(callee.blocker_effect_free)
                } else {
                    callee.blocker_after_effect
                };
                if let Some(blocker) = after {
                    summary.blocker_after_effect.get_or_insert(blocker);
                } else if let Some(blocker) = callee.blocker_effect_free {
                    summary.blocker_effect_free.get_or_insert(blocker);
                }
                if callee.may_execute_effect {
                    effect = true;
                    summary.first_effect_pc.get_or_insert(d.pc);
                }
                // The descent enters this callee, so a region of it the callee's
                // own scan could not read is a region of this descent.
                summary.body_not_walked |= callee.body_not_walked;
            } else {
                // The `d` operand did not resolve to a JitCode descr, so this
                // op's callee is a body the scan holds nothing about.  Falling
                // through silently would read it as a callee that executes no
                // effect and holds no blocker — the most permissive answer
                // available, arrived at by not looking.  It is the same
                // condition `body_not_walked` already names one level down: a
                // region of this descent the scan did not walk.
                summary.body_not_walked = true;
                effect = true;
                summary.first_effect_pc.get_or_insert(d.pc);
            }
        }
        // A store into an object this body allocated is not an effect: the
        // allocation goes with the trace, so a rewind leaves nothing that could
        // read what the store wrote.  The three `_gc` store families all put
        // the written object in their first operand.  A residual is spared
        // when its calldescr reports no undoable effect (`rlib/jit.py
        // elidable` / `not_in_trace`), or when its funcbox names a helper
        // the walk's `provably_side_effect_free` already spares by address.
        let residual_odometer_exempt = d.opname.starts_with("residual_call")
            && (descr_operand_index(code, &d).is_some_and(|index| call_effect_free(index))
                || code
                    .get(d.pc + 1)
                    .and_then(|&slot| known_i.get(slot as usize).copied().flatten())
                    .is_some_and(effect_free_residual_fnaddr));
        let applies_effect = descent_op_applies_effect(d.opname)
            && !residual_odometer_exempt
            && !heap_write_into_fresh_object(code, &d, &fresh_r);
        // Every blocker a body reports reads as `after_effect` once one early
        // op arms the flag, so the reachable set alone does not say what to
        // repair.  Name each op that arms it — with the helper a residual's
        // funcbox resolves to, and the field a store writes — so the first one
        // on the entry path is readable.
        if !effect && applies_effect && fbw_inline_diag_enabled() {
            let funcbox = d
                .opname
                .starts_with("residual_call")
                .then(|| code.get(d.pc + 1).copied())
                .flatten()
                .and_then(|slot| known_i.get(slot as usize).copied())
                .flatten();
            let field = descr_operand_index(code, &d).and_then(|index| {
                crate::jitcode_runtime::descr_ref_table()
                    .at(index)
                    .and_then(|descr| descr.as_field_descr().map(|f| f.field_name().to_string()))
            });
            // A bare address names nothing on its own, and the registry that
            // resolves it is the same one the funcbox constant came from.
            let helper = funcbox.and_then(|addr| {
                pyre_interpreter::jit_trace_fnaddrs()
                    .iter()
                    .find(|(_, registered)| *registered == addr)
                    .map(|(path, _)| (*path).to_string())
            });
            eprintln!(
                "[descent-effect-origin] pc={} op={} funcbox={} field={} helper={}",
                d.pc,
                d.opname,
                funcbox.map_or_else(|| "-".to_string(), |addr| format!("{addr:#x}")),
                field.as_deref().unwrap_or("-"),
                helper.as_deref().unwrap_or("-"),
            );
        }
        if applies_effect {
            effect = true;
            summary.first_effect_pc.get_or_insert(d.pc);
        }
        if effect {
            summary.may_execute_effect = true;
        }

        // An op writes at most one register, named by the argcode suffix after
        // `>` and encoded as the instruction's last byte.  Carry every Int
        // result whose inputs are already known; a red input or an unmodelled
        // operation clears the destination back to unknown.
        if d.argcodes
            .split_once('>')
            .is_some_and(|(_, dst)| dst == "i")
            && let Some(&dst) = code.get(d.next_pc.wrapping_sub(1))
        {
            let carried = known_int_result(code, &d, &known_i, &known_array_len_r);
            if let Some(slot) = known_i.get_mut(dst as usize) {
                *slot = carried;
            }
        }
        // The Ref-bank destination color, if this op writes one: the dst byte
        // is the last operand.  Both facts below are keyed on it.
        if d.argcodes
            .split_once('>')
            .is_some_and(|(_, dst)| dst == "r")
            && let Some(&dst) = code.get(d.next_pc.wrapping_sub(1))
        {
            // A Ref-bank write is fresh when a `new*` op produced it or a
            // `ref_copy` carried that ownership; any other producer may name
            // live heap.
            let fresh = d.opname.starts_with("new")
                || (d.key == "ref_copy/r>r"
                    && code
                        .get(d.pc + 1)
                        .and_then(|&src| fresh_r.get(src as usize).copied())
                        .unwrap_or(false));
            if let Some(slot) = fresh_r.get_mut(dst as usize) {
                *slot = fresh;
            }
            // The wrapper-argument array can move between Ref colors before its
            // length check.  Only a plain ref copy preserves that entry fact.
            let carried = (d.key == "ref_copy/r>r")
                .then(|| code.get(d.pc + 1))
                .flatten()
                .and_then(|&src| known_array_len_r.get(src as usize).copied())
                .flatten();
            if let Some(slot) = known_array_len_r.get_mut(dst as usize) {
                *slot = carried;
            }
        }

        let state = DescentPoint {
            effect,
            known_i: known_i.clone(),
            known_array_len_r: known_array_len_r.clone(),
            fresh_r: fresh_r.clone(),
        };
        let label = label_operand_offset(d.argcodes).map(|off| read_label(code, &d, off));
        match d.opname {
            // Frame exits: the walk leaves this body here.
            "ref_return" | "int_return" | "float_return" | "void_return" => {}
            // A raise leaves the body only when nothing in it catches.
            // `finishframe_lookahead_at` skips a `live/` and takes the
            // `catch_exception/L` that follows, so the handler edge of a caught
            // raise runs forward over bytes no fallthrough reaches — the walker
            // routes `raise/r` and `reraise/` through exactly this lookahead
            // (`mod.rs`, the current-frame arm).  Seeded like `catch_exception`
            // below and for the same reason: the state at the raise is one of
            // several the handler can be entered with.
            //
            // No body in the current build takes this edge — every one of the
            // 1764 `catch_exception` ops sits after a call, which the
            // fallthrough already reaches, and none of the 3473 `raise` /
            // `reraise` ops has one behind it.
            "raise" | "reraise" => {
                if let Some(target) =
                    crate::jitcode_dispatch::try_catch_exception_at(code, d.next_pc)
                {
                    push!(
                        points,
                        work,
                        target,
                        DescentPoint {
                            effect: true,
                            known_i: handler_known.clone(),
                            known_array_len_r: vec![None; FRESH_SLOTS],
                            fresh_r: vec![false; FRESH_SLOTS],
                        }
                    );
                }
            }
            "goto" => {
                if let Some(target) = label {
                    push!(points, work, target, state);
                }
            }
            name if name.starts_with("goto_if") => match known_goto_if_taken(code, &d, &known_i) {
                Some(true) => {
                    if let Some(target) = label {
                        push!(points, work, target, state);
                    }
                }
                Some(false) => push!(points, work, d.next_pc, state),
                None => {
                    if let Some(target) = label {
                        push!(points, work, target, state.clone());
                    }
                    push!(points, work, d.next_pc, state);
                }
            },
            "catch_exception" => {
                // The exception this handler catches can be raised by any op in
                // the region it protects, so the handler is entered with the
                // join of every state in the body rather than this op's.
                // Seeding it as effectful is the sound over-approximation of
                // that join.
                if let Some(target) = label {
                    push!(
                        points,
                        work,
                        target,
                        DescentPoint {
                            effect: true,
                            known_i: handler_known.clone(),
                            known_array_len_r: vec![None; FRESH_SLOTS],
                            fresh_r: vec![false; FRESH_SLOTS],
                        }
                    );
                }
                push!(points, work, d.next_pc, state);
            }
            "switch" => {
                // `bhimpl_switch`: a known discriminant takes one arm, or
                // the fallthrough on a miss — the same one-successor
                // reading `goto_if` already applies to a known condition.
                // A red discriminant still joins every listed arm plus
                // fallthrough (`bhimpl_switch` KeyError path).
                let disc = code
                    .get(d.pc + 1)
                    .and_then(|&slot| known_i.get(slot as usize).copied())
                    .flatten();
                let descr_index = descr_operand_index(code, &d);
                if let (Some(value), Some(index)) = (disc, descr_index)
                    && let Some(hit) = switch_descr_lookup(index, value)
                {
                    match hit {
                        Some(target) => push!(points, work, target, state),
                        None => push!(points, work, d.next_pc, state),
                    }
                } else {
                    match descr_index.and_then(|index| switch_targets(index)) {
                        Some(targets) => {
                            for target in targets {
                                push!(points, work, target, state.clone());
                            }
                            push!(points, work, d.next_pc, state);
                        }
                        None => {
                            for &target in &starts {
                                push!(points, work, target, state.clone());
                            }
                        }
                    }
                }
            }
            _ => {
                // An opcode this match does not name but that carries an `L`
                // operand is a branch nobody modelled here.  The codewriter
                // emits `int_{add,sub,mul}_jump_if_ovf` (`flatten.rs`), none of
                // which reaches an assembled body in this build — every `L`
                // carrier in the insns table is claimed above.  Following the
                // target as well as the fallthrough over-approximates the
                // graph, which can only add blockers; dropping the edge would
                // remove a region, and a region the scan never enters reports
                // no blocker.
                if let Some(target) = label {
                    push!(points, work, target, state.clone());
                }
                push!(points, work, d.next_pc, state);
            }
        }
    }

    summary
}

/// Whether an exception string-override body issues a nested Python call.  The
/// bounded string-override route inlines the override as a leaf; a nested call
/// (`CallFn` residual, or a `cond_call`/`call_assembler`/`inline_call`) forces a
/// multi-frame guard-resume snapshot the sub-walk cannot build, aborting
/// mid-recording (`LoopBearingCalleeInlineUnsupported`) and discarding the whole
/// loop.  Such a body must stay on the residual dispatch path.
pub(crate) fn exception_string_override_has_nested_call(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return true;
        };
        if d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            return true;
        }
        if d.opname.starts_with("residual_call")
            && residual_call_helper_kind_in_body(body_code, &d, callee_descr_refs)
                == Some(majit_ir::RuntimeHelperKind::CallFn)
        {
            return true;
        }
        pc = d.next_pc;
    }
    false
}

/// Active boxes for an inlined callee's OWN frame in a multi-frame snapshot
/// (#68).  The fast-path inline predicate guarantees the callee does not own a
/// virtualizable (any vable op declines the inline), so the owns_vable /
/// portal-reg / semantic-slot machinery in [`collect_outer_active_boxes`]
/// reduces to a plain per-bank `registers_{i,r,f}[live_color]` read — RPython
/// `pyjitpl.py _get_list_of_active_boxes`, banks in int → ref → float
/// order to match the `all_liveness` header layout the decoder consumes.  A
/// liveness-active register holding `OpRef::NONE` is a tracer-side invariant
/// violation (callee banks are sized to the jitcode num_regs co-published with
/// liveness), so panic loudly rather than bleed NONE into the encoder.
/// Build the inlined callee (top/innermost) snapshot frame's live box list
/// from the sub-walk register banks at the guard's carried resume coordinate.
///
/// Unlike [`collect_outer_active_boxes`], the callee sub-walk is sym-less and
/// owns no virtualizable, so none of the vable-shadow / portal-red / #124
/// kept-stack recovery applies — every live color must be present directly in
/// the sub-walk's `registers_*`.  A liveness-active color the sub-walk never
/// wrote (`OpRef::NONE` — e.g. a static-ref operand-stack slot that trace-time
/// int-specialization left only in the int bank, or a py_pc↔jit_pc round-trip
/// landing on a different liveness window) cannot be sourced, so return `Err`
/// to abort the multi-frame inline and interpret rather than
/// encode a NONE box.  `PYRE_FBW_MF_DIAG` dumps the missing color.
pub(crate) fn collect_callee_active_boxes(
    regs_i: &(impl RegisterValues + ?Sized),
    regs_r: &(impl RegisterValues + ?Sized),
    regs_f: &(impl RegisterValues + ?Sized),
    callee_jitcode_index: u32,
    callee_op_pc: usize,
    carried_jitcode_pc: i32,
) -> Result<Vec<OpRef>, DispatchError> {
    // Without a carried coordinate there is no `-live-` window to size this
    // frame's box section from, and the decoder would resume on one anyway;
    // decline the inline rather than encode against an empty window.
    if carried_jitcode_pc == majit_ir::resumedata::NO_JITCODE_PC {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    }
    // The resume decoder consumes this frame's section per the liveness at the
    // carried `jitcode_pc` (`setposition` → `get_current_position_info`), not
    // a Python-pc translation. Query the same carried coordinate so the
    // encoder's box bank set agrees with the decoder's section sizes. A
    // mismatched window let a callee that int-specializes a param encode a Ref
    // where the decoder expects an int → `getvirtual_int: not a raw virtual`.
    let banks = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
        callee_jitcode_index as i32,
        carried_jitcode_pc,
    );
    let mut active = Vec::with_capacity(banks.int.len() + banks.ref_.len() + banks.float.len());
    let diag = fbw_mf_diag_enabled();
    let read =
        |value: Option<OpRef>, len: usize, idx: u32, name: &str| -> Result<OpRef, DispatchError> {
            match value {
                Some(v) if v != OpRef::NONE => Ok(v),
                other => {
                    if diag {
                        eprintln!(
                            "[fbw-mf-diag] decline: callee {name} reg {idx} {} \
                         (callee_jitcode_index={callee_jitcode_index}, \
                         bank_len={}, live_i={:?} live_r={:?} live_f={:?})",
                            if other.is_none() {
                                "out-of-range"
                            } else {
                                "holds OpRef::NONE"
                            },
                            len,
                            banks.int,
                            banks.ref_,
                            banks.float,
                        );
                    }
                    Err(DispatchError::callee_inline_unsupported(callee_op_pc))
                }
            }
        };
    for &idx in &banks.int {
        active.push(read(
            regs_i.get_box(idx as usize),
            regs_i.len(),
            idx,
            "int",
        )?);
    }
    for &idx in &banks.ref_ {
        active.push(read(
            regs_r.get_box(idx as usize),
            regs_r.len(),
            idx,
            "ref",
        )?);
    }
    for &idx in &banks.float {
        active.push(read(
            regs_f.get_box(idx as usize),
            regs_f.len(),
            idx,
            "float",
        )?);
    }
    Ok(active)
}

/// Whether a loop-carried operand parked below the call keeps the fold
/// worthwhile.
///
/// A numeric accumulator (`a` in `a += r(d)`) does: the fold's resume rebuilds
/// it from the caller's own register writeback, and folding the call is what
/// turns a residual recursion back into an assembler-to-assembler jump.
///
/// A live iterator does not, for a recursive callee.  Folding a self-call that
/// sits under a `FOR_ITER` iterator costs the caller a compiled loop and a
/// bridge — measured on
/// `recursive_call_frame_relocation` (`for k in range(n): r += cat(n - 1)`) as
/// `loops_compiled 3 -> 2`, `bridges_compiled 3 -> 2` and 0.18s -> 0.22s.  The
/// residual path keeps that loop, so leave the iterator-bearing shape to it.
fn loop_carried_slot_keeps_fold_profitable<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    slot: OpRef,
) -> bool {
    match ctx.trace_ctx.box_value(slot) {
        Some(majit_ir::Value::Int(_)) | Some(majit_ir::Value::Float(_)) => true,
        Some(majit_ir::Value::Ref(r)) => {
            let obj = r.as_usize();
            obj != 0 && unsafe { pyre_object::is_int(obj as pyre_object::PyObjectRef) }
        }
        _ => false,
    }
}

/// Whether the call at `op_pc` in the walk's own code sits inside a protected
/// region its `GUARD_NO_EXCEPTION` deopt would have to resume into.
///
/// Asked of the CALL's own coordinate, not of the whole body: the deopt
/// resumes at this call site, so a `try` elsewhere in the function is not a
/// handler it can reach.  `lookup_exceptiontable` answers exactly that question
/// and returns the innermost covering entry.
///
/// The `depth != 0 || !lasti` test excludes the whole-body, depth-zero,
/// `lasti` entry 3.14 wraps around every generator, which exists only to
/// convert an escaping `StopIteration` at the generator boundary and catches
/// nothing at a call site.  Same discrimination, same reason, as
/// `code_yields_inside_try` (`pycode.rs`) applies to decide whether a
/// generator really yields inside a `try`.
fn call_site_inside_protected_region<Sym: WalkSym>(sym: &Sym, op_pc: usize) -> bool {
    if sym.jitcode().is_null() {
        return false;
    }
    let jitcode = unsafe { &*sym.jitcode() };
    let raw = jitcode.payload.code_ptr;
    if raw.is_null() {
        return false;
    }
    let code = unsafe { &*raw };
    // `lookup_exceptiontable` keys on the byte offset of the instruction
    // itself, and `containing_py_pc_for_jitcode_pc` gives the CALL's own
    // Python instruction index — so the offset is `py_pc * 2`, without the
    // `next_instr` back-step a resume coordinate would need.
    let call_py_pc =
        crate::py_coord::containing_py_pc_for_jitcode_pc(&jitcode.payload.metadata, op_pc);
    matches!(
        pyre_interpreter::pycode::lookup_exceptiontable(&code.exceptiontable, call_py_pc * 2),
        Some((_target, depth, lasti)) if depth != 0 || !lasti
    )
}

/// Whether a callee that is not the walk's own code carries a body this fold
/// can represent.
///
/// This screen is the fold's own precondition, not upstream's.  The whole call
/// collapses into one `CALL_ASSEMBLER` plus
/// `GUARD_NOT_FORCED`/`GUARD_NO_EXCEPTION`.  A `raise` that crosses the
/// boundary needs nothing more: the callee's compiled trace propagates it and
/// `GUARD_NO_EXCEPTION` resumes the caller into its unwind, as
/// `do_recursive_call(assembler_call=True)` (`pyjitpl.py`) does for every
/// callee it reaches.  What the fold cannot represent is a protected region
/// inside the callee or an `abort_permanent` marker; those stay a residual,
/// the direction `look_inside_graph` (`codewriter/policy.py`) answers a "no"
/// with.
///
/// Not asked of the walk's own code: `call_site_inside_protected_region` already
/// read that body's table at this very call site.
fn foreign_callee_admits_call_assembler(w_code: *const ()) -> bool {
    let Some(facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return false;
    };
    !facts.has_exception_table && !facts.has_abort_permanent
}

/// #62: full-body-walk direct `CALL_ASSEMBLER` for a self-recursive call
/// at the inline recursion-bound boundary.
///
/// When the FBW inline depth for a callee reaches the warmstate's live
/// `max_unroll_recursion`
/// the call would otherwise degrade to a generic may-force residual, which
/// re-enters the callee through the func-entry residency door — one
/// heavyweight frame build + entry-bridge per recursive call (the
/// `fib_recursive` ~30x slowdown).  This emits instead the direct
/// assembler->assembler jump: `CALL_ASSEMBLER_R` to the callee's own
/// loop/pending token (mirror of `_opimpl_recursive_call`
/// `recursion_exceeded -> assembler_call`, `pyjitpl.py`, and
/// `do_residual_call`'s assembler branch, `pyjitpl.py`).
///
/// The `fib` shape and any other red-box arguments: positional arguments to a
/// self-recursive (`callee code == portal code`) callee whose frame is
/// `ncells == 0`, non-global-storing, and inline-buildable via
/// [`crate::helpers::emit_new_pyframe_inline_self_recursive`]
/// (Branch A of the retired trait-side callee-frame path).  Any unmet
/// precondition returns `Ok(None)` *before* recording any IR, so the call
/// falls back to the proven (slow) residual path.  No callable-identity
/// guard is emitted: matching the trait's self-recursive arm, the function
/// identity is pinned upstream by the same `LOAD_GLOBAL` machinery the
/// residual path already relies on.
///
/// Parity note: upstream `_opimpl_recursive_call` (`pyjitpl.py`)
/// counts same-greenkey portal frames on the framestack and flips to
/// `assembler_call` only at `count >= memmgr.max_unroll_recursion`,
/// inlining (`perform_call`) below the bound.  The inline attempt immediately
/// preceding this function applies the same count, including reconstructed
/// carrier frames, so reaching this fold at the recursion boundary has the
/// same transition.
pub(crate) fn try_walker_call_assembler_self_recursive<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    runtime_helper: majit_ir::RuntimeHelperKind,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // non-emitting eligibility checks (free to bail with Ok(None))
    // Authoritative walks only: the CALL_ASSEMBLER record + walk-commit
    // bookkeeping is FBW machinery; a non-authoritative context (the
    // diagnostic probe, tests) records the plain residual instead.
    if !ctx.is_authoritative_executor {
        return Ok(None);
    }
    // A `CALL_ASSEMBLER` jumps straight into the callee's own compiled trace,
    // which begins at the callee's first bytecode — past `pyframe.py
    // execute_frame` and so past the `ec.call_trace` / `ec.return_trace` it
    // brackets the eval loop with.  While a hook is installed there is nothing
    // that can report those events from here, so the call stays a residual and
    // reaches `execute_frame`.  Same test and same reason as the inline
    // decline in `try_walker_inline_resolved_user_call_inner`.
    if super::ec_hook_installed() {
        return Ok(None);
    }
    let is_being_profiled = ctx.session.borrow().is_being_profiled;
    // Only a genuine `call_fn` residual is a candidate — every
    // container/builtin helper carries a distinct tag.
    if runtime_helper != majit_ir::RuntimeHelperKind::CallFn {
        return Ok(None);
    }
    // Positional args only (`r_args = [callable, null_or_self, arg0, ..]`).
    // A statement call (`plain(n+1)`) is void (`residual_call_r_v`); a
    // value-producing call is Ref.  Declining void kept that self-rec
    // on an inlined jump with no `execute_frame` depth charge, so a
    // JIT-hot recursion ran past `sys.getrecursionlimit()`.
    if (dst_bank != 'r' && dst_bank != 'v') || r_args.len() < 3 {
        return Ok(None);
    }
    // The handler test this fold needs is positional and lives below, once the
    // caller's jitcode is in hand: see `call_site_inside_protected_region`.
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let ConcreteValue::Ref(callable) = arg_concretes[0] else {
        return Ok(None);
    };
    if callable.is_null() {
        return Ok(None);
    }
    // Plain-call shape only: a non-null `null_or_self` is a method
    // receiver `bh_call_fn_impl` would prepend as arg0; an unknown
    // concrete cannot be proven plain.  Either way, decline to the
    // residual call.
    let ConcreteValue::Ref(null_or_self) = arg_concretes[1] else {
        return Ok(None);
    };
    if !null_or_self.is_null() {
        return Ok(None);
    }
    // The callable must be a plain Python function with N positional
    // parameters and no closure.  Unlike the inline path this does NOT
    // require a leaf body: the callee runs as its own compiled loop reached
    // through `CALL_ASSEMBLER`, not traced through — so a branchy
    // self-recursive body (`fib`'s `if n < 2`) is eligible here even though
    // `callee_fast_path_inlinable` declines it for inlining.
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(callable) })
    else {
        return Ok(None);
    };
    // Positional args fill `locals[0..co_argcount]`. Keyword-only defaults,
    // the empty `*args` tuple and the empty `**kwargs` dict are filled below
    // from the function object (`_match_signature`), so a scope that owns
    // those locals is not declined here. A count that is not exactly the
    // positional arity still leaves a hole this builder does not fill.
    let npositional = r_args.len() - 2;
    if has_closure || npositional != nparams || nparams > FBW_REC_CA_MAX_PARAMS {
        return Ok(None);
    }
    // Positional arguments are red boxes. `_opimpl_recursive_call` /
    // `direct_assembler_call` (`pyjitpl.py`) record `call_assembler` for any
    // red, including a `W_LongObject` accumulator. A null or non-ref concrete
    // is not a box this frame build can store. The box is stored as itself:
    // the callee portal's entry (`GuardClass` / `guard_value` on `w_class`,
    // recorded from the red local) is what rejects a later class. Re-boxing
    // an exact int into a fresh `W_IntObject` would make that guard see only
    // the manufactured class.
    for i in 0..nparams {
        let ConcreteValue::Ref(arg_obj) = arg_concretes[2 + i] else {
            return Ok(None);
        };
        if arg_obj.is_null() {
            return Ok(None);
        }
    }
    // The outer portal sym (the only materialized frame across sub-walks)
    // via the FBW thread-local — the same read mechanism
    // `walker_capture_snapshot_for_last_guard` uses.  Null outside a
    // production full-body walk (arm/shadow/diagnostic), in which case the
    // sym.frame / sym.execution_context reds are unavailable: bail.
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let caller_frame = sym.frame();
    // A CALL_ASSEMBLER raising inside a `try` body must route its
    // GUARD_NO_EXCEPTION deopt into the handler, and this fold cannot encode
    // that resume in its snapshot.  Ask whether THIS call site is protected
    // rather than whether the body carries a handler anywhere: the latter
    // reads a property of the function, not of the call, and it excludes
    // every generator outright — 3.14 wraps each generator body in a
    // whole-body entry, so a `while` loop in a generator that calls nothing
    // protected was declining on a handler it can never reach.
    if call_site_inside_protected_region(sym, op.pc) {
        if p2_diag_enabled() {
            eprintln!("[p2-ca] decline pc={} reason=call-inside-try", op.pc);
        }
        return Ok(None);
    }
    // `is_self_recursive = callee code == portal code`. During
    // recording `we_are_jitted()` is false, so `function_get_code` (the
    // `w_code` already in hand) equals `getcode` — the pointer the
    // jit_merge_point green key and the portal jitcode were registered
    // under.
    let caller_code = unsafe {
        pyre_interpreter::live_code_wrapper((*sym.jitcode()).raw_code() as *const ()) as *const ()
    };
    // Self-fold requires callee code == portal code.  The full-portal cutover
    // additionally admits a *mutual*-recursive callee — one whose code is
    // already on the inline framestack, i.e. a
    // genuine recursion cycle (`is_even` → `is_odd` → `is_even` at the unroll
    // cap).  The emit below keys on `w_code` (callee-agnostic); the token
    // is resolved / synthesised per `callee_key` via `get_assembler_token`.
    //
    // Any other callee is admitted only through
    // [`foreign_callee_admits_call_assembler`].  That screen is this fold's own
    // precondition and has no counterpart in `_opimpl_recursive_call`
    // (`pyjitpl.py`), which puts no condition at all on `assembler_call` — once
    // control reaches the fall-through it emits unconditionally.  Without the
    // screen an arbitrary foreign call (e.g. a CALL_KW-bearing leaf) would fold
    // to CALL_ASSEMBLER, building and entering a frame the callee's own loop
    // was never traced against.
    if w_code as usize != caller_code as usize {
        let admit_mutual = ctx
            .session
            .borrow()
            .framestack
            .iter()
            .any(|f| f.w_code == w_code as usize);
        if !admit_mutual && !foreign_callee_admits_call_assembler(w_code) {
            if p2_diag_enabled() {
                eprintln!("[p2-ca] decline pc={} reason=not-self-nor-mutual", op.pc);
            }
            return Ok(None);
        }
    }
    // The operand stack below the call's own operands (`r_args = [callable,
    // null_or_self, arg0..]`) may hold loop-carried input args: the enclosing
    // loop's `FOR_ITER` iterator, or an accumulator reloaded for `+=`.  Only
    // the iterator disqualifies the fold — see
    // `loop_carried_slot_keeps_fold_profitable`.  Declining every loop-carried
    // operand instead kept a recursion called from a `while` body on the
    // residual path permanently: `r(8)` a million times over measured 1.23s
    // against 0.07s for the same run with the numeric shape admitted, because
    // every recursive call re-entered through the func-entry residency door.
    // The loopless `fib` shape keeps only within-iteration temps (a prior call
    // result), no InputArg, and was foldable either way.
    //
    // The screen is about a recursion re-entering through the loop it sits
    // in.  A foreign, non-recursive callee called from a `for` body is the
    // shape `do_recursive_call(assembler_call=True)` (`pyjitpl.py`) folds for
    // any callee with its own loop token; declining it left the whole callee
    // as a residual call per iteration.
    let recursive_callee = w_code as usize == caller_code as usize
        || ctx
            .session
            .borrow()
            .framestack
            .iter()
            .any(|f| f.w_code == w_code as usize);
    if recursive_callee && ctx.vstack_valid {
        let kept_below = ctx
            .frame_state
            .borrow()
            .vstack_boxes
            .len()
            .saturating_sub(r_args.len());
        if ctx.frame_state.borrow().vstack_boxes[..kept_below]
            .iter()
            .any(|slot| slot.is_input_arg() && !loop_carried_slot_keeps_fold_profitable(ctx, *slot))
        {
            if p2_diag_enabled() {
                eprintln!(
                    "[p2-ca] decline pc={} reason=loop-carried-inputarg-below",
                    op.pc
                );
            }
            return Ok(None);
        }
    }
    // A foreign (non self-recursive) non-pure residual already executed
    // concretely earlier in this walk (e.g. `events.append(n)` ahead of the
    // self-call).  Folding to CALL_ASSEMBLER terminates the walk symbolically;
    // a later value-unavailable decline then leaves it uncommittable, so the
    // interpreter replays the region and double-applies that mutation.  Decline
    // to the plain residual path, which eagerly executes and commits the call.
    if fbw_executed_body_residual() {
        if p2_diag_enabled() {
            eprintln!("[p2-ca] decline pc={} reason=executed-body-residual", op.pc);
        }
        return Ok(None);
    }
    // Branch A frame shape only: `ncells == 0`, non-global-storing callee.
    let raw = unsafe { pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef) }
        as *const pyre_interpreter::CodeObject;
    if raw.is_null() {
        return Ok(None);
    }
    let callee_code = unsafe { &*raw };
    if pyre_interpreter::ncells(callee_code) != 0 {
        return Ok(None);
    }
    let kwonly_count = callee_code.kwonlyarg_count as usize;
    let has_varargs = callee_code
        .flags
        .contains(pyre_interpreter::CodeFlags::VARARGS);
    let has_varkeywords = callee_code
        .flags
        .contains(pyre_interpreter::CodeFlags::VARKEYWORDS);
    // `def f():` has nothing to store. A zero-arg scope that owns kwonly /
    // `*args` / `**kwargs` still has slots `_match_signature` fills.
    if nparams == 0 && kwonly_count == 0 && !has_varargs && !has_varkeywords {
        return Ok(None);
    }
    let total_slots =
        nparams + kwonly_count + usize::from(has_varargs) + usize::from(has_varkeywords);
    if total_slots > FBW_REC_CA_MAX_PARAMS {
        return Ok(None);
    }
    // Resolve before any IR. A missing kwonly default is a TypeError the
    // residual call raises; this builder only fills entries the mapping has.
    let kwonly_resolved = if kwonly_count == 0 {
        None
    } else {
        match unsafe { kwonly_defaults_for_inline(callable, w_code, nparams, kwonly_count) } {
            Some(resolved) => Some(resolved),
            None => return Ok(None),
        }
    };
    let empty_vararg = if has_varargs {
        let concrete = pyre_object::w_tuple_new_array_backed(Vec::new());
        if concrete.is_null() || majit_gc::can_move(majit_ir::GcRef(concrete as usize)) {
            return Ok(None);
        }
        Some(concrete)
    } else {
        None
    };
    // The callee's globals OBJECT (`function.w_func_globals_obj`) for the
    // `frame_stores_global` stamp.
    let callee_globals_obj = unsafe { pyre_interpreter::function_get_globals_obj(callable) };
    if unsafe {
        pyre_interpreter::w_code_frame_stores_global(
            w_code as pyre_object::PyObjectRef,
            callee_globals_obj,
        )
    } {
        return Ok(None);
    }
    // Resolve the callee's own loop or trace-in-progress marker with the live
    // W_Code object used by `PyFrame.pycode`, the portal greens, and
    // `JitCell.comparekey`, at `pc = 0` (function entry).  A pending token
    // only proves the callee is
    // being traced; emission below resolves compiled-or-tmp so the descr never
    // carries a bodyless token.
    let (driver, _) = crate::driver::driver_pair();
    let callee_key = crate::driver::make_green_key_typed(w_code, 0, is_being_profiled);
    // `get_jitcell(*greenargs)` — the token and the later loop attach share
    // this cell. The hash alone would file a comparekey-less sibling.
    let greenboxes = [
        majit_ir::Value::Int(0),
        majit_ir::Value::Int(is_being_profiled as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(w_code as usize)),
    ];
    let red_types = [Type::Ref, Type::Ref];
    let token =
        match driver.get_or_make_portal_assembler_token_arc(&callee_key, &greenboxes, &red_types) {
            Some(token) => token,
            None => {
                if p2_diag_enabled() {
                    eprintln!("[p2-ca] decline pc={} reason=synth-failed", op.pc);
                }
                return Ok(None);
            }
        };
    if p2_diag_enabled() {
        eprintln!("[p2-ca] EMIT pc={} token={}", op.pc, token.number);
    }

    // emission
    // The token is the callee this callable resolved to. A loop-carried
    // callable (`op` in `op(value)`) is not pinned by LOAD_GLOBAL, so a later
    // function must fail this guard instead of entering the traced callee.
    // Same `GuardValue` the inline path records on a non-constant callable.
    if !r_args[0].is_constant() {
        ctx.trace_ctx.set_opref_concrete(
            r_args[0],
            majit_ir::Value::Ref(majit_ir::GcRef(callable as usize)),
        );
        let expected = ctx.trace_ctx.const_ref(callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[r_args[0], expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(r_args[0], expected);
    }
    // Past this point every step records IR; `?` propagation aborts the
    // whole walk (the trace is discarded), the correct failure mode for a
    // recording error.
    let nlocals = callee_code.varnames.len();
    let max_stack = callee_code.max_stackdepth as usize;

    // `direct_assembler_call` stores the red boxes it was given. The callee
    // entry guards their class; manufacturing a `W_IntObject` here would
    // hide a subclass from that guard.
    //
    // `_match_signature` then fills keyword-only locals from `w_kw_defs`,
    // writes `()` into the `*args` slot, and a fresh dict into `**kwargs`.
    // The cell read is the same live load the inline seeder emits, so an
    // in-place `__kwdefaults__` store is visible on the next call.
    let mut param_boxes: Vec<OpRef> = r_args[2..2 + nparams].to_vec();
    if let Some(resolved) = kwonly_resolved {
        let kw_defs_descr = crate::descr::function_w_kw_defs_descr();
        if r_args[0].is_constant() {
            let callable_const = ctx.trace_ctx.const_ref(callable as i64);
            crate::state::record_quasiimmut_field(ctx.trace_ctx, callable_const, kw_defs_descr);
        } else {
            walker_guard_function_field(
                ctx,
                op.pc,
                r_args[0],
                kw_defs_descr,
                resolved.mapping as i64,
            )?;
        }
        let strategy_const = ctx.trace_ctx.const_ref(resolved.strategy as i64);
        crate::state::record_quasiimmut_field(
            ctx.trace_ctx,
            strategy_const,
            crate::descr::module_dict_version_descr(),
        );
        walker_flush_guard_not_invalidated(ctx, op.pc)?;
        for kwonly in &resolved.values {
            if crate::state::module_dict_cell_value_direct(resolved.mapping, kwonly.slot)
                != Some(kwonly.stored)
            {
                return Err(DispatchError::KwonlyDefaultsMappingRacedRecord { pc: op.pc });
            }
            let Some((value_op, _)) = emit_namespace_cell_value(ctx, op.pc, kwonly.stored)? else {
                return Err(DispatchError::KwonlyDefaultsMappingRacedRecord { pc: op.pc });
            };
            param_boxes.push(value_op);
        }
    }
    if let Some(concrete) = empty_vararg {
        param_boxes.push(ctx.trace_ctx.const_ref(concrete as i64));
    }
    if has_varkeywords {
        // `_match_signature` does `space.newdict(kwargs=True)` on every call.
        // The dict is mutable, so it cannot be the shared empty constant.
        // Walk `w_dict_new_kwargs` so the constructor records instead of a
        // residual helper standing in for it.
        let Some(dict_op) = super::specialize::try_walker_orthodox_kwargs_dict_new(ctx, op.pc)?
        else {
            return Ok(None);
        };
        param_boxes.push(dict_op);
    }

    // `ec` is the portal's second red (`interp_jit.py reds=['frame', 'ec']`).
    // It is shared by every recursive MIFrame and never recovered from the
    // app-level frame object.
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    let ec = unsafe { &*sym_ptr }.execution_context();
    if ec.is_none() {
        return Ok(None);
    }

    // `function.py` `_immutable_fields_ = ['code?', ...]`. The assembler
    // token and this frame's pycode come from the function's code. A
    // `GuardValue` on the function object does not watch that field, so
    // `f.__code__ = ...` would keep entering the old body. The `?` marker
    // invalidates the trace instead.
    walker_pin_function_code(ctx, op.pc, callable)?;

    // Build the callee PyFrame inline (Branch A): a single positional
    // local, no cells, constant code / globals.
    let pycode_const = ctx.trace_ctx.const_ref(w_code as i64);
    let w_globals_obj_const = ctx.trace_ctx.const_ref(callee_globals_obj as i64);
    // The `frame_stores_global` decline above is what makes this frame shape
    // able to answer for `callee_globals_obj`; emission has already started, so
    // a disagreement here is an aborted walk rather than a residual.
    let Some(callee_frame) = crate::helpers::emit_new_pyframe_inline_with_params(
        ctx.trace_ctx,
        &param_boxes,
        &[],
        &[],
        0,
        nlocals + max_stack,
        nlocals,
        pycode_const,
        w_globals_obj_const,
    ) else {
        return Err(DispatchError::callee_inline_unsupported(op.pc));
    };

    // `pyframe.py execute_frame`'s `ec.enter(self)`, which upstream records
    // ahead of the callee's `jit_merge_point` and this fold otherwise jumps
    // over: without it the activation the CALL_ASSEMBLER runs never reaches
    // `topframeref`, so `sys._getframe().f_back` walks straight from the
    // callee's callee to this caller and every level in between is missing.
    // `record_activation_charge` records the stack check that stands in front
    // of that `enter`, and documents why.  It pays for the inlined levels
    // above it as well: this fold re-enters the fragment that inlined them, so
    // charging one unit here would count one activation per fragment and
    // divide the recursion's depth by the inline width.
    // `open_inline_activations` is that width, and it is charged here because
    // this is the seam a guard cannot leave unbalanced.
    let units = ctx
        .trace_ctx
        .const_int(i64::from(open_inline_activations(ctx.session) + 1));
    let (saved_depth, displaced_activation) =
        record_activation_charge(ctx, op.pc, ec, callee_frame, units)?;
    record_ec_enter_frame_chain(ctx.trace_ctx, callee_frame, ec);

    // do_residual_call step 1 (`pyjitpl.py`): FORCE_TOKEN +
    // SETFIELD_GC(vable_token) before the assembler call.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);

    // pyjitpl.py `execute_and_record_varargs(CALL_MAY_FORCE_R)`:
    // the forces branch EXECUTES the call during tracing —
    // `direct_assembler_call` (pyjitpl.py) only rewrites the
    // already-recorded op into CALL_ASSEMBLER afterwards, so the result
    // box always carries the executed value. The retired call-replay leg's
    // `trace_guarded_int_payload(ca_result)` consumed the same concrete
    // result (trace_opcode.rs).
    // Without the stamp the downstream BINARY_OP on two recursive-call
    // results cannot take the int specialization and records the generic
    // dunder-dispatch residual instead — the compiled loop then runs the
    // full `lookup_where`/type-dispatch chain per call.  Reuse the
    // residual executor primitive: it brackets the active vable with the
    // TOKEN_TRACING_RESCALL protocol, suspends re-entrant trace
    // continuation across the callee's `jit_merge_point`, stamps
    // `ca_result` with the executed concrete on success, and seeds the
    // standing exception state on a raise.
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let allboxes = build_allboxes(funcptr, r_args, &argbox_types, call_descr.arg_types(), None);
    let exec = {
        let _selfrec_ca_fold_guard = SelfRecCaFoldGuard::enter();
        try_execute_residual_call_via_executor(
            ctx,
            OpCode::CallMayForceR,
            &allboxes,
            call_descr,
            OpRef::NONE,
            op.pc,
            None,
            false,
        )?
    };
    // `pyjitpl.py:2049-2079` checks vrefs after concrete execution, records
    // CALL_ASSEMBLER, then emits GUARD_NOT_FORCED.  In particular,
    // VIRTUAL_REF_FINISH must precede the call so the call and guard remain
    // adjacent and the backend can arm the JIT frame's force descriptor.
    let ca_result = ctx.trace_ctx.call_assembler_red_only_ref_arc(
        token,
        &[callee_frame, ec],
        &[Type::Ref, Type::Ref],
    );
    if let ResidualExecOutcome::Executed(Ok(result)) = exec {
        ctx.trace_ctx.set_opref_concrete(
            ca_result,
            majit_ir::Value::Ref(majit_ir::GcRef(result as usize)),
        );
    }
    // A decline leaves the CALL_ASSEMBLER recorded symbolically WITHOUT
    // running it — a side effect only the legacy replay applies, so the
    // walk-end no-replay commit must stay off for this trace (see
    // `fbw_has_unjournaled_effect`).
    let exec_raised = match exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };

    // `pyjitpl.py do_residual_call` step 5: invalidate on the
    // CALL_MAY_FORCE executed above, not the recorded CALL_ASSEMBLER.
    ctx.trace_ctx.heapcache_invalidate_caches_varargs(
        OpCode::CallMayForceR,
        Some(call_descr.get_extra_info()),
        &allboxes,
    );

    // pyjitpl.py `make_result_of_lastop`: the result lands in
    // `registers_*[reg_index]` BEFORE GUARD_NOT_FORCED (2079) and
    // `handle_possible_exception` (2082).  The writeback MUST precede the
    // two guards so their after-call resume snapshots read the recorded
    // OpRef in the dst slot the resume position points at — deferring it
    // past the guards surfaces a stale box in the fail_args for the `>X`
    // slot on a raising/forcing deopt.  Mirror of the sibling residual
    // path (jitcode_dispatch.rs) and
    // `do_residual_call_walker_emit`.  `CALL_ASSEMBLER_R` yields the boxed
    // PyObject return value, taken as-is by the Ref dst (the consuming
    // BINARY_OP unboxes); a void callsite (`dst_bank == 'v'`) is eligible too
    // and its write lands nowhere.
    // Written REGARDLESS of `exec_raised`: on a raise `ca_result` is still
    // the recorded CALL_ASSEMBLER OpRef (carrying a Null concrete shadow,
    // never read on the exception path), and the after-call resume snapshots
    // must see it in the dst slot — the same unconditional non-void writeback
    // as the residual dispatcher (pyjitpl.py / 2074-2077:
    // make_result_of_lastop before handle_possible_exception for
    // get_list_of_active_boxes).
    write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, ca_result)?;

    // The CALL_ASSEMBLER fold records both post-call guards before the walk
    // reaches the next Python-opcode boundary, so advance the caller's stack
    // mirror here.  Keep any operands below the call, discard the callable,
    // NULL marker, and arguments, then put the assembler result on the new
    // TOS.  The register writeback above already carries the same result, but
    // guard snapshotting sources operand-stack slots from this mirror.
    if ctx.vstack_valid {
        let caller_jitcode = unsafe { &*sym.jitcode() };
        let caller_code = unsafe { &*caller_jitcode.payload.code_ptr };
        // #73 Slice 4: forward after-residual fallthrough coordinate; the
        // containing lookup survives only for the empty-twin class and as the
        // audit oracle.
        let legacy_resume_py_pc = || {
            let call_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(
                &caller_jitcode.payload.metadata,
                op.pc,
            ) as usize;
            crate::pyjitpl::semantic_fallthrough_pc(caller_code, call_py_pc) as u32
        };
        let resume_py_pc = match caller_jitcode
            .payload
            .after_residual_fallthrough_py_pc_populated()
            .then(|| {
                caller_jitcode
                    .payload
                    .after_residual_fallthrough_py_pc_for_jitcode_pc(op.pc)
            })
            .flatten()
        {
            Some(ft) => {
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        ft,
                        legacy_resume_py_pc(),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: self-recursive CA vstack fallthrough-py twin diverged at jit_pc {}",
                        op.pc
                    );
                }
                ft
            }
            None => legacy_resume_py_pc(),
        };
        let raw_depth = || {
            crate::liveness::liveness_for(caller_jitcode.payload.code_ptr)
                .depth_at_py_pc()
                .get(resume_py_pc as usize)
                .copied()
                .unwrap_or(0) as usize
        };
        let resume_depth = if caller_jitcode.payload.depth_after_residual_populated() {
            let depth = caller_jitcode
                .payload
                .depth_after_residual_for_jitcode_pc(op.pc)
                .unwrap_or(0) as usize;
            if pcmap_afterresidual_audit_enabled() {
                assert_eq!(
                    depth,
                    raw_depth(),
                    "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: self-recursive CA vstack depth twin diverged at jit_pc {} (py {resume_py_pc})",
                    op.pc
                );
            }
            depth
        } else {
            raw_depth()
        };
        ctx.frame_state
            .borrow_mut()
            .vstack_boxes
            .truncate(resume_depth);
        ctx.frame_state
            .borrow_mut()
            .vstack_boxes
            .resize(resume_depth, OpRef::NONE);
        if resume_depth > 0 {
            ctx.frame_state.borrow_mut().vstack_boxes[resume_depth - 1] = ca_result;
        }
        ctx.vstack_cur_pypc = resume_py_pc;
        ctx.vstack_depth = resume_depth;
        ctx.frame_state.borrow_mut().vstack_last_ref = OpRef::NONE;
        super::vstack_mirror::disarm_vstack_reorder_region(ctx);
    }

    // pyjitpl.py: GUARD_NOT_FORCED + resume snapshot advanced past
    // the call (`capture_resumedata(after_residual_call=True)`).
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    // `execute_frame`'s `finally: ec.leave(...)`.  A `finally` runs on every
    // exit, and compiled code has no such construct, so the restore is
    // recorded at the earliest point the call is known to have returned:
    // after GUARD_NOT_FORCED, which `pyjitpl.py:2049-2079` keeps adjacent to
    // the call, and BEFORE the exception guard.  Recording it past the
    // exception guard instead would skip the restore on exactly the exit that
    // needs it — a raising callee deopts at the guard, leaving the callee
    // frame published in `topframeref` for the next `enter` to chain onto, and
    // the root walker then traverses a chain that grows once per raise
    // (`walk_pyframe_roots_area` took 100% of a profile of
    // `synth/list_length_hint_validate`, which raises on four of every six
    // calls).  The residual exposure — a failing GUARD_NOT_FORCED — is the one
    // `TopFrameRefGuard` (`pyre-jit/src/eval.rs`) already covers.
    record_ec_leave_frame_chain(ctx.trace_ctx, callee_frame, ec);
    // `execute_frame`'s `finally` half of the same seam: give the activation
    // back before either exception arm leaves this compiled caller.
    record_activation_release(ctx.trace_ctx, ec, saved_depth, displaced_activation);
    // `MetaInterp.direct_assembler_call` keeps the assembler virtualizable
    // alive between GUARD_NOT_FORCED and `handle_possible_exception`. Keep it
    // last in that interval: besides preserving the frame, it prevents the
    // ordinary activation stores above from becoming the optimizer's
    // "previous op" for GUARD_NO_EXCEPTION.
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[callee_frame]);
    // pyjitpl.py `handle_possible_exception`.
    if exec_raised {
        // Raising branch (pyjitpl.py): `GUARD_EXCEPTION` with
        // the const class pin, then `finishframe_exception()` — the
        // remaining bytes of the arm never run.  Mirror of the residual
        // dispatcher's raising tail: surface `SubRaise` so `walk_loop`
        // emits the outer `FINISH(exc)` (or an outer inline frame's
        // handler catches it).
        walker_record_guard_exception(ctx, op.pc)?;
        let exc = ctx
            .last_exc_value()
            .expect("exec_raised implies last_exc_value seeded by the Err branch");
        let exc_concrete = ctx.last_exc_value_concrete();
        return Ok(Some((
            DispatchOutcome::SubRaise { exc, exc_concrete },
            op.next_pc,
        )));
    }
    // GUARD_NO_EXCEPTION on the non-raising recording path.
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;

    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// `do_recursive_call`'s call target (`pyjitpl.py`): the portal runner address
/// `warmspot.py` stamped on the driver, and the portal ABI
/// `build_portal_calldescr` derived from its `vars`.
///
/// The portal driver is the one with greens; `ensure_default_driver_sd`'s
/// placeholder has none and carries no runner address.
fn portal_runner_call_target() -> Option<(i64, majit_ir::DescrRef)> {
    let (driver, _) = crate::driver::try_driver_pair()?;
    let jd = driver
        .meta_interp()
        .staticdata
        .jitdrivers_sd
        .iter()
        .find(|jd| jd.num_greens() > 0)?;
    if jd.portal_runner_adr == 0 {
        return None;
    }
    Some((jd.portal_runner_adr, jd.portal_calldescr.clone()?))
}

/// Result of [`record_walker_loop_callee_portal_call`].
///
/// `raised` is set when the portal runner raised. The result op is recorded
/// either way: `make_result_of_lastop` runs before `handle_possible_exception`.
/// The inline wrapper stores that op into `dst`. The bridge drain threads
/// `SubRaise` and does not plant it on the caller.
pub(crate) struct WalkerLoopCalleePortalRecord {
    pub result: OpRef,
    pub raised: Option<(OpRef, ConcreteValue)>,
}

/// Inline-route wrapper around [`record_walker_loop_callee_portal_call`].
///
/// Writes the caller's result register and returns `Continue` or `SubRaise`
/// at `op.next_pc`. Early refusals (no portal runner, no concrete frame, a
/// resume depth the frame does not carry) decline here, after the shared
/// function has recorded nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_walker_loop_callee_call_assembler<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    dst_bank: char,
    dst: usize,
    callee_frame: OpRef,
    callee_ec: OpRef,
    nlocals: usize,
    token: std::sync::Arc<majit_backend::JitCellToken>,
    target_pc: usize,
    w_code: *const (),
    is_being_profiled: bool,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let _ = nlocals;
    let Some(recorded) = record_walker_loop_callee_portal_call(
        ctx,
        op.pc,
        Some((dst_bank, dst)),
        callee_frame,
        callee_ec,
        Some(token),
        target_pc,
        w_code,
        is_being_profiled,
    )?
    else {
        return resolved_inline_decline(op.pc, line!());
    };
    if let Some((exc, exc_concrete)) = recorded.raised {
        return Ok(Some((
            DispatchOutcome::SubRaise { exc, exc_concrete },
            op.next_pc,
        )));
    }
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Walker mirror of `opimpl_recursive_call_assembler`
/// (`metainterp.rs`): a multi-frame inlined callee sub-walk reached its
/// OWN loop header (surfaced as `SubLoopCalleeCallAssembler`) and a compiled
/// loop token already exists for it. The inlined prologue already populated
/// the seeded virtual callee frame's locals via `setarrayitem_vable`, so this
/// only pins the loop-entry resume position (`last_instr = target_pc - 1`) on
/// the frame, then emits `CALL_ASSEMBLER([frame, ec])` into the token —
/// forcing the virtual frame materializes the locals the compiled loop reads
/// at entry. The op sequence (vable/vref-before, CALL_ASSEMBLER + KEEPALIVE,
/// residual executor to run the callee concretely and stamp `ca_result`, dst
/// writeback, GUARD_NOT_FORCED + GUARD_NO_EXCEPTION) mirrors
/// [`try_walker_call_assembler_self_recursive`].
///
/// What the executor runs is `do_recursive_call`'s own target: the portal
/// runner on the callee frame the sub-walk already advanced to this header,
/// not the caller's original CALL.  Re-entering the callee at its entry would
/// replay the prologue the sub-walk has already executed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_walker_loop_callee_portal_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: Option<(char, usize)>,
    callee_frame: OpRef,
    callee_ec: OpRef,
    token: Option<std::sync::Arc<majit_backend::JitCellToken>>,
    target_pc: usize,
    w_code: *const (),
    is_being_profiled: bool,
) -> Result<Option<WalkerLoopCalleePortalRecord>, DispatchError> {
    debug_assert!(callee_frame != OpRef::NONE && callee_ec != OpRef::NONE);
    // `do_recursive_call`'s funcbox and ABI, resolved before the first
    // recorded op so an unwired driver declines the inline instead of
    // leaving a half-emitted CALL behind.
    let Some((portal_runner_adr, portal_descr)) = portal_runner_call_target() else {
        return Ok(None);
    };
    let Some(portal_view) = portal_descr.as_call_descr() else {
        return Ok(None);
    };
    // The live frame this resume runs, resolved under the same rule as the
    // portal target above: a callee_frame with no concrete shadow declines
    // before the `last_instr` pin and the vable/vref bookkeeping are recorded,
    // so the generic residual path that then re-enters the callee at its entry
    // does not run against a trace already carrying this fold's ops.
    let concrete_callee_frame = match ctx.trace_ctx.concrete_of_opref(callee_frame) {
        Some(majit_ir::Value::Ref(gcref)) if gcref.0 != 0 => {
            gcref.0 as *mut pyre_interpreter::PyFrame
        }
        _ => return Ok(None),
    };
    // A resume coordinate is `(last_instr, valuestackdepth)`, not `last_instr`
    // alone: a header entered with operands on the stack — a `for` header holds
    // its iterator there — executes against a height the frame has to
    // advertise, and the portal runner hands that frame to the interpreter
    // whenever its entry gate declines.  The sub-walk's own `setfield_vable_i`
    // writes keep the frame in step, so this is a check and not a publish;
    // decline rather than resume at a coordinate the frame does not carry.
    if let Some(depth_vsd) = crate::state::depth_based_vsd_for_wcode(w_code as usize, target_pc)
        && depth_vsd != unsafe { (*concrete_callee_frame).valuestackdepth }
    {
        return Ok(None);
    }
    // Bridge resume (`dst` is None). `emit_reconstructed_callee_pyframe`
    // seeds the virtual `valuestackdepth` at `stack_base`, and this walk
    // does not install `InlineConcreteFrameGuard`, so the fold that keeps
    // an inline callee's virtual in step does not run. Pin the depth the
    // check just accepted before `direct_assembler_call` forces the frame.
    // The inline path passes a dst and keeps its own setfield sequence.
    if dst.is_none()
        && let Some(depth_vsd) = crate::state::depth_based_vsd_for_wcode(w_code as usize, target_pc)
    {
        let vsd_box = ctx.trace_ctx.const_int(depth_vsd as i64);
        let vsd_descr = crate::descr::pyframe_stack_depth_descr();
        let vsd_idx = vsd_descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[callee_frame, vsd_box], vsd_descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(callee_frame, vsd_idx, vsd_box);
    }

    // Pin the loop-entry resume position on the (still-virtual) callee frame:
    // override `last_instr` from -1 (the fresh-frame entry value
    // `emit_new_pyframe_inline_with_params` wrote) to `target_pc - 1`, so the
    // compiled loop's `next_instr()` lands on the merge point. The frame's
    // `valuestackdepth` was already seeded to `nlocals` at construction (empty
    // stack at the while-header), and the locals themselves are recorded as
    // virtual-frame items by the inlined prologue stores — both flow through
    // when the CALL_ASSEMBLER forces the virtual. Uses `SetfieldGc` + a real
    // `FieldDescr` (the same field-set the builder uses), so
    // `optimize_setfield_gc` records it into the virtual's `vinfo.fields`.
    //
    // The seeded depth reads as the whole answer only for a header whose static
    // operand-stack depth is zero, which is what "empty stack at the
    // while-header" above assumes; a `for` header is entered with the iterator
    // on the stack.  Census over the 447 `pyre/bench/synth` fixtures (dynasm,
    // 84131dc4da8): 36 emits reach here, 25 at static depth 0 and 11 at depth 1
    // — all 11 in `str_search_index_bounds.py`, all with `ForIter` at
    // `target_pc`.  Measured on those 11, the frame this resume runs reads
    // `valuestackdepth` 4 against a `depth_at_py_pc` of 4: the sub-walk's own
    // `setfield_vable_i` writes carried the push, so the seed is not what the
    // frame still holds by the time the header is reached.  The gate above is
    // what keeps that true rather than assumed.  See
    // `pyre/bench/synth/_pending/loop_callee_for_header_resume.py` for the same
    // "resume pc pinned without its operand height" shape that DOES produce a
    // wrong answer, through a different writer.
    let last_instr = ctx.trace_ctx.const_int(target_pc as i64 - 1);
    let last_instr_descr = crate::descr::pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, last_instr],
        last_instr_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(callee_frame, last_instr_idx, last_instr);

    // `pyframe.py execute_frame`'s activation entry, the same seam
    // [`record_activation_charge`] records for the arm that builds its own
    // callee frame. Every compiled door begins past `execute_frame`, so the
    // test against `sys.getrecursionlimit()` has no other home. Without it the
    // limit binds neither recursion nor a deep acyclic chain for as long as
    // the calls stay compiled -- measured on a self-recursive callee that
    // carries its own loop,
    // `sys.setrecursionlimit(150)` followed by a 400-deep call returned a
    // value, and the counter it had meanwhile climbed past the limit made the
    // next ordinary activation raise `RecursionError` from a depth nothing was
    // left to unwind.
    //
    // The callee's own `OpenInlineActivation` is already closed by the time
    // this arm emits -- the sub-walk's `walker_ec_leave` drops it -- so this
    // level is the `+ 1`, exactly as in the sibling arm. The width belongs to
    // this walk's `WalkSession`; a nested residual trace on the same thread
    // owns a different session and cannot leak implementation frames into it.
    let units = ctx
        .trace_ctx
        .const_int(i64::from(open_inline_activations(ctx.session) + 1));
    let activation = record_activation_charge(ctx, pc, callee_ec, callee_frame, units)?;

    // do_residual_call step 1 (`pyjitpl.py`): FORCE_TOKEN +
    // SETFIELD_GC(vable_token) before the assembler call.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, pc);

    // `direct_assembler_call` (`pyjitpl.py`) records the CALL_ASSEMBLER with
    // exactly the target jitdriver's red args — `assert len(args) ==
    // targetjitdriver_sd.num_red_args` — and the portal reds are
    // `['frame', 'ec']` (`interp_jit.py`), the frame at
    // `index_of_virtualizable`. The callee, not the caller, unpacks the
    // virtualizable: `patch_new_loop_to_load_virtualizable_fields`
    // (`compile.py`) truncates the callee loop to `inputargs[:num_red_args]`
    // and prepends a GETFIELD_GC / GETARRAYITEM_GC per field read off that
    // vable arg. So the callee loop head dereferences a real heap frame, which
    // is why forcing the still-virtual frame here — allocation plus one
    // SETARRAYITEM_GC per known element — is the upstream op sequence rather
    // than a decline.
    // Run the callee concretely to stamp `ca_result` (same rationale as the
    // self-recursive arm: the downstream consumer needs the real concrete to
    // take its int specialization).
    //
    // What is run is the PORTAL RUNNER on the callee frame the sub-walk has
    // already advanced to this loop header — `do_recursive_call`'s funcbox is
    // `targetjitdriver_sd.portal_runner_adr` and its arguments are the merge
    // point's own greens and reds (`opimpl_jit_merge_point`'s recursive arm,
    // `pyjitpl.py`), not the caller's original CALL operands.  Resuming is the
    // whole point: the inlined prologue already ran the callee's pre-loop
    // bytecode concretely during the sub-walk, so re-entering the callee at
    // its ENTRY would apply every one of those effects a second time at trace
    // time.  `ll_portal_runner_shim` takes the frame and continues it, which
    // is also exactly what the `CALL_ASSEMBLER` recorded below does at
    // runtime, so the trace-time and compiled executions are the same
    // operation on the same frame.
    //
    // GC-rooting of the materialized callee virtualizable frame is equivalent
    // to the GC-clean self-recursive arm (a four-lens audit found no
    // content-dependent rooting defect): the frame is built with the same
    // `pyframe_size_descr()` + `pyobject_gcarray_descr()` locals array as the
    // fib frame, is JUMP-loop-carried so its slot is in every inner
    // residual-call gcmap, and the runtime `PyFrame`/array GC type registration
    // traces frame->array->elements with no int-vs-ref branch. A historical
    // GC-stress SEGV (freed, not-forwarded receiver under nursery pressure)
    // reproduced ONLY on layout-shifting diagnostic-probe builds; on clean
    // binaries it does not reproduce across the GC-stress matrix
    // (r1/r5/r6/r2/r4 × nursery {default,1M,256K,64K,16K,4K} × dynasm+x86, all
    // clean) — a diagnostic-build layout artifact, with content-agnostic
    // rooting ruling out a ref-specific defect here.
    // The recorded `last_instr` pin above describes the frame the compiled
    // trace builds; the live frame this resume runs has to carry the same
    // coordinate.  Its locals, stack slots and `valuestackdepth` are already
    // in step — `setarrayitem_vable_*` and `setfield_vable_i` mirror every
    // own-frame write into it as the sub-walk records them — but the walk
    // stopped AT the merge point, before the opcode that would have spilled
    // `last_instr` for it.
    unsafe { (*concrete_callee_frame).last_instr = target_pc as isize - 1 };
    // `do_recursive_call` bakes `portal_runner_adr`.  That address is what the
    // runtime CALL_ASSEMBLER calls when the callee has no compiled loop yet:
    // `compile_tmp_callback` routes it through `ll_portal_runner`, which begins
    // an activation.  `_portal_ptr` names the body below that entry and is
    // reached only from inside `ll_portal_runner` itself, so baking it here
    // resumes a body whose prologue never ran — the callee never enters the
    // JIT and the frame is forced instead.
    let funcbox = ctx.trace_ctx.const_int(portal_runner_adr);
    let next_instr_box = ctx.trace_ctx.const_int(target_pc as i64);
    let profiled_box = ctx.trace_ctx.const_int(is_being_profiled as i64);
    let pycode_box = ctx.trace_ctx.const_ref(w_code as i64);
    // The `ec` red is the portal's second red (`interp_jit.py`
    // `PyPyJitDriver.reds = ['frame', 'ec']`). `do_residual_call` records
    // that box. A `ConstPtr` of `getexecutioncontext()` would bake the
    // tracing thread's context into `CALL_MAY_FORCE_R`. Stamp a concrete
    // shadow on the red when the sub-walk left it empty so the trace-time
    // executor can run; the recorded operand stays `callee_ec`.
    if ctx.trace_ctx.concrete_of_opref(callee_ec).is_none() {
        super::stamp_live_execution_context(ctx.trace_ctx, callee_ec);
    }
    // `_build_allboxes` order for the portal ABI, which
    // `build_portal_calldescr` lays out in `vars` declaration order:
    // `[funcbox] + greens[next_instr, is_being_profiled, pycode] +
    // reds[frame, ec]`.
    let allboxes = [
        funcbox,
        next_instr_box,
        profiled_box,
        pycode_box,
        callee_frame,
        callee_ec,
    ];
    let exec = try_execute_residual_call_via_executor(
        ctx,
        OpCode::CallMayForceR,
        &allboxes,
        portal_view,
        OpRef::NONE,
        pc,
        None,
        false,
    )?;
    // `pyjitpl.py:2049-2079` records a forced VIRTUAL_REF_FINISH before the
    // selected CALL_ASSEMBLER, followed immediately by GUARD_NOT_FORCED.
    // `direct_assembler_call` records CALL_ASSEMBLER on the portal reds.
    // With no token, `do_recursive_call`'s `assembler_call=False` arm records
    // `direct_call_may_force` of the same portal allboxes. `allboxes` already
    // starts with the funcbox, so the descr is the portal call descr and the
    // list is not passed through `call_may_force_ref_typed` (that prepends
    // the funcbox again).
    // `do_residual_call` step 5 needs the CALL_MAY_FORCE extra_info after
    // `record_op_with_descr` takes `portal_descr`.
    let ei = portal_view.get_extra_info().clone();
    let ca_result = if let Some(token) = token {
        ctx.trace_ctx.call_assembler_red_only_ref_arc(
            token,
            &[callee_frame, callee_ec],
            &[Type::Ref, Type::Ref],
        )
    } else {
        ctx.trace_ctx
            .record_op_with_descr(OpCode::CallMayForceR, &allboxes, portal_descr)
    };
    if let ResidualExecOutcome::Executed(Ok(result)) = exec {
        ctx.trace_ctx.set_opref_concrete(
            ca_result,
            majit_ir::Value::Ref(majit_ir::GcRef(result as usize)),
        );
    }
    let exec_raised = match exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };

    // `pyjitpl.py do_residual_call` step 5: invalidate on the
    // CALL_MAY_FORCE executed above, not the recorded CALL_ASSEMBLER.
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(OpCode::CallMayForceR, Some(&ei), &allboxes);
    if let Some((dst_bank, dst)) = dst {
        write_residual_call_result_to_dst(ctx, pc, dst, dst_bank, ca_result)?;
    }

    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    // Restore before either exception arm leaves the compiled caller.  Put
    // PyPy's KEEPALIVE last in the interval so GUARD_NO_EXCEPTION still sees
    // an emitted call-adjacent operation even when the restore stores fold.
    let (saved_depth, displaced_activation) = activation;
    record_activation_release(ctx.trace_ctx, callee_ec, saved_depth, displaced_activation);
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[callee_frame]);
    if exec_raised {
        walker_record_guard_exception(ctx, pc)?;
        let exc = ctx
            .last_exc_value()
            .expect("exec_raised implies last_exc_value seeded by the Err branch");
        let exc_concrete = ctx.last_exc_value_concrete();
        return Ok(Some(WalkerLoopCalleePortalRecord {
            result: ca_result,
            raised: Some((exc, exc_concrete)),
        }));
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;

    Ok(Some(WalkerLoopCalleePortalRecord {
        result: ca_result,
        raised: None,
    }))
}

/// Fold a keyword call's `kwnames`->parameter permutation at trace time so a
/// `call_kw` reuses the positional inline path.  The constant `kwnames` tuple
/// and the callee's static parameter names are both known at record time, so
/// the reorder is a pure trace-time permutation of the argument boxes into
/// parameter order — `_match_signature` (`@jit.unroll_safe`) unrolled and
/// folded.  Once reordered the seeding is identical to a positional call.
///
/// `r_args` layout is `[callable, self_or_null, kwnames, arg0..argN-1]`; the
/// trailing `nkw` args are the keyword values, `kwnames[j]` naming
/// `arg[n_pos + j]` where `n_pos = nargs - nkw`.  `receiver`, when present,
/// is the implicit leading argument inserted by `call_kw` for method-form
/// calls (`Arguments.prepend` / `funcrun_obj` upstream).
///
/// A keyword that names no parameter is returned separately for a `**kwargs`
/// callee, which collects it into its mapping (`_match_signature`).
///
/// Returns `None` (declining to the residual call, no behavior change) for any
/// shape the seeding cannot serve: a non-constant / non-tuple `kwnames`, a
/// callee with `*args` / keyword-only parameters, more positional arguments
/// than the callee takes, a keyword naming an unknown parameter of a callee
/// without `**kwargs`, or a keyword colliding with a positionally-filled
/// parameter ("multiple values").
///
/// # Safety
/// `w_code` must be the live code object pointer for the resolved callable.
/// Whether every local of `w_code` that a call must bind lives in
/// `registers_r[0..co_argcount]`, which is all the inline seeding fills.
///
/// `*args` and `**kwargs` own locals the seeding never packs, and a
/// keyword-only parameter owns one it never reaches — and none of the three is
/// counted by `co_argcount`, so an arity check alone lets them through.
fn fbw_callee_scope_is_positional_only(w_code: *const ()) -> bool {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return false;
    }
    let flags = unsafe { (*raw).flags };
    !flags.contains(pyre_interpreter::CodeFlags::VARARGS)
        && !flags.contains(pyre_interpreter::CodeFlags::VARKEYWORDS)
        && unsafe { (*raw).kwonlyarg_count } == 0
}

/// Whether the only local of `w_code` outside `registers_r[0..co_argcount]`
/// is the `**kwargs` mapping.
fn fbw_callee_scope_is_varkw_only(w_code: *const ()) -> bool {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return false;
    }
    let flags = unsafe { (*raw).flags };
    !flags.contains(pyre_interpreter::CodeFlags::VARARGS)
        && flags.contains(pyre_interpreter::CodeFlags::VARKEYWORDS)
        && unsafe { (*raw).kwonlyarg_count } == 0
}

/// The scope slot `_match_signature` writes the vararg tuple into
/// (`_match_signature`): `co_argcount + co_kwonlyargcount`.
fn fbw_callee_vararg_slot(w_code: *const ()) -> Option<usize> {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    let flags = unsafe { (*raw).flags };
    if flags.contains(pyre_interpreter::CodeFlags::VARARGS) {
        Some(unsafe { (*raw).arg_count as usize + (*raw).kwonlyarg_count as usize })
    } else {
        None
    }
}

/// The scope slot `_match_signature` writes the `**kwargs` mapping into
/// (`_match_signature`): `co_argcount + co_kwonlyargcount + has_vararg`.
fn fbw_callee_varkw_slot(w_code: *const ()) -> Option<usize> {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    let flags = unsafe { (*raw).flags };
    if flags.contains(pyre_interpreter::CodeFlags::VARKEYWORDS) {
        Some(unsafe {
            (*raw).arg_count as usize
                + (*raw).kwonlyarg_count as usize
                + usize::from(flags.contains(pyre_interpreter::CodeFlags::VARARGS))
        })
    } else {
        None
    }
}

/// How many keyword-only locals the seeding has to fill, or `None` when the
/// callee's code object is not readable.
fn fbw_callee_kwonly_count(w_code: *const ()) -> Option<usize> {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    Some(unsafe { (*raw).kwonlyarg_count } as usize)
}

/// One keyword `_match_signature` (`argument.py`) collects into the callee's
/// `**kwargs` mapping because it names no parameter, in call order.
struct VarkwExtra {
    /// The `kwnames` entry, a `str` constant of the calling code.
    name: pyre_object::PyObjectRef,
    value: OpRef,
    concrete: ConcreteValue,
}

unsafe fn fbw_reorder_call_kw_args(
    r_args: &[OpRef],
    arg_concretes: &[ConcreteValue],
    w_code: *const (),
    nparams: usize,
    receiver: Option<(OpRef, ConcreteValue)>,
) -> Option<(Vec<OpRef>, Vec<ConcreteValue>, Vec<VarkwExtra>)> {
    if r_args.len() < 3 || arg_concretes.len() < 3 {
        return None;
    }
    let ConcreteValue::Ref(kwnames) = arg_concretes[2] else {
        return None;
    };
    if kwnames.is_null() || !unsafe { pyre_object::is_tuple(kwnames) } {
        return None;
    }
    let args = &r_args[3..];
    let arg_conc = &arg_concretes[3..];
    let nargs = args.len();
    if arg_conc.len() != nargs {
        return None;
    }
    let nkw = unsafe { pyre_object::w_tuple_len(kwnames) };
    // A `**kwargs` callee takes every keyword that names no parameter into its
    // mapping (`_match_signature`), so only the positional arguments have to
    // fit its parameters.
    let has_varkw = fbw_callee_varkw_slot(w_code).is_some();
    // No positional parameter may be filled more than once, and the call may
    // not pass more than the callee takes — `*args` / keyword-only slots are
    // ruled out separately by `fbw_callee_scope_is_positional_only`.  A
    // parameter left unbound is allowed through as a hole; the caller fills it
    // from `defs_w` or declines when it has no default.
    let receiver_count = usize::from(receiver.is_some());
    if nkw > nargs {
        return None;
    }
    if has_varkw {
        if nargs - nkw + receiver_count > nparams {
            return None;
        }
    } else if nparams == 0 || nargs + receiver_count > nparams {
        return None;
    }
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    if !fbw_callee_scope_is_positional_only(w_code) && !fbw_callee_scope_is_varkw_only(w_code) {
        return None;
    }
    let varnames = unsafe { &(*raw).varnames };
    if varnames.len() < nparams {
        return None;
    }
    let n_pos = nargs - nkw;
    let mut extras = Vec::new();
    let mut slot_args: Vec<Option<OpRef>> = vec![None; nparams];
    let mut slot_conc: Vec<Option<ConcreteValue>> = vec![None; nparams];
    if let Some((receiver_arg, receiver_concrete)) = receiver {
        slot_args[0] = Some(receiver_arg);
        slot_conc[0] = Some(receiver_concrete);
    }
    for k in 0..n_pos {
        let pi = receiver_count + k;
        slot_args[pi] = Some(args[k]);
        slot_conc[pi] = Some(arg_conc[k]);
    }
    for j in 0..nkw {
        let name_obj = unsafe { pyre_object::w_tuple_getitem(kwnames, j as i64) }?;
        if !unsafe { pyre_object::is_str(name_obj) } {
            return None;
        }
        let name = unsafe { pyre_object::w_str_get_wtf8(name_obj) }
            .as_str()
            .ok()?;
        let posonly = unsafe { (*raw).posonlyarg_count } as usize;
        // A keyword naming no parameter — or, under PEP 570, naming a
        // positional-only one — goes to the `**kwargs` mapping when the callee
        // has one (`_match_signature`); without it the call is a TypeError the
        // interpreter reports.
        let pi = varnames[..nparams]
            .iter()
            .position(|v| v.as_str() == name)
            .filter(|&pi| pi >= posonly);
        let Some(pi) = pi else {
            // A name that binds no parameter belongs in the callee's
            // `**kwargs` mapping (`_match_signature`). Building that dict
            // here would trace the binding by hand. Leave the call residual.
            return None;
        };
        // A keyword may only bind a parameter past the positional fill, and each
        // parameter at most once (else Python raises "multiple values for
        // argument").  A name in the positional-only range is not bindable by
        // keyword at all — `def f(x, /)` called as `f(x=1)` is a TypeError, so
        // binding slot 0 here would inline a call the interpreter rejects.
        if pi < receiver_count + n_pos || slot_args[pi].is_some() {
            return None;
        }
        slot_args[pi] = Some(args[n_pos + j]);
        slot_conc[pi] = Some(arg_conc[n_pos + j]);
    }
    let mut out_args = Vec::with_capacity(nparams);
    let mut out_conc = Vec::with_capacity(nparams);
    for k in 0..nparams {
        out_args.push(slot_args[k].unwrap_or(OpRef::NONE));
        out_conc.push(slot_conc[k].unwrap_or(ConcreteValue::Null));
    }
    Some((out_args, out_conc, extras))
}

/// Permute a builtin CALL_KW into the gateway Signature's parameter
/// order.  Keyword VALUES stay the traced boxes; names come from the
/// constant `kwnames` tuple.  Missing slots stay `PY_NULL` for the
/// wrapper to default, never a ConstPtr baked from a recording-time
/// object.
///
/// `r_args` layout is `[callable, self_or_null, kwnames, arg0..argN-1]`.
fn builtin_call_kw_bind_signature(
    r_args: &[OpRef],
    arg_concretes: &[ConcreteValue],
    sig: &pyre_interpreter::gateway::Signature,
    receiver: Option<(OpRef, ConcreteValue)>,
) -> Option<(Vec<OpRef>, Vec<ConcreteValue>)> {
    // The permutation below is keyed on the recording-time names, which is
    // only sound when the `kwnames` box is a constant: then no guard is
    // needed for the trace to replay the same binding.
    if r_args.len() < 3 || arg_concretes.len() < 3 || !r_args[2].is_constant() {
        return None;
    }
    let ConcreteValue::Ref(kwnames) = arg_concretes[2] else {
        return None;
    };
    if kwnames.is_null() || !unsafe { pyre_object::is_tuple(kwnames) } {
        return None;
    }
    let args = &r_args[3..];
    let arg_conc = &arg_concretes[3..];
    let nargs = args.len();
    if arg_conc.len() != nargs {
        return None;
    }
    let nkw = unsafe { pyre_object::w_tuple_len(kwnames) };
    if nkw > nargs {
        return None;
    }
    let nparams = sig.argnames.len();
    // `bind_kwargs_to_signature` appends packed `*args` / `**kwargs` slots
    // after the named ones; this permutation builds only the named slots.
    if nparams == 0 || sig.varargname.is_some() || sig.kwargname.is_some() {
        return None;
    }
    let n_pos_params = sig.num_argnames();
    let receiver_count = usize::from(receiver.is_some());
    let n_pos = nargs - nkw;
    if n_pos + receiver_count > n_pos_params {
        return None;
    }
    let mut slot_args: Vec<Option<OpRef>> = vec![None; nparams];
    let mut slot_conc: Vec<Option<ConcreteValue>> = vec![None; nparams];
    if let Some((receiver_arg, receiver_concrete)) = receiver {
        slot_args[0] = Some(receiver_arg);
        slot_conc[0] = Some(receiver_concrete);
    }
    for k in 0..n_pos {
        let pi = receiver_count + k;
        slot_args[pi] = Some(args[k]);
        slot_conc[pi] = Some(arg_conc[k]);
    }
    for j in 0..nkw {
        let name_obj = unsafe { pyre_object::w_tuple_getitem(kwnames, j as i64) }?;
        if !unsafe { pyre_object::is_str(name_obj) } {
            return None;
        }
        let name = unsafe { pyre_object::w_str_get_wtf8(name_obj) }
            .as_str()
            .ok()?;
        let pi = sig.find_argname(name);
        if pi < 0 {
            return None;
        }
        let pi = pi as usize;
        if pi < sig.posonlyargcount || slot_args[pi].is_some() {
            return None;
        }
        slot_args[pi] = Some(args[n_pos + j]);
        slot_conc[pi] = Some(arg_conc[j + n_pos]);
    }
    let mut out_args = Vec::with_capacity(nparams);
    let mut out_conc = Vec::with_capacity(nparams);
    for k in 0..nparams {
        match (slot_args[k], slot_conc[k]) {
            (Some(op), Some(conc)) => {
                out_args.push(op);
                out_conc.push(conc);
            }
            _ => {
                out_args.push(OpRef::NONE);
                out_conc.push(ConcreteValue::Null);
            }
        }
    }
    Some((out_args, out_conc))
}

/// `PYRE_FBW_INLINE_DIAG`: whether a specialised pair's `value0` / `value1`
/// were still in the heap cache at the star-call. A miss leaves the call
/// residual; the line is the only signal, because the unpack itself returns
/// `None` without a decline reason.
fn diag_spec_tuple_cache(kind: &str, hit: bool) {
    if fbw_inline_diag_enabled() {
        eprintln!(
            "[inline-spec-tuple] {kind} {}",
            if hit { "hit" } else { "miss" }
        );
    }
}

/// Read a trace-local `W_SpecialisedTupleObject_ii` / `_oo` into the two
/// positional boxes `w_tuple_getitem` would hand `call_function_ex`.
///
/// A traced `setfield_gc` of `value0` / `value1` — `w_specialised_tuple_ii_new`
/// for an `ii` pair, `emit_specialised_tuple_oo_inline` for an `oo` pair —
/// leaves those slots in the heap cache. `specialisedtupleobject.py
/// tolist` / `getitem` box an `ii` slot through `wraps` (`w_int_new`) and
/// return an `oo` slot unchanged. `ObjSpace.fixedview` takes that `tolist`
/// for a tuple whose iterator is the builtin one. A pair this trace did not
/// just build has no cached fields and declines. `Cls_ff` has no producer
/// (`w_tuple_new` keeps a float pair as `Cls_oo`), so it is not a third arm.
fn fbw_unpack_specialised_tuple_pair<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    starargs_obj: pyre_object::PyObjectRef,
    starargs: OpRef,
) -> Option<(Vec<OpRef>, Vec<ConcreteValue>)> {
    if unsafe { pyre_object::specialisedtupleobject::is_specialised_tuple_ii(starargs_obj) } {
        let Some(raw0) = ctx.trace_ctx.heapcache_getfield_cached(
            starargs,
            crate::descr::specialised_tuple_ii_value0_descr().index(),
        ) else {
            diag_spec_tuple_cache("ii", false);
            return None;
        };
        let Some(raw1) = ctx.trace_ctx.heapcache_getfield_cached(
            starargs,
            crate::descr::specialised_tuple_ii_value1_descr().index(),
        ) else {
            diag_spec_tuple_cache("ii", false);
            return None;
        };
        diag_spec_tuple_cache("ii", true);
        // Copy both payloads before `w_int_new`. That constructor can collect,
        // and the concrete pair from `malloc_typed_managed` may sit in the
        // nursery, so a later read through `starargs_obj` would see a moved
        // address. The raw field OpRefs stay the machine-code inputs.
        let value0 = unsafe {
            pyre_object::specialisedtupleobject::w_specialised_tuple_ii_getvalue(starargs_obj, 0)
        };
        let value1 = unsafe {
            pyre_object::specialisedtupleobject::w_specialised_tuple_ii_getvalue(starargs_obj, 1)
        };
        let mut args = Vec::with_capacity(2);
        let mut concretes = Vec::with_capacity(2);
        for (raw, value) in [(raw0, value0), (raw1, value1)] {
            // `w_tuple_getitem` boxes the slot with `w_int_new`. The concrete
            // has to be a heap pointer: a small value comes back tagged, and
            // `box_int_concrete` re-homes that onto `w_int_new_unique`.
            // The previous box lives only in `concretes`. Pin it across this
            // allocation and re-read the forwarded slot before the next one.
            let concrete = with_arg_refs_pinned(&mut concretes, |_| {
                let boxed_ptr = pyre_object::intobject::w_int_new(value);
                box_int_concrete(value, boxed_ptr as i64)
            });
            let boxed = walker_box_int(ctx, op_pc, raw, value).ok()?;
            ctx.trace_ctx.set_opref_concrete(boxed, concrete);
            let majit_ir::Value::Ref(gcref) = concrete else {
                return None;
            };
            args.push(boxed);
            concretes.push(ConcreteValue::Ref(
                gcref.as_usize() as pyre_object::PyObjectRef
            ));
        }
        return Some((args, concretes));
    }
    if unsafe { pyre_object::specialisedtupleobject::is_specialised_tuple_oo(starargs_obj) } {
        let Some(elem0) = ctx.trace_ctx.heapcache_getfield_cached(
            starargs,
            crate::descr::specialised_tuple_oo_value0_descr().index(),
        ) else {
            diag_spec_tuple_cache("oo", false);
            return None;
        };
        let Some(elem1) = ctx.trace_ctx.heapcache_getfield_cached(
            starargs,
            crate::descr::specialised_tuple_oo_value1_descr().index(),
        ) else {
            diag_spec_tuple_cache("oo", false);
            return None;
        };
        diag_spec_tuple_cache("oo", true);
        let mut args = Vec::with_capacity(2);
        let mut concretes = Vec::with_capacity(2);
        for elem in [elem0, elem1] {
            let concrete = walker_concrete_ref_object(ctx, elem)?;
            args.push(elem);
            concretes.push(ConcreteValue::Ref(concrete));
        }
        return Some((args, concretes));
    }
    None
}

/// A `f(*args)` star tuple whose elements the resolved half reads off the
/// tuple, once every non-emitting eligibility check has passed.
///
/// `Arguments._combine_starargs_wrapped` takes the positional arguments from
/// `space.fixedview(w_stararg)`, which for an exact tuple is `w_obj.tolist()`:
/// the `wrappeditems` elements of a `W_TupleObject`, or the `wraps[i]` of a
/// specialised pair.  The emit re-proves the layout with a class guard and,
/// for the array-backed tuple, pins its length to the callee's arity.
struct StarArgsInline {
    starargs_op: OpRef,
    repr: DefaultsRepr,
    /// The elements at resolve time, in parameter order.
    values: Vec<pyre_object::PyObjectRef>,
}

/// The `**` mapping of a `CALL_FUNCTION_EX`, bound to the callee's parameters
/// by name once every non-emitting eligibility check has passed.
///
/// `Arguments._combine_starstarargs_wrapped` takes the keywords of an exact
/// dict through `view_as_kwargs`, and `_match_keywords` binds each one to the
/// parameter of that name.  The emit re-proves the dict's layout and strategy,
/// pins its length to the number of keywords bound here, and reads each bound
/// parameter with a lookup that must hit: with the length pinned, those hits
/// leave no keyword for `_match_keywords` to reject.
struct StarKwargsInline {
    dict_op: OpRef,
    /// `(parameter index, interned parameter name, value at resolve time)`.
    bound: Vec<(usize, pyre_object::PyObjectRef, pyre_object::PyObjectRef)>,
}

/// Unpack a `bh_call_function_ex_fn(callable, self_or_null, starargs,
/// kwargs_or_null)` call into the argument boxes the inline path seeds from,
/// or `None` to leave the call a residual.
///
/// When the star tuple is virtual at the call — the `args = (...)` /
/// `f(*args)` pair the walker just recorded, whose `wrappeditems` block and
/// per-index stores are still cached (`try_walker_specialize_newtuple_object`),
/// or whose `value0` / `value1` fields are cached on a specialised pair
/// (`fbw_unpack_specialised_tuple_pair`) — the elements come out of the heap
/// cache and the tuple keeps no consumer. Any other exact tuple is read the
/// way `space.fixedview` reads it; those reads are recorded by the resolved
/// half, so the argument boxes returned here are the tuple itself as a
/// placeholder and the [`StarArgsInline`] names what replaces them.
///
/// Without a `**` mapping the star tuple has to fill every parameter. With
/// one, it fills a prefix and the mapping binds the rest by name
/// ([`fbw_bind_star_kwargs`]); a parameter neither fills is left as a hole for
/// the caller to fill from `defs_w`.
fn fbw_unpack_call_function_ex_args<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    arg_concretes: &[ConcreteValue],
    w_code: *const (),
    nparams: usize,
) -> Option<(
    Vec<OpRef>,
    Vec<ConcreteValue>,
    Option<StarArgsInline>,
    Option<StarKwargsInline>,
)> {
    if r_args.len() < 4 || arg_concretes.len() < 4 || nparams == 0 {
        return None;
    }
    // `kwargs_or_null` is absent only when the concrete is the checked
    // `PY_NULL` sentinel. `ConcreteValue::Null` means the box is untracked,
    // not that the runtime pointer is that sentinel. An opaque mapping
    // (`f(*(1,), **mapping)`) must stay residual: treating it as no mapping
    // would run the call as `f(1)` and skip duplicate-key and keyword binding.
    let kwargs = match arg_concretes[3] {
        ConcreteValue::Ref(kwargs) if kwargs.is_null() || kwargs == pyre_object::PY_NULL => None,
        ConcreteValue::Ref(kwargs) => Some(kwargs),
        ConcreteValue::Null => match ctx.trace_ctx.box_value(r_args[3]) {
            Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
                let obj = r.as_usize() as pyre_object::PyObjectRef;
                if obj.is_null() || obj == pyre_object::PY_NULL {
                    None
                } else {
                    Some(obj)
                }
            }
            _ => return None,
        },
        _ => return None,
    };
    let arity_fits = |npos: usize| {
        if kwargs.is_some() {
            npos <= nparams
        } else {
            npos == nparams
        }
    };
    // `wrappeditems` resolves to a structural field index (offset / size /
    // type), so an object whose slot 0 happens to match — a list's backing
    // store, say — would hit the same cache entry and be unpacked as if its
    // slots were tuple element refs.  `f(*some_list)` is ordinary Python, so
    // pin the concrete to a real tuple the way the `kwnames` path does before
    // reading the field.
    // `fixedview` (`objspace.py`) reads storage only when `_uses_tuple_iter`
    // holds. A tuple subclass that overrides `__iter__` does not, so only an
    // exact tuple (the specialised arity-2 variants included) is unpacked
    // here. Anything else stays a residual and runs its iterator.
    let ConcreteValue::Ref(starargs_obj) = arg_concretes[2] else {
        return None;
    };
    if starargs_obj.is_null() || unsafe { !pyre_object::is_exact_tuple(starargs_obj) } {
        return None;
    }
    let starargs = r_args[2];
    // `fixedview` on a specialised pair is `tolist` / `getitem`, not
    // `wrappeditems`. Only an arity-2 callee with no `**` mapping matches
    // that layout: the two fields already fill every parameter. A mapping
    // still has to run `fbw_bind_star_kwargs`, which reports a duplicate or
    // unexpected key.
    if kwargs.is_none()
        && nparams == 2
        && let Some(unpacked) =
            fbw_unpack_specialised_tuple_pair(ctx, op_pc, starargs_obj, starargs)
    {
        return Some((unpacked.0, unpacked.1, None, None));
    }
    let items_descr = crate::descr::tuple_wrappeditems_descr();
    // Lives across the uncached `w_tuple_getitem` loop and the `**` bind
    // below it. `Cls_ii` / `Cls_ff` box on each read, and the bind can
    // allocate while those boxes are still only in a `Vec`.
    let extract_roots = pyre_object::gc_roots::push_roots();
    // Publish the tuple, the mapping, and the code object together.
    // Sequential `pin_root` normalizes after the first word, and a collection
    // there relocates the next pointer before it is published. `w_code` is
    // read again after `w_tuple_getitem`, which boxes and can collect.
    let code_obj = w_code as pyre_object::PyObjectRef;
    let (tuple_slot, kwargs_slot, code_slot) = if let Some(obj) = kwargs {
        let base = extract_roots.pin_roots(&[starargs_obj, obj, code_obj]);
        (base, Some(base + 1), base + 2)
    } else {
        let base = extract_roots.pin_roots(&[starargs_obj, code_obj]);
        (base, None, base + 1)
    };
    let mut extracted_base: Option<usize> = None;
    let (mut args, mut concretes, mut star) = if let Some(block) = ctx
        .trace_ctx
        .heapcache_getfield_cached(starargs, items_descr.index())
    {
        // The cached length pins the arity: a count the callee cannot take
        // is a runtime TypeError the inline path does not model.
        let len_op = ctx.trace_ctx.heap_cache().arraylen(block)?;
        let npos = match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 0 && arity_fits(n as usize) => n as usize,
            _ => return None,
        };
        let array_descr_index = crate::state::pyobject_gcarray_descr().index();
        let mut args = Vec::with_capacity(nparams);
        let mut concretes = Vec::with_capacity(nparams);
        for index in 0..npos {
            let elem = ctx.trace_ctx.heapcache_getarrayitem(
                block,
                OpRef::ConstInt(index as i64),
                array_descr_index,
            )?;
            // Take each concrete from the element box itself, not from the
            // tuple, so the seeded shadow is the one the symbolic argument
            // carries.
            let concrete = walker_concrete_ref_object(ctx, elem)?;
            args.push(elem);
            concretes.push(ConcreteValue::Ref(concrete));
        }
        (args, concretes, None)
    } else {
        // `_uses_tuple_iter` holds for the exact tuple classes alone, and
        // those are the layouts `defaults_repr_of` has an element read for.
        let starargs_live = extract_roots.get(tuple_slot);
        let repr = defaults_repr_of(unsafe { (*starargs_live).ob_type })?;
        let npos = unsafe { pyre_object::w_tuple_len(starargs_live) };
        if !arity_fits(npos) {
            return None;
        }
        let values_base = pyre_object::gc_roots::shadow_stack_len();
        extracted_base = Some(values_base);
        let mut values = Vec::with_capacity(npos);
        for index in 0..npos {
            let tuple = extract_roots.get(tuple_slot);
            let item = unsafe { pyre_object::w_tuple_getitem(tuple, index as i64) }?;
            let item = extract_roots.pin_root(item);
            values.push(item);
        }
        let concretes = values.iter().map(|&v| ConcreteValue::Ref(v)).collect();
        (
            vec![starargs; npos],
            concretes,
            Some(StarArgsInline {
                starargs_op: starargs,
                repr,
                values,
            }),
        )
    };
    // Cached elements are already boxes. Pin them before the bind: interning
    // a parameter name can collect, and these are not on `extract_roots` yet.
    let cached_pin = if extracted_base.is_none() {
        let live: Vec<pyre_object::PyObjectRef> = concretes
            .iter()
            .filter_map(|concrete| match concrete {
                ConcreteValue::Ref(obj) if !obj.is_null() => Some(*obj),
                _ => None,
            })
            .collect();
        let base = extract_roots.pin_roots(&live);
        Some((base, live.len()))
    } else {
        None
    };
    let Some(kwargs) = kwargs else {
        if let Some(base) = extracted_base {
            rehome_extracted_star(&extract_roots, base, &mut concretes, star.as_mut());
        }
        rehome_cached_star(&extract_roots, cached_pin, &mut concretes);
        return Some((args, concretes, star, None));
    };
    let kwargs_live = kwargs_slot
        .map(|slot| extract_roots.get(slot))
        .unwrap_or(kwargs);
    let w_code = extract_roots.get(code_slot) as *const ();
    let mut star_kwargs =
        unsafe { fbw_bind_star_kwargs(r_args[3], kwargs_live, w_code, args.len(), nparams) }?;
    let bound_live: Vec<pyre_object::PyObjectRef> =
        star_kwargs.bound.iter().map(|entry| entry.2).collect();
    let bound_base = extract_roots.pin_roots(&bound_live);
    for (offset, entry) in star_kwargs.bound.iter_mut().enumerate() {
        entry.2 = extract_roots.get(bound_base + offset);
    }
    if let Some(base) = extracted_base {
        rehome_extracted_star(&extract_roots, base, &mut concretes, star.as_mut());
    }
    rehome_cached_star(&extract_roots, cached_pin, &mut concretes);
    args.resize(nparams, OpRef::NONE);
    concretes.resize(nparams, ConcreteValue::Null);
    for &(index, _, value) in &star_kwargs.bound {
        // Placeholder until the resolved half records the lookup.
        args[index] = star_kwargs.dict_op;
        concretes[index] = ConcreteValue::Ref(value);
    }
    Some((args, concretes, star, Some(star_kwargs)))
}

fn rehome_cached_star(
    roots: &pyre_object::gc_roots::RootScope,
    pin: Option<(usize, usize)>,
    concretes: &mut [ConcreteValue],
) {
    let Some((base, len)) = pin else {
        return;
    };
    if len == 0 {
        return;
    }
    let mut offset = 0;
    for concrete in concretes.iter_mut() {
        if let ConcreteValue::Ref(obj) = concrete
            && !obj.is_null()
        {
            *obj = roots.get(base + offset);
            offset += 1;
        }
    }
}

fn rehome_extracted_star(
    roots: &pyre_object::gc_roots::RootScope,
    base: usize,
    concretes: &mut [ConcreteValue],
    star: Option<&mut StarArgsInline>,
) {
    let Some(star) = star else {
        return;
    };
    for (offset, value) in star.values.iter_mut().enumerate() {
        *value = roots.get(base + offset);
        if let Some(ConcreteValue::Ref(obj)) = concretes.get_mut(offset) {
            *obj = *value;
        }
    }
}

/// Bind the keywords of the `**` mapping `kwargs` to the parameters of
/// `w_code` past the `npos` the star tuple filled, the way `_match_keywords`
/// (`argument.py`) binds the `view_as_kwargs` keys.
///
/// Only an exact dict under the unicode strategy is taken, and only by a
/// callee whose locals are its positional parameters alone.  Every key has to
/// name a parameter it may bind — past the positional fill and past the
/// positional-only range — or the call is a TypeError the interpreter
/// reports, so the inline declines.
///
/// # Safety
/// `kwargs` must be a live object.
unsafe fn fbw_bind_star_kwargs(
    dict_op: OpRef,
    kwargs: pyre_object::PyObjectRef,
    w_code: *const (),
    npos: usize,
    nparams: usize,
) -> Option<StarKwargsInline> {
    let exact_dict = unsafe {
        (*kwargs).ob_type == &pyre_object::pyobject::DICT_TYPE as *const _
            && (*kwargs).w_class == pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE)
    };
    if !exact_dict
        || !std::ptr::eq(
            unsafe { (*(kwargs as *const pyre_object::W_DictObject)).dstrategy },
            &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF,
        )
    {
        return None;
    }
    if !fbw_callee_scope_is_positional_only(w_code) {
        return None;
    }
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    let varnames = unsafe { &(*raw).varnames };
    if varnames.len() < nparams {
        return None;
    }
    let posonly = unsafe { (*raw).posonlyarg_count } as usize;
    // Copy the names out before any interning. `get_interned_wtf8` can
    // collect, and `varnames` is a borrow of the code object.
    let names: Vec<(usize, String)> = varnames[..nparams]
        .iter()
        .enumerate()
        .skip(npos.max(posonly))
        .map(|(index, name)| (index, name.as_str().to_string()))
        .collect();
    let roots = pyre_object::gc_roots::push_roots();
    let kwargs_slot = roots.base();
    let _ = roots.pin_root(kwargs);
    let mut bound = Vec::new();
    let mut value_slots = Vec::new();
    for (index, name) in names {
        // The name is baked into the lookup, so it has to be the canonical
        // object a collection cannot move.  Parameter names are interned when
        // the code object is built; one that is not declines rather than
        // interning a value at trace time.
        let name_obj = pyre_object::get_interned_wtf8(Wtf8::new(name.as_str()))?;
        if majit_gc::can_move(majit_ir::GcRef(name_obj as usize)) {
            return None;
        }
        let kwargs = roots.get(kwargs_slot);
        if let Some(value) = unsafe {
            pyre_object::dictmultiobject::w_dict_lookup_or_null_unicode_strategy(kwargs, name_obj)
        } {
            value_slots.push(pyre_object::gc_roots::shadow_stack_len());
            let _ = roots.pin_root(value);
            bound.push((index, name_obj, value));
        }
    }
    for (entry, slot) in bound.iter_mut().zip(&value_slots) {
        entry.2 = roots.get(*slot);
    }
    // A key no parameter took is either unknown or names a parameter the
    // positional fill already bound.
    let kwargs = roots.get(kwargs_slot);
    if bound.len() != unsafe { pyre_object::dictmultiobject::w_dict_len(kwargs) } {
        return None;
    }
    Some(StarKwargsInline { dict_op, bound })
}

/// Body-shape verdicts for a callee's per-fn JitCode, decided once and kept on
/// the payload ([`crate::pyjitcode::InlineBodyFacts`]).
///
/// Every predicate behind this decodes the whole callee body, and the inline
/// recipes ask them per call site, on every trace that reaches the call — so
/// answering them here is the difference between one scan per callee and one
/// per call site. `None` when the code has no installed jitcode body or descr
/// pool, which is the same condition the callers already decline on.
/// Whether `w_code` commits nothing a discarded sub-walk would double.
///
/// A route whose entry opcode is not a call boundary — `COMPARE_OP`,
/// `BINARY_OP` — has no CALL to convert past once the body has run.
/// [`BinopRewindInlineGuard`] refuses the first commit before it runs, so a
/// discarded body rewinds cleanly.  Admission is what has to carry that
/// decision.
///
/// `DeferredCall` is not enough here.  A residual the lever cannot inline
/// is executed and recorded, or converts in place through
/// `convert_and_run_from_pyjitpl` (`blackhole.py`) when the callee already
/// has a seeded `PyFrame`.  A `BINARY_OP` / `COMPARE_OP` entry is a
/// separate rewind region: [`BinopRewindInlineGuard`] refuses the first
/// commit so a discarded `NotImplemented` descent does not keep its prefix.
/// Whether a dunder body at a `BINARY_OP` / `COMPARE_OP` entry is a shape
/// worth attempting on the strength of the entry's rewind.
///
/// This is NOT what makes the rewind sound.  [`BinopRewindInlineGuard`] is:
/// it refuses the first commit before it runs, which is the only form the
/// question has an answer in -- whether a body commits is a property of the
/// path it walks, and a body that delegates through `+` is statically
/// indistinguishable from `return self.v + o.v` over ints.
///
/// What the nested-call refusal covers, and the one nested shape it lets
/// through.  A body making a nested Python call and then returning inlines
/// into a peeled loop whose Phase 2 can still die on `phase2 snapshot remap
/// cache miss` (`unroll.rs`), measured on
/// `synth/inline_freevar_after_mayforce` with a `Fraction.__gt__`-shaped body
/// -- `return a._richcmp(b, operator.gt)`.  That returning shape is what the
/// refusal is for.
///
/// A straight-line, handler-free body that raises is different: its nested
/// call constructs the exception and `raise/r` finishes the inlined MIFrame;
/// no callee continuation enters the peeled loop.  This is PyPy's ordinary
/// `perform_call` -> `finishframe_exception` path.  Admit it so each dunder has
/// its own live frame and exception state instead of compiling a second root
/// portal.  `BinopRewindInlineGuard` still refuses a nested residual before it
/// commits, and `has_exception_table == false` excludes a caught raise whose
/// body could continue to a return.
pub(crate) fn dunder_body_facts_admissible_on_rewind(
    facts: crate::pyjitcode::InlineBodyFacts,
) -> bool {
    !facts.exc_override_has_nested_call
        || (facts.contains_raise && facts.exc_override_straight_line && !facts.has_exception_table)
}

/// The filter this carried unconditionally was that nested-call refusal, whose
/// stated cost was a trace dying in the Phase 2 unroll on `phase2 snapshot
/// remap cache miss`.  That no longer reproduces:
/// `synth/inline_freevar_after_mayforce` passes with the nested body admitted,
/// and all 537 bench scripts answer identically with and without it.  Since
/// the filter also refused every value type the pure-Python stdlib is made of
/// -- `date.__add__` returns `date(...)`, `Decimal.__add__` returns
/// `_dec_from_triple(...)` -- it is off, and `PYRE_BINOP_NO_NESTED_INLINE`
/// puts it back so one binary carries both arms.
fn dunder_body_admissible_on_rewind(w_code: *const ()) -> bool {
    let Some(facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return false;
    };
    !binop_nested_filter_enabled() || dunder_body_facts_admissible_on_rewind(facts)
}

/// `PYRE_NO_BINOP_DEFAULTED_PARAMS`: go back to refusing a `BINARY_OP` /
/// `COMPARE_OP` dunder whose signature has more than the two parameters the
/// operands bind, instead of leaving the tail to the resolved descent's
/// `__defaults__` seeding.
///
/// On by default; the variable only turns it off, so a bisection can name this
/// change in one command and one binary carries both arms.
fn binop_defaulted_params_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_NO_BINOP_DEFAULTED_PARAMS").is_none())
}

/// `PYRE_BINOP_NO_NESTED_INLINE`: refuse a `BINARY_OP` / `COMPARE_OP` dunder
/// whose body makes a nested Python call, in the shape
/// [`dunder_body_facts_admissible_on_rewind`] states, as
/// [`dunder_body_admissible_on_rewind`] did before.
fn binop_nested_filter_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_BINOP_NO_NESTED_INLINE").is_some())
}

/// Discard a declined inline sub-walk: the operations it recorded, the
/// snapshots those operations named, and the heap cache they populated.
///
/// The eight call sites all spelled the operations and the cache out as the
/// same two statements.
///
/// [`majit_metainterp::history::TraceCtx::cut_trace`] leaves the snapshot side
/// table alone, and a guard the speculative walk attached before it declined
/// then keeps naming positions the cut has since handed to other operations —
/// the shape the unroll's Phase 2 remap reports as a `phase2 snapshot remap
/// cache miss`.  So the cut here takes both.
///
/// Upstream restores neither: `Trace.cut_point` returns the two snapshot
/// lengths as the last elements of its five-tuple and `Trace.cut_at` restores
/// only `_pos`, `_count` and `_index` from it, leaving `_snapshot_data` and
/// `_snapshot_array_data` as they stand -- the same two elements
/// `cut_trace_from` destructures and never reads.  It is not exposed by that
/// because its snapshots live inline in the trace byte stream, where anything
/// past the restored `_pos` is unreachable and the next append overwrites it;
/// pyre owns them in a `Vec<Snapshot>` beside the stream, where a stale entry
/// keeps its index.  Truncating is what that representation costs, and it is
/// already what every other speculative rewind in the walker does --
/// `cut_trace_with_snapshots` has eighteen further call sites across this file
/// and `specialize.rs`, the `BINARY_OP` entry's own rewind among them.  The
/// migration to the byte-stream form `TraceRecordBuffer` already carries is
/// recorded on `TraceCtx::snapshots`.
///
/// `cut_trace`'s own note records that truncating regressed a bench.  That is
/// about `cut_trace` truncating for all of its callers, which this does not
/// change: the seven outside this helper still take the operations alone.
fn cut_declined_subwalk<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pre_fold_pos: majit_metainterp::recorder::TracePosition,
) {
    ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
    ctx.trace_ctx.heap_cache_mut().reset();
}

/// The point a builtin descent cuts back to, with the effect state it was
/// entered with.
#[derive(Clone, Copy)]
struct SubWalkRewind {
    pre_fold_pos: majit_metainterp::recorder::TracePosition,
    effects_before: usize,
    unjournaled_before: bool,
}

impl SubWalkRewind {
    /// Whether the walk applied nothing since the rewind point, so the
    /// ordinary residual call applies every effect exactly once.  The
    /// gateway's rollback arm in [`try_walker_inline_builtin_call`] says why
    /// an unjournaled effect from before the rewind point also refuses.
    fn nothing_applied(&self) -> bool {
        fbw_executed_effect_count() == self.effects_before
            && !self.unjournaled_before
            && !fbw_has_unjournaled_effect()
    }
}

fn sub_jitcode_body_facts_for_code(code: *const ()) -> Option<crate::pyjitcode::InlineBodyFacts> {
    let body = crate::state::sub_jitcode_body_for_code(code)?;
    let (descr_refs, _, _) = crate::state::sub_jitcode_descr_pool_for_code(code)?;
    // Resolved after the two lookups above so that a code object without an
    // installed body never caches a verdict computed from an absent one.
    let pjc = crate::state::pyjitcode_for_code(code)?;
    Some(
        *pjc.inline_body_facts
            .get_or_init(|| crate::pyjitcode::InlineBodyFacts {
                contains_raise: callee_body_contains_raise(body.code),
                has_abort_permanent: crate::jitcode_runtime::decoded_ops(body.code)
                    .any(|op| op.opname == "abort_permanent"),
                exc_override_straight_line: exception_string_override_straight_line(body.code),
                exc_override_sample_safe: exception_string_override_sample_safe(
                    body.code,
                    descr_refs,
                    body.num_regs_i,
                    body.constants_i,
                ),
                index_sample_safe: index_inline_sample_safe(
                    body.code,
                    descr_refs,
                    body.num_regs_i,
                    body.constants_i,
                ),
                exc_override_has_nested_call: exception_string_override_has_nested_call(
                    body.code, descr_refs,
                ),
                owns_loop_header: callee_body_owns_loop_header(body.code),
                has_exception_table: unsafe {
                    let raw = pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject;
                    !raw.is_null() && !(&(*raw).exceptiontable).is_empty()
                },
            }),
    )
}

pub(crate) fn try_walker_inline_user_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    ref_operand_offset: usize,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    runtime_helper: majit_ir::RuntimeHelperKind,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // Authoritative walks only: inline sub-walks lean on FBW multi-frame
    // snapshot plumbing a non-authoritative context does not carry.
    if !ctx.is_authoritative_executor {
        return Ok(None);
    }
    // Only a genuine Python call helper is an inline target: positional
    // `call_fn` / `call_fn_N` (`RuntimeHelperKind::CallFn`) and keyword `call_kw_N`
    // (`RuntimeHelperKind::CallKw`), both tagged by the flatten lowering.  Every
    // container/builtin helper routed here carries a different tag or `None`
    // (`store_subscr_fn` -> StoreSubscr, `normalize_raise_varargs_fn` /
    // `set_current_exception` -> None).  Without this guard `d[f] = v` with a
    // 1-arg function key `f` lowers to `residual_call_r_v(store_subscr_fn, [d,
    // f, v])`, whose ref args pass the function sniff below and are mis-inlined
    // as `f(v)`, skipping the store.  Upstream never inlines a Python call at a
    // residual_call site (inlinable calls get their own inline_call jitcodes);
    // this restores that invariant for the pyre FBW inline-at-residual lever.
    let is_call_kw = runtime_helper == majit_ir::RuntimeHelperKind::CallKw;
    let is_call_function_ex = runtime_helper == majit_ir::RuntimeHelperKind::CallFunctionEx;
    if runtime_helper != majit_ir::RuntimeHelperKind::CallFn && !is_call_kw && !is_call_function_ex
    {
        return Ok(None);
    }
    if fbw_inline_diag_enabled() {
        eprintln!(
            "[inline-entry] pc={} helper={:?} nrefargs={} subwalk={} dst_bank={dst_bank}",
            op.pc,
            runtime_helper,
            r_args.len(),
            ctx.fbw_mode.inline_subwalk,
        );
    }
    // Name every bail between the entry print and callee resolution.  An
    // `[inline-entry]` with no follow-up line otherwise leaves the reason
    // unobservable, which is the whole distance between "this call did not
    // inline" and knowing why.
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }
    if r_args.is_empty() {
        decline!("no ref args");
    }
    let mut arg_concretes = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    if r_args.len() < 2 {
        decline!("fewer than two ref args");
    }
    for i in 0..2 {
        if matches!(arg_concretes.get(i), Some(ConcreteValue::Null)) {
            if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(r_args[i]) {
                let bits = r.as_usize();
                // `cast_int_to_ptr(21)` folds to a non-null `Ref` that is not
                // an object. `is_function` loads the type word from it.
                if r != majit_ir::GcRef::NO_CONCRETE
                    && bits != 0
                    && bits % std::mem::align_of::<usize>() == 0
                {
                    arg_concretes[i] = ConcreteValue::Ref(bits as pyre_object::PyObjectRef);
                }
            }
        }
    }
    let ConcreteValue::Ref(callable) = arg_concretes[0] else {
        decline!("callable slot carries no concrete ref");
    };
    if callable.is_null() || (callable as usize) % std::mem::align_of::<usize>() != 0 {
        decline!("callable is null");
    }
    // The receiver slot is a checked `PY_NULL` sentinel for a plain no-receiver
    // call; its concrete shadow arrives as `Null` (`call_kw`) or `Ref(PY_NULL)`
    // (`call_fn`).  Both mean "no receiver" (`method_form = false`).
    let null_or_self = match arg_concretes[1] {
        ConcreteValue::Ref(r) => r,
        ConcreteValue::Null => pyre_object::PY_NULL,
        _ => decline!("receiver slot carries no concrete ref"),
    };
    let mut method_form = !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL;
    // baseobjspace.py:1254-1259 unwraps `_Method` before the Function
    // valuestack fast path.  CALLs through a stored bound method (notably the
    // module aliases in random.py) arrive as `[Method, PY_NULL, args...]`, so
    // recover its immutable function/receiver fields and feed the same
    // method-form callee shape used by LOAD_METHOD.
    let bound_method = if !method_form && unsafe { pyre_object::is_method(callable) } {
        let function = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable) };
        if function.is_null() || receiver.is_null() {
            return Ok(None);
        }
        method_form = true;
        Some(BoundMethodInline {
            method_op: r_args[0],
            function,
            receiver,
        })
    } else {
        None
    };
    let resolved_callable = bound_method.map_or(callable, |bound| bound.function);
    // `descroperation.py DescrOperation.call_args` has three arms, and
    // the two above are its `Function` and `Method` fast paths.  Anything else
    // resolves `space.lookup(w_obj, '__call__')` and calls that with the object
    // as the receiver.  `__call__` is an ordinary function in the class dict, so
    // the call inlines like a bound method whose receiver is the callable
    // itself — pinned on the callable's class version, which is what
    // `space.lookup` promotes, instead of read off a `Method`'s immutable
    // fields.
    let mut instance_call_pin = None;
    let (mut resolved_callable, mut w_code, nparams, has_closure) = match unsafe {
        resolve_inlinable_callee(resolved_callable)
    } {
        Some((w_code, nparams, has_closure)) => (resolved_callable, w_code, nparams, has_closure),
        None => {
            let Some((method, w_class, version_tag, w_code, nparams, has_closure)) = (!method_form)
                .then(|| unsafe { resolve_instance_dunder_call(callable) })
                .flatten()
            else {
                // The callable's type is the whole answer here:
                // `resolve_inlinable_callee` takes `function` and a
                // MixedModule `BuiltinFunction` that still carries PyCode,
                // so a BuiltinCode builtin or a `method` reads as "not
                // inlinable" for a reason no pc can convey.
                decline!(format_args!(
                    "callee not inlinable (callable type {})",
                    unsafe { pyre_object::type_name_of(callable) }
                ));
            };
            method_form = true;
            instance_call_pin = Some((r_args[0], callable, w_class, version_tag, unsafe {
                inline_attr_cell_guard(w_class, "__call__", method)
            }));
            (method, w_code, nparams, has_closure)
        }
    };
    if fbw_inline_diag_enabled() {
        // Name the callee: a pc alone does not say which function a decline
        // cost, and one trace reaches several.
        let name = unsafe {
            let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject;
            if raw.is_null() {
                String::new()
            } else {
                (*raw).qualname.clone()
            }
        };
        eprintln!(
            "[inline-resolved] pc={} callee={name} nparams={nparams} has_closure={has_closure} method_form={method_form}",
            op.pc,
        );
    }
    let mut star_args = None;
    let mut star_kwargs = None;
    let mut varkw_extra = Vec::new();
    let (callee_args, callee_arg_concretes) = if is_call_kw {
        // A keyword call folds its `kwnames`->parameter permutation at trace
        // time (`fbw_reorder_call_kw_args`) so the reordered param-order args
        // seed the callee exactly like a positional call.  Method form
        // prepends its receiver before that permutation, matching
        // `call_kw`'s `Arguments.prepend` / `funcrun_obj` path.
        let receiver = if let Some(bound) = bound_method {
            // Placeholder until the resolved half reads Method.w_self live.
            Some((bound.method_op, ConcreteValue::Ref(bound.receiver)))
        } else if instance_call_pin.is_some() {
            // `get_and_call_args(w_descr, w_obj, args)` binds the object the
            // lookup ran on, and that object is the CALL's own callable operand.
            Some((r_args[0], ConcreteValue::Ref(callable)))
        } else if method_form {
            Some((r_args[1], arg_concretes[1]))
        } else {
            None
        };
        let Some((args, concretes, extras)) = (unsafe {
            fbw_reorder_call_kw_args(r_args, &arg_concretes, w_code, nparams, receiver)
        }) else {
            return Ok(None);
        };
        varkw_extra = extras;
        (args, concretes)
    } else if is_call_function_ex {
        // `f(*args)` unpacks the star tuple into positional arguments, which is
        // a trace-time reorder whenever the tuple is the one this trace just
        // built: its element boxes are still in the heap cache, so the callee
        // seeds from them and the tuple keeps no consumer.
        //
        // The unpacked arguments come from the star tuple alone, so a
        // method-form call — whose receiver is an implicit leading argument the
        // unpack never sees — stays residual.  The callee-scope decline this
        // seeding also needs is applied once in the resolved half.
        if method_form {
            return Ok(None);
        }
        // `w_tuple_getitem` on a specialised tuple allocates. Pin the callee
        // and its code across that unpack, then reload the forwarded
        // addresses. The unpack also reloads `w_code` before keyword binding.
        let callee_roots = pyre_object::gc_roots::push_roots();
        let callee_base =
            callee_roots.pin_roots(&[resolved_callable, w_code as pyre_object::PyObjectRef]);
        let unpacked = fbw_unpack_call_function_ex_args(
            ctx,
            op.pc,
            r_args,
            &arg_concretes,
            callee_roots.get(callee_base + 1) as *const (),
            nparams,
        );
        resolved_callable = callee_roots.get(callee_base);
        w_code = callee_roots.get(callee_base + 1) as *const ();
        let Some((args, concretes, star, kwargs)) = unpacked else {
            return Ok(None);
        };
        star_args = star;
        star_kwargs = kwargs;
        (args, concretes)
    } else {
        let mut callee_args = Vec::with_capacity(r_args.len().saturating_sub(1));
        let mut callee_arg_concretes = Vec::with_capacity(arg_concretes.len().saturating_sub(1));
        if let Some(bound) = bound_method {
            // Placeholder until the non-emitting eligibility checks finish;
            // the resolved half replaces it with GetfieldGcR(Method.w_self).
            callee_args.push(bound.method_op);
            callee_arg_concretes.push(ConcreteValue::Ref(bound.receiver));
        } else if instance_call_pin.is_some() {
            // `get_and_call_args(w_descr, w_obj, args)` binds the object the
            // lookup ran on, and that object is the CALL's own callable operand.
            callee_args.push(r_args[0]);
            callee_arg_concretes.push(ConcreteValue::Ref(callable));
        } else if method_form {
            callee_args.push(r_args[1]);
            callee_arg_concretes.push(arg_concretes[1]);
        }
        callee_args.extend_from_slice(&r_args[2..]);
        callee_arg_concretes.extend_from_slice(&arg_concretes[2..]);
        (callee_args, callee_arg_concretes)
    };
    try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        resolved_callable,
        r_args[0],
        // The operand this call dispatched on is the callable itself, and for
        // the `__call__` arm that is not the resolved callee: reading
        // `Function.code` off it would be a type-confused load, so the resolved
        // half must see the instance and pin the callee's `code?` marker
        // instead.
        if instance_call_pin.is_some() {
            callable
        } else {
            resolved_callable
        },
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        method_form,
        bound_method,
        w_code,
        nparams,
        has_closure,
        instance_call_pin,
        None,
        true,
        false,
        None,
        None,
        false,
        None,
        None,
        None,
        star_args,
        varkw_extra,
        star_kwargs,
    )
}

/// How many activations the fragment being recorded has minted by inlining a
/// callee body, at the point the walk has reached.
///
/// Record-time only: it costs the compiled code nothing and exists so the one
/// seam that does survive to run time can name the width it stands for. It is
/// owned by the per-attempt [`WalkSession`], matching `MetaInterp.framestack`:
/// a residual may re-enter the tracer on the same OS thread, but that nested
/// walk must start with width zero rather than inherit the outer attempt.
fn open_inline_activations(session: &std::cell::RefCell<WalkSession>) -> u32 {
    session.borrow().open_inline_activations
}

/// Descriptor for the activation seam's load of the recursion-limit word.
///
/// The twin of `eval_breaker_word_descr`: the load is exactly as wide as the
/// word it reads, taking the width from the word itself rather than restating
/// it.  Non-pure like that one, and for the same reason -- a pure load is a
/// candidate for CSE against the value the recording observed, which would
/// bake the limit in force at trace time into the fragment and leave
/// `sys.setrecursionlimit` unable to reach it.
fn recursion_limit_word_descr() -> DescrRef {
    static DESCR: std::sync::OnceLock<DescrRef> = std::sync::OnceLock::new();
    DESCR
        .get_or_init(|| {
            majit_ir::descr::make_array_descr_signed(
                0,
                pyre_interpreter::module::sys::state::RECURSION_LIMIT_WORD_SIZE,
                Type::Int,
                true,
            )
        })
        .clone()
}

/// `pyframe.py execute_frame`'s activation entry, in the form a trace carries.
///
/// Ahead of `ec.enter` stands `execute_frame`'s operation 0, which
/// `insert_ll_stackcheck` (`rpython/translator/transform.py`) puts there.
/// Upstream keeps it out of jitcodes -- `driver.py`'s `pyjitpl_lltype` task
/// depends on `rtype` alone while `stackcheckinsertion_lltype` runs after
/// `backendopt` -- and lets the backend's per-fragment SP probe stand for it.
/// That probe answers the native half only.  The logical half, the test
/// against `sys.getrecursionlimit()`, has `execute_frame`'s entry as its only
/// home, so a fold that mints a callee activation inside compiled code has to
/// run it here or the limit binds nothing for as long as the recursion stays
/// compiled -- the shape `insert_ll_stackcheck` names when it sets
/// `inhibit_tail_call`, a recursion that consumes no stack and so answers no
/// stack probe.
///
/// The counters live on the `ExecutionContext`, which this seam already holds
/// and already reads a field of one line further down, so the charge is the
/// same GETFIELD_GC/SETFIELD_GC pair rather than a residual call into a
/// thread-local.  The limit is read the way the back-edge poll reads the
/// eval-breaker word, a raw load of a published word, so lowering the limit
/// reaches compiled code without invalidating it.
///
/// Claim and charge both land before the guard, and neither is given back on
/// it. That is the state a deopt wants: the guard exit resumes the very frame
/// the claim names, so the interpreter's own entry finds it already accounted,
/// spends nothing, and raises off the depth this charge produced. Keep the
/// conservative `charged >= limit` boundary: `sys.setrecursionlimit` documents
/// PyPy's limit as approximative and lower-level, and its `@jit.dont_look_inside`
/// is load-bearing for that policy.
///
/// Returns the depth to restore and the claim to put back -- the two words the
/// release takes, threaded to it as dataflow rather than respelled there.
fn record_activation_charge<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    ec: OpRef,
    callee_frame: OpRef,
    units: OpRef,
) -> Result<(OpRef, OpRef), DispatchError> {
    let saved_depth = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcI,
        &[ec],
        crate::descr::ec_py_recursion_depth_descr(),
    );
    let charged = ctx
        .trace_ctx
        .record_op(OpCode::IntAdd, &[saved_depth, units]);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, charged],
        crate::descr::ec_py_recursion_depth_descr(),
    );

    let displaced_activation = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcI,
        &[ec],
        crate::descr::ec_accounted_activation_descr(),
    );
    // The claim is an identity token, never dereferenced, so it is banked as
    // an integer rather than as the reference the frame arrives in.
    let claim = ctx
        .trace_ctx
        .record_op(OpCode::CastPtrToInt, &[callee_frame]);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, claim],
        crate::descr::ec_accounted_activation_descr(),
    );

    let limit_addr = ctx
        .trace_ctx
        .const_int(pyre_interpreter::module::sys::state::recursion_limit_addr() as i64);
    let zero = ctx.trace_ctx.const_int(0);
    let limit = ctx.trace_ctx.record_op_with_descr(
        OpCode::RawLoadI,
        &[limit_addr, zero],
        recursion_limit_word_descr(),
    );
    // Strictly greater, because the check this stands in for runs *ahead* of
    // the charge: `stack_check` reads the depth before `enter_recursive_frame`
    // spends the unit, so `depth >= limit` refuses only a frame that would
    // take the depth past the limit and a recursion is allowed to reach the
    // limit exactly.  Charging first and then testing `charged >= limit` would
    // refuse at `limit - 1` what the plain evaluator admits, and the two paths
    // would enforce limits one apart.  `units` covers the inlined levels above
    // this seam, each of which would have run its own ahead-of-charge check, so
    // the last of them is the one this test decides.
    let over_limit = ctx.trace_ctx.record_op(OpCode::UintGt, &[charged, limit]);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[over_limit])?;

    Ok((saved_depth, displaced_activation))
}

/// The `finally` half of the same seam, where the frame chain is given back.
///
/// It restores the depth the activation was entered with rather than
/// subtracting the units it spent, matching `RecursionDepthGuard`: an exit
/// that leaves compiled code between the charge and here -- a guard failure, a
/// raise -- skips this store the same way it skips the `ec.topframeref`
/// restore, and an absolute restore at the next activation boundary is what
/// absorbs the drift.
fn record_activation_release(
    ctx: &mut TraceCtx,
    ec: OpRef,
    saved_depth: OpRef,
    displaced_activation: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, saved_depth],
        crate::descr::ec_py_recursion_depth_descr(),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, displaced_activation],
        crate::descr::ec_accounted_activation_descr(),
    );
}

/// Holds one inlined level open for as long as the walk is inside its body.
///
/// A guard rather than a pair of calls because the region between
/// [`walker_ec_enter`] and [`walker_ec_leave`] has many ways out — a declined
/// sub-walk, a rewind — and a width that leaks would price every later
/// `CALL_ASSEMBLER` in the same trace too high.
struct OpenInlineActivation<'a>(&'a std::cell::RefCell<WalkSession>);

impl<'a> OpenInlineActivation<'a> {
    fn open(session: &'a std::cell::RefCell<WalkSession>) -> Self {
        let mut walk = session.borrow_mut();
        walk.open_inline_activations += 1;
        drop(walk);
        Self(session)
    }
}

impl Drop for OpenInlineActivation<'_> {
    fn drop(&mut self) {
        let mut walk = self.0.borrow_mut();
        walk.open_inline_activations = walk.open_inline_activations.saturating_sub(1);
    }
}

/// `executioncontext.py ExecutionContext.enter`, at an inlined call.
///
/// ```python
/// def enter(self, frame):
///     frame.f_backref = self.topframeref
///     self.topframeref = jit.virtual_ref(frame)
/// ```
///
/// Both halves run.  The recorded ops are what a guard resumes into; the
/// concrete stores are what the callee body observes while this walk records
/// it, because the walk IS the interpreter running — a `sys._getframe()` in
/// the body executes as a residual against the live `ec`, and without the
/// concrete store it would read the CALLER and commit the wrong frame.
///
/// The concrete slot holds the `JitVirtualRef`, not the frame:
/// `executioncontext::force_vref` resolves it for every reader, and a vref
/// built by `virtual_ref_during_tracing` already carries `forced = frame` with
/// `virtual_token = TOKEN_NONE` (`virtualref.py`), so the resolution is
/// exact and cannot fail.  Storing the vref rather than the frame is what lets
/// the optimizer keep the frame virtual: nothing reads the frame itself unless
/// something forces it.
///
/// Returns the vref's OpRef for the matching [`walker_ec_leave`].
fn walker_ec_enter(
    ctx: &mut TraceCtx,
    callee_frame: OpRef,
    callee_ec: OpRef,
    concrete_frame: *mut pyre_interpreter::PyFrame,
    concrete_ec: *mut pyre_interpreter::PyExecutionContext,
) -> OpRef {
    // `frame.f_backref = self.topframeref` — the caller's vref moves into the
    // callee, unforced.  `emit_new_pyframe_inline_with_params` leaves the slot
    // at its constructor default, so this is the store that links the chain.
    let concrete_caller_topframeref = unsafe { (*concrete_ec).topframeref };
    let caller_topframeref = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[callee_ec],
        crate::descr::ec_topframeref_descr(),
    );
    ctx.set_opref_concrete(
        caller_topframeref,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_caller_topframeref as usize)),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, caller_topframeref],
        crate::descr::pyframe_f_backref_descr(),
    );
    // `self.topframeref = jit.virtual_ref(frame)`.
    let (vref, concrete_vref) = ctx.opimpl_virtual_ref(callee_frame, concrete_frame as usize);
    ctx.set_opref_concrete(
        vref,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_vref as usize)),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_ec, vref],
        crate::descr::ec_topframeref_descr(),
    );
    // The recording-time shadow of the `SetfieldGc` above: `PyFrame.f_backref`
    // is a `Type::Ref` field, so the emitted store carries the generational
    // barrier and the concrete store has to carry it too.  This frame is an
    // old-gen `FrameBox` and the caller's vref can be young.
    pyre_object::gc_hook::try_gc_write_barrier(concrete_frame as pyre_object::gc_hook::GCREF);
    majit_gc::bh_probe_note_store(
        concrete_frame as usize,
        crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
        4,
    );
    unsafe {
        (*concrete_frame).f_backref = concrete_caller_topframeref;
        (*concrete_ec).topframeref = concrete_vref as *mut pyre_interpreter::PyFrame;
    }
    vref
}

/// The box this walk holds for the frame at `frame_ptr`: the virtual of a
/// `virtual_ref` pair still open (an inlined caller), else the portal
/// virtualizable.  `None` when the walk has no box for it.
fn caller_frame_box_for(ctx: &TraceCtx, frame_ptr: usize) -> Option<OpRef> {
    if let Some(virtual_box) = ctx.virtualref_virtual_for_object_ptr(frame_ptr) {
        return Some(virtual_box);
    }
    let vable = ctx.standard_virtualizable_box()?;
    match ctx.standard_virtualizable_concrete()? {
        majit_ir::Value::Ref(gc) if gc.as_usize() == frame_ptr => Some(vable),
        _ => None,
    }
}

/// `executioncontext.py ExecutionContext.leave`'s frame-chain half, at
/// the return from an inlined call.
///
/// ```python
/// frame_vref = self.topframeref
/// self.topframeref = frame.f_backref
/// if frame.escaped or got_exception:
///     f_back = frame.f_backref()
///     if f_back:
///         f_back.mark_as_escaped()
///     frame_vref()
/// jit.virtual_ref_finish(frame_vref, frame)
/// ```
///
/// The profile-hook half (`if self.profilefunc: self._trace(frame,
/// 'leaveframe', w_exitvalue)`) stays with the interpreter's own
/// [`pyre_interpreter::PyExecutionContext::leave`].  Omitting it here does not
/// lose a PROFILE leave event, because `is_being_profiled` is a portal-driver
/// GREEN (`interp_jit.py greens = ['next_instr', 'is_being_profiled',
/// 'pycode']`): a trace is keyed on it, so one recorded with profiling off is
/// only ever entered with profiling off, and turning profiling on selects a
/// different green key rather than reusing this trace.
///
/// ⛔ That argument is sound for profiling and for nothing else, and it is not
/// licence to omit another hook here.  TRACING has no green, so no key change
/// separates a traced re-entry from an untraced one; and the events an inlined
/// callee owes a tracer are `call_trace` / `return_trace`, which upstream runs
/// from `pyframe.py execute_frame` rather than from `enter` / `leave`.
/// Whatever inlines the callee inlines those `ec.gettrace()` reads with it, so
/// upstream's trace carries a guard on them and a tracer installed later cannot
/// be missed.  This walker inlines neither, so an inlined callee reports
/// nothing for as long as the loop stays compiled.
///
/// What keeps the omission sound is that there is no inlined callee to lose
/// events for while a hook is installed:
/// `try_walker_inline_resolved_user_call_inner` declines the inline outright
/// once `ec.w_tracefunc` or `ec.profilefunc` is set, so the callee becomes a
/// residual call and reaches the interpreter's own `execute_frame`, which runs
/// `call_trace` / `return_trace` / `leave`.  A hook installed AFTER the trace
/// was recorded is answered by `record_portal_tracefunc_guard`
/// (`jitcode_dispatch/mod.rs`) for a trace function and by the green for a
/// profiler.
///
/// ⛔ Neither of the two gates that USED to hide this is a correctness
/// argument for it, and neither should be cited as one.
/// `eval_with_jit_inner` (`pyre-jit/src/eval.rs`) routing every frame for
/// which `frame_tracing_active` holds to `execute_frame_plain` is a
/// compilation gate, not a reporting one.  And the portal frame's `debugdata`
/// guard (`record_portal_debugdata_guard`) is about the CALLER's frame: it
/// says nothing about a callee whose own `call_trace` / `return_trace` this
/// walker declines to inline, and it does not fire at all for a global tracer
/// that leaves the caller's `f_trace` unset.
///
/// The escape branch runs in both worlds.  Concretely it marks the caller and
/// forces the leaving vref; in the trace it records the force as
/// `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` — upstream's "already forced
/// during tracing" form — rather than the NULL form, so `vref.forced` ends up
/// pointing at the virtual instead of staying NULL.  That is what keeps a
/// later read through a deeper escaped frame's `f_backref` from hitting
/// `InvalidVirtualRef`.
///
/// Upstream runs this from a `finally`, so the caller must too: every path out
/// of the callee level — normal return, raised exception, or a declined
/// sub-walk — has to reach it, or `virtualref_boxes` is left unbalanced and the
/// loop header trips `assert len(self.virtualref_boxes) == 0`.
///
/// `SwitchToBlackhole(ABORT_TOO_LONG)` is the exception. It leaves `_interpret`
/// before `leave`, and the trace is discarded whole, so that path uses
/// [`abandon_entered_frame`] and must not reach this function.
pub(crate) fn walker_ec_leave(
    ctx: &mut TraceCtx,
    callee_frame: OpRef,
    callee_ec: OpRef,
    concrete_frame: *mut pyre_interpreter::PyFrame,
    concrete_ec: *mut pyre_interpreter::PyExecutionContext,
    got_exception: bool,
) {
    // `pyopcode.py handle_operation_error` marks the frame finished before
    // an exception escapes into `ExecutionContext.leave`.  Ordinary returns
    // already publish `PyFrame.finish_value` at the lowered `*_return`
    // operation; this boundary supplies the exception sibling on this
    // inlined callee's own red frame.  A declined sub-walk or callee-loop
    // handoff has not finished executing and deliberately skips it.
    if got_exception {
        let flags_descr = crate::descr::pyframe_flags_descr();
        let live_flags = crate::state::opimpl_getfield_gc_i(ctx, callee_frame, flags_descr.clone());
        let finished_bit = ctx.const_int(i64::from(pyre_interpreter::PyFrame::FLAG_FRAME_FINISHED));
        let new_flags = ctx.record_op(OpCode::IntOr, &[live_flags, finished_bit]);
        ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[callee_frame, new_flags],
            flags_descr.clone(),
        );
        ctx.heapcache_setfield_cached(callee_frame, flags_descr.index(), new_flags);
        unsafe { (*concrete_frame).set_frame_finished_execution(true) };
    }
    // `self.topframeref = frame.f_backref` — no parens: the caller's vref
    // moves back unforced, so a caller frame that stayed virtual stays virtual.
    let concrete_f_backref = unsafe { (*concrete_frame).f_backref };
    let f_backref = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[callee_frame],
        crate::descr::pyframe_f_backref_descr(),
    );
    ctx.set_opref_concrete(
        f_backref,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_f_backref as usize)),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_ec, f_backref],
        crate::descr::ec_topframeref_descr(),
    );
    let escaped = unsafe {
        let frame_vref = (*concrete_ec).topframeref;
        (*concrete_ec).topframeref = concrete_f_backref;
        let escaped = (*concrete_frame).escaped() || got_exception;
        if escaped {
            // A frame that reached app level must keep its caller reachable
            // too, or the next `_getframe().f_back` walks into a frame the JIT
            // was still free to keep virtual.  `get_f_back` forces, which is
            // `f_back = frame.f_backref()` with the parens.
            let f_back = (*concrete_frame).get_f_back();
            if !f_back.is_null() {
                // `f_back.mark_as_escaped()` is traced: the store lands in
                // the trace and in the heapcache, on the box this walk holds
                // for the caller frame — the pair its `enter` pushed, or the
                // portal virtualizable.  Left to the concrete side alone, a
                // caller frame the bridge rebuilt keeps `flags` cached at
                // its rebuilt value while the object already carries the
                // bit, and the next read of it trips the cache-hit check.
                if let Some(f_back_box) = caller_frame_box_for(ctx, f_back as usize) {
                    let flags_descr = crate::descr::pyframe_flags_descr();
                    let live_flags =
                        crate::state::opimpl_getfield_gc_i(ctx, f_back_box, flags_descr.clone());
                    let escaped_bit =
                        ctx.const_int(i64::from(pyre_interpreter::PyFrame::FLAG_ESCAPED));
                    let new_flags = ctx.record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[f_back_box, new_flags],
                        flags_descr.clone(),
                    );
                    ctx.heapcache_setfield_cached(f_back_box, flags_descr.index(), new_flags);
                }
                (*f_back).mark_as_escaped();
            }
            // `frame_vref()` — force the leaving frame's own vref so it
            // outlives the JIT frame.
            let _ = pyre_interpreter::executioncontext::force_vref(frame_vref);
        }
        escaped
    };
    if escaped {
        // The concrete force above is only half of `frame_vref()`: the
        // optimizer reads the trace, not the heap.  Record the force as
        // `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` — the non-null second
        // operand is upstream's "this vref was forced during tracing already"
        // encoding, which `optimize_VIRTUAL_REF_FINISH` lowers to storing the
        // virtual into `vref.forced` (`virtualize.py`).
        //
        // Without it the finish below would emit the NULL form, leaving
        // `forced` NULL and `virtual_token` cleared, and a later read through
        // a deeper escaped frame's `f_backref` would hit `InvalidVirtualRef`.
        // `stop_tracking_virtualref` also replaces the vrefbox with
        // ConstPtr(NULL), so the finish that follows sees a non-vref and
        // records nothing — one finish, not two.
        //
        // Upstream reaches the same end state by a different route: it records
        // `frame_vref()` as a force and then the ordinary NULL finish, so
        // `forced` is written at runtime by `force_now` rather than by the
        // optimizer.  That route needs the `jit_force_virtual` lowering, which
        // is the one piece of this protocol pyre has not wired
        // (`jitcode_dispatch/mod.rs` item c — `_do_jit_force_virtual` is
        // tests-only, production reach 0).  Both forms are already understood
        // by `optimize_VIRTUAL_REF_FINISH`, so this uses the one that is
        // reachable; converge on the upstream spelling when the force lowering
        // lands.
        let live = ctx.virtualref_boxes_len();
        let vref_is_live = ctx
            .innermost_virtualref_vref()
            .is_some_and(|(vrefbox, _)| vrefbox.as_const_ptr().is_none_or(|vref| vref.0 != 0));
        if live >= 2 && vref_is_live {
            ctx.stop_tracking_virtualref(live - 2);
        }
    }
    // `jit.virtual_ref_finish(frame_vref, frame)`.
    ctx.opimpl_virtual_ref_finish(callee_frame);
}

/// `SwitchToBlackhole` leaves `_interpret` without `ExecutionContext.leave`.
/// Undo the concrete `topframeref` this enter installed and drop the
/// tracing-only vref pair. Do not record `VIRTUAL_REF_FINISH` or
/// `LEAVE_PORTAL_FRAME`: the trace is discarded whole.
///
/// `concrete_frame` is dereferenced (`f_backref`, `escaped`). Callers pass
/// the live address from [`TraceCtx::virtualref_entry_ptr`]: the `usize`
/// `opimpl_virtual_ref` stored beside the pair is the address at the push,
/// and a moving collection forwards the stamp through `concrete_of_opref`
/// while leaving that copy behind.
pub(crate) fn abandon_entered_frame(
    ctx: &mut TraceCtx,
    concrete_frame: *mut pyre_interpreter::PyFrame,
    concrete_ec: *mut pyre_interpreter::PyExecutionContext,
) {
    if concrete_frame.is_null() || concrete_ec.is_null() {
        return;
    }
    unsafe {
        let frame_vref = (*concrete_ec).topframeref;
        let concrete_f_backref = (*concrete_frame).f_backref;
        (*concrete_ec).topframeref = concrete_f_backref;
        if (*concrete_frame).escaped() {
            let f_back = (*concrete_frame).get_f_back();
            if !f_back.is_null() {
                (*f_back).mark_as_escaped();
            }
            let _ = pyre_interpreter::executioncontext::force_vref(frame_vref);
        }
    }
    ctx.discard_innermost_virtualref_if_frame(concrete_frame as usize);
}

/// Concrete-unwind vref scopes this walk opened, without recording a leave.
/// A scope whose pair does not name the frame it claims stops the loop so an
/// outer scope is not popped.
pub(crate) fn unwind_entered_scopes_above<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &Sym,
    entry_depth: usize,
) {
    let concrete_ec = sym.concrete_execution_context() as *mut pyre_interpreter::PyExecutionContext;
    while ctx.virtualref_boxes_len() > entry_depth {
        let Some(entry) = ctx.innermost_virtualref_virtual() else {
            break;
        };
        let frame_ptr = ctx.virtualref_entry_ptr(entry) as *mut pyre_interpreter::PyFrame;
        let before = ctx.virtualref_boxes_len();
        abandon_entered_frame(ctx, frame_ptr, concrete_ec);
        if ctx.virtualref_boxes_len() == before {
            break;
        }
    }
}

/// `executioncontext.py ExecutionContext.enter`'s frame-chain half, recorded
/// with no concrete shadow.
///
/// ```python
/// frame.f_backref = self.topframeref
/// self.topframeref = jit.virtual_ref(frame)
/// ```
///
/// [`walker_ec_enter`] is the seeded-callee form: it has a live callee frame
/// at recording time, so it mirrors every store concretely and takes a real
/// vref of the frame.  This fold builds its callee out of recorded operations
/// alone, so what is recorded here is the interpreter-level reading of
/// `jit.virtual_ref(frame)`: the frame pointer itself (`_jit_vref.py
/// lowleveltype = OBJECTPTR`, no allocation), which is exactly what
/// [`pyre_interpreter::PyExecutionContext::enter`] stores.
///
/// ⛔ Recording a `VIRTUAL_REF` here instead is measurable and is a net loss.
/// The callee this fold builds is read back through `gettopframe_nohidden`,
/// which is `force_vref`: `locals()` in a self-recursive callee forces on
/// every call, and an unwind that crosses suspended copies of the frame
/// forces once per crossing.  Each force is a GUARD_NOT_FORCED failure, and
/// `ResumeGuardForcedDescr.handle_fail` never compiles one — it blackholes.
/// Measured over the synthetic corpus, publishing the vref moves eight
/// fixtures by one to three guard failures and moves two by +273
/// (`recursive_forced_frame_kept_stack`) and +169
/// (`selfrec_tail_exception_unwind`): +430 net, all of it blackholed.  The
/// portal doors no longer need it — [`crate::PortalEntry`]'s caller names the
/// door — so the remaining question is what makes those forces unnecessary,
/// not how to spell the enter.
///
/// Upstream reaches `ec.enter` on this path too: `interp_jit.py` puts
/// `jit_merge_point` on `PyFrame.dispatch`, so the CALL_ASSEMBLER that
/// replaces the callee's merge point is recorded with `pyframe.py
/// execute_frame`'s `ec.enter(self)` already traced ahead of it.  pyre's
/// portal sits one level up, at `execute_frame` itself, so the fold jumps
/// over the enter unless it records it here.
fn record_ec_enter_frame_chain(ctx: &mut TraceCtx, callee_frame: OpRef, callee_ec: OpRef) {
    let caller_topframeref = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[callee_ec],
        crate::descr::ec_topframeref_descr(),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, caller_topframeref],
        crate::descr::pyframe_f_backref_descr(),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_ec, callee_frame],
        crate::descr::ec_topframeref_descr(),
    );
}

/// `executioncontext.py ExecutionContext.leave`'s frame-chain half, in
/// `execute_frame`'s `finally` position, recorded with no concrete shadow.
///
/// ```python
/// self.topframeref = frame.f_backref
/// ```
///
/// No parens on `f_backref`, so a caller frame that stayed virtual stays
/// virtual.  The escape branch and `virtual_ref_finish` that [`walker_ec_leave`]
/// carries have no counterpart here for the same reason the enter takes no
/// vref: this level never held one.  The profile-hook half stays with
/// [`pyre_interpreter::PyExecutionContext::leave`] — see [`walker_ec_leave`]
/// for why the portal-driver GREEN makes that split sound.
///
/// Record this at the earliest point the call is known to have returned, not
/// on the trace's fall-through tail: a guard recorded ahead of it skips it on
/// the exit it is needed for, and every skipped restore leaves a frame
/// published in `topframeref` that the next `enter` chains onto.  The portal's
/// `TopFrameRefGuard` (`pyre-jit/src/eval.rs`) balances the slot only when the
/// activation itself ends, which is far too late for a loop that raises.
fn record_ec_leave_frame_chain(ctx: &mut TraceCtx, callee_frame: OpRef, callee_ec: OpRef) {
    let f_backref = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[callee_frame],
        crate::descr::pyframe_f_backref_descr(),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_ec, f_backref],
        crate::descr::ec_topframeref_descr(),
    );
}

#[cfg(test)]
mod portal_frame_chain_tests {
    use super::*;

    /// The fold's recorded `enter` publishes the callee frame itself, and its
    /// recorded `leave` restores `f_backref` — neither takes a vref.
    ///
    /// The op shape is the contract because publishing a `VIRTUAL_REF` here
    /// is a measured net loss: the forces it admits are GUARD_NOT_FORCED
    /// failures, which are blackholed rather than compiled. See
    /// [`record_ec_enter_frame_chain`] for the census.
    #[test]
    fn the_recorded_enter_and_leave_take_no_virtual_ref() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let frame = OpRef::input_arg_ref(0);
        let ec = OpRef::input_arg_ref(1);

        record_ec_enter_frame_chain(&mut ctx, frame, ec);
        record_ec_leave_frame_chain(&mut ctx, frame, ec);
        assert_eq!(
            ctx.virtualref_boxes_len(),
            0,
            "this level holds no vref, so it owes no virtual_ref_finish and \
             must not leave `virtualref_boxes` unbalanced at the loop header",
        );

        let tree_loop = ctx.into_tree_loop();
        let opcodes: Vec<OpCode> = tree_loop.ops.iter().map(|op| op.opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                // `frame.f_backref = self.topframeref`
                OpCode::GetfieldGcR,
                OpCode::SetfieldGc,
                // `self.topframeref = frame`
                OpCode::SetfieldGc,
                // `self.topframeref = frame.f_backref`
                OpCode::GetfieldGcR,
                OpCode::SetfieldGc,
            ],
        );
    }
}

/// `BuiltinCode.func` is an RPython PBC: the codewriter turns its finite
/// target family into an indirect call whose address is resolved back to the
/// generated target JitCode by `MetaInterpStaticData.bytecode_for_address`
/// (`pyjitpl.py`).  The interpreter-level `call_fn` helper hides
/// that indirect call behind `Function -> BuiltinCode -> func`, so recover
/// the same target here and enter the generated wrapper with its one red
/// `&[PyObjectRef]` argument.
///
/// The slice is represented to the translated body as a GC array.  Build that
/// array in trace IR and seed its heap-cache entries from the live CALL
/// operands; this preserves a distinct red receiver for every method call.
/// In particular, a bound Method's receiver is read from its immutable
/// `w_self` field rather than baked from the recording-time object.
pub(crate) fn try_walker_inline_builtin_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    ref_operand_offset: usize,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    runtime_helper: majit_ir::RuntimeHelperKind,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let is_call_kw = runtime_helper == majit_ir::RuntimeHelperKind::CallKw;
    if !ctx.is_authoritative_executor
        // `CallKw` is admitted when the builtin has a Signature: keyword
        // VALUES stay the traced boxes and are permuted into parameter
        // slots (`Arguments` + gateway unwrap).  Building a marker dict
        // from recording-time concretes as ConstPtr is forbidden.
        || (runtime_helper != majit_ir::RuntimeHelperKind::CallFn && !is_call_kw)
        || r_args.len() < 2
        || dst_bank != 'r'
    {
        return Ok(None);
    }

    let mut arg_concretes = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    let concrete_lead = if is_call_kw { 3 } else { 2 };
    for i in 0..concrete_lead.min(r_args.len()) {
        if matches!(arg_concretes.get(i), Some(ConcreteValue::Null)) {
            if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(r_args[i]) {
                let bits = r.as_usize();
                // Same unaligned `cast_int_to_ptr` concrete as the plain
                // inline path. A bool bit-cast is not a callable.
                if r != majit_ir::GcRef::NO_CONCRETE
                    && bits != 0
                    && bits % std::mem::align_of::<usize>() == 0
                {
                    arg_concretes[i] = ConcreteValue::Ref(bits as pyre_object::PyObjectRef);
                }
            }
        }
    }
    let ConcreteValue::Ref(callable_operand) = arg_concretes[0] else {
        return Ok(None);
    };
    if callable_operand.is_null()
        || (callable_operand as usize) % std::mem::align_of::<usize>() != 0
    {
        return Ok(None);
    }
    // `try_walker_trace_exception_new` turns `Exc(args)` into a virtual
    // instance the raise fold can chain.  Descending `type.__call__` instead
    // records `value_error_one_arg` and `exc_init_one_positional`
    // (`dont_look_inside`), so the instance stays concrete and a caught
    // traceback cannot leave the bridge.  A keyword call is not that CallFn
    // shape and keeps the descent.
    if !is_call_kw
        && unsafe {
            pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(callable_operand)
        }
    {
        return Ok(None);
    }
    let null_or_self = match arg_concretes[1] {
        ConcreteValue::Ref(value) => value,
        ConcreteValue::Null => pyre_object::PY_NULL,
        _ => return Ok(None),
    };
    let method_form = !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL;
    let bound_method = !method_form && unsafe { pyre_object::is_method(callable_operand) };
    let (callable, receiver) = if bound_method {
        let function = unsafe { pyre_object::w_method_get_func(callable_operand) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable_operand) };
        if function.is_null() || receiver.is_null() {
            return Ok(None);
        }
        (function, Some(receiver))
    } else {
        (callable_operand, method_form.then_some(null_or_self))
    };
    // `descroperation.py DescrOperation.call_args`'s third arm: a callable
    // that is neither a `Function` nor a `Method` dispatches through
    // `space.lookup(w_obj, '__call__')`, bound to the object.  When that
    // `__call__` is a builtin gateway, the wrapper is entered below with the
    // instance as its receiver, and the lookup is pinned on the instance's
    // class the way `space.lookup` promotes the type.  An app-level `__call__`
    // belongs to `try_walker_inline_call`'s `resolve_instance_dunder_call`.
    let mut instance_call_class = None;
    // `descr_call` on a class whose `__new__` is a builtin gateway: the class
    // is the wrapper's receiver.  See `resolve_type_call_builtin_new`.
    let mut type_call_class = None;
    // When `__init__` is also a builtin gateway (`list.__init__`), walk it
    // after `__new__` and keep the instance — `typeobject.py descr_call`.
    let mut type_call_init = None;
    // A type-dict `__call__` installed from a `BuiltinCode` is a slot wrapper:
    // the same `Function` carrier under `SLOT_WRAPPER_TYPE`, so the carrier
    // test admits it and the `Function.code` read below is the same load.
    // One-argument `type(x)` enters `type_descr_call_impl`'s graph
    // (`__majit_wrap_type_query`) instead of the instantiation emit.
    // Three-argument `type(name, bases, dict)` walks `type_descr_new`
    // (`descr__new__`); `_create_new_type` is residual because
    // `look_inside_graph` refuses its loops, so `_check_surrogate` is
    // not in-trace. `resolve_type_call_builtin_new` returns None for
    // `space.w_type` itself (that path is user-class `P()`), so the
    // 3-arg metatype call has to select `__new__` here.
    let type_is_metatype = unsafe {
        pyre_object::is_type(callable_operand)
            && std::ptr::eq(callable_operand, pyre_interpreter::typedef::w_type())
    };
    let type_query =
        !method_form && !bound_method && !is_call_kw && r_args.len() == 3 && type_is_metatype;
    let type_new_3arg = !method_form
        && !bound_method
        && !is_call_kw
        && r_args.len() == 5
        && type_is_metatype
        && arg_concretes.len() >= 5
        && arg_concretes[2..5].iter().all(|value| match value {
            ConcreteValue::Ref(obj) => !obj.is_null() && *obj != pyre_object::PY_NULL,
            _ => false,
        });
    let (callable, receiver) = if type_query {
        (callable, None)
    } else if type_new_3arg {
        let tp_new = unsafe {
            let found = pyre_interpreter::baseobjspace::lookup_in_type(callable_operand, "__new__");
            match found {
                Some(w) if pyre_object::function::is_exact_staticmethod(w) => {
                    let func = pyre_object::function::w_staticmethod_get_func(w);
                    if func.is_null() { None } else { Some(func) }
                }
                other => other,
            }
        };
        match tp_new {
            Some(tp_new) if unsafe { pyre_interpreter::is_function_carrier(tp_new) } => {
                type_call_class = Some(callable_operand);
                (tp_new, Some(callable_operand))
            }
            _ => (callable, receiver),
        }
    } else if method_form || bound_method || unsafe { pyre_interpreter::is_function(callable) } {
        (callable, receiver)
    } else {
        match unsafe { lookup_instance_dunder_call(callable) } {
            Some((method, w_class, version_tag)) => {
                instance_call_class = Some((w_class, version_tag));
                (method, Some(callable_operand))
            }
            // A CALL_KW's third operand is the kwnames tuple, not an
            // argument, so the type-call path admits positional calls only.
            None => match (!is_call_kw)
                .then(|| unsafe { resolve_type_call_builtin_new(callable, &arg_concretes[2..]) })
                .flatten()
            {
                Some((tp_new, tp_init)) => {
                    // `__new__` allocates, which is an effect, so `__init__`
                    // must be known-descendable before that walk starts:
                    // a later decline cannot rewind the instance.
                    if let Some(init) = tp_init {
                        let init_items = 1 + r_args.len().saturating_sub(2);
                        if let Some(why) = builtin_gateway_undescendable_reason(init, init_items) {
                            if fbw_inline_diag_enabled() {
                                eprintln!(
                                    "[builtin-inline-decline] pc={} why=type.__call__ __init__ not descendable class={} detail={}",
                                    op.pc,
                                    unsafe { pyre_object::w_type_get_name(callable_operand) },
                                    why,
                                );
                            }
                            return Ok(None);
                        }
                        type_call_init = Some(init);
                    }
                    type_call_class = Some(callable_operand);
                    (tp_new, Some(callable_operand))
                }
                None => (callable, receiver),
            },
        }
    };
    // Every decline below is silent otherwise, and they are not
    // interchangeable: `not is_function` is a class call or another
    // non-Function callable, while `no jitcode for address` names a builtin
    // whose `BuiltinCode.func` is not a member of the PBC family
    // `builtin_wrapper_indirect_graphs` builds — i.e. one registered without a
    // `__majit_wrap_*` gateway.  The address is what identifies the missing
    // member, so print it.
    //
    // Every bail between here and the descent below names itself for the same
    // reason.  With the tail of this function silent, a run of
    // `abc_instancecheck_weak_cache` with the scan's decline withdrawn printed
    // no builtin-inline line at
    // all and recorded the same 69 ops as the gated run, which reads as "the
    // descent scan is the wall" when the scan had already been switched off and
    // something downstream refused instead.
    macro_rules! builtin_inline_decline {
        ($why:expr, $addr:expr) => {
            if fbw_inline_diag_enabled() {
                let name = if unsafe { pyre_interpreter::is_function(callable) } {
                    unsafe { pyre_interpreter::function_get_name(callable) }
                } else {
                    ""
                };
                eprintln!(
                    "[builtin-inline-decline] pc={} why={} callable={} operand={} fnaddr={:#x} name={}",
                    op.pc,
                    $why,
                    unsafe { pyre_object::type_name_of(callable) },
                    unsafe { pyre_object::type_name_of(callable_operand) },
                    $addr,
                    name,
                );
            }
        };
    }
    let fnaddr = if type_query {
        pyre_interpreter::call::__majit_wrap_type_query as *const () as usize
    } else {
        if !unsafe { pyre_interpreter::is_function_carrier(callable) } {
            builtin_inline_decline!("not is_function", 0usize);
            return Ok(None);
        }
        let builtin_code =
            unsafe { pyre_interpreter::function_get_code(callable) as pyre_object::PyObjectRef };
        if builtin_code.is_null() || !unsafe { pyre_interpreter::is_builtin_code(builtin_code) } {
            builtin_inline_decline!("not builtin_code", 0usize);
            return Ok(None);
        }
        unsafe { pyre_interpreter::builtin_code_get(builtin_code) as usize }
    };
    let builtin_code = if type_query {
        pyre_object::PY_NULL
    } else {
        unsafe { pyre_interpreter::function_get_code(callable) as pyre_object::PyObjectRef }
    };
    let Some(jitcode) = crate::state::bytecode_for_address(fnaddr) else {
        builtin_inline_decline!("no jitcode for address", fnaddr);
        return Ok(None);
    };
    let Some(body) = crate::jitcode_dispatch::sub_jitcode_body_by_index(jitcode.index()) else {
        builtin_inline_decline!("no sub jitcode body", fnaddr);
        return Ok(None);
    };
    // `warmspot.py:281-282` publishes every CodeWriter JitCode into the one
    // MetaInterpStaticData registry.  The frozen table is lazy in pyre, so
    // publish this wrapper before any guard snapshot names its absolute index.
    if crate::state::ensure_build_time_jitcode_at(jitcode.index()).is_none() {
        builtin_inline_decline!("wrapper jitcode absent from the build-time table", fnaddr);
        return Ok(None);
    }
    if body.num_regs_r < 1 {
        builtin_inline_decline!("wrapper body has no ref register", fnaddr);
        return Ok(None);
    }
    // PyPy installs the builtin `__len__` implementations through
    // `use_special_method_shortcut`, so an exact builtin receiver enters its
    // layout body without visiting the generic lookup/error arms.  Admit that
    // execution path even when the body-wide safety scan sees a blocker on a
    // different arm.  Suppressing the row restores the conservative scan and
    // is the A/B proof that the generated descent, rather than a hand emitter,
    // supplies the trace.
    let builtin_len_call = receiver.is_none()
        && r_args.len() == 3
        && pyre_interpreter::builtins::is_builtin_len_function(callable);
    let len_receiver = match arg_concretes.get(2) {
        Some(ConcreteValue::Ref(obj)) if builtin_len_call => *obj,
        _ => pyre_object::PY_NULL,
    };
    // The receivers the shortcut below does not reach are the ones whose
    // length is a Python call rather than a layout read.  They fail the same
    // body-wide scan for the same reason, and the descent cannot serve them
    // either, so they take their own route into the resolved `__len__`.
    if builtin_len_call
        && let Some(inlined) = spec_gate(SpecFold::LenUserDunder, || {
            try_walker_inline_len_dunder(
                ctx,
                op,
                code,
                funcptr,
                r_args,
                call_descr,
                dst,
                r_args[2],
                len_receiver,
            )
        })?
    {
        return Ok(Some(inlined));
    }
    let builtin_len_shortcut = builtin_len_call
        && spec_gate(SpecFold::BuiltinLenDescent, || {
            Ok::<Option<()>, DispatchError>(
                unsafe { exact_builtin_len_shortcut_receiver(len_receiver) }.then_some(()),
            )
        })?
        .is_some();
    let builtin_sig = if type_query || builtin_code.is_null() {
        None
    } else {
        unsafe { pyre_interpreter::gateway::builtin_code_get_signature(builtin_code) }
    };
    if is_call_kw && builtin_sig.is_none() {
        builtin_inline_decline!("call_kw builtin has no Signature", fnaddr);
        return Ok(None);
    }
    let wrapper_item_count = if is_call_kw {
        builtin_sig.map(|s| s.argnames.len()).unwrap_or(0)
    } else {
        usize::from(receiver.is_some()) + (r_args.len() - 2)
    };
    // `W_CData.call` on a function pointer. The scan's red `kind` join
    // reaches `allocate_stable` on an arm `W_CTypeFunc.call` does not take
    // after it promotes the ctype. The generated descent records the call.
    let cdata_func_descent =
        receiver.is_some_and(pyre_interpreter::importing::cdata_is_function_pointer);
    if !builtin_len_shortcut
        && !cdata_func_descent
        && let Some(decline) = descent_decline(jitcode.index(), &[(0, wrapper_item_count)])
    {
        if matches!(decline, DescentDecline::Helper(_)) {
            log_descent_unlowered_helper_blockers(jitcode.index());
        }
        builtin_inline_decline!(decline, fnaddr);
        return Ok(None);
    }
    let nested_helper = ctx.fbw_mode.inline_subwalk;
    // A translated helper is transparent to blackhole execution, but its
    // Python caller is not, so a nested inline preserves that caller at the
    // helper's CALL entry.
    //
    // The root portal frame does NOT get the same treatment.  Pushing an entry
    // frame there makes `parent_frames` non-empty, which is exactly the
    // precondition `walker_capture_transparent_helper_snapshot` needs, so
    // root-level helper guards start emitting a two-section snapshot whose
    // innermost section is the helper JitCode with the no-Python-coordinate
    // `py_pc`.  pyre's resume chain is Python-level: `build_resumed_frames`
    // (`pyre-jit/src/eval.rs`) turns each section into a `ResumedFrame` and has
    // nothing that executes a JitCode section, so the helper section decodes to
    // a null `code`, borrows the caller's `vable_ni`, and — by making
    // `resumed_frames.len() > 1` — reroutes the guard onto the multi-frame
    // resume, where `resumed_catch_level` bails at the null `code` and the
    // Python-level values come back wrong (`synth/gc_deque_backing_list`
    // reported `grow_each_round 987334` against `2966`, and `test.test_pickle`
    // lost its memo/`struct` round trip).  Keep the root level on the
    // caller-boundary resume until the resume side can carry a non-Python
    // section.
    //
    // Inside another canonical helper body there is no Python caller to
    // preserve at this CALL: the enclosing helper already published its own
    // caller's resume point, and this call inherits it
    // (`specialize::HelperEntry::EnclosingHelper`).
    let nested_helper_entry = if nested_helper && !ctx.fbw_mode.transparent_helper_subwalk {
        match compute_inline_helper_call_entry_frame(ctx, op.pc) {
            Ok(frame) => Some(frame),
            Err(_) => {
                builtin_inline_decline!("no entry frame for the nested helper call", fnaddr);
                return Ok(None);
            }
        }
    } else {
        None
    };
    // Guards inside the generated wrapper must resume at the outer Python
    // CALL, because helper JitCodes have no blackhole entry point of their
    // own.  The full-body symbol is the authority for that caller frame's
    // liveness and resume coordinate (the same setup used by the orthodox
    // w_list_append descent below).  Resolve it before recording any guards
    // or synthetic allocations so a missing coordinate is a clean decline.
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        builtin_inline_decline!("walk carries no snapshot symbol", fnaddr);
        return Ok(None);
    }
    // SAFETY: snapshot_sym is installed for the lifetime of the enclosing
    // full-body walk and is read-only here.
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        builtin_inline_decline!("snapshot symbol names no jitcode", fnaddr);
        return Ok(None);
    }
    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker) = if nested_helper {
        (
            ctx.entry_py_pc(),
            0,
            ctx.outer_jitcode_index,
            ctx.outer_resume_marker_jit_pc,
        )
    } else {
        unsafe {
            let jc = &*sym.jitcode();
            let jc_index = jc.index as u32;
            let marker = jc.payload.resume_marker_for_jitcode_pc(op.pc);
            // Forward py twin first (#73 phase-3): equals the
            // containing coordinate plus trivia normalization by
            // construction; the containing lookup survives for the
            // empty-twin class, and the trivia skip below is an identity
            // on the twin path.
            let mut py = jc
                .payload
                .forward_py_pc_for_jitcode_pc(op.pc)
                .unwrap_or_else(|| {
                    crate::py_coord::note_empty_twin_fallback(
                        "builtin_call",
                        jc.index,
                        op.pc as i32,
                    );
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op.pc)
                });
            if jc.payload.code_ptr.is_null() {
                (py, sym.valuestackdepth() as i64, jc_index, marker)
            } else {
                let codeobj = &*jc.payload.code_ptr;
                py = skip_python_trivia_forward(codeobj, py as usize) as u32;
                let depth = if jc.payload.depth_trivia_populated() {
                    jc.payload.depth_trivia_for_jitcode_pc(op.pc)
                } else {
                    crate::liveness::liveness_for(jc.payload.code_ptr)
                        .depth_at_py_pc()
                        .get(py as usize)
                        .copied()
                };
                let vsd = depth
                    .map(|d| (sym.nlocals() + d as usize) as i64)
                    .unwrap_or(sym.valuestackdepth() as i64);
                (py, vsd, jc_index, marker)
            }
        }
    };
    // Rewind point for the un-lowered-helper decline below.  Nothing above
    // this line records IR or touches the heap cache, so cutting back to it
    // leaves the caller's trace exactly as the ordinary residual call found it.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let call_site_active = if nested_helper_entry.is_some() {
        ctx.frame_state.borrow().outer_active_boxes.clone()
    } else {
        let call_site_word = call_site_marker
            .map(|marker| marker as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let vstack_boxes = ctx.frame_state.borrow().vstack_boxes.clone();
        let vstack = ctx.vstack_valid.then_some(vstack_boxes.as_slice());
        collect_outer_active_boxes(
            sym,
            ctx.trace_ctx,
            ctx.registers_i,
            ctx.registers_r,
            ctx.registers_f,
            outer_jitcode_index,
            false,
            call_site_word,
            op.pc as i32,
            OuterActiveBoxesEntryTwin::Plain,
            "builtin_wrapper_call_site",
            vstack,
            &[],
            None,
        )
    };

    let mut callable_guard_op = r_args[0];
    let mut receiver_op = method_form.then_some(r_args[1]);
    if let Some((w_class, version_tag)) = instance_call_class {
        // The promoted-class lookup shape: the class and its version tag
        // together fix the `__call__` answer, so the wrapper this call
        // dispatches to is a constant of the pinned class rather than a field
        // the `GuardValue` below would have to read off the operand.
        walker_guard_exception_attr_slot(
            ctx,
            op.pc,
            r_args[0],
            callable_operand,
            w_class,
            version_tag,
        )?;
        // After the pin, never before: the pin is what makes the cell pointer a
        // constant.  The tag covers a REPLACING store, which builds a fresh
        // cell and revokes the tag here and in every subclass; an in-place
        // write returns `None` from `write_cell` and moves nothing, so the
        // entry that resolved this wrapper is read and promoted.  The same pair
        // `try_walker_inline_call`'s app-level `__call__` arm carries.
        if let Some((cell, expected)) =
            unsafe { inline_attr_cell_guard(w_class, "__call__", callable) }
        {
            walker_promote_object_mutable_cell(ctx, op.pc, cell, expected)?;
        }
        callable_guard_op = ctx.trace_ctx.const_ref(callable as i64);
        receiver_op = Some(r_args[0]);
    }
    if let Some(w_type) = type_call_class {
        // Pin the class and the type-dict version the `__new__` / `__init__`
        // lookups resolved against, then each argument's class: the wrapper
        // this call dispatches to is a constant of the pinned class.
        let type_const = ctx.trace_ctx.const_ref(w_type as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[r_args[0], type_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(r_args[0], type_const);
        walker_pin_type_version_tag(ctx, op.pc, type_const)?;
        for (&arg, concrete) in r_args[2..].iter().zip(&arg_concretes[2..]) {
            let ConcreteValue::Ref(concrete) = *concrete else {
                unreachable!("resolve_type_call_builtin_new admits ref arguments only")
            };
            let w_class = unsafe { (*concrete).w_class };
            let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
            walker_guard_exception_attr_slot(ctx, op.pc, arg, concrete, w_class, version_tag)?;
        }
        callable_guard_op = ctx.trace_ctx.const_ref(callable as i64);
        receiver_op = Some(type_const);
    }
    if bound_method {
        // pypy/interpreter/function.py `_Method._immutable_fields_`:
        // guard the carrier layout, read both fields live, and only promote
        // the immutable function identity used to select BuiltinCode.func.
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        walker_guard_class(ctx, op.pc, r_args[0], method_type_addr)?;
        callable_guard_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_function_descr(),
        );
        let live_receiver = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            live_receiver,
            majit_ir::Value::Ref(majit_ir::GcRef(receiver.unwrap() as usize)),
        );
        receiver_op = Some(live_receiver);
    }
    if !callable_guard_op.is_constant() {
        // `callable_guard_op` can be the live `Method.w_function` field read
        // above.  The GuardValue pins it to `callable` at runtime; retain the
        // same recording-time Box.value so a multi-frame bridge recipe does
        // not mistake an unstamped field-read shadow for Python NULL.
        ctx.trace_ctx.set_opref_concrete(
            callable_guard_op,
            majit_ir::Value::Ref(majit_ir::GcRef(callable as usize)),
        );
        let expected = ctx.trace_ctx.const_ref(callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_guard_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_guard_op, expected);
    }

    let mut wrapper_items = Vec::with_capacity(r_args.len().saturating_sub(1));
    let mut wrapper_item_concretes = Vec::with_capacity(arg_concretes.len().saturating_sub(1));
    if is_call_kw {
        // The bind decline below comes after the callable `GuardValue`, so it
        // rewinds the way every other decline past `pre_fold_pos` does: the
        // guard named a snapshot the cut hands to another operation, which
        // the unroll's Phase 2 remap reports as a cache miss.  The missing
        // `Signature` arm is already answered before that guard is recorded
        // and only restates the invariant here.
        let Some(sig) = builtin_sig else {
            builtin_inline_decline!("call_kw builtin has no Signature", fnaddr);
            cut_declined_subwalk(ctx, pre_fold_pos);
            return Ok(None);
        };
        let bound_receiver = receiver_op.map(|op| {
            (
                op,
                ConcreteValue::Ref(receiver.unwrap_or(pyre_object::PY_NULL)),
            )
        });
        let Some((bound_ops, bound_conc)) =
            builtin_call_kw_bind_signature(r_args, &arg_concretes, sig, bound_receiver)
        else {
            builtin_inline_decline!("call_kw signature bind declined", fnaddr);
            cut_declined_subwalk(ctx, pre_fold_pos);
            return Ok(None);
        };
        wrapper_items = bound_ops;
        wrapper_item_concretes = bound_conc;
        for (op, conc) in wrapper_items
            .iter_mut()
            .zip(wrapper_item_concretes.iter_mut())
        {
            if *op == OpRef::NONE {
                // Missing keyword-only slot: the wrapper defaults it.  A
                // runtime PY_NULL box, not a recording-time value.
                *op = ctx.trace_ctx.const_ref(0);
                *conc = ConcreteValue::Null;
            }
        }
    } else {
        if let Some(receiver_op) = receiver_op {
            wrapper_items.push(receiver_op);
            wrapper_item_concretes.push(ConcreteValue::Ref(receiver.unwrap()));
        }
        wrapper_items.extend_from_slice(&r_args[2..]);
        wrapper_item_concretes.extend_from_slice(&arg_concretes[2..]);
    }
    for (&item, concrete) in wrapper_items.iter().zip(&wrapper_item_concretes) {
        if let ConcreteValue::Ref(value) = concrete
            && !value.is_null()
        {
            // Box.value is the recording-time shadow, not a compile-time
            // constant.  The generated wrapper's getarrayitem returns this
            // same live box; seeding it lets py_type_check choose its observed
            // arm and emit the corresponding guards while retaining `self`
            // as a red input.
            ctx.trace_ctx.try_set_opref_concrete(
                item,
                majit_ir::Value::Ref(majit_ir::GcRef(*value as usize)),
            );
        }
    }

    let array_descr = crate::state::pyobject_gcarray_descr();
    let len = ctx.trace_ctx.const_int(wrapper_items.len() as i64);
    let args_array =
        ctx.trace_ctx
            .record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr.clone());
    ctx.trace_ctx
        .heap_cache_mut()
        .new_array(args_array, len, true);
    for (index, &item) in wrapper_items.iter().enumerate() {
        let index = ctx.trace_ctx.const_int(index as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[args_array, index, item],
            array_descr.clone(),
        );
        // The wrapper reads its `&[PyObjectRef]` argument through this same
        // ArrayDescr, so its getarrayitem returns the live CALL operand.
        ctx.trace_ctx
            .heapcache_setarrayitem(args_array, index, array_descr.index(), item);
    }

    let saved_vable = if !nested_helper && sym.owns_virtualizable_shadow() {
        let saved = crate::trace_opcode::save_vable_resume_scalars(ctx.trace_ctx);
        let last_instr = call_site_py_pc as i64 - 1;
        let last_instr_op = ctx.trace_ctx.const_int(last_instr);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            last_instr_op,
            Value::Int(last_instr),
        );
        let vsd_op = ctx.trace_ctx.const_int(vsd_value);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "valuestackdepth",
            vsd_op,
            Value::Int(vsd_value),
        );
        Some(saved)
    } else {
        None
    };

    // Build-time canonical helper JitCodes use the one global Assembler
    // descriptor pool.  Temporarily give the wrapper sub-frame that pool;
    // its nested inline_call descriptors then resolve the generated child
    // JitCodes (e.g. W_Random::random -> Random::random) by global index.
    let saved_entry = ctx.entry_py_pc;
    let saved_marker = ctx.outer_resume_marker_jit_pc;
    let saved_oji = ctx.outer_jitcode_index;
    let saved_active = std::mem::take(&mut ctx.frame_state.borrow_mut().outer_active_boxes);
    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    let saved_fbw_mode = ctx.fbw_mode;
    // The odometer counts every journaled effect — a store, a list append/pop,
    // a cell store — where the store journal counts only the first.  The
    // rewind arm below reads it to decide whether this descent already applied
    // something.
    let effects_before = fbw_executed_effect_count();
    let unjournaled_before = fbw_has_unjournaled_effect();
    let rewind = SubWalkRewind {
        pre_fold_pos,
        effects_before,
        unjournaled_before,
    };
    // A nested helper's byte offset is not a coordinate in the Python
    // jitcode. Keep the outermost CALL so a portal entered from inside
    // the helper still names that call.
    if !(ctx.fbw_mode.inline_subwalk && matches!(saved_entry, EntryPyPc::Jit(_))) {
        ctx.entry_py_pc = EntryPyPc::Jit(op.pc);
    }
    ctx.outer_resume_marker_jit_pc = call_site_marker;
    ctx.outer_jitcode_index = outer_jitcode_index;
    ctx.frame_state.borrow_mut().outer_active_boxes = call_site_active;
    ctx.descr_refs = crate::jitcode_runtime::descr_ref_table();
    ctx.raw_descrs = RawDescrPool::Global;
    ctx.sub_jitcode_lookup = &GLOBAL_SUB_JITCODE_LOOKUP_FN;
    ctx.fbw_mode.inline_subwalk = true;
    ctx.fbw_mode.inline_caller_py_pc = Some(call_site_py_pc);
    ctx.fbw_mode.transparent_helper_jitcode_index = Some(jitcode.index());
    // The control for `[fbw-mergepoint-in-helper]` in `mod.rs`: that probe answers
    // 0, and this line is what says the 0 describes the arm rather than a helper
    // walk that never happens.  Over the 491 synthetic fixtures this one fires
    // on `pickle_ctor_args` alone, for `__majit_wrap_load`.  One line per helper
    // jitcode, not per walk, so the pair stays readable next to the rest of what
    // this knob prints.
    if fbw_debug_abort_enabled() {
        thread_local! {
            static SEEN: std::cell::RefCell<std::collections::HashSet<usize>> =
                std::cell::RefCell::new(std::collections::HashSet::new());
        }
        if SEEN.with(|seen| seen.borrow_mut().insert(jitcode.index())) {
            eprintln!(
                "[fbw-helper-walk-enter] jitcode_index={} name={}",
                jitcode.index(),
                jitcode.name
            );
        }
    }
    // `transparent_helper_subwalk` is set by `run_sub_jitcode_walk` on the
    // sub-context it builds, so every descent into a canonical helper body
    // carries it — not just the ones entered from another sub-walk.
    let helper_frame = nested_helper_entry
        .map(|frame| InlineFrameGuard::enter(ctx.session, 0, false, vec![frame]));
    // Bracket the session-wide exception slot around the callee so a NULL
    // return only reads as a raise when the callee installed the exception.
    let exc_before_subwalk = ctx.last_exc_value();
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op.pc,
        &body,
        &[],
        &[],
        &[args_array],
        &[ConcreteValue::Null],
        &[],
    );
    // The helper's level ends with its walk.  A type-call `__init__` walked
    // next computes its own entry frame, which must see the caller as the
    // innermost level, not this finished helper.
    drop(helper_frame);
    ctx.fbw_mode = saved_fbw_mode;
    ctx.entry_py_pc = saved_entry;
    ctx.outer_resume_marker_jit_pc = saved_marker;
    ctx.outer_jitcode_index = saved_oji;
    ctx.frame_state.borrow_mut().outer_active_boxes = saved_active;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;
    if let Some(saved) = saved_vable {
        crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved);
    }

    let walk_result = match walk_result {
        Ok(outcome) => outcome,
        // `try_execute_residual_call_via_executor` declines an un-lowered
        // in-body helper (a tagged symbolic fnaddr) while inlining a
        // sub-jitcode, so the descent aborts instead of baking the hash as a
        // code address.  Propagating that abort from here strands the CALL:
        // this walk is the authoritative executor and the descent declined
        // *before* running the call, so the aborted trace resumes past a
        // Python instruction whose effect never happened — `d.popleft()`
        // returns its value and leaves the element in place.  Roll the partial
        // descent back and let the ordinary residual call run, the same way
        // the orthodox `w_list_append` descent does.  A descent that already
        // applied an effect cannot be rewound this way, so it keeps the abort.
        // The `unjournaled` requirement is an all-clear walk, not a
        // same-as-before walk: a walk that already carries an unjournaled
        // effect can no longer flush a `CloseLoop` end at all, so continuing
        // past the rolled back call strands the recorded tail of the loop
        // body when the close is declined and the FOR_ITER delivery refusal
        // drops the consumed iteration.  Propagating instead hands the abort
        // to the frame latch, which resumes forward at this opcode losing
        // nothing.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, symbolic })
            if rewind.nothing_applied() =>
        {
            if fbw_inline_diag_enabled() {
                eprintln!(
                    "[subwalk-abort] name={} pc={} abort_pc={} symbolic={:#x} \
                     disp=rollback effects={}/{} unjournaled={}/{}",
                    unsafe { pyre_interpreter::function_get_name(callable) },
                    op.pc,
                    pc,
                    symbolic as u64,
                    effects_before,
                    fbw_executed_effect_count(),
                    unjournaled_before,
                    fbw_has_unjournaled_effect(),
                );
            }
            cut_declined_subwalk(ctx, pre_fold_pos);
            return Ok(None);
        }
        Err(error) => {
            // The arm above withdrew its descent and returned the call to the
            // ordinary residual, so it is not an abort and leaves the body
            // eligible.  Reaching here does not: the walk stops, and the next
            // attempt would rebuild the same shape and stop again.
            if let DispatchError::OrthodoxSubWalkTraceUnsupported { pc, symbolic } = &error {
                if fbw_inline_diag_enabled() {
                    eprintln!(
                        "[subwalk-abort] name={} pc={} abort_pc={} symbolic={:#x} \
                         disp=propagate effects={}/{} unjournaled={}/{}",
                        unsafe { pyre_interpreter::function_get_name(callable) },
                        op.pc,
                        pc,
                        *symbolic as u64,
                        effects_before,
                        fbw_executed_effect_count(),
                        unjournaled_before,
                        fbw_has_unjournaled_effect(),
                    );
                }
                // The descent's `pc` is an offset into the callee's own
                // jitcode.  Crossing back into this frame re-raises the abort
                // where the enclosing walk owns the coordinates, and its
                // consumers read `stop_pc` as a position in THIS jitcode:
                // `latch_abort_blackhole` hands it to `setposition`, which
                // starts the blackhole's dispatch loop there.  Left unchanged,
                // a callee offset lands mid-instruction in a body of a
                // different length and the next byte read as an opcode is an
                // operand.  This frame stopped at its CALL, so that is the
                // position the abort carries out of here.
                let symbolic = *symbolic;
                return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                    pc: op.pc,
                    symbolic,
                });
            }
            return Err(error);
        }
    };
    // Look-inside of `str.startswith` / `str.endswith` is
    // `try_walker_inline_*`, not the residual prefix-match fold.  Recording
    // it on `str_startswith` / `str_endswith` made
    // `PYRE_FBW_NO_SPECIALIZE=str_startswith` report `fired` after the walk
    // had already replaced the call.  `spec_suppression` leaves these
    // descent entry points live even under `all`; the residual `spec_gate`
    // still covers leftover `jit_str_startswith` / `jit_str_endswith`.
    match promote_published_null_return_since(ctx, walk_result, op.pc, exc_before_subwalk)? {
        DispatchOutcome::SubReturn { result } => match finish_inline_callee_return(ctx, result) {
            Some(value) => {
                let concrete = match concrete_from_recorded_opref(ctx, value) {
                    ConcreteValue::Ref(p) if !p.is_null() => ConcreteValue::Ref(p),
                    other => walker_concrete_ref_object(ctx, value)
                        .map(ConcreteValue::Ref)
                        .unwrap_or(other),
                };
                if let Some(tp_init) = type_call_init {
                    return try_walker_inline_type_call_builtin_init(
                        ctx,
                        op,
                        code,
                        r_args,
                        dst,
                        tp_init,
                        value,
                        concrete,
                        &arg_concretes,
                        rewind,
                    );
                }
                write_ref_reg(ctx, op.pc, dst, value, concrete)?;
                Ok(Some((DispatchOutcome::Continue, op.next_pc)))
            }
            None => Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc }),
        },
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.set_last_exc_value(exc, exc_concrete);
                Ok(Some((DispatchOutcome::Continue, target)))
            } else {
                Ok(Some((
                    DispatchOutcome::SubRaise { exc, exc_concrete },
                    op.next_pc,
                )))
            }
        }
        DispatchOutcome::Terminate {
            finish_arg,
            finish_arg_type,
        } => Ok(Some((
            DispatchOutcome::Terminate {
                finish_arg,
                finish_arg_type,
            },
            op.next_pc,
        ))),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok(Some((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        ))),
        DispatchOutcome::CloseLoop { .. }
        | DispatchOutcome::CompileTracePending { .. }
        | DispatchOutcome::SubLoopCalleeCallAssembler { .. }
        | DispatchOutcome::SegmentTrace { .. } => {
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// Why `callable` cannot be descended with `wrapper_item_count` arguments,
/// or `None` if it can.  `__new__` allocates, so `__init__` must pass this
/// before that walk starts.
fn builtin_gateway_undescendable_reason(
    callable: pyre_object::PyObjectRef,
    wrapper_item_count: usize,
) -> Option<&'static str> {
    if !unsafe { pyre_interpreter::is_function_carrier(callable) } {
        return Some("not is_function_carrier");
    }
    let builtin_code =
        unsafe { pyre_interpreter::function_get_code(callable) as pyre_object::PyObjectRef };
    if builtin_code.is_null() || !unsafe { pyre_interpreter::is_builtin_code(builtin_code) } {
        return Some("not builtin_code");
    }
    let fnaddr = unsafe { pyre_interpreter::builtin_code_get(builtin_code) as usize };
    let Some(jitcode) = crate::state::bytecode_for_address(fnaddr) else {
        return Some("no jitcode for address");
    };
    if crate::jitcode_dispatch::sub_jitcode_body_by_index(jitcode.index()).is_none() {
        return Some("no sub jitcode body");
    }
    if crate::state::ensure_build_time_jitcode_at(jitcode.index()).is_none() {
        return Some("wrapper jitcode absent from the build-time table");
    }
    if !descent_unlowered_helper_scan_enabled() {
        return None;
    }
    // The walk of `__init__` starts after `__new__` allocated, so every
    // blocker in it is after an effect: an `__init__` whose blocker the scan
    // counts as effect-free would still abort past the allocation.
    let summary = descent_blocker_summary_for_entry_len(jitcode.index(), wrapper_item_count);
    if summary.body_not_walked {
        return Some("body holds bytes the scan cannot decode");
    }
    if summary
        .blocker_after_effect
        .or(summary.blocker_effect_free)
        .is_some()
    {
        log_descent_unlowered_helper_blockers(jitcode.index());
        return Some("un-lowered helper");
    }
    None
}

/// `typeobject.py descr_call`'s `__init__` half after a builtin `__new__`
/// already ran: walk the builtin `__init__` with the instance as `self` and
/// keep the instance as the CALL result.
///
/// Returning the instance without running `__init__` would be a wrong answer,
/// so a decline here aborts the walk.  The one exception is the `__init__`
/// walk declining a call before it ran it, with nothing applied since
/// `rewind` was taken: `__new__`'s unreferenced allocation is no effect, so
/// the whole CALL is cut back and runs as the ordinary residual.  The caller
/// checked [`builtin_gateway_undescendable_reason`] first.
#[allow(clippy::too_many_arguments)]
fn try_walker_inline_type_call_builtin_init<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    dst: usize,
    tp_init: pyre_object::PyObjectRef,
    instance: OpRef,
    instance_concrete: ConcreteValue,
    arg_concretes: &[ConcreteValue],
    rewind: SubWalkRewind,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let ConcreteValue::Ref(instance_obj) = instance_concrete else {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx, op.pc, None,
        ));
    };
    // `descr_call` runs `__init__` only when `__new__` returned an instance
    // of the called class.
    if let Some(w_type) = walker_concrete_ref_object(ctx, r_args[0])
        && !unsafe { pyre_interpreter::baseobjspace::isinstance_w(instance_obj, w_type) }
    {
        write_ref_reg(ctx, op.pc, dst, instance, instance_concrete)?;
        return Ok(Some((DispatchOutcome::Continue, op.next_pc)));
    }
    let builtin_code =
        unsafe { pyre_interpreter::function_get_code(tp_init) } as pyre_object::PyObjectRef;
    let fnaddr = unsafe { pyre_interpreter::builtin_code_get(builtin_code) as usize };
    let Some(jitcode) = crate::state::bytecode_for_address(fnaddr) else {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx, op.pc, None,
        ));
    };
    let Some(body) = crate::jitcode_dispatch::sub_jitcode_body_by_index(jitcode.index()) else {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx, op.pc, None,
        ));
    };
    if crate::state::ensure_build_time_jitcode_at(jitcode.index()).is_none() {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx, op.pc, None,
        ));
    }
    if fbw_inline_diag_enabled() {
        eprintln!(
            "[type-call-init] pc={} class={} init={}",
            op.pc,
            unsafe {
                walker_concrete_ref_object(ctx, r_args[0])
                    .map(|t| pyre_object::w_type_get_name(t))
                    .unwrap_or("")
            },
            unsafe { pyre_interpreter::function_get_name(tp_init) },
        );
    }

    let nested_helper = ctx.fbw_mode.inline_subwalk;
    // Same entry rule as `try_walker_inline_builtin_call`: a call from inside
    // a canonical helper body inherits that helper's resume point.
    let nested_helper_entry = if nested_helper && !ctx.fbw_mode.transparent_helper_subwalk {
        match compute_inline_helper_call_entry_frame(ctx, op.pc) {
            Ok(frame) => Some(frame),
            Err(_) => {
                return Err(super::fbw_state::fbw_decline_inline_callee(
                    ctx, op.pc, None,
                ));
            }
        }
    } else {
        None
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx, op.pc, None,
        ));
    }
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx, op.pc, None,
        ));
    }
    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker) = if nested_helper {
        (
            ctx.entry_py_pc(),
            0,
            ctx.outer_jitcode_index,
            ctx.outer_resume_marker_jit_pc,
        )
    } else {
        unsafe {
            let jc = &*sym.jitcode();
            let jc_index = jc.index as u32;
            let marker = jc.payload.resume_marker_for_jitcode_pc(op.pc);
            let mut py = jc
                .payload
                .forward_py_pc_for_jitcode_pc(op.pc)
                .unwrap_or_else(|| {
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op.pc)
                });
            if jc.payload.code_ptr.is_null() {
                (py, sym.valuestackdepth() as i64, jc_index, marker)
            } else {
                let codeobj = &*jc.payload.code_ptr;
                py = skip_python_trivia_forward(codeobj, py as usize) as u32;
                let depth = if jc.payload.depth_trivia_populated() {
                    jc.payload.depth_trivia_for_jitcode_pc(op.pc)
                } else {
                    crate::liveness::liveness_for(jc.payload.code_ptr)
                        .depth_at_py_pc()
                        .get(py as usize)
                        .copied()
                };
                let vsd = depth
                    .map(|d| (sym.nlocals() + d as usize) as i64)
                    .unwrap_or(sym.valuestackdepth() as i64);
                (py, vsd, jc_index, marker)
            }
        }
    };
    let call_site_active = if nested_helper_entry.is_some() {
        ctx.frame_state.borrow().outer_active_boxes.clone()
    } else {
        let call_site_word = call_site_marker
            .map(|marker| marker as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let vstack_boxes = ctx.frame_state.borrow().vstack_boxes.clone();
        let vstack = ctx.vstack_valid.then_some(vstack_boxes.as_slice());
        collect_outer_active_boxes(
            sym,
            ctx.trace_ctx,
            ctx.registers_i,
            ctx.registers_r,
            ctx.registers_f,
            outer_jitcode_index,
            false,
            call_site_word,
            op.pc as i32,
            OuterActiveBoxesEntryTwin::Plain,
            "builtin_wrapper_call_site",
            vstack,
            &[],
            None,
        )
    };

    let mut wrapper_items = Vec::with_capacity(1 + r_args.len().saturating_sub(2));
    let mut wrapper_item_concretes = Vec::with_capacity(1 + arg_concretes.len().saturating_sub(2));
    wrapper_items.push(instance);
    wrapper_item_concretes.push(instance_concrete);
    wrapper_items.extend_from_slice(&r_args[2..]);
    wrapper_item_concretes.extend_from_slice(&arg_concretes[2..]);
    for (&item, concrete) in wrapper_items.iter().zip(&wrapper_item_concretes) {
        if let ConcreteValue::Ref(value) = concrete
            && !value.is_null()
        {
            ctx.trace_ctx.try_set_opref_concrete(
                item,
                majit_ir::Value::Ref(majit_ir::GcRef(*value as usize)),
            );
        }
    }

    let array_descr = crate::state::pyobject_gcarray_descr();
    let len = ctx.trace_ctx.const_int(wrapper_items.len() as i64);
    let args_array =
        ctx.trace_ctx
            .record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr.clone());
    ctx.trace_ctx
        .heap_cache_mut()
        .new_array(args_array, len, true);
    for (index, &item) in wrapper_items.iter().enumerate() {
        let index = ctx.trace_ctx.const_int(index as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[args_array, index, item],
            array_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setarrayitem(args_array, index, array_descr.index(), item);
    }

    let saved_vable = if !nested_helper && sym.owns_virtualizable_shadow() {
        let saved = crate::trace_opcode::save_vable_resume_scalars(ctx.trace_ctx);
        let last_instr = call_site_py_pc as i64 - 1;
        let last_instr_op = ctx.trace_ctx.const_int(last_instr);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            last_instr_op,
            Value::Int(last_instr),
        );
        let vsd_op = ctx.trace_ctx.const_int(vsd_value);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "valuestackdepth",
            vsd_op,
            Value::Int(vsd_value),
        );
        Some(saved)
    } else {
        None
    };

    let saved_entry = ctx.entry_py_pc;
    let saved_marker = ctx.outer_resume_marker_jit_pc;
    let saved_oji = ctx.outer_jitcode_index;
    let saved_active = std::mem::take(&mut ctx.frame_state.borrow_mut().outer_active_boxes);
    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    let saved_fbw_mode = ctx.fbw_mode;
    // A nested helper's byte offset is not a coordinate in the Python
    // jitcode. Keep the outermost CALL so a portal entered from inside
    // the helper still names that call.
    if !(ctx.fbw_mode.inline_subwalk && matches!(saved_entry, EntryPyPc::Jit(_))) {
        ctx.entry_py_pc = EntryPyPc::Jit(op.pc);
    }
    ctx.outer_resume_marker_jit_pc = call_site_marker;
    ctx.outer_jitcode_index = outer_jitcode_index;
    ctx.frame_state.borrow_mut().outer_active_boxes = call_site_active;
    ctx.descr_refs = crate::jitcode_runtime::descr_ref_table();
    ctx.raw_descrs = RawDescrPool::Global;
    ctx.sub_jitcode_lookup = &GLOBAL_SUB_JITCODE_LOOKUP_FN;
    ctx.fbw_mode.inline_subwalk = true;
    ctx.fbw_mode.inline_caller_py_pc = Some(call_site_py_pc);
    ctx.fbw_mode.transparent_helper_jitcode_index = Some(jitcode.index());
    let helper_frame = nested_helper_entry
        .map(|frame| InlineFrameGuard::enter(ctx.session, 0, false, vec![frame]));
    let exc_before_subwalk = ctx.last_exc_value();
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op.pc,
        &body,
        &[],
        &[],
        &[args_array],
        &[ConcreteValue::Null],
        &[],
    );
    drop(helper_frame);
    ctx.fbw_mode = saved_fbw_mode;
    ctx.entry_py_pc = saved_entry;
    ctx.outer_resume_marker_jit_pc = saved_marker;
    ctx.outer_jitcode_index = saved_oji;
    ctx.frame_state.borrow_mut().outer_active_boxes = saved_active;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;
    if let Some(saved) = saved_vable {
        crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved);
    }

    let walk_result = match walk_result {
        Ok(outcome) => outcome,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { .. }) if rewind.nothing_applied() => {
            cut_declined_subwalk(ctx, rewind.pre_fold_pos);
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    match promote_published_null_return_since(ctx, walk_result, op.pc, exc_before_subwalk)? {
        DispatchOutcome::SubReturn { result: _ } => {
            // `descr_call` discards `__init__`'s None and returns the instance.
            write_ref_reg(ctx, op.pc, dst, instance, instance_concrete)?;
            Ok(Some((DispatchOutcome::Continue, op.next_pc)))
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.set_last_exc_value(exc, exc_concrete);
                Ok(Some((DispatchOutcome::Continue, target)))
            } else {
                Ok(Some((
                    DispatchOutcome::SubRaise { exc, exc_concrete },
                    op.next_pc,
                )))
            }
        }
        DispatchOutcome::Terminate {
            finish_arg,
            finish_arg_type,
        } => Ok(Some((
            DispatchOutcome::Terminate {
                finish_arg,
                finish_arg_type,
            },
            op.next_pc,
        ))),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok(Some((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        ))),
        DispatchOutcome::CloseLoop { .. }
        | DispatchOutcome::CompileTracePending { .. }
        | DispatchOutcome::SubLoopCalleeCallAssembler { .. }
        | DispatchOutcome::SegmentTrace { .. } => {
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// Read one of the callee function's `_immutable_fields_` slots live and pin
/// the value this inline baked, replacing the box so later reads fold.
///
/// `function.py` declares `code?` / `w_func_globals?` / `closure?[*]` /
/// `defs_w?[*]`; pyre's setters do not yet force the quasi-immutable
/// invalidation, so a `GuardValue` stands in for it.
fn walker_guard_function_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    callable: OpRef,
    descr: majit_ir::DescrRef,
    expected: i64,
) -> Result<(), DispatchError> {
    let field_op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, callable, descr);
    ctx.trace_ctx.try_set_opref_concrete(
        field_op,
        majit_ir::Value::Ref(majit_ir::GcRef(expected as usize)),
    );
    let expected_op = ctx.trace_ctx.const_ref(expected);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[field_op, expected_op], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(field_op, expected_op);
    Ok(())
}

fn immediate_inline_caller_py_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    call_jitcode_pc: usize,
) -> Option<u32> {
    // A canonical helper has no Python pc. Remapping its CALL through the
    // portal snapshot would stamp the surrounding user frame with an
    // unrelated offset. Inherit the Python caller's coordinate instead.
    if ctx.fbw_mode.transparent_helper_subwalk {
        // Nested helper entry stores the Python CALL that invoked the
        // helper on `InlineParentFrame.caller_py_pc`. That is the site
        // `_getframe(1)` should report, not the CALL that entered the
        // surrounding Python frame.
        if let Some(pc) = ctx
            .session
            .borrow()
            .framestack
            .last()
            .and_then(|frame| frame.parents.last().and_then(|parent| parent.caller_py_pc))
        {
            return Some(pc);
        }
        return ctx
            .fbw_mode
            .immediate_inline_caller_py_pc
            .or(ctx.fbw_mode.inline_caller_py_pc);
    }
    if let Some(consts) = ctx.inline_callee_consts {
        crate::py_coord::containing_py_pc_for_jitcode_pc_public(
            consts.jitcode_index,
            call_jitcode_pc as i32,
        )
        .map(|py| py as u32)
    } else {
        inline_caller_py_pc_from_snapshot(ctx, call_jitcode_pc)
    }
}

fn inline_caller_py_pc_from_snapshot<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    call_jitcode_pc: usize,
) -> Option<u32> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        return None;
    }
    let jc = unsafe { &*sym.jitcode() };
    Some(
        crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, call_jitcode_pc)
            as u32,
    )
}

/// Fill in the `name` / `obj` pair on an `AttributeError` raised out of an
/// inlined attribute hook, reproducing `enrich_attribute_error` (`error.rs`).
///
/// `StdObjSpace.getattr` (`objspace/std/objspace.py`) performs it on the
/// `__getattr__` fallback call, and pyre's `getattr_str` (`baseobjspace.rs`)
/// wraps the whole dispatch in it, so the `load_attr_fn` residual the hook
/// routes replace reached it either way.  Without this the compiled iterations
/// answer `e.name is None` where both the interpreter and a deopt answer the
/// attribute name.
///
/// The pair is written with `SetfieldGc` rather than by calling the enricher:
/// a call takes the exception as an argument and so forces the allocation the
/// optimizer had removed, the cost `record_inline_exception_context` records as
/// doubling the loops that catch without touching the object.  Both slots sit at
/// one offset for every kind (`W_BaseException` is a single flattened struct), so
/// the recording kind names the right bytes; only the descr identity the
/// optimizer aliases on is narrower, as in `record_prepend_application_traceback`.
fn record_inline_attribute_error_context<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    attr: &AttributeErrorInlineContext,
) -> Result<(), DispatchError> {
    use pyre_interpreter::baseobjspace::ExceptionAttrSlot as Slot;
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return Ok(());
    };
    if exc_ptr.is_null() || !unsafe { pyre_object::is_exception(exc_ptr) } {
        return Ok(());
    }
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc_ptr) };
    if kind != pyre_object::interp_exceptions::ExcKind::AttributeError {
        return Ok(());
    }
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc_ptr) };
    let current_name = unsafe { pyre_object::interp_exceptions::w_exception_get_name(exc_ptr) };
    let current_obj = unsafe { pyre_object::interp_exceptions::w_exception_get_attr_obj(exc_ptr) };
    let fills = current_name.is_null() && current_obj.is_null();
    // A `Const` exception box freezes the recording iteration's address
    // (`walker_record_guard_exception` pins every raise after the first one in a
    // walk), so a field op through it would reach a stale object on every later
    // iteration -- the shape `record_prepend_application_traceback` declines for
    // the same reason.  The concrete write below still runs: this walk is the
    // authoritative execution path.
    if !exc.is_none() && !exc.is_constant() {
        // Pin the branch `enrich_attribute_error` takes.  Both reads answer
        // statically for the virtual an inlined `raise` builds -- the shape every
        // hook constructing its own `AttributeError` produces -- so the guards
        // cost nothing there and are real only for an exception that reached the
        // walk from outside.
        let read_name = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            exc,
            crate::descr::w_exception_attr_slot_descr_for(kind, Slot::Name, user),
        );
        ctx.trace_ctx.set_opref_concrete(
            read_name,
            majit_ir::Value::Ref(majit_ir::GcRef(current_name as usize)),
        );
        let read_obj = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            exc,
            crate::descr::w_exception_attr_slot_descr_for(kind, Slot::AttrObj, user),
        );
        ctx.trace_ctx.set_opref_concrete(
            read_obj,
            majit_ir::Value::Ref(majit_ir::GcRef(current_obj as usize)),
        );
        for (read, is_null) in [
            (read_name, current_name.is_null()),
            (read_obj, current_obj.is_null()),
        ] {
            let guard = if is_null {
                OpCode::GuardIsnull
            } else {
                OpCode::GuardNonnull
            };
            walker_emit_fold_guard_with_snapshot(ctx, op_pc, guard, &[read])?;
        }
        if fills {
            for (slot, value) in [(Slot::Name, attr.name), (Slot::AttrObj, attr.obj)] {
                let descr = crate::descr::w_exception_attr_slot_descr_for(kind, slot, user);
                let descr_index = descr.index();
                ctx.trace_ctx
                    .record_op_with_descr(OpCode::SetfieldGc, &[exc, value], descr);
                ctx.trace_ctx
                    .heapcache_setfield_cached(exc, descr_index, value);
            }
        }
    }
    if fills {
        // The setters carry the host-side remembered-set barrier themselves;
        // compiled `SetfieldGc` reference stores get `CondCallGcWb` from
        // majit-gc's rewrite pass.
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_name(exc_ptr, attr.name_concrete);
            pyre_object::interp_exceptions::w_exception_set_attr_obj(exc_ptr, attr.obj_concrete);
        }
    }
    Ok(())
}

/// Shared post-resolution half of the FBW inline lever. Ordinary Python calls
/// resolve their callee from the CALL operand; builtin-dispatch specializers
/// resolve an app-level descriptor first and enter here with that function as
/// the callee while independently pinning the original builtin callable.
pub(crate) fn try_walker_inline_resolved_user_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    callable: pyre_object::PyObjectRef,
    callable_guard_op: OpRef,
    callable_guard_value: pyre_object::PyObjectRef,
    arg_concretes: Vec<ConcreteValue>,
    callee_args: Vec<OpRef>,
    callee_arg_concretes: Vec<ConcreteValue>,
    method_form: bool,
    bound_method: Option<BoundMethodInline>,
    w_code: *const (),
    nparams: usize,
    has_closure: bool,
    exception_receiver_guard: Option<ExceptionInlineReceiverGuard>,
    arg_class_guard: Option<ArgClassGuard>,
    entry_is_call_boundary: bool,
    require_str_result: bool,
    constructor_result: Option<(OpRef, ConcreteValue)>,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        callable,
        callable_guard_op,
        callable_guard_value,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        method_form,
        bound_method,
        w_code,
        nparams,
        has_closure,
        exception_receiver_guard,
        arg_class_guard,
        entry_is_call_boundary,
        require_str_result,
        constructor_result,
        None,
        false,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
    )
}

/// The resolved-call inliner with an optional receiver for a nested call's
/// result.  RPython's `MIFrame.finishframe` hands a returned box to the
/// previous frame (`pyjitpl.py`) before that caller continues its
/// current operation.  `intermediate_result` represents that hand-off for a
/// user call nested inside a specialized builtin: it deliberately leaves the
/// outer residual's destination untouched, so guards still snapshot the
/// caller at the builtin-call boundary.
/// Report which decline in [`try_walker_inline_resolved_user_call_inner`] a
/// call hit.  The function has three dozen of them and reaches them from every
/// call shape, so a caller that only learns "declined" has to guess; the
/// `[binop-inline-decline]` and `[type-call-decline]` lines above name the
/// call, and this names the test inside it that refused.
#[inline]
fn resolved_inline_decline<T>(op_pc: usize, line: u32) -> Result<Option<T>, DispatchError> {
    if fbw_inline_diag_enabled() {
        eprintln!("[resolved-inline-decline] pc={op_pc} inline_call.rs:{line}");
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn try_walker_inline_resolved_user_call_inner<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    callable: pyre_object::PyObjectRef,
    callable_guard_op: OpRef,
    callable_guard_value: pyre_object::PyObjectRef,
    arg_concretes: Vec<ConcreteValue>,
    mut callee_args: Vec<OpRef>,
    mut callee_arg_concretes: Vec<ConcreteValue>,
    method_form: bool,
    bound_method: Option<BoundMethodInline>,
    w_code: *const (),
    nparams: usize,
    has_closure: bool,
    exception_receiver_guard: Option<ExceptionInlineReceiverGuard>,
    arg_class_guard: Option<ArgClassGuard>,
    entry_is_call_boundary: bool,
    require_str_result: bool,
    constructor_result: Option<(OpRef, ConcreteValue)>,
    intermediate_result: Option<&mut Option<(OpRef, ConcreteValue)>>,
    require_exact_int_result: bool,
    instance_next_foriter_green_key: Option<u64>,
    attribute_error_context: Option<AttributeErrorInlineContext>,
    // The operator this call sits under, when its result is not the callee's
    // return value.  It decides two things together: the checks emitted over
    // the returned box, and the resume level pushed under the callee so a
    // guard inside the BODY runs those same checks on the way out
    // (`crate::operator_continuation`).
    operator_tail: Option<crate::operator_continuation::OperatorTail>,
    // A `f(*args)` star tuple the positional parameters are read from; its
    // placeholders in `callee_args` are replaced after the eligibility checks.
    star_args: Option<StarArgsInline>,
    // The keywords a `**kwargs` callee collects into its mapping.
    mut varkw_extra: Vec<VarkwExtra>,
    // A `f(**mapping)` dict the named parameters are read from; its
    // placeholders in `callee_args` are replaced after the eligibility checks.
    star_kwargs: Option<StarKwargsInline>,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // Argument boxes and surplus keyword values stay reachable across every
    // allocation this inline records (`wrapint`, the kwargs dict, a default
    // box). The shadow slots, not the `Vec` copies, are what a collection
    // forwards.
    let _arg_roots = pyre_object::gc_roots::push_roots();
    let mut arg_root_slots = Vec::with_capacity(callee_arg_concretes.len());
    {
        let mut live = Vec::new();
        for concrete in callee_arg_concretes.iter().copied() {
            if let ConcreteValue::Ref(obj) = concrete
                && !obj.is_null()
            {
                arg_root_slots.push(Some(live.len()));
                live.push(obj);
            } else {
                arg_root_slots.push(None);
            }
        }
        let base = _arg_roots.pin_roots(&live);
        for slot in &mut arg_root_slots {
            if let Some(offset) = *slot {
                *slot = Some(base + offset);
            }
        }
        refresh_rooted_refs(&mut callee_arg_concretes, &arg_root_slots);
    }
    let mut varkw_root_slots = Vec::with_capacity(varkw_extra.len());
    {
        let mut live = Vec::new();
        for extra in &varkw_extra {
            if let ConcreteValue::Ref(obj) = extra.concrete
                && !obj.is_null()
            {
                varkw_root_slots.push(Some(live.len()));
                live.push(obj);
            } else {
                varkw_root_slots.push(None);
            }
        }
        if !live.is_empty() {
            let base = _arg_roots.pin_roots(&live);
            for slot in &mut varkw_root_slots {
                if let Some(offset) = *slot {
                    *slot = Some(base + offset);
                }
            }
            refresh_rooted_refs(&mut callee_arg_concretes, &arg_root_slots);
            refresh_varkw_concretes(&mut varkw_extra, &varkw_root_slots);
        }
    }
    let is_being_profiled = ctx.session.borrow().is_being_profiled;
    // `_compute_flatcall` (`pycode.py`) leaves `fast_natural_arity`
    // HOPELESS for a `*args` / `**kwargs` / keyword-only callee.  The general
    // `funcrun` path still traces through `_match_signature`, which writes a
    // surplus tuple for `*args`, fills each keyword-only local from
    // `w_kw_defs`, and binds a fresh mapping of the unmatched keywords to
    // `**kwargs` (`_match_signature`); seed all three of those extra locals
    // here.
    let positional_only = fbw_callee_scope_is_positional_only(w_code);
    let vararg_slot = fbw_callee_vararg_slot(w_code);
    let varkw_slot = fbw_callee_varkw_slot(w_code);
    let Some(kwonly_count) = fbw_callee_kwonly_count(w_code) else {
        return resolved_inline_decline(op.pc, line!());
    };
    if !positional_only && vararg_slot.is_none() && kwonly_count == 0 && varkw_slot.is_none() {
        return resolved_inline_decline(op.pc, line!());
    }
    // Each collected keyword is stored under its `kwnames` constant, which the
    // emit bakes, so it has to be an object a collection cannot move; its
    // value is read off the live register at the emit.
    if !varkw_extra.is_empty() && varkw_slot.is_none() {
        return resolved_inline_decline(op.pc, line!());
    }
    for extra in &varkw_extra {
        let ConcreteValue::Ref(value) = extra.concrete else {
            return resolved_inline_decline(op.pc, line!());
        };
        if value.is_null() || majit_gc::can_move(majit_ir::GcRef(extra.name as usize)) {
            return resolved_inline_decline(op.pc, line!());
        }
    }
    // Not every caller pins the callee function itself.  A specializer that
    // resolves an app-level method behind a builtin — `str(e)` reaching an
    // exception subclass's `__str__` — passes the CALL's own operand, which is
    // the `str` builtin, while `callable` is the resolved `Function`.  Reading
    // `Function.code` off that operand is a type-confused load: it returns
    // whatever sits at the same offset in a `PyCFunction`, so the guard
    // compares a value that is not `code` and fails every iteration (99480
    // failures and 497 bridges on `synth/exception_subclass_attrs`, a 31x
    // slowdown).  Guard the fields only when the pinned object really is the
    // function whose code this inline resolved.
    let pinned_object_is_the_callee = unsafe {
        (*callable_guard_value).ob_type as *const () as usize
            == &pyre_interpreter::FUNCTION_TYPE as *const _ as usize
            && pyre_interpreter::function_get_code(callable_guard_value) as usize
                == w_code as pyre_object::PyObjectRef as usize
    };
    // A trace-constant callable is excluded for a second reason: the field
    // reads would dereference a baked `ConstPtr`, and loading through one
    // dangles as soon as a minor collection moves the object
    // (`synth/inline_subwalk_property_mutates` — a property getter that
    // allocates on every iteration — segfaulted on cranelift under CI's macOS
    // runner with the reads in place).  Comparing against such a constant is
    // fine; that is why the `code?` marker below covers this arm instead.
    let guards_the_callee_function =
        !callable_guard_op.is_constant() && pinned_object_is_the_callee;
    // The `__call__` arm pins the instance it dispatched on by class and type
    // version, and that pin selects the resolved `__call__` the way
    // `space.lookup` does.  The other user of this guard, the `str(e)`
    // specializer, pins a different operand than its receiver.
    let receiver_pinned_by_class =
        exception_receiver_guard.is_some_and(|(receiver, ..)| receiver == callable_guard_op);
    // `Function.funccall_valuestack` fills every parameter the call left
    // unbound from `defs_w` before entering the frame
    // (`function.py:188-193,217-231`); `Arguments.parse` reaches the same frame
    // shape for a keyword call.  Mirror that shape here.  Placeholder boxes are
    // replaced by live guarded tuple-item reads after all non-emitting
    // eligibility checks.
    //
    // A positional call leaves the unbound parameters as a missing tail, but a
    // keyword one can leave a hole anywhere — `f(i, c=7)` on `def f(a, b=3,
    // c=5)` binds slots 0 and 2 and leaves 1 — so the seeding vectors carry
    // `OpRef::NONE` for a hole and the set is collected rather than assumed
    // contiguous.  Only the seeding writes that sentinel, so it cannot collide
    // with a genuinely passed argument.
    let missing: Vec<usize> = (0..nparams)
        .filter(|&i| callee_args.get(i).is_none_or(|arg| *arg == OpRef::NONE))
        .collect();
    let positional_defaults = if missing.is_empty() {
        None
    } else {
        // The defaults are read live off `callable_guard_op` below, while the
        // tuple they are checked against comes from `callable`.  Those are the
        // same object for an ordinary call, but not for a specializer that
        // resolves an app-level method behind a builtin: `str(e)` reaching an
        // exception subclass's `__str__` passes the CALL's own operand — the
        // `str` builtin — as the guard operand while `callable` is the resolved
        // `Function`.  Reading `Function.defs_w` off that operand loads
        // whatever sits at the same offset in another type, so decline the
        // inline rather than emit the load.  Measured on a `__str__(self,
        // value=7)` of an `Exception` subclass: 122 bridges and 24480 guard
        // failures over 50000 iterations, against 0 and 1 for the same loop
        // with no defaulted parameter.
        if callable_guard_value != callable {
            return Ok(None);
        }
        let Some(defaults) =
            (unsafe { positional_defaults_for_inline(&_arg_roots, callable, &missing, nparams) })
        else {
            return resolved_inline_decline(op.pc, line!());
        };
        callee_args.resize(nparams, OpRef::NONE);
        callee_arg_concretes.resize(nparams, ConcreteValue::Null);
        arg_root_slots.resize(nparams, None);
        for &(param_index, _, value) in &defaults.values {
            adopt_rooted_ref(
                &_arg_roots,
                &mut callee_arg_concretes,
                &mut arg_root_slots,
                &mut varkw_extra,
                &varkw_root_slots,
                param_index,
                value,
            );
        }
        Some(defaults)
    };
    // `_match_signature` fills every keyword-only local from `w_kw_defs`
    // (`_match_signature`) before it writes the vararg slot.  Resolved here
    // so a missing default still declines without having recorded anything;
    // the live reads are emitted alongside the positional defaults below.
    let kwonly_defaults = if kwonly_count == 0 {
        None
    } else {
        // The dict is read off `callable_guard_op` while the resolve below
        // reads it off `callable`, so unless the two name the same object the
        // walk would stamp one function's defaults onto the other's live
        // reads.  `guards_the_callee_function` is not that test: it proves the
        // pinned object is *a* function carrying this code, which two closures
        // minted from one `def` both are.  It is also not the test to gate on
        // — a module-level callee reaches here with its function baked, which
        // clears the flag and is the ordinary shape, not a corner.
        if callable_guard_value != callable {
            return resolved_inline_decline(op.pc, line!());
        }
        // Seeding from baked cells stands on a `GuardValue` that `w_kw_defs`
        // still holds the mapping those cells belong to, so the callee has to
        // be a function that outlived the recording.  A callee this trace
        // allocated itself is a fresh function every iteration, and the walker
        // reads the slot off its raw address: the allocator hands the next
        // incarnation the same address, so the peeled body's load disagrees
        // with what the heap cache holds for that address and trips
        // `_opimpl_getfield_gc_any_pureornot`'s sanity check.
        if ctx.trace_ctx.heap_cache().is_unescaped(callable_guard_op) {
            return resolved_inline_decline(op.pc, line!());
        }
        let resolved: Option<KwonlyDefaultsInline> = crate::jitcode_dispatch::diag::spec_gate(
            crate::jitcode_dispatch::diag::SpecFold::KwonlyDefaultsInline,
            || {
                Ok::<_, DispatchError>(unsafe {
                    kwonly_defaults_for_inline(callable, w_code, nparams, kwonly_count)
                })
            },
        )?;
        let Some(resolved) = resolved else {
            return resolved_inline_decline(op.pc, line!());
        };
        Some(resolved)
    };
    // The surplus positional arguments `_match_signature` packs into the
    // vararg.  Collected here and emitted alongside the defaults below, so
    // every remaining eligibility check still declines without having recorded
    // the tuple build; the placeholder box mirrors how a default is carried.
    let vararg_surplus = if vararg_slot.is_some() {
        // `_match_signature` (`argument.py:194-201`) prepends a receiver to
        // `args_w` — and so to the vararg tuple — when the callee has no
        // positional parameter to hold it.  A bound method reaches here with
        // `callee_args[0]` still the placeholder the resolved half replaces
        // with `GetfieldGcR(Method.w_self)`, which is emitted after the split
        // below; folding the placeholder into the tuple would put the Method
        // object where the receiver belongs.  Decline that one shape.
        if bound_method.is_some() && nparams == 0 {
            return resolved_inline_decline(op.pc, line!());
        }
        if callee_arg_concretes.len() != callee_args.len() || callee_args.len() < nparams {
            return resolved_inline_decline(op.pc, line!());
        }
        let surplus_ops: Vec<OpRef> = callee_args[nparams..].to_vec();
        let mut surplus_concretes = Vec::with_capacity(surplus_ops.len());
        for concrete in &callee_arg_concretes[nparams..] {
            let ConcreteValue::Ref(obj) = *concrete else {
                return resolved_inline_decline(op.pc, line!());
            };
            if obj.is_null() {
                return resolved_inline_decline(op.pc, line!());
            }
            surplus_concretes.push(obj);
        }
        // The constructor the interpreter's own `pack_varargs` reaches, so the
        // walk binds the object the residual would have: a new allocation with
        // no heap mutation for a surplus — the one `emit_object_tuple_inline`
        // reproduces below — and the shared `()` for none.
        let concrete = pyre_object::w_tuple_new_array_backed(surplus_concretes);
        if concrete.is_null() {
            return resolved_inline_decline(op.pc, line!());
        }
        // A zero-surplus vararg is baked rather than built (see the emit
        // below).  That `ConstPtr` lands in the walker's `registers_r`, which
        // `InlineRegisterBankGuard` (`miframe_registers`) forwards, and then
        // in the recorded op / gcref table.  Movability does not decide it.
        callee_args.truncate(nparams);
        callee_arg_concretes.truncate(nparams);
        // The surplus elements' root slots must leave with them. A later
        // refresh zips this slice with the concretes, and the vararg index
        // is about to be reused for the tuple.
        arg_root_slots.truncate(nparams);
        Some((surplus_ops, concrete))
    } else {
        None
    };
    // Frame order is positional, then keyword-only, then the vararg
    // (`_match_signature`), so the placeholders are appended in that order and
    // the tuple's own slot moves out past the keyword-only block.
    if let Some(resolved) = kwonly_defaults.as_ref() {
        for kwonly in &resolved.values {
            let index = callee_args.len();
            callee_args.push(OpRef::NONE);
            callee_arg_concretes.push(ConcreteValue::Null);
            arg_root_slots.push(None);
            adopt_rooted_ref(
                &_arg_roots,
                &mut callee_arg_concretes,
                &mut arg_root_slots,
                &mut varkw_extra,
                &varkw_root_slots,
                index,
                kwonly.value,
            );
        }
    }
    if let Some((_, concrete)) = vararg_surplus.as_ref() {
        let index = callee_args.len();
        callee_args.push(OpRef::NONE);
        callee_arg_concretes.push(ConcreteValue::Null);
        arg_root_slots.push(None);
        adopt_rooted_ref(
            &_arg_roots,
            &mut callee_arg_concretes,
            &mut arg_root_slots,
            &mut varkw_extra,
            &varkw_root_slots,
            index,
            *concrete,
        );
    }
    // The `**kwargs` mapping is built at the emit, which fills both halves.
    if varkw_slot.is_some() {
        callee_args.push(OpRef::NONE);
        callee_arg_concretes.push(ConcreteValue::Null);
        arg_root_slots.push(None);
    }
    let vararg_index = nparams + kwonly_count;
    let varkw_index = vararg_index + usize::from(vararg_slot.is_some());
    let seeded_locals = varkw_index + usize::from(varkw_slot.is_some());
    // Does any incoming binding land a value the callee's register banks can
    // hold unboxed?  Only the `is`-against-None scan below consults this; see
    // its hazard-2 arm for why an int-specialized tested local is unsafe to
    // inline.  A default is pushed above as a raw `Ref` without going through
    // `ConcreteValue::from_pyobj`, so re-classify here rather than matching the
    // variant alone — `def _read_from_buffer(self, size=-1)` reaches the scan
    // with `size` as `Ref(<int object>)` and is exactly the shape that
    // miscompiled.
    let binding_is_unboxed = |concrete: ConcreteValue| {
        let classified = match concrete {
            ConcreteValue::Ref(obj) => ConcreteValue::from_pyobj(obj),
            other => other,
        };
        matches!(
            classified,
            ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_)
        )
    };
    let callee_binds_an_unboxed_local = callee_arg_concretes
        .iter()
        .copied()
        .any(&binding_is_unboxed);
    // Vararg/over-arity calls still use the ordinary residual path. A closure
    // is admissible with or without cellvars of its own: the existing cell
    // objects are threaded into this callee's own frame and each pure cellvar
    // takes a freshly emitted cell, exactly as
    // PyFrame::finish_for_call_with_globals_obj does.
    if callee_args.len() != seeded_locals {
        return resolved_inline_decline(op.pc, line!());
    }
    let raw_callee_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_callee_code.is_null() {
        return resolved_inline_decline(op.pc, line!());
    }
    let callee_code = unsafe { &*raw_callee_code };
    let mut concrete_freevar_cells = Vec::new();
    let concrete_closure = if has_closure {
        let closure = unsafe { pyre_interpreter::function_get_closure(callable) };
        if closure.is_null()
            || !unsafe { pyre_object::is_tuple(closure) }
            || unsafe { pyre_object::w_tuple_len(closure) } != callee_code.freevars.len()
        {
            return resolved_inline_decline(op.pc, line!());
        }
        for i in 0..callee_code.freevars.len() {
            let Some(cell) = (unsafe { pyre_object::w_tuple_getitem(closure, i as i64) }) else {
                return resolved_inline_decline(op.pc, line!());
            };
            concrete_freevar_cells.push(cell);
        }
        closure
    } else {
        if !callee_code.freevars.is_empty() {
            return resolved_inline_decline(op.pc, line!());
        }
        pyre_object::PY_NULL
    };
    // `pyjitpl.py`: consult `can_inline_callable` first, then bound
    // recursive inlining at `max_unroll_recursion`.  Reaching the bound marks
    // this callee's entry greenkey `dont_trace_here`; that state transition is
    // essential for branching recursion.  Once one arm reaches the bound,
    // later sibling calls in the same trace go straight to CALL_ASSEMBLER
    // instead of starting a fresh unroll from their now-shallower framestack.
    let callee_code_key = w_code as pyre_object::PyObjectRef as usize;
    let callee_green_key = crate::driver::make_green_key_typed(w_code, 0, is_being_profiled);
    if let Some((driver, _)) = crate::driver::try_driver_pair()
        && !driver
            .meta_interp_mut()
            .warm_state_for_driver(crate::state::PyreJitState::PYPYJIT_JD_INDEX)
            .can_inline_callable_for_key(&callee_green_key)
    {
        return resolved_inline_decline(op.pc, line!());
    }
    // `_opimpl_recursive_call` (`pyjitpl.py`) reads
    // `memory_manager.max_unroll_recursion` at this decision. Do not bake the
    // RPython default here: `pypyjit.set_param(max_unroll_recursion=...)`
    // mutates the live warmstate and must steer the production FBW path too.
    // A skeleton walk has no installed driver, so only that diagnostic path
    // falls back to the upstream default.
    let max_unroll_recursion = crate::driver::try_driver_pair()
        .map(|(driver, _)| driver.meta_interp().memory_manager.max_unroll_recursion() as usize)
        .unwrap_or(FBW_DEFAULT_MAX_INLINE_RECURSION);
    let inline_recursion_count = fbw_inline_recursion_count(ctx, callee_code_key);
    let recursive_portal_present = fbw_recursive_portal_present(ctx, callee_code_key);
    if inline_recursion_count >= max_unroll_recursion {
        if let Some((driver, _)) = crate::driver::try_driver_pair() {
            driver
                .meta_interp_mut()
                .warm_state_for_driver(crate::state::PyreJitState::PYPYJIT_JD_INDEX)
                .disable_noninlinable_function_for_key(&callee_green_key);
        }
        return resolved_inline_decline(op.pc, line!());
    }
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return resolved_inline_decline(op.pc, line!());
    };
    if nparams > body.num_regs_r {
        return resolved_inline_decline(op.pc, line!());
    }
    // Inlining a callee whose body carries an `abort_permanent` marker walks
    // the sub-walk straight into it.  That surfaces as
    // `TraceAction::AbortPermanent`, which stamps `DONT_TRACE_HERE` on the
    // CALLER loop's green key — so one unported opcode anywhere in a callee
    // permanently un-JITs the loop that calls it.  Upstream keeps this
    // decision static and on the callee: `codewriter/policy.py look_inside_graph`
    // `look_inside_graph` reads whole-graph properties before tracing, and its
    // own comment (:78-79) spells out the consequence of a "no" — "the call
    // will be turned into a residual call".  Answer the same way.
    //
    // `loop_inlines_abort_permanent_callee` (`trace.rs`) already screens this
    // up front, but only for callees it can resolve statically out of globals
    // and frame slots; a bound method, a container element or a call result
    // reaches here unscreened.  At this point the callee is concrete, so the
    // screen is exact.
    //
    // Whole-body, like upstream's whole-graph test: a marker on a path this
    // trace happens not to take still costs only the inline if we decline,
    // where walking into it costs the whole loop, permanently.
    //
    // Decided once per callee and kept on its jitcode payload rather than
    // re-scanned here on every call site that reaches this recipe.  `None`
    // means no installed body or descr pool, which the pool fetch immediately
    // below declines on regardless.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return resolved_inline_decline(op.pc, line!());
    };
    if body_facts.has_abort_permanent {
        return resolved_inline_decline(op.pc, line!());
    }
    // `pyframe.py execute_frame` brackets the callee's eval loop with
    // `ec.call_trace(self)` and `ec.return_trace(self, w_exitvalue)`, and
    // `executioncontext.py leave` adds `_trace(frame, 'leaveframe', ...)` when
    // a profiler is installed.  `walker_ec_enter` / `walker_ec_leave` port the
    // frame-chain half of that bracket and nothing else, and the walker has no
    // route to record `_trace` — it calls back into app-level Python with the
    // callee frame as an argument.  So while a hook is installed there is no
    // shape of this inline that can report what the callee owes, and the
    // answer is the one `codewriter/policy.py look_inside_graph` gives for a
    // graph it will not look inside: decline, and let the residual call reach
    // the interpreter's own `execute_frame`, which runs both.
    //
    if super::ec_hook_installed() {
        return resolved_inline_decline(op.pc, line!());
    }
    // The callee body resolves its `d`/`j` descr operands through its OWN
    // per-fn pool, not the caller's.  Without this the sub-walk reads the
    // wrong descr at the first `getfield_vable_r` / `residual_call`
    // (`VableArrayDescrMalformed` / `ResidualCallDescrNotCallDescr`).
    let Some((callee_descr_refs, callee_perfn_descrs, callee_lookup)) =
        crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return resolved_inline_decline(op.pc, line!());
    };
    let args_all_builtin_integer = callee_arg_concretes.iter().all(|concrete| match concrete {
        ConcreteValue::Int(_) | ConcreteValue::Bool(_) => true,
        ConcreteValue::Ref(obj) if !obj.is_null() => unsafe { pyre_object::is_int_or_long(*obj) },
        ConcreteValue::Float(_) | ConcreteValue::Ref(_) | ConcreteValue::Null => false,
    });
    // Keep exact-integer arithmetic callees as one residual call when tracing
    // a guard-origin bridge.  Re-inlining their BinaryOp body would create a
    // second virtual frame whose operand stack is not a red bridge input; an
    // overflow path can then compile NULL vable stack slots into the bridge.
    // The primary loop still inlines the callee, and non-integer/user-
    // overridable calls continue through the ordinary inline/abort policy.
    //
    // A plain ROOT bridge walk — no carrier resume, not an inline sub-walk,
    // an empty framestack, and
    // a live root portal — is uniform with a primary trace, so its second
    // virtual frame is seeded and snapshot-covered exactly as the loop's is.
    // There the decline is lifted: the call falls through to the self-recursive
    // unroll gate and multiframe seed as if walked from a primary trace.
    // True once this attempt takes the root-bridge admission for a
    // self-recursive callee.  The admitted top-level inline's
    // body sub-walk reaches its own recursive CALL as a nested residual, which
    // `fbw_abort_nested_unjournaled_residual` declines on the self-recursive
    // hazard arm — an abort storm that folds the whole guard bridge back to
    // residual.  The `CALL_ASSEMBLER` self-recursion fold already exempts that
    // decline via `SELFREC_CA_FOLD_ACTIVE`; the same exemption applies to this
    // admitted inline, whose recursive residual runs concretely at the
    // pre-execute site (executed, so no replay double-apply).
    let mut bridge_rec_root_selfrec = false;
    if ctx.trace_ctx.is_bridge_trace
        && args_all_builtin_integer
        && fbw_callee_body_has_binary_op_residual(body.code, callee_descr_refs)
    {
        // `_opimpl_recursive_call` runs in the one `MetaInterp.interpret`
        // loop after `perform_call` pushes each MIFrame; it has no separate
        // "root bridge may inline, nested bridge sub-walk may not" gate.  FBW
        // now carries the same frame chain explicitly: `InlineFrameGuard`
        // keeps this callee and its `parent_frame` live while the nested walk
        // runs, and multiframe guard snapshots encode that chain.  Refusing a
        // nested root-bridge call here dropped the parent's continuation from
        // the compiled shape and capped fib one unroll short of PyPy.
        bridge_rec_root_selfrec = unsafe {
            let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject;
            !raw.is_null() && pyre_interpreter::code_is_self_recursive(&*raw)
        };
    }
    // A callee `fbw_abort_nested_unjournaled_residual` already named on its
    // hazard arm residualizes from here on.  The hazard is a static property of
    // the callee's `CodeObject` (self-recursive), so re-inlining
    // it rebuilds the identical framestack and reaches the identical abort —
    // the enclosing loop is retired for a decline that belongs to the callee.
    // `WarmEnterState::disable_noninlinable_function_for_key` carries the
    // general answer: its flag means "do not inline calls to this function",
    // and the enclosing loop is left free to retrace.
    //
    // `bridge_rec_root_selfrec` is exempt: that admission carries its own
    // `SELFREC_CA_FOLD_ACTIVE` exemption from
    // `fbw_abort_nested_unjournaled_residual`, so its recursive residual is
    // not what named the callee here.  The warmstate verdict is deferred to
    // this same point so its typed lookup cannot hide the exemption.
    if !bridge_rec_root_selfrec && fbw_hazardous_inline_denied(callee_code_key) {
        return resolved_inline_decline(op.pc, line!());
    }
    // Inlining a self-recursive callee into a foreign caller (the
    // module `for` around `walk(...)`) starts a sub-walk whose
    // nested recursive CALL residualizes and then hits
    // `fbw_abort_nested_unjournaled_residual`'s self-recursive
    // hazard, aborting the enclosing loop
    // (`selfrec_bridge_nontail_promote`).  Same-function unroll
    // (`fib`) and the root-bridge admission keep the inline.
    if !bridge_rec_root_selfrec && !recursive_portal_present {
        let raw = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject
        };
        if !raw.is_null() && unsafe { pyre_interpreter::code_is_self_recursive(&*raw) } {
            let root_code = {
                let session = ctx.session.borrow();
                let frame = session.recording_frame_ptr as *const pyre_interpreter::PyFrame;
                if frame.is_null() {
                    0
                } else {
                    unsafe { (*frame).pycode as usize }
                }
            };
            if root_code != 0 && root_code != callee_code_key {
                return resolved_inline_decline(op.pc, line!());
            }
        }
    }
    // An unseeded inline sub-walk inside a FOR_ITER body resumes a guard at the
    // caller's CALL boundary and replays the whole callee, so a live-heap write
    // would execute twice.  A seeded callee frame answers that hazard exactly:
    // the guard carries the callee's own resume coordinate, matching the
    // per-call MIFrame shape of `MetaInterp.perform_call`.  The Dirty admission
    // therefore asks for that resume shape directly.  Of the three terms that
    // once stood beside it, two are gone -- the callee's own exception table
    // and the depth cap, both vestigial -- and the third, a constant callable,
    // is a profitability screen rather than part of this answer
    // (`seeded_callee_resume` below states what it measures).  The keyed
    // instance-`__next__` route is excluded because it already resumes keyed
    // guards through the seeded multi-frame snapshot.
    //
    // A body whose only unproven ops are Python-level CALL residuals is
    // admitted too: this same gate re-runs for each callee the lever resolves
    // below it, and one it cannot inline aborts before executing
    // (`fbw_abort_nested_unjournaled_residual`) and denies this callee, so a
    // deferred body commits nothing either.  Without that the whole nest
    // declines — `helper(i)` calling `add(i, 1, 2)` residualizes both calls
    // per iteration, though each body on its own is pure arithmetic.
    // The inlined callee body is entered at pc=0 with the fast-path
    // register convention `registers_r[0..nparams] = positional args` —
    // the same seeding `dispatch_inline_call_dr_kind` uses for `n_*`
    // inline calls and the retired `can_skip_traced_callee_frame` branch used
    // (`sym.registers_r = args.to_vec()`). This only holds for a callee
    // that reads its params straight from `r0`/`r1` (ref_copy +
    // residual_call args).  A callee that materializes a frame — any
    // `*_vable_*` op, emitted when a local must survive a sub-call —
    // reads from the unseeded frame box; inlining it would abort the
    // *whole* enclosing trace with `VableBoxNotSeeded`.
    //
    // A param-bearing Python callee that is otherwise inline-eligible but
    // whose body is not a straight-line leaf (loop / branch / non-static
    // vable) cannot be served by the fast-path register seeding.  Emitting
    // the residual leaves it re-interpreted per iteration and lets its short
    // inner loops compile + deopt-storm — strictly slower than interpreting.
    // Decline the enclosing key to interpretation
    // (`FBW_DECLINED_KEYS`) instead of recording the slow residual.
    // Resolve the callee's own portal frame register up-front so both the
    // strict predicate (own-frame vable acceptance) and the multiframe gate
    // share one `ensure_jitcode_index` + `portal_red_regs_at` lookup.  A
    // portal-shaped strict straight-line leaf's LOAD_FAST / STORE_FAST carry
    // the frame-vable locals prologue, folded register-to-register against
    // this frame reg (see the `*_vable_via_metainterp` short-circuits).
    // `u16::MAX` for a non-portal callee keeps the strict predicate
    // byte-identical (`inline_resolvable_seeded_frame_op` declines).
    let callee_portal_frame_reg = crate::state::ensure_jitcode_index(callee_code_key as *const ())
        .filter(|&jc| crate::state::built_as_portal_at(jc))
        .map(|jc| crate::state::portal_red_regs_at(jc).0)
        .unwrap_or(u16::MAX);
    let strict_inlinable =
        callee_fast_path_inlinable(body.code, callee_descr_refs, ctx, callee_portal_frame_reg);
    // #68: a forward-branch-bearing callee is inlinable with a multi-frame
    // guard snapshot (its in-callee branch
    // guard resumes through `walker_capture_multi_frame_inline_snapshot` rather
    // than collapsing to the caller boundary).  The relaxed predicate also
    // accepts a callee whose only non-strict ops are reads off its OWN seeded
    // frame register, so resolve that register up-front (the same
    // `ensure_jitcode_index` + `portal_red_regs_at` the seeding below uses).
    // A multiframe caller no longer needs to be TOP-LEVEL: a nested caller's
    // paused frame is computed from the framestack's top (the live
    // intermediate callee jitcode) by `compute_inline_caller_frame`, bounded by
    // a depth cap on the inline stack (the `n_parents == n_callees` valve in
    // the snapshot path is the real desync safety net).
    let multiframe_eligible = !strict_inlinable;
    let callee_frame_reg = if multiframe_eligible {
        crate::state::ensure_jitcode_index(callee_code_key as *const ())
            .map(|jc| crate::state::portal_red_regs_at(jc).0)
            .unwrap_or(u16::MAX)
    } else {
        u16::MAX
    };
    let inline_depth = ctx.session.borrow().inline_depth();

    // A strict straight-line callee is seeded with its own frame red so guards
    // carry a real two-frame snapshot.  A callee beyond the currently-supported
    // resume depth remains a residual call: there is no single-frame inline
    // fallback.  PyPy's `MetaInterp.perform_call` always pushes a distinct
    // `MIFrame`, and `resume.rebuild_from_resumedata` rebuilds one frame per
    // encoded section; collapsing an unseeded callee onto its caller loses the
    // callee's pycode/globals/locals identity.
    //
    // Computed here rather than beside its first use in the seed block, because
    // `seeded_callee_resume` below asks the same question and is 200 lines
    // earlier.
    let strict_seed = strict_inlinable && inline_depth < fbw_max_multiframe_depth();
    let contains_raise = body_facts.contains_raise;
    // A callee that raises inline needs the cross-frame bridge the carrier
    // drain builds once a guard inside the compiled chain fails.  The drain
    // walks the paused middle frames between the raising leaf and the root
    // the way `finishframe_exception` does, including `ChangeFrame` into a
    // middle that catches and `finishframe_exception` from a middle that
    // raises after `finishframe`.  Distinct raising depth is 3; recursive
    // / duplicate raising is unbounded here (`max_unroll_recursion` is the
    // sole remaining bound).  `perform_call` has no such cap; see
    // `fbw_effective_multiframe_depth`.  A value-returning chain (no raise)
    // inlines to the full depth either way.
    // Value-returning recursion still keys only on this callee's own
    // greenkey (`_opimpl_recursive_call`).  The duplicate-w_code bit is
    // only a raising-chain safety valve: it must not promote an unrelated
    // value-returning callee to the unbounded recursive cap.
    let effective_multiframe_depth = fbw_effective_multiframe_depth(
        contains_raise,
        if contains_raise {
            recursive_portal_present
                || framestack_has_duplicate_w_code(&ctx.session.borrow().framestack)
        } else {
            recursive_portal_present
        },
    );
    // The instance-`__next__` FOR_ITER route uses the same seeded-frame shape
    // as other CALL-entered inlines.  Its catch arm owns exception-to-exhaustion
    // conversion, so an unseeded caller-boundary resume is not part of that
    // route's deopt discipline.
    // A zero-param `for` body's back-edge is a plain `goto`, so the
    // forward-branch scan would walk the loop and skip the portal activation
    // a residual call charges. That chain runs past the recursion limit.
    // Parameter-bearing loop callees stay on the multiframe path: refusing
    // them crashes the recursive-portal handler fixtures. `step()` has no
    // `for`, so it stays eligible.
    let try_multiframe = multiframe_eligible
        && !(nparams == 0 && pyre_interpreter::code_has_for_iter(callee_code))
        && inline_depth < effective_multiframe_depth
        && callee_fast_path_inlinable_allowing_forward_branch(
            body.code,
            callee_descr_refs,
            ctx,
            callee_frame_reg,
        );
    let instance_next_seeded_route = instance_next_foriter_green_key.is_some();
    // `can_inline_callable` (`warmstate.py`) tests only `can_never_inline` /
    // `JC_DONT_TRACE_HERE`; `perform_call` (`pyjitpl.py`) pushes an MIFrame.
    // A seeded frame already exists here, and a walk abort converts through
    // `convert_and_run_from_pyjitpl` (`blackhole.py`). The remaining terms
    // are unported OTHER: `code_has_for_iter`, `code_is_self_recursive`,
    // and the constant-callable screen. That screen pins `Function.code`
    // (`guards_the_callee_function`) — the greenkey `can_inline_callable`
    // unwraps — not function identity, so a fresh lambda with one code
    // object is admitted the same way a closure is.
    if fbw_foriter_inflight_active() && !instance_next_seeded_route {
        let seeded_callee_resume = (try_multiframe || strict_seed)
            && (callable_guard_op.is_constant()
                || receiver_pinned_by_class
                || guards_the_callee_function);
        let foriter_seeded_admit = seeded_callee_resume
            && !pyre_interpreter::code_has_for_iter(callee_code)
            && !pyre_interpreter::code_is_self_recursive(callee_code);
        if !foriter_seeded_admit && fbw_inline_diag_enabled() {
            eprintln!(
                "[inline-foriter-gate] pc={} seeded_admit={foriter_seeded_admit} \
                 const_callable={} mf={try_multiframe} strict={strict_inlinable} \
                 for_iter={} selfrec={} guards_fn={guards_the_callee_function}",
                op.pc,
                callable_guard_op.is_constant(),
                pyre_interpreter::code_has_for_iter(callee_code),
                pyre_interpreter::code_is_self_recursive(callee_code),
            );
        }
        if !foriter_seeded_admit {
            return resolved_inline_decline(op.pc, line!());
        }
    }
    // A widened method-form body that also raises was declined here until the
    // resume-liveness filter started keeping the raise operand.  The decline
    // existed because a guard whose resume coordinate landed on the `Reraise`
    // needed ref registers the recorded path never wrote
    // (`collect_callee_active_boxes`), and that decline arrived mid-recording
    // on a non-effect-free opcode with no mid-body carrier, discarding the
    // whole enclosing loop.  With the operand retained the sub-walk records the
    // handler region with the registers the coordinate names, so the body
    // inlines like any other.
    //
    // The raise need not even execute to have been caught by it: a dead
    // `if self.i < 0: raise` in an `o.m()` callee measured 1705 ns/call against
    // 15.3 once admitted, and `self.i >= self.n` 1207 -> 15.9.  Swapping that
    // `raise` for a `return` already measured 17.7, which is what named the
    // token rather than the branch or the attribute compare.
    if fbw_inline_diag_enabled() {
        let mut pc = 0usize;
        let mut shown = 0;
        while pc < body.code.len() && shown < 8 {
            let Some(d) = crate::jitcode_runtime::decode_op_at(body.code, pc) else {
                break;
            };
            let ops: Vec<u8> = body.code[d.pc + 1..d.next_pc.min(body.code.len())].to_vec();
            eprintln!("[inline-body] pc={} {} operands={:?}", d.pc, d.key, ops);
            pc = d.next_pc;
            shown += 1;
        }
    }
    // A zero-param callee has no positional argument to seed. `perform_call`
    // (`pyjitpl.py`) still pushes its MIFrame: parameter count is not an
    // admission test. A body neither `try_multiframe` nor `strict_seed`
    // accepts residualizes at the `!seeded_inline` decline below (`Ok(None)`),
    // which records a residual call instead of stamping the caller key into
    // `FBW_DECLINED_KEYS`.

    // A self-recursive callee unrolls until its own frame count reaches
    // `max_unroll_recursion`, then routes to the direct `CALL_ASSEMBLER` arm
    // (`try_walker_call_assembler_self_recursive`, reached when this inline
    // attempt returns `Ok(None)`).  That bound is the one
    // `fbw_inline_recursion_count` already applied above, matching
    // `opimpl_recursive_call` (`pyjitpl.py:1390-1416`), which counts the portal
    // frames already on the framestack and inlines while the count is below
    // `max_unroll_recursion`.  Carrier-resume frames participate in that same
    // count: `drive_bridge_frame_subwalk` reconstructs their `W_Code` green
    // identities before entering this path.
    // `can_inline_callable` (`warmstate.py`) tests only `can_never_inline` /
    // `JC_DONT_TRACE_HERE`; `perform_call` (`pyjitpl.py`) traces the taken
    // path, including `except: raise` and `except E: return`. A seeded
    // callee with an exception table is admitted; an abort inside converts
    // through `convert_and_run_from_pyjitpl` (`blackhole.py`).
    let seeded_inline = try_multiframe || strict_seed;
    // Preflight the caller frame BEFORE the seed below records a virtual
    // PyFrame.  A CALL covered by a try/catch marker must remain residual so
    // its post-call catch resume routes an exception; returning after frame
    // emission would strand dead seed IR and force an abort/replay after the
    // callee already consumed external state.  RPython decides whether to
    // inline the graph before `perform_call` pushes its MIFrame.
    if !seeded_inline {
        // The remaining strict cases are deeper than the resume chain
        // currently supports. Keep the CALL residual instead of entering an
        // inlined JitCode with an empty frame red. This is the same structural
        // choice as `pyjitpl.py do_residual_or_indirect_call` when a callee
        // cannot be followed: every call that *is* inlined keeps its own red
        // frame.
        //
        // Two populations reach this one decline and they are not the same
        // work to serve, so name which one in the `[fbw-census]` tally: a body
        // neither predicate accepts declined here before this gate existed,
        // while the depth case is the one the seeded-only admission newly
        // residualizes.
        let seeded_why = if !strict_inlinable {
            "SeededInline::NeitherStrictNorMultiframe"
        } else {
            "SeededInline::OverMultiframeDepth"
        };
        if fbw_debug_abort_enabled() {
            crate::jitcode_dispatch::census_record(seeded_why);
        }
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[seeded-inline-decline] pc={} why={seeded_why} \
                 strict={strict_inlinable} mf={try_multiframe} depth={inline_depth}",
                op.pc
            );
        }
        return resolved_inline_decline(op.pc, line!());
    }
    // `seeded_inline` is the admission: everything else returned above, so the
    // caller frame is computed unconditionally here and every inlined callee
    // reaches the seed with a real parent.
    // A canonical-helper subwalk's `op.pc` is a byte offset in the helper
    // jitcode. The paused caller is the Python CALL that entered the
    // outermost helper, kept on `entry_py_pc`.
    let caller_frame_pc =
        if ctx.fbw_mode.inline_subwalk && ctx.session.borrow().last_inline().is_none() {
            match ctx.entry_py_pc {
                EntryPyPc::Jit(pc) => pc,
                EntryPyPc::Py(_) => op.pc,
            }
        } else {
            op.pc
        };
    let precomputed_parent_frame =
        match compute_inline_caller_frame(ctx, caller_frame_pc, !callee_code.freevars.is_empty()) {
            Ok(parent) => parent,
            Err(InlineCallerFrameDecline::TryBlockCatchMarker)
            | Err(InlineCallerFrameDecline::Unavailable) => {
                return resolved_inline_decline(op.pc, line!());
            }
        };

    let mut callable_guard_op = callable_guard_op;
    let mut callable_guard_value = callable_guard_value;
    if let Some(bound) = bound_method {
        // `_Method._immutable_fields_ = ['w_function', 'w_instance']`
        // (pypy/interpreter/function.py).  Preserve those as red field
        // reads: guard only the Method layout and underlying function, then
        // pass the live receiver field into the callee.  Baking the receiver
        // concrete would collapse bound methods with different `self` values.
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        walker_guard_class(ctx, op.pc, bound.method_op, method_type_addr)?;
        let function_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            bound.method_op,
            crate::descr::method_w_function_descr(),
        );
        let receiver_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            bound.method_op,
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            receiver_op,
            majit_ir::Value::Ref(majit_ir::GcRef(bound.receiver as usize)),
        );
        callee_args[0] = receiver_op;
        callable_guard_op = function_op;
        callable_guard_value = bound.function;
    }
    // Nursery-born `Function.__globals__` / `W_Code` must survive the
    // allocations below (a specialised pair's `wrapint`, the kwargs dict)
    // until they land in `WalkFrameState`, which `InlineFrameStateGuard`
    // walks. `PyCode.frame_stores_global` has no frame field in this
    // encoder, so that rare shape stays residual.
    let callee_globals_obj = unsafe { pyre_interpreter::function_get_globals_obj(callable) };
    if unsafe {
        pyre_interpreter::w_code_frame_stores_global(
            w_code as pyre_object::PyObjectRef,
            callee_globals_obj,
        )
    } {
        return resolved_inline_decline(op.pc, line!());
    }
    let _inline_const_roots = pyre_object::gc_roots::push_roots();
    let inline_globals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(callee_globals_obj);
    let inline_code_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_code as pyre_object::PyObjectRef);
    refresh_rooted_refs(&mut callee_arg_concretes, &arg_root_slots);
    refresh_varkw_concretes(&mut varkw_extra, &varkw_root_slots);
    if let Some(star) = star_args {
        // `space.fixedview(w_stararg)` -> `w_obj.tolist()`: the class guard
        // re-proves the layout the resolve picked the reads for, and an exact
        // tuple class is also what `_uses_tuple_iter` needs to take this path.
        walker_guard_class(
            ctx,
            op.pc,
            star.starargs_op,
            defaults_repr_class_addr(star.repr),
        )?;
        if star.repr == DefaultsRepr::ItemsBlock {
            let items = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                star.starargs_op,
                crate::descr::tuple_wrappeditems_descr(),
            );
            // `_match_signature` raises on any other count; the pair classes
            // fix theirs at two, so only the array-backed tuple needs this.
            let len_op = crate::state::opimpl_arraylen_gc(
                ctx.trace_ctx,
                items,
                crate::state::pyobject_gcarray_descr(),
            );
            let expected = ctx.trace_ctx.const_int(star.values.len() as i64);
            let same_len = ctx.trace_ctx.record_op(OpCode::IntEq, &[len_op, expected]);
            ctx.trace_ctx
                .set_opref_concrete(same_len, majit_ir::Value::Int(1));
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[same_len])?;
            for index in 0..star.values.len() {
                let index_op = ctx.trace_ctx.const_int(index as i64);
                callee_args[index] = crate::state::trace_items_block_getitem_value_pure(
                    ctx.trace_ctx,
                    items,
                    index_op,
                );
            }
        } else {
            for index in 0..star.values.len() {
                let (elem, concrete) =
                    with_arg_refs_pinned(&mut callee_arg_concretes, |concretes| {
                        let ConcreteValue::Ref(value) = concretes[index] else {
                            unreachable!("unpacked star element is a ref");
                        };
                        emit_pair_tuple_item(ctx, op.pc, star.starargs_op, star.repr, index, value)
                    })?;
                callee_args[index] = elem;
                if let Some(ConcreteValue::Ref(obj)) = concrete {
                    adopt_rooted_ref(
                        &_arg_roots,
                        &mut callee_arg_concretes,
                        &mut arg_root_slots,
                        &mut varkw_extra,
                        &varkw_root_slots,
                        index,
                        obj,
                    );
                }
            }
        }
    }

    if let Some(kw) = star_kwargs {
        // `view_as_kwargs` reads the keys and values of an exact dict under
        // the unicode strategy; the class and strategy guards re-prove the
        // layout the resolve bound the names against.
        walker_guard_class(
            ctx,
            op.pc,
            kw.dict_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(
            ctx,
            op.pc,
            kw.dict_op,
            pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE),
        )?;
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            kw.dict_op,
            crate::descr::dict_strategy_word_descr(),
        );
        let strategy_const = ctx
            .trace_ctx
            .const_int(&pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64);
        walker_emit_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[strategy, strategy_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, strategy_const);
        // `_match_keywords` rejects a keyword no parameter takes; with every
        // bound name required to hit below, pinning the count leaves none.
        let effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        );
        let len_op = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            crate::helpers::jit_dict_len as *const (),
            &[kw.dict_op],
            &[Type::Ref],
            Type::Int,
            effect.clone(),
        );
        ctx.trace_ctx
            .set_opref_concrete(len_op, majit_ir::Value::Int(kw.bound.len() as i64));
        let expected = ctx.trace_ctx.const_int(kw.bound.len() as i64);
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[len_op, expected])?;
        for &(index, name, _) in &kw.bound {
            let value = match arg_root_slots.get(index).copied().flatten() {
                Some(slot) => pyre_object::gc_roots::shadow_stack_get(slot),
                None => match callee_arg_concretes.get(index).copied() {
                    Some(ConcreteValue::Ref(obj)) if !obj.is_null() => obj,
                    _ => continue,
                },
            };
            let name_op = ctx.trace_ctx.const_ref(name as i64);
            let value_op = ctx.trace_ctx.call_ref_typed_with_effect(
                crate::helpers::jit_dict_exact_unicode_lookup_or_null as *const (),
                &[kw.dict_op, name_op],
                &[Type::Ref, Type::Ref],
                effect.clone(),
            );
            ctx.trace_ctx.set_opref_concrete(
                value_op,
                majit_ir::Value::Ref(majit_ir::GcRef(value as usize)),
            );
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[value_op])?;
            callee_args[index] = value_op;
            callee_arg_concretes[index] = ConcreteValue::Ref(value);
        }
    }

    // Path-1 (#68): the inlined callee's promoted `pycode` static field and
    // its code-derived globals semantic constant.  The latter is no longer a
    // physical virtualizable scalar; LOAD_GLOBAL derives it through the
    // callee frame's pycode.
    let inline_consts = InlineCalleeConsts {
        jitcode_index: crate::state::ensure_jitcode_index(callee_code_key as *const ())
            .map_or(-1, |index| index as i32),
    };

    // Specialize the inlined body on this exact callable: a later
    // iteration calling a different function at this site must deopt
    // rather than run the wrong body.  The guard resumes at the caller's
    // CALL boundary (single outer Python frame — re-execute the whole
    // call on deopt), captured via `fbw_mode.inline_subwalk` for
    // the sub-walk guards below.
    if let Some((receiver, concrete_receiver, w_class, version_tag, attr_cell)) =
        exception_receiver_guard
    {
        walker_guard_exception_attr_slot(
            ctx,
            op.pc,
            receiver,
            concrete_receiver,
            w_class,
            version_tag,
        )?;
        // After the class guards and the pin, never before: the pin is what
        // makes the cell pointer a constant, and an in-place write to the cell
        // is the one rebind the pin does not see.
        if let Some((cell, expected)) = attr_cell {
            walker_promote_object_mutable_cell(ctx, op.pc, cell, expected)?;
        }
    }
    if let Some((arg, concrete_arg, w_type)) = arg_class_guard {
        // `GuardClass` compares the object's physical `ob_type`, not its Python
        // `W_TypeObject`.  Pin the physical type: a boxed builtin whose
        // `ob_type` the optimizer already knows would make a `GuardClass`
        // against the heap type object provably fail, discarding the loop
        // (`InvalidLoop`).  `walker_guard_class` also emits the tagged-int
        // low-bit test, needed when `arg` may arrive as a tagged int.
        let physical_type = unsafe { (*concrete_arg).ob_type } as i64;
        walker_guard_class(ctx, op.pc, arg, physical_type)?;
        // A builtin subclass / user instance shares its `ob_type` with the base
        // layout, so `ob_type` alone does not pin the class the reflected-op
        // decline (`w_type_issubtype`) was computed against.  Guard the live
        // `w_class` too so an arg of a distinct class deopts.  Singletons with a
        // null `w_class` (`Ellipsis`/`NotImplemented`) are pinned exactly by
        // `ob_type`, and guarding their null slot against `w_type` would itself
        // be provably false — so guard `w_class` only when it is populated.
        if !unsafe { (*concrete_arg).w_class }.is_null() {
            let live_w_class = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                arg,
                crate::descr::w_class_descr(),
            );
            let w_class_const = ctx.trace_ctx.const_ref(w_type as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op.pc,
                OpCode::GuardValue,
                &[live_w_class, w_class_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .replace_box(live_w_class, w_class_const);
        }
    }

    // Keep the closure cells as red operands.  A MAKE_FUNCTION in the caller's
    // loop creates a fresh function and fresh enclosing cells on every
    // iteration; the trace-time cell pointers are only the concrete shadow
    // used while walking.  PyPy threads the live closure cells into each
    // callee MIFrame through setup_call, so folding them to constants here
    // collapses distinct inlined frames onto the tracing iteration's cells.
    // Specializer-owned callees which cannot be read through
    // `callable_guard_op` retain the already-pinned concrete cells.
    let mut freevar_cell_ops: Vec<OpRef> = concrete_freevar_cells
        .iter()
        .map(|&cell| ctx.trace_ctx.const_ref(cell as i64))
        .collect();

    // The `__call__` arm dispatches on the instance itself, and the receiver
    // guard above already pinned THAT operand by class and type version --
    // which is exactly what `space.lookup` promotes to select the resolved
    // `__call__`.  Pinning its identity on top specializes the body on one
    // instance, so a loop over several instances of one callable class
    // side-exits per instance.  Measured over 200000 calls: one instance reads
    // 0 bridges and 1 guard failure, 64 instances read 87424 failures with the
    // bridge count saturated at 29 -- 17.3x CPython against 3.0x for the same
    // loop on one instance.  `self` reaches the callee as the red operand
    // `r_args[0]` (the concrete is only the walk's shadow), so the class and
    // version guard is the whole requirement (`receiver_pinned_by_class`).
    if !guards_the_callee_function {
        // Those sites resolve the callee through their own guarded path (a
        // type version tag, a receiver class guard); all this has to pin is
        // the operand they dispatched on.
        if !callable_guard_op.is_constant() && !receiver_pinned_by_class {
            let expected = ctx
                .trace_ctx
                .const_ref(callable_guard_value as usize as i64);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[callable_guard_op, expected], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        // Pinning the operand pins none of the callee's fields, and this inline
        // bakes `code` in the strongest form there is — it selects which callee
        // body the trace walks into.  `function.py:47 _immutable_fields_ =
        // ['code?', ...]` is what covers that, and the `?` costs one marker
        // plus one `GUARD_NOT_INVALIDATED` per trace instead of a load and a
        // `GUARD_VALUE` per iteration.  It is also the only form available
        // here: the guard arm below reads the field off the pinned operand,
        // which this arm either cannot do (the operand is not the callee) or
        // must not do (a baked `ConstPtr`).
        walker_pin_function_code(ctx, op.pc, callable)?;
    } else {
        // `function.py getcode()` promotes `self.code`, never `self`.
        // The code object below and globals namespace in `InlineCalleeConsts`
        // are baked constants, so guard those `?`-fields off the live
        // function. `defs_w` has its own guard below. Closure cells differ:
        // `closure?[*]` permits a pure live read, but the cells themselves are
        // red inputs to the callee frame and must not be pinned to the tracing
        // iteration.
        //
        // Reading them live is what `f.__code__ = g.__code__` needs: pinning
        // the object pins none of its fields.  `code?` is quasi-immutable, so
        // `opimpl_getfield_gc_r` pairs each read with `record_quasiimmut_field`
        // and assigning the field invalidates the traces that folded it; the
        // value guards stay on top as the stricter identity check the reads
        // below assume.  They also stay because this arm exists for a callee
        // whose identity changes every iteration, so the field is re-read
        // anyway, and because the marker resolves its owner by raw address —
        // the guard is the only answer that keeps working for a callee the
        // collector can relocate.
        //
        // Guarding the function OBJECT instead pinned its identity, which a
        // callee built by a `MAKE_FUNCTION` in the caller's own loop body can
        // never match twice: every iteration allocates a fresh function, so
        // the guard failed forever while the code object it stands for is
        // loop-invariant.
        //
        // Keep GETFIELD+GUARD_VALUE even on a ConstPtr callable.
        // `code?` matches `function.py getcode()` but Linux `test_uuid`
        // SIGSEGV's with the marker: after `test_UUID` warms the JIT,
        // `jit_bigint_int_eq` treats a small int as `*const BigInt`.
        walker_guard_function_field(
            ctx,
            op.pc,
            callable_guard_op,
            crate::descr::function_code_descr(),
            callee_code_key as i64,
        )?;
        walker_guard_function_field(
            ctx,
            op.pc,
            callable_guard_op,
            crate::descr::function_w_globals_descr(),
            pyre_object::gc_roots::shadow_stack_get(inline_globals_slot) as i64,
        )?;
        if !concrete_freevar_cells.is_empty() {
            // `closure?[*]`: the tuple itself is rebuilt by every
            // `MAKE_FUNCTION`, so read its cells live and thread those red
            // operands into the new callee frame. The element count needs no
            // guard of its own: `code` above pins `co_freevars`, and a closure
            // always has exactly that many cells.
            //
            // `specialisedtupleobject.py` stores an object pair inline as
            // `W_SpecialisedTupleObject_oo.value0/value1`; only the ordinary
            // `W_TupleObject` has `wrappeditems`.  The two structs deliberately
            // put those fields at the same offset.  Treating an `_oo` closure
            // as array-backed therefore reads `value0` as the array pointer,
            // then reads the first cell's `w_class` and value as the two
            // closure entries.  Select and guard the concrete storage class
            // before emitting either upstream layout's reads, exactly as the
            // positional-default path below does.
            let closure_ob_type = unsafe { (*concrete_closure).ob_type };
            let closure_is_pair_object = std::ptr::eq(
                closure_ob_type,
                &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
            );
            if !closure_is_pair_object
                && !std::ptr::eq(closure_ob_type, &pyre_object::pyobject::TUPLE_TYPE)
            {
                return resolved_inline_decline(op.pc, line!());
            }
            let closure_op = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                callable_guard_op,
                crate::descr::function_closure_descr(),
            );
            ctx.trace_ctx.try_set_opref_concrete(
                closure_op,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_closure as usize)),
            );
            walker_guard_class(ctx, op.pc, closure_op, closure_ob_type as i64)?;
            freevar_cell_ops.clear();
            if closure_is_pair_object {
                debug_assert_eq!(concrete_freevar_cells.len(), 2);
                for (i, &cell) in concrete_freevar_cells.iter().enumerate() {
                    let descr = if i == 0 {
                        crate::descr::specialised_tuple_oo_value0_descr()
                    } else {
                        crate::descr::specialised_tuple_oo_value1_descr()
                    };
                    let cell_op =
                        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, closure_op, descr);
                    ctx.trace_ctx.try_set_opref_concrete(
                        cell_op,
                        majit_ir::Value::Ref(majit_ir::GcRef(cell as usize)),
                    );
                    freevar_cell_ops.push(cell_op);
                }
            } else {
                let items_op = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    closure_op,
                    crate::descr::tuple_wrappeditems_descr(),
                );
                for (i, &cell) in concrete_freevar_cells.iter().enumerate() {
                    let index_op = ctx.trace_ctx.const_int(i as i64);
                    let cell_op = crate::state::trace_items_block_getitem_value_pure(
                        ctx.trace_ctx,
                        items_op,
                        index_op,
                    );
                    ctx.trace_ctx.try_set_opref_concrete(
                        cell_op,
                        majit_ir::Value::Ref(majit_ir::GcRef(cell as usize)),
                    );
                    freevar_cell_ops.push(cell_op);
                }
            }
        }
    }

    if let Some(mut defaults) = positional_defaults {
        // Element pins and the vararg tuple ran after the resolve copied
        // `tuple`. The shadow slot is the address a collection forwarded.
        defaults.tuple = pyre_object::gc_roots::shadow_stack_get(defaults.tuple_root);
        // `defs_w?[*]`: read the live field on every compiled iteration.
        //
        // `jtransform.py:895-903` prefixes a quasi-immutable read with
        // `record_quasiimmut_field`; the read below records the op directly
        // rather than through `opimpl_getfield_gc_r`, so the pairing is spelled
        // out here.  It is what installs the watcher on this callee and puts the
        // field in `quasi_immutable_deps`, so a later `f.__defaults__ = ...`
        // retires this loop instead of leaving the guard below to fail forever.
        //
        // Gated on the same flag as the field guards above.  Reaching a
        // `Some(defaults)` already proved the guard operand is `callable`
        // itself, so what the flag still adds here is the trace-constant
        // exclusion: the install dereferences `callable_guard_op` to reach the
        // `mutate_defs_w` slot, and a baked `ConstPtr` is the pointer the note
        // above says must not be loaded through.
        if guards_the_callee_function {
            crate::state::record_quasiimmut_field(
                ctx.trace_ctx,
                callable_guard_op,
                crate::descr::function_defs_w_descr(),
            );
        }
        let defaults_op = ctx.trace_ctx.record_op_with_descr(
            OpCode::GetfieldGcR,
            &[callable_guard_op],
            crate::descr::function_defs_w_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            defaults_op,
            majit_ir::Value::Ref(majit_ir::GcRef(defaults.tuple as usize)),
        );

        // `defs_w?[*]`: the elements are read live below.  Trace time
        // picks the SHAPE (`ob_type` → which field/array read) and the
        // missing-parameter set; `_flat_pycall_defaults` then recomputes
        // each index from the live `len(self.defs_w)`.
        //
        // For a callee whose identity is not a baked `ConstPtr`, re-check
        // the shape directly.  Upstream never pins the list:
        // `funccall_valuestack` tests
        // `natural_arity > nargs >= natural_arity - len(self.defs_w)` live,
        // and `_flat_pycall_defaults` reads `ndefs = len(self.defs_w)` and
        // `self.defs_w[j]` live too (`function.py`).  Pinning the exact
        // length was a guard a mid-loop `f.__defaults__ = (…)` — or a
        // `MAKE_FUNCTION` whose default expression rebuilds the tuple
        // every iteration (`def add(value=i)`) — can never satisfy twice.
        // `synth/foriter_make_function_body` paid 980 guard failures in
        // 3000 iterations for that pin, with no bridge.
        //
        // `GuardClass` dereferences its operand and a callee with no
        // defaults at all carries `PY_NULL` here, so the nullity check
        // comes first.  The `ItemsBlock` arm below guards only
        // "enough defaults"; the two specialised reprs are arity-2
        // classes, so the class guard already fixes their length.
        //
        // A constant callable keeps the identity `GuardValue`.  There the
        // function is pinned by its own guard, `defs_w` carries no
        // quasi-immutable watcher (a baked `ConstPtr` must not be loaded
        // through to reach the `mutate_defs_w` slot), and this guard is what
        // stands in for the invalidation a later `f.__defaults__ = ...` would
        // otherwise slip past.
        if guards_the_callee_function {
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[defaults_op])?;
            walker_guard_class(
                ctx,
                op.pc,
                defaults_op,
                defaults_repr_class_addr(defaults.repr),
            )?;
        } else {
            let tuple_expected = ctx.trace_ctx.const_ref(defaults.tuple as i64);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[defaults_op, tuple_expected], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }

        // Preserve the actual Ref boxes instead of baking one anchor's
        // concrete value.  Which read that is depends on where the element
        // lives; the guards above already proved the layout, so no arm needs a
        // class guard of its own.
        match defaults.repr {
            DefaultsRepr::ItemsBlock => {
                let items = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    defaults_op,
                    crate::descr::tuple_wrappeditems_descr(),
                );
                if guards_the_callee_function {
                    // `funccall_valuestack` / `_flat_pycall_defaults`:
                    // parameter `p` reads `defs_w[len - (nparams - p)]`.
                    // A star-mapping call can leave an early parameter
                    // unbound and fill a later one, so the live tuple has
                    // to reach the earliest hole, not merely contain as
                    // many elements as there are holes.  `arraylen_gc` is
                    // `len(self.defs_w)`.  Pinning the exact length would
                    // deopt every later iteration of a mid-loop
                    // `__defaults__` swap whose new tuple is still long
                    // enough.
                    let len_op = crate::state::opimpl_arraylen_gc(
                        ctx.trace_ctx,
                        items,
                        crate::state::pyobject_gcarray_descr(),
                    );
                    let required_len = defaults
                        .values
                        .iter()
                        .map(|&(_, tuple_index, _)| (defaults.len - tuple_index) as i64)
                        .max()
                        .unwrap_or(0);
                    let min_len = ctx.trace_ctx.const_int(required_len);
                    let enough = ctx.trace_ctx.record_op(OpCode::IntGe, &[len_op, min_len]);
                    ctx.trace_ctx.set_opref_concrete(
                        enough,
                        majit_ir::Value::Int(i64::from((defaults.len as i64) >= required_len)),
                    );
                    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[enough])?;
                    for (param_index, tuple_index, _) in defaults.values {
                        let from_end = (defaults.len - tuple_index) as i64;
                        let from_end_op = ctx.trace_ctx.const_int(from_end);
                        let index = ctx
                            .trace_ctx
                            .record_op(OpCode::IntSub, &[len_op, from_end_op]);
                        ctx.trace_ctx
                            .set_opref_concrete(index, majit_ir::Value::Int(tuple_index as i64));
                        callee_args[param_index] =
                            crate::state::trace_items_block_getitem_value_pure(
                                ctx.trace_ctx,
                                items,
                                index,
                            );
                    }
                } else {
                    for (param_index, tuple_index, _) in defaults.values {
                        let index = ctx.trace_ctx.const_int(tuple_index as i64);
                        callee_args[param_index] =
                            crate::state::trace_items_block_getitem_value_pure(
                                ctx.trace_ctx,
                                items,
                                index,
                            );
                    }
                }
            }
            DefaultsRepr::PairObject | DefaultsRepr::PairInt => {
                for (param_index, tuple_index, _) in defaults.values {
                    let (elem, concrete) =
                        with_arg_refs_pinned(&mut callee_arg_concretes, |concretes| {
                            let ConcreteValue::Ref(value) = concretes[param_index] else {
                                unreachable!("defaulted parameter is a ref");
                            };
                            emit_pair_tuple_item(
                                ctx,
                                op.pc,
                                defaults_op,
                                defaults.repr,
                                tuple_index,
                                value,
                            )
                        })?;
                    callee_args[param_index] = elem;
                    if let Some(ConcreteValue::Ref(obj)) = concrete {
                        adopt_rooted_ref(
                            &_arg_roots,
                            &mut callee_arg_concretes,
                            &mut arg_root_slots,
                            &mut varkw_extra,
                            &varkw_root_slots,
                            param_index,
                            obj,
                        );
                    }
                }
            }
        }
    }

    // `scope_w[i] = space.finditem_str(w_kw_defs, name)` (`_match_signature`),
    // replacing the keyword-only placeholders pushed above.
    //
    // Nothing is read at run time.  Two pins stand in for the read: the
    // callee's `w_kw_defs` slot, which a `f.__kwdefaults__ = {...}` rebind
    // fails, and the mapping's own strategy version, which a `del`, a new
    // key, a replaced entry or a storage switch revokes.  What is left --
    // storing over an entry that is already there -- is absorbed by the cell
    // without a version bump, and the fold below reads that cell's field live,
    // so it is observed on the very next call the way every runtime does it.
    if let Some(resolved) = kwonly_defaults {
        // Taken off `callable_guard_op`, which the resolve proved names
        // `callable`.
        //
        // `function.py` declares `w_kw_defs?`, and every writer of the slot
        // funnels through `function_set_kwdefaults` -- `fset_func_kwdefaults`,
        // `fdel_func_kwdefaults` and the definition path all reach it -- which
        // notifies `QuasiImmutSlot::WKwDefs`.  The orthodox pin is therefore
        // the marker, and it costs nothing to flush: the guard below drains
        // every marker recorded this epoch, and one is going out for the
        // mapping version regardless.
        //
        // Available on a constant callable for the reason
        // `walker_pin_function_code` gives: a marker resolves its owner by raw
        // address at record and compile time, so it never loads through the
        // baked `ConstPtr` the guard arm must stay off.  What being constant
        // adds is the part a marker needs and cannot check for itself -- one
        // object on every execution, which is what makes watching that object's
        // slot answer for all of them.  The owner word lives on
        // `QuasiImmutDescr.struct_ptr`, which `walk_const_ptr_refs` forwards.
        //
        // Off a live operand the callee's identity is not fixed -- two closures
        // minted from one `def` both satisfy the class-and-code test -- so the
        // slot is re-read per execution and the `GuardValue` stays.  It is
        // stricter than the hint and sound for the reason the marker is: the
        // cells baked below belong to one mapping, and only pinning the slot
        // that holds it makes that true on every later execution.
        let kw_defs_descr = crate::descr::function_w_kw_defs_descr();
        if callable_guard_op.is_constant() {
            let callable_const = ctx.trace_ctx.const_ref(callable as i64);
            crate::state::record_quasiimmut_field(ctx.trace_ctx, callable_const, kw_defs_descr);
        } else {
            walker_guard_function_field(
                ctx,
                op.pc,
                callable_guard_op,
                kw_defs_descr,
                resolved.mapping as i64,
            )?;
        }
        // `walker_pin_namespace_version`'s body, opened up: the resolve already
        // read the strategy box, and there is no way back from here to act on
        // the `Ok(false)` the wrapper would return.  The marker names that
        // box through `QuasiImmutDescr.struct_ptr`, which is forwarded.
        let strategy_const = ctx.trace_ctx.const_ref(resolved.strategy as i64);
        crate::state::record_quasiimmut_field(
            ctx.trace_ctx,
            strategy_const,
            crate::descr::module_dict_version_descr(),
        );
        // One guard drains both markers: it asks whether anything recorded
        // this epoch was invalidated, not which.
        walker_flush_guard_not_invalidated(ctx, op.pc)?;
        for (offset, kwonly) in resolved.values.into_iter().enumerate() {
            // The resolve read this cell before the markers above stood, so a
            // store from another thread in between bumped a version nothing
            // was watching and owes no invalidation for it.  Ask the mapping
            // again now that the watcher is installed: agreeing means the
            // markers answer for what is about to be baked.
            if crate::state::module_dict_cell_value_direct(resolved.mapping, kwonly.slot)
                != Some(kwonly.stored)
            {
                return Err(DispatchError::KwonlyDefaultsMappingRacedRecord { pc: op.pc });
            }
            let Some((value_op, _)) = emit_namespace_cell_value(ctx, op.pc, kwonly.stored)? else {
                return Err(DispatchError::KwonlyDefaultsMappingRacedRecord { pc: op.pc });
            };
            callee_args[nparams + offset] = value_op;
        }
    }

    // `scope_w[co_argcount + co_kwonlyargcount] = space.newtuple(starargs_w)`
    // (`argument.py:233-234`), replacing the placeholder pushed above.
    if let Some((surplus_ops, concrete)) = vararg_surplus {
        let tuple_op = if surplus_ops.is_empty() {
            // `space.newtuple([])` hands back one process-wide object — a
            // call that passes no surplus binds the same `()` every time, and
            // `is_w` is pointer identity for a tuple — so building a fresh
            // empty tuple here would bind an object the interpreter never
            // installs.  Bake the one it does.  This is the ordinary way a
            // `*args` callee is called, not a corner: `def f(a, *rest)` reached
            // as `f(i)` was the whole shape being refused, and it ran 509x
            // slower than the same callee reached as `f(i, 9)`, which inlined.
            ctx.trace_ctx.const_ref(concrete as i64)
        } else {
            let op = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &surplus_ops);
            ctx.trace_ctx
                .set_opref_concrete(op, majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)));
            op
        };
        callee_args[vararg_index] = tuple_op;
    }

    // `w_kwds = space.newdict(kwargs=True)` and one `space.setitem(w_kwds,
    // ...)` per keyword that named no parameter (`_match_signature`),
    // replacing the placeholder pushed above.
    if varkw_slot.is_some() {
        // `newdict` / `setitem` allocate. The vararg tuple and the keyword
        // values are already live objects; pin them before that, then read
        // the slots the collector rewrites. A raw copy in
        // `callee_arg_concretes` would otherwise keep the pre-move address.
        refresh_rooted_refs(&mut callee_arg_concretes, &arg_root_slots);
        refresh_varkw_concretes(&mut varkw_extra, &varkw_root_slots);
        let mut live = Vec::new();
        let mut concrete_slots = Vec::new();
        for (i, concrete) in callee_arg_concretes.iter().copied().enumerate() {
            if let ConcreteValue::Ref(obj) = concrete
                && !obj.is_null()
            {
                concrete_slots.push((i, live.len()));
                live.push(obj);
            }
        }
        let mut extra_slots = Vec::with_capacity(varkw_extra.len());
        for extra in &varkw_extra {
            let name_slot = live.len();
            live.push(extra.name);
            let value_slot = match extra.concrete {
                ConcreteValue::Ref(obj) if !obj.is_null() => {
                    let slot = live.len();
                    live.push(obj);
                    Some(slot)
                }
                _ => None,
            };
            extra_slots.push((name_slot, value_slot));
        }
        let pre_kw_roots = pyre_object::gc_roots::push_roots();
        let pre_kw_base = pre_kw_roots.pin_roots(&live);
        // `argument.py` `_match_signature`: `w_kwds = space.newdict(kwargs=True)`
        // then one `space.setitem` per unmatched keyword.  Walk those
        // interpreter calls so the kwargs strategy's construction records.
        let Some(dict_op) = super::specialize::try_walker_orthodox_kwargs_dict_new(ctx, op.pc)?
        else {
            return resolved_inline_decline(op.pc, line!());
        };
        let Some(concrete_dict) = walker_concrete_ref_object(ctx, dict_op) else {
            return resolved_inline_decline(op.pc, line!());
        };
        // `argument.py` `_match_signature`: `w_kwds` is stored in `scope_w`
        // at `space.newdict(kwargs=True)` and stays live through every
        // `space.setitem` (`_collect_keyword_args`). Pin the freshly
        // allocated mapping before the stores, which may allocate, and
        // re-read the slot after each one.
        let concrete_dict = pre_kw_roots.pin_root(concrete_dict);
        let dict_root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        ctx.trace_ctx.set_opref_concrete(
            dict_op,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_dict as usize)),
        );
        for (extra, (name_slot, value_slot)) in varkw_extra.iter().zip(extra_slots) {
            let name = pyre_object::gc_roots::shadow_stack_get(pre_kw_base + name_slot);
            let value = match value_slot {
                Some(slot) => pyre_object::gc_roots::shadow_stack_get(pre_kw_base + slot),
                None => match ctx.trace_ctx.concrete_of_opref(extra.value) {
                    Some(majit_ir::Value::Ref(gcref)) if gcref.0 != 0 => {
                        gcref.0 as pyre_object::PyObjectRef
                    }
                    _ => unreachable!("checked by the resolve above"),
                },
            };
            let concrete_dict = pyre_object::gc_roots::shadow_stack_get(dict_root_slot);
            ctx.trace_ctx.set_opref_concrete(
                dict_op,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_dict as usize)),
            );
            let name_op = ctx.trace_ctx.const_ref(name as i64);
            if super::specialize::try_walker_orthodox_kwargs_dict_setitem(
                ctx,
                op.pc,
                dict_op,
                concrete_dict,
                name_op,
                name,
                extra.value,
                value,
            )?
            .is_none()
            {
                return resolved_inline_decline(op.pc, line!());
            }
            let concrete_dict = pyre_object::gc_roots::shadow_stack_get(dict_root_slot);
            ctx.trace_ctx.set_opref_concrete(
                dict_op,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_dict as usize)),
            );
        }
        callee_args[varkw_index] = dict_op;
        for (i, slot) in concrete_slots {
            if let ConcreteValue::Ref(obj) = &mut callee_arg_concretes[i] {
                *obj = pyre_object::gc_roots::shadow_stack_get(pre_kw_base + slot);
            }
        }
        // `pre_kw_roots` only covers the dict build. Frame setup still
        // allocates (closure cells), so the finished mapping joins the
        // argument roots that live until the frame is built.
        let concrete_dict = pyre_object::gc_roots::shadow_stack_get(dict_root_slot);
        drop(pre_kw_roots);
        adopt_rooted_ref(
            &_arg_roots,
            &mut callee_arg_concretes,
            &mut arg_root_slots,
            &mut varkw_extra,
            &varkw_root_slots,
            varkw_index,
            concrete_dict,
        );
    }

    let (callee_regs_r, callee_regs_i, callee_regs_f, callee_concrete_r, mut callee_concrete_i) =
        allocate_callee_register_banks(&body, ctx.trace_ctx);
    let callee_regs_r = RegisterBank::with_constants(callee_regs_r, body.num_regs_r);
    let _setup_bank_guard = crate::trace::InlineRegisterBankGuard::enter(&callee_regs_r);
    // MIFrame.setup_call owns the concrete halves before frame construction
    // can allocate. The later WalkContext uses these same slots.
    let callee_state = WalkFrameState::new(WalkFrameStateData {
        concrete_registers_r: callee_concrete_r,
        inline_w_globals: pyre_object::gc_roots::shadow_stack_get(inline_globals_slot) as usize,
        inline_w_code: pyre_object::gc_roots::shadow_stack_get(inline_code_slot) as usize,
        ..Default::default()
    });
    let _setup_state_guard = crate::trace::InlineFrameStateGuard::enter(&callee_state);
    // Fast-path arg seeding: positional args land in the callee's
    // param registers with their concrete shadow (mirror of
    // `dispatch_inline_call_dr_kind`).  The canonical splice regalloc does
    // not pin local-i inputargs to identity colors, so the register the
    // body reads param i from is its per-PC pcdep color at the callee entry
    // (`pcdep_color_slots[0]`), not `r{i}`; an empty fixture map is identity.
    let entry_colors = crate::state::sub_jitcode_entry_param_colors(w_code);
    for i in 0..seeded_locals {
        // RPython passes the same RefFrontendOp into the callee MIFrame, so
        // the Box keeps its recording-time pointer across the frame boundary.
        // Pyre also carries a register-local concrete shadow; mirror that
        // value back onto the canonical OpRef before constructing the callee
        // register file.  Loop-sidecar locals in particular can have a live
        // register concrete even when an intervening vable load returned a
        // fresh, previously unstamped OpRef.
        if let ConcreteValue::Ref(value) = callee_arg_concretes[i]
            && !value.is_null()
        {
            ctx.trace_ctx.try_set_opref_concrete(
                callee_args[i],
                majit_ir::Value::Ref(majit_ir::GcRef(value as usize)),
            );
        }
        let reg = match &entry_colors {
            // Colored jitcode: seed param `i` at the register it occupies at
            // the callee entry PC.  A param dead at entry carries no entry
            // color — the body never reads it, so skip seeding rather than
            // clobber a live register.
            Some(entries) => match entries
                .iter()
                .find(|&&(b, _, slot)| b == 1 && slot as usize == i)
            {
                Some(&(_, color, _)) => color as usize,
                None => continue,
            },
            // Portal / skeleton install (empty `pcdep_color_slots`): colors
            // are slot-identity, so param `i` lives in register `i`.
            None => i,
        };
        if reg >= callee_regs_r.len() {
            return resolved_inline_decline(op.pc, line!());
        }
        callee_regs_r.set(reg, callee_args[i]);
        callee_state.borrow_mut().concrete_registers_r[reg] = callee_arg_concretes[i];
    }

    // #68: seed the callee's `frame` / `ec` reds that the codewriter
    // force-alives at every pc (portal_frame_reg / portal_ec_reg).  The
    // sym-less fast path seeds only the param colors above, leaving the reds
    // OpRef::NONE so an in-callee guard snapshot cannot source them
    // (`collect_callee_active_boxes` declines).  RPython seeds these reds as
    // part of `setup_call(allboxes)` for a recursive-portal inline
    // (`pyjitpl.py`, reds=['frame','ec'] `interp_jit.py`): a
    // freshly-built (virtual) callee `PyFrame` plus the caller's shared `ec`.
    // pyre's "every function is its own portal" model makes every inlined
    // callee portal-shaped, so the same seeding applies.  The frame box stays
    // virtual on the hot path (the optimizer folds the NewWithVtable away) and
    // is materialized only on guard failure; `collect_callee_active_boxes` is
    // then unchanged (it finds real boxes).
    //
    // Seeded for BOTH the forward-branch multiframe callee (`try_multiframe`)
    // AND a STRICT straight-line callee (`strict_seed`).
    // With the reds seeded, an in-callee guard resumes at the callee's OWN
    // coordinate through `walker_capture_multi_frame_inline_snapshot` instead of
    // collapsing to the caller boundary and re-executing the whole call — which
    // re-materializes it at a stale `valuestackdepth` (a resume `LOAD_FAST` push
    // overflows the frame, an `rd_numb` decode overruns) and re-applies a
    // committed heap side effect (visible on the wasm resume path, where a
    // guard-failure deopt is not absorbed by a compiled bridge). Every callee
    // declines the inline when a precondition below fails; there is no
    // single-frame collapse. Every bail below precedes any IR recording, so a
    // residualized call leaves no dead seed op.
    //
    // The seeded virtual callee frame /
    // shared ec / local count are hoisted so the sub-walk return site can
    // emit a `CALL_ASSEMBLER` into the callee loop token when the sub-walk
    // surfaces `SubLoopCalleeCallAssembler` (the callee reached its own loop
    // header).  A strict straight-line callee has no loop, so that outcome
    // never arises for it and the hoisted values are simply unused.
    // Sentinels: every read is dominated by the seed write, but the two are
    // correlated through `callee_frame_seeded`, which definite assignment does
    // not track.
    #[allow(unused_assignments)]
    let mut ca_callee_frame = OpRef::NONE;
    #[allow(unused_assignments)]
    let mut ca_callee_ec = OpRef::NONE;
    #[allow(unused_assignments)]
    let mut ca_nlocals = 0usize;
    // The seeded callee frame's runtime object, for the `enter`/`leave`
    // bracket below — the OpRef alone cannot carry it out of the seed block.
    #[allow(unused_assignments)]
    let mut ca_concrete_frame = std::ptr::null_mut::<pyre_interpreter::PyFrame>();
    // A strict straight-line callee is seeded the same way, so its in-callee
    // guards route through the multi-frame snapshot. A deeper callee beyond the
    // supported resume chain was residualized before this block.
    // True once the callee frame reds are actually seeded (all preconditions
    // below met). It gates routing guards through the multi-frame snapshot.
    #[allow(unused_assignments)]
    let mut callee_frame_seeded = false;
    // The concrete callee frame the seed block materializes, retained so the
    // sub-walk can put it on the interpreter frame chain: the walk executes
    // the callee's residuals for real, and a residual that reads the chain
    // (`sys._getframe`, a traceback) must see the callee it is running in.
    #[allow(unused_assignments)]
    let mut concrete_callee_frame = std::ptr::null_mut::<pyre_interpreter::PyFrame>();
    // Each precondition below answers "can the multiframe seed serve this
    // callee".  A "no" declines the INLINE — `Ok(None)`, this function's own
    // did-not-inline answer, which every caller follows to the ordinary
    // residual call.  It must not be `Err`: that is
    // `LoopBearingCalleeInlineUnsupported`, which `trace.rs` maps to
    // `TraceAction::Abort`, discarding the whole enclosing loop trace.  And
    // because the arm does not call `fbw_decline`, the same static, callee-shaped
    // precondition failed identically on every retrace, so the loop kept
    // re-tracing and re-aborting instead of settling.
    //
    // Both paths decline the callee locally here. Upstream never has this state:
    // `pyjitpl.py` `do_residual_or_indirect_call` residualizes the callee it
    // cannot follow, and the recursion-budget path calls `dont_trace_here` and
    // then still falls through to `do_residual_call` — the enclosing trace
    // survives either way.  `rlib/jit.py`'s `ABORT_*` set (TOO_LONG, BRIDGE,
    // BAD_LOOP, ESCAPE, FORCE_QUASIIMMUT, SEGMENTED_TRACE) has no
    // cannot-inline-this-callee reason at all.
    //
    // The variant's own doc justifies abort-over-residual for a callee whose
    // short inner loops would compile and deopt-storm — but
    // `callee_fast_path_inlinable_allowing_forward_branch` already rejects every
    // backward `goto_if_not` and every `switch`, so a `try_multiframe` callee
    // provably has no inner loop and that rationale does not reach here.
    //
    // All of these sit before the first recorded op (the `GETFIELD_GC_R`
    // below), so returning costs nothing but the inline.
    {
        // POP_JUMP_IF_NONE / POP_JUMP_IF_NOT_NONE lower to a bare
        // `ptr_eq`/`ptr_ne` against the None singleton whose `Kind::Int`
        // result feeds the exitswitch -- no residual call and no boxed
        // bool.  What the shape does need is the tested value popped as a
        // Ref (`pop_ref_or_fresh`, the codewriter PopJumpIfNone arm).  When
        // the multiframe inline int-specializes the tested local, the
        // mid-body guard resume cannot source that operand's Ref form from
        // the callee register banks (`collect_callee_active_boxes` would
        // read a stale/mismatched box), so the encoded liveness stream
        // disagrees with the decoder (`resume.rs decode_ref: unexpected
        // tag`) and the caller frame is corrupted.  Decline to the ordinary
        // residual call until the multi-frame resume reboxes
        // int-specialized identity operands.
        // POP_JUMP_IF_TRUE/FALSE stay inlinable: their `bool` truth folds in the
        // int bank, so no Ref rebox is needed.  A strict straight-line callee
        // has no branch at all, so this scan never fires for it.
        // Stored bound methods carry their explicit receiver and callee frame,
        // so their Ref operands remain available to the resume path.
        //
        // This precondition used to abort the enclosing trace rather than
        // decline the inline, because residualizing it let loops compile that
        // then printed traceback tuples missing their OUTERMOST frame.  That
        // node is now recorded — the two bridge handler-entry arms attach the
        // catching frame's own node — so the decline joins every other
        // precondition here and returns `Ok(None)`.
        //
        // The abort was expensive out of all proportion to the inline it was
        // protecting: a callee that walks a traceback (`while tb is not None`)
        // lowers to exactly this instruction, so any handler calling such a
        // helper aborted every retrace of the enclosing loop.  The guard whose
        // bridge the retrace was building therefore never got one and deopted
        // on every delivery.
        if bound_method.is_none() {
            let liveness = crate::liveness::liveness_for(raw_callee_code);
            let has_is_none_branch = (0..callee_code.instructions.len()).any(|pc| {
                matches!(
                    pyre_interpreter::decode_instruction_at(callee_code, pc),
                    Some((
                        pyre_interpreter::bytecode::Instruction::PopJumpIfNone { .. }
                            | pyre_interpreter::bytecode::Instruction::PopJumpIfNotNone { .. },
                        _
                    ))
                ) && (
                    // Hazard 1 — kept operands. A branch that leaves slots
                    // on the value stack needs its guard resume to restore
                    // them, and the inline sub-walk's mirror does not model
                    // them, so the kept Ref reads NULL and
                    // `walker_branch_guard` raises
                    // BranchGuardUnrestorableKeptStackPermanent — a
                    // permanent abort that discards the enclosing loop
                    // trace.  `stack_depth_at` is the depth BEFORE the
                    // instruction and the branch pops the tested value, so
                    // `depth > 1` is exactly "a kept slot survives".  An
                    // unreachable pc has no depth and cannot fire a guard.
                    liveness.stack_depth_at(pc).is_some_and(|depth| depth > 1)
                            // Hazard 2 — the tested operand itself.  When the
                            // multiframe inline int-specializes the tested
                            // local, the mid-body guard resume cannot source
                            // that operand's Ref form from the callee register
                            // banks (`collect_callee_active_boxes` reads a
                            // stale/mismatched box), so the encoded liveness
                            // stream disagrees with the decoder and the caller
                            // frame is corrupted.  This is independent of kept
                            // depth: a statement-level `if x is None:` reads
                            // depth 1 and still carries it.
                            || match crate::liveness::branch_operand_local(callee_code, pc) {
                                // The branch reads its operand straight out of
                                // a parameter slot the body never reassigns, so
                                // that one binding answers which bank holds it.
                                // `if o is None:` over `def g(x, *, o=None)`
                                // tests `o`; `x` being an int says nothing about
                                // `o`.  The shape that miscompiled tests the int
                                // itself -- `_read_from_buffer(self, size=-1)`
                                // with `if size is None` -- and still declines.
                                Some(local) if local < seeded_locals => {
                                    binding_is_unboxed(callee_arg_concretes[local])
                                }
                                // Anything else on top of the stack is not
                                // described by the incoming bindings, so keep
                                // the whole-signature answer.
                                _ => callee_binds_an_unboxed_local,
                            }
                )
            });
            if has_is_none_branch {
                return resolved_inline_decline(op.pc, line!());
            }
        }
        let nlocals = callee_code.varnames.len();
        let ncells = pyre_interpreter::ncells(callee_code);
        let frame_array_size = nlocals + ncells + callee_code.max_stackdepth as usize;

        let Some(callee_jitcode_index) =
            crate::state::ensure_jitcode_index(callee_code_key as *const ())
        else {
            return resolved_inline_decline(op.pc, line!());
        };
        let (frame_reg, ec_reg) = crate::state::portal_red_regs_at(callee_jitcode_index as i32);
        if frame_reg == u16::MAX
            || ec_reg == u16::MAX
            || frame_reg as usize >= callee_regs_r.len()
            || ec_reg as usize >= callee_regs_r.len()
        {
            return resolved_inline_decline(op.pc, line!());
        }

        // ec red: `perform_call` threads the caller's second portal red
        // down unchanged, just as PyPy shares `ec` between MIFrames.
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        if sym_ptr.is_null() {
            return resolved_inline_decline(op.pc, line!());
        }
        let sym = unsafe { &*sym_ptr };
        let callee_ec = sym.execution_context();
        if callee_ec.is_none() {
            return resolved_inline_decline(op.pc, line!());
        }

        // Copy before each call: a `borrow()` temporary lives until the end of
        // the statement, and later `FrameBox::new` collects.
        let inline_w_code = callee_state.borrow().inline_w_code;
        let inline_w_globals = callee_state.borrow().inline_w_globals;
        let pycode_const = ctx.trace_ctx.const_ref(inline_w_code as i64);
        let w_globals_obj_const = ctx.trace_ctx.const_ref(inline_w_globals as i64);
        let param_boxes: Vec<OpRef> = (0..seeded_locals).map(|i| callee_args[i]).collect();
        // `finish_for_call_with_globals_obj` fills the cell band in two
        // halves: one fresh `w_cell_new(PY_NULL, family)` per pure cellvar,
        // then the closure's existing cells.  Emit the first half as virtual
        // allocations so the callee's own LOAD_DEREF / STORE_DEREF pair folds
        // against the trace's object instead of forcing one.  The family
        // address is a trace-time constant: `PyCode._initialize` builds the
        // table once with the code object, which `pycode_const` has already
        // pinned.  A cellvar that also names a varname is not in this band --
        // it shares that varname's slot and `MAKE_CELL` wraps it in the
        // callee's own prologue.
        let npure_cellvars = pyre_interpreter::npure_cellvars(callee_code);
        let mut cell_slots: Vec<OpRef> = Vec::with_capacity(ncells);
        if npure_cellvars > 0 {
            let cell_header_w_class = ctx.trace_ctx.const_ref(pyre_object::get_instantiate(
                &pyre_object::nestedscope::CELL_TYPE,
            ) as i64);
            for i in 0..npure_cellvars {
                let family = unsafe {
                    pyre_interpreter::pycode::w_code_cell_family(
                        w_code as pyre_object::PyObjectRef,
                        nlocals + i,
                    )
                };
                let family_const = ctx.trace_ctx.const_int(family as i64);
                cell_slots.push(crate::helpers::emit_new_cell_inline(
                    ctx.trace_ctx,
                    family_const,
                    cell_header_w_class,
                ));
            }
        }
        cell_slots.extend_from_slice(&freevar_cell_ops);
        // Same pairing as the Branch A site: the `frame_stores_global`
        // decline earlier in this walk is what lets a `debugdata`-less
        // frame answer for `inline_w_globals`.
        let Some(callee_frame) = crate::helpers::emit_new_pyframe_inline_with_params(
            ctx.trace_ctx,
            &param_boxes,
            &cell_slots,
            &[],
            nlocals,
            frame_array_size,
            nlocals + ncells,
            pycode_const,
            w_globals_obj_const,
        ) else {
            return resolved_inline_decline(op.pc, line!());
        };

        callee_regs_r.set(frame_reg as usize, callee_frame);
        // `perform_call` creates one concrete frame per MIFrame before
        // `setup_call` installs the argument boxes (pyjitpl.py,
        // 1862-1874).  Mirror that recording-time object.  `setup_call`
        // installs the whole box list, so seed every local the symbolic
        // frame above got from `param_boxes` — a `*args` callee's packed
        // vararg tuple is one of them, and a frame short of it publishes
        // that name as unbound to any residual the sub-walk runs.
        // Root every argument the frame constructor will own before any of
        // them is boxed. `to_pyobj` allocates for an int or a float, and that
        // collection moves a kwargs dict or a vararg tuple that is still only
        // a raw pointer later in `callee_arg_concretes`.
        let arg_roots = pyre_object::gc_roots::push_roots();
        let mut raw_args = Vec::with_capacity(seeded_locals);
        let mut box_later = Vec::new();
        for (i, concrete) in callee_arg_concretes
            .iter()
            .take(seeded_locals)
            .copied()
            .enumerate()
        {
            match concrete {
                ConcreteValue::Ref(obj) => raw_args.push(obj),
                ConcreteValue::Null => raw_args.push(pyre_object::PY_NULL),
                ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => {
                    raw_args.push(pyre_object::PY_NULL);
                    box_later.push(i);
                }
            }
        }
        let arg_root_base = arg_roots.pin_roots(&raw_args);
        for i in box_later {
            let boxed = callee_arg_concretes[i].to_pyobj();
            pyre_object::gc_roots::shadow_stack_set(arg_root_base + i, boxed);
        }
        let concrete_args: Vec<pyre_object::PyObjectRef> = (0..seeded_locals)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(arg_root_base + i))
            .collect();
        let concrete_ec = sym.concrete_execution_context();
        // Use a GC-managed frame, not `new_boxed`: the concrete pointer is
        // stamped onto the active trace's frontend op below, and
        // `MetaInterp::walk_active_trace_refs` is then its RPython-style
        // GC root through optimization.  A scope-owned tracer snapshot
        // would be freed when this function returns while the Box value
        // still exists, leaving a dangling recording-time pointer.
        // Re-read after `to_pyobj` / cell emission, which can collect and
        // forward the cell. `new_for_call_with_closure_and_globals_obj` and
        // `FrameBox::new` also collect (`maybe_collect_for_external_malloc`).
        let inline_w_globals = callee_state.borrow().inline_w_globals;
        let mut frame = pyre_interpreter::pyframe::FrameBox::new(
            pyre_interpreter::pyframe::PyFrame::new_for_call_with_closure_and_globals_obj(
                w_code,
                &concrete_args,
                inline_w_globals as pyre_object::PyObjectRef,
                concrete_ec,
                concrete_closure,
                pyre_interpreter::pyframe::FrameLocalsArrayAllocation::OldGenGc,
            ),
        );
        drop(arg_roots);
        let concrete_frame_ptr = frame.as_mut_ptr();
        concrete_callee_frame = concrete_frame_ptr;
        callee_state.borrow_mut().concrete_registers_r[frame_reg as usize] =
            ConcreteValue::Ref(concrete_frame_ptr as pyre_object::PyObjectRef);
        ctx.trace_ctx.set_opref_concrete(
            callee_frame,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame_ptr as usize)),
        );
        // The emitted cells above are the trace's objects; the sub-walk runs
        // the callee's residuals for real, and one that reads a cell
        // (`bh_load_deref_value_fn`) must reach the object the constructor
        // just allocated.  Pair them the way the closure cells are paired at
        // their read.
        for (i, &cell_op) in cell_slots.iter().take(npure_cellvars).enumerate() {
            let concrete_cell =
                unsafe { &*(*concrete_frame_ptr).locals_cells_stack_w }.as_slice()[nlocals + i];
            ctx.trace_ctx.try_set_opref_concrete(
                cell_op,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_cell as usize)),
            );
        }
        // GC-managed FrameBox::drop intentionally relinquishes only the
        // host handle; the frontend op above keeps the frame reachable.
        drop(frame);
        callee_regs_r.set(ec_reg as usize, callee_ec);
        // `perform_call` threads the same concrete ExecutionContext into
        // every MIFrame.  The symbolic second red above and its concrete
        // shadow are one value; leaving only the shadow unknown makes
        // `build_single_frame_miframe` reject an otherwise complete
        // callee image during an escape, after which the legacy caller
        // replay resumes past CALL without its result.
        callee_state.borrow_mut().concrete_registers_r[ec_reg as usize] =
            ConcreteValue::Ref(concrete_ec as pyre_object::PyObjectRef);

        // Retain for a possible `SubLoopCalleeCallAssembler` emit.
        ca_callee_frame = callee_frame;
        ca_callee_ec = callee_ec;
        ca_nlocals = nlocals + ncells;
        ca_concrete_frame = concrete_frame_ptr;
        callee_frame_seeded = true;
    }
    // gh#467 forward-flush inputs are captured AT the CALL, after this
    // iteration's pre-CALL effects and before any callee sub-walk.  Hoisting
    // them above the paused-caller-frame gate lets its try-block decline use
    // the same Entry-carrier predicates as a zero-effect sub-walk abort.
    let unjournaled_before_subwalk = fbw_has_unjournaled_effect();
    let executed_effects_before = fbw_executed_effect_count();
    let is_top_inline = !ctx.fbw_mode.inline_subwalk;
    let abort_flush_call_jitcode_coord: Option<(u32, usize)> = if is_top_inline {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        if sym_ptr.is_null() {
            None
        } else {
            let sym = unsafe { &*sym_ptr };
            if sym.jitcode().is_null() {
                None
            } else {
                unsafe {
                    let jc = &*sym.jitcode();
                    Some((jc.index as u32, op.pc))
                }
            }
        }
    } else {
        None
    };

    // #68: a forward-branch callee inlined under the multi-frame path needs a
    // paused caller frame on the framestack so its in-callee guards snapshot
    // both frames.  The caller's live register banks were preflighted above,
    // before seed IR; at guard-capture time the walk context is the callee's.
    // Every inlined call reaches this point with both its own callee frame and
    // a paused caller image.  `compute_inline_caller_frame` failures were
    // residualized before the seed, so there is no caller-boundary collapse.
    let parent_frame = precomputed_parent_frame;
    let callee_frame_materialized_has_resume = callee_frame_seeded;

    // CODEX1 parity: snapshot the heap-effect state before the callee
    // sub-walk.  If the prologue (callee pc 0 → its loop header) mutates the
    // heap, the `SubLoopCalleeCallAssembler` arm below would re-run the WHOLE
    // call through the residual executor to stamp `ca_result`, applying the
    // prologue's side effects a second time at trace time.  RPython's
    // `do_residual_call` runs the call exactly once (`pyjitpl.py`), so a
    // side-effecting prologue must decline the CA inline (see the arm).
    //
    // The odometer, not the store journal alone: a store, a list append/pop
    // and a cell store each bump it, and the store journal counts only the
    // first.  The abort path reads it for the same "did this sub-walk already
    // run an effect" question.
    let prologue_effects_before = fbw_executed_effect_count();
    // Compute fresh outer_active_boxes for the inline sub-walk when the
    // parent FBW walk carries an empty set (`dispatch_via_miframe`
    // initializes `outer_active_boxes: Vec::new()`; it is computed
    // dynamically per guard by the FBW path).  A callee guard falls
    // through to the per-opcode arm path which reads `ctx.outer_active_boxes`,
    // so an empty inherited set produces a resume snapshot with zero frame
    // boxes while the decoder expects the full liveness-derived set — the
    // same defect class as the LOAD_ATTR fold empty-boxes bug.  Mirror
    // `try_walker_list_append_inline`: read the caller's live register
    // banks from `fbw_mode.snapshot_sym` at the CALL-site py_pc.
    //
    // The snapshot header coordinate (`sub_wc.entry_py_pc` /
    // `sub_wc.outer_jitcode_index`, stamped below) MUST be the SAME coordinate
    // these boxes are collected at.  A callee guard that collapses to the
    // caller boundary stamps that coordinate as its resume `SnapshotFrame`
    // header, and the decoder (`setup_bridge_sym`) reads the liveness window at
    // that header to size and place the stored boxes
    // (`reg_indices.total_len() == frame.values.len()`).  Collecting boxes at
    // the CALL site but stamping the walk-entry header desyncs the two windows
    // (count-mismatch assert / wrong slot layout), so carry the box coordinate
    // to the header alongside the boxes.
    // The coordinate published onto the outer portal frame's `last_instr`
    // while a residual runs inside the callee.  `op.pc` indexes the snapshot
    // sym's jitcode only while the walk is the portal's own; one level down it
    // is the INTERMEDIATE callee's offset, and mapping it through the portal's
    // pc tables stamps whatever py_pc that byte happens to land on onto a frame
    // that is still paused at the CALL the outermost sub-walk entered under —
    // which is exactly what the inherited coordinate already names.
    let inherited_caller_py_pc = if ctx.fbw_mode.inline_subwalk {
        ctx.fbw_mode.inline_caller_py_pc
    } else {
        inline_caller_py_pc_from_snapshot(ctx, op.pc)
    };
    let (
        inline_outer_active_boxes,
        inline_outer_entry_py_pc,
        inline_outer_jc_index,
        inline_outer_resume_marker_jit_pc,
        inline_caller_py_pc,
    ) = if ctx.frame_state.borrow().outer_active_boxes.is_empty() {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        if sym_ptr.is_null() {
            (
                ctx.frame_state.borrow().outer_active_boxes.clone(),
                ctx.entry_py_pc,
                ctx.outer_jitcode_index,
                ctx.outer_resume_marker_jit_pc,
                inherited_caller_py_pc,
            )
        } else {
            let sym = unsafe { &*sym_ptr };
            if sym.jitcode().is_null() {
                (
                    ctx.frame_state.borrow().outer_active_boxes.clone(),
                    ctx.entry_py_pc,
                    ctx.outer_jitcode_index,
                    ctx.outer_resume_marker_jit_pc,
                    inherited_caller_py_pc,
                )
            } else {
                // Liveness coordinate is the CALL op's own (jitcode index,
                // py_pc) — NOT the `ctx` sentinels.  `dispatch_via_miframe`
                // initializes `ctx.outer_jitcode_index` to 0 and
                // `ctx.entry_py_pc` to the walk-entry py_pc, so for a CALL in
                // a non-root jitcode, or a CALL not at the walk-entry pc,
                // those select the wrong liveness window and the callee guard
                // snapshot encodes the wrong frame boxes.  Derive the
                // coordinate from the snapshot sym's jitcode at the CALL op's
                // pc, matching `orthodox_list_append_commit`.
                let (call_site_jc_index, call_site_marker, call_site_py_pc) = unsafe {
                    let jc = &*sym.jitcode();
                    let jc_index = jc.index as u32;
                    let py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(
                        &jc.payload.metadata,
                        op.pc,
                    );
                    (
                        jc_index,
                        jc.payload.resume_marker_for_jitcode_pc(op.pc),
                        py_pc as u32,
                    )
                };
                let call_site_word = match call_site_marker {
                    Some(m) => m as i32,
                    None => majit_ir::resumedata::NO_JITCODE_PC,
                };
                let boxes = collect_outer_active_boxes(
                    sym,
                    ctx.trace_ctx,
                    ctx.registers_i,
                    ctx.registers_r,
                    ctx.registers_f,
                    call_site_jc_index,
                    false,
                    call_site_word,
                    // Keep the marker for the liveness-bank query, but key
                    // entry metadata to the raw CALL offset that produced the
                    // pre-adjustment Python coordinate.
                    op.pc as i32,
                    OuterActiveBoxesEntryTwin::Plain,
                    "call_site_capture",
                    None,
                    &[],
                    None,
                );
                (
                    boxes,
                    EntryPyPc::Jit(op.pc),
                    call_site_jc_index,
                    call_site_marker,
                    Some(call_site_py_pc),
                )
            }
        }
    } else {
        // No CALL-site coordinate is derived in these fallbacks; the outer
        // coordinate is inherited from `ctx` verbatim, so the twin is too
        // (e.g. an inline CALL inside a carrier sub-walk whose outer
        // coordinate is the paused root).
        (
            ctx.frame_state.borrow().outer_active_boxes.clone(),
            ctx.entry_py_pc,
            ctx.outer_jitcode_index,
            ctx.outer_resume_marker_jit_pc,
            inherited_caller_py_pc,
        )
    };
    // `executioncontext.py enter` — emitted here, past every decline gate,
    // so a callee that never runs leaves no half-entered chain behind.  A
    // admitted level has the real frame object seeded above, matching
    // upstream's app-level frame construction, so `enter` runs on that same
    // object. The frame `perform_call` →
    // `newframe` pushes (`pyjitpl.py, 1862-1874`) is the tracer's
    // `MIFrame` instead, and is not what `enter` takes its vref of.
    let entered_ec = if callee_frame_seeded && !ca_concrete_frame.is_null() {
        let concrete_ec = pyre_interpreter::call::getexecutioncontext()
            as *mut pyre_interpreter::PyExecutionContext;
        if concrete_ec.is_null() {
            None
        } else {
            walker_ec_enter(
                ctx.trace_ctx,
                ca_callee_frame,
                ca_callee_ec,
                ca_concrete_frame,
                concrete_ec,
            );
            // This inlined level is an activation `execute_frame` would have
            // charged the recursion counter for.  Counting it at RUN time is
            // what a recorded call would do, and that is exactly wrong here: a
            // guard leaving this callee skips the recorded release the same
            // way it skips the `ec.topframeref` restore below, and a loop that
            // keeps running inside one portal activation never reaches the
            // boundary that repairs it — the depth then climbs until an
            // unrelated call raises `RecursionError`.
            //
            // So the level is counted at RECORD time instead, as a width the
            // seam that does survive to run time — the `CALL_ASSEMBLER` fold,
            // which runs once per fragment entry — charges in one go.  The
            // guard makes the width balanced across every early return between
            // here and the `leave`.
            Some(OpenInlineActivation::open(ctx.session))
        }
    } else {
        None
    };
    // The paused caller's banks live on `InlineParentFrame` (`attach_live_caller`
    // in `compute_*_caller_frame`). `replace_box` walks `framestack` the way
    // `MetaInterp.replace_box` walks `MIFrame`s; no Weak inbox is needed here.
    // Bracket the session-wide exception slot around the callee so a NULL
    // return only reads as a raise when the callee installed the exception.
    // Nothing between here and the sub-walk writes the slot.
    let exc_before_subwalk = ctx.last_exc_value();
    let (mut callee_outcome, callee_class_of_last_exc_is_const) = {
        {
            let parent_state = ctx.frame_state.borrow();
            let mut state = callee_state.borrow_mut();
            state.callee_shadow = Some(super::CalleeLocalsShadow {
                code_ptr: raw_callee_code,
                ..Default::default()
            });
            state.current_exception_seed = parent_state.current_exception_seed;
            state.current_exception_seed_concrete = parent_state.current_exception_seed_concrete;
            state.outer_active_boxes = inline_outer_active_boxes;
        }
        let mut sub_wc = WalkContext {
            frame_state: callee_state,
            // Path-1: resolve scalar static-field reads off this callee's own
            // unseeded portal frame to its compile-time constants.
            inline_callee_consts: Some(inline_consts),
            // Guards emitted inside the callee body — both the walker's own
            // and the `_nonstandard_virtualizable` PTR_EQ promote that
            // `vable_getfield_*` records internally — resume at this CALL
            // boundary (`sub_wc.entry_py_pc` / `outer_active_boxes`, both
            // stamped at the CALL-site coordinate above), not at a callee
            // `op_pc` that has no meaning in the outer jitcode's py_pc→jitcode
            // tables.
            fbw_mode: FbwWalkMode {
                inline_subwalk: true,
                inline_caller_py_pc,
                immediate_inline_caller_py_pc: immediate_inline_caller_py_pc(ctx, op.pc),
                instance_next_foriter_green_key,
                instance_next_foriter_census_active: instance_next_seeded_route,
                // Callee MIFrame. `_interpret` runs `blackhole_if_trace_too_long`
                // after every one of its `run_one_step`s, so this walk must not
                // inherit a helper's deferred limit check.
                transparent_helper_subwalk: false,
                transparent_helper_jitcode_index: None,
                ..ctx.fbw_mode
            },
            session: ctx.session,
            registers_r: &callee_regs_r,
            registers_i: &RegisterBank::with_constants(callee_regs_i, body.num_regs_i),
            registers_f: &RegisterBank::with_constants(callee_regs_f, body.num_regs_f),

            concrete_registers_i: &mut callee_concrete_i,
            descr_refs: &callee_descr_refs,
            raw_descrs: RawDescrPool::PerFn(callee_perfn_descrs),
            is_authoritative_executor: ctx.is_authoritative_executor,
            pending_guard_snapshot_error: None,

            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,

            vstack_reorder_ceiling: u32::MAX,

            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
            trace_ctx: ctx.trace_ctx,
            is_top_level: false,
            sub_jitcode_lookup: callee_lookup,
            entry_py_pc: inline_outer_entry_py_pc,
            outer_resume_marker_jit_pc: inline_outer_resume_marker_jit_pc,
            outer_jitcode_index: inline_outer_jc_index,
        };
        // Track this callee for the lifetime of the sub-walk so nested
        // self-calls see the correct recursion depth.
        //
        // A constructor pauses TWO levels here, outermost-first: the caller at
        // its CALL, then `descr_call`'s tail, which upstream is an ordinary
        // `MIFrame` between them.  Recording it on this level is what puts it
        // at its own depth for every guard below, including one inside a
        // callee `__init__` itself inlines.
        let mut parents: Vec<InlineParentFrame> = vec![parent_frame];
        if let Some((instance, _)) = constructor_result
            && callee_frame_materialized_has_resume
        {
            let Some(tail) = super::ctor_continuation_parent_frame(instance) else {
                return Err(DispatchError::callee_inline_unsupported(op.pc));
            };
            parents.push(tail);
        }
        // An operator that post-processes the dunder's result pauses a level
        // for the same span and the same reason: its tail runs between the
        // callee's return and the caller's result slot, so without it
        // `_setup_return_value_r` hands the caller the RAW return and the
        // operator's own conversion and checks are simply skipped.
        if let Some(operator_tail) = operator_tail
            && callee_frame_materialized_has_resume
        {
            let Some(tail) = super::operator_continuation_parent_frame(operator_tail) else {
                return Err(DispatchError::callee_inline_unsupported(op.pc));
            };
            parents.push(tail);
        }
        let _inline_frame = InlineFrameGuard::enter(ctx.session, callee_code_key, true, parents);
        // Name the frame this sub-walk executes concretely, so each residual
        // it runs can `enter`/`leave` it on the interpreter frame chain.
        let _inline_concrete_frame = InlineConcreteFrameGuard::enter(concrete_callee_frame);
        if let Some(frame) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) {
            if frame.body_matches(&body) {
                seed_callee_vstack_mirror(&mut sub_wc, &frame);
            }
        }
        // Seed the callee's per-slot concrete-locals shadow from the param
        // boxes.  Two distinct consumers, gated differently:
        //
        // 1. Register-to-register fold (`!try_multiframe` only): a branchless
        //    leaf inlined without a materialized virtual frame folds its own
        //    `getarrayitem_vable_r` / `setarrayitem_vable_r` through the per-slot
        //    OpRef shadow (`fold_frame_reg` + `set_opref`), so the callee's first
        //    LOAD_FAST of a param folds to the arg OpRef instead of reading its
        //    unseeded frame box.  A `try_multiframe` callee HAS a real virtual
        //    frame, so this fold must stay off (its reads go through the frame).
        //
        // 2. Concrete-locals fallback (BOTH paths): the `getarrayitem_vable`
        //    read fallback and the `setarrayitem_vable` re-seed
        //    (`getarrayitem_vable_via_metainterp` / `setarrayitem_vable`) supply
        //    the local's recording-time concrete when the heapcache holds no
        //    entry.  A `try_multiframe` callee's param reads forward through the
        //    heapcache only until an in-callee may-force op runs
        //    `reset_keep_likely_virtuals` (heapcache.py) and clears the array
        //    cache; the post-call LOAD_FAST re-read then misses and the branch
        //    value goes non-concrete (`GotoIfNotValueNotConcrete`).  Seeding the
        //    shadow for `try_multiframe` too gives that re-read a fallback — the
        //    analog of the callee MIFrame register box RPython reads
        //    `box.getint()` off (registers survive a residual call; the heapcache
        //    does not).  STORE_FAST keeps both maps current (the
        //    `setarrayitem_vable` handler re-seeds `set_opref` + `set_concrete`
        //    on every store).
        //
        //    `set_opref` is seeded on BOTH paths (not just the fold): the read
        //    fallback re-resolves the slot's concrete through `concrete_of_opref`
        //    on this OpRef — a GC-forwarded, rooted channel — in preference to
        //    the raw `Value` copy in `concrete`, which the trace-ref walker does
        //    not visit and so dangles if a minor collection moves a nursery Ref
        //    across the may-force residual.  The fold consumer stays gated by
        //    `fold_frame_reg` (kept `!try_multiframe`), so seeding `opref` here is
        //    inert for folding on the multiframe path.
        //
        // Inert when `callee_portal_frame_reg == u16::MAX` (flip OFF / frame reg
        // unresolved).
        if callee_portal_frame_reg != u16::MAX {
            {
                let mut state = sub_wc.frame_state.borrow_mut();
                let shadow = state.callee_shadow.as_mut().unwrap();
                shadow.concrete_frame = concrete_callee_frame as usize;
                shadow.frame_box = sub_wc
                    .registers_r
                    .get(callee_portal_frame_reg as usize)
                    .expect("ref register in range");
            }
            if !try_multiframe {
                let mut state = sub_wc.frame_state.borrow_mut();
                let shadow = state.callee_shadow.as_mut().unwrap();
                shadow.fold_frame_reg = callee_portal_frame_reg;
                // The fold's premise (`setarrayitem_vable_via_metainterp`) is
                // that it writes away from an UNSEEDED portal frame — a pure
                // SSA mirror with no heap array behind it.  The seed block
                // above may have materialized a real callee `PyFrame`, whose
                // `NewArrayClear` locals array is stored into only for the
                // parameters and freevar cells; folding away the in-callee
                // STORE_FASTs would leave every other local holding the
                // zero-fill, and a frame reachable afterwards through a
                // traceback, `f_locals` or `sys._getframe` reads them as
                // unbound.  Record that here so the store handler demotes just
                // the LOCAL region to a recorded `SETARRAYITEM_GC`, which is
                // what `_opimpl_setarrayitem_vable` does for a
                // `_nonstandard_virtualizable` (`pyjitpl.py`). Recording
                // those stores also emits the promote guard in
                // `vable_getfield_*` (`pyjitpl.py`), whose resume
                // image must include the paused caller frame
                // (`opencoder.py capture_resumedata`). Every admitted inline
                // now carries one, so the remaining question is only whether
                // the callee's own frame reds were seeded; an unseeded callee
                // keeps folding.
                shadow.frame_materialized = callee_frame_materialized_has_resume;
            }
            for i in 0..seeded_locals {
                let slot = i as i64;
                let value = callee_args[i];
                let concrete = sub_wc
                    .trace_ctx
                    .concrete_of_opref(callee_args[i])
                    .unwrap_or(majit_ir::Value::Void);
                let mut state = sub_wc.frame_state.borrow_mut();
                let shadow = state.callee_shadow.as_mut().unwrap();
                shadow.set_opref(slot, value);
                shadow.set_concrete(callee_portal_frame_reg, slot, concrete);
            }
            // `MIFrame.registers_r` retains cell/freevar slots across a
            // may-force call just like ordinary locals.  The heapcache entry
            // installed by `emit_new_pyframe_inline_with_params` is only a
            // forwarding optimization and is invalidated by that call; seed
            // the frame-local shadow too so a later LOAD_DEREF re-reads the
            // same live cell instead of manufacturing an unstamped
            // GetarrayitemGcR result.  This callee shape has no fresh cellvars
            // (rejected above), so existing freevars begin at `nlocals`.
            for (i, (&cell, &value)) in concrete_freevar_cells
                .iter()
                .zip(&freevar_cell_ops)
                .enumerate()
            {
                let slot = (callee_code.varnames.len() + i) as i64;
                let mut state = sub_wc.frame_state.borrow_mut();
                let shadow = state.callee_shadow.as_mut().unwrap();
                shadow.set_opref(slot, value);
                shadow.set_concrete(
                    callee_portal_frame_reg,
                    slot,
                    majit_ir::Value::Ref(majit_ir::GcRef(cell as usize)),
                );
            }
        }
        // Capture a depth-1 live callee before these guards drop. This is the
        // two-frame specialization of `run_blackhole_interp_to_cancel_tracing`:
        // `_copy_data_from_miframe` preserves the callee's own position and
        // live registers instead of collapsing it onto the caller frame.
        // `newframe(jitcode, greenkey)` (pyjitpl.py) — the callee this
        // walk is about to inline is what upstream logs there.  The pair
        // brackets the `walk` call and nothing else: every decline is already
        // behind us and every exit below reads `result`, so the sequence
        // `find_biggest_function` pairs off cannot go out of step.  The green
        // key is the callee's function-entry key, the one
        // `disable_noninlinable_function` is applied to — `callee_green_key`
        // above, minted from the `PyCode` object.
        //
        // `w_code`, not the inner `CodeObject` `w_code_get_ptr` hands back:
        // the third green is the code OBJECT (`interp_jit.py` `greens =
        // ['next_instr', 'is_being_profiled', 'pycode']`), and every other
        // mint in the tree — the function-entry key, the back-edge merge
        // point, `can_inline_callable` and `disable_noninlinable_function`
        // right above — uses it.  Keying the log on the inner pointer put
        // `find_biggest_function`'s answer in a hash no reader computes, so
        // `note_root_trace_too_long`'s huge-function arm filed a cell that
        // `can_inline_callable` never found, the same callee was re-inlined on
        // the next attempt, and the root took neither the disable nor
        // `prepare_trace_segmenting`'s permanent stamp.
        // pyjitpl.py `perform_call` → `newframe` then `ChangeFrame`, then
        // `_interpret` runs `blackhole_if_trace_too_long` after that step.
        // The log entry must exist before the abort: a too-long check that
        // ran first left `portal_trace_positions` empty, so
        // `find_biggest_function` named no callee and
        // `prepare_trace_segmenting` wrote `JC_DONT_TRACE_HERE` on the portal.
        let subwalk_jd_no = crate::state::note_inline_subwalk_start(
            (
                callee_green_key.get_uhash(),
                // The callee's greens are in scope here, so the log carries
                // them: `disable_noninlinable_function` applies to this key,
                // and it reaches a cell.
                Some(crate::driver::make_green_key_typed(
                    w_code,
                    0,
                    is_being_profiled,
                )),
            ),
            None,
        );
        // `_interpret` checks `blackhole_if_trace_too_long` after the
        // `run_one_step` that `newframe`s, before the callee's first
        // instruction. A zero-size just-opened frame does not win
        // `find_biggest_function` (`size > max_size`).
        if sub_wc.trace_ctx.is_too_long() {
            // `blackhole_if_trace_too_long` after the `newframe` step.
            // `trace_too_long_abort_safe`: latch the live framestack, and
            // keep recording when that image cannot be built once effects
            // have already run. Returning `TraceTooLong` with effects and
            // no image is what `try_adopt_single_frame_blackhole` rejects.
            // `sub_wc` is the callee `newframe` just opened.
            // `build_multi_frame_miframe` stamps `resume_pc` onto that
            // callee's jitcode, so the coordinate is the body's entry
            // (`walk` starts at 0), not the caller's `op.pc`.
            let latched = super::residual_call::latch_abort_blackhole(&sub_wc, 0, "pre-callee");
            if !latched && fbw_executed_effect_count() != 0 {
                majit_metainterp::mc_diag_bump(26);
            } else {
                let ops = sub_wc.trace_ctx.num_recorded_ops();
                crate::state::note_root_trace_too_long(
                    sub_wc.trace_ctx.current_merge_points_first_green_key_pair(),
                    sub_wc.trace_ctx.resumekey_original_loop_token().cloned(),
                );
                sub_wc.session.borrow_mut().trace_too_long = true;
                // `enter` already ran. `SwitchToBlackhole` never reaches `leave`,
                // so restore the concrete chain without recording the finish.
                if entered_ec.is_some() {
                    let frame_ptr = sub_wc
                        .trace_ctx
                        .virtualref_entry_ptr((ca_callee_frame, ca_concrete_frame as usize))
                        as *mut pyre_interpreter::PyFrame;
                    abandon_entered_frame(
                        sub_wc.trace_ctx,
                        frame_ptr,
                        pyre_interpreter::call::getexecutioncontext()
                            as *mut pyre_interpreter::PyExecutionContext,
                    );
                }
                return Err(DispatchError::TraceTooLong { pc: op.pc, ops });
            }
        }
        let result = {
            // #704 root-bridge self-recursive inline: exempt this callee body
            // sub-walk's nested recursive residual from the self-recursive
            // nested-residual decline, mirroring the native `CALL_ASSEMBLER`
            // fold's `SELFREC_CA_FOLD_ACTIVE` exemption.
            let _bridge_rec_selfrec_guard = bridge_rec_root_selfrec.then(SelfRecCaFoldGuard::enter);
            // Root this sub-walk's own Ref bank: a callee fold mints its
            // `ConstPtr` into `sub_wc.registers_r`, which is not the anchored
            // `sym.registers_r`.
            let _callee_bank_guard =
                crate::trace::InlineRegisterBankGuard::enter(sub_wc.registers_r);
            walk(body.code, 0, &mut sub_wc)
        };
        // `blackhole_if_trace_too_long` raises out of `_interpret`. `popframe`
        // and `ExecutionContext.leave` are after that point and must not run.
        if let Err(error @ DispatchError::TraceTooLong { .. }) = result {
            if entered_ec.is_some() {
                let frame_ptr = sub_wc
                    .trace_ctx
                    .virtualref_entry_ptr((ca_callee_frame, ca_concrete_frame as usize))
                    as *mut pyre_interpreter::PyFrame;
                abandon_entered_frame(
                    sub_wc.trace_ctx,
                    frame_ptr,
                    pyre_interpreter::call::getexecutioncontext()
                        as *mut pyre_interpreter::PyExecutionContext,
                );
            }
            return Err(error);
        }
        if subwalk_jd_no.is_some() {
            crate::state::note_inline_subwalk_end();
        }
        let prologue_cannot_call_assembler = fbw_executed_effect_count() != prologue_effects_before
            || (!unjournaled_before_subwalk && fbw_has_unjournaled_effect());
        // `descr_call` is a level of its own upstream: it discards `__init__`'s
        // result, checks it is None, and returns the instance. The mid-body leg
        // resumes INSIDE the rebuilt callee and hands its return to the caller
        // PAST the CALL, and the tail (`crate::ctor_continuation`) carries no
        // Python code object, so the interpreter cannot run it as a frame
        // between the two -- the caller would take `__init__`'s None as the
        // instantiation's result. Leave the carrier to the entry leg, which
        // replays the CALL: the half-built instance is unreachable, so the
        // residual re-does the whole instantiation, the same answer the fold's
        // own decline path gives.
        let midbody_abort = if constructor_result.is_some() {
            None
        } else {
            match &result {
                Err(DispatchError::AbortPermanentMarkerReached { pc }) => {
                    Some((*pc, MidBodyAbortKind::Marker))
                }
                Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc, .. })
                    if fbw_structural_abort_opcode_is_effect_free(*pc) =>
                {
                    Some((*pc, MidBodyAbortKind::Structural))
                }
                // `run_blackhole_interp_to_cancel_tracing` keeps the live callee
                // MIFrame and continues it at its own merge point.  When a callee
                // prologue has already acquired a lock or committed another
                // effect, rejecting CALL_ASSEMBLER below must use that same
                // mid-body handoff.  Falling through without a carrier replays the
                // callee from its entry and, for `_pyio.BufferedReader.read`, tries
                // to acquire its non-reentrant `_read_lock` a second time.
                Ok((DispatchOutcome::SubLoopCalleeCallAssembler { target_pc, .. }, _))
                    if prologue_cannot_call_assembler =>
                {
                    crate::state::pyjitcode_for_code(w_code)
                        .and_then(|pjc| pjc.merge_entry_for(*target_pc))
                        .map(|pc| (pc, MidBodyAbortKind::Structural))
                }
                _ => None,
            }
        };
        if let Some((abort_pc, abort_kind)) = midbody_abort {
            if is_top_inline && !unjournaled_before_subwalk {
                crate::trace::fbw_diag::bump(crate::trace::fbw_diag::MIDBODY_LATCH);
                if fbw_has_unjournaled_effect() {
                    crate::trace::fbw_diag::bump(
                        crate::trace::fbw_diag::MIDBODY_LATCH_NEW_UNJOURNALED,
                    );
                }
            }
            // This leg resumes INSIDE the rebuilt callee, so a residual the
            // callee recorded only symbolically before the abort pc lies
            // BEHIND the resume point and the discarded trace was its only
            // carrier.  `unjournaled_before_subwalk` is sampled before the
            // sub-walk and cannot see such a mark; read the flags again, as
            // the loop-header, abort-pc and branch-guard legs do.
            //
            // Attempted whether or not the callee executed anything.
            // `convert_and_run_from_pyjitpl` (`blackhole.py`) rebuilds
            // every framestack frame at its own pc unconditionally —
            // `run_blackhole_interp_to_cancel_tracing` ends `assert False`
            // (`pyjitpl.py`) — and the caller is resumed PAST its call
            // (`blackhole.py`).  This outer gate sits OUTSIDE the
            // reason-producing closure below, so until it was instrumented
            // its two narrowings were the only ones that could keep a callee
            // off leg 4 without saying so — the census could not tell
            // "never fired" from "not measured".
            if fbw_debug_abort_enabled() && !(is_top_inline && !fbw_has_unjournaled_effect()) {
                eprintln!(
                    "[fbw-abort-flush] gh#467 callee-rebuild NOT LATCHED ({})",
                    if !is_top_inline {
                        "inline sub-walk (depth>=2)"
                    } else {
                        "unjournaled effect pending"
                    },
                );
            }
            if is_top_inline && !fbw_has_unjournaled_effect() {
                // Each refusal names itself so the debug log can say WHICH
                // narrowing keeps a callee off this leg.
                let payload = (|| {
                    let (outer_jitcode_index, call_jitcode_pc) =
                        abort_flush_call_jitcode_coord.ok_or("no call jitcode coord")?;
                    let callee_pjc =
                        crate::state::pyjitcode_for_code(w_code).ok_or("no callee pyjitcode")?;
                    let metadata = &callee_pjc.metadata;
                    let callee_py_pc =
                        crate::py_coord::containing_py_pc_for_jitcode_pc(metadata, abort_pc)
                            as usize;
                    // Both abort kinds sit at the head of an opcode the walker
                    // could not take, behind at most that opcode's own vable
                    // spill; the marker kind is the narrower of the two.
                    let anchor_ok = portal_vable_bookkeeping_anchor(
                        metadata,
                        metadata.built_as_portal,
                        metadata.portal_frame_reg,
                        callee_perfn_descrs,
                        body.code,
                        callee_py_pc,
                        abort_pc,
                        |op_pc| {
                            crate::py_coord::containing_py_pc_for_jitcode_pc(metadata, op_pc)
                                as usize
                        },
                    );
                    if !anchor_ok {
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] gh#467 inexact anchor at abort_pc={abort_pc} \
                                 callee_py_pc={callee_py_pc} kind={abort_kind:?} \
                                 portal_frame_reg={} callee_portal_frame_reg={} would re-run {:?}",
                                metadata.portal_frame_reg,
                                callee_portal_frame_reg,
                                floor_segment_ops_before(metadata, body.code, abort_pc),
                            );
                        }
                        return Err("abort pc is not an exact segment anchor");
                    }
                    let raw = unsafe {
                        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                            as *const pyre_interpreter::CodeObject
                    };
                    if raw.is_null() {
                        return Err("null callee code ptr");
                    }
                    let callee_code = unsafe { &*raw };
                    if pyre_interpreter::pyframe::code_flags_make_generator(callee_code.flags) {
                        return Err("callee is a generator");
                    }
                    if !callee_code.cellvars.is_empty() || !callee_code.freevars.is_empty() {
                        return Err("callee has cellvars/freevars");
                    }
                    if !unsafe { pyre_interpreter::function_get_closure(callable) }.is_null() {
                        return Err("callee has a closure");
                    }
                    let depth = callee_pjc
                        .depth_for_jitcode_pc_pred(abort_pc)
                        .ok_or("no stack depth for the abort pc")?
                        as usize;
                    let nlocals = callee_code.varnames.len();
                    let entries = callee_pjc
                        .pcdep_for_jitcode_pc(abort_pc)
                        .ok_or("no pcdep entries for the abort pc")?;
                    let mut live_stack = Vec::with_capacity(depth);
                    for rel in 0..depth {
                        let semantic_slot = nlocals + rel;
                        let register_value =
                            crate::state::semantic_slot_color_for_ref_slot(&entries, semantic_slot)
                                .and_then(|color| {
                                    sub_wc
                                        .frame_state
                                        .borrow()
                                        .concrete_registers_r
                                        .get(color)
                                        .copied()
                                });
                        let value = register_value
                            .or_else(|| {
                                (metadata.built_as_portal && abort_kind == MidBodyAbortKind::Marker)
                                    .then(|| {
                                        callee_vable_ref_at(
                                            sub_wc.frame_state.borrow().callee_shadow.as_ref(),
                                            metadata.portal_frame_reg,
                                            semantic_slot,
                                        )
                                    })
                                    .flatten()
                            })
                            .ok_or("live stack slot has no concrete value")?;
                        if !matches!(value, ConcreteValue::Ref(r) if !r.is_null()) {
                            return Err("live stack slot is not a non-null Ref");
                        }
                        live_stack.push(value);
                    }
                    if live_stack.len() != depth {
                        return Err("live stack depth mismatch");
                    }
                    let lv = crate::state::liveness_for(raw);
                    let mut live_locals = vec![None; nlocals];
                    for (slot, dst) in live_locals.iter_mut().enumerate() {
                        if !lv.is_local_live(callee_py_pc, slot) {
                            continue;
                        }
                        let value = sub_wc
                            .frame_state
                            .borrow()
                            .callee_shadow
                            .as_ref()
                            .and_then(|shadow| shadow.concrete.get(&(slot as i64)).copied())
                            .filter(|entry| entry.frame_reg == metadata.portal_frame_reg)
                            .and_then(|entry| match entry.value {
                                Value::Ref(r) => Some(ConcreteValue::Ref(
                                    r.as_usize() as pyre_object::PyObjectRef
                                )),
                                Value::Int(v) => Some(ConcreteValue::Int(v)),
                                Value::Float(v) => Some(ConcreteValue::Float(v)),
                                Value::Void => None,
                            })
                            .or_else(|| callee_arg_concretes.get(slot).copied())
                            .ok_or("live local has no concrete value")?;
                        if matches!(value, ConcreteValue::Null | ConcreteValue::Bool(_)) {
                            return Err("live local is Null/Bool");
                        }
                        *dst = Some(value);
                    }
                    // A constructor's CALL result is the instance, and the
                    // rebuild delivers the callee's own return.  Carry the
                    // instance so the flush can play `type_descr_call_impl`'s
                    // tail; without a concrete Ref to carry there is nothing to
                    // substitute and `__init__`'s `None` would reach the
                    // caller, so refuse the rebuild instead.
                    let constructor_instance = match constructor_result {
                        None => pyre_object::PY_NULL,
                        Some((_, ConcreteValue::Ref(instance))) if !instance.is_null() => instance,
                        Some(_) => return Err("constructor instance has no concrete Ref"),
                    };
                    Ok(MidBodyPayload {
                        abort_kind,
                        outer_jitcode_index,
                        call_jitcode_pc,
                        call_stack_len: arg_concretes.len(),
                        callee_jitcode_index: callee_pjc.jitcode.index() as u32,
                        abort_jitcode_pc: abort_pc,
                        callee_py_pc,
                        w_code: w_code as pyre_object::PyObjectRef,
                        w_globals: unsafe { pyre_interpreter::function_get_globals_obj(callable) },
                        live_locals,
                        live_stack,
                        return_value: pyre_object::PY_NULL,
                        constructor_instance,
                    })
                })();
                match payload {
                    Ok(payload) => fbw_set_midbody_abort_resume(payload),
                    Err(reason) => {
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] gh#467 callee-rebuild NOT LATCHED at \
                                 abort_pc={abort_pc} ({reason})"
                            );
                        }
                    }
                }
            }
        }
        let class_of_last_exc_is_const = sub_wc.fbw_mode.class_of_last_exc_is_const;
        (result, class_of_last_exc_is_const)
    };
    // `executioncontext.py leave`, in the original's `finally`
    // position: the sub-walk block above is an expression that always
    // completes, so every callee exit — return, exception, or decline —
    // arrives here before any of the early returns below.
    //
    // A callee that stopped at its OWN loop header has not returned, so this
    // `leave` is recorded before the `CALL_ASSEMBLER` that runs the rest of it,
    // and a loop body reading the live frame (`sys._getframe()`, a traceback)
    // names the caller rather than its own frame.  MEASURED 2026-08-27: moving
    // the `leave` past that op — the obvious fix — hangs
    // `synth/exception_traceback_frame_lineno` (dynasm, 6/6; the deferral
    // switched off in the same binary is 3/3 clean).  The guards between the
    // two, and `GUARD_NO_EXCEPTION` above all in a callee that raises every
    // iteration, leave the trace before the deferred `leave` is reached, so
    // `ec.topframeref` keeps the callee and the frame chain never unwinds.
    // Converging needs the `leave` reachable from those guard exits — a
    // resume-side leave, or the exception path recording its own — not a
    // reorder.
    if let Some(open_activation) = entered_ec {
        let concrete_ec = pyre_interpreter::call::getexecutioncontext()
            as *mut pyre_interpreter::PyExecutionContext;
        // `leave(frame, w_exitvalue, got_exception)` — the caller passes true
        // only when the frame is unwinding an exception, which for an inlined
        // callee is `SubRaise` and nothing else.  A tracing decline (`Err`) or
        // a loop transition is not an exception exit: treating it as one would
        // permanently `mark_as_escaped` the caller and force a vref that never
        // needed forcing.
        let got_exception = matches!(callee_outcome, Ok((DispatchOutcome::SubRaise { .. }, _)));
        // `SubRaise`'s concrete word is a Rust copy. `leave` forces the
        // frame vref (`force_vref`). `registers_r` is not a root. Publish
        // the copy for the leave and write the forwarded address back
        // before `record_prepend_application_traceback` reads it.
        let raised_ptr = if let Ok((
            DispatchOutcome::SubRaise {
                exc_concrete: ConcreteValue::Ref(exc_ptr),
                ..
            },
            _,
        )) = &callee_outcome
        {
            (!exc_ptr.is_null()).then_some(*exc_ptr)
        } else {
            None
        };
        let _raise_roots = raised_ptr.map(|_| pyre_object::gc_roots::push_roots());
        let raise_slot = raised_ptr.map(|exc_ptr| {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(exc_ptr);
            slot
        });
        walker_ec_leave(
            ctx.trace_ctx,
            ca_callee_frame,
            ca_callee_ec,
            ca_concrete_frame,
            concrete_ec,
            got_exception,
        );
        if let Some(slot) = raise_slot {
            let live = pyre_object::gc_roots::shadow_stack_get(slot);
            if let Ok((DispatchOutcome::SubRaise { exc_concrete, .. }, _)) = &mut callee_outcome {
                let old = *exc_concrete;
                *exc_concrete = ConcreteValue::Ref(live);
                if ctx.last_exc_value_concrete() == old {
                    ctx.set_last_exc_value_concrete(ConcreteValue::Ref(live));
                }
            }
        }
        drop(open_activation);
    }
    // RPython has one MetaInterp shared by every MIFrame.  The sub-walk uses
    // a copied FbwWalkMode only to satisfy Rust's nested borrow, so write the
    // MetaInterp-owned exception state back across the frame boundary.
    ctx.fbw_mode.class_of_last_exc_is_const = callee_class_of_last_exc_is_const;
    let (outcome, _end_pc) = match callee_outcome {
        Ok(v) => v,
        Err(e) => {
            if fbw_inline_diag_enabled() {
                eprintln!("[inline-abort] callee sub-walk err: {e:?}");
            }
            // A seeded inline abort latches the live framestack in `walk`
            // and converts through `convert_and_run_from_pyjitpl`
            // (`blackhole.py`).  Returning the error is what lets that latch
            // see the abort class.
            return Err(e);
        }
    };

    match promote_published_null_return_since(ctx, outcome, op.pc, exc_before_subwalk)? {
        DispatchOutcome::SubReturn { result } => match finish_inline_callee_return(ctx, result) {
            Some(value) => {
                let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
                if require_str_result
                    && !matches!(
                        concrete_for_shadow,
                        ConcreteValue::Ref(obj) if !obj.is_null() && unsafe { pyre_object::is_str(obj) }
                    )
                {
                    if !entry_is_call_boundary
                        && fbw_executed_effect_count() == executed_effects_before
                    {
                        // A descriptor opcode such as FORMAT_WITH_SPEC owns
                        // its result check and has no Python CALL boundary to
                        // latch.  With no concrete effect, let its caller
                        // rewind the attempt and emit the faithful residual.
                        return resolved_inline_decline(op.pc, line!());
                    }
                    // descroperation.py checks the app-level result before
                    // returning from `space.str` / `space.repr`. Abort so
                    // `convert_and_run_from_pyjitpl` (`blackhole.py`) raises
                    // the faithful TypeError from the live framestack.
                    return Err(DispatchError::callee_inline_unsupported(op.pc));
                }
                if require_exact_int_result
                    && !matches!(
                        concrete_for_shadow,
                        ConcreteValue::Ref(obj)
                            if walker_is_exact_machine_int_concrete(obj)
                    )
                {
                    // `descroperation.py _index` validates the app-level
                    // result before its caller continues, and a long, a bool or an
                    // int subclass are all legal there — only the machine-int
                    // arithmetic downstream cannot take them.  Decline instead of
                    // aborting: the caller rewinds the emission and falls through
                    // to its residual, which re-runs the whole builtin.  That is
                    // sound because the body is admitted only when re-running it
                    // observes and changes nothing (`exc_override_sample_safe`),
                    // and it keeps a legal program from killing the enclosing
                    // loop's trace, which `callee_inline_unsupported` would.
                    return resolved_inline_decline(op.pc, line!());
                }
                if operator_tail == Some(crate::operator_continuation::OperatorTail::Format)
                    && fbw_executed_effect_count() == executed_effects_before
                {
                    // `descroperation.py format` checks `isinstance_w(w_res,
                    // w_unicode)` after `__format__` returns. Pin the observed
                    // str class before the destination write so a later
                    // non-str deopts with the FORMAT_WITH_SPEC result slot
                    // still pending; `OperatorTail::Format` is the resume
                    // level that runs the same check when a guard inside the
                    // inlined body fails (`crate::operator_continuation`).
                    // This GuardClass itself resumes at `op.pc`, so a deopt
                    // re-runs FORMAT_WITH_SPEC. Emit it only over a body that
                    // committed nothing. An effectful body cannot decline
                    // here: `cut_declined_subwalk` drops recorded ops and
                    // does not undo the effect, so the FORMAT_WITH_SPEC
                    // residual would run `__format__` again. Skip the pin
                    // and keep the recorded prefix.
                    let concrete = match concrete_for_shadow {
                        ConcreteValue::Ref(obj) if !obj.is_null() => obj,
                        _ => return resolved_inline_decline(op.pc, line!()),
                    };
                    let result_type = unsafe { (*concrete).ob_type } as i64;
                    let result_type_const = ctx.trace_ctx.const_int(result_type);
                    walker_emit_guard_with_snapshot(
                        ctx,
                        op.pc,
                        OpCode::GuardClass,
                        &[value, result_type_const],
                    )?;
                    ctx.trace_ctx.heap_cache_mut().class_now_known(value);
                }
                if operator_tail == Some(crate::operator_continuation::OperatorTail::Len) {
                    // `len` runs `space.index` on what `__len__` returned,
                    // then `_check_len_result` on that, and boxes the machine
                    // length it checked out to (`builtins.rs builtin_len`).
                    // An exact machine int makes the index the identity,
                    // reduces the check to its nonnegative test, and makes the
                    // box an int of the same value, so admit that one shape
                    // and guard it at runtime below.  Every other box — a
                    // long, a bool, an int subclass, something carrying
                    // `__index__`, and a negative — hands the call back to the
                    // interpreter, which runs the whole operator and produces
                    // the faithful conversion, TypeError, ValueError or
                    // OverflowError.
                    let len_value = match concrete_for_shadow {
                        ConcreteValue::Ref(obj) if walker_is_exact_machine_int_concrete(obj) => {
                            walker_machine_int_value(obj)
                        }
                        _ => None,
                    };
                    // Two refusals, one exit.  A length this route cannot
                    // answer — a long, a bool, an int subclass, something
                    // carrying `__index__`, a negative — owes the interpreter
                    // its faithful conversion or error.  A body whose sub-walk
                    // moved the executed-effect odometer owes it the operator
                    // too: the guard below resumes at the CALL boundary with
                    // the result slot pending, so the blackhole re-runs the
                    // whole operator, `__len__` included, and the odometer is
                    // what says that re-run is not free.  The traced prefix is
                    // the only prefix a deopt can have, so it answers the
                    // question exactly, and the guard is emitted only over a
                    // body that committed nothing.
                    //
                    // Both leave by declining, not aborting.  Declining cuts
                    // the emission and lets the caller's residual run the
                    // whole operator; aborting would ban the enclosing loop
                    // for a `__len__` that answers this way every iteration,
                    // which is worse than the residual this route was opened
                    // to replace.
                    if !len_value.is_some_and(|value| value >= 0)
                        || fbw_executed_effect_count() != executed_effects_before
                    {
                        return resolved_inline_decline(op.pc, line!());
                    }
                    // Both checks as guards, before the destination write, so a
                    // later iteration whose `__len__` answers differently
                    // resumes with the CALL's result slot still pending rather
                    // than carrying a length the operator would have refused.
                    let concrete = match concrete_for_shadow {
                        ConcreteValue::Ref(obj) => obj,
                        _ => pyre_object::PY_NULL,
                    };
                    let (int_type, intval_descr) =
                        crate::state::int_or_bool_unbox_type_descr(concrete);
                    let expected_class = walker_numeric_builtin_class(concrete);
                    let raw = walker_unbox_int_exact(
                        ctx,
                        op.pc,
                        value,
                        int_type,
                        intval_descr,
                        expected_class,
                    )?;
                    let zero = ctx.trace_ctx.const_int(0);
                    let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw, zero]);
                    ctx.trace_ctx
                        .set_opref_concrete(nonnegative, majit_ir::Value::Int(1));
                    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[nonnegative])?;
                }
                // `descr_call` discards `__init__`'s result after checking it is
                // None and returns the instance instead (`check_init_returned_none`).
                // A non-None result is a TypeError the inlined body cannot raise, so
                // abort and let `convert_and_run_from_pyjitpl` (`blackhole.py`)
                // raise the faithful message from the live framestack.
                let (value, concrete_for_shadow) = match constructor_result {
                    Some(instance) => {
                        if !matches!(concrete_for_shadow,
                        ConcreteValue::Ref(obj) if unsafe { pyre_object::is_none(obj) })
                        {
                            return Err(DispatchError::callee_inline_unsupported(op.pc));
                        }
                        instance
                    }
                    None => (value, concrete_for_shadow),
                };
                // `is_true` after `__bool__` answers an int, not the bool
                // object. A non-bool owes the TypeError, and a body that
                // already ran an effect cannot be cut and re-executed by the
                // Truth residual. Both decline, the same way `Len` does.
                let (value, concrete_for_shadow) =
                    if operator_tail == Some(crate::operator_continuation::OperatorTail::Truth) {
                        match truth_bool_result_int(
                            ctx,
                            op.pc,
                            value,
                            concrete_for_shadow,
                            executed_effects_before,
                        )? {
                            Some(pair) => pair,
                            None => return resolved_inline_decline(op.pc, line!()),
                        }
                    } else {
                        (value, concrete_for_shadow)
                    };
                if let Some(result) = intermediate_result {
                    // The caller stays pinned at its own CALL boundary, so a guard
                    // it emits after this hand-off resumes by re-entering that CALL
                    // and running the callee a second time, and its rewinding
                    // declines cut the trace without undoing what the body already
                    // did.  `FBW_EXECUTED_EFFECT_COUNT` is the odometer that
                    // answers whether that is survivable: "a nonzero count delta
                    // means the callee attempt cannot be discarded and re-executed
                    // without risking a double".  Abort rather than decline so
                    // `convert_and_run_from_pyjitpl` (`blackhole.py`) resumes
                    // the live framestack past the call.
                    if fbw_executed_effect_count() != executed_effects_before {
                        return Err(DispatchError::callee_inline_unsupported(op.pc));
                    }
                    *result = Some((value, concrete_for_shadow));
                    return Ok(Some((DispatchOutcome::Continue, op.next_pc)));
                }
                match dst_bank {
                    'r' => write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?,
                    'i' => write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?,
                    'v' => {}
                    _ => return Ok(None),
                }
                Ok(Some((DispatchOutcome::Continue, op.next_pc)))
            }
            None => {
                if dst_bank == 'v' {
                    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
                } else {
                    Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
                }
            }
        },
        DispatchOutcome::SubRaise {
            exc,
            mut exc_concrete,
        } => {
            // Ahead of the traceback nodes, matching the interpreter order:
            // `getattr_str` enriches as the error leaves the attribute
            // dispatch, before the caller frame's `handle_exception` records a
            // node for the opcode that made the call.
            if let Some(attr) = attribute_error_context.as_ref() {
                record_inline_attribute_error_context(ctx, op.pc, exc, exc_concrete, attr)?;
            }
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let node = record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc)?;
                let emit_runtime = node.is_none();
                record_inline_exception_context(ctx.trace_ctx, exc, exc_concrete);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                ctx.set_last_exc_value(exc, exc_concrete);
                Ok(Some((DispatchOutcome::Continue, target)))
            } else {
                Ok(Some((
                    DispatchOutcome::SubRaise { exc, exc_concrete },
                    op.next_pc,
                )))
            }
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { token, target_pc } => {
            // `descr_call`'s tail has no place in this fold.  The emit hands
            // the caller `ca_result`, the compiled `__init__`'s own return,
            // and for a constructor the CALL has to evaluate to the instance
            // instead -- the discard, the None check and the `w_newobject`
            // answer that `type_descr_call_impl` performs between the two
            // frames.  The three legs that DO play it each own a resume
            // coordinate to play it at ([`crate::ctor_continuation`] for the
            // blackhole, `bridge_subwalk` for a bridge,
            // `MidBodyPayload::constructor_instance` for the gh#467 rebuild);
            // `CALL_ASSEMBLER` has none, and its `GUARD_NOT_FORCED` would
            // resume the forced callee straight into the caller's call-result
            // slot as well.  Residualize the whole instantiation, which is
            // what `do_residual_call` (`pyjitpl.py`) does with a callee the
            // tracer cannot follow: `type_descr_call_impl` then runs the tail
            // for real, TypeError included.
            //
            // CONVERGENCE PATH: give the fold the tail as a paused level, the
            // way the inline path pushes it on `InlineFrame::parents`, and the
            // fold can be admitted for a constructor too.
            if constructor_result.is_some() {
                return resolved_inline_decline(op.pc, line!());
            }
            // A nested inline that inherited `carrier_resume` can surface this
            // with no token. Residualize the original CALL. The bridge drain
            // is the consumer that records `direct_call_may_force`.
            let Some(token) = token else {
                return resolved_inline_decline(op.pc, line!());
            };
            emit_walker_loop_callee_call_assembler(
                ctx,
                op,
                dst_bank,
                dst,
                ca_callee_frame,
                ca_callee_ec,
                ca_nlocals,
                token,
                target_pc,
                w_code,
                is_being_profiled,
            )
        }
        other => Ok(Some((other, op.next_pc))),
    }
}

/// Report why the instantiation emit declined, under `PYRE_FBW_INLINE_DIAG`.
fn type_call_decline(reason: &str) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if fbw_inline_diag_enabled() {
        eprintln!("[type-call-decline] {reason}");
    }
    Ok(None)
}

/// Instantiate a user-defined class inside the trace instead of leaving `P()`
/// an opaque `bh_call_fn` residual that re-enters `type_descr_call_impl`,
/// `object.__new__` and an interpreted `__init__` frame every iteration.
///
/// This is the walker counterpart of `typeobject.py descr_call`: promote the
/// class, run `__new__`, then `__init__` with the fresh instance as `self`.
/// Only the shape where `__new__` is `object`'s is emitted — that is exactly
/// the case whose allocation `object_descr_new` performs unconditionally
/// (`w_instance_new`), so the emitted `NewWithVtable` + header/`map` stores
/// reproduce it field for field.  A class that overrides `__new__`, is
/// abstract, refuses instantiation, or carries `__del__` (whose instances
/// `w_instance_new` registers on the finalizer queue) declines to the residual.
///
/// The payoff is not the removed dispatch alone: an instance built by
/// `new_with_vtable` is a virtual, so a constructor whose result never escapes
/// the loop optimizes away entirely, as it does upstream.
///
/// This applies at every inline depth.  `MetaInterp.perform_call` gives each
/// Python call its own `MIFrame`, and `capture_resumedata` preserves the whole
/// framestack; being reached from another inlined Python frame is therefore not
/// a reason to leave `type.__call__` residual.  Pyre's matching shape is the
/// callee-owned frame plus the `descr_call` parent level installed by
/// `ctor_continuation_parent_frame`, which keeps the instance (rather than
/// `__init__`'s return) live across a guard resume.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_type_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        // These two reject far more calls than the instantiations this emit is
        // about, so name the reason only for a call that does resolve to a
        // class, and only while the reasons are being collected — the extra
        // resolution below is diagnostic cost, not tracing cost.
        if fbw_inline_diag_enabled()
            && r_args.len() >= 2
            && walker_concrete_ref_object(ctx, r_args[1]).is_none()
            && walker_concrete_ref_object(ctx, r_args[0])
                .is_some_and(|w_type| unsafe { pyre_object::is_type(w_type) })
        {
            return type_call_decline(if !ctx.is_authoritative_executor {
                "not the authoritative executor"
            } else {
                "destination is not a ref register"
            });
        }
        return Ok(None);
    }
    // `[callable, null_or_self, args...]`.  A method-form call (`null_or_self`
    // populated) never names a class as its callable.
    if r_args.len() < 2 {
        return Ok(None);
    }
    // "No receiver" has two spellings in that slot: `call_kw` leaves it with no
    // concrete shadow at all, `call_fn` fills it with the checked `PY_NULL`
    // sentinel.  Reading a present shadow as a receiver rejects the whole
    // `call_fn` spelling, which is the one an ordinary `C(...)` lowers to.
    if walker_concrete_ref_object(ctx, r_args[1])
        .is_some_and(|null_or_self| !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL)
    {
        return type_call_decline("receiver slot is populated");
    }
    let Some(w_type) = walker_concrete_ref_object(ctx, r_args[0]) else {
        // Whether this even was an instantiation is unknowable without the
        // callable, so the reason is reported as the open question it is.
        return type_call_decline("callable is not a concrete ref");
    };
    if !unsafe { pyre_object::is_type(w_type) } {
        return Ok(None);
    }
    // `type(x)` is the one-argument class introspection shortcut, not an
    // instantiation (`type_call_type_x_shortcut`). Three-argument
    // `type(name, bases, dict)` is `descr__new__` (`type_descr_new`), walked
    // by the builtin-gateway arm rather than this object-`__new__` emit.
    let w_metatype = pyre_interpreter::typedef::w_type();
    if std::ptr::eq(w_type, w_metatype) {
        return type_call_decline(if r_args.len() == 3 {
            "type(x) shortcut"
        } else {
            "type(name, bases, dict) is descr__new__"
        });
    }
    // What follows is `type.__call__`.  A metaclass that overrides `__call__`
    // runs instead of it and may return anything at all, so it stays residual.
    //
    // The question is which `__call__` the metatype resolves to, not whether it
    // is `type` itself: `ABCMeta` supplies `__instancecheck__`,
    // `__subclasscheck__` and `register` and leaves `__call__` alone, so every
    // class that registers with a `numbers` / `collections.abc` ABC — which is
    // every `Fraction`, `Decimal` and `deque` construction — resolves to the
    // same `type.__call__` a plain class does.  Comparing the metatype's
    // identity refused all of them.
    let w_metaclass = unsafe { (*w_type).w_class };
    let metaclass_to_pin = if std::ptr::eq(w_metaclass, w_metatype) {
        None
    } else {
        let meta_call =
            unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_metaclass, "__call__") };
        let type_call =
            unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_metatype, "__call__") };
        if meta_call != type_call {
            return type_call_decline("metaclass overrides __call__");
        }
        // The answer is a lookup, so a metaclass `__call__` parked in a cell
        // could be rebound in place after this comparison without moving the
        // tag that is about to pin it.
        if unsafe { type_attr_is_cell_backed(w_metaclass, "__call__") } {
            return type_call_decline("metaclass __call__ is in a mutable cell");
        }
        // The answer above is a dict lookup, so it needs the same pin the
        // `__new__` / `__init__` answers get.  A metaclass whose dict changes
        // are untracked cannot supply one.
        if unsafe { pyre_object::typeobject::w_type_get_version_tag(w_metaclass) } == 0 {
            return type_call_decline("metaclass has no version tag");
        }
        Some(w_metaclass)
    };
    // A version tag of 0 is a type whose dict changes are not tracked, so the
    // `__new__` / `__init__` / `__del__` lookups below cannot be pinned.
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return type_call_decline("no version tag");
    }
    if unsafe {
        pyre_object::w_type_disallows_instantiation(w_type)
            || pyre_object::w_type_is_abstract(w_type)
            || pyre_object::typeobject::w_type_get_hasuserdel(w_type)
    } {
        return type_call_decline("not instantiable, abstract, or has __del__");
    }
    // `Py_tp_vectorcall` replaces the whole of `type.__call__` for the class
    // that declares it: the body this emit builds -- allocate, then run
    // `__init__` -- is not what such a call does at all.
    if unsafe { pyre_object::typeobject::w_type_has_vectorcall(w_type) } {
        return type_call_decline("type declares a vectorcall");
    }
    // `map` is baked as a constant, so install the terminator now if the type
    // has not been asked for one yet — a class nobody has read an attribute off
    // still carries null, and baking that would hand every traced instance a
    // map the interpreter's own instances do not have.  The terminator is
    // created once and never replaced, so the constant stays valid.
    let terminator =
        unsafe { pyre_interpreter::objspace::std::mapdict::ensure_type_terminator(w_type) };
    if terminator.is_null() {
        return type_call_decline("no terminator");
    }
    let w_object = pyre_interpreter::typedef::w_object();
    if w_object.is_null() {
        return type_call_decline("object type unavailable");
    }
    // Only `object.__new__` allocates the plain `[ob_type | w_class | map |
    // storage]` instance this emit builds; any other `__new__` picks its own
    // layout (a builtin subclass) or runs arbitrary code.
    //
    // A `__new__` written in Python is not that -- it ends in
    // `object.__new__(cls)`, which [`try_walker_inline_object_new`] folds --
    // but walking it from here in place of the allocation, and letting its
    // return value stand for the whole of `type.__call__`, was measured to
    // produce a wrong answer (`TypeError: 'NewOv' object is not callable`, an
    // instance reaching the callable slot) and is not what this does.
    let tp_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__new__") };
    let obj_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__new__") };
    if tp_new != obj_new {
        return type_call_decline("__new__ overridden");
    }
    let tp_init = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__init__") };
    let obj_init = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__init__") };
    let init_override = (tp_init != obj_init).then_some(tp_init).flatten();
    // Both answers are baked by identity -- `__new__` as "object's, so the
    // allocation is this emit's", `__init__` as the body walked below -- and
    // this route has no receiver guard to carry a promote on.
    if unsafe { type_attr_is_cell_backed(w_type, "__new__") }
        || unsafe { type_attr_is_cell_backed(w_type, "__init__") }
    {
        return type_call_decline("__new__/__init__ is in a mutable cell");
    }
    // `object.__new__`/`object.__init__` both reject surplus arguments when
    // neither is overridden; leave that TypeError to the interpreter.
    if init_override.is_none() && r_args.len() != 2 {
        return type_call_decline("surplus args without __init__");
    }
    let inlinable_init = match init_override {
        Some(init) => match unsafe { resolve_inlinable_callee(init) } {
            Some(resolved) => Some((init, resolved)),
            // An overridden `__init__` that is not a plain Python function
            // (a builtin, or a callable object) has no body to walk.
            None => return type_call_decline("__init__ not inlinable"),
        },
        None => None,
    };

    let mut arg_concretes = vec![ConcreteValue::Ref(w_type), ConcreteValue::Null];
    let mut callee_arg_concretes = Vec::with_capacity(r_args.len() - 1);
    for (i, &arg) in r_args[2..].iter().enumerate() {
        let Some(concrete) = walker_concrete_ref_object(ctx, arg) else {
            return type_call_decline(&format!("argument {i} is not a concrete ref"));
        };
        arg_concretes.push(ConcreteValue::Ref(concrete));
        callee_arg_concretes.push(ConcreteValue::Ref(concrete));
    }

    // Everything below emits.  `try_walker_inline_resolved_user_call` has
    // decline paths of its own past this point, so keep a rewind point and cut
    // back to it — the orphaned instance is unreachable and `__init__` has not
    // run, so the residual re-does the whole instantiation from scratch.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();

    // Pin the class and the type-dict version the two lookups above resolved
    // against, so redefining `__new__` / `__init__` / `__del__` deopts.
    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[r_args[0], type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(r_args[0], type_const);
    walker_pin_type_version_tag(ctx, op.pc, type_const)?;
    // A metaclass that does not override `__call__` today can be given one, and
    // that changes its own version tag rather than the class's.
    if let Some(w_metaclass) = metaclass_to_pin {
        let metaclass_const = ctx.trace_ctx.const_ref(w_metaclass as i64);
        walker_pin_type_version_tag(ctx, op.pc, metaclass_const)?;
    }

    // The walker is the executor here, so the instance the rest of this walk
    // reads has to be a real one — the same split `trace_box_int` makes between
    // the recorded allocation and the concrete object it hands back.
    // The `__init__` below writes this instance's slots through the store-attr
    // resolver, which inside a rewind region refuses an unjournaled commit
    // unless the receiver is one of the region's own allocations.
    let (instance, concrete_instance) = emit_walker_instance(ctx, w_type, type_const, terminator);
    if fbw_inline_diag_enabled() {
        eprintln!(
            "[type-call-inline] pc={} class={} init={}",
            op.pc,
            unsafe { pyre_object::w_type_get_name(w_type) },
            inlinable_init.is_some(),
        );
    }

    let Some((init, (w_code, nparams, has_closure))) = inlinable_init else {
        write_ref_reg(
            ctx,
            op.pc,
            dst,
            instance,
            ConcreteValue::Ref(concrete_instance),
        )?;
        return Ok(Some((DispatchOutcome::Continue, op.next_pc)));
    };

    let mut callee_args = vec![instance];
    callee_args.extend_from_slice(&r_args[2..]);
    callee_arg_concretes.insert(0, ConcreteValue::Ref(concrete_instance));
    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        init,
        r_args[0],
        w_type,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        // `__init__` bodies are `self.x = ...` stores; the sub-walk folds them
        // to slot writes on the fresh instance exactly as the property-setter
        // route folds its own.
        false,
        Some((instance, ConcreteValue::Ref(concrete_instance))),
    )?;
    if inlined.is_none() {
        // The `[type-call-inline]` line above is printed before this sub-walk is
        // attempted, because that is where the class and the `init` shape are
        // known — so on its own it reports that the fold *began*, not that it
        // stood.  Say so when it does not: without this line the diagnostic
        // reads as a successful fold on a trace that ends up carrying the whole
        // instantiation as a residual.
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[type-call-rewind] pc={} class={} why=__init__ sub-walk declined",
                op.pc,
                unsafe { pyre_object::w_type_get_name(w_type) },
            );
        }
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Emit the instance allocation both instantiation folds perform, and bind the
/// emitted box to the object the walker actually made.
///
/// The walker is the executor here, so the instance the rest of the walk reads
/// has to be a real one -- the same split `trace_box_int` makes between the
/// recorded allocation and the concrete object it hands back.  Without the
/// concrete binding the instance reaches a residual as a box with no value and
/// the residual declines (`[fbw-resid-decline] box_value=None`), which costs
/// the recording iteration; without the class the receiver re-guards its own
/// freshly emitted layout.
///
/// The caller pins the class version tag first -- [`try_walker_inline_type_call`]
/// pins a metaclass beside it, which is why that stays at the call site.
fn emit_walker_instance<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    w_type: pyre_object::PyObjectRef,
    type_const: OpRef,
    terminator: *const u8,
) -> (OpRef, pyre_object::PyObjectRef) {
    let concrete_instance = pyre_object::w_instance_new(w_type);
    let terminator_const = ctx.trace_ctx.const_int(terminator as i64);
    // The vtable `alloc_instance_object` stamps for this `w_type`. Exact
    // `object()` is `INSTANCE_TYPE`; every other carrier is
    // `INSTANCE_USER_TYPE` (`typedef.py` `_getusercls(W_ObjectObject)`).
    let (typeptr, _) = pyre_object::instance_typeptr_for(w_type);
    let user = !std::ptr::eq(typeptr, &pyre_object::pyobject::INSTANCE_TYPE);
    // Nursery-born (`alloc_instance_object`). `emit_instance_inline`
    // records a collecting `NewWithVtable`; reload after
    // (`shadowstack.py expand_pop_roots`).
    let _roots = pyre_object::gc_roots::push_roots();
    let concrete_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(concrete_instance);
    let instance =
        crate::helpers::emit_instance_inline(ctx.trace_ctx, type_const, terminator_const, user);
    let concrete_instance = pyre_object::gc_roots::shadow_stack_get(concrete_slot);
    ctx.trace_ctx.set_opref_concrete(
        instance,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_instance as usize)),
    );
    ctx.trace_ctx.heap_cache_mut().class_now_known(instance);
    (instance, concrete_instance)
}

/// Fold `object.__new__(cls)` to the allocation it performs.
///
/// For a one-argument call on a concrete class, `object_descr_new` reduces to
/// `w_instance_new(cls)` behind four record-time tests: `cls` is a type, it is
/// instantiable, it is not abstract, and it is laid out by `object` itself
/// (`check_user_subclass`).  Pinning the class is what keeps those answers, so
/// the emit is the same `NewWithVtable` + header/`map` pair
/// [`try_walker_inline_type_call`] builds for a class it did not have to run a
/// `__new__` for.
///
/// This is how every Python `__new__` ends -- `self = object.__new__(cls)` --
/// so without it the descent into an overridden `__new__` stops on a residual
/// whose body reaches `object_descr_new`'s error paths
/// (`__getslice_minusone`, `abstract_instantiation_error`,
/// `type_repr_qualified_name`), which is where the builtin route declines with
/// `un-lowered helper call in body`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_object_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 3 {
        return Ok(None);
    }
    // Same two spellings of "no receiver" the type call reads.
    if walker_concrete_ref_object(ctx, r_args[1])
        .is_some_and(|null_or_self| !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL)
    {
        return Ok(None);
    }
    let Some(callable) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    let w_object = pyre_interpreter::typedef::w_object();
    if w_object.is_null() {
        return Ok(None);
    }
    let obj_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__new__") };
    if obj_new != Some(callable) {
        return Ok(None);
    }
    let Some(w_type) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_type(w_type) } {
        return Ok(None);
    }
    // The class pin below is what holds `__del__` and the abstract flag, so it
    // has to be a class whose dict changes are tracked.
    if unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) } == 0 {
        return Ok(None);
    }
    if unsafe {
        pyre_object::w_type_disallows_instantiation(w_type)
            || pyre_object::w_type_is_abstract(w_type)
            // `w_instance_new` puts an instance of such a class on the
            // finalizer queue, which `NewWithVtable` does not do.
            || pyre_object::typeobject::w_type_get_hasuserdel(w_type)
    } {
        return Ok(None);
    }
    // `check_user_subclass(object, cls)`: the base allocator only knows how to
    // fill the parent layout, so a subtype that introduces its own is refused.
    let typedef_of = |w: pyre_object::PyObjectRef| {
        let layout = unsafe { pyre_object::typeobject::w_type_get_layout_ptr(w) };
        if layout.is_null() {
            std::ptr::null()
        } else {
            unsafe { (*layout).typedef }
        }
    };
    if !std::ptr::eq(typedef_of(w_object), typedef_of(w_type)) {
        return Ok(None);
    }
    let terminator =
        unsafe { pyre_interpreter::objspace::std::mapdict::ensure_type_terminator(w_type) };
    if terminator.is_null() {
        return Ok(None);
    }

    let callable_const = ctx.trace_ctx.const_ref(callable as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op.pc,
        OpCode::GuardValue,
        &[r_args[0], callable_const],
    )?;
    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[r_args[2], type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(r_args[2], type_const);
    walker_pin_type_version_tag(ctx, op.pc, type_const)?;

    let (instance, concrete_instance) = emit_walker_instance(ctx, w_type, type_const, terminator);
    if fbw_inline_diag_enabled() {
        eprintln!("[object-new-inline] pc={} class={}", op.pc, unsafe {
            pyre_object::w_type_get_name(w_type)
        });
    }
    write_ref_reg(
        ctx,
        op.pc,
        dst,
        instance,
        ConcreteValue::Ref(concrete_instance),
    )?;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Route `str(exc)` / `repr(exc)` through an app-level exception override.
///
/// Pyre's exact `str` type call follows `str_descr_new` → `builtin_str` →
/// `exc_user_dunder_obj`; the builtin `repr` follows `builtin_repr` →
/// `py_repr_obj`. Both paths look up the receiver dunder before builtin
/// exception formatting. This is the walker counterpart of
/// `descroperation.py`'s `space.lookup` + `get_and_call_function`: pin the
/// promoted exception class, then enter the ordinary resolved-callee inline
/// plumbing with the receiver as `self`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_exception_string_override<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let Some(concrete_callable) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if walker_concrete_ref_object(ctx, r_args[1]).is_some() {
        return Ok(None);
    }
    let Some(concrete_receiver) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return Ok(None);
    };
    // `is_exception` → `ll_isinstance` reads `ob_type`. A walk concrete
    // that has already been collected (or never had a header) is not
    // an exception override target.
    if unsafe { (*concrete_receiver).ob_type.is_null() } {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_exception(concrete_receiver) } {
        return Ok(None);
    }

    let dunder = if std::ptr::eq(
        concrete_callable,
        pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::STR_TYPE),
    ) {
        "__str__"
    } else if pyre_interpreter::builtins::is_builtin_repr_function(concrete_callable) {
        "__repr__"
    } else {
        return Ok(None);
    };

    let w_class = unsafe { (*concrete_receiver).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        return Ok(None);
    }
    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) })
    else {
        return Ok(None);
    };
    let Some(base_exception) = pyre_interpreter::builtins::lookup_exc_class("BaseException") else {
        return Ok(None);
    };
    let Some(default_method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(base_exception, dunder) })
    else {
        return Ok(None);
    };
    if std::ptr::eq(method, default_method) {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };

    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    // A nested call in the override body (e.g. `return repr(self.args)`) is
    // inlined like any other: the nested callee records its own residual and
    // guard-resume snapshot on this route without aborting, and a raise from it
    // reaches the enclosing handler unchanged.
    let Some((_override_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };

    // A straight-line, effect-free override can be sampled before any IR is
    // emitted. If its observed result is not a string, decline to the original
    // builtin residual so the interpreter's result check raises TypeError and
    // the exception-handler loop remains traceable. More complex bodies are
    // not executed speculatively; their inlined result is guarded below.
    if let (Some(body), Some((callee_descr_refs, _, _))) = (
        crate::state::sub_jitcode_body_for_code(w_code),
        crate::state::sub_jitcode_descr_pool_for_code(w_code),
    ) {
        if body_facts.exc_override_sample_safe {
            let sampled = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(method, &[concrete_receiver])
            };
            let sampled_is_acceptable = matches!(sampled, Ok(result)
                if !result.is_null() && unsafe { pyre_object::is_str(result) });
            if !sampled_is_acceptable {
                return Ok(None);
            }
        }
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
    ];
    let _exception_string_inline = ExceptionStringInlineGuard::enter();
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        r_args[0],
        concrete_callable,
        arg_concretes,
        vec![r_args[2]],
        vec![ConcreteValue::Ref(concrete_receiver)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some(
            (r_args[2], concrete_receiver, w_class, version_tag, unsafe {
                inline_attr_cell_guard(w_class, dunder, method)
            }),
        ),
        None,
        true,
        true,
        None,
    )?
    else {
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r.get(dst).expect("ref register in range");
        let str_type = &pyre_object::pyobject::STR_TYPE as *const _ as i64;
        let str_type_const = ctx.trace_ctx.const_int(str_type);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[result, str_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(result);
    }
    Ok(Some(inlined))
}

/// Machine-int digest of a `__hash__` result: a tagged immediate or an exact
/// heap `W_IntObject`; bool/long take `hash_call_normalize`'s other arms.
fn walker_machine_int_value(obj: pyre_object::PyObjectRef) -> Option<i64> {
    if obj.is_null() {
        return None;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Some(pyre_object::tagged_int::untag_int(obj));
    }
    unsafe {
        if pyre_object::is_int(obj) && !pyre_object::is_bool(obj) && !pyre_object::is_long(obj) {
            Some(pyre_object::w_int_get_value(obj))
        } else {
            None
        }
    }
}

/// `hash(x)` over a user instance — the hash sibling of
/// [`try_walker_inline_exception_string_override`]: pin the receiver class,
/// inline the resolved `__hash__` body in place of the opaque call residual,
/// then emit `hash_call_normalize`'s digest check and `-1 -> -2` map as int
/// ops.  Declines to the residual before any IR is emitted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_hash_builtin<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let Some(concrete_callable) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if walker_concrete_ref_object(ctx, r_args[1]).is_some() {
        return Ok(None);
    }
    let Some(concrete_receiver) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_hash_function(concrete_callable) {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_instance(concrete_receiver) } {
        return Ok(None);
    }
    let w_type = unsafe { pyre_object::w_instance_get_type(concrete_receiver) };
    if w_type.is_null() || !unsafe { pyre_object::is_type(w_type) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return Ok(None);
    }
    let Some(method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__hash__") })
    else {
        return Ok(None);
    };
    // `__hash__ = None` raises in the residual; a non-Python `__hash__`
    // has no body to walk.
    if unsafe { pyre_object::is_none(method) } {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    if crate::state::sub_jitcode_body_for_code(w_code).is_none()
        || crate::state::sub_jitcode_descr_pool_for_code(w_code).is_none()
    {
        return Ok(None);
    }
    // No pre-sampling: `hash_w` calls `__hash__` exactly once, so the digest
    // is checked after the single authoritative sub-walk run instead.
    let effect_free = body_facts.exc_override_sample_safe;

    let arg_concretes = vec![
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
    ];
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        r_args[0],
        concrete_callable,
        arg_concretes,
        vec![r_args[2]],
        vec![ConcreteValue::Ref(concrete_receiver)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((r_args[2], concrete_receiver, w_type, version_tag, unsafe {
            inline_attr_cell_guard(w_type, "__hash__", method)
        })),
        None,
        true,
        false,
        None,
    )?
    else {
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r.get(dst).expect("ref register in range");
        let concrete_result = walker_concrete_ref_object(ctx, result);
        let live = concrete_result.and_then(walker_machine_int_value);
        // The inline unbox is guard-free only against a known-class or
        // trace-built box; a live post-body guard on a side-effecting body
        // would re-run its effects on failure, so those shapes — and every
        // bool/long digest — take the fallible normalize residual instead.
        let inline_unbox = live.is_some()
            && (effect_free
                || ctx.trace_ctx.heap_cache().is_class_known(result)
                || ctx.trace_ctx.heap_cache().is_unescaped(result));
        let (norm, live_norm) = if inline_unbox {
            let live = live.unwrap();
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let raw = walker_unbox_int(ctx, op.pc, result, int_type_addr)?;
            // `hash_call_normalize`'s `-1 -> -2` map as `raw - (raw == -1)`.
            let neg1 = ctx.trace_ctx.const_int(-1);
            let is_neg1 = ctx.trace_ctx.record_op(OpCode::IntEq, &[raw, neg1]);
            ctx.trace_ctx
                .set_opref_concrete(is_neg1, majit_ir::Value::Int((live == -1) as i64));
            let norm = ctx.trace_ctx.record_op(OpCode::IntSub, &[raw, is_neg1]);
            let live_norm = if live == -1 { -2 } else { live };
            ctx.trace_ctx
                .set_opref_concrete(norm, majit_ir::Value::Int(live_norm));
            (norm, live_norm)
        } else {
            let Some(concrete_result) = concrete_result else {
                return Err(DispatchError::callee_inline_unsupported(op.pc));
            };
            let raw = crate::helpers::emit_trace_call_int_typed(
                ctx.trace_ctx,
                crate::helpers::jit_hash_normalize_digest as *const (),
                &[result],
                &[majit_ir::Type::Ref],
            );
            match pyre_interpreter::builtins::normalize_hash_digest(concrete_result) {
                Ok(live_norm) => {
                    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNoException, &[])?;
                    ctx.trace_ctx
                        .set_opref_concrete(raw, majit_ir::Value::Int(live_norm));
                    (raw, live_norm)
                }
                // A raising digest completes the recording the way the
                // generic raising residual does: publish the exception,
                // pin it with GuardException, surface SubRaise.
                Err(mut err) => {
                    let exc = err.to_exc_object();
                    fbw_count_executed_residual(true, true);
                    ctx.set_last_exc_value_concrete(ConcreteValue::Ref(exc));
                    ctx.fbw_mode.class_of_last_exc_is_const = false;
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));
                    walker_record_guard_exception(ctx, op.pc)?;
                    let exc_box = ctx
                        .last_exc_value()
                        .expect("guard_exception seeds last_exc_value");
                    return Ok(Some((
                        DispatchOutcome::SubRaise {
                            exc: exc_box,
                            exc_concrete: ConcreteValue::Ref(exc),
                        },
                        op.next_pc,
                    )));
                }
            }
        };
        let boxed = walker_box_int(ctx, op.pc, norm, live_norm)?;
        let live_ptr = pyre_object::w_int_new(live_norm) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live_norm, live_ptr));
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    }
    Ok(Some(inlined))
}

/// Inline an exact `property` getter selected by
/// `W_Super.getattribute`'s MRO suffix walk.
///
/// PyPy traces through `lookup_starting_at` and `W_Property.get`, so the
/// getter enters the same MIFrame inline path as an ordinary property read.
/// The fused 3.14 `LOAD_SUPER_ATTR` reaches pyre as one opaque may-force
/// residual; this reconstructs that upstream shape under the same start-class,
/// receiver-class and type-version guards as the non-Python `super` folds.
///
/// The zero-argument spelling is the only one that arrives fused.
/// [`try_walker_inline_super_proxy_property_get`] takes the two-argument one,
/// which `codewriter.rs` lowers to a call and an attribute load.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_super_property_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    global_super: OpRef,
    self_obj: OpRef,
    cls: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_super) = walker_concrete_ref_object(ctx, global_super) else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_super_type(concrete_super) {
        return Ok(None);
    }
    let Some(concrete_self) = walker_concrete_ref_object(ctx, self_obj) else {
        return Ok(None);
    };
    let Some(concrete_cls) = walker_concrete_ref_object(ctx, cls) else {
        return Ok(None);
    };
    let python_free = unsafe {
        pyre_interpreter::baseobjspace::super_attr_fast_path(concrete_cls, concrete_self, name)
    };
    let apparent = if python_free.is_none() {
        super::specialize::walker_apparent_super_class(concrete_cls, concrete_self)
    } else {
        None
    };
    let (objtype, w_descr, class_mode) =
        if let Some((objtype, _, w_descr, class_mode)) = python_free {
            (objtype, w_descr, class_mode)
        } else if let Some(apparent) = apparent {
            let Some((_, w_descr, class_mode)) = (unsafe {
                pyre_interpreter::baseobjspace::super_attr_proxy_fast_path(
                    concrete_cls,
                    apparent.objtype,
                    concrete_self,
                    name,
                )
            }) else {
                return Ok(None);
            };
            (apparent.objtype, w_descr, class_mode)
        } else {
            return Ok(None);
        };
    // Class access asks `property.__get__(None, objtype)` and gets the
    // descriptor itself; the constant result fold owns that arm.
    if class_mode || !unsafe { pyre_object::descriptor::is_exact_property(w_descr) } {
        return Ok(None);
    }
    let fget = unsafe { pyre_object::descriptor::w_property_get_fget(w_descr) };
    if fget.is_null() || unsafe { pyre_object::is_none(fget) } {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(fget) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(fget),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_self),
    ];
    let fget_const = ctx.trace_ctx.const_ref(fget as i64);
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();

    // LOAD_SUPER_ATTR names the global callable explicitly.  Pin it before
    // baking the builtin implementation, just as the non-Python result fold
    // does; the rewind below removes this guard if the callee walk declines.
    if !global_super.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_super as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[global_super, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(global_super, expected);
    }
    if let Some(apparent) = apparent {
        let cls_const = ctx.trace_ctx.const_ref(concrete_cls as i64);
        if !cls.is_constant() {
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op.pc,
                OpCode::GuardValue,
                &[cls, cls_const],
            )?;
            ctx.trace_ctx.heap_cache_mut().replace_box(cls, cls_const);
        }
        super::specialize::walker_guard_apparent_super_class(
            ctx,
            op.pc,
            self_obj,
            concrete_self,
            apparent,
        )?;
    } else {
        super::specialize::walker_emit_super_attr_lookup_guards(
            ctx,
            op.pc,
            self_obj,
            cls,
            concrete_self,
            concrete_cls,
            objtype,
            class_mode,
        )?;
    }
    walker_pin_descriptor_slot(ctx, op.pc, w_descr, crate::descr::property_fget_descr())?;

    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        fget_const,
        r_args,
        call_descr,
        'r',
        dst,
        fget,
        fget_const,
        fget,
        arg_concretes,
        vec![self_obj],
        vec![ConcreteValue::Ref(concrete_self)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        false,
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Inline the same exact `property` getter through an already-built
/// `super` proxy (`proxy = super(); proxy.name`).  Splitting proxy creation
/// from the attribute opcode must not turn PyPy's traced
/// `W_Super.getattribute -> W_Property.get` chain back into an opaque
/// `getattr_fn` residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_super_proxy_property_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    proxy: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_proxy) = walker_concrete_ref_object(ctx, proxy) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::descriptor::is_super(concrete_proxy) } {
        return Ok(None);
    }
    let Some(proxy_w_class) = (unsafe { walker_exact_builtin_class(concrete_proxy) }) else {
        return Ok(None);
    };
    let concrete_cls = unsafe { pyre_object::descriptor::w_super_get_type(concrete_proxy) };
    let concrete_self = unsafe { pyre_object::descriptor::w_super_get_obj(concrete_proxy) };
    let objtype = unsafe { pyre_object::descriptor::w_super_get_obj_type(concrete_proxy) };
    let Some((_version_tag, w_descr, class_mode)) = (unsafe {
        pyre_interpreter::baseobjspace::super_attr_proxy_fast_path(
            concrete_cls,
            objtype,
            concrete_self,
            name,
        )
    }) else {
        return Ok(None);
    };
    if class_mode || !unsafe { pyre_object::descriptor::is_exact_property(w_descr) } {
        return Ok(None);
    }
    let fget = unsafe { pyre_object::descriptor::w_property_get_fget(w_descr) };
    if fget.is_null() || unsafe { pyre_object::is_none(fget) } {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(fget) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(fget),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_self),
    ];
    let fget_const = ctx.trace_ctx.const_ref(fget as i64);
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let (self_op, cls_op) = super::specialize::walker_guard_and_read_super_proxy(
        ctx,
        op.pc,
        proxy,
        proxy_w_class,
        concrete_self,
        concrete_cls,
        objtype,
    )?;
    super::specialize::walker_emit_super_attr_lookup_guards(
        ctx,
        op.pc,
        self_op,
        cls_op,
        concrete_self,
        concrete_cls,
        objtype,
        false,
    )?;
    walker_pin_descriptor_slot(ctx, op.pc, w_descr, crate::descr::property_fget_descr())?;

    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        fget_const,
        r_args,
        call_descr,
        'r',
        dst,
        fget,
        fget_const,
        fget,
        arg_concretes,
        vec![self_op],
        vec![ConcreteValue::Ref(concrete_self)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        false,
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Inline a `property` getter read (`obj.value`) after the plain-attribute
/// mapdict fold declines because the attribute is a data descriptor.  PyPy
/// traces *through* `space.getattr` → `property.__get__` →
/// `space.call_function(fget, obj)`, inlining the pure-Python getter body; pyre
/// leaves LOAD_ATTR an opaque `getattr` `CALL_MAY_FORCE` residual whose fget
/// runs as a fresh interpreter frame every iteration.  This fold reproduces
/// PyPy's shape: pin the receiver class + version tag (so the property lookup
/// const-folds), then enter the resolved-callee inline plumbing with the
/// receiver as `self` so the getter body (commonly `return self._value`) folds
/// to a guarded slot read inside the trace.
///
/// Restricted to a straight-line getter whose body owns no loop: the inline
/// then stays off the `CALL_ASSEMBLER` route (the only consumer of this
/// LOAD_ATTR residual's non-call `r_args`).  Nested calls are walked normally;
/// in particular, `raise ValueError(...)` constructs the exception in the
/// callee sub-walk and returns `SubRaise` to this LOAD_ATTR's enclosing
/// `catch_exception`, matching `descroperation.py` tracing through the
/// property's `__get__` and Python fget.
/// Top full-body frame only, for the resume-doubling reason
/// [`try_walker_specialize_load_bound_method_attr`] documents.  Every other
/// shape declines to the residual (SAFE — no acceleration, unchanged
/// semantics).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_property_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    try_walker_inline_property_get_named(
        ctx,
        op,
        code,
        r_args,
        call_descr,
        obj,
        Wtf8::new(name.as_str()),
        dst,
        dst_bank,
        None,
    )
}

/// Builtin `getattr(obj, name)` counterpart of
/// [`try_walker_inline_property_get`].  PyPy traces through the builtin and
/// the property's Python getter regardless of whether `name` has a UTF-8
/// spelling; keep that route for pyre's opaque builtin-call boundary.
pub(crate) fn try_walker_inline_builtin_getattr_property<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if r_args.len() != 4 {
        return Ok(None);
    }
    let concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_name),
    ) = (concretes[0], concretes[1], concretes[2], concretes[3])
    else {
        return Ok(None);
    };
    if callable.is_null()
        || !null_or_self.is_null()
        || concrete_obj.is_null()
        || concrete_name.is_null()
        || !pyre_interpreter::builtins::is_builtin_getattr_function(callable)
        || !unsafe { pyre_object::is_exact_type(concrete_name, &pyre_object::STR_TYPE) }
    {
        return Ok(None);
    }
    let name = unsafe { pyre_object::w_str_get_wtf8(concrete_name) };
    try_walker_inline_property_get_named(
        ctx,
        op,
        code,
        r_args,
        call_descr,
        r_args[2],
        name,
        dst,
        'r',
        Some((r_args[0], callable, r_args[3], concrete_name)),
    )
}

#[allow(clippy::too_many_arguments)]
fn try_walker_inline_property_get_named<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    name: &Wtf8,
    dst: usize,
    dst_bank: char,
    builtin_guards: Option<(
        OpRef,
        pyre_object::PyObjectRef,
        OpRef,
        pyre_object::PyObjectRef,
    )>,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    // Instance map first. A class object has no map; `descr_getattribute`
    // then answers an exact metaclass `property` by calling `fget(cls)`.
    // The tuple is the same shape either way: `w_type` is the class whose
    // version tag pins the lookup (the instance's class, or the metaclass).
    let Some((w_type, version_tag, w_descr, fget)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::property_get_fast_path_wtf8(concrete_obj, name)
    })
    .or_else(|| unsafe {
        pyre_interpreter::objspace::std::mapdict::type_property_get_fast_path(concrete_obj, name)
    }) else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(fget) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    // A branch still needs the bounded property-entry control-flow port.  A
    // nested call does not: the ordinary sub-walk handles it, including the
    // exception-constructor call in a raising getter.
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    // `[fget, <self-placeholder>, obj]`: the method-form call header the inline
    // plumbing expects (`callable`, unused self slot, then the receiver).
    let arg_concretes = vec![
        ConcreteValue::Ref(fget),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_obj),
    ];
    let fget_const = ctx.trace_ctx.const_ref(fget as i64);
    // Everything below emits, and the callee inline has decline paths of its
    // own past this point, so keep a rewind point the way the type-call fold
    // does.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    if let Some((callable_op, callable, name_op, concrete_name)) = builtin_guards {
        if !callable_op.is_constant() {
            let expected = ctx.trace_ctx.const_ref(callable as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op.pc,
                OpCode::GuardValue,
                &[callable_op, expected],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .replace_box(callable_op, expected);
        }
        if !name_op.is_constant() {
            let expected = ctx.trace_ctx.const_ref(concrete_name as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op.pc,
                OpCode::GuardValue,
                &[name_op, expected],
            )?;
        }
    }
    walker_pin_descriptor_slot(ctx, op.pc, w_descr, crate::descr::property_fget_descr())?;
    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        fget_const,
        r_args,
        call_descr,
        'r',
        dst,
        fget,
        fget_const,
        fget,
        arg_concretes,
        vec![obj],
        vec![ConcreteValue::Ref(concrete_obj)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag, None)),
        None,
        // LOAD_ATTR is not a CALL either, but this entry is left admitted
        // as it was: only the subscript one below has a witness.
        true,
        // Getter bodies commonly read `self._slot` — a LOAD_ATTR the method-form
        // support gate would otherwise reject; the sub-walk folds it to a slot
        // read (same allowance the exception `__str__`/`__repr__` override uses).
        false,
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Inline the general user data-descriptor call selected by
/// `Object.descr__getattribute__`:
/// `get_and_call_function(w_get, w_descr, w_obj, w_type)`.
///
/// Unlike the exact-`property` arm, the callee is the descriptor type's
/// ordinary Python `__get__` function and receives all three explicit
/// arguments.  The receiver class/version pins the named descriptor lookup;
/// the descriptor's own map/class/version pins both its data-descriptor status
/// and the `__get__` lookup.  The descriptor and owner type are green, while
/// the accessed object remains the caller's red operand and is threaded into
/// the callee's distinct MIFrame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_data_descriptor_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_descr, descr_type, descr_version_tag, descr_map, w_get)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::data_descriptor_get_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_get) }) else {
        return Ok(None);
    };
    // `get_and_call_function`'s exact-Function arm supplies precisely the
    // descriptor, receiver and receiver type.
    if nparams != 3 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }

    let get_const = ctx.trace_ctx.const_ref(w_get as i64);
    let descr_const = ctx.trace_ctx.const_ref(w_descr as i64);
    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    let arg_concretes = vec![
        ConcreteValue::Ref(w_get),
        ConcreteValue::Null,
        ConcreteValue::Ref(w_descr),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(w_type),
    ];
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();

    // The descriptor itself is mutable through `__class__` assignment.  Its
    // live map/class guard is therefore load-bearing even though the receiver
    // type lookup produced a constant pointer to it.
    walker_guard_mapdict_instance_shape(
        ctx,
        op.pc,
        descr_const,
        w_descr,
        descr_type,
        descr_version_tag,
        descr_map,
    )?;
    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        get_const,
        r_args,
        call_descr,
        'r',
        dst,
        w_get,
        get_const,
        w_get,
        arg_concretes,
        vec![descr_const, obj, type_const],
        vec![
            ConcreteValue::Ref(w_descr),
            ConcreteValue::Ref(concrete_obj),
            ConcreteValue::Ref(w_type),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag, None)),
        None,
        true,
        false,
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Which gate made [`nondatadescr_get_fast_path`](pyre_interpreter::objspace::std::mapdict::nondatadescr_get_fast_path)
/// answer `None`. The instance arm and the type arm share the descriptor pins,
/// so a miss that only one receiver hits is one of the probes printed here.
fn diag_nondatadescr_miss(pc: usize, name: &str, obj: pyre_object::PyObjectRef) {
    if !fbw_inline_diag_enabled() {
        return;
    }
    let is_type = unsafe { pyre_object::is_type(obj) };
    let owns_dict = unsafe { pyre_interpreter::objspace::std::mapdict::typedef_owns_dict(obj) };
    let mapdict = unsafe { pyre_interpreter::objspace::std::mapdict::has_mapdict_storage(obj) };
    let w_type = if is_type {
        obj
    } else {
        unsafe { pyre_object::w_instance_get_type(obj) }
    };
    let version = if w_type.is_null() {
        0
    } else {
        unsafe { pyre_interpreter::baseobjspace::w_type_version_tag(w_type) }
    };
    let getattr = !w_type.is_null()
        && unsafe { pyre_interpreter::baseobjspace::lookup_in_type_where(w_type, "__getattr__") }
            .is_some();
    let looked = if w_type.is_null() {
        None
    } else {
        unsafe { pyre_interpreter::baseobjspace::lookup_in_type_where(w_type, name) }
    };
    let cell_null = w_type.is_null()
        || unsafe {
            pyre_interpreter::baseobjspace::type_attr_object_cell(
                w_type,
                rustpython_wtf8::Wtf8::new(name),
            )
        }
        .is_null();
    let pins = looked.is_some_and(|descr| {
        unsafe { pyre_interpreter::objspace::std::mapdict::python_descr_get_pins(descr) }.is_some()
    });
    eprintln!(
        "[nondatadescr-none] pc={pc} name={name} is_type={is_type} owns_dict={owns_dict} \
         mapdict={mapdict} version={version} getattr={getattr} lookup={} cell_null={cell_null} \
         pins={pins}",
        looked.is_some(),
    );
}

/// Inline `type(descr).__get__(descr, obj_or_None, owner)` for a non-data
/// descriptor.
///
/// `typeobject.py W_TypeObject.descr_getattribute` and
/// `objectobject.py Object.descr__getattribute__` both finish in
/// `descroperation.py DescrOperation.get`. Exact `classmethod` /
/// `staticmethod` unwrap `w_function` on their own folds; a subclass that
/// overrides `__get__` has to run that override, which is the body recorded
/// here. The following `CALL` then sees whatever the override returned.
///
/// Class access passes the real `None` object (`get` turns a null receiver
/// into `space.w_None` only for the Python-visible call). Instance access
/// also pins the receiver map: a non-data descriptor loses to an instance
/// store, and the version tag does not move when that store happens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_nondatadescr_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[nondatadescr-none] pc={} name={name} why=no-concrete",
                op.pc
            );
        }
        return Ok(None);
    };
    let Some(pins) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::nondatadescr_get_fast_path(concrete_obj, &name)
    }) else {
        diag_nondatadescr_miss(op.pc, &name, concrete_obj);
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(pins.w_get) })
    else {
        return Ok(None);
    };
    // `DescrOperation.get` calls `__get__(descr, obj, type)`.
    if nparams != 3 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }

    let get_const = ctx.trace_ctx.const_ref(pins.w_get as i64);
    let descr_const = ctx.trace_ctx.const_ref(pins.w_descr as i64);
    let owner_const = ctx.trace_ctx.const_ref(pins.w_owner as i64);
    let none = pyre_object::w_none();
    let (instance_op, instance_concrete) = if pins.class_access {
        (ctx.trace_ctx.const_ref(none as i64), none)
    } else {
        (obj, concrete_obj)
    };
    let arg_concretes = vec![
        ConcreteValue::Ref(pins.w_get),
        ConcreteValue::Null,
        ConcreteValue::Ref(pins.w_descr),
        ConcreteValue::Ref(instance_concrete),
        ConcreteValue::Ref(pins.w_owner),
    ];
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    if pins.class_access {
        // The receiver is the class object. `GuardClass` on a type only
        // names `type`, so pin this class the way `load_type_attr` does.
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[obj, owner_const])?;
        ctx.trace_ctx.heap_cache_mut().replace_box(obj, owner_const);
        walker_pin_type_version_tag(ctx, op.pc, owner_const)?;
    } else {
        walker_guard_mapdict_instance_shape(
            ctx,
            op.pc,
            obj,
            concrete_obj,
            pins.w_owner,
            pins.owner_version_tag,
            pins.receiver_map,
        )?;
    }
    // A class-body assignment is an `ObjectMutableCell`. The version pin above
    // does not move on `write_cell`; getfield the payload before baking it.
    if !pins.attr_cell.is_null() {
        super::walker_promote_object_mutable_cell(ctx, op.pc, pins.attr_cell, pins.w_descr)?;
    }
    // `__class__` assignment on the descriptor retargets its map. The class
    // lookup handed back a constant pointer, which does not pin that.
    walker_guard_mapdict_instance_shape(
        ctx,
        op.pc,
        descr_const,
        pins.w_descr,
        pins.descr_type,
        pins.descr_version_tag,
        pins.descr_map,
    )?;
    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        get_const,
        r_args,
        call_descr,
        'r',
        dst,
        pins.w_get,
        get_const,
        pins.w_get,
        arg_concretes,
        vec![descr_const, instance_op, owner_const],
        vec![
            ConcreteValue::Ref(pins.w_descr),
            ConcreteValue::Ref(instance_concrete),
            ConcreteValue::Ref(pins.w_owner),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        false,
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Inline the receiver type's `__getattr__` hook for an attribute the type and
/// the instance both lack — the miss twin of [`try_walker_inline_property_get`].
///
/// `descroperation.py:242-245` reaches the hook only after the descriptor
/// protocol has raised, so pyre runs the whole `object_getattr_miss` walk (the
/// `__dict__` / `__doc__` / `__class__` special names, the metaclass loops, the
/// terminal miss) and then a fresh interpreter frame for the hook, on every
/// access, behind one opaque `CALL_MAY_FORCE`. PyPy traces through all of it,
/// so the miss const-folds and only the hook body is left.
/// [`getattr_hook_fast_path`](pyre_interpreter::objspace::std::mapdict::getattr_hook_fast_path)
/// is the oracle for that fold: the version-tag and map pins it asks for are
/// what make the miss a compile-time answer.
///
/// All three spellings `get_and_call_function` binds are folded, because the
/// version-tag pin makes the binding decision itself constant: a plain
/// `Function` takes `funccall(w_obj, w_name)`, a `classmethod` is entered with
/// the class the descriptor would bind, and a `staticmethod` with the name
/// alone. A custom-descriptor hook stays on the residual.
///
/// The name argument is an interned immortal block rather than the fresh
/// `w_str_new` the residual path allocates per access, which is the shape
/// `pyopcode.py LOAD_ATTR` passes (`space.getattr(w_obj, w_name)` hands over
/// `co_names_w[oparg]`, one object for the life of the code object).
///
/// A branching, raising body is admitted: a hook that raises `AttributeError`
/// for an unknown name is the shape worth inlining, not an edge case. Same
/// loop-header and top-frame restrictions as the sibling routes; every other
/// shape declines to the residual (SAFE — no acceleration, unchanged
/// semantics).
/// The leading argument `get_and_call_function` binds ahead of the attribute
/// name, one variant per descriptor spelling of a `__getattr__` hook.
#[derive(Clone, Copy)]
enum HookLeading {
    /// Plain `Function`: `funccall(w_obj, w_name)` leads with the receiver.
    Receiver,
    /// `ClassMethod.__get__` leads with the class.
    Class,
    /// `StaticMethod.__get__` binds nothing; the name is the only argument.
    None,
}

/// The wrapper slot a descriptor spelling of the hook unwrapped, so the fold
/// can pin the value it read.  `function.py:673`/`:720`
/// `_immutable_fields_ = ['w_function?']`.
#[derive(Clone, Copy)]
enum WrapperField {
    ClassMethod,
    StaticMethod,
}

impl WrapperField {
    fn quasi_descr(&self) -> majit_ir::DescrRef {
        match self {
            Self::ClassMethod => crate::descr::classmethod_w_function_quasi_descr(),
            Self::StaticMethod => crate::descr::staticmethod_w_function_quasi_descr(),
        }
    }
}

/// Resolve the three descriptor spellings `get_and_call_function` gives an
/// attribute hook.  Shared by `__getattribute__` and `__getattr__`, which are
/// dispatched identically in `descroperation.py:_handle_getattribute` once
/// their respective lookup has selected a descriptor.
unsafe fn resolve_attribute_hook(
    w_hook: pyre_object::PyObjectRef,
) -> Option<(pyre_object::PyObjectRef, HookLeading, Option<WrapperField>)> {
    let resolved = unsafe {
        if pyre_object::function::is_exact_classmethod(w_hook) {
            (
                pyre_object::function::w_classmethod_get_func(w_hook),
                HookLeading::Class,
                Some(WrapperField::ClassMethod),
            )
        } else if pyre_object::function::is_exact_staticmethod(w_hook) {
            (
                pyre_object::function::w_staticmethod_get_func(w_hook),
                HookLeading::None,
                Some(WrapperField::StaticMethod),
            )
        } else {
            (w_hook, HookLeading::Receiver, None)
        }
    };
    (!resolved.0.is_null()).then_some(resolved)
}

/// Inline a custom Python `__getattribute__` selected by LOAD_ATTR / the
/// attribute half of LOAD_METHOD.
///
/// PyPy's `descroperation.py getattr` traces the receiver-type lookup and then
/// enters `get_and_call_function(w_descr, w_obj, w_name)`.  The optimized
/// trace therefore carries the same map promotion and type-version dependency
/// as mapdict LOAD_ATTR, followed by a distinct inlined MIFrame for the hook.
/// Pyre's fused `load_attr_fn` otherwise hides both stages in one
/// `CALL_MAY_FORCE`, so a hot override executes in the blackhole interpreter
/// on every iteration.
///
/// This is the hit-side predecessor of [`try_walker_inline_getattr_hook`].  It
/// admits the same exact Function/classmethod/staticmethod spellings, pins the
/// same wrapper fields, and uses the same per-frame resolved-call path.  A loop
/// in the hook, a type whose `__getattr__` would have to catch AttributeError,
/// or any shape the oracle cannot guard remains residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_getattribute_hook<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    // Instance map first. A class object has none; `getattr_str` still
    // runs a metaclass `__getattribute__` that is not
    // `descr_getattribute`, pinned by `w_class` and the metaclass version
    // tag rather than a map.
    let instance_hook = unsafe {
        pyre_interpreter::objspace::std::mapdict::getattribute_hook_fast_path(concrete_obj)
    };
    let type_hook = if instance_hook.is_none() {
        unsafe {
            pyre_interpreter::objspace::std::mapdict::type_getattribute_hook_fast_path(concrete_obj)
        }
    } else {
        None
    };
    let (w_type, version_tag, receiver_map, w_getattribute, attr_cell) =
        if let Some((w_type, version_tag, map, w_getattribute, attr_cell)) = instance_hook {
            (w_type, version_tag, Some(map), w_getattribute, attr_cell)
        } else if let Some((w_type, version_tag, w_getattribute, attr_cell)) = type_hook {
            (w_type, version_tag, None, w_getattribute, attr_cell)
        } else {
            return Ok(None);
        };
    let Some((w_func, leading, wrapper_field)) =
        (unsafe { resolve_attribute_hook(w_getattribute) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_func) }) else {
        return Ok(None);
    };
    if nparams != usize::from(!matches!(leading, HookLeading::None)) + 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    if let Some(map) = receiver_map {
        walker_guard_mapdict_instance_shape(
            ctx,
            op.pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;
    } else {
        // The receiver is the class. `GuardClass` only names the shared
        // type layout, so pin the metaclass through `w_class`.
        walker_guard_exception_attr_slot(ctx, op.pc, obj, concrete_obj, w_type, version_tag)?;
    }
    // The version-tag pin above does not cover an in-place cell write.  Same
    // getfield and `guard_value` as [`super::walker_promote_object_mutable_cell`]
    // on `ExceptionInlineReceiverGuard::attr_cell`.
    if !attr_cell.is_null() {
        super::walker_promote_object_mutable_cell(ctx, op.pc, attr_cell, w_getattribute)?;
    }
    if let Some(field) = wrapper_field {
        walker_pin_descriptor_slot(ctx, op.pc, w_getattribute, field.quasi_descr())?;
    }

    let name_obj =
        pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(name.as_str()))
            as pyre_object::PyObjectRef;
    let name_const = ctx.trace_ctx.const_ref(name_obj as i64);
    let leading_arg = match leading {
        HookLeading::Receiver => Some((obj, concrete_obj)),
        HookLeading::Class => Some((ctx.trace_ctx.const_ref(w_type as i64), w_type)),
        HookLeading::None => None,
    };
    let mut arg_concretes = vec![ConcreteValue::Ref(w_func), ConcreteValue::Null];
    let mut callee_args = Vec::with_capacity(2);
    let mut callee_arg_concretes = Vec::with_capacity(2);
    if let Some((arg, concrete)) = leading_arg {
        arg_concretes.push(ConcreteValue::Ref(concrete));
        callee_args.push(arg);
        callee_arg_concretes.push(ConcreteValue::Ref(concrete));
    }
    arg_concretes.push(ConcreteValue::Ref(name_obj));
    callee_args.push(name_const);
    callee_arg_concretes.push(ConcreteValue::Ref(name_obj));
    let hook_const = ctx.trace_ctx.const_ref(w_func as i64);
    let inlined = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        hook_const,
        r_args,
        call_descr,
        'r',
        dst,
        w_func,
        hook_const,
        w_func,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        false,
        None,
        None,
        false,
        None,
        // pyre's `getattr_str` (`baseobjspace.rs`) enriches the whole
        // `__getattribute__` / `__getattr__` chain, so the `load_attr_fn`
        // residual this route replaces filled the pair in here as well.
        // `StdObjSpace.getattr` (`objspace/std/objspace.py`) returns
        // through `_handle_getattribute` without enriching, so the interpreter
        // this trace has to agree with is ahead of upstream on this arm; the
        // difference is the interpreter's to settle, not the trace's.
        Some(AttributeErrorInlineContext {
            obj,
            obj_concrete: concrete_obj,
            name: name_const,
            name_concrete: name_obj,
        }),
        None,
        None,
        Vec::new(),
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_getattr_hook<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, map, w_getattr, attr_cell)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::getattr_hook_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    // `get_and_call_function` (`descroperation.py`) leads the
    // positionals with the receiver only for an exact `Function`; every other
    // descriptor goes through `space.get` first and is called with the name
    // alone.  The version-tag pin makes that binding decision a constant of the
    // trace, so each spelling resolves to its own function and leading argument
    // here instead of declining.
    //
    // The type tests are EXACT.  Upstream is explicit that they have to be
    // ("isinstance(typ, Function) would not be correct here … because a builtin
    // function binds differently than a normal function"), and the same holds
    // for the two wrappers: a `classmethod` subclass overriding `__get__` binds
    // through that override, so unwrapping `w_function` in its place calls the
    // wrong callable.  `wrapper_field` names the slot that unwrapping read, for
    // the guard below; the plain arm reads no field.
    let Some((w_func, leading, wrapper_field)) = (unsafe { resolve_attribute_hook(w_getattr) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_func) }) else {
        // A `classmethod` / `staticmethod` subclass is not a function.
        // `get_and_call_function` binds it through `__get__` and then calls
        // the result with the name. Unwrapping `w_function` would skip the
        // override.
        return try_walker_inline_getattr_hook_through_get(
            ctx,
            op,
            code,
            r_args,
            call_descr,
            obj,
            concrete_obj,
            &name,
            w_type,
            version_tag,
            map,
            w_getattr,
            attr_cell,
            dst,
        );
    };
    // The name, plus the bound leading argument when the descriptor supplies
    // one.  Any other arity is a shape the call would reject before the body
    // runs.
    if nparams != usize::from(!matches!(leading, HookLeading::None)) + 1 {
        return Ok(None);
    }
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }

    // Everything below emits, and the callee inline has decline paths of its
    // own past this point, so keep a rewind point the way the property twins
    // do.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Both pins the oracle asked for, plus the layout guard its map read needs.
    walker_guard_mapdict_instance_shape(ctx, op.pc, obj, concrete_obj, w_type, version_tag, map)?;
    // The version-tag pin above does not cover an in-place cell write.  Same
    // getfield and `guard_value` as [`super::walker_promote_object_mutable_cell`]
    // on `ExceptionInlineReceiverGuard::attr_cell`.
    if !attr_cell.is_null() {
        super::walker_promote_object_mutable_cell(ctx, op.pc, attr_cell, w_getattr)?;
    }
    // The pins above make the DESCRIPTOR a constant; they say nothing about the
    // callable inside it.  Re-initialising an installed wrapper swaps
    // `w_function` without touching the owner type's version tag, which is the
    // only thing those pins hold, so the unwrapped callable needs a pin of its
    // own.  `function.py:673`/`:720` declares that slot `w_function?`, and
    // `w_*_set_func` now forces the invalidation, so this is the declared
    // marker rather than the `GuardValue` that stood in for it — the fold bakes
    // `w_func` below and never reads the slot back.
    if let Some(field) = wrapper_field {
        walker_pin_descriptor_slot(ctx, op.pc, w_getattr, field.quasi_descr())?;
    }

    let name_obj =
        pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(name.as_str()))
            as pyre_object::PyObjectRef;
    let name_const = ctx.trace_ctx.const_ref(name_obj as i64);
    let leading_arg = match leading {
        // The live receiver box: baking it would collapse instances that share
        // this shape but not this identity.
        HookLeading::Receiver => Some((obj, concrete_obj)),
        // The class the `w_class` guard above already pinned.
        HookLeading::Class => Some((ctx.trace_ctx.const_ref(w_type as i64), w_type)),
        HookLeading::None => None,
    };
    // `[__getattr__, <self-placeholder>, <bound arg>?, name]`: the method-form
    // call header the inline plumbing expects, then the positional args.
    let mut arg_concretes = vec![ConcreteValue::Ref(w_func), ConcreteValue::Null];
    let mut callee_args = Vec::with_capacity(2);
    let mut callee_arg_concretes = Vec::with_capacity(2);
    if let Some((arg, concrete)) = leading_arg {
        arg_concretes.push(ConcreteValue::Ref(concrete));
        callee_args.push(arg);
        callee_arg_concretes.push(ConcreteValue::Ref(concrete));
    }
    arg_concretes.push(ConcreteValue::Ref(name_obj));
    callee_args.push(name_const);
    callee_arg_concretes.push(ConcreteValue::Ref(name_obj));
    let getattr_const = ctx.trace_ctx.const_ref(w_func as i64);
    let inlined = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        getattr_const,
        r_args,
        call_descr,
        'r',
        dst,
        w_func,
        getattr_const,
        w_func,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        true,
        None,
        w_code,
        nparams,
        has_closure,
        // The class and version pins are already emitted above, alongside the
        // map pin this route additionally owes.
        None,
        None,
        // The same LOAD_ATTR entry [`try_walker_inline_property_get`] admits.
        true,
        false,
        None,
        None,
        false,
        None,
        // `StdObjSpace.getattr` (`objspace/std/objspace.py`) wraps
        // the `__getattr__` fallback call in `enrich_attribute_error`, which
        // the `load_attr_fn` residual this route replaces reached through
        // `getattr_str`.
        Some(AttributeErrorInlineContext {
            obj,
            obj_concrete: concrete_obj,
            name: name_const,
            name_concrete: name_obj,
        }),
        None,
        None,
        Vec::new(),
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// `descroperation.py DescrOperation.get_and_call_function` for a
/// `__getattr__` hook that is not an exact `Function`, `classmethod`, or
/// `staticmethod`.
///
/// Those three spellings unwrap above and call the function with the name.
/// Every other descriptor is `space.get(w_descr, w_obj)` followed by
/// `space.call_args` on that result. The `LOAD_ATTR` result is the call, so
/// recording only `__get__` would publish the bound callable where the hook
/// returns the call's value.
///
/// `__get__` is handed off through `intermediate_result` and then called.
/// That hand-off aborts when `__get__` committed an effect, because a later
/// decline would re-enter this `LOAD_ATTR` and run the effect again.
#[allow(clippy::too_many_arguments)]
fn try_walker_inline_getattr_hook_through_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
    w_type: pyre_object::PyObjectRef,
    version_tag: u64,
    map: pyre_interpreter::objspace::std::mapdict::MapRef,
    w_getattr: pyre_object::PyObjectRef,
    attr_cell: pyre_object::PyObjectRef,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let Some((descr_type, descr_version_tag, descr_map, w_get)) =
        (unsafe { pyre_interpreter::objspace::std::mapdict::python_descr_get_pins(w_getattr) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_get) }) else {
        return Ok(None);
    };
    // `DescrOperation.get` calls `__get__(descr, obj, type)`.
    if nparams != 3 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }

    let get_const = ctx.trace_ctx.const_ref(w_get as i64);
    let descr_const = ctx.trace_ctx.const_ref(w_getattr as i64);
    let owner_const = ctx.trace_ctx.const_ref(w_type as i64);
    let name_obj = pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(name))
        as pyre_object::PyObjectRef;
    let name_const = ctx.trace_ctx.const_ref(name_obj as i64);
    let attr_ctx = AttributeErrorInlineContext {
        obj,
        obj_concrete: concrete_obj,
        name: name_const,
        name_concrete: name_obj,
    };
    let arg_concretes = vec![
        ConcreteValue::Ref(w_get),
        ConcreteValue::Null,
        ConcreteValue::Ref(w_getattr),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(w_type),
    ];
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    walker_guard_mapdict_instance_shape(ctx, op.pc, obj, concrete_obj, w_type, version_tag, map)?;
    if !attr_cell.is_null() {
        super::walker_promote_object_mutable_cell(ctx, op.pc, attr_cell, w_getattr)?;
    }
    walker_guard_mapdict_instance_shape(
        ctx,
        op.pc,
        descr_const,
        w_getattr,
        descr_type,
        descr_version_tag,
        descr_map,
    )?;

    let mut handed: Option<(OpRef, ConcreteValue)> = None;
    let got = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        get_const,
        r_args,
        call_descr,
        'r',
        dst,
        w_get,
        get_const,
        w_get,
        arg_concretes,
        vec![descr_const, obj, owner_const],
        vec![
            ConcreteValue::Ref(w_getattr),
            ConcreteValue::Ref(concrete_obj),
            ConcreteValue::Ref(w_type),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        false,
        None,
        Some(&mut handed),
        false,
        None,
        Some(attr_ctx),
        None,
        None,
        Vec::new(),
        None,
    )?;
    let Some((outcome, next_pc)) = got else {
        cut_declined_subwalk(ctx, pre_fold_pos);
        return Ok(None);
    };
    // `__get__` raised, or the opcode otherwise finished. That outcome is
    // the attribute access; calling a return value that does not exist
    // would drop it.
    if !matches!(outcome, DispatchOutcome::Continue) {
        return Ok(Some((outcome, next_pc)));
    }
    let Some((bound_op, ConcreteValue::Ref(bound_fn))) = handed else {
        cut_declined_subwalk(ctx, pre_fold_pos);
        return Ok(None);
    };
    if bound_fn.is_null() {
        cut_declined_subwalk(ctx, pre_fold_pos);
        return Ok(None);
    }
    let Some((call_code, call_nparams, call_closure)) =
        (unsafe { resolve_inlinable_callee(bound_fn) })
    else {
        cut_declined_subwalk(ctx, pre_fold_pos);
        return Ok(None);
    };
    // A loop in the bound callable is the same refusal as the exact-wrapper
    // arm: the hook stays one residual rather than a bridge per backedge.
    if sub_jitcode_body_facts_for_code(call_code).is_none_or(|facts| facts.owns_loop_header) {
        cut_declined_subwalk(ctx, pre_fold_pos);
        return Ok(None);
    }
    let called = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        bound_op,
        r_args,
        call_descr,
        'r',
        dst,
        bound_fn,
        bound_op,
        bound_fn,
        vec![
            ConcreteValue::Ref(bound_fn),
            ConcreteValue::Null,
            ConcreteValue::Ref(name_obj),
        ],
        vec![name_const],
        vec![ConcreteValue::Ref(name_obj)],
        true,
        None,
        call_code,
        call_nparams,
        call_closure,
        None,
        None,
        true,
        false,
        None,
        None,
        false,
        None,
        Some(attr_ctx),
        None,
        None,
        Vec::new(),
        None,
    )?;
    if called.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(called)
}

/// Inline a `property` setter store (`obj.value = x`) after the plain-attribute
/// mapdict store fold declines because the attribute is a data descriptor — the
/// setter twin of [`try_walker_inline_property_get`].  Pin the receiver class +
/// version tag (so the property lookup const-folds), then inline `fset(obj, x)`
/// with the receiver as `self`; the setter body (commonly `self._value =
/// value`) folds to a guarded slot store instead of the opaque `setattr`
/// residual.  Same leaf-only / top-frame restrictions and residual fall-through
/// as the getter fold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_property_set<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    value: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(concrete_value) = walker_concrete_ref_object(ctx, value) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_descr, fset)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::property_set_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(fset) }) else {
        return Ok(None);
    };
    if nparams != 2 {
        return Ok(None);
    }
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    let Some((setter_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };
    if body_facts.exc_override_has_nested_call {
        return Ok(None);
    }

    // `[fset, <self-placeholder>, obj, value]`: the method-form call header the
    // inline plumbing expects, then the two positional args (`self`, `value`).
    let arg_concretes = vec![
        ConcreteValue::Ref(fset),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_value),
    ];
    let fset_const = ctx.trace_ctx.const_ref(fset as i64);
    // Rewind point for the same reason as the getter twin.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    walker_pin_descriptor_slot(ctx, op.pc, w_descr, crate::descr::property_fset_descr())?;
    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        fset_const,
        r_args,
        call_descr,
        dst_bank,
        dst,
        fset,
        fset_const,
        fset,
        arg_concretes,
        vec![obj, value],
        vec![
            ConcreteValue::Ref(concrete_obj),
            ConcreteValue::Ref(concrete_value),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag, None)),
        None,
        // STORE_ATTR, same standing as the getter above.
        true,
        false,
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Whether a concrete object is the canonical machine-word `int` layout that
/// the range virtualizer can unbox.  Longs and int subclasses stay on the
/// residual path even though Python-level `type(x) is int` may hold for the
/// separate long payload layout.
pub(crate) fn walker_is_exact_machine_int_concrete(obj: pyre_object::PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return true;
    }
    if obj.is_null() {
        return false;
    }
    unsafe {
        std::ptr::eq((*obj).ob_type, &pyre_object::pyobject::INT_TYPE)
            && std::ptr::eq(
                (*obj).w_class,
                pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE),
            )
    }
}

/// A fully preflighted user `__index__` call nested inside a builtin.
/// Keeping the resolved pieces together lets `range` preflight every bound
/// before the first conversion emits guards or enters a callee frame.
pub(crate) struct IndexInlineCandidate {
    arg: OpRef,
    concrete_arg: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
    version_tag: u64,
    method: pyre_object::PyObjectRef,
    attr_cell: pyre_object::PyObjectRef,
    w_code: *const (),
    nparams: usize,
    has_closure: bool,
}

/// Resolve and statically preflight the call made by
/// `descroperation.py _index`.  The body is held to the existing
/// strict straight-line inline shape; a non-user descriptor, loop/raise body,
/// or other unsupported function shape leaves the enclosing builtin on its
/// ordinary residual path without having emitted anything.
///
/// The body must additionally pass [`index_inline_sample_safe`].  This call is
/// nested inside a builtin the caller keeps pinned at its own CALL boundary, so
/// every guard the caller emits afterwards resumes by re-entering that CALL, and
/// the interpreter then performs `space.index` again — a second `__index__` on
/// an object the inlined body already ran once.  The caller also declines on
/// paths that only rewind the trace, which likewise cannot undo an executed
/// body.  Both are harmless exactly when re-running the body changes nothing.
///
/// That predicate settles it before the body runs for every shape it can read
/// off the bytecode, which is the only place a writing body can be caught: the
/// executed-effect odometer reports a write only after it has happened.  It also
/// rejects a branch, so the odometer's verdict on the one recorded path speaks
/// for every later execution as well.
fn index_prepare_diag(why: &str) {
    if fbw_inline_diag_enabled() {
        eprintln!("[index-prepare-decline] why={why}");
    }
}

pub(crate) fn prepare_walker_inline_index<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    arg: OpRef,
    concrete_arg: pyre_object::PyObjectRef,
) -> Option<IndexInlineCandidate> {
    // Every other lever that inlines app-level Python out of a specializer
    // declines once it is itself inside a callee sub-walk — `try_walker_inline_
    // type_call`, `..._property_get`, `..._property_set` and
    // `..._subscr_getitem` all lead with this.
    if !ctx.is_authoritative_executor || ctx.fbw_mode.inline_subwalk {
        index_prepare_diag("not-authoritative-or-subwalk");
        return None;
    }
    let Some((w_type, version_tag, method, attr_cell)) =
        (unsafe { pyre_interpreter::baseobjspace::index_fast_path(concrete_arg) })
    else {
        index_prepare_diag("index-fast-path");
        return None;
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        index_prepare_diag("callee-not-inlinable");
        return None;
    };
    // `get_and_call_function(w_impl, w_obj)` supplies exactly `self`.
    if nparams != 1 || has_closure {
        index_prepare_diag("arity-or-closure");
        return None;
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        index_prepare_diag("no-body-facts");
        return None;
    };
    if body_facts.has_abort_permanent
        || body_facts.owns_loop_header
        || body_facts.contains_raise
        || !body_facts.index_sample_safe
    {
        if fbw_inline_diag_enabled() {
            let keys = (!body_facts.index_sample_safe)
                .then(|| crate::state::sub_jitcode_body_for_code(w_code))
                .flatten()
                .map(|body| {
                    crate::jitcode_runtime::decoded_ops(body.code)
                        .map(|op| op.key)
                        .collect::<Vec<_>>()
                });
            index_prepare_diag(&format!(
                "body-facts abort={} loop={} raise={} sample_safe={} keys={keys:?}",
                body_facts.has_abort_permanent,
                body_facts.owns_loop_header,
                body_facts.contains_raise,
                body_facts.index_sample_safe,
            ));
        }
        return None;
    }
    // `index_inline_sample_safe` admits `LOAD_ATTR` because the residual covers
    // both a plain instance read and a descriptor whose getter runs app-level
    // code, and the bytecode cannot tell them apart.  Separate them here, while
    // declining is still free: every name the body can name has to be a plain
    // mapdict slot on this very receiver, which is the question
    // `mapdict.py LOAD_ATTR_caching` asks.  A property, a slot this
    // receiver does not carry, and a read chained through some other object all
    // fail it.  Deciding it after the body has run would be too late — a
    // getter's effect has happened by then.  Both storage shapes count: an
    // `__index__` returning an int attribute holds it unboxed
    // (`mapdict.py _prim_direct_read`), which is the common case here.
    let mut name_idx = 0usize;
    while let Some(name) =
        crate::jitcode_dispatch::walker_load_name_from_code(w_code as usize, name_idx)
    {
        let plain_slot = unsafe {
            pyre_interpreter::objspace::std::mapdict::load_attr_fast_path(concrete_arg, &name)
                .is_some()
                || pyre_interpreter::objspace::std::mapdict::load_attr_unboxed_fast_path(
                    concrete_arg,
                    &name,
                )
                .is_some()
        };
        if !plain_slot {
            index_prepare_diag("load-attr-not-plain-slot");
            return None;
        }
        name_idx += 1;
    }
    let body = crate::state::sub_jitcode_body_for_code(w_code)?;
    let (callee_descr_refs, _, _) = crate::state::sub_jitcode_descr_pool_for_code(w_code)?;
    let callee_frame_reg = crate::state::ensure_jitcode_index(w_code)
        .filter(|&jc| crate::state::built_as_portal_at(jc))
        .map(|jc| crate::state::portal_red_regs_at(jc).0)
        .unwrap_or(u16::MAX);
    if !callee_fast_path_inlinable(body.code, callee_descr_refs, ctx, callee_frame_reg) {
        index_prepare_diag("fast-path");
        return None;
    }
    Some(IndexInlineCandidate {
        arg,
        concrete_arg,
        w_type,
        version_tag,
        method,
        attr_cell,
        w_code,
        nparams,
        has_closure,
    })
}

/// Inline one preflighted `_index` user call and return its box to the
/// enclosing builtin without writing the builtin residual's destination.
pub(crate) fn try_walker_inline_index<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    candidate: IndexInlineCandidate,
) -> Result<Option<(OpRef, ConcreteValue)>, DispatchError> {
    let IndexInlineCandidate {
        arg,
        concrete_arg,
        w_type,
        version_tag,
        method,
        attr_cell,
        w_code,
        nparams,
        has_closure,
    } = candidate;
    let cell_guard = (!attr_cell.is_null()).then_some((attr_cell, method));
    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_arg),
    ];
    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let mut result = None;
    let inlined = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![arg],
        vec![ConcreteValue::Ref(concrete_arg)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((arg, concrete_arg, w_type, version_tag, cell_guard)),
        None,
        // The entry is the `range(...)` CALL this `__index__` is nested in.
        // `index_inline_sample_safe` is what makes the re-run free.
        true,
        false,
        None,
        Some(&mut result),
        true,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
    )?;
    match (inlined, result) {
        (Some((DispatchOutcome::Continue, next_pc)), Some(result)) if next_pc == op.next_pc => {
            Ok(Some(result))
        }
        (None, None) => Ok(None),
        _ => Err(DispatchError::callee_inline_unsupported(op.pc)),
    }
}

/// Inline `obj[key]` into the receiver type's Python `__getitem__`.
///
/// `descroperation.py DescrOperation.getitem` resolves `__getitem__`
/// on the receiver's type and calls it; PyPy traces through that call and
/// inlines the body.  pyre lowers BINARY_SUBSCR to one `binary_op` residual,
/// and `try_walker_specialize_subscr` only recognizes the canonical tuple and
/// exact-`list[int]` storage shapes — so a user `__getitem__` never reaches the
/// inline plumbing at all and runs as a fresh interpreter frame behind a
/// `CALL_MAY_FORCE` on every iteration.  This route closes that gap: pin the
/// receiver class + version tag so the MRO lookup const-folds, then enter the
/// resolved-callee inline with the receiver as `self`.
///
/// The gate is `getitem_fast_path`, which admits exactly the receivers whose
/// subscript the type's own MRO owns — a user instance, or a builtin sequence
/// layout that overrides `__getitem__`.  A callee owning a loop header is
/// declined for the reason [`callee_body_owns_loop_header`] documents.  Unlike
/// the property folds this admits a branching, raising body: `__getitem__`
/// raising `IndexError` at the end of a sequence is the shape worth inlining,
/// not an edge case.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_subscr_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: Option<OpRef>,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    // `binary_op_fn(obj, key, tag)` — the two Ref operands are the whole
    // subscript; the operator tag rides the Int list.
    let [obj, key] = r_args else {
        return Ok(None);
    };
    let (obj, key) = (*obj, *key);
    let (Some(concrete_obj), Some(concrete_key)) = (
        walker_concrete_ref_object(ctx, obj),
        walker_concrete_ref_object(ctx, key),
    ) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_getitem, attr_cell)) =
        (unsafe { pyre_interpreter::baseobjspace::getitem_fast_path(concrete_obj) })
    else {
        return Ok(None);
    };
    // Null unless the namespace entry is an `ObjectMutableCell`, which is what
    // a rebound `__getitem__` leaves behind.  Baking the method on the version
    // tag alone would keep calling the function the first store installed.
    let cell_guard = (!attr_cell.is_null()).then_some((attr_cell, w_getitem));
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_getitem) })
    else {
        return Ok(None);
    };
    // `__getitem__(self, key)` — any other arity is a shape
    // `get_and_call_function` would reject before the body runs.
    if nparams != 2 {
        return Ok(None);
    }
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }

    // `[__getitem__, <self-placeholder>, obj, key]`: the method-form call
    // header the inline plumbing expects, then the two positional args.
    let arg_concretes = vec![
        ConcreteValue::Ref(w_getitem),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_key),
    ];
    let getitem_const = ctx.trace_ctx.const_ref(w_getitem as i64);
    try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        // A helper-descent entry carries no funcptr operand of its own; the
        // method constant stands for the call the way the binop route's does.
        funcptr.unwrap_or(getitem_const),
        r_args,
        call_descr,
        dst_bank,
        dst,
        w_getitem,
        getitem_const,
        w_getitem,
        arg_concretes,
        vec![obj, key],
        vec![
            ConcreteValue::Ref(concrete_obj),
            ConcreteValue::Ref(concrete_key),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag, cell_guard)),
        None,
        // `entry_is_call_boundary`. What decides it is whether the abort
        // can name this entry as a resume coordinate, not whether the entry
        // is spelled CALL. `__getitem__` has no reflected half, so no body
        // reached here can answer `NotImplemented` and demand its descent
        // be taken back.
        binop_rewind_enabled(),
        false,
        None,
    )
}

/// Inline `len(obj)` into the receiver type's Python `__len__`.
///
/// `descroperation.py _len` resolves `__len__` on the receiver's type and
/// calls it; `len` then runs `space.index` over the answer and checks it with
/// `_check_len_result`.  pyre reaches that whole chain through the generated
/// `space.len` wrapper, and the wrapper's generic lookup and error arms carry
/// un-lowered helper calls, so the body-wide descent scan refuses the descent
/// and every `len(obj)` over a user instance runs as a fresh interpreter frame
/// behind a `CALL_MAY_FORCE`.  [`exact_builtin_len_shortcut_receiver`] walks
/// the receivers whose length is a layout read past that same scan; this route
/// is the other half, for the receivers whose length really is a call.
///
/// The gate is `len_fast_path`, which admits exactly the receivers `_len`
/// dispatches a method for — a user instance, or a builtin subclass overriding
/// `__len__`.  Pinning the receiver's class and version tag is what makes that
/// MRO answer constant for the trace, and the inline plumbing owes those
/// guards the way the subscript route does.
///
/// Neither check around the call is dropped, on EITHER exit.  On the traced
/// path `OperatorTail::Len` admits only the box that makes `space.index` the
/// identity and emits `_check_len_result`'s nonnegative test as a guard; on
/// the deopt path the same `OperatorTail::Len` is the resume level that runs
/// `baseobjspace::len_result_tail` for real.  Without the second half the
/// first is decoration: a guard inside the inlined body returns the callee's
/// raw box straight into `len`'s result register
/// (`crate::operator_continuation`).
#[allow(clippy::too_many_arguments)]
fn try_walker_inline_len_dunder<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    receiver_op: OpRef,
    concrete_receiver: pyre_object::PyObjectRef,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some((w_type, version_tag, w_len, attr_cell)) =
        (unsafe { pyre_interpreter::baseobjspace::len_fast_path(concrete_receiver) })
    else {
        return Ok(None);
    };
    // A rebound `__len__` leaves an `ObjectMutableCell` in the namespace; the
    // version tag does not move when the next store updates it in place, so the
    // payload is promoted rather than baked.  See the `__getitem__` route.
    let cell_guard = (!attr_cell.is_null()).then_some((attr_cell, w_len));
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_len) }) else {
        return Ok(None);
    };
    // `__len__(self)` — any other arity is a shape `get_and_call_function`
    // would reject before the body runs.
    if nparams != 1 {
        return Ok(None);
    }
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }

    // `[__len__, <self-placeholder>, obj]`: the method-form call header the
    // inline plumbing expects, then the one positional argument.
    let arg_concretes = vec![
        ConcreteValue::Ref(w_len),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
    ];
    let len_const = ctx.trace_ctx.const_ref(w_len as i64);
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let inlined = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        'r',
        dst,
        w_len,
        len_const,
        w_len,
        arg_concretes,
        vec![receiver_op],
        vec![ConcreteValue::Ref(concrete_receiver)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((
            receiver_op,
            concrete_receiver,
            w_type,
            version_tag,
            cell_guard,
        )),
        None,
        // The entry is a Python CALL of its own, which is what the abort
        // rewind names.
        true,
        false,
        None,
        None,
        false,
        None,
        None,
        Some(crate::operator_continuation::OperatorTail::Len),
        None,
        Vec::new(),
        None,
    )?;
    // A refused result — `__len__` answering a non-int or a negative — declines
    // from inside the call above after the body has been walked.  Cut that
    // emission back so the caller's residual runs the operator itself.
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// Turn `__bool__`'s returned box into the int `is_true` answers.
///
/// `descroperation.py is_true` accepts only `w_False` and `w_True`. The
/// unbox guards that class and reads `intval`, so a later iteration that
/// returns the other singleton or a non-bool resumes the Truth residual
/// with its result slot still pending. `None` is a decline: the caller cuts
/// the emission and the residual runs `is_true_lookup`.
fn truth_bool_result_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    value: OpRef,
    concrete_for_shadow: ConcreteValue,
    executed_effects_before: usize,
) -> Result<Option<(OpRef, ConcreteValue)>, DispatchError> {
    if fbw_executed_effect_count() != executed_effects_before {
        return Ok(None);
    }
    let ConcreteValue::Ref(obj) = concrete_for_shadow else {
        return Ok(None);
    };
    if obj.is_null() || unsafe { !pyre_object::is_bool(obj) } {
        return Ok(None);
    }
    let bit = i64::from(unsafe { pyre_object::w_bool_get_value(obj) });
    let (int_type, intval_descr) = crate::state::int_or_bool_unbox_type_descr(obj);
    let expected_class = walker_numeric_builtin_class(obj);
    let raw = walker_unbox_int_exact(ctx, op_pc, value, int_type, intval_descr, expected_class)?;
    Ok(Some((raw, ConcreteValue::Int(bit))))
}

/// Inline an overriding Python `__bool__` under the Truth residual.
///
/// `is_true` sends anything that is not an exact builtin to
/// `is_true_lookup`, and that function is `dont_look_inside` because the
/// generic lookup also reaches `space.index` and the TypeError formatter.
/// The override itself is ordinary Python: `subclass_special_override`
/// distinguishes it from an inherited slot, and `DescrOperation.is_true`
/// (`descroperation.py`) calls it through `get_and_call_function`. This
/// route inlines that call and leaves the bool-to-int check to
/// [`OperatorTail::Truth`].
///
/// Exact builtins stay on `try_walker_orthodox_truth`. A missing `__bool__`,
/// an inherited slot, or a callee this inline cannot enter declines, and
/// the Truth residual runs `is_true_lookup` unchanged. The payload is never
/// the answer: `LiarBool(0)` is falsy as an int and true by `__bool__`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_truth_bool<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if ctx.fbw_mode.inline_subwalk || !ctx.is_authoritative_executor || r_args.len() != 1 {
        return Ok(None);
    }

    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[truth-bool-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    let receiver_op = r_args[0];
    let Some(concrete_receiver) = walker_concrete_ref_object(ctx, receiver_op) else {
        decline!("operand has no concrete ref");
    };
    if concrete_receiver.is_null()
        || (pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(concrete_receiver))
        || unsafe { pyre_object::is_exact_builtin_instance(concrete_receiver) }
    {
        decline!("exact builtin stays on truth_value");
    }
    let Some(w_type_nn) = pyre_interpreter::typedef::r#type(concrete_receiver) else {
        decline!("operand has no type");
    };
    let w_type = w_type_nn.as_ptr();
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        decline!(format_args!("class {} has no version tag", unsafe {
            pyre_object::typeobject::w_type_get_name(w_type)
        }));
    }
    // `subclass_special_override`: an inherited slot is the layout truth,
    // not an override. Inlining it would call the builtin `__bool__` and
    // skip a subclass that only overrides `__len__`.
    let Some(method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__bool__") })
    else {
        decline!("no __bool__");
    };
    let base = unsafe { pyre_object::get_instantiate(&*(*concrete_receiver).ob_type) };
    if !base.is_null()
        && let Some(inherited) =
            (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(base, "__bool__") })
        && std::ptr::eq(inherited, method)
    {
        decline!("__bool__ is the inherited slot");
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        decline!("__bool__ is not inlinable Python");
    };
    if nparams != 1 {
        decline!(format_args!("__bool__ arity {nparams}"));
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        decline!("__bool__ has no jitcode body");
    };
    if body_facts.owns_loop_header {
        decline!("__bool__ owns a loop header");
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let receiver_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(concrete_receiver);
    let type_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let method_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(method);
    let concrete_receiver = pyre_object::gc_roots::shadow_stack_get(receiver_root);
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_root);
    let method = pyre_object::gc_roots::shadow_stack_get(method_root);
    let cell_guard = unsafe { inline_attr_cell_guard(w_type, "__bool__", method) };

    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
    ];
    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let inlined = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        'i',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![receiver_op],
        vec![ConcreteValue::Ref(concrete_receiver)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((
            receiver_op,
            concrete_receiver,
            w_type,
            version_tag,
            cell_guard,
        )),
        None,
        // Not a Python CALL. A `Clean` body needs no boundary; a dirty one
        // stays on the Truth residual rather than latching a CALL resume
        // this opcode does not have.
        false,
        false,
        None,
        None,
        false,
        None,
        None,
        Some(crate::operator_continuation::OperatorTail::Truth),
        None,
        Vec::new(),
        None,
    )?;
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    } else if fbw_inline_diag_enabled() {
        eprintln!("[truth-bool] pc={} class={}", op.pc, unsafe {
            pyre_object::typeobject::w_type_get_name(w_type)
        });
    }
    Ok(inlined)
}

/// Inline a user instance's Python `__iter__` directly under GET_ITER.
///
/// `iter`'s instance arm dispatches the method and then runs
/// `iter_check_is_iterator` on what it returned.  Upstream pays nothing for
/// either: `DescrOperation.iter` resolves `__iter__` off the promoted class,
/// and the `space.lookup(w_iter, '__next__')` that follows folds away because
/// the returned box's class is promoted too.  Here the method stayed an opaque
/// `CallMayForce` — one interpreter frame, plus the virtualref bracket that
/// forces the caller's frame — for every `for x in obj` over a user iterator.
///
/// The check is reproduced rather than dropped: this route admits only the
/// shape where the inlined body hands back the receiver's own box, so the
/// class the receiver is already guarded to, under the `_version_tag?` pin the
/// same guard takes, is the class the check would have looked `__next__` up
/// on.  The lookup is then a record-time question, and any other returned box
/// falls back to the residual, which runs the check itself. GET_ITER's caller
/// image carries its consumed iterable as the one proof operand, so a seeded
/// callee guard resumes after GET_ITER with the result slot pending just like
/// LOAD_ATTR; that is what also admits a body whose effects are protected by
/// the per-frame resume path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_instance_iter<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }

    let obj_op = r_args[0];
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj_op) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_iter, attr_cell)) =
        (unsafe { pyre_interpreter::baseobjspace::iter_fast_path(concrete_obj) })
    else {
        return Ok(None);
    };
    // `iter_check_is_iterator` on the receiver itself, decided here because
    // the guard below pins the class it reads and the version tag that would
    // revoke the answer.  A receiver that is not its own iterator declines
    // rather than reaching the identity test after the body has run.
    if unsafe { pyre_interpreter::baseobjspace::next_fast_path(concrete_obj) }.is_none() {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_iter) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }

    let pre_emit_pos = ctx.trace_ctx.get_trace_position();
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(concrete_obj);
    let type_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let iter_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_iter);
    let cell_root = (!attr_cell.is_null()).then(|| {
        let cell_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(attr_cell);
        cell_root
    });
    let code_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_code as pyre_object::PyObjectRef);

    let concrete_obj = pyre_object::gc_roots::shadow_stack_get(obj_root);
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_root);
    let w_iter = pyre_object::gc_roots::shadow_stack_get(iter_root);
    let attr_cell = cell_root.map_or(std::ptr::null_mut(), |cell_root| {
        pyre_object::gc_roots::shadow_stack_get(cell_root)
    });
    let cell_guard = (!attr_cell.is_null()).then_some((attr_cell, w_iter));
    let w_code = pyre_object::gc_roots::shadow_stack_get(code_root) as *const ();

    let iter_const = ctx.trace_ctx.const_ref(w_iter as i64);
    let executed_effects_before = fbw_executed_effect_count();
    let inline = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        w_iter,
        iter_const,
        w_iter,
        vec![
            ConcreteValue::Ref(w_iter),
            ConcreteValue::Null,
            ConcreteValue::Ref(concrete_obj),
        ],
        vec![obj_op],
        vec![ConcreteValue::Ref(concrete_obj)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj_op, concrete_obj, w_type, version_tag, cell_guard)),
        None,
        // GET_ITER consumes one iterable and produces one iterator. Its
        // caller-operand shape is recorded by `caller_operand_slots`, so the
        // seeded callee can resume forward with the pending result slot; this
        // is the same boundary property LOAD_ATTR relies on.
        true,
        false,
        None,
        None,
        false,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
    );
    let inline_resume_pc = match inline {
        Ok(Some((DispatchOutcome::Continue, next))) => next,
        Err(error @ DispatchError::TraceTooLong { .. }) => return Err(error),
        outcome => {
            // The same odometer the `__next__` route reads: once the sub-walk
            // has executed a concrete effect, `cut_trace_with_snapshots`
            // cannot undo it, and falling through to the residual would run
            // `__iter__` a second time.
            if fbw_executed_effect_count() != executed_effects_before {
                return Err(match outcome {
                    Err(e) => e,
                    _ => DispatchError::callee_inline_unsupported(op.pc),
                });
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    };
    // A `SubRaise` routed to the caller's handler wrote no `dst`, and the
    // check upstream would not have run either.
    if inline_resume_pc != op.next_pc {
        return Ok(Some((DispatchOutcome::Continue, inline_resume_pc)));
    }

    // The identity the `iter_check_is_iterator` decision above rests on, as a
    // property of the trace rather than of this recording: the body handed
    // back the very box the receiver guard pins, so the class the check reads
    // is that guard's class on every entry.  Any other result reaches the
    // residual, which runs the check itself.
    if ctx.registers_r.get(dst).expect("ref register in range") != obj_op {
        if fbw_executed_effect_count() != executed_effects_before {
            return Err(DispatchError::callee_inline_unsupported(op.pc));
        }
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    ctx.frame_state.borrow_mut().vstack_last_ref = obj_op;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Inline a user instance's Python `__next__` directly under FOR_ITER.
///
/// The instance arm of `space.next` forwards the method's exception to the
/// FOR_ITER catch arm.  The callee is resumed through a seeded multi-frame
/// snapshot, and every route/callee guard is keyed so failure resumes in the
/// blackhole without compiling the unsafe mid-callee exhaustion bridge.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_instance_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }

    // A sub-walk is admitted: `walker_foriter_green_key` keys the demotion off
    // the JitCode the walk is executing, so a `for` loop in an inlined callee
    // gets its own key rather than a caller offset read as a callee one.
    let Some(foriter_green_key) = walker_foriter_green_key(ctx, op.pc) else {
        return Ok(None);
    };
    if ctx.trace_ctx.is_bridge_trace
        && crate::trace::instance_next_foriter_bridge_demoted(foriter_green_key)
    {
        return Ok(None);
    }

    let iter_op = r_args[0];
    let Some(iter_obj) = walker_concrete_ref_object(ctx, iter_op) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_next, attr_cell)) =
        (unsafe { pyre_interpreter::baseobjspace::next_fast_path(iter_obj) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_next) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    // Admit a raising `__next__` the way `try_walker_inline_subscr_getitem`
    // admits a raising `__getitem__`: StopIteration at the end of the
    // iterator is the shape worth inlining.  A callee that owns a portal
    // loop header is still declined — that body is a trace of its own.
    if body_facts.owns_loop_header {
        return Ok(None);
    }
    let body_coord = fbw_foriter_body_from_op_pc(ctx, op.pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body_coord);
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();

    let _roots = pyre_object::gc_roots::push_roots();
    let iter_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(iter_obj);
    let type_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let next_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_next);
    let cell_root = (!attr_cell.is_null()).then(|| {
        let cell_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(attr_cell);
        cell_root
    });
    let code_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_code as pyre_object::PyObjectRef);

    let iter_obj = pyre_object::gc_roots::shadow_stack_get(iter_root);
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_root);
    let w_next = pyre_object::gc_roots::shadow_stack_get(next_root);
    let attr_cell = cell_root.map_or(std::ptr::null_mut(), |cell_root| {
        pyre_object::gc_roots::shadow_stack_get(cell_root)
    });
    let cell_guard = (!attr_cell.is_null()).then_some((attr_cell, w_next));
    let w_code = pyre_object::gc_roots::shadow_stack_get(code_root) as *const ();
    let iter_layout = unsafe { (*iter_obj).ob_type } as i64;
    if !iter_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(iter_op) {
        let type_const = ctx.trace_ctx.const_int(iter_layout);
        let descr = majit_metainterp::make_resume_guard_descr_instance_next_foriter(
            Some(OpCode::GuardClass),
            foriter_green_key,
        );
        ctx.trace_ctx
            .record_guard_with_descr(OpCode::GuardClass, &[iter_op, type_const], descr);
        spec_census_record_instance_next_route_guard_keyed();
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx.heap_cache_mut().class_now_known(iter_op);

    let next_const = ctx.trace_ctx.const_ref(w_next as i64);
    let executed_effects_before = fbw_executed_effect_count();
    let inline = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        w_next,
        next_const,
        w_next,
        vec![
            ConcreteValue::Ref(w_next),
            ConcreteValue::Null,
            ConcreteValue::Ref(iter_obj),
        ],
        vec![iter_op],
        vec![ConcreteValue::Ref(iter_obj)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((iter_op, iter_obj, w_type, version_tag, cell_guard)),
        None,
        true,
        false,
        None,
        None,
        false,
        Some(foriter_green_key),
        None,
        None,
        None,
        Vec::new(),
        None,
    );
    let inline_resume_pc = match inline {
        Ok(Some((DispatchOutcome::Continue, next))) => next,
        Err(error @ DispatchError::TraceTooLong { .. }) => return Err(error),
        outcome => {
            // Rewinding the emission and falling through to the caller's
            // residual re-runs the whole `__next__`. Once the sub-walk has
            // executed a concrete effect, `cut_trace_with_snapshots` cannot
            // undo it (it truncates recorded ops and snapshots, nothing else)
            // and the residual `next` compiles a loop that advances the
            // iterator twice per iteration. Surface the decline as an abort
            // there, the same disposition every other caller of the inliner
            // already takes.
            if fbw_executed_effect_count() != executed_effects_before {
                return Err(match outcome {
                    Err(e) => e,
                    _ => DispatchError::callee_inline_unsupported(op.pc),
                });
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    };
    // `Continue` represents either a value return at `op.next_pc`, with `dst`
    // holding the item, or a caught `SubRaise` routed to the caller's handler
    // target, where `dst` was not written (`pyjitpl.py`).  Only the
    // value-return shape can update the FOR_ITER item bookkeeping below.
    if inline_resume_pc != op.next_pc {
        return Ok(Some((DispatchOutcome::Continue, inline_resume_pc)));
    }

    let item_op = ctx.registers_r.get(dst).expect("ref register in range");
    let Some(concrete_item) = walker_concrete_ref_object(ctx, item_op) else {
        return Err(DispatchError::callee_inline_unsupported(op.pc));
    };
    fbw_foriter_inflight_capture(concrete_item, body_coord, false);
    ctx.frame_state.borrow_mut().vstack_last_ref = item_op;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// End a generator-resume sub-walk at the marker its own `yield` lowered to,
/// with the value the generator is suspending on.
///
/// `Instruction::YieldValue` has no residual a trace can carry — the opcode
/// suspends the frame and resumes it in a different stack context — so the
/// codewriter lowers it to `abort_permanent`, and every walk that reaches one
/// gives up on the location for good.  For the walk that is RESUMING that same
/// generator the marker is not a failure: it is the suspension the resume was
/// walked to reach, and the value it suspends with is the `FOR_ITER` item.
///
/// Nothing has to be reconstructed to read that value.  `emit_abort_permanent!`
/// already materializes the pre-opcode operand stack into the virtualizable —
/// one `setarrayitem_vable_r` per live slot, ascending — and syncs
/// `valuestackdepth`, precisely so the interpreter it bails to resumes on a
/// real stack.  The top slot of that stack is what `YIELD_VALUE` pops, so the
/// last of those stores is the yielded value, and the store site latched it
/// ([`fbw_note_generator_stack_store`]).
///
/// `Ok(None)` means "not a yield this walk can take", and the caller falls
/// through to the permanent abort the marker has always raised.  That is the
/// disposition for every precondition below, so a resume that cannot read its
/// own suspension is never worse than no resume at all.
pub(super) fn generator_resume_yield<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(resume) = ctx.fbw_mode.generator_resume else {
        return Ok(None);
    };
    // The body's own markers for unported opcodes are still aborts.  Only the
    // offset the admission resolved the `yield` to is the suspension.
    if op_pc != resume.yield_marker_jit_pc {
        return Ok(None);
    }
    // `valuestackdepth` is absolute (`stack_base_absolute + stack length`), so
    // the value the opcode pops is the slot below the published depth.
    let Some(top) =
        crate::state::concrete_stack_depth(resume.frame).and_then(|depth| depth.checked_sub(1))
    else {
        return Ok(None);
    };
    let Some(store) = fbw_generator_stack_store_at(resume.frame, top) else {
        return Ok(None);
    };
    // The marker leaves the frame in the state the INTERPRETER resumes on: the
    // yielded value still on the stack, and `last_instr` one BEFORE the yield
    // so that a bail re-runs it.  A walk that has taken the value here has
    // already run it, so publish the suspension instead — pop the value and put
    // the coordinate AT the yield, which is what the next `next()` reads back.
    // Without this the generator would hand out the same item twice.
    //
    // The undos are already armed — every one of the marker's own stores went
    // through `durable_resume_frame` first — but arm again rather than depend
    // on that: the note is first-write-per-frame wins, so a redundant call
    // keeps the earlier, pre-walk image.
    fbw_arm_durable_frame_undo(resume.frame);
    // The generator frame is a nonstandard virtualizable: the trace's
    // standard vable stays the caller's.  Write the suspension through
    // SETFIELD_GC on the generator frame so the compiled loop updates that
    // object, not the caller's `last_instr` / `valuestackdepth` slots.
    publish_generator_suspension(
        ctx,
        resume.frame,
        resume.frame_box,
        "valuestackdepth",
        top as i64,
    );
    publish_generator_suspension(
        ctx,
        resume.frame,
        resume.frame_box,
        "last_instr",
        resume.yield_py_pc as i64,
    );
    // `popvalue_maybe_none` (`pyframe.py`) writes None into the slot it pops.
    // The depth store above is that pop's index write; the yielded object is
    // still in the slot, and the next `pushvalue_none` requires the slot to
    // be null. The compiled loop has to store that too: the GC scans the
    // array independently of `valuestackdepth`.
    publish_generator_slot_null(ctx, resume.frame, resume.frame_box, top);
    Ok(Some(store.value))
}

/// A walk that is not resuming a generator ends at `YIELD_VALUE` the way
/// `RETURN_VALUE` ends a frame.
///
/// `pyopcode.py` `YIELD_VALUE` raises `Yield`; `interp_jit.py` `dispatch`
/// catches it with `popvalue()`, `jit.hint(..., force_virtualizable=True)`,
/// and returns that value. The portal then `compile_done_with_this_frame`s.
/// The marker stays `abort_permanent` so a resume sub-walk can still stop on
/// it (`generator_resume_yield`); this path is every other walk.
///
/// `last_instr` is the yield's own Python pc. `dispatch_bytecode` stores
/// `set_last_instr_from_next_instr(fallthrough)` before the opcode, and
/// `execute_yield_value` does not move it, so the next `send` starts at the
/// instruction after the yield. The yielded value is popped: depth is one
/// below the marker's pre-opcode stack, whose top slot is that value.
pub(super) fn portal_yield_frame_exit<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Option<(OpRef, i64)> {
    if ctx.fbw_mode.generator_resume.is_some() {
        return None;
    }
    let Some((py_pc, code_ptr)) = yield_marker_py_pc(ctx, op_pc) else {
        return None;
    };
    let is_yield = unsafe {
        pyre_interpreter::decode_instruction_at(code_ptr.as_ref()?, py_pc).is_some_and(
            |(instr, _)| matches!(instr, pyre_interpreter::Instruction::YieldValue { .. }),
        )
    };
    if !is_yield {
        return None;
    }
    // `vstack_last_ref` is the producer of the current opcode and is cleared
    // at the opcode boundary. The yield marker is the first op of its own
    // opcode, so that clear has already moved the yielded ref onto
    // `vstack_boxes`. An int-bank temporary stays `NONE` there; then there
    // is no ref for this exit to return.
    let state = ctx.frame_state.borrow();
    let mirrored = if ctx.vstack_valid && ctx.vstack_depth > 0 {
        state
            .vstack_boxes
            .get(ctx.vstack_depth - 1)
            .copied()
            .unwrap_or(OpRef::NONE)
    } else {
        OpRef::NONE
    };
    let value = if !mirrored.is_none() {
        mirrored
    } else if !state.vstack_last_ref.is_none() {
        state.vstack_last_ref
    } else {
        OpRef::NONE
    };
    drop(state);
    // An int-bank temporary never lands in the Ref mirror. The marker's
    // `setarrayitem_vable_r` has already written that boxed value into the
    // standard virtualizable, at the slot just below `valuestackdepth`.
    // A loop-header walk can also leave the mirror invalid (`ReturnGenerator`
    // is unmodeled, so the boundary latch drops it) while the value register
    // the marker just stored is still live. That register is the yielded
    // ref even when the virtualizable slot index does not match.
    let value = if value.is_none() {
        portal_yield_tos_from_vable(ctx).unwrap_or(OpRef::NONE)
    } else {
        value
    };
    if value.is_none() {
        return None;
    }
    Some((value, py_pc as i64))
}

/// The yielded ref the `abort_permanent` marker just flushed into the portal
/// virtualizable. `valuestackdepth` is absolute, so the popped slot is one
/// below it.
fn portal_yield_tos_from_vable<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> Option<OpRef> {
    let info = ctx.trace_ctx.virtualizable_info()?;
    let depth_index = info.static_field_index_by_name("valuestackdepth")?;
    let (_, majit_ir::Value::Int(depth)) = ctx.trace_ctx.virtualizable_entry_at(depth_index)?
    else {
        return None;
    };
    let slot = usize::try_from(depth).ok()?.checked_sub(1)?;
    let flat = crate::virtualizable_gen::NUM_VABLE_SCALARS + slot;
    let (op, _) = ctx.trace_ctx.virtualizable_entry_at(flat)?;
    if op.is_none() { None } else { Some(op) }
}

/// Flush the yield suspension after the trace finish has published its own
/// `last_instr`. `emit_abort_permanent` left `py_pc - 1` and the yielded
/// value on the stack; `fbw_terminate_with_finish` may have published the
/// marker's resolved coordinate. The hint runs last so the flushed
/// virtualizable is the post-yield frame.
pub(super) fn flush_yield_exit<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>, py_pc: i64) {
    publish_yield_exit_state(ctx, py_pc);
    // `gen_store_back_in_vable` (`hint_force_virtualizable`) records every
    // virtualizable field, array items included. The walker is not the
    // interpreter, and `synchronize_virtualizable` skips the write when an
    // outer executor owns the struct — a no-replay `Finish` then returns the
    // yielded value without replaying those stores. The live frame the
    // interpreter resumes is `recording_frame_ptr`, so the same array image
    // the hint publishes has to land there before the trace ends.
    publish_yield_vable_array(ctx);
    if let Some(vbox) = ctx.trace_ctx.standard_virtualizable_box() {
        ctx.trace_ctx.gen_store_back_in_vable(vbox);
    }
}

/// Copy every `locals_cells_stack_w` slot of the standard virtualizable onto
/// the live portal frame. Scalar fields are published by
/// [`publish_yield_exit_state`]; this is the array half of
/// `gen_store_back_in_vable`.
fn publish_yield_vable_array<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>) {
    let frame = ctx.session.borrow().recording_frame_ptr;
    if frame == 0 {
        return;
    }
    let len = {
        let Some(lengths) = ctx.trace_ctx.virtualizable_array_lengths() else {
            return;
        };
        lengths.first().copied().unwrap_or(0)
    };
    if len == 0 {
        return;
    }
    // First note wins, so an earlier `f_locals` mirror keeps the pre-walk
    // image. A yield that is the first writer captures the whole array,
    // stack slots included, because this flush writes those too.
    fbw_note_locals_mirror_undo(frame, len);
    let base = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    for slot in 0..len {
        let Some((op, mut value)) = ctx.trace_ctx.virtualizable_entry_at(base + slot) else {
            continue;
        };
        // The array half of `gen_store_back_in_vable` stores the box. A slot
        // whose concrete half is still null keeps that box's stamped value,
        // which is what the recorded `setarrayitem_gc` writes.
        if matches!(value, majit_ir::Value::Void)
            || matches!(value, majit_ir::Value::Ref(r) if r.is_null())
        {
            if let Some(stamped) = ctx.trace_ctx.concrete_of_opref(op) {
                value = stamped;
            }
        }
        let boxed = crate::state::boxed_slot_value_for_type(majit_ir::Type::Ref, &value);
        crate::state::store_live_frame_array_slot(
            frame,
            slot,
            majit_ir::Value::Ref(majit_ir::GcRef(boxed as usize)),
        );
    }
}

/// Python pc and code object of an `abort_permanent` at `op_pc`, when that
/// marker belongs to the jitcode this walk is executing.
fn yield_marker_py_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Option<(usize, *const pyre_interpreter::CodeObject)> {
    if ctx.is_top_level {
        let sym = ctx.fbw_mode.snapshot_sym;
        if sym.is_null() {
            return None;
        }
        // SAFETY: the sym and its jitcode live for this walk; read-only.
        let code_ptr = unsafe {
            let jitcode = (*sym).jitcode();
            if jitcode.is_null() {
                return None;
            }
            let jitcode = &*jitcode;
            jitcode.payload.code_ptr
        };
        let py_pc = fbw_abort_resume_py_pc(unsafe { &*sym }, op_pc)?;
        return Some((py_pc, code_ptr));
    }
    let consts = ctx.inline_callee_consts?;
    let pjc = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)?;
    let py_pc = pjc
        .metadata
        .abort_permanent_py_pc_by_jit_pc
        .iter()
        .find(|&&(off, _)| off as usize == op_pc)
        .map(|&(_, py)| py as usize)?;
    // The callee's `W_Code` lives on `WalkFrameStateData` (`inline_w_code`),
    // not on the consts.
    let w_code = ctx.inline_w_code();
    if w_code == 0 {
        return None;
    }
    let code_ptr = unsafe { pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef) }
        as *const pyre_interpreter::CodeObject;
    Some((py_pc, code_ptr))
}

/// Store the suspension `StepResult::Yield` leaves: `last_instr` at the yield,
/// operand stack one slot shorter.
fn publish_yield_exit_state<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>, py_pc: i64) {
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return;
    };
    let Some(vbox) = ctx.trace_ctx.standard_virtualizable_box() else {
        return;
    };
    if let Some(field_index) = info.static_field_index_by_name("last_instr") {
        let value = ctx.trace_ctx.const_int(py_pc);
        let descr = info.static_field_struct_descr(field_index);
        ctx.trace_ctx.vable_setfield_descr(vbox, value, descr);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            value,
            majit_ir::Value::Int(py_pc),
        );
    }
    let frame = ctx.session.borrow().recording_frame_ptr;
    if frame != 0 {
        fbw_note_last_instr_undo(frame);
        // SAFETY: `frame_layout` pins `last_instr` at this offset.
        unsafe {
            *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize) =
                py_pc as isize;
        }
    }
    let Some(depth_index) = info.static_field_index_by_name("valuestackdepth") else {
        return;
    };
    let Some((_, majit_ir::Value::Int(depth))) = ctx.trace_ctx.virtualizable_entry_at(depth_index)
    else {
        return;
    };
    if depth <= 0 {
        return;
    }
    let popped = depth - 1;
    let value = ctx.trace_ctx.const_int(popped);
    let descr = info.static_field_struct_descr(depth_index);
    ctx.trace_ctx.vable_setfield_descr(vbox, value, descr);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "valuestackdepth",
        value,
        majit_ir::Value::Int(popped),
    );
    if frame != 0 {
        fbw_note_frame_vsd_undo(frame);
        crate::state::set_concrete_stack_depth(frame, popped as usize);
    }
}

/// Write one static field of the suspended generator frame.
///
/// The generator is not the trace's standard virtualizable — that stays the
/// caller's frame — so this is a SETFIELD_GC on the generator object plus the
/// matching live-frame store, not a `mirror_vable_static_to_boxes` into the
/// caller's shadow.
fn publish_generator_suspension<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame: usize,
    frame_box: OpRef,
    static_field_name: &str,
    value: i64,
) {
    let field_index = match static_field_name {
        "last_instr" => 0,
        "valuestackdepth" => 2,
        _ => return,
    };
    crate::state::store_live_frame_static_int(frame, field_index, value);
    let descr = match static_field_name {
        "last_instr" => crate::descr::pyframe_next_instr_descr(),
        "valuestackdepth" => crate::descr::pyframe_stack_depth_descr(),
        _ => return,
    };
    let value_op = ctx.trace_ctx.const_int(value);
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[frame_box, value_op], descr);
}

/// `popvalue_maybe_none` (`pyframe.py`) stores None over the popped slot.
///
/// The generator frame is not the trace's standard virtualizable, so the
/// marker's `setarrayitem_vable_r` did not write this array. Null the live
/// slot and record the same store (`GetfieldGc` of `locals_cells_stack_w`,
/// then `SetarrayitemGc` of `PY_NULL`).
fn publish_generator_slot_null<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame: usize,
    frame_box: OpRef,
    index: usize,
) {
    unsafe {
        (*(frame as *mut pyre_interpreter::PyFrame)).set_locals_w(index, std::ptr::null_mut());
    }
    let locals_descr = crate::descr::pyframe_locals_cells_stack_descr();
    let locals_array =
        ctx.trace_ctx
            .record_op_with_descr(OpCode::GetfieldGcR, &[frame_box], locals_descr);
    let array_ptr =
        unsafe { (*(frame as *mut pyre_interpreter::PyFrame)).locals_cells_stack_w as usize };
    ctx.trace_ctx.set_opref_concrete(
        locals_array,
        majit_ir::Value::Ref(majit_ir::GcRef(array_ptr)),
    );
    let idx = ctx.trace_ctx.const_int(index as i64);
    let null_ref = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetarrayitemGc,
        &[locals_array, idx, null_ref],
        crate::state::pyobject_gcarray_descr(),
    );
}

/// Why a `FOR_ITER` over a suspended generator was not resumed into the trace.
///
/// Named rather than counted so the corpus census answers WHICH precondition
/// stands between the shape and the resume; `Admissible` is the row that
/// measures the reachable population.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GeneratorResumeVerdict {
    /// The `FOR_ITER` operand is not a suspended, resumable generator:
    /// exhausted, already running, not yet started, frameless, or not a
    /// generator at all (`generator_resume_fast_path`).
    NotResumable,
    /// `should_not_inline` — two or more `yield`s. Upstream gives this body
    /// `generatorentry_driver`'s own merge point instead of inlining it.
    SeveralYields,
    /// The body owns cells or reads free variables. The seeded-frame inline
    /// admits existing freevar cells but not fresh cellvar allocation, and the
    /// resume has no seeding step of its own yet.
    CellsOrFreevars,
    /// The body carries an exception table beyond the synthetic wrapper 3.14
    /// puts around every generator to convert an escaping `StopIteration`.
    RealExceptionTable,
    /// No jitcode is installed for the body, so there is nothing to walk.
    NoJitcode,
    /// The body's jitcode publishes no trace-entry offset for the resume
    /// coordinate (`merge_entry_for`), so a walk has no place to start.
    NoResumeEntry,
    /// Every precondition holds.
    Admissible,
}

impl GeneratorResumeVerdict {
    fn label(self) -> &'static str {
        match self {
            Self::NotResumable => "not_resumable",
            Self::SeveralYields => "several_yields",
            Self::CellsOrFreevars => "cells_or_freevars",
            Self::RealExceptionTable => "real_exception_table",
            Self::NoJitcode => "no_jitcode",
            Self::NoResumeEntry => "no_resume_entry",
            Self::Admissible => "admissible",
        }
    }
}

/// Whether this code object's exception table is nothing but the synthetic
/// entry 3.14 wraps every generator body in.
///
/// `codegen.py` emits one depth-zero, `lasti` entry spanning the whole body
/// whose handler is the `CALL_INTRINSIC_1 INTRINSIC_STOPITERATION_ERROR` /
/// `RERAISE` pair, so EVERY generator has a table and the plain
/// "`exceptiontable` is empty" test the seeded inline uses refuses all of
/// them.  An entry written for an actual `try` around the body either omits
/// `lasti` at depth zero or carries a non-zero unwind depth — the same
/// discriminator `code_yields_inside_try` reads.
fn generator_table_is_only_the_stopiteration_wrapper(code: &pyre_interpreter::CodeObject) -> bool {
    pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable)
        .all(|entry| entry.depth == 0 && entry.lasti)
}

/// What the census reports about one `FOR_ITER` over a generator.
pub(crate) struct GeneratorResumeCensus {
    verdict: GeneratorResumeVerdict,
    /// `last_instr + 1`, the coordinate a resume walk would enter at.
    resume_py_pc: usize,
    /// The JitCode byte offset that coordinate resolves to, once it does.
    marker_offset: Option<usize>,
    /// Whether the code object owns a Python-level loop header, read as a
    /// `merge_entry_by_green` entry other than function entry.  This is the
    /// one gate a resume cannot simply refuse: a `while` generator's own loop
    /// header sits between the resume and the yield, and upstream crosses it
    /// exactly once because `should_unroll_one_iteration` answers True for a
    /// generator code object.  Counting the bodies that have one prices that
    /// hook against the ones a straight-line resume would already serve.
    ///
    /// NOT [`callee_body_owns_loop_header`], which asks the jitcode for a
    /// `jit_merge_point` op.  That op is emitted only under `is_true_portal`,
    /// and a generator body is never one, so the jitcode test answers `false`
    /// for every generator whatever its source says.  `merge_entry_by_green`
    /// is `find_loop_header_pcs` plus function entry and is built for every
    /// code object, so it survives the portal distinction.
    owns_loop_header: bool,
    /// `n_py_instrs`, so a `NoResumeEntry` naming an out-of-range coordinate is
    /// distinguishable from one whose tables simply do not resolve it.
    n_py_instrs: usize,
    /// `(py_exact_by_jit_pc.len(), block_head_py_by_jit_pc.len())` — an empty
    /// pair means skeleton metadata, not an unresolvable coordinate.
    tables: (usize, usize),
    /// The `abort_permanent` offset the yield itself lowered to.  It is the
    /// coordinate a resume walk would stop at, and its presence is what says
    /// the marker the yield bakes is reachable from the resume side.
    yield_marker_offset: Option<usize>,
    /// `(ops, merge_points, residual_calls)` over the whole body.
    ops_to_yield: (usize, usize, usize),
}

/// Decide whether a `FOR_ITER` over `iter_obj` names a generator resume the
/// tracer could walk instead of leaving the `jit_next` residual in place.
///
/// This is the admission half of the route, split out so the corpus census
/// can ask the question without emitting anything.
fn generator_resume_verdict(iter_obj: pyre_object::PyObjectRef) -> GeneratorResumeCensus {
    use GeneratorResumeVerdict as V;
    let mut census = GeneratorResumeCensus {
        verdict: V::NotResumable,
        resume_py_pc: usize::MAX,
        marker_offset: None,
        owns_loop_header: false,
        n_py_instrs: 0,
        tables: (0, 0),
        yield_marker_offset: None,
        ops_to_yield: (0, 0, 0),
    };
    macro_rules! decline {
        ($v:expr) => {{
            census.verdict = $v;
            return census;
        }};
    }
    let Some(shape) =
        (unsafe { pyre_interpreter::baseobjspace::generator_resume_fast_path(iter_obj) })
    else {
        decline!(V::NotResumable)
    };
    // `execute_frame` documents execution as starting just after `last_instr`,
    // and `resume_execute_frame` returns exactly `last_instr + 1` after
    // pushing the sent value.
    census.resume_py_pc = shape.last_instr as usize + 1;
    let raw = unsafe { pyre_interpreter::w_code_get_ptr(shape.w_pycode) }
        as *const pyre_interpreter::CodeObject;
    if raw.is_null() {
        decline!(V::NoJitcode)
    }
    let code = unsafe { &*raw };
    if pyre_interpreter::baseobjspace::should_not_inline(code) {
        decline!(V::SeveralYields)
    }
    if !code.cellvars.is_empty() || !code.freevars.is_empty() {
        decline!(V::CellsOrFreevars)
    }
    if !generator_table_is_only_the_stopiteration_wrapper(code) {
        decline!(V::RealExceptionTable)
    }
    let w_pycode = shape.w_pycode as *const ();
    // `sub_jitcode_body_for_code` first: it is the one that BUILDS the per-fn
    // jitcode on demand (`jitcode_for`).  Asking `pyjitcode_for_code` ahead of
    // it reads the store before anything has installed a payload for a code
    // object nothing has ever inlined — which every generator is — so the
    // route answered `NoJitcode` for a body that was merely unbuilt.
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_pycode) else {
        decline!(V::NoJitcode)
    };
    let Some(pjc) = crate::state::pyjitcode_for_code(w_pycode) else {
        decline!(V::NoJitcode)
    };
    census.owns_loop_header = pjc
        .metadata
        .merge_entry_by_green
        .iter()
        .any(|&(py, _)| py != 0);
    census.n_py_instrs = pjc.metadata.n_py_instrs as usize;
    census.tables = (
        pjc.metadata.py_exact_by_jit_pc.len(),
        pjc.metadata.block_head_py_by_jit_pc.len(),
    );
    census.yield_marker_offset = pjc
        .metadata
        .abort_permanent_py_pc_by_jit_pc
        .iter()
        .find(|&&(_, py)| py as usize == shape.last_instr as usize)
        .map(|&(off, _)| off as usize);
    census.marker_offset = generator_resume_marker_offset(&pjc, census.resume_py_pc);
    if census.marker_offset.is_none() {
        decline!(V::NoResumeEntry)
    }
    census.ops_to_yield = generator_resume_op_scan(body.code);
    census.verdict = V::Admissible;
    census
}

/// Whole-body op reconnaissance, reported as
/// `(ops, merge_points, residual_calls)`.
///
/// It prices two questions a resume has to answer before any IR is emitted,
/// without emitting any: whether the body carries the generator's own
/// `jit_merge_point` (upstream crosses one exactly once, via
/// `should_unroll_one_iteration`), and whether it carries a residual call,
/// which would commit an effect the caller's `FOR_ITER` cannot re-run.
///
/// The whole body, not the resume-to-yield span: the resume coordinate is one
/// byte PAST the yield's `abort_permanent` in every case the corpus produces,
/// because the resume block head follows the suspension it resumes from.  The
/// path between them therefore runs FORWARD off the end of the resume block
/// and back through the body's own backward jump, which no linear scan
/// describes — so scan everything and let the totals be an upper bound.
fn generator_resume_op_scan(body_code: &[u8]) -> (usize, usize, usize) {
    let end = body_code.len();
    let mut pc = 0usize;
    let (mut ops, mut merge_points, mut residual_calls) = (0usize, 0usize, 0usize);
    while pc < end {
        let Some(op) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            break;
        };
        ops += 1;
        if op.opname == "jit_merge_point" {
            merge_points += 1;
        } else if op.opname.starts_with("residual_call") {
            residual_calls += 1;
        }
        if op.next_pc <= pc {
            break;
        }
        pc = op.next_pc;
    }
    (ops, merge_points, residual_calls)
}

/// The JitCode byte offset a walk of this body enters at to resume at
/// `resume_py_pc`.
///
/// `merge_entry_for` cannot answer it: `merge_entry_by_green` is built from
/// `find_loop_header_pcs` plus function entry, and a generator's resume
/// coordinate is neither.  Widening THAT table is not the fix — the portal
/// reads it, so publishing resume coordinates there would start portal traces
/// that walk straight into the `abort_permanent` the yield bakes.
///
/// The shipped per-Python-PC tables already carry what the codewriter's own
/// `derive_resume_marker` reads.  `py_exact_by_jit_pc` records one entry per
/// contiguous emission run, so the smallest offset a Python PC owns is that
/// PC's first emitted op — the `first_op_by_py_pc` value — and
/// `block_head_py_by_jit_pc` is the table it resolves against.  Rebuilding the
/// first-op column here and calling the same derivation keeps one definition
/// of the resolution rather than a second spelling of it.
fn generator_resume_marker_offset(
    pjc: &crate::pyjitcode::PyJitCode,
    resume_py_pc: usize,
) -> Option<usize> {
    let metadata = &pjc.metadata;
    let n_py = metadata.n_py_instrs as usize;
    if metadata.py_exact_by_jit_pc.is_empty()
        || metadata.block_head_py_by_jit_pc.is_empty()
        || resume_py_pc >= n_py
    {
        return None;
    }
    let mut first_op_by_py_pc = vec![usize::MAX; n_py];
    for &(off, py) in &metadata.py_exact_by_jit_pc {
        if let Some(slot) = first_op_by_py_pc.get_mut(py as usize) {
            *slot = (*slot).min(off as usize);
        }
    }
    crate::pyjitcode::derive_resume_marker(
        &first_op_by_py_pc,
        &metadata.block_head_py_by_jit_pc,
        resume_py_pc,
    )
}

/// `PYRE_GEN_CENSUS=1`: print one line per `FOR_ITER` whose operand is a
/// generator, naming the verdict.  Reading the population is what says whether
/// the resume route is worth its guards, and a decline reason cannot be read
/// off `spec_gate`'s consulted/fired pair.
fn gen_census_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("PYRE_GEN_CENSUS").is_some());
    *ON
}

fn gen_resume_decline(reason: &str) {
    if gen_census_enabled() || fbw_debug_abort_enabled() {
        eprintln!("[gen-resume] decline {reason}");
    }
}

/// Whether this tracer can look inside `execute_frame` the way
/// `_opimpl_recursive_call` does.
///
/// `can_inline_callable` (`warmstate.py`) is true only when `perform_call`
/// (`pyjitpl.py`) can `newframe` and `capture_resumedata` the resulting
/// MIFrame, and the walk's yield is `dispatch`'s `except Yield` `popvalue()`.
/// That `popvalue` is the top slot `YIELD_VALUE` leaves on the entry stack:
/// `emit_abort_permanent!` stores `pre_opcode_stack` before the fall-through
/// placeholder, and `generator_resume_yield` returns that slot. A `newframe`
/// without it compiled `FOR_ITER` to hand the iterator back as the item
/// (`int + generator` on `generator_iteration__main`) and let the generator
/// `while` start traces that die on the yield `abort_permanent`
/// (`calls_closures` `loops_aborted` 1→11).
#[inline(never)]
fn generator_resume_can_perform_call<Sym: WalkSym>(_ctx: &WalkContext<'_, '_, Sym>) -> bool {
    true
}

/// Resume a suspended generator into the trace at a `FOR_ITER`, in place of
/// the opaque `jit_next` residual.
///
/// Upstream reaches this without a route of its own: the tracer walks into
/// `send_ex` (which takes `generatorentry_driver`'s merge point only when
/// `should_not_inline`), through `_invoke_execute_frame` into
/// `execute_frame`, and the generator's OWN loop header is crossed once
/// because `should_unroll_one_iteration` (`interp_jit.py`) answers True for
/// `CO_GENERATOR`.  The walk ends at `YIELD_VALUE`, where `dispatch`'s
/// `except Yield` arm forces the virtualizable and returns the value.
///
/// That look-inside is `_opimpl_recursive_call` → `perform_call` →
/// `newframe` (`pyjitpl.py`) and ends at `except Yield` `popvalue()`.
/// `generator_resume_can_perform_call` is that `popvalue`: the yield
/// marker's entry-stack top. When it declines, the residual `jit_next`
/// stays in place — `do_residual_call`, not a start-then-abort of the
/// parent `FOR_ITER` loop and not a miscompiled item.
pub(crate) fn try_walker_specialize_generator_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }
    let Some(iter_obj) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::generator::is_generator(iter_obj) } {
        return Ok(None);
    }
    if gen_census_enabled() {
        let census = generator_resume_verdict(iter_obj);
        let (ops, merge_points, residual_calls) = census.ops_to_yield;
        eprintln!(
            "[gen-census] pc={} {} resume_py={} n_py={} tables={:?} marker={:?} \
             yield_marker={:?} loop_header={} ops={} mp={} residual={}",
            op.pc,
            census.verdict.label(),
            census.resume_py_pc as isize,
            census.n_py_instrs,
            census.tables,
            census.marker_offset,
            census.yield_marker_offset,
            census.owns_loop_header,
            ops,
            merge_points,
            residual_calls,
        );
    }
    walk_generator_resume(ctx, op, funcptr, r_args[0], iter_obj, call_descr, dst)
}

/// `'static` `DescrRefTable` for an already-leaked per-fn descr slice.
///
/// `&[DescrRef]` cannot be unsized-cast to `&dyn DescrRefTable` (the blanket
/// impl is on the slice itself); `&&[DescrRef]` can.  Memoize one fat pointer
/// per distinct pool so a resume does not leak on every `FOR_ITER`.
fn static_descr_table(refs: &'static [DescrRef]) -> &'static dyn DescrRefTable {
    thread_local! {
        static TABLES: std::cell::RefCell<Vec<(usize, *const dyn DescrRefTable)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = refs.as_ptr() as usize;
    TABLES.with(|tables| {
        let mut tables = tables.borrow_mut();
        if let Some((_, table)) = tables.iter().find(|(k, _)| *k == key) {
            // SAFETY: leaked Box<&'static [DescrRef]> lives for the process.
            return unsafe { &**table };
        }
        let table: &'static dyn DescrRefTable = Box::leak(Box::new(refs));
        tables.push((key, table));
        table
    })
}

/// `should_not_inline`: do not inline the generator body.
///
/// `opimpl_jit_merge_point` with `portal_call_depth > 0` records
/// `CALL_ASSEMBLER` (`do_recursive_call(assembler_call=True)` →
/// `direct_assembler_call`). The token is `warmstate.py`
/// `get_assembler_token`: a cell with no procedure yet gets the
/// `compile.py` `compile_tmp_callback` loop, whose body is
/// `ll_generatorentry_portal_runner_shim`. A token whose only address is
/// `generatorentry_portal_c` has no assembled entry for
/// `redirect_call_assembler` to patch, so the call would stay on that
/// function after the real trace compiled.
///
/// Op order matches `emit_walker_loop_callee_call_assembler`: vable/vref
/// bookkeeping, concrete `jit_next`, `CALL_ASSEMBLER` + `KEEPALIVE`,
/// result stamp, dst writeback, `GUARD_NOT_FORCED`, then
/// `handle_possible_exception`. The green key is
/// `genentry_resolved_cell_key`, the same key `genentry_merge_point_jit`
/// passes to `force_start_tracing`.
fn descend_generatorentry<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    funcptr: OpRef,
    iter_op: OpRef,
    iter_obj: pyre_object::PyObjectRef,
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let Some(canonical) =
        crate::jitcode_runtime::portal_jitcode_for_key(crate::genentry_state::GENENTRY_PORTAL_KEY)
    else {
        gen_resume_decline("several_yields");
        return Ok(None);
    };
    let has_merge = crate::jitcode_runtime::decoded_ops(&canonical.code)
        .any(|decoded| decoded.opname == "jit_merge_point");
    if !has_merge {
        gen_resume_decline("several_yields_no_merge");
        return Ok(None);
    }
    let pycode = unsafe { pyre_object::generator::w_generator_get_pycode(iter_obj) };
    if pycode.is_null() {
        gen_resume_decline("several_yields_no_pycode");
        return Ok(None);
    }
    // `next()` sends `w_None`. Same object the portal red `w_arg` carries.
    let w_arg = pyre_object::w_none();
    // Same cell key `genentry_merge_point_jit` resolves (`warmstate.py
    // JitCell`): `(pycode, jd index)` on `jitdrivers_sd[2].warmstate`,
    // not jd0's `make_green_key(pycode, 0, false)`.
    let green_key = {
        let (driver, _) = crate::driver::driver_pair();
        crate::genentry_state::genentry_resolved_cell_key(
            &driver.meta_interp_mut().warm_state_for_driver(2),
            pycode,
        )
    };
    let (driver, _) = crate::driver::driver_pair();
    let greenboxes = [majit_ir::Value::Ref(majit_ir::GcRef(pycode as usize))];
    let red_types = [majit_ir::Type::Ref, majit_ir::Type::Ref];
    // `jitdrivers_sd[2]` is `generatorentry` (registered immediately after jd1).
    let Some(token) = driver
        .meta_interp_mut()
        .get_or_make_jitdriver_assembler_token_arc(2, green_key, &greenboxes, &red_types)
    else {
        gen_resume_decline("several_yields_no_token");
        return Ok(None);
    };
    let none_op = ctx.trace_ctx.const_ref(w_arg as i64);
    ctx.trace_ctx.set_opref_concrete(
        none_op,
        majit_ir::Value::Ref(majit_ir::GcRef(w_arg as usize)),
    );

    // `space.next` dispatches on the iterator's type, and `send_ex` reads
    // the immutable `self.pycode` that `generatorentry_driver.jit_merge_point`
    // takes as its green: `promote_greens` guards it to the constant the
    // token was chosen for.
    if !iter_op.is_constant() {
        let type_const = ctx
            .trace_ctx
            .const_int(unsafe { (*iter_obj).ob_type } as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[iter_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        let pycode_descr = crate::descr::make_immutable_field_descr(
            std::mem::offset_of!(pyre_object::generator::GeneratorIterator, pycode),
            std::mem::size_of::<pyre_object::PyObjectRef>(),
            majit_ir::Type::Ref,
            false,
        );
        let pycode_op =
            ctx.trace_ctx
                .record_op_with_descr(OpCode::GetfieldGcR, &[iter_op], pycode_descr);
        ctx.trace_ctx.set_opref_concrete(
            pycode_op,
            majit_ir::Value::Ref(majit_ir::GcRef(pycode as usize)),
        );
        let pycode_const = ctx.trace_ctx.const_ref(pycode as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[pycode_op, pycode_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }

    // `pyjitpl.py` `vable_and_vrefs_before_residual_call`, before the call.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
    // `direct_assembler_call` executes the call, then rewrites the recorded
    // op to `CALL_ASSEMBLER`. `jit_next` is that execution for `next(gen)`,
    // run through the residual executor so the walk's virtualizable heap
    // pointer is saved and restored around the callee's own portal
    // activation and `tracing_after_residual_call` runs after it (the same
    // route `emit_walker_loop_callee_call_assembler` takes). A raw call
    // left the callee's frame seeded as this walk's vable, and the next
    // snapshot wrote that frame's slot count into the caller's array.
    let exec = try_execute_residual_call_via_executor(
        ctx,
        OpCode::CallMayForceR,
        &[funcptr, iter_op],
        call_descr,
        OpRef::NONE,
        op.pc,
        None,
        false,
    )?;
    // `execute_residual_call` consumes `BH_LAST_EXC_VALUE` and returns the
    // exception word. Carry that word; do not read the TLS slot afterwards.
    // `pyjitpl.py handle_possible_exception` / `emit_walker_loop_callee_call_assembler`.
    let (concrete, raised) = match exec {
        ResidualExecOutcome::Executed(Ok(result)) => (result, 0),
        ResidualExecOutcome::Executed(Err(exc)) => (0, exc),
        ResidualExecOutcome::Declined(cause) => {
            fbw_mark_unjournaled_effect(cause);
            gen_resume_decline("several_yields_exec_declined");
            return Ok(None);
        }
    };
    let ca_result = ctx.trace_ctx.call_assembler_red_only_ref_arc(
        token,
        &[iter_op, none_op],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
    );
    // `execute_and_record` (`pyjitpl.py`) stamps the executed result on the
    // box before `opimpl_goto_if_not` reads `box.getint()`. `jit_next`
    // answers exhaustion with a null return; that is still a concrete Ref,
    // and leaving the CALL_ASSEMBLER unstamped makes the trailing
    // `ptr_nonzero` / `goto_if_not` a missing-value abort.
    if raised == 0 {
        ctx.trace_ctx.set_opref_concrete(
            ca_result,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
        );
    }
    // `pyjitpl.py do_residual_call` step 5: invalidate on the
    // CALL_MAY_FORCE executed above, not the recorded CALL_ASSEMBLER.
    ctx.trace_ctx.heapcache_invalidate_caches_varargs(
        OpCode::CallMayForceR,
        Some(call_descr.get_extra_info()),
        &[funcptr, iter_op],
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', ca_result)?;
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    // `direct_assembler_call` keeps the assembler target alive between
    // `GUARD_NOT_FORCED` and `handle_possible_exception`.
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[iter_op]);
    if raised != 0 {
        // A returning generator raises `StopIteration` out of `send_ex`.
        // The executor already returned that exception word.
        let exc = ctx.trace_ctx.const_ref(raised);
        let exc_concrete = ConcreteValue::Ref(raised as pyre_object::PyObjectRef);
        ctx.set_last_exc_value(exc, exc_concrete);
        walker_record_guard_exception(ctx, op.pc)?;
        return Ok(Some((
            DispatchOutcome::SubRaise { exc, exc_concrete },
            op.next_pc,
        )));
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Walk the already-installed portal-shaped body of a suspended generator
/// from `resume_execute_frame` to its next `yield`.
fn walk_generator_resume<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    funcptr: OpRef,
    iter_op: OpRef,
    iter_obj: pyre_object::PyObjectRef,
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let Some(shape) =
        (unsafe { pyre_interpreter::baseobjspace::generator_resume_fast_path(iter_obj) })
    else {
        gen_resume_decline("not_resumable");
        return Ok(None);
    };
    let raw = unsafe { pyre_interpreter::w_code_get_ptr(shape.w_pycode) }
        as *const pyre_interpreter::CodeObject;
    if raw.is_null() {
        gen_resume_decline("no_code");
        return Ok(None);
    }
    let code = unsafe { &*raw };
    // `should_unroll_one_iteration` (`interp_jit.py`) is true for every
    // `CO_GENERATOR` body; `should_not_inline` is the 2+ yield case that
    // upstream sends to `generatorentry_driver` instead.
    if pyre_interpreter::baseobjspace::should_not_inline(code) {
        return descend_generatorentry(ctx, op, funcptr, iter_op, iter_obj, call_descr, dst);
    }
    if !code.cellvars.is_empty() || !code.freevars.is_empty() {
        gen_resume_decline("cells_or_freevars");
        return Ok(None);
    }
    if !generator_table_is_only_the_stopiteration_wrapper(code) {
        gen_resume_decline("real_exception_table");
        return Ok(None);
    }
    // `_opimpl_recursive_call` (`pyjitpl.py`): `can_inline_callable` then
    // `perform_call` → `newframe` → `framestack.append`, and the walk
    // ends at `dispatch`'s `except Yield` `popvalue()`.  `newframe`
    // without that `popvalue` compiled the iterator as the `FOR_ITER`
    // item.  `sub_jitcode_body_for_code` also *installs* a portal-shaped
    // body; doing that without a successful `perform_call` lets the
    // generator's own `while` start traces that die on the yield
    // `abort_permanent` (SNAPDIFF `loops_aborted` 1→11).  Residual like
    // `do_residual_call`, and do not install.
    if !generator_resume_can_perform_call(ctx) {
        gen_resume_decline("cannot_capture_resumedata");
        return Ok(None);
    }
    let w_pycode = shape.w_pycode as *const ();
    // Same on-demand install as a user-call inline: the body is the thing
    // we are about to walk.  Census-only must not take this path; the walk
    // itself is the reason to install.
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_pycode) else {
        gen_resume_decline("no_jitcode");
        return Ok(None);
    };
    let Some(pjc) = crate::state::pyjitcode_for_code(w_pycode) else {
        gen_resume_decline("no_pjc");
        return Ok(None);
    };
    if !pjc.metadata.built_as_portal {
        gen_resume_decline("not_portal_shaped");
        return Ok(None);
    }
    let resume_py_pc = shape.last_instr as usize + 1;
    let Some(resume_marker) = generator_resume_marker_offset(&pjc, resume_py_pc) else {
        gen_resume_decline("no_resume_marker");
        return Ok(None);
    };
    let Some(yield_marker) = pjc
        .metadata
        .abort_permanent_py_pc_by_jit_pc
        .iter()
        .find(|&&(_, py)| py as usize == shape.last_instr as usize)
        .map(|&(off, _)| off as usize)
    else {
        gen_resume_decline("no_yield_marker");
        return Ok(None);
    };
    if pjc.metadata.portal_frame_reg == u16::MAX {
        gen_resume_decline("no_portal_frame_reg");
        return Ok(None);
    }
    let Some((callee_descr_refs, callee_perfn_descrs, callee_lookup)) =
        crate::state::sub_jitcode_descr_pool_for_code(w_pycode)
    else {
        gen_resume_decline("no_descr_pool");
        return Ok(None);
    };

    let body_coord = fbw_foriter_body_from_op_pc(ctx, op.pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body_coord);
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();
    let executed_effects_before = fbw_executed_effect_count();

    let _roots = pyre_object::gc_roots::push_roots();
    let iter_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(iter_obj);
    let frame_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(shape.frame as pyre_object::PyObjectRef);
    let code_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(shape.w_pycode);
    let iter_obj = pyre_object::gc_roots::shadow_stack_get(iter_root);
    let gen_frame =
        pyre_object::gc_roots::shadow_stack_get(frame_root) as *mut pyre_interpreter::PyFrame;
    if gen_frame.is_null() {
        gen_resume_decline("null_frame");
        return Ok(None);
    }

    if !iter_op.is_constant() {
        let type_const = ctx
            .trace_ctx
            .const_int(unsafe { (*iter_obj).ob_type } as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[iter_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        let gen_const = ctx.trace_ctx.const_ref(iter_obj as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[iter_op, gen_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }

    // `resume_execute_frame`: push the sent `None` and start at
    // `last_instr + 1`.  Arm the durable undo first so a declined walk
    // puts the frame back whole.
    fbw_arm_durable_frame_undo(gen_frame as usize);
    unsafe {
        // `pushvalue_none` (`pyframe.py`) counts a slot that `popvalue` already
        // set to None. A traced yield publishes the depth and leaves the
        // yielded int in that slot; clear it before the count.
        let depth = (*gen_frame).valuestackdepth;
        if !(*(*gen_frame).locals_cells_stack_w).as_slice()[depth].is_null() {
            (*gen_frame).set_locals_w(depth, std::ptr::null_mut());
        }
        (*gen_frame).pushvalue_none();
    };

    let saved_fbw_mode = ctx.fbw_mode;
    ctx.fbw_mode.inline_subwalk = true;

    // Load `gen.frame` from the guarded iterator so the compiled loop
    // re-reads the suspended frame each iteration instead of baking the
    // address from this recording.
    let frame_offset = std::mem::offset_of!(pyre_object::generator::GeneratorIterator, frame_ptr);
    let frame_descr = crate::descr::make_field_descr(
        frame_offset,
        std::mem::size_of::<*mut u8>(),
        majit_ir::Type::Ref,
        false,
    );
    let frame_box =
        ctx.trace_ctx
            .record_op_with_descr(OpCode::GetfieldGcR, &[iter_op], frame_descr);
    ctx.trace_ctx.set_opref_concrete(
        frame_box,
        majit_ir::Value::Ref(majit_ir::GcRef(gen_frame as usize)),
    );
    // The slot `pushvalue_none` just counted may still hold the object the
    // previous traced yield left behind. Recording already nulled it; the
    // compiled resume has to as well, before anything reads the frame back.
    let resumed_depth = unsafe { (*gen_frame).valuestackdepth.saturating_sub(1) };
    publish_generator_slot_null(ctx, gen_frame as usize, frame_box, resumed_depth);
    ctx.fbw_mode.generator_resume = Some(GeneratorResumeSubwalk {
        frame: gen_frame as usize,
        frame_box,
        yield_marker_jit_pc: yield_marker,
        yield_py_pc: shape.last_instr as u32,
    });
    let ec_box = ctx
        .trace_ctx
        .const_ref(pyre_interpreter::call::getexecutioncontext() as i64);
    let mut extra_ref_seeds: Vec<(usize, OpRef, ConcreteValue)> = Vec::new();
    let portal_frame_reg = pjc.metadata.portal_frame_reg;
    let portal_ec_reg = pjc.metadata.portal_ec_reg;
    if portal_frame_reg != u16::MAX {
        extra_ref_seeds.push((
            portal_frame_reg as usize,
            frame_box,
            ConcreteValue::Ref(gen_frame as pyre_object::PyObjectRef),
        ));
    }
    if portal_ec_reg != u16::MAX {
        extra_ref_seeds.push((
            portal_ec_reg as usize,
            ec_box,
            ConcreteValue::Ref(
                pyre_interpreter::call::getexecutioncontext() as pyre_object::PyObjectRef
            ),
        ));
    }
    if let Some(pcdep) =
        crate::state::pcdep_trivia_at(pjc.jitcode.index() as i32, resume_marker as i32)
    {
        let frame_ref = unsafe { &*gen_frame };
        for &(bank, color, slot) in &pcdep {
            if bank != 1 {
                continue;
            }
            if let Some(&value) = locals_w!(frame_ref).as_slice().get(slot as usize)
                && !value.is_null()
            {
                let opref = ctx.trace_ctx.const_ref(value as i64);
                extra_ref_seeds.push((color as usize, opref, ConcreteValue::Ref(value)));
            }
        }
    }

    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    ctx.descr_refs = static_descr_table(callee_descr_refs);
    ctx.raw_descrs = RawDescrPool::PerFn(callee_perfn_descrs);
    ctx.sub_jitcode_lookup = callee_lookup;

    let _inline_concrete = InlineConcreteFrameGuard::enter(gen_frame);
    let walk_result = run_sub_jitcode_walk_from(
        ctx,
        op.pc,
        resume_marker,
        &body,
        &[],
        &[],
        &[],
        &[],
        &[],
        &extra_ref_seeds,
    );

    ctx.fbw_mode = saved_fbw_mode;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;

    let walk_result = match walk_result {
        Ok(outcome) => outcome,
        Err(error @ DispatchError::TraceTooLong { .. }) => return Err(error),
        Err(error) => {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[gen-resume] walk err pc={} resume_marker={} yield_marker={} err={error:?}",
                    op.pc, resume_marker, yield_marker,
                );
            }
            fbw_durable_frame_rollback_one(gen_frame as usize);
            if fbw_executed_effect_count() != executed_effects_before {
                return Err(error);
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    };
    let DispatchOutcome::SubReturn { result: Some(item) } = walk_result else {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[gen-resume] walk outcome pc={} resume_marker={} yield_marker={} outcome={walk_result:?}",
                op.pc, resume_marker, yield_marker,
            );
        }
        fbw_durable_frame_rollback_one(gen_frame as usize);
        if fbw_executed_effect_count() != executed_effects_before {
            return Err(DispatchError::callee_inline_unsupported(op.pc));
        }
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    };
    let Some(concrete_item) = walker_concrete_ref_object(ctx, item) else {
        fbw_durable_frame_rollback_one(gen_frame as usize);
        if fbw_executed_effect_count() != executed_effects_before {
            return Err(DispatchError::callee_inline_unsupported(op.pc));
        }
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    };
    write_ref_reg(ctx, op.pc, dst, item, ConcreteValue::Ref(concrete_item))?;
    fbw_foriter_inflight_capture(concrete_item, body_coord, false);
    ctx.frame_state.borrow_mut().vstack_last_ref = item;
    let _ = code_root;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Forward dunder selected by `try_dispatch_binary_special` for a non-inplace
/// BINARY_OP. In-place operators have a distinct `__i*__` then binary fallback
/// protocol and therefore stay on the generic path until that protocol is
/// ported as a unit.
pub(super) fn user_binop_forward_dunder(
    op: pyre_interpreter::bytecode::BinaryOperator,
) -> Option<&'static str> {
    use pyre_interpreter::bytecode::BinaryOperator;

    match op {
        BinaryOperator::Add => Some("__add__"),
        BinaryOperator::And => Some("__and__"),
        BinaryOperator::FloorDivide => Some("__floordiv__"),
        BinaryOperator::Lshift => Some("__lshift__"),
        BinaryOperator::MatrixMultiply => Some("__matmul__"),
        BinaryOperator::Multiply => Some("__mul__"),
        BinaryOperator::Or => Some("__or__"),
        BinaryOperator::Power => Some("__pow__"),
        BinaryOperator::Remainder => Some("__mod__"),
        BinaryOperator::Rshift => Some("__rshift__"),
        BinaryOperator::Subtract => Some("__sub__"),
        BinaryOperator::TrueDivide => Some("__truediv__"),
        BinaryOperator::Xor => Some("__xor__"),
        BinaryOperator::Subscr
        | BinaryOperator::InplaceAdd
        | BinaryOperator::InplaceAnd
        | BinaryOperator::InplaceFloorDivide
        | BinaryOperator::InplaceLshift
        | BinaryOperator::InplaceMatrixMultiply
        | BinaryOperator::InplaceMultiply
        | BinaryOperator::InplaceOr
        | BinaryOperator::InplacePower
        | BinaryOperator::InplaceRemainder
        | BinaryOperator::InplaceRshift
        | BinaryOperator::InplaceSubtract
        | BinaryOperator::InplaceTrueDivide
        | BinaryOperator::InplaceXor => None,
    }
}

/// Reflected dunder selected by `try_dispatch_binary_special` for a
/// non-inplace BINARY_OP. In-place operators and `Subscr` have no `__r*__`
/// arm in `_call_binop_impl` (`descroperation.py`) and stay on the generic
/// path, matching [`user_binop_forward_dunder`].
pub(super) fn user_binop_reflected_dunder(
    op: pyre_interpreter::bytecode::BinaryOperator,
) -> Option<&'static str> {
    use pyre_interpreter::bytecode::BinaryOperator;

    match op {
        BinaryOperator::Add => Some("__radd__"),
        BinaryOperator::And => Some("__rand__"),
        BinaryOperator::FloorDivide => Some("__rfloordiv__"),
        BinaryOperator::Lshift => Some("__rlshift__"),
        BinaryOperator::MatrixMultiply => Some("__rmatmul__"),
        BinaryOperator::Multiply => Some("__rmul__"),
        BinaryOperator::Or => Some("__ror__"),
        BinaryOperator::Power => Some("__rpow__"),
        BinaryOperator::Remainder => Some("__rmod__"),
        BinaryOperator::Rshift => Some("__rrshift__"),
        BinaryOperator::Subtract => Some("__rsub__"),
        BinaryOperator::TrueDivide => Some("__rtruediv__"),
        BinaryOperator::Xor => Some("__rxor__"),
        BinaryOperator::Subscr
        | BinaryOperator::InplaceAdd
        | BinaryOperator::InplaceAnd
        | BinaryOperator::InplaceFloorDivide
        | BinaryOperator::InplaceLshift
        | BinaryOperator::InplaceMatrixMultiply
        | BinaryOperator::InplaceMultiply
        | BinaryOperator::InplaceOr
        | BinaryOperator::InplacePower
        | BinaryOperator::InplaceRemainder
        | BinaryOperator::InplaceRshift
        | BinaryOperator::InplaceSubtract
        | BinaryOperator::InplaceTrueDivide
        | BinaryOperator::InplaceXor => None,
    }
}

/// True when `method` is an app-level `function` — the public type
/// [`resolve_inlinable_callee`] admits. Builtin slots share `is_function`
/// but carry a different type object.
fn binop_impl_is_app_level_function(method: pyre_object::PyObjectRef) -> bool {
    !method.is_null()
        && unsafe {
            (*method).ob_type as *const () as usize
                == &pyre_interpreter::FUNCTION_TYPE as *const _ as usize
        }
}

/// True when `method` is a builtin operator slot (`builtin_function_or_method`,
/// `method_descriptor`, or `wrapper_descriptor`). Those are the forward impls
/// `_invoke_binop` (`descroperation.py`) may run without entering user Python.
fn binop_impl_is_builtin_slot(method: pyre_object::PyObjectRef) -> bool {
    if method.is_null() {
        return false;
    }
    unsafe {
        (pyre_interpreter::is_function(method) && !binop_impl_is_app_level_function(method))
            || pyre_interpreter::is_slot_wrapper(method)
    }
}

/// True when `_call_binop_impl` (`descroperation.py`) tries the rhs reflected
/// impl before the lhs forward impl.
///
/// The caller has already established that the rhs type is a proper subtype of
/// the lhs type. The swap still requires the two methods to be found on
/// different defining classes, and `abstract_issubclass_w` (`abstractinst.py`,
/// `allow_override` false) cancels it when the forward impl's class or the lhs
/// type is a subclass of the reflected impl's class. For two type objects that
/// test is `issubtype_w` (`p_recursive_issubclass_w`).
fn binop_reflected_runs_first(
    op_kind: pyre_interpreter::bytecode::BinaryOperator,
    w_typ_l: pyre_object::PyObjectRef,
    w_typ_r: pyre_object::PyObjectRef,
) -> bool {
    let Some(forward) = user_binop_forward_dunder(op_kind) else {
        return false;
    };
    let Some(reflected) = user_binop_reflected_dunder(op_kind) else {
        return false;
    };
    let Some((lsrc, _)) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_where_pair(w_typ_l, forward) })
    else {
        return false;
    };
    let Some((rsrc, _)) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_where_pair(w_typ_r, reflected) })
    else {
        return false;
    };
    if lsrc.is_null() || rsrc.is_null() || std::ptr::eq(lsrc, rsrc) {
        return false;
    }
    let keep_order = unsafe {
        pyre_object::typeobject::w_type_issubtype(lsrc, rsrc)
            || pyre_object::typeobject::w_type_issubtype(w_typ_l, rsrc)
    };
    !keep_order
}

/// Inline a plain Python arithmetic dunder after the exact numeric BINARY_OP
/// specializations decline. The forward arm looks the dunder up on the lhs
/// class, matching `try_dispatch_binary_special`'s first `_invoke_binop`
/// (`descroperation.py` `_call_binop_impl`). When that impl is absent or a
/// builtin slot, the reflected arm looks `__r*__` up on the rhs and descends
/// with the operands swapped. A proper-subclass rhs whose reflected impl
/// `_call_binop_impl` would run first takes that arm without sampling the
/// forward slot; a traced `NotImplemented` result guards and deopts to the
/// generic dispatcher.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_user_binop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    op_tag: i64,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 2 {
        return Ok(None);
    }

    // Name every bail, as the CallFn inline path does: a BINARY_OP that stays
    // residual costs a full interpreter dispatch per execution, and without a
    // reason line the only observable is the runtime.
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[binop-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    let Some(op_kind) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        decline!(format_args!("unknown binary op tag {op_tag}"));
    };
    let Some(dunder) = user_binop_forward_dunder(op_kind) else {
        decline!(format_args!("no forward dunder for {op_kind:?}"));
    };

    let lhs = r_args[0];
    let rhs = r_args[1];
    let Some(concrete_lhs) = walker_concrete_ref_object(ctx, lhs) else {
        decline!("lhs has no concrete ref");
    };
    let Some(concrete_rhs) = walker_concrete_ref_object(ctx, rhs) else {
        decline!("rhs has no concrete ref");
    };

    // A tagged immediate is an exact builtin `int` with C-level operator slots:
    // it has no heap `ob_type`/`w_class` to pin, and its dunder is not inlinable
    // Python code.  Decline before the concrete derefs below, which would fault
    // on the immediate (`typedef::r#type` stays the tagged-safe typing path).
    // Inert behind `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(concrete_lhs)
            || pyre_object::tagged_int::is_tagged_int(concrete_rhs))
    {
        decline!("tagged immediate operand");
    }

    let w_class = unsafe { (*concrete_lhs).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        decline!("lhs w_class is not a type");
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        decline!(format_args!("lhs class {} has no version tag", unsafe {
            pyre_object::typeobject::w_type_get_name(w_class)
        }));
    }

    let Some(w_typ_r) = pyre_interpreter::typedef::r#type(concrete_rhs) else {
        decline!("rhs has no type");
    };
    // `_call_binop_impl` runs the reflected impl first when the rhs type is a
    // proper subtype and the defining-class test does not keep the original
    // order. The forward slot must not be sampled here: `float.__add__`
    // returns a value for a float subclass, so a NotImplemented probe would
    // refuse the reflected Python method the language actually calls.
    if !std::ptr::eq(w_class, w_typ_r.as_ptr())
        && unsafe { pyre_object::typeobject::w_type_issubtype(w_typ_r.as_ptr(), w_class) }
        && binop_reflected_runs_first(op_kind, w_class, w_typ_r.as_ptr())
    {
        return try_walker_inline_user_binop_reflected(
            ctx,
            op,
            code,
            op_kind,
            dunder,
            None,
            r_args,
            call_descr,
            dst,
            lhs,
            concrete_lhs,
            w_class,
            rhs,
            concrete_rhs,
            w_typ_r.as_ptr(),
        );
    }

    let forward_method = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) };
    if let Some(method) = forward_method
        && let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) })
    {
        return try_walker_inline_user_binop_dunder(
            ctx,
            op,
            code,
            r_args,
            call_descr,
            dst,
            dunder,
            method,
            w_code,
            nparams,
            has_closure,
            lhs,
            concrete_lhs,
            w_class,
            version_tag,
            rhs,
            concrete_rhs,
            w_typ_r.as_ptr(),
        );
    }
    // Forward lookup found nothing, or found a slot that is not inlinable
    // Python.  An app-level `function` that `resolve_inlinable_callee`
    // refused is still user Python — do not invoke it, and do not fall
    // through to `__r*__`.  A builtin slot (or a missing impl) is the
    // `_invoke_binop` miss that `_call_binop_impl` (`descroperation.py`)
    // follows with the reflected arm.
    if let Some(method) = forward_method {
        if binop_impl_is_app_level_function(method) {
            decline!(format_args!(
                "{}.{dunder} is not inlinable Python code",
                unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
            ));
        }
        if !binop_impl_is_builtin_slot(method) {
            decline!(format_args!("{}.{dunder} is not a builtin slot", unsafe {
                pyre_object::typeobject::w_type_get_name(w_class)
            }));
        }
    }
    return try_walker_inline_user_binop_reflected(
        ctx,
        op,
        code,
        op_kind,
        dunder,
        forward_method,
        r_args,
        call_descr,
        dst,
        lhs,
        concrete_lhs,
        w_class,
        rhs,
        concrete_rhs,
        w_typ_r.as_ptr(),
    );
}

/// Reflected arm of [`try_walker_inline_user_binop`]: look `__r*__` up on the
/// rhs type and descend with the operands swapped. Taken when the forward
/// impl cannot run user Python, and when [`binop_reflected_runs_first`] says
/// the reflected impl runs first — that caller passes no forward method, so
/// the slot probe below does not sample an impl the language does not call.
#[allow(clippy::too_many_arguments)]
fn try_walker_inline_user_binop_reflected<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    op_kind: pyre_interpreter::bytecode::BinaryOperator,
    forward_dunder: &'static str,
    forward_method: Option<pyre_object::PyObjectRef>,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    lhs: OpRef,
    mut concrete_lhs: pyre_object::PyObjectRef,
    mut w_class_l: pyre_object::PyObjectRef,
    rhs: OpRef,
    mut concrete_rhs: pyre_object::PyObjectRef,
    mut w_typ_r: pyre_object::PyObjectRef,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[binop-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    let Some(dunder) = user_binop_reflected_dunder(op_kind) else {
        decline!(format_args!("no reflected dunder for {op_kind:?}"));
    };
    // `_call_binop_impl` (`descroperation.py`) never considers the reflected
    // impl when the operand types are the same.
    if std::ptr::eq(w_class_l, w_typ_r) {
        decline!("same types; reflected dunder is not considered");
    }
    let version_tag_r = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_typ_r) };
    if version_tag_r == 0 {
        decline!(format_args!("rhs class {} has no version tag", unsafe {
            pyre_object::typeobject::w_type_get_name(w_typ_r)
        }));
    }
    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_typ_r, dunder) })
    else {
        decline!(format_args!("{} has no {dunder}", unsafe {
            pyre_object::typeobject::w_type_get_name(w_typ_r)
        }));
    };
    let Some((mut w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) })
    else {
        decline!(format_args!(
            "{}.{dunder} is not inlinable Python code",
            unsafe { pyre_object::typeobject::w_type_get_name(w_typ_r) }
        ));
    };
    let mut method = method;
    // `_invoke_binop` (`descroperation.py` `_call_binop_impl`) treats a
    // missing impl as no result.  A builtin slot is invoked at record
    // time: NotImplemented is no result, anything else (a value or a
    // raise) is a result we cannot skip.  The compiled body will not
    // re-run that slot, so the skip has to hold for every pair the
    // emitted guards admit.  Those guards pin the rhs class/version tag
    // (the reflected receiver) and the lhs `w_class`; builtin numeric
    // slots decide NotImplemented from those types.  The odometer
    // reading around the invocation — not an assumption — is what
    // establishes that the skipped slot applied nothing.  A Python
    // forward dunder is never invoked here.
    if let Some(fwd) = forward_method {
        let effects_before = fbw_executed_effect_count();
        let unjournaled_before = fbw_has_unjournaled_effect();
        // The slot runs arbitrary builtin code, so it can allocate and a minor
        // collection then moves every operand the route still needs after it.
        // Pin them for the invocation and read them back from the shadow
        // stack, as the `for_iter` and resolved-call descents do.
        let _roots = pyre_object::gc_roots::push_roots();
        let lhs_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(concrete_lhs);
        let lhs_class_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_class_l);
        let rhs_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(concrete_rhs);
        let rhs_type_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_typ_r);
        let method_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(method);
        let code_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_code as pyre_object::PyObjectRef);

        let forward_result = unsafe {
            pyre_interpreter::baseobjspace::get_and_call_function(
                fwd,
                concrete_lhs,
                w_class_l,
                &[concrete_rhs],
            )
        };

        concrete_lhs = pyre_object::gc_roots::shadow_stack_get(lhs_root);
        w_class_l = pyre_object::gc_roots::shadow_stack_get(lhs_class_root);
        concrete_rhs = pyre_object::gc_roots::shadow_stack_get(rhs_root);
        w_typ_r = pyre_object::gc_roots::shadow_stack_get(rhs_type_root);
        method = pyre_object::gc_roots::shadow_stack_get(method_root);
        w_code = pyre_object::gc_roots::shadow_stack_get(code_root) as *const ();

        match forward_result {
            Ok(result)
                if pyre_interpreter::baseobjspace::is_w(
                    result,
                    pyre_object::special::w_not_implemented(),
                ) => {}
            Ok(_) => {
                decline!(format_args!(
                    "{}.{forward_dunder} produced a value for these operand types",
                    unsafe { pyre_object::typeobject::w_type_get_name(w_class_l) }
                ));
            }
            Err(_) => {
                decline!(format_args!(
                    "{}.{forward_dunder} raised for these operand types",
                    unsafe { pyre_object::typeobject::w_type_get_name(w_class_l) }
                ));
            }
        }
        if fbw_executed_effect_count() != effects_before
            || unjournaled_before
            || fbw_has_unjournaled_effect()
        {
            decline!(format_args!(
                "{}.{forward_dunder} applied an effect before answering NotImplemented",
                unsafe { pyre_object::typeobject::w_type_get_name(w_class_l) }
            ));
        }
        // The skip of the forward slot stands on the lhs type still
        // resolving that name to the same builtin slot.  Only the
        // version tag witnesses that; a zero tag is uncacheable and
        // cannot be pinned.
        if unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class_l) } == 0 {
            decline!(format_args!("lhs class {} has no version tag", unsafe {
                pyre_object::typeobject::w_type_get_name(w_class_l)
            }));
        }
        let w_class_l_const = ctx.trace_ctx.const_ref(w_class_l as i64);
        walker_pin_type_version_tag(ctx, op.pc, w_class_l_const)?;
    }
    try_walker_inline_user_binop_dunder(
        ctx,
        op,
        code,
        r_args,
        call_descr,
        dst,
        dunder,
        method,
        w_code,
        nparams,
        has_closure,
        rhs,
        concrete_rhs,
        w_typ_r,
        version_tag_r,
        lhs,
        concrete_lhs,
        w_class_l,
    )
}

/// Shared descent for a resolved Python binop dunder. `receiver` is `self`;
/// `other` is the remaining operand. The class/version-tag guard pins the
/// receiver type; the other-operand guard pins the remaining operand's type.
#[allow(clippy::too_many_arguments)]
fn try_walker_inline_user_binop_dunder<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dunder: &'static str,
    method: pyre_object::PyObjectRef,
    w_code: *const (),
    nparams: usize,
    has_closure: bool,
    receiver: OpRef,
    concrete_receiver: pyre_object::PyObjectRef,
    w_class: pyre_object::PyObjectRef,
    version_tag: u64,
    other: OpRef,
    concrete_other: pyre_object::PyObjectRef,
    w_typ_other: pyre_object::PyObjectRef,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[binop-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    // The two operands bind the first two parameters.  A longer signature is
    // not itself unbindable: `Function.funccall_valuestack` fills every
    // parameter the call leaves unbound from `defs_w`, and the
    // resolved descent below already seeds that tail from `__defaults__` --
    // it declines on its own when the defaults do not cover the whole tail.
    // `_pydecimal`'s dunders are all `(self, other, context=None)`, so a hard
    // `!= 2` refuses every arithmetic operator that module defines.
    if nparams < 2 || (nparams != 2 && !binop_defaulted_params_enabled()) {
        decline!(format_args!("{}.{dunder} takes {nparams} params", unsafe {
            pyre_object::typeobject::w_type_get_name(w_class)
        }));
    }
    // A `NotImplemented` result has to go back to the operator protocol.
    // `perform_call` (`pyjitpl.py`) traces the dunder as an MIFrame; the
    // walker admits the same way. [`BinopRewindInlineGuard`] refuses the
    // first commit before the helper runs, so the cut is empty of effects
    // when the result is handed back.
    if !dunder_body_admissible_on_rewind(w_code) {
        decline!(format_args!(
            "{}.{dunder} delegates through a nested call",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
        ConcreteValue::Ref(concrete_other),
    ];
    // Rewind point for the `NotImplemented` arm below.  Nothing above this
    // line records IR or touches the heap cache.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // `MetaInterp.virtualref_boxes` is state beside the history, not part of
    // `History.cut`.  This speculative descent can run
    // `opimpl_virtual_ref` before returning `NotImplemented`, so its rewind
    // must restore the pair stack together with the recorded operations.
    let pre_fold_virtualrefs = ctx.trace_ctx.snapshot_virtualref_boxes();
    let effects_before = fbw_executed_effect_count();
    let unjournaled_before = fbw_has_unjournaled_effect();
    // A decline below returns to the residual, which re-executes the dunder.
    // Journal entries pushed after this mark are undone there
    // (`fbw_effect_journal_rollback_since`); the walk can still commit, so
    // `fbw_store_journal_rollback` does not.
    let journal_mark = fbw_effect_journal_mark();
    let method_const = ctx.trace_ctx.const_ref(method as i64);
    // Copied out before the descent takes `ctx` mutably; the region state is
    // the session's, not this frame's.
    // Value-producing boxes are numbered by opencoder `_index` (`OpRef::RefOp`
    // `raw()`); `_count` also counts voids, so a store into a region-fresh
    // allocation would fail `obj.raw() >= from` and residualize the dunder.
    let region_from = ctx.trace_ctx.get_trace_position()._index;
    let session = ctx.session;
    let rewind_guard = BinopRewindInlineGuard::enter(session, region_from);
    let descent = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![receiver, other],
        vec![
            ConcreteValue::Ref(concrete_receiver),
            ConcreteValue::Ref(concrete_other),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((receiver, concrete_receiver, w_class, version_tag, unsafe {
            inline_attr_cell_guard(w_class, dunder, method)
        })),
        Some((other, concrete_other, w_typ_other)),
        // `entry_is_call_boundary`.  What decides it is whether the abort
        // can name this entry as a resume coordinate, not whether the entry
        // is spelled CALL.  `binop_rewind_enabled` is that name.
        binop_rewind_enabled(),
        false,
        None,
    );
    // The refusal that sets this happens BEFORE the commit runs, so the region
    // has applied nothing and the cut is the legal exit.  Asking here rather
    // than at the `NotImplemented` arm below is the whole point: by the time a
    // result exists, the write that had to be prevented has happened.
    //
    // The cut takes the snapshots with it.  The descent attaches guards before
    // it declines, and a snapshot minted inside it names the discarded
    // operation namespace — including the `virtual_ref` pair the callee's
    // `enter` pushed.  `cut_trace` leaves the side table alone, so those
    // entries survive into the optimizer, which remaps every published
    // snapshot and resolves the stale `OpRef` against a position the cut has
    // since handed to another operation.
    let descent = match descent {
        Err(error @ DispatchError::TraceTooLong { .. }) => return Err(error),
        other => other,
    };
    let refused_a_commit = rewind_guard.refused();
    drop(rewind_guard);
    if refused_a_commit {
        fbw_effect_journal_rollback_since(journal_mark);
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx
            .restore_virtualref_boxes(pre_fold_virtualrefs.clone());
        ctx.trace_ctx.heap_cache_mut().reset();
        decline!(format_args!(
            "{}.{dunder} may commit before its result is known",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    }
    let Some(inlined) = (match descent {
        Ok(inlined) => inlined,
        Err(err) => {
            fbw_effect_journal_rollback_since(journal_mark);
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.restore_virtualref_boxes(pre_fold_virtualrefs);
            ctx.trace_ctx.heap_cache_mut().reset();
            decline!(format_args!(
                "callee inline of {}.{dunder} aborted: {err:?}",
                unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
            ));
        }
    }) else {
        fbw_effect_journal_rollback_since(journal_mark);
        decline!(format_args!(
            "callee inline of {}.{dunder} declined",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r.get(dst).expect("ref register in range");
        if matches!(
            concrete_from_recorded_opref(ctx, result),
            ConcreteValue::Ref(obj)
                if std::ptr::eq(obj, pyre_object::special::w_not_implemented())
        ) {
            if fbw_inline_diag_enabled() {
                eprintln!(
                    "[binop-inline-abort] pc={} why={}.{dunder} returned NotImplemented",
                    op.pc,
                    unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
                );
            }
            // Hand the result back to the full binary protocol.  Cutting the
            // recorded descent away and letting the ordinary residual run
            // applies the dunder exactly once -- but only while this walk has
            // applied nothing since `pre_fold_pos`, because the residual
            // re-executes the body and there is no journal undo for what user
            // code did.  The same all-clear reading the un-lowered-helper
            // rollback takes, and for its reason too: a walk already carrying
            // an unjournaled effect can no longer flush a `CloseLoop` end.
            //
            // The snapshots go with the operations, for the reason the refusal
            // arm above records.
            if binop_rewind_enabled()
                && fbw_executed_effect_count() == effects_before
                && !unjournaled_before
                && !fbw_has_unjournaled_effect()
            {
                fbw_effect_journal_rollback_since(journal_mark);
                ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
                ctx.trace_ctx.restore_virtualref_boxes(pre_fold_virtualrefs);
                ctx.trace_ctx.heap_cache_mut().reset();
                return Ok(None);
            }
            return Err(DispatchError::callee_inline_unsupported(op.pc));
        }
        if !result.is_constant() {
            let not_implemented = ctx
                .trace_ctx
                .const_ref(pyre_object::special::w_not_implemented() as i64);
            let is_not_implemented = ctx
                .trace_ctx
                .record_op(OpCode::PtrEq, &[result, not_implemented]);
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[is_not_implemented])?;
        }
    }
    Ok(Some(inlined))
}

/// Inline a plain Python rich-compare dunder after the numeric COMPARE_OP
/// specializations decline. The receiver class and its version tag pin the
/// descriptor lookup; a proper-subclass rhs declines so its reflected dunder
/// retains priority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_user_compareop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    op_tag: i64,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 2 {
        return Ok(None);
    }

    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    // Forward rich-compare dunder only; a proper-subclass rhs declines below so
    // its reflected dunder (__lt__/__gt__, __le__/__ge__, __eq__/__ne__ self)
    // keeps priority, matching try_compare_override's forward-first dispatch.
    let dunder = match cmp_op {
        pyre_interpreter::bytecode::ComparisonOperator::Less => "__lt__",
        pyre_interpreter::bytecode::ComparisonOperator::LessOrEqual => "__le__",
        pyre_interpreter::bytecode::ComparisonOperator::Greater => "__gt__",
        pyre_interpreter::bytecode::ComparisonOperator::GreaterOrEqual => "__ge__",
        pyre_interpreter::bytecode::ComparisonOperator::Equal => "__eq__",
        pyre_interpreter::bytecode::ComparisonOperator::NotEqual => "__ne__",
    };

    let lhs = r_args[0];
    let rhs = r_args[1];
    let Some(concrete_lhs) = walker_concrete_ref_object(ctx, lhs) else {
        return Ok(None);
    };
    let Some(concrete_rhs) = walker_concrete_ref_object(ctx, rhs) else {
        return Ok(None);
    };

    // A tagged immediate is an exact builtin `int` with C-level operator slots:
    // it has no heap `ob_type`/`w_class` to pin, and its dunder is not inlinable
    // Python code.  Decline before the concrete derefs below, which would fault
    // on the immediate (`typedef::r#type` stays the tagged-safe typing path).
    // Inert behind `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(concrete_lhs)
            || pyre_object::tagged_int::is_tagged_int(concrete_rhs))
    {
        return Ok(None);
    }

    let w_class = unsafe { (*concrete_lhs).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        return Ok(None);
    }

    let Some(w_typ_r) = pyre_interpreter::typedef::r#type(concrete_rhs) else {
        return Ok(None);
    };
    if !std::ptr::eq(w_class, w_typ_r.as_ptr())
        && unsafe { pyre_object::typeobject::w_type_issubtype(w_typ_r.as_ptr(), w_class) }
    {
        return Ok(None);
    }

    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };
    // Same reading as the binary-op route: the two operands bind the first two
    // parameters and the resolved descent seeds the rest from `__defaults__`.
    if nparams < 2 || (nparams != 2 && !binop_defaulted_params_enabled()) {
        return Ok(None);
    }
    // A `NotImplemented` result has to go back to the operator protocol.
    // `perform_call` (`pyjitpl.py`) traces the dunder as an MIFrame; the
    // walker admits the same way. [`BinopRewindInlineGuard`] refuses the
    // first commit before the helper runs, so the cut is empty of effects
    // when the result is handed back.
    if !dunder_body_admissible_on_rewind(w_code) {
        return Ok(None);
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_lhs),
        ConcreteValue::Ref(concrete_rhs),
    ];
    // Rewind point for the `NotImplemented` arm below.  Nothing above this
    // line records IR or touches the heap cache.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Keep the `MetaInterp.virtualref_boxes` side state at the same boundary
    // as the history: the attempted compare dunder can open a virtual-ref
    // scope before returning `NotImplemented` and being discarded.
    let pre_fold_virtualrefs = ctx.trace_ctx.snapshot_virtualref_boxes();
    let effects_before = fbw_executed_effect_count();
    let unjournaled_before = fbw_has_unjournaled_effect();
    // A decline below returns to the residual, which re-executes the dunder.
    // Journal entries pushed after this mark are undone there
    // (`fbw_effect_journal_rollback_since`); the walk can still commit, so
    // `fbw_store_journal_rollback` does not.
    let journal_mark = fbw_effect_journal_mark();
    let method_const = ctx.trace_ctx.const_ref(method as i64);
    // Copied out before the descent takes `ctx` mutably; the region state is
    // the session's, not this frame's.
    // Value-producing boxes are numbered by opencoder `_index` (`OpRef::RefOp`
    // `raw()`); `_count` also counts voids, so a store into a region-fresh
    // allocation would fail `obj.raw() >= from` and residualize the dunder.
    let region_from = ctx.trace_ctx.get_trace_position()._index;
    let session = ctx.session;
    let rewind_guard = BinopRewindInlineGuard::enter(session, region_from);
    let descent = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![lhs, rhs],
        vec![
            ConcreteValue::Ref(concrete_lhs),
            ConcreteValue::Ref(concrete_rhs),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((lhs, concrete_lhs, w_class, version_tag, unsafe {
            inline_attr_cell_guard(w_class, dunder, method)
        })),
        Some((rhs, concrete_rhs, w_typ_r.as_ptr())),
        // `entry_is_call_boundary`.  What decides it is whether the abort
        // can name this entry as a resume coordinate, not whether the entry
        // is spelled CALL.  `binop_rewind_enabled` is that name.
        binop_rewind_enabled(),
        false,
        None,
    );
    // The refusal that sets this happens BEFORE the commit runs, so the region
    // has applied nothing and the cut is the legal exit.  Asking here rather
    // than at the `NotImplemented` arm below is the whole point: by the time a
    // result exists, the write that had to be prevented has happened.
    //
    // The cut takes the snapshots with it.  The descent attaches guards before
    // it declines, and a snapshot minted inside it names the discarded
    // operation namespace — including the `virtual_ref` pair the callee's
    // `enter` pushed.  `cut_trace` leaves the side table alone, so those
    // entries survive into the optimizer, which remaps every published
    // snapshot and resolves the stale `OpRef` against a position the cut has
    // since handed to another operation.
    let descent = match descent {
        Err(error @ DispatchError::TraceTooLong { .. }) => return Err(error),
        other => other,
    };
    let refused_a_commit = rewind_guard.refused();
    drop(rewind_guard);
    if refused_a_commit {
        fbw_effect_journal_rollback_since(journal_mark);
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx
            .restore_virtualref_boxes(pre_fold_virtualrefs.clone());
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    let Some(inlined) = descent? else {
        fbw_effect_journal_rollback_since(journal_mark);
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r.get(dst).expect("ref register in range");
        if matches!(
            concrete_from_recorded_opref(ctx, result),
            ConcreteValue::Ref(obj)
                if std::ptr::eq(obj, pyre_object::special::w_not_implemented())
        ) {
            // Hand the result back to the full binary protocol.  Cutting the
            // recorded descent away and letting the ordinary residual run
            // applies the dunder exactly once -- but only while this walk has
            // applied nothing since `pre_fold_pos`, because the residual
            // re-executes the body and there is no journal undo for what user
            // code did.  The same all-clear reading the un-lowered-helper
            // rollback takes, and for its reason too: a walk already carrying
            // an unjournaled effect can no longer flush a `CloseLoop` end.
            //
            // The snapshots go with the operations, for the reason the refusal
            // arm above records.
            if binop_rewind_enabled()
                && fbw_executed_effect_count() == effects_before
                && !unjournaled_before
                && !fbw_has_unjournaled_effect()
            {
                fbw_effect_journal_rollback_since(journal_mark);
                ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
                ctx.trace_ctx.restore_virtualref_boxes(pre_fold_virtualrefs);
                ctx.trace_ctx.heap_cache_mut().reset();
                return Ok(None);
            }
            return Err(DispatchError::callee_inline_unsupported(op.pc));
        }
        if !result.is_constant() {
            let not_implemented = ctx
                .trace_ctx
                .const_ref(pyre_object::special::w_not_implemented() as i64);
            let is_not_implemented = ctx
                .trace_ctx
                .record_op(OpCode::PtrEq, &[result, not_implemented]);
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[is_not_implemented])?;
        }
    }
    Ok(Some(inlined))
}

/// Route `f"{x:spec}"` through an app-level `__format__`.
///
/// FORMAT_WITH_SPEC lowers to `residual_call_r_r(format_with_spec_fn,
/// [value, spec])`, and `bh_format_with_spec_fn` reaches the receiver dunder
/// through `runtime_ops::format_value` → `space.format`. When that dunder is
/// Python, the whole method body sits behind an opaque MayForce residual and
/// runs interpreted once per iteration. This is the `__format__` sibling of
/// [`try_walker_inline_user_binop`]: pin the promoted receiver class, then
/// enter the ordinary resolved-callee plumbing with the value as `self` and
/// the spec as the second parameter.
///
/// Unlike a binary operator there is no `NotImplemented` escape hatch to hand
/// back to a wider protocol — `__format__` either returns a string or raises —
/// so this route needs none of binop's rewind machinery, and a body that
/// commits an effect is admitted like any other inlined callee.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_format<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 2 {
        return Ok(None);
    }

    // Name every bail, as the binop route does: a FORMAT_WITH_SPEC that stays
    // residual costs a full interpreted `__format__` per execution, and
    // without a reason line the only observable is the runtime.
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[format-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    let value = r_args[0];
    let spec = r_args[1];
    let Some(concrete_value) = walker_concrete_ref_object(ctx, value) else {
        decline!("value has no concrete ref");
    };
    let Some(concrete_spec) = walker_concrete_ref_object(ctx, spec) else {
        decline!("spec has no concrete ref");
    };

    // A tagged immediate is an exact builtin `int` with C-level slots: no heap
    // `w_class` to pin, and its `__format__` is not inlinable Python code.
    // Decline before the deref below, which would fault on the immediate.
    // Inert behind `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && pyre_object::tagged_int::is_tagged_int(concrete_value)
    {
        decline!("tagged immediate receiver");
    }

    let w_class = unsafe { (*concrete_value).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        decline!("receiver w_class is not a type");
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        decline!(format_args!(
            "receiver class {} has no version tag",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    }

    let Some(method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, "__format__") })
    else {
        decline!(format_args!("{} has no __format__", unsafe {
            pyre_object::typeobject::w_type_get_name(w_class)
        }));
    };
    // `object.__format__` and the builtin type overrides are C slots, so this
    // is also what refuses a receiver that has no app-level `__format__` at
    // all — no separate comparison against the default is needed.
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        decline!(format_args!(
            "{}.__format__ is not inlinable Python code",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    };
    if nparams != 2 {
        decline!(format_args!(
            "{}.__format__ takes {nparams} params",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    }

    // If this straight-line body is safe to sample, refuse a bad result
    // before the sub-walk emits anything.  `require_str_result` still
    // declines a record-time non-str; this sample is the format sibling of
    // the one in `try_walker_inline_exception_string_override`.
    if let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) {
        if body_facts.exc_override_sample_safe {
            let sampled = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(
                    method,
                    &[concrete_value, concrete_spec],
                )
            };
            let sampled_is_str = matches!(sampled, Ok(result)
                if !result.is_null() && unsafe { pyre_object::is_str(result) });
            if !sampled_is_str {
                decline!("sampled __format__ result is not str");
            }
        }
    }

    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_value),
        ConcreteValue::Ref(concrete_spec),
    ];
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let inlined = try_walker_inline_resolved_user_call_inner(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![value, spec],
        vec![
            ConcreteValue::Ref(concrete_value),
            ConcreteValue::Ref(concrete_spec),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((value, concrete_value, w_class, version_tag, unsafe {
            inline_attr_cell_guard(w_class, "__format__", method)
        })),
        None,
        // FORMAT_WITH_SPEC is not a Python CALL. A record-time non-str
        // declines to the residual (`require_str_result`); a guard inside
        // the inlined body resumes through `OperatorTail::Format`, which
        // runs `DescrOperation.format`'s post-call str check.
        false,
        // `__format__` returning a non-string is a TypeError
        // `call_format_dispatch_w` raises; `require_str_result` keeps that
        // at record time, and `OperatorTail::Format` keeps it on deopt.
        true,
        None,
        None,
        false,
        None,
        None,
        Some(crate::operator_continuation::OperatorTail::Format),
        None,
        Vec::new(),
        None,
    )?;
    // A refused result — `__format__` answering a non-str — declines from
    // inside the call above after the body has been walked. Cut that
    // emission back so the caller's residual runs the operator itself.
    if inlined.is_none() {
        cut_declined_subwalk(ctx, pre_fold_pos);
    }
    Ok(inlined)
}

/// State taken before a protocol-dunder inline so a non-bool result can be
/// cut back to the residual.  [`try_walker_inline_format`] declines a bad
/// result from inside the walk (`require_str_result`); `__contains__` and
/// `__instancecheck__` have no such flag, and `contains()` / `isinstance()`
/// bool-check the method's return after the call.
struct ProtocolBoolRewind {
    pre_fold_pos: majit_metainterp::recorder::TracePosition,
    virtualrefs: Vec<(OpRef, usize)>,
    effects_before: usize,
    unjournaled_before: bool,
    journal_mark: FbwEffectJournalMark,
}

fn protocol_bool_rewind<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>) -> ProtocolBoolRewind {
    ProtocolBoolRewind {
        pre_fold_pos: ctx.trace_ctx.get_trace_position(),
        virtualrefs: ctx.trace_ctx.snapshot_virtualref_boxes(),
        effects_before: fbw_executed_effect_count(),
        unjournaled_before: fbw_has_unjournaled_effect(),
        journal_mark: fbw_effect_journal_mark(),
    }
}

/// After a protocol dunder returns, pin a bool result.  A non-bool is cut
/// back to the residual when this walk has applied nothing since `rewind`,
/// because the residual re-executes the method and `is_true`s the answer.
/// A committed effect has no such exit.
fn finish_inlined_bool<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    dst: usize,
    label: &str,
    inlined: (DispatchOutcome, usize),
    rewind: ProtocolBoolRewind,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if matches!(inlined.0, DispatchOutcome::Continue) && inlined.1 == op.next_pc {
        let result = ctx.registers_r.get(dst).expect("ref register in range");
        let bool_obj = match concrete_from_recorded_opref(ctx, result) {
            ConcreteValue::Ref(obj) if !obj.is_null() && unsafe { pyre_object::is_bool(obj) } => {
                Some(obj)
            }
            _ => None,
        };
        let Some(bool_obj) = bool_obj else {
            if fbw_inline_diag_enabled() {
                eprintln!(
                    "[{label}-inline-decline] pc={} why=result is not bool",
                    op.pc
                );
            }
            if fbw_executed_effect_count() == rewind.effects_before
                && !rewind.unjournaled_before
                && !fbw_has_unjournaled_effect()
            {
                fbw_effect_journal_rollback_since(rewind.journal_mark);
                cut_declined_subwalk(ctx, rewind.pre_fold_pos);
                ctx.trace_ctx.restore_virtualref_boxes(rewind.virtualrefs);
                return Ok(None);
            }
            return Err(DispatchError::callee_inline_unsupported(op.pc));
        };
        walker_guard_class(ctx, op.pc, result, unsafe { (*bool_obj).ob_type } as i64)?;
    }
    Ok(Some(inlined))
}

/// Inline `haystack.__contains__(needle)` for CONTAINS_OP.
///
/// `compare_op_from_tag` only decodes the six rich comparisons, so
/// [`try_walker_inline_user_compareop`] never enters this opcode.
/// `COMPARE_OP_NOT_CONTAINS` stays residual: the helper inverts the
/// method's answer, and publishing that answer raw would flip `not in`.
///
/// No `NotImplemented` arm and no [`BinopRewindInlineGuard`].  The method
/// returns a bool or raises, the same shape as [`try_walker_inline_format`].
/// `entry_is_call_boundary` is [`binop_rewind_enabled`]: that flag names this
/// opcode as a resume coordinate a seeded callee can convert at.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_contains_dunder<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    op_tag: i64,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 2 {
        return Ok(None);
    }
    // `runtime_ops::compare_value` inverts the protocol result for `not in`.
    if op_tag != pyre_interpreter::runtime_ops::COMPARE_OP_CONTAINS {
        return Ok(None);
    }

    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[contains-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    let needle = r_args[0];
    let haystack = r_args[1];
    let Some(concrete_needle) = walker_concrete_ref_object(ctx, needle) else {
        decline!("needle has no concrete ref");
    };
    let Some(concrete_haystack) = walker_concrete_ref_object(ctx, haystack) else {
        decline!("haystack has no concrete ref");
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(concrete_needle)
            || pyre_object::tagged_int::is_tagged_int(concrete_haystack))
    {
        decline!("tagged immediate operand");
    }

    let w_class = unsafe { (*concrete_haystack).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        decline!("haystack w_class is not a type");
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        decline!("haystack class has no version tag");
    }

    let Some(method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, "__contains__") })
    else {
        decline!("haystack class has no __contains__");
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        decline!("__contains__ is not inlinable Python code");
    };
    if nparams != 2 {
        decline!("__contains__ arity is not 2");
    }

    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_haystack),
        ConcreteValue::Ref(concrete_needle),
    ];
    let rewind = protocol_bool_rewind(ctx);
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![haystack, needle],
        vec![
            ConcreteValue::Ref(concrete_haystack),
            ConcreteValue::Ref(concrete_needle),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((haystack, concrete_haystack, w_class, version_tag, unsafe {
            inline_attr_cell_guard(w_class, "__contains__", method)
        })),
        None,
        binop_rewind_enabled(),
        false,
        None,
    )?
    else {
        return Ok(None);
    };
    finish_inlined_bool(ctx, op, dst, "contains", inlined, rewind)
}

/// Inline `type(classinfo).__instancecheck__(classinfo, obj)`.
///
/// The `type`-metaclass bake ([`try_walker_specialize_builtin_isinstance`])
/// has already declined.  `abc.py ABCMeta.__instancecheck__` is the one-line
/// forwarder to `_abc_instancecheck`; entering it here is what lets that
/// body trace.  The wrapper descent runs only when this returns `None`:
/// its `Err` aborts the trace, and `type.__instancecheck__` is a slot
/// [`resolve_inlinable_callee`] refuses, so an ordinary class stays on that
/// descent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_isinstance_dunder<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 4 {
        return Ok(None);
    }

    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[isinstance-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    // `try_walker_specialize_builtin_isinstance` reads the call through the
    // frame's register shadow (`read_ref_var_list_concrete`). The traced
    // OpRef's own concrete is a different object here, so consulting it
    // reports a callable that is not `isinstance` and leaves the residual.
    let arg_shadow = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_classinfo),
    ) = (
        arg_shadow.first().copied().unwrap_or(ConcreteValue::Null),
        arg_shadow.get(1).copied().unwrap_or(ConcreteValue::Null),
        arg_shadow.get(2).copied().unwrap_or(ConcreteValue::Null),
        arg_shadow.get(3).copied().unwrap_or(ConcreteValue::Null),
    )
    else {
        decline!("isinstance args are not refs");
    };
    if concrete_callable.is_null()
        || !pyre_interpreter::builtins::is_builtin_isinstance_function(concrete_callable)
    {
        decline!("callable is not isinstance");
    }
    if !null_or_self.is_null() {
        decline!("isinstance self slot is not null");
    }
    if concrete_obj.is_null() {
        decline!("obj has no concrete ref");
    }
    if concrete_classinfo.is_null() {
        decline!("classinfo has no concrete ref");
    }
    let obj = r_args[2];
    let classinfo = r_args[3];
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(concrete_obj)
            || pyre_object::tagged_int::is_tagged_int(concrete_classinfo))
    {
        decline!("tagged immediate operand");
    }
    let Some(meta) = pyre_interpreter::typedef::r#type(concrete_classinfo) else {
        decline!("classinfo has no type");
    };
    let metaclass = meta.as_ptr();
    let w_class = unsafe { (*concrete_classinfo).w_class };
    if w_class.is_null()
        || !std::ptr::eq(w_class, metaclass)
        || !unsafe { pyre_object::is_type(metaclass) }
    {
        decline!("classinfo class is not its type");
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(metaclass) };
    if version_tag == 0 {
        decline!("metaclass has no version tag");
    }
    let Some(method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(metaclass, "__instancecheck__") })
    else {
        decline!("metaclass has no __instancecheck__");
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        decline!("__instancecheck__ is not inlinable Python code");
    };
    if nparams != 2 {
        decline!("__instancecheck__ arity is not 2");
    }

    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_classinfo),
        ConcreteValue::Ref(concrete_obj),
    ];
    let rewind = protocol_bool_rewind(ctx);
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![classinfo, obj],
        vec![
            ConcreteValue::Ref(concrete_classinfo),
            ConcreteValue::Ref(concrete_obj),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((
            classinfo,
            concrete_classinfo,
            metaclass,
            version_tag,
            unsafe { inline_attr_cell_guard(metaclass, "__instancecheck__", method) },
        )),
        None,
        true,
        false,
        None,
    )?
    else {
        return Ok(None);
    };
    finish_inlined_bool(ctx, op, dst, "isinstance", inlined, rewind)
}

/// Allocate the callee's three symbolic register banks for a sub-walk
/// entered through any `inline_call_*` arm.
///
/// Each bank is sized to `num_regs_X + constants_X.len()`
/// (RPython `JitCode.num_regs_and_consts_X`) so callee bytecode that
/// reads the post-regs constant window (indices
/// `[num_regs_X, num_regs_and_consts_X)`) finds a populated slot.
/// Constant slots are filled via `TraceCtx::const_int` / `const_ref` /
/// `const_float`, matching RPython
/// `pyjitpl.py MIFrame.copy_constants`.
///
/// Also returns Ref- and Int-bank concrete shadows sized to match
/// `registers_r` / `registers_i`.  Constant slots seed their concrete
/// directly from the pools: `ConcreteValue::Int(v)` from
/// `body.constants_i` (so a `goto_if_not/iL` reading a constant input
/// can fold the branch) and `ConcreteValue::Ref(v)` from
/// `body.constants_r` — a Ref constant's runtime value IS the pooled
/// object pointer (kept alive by the jitcode), and the nested
/// call-inline gate (`try_walker_inline_user_call`) reads the callable
/// through this shadow when a callee body calls another function
/// through its own baked const-pool callable.
pub(crate) fn allocate_callee_register_banks(
    body: &SubJitCodeBody,
    trace_ctx: &mut TraceCtx,
) -> (
    Vec<OpRef>,
    Vec<OpRef>,
    Vec<OpRef>,
    Vec<ConcreteValue>,
    Vec<ConcreteValue>,
) {
    let total_r = body.num_regs_r + body.constants_r.len();
    let total_i = body.num_regs_i + body.constants_i.len();
    let total_f = body.num_regs_f + body.constants_f.len();
    let mut regs_r = vec![OpRef::NONE; total_r];
    let mut regs_i = vec![OpRef::NONE; total_i];
    let mut regs_f = vec![OpRef::NONE; total_f];
    let mut concrete_r = vec![ConcreteValue::Null; total_r];
    let mut concrete_i = vec![ConcreteValue::Null; total_i];
    for (i, &v) in body.constants_i.iter().enumerate() {
        regs_i[body.num_regs_i + i] = trace_ctx.const_int(v);
        concrete_i[body.num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, v) in body.constants_r.iter().enumerate() {
        let v = v.get();
        regs_r[body.num_regs_r + i] = trace_ctx.const_ref(v);
        concrete_r[body.num_regs_r + i] =
            ConcreteValue::Ref(v as usize as pyre_object::PyObjectRef);
    }
    for (i, &v) in body.constants_f.iter().enumerate() {
        regs_f[body.num_regs_f + i] = trace_ctx.const_float(v);
    }
    (regs_r, regs_i, regs_f, concrete_r, concrete_i)
}

/// Apply the non-exceptional frame-exit state transition and return the
/// callee's value for caller-side shape handling.
///
/// `pyjitpl.py finishframe` clears `last_exc_value` before `popframe()`. The
/// slot lives on the `WalkSession`, shared by every frame in the walk, so this
/// one clear is what the resumed caller observes.
pub(crate) fn finish_inline_callee_return<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    result: Option<OpRef>,
) -> Option<OpRef> {
    ctx.clear_last_exc_value();
    result
}

/// The `*_from_tag` helpers follow a publish-and-return-NULL convention: on a
/// raise they set the exception and hand back NULL, so a sub-walk that
/// recorded that arm looks like `SubReturn` with `last_exc_value` set.
/// `finishframe` would dest-write the NULL and clear the exception; promote to
/// `finishframe_exception` instead.
///
/// `exc_before` is the slot as it stood before the sub-walk.  The slot is
/// session-wide and survives a `SubRaise` that routed into a Python handler
/// (`try_catch_exception_at` re-installs it and continues), so inside an
/// `except` body a helper that legitimately returns NULL would otherwise
/// re-raise the exception that body already caught.  Only a value the callee
/// itself installed is the published-NULL convention.
pub(crate) fn promote_published_null_return_since<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    outcome: DispatchOutcome,
    pc: usize,
    exc_before: Option<OpRef>,
) -> Result<DispatchOutcome, DispatchError> {
    let DispatchOutcome::SubReturn { result: Some(op) } = outcome else {
        return Ok(outcome);
    };
    // A void return (`None`) or a real box must not be treated as a raise
    // just because the caller still holds a stale last_exc.  Only the
    // published-NULL arm of `jit_*_from_tag` is that shape.
    let published_null = matches!(
        ctx.trace_ctx.box_value(op),
        Some(majit_ir::Value::Ref(r)) if r.0 == 0
    );
    if !published_null {
        return Ok(DispatchOutcome::SubReturn { result: Some(op) });
    }
    match ctx.last_exc_value().filter(|&exc| Some(exc) != exc_before) {
        Some(_) => {
            // Recording-time cells must not leak into a later loop.
            // The compiled path consumes them via `GUARD_EXCEPTION`
            // (`handle_possible_exception`) so `signature()` walking a
            // 2-frame tb after `bigint_same` does not re-raise.
            if let Some(cb) = crate::callbacks::try_get() {
                (cb.drain_backend_jit_exc)();
            }
            super::walker_record_guard_exception(ctx, pc)?;
            Ok(DispatchOutcome::SubRaise {
                exc: ctx
                    .last_exc_value()
                    .expect("GUARD_EXCEPTION keeps last_exc_value"),
                exc_concrete: ctx.last_exc_value_concrete(),
            })
        }
        None => Ok(DispatchOutcome::SubReturn { result: Some(op) }),
    }
}

/// `//` / `%` and the other exact-int raisers publish + return NULL.
/// After a successful inline of those tags, record `GUARD_NO_EXCEPTION`
/// so a later zero divisor deopts instead of dest-writing NULL into
/// the caller's `+=` slot (`flip_floor`).
pub(crate) fn maybe_guard_no_exception_after_raising_binop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    int_arg_concretes: &[ConcreteValue],
) -> Result<(), DispatchError> {
    let Some(ConcreteValue::Int(tag)) = int_arg_concretes.first().copied() else {
        return Ok(());
    };
    use pyre_interpreter::bytecode::BinaryOperator as B;
    let raises = matches!(
        pyre_interpreter::runtime_ops::binary_op_from_tag(tag),
        Some(
            B::FloorDivide
                | B::InplaceFloorDivide
                | B::Remainder
                | B::InplaceRemainder
                | B::TrueDivide
                | B::InplaceTrueDivide
                | B::Power
                | B::InplacePower
                | B::Lshift
                | B::InplaceLshift
                | B::Subscr
        )
    );
    if !raises {
        return Ok(());
    }
    ctx.trace_ctx
        .record_guard(majit_ir::OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_inline_call_subwalk<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    pc: usize,
    descr_index: usize,
    sub_index: usize,
    sub_body: &SubJitCodeBody,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
    float_args: &[OpRef],
) -> Result<InlineCallOutcome, DispatchError> {
    // `pyjitpl.py MIFrame.setup` walks the *callee* jitcode with that
    // jitcode's own descrs.  Build-time helpers (`from_canonical`)
    // resolve `d`/`j` through the process-wide table.  The caller's
    // per-fn pool is the wrong namespace: a switch/id inside
    // `binary_value_from_tag` then jumps to the wrong arm and the
    // portal guard snapshot misses live locals (`checksum += i // d`).
    // Production full-body walks always have `snapshot_sym`, so they
    // take the orthodox helper entry even when the per-fn descr slot
    // does not carry the `uses_global_descr_pool` flag.
    //
    // A synthetic fixture's descr index is a slot in its own
    // `descr_refs`, and the same small index in `ALL_DESCRS` names an
    // unrelated canonical helper (`RawDescrPool::runtime_jitcode_at`).
    // Honor `uses_global_descr_pool` only when that global slot is the
    // callee body this opcode is about to walk.  A per-fn pool numbers
    // its own slots, so the index is already the callee.
    let global_slot_is_callee = match ctx.raw_descrs {
        super::RawDescrPool::PerFn(_) => true,
        super::RawDescrPool::Global => {
            crate::jitcode_runtime::get_descr_by_index(descr_index).is_some_and(|descr| {
                matches!(
                    descr,
                    majit_jitcode::jitcode::BhDescr::JitCode { jitcode_index, .. }
                        if *jitcode_index == sub_index
                )
            }) && crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
                .is_some_and(|jitcode| jitcode.code.as_ptr() == sub_body.code.as_ptr())
        }
    };
    let uses_global_descr_pool = (ctx
        .raw_descrs
        .runtime_jitcode_at(descr_index)
        .is_some_and(|jitcode| jitcode.uses_global_descr_pool())
        && global_slot_is_callee)
        || !ctx.fbw_mode.snapshot_sym.is_null();
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let walk = if !uses_global_descr_pool {
        run_sub_jitcode_walk(
            ctx,
            pc,
            sub_body,
            int_args,
            int_arg_concretes,
            ref_args,
            ref_arg_concretes,
            float_args,
        )
    } else {
        super::specialize::run_codewriter_helper_inline_call(
            ctx,
            pc,
            sub_body,
            int_args,
            int_arg_concretes,
            ref_args,
            ref_arg_concretes,
            float_args,
        )
    };
    match walk {
        // Same cut as `try_walker_orthodox_descent`: an un-lowered helper
        // inside the callee is not a portal abort.  Residualize through
        // `jitcode.fnaddr` (`blackhole.py bhimpl_inline_call_*`).
        //
        // `GuardResumeCoordinateUnavailable` is the same class of decline:
        // `run_codewriter_helper_inline_call` cannot stamp the helper's
        // parent resume word (`pyjitpl.py capture_resumedata`).  Aborting
        // the caller would drop a recursive carrier before `getframe`
        // forces it; residualize the helper and keep walking the caller.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { .. })
        | Err(DispatchError::GuardResumeCoordinateUnavailable { .. }) => {
            cut_declined_subwalk(ctx, pre_fold_pos);
            // A declined helper walk must not become an opaque
            // `CallMayForce` when the callee is `int_add` / `add` /
            // `binary_value_from_tag`.  Emit the machine-int body
            // (`int_add_ovf` + box) the walk would have recorded —
            // do not re-enter `binary_value_from_tag`, which inlines
            // `add` and recurses.
            if ref_args.len() == 2
                && let Some(op_tag) = super::specialize::binary_op_tag_for_helper_index(
                    ctx.raw_descrs,
                    sub_index,
                    int_arg_concretes,
                )
                && let Some((dst_bank, dst, _)) = call_opcode_result_dst(code, pc)
                && dst_bank == 'r'
            {
                if fbw_debug_abort_enabled() {
                    let name = crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
                        .map(|jc| jc.name.as_str())
                        .unwrap_or("<missing>");
                    eprintln!("[bin-emit] pc={pc} name={name} tag={op_tag}");
                }
                if let Some(outcome) = spec_gate(SpecFold::BinaryOpDescent, || {
                    super::specialize::try_emit_exact_int_binop(
                        ctx, pc, op_tag, ref_args, dst, dst_bank,
                    )
                })? {
                    return Ok(outcome.into());
                }
                if matches!(
                    pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
                    Some(pyre_interpreter::bytecode::BinaryOperator::Subscr)
                ) && let Some(outcome) = spec_gate(SpecFold::Subscr, || {
                    super::specialize::try_emit_list_int_getitem(ctx, pc, ref_args, dst, dst_bank)
                })? {
                    return Ok(outcome.into());
                }
            } else if fbw_debug_abort_enabled() {
                let name = crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
                    .map(|jc| jc.name.as_str())
                    .unwrap_or("<missing>");
                eprintln!(
                    "[bin-emit] miss pc={pc} name={name} i={} r={}",
                    int_args.len(),
                    ref_args.len()
                );
            }
            residualize_inline_call_via_fnaddr(
                ctx,
                code,
                pc,
                descr_index,
                int_args,
                ref_args,
                float_args,
            )
        }
        other => other.map(InlineCallOutcome::from),
    }
}

struct InlineFnaddrCall {
    fnaddr: i64,
    allboxes: Vec<OpRef>,
    descr: majit_ir::DescrRef,
}

fn inline_fnaddr_call_setup<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    descr_index: usize,
    int_args: &[OpRef],
    ref_args: &[OpRef],
    float_args: &[OpRef],
) -> Result<InlineFnaddrCall, DispatchError> {
    let Some(jc) = ctx.raw_descrs.runtime_jitcode_at(descr_index) else {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, symbolic: 0 });
    };
    inline_fnaddr_call_setup_from_jc(ctx, pc, &jc, int_args, ref_args, float_args)
}

fn inline_fnaddr_call_setup_binary_helper<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    int_args: &[OpRef],
    ref_args: &[OpRef],
) -> Result<InlineFnaddrCall, DispatchError> {
    let Some(jc) = super::specialize::binary_value_from_tag_jitcode() else {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, symbolic: 0 });
    };
    inline_fnaddr_call_setup_from_jc(ctx, pc, &jc, int_args, ref_args, &[])
}

fn inline_fnaddr_call_setup_from_jc<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    jc: &majit_metainterp::jitcode::JitCode,
    int_args: &[OpRef],
    ref_args: &[OpRef],
    float_args: &[OpRef],
) -> Result<InlineFnaddrCall, DispatchError> {
    let fnaddr = jc.fnaddr;
    if fnaddr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr) {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
            pc,
            symbolic: fnaddr,
        });
    }
    let calldescr = jc.calldescr().clone();
    let descr = crate::descr::make_descr_from_bh(&majit_jitcode::jitcode::BhDescr::Call {
        calldescr: calldescr.clone(),
    });
    let funcbox = ctx.trace_ctx.const_int(fnaddr);
    let mut allboxes = Vec::with_capacity(1 + calldescr.arg_classes.len());
    allboxes.push(funcbox);
    let mut i_cur = 0usize;
    let mut r_cur = 0usize;
    let mut f_cur = 0usize;
    for ch in calldescr.arg_classes.chars() {
        let arg =
            match ch {
                'i' => {
                    let arg = *int_args.get(i_cur).ok_or(
                        DispatchError::OrthodoxSubWalkTraceUnsupported {
                            pc,
                            symbolic: fnaddr,
                        },
                    )?;
                    i_cur += 1;
                    arg
                }
                'r' => {
                    let arg = *ref_args.get(r_cur).ok_or(
                        DispatchError::OrthodoxSubWalkTraceUnsupported {
                            pc,
                            symbolic: fnaddr,
                        },
                    )?;
                    r_cur += 1;
                    arg
                }
                'f' => {
                    let arg = *float_args.get(f_cur).ok_or(
                        DispatchError::OrthodoxSubWalkTraceUnsupported {
                            pc,
                            symbolic: fnaddr,
                        },
                    )?;
                    f_cur += 1;
                    arg
                }
                _ => {
                    return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                        pc,
                        symbolic: fnaddr,
                    });
                }
            };
        allboxes.push(arg);
    }
    Ok(InlineFnaddrCall {
        fnaddr,
        allboxes,
        descr,
    })
}

/// A helper CALL's outcome plus whether the CALL was turned into a residual
/// that already recorded its own post-call exception guard.
///
/// A caller that adds a second guard of its own from the operator tag would
/// record `GUARD_NO_EXCEPTION` twice for one call.
struct InlineCallOutcome {
    outcome: DispatchOutcome,
    guard_no_exception_recorded: bool,
}

impl From<DispatchOutcome> for InlineCallOutcome {
    fn from(outcome: DispatchOutcome) -> Self {
        Self {
            outcome,
            guard_no_exception_recorded: false,
        }
    }
}

/// `cpu.bh_call_*(jitcode.fnaddr, ...)` for a helper body the walker
/// could not record.  The codewriter already required a callable
/// `fnaddr` (`fully_bound_callee_body`); blackhole uses the same
/// address on guard-failure resume.
fn residualize_inline_call_via_fnaddr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    pc: usize,
    descr_index: usize,
    int_args: &[OpRef],
    ref_args: &[OpRef],
    float_args: &[OpRef],
) -> Result<InlineCallOutcome, DispatchError> {
    let InlineFnaddrCall {
        fnaddr,
        allboxes,
        descr,
    } = inline_fnaddr_call_setup(ctx, pc, descr_index, int_args, ref_args, float_args)?;
    let Some(call_descr) = descr.as_call_descr() else {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
            pc,
            symbolic: fnaddr,
        });
    };
    // `bhimpl_inline_call_*` uses `jitcode.fnaddr` and the callee's
    // own `calldescr`.  Pick the same `Call*` / `CallMayForce*` /
    // `CallPure*` family `select_residual_call_opcode` would emit for
    // that descr (`do_residual_call` / `_opimpl_residual_call*`),
    // including FORCE_TOKEN + `GuardNotForced` when the helper can force.
    let ei = call_descr.get_extra_info();
    let dst_bank = match call_descr.result_type() {
        majit_ir::Type::Int => 'i',
        majit_ir::Type::Float => 'f',
        majit_ir::Type::Void => 'v',
        majit_ir::Type::Ref => 'r',
    };
    let (call_opcode, can_raise, emit_guard_not_forced) =
        super::residual_call::select_residual_call_opcode(
            &ei,
            dst_bank,
            "residualize_inline_call_via_fnaddr",
        );
    super::residual_call::walker_abort_if_mayforce_null_ref_arg(
        call_opcode,
        &allboxes,
        call_descr,
        ctx,
        pc,
    )?;
    if emit_guard_not_forced {
        super::residual_call::maybe_walker_vable_and_vrefs_before_residual_call(ctx, pc);
    }
    // Flatten BINARY_SUBSCR of a pre-bound `f_locals` proxy does not
    // go through residual_call's BINARY_OP arm, so the write-back
    // that keeps `p["x"]` live must run here too.
    super::residual_call::write_back_locals_for_proxy_reader(ctx, &allboxes);
    let profiled_call = matches!(
        call_opcode,
        OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
    );
    if profiled_call {
        ctx.trace_ctx
            .profiler()
            .count_ops(call_opcode, majit_metainterp::counters::OPS);
    }
    // `pyjitpl.py:1943` takes `patch_pos` before recording the call so
    // `record_result_of_call_pure` can cut it back out.
    let patch_pos = ctx.trace_ctx.get_trace_position();
    let recorded = ctx
        .trace_ctx
        .record_op_with_descr(call_opcode, &allboxes, descr.clone());
    // `MIFrame.execute_varargs(pure=True)` parity — see
    // `dispatch_residual_call_iRd_kind` for the upstream walk.
    let recorded = super::residual_call::try_fold_pure_call_via_executor(
        ctx,
        call_opcode,
        &allboxes,
        call_descr,
        descr.clone(),
        patch_pos,
        recorded,
    );
    // See `dispatch_residual_call_iRd_kind`: count only a standing call.
    if profiled_call && recorded.inline_const_to_value().is_none() {
        ctx.trace_ctx
            .profiler()
            .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
    }
    let resid = super::residual_call::try_execute_residual_call_via_executor(
        ctx,
        call_opcode,
        &allboxes,
        call_descr,
        recorded,
        pc,
        None,
        false,
    )?;
    let resid_raised = match resid {
        super::ResidualExecOutcome::Executed(result) => result.is_err(),
        super::ResidualExecOutcome::Declined(_) => {
            return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
                pc,
                symbolic: fnaddr,
            });
        }
    };
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
    // `do_residual_call` writes dest before `GUARD_NOT_FORCED` so the
    // fail_args snapshot sees the result box.  The caller still dest-writes
    // again from `SubReturn` (`finishframe`); the second write is the same
    // OpRef.
    if let Some((op_dst_bank, dst, next_pc)) = call_opcode_result_dst(code, pc) {
        super::residual_call::write_residual_call_result_to_dst(
            ctx,
            pc,
            dst,
            op_dst_bank,
            recorded,
        )?;
        if emit_guard_not_forced {
            ctx.live_after_jit_pc = next_pc;
        }
    }
    if emit_guard_not_forced {
        ctx.trace_ctx
            .record_guard(majit_ir::OpCode::GuardNotForced, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    // `pyjitpl.py handle_possible_exception`: a standing last_exc_value
    // always `finishframe_exception`s.  The EI `can_raise` flag only
    // chooses whether a successful call records `GUARD_NO_EXCEPTION`.
    // `jit_binary_value_from_tag` publishes + returns null on raise;
    // treating that as `SubReturn` dest-writes NULL and swallows the
    // TypeError / OverflowError (`keys() & None`, `[0] * 2**63`).
    if resid_raised {
        walker_record_guard_exception(ctx, pc)?;
        let exc = ctx
            .last_exc_value()
            .ok_or(DispatchError::OrthodoxSubWalkTraceUnsupported {
                pc,
                symbolic: fnaddr,
            })?;
        return Ok(DispatchOutcome::SubRaise {
            exc,
            exc_concrete: ctx.last_exc_value_concrete(),
        }
        .into());
    }
    if can_raise {
        ctx.trace_ctx
            .record_guard(majit_ir::OpCode::GuardNoException, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    Ok(InlineCallOutcome {
        outcome: DispatchOutcome::SubReturn {
            result: (dst_bank != 'v').then_some(recorded),
        },
        guard_no_exception_recorded: can_raise,
    })
}

thread_local! {
    /// Active explicit sub-walk driver for this OS thread.  The pointer is
    /// scoped by `SubWalkDriverGuard` and is only dereferenced synchronously by
    /// a nested `run_sub_jitcode_walk` using the same `Sym` monomorphization.
    /// It replaces Rust call-stack recursion; semantic state remains on the
    /// walk session / heap-owned frames, like PyPy's `MetaInterp.framestack`.
    static SUBWALK_DRIVER: std::cell::Cell<*mut ()> = const {
        std::cell::Cell::new(std::ptr::null_mut())
    };
}

static SUBWALK_DIRECT_RESUME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUBWALK_CALL_REPLAY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUBWALK_HEAP_CLONES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Nested-helper dest-write resumes, CALL replays, and heap-cache clones.
pub fn subwalk_resume_counts() -> (u64, u64) {
    (
        SUBWALK_DIRECT_RESUME.load(std::sync::atomic::Ordering::Relaxed),
        SUBWALK_CALL_REPLAY.load(std::sync::atomic::Ordering::Relaxed),
    )
}

pub fn subwalk_heap_clone_count() -> u64 {
    SUBWALK_HEAP_CLONES.load(std::sync::atomic::Ordering::Relaxed)
}

struct SubWalkDriverGuard {
    previous: *mut (),
}

impl SubWalkDriverGuard {
    fn install<Sym: WalkSym>(exchange: &mut SubWalkExchange<'_, Sym>) -> Self {
        let pointer = exchange as *mut SubWalkExchange<'_, Sym> as *mut ();
        let previous = SUBWALK_DRIVER.with(|slot| slot.replace(pointer));
        debug_assert!(previous.is_null(), "sub-walk drivers must not nest");
        Self { previous }
    }
}

impl Drop for SubWalkDriverGuard {
    fn drop(&mut self) {
        SUBWALK_DRIVER.with(|slot| slot.set(self.previous));
    }
}

struct HelperLiveGuard<'a> {
    session: &'a std::cell::RefCell<WalkSession>,
}

impl Drop for HelperLiveGuard<'_> {
    fn drop(&mut self) {
        super::vable_ops::pop_helper_live(self.session);
    }
}

struct SubWalkFrame<'a, Sym: WalkSym> {
    _helper_live: HelperLiveGuard<'a>,
    id: usize,
    caller_pc: usize,
    /// The caller's `last_exc_value` when this frame was pushed.  The slot is
    /// session-wide, so only a value this frame installed marks its NULL
    /// return as the publish-and-return-NULL convention.
    caller_exc_before: Option<OpRef>,
    pc: usize,
    body: SubJitCodeBody,
    seed_from_active_resume: bool,
    registers_r: RegisterBank,
    registers_i: RegisterBank,
    registers_f: RegisterBank,

    concrete_registers_i: Vec<ConcreteValue>,
    /// The active and paused adapters share the same frame-owned slots.
    frame_state: WalkFrameState,
    inline_callee_consts: Option<InlineCalleeConsts>,
    fbw_mode: FbwWalkMode<Sym>,
    session: &'a std::cell::RefCell<WalkSession>,
    descr_refs: &'a dyn DescrRefTable,
    raw_descrs: RawDescrPool<'a>,
    is_authoritative_executor: bool,
    sub_jitcode_lookup: &'a SubJitCodeLookup,
    entry_py_pc: EntryPyPc,
    outer_resume_marker_jit_pc: Option<usize>,
    outer_jitcode_index: u32,

    pending_guard_snapshot_error: Option<DispatchError>,

    vstack_depth: usize,
    vstack_cur_pypc: u32,
    vstack_valid: bool,

    vstack_reorder_ceiling: u32,

    vstack_handler_landing_py: Option<u32>,
    live_before_jit_pc: usize,
    live_after_jit_pc: usize,
    /// `pyjitpl.py finishframe` + `make_result_of_lastop`: dest write for an
    /// `inline_call_*` SubReturn, applied when this frame is next driven.
    /// Residual SubReturn still replays the CALL so a cut specialize
    /// preamble is recorded again.
    pending_subreturn: Option<PendingSubReturn>,
    /// Results of the callees this frame's current CALL step already
    /// entered, in entry order.  The step replays from its start after each
    /// callee finishes, so every entry before the one that suspended must
    /// receive its own earlier result again rather than walk the callee a
    /// second time; `finishframe` hands each result to the call that made it.
    replay_log: Vec<CompletedSubWalk>,
}

struct PendingSubReturn {
    dst_bank: char,
    dst: usize,
    caller_pc: usize,
    result: Option<OpRef>,
    /// `SubWalkFrame::caller_exc_before` of the callee that produced `result`.
    exc_before: Option<OpRef>,
}

impl<'a, Sym: WalkSym> SubWalkFrame<'a, Sym> {
    fn drive(
        &mut self,
        trace_ctx: &mut TraceCtx,
    ) -> Result<(DispatchOutcome, usize), DispatchError> {
        let mut walk_ctx = WalkContext {
            frame_state: self.frame_state.clone(),
            inline_callee_consts: self.inline_callee_consts,
            fbw_mode: self.fbw_mode,
            session: self.session,
            registers_r: &self.registers_r,
            registers_i: &self.registers_i,
            registers_f: &self.registers_f,

            concrete_registers_i: &mut self.concrete_registers_i,
            descr_refs: self.descr_refs,
            raw_descrs: self.raw_descrs,
            is_authoritative_executor: self.is_authoritative_executor,
            trace_ctx,
            is_top_level: false,
            sub_jitcode_lookup: self.sub_jitcode_lookup,
            entry_py_pc: self.entry_py_pc,
            outer_resume_marker_jit_pc: self.outer_resume_marker_jit_pc,
            outer_jitcode_index: self.outer_jitcode_index,

            pending_guard_snapshot_error: self.pending_guard_snapshot_error.take(),

            vstack_depth: self.vstack_depth,
            vstack_cur_pypc: self.vstack_cur_pypc,
            vstack_valid: self.vstack_valid,

            vstack_reorder_ceiling: self.vstack_reorder_ceiling,

            vstack_handler_landing_py: self.vstack_handler_landing_py,
            live_before_jit_pc: self.live_before_jit_pc,
            live_after_jit_pc: self.live_after_jit_pc,
        };
        if self.seed_from_active_resume {
            self.seed_from_active_resume = false;
            if let Some(frame) =
                ActiveResumeFrame::current(walk_ctx.session, walk_ctx.fbw_mode.snapshot_sym)
                && frame.body_matches(&self.body)
            {
                seed_callee_vstack_mirror(&mut walk_ctx, &frame);
            }
        }
        // ChangeFrame: dest-write from the finished callee, then walk at
        // the opcode after the CALL.  The first CALL step already ran
        // vstack/live bookkeeping before it suspended.
        let mut published_raise = None;
        let dest_err = if let Some(pending) = self.pending_subreturn.take() {
            if let DispatchOutcome::SubRaise { exc, exc_concrete } =
                promote_published_null_return_since(
                    &mut walk_ctx,
                    DispatchOutcome::SubReturn {
                        result: pending.result,
                    },
                    pending.caller_pc,
                    pending.exc_before,
                )?
            {
                published_raise = Some(DispatchOutcome::SubRaise { exc, exc_concrete });
                None
            } else {
                let applied = match finish_inline_callee_return(&mut walk_ctx, pending.result) {
                    Some(value) => super::residual_call::write_residual_call_result_to_dst(
                        &mut walk_ctx,
                        pending.caller_pc,
                        pending.dst,
                        pending.dst_bank,
                        value,
                    )
                    .is_ok(),
                    None => pending.dst_bank == 'v',
                };
                if applied {
                    SUBWALK_DIRECT_RESUME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                } else {
                    Some(DispatchError::UnexpectedVoidSubReturn {
                        pc: pending.caller_pc,
                    })
                }
            }
        } else {
            None
        };
        // The bank is rooted by `SubWalkDriver::push_frame` for this frame's
        // whole residency, which outlasts this call.
        let result = if let Some(raised) = published_raise {
            Ok((raised, self.pc))
        } else if let Some(err) = dest_err {
            Err(err)
        } else {
            walk(self.body.code, self.pc, &mut walk_ctx)
        };

        self.inline_callee_consts = walk_ctx.inline_callee_consts;
        self.fbw_mode = walk_ctx.fbw_mode;
        self.entry_py_pc = walk_ctx.entry_py_pc;
        self.outer_resume_marker_jit_pc = walk_ctx.outer_resume_marker_jit_pc;
        self.outer_jitcode_index = walk_ctx.outer_jitcode_index;

        self.pending_guard_snapshot_error = walk_ctx.pending_guard_snapshot_error.take();

        self.vstack_depth = walk_ctx.vstack_depth;
        self.vstack_cur_pypc = walk_ctx.vstack_cur_pypc;
        self.vstack_valid = walk_ctx.vstack_valid;

        self.vstack_reorder_ceiling = walk_ctx.vstack_reorder_ceiling;

        self.vstack_handler_landing_py = walk_ctx.vstack_handler_landing_py;
        self.live_before_jit_pc = walk_ctx.live_before_jit_pc;
        self.live_after_jit_pc = walk_ctx.live_after_jit_pc;
        result
    }
}

/// Dest of an `inline_call_*` at `caller_pc`.  Residual calls stay on
/// the replay path so a cut specialize preamble is recorded again.
fn inline_call_subreturn_dst(code: &[u8], caller_pc: usize) -> Option<(char, usize, usize)> {
    let op = crate::jitcode_runtime::decode_op_at(code, caller_pc)?;
    if !op.opname.starts_with("inline_call_") {
        return None;
    }
    call_opcode_result_dst(code, caller_pc)
}

/// Dest bank, dest register, and the pc after a CALL-family opcode.
/// `None` when the byte at `caller_pc` is not a decodable call.
fn call_opcode_result_dst(code: &[u8], caller_pc: usize) -> Option<(char, usize, usize)> {
    let op = crate::jitcode_runtime::decode_op_at(code, caller_pc)?;
    let dst_bank = match op.argcodes.rsplit_once('>') {
        Some((_, dst)) if dst.starts_with('r') => 'r',
        Some((_, dst)) if dst.starts_with('i') => 'i',
        Some((_, dst)) if dst.starts_with('f') => 'f',
        _ => 'v',
    };
    if dst_bank == 'v' {
        return Some(('v', 0, op.next_pc));
    }
    let dst = *code.get(op.next_pc.checked_sub(1)?)? as usize;
    Some((dst_bank, dst, op.next_pc))
}

struct CompletedSubWalk {
    parent_id: usize,
    caller_pc: usize,
    result: Result<DispatchOutcome, DispatchError>,
    class_of_last_exc_is_const: bool,
}

/// The only state nested `walk()` activations exchange with the outer driver.
/// Keeping it separate from `SubWalkDriver.frames` matters: the active frame
/// is already mutably borrowed while `step()` calls back into
/// `run_sub_jitcode_walk`, so a TLS pointer to the whole driver would create a
/// second overlapping `&mut SubWalkDriver`.  PyPy's `_interpret` likewise
/// communicates through framestack operations, not by re-entering an already
/// borrowed frame object.
struct SubWalkExchange<'a, Sym: WalkSym> {
    pending: Option<SubWalkFrame<'a, Sym>>,
    /// The active frame's [`SubWalkFrame::replay_log`], lent to it while it
    /// is driven.
    completed: Vec<CompletedSubWalk>,
    /// How many `completed` entries the current pass of the step consumed.
    completed_cursor: usize,
    active_frame_id: usize,
    next_frame_id: usize,
    step_trace_position: Option<majit_metainterp::recorder::TracePosition>,
    step_effect_count: usize,
    step_unjournaled: bool,
    /// Heap-cache state at the start of the CALL step.  Parked on the
    /// paused parent (`SubWalkFrame.replay_heap_cache`) so a later
    /// `SubRaise` can still replay the CALL; a `SubReturn` drops it.
    step_heap_cache: Option<majit_metainterp::heapcache::HeapCache>,
    residual_step: bool,
}

struct SubWalkDriver<'a, Sym: WalkSym> {
    frames: Vec<SubWalkFrame<'a, Sym>>,
    /// One entry per live frame, pushed and dropped with `frames`.
    bank_guards: Vec<crate::trace::InlineRegisterBankGuard>,
    state_guards: Vec<crate::trace::InlineFrameStateGuard>,
    exchange: SubWalkExchange<'a, Sym>,
}

impl<'a, Sym: WalkSym> SubWalkDriver<'a, Sym> {
    fn new(root: SubWalkFrame<'a, Sym>) -> Self {
        let next_frame_id = root.id + 1;
        let mut driver = Self {
            frames: Vec::new(),
            bank_guards: Vec::new(),
            state_guards: Vec::new(),
            exchange: SubWalkExchange {
                pending: None,
                completed: Vec::new(),
                completed_cursor: 0,
                active_frame_id: 0,
                next_frame_id,
                step_trace_position: None,
                step_effect_count: 0,
                step_unjournaled: false,
                step_heap_cache: None,
                residual_step: false,
            },
        };
        driver.push_frame(root);
        driver
    }

    /// Push a frame and root its Ref bank for as long as it stays on
    /// `frames`.
    ///
    /// The bank has to be walkable for the frame's whole residency, not only
    /// while its own `drive` call is on the host stack: a parent suspended at
    /// a CALL keeps the `ConstPtr`s its folds minted in `registers_r` while
    /// the child walks and allocates.  The recursive form this driver replaced
    /// got that by nesting one guard activation inside the next, so
    /// `inline_register_banks` already held every live bank at once — its only
    /// reader walks the whole list (`trace.rs walk_active_sym_register_area`).
    ///
    /// Each registration retains the frame's actual shared slots. A `frames`
    /// realloc cannot move those slots, and guard retirement finds its bank
    /// by identity, including when setup and driver registrations overlap.
    fn push_frame(&mut self, frame: SubWalkFrame<'a, Sym>) {
        self.frames.push(frame);
        let bank = &self
            .frames
            .last()
            .expect("the frame just pushed")
            .registers_r;
        self.bank_guards
            .push(crate::trace::InlineRegisterBankGuard::enter(bank));
        self.state_guards
            .push(crate::trace::InlineFrameStateGuard::enter(
                &self
                    .frames
                    .last()
                    .expect("the frame just pushed")
                    .frame_state,
            ));
    }

    /// Pop a frame, retiring its bank root first so the two stacks stay in
    /// step.
    fn pop_frame(&mut self) -> SubWalkFrame<'a, Sym> {
        self.state_guards
            .pop()
            .expect("a live sub-walk frame lost its state root");
        self.bank_guards
            .pop()
            .expect("a live sub-walk frame lost its register-bank root");
        self.frames.pop().expect("completed sub-walk frame missing")
    }

    fn drive(
        &mut self,
        trace_ctx: &mut TraceCtx,
    ) -> Result<(DispatchOutcome, bool), DispatchError> {
        loop {
            let frame = self
                .frames
                .last_mut()
                .expect("sub-walk driver lost its root frame");
            self.exchange.active_frame_id = frame.id;
            self.exchange.completed = std::mem::take(&mut frame.replay_log);
            self.exchange.completed_cursor = 0;
            let driven = frame.drive(trace_ctx);
            match driven {
                // `SwitchToBlackhole` unwinds `_interpret`. Replaying the
                // parent CALL would record another step after the raise.
                Err(error @ DispatchError::TraceTooLong { .. }) => {
                    // `frame` is `&mut` into `self.frames`. Moving the
                    // reference ends that borrow so the stack can be popped.
                    // `drop` on a reference does not.
                    let _ = frame;
                    self.exchange.completed.clear();
                    while !self.frames.is_empty() {
                        let _ = self.pop_frame();
                    }
                    return Err(error);
                }
                Err(DispatchError::SubWalkSuspended { pc }) => {
                    assert_eq!(
                        fbw_executed_effect_count(),
                        self.exchange.step_effect_count,
                        "sub-walk push occurred after a concrete effect"
                    );
                    assert_eq!(
                        fbw_has_unjournaled_effect(),
                        self.exchange.step_unjournaled,
                        "sub-walk push occurred after an unjournaled effect"
                    );
                    // Residual first entries may have recorded before
                    // suspend; cut that preamble and restore the heap
                    // cache so replay records the CALL once.  `inline_call_*`
                    // records nothing before suspend and dest-write resume
                    // does not rewind it.
                    if let Some(heap_cache) = self.exchange.step_heap_cache.take() {
                        trace_ctx.cut_trace_with_snapshots(
                            self.exchange
                                .step_trace_position
                                .expect("residual sub-walk missed the parent step boundary"),
                        );
                        *trace_ctx.heap_cache_mut() = heap_cache;
                    }
                    frame.pc = pc;
                    // Keep the results this pass consumed; the callee that
                    // just suspended appends its own when it finishes.
                    let mut replay_log = std::mem::take(&mut self.exchange.completed);
                    replay_log.truncate(self.exchange.completed_cursor);
                    frame.replay_log = replay_log;
                    let child = self
                        .exchange
                        .pending
                        .take()
                        .expect("SubWalkSuspended without a pending callee frame");
                    self.push_frame(child);
                }
                result => {
                    self.exchange.completed.clear();
                    let frame = self.pop_frame();
                    let class_state = frame.fbw_mode.class_of_last_exc_is_const;
                    let caller_pc = frame.caller_pc;
                    let caller_exc_before = frame.caller_exc_before;
                    let result = result.map(|(outcome, _)| outcome);
                    let Some(parent) = self.frames.last_mut() else {
                        return result.map(|outcome| (outcome, class_state));
                    };
                    if let Ok(DispatchOutcome::SubReturn { result: ret }) = &result
                        && let Some((dst_bank, dst, next_pc)) =
                            inline_call_subreturn_dst(parent.body.code, caller_pc)
                    {
                        parent.fbw_mode.class_of_last_exc_is_const = class_state;
                        parent.pending_subreturn = Some(PendingSubReturn {
                            dst_bank,
                            dst,
                            caller_pc,
                            result: *ret,
                            exc_before: caller_exc_before,
                        });
                        parent.pc = next_pc;
                        continue;
                    }
                    SUBWALK_CALL_REPLAY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    parent.pc = caller_pc;
                    debug_assert!(
                        parent
                            .replay_log
                            .iter()
                            .all(|entry| entry.caller_pc == caller_pc),
                        "a replay log spans two CALL steps"
                    );
                    parent.replay_log.push(CompletedSubWalk {
                        parent_id: parent.id,
                        caller_pc,
                        result,
                        class_of_last_exc_is_const: class_state,
                    });
                }
            }
        }
    }
}

/// Called by `walk` immediately before each opcode step while an explicit
/// sub-walk driver is active.  A nested CALL yields and replays that one step;
/// these marks prove the replay crosses no concrete effect and delimit the IR
/// preamble to cut.
///
/// `pyjitpl.py MetaInterp.perform_call` changes frames only at a call. The
/// local replay continuation needs a heap-cache checkpoint for that CALL's
/// preamble, not for scalar operations, field reads or control flow. Both
/// direct inline calls and descents entered through residual-call handlers
/// can suspend. A non-call clears the checkpoint, so a suspension from an
/// opcode that took none finds the slot empty and the driver cuts nothing,
/// rather than replaying against another opcode's heap state.
///
/// The checkpoint itself is cloned later, by
/// [`snapshot_residual_heap_before_suspend`] at the suspend, so this mark
/// needs only the opname and the trace position.
pub(crate) fn note_subwalk_driver_step<Sym: WalkSym>(
    opname: &str,
    pc: usize,
    trace_position: majit_metainterp::recorder::TracePosition,
) {
    SUBWALK_DRIVER.with(|slot| {
        let pointer = slot.get();
        if pointer.is_null() {
            return;
        }
        // SAFETY: `SubWalkDriverGuard` installs only the driver's exchange
        // object and clears it before the enclosing stack frame returns.
        // `frames` is a disjoint field and remains exclusively borrowed by
        // the driver while this callback runs. Nested walks keep the same
        // `Sym` monomorphization.
        let exchange = unsafe { &mut *(pointer as *mut SubWalkExchange<'_, Sym>) };
        // A replay of the CALL step starts consuming its log from the first
        // entry; any other step means that CALL finished and its log is spent.
        exchange.completed_cursor = 0;
        if exchange
            .completed
            .first()
            .is_some_and(|entry| entry.caller_pc != pc)
        {
            exchange.completed.clear();
        }
        exchange.step_trace_position = Some(trace_position);
        exchange.step_effect_count = fbw_executed_effect_count();
        exchange.step_unjournaled = fbw_has_unjournaled_effect();
        // `inline_call_*` records nothing before suspend.  A residual
        // fold that fires and continues must not copy the heap cache;
        // only a residual that actually yields clones, at the yield.
        exchange.step_heap_cache = None;
        exchange.residual_step = opname.starts_with("residual_call_");
    });
}

/// Snapshot the heap cache when a residual step yields a nested walk.
/// Folds that fire without suspending never reach here.
pub(crate) fn snapshot_residual_heap_before_suspend<Sym: WalkSym>(
    heap_cache: &majit_metainterp::heapcache::HeapCache,
) {
    SUBWALK_DRIVER.with(|slot| {
        let pointer = slot.get();
        if pointer.is_null() {
            return;
        }
        // SAFETY: same scoped driver pointer as `note_subwalk_driver_step`.
        let exchange = unsafe { &mut *(pointer as *mut SubWalkExchange<'_, Sym>) };
        if exchange.residual_step && exchange.step_heap_cache.is_none() {
            SUBWALK_HEAP_CLONES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            exchange.step_heap_cache = Some(heap_cache.clone());
        }
    });
}

/// Whether `run_sub_jitcode_walk` will yield to an already-active driver
/// instead of walking the callee inline.
pub(crate) fn subwalk_driver_is_active() -> bool {
    SUBWALK_DRIVER.with(|slot| !slot.get().is_null())
}

/// Keep ops the active residual handler recorded before it yields.
///
/// `SubWalkDriver::drive` cuts a residual step back to the opcode start and
/// restores the heap cache snapshotted at the yield. That cache still names
/// the cut ops. A helper body that then uses one as a value box dies in
/// `Trace::arg_to_box`. `orthodox_list_append_commit` records the Empty-list
/// promotion and the class pins before `run_sub_jitcode_walk`; those boxes
/// are the append body's inputs, so this step must not be cut. Replay of the
/// same opcode sees the completed callee and must not record the prefix again.
pub(crate) fn keep_residual_recordings_across_suspend<Sym: WalkSym>() {
    SUBWALK_DRIVER.with(|slot| {
        let pointer = slot.get();
        if pointer.is_null() {
            return;
        }
        // SAFETY: same scoped driver pointer as `note_subwalk_driver_step`.
        let exchange = unsafe { &mut *(pointer as *mut SubWalkExchange<'_, Sym>) };
        exchange.residual_step = false;
        exchange.step_heap_cache = None;
    });
}

/// Consume the completed nested helper for `pc`, if this residual opcode is
/// replaying it. Mirrors the completed-result arm of `run_sub_jitcode_walk_from`.
pub(crate) fn take_completed_nested_subwalk<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Option<Result<DispatchOutcome, DispatchError>> {
    SUBWALK_DRIVER.with(|slot| {
        let pointer = slot.get();
        if pointer.is_null() {
            return None;
        }
        // SAFETY: same scoped driver pointer as `note_subwalk_driver_step`.
        let exchange = unsafe { &mut *(pointer as *mut SubWalkExchange<'_, Sym>) };
        let completed = exchange.completed.get(exchange.completed_cursor)?;
        if completed.parent_id != exchange.active_frame_id || completed.caller_pc != pc {
            return None;
        }
        let class_of_last_exc_is_const = completed.class_of_last_exc_is_const;
        let result = completed.result.clone();
        exchange.completed_cursor += 1;
        if result.is_ok() {
            ctx.fbw_mode.class_of_last_exc_is_const = class_of_last_exc_is_const;
        }
        Some(result)
    })
}

#[cfg(test)]
mod subwalk_checkpoint_tests {
    use super::*;

    #[test]
    fn only_calls_keep_a_heap_cache_replay_checkpoint() {
        let mut exchange = SubWalkExchange::<crate::state::PyreSym> {
            pending: None,
            completed: Vec::new(),
            completed_cursor: 0,
            active_frame_id: 0,
            next_frame_id: 1,
            step_trace_position: None,
            step_effect_count: 0,
            step_unjournaled: false,
            step_heap_cache: None,
            residual_step: false,
        };
        let _driver = SubWalkDriverGuard::install(&mut exchange);
        let position = majit_metainterp::recorder::TracePosition {
            _pos: 0,
            _count: 0,
            _index: 0,
            snapshot_data_len: 0,
            snapshot_array_data_len: 0,
        };
        let heap_cache = majit_metainterp::heapcache::HeapCache::new();
        for (opname, keeps_checkpoint) in [
            ("inline_call_r_r", false),
            ("int_add", false),
            ("residual_call_ir_i", false),
            ("getfield_gc_r", false),
            ("goto_if_not", false),
            ("live", false),
            ("inline_call_irf_v", false),
            ("int_return", false),
        ] {
            note_subwalk_driver_step::<crate::state::PyreSym>(opname, 0, position);
            assert_eq!(
                exchange.step_heap_cache.is_some(),
                keeps_checkpoint,
                "{opname}"
            );
            assert_eq!(exchange.step_trace_position, Some(position));
        }
        note_subwalk_driver_step::<crate::state::PyreSym>("residual_call_ir_r", 0, position);
        assert!(exchange.step_heap_cache.is_none());
        snapshot_residual_heap_before_suspend::<crate::state::PyreSym>(&heap_cache);
        assert!(exchange.step_heap_cache.is_some());
        keep_residual_recordings_across_suspend::<crate::state::PyreSym>();
        assert!(exchange.step_heap_cache.is_none());
        assert!(!exchange.residual_step);
        snapshot_residual_heap_before_suspend::<crate::state::PyreSym>(&heap_cache);
        assert!(exchange.step_heap_cache.is_none());
    }

    /// Two callees entered by one CALL step: the replay after the second
    /// finishes has to hand the first entry its own result again, and only
    /// leaving the step spends the log.
    #[test]
    fn a_replayed_call_step_keeps_every_earlier_callee_result() {
        let entry = |caller_pc: usize, leaf_pc: usize| CompletedSubWalk {
            parent_id: 0,
            caller_pc,
            result: Err(DispatchError::UndecodableOpcode { pc: leaf_pc }),
            class_of_last_exc_is_const: false,
        };
        let mut exchange = SubWalkExchange::<crate::state::PyreSym> {
            pending: None,
            completed: vec![entry(7, 0), entry(7, 140)],
            completed_cursor: 2,
            active_frame_id: 0,
            next_frame_id: 1,
            step_trace_position: None,
            step_effect_count: 0,
            step_unjournaled: false,
            step_heap_cache: None,
            residual_step: false,
        };
        let _driver = SubWalkDriverGuard::install(&mut exchange);
        let position = majit_metainterp::recorder::TracePosition {
            _pos: 0,
            _count: 0,
            _index: 0,
            snapshot_data_len: 0,
            snapshot_array_data_len: 0,
        };
        note_subwalk_driver_step::<crate::state::PyreSym>("residual_call_ir_r", 7, position);
        assert_eq!(exchange.completed_cursor, 0);
        assert_eq!(
            exchange
                .completed
                .iter()
                .map(|entry| entry.result.clone())
                .collect::<Vec<_>>(),
            vec![
                Err(DispatchError::UndecodableOpcode { pc: 0 }),
                Err(DispatchError::UndecodableOpcode { pc: 140 }),
            ]
        );
        note_subwalk_driver_step::<crate::state::PyreSym>("int_add", 9, position);
        assert!(exchange.completed.is_empty());
    }

    #[test]
    fn undecodable_call_site_has_no_result_dst() {
        // The driver falls back to CALL replay when the paused pc is not a
        // call opcode.  Empty code is that miss, not a void-return shape.
        assert_eq!(call_opcode_result_dst(&[], 0), None);
        assert_eq!(inline_call_subreturn_dst(&[], 0), None);
    }
}

/// Seed a callee jitcode's register banks with positional args and walk
/// its body, returning the callee's terminal [`DispatchOutcome`]
/// (`SubReturn` / `SubRaise` / `Terminate` / `SwitchToBlackhole`).
///
/// Shared descent core of the `inline_call_*` handlers
/// ([`dispatch_inline_call_dr_kind`], `_dir`, `_dirf`) — they read the
/// callee index + arglists from the caller bytecode, then delegate the
/// bank allocation, arity check, arg seeding, sub-`WalkContext`
/// construction, and `walk()` to here. A trace-time specialization can
/// also call this directly to synthesize a descent into a charon helper
/// body (e.g. `w_list_append`), passing args it already holds rather
/// than reading them from bytecode.
///
/// `pc` is the caller-site pc, used only for arity-mismatch error
/// reporting. An empty arg slice for an unused bank passes its arity
/// check trivially. The callee runs with `is_top_level == false` and
/// inherits the caller's descr pool + sub-jitcode lookup (RPython
/// `pyjitpl.py setup_call(argboxes_i, argboxes_r, argboxes_f)`).
/// Only Ref-bank concrete shadows are seeded — matching the
/// `inline_call_*` handlers, which thread `ref_arg_concretes` but no
/// Int/Float concrete shadows across the frame boundary.
pub(crate) fn run_sub_jitcode_walk<'frame, 'a: 'frame, Sym: WalkSym>(
    ctx: &mut WalkContext<'frame, 'a, Sym>,
    pc: usize,
    sub_body: &SubJitCodeBody,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
    float_args: &[OpRef],
) -> Result<DispatchOutcome, DispatchError> {
    run_sub_jitcode_walk_from(
        ctx,
        pc,
        0,
        sub_body,
        int_args,
        int_arg_concretes,
        ref_args,
        ref_arg_concretes,
        float_args,
        &[],
    )
}

/// Same as [`run_sub_jitcode_walk`], but enter the callee body at
/// `start_pc` and seed extra Ref-bank registers after the call args.
///
/// A generator resume starts at `resume_execute_frame` (`last_instr + 1`),
/// not at function entry, and needs the portal frame / ec colors plus the
/// live locals the resume block reads.  `extra_ref_seeds` is `(reg, box,
/// concrete)` — later entries overwrite earlier ones at the same color.
pub(crate) fn run_sub_jitcode_walk_from<'frame, 'a: 'frame, Sym: WalkSym>(
    ctx: &mut WalkContext<'frame, 'a, Sym>,
    pc: usize,
    start_pc: usize,
    sub_body: &SubJitCodeBody,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
    float_args: &[OpRef],
    extra_ref_seeds: &[(usize, OpRef, ConcreteValue)],
) -> Result<DispatchOutcome, DispatchError> {
    // A parent frame re-enters its inline_call opcode after the explicit
    // driver popped the callee. Consume that result before allocating or
    // reseeding anything, exactly as PyPy's `_interpret` delivers the value to
    // the now-top `framestack[-1]`.
    let driver_pointer = SUBWALK_DRIVER.with(|slot| slot.get());
    if !driver_pointer.is_null() {
        // SAFETY: installed by the enclosing invocation of this same generic
        // function; see `SubWalkDriverGuard`.
        let exchange = unsafe { &mut *(driver_pointer as *mut SubWalkExchange<'a, Sym>) };
        if let Some(completed) = exchange.completed.get(exchange.completed_cursor)
            && completed.parent_id == exchange.active_frame_id
            && completed.caller_pc == pc
        {
            exchange.completed_cursor += 1;
            if completed.result.is_ok() {
                ctx.fbw_mode.class_of_last_exc_is_const = completed.class_of_last_exc_is_const;
            }
            return completed.result.clone();
        }
    }

    let (
        callee_regs_r,
        mut callee_regs_i,
        mut callee_regs_f,
        mut callee_concrete_r,
        mut callee_concrete_i,
    ) = allocate_callee_register_banks(sub_body, ctx.trace_ctx);
    let callee_regs_r = RegisterBank::with_constants(callee_regs_r, sub_body.num_regs_r);
    let _setup_bank_guard = crate::trace::InlineRegisterBankGuard::enter(&callee_regs_r);

    if int_args.len() > sub_body.num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc,
            provided: int_args.len(),
            callee_num_regs_i: sub_body.num_regs_i,
        });
    }
    if ref_args.len() > sub_body.num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc,
            provided: ref_args.len(),
            callee_num_regs_r: sub_body.num_regs_r,
        });
    }
    if float_args.len() > sub_body.num_regs_f {
        return Err(DispatchError::InlineCallFloatArityMismatch {
            pc,
            provided: float_args.len(),
            callee_num_regs_f: sub_body.num_regs_f,
        });
    }
    for (i, arg) in int_args.iter().enumerate() {
        callee_regs_i[i] = *arg;
    }
    for (i, arg) in ref_args.iter().enumerate() {
        callee_regs_r.set(i, *arg);
    }
    for (i, arg) in float_args.iter().enumerate() {
        callee_regs_f[i] = *arg;
    }
    // Seed the callee's concrete shadows from the caller's per-arg
    // shadows (`setup_call` parity for the Int + Ref banks; the Float
    // bank has no concrete shadow companion).  A callee body folds a
    // `goto_if_not/iL` / `switch/id` over a concrete int arg, or a
    // `guard_class` over a concrete ref arg, only when its shadow is
    // seeded here.
    for (i, concrete) in int_arg_concretes.iter().enumerate() {
        callee_concrete_i[i] = *concrete;
        if let ConcreteValue::Int(value) = concrete {
            // RPython `MIFrame.setup_call` passes the original BoxInt into
            // the callee, and that box retains its recording-time `.value`.
            // Pyre's register-local concrete shadow is separate from OpRef,
            // so mirror it onto the shared OpRef as well: canonical helper
            // bodies execute field/arithmetic ops through
            // `TraceCtx::box_value`, just as RPython's executor calls
            // `box.getint()`.
            ctx.trace_ctx
                .try_set_opref_concrete(int_args[i], majit_ir::Value::Int(*value));
        }
    }
    for (i, concrete) in ref_arg_concretes.iter().enumerate() {
        callee_concrete_r[i] = *concrete;
        if let ConcreteValue::Ref(value) = concrete
            && !value.is_null()
        {
            // Same `setup_call` Box.value parity for RefFrontendOp.  Without
            // this, a canonical sub-jitcode sees the pointer only in its
            // side shadow while `getfield_gc_*` asks the OpRef for the live
            // object, losing concrete length/capacity reads and aborting a
            // data-dependent branch after the helper already mutated state.
            ctx.trace_ctx.try_set_opref_concrete(
                ref_args[i],
                majit_ir::Value::Ref(majit_ir::GcRef(*value as usize)),
            );
        }
    }
    for &(reg, opref, concrete) in extra_ref_seeds {
        if reg >= callee_regs_r.len() {
            continue;
        }
        callee_regs_r.set(reg, opref);
        callee_concrete_r[reg] = concrete;
        if let ConcreteValue::Ref(value) = concrete
            && !value.is_null()
        {
            ctx.trace_ctx.try_set_opref_concrete(
                opref,
                majit_ir::Value::Ref(majit_ir::GcRef(value as usize)),
            );
        }
    }

    let frame_id = if driver_pointer.is_null() {
        0
    } else {
        // SAFETY: same scoped driver cast as the completed-result arm above.
        let exchange = unsafe { &mut *(driver_pointer as *mut SubWalkExchange<'a, Sym>) };
        let id = exchange.next_frame_id;
        exchange.next_frame_id += 1;
        id
    };
    let registers_i = RegisterBank::with_constants(callee_regs_i, sub_body.num_regs_i);
    let registers_f = RegisterBank::with_constants(callee_regs_f, sub_body.num_regs_f);
    let frame_state = WalkFrameState::new(WalkFrameStateData {
        callee_shadow: None,
        concrete_registers_r: callee_concrete_r,
        current_exception_seed: ctx.frame_state.borrow().current_exception_seed,
        current_exception_seed_concrete: ctx.frame_state.borrow().current_exception_seed_concrete,
        outer_active_boxes: ctx.frame_state.borrow().outer_active_boxes.clone(),
        list_iter_class_guard_resume: ctx
            .frame_state
            .borrow()
            .list_iter_class_guard_resume
            .clone(),
        vstack_boxes: Vec::new(),
        vstack_last_ref: OpRef::NONE,
        vstack_reorder_saved: None,
        ..Default::default()
    });
    super::vable_ops::push_helper_live(
        ctx.session,
        &callee_regs_r,
        &registers_i,
        &registers_f,
        &frame_state,
    );
    let frame = SubWalkFrame {
        _helper_live: HelperLiveGuard {
            session: ctx.session,
        },
        id: frame_id,
        caller_pc: pc,
        caller_exc_before: ctx.last_exc_value(),
        pc: start_pc,
        body: sub_body.clone(),
        seed_from_active_resume: start_pc == 0,
        registers_r: callee_regs_r,
        registers_i,
        registers_f,
        frame_state,
        concrete_registers_i: callee_concrete_i,

        inline_callee_consts: None,
        // `op_pc` belongs to the callee JitCode.  A canonical helper has no
        // Python blackhole entry of its own, so it stays transparent while its
        // actual MIFrame/register state lives in this heap-owned frame.
        fbw_mode: FbwWalkMode {
            inline_subwalk: true,
            transparent_helper_subwalk: true,
            // A helper has no Python PC. Residual last_instr publication
            // reads this field when `current_inline_concrete_frame()==0`.
            inline_caller_py_pc: ctx
                .fbw_mode
                .inline_caller_py_pc
                .or_else(|| (ctx.vstack_cur_pypc != 0).then_some(ctx.vstack_cur_pypc)),
            ..ctx.fbw_mode
        },
        session: ctx.session,
        descr_refs: ctx.descr_refs,
        raw_descrs: ctx.raw_descrs,
        is_authoritative_executor: ctx.is_authoritative_executor,
        sub_jitcode_lookup: ctx.sub_jitcode_lookup,
        entry_py_pc: ctx.entry_py_pc,
        outer_resume_marker_jit_pc: ctx.outer_resume_marker_jit_pc,
        outer_jitcode_index: ctx.outer_jitcode_index,

        pending_guard_snapshot_error: None,

        vstack_depth: 0,
        vstack_cur_pypc: 0,
        vstack_valid: false,

        vstack_reorder_ceiling: u32::MAX,

        vstack_handler_landing_py: None,
        live_before_jit_pc: usize::MAX,
        live_after_jit_pc: usize::MAX,
        pending_subreturn: None,
        replay_log: Vec::new(),
    };

    if !driver_pointer.is_null() {
        // Nested descent: publish the heap frame and yield the parent at its
        // CALL.  The outer driver pushes it without another Rust `walk()`
        // activation.
        let exchange = unsafe { &mut *(driver_pointer as *mut SubWalkExchange<'a, Sym>) };
        snapshot_residual_heap_before_suspend::<Sym>(&*ctx.trace_ctx.heap_cache());
        assert!(exchange.pending.replace(frame).is_none());
        return Err(DispatchError::SubWalkSuspended { pc });
    }

    let descent_journal_before = fbw_store_journal_len();
    let descent_unjournaled_before = fbw_has_unjournaled_effect();
    let mut driver = SubWalkDriver::new(frame);
    let _driver_guard = SubWalkDriverGuard::install(&mut driver.exchange);
    // A nested helper's banks are not the Python frame's. Rebinding here
    // would replace the paused `MIFrame` (`pyjitpl.py` `framestack`) with
    // the helper register file, and a portal entered inside the helper
    // would rebuild the caller from that file.
    if !ctx.fbw_mode.transparent_helper_subwalk {
        super::vable_ops::bind_paused_caller_regs(
            ctx.session,
            ctx.registers_r,
            ctx.registers_i,
            ctx.registers_f,
            &ctx.frame_state,
        );
    }
    let result = driver.drive(ctx.trace_ctx);
    match result {
        Ok((outcome, class_state)) => {
            // `MetaInterp.class_of_last_exc_is_const` is shared across the
            // whole explicit framestack.
            ctx.fbw_mode.class_of_last_exc_is_const = class_state;
            Ok(outcome)
        }
        Err(error) => {
            // The callee pc is not a coordinate of this frame.
            // `convert_and_run_from_pyjitpl` resumes every frame
            // `aborted_tracing` captured; a single-frame adopt of the
            // caller is not that resume.
            ctx.session.borrow_mut().crossed_inline_subwalk = true;
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[subwalk-propagate] jitcode_pc={pc} effects={} unjournaled={} err={error:?}",
                    fbw_store_journal_len() as i64 - descent_journal_before as i64,
                    fbw_has_unjournaled_effect() != descent_unjournaled_before,
                );
            }
            Err(error)
        }
    }
}

/// Operand layout `dR>X`:
///   2B descr index + 1B varlen + N×1B Ref args + 1B `>X` dst.
///
/// RPython parity: `pyjitpl.py _opimpl_inline_call*`. The
/// `_X` suffix is the callee's *return kind* — e.g. `_opimpl_inline_call_r_i`
/// dispatches an inline call whose callee body returns via
/// `int_return/i`. Walker semantics are otherwise identical to the
/// `_r_r` arm (which originally landed inline; this helper extracts the
/// shared body so kind variants can share the dispatch logic).
///
/// `dst_bank` selects where the SubReturn value lands:
/// * `'r'`: caller's `registers_r[dst]` — pairs with callee `ref_return/r`.
/// * `'i'`: caller's `registers_i[dst]` — pairs with callee `int_return/i`.
/// * `'f'`: would pair with callee `float_return/f` — not handled by
///   this helper because the codewriter doesn't emit a `dR>f` shape
///   (float return paths use the `dIRF` arglist family).
///
/// `kind_label` mirrors `dst_bank` as a static `&str` for typed-error
/// reporting (`RegisterOutOfRange::bank`).
pub(crate) fn dispatch_inline_call_dr_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    let (args, arg_width) = read_ref_var_list(code, op, 2, ctx)?;
    let arg_concretes = read_ref_var_list_concrete(code, op, 2, ctx);

    if dst_bank == 'r'
        && args.len() == 1
        && crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index).is_some_and(|jc| {
            jc.code.as_ptr() == sub_body.code.as_ptr()
                && (jc.name == "invert"
                    || jc.name.ends_with("::invert")
                    || jc.name.ends_with("::invert_inner"))
        })
    {
        let dst = code[op.pc + 1 + 2 + arg_width] as usize;
        if let Some(outcome) =
            super::specialize::try_walker_orthodox_unary_invert(ctx, op.pc, &args, dst, dst_bank)?
        {
            return Ok((outcome, op.next_pc));
        }
    }

    if dst_bank == 'r'
        && args.len() == 1
        && crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index).is_some_and(|jc| {
            jc.code.as_ptr() == sub_body.code.as_ptr()
                && (jc.name == "pos"
                    || jc.name.ends_with("::pos")
                    || jc.name.ends_with("::pos_inner"))
        })
    {
        let dst = code[op.pc + 1 + 2 + arg_width] as usize;
        if let Some(outcome) =
            super::specialize::try_walker_orthodox_unary_pos(ctx, op.pc, &args, dst, dst_bank)?
        {
            return Ok((outcome, op.next_pc));
        }
    }

    if dst_bank == 'r'
        && args.len() == 2
        && crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
            .is_some_and(|jc| jc.code.as_ptr() == sub_body.code.as_ptr())
        && let Some(op_tag) =
            super::specialize::binary_op_tag_for_helper_index(ctx.raw_descrs, sub_index, &[])
    {
        let dst = code[op.pc + 1 + 2 + arg_width] as usize;
        let write_boxed = |ctx: &mut WalkContext<'_, '_, Sym>, boxed: OpRef| {
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, boxed);
            write_ref_reg(ctx, op.pc, dst, boxed, concrete_for_shadow)
        };
        let is_subscr = matches!(
            pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
            Some(pyre_interpreter::bytecode::BinaryOperator::Subscr)
        );
        // A receiver whose own type owns `__getitem__` resolves to a Python
        // body.  Residual BINARY_OP admitted it here; flatten now lowers
        // BINARY to an `inline_call` of the helper, so the residual gate no
        // longer sees the subscript at all -- the same move the forward-dunder
        // admission below had to be re-run for, and the storage folds around
        // it are for builtin containers this declines.
        if is_subscr
            && let Ok(setup) = inline_fnaddr_call_setup(ctx, op.pc, descr_index, &[], &args, &[])
            && let Some(call_descr) = setup.descr.as_call_descr()
            && let Some(inlined) = spec_gate(SpecFold::SubscrUserGetitem, || {
                try_walker_inline_subscr_getitem(
                    ctx, op, code, None, &args, call_descr, dst, dst_bank,
                )
            })?
        {
            return Ok(inlined);
        }
        if is_subscr
            && let Some(DispatchOutcome::SubReturn {
                result: Some(boxed),
            }) = spec_gate(SpecFold::Subscr, || {
                super::specialize::try_emit_list_int_getitem(ctx, op.pc, &args, dst, dst_bank)
            })?
        {
            write_boxed(ctx, boxed)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // Named `add`/`mul`/… helpers arrive as `dR>r`, not
        // `binary_value_from_tag`.  Descend the generated body the same
        // way the tagged helper does; do not emit `try_emit_exact_*`
        // under this descent row.
        //
        // This frame is already that helper walk (`perform_call` of
        // `binary_value_from_tag` → `and_` / `add` / …).  Opening the
        // wrapper again suspends at this inline_call and the driver
        // pushes it forever.  Walk the callee body instead.
        if !ctx.fbw_mode.transparent_helper_subwalk {
            let tag_op = ctx.trace_ctx.const_int(op_tag);
            if let Some(outcome) = spec_gate(SpecFold::BinaryOpDescent, || {
                super::specialize::try_walker_orthodox_binary_op(
                    ctx, op.pc, op_tag, tag_op, &args, dst, dst_bank,
                )
            })? {
                return Ok((outcome, op.next_pc));
            }
        }
        // Named `add`/`mul` helpers are the same BINARY family as
        // `binary_value_from_tag`.  A Python forward dunder must be
        // admitted here too, or `p + i` residualizes the whole helper.
        if let Ok(setup) = inline_fnaddr_call_setup(ctx, op.pc, descr_index, &[], &args, &[])
            && let Some(call_descr) = setup.descr.as_call_descr()
            && let Some(inlined) = try_walker_inline_user_binop(
                ctx, op, code, op_tag, &args, call_descr, dst, dst_bank,
            )?
        {
            return Ok(inlined);
        }
    }

    // Bracket the session-wide exception slot around the callee so a NULL
    // return only reads as a raise when the callee installed the exception.
    let exc_before_subwalk = ctx.last_exc_value();
    let callee_result = run_inline_call_subwalk(
        ctx,
        code,
        op.pc,
        descr_index,
        sub_index,
        &sub_body,
        &[],
        &[],
        &args,
        &arg_concretes,
        &[],
    );
    let callee_outcome = promote_published_null_return_since(
        ctx,
        callee_result?.outcome,
        op.pc,
        exc_before_subwalk,
    )?;

    match callee_outcome {
        DispatchOutcome::SubReturn { result } => match finish_inline_callee_return(ctx, result) {
            Some(value) => {
                if dst_bank == 'v' {
                    // `inline_call_r_v/dR`
                    // (`bhimpl_inline_call_r_v` `blackhole.py`)
                    // expects a void-return callee. A `Some` return here is
                    // a codewriter shape mismatch.
                    return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
                }
                let dst = code[op.pc + 1 + 2 + arg_width] as usize;
                // inline_call_* dst writeback — `value` is the callee's
                // SubReturn OpRef.  The callee's matching concrete shadow
                // was dropped at sub-walk exit; `concrete_of_opref` still
                // sees through to `constants.get_value` for callees that
                // return a constant (e.g. `LoadConst` tail), so route via
                // the unified shadow channel.  Non-constant returns surface
                // as the sentinel `GcRef(usize::MAX)` → Null fallback.
                let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
                match dst_bank {
                    'r' => {
                        write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                    }
                    'i' => {
                        write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                    }
                    _ => unreachable!(
                        "dispatch_inline_call_dr_kind dst_bank must be 'r', 'i' or 'v' (\
                     codewriter does not emit dR>f shape today)"
                    ),
                }
                Ok((DispatchOutcome::Continue, op.next_pc))
            }
            None => {
                if dst_bank == 'v' {
                    // `inline_call_r_v/dR` expects exactly this — callee
                    // exits via `void_return/`, no SubReturn writeback.
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // Same shape contract as `_r_r`: a `_r_<X>` variant promises
                // a non-void result for the dst's `>X` slot. A void return
                // reaching here is a codewriter shape mismatch.
                Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
            }
        },
        DispatchOutcome::SubRaise {
            exc,
            mut exc_concrete,
        } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let node = record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc)?;
                let emit_runtime = node.is_none();
                record_inline_exception_context(ctx.trace_ctx, exc, exc_concrete);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                // `finishframe_exception` hands the handler the exception
                // the callee raised, concrete shadow included, so a
                // downstream `raise/r` / `reraise/` in the caller's handler
                // keeps the class-known pin the callee's leg established.
                ctx.set_last_exc_value(exc, exc_concrete);
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate {
            finish_arg,
            finish_arg_type,
        } => Ok((
            DispatchOutcome::Terminate {
                finish_arg,
                finish_arg_type,
            },
            op.next_pc,
        )),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::CloseLoop { .. } => {
            // An inlined callee body must not close a loop — see
            // `SubWalkClosedLoop`.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::CompileTracePending { .. } => {
            // The compile_trace attempt is gated on `is_top_level`
            // (sub-walks run with `is_top_level == false`), so a callee
            // body can never surface it; fail loud like the CloseLoop
            // arm if that invariant ever breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            // The loop-callee CALL_ASSEMBLER request is surfaced from a
            // multi-frame inline at a `residual_call` site and consumed by
            // `try_walker_inline_user_call`; it cannot reach the `inline_call_*`
            // jitcode-op path. Fail loud (safe decline) if that invariant ever
            // breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SegmentTrace { .. } => {
            // The segmenting cut is gated on `is_top_level` for the same
            // reason as the compile_trace attempt above: its compile half
            // lives in the driver, which a callee body has no route to.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// `inline_call_ir_<X>/dIR>X` handler shared by `dIR>i` (Int result)
/// and `dIR>r` (Ref result). Same control-flow shape as
/// [`dispatch_inline_call_dr_kind`], extended with an I-list arglist
/// preceding the R-list.
///
/// Operand layout `dIR>X`:
///   2B descr index +
///   1B I-len + N×1B int args +
///   1B R-len + M×1B ref args +
///   1B `>X` dst.
///
/// RPython parity: `pyjitpl.py _opimpl_inline_call*` —
/// kind-aware variants call `setup_call(argboxes_i, argboxes_r,
/// argboxes_f)` which distributes args into the callee's typed banks
/// (`pyjitpl.py`).
///
/// `dst_bank` selects where the SubReturn value lands: `'r'` writes to
/// `registers_r[dst]` (paired with callee `ref_return/r`), `'i'`
/// writes to `registers_i[dst]` (paired with callee `int_return/i`).
///
/// Inplace add/sub/mul/and/or/xor, excluding inplace `//` and `%`: `acc //= 0`
/// in a compiled-then-raise loop must remain a residual so the except bridge
/// can blackhole (`exception_loop_warmup`).
///
/// This is the admission test for the NAMED-helper arms only — a body such as
/// `inplace_add` whose tag has to be recovered from the helper index.  A call
/// the walk has identified as `binary_value_from_tag` carries its tag in the
/// I-list and is admitted whatever that tag is.
fn inplace_int_arith_tag(tag: i64) -> bool {
    use pyre_interpreter::bytecode::BinaryOperator as B;
    matches!(
        pyre_interpreter::runtime_ops::binary_op_from_tag(tag),
        Some(
            B::InplaceAdd
                | B::InplaceSubtract
                | B::InplaceMultiply
                | B::InplaceAnd
                | B::InplaceOr
                | B::InplaceXor
        )
    )
}

pub(crate) fn dispatch_inline_call_dir_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    // I-list at offset 2 (skip descr).
    let (int_args, int_width) = read_int_var_list(code, op, 2, ctx)?;
    let int_arg_concretes = read_int_var_list_concrete(code, op, 2, ctx);
    // R-list immediately after the I-list.
    let (ref_args, ref_width) = read_ref_var_list(code, op, 2 + int_width, ctx)?;
    let ref_arg_concretes = read_ref_var_list_concrete(code, op, 2 + int_width, ctx);

    if dst_bank == 'r'
        && int_args.len() == 1
        && ref_args.len() == 2
        && let Some(ConcreteValue::Int(op_tag)) = int_arg_concretes.first().copied()
        && matches!(
            pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
            Some(pyre_interpreter::bytecode::BinaryOperator::Subscr)
        )
    {
        let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
        // A receiver whose own type owns `__getitem__` resolves to a Python
        // body.  Residual BINARY_OP admitted it; this lowering replaced that
        // gate, so the route has to be re-run here the way the forward-dunder
        // admission below is.  The storage folds after it are for builtin
        // containers, which this declines.
        if let Ok(setup) =
            inline_fnaddr_call_setup(ctx, op.pc, descr_index, &int_args, &ref_args, &[])
            && let Some(call_descr) = setup.descr.as_call_descr()
            && let Some(inlined) = spec_gate(SpecFold::SubscrUserGetitem, || {
                try_walker_inline_subscr_getitem(
                    ctx, op, code, None, &ref_args, call_descr, dst, dst_bank,
                )
            })?
        {
            return Ok(inlined);
        }
        // Residual BINARY_OP runs the full tuple/str/list/dict fold.
        // Flatten must not keep only list-int: `t[i]` / `s[i]` would
        // residualize (`pure_tupleload`, `str_subscr_hot`).  Identify
        // SUBSCR by the I-list tag, not `get_jitcode_ref_by_index` —
        // that lookup is a global index and misses the per-fn pool.
        if let Ok(setup) = inline_fnaddr_call_setup_binary_helper(ctx, op.pc, &int_args, &ref_args)
            && let Some(call_descr) = setup.descr.as_call_descr()
            && spec_gate(SpecFold::Subscr, || {
                super::specialize::try_walker_specialize_subscr(
                    ctx,
                    op.pc,
                    &ref_args,
                    &setup.allboxes,
                    call_descr,
                    dst,
                    dst_bank,
                )
            })?
            .is_some()
        {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        if let Some(DispatchOutcome::SubReturn {
            result: Some(boxed),
        }) = spec_gate(SpecFold::Subscr, || {
            super::specialize::try_emit_list_int_getitem(ctx, op.pc, &ref_args, dst, dst_bank)
        })? {
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, boxed);
            write_ref_reg(ctx, op.pc, dst, boxed, concrete_for_shadow)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // RPython `rclass.py`/`rbuiltin.py` allocation lowering: entering the
    // canonical `w_int_new` helper with one signed argument records the
    // `NEW_WITH_VTABLE + SETFIELD_GC` box directly.  The LLBC fallback body
    // spells the same operation as a zero-argument allocation residual plus
    // header/payload stores; its generic allocator has no host fnaddr (it is
    // an lltype allocation opcode, not a callable), so descending that legacy
    // spelling would abort at the symbolic funcptr.  Treat the translated
    // helper call as the allocation intrinsic at this frame boundary, exactly
    // where RPython's rtyper has already replaced `malloc(W_IntObject)`.
    let is_w_int_new = crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
        .is_some_and(|jc| jc.name == "w_int_new" && jc.code.as_ptr() == sub_body.code.as_ptr());
    if is_w_int_new
        && dst_bank == 'r'
        && int_args.len() == 1
        && ref_args.is_empty()
        && let Some(ConcreteValue::Int(value)) = int_arg_concretes.first().copied()
    {
        let boxed_ptr = pyre_object::w_int_new(value) as i64;
        let boxed = walker_box_int(ctx, op.pc, int_args[0], value)?;
        let boxed_concrete = box_int_concrete(value, boxed_ptr);
        ctx.trace_ctx.set_opref_concrete(boxed, boxed_concrete);
        let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
        let ConcreteValue::Ref(boxed_shadow) = concrete_from_recorded_opref(ctx, boxed) else {
            unreachable!("box_int_concrete must produce a Ref concrete")
        };
        write_ref_reg(ctx, op.pc, dst, boxed, ConcreteValue::Ref(boxed_shadow))?;
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // Flatten lowered BINARY to `inline_call_ir_r` of
    // `binary_value_from_tag`, so the residual_call BINARY_OP descent
    // gate is never consulted.  Run the same orthodox descent here
    // before residualizing the whole helper via fnaddr.
    //
    // Inplace tags (`a += i`) lower as a named `inplace_add` body, not
    // `binary_value_from_tag`.  Without the I-list / helper-name tag
    // those calls never reached `try_emit_exact_int_binop`.
    let is_binary_from_tag =
        super::specialize::jitcode_is_binary_value_from_tag(ctx.raw_descrs, sub_index, &sub_body);
    let op_tag = match int_arg_concretes.first() {
        Some(ConcreteValue::Int(tag)) if is_binary_from_tag => Some(*tag),
        Some(ConcreteValue::Int(tag)) if inplace_int_arith_tag(*tag) => Some(*tag),
        _ => super::specialize::binary_op_tag_for_helper_index(
            ctx.raw_descrs,
            sub_index,
            &int_arg_concretes,
        )
        .filter(|&tag| inplace_int_arith_tag(tag)),
    };
    // Residual BINARY_OP records the raising arm before dest-write
    // (`try_walker_specialize_binary_op_int_zero_div`).  Flatten lands
    // `acc //= 0` here, so the same raise must fire before a helper
    // walk dest-writes a wrap.  Skipping `GUARD_FALSE` of a proven
    // `int_eq(0, 0)` avoided InvalidLoop but left that wrap as the
    // compiled result (`exception_loop_warmup`).
    //
    // The raw I-list concrete is a BINARY tag only for `binary_value_from_tag`.
    // `compare_value_from_tag` puts a COMPARE tag in the same slot, so reading
    // it as a BINARY tag makes every `a < b` look like a divisor operand.  The
    // helper-index fallback keys on the callee's name instead, so it needs no
    // such gate.
    let zero_div_tag = match int_arg_concretes.first() {
        Some(ConcreteValue::Int(tag)) if is_binary_from_tag => Some(*tag),
        _ => super::specialize::binary_op_tag_for_helper_index(
            ctx.raw_descrs,
            sub_index,
            &int_arg_concretes,
        ),
    };
    if dst_bank == 'r'
        && ref_args.len() == 2
        && let Some(div_tag) = zero_div_tag
        && let Ok(setup) =
            inline_fnaddr_call_setup(ctx, op.pc, descr_index, &int_args, &ref_args, &[])
        && let Some(call_descr) = setup.descr.as_call_descr()
        && let Some(outcome) = spec_gate(SpecFold::BinaryOpIntZeroDiv, || {
            super::specialize::try_walker_specialize_binary_op_int_zero_div(
                ctx,
                op.pc,
                div_tag,
                &ref_args,
                &setup.allboxes,
                call_descr,
                dst_bank,
            )
        })?
    {
        return Ok((outcome, op.next_pc));
    }
    if dst_bank == 'r'
        && ref_args.len() == 2
        && let Some(op_tag) = op_tag
    {
        let int_args_tag = int_args
            .first()
            .copied()
            .unwrap_or_else(|| ctx.trace_ctx.const_int(op_tag));
        let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
        // Emit the machine-int body before descending `binary_value_from_tag`.
        // The descent walks `int_add_ovf`; a bridge InputArg for `total` has
        // no sidecar stamp, so that walk used to abort the except path before
        // this emit ran.  Boxed concretes are available here
        // (`try_emit_exact_int_binop` reads the heap objects and passes them
        // as `known`), so `total += 2` after a caught raise records
        // `int_add_ovf` instead of aborting the bridge.
        if let Some(DispatchOutcome::SubReturn {
            result: Some(boxed),
        }) = spec_gate(SpecFold::BinaryOpDescent, || {
            super::specialize::try_emit_exact_int_binop(
                ctx, op.pc, op_tag, &ref_args, dst, dst_bank,
            )
        })? {
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, boxed);
            write_ref_reg(ctx, op.pc, dst, boxed, concrete_for_shadow)?;
            // `pyjitpl.py finishframe` clears `last_exc_value` before popframe
            // on a non-exceptional return. This emit replaces that inline
            // call, so the following `opimpl_catch_exception` sees a clear
            // slot the way the residual success arm does.
            ctx.clear_last_exc_value();
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // Inside a transparent helper subwalk this frame already walks the
        // `binary_value_from_tag` body; descending again re-enters the same
        // inline_call.  Walk the callee body instead (see the `iRd` site).
        if !ctx.fbw_mode.transparent_helper_subwalk
            && let Some(outcome) = spec_gate(SpecFold::BinaryOpDescent, || {
                super::specialize::try_walker_orthodox_binary_op(
                    ctx,
                    op.pc,
                    op_tag,
                    int_args_tag,
                    &ref_args,
                    dst,
                    dst_bank,
                )
            })?
        {
            return Ok((outcome, op.next_pc));
        }
        // Residual BINARY_OP used to admit a Python forward dunder here.
        // Flatten now lowers BINARY to `inline_call` of
        // `binary_value_from_tag`, so the residual gate never sees
        // `p + i`.  Re-run the same admission before residualizing the
        // helper — `synth/binop_dunder_leaf_inline` and the defaulted
        // `_pydecimal` shape (`binop_dunder_defaulted_param`).
        // The inline_call descr is a JitCode descr, not a CallDescr;
        // borrow the helper's own `calldescr` the fnaddr residual uses.
        if let Ok(setup) =
            inline_fnaddr_call_setup(ctx, op.pc, descr_index, &int_args, &ref_args, &[])
            && let Some(call_descr) = setup.descr.as_call_descr()
            && let Some(inlined) = try_walker_inline_user_binop(
                ctx, op, code, op_tag, &ref_args, call_descr, dst, dst_bank,
            )?
        {
            return Ok(inlined);
        }
    }

    // `w_complex_new` is `malloc_typed` of a header-plus-payload struct.
    // Walking `binary_value_from_tag` records that allocation as a bare
    // `object` (`<object object at ...>`), so a compiled `abs(x - y)`
    // over complexes TypeErrors. Residualize the helper; the real
    // `complex_sub` builds a `W_ComplexObject`.
    if is_binary_from_tag
        && ref_args.len() == 2
        && ref_args.iter().any(|&operand| {
            walker_concrete_ref_object(ctx, operand)
                .is_some_and(|obj| unsafe { pyre_object::is_complex(obj) })
        })
    {
        let residualized = residualize_inline_call_via_fnaddr(
            ctx,
            code,
            op.pc,
            descr_index,
            &int_args,
            &ref_args,
            &[],
        )?;
        return Ok((residualized.outcome, op.next_pc));
    }

    // Bracket the session-wide exception slot around the callee so a NULL
    // return only reads as a raise when the callee installed the exception.
    let exc_before_subwalk = ctx.last_exc_value();
    let walked = run_inline_call_subwalk(
        ctx,
        code,
        op.pc,
        descr_index,
        sub_index,
        &sub_body,
        &int_args,
        &int_arg_concretes,
        &ref_args,
        &ref_arg_concretes,
        &[],
    )?;
    let guard_no_exception_recorded = walked.guard_no_exception_recorded;
    let callee_outcome =
        promote_published_null_return_since(ctx, walked.outcome, op.pc, exc_before_subwalk)?;

    match callee_outcome {
        DispatchOutcome::SubReturn { result } => match finish_inline_callee_return(ctx, result) {
            Some(value) => {
                if dst_bank == 'v' {
                    return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
                }
                // dst register byte sits after descr (2B) + I-list (int_width)
                // + R-list (ref_width) bytes.
                let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
                // See dispatch_inline_call_dr_kind: route the SubReturn
                // OpRef through the unified shadow channel so constant
                // return values propagate.
                let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
                match dst_bank {
                    'r' => {
                        write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                    }
                    'i' => {
                        write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                    }
                    _ => unreachable!(
                        "dispatch_inline_call_dir_kind dst_bank must be 'r', 'i' or 'v'"
                    ),
                }
                // The first I-list concrete reads as a BINARY tag only when
                // the callee is `binary_value_from_tag`.  The same slot holds
                // a COMPARE tag for `compare_value_from_tag` — `is` and
                // `is_not` are 8 and 9, which decode as `<<` and `>>` — so
                // every `x is y` recorded a guard for an operator the call
                // never performs.
                if is_binary_from_tag {
                    // A CALL residualized through `jitcode.fnaddr` already
                    // recorded its own `GUARD_NO_EXCEPTION`.
                    if !guard_no_exception_recorded {
                        maybe_guard_no_exception_after_raising_binop(
                            ctx,
                            op.pc,
                            &int_arg_concretes,
                        )?;
                    }
                    if let Some(ConcreteValue::Int(tag)) = int_arg_concretes.first().copied() {
                        use pyre_interpreter::bytecode::BinaryOperator as B;
                        if matches!(
                            pyre_interpreter::runtime_ops::binary_op_from_tag(tag),
                            Some(
                                B::FloorDivide
                                    | B::InplaceFloorDivide
                                    | B::Remainder
                                    | B::InplaceRemainder
                            )
                        ) {
                            super::specialize::walker_guard_int_div_domain_if_exact(
                                ctx, op.pc, &ref_args,
                            )?;
                        }
                    }
                }
                Ok((DispatchOutcome::Continue, op.next_pc))
            }
            None => {
                if dst_bank == 'v' {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
            }
        },
        DispatchOutcome::SubRaise {
            exc,
            mut exc_concrete,
        } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let node = record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc)?;
                let emit_runtime = node.is_none();
                record_inline_exception_context(ctx.trace_ctx, exc, exc_concrete);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                // `finishframe_exception` hands the handler the exception
                // the callee raised, concrete shadow included, so a
                // downstream `raise/r` / `reraise/` in the caller's handler
                // keeps the class-known pin the callee's leg established.
                ctx.set_last_exc_value(exc, exc_concrete);
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate {
            finish_arg,
            finish_arg_type,
        } => Ok((
            DispatchOutcome::Terminate {
                finish_arg,
                finish_arg_type,
            },
            op.next_pc,
        )),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::CloseLoop { .. } => {
            // An inlined callee body must not close a loop — see
            // `SubWalkClosedLoop`.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::CompileTracePending { .. } => {
            // The compile_trace attempt is gated on `is_top_level`
            // (sub-walks run with `is_top_level == false`), so a callee
            // body can never surface it; fail loud like the CloseLoop
            // arm if that invariant ever breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            // The loop-callee CALL_ASSEMBLER request is surfaced from a
            // multi-frame inline at a `residual_call` site and consumed by
            // `try_walker_inline_user_call`; it cannot reach the `inline_call_*`
            // jitcode-op path. Fail loud (safe decline) if that invariant ever
            // breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SegmentTrace { .. } => {
            // The segmenting cut is gated on `is_top_level` for the same
            // reason as the compile_trace attempt above: its compile half
            // lives in the driver, which a callee body has no route to.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// `inline_call_irf_<X>/dIRF>X` handler shared by `dIRF>f` (Float
/// result) and `dIRF>r` (Ref result). Extends
/// [`dispatch_inline_call_dir_kind`] with an F-list arglist following
/// the R-list.
///
/// Operand layout `dIRF>X`:
///   2B descr index +
///   1B I-len + N×1B int args +
///   1B R-len + M×1B ref args +
///   1B F-len + K×1B float args +
///   1B `>X` dst.
///
/// RPython parity: same `pyjitpl.py setup_call(argboxes_i,
/// argboxes_r, argboxes_f)` distribution — all three kind banks
/// populated from the three lists.
///
/// `dst_bank` selects where the SubReturn value lands: `'f'` writes
/// `registers_f[dst]` (paired with callee `float_return/f`), `'r'`
/// writes `registers_r[dst]` (paired with callee `ref_return/r`).
pub(crate) fn dispatch_inline_call_dirf_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    let (int_args, int_width) = read_int_var_list(code, op, 2, ctx)?;
    let int_arg_concretes = read_int_var_list_concrete(code, op, 2, ctx);
    let (ref_args, ref_width) = read_ref_var_list(code, op, 2 + int_width, ctx)?;
    let ref_arg_concretes = read_ref_var_list_concrete(code, op, 2 + int_width, ctx);
    let (float_args, float_width) = read_float_var_list(code, op, 2 + int_width + ref_width, ctx)?;

    // The `w_int_new` boundary of `dispatch_inline_call_dir_kind`, for the
    // float box.  `w_float_new`'s translated body builds the whole
    // `W_FloatObject` and writes it into the collector block in one copy;
    // that is neither the `malloc_typed` shape `fuse_boxing_alloc` rewrites
    // into a `NewWithVtable` nor a field-wise store sequence, so the header
    // constructor and the copy both stay symbolic funcptr calls the sub-walk
    // cannot record.  Take the helper call as the allocation intrinsic here,
    // where the rtyper has already replaced `malloc(W_FloatObject)`.
    let is_w_float_new = crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
        .is_some_and(|jc| jc.name == "w_float_new" && jc.code.as_ptr() == sub_body.code.as_ptr());
    if is_w_float_new
        && dst_bank == 'r'
        && int_args.is_empty()
        && ref_args.is_empty()
        && float_args.len() == 1
        && let ConcreteValue::Float(value) = concrete_from_recorded_opref(ctx, float_args[0])
    {
        let boxed = walker_box_float(ctx, float_args[0]);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(
                pyre_object::floatobject::w_float_new(value) as usize,
            )),
        );
        let dst = code[op.pc + 1 + 2 + int_width + ref_width + float_width] as usize;
        let ConcreteValue::Ref(boxed_shadow) = concrete_from_recorded_opref(ctx, boxed) else {
            unreachable!("the stamped concrete of a float box is a Ref")
        };
        write_ref_reg(ctx, op.pc, dst, boxed, ConcreteValue::Ref(boxed_shadow))?;
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // Bracket the session-wide exception slot around the callee so a NULL
    // return only reads as a raise when the callee installed the exception.
    let exc_before_subwalk = ctx.last_exc_value();
    let callee_result = run_inline_call_subwalk(
        ctx,
        code,
        op.pc,
        descr_index,
        sub_index,
        &sub_body,
        &int_args,
        &int_arg_concretes,
        &ref_args,
        &ref_arg_concretes,
        &float_args,
    );
    let callee_outcome = promote_published_null_return_since(
        ctx,
        callee_result?.outcome,
        op.pc,
        exc_before_subwalk,
    )?;

    match callee_outcome {
        DispatchOutcome::SubReturn { result } => match finish_inline_callee_return(ctx, result) {
            Some(value) => {
                if dst_bank == 'v' {
                    return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
                }
                let dst = code[op.pc + 1 + 2 + int_width + ref_width + float_width] as usize;
                // See dispatch_inline_call_dr_kind: route the SubReturn
                // OpRef through the unified shadow channel so constant
                // return values propagate.
                let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
                match dst_bank {
                    'i' => {
                        write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                    }
                    'r' => {
                        write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                    }
                    'f' => {
                        let len = ctx.registers_f.len();
                        let _ =
                            ctx.registers_f
                                .get(dst)
                                .ok_or(DispatchError::RegisterOutOfRange {
                                    pc: op.pc,
                                    reg: dst,
                                    len,
                                    bank: "f",
                                })?;
                        ctx.registers_f.set(dst, value);
                    }
                    _ => unreachable!(
                        "dispatch_inline_call_dirf_kind dst_bank must be 'i', 'r', 'f' or 'v'"
                    ),
                }
                Ok((DispatchOutcome::Continue, op.next_pc))
            }
            None => {
                if dst_bank == 'v' {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
            }
        },
        DispatchOutcome::SubRaise {
            exc,
            mut exc_concrete,
        } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let node = record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc)?;
                let emit_runtime = node.is_none();
                record_inline_exception_context(ctx.trace_ctx, exc, exc_concrete);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    &mut exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                    node,
                );
                // `finishframe_exception` hands the handler the exception
                // the callee raised, concrete shadow included, so a
                // downstream `raise/r` / `reraise/` in the caller's handler
                // keeps the class-known pin the callee's leg established.
                ctx.set_last_exc_value(exc, exc_concrete);
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate {
            finish_arg,
            finish_arg_type,
        } => Ok((
            DispatchOutcome::Terminate {
                finish_arg,
                finish_arg_type,
            },
            op.next_pc,
        )),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::CloseLoop { .. } => {
            // An inlined callee body must not close a loop — see
            // `SubWalkClosedLoop`.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::CompileTracePending { .. } => {
            // The compile_trace attempt is gated on `is_top_level`
            // (sub-walks run with `is_top_level == false`), so a callee
            // body can never surface it; fail loud like the CloseLoop
            // arm if that invariant ever breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            // The loop-callee CALL_ASSEMBLER request is surfaced from a
            // multi-frame inline at a `residual_call` site and consumed by
            // `try_walker_inline_user_call`; it cannot reach the `inline_call_*`
            // jitcode-op path. Fail loud (safe decline) if that invariant ever
            // breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SegmentTrace { .. } => {
            // The segmenting cut is gated on `is_top_level` for the same
            // reason as the compile_trace attempt above: its compile half
            // lives in the driver, which a callee body has no route to.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}
