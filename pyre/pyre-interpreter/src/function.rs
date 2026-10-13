//! Function object.
#![allow(non_snake_case)]
//!
//! Wraps a code object pointer, a function name, a pointer to the
//! defining module's globals namespace, and an optional closure tuple.
//! When called, the interpreter creates a new PyFrame that *shares*
//! the globals pointer (no clone).

use pyre_object::pyobject::*;
use pyre_object::quasiimmut::QuasiImmutField;
use rustpython_wtf8::Wtf8Buf;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Type descriptor for user-defined functions.
pub static FUNCTION_TYPE: PyType = pyre_object::pyobject::new_pytype_with_weakref(
    "function",
    std::mem::offset_of!(Function, lifeline),
);
/// Type descriptor for module-level builtins.
pub static BUILTIN_FUNCTION_TYPE: PyType = pyre_object::pyobject::new_pytype_with_weakref(
    "builtin_function_or_method",
    std::mem::offset_of!(Function, lifeline),
);
/// CPython-compatible interp-level method descriptor.
///
/// PyPy represents these as `FunctionWithFixedCode`; the separate public
/// type preserves that payload and call ABI while exposing CPython's
/// `method_descriptor` surface.  Staying distinct from a user `function` is
/// load-bearing: the stdlib reducers classify descriptors by this type.
pub static METHOD_DESCRIPTOR_TYPE: PyType = pyre_object::pyobject::new_pytype("method_descriptor");
/// CPython-compatible slot wrapper descriptor.
///
/// The public type is distinct from PyPy's `FunctionWithFixedCode`, while the
/// payload deliberately reuses [`Function`]: both are immutable BuiltinCode
/// carriers and therefore have identical GC edges and call ABI.
pub static SLOT_WRAPPER_TYPE: PyType = pyre_object::pyobject::new_pytype("wrapper_descriptor");
/// The bound form of a slot wrapper — `descrobject.c PyMethodWrapper_Type`.
///
/// A `wrapper_descriptor` accessed on an instance yields this, the way a
/// `method_descriptor` yields a `builtin_function_or_method`.  Like that pair
/// the payload stays PyPy's [`Method`], so only the public class differs; the
/// separate type is what lets `inspect.ismethodwrapper` and `inspect.isbuiltin`
/// tell the two bound forms apart.
pub static METHOD_WRAPPER_TYPE: PyType = pyre_object::pyobject::new_pytype("method-wrapper");
/// A `METH_CLASS` entry of a builtin type — `descrobject.c
/// PyClassMethodDescr_Type`.
///
/// `type_ready_fill_dict` wraps such an entry in the same [`ClassMethod`]
/// carrier a `@classmethod` uses, so the payload is shared and only the public
/// class differs.  Keeping the two apart is what makes `dict.fromkeys` a
/// descriptor while a user `@classmethod` stays an ordinary wrapper, which
/// `inspect._signature_get_user_defined_method` relies on to decide whether a
/// `__init__` / `__new__` / `__call__` was written in Python.
///
/// [`ClassMethod`]: pyre_object::function::ClassMethod
pub static CLASSMETHOD_DESCRIPTOR_TYPE: PyType =
    pyre_object::pyobject::new_pytype("classmethod_descriptor");

/// User-defined function object.
///
/// Layout: `[ob_type | code | can_change_code | name_ptr | w_name | closure]`
/// - `code`: pointer to a Code object (PyCode for user funcs, BuiltinCode for builtins).
///   function.py:47 — `_immutable_fields_ = ['code?', ...]`
/// - `can_change_code`: function.py:33 — True by default; False for
///   `FunctionWithFixedCode` subclass (used by builtins).
/// - `name_ptr`: off-GC `Box<String>` containing the function name
/// - `closure`:  tuple of cell objects, or PY_NULL if no closure
/// - `w_func_globals_obj`: the module namespace dict object (`__globals__`)
#[repr(C)]
pub struct Function {
    pub ob: PyObject,
    /// Pointer to a Code object (PyCode or BuiltinCode).
    /// function.py:47 — `_immutable_fields_ = ['code?', ...]`
    pub code: *const (),
    /// function.py — `can_change_code = True`
    /// False for FunctionWithFixedCode subclass.
    pub can_change_code: bool,
    /// Function name (off-GC Box<String>).
    pub name: *const String,
    /// Application-level `__name__` object.
    ///
    /// The raw `name` pointer remains the canonical fast text carrier for
    /// frames/repr/tracebacks, while this slot preserves object identity across
    /// repeated `function.__name__` reads. `PY_NULL` means a legacy construction
    /// path has not stamped it yet.
    pub w_name: PyObjectRef,
    /// Closure: tuple of cell objects from the enclosing scope,
    /// or PY_NULL if this function has no free variables.
    pub closure: PyObjectRef,
    /// Default argument values.
    /// PyPy: Function.defs_w
    pub defs_w: PyObjectRef,
    /// Keyword-only default values.
    /// PyPy: Function.w_kw_defs
    pub w_kw_defs: PyObjectRef,
    /// function.py — `self.w_module = None`
    pub w_module: PyObjectRef,
    /// PyPy: Function.w_func_globals — the module namespace dict object.
    ///
    /// `function.py self.w_func_globals = w_globals` stores the dict
    /// object directly; this is the function's sole globals carrier, so
    /// `function.__globals__` returns the same identity as the module's
    /// `__dict__` and frames built from this function share globals.
    /// `PY_NULL` for globals-less carriers (gateway builtins).
    pub w_func_globals_obj: PyObjectRef,
    /// CPython 3.14 `PyFunctionObject.func_builtins` — resolved once from
    /// globals at function construction and then exposed by the direct
    /// read-only `__builtins__` member.  This is intentionally distinct from
    /// frame builtin selection: replacing `globals['__builtins__']` later
    /// does not change `function.__builtins__`.
    pub w_builtins: PyObjectRef,
    /// `function.py w_ann=None` constructor default plus
    /// `function.py fget_func_annotations` lazy-init shape:
    /// PyPy stores the annotations dict directly on the function and
    /// allocates an empty dict on first read when none was set.
    /// Pyre mirrors that by keeping the slot `PY_NULL` until either
    /// `MAKE_FUNCTION ANNOTATIONS` flag stamps the compile-time dict
    /// (`function.py fset_func_annotations`) or the getter
    /// lazy-fills with `w_dict_new()`.  `f.__annotations__ is
    /// f.__annotations__` identity holds because both reads see the
    /// cached slot after the first allocation.
    pub w_ann: PyObjectRef,
    /// PEP 649 `__annotate__` callable (CPython 3.14 `func_annotate`
    /// slot; `MAKE_FUNCTION` `Annotate` flag) — evaluated with
    /// `format=1` by the `__annotations__` getter when `w_ann` is
    /// still unset, then discarded in favour of the stamped dict.
    /// `PY_NULL` when the compiler emitted eager annotations or none.
    /// No PyPy counterpart (upstream function.py targets 3.11, before
    /// PEP 649); the typed slot mirrors how `w_ann` sits on the
    /// function object rather than a side table.
    pub w_annotate: PyObjectRef,
    /// `function.py self.w_func_dict = None` — lazily allocated
    /// per-function attribute dictionary.  This is a field on the function,
    /// not mapdict side storage, matching PyPy's `Function.getdict`.
    pub w_func_dict: PyObjectRef,
    /// CPython 3.14 `PyFunctionObject.func_typeparams` — the declared type
    /// parameters tuple.  `PY_NULL` represents the default empty tuple.
    pub w_typeparams: PyObjectRef,
    /// `function.py:375 self.w_doc = w_doc` constructor slot plus
    /// `function.py fget_func_doc` cache:
    ///
    /// ```python
    /// def fget_func_doc(self, space):
    ///     if self.w_doc is None:
    ///         self.w_doc = self.code.getdocstring(space)
    ///     return self.w_doc
    /// ```
    ///
    /// `PY_NULL` means "not yet resolved" (PyPy `None`); first reader
    /// caches `code.getdocstring()` here so subsequent
    /// `f.__doc__ is f.__doc__` holds.  `function_del_doc` writes
    /// `w_none()` to make the deleted state sticky against the lazy
    /// fallback (`function.py fdel_func_doc`).
    pub w_doc: PyObjectRef,
    /// `function.py self.qualname = qualname or self.name` slot —
    /// PyPy stores the qualified name as a regular field on the
    /// function object.  `PY_NULL` means "not explicitly set"; the
    /// getter falls back to `code.co_qualname` then `name` to mirror
    /// the constructor's `qualname or self.name` short-circuit when
    /// the compile path did not stamp a value at MAKE_FUNCTION time.
    pub w_qualname: PyObjectRef,
    /// `function.py fget_func_objclass / set_objclass` slot —
    /// PyPy stores the bound class on the function for descriptor
    /// introspection (`inspect.getfullargspec` etc.).  `PY_NULL`
    /// means "not bound to a class" and the getter raises
    /// AttributeError per `function.py:500`.
    pub w_objclass: PyObjectRef,
    /// `function.py fget_func_text_signature /
    /// fset_func_text_signature` slot — PyPy stores the text
    /// signature (the docstring-prefix `(...)` line that PEP 437
    /// describes for builtins) directly on the function.  `PY_NULL`
    /// means "no signature recorded" and the getter returns
    /// `space.w_None` per `function.py:488`.
    pub w_text_signature: PyObjectRef,
    /// `__self__` of a builtin `__new__` descriptor — the type whose
    /// `tp_new` this function wraps.  `typeobject.c add_tp_new_wrapper`
    /// binds `int.__new__.__self__ is int`, so `copyreg._reduce_ex`
    /// can ask `base.__new__.__self__ is base` to find the defining
    /// type.  `PY_NULL` for every ordinary function (a plain `def`
    /// has no `__self__`); only stamped on the per-type builtin
    /// `__new__` carriers at type-finalisation time.
    pub w_new_self: PyObjectRef,
    /// function.py `BuiltinFunction.w_moduleobj` — the module object
    /// bound as `__self__` for an interp-level module function. PyPy stores
    /// this on each BuiltinFunction; CPython 3.14 likewise exposes the
    /// defining module object from `PyCFunction_GET_SELF`.
    pub w_moduleobj: PyObjectRef,
    /// The hidden `mutate_<name>` field `quasiimmut.py get_mutate_field_name`
    /// synthesises for each `?` entry of `function.py:34-42
    /// _immutable_fields_`. Upstream's rtyper makes one per declared field, so
    /// mutating `__defaults__` revokes only the loops that folded `defs_w`;
    /// collapsing them onto a shared slot would still be sound but would
    /// over-invalidate.
    ///
    /// Behind one lazily allocated block rather than nine inline slots, because
    /// this word is the only part a fresh `Function` must have right.
    /// `emit_make_function_inline` allocates with `NewWithVtable`, and
    /// `rewrite.py clear_gc_fields` writes only the census's GC-pointer
    /// slots — incminimark does not zero-fill (`gc.rs malloc_zero_filled =
    /// false`), so anything else keeps whatever the address last held.  Upstream
    /// is safe because its `mutate_<field>` IS a GC pointer and
    /// `clear_gc_fields` therefore covers it, and one pointer-shaped word puts
    /// pyre on that same footing: `FUNCTION_DESCR_GROUP` declares it
    /// `Type::Ref`, so it joins `gc_fielddescrs` and the allocation clear NULLs
    /// it.  [`FUNCTION_GC_PTR_OFFSETS`] deliberately omits it, so the collector
    /// — whose census is that list, not the descriptor group — never follows
    /// the `Box` behind it.  Nine inline `QuasiImmutField`s had no such shape:
    /// each is `AtomicPtr` + `parking_lot::Mutex`, 144 bytes with no
    /// descriptor, and nothing initialised them.
    ///
    /// Null means no read has been recorded against this function yet, which is
    /// exactly `quasiimmut.py`'s null `mutate_<name>`: nothing folded the field,
    /// so a write revokes nothing.  The block is created on first install and
    /// freed by [`function_destructor`] on sweep.
    pub mutate_slots: AtomicPtr<FunctionQuasiImmutSlots>,
    /// `function.py` `make_weakref_descr(Function)` `_lifeline_`.
    /// `builtin_function_or_method` shares this payload.
    pub lifeline: PyObjectRef,
}

/// One `QuasiImmutField` per `function.py:34-42` `?` entry.
///
/// Upstream's rtyper makes one hidden field per declared field, so mutating
/// `__defaults__` revokes only the loops that folded `defs_w`; collapsing them
/// onto a shared slot would still be sound but would over-invalidate.
#[repr(C)]
pub struct FunctionQuasiImmutSlots([QuasiImmutField; QUASI_IMMUT_SLOT_COUNT]);

impl FunctionQuasiImmutSlots {
    fn new() -> Self {
        // Not `[const { QuasiImmutField::new() }; N]`: Charon rejects the
        // pointer-null Cast that const-repeat lowers. Length is checked
        // against [`QUASI_IMMUT_SLOT_COUNT`].
        Self([
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
            QuasiImmutField::new(),
        ])
    }
}

/// Number of `?` entries in `function.py:34-42 _immutable_fields_`.
///
/// Anchored on the highest-numbered [`QuasiImmutSlot`] variant, so an entry
/// added at the end of that enum extends the bound rather than needing the
/// count restated here.
pub const QUASI_IMMUT_SLOT_COUNT: usize = (QuasiImmutSlot::WKwDefs as usize) + 1;

/// Which `_immutable_fields_` `?` entry a quasi-immutable operation names.
///
/// `quasiimmut.py` keys everything off the mutate-field *name*; pyre needs a
/// value it can pass through the descr-index dispatch in
/// `pyre-jit-trace state.rs` / `pyre-jit eval.rs`, so the name becomes a tag.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuasiImmutSlot {
    Code,
    WFuncGlobalsObj,
    Closure,
    DefsW,
    Name,
    WQualname,
    WObjclass,
    WTextSignature,
    WKwDefs,
}

impl QuasiImmutSlot {
    /// Position in [`FunctionQuasiImmutSlots`]. The discriminant order is the
    /// `_immutable_fields_` order, so `as usize` is the index.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// The `mutate_<name>` field for one `?` entry, or `None` when `obj` does not
/// carry the `Function` layout the slot lives in.
///
/// Reaching a slot casts `obj` to `*const Function`, so the carrier test is the
/// precondition for the cast, not a convenience. `is_function_carrier` is the
/// right predicate rather than a bare `FUNCTION_TYPE` comparison: the four
/// types `function_new_impl` stamps — `function`, `builtin_function_or_method`,
/// `method_descriptor`, `wrapper_descriptor` — all allocate a real `Function`
/// whose slots are initialised, and skipping any of them would leave a write
/// unannounced. Under-invalidation is the unsound direction; over-invalidation
/// is not (`quasiimmut.py`).
///
/// Answers `None` while [`Function::mutate_slots`] is still null, which is the
/// null `mutate_<name>` of `quasiimmut.py`: nothing has folded the field, so
/// there is nothing to revoke. Only [`function_current_qmut_instance`] creates
/// the block.
///
/// # Safety
/// `obj` must be null or point at a live `PyObject`.
pub unsafe fn function_quasi_immut_field<'a>(
    obj: PyObjectRef,
    slot: QuasiImmutSlot,
) -> Option<&'a QuasiImmutField> {
    if obj.is_null() || !unsafe { is_function_carrier(obj) } {
        return None;
    }
    let f = obj as *const Function;
    let slots = unsafe { (*f).mutate_slots.load(Ordering::Acquire) };
    if slots.is_null() {
        return None;
    }
    Some(unsafe { &(*slots).0[slot.index()] })
}

/// `quasiimmut.py get_current_qmut_instance` — resolve the instance
/// while the read is still being recorded, so a write reached later in the same
/// trace sees a non-null mutate field and aborts the attempt, and the recording
/// can carry the instance to `compile.py:204-207`.
///
/// Creates the slot block on first use. The losing racer frees its own
/// allocation rather than the published one, so the winner's block is the only
/// one any watcher ever reaches.
///
/// # Safety
/// `obj` must be null or point at a live `PyObject`.
pub unsafe fn function_current_qmut_instance(
    obj: PyObjectRef,
    slot: QuasiImmutSlot,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if obj.is_null() || !unsafe { is_function_carrier(obj) } {
        return None;
    }
    let f = obj as *const Function;
    let cell = unsafe { &(*f).mutate_slots };
    let mut slots = cell.load(Ordering::Acquire);
    if slots.is_null() {
        let fresh = Box::into_raw(Box::new(FunctionQuasiImmutSlots::new()));
        match cell.compare_exchange(
            std::ptr::null_mut(),
            fresh,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => slots = fresh,
            Err(published) => {
                drop(unsafe { Box::from_raw(fresh) });
                slots = published;
            }
        }
    }
    Some(unsafe { &(*slots).0[slot.index()] }.get_current_qmut_instance())
}

/// `quasiimmut.py make_invalidation_function._invalidate_now` — revoke
/// every loop that folded this field. Runs BEFORE the store, the way
/// `typeobject.rs w_type_set_version_tag` calls its notify first.
///
/// # Safety
/// `obj` must be null or point at a live `PyObject`.
pub unsafe fn function_notify_quasi_immut(obj: PyObjectRef, slot: QuasiImmutSlot) {
    if let Some(field) = unsafe { function_quasi_immut_field(obj, slot) } {
        field.invalidate();
    }
}

/// Drop glue for the `mutate_<name>` block, registered on
/// `FUNCTION_GC_TYPE_ID`.
///
/// Touches only that block. The `name` box is its own allocation — a GC
/// storage box when the collector owns the function, `malloc_raw` on the
/// pre-hook fallback — reclaimed by that box's drop glue when it is managed,
/// so this must not reach it. A holder destructor that did would double-free
/// a box swept before its owner.
///
/// Reads the pointer with `swap`, so a `Function` reached twice frees once. The
/// word is null on every function that no trace ever read a `?` field off,
/// including one `emit_make_function_inline` allocated, which is what makes this
/// safe to run over a JIT-built object.
///
/// # Safety
/// `addr` must point at a live `Function` the collector is reclaiming, exactly
/// once.
pub unsafe fn function_destructor(addr: usize) {
    let f = addr as *mut Function;
    let slots = unsafe {
        (*f).mutate_slots
            .swap(std::ptr::null_mut(), Ordering::AcqRel)
    };
    if !slots.is_null() {
        drop(unsafe { Box::from_raw(slots) });
    }
}

/// function.py — `class BuiltinFunction(Function): can_change_code = False`
pub type BuiltinFunction = Function;
/// function.py — `class FunctionWithFixedCode(Function): can_change_code = False`
pub type FunctionWithFixedCode = Function;
pub type Method = pyre_object::function::Method;
pub type StaticMethod = pyre_object::function::StaticMethod;
pub type ClassMethod = pyre_object::function::ClassMethod;

#[inline]
fn function_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
    // A Box-immortal function's children are reached only through
    // `walk_raw_function_roots`, which clean minor collections skip;
    // record every field store (gc_roots.rs prebuilt-root tracking).
    // RPython's GC transform inserts the old-to-young `write_barrier`
    // (minimark.py) after such post-alloc field stores; pyre has no
    // transform pass, so callers run it by hand through this helper.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
}

/// Field offset of `code` within `Function`, for JIT field access.
pub const FUNCTION_CODE_OFFSET: usize = std::mem::offset_of!(Function, code);
/// Field offset of `name` within `Function`.
pub const FUNCTION_NAME_OFFSET: usize = std::mem::offset_of!(Function, name);
/// Field offset of wrapped `__name__` within `Function`.
pub const FUNCTION_W_NAME_OFFSET: usize = std::mem::offset_of!(Function, w_name);
/// Field offset of `closure` within `Function`.
pub const FUNCTION_CLOSURE_OFFSET: usize = std::mem::offset_of!(Function, closure);
/// Field offset of `defs_w` within `Function`.
pub const FUNCTION_DEFS_W_OFFSET: usize = std::mem::offset_of!(Function, defs_w);
/// Field offset of `w_kw_defs` within `Function`.
pub const FUNCTION_W_KW_DEFS_OFFSET: usize = std::mem::offset_of!(Function, w_kw_defs);
/// Field offset of `w_module` within `Function`.
pub const FUNCTION_W_MODULE_OFFSET: usize = std::mem::offset_of!(Function, w_module);
/// Field offset of `w_func_globals_obj` within `Function` — the
/// lazy-cached canonical W_DictObject for `w_func_globals`.
pub const FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET: usize =
    std::mem::offset_of!(Function, w_func_globals_obj);
/// Field offset of CPython 3.14 `func_builtins`.
pub const FUNCTION_W_BUILTINS_OFFSET: usize = std::mem::offset_of!(Function, w_builtins);
/// Field offset of `w_ann` within `Function` — the
/// `function.py:50 w_ann` annotations dict slot.
pub const FUNCTION_W_ANN_OFFSET: usize = std::mem::offset_of!(Function, w_ann);
/// Field offset of `w_annotate` within `Function` — the PEP 649
/// `__annotate__` callable slot.
pub const FUNCTION_W_ANNOTATE_OFFSET: usize = std::mem::offset_of!(Function, w_annotate);
/// Field offset of PyPy `Function.w_func_dict`.
pub const FUNCTION_W_FUNC_DICT_OFFSET: usize = std::mem::offset_of!(Function, w_func_dict);
/// Field offset of CPython 3.14 `func_typeparams`.
pub const FUNCTION_W_TYPEPARAMS_OFFSET: usize = std::mem::offset_of!(Function, w_typeparams);
/// Field offset of `w_doc` within `Function` — the
/// `function.py:375 w_doc` docstring cache slot.
pub const FUNCTION_W_DOC_OFFSET: usize = std::mem::offset_of!(Function, w_doc);
/// Field offset of `w_qualname` within `Function` — the
/// `function.py:54 qualname` slot.
pub const FUNCTION_W_QUALNAME_OFFSET: usize = std::mem::offset_of!(Function, w_qualname);
/// Field offset of `w_objclass` within `Function` — the
/// `function.py:498-504 w_objclass` slot.
pub const FUNCTION_W_OBJCLASS_OFFSET: usize = std::mem::offset_of!(Function, w_objclass);
/// Field offset of `w_text_signature` within `Function` — the
/// `function.py:487-496 w_text_signature` slot.
pub const FUNCTION_W_TEXT_SIGNATURE_OFFSET: usize =
    std::mem::offset_of!(Function, w_text_signature);
/// Field offset of `w_new_self` within `Function` — the builtin
/// `__new__` descriptor's `__self__` (defining type).
pub const FUNCTION_W_NEW_SELF_OFFSET: usize = std::mem::offset_of!(Function, w_new_self);
/// Field offset of PyPy `BuiltinFunction.w_moduleobj`.
pub const FUNCTION_W_MODULEOBJ_OFFSET: usize = std::mem::offset_of!(Function, w_moduleobj);
/// Field offset of the `make_weakref_descr` `_lifeline_` slot.
pub const FUNCTION_LIFELINE_OFFSET: usize = std::mem::offset_of!(Function, lifeline);
/// Field offset of the `mutate_<name>` block pointer within `Function`.
/// Declared `Type::Ref` on `FUNCTION_DESCR_GROUP` so `clear_gc_fields` NULLs it
/// behind a JIT allocation; absent from [`FUNCTION_GC_PTR_OFFSETS`] so the
/// collector never follows the `Box` it addresses.
pub const FUNCTION_MUTATE_SLOTS_OFFSET: usize = std::mem::offset_of!(Function, mutate_slots);

/// GC type id assigned to `Function` at JitDriver init time. Held as
/// a constant alongside the struct (rather than runtime-queried) so
/// the allocation hook can reach it without a back-channel, mirroring
/// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` / `BUILTIN_CODE_GC_TYPE_ID`.
/// `pyre/pyre-jit/src/eval.rs` asserts the same id is returned by
/// `gc.register_type(...)` so any drift panics on startup.
///
/// `BuiltinFunction` (gateway.py module-level builtins) and
/// `FunctionWithFixedCode` are Rust type aliases of `Function`, so
/// instances of those PyTypes share this tid via the
/// `register_vtable_for_type` mapping.
pub const FUNCTION_GC_TYPE_ID: u32 = 14;

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`).
pub const FUNCTION_OBJECT_SIZE: usize = std::mem::size_of::<Function>();

/// Byte offsets of the inline `PyObjectRef`-shaped fields the GC must
/// trace during minor collection. `code` is included because
/// `BuiltinCode` is now allocated through `malloc_typed`
/// (`gateway.rs`'s `builtin_code_new_full`) and therefore lives in the GC
/// heap; `PyCode` is likewise a normal movable GC object, so tracing this slot
/// is what forwards it when a minor collection moves it.
/// `function.py:47 _immutable_fields_ = ['code?', ...]` matches PyPy's
/// Function.code? — an immutable GC reference traced as part of the
/// closure / defs_w / w_kw_defs / w_module set.
///
/// `can_change_code` is a `bool` and thus non-GC. `name` points at a
/// GC-managed leaf storage box (`NameStorage`) whenever the collector owns
/// the function, builtin or user, so its slot is forwarded here; the box
/// tid's drop glue reclaims it on sweep. A function minted before the
/// collector hook exists keeps a `malloc_raw` name, which the walker's
/// `is_managed_heap_object` guard skips (the same immortal-holder skip as
/// W_UnicodeObject's `value`).
///
/// `ob.w_class` is intentionally absent, mirroring how W_IntObject /
/// W_FloatObject leave the typeptr-shaped header field out of their
/// `gc_ptr_offsets`. W_TypeObject instances are static-region and
/// not subject to nursery relocation.
pub const FUNCTION_GC_PTR_OFFSETS: [usize; 19] = [
    FUNCTION_CODE_OFFSET,
    // `name` — GC-managed `NameStorage` box when the collector owns the function
    // (skipped by the walker's managed-object guard for a pre-hook `malloc_raw` name).
    FUNCTION_NAME_OFFSET,
    // `function.py:51 self.name` — wrapped `__name__` stamped at construction
    // or via `f.__name__ = ...`.
    FUNCTION_W_NAME_OFFSET,
    FUNCTION_CLOSURE_OFFSET,
    FUNCTION_DEFS_W_OFFSET,
    FUNCTION_W_KW_DEFS_OFFSET,
    FUNCTION_W_MODULE_OFFSET,
    // `function.py:57 w_func_globals` — the module namespace dict object,
    // the function's sole globals carrier; traced for the lifetime of the
    // function so its `__globals__` identity survives minor collection.
    FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET,
    // CPython 3.14 `func_builtins` — frozen at Function construction.
    FUNCTION_W_BUILTINS_OFFSET,
    // `function.py:50 w_ann` — annotations dict, allocated lazily on
    // first read by the getter or stamped at MAKE_FUNCTION time.
    FUNCTION_W_ANN_OFFSET,
    // PEP 649 `__annotate__` callable stamped by MAKE_FUNCTION's
    // Annotate flag; live until the first `__annotations__` read
    // materialises `w_ann`.
    FUNCTION_W_ANNOTATE_OFFSET,
    // `function.py:68 w_func_dict` — lazily allocated instance dict.
    FUNCTION_W_FUNC_DICT_OFFSET,
    // CPython 3.14 `func_typeparams` — tuple or null for the empty default.
    FUNCTION_W_TYPEPARAMS_OFFSET,
    // `function.py:375 w_doc` — docstring slot cached on first read.
    FUNCTION_W_DOC_OFFSET,
    // `function.py:54 qualname` — qualified name slot stamped at
    // construction or via `f.__qualname__ = ...`.
    FUNCTION_W_QUALNAME_OFFSET,
    // `function.py:498-504 w_objclass` — bound class for descriptor
    // introspection (`inspect.getfullargspec` etc.).
    FUNCTION_W_OBJCLASS_OFFSET,
    // `function.py:487-496 w_text_signature` — PEP 437 text signature
    // line stripped from the docstring.
    FUNCTION_W_TEXT_SIGNATURE_OFFSET,
    // `w_new_self` is intentionally absent: it holds the defining type
    // of a builtin `__new__` carrier, a static-region W_TypeObject that
    // is never nursery-relocated (same reasoning as `ob.w_class`).
    // PyPy `BuiltinFunction.w_moduleobj` is an ordinary GC module reference.
    FUNCTION_W_MODULEOBJ_OFFSET,
    FUNCTION_LIFELINE_OFFSET,
];

impl pyre_object::lltype::GcType for Function {
    fn type_id() -> u32 {
        FUNCTION_GC_TYPE_ID
    }
    const SIZE: usize = FUNCTION_OBJECT_SIZE;
}

/// Allocate a new `Function`.
///
/// `code` is a pointer to a Code object (PyCode) cast to `*const ()`.
/// `name` is the function name string stored in an off-GC Box.
/// `w_func_globals_obj` is the defining module's namespace dict object
/// (shared), or `PY_NULL` for a globals-less carrier.
pub fn function_new(code: *const (), name: String, w_func_globals_obj: PyObjectRef) -> PyObjectRef {
    function_new_with_closure(code, name, w_func_globals_obj, PY_NULL)
}

/// Allocate a new `Function` with a closure.
///
/// `pypy/interpreter/function.py Function.__init__` —
/// `self.w_func_globals = w_globals` stores the user-visible
/// `__globals__` dict object directly, so the function's `__globals__`
/// identity IS the supplied `PyObjectRef`.  `closure` is a tuple of cell
/// objects, or PY_NULL if no closure.
pub fn function_new_with_closure(
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
    closure: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(
        &FUNCTION_TYPE,
        code,
        FunctionName::Owned(name),
        w_func_globals_obj,
        closure,
        true,
    )
}

/// Where a new function's `name` slot points.
///
/// `function.py self.name = forcename or code.co_name` copies the code
/// object's own string; RPython strings are immutable and shared, so no second
/// allocation exists upstream.  Pyre's names live in explicit storage, so the
/// two shapes are named here: `Owned` boxes a string the caller built, while
/// `Borrowed` hands over a pointer whose storage outlives every function that
/// names it.
pub enum FunctionName {
    /// Box the string — a GC storage box for a mortal (`PyCode`) function, a
    /// `malloc_raw` box for an immortal builtin.
    Owned(String),
    /// Point at existing permanent storage: a `Box::into_raw`'d `CodeObject`'s
    /// `co_name`, which is never rewritten in place (`code.replace()` clones
    /// first) and is never freed.  The GC walker's managed-heap guard skips the
    /// edge for the same reason it skips an immortal builtin's `malloc_raw`
    /// name, and `function_set_func_name` replaces this pointer instead of
    /// writing through it, so several functions may share one string.
    Borrowed(*const String),
}

/// `pyopcode.py MAKE_FUNCTION` construction: both the name and the
/// qualified name come off the code object, so neither allocates.
///
/// `function.py:51-54`:
///
/// ```python
/// self.name = forcename or code.co_name
/// self.qualname = qualname or self.name
/// ```
///
/// `w_code` is the code wrapper the new function keeps in its `code` slot; it
/// owns the borrowed `co_name` and the shared realized `co_qualname`.
pub fn function_new_from_code(w_code: PyObjectRef, w_func_globals_obj: PyObjectRef) -> PyObjectRef {
    // `PyCode` is a normal movable `W_Root` in `pycode.py`. Keep both inputs
    // in the shadow stack across lazy name/qualname realization and Function
    // allocation, and reload the code pointer before every use.
    let roots = pyre_object::gc_roots::push_roots();
    let root_base = roots.publish(&[w_code, w_func_globals_obj]);
    roots.normalize(root_base, 2);
    let w_code = roots.get(root_base);
    let code_ptr = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
    let name = if code_ptr.is_null() {
        FunctionName::Owned(String::new())
    } else {
        FunctionName::Borrowed(unsafe { &(*code_ptr).obj_name } as *const String)
    };
    // `function.py:51 self.name = code.co_name`; realize it BEFORE the
    // function so the allocation that can collect runs while no unrooted
    // function is live.
    unsafe { crate::pycode::w_code_name_obj(w_code) };
    // `function.py self.qualname = qualname or self.name`.  The code object
    // realizes one wrapped qualname and every function built from it names that
    // object, so a later `__code__ = new_code` assignment leaves
    // `__qualname__` alone (`pyopcode.py:1457` stamps it at construction).
    // Realize it BEFORE the function so the one allocation that can collect
    // runs while no unrooted function is live.
    unsafe { crate::pycode::w_code_qualname_obj(roots.get(root_base)) };
    let w_code = roots.get(root_base);
    let func = function_new_impl(
        &FUNCTION_TYPE,
        w_code as *const (),
        name,
        roots.get(root_base + 1),
        PY_NULL,
        true,
    );
    // `pick_builtin_obj` inside the constructor is a collection point, so read
    // the realized values back through the forwarded code root rather than a
    // stale pre-call copy. These calls are plain cache hits.
    let w_name = unsafe { crate::pycode::w_code_name_obj(roots.get(root_base)) };
    if !w_name.is_null() {
        unsafe { function_set_name_obj(func, w_name) };
    }
    let w_qualname = unsafe { crate::pycode::w_code_qualname_obj(roots.get(root_base)) };
    if !w_qualname.is_null() {
        unsafe { function_set_qualname(func, w_qualname) };
    }
    func
}

/// Field block `function_new_impl` publishes for a new `Function`.
///
/// `Function.__init__` leaves the attribute slots null. `mutate_slots` is a
/// fresh null pointer — `quasiimmut` `mutate_<name>` stays null until the
/// first read is recorded — so a caller that wraps an existing function must
/// assign copied slots afterwards. Copying that function by value would share
/// its watcher block.
fn function_object_value(
    ob_type: &'static PyType,
    code: *const (),
    name: *const String,
    w_func_globals_obj: PyObjectRef,
    closure: PyObjectRef,
    w_builtins: PyObjectRef,
    can_change_code: bool,
) -> Function {
    Function {
        ob: PyObject {
            ob_type: ob_type as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(ob_type),
        },
        code,
        can_change_code,
        name,
        w_name: PY_NULL,
        closure,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
        w_module: PY_NULL,
        w_func_globals_obj,
        w_builtins,
        w_ann: PY_NULL,
        w_annotate: PY_NULL,
        w_func_dict: PY_NULL,
        w_typeparams: PY_NULL,
        w_doc: PY_NULL,
        w_qualname: PY_NULL,
        w_objclass: PY_NULL,
        w_text_signature: PY_NULL,
        w_new_self: PY_NULL,
        w_moduleobj: PY_NULL,
        // quasiimmut `mutate_<name>` — null until the first read is recorded.
        mutate_slots: AtomicPtr::new(std::ptr::null_mut()),
        lifeline: PY_NULL,
    }
}

/// Allocate a `Function` object.
///
/// `function.py` `class Function(W_Root)` — one GC object for builtin and
/// user code. `try_gc_alloc_stable_raw` is that allocation once the collector
/// hook is installed; `lltype::malloc_typed` is only the pre-hook fallback.
/// The tracer cannot model either call, so the JIT residualises this function
/// (`@dont_look_inside`, `rlib/jit.py`), the `box_str_constant` /
/// `try_gc_add_root` twin.
#[majit_macros::dont_look_inside]
pub(crate) fn function_new_impl(
    ob_type: &'static PyType,
    code: *const (),
    name: FunctionName,
    w_func_globals_obj: PyObjectRef,
    closure: PyObjectRef,
    can_change_code: bool,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`) for
    // the `lltype::malloc_typed` call below. `closure`, `code`, and
    // `w_func_globals_obj` are PyObjectRef-shaped GC roots across the
    // alloc — `BuiltinCode` lives in the GC heap (`gateway.rs`'s
    // `builtin_code_new_full` malloc_typed) and `PyCode` is movable; the
    // walker's `is_in_nursery` filter (`majit-gc/src/collector.rs`)
    // is what makes the heterogeneous case safe. `name_ptr` is allocated
    // below via `malloc_raw` (non-GC) and stored into the struct as
    // part of the same `malloc_typed` call, so it never spans a
    // collection point.
    let _roots = pyre_object::gc_roots::push_roots();
    let closure_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(closure);
    let code_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(code as PyObjectRef);
    // `function.py self.w_func_globals = w_globals` stores the dict
    // object directly as the function's sole globals carrier.
    let globals_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_func_globals_obj = pyre_object::gc_roots::pin_root(w_func_globals_obj);
    // CPython 3.14 `_PyDict_LoadBuiltinsFromGlobals` at function construction:
    // retain the selected mapping identity even if the globals entry changes
    // later.  Its `_Py_dict_lookup_threadsafe_stackref` reads the native dict
    // backing directly (including for a dict subclass), and a missing key
    // inherits `PyEval_GetBuiltins()` from the live caller frame.  This differs
    // from PyPy's `pick_builtin`, whose missing-key result is a fresh anonymous
    // module containing only `None`; 3.14 is pyre's compatibility target for
    // the observable `function.__builtins__` field.
    //
    // Builtin/gateway carriers have no globals and keep a null slot.
    let w_builtins = if w_func_globals_obj.is_null() {
        PY_NULL
    } else {
        let globals_backing = crate::type_methods::resolve_dict_backing(w_func_globals_obj);
        let selected = if globals_backing.is_null() {
            None
        } else {
            unsafe { pyre_object::w_dict_getitem_str(globals_backing, "__builtins__") }
        }
        .unwrap_or_else(|| {
            let exec_ctx = crate::call::take_last_exec_ctx();
            if exec_ctx.is_null() {
                return PY_NULL;
            }
            let frame = unsafe { (*exec_ctx).gettopframe_raw() };
            if frame.is_null() {
                unsafe { (*exec_ctx).get_builtin_dict() }
            } else {
                unsafe { (*frame).fget_f_builtins() }
            }
        });
        if !selected.is_null() && unsafe { pyre_object::is_module(selected) } {
            unsafe { pyre_object::w_module_get_w_dict(selected) }
        } else {
            selected
        }
    };
    let builtins_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_builtins);
    // Resolving the caller vref and later allocations may collect.  Reload every
    // pinned input before embedding it in the new Function; the original raw
    // locals are not rewritten when the shadow-stack slots are forwarded.
    let closure = pyre_object::gc_roots::shadow_stack_get(closure_slot);
    let code = pyre_object::gc_roots::shadow_stack_get(code_slot) as *const ();
    let w_func_globals_obj = pyre_object::gc_roots::shadow_stack_get(globals_slot);
    let w_builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);

    // `function.py` `Function` is one GC object whether or not its code is a
    // `BuiltinCode`. `try_gc_alloc_stable_raw` returns null only when no
    // collector hook is installed (`gc_hook.rs`); that call does not collect,
    // so the name box below can be chosen from the result and stored before
    // anything else runs. A managed carrier boxes the name in `NameStorage`,
    // greyed through `FUNCTION_NAME_OFFSET`. The pre-hook carrier is immortal
    // and keeps a `malloc_raw` name: nothing will ever grey it.
    let raw =
        pyre_object::gc_hook::try_gc_alloc_stable_raw(FUNCTION_GC_TYPE_ID, FUNCTION_OBJECT_SIZE);
    let managed = !raw.is_null();
    let name_ptr = match name {
        // Already-permanent storage owned by the code object; nothing to box.
        FunctionName::Borrowed(ptr) => ptr,
        FunctionName::Owned(name) if !managed => {
            pyre_object::lltype::malloc_raw(name) as *const String
        }
        FunctionName::Owned(name) => pyre_object::gc_storage::gc_alloc_storage_box(
            name,
            pyre_object::typeobject::name_storage_gc_type_id(),
        ) as *const String,
    };
    let function = function_object_value(
        ob_type,
        code,
        name_ptr,
        w_func_globals_obj,
        closure,
        w_builtins,
        can_change_code,
    );

    // `function.py` `class Function(W_Root)`: builtin and user functions are
    // the same GC object. The stable hook is old-gen and non-moving, and the
    // builtin-type / module-dict root walks mark the pointer, so a major
    // traces `FUNCTION_GC_PTR_OFFSETS` — including `w_module` stamped later
    // by `builtin_function_set_module`.
    if managed {
        unsafe {
            std::ptr::write(raw as *mut Function, function);
        }
        // The fields were published by the `ptr::write` above, which no
        // barrier sees. An old-gen Function that never joins the
        // remembered set is not scanned by a minor collection, so a
        // freshly nursery-born `w_func_globals_obj` reached only through
        // it is neither forwarded nor kept alive. The setters barrier
        // just ahead of their single store; a bulk write has no such
        // point, so barrier once behind it for the whole initial set.
        // Nothing collectable runs in between.
        function_write_barrier(raw as PyObjectRef);
        return raw as PyObjectRef;
    }

    // Immortal holder with the prebuilt header (`init_gc_object_immortal`):
    // every later field store (`__name__`, `__qualname__`, `__module__`,
    // `__doc__` materialised on first read, the JIT's `SETFIELD_GC`) goes
    // through a barrier on this box, and the barrier just below covers the
    // construction-time fields. `remember_young_pointer` then puts the box in
    // `old_objects_pointing_to_young` and `prebuilt_root_objects`, so the
    // collector traces it by `FUNCTION_GC_TYPE_ID` whichever holder reaches
    // it (`incminimark.py` `prebuilt_root_objects`).
    let obj = pyre_object::lltype::malloc_typed_immortal(function) as PyObjectRef;
    function_write_barrier(obj);
    obj
}

/// function.py — `class FunctionWithFixedCode(Function): can_change_code = False`
/// Allocate a function whose code pointer the JIT can treat as immutable.
pub fn function_new_with_fixed_code(
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(
        &FUNCTION_TYPE,
        code,
        FunctionName::Owned(name),
        w_func_globals_obj,
        PY_NULL,
        false,
    )
}

/// Expose a type-owned `FunctionWithFixedCode` as a method descriptor.
///
/// PyPy performs the ownership step in `TypeCache.build`, where it finds
/// `FunctionWithFixedCode` values in a `TypeDef` and stamps their
/// `objclass`/qualified name.  A fixed-code function can also be a standalone
/// callable (for example `itertools.chain`), so allocation alone cannot decide
/// its public descriptor kind.  Retag it only when that same TypeCache step
/// establishes a defining type.
///
/// The payload and GC layout are unchanged: `METHOD_DESCRIPTOR_TYPE` is an
/// alias of the `Function` GC type, just as `SLOT_WRAPPER_TYPE` is.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_mark_method_descriptor(function: PyObjectRef) {
    unsafe {
        (*function).ob_type = &METHOD_DESCRIPTOR_TYPE;
        (*function).w_class = pyre_object::get_instantiate(&METHOD_DESCRIPTOR_TYPE);
    }
}

/// function.py — `class BuiltinFunction(Function): can_change_code = False`
/// Allocate a module builtin whose typedef intentionally omits `__get__`.
pub fn function_new_builtin(
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(
        &BUILTIN_FUNCTION_TYPE,
        code,
        FunctionName::Owned(name),
        w_func_globals_obj,
        PY_NULL,
        false,
    )
}

/// Allocate an immutable slot-wrapper descriptor backed by `BuiltinCode`.
pub fn function_new_slot_wrapper(code: *const (), name: String) -> PyObjectRef {
    function_new_impl(
        &SLOT_WRAPPER_TYPE,
        code,
        FunctionName::Owned(name),
        PY_NULL,
        PY_NULL,
        false,
    )
}

/// Retag a freshly materialised `FunctionWithFixedCode` as the public
/// method-descriptor carrier. TypeDef builds its namespace before the owner
/// type exists, so this is deliberately performed during materialisation.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_retag_method_descriptor(obj: PyObjectRef) {
    unsafe {
        (*obj).ob_type = &METHOD_DESCRIPTOR_TYPE as *const PyType;
        (*obj).w_class = get_instantiate(&METHOD_DESCRIPTOR_TYPE);
    }
}

/// Retag a freshly materialised `FunctionWithFixedCode` as the public
/// slot-wrapper carrier — the `add_operators` half of the same namespace sweep
/// [`function_retag_method_descriptor`] serves.
///
/// # Safety
/// `obj` must be a valid, live `FunctionWithFixedCode` carrier.
pub unsafe fn function_retag_slot_wrapper(obj: PyObjectRef) {
    unsafe {
        (*obj).ob_type = &SLOT_WRAPPER_TYPE as *const PyType;
        (*obj).w_class = get_instantiate(&SLOT_WRAPPER_TYPE);
    }
}

/// Allocate an immutable ordinary method descriptor backed by `BuiltinCode`.
pub fn function_new_method_descriptor(code: *const (), name: String) -> PyObjectRef {
    function_new_impl(
        &METHOD_DESCRIPTOR_TYPE,
        code,
        FunctionName::Owned(name),
        PY_NULL,
        PY_NULL,
        false,
    )
}

/// function.py — `_check_code_mutable(attr)`:
///
/// ```python
/// def _check_code_mutable(self, attr):
///     if not self.can_change_code:
///         raise oefmt(self.space.w_AttributeError,
///                     "Cannot change %s attribute of builtin functions", attr)
/// ```
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn _check_code_mutable(func: PyObjectRef, attr: &str) -> Result<(), crate::PyError> {
    unsafe {
        if (*(func as *const Function)).can_change_code {
            Ok(())
        } else {
            Err(crate::PyError::attribute_error(format!(
                "Cannot change {} attribute of builtin functions",
                attr
            )))
        }
    }
}

/// function.py — `@jit.elidable_promote()`
/// Only valid when `can_change_code == false`.
#[majit_macros::elidable_promote]
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn _get_immutable_code(func: PyObjectRef) -> PyObjectRef {
    // function.py:25
    debug_assert!(
        !unsafe { (*(func as *const Function)).can_change_code },
        "_get_immutable_code called on function with can_change_code=true"
    );
    unsafe { (*(func as *const Function)).code as PyObjectRef }
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &FUNCTION_TYPE)
            || py_type_check(obj, &BUILTIN_FUNCTION_TYPE)
            || py_type_check(obj, &METHOD_DESCRIPTOR_TYPE)
    }
}

/// Whether `obj` is a CPython-compatible slot-wrapper descriptor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_slot_wrapper(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SLOT_WRAPPER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_method_descriptor(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &METHOD_DESCRIPTOR_TYPE) }
}

/// PyPy `_Method(w_function, w_instance)` binding, restamped so the binding
/// publishes `public_class` instead of the `Method` typedef and carries
/// `m_module = None`.
///
/// The two `get` arms that restamp differ only in `public_class`, so both go
/// through here.  The tracer's `super` fold reproduces this call with the same
/// argument once it has chosen the class, which keeps the value it records and
/// the value the emitted stores build identical by construction.
pub fn restamped_bound_method_new(
    descr: PyObjectRef,
    obj: PyObjectRef,
    w_class: PyObjectRef,
    public_class: PyObjectRef,
) -> PyObjectRef {
    let method = pyre_object::w_method_new(descr, obj, w_class);
    unsafe {
        pyre_object::function::w_method_set_public_class(method, public_class);
        pyre_object::function::w_method_set_module(method, pyre_object::w_none());
    }
    method
}

/// PyPy `_Method(w_function, w_instance)` binding with CPython 3.14's
/// `builtin_function_or_method` public surface and `m_module = None`.
pub fn builtin_bound_method_new(
    descr: PyObjectRef,
    obj: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    restamped_bound_method_new(
        descr,
        obj,
        w_class,
        crate::typedef::gettypeobject(&BUILTIN_FUNCTION_TYPE),
    )
}

/// `descrobject.c PyWrapper_New` — bind a slot wrapper to its receiver.
///
/// Same `Method` payload as [`builtin_bound_method_new`]; only the public class
/// differs, so `type(object().__str__)` reports `method-wrapper` while
/// `type([].append)` reports `builtin_function_or_method`.  `PyMethodWrapper_Type`
/// has no `__module__`, but the field is part of the shared payload, so it is
/// filled the same way and simply left unpublished by the typedef.
pub fn method_wrapper_new(
    descr: PyObjectRef,
    obj: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    restamped_bound_method_new(
        descr,
        obj,
        w_class,
        crate::typedef::gettypeobject(&METHOD_WRAPPER_TYPE),
    )
}

/// Publish a `ClassMethod` carrier built for a `METH_CLASS` entry as
/// [`CLASSMETHOD_DESCRIPTOR_TYPE`].
///
/// The `ob_type` stays `CLASSMETHOD_TYPE`, so `is_classmethod` and every
/// binding path that reads the payload keep working; only the type the
/// namespace publishes changes.
///
/// A namespace swept before the type object exists is retagged again by the
/// closing pass of `init_typeobjects`, so this is a no-op until then.
pub fn classmethod_retag_descriptor(obj: PyObjectRef) {
    let w_class = get_instantiate(&CLASSMETHOD_DESCRIPTOR_TYPE);
    if w_class.is_null() {
        return;
    }
    unsafe {
        pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
        (*obj).w_class = w_class;
    }
}

/// Whether `obj` is a `ClassMethod` carrier published as a
/// `classmethod_descriptor` rather than as an ordinary `classmethod`.
pub fn is_classmethod_descriptor(obj: PyObjectRef) -> bool {
    !obj.is_null()
        && unsafe { pyre_object::function::is_classmethod(obj) }
        && std::ptr::eq(
            unsafe { (*obj).w_class },
            get_instantiate(&CLASSMETHOD_DESCRIPTOR_TYPE),
        )
}

/// Function-layout carriers accepted by the interpreter call machinery.
///
/// Keep this separate from [`is_function`]: wrapper descriptors must not
/// expose Python-function metadata such as `__code__` or `__globals__`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_function_carrier(obj: PyObjectRef) -> bool {
    unsafe { is_function(obj) || is_slot_wrapper(obj) || is_method_descriptor(obj) }
}

/// Whether this Function carrier wraps an interp-level BuiltinCode rather
/// than Python bytecode.  PyPy uses FunctionWithFixedCode for these objects,
/// but CPython-only function metadata such as PEP 649 `__annotate__` must not
/// leak onto their public surface.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_has_builtin_code(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let code = unsafe { (*(obj as *const Function)).code } as PyObjectRef;
    !code.is_null() && unsafe { crate::gateway::is_builtin_code(code) }
}

/// Stamp a module-level builtin function's `__module__` with its
/// containing module at install time — `MixedModule` registration sets
/// `func.w_module = w(modulename)` for each interp-level definition.
/// Touches `BUILTIN_FUNCTION_TYPE` objects (the `py_module!`
/// `inline_functions` / `functions` / `module_functions` carriers) and any
/// globals-less `FUNCTION_TYPE` object whose module is still unset.
/// App-level functions carry globals and derive `__module__` lazily from
/// `globals['__name__']`, so a function with globals is left alone.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
pub unsafe fn builtin_function_set_module(obj: PyObjectRef, w_module: PyObjectRef) {
    unsafe {
        let stampable = py_type_check(obj, &BUILTIN_FUNCTION_TYPE)
            || (py_type_check(obj, &FUNCTION_TYPE)
                && (*(obj as *const Function)).w_func_globals_obj.is_null());
        if stampable {
            let func = obj as *mut Function;
            if (*func).w_module.is_null() {
                function_write_barrier(obj);
                (*func).w_module = w_module;
            }
        }
    }
}

/// CPython 3.14 `PyCFunctionObject.m_module` member assignment. Unlike
/// PyPy's shared Function getset, builtin `__module__` is a writable direct
/// member and accepts any object; deletion stores `None`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn builtin_function_set_module_attr(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = value;
    }
}

/// mixedmodule.py:_load_lazily — "all functions that are directly in a
/// mixed-module are 'builtin', e.g. they get a special type without a
/// `__get__`".  A module dict entry that is a globals-less
/// `FunctionWithFixedCode` is retagged in place to `BuiltinFunction`
/// (`BUILTIN_FUNCTION_TYPE`, whose typedef omits `__get__`) so storing it on a
/// user class and reading it through an instance does not synthesize a bound
/// method (`typedef.py del BuiltinFunction.typedef.rawdict['__get__']`).
///
/// Both PyTypes share the `Function` layout and GC type id
/// (`eval.rs` maps both to `function_tid`), so only `ob_type` (the binding
/// dispatch tag read by `getdescriptor`) and `w_class` (the Python-visible
/// `type()`) are retagged — code, name, and every other field are preserved
/// and object identity is kept, so two dict keys aliasing one function stay
/// identical.  Callers pass module dict values only; builtin-type methods live
/// in type dicts, never reach this, and keep `Function.__get__` to bind as
/// methods.  App-level functions carry globals and are left alone.
///
/// A function carrying instance attributes (`w_func_dict` populated) is also
/// left alone: `BUILTIN_FUNCTION_TYPE` has no instance `__dict__` surface, so
/// retagging would hide those attributes.  `itertools.chain` is a function-
/// shaped constructor whose `from_iterable` alternate constructor is attached
/// as an attribute (`itertools::register_module`), so demoting it would drop
/// `chain.from_iterable` — kept binding here rather than lose the attribute.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
pub unsafe fn demote_module_function_to_builtin(obj: PyObjectRef) {
    unsafe {
        if py_type_check(obj, &FUNCTION_TYPE)
            && (*(obj as *const Function)).w_func_globals_obj.is_null()
            && (*(obj as *const Function)).w_func_dict.is_null()
        {
            (*obj).ob_type = &BUILTIN_FUNCTION_TYPE as *const PyType;
            (*obj).w_class = pyre_object::pyobject::get_instantiate(&BUILTIN_FUNCTION_TYPE);
        }
    }
}

/// GC-managed `BuiltinFunction.__init__` for `MixedModule._load_lazily`.
///
/// `_load_lazily` does not retag the module's function in place. It builds a
/// new `BuiltinFunction(func)` and stores that object in the module dict, so
/// the function is traced from the dict and dies with the module.
/// `BuiltinFunction.__init__` copies `code`, `w_func_globals`, `defs_w`,
/// `closure`, and `name` through `Function.__init__`, then copies `w_doc`,
/// `w_func_dict`, `w_module`, `w_kw_defs`, and `w_text_signature`. pyre also
/// copies `w_name`, the wrapped text kept beside `name`. `w_moduleobj` stays
/// null; the caller binds the module once it exists. `w_qualname` stays null,
/// the representation `function_new_impl` uses when a builtin's qualname is
/// derived from `name` (`_load_lazily` sets `qualname = name`).
///
/// The value is a fresh `Function`: `can_change_code` is false
/// (`BuiltinFunction`) and `mutate_slots` is a new null pointer. A bytewise
/// copy of `func` would share its quasi-immutable watcher block. `ob_type`
/// and `w_class` are tagged `BUILTIN_FUNCTION_TYPE` the same way
/// `demote_module_function_to_builtin` tags them. Allocation is the
/// GC-managed arm of `function_new_impl` (`try_gc_alloc_stable_raw` of
/// `FUNCTION_GC_TYPE_ID` / `FUNCTION_OBJECT_SIZE`, then `ptr::write` and
/// `function_write_barrier`). A null result means no GC hook is installed;
/// the call then falls back to `demote_module_function_to_builtin` and
/// returns `func`.
///
/// # Safety
/// `func` must be a live `Function` carrier (`FUNCTION_TYPE` or
/// `BUILTIN_FUNCTION_TYPE`).
#[majit_macros::dont_look_inside]
pub unsafe fn builtin_function_new_managed(func: PyObjectRef) -> PyObjectRef {
    // `try_gc_alloc_stable_raw` does not collect, so the copied slots stay
    // valid across it. End the source borrow before that allocation.
    let src = unsafe { &*(func as *const Function) };
    let code = src.code;
    let name = src.name;
    let w_name = src.w_name;
    let closure = src.closure;
    let defs_w = src.defs_w;
    let w_func_globals_obj = src.w_func_globals_obj;
    let w_doc = src.w_doc;
    let w_func_dict = src.w_func_dict;
    let w_module = src.w_module;
    let w_kw_defs = src.w_kw_defs;
    let w_text_signature = src.w_text_signature;
    let mut function = function_object_value(
        &BUILTIN_FUNCTION_TYPE,
        code,
        name,
        w_func_globals_obj,
        closure,
        PY_NULL,
        false,
    );
    function.w_name = w_name;
    function.defs_w = defs_w;
    function.w_doc = w_doc;
    function.w_func_dict = w_func_dict;
    function.w_module = w_module;
    function.w_kw_defs = w_kw_defs;
    function.w_text_signature = w_text_signature;

    let raw =
        pyre_object::gc_hook::try_gc_alloc_stable_raw(FUNCTION_GC_TYPE_ID, FUNCTION_OBJECT_SIZE);
    if raw.is_null() {
        unsafe { demote_module_function_to_builtin(func) };
        return func;
    }
    unsafe {
        std::ptr::write(raw as *mut Function, function);
    }
    // Same bulk-write barrier as the managed arm of `function_new_impl`:
    // `ptr::write` publishes every slot before any setter runs.
    function_write_barrier(raw as PyObjectRef);
    raw as PyObjectRef
}

/// `mixedmodule.py _load_lazily` after `BuiltinFunction(func)`:
/// `w_module` is the module name, and `name` / `qualname` are the name
/// the module dict stores. `w_moduleobj` stays null until the module
/// object exists (`builtin_function_set_module_obj`).
///
/// Install runs once per name, so the `_builtinversion_` cache is not kept.
/// A failed managed allocation returns `func` unchanged (`demote` does not
/// retag a function that still has globals).
///
/// # Safety
/// `func` must be a live plain `Function`. `w_module` must be the module-name
/// string, kept alive by the caller for the duration of the call.
pub unsafe fn builtin_function_from_function(
    func: PyObjectRef,
    name: &str,
    w_module: PyObjectRef,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let func_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(func);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_module);
    let bltin = builtin_function_new_managed(pyre_object::gc_roots::shadow_stack_get(func_slot));
    if !py_type_check(bltin, &BUILTIN_FUNCTION_TYPE) {
        return pyre_object::gc_roots::shadow_stack_get(func_slot);
    }
    let bltin_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(bltin);
    let w_name = pyre_object::w_str_new_managed(name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_name);
    function_set_func_name(
        pyre_object::gc_roots::shadow_stack_get(bltin_slot),
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    );
    function_set_qualname(
        pyre_object::gc_roots::shadow_stack_get(bltin_slot),
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    );
    builtin_function_set_module_attr(
        pyre_object::gc_roots::shadow_stack_get(bltin_slot),
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    );
    pyre_object::gc_roots::shadow_stack_get(bltin_slot)
}

/// Stamp the `__self__` of a builtin `__new__` carrier — the defining
/// type whose `tp_new` it wraps (`typeobject.c add_tp_new_wrapper`).
/// Only touches functions whose `w_new_self` is still unset, so an
/// inherited `__new__` keeps the ancestor that defined it.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `Function`.
pub unsafe fn function_set_new_self(obj: PyObjectRef, w_type: PyObjectRef) {
    unsafe {
        let func = obj as *mut Function;
        if (*func).w_new_self.is_null() {
            function_write_barrier(obj);
            (*func).w_new_self = w_type;
        }
    }
}

/// `__self__` getter for the builtin-function type.  Returns the
/// stamped `w_new_self` (the defining type) for a builtin `__new__`
/// carrier, else `None` — `typedef.py GetSetProperty(always_none,
/// cls=BuiltinFunction)` returns `None` for an ordinary builtin.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `Function`.
pub unsafe fn function_get_self_or_none(obj: PyObjectRef) -> PyObjectRef {
    let func = unsafe { &*(obj as *const Function) };
    let w_self = if !func.w_moduleobj.is_null() {
        func.w_moduleobj
    } else {
        func.w_new_self
    };
    if w_self.is_null() {
        pyre_object::w_none()
    } else {
        w_self
    }
}

/// `meth_repr` — the receiver decides the wording, not how the carrier is
/// spelled: a null or module `m_self` reports as a plain function, anything
/// else as a method of that receiver.  A `__new__` entry carries its owning
/// type, so it reads `<built-in method __new__ of type object at 0x...>`.
///
/// # Safety
/// `w_self` must be null or a valid object pointer.
pub unsafe fn builtin_function_repr_text(name: &str, w_self: PyObjectRef) -> String {
    let bound = !w_self.is_null()
        && !unsafe { pyre_object::is_none(w_self) }
        && !unsafe { pyre_object::is_module(w_self) };
    if !bound {
        return format!("<built-in function {name}>");
    }
    // `r#type` walks the class and can allocate; the bound receiver is the
    // address `getaddrstring` prints, so keep it live until that id is taken.
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_self = pyre_object::gc_roots::pin_root(w_self);
    let type_name = crate::typedef::r#type(w_self)
        .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()) })
        .unwrap_or("object")
        .to_string();
    let w_self = pyre_object::gc_roots::shadow_stack_get(self_slot);
    format!(
        "<built-in method {name} of {type_name} object at {}>",
        crate::display::repr_gc_addr(w_self)
    )
}

/// CPython 3.14 `meth_reduce`: type-bound builtins reconstruct through
/// `getattr(__self__, __name__)`; module-level builtins reduce by qualname.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_builtin_function_reduce(obj: PyObjectRef) -> crate::PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let w_self = unsafe { function_get_self_or_none(obj()) };
    if !w_self.is_null()
        && !unsafe { pyre_object::is_none(w_self) }
        && !unsafe { pyre_object::is_module(w_self) }
    {
        let self_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_self);
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(unsafe {
            crate::function_get_name(obj())
        }));
        let args = pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        ]);
        let args_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(args);
        let getattr_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(crate::baseobjspace::builtin_callable("getattr"));
        return Ok(pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(getattr_slot),
            pyre_object::gc_roots::shadow_stack_get(args_slot),
        ]));
    }
    Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
        function_get_qualname(obj())
    }))
}

/// CPython `methodobject.c:meth_reduce` / `descrobject.c` reduction shape for
/// an unbound builtin method or slot descriptor:
/// `(<builtins.getattr>, (__objclass__, __name__))`.
///
/// PyPy represents these descriptors as `FunctionWithFixedCode`; its
/// `w_objclass` field is stamped by `TypeCache.build`, so use that existing
/// owner instead of inventing a descriptor side table.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_fixed_code_reduce(obj: PyObjectRef) -> crate::PyResult {
    let owner = unsafe { fget_func_objclass(obj)? };
    let _roots = pyre_object::gc_roots::push_roots();
    let owner_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(owner);
    let name = pyre_object::w_str_new_managed(unsafe { function_get_name(obj) });
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    let getattr = crate::baseobjspace::builtin_callable("getattr");
    let getattr_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(getattr);
    let args = pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(owner_slot),
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    ]);
    let _ = pyre_object::gc_roots::pin_root(args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(getattr_slot),
        pyre_object::gc_roots::shadow_stack_get(args_slot),
    ]))
}

/// Stamp PyPy `BuiltinFunction.w_moduleobj` after the defining module object
/// has been created. The module name is installed first through
/// `builtin_function_set_module`; this second field preserves the distinct
/// `__module__` (string) / `__self__` (module object) identities.
///
/// # Safety
/// `obj` must be a valid PyObjectRef. Non-builtin functions are ignored.
pub unsafe fn builtin_function_set_module_obj(obj: PyObjectRef, w_module: PyObjectRef) {
    unsafe {
        if py_type_check(obj, &BUILTIN_FUNCTION_TYPE) {
            let func = obj as *mut Function;
            function_write_barrier(obj);
            (*func).w_moduleobj = w_module;
        }
    }
}

/// `isinstance(obj, FunctionWithFixedCode)` parity.
///
/// function.py — `class FunctionWithFixedCode(Function):
///     can_change_code = False`.  Pyre encodes the `Function`
/// vs `FunctionWithFixedCode` distinction through the
/// `can_change_code` flag (true for user `def`s built via
/// `function_new_with_closure`, false for gateway-built builtins
/// via `function_new_with_fixed_code`).  `BuiltinFunction` has its
/// own `BUILTIN_FUNCTION_TYPE`, so a strict `FUNCTION_TYPE` check
/// excludes it (function.py makes BuiltinFunction a sibling
/// subclass of Function, not a subclass of FunctionWithFixedCode).
/// `METHOD_DESCRIPTOR_TYPE` is the public retag a type-owned fixed-code
/// function receives in `function_mark_method_descriptor`; it does not
/// change what the object is, so it stays inside this check.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function_with_fixed_code(obj: PyObjectRef) -> bool {
    unsafe {
        (py_type_check(obj, &FUNCTION_TYPE) || py_type_check(obj, &METHOD_DESCRIPTOR_TYPE))
            && !(*(obj as *const Function)).can_change_code
    }
}

/// function.py — `getcode(self)`: three-way dispatch.
///   - JIT + immutable code → _get_immutable_code (elidable_promote)
///   - JIT + mutable code  → promote(self.code)
///   - interpreter         → self.code
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn getcode(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *const Function;
        if majit_metainterp::jit::we_are_jitted() {
            if !(*func).can_change_code {
                // function.py `Function.getcode`: the immutable-code arm.
                return _get_immutable_code(obj);
            }
            // function.py `Function.getcode`: `jit.promote(self.code)`.
            return majit_metainterp::jit::promote((*func).code) as PyObjectRef;
        }
        // function.py `Function.getcode`: the untraced `return self.code`.
        (*func).code as PyObjectRef
    }
}

/// Get the Code object pointer from a function object.
///
/// Returns a pointer to the Code-level object (PyCode or BuiltinCode).
///
/// # Safety
/// `obj` must point to a valid `Function`.
/// NOTE: NOT elidable — code field can change (can_change_code).
/// Use _get_immutable_code() for the elidable path.
#[inline]
pub unsafe fn function_get_code(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const Function)).code }
}

/// Extract the raw bytecode CodeObject pointer from a user function.
///
/// Equivalent to accessing `self.getcode().code_ptr` in PyPy terms:
/// `getcode()` returns the Code wrapper (PyCode), and this
/// dereferences through it to the underlying CodeObject.
///
/// # Safety
/// `obj` must point to a valid `Function` whose `code` field is a `PyCode`
/// (i.e., NOT a BuiltinCode). Only call on user-defined functions.
#[inline]
pub unsafe fn get_pycode(obj: PyObjectRef) -> *const () {
    unsafe {
        let code = getcode(obj);
        debug_assert!(
            !crate::is_builtin_code(code as PyObjectRef),
            "get_pycode called on a builtin function"
        );
        crate::w_code_get_ptr(code as PyObjectRef)
    }
}

/// Get the function name.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const Function)).name }
}

/// `function.py fset_func_qualname` parity:
///
/// ```python
/// def fset_func_qualname(self, space, w_name):
///     self._check_code_mutable("__qualname__")
///     try:
///         qualname = space.realutf8_w(w_name)
///     except OperationError as e:
///         if e.match(space, space.w_TypeError):
///             raise oefmt(space.w_TypeError,
///                         "__qualname__ must be set to a string object")
///         raise
///     self.set_qualname(qualname)
/// ```
///
/// `space.realutf8_w` accepts any `isinstance_w(w_name, w_text)` —
/// `str` and its subclasses.  `isinstance_str_w` mirrors that.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_qualname(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__qualname__")?; // function.py:477
        if !crate::baseobjspace::isinstance_str_w(value) {
            return Err(crate::PyError::type_error(
                "__qualname__ must be set to a string object",
            ));
        }
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::WQualname);
        (*(obj as *mut Function)).w_qualname = value;
        Ok(())
    }
}

/// Return the function-owned qualified-name object.
///
/// `function.py fget_func_qualname` parity:
///
/// ```python
/// def fget_func_qualname(self, space):
///     return space.newtext(self.qualname)
/// ```
///
/// `self.qualname` is initialised from `qualname or self.name` at
/// `function.py __init__`.  MAKE_FUNCTION
/// (`runtime_ops::make_function_from_code_obj_with_globals_obj`) stamps `w_qualname`
/// from `codeobj.co_qualname` at construction, so subsequent
/// `__code__ = new_code` assignments do NOT alter `__qualname__`
/// (matching `pypy/interpreter/pyopcode.py` + `function.py:54`).  allow-line-citation
/// Pyre keeps the wrapped text object in `w_qualname`, the direct equivalent
/// of that object-owned PyPy field.  Returning it directly preserves the
/// identity installed by `f.__qualname__ = value`, which `update_wrapper`
/// observes.  When the field is still `PY_NULL` (legacy gateway/unit-test
/// callers), materialise the `qualname or self.name` fallback once and cache
/// it on the function instead of manufacturing a fresh string per read.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fget_func_qualname(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = &*(obj as *const Function);
        let cached = func.w_qualname;
        if !cached.is_null() && pyre_object::is_str(cached) {
            return cached;
        }
        // CPython methodobject.c exposes a type-bound builtin's qualified
        // name as `<self type>.<name>` (`object.__new__`), which is also what
        // pickle.save_global needs in order to reconstruct the carrier via
        // `getattr(object, "__new__")`.  `w_new_self` is the defining type
        // stamped by type finalisation; ordinary/module builtins retain the
        // plain-name fallback.
        let qualname =
            if !func.w_new_self.is_null() && pyre_object::typeobject::is_type(func.w_new_self) {
                format!(
                    "{}.{}",
                    pyre_object::w_type_get_qualname(func.w_new_self),
                    function_get_name(obj)
                )
            } else {
                function_get_name(obj).to_owned()
            };
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let value = pyre_object::w_str_new_managed(&qualname);
        function_set_qualname(pyre_object::gc_roots::shadow_stack_get(obj_slot), value);
        value
    }
}

/// Get the qualified-name text for formatting/error-reporting paths.
///
/// Attribute access itself uses [`fget_func_qualname`] so it returns the
/// function-owned Python object rather than a re-wrapped copy.
/// `function.py` reads the name with `space.realutf8_w`, the
/// surrogate-preserving spelling, and `:283` interpolates it into the repr
/// verbatim; `argument.py` builds its TypeErrors from the same string. So the
/// text a lone surrogate is set into has to come back out of here intact --
/// every consumer either mints a `str` from it or puts it in `e.args[0]`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_qualname(obj: PyObjectRef) -> Wtf8Buf {
    unsafe { pyre_object::w_str_get_wtf8(fget_func_qualname(obj)).to_wtf8_buf() }
}

/// Return the application-level `__qualname__` object.
///
/// PyPy keeps the qualified name on the `Function` itself rather than in a
/// side table.  CPython 3.14 additionally preserves the exact unicode object:
/// `func_get_qualname` returns `func_qualname`, and `func_set_qualname` stores
/// the supplied object with `Py_NewRef`.  Thus, after `f.__qualname__ = value`,
/// `f.__qualname__ is value` must hold, including for a `str` subclass.
///
/// Legacy construction paths leave `w_qualname` as `PY_NULL`; materialise the
/// `qualname or self.name` fallback once and retain it in the same owner field.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn function_get_qualname_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if !(*(obj as *const Function)).w_qualname.is_null() {
            return (*(obj as *const Function)).w_qualname;
        }
        // `w_str_new` is a collection point that can relocate the function, so
        // pin the receiver and reload it before storing through it.  The name
        // itself lives in the off-GC `Box<String>` and does not move.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let obj = pyre_object::gc_roots::pin_root(obj);
        let w_qualname = pyre_object::w_str_new_managed(function_get_name(obj));
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::WQualname);
        (*(obj as *mut Function)).w_qualname = w_qualname;
        w_qualname
    }
}

/// `function.py self.qualname = qualname or self.name` setter —
/// used by MAKE_FUNCTION (`runtime_ops::make_function_from_code_obj_with_globals_obj`)
/// to freeze the qualified name immediately after `function_new`.
/// Bypasses `_check_code_mutable` because this is the construction
/// path; user code goes through `fset_func_qualname` instead.
///
/// # Safety
/// `obj` must point to a valid `Function`; `value` must be a string
/// `PyObjectRef`.
#[inline]
pub unsafe fn function_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::WQualname);
        (*(obj as *mut Function)).w_qualname = value;
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn _eq(_obj: PyObjectRef, other: PyObjectRef) -> bool {
    _obj == other
}

/// PyPy-compatible descriptor accessor for function name.
///
/// CPython 3.14 stores and returns `func_name` directly, so repeated
/// `f.__name__` reads preserve object identity.  Pyre keeps the raw `name`
/// pointer for fast internal `&str` users and mirrors the application-level
/// owner with `w_name`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget_func_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_get_name_obj(obj) }
}

/// Return the application-level `__name__` object, materialising a legacy
/// fallback exactly once from the raw name pointer.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn function_get_name_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if !(*(obj as *const Function)).w_name.is_null() {
            return (*(obj as *const Function)).w_name;
        }
        // `w_str_new` is a collection point that can relocate the function, so
        // pin the receiver and reload it before storing through it.  The raw
        // name itself lives in off-GC storage and does not move.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let obj = pyre_object::gc_roots::pin_root(obj);
        let w_name = pyre_object::w_str_new_managed(function_get_name(obj));
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        function_set_name_obj(obj, w_name);
        w_name
    }
}

/// Return the canonical W_DictObject stored as `function.w_func_globals`.
///
/// `function.py self.w_func_globals = w_globals` stores the dict
/// object directly; this is a plain field load.  Returns `PY_NULL` for
/// a globals-less carrier (gateway builtins).
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_globals_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_func_globals_obj }
}

/// CPython 3.14 `PyFunctionObject.func_builtins`, resolved once during
/// construction. Returns `PY_NULL` for globals-less builtin carriers.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_builtins(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_builtins }
}

/// Get the closure tuple from a function object.
/// Returns PY_NULL if the function has no closure.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).closure }
}

/// Set the closure on a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::Closure);
        (*(obj as *mut Function)).closure = closure;
    }
}

/// Get defaults tuple.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).defs_w }
}

/// Set defaults tuple.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_defaults(obj: PyObjectRef, defaults: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::DefsW);
        (*(obj as *mut Function)).defs_w = defaults;
    }
}

/// Get kwdefaults dict.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_kw_defs }
}

/// Set kwdefaults dict.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_kwdefaults(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::WKwDefs);
        (*(obj as *mut Function)).w_kw_defs = kwdefaults;
    }
}

/// `function.py Function.init_kwdefaults_dict` — a definition's keyword-only
/// defaults are rebuilt into the namespace mapping, whose entries carry a
/// version a trace can pin, so filling one reads a normally-constant entry
/// rather than probing an ordinary dict on every call.
///
/// Upstream consumes the `(name, value)` pairs `MAKE_FUNCTION` unpacks and
/// never sees the mapping the bytecode built.  The 3.14 opcode hands that
/// mapping over already built, so the rebuild reads it back out here instead.
///
/// Only a definition converts.  `f.__kwdefaults__ = d` stores `d` itself, as
/// `fset_func_kwdefaults` does, which is why this cannot move into
/// `function_set_kwdefaults` — both paths share that sink.  Rebuilding twice is
/// a no-op: what comes back is no longer an exact `dict` and returns unchanged.
///
/// A mapping holding a key that is not an exact `str` is returned as it
/// arrived; the namespace storage would only switch back to the general one.
///
/// # Safety
/// `w_kw_defs` must be null, `None`, or a live mapping.
pub unsafe fn init_kwdefaults_dict(w_kw_defs: PyObjectRef) -> PyObjectRef {
    unsafe {
        if w_kw_defs.is_null()
            || pyre_object::is_none(w_kw_defs)
            || !pyre_object::is_exact_type(w_kw_defs, &pyre_object::DICT_TYPE)
        {
            return w_kw_defs;
        }
        let items = pyre_object::dictmultiobject::w_dict_items(w_kw_defs);
        if !items
            .iter()
            .all(|&(key, _)| pyre_object::is_exact_type(key, &pyre_object::STR_TYPE))
        {
            return w_kw_defs;
        }
        // Every store mints a cell, so the pairs read out above are published
        // before the mapping is allocated, and both they and the receiver are
        // re-read from the shadow stack after each store.
        let roots = pyre_object::gc_roots::push_roots();
        let flat: Vec<PyObjectRef> = items
            .iter()
            .flat_map(|&(key, value)| [key, value])
            .collect();
        let pairs_base = roots.publish(&flat);
        let dict_slot = roots.publish(&[pyre_object::dictmultiobject::w_module_dict_new()]);
        for index in 0..items.len() {
            pyre_object::dictmultiobject::w_dict_store(
                roots.get(dict_slot),
                roots.get(pairs_base + index * 2),
                roots.get(pairs_base + index * 2 + 1),
            );
        }
        roots.get(dict_slot)
    }
}

/// Store the keyword-only defaults a definition installs, rebuilding them
/// through [`init_kwdefaults_dict`] on the way in.  The rebuild allocates, so
/// the function is published across it.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_kwdefaults_from_definition(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe {
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.publish(&[obj]);
        let w_kw_defs = init_kwdefaults_dict(kwdefaults);
        function_set_kwdefaults(roots.get(obj_slot), w_kw_defs);
    }
}

/// PyPy-compatible `__dict__` storage field alias.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *mut Function;
        if (*func).w_func_dict.is_null() {
            // Allocate before the barrier: `w_dict_new` can collect, and that
            // collection consumes the prebuilt-dirty bit this store needs.
            // Marking first would leave the young dict stored under a clean
            // bit, so the next minor walk skips the slot and the dict goes
            // stale (`walk_raw_function_roots` then drags a dead pointer).
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(obj);
            let w_dict = crate::objspace::std::mapdict::make_instance_dict();
            let dict_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_dict);
            function_write_barrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
            let func = pyre_object::gc_roots::shadow_stack_get(obj_slot) as *mut Function;
            (*func).w_func_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
            return pyre_object::gc_roots::shadow_stack_get(dict_slot);
        }
        (*func).w_func_dict
    }
}

/// `function.py Function.setdict` — replace the function's instance
/// dict, raising `TypeError` when `value` is not a dict.  Routes through
/// `setdict` (the wholesale dict replacement), not `setattr_str(obj,
/// "__dict__", ..)` which would store a literal `"__dict__"` dict entry.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_setdict(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    crate::baseobjspace::require_dict_for_setdict(value)?;
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_func_dict = value;
    }
    Ok(())
}

/// CPython 3.14 `function.__annotate__` getter.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_annotate(obj: PyObjectRef) -> PyObjectRef {
    let value = unsafe { (*(obj as *const Function)).w_annotate };
    if value.is_null() {
        pyre_object::w_none()
    } else {
        value
    }
}

/// CPython 3.14 `function.__annotate__` setter.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_annotate(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if value.is_null() {
        return Err(crate::PyError::type_error("__annotate__ cannot be deleted"));
    }
    if !pyre_object::is_none(value) && !crate::baseobjspace::callable_w(value) {
        return Err(crate::PyError::type_error(
            "__annotate__ must be callable or None",
        ));
    }
    unsafe {
        function_set_annotate_unchecked(obj, value);
    }
    Ok(())
}

/// MAKE_FUNCTION helper for the compiler-provided PEP 649 callable.  The
/// callable is already validated by bytecode construction, but the GC write
/// barrier is still required: Function objects are old-generation carriers
/// and the annotate function may be in the nursery.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_annotate_unchecked(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_annotate = value;
        if !pyre_object::is_none(value) {
            (*func).w_ann = PY_NULL;
        }
    }
}

/// CPython 3.14 `function.__type_params__` getter.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_typeparams(obj: PyObjectRef) -> PyObjectRef {
    let value = unsafe { (*(obj as *const Function)).w_typeparams };
    if value.is_null() {
        pyre_object::w_tuple_new(vec![])
    } else {
        value
    }
}

/// CPython 3.14 `function.__type_params__` setter and
/// `_Py_set_function_type_params` opcode helper.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_typeparams(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if value.is_null() || !pyre_object::is_tuple(value) {
        return Err(crate::PyError::type_error(
            "__type_params__ must be set to a tuple",
        ));
    }
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_typeparams = value;
    }
    Ok(())
}

/// PyPy-compatible `getdict()` descriptor helper.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_getdict(obj) }
}

/// PyPy-compatible `setdict()` helper.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn setdict(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_setdict(obj, value) }
}

/// `function.py fget_func_doc` parity:
///
/// ```python
/// def fget_func_doc(self, space):
///     if self.w_doc is None:
///         self.w_doc = self.code.getdocstring(space)
///     return self.w_doc
/// ```
///
/// `(*func).w_doc == PY_NULL` mirrors PyPy's `self.w_doc is None`
/// (the unset / not-yet-cached state).  First reader resolves
/// `code.getdocstring(space)` and stamps the result back so subsequent
/// `f.__doc__ is f.__doc__` holds; `function_del_doc` writes
/// `w_none()` to make the deleted state sticky against the lazy
/// fallback (`function.py fdel_func_doc`).
///
/// `code.getdocstring(space)` has two shapes in pyre:
///   - `BuiltinCode`: stores `docstring` directly (`gateway.rs`'s
///     `BuiltinCode.docstring`).
///   - `PyCode`: docstring is the first const when `code.flags`
///     has `HAS_DOCSTRING` set, mirroring `pycode.py
///     PyCode.getdocstring`.
pub fn function_get_doc(mut obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::w_none();
    }
    let func = obj as *mut Function;
    let cached = unsafe { (*func).w_doc };
    if !cached.is_null() {
        return cached;
    }
    // Lazy fallback: `code.getdocstring(space)` (function.py).
    let resolved = pyre_object::with_roots!(obj => code_getdocstring(obj));
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = resolved;
    }
    resolved
}

/// `pycode.py PyCode.getdocstring(space)` parity for the two code
/// shapes pyre carries — extracted out of `function_get_doc` so
/// `fset_func_code`'s pre-assignment cache step
/// (`function.py self.fget_func_doc(space)`) can reach the same
/// path without going through the cache write.
fn code_getdocstring(obj: PyObjectRef) -> PyObjectRef {
    let code = unsafe { function_get_code(obj) } as PyObjectRef;
    if code.is_null() {
        return pyre_object::w_none();
    }
    if unsafe { crate::gateway::is_builtin_code(code) } {
        return unsafe { crate::gateway::builtin_code_get_docstring(code) };
    }
    if unsafe { crate::pycode::is_code(code) } {
        let raw = unsafe { crate::pycode::w_code_get_ptr(code) } as *const crate::CodeObject;
        if !raw.is_null() {
            let code_ref = unsafe { &*raw };
            if code_ref.flags.contains(crate::CodeFlags::HAS_DOCSTRING)
                && !code_ref.constants.is_empty()
            {
                let first = unsafe { crate::pycode::w_code_const(code, 0) };
                if !first.is_null() && unsafe { pyre_object::is_str(first) } {
                    return first;
                }
            }
        }
    }
    pyre_object::w_none()
}

/// `function.py fset_func_doc` parity:
///
/// ```python
/// def fset_func_doc(self, space, w_doc):
///     self._check_code_mutable("__doc__")
///     self.w_doc = w_doc
/// ```
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__doc__")?; // function.py:452
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = value;
        Ok(())
    }
}

/// `function.py fdel_func_doc` parity:
///
/// ```python
/// def fdel_func_doc(self, space):
///     self._check_code_mutable("__doc__")
///     self.w_doc = space.w_None
/// ```
///
/// Stamps `w_none()` so the next `function_get_doc` short-circuits
/// the lazy `code.getdocstring(space)` fallback — `w_doc` is no
/// longer null, so the cache hit returns `w_none()` directly.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_del_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__doc__")?; // function.py:456
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = pyre_object::w_none();
        Ok(())
    }
}

/// `function.py fget_func_annotations` parity — returns the
/// stored annotations dict, lazily allocating an empty dict on the
/// first read when none was set.  PyPy mutates `self.w_ann = space
/// .newdict()` in place so subsequent reads return the same dict
/// (`f.__annotations__ is f.__annotations__`); pyre stamps the slot
/// the same way through `function_set_annotations`.
///
/// PEP 649 lazy annotations (no PyPy counterpart — upstream targets
/// 3.11): when `w_ann` is unset but a `w_annotate` callable was
/// stamped at MAKE_FUNCTION time, evaluate it with `format=1` and
/// stamp the resulting dict, mirroring CPython 3.14
/// `func_get_annotations`.
///
/// Errors raised by the annotation thunk propagate without caching.  This is
/// required by annotationlib's FORWARDREF path, which retries a VALUE
/// `NameError` in a Stringifier globals environment.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_annotations(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        if obj.is_null() {
            return Ok(pyre_object::w_dict_new());
        }
        let func = obj as *mut Function;
        let cached = (*func).w_ann;
        if !cached.is_null() {
            return Ok(cached);
        }
        let annotate_fn = (*func).w_annotate;
        if !annotate_fn.is_null() && !pyre_object::is_none(annotate_fn) {
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(obj);
            let annotate_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(annotate_fn);
            let idx_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(1));
            let dict = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(annotate_slot),
                &[pyre_object::gc_roots::shadow_stack_get(idx_slot)],
            )?;
            if !pyre_object::is_dict(dict) {
                return Err(crate::PyError::type_error(format!(
                    "__annotate__ returned non-dict of type '{}'",
                    crate::baseobjspace::object_functionstr_type_name(dict),
                )));
            }
            let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
            function_write_barrier(obj);
            (*(obj as *mut Function)).w_ann = dict;
            return Ok(dict);
        }
        let fresh = pyre_object::w_dict_new();
        function_write_barrier(obj);
        (*func).w_ann = fresh;
        Ok(fresh)
    }
}

/// MAKE_FUNCTION ANNOTATIONS opcode helper — stamps the
/// compile-time annotations dict directly into `Function.w_ann`
/// without running `function.py fset_func_annotations`'s
/// type validation.  The compiler always emits a real dict here, so
/// the `isinstance(w_new, dict)` check would never fail; bypassing
/// it keeps the opcode hot path free of error-handling overhead.
///
/// User-level `f.__annotations__ = X` writes go through
/// `fset_func_annotations` (typedef getset descriptor) instead.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_annotations(obj: PyObjectRef, w_ann: PyObjectRef) {
    unsafe {
        if obj.is_null() {
            return;
        }
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_ann = w_ann;
        (*func).w_annotate = PY_NULL;
    }
}

/// `function.py fset_func_annotations` parity:
///
/// ```python
/// def fset_func_annotations(self, space, w_new):
///     self._check_code_mutable("__annotations__")
///     if space.is_w(w_new, space.w_None):
///         w_new = None
///     elif not space.isinstance_w(w_new, space.w_dict):
///         raise oefmt(space.w_TypeError, "__annotations__ must be a dict")
///     self.w_ann = w_new
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_annotations(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__annotations__")?; // function.py:554
        if obj.is_null() {
            return Ok(());
        }
        // function.py:555-558 — None clears the slot, anything else
        // must be a dict.
        let stored = if value.is_null() || pyre_object::is_none(value) {
            PY_NULL
        } else if pyre_object::is_dict(value) {
            value
        } else {
            return Err(crate::PyError::type_error("__annotations__ must be a dict"));
        };
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_ann = stored;
        // CPython 3.14 function___annotations___set_impl clears the lazy
        // annotation callable whenever eager annotations are assigned.
        (*func).w_annotate = PY_NULL;
        Ok(())
    }
}

/// `function.py fdel_func_annotations` parity:
///
/// ```python
/// def fdel_func_annotations(self, space):
///     self._check_code_mutable("__annotations__")
///     self.w_ann = None
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fdel_func_annotations(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__annotations__")?; // function.py:562
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_ann = PY_NULL;
        (*func).w_annotate = PY_NULL;
        Ok(())
    }
}

/// PyPy `fget_func_defaults` accessor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget_func_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_defaults(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// `function.py fset_func_defaults` parity:
///
/// ```python
/// def fset_func_defaults(self, space, w_defaults):
///     self._check_code_mutable("__defaults__")
///     if space.is_w(w_defaults, space.w_None):
///         self.defs_w = []
///         return
///     if not space.isinstance_w(w_defaults, space.w_tuple):
///         raise oefmt(space.w_TypeError,
///                     "__defaults__ must be set to a tuple object or None")
///     self.defs_w = space.fixedview(w_defaults)
/// ```
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fset_func_defaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__defaults__")?; // function.py:409
        if value.is_null() || pyre_object::is_none(value) {
            function_set_defaults(obj, pyre_object::PY_NULL);
            return Ok(());
        }
        if !pyre_object::is_tuple(value) {
            return Err(crate::PyError::type_error(
                "__defaults__ must be set to a tuple object or None",
            ));
        }
        function_set_defaults(obj, value);
        Ok(())
    }
}

/// `function.py fdel_func_defaults` parity:
///
/// ```python
/// def fdel_func_defaults(self, space):
///     self._check_code_mutable("__defaults__")
///     self.defs_w = []
/// ```
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fdel_func_defaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__defaults__")?; // function.py:419
        function_set_defaults(obj, pyre_object::PY_NULL);
        Ok(())
    }
}

/// PyPy `fget_func_kwdefaults` accessor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget_func_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_kwdefaults(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// `function.py fset_func_kwdefaults` parity:
///
/// ```python
/// def fset_func_kwdefaults(self, space, w_new):
///     if space.is_w(w_new, space.w_None):
///         self.w_kw_defs = None
///     else:
///         if not space.isinstance_w(w_new, space.w_dict):
///             raise oefmt(space.w_TypeError, "__kwdefaults__ must be a dict")
///         self.w_kw_defs = w_new
/// ```
///
/// PyPy intentionally omits `_check_code_mutable` here — `__kwdefaults__`
/// is settable on builtins too.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fset_func_kwdefaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        if value.is_null() || pyre_object::is_none(value) {
            function_set_kwdefaults(obj, pyre_object::PY_NULL);
            return Ok(());
        }
        if !pyre_object::is_dict(value) {
            return Err(crate::PyError::type_error("__kwdefaults__ must be a dict"));
        }
        function_set_kwdefaults(obj, value);
        Ok(())
    }
}

/// `function.py fdel_func_kwdefaults` parity:
///
/// ```python
/// def fdel_func_kwdefaults(self, space):
///     self.w_kw_defs = None
/// ```
///
/// PyPy intentionally omits `_check_code_mutable` here — symmetric
/// with `fset_func_kwdefaults` at function.py.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fdel_func_kwdefaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        function_set_kwdefaults(obj, pyre_object::PY_NULL);
        Ok(())
    }
}

/// function.py — `fget_func_code(self, space): return self.getcode()`
/// Uses getcode() for JIT elidable_promote / promote path.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_func_code(obj: PyObjectRef) -> PyObjectRef {
    unsafe { getcode(obj) }
}

/// PyPy-compatible `__code__` setter.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_func_code(obj: PyObjectRef, code: *const ()) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::Code);
        (*(obj as *mut Function)).code = code;
    }
}

/// PyPy-compatible `__name__` getter alias.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_func_name(obj: PyObjectRef) -> &'static str {
    unsafe { function_get_name(obj) }
}

/// The UTF-8 mirror a `Function`'s raw `name` slot holds.
///
/// `NameStorage` is a `String`, so a name carrying a lone surrogate — which
/// `surrogateescape` puts there for any byte an operating-system name failed
/// to decode — has no exact form in that slot. The `w_name` slot keeps the
/// object the caller supplied and is what `__name__` reads, so the stored
/// value survives; this is the `&str` view `repr` and the `__qualname__`
/// default consume, and it renders the surrogate rather than aborting the
/// process on it.
unsafe fn name_utf8_mirror(w_name: PyObjectRef) -> String {
    match unsafe { pyre_object::w_str_get_value_opt(w_name) } {
        Some(value) => value.to_string(),
        None => unsafe { pyre_object::w_str_get_wtf8(w_name) }
            .to_string_lossy()
            .into_owned(),
    }
}

/// PyPy-compatible `__name__` setter.
///
/// A GC-managed (user) function boxes the new name in a GC-managed storage box
/// (`NameStorage`), reclaimed by the box tid's drop glue and greyed through the
/// `FUNCTION_NAME_OFFSET` edge; an immortal function keeps a `malloc_raw` name
/// (an immortal holder can never grey an old-gen box). The old name slot is not
/// freed here — a GC box is reclaimed on sweep once it becomes unreachable, and
/// an immortal `malloc_raw` name was never collector-freed — so the box tid's
/// drop glue stays the sole reclaimer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_set_func_name(obj: PyObjectRef, name: PyObjectRef) {
    unsafe {
        if !crate::baseobjspace::isinstance_str_w(name) {
            return;
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_name = name;
        let raw_name = name_utf8_mirror(name);
        let raw_name =
            if pyre_object::gc_hook::try_gc_owns_object(obj as pyre_object::gc_hook::GCREF) {
                pyre_object::gc_storage::gc_alloc_storage_box(
                    raw_name,
                    pyre_object::typeobject::name_storage_gc_type_id(),
                ) as *const String
            } else {
                pyre_object::lltype::malloc_raw(raw_name) as *const String
            };
        function_notify_quasi_immut(obj, QuasiImmutSlot::Name);
        (*(obj as *mut Function)).name = raw_name;
    }
}

/// Construction-time `__name__` stamp from the code object's shared wrapped
/// `co_name`. The raw `name` pointer is set separately from `CodeObject.obj_name`.
///
/// # Safety
/// `obj` must point to a valid `Function`; `value` must be a string object.
#[inline]
pub unsafe fn function_set_name_obj(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_name = value;
    }
}

/// `function.py fget_func_objclass` parity:
///
/// ```python
/// def fget_func_objclass(self, space):
///     if self.w_objclass is None:
///         raise oefmt(space.w_AttributeError, "__objclass__")
///     return self.w_objclass
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fget_func_objclass(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let value = (*(obj as *const Function)).w_objclass;
        if value.is_null() {
            return Err(crate::PyError::attribute_error("__objclass__"));
        }
        Ok(value)
    }
}

/// `function.py set_objclass(w_type)` parity — direct field
/// write used by descriptor-bind helpers.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_objclass(obj: PyObjectRef, w_type: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::WObjclass);
        (*(obj as *mut Function)).w_objclass = w_type;
    }
}

/// `function.py fget_func_text_signature` parity:
///
/// ```python
/// def fget_func_text_signature(self, space):
///     if self.w_text_signature is None:
///         raise oefmt(space.w_AttributeError, "__text_signature__")
///     return self.w_text_signature
/// ```
///
/// PyPy distinguishes the RPython-level `None` (unset, raises
/// `AttributeError`) from `space.w_None` (explicitly stored). Pyre
/// uses `PY_NULL` for the former; `space.w_None` survives as a
/// real PyObjectRef.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fget_func_text_signature(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let value = (*(obj as *const Function)).w_text_signature;
        if value.is_null() {
            return Err(crate::PyError::attribute_error("__text_signature__"));
        }
        Ok(value)
    }
}

/// `function.py fset_func_text_signature` parity:
///
/// ```python
/// def fset_func_text_signature(self, space, w_value):
///     self.w_text_signature = w_value
/// ```
///
/// Direct field write — even `space.w_None` is preserved as a real
/// value (only the RPython-level `None`, i.e. `PY_NULL`, is treated
/// as "unset" by `fget_func_text_signature`).
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fset_func_text_signature(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        function_notify_quasi_immut(obj, QuasiImmutSlot::WTextSignature);
        (*(obj as *mut Function)).w_text_signature = value;
    }
}

/// `function.py fset_func_name` parity:
///
/// ```python
/// def fset_func_name(self, space, w_name):
///     self._check_code_mutable("__name__")
///     if space.isinstance_w(w_name, space.w_text):
///         self.name = space.text_w(w_name)
///     else:
///         raise oefmt(space.w_TypeError,
///                     "__name__ must be set to a string object")
/// ```
///
/// `isinstance_str_w` mirrors `space.isinstance_w(w_name, space.w_text)`
/// — accepts `str` and any `str` subclass.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fset_func_name(obj: PyObjectRef, name: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__name__")?; // function.py:463
        if !crate::baseobjspace::isinstance_str_w(name) {
            return Err(crate::PyError::type_error(
                "__name__ must be set to a string object",
            ));
        }
        function_set_func_name(obj, name);
        Ok(())
    }
}

// _check_code_mutable is defined above (function.py parity).

/// `function.py fset_func_code` parity:
///
/// ```python
/// def fset_func_code(self, space, w_code):
///     from pypy.interpreter.pycode import PyCode
///     if not self.can_change_code:
///         raise oefmt(space.w_AttributeError,
///                     "Cannot change code attribute of builtin functions")
///     code = space.interp_w(Code, w_code)
///     closure_len = 0
///     if self.closure:
///         closure_len = len(self.closure)
///     if isinstance(code, PyCode) and closure_len != len(code.co_freevars):
///         raise oefmt(space.w_ValueError,
///                     "%N() requires a code object with %d free vars, not "
///                     "%d", self, closure_len, len(code.co_freevars))
///     self.code = code
/// ```
///
/// `w_code` is the user-provided value as a `PyObjectRef`; this
/// helper enforces the `is_code(...)` (interp_w PyCode) check and
/// the closure / freevar arity invariant before reaching for the
/// inner code pointer.  Without these checks the previous setter
/// would happily reinterpret arbitrary objects as `*const CodeObject`
/// and corrupt the function's internal pointer (worse than main's
/// pre-fix shadowing-via-instance-dict regression).
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_code(obj: PyObjectRef, w_code: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        // function.py:527-529 — `can_change_code = False` on
        // BuiltinFunction / FunctionWithFixedCode.  pyre exposes
        // this via `_check_code_mutable` which raises the same
        // AttributeError.
        _check_code_mutable(obj, "__code__")?;
        // function.py:530 — `space.interp_w(Code, w_code)` raises
        // TypeError when `w_code` is not a `PyCode` instance.
        if w_code.is_null() || !crate::pycode::is_code(w_code) {
            return Err(crate::PyError::type_error(
                "__code__ must be set to a code object",
            ));
        }
        // function.py:531-537 — closure-vs-freevars arity check.
        // PyPy raises ValueError if the function has N closure cells
        // and the new code object declares M co_freevars with N != M;
        // mismatch leaves dangling cells / unbound freevars at runtime.
        let closure = function_get_closure(obj);
        let closure_len = if closure.is_null() || pyre_object::is_none(closure) {
            0
        } else {
            pyre_object::w_tuple_len(closure)
        };
        let raw_code = crate::pycode::w_code_get_ptr(w_code) as *const crate::CodeObject;
        let freevars_len = if raw_code.is_null() {
            0
        } else {
            (&(*raw_code).freevars).len()
        };
        if closure_len != freevars_len {
            let name = function_get_name(obj);
            return Err(crate::PyError::value_error(format!(
                "{name}() requires a code object with {closure_len} free vars, not {freevars_len}"
            )));
        }
        // CPython 3.14 `func_set_code`: changing between a plain function,
        // generator, coroutine, and async generator remains permitted for
        // compatibility, but warns because the callable's result protocol
        // changes underneath existing references.
        let old_code = &*(get_pycode(obj) as *const crate::CodeObject);
        let kind_mask = crate::CodeFlags::GENERATOR
            | crate::CodeFlags::COROUTINE
            | crate::CodeFlags::ASYNC_GENERATOR;
        // `warn_deprecation` runs the app-level warnings machinery and
        // `function_get_doc` realizes the docstring; both allocate, so publish
        // the two operands and read them back from their root slots rather
        // than storing through a pre-move address.
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[obj, w_code]);
        if old_code.flags.intersection(kind_mask).bits()
            != (*raw_code).flags.intersection(kind_mask).bits()
        {
            crate::warn::warn_deprecation(
                "Assigning a code object of non-matching type is deprecated (e.g., from a generator to a plain function)",
            )?;
        }
        // function.py self.fget_func_doc(space) — see test_issue1293.
        // Resolves the OLD code's docstring into `w_doc` *before* the
        // pointer flip so the cached value reflects the function's
        // original docstring, not the new code's first const.
        let _ = function_get_doc(pyre_object::gc_roots::shadow_stack_get(base));
        function_set_func_code(
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(base + 1) as *const (),
        );
        Ok(())
    }
}

/// PyPy-compatible `__closure__` getter alias.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn function_get_func_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_get_closure(obj) }
}

/// PyPy-compatible `fget_func_closure`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget_func_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_func_closure(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// PyPy-compatible `__closure__` setter alias.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fset_func_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe {
        function_set_closure(obj, closure);
    }
}

/// `function.py fget___module__`:
///
/// ```python
/// def fget___module__(self, space):
///     if self.w_module is None:
///         if self.w_func_globals is not None and not space.is_w(
///                 self.w_func_globals, space.w_None):
///             self.w_module = space.call_method(
///                 self.w_func_globals, "get", space.newtext("__name__"))
///         else:
///             self.w_module = space.w_None
///     return self.w_module
/// ```
///
/// Caches on first read: if w_module is PY_NULL (RPython-level None,
/// unset), computes from globals["__name__"] and stores into
/// self.w_module.  PY_NULL = RPython `None`, w_none() = space.w_None.
/// After fdel___module__ writes w_none(), subsequent reads return
/// None without re-computing from globals.
///
/// Routes through `w_func_globals_obj` (the canonical PyObjectRef the
/// user sees via `__globals__`) and dispatches `dict.get` so dict
/// subclasses that override `get` are observed.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget___module__(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *mut Function;
        // function.py:504: if self.w_module is None
        if (*func).w_module.is_null() {
            // function.py:505-506: space.call_method(self.w_func_globals,
            // "get", space.newtext("__name__"))
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(obj);
            let name_key = pyre_object::unicodeobject::intern_str_value("__name__");
            let func = pyre_object::gc_roots::shadow_stack_get(obj_slot) as *mut Function;
            let w_globals = (*func).w_func_globals_obj;
            if !w_globals.is_null() && !pyre_object::is_none(w_globals) {
                // Dispatch through `dict.get` so dict subclasses that
                // override `get` are observed.  When the lookup yields
                // PY_NULL we fall back to `space.w_None` per the
                // upstream attribute-not-found branch.
                //
                // `call_method` is a collection point.  Pin the function, the
                // globals dict, the interned key, and the returned name, then
                // reload each after the call — a raw `w_globals` across that
                // call is the dict's pre-move address, and storing the
                // unrooted result writes an interior word into `w_module`
                // that the next minor copies as a root.
                let pair = pyre_object::gc_roots::pin_roots(&[w_globals, name_key]);
                let result = crate::baseobjspace::call_method(
                    pyre_object::gc_roots::shadow_stack_get(pair),
                    "get",
                    &[pyre_object::gc_roots::shadow_stack_get(pair + 1)],
                );
                let stored = if result.is_null() {
                    pyre_object::w_none()
                } else {
                    result
                };
                let stored_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(stored);
                let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
                function_write_barrier(obj);
                let func = obj as *mut Function;
                (*func).w_module = pyre_object::gc_roots::shadow_stack_get(stored_slot);
                return (*func).w_module;
            }
            // function.py `fget___module__`: self.w_module = space.w_None
            function_write_barrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
            let func = pyre_object::gc_roots::shadow_stack_get(obj_slot) as *mut Function;
            (*func).w_module = pyre_object::w_none();
            return (*func).w_module;
        }
        // function.py:509: return self.w_module
        let func = obj as *mut Function;
        (*func).w_module
    }
}

/// PyPy-compatible `descr_function__new__` helper.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_function__new__(
    code: *const (),
    w_globals: PyObjectRef,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = _argdefs;
        let name = if !w_name.is_null() && !pyre_object::is_none(w_name) {
            name_utf8_mirror(w_name)
        } else {
            String::new()
        };
        let closure = if w_closure.is_null() || pyre_object::is_none(w_closure) {
            pyre_object::PY_NULL
        } else {
            w_closure
        };
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[w_name, w_globals, closure]);
        let func = function_new_with_closure(
            code,
            name,
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            pyre_object::gc_roots::shadow_stack_get(base + 2),
        );
        let func_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        let w_name = pyre_object::gc_roots::shadow_stack_get(base);
        // `name` above is only the UTF-8 mirror. The object the caller passed
        // is what `__name__` must answer with, so store it: otherwise the
        // getter rebuilds one from the mirror and a name carrying a lone
        // surrogate comes back escaped instead of equal to what went in.
        if !w_name.is_null() && !pyre_object::is_none(w_name) {
            function_set_name_obj(
                pyre_object::gc_roots::shadow_stack_get(func_slot),
                pyre_object::gc_roots::shadow_stack_get(base),
            );
        }
        pyre_object::gc_roots::shadow_stack_get(func_slot)
    }
}

/// `function.py fset___module__` parity:
///
/// ```python
/// def fset___module__(self, space, w_module):
///     self._check_code_mutable("__module__")
///     self.w_module = w_module
/// ```
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fset___module__(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__module__")?; // function.py:512
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = value;
        Ok(())
    }
}

/// `function.py fdel___module__` parity:
///
/// ```python
/// def fdel___module__(self, space):
///     self._check_code_mutable("__module__")
///     self.w_module = space.w_None
/// ```
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fdel___module__(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__module__")?; // function.py:516
        // function.py:517: self.w_module = space.w_None
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = pyre_object::w_none();
        Ok(())
    }
}

/// PyPy-compatible `descr_function__new__` overload.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn _cleanup_(_obj: PyObjectRef) -> bool {
    true
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_builtinfunction__new__(
    code: *const (),
    w_globals: PyObjectRef,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    unsafe { descr_function__new__(code, w_globals, w_name, _argdefs, w_closure) }
}

/// `funcobject.c func_new_impl` — `FunctionType(code, globals,
/// name=None, argdefs=None, closure=None, kwdefaults=None)`.
///
/// `args[0]` is the class (the `function` type); the remaining
/// positional args plus the trailing `__pyre_kw__` dict
/// (`split_builtin_kwargs`) supply the constructor parameters, so both
/// `FunctionType(code, g)` and `FunctionType(code, g, closure=c,
/// kwdefaults=k)` reach the same slot resolution.
pub fn descr_function_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `args[0]` is `cls`; the constructor parameters start at index 1. All six
    // are positional-or-keyword, so the shared binder owns the arity count,
    // the by-name-and-position duplicate, and the missing-required report —
    // resolving each slot by hand let the positional value win a duplicate and
    // let a seventh argument through.
    let scope = crate::builtins::bind_builtin_kwargs(
        args.get(1..).unwrap_or(&[]),
        &[
            "code",
            "globals",
            "name",
            "argdefs",
            "closure",
            "kwdefaults",
        ],
        &[true, true, false, false, false, false],
        "function",
    )?;
    let w_code = scope[0];
    let w_globals = scope[1];
    let w_name = scope[2];
    let w_argdefs = scope[3];
    let w_closure = scope[4];
    let w_kwdefaults = scope[5];

    if w_code.is_null() || !unsafe { crate::pycode::is_code(w_code) } {
        return Err(crate::PyError::type_error(
            "function() argument 'code' must be code, not ...",
        ));
    }
    // `PyDict_Check` accepts dict subclasses (annotationlib hands a
    // `_StringifierDict`); resolve the backing storage rather than
    // demanding an exact `dict`. The closure is addressed later as a
    // tuple, so publish it before this call and read it back after.
    let _closure_roots = pyre_object::gc_roots::push_roots();
    // All six words exist. `RootScope::pin_roots` publishes them once;
    // each `pin_root` would normalize after the first write.
    let base = pyre_object::gc_roots::pin_roots(&[
        w_code,
        w_globals,
        w_name,
        w_argdefs,
        w_kwdefaults,
        w_closure,
    ]);
    let closure_slot = base + 5;
    let w_globals = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if crate::type_methods::resolve_dict_backing(w_globals).is_null() {
        return Err(crate::PyError::type_error(
            "function() argument 'globals' must be dict, not ...",
        ));
    }
    // `pop_roots` writes every live local. `resolve_dict_backing` can collect,
    // so the words below are the slots, not the pre-pin copies.
    let w_code = pyre_object::gc_roots::shadow_stack_get(base);
    let w_name = pyre_object::gc_roots::shadow_stack_get(base + 2);
    let code_ptr = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
    let name = if w_name.is_null() || unsafe { pyre_object::is_none(w_name) } {
        unsafe { (*code_ptr).obj_name.to_string() }
    } else if unsafe { pyre_object::is_str(w_name) } {
        unsafe { name_utf8_mirror(w_name) }
    } else {
        return Err(crate::PyError::type_error(
            "arg 3 (name) must be None or string",
        ));
    };
    // `name_utf8_mirror` can collect. Reload before the defaults and closure
    // reads (`ShadowStackFrameworkGCTransformer.pop_roots`).
    let w_code = pyre_object::gc_roots::shadow_stack_get(base);
    let w_argdefs = pyre_object::gc_roots::shadow_stack_get(base + 3);
    let code_ptr = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
    // [3.14-spec] PyPy `Function.descr_function__new__` calls
    // `space.fixedview(argdefs)` before it inspects `closure`, but that accepts
    // lists and then reports PyPy's `TypeError("invalid closure")` for
    // `FunctionType(code, {}, None, [], object())`.  CPython 3.14.2 instead
    // reports `TypeError("arg 4 (defaults) must be None or tuple")`: validate
    // the public defaults contract at the same PyPy-shaped constructor site,
    // before the unchanged closure processing below.
    if !w_argdefs.is_null()
        && !unsafe { pyre_object::is_none(w_argdefs) }
        && !unsafe { pyre_object::is_tuple(w_argdefs) }
    {
        return Err(crate::PyError::type_error(
            "arg 4 (defaults) must be None or tuple",
        ));
    }
    // [3.14-spec] `function.py Function.descr_function__new__` derives the
    // closure contract from `PyCode.co_freevars`, then refuses anything but an
    // exact tuple (`TypeError("invalid closure")`) and refuses a closure at all
    // for a code object with no free variables (`ValueError("no closure
    // needed")`).  Measured on 3.14.2: `FunctionType(f.__code__, {}, None,
    // None, ())` returns a function whose `__closure__` reads back as `()`, and
    // a `tuple` subclass is accepted and read back as that subclass.  The two
    // widenings only decide what may be stored once, at construction; the
    // `closure?[*]` hint (function.py) governs rebinding the field and the
    // immutability of the cells it holds, and every accepted element is still
    // checked to be a `Cell` below.
    let nfreevars = unsafe { (&(*code_ptr).freevars).len() };
    let w_closure = pyre_object::gc_roots::shadow_stack_get(closure_slot);
    let closure = if w_closure.is_null() || unsafe { pyre_object::is_none(w_closure) } {
        if nfreevars != 0 {
            return Err(crate::PyError::type_error("arg 5 (closure) must be tuple"));
        }
        PY_NULL
    } else {
        if !unsafe { pyre_object::is_tuple(w_closure) } {
            return Err(crate::PyError::type_error(
                "arg 5 (closure) must be None or tuple",
            ));
        }
        let nclosure = unsafe { pyre_object::w_tuple_len(w_closure) };
        if nclosure != nfreevars {
            let w_shown = unsafe { crate::pycode::w_code_name_obj(w_code) };
            let mut msg = rustpython_wtf8::Wtf8Buf::new();
            if !w_shown.is_null() {
                msg.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_shown) });
            }
            msg.push_str(" requires closure of length ");
            msg.push_str(&nfreevars.to_string());
            msg.push_str(", not ");
            msg.push_str(&nclosure.to_string());
            return Err(crate::PyError::value_error(msg));
        }
        for index in 0..nclosure {
            let cell = unsafe { pyre_object::w_tuple_getitem(w_closure, index as i64) }
                .expect("closure tuple index is in bounds");
            if !unsafe { pyre_object::is_cell(cell) } {
                return Err(crate::PyError::type_error(format!(
                    "arg 5 (closure) expected cell, found {}",
                    crate::type_methods::arg_type_name(cell)
                )));
            }
        }
        w_closure
    };
    // `function.py self.w_func_globals = w_globals` stores the dict
    // object itself as the function's sole globals carrier.  Preserving a
    // dict subclass here is observable: annotationlib's `_StringifierDict`
    // supplies unresolved names through `__missing__` when it clones a PEP
    // 649 annotate function with `types.FunctionType`.
    let w_code = pyre_object::gc_roots::shadow_stack_get(base);
    let w_globals = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let closure = if closure.is_null() {
        PY_NULL
    } else {
        pyre_object::gc_roots::shadow_stack_get(closure_slot)
    };
    let func = function_new_with_closure(w_code as *const (), name, w_globals, closure);
    // `function.py` `self.name = forcename or code.co_name`. The raw `name`
    // mirror cannot hold a lone surrogate, so `__name__` reads the str object.
    let _qual_roots = pyre_object::gc_roots::push_roots();
    let code_slot =
        pyre_object::gc_roots::pin_roots(&[pyre_object::gc_roots::shadow_stack_get(base), func]);
    let func_slot = code_slot + 1;
    let w_name = pyre_object::gc_roots::shadow_stack_get(base + 2);
    let name_obj = if !w_name.is_null() && !unsafe { pyre_object::is_none(w_name) } {
        w_name
    } else {
        unsafe {
            crate::pycode::w_code_name_obj(pyre_object::gc_roots::shadow_stack_get(code_slot))
        }
    };
    if !name_obj.is_null() {
        unsafe {
            function_set_name_obj(pyre_object::gc_roots::shadow_stack_get(func_slot), name_obj)
        };
    }
    let qualname_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(unsafe {
        crate::pycode::w_code_qualname_obj(pyre_object::gc_roots::shadow_stack_get(code_slot))
    });
    unsafe {
        function_set_qualname(
            pyre_object::gc_roots::shadow_stack_get(func_slot),
            pyre_object::gc_roots::shadow_stack_get(qualname_slot),
        )
    };
    let w_argdefs = pyre_object::gc_roots::shadow_stack_get(base + 3);
    if !w_argdefs.is_null() && !unsafe { pyre_object::is_none(w_argdefs) } {
        unsafe {
            function_set_defaults(
                pyre_object::gc_roots::shadow_stack_get(func_slot),
                w_argdefs,
            )
        };
    }
    let w_kwdefaults = pyre_object::gc_roots::shadow_stack_get(base + 4);
    if !w_kwdefaults.is_null() && !unsafe { pyre_object::is_none(w_kwdefaults) } {
        if !unsafe { pyre_object::is_dict(w_kwdefaults) } {
            return Err(crate::PyError::type_error(
                "arg 6 (kwdefaults) must be None or dict",
            ));
        }
        unsafe {
            function_set_kwdefaults(
                pyre_object::gc_roots::shadow_stack_get(func_slot),
                w_kwdefaults,
            )
        };
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(func_slot))
}

/// PyPy-compatible static registry hook.
#[inline]
pub fn add_to_table() {}

/// PyPy-compatible `__doc__` getter.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget_func_doc(obj: PyObjectRef) -> PyObjectRef {
    function_get_doc(obj)
}

/// function.py — `fset_func_doc` descriptor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fset_func_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_set_doc(obj, value) }
}

/// function.py — `fdel_func_doc` descriptor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fdel_func_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_del_doc(obj) }
}

/// `pypy/objspace/std/util.py:6-13` — `id()` of a plain `int` / `float`
/// / `complex` is its value tagged `(value << IDTAG_SHIFT) | IDTAG_*`;
/// the unique-ified immutables (empty/short `bytes`/`str`, empty
/// `tuple`/`frozenset`) use `IDTAG_SPECIAL`.
const IDTAG_SHIFT: i64 = 4;
const IDTAG_INT: i64 = 1;
const IDTAG_ALT_UID: i64 = 2;
const IDTAG_FLOAT: i64 = 5;
const IDTAG_SPECIAL: i64 = 11;

#[inline]
pub fn immutable_unique_id(obj: PyObjectRef) -> Option<PyObjectRef> {
    // `W_AbstractIntObject.immutable_unique_id` (intobject.py): a
    // plain `int` — `W_IntObject` or the BigInt-backed `W_LongObject` —
    // has a value-derived id `(bigint_w << IDTAG_SHIFT) | IDTAG_INT`,
    // wrapped as an `int` (a `long` when it overflows i64).
    // `W_FloatObject.immutable_unique_id` (floatobject.py) does
    // the same with the float bit pattern (`float2longlong`) and
    // `IDTAG_FLOAT`. `bool` (`W_BoolObject.immutable_unique_id` returns
    // None, boolobject.py) and `int`/`float` subclasses
    // (`user_overridden_class`) return `None`, so `space.id` falls back
    // to the address-based uid.
    unsafe {
        if is_exact_type(obj, &INT_TYPE) {
            // `b.lshift(IDTAG_SHIFT).int_or_(IDTAG_INT)`.
            let b = (pyre_object::functional::range_obj_to_bigint(obj) << IDTAG_SHIFT as usize)
                .int_or_(IDTAG_INT);
            return Some(pyre_object::functional::range_bigint_to_obj(b));
        }
        if is_exact_type(obj, &FLOAT_TYPE) {
            // `float2longlong(float_w(self))` reinterprets the f64 bits as
            // a signed i64, then `lshift(IDTAG_SHIFT).int_or_(IDTAG_FLOAT)`.
            // NaNs use the address uid, matching `is_w`'s pointer identity.
            let value = pyre_object::floatobject::w_float_get_value(obj);
            if value.is_nan() {
                return None;
            }
            let bits = value.to_bits() as i64;
            let b = (majit_rlib::rbigint::RBigInt::from(bits) << IDTAG_SHIFT as usize)
                .int_or_(IDTAG_FLOAT);
            return Some(pyre_object::functional::range_bigint_to_obj(b));
        }
        // Unlike `W_ComplexObject.immutable_unique_id` (complexobject.py),
        // CPython 3.14 complex identity is address-based, so there is no
        // IDTAG_COMPLEX branch.
        if is_exact_type(obj, &TUPLE_TYPE) {
            // `W_AbstractTupleObject.immutable_unique_id`
            // (tupleobject.py:57-62): only the empty tuple is unique-ified
            // — a non-empty tuple (`length() > 0`) returns `None` and
            // `space.id` falls back to the address-based uid. `tuple`
            // subclasses (`user_overridden_class`) also return `None`; the
            // exact-type gate excludes them. The empty tuple has base value
            // 258: `(258 << IDTAG_SHIFT) | IDTAG_SPECIAL`; the shifted value
            // has low 4 bits clear, so `| IDTAG_SPECIAL` == `+`. The uid
            // fits i64, so it is wrapped with `w_int_new` (`space.newint`).
            if pyre_object::tupleobject::w_tuple_len(obj) > 0 {
                return None;
            }
            let uid = (258i64 << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
        if is_exact_type(obj, &pyre_object::bytesobject::BYTES_TYPE) {
            // `W_AbstractBytesObject.immutable_unique_id`
            // (bytesobject.py): `len(s) > 1` is `compute_unique_id(s)` — the
            // id of the STORAGE, which is what `is_w` compares at that length.
            // BytesListStrategy re-wraps one erased rpython string, so the
            // wrapper address would answer a fresh value per read and make
            // `id(a) == id(b)` disagree with `a is b`. `len(s) <= 1` is
            // unique-ified: `base = ord(s[0])` (0..255) for one byte,
            // `base = 256` for the empty bytes,
            // `uid = (base << IDTAG_SHIFT) | IDTAG_SPECIAL`.
            let len = pyre_object::bytesobject::w_bytes_len(obj);
            if len > 1 {
                let block = pyre_object::bytesobject::w_bytes_block(obj);
                return Some(pyre_object::intobject::w_int_new(
                    pyre_object::gc_hook::gc_identity_hash(block as usize) as i64,
                ));
            }
            let base: i64 = if len == 1 {
                pyre_object::bytesobject::w_bytes_getitem(obj, 0) as i64
            } else {
                256
            };
            let uid = (base << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
        if is_exact_type(obj, &STR_TYPE) {
            // `W_UnicodeObject.immutable_unique_id` (unicodeobject.py): more
            // than one code point uses the `_utf8` storage identity, which
            // AsciiListStrategy preserves across wrapper allocation. Empty
            // and one-code-point strings are unique-ified from the code point.
            let len = pyre_object::unicodeobject::w_str_len(obj);
            if len > 1 {
                let storage = pyre_object::unicodeobject::w_str_storage(obj);
                return Some(pyre_object::intobject::w_int_new(
                    pyre_object::gc_hook::gc_identity_hash(storage as usize) as i64 + IDTAG_ALT_UID,
                ));
            }
            let base: i64 = if len == 1 {
                let cp = pyre_object::rutf8::codepoint_at_pos(
                    pyre_object::unicodeobject::w_str_get_wtf8(obj),
                    0,
                );
                // `(neg << IDTAG_SHIFT) | IDTAG_SPECIAL` == `+` (low 4 bits 0).
                !cp
            } else {
                257
            };
            let uid = (base << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
        if is_exact_type(obj, &pyre_object::setobject::FROZENSET_TYPE) {
            // `W_FrozensetObject.immutable_unique_id` (setobject.py):
            // a non-empty frozenset (`length() > 0`) and `frozenset`
            // subclasses (excluded by the exact-type gate) return `None`.
            // The empty frozenset is unique-ified with base value 259. The
            // mutable `set` has its own type and does not override this, so
            // the `FROZENSET_TYPE` gate does not match it.
            if pyre_object::setobject::w_set_len(obj) > 0 {
                return None;
            }
            let uid = (259i64 << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
    }
    None
}

/// PyPy-compatible `find` helper.
#[inline]
pub fn find(_identifier: &str) -> PyObjectRef {
    let _ = _identifier;
    pyre_object::PY_NULL
}

/// function.py `is_builtin_code`: unwrap a `_Method`, take the function's
/// code, and report whether that code is a `BuiltinCode`.
///
/// Not the same predicate as `gateway::is_builtin_code`, which asks whether
/// the object handed to it *is* a builtin code.  This one is asked about a
/// callable, which is why it has to reach the code through `getcode` first.
///
/// `getcode`, not the `code` field: it is the accessor that promotes a
/// changeable code and takes the elidable one otherwise, and a profiled call
/// reaches this while a trace records.
#[inline]
pub fn is_builtin_code(w_func: PyObjectRef) -> bool {
    unsafe {
        let mut w_func = w_func;
        if !w_func.is_null() && pyre_object::is_method(w_func) {
            w_func = pyre_object::w_method_get_func(w_func);
        }
        if w_func.is_null() || !crate::is_function(w_func) {
            return false;
        }
        let code = crate::getcode(w_func) as PyObjectRef;
        !code.is_null() && crate::gateway::is_builtin_code(code)
    }
}

#[inline]
pub fn descr_init() {}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_classmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::function::w_classmethod_new(w_function)
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_classmethod_get(
    w_obj: PyObjectRef,
    obj: PyObjectRef,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = w_cls;
        if obj.is_null() || pyre_object::is_none(obj) {
            w_obj
        } else {
            let func = pyre_object::w_classmethod_get_func(w_obj);
            let cls = if w_obj.is_null() {
                pyre_object::w_none()
            } else {
                obj
            };
            pyre_object::w_method_new(func, cls, cls)
        }
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_staticmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::function::w_staticmethod_new(w_function)
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_staticmethod_get(
    obj: PyObjectRef,
    _obj: PyObjectRef,
    _cls: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = (_obj, _cls);
        if obj.is_null() {
            pyre_object::w_none()
        } else {
            pyre_object::w_staticmethod_get_func(obj)
        }
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
    w_instance: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let _ = _subtype;
    if w_function.is_null() || !crate::baseobjspace::callable_w(w_function) {
        return Err(crate::PyError::type_error(
            "first argument must be callable",
        ));
    }
    if w_instance.is_null() || unsafe { pyre_object::is_none(w_instance) } {
        return Err(crate::PyError::type_error("instance must not be None"));
    }
    let w_class = crate::typedef::r#type(w_instance).map_or(pyre_object::PY_NULL, |p| p.as_ptr());
    Ok(pyre_object::w_method_new(w_function, w_instance, w_class))
}

/// `interp2app` unwraps the receiver as `Method` before entering PyPy's
/// method bodies.  Reproduce that gateway contract before reading the Rust
/// payload, including for direct descriptor calls with a foreign receiver.
#[inline]
pub fn require_method(method: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if method.is_null() || !unsafe { pyre_object::function::is_method(method) } {
        let received = crate::typedef::r#type(method)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()) })
            .unwrap_or("object");
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a 'method' object but received a '{received}'"
        )));
    }
    Ok(method)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method_get(
    method: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let method = require_method(method, "__get__")?;
    let obj_is_none = obj.is_null() || unsafe { pyre_object::is_none(obj) };
    let cls_is_none = cls.is_null() || unsafe { pyre_object::is_none(cls) };
    if obj_is_none && cls_is_none {
        return Err(crate::PyError::type_error("__get__(None, None) is invalid"));
    }
    Ok(method)
}

#[inline]
pub fn descr_method_call(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let method = require_method(
        positional.first().copied().unwrap_or(pyre_object::PY_NULL),
        "__call__",
    )?;
    let call_args = positional.get(1..).unwrap_or(&[]);
    if !crate::builtins::has_real_kwargs(kwargs) {
        return crate::call::call_function_impl_result(method, call_args);
    }
    let keyword_args: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = unsafe {
        pyre_object::w_dict_str_entries(kwargs.unwrap())
            .into_iter()
            .filter(|(name, _)| name != "__pyre_kw__")
            .map(|(name, value)| (rustpython_wtf8::Wtf8Buf::from_string(name), value))
            .collect()
    };
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "method call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(unsafe { &mut *frame }, method, call_args, &keyword_args)
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method_eq(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let this = require_method(this, "__eq__")?;
    if !unsafe { pyre_object::is_method(other) } {
        return Ok(pyre_object::special::w_not_implemented());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let pair = pyre_object::gc_roots::pin_roots(&[this, other]);
    let fa =
        unsafe { pyre_object::w_method_get_func(pyre_object::gc_roots::shadow_stack_get(pair)) };
    let fb = unsafe {
        pyre_object::w_method_get_func(pyre_object::gc_roots::shadow_stack_get(pair + 1))
    };
    let funcs = pyre_object::gc_roots::pin_roots(&[fa, fb]);
    let funcs_equal = crate::baseobjspace::eq_w(
        pyre_object::gc_roots::shadow_stack_get(funcs),
        pyre_object::gc_roots::shadow_stack_get(funcs + 1),
    )?;
    let selves_identical = unsafe {
        pyre_object::w_method_get_self(pyre_object::gc_roots::shadow_stack_get(pair))
            == pyre_object::w_method_get_self(pyre_object::gc_roots::shadow_stack_get(pair + 1))
    };
    Ok(pyre_object::w_bool_from(funcs_equal && selves_identical))
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method_ne(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    require_method(this, "__ne__")?;
    let equal = unsafe { descr_method_eq(this, other)? };
    if unsafe { pyre_object::is_not_implemented(equal) } {
        Ok(equal)
    } else {
        Ok(pyre_object::w_bool_from(!unsafe {
            pyre_object::w_bool_get_value(equal)
        }))
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method_repr(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let obj = require_method(obj, "__repr__")?;
    let mut function = unsafe { pyre_object::w_method_get_func(obj) };
    let mut instance = unsafe { pyre_object::w_method_get_self(obj) };
    let w_name = match pyre_object::with_roots!(function, instance =>
        crate::baseobjspace::getattr_str(function, "__qualname__")
    ) {
        Ok(value) => Some(value),
        Err(err) if err.kind == crate::PyErrorKind::AttributeError => {
            match pyre_object::with_roots!(instance =>
                crate::baseobjspace::getattr_str(function, "__name__")
            ) {
                Ok(value) => Some(value),
                Err(err) if err.kind == crate::PyErrorKind::AttributeError => None,
                Err(err) => return Err(err),
            }
        }
        Err(err) => return Err(err),
    };
    // The name is spliced in as itself, so a lone surrogate in a `__qualname__`
    // reaches the repr rather than collapsing the whole name to the placeholder.
    // The `is_str` filter is what makes the unchecked WTF-8 read sound.
    let name = w_name
        .filter(|&value| unsafe { pyre_object::is_str(value) })
        .map(|value| unsafe { pyre_object::w_str_get_wtf8(value) }.to_wtf8_buf())
        .unwrap_or_else(|| Wtf8Buf::from_string("?".to_owned()));
    let instance_repr = unsafe { crate::display::py_repr_wtf8(instance)? };
    Ok(pyre_object::w_str_from_wtf8_managed(
        crate::display::wtf8_format!("<bound method ", name, " of ", instance_repr, ">"),
    ))
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method_getattribute(
    obj: PyObjectRef,
    name: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let mut obj = require_method(obj, "__getattribute__")?;
    if !unsafe { pyre_object::is_str(name) } {
        return Err(crate::PyError::type_error("attribute name must be string"));
    }
    let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(name) }) else {
        return Err(crate::PyError::type_error("attribute name must be string"));
    };
    let mut function = unsafe { pyre_object::w_method_get_func(obj) };
    // PyPy's Method.descr_method_getattribute delegates every miss to the
    // wrapped function, so an inherited object.__subclasshook__ keeps the
    // defining qualname.  CPython 3.14 exposes object.__subclasshook__ and
    // object.__init_subclass__ as METH_CLASS builtins: their bound-method
    // qualname uses the class they are currently bound to (`type` / `int` /
    // ...), while the definitions still remain absent from those type dicts.
    // Preserve that selected 3.14 delta without changing ordinary Python
    // classmethods, whose qualname continues to name the defining class.
    if name == "__qualname__"
        && let Some(qualname) =
            unsafe { pyre_object::with_roots!(function, obj => method_class_bound_qualname(obj))? }
    {
        return Ok(qualname);
    }
    // function.py:604-614 — method attributes win, except `__doc__`;
    // an AttributeError falls back to the wrapped function.
    if name != "__doc__" {
        match pyre_object::with_roots!(function =>
            crate::baseobjspace::object_getattribute(obj, name)
        ) {
            Ok(value) => return Ok(value),
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }
    }
    crate::baseobjspace::getattr_str(function, name)
}

/// The callables `object` publishes as class methods, paired with the name
/// each is published under.
///
/// Only these two take a bound qualname naming the class they were read off,
/// and `__qualname__` is writable on an ordinary function, so they have to be
/// recognised by identity.  `Box` so a slot keeps one address for the
/// collector's whole lifetime; the `Vec` only owns them.
static OBJECT_CLASS_METHODS: parking_lot::Mutex<Vec<(&'static str, Box<usize>)>> =
    parking_lot::Mutex::new(Vec::new());

/// Publish `function` as `object.<name>` for [`method_class_bound_qualname`].
///
/// The address is registered as a process-lifetime GC root so the collector
/// forwards it, the `intern_str` idiom: nothing else keeps this raw copy in
/// step with a moving collection.
pub fn register_object_class_method(name: &'static str, function: PyObjectRef) {
    let mut slot = Box::new(function as usize);
    let root_slot = (&mut *slot) as *mut usize as *mut pyre_object::gc_hook::GCREF;
    unsafe { pyre_object::gc_hook::try_gc_add_root(root_slot) };
    OBJECT_CLASS_METHODS.lock().push((name, slot));
}

/// The name `object` publishes `function` under, if it publishes it at all.
///
/// This process-global registry is the runtime metadata carrier for the
/// documented CPython-3.14 METH_CLASS observable above.  It is not a
/// trace-time constant: keep its mutex and moving-GC root table behind one
/// residual lookup, just as PyPy keeps host/runtime metadata operations out of
/// the object-space trace.  The returned static name remains ordinary data to
/// the caller.
#[majit_macros::dont_look_inside]
fn object_class_method_name(function: PyObjectRef) -> Option<&'static str> {
    OBJECT_CLASS_METHODS
        .lock()
        .iter()
        .find(|(_, slot)| **slot == function as usize)
        .map(|(name, _)| *name)
}

/// Python 3.14's dynamic qualname for the two METH_CLASS methods inherited
/// from `object`; `None` for every ordinary bound method.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn method_class_bound_qualname(
    obj: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    let instance = unsafe { pyre_object::w_method_get_self(obj) };
    if !unsafe { pyre_object::is_type(instance) } {
        return Ok(None);
    }
    // Identify the two by the callables `object` publishes, not by their
    // qualname: `__qualname__` is writable on an ordinary function, so a
    // classmethod over one that spells itself `object.__subclasshook__` would
    // otherwise be rewritten to name its own owner instead of keeping the
    // string it was given.
    let Some(inherited_name) = object_class_method_name(function) else {
        return Ok(None);
    };
    let owner_qualname = crate::baseobjspace::getattr_str(instance, "__qualname__")?;
    let Some(owner_qualname) = (unsafe { pyre_object::w_str_get_value_opt(owner_qualname) }) else {
        return Err(crate::PyError::type_error(
            "type.__qualname__ is not a unicode object",
        ));
    };
    Ok(Some(pyre_object::w_str_new_managed(&format!(
        "{owner_qualname}.{inherited_name}"
    ))))
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method_hash(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let obj = require_method(obj, "__hash__")?;
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    let instance = unsafe { pyre_object::w_method_get_self(obj) };
    let x = pyre_object::gc_hook::gc_identity_hash(instance as usize) as i64;
    let y = crate::baseobjspace::hash_w_strict(function)?;
    let value = x ^ y;
    Ok(if value == -1 { -2 } else { value })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_method__reduce__(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let obj = require_method(obj, "__reduce__")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    let instance = unsafe { pyre_object::w_method_get_self(obj) };
    let pair = pyre_object::gc_roots::pin_roots(&[function, instance]);
    let name = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(pair),
        "__name__",
    )?;
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    let args = pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(pair + 1),
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    ]);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args);
    let getattr_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(crate::baseobjspace::builtin_callable("getattr"));
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(getattr_slot),
        pyre_object::gc_roots::shadow_stack_get(args_slot),
    ]))
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_w(_obj: PyObjectRef, _other: PyObjectRef) -> bool {
    _obj == _other
}

/// PyPy-compatible `descr_function_call` helper.
#[inline]
pub fn descr_function_call(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        call_args(args[0], &args[1..])
    }
}

/// PyPy-compatible `descr_function_get` helper.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_function_get(
    _func: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> PyObjectRef {
    let _ = cls;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        _func
    } else {
        pyre_object::w_method_new(_func, obj, cls)
    }
}

/// PyPy-compatible `descr_function_repr` helper.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn descr_function_repr(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let name = function_get_name(obj);
        pyre_object::w_str_new_managed(&format!("function {name}"))
    }
}

/// PyPy-compatible `__code__` getter for direct descriptors.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn fget_func_code(obj: PyObjectRef) -> *const () {
    unsafe { function_get_code(obj) }
}

/// PyPy-compatible `descr__reduce__` helper.
#[inline]
pub fn descr_function__reduce__(_obj: PyObjectRef) -> PyObjectRef {
    // Empty tuples are nursery-allocated; pin each inner one before the
    // next constructor, then build the outer tuple from the reloaded slots.
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_tuple_new(vec![]));
    fields.push(pyre_object::w_tuple_new(vec![]));
    pyre_object::w_tuple_new(fields.take())
}

/// PyPy-compatible `descr__setstate__` helper.
#[inline]
pub fn descr_function__setstate__(_obj: PyObjectRef, _state: PyObjectRef) {
    let _ = _state;
}

#[inline]
pub fn __init__() {}

#[inline]
pub fn __repr__() -> String {
    "function".to_string()
}

/// PyPy-compatible `__call__` alias helper.
#[inline]
pub fn call(frame: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(frame, args)
}

/// PyPy-compatible call fast-path hooks.
#[inline]
pub fn call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call_function(func, args)
}

/// PyPy-compatible `call_obj_args` helper.
#[inline]
pub fn call_obj_args(func: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::baseobjspace::call_obj_args(func, obj, args)
}

/// PyPy-compatible `call_args` instance method.
#[inline]
pub fn function_call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// PyPy-compatible `call_obj_args` instance method.
#[inline]
#[allow(dead_code)]
pub(crate) fn function_call_obj_args(
    func: PyObjectRef,
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    call_obj_args(func, obj, args)
}

/// `function.py Function.funccall` — the Result-returning body.
///
/// Fixed-arity 0..=4 builtins go through `BuiltinCodeN.fastcall_N`:
/// `getcode()` then `code.fastfunc_N(...)`.  `builtin_code_call` is the
/// gateway that holds that function-pointer field; the caller's live
/// arguments are pinned here because that gateway roots nothing of its
/// own (`funccall_valuestack`'s same pin).
///
/// `funccall_valuestack` uses this same `BuiltinCodeN.fastcall_N` arm
/// whether or not a trace is recording: the CALL walker intercepts the
/// `CallFn` residual, and after `abort: force quasi-immut` the regular
/// interpreter still wants the gateway rather than `space.call_function`.
pub fn funccall_result(
    func: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { crate::getcode(func) };
    let nargs = args.len();
    let fast_natural_arity =
        unsafe { crate::pycode::code_get_fast_natural_arity(code as PyObjectRef) } as usize;
    if nargs == fast_natural_arity
        && nargs <= 4
        && unsafe { crate::gateway::is_builtin_code(code as PyObjectRef) }
    {
        // One-shot `pin_roots`: `gc_push_roots` publishes the whole live set.
        // Sequential `pin_root(code)` then per-arg is a safepoint window.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = match nargs {
            0 => _roots.pin_roots(&[code as PyObjectRef]),
            1 => _roots.pin_roots(&[code as PyObjectRef, args[0]]),
            2 => _roots.pin_roots(&[code as PyObjectRef, args[0], args[1]]),
            3 => _roots.pin_roots(&[code as PyObjectRef, args[0], args[1], args[2]]),
            4 => _roots.pin_roots(&[code as PyObjectRef, args[0], args[1], args[2], args[3]]),
            _ => unreachable!(),
        };
        let code = _roots.get(root_base);
        let rooted_arg = |index| _roots.get(root_base + 1 + index);
        return match nargs {
            0 => unsafe { crate::builtin_code_call(code, &[]) },
            1 => unsafe { crate::builtin_code_call(code, &[rooted_arg(0)]) },
            2 => unsafe { crate::builtin_code_call(code, &[rooted_arg(0), rooted_arg(1)]) },
            3 => unsafe {
                crate::builtin_code_call(code, &[rooted_arg(0), rooted_arg(1), rooted_arg(2)])
            },
            4 => unsafe {
                crate::builtin_code_call(
                    code,
                    &[rooted_arg(0), rooted_arg(1), rooted_arg(2), rooted_arg(3)],
                )
            },
            _ => unreachable!(),
        };
    }
    // function.py `Function.funccall`:
    //   elif (nargs | PyCode.FLATPYCALL) == fast_natural_arity:
    //       assert isinstance(code, PyCode)
    //       if nargs < 5:
    //           createframe; funccallunrolling local copy; new_frame.run
    // A user dunder (`__add__` and the rest) has `fast_natural_arity =
    // FLATPYCALL | co_argcount`, so the builtin arm above does not match
    // and this arm is what looks inside to `PyFrame.run` instead of
    // residualising `call_args` / `fill_user_function_args`.
    if (nargs | crate::BuiltinCodeFlags::FLATPYCALL.bits() as usize) == fast_natural_arity
        && nargs < 5
    {
        return funccall_flat_from_args(func, code, args);
    }
    // function.py `Function.funccall` PASSTHROUGHARGS1:
    //   elif nargs >= 1 and fast_natural_arity == Code.PASSTHROUGHARGS1:
    //       return code.funcrun_obj(self, args_w[0], Arguments(..., list(args_w[1:])))
    // BuiltinCodePassThroughArguments1.funcrun_obj calls
    // `self.func__args__(space, w_obj, args)` with the receiver separate
    // from the Arguments rest. It prepends the receiver only on
    // DescrMismatch (`args.prepend(w_obj)`). Pyre's BuiltinCodeFn takes a
    // flat slice, so the call is `[args[0], ...args[1:]]`.
    if nargs >= 1 && fast_natural_arity == crate::BuiltinCodeFlags::PASSTHROUGHARGS1.bits() as usize
    {
        let _roots = pyre_object::gc_roots::push_roots();
        let mut live = Vec::with_capacity(1 + nargs);
        live.push(code as PyObjectRef);
        live.extend_from_slice(args);
        let root_base = _roots.pin_roots(&live);
        let args_w: Vec<PyObjectRef> = (0..nargs).map(|i| _roots.get(root_base + 1 + i)).collect();
        return unsafe { crate::builtin_code_call(_roots.get(root_base), &args_w) };
    }
    crate::call::call_function_impl_result(func, args)
}

/// function.py `Function.funccall` FLATPYCALL arm (`nargs < 5`).
///
/// `space.createframe(code, self.w_func_globals, self)` then
/// `funccallunrolling` writes `args_w[i]` into
/// `locals_cells_stack_w[i]`, then `new_frame.run`.
///
/// `funccallunrolling = unrolling_iterable(range(4))` expands the copy in
/// flowspace, so the graph has no loop and `Function.funccall` carries no
/// `@jit.unroll_safe`. The `match nargs` below is that expansion. PyPy has
/// no extra GC API on that path; pyre is 3.14t, so the converted
/// `PyObjectRef`s stay pinned across `createframe` /
/// `try_new_for_call_with_closure_and_globals_obj` with one-shot
/// `pin_roots` (`gc_push_roots` publishes the whole live set), matching
/// `_flat_pycall_defaults`.
fn funccall_flat_from_args(
    func: PyObjectRef,
    code: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let nargs = args.len();
    debug_assert!(
        !unsafe { crate::gateway::is_builtin_code(code) },
        "function.py funccall FLATPYCALL arm is PyCode only"
    );
    // One-shot publish before any allocating call: sequential `pin_root(func)`
    // then `pin_root(code)` then per-arg is a safepoint window.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = match nargs {
        0 => _roots.pin_roots(&[func, code]),
        1 => _roots.pin_roots(&[func, code, args[0]]),
        2 => _roots.pin_roots(&[func, code, args[0], args[1]]),
        3 => _roots.pin_roots(&[func, code, args[0], args[1], args[2]]),
        4 => _roots.pin_roots(&[func, code, args[0], args[1], args[2], args[3]]),
        _ => unreachable!(),
    };
    let func = _roots.get(root_base);
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };
    // Pin globals/closure on the same scope as func/code/args so
    // `createframe` / `getexecutioncontext` / `FrameBox::new` /
    // `frame_into_generator_for_function` / `get_eval_fn` stay under one
    // `push_roots`, matching `_flat_pycall_defaults`.
    let env_base = _roots.pin_roots(&[w_globals, closure]);
    let mut func = _roots.get(root_base);

    let mut new_frame = crate::pyframe::FrameBox::new(
        match pyre_object::with_roots!(func => {
            crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
                _roots.get(root_base + 1) as *const (),
                &[],
                _roots.get(env_base),
                crate::call::getexecutioncontext(),
                _roots.get(env_base + 1),
                crate::pyframe::FrameLocalsArrayAllocation::NurseryGc,
            )
        }) {
            Ok(f) => f,
            Err(e) => return Err(e),
        },
    );

    // funccallunrolling: `for i in unrolling_iterable(range(4)): if i < nargs`
    match nargs {
        0 => {}
        1 => new_frame.set_locals_w(0, _roots.get(root_base + 2)),
        2 => {
            new_frame.set_locals_w(0, _roots.get(root_base + 2));
            new_frame.set_locals_w(1, _roots.get(root_base + 3));
        }
        3 => {
            new_frame.set_locals_w(0, _roots.get(root_base + 2));
            new_frame.set_locals_w(1, _roots.get(root_base + 3));
            new_frame.set_locals_w(2, _roots.get(root_base + 4));
        }
        4 => {
            new_frame.set_locals_w(0, _roots.get(root_base + 2));
            new_frame.set_locals_w(1, _roots.get(root_base + 3));
            new_frame.set_locals_w(2, _roots.get(root_base + 4));
            new_frame.set_locals_w(3, _roots.get(root_base + 5));
        }
        _ => unreachable!(),
    }
    crate::pyframe::remember_frame_locals_array(new_frame.locals_cells_stack_w);
    new_frame.fix_array_ptrs();
    // function.py `Function.funccall` — `new_frame.run(self.name, self.qualname)`
    // → `initialize_as_generator(name, qualname)` / `GeneratorIterator`.
    // Same arm as `_flat_pycall`: transfer the owned `FrameBox`, do not
    // snapshot through `run_with_jit` / `initialize_as_generator`. Re-read
    // `func` off the live publish the way `_flat_pycall_defaults` does.
    let mut func = _roots.get(root_base);
    if new_frame._is_generator_or_coroutine() {
        pyre_object::with_roots!(func => {
            crate::call::frame_into_generator_for_function(new_frame, func)
        })
    } else {
        pyre_object::with_roots!(func => {
            let eval_fn = crate::call::get_eval_fn();
            eval_fn(&mut new_frame, None)
        })
    }
}

/// PyPy-compatible `funccall` helper.
#[inline]
pub fn funccall(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    match funccall_result(func, args) {
        Ok(v) => v,
        Err(e) => {
            crate::call::set_call_error(e);
            pyre_object::PY_NULL
        }
    }
}

/// baseobjspace.py: `space._code_of_sys_exc_info` — the BuiltinCode object
/// backing `sys.exc_info`, captured at sys-module init time so the JIT
/// fast-path in `funccall_valuestack` can recognize this specific call and
/// inline `exc_info_direct` without going through the regular dispatch.
///
/// PyPy stores this on the shared object space. Pyre's process-global slots
/// preserve the same identity across execution contexts and OS threads. The
/// paired `direct_fn` returns the same `(type, value, traceback)` tuple as the
/// regular closure but skips the builtin-call setup.
type ExcInfoDirectFn = fn(&mut crate::pyframe::PyFrame) -> PyObjectRef;
static SYS_EXC_INFO_CODE: std::sync::atomic::AtomicPtr<PyObject> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static SYS_EXC_INFO_DIRECT_FN: std::sync::OnceLock<ExcInfoDirectFn> = std::sync::OnceLock::new();

/// Register the BuiltinCode pointer + direct helper for `sys.exc_info`.
///
/// Called once during sys module init (after the builtin is created). The
/// `code` pointer is the BuiltinCode object underlying the `exc_info`
/// builtin function; `direct_fn` is the JIT-direct equivalent of the
/// closure body. `funccall_valuestack` consults both to take the fast path.
pub fn register_sys_exc_info_path(code: PyObjectRef, direct_fn: ExcInfoDirectFn) {
    SYS_EXC_INFO_CODE.store(code, std::sync::atomic::Ordering::Release);
    // Reinitializing the sys module may replace its BuiltinCode object, but
    // every instance is backed by the same `exc_info_direct` symbol. Preserve
    // the object-space-wide helper installed by the first initialization.
    let _ = SYS_EXC_INFO_DIRECT_FN.set(direct_fn);
}

#[inline]
fn sys_exc_info_code() -> PyObjectRef {
    SYS_EXC_INFO_CODE.load(std::sync::atomic::Ordering::Acquire)
}

#[inline]
fn sys_exc_info_direct_fn() -> Option<ExcInfoDirectFn> {
    SYS_EXC_INFO_DIRECT_FN.get().copied()
}

/// function.py `funccall_valuestack` — fast-path call dispatcher.
///
/// Dispatches based on `code.fast_natural_arity`:
/// - nargs == arity (0-4): direct builtin fastcall from stack (no Vec alloc)
/// - (nargs | FLATPYCALL) == arity: _flat_pycall (user code, exact arity)
/// - FLATPYCALL + defaults: _flat_pycall_defaults
/// - Fallback: allocate Vec via peekvalues + generic call path
pub fn funccall_valuestack(
    func: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    // Root the caller before any allocation in this dispatcher.  `_flat_pycall`
    // `space.createframe` is a nursery safepoint; RPython's GC transform then
    // reloads `frame` for the peek/dropvalues that follow.  The raw `&mut`
    // would otherwise name the abandoned copy, so dropvalues would miss the
    // live valuestackdepth and the CALL result push would write past the
    // array.  `OpcodeStepExecutor::call` already holds an outer FrameAnchor
    // for that push; this inner one is what the peek/dropvalues after
    // createframe read through.
    let caller = crate::eval::FrameAnchor::new(frame);
    // `function.py funccall_valuestack` does not register an extra root: the
    // caller frame is already a GC object kept live by `FrameAnchor` and the
    // execution-context chain, and `pyframe_object_custom_trace` visits
    // `locals_cells_stack_w`. Value-stack stores can still leave a young
    // pointer in an old array, so the write barrier runs before anything allocates.
    crate::pyframe::remember_frame_locals_array(frame.locals_cells_stack_w);
    // A compiled callee prologue can publish an overflow before control
    // returns to this dispatcher.  The fresh stack check belongs to the
    // Python frame entry (`PyFrame.execute_frame.insert_stack_check_here`),
    // not to every CALL: builtin calls do not create a recursive frame.
    if let Err(e) = crate::stack_check::drain_jit_pending_exception() {
        crate::call::set_call_error(e);
        return pyre_object::PY_NULL;
    }

    let code = unsafe { crate::getcode(func) };

    // function.py:146-150 — JIT direct path for `sys.exc_info()` with no
    // arguments: skip the builtin call entirely and inline the tuple
    // construction. PyPy uses `space._code_of_sys_exc_info`; pyre uses the
    // shared cache populated during sys module init.
    if nargs == 0
        && majit_metainterp::jit::we_are_jitted()
        && std::ptr::eq(code, sys_exc_info_code())
        && let Some(direct_fn) = sys_exc_info_direct_fn()
    {
        let frame = unsafe { &mut *caller.live() };
        frame.dropvalues(dropvalues);
        return direct_fn(frame);
    }

    let fast_natural_arity =
        unsafe { crate::pycode::code_get_fast_natural_arity(code as PyObjectRef) } as usize;

    // function.py:153-199 — the four fast-path arms are an `elif` chain, so an
    // arm whose *outer* test matches consumes the call even when its inner
    // dispatch declines: `function.py:154-184` dispatches only arities 0..=4,
    // and a natural-arity call above that falls out of the whole chain to
    // `make_arguments` below rather than into a later arm.  The exclusivity is
    // load-bearing, not stylistic — `fast_natural_arity` carries FLATPYCALL in
    // its 0x100 bit, so an `nargs` equal to it also satisfies the next arm's
    // `nargs | FLATPYCALL == fast_natural_arity` and would enter `_flat_pycall`
    // with 0x100 more arguments than the callee has local slots.
    let natural_arity_call = nargs == fast_natural_arity;

    // function.py:153-184 — nargs == fast_natural_arity: builtin fast path
    // baseobjspace.py call_valuestack — skip when profiling (c_call/c_return events)
    if natural_arity_call && nargs <= 4 && !frame.get_is_being_profiled() {
        debug_assert!(
            (fast_natural_arity & crate::BuiltinCodeFlags::FLATPYCALL.bits() as usize) == 0,
            "BuiltinCodeFlags::FLATPYCALL.bits() bit set on arity {fast_natural_arity} — not a builtin code"
        );
        // function.py:154-184 — BuiltinCodeN.fastcall_N dispatch.
        // RPython translation keeps the BuiltinCode and each peeked argument
        // live across dropvalues() and the gateway call.  The Python frame
        // slots cease to be roots when dropvalues() clears them, so publish
        // the same live-variable set on pyre's shadow stack before doing so.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = _roots.base();
        let _ = _roots.pin_root(code as PyObjectRef);
        match nargs {
            0 => {}
            1 => {
                let _ = _roots.pin_root(frame.peekvalue(0));
            }
            2 => {
                let _ = _roots.pin_root(frame.peekvalue(1));
                let _ = _roots.pin_root(frame.peekvalue(0));
            }
            3 => {
                let _ = _roots.pin_root(frame.peekvalue(2));
                let _ = _roots.pin_root(frame.peekvalue(1));
                let _ = _roots.pin_root(frame.peekvalue(0));
            }
            4 => {
                let _ = _roots.pin_root(frame.peekvalue(3));
                let _ = _roots.pin_root(frame.peekvalue(2));
                let _ = _roots.pin_root(frame.peekvalue(1));
                let _ = _roots.pin_root(frame.peekvalue(0));
            }
            _ => unreachable!(),
        }
        unsafe { &mut *caller.live() }.dropvalues(dropvalues);

        let code = _roots.get(root_base);
        let rooted_arg = |index| _roots.get(root_base + 1 + index);
        // Pyre builtins share a single fn(&[PyObjectRef]) signature, so use
        // fixed-size stack arrays instead of heap-allocating a Vec.
        let result = match nargs {
            0 => unsafe { crate::builtin_code_call(code, &[]) },
            1 => unsafe { crate::builtin_code_call(code, &[rooted_arg(0)]) },
            2 => unsafe { crate::builtin_code_call(code, &[rooted_arg(0), rooted_arg(1)]) },
            3 => unsafe {
                crate::builtin_code_call(code, &[rooted_arg(0), rooted_arg(1), rooted_arg(2)])
            },
            4 => unsafe {
                crate::builtin_code_call(
                    code,
                    &[rooted_arg(0), rooted_arg(1), rooted_arg(2), rooted_arg(3)],
                )
            },
            _ => unreachable!(),
        };
        return match result {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        };
    }

    // function.py:185-187 — (nargs | FLATPYCALL) == fast_natural_arity
    if !natural_arity_call
        && (nargs | crate::BuiltinCodeFlags::FLATPYCALL.bits() as usize) == fast_natural_arity
    {
        return _flat_pycall(func, code, nargs, &caller, dropvalues);
    }

    // function.py:188-193 — FLATPYCALL bit set + nargs within defaults range
    if !natural_arity_call
        && (fast_natural_arity & crate::BuiltinCodeFlags::FLATPYCALL.bits() as usize) != 0
    {
        let natural_arity = fast_natural_arity & 0xff;
        if nargs < natural_arity {
            let raw_defs = unsafe { crate::function_get_defaults(func) };
            let defs = raw_defs;
            let defs_len = if defs.is_null() || !unsafe { pyre_object::is_tuple(defs) } {
                0
            } else {
                unsafe { pyre_object::w_tuple_len(defs) }
            };
            if nargs >= natural_arity.saturating_sub(defs_len) {
                return _flat_pycall_defaults(
                    func,
                    code,
                    nargs,
                    &caller,
                    defs,
                    natural_arity - nargs,
                    dropvalues,
                );
            }
        }
    }

    // function.py:194-199 — PASSTHROUGHARGS1 dispatch.
    // BuiltinCodePassThroughArguments1.funcrun_obj calls
    // `self.func__args__(space, w_obj, args)` with the receiver separate
    // from the Arguments rest. It prepends the receiver only on
    // DescrMismatch (`args.prepend(w_obj)`). Pyre's BuiltinCodeFn takes a
    // flat slice, so the peek/Arguments split is structural — the closure
    // sees `[w_obj, ...rest]`.
    if !natural_arity_call
        && fast_natural_arity == crate::BuiltinCodeFlags::PASSTHROUGHARGS1.bits() as usize
        && nargs >= 1
    {
        let w_obj = frame.peekvalue(nargs - 1);
        let rest = frame.make_arguments(nargs - 1, false, func);
        // Same live-variable set as the fixed-arity arm above, for the same
        // two reasons: `dropvalues()` retires the frame slots that root these
        // values, and `args_w` is a native Vec no root walker updates.
        // `builtin_code_call` roots nothing of its own, so the whole
        // `[code, w_obj, ...rest]` set has to be published here and read back
        // for the call.
        let _roots = pyre_object::gc_roots::push_roots();
        let mut live = Vec::with_capacity(2 + rest.len());
        live.push(code as PyObjectRef);
        live.push(w_obj);
        live.extend_from_slice(&rest);
        let root_base = _roots.pin_roots(&live);
        unsafe { &mut *caller.live() }.dropvalues(dropvalues);
        let args_w: Vec<PyObjectRef> = (0..nargs).map(|i| _roots.get(root_base + 1 + i)).collect();
        return match unsafe { crate::builtin_code_call(_roots.get(root_base), &args_w) } {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        };
    }

    // function.py:201-203 — fallback: build Arguments via make_arguments
    // (carries methodcall + w_function for diagnostics) and dispatch through
    // call_args.
    let args = frame.make_arguments(nargs, methodcall, func);
    unsafe { &mut *caller.live() }.dropvalues(dropvalues);
    funccall(func, &args)
}

/// function.py `_flat_pycall` — create frame directly from stack.
///
/// For user functions with exact arity match (no defaults needed).
/// Copies args from caller's value stack into the new frame's locals
/// without intermediate Vec allocation.
/// `function.py _flat_pycall` is `@jit.unroll_safe`.
#[majit_macros::unroll_safe]
fn _flat_pycall(
    mut func: PyObjectRef,
    code: PyObjectRef,
    nargs: usize,
    caller: &crate::eval::FrameAnchor,
    dropvalues: usize,
) -> PyObjectRef {
    // `function.py _flat_pycall` copies positional arguments off the caller
    // value stack after `space.createframe`. The caller is the executing
    // frame (`FrameAnchor`, execution-context chain); its trace reaches the
    // stack, so the copy sees forwarded operands.
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };

    // function.py:208-209 — createframe(code, w_func_globals, self)
    // FrameBox: the callee runs through the JIT, so it must be a
    // header-bearing heap frame (write barrier reads a valid header at
    // frame - GC_HEADER_SIZE) rather than a bare interpreter-stack frame.
    let mut new_frame = crate::pyframe::FrameBox::new(
        match pyre_object::with_roots!(func =>
            crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
                code as *const (),
                &[], // locals filled below directly from stack
                w_globals,
                crate::call::getexecutioncontext(),
                closure,
                crate::pyframe::FrameLocalsArrayAllocation::NurseryGc,
            )
        ) {
            Ok(f) => f,
            Err(e) => {
                crate::call::set_call_error(e);
                return pyre_object::PY_NULL;
            }
        },
    );

    // createframe is a nursery safepoint.  RPython reloads `frame` here via
    // the GC transform; read the forwarded caller for peekvalue / dropvalues
    // so those writes land on the live valuestackdepth.
    let frame = unsafe { &mut *caller.live() };
    // function.py:210-211 — copy from stack into locals directly
    // peekvalue(nargs-1-i) gives bottom-to-top order (matching local slot order)
    for i in 0..nargs {
        new_frame.set_locals_w(i, frame.peekvalue(nargs - 1 - i));
    }
    // `PyFrame.__init__` allocates `locals_cells_stack_w` as a fresh
    // `[None] * size` nursery array.  Arguments just written into it are
    // also young.  `remember_frame_locals_array` still runs: a full nursery
    // can spill the array to old-gen, and until the callee sits on
    // `f_backref` nothing else exposes these slots.
    crate::pyframe::remember_frame_locals_array(new_frame.locals_cells_stack_w);
    let frame = unsafe { &mut *caller.live() };
    frame.dropvalues(dropvalues);
    new_frame.fix_array_ptrs();

    // function.py `_flat_pycall` — return new_frame.run(self.name, self.qualname)
    // Generator/coroutine: wrap the frame in a generator object and hand it
    // ownership (no execution). The callee `FrameBox` keeps a GC frame live
    // through `OwnerRootGuard`; `pyframe_object_custom_trace` visits its
    // locals array, so eval does not register that slot again.
    if new_frame._is_generator_or_coroutine() {
        match crate::call::frame_into_generator_for_function(new_frame, func) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    } else {
        let eval_fn = crate::call::get_eval_fn();
        match eval_fn(&mut new_frame, None) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    }
}

/// function.py `_flat_pycall_defaults` — flat call with defaults.
///
/// Same as `_flat_pycall` but also fills missing positional args from
/// `self.defs_w[ndefs - defs_to_load ..]`.
/// `defs` is the pre-unwrapped defaults tuple (already null-checked and
/// verified as a tuple by the caller in `funccall_valuestack`).
/// `function.py _flat_pycall_defaults` is `@jit.unroll_safe`.
#[majit_macros::unroll_safe]
fn _flat_pycall_defaults(
    func: PyObjectRef,
    code: PyObjectRef,
    nargs: usize,
    caller: &crate::eval::FrameAnchor,
    defs: PyObjectRef,
    defs_to_load: usize,
    dropvalues: usize,
) -> PyObjectRef {
    // Same live caller frame as `_flat_pycall`: positional stack entries are
    // copied only after the callee allocation returns, and the executing
    // frame's trace keeps them. Normalizing a published root is a safepoint,
    // so every call input goes out in one publish and is read back from its
    // slot before use.
    let _roots = pyre_object::gc_roots::push_roots();
    let inputs_base = _roots.publish(&[func, code, defs]);
    _roots.normalize(inputs_base, 3);
    let func = _roots.get(inputs_base);
    let code = _roots.get(inputs_base + 1);
    let defs_idx = inputs_base + 2;
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };

    // FrameBox: header-bearing heap frame for the JIT write barrier.
    let mut new_frame = crate::pyframe::FrameBox::new(
        match crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
            code as *const (),
            &[], // locals filled below
            w_globals,
            crate::call::getexecutioncontext(),
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::NurseryGc,
        ) {
            Ok(f) => f,
            Err(e) => {
                crate::call::set_call_error(e);
                return pyre_object::PY_NULL;
            }
        },
    );

    // createframe is a nursery safepoint; reload the caller the way the GC
    // transform reloads `frame` after `space.createframe` in
    // `function.py _flat_pycall_defaults`.
    let frame = unsafe { &mut *caller.live() };
    // function.py:221-222 — copy positional args from stack
    for i in 0..nargs {
        new_frame.set_locals_w(i, frame.peekvalue(nargs - 1 - i));
    }

    // function.py:224-229 — fill remaining from defs_w
    let defs = pyre_object::gc_roots::shadow_stack_get(defs_idx);
    if !defs.is_null() {
        let ndefs = unsafe { pyre_object::w_tuple_len(defs) };
        let start = ndefs - defs_to_load;
        let mut i = nargs;
        for j in start..ndefs {
            if let Some(val) = unsafe { pyre_object::w_tuple_getitem(defs, j as i64) } {
                new_frame.set_locals_w(i, val);
            }
            i += 1;
        }
    }

    // Same barrier as `_flat_pycall`: a full nursery can spill the callee's
    // locals array to old-gen, and the arguments and defaults in it are young.
    crate::pyframe::remember_frame_locals_array(new_frame.locals_cells_stack_w);
    let frame = unsafe { &mut *caller.live() };
    frame.dropvalues(dropvalues);
    new_frame.fix_array_ptrs();

    // function.py `_flat_pycall_defaults` — return new_frame.run.
    // Same callee root as `_flat_pycall`: `FrameBox`'s `OwnerRootGuard`.
    if new_frame._is_generator_or_coroutine() {
        match crate::call::frame_into_generator_for_function(new_frame, _roots.get(inputs_base)) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    } else {
        let eval_fn = crate::call::get_eval_fn();
        match eval_fn(&mut new_frame, None) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_unique_id_matches_pypy_storage_and_codepoint_rules() {
        crate::test_hooks::install_hash_hook();

        let ascii = pyre_object::unicodeobject::w_str_new_managed("ascii");
        let ascii_uid = immutable_unique_id(ascii).expect("exact str has a unique id");
        let storage = unsafe { pyre_object::unicodeobject::w_str_storage(ascii) };
        assert_eq!(
            unsafe { pyre_object::intobject::w_int_get_value(ascii_uid) },
            pyre_object::gc_hook::gc_identity_hash(storage as usize) as i64 + IDTAG_ALT_UID
        );

        // The branch is based on `_len()`, so one non-ASCII code point uses
        // the value-derived special id even though UTF-8 encodes it in bytes.
        let one = pyre_object::unicodeobject::w_str_new_managed("é");
        let one_uid = immutable_unique_id(one).expect("exact str has a unique id");
        assert_eq!(
            unsafe { pyre_object::intobject::w_int_get_value(one_uid) },
            (!(0xe9_i64) << IDTAG_SHIFT) + IDTAG_SPECIAL
        );
    }

    #[test]
    fn test_function_create() {
        crate::test_hooks::install_hash_hook();
        // Function.code now stores a Code-level wrapper (PyCode).
        let raw_code = 0xDEAD_BEEF as *const ();
        let w_code = crate::w_code_new(raw_code);
        let w_globals = pyre_object::w_module_dict_new();
        let obj = function_new(w_code as *const (), "myfunc".to_string(), w_globals);
        unsafe {
            assert!(is_function(obj));
            assert!(!is_int(obj));
            assert_eq!(function_get_code(obj), w_code as *const ());
            assert_eq!(function_get_name(obj), "myfunc");
            assert_eq!(function_get_globals_obj(obj), w_globals);
            assert!(function_get_closure(obj).is_null());
        }
    }

    #[test]
    fn function_qualname_getter_preserves_the_stored_object() {
        crate::test_hooks::install_hash_hook();
        let raw_code = 0xDEAD_BEEF as *const ();
        let w_code = crate::w_code_new(raw_code);
        let w_globals = pyre_object::w_module_dict_new();
        let obj = function_new(w_code as *const (), "myfunc".to_string(), w_globals);
        let w_qualname = pyre_object::w_str_new("pkg.myfunc");

        unsafe {
            function_set_qualname(obj, w_qualname);
            assert_eq!(function_get_qualname_obj(obj), w_qualname);
            assert_eq!(function_get_qualname_obj(obj), w_qualname);
        }
    }

    #[test]
    fn function_name_getter_preserves_the_stored_object() {
        crate::test_hooks::install_hash_hook();
        let raw_code = 0xDEAD_BEEF as *const ();
        let w_code = crate::w_code_new(raw_code);
        let w_globals = pyre_object::w_module_dict_new();
        let obj = function_new(w_code as *const (), "myfunc".to_string(), w_globals);
        let w_name = pyre_object::w_str_new("renamed");

        unsafe {
            function_set_func_name(obj, w_name);
            assert_eq!(function_get_name_obj(obj), w_name);
            assert_eq!(function_get_name_obj(obj), w_name);
            assert_eq!(function_get_name(obj), "renamed");
        }
    }

    #[test]
    fn test_function_field_offsets() {
        assert_eq!(FUNCTION_CODE_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
        assert_eq!(FUNCTION_NAME_OFFSET, 32); // after code(8) + can_change_code(1) + padding(7)
        assert_eq!(FUNCTION_W_NAME_OFFSET, 40); // after name
        assert_eq!(FUNCTION_CLOSURE_OFFSET, 48); // after w_name
    }

    /// Guard against drift between the constant colocated with
    /// `Function` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT/BuiltinCode
    /// trip-wire tests.
    #[test]
    fn function_gc_type_id_matches_descr() {
        assert_eq!(FUNCTION_GC_TYPE_ID, 14);
        assert_eq!(
            <Function as pyre_object::lltype::GcType>::type_id(),
            FUNCTION_GC_TYPE_ID
        );
        assert_eq!(
            <Function as pyre_object::lltype::GcType>::SIZE,
            std::mem::size_of::<Function>()
        );
    }

    /// `FUNCTION_GC_PTR_OFFSETS` must list every inline
    /// `PyObjectRef`-shaped field the GC traces (`code` is `*const ()`
    /// but points at a `[ob: PyObject, ...]`-prefixed Code object so
    /// the walker can interpret it as a typed reference). If a new GC
    /// field is added to `Function` (or one of these fields is removed)
    /// the array has to follow — this test makes the change a
    /// compile-time failure rather than a silent traversal gap.
    #[test]
    fn function_gc_ptr_offsets_cover_inline_pyobjectref_fields() {
        assert_eq!(
            FUNCTION_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(Function, code),
                std::mem::offset_of!(Function, name),
                std::mem::offset_of!(Function, w_name),
                std::mem::offset_of!(Function, closure),
                std::mem::offset_of!(Function, defs_w),
                std::mem::offset_of!(Function, w_kw_defs),
                std::mem::offset_of!(Function, w_module),
                std::mem::offset_of!(Function, w_func_globals_obj),
                std::mem::offset_of!(Function, w_builtins),
                std::mem::offset_of!(Function, w_ann),
                std::mem::offset_of!(Function, w_annotate),
                std::mem::offset_of!(Function, w_func_dict),
                std::mem::offset_of!(Function, w_typeparams),
                std::mem::offset_of!(Function, w_doc),
                std::mem::offset_of!(Function, w_qualname),
                std::mem::offset_of!(Function, w_objclass),
                std::mem::offset_of!(Function, w_text_signature),
                std::mem::offset_of!(Function, w_moduleobj),
                std::mem::offset_of!(Function, lifeline),
            ]
        );
    }
}
