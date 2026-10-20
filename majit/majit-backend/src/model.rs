//! Backend CPU abstraction per `rpython/jit/backend/model.py`.
//!
//! RPython's `AbstractCPU` (model.py+) hosts the services every
//! `Optimization` sub-class reaches via `self.optimizer.cpu.<method>()`:
//! `cls_of_box(box)` (model.py), `bh_*` runtime calls
//! (model.py:209+), GC type-info accessors, and so on.  Pyre currently
//! exposes only `cls_of_box` here; future expansion ports the rest of
//! the AbstractCPU surface onto the same trait so the carrier chain
//! `MetaInterp.cpu → UnrollOpt.cpu → Optimizer.cpu → OptContext.cpu`
//! threads a single trait object instead of an N-tuple of `fn` pointers.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::operand::Operand;

/// `llmodel.py` `AbstractLLCPU.subclassrange_min_offset`, beside
/// `vtable_offset`. `usize::MAX` means the CPU has not been configured.
static CPU_SUBCLASSRANGE_MIN_OFFSET: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Publish `cpu.subclassrange_min_offset`. The optimizer's
/// `_check_subclass` and the assembler read the same value.
pub fn set_cpu_subclassrange_min_offset(offset: Option<usize>) {
    CPU_SUBCLASSRANGE_MIN_OFFSET.store(offset.unwrap_or(usize::MAX), Ordering::Release);
}

/// `None` when [`set_cpu_subclassrange_min_offset`] has not been called.
pub fn cpu_subclassrange_min_offset() -> Option<usize> {
    match CPU_SUBCLASSRANGE_MIN_OFFSET.load(Ordering::Acquire) {
        usize::MAX => None,
        offset => Some(offset),
    }
}

/// `vtable_ptr.subclassrange_min` / `subclassrange_max`. The two `Signed`
/// fields are adjacent (`rclass.OBJECT_VTABLE`); `min_offset` is the first.
pub fn read_vtable_subclass_range(classptr: i64, min_offset: usize) -> (i64, i64) {
    if classptr == 0 {
        return (0, 0);
    }
    unsafe {
        let ptr = (classptr as usize).wrapping_add(min_offset) as *const i64;
        (ptr.read_unaligned(), ptr.add(1).read_unaligned())
    }
}
use majit_ir::{ArrayDescr, FieldDescr, GcRef, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeError;

/// `model.py AbstractCPU` (subset) — services hosted on
/// `optimizer.cpu` and reached from any `Optimization` sub-class.
pub trait Cpu: Send + Sync {
    /// `model.py cpu.cls_of_box(box)`:
    ///
    /// ```python
    /// def cls_of_box(self, box):
    ///     obj = lltype.cast_opaque_ptr(OBJECTPTR, box.getref_base())
    ///     return ConstInt(ptr2int(obj.typeptr))
    /// ```
    ///
    /// Unwraps the box's Ref payload and reads its class with
    /// `bh_classof`.  Returns 0 when the box does not carry a concrete
    /// `Value::Ref` or when the Ref is null.  A backend with a different
    /// object model overrides `bh_classof`, not this.
    fn cls_of_box(&self, box_: &Operand) -> i64 {
        // `AbstractValue.get_box_replacement` (`resoperation.py`) walker, then
        // read the concrete ref off the resolved operand. `cls_of_box`
        // (`llsupport/model.py`) reads `box.getref_base().typeptr`
        // reads any ref-carrying box, so a bound `Op` / `InputArg` with a
        // stamped `Value::Ref` must resolve too — `get_value()`, not the
        // const-only `const_value()`.
        match box_.get_box_replacement(false).get_value() {
            Some(Value::Ref(gcref)) if !gcref.is_null() => self.bh_classof(gcref),
            _ => 0,
        }
    }

    /// `llmodel.py AbstractLLCPU.bh_classof`:
    ///
    /// ```python
    /// def bh_classof(self, struct):
    ///     struct = lltype.cast_opaque_ptr(rclass.OBJECTPTR, struct)
    ///     return ptr2int(struct.typeptr)
    /// ```
    ///
    /// Raw typeptr read of a `GcRef`. `cls_of_box` unwraps the box and
    /// delegates here. Backends with a different object model override this.
    fn bh_classof(&self, gcref: GcRef) -> i64;

    /// `model.py+ cpu.bh_getfield_gc_i / _r / _f`:
    /// `llmodel.py read_int_at_mem / read_ref_at_mem / read_float_at_mem`.
    /// Read the field at `struct_ptr + fielddescr.offset()` honoring
    /// `field_size` + `is_field_signed`. The pure-getfield constant
    /// folder (`executor::execute_nonspec_const`) calls these after
    /// `protect_speculative_field` has validated that `struct_ptr` is
    /// non-null and of the expected type.
    fn bh_getfield_gc_i(&self, struct_ptr: usize, fielddescr: &dyn FieldDescr) -> i64;
    fn bh_getfield_gc_r(&self, struct_ptr: usize, fielddescr: &dyn FieldDescr) -> GcRef;
    fn bh_getfield_gc_f(&self, struct_ptr: usize, fielddescr: &dyn FieldDescr) -> f64;

    /// `llmodel.py protect_speculative_field`. Line-by-line:
    ///
    /// ```python
    /// def protect_speculative_field(self, gcptr, fielddescr):
    ///     if not gcptr:
    ///         raise SpeculativeError
    ///     if self.supports_guard_gc_type:
    ///         assert isinstance(fielddescr, FieldDescr)
    ///         sizedescr = fielddescr.parent_descr
    ///         if sizedescr.is_object():
    ///             if (not self.check_is_object(gcptr) or
    ///                 not sizedescr.is_valid_class_for(gcptr)):
    ///                 raise SpeculativeError
    ///         else:
    ///             if self.get_actual_typeid(gcptr) != sizedescr.tid:
    ///                 raise SpeculativeError
    /// ```
    ///
    /// `is_valid_class_for(gcptr)` (`descr.py`) compares the
    /// runtime typeptr's `subclassrange_min` against the descr's
    /// vtable's `[subclassrange_min, subclassrange_max]` inclusive
    /// interval.  Pyre routes both lookups through
    /// `majit_gc::subclass_range`.
    ///
    /// When `supports_guard_gc_type == False` (the boehm-style pyre
    /// default) only the null check fires, matching `llmodel.py`.
    ///
    /// **Fail-closed under `supports_guard_gc_type == true`:** PyPy
    /// asserts the field descr has a `parent_descr` and reaches into
    /// it unconditionally; if pyre cannot resolve the same metadata
    /// (parent_descr missing, vtable subclassrange unknown, typeid
    /// unresolvable, downcast fails), the speculative check fails
    /// closed so the fold caller declines.  Matching PyPy's "raise
    /// SpeculativeError" on any path where the type-validity verdict
    /// cannot be produced.  The `Err` names which of those paths fired, so
    /// the `InvalidLoop` the caller signals says more than that one did.
    fn protect_speculative_field(
        &self,
        gcptr: GcRef,
        fielddescr: &dyn FieldDescr,
    ) -> Result<(), &'static str> {
        if gcptr.is_null() {
            return Err("protect_speculative_field: null gcptr");
        }
        if !majit_gc::supports_guard_gc_type() {
            return Ok(());
        }
        // Header words (`w_class`, typeptr) are not in the parent's
        // field list (`heaptracker.py` skips them), so they carry no
        // `parent_descr`. They sit on every PyObject; the only
        // speculative check that applies is "this is an object".
        if fielddescr.is_w_class() || fielddescr.is_typeptr() {
            if !majit_gc::check_is_object(gcptr) {
                return Err("protect_speculative_field: gcptr is not an object");
            }
            return Ok(());
        }
        let parent = fielddescr
            .get_parent_descr()
            .ok_or("protect_speculative_field: field descr has no parent_descr")?;
        let sizedescr = parent
            .as_size_descr()
            .ok_or("protect_speculative_field: parent_descr is not a size descr")?;
        // A raw or headerless struct has no GC type-id word to validate.
        // RPython reaches this shape through raw field operations; pyre can
        // still carry the pointer in the Ref bank, so its GETFIELD_GC spelling
        // must consult the descr's storage shape before asking the collector
        // for a header that does not exist.
        if !sizedescr.is_gc_managed() || sizedescr.headerless() {
            return Ok(());
        }
        if sizedescr.is_object() {
            if !majit_gc::check_is_object(gcptr) {
                return Err("protect_speculative_field: gcptr is not an object");
            }
            // descr.py is_valid_class_for — subclassrange
            // containment of gcref's typeptr inside sizedescr.vtable's
            // range.
            let (expected_min, expected_max) = majit_gc::subclass_range(sizedescr.vtable())
                .ok_or("protect_speculative_field: descr vtable has no range")?;
            let actual_vtable = self.bh_classof(gcptr);
            if actual_vtable == 0 {
                return Err("protect_speculative_field: gcptr has no class");
            }
            let (actual_min, _) = majit_gc::subclass_range(actual_vtable as usize)
                .ok_or("protect_speculative_field: gcptr class has no range")?;
            if !(expected_min <= actual_min && actual_min <= expected_max) {
                return Err("protect_speculative_field: class outside the descr's subclass range");
            }
        } else {
            let actual_tid = majit_gc::get_actual_typeid(gcptr)
                .ok_or("protect_speculative_field: gcptr has no typeid")?;
            if actual_tid != sizedescr.type_id() {
                return Err("protect_speculative_field: typeid does not match the descr");
            }
        }
        Ok(())
    }

    /// `llmodel.py protect_speculative_array`. Line-by-line:
    ///
    /// ```python
    /// def protect_speculative_array(self, gcptr, arraydescr):
    ///     if not gcptr:
    ///         raise SpeculativeError
    ///     if self.supports_guard_gc_type:
    ///         assert isinstance(arraydescr, ArrayDescr)
    ///         if self.get_actual_typeid(gcptr) != arraydescr.tid:
    ///             raise SpeculativeError
    /// ```
    ///
    /// **Fail-closed under `supports_guard_gc_type == true`:**
    /// if `get_actual_typeid(gcptr)` returns `None` (no GC type info
    /// for this gcref), the check fails closed — matching PyPy's
    /// behavior where `None != arraydescr.tid` triggers
    /// SpeculativeError.
    fn protect_speculative_array(
        &self,
        gcptr: GcRef,
        arraydescr: &dyn ArrayDescr,
    ) -> Result<(), SpeculativeError> {
        if gcptr.is_null() {
            return Err(SpeculativeError);
        }
        if !majit_gc::supports_guard_gc_type() {
            return Ok(());
        }
        let actual_tid = majit_gc::get_actual_typeid(gcptr).ok_or(SpeculativeError)?;
        if actual_tid != arraydescr.type_id() {
            return Err(SpeculativeError);
        }
        Ok(())
    }

    /// `llmodel.py protect_speculative_string`:
    ///
    /// ```python
    /// def protect_speculative_string(self, gcptr):
    ///     self.protect_speculative_array(gcptr, self.gc_ll_descr.str_descr)
    /// ```
    ///
    /// PyPy delegates to `protect_speculative_array` with
    /// `gc_ll_descr.str_descr` cached at backend init.  Pyre routes
    /// through the same delegate when a backend has registered its
    /// string layout via `str_descr()`.  Backends without a typed
    /// string descr (the trait default) fail closed under
    /// `supports_guard_gc_type == true` — the upstream
    /// `protect_speculative_operation` gate already skips the
    /// string/unicode branch when `supports_guard_gc_type == false`
    /// (`mod.rs:5430` "we don't unroll in that case" port), so only
    /// the null check is reachable in that mode.
    fn protect_speculative_string(&self, gcptr: GcRef) -> Result<(), SpeculativeError> {
        if gcptr.is_null() {
            return Err(SpeculativeError);
        }
        if !majit_gc::supports_guard_gc_type() {
            return Ok(());
        }
        match self.str_descr() {
            Some(d) => self.protect_speculative_array(gcptr, d),
            None => Err(SpeculativeError),
        }
    }

    /// `llmodel.py protect_speculative_unicode`.  Mirror of
    /// `protect_speculative_string` for unicode storage; routes
    /// through `unicode_descr()` when registered.
    fn protect_speculative_unicode(&self, gcptr: GcRef) -> Result<(), SpeculativeError> {
        if gcptr.is_null() {
            return Err(SpeculativeError);
        }
        if !majit_gc::supports_guard_gc_type() {
            return Ok(());
        }
        match self.unicode_descr() {
            Some(d) => self.protect_speculative_array(gcptr, d),
            None => Err(SpeculativeError),
        }
    }

    /// `llmodel.py AbstractLLCPU.protect_speculative_string gc_ll_descr.str_descr` — the typed `ArrayDescr`
    /// for the runtime string layout (basesize + length offset + char
    /// item size + `STR` type id).  Cached on the gc_ll_descr at
    /// backend init upstream; pyre exposes it on the `Cpu` trait so
    /// pyre-side bootstrap can register the concrete string layout
    /// (e.g. `W_UnicodeObject` / `W_BytesObject`) once and route
    /// `protect_speculative_string` through the same typed descr. The
    /// default `bh_strlen` / `bh_strgetitem` also use this descr; backends
    /// whose physical string storage is not inline can override those
    /// helpers while still reusing the same speculative type check.
    ///
    /// Default returns `None` — backends without a typed string
    /// layout opt out, and the four downstream helpers fall back to
    /// their fail-closed / `None` shape.
    fn str_descr(&self) -> Option<&dyn ArrayDescr> {
        None
    }

    /// `llmodel.py AbstractLLCPU.protect_speculative_unicode gc_ll_descr.unicode_descr` — mirror of
    /// `str_descr()` for the unicode layout.  Default `None`.
    fn unicode_descr(&self) -> Option<&dyn ArrayDescr> {
        None
    }

    /// `model.py+ cpu.bh_arraylen_gc` /
    /// `llmodel.py read_int_at_mem(array, lendescr.offset, WORD, 1)`.
    /// Default impl reads i64 at `arraydescr.len_descr().offset()`.
    /// Returns `None` when no `len_descr` is registered (matches the
    /// `assert isinstance(arraydescr, ArrayDescr)` failure mode upstream
    /// would hit with a misconfigured descr).
    fn bh_arraylen_gc(&self, array: GcRef, arraydescr: &dyn ArrayDescr) -> Option<i64> {
        let lendescr = arraydescr.len_descr()?;
        let addr = array.0 + lendescr.offset();
        // SAFETY: caller has guaranteed `array` is a valid array gcref
        // and `lendescr.offset()` is the registered length field offset.
        // The length is a machine-word (`Signed`/`WORD`) field
        // (`llmodel.py:587`): read it at `usize` width so a 32-bit target
        // (`WORD == 4`) does not pull the adjacent field into the high
        // half. On 64-bit this is identical to the previous `i64` read.
        Some(unsafe { *(addr as *const usize) as i64 })
    }

    /// `model.py+ cpu.bh_strlen` /
    /// `llmodel.py read_int_at_mem(string, str_descr.lendescr.offset, WORD, 1)`.
    /// Routes through `str_descr()` → `bh_arraylen_gc` so any backend
    /// that registered a typed string layout reaches the same length
    /// read as the rest of the array family.  Backends without a
    /// registered `str_descr()` return `None`, declining the fold.
    fn bh_strlen(&self, string: GcRef) -> Option<i64> {
        let descr = self.str_descr()?;
        self.bh_arraylen_gc(string, descr)
    }

    /// `llmodel.py:594-595` mirror for unicode.  Routes through
    /// `unicode_descr()` → `bh_arraylen_gc`.
    fn bh_unicodelen(&self, unicode: GcRef) -> Option<i64> {
        let descr = self.unicode_descr()?;
        self.bh_arraylen_gc(unicode, descr)
    }

    /// `model.py+ cpu.bh_strgetitem` /
    /// `llmodel.py read_int_at_mem(string, basesize + index, 1, 0)`.
    /// Routes through `str_descr()` → `bh_getarrayitem_gc_i` so any
    /// backend that registered a typed string layout reaches the same
    /// per-character read as the rest of the array family.  Backends
    /// whose physical char data is not in-line at `base + index *
    /// item_size` (e.g. pyre's `W_UnicodeObject`, whose chars sit behind a
    /// `*mut String` indirection) must override.
    fn bh_strgetitem(&self, string: GcRef, index: i64) -> Option<i64> {
        let descr = self.str_descr()?;
        self.bh_getarrayitem_gc_i(string, index, descr)
    }

    /// `llmodel.py:609-612` mirror for unicode.  Routes through
    /// `unicode_descr()` → `bh_getarrayitem_gc_i`.  Same override
    /// requirement as `bh_strgetitem` for indirected char-data
    /// layouts.
    fn bh_unicodegetitem(&self, unicode: GcRef, index: i64) -> Option<i64> {
        let descr = self.unicode_descr()?;
        self.bh_getarrayitem_gc_i(unicode, index, descr)
    }

    /// `model.py+ cpu.bh_getarrayitem_gc_i` /
    /// `llmodel.py read_int_at_mem(gcref, ofs + index * size, size, sign)`.
    /// Default impl reads the int item at
    /// `array + ad.base_size() + index * ad.item_size()`, dispatching on
    /// `item_size` × `is_item_signed`.  `index` is assumed to be in
    /// bounds — `protect_speculative_operation` validates first.
    fn bh_getarrayitem_gc_i(&self, array: GcRef, index: i64, ad: &dyn ArrayDescr) -> Option<i64> {
        let addr = array.0 + ad.base_size() + (index as usize) * ad.item_size();
        match (ad.item_size(), ad.is_item_signed()) {
            (8, true) => Some(unsafe { *(addr as *const i64) }),
            (8, false) => Some(unsafe { *(addr as *const u64) as i64 }),
            (4, true) => Some(unsafe { *(addr as *const i32) as i64 }),
            (4, false) => Some(unsafe { *(addr as *const u32) as i64 }),
            (2, true) => Some(unsafe { *(addr as *const i16) as i64 }),
            (2, false) => Some(unsafe { *(addr as *const u16) as i64 }),
            (1, true) => Some(unsafe { *(addr as *const i8) as i64 }),
            (1, false) => Some(unsafe { *(addr as *const u8) as i64 }),
            // llmodel.py:478 `else: raise NotImplementedError(...)` —
            // mirror with `None` so the fold caller skips and the op
            // is emitted verbatim.
            _ => None,
        }
    }

    /// `model.py+ cpu.bh_getarrayitem_gc_r` /
    /// `llmodel.py read_ref_at_mem(gcref, index * WORD + ofs)`.
    fn bh_getarrayitem_gc_r(&self, array: GcRef, index: i64, ad: &dyn ArrayDescr) -> GcRef {
        let addr = array.0 + ad.base_size() + (index as usize) * ad.item_size();
        GcRef(unsafe { *(addr as *const usize) })
    }

    /// `model.py+ cpu.bh_getarrayitem_gc_f` /
    /// `llmodel.py read_float_at_mem(gcref, index * FLOATSTORAGE + ofs)`.
    fn bh_getarrayitem_gc_f(&self, array: GcRef, index: i64, ad: &dyn ArrayDescr) -> f64 {
        let addr = array.0 + ad.base_size() + (index as usize) * ad.item_size();
        let bits = unsafe { *(addr as *const u64) };
        f64::from_bits(bits)
    }
}

/// Default `Cpu` implementing `cls_of_box` against the lltype-typeptr-
/// at-offset-0 layout (model.py).  Production paths that did
/// not install a custom backend hook fall through to this.
pub struct DefaultCpu;

impl Cpu for DefaultCpu {
    fn bh_classof(&self, gcref: GcRef) -> i64 {
        if gcref.is_null() {
            return 0;
        }
        if majit_gc::is_tagged_immediate(gcref.as_usize()) {
            // A tagged immediate has no object header to read at offset 0; report
            // "no class" (0) so the optimizer keeps the runtime type guard instead
            // of dereferencing. Inert when taggedpointers is disabled.
            return 0;
        }
        // SAFETY: caller has guaranteed `gcref` is a valid OBJECTPTR
        // payload pointer; the lltype OBJECTPTR layout has the typeptr
        // at offset 0 (model.py `box.getref_base().typeptr`).
        unsafe { *(gcref.0 as *const usize) as i64 }
    }

    fn bh_getfield_gc_i(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> i64 {
        // llmodel.py read_int_at_mem signed/unsigned width
        // dispatch. RPython's loop falls through to `else: raise
        // NotImplementedError("size = %d" % size)` when no `itemsize`
        // matches; mirror that with a panic. Callers that may receive
        // exotic field sizes (e.g. the trace-time fold path) MUST
        // pre-filter via `fd.field_size()` before invoking this method.
        // `load_is_acquire` selects the Acquire arm of `read_int_at_mem`.
        unsafe {
            crate::llmodel::read_int_at_mem(
                struct_ptr,
                fd.offset(),
                fd.field_size(),
                fd.is_field_signed(),
                fd.load_is_acquire(),
            )
        }
    }

    fn bh_getfield_gc_r(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> GcRef {
        let addr = struct_ptr + fd.offset();
        // `llmodel.py bh_getfield_gc_r` / `read_ref_at_mem`: pointer-width
        // value load. `rewrite_op_getsubstruct` handles address projections
        // before execution; an inline layout alone does not request one.
        // `load_is_acquire` reads the same slot with Acquire.
        GcRef(unsafe { crate::llmodel::read_ref_at_mem(addr, fd.load_is_acquire()) })
    }

    fn bh_getfield_gc_f(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> f64 {
        // Size 8 is llmodel.py FLOATSTORAGE; size 4 widens f32.
        unsafe { crate::llmodel::read_float_at_mem_sized(struct_ptr, fd.offset(), fd.field_size()) }
    }
}

/// `Arc<dyn Cpu>` whose `bh_classof` is a bare `fn(i64) -> i64` over the
/// raw Ref payload; `cls_of_box` reaches it through the provided method.
pub fn cpu_from_bh_classof_fn(f: fn(i64) -> i64) -> Arc<dyn Cpu> {
    struct ClosureCpu(fn(i64) -> i64);
    impl Cpu for ClosureCpu {
        fn bh_classof(&self, gcref: GcRef) -> i64 {
            if majit_gc::is_tagged_immediate(gcref.as_usize()) {
                // A tagged immediate has no object header to read at offset 0; report
                // "no class" (0) so the optimizer keeps the runtime type guard instead
                // of dereferencing. Inert when taggedpointers is disabled.
                return 0;
            }
            (self.0)(gcref.0 as i64)
        }
        fn bh_getfield_gc_i(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> i64 {
            DefaultCpu.bh_getfield_gc_i(struct_ptr, fd)
        }
        fn bh_getfield_gc_r(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> GcRef {
            DefaultCpu.bh_getfield_gc_r(struct_ptr, fd)
        }
        fn bh_getfield_gc_f(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> f64 {
            DefaultCpu.bh_getfield_gc_f(struct_ptr, fd)
        }
    }
    Arc::new(ClosureCpu(f))
}

/// `Arc<dyn Cpu>` to the default lltype backend, for production paths
/// + tests that want the model.py typeptr-at-offset-0 read.
pub fn default_cpu() -> Arc<dyn Cpu> {
    Arc::new(DefaultCpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_w_globals_load_reads_the_published_pointer() {
        let published = 0xABC0usize;
        let mut slot = 0usize;
        unsafe {
            std::sync::atomic::AtomicPtr::<()>::from_ptr(&mut slot as *mut usize as *mut *mut ())
                .store(published as *mut (), std::sync::atomic::Ordering::Release);
        }
        let fd = majit_ir::descr::SimpleFieldDescr::new_with_name(
            0,
            0,
            std::mem::size_of::<usize>(),
            majit_ir::Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Pointer,
            "PyCode.w_globals".to_string(),
            "w_globals",
        )
        .with_quasi_immutable(true);
        assert!(fd.load_is_acquire());
        assert_eq!(
            DefaultCpu.bh_getfield_gc_r(&mut slot as *mut usize as usize, &fd),
            GcRef(published)
        );

        let plain = majit_ir::descr::SimpleFieldDescr::new_with_name(
            0,
            0,
            std::mem::size_of::<usize>(),
            majit_ir::Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Pointer,
            "PyCode.co_consts".to_string(),
            "co_consts",
        );
        assert!(!plain.load_is_acquire());
        assert_eq!(
            DefaultCpu.bh_getfield_gc_r(&mut slot as *mut usize as usize, &plain),
            GcRef(published)
        );
    }

    #[test]
    fn acquire_integer_field_load_reads_the_published_value() {
        let published: u64 = 0x1111_2222_3333_4444;
        let mut slot: u64 = 0;
        unsafe {
            std::sync::atomic::AtomicU64::from_ptr(&mut slot as *mut u64)
                .store(published, std::sync::atomic::Ordering::Release);
        }
        let fd = majit_ir::descr::SimpleFieldDescr::new_with_name(
            0,
            0,
            8,
            majit_ir::Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "W_TypeObject.version_tag".to_string(),
            "version_tag",
        )
        .with_load_is_acquire(true);
        assert!(fd.load_is_acquire());
        assert_eq!(
            DefaultCpu.bh_getfield_gc_i(&mut slot as *mut u64 as usize, &fd),
            published as i64
        );
    }

    #[test]
    fn reference_value_read_does_not_become_a_substructure_address() {
        let referent = 123usize;
        let field_words = [0usize, &referent as *const usize as usize];
        for flag in [
            majit_ir::descr::ArrayFlag::Pointer,
            majit_ir::descr::ArrayFlag::Struct,
        ] {
            let fd = majit_ir::descr::SimpleFieldDescr::new_with_name(
                0,
                std::mem::size_of::<usize>(),
                std::mem::size_of::<usize>(),
                majit_ir::Type::Ref,
                false,
                flag,
                "Node.value".to_string(),
                "value",
            );
            assert_eq!(
                DefaultCpu.bh_getfield_gc_r(field_words.as_ptr() as usize, &fd),
                GcRef(field_words[1])
            );
        }
    }

    /// `AbstractCPU.cls_of_box` unwraps the box; `AbstractLLCPU.bh_classof`
    /// reads `struct.typeptr`. Both entries on one object return that class.
    #[test]
    fn cls_of_box_and_bh_classof_agree_on_one_object() {
        let class_word = 0xC1A5_u64;
        let object = [class_word as usize];
        let gcref = GcRef(object.as_ptr() as usize);
        let box_ = Operand::const_from_value(Value::Ref(gcref));
        let cpu = DefaultCpu;
        assert_eq!(cpu.cls_of_box(&box_), class_word as i64);
        assert_eq!(cpu.bh_classof(gcref), class_word as i64);
    }
}
