use crate::optimizeopt::intutils::{IntBound, IntBoundMakeGuards};
pub use crate::optimizeopt::rawbuffer::{
    InvalidRawOperation, InvalidRawRead, InvalidRawWrite, RawBuffer,
};
/// Abstract information attached to operations during optimization.
///
/// Translated from rpython/jit/metainterp/optimizeopt/info.py.
/// Each operation can have associated analysis info (e.g., known integer bounds,
/// pointer info, virtual object state).
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, FieldDescr, GcRef, Op, OpCode, OpRc, OpRef, Type, Value};

fn lookup_field_descr(field_descrs: &[DescrRef], field_idx: u32) -> Option<DescrRef> {
    field_descrs.get(field_idx as usize).cloned()
}

/// `info.py _force_elements` walks `descr.get_all_fielddescrs()` by
/// enumerate index. A leftover sparse key is a numbering that is not
/// that list; name both the leftover and the list it failed to index.
fn force_box_field_keys(field_descrs: &[DescrRef]) -> Vec<(String, usize, u32)> {
    field_descrs
        .iter()
        .map(|d| {
            d.as_field_descr()
                .map(|f| (f.field_key().to_string(), f.index_in_parent(), d.index()))
                .unwrap_or_else(|| ("?".to_string(), 0, d.index()))
        })
        .collect()
}

fn force_box_size_identity(descr: &DescrRef) -> String {
    match descr.as_size_descr() {
        Some(sd) => format!(
            "index={} cache_key={} type_id={} size={} nfields={}",
            descr.index(),
            sd.cache_key(),
            sd.type_id(),
            sd.size(),
            sd.all_fielddescrs().len(),
        ),
        None => format!("index={} not-size", descr.index()),
    }
}

/// The dense runtime type id a `GUARD_GC_TYPE` may be built against, or `None`
/// when the descr cannot name one.
///
/// A stamped 0 is never a legitimate tid: the allocator hands out identities
/// starting at 1 (`gctypelayout.py` "don't use typeid 0"), so 0
/// means this descr was rehydrated from a serialized form and never stamped.
/// It is, however, a live value in a runtime header — the `rclass.OBJECT` root
/// — which makes `GUARD_GC_TYPE(x, 0)` wrong in both directions: it fails on
/// every object carrying a real header, and it *passes* on a plain `object`,
/// certifying a layout nothing ever named. Emitting no guard at all is worse
/// still, since it is the only layout check on the short-preamble entry. So the
/// two admissible outcomes are the real tid or a decline, never 0 and never a
/// silent skip.
///
/// The tid is recoverable because optimization runs inside the live interpreter,
/// after `gc.register_type` has handed the runtime tids out — the build-time
/// analyzer that wrote the serialized descr could not see them, but the
/// structural `cache_key` still resolves to the dense tid through the same GC
/// cache the backends use (`BhDescr::resolve_gc_tid`).
///
/// That recovery answers for the id the JIT's own allocations write. It is the
/// same id the runtime header holds only while the JIT is what allocates the
/// struct: a struct the host allocates under an id of its own resolves here to
/// the cache's, and the guard then fails on every object. A descr for such a
/// struct has to say so with `headerless`, which is checked before this is
/// called, rather than by leaving its `type_id` at 0.
pub(crate) fn resolve_gc_tid(
    stamped_tid: u32,
    cache_key: u64,
    resolve: impl FnOnce(&majit_ir::descr::GcCache, u64) -> Option<u32>,
) -> Option<u32> {
    if stamped_tid != 0 && !majit_ir::descr::array_tid_is_unresolved(stamped_tid) {
        return Some(stamped_tid);
    }
    // Zero is the no-STRUCT-identity sentinel, not a key: resolving it would
    // certify this descr against whichever group was published under the
    // sentinel, and the guard would then pin a layout nothing here has.
    // `UNSET_GC_TYPE_ID` is the array mint placeholder until
    // `register_unresolved_array_tids` (`gc.py` `init_array_descr`).
    if cache_key == 0 {
        return None;
    }
    let cache = majit_ir::descr::gc_cache().lock();
    resolve(&cache, cache_key).filter(|&tid| !majit_ir::descr::array_tid_is_unresolved(tid))
}

/// Whether allocation lowering already writes exactly the `w_class` value
/// represented by this virtual field.
///
/// RPython's `AbstractStructPtrInfo._force_elements` only visits
/// `descr.get_all_fielddescrs()` (info.py), where `typeptr` is absent,
/// so upstream never emits this header store from the force path. This is the
/// same optimizer-layer "value already there" elision as heap.py:88-101.
pub(crate) fn w_class_value_is_covered_by_alloc(
    size_descr: &DescrRef,
    field_descr: &DescrRef,
    value: Option<Value>,
) -> bool {
    let Some(field) = field_descr.as_field_descr().filter(|fd| fd.is_w_class()) else {
        return false;
    };
    let Some(size) = size_descr.as_size_descr() else {
        return false;
    };
    let Some(w_class) = size.w_class_obj().filter(|&w| w != 0) else {
        return false;
    };
    let Some(init_field) = size.class_word_field() else {
        return false;
    };
    // Keep this comparison even though the slot is now declared. The
    // declared slot replaces the *name* test and nothing else. `field` comes
    // from the operation and `init_field` from the layout, and pyre mints
    // those from two independent producers, so they are distinct objects that
    // may or may not describe the same bytes. This is the check that they do;
    // removing it would let a store to any field of a class-word-bearing
    // struct be elided as "already written by the allocation".
    if init_field.offset() != field.offset() || init_field.field_size() != field.field_size() {
        return false;
    }
    matches!(
        value,
        Some(Value::Ref(value)) if value == GcRef(w_class as usize)
    )
}

fn w_class_store_is_covered_by_alloc(
    size_descr: &DescrRef,
    field_descr: &DescrRef,
    value: &Operand,
    ctx: &crate::optimizeopt::OptContext,
) -> bool {
    w_class_value_is_covered_by_alloc(
        size_descr,
        field_descr,
        ctx.resolve_operand_operand_opt(value)
            .and_then(|resolved| resolved.const_value()),
    )
}

/// Cache the Python-level class word that `NEW_WITH_VTABLE` initializes.
///
/// PyPy keeps the allocation-owned class fact on `InstancePtrInfo` as
/// `_known_class` (`info.py` `InstancePtrInfo.__init__`).  Pyre additionally
/// has the public `PyObject.w_class` header, so the same allocation fact must
/// be visible through that field's disjoint heap slot as well.  Keeping it on
/// the box's `PtrInfo` lets `OptHeap` fold a short-preamble GETFIELD after a
/// virtual is forced, while a later explicit class-word SETFIELD still
/// overwrites this entry normally (required for user-subclass instances).
fn allocation_initialized_w_class(
    size_descr: &DescrRef,
) -> Option<(u32, majit_ir::field_entry::FieldEntry)> {
    let size = size_descr.as_size_descr()?;
    let w_class = size.w_class_obj().filter(|&w| w != 0)?;
    let field = size.class_word_field()?;
    let slot = crate::optimizeopt::heap::OptHeap::header_field_slot(field.as_ref());
    Some((
        slot,
        majit_ir::field_entry::FieldEntry::Value(Operand::const_from_value(Value::Ref(GcRef(
            w_class as usize,
        )))),
    ))
}

pub use majit_ir::field_entry::{FieldEntry, FieldEntryKind, PreambleOp};
pub use majit_ir::op_info::{EmptyInfo, FloatConstInfo, OpInfo};
pub use majit_ir::ptr_info::reasonable_array_index;
pub use majit_ir::ptr_info::{PtrInfo, StrPtrInfo};

pub use majit_ir::ptr_info::{
    CachedFieldList, StringConstantAllocator, StringContentResolver, StringLengthResolver,
    VirtualFieldList,
};

/// Result of `OptContext::ensure_ptr_info_arg0(op)` — direct line-by-line
/// equivalent of PyPy's `ensure_ptr_info_arg0` return value
/// (`optimizer.py`).
///
/// PyPy returns a Python `PtrInfo` object that the caller invokes methods on
/// (`structinfo.setfield(...)`, `arrayinfo.getlenbound(None).make_gt_const(...)`).
/// The Rust port can't expose `&mut PtrInfo` directly when the arg0 is a
/// constant — there's no `Forwarded::Info` slot to borrow from — so the enum
/// distinguishes the two cases:
///
/// - **`Constant { gcref, .. }`** — `arg0.is_constant()`
///   (`optimizer.py`). PyPy returns a freshly-constructed
///   `info.ConstPtrInfo(arg0)`. The Rust variant carries the resolved
///   `GcRef` so methods like `getlenbound` can synthesize the same answer
///   on demand. The optional `string_length_resolver` Arc allows
///   `getlenbound(Some(mode))` to return an exact constant length when the
///   runtime can read the underlying string object — matching PyPy's
///   `getstrlen1(mode)` path through `_unpack_str(mode)`.
///
/// - **`Forwarded(&mut PtrInfo)`** — `arg0.get_forwarded()` returns either an
///   existing `AbstractVirtualPtrInfo` subclass (early-return path) or a
///   freshly-installed Instance/Struct/Array/Str etc. (`optimizer.py`).
///   The mutable reference is backed by the bound `Op`/`InputArg`'s
///   `_forwarded` slot, so `info.setfield()` / `info.setitem()` mutate the
///   canonical PtrInfo in-place — matching PyPy's
///   `arg0.set_forwarded(opinfo)` followed by `opinfo.setfield(...)`.
pub enum EnsuredPtrInfo {
    /// `info.ConstPtrInfo(arg0)` — synthesized from a constant Ref / raw-pointer
    /// Int OpRef. Read-only by construction.
    Constant {
        gcref: GcRef,
        /// Optional runtime hook for `getstrlen1(mode)` lookups.
        string_length_resolver: Option<StringLengthResolver>,
    },
    /// `arg0.get_forwarded()` — operand-routed mutable handle. Each
    /// `as_mut()` call re-borrows the inner `RefCell`. Produced when the
    /// opref resolves to a bound `Op`/`InputArg`.
    Forwarded(majit_ir::operand::Operand),
}

impl EnsuredPtrInfo {
    /// `info.py PtrInfo.getlenbound(mode)` — direct delegation to the underlying
    /// PtrInfo. For `Constant` the call routes through the optional
    /// `string_length_resolver` so an exact constant length can be returned
    /// when the runtime knows it (PyPy `ConstPtrInfo.getlenbound` →
    /// `getstrlen1(mode)` → `_unpack_str(mode)` at info.py).
    pub fn getlenbound(&mut self, mode: Option<u8>) -> Option<IntBound> {
        match self {
            EnsuredPtrInfo::Constant {
                gcref,
                string_length_resolver,
            } => {
                // info.py ConstPtrInfo.getlenbound(mode):
                //
                //     def getlenbound(self, mode):
                //         length = self.getstrlen1(mode)
                //         if length < 0:
                //             return IntBound.nonnegative()
                //         return IntBound.from_constant(length)
                //
                // info.py ConstPtrInfo.getstrlen1(mode):
                //
                //     def getstrlen1(self, mode):
                //         if mode is vstring.mode_string:    ...
                //         elif mode is vstring.mode_unicode: ...
                //         else:
                //             return -1
                //
                // PyPy returns `IntBound.nonnegative()` regardless of
                // mode whenever `getstrlen1` cannot supply an exact
                // length. The Rust port mirrors that:
                //   * mode == None        → getstrlen1 returns -1 →
                //                           nonnegative()
                //   * mode == Some(0|1)   → resolver returns Some(len) →
                //                           from_constant(len);
                //                           else nonnegative()
                let length = match mode {
                    Some(mode_value) => {
                        if gcref.is_null() {
                            -1
                        } else if let Some(resolver) = string_length_resolver.as_deref() {
                            resolver(*gcref, mode_value).unwrap_or(-1)
                        } else {
                            -1
                        }
                    }
                    // info.py:823-824 `else: return -1` for mode == None.
                    None => -1,
                };
                if length < 0 {
                    Some(IntBound::nonnegative())
                } else {
                    Some(IntBound::from_constant(length))
                }
            }
            EnsuredPtrInfo::Forwarded(o) => o.ptr_info_mut().and_then(|mut p| p.getlenbound(mode)),
        }
    }

    /// Mutable access to the underlying `PtrInfo`. Returns `None` for the
    /// `Constant` variant — PyPy's `ConstPtrInfo.setfield/setitem` route
    /// through `optheap.const_infos`, not through the constant box's own
    /// info slot (info.py). The `Forwarded` variant returns
    /// `None` if the operand's `_forwarded` slot does not currently hold
    /// `Forwarded::Info(OpInfo::Ptr(_))`. The returned guard owns an `Rc`
    /// clone of the live `Rc<RefCell<PtrInfo>>` cell and an exclusive
    /// `RefCell` borrow — drop it before any sibling write to the same
    /// box's `_forwarded` slot.
    pub fn as_mut(&mut self) -> Option<majit_ir::forwarding::PtrInfoBorrowMut> {
        match self {
            EnsuredPtrInfo::Constant { .. } => None,
            EnsuredPtrInfo::Forwarded(o) => o.ptr_info_mut(),
        }
    }

    /// Whether the helper produced a synthesized `ConstPtrInfo` rather than a
    /// real forwarded entry. Mirrors `isinstance(opinfo, ConstPtrInfo)` at
    /// the call site.
    pub fn is_constant(&self) -> bool {
        matches!(self, EnsuredPtrInfo::Constant { .. })
    }
}

pub use majit_ir::ptr_info::{
    VStringConcatInfo, VStringPlainInfo, VStringSliceInfo, VStringVariant,
};

/// Extension trait carrying the OptContext-coupled methods that used
/// to live on `impl StrPtrInfo` in metainterp. The data type itself
/// is in `majit-ir`; only methods that depend on metainterp-side
/// helpers (`get_constant_int_or_bound`, `get_box_replacement_operand_opt`,
/// `getptrinfo`) stay here as a trait.
pub trait StrPtrInfoExt {
    fn getstrlen(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64>;
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>>;
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<Operand>;
}

impl StrPtrInfoExt for StrPtrInfo {
    /// vstring.py/171/251/281 `getstrlen()` on the string ptrinfo classes.
    ///
    /// Returns the structurally-known constant length for virtual variants
    /// (Plain/Slice/Concat). For the non-virtual `Ptr` variant, returns
    /// `None` — RPython's base StrPtrInfo.getstrlen() (vstring.py)
    /// always emits STRLEN and attaches lenbound as metadata; it never
    /// extracts a constant from lenbound directly.
    fn getstrlen(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64> {
        // vstring.py:112: if self.lgtop is not None: return self.lgtop — the
        // length BOX. Callers fold only when it `isinstance(ConstInt)`, so a
        // length variable whose IntBound merely happens to be constant is NOT
        // a known constant length: read only an actual ConstInt, not the bound.
        if let Some(lgtop) = self.lgtop.as_ref() {
            return ctx
                .resolve_operand_operand_opt(lgtop)
                .and_then(|b| ctx.get_constant_int_box(&b));
        }
        match &self.variant {
            // vstring.py: base StrPtrInfo.getstrlen always emits
            // STRLEN and caches in lgtop; never returns a constant from
            // lenbound. The caller (getstrlen_opref) handles STRLEN emission.
            VStringVariant::Ptr => None,
            // vstring.py: VStringPlainInfo.getstrlen
            VStringVariant::Plain(info) => Some(info._chars.len() as i64),
            // vstring.py: VStringSliceInfo.getstrlen → self.lgtop;
            // constant only when it is an actual ConstInt (isinstance check).
            VStringVariant::Slice(info) => {
                let b = ctx.resolve_operand_operand_opt(&info.lgtop)?;
                ctx.get_constant_int_box(&b)
            }
            // vstring.py: VStringConcatInfo.getstrlen
            VStringVariant::Concat(info) => {
                let vleft_box = ctx.resolve_operand_operand_opt(&info.vleft);
                let vright_box = ctx.resolve_operand_operand_opt(&info.vright);
                let left = vleft_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let right = vright_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let len1 = left.get_known_str_length(ctx, mode)?;
                let len2 = right.get_known_str_length(ctx, mode)?;
                Some(len1 + len2)
            }
        }
    }

    /// vstring.py / 172 / 298 `get_constant_string_spec()` on the string
    /// ptrinfo classes.
    ///
    /// The upstream method returns either a low-level string or unicode object.
    /// majit keeps the same recursive shape but represents the constant string
    /// as character/codepoint integers until a runtime string allocator is
    /// wired in.
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>> {
        let _ = mode;
        match &self.variant {
            VStringVariant::Ptr => None,
            VStringVariant::Plain(info) => {
                let mut chars = Vec::with_capacity(info._chars.len());
                for ch in &info._chars {
                    let ch_box = ch.as_ref()?;
                    // vstring.py: `if c is None or not c.is_constant()`.
                    // RPython stores the resolved char box in `_chars`, so a
                    // bare `c.is_constant()` suffices. Pyre stores the operand's
                    // position box (which forwards to its Const), so resolving
                    // the chain first is the faithful equivalent of RPython's
                    // direct check. `const_int()` (not `get_constant_int_box`)
                    // keeps parity strict: a Const terminal folds, a box with
                    // only a constant IntBound does NOT (no synthesized
                    // ConstInt).
                    chars.push(
                        ctx.resolve_operand_operand_opt(ch_box)
                            .and_then(|cb| cb.const_int())?,
                    );
                }
                Some(chars)
            }
            VStringVariant::Slice(info) => {
                // vstring.py get_constant_string_spec: use getintbound().is_constant()
                let s_box = ctx.resolve_operand_operand_opt(&info.s);
                let source = s_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let source_chars = source.get_constant_string_spec(ctx, mode)?;
                let start_box = ctx.resolve_operand_operand_opt(&info.start)?;
                let start = usize::try_from(ctx.get_constant_int_or_bound_box(&start_box)?).ok()?;
                let lgtop_box = ctx.resolve_operand_operand_opt(&info.lgtop)?;
                let length =
                    usize::try_from(ctx.get_constant_int_or_bound_box(&lgtop_box)?).ok()?;
                let stop = start.checked_add(length)?;
                if stop > source_chars.len() {
                    return None;
                }
                Some(source_chars[start..stop].to_vec())
            }
            VStringVariant::Concat(info) => {
                let vleft_box = ctx.resolve_operand_operand_opt(&info.vleft);
                let vright_box = ctx.resolve_operand_operand_opt(&info.vright);
                let left = vleft_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let right = vright_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let mut chars = left.get_constant_string_spec(ctx, mode)?;
                chars.extend(right.get_constant_string_spec(ctx, mode)?);
                Some(chars)
            }
        }
    }

    /// vstring.py / 172 / 230 `strgetitem()` shape, collapsed into a single
    /// variant-dispatch method on the Rust side.
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<Operand> {
        let index = usize::try_from(index).ok()?;
        match &self.variant {
            VStringVariant::Ptr => None,
            VStringVariant::Plain(info) => info._chars.get(index).and_then(|o| o.clone()),
            VStringVariant::Slice(info) => {
                // vstring.py: index = _int_add(sinfo.start, index), then
                // the fold proceeds on `getintbound(index).is_constant()` — the
                // IntBound of the COMPUTED index, not `isinstance(index, ConstInt)`.
                // So a start whose IntBound is constant (even if the box is not a
                // literal ConstInt) still yields a constant index and folds. Read
                // the start via the intbound-aware accessor, not ConstInt-only.
                let start_box = ctx.resolve_operand_operand_opt(&info.start)?;
                let start = ctx.get_constant_int_or_bound_box(&start_box)?;
                let s_box = ctx.resolve_operand_operand_opt(&info.s);
                let source = s_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                source.strgetitem(index as i64 + start, ctx)
            }
            VStringVariant::Concat(info) => {
                let vleft_box = ctx.resolve_operand_operand_opt(&info.vleft);
                let left = vleft_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                let left_len = usize::try_from(left.get_known_str_length(ctx, self.mode)?).ok()?;
                if index < left_len {
                    left.strgetitem(index as i64, ctx)
                } else {
                    let vright_box = ctx.resolve_operand_operand_opt(&info.vright);
                    let right = vright_box.as_ref().and_then(|b| ctx.getptrinfo(b))?;
                    right.strgetitem((index - left_len) as i64, ctx)
                }
            }
        }
    }
}

/// Extension trait carrying the `OptContext` / `majit_gc` /
/// `VirtualVisitor`-coupled methods that used to live on `impl PtrInfo`.
/// The data type and pure-leaf methods live in `majit-ir::ptr_info`; only
/// methods that depend on metainterp-side helpers stay here as a trait.
pub trait PtrInfoExt {
    /// info.py `ConstPtrInfo.get_known_class(cpu)` + the other
    /// PtrInfo subclasses' `_known_class` accessors. Probes
    /// `majit_gc::supports_guard_gc_type` / `majit_gc::check_is_object`
    /// when discriminating constant pointers. The `cpu` argument routes
    /// `cls_of_box` through the `Cpu` trait so backends overriding the
    /// typeptr-at-offset-0 read (`gcremovetypeptr`) are honored.
    fn get_known_class(&self, cpu: &dyn crate::cpu::Cpu) -> Option<i64>;

    /// info.py `make_guards(op, short, optimizer)`. Returns false when a
    /// required runtime layout guard cannot be expressed.
    fn make_guards(
        &self,
        op: &Operand,
        short: &mut Vec<majit_ir::OpRc>,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> bool;

    /// info.py getstrlen / vstring.py / 249-258 — common string-length
    /// query across `ConstPtrInfo` and `StrPtrInfo`.
    fn get_known_str_length(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64>;

    /// info.py ConstPtrInfo.get_constant_string_spec and
    /// vstring.py / 236 / 298 — recursive constant string extraction.
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>>;

    /// vstring.py / 230 `strgetitem()` on string ptrinfo —
    /// virtual dispatch only.
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<Operand>;

    /// info.py visitor_dispatch_virtual_type / 369 / 376 / 445 / 485 / 598 / 701 +
    /// vstring.py / 263 / 333 `visitor_dispatch_virtual_type`.
    fn visitor_dispatch_virtual_type<V: crate::walkvirtual::VirtualVisitor>(
        &self,
        visitor: &mut V,
    ) -> Option<V::VInfo>;

    /// info.py / 222-226: force_box() emits the allocation and
    /// field writes via emit_extra(), recursively forcing child virtuals.
    /// `op` is the (bound) operand of the virtual being forced — RPython
    /// `force_box(self, op, optforce)` passes the op directly
    /// (info.py), so the make_equal_to / set_forwarded receiver
    /// needs no lookup.
    fn force_box(&mut self, op: &Operand, ctx: &mut crate::optimizeopt::OptContext) -> Operand;

    /// info.py: `_is_immutable_and_filled_with_constants`
    /// — used by `force_box` to decide whether a virtual can be
    /// constant-folded.
    fn is_immutable_and_filled_with_constants(&self, ctx: &crate::optimizeopt::OptContext) -> bool;
}

impl PtrInfoExt for PtrInfo {
    /// info.py `ConstPtrInfo.get_known_class(cpu)` +
    /// the other PtrInfo subclasses' `_known_class` accessors:
    ///
    /// ```text
    /// def get_known_class(self, cpu):
    ///     if not self._const.nonnull():
    ///         return None
    ///     if cpu.supports_guard_gc_type:
    ///         if not cpu.check_is_object(self._const.getref_base()):
    ///             return None
    ///     return cpu.cls_of_box(self._const)
    /// ```
    ///
    /// - `Instance`/`Virtual`: return the stored `known_class` field
    ///   (PyPy `InstancePtrInfo._known_class`). A class-only result of
    ///   `make_constant_class` is also stored as `Instance(descr=None,
    ///   known_class=Some(...))`.
    /// - `Constant`: null constants → `None`; otherwise, when the
    ///   backend supports `guard_gc_type` (`majit_gc::supports_guard_gc_type`),
    ///   gate `cls_of_box` on `majit_gc::check_is_object` so that
    ///   non-object constant pointers are rejected and the optimizer
    ///   does not read garbage at offset 0. When the backend does
    ///   not support `guard_gc_type`, RPython skips the
    ///   `check_is_object` call entirely and still returns
    ///   `cls_of_box(self._const)`; this port follows that.
    /// - Everything else: `None`.
    fn get_known_class(&self, cpu: &dyn crate::cpu::Cpu) -> Option<i64> {
        match self {
            // A stored class of 0 means the allocation's vtable address was unavailable at
            // build time, so the value reads as no known class while the flag stays valid.
            PtrInfo::Instance(v) => v.known_class.filter(|&c| c != 0),
            PtrInfo::Virtual(v) => v.known_class.filter(|&c| c != 0),
            PtrInfo::VirtualStruct(v) => v.known_class.filter(|&c| c != 0),
            PtrInfo::Constant(gcref) => {
                // info.py:764: `if not self._const.nonnull(): return None`
                if gcref.is_null() {
                    return None;
                }
                // A tagged immediate (odd-valued constant address) is an
                // unboxed `int`, not a heap object; reading a typeptr at
                // offset 0 would fault. Return `None` ("class unknown")
                // so the optimizer declines to fold the guard while the
                // runtime tag/GuardClass check still runs. This crate
                // cannot name the pyre `int` classptr, and no `Cpu`
                // accessor yields it without an offset-0 deref, so `None`
                // is the conservative, deref-free verdict. Gated on the
                // active GC's taggedpointers config (mirror of
                // CAN_BE_TAGGED, default false), so it is inert until the
                // enablement flip.
                if majit_gc::is_tagged_immediate(gcref.as_usize()) {
                    return None;
                }
                // info.py: gate the `check_is_object` call on
                // `supports_guard_gc_type`. When the backend doesn't
                // support guard_gc_type, RPython simply skips the
                // `check_is_object` step and still calls `cls_of_box`.
                if majit_gc::supports_guard_gc_type() && !majit_gc::check_is_object(*gcref) {
                    return None;
                }
                // info.py `return cpu.cls_of_box(self._const)` —
                // routes through the Cpu trait so backends that override
                // the typeptr-at-offset-0 read (e.g. `gcremovetypeptr`)
                // are honored here too.
                let vtable = cpu.bh_classof(*gcref);
                if vtable == 0 { None } else { Some(vtable) }
            }
            _ => None,
        }
    }

    /// info.py: make_guards(op, short, optimizer)
    /// info.py: make_guards(self, op, short, optimizer)
    ///
    /// Append guard operations to `short` that check this PtrInfo's
    /// properties hold for `op`. Used by use_box (shortpreamble.py).
    /// `ctx` plays the role of `optimizer` in the upstream signature:
    /// constants ride inline on their operand, and a producer pushed into
    /// `short` (lenop / eq_op below) takes a fresh position from
    /// `alloc_op_position_typed`; its consumer guard binds the producer's
    /// `OpRc` itself.
    fn make_guards(
        &self,
        op: &Operand,
        short: &mut Vec<majit_ir::OpRc>,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> bool {
        use majit_ir::OpRc;
        // history.py Const{Int,Float,Ptr}.value inline.
        let alloc_const = |value: Value| Operand::const_from_value(value);
        let op_b = op.clone();
        match self {
            // info.py: PtrInfo base — no-op
            PtrInfo::NonNull { .. } => {
                // info.py: NonNullPtrInfo.make_guards
                short.push(OpRc::new(Op::new(
                    OpCode::GuardNonnull,
                    std::slice::from_ref(&op_b),
                )));
            }
            PtrInfo::Instance(info) => {
                // info.py InstancePtrInfo.make_guards line-by-line.
                //
                //   def make_guards(self, op, short, optimizer):
                //       if self._known_class is not None:
                //           if not optimizer.cpu.remove_gctypeptr:
                //               short.append(GUARD_NONNULL[op])
                //               short.append(GUARD_IS_OBJECT[op])
                //               short.append(GUARD_CLASS[op, self._known_class])
                //           else:
                //               short.append(GUARD_NONNULL_CLASS[op, self._known_class])
                //       elif self.descr is not None:
                //           short.append(GUARD_NONNULL[op])
                //           if not optimizer.cpu.remove_gctypeptr:
                //               short.append(GUARD_IS_OBJECT[op])
                //           short.append(GUARD_SUBCLASS[op, ConstInt(descr.get_vtable())])
                //       else:
                //           AbstractStructPtrInfo.make_guards(...)
                //
                // `ctx.remove_gctypeptr` is the
                // `optimizer.cpu.remove_gctypeptr` analogue (llmodel.py:55
                // — translator config `gcremovetypeptr`). Pyre defaults
                // to True because its PyObject layout has static
                // singletons (INSTANCE_TYPE, INT_TYPE, …) with no GC
                // header, and the False-branch GUARD_IS_OBJECT reads
                // `obj - GcHeader::SIZE` (the `GuardIsObject` arm of
                // `majit-backend-wasm/src/codegen.rs`) which
                // SIGBUSes on those statics. The False branch is still
                // emitted line-by-line so a backend that flips
                // `remove_gctypeptr=false` (e.g. a future heap-only
                // PyObject layout) gets the upstream guard sequence
                // without further changes.
                if let Some(cls) = info.known_class {
                    // info.py stores `_known_class` on PtrInfo, but
                    // the emitted guard operand is the same ConstInt vtable
                    // address produced by backend/model.py cls_of_box
                    // `cls_of_box()`.
                    let class_ref = alloc_const(Value::Int(cls));
                    if !ctx.remove_gctypeptr {
                        short.push(OpRc::new(Op::new(
                            OpCode::GuardNonnull,
                            std::slice::from_ref(&op_b),
                        )));
                        short.push(OpRc::new(Op::new(
                            OpCode::GuardIsObject,
                            std::slice::from_ref(&op_b),
                        )));
                        short.push(OpRc::new(Op::new(
                            OpCode::GuardClass,
                            &[op_b.clone(), class_ref.clone()],
                        )));
                    } else {
                        short.push(OpRc::new(Op::new(
                            OpCode::GuardNonnullClass,
                            &[op_b.clone(), class_ref.clone()],
                        )));
                    }
                } else if let Some(descr) = &info.descr {
                    // info.py reads `self.descr.get_vtable()`
                    // unconditionally once the descr is present. Every pyre
                    // producer of an `InstancePtrInfo` descr proves the
                    // downcast: `ensure_ptr_info_arg0` unwraps the identical
                    // `parent_descr` with `.expect` before it builds the info
                    // (`optimizeopt/mod.rs`), and `deserialize_optheap` asserts
                    // the same shape on the same value (`heap.rs`'s
                    // `import_cached_fields`).
                    // A fabricated `0` would reach the backend as classptr 0
                    // and panic in the subclass-range lookup
                    // (`aarch64/assembler.rs`'s `emit_guard_subclass`),
                    // naming the wrong layer.
                    // A real `SizeDescr` whose `vtable()` is 0 is left alone:
                    // that is the headerless state, not a manufactured one.
                    let vtable = descr
                        .as_size_descr()
                        .expect(
                            "InstancePtrInfo.descr must be a SizeDescr \
                             (optimizer.py:480-481 InstancePtrInfo(parent_descr))",
                        )
                        .vtable() as i64;
                    let vtable_const = alloc_const(Value::Int(vtable));
                    short.push(OpRc::new(Op::new(
                        OpCode::GuardNonnull,
                        std::slice::from_ref(&op_b),
                    )));
                    if !ctx.remove_gctypeptr {
                        short.push(OpRc::new(Op::new(
                            OpCode::GuardIsObject,
                            std::slice::from_ref(&op_b),
                        )));
                    }
                    short.push(OpRc::new(Op::new(
                        OpCode::GuardSubclass,
                        &[op_b.clone(), vtable_const.clone()],
                    )));
                } else {
                    // info.py:353 fall-through with neither class nor
                    // descr — base NonNullPtrInfo.make_guards.
                    short.push(OpRc::new(Op::new(
                        OpCode::GuardNonnull,
                        std::slice::from_ref(&op_b),
                    )));
                }
            }
            PtrInfo::Struct(info) => {
                // info.py: StructPtrInfo.make_guards.
                //   if self.descr is not None:
                //       c_typeid = ConstInt(self.descr.get_type_id())
                //       short.extend([GUARD_NONNULL[op],
                //                     GUARD_GC_TYPE[op, c_typeid]])
                // `StructPtrInfo.descr` is a plain `DescrRef`
                // (`ptr_info.rs`), so info.py `if self.descr is not
                // None` is unconditionally true here and the `SizeDescr`
                // downcast cannot legitimately fail: `ensure_ptr_info_arg0`
                // proves it on the same `parent_descr` one line before it
                // builds the info (`optimizeopt/mod.rs`), and the VirtualStruct
                // force path hands its descr to `handle_new`, which unwraps a
                // `SizeDescr` (`majit-gc/src/rewrite.rs`).  A tid may
                // not be invented for a descr that cannot name one: the
                // allocator never mints 0 (`gctypelayout.py` DUMMY at
                // index 0) while the
                // runtime header tid 0 is the live `rclass.OBJECT` root
                // (`pyre-jit-trace/src/descr.rs`'s `OBJECT_GC_TYPE_ID`), so a
                // fabricated 0 would pass on any plain `object` instance and
                // certify a layout the
                // optimizer never named.
                let sd = info.descr.as_size_descr().expect(
                    "StructPtrInfo.descr must be a SizeDescr \
                     (optimizer.py:481 info.StructPtrInfo(parent_descr))",
                );
                // A serialized descr may carry only its structural cache key;
                // recover the runtime tid from the live GC cache before
                // constructing the layout guard. Headerless structs have no
                // header word and require only GUARD_NONNULL.
                let type_id = if sd.is_gc_managed() && !sd.headerless() {
                    let Some(type_id) =
                        resolve_gc_tid(sd.type_id(), sd.cache_key(), |cache, key| {
                            cache.resolve_struct_tid(key)
                        })
                    else {
                        return false;
                    };
                    Some(type_id)
                } else {
                    None
                };
                short.push(OpRc::new(Op::new(
                    OpCode::GuardNonnull,
                    std::slice::from_ref(&op_b),
                )));
                if let Some(type_id) = type_id {
                    let type_id = type_id as i64;
                    let type_id_const = alloc_const(Value::Int(type_id));
                    short.push(OpRc::new(Op::new(
                        OpCode::GuardGcType,
                        &[op_b.clone(), type_id_const.clone()],
                    )));
                }
            }
            PtrInfo::Constant(gcref) => {
                // info.py: ConstPtrInfo.make_guards
                let c = alloc_const(Value::Ref(*gcref));
                short.push(OpRc::new(Op::new(
                    OpCode::GuardValue,
                    &[op_b.clone(), c.clone()],
                )));
            }
            PtrInfo::Array(info) => {
                // info.py: ArrayPtrInfo.make_guards.
                //   AbstractVirtualPtrInfo.make_guards → NonNullPtrInfo.make_guards
                //   short.append(GUARD_GC_TYPE[op, ConstInt(descr.get_type_id())])
                //   if self.lenbound is not None:
                //       lenop = ARRAYLEN_GC[op] (descr=self.descr)
                //       short.append(lenop)
                //       self.lenbound.make_guards(lenop, short, optimizer)
                // `ArrayPtrInfo.descr` is a plain `DescrRef`
                // (`ptr_info.rs`) and info.py reads
                // `self.descr.get_type_id()` unconditionally.  Every producer
                // feeds an array-op descr (optimizer.py:485-487
                // `info.ArrayPtrInfo(op.getdescr())`, `ensure_ptr_info_arg0`) —
                // the same Arc the mandatory GC rewrite unwraps as an
                // `ArrayDescr`
                // (`majit-gc/src/rewrite.rs`'s `handle_setarrayitem` /
                // `handle_getarrayitem` / `transform_to_gc_load`),
                // including for the ARRAYLEN_GC this arm itself emits below.
                let ad = info.descr.as_array_descr().expect(
                    "ArrayPtrInfo.descr must be an ArrayDescr \
                     (optimizer.py:487 info.ArrayPtrInfo(op.getdescr()))",
                );
                // Raw native arrays have no header word. For GC arrays, a
                // missing stamped tid must be resolved from the structural
                // cache key; otherwise this short-preamble entry is unsafe.
                let stamped_tid = ad.type_id();
                let type_id = if ad.is_gc_managed() {
                    let Some(type_id) =
                        resolve_gc_tid(stamped_tid, ad.cache_key(), |cache, key| {
                            cache.resolve_array_tid(key)
                        })
                    else {
                        return false;
                    };
                    if stamped_tid == 0 {
                        ad.set_type_id(type_id);
                    }
                    Some(type_id)
                } else {
                    None
                };
                short.push(OpRc::new(Op::new(
                    OpCode::GuardNonnull,
                    std::slice::from_ref(&op_b),
                )));
                if let Some(type_id) = type_id {
                    let type_id = type_id as i64;
                    let type_id_const = alloc_const(Value::Int(type_id));
                    short.push(OpRc::new(Op::new(
                        OpCode::GuardGcType,
                        &[op_b.clone(), type_id_const.clone()],
                    )));
                }
                // Emit ARRAYLEN_GC + bound guards: pyre's ArrayPtrInfo.lenbound
                // is a plain `IntBound`, not an `Option`, so the parity check
                // is on `is_unbounded()` rather than `is None`.
                //
                // A length-less array descr (`len_offset = None`, the shape a
                // header-less buffer takes) has no length word to read, and
                // the mandatory GC rewrite unwraps the lendescr unconditionally
                // (`majit-gc/src/rewrite.rs`'s `transform_to_gc_load`) — emitting
                // the op there panics.  The bound itself is reachable on such an
                // array: a
                // constant-index element read narrows it through
                // `getlenbound().make_gt_const(index)` (`heap.py:676-681`,
                // ported in `optimizeopt/heap.rs`'s `optimize_getarrayitem`)
                // without ever consulting the descr.  Upstream cannot reach this
                // because a raw array
                // gets `getarrayitem_raw_*` and an `AbstractRawPtrInfo`, never
                // `ArrayPtrInfo`; pyre routes both families through one info
                // type, so the gate belongs here, alongside the `is_gc_managed`
                // gate above.  Nothing is lost by skipping it — with no length
                // in memory there is no length any guard could establish.
                let has_length = ad.len_descr().is_some();
                if has_length && !info.lenbound.is_unbounded() {
                    let lenop = OpRc::new(Op::with_descr(
                        OpCode::ArraylenGc,
                        std::slice::from_ref(&op_b),
                        info.descr.clone(),
                    ));
                    // ArrayPtrInfo.make_guards `lenop = ResOperation(
                    // ARRAYLEN_GC, [op])` followed by
                    // `lenbound.make_guards(lenop, ...)`: the `lenop` object
                    // is the chained guards' box arg.
                    lenop.pos().set(ctx.alloc_op_position_typed(Type::Int));
                    short.push(lenop.clone());
                    info.lenbound
                        .make_guards(&Operand::from_bound_op(&lenop), short, ctx);
                }
            }
            // info.py `AbstractRawPtrInfo.make_guards`:
            //
            // ```python
            // def make_guards(self, op, short, optimizer):
            //     from rpython.jit.metainterp.optimizeopt.optimizer import CONST_0
            //     op = ResOperation(rop.INT_EQ, [op, CONST_0])
            //     short.append(op)
            //     op = ResOperation(rop.GUARD_FALSE, [op])
            //     short.append(op)
            // ```
            //
            // Emits "must not be 0" check (null-pointer equivalent for
            // Int-typed raw pointers) at the short-preamble entry.
            // Both `RawBufferPtrInfo` (info.py) and
            // `RawSlicePtrInfo` (info.py) inherit this override.
            PtrInfo::VirtualRawBuffer(_) | PtrInfo::VirtualRawSlice(_) => {
                let zero = alloc_const(Value::Int(0));
                // `op = ResOperation(INT_EQ, [op, CONST_0])` is then the
                // GUARD_FALSE arg itself.
                let eq_op = OpRc::new(Op::new(OpCode::IntEq, &[op_b.clone(), zero]));
                eq_op.pos().set(ctx.alloc_op_position_typed(Type::Int));
                short.push(eq_op.clone());
                short.push(OpRc::new(Op::new(
                    OpCode::GuardFalse,
                    &[Operand::from_bound_op(&eq_op)],
                )));
            }
            PtrInfo::Str(sinfo) => {
                // vstring.py: StrPtrInfo.make_guards
                short.push(OpRc::new(Op::new(
                    OpCode::GuardNonnull,
                    std::slice::from_ref(&op_b),
                )));
                if let Some(ref bound) = sinfo.lenbound
                    && bound.lower >= 1
                {
                    let lenop_code = if sinfo.mode == 0 {
                        OpCode::Strlen
                    } else {
                        OpCode::Unicodelen
                    };
                    // StrPtrInfo.make_guards `lenop = ResOperation(STRLEN,
                    // [op])` is the box of `bound.make_guards(lenop, ...)`,
                    // which chains INT_GE/INT_LE/INT_AND →
                    // GUARD_TRUE/GUARD_VALUE pairs against it.
                    let lenop = OpRc::new(Op::new(lenop_code, std::slice::from_ref(&op_b)));
                    lenop.pos().set(ctx.alloc_op_position_typed(Type::Int));
                    short.push(lenop.clone());
                    bound.make_guards(&Operand::from_bound_op(&lenop), short, ctx);
                }
            }
            // Virtuals/Virtualizable: no guards needed in short preamble
            _ => {}
        }
        true
    }

    /// info.py getstrlen / vstring.py / 249-258 — common string-length
    /// query across `ConstPtrInfo` and `StrPtrInfo`.
    fn get_known_str_length(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64> {
        match self {
            PtrInfo::Str(info) => info.getstrlen(ctx, mode),
            // info.py ConstPtrInfo.getstrlen — delegate to
            // the runtime resolver for constant string pointers.
            PtrInfo::Constant(gcref) if !gcref.is_null() => ctx
                .string_length_resolver
                .as_deref()
                .and_then(|resolver| resolver(*gcref, mode)),
            _ => None,
        }
    }

    /// info.py ConstPtrInfo.get_constant_string_spec and
    /// vstring.py / 236 / 298 — recursive constant string extraction.
    fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>> {
        match self {
            PtrInfo::Str(info) => info.get_constant_string_spec(ctx, mode),
            // info.py: ConstPtrInfo.get_constant_string_spec
            // delegates to _unpack_str(mode) → extracts chars from the
            // constant GcRef.
            PtrInfo::Constant(gcref) if !gcref.is_null() => ctx
                .string_content_resolver
                .as_deref()
                .and_then(|resolver| resolver(*gcref, mode)),
            PtrInfo::Constant(_) => None,
            _ => None,
        }
    }

    /// vstring.py / 230 `strgetitem()` on string ptrinfo — virtual dispatch only.
    /// ConstPtr constant resolution is handled by `OptString::strgetitem`
    /// (vstring.py `_strgetitem`), which needs `&mut OptContext`.
    fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<Operand> {
        match self {
            PtrInfo::Str(info) => info.strgetitem(index, ctx),
            _ => None,
        }
    }

    /// info.py visitor_dispatch_virtual_type / 369 / 376 / 445 / 485 / 598 / 701 +
    /// vstring.py / 263 / 333 `visitor_dispatch_virtual_type`.
    ///
    /// Each virtual `PtrInfo` subclass implements `visitor_dispatch_virtual_type(visitor)`
    /// which calls the corresponding `visitor.visit_*()` method with the
    /// subclass's static metadata (descr, fielddescrs, array clear flag,
    /// raw buffer offsets, etc.). The visitor is free to produce a
    /// `VInfo` per call; the same visitor pattern is shared by
    /// `ResumeDataVirtualAdder` (resume.py) and `VirtualStateConstructor`
    /// (virtualstate.py).
    ///
    /// Returns `None` for non-virtual `PtrInfo` variants — RPython's
    /// `visitor_dispatch_virtual_type` is only defined on
    /// `AbstractVirtualPtrInfo` subclasses, so callers must check
    /// `is_virtual()` first.
    fn visitor_dispatch_virtual_type<V: crate::walkvirtual::VirtualVisitor>(
        &self,
        visitor: &mut V,
    ) -> Option<V::VInfo> {
        match self {
            // info.py InstancePtrInfo.visitor_dispatch_virtual_type.
            // `fields` still stores sparse `(field_index, OpRef)` entries, but
            // the visitor now rebuilds the full descriptor-order slot list so
            // resume.py can pair `fielddescrs` and `fieldnums` 1:1 again.
            PtrInfo::Virtual(info) => {
                let indices: Vec<u32> = info.fields.iter().map(|(fi, _)| *fi).collect();
                let fielddescrs = self.all_fielddescrs_from_descr();
                Some(visitor.visit_virtual(&info.descr, &indices, &fielddescrs))
            }
            // info.py StructPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualStruct(info) => {
                let indices: Vec<u32> = info.fields.iter().map(|(fi, _)| *fi).collect();
                let fielddescrs = self.all_fielddescrs_from_descr();
                Some(visitor.visit_vstruct(&info.descr, &indices, &fielddescrs))
            }
            // info.py ArrayPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualArray(info) => Some(visitor.visit_varray(&info.descr, info.clear)),
            // info.py ArrayStructInfo.visitor_dispatch_virtual_type.
            // The visitor consumes the canonical `fielddescrs` ordering; the
            // compatibility indices are the same descriptor-order slot numbers.
            PtrInfo::VirtualArrayStruct(info) => {
                let indices: Vec<u32> = (0..info.fielddescrs.len()).map(|i| i as u32).collect();
                Some(visitor.visit_varraystruct(
                    &info.descr,
                    info.element_fields.len(),
                    &indices,
                    &info.fielddescrs,
                ))
            }
            // info.py RawBufferPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualRawBuffer(info) => Some(visitor.visit_vrawbuffer(
                info.func,
                info.size,
                info.buffer.offsets(),
                info.buffer.descrs(),
            )),
            // info.py RawSlicePtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualRawSlice(info) => Some(visitor.visit_vrawslice(info.offset)),
            // vstring.py visitor_dispatch_virtual_type / 263-264 / 333-334 per-variant dispatch
            PtrInfo::Str(info) if info.is_virtual() => {
                let is_unicode = info.mode != 0;
                Some(match &info.variant {
                    VStringVariant::Plain(_) => visitor.visit_vstrplain(is_unicode),
                    VStringVariant::Concat(_) => visitor.visit_vstrconcat(is_unicode),
                    VStringVariant::Slice(_) => visitor.visit_vstrslice(is_unicode),
                    VStringVariant::Ptr => unreachable!("non-virtual Str reached virtual arm"),
                })
            }
            _ => None,
        }
    }

    /// info.py / 222-226: force_box() emits the allocation and
    /// field writes via emit_extra(), recursively forcing child virtuals.
    ///
    /// Generated ops are routed via emit_extra() (RPython
    /// emit_extra parity) so downstream passes can observe them.
    fn force_box(&mut self, op: &Operand, ctx: &mut crate::optimizeopt::OptContext) -> Operand {
        force_box_impl(self, op, ctx)
    }

    /// info.py: _is_immutable_and_filled_with_constants
    ///
    /// ```text
    /// if not self.descr.is_immutable():
    ///     return False
    /// for op in self._fields:
    ///     if op is None:
    ///         return False     # uninitialized field
    ///     ...
    /// ```
    ///
    /// Check if this virtual is immutable and all fields are constants.
    /// Used by force_box to determine if the virtual can be constant-folded.
    fn is_immutable_and_filled_with_constants(&self, ctx: &crate::optimizeopt::OptContext) -> bool {
        // typeptr is excluded from `_fields` (`heaptracker.py
        // all_fielddescrs`). A residual `new` may still hold a runtime
        // header value there; treating the vstruct as const-foldable
        // would allocate without writing it.
        if let PtrInfo::VirtualStruct(v) = self
            && let Some(tp) = &v.typeptr_value
            && ctx
                .resolve_operand_operand_opt(tp)
                .and_then(|cb| cb.const_value())
                .is_none()
        {
            return false;
        }
        let (fields, descr) = match self {
            PtrInfo::Virtual(v) => (&v.fields, &v.descr),
            PtrInfo::VirtualStruct(v) => (&v.fields, &v.descr),
            _ => return false,
        };
        // info.py: `if not self.descr.is_immutable()`.
        let Some(size_descr) = descr.as_size_descr() else {
            return false;
        };
        if !size_descr.is_immutable() {
            return false;
        }
        // info.py:286-288: `for op in self._fields: if op is None: return False`.
        // RPython's _fields is pre-allocated to len(descr.get_all_fielddescrs())
        // with None for unset slots; pyre stores only set entries in `fields`,
        // so parity requires fields.len() to match all_fielddescrs().len()
        // before treating the virtual as filled.
        if fields.len() != size_descr.all_fielddescrs().len() {
            return false;
        }
        for (_, val) in fields {
            let resolved_box = ctx.resolve_operand_operand_opt(val);
            if resolved_box
                .as_ref()
                .and_then(|cb| cb.const_value())
                .is_none()
            {
                // Check if it's a virtual that is also immutable+constant.
                // info.py:282-303 threads a `memo` keyed on the receiver info
                // (`if self in memo: return True`) to break recursion on a
                // cyclic immutable virtual (A.field -> B, B.field -> A). pyre
                // omits the memo: a JIT-created virtual cannot close such a
                // cycle, since immutable fields are written by SETFIELD in
                // construction order and the second box does not exist when the
                // first is sealed. Convergence path if a prebuilt cyclic
                // immutable virtual ever reaches here: thread a
                // `memo: &mut Vec<Operand>` keyed on the receiver box (operand
                // Rc-identity) mirroring info.py:282-284.
                if let Some(info) = resolved_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
                    && info.is_virtual()
                    && info.is_immutable_and_filled_with_constants(ctx)
                {
                    continue;
                }
                return false;
            }
        }
        true
    }
}

fn force_box_impl(
    self_: &mut PtrInfo,
    op: &Operand,
    ctx: &mut crate::optimizeopt::OptContext,
) -> Operand {
    use majit_ir::{Op, OpCode};

    // `op` is the bound operand of the virtual being forced (callers resolve to
    // the chain terminal before delegating). The OpRef view drives op identity
    // (pos, logging, alloc-vs-original comparisons); the operand drives every
    // make_equal_to / set_ptr_info receiver, so no `materialize_operand_at` round-trip is
    // needed for the forwarding writes.
    let opref = op.to_opref();

    // info.py: subbox = optimizer.force_box(fld) — `fld` is the field's
    // own box. Force it (materialising a nested virtual) and return the
    // resulting box, which the parent uses as a SETFIELD/SETARRAYITEM value
    // operand. Box-native: the field box is passed and returned directly,
    // matching `force_box(fld)` rather than an OpRef round-trip.
    fn force_child(orig: &Operand, ctx: &mut crate::optimizeopt::OptContext) -> Operand {
        let value_box = ctx.resolve_operand_operand_opt(orig);
        // optimizer.py:356-357: `info = op.get_forwarded()` then `if op.type ==
        // 'i' and info.is_constant(): return ConstInt(info.get_constant_int())`.
        // The const-int synthesis is gated on an info that is ALREADY forwarded
        // — force_box only reads the slot. `peek_intbound_box` reads the
        // forwarded IntBound without the lazy `set_forwarded(IntBound())`
        // install that `getintbound_handle` performs (optimizer.py), so a
        // plain int box with no info is returned as-is below rather than given a
        // fresh unbounded slot.
        if let Some(b) = &value_box
            && b.const_value().is_none()
            && b.type_() == majit_ir::Type::Int
            && let Some(bound) = ctx.peek_intbound_box(b)
            && bound.is_constant()
        {
            return Operand::const_(majit_ir::Const::Int(bound.get_constant_int()));
        }
        let value_is_virtual = match &value_box {
            Some(b) => ctx.is_virtual(b),
            None => false,
        };
        if value_is_virtual {
            let vb_op = value_box.expect("recorder-populated");
            let mut info = ctx.take_ptr_info(&vb_op).unwrap();
            let forced = force_box_impl(&mut info, &vb_op, ctx);
            return ctx.resolve_operand_operand(&forced);
        }
        value_box.unwrap_or_else(|| orig.clone())
    }

    // RPython info.py:148,226: optforce.emit_extra(op)
    // `optforce` determines where emitted ops enter the pass chain:
    //   optforce=Optimizer (in_final_emission) → emit directly
    //   optforce=OptEarlyForce → route from earlyforce.next (= heap)
    // When called from EarlyForce pass, current_pass_idx == earlyforce_idx
    // so emit_extra automatically routes from earlyforce.next.
    // When called from _emit_operation, in_final_emission=true → direct.
    fn emit_op_rc(ctx: &mut crate::optimizeopt::OptContext, op: majit_ir::OpRc) -> majit_ir::OpRc {
        if ctx.in_final_emission {
            // optimizer.py `Optimizer.emit_extra` → `emit` →
            // `_emit_operation`: `arg = self.force_box(op.getarg(i))`.
            for i in 0..op.num_args() {
                let arg = ctx.force_box_inline(&op.arg(i));
                op.setarg(i, arg);
            }
            ctx.emit_rc(op.clone());
        } else {
            ctx.emit_extra_rc(ctx.current_pass_idx, op.clone());
        }
        op
    }
    let emit_op = |ctx: &mut crate::optimizeopt::OptContext, op: Op| -> majit_ir::OpRc {
        emit_op_rc(ctx, majit_ir::OpRc::new(op))
    };
    // info.py `AbstractVirtualPtrInfo.force_box`: `op.set_forwarded(None);
    // optforce.emit_extra(op)` — the allocation is the virtual's own
    // operation, so `newop is op` and no forwarding is needed.
    let emit_forced = |ctx: &mut crate::optimizeopt::OptContext, op: &Operand| -> majit_ir::OpRc {
        let newop = op
            .bound_op()
            .expect("a virtual is the result of an operation");
        op.clear_forwarded();
        emit_op_rc(ctx, newop)
    };

    // Descr-derived view of the full fielddescr slot list, used by both
    // the constant-fold path and the per-field SETFIELD_GC emission in the
    // Virtual/VirtualStruct match arms below. Computed once so the call
    // sites don't need to re-borrow `self` while `vinfo` is borrowed.
    let cached_fielddescrs = self_.all_fielddescrs_from_descr();

    // RPython info.py:140-145: immutable virtual filled with constants
    // → constant fold to a compile-time constant pointer.
    if self_.is_immutable_and_filled_with_constants(ctx) {
        // info.py _force_elements_immutable: `subbox =
        // optimizer.force_box(fld)` materializes EACH field — including a
        // nested immutable virtual — into a Const before the parent's slots
        // are written. `is_immutable_and_filled_with_constants` guarantees a
        // non-Const field is itself an immutable virtual filled with
        // constants, so `force_child` recurses through this same fold and
        // const-folds it (no SETFIELD ops emitted). Run before borrowing
        // `constant_fold_alloc` (force_child needs `&mut ctx`); a field that
        // already resolves to a Const is left untouched.
        // Should a nested field fail to const-fold (its own fold fell back to
        // materialization), the parent's constant slot would be left
        // uninitialized, so fold the parent only when every field is constant.
        let mut can_fold_parent = true;
        if ctx.constant_fold_alloc.is_some() {
            let field_boxes: Vec<Operand> = match self_ {
                PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| r.clone()).collect(),
                PtrInfo::VirtualStruct(v) => v.fields.iter().map(|(_, r)| r.clone()).collect(),
                _ => Vec::new(),
            };
            for fb in field_boxes {
                if ctx
                    .resolve_operand_operand_opt(&fb)
                    .and_then(|b| b.const_value())
                    .is_none()
                {
                    force_child(&fb, ctx);
                    if ctx
                        .resolve_operand_operand_opt(&fb)
                        .and_then(|b| b.const_value())
                        .is_none()
                    {
                        can_fold_parent = false;
                    }
                }
            }
        }
        if let Some(alloc_fn) = ctx.constant_fold_alloc.as_ref().filter(|_| can_fold_parent) {
            let field_descrs = &cached_fielddescrs;
            let (descr, fields) = match self_ {
                PtrInfo::Virtual(v) => (&v.descr, &v.fields),
                PtrInfo::VirtualStruct(v) => (&v.descr, &v.fields),
                _ => unreachable!(),
            };
            let obj_size = descr.as_size_descr().map(|sd| sd.size()).unwrap_or(0);
            if obj_size > 0 {
                let ptr = alloc_fn(obj_size);
                if !ptr.is_null() {
                    // info.py: _force_elements_immutable
                    // Write constant field values directly to the allocated memory.
                    for (field_idx, val_ref) in fields.iter() {
                        let field_idx = *field_idx;
                        if let Some(value) = ctx
                            .resolve_operand_operand_opt(val_ref)
                            .and_then(|cb| cb.const_value())
                            && let Some(fd) = lookup_field_descr(field_descrs, field_idx)
                            && let Some(field_d) = fd.as_field_descr()
                        {
                            let offset = field_d.offset();
                            match value {
                                Value::Int(v) => unsafe {
                                    let dest = (ptr.0 as *mut u8).add(offset) as *mut i64;
                                    *dest = v;
                                },
                                Value::Ref(r) => unsafe {
                                    let dest = (ptr.0 as *mut u8).add(offset) as *mut usize;
                                    *dest = r.0;
                                },
                                _ => {}
                            }
                        }
                    }
                    // info.py:142: op.set_forwarded(constptr) — write
                    // unconditional. `set_ptr_info` on `op` walks the
                    // chain to the just-installed Const target (where it
                    // is a no-op per Const-box invariant).
                    let const_ref = GcRef(ptr.0);
                    ctx.make_constant_arg(op, Value::Ref(const_ref));
                    ctx.set_ptr_info(op, PtrInfo::Constant(const_ref));
                    return op.get_box_replacement(false);
                }
            }
        }
        // No allocator or size unknown: fall through to normal force.
    }

    match self_ {
        PtrInfo::VirtualStruct(vinfo) => {
            // RPython info.py _force_elements clears each
            // `self._fields[i] = None` BEFORE `optforce.emit_extra(setfieldop)`.
            // After force, the non-virtual structinfo carries no field cache,
            // so heap.py do_setfield records the SETFIELD_GC as a lazy_set
            // instead of MUST_ALIAS-eliding it against the preserved value.
            let preserved = PtrInfo::Struct(StructPtrInfo {
                descr: vinfo.descr.clone(),
                fields: CachedFieldList::new(),
                last_guard_pos: -1,
            });
            let alloc_rc = emit_forced(ctx, op);
            // info.py `newop.set_forwarded(self)` — unconditional.
            ctx.set_ptr_info(op, preserved);
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit][force-box] virtual-struct {:?} in_final_emission={} pass_idx={}",
                    opref, ctx.in_final_emission, ctx.current_pass_idx
                );
            }
            if let Some(value_ref) = vinfo.typeptr_value.take() {
                let value_ref = force_child(&value_ref, ctx);
                if let Some(descr) = vinfo.ob_type_descr.clone() {
                    let arg_alloc = Operand::from_bound_op(&alloc_rc);
                    let arg_value = ctx.resolve_operand_operand(&value_ref);
                    let mut set_op = Op::new(OpCode::SetfieldGc, &[arg_alloc, arg_value]);
                    set_op.setdescr(descr);
                    emit_op(ctx, set_op);
                }
            }
            // info.py _force_elements:
            //   for i, fielddescr in enumerate(descr.get_all_fielddescrs()):
            //       fld = self._fields[i]
            let mut leftover = std::mem::take(&mut vinfo.fields);
            for (i, descr) in cached_fielddescrs.iter().enumerate() {
                let field_idx = i as u32;
                let Some(pos) = leftover.iter().position(|(idx, _)| *idx == field_idx) else {
                    continue;
                };
                let (_, value_ref) = leftover.swap_remove(pos);
                let value_ref = force_child(&value_ref, ctx);
                if w_class_store_is_covered_by_alloc(&vinfo.descr, descr, &value_ref, ctx) {
                    continue;
                }
                let arg_alloc = Operand::from_bound_op(&alloc_rc);
                let arg_value = ctx.resolve_operand_operand(&value_ref);
                let mut set_op =
                    Op::new(OpCode::SetfieldGc, &[arg_alloc.clone(), arg_value.clone()]);
                set_op.setdescr(descr.clone());
                emit_op(ctx, set_op);
            }
            if !leftover.is_empty() {
                panic!(
                    "force_box: field_idx must resolve through descr.get_all_fielddescrs()[i] \
                     (leftover={:?} len={} keys={:?} {})",
                    leftover.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                    cached_fielddescrs.len(),
                    force_box_field_keys(&cached_fielddescrs),
                    force_box_size_identity(&vinfo.descr),
                );
            }
            Operand::from_bound_op(&alloc_rc)
        }
        PtrInfo::Virtual(vinfo) => {
            // info.py _force_elements — see VirtualStruct branch above. Build the
            // non-virtual replacement with no field cache so heap.py
            // do_setfield does not MUST_ALIAS-elide the materialization
            // SETFIELD_GC against the preserved value.
            let allocation_fields = allocation_initialized_w_class(&vinfo.descr)
                .into_iter()
                .collect();
            let preserved = PtrInfo::Instance(InstancePtrInfo {
                descr: Some(vinfo.descr.clone()),
                known_class: vinfo.known_class,
                // info.py keeps `_fields` empty here so initializing payload
                // SETFIELDs are not elided.  The sole pyre-only entry is the
                // public class header written by NEW_WITH_VTABLE itself; its
                // header-band slot cannot collide with any payload position.
                fields: allocation_fields,
                last_guard_pos: -1,
            });
            let alloc_rc = emit_forced(ctx, op);
            // info.py `newop.set_forwarded(self)` — unconditional.
            ctx.set_ptr_info(op, preserved);
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit][force-box] virtual {:?} in_final_emission={} pass_idx={}",
                    opref, ctx.in_final_emission, ctx.current_pass_idx
                );
            }
            // info.py _force_elements — same enumerate walk as VirtualStruct.
            let mut leftover = std::mem::take(&mut vinfo.fields);
            for (i, descr) in cached_fielddescrs.iter().enumerate() {
                let field_idx = i as u32;
                let Some(pos) = leftover.iter().position(|(idx, _)| *idx == field_idx) else {
                    continue;
                };
                let (_, value_ref) = leftover.swap_remove(pos);
                let value_ref = force_child(&value_ref, ctx);
                if w_class_store_is_covered_by_alloc(&vinfo.descr, descr, &value_ref, ctx) {
                    continue;
                }
                let arg_alloc = Operand::from_bound_op(&alloc_rc);
                let arg_value = ctx.resolve_operand_operand(&value_ref);
                let mut set_op =
                    Op::new(OpCode::SetfieldGc, &[arg_alloc.clone(), arg_value.clone()]);
                set_op.setdescr(descr.clone());
                emit_op(ctx, set_op);
            }
            if !leftover.is_empty() {
                panic!(
                    "force_box: field_idx must resolve through descr.get_all_fielddescrs()[i] \
                     (leftover={:?} len={} keys={:?} {})",
                    leftover.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                    cached_fielddescrs.len(),
                    force_box_field_keys(&cached_fielddescrs),
                    force_box_size_identity(&vinfo.descr),
                );
            }
            Operand::from_bound_op(&alloc_rc)
        }
        PtrInfo::VirtualArray(vinfo) => {
            // info.py `AbstractVirtualPtrInfo.force_box` forwards the
            // allocation at this array info and then clears `_is_virtual`.
            // The forced box keeps `descr` and the constant length, with no
            // cached items: `_force_elements` forgets every item, so a later
            // SETARRAYITEM_GC is not elided against the store just emitted.
            // `optimize_ARRAYLEN_GC` only folds a still-virtual array, so the
            // ARRAYLEN_GC pure registration below still publishes the length.
            let len = vinfo.items.len();
            let preserved = PtrInfo::Array(ArrayPtrInfo {
                descr: vinfo.descr.clone(),
                lenbound: IntBound::from_constant(len as i64),
                items: Vec::new(),
                last_guard_pos: -1,
            });

            let alloc_rc = emit_forced(ctx, op);
            // Install before the item stores: a child that points back at
            // this box must see the non-virtual array.
            ctx.set_ptr_info(op, preserved);

            // info.py:542: const = optforce.optimizer.new_const_item(self.descr)
            // info.py:546-548: skip items equal to the default when _clear=True
            let items = std::mem::take(&mut vinfo.items);
            let clear = vinfo.clear;
            let descr = vinfo.descr.clone();
            for (i, item_ref) in items.into_iter().enumerate() {
                if item_ref.is_none() {
                    continue;
                }
                // info.py:543: const = optforce.optimizer.new_const_item(self.descr)
                // info.py:546-548: if self._clear and const.same_constant(item)
                // new_const_item returns CONST_0/CONST_NULL/CONST_ZERO_FLOAT
                // (all raw=0).
                if clear {
                    let is_default = ctx
                        .resolve_operand_operand_opt(&item_ref)
                        .as_ref()
                        .and_then(|b| ctx.getconst(b))
                        .is_some_and(|(raw, _)| raw == 0);
                    if is_default {
                        continue;
                    }
                }
                let subbox = force_child(&item_ref, ctx);
                let idx_ref = Operand::const_(majit_ir::Const::Int(i as i64));
                let arg_alloc = Operand::from_bound_op(&alloc_rc);
                let arg_idx = idx_ref.clone();
                let arg_sub = ctx.resolve_operand_operand(&subbox);
                let mut set_op = Op::new(
                    OpCode::SetarrayitemGc,
                    &[arg_alloc.clone(), arg_idx.clone(), arg_sub.clone()],
                );
                set_op.setdescr(descr.clone());
                emit_op(ctx, set_op);
            }
            // pure_from_args(ARRAYLEN_GC, [op], ConstInt(len))
            ctx.pure_from_args_arraylen(
                majit_ir::operand::Operand::from_bound_op(&alloc_rc),
                len as i64,
            );
            Operand::from_bound_op(&alloc_rc)
        }
        PtrInfo::VirtualArrayStruct(vinfo) => {
            // `AbstractVirtualPtrInfo.force_box` forwards the allocation at
            // this `ArrayStructInfo` and clears `_is_virtual`. There is no
            // non-virtual array-struct variant. After the force, interior
            // field get/set and `VArrayStructStateInfo.enum_forced_boxes`
            // both require `is_virtual()`, and the heap cache does not
            // record interior fields, so the readers that still run
            // (`getlenbound`, `make_guards`, `ensure_ptr_info_arg0`) see
            // the same surface as a plain `ArrayPtrInfo`: this descr, a
            // constant length, and no cached items. `_force_elements`
            // emits the `SETINTERIORFIELD_GC` stores and does not publish
            // those fields as array items, so the forced box carries
            // `PtrInfo::Array`.
            // An array struct is always allocated cleared, so the op is
            // `NEW_ARRAY_CLEAR`.
            let num_elements = vinfo.element_fields.len();
            let preserved = PtrInfo::Array(ArrayPtrInfo {
                descr: vinfo.descr.clone(),
                lenbound: IntBound::from_constant(num_elements as i64),
                items: Vec::new(),
                last_guard_pos: -1,
            });

            let alloc_rc = emit_forced(ctx, op);
            // Install before the interior stores: a child that points back
            // at this box must see the non-virtual array.
            ctx.set_ptr_info(op, preserved);

            // info.py:672: fielddescrs = op.getdescr().get_all_fielddescrs()
            let fielddescrs: Vec<majit_ir::DescrRef> = vinfo
                .descr
                .as_array_descr()
                .and_then(|ad| ad.get_all_interiorfielddescrs())
                .map(|fds| fds.to_vec())
                .unwrap_or_else(|| vinfo.fielddescrs.clone());
            let element_fields = std::mem::take(&mut vinfo.element_fields);
            // info.py:673-684:
            //   for index in range(self.length):
            //       for fielddescr in fielddescrs:
            //           fld = self._items[i]
            //           if fld is not None:
            //               subbox = optforce.optimizer.force_box(fld)
            //               setfieldop = ResOperation(SETINTERIORFIELD_GC,
            //                   [op, ConstInt(index), subbox], descr=fielddescr)
            //               optforce.emit_extra(setfieldop)
            //           i += 1
            for (elem_idx, fields) in element_fields.into_iter().enumerate() {
                let idx_ref = Operand::const_(majit_ir::Const::Int(elem_idx as i64));
                for (field_idx, value_ref) in fields {
                    if value_ref.is_none() {
                        continue;
                    }
                    let subbox = force_child(&value_ref, ctx);
                    let arg_alloc = Operand::from_bound_op(&alloc_rc);
                    let arg_idx = idx_ref.clone();
                    let arg_sub = ctx.resolve_operand_operand(&subbox);
                    let mut set_op = Op::new(
                        OpCode::SetinteriorfieldGc,
                        &[arg_alloc.clone(), arg_idx.clone(), arg_sub.clone()],
                    );
                    if let Some(d) = fielddescrs.get(field_idx as usize).cloned() {
                        set_op.setdescr(d);
                    }
                    emit_op(ctx, set_op);
                }
            }
            Operand::from_bound_op(&alloc_rc)
        }
        PtrInfo::VirtualRawBuffer(vinfo) => {
            // info.py: RawBufferPtrInfo._force_elements()
            // info.py: self.size = -1 (mark as no longer virtual)
            let entries = vinfo.buffer.drain_entries();

            // info.py `force_box`: emit the raw-malloc CALL_I itself.
            let alloc_rc = emit_forced(ctx, op);

            // info.py:152/421 set_forwarded(self). RPython keeps the SAME
            // RawBufferPtrInfo on the forced op and flips it non-virtual via
            // `self.size = -1` (is_virtual = `size != -1`), so a later
            // `getrawptrinfo` still recovers the raw-buffer identity. pyre
            // installs `nonnull()` instead — a PRE-EXISTING-ADAPTATION: the
            // RawSlice force keeps identity by reinstalling
            // `RawSlicePtrInfo` with `parent` set to the none sentinel (the
            // `VirtualRawSlice` arm of `force_box_impl`), but a non-virtual
            // RawBuffer would need a size=-1 sentinel on `RawBufferPtrInfo` AND
            // an `is_virtual()` gate at
            // every `matches!(PtrInfo::VirtualRawBuffer(_))` site in
            // virtualize.rs (`optimize_raw_load` / `optimize_raw_store` /
            // `optimize_getarrayitem_raw` / `optimize_setarrayitem_raw`),
            // which currently assume virtual
            // and would re-virtualize a forced buffer. Raw buffers are absent
            // from the check.py corpus, so that cascade is ungateable here.
            // nonnull() preserves nonnull-ness; a later RAW_LOAD/RAW_STORE just
            // residualizes (getrawptrinfo→None), which is the correct result.
            ctx.set_ptr_info(op, PtrInfo::nonnull());

            // info.py:425: CHECK_MEMORY_ERROR
            let arg_alloc = Operand::from_bound_op(&alloc_rc);
            let check_op = Op::new(OpCode::CheckMemoryError, std::slice::from_ref(&arg_alloc));
            emit_op(ctx, check_op);

            // info.py:429-436: emit RAW_STORE for each buffered write.
            // info.py uses `itembox = buffer.values[i]` DIRECTLY in the
            // RAW_STORE — unlike the struct/array paths (info.py:222/548),
            // RawBufferPtrInfo._force_elements does NOT optimizer.force_box the
            // value (info.py: "there can be no virtuals stored in raw
            // buffer"). Use the stored value box as-is; routing it through
            // force_child would force-box it and synthesize a ConstInt for a
            // constant-bound int, neither of which upstream does here.
            for (offset, _length, descr, value) in entries {
                let offset_ref = Operand::const_(majit_ir::Const::Int(offset));
                let arg_alloc = Operand::from_bound_op(&alloc_rc);
                let arg_offset = offset_ref.clone();
                let arg_value = ctx.resolve_operand_operand(&value);
                let mut store_op = Op::new(
                    OpCode::RawStore,
                    &[arg_alloc.clone(), arg_offset.clone(), arg_value.clone()],
                );
                store_op.setdescr(descr);
                emit_op(ctx, store_op);
            }

            Operand::from_bound_op(&alloc_rc)
        }
        PtrInfo::VirtualRawSlice(slice) => {
            // `info.py` `RawSlicePtrInfo._force_elements`:
            //
            // ```python
            // def _force_elements(self, op, optforce, descr):
            //     if self.parent.is_virtual():
            //         self.parent._force_elements(op, optforce, descr)
            //     self.parent = None
            // ```
            //
            // RPython keeps the `RawSlicePtrInfo` attached to the op and
            // flips it to non-virtual by setting `self.parent = None`
            // (`is_virtual` at info.py is `self.parent is not None`).
            // The info class stays RawSlicePtrInfo so subsequent
            // `getrawptrinfo` lookups still identify it as a raw slice.
            //
            // pyre's `RawSlicePtrInfo` stores `parent: OpRef`; the
            // `OpRef::NONE` sentinel plays the role of `None`, and
            // `PtrInfo::is_virtual` gates on `slice.parent.is_none()`.
            // Overwriting with `PtrInfo::nonnull()` would lose the
            // raw-slice identity and mis-route any later
            // `get_virtual_fields` / raw-guard path.
            force_child(&slice.parent, ctx);
            // info.py `force_box`: emit the slice's INT_ADD itself.
            let new_rc = emit_forced(ctx, op);
            // Preserve raw-slice identity; mark non-virtual via
            // `parent = OpRef::NONE` (RPython `self.parent = None`).
            // info.py `newop.set_forwarded(self)`, unconditional.
            ctx.set_ptr_info(
                op,
                PtrInfo::VirtualRawSlice(RawSlicePtrInfo {
                    offset: slice.offset,
                    parent: Operand::None,
                    last_guard_pos: slice.last_guard_pos,
                    avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                }),
            );
            Operand::from_bound_op(&new_rc)
        }
        PtrInfo::Str(sinfo) if sinfo.is_virtual() => {
            // vstring.py StrPtrInfo.force_box
            let mode = sinfo.mode;
            let is_unicode = mode != 0;

            // vstring.py: if self.mode is mode_string / else
            let c_s = if mode == crate::optimizeopt::vstring::mode_string {
                // vstring.py:80-84
                sinfo
                    .get_constant_string_spec(&*ctx, mode)
                    .and_then(|chars| crate::history::get_const_ptr_for_string(&chars, ctx))
            } else {
                // vstring.py:86-90
                sinfo
                    .get_constant_string_spec(&*ctx, mode)
                    .and_then(|chars| crate::history::get_const_ptr_for_unicode(&chars, ctx))
            };
            if let Some(gcref) = c_s {
                // vstring.py:83: get_box_replacement(op).set_forwarded(c_s)
                ctx.make_constant_arg(op, Value::Ref(gcref));
                return op.get_box_replacement(false);
            }

            // vstring.py:91: self._is_virtual = False
            let sinfo_full = match std::mem::replace(self_, PtrInfo::nonnull()) {
                PtrInfo::Str(s) => s,
                _ => unreachable!(),
            };
            let variant = sinfo_full.variant;

            // vstring.py: lengthbox = self.getstrlen(op, optstring, mode)
            let lengthbox = match &variant {
                VStringVariant::Plain(info) => {
                    Operand::const_(majit_ir::Const::Int(info._chars.len() as i64))
                }
                VStringVariant::Slice(info) => ctx
                    .resolve_operand_operand_opt(&info.lgtop)
                    .unwrap_or_else(|| info.lgtop.clone()),
                VStringVariant::Concat(info) => {
                    let left_len = ctx.getstrlen_opref(info.vleft.to_opref(), mode);
                    let right_len = ctx.getstrlen_opref(info.vright.to_opref(), mode);
                    crate::optimizeopt::vstring::_int_add(&left_len, &right_len, ctx)
                }
                VStringVariant::Ptr => unreachable!(),
            };

            // vstring.py:93-96: newop = ResOperation(mode.NEWSTR, [lengthbox])
            let new_opcode = if is_unicode {
                OpCode::Newunicode
            } else {
                OpCode::Newstr
            };
            let arg_length = lengthbox;
            let newstr_op = Op::new(new_opcode, std::slice::from_ref(&arg_length));
            let newop_rc = emit_op(ctx, newstr_op);
            let newop_box = Operand::from_bound_op(&newop_rc);

            // vstring.py:98: newop.set_forwarded(self) — unconditional.
            ctx.set_ptr_info(
                &newop_box,
                PtrInfo::Str(StrPtrInfo {
                    lenbound: sinfo_full.lenbound,
                    lgtop: Some(arg_length.clone()), // `StrPtrInfo.force_box` keeps the computed length
                    mode: sinfo_full.mode,
                    length: sinfo_full.length,
                    variant: VStringVariant::Ptr, // non-virtual
                    last_guard_pos: sinfo_full.last_guard_pos,
                    avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                }),
            );

            // vstring.py `StrPtrInfo.force_box`: op = get_box_replacement(op);
            // op.set_forwarded(newop)
            op.get_box_replacement(false).set_forwarded_op(&newop_rc);

            // vstring.py: initialize_forced_string(op, optstring, op, CONST_0, mode)
            let zero = Operand::const_(majit_ir::Const::Int(0));
            let set_opcode = if is_unicode {
                OpCode::Unicodesetitem
            } else {
                OpCode::Strsetitem
            };

            match variant {
                VStringVariant::Plain(info) => {
                    // vstring.py VStringPlainInfo.initialize_forced_string
                    let mut offset = zero.clone();
                    let one = Operand::const_(majit_ir::Const::Int(1));
                    let one = one.clone();
                    for ch in &info._chars {
                        if let Some(ch_ref) = ch {
                            // vstring.py initialize_forced_string get_box_replacement(charbox) — walk the
                            // char operand's own forwarding (object-native).
                            let ch_resolved = ctx.resolve_operand_operand(ch_ref).to_opref();
                            let arg_newop = Operand::from_bound_op(&newop_rc);
                            let arg_offset = ctx.resolve_operand_operand(&offset);
                            let arg_ch = ctx.materialize_operand_at(ch_resolved);
                            let setitem_op = Op::new(
                                set_opcode,
                                &[arg_newop.clone(), arg_offset.clone(), arg_ch.clone()],
                            );
                            emit_op(ctx, setitem_op);
                        }
                        offset = crate::optimizeopt::vstring::_int_add(&offset, &one, ctx);
                    }
                }
                VStringVariant::Concat(info) => {
                    // vstring.py VStringConcatInfo.string_copy_parts
                    let newop_box = Operand::from_bound_op(&newop_rc);
                    let zero_box = zero.clone();
                    let offset = crate::optimizeopt::vstring::string_copy_parts(
                        &info.vleft.clone(),
                        &newop_box,
                        &zero_box,
                        mode,
                        ctx,
                    );
                    crate::optimizeopt::vstring::string_copy_parts(
                        &info.vright.clone(),
                        &newop_box,
                        &offset,
                        mode,
                        ctx,
                    );
                }
                VStringVariant::Slice(info) => {
                    // vstring.py VStringSliceInfo.string_copy_parts
                    let newop_box = Operand::from_bound_op(&newop_rc);
                    let zero_box = zero.clone();
                    crate::optimizeopt::vstring::copy_str_content(
                        ctx,
                        &info.s.clone(),
                        &newop_box,
                        &info.start.clone(),
                        &zero_box,
                        &info.lgtop.clone(),
                        mode,
                        true,
                    );
                }
                VStringVariant::Ptr => unreachable!(),
            }

            Operand::from_bound_op(&newop_rc)
        }
        _ => op.clone(),
    }
}

/// info.py `AbstractVirtualPtrInfo` line-by-line shared state.
///
/// ```python
/// class AbstractVirtualPtrInfo(NonNullPtrInfo):
///     _attrs_ = ('_cached_vinfo', 'descr', '_is_virtual')
///     _cached_vinfo = None
/// ```
///
/// Every concrete virtual-flavoured PtrInfo (Virtual, VirtualStruct,
/// VirtualArray, VirtualArrayStruct, VirtualRawBuffer, VirtualRawSlice,
/// Str) inherits `_cached_vinfo` from `AbstractVirtualPtrInfo`. Pyre
/// embeds this shared struct as `pub avpi: AbstractVirtualPtrInfo` in
/// each concrete variant so the inheritance contract is structural,
/// not per-variant copy-paste.
///
/// `descr` and `_is_virtual` are NOT lifted here:
///   - `descr` is variant-specific (SizeDescr for Virtual, ArrayDescr
///     for VirtualArray, etc.) — RPython's `_attrs_` is a hint to the
///     translator's slot allocator, not a parity constraint on the
///     storage *type*. Each pyre variant keeps its own typed `descr`.
///   - `_is_virtual` collapses into the pyre enum tag itself
///     (`PtrInfo::Virtual(_)` IS the truthy carrier of `_is_virtual`);
///     no separate slot is needed.
///
/// `make_virtual_info` (resume.py) reads `cached_vinfo` to
/// dedup RdVirtualInfo allocations across multiple finish() calls
/// referencing the same virtual. `RefCell` provides interior
/// mutability so the immutable-receiver accessor can populate the
/// cache on first miss.
pub use majit_ir::ptr_info::{
    AbstractVirtualPtrInfo, ArrayPtrInfo, ArrayStructInfo, InstancePtrInfo, RawBufferPtrInfo,
    RawSlicePtrInfo, StructPtrInfo, VirtualArrayInfo, VirtualInfo, VirtualStructInfo,
    VirtualizableFieldState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::OptContext;
    use majit_ir::descr::{SimpleArrayDescr, SimpleSizeDescr, gc_cache};
    use majit_ir::{ArrayDescr, Descr, FieldDescr, LLType, OpCode, SizeDescr, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr;
    impl Descr for TestDescr {}

    #[derive(Debug)]
    struct ForceFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
        name: &'static str,
    }

    impl Descr for ForceFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for ForceFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
        fn is_pointer_field(&self) -> bool {
            self.field_type == Type::Ref
        }
        fn field_name(&self) -> &str {
            self.name
        }
        // The fixture builds an op-side and a layout-side descr separately and
        // tells them apart by name, so the mock declares from the name it was
        // constructed with. Production producers store the flag instead.
        fn is_w_class(&self) -> bool {
            self.name == "w_class" || self.name.ends_with(".w_class")
        }
    }

    #[derive(Debug)]
    struct ForceSizeDescr {
        all_fields: Vec<Arc<dyn FieldDescr>>,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
        w_class: i64,
    }

    impl Descr for ForceSizeDescr {
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for ForceSizeDescr {
        fn size(&self) -> usize {
            32
        }
        fn type_id(&self) -> u32 {
            7
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.all_fields
        }
        fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.gc_fields
        }
        fn w_class_obj(&self) -> Option<i64> {
            Some(self.w_class)
        }
    }

    fn force_virtual_with_w_class(
        stored_w_class: usize,
        stored_offset: usize,
        init_offset: usize,
    ) -> Vec<Op> {
        let stored_field: Arc<dyn FieldDescr> = Arc::new(ForceFieldDescr {
            offset: stored_offset,
            field_size: 8,
            field_type: Type::Ref,
            name: "PyObject.w_class",
        });
        let other_field: Arc<dyn FieldDescr> = Arc::new(ForceFieldDescr {
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            name: "Object.payload",
        });
        // Deliberately use an independent descriptor for the allocation-side
        // lookup: name equality alone must not establish byte identity.
        let init_field: Arc<dyn FieldDescr> = Arc::new(ForceFieldDescr {
            offset: init_offset,
            field_size: 8,
            field_type: Type::Ref,
            name: "w_class",
        });
        let w_class = 0xCAFEusize;
        let size_descr: DescrRef = Arc::new(ForceSizeDescr {
            all_fields: vec![stored_field, other_field],
            gc_fields: vec![init_field],
            w_class: w_class as i64,
        });
        let mut info = PtrInfo::virtual_obj(size_descr.clone(), Some(0xDEAD));
        info.setfield(
            0,
            Operand::const_from_value(Value::Ref(GcRef(stored_w_class))),
        );
        info.setfield(1, Operand::const_from_value(Value::Int(42)));

        let mut ctx = OptContext::new(8);
        ctx.in_final_emission = true;
        // The virtual is the result of its NEW_WITH_VTABLE, which force_box
        // emits.
        let new_rc = OpRc::new(Op::with_descr(OpCode::NewWithVtable, &[], size_descr));
        new_rc.pos().set(OpRef::ref_op(10));
        ctx.register_extra_producer(&new_rc);
        let virtual_box = Operand::from_bound_op(&new_rc);
        info.force_box(&virtual_box, &mut ctx);
        ctx.new_operations
            .iter()
            .map(|op| op.as_ref().clone())
            .collect()
    }

    fn setfield_offsets(ops: &[Op]) -> Vec<usize> {
        ops.iter()
            .filter(|op| op.opcode == OpCode::SetfieldGc)
            .map(|op| {
                op.getdescr()
                    .and_then(|descr| descr.as_field_descr().map(|fd| fd.offset()))
                    .expect("SETFIELD_GC must carry a field descriptor")
            })
            .collect()
    }

    /// Bound-producer `Operand` at position `int_op(pos)` / `ref_op(pos)`,
    /// the field-value analog of the old `from_opref` test stand-ins.
    fn field_op(tp: Type, pos: u32) -> Operand {
        crate::history::test_support::rooted_resop_operand(tp, pos)
    }

    /// A one-byte integer array descr, with or without a length word.
    /// `lendescr: None` is the shape a header-less buffer takes — the `&[u8]`
    /// `env` slice and every `array_fields` element array.
    fn byte_array_descr(with_length: bool) -> DescrRef {
        let lendescr = with_length.then(|| {
            majit_ir::descr::make_field_descr(
                /* offset */ 0,
                /* field_size */ std::mem::size_of::<usize>(),
                Type::Int,
                majit_ir::descr::ArrayFlag::Unsigned,
            )
        });
        majit_ir::descr::make_array_descr_from_lltype_shape(
            /* type_id */ 0,
            /* base_size */ if with_length { 8 } else { 0 },
            /* item_size */ 1,
            /* len_offset */ if with_length { Some(0) } else { None },
            Type::Int,
            /* is_array_of_pointers */ false,
            /* is_array_of_structs */ false,
            /* is_item_signed */ false,
            /* is_gc_managed */ false,
            lendescr,
            /* is_pure */ false,
            /* ei_index */ u32::MAX,
            /* interior_field_descrs */ Vec::new(),
        )
    }

    /// The short-preamble guards `ArrayPtrInfo::make_guards` emits for an array
    /// whose length bound was narrowed.
    fn array_short_preamble_opcodes(with_length: bool) -> Vec<OpCode> {
        let info = PtrInfo::Array(majit_ir::ptr_info::ArrayPtrInfo {
            descr: byte_array_descr(with_length),
            // What a constant-index element read leaves behind:
            // `getlenbound().make_gt_const(index)` (`heap.py:676-681`).
            lenbound: IntBound::from_constant(8),
            items: Vec::new(),
            last_guard_pos: -1,
        });
        let mut ctx = OptContext::new(8);
        let array_box = field_op(Type::Ref, 3);
        let mut short = Vec::new();
        info.make_guards(&array_box, &mut short, &mut ctx);
        short.iter().map(|op| op.opcode).collect()
    }

    /// A length-less array descr has no length word to read, and the mandatory
    /// GC rewrite unwraps the lendescr unconditionally
    /// (`majit-gc/src/rewrite.rs`'s `transform_to_gc_load`), so emitting
    /// `ARRAYLEN_GC` for one panics the compile. The bound that triggers the
    /// emission is reachable —
    /// a constant-index element read narrows it without ever consulting the
    /// descr — which is why the gate has to be here rather than at the
    /// narrowing site.
    #[test]
    fn a_lengthless_array_emits_no_arraylen_guard() {
        let opcodes = array_short_preamble_opcodes(/* with_length */ false);
        assert!(
            !opcodes.contains(&OpCode::ArraylenGc),
            "a length-less array descr must not get an ARRAYLEN_GC guard; \
             emitted {opcodes:?}"
        );
    }

    /// The control: with a length word the guard must still be emitted, or the
    /// test above passes because `make_guards` stopped emitting it for
    /// everything.
    #[test]
    fn an_array_with_a_length_word_still_gets_its_arraylen_guard() {
        let opcodes = array_short_preamble_opcodes(/* with_length */ true);
        assert!(
            opcodes.contains(&OpCode::ArraylenGc),
            "a length-carrying array must keep its ARRAYLEN_GC guard; \
             emitted {opcodes:?}"
        );
    }

    #[test]
    fn test_ptr_info_factories() {
        let nonnull = PtrInfo::nonnull();
        assert!(nonnull.is_nonnull());
        assert!(!nonnull.is_virtual());

        let constant = PtrInfo::constant(GcRef(0x1000));
        assert!(constant.is_nonnull());
        assert!(constant.is_constant());

        let kc = PtrInfo::known_class(0x2000, true);
        assert!(kc.is_nonnull());
        // Instance arm — no cpu read; DefaultCpu is a placeholder.
        assert!(kc.get_known_class(&crate::cpu::DefaultCpu).is_some());
    }

    #[test]
    fn test_ptr_info_virtual_factories() {
        let descr: DescrRef = Arc::new(TestDescr);

        let virtual_obj = PtrInfo::virtual_obj(descr.clone(), Some(0x3000));
        assert!(virtual_obj.is_virtual());
        assert!(virtual_obj.is_nonnull());
        assert!(virtual_obj.get_descr().is_some());

        let virtual_arr = PtrInfo::virtual_array(descr.clone(), 5, false);
        assert!(virtual_arr.is_virtual());
        assert_eq!(virtual_arr.num_fields(), 5);

        let virtual_struct = PtrInfo::virtual_struct(descr);
        assert!(virtual_struct.is_virtual());
    }

    #[test]
    fn make_guards_resolves_zero_stamped_layout_tids() {
        let key = 0xFFFF_FF00_0000_0042;
        let struct_tid = 71;
        let array_tid = 72;

        let cached_struct: DescrRef = Arc::new(SimpleSizeDescr::new(0, 16, struct_tid));
        let cached_array: DescrRef =
            Arc::new(SimpleArrayDescr::new(0, 16, 8, array_tid, Type::Int));
        {
            let mut cache = gc_cache().lock();
            cache._cache_size.insert(LLType::Struct(key), cached_struct);
            cache._cache_array.insert(LLType::Array(key), cached_array);
        }

        let mut rehydrated_struct = SimpleSizeDescr::new(1, 16, 0);
        rehydrated_struct.set_cache_key(key);
        let struct_info = PtrInfo::struct_ptr(Arc::new(rehydrated_struct));
        let mut rehydrated_array = SimpleArrayDescr::new(2, 16, 8, 0, Type::Int);
        rehydrated_array.set_cache_key(key);
        let rehydrated_array: DescrRef = Arc::new(rehydrated_array);
        let array_info = PtrInfo::array(rehydrated_array.clone(), IntBound::unbounded());

        let mut ctx = OptContext::new(8);
        let mut struct_guards = Vec::new();
        let struct_box = ctx.materialize_operand_at(OpRef::ref_op(1));
        assert!(struct_info.make_guards(&struct_box, &mut struct_guards, &mut ctx));
        assert_eq!(struct_guards[1].opcode, OpCode::GuardGcType);
        assert_eq!(struct_guards[1].arg(1).const_value(), Some(Value::Int(71)));

        let mut array_guards = Vec::new();
        let array_box = ctx.materialize_operand_at(OpRef::ref_op(2));
        assert!(array_info.make_guards(&array_box, &mut array_guards, &mut ctx));
        assert_eq!(array_guards[1].opcode, OpCode::GuardGcType);
        assert_eq!(array_guards[1].arg(1).const_value(), Some(Value::Int(72)));
        assert_eq!(
            rehydrated_array.as_array_descr().unwrap().type_id(),
            array_tid
        );

        let mut cache = gc_cache().lock();
        cache._cache_size.shift_remove(&LLType::Struct(key));
        cache._cache_array.shift_remove(&LLType::Array(key));
    }

    #[test]
    fn make_guards_declines_an_unresolved_gc_layout() {
        let mut descr = SimpleArrayDescr::new(0, 16, 8, 0, Type::Int);
        descr.set_cache_key(0xFFFF_FF00_0000_0043);
        let info = PtrInfo::array(Arc::new(descr), IntBound::unbounded());
        let mut guards = Vec::new();
        let mut ctx = OptContext::new(4);

        let op = ctx.materialize_operand_at(OpRef::ref_op(1));
        assert!(!info.make_guards(&op, &mut guards, &mut ctx));
        assert!(guards.is_empty());
    }

    #[test]
    fn test_force_virtual_array_keeps_array_info_and_folds_arraylen() {
        use crate::optimizeopt::Optimization;
        use crate::optimizeopt::pure::OptPure;

        let descr = byte_array_descr(true);
        let length = 4usize;
        let mut info = PtrInfo::virtual_array(descr.clone(), length, false);
        // A written slot must not survive: `_force_elements` forgets items
        // while emitting the initializing stores.
        info.setitem(1, field_op(Type::Int, 7));

        let mut ctx = OptContext::new(32);
        ctx.in_final_emission = true;
        let virtual_box = field_op(Type::Ref, 10);
        let forced = info.force_box(&virtual_box, &mut ctx);

        let forced_box = ctx.resolve_operand_operand(&forced);
        let ptr = ctx
            .peek_ptr_info(&forced_box)
            .expect("forced box carries ptr info");
        let PtrInfo::Array(array) = &ptr else {
            panic!("forced virtual array must be PtrInfo::Array, got {ptr:?}");
        };
        assert!(
            Arc::ptr_eq(&array.descr, &descr),
            "forced array keeps the virtual array descr"
        );
        assert!(
            array.lenbound.is_constant(),
            "forced array length is the constant virtual length"
        );
        assert_eq!(array.lenbound.get_constant_int(), length as i64);
        assert!(
            array.items.is_empty(),
            "forced array forgets cached items, got {:?}",
            array.items
        );

        // The pure registration is descr-less (`pure_from_args` without a
        // descr), so the following ARRAYLEN_GC carries no descr either.
        let mut pure = OptPure::new();
        let array_arg = forced.clone();
        let mut len_op = Op::new(OpCode::ArraylenGc, &[array_arg]);
        len_op.pos().set(OpRef::int_op(40));
        let len_rc = OpRc::new(len_op);
        let result = pure.propagate_forward(len_rc.as_ref(), &len_rc, &mut ctx);
        assert!(
            matches!(result, crate::optimizeopt::OptimizationResult::Remove),
            "ARRAYLEN_GC on the forced array must fold, got {result:?}"
        );
        let folded = Operand::from_bound_op(&len_rc).get_box_replacement(false);
        assert_eq!(folded.const_value(), Some(Value::Int(length as i64)));
    }

    #[test]
    fn test_force_virtual_array_struct_keeps_array_info() {
        let descr = byte_array_descr(true);
        let length = 2usize;
        let mut info = PtrInfo::VirtualArrayStruct(ArrayStructInfo {
            descr: descr.clone(),
            element_fields: vec![
                majit_ir::ptr_info::VirtualFieldList::new(),
                majit_ir::ptr_info::VirtualFieldList::new(),
            ],
            fielddescrs: vec![Arc::new(ForceFieldDescr {
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                name: "real",
            })],
            last_guard_pos: -1,
            avpi: AbstractVirtualPtrInfo::new(),
        });
        // A written interior field must not survive: the forced info keeps
        // no item cache, so a later read is not elided against this store.
        info.setinteriorfield_virtual(0, 0, field_op(Type::Int, 7));

        let mut ctx = OptContext::new(32);
        ctx.in_final_emission = true;
        let virtual_box = field_op(Type::Ref, 10);
        let forced = info.force_box(&virtual_box, &mut ctx);

        let forced_box = ctx.resolve_operand_operand(&forced);
        let ptr = ctx
            .peek_ptr_info(&forced_box)
            .expect("forced box carries ptr info");
        let PtrInfo::Array(array) = &ptr else {
            panic!("forced virtual array struct must be PtrInfo::Array, got {ptr:?}");
        };
        assert!(
            Arc::ptr_eq(&array.descr, &descr),
            "forced array struct keeps the virtual array descr"
        );
        assert!(
            array.lenbound.is_constant(),
            "forced array struct length is the constant virtual length"
        );
        assert_eq!(array.lenbound.get_constant_int(), length as i64);
        assert!(
            array.items.is_empty(),
            "forced array struct forgets cached items, got {:?}",
            array.items
        );
        assert_eq!(array.last_guard_pos, -1);
        assert!(
            ctx.new_operations
                .iter()
                .any(|op| op.opcode == OpCode::SetinteriorfieldGc),
            "forcing an array struct emits SETINTERIORFIELD_GC for a written field"
        );
    }

    #[test]
    fn test_force_box_elides_alloc_covered_w_class_store() {
        let ops = force_virtual_with_w_class(0xCAFE, 8, 8);

        assert_eq!(ops[0].opcode, OpCode::NewWithVtable);
        assert_eq!(setfield_offsets(&ops), vec![16]);
    }

    #[test]
    fn test_force_box_keeps_reassigned_w_class_store() {
        let ops = force_virtual_with_w_class(0xBEEF, 8, 8);

        assert_eq!(ops[0].opcode, OpCode::NewWithVtable);
        assert_eq!(setfield_offsets(&ops), vec![8, 16]);
    }

    #[test]
    fn test_force_box_keeps_w_class_store_at_different_offset() {
        let ops = force_virtual_with_w_class(0xCAFE, 24, 8);

        assert_eq!(ops[0].opcode, OpCode::NewWithVtable);
        assert_eq!(setfield_offsets(&ops), vec![24, 16]);
    }

    #[test]
    fn new_with_vtable_publishes_its_initialized_w_class_on_ptr_info() {
        let init_field: Arc<dyn FieldDescr> = Arc::new(ForceFieldDescr {
            offset: 8,
            field_size: 8,
            field_type: Type::Ref,
            name: "PyObject.w_class",
        });
        let size_descr: DescrRef = Arc::new(ForceSizeDescr {
            all_fields: vec![],
            gc_fields: vec![init_field],
            w_class: 0xCAFE,
        });

        let (_, entry) = allocation_initialized_w_class(&size_descr)
            .expect("NEW_WITH_VTABLE should publish its class header value");

        assert_eq!(entry.as_opref(), Some(OpRef::const_ptr(GcRef(0xCAFE))));
    }

    #[test]
    fn test_const_ptr_info_getlenbound_returns_none_at_base() {
        // The base `PtrInfo::getlenbound` returns None for `PtrInfo::Constant`
        // — the constant string-length lookup runs through
        // `EnsuredPtrInfo::Constant::getlenbound`, which threads in the
        // runtime `string_length_resolver`. Callers that bypass
        // EnsuredPtrInfo (and thus skip the resolver) must not get a
        // misleading nonnegative answer here.
        let mut info = PtrInfo::constant(GcRef(0x1000));

        assert_eq!(info.getlenbound(Some(0)), None);
        assert_eq!(info.getlenbound(Some(1)), None);
        assert_eq!(info.getlenbound(None), None);
    }

    #[test]
    fn test_str_ptr_info_virtual_variants() {
        let mut ctx = OptContext::new(32);
        let plain = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 2,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![None, None],
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(plain.is_virtual());

        let slice = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: Some(ctx.materialize_operand_at(OpRef::int_op(3))), // vstring.py:223: self.lgtop = length
            mode: 0,
            length: -1,
            variant: VStringVariant::Slice(VStringSliceInfo {
                s: ctx.materialize_operand_at(OpRef::int_op(1)),
                start: ctx.materialize_operand_at(OpRef::int_op(2)),
                lgtop: ctx.materialize_operand_at(OpRef::int_op(3)),
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(slice.is_virtual());

        let concat = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Concat(VStringConcatInfo {
                vleft: ctx.materialize_operand_at(OpRef::int_op(4)),
                vright: ctx.materialize_operand_at(OpRef::int_op(5)),
                _is_virtual: true,
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(concat.is_virtual());

        let ptr = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Ptr,
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert!(!ptr.is_virtual());
    }

    #[test]
    fn test_str_ptr_info_constant_string_spec_and_strgetitem() {
        let mut ctx = OptContext::new(16);
        let b = ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.make_constant_box(&b, Value::Int(97));
        let b = ctx.materialize_operand_at(OpRef::int_op(11));
        ctx.make_constant_box(&b, Value::Int(98));
        let b = ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.make_constant_box(&b, Value::Int(99));

        let info = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 3,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![
                    Some(ctx.materialize_operand_at(OpRef::int_op(10))),
                    Some(ctx.materialize_operand_at(OpRef::int_op(11))),
                    Some(ctx.materialize_operand_at(OpRef::int_op(12))),
                ],
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });

        assert_eq!(
            info.get_constant_string_spec(&ctx, 0),
            Some(vec![97, 98, 99])
        );
        assert_eq!(info.get_known_str_length(&ctx, 0), Some(3));
        assert_eq!(
            info.strgetitem(1, &ctx).map(|b| b.to_opref()),
            Some(OpRef::int_op(11))
        );
    }

    #[test]
    fn test_str_ptr_info_plain_constant_string_spec_rejects_intbound_constant_chars() {
        let mut ctx = OptContext::new(16);
        let ch = OpRef::int_op(10);
        let ch_box = ctx.materialize_operand_at(ch);
        ctx.setintbound(&ch_box, &IntBound::from_constant(97));

        assert_eq!(
            ctx.get_box_replacement_operand_opt(ch)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            Some(97),
            "test setup should expose a get_constant_box-style IntBound constant",
        );

        let info = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 1,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![Some(ctx.materialize_operand_at(ch))],
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });

        // vstring.py get_constant_string_spec checks `c.is_constant()` directly for Plain
        // strings; a constant IntBound is not enough here.
        assert_eq!(info.get_constant_string_spec(&ctx, 0), None);
    }

    #[test]
    fn test_str_ptr_info_slice_and_concat_dispatch() {
        let mut ctx = OptContext::new(32);
        let b = ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.make_constant_box(&b, Value::Int(97));
        let b = ctx.materialize_operand_at(OpRef::int_op(11));
        ctx.make_constant_box(&b, Value::Int(98));
        let b = ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.make_constant_box(&b, Value::Int(99));
        let b = ctx.materialize_operand_at(OpRef::int_op(20));
        ctx.make_constant_box(&b, Value::Int(1));
        let b = ctx.materialize_operand_at(OpRef::int_op(21));
        ctx.make_constant_box(&b, Value::Int(2));

        let source = OpRef::int_op(1);
        let source_box = ctx.materialize_operand_at(source);
        let source_chars = vec![
            Some(ctx.materialize_operand_at(OpRef::int_op(10))),
            Some(ctx.materialize_operand_at(OpRef::int_op(11))),
            Some(ctx.materialize_operand_at(OpRef::int_op(12))),
        ];
        ctx.set_ptr_info(
            &source_box,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: 3,
                variant: VStringVariant::Plain(VStringPlainInfo {
                    _chars: source_chars,
                }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let slice = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: Some(ctx.materialize_operand_at(OpRef::int_op(21))), // vstring.py:223: self.lgtop = length
            mode: 0,
            length: -1,
            variant: VStringVariant::Slice(VStringSliceInfo {
                s: ctx.materialize_operand_at(source),
                start: ctx.materialize_operand_at(OpRef::int_op(20)),
                lgtop: ctx.materialize_operand_at(OpRef::int_op(21)),
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        assert_eq!(slice.get_known_str_length(&ctx, 0), Some(2));
        assert_eq!(slice.get_constant_string_spec(&ctx, 0), Some(vec![98, 99]));
        assert_eq!(
            slice.strgetitem(0, &ctx).map(|b| b.to_opref()),
            Some(OpRef::int_op(11))
        );

        let concat = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Concat(VStringConcatInfo {
                vleft: ctx.materialize_operand_at(source),
                vright: ctx.materialize_operand_at(OpRef::int_op(2)),
                _is_virtual: true,
            }),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        let pos2 = ctx.materialize_operand_at(OpRef::int_op(2));
        let pos2_chars = vec![
            Some(ctx.materialize_operand_at(OpRef::int_op(11))),
            Some(ctx.materialize_operand_at(OpRef::int_op(12))),
        ];
        ctx.set_ptr_info(
            &pos2,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: 2,
                variant: VStringVariant::Plain(VStringPlainInfo { _chars: pos2_chars }),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        assert_eq!(concat.get_known_str_length(&ctx, 0), Some(5));
        assert_eq!(
            concat.get_constant_string_spec(&ctx, 0),
            Some(vec![97, 98, 99, 98, 99])
        );
        assert_eq!(
            concat.strgetitem(3, &ctx).map(|b| b.to_opref()),
            Some(OpRef::int_op(11))
        );
    }

    #[test]
    fn test_ptr_info_set_getfield() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);

        assert!(info.getfield(0).is_none());
        info.setfield(0, field_op(Type::Int, 10));
        assert_eq!(
            info.getfield(0).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(10))
        );
        info.setfield(0, field_op(Type::Int, 20)); // overwrite
        assert_eq!(
            info.getfield(0).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(20))
        );
        info.setfield(1, field_op(Type::Int, 30));
        assert_eq!(
            info.getfield(1).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(30))
        );
    }

    #[test]
    fn test_ptr_info_set_getitem() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_array(descr, 3, false);

        assert_eq!(
            info.getitem(0).and_then(|e| e.as_opref()),
            Some(OpRef::NONE)
        ); // initialized to NONE
        info.setitem(0, field_op(Type::Int, 10));
        assert_eq!(
            info.getitem(0).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(10))
        );
        info.setitem(2, field_op(Type::Int, 30));
        assert_eq!(
            info.getitem(2).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(30))
        );
        assert!(info.getitem(5).is_none()); // out of bounds
    }

    #[test]
    fn test_preamble_item_keeps_regular_array_item_visible() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::array(descr, crate::optimizeopt::intutils::IntBound::nonnegative());
        info.setitem(1, field_op(Type::Int, 77));
        assert_eq!(
            info.getitem(1).and_then(|e| e.as_opref()),
            Some(OpRef::int_op(77))
        );

        let mut replay = Op::new(
            OpCode::GetarrayitemGcI,
            &[
                crate::history::test_support::rooted_resop_operand(Type::Int, 10),
                majit_ir::operand::Operand::from_opref(OpRef::const_int(0)),
            ],
        );
        replay.pos().set(OpRef::int_op(88));
        let pop = PreambleOp {
            op: majit_ir::operand::Operand::bound_from_opref(OpRef::int_op(88)),
            invented_name: false,
            preamble_op: OpRc::new(replay),
            same_as_source: None,
        };
        info.set_preamble_item(1, pop.clone());

        assert!(info.has_preamble_item(1));
        // After set_preamble_item, getitem returns Preamble (not the old Value)
        assert!(info.getitem(1).is_some_and(|e| e.is_preamble()));
        let recovered = info
            .take_preamble_item(1)
            .expect("preamble item should be recoverable");
        assert_eq!(recovered.op.to_opref(), OpRef::int_op(88));
        // After take_preamble_item, slot is Value(NONE)
        assert_eq!(
            info.getitem(1).and_then(|e| e.as_opref()),
            Some(OpRef::NONE)
        );
    }

    #[test]
    fn test_all_items_exposes_preamble_source_box() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::instance(Some(descr), None);
        let replay = Op::new(
            OpCode::GetfieldGcI,
            &[crate::history::test_support::rooted_resop_operand(
                Type::Int,
                10,
            )],
        );
        let pop = PreambleOp {
            op: majit_ir::operand::Operand::bound_from_opref(OpRef::int_op(88)),
            invented_name: false,
            preamble_op: OpRc::new(replay),
            same_as_source: None,
        };
        info.set_preamble_field(3, pop);

        // all_items includes Preamble entries (RPython parity: _fields returns raw)
        let items = info.all_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, 3);
        assert!(items[0].1.is_preamble());
    }

    #[test]
    fn test_ptr_info_guard_opcodes() {
        let nonnull = PtrInfo::nonnull();
        let guards = nonnull.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnull));

        let constant = PtrInfo::constant(GcRef(0x1000));
        let guards = constant.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardValue));

        let kc = PtrInfo::known_class(0x2000, true);
        let guards = kc.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnullClass));
    }

    #[test]
    fn test_ptr_info_visitor_walk() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.setfield(0, field_op(Type::Int, 10));
        info.setfield(1, field_op(Type::Int, 20));
        let refs = info.visitor_walk_recursive();
        assert_eq!(refs, vec![OpRef::int_op(10), OpRef::int_op(20)]);
    }

    #[test]
    fn test_opinfo_is_nonnull() {
        assert!(!OpInfo::Unknown.is_nonnull());
        assert!(OpInfo::ptr(PtrInfo::nonnull()).is_nonnull());
    }

    #[test]
    fn test_opinfo_float_const() {
        let info = OpInfo::FloatConstInfo(FloatConstInfo::new(3.25));
        assert!(info.is_constant());
        assert_eq!(info.get_constant_float(), Some(3.25));
    }
}
