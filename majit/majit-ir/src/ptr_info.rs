//! Building blocks for `PtrInfo` — the pointer-analysis info type
//! attached to each `_forwarded` slot. Hosted in `majit-ir` so the
//! `Forwarded` move that follows can reference these types without
//! a `majit-metainterp → majit-ir` circular dep.
//!
//! Pure data + leaf methods only. Methods that need `Op` / `OptContext`
//! from `majit-metainterp` live as extension traits in
//! `metainterp::optimizeopt::info`.

use crate::field_entry::{FieldEntry, PreambleOp};
use crate::intbound::IntBound;
use crate::operand::Operand;
use crate::rawbuffer::{InvalidRawOperation, RawBuffer};
use crate::{DescrRef, GcRef, Op, OpCode, OpRef, RdVirtualInfo, Type};

fn lookup_field_descr(field_descrs: &[DescrRef], field_idx: u32) -> Option<DescrRef> {
    field_descrs.get(field_idx as usize).cloned()
}

/// Sparse virtual/instance field lists stay at 1–4 entries on the
/// regex `and`/`or` leaf. A `Vec` grow 16→32→64 is the leftover
/// 0.16 allocs/char of 64 B (and the matching 16/32 B classes).
pub type VirtualFieldList = smallvec::SmallVec<[(u32, Operand); 4]>;
pub type CachedFieldList = smallvec::SmallVec<[(u32, FieldEntry); 4]>;

/// info.py `init_fields`: one list sized to the descr, not a grow
/// chain. Kept for the remaining Vec field lists (virtualizable heap
/// fields). Virtual/instance lists are inline and do not reserve.
#[allow(dead_code)]
fn reserve_assoc<T>(fields: &mut Vec<T>, n: usize) {
    if n > fields.capacity() {
        fields.reserve(n - fields.len());
    }
}

fn push_assoc<V>(fields: &mut Vec<(u32, V)>, field_idx: u32, value: V, reserve_n: usize) {
    for entry in fields.iter_mut() {
        if entry.0 == field_idx {
            entry.1 = value;
            return;
        }
    }
    if fields.capacity() == 0 {
        fields.reserve(reserve_n.max(1));
    }
    fields.push((field_idx, value));
}

fn push_assoc_virtual(fields: &mut VirtualFieldList, field_idx: u32, value: Operand) {
    for entry in fields.iter_mut() {
        if entry.0 == field_idx {
            entry.1 = value;
            return;
        }
    }
    fields.push((field_idx, value));
}

fn push_assoc_cached(fields: &mut CachedFieldList, field_idx: u32, value: FieldEntry) {
    for entry in fields.iter_mut() {
        if entry.0 == field_idx {
            entry.1 = value;
            return;
        }
    }
    fields.push((field_idx, value));
}

#[allow(dead_code)]
fn descr_field_len(descr: Option<&DescrRef>) -> usize {
    descr
        .and_then(|d| d.as_size_descr())
        .map(|sd| sd.all_fielddescrs().len())
        .unwrap_or(8)
}

/// info.py `reasonable_array_index(index)` — sanity gate on a
/// constant array index or array size. Returns false for negative
/// values or values above 150_000 so invalid loops and pathological
/// allocations are not optimized.
///
/// Used by `virtualize.py:28` (NEW_ARRAY size gate) and
/// `info.py` (per-element initialization gate).
pub fn reasonable_array_index(index: i64) -> bool {
    (0..=150_000).contains(&index)
}

/// Runtime hook for `ConstPtrInfo.getstrlen1(mode)` (info.py).
/// Returns `Some(length)` when `gcref` points at a known string of the
/// requested mode, `None` otherwise.
pub type StringLengthResolver = std::sync::Arc<dyn Fn(GcRef, u8) -> Option<i64> + Send + Sync>;

/// info.py `ConstPtrInfo._unpack_str(mode)` — runtime hook for
/// extracting characters from a constant string GcRef. Returns the char
/// values as `Vec<i64>`. Set by the host runtime (pyre etc.).
pub type StringContentResolver =
    std::sync::Arc<dyn Fn(GcRef, u8) -> Option<Vec<i64>> + Send + Sync>;

/// history.py `get_const_ptr_for_string(s)` — runtime hook for
/// creating a constant string GcRef from char values. The bool indicates
/// unicode (true) vs byte-string (false). Set by the host runtime.
pub type StringConstantAllocator = std::sync::Arc<dyn Fn(&[i64], bool) -> GcRef + Send + Sync>;

/// info.py: `AbstractVirtualPtrInfo` (RPython base class hint). Pyre
/// hoists only the fields shared by every Virtual* variant so each
/// `PtrInfo::Virtual*` carries a single embedded slot instead of N
/// independent copies of the same field set.
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
#[derive(Clone, Debug, Default)]
pub struct AbstractVirtualPtrInfo {
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<RdVirtualInfo>>>,
}

impl AbstractVirtualPtrInfo {
    pub fn new() -> Self {
        Self {
            cached_vinfo: std::cell::RefCell::new(None),
        }
    }
}

/// vstring.py: StrPtrInfo
#[derive(Clone, Debug)]
pub struct StrPtrInfo {
    /// vstring.py: self.lenbound — IntBound for string length.
    pub lenbound: Option<IntBound>,
    /// vstring.py self.lgtop — cached length OpRef (set by getstrlen).
    /// After force_box, this preserves the computed length so subsequent
    /// STRLEN queries reuse it instead of emitting a new STRLEN op.
    pub lgtop: Option<Operand>,
    /// vstring.py: self.mode — 0 = mode_string, 1 = mode_unicode.
    pub mode: u8,
    /// vstring.py: self.length — known exact length (-1 if unknown).
    pub length: i32,
    /// vstring.py: subclass-specific state
    /// (`VStringPlainInfo` / `VStringSliceInfo` / `VStringConcatInfo`).
    pub variant: VStringVariant,
    /// info.py:91-92: last_guard_pos
    pub last_guard_pos: i32,
    /// info.py `AbstractVirtualPtrInfo._cached_vinfo` — inherited
    /// through `StrPtrInfo(AbstractVirtualPtrInfo)` (vstring.py).
    /// Lifted into `AbstractVirtualPtrInfo` per RPython `_attrs_`
    /// inheritance contract; `make_virtual_info` dedups across finish()
    /// calls by comparing fieldnums (resume.py).
    pub avpi: AbstractVirtualPtrInfo,
}

impl StrPtrInfo {
    /// vstring.py / 227 / 278 `is_virtual()` on the string ptrinfo classes.
    pub fn is_virtual(&self) -> bool {
        match &self.variant {
            VStringVariant::Ptr => false,
            VStringVariant::Plain(_) | VStringVariant::Slice(_) => true,
            VStringVariant::Concat(info) => info._is_virtual,
        }
    }
}

/// vstring.py subclass state carried by `StrPtrInfo`.
#[derive(Clone, Debug)]
pub enum VStringVariant {
    /// Non-virtual base `StrPtrInfo`.
    Ptr,
    /// vstring.py `VStringPlainInfo`.
    Plain(VStringPlainInfo),
    /// vstring.py `VStringSliceInfo`.
    Slice(VStringSliceInfo),
    /// vstring.py `VStringConcatInfo`.
    Concat(VStringConcatInfo),
}

/// vstring.py `VStringPlainInfo`
#[derive(Clone, Debug)]
pub struct VStringPlainInfo {
    pub _chars: Vec<Option<Operand>>,
}

/// vstring.py `VStringSliceInfo`
#[derive(Clone, Debug)]
pub struct VStringSliceInfo {
    pub s: Operand,
    pub start: Operand,
    pub lgtop: Operand,
}

/// vstring.py `VStringConcatInfo`
#[derive(Clone, Debug)]
pub struct VStringConcatInfo {
    pub vleft: Operand,
    pub vright: Operand,
    pub _is_virtual: bool,
}

/// A virtual object whose allocation has been removed.
///
/// Fields are tracked as OpRefs to the operations that produce their values.
///
/// ## Invariant: an entry's key is a slot of `all_fielddescrs()`
///
/// Matches RPython upstream: `heaptracker.py all_fielddescrs()` skips
/// `typeptr`, so `info.py AbstractStructPtrInfo.init_fields` sizes
/// `_fields` with it excluded from the indexable range. The typeptr (offset 0)
/// is therefore never a key here; it is tracked via `known_class` and emitted
/// by the GC rewriter's `gen_initialize_vtable` path (rewrite.py), NOT from
/// the force-path field loop.
///
/// Pyre's additional Python-level `w_class` header is not the same case: a
/// layout that lists it holds it at an ordinary slot (`W_BaseException` at 1),
/// and an entry keyed by that slot resolves like any other. What must never
/// appear is a key that names no slot — `index_in_parent` off a gc-only header
/// edge (always 0, which is some value field), or one of
/// `heap::OptHeap::field_slot_index`'s header/unslotted bands, which are keys
/// for an association lookup and not positions in any list. `force_box` reads
/// this list back through `all_fielddescrs()` BY POSITION, so such a key
/// resolves to the wrong field or to nothing at all.
///
/// Enforced by:
/// - `virtualize.rs optimize_setfield_gc` Virtual arm: returns early on
///   typeptr, and keys a class-word store off the layout's
///   `class_word_index_in_parent` rather than the op's descr.
/// - `optimizeopt/mod.rs structinfo_setfield`: declines a header word on a
///   virtual, whose key there comes from `heap::OptHeap::field_slot_index`'s
///   band and names no slot.
/// - `virtualize.rs force_virtual_instance`: `debug_assert_no_typeptr`
///   at the entry of the field-emit loop.
/// - `virtualstate.rs export_single_value`:
///   `debug_assert_no_typeptr` on the fields collection boundary.
#[derive(Clone, Debug)]
pub struct VirtualInfo {
    /// The size descriptor of this object.
    pub descr: DescrRef,
    /// Known class, as the immortal vtable address (`ConstInt(ptr2int(typeptr))`,
    /// model.py:199-201). Held as a plain integer — never a traced `GcRef` —
    /// because the vtable is a prebuilt static the GC never moves.
    pub known_class: Option<i64>,
    /// ob_type field descriptor for force path. In RPython the vtable is
    /// set by allocate_with_vtable, not as a struct field. pyre stores
    /// ob_type at offset 0 explicitly. This descr lets force emit
    /// SetfieldGc(ob_type) without polluting `fields` (which feeds rd_virtuals).
    pub ob_type_descr: Option<DescrRef>,
    /// Field values: `(field_descr_index, value_opref)`.
    /// **Invariant**: every key is a slot of the descr's `all_fielddescrs()`;
    /// typeptr is never one — see struct-level docs.
    pub fields: VirtualFieldList,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `AbstractVirtualPtrInfo._cached_vinfo` inherited
    /// state. Lifted into `AbstractVirtualPtrInfo` per RPython `_attrs_`
    /// inheritance — see the [`AbstractVirtualPtrInfo`] doc.
    pub avpi: AbstractVirtualPtrInfo,
}

/// A virtual array.
#[derive(Clone, Debug)]
pub struct VirtualArrayInfo {
    /// The array descriptor.
    pub descr: DescrRef,
    /// Whether this was created by NewArrayClear (zero-initialized).
    pub clear: bool,
    /// Element values.
    pub items: Vec<Operand>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// A non-virtual object with cached field info.
///
/// Mirrors RPython's InstancePtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct InstancePtrInfo {
    /// Best-known instance descriptor, if any.
    pub descr: Option<DescrRef>,
    /// Known class, as the immortal vtable address (`ConstInt(ptr2int(typeptr))`,
    /// model.py:199-201) — a plain integer, not a traced `GcRef`.
    pub known_class: Option<i64>,
    /// info.py AbstractStructPtrInfo _fields — cached field values.
    /// RPython stores both normal Boxes and PreambleOp sentinels in the
    /// same list. Four inline slots keep the 16→32→64 grow off the heap.
    pub fields: CachedFieldList,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC struct with cached field info.
///
/// Mirrors RPython's StructPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct StructPtrInfo {
    /// Exact struct descriptor.
    pub descr: DescrRef,
    /// info.py _fields — cached field values (same as InstancePtrInfo).
    pub fields: CachedFieldList,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC array with cached item info and lenbound.
///
/// Mirrors RPython's ArrayPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct ArrayPtrInfo {
    /// Exact array descriptor.
    pub descr: DescrRef,
    /// Known bounds on the array length.
    pub lenbound: IntBound,
    /// info.py getitem _items — cached item values for constant indices.
    /// RPython stores both normal Boxes and PreambleOp sentinels.
    pub items: Vec<FieldEntry>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual struct (no vtable).
#[derive(Clone, Debug)]
pub struct VirtualStructInfo {
    /// The size descriptor.
    pub descr: DescrRef,
    /// Residual `new` does not stamp the class word. A typeptr SETFIELD on
    /// this vstruct is the header write `new_with_vtable` would have done;
    /// it is metadata here, never a `_fields` slot (`heaptracker.py
    /// all_fielddescrs` skips typeptr). `PassOn` of that store would force
    /// the vstruct before sibling fields are written.
    pub known_class: Option<i64>,
    /// Runtime typeptr value to write at `force_box`, when the store is not
    /// a compile-time constant. Immortal vtables stay in `known_class`.
    pub typeptr_value: Option<Operand>,
    /// Descr for the header store `force_box` emits from `typeptr_value`.
    pub ob_type_descr: Option<DescrRef>,
    /// Field values: (field_index, value, optional original field descriptor).
    pub fields: VirtualFieldList,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

impl VirtualStructInfo {
    pub fn new(descr: DescrRef) -> Self {
        Self {
            descr,
            known_class: None,
            typeptr_value: None,
            ob_type_descr: None,
            fields: VirtualFieldList::new(),
            last_guard_pos: -1,
            avpi: AbstractVirtualPtrInfo::new(),
        }
    }
}

/// A virtual array of structs (interior field access pattern).
///
/// Mirrors RPython's VArrayStructInfo where each array element
/// is a fixed-size struct with named fields. Used for RPython arrays
/// with complex item types (e.g., hash table entries with key+value fields).
#[derive(Clone, Debug)]
pub struct ArrayStructInfo {
    /// The array descriptor (arraydescr).
    pub descr: DescrRef,
    /// Per-element fields: outer Vec = elements, inner list = (field_descr_index, value_opref).
    pub element_fields: Vec<VirtualFieldList>,
    /// resume.py VArrayStructInfo.fielddescrs — InteriorFieldDescr per field.
    /// Used by _number_virtuals to extract item_size/field_offset/field_size.
    pub fielddescrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// info.py:RawSlicePtrInfo — alias view into a parent virtual raw buffer.
///
/// Created by `make_virtual_raw_slice` (virtualize.py) when an
/// `INT_ADD(rawbuf, const_offset)` is folded against a virtual raw buffer.
/// Reads / writes through a slice add `offset` to the requested byte
/// offset and forward to the parent buffer.
#[derive(Clone, Debug)]
pub struct RawSlicePtrInfo {
    /// Slice offset relative to the parent buffer's base. Signed because
    /// `info.py RawSlicePtrInfo.__init__(offset, parent)` accepts an
    /// unbounded RPython int — `optimize_INT_ADD` folds the addend as a
    /// signed `getint()` and a negative addend is a valid (if rare)
    /// slice base.
    pub offset: i64,
    /// OpRef of the parent VirtualRawBuffer (or another VirtualRawSlice
    /// — `optimize_int_add` flattens chained slices when the underlying
    /// info is `RawBufferPtrInfo`/`RawSlicePtrInfo`).
    pub parent: Operand,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// info.py RawBufferPtrInfo — pointer info for virtual raw memory.
///
/// RPython stores the byte-write tracking in a separate `RawBuffer` object
/// (`self.buffer = RawBuffer(cpu, None)` in info.py). Rust mirrors
/// that by keeping the rawbuffer.py parallel-list state in `buffer`, while
/// this struct owns the RawBufferPtrInfo metadata.
#[derive(Clone, Debug)]
pub struct RawBufferPtrInfo {
    /// info.py:390 self.func — raw malloc function pointer.
    pub func: i64,
    /// info.py:391 self.size — size of the virtual raw buffer.
    pub size: usize,
    /// info.py:387/392 self.buffer — rawbuffer.py RawBuffer.
    pub buffer: RawBuffer,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py _force_elements: calldescr for CALL_I(func, size) raw malloc.
    /// Saved from the original CALL_I op during virtualization.
    pub calldescr: Option<DescrRef>,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

impl RawBufferPtrInfo {
    /// virtualize.py make_virtual_raw_memory creates RawBufferPtrInfo(cpu, func, size),
    /// whose constructor initializes `self.buffer = RawBuffer(cpu, None)`.
    pub fn new(func: i64, size: usize, calldescr: Option<DescrRef>) -> Self {
        Self {
            func,
            size,
            buffer: RawBuffer::new(),
            last_guard_pos: -1,
            calldescr,
            avpi: AbstractVirtualPtrInfo::new(),
        }
    }

    /// info.py RawBufferPtrInfo.getitem_raw delegates to RawBuffer.
    pub fn read_value(
        &self,
        offset: i64,
        length: usize,
        descr: &DescrRef,
    ) -> Result<Operand, InvalidRawOperation> {
        self.buffer.read_value(offset, length, descr)
    }

    /// info.py RawBufferPtrInfo.setitem_raw delegates to RawBuffer.
    pub fn write_value(
        &mut self,
        offset: i64,
        length: usize,
        descr: DescrRef,
        value: Operand,
    ) -> Result<(), InvalidRawOperation> {
        self.buffer.write_value(offset, length, descr, value)
    }
}

/// Tracked field state for a virtualizable object (interpreter frame).
///
/// Mirrors RPython's virtualizable handling in the optimizer:
/// the frame already exists on the heap, but during JIT execution its
/// fields are kept in registers. The optimizer tracks the current value
/// of each field so that redundant setfield/getfield ops are eliminated.
///
/// When the virtualizable is "forced" (escapes to non-JIT code), field
/// values are written back to the heap via SETFIELD_RAW ops.
#[derive(Clone, Debug)]
pub struct VirtualizableFieldState {
    /// Tracked static field values: (field_descr_index, current_value).
    /// Indices correspond to VirtualizableInfo::static_fields order.
    pub fields: Vec<(u32, Operand)>,
    /// Original field descriptors: (field_descr_index, original_descr).
    /// Used to emit correct SetfieldRaw ops when forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// Tracked array field values: (array_field_index, element_values).
    /// Indices correspond to VirtualizableInfo::array_fields order.
    pub arrays: Vec<(u32, Vec<Operand>)>,
    /// Ordinary (non-virtualizable) heap fields read off this object, keyed by
    /// `FieldDescr::index_in_parent` — the same `_fields` role
    /// `AbstractStructPtrInfo` plays at `info.py`.
    ///
    /// `fields` above is a different index space (`VirtualizableInfo::
    /// static_fields` order), so the two cannot share storage. Upstream needs
    /// no such split: it has no virtualizable-specific `PtrInfo` subclass, so
    /// a virtualizable frame carries a plain `InstancePtrInfo` and every field
    /// read off it lands in the one `_fields` list the heap cache consults.
    pub heap_fields: Vec<(u32, FieldEntry)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// info.py: `PtrInfo` hierarchy collapsed into a Rust enum.
///
/// Each variant corresponds to a concrete RPython subclass; the enum tag
/// replaces RPython's runtime class dispatch. Methods coupled to
/// `OptContext` / `majit_gc` / `VirtualVisitor` live as the `PtrInfoExt`
/// extension trait in `majit-metainterp::optimizeopt::info`. The methods
/// hosted here are pure leaves that depend only on `Op`/`OpRef`/`Descr`
/// already in `majit-ir`.
#[derive(Clone, Debug)]
pub enum PtrInfo {
    /// Known to be non-null, nothing else.
    /// info.py: NonNullPtrInfo
    NonNull {
        /// info.py: NonNullPtrInfo.last_guard_pos = -1
        last_guard_pos: i32,
    },
    /// Known constant pointer.
    /// info.py: ConstPtrInfo (does NOT inherit NonNullPtrInfo)
    Constant(GcRef),
    /// Non-virtual GC object with cached field info.
    /// info.py: InstancePtrInfo (is_virtual = False).
    /// `make_constant_class` results — class set, no descr, no fields —
    /// are also stored here as `Instance(descr=None, known_class=Some(...))`,
    /// matching PyPy's `info.InstancePtrInfo(None, class_const)` factory
    /// at optimizer.py.
    Instance(InstancePtrInfo),
    /// Non-virtual GC struct with cached field info.
    /// info.py: StructPtrInfo (is_virtual = False)
    Struct(StructPtrInfo),
    /// Non-virtual GC array with cached item info and lenbound.
    /// info.py: ArrayPtrInfo (is_virtual = False)
    Array(ArrayPtrInfo),
    /// Virtual object (allocation removed by the optimizer).
    /// info.py: InstancePtrInfo
    Virtual(VirtualInfo),
    /// Virtual array.
    /// info.py: ArrayPtrInfo
    VirtualArray(VirtualArrayInfo),
    /// Virtual struct (no vtable).
    /// info.py: StructPtrInfo
    VirtualStruct(VirtualStructInfo),
    /// Virtual array of structs (interior field access).
    /// info.py: ArrayStructInfo
    VirtualArrayStruct(ArrayStructInfo),
    /// Virtual raw buffer.
    /// info.py: RawBufferPtrInfo
    VirtualRawBuffer(RawBufferPtrInfo),
    /// Virtual raw slice (offset alias into a parent raw buffer).
    /// info.py: RawSlicePtrInfo
    VirtualRawSlice(RawSlicePtrInfo),
    /// Virtualizable object (interpreter frame).
    Virtualizable(VirtualizableFieldState),
    /// vstring.py: StrPtrInfo — string with known length bounds.
    /// Tracks lenbound (IntBound) and mode (string vs unicode).
    Str(StrPtrInfo),
}

/// vstring.py _visitor_walk_recursive / 255-257 / 319-324: enumerate the child OpRefs that
/// each `StrPtrInfo` variant registers via `_visitor_walk_recursive`.  Used
/// by the generic walkers (`PtrInfo::visitor_walk_recursive`, `num_fields`)
/// so Str-typed virtuals participate in GC rooting and resume encoding.
fn str_child_oprefs(s: &StrPtrInfo) -> Vec<OpRef> {
    match &s.variant {
        VStringVariant::Ptr => Vec::new(),
        VStringVariant::Plain(p) => p
            ._chars
            .iter()
            .filter_map(|slot| slot.as_ref().map(|b| b.to_opref()))
            .collect(),
        VStringVariant::Slice(sl) => {
            vec![sl.s.to_opref(), sl.start.to_opref(), sl.lgtop.to_opref()]
        }
        VStringVariant::Concat(c) => vec![c.vleft.to_opref(), c.vright.to_opref()],
    }
}

fn str_child_count(s: &StrPtrInfo) -> usize {
    match &s.variant {
        VStringVariant::Ptr => 0,
        VStringVariant::Plain(p) => p._chars.len(),
        VStringVariant::Slice(_) => 3,
        VStringVariant::Concat(_) => 2,
    }
}

impl PtrInfo {
    /// Visit every inline `ConstPtr.value` slot reachable from this PtrInfo.
    ///
    /// RPython stores these references in normal object fields below
    /// `PtrInfo`/virtual info objects, so the translated GC updates the fields
    /// in place. Pyre keeps the same structural data in Rust containers and
    /// must walk the actual `OpRef` / `GcRef` slots explicitly.
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_field(entry: &mut FieldEntry, visitor: &mut dyn FnMut(&mut GcRef)) {
            if let Some(pop) = entry.as_preamble_mut() {
                pop.op.walk_const_ptr_refs(visitor);
                pop.preamble_op.walk_const_ptr_refs_mut(visitor);
                // An invented ref-typed alias can carry a `ConstPtr`
                // `same_as_source`; walk it so a moving GC updates the
                // pointer before the cached preamble emits its `SameAs`.
                if let Some(src) = &pop.same_as_source {
                    src.walk_const_ptr_refs(visitor);
                }
            } else if let Some(b) = entry.as_value() {
                b.walk_const_ptr_refs(visitor);
            }
        }

        match self {
            PtrInfo::Constant(gcref) => visitor(gcref),
            PtrInfo::Instance(info) => {
                // known_class is an immortal vtable integer, not a traced ref.
                for (_, entry) in &mut info.fields {
                    visit_field(entry, visitor);
                }
            }
            PtrInfo::Struct(info) => {
                for (_, entry) in &mut info.fields {
                    visit_field(entry, visitor);
                }
            }
            PtrInfo::Array(info) => {
                for entry in &mut info.items {
                    visit_field(entry, visitor);
                }
            }
            PtrInfo::Virtual(info) => {
                // known_class is an immortal vtable integer, not a traced ref.
                for (_, b) in &info.fields {
                    b.walk_const_ptr_refs(visitor);
                }
            }
            PtrInfo::VirtualArray(info) => {
                for b in &info.items {
                    b.walk_const_ptr_refs(visitor);
                }
            }
            PtrInfo::VirtualStruct(info) => {
                if let Some(b) = &info.typeptr_value {
                    b.walk_const_ptr_refs(visitor);
                }
                for (_, b) in &info.fields {
                    b.walk_const_ptr_refs(visitor);
                }
            }
            PtrInfo::VirtualArrayStruct(info) => {
                for fields in &info.element_fields {
                    for (_, b) in fields {
                        b.walk_const_ptr_refs(visitor);
                    }
                }
            }
            PtrInfo::VirtualRawBuffer(info) => info.buffer.walk_const_ptr_refs(visitor),
            PtrInfo::VirtualRawSlice(info) => info.parent.walk_const_ptr_refs(visitor),
            PtrInfo::Virtualizable(info) => {
                for (_, b) in &info.fields {
                    b.walk_const_ptr_refs(visitor);
                }
                for (_, items) in &info.arrays {
                    for b in items {
                        b.walk_const_ptr_refs(visitor);
                    }
                }
            }
            PtrInfo::Str(info) => {
                if let Some(b) = info.lgtop.as_ref() {
                    b.walk_const_ptr_refs(visitor);
                }
                match &info.variant {
                    VStringVariant::Ptr => {}
                    VStringVariant::Plain(plain) => {
                        for slot in &plain._chars {
                            if let Some(b) = slot.as_ref() {
                                b.walk_const_ptr_refs(visitor);
                            }
                        }
                    }
                    VStringVariant::Slice(slice) => {
                        slice.s.walk_const_ptr_refs(visitor);
                        slice.start.walk_const_ptr_refs(visitor);
                        slice.lgtop.walk_const_ptr_refs(visitor);
                    }
                    VStringVariant::Concat(concat) => {
                        concat.vleft.walk_const_ptr_refs(visitor);
                        concat.vright.walk_const_ptr_refs(visitor);
                    }
                }
            }
            PtrInfo::NonNull { .. } => {}
        }
    }

    // ── Constructors (info.py: factory methods) ──

    /// Create a NonNull PtrInfo.
    pub fn nonnull() -> Self {
        PtrInfo::NonNull { last_guard_pos: -1 }
    }

    // ── info.py get_last_guard: last_guard_pos methods ──

    /// info.py: get_last_guard
    pub fn get_last_guard_pos(&self) -> Option<usize> {
        let pos = match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::Instance(i) => i.last_guard_pos,
            PtrInfo::Struct(s) => s.last_guard_pos,
            PtrInfo::Array(a) => a.last_guard_pos,
            PtrInfo::Virtual(v) => v.last_guard_pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos,
            PtrInfo::VirtualRawSlice(v) => v.last_guard_pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos,
            PtrInfo::Str(s) => s.last_guard_pos,
            PtrInfo::Constant(_) => return None,
        };
        if pos < 0 { None } else { Some(pos as usize) }
    }

    /// Raw last_guard_pos value as i32 (-1 if none).
    pub fn last_guard_pos(&self) -> Option<i32> {
        let pos = match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::Instance(i) => i.last_guard_pos,
            PtrInfo::Struct(s) => s.last_guard_pos,
            PtrInfo::Array(a) => a.last_guard_pos,
            PtrInfo::Virtual(v) => v.last_guard_pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos,
            PtrInfo::VirtualRawSlice(v) => v.last_guard_pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos,
            PtrInfo::Str(s) => s.last_guard_pos,
            PtrInfo::Constant(_) => return None,
        };
        Some(pos)
    }

    /// info.py: mark_last_guard
    pub fn set_last_guard_pos(&mut self, pos: i32) {
        match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos = pos,
            PtrInfo::Instance(i) => i.last_guard_pos = pos,
            PtrInfo::Struct(s) => s.last_guard_pos = pos,
            PtrInfo::Array(a) => a.last_guard_pos = pos,
            PtrInfo::Virtual(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualRawSlice(v) => v.last_guard_pos = pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos = pos,
            PtrInfo::Str(s) => s.last_guard_pos = pos,
            PtrInfo::Constant(_) => {}
        }
    }

    /// info.py: reset_last_guard_pos
    pub fn reset_last_guard_pos(&mut self) {
        self.set_last_guard_pos(-1);
    }

    /// Create a Constant PtrInfo.
    pub fn constant(gcref: GcRef) -> Self {
        PtrInfo::Constant(gcref)
    }

    /// `optimizer.py make_constant_class` parity. PyPy stores
    /// known-class state on `InstancePtrInfo` itself (with `descr=None` and
    /// an empty `_fields`).
    pub fn known_class(class_ptr: i64, _is_nonnull: bool) -> Self {
        PtrInfo::Instance(InstancePtrInfo {
            descr: None,
            known_class: Some(class_ptr),
            fields: CachedFieldList::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual InstancePtrInfo.
    pub fn instance(descr: Option<DescrRef>, known_class: Option<i64>) -> Self {
        PtrInfo::Instance(InstancePtrInfo {
            descr,
            known_class,
            fields: CachedFieldList::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual StructPtrInfo.
    pub fn struct_ptr(descr: DescrRef) -> Self {
        PtrInfo::Struct(StructPtrInfo {
            descr,
            fields: CachedFieldList::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual ArrayPtrInfo.
    pub fn array(descr: DescrRef, lenbound: IntBound) -> Self {
        PtrInfo::Array(ArrayPtrInfo {
            descr,
            lenbound,
            items: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a Virtual PtrInfo (allocation removed).
    pub fn virtual_obj(descr: DescrRef, known_class: Option<i64>) -> Self {
        PtrInfo::Virtual(VirtualInfo {
            descr,
            known_class,
            ob_type_descr: None,
            fields: VirtualFieldList::new(),
            last_guard_pos: -1,
            avpi: AbstractVirtualPtrInfo::new(),
        })
    }

    /// Create a VirtualArray PtrInfo.
    pub fn virtual_array(descr: DescrRef, length: usize, clear: bool) -> Self {
        PtrInfo::VirtualArray(VirtualArrayInfo {
            descr,
            clear,
            items: vec![Operand::None; length],
            last_guard_pos: -1,
            avpi: AbstractVirtualPtrInfo::new(),
        })
    }

    /// Create a VirtualStruct PtrInfo.
    pub fn virtual_struct(descr: DescrRef) -> Self {
        PtrInfo::VirtualStruct(VirtualStructInfo::new(descr))
    }

    // ── Query methods ──

    /// Whether this pointer is known to be non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            PtrInfo::NonNull { .. } => true,
            PtrInfo::Constant(gcref) => !gcref.is_null(),
            PtrInfo::Instance(_)
            | PtrInfo::Struct(_)
            | PtrInfo::Array(_)
            | PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_)
            | PtrInfo::VirtualRawSlice(_)
            | PtrInfo::Virtualizable(_)
            | PtrInfo::Str(_) => true,
        }
    }

    /// Whether this pointer is a virtual (allocation removed).
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        match self {
            PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_) => true,
            PtrInfo::VirtualRawSlice(slice) => !slice.parent.is_none(),
            PtrInfo::Str(sinfo) => sinfo.is_virtual(),
            _ => false,
        }
    }

    /// `isinstance(x, info.AbstractVirtualPtrInfo)` — whether this info can
    /// carry cached fields/items, which is a WIDER test than `is_virtual`.
    ///
    /// `AbstractVirtualPtrInfo` (info.py) is the base of
    /// `AbstractStructPtrInfo` (:175, so `InstancePtrInfo` :313 and
    /// `StructPtrInfo`), `ArrayPtrInfo` (:496), `AbstractRawPtrInfo` (:374)
    /// and `StrPtrInfo` (vstring.py). Only `NonNullPtrInfo` (:90) and
    /// `ConstPtrInfo` (:706, which does not even inherit `NonNullPtrInfo`)
    /// fall outside it. A NON-virtual `InstancePtrInfo` therefore answers
    /// `True` here and `False` to `is_virtual`.
    ///
    /// Call sites that decide whether to REPLACE an existing info want this
    /// one: `is_virtual` would throw away a non-virtual `Instance` that
    /// already carries a known class or cached fields.
    pub fn is_abstract_virtual_ptr_info(&self) -> bool {
        !matches!(self, PtrInfo::NonNull { .. } | PtrInfo::Constant(_))
    }

    /// Whether this is a constant pointer.
    pub fn is_constant(&self) -> bool {
        matches!(self, PtrInfo::Constant(_))
    }

    /// Get constant GcRef value if this is a constant pointer.
    pub fn get_constant_ref(&self) -> Option<&GcRef> {
        match self {
            PtrInfo::Constant(r) => Some(r),
            _ => None,
        }
    }

    /// vstring.py `StrPtrInfo.getstrlen`: return self.lgtop — cached length
    /// box if available.
    pub fn get_cached_lgtop(&self) -> Option<Operand> {
        match self {
            PtrInfo::Str(info) => info.lgtop.clone(),
            _ => None,
        }
    }

    /// info.py ConstPtrInfo.getstrhash — closure-resolved variant.
    pub fn getstrhash<F>(&self, mode: u8, mut resolver: F) -> Option<i64>
    where
        F: FnMut(GcRef, u8) -> Option<i64>,
    {
        match self {
            PtrInfo::Constant(gcref) if !gcref.is_null() => resolver(*gcref, mode),
            _ => None,
        }
    }

    /// Count the number of fields/items in this virtual object.
    pub fn num_fields(&self) -> usize {
        match self {
            PtrInfo::Instance(v) => v.fields.len(),
            PtrInfo::Struct(v) => v.fields.len(),
            PtrInfo::Array(v) => v.items.len(),
            PtrInfo::Virtual(v) => v.fields.len(),
            PtrInfo::VirtualArray(v) => v.items.len(),
            PtrInfo::VirtualStruct(v) => v.fields.len(),
            PtrInfo::VirtualArrayStruct(v) => v.element_fields.iter().map(|f| f.len()).sum(),
            PtrInfo::VirtualRawBuffer(v) => v.buffer.len(),
            PtrInfo::Str(s) => str_child_count(s),
            _ => 0,
        }
    }

    /// Enumerate all OpRef values stored in this virtual's fields/items.
    pub fn visitor_walk_recursive(&self) -> Vec<OpRef> {
        match self {
            PtrInfo::Instance(v) => v.fields.iter().filter_map(|(_, e)| e.as_opref()).collect(),
            PtrInfo::Struct(v) => v.fields.iter().filter_map(|(_, e)| e.as_opref()).collect(),
            PtrInfo::Array(v) => v.items.iter().filter_map(|e| e.as_opref()).collect(),
            PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| r.to_opref()).collect(),
            PtrInfo::VirtualArray(v) => v.items.iter().map(|b| b.to_opref()).collect(),
            PtrInfo::VirtualStruct(v) => {
                let mut refs: Vec<OpRef> = v.fields.iter().map(|(_, r)| r.to_opref()).collect();
                if let Some(b) = &v.typeptr_value {
                    refs.push(b.to_opref());
                }
                refs
            }
            PtrInfo::VirtualArrayStruct(v) => v
                .element_fields
                .iter()
                .flat_map(|fields| fields.iter().map(|(_, r)| r.to_opref()))
                .collect(),
            PtrInfo::VirtualRawBuffer(v) => {
                v.buffer.values().iter().map(|b| b.to_opref()).collect()
            }
            PtrInfo::VirtualRawSlice(v) => vec![v.parent.to_opref()],
            PtrInfo::Virtualizable(v) => {
                let mut refs: Vec<OpRef> = v.fields.iter().map(|(_, r)| r.to_opref()).collect();
                for (_, items) in &v.arrays {
                    refs.extend(items.iter().map(|b| b.to_opref()));
                }
                refs
            }
            // vstring.py / 255-257 / 319-324: each `StrPtrInfo`
            // variant registers its child OpRefs via
            // `_visitor_walk_recursive`. Mirror that here so GC rooting
            // (`unroll.rs` exported_infos walk) and other generic-walker
            // consumers see them.
            PtrInfo::Str(s) => str_child_oprefs(s),
            _ => Vec::new(),
        }
    }

    /// info.py: force_at_the_end_of_preamble — recurses into pointer
    /// children of struct-like virtuals via the supplied closure.
    pub fn force_at_the_end_of_preamble<F>(&mut self, mut recurse: F)
    where
        F: FnMut(Operand) -> Operand,
    {
        match self {
            PtrInfo::Virtual(v) => {
                for (_, field) in &mut v.fields {
                    if !field.is_none() {
                        *field = recurse(field.clone());
                    }
                }
            }
            PtrInfo::VirtualStruct(v) => {
                if let Some(field) = v.typeptr_value.as_mut()
                    && !field.is_none()
                {
                    *field = recurse(field.clone());
                }
                for (_, field) in &mut v.fields {
                    if !field.is_none() {
                        *field = recurse(field.clone());
                    }
                }
            }
            PtrInfo::VirtualArray(v) => {
                for item in &mut v.items {
                    if !item.is_none() {
                        *item = recurse(item.clone());
                    }
                }
            }
            PtrInfo::VirtualArrayStruct(v) => {
                for fields in &mut v.element_fields {
                    for (_, field) in fields {
                        if !field.is_none() {
                            *field = recurse(field.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// info.py `_cached_vinfo` accessor — per-instance dedup cell.
    pub fn cached_vinfo(&self) -> Option<&std::cell::RefCell<Option<std::rc::Rc<RdVirtualInfo>>>> {
        match self {
            PtrInfo::Virtual(v) => Some(&v.avpi.cached_vinfo),
            PtrInfo::VirtualStruct(v) => Some(&v.avpi.cached_vinfo),
            PtrInfo::VirtualArray(v) => Some(&v.avpi.cached_vinfo),
            PtrInfo::VirtualArrayStruct(v) => Some(&v.avpi.cached_vinfo),
            PtrInfo::VirtualRawBuffer(v) => Some(&v.avpi.cached_vinfo),
            PtrInfo::VirtualRawSlice(v) => Some(&v.avpi.cached_vinfo),
            PtrInfo::Str(v) => Some(&v.avpi.cached_vinfo),
            _ => None,
        }
    }

    /// info.py `AbstractStructPtrInfo.init_fields`.
    pub fn all_fielddescrs_from_descr(&self) -> Vec<DescrRef> {
        let sd = match self {
            PtrInfo::Virtual(v) => v.descr.as_size_descr(),
            PtrInfo::VirtualStruct(v) => v.descr.as_size_descr(),
            PtrInfo::Instance(v) => v.descr.as_ref().and_then(|d| d.as_size_descr()),
            PtrInfo::Struct(v) => v.descr.as_size_descr(),
            _ => None,
        };
        match sd {
            Some(sd) => sd
                .all_fielddescrs()
                .iter()
                .map(|fd| std::sync::Arc::clone(fd) as std::sync::Arc<dyn crate::Descr>)
                .collect(),
            None => Vec::new(),
        }
    }

    /// info.py per-variant guard opcode lists used by structural diffing.
    pub fn guard_opcodes(&self) -> Vec<OpCode> {
        match self {
            PtrInfo::NonNull { .. } => vec![OpCode::GuardNonnull],
            PtrInfo::Instance(info) if info.known_class.is_some() => {
                vec![OpCode::GuardNonnullClass]
            }
            PtrInfo::Instance(info) if info.descr.is_some() => vec![
                OpCode::GuardNonnull,
                OpCode::GuardIsObject,
                OpCode::GuardSubclass,
            ],
            PtrInfo::Struct(_) | PtrInfo::Array(_) => {
                vec![OpCode::GuardNonnull, OpCode::GuardGcType]
            }
            PtrInfo::Constant(_) => vec![OpCode::GuardValue],
            _ => Vec::new(),
        }
    }

    /// info.py: is_null() — whether this pointer is known to be null.
    pub fn is_null(&self) -> bool {
        match self {
            PtrInfo::Constant(gcref) => gcref.is_null(),
            _ => false,
        }
    }

    /// info.py `PtrInfo.getnullness()` parity.
    pub fn getnullness(&self) -> i8 {
        if self.is_null() {
            crate::optimize::INFO_NULL
        } else if self.is_nonnull() {
            crate::optimize::INFO_NONNULL
        } else {
            crate::optimize::INFO_UNKNOWN
        }
    }

    /// `info.py` `PtrInfo.is_about_object()` default `False` /
    /// `info.py` `InstancePtrInfo.is_about_object()` override `True`.
    pub fn is_about_object(&self) -> bool {
        matches!(self, PtrInfo::Instance(_) | PtrInfo::Virtual(_))
    }

    /// `info.py` `PtrInfo.is_precise()` default `False` / virtual
    /// subclass override `True`.
    pub fn is_precise(&self) -> bool {
        matches!(
            self,
            PtrInfo::Instance(_)
                | PtrInfo::Struct(_)
                | PtrInfo::Array(_)
                | PtrInfo::Virtual(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_)
                | PtrInfo::VirtualRawSlice(_)
                | PtrInfo::Str(_)
        )
    }

    /// info.py: get_descr() — size/type descriptor for virtual objects.
    pub fn get_descr(&self) -> Option<&DescrRef> {
        match self {
            PtrInfo::Instance(v) => v.descr.as_ref(),
            PtrInfo::Struct(v) => Some(&v.descr),
            PtrInfo::Array(v) => Some(&v.descr),
            PtrInfo::Virtual(v) => Some(&v.descr),
            PtrInfo::VirtualArray(v) => Some(&v.descr),
            PtrInfo::VirtualStruct(v) => Some(&v.descr),
            PtrInfo::VirtualArrayStruct(v) => Some(&v.descr),
            _ => None,
        }
    }

    /// `getlenbound(mode)` polymorphic dispatch matching the PyPy class
    /// hierarchy. ConstPtrInfo's `getstrlen1(mode)` path is handled by
    /// `EnsuredPtrInfo` (which carries the runtime string-length resolver).
    pub fn getlenbound(&mut self, mode: Option<u8>) -> Option<IntBound> {
        match self {
            PtrInfo::Array(v) => {
                debug_assert!(
                    mode.is_none(),
                    "ArrayPtrInfo.getlenbound: mode must be None"
                );
                Some(v.lenbound.clone())
            }
            PtrInfo::VirtualArray(v) => {
                debug_assert!(
                    mode.is_none(),
                    "VirtualArrayInfo.getlenbound: mode must be None"
                );
                Some(IntBound::from_constant(v.items.len() as i64))
            }
            PtrInfo::VirtualArrayStruct(v) => {
                debug_assert!(
                    mode.is_none(),
                    "ArrayStructInfo.getlenbound: mode must be None"
                );
                Some(IntBound::from_constant(v.element_fields.len() as i64))
            }
            PtrInfo::Str(sinfo) => {
                if sinfo.lenbound.is_none() {
                    sinfo.lenbound = Some(if sinfo.length == -1 {
                        IntBound::nonnegative()
                    } else {
                        IntBound::from_constant(sinfo.length as i64)
                    });
                }
                sinfo.lenbound.clone()
            }
            _ => None,
        }
    }

    /// info.py `AbstractStructPtrInfo.init_fields` — upgrade the
    /// descr when a more-precise one shows up via `setfield`, and
    /// size `_fields` to `len(descr.get_all_fielddescrs())` so
    /// subsequent `setfield`s do not grow a 96 B `Vec` mid-guard.
    pub fn init_fields(&mut self, descr: DescrRef, index: usize) {
        let Some(size_descr) = descr.as_size_descr() else {
            return;
        };
        let new_len = size_descr.all_fielddescrs().len();
        match self {
            PtrInfo::Instance(v) => {
                let cur_len = v
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_size_descr())
                    .map(|sd| sd.all_fielddescrs().len())
                    .unwrap_or(0);
                if (v.descr.is_none() || index >= cur_len) && (cur_len == 0 || new_len > cur_len) {
                    v.descr = Some(descr);
                }
                // `(u32, FieldEntry)` is 24 B; reserving `new_len` for a
                // 6-field NodeRec was 144 B. Header-band keys are not
                // positions, so this list stays sparse and grows on
                // setfield (`info.py` `_fields[index] = op`).
            }
            PtrInfo::Struct(v) => {
                let cur_len = v
                    .descr
                    .as_size_descr()
                    .map(|sd| sd.all_fielddescrs().len())
                    .unwrap_or(0);
                if cur_len == 0 || (index >= cur_len && new_len > cur_len) {
                    v.descr = descr;
                }
                // Same 144 B spare as Instance: do not reserve `new_len`.
            }
            PtrInfo::Virtual(v) => {
                let cur_len = v
                    .descr
                    .as_size_descr()
                    .map(|sd| sd.all_fielddescrs().len())
                    .unwrap_or(0);
                if cur_len == 0 || (index >= cur_len && new_len > cur_len) {
                    v.descr = descr;
                }
            }
            PtrInfo::VirtualStruct(v) => {
                let cur_len = v
                    .descr
                    .as_size_descr()
                    .map(|sd| sd.all_fielddescrs().len())
                    .unwrap_or(0);
                if cur_len == 0 || (index >= cur_len && new_len > cur_len) {
                    v.descr = descr;
                }
            }
            _ => {}
        }
    }

    /// info.py: setfield(field_descr, value).
    pub fn setfield(&mut self, field_idx: u32, value: Operand) {
        match self {
            PtrInfo::Instance(v) => {
                push_assoc_cached(&mut v.fields, field_idx, FieldEntry::Value(value));
            }
            PtrInfo::Struct(v) => {
                push_assoc_cached(&mut v.fields, field_idx, FieldEntry::Value(value));
            }
            PtrInfo::Virtual(v) => {
                push_assoc_virtual(&mut v.fields, field_idx, value);
            }
            PtrInfo::VirtualStruct(v) => {
                push_assoc_virtual(&mut v.fields, field_idx, value);
            }
            PtrInfo::Virtualizable(v) => {
                push_assoc(&mut v.heap_fields, field_idx, FieldEntry::Value(value), 8);
            }
            _ => {}
        }
    }

    /// shortpreamble.py: HeapOp.produce_op stores PreambleOp in _fields.
    pub fn set_preamble_field(&mut self, field_idx: u32, pop: PreambleOp) {
        assert!(!self.is_virtual(), "set_preamble_field on virtual");
        match self {
            PtrInfo::Instance(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
                v.fields.push((field_idx, FieldEntry::preamble(pop)));
            }
            PtrInfo::Struct(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
                v.fields.push((field_idx, FieldEntry::preamble(pop)));
            }
            // The catch-all below re-seats `self` as an `InstancePtrInfo`,
            // which would drop the tracked virtualizable state. Store the
            // hoisted field alongside the other ordinary heap fields instead.
            PtrInfo::Virtualizable(v) => {
                v.heap_fields.retain(|(k, _)| *k != field_idx);
                v.heap_fields.push((field_idx, FieldEntry::preamble(pop)));
            }
            _ => {
                *self = PtrInfo::Instance(InstancePtrInfo {
                    descr: None,
                    known_class: None,
                    fields: smallvec::smallvec![(field_idx, FieldEntry::preamble(pop))],
                    last_guard_pos: -1,
                });
            }
        }
    }

    /// shortpreamble.py stores PreambleOp in array `_items[index]`.
    pub fn set_preamble_item(&mut self, index: usize, pop: PreambleOp) {
        assert!(!self.is_virtual(), "set_preamble_item on virtual");
        if let PtrInfo::Array(v) = self {
            if index >= v.items.len() {
                v.items.resize(index + 1, FieldEntry::Value(Operand::None));
            }
            v.items[index] = FieldEntry::preamble(pop);
        }
    }

    /// RPython: `isinstance(res, PreambleOp)` check in _getfield.
    pub fn has_preamble_field(&self, field_idx: u32) -> bool {
        match self {
            PtrInfo::Instance(v) => v
                .fields
                .iter()
                .any(|(k, e)| *k == field_idx && e.is_preamble()),
            PtrInfo::Struct(v) => v
                .fields
                .iter()
                .any(|(k, e)| *k == field_idx && e.is_preamble()),
            PtrInfo::Virtualizable(v) => v
                .heap_fields
                .iter()
                .any(|(k, e)| *k == field_idx && e.is_preamble()),
            _ => false,
        }
    }

    /// RPython: `isinstance(res, PreambleOp)` check in ArrayCachedItem._getfield.
    pub fn has_preamble_item(&self, index: usize) -> bool {
        match self {
            PtrInfo::Array(v) => v.items.get(index).is_some_and(|e| e.is_preamble()),
            _ => false,
        }
    }

    /// heap.py: CachedField._getfield detects PreambleOp in _fields.
    pub fn take_preamble_field(&mut self, field_idx: u32) -> Option<PreambleOp> {
        match self {
            PtrInfo::Instance(v) => {
                if let Some(pos) = v
                    .fields
                    .iter()
                    .position(|(k, e)| *k == field_idx && e.is_preamble())
                {
                    v.fields.remove(pos).1.into_preamble()
                } else {
                    None
                }
            }
            PtrInfo::Struct(v) => {
                if let Some(pos) = v
                    .fields
                    .iter()
                    .position(|(k, e)| *k == field_idx && e.is_preamble())
                {
                    v.fields.remove(pos).1.into_preamble()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// heap.py: ArrayCachedItem._getfield detects PreambleOp in
    /// `_items[index]`, forces it, and writes the resolved result back.
    pub fn take_preamble_item(&mut self, index: usize) -> Option<PreambleOp> {
        match self {
            PtrInfo::Array(v) => {
                if let Some(entry) = v.items.get_mut(index) {
                    if entry.is_preamble() {
                        let taken = std::mem::replace(entry, FieldEntry::Value(Operand::None));
                        taken.into_preamble()
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// heap.py:194: opinfo._fields[descr.get_index()] = None
    pub fn clear_field(&mut self, field_idx: u32) {
        match self {
            PtrInfo::Instance(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
            }
            PtrInfo::Struct(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
            }
            PtrInfo::Virtual(v) => v.fields.retain(|(k, _)| *k != field_idx),
            PtrInfo::VirtualStruct(v) => v.fields.retain(|(k, _)| *k != field_idx),
            PtrInfo::Virtualizable(v) => v.heap_fields.retain(|(k, _)| *k != field_idx),
            _ => {}
        }
    }

    /// info.py: all_items — returns _fields directly.
    pub fn all_items(&self) -> Vec<(u32, FieldEntry)> {
        match self {
            PtrInfo::Instance(v) => v.fields.to_vec(),
            PtrInfo::Struct(v) => v.fields.to_vec(),
            PtrInfo::Virtual(v) => v
                .fields
                .iter()
                .map(|(k, v)| (*k, FieldEntry::Value(v.clone())))
                .collect(),
            PtrInfo::VirtualStruct(v) => v
                .fields
                .iter()
                .map(|(k, v)| (*k, FieldEntry::Value(v.clone())))
                .collect(),
            PtrInfo::Array(v) => v
                .items
                .iter()
                .enumerate()
                .map(|(i, e)| (i as u32, e.clone()))
                .collect(),
            PtrInfo::VirtualArray(v) => v
                .items
                .iter()
                .enumerate()
                .map(|(i, val)| (i as u32, FieldEntry::Value(val.clone())))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// info.py AbstractStructPtrInfo.getfield.
    pub fn getfield(&self, field_idx: u32) -> Option<FieldEntry> {
        match self {
            PtrInfo::Instance(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, e)| e.clone()),
            PtrInfo::Struct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, e)| e.clone()),
            PtrInfo::Virtual(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| FieldEntry::Value(v.clone())),
            PtrInfo::VirtualStruct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| FieldEntry::Value(v.clone())),
            PtrInfo::Virtualizable(v) => v
                .heap_fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, e)| e.clone()),
            _ => None,
        }
    }

    /// info.py: setitem(index, value).
    pub fn setitem(&mut self, index: usize, value: Operand) {
        match self {
            PtrInfo::Array(v) => {
                if index >= v.items.len() {
                    v.items.resize(index + 1, FieldEntry::Value(Operand::None));
                }
                v.items[index] = FieldEntry::Value(value);
            }
            PtrInfo::VirtualArray(v)
                // info.py `if self.is_virtual(): return  # bogus
                // setarrayitem_gc into virtual, drop the operation`.
                if index < v.items.len() => {
                    v.items[index] = value;
                }
            _ => {}
        }
    }

    /// info.py: getitem(index).
    pub fn getitem(&self, index: usize) -> Option<FieldEntry> {
        match self {
            PtrInfo::Array(v) => v.items.get(index).cloned(),
            PtrInfo::VirtualArray(v) => v.items.get(index).map(|r| FieldEntry::Value(r.clone())),
            _ => None,
        }
    }

    /// heap.py: ArrayCachedItem.invalidate clears the cached slot.
    pub fn clear_item(&mut self, index: usize) {
        match self {
            PtrInfo::Array(v) => {
                if index < v.items.len() {
                    v.items[index] = FieldEntry::Value(Operand::None);
                }
            }
            PtrInfo::VirtualArray(v) if index < v.items.len() => {
                v.items[index] = Operand::None;
            }
            _ => {}
        }
    }

    /// info.py: getinteriorfield_virtual(index, fielddescr).
    pub fn getinteriorfield_virtual(
        &self,
        element_index: usize,
        field_descr_index: u32,
    ) -> Option<OpRef> {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    return None;
                }
                v.element_fields[element_index]
                    .iter()
                    .find(|(fdidx, _)| *fdidx == field_descr_index)
                    .map(|(_, b)| b.to_opref())
            }
            _ => None,
        }
    }

    /// info.py: setinteriorfield_virtual(index, fielddescr, fld).
    pub fn setinteriorfield_virtual(
        &mut self,
        element_index: usize,
        field_descr_index: u32,
        value: Operand,
    ) {
        if let PtrInfo::VirtualArrayStruct(v) = self {
            if element_index >= v.element_fields.len() {
                v.element_fields
                    .resize(element_index + 1, VirtualFieldList::new());
            }
            let fields = &mut v.element_fields[element_index];
            if let Some(entry) = fields
                .iter_mut()
                .find(|(fdidx, _)| *fdidx == field_descr_index)
            {
                entry.1 = value;
            } else {
                fields.push((field_descr_index, value));
            }
        }
    }

    /// info.py: produce_short_preamble_ops — register cached field reads
    /// into the short preamble builder.
    ///
    /// The emitted opcode tracks the field's declared type
    /// (`FieldDescr::field_type`) so ref / float fields land as
    /// `GetfieldGcR` / `GetfieldGcF`; otherwise the short preamble would
    /// reconstruct non-int virtual fields with the wrong result type.
    pub fn produce_short_preamble_ops(&self, structbox: crate::operand::Operand) -> Vec<Op> {
        let mut result = Vec::new();
        let field_descrs = self.all_fielddescrs_from_descr();
        let push_for = |result: &mut Vec<Op>, field_idx: u32, missing_msg: &str| {
            let descr = lookup_field_descr(&field_descrs, field_idx).expect(missing_msg);
            let tp = descr
                .as_field_descr()
                .map(|fd| fd.field_type())
                .unwrap_or(Type::Int);
            let opcode = OpCode::getfield_for_type(tp);
            result.push(Op::with_descr(
                opcode,
                std::slice::from_ref(&structbox),
                descr,
            ));
        };
        if let PtrInfo::Virtual(v) = self {
            for (field_idx, value) in &v.fields {
                if !value.is_none() {
                    push_for(
                        &mut result,
                        *field_idx,
                        "produce_short_preamble_ops: virtual field descr missing",
                    );
                }
            }
        }
        if let PtrInfo::VirtualStruct(v) = self {
            for (field_idx, value) in &v.fields {
                if !value.is_none() {
                    push_for(
                        &mut result,
                        *field_idx,
                        "produce_short_preamble_ops: virtual struct field descr missing",
                    );
                }
            }
        }
        result
    }

    #[cfg(test)]
    fn fields_capacity(&self) -> usize {
        match self {
            PtrInfo::Instance(v) => v.fields.capacity(),
            PtrInfo::Struct(v) => v.fields.capacity(),
            PtrInfo::Virtual(v) => v.fields.capacity(),
            PtrInfo::VirtualStruct(v) => v.fields.capacity(),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setfield_does_not_reserve_a_six_field_144b_list() {
        let mut info = PtrInfo::instance(None, None);
        let value = Operand::from_opref(OpRef::ConstInt(1));
        assert_eq!(
            info.fields_capacity(),
            4,
            "CachedFieldList must start with four inline slots, cap={}",
            info.fields_capacity()
        );
        for i in 0..4 {
            info.setfield(i, value.clone());
            assert_eq!(
                info.fields_capacity(),
                4,
                "four sparse setfields must stay inline, cap={}",
                info.fields_capacity()
            );
        }
    }

    #[test]
    fn field_list_pairs_stay_16b_so_four_fit_inline() {
        assert!(
            std::mem::size_of::<(u32, Operand)>() <= 16,
            "(u32, Operand) is {} B",
            std::mem::size_of::<(u32, Operand)>()
        );
        assert!(
            std::mem::size_of::<(u32, FieldEntry)>() <= 16,
            "(u32, FieldEntry) is {} B",
            std::mem::size_of::<(u32, FieldEntry)>()
        );
    }
}
