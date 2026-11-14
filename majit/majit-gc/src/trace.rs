//! Object tracing for GC reference discovery.
//!
//! During collection, the GC needs to find all GC references within
//! a live object so it can update them (for copying collection) or
//! mark the targets (for mark-sweep).
//!
//! Instead of a closure-based trace approach (which causes lifetime issues
//! with the borrow checker), we use an offset-based approach: each type
//! declares the offsets of its GC pointer fields relative to the object
//! payload start. The collector reads/writes GcRef values at these offsets
//! directly.
use majit_ir::GcRef;
use std::collections::HashMap;

/// One `gctypelayout.GCData.TYPE_INFO` entry of the materialized
/// type-info group (`GcLLDescr_framework.get_translated_info_for_typeinfo`,
/// `Assembler386.genop_guard_guard_is_object`).
///
/// Mirrors `GCData.TYPE_INFO` field-for-field.  The pointer-valued fields are
/// stored as native addresses so the row remains `Send` while retaining the
/// C/lltype memory shape consumed by generated code.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct TypeInfoLayout {
    pub infobits: usize,
    pub customdata: usize,
    pub fixedsize: usize,
    pub ofstoptrs: usize,
}

/// `GCData.CUSTOM_DATA_STRUCT` materialized behind
/// [`TypeInfoLayout::customdata`].
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct CustomDataLayout {
    pub customfunc: usize,
    pub memory_pressure_offset: isize,
}

/// The four fields appended by `GCData.VARSIZE_TYPE_INFO` after its
/// `TYPE_INFO header`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct VarSizeTypeInfoLayout {
    pub varitemsize: usize,
    pub ofstovar: usize,
    pub ofstolength: usize,
    pub varofstoptrs: usize,
}

/// One `rclass.CLASSTYPE` entry of the materialized type-info group.
///
/// RPython's `GcLLDescr_framework.add_vtable_after_typeinfo`
/// (gctypelayout.py) appends the vtable struct directly after
/// each `TYPE_INFO` entry. `genop_guard_guard_subclass`
/// (x86/assembler.py) then addresses `subclassrange_min` as
/// `base + (typeid << shift_by) + sizeof_ti + offset2` — i.e. "skip
/// past the TYPE_INFO to read the CLASSTYPE that follows it".
///
/// majit mirrors that layout: each registered type gets a `TypeEntry`
/// pair whose `TypeInfoLayout` is immediately followed by
/// `ClassTypeLayout`. `subclassrange_min` / `subclassrange_max` mirror
/// the equally-named `rclass.CLASSTYPE` fields that RPython populates
/// via preorder numbering (rtyper/normalizecalls.py:385-389).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct ClassTypeLayout {
    pub subclassrange_min: i64,
    pub subclassrange_max: i64,
}

/// The bytes following one [`TypeInfoLayout`] row.
///
/// `TypeLayoutBuilder.add_vtable_after_typeinfo` asserts that an RPython
/// instance carrying a `CLASSTYPE` is not varsize.  The same four-word tail can
/// therefore hold either the varsize extension or the paired class metadata.
#[repr(C)]
#[derive(Copy, Clone)]
pub union TypeEntryTail {
    pub varsize: VarSizeTypeInfoLayout,
    pub classtype: ClassTypeLayout,
    raw: [usize; 4],
}

impl Default for TypeEntryTail {
    fn default() -> Self {
        Self { raw: [0; 4] }
    }
}

/// One materialized type-info group slot.
///
/// RPython's heterogeneous group member is either `(TYPE_INFO, CLASSTYPE)` or
/// `VARSIZE_TYPE_INFO`.  majit's typeids are dense integer indices rather than
/// pre-scaled `GROUP_MEMBER_OFFSET`s, so every slot has the maximum eight-word
/// size and generated code can address it with one power-of-two shift.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct TypeEntry {
    pub type_info: TypeInfoLayout,
    pub tail: TypeEntryTail,
}

impl TypeInfoLayout {
    /// `T_MEMBER_INDEX` (gctypelayout.py). Lowest 16 bits of
    /// `infobits` carry the group member index; majit doesn't use
    /// this field today but reserves the bits to keep the bit layout
    /// compatible with the RPython constants below.
    pub const T_MEMBER_INDEX: usize = 0xffff;
    /// `T_IS_VARSIZE` (gctypelayout.py).
    pub const T_IS_VARSIZE: usize = 0x010000;
    /// `T_HAS_GCPTR_IN_VARSIZE` (gctypelayout.py).
    pub const T_HAS_GCPTR_IN_VARSIZE: usize = 0x020000;
    /// `T_IS_GCARRAY_OF_GCPTR` (gctypelayout.py).
    pub const T_IS_GCARRAY_OF_GCPTR: usize = 0x040000;
    /// `T_IS_WEAKREF` (gctypelayout.py).
    pub const T_IS_WEAKREF: usize = 0x080000;
    /// `T_IS_RPYTHON_INSTANCE` — the type is a subclass of OBJECT
    /// (gctypelayout.py). Stored in `infobits` and tested by
    /// `genop_guard_guard_is_object` via the byte mask returned by
    /// `gc_ll_descr.get_translated_info_for_guard_is_object`
    /// (gc.py:603-622).
    pub const T_IS_RPYTHON_INSTANCE: usize = 0x100000;
    /// `T_HAS_CUSTOM_TRACE` (gctypelayout.py).
    pub const T_HAS_CUSTOM_TRACE: usize = 0x200000;
    /// `T_HAS_OLDSTYLE_FINALIZER` (gctypelayout.py).
    pub const T_HAS_OLDSTYLE_FINALIZER: usize = 0x400000;
    /// Lightweight-destructor marker (bit 23, the otherwise-unused slot
    /// between `T_HAS_OLDSTYLE_FINALIZER` and `T_HAS_GCPTR`). RPython
    /// folds the lightweight-destructor case into its `q_finalizer`
    /// machinery rather than a distinct infobit; majit models the
    /// `TypeInfo.destructor` half with its own bit so the materialized
    /// table reflects which types carry a destructor.
    pub const T_HAS_DESTRUCTOR: usize = 0x800000;
    /// `T_HAS_GCPTR` (gctypelayout.py).
    pub const T_HAS_GCPTR: usize = 0x1000000;
    /// `T_HAS_MEMORY_PRESSURE` (gctypelayout.py) — first field is
    /// memory pressure field.
    pub const T_HAS_MEMORY_PRESSURE: usize = 0x2000000;
    /// `T_KEY_MASK` (gctypelayout.py) — bug detection mask.
    /// `_check_valid_type_info` asserts `infobits & T_KEY_MASK ==
    /// T_KEY_VALUE`.
    pub const T_KEY_MASK: usize = 0xFC000000;
    /// `T_KEY_VALUE` (gctypelayout.py) — bug detection sentinel
    /// stored in every valid `TYPE_INFO.infobits`.
    pub const T_KEY_VALUE: usize = 0x58000000;

    pub const INFOBITS_OFFSET: usize = 0;
    /// `rffi.sizeof(GCData.TYPE_INFO)` equivalent. Reported by
    /// `get_translated_info_for_typeinfo` as `sizeof_ti` so the
    /// backend `genop_guard_guard_subclass` formula
    /// `base + (typeid << shift_by) + sizeof_ti + offset2` lands on
    /// the `ClassTypeLayout` that follows this struct in memory.
    pub const SIZE_OF_TI: usize = std::mem::size_of::<TypeInfoLayout>();
}

impl ClassTypeLayout {
    pub const SUBCLASSRANGE_MIN_OFFSET: usize =
        std::mem::offset_of!(ClassTypeLayout, subclassrange_min);
    pub const SUBCLASSRANGE_MAX_OFFSET: usize =
        std::mem::offset_of!(ClassTypeLayout, subclassrange_max);
}

impl TypeEntry {
    /// Stride between consecutive materialized type entries. Used as
    /// the `<< shift_by` scale the backend applies to a small-integer
    /// typeid before adding it to the table base — see the
    /// `get_translated_info_for_typeinfo` docstring.
    pub const STRIDE: usize = std::mem::size_of::<TypeEntry>();
    /// `log2(STRIDE)`. The backend uses this as `shift_by`
    /// (`Assembler386.genop_guard_guard_is_object`) so the formula
    /// `base + (typeid << shift_by) + offset` lands on the right
    /// entry. RPython's 64-bit port sets `shift_by = 0` because its
    /// typeid is already pre-scaled to a byte offset
    /// (`GROUP_MEMBER_OFFSET`, translator/c/src/llgroup.h:36); majit
    /// keeps typeid as a small integer index and encodes the scale
    /// here instead.
    pub const SHIFT_BY: u8 = {
        // Compile-time assertion that STRIDE is a power of two.
        assert!(Self::STRIDE.is_power_of_two());
        Self::STRIDE.trailing_zeros() as u8
    };

    fn from_type_info(
        info: &TypeInfo,
        index: u32,
        customdata: usize,
        fixed_offsets: usize,
        var_offsets: usize,
    ) -> Self {
        let mut tail = TypeEntryTail::default();
        if info.item_size > 0 {
            tail.varsize = VarSizeTypeInfoLayout {
                varitemsize: info.item_size,
                ofstovar: info.size,
                ofstolength: info.length_offset,
                varofstoptrs: var_offsets,
            };
        } else {
            // Start from the all-zero raw tail so the two padding words in
            // majit's uniform eight-word slot do not contain indeterminate
            // bytes after writing the narrower CLASSTYPE member.
            tail.classtype = ClassTypeLayout {
                subclassrange_min: info.subclassrange_min,
                subclassrange_max: info.subclassrange_max,
            };
        }
        TypeEntry {
            type_info: TypeInfoLayout {
                infobits: encode_type_shape(info, index),
                customdata,
                fixedsize: info.size,
                ofstoptrs: fixed_offsets,
            },
            tail,
        }
    }
}

/// `gctypelayout.encode_type_shape` parity (gctypelayout.py).
///
/// Builds the `infobits` word from the type's structural flags. The
/// resulting bit layout is:
///
/// ```text
///   bits  0..15 : T_MEMBER_INDEX (typeid index)
///   bit   16    : T_IS_VARSIZE
///   bit   17    : T_HAS_GCPTR_IN_VARSIZE
///   bit   18    : T_IS_GCARRAY_OF_GCPTR
///   bit   19    : T_IS_WEAKREF
///   bit   20    : T_IS_RPYTHON_INSTANCE
///   bit   21    : T_HAS_CUSTOM_TRACE
///   bit   22    : T_HAS_OLDSTYLE_FINALIZER
///   bit   23    : T_HAS_DESTRUCTOR (majit lightweight-destructor bit)
///   bit   24    : T_HAS_GCPTR
///   bit   25    : T_HAS_MEMORY_PRESSURE
///   bits 26..31 : T_KEY_VALUE  (T_KEY_MASK sanity check)
/// ```
///
/// majit's `TypeInfo` carries the same structural flags the collector
/// inspector consumes. The bits are mapped one-for-one against the
/// RPython source for line-by-line parity:
///
/// ```python
///   infobits = index
///   if len(offsets) > 0: infobits |= T_HAS_GCPTR
///   if has custom trace: infobits |= T_HAS_CUSTOM_TRACE / T_HAS_OLDSTYLE_FINALIZER
///   if varsize: infobits |= T_IS_VARSIZE
///   if varsize array of gc ptrs: infobits |= T_IS_GCARRAY_OF_GCPTR
///   if varsize has gc ptrs: infobits |= T_HAS_GCPTR_IN_VARSIZE | T_HAS_GCPTR
///   if weakref: infobits |= T_IS_WEAKREF
///   if subclass of OBJECT: infobits |= T_IS_RPYTHON_INSTANCE
///   info.infobits = infobits | T_KEY_VALUE
/// ```
pub fn encode_type_shape(info: &TypeInfo, index: u32) -> usize {
    // gctypelayout.py `infobits = index` — typeid in low 16 bits.
    let mut infobits: usize = (index as usize) & TypeInfoLayout::T_MEMBER_INDEX;
    // gctypelayout.py: T_HAS_GCPTR if there are GC pointer
    // fields in the fixed part.
    if !info.gc_ptr_offsets.is_empty() {
        infobits |= TypeInfoLayout::T_HAS_GCPTR;
    }
    // `encode_type_shape` customdata / destructor / oldstyle finalizer /
    // memory pressure. `custom_trace` selects T_HAS_CUSTOM_TRACE
    // (info.py `q_has_custom_trace`); `old_style_finalizer` selects its own
    // mutually-exclusive bit below.
    if info.custom_trace.is_some() {
        infobits |= TypeInfoLayout::T_HAS_CUSTOM_TRACE;
        // jitframe.py registers JITFRAME via custom_trace and is also
        // T_HAS_GCPTR by virtue of holding GC refs the tracer
        // discovers; mark T_HAS_GCPTR so q_has_gcptr returns True for
        // it (gctypelayout.py).
        infobits |= TypeInfoLayout::T_HAS_GCPTR;
    }
    // `encode_type_shape` varsize encoding.
    if info.item_size > 0 {
        infobits |= TypeInfoLayout::T_IS_VARSIZE;
        if info.items_have_gc_ptrs {
            // `encode_type_shape`: a variable-size array carrying
            // GC pointers sets both flags.
            infobits |= TypeInfoLayout::T_HAS_GCPTR_IN_VARSIZE;
            infobits |= TypeInfoLayout::T_HAS_GCPTR;
            // `encode_type_shape`: pure GcArray-of-GcPtr.
            if info.gc_ptr_offsets.is_empty() {
                infobits |= TypeInfoLayout::T_IS_GCARRAY_OF_GCPTR;
            }
        }
    }
    // gctypelayout.py `if builder.is_weakref_type(TYPE):
    // infobits |= T_IS_WEAKREF`.
    if info.is_weakref {
        infobits |= TypeInfoLayout::T_IS_WEAKREF;
    }
    // gctypelayout.py `is_subclass_of_object`.
    if info.is_object {
        infobits |= TypeInfoLayout::T_IS_RPYTHON_INSTANCE;
    }
    // Lightweight-destructor marker (majit's half of the
    // gctypelayout.py:245-257 destructor surface).
    if info.destructor.is_some() {
        infobits |= TypeInfoLayout::T_HAS_DESTRUCTOR;
    }
    if info.old_style_finalizer.is_some() {
        infobits |= TypeInfoLayout::T_HAS_OLDSTYLE_FINALIZER;
    }
    // `encode_type_shape`: the translated type's custom data points at
    // its inherited `special_memory_pressure` field.
    if info.memory_pressure_offset.is_some() {
        infobits |= TypeInfoLayout::T_HAS_MEMORY_PRESSURE;
    }
    // gctypelayout.py — T_KEY_VALUE sanity tag.
    infobits | TypeInfoLayout::T_KEY_VALUE
}

/// One element of a `TotalOrderSymbolic.orderwitness` list
/// (rtyper/normalizecalls.py). RPython mixes Python ints
/// (the `_unique_cdef_id` of each MRO entry) with the float
/// `MAX = 1E100` sentinel that's appended to the maxid witness, and
/// relies on tuple comparison treating any int as smaller than `MAX`.
///
/// majit replaces the implicit-Python type juggling with an explicit
/// enum: `Cdef(_) < Max` follows from the variant order under the
/// derived `Ord` impl, which line-by-line preserves RPython's
/// "real id sorts before MAX sentinel" invariant.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum WitnessElement {
    Cdef(u32),
    Max,
}

/// lltype `TYPE` identity `TypeLayoutBuilder.id_of_type` is keyed by.
///
/// The referenced object must be a `static` (not a `const`, whose address
/// is not unique).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct GcTypeKey(usize);

impl GcTypeKey {
    /// Address of a `'static` object used as the `id_of_type` key.
    /// The referenced object must be a `static` (not a `const`, whose
    /// address is not unique).
    pub fn of<T>(identity: &'static T) -> Self {
        Self(identity as *const T as usize)
    }
}

/// Registry mapping type IDs to their type descriptors and object sizes.
///
/// Mirrors RPython's `gctypelayout.TypeLayoutBuilder` and the surrounding
/// `type_info_group` machinery (gctypelayout.py). The translator
/// adds members during the "build types" phase and freezes the table
/// before any compiled code runs; majit reproduces the same lifecycle
/// with `register` (matches `gctypelayout.add_vtable_after_typeinfo`)
/// and `freeze_types` (matches the implicit translator-end freeze
/// captured by `can_add_new_types`).
pub struct TypeRegistry {
    /// One `TypeInfo` per registered class. Indexed by typeid as a small
    /// integer (the majit-internal counterpart to `_unique_cdef_id`).
    entries: Vec<TypeInfo>,
    /// `gctypelayout.py` `TypeLayoutBuilder.id_of_type` — `{LLTYPE: type_id}`.
    id_of_type: HashMap<GcTypeKey, u32>,
    /// `gctypelayout.type_info_group` parity — the contiguous
    /// `(TYPE_INFO, CLASSTYPE)` member array the backend reads at
    /// codegen time via `get_translated_info_for_typeinfo`.
    ///
    /// Mutable build-time rows. RPython's `TypeLayoutBuilder` grows its
    /// `type_info_group` while `can_add_new_types` is true, then
    /// `close_table()` makes the completed group immutable. Majit follows the
    /// same two-phase lifecycle: this Vec may reallocate while registering,
    /// and `freeze_types` moves it into `frozen_layout_table`.
    layout_table: Vec<TypeEntry>,
    /// Exact-size, stable-address table published by `freeze_types`. Compiled
    /// guards may embed this slice's base pointer; a boxed slice cannot grow or
    /// reallocate and is reclaimed with the registry.
    frozen_layout_table: Option<Box<[TypeEntry]>>,
    /// Immortal `OFFSETS_TO_GC_PTR` arrays referenced by materialized rows.
    /// Word zero is the lltype Array length, followed by Signed offsets.
    offset_tables: Vec<Box<[usize]>>,
    /// Immortal `CUSTOM_DATA_STRUCT` objects referenced by materialized rows.
    custom_data: Vec<Box<CustomDataLayout>>,
    /// `gctypelayout.can_add_new_types` parity. `true` until the
    /// frontend calls `freeze_types`; after that, `register_type`
    /// panics. Mirrors how RPython's translator stops accepting new
    /// types at the end of `make_type_info_group` /
    /// `encode_type_shapes_now` (gctypelayout.py).
    can_add_new_types: bool,
    /// Set once `assign_inheritance_ids_now` has run
    /// (`normalizecalls.py assign_inheritance_ids`). `register` then
    /// rejects a type with `has_subclass_range`, because those preorder
    /// bounds are the whole classdef census. Types without a subclass
    /// range can still be registered until `freeze_types`.
    inheritance_ids_assigned: bool,
}

/// Custom trace function type.
///
/// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
/// `GCBase.trace` calls this from `_trace_slow_path` when
/// `has_custom_trace`, then still walks `offsets_to_gc_pointers`.
/// Used for types with dynamic GC reference layouts (e.g. JitFrame
/// with a gcmap bitmap).
///
/// `obj_addr` is the object payload start. The callback `f` must be
/// called for each GC reference slot address.
pub type CustomTraceFn = unsafe fn(obj_addr: usize, f: &mut dyn FnMut(*mut GcRef));

/// Lightweight destructor function type.
///
/// RPython parity: the `destructor` half of
/// `gctypelayout.py:245-257` (the `q_finalizer` / lightweight
/// destructor path). When set, the collector calls this on an object
/// that is about to be reclaimed — at minor collection for a dead
/// nursery object and at major collection for a dead old-gen object —
/// so a payload owning non-GC heap memory (for example a boxed buffer view)
/// gets its drop glue run instead of leaking. As in
/// incminimark's `deal_with_young/old_objects_with_destructors`
/// (incminimark.py deal_with_young_objects_with_destructors), a destructor "just" runs and does not
/// resurrect the object.
///
/// `obj_addr` is the object payload start (post-header), the same
/// convention as [`CustomTraceFn`].
pub type DestructorFn = unsafe fn(obj_addr: usize);

/// Information about a GC-managed type.
pub struct TypeInfo {
    /// Fixed size of the object (excluding header), or base size for varsize objects.
    pub size: usize,
    /// Whether this type contains GC pointers.
    pub has_gc_ptrs: bool,
    /// Byte offsets of GC pointer fields within the object payload (fixed part).
    pub gc_ptr_offsets: Vec<usize>,
    /// For variable-size objects: item size. 0 for fixed-size objects.
    pub item_size: usize,
    /// For variable-size objects: offset from object start to the length field.
    pub length_offset: usize,
    /// For variable-size objects: whether items contain GC pointers.
    /// Derived from `var_gc_ptr_offsets`; retained as the infobit/card-layout
    /// predicate used by the JIT-facing interfaces.
    pub items_have_gc_ptrs: bool,
    /// `VARSIZE_TYPE_INFO.varofstoptrs`: byte offsets of every GC pointer
    /// inside one variable-size item.  A plain `GcArray(GcRef)` is `[0]`; an
    /// array of structs may contain several entries at non-zero offsets.
    pub var_gc_ptr_offsets: Vec<usize>,
    /// RPython `rgc.register_custom_trace_hook` parity.
    /// `GCBase._trace_slow_path` runs this when `has_custom_trace`;
    /// `GCBase.trace` still walks `offsets_to_gc_pointers` afterwards.
    pub custom_trace: Option<CustomTraceFn>,
    /// `TypeRegistry::register` merges the parent's fixed
    /// `gc_ptr_offsets` into this object subclass even when
    /// `custom_trace` is set. Only
    /// `object_subclass_with_gc_ptrs_and_custom_trace` sets this, and
    /// only for a non-empty offset list. `object_subclass_with_custom_trace`
    /// leaves it clear: those hooks already walk the base fields, and
    /// merging them would trace the same slots twice.
    pub inherits_parent_offsets: bool,
    /// gc.py:642 `T_IS_RPYTHON_INSTANCE` parity. True when this type has
    /// `rclass.OBJECT` layout — the first word of the payload is the
    /// `typeptr` (ob_type) and `cls_of_box(gcref)` is valid. False for
    /// raw GC structs / arrays whose first word is not a class pointer.
    /// Consumed by `check_is_object(gcref)` (llmodel.py) and by
    /// `genop_guard_guard_is_object` (x86/assembler.py) which
    /// reads it through the materialized `TYPE_INFO` table's `infobits`
    /// byte (gc.py:631-642).
    pub is_object: bool,
    /// Whether this type corresponds to an RPython classdef and therefore
    /// contributes peers to `assign_inheritance_ids`. Some frontend-private
    /// storage layouts still carry an `rclass.OBJECT` header (and must set
    /// `is_object`) while sharing a classdef/vtable with another registered
    /// type; those layouts must not perturb the classdef census.
    pub has_subclass_range: bool,
    /// CPython-internal object that remains an OBJECT-layout GC boundary but
    /// is omitted from app-level `gc.get_objects()` enumeration.  This models
    /// 3.11+ interpreter frames: owning coroutine/traceback objects expose
    /// their relevant edges, while an unmaterialized execution frame is not
    /// itself reported as an app-level referrer.  The frame still has a
    /// typedef, so `referents._list_w_obj_referents` stops on it.
    pub hide_from_app_level_inspector: bool,
    /// `referents.py try_cast_gcref_to_w_root` returns None for a `W_Root`
    /// whose typedef is null (`MutableCell`, `WeakrefLifeline`).  Such an
    /// instance is not an app-level object: `gc.get_objects` omits it and
    /// `get_rpy_roots` / `get_rpy_referents` wrap the raw gcref, while
    /// `_list_w_obj_referents` looks through it to the value it holds.
    /// Distinct from `hide_from_app_level_inspector`, which still stops the
    /// referents walk.
    pub has_no_typedef: bool,
    /// One frontend-owned GC edge in the common OBJECT header that keeps
    /// runtime class metadata alive but is not an app-level referent.  PyPy's
    /// RPython objects carry their class in the non-GC `typeptr`; pyre's
    /// augmented `PyObject` header additionally stores the corresponding
    /// Python class in a managed `w_class` slot.  The collector must trace
    /// that slot, while `gc.get_referents()` must omit it.  Set this on the
    /// OBJECT root; subclasses inherit it through [`Self::parent`].
    pub app_level_inspector_hidden_edge_offset: Option<usize>,
    /// Parent class typeid in the `rclass.OBJECT` hierarchy, or
    /// `None` for `rclass.OBJECT` itself / non-OBJECT types.
    /// Mirrors what `classdef.getmro()` traverses in
    /// `rtyper/normalizecalls.py:assign_inheritance_ids`. The parent
    /// chain feeds into `freeze_types`, which sorts the types
    /// lexicographically by reversed-MRO and assigns
    /// `subclassrange_{min,max}` from the resulting peer order
    /// (TotalOrderSymbolic, normalizecalls.py).
    pub parent: Option<u32>,
    /// `rclass.CLASSTYPE.subclassrange_min` for this type's class —
    /// i.e. the lowest preorder index of the class subtree rooted at
    /// this type. Populated by `freeze_types`. Used by
    /// `genop_guard_guard_subclass` (x86/assembler.py) and
    /// `execute_guard_subclass` (llgraph/runner.py).
    /// Encoded into the materialized `TYPE_INFO` table at
    /// `subclassrange_min_offset`.
    pub subclassrange_min: i64,
    /// `rclass.CLASSTYPE.subclassrange_max` for this type's class —
    /// the exclusive upper bound of the same subtree. Populated by
    /// `freeze_types`.
    pub subclassrange_max: i64,
    /// `gctypelayout.is_weakref_type(TYPE)` parity
    /// (gctypelayout.py): set to `true` for the WEAKREF
    /// GcStruct itself (gctypelayout.py). `encode_type_shape`
    /// emits the `T_IS_WEAKREF` infobits bit (gctypelayout.py)
    /// so the collector's minor / major cycles can locate weakref
    /// objects when scanning the type-info group.
    pub is_weakref: bool,
    /// `gctypelayout.py:245-257` lightweight-destructor parity. When
    /// `Some`, an instance of this type is recorded on the GC's
    /// `young_objects_with_destructors` list at allocation, and the
    /// collector runs this hook when the instance dies (minor cycle for
    /// a dead nursery object, major cycle for a dead old-gen object) so
    /// payloads owning non-GC heap memory get their drop glue run. See
    /// [`DestructorFn`].
    pub destructor: Option<DestructorFn>,
    /// `gctypelayout.py`'s `old_style_finalizer` custom function. Unlike a
    /// lightweight destructor, this may run arbitrary code and resurrect its
    /// object. IncrementalMiniMark therefore allocates the object old and
    /// queues the callback until `GCBase.execute_finalizers`.
    pub old_style_finalizer: Option<DestructorFn>,
    /// `gctypelayout.get_memory_pressure_ofs(TYPE)` parity. `Some(offset)`
    /// marks a fixed-size type carrying the inherited
    /// `special_memory_pressure` signed field read by
    /// `inspector.count_memory_pressure`.
    pub memory_pressure_offset: Option<usize>,
}

impl TypeInfo {
    /// Create a type info for a fixed-size object with no GC pointers.
    pub fn simple(size: usize) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create a type info for a fixed-size object with no GC pointers
    /// that owns non-GC heap memory reclaimed by a lightweight
    /// `destructor`. The collector runs `destructor` when an instance
    /// dies (see [`DestructorFn`]).
    pub fn with_destructor(size: usize, destructor: DestructorFn) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: Some(destructor),
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Builder: attach a lightweight `destructor` (see [`DestructorFn`]) to a
    /// type built by another constructor. Composes with
    /// `object_subclass_with_custom_trace` for an instance that both owns
    /// inline GC refs discovered by the trace hook *and* an off-heap
    /// allocation whose drop glue must run when the instance dies — e.g. the
    /// `W_MemoryView` header owning its `*const BufferView` box.
    pub fn with_destructor_fn(mut self, destructor: DestructorFn) -> Self {
        assert!(
            self.old_style_finalizer.is_none(),
            "a type cannot have both a light and an old-style finalizer"
        );
        self.destructor = Some(destructor);
        self
    }

    /// Create a fixed-size type with PyPy's heavyweight, old-style finalizer.
    /// The collector forces instances out of the nursery and invokes this only
    /// after finalization ordering has kept the object graph alive.
    pub fn with_old_style_finalizer(size: usize, finalizer: DestructorFn) -> Self {
        Self::simple(size).with_old_style_finalizer_fn(finalizer)
    }

    /// Attach an old-style finalizer to another fixed-size type layout.
    pub fn with_old_style_finalizer_fn(mut self, finalizer: DestructorFn) -> Self {
        assert_eq!(self.item_size, 0, "finalizer types must be fixed-size");
        assert!(!self.is_weakref, "a finalizer type cannot be a weakref");
        assert!(
            self.custom_trace.is_none(),
            "an old-style finalizer cannot share the custom function slot"
        );
        assert!(
            self.destructor.is_none(),
            "a type cannot have both a light and an old-style finalizer"
        );
        self.old_style_finalizer = Some(finalizer);
        self
    }

    /// Attach the translated `special_memory_pressure` field offset.
    /// `gctypelayout.has_special_memory_pressure` rejects varsize types;
    /// preserve that invariant at registration time.
    pub fn with_memory_pressure_offset(mut self, offset: usize) -> Self {
        assert_eq!(
            self.item_size, 0,
            "memory-pressure types must be fixed-size"
        );
        assert!(
            offset
                .checked_add(std::mem::size_of::<isize>())
                .is_some_and(|end| end <= self.size),
            "memory-pressure field exceeds the object payload"
        );
        self.memory_pressure_offset = Some(offset);
        self
    }

    /// Keep an OBJECT-layout type as a referent boundary while omitting its
    /// instances from app-level `gc.get_objects()` enumeration.
    pub fn hidden_from_app_level_inspector(mut self) -> Self {
        self.hide_from_app_level_inspector = true;
        self
    }

    /// Mark an OBJECT-layout `W_Root` that has no typedef.
    /// `referents.py try_cast_gcref_to_w_root` rejects it, so app-level
    /// inspection expands the instance instead of stopping on it, and
    /// `gc.get_objects` does not list it.
    pub fn without_app_level_typedef(mut self) -> Self {
        self.has_no_typedef = true;
        self
    }

    /// Mark the frontend's managed mirror of RPython's non-GC type pointer.
    /// This changes app-level inspection only; ordinary GC tracing continues
    /// to visit the edge.
    pub fn with_app_level_inspector_hidden_edge(mut self, offset: usize) -> Self {
        assert!(
            offset
                .checked_add(std::mem::size_of::<crate::GcRef>())
                .is_some_and(|end| end <= self.size),
            "app-level inspector hidden edge exceeds the object payload"
        );
        self.app_level_inspector_hidden_edge_offset = Some(offset);
        self
    }

    /// `gctypelayout.is_weakref_type(WEAKREF)` parity — TypeInfo for
    /// the singleton WEAKREF GcStruct itself (gctypelayout.py).
    /// The struct payload is one `weakptr: GcRef` slot; the collector
    /// invalidates it during minor / major cycles instead of tracing
    /// it as a strong reference, so `gc_ptr_offsets` stays empty.
    pub fn weakref() -> Self {
        TypeInfo {
            size: crate::weakref::SIZEOF_WEAKREF,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: true,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create a type info for a fixed-size `rclass.OBJECT`-layout
    /// instance (gc.py:642 `T_IS_RPYTHON_INSTANCE`). The first word of
    /// the payload is the `typeptr` (ob_type). `cls_of_box` and
    /// `guard_class` rely on this invariant, and
    /// `check_is_object(gcref)` returns true for such types.
    ///
    /// The class is treated as a leaf with no `rclass.OBJECT` parent
    /// in this registry. Use `object_subclass(size, parent_typeid)`
    /// to attach the type into an `rclass.OBJECT` inheritance chain
    /// before calling `freeze_types`.
    pub fn object(size: usize) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: true,
            has_subclass_range: true,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create an `rclass.OBJECT`-layout root whose fixed payload also
    /// contains GC pointer fields.
    ///
    /// RPython's `rclass.OBJECT` itself contains only `typeptr`, but a
    /// frontend may extend that common root header with managed fields. The
    /// concrete subclass layout then inherits these offsets just as an
    /// RPython `GcStruct(..., ('super', OBJECT), ...)` inherits the GC fields
    /// of its embedded `super`.
    pub fn object_with_gc_ptrs(size: usize, offsets: Vec<usize>) -> Self {
        let has_gc_ptrs = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: true,
            has_subclass_range: true,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create a type info for an `rclass.OBJECT`-layout instance
    /// that inherits from a previously-registered class.
    ///
    /// Mirrors how RPython attaches `classdef.bases[]` to its
    /// inheritance graph; `freeze_types` later walks the parent chain
    /// to compute reversed-MRO witnesses
    /// (rtyper/normalizecalls.py:assign_inheritance_ids).
    pub fn object_subclass(size: usize, parent_typeid: u32) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: true,
            has_subclass_range: true,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// `object_subclass` variant for instances whose payload contains
    /// inline `GcRef`-typed fields. `offsets` lists their byte offsets
    /// within the struct (after `ob_type` at offset 0). The collector
    /// walks these during minor collection so the parent-chain
    /// inheritance information is preserved alongside the trace
    /// metadata.
    ///
    /// Use this when an `rclass.OBJECT`-shaped class has fields whose
    /// rtyped form is `Ptr(rclass.OBJECT)` — for example
    /// `pypy/interpreter/function.py Function`'s `closure`,
    /// `defs_w`, `w_kw_defs`, `w_module`.
    pub fn object_subclass_with_gc_ptrs(
        size: usize,
        parent_typeid: u32,
        offsets: Vec<usize>,
    ) -> Self {
        let has_gc_ptrs = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: true,
            has_subclass_range: true,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// `object_subclass` variant for instances whose GC references are
    /// discovered by a custom trace hook rather than fixed inline offsets.
    pub fn object_subclass_with_custom_trace(
        size: usize,
        parent_typeid: u32,
        trace_fn: CustomTraceFn,
    ) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: Some(trace_fn),
            inherits_parent_offsets: false,
            is_object: true,
            has_subclass_range: true,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// `object_subclass` variant with both fixed GC offsets and a custom
    /// trace. `GCBase.trace` walks `offsets_to_gc_pointers` after
    /// `_trace_slow_path` runs `custom_trace_dispatcher`.
    ///
    /// `offsets` are this subclass's own fixed slots. When that list is
    /// non-empty, `inherits_parent_offsets` makes `TypeRegistry::register`
    /// merge the parent's fixed offsets the same way an offset-traced
    /// subclass does. `object_subclass_with_custom_trace` does not set
    /// the flag: those hooks already walk the base fields.
    pub fn object_subclass_with_gc_ptrs_and_custom_trace(
        size: usize,
        parent_typeid: u32,
        offsets: Vec<usize>,
        trace_fn: CustomTraceFn,
    ) -> Self {
        let inherits_parent_offsets = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs: true,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: Some(trace_fn),
            inherits_parent_offsets,
            is_object: true,
            has_subclass_range: true,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create a type info for a fixed-size object with GC pointer fields.
    /// `offsets` lists byte offsets of GcRef fields within the payload.
    pub fn with_gc_ptrs(size: usize, offsets: Vec<usize>) -> Self {
        let has_gc_ptrs = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create a type info for a variable-size object.
    pub fn varsize(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        items_have_gc_ptrs: bool,
        gc_ptr_offsets: Vec<usize>,
    ) -> Self {
        let var_gc_ptr_offsets = if items_have_gc_ptrs {
            vec![0]
        } else {
            Vec::new()
        };
        Self::varsize_with_gc_ptr_offsets(
            base_size,
            item_size,
            length_offset,
            var_gc_ptr_offsets,
            gc_ptr_offsets,
        )
    }

    /// Create a variable-size type whose item is itself a structure.
    ///
    /// This is `q_varsize_offsets_to_gcpointers_in_var_part`: each offset is
    /// relative to the start of one item and every item is visited with the
    /// same offset table.
    pub fn varsize_with_gc_ptr_offsets(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        var_gc_ptr_offsets: Vec<usize>,
        gc_ptr_offsets: Vec<usize>,
    ) -> Self {
        for &offset in &var_gc_ptr_offsets {
            assert!(
                offset
                    .checked_add(std::mem::size_of::<crate::GcRef>())
                    .is_some_and(|end| end <= item_size),
                "variable-item GC pointer at offset {offset} exceeds item size {item_size}",
            );
        }
        let items_have_gc_ptrs = !var_gc_ptr_offsets.is_empty();
        TypeInfo {
            size: base_size,
            has_gc_ptrs: !gc_ptr_offsets.is_empty() || items_have_gc_ptrs,
            gc_ptr_offsets,
            item_size,
            length_offset,
            items_have_gc_ptrs,
            var_gc_ptr_offsets,
            custom_trace: None,
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Create a type info with a custom trace hook (fixed-size).
    ///
    /// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
    pub fn with_custom_trace(size: usize, trace_fn: CustomTraceFn) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: Some(trace_fn),
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Mark an `rclass.OBJECT`-shaped private storage layout that shares the
    /// classdef/vtable of another registered type. It remains visible to
    /// `check_is_object` and app-level GC inspection, but contributes no
    /// duplicate peer to the subclass-range numbering.
    pub fn object_layout_without_subclass_range(mut self) -> Self {
        self.is_object = true;
        self.has_subclass_range = false;
        self
    }

    /// Create a varsize type info with a custom trace hook.
    ///
    /// RPython parity: `rgc.register_custom_trace_hook` on a
    /// `GcStruct(..., Array(Signed))` — e.g. JITFRAME.
    pub fn varsize_with_custom_trace(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        trace_fn: CustomTraceFn,
    ) -> Self {
        TypeInfo {
            size: base_size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size,
            length_offset,
            // This constructor lists no variable-part offsets. `GCBase.trace`
            // still walks `offsets_to_gc_pointers` after `_trace_slow_path`.
            items_have_gc_ptrs: false,
            var_gc_ptr_offsets: Vec::new(),
            custom_trace: Some(trace_fn),
            inherits_parent_offsets: false,
            is_object: false,
            has_subclass_range: false,
            hide_from_app_level_inspector: false,
            has_no_typedef: false,
            app_level_inspector_hidden_edge_offset: None,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
            is_weakref: false,
            destructor: None,
            old_style_finalizer: None,
            memory_pressure_offset: None,
        }
    }

    /// Compute the total size of an instance (excluding GC header).
    /// RPython lltypelayout.py sizeof(TYPE, i):
    ///   fixedsize = get_fixed_size(TYPE)
    ///   varsize = get_variable_size(TYPE)
    ///   return fixedsize + i * varsize
    ///
    /// Both `get_fixed_size(lltype.Array)` and the `_size` field of a
    /// `get_layout(Struct-with-array)` already account for the length
    /// word, so no extra WORD is added here.
    pub fn total_instance_size(&self, length: usize) -> usize {
        self.size + self.item_size * length
    }

    /// Iterate all GC pointer slot addresses for a given object.
    ///
    /// # Safety
    /// `obj_addr` must point to valid memory of at least `self.size` bytes (plus
    /// variable part if applicable).
    pub unsafe fn for_each_gc_ptr(&self, obj_addr: usize, mut f: impl FnMut(*mut GcRef)) {
        // `GCBase.trace` always walks `offsets_to_gc_pointers` after
        // `_trace_slow_path`. The hook (`has_custom_trace`) and the
        // variable-part items add slots; neither replaces the fixed offsets.
        if let Some(trace_fn) = self.custom_trace {
            unsafe { trace_fn(obj_addr, &mut f) };
        }

        // Fixed-part GC pointer fields.
        for &offset in &self.gc_ptr_offsets {
            f((obj_addr + offset) as *mut GcRef);
        }

        // Variable-part GC pointer items.
        // Callers reject a forwarded header before this load: its type id
        // is outside the table (`FORWARDED_MARKER`), so `debug_check_*`
        // returns first. `_get_size_for_typeid` reads the length raw once
        // the address is the live object; `get_possibly_forwarded_type_id`
        // is what substitutes the copy, and that already happened.
        if self.items_have_gc_ptrs && self.item_size > 0 {
            let length = unsafe { *((obj_addr + self.length_offset) as *const usize) };
            let items_start = obj_addr + self.size;
            for i in 0..length {
                let item = items_start + i * self.item_size;
                for &offset in &self.var_gc_ptr_offsets {
                    f((item + offset) as *mut GcRef);
                }
            }
        }
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        TypeRegistry {
            entries: Vec::new(),
            id_of_type: HashMap::new(),
            layout_table: Vec::new(),
            frozen_layout_table: None,
            offset_tables: Vec::new(),
            custom_data: Vec::new(),
            can_add_new_types: true,
            inheritance_ids_assigned: false,
        }
    }

    /// `gctypelayout.py` `TypeLayoutBuilder.get_type_id`.
    ///
    /// Return the stored id on a hit. On a miss, assert `can_add_new_types`
    /// before building the `TypeInfo`, `register` it, store the id, and
    /// return it.
    ///
    /// Upstream derives the shape from `TYPE` itself (`encode_type_shape`), so
    /// one key cannot carry two layouts. Here the caller still supplies the
    /// shape, and a hit skips `info` without comparing it: one key must name
    /// one layout until the shape is derived from the type.
    pub fn get_type_id(&mut self, key: GcTypeKey, info: impl FnOnce() -> TypeInfo) -> u32 {
        if let Some(&type_id) = self.id_of_type.get(&key) {
            return type_id;
        }
        assert!(
            self.can_add_new_types,
            "TypeRegistry::get_type_id: new type after freeze_types \
             (gctypelayout.can_add_new_types == False)"
        );
        let type_id = self.register(info());
        self.id_of_type.insert(key, type_id);
        type_id
    }

    /// Lookup in `TypeLayoutBuilder.id_of_type` without allocating a new id.
    pub fn id_of_type(&self, key: GcTypeKey) -> Option<u32> {
        self.id_of_type.get(&key).copied()
    }

    /// Register a new type and return its type ID.
    ///
    /// `gctypelayout.add_vtable_after_typeinfo` parity (gctypelayout.
    /// py:359-374): pushes the `TypeInfo` and reserves a paired
    /// `TypeEntry` slot in the `type_info_group` array. Both the
    /// `entries` and `layout_table` grow only during this build phase.
    /// `freeze_types` publishes an exact-size boxed table afterwards,
    /// matching `TypeLayoutBuilder.close_table` rather than imposing an
    /// arbitrary registration cap to keep a build-time pointer stable.
    ///
    /// `subclassrange_{min,max}` are intentionally left at 0 here.
    /// `assign_inheritance_ids_now` walks the inheritance tree and
    /// assigns preorder bounds (`normalizecalls.py assign_inheritance_ids`);
    /// `freeze_types` calls that and then refreshes the materialized
    /// `TypeEntry` rows (`gctypelayout.py encode_type_shapes_now`).
    pub fn register(&mut self, mut info: TypeInfo) -> u32 {
        assert!(
            self.can_add_new_types,
            "TypeRegistry::register called after freeze_types \
             (gctypelayout.can_add_new_types == False)"
        );
        if self.inheritance_ids_assigned {
            assert!(
                !info.has_subclass_range,
                "a type with a subclass range cannot be registered after \
                 inheritance ids were assigned"
            );
        }
        let id = self.entries.len();
        // `T_MEMBER_INDEX` is the low 16 bits of `infobits`, so an id past
        // `u16::MAX` cannot round-trip through `encode_type_shape` — it would
        // be masked back to a smaller index and mislabel the row.  The width
        // is `gctypelayout.py`'s, so the bound belongs here rather than in a
        // wider field.
        assert!(
            id <= u16::MAX as usize,
            "TypeRegistry exhausted the 16-bit T_MEMBER_INDEX field: id {id}"
        );
        // `GCBase.trace` walks `offsets_to_gc_pointers` after
        // `_trace_slow_path`. An offset-traced subclass always inherits
        // its parent's fixed offsets. A custom-traced subclass inherits
        // them only when `inherits_parent_offsets` is set
        // (`object_subclass_with_gc_ptrs_and_custom_trace` with a
        // non-empty offset list). `object_subclass_with_custom_trace`
        // leaves the flag clear so a hook that already walks the base
        // fields is not traced twice.
        if (info.custom_trace.is_none() || info.inherits_parent_offsets)
            && let Some(parent_id) = info.parent
        {
            let parent = self
                .entries
                .get(parent_id as usize)
                .expect("object parent must be registered before its subclass");
            assert!(
                parent.custom_trace.is_none(),
                "offset-traced object subclass cannot inherit a custom-traced parent"
            );
            let mut offsets = parent.gc_ptr_offsets.clone();
            for offset in std::mem::take(&mut info.gc_ptr_offsets) {
                if !offsets.contains(&offset) {
                    offsets.push(offset);
                }
            }
            info.gc_ptr_offsets = offsets;
            info.has_gc_ptrs = !info.gc_ptr_offsets.is_empty() || info.items_have_gc_ptrs;
        }
        // `offsets_to_gc_pointers` only records offsets inside the fixed
        // part. An offset at or past `size` is the variable part (or a
        // field list borrowed from another object) and `mark_object`'s
        // `major_fixed_field` loop would dereference it.
        for &offset in &info.gc_ptr_offsets {
            assert!(
                offset < info.size,
                "TypeRegistry::register gc_ptr offset {offset} is past fixed size {} \
                 (gctypelayout.offsets_to_gc_pointers)",
                info.size
            );
        }
        self.entries.push(info);
        // No row address is published until `freeze_types`; fill it there once
        // the pointer-bearing auxiliary tables have stable storage.
        self.layout_table.push(TypeEntry::default());
        id as u32
    }

    /// Attach a sweep-time destructor to an already-registered type.
    /// For `#[pyre_class]` types registered through the generic
    /// `object_subclass_with_gc_ptrs` path (which sets no destructor) that
    /// nonetheless own Rust heap needing drop glue when the collector
    /// reclaims a dead instance.
    ///
    /// Sweep reads `get(type_id).destructor`, so the `entries` slot is the
    /// dispatch source. After `freeze_types` the published `TypeEntry` also
    /// carries `T_HAS_DESTRUCTOR` and `customdata`; rematerialize that row so
    /// a late attach cannot leave the group describing a type without a
    /// destructor.
    pub fn set_destructor(&mut self, type_id: u32, destructor: DestructorFn) {
        assert!(
            self.entries[type_id as usize].old_style_finalizer.is_none(),
            "a type cannot have both a light destructor and an old-style finalizer"
        );
        self.entries[type_id as usize].destructor = Some(destructor);
        if self.frozen_layout_table.is_some() {
            self.rematerialize_type_entry(type_id as usize);
        }
    }

    /// Rebuild one frozen `type_info_group` row from the live `TypeInfo`.
    /// Offsets stay where `freeze_types` published them; only `infobits` and
    /// `customdata` can change when a destructor is attached later.
    fn rematerialize_type_entry(&mut self, index: usize) {
        let info = &self.entries[index];
        let existing = self
            .frozen_layout_table
            .as_ref()
            .expect("rematerialize_type_entry after freeze_types")[index];
        let need_custom = info.custom_trace.is_some()
            || info.destructor.is_some()
            || info.old_style_finalizer.is_some()
            || info.memory_pressure_offset.is_some();
        let customfunc = info
            .custom_trace
            .map(|f| f as usize)
            .or_else(|| info.old_style_finalizer.map(|f| f as usize))
            .or_else(|| info.destructor.map(|f| f as usize))
            .unwrap_or(0);
        let pressure = info.memory_pressure_offset.unwrap_or(0) as isize;
        let item_size = info.item_size;
        let customdata = if need_custom {
            if existing.type_info.customdata != 0 {
                // The Box in `custom_data` owns this row for the registry
                // lifetime; rewrite the fields in place rather than allocating
                // a second CUSTOM_DATA_STRUCT the old address would abandon.
                let data =
                    unsafe { &mut *(existing.type_info.customdata as *mut CustomDataLayout) };
                data.customfunc = customfunc;
                data.memory_pressure_offset = pressure;
                existing.type_info.customdata
            } else {
                let data = Box::new(CustomDataLayout {
                    customfunc,
                    memory_pressure_offset: pressure,
                });
                let address = (&*data) as *const CustomDataLayout as usize;
                self.custom_data.push(data);
                address
            }
        } else {
            0
        };
        let fixed_offsets = existing.type_info.ofstoptrs;
        let var_offsets = if item_size > 0 {
            unsafe { existing.tail.varsize.varofstoptrs }
        } else {
            0
        };
        let info = &self.entries[index];
        let table = self
            .frozen_layout_table
            .as_mut()
            .expect("rematerialize_type_entry after freeze_types");
        table[index] =
            TypeEntry::from_type_info(info, index as u32, customdata, fixed_offsets, var_offsets);
    }

    /// `gctypelayout.encode_type_shapes_now` parity
    /// (gctypelayout.py): freezes the registry so subsequent
    /// `register_type` calls panic and assigns each registered classdef
    /// type its preorder `subclassrange_{min,max}`.
    ///
    /// The bounds come from `rtyper/normalizecalls.py assign_inheritance_ids
    /// assign_inheritance_ids`: each class is sorted lexicographically
    /// by reversed MRO (witness), and its position in that sorted list
    /// becomes the preorder index. The exclusive upper bound is the
    /// position of the matching `[witness + MAX]` peer, which lies
    /// just past the class's last descendant. The resulting
    /// `subclassrange_{min,max}` satisfy
    /// `int_between(cls.min, subcls.min, cls.max)` from
    /// `rclass.py ll_issubclass`.
    pub fn freeze_types(&mut self) {
        if !self.can_add_new_types {
            return;
        }
        self.can_add_new_types = false;
        self.assign_inheritance_ids_now();
        self.offset_tables.reserve(self.entries.len() * 2);
        self.custom_data.reserve(self.entries.len());
        for (i, info) in self.entries.iter().enumerate() {
            let fixed_offsets =
                Self::materialize_offsets(&mut self.offset_tables, &info.gc_ptr_offsets);
            let var_offsets =
                Self::materialize_offsets(&mut self.offset_tables, &info.var_gc_ptr_offsets);
            let customdata = if info.custom_trace.is_some()
                || info.destructor.is_some()
                || info.old_style_finalizer.is_some()
                || info.memory_pressure_offset.is_some()
            {
                // Pyre permits its Rust-side custom tracer and lightweight
                // destructor side channels to coexist.  Upstream has one
                // CUSTOM_FUNC slot; when T_HAS_CUSTOM_TRACE is set, materialize
                // that function and retain destructor dispatch on TypeInfo.
                let customfunc = info
                    .custom_trace
                    .map(|f| f as usize)
                    .or_else(|| info.old_style_finalizer.map(|f| f as usize))
                    .or_else(|| info.destructor.map(|f| f as usize))
                    .unwrap_or(0);
                let data = Box::new(CustomDataLayout {
                    customfunc,
                    memory_pressure_offset: info.memory_pressure_offset.unwrap_or(0) as isize,
                });
                let address = (&*data) as *const CustomDataLayout as usize;
                self.custom_data.push(data);
                address
            } else {
                0
            };
            self.layout_table[i] =
                TypeEntry::from_type_info(info, i as u32, customdata, fixed_offsets, var_offsets);
        }
        self.entries.shrink_to_fit();
        self.frozen_layout_table = Some(std::mem::take(&mut self.layout_table).into_boxed_slice());
    }

    fn materialize_offsets(storage: &mut Vec<Box<[usize]>>, offsets: &[usize]) -> usize {
        let mut words = Vec::with_capacity(offsets.len() + 1);
        words.push(offsets.len());
        words.extend_from_slice(offsets);
        let table = words.into_boxed_slice();
        let address = table.as_ptr() as usize;
        storage.push(table);
        address
    }

    /// Run `normalizecalls.py assign_inheritance_ids` once.
    ///
    /// `freeze_types` (`gctypelayout.py encode_type_shapes_now`) calls this
    /// before it publishes `frozen_layout_table`. A startup caller can assign
    /// the preorder `subclassrange_{min,max}` first and still register later
    /// types that have no subclass range. Idempotent.
    ///
    /// `assign_inheritance_ids` writes min/max only for entries with
    /// `has_subclass_range`. Every constructor leaves the other entries at
    /// `(0, 0)`, and a type without the flag adds no peer, so skipping a
    /// second pass leaves a late entry exactly where a full pass would.
    pub fn assign_inheritance_ids_now(&mut self) {
        if self.inheritance_ids_assigned {
            return;
        }
        self.assign_inheritance_ids();
        self.inheritance_ids_assigned = true;
    }

    /// `rtyper/normalizecalls.py assign_inheritance_ids` /
    /// `TotalOrderSymbolic` parity (normalizecalls.py).
    ///
    /// For each type representing a registered classdef, builds
    /// `witness = reversed(MRO of cdef ids)` and pairs it with a
    /// matching `witness + [MAX]` peer:
    ///
    /// ```python
    /// classdef.minid = TotalOrderSymbolic(witness, lst)
    /// classdef.maxid = TotalOrderSymbolic(witness + [MAX], lst)
    /// MAX = 1E100
    /// ```
    ///
    /// All peers are then sorted lexicographically; the position of
    /// each peer in the sorted list becomes its preorder value
    /// (`compute_fn` in normalizecalls.py). The result
    /// satisfies `int_between(cls.min, subcls.min, cls.max)` from
    /// `rclass.py ll_issubclass`.
    ///
    /// RPython encodes the sentinel as the float `1E100` so that any
    /// real cdef-id (a Python int) sorts before it under tuple
    /// comparison. majit replicates this with the explicit
    /// `WitnessElement::Max` variant — `Cdef(_) < Max` falls out of
    /// the derived `Ord`, structurally matching the RPython peer
    /// list without relying on a magic numeric sentinel.
    fn assign_inheritance_ids(&mut self) {
        let n = self.entries.len();
        // Build reversed-MRO witness for each registered classdef type.
        // Witness for typeid T = [T_root, ..., T_grandparent, T_parent, T]
        // where T_root is the topmost rclass.OBJECT-layout ancestor.
        let mut witness: Vec<Option<Vec<WitnessElement>>> = vec![None; n];
        for (id, witness_slot) in witness.iter_mut().enumerate().take(n) {
            if !self.entries[id].has_subclass_range {
                continue;
            }
            let mut mro: Vec<WitnessElement> = Vec::new();
            let mut cur = Some(id as u32);
            while let Some(c) = cur {
                mro.push(WitnessElement::Cdef(c));
                cur = self.entries[c as usize].parent;
            }
            mro.reverse();
            *witness_slot = Some(mro);
        }

        // Build the peer list: each registered classdef contributes a
        // `(witness, owner)` Min peer and a `(witness + [Max], owner)`
        // Max peer. Sort lexicographically by witness; the peer's
        // position becomes its preorder value (TotalOrderSymbolic
        // .compute_fn, normalizecalls.py).
        #[derive(Clone)]
        struct Peer {
            witness: Vec<WitnessElement>,
            owner: u32,
            is_max: bool,
        }
        let mut peers: Vec<Peer> = Vec::new();
        for (id, witness_slot) in witness.iter().enumerate().take(n) {
            let Some(w) = witness_slot.as_ref() else {
                continue;
            };
            peers.push(Peer {
                witness: w.clone(),
                owner: id as u32,
                is_max: false,
            });
            let mut w_max = w.clone();
            w_max.push(WitnessElement::Max);
            peers.push(Peer {
                witness: w_max,
                owner: id as u32,
                is_max: true,
            });
        }
        peers.sort_by(|a, b| a.witness.cmp(&b.witness));

        // Assign preorder values. minid = index of the Min peer,
        // maxid = index of the Max peer. The exclusive bound matches
        // RPython's int_between (a <= b < c) semantics from
        // rclass.py ll_issubclass.
        for (idx, peer) in peers.iter().enumerate() {
            let owner = peer.owner as usize;
            if peer.is_max {
                self.entries[owner].subclassrange_max = idx as i64;
            } else {
                self.entries[owner].subclassrange_min = idx as i64;
            }
        }
    }

    /// Whether `freeze_types` has published `frozen_layout_table`.
    ///
    /// Compiled guards embed that table's base address
    /// (`llsupport/gc.py get_translated_info_for_typeinfo`).
    /// `assign_inheritance_ids_now` does not publish it.
    pub fn is_frozen(&self) -> bool {
        self.frozen_layout_table.is_some()
    }

    /// Look up type info by ID.
    pub fn get(&self, type_id: u32) -> &TypeInfo {
        &self.entries[type_id as usize]
    }

    /// Collector tid of a varsize type with this `encode_type_shape` layout,
    /// or `None`. Used by `init_array_descr`'s layoutbuilder half so a
    /// `DICTENTRYARRAY` already registered by the host is not registered
    /// again (`TypeLayoutBuilder.get_type_id` returns the existing id).
    pub fn find_varsize_type(
        &self,
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        var_gc_ptr_offsets: &[usize],
    ) -> Option<u32> {
        if item_size == 0 {
            return None;
        }
        self.entries.iter().enumerate().find_map(|(id, info)| {
            (info.item_size == item_size
                && info.size == base_size
                && info.length_offset == length_offset
                && info.var_gc_ptr_offsets == var_gc_ptr_offsets)
                .then_some(id as u32)
        })
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the `type_info_group` array of paired
    /// `(TYPE_INFO, CLASSTYPE)` entries.
    ///
    /// Mirrors `llop.gc_get_type_info_group` (gc.py:585): RPython's
    /// translator emits a contiguous group at translation time whose
    /// members alternate between `TYPE_INFO` and `CLASSTYPE`
    /// (gctypelayout.py `add_vtable_after_typeinfo`). majit's
    /// equivalent is the exact-size boxed table published by
    /// `freeze_types`.  No address is exposed while the build-time Vec can
    /// still grow; after freezing, the boxed slice has the same stable,
    /// immutable shape as RPython's closed `type_info_group` and compiled
    /// guards may embed its base address.
    pub fn type_info_table(&self) -> &[TypeEntry] {
        self.frozen_layout_table
            .as_deref()
            .expect("type_info_table requested before freeze_types")
    }

    /// Translation-time `framework.py write_typeid_list` metadata.
    /// Pyre's type-info group contains one row per registered type after the
    /// dummy member, so line `type_id + 1` is the index written to heap dumps.
    pub fn typeids_text(&self) -> Vec<u8> {
        let mut text = String::from("member0    ?\n");
        for (type_id, info) in self.entries.iter().enumerate() {
            use std::fmt::Write as _;
            let member = type_id + 1;
            if info.item_size == 0 {
                let _ = writeln!(
                    text,
                    "member{member:<4} GcStruct gc_type_{type_id} {{ size={}, gcptr_offsets={:?} }}",
                    info.size, info.gc_ptr_offsets,
                );
            } else {
                let _ = writeln!(
                    text,
                    "member{member:<4} GcArray gc_type_{type_id} {{ base_size={}, item_size={}, gcptr_offsets={:?} }}",
                    info.size, info.item_size, info.gc_ptr_offsets,
                );
            }
        }
        text.into_bytes()
    }

    /// `framework.py:739-747 list_data`: map each group-member index back to
    /// the type-id representation stored in object headers. Member zero is the
    /// dummy; pyre headers use the compact zero-based registry row directly.
    pub fn typeids_list(&self) -> Vec<usize> {
        std::iter::once(0).chain(0..self.entries.len()).collect()
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn materialized_destructor(_obj_addr: usize) {}
    unsafe fn materialized_old_style_finalizer(_obj_addr: usize) {}

    #[test]
    fn test_type_registry() {
        let mut reg = TypeRegistry::new();
        let id0 = reg.register(TypeInfo::simple(16));
        let id1 = reg.register(TypeInfo::simple(32));

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(reg.get(id0).size, 16);
        assert_eq!(reg.get(id1).size, 32);
        assert!(!reg.get(id0).has_gc_ptrs);
    }

    #[test]
    fn get_type_id_same_key_twice_returns_same_id() {
        static KEY: u8 = 0;
        let mut reg = TypeRegistry::new();
        let key = GcTypeKey::of(&KEY);
        let id0 = reg.get_type_id(key, || TypeInfo::simple(16));
        let id1 = reg.get_type_id(key, || TypeInfo::simple(32));
        assert_eq!(id0, id1);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.id_of_type(key), Some(id0));
        assert_eq!(reg.get(id0).size, 16);
    }

    #[test]
    fn get_type_id_distinct_static_keys_get_ids_in_call_order() {
        static KEY_A: u8 = 0;
        static KEY_B: u8 = 1;
        let mut reg = TypeRegistry::new();
        let id_a = reg.get_type_id(GcTypeKey::of(&KEY_A), || TypeInfo::simple(8));
        let id_b = reg.get_type_id(GcTypeKey::of(&KEY_B), || TypeInfo::simple(16));
        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    #[should_panic(expected = "TypeRegistry::get_type_id: new type after freeze_types")]
    fn get_type_id_miss_after_freeze_panics() {
        static KEY: u8 = 0;
        let mut reg = TypeRegistry::new();
        reg.freeze_types();
        let _ = reg.get_type_id(GcTypeKey::of(&KEY), || TypeInfo::simple(8));
    }

    #[test]
    fn test_materialized_type_info_matches_gctypelayout_rows() {
        let word = std::mem::size_of::<usize>();
        assert_eq!(std::mem::size_of::<TypeInfoLayout>(), 4 * word);
        assert_eq!(std::mem::size_of::<VarSizeTypeInfoLayout>(), 4 * word);
        assert_eq!(std::mem::size_of::<TypeEntry>(), 8 * word);

        let mut reg = TypeRegistry::new();
        let fixed = reg.register(
            TypeInfo::with_gc_ptrs(32, vec![0, 24])
                .with_destructor_fn(materialized_destructor)
                .with_memory_pressure_offset(16),
        );
        let var = reg.register(TypeInfo::varsize_with_gc_ptr_offsets(
            16,
            24,
            8,
            vec![0, 16],
            vec![0],
        ));
        reg.freeze_types();

        let fixed_row = &reg.type_info_table()[fixed as usize];
        assert_eq!(fixed_row.type_info.fixedsize, 32);
        assert_ne!(fixed_row.type_info.customdata, 0);
        let custom = unsafe { &*(fixed_row.type_info.customdata as *const CustomDataLayout) };
        assert_eq!(
            custom.customfunc,
            materialized_destructor as *const () as usize
        );
        assert_eq!(custom.memory_pressure_offset, 16);
        let fixed_offsets = fixed_row.type_info.ofstoptrs as *const usize;
        assert_eq!(unsafe { *fixed_offsets }, 2);
        assert_eq!(unsafe { *fixed_offsets.add(1) }, 0);
        assert_eq!(unsafe { *fixed_offsets.add(2) }, 24);

        let var_row = &reg.type_info_table()[var as usize];
        assert_eq!(var_row.type_info.fixedsize, 16);
        let var_layout = unsafe { var_row.tail.varsize };
        assert_eq!(var_layout.varitemsize, 24);
        assert_eq!(var_layout.ofstovar, 16);
        assert_eq!(var_layout.ofstolength, 8);
        let var_offsets = var_layout.varofstoptrs as *const usize;
        assert_eq!(unsafe { *var_offsets }, 2);
        assert_eq!(unsafe { *var_offsets.add(1) }, 0);
        assert_eq!(unsafe { *var_offsets.add(2) }, 16);
    }

    #[test]
    fn old_style_finalizer_sets_infobit_and_custom_function() {
        let mut reg = TypeRegistry::new();
        let type_id = reg.register(TypeInfo::with_old_style_finalizer(
            16,
            materialized_old_style_finalizer,
        ));
        assert_ne!(
            encode_type_shape(reg.get(type_id), type_id) & TypeInfoLayout::T_HAS_OLDSTYLE_FINALIZER,
            0
        );
        assert_eq!(
            encode_type_shape(reg.get(type_id), type_id) & TypeInfoLayout::T_HAS_DESTRUCTOR,
            0
        );

        reg.freeze_types();
        let row = &reg.type_info_table()[type_id as usize];
        let custom = unsafe { &*(row.type_info.customdata as *const CustomDataLayout) };
        assert_eq!(
            custom.customfunc,
            materialized_old_style_finalizer as *const () as usize
        );
    }

    #[test]
    fn set_destructor_after_freeze_rematerializes_type_entry() {
        let mut reg = TypeRegistry::new();
        let type_id = reg.register(TypeInfo::simple(16));
        reg.freeze_types();

        let row = &reg.type_info_table()[type_id as usize];
        assert_eq!(row.type_info.infobits & TypeInfoLayout::T_HAS_DESTRUCTOR, 0);
        assert_eq!(row.type_info.customdata, 0);

        reg.set_destructor(type_id, materialized_destructor);

        assert!(reg.get(type_id).destructor.is_some());
        let row = &reg.type_info_table()[type_id as usize];
        assert_ne!(row.type_info.infobits & TypeInfoLayout::T_HAS_DESTRUCTOR, 0);
        assert_ne!(row.type_info.customdata, 0);
        let custom = unsafe { &*(row.type_info.customdata as *const CustomDataLayout) };
        assert_eq!(
            custom.customfunc,
            materialized_destructor as *const () as usize
        );
    }

    #[test]
    fn test_type_registry_grows_then_publishes_stable_exact_table() {
        // `TypeLayoutBuilder.type_info_group` grows while types are being
        // added; the old fixed reservation of 4096 entries was a pyre-only
        // limit, not an RPython invariant.
        let mut reg = TypeRegistry::new();
        for _ in 0..4097 {
            reg.register(TypeInfo::simple(16));
        }
        reg.freeze_types();

        let first = reg.type_info_table();
        assert_eq!(first.len(), 4097);
        let base = first.as_ptr();
        assert_eq!(reg.type_info_table().as_ptr(), base);
    }

    #[test]
    #[should_panic(expected = "type_info_table requested before freeze_types")]
    fn test_type_registry_does_not_publish_build_time_table() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::simple(16));
        let _ = reg.type_info_table();
    }

    #[test]
    fn test_type_with_gc_ptrs() {
        let mut reg = TypeRegistry::new();
        let id = reg.register(TypeInfo::with_gc_ptrs(24, vec![0, 8, 16]));
        assert!(reg.get(id).has_gc_ptrs);
        assert_eq!(reg.get(id).gc_ptr_offsets.len(), 3);
    }

    #[test]
    fn test_object_subclass_inherits_fixed_gc_ptrs() {
        // rclass.py embeds the complete parent GcStruct as `super`; GC fields
        // from that fixed prefix therefore remain part of every concrete
        // subclass's trace shape.
        let mut reg = TypeRegistry::new();
        let root = reg.register(TypeInfo::object_with_gc_ptrs(16, vec![8]));
        let child = reg.register(TypeInfo::object_subclass_with_gc_ptrs(32, root, vec![24]));
        let grandchild = reg.register(TypeInfo::object_subclass(40, child));

        assert_eq!(reg.get(child).gc_ptr_offsets, vec![8, 24]);
        assert_eq!(reg.get(grandchild).gc_ptr_offsets, vec![8, 24]);
        assert!(reg.get(child).has_gc_ptrs);
        assert!(reg.get(grandchild).has_gc_ptrs);
    }

    #[test]
    fn test_varsize_type() {
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());
        assert_eq!(info.total_instance_size(10), 88); // 8 + 8*10
        assert_eq!(info.total_instance_size(0), 8);
    }

    #[test]
    #[should_panic(expected = "gc_ptr offset 8 is past fixed size 8")]
    fn register_rejects_gc_ptr_offset_past_fixed_size() {
        let mut reg = TypeRegistry::new();
        let _ = reg.register(TypeInfo::with_gc_ptrs(8, vec![8]));
    }

    #[test]
    fn test_for_each_gc_ptr() {
        // Object with two GcRef fields at offsets 0 and 8.
        let info = TypeInfo::with_gc_ptrs(16, vec![0, 8]);

        let mut data = [0u8; 16];
        let obj_addr = data.as_mut_ptr() as usize;

        let mut visited = Vec::new();
        unsafe {
            info.for_each_gc_ptr(obj_addr, |ptr| {
                visited.push(ptr as usize);
            });
        }

        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0], obj_addr);
        assert_eq!(visited[1], obj_addr + 8);
    }

    #[test]
    fn test_encode_type_shape_object_flag() {
        // gctypelayout.py: T_IS_RPYTHON_INSTANCE is set on
        // is_object types and stored in the byte that
        // _setup_guard_is_object pulls out — byte 2 of a
        // little-endian Signed for T_IS_RPYTHON_INSTANCE = 0x100000.
        let info = TypeInfo::object(16);
        let bits = encode_type_shape(&info, 5);
        // Lower 16 bits hold the index.
        assert_eq!(bits & TypeInfoLayout::T_MEMBER_INDEX, 5);
        // T_IS_RPYTHON_INSTANCE bit is set.
        assert!(bits & TypeInfoLayout::T_IS_RPYTHON_INSTANCE != 0);
        // T_KEY_VALUE sanity tag is set.
        assert_eq!(
            bits & TypeInfoLayout::T_KEY_MASK,
            TypeInfoLayout::T_KEY_VALUE
        );
        // Byte 2 of the little-endian word carries 0x10
        // (T_IS_RPYTHON_INSTANCE >> 16 = 0x10).
        let bytes = bits.to_le_bytes();
        assert_eq!(bytes[2] & 0x10, 0x10);
    }

    #[test]
    fn test_encode_type_shape_varsize_with_gcptrs() {
        // gctypelayout.py: varsize encoding sets T_IS_VARSIZE
        // and, when items carry GcRefs, both T_HAS_GCPTR_IN_VARSIZE
        // and T_HAS_GCPTR. A pure GcArray-of-GcPtr also sets
        // T_IS_GCARRAY_OF_GCPTR.
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());
        let bits = encode_type_shape(&info, 0);
        assert!(bits & TypeInfoLayout::T_IS_VARSIZE != 0);
        assert!(bits & TypeInfoLayout::T_HAS_GCPTR_IN_VARSIZE != 0);
        assert!(bits & TypeInfoLayout::T_HAS_GCPTR != 0);
        assert!(bits & TypeInfoLayout::T_IS_GCARRAY_OF_GCPTR != 0);
    }

    #[test]
    fn test_assign_inheritance_ids_singleton_for_leaf() {
        // Pure leaf classes get singleton ranges so
        // ll_issubclass(self, self) holds while subclasses are
        // excluded. Mirrors the int_between(a, b, c) = a <= b < c
        // semantics.
        let mut reg = TypeRegistry::new();
        let a = reg.register(TypeInfo::object(16));
        let b = reg.register(TypeInfo::object(16));
        reg.freeze_types();
        let info_a = reg.get(a);
        let info_b = reg.get(b);
        assert!(info_a.subclassrange_min < info_a.subclassrange_max);
        assert!(info_b.subclassrange_min < info_b.subclassrange_max);
        // No overlap between distinct leaf classes.
        assert!(
            info_a.subclassrange_max <= info_b.subclassrange_min
                || info_b.subclassrange_max <= info_a.subclassrange_min
        );
    }

    #[test]
    fn test_assign_inheritance_ids_parent_contains_child() {
        // For a parent->child chain, the parent's preorder range
        // [min, max) must contain the child's range. This is what
        // makes ll_issubclass(child, parent) true and
        // ll_issubclass(parent, child) false.
        let mut reg = TypeRegistry::new();
        let parent = reg.register(TypeInfo::object(16));
        let child = reg.register(TypeInfo::object_subclass(16, parent));
        reg.freeze_types();
        let p = reg.get(parent);
        let c = reg.get(child);
        // child is inside parent's range.
        assert!(p.subclassrange_min <= c.subclassrange_min);
        assert!(c.subclassrange_min < p.subclassrange_max);
        // parent is NOT inside child's range (unless they're equal).
        assert!(
            p.subclassrange_min < c.subclassrange_min || p.subclassrange_max > c.subclassrange_max
        );
    }

    #[test]
    fn test_private_object_layout_does_not_perturb_subclass_ranges() {
        unsafe fn trace_nothing(_: usize, _: &mut dyn FnMut(*mut majit_ir::GcRef)) {}

        let build = |with_private_layout| {
            let mut reg = TypeRegistry::new();
            let parent = reg.register(TypeInfo::object(16));
            let child = reg.register(TypeInfo::object_subclass(16, parent));
            if with_private_layout {
                let private = reg.register(
                    TypeInfo::with_custom_trace(24, trace_nothing)
                        .object_layout_without_subclass_range(),
                );
                assert!(reg.get(private).is_object);
                assert!(!reg.get(private).has_subclass_range);
            }
            reg.freeze_types();
            (
                (
                    reg.get(parent).subclassrange_min,
                    reg.get(parent).subclassrange_max,
                ),
                (
                    reg.get(child).subclassrange_min,
                    reg.get(child).subclassrange_max,
                ),
            )
        };

        assert_eq!(build(false), build(true));
    }

    #[test]
    fn test_for_each_gc_ptr_varsize() {
        // Variable-size object: base_size=8 (length field at offset 0),
        // items are GcRef (item_size=8).
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());

        // Layout: [length: usize][item0: GcRef][item1: GcRef][item2: GcRef]
        let mut data = [0u8; 32]; // 8 base + 3*8 items
        let obj_addr = data.as_mut_ptr() as usize;

        // Write length = 3.
        unsafe {
            *(obj_addr as *mut usize) = 3;
        }

        let mut visited = Vec::new();
        unsafe {
            info.for_each_gc_ptr(obj_addr, |ptr| {
                visited.push(ptr as usize);
            });
        }

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], obj_addr + 8);
        assert_eq!(visited[1], obj_addr + 16);
        assert_eq!(visited[2], obj_addr + 24);
    }

    #[test]
    fn test_for_each_gc_ptr_varsize_struct_items_uses_varofstoptrs() {
        let word = std::mem::size_of::<GcRef>();
        let info = TypeInfo::varsize_with_gc_ptr_offsets(
            word,
            3 * word,
            0,
            vec![word, 2 * word],
            Vec::new(),
        );

        let mut data = vec![0u8; word + 2 * 3 * word];
        let obj_addr = data.as_mut_ptr() as usize;
        unsafe { *(obj_addr as *mut usize) = 2 };

        let mut visited = Vec::new();
        unsafe { info.for_each_gc_ptr(obj_addr, |ptr| visited.push(ptr as usize)) };

        assert_eq!(
            visited,
            vec![
                obj_addr + 2 * word,
                obj_addr + 3 * word,
                obj_addr + 5 * word,
                obj_addr + 6 * word,
            ]
        );
    }

    fn recorded_ranges(reg: &TypeRegistry, ids: &[u32]) -> Vec<(i64, i64)> {
        ids.iter()
            .map(|&id| {
                let info = reg.get(id);
                (info.subclassrange_min, info.subclassrange_max)
            })
            .collect()
    }

    /// Logical `type_info_group` rows. Offset-table addresses differ between
    /// registries; the length word and the offsets themselves do not.
    fn recorded_table(reg: &TypeRegistry) -> Vec<(usize, usize, Vec<usize>, i64, i64)> {
        reg.type_info_table()
            .iter()
            .map(|row| {
                let offsets_ptr = row.type_info.ofstoptrs as *const usize;
                let header = unsafe { *offsets_ptr };
                let offsets = (1..=header)
                    .map(|i| unsafe { *offsets_ptr.add(i) })
                    .collect();
                let class = unsafe { row.tail.classtype };
                (
                    row.type_info.fixedsize,
                    row.type_info.infobits,
                    offsets,
                    class.subclassrange_min,
                    class.subclassrange_max,
                )
            })
            .collect()
    }

    #[test]
    fn assign_inheritance_ids_now_keeps_ranges_when_a_late_type_is_frozen() {
        let mut reg = TypeRegistry::new();
        let parent = reg.register(TypeInfo::object(16));
        let child = reg.register(TypeInfo::object_subclass(24, parent));
        let other = reg.register(TypeInfo::object(16));
        reg.assign_inheritance_ids_now();
        let ids = [parent, child, other];
        let ranges = recorded_ranges(&reg, &ids);
        assert!(ranges.iter().all(|(min, max)| min < max));

        let late = reg.register(TypeInfo::with_gc_ptrs(24, vec![0, 8]));
        assert!(!reg.is_frozen());
        reg.freeze_types();
        assert!(reg.is_frozen());
        assert_eq!(recorded_ranges(&reg, &ids), ranges);

        let late_info = reg.get(late);
        assert_eq!(
            (late_info.subclassrange_min, late_info.subclassrange_max),
            (0, 0)
        );
        let row = &reg.type_info_table()[late as usize];
        assert_eq!(row.type_info.fixedsize, late_info.size);
        let offsets_ptr = row.type_info.ofstoptrs as *const usize;
        unsafe {
            assert_eq!(*offsets_ptr, late_info.gc_ptr_offsets.len());
            for (i, offset) in late_info.gc_ptr_offsets.iter().enumerate() {
                assert_eq!(*offsets_ptr.add(i + 1), *offset);
            }
        }
    }

    #[test]
    #[should_panic(
        expected = "a type with a subclass range cannot be registered after inheritance ids were assigned"
    )]
    fn register_subclass_range_after_assign_inheritance_ids_panics() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::object(16));
        reg.assign_inheritance_ids_now();
        reg.register(TypeInfo::object_subclass(16, 0));
    }

    #[test]
    fn freeze_after_assign_inheritance_ids_matches_plain_freeze() {
        fn build(
            assign_first: bool,
        ) -> (Vec<(i64, i64)>, Vec<(usize, usize, Vec<usize>, i64, i64)>) {
            let mut reg = TypeRegistry::new();
            let parent = reg.register(TypeInfo::object(16));
            let child = reg.register(TypeInfo::object_subclass(24, parent));
            let other = reg.register(TypeInfo::object(32));
            if assign_first {
                reg.assign_inheritance_ids_now();
                reg.assign_inheritance_ids_now();
            }
            reg.freeze_types();
            let ids = [parent, child, other];
            (recorded_ranges(&reg, &ids), recorded_table(&reg))
        }

        assert_eq!(build(true), build(false));
    }
}
