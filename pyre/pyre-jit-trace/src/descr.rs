//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use parking_lot::Mutex;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{
    ArrayDescr, Descr, DescrRef, FieldDescr, IndexMapExt, JitCodeDescr, SizeDescr, SwitchDescr,
    Type,
};

// TODO: tag bits in the high nibble of the descr
// index discriminate Field/Array/Size descrs. RPython stores all descrs
// in `setup_descrs`'s flat `all_descrs` list (descr.py) and
// recovers the type via `isinstance` on the descr object. Pyre cannot
// downcast `Arc<dyn Descr>` to a specific concrete trait via type id,
// so the index itself encodes the discriminant.
//
// Nothing decodes the tag any more. It once was load-bearing for a
// synthetic `FieldIndexDescr` that reconstructed
// `offset`/`field_size`/`field_type`/`signed` out of the packed bits,
// but that descriptor and its helpers are gone (`majit-ir`'s `descr`
// module records the removal): `VirtualizableFieldState.fields` is keyed
// by `FieldDescr::index_in_parent()` now, matching
// `info.AbstractStructPtrInfo._fields[fielddescr.get_index()]`
// (`info.py`). What the tags still buy is disjoint index ranges,
// so two descr kinds cannot collide on one `HeapCache` key — a flat
// counter would have to preserve that much and nothing else.
/// `symbolic.py WORD` — the target pointer width. Every `Type::Ref`
/// field is one pointer, so its descr width derives from the target
/// instead of spelling a 64-bit literal.
const WORD: usize = core::mem::size_of::<usize>();

const FIELD_DESCR_TAG: u32 = 0x1000_0000;
const ARRAY_DESCR_TAG: u32 = 0x2000_0000;
const SIZE_DESCR_TAG: u32 = 0x3000_0000;
const HEADERLESS_SIZE_OWNER_MARKER: &str = "__majit_headerless_size__";

// Reserved, hand-assigned indices for mutable-cell payload fields. The
// runtime `HeapCache` keys entries by `descr.index()`; `stable_field_index`
// derives that key from `(offset, field_size, field_type, signed)` alone, so
// two distinct structs sharing a field layout collapse onto one `CacheEntry`.
// `IntMutableCell.intvalue` (offset 16 / 8 / Int) lands on the exact slot of
// `W_IntObject.intval` (signed) or `W_ListObject.length` (unsigned) — no
// `signed` value avoids both. The read-only LOAD fold tolerated the collision,
// but the store fold's `SetfieldGc` plus the const-cell `last_const_box`
// heuristic thrash the shared `CacheEntry` against colliding list/int reads.
// A reserved index above the FIELD/ARRAY/SIZE tag ranges gives the cell
// payload a private `CacheEntry`; no code inspects the tag bits of `index()`.
const CELL_DESCR_TAG: u32 = 0x4000_0000;
const INT_MUTABLE_CELL_VALUE_INDEX: u32 = CELL_DESCR_TAG;

// FIELD spans [0x1000_0000, 0x10FF_FFFF], ARRAY 0x2xxx_xxxx,
// SIZE 0x3xxx_xxxx, CELL 0x4000_0000 — 0x5000_0000 is disjoint from all four.
// These reserved indices cannot use `make_quasi_immutable_field_descr`:
// `Terminator.allow_unboxing` and `PlainAttribute.ever_mutated` are both
// `Cell<bool>`, so `stable_field_index(offset, field_size, field_type, signed)`
// could collide whenever their `repr(Rust)` offsets coincide. Because the
// index selects the pointer cast in `quasi_immut_descr`, that collision
// would be type confusion rather than merely a shared `HeapCache` key.
const MAPDICT_DESCR_TAG: u32 = 0x5000_0000;
const TERMINATOR_ALLOW_UNBOXING_INDEX: u32 = MAPDICT_DESCR_TAG;
const PLAIN_ATTRIBUTE_EVER_MUTATED_INDEX: u32 = MAPDICT_DESCR_TAG | 1;
const HOLDER_ATTR_INDEX: u32 = MAPDICT_DESCR_TAG | 2;
const HOLDER_TYP_INDEX: u32 = MAPDICT_DESCR_TAG | 3;
// `AuditHolder.hooks_armed` is the same kind of owner as the map nodes — not a
// `PyObject`, and a `Cell<bool>` whose `stable_field_index` could collide with
// theirs — so it takes the next index in this block rather than a tag of its
// own: the other disjoint tags are already spoken for, `0x6000_0000` by
// `object.typeptr` and `0x7000_0000` by the GC header's tid (`majit-ir`
// `descr.rs`).
const AUDIT_HOLDER_HOOKS_INDEX: u32 = MAPDICT_DESCR_TAG | 4;

// `W_Property.fget` / `.fset` are `PyObjectRef` at the first two offsets past
// the object header — the single most crowded coordinate in the tree, since
// every `W_*` class's first reference field lands there.  A
// `stable_field_index(offset, size, type, signed)` would therefore name a
// layout, not an owner, and the index is what selects the pointer cast in
// `quasi_immut_descr`.  Reserved for
// the same reason as the map-node block above, in a tag of its own because the
// owner here IS a `PyObject` and the reasoning that groups those four does not
// apply.  Disjoint from FIELD (0x10xx_xxxx), ARRAY, SIZE, CELL, MAPDICT,
// `object.typeptr` (0x6000_0000), the native mapdict block, and the GC tid.
const PROPERTY_DESCR_TAG: u32 = 0x5100_0000;
const PROPERTY_FGET_INDEX: u32 = PROPERTY_DESCR_TAG;
const PROPERTY_FSET_INDEX: u32 = PROPERTY_DESCR_TAG | 1;

// `StaticMethod.w_function` and `ClassMethod.w_function` sit at the same
// coordinate as each other and as every other `W_*` class's first reference
// field, and the two wrappers have byte-identical layouts, so
// `stable_field_index` cannot even tell THEM apart — it names a layout, not an
// owner, and the index is what selects the pointer cast in
// `quasi_immut_descr`.  Reserved for
// the same reason as the property block above, in a tag of its own.  Disjoint
// from FIELD (0x10xx_xxxx), ARRAY, SIZE, CELL, MAPDICT, PROPERTY,
// `object.typeptr` (0x6000_0000), the native mapdict block, and the GC tid.
const WRAPPER_DESCR_TAG: u32 = 0x5200_0000;
const STATICMETHOD_W_FUNCTION_INDEX: u32 = WRAPPER_DESCR_TAG;
const CLASSMETHOD_W_FUNCTION_INDEX: u32 = WRAPPER_DESCR_TAG | 1;

// `CellFamily.ever_mutated` is a `Cell<bool>` on an owner that is not a
// `PyObject` — the same shape as the map-node block, so `stable_field_index`
// could collide with `Terminator.allow_unboxing` or
// `PlainAttribute.ever_mutated` and the index is what selects the pointer cast
// in `install_quasiimmut_field` / `register_quasi_immutable_deps`.  Reserved in
// a tag of its own because the owner belongs to a code object rather than to
// mapdict.  Disjoint from FIELD (0x10xx_xxxx), ARRAY, SIZE, CELL, MAPDICT,
// PROPERTY, WRAPPER, `object.typeptr` (0x6000_0000), the native mapdict block,
// and the GC tid.
const CELL_FAMILY_DESCR_TAG: u32 = 0x5300_0000;
const CELL_FAMILY_EVER_MUTATED_INDEX: u32 = CELL_FAMILY_DESCR_TAG;

// `ExecutionContext.w_tracefunc` is a new `?` field.  `quasi_immut_descr`
// and `function_quasi_immut_slot` select the pointer cast from
// `descr.index()` alone, and the EC group's ordinal `2` is not an owner
// identity — any other group's field 2 would take the same arm.  Reserved
// in a tag of its own, disjoint from FIELD / ARRAY / SIZE / CELL / MAPDICT /
// PROPERTY / WRAPPER / CELL_FAMILY / `object.typeptr` / NATIVE_MAPDICT /
// the GC tid.
const EC_DESCR_TAG: u32 = 0x5400_0000;
const EC_W_TRACEFUNC_INDEX: u32 = EC_DESCR_TAG;
const EC_PROFILEFUNC_INDEX: u32 = EC_DESCR_TAG + 1;

// `PyObject.w_class` sits at the same first-Ref coordinate as every other
// `W_*` header field, so `stable_field_index` names a layout, not an owner.
// `quasi_immut_descr` casts from the index alone. Reserved in a tag of its
// own, disjoint from EC / PROPERTY / WRAPPER / MAPDICT / `object.typeptr`.
const OBJECT_W_CLASS_DESCR_TAG: u32 = 0x5500_0000;
const OBJECT_W_CLASS_INDEX: u32 = OBJECT_W_CLASS_DESCR_TAG;

// `GilReadyState.gil_ready` is a process-global `i64`, not a `PyObject`
// field. `quasi_immut_descr` selects the watcher from `descr.index()`
// alone, so the index has to be an owner identity. Next free tag after
// `PyObject.w_class` (`0x5500_0000`); `0x6100_0000` is native mapdict.
const GIL_READY_DESCR_TAG: u32 = 0x5600_0000;
const GIL_READY_INDEX: u32 = GIL_READY_DESCR_TAG;

// `PyCode.w_globals` is `pycode.py` `"w_globals?"`. The index selects the
// pointer cast in `quasi_immut_descr`, and the PyCode descr group's row
// derives its index from the layout (`stable_field_index`) and its parent
// ordinal, so that number is not an owner identity. Reserved in a tag of
// its own. Disjoint from FIELD (0x10xx_xxxx), ARRAY, SIZE, CELL, MAPDICT,
// PROPERTY, WRAPPER, CELL_FAMILY, EC, OBJECT_W_CLASS (`0x5500_0000`),
// GIL_READY (`0x5600_0000`), `object.typeptr` (0x6000_0000), the native
// mapdict block, and the GC tid.
const PYCODE_DESCR_TAG: u32 = 0x5700_0000;
const PYCODE_W_GLOBALS_INDEX: u32 = PYCODE_DESCR_TAG;

// The generated native user layouts append mapdict fields at different base
// sizes. HeapCache keys by descriptor index; give each translated STRUCT field
// the distinct identity provided by descr.py's per-STRUCT cache.
const NATIVE_MAPDICT_DESCR_TAG: u32 = 0x6100_0000;

fn type_bits(tp: Type) -> u32 {
    match tp {
        Type::Int => 0,
        Type::Ref => 1,
        Type::Float => 2,
        Type::Void => 3,
    }
}

fn stable_field_index(offset: usize, field_size: usize, field_type: Type, signed: bool) -> u32 {
    FIELD_DESCR_TAG
        | (((offset as u32) & 0x000f_ffff) << 4)
        | (((field_size as u32) & 0x7) << 1)
        | ((signed as u32) << 3)
        | type_bits(field_type)
}

/// Concrete field descriptor for pyre object fields.
/// RPython FieldDescr: describes a field in a GC/raw struct.
#[derive(Debug)]
pub struct PyreFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
    /// RPython: is_immutable_field(). Immutable fields survive cache invalidation.
    immutable: bool,
    /// RPython: _is_quasi_immutable(). Fields that rarely change but CAN change.
    /// When read during tracing, emits QUASIIMMUT_FIELD + GUARD_NOT_INVALIDATED.
    /// If mutated at runtime, invalidates all compiled loops watching this field.
    quasi_immutable: bool,
    /// RPython descr.py:227 — field name for heaptracker.py:66 filtering.
    name: &'static str,
    /// Whether this field is the owning struct's class word — the declared
    /// answer behind `FieldDescr::is_w_class()`.  Stated per construction
    /// site rather than recovered from `name`, so a consumer never has to
    /// know how this host spells its header.
    is_class_word: bool,
    index_in_parent: usize,
    parent_descr: Option<Weak<dyn Descr>>,
    /// `effectinfo.py compute_bitstrings` ei_index. `u32::MAX` until
    /// the codewriter publishes its `field_index` (`effectinfo.py`)
    /// onto this descr.
    ei_index: AtomicU32,
}

/// Structural key for `ARRAY_DESCR_REGISTRY`. Combination of all fields
/// that PyPy treats as part of `ArrayDescr` identity (`descr.py
/// + lendescr`). Two array descrs sharing this tuple share the same
/// `descr_id`.
///
/// `array_type_id` carries the codewriter lltype-identity proxy
/// (`majit-translate/src/codewriter/call.rs::DescrIndexRegistry::array_index`
/// key) so the runtime registry's identity domain matches PyPy's
/// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py`) keyed
/// on the actual lltype object: two BhDescr::Array entries that
/// disagree only on the Rust type spelling
/// (e.g. `"Vec<Foo>"` vs `"Vec<Bar>"` with both at `type_id == 0`)
/// land on distinct registry slots, preventing the second
/// `set_ei_index` from clobbering the first.
///
/// `None` for legacy descrs minted by pyre-jit-trace internal
/// factories with no source-level array_type_id context; two `None`
/// entries still collide on the remaining structural tuple just as
/// the pre-bridge baseline did.
#[derive(Hash, Eq, PartialEq, Clone)]
struct ArrayDescrKey {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type_bits: u32,
    signed: bool,
    len_offset: Option<usize>,
    array_type_id: Option<String>,
}

static NEXT_ARRAY_DESCR_ID: AtomicU32 = AtomicU32::new(0);

/// Maximum sequential ARRAY descr id. Bits 0-27 of the index are
/// available below `ARRAY_DESCR_TAG`; bit 28 is reserved by
/// `FIELD_DESCR_TAG`.
const ARRAY_DESCR_ID_MAX: u32 = 1 << 28;

static ARRAY_DESCR_REGISTRY: LazyLock<Mutex<indexmap::IndexMap<ArrayDescrKey, DescrRef>>> =
    LazyLock::new(|| Mutex::new(indexmap::IndexMap::new()));

fn alloc_array_descr_id() -> u32 {
    let id = NEXT_ARRAY_DESCR_ID.fetch_add(1, Ordering::Relaxed);
    assert!(
        id < ARRAY_DESCR_ID_MAX,
        "array descr registry exhausted (>2^28 instances) — index() bit 28 belongs to FIELD_DESCR_TAG"
    );
    id
}

/// `descr.py get_type_flag(ARRAY.OF)` — element classification
/// for the runtime array mint.  Preserves the predicates the dropped
/// `PyreArrayDescr` reported: `is_item_signed` (Signed vs Unsigned for
/// ints), `is_array_of_pointers` (Pointer), `is_array_of_floats`
/// (Float).  Non-struct only — struct arrays are minted as
/// `SimpleArrayDescr(FLAG_STRUCT)` in `make_descr_from_bh`.
fn runtime_array_flag(item_type: Type, signed: bool) -> majit_ir::descr::ArrayFlag {
    use majit_ir::descr::ArrayFlag;
    match item_type {
        Type::Ref => ArrayFlag::Pointer,
        Type::Float => ArrayFlag::Float,
        Type::Int if signed => ArrayFlag::Signed,
        Type::Int => ArrayFlag::Unsigned,
        Type::Void => ArrayFlag::Void,
    }
}

fn get_or_create_array_descr(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_offset: Option<usize>,
) -> DescrRef {
    get_or_create_array_descr_with_full_id(
        base_size, item_size, type_id, item_type, signed, len_offset, None,
    )
}

fn get_or_create_array_descr_with_full_id(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_offset: Option<usize>,
    array_type_id: Option<String>,
) -> DescrRef {
    let key = ArrayDescrKey {
        base_size,
        item_size,
        type_id,
        item_type_bits: type_bits(item_type),
        signed,
        len_offset,
        array_type_id,
    };
    let mut cache = ARRAY_DESCR_REGISTRY.lock();
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    // `descr.py get_array_descr(gccache, ARRAY)`: when the
    // caller has an `array_type_id` (the codewriter lltype identity
    // proxy), `gc_cache._cache_array[LLType::Array(path_hash(atid))]`
    // is the authoritative cache slot — consult it FIRST so a prior
    // analyzer-side `gc_cache.get_array_descr` mint
    // (`SimpleArrayDescr`) is reused instead of layered under a fresh
    // runtime `SimpleArrayDescr`.  Matches PyPy `cpu.arraydescrof(ARRAY)`
    // per-ARRAY object identity — both analyzer and pyre runtime
    // consumers share one Arc per `LLType::Array(path_hash(atid))`.
    if let Some(ref atid) = key.array_type_id {
        let gc_key = majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(atid));
        if let Some(existing) = majit_ir::descr::gc_cache()
            .lock()
            ._cache_array
            .get(&gc_key)
            .cloned()
        {
            // Analyzer `arraydescrof_concrete` mints `type_id = 0`.
            // `init_array_descr` leaves `UNSET_GC_TYPE_ID` until
            // `register_unresolved_array_tids`. Runtime
            // `pyobject_gcarray_descr` carries the collector tid; stamp
            // it onto the shared Arc so `gen_initialize_tid` does not
            // fall back to `OBJECT_GC_TYPE_ID` (frame-locals walk as a
            // 16-byte PyObject — `fib_recursive` SIGSEGV).
            if !majit_ir::descr::array_tid_is_unresolved(type_id)
                && type_id != 0
                && let Some(ad) = existing.as_array_descr()
                && (majit_ir::descr::array_tid_is_unresolved(ad.type_id()) || ad.type_id() == 0)
            {
                ad.set_type_id(type_id);
            }
            // Memoise into the local structural cache so subsequent
            // `get_or_create_array_descr_with_full_id` calls with the
            // same structural key hit the local fast path without
            // re-consulting gc_cache.
            cache.insert(key.clone(), existing.clone());
            return existing;
        }
    }
    let descr_id = alloc_array_descr_id();
    // `array_type_id` Some → `LLType::Array(path_hash(atid))` cache
    // slot (analyzer ↔ runtime convergence path).
    // `array_type_id` None but `type_id != 0` → no codewriter
    // lltype-identity carrier but a stable GC-tid is available
    // (`make_array_descr_with_type` path).  Widening that tid to
    // u64 preserves per-tid identity in `BhDescr::Array.type_id`,
    // matching the behaviour producer sites in `eval.rs` /
    // `assembler.rs` / `jitcode.rs` relied on before the
    // `cache_key()` migration — without this fallback, every
    // `PY_OBJECT_ARRAY_GC_TYPE_ID`-class runtime descr collapsed
    // onto slot 0 at the `BhDescr` boundary.
    // `array_type_id` None and `type_id == 0` → no identity carrier
    // at all (legacy `make_array_descr` no-identity path); stay 0.
    let cache_key = match key.array_type_id.as_deref() {
        Some(atid) => majit_ir::descr::path_hash(atid),
        None if type_id != 0 => type_id as u64,
        None => 0,
    };
    // `descr.py` ArrayDescr — a single flag-bearing descriptor.
    // The runtime mint produces the same `SimpleArrayDescr` the
    // analyzer / `make_descr_from_bh` path mints, stamping the
    // content-addressed `ARRAY_DESCR_TAG | descr_id` index so `index()`
    // keeps the value the retired `PyreArrayDescr::index()` returned
    // (`ARRAY_DESCR_REGISTRY` still returns one Arc per structural key,
    // preserving `Arc::as_ptr` identity).
    let index = ARRAY_DESCR_TAG | (descr_id & 0x0FFF_FFFF);
    let mut array_descr = majit_ir::descr::SimpleArrayDescr::with_flag(
        index,
        base_size,
        item_size,
        type_id,
        item_type,
        runtime_array_flag(item_type, signed),
    );
    array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
    array_descr.set_cache_key(cache_key);
    let arc: DescrRef = Arc::new(array_descr);
    cache.insert(key.clone(), arc.clone());
    // Publish the freshly-minted SimpleArrayDescr into
    // `gc_cache._cache_array` keyed on `LLType::Array(path_hash(atid))`
    // so later analyzer-side `gc_cache.get_array_descr` cache-hit
    // returns this exact Arc.  Without an `array_type_id` (legacy
    // `make_array_descr` callers), only the local
    // `ARRAY_DESCR_REGISTRY` carries the descr — gc_cache cannot
    // identify it.
    if let Some(ref atid) = key.array_type_id {
        majit_ir::descr_registry::register_keyed_array(
            majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(atid)),
            arc.clone(),
        );
    }
    arc
}

/// `cpu.arraydescrof(rffi.CArray(TYPE))` for a raw C array of `item_size`
/// bytes per element — the descr shape `ffisupport.py _get_ffi2descr_dict`
/// hands back for one libffi argument slot.
///
/// A `CArray` has no header and no length field, so `base_size` is zero and
/// the index a `GETARRAYITEM_RAW` carries scales straight by `item_size`.
/// The identity string is the one the codewriter stamps on the raw
/// load/store descrs of the same width, so a store recorded through the
/// codewriter and a read recorded here land on one cache slot and the
/// virtual raw buffer sees a single descr per width.
pub fn raw_carray_descr(item_type: Type, item_size: usize, is_item_signed: bool) -> DescrRef {
    let prefix = match item_type {
        Type::Float => "f",
        _ if is_item_signed => "i",
        _ => "u",
    };
    get_or_create_array_descr_with_full_id(
        0,
        item_size,
        0,
        item_type,
        is_item_signed,
        None,
        Some(format!("majit::raw_carray_{prefix}{}", item_size * 8)),
    )
}

impl Descr for PyreFieldDescr {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn index(&self) -> u32 {
        if self.is_class_word {
            // `w_class?` — reserved so `quasi_immut_descr` can tell this
            // header field from every other first-Ref at the same offset.
            OBJECT_W_CLASS_INDEX
        } else {
            stable_field_index(self.offset, self.field_size, self.field_type, self.signed)
        }
    }

    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }

    fn set_ei_index(&self, ei_index: u32) {
        self.ei_index.store(ei_index, Ordering::Relaxed);
    }

    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }

    /// PyPy FieldDescr.is_always_pure(): immutable fields survive cache invalidation.
    fn is_always_pure(&self) -> bool {
        self.immutable
    }

    fn is_quasi_immutable(&self) -> bool {
        self.quasi_immutable
    }
}

impl FieldDescr for PyreFieldDescr {
    fn is_w_class(&self) -> bool {
        self.is_class_word
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
    fn field_name(&self) -> &str {
        self.name
    }
    fn index_in_parent(&self) -> usize {
        self.index_in_parent
    }
    fn get_parent_descr(&self) -> Option<DescrRef> {
        self.parent_descr
            .as_ref()
            .and_then(|parent| parent.upgrade())
    }
}

/// Create a field descriptor for an object field.
pub fn make_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: false,
        quasi_immutable: false,
        name: "",
        is_class_word: false,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Create a field descr with an explicit parent SizeDescr.
///
/// RPython parity: `fielddescr.get_parent_descr()` returns the owning
/// struct's SizeDescr, enabling `info.py init_fields(parent_descr,
/// index)`. Without parent_descr, `descr_index()` falls back to
/// `stable_field_index` (a hash) instead of `index_in_parent` (a small
/// sequential index), causing OOM in `ensure_field_descr_slot`.
///
/// The `index_in_parent` is computed by scanning the parent SizeDescr's
/// `all_fielddescrs` for a matching offset.
pub fn make_field_descr_with_parent(parent: DescrRef, offset: usize) -> DescrRef {
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

pub fn make_field_descr_full(
    _index: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    immutable: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed: false,
        immutable,
        quasi_immutable: false,
        name: "",
        is_class_word: false,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Create a field descriptor for an immutable field (RPython is_immutable_field).
/// Cache entries for immutable fields survive call invalidation.
pub fn make_immutable_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: true,
        quasi_immutable: false,
        name: "",
        is_class_word: false,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Create a field descriptor for a quasi-immutable field.
/// When read during tracing, emits QUASIIMMUT_FIELD + GUARD_NOT_INVALIDATED.
pub fn make_quasi_immutable_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: false,
        quasi_immutable: true,
        name: "",
        is_class_word: false,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Concrete size descriptor for fixed-size object allocations.
#[derive(Debug)]
pub struct PyreSizeDescr {
    obj_size: usize,
    type_id: u32,
    /// Slot key for `_cache_size[LLType::Struct(cache_key)]` — the STRUCT
    /// structural identity built by `path_hash` (the same publish slot).
    /// `type_id` is the dense u32 GC tid for `gc.alloc_gc_typed`, and
    /// `cache_key` corresponds 1:1 to the lltype-object identity of
    /// `descr.py` `get_size_descr` `cache[STRUCT]`. `SizeDescr.cache_key()` returns
    /// this value so the reverse reader `bh_size_spec_from_descr` round-trips
    /// to the same publish slot as `LLType::Struct(cache_key)`. init 0 is the
    /// one-shot fixture fallback (no structural identity).
    cache_key: u64,
    /// descr.get_vtable() parity: ob_type pointer for NewWithVtable.
    /// optimize_new_with_vtable reads this to set VirtualInfo.known_class.
    vtable: usize,
    /// descr.py `self.all_fielddescrs = all_fielddescrs`.
    all_fielddescrs: Vec<Arc<dyn FieldDescr>>,
    /// descr.py `self.gc_fielddescrs = gc_fielddescrs` — precomputed
    /// subset of `all_fielddescrs` via `is_pointer_field()`
    /// (heaptracker.py + :70 filter).
    gc_fielddescrs: Vec<Arc<dyn FieldDescr>>,
}

struct PyreObjectDescrGroup {
    size_descr: Arc<majit_ir::descr::SimpleSizeDescr>,
    /// This group's own fields, in the order its static table declared them.
    ///
    /// Static accessors index by their table position, so they must read this
    /// list. `descr.py get_field_descr` still makes each Arc shared by
    /// `(STRUCT, fieldname)`, while each SizeDescr keeps its own frozen
    /// positional field list.
    field_descrs: Vec<Arc<majit_ir::descr::SimpleFieldDescr>>,
}

trait FieldDescrGroup {
    fn field_descrs(&self) -> &[Arc<majit_ir::descr::SimpleFieldDescr>];
}

impl FieldDescrGroup for PyreObjectDescrGroup {
    fn field_descrs(&self) -> &[Arc<majit_ir::descr::SimpleFieldDescr>] {
        &self.field_descrs
    }
}

impl FieldDescrGroup for majit_ir::descr::SimpleDescrGroup {
    fn field_descrs(&self) -> &[Arc<majit_ir::descr::SimpleFieldDescr>] {
        &self.field_descrs
    }
}

impl<T: FieldDescrGroup> FieldDescrGroup for LazyLock<T> {
    fn field_descrs(&self) -> &[Arc<majit_ir::descr::SimpleFieldDescr>] {
        (**self).field_descrs()
    }
}

/// GC type id for the `rclass.OBJECT` root (`W_Root`, `baseobjspace.py`).
/// The vtable is `W_ROOT_TYPE`: never an app-level type, never stamped
/// into `ob_type`. Every interp class chains its parent to this id so
/// `assign_inheritance_ids` (normalizecalls.py) emits a
/// `subclassrange_{min,max}` covering every descendant.
/// `GUARD_SUBCLASS(obj, &W_ROOT_TYPE)` succeeds for any rclass instance
/// via `int_between(root.min, obj_typeid.min, root.max)`
/// (`rclass.py` `ll_issubclass`). `INSTANCE_TYPE` (`W_ObjectObject`) is
/// a child of this root, at `W_OBJECT_OBJECT_GC_TYPE_ID`.
pub const OBJECT_GC_TYPE_ID: u32 = 0;
// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` live in `pyre-object`
// alongside the `W_IntObject` / `W_FloatObject` structs they describe,
// so `pyre-object`'s host-side allocators can reach them without a
// back-channel. Re-exported here for existing call sites.
pub use pyre_object::floatobject::W_FLOAT_GC_TYPE_ID;
pub use pyre_object::intobject::W_INT_GC_TYPE_ID;
/// GC type id for JitFrame (jitframe.py:49 register_custom_trace_hook).
pub const JITFRAME_GC_TYPE_ID: u32 = 3;
/// GC type id for JitVirtualRef (virtualref.py — JIT_VIRTUAL_REF).
pub const VREF_GC_TYPE_ID: u32 = 4;
/// GC type id for W_BoolObject. `bool` inherits from `int` per
/// `objectobject.py W_BoolObject.typedef`, so this chains to
/// `W_INT_GC_TYPE_ID` as its parent via `TypeInfo::object_subclass`
/// (heaptracker.py setup_cache_gcstruct2vtable — one typeid per
/// distinct STRUCT, not per root layout).
pub const W_BOOL_GC_TYPE_ID: u32 = 5;
/// GC type id for W_IntRangeIterator. Inherits from `object`
/// (`RANGE_ITER_TYPE` in functional.rs).
pub const RANGE_ITER_GC_TYPE_ID: u32 = 6;
/// GC type ids for the two `step == 1` iterator shapes.  They are separate
/// layouts behind one `range_iterator` Python type, so each needs its own id
/// for `GuardClass` to tell them apart and reach `stop` at a fixed offset.
pub use pyre_object::functional::{
    W_RANGE_ITER_ONE_ARG_GC_TYPE_ID, W_RANGE_ITER_STEP_ONE_GC_TYPE_ID,
};
// `W_LIST_GC_TYPE_ID` / `W_TUPLE_GC_TYPE_ID` live in `pyre-object`
// alongside their structs (matching W_INT/W_FLOAT pattern); re-exported
// here for existing call sites.
pub use pyre_object::listobject::W_LIST_GC_TYPE_ID;
/// GC type id for the variable-length backing block of `PyObjectArray`
/// (the list/tuple items storage). Shape matches `rlist.py:84,116`
/// `GcArray(OBJECTPTR)` — a `T_IS_VARSIZE` block with a single-slot
/// `capacity` header (= upstream's GcArray length header,
/// rlist.py:251 `len(l.items)`) followed by inline `PyObjectRef`
/// items. Registered from `pyre_object::ITEMS_BLOCK_TOKEN` with
/// `items_have_gc_ptrs=true` so the GC walks each item slot as a
/// Ref (`gctypelayout.py T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR`);
/// live list length is stored on the enclosing `W_ListObject` wrapper
/// (`PyObjectArray.len`) to match rlist.py `("length", Signed)`.
///
// Array GC type ids live in `pyre-object` alongside the backing storage
// structs/constants they describe (matching W_INT/W_FLOAT/W_LIST/W_TUPLE
// pattern). Re-exported here for existing call sites.
// The two `GcArray` tids are accessors, not constants: pyre declares them from
// its own `gc.register_type` results at GC build time (`pyre-jit/src/eval.rs`).
pub use pyre_object::object_array::{
    PY_OBJECT_ARRAY_GC_TYPE_ID, gc_float_array_gc_type_id, gc_int_array_gc_type_id,
};
pub use pyre_object::tupleobject::W_TUPLE_GC_TYPE_ID;
// GC type ids for `W_SpecialisedTupleObject_{ii,ff,oo}` live in
// `pyre-object` alongside the structs they describe; re-exported here
// for existing call sites. See
// `pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_*_GC_TYPE_ID`.
pub use pyre_object::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_GC_TYPE_ID, SPECIALISED_TUPLE_II_GC_TYPE_ID,
    SPECIALISED_TUPLE_OO_GC_TYPE_ID,
};
// `BUILTIN_CODE_GC_TYPE_ID` lives in `pyre-interpreter::gateway`
// alongside the `BuiltinCode` struct it describes. `FUNCTION_GC_TYPE_ID`
// lives in `pyre-interpreter::function` for the same reason and covers
// `Function`, `BuiltinFunction`, and `FunctionWithFixedCode` (the
// latter two are Rust type aliases of `Function`). Re-exported here
// for the JIT registration site (`pyre-jit/src/eval.rs`).
pub use pyre_interpreter::function::FUNCTION_GC_TYPE_ID;
pub use pyre_interpreter::gateway::BUILTIN_CODE_GC_TYPE_ID;
// `W_CELL_GC_TYPE_ID` lives in `pyre-object::nestedscope` alongside the
// `Cell` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::nestedscope::W_CELL_GC_TYPE_ID;
// `W_METHOD_GC_TYPE_ID` lives in `pyre-object::function` alongside
// the `Method` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::function::W_METHOD_GC_TYPE_ID;
// `W_SLICE_GC_TYPE_ID` lives in `pyre-object::sliceobject` alongside
// the `W_SliceObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::sliceobject::W_SLICE_GC_TYPE_ID;
// `W_SUPER_GC_TYPE_ID` lives in `pyre-object::descriptor` alongside
// the `W_Super` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::descriptor::W_SUPER_GC_TYPE_ID;
// `W_PROPERTY_GC_TYPE_ID` lives in `pyre-object::descriptor`, while
// `W_STATICMETHOD_GC_TYPE_ID` / `W_CLASSMETHOD_GC_TYPE_ID` live in
// `pyre-object::function` alongside their structs. Re-exported for the JIT
// registration site.
pub use pyre_object::descriptor::W_PROPERTY_GC_TYPE_ID;
pub use pyre_object::function::{W_CLASSMETHOD_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID};
// `W_UNION_GC_TYPE_ID` lives in `pyre-object::_pypy_generic_alias` alongside
// the `UnionType` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::_pypy_generic_alias::W_UNION_GC_TYPE_ID;
// `W_SEQ_ITER_GC_TYPE_ID` lives in `pyre-object::iterobject`
// alongside the `W_SeqIterObject` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::iterobject::W_SEQ_ITER_GC_TYPE_ID;
// `W_COUNT_GC_TYPE_ID` / `W_REPEAT_GC_TYPE_ID` live in
// `pyre-object::interp_itertools` alongside the `W_Count` /
// `W_Repeat` structs they describe. Re-exported for the JIT
// registration site.
pub use pyre_object::interp_itertools::{W_COUNT_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID};
// `W_MEMBER_GC_TYPE_ID` lives in `pyre-object::typedef`
// alongside the `W_MemberDescr` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::typedef::W_MEMBER_GC_TYPE_ID;
// `W_BYTES_GC_TYPE_ID` lives in `pyre-object::bytesobject` alongside
// the `W_BytesObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::bytesobject::W_BYTES_GC_TYPE_ID;
// `W_BYTEARRAY_GC_TYPE_ID` lives in `pyre-object::bytearrayobject`
// alongside the `W_BytearrayObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::bytearrayobject::W_BYTEARRAY_GC_TYPE_ID;
// `W_DICT_GC_TYPE_ID` lives in `pyre-object::dictmultiobject` alongside
// the `W_DictObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::dictmultiobject::W_DICT_GC_TYPE_ID;
// `W_MODULE_DICT_GC_TYPE_ID` lives in `pyre-object::dictmultiobject`
// alongside the `W_ModuleDictObject` struct it describes (the PyPy
// `dictmultiobject.py W_ModuleDictObject` port).  Re-exported
// for the JIT registration site.
pub use pyre_object::dictmultiobject::W_MODULE_DICT_GC_TYPE_ID;
// `W_SET_GC_TYPE_ID` lives in `pyre-object::setobject` alongside the
// `W_SetObject` struct it describes (covers both `set` and
// `frozenset` PyTypes — same Rust struct). Re-exported for the JIT
// registration site.
pub use pyre_object::setobject::W_SET_GC_TYPE_ID;
// `W_BASE_EXCEPTION_GC_TYPE_ID` lives in `pyre-object::interp_exceptions`
// alongside the `W_BaseException` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::interp_exceptions::W_BASE_EXCEPTION_GC_TYPE_ID;
// `W_GENERATOR_GC_TYPE_ID` lives in `pyre-object::generator`
// alongside the `GeneratorIterator` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::generator::W_GENERATOR_GC_TYPE_ID;
// `W_TYPE_GC_TYPE_ID` lives in `pyre-object::typeobject` alongside
// the `W_TypeObject` struct it describes. Re-exported for the JIT
// registration site. (`TYPE_TYPE` is in `all_foreign_pytypes()` but
// the foreign-pytype loop's `sizeof(PyObject)` approximation would
// drastically under-count the W_TypeObject payload.)
pub use pyre_object::typeobject::W_TYPE_GC_TYPE_ID;
// `W_UNICODE_GC_TYPE_ID` / `W_LONG_GC_TYPE_ID` / `W_MODULE_GC_TYPE_ID`
// live alongside their structs in
// `pyre-object::{unicodeobject, longobject, module}`. Re-exported
// for the JIT registration site. `W_ObjectObject` is
// `W_OBJECT_OBJECT_GC_TYPE_ID`, a child of the rclass root, with
// vtable `INSTANCE_TYPE`.
pub use pyre_object::longobject::W_LONG_GC_TYPE_ID;
pub use pyre_object::module::W_MODULE_GC_TYPE_ID;
// `W_DICT_PROXY_GC_TYPE_ID` lives in `pyre-object::dictproxyobject`
// alongside the `W_DictProxyObject` struct it describes.  Re-exported
// for the JIT registration site so the typeid stays in the
// pyre-jit-trace exports table next to its sibling Module/PyFrame
// entries.
pub use pyre_object::dictproxyobject::W_DICT_PROXY_GC_TYPE_ID;
pub use pyre_object::unicodeobject::W_UNICODE_GC_TYPE_ID;
// `PYFRAME_GC_TYPE_ID` lives in `pyre-interpreter::pyframe` alongside
// the `PyFrame` struct it describes. Re-exported for the JIT
// registration site (`pyre-jit/src/eval.rs`).
// Registered ahead of any future
// `NewWithVtable(PyFrame)` in trace IR.
pub use pyre_interpreter::pyframe::PYFRAME_GC_TYPE_ID;
// Appended tail registrations for PyFrame-owned auxiliary objects. These live
// with their Rust layouts in pyre-interpreter and are re-exported here beside
// PYFRAME_GC_TYPE_ID for the runtime registration census.
pub use pyre_interpreter::pyframe::FRAME_DEBUG_DATA_GC_TYPE_ID;

fn field_descr_from_group(group: &dyn FieldDescrGroup, index: usize) -> DescrRef {
    group
        .field_descrs()
        .get(index)
        .expect("field descriptor index out of bounds")
        .clone() as DescrRef
}

/// Build a SizeDescr group for a runtime PyObject layout and publish
/// it into `gc_cache._cache_size` under both the simple-name slot
/// AND the crate-stripped def-path slot.  PyPy `cache[STRUCT]`
/// collapses both into a single lltype-object identity; pyre's
/// analyzer currently hashes the simple name (use-site bare
/// identifier — collect_struct_names registers top-level structs by
/// `simple_name`) so that slot is the de-facto convergence point.
/// The def-path slot is published alongside as a forward-compatible
/// alias for the future analyzer use-import resolver (B-5 follow-up):
/// when that lands, analyzer's `owner_root` switches to qualified
/// form and the SAME `Arc<PyreSizeDescr>` is reachable via the
/// qualified hash. `register_keyed_size` keeps one `_cache_size`
/// entry per logical SizeDescr while allowing fuller-layout upgrades.
///
/// `def_path` empty (or equal to `simple_name`) → single publish.
fn build_object_descr_group_with_def_path(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
) -> PyreObjectDescrGroup {
    build_object_descr_group_with_extra_gc_edges(
        obj_size,
        type_id,
        vtable,
        fields,
        simple_name,
        def_path,
        &[],
        &[],
        "",
        false,
    )
}

/// `build_object_descr_group_with_def_path` for a struct the JIT never
/// allocates and whose pointer may not carry a `GcHeader`.  `headerless` is
/// what keeps `StructPtrInfo::make_guards` from certifying a runtime type id
/// off `ref - GcHeader::SIZE`; the `NEW` rewrite shape the flag also selects is
/// out of reach with no `NEW` to rewrite.
fn build_headerless_object_descr_group_with_def_path(
    obj_size: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
) -> PyreObjectDescrGroup {
    build_object_descr_group_with_extra_gc_edges(
        obj_size,
        0,
        0,
        fields,
        simple_name,
        def_path,
        &[],
        &[],
        "",
        true,
    )
}

/// `descr.py` `get_field_descr` display name: `STRUCT._name + '.' + fieldname`.
/// A key that already carries that prefix is the fieldname, so it is stored
/// as-is. The prefix test does not allocate a temporary string.
fn field_display_name(simple_name: &str, field_key: &str) -> String {
    let already_qualified = !simple_name.is_empty()
        && field_key.len() > simple_name.len()
        && field_key.as_bytes()[simple_name.len()] == b'.'
        && field_key.starts_with(simple_name);
    if simple_name.is_empty() || already_qualified {
        field_key.to_string()
    } else {
        let mut name = String::with_capacity(simple_name.len() + 1 + field_key.len());
        name.push_str(simple_name);
        name.push('.');
        name.push_str(field_key);
        name
    }
}

/// Same display name as [`field_display_name`]. A key that already carries
/// `STRUCT._name + '.' + fieldname` is borrowed; a missing prefix is owned
/// on the descr (`descr.py` `get_field_descr` stores that string).
fn borrowed_field_display_name(simple_name: &str, field_key: &'static str) -> Cow<'static, str> {
    let already_qualified = !simple_name.is_empty()
        && field_key.len() > simple_name.len()
        && field_key.as_bytes()[simple_name.len()] == b'.'
        && field_key.starts_with(simple_name);
    if simple_name.is_empty() || already_qualified {
        Cow::Borrowed(field_key)
    } else {
        let mut name = String::with_capacity(simple_name.len() + 1 + field_key.len());
        name.push_str(simple_name);
        name.push('.');
        name.push_str(field_key);
        Cow::Owned(name)
    }
}

#[cfg(test)]
mod field_display_name_tests {
    use super::field_display_name;

    #[test]
    fn a_qualified_key_is_the_published_name() {
        {
            // a_qualified_key_is_not_prefixed_again
            assert_eq!(
                field_display_name("PyFrame", "PyFrame.flags"),
                "PyFrame.flags"
            );
            assert_eq!(field_display_name("PyFrame", "flags"), "PyFrame.flags");
            assert_eq!(field_display_name("", "flags"), "flags");
            assert_eq!(
                field_display_name("Py", "PyFrame.flags"),
                "Py.PyFrame.flags"
            );
        }
        {
            // a_qualified_static_key_is_the_published_name
            let key: &'static str = "QualStaticOwner.slot";
            let group = super::build_object_descr_group_with_def_path(
                24,
                0x5155_A701,
                0x1000,
                &[(key, 16, 8, majit_ir::value::Type::Int, true, false, false)],
                "QualStaticOwner",
                "pyre_jit_trace::descr::QualStaticOwner",
            );
            let field = &group.field_descrs[0];
            assert!(std::ptr::eq(
                majit_ir::descr::FieldDescr::field_name(field.as_ref()),
                key
            ));
            assert_eq!(majit_ir::descr::FieldDescr::field_key(field.as_ref()), key);
        }
    }
}

fn build_object_descr_group_with_field_indices(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
    field_indices: &[u32],
) -> PyreObjectDescrGroup {
    build_object_descr_group_with_extra_gc_edges(
        obj_size,
        type_id,
        vtable,
        fields,
        simple_name,
        def_path,
        &[],
        field_indices,
        "",
        false,
    )
}

/// `build_object_descr_group_with_def_path` plus GC edges that the
/// positional `fields` census does not name. `PyObject.w_class` is the
/// nested header leaf `heaptracker.py all_fielddescrs` puts at
/// `all_fielddescrs[0]`. Other extra edges (`vable_token`) stay
/// gc-only: `rewrite.py clear_gc_fields` walks `gc_fielddescrs`.
fn build_object_descr_group_with_extra_gc_edges(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
    extra_gc_edges: &[Arc<dyn FieldDescr>],
    field_indices: &[u32],
    cache_key_name: &str,
    headerless: bool,
) -> PyreObjectDescrGroup {
    // `cache_key_name` names the `gc_cache._cache_size` STRUCT identity for a
    // group that publishes under neither name registry.  Zero is the
    // no-STRUCT-identity sentinel, not a key: a group keyed there collapses
    // onto whichever no-name group is inserted first, and `resolve_gc_tid`
    // then hands that group's `type_id` to every header write whose descr
    // carries no key — stamping one STRUCT's tid on another STRUCT's block.
    let cache_key = if !cache_key_name.is_empty() {
        majit_ir::descr::path_hash(cache_key_name)
    } else if !def_path.is_empty() {
        majit_ir::descr::path_hash(def_path)
    } else if !simple_name.is_empty() {
        majit_ir::descr::path_hash(simple_name)
    } else {
        0
    };
    // descr.py `get_field_descr` fills `cache[STRUCT][fieldname]` when the
    // name is requested. An absent STRUCT publishes the static field key
    // directly; the owned-spec path below is the second publication.
    if majit_ir::descr::struct_layout_absent(cache_key) {
        let borrowed: Vec<majit_ir::descr::BorrowedField> =
            fields
                .iter()
                .enumerate()
                .map(
                    |(
                        index_in_parent,
                        &(
                            field_key,
                            offset,
                            field_size,
                            field_type,
                            signed,
                            immutable,
                            quasi_immutable,
                        ),
                    )| {
                        majit_ir::descr::BorrowedField {
                            index: field_indices.get(index_in_parent).copied().unwrap_or_else(
                                || stable_field_index(offset, field_size, field_type, signed),
                            ),
                            name: borrowed_field_display_name(simple_name, field_key),
                            field_key,
                            offset,
                            field_size,
                            field_type,
                            flag: runtime_array_flag(field_type, signed),
                            is_immutable: immutable,
                            is_quasi_immutable: quasi_immutable,
                            index_in_parent,
                            is_class_word: Some(offset == pyre_object::pyobject::W_CLASS_OFFSET),
                        }
                    },
                )
                .collect();
        let mut gc_edges: Vec<Arc<dyn majit_ir::descr::FieldDescr>> =
            vec![W_CLASS_FIELD_DESCR.clone()];
        gc_edges.extend(extra_gc_edges.iter().cloned());
        let group = majit_ir::descr::publish_borrowed_struct_layout(
            SIZE_DESCR_TAG | (obj_size as u32 & 0x0FFF_FFFF),
            obj_size,
            type_id,
            cache_key,
            vtable,
            true,
            headerless,
            &gc_edges,
            borrowed,
        );
        let field_descrs = group.field_descrs;
        let size_descr = group.size_descr;
        if !simple_name.is_empty() {
            let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(simple_name));
            majit_ir::descr_registry::register_keyed_size(
                key,
                size_descr.clone() as majit_ir::DescrRef,
            );
        }
        if !def_path.is_empty() && def_path != simple_name {
            let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(def_path));
            majit_ir::descr_registry::register_keyed_size(
                key,
                size_descr.clone() as majit_ir::DescrRef,
            );
        }
        return PyreObjectDescrGroup {
            size_descr,
            field_descrs,
        };
    }
    let specs: Vec<majit_ir::descr::SimpleFieldDescrSpec> = fields
        .iter()
        .enumerate()
        .map(
            |(
                index_in_parent,
                &(field_key, offset, field_size, field_type, signed, immutable, quasi_immutable),
            )| majit_ir::descr::SimpleFieldDescrSpec {
                index: field_indices
                    .get(index_in_parent)
                    .copied()
                    .unwrap_or_else(|| stable_field_index(offset, field_size, field_type, signed)),
                field_key: field_key.to_string(),
                // `descr.py` `get_field_descr` stores one FieldDescr per
                // `(STRUCT, fieldname)`. The display name is
                // `STRUCT._name + '.' + fieldname`. A key that already
                // begins with that prefix is the fieldname (`PyFrame.flags`
                // versus `PyFrame.PyFrame.flags`).
                name: field_display_name(simple_name, field_key),
                offset,
                field_size,
                field_type,
                is_immutable: immutable,
                is_quasi_immutable: quasi_immutable,
                flag: runtime_array_flag(field_type, signed),
                virtualizable: false,
                // pyre's layout invariant, not a name rule: the inherited
                // `PyObject` header sits at `W_CLASS_OFFSET` in every object
                // group. A name rule cannot express this — `Method` has a
                // *payload* field called `w_class` (the class the bound
                // function was found on) whose qualified name
                // `"Method.w_class"` matches the header spelling exactly, and
                // it sits ahead of the real header in the field list.
                is_class_word: Some(offset == pyre_object::pyobject::W_CLASS_OFFSET),
                index_in_parent,
            },
        )
        .collect();
    debug_assert!(
        specs
            .iter()
            .all(|s| s.field_type != Type::Ref || s.field_size == WORD),
        "a Ref field is one pointer; size it WORD, not a literal"
    );
    let mut gc_edges: Vec<Arc<dyn FieldDescr>> = vec![W_CLASS_FIELD_DESCR.clone()];
    gc_edges.extend(extra_gc_edges.iter().cloned());
    let group = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
        SIZE_DESCR_TAG | (obj_size as u32 & 0x0FFF_FFFF),
        obj_size,
        type_id,
        cache_key,
        vtable,
        true,
        headerless,
        &specs,
        &gc_edges,
    );
    let field_descrs = group.field_descrs;
    let size_descr = group.size_descr;
    // The class word is the positional prefix. Any other extra edge stays
    // a gc-only pointer.
    // Dual-publish: register under BOTH the simple-name slot AND
    // (when supplied) the crate-stripped def-path slot.
    //
    // PyPy `cache[STRUCT]` collapses both namespaces into a single
    // lltype-object identity; pyre's analyzer currently hashes the
    // use-site bare identifier (collect_struct_names registers
    // top-level structs at simple-name) so the simple-name slot is
    // the primary cache hit point.  The def-path slot is published
    // alongside as a forward-compatible alias for the future
    // analyzer use-import resolver (B-5 follow-up): when that lands,
    // analyzer's `owner_root` switches to qualified form and the
    // SAME `Arc<PyreSizeDescr>` is reachable via the qualified
    // hash. `register_keyed_size` keeps one `_cache_size` entry
    // per logical SizeDescr while allowing fuller-layout upgrades.
    if !simple_name.is_empty() {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(simple_name));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    if !def_path.is_empty() && def_path != simple_name {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(def_path));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    PyreObjectDescrGroup {
        size_descr,
        field_descrs,
    }
}

/// Build + publish a SizeDescr group for a **bare** low-level `GcStruct` — a
/// headered, movable GC object that is NOT a `PyObject` (no `ob_header` /
/// `w_class`), unlike [`build_object_descr_group_with_def_path`]. Used by
/// `New{owner}` for synthetic rtyper structs such as the rbuilder
/// `StringBuilder`. No `W_CLASS` GC edge is added and no field is flagged the
/// class word; `is_gc_managed = true` + `headerless = false` keep it forwardable
/// under a tid the caller assigned. Any `Type::Ref` field enters `gc_fielddescrs`
/// (via `with_all_fielddescrs`' `is_pointer_field` filter) as a traced edge; a
/// pure-scalar struct therefore yields an empty `gc_fielddescrs` (a GC leaf).
/// Published under `LLType::Struct(path_hash(simple_name))` so the analyzer's
/// `New{simple_name}` resolves it.
fn build_bare_gcstruct_descr_group(
    obj_size: usize,
    type_id: u32,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
) -> PyreObjectDescrGroup {
    let cache_key = majit_ir::descr::path_hash(simple_name);
    let specs: Vec<majit_ir::descr::SimpleFieldDescrSpec> = fields
        .iter()
        .enumerate()
        .map(
            |(
                index_in_parent,
                &(field_key, offset, field_size, field_type, signed, immutable, quasi_immutable),
            )| majit_ir::descr::SimpleFieldDescrSpec {
                index: stable_field_index(offset, field_size, field_type, signed),
                field_key: field_key.to_string(),
                name: field_display_name(simple_name, field_key),
                offset,
                field_size,
                field_type,
                is_immutable: immutable,
                is_quasi_immutable: quasi_immutable,
                flag: runtime_array_flag(field_type, signed),
                virtualizable: false,
                // A bare GcStruct has no class word.
                is_class_word: Some(false),
                index_in_parent,
            },
        )
        .collect();
    debug_assert!(
        specs
            .iter()
            .all(|s| s.field_type != Type::Ref || s.field_size == WORD),
        "a Ref field is one pointer; size it WORD, not a literal"
    );
    let group = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
        SIZE_DESCR_TAG | (obj_size as u32 & 0x0FFF_FFFF),
        obj_size,
        type_id,
        cache_key,
        0,     // vtable: a bare GcStruct is not an object
        true,  // is_gc_managed: headered + forwardable
        false, // headerless: carries a GcHeader tid word
        &specs,
        // No edge beyond the positional fields: `with_all_fielddescrs` already
        // puts every `Type::Ref` spec (e.g. `extra_pieces`) into gc_fielddescrs.
        &[],
    );
    let field_descrs = group.field_descrs;
    let size_descr = group.size_descr;
    if !simple_name.is_empty() {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(simple_name));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    PyreObjectDescrGroup {
        size_descr,
        field_descrs,
    }
}

/// Upstream `rpython/rtyper/lltypesystem/rbuilder.py` STRINGBUILDER:
///   `GcStruct("stringbuilder", ("current_buf", STRPTR), ("current_pos", Signed),
///     ("current_end", Signed), ("total_size", Signed), ("extra_pieces", STRINGPIECEPTR))`
/// A bare GcStruct (not a PyObject). Both `current_buf: Ptr(STR)` and
/// `extra_pieces: Ptr(STRINGPIECE)` are ordinary traced references, matching the
/// upstream lltype fields. The `Ref` flags put both into `gc_fielddescrs`; the
/// runtime tid carries the same offsets. The
/// body size and field offsets are the `pyre_object::rbuilder` constants that
/// `pyre-jit::eval` pins to `StringBuilderBox` with `offset_of!` const asserts.
static STRINGBUILDER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::rbuilder as rb;
    build_bare_gcstruct_descr_group(
        rb::STRINGBUILDER_SIZE,
        rb::stringbuilder_gc_type_id(),
        &[
            (
                "current_buf",
                rb::STRINGBUILDER_CURRENT_BUF_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "current_pos",
                rb::STRINGBUILDER_CURRENT_POS_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "current_end",
                rb::STRINGBUILDER_CURRENT_END_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "total_size",
                rb::STRINGBUILDER_TOTAL_SIZE_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "extra_pieces",
                rb::STRINGBUILDER_EXTRA_PIECES_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "stringbuilder",
    )
});

/// Force + return the `StringBuilder` bare-GcStruct size descriptor, keyed at
/// `path_hash("stringbuilder")` so `New{"stringbuilder"}` resolves it. Forced by
/// `pyre-jit::eval build_gc` right after the box tid is published, so it lands in
/// gc_cache RESOLVED before the unresolved-struct walker.
pub fn stringbuilder_size_descr() -> DescrRef {
    STRINGBUILDER_DESCR_GROUP.size_descr.clone()
}

/// Upstream `rpython/rtyper/lltypesystem/rbuilder.py` STRINGPIECE:
///   `GcStruct("stringpiece", ("buf", STRPTR), ("prev_piece", Ptr(STRINGPIECE)))`
/// A bare GcStruct chain node the builder's grow path allocates
/// (`malloc(STRINGPIECE)` = `New{"stringpiece"}`) when `current_buf` fills.
/// `buf: Ptr(STR)` and `prev_piece: Ptr(STRINGPIECE)` are both traced references
/// (`gc_fielddescrs = [0, 8]`, mirrored by the runtime tid). 16-byte body.
static STRINGPIECE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::rbuilder as rb;
    build_bare_gcstruct_descr_group(
        rb::STRINGPIECE_SIZE,
        rb::stringpiece_gc_type_id(),
        &[
            (
                "buf",
                rb::STRINGPIECE_BUF_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "prev_piece",
                rb::STRINGPIECE_PREV_PIECE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "stringpiece",
    )
});

/// Force + return the `StringPiece` bare-GcStruct size descriptor, keyed at
/// `path_hash("stringpiece")` so `New{"stringpiece"}` (the grow path's
/// `malloc(STRINGPIECE)`) resolves it. Forced by `pyre-jit::eval build_gc` right
/// after the node tid is published, before the unresolved-struct walker.
pub fn stringpiece_size_descr() -> DescrRef {
    STRINGPIECE_DESCR_GROUP.size_descr.clone()
}

/// rlist.py LIST header. Live exception `args_w` is a fixed `GcArray`;
/// the header stays registered so later type ids do not move.
static RLIST_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::interp_exceptions as exc;
    build_bare_gcstruct_descr_group(
        exc::RLIST_SIZE,
        exc::rlist_gc_type_id(),
        &[
            (
                "length",
                exc::RLIST_LENGTH_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "items",
                exc::RLIST_ITEMS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "rlist.LIST",
    )
});

pub fn rlist_size_descr() -> DescrRef {
    RLIST_DESCR_GROUP.size_descr.clone()
}

pub fn rlist_length_descr() -> DescrRef {
    field_descr_from_group(&RLIST_DESCR_GROUP, 0)
}

pub fn rlist_items_descr() -> DescrRef {
    field_descr_from_group(&RLIST_DESCR_GROUP, 1)
}

static W_INT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_IntObject>(),
        W_INT_GC_TYPE_ID,
        &INT_TYPE as *const _ as usize,
        &[("intval", INT_INTVAL_OFFSET, 8, Type::Int, true, true, false)],
        "W_IntObject",
        "intobject::W_IntObject",
    )
});

static W_FLOAT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_FloatObject>(),
        W_FLOAT_GC_TYPE_ID,
        &FLOAT_TYPE as *const _ as usize,
        &[(
            "floatval",
            FLOAT_FLOATVAL_OFFSET,
            8,
            Type::Float,
            false,
            true,
            false,
        )],
        "W_FloatObject",
        "floatobject::W_FloatObject",
    )
});

static W_COMPLEX_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_ComplexObject>(),
        W_COMPLEX_GC_TYPE_ID,
        &COMPLEX_TYPE as *const _ as usize,
        &[
            (
                "real",
                COMPLEX_REAL_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "imag",
                COMPLEX_IMAG_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
        ],
        "W_ComplexObject",
        "complexobject::W_ComplexObject",
    )
});

static W_LONG_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::longobject::W_LongObject>(),
        pyre_object::longobject::W_LONG_GC_TYPE_ID,
        &pyre_object::pyobject::LONG_TYPE as *const _ as usize,
        &[(
            "value",
            pyre_object::longobject::LONG_VALUE_OFFSET,
            WORD,
            // The `value` slot is a gc-pointer to the BigInt payload, so it
            // enters `gc_fielddescrs` (the boxing SetfieldGc emits the write
            // barrier). Immutable: a long's payload is set once at creation.
            Type::Ref,
            false,
            true,
            false,
        )],
        "W_LongObject",
        "longobject::W_LongObject",
    )
});

static W_BOOL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
        W_BOOL_GC_TYPE_ID,
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        &[(
            "intval",
            BOOL_INTVAL_OFFSET,
            8,
            Type::Int,
            false,
            true,
            false,
        )],
        "W_BoolObject",
        "boolobject::W_BoolObject",
    )
});

// RPython `descr.py get_field_descr` returns a FieldDescr owned by
// the cached parent SizeDescr.  Keep the Unicode fields in the same
// object-descriptor group as every other runtime PyObject layout: in
// particular, `str_len_descr()` must not mint a detached FieldDescr whose
// weak `parent_descr` immediately upgrades to `None`.  The optimizer's
// `protect_speculative_field` asks that parent for the expected type before a
// pure length read, exactly as `llmodel.py:protect_speculative_field` does.
static W_UNICODE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::unicodeobject::W_UNICODE_OBJECT_SIZE,
        W_UNICODE_GC_TYPE_ID,
        &pyre_object::pyobject::STR_TYPE as *const _ as usize,
        &[
            (
                "value",
                pyre_object::unicodeobject::UNICODE_VALUE_OFFSET,
                std::mem::size_of::<*mut pyre_object::unicodeobject::UnicodeValueStorage>(),
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "byte_len",
                pyre_object::unicodeobject::UNICODE_BYTE_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "len",
                pyre_object::unicodeobject::UNICODE_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "index_storage",
                pyre_object::unicodeobject::UNICODE_INDEX_STORAGE_OFFSET,
                std::mem::size_of::<*mut pyre_object::rutf8::Utf8IndexStorage>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(pyre_object::unicodeobject::W_UnicodeObject, hash),
                std::mem::size_of::<i64>(),
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_UnicodeObject",
        "unicodeobject::W_UnicodeObject",
    )
});

/// `W_BytesObject`, laid out the same way as [`W_UNICODE_DESCR_GROUP`]: a
/// pointer to the heap byte buffer plus a precomputed length.  `len` is
/// immutable because bytes is, so a length read hoists out of a loop the way
/// `bytesobject.py`'s `strlen(self._value)` does once the receiver's class is
/// pinned.
///
/// The group exists so `bytes_len_descr()` mints its FieldDescr with a live
/// `parent_descr`, which `protect_speculative_field` asks for the expected
/// type before a pure length read.
static W_BYTES_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::bytesobject::W_BYTES_OBJECT_SIZE,
        W_BYTES_GC_TYPE_ID,
        &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
        &[
            (
                "data",
                pyre_object::bytesobject::BYTES_DATA_OFFSET,
                std::mem::size_of::<*const pyre_object::bytesobject::BytesBlock>(),
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "len",
                pyre_object::bytesobject::BYTES_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "ctypes_keepalive_refs",
                pyre_object::bytesobject::BYTES_CTYPES_KEEPALIVE_REFS_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "W_BytesObject",
        "bytesobject::W_BytesObject",
    )
});

/// `W_BytearrayObject`, the mutable sibling of [`W_BYTES_DESCR_GROUP`].
///
/// `length` is the field `bytearrayobject.py`'s `_len` reaches through
/// `self._data`, where the storage is an RPython list and the read is
/// `rlist.py`'s `("length", Signed)`.  Unlike the bytes `len` it is
/// MUTABLE: append / insert / remove / pop / clear all move it, so a length
/// read must not hoist out of a loop that can mutate the receiver.  Same
/// reason `W_ListObject.length` is mutable.
static W_BYTEARRAY_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::bytearrayobject::W_BYTEARRAY_OBJECT_SIZE,
        W_BYTEARRAY_GC_TYPE_ID,
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
        &[
            (
                "data",
                pyre_object::bytearrayobject::BYTEARRAY_DATA_OFFSET,
                std::mem::size_of::<*mut Vec<u8>>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "length",
                pyre_object::bytearrayobject::BYTEARRAY_LENGTH_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "alloc",
                pyre_object::bytearrayobject::BYTEARRAY_ALLOC_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "logical_offset",
                pyre_object::bytearrayobject::BYTEARRAY_LOGICAL_OFFSET_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "exports",
                pyre_object::bytearrayobject::BYTEARRAY_EXPORTS_OFFSET,
                std::mem::size_of::<i64>(),
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_BytearrayObject",
        "bytearrayobject::W_BytearrayObject",
    )
});

/// `W_SetObject`, the body `set` and `frozenset` share.
///
/// `setobject.py W_BaseSetObject.length` is `self.strategy.length(self)`; pyre
/// keeps the count in a field of the body instead, so the answer is one read
/// once the receiver's class is pinned.  `len` is declared MUTABLE even though
/// a published `frozenset` never moves it: the field is the same slot on both
/// types and every mutator that moves the count -- `w_set_add`,
/// `w_set_discard`, `w_set_clear`, `w_set_popitem` and the bulk updates --
/// stores through `set_len_relaxed`, so a read must not hoist out of a loop
/// that can mutate the receiver.  The slot is an `AtomicUsize` written
/// `Relaxed`; the descr declares it `Type::Int` at `offset_of!`, the same
/// declaration `W_ListObject.length` and `W_BytearrayObject.length` carry
/// over their own atomic slots.
///
/// `content_gen` is the same mutable `AtomicUsize` shape. The integer `add`
/// hit guard re-reads it every iteration; an immutable or pure descr would
/// fold the load away (`opimpl_getfield_gc_i` const-folds only a constant
/// receiver through an always-pure descr). `set_id` is immutable: it is
/// written once in `alloc_set_object` and a nursery move does not change it.
///
/// One STRUCT gets one def-path-keyed group (see [`DECLARED_GROUPS`]), so the
/// vtable here is `SET_TYPE` and a `frozenset` receiver reads its length
/// through the same field descr.  The parent vtable is consulted in exactly
/// one place — `protect_speculative_field`, off `optimizeopt`'s
/// `GETFIELD_GC_PURE_*` branch — and a mutable field never reaches it, which
/// is the same declaration `set` requires on its own.  Registering a second
/// group under this def path would not give `frozenset` its own vtable
/// anyway: `register_keyed_size` arbitrates by field count with both vtables
/// non-zero, so an equal-length twin loses the slot, and `_cache_field`
/// re-parenting would cross-wire the two.
static W_SET_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::setobject::W_SET_OBJECT_SIZE,
        W_SET_GC_TYPE_ID,
        &pyre_object::setobject::SET_TYPE as *const _ as usize,
        &[
            (
                "sstorage",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, sstorage),
                std::mem::size_of::<*mut u8>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            // `W_BaseSetObject.strategy`. Keyed by the field name and typed as
            // the analyzer types `&'static SetStrategyRef` (an instance), so the
            // residual calls whose EffectInfo writes it resolve their member.
            (
                "strategy",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, strategy),
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "len",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, len),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, hash),
                std::mem::size_of::<i64>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "set_id",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, set_id),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "content_gen",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, content_gen),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "lifeline",
                std::mem::offset_of!(pyre_object::setobject::W_SetObject, lifeline),
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_SetObject",
        "setobject::W_SetObject",
    )
});

static RANGE_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::functional::W_IntRangeIterator>(),
        RANGE_ITER_GC_TYPE_ID,
        &pyre_object::functional::RANGE_ITER_TYPE as *const _ as usize,
        &[
            (
                "current",
                RANGE_ITER_CURRENT_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "remaining",
                RANGE_ITER_REMAINING_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "step",
                RANGE_ITER_STEP_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_IntRangeIterator",
        "functional::W_IntRangeIterator",
    )
});

/// `functional.py W_IntRangeStepOneIterator` — `_immutable_fields_ =
/// ['stop']`, and `start` never moves either, so both carry the immutable
/// column and a compiled loop keeps the bound in a register.
static RANGE_ITER_STEP_ONE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::functional::W_IntRangeStepOneIterator>(),
        W_RANGE_ITER_STEP_ONE_GC_TYPE_ID,
        &pyre_object::functional::RANGE_ITER_STEP_ONE_TYPE as *const _ as usize,
        &[
            (
                "current",
                RANGE_ITER_STEP_ONE_CURRENT_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "stop",
                RANGE_ITER_STEP_ONE_STOP_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "start",
                RANGE_ITER_STEP_ONE_START_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_IntRangeStepOneIterator",
        "functional::W_IntRangeStepOneIterator",
    )
});

/// `functional.py W_IntRangeOneArgIterator` — the `range(stop)` shape.
static CALLABLE_ITERATOR_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::operation::_CallableIterator>(),
        <pyre_object::operation::_CallableIterator as pyre_object::lltype::GcType>::type_id(),
        &pyre_object::operation::CALLABLE_ITERATOR_TYPE as *const _ as usize,
        &[
            (
                "callable",
                std::mem::offset_of!(pyre_object::operation::_CallableIterator, callable),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "sentinel",
                std::mem::offset_of!(pyre_object::operation::_CallableIterator, sentinel),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "_CallableIterator",
        "operation::_CallableIterator",
    )
});

static RANGE_ITER_ONE_ARG_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::functional::W_IntRangeOneArgIterator>(),
        W_RANGE_ITER_ONE_ARG_GC_TYPE_ID,
        &pyre_object::functional::RANGE_ITER_ONE_ARG_TYPE as *const _ as usize,
        &[
            (
                "current",
                RANGE_ITER_ONE_ARG_CURRENT_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "stop",
                RANGE_ITER_ONE_ARG_STOP_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
        ],
        "W_IntRangeOneArgIterator",
        "functional::W_IntRangeOneArgIterator",
    )
});

static SEQ_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::iterobject::W_SeqIterObject>(),
        W_SEQ_ITER_GC_TYPE_ID,
        &pyre_object::iterobject::SEQ_ITER_TYPE as *const _ as usize,
        &[
            (
                "seq",
                pyre_object::iterobject::SEQ_ITER_SEQ_OFFSET,
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "index",
                pyre_object::iterobject::SEQ_ITER_INDEX_OFFSET,
                std::mem::size_of::<isize>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "length",
                pyre_object::iterobject::SEQ_ITER_LENGTH_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "empty_kind",
                pyre_object::iterobject::SEQ_ITER_EMPTY_KIND_OFFSET,
                1,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "W_SeqIterObject",
        "iterobject::W_SeqIterObject",
    )
});

/// PyPy `iterobject.py W_TupleIterObject` fields.
static TUPLE_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::iterobject::W_TupleIterObject>(),
        <pyre_object::iterobject::W_TupleIterObject as pyre_object::lltype::GcType>::type_id(),
        &pyre_object::iterobject::TUPLE_ITER_TYPE as *const _ as usize,
        &[
            (
                "seq",
                std::mem::offset_of!(pyre_object::iterobject::W_TupleIterObject, seq),
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "index",
                std::mem::offset_of!(pyre_object::iterobject::W_TupleIterObject, index),
                std::mem::size_of::<isize>(),
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_TupleIterObject",
        "iterobject::W_TupleIterObject",
    )
});

/// PyPy `iterobject.py W_FastListIterObject` fields.
static LIST_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::iterobject::W_ListIterObject>(),
        <pyre_object::iterobject::W_ListIterObject as pyre_object::lltype::GcType>::type_id(),
        &pyre_object::iterobject::LIST_ITER_TYPE as *const _ as usize,
        &[
            (
                "seq",
                std::mem::offset_of!(pyre_object::iterobject::W_ListIterObject, seq),
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "index",
                std::mem::offset_of!(pyre_object::iterobject::W_ListIterObject, index),
                std::mem::size_of::<isize>(),
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_ListIterObject",
        "iterobject::W_ListIterObject",
    )
});

/// `FrameLocalsProxy.w_frame` — the live view the 3.14 proxy holds.
static FRAME_LOCALS_PROXY_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_interpreter::pyframe::frame_locals_proxy::FrameLocalsProxy>(),
        <pyre_interpreter::pyframe::frame_locals_proxy::FrameLocalsProxy as pyre_object::lltype::GcType>::type_id(),
        <pyre_interpreter::pyframe::frame_locals_proxy::FrameLocalsProxy as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE
            as *const _ as usize,
        &[(
            "w_frame",
            pyre_interpreter::pyframe::frame_locals_proxy::W_FRAME_OFFSET,
            std::mem::size_of::<pyre_object::PyObjectRef>(),
            Type::Ref,
            false,
            false,
            false,
        )],
        "FrameLocalsProxy",
        "pyframe::frame_locals_proxy::FrameLocalsProxy",
    )
});

/// PyPy `functional.py W_Zip` fields.
static W_ZIP_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::functional::W_Zip>(),
        <pyre_object::functional::W_Zip as pyre_object::lltype::GcType>::type_id(),
        &pyre_object::functional::ZIP_TYPE as *const _ as usize,
        &[
            (
                "w_iterators",
                std::mem::offset_of!(pyre_object::functional::W_Zip, w_iterators),
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "strict",
                std::mem::offset_of!(pyre_object::functional::W_Zip, strict),
                std::mem::size_of::<bool>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "iteration_progress",
                std::mem::offset_of!(pyre_object::functional::W_Zip, iteration_progress),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "W_Zip",
        "functional::W_Zip",
    )
});

/// `stop` carries no accessor of its own — the GET_ITER virtualization derives
/// the cursor from `start` / `step` / `length` — but it is a pointer-shaped
/// slot, so it belongs in the census for the same reason `FUNCTION_DESCR_GROUP`
/// lists every slot: `clear_gc_fields` walks exactly this group's
/// `gc_fielddescrs` to emit a fresh object's delayed NULL stores, and a slot
/// left out would keep recycled nursery bytes that the collector then follows
/// as a child reference.  Its presence also keeps the list in byte-offset
/// order, which is what makes the positional numbering agree with the
/// analyzer's.
static RANGE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_Range>(),
        pyre_object::functional::W_RANGE_GC_TYPE_ID,
        &pyre_object::functional::RANGE_TYPE as *const _ as usize,
        &[
            (
                "start",
                RANGE_START_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "stop",
                RANGE_STOP_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "step",
                RANGE_STEP_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "length",
                RANGE_LENGTH_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "promote_step",
                RANGE_PROMOTE_STEP_OFFSET,
                1,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "W_Range",
        "functional::W_Range",
    )
});

/// The `Function` fields PyPy declares quasi-immutable — `function.py`
/// `_immutable_fields_ = ['code?', 'w_func_globals?', 'closure?[*]',
/// 'defs_w?[*]', 'name?', 'qualname?', 'w_objclass?', 'w_text_signature?',
/// 'w_kw_defs?']`.
///
/// The `?` makes each field quasi-immutable and `[*]` makes the selected
/// elements immutable after the field has been promoted.  All nine are marked
/// here through the `quasi` row: `descr.py:229 is_pure =
/// STRUCT._immutable_field(fieldname) != False` answers true for a
/// quasi-immutable field — `IR_QUASIIMMUTABLE` is an object, so `!= False`
/// holds — which is what the `is_immutable` column carries.  The rank is what
/// keeps the read from folding on that column alone:
/// `SimpleFieldDescr::is_always_pure` is `is_immutable && !is_quasi_immutable`,
/// so the fold stays behind `record_quasiimmut_field` + `GUARD_NOT_INVALIDATED`
/// exactly as `jtransform.py:895-903` sequences it.
///
/// The write half these descriptors depend on is `function_notify_quasi_immut`
/// (pyre-interpreter `function.rs`), called immediately before every store to
/// one of the nine — RPython gets it from the rtyper rewriting writes to a `?`
/// field, pyre spells it out.  `f.__defaults__ = ...` / `f.__code__ = ...`
/// therefore runs `QuasiImmut::invalidate()` and retires every loop that
/// folded the field, which is the termination guarantee a value guard cannot
/// give: a guard that can no longer pass is re-derived by the next trace
/// forever.  The inline-call path's `GuardValue`s stay on top; over-strictness
/// is sound, and they still carry the identity checks that path's later reads
/// assume.  A tuple's backing array has its own immutable descriptor and is
/// read with `GetarrayitemGcPureR`.
///
/// The declarative attribute stays off this struct. Emitting
/// `record_quasiimmut_field` for these nine fields would put the opcode in
/// every interpreter jitcode that reads them, and `pyjitpl/dispatch.rs` still
/// has no arm for that byte. The descriptors in this group are what the
/// hand-written trace paths resolve.
///
/// The entries are in byte-offset order and each key is the struct's own field
/// name, which is what makes the group's numbering agree with the analyzer's.
/// `index_in_parent` is a field's slot number in the virtual
/// (`virtualize.py:210 fielddescr.get_index()`) and its `PtrInfo._fields` key
/// in the heap optimizer, so two fields sharing one number makes a read of
/// either resolve to the other's cached value and a virtual keep only the
/// later of the two stores.  Two things assign it: `descr.py` caches a
/// FieldDescr per `(STRUCT, fieldname)` and a cache hit reports the analyzer's
/// number, which counts the struct's own fields past the flattened header;
/// a miss numbers the field by its position in this list.  Keeping the list in
/// offset order — with the inherited `PyObject.w_class` last, since the
/// analyzer's count does not include it — makes both answers the same one.
/// Look fields up by offset (`function_field_descr`) so the accessors below
/// cannot drift out of that order.
///
/// The census is COMPLETE — every pointer-shaped slot of `Function` is listed,
/// not only the four the inline-call path reads.  `emit_make_function_inline`
/// allocates a `Function` with `NewWithVtable`, and the nursery is not
/// zero-filled: `rewrite.py clear_gc_fields` emits the delayed NULL
/// stores, and it walks exactly this group's `gc_fielddescrs`.  A pointer slot
/// missing from the census would keep recycled nursery bytes and the collector
/// would follow that word as a child reference.  `can_change_code` is a plain
/// byte, so it gets no NULL store and the emit writes it explicitly.
static FUNCTION_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_interpreter::function as f;
    let field = |key, offset| {
        (
            key,
            offset,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            false,
            false,
        )
    };
    // A `function.py:34-42` entry: pure by `descr.py:229`, quasi by its `?`.
    let quasi = |key, offset| {
        (
            key,
            offset,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            true,
            true,
        )
    };
    build_object_descr_group_with_def_path(
        f::FUNCTION_OBJECT_SIZE,
        FUNCTION_GC_TYPE_ID,
        &pyre_interpreter::FUNCTION_TYPE as *const _ as usize,
        &[
            quasi("code", f::FUNCTION_CODE_OFFSET),
            // `function.py can_change_code = True`; `False` for the
            // `FunctionWithFixedCode` / `BuiltinFunction` subclasses.  A plain
            // byte, so `clear_gc_fields` leaves it alone and the value a
            // materialized function reports comes from the emit's own store.
            (
                "can_change_code",
                std::mem::offset_of!(pyre_interpreter::function::Function, can_change_code),
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            // `function.py self.name = forcename or code.co_name` — a
            // pointer to the name string's storage.  It is pointer-shaped and
            // the runtime walker visits it, so it belongs in the census; the
            // storage itself may be GC-managed (a mortal function's own box) or
            // permanent (a builtin's `malloc_raw` box, or the code object's
            // borrowed `co_name`), which the walker's managed-heap guard sorts
            // out.
            quasi("name", f::FUNCTION_NAME_OFFSET),
            // `w_name` has no `function.py:34-42` twin — upstream's `name?` is
            // the str this field wraps — so it stays a plain mutable slot.
            field("w_name", f::FUNCTION_W_NAME_OFFSET),
            quasi("closure", f::FUNCTION_CLOSURE_OFFSET),
            quasi("defs_w", f::FUNCTION_DEFS_W_OFFSET),
            quasi("w_kw_defs", f::FUNCTION_W_KW_DEFS_OFFSET),
            field("w_module", f::FUNCTION_W_MODULE_OFFSET),
            quasi("w_func_globals_obj", f::FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET),
            field("w_builtins", f::FUNCTION_W_BUILTINS_OFFSET),
            field("w_ann", f::FUNCTION_W_ANN_OFFSET),
            field("w_annotate", f::FUNCTION_W_ANNOTATE_OFFSET),
            field("w_func_dict", f::FUNCTION_W_FUNC_DICT_OFFSET),
            field("w_typeparams", f::FUNCTION_W_TYPEPARAMS_OFFSET),
            field("w_doc", f::FUNCTION_W_DOC_OFFSET),
            quasi("w_qualname", f::FUNCTION_W_QUALNAME_OFFSET),
            quasi("w_objclass", f::FUNCTION_W_OBJCLASS_OFFSET),
            quasi("w_text_signature", f::FUNCTION_W_TEXT_SIGNATURE_OFFSET),
            // `w_new_self` is absent from `FUNCTION_GC_PTR_OFFSETS` (it names a
            // static-region type that is never relocated), but it is still a
            // pointer slot an app-level `__self__` read dereferences, so it
            // needs the census entry that gets it NULLed behind the allocation.
            field("w_new_self", f::FUNCTION_W_NEW_SELF_OFFSET),
            field("w_moduleobj", f::FUNCTION_W_MODULEOBJ_OFFSET),
            // The `mutate_<name>` block pointer.  `Type::Ref` puts it in
            // `gc_fielddescrs`, which is what `rewrite.py
            // clear_gc_fields` walks, so a JIT-allocated function gets its NULL
            // from the allocation clear — the same footing upstream's
            // `mutate_<field>` sits on, that one being a real GC pointer.  The
            // collector does NOT follow it: its census is the hand-maintained
            // `FUNCTION_GC_PTR_OFFSETS`, which this word is deliberately absent
            // from, so the `Box` behind it is never traced as a child.
            field("mutate_slots", f::FUNCTION_MUTATE_SLOTS_OFFSET),
            // `make_weakref_descr(Function)` `_lifeline_` follows `mutate_slots`
            // in the struct, so it follows it here. Offset order is what the
            // analyzer's `index_in_parent` matches.
            field("lifeline", f::FUNCTION_LIFELINE_OFFSET),
            // The inline emit can escape a guard and be materialized, so the
            // inherited Python class is a proper virtual field of this group —
            // same reasoning as the `Method` / `W_ListObject` entries.  It sits
            // last, outside the offset ordering, because the analyzer counts
            // only the struct's own fields.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "Function",
        "function::Function",
    )
});

/// Dict storage descriptors, read after a trace has pinned the live strategy.
///
/// Positional `all_fielddescrs` is the heaptracker walk (`heaptracker.py
/// all_fielddescrs`): the struct's own field names in offset order
/// (`dstorage`, `dstrategy`, `keys_version`, `clear_gen`), with the
/// inherited class word prepended by the group factory. `newdict_empty` and
/// `w_dict_new_kwargs` store those fields, and `force_box` writes a virtual
/// field back through the slot `derive_index_in_parent` finds for its name.
///
/// `dstorage` is the GC storage box (`Type::Ref`, FLAG_POINTER from
/// `runtime_array_flag`). `dstrategy` is the static singleton pointer
/// (`kwargsdict.py` immortal strategy): constructors write it as a `Ref`
/// (`setfield_gc_r`) and the group field uses the same FLAG_POINTER, one
/// coherent pointer descr. `dstrategy_word` / `dstorage_lookup_ns` are
/// distinct Int cache keys minted outside the group by
/// `mint_w_dict_int_alias`, so `getfield_gc_i` folds and the `dict.lookup`
/// oopspec keep their identity without occupying a virtual slot.
static W_DICT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::dictmultiobject::W_DICT_OBJECT_SIZE,
        W_DICT_GC_TYPE_ID,
        &pyre_object::pyobject::DICT_TYPE as *const _ as usize,
        &[
            (
                "dstorage",
                std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, dstorage),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "dstrategy",
                std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, dstrategy),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "keys_version",
                std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, keys_version),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "clear_gen",
                std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, clear_gen),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "W_DictObject",
        "dictmultiobject::W_DictObject",
    )
});

fn mint_w_dict_int_alias(field_key: &str, display: &str, offset: usize) -> DescrRef {
    let _ = &*W_DICT_DESCR_GROUP;
    let struct_key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(
        "dictmultiobject::W_DictObject",
    ));
    let mut gc = majit_ir::descr::gc_cache().lock();
    gc.get_field_descr(
        struct_key,
        field_key,
        Some(display),
        offset,
        WORD,
        Type::Int,
        false,
        false,
        runtime_array_flag(Type::Int, false),
        stable_field_index(offset, WORD, Type::Int, false),
        false,
        None,
    ) as DescrRef
}

/// `dict.lookup` oopspec `extradescrs[0]` (`heap.py descrs[0]`).  A
/// distinct `(STRUCT, fieldname)` slot from positional `dstorage` so the
/// oopspec's identity is not the GC-ref view the constructor writes.
static W_DICT_LOOKUP_NS_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    mint_w_dict_int_alias(
        "dstorage_lookup_ns",
        "W_DictObject.dstorage_lookup_ns",
        std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, dstorage),
    )
});

/// Int view of `dstrategy` for `getfield_gc_i` / `GuardValue` folds.
static W_DICT_STRATEGY_WORD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    mint_w_dict_int_alias(
        "dstrategy_word",
        "W_DictObject.dstrategy_word",
        std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, dstrategy),
    )
});

/// `Method` field layout — `w_function`, `w_self`, `w_class`, `w_module`,
/// `lifeline`. All five are Ref slots; the JIT only consumes `w_function` (for guarding
/// which method) and `w_self` (for recovering the receiver `OpRef` discarded
/// by `LOAD_METHOD`). `w_class` and `w_module` are included for layout
/// completeness so the descrs match the struct order — a field the struct
/// declares but this census omits has no `index_in_parent` to rederive, so
/// the two sides that mint its descr disagree on the number.
///
/// `w_function` and `w_self` are marked immutable per
/// `pypy/interpreter/function.py`
/// `_Method._immutable_fields_ = ['w_function', 'w_instance']`. `w_class`
/// is not listed there and stays mutable; `w_module` is written after
/// construction by `w_method_set_module` and is mutable for that reason.
static W_METHOD_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::function::{
        METHOD_LIFELINE_OFFSET, METHOD_W_CLASS_OFFSET, METHOD_W_FUNCTION_OFFSET,
        METHOD_W_MODULE_OFFSET, METHOD_W_SELF_OFFSET, W_METHOD_GC_TYPE_ID, W_METHOD_OBJECT_SIZE,
    };
    build_object_descr_group_with_def_path(
        W_METHOD_OBJECT_SIZE,
        W_METHOD_GC_TYPE_ID,
        &pyre_object::function::METHOD_TYPE as *const _ as usize,
        &[
            (
                "w_function",
                METHOD_W_FUNCTION_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_self",
                METHOD_W_SELF_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_class",
                METHOD_W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_module",
                METHOD_W_MODULE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "lifeline",
                METHOD_LIFELINE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            // The inline bound-method emit (`emit_bound_method_inline`) can
            // escape a guard and be materialized, so the inherited Python
            // class needs to be a proper virtual field of this group — same
            // reasoning as the `PyObject.w_class` entry on W_ListObject.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "Method",
        "function::Method",
    )
});

/// `pypy/interpreter/function.py:673` / `:720`
/// `_immutable_fields_ = ['w_function?']` for `StaticMethod` and `ClassMethod`.
/// The `?` lives on [`STATICMETHOD_W_FUNCTION_QUASI_DESCR`] and its twin rather
/// than on this row: those two carry an owner-reserved index the layout cannot
/// supply, and `w_*_set_func` forces the invalidation.  This row stays LIVE and
/// describes the same word for an ordinary read of the slot.
///
/// Both censuses are COMPLETE — `w_dict` is listed even though nothing reads
/// it, because a field the struct declares but a group omits has no
/// `index_in_parent` to rederive and the two sides that mint its descr then
/// disagree on the number.  `PyObject.w_class` is absent because no emit
/// allocates either wrapper, so the analyzer's count is the whole answer.
static W_STATICMETHOD_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::function::{
        STATICMETHOD_W_DICT_OFFSET, STATICMETHOD_W_FUNCTION_OFFSET, W_STATICMETHOD_GC_TYPE_ID,
        W_STATICMETHOD_OBJECT_SIZE,
    };
    let field = |key, offset| {
        (
            key,
            offset,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            false,
            false,
        )
    };
    build_object_descr_group_with_def_path(
        W_STATICMETHOD_OBJECT_SIZE,
        W_STATICMETHOD_GC_TYPE_ID,
        &pyre_object::function::STATICMETHOD_TYPE as *const _ as usize,
        &[
            field("w_function", STATICMETHOD_W_FUNCTION_OFFSET),
            field("w_dict", STATICMETHOD_W_DICT_OFFSET),
        ],
        "StaticMethod",
        "function::StaticMethod",
    )
});

/// The `classmethod` twin of [`W_STATICMETHOD_DESCR_GROUP`]; see it for why
/// `w_function` is mutable and why `w_dict` is listed.
static W_CLASSMETHOD_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::function::{
        CLASSMETHOD_W_DICT_OFFSET, CLASSMETHOD_W_FUNCTION_OFFSET, W_CLASSMETHOD_GC_TYPE_ID,
        W_CLASSMETHOD_OBJECT_SIZE,
    };
    let field = |key, offset| {
        (
            key,
            offset,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            false,
            false,
        )
    };
    build_object_descr_group_with_def_path(
        W_CLASSMETHOD_OBJECT_SIZE,
        W_CLASSMETHOD_GC_TYPE_ID,
        &pyre_object::function::CLASSMETHOD_TYPE as *const _ as usize,
        &[
            field("w_function", CLASSMETHOD_W_FUNCTION_OFFSET),
            field("w_dict", CLASSMETHOD_W_DICT_OFFSET),
        ],
        "ClassMethod",
        "function::ClassMethod",
    )
});

/// `pypy/objspace/std/typeobject.py ObjectMutableCell`. The single
/// `w_value` field is read LIVE on the module-global cell fast path: a
/// frequently-rewritten global mutates the cell payload in place without
/// bumping `mstrategy.version` (`typeobject.py write_cell`), so the
/// JIT cell read must observe the current value and stays MUTABLE (not
/// immutable / quasi-immutable). Cell identity is what the `version?`
/// quasi-immutable guard protects.
static W_OBJECT_MUTABLE_CELL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::celldict::{
        OBJECT_MUTABLE_CELL_TYPE, W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS,
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID, W_OBJECT_MUTABLE_CELL_OBJECT_SIZE,
    };
    build_object_descr_group_with_def_path(
        W_OBJECT_MUTABLE_CELL_OBJECT_SIZE,
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID,
        &OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        &[(
            "w_value",
            W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS[0],
            WORD,
            Type::Ref,
            false,
            false,
            false,
        )],
        "ObjectMutableCell",
        "celldict::ObjectMutableCell",
    )
});

/// `typeobject.py ObjectMutableCell.w_value`. Mutable: `write_cell` rewrites
/// the payload in place and does not move `_version_tag`. The cell pointer is
/// the constant; this field is a live `getfield`.
pub fn object_mutable_cell_value_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_MUTABLE_CELL_DESCR_GROUP, 0)
}

/// `nestedscope.py Cell` field layout — `contents`, `family`.
///
/// `contents` is MUTABLE: `w_cell_set` rewrites it in place on every rebinding
/// of the closure variable, and `w_cell_delete` nulls it.
/// `CellFamily.ever_mutated` is the quasi-immutable that says whether any such
/// write has EVER happened; it says nothing about a given read, so this field
/// carries no immutability of its own.
///
/// `family` is `nestedscope.py Cell._immutable_fields_ = ['family']`, and it is
/// declared `Int` rather than `Ref` because the families are leaked rather than
/// GC-managed — the word is absent from this layout's traced pointer offsets.
/// Nothing reads it through this group; the census is COMPLETE because a field
/// the struct declares but a group omits has no `index_in_parent` to rederive,
/// and the two sides that mint its descr then disagree on the number.
static W_CELL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::nestedscope::{Cell, W_CELL_GC_TYPE_ID, W_CELL_OBJECT_SIZE};
    build_object_descr_group_with_def_path(
        W_CELL_OBJECT_SIZE,
        W_CELL_GC_TYPE_ID,
        &pyre_object::nestedscope::CELL_TYPE as *const _ as usize,
        &[
            (
                "contents",
                core::mem::offset_of!(Cell, contents),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "family",
                core::mem::offset_of!(Cell, family),
                8,
                Type::Int,
                false,
                true,
                false,
            ),
            // The cell a frame constructor allocates is emitted inline
            // ([`crate::helpers::emit_new_cell_inline`]) and can escape a guard
            // and be materialized, so the header class `w_cell_new` stamps has
            // to be a proper virtual field of this group -- the same reason
            // `W_Super` carries one.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "Cell",
        "nestedscope::Cell",
    )
});

/// `descriptor.py W_Super` field layout — `w_starttype`, `w_objtype`,
/// `w_self`, plus the inherited header `PyObject.w_class`.
///
/// All three are plain mutable Ref slots: `descriptor.py` declares no
/// `_immutable_fields_` for the proxy, and nothing here needs one — the
/// emitted proxy ([`crate::helpers::emit_super_inline`]) is read back through
/// the same trace that wrote it, where the optimizer answers from the
/// `SetfieldGc` it already has.
///
/// The header entry is there for the same reason the `Method` group carries
/// one: the inline emit can escape a guard and be materialized, so the
/// inherited Python class has to be a proper virtual field of this group.
static W_SUPER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::descriptor::{W_SUPER_GC_TYPE_ID, W_SUPER_OBJECT_SIZE, W_Super};
    let field = |key, offset| (key, offset, WORD, Type::Ref, false, false, false);
    build_object_descr_group_with_def_path(
        W_SUPER_OBJECT_SIZE,
        W_SUPER_GC_TYPE_ID,
        &pyre_object::descriptor::SUPER_TYPE as *const _ as usize,
        &[
            field("super_type", core::mem::offset_of!(W_Super, super_type)),
            field("obj_type", core::mem::offset_of!(W_Super, obj_type)),
            field("obj", core::mem::offset_of!(W_Super, obj)),
            field("PyObject.w_class", pyre_object::pyobject::W_CLASS_OFFSET),
        ],
        "W_Super",
        "descriptor::W_Super",
    )
});

static W_LIST_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // Upstream `rpython/rtyper/lltypesystem/rlist.py:116`
    //     GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))
    // The parity-field pair is `(length, items)`. `strategy` +
    // `int_items` / `float_items` / `bytes_items` are pyre-only PRE-EXISTING-
    // ADAPTATIONs for the PyPy interp-level strategy split.
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_ListObject>(),
        W_LIST_GC_TYPE_ID,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        &[
            // rlist.py `("length", Signed)`. Mutable: Object-strategy
            // push/pop/insert/remove/drain update it.
            (
                // `length` is a `usize` (8 bytes on 64-bit, 4 on wasm32). A
                // hardcoded 8 makes `GetfieldGcI` read the adjacent `items`
                // pointer into the high half on wasm32 → a garbage length →
                // out-of-bounds list access. Same fix as `str_len_descr`; the
                // `usize`/pointer fields below follow suit (the `Type::Ref`
                // fields are safe — read at pointer width regardless of size).
                "length",
                std::mem::offset_of!(W_ListObject, length),
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            // rlist.py:116 `("items", Ptr(GcArray(OBJECTPTR)))`. Points
            // at the `ItemsBlock` GcArray body. Mutable: re-pointed when
            // the Object-strategy storage is reallocated
            // (`list.object_grow` → `grow_list_items_block`) or when the
            // strategy switches.
            (
                "items",
                std::mem::offset_of!(W_ListObject, items),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                // `W_ListObject.strategy` is MUTABLE: `switch_to_object_strategy`
                // flips it from Integer/Float to Object when an
                // incompatible item is stored. A trace that folded
                // `strategy == Float` at trace-time into a constant would
                // then read from `float_items.block` (empty after the
                // switch) and dereference garbage — the spectral_norm n=10
                // SIGSEGV root cause.
                //
                // Upstream PyPy handles this with a quasi-immutable flag
                // + invalidate_compiled_code hook on strategy change;
                // pyre has no such hook yet, so `strategy` stays
                // plain-mutable. TODO — strategy split itself
                // is a pyre-only adaptation vs rlist.py.
                "strategy",
                std::mem::offset_of!(W_ListObject, strategy),
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            // Integer-strategy typed storage (pyre-only
            // TODO vs listobject.py's
            // IntegerListStrategy at the interp level — upstream keeps
            // the unwrap inline and doesn't add a separate backing
            // array).
            (
                "int_items.len",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            // Float-strategy typed storage.
            (
                "float_items.len",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            // `Ptr(GcArray(Signed))` / `Ptr(GcArray(Float))` — the typed
            // strategy backing blocks (`erase([int])` / `erase([float])`).
            // Read as a Ref so `GetarrayitemGcI` / `GetarrayitemGcF` address
            // items[i] through the heap cache. Mutable: re-pointed on grow /
            // strategy switch (like `W_ListObject.items`).
            (
                "int_items.block",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_BLOCK_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "float_items.block",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_BLOCK_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            // The inline list emit can escape through an exception's args_w
            // slot and be materialized.  Track the inherited Python class in
            // the same parent group as tuple objects so its SetfieldGc is a
            // proper virtual field and materialization reproduces w_list_new.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            // CPython 3.14 `PyListObject.allocated`.  The orthodox append
            // descent now walks `W_ListObject::sync_allocated`, so this field
            // must belong to the canonical list descr group just like every
            // other mutable field reached by the translated body.  Keep it at
            // the end so the established list field indices remain stable.
            (
                "allocated",
                std::mem::offset_of!(W_ListObject, allocated),
                std::mem::size_of::<isize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            // listobject.py `BytesListStrategy` stores erased `rpython str`
            // pointers in its own GcArray(GCREF).  Keep these entries at the
            // end so the established descriptor indices above remain stable.
            (
                "bytes_items.len",
                std::mem::offset_of!(W_ListObject, bytes_items)
                    + pyre_object::bytes_array::BYTES_ARRAY_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "bytes_items.block",
                std::mem::offset_of!(W_ListObject, bytes_items)
                    + pyre_object::bytes_array::BYTES_ARRAY_BLOCK_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            // AsciiListStrategy stores exact ASCII `_utf8` rpython-string
            // pointers in the same GcArray(GCREF) representation.
            (
                "ascii_items.len",
                std::mem::offset_of!(W_ListObject, ascii_items)
                    + pyre_object::unicode_array::UNICODE_ARRAY_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "ascii_items.block",
                std::mem::offset_of!(W_ListObject, ascii_items)
                    + pyre_object::unicode_array::UNICODE_ARRAY_BLOCK_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_ListObject",
        "listobject::W_ListObject",
    )
});

static W_TUPLE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `pypy/objspace/std/tupleobject.py` `W_TupleObject` stores
    // `wrappeditems: list` with `_immutable_fields_ =
    // ['wrappeditems[*]']`. After translation this becomes
    // `Ptr(GcArray(OBJECTPTR))`; `wrappeditems[*]` flows into both
    // the field descr (`immutable: true`) AND the GcArray contents
    // (read via `getfield_gc_pure_r`). Length comes from the GcArray
    // header via `arraylen_gc(items_block)` — no inline length cache.
    // Python 3.14's mutable-once `hash` cache is the intentional version
    // delta from that PyPy layout.
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_TupleObject>(),
        W_TUPLE_GC_TYPE_ID,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        &[
            // `Ptr(GcArray(OBJECTPTR))` — wrappeditems body. Immutable.
            (
                "wrappeditems",
                std::mem::offset_of!(W_TupleObject, wrappeditems),
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_TupleObject, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_TupleObject",
        "tupleobject::W_TupleObject",
    )
});

static SPECIALISED_TUPLE_II_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `specialisedtupleobject.py cls` `_immutable_fields_ = ['value0',
    // 'value1']` — both fields immutable. Inline-field shape, no array
    // indirection.
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_SpecialisedTupleObject_ii>(),
        SPECIALISED_TUPLE_II_GC_TYPE_ID,
        &SPECIALISED_TUPLE_II_TYPE as *const _ as usize,
        &[
            (
                "value0",
                SPECIALISED_TUPLE_II_VALUE0_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "value1",
                SPECIALISED_TUPLE_II_VALUE1_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_SpecialisedTupleObject_ii, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_ii",
        "specialisedtupleobject::W_SpecialisedTupleObject_ii",
    )
});

static SPECIALISED_TUPLE_FF_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_SpecialisedTupleObject_ff>(),
        SPECIALISED_TUPLE_FF_GC_TYPE_ID,
        &SPECIALISED_TUPLE_FF_TYPE as *const _ as usize,
        &[
            (
                "value0",
                SPECIALISED_TUPLE_FF_VALUE0_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "value1",
                SPECIALISED_TUPLE_FF_VALUE1_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_SpecialisedTupleObject_ff, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_ff",
        "specialisedtupleobject::W_SpecialisedTupleObject_ff",
    )
});

static SPECIALISED_TUPLE_OO_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_SpecialisedTupleObject_oo>(),
        SPECIALISED_TUPLE_OO_GC_TYPE_ID,
        &SPECIALISED_TUPLE_OO_TYPE as *const _ as usize,
        &[
            (
                "value0",
                SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "value1",
                SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_SpecialisedTupleObject_oo, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_oo",
        "specialisedtupleobject::W_SpecialisedTupleObject_oo",
    )
});

/// The `[capacity][items…]` header shared by every list/tuple backing block.
///
/// Headerless, and both halves of that are load-bearing.
/// `items_block_capacity_descr()` is the capacity read for all three list
/// strategies, whose blocks carry three different runtime tids
/// (`GC_INT_ARRAY_GC_TYPE_ID`,
/// `GC_FLOAT_ARRAY_GC_TYPE_ID`, `PY_OBJECT_ARRAY_GC_TYPE_ID` — see the typed
/// arms of `helpers::emit_promote_empty_list_inline`), so one descr fronting
/// them can name none; and `alloc_items_block`, the `PYRE_GC_ITEMSBLOCK=0` and
/// no-hook fallback, hands back a `std::alloc` block with no `GcHeader` at all.
/// Either way the loaded pointer must never reach `GUARD_GC_TYPE`, which reads
/// at `ref - GcHeader::SIZE`.
///
/// Leaving the type id at 0 does not express that on its own: the flag
/// `StructPtrInfo::make_guards` (`optimizeopt/info.rs`) consults is
/// `headerless`, and an unstamped gc-managed descr is one a second producer for
/// this key can have a collector identity stamped onto
/// (`GcCache::register_unresolved_struct_tids`).
static ITEMS_BLOCK_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_headerless_object_descr_group_with_def_path(
        pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET,
        &[(
            "capacity",
            pyre_object::object_array::ITEMS_BLOCK_LEN_OFFSET,
            std::mem::size_of::<usize>(),
            Type::Int,
            false,
            true,
            false,
        )],
        "ItemsBlock",
        "object_array::ItemsBlock",
    )
});

// The RPython `tuple2` returned by `rbigint.divmod` / `int_divmod`
// (`rbigint.py` `divmod` / `int_divmod`). Not a PyObject — no vtable, and the
// trace never NEWs one: it arrives as an elidable's result and is only read.
//
// Headerless, because `alloc_rbigint_pair_nursery_collecting` ends in
// `malloc_raw` when the collector has no tid for the pair or the nursery
// cannot satisfy the request, so a loaded pointer may have no `GcHeader` for
// `GUARD_GC_TYPE` to read.
//
// Both fields are `Type::Ref`, which is what puts them in `gc_fielddescrs`, and
// both are immutable, so the two reads CSE the way upstream's pair of
// `getfield_gc_r` off one `call_pure_r` does.
static RBIGINT_PAIR_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_headerless_object_descr_group_with_def_path(
        pyre_object::longobject::BIGINT_PAIR_SIZE,
        &[
            (
                "item0",
                pyre_object::longobject::BIGINT_PAIR_ITEM0_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "item1",
                pyre_object::longobject::BIGINT_PAIR_ITEM1_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
        "RBigIntPair",
        "rbigint::RBigIntPair",
    )
});

// `pyframe.py FrameDebugData` — the payload behind `self.getorcreatedebug()`,
// holding the frame's own locals mapping (read at the head of `fast2locals`)
// and the rare per-frame globals override (read by `get_w_globals`).  Not a
// PyObject — no vtable, and the trace never NEWs one: it arrives as the
// `debugdata` virtualizable field and is only read.
//
// Headerless, because `getorcreate_debug_data` falls back to `malloc_raw`
// whenever the owning frame is not collector-owned, so a loaded pointer may
// have no `GcHeader` to read.  A headered declaration has
// `StructPtrInfo::make_guards` certify the loaded pointer against a type id
// the GcCache mints from the structural key, which names nothing in the
// collector's type table and so fails on every check.
//
// Both fields are MUTABLE — `setdictscope` and `fast2locals`' lazy
// materialisation rebind `w_locals`, `set_w_globals` rebinds `w_globals` — so
// neither read may be treated as always-pure.
static FRAME_DEBUG_DATA_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_headerless_object_descr_group_with_def_path(
        pyre_interpreter::pyframe::FRAME_DEBUG_DATA_SIZE,
        &[
            (
                "w_globals",
                pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_GLOBALS_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_locals",
                pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_LOCALS_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_extra_locals",
                pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_EXTRA_LOCALS_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_f_trace",
                pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_F_TRACE_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "FrameDebugData",
        "pyframe::FrameDebugData",
    )
});

// `pypy/objspace/std/sliceobject.py` `W_SliceObject._immutable_fields_ =
// ['w_start', 'w_stop', 'w_step']` — all three Ref fields are immutable
// once `__init__` runs.  The `space.newslice(w_start, w_end, w_step)` JIT
// shape allocates the W_SliceObject inline so the optimizer can virtualize
// the three SetfieldGc writes when the slice never escapes (per
// `optimizeopt/virtualize.py optimize_NEW_WITH_VTABLE`).
static W_SLICE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::sliceobject::*;
    build_object_descr_group_with_def_path(
        W_SLICE_OBJECT_SIZE,
        W_SLICE_GC_TYPE_ID,
        &pyre_object::sliceobject::SLICE_TYPE as *const _ as usize,
        &[
            (
                "w_start",
                SLICE_START_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_stop",
                SLICE_STOP_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_step",
                SLICE_STEP_OFFSET,
                WORD,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
        "W_SliceObject",
        "sliceobject::W_SliceObject",
    )
});

/// `rvirtualizable.py` appends `('vable_token', llmemory.GCREF)` to the
/// virtualizable's own fields, so upstream's `gc_fielddescrs` names it and
/// `clear_gc_fields` zeroes the slot on every `new`.  pyre declares
/// `PyFrame.vable_token` as a plain `usize` and the positional census below
/// does not list it, so without this edge a JIT-inlined
/// `NewWithVtable(pyframe_size_descr())` leaves the slot holding recycled
/// nursery bytes — which `emit_force_virtualizable`'s `GETFIELD_GC_R` then
/// reads as a live GC reference (`pyjitpl.py` `emit_force_virtualizable`).
static PYFRAME_VABLE_TOKEN_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(|| {
    Arc::new(PyreFieldDescr {
        offset: crate::frame_layout::PYFRAME_VABLE_TOKEN_OFFSET,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Ref,
        signed: false,
        immutable: false,
        quasi_immutable: false,
        name: "vable_token",
        is_class_word: false,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
});

static PYFRAME_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_extra_gc_edges(
        std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        PYFRAME_GC_TYPE_ID,
        // `NewWithVtable` writes this typeptr at `cpu.vtable_offset`
        // (`OB_TYPE_OFFSET = 0`), populating the frame's `ob_header.ob_type`
        // so a JIT-built inline callee frame carries the same `frame` type
        // tag as a `FrameBox`-constructed one.
        &pyre_interpreter::pyframe::FRAME_TYPE as *const _ as usize,
        &[
            (
                "locals_cells_stack_w",
                crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "valuestackdepth",
                crate::frame_layout::PYFRAME_VALUESTACKDEPTH_OFFSET,
                // `usize`, not a fixed 64-bit int.  The other `Type::Int`
                // fields in this table are all `i64`, so 8 is right for
                // them; this one and `last_instr` below are the two that
                // are a machine word wide.  A literal 8 makes the store a
                // byte pair too wide on a 32-bit target, and the overrun
                // lands on `last_instr`, which sits immediately after it.
                std::mem::size_of::<usize>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "last_instr",
                crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET,
                // `isize` — see the width note on `valuestackdepth` above.
                std::mem::size_of::<isize>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "pycode",
                crate::frame_layout::PYFRAME_PYCODE_OFFSET,
                WORD,
                Type::Ref,
                true,
                false,
                false,
            ),
            (
                "PyFrame.debugdata",
                crate::frame_layout::PYFRAME_DEBUGDATA_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.f_generator_wref",
                crate::frame_layout::PYFRAME_F_GENERATOR_WREF_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.w_yielding_from",
                crate::frame_layout::PYFRAME_W_YIELDING_FROM_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.f_backref",
                crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.w_builtin",
                crate::frame_layout::PYFRAME_W_BUILTIN_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            // `pyframe.py:80 escaped` and its neighbours live in one `u8`, so
            // the traced `mark_as_escaped` is a byte-wide read-or-write, not a
            // word store.  Appended last: the accessors above address this
            // group by index.
            (
                "PyFrame.flags",
                crate::frame_layout::PYFRAME_FLAGS_OFFSET,
                std::mem::size_of::<u8>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "PyFrame.failed_attr_cleanup",
                crate::frame_layout::PYFRAME_FAILED_ATTR_CLEANUP_OFFSET,
                std::mem::size_of::<u8>(),
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "PyFrame",
        "pyframe::PyFrame",
        std::slice::from_ref(&PYFRAME_VABLE_TOKEN_FIELD_DESCR),
        &[],
        "",
        false,
    )
});

impl Descr for PyreSizeDescr {
    fn index(&self) -> u32 {
        SIZE_DESCR_TAG | (self.obj_size as u32 & 0x0FFF_FFFF)
    }

    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        Some(self)
    }
}

impl SizeDescr for PyreSizeDescr {
    fn size(&self) -> usize {
        self.obj_size
    }

    fn type_id(&self) -> u32 {
        self.type_id
    }

    /// Line-by-line equivalent of the `descr.py get_size_descr` cache
    /// identity: the slot key published by `register_keyed_size` (currently
    /// `path_hash_stripped_crate(module_path!(), bare_name)` —
    /// i.e. the def-path `path_hash("module::Bare")`). Returns the value
    /// computed by `__majit_type_id()` at macro publish time and stored in
    /// the instance slot. The reverse reader `bh_size_spec_from_descr` puts
    /// this value in `BhSizeSpec.type_id`, and `simple_descr_group_from_bh_size`
    /// round-trips to the publish slot as `LLType::Struct(spec.type_id)`.
    /// `type_id` (dense GC tid) and `cache_key` (structural identity) follow
    /// the split contract in the `SizeDescr::cache_key` trait doc.
    fn cache_key(&self) -> u64 {
        self.cache_key
    }

    fn vtable(&self) -> usize {
        self.vtable
    }

    fn w_class_obj(&self) -> Option<i64> {
        w_class_obj_for_vtable(self.vtable)
    }

    fn is_immutable(&self) -> bool {
        false
    }
    fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.all_fielddescrs
    }
    fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.gc_fielddescrs
    }
    /// descr.py SizeDescr.is_object: every PyreSizeDescr that ships a
    /// vtable corresponds to a Python object (W_IntObject / W_ListObject /
    /// W_IntRangeIterator / …). `ensure_ptr_info_arg0` (optimizer.py)
    /// uses this to dispatch InstancePtrInfo vs StructPtrInfo.
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
}

/// Empty-struct fallback for `BhDescr::Size` decode (`make_descr_from_bh`).
/// RPython `descr.py init_size_descr` records an empty
/// `all_fielddescrs` list when the underlying STRUCT has no GC fields, so
/// the consumer-side decoder still needs a constructor that produces a
/// `PyreSizeDescr` with an empty field-list rather than refusing to build
/// one. Producers carrying a populated field-list go through
/// `simple_descr_group_from_bh_size` instead.
pub fn make_size_descr_with_type_and_vtable(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
) -> DescrRef {
    // Empty fielddescr fallback — where the `BhDescr::Size` decoder is
    // called with no structural-identity carrier. `cache_key = 0` is safe
    // on round-trip because the no-identity branch of
    // `simple_descr_group_from_bh_size` treats each call as distinct.
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        cache_key: 0,
        vtable,
        all_fielddescrs: Vec::new(),
        gc_fielddescrs: Vec::new(),
    })
}

/// Synthetic `len` field descriptor matching upstream
/// `descr.py FieldDescr("len", ofs, WORD, FLAG_SIGNED)`. Lives at
/// offset 0 of the `Ptr(GcArray(T))` block (FixedObjectArray /
/// pyobject_gcarray layout): items start at `base_size`, so the word
/// before items is the length header. Stored as
/// `SimpleArrayDescr.lendescr` so `gen_initialize_len`
/// (`rewrite.py:565,572`) emits the runtime length store after
/// `CallMallocNurseryVarsize`.
fn array_lendescr_at_offset(offset: usize) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Int,
        signed: true,
        immutable: false,
        quasi_immutable: false,
        name: "len",
        is_class_word: false,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Lift `Option<usize>` ↔ `Option<DescrRef>` so `make_array_descr*`
/// callers express nolength/length-prefixed shapes directly. PyPy
/// `descr.py:359-362` decides this from
/// `ARRAY_INSIDE._hints.get('nolength', False)`; the explicit
/// `Option<usize>` is the structural equivalent.
fn maybe_array_lendescr_at_offset(len_offset: Option<usize>) -> Option<DescrRef> {
    len_offset.map(array_lendescr_at_offset)
}

/// Create a fresh ARRAY descriptor without identity carrier.
///
/// `len_offset`: `None` for the `nolength=True` shape (descr.py:360);
/// `Some(off)` for length-prefixed layouts (descr.py:362).
///
/// PyPy `descr.py get_array_descr(gccache, ARRAY)` keys
/// `_cache_array[ARRAY_OR_STRUCT]` on the ARRAY object identity, never
/// on its structural shape — two distinct lltype ARRAYs that share
/// `(base_size, item_size, item_type, signed, len_offset)` get
/// distinct `ArrayDescr` Arcs.  Pyre's no-identity-carrier callers
/// (this function: `array_type_id = None`, `type_id = 0`) cannot
/// participate in the keyed cache because they have no ARRAY-object
/// surrogate to hash; the orthodox behaviour is therefore "each call
/// is a distinct ARRAY" — mint fresh `SimpleArrayDescr` per call so
/// shape-coincident-but-logically-distinct ARRAYs receive distinct
/// `descr_id` slots.  Callers that need singleton semantics
/// (`int_array_descr`, `float_array_descr`, `pyobject_array_descr`,
/// …) route through [`make_array_descr_with_full_id`] with a stable
/// identity string instead — the keyed cache canonicalises by that
/// string.
pub fn make_array_descr(
    base_size: usize,
    item_size: usize,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    let descr_id = alloc_array_descr_id();
    // No identity carrier — fresh mint per call (cache_key = 0 means "no
    // cache slot").  Single flag-bearing `SimpleArrayDescr` (descr.py ArrayDescr);
    // index carries the content-addressed `ARRAY_DESCR_TAG | descr_id`.
    let index = ARRAY_DESCR_TAG | (descr_id & 0x0FFF_FFFF);
    let mut array_descr = majit_ir::descr::SimpleArrayDescr::with_flag(
        index,
        base_size,
        item_size,
        0,
        item_type,
        runtime_array_flag(item_type, signed),
    );
    array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
    Arc::new(array_descr)
}

pub fn make_array_descr_with_type(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    get_or_create_array_descr(base_size, item_size, type_id, item_type, signed, len_offset)
}

/// Bridge-only factory that threads the codewriter's `array_type_id`
/// (`majit-translate::codewriter::call::DescrIndexRegistry::array_index`
/// key) into `ArrayDescrKey` so two BhDescr::Array entries with
/// identical structural fields but different lltype spellings receive
/// distinct registry slots — matching upstream
/// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py get_array_descr`).
pub fn make_array_descr_with_full_id(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
    array_type_id: Option<String>,
) -> DescrRef {
    get_or_create_array_descr_with_full_id(
        base_size,
        item_size,
        type_id,
        item_type,
        signed,
        len_offset,
        array_type_id,
    )
}

// Range iterator field descriptors.

use pyre_object::complexobject::{
    COMPLEX_IMAG_OFFSET, COMPLEX_REAL_OFFSET, W_COMPLEX_GC_TYPE_ID, W_ComplexObject,
};
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::functional::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_ONE_ARG_CURRENT_OFFSET, RANGE_ITER_ONE_ARG_STOP_OFFSET,
    RANGE_ITER_REMAINING_OFFSET, RANGE_ITER_STEP_OFFSET, RANGE_ITER_STEP_ONE_CURRENT_OFFSET,
    RANGE_ITER_STEP_ONE_START_OFFSET, RANGE_ITER_STEP_ONE_STOP_OFFSET, RANGE_LENGTH_OFFSET,
    RANGE_PROMOTE_STEP_OFFSET, RANGE_START_OFFSET, RANGE_STEP_OFFSET, RANGE_STOP_OFFSET, W_Range,
};
use pyre_object::interp_exceptions::{
    EXC_ARGS_W_OFFSET, EXC_EXTENDED_USER_MAP_OFFSET, EXC_EXTENDED_USER_STORAGE_OFFSET,
    EXC_KIND_COUNT, EXC_KIND_OFFSET, EXC_SUPPRESS_CONTEXT_OFFSET, EXC_USER_MAP_OFFSET,
    EXC_USER_STORAGE_OFFSET, EXC_W_ATTR_OBJ_OFFSET, EXC_W_CAUSE_OFFSET, EXC_W_CODE_OFFSET,
    EXC_W_CONTEXT_OFFSET, EXC_W_DICT_OFFSET, EXC_W_ENCODING_OFFSET, EXC_W_END_OFFSET,
    EXC_W_ERRNO_OFFSET, EXC_W_FILENAME_OFFSET, EXC_W_FILENAME2_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_OFFSET, EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET, EXC_W_GROUP_MESSAGE_OFFSET,
    EXC_W_IMPORT_MSG_OFFSET, EXC_W_IMPORT_NAME_FROM_OFFSET, EXC_W_IMPORT_PATH_OFFSET,
    EXC_W_NAME_OFFSET, EXC_W_OBJECT_OFFSET, EXC_W_REASON_OFFSET, EXC_W_START_OFFSET,
    EXC_W_STRERROR_OFFSET, EXC_W_SYNTAX_END_LINENO_OFFSET, EXC_W_SYNTAX_END_OFFSET_OFFSET,
    EXC_W_SYNTAX_FILENAME_OFFSET, EXC_W_SYNTAX_LINENO_OFFSET, EXC_W_SYNTAX_METADATA_OFFSET,
    EXC_W_SYNTAX_MSG_OFFSET, EXC_W_SYNTAX_OFFSET_OFFSET, EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
    EXC_W_SYNTAX_TEXT_OFFSET, EXC_W_TRACEBACK_OFFSET, EXC_W_VALUE_OFFSET, EXC_W_WINERROR_OFFSET,
    EXC_WRITTEN_OFFSET, ExcKind, W_BASE_EXCEPTION_GC_PTR_OFFSETS, W_BASE_EXCEPTION_SIZE,
    W_BASE_EXCEPTION_USER_GC_TYPE_ID, W_BASE_EXCEPTION_USER_SIZE, W_EXCEPTION_EXTENDED_GC_TYPE_ID,
    W_EXCEPTION_EXTENDED_SIZE, W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID,
    W_EXCEPTION_EXTENDED_USER_SIZE, exc_kind_canonical_is_user_layout,
    exc_kind_uses_extended_layout, exc_realbase_pytype,
};
use pyre_object::intobject::W_IntObject;
use pyre_object::pyobject::W_CLASS_OFFSET;
use pyre_object::{
    BOOL_INTVAL_OFFSET, FLOAT_ARRAY_BLOCK_OFFSET, FLOAT_ARRAY_LEN_OFFSET, INT_ARRAY_BLOCK_OFFSET,
    INT_ARRAY_LEN_OFFSET, INT_INTVAL_OFFSET, W_ListObject, W_TupleObject,
};
// Re-import the rest without duplication
use pyre_object::{COMPLEX_TYPE, FLOAT_TYPE, INT_TYPE};

/// Field descriptor for `PyObject.w_class` (Ref, `w_class?`).
///
/// PyObject layout: [ob_type(8)] [w_class(8)]
/// The w_class field holds the Python class for all object types.
///
/// RPython parity: jit.promote(w_obj.__class__) reads typeptr via
/// getfield_gc_r then GUARD_VALUE. This is the pyre equivalent — a
/// field read on the common PyObject header.
///
/// `__class__` assignment can change it (`objectobject.py
/// descr_set___class__`), so the field is `?` not frozen: the optimizer
/// folds the load, and `notify_w_class_mutated` revokes those loops.
fn new_w_class_field_descr() -> Arc<dyn FieldDescr> {
    // Declares itself the class word (`is_class_word`), which is what
    // `FieldDescr::is_w_class()` reports; the name is for diagnostics only
    // and no consumer reads it. OptVirtualize must resolve this field from
    // the object's class identity rather than indexing it against a
    // virtual's value fields (its `index_in_parent` of 0 would otherwise
    // collide with the first value field, e.g. `W_IntObject.intval`).
    Arc::new(PyreFieldDescr {
        offset: pyre_object::pyobject::W_CLASS_OFFSET,
        // One pointer, sized from the target: the build-time descr pool
        // sizes this field by `layout::target_word_size()` (`call.rs
        // get_type_flag`), and the two universes must agree for the
        // canonical-descr bridge (`make_descr_from_bh`) and for
        // `state.rs materialize_virtual_object`'s w_class branch, both of
        // which compare widths. A fixed 8 would also overlap the first
        // payload field on wasm32, where the header is 8 bytes.
        field_size: WORD,
        field_type: Type::Ref,
        signed: false,
        immutable: false,
        quasi_immutable: true,
        name: "w_class",
        is_class_word: true,
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// The single `w_class` field descriptor for the shared `PyObject` header.
///
/// `descr.py get_field_descr` caches on `(STRUCT, fieldname)`, so a
/// given field is one object for the whole run and every consumer compares it
/// by identity: `heap.py` keys `cached_fields` by the descriptor itself
/// (`heap.rs`'s `cached_field_pos_for_descr` matches on `descr_identity`), and
/// so do the short-preamble export and `force_from_effectinfo`. Minting a
/// fresh `Arc` per call gave each `w_class` read in a trace its own identity,
/// so two reads of the same header could never share a cache entry.
static W_CLASS_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(new_w_class_field_descr);

pub fn w_class_descr() -> DescrRef {
    W_CLASS_FIELD_DESCR.clone() as DescrRef
}

/// The canonical `w_class` (Python class object) for instances of the type
/// `vtable` names — `get_instantiate(vtable_type)`. Read live (not cached at
/// construction) since the type objects are installed after the descrs are
/// built. `None` before `init_typeobjects()` runs.
///
/// Registered as majit's [`majit_ir::descr::WClassObjFn`] so the generic
/// `SimpleSizeDescr` — which every runtime PyObject group and every
/// blackhole-dispatch size descr is built as — answers `w_class_obj` the same
/// way `PyreSizeDescr` does. Without it `OptVirtualize` cannot fold the
/// `w_class` header read off a `new_with_vtable` virtual and forces the
/// virtual instead.
pub fn w_class_obj_for_vtable(vtable: usize) -> Option<i64> {
    if vtable == 0 {
        return None;
    }
    // A `JitVirtualRef` names itself with a type tag, not a `PyType` address,
    // and has no `w_class` to answer with — reading the tag as a type would
    // dereference a constant.
    if vtable == majit_metainterp::virtualref::JIT_VIRTUAL_REF_VTABLE as usize {
        return None;
    }
    let tp = vtable as *const pyre_object::pyobject::PyType;
    let w_class = unsafe { pyre_object::pyobject::get_instantiate(&*tp) };
    if w_class.is_null() {
        None
    } else {
        Some(w_class as i64)
    }
}

/// Alias for backward compatibility — same as w_class_descr().
pub fn instance_w_type_descr() -> DescrRef {
    w_class_descr()
}

/// Field descriptor for `W_IntRangeIterator.current` (i64, signed).
pub fn range_iter_current_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_IntRangeIterator.remaining` (i64, signed).
pub fn range_iter_remaining_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 1)
}

/// Field descriptor for `W_IntRangeIterator.step` (i64, signed).
pub fn range_iter_step_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 2)
}

/// Field descriptor for `W_IntRangeStepOneIterator.current` (i64, signed).
pub fn range_iter_step_one_current_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_STEP_ONE_DESCR_GROUP, 0)
}

/// Field descriptor for `W_IntRangeStepOneIterator.stop` (i64, immutable).
pub fn range_iter_step_one_stop_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_STEP_ONE_DESCR_GROUP, 1)
}

/// Field descriptor for `W_IntRangeStepOneIterator.start` (i64, immutable).
pub fn range_iter_step_one_start_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_STEP_ONE_DESCR_GROUP, 2)
}

/// Field descriptor for `W_IntRangeOneArgIterator.current` (i64, signed).
pub fn range_iter_one_arg_current_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_ONE_ARG_DESCR_GROUP, 0)
}

/// Field descriptor for `W_IntRangeOneArgIterator.stop` (i64, immutable).
pub fn range_iter_one_arg_stop_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_ONE_ARG_DESCR_GROUP, 1)
}

/// Field descriptor for `W_SeqIterObject.seq`.
pub fn seq_iter_seq_descr() -> DescrRef {
    field_descr_from_group(&SEQ_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_SeqIterObject.index`.
pub fn seq_iter_index_descr() -> DescrRef {
    field_descr_from_group(&SEQ_ITER_DESCR_GROUP, 1)
}

/// `iterobject.py W_FastListIterObject.w_seq`.
pub fn list_iter_seq_descr() -> DescrRef {
    field_descr_from_group(&LIST_ITER_DESCR_GROUP, 0)
}

/// `iterobject.py W_FastListIterObject.index`.
pub fn list_iter_index_descr() -> DescrRef {
    field_descr_from_group(&LIST_ITER_DESCR_GROUP, 1)
}

/// `FrameLocalsProxy.w_frame`.
pub fn frame_locals_proxy_w_frame_descr() -> DescrRef {
    field_descr_from_group(&FRAME_LOCALS_PROXY_DESCR_GROUP, 0)
}

/// Field descriptor for `W_TupleIterObject.seq`.
pub fn tuple_iter_seq_descr() -> DescrRef {
    field_descr_from_group(&TUPLE_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_TupleIterObject.index`.
pub fn tuple_iter_index_descr() -> DescrRef {
    field_descr_from_group(&TUPLE_ITER_DESCR_GROUP, 1)
}

/// Immutable iterator-list edge of `functional.py W_Zip`.
pub fn zip_iterators_descr() -> DescrRef {
    field_descr_from_group(&W_ZIP_DESCR_GROUP, 0)
}

/// Mutable `strict` flag of `functional.py W_Zip` (`__setstate__`).
pub fn zip_strict_descr() -> DescrRef {
    field_descr_from_group(&W_ZIP_DESCR_GROUP, 1)
}

/// `_iteration_progress`, updated while one zip row is fetched.
pub fn zip_iteration_progress_descr() -> DescrRef {
    field_descr_from_group(&W_ZIP_DESCR_GROUP, 2)
}

/// Resolve one [`RANGE_DESCR_GROUP`] field by byte offset, so the accessors
/// below stay correct however the census is ordered.  They were positional
/// until `stop` joined the census and shifted every later slot.
fn range_field_descr(offset: usize) -> DescrRef {
    let parent = RANGE_DESCR_GROUP.size_descr.clone() as DescrRef;
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

/// Field descriptor for `W_Range.start` (wrapped PyObjectRef).
pub fn range_start_descr() -> DescrRef {
    range_field_descr(RANGE_START_OFFSET)
}

/// Field descriptor for `W_Range.stop` (wrapped PyObjectRef).
pub fn range_stop_descr() -> DescrRef {
    range_field_descr(RANGE_STOP_OFFSET)
}

/// Field descriptor for `W_Range.step` (wrapped PyObjectRef).
pub fn range_step_descr() -> DescrRef {
    range_field_descr(RANGE_STEP_OFFSET)
}

/// Field descriptor for `W_Range.length` (wrapped PyObjectRef).
pub fn range_length_descr() -> DescrRef {
    range_field_descr(RANGE_LENGTH_OFFSET)
}

/// Field descriptor for `W_Range.promote_step` (one byte).
pub fn range_promote_step_descr() -> DescrRef {
    range_field_descr(RANGE_PROMOTE_STEP_OFFSET)
}

/// `Method.w_function` — the underlying function (`Function` or
/// `BuiltinFunction`) bound by `getattr(obj, name)`. Marked immutable
/// per `pypy/interpreter/function.py` `_Method._immutable_fields_`,
/// so reads survive cache invalidation across calls. Used by the
/// bound-method specialization in `call_callable_value`.
pub fn method_w_function_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 0)
}

/// `function.py StaticMethod._immutable_fields_ = ['w_function?']` — the
/// callable a descriptor fold unwraps in place of invoking `__get__`.
///
/// Minted standalone rather than read out of [`W_STATICMETHOD_DESCR_GROUP`],
/// the way the `w_fget?` / `w_fset?` twins are: the group's own row derives its
/// index from the layout, and this owner needs an index of its own for the
/// reason [`WRAPPER_DESCR_TAG`] records.  The group keeps its census row for
/// the field's ordinary reads.
static STATICMETHOD_W_FUNCTION_QUASI_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            STATICMETHOD_W_FUNCTION_INDEX,
            pyre_object::function::STATICMETHOD_W_FUNCTION_OFFSET,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "StaticMethod.w_function".to_string(),
            "w_function".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn staticmethod_w_function_quasi_descr() -> DescrRef {
    STATICMETHOD_W_FUNCTION_QUASI_DESCR.clone()
}

/// The `function.py ClassMethod` twin of
/// [`STATICMETHOD_W_FUNCTION_QUASI_DESCR`].
static CLASSMETHOD_W_FUNCTION_QUASI_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            CLASSMETHOD_W_FUNCTION_INDEX,
            pyre_object::function::CLASSMETHOD_W_FUNCTION_OFFSET,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "ClassMethod.w_function".to_string(),
            "w_function".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn classmethod_w_function_quasi_descr() -> DescrRef {
    CLASSMETHOD_W_FUNCTION_QUASI_DESCR.clone()
}

/// Resolve one [`FUNCTION_DESCR_GROUP`] field by byte offset, so the accessors
/// below stay correct however the census is ordered.
fn function_field_descr(offset: usize) -> DescrRef {
    let parent = FUNCTION_DESCR_GROUP.size_descr.clone() as DescrRef;
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

/// `Function.defs_w` — the positional defaults the inline path reads.
/// `defs_w?[*]`, so the read is paired with `record_quasiimmut_field`.
pub fn function_defs_w_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_DEFS_W_OFFSET)
}

/// Live `Function.code` — the field `Function.getcode()` promotes
/// (`function.py jit.promote(self.code)`).  This is what identifies an
/// inlined body, so the inline lever guards it instead of the function object.
pub fn function_code_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_CODE_OFFSET)
}

/// Live `Function.w_func_globals_obj` — the namespace an inlined callee's
/// LOAD_GLOBAL folds against.
pub fn function_w_globals_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET)
}

/// Live `Function.closure` — the freevar cell tuple threaded into the inlined
/// callee's own frame.
pub fn function_closure_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_CLOSURE_OFFSET)
}

/// `function.py:51 self.name` — the pointer to the name string's storage.
pub fn function_name_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_NAME_OFFSET)
}

/// Wrapped `Function.__name__` object stamped at construction from `co_name`.
pub fn function_w_name_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_NAME_OFFSET)
}

/// `Function.w_builtins` — the builtin mapping resolved from the globals
/// once at construction and then frozen (rebinding `globals['__builtins__']`
/// afterwards does not change it).
pub fn function_w_builtins_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_BUILTINS_OFFSET)
}

/// `function.py self.qualname = qualname or self.name` — the wrapped
/// qualified name stamped at construction from the code object.
pub fn function_w_qualname_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_QUALNAME_OFFSET)
}

/// `Function.w_ann` — the eager `__annotations__` dict.  A plain mutable slot:
/// it has no `function.py:34-42` twin, so no watcher pairs with it.
pub fn function_w_ann_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_ANN_OFFSET)
}

/// `Function.w_annotate` — the PEP 649 `__annotate__` callable the runtime
/// `__annotations__` getter evaluates lazily.  Plain and mutable, like its
/// `w_ann` sibling.
pub fn function_w_annotate_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_ANNOTATE_OFFSET)
}

/// `Function.w_kw_defs` — `w_kw_defs?`, the keyword-only defaults mapping.
pub fn function_w_kw_defs_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_KW_DEFS_OFFSET)
}

/// `Function.w_objclass` — `w_objclass?`, the owning class a descriptor-backed
/// builtin reports through `__objclass__`.
pub fn function_w_objclass_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_OBJCLASS_OFFSET)
}

/// `Function.w_text_signature` — `w_text_signature?`, the parsed `__text_signature__`.
pub fn function_w_text_signature_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_TEXT_SIGNATURE_OFFSET)
}

/// Resolve a field-descr index to the `Function` watcher slot it names, or
/// `None` when the index is not one of the nine `function.py:34-42` fields.
///
/// `quasiimmut.py get_current_qmut_instance` reaches the hidden
/// `mutate_<name>` field through a `mutatefielddescr` the rtyper minted next to
/// the field's own descr.  Pyre carries the same pairing as this table: both
/// the record-time install (`state.rs`) and the compile-time watcher
/// registration (`pyre-jit/src/eval.rs`) key on `descr.index()`, so they resolve
/// through one list instead of two chains that could drift apart.
///
/// The index carries no struct identity — `stable_field_index` derives it from
/// `(offset, field_size, field_type, signed)` — so keying an arm on it is only
/// safe while no other quasi-immutable descriptor can land on the same number.
/// The other owners that address a real struct field either use a different
/// `Type` ([`TYPE_VERSION_TAG_FIELD_DESCR`], [`MODULE_DICT_VERSION_FIELD_DESCR`])
/// or carry a reserved tag outside `FIELD_DESCR_TAG`
/// ([`MAPDICT_DESCR_TAG`], [`PROPERTY_DESCR_TAG`], [`WRAPPER_DESCR_TAG`],
/// [`CELL_FAMILY_DESCR_TAG`], [`EC_W_TRACEFUNC_INDEX`]).  A descriptor that
/// is not quasi-immutable reaches neither caller.
pub fn function_quasi_immut_slot(index: u32) -> Option<pyre_interpreter::function::QuasiImmutSlot> {
    use pyre_interpreter::function::QuasiImmutSlot;
    static SLOTS: LazyLock<Vec<(u32, QuasiImmutSlot)>> = LazyLock::new(|| {
        vec![
            (function_code_descr().index(), QuasiImmutSlot::Code),
            (
                function_w_globals_descr().index(),
                QuasiImmutSlot::WFuncGlobalsObj,
            ),
            (function_closure_descr().index(), QuasiImmutSlot::Closure),
            (function_defs_w_descr().index(), QuasiImmutSlot::DefsW),
            (function_name_descr().index(), QuasiImmutSlot::Name),
            (
                function_w_qualname_descr().index(),
                QuasiImmutSlot::WQualname,
            ),
            (
                function_w_objclass_descr().index(),
                QuasiImmutSlot::WObjclass,
            ),
            (
                function_w_text_signature_descr().index(),
                QuasiImmutSlot::WTextSignature,
            ),
            (function_w_kw_defs_descr().index(), QuasiImmutSlot::WKwDefs),
        ]
    });
    SLOTS
        .iter()
        .find(|(slot_index, _)| *slot_index == index)
        .map(|(_, slot)| *slot)
}

/// `function.py can_change_code = True` — a plain byte, so a fresh
/// allocation does not zero it and an emit must write it.
pub fn function_can_change_code_descr() -> DescrRef {
    function_field_descr(std::mem::offset_of!(
        pyre_interpreter::function::Function,
        can_change_code
    ))
}

/// Inherited `PyObject.w_class` on a `Function` — the Python-level `function`
/// class the constructor's header stamps.  Kept in the Function group (not the
/// standalone `w_class_descr`) so an inline emit's store is a virtual field of
/// the same size descr and materialization reproduces the header.
pub fn function_header_w_class_descr() -> DescrRef {
    function_field_descr(pyre_object::pyobject::W_CLASS_OFFSET)
}

/// Size descriptor for `Function` allocation via `NewWithVtable`
/// (vtable = `&FUNCTION_TYPE`).
pub fn w_function_size_descr() -> DescrRef {
    FUNCTION_DESCR_GROUP.size_descr.clone()
}

/// `W_DictObject.dstorage` — the strategy-erased storage box
/// (`dictmultiobject.py`). Read as a `Ref` only after the trace has pinned
/// the strategy through [`dict_strategy_word_descr`], because what the pointer
/// addresses is whatever the live strategy erased into it.
pub fn dict_dstorage_descr() -> DescrRef {
    field_descr_from_group(&W_DICT_DESCR_GROUP, 0)
}

/// `W_DictObject.dstrategy` as a raw word, for the `GuardValue` that pins a
/// dict to one strategy singleton.
///
/// `keys_version` is not an alternative for that guard — `MapDictStrategy`
/// never bumps it, which is why the `dict.get` fold declines a Map-backed
/// dictionary outright.
pub fn dict_strategy_word_descr() -> DescrRef {
    W_DICT_STRATEGY_WORD_DESCR.clone()
}

/// The cache-namespace half of the `dict.lookup` oopspec's `extradescrs`
/// (`heap.py descrs[0]`), naming the entry table a lookup probes.
/// Only its identity is read; the slot is never loaded.
pub fn dict_lookup_namespace_descr() -> DescrRef {
    W_DICT_LOOKUP_NS_DESCR.clone()
}

/// `extradescrs[1]` — the entry-array descr `_optimize_CALL_DICT_LOOKUP`
/// registers in `corresponding_array_descrs` so a call that writes the table
/// drops the cached lookups.  No pyre helper declares it in
/// `write_descrs_arrays` yet, so its `ei_index` stays `u32::MAX` and
/// `check_write_descr_array` never fires for it; the cache is dropped instead
/// by `clean_caches`, which every dict-mutating residual triggers.
pub fn dict_lookup_entries_array_descr() -> DescrRef {
    static DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
        make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Ref,
            false,
            Some("dict.lookup.entries".to_string()),
        )
    });
    DESCR.clone()
}

/// `Method.w_self` — the receiver object. The bound-method
/// specialization extracts this via `GetfieldGcR` to recover the receiver
/// `OpRef` after `LOAD_METHOD` discarded it (`compute_load_method_bound` in
/// `eval.rs` yields `null_value` for `is_method` attrs). Immutable per
/// `_Method._immutable_fields_`.
pub fn method_w_self_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 1)
}

/// `Method.w_class` — the class the bound function was found on
/// (`function.py` `Method.w_class`), distinct from the inherited
/// `PyObject.w_class` naming the Method's own type.
pub fn method_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 2)
}

/// `Method.w_module` — `__module__` storage for a bound builtin method,
/// written after construction by `w_method_set_module`. The JIT does not
/// read it; the census entry exists so the struct's field order is complete.
pub fn method_w_module_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 3)
}

/// Inherited `PyObject.w_class` on a `Method` — the Python-level `method`
/// class stamped by `w_method_new`'s header. Kept in the Method group (not
/// the standalone `w_class_descr`) so an inline emit's store is a virtual
/// field of the same size descr and materialization reproduces the header.
pub fn method_header_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 5)
}

/// Size descriptor for `Method` allocation via `NewWithVtable`
/// (vtable = `&METHOD_TYPE`); `w_function` / `w_self` / `w_class` and the
/// inherited header `w_class` are `SetfieldGc`'d after.
pub fn w_method_size_descr() -> DescrRef {
    W_METHOD_DESCR_GROUP.size_descr.clone()
}

/// `nestedscope.py Cell.contents` — what `w_cell_get` reads.  The modelled
/// `fast2locals` expansion reads it once per cell slot, in place of the
/// residual `bh_call_fn(locals)` whose frame force loses the enclosing loop.
pub fn cell_contents_descr() -> DescrRef {
    field_descr_from_group(&W_CELL_DESCR_GROUP, 0)
}

/// `nestedscope.py Cell.family` — the [`pyre_object::nestedscope::CellFamily`]
/// every cell of one cellvar shares.  Immutable per
/// `_immutable_fields_ = ['family']`, and a leaked (untraced) address rather
/// than a managed reference, so the descriptor spells it `Type::Int`.
pub fn cell_family_descr() -> DescrRef {
    field_descr_from_group(&W_CELL_DESCR_GROUP, 1)
}

/// Inherited `PyObject.w_class` on a `Cell` — kept in this group for the same
/// reason [`super_header_w_class_descr`] is kept in the `W_Super` one.
pub fn cell_header_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_CELL_DESCR_GROUP, 2)
}

/// Size descriptor for `nestedscope.py Cell` allocation via `NewWithVtable`
/// (vtable = `&CELL_TYPE`); `family` and the inherited header `w_class` are
/// `SetfieldGc`'d after, while `contents` keeps the null a virtual reads for
/// an unwritten reference field.
pub fn w_cell_size_descr() -> DescrRef {
    W_CELL_DESCR_GROUP.size_descr.clone()
}

/// `descriptor.py W_Super.w_starttype` — the class `super()` was given, whose
/// MRO position the attribute walk starts AFTER.
pub fn super_start_type_descr() -> DescrRef {
    field_descr_from_group(&W_SUPER_DESCR_GROUP, 0)
}

/// `descriptor.py W_Super.w_objtype` — `_super_check`'s answer, the type whose
/// MRO is walked.
pub fn super_obj_type_descr() -> DescrRef {
    field_descr_from_group(&W_SUPER_DESCR_GROUP, 1)
}

/// `descriptor.py W_Super.w_self` — the receiver a found descriptor binds to.
pub fn super_obj_descr() -> DescrRef {
    field_descr_from_group(&W_SUPER_DESCR_GROUP, 2)
}

/// Inherited `PyObject.w_class` on a `W_Super` — the Python-level `super`
/// class stamped by `w_super_new`'s header, kept in this group for the same
/// reason [`method_header_w_class_descr`] is kept in the `Method` one.
pub fn super_header_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_SUPER_DESCR_GROUP, 3)
}

/// Size descriptor for `W_Super` allocation via `NewWithVtable`
/// (vtable = `&SUPER_TYPE`); the three proxy fields and the inherited header
/// `w_class` are `SetfieldGc`'d after.
pub fn w_super_size_descr() -> DescrRef {
    W_SUPER_DESCR_GROUP.size_descr.clone()
}

/// `typeobject.py IntMutableCell.intvalue` — the unboxed `Signed`
/// payload of a module-global int cell, read LIVE on the cell fast path.
/// `write_cell` rewrites it in place when a hot int global is reassigned to
/// another int (no strategy-version bump), so this descriptor is mutable
/// (not immutable / quasi-immutable); the `version?` guard protects cell
/// identity, not the payload. Reassigning to a non-int replaces the cell,
/// bumping the version and invalidating the fold.
/// Descriptor for `IntMutableCell.intvalue`.  Minted with a reserved unique
/// [`INT_MUTABLE_CELL_VALUE_INDEX`] rather than `stable_field_index` because
/// that field's `(offset 16, size 8, Int)` layout collides with
/// `W_IntObject.intval` / `W_ListObject.length` in the runtime `HeapCache`
/// key space (which keys by `descr.index()`; see [`CELL_DESCR_TAG`]).
///
/// A SINGLETON `Arc`, one descr per field exactly as upstream's codewriter
/// produces one `FieldDescr` per `IntMutableCell.inst_intvalue`.  The
/// optimizer's `cached_fields` is keyed by `descr_identity`
/// (`Arc::as_ptr`), so the LOAD `getfield_gc_i` and the STORE
/// `setfield_gc_i` MUST share the Arc: with per-call fresh Arcs the store's
/// lazy `setfield` lives in a `CachedField` the load's lookup never finds,
/// the load skips heap.py `possible_aliasing_two_infos` entirely, and
/// `force_lazy_sets_for_guard` later flushes the store BELOW the emitted
/// load — reordering a store past a load of the same location (the nested
/// module-loop `i = i + 1; while i < n` reads the pre-increment value and
/// runs one extra iteration).  Distinct cells (`i`/`j`/`k`) do NOT
/// cross-forward under the shared descr: `CachedField` distinguishes
/// structs by the obj operand (`same_box` MUST_ALIAS / UNKNOWN_ALIAS →
/// `force_lazy_set`, heap.py).  Signed `i64` payload, mutable
/// (`write_cell` rewrites `intvalue` in place for an int->int reassign with
/// no version bump).
pub fn int_mutable_cell_value_descr() -> DescrRef {
    static DESCR: std::sync::OnceLock<DescrRef> = std::sync::OnceLock::new();
    DESCR
        .get_or_init(|| {
            Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
                INT_MUTABLE_CELL_VALUE_INDEX,
                core::mem::offset_of!(pyre_object::celldict::IntMutableCell, intvalue),
                8,
                Type::Int,
                false,
                majit_ir::descr::ArrayFlag::Signed,
                "IntMutableCell.intvalue".to_string(),
                "intvalue".to_string(),
            ))
        })
        .clone()
}

/// Size descriptor for `W_ListObject` allocation via NewWithVtable.
/// vtable = &LIST_TYPE; the Object-strategy fields `length` / `items` /
/// `strategy` are SetField'd after. `int_items.block` / `float_items.block` /
/// `bytes_items.block`
/// are GC-pointer fields of this descr, so `rewrite.py:498-504
/// clear_gc_fields` zeroes them behind the allocation (== empty, never read
/// under the Object strategy); their `len` halves are plain ints and stay at
/// whatever the recycled nursery bytes held, which no strategy reads while the
/// block slot is null.
pub fn w_list_size_descr() -> DescrRef {
    W_LIST_DESCR_GROUP.size_descr.clone()
}

/// `typeobject.py:162 _version_tag` — the method-cache version (`u64`, 8
/// bytes, unsigned) on `W_TypeObject`. The `LOAD_METHOD` fast path reads it
/// through `promote(self.version_tag())` (typeobject.py) so the
/// `_pure_lookup_where_with_method_cache` `CALL_PURE_R` folds on a green
/// version.
///
/// Quasi-immutable, per `typeobject.py:177 _immutable_fields_ =
/// ['_version_tag?']`: the read is replaced by a `QUASIIMMUT_FIELD` plus one
/// `GUARD_NOT_INVALIDATED` per trace, and `mutated()` (typeobject.py)
/// revokes the dependent loops through
/// [`pyre_object::typeobject::w_type_set_version_tag`] instead of the trace
/// re-checking a live value every iteration. A residual call cannot flush a
/// quasi-immutable field, which is the whole point — the interim live read +
/// `guard_value` had to be re-done after every un-inlined call in the body.
///
/// One object per run, for the identity reason documented on
/// [`W_CLASS_FIELD_DESCR`], and load-bearing here: `heap.rs` keys
/// `quasi_immut_cache` on `field_cache_identity`, which is the `Arc` pointer,
/// so a per-call descriptor would miss its own cache on every read.
static TYPE_VERSION_TAG_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    make_quasi_immutable_field_descr(
        core::mem::offset_of!(pyre_object::typeobject::W_TypeObject, version_tag),
        8,
        Type::Int,
        false,
    )
});

pub fn type_version_tag_descr() -> DescrRef {
    TYPE_VERSION_TAG_FIELD_DESCR.clone()
}

/// `typeobject.py W_TypeObject.name?` — the app-level `type.__name__`
/// object, which the `Cls.__name__` fold reads instead of the opaque `getattr`
/// residual.
///
/// Quasi-immutable: `descr_set__name__` (typeobject.py) replaces the slot
/// without calling `mutated()`, so a rename moves this field and nothing
/// else.  `QUASIIMMUT_FIELD` on this descr is what revokes a baked name.
///
/// One object per run for the identity reason [`W_CLASS_FIELD_DESCR`]
/// documents — `heap.rs` keys its field cache on the `Arc` pointer, so a
/// per-call descriptor would miss its own cache on every read.  The size
/// follows the same descriptor's: one pointer, from the target.
static TYPE_NAME_OBJ_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    make_quasi_immutable_field_descr(
        core::mem::offset_of!(pyre_object::typeobject::W_TypeObject, w_name),
        WORD,
        Type::Ref,
        false,
    )
});

pub fn type_name_obj_descr() -> DescrRef {
    TYPE_NAME_OBJ_FIELD_DESCR.clone()
}

/// `celldict.py ModuleDictStrategy.version` — the module-namespace version
/// tag (`u64`, 8 bytes, unsigned) on the strategy box.
///
/// Quasi-immutable, per `celldict.py _immutable_fields_ = ["version?"]`,
/// which is the same declaration `getdictvalue_no_unwrapping` promotes before
/// its elidable lookup (`celldict.py`). The `LOAD_GLOBAL` / `STORE_GLOBAL`
/// cell folds bake the slot's stored cell as a `ConstPtr` under a
/// `QUASIIMMUT_FIELD` on this field: `_setitem_str_cell_known`
/// (`celldict.py`) calls `mutated()` before every write that replaces the
/// stored pointer, and an in-place cell write leaves the pointer alone, so the
/// version is exactly the datum that proves the baked address still stands.
///
/// `offset_of!` rather than a literal because `ModuleDictStrategy` is not
/// `repr(C)`.
///
/// One object per run, for the reason spelled out on
/// [`TYPE_VERSION_TAG_FIELD_DESCR`].
static MODULE_DICT_VERSION_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    make_quasi_immutable_field_descr(
        core::mem::offset_of!(pyre_object::celldict::ModuleDictStrategy, version),
        8,
        Type::Int,
        false,
    )
});

pub fn module_dict_version_descr() -> DescrRef {
    MODULE_DICT_VERSION_FIELD_DESCR.clone()
}

// These headerless-map-node descrs deliberately retain `parent_descr = None`
// and `index_in_parent = 0`. This is sound only because arg0 of every one of
// their `QUASIIMMUT_FIELD` ops is a freshly minted `const_ref`:
// optimizer.py would otherwise ask for `get_parent_descr()` and
// `get_index()`, but the constant-arg0 `ConstPtrInfo` short-circuit at
// optimizer.py:468-469 skips that path. Therefore these four descrs may appear
// ONLY on `QUASIIMMUT_FIELD`, never on `GetfieldGc*` or `SetfieldGc*`:
// `field_slot_index` falls back to `descr.index()` without a parent, and a
// 0x5000_000X slot would blow out `_fields`. They also stay singleton Arcs for
// the identity reason documented on `TYPE_VERSION_TAG_FIELD_DESCR`.
static TERMINATOR_ALLOW_UNBOXING_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            TERMINATOR_ALLOW_UNBOXING_INDEX,
            core::mem::offset_of!(
                pyre_interpreter::objspace::std::mapdict::Terminator,
                allow_unboxing
            ),
            1,
            Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "Terminator.allow_unboxing".to_string(),
            "allow_unboxing".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn terminator_allow_unboxing_descr() -> DescrRef {
    TERMINATOR_ALLOW_UNBOXING_FIELD_DESCR.clone()
}

static PLAIN_ATTRIBUTE_EVER_MUTATED_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            PLAIN_ATTRIBUTE_EVER_MUTATED_INDEX,
            core::mem::offset_of!(
                pyre_interpreter::objspace::std::mapdict::PlainAttribute,
                ever_mutated
            ),
            1,
            Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "PlainAttribute.ever_mutated".to_string(),
            "ever_mutated".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn plain_attribute_ever_mutated_descr() -> DescrRef {
    PLAIN_ATTRIBUTE_EVER_MUTATED_FIELD_DESCR.clone()
}

/// `nestedscope.py `CellFamily` _immutable_fields_ = ['ever_mutated?']` — the field
/// `Cell.get`'s fold reads before returning a constant cell's contents.
static CELL_FAMILY_EVER_MUTATED_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            CELL_FAMILY_EVER_MUTATED_INDEX,
            core::mem::offset_of!(pyre_object::nestedscope::CellFamily, ever_mutated),
            1,
            Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "CellFamily.ever_mutated".to_string(),
            "ever_mutated".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn cell_family_ever_mutated_descr() -> DescrRef {
    CELL_FAMILY_EVER_MUTATED_FIELD_DESCR.clone()
}

static HOLDER_ATTR_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            HOLDER_ATTR_INDEX,
            core::mem::offset_of!(
                pyre_interpreter::objspace::std::mapdict::CachedAttributeHolder,
                attr
            ),
            // A map node is an opaque pointer-sized Int, including on wasm32.
            core::mem::size_of::<usize>(),
            Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "CachedAttributeHolder.attr".to_string(),
            "attr".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn holder_attr_descr() -> DescrRef {
    HOLDER_ATTR_FIELD_DESCR.clone()
}

// `field_sanity_load` / `bh_getfield_gc_i` only handle power-of-two widths;
// make an `Option<UnboxType>` layout change a compile error instead of a
// silent wrong-width read.
const _: () = assert!(matches!(
    core::mem::size_of::<
        std::cell::Cell<Option<pyre_interpreter::objspace::std::mapdict::UnboxType>>,
    >(),
    1 | 2 | 4 | 8
));

static HOLDER_TYP_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            HOLDER_TYP_INDEX,
            core::mem::offset_of!(
                pyre_interpreter::objspace::std::mapdict::CachedAttributeHolder,
                typ
            ),
            core::mem::size_of::<
                std::cell::Cell<Option<pyre_interpreter::objspace::std::mapdict::UnboxType>>,
            >(),
            Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "CachedAttributeHolder.typ".to_string(),
            "typ".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn holder_typ_descr() -> DescrRef {
    HOLDER_TYP_FIELD_DESCR.clone()
}

/// `vm.py AuditHolder._immutable_fields_ = ['hooks_w?[:]']` — whether any
/// audit hook is installed.
///
/// The declaration is on the hook list itself; the field watched here is the
/// byte `AuditHolder` projects it onto, because a `Box<[W_Root]>` has no
/// target-stable null spelling.  That projection carries the whole datum a
/// reader needs: `vm.py` tests only `hooks_w is None`, and the no-hook
/// fold (`try_walker_specialize_sys_getframe`) pins that read. Once a hook
/// exists the same fold residuals `trigger_audit_events` instead.
///
/// A reserved index and `QUASIIMMUT_FIELD`-only, for the reasons the map-node
/// descrs above document.
static AUDIT_HOLDER_HOOKS_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            AUDIT_HOLDER_HOOKS_INDEX,
            core::mem::offset_of!(pyre_interpreter::module::sys::vm::AuditHolder, hooks_armed),
            1,
            Type::Int,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "AuditHolder.hooks_w".to_string(),
            "hooks_w".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn audit_holder_hooks_descr() -> DescrRef {
    AUDIT_HOLDER_HOOKS_FIELD_DESCR.clone()
}

/// `descriptor.py W_Property._immutable_fields_ = ["w_fget?", "w_fset?",
/// "w_fdel?"]` — the property's getter slot.
///
/// The LOAD_ATTR property fold inlines `fget(obj)` against a descriptor the
/// receiver's class + `_version_tag?` already pin, which makes the descriptor
/// object constant but says nothing about its accessor slots: `__init__` on an
/// installed property replaces them in place and bumps no type's version.  The
/// `?` is what covers the slot, and it costs a `QUASIIMMUT_FIELD` marker plus
/// one `GUARD_NOT_INVALIDATED` per trace rather than a load and a `GUARD_VALUE`
/// per iteration — see [`walker_pin_type_version_tag`](crate::jitcode_dispatch)
/// for why the difference is load-bearing across a residual call.
///
/// A marker only: the fold never loads through the baked descriptor pointer,
/// which is what keeps it clear of the baked-`ConstPtr` hazard that made the
/// inline-call path skip its `Function.code` reads for a constant callable.
static PROPERTY_FGET_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            PROPERTY_FGET_INDEX,
            core::mem::offset_of!(pyre_object::descriptor::W_Property, fget),
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "W_Property.fget".to_string(),
            "fget".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn property_fget_descr() -> DescrRef {
    PROPERTY_FGET_FIELD_DESCR.clone()
}

/// The `w_fset?` twin of [`PROPERTY_FGET_FIELD_DESCR`], read by the STORE_ATTR
/// property fold.
static PROPERTY_FSET_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            PROPERTY_FSET_INDEX,
            core::mem::offset_of!(pyre_object::descriptor::W_Property, fset),
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Unsigned,
            "W_Property.fset".to_string(),
            "fset".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn property_fset_descr() -> DescrRef {
    PROPERTY_FSET_FIELD_DESCR.clone()
}

/// `W_ObjectObject` SizeDescr group (`objectobject.rs`) — the instance
/// layout `[ob_type | w_class | map | storage]`.  Built with a parent SizeDescr
/// (unlike a bare [`make_field_descr`]) so a `getfield_gc` on `map` / `storage`
/// resolves `FieldDescr.get_parent_descr()` in the optimizer's
/// `ensure_ptr_info_arg0` (`optimizer.py`); the LOAD_ATTR fold reads these
/// fields inline, so they must carry the owning struct's SizeDescr.
///
/// `map` is an opaque `Int` word (interned immortal map nodes — not a GC ref,
/// stays off `gc_fielddescrs`); `storage` is a `Ref` block pointer (enters
/// `gc_fielddescrs` so a `setfield_gc` emits the write barrier).
///
/// The inherited header `w_class` is a member of the group — not just the
/// standalone `w_class_descr` — because the instantiation emit
/// (`try_walker_inline_type_call`) builds instances with `NewWithVtable`, and
/// the class a `getfield_gc(w_class)` off such a virtual must answer with is
/// the *stored* one.  Exact `object()` stamps `INSTANCE_TYPE`. Every other
/// carrier stamps `INSTANCE_USER_TYPE` (`typedef.py`
/// `_getusercls(W_ObjectObject)`), whose Python class still varies per
/// instance, so the vtable-derived fallback (`w_class_obj`, which resolves
/// `get_instantiate` to `object` or to null on the user vtable) is wrong
/// here; only a field the virtual actually tracks gives `OptVirtualize` the
/// right answer, and materialization then reproduces the header. The user
/// size descr below shares these field Arcs, so both vtables use one index
/// space.
static W_OBJECT_OBJECT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    let group = build_object_descr_group_with_def_path(
        pyre_object::W_OBJECT_OBJECT_SIZE,
        pyre_object::objectobject::W_OBJECT_OBJECT_GC_TYPE_ID,
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
        &[
            (
                "W_ObjectObject.map",
                core::mem::offset_of!(pyre_object::W_ObjectObject, map),
                // `map` is a `*const MapNode` erased to an opaque Int word, so
                // its width is one machine word — 4 bytes on wasm32, 8 on
                // 64-bit. Hardcoding 8 would read/write past the field on
                // wasm32, folding the adjacent `storage` pointer into the high
                // half of a `guard_value(map)` load (and clobbering it on a
                // `setfield_gc(map)` store).
                core::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ObjectObject.storage",
                core::mem::offset_of!(pyre_object::W_ObjectObject, storage),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_ObjectObject",
        "objectobject::W_ObjectObject",
    );
    // `alloc_instance_object` (`objectobject.rs`) allocates every interpreter
    // instance through the stable, non-moving old-gen allocator, because the
    // instance layer reaches an instance through raw pointers it does not root
    // — `store_attr_caching` holds one across the allocation of the storage
    // block it is about to install.  A JIT-emitted instance has to agree: put
    // it in the nursery and the first minor collection inside such a residual
    // moves it, after which the caller finishes writing `map` / `storage` into
    // the dead pre-move copy and the attribute is lost.
    group.size_descr.set_non_moving(true);
    group
});

/// Size index for `W_ObjectObjectUserDictWeakrefable`. Bit 27 sits inside
/// the size mask and no struct is 128MB, so it cannot collide with a
/// size-derived `SIZE_DESCR_TAG | size` index. Field indices stay the
/// base group's: `set_parent_descr` is not called on the shared Arcs.
const W_OBJECT_OBJECT_USER_SIZE_INDEX: u32 = SIZE_DESCR_TAG | 0x0800_0000;

/// `NewWithVtable` descr for a non-exact `W_ObjectObject` carrier.
/// Same field Arcs, size, and `non_moving` bit as
/// [`W_OBJECT_OBJECT_DESCR_GROUP`]; vtable is `INSTANCE_USER_TYPE` and
/// the GC tid is `W_OBJECT_OBJECT_USER_GC_TYPE_ID`. Registered under
/// `objectobject::W_ObjectObjectUser`, not the exact-object STRUCT key.
static W_OBJECT_OBJECT_USER_SIZE_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    let base = W_OBJECT_OBJECT_DESCR_GROUP.size_descr.clone();
    let fields = base
        .as_size_descr()
        .expect("W_ObjectObject size descr")
        .all_fielddescrs()
        .to_vec();
    let mut sd = majit_ir::descr::SimpleSizeDescr::with_vtable(
        W_OBJECT_OBJECT_USER_SIZE_INDEX,
        pyre_object::W_OBJECT_OBJECT_SIZE,
        pyre_object::objectobject::W_OBJECT_OBJECT_USER_GC_TYPE_ID,
        &pyre_object::pyobject::INSTANCE_USER_TYPE as *const _ as usize,
    );
    let cache_key = majit_ir::descr::path_hash("objectobject::W_ObjectObjectUser");
    sd.set_cache_key(cache_key);
    let sd = sd.with_all_fielddescrs(fields);
    sd.set_non_moving(true);
    let descr = Arc::new(sd) as DescrRef;
    majit_ir::descr_registry::register_keyed_size(
        majit_ir::descr::LLType::Struct(cache_key),
        descr.clone(),
    );
    descr
});

fn build_native_user_mapdict_group(
    size: usize,
    type_id: u32,
    vtable: usize,
    map_offset: usize,
    storage_offset: usize,
    simple_name: &'static str,
    def_path: &'static str,
    tag: u32,
) -> PyreObjectDescrGroup {
    let group = build_object_descr_group_with_field_indices(
        size,
        type_id,
        vtable,
        &[
            (
                "map",
                map_offset,
                core::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "storage",
                storage_offset,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        simple_name,
        def_path,
        &[tag, tag | 1],
    );
    // The annotator can publish the shared `(STRUCT, fieldname)` shells before
    // this runtime group is forced. `get_field_descr` correctly reuses those
    // objects (descr.py), but their placeholder index is `u32::MAX`.
    // Stamp the two shared descriptors after the cache lookup so HeapCache
    // cannot alias map and storage through that placeholder.
    group.field_descrs[0].set_index(tag);
    group.field_descrs[1].set_index(tag | 1);
    group
}

static W_INT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `W_IntObjectUser` embeds `W_IntObject` ahead of the mapdict tail, so
    // `intval` shares `INT_INTVAL_OFFSET` with the exact int. It is a
    // different `(STRUCT, fieldname)` from `int_intval_descr`: a virtual
    // built on this group folds only a getfield of this field. Fields 0 and 1
    // stay map and storage; `mapdict_map_descr` / `mapdict_storage_descr`
    // index those slots.
    //
    // Field 3 is the inherited header, listed the way `W_FloatObjectUser`
    // lists `PyObject.w_class`. `tag_subclass_instance` stores a class the
    // allocation's canonical word does not already hold; without the slot
    // `optimize_setfield_gc` forces a real `NewWithVtable` every iteration.
    let tag = NATIVE_MAPDICT_DESCR_TAG;
    let group = build_object_descr_group_with_field_indices(
        pyre_object::intobject::W_INT_USER_OBJECT_SIZE,
        pyre_object::intobject::W_INT_USER_GC_TYPE_ID,
        &pyre_object::pyobject::INT_USER_TYPE as *const _ as usize,
        &[
            (
                "map",
                std::mem::offset_of!(pyre_object::intobject::W_IntObjectUser, map),
                core::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "storage",
                std::mem::offset_of!(pyre_object::intobject::W_IntObjectUser, storage),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            ("intval", INT_INTVAL_OFFSET, 8, Type::Int, true, true, false),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_IntObjectUser",
        "intobject::W_IntObjectUser",
        &[tag, tag | 1, tag | 2, tag | 3],
    );
    // `get_field_descr` reuses a placeholder shell whose index is `u32::MAX`.
    // Stamp after the cache lookup so HeapCache cannot alias the fields.
    group.field_descrs[0].set_index(tag);
    group.field_descrs[1].set_index(tag | 1);
    group.field_descrs[2].set_index(tag | 2);
    group.field_descrs[3].set_index(tag | 3);
    group
});

static W_UNICODE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::unicodeobject::W_UNICODE_USER_OBJECT_SIZE,
        pyre_object::unicodeobject::W_UNICODE_USER_GC_TYPE_ID,
        &pyre_object::pyobject::STR_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::unicodeobject::W_UnicodeObjectUser, map),
        std::mem::offset_of!(pyre_object::unicodeobject::W_UnicodeObjectUser, storage),
        "W_UnicodeObjectUser",
        "unicodeobject::W_UnicodeObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x10,
    )
});

static W_TUPLE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::tupleobject::W_TUPLE_USER_OBJECT_SIZE,
        pyre_object::tupleobject::W_TUPLE_USER_GC_TYPE_ID,
        &pyre_object::pyobject::TUPLE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::tupleobject::W_TupleObjectUser, map),
        std::mem::offset_of!(pyre_object::tupleobject::W_TupleObjectUser, storage),
        "W_TupleObjectUser",
        "tupleobject::W_TupleObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x20,
    )
});

static W_FLOAT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `W_FloatObjectUser` embeds `W_FloatObject` ahead of the mapdict tail, so
    // `floatval` shares `FLOAT_FLOATVAL_OFFSET` with the exact float. It is a
    // different `(STRUCT, fieldname)` from `float_floatval_descr`: a virtual
    // built on this group folds only a getfield of this field. Fields 0 and 1
    // stay map and storage; `mapdict_map_descr` / `mapdict_storage_descr`
    // index those slots.
    //
    // The spec list ends with `PyObject.w_class`, the same row
    // `W_LIST_DESCR_GROUP` lists. A headered group also prepends the nested
    // `PyObject.w_class` leaf (`heaptracker.py` `all_fielddescrs`), and
    // `class_word_index_in_parent` answers that leading row.
    // `tag_subclass_instance`
    // stores a class the allocation's canonical word does not already hold;
    // without the slot `optimize_setfield_gc` cannot record the store on the
    // virtual and forces a real `NewWithVtable` every iteration. The flags
    // match the other positional header rows. The trace still writes the
    // shared quasi-immutable singleton; this row is only the slot that
    // records that store.
    let tag = NATIVE_MAPDICT_DESCR_TAG | 0x30;
    let group = build_object_descr_group_with_field_indices(
        pyre_object::floatobject::W_FLOAT_USER_OBJECT_SIZE,
        pyre_object::floatobject::W_FLOAT_USER_GC_TYPE_ID,
        &pyre_object::pyobject::FLOAT_USER_TYPE as *const _ as usize,
        &[
            (
                "map",
                std::mem::offset_of!(pyre_object::floatobject::W_FloatObjectUser, map),
                core::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "storage",
                std::mem::offset_of!(pyre_object::floatobject::W_FloatObjectUser, storage),
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "floatval",
                FLOAT_FLOATVAL_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_FloatObjectUser",
        "floatobject::W_FloatObjectUser",
        &[tag, tag | 1, tag | 2, tag | 3],
    );
    // `get_field_descr` reuses a placeholder shell whose index is `u32::MAX`.
    // Stamp after the cache lookup so HeapCache cannot alias the fields.
    group.field_descrs[0].set_index(tag);
    group.field_descrs[1].set_index(tag | 1);
    group.field_descrs[2].set_index(tag | 2);
    group.field_descrs[3].set_index(tag | 3);
    group
});

static W_COMPLEX_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::complexobject::W_COMPLEX_USER_OBJECT_SIZE,
        pyre_object::complexobject::W_COMPLEX_USER_GC_TYPE_ID,
        &pyre_object::pyobject::COMPLEX_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::complexobject::W_ComplexObjectUser, map),
        std::mem::offset_of!(pyre_object::complexobject::W_ComplexObjectUser, storage),
        "W_ComplexObjectUser",
        "complexobject::W_ComplexObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x40,
    )
});

static W_BYTES_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::bytesobject::W_BYTES_USER_OBJECT_SIZE,
        pyre_object::bytesobject::W_BYTES_USER_GC_TYPE_ID,
        &pyre_object::bytesobject::BYTES_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::bytesobject::W_BytesObjectUser, map),
        std::mem::offset_of!(pyre_object::bytesobject::W_BytesObjectUser, storage),
        "W_BytesObjectUser",
        "bytesobject::W_BytesObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x50,
    )
});

static W_BYTEARRAY_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::bytearrayobject::W_BYTEARRAY_USER_OBJECT_SIZE,
        pyre_object::bytearrayobject::W_BYTEARRAY_USER_GC_TYPE_ID,
        &pyre_object::bytearrayobject::BYTEARRAY_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::bytearrayobject::W_BytearrayObjectUser, map),
        std::mem::offset_of!(pyre_object::bytearrayobject::W_BytearrayObjectUser, storage),
        "W_BytearrayObjectUser",
        "bytearrayobject::W_BytearrayObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x60,
    )
});

static W_LIST_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::listobject::W_LIST_USER_OBJECT_SIZE,
        pyre_object::listobject::W_LIST_USER_GC_TYPE_ID,
        &pyre_object::pyobject::LIST_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::listobject::W_ListObjectUser, map),
        std::mem::offset_of!(pyre_object::listobject::W_ListObjectUser, storage),
        "W_ListObjectUser",
        "listobject::W_ListObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x70,
    )
});

/// `W_SetObjectUser` (`typedef.py` `_getusercls`). `set` and `frozenset`
/// share this layout and this tid. The vtable is `SET_USER_TYPE`; a
/// `frozenset` subclass reads `map`/`storage` through the same field descrs,
/// the same sharing [`W_SET_DESCR_GROUP`] gives the base payloads.
static W_SET_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::setobject::W_SET_USER_OBJECT_SIZE,
        pyre_object::setobject::W_SET_USER_GC_TYPE_ID,
        &pyre_object::setobject::SET_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::setobject::W_SetObjectUser, map),
        std::mem::offset_of!(pyre_object::setobject::W_SetObjectUser, storage),
        "W_SetObjectUser",
        "setobject::W_SetObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x80,
    )
});

/// `W_ArrayUser` (`typedef.py` `_getusercls`).
static W_ARRAY_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::interp_array::W_ARRAY_USER_OBJECT_SIZE,
        pyre_object::interp_array::W_ARRAY_USER_GC_TYPE_ID,
        &pyre_object::interp_array::ARRAY_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_array::W_ArrayUser, map),
        std::mem::offset_of!(pyre_object::interp_array::W_ArrayUser, storage),
        "W_ArrayUser",
        "interp_array::W_ArrayUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x90,
    )
});

/// `W_WeakrefUser` (`typedef.py` `_getusercls`).
static W_WEAKREF_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::weakref::W_WEAKREF_USER_OBJECT_SIZE,
        pyre_object::weakref::W_WEAKREF_USER_GC_TYPE_ID,
        &pyre_object::weakref::WEAKREF_LAYOUT_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::weakref::W_WeakrefUser, map),
        std::mem::offset_of!(pyre_object::weakref::W_WeakrefUser, storage),
        "W_WeakrefUser",
        "weakref::W_WeakrefUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xA0,
    )
});

static W_ENUMERATE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::functional::W_EnumerateUser>(),
        pyre_object::functional::W_ENUMERATE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::functional::ENUMERATE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::functional::W_EnumerateUser, map),
        std::mem::offset_of!(pyre_object::functional::W_EnumerateUser, storage),
        "W_EnumerateUser",
        "functional::W_EnumerateUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xB0,
    )
});

static W_MAP_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::functional::W_MapUser>(),
        pyre_object::functional::W_MAP_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::functional::MAP_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::functional::W_MapUser, map),
        std::mem::offset_of!(pyre_object::functional::W_MapUser, storage),
        "W_MapUser",
        "functional::W_MapUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xC0,
    )
});

static W_FILTER_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::functional::W_FilterUser>(),
        pyre_object::functional::W_FILTER_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::functional::FILTER_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::functional::W_FilterUser, map),
        std::mem::offset_of!(pyre_object::functional::W_FilterUser, storage),
        "W_FilterUser",
        "functional::W_FilterUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xD0,
    )
});

static W_ZIP_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::functional::W_ZipUser>(),
        pyre_object::functional::W_ZIP_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::functional::ZIP_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::functional::W_ZipUser, map),
        std::mem::offset_of!(pyre_object::functional::W_ZipUser, storage),
        "W_ZipUser",
        "functional::W_ZipUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xE0,
    )
});

static W_REVERSED_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::functional::W_ReversedIteratorUser>(),
        pyre_object::functional::W_REVERSED_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::functional::REVERSED_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::functional::W_ReversedIteratorUser, map),
        std::mem::offset_of!(pyre_object::functional::W_ReversedIteratorUser, storage),
        "W_ReversedIteratorUser",
        "functional::W_ReversedIteratorUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF0,
    )
});

static W_SUPER_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::descriptor::W_SuperUser>(),
        pyre_object::descriptor::W_SUPER_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::descriptor::SUPER_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::descriptor::W_SuperUser, map),
        std::mem::offset_of!(pyre_object::descriptor::W_SuperUser, storage),
        "W_SuperUser",
        "descriptor::W_SuperUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x100,
    )
});

static W_PROPERTY_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::descriptor::W_PropertyUser>(),
        pyre_object::descriptor::W_PROPERTY_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::descriptor::PROPERTY_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::descriptor::W_PropertyUser, map),
        std::mem::offset_of!(pyre_object::descriptor::W_PropertyUser, storage),
        "W_PropertyUser",
        "descriptor::W_PropertyUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x110,
    )
});

static W_COUNT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_CountUser>(),
        pyre_object::interp_itertools::W_COUNT_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::COUNT_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_CountUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_CountUser, storage),
        "W_CountUser",
        "interp_itertools::W_CountUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x120,
    )
});

static W_REPEAT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_RepeatUser>(),
        pyre_object::interp_itertools::W_REPEAT_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::REPEAT_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_RepeatUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_RepeatUser, storage),
        "W_RepeatUser",
        "interp_itertools::W_RepeatUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x130,
    )
});

static W_TAKEWHILE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_TakeWhileUser>(),
        pyre_object::interp_itertools::W_TAKEWHILE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::TAKEWHILE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_TakeWhileUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_TakeWhileUser, storage),
        "W_TakeWhileUser",
        "interp_itertools::W_TakeWhileUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x140,
    )
});

static W_DROPWHILE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_DropWhileUser>(),
        pyre_object::interp_itertools::W_DROPWHILE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::DROPWHILE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_DropWhileUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_DropWhileUser, storage),
        "W_DropWhileUser",
        "interp_itertools::W_DropWhileUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x150,
    )
});

static W_FILTERFALSE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_FilterFalseUser>(),
        pyre_object::interp_itertools::W_FILTERFALSE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::FILTERFALSE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_FilterFalseUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_FilterFalseUser, storage),
        "W_FilterFalseUser",
        "interp_itertools::W_FilterFalseUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x160,
    )
});

static W_ISLICE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_ISliceUser>(),
        pyre_object::interp_itertools::W_ISLICE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::ISLICE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_ISliceUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_ISliceUser, storage),
        "W_ISliceUser",
        "interp_itertools::W_ISliceUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x170,
    )
});

static W_BATCHED_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_BatchedUser>(),
        pyre_object::interp_itertools::W_BATCHED_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::BATCHED_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_BatchedUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_BatchedUser, storage),
        "W_BatchedUser",
        "interp_itertools::W_BatchedUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x180,
    )
});

static W_PRODUCT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_ProductUser>(),
        pyre_object::interp_itertools::W_PRODUCT_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::PRODUCT_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_ProductUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_ProductUser, storage),
        "W_ProductUser",
        "interp_itertools::W_ProductUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x190,
    )
});

static W_COMBINATIONS_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_CombinationsUser>(),
        pyre_object::interp_itertools::W_COMBINATIONS_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::COMBINATIONS_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_CombinationsUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_CombinationsUser, storage),
        "W_CombinationsUser",
        "interp_itertools::W_CombinationsUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x1A0,
    )
});

static W_COMBINATIONS_WITH_REPLACEMENT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> =
    LazyLock::new(|| {
        build_native_user_mapdict_group(
            std::mem::size_of::<pyre_object::interp_itertools::W_CombinationsWithReplacementUser>(),
            pyre_object::interp_itertools::W_COMBINATIONS_WITH_REPLACEMENT_USER_GC_TYPE_ID_CELL
                .get(),
            &pyre_object::interp_itertools::COMBINATIONS_WITH_REPLACEMENT_USER_TYPE as *const _
                as usize,
            std::mem::offset_of!(
                pyre_object::interp_itertools::W_CombinationsWithReplacementUser,
                map
            ),
            std::mem::offset_of!(
                pyre_object::interp_itertools::W_CombinationsWithReplacementUser,
                storage
            ),
            "W_CombinationsWithReplacementUser",
            "interp_itertools::W_CombinationsWithReplacementUser",
            NATIVE_MAPDICT_DESCR_TAG | 0x1B0,
        )
    });

static W_PERMUTATIONS_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_PermutationsUser>(),
        pyre_object::interp_itertools::W_PERMUTATIONS_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::PERMUTATIONS_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_PermutationsUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_PermutationsUser, storage),
        "W_PermutationsUser",
        "interp_itertools::W_PermutationsUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x1C0,
    )
});

static W_GROUPBY_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_GroupByUser>(),
        pyre_object::interp_itertools::W_GROUPBY_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::GROUPBY_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_GroupByUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_GroupByUser, storage),
        "W_GroupByUser",
        "interp_itertools::W_GroupByUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x1D0,
    )
});

static W_COMPRESS_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_CompressUser>(),
        pyre_object::interp_itertools::W_COMPRESS_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::COMPRESS_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_CompressUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_CompressUser, storage),
        "W_CompressUser",
        "interp_itertools::W_CompressUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x1E0,
    )
});

static W_STARMAP_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_StarMapUser>(),
        pyre_object::interp_itertools::W_STARMAP_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::STARMAP_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_StarMapUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_StarMapUser, storage),
        "W_StarMapUser",
        "interp_itertools::W_StarMapUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x1F0,
    )
});

static W_ACCUMULATE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_AccumulateUser>(),
        pyre_object::interp_itertools::W_ACCUMULATE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::ACCUMULATE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_AccumulateUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_AccumulateUser, storage),
        "W_AccumulateUser",
        "interp_itertools::W_AccumulateUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x200,
    )
});

static W_ZIP_LONGEST_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_ZipLongestUser>(),
        pyre_object::interp_itertools::W_ZIP_LONGEST_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::ZIP_LONGEST_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_ZipLongestUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_ZipLongestUser, storage),
        "W_ZipLongestUser",
        "interp_itertools::W_ZipLongestUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x210,
    )
});

static W_PAIRWISE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_PairwiseUser>(),
        pyre_object::interp_itertools::W_PAIRWISE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::PAIRWISE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_PairwiseUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_PairwiseUser, storage),
        "W_PairwiseUser",
        "interp_itertools::W_PairwiseUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x220,
    )
});

static W_CYCLE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_CycleUser>(),
        pyre_object::interp_itertools::W_CYCLE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::CYCLE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_CycleUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_CycleUser, storage),
        "W_CycleUser",
        "interp_itertools::W_CycleUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x230,
    )
});

static W_CHAIN_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::interp_itertools::W_ChainUser>(),
        pyre_object::interp_itertools::W_CHAIN_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::interp_itertools::CHAIN_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::interp_itertools::W_ChainUser, map),
        std::mem::offset_of!(pyre_object::interp_itertools::W_ChainUser, storage),
        "W_ChainUser",
        "interp_itertools::W_ChainUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x240,
    )
});

static W_BUFFERABLE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::W_BufferableUser>(),
        pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::W_BUFFERABLE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::BUFFERABLE_USER_TYPE
            as *const _ as usize,
        std::mem::offset_of!(
            pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::W_BufferableUser,
            map
        ),
        std::mem::offset_of!(
            pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::W_BufferableUser,
            storage
        ),
        "W_BufferableUser",
        "module::__pypy__::interp_buffer::bufferable_impl::W_BufferableUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x250,
    )
});

static W_DEQUE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_collections::W_DequeUser>(),
        pyre_interpreter::module::_collections::W_DEQUE_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_collections::DEQUE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_collections::W_DequeUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_collections::W_DequeUser, storage),
        "W_DequeUser",
        "module::_collections::W_DequeUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF00,
    )
});

static W_STRUCT_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::r#struct::W_StructUser>(),
        pyre_interpreter::module::r#struct::W_STRUCT_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::r#struct::STRUCT_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::r#struct::W_StructUser, map),
        std::mem::offset_of!(pyre_interpreter::module::r#struct::W_StructUser, storage),
        "W_StructUser",
        "module::struct::W_StructUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF10,
    )
});

static W_GENERIC_ALIAS_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::_pypy_generic_alias::GenericAliasUser>(),
        pyre_object::_pypy_generic_alias::W_GENERIC_ALIAS_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::_pypy_generic_alias::GENERIC_ALIAS_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::_pypy_generic_alias::GenericAliasUser, map),
        std::mem::offset_of!(pyre_object::_pypy_generic_alias::GenericAliasUser, storage),
        "GenericAliasUser",
        "_pypy_generic_alias::GenericAliasUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF20,
    )
});

static W_LONG_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::longobject::W_LONG_USER_OBJECT_SIZE,
        pyre_object::longobject::W_LONG_USER_GC_TYPE_ID,
        &pyre_object::pyobject::LONG_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::longobject::W_LongObjectUser, map),
        std::mem::offset_of!(pyre_object::longobject::W_LongObjectUser, storage),
        "W_LongObjectUser",
        "longobject::W_LongObjectUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF30,
    )
});

static W_STATICMETHOD_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::function::StaticMethodUser>(),
        pyre_object::function::W_STATICMETHOD_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::function::STATICMETHOD_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::function::StaticMethodUser, map),
        std::mem::offset_of!(pyre_object::function::StaticMethodUser, storage),
        "StaticMethodUser",
        "function::StaticMethodUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF40,
    )
});

static W_CLASSMETHOD_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_object::function::ClassMethodUser>(),
        pyre_object::function::W_CLASSMETHOD_USER_GC_TYPE_ID_CELL.get(),
        &pyre_object::function::CLASSMETHOD_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::function::ClassMethodUser, map),
        std::mem::offset_of!(pyre_object::function::ClassMethodUser, storage),
        "ClassMethodUser",
        "function::ClassMethodUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF50,
    )
});

static W_MODULE_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        pyre_object::module::W_MODULE_USER_OBJECT_SIZE,
        pyre_object::module::W_MODULE_USER_GC_TYPE_ID,
        &pyre_object::pyobject::MODULE_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_object::module::ModuleUser, map),
        std::mem::offset_of!(pyre_object::module::ModuleUser, storage),
        "ModuleUser",
        "module::ModuleUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF60,
    )
});

static W_LOCAL_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::thread::W_LocalUser>(),
        pyre_interpreter::module::thread::W_LOCAL_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::thread::LOCAL_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::thread::W_LocalUser, map),
        std::mem::offset_of!(pyre_interpreter::module::thread::W_LocalUser, storage),
        "W_LocalUser",
        "module::thread::W_LocalUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF70,
    )
});

static W_BYTESIO_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_BytesIOUser>(),
        pyre_interpreter::module::_io::W_BYTESIO_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::BYTESIO_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BytesIOUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BytesIOUser, storage),
        "W_BytesIOUser",
        "module::_io::W_BytesIOUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF80,
    )
});

static W_STRINGIO_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_StringIOUser>(),
        pyre_interpreter::module::_io::W_STRINGIO_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::STRINGIO_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_StringIOUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_StringIOUser, storage),
        "W_StringIOUser",
        "module::_io::W_StringIOUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xF90,
    )
});

static W_BUFFEREDREADER_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_BufferedReaderUser>(),
        pyre_interpreter::module::_io::W_BUFFEREDREADER_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::BUFFEREDREADER_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedReaderUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedReaderUser, storage),
        "W_BufferedReaderUser",
        "module::_io::W_BufferedReaderUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xFA0,
    )
});

static W_BUFFEREDWRITER_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_BufferedWriterUser>(),
        pyre_interpreter::module::_io::W_BUFFEREDWRITER_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::BUFFEREDWRITER_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedWriterUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedWriterUser, storage),
        "W_BufferedWriterUser",
        "module::_io::W_BufferedWriterUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xFB0,
    )
});

static W_BUFFEREDRWPAIR_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_BufferedRWPairUser>(),
        pyre_interpreter::module::_io::W_BUFFEREDRWPAIR_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::BUFFEREDRWPAIR_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedRWPairUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedRWPairUser, storage),
        "W_BufferedRWPairUser",
        "module::_io::W_BufferedRWPairUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xFC0,
    )
});

static W_BUFFEREDRANDOM_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_BufferedRandomUser>(),
        pyre_interpreter::module::_io::W_BUFFEREDRANDOM_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::BUFFEREDRANDOM_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedRandomUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_BufferedRandomUser, storage),
        "W_BufferedRandomUser",
        "module::_io::W_BufferedRandomUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xFD0,
    )
});

static W_FILEIO_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_FileIOUser>(),
        pyre_interpreter::module::_io::W_FILEIO_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::FILEIO_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_FileIOUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_FileIOUser, storage),
        "W_FileIOUser",
        "module::_io::W_FileIOUser",
        NATIVE_MAPDICT_DESCR_TAG | 0x1000,
    )
});

static W_TEXTIOWRAPPER_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_TextIOWrapperUser>(),
        pyre_interpreter::module::_io::W_TEXTIOWRAPPER_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::TEXTIOWRAPPER_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_TextIOWrapperUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_TextIOWrapperUser, storage),
        "W_TextIOWrapperUser",
        "module::_io::W_TextIOWrapperUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xFE0,
    )
});

unsafe fn winconsoleio_user_mapdict_descr(
    obj: pyre_object::PyObjectRef,
    field: usize,
) -> Option<DescrRef> {
    #[cfg(all(windows, not(feature = "sandbox")))]
    {
        if unsafe {
            pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::WINCONSOLEIO_USER_TYPE)
        } {
            return Some(field_descr_from_group(
                &W_WINCONSOLEIO_USER_DESCR_GROUP,
                field,
            ));
        }
    }
    let _ = (obj, field);
    None
}

#[cfg(all(windows, not(feature = "sandbox")))]
static W_WINCONSOLEIO_USER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_native_user_mapdict_group(
        std::mem::size_of::<pyre_interpreter::module::_io::W_WinConsoleIOUser>(),
        pyre_interpreter::module::_io::W_WINCONSOLEIO_USER_GC_TYPE_ID_CELL.get(),
        &pyre_interpreter::module::_io::WINCONSOLEIO_USER_TYPE as *const _ as usize,
        std::mem::offset_of!(pyre_interpreter::module::_io::W_WinConsoleIOUser, map),
        std::mem::offset_of!(pyre_interpreter::module::_io::W_WinConsoleIOUser, storage),
        "W_WinConsoleIOUser",
        "module::_io::W_WinConsoleIOUser",
        NATIVE_MAPDICT_DESCR_TAG | 0xFF0,
    )
});

struct ModuleUserLayoutGroup {
    /// `ob_type` of a `typedef.py` `_getusercls` instance.
    pytype: usize,
    /// Crate-stripped `struct_path`, the `DECLARED_GROUPS` def-path spelling.
    def_path: &'static str,
    group: PyreObjectDescrGroup,
}

/// `MapdictStorageMixin` groups for module user layouts (`typedef.py`
/// `_getusercls`).
///
/// Built from `for_each_class_descriptor`, keeping `mapdict_user_layout`
/// descriptors whose `struct_path` starts with `pyre_module`. Sorted by
/// `struct_path`: `linkme` order differs between the build-script process
/// and the runtime binary, and the field index has to agree across that
/// boundary. With no such descriptors linked the vector is empty.
static MODULE_USER_LAYOUT_DESCR_GROUPS: LazyLock<Vec<ModuleUserLayoutGroup>> =
    LazyLock::new(|| {
        const PREFIX: &str = "pyre_module::";
        let mut descrs: Vec<&'static pyre_object::lltype::PyreClassDescriptor> = Vec::new();
        pyre_object::lltype::for_each_class_descriptor(|descr| {
            if descr.mapdict_user_layout && descr.struct_path.starts_with(PREFIX) {
                descrs.push(descr);
            }
        });
        descrs.sort_by(|left, right| left.struct_path.cmp(right.struct_path));
        descrs
            .into_iter()
            .enumerate()
            .map(|(index, descr)| {
                let map_off = unsafe { (*descr.pytype_ptr).mapdict_offset };
                assert!(map_off != 0, "{}", descr.struct_path);
                let def_path = &descr.struct_path[PREFIX.len()..];
                let simple_name = match def_path.rsplit_once("::") {
                    Some((_, name)) => name,
                    None => def_path,
                };
                let tag = NATIVE_MAPDICT_DESCR_TAG | (0x260 + 0x10 * (index as u32));
                let group = build_native_user_mapdict_group(
                    descr.object_size,
                    descr.gc_type_id.get(),
                    descr.pytype_ptr as usize,
                    map_off,
                    map_off + WORD,
                    simple_name,
                    def_path,
                    tag,
                );
                ModuleUserLayoutGroup {
                    pytype: descr.pytype_ptr as usize,
                    def_path,
                    group,
                }
            })
            .collect()
    });

/// Map or storage descr for a module user layout. Exact instances carry the
/// base typeptr and return `None`. `field` is 0 for `map` and 1 for `storage`.
///
/// # Safety
/// `obj` must be a live object.
unsafe fn module_user_layout_mapdict_descr(
    obj: pyre_object::PyObjectRef,
    field: usize,
) -> Option<DescrRef> {
    let ob_type = unsafe { (*obj).ob_type } as usize;
    for entry in MODULE_USER_LAYOUT_DESCR_GROUPS.iter() {
        if entry.pytype == ob_type {
            return Some(field_descr_from_group(&entry.group, field));
        }
    }
    None
}

/// `W_ObjectObject.map` (`objectobject.rs`) — the instance shape word,
/// `self.map` of PyPy's `MapdictStorageMixin` (`mapdict.py`). Read as an
/// `Int` word so the LOAD_ATTR fast path can `guard_value` it to a constant map
/// (`jit.promote(self.map)`, mapdict.py), after which the resolved
/// `storageindex` is a green constant. The map nodes are interned + immortal,
/// so the address is a stable identity and the guard need not treat it as a GC
/// ref — the field is declared `usize` so the translator lowers its reads to
/// the same `Int` kind rather than the `Ref` a pointer type would imply.
pub fn object_map_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_OBJECT_DESCR_GROUP, 0)
}

/// `W_ObjectObject.storage` (`objectobject.rs`) — `self.storage` of
/// `MapdictStorageMixin` (`mapdict.py`), a `Ptr(GcArray(OBJECTPTR))` block
/// of attribute values. Read as a `Ref` (the block pointer) so the LOAD_ATTR
/// fast path can then `getarrayitem_gc_r` the value at the green-constant
/// `storageindex` (`mapdict.py` `_mapdict_read_storage`), mirroring
/// `list_items_descr` → `pyobject_gcarray_descr`. Mutable: STORE_ATTR grows /
/// replaces the block.
pub fn object_storage_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_OBJECT_DESCR_GROUP, 1)
}

/// Map field for a receiver already proven to carry a translated mapdict
/// layout.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn mapdict_map_descr(obj: pyre_object::PyObjectRef) -> DescrRef {
    if unsafe { pyre_object::is_int(obj) } {
        field_descr_from_group(&W_INT_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::is_float(obj) } {
        field_descr_from_group(&W_FLOAT_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::bytesobject::is_bytes(obj) } {
        field_descr_from_group(&W_BYTES_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::bytearrayobject::is_bytearray(obj) } {
        field_descr_from_group(&W_BYTEARRAY_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::is_list(obj) } {
        field_descr_from_group(&W_LIST_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::is_set_or_frozenset(obj) } {
        field_descr_from_group(&W_SET_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_array::is_array(obj) } {
        field_descr_from_group(&W_ARRAY_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        field_descr_from_group(&W_WEAKREF_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::is_complex(obj) } {
        field_descr_from_group(&W_COMPLEX_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::is_str(obj) } {
        field_descr_from_group(&W_UNICODE_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::is_tuple(obj)
            && !pyre_object::specialisedtupleobject::is_specialised_tuple(obj)
    } {
        field_descr_from_group(&W_TUPLE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::functional::is_enumerate(obj) } {
        field_descr_from_group(&W_ENUMERATE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::functional::is_map(obj) } {
        field_descr_from_group(&W_MAP_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::functional::is_filter(obj) } {
        field_descr_from_group(&W_FILTER_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::functional::is_zip(obj) } {
        field_descr_from_group(&W_ZIP_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::functional::is_reversed(obj) } {
        field_descr_from_group(&W_REVERSED_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::descriptor::is_super(obj) } {
        field_descr_from_group(&W_SUPER_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::descriptor::is_property(obj) } {
        field_descr_from_group(&W_PROPERTY_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_count(obj) } {
        field_descr_from_group(&W_COUNT_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_repeat(obj) } {
        field_descr_from_group(&W_REPEAT_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_takewhile(obj) } {
        field_descr_from_group(&W_TAKEWHILE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_dropwhile(obj) } {
        field_descr_from_group(&W_DROPWHILE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_filterfalse(obj) } {
        field_descr_from_group(&W_FILTERFALSE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_islice(obj) } {
        field_descr_from_group(&W_ISLICE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_batched(obj) } {
        field_descr_from_group(&W_BATCHED_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_product(obj) } {
        field_descr_from_group(&W_PRODUCT_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_combinations(obj) } {
        field_descr_from_group(&W_COMBINATIONS_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_combinations_with_replacement(obj) } {
        field_descr_from_group(&W_COMBINATIONS_WITH_REPLACEMENT_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_permutations(obj) } {
        field_descr_from_group(&W_PERMUTATIONS_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_groupby(obj) } {
        field_descr_from_group(&W_GROUPBY_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_compress(obj) } {
        field_descr_from_group(&W_COMPRESS_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_starmap(obj) } {
        field_descr_from_group(&W_STARMAP_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_accumulate(obj) } {
        field_descr_from_group(&W_ACCUMULATE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_zip_longest(obj) } {
        field_descr_from_group(&W_ZIP_LONGEST_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_pairwise(obj) } {
        field_descr_from_group(&W_PAIRWISE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_cycle(obj) } {
        field_descr_from_group(&W_CYCLE_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::interp_itertools::is_chain(obj) } {
        field_descr_from_group(&W_CHAIN_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::BUFFERABLE_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFERABLE_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_collections::DEQUE_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_DEQUE_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::r#struct::STRUCT_USER_TYPE)
    } {
        field_descr_from_group(&W_STRUCT_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_object::_pypy_generic_alias::GENERIC_ALIAS_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_GENERIC_ALIAS_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::is_long(obj) } {
        field_descr_from_group(&W_LONG_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_object::function::STATICMETHOD_USER_TYPE)
    } {
        field_descr_from_group(&W_STATICMETHOD_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_object::function::CLASSMETHOD_USER_TYPE)
    } {
        field_descr_from_group(&W_CLASSMETHOD_USER_DESCR_GROUP, 0)
    } else if unsafe { pyre_object::py_type_check(obj, &pyre_object::pyobject::MODULE_USER_TYPE) } {
        field_descr_from_group(&W_MODULE_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::thread::LOCAL_USER_TYPE)
    } {
        field_descr_from_group(&W_LOCAL_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::BYTESIO_USER_TYPE)
    } {
        field_descr_from_group(&W_BYTESIO_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::STRINGIO_USER_TYPE)
    } {
        field_descr_from_group(&W_STRINGIO_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDREADER_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDREADER_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDWRITER_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDWRITER_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDRWPAIR_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDRWPAIR_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDRANDOM_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDRANDOM_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::TEXTIOWRAPPER_USER_TYPE)
    } {
        field_descr_from_group(&W_TEXTIOWRAPPER_USER_DESCR_GROUP, 0)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::FILEIO_USER_TYPE)
    } {
        field_descr_from_group(&W_FILEIO_USER_DESCR_GROUP, 0)
    } else if let Some(descr) = unsafe { winconsoleio_user_mapdict_descr(obj, 0) } {
        descr
    } else if let Some(descr) = unsafe { module_user_layout_mapdict_descr(obj, 0) } {
        descr
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_object::interp_exceptions::BASE_EXCEPTION_USER_TYPE,
        )
    } {
        exception_user_mapdict_descr(false, EXC_USER_MAP_OFFSET)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_object::interp_exceptions::EXCEPTION_EXTENDED_USER_TYPE,
        )
    } {
        exception_user_mapdict_descr(true, EXC_EXTENDED_USER_MAP_OFFSET)
    } else {
        object_map_descr()
    }
}

/// Storage-field sibling of [`mapdict_map_descr`].
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn mapdict_storage_descr(obj: pyre_object::PyObjectRef) -> DescrRef {
    if unsafe { pyre_object::is_int(obj) } {
        field_descr_from_group(&W_INT_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::is_float(obj) } {
        field_descr_from_group(&W_FLOAT_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::bytesobject::is_bytes(obj) } {
        field_descr_from_group(&W_BYTES_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::bytearrayobject::is_bytearray(obj) } {
        field_descr_from_group(&W_BYTEARRAY_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::is_list(obj) } {
        field_descr_from_group(&W_LIST_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::is_set_or_frozenset(obj) } {
        field_descr_from_group(&W_SET_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_array::is_array(obj) } {
        field_descr_from_group(&W_ARRAY_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::weakref::is_typed_weakref(obj) } {
        field_descr_from_group(&W_WEAKREF_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::is_complex(obj) } {
        field_descr_from_group(&W_COMPLEX_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::is_str(obj) } {
        field_descr_from_group(&W_UNICODE_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::is_tuple(obj)
            && !pyre_object::specialisedtupleobject::is_specialised_tuple(obj)
    } {
        field_descr_from_group(&W_TUPLE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::functional::is_enumerate(obj) } {
        field_descr_from_group(&W_ENUMERATE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::functional::is_map(obj) } {
        field_descr_from_group(&W_MAP_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::functional::is_filter(obj) } {
        field_descr_from_group(&W_FILTER_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::functional::is_zip(obj) } {
        field_descr_from_group(&W_ZIP_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::functional::is_reversed(obj) } {
        field_descr_from_group(&W_REVERSED_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::descriptor::is_super(obj) } {
        field_descr_from_group(&W_SUPER_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::descriptor::is_property(obj) } {
        field_descr_from_group(&W_PROPERTY_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_count(obj) } {
        field_descr_from_group(&W_COUNT_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_repeat(obj) } {
        field_descr_from_group(&W_REPEAT_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_takewhile(obj) } {
        field_descr_from_group(&W_TAKEWHILE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_dropwhile(obj) } {
        field_descr_from_group(&W_DROPWHILE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_filterfalse(obj) } {
        field_descr_from_group(&W_FILTERFALSE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_islice(obj) } {
        field_descr_from_group(&W_ISLICE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_batched(obj) } {
        field_descr_from_group(&W_BATCHED_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_product(obj) } {
        field_descr_from_group(&W_PRODUCT_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_combinations(obj) } {
        field_descr_from_group(&W_COMBINATIONS_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_combinations_with_replacement(obj) } {
        field_descr_from_group(&W_COMBINATIONS_WITH_REPLACEMENT_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_permutations(obj) } {
        field_descr_from_group(&W_PERMUTATIONS_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_groupby(obj) } {
        field_descr_from_group(&W_GROUPBY_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_compress(obj) } {
        field_descr_from_group(&W_COMPRESS_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_starmap(obj) } {
        field_descr_from_group(&W_STARMAP_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_accumulate(obj) } {
        field_descr_from_group(&W_ACCUMULATE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_zip_longest(obj) } {
        field_descr_from_group(&W_ZIP_LONGEST_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_pairwise(obj) } {
        field_descr_from_group(&W_PAIRWISE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_cycle(obj) } {
        field_descr_from_group(&W_CYCLE_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::interp_itertools::is_chain(obj) } {
        field_descr_from_group(&W_CHAIN_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::__pypy__::interp_buffer::bufferable_impl::BUFFERABLE_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFERABLE_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_collections::DEQUE_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_DEQUE_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::r#struct::STRUCT_USER_TYPE)
    } {
        field_descr_from_group(&W_STRUCT_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_object::_pypy_generic_alias::GENERIC_ALIAS_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_GENERIC_ALIAS_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::is_long(obj) } {
        field_descr_from_group(&W_LONG_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_object::function::STATICMETHOD_USER_TYPE)
    } {
        field_descr_from_group(&W_STATICMETHOD_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_object::function::CLASSMETHOD_USER_TYPE)
    } {
        field_descr_from_group(&W_CLASSMETHOD_USER_DESCR_GROUP, 1)
    } else if unsafe { pyre_object::py_type_check(obj, &pyre_object::pyobject::MODULE_USER_TYPE) } {
        field_descr_from_group(&W_MODULE_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::thread::LOCAL_USER_TYPE)
    } {
        field_descr_from_group(&W_LOCAL_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::BYTESIO_USER_TYPE)
    } {
        field_descr_from_group(&W_BYTESIO_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::STRINGIO_USER_TYPE)
    } {
        field_descr_from_group(&W_STRINGIO_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDREADER_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDREADER_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDWRITER_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDWRITER_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDRWPAIR_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDRWPAIR_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_interpreter::module::_io::BUFFEREDRANDOM_USER_TYPE,
        )
    } {
        field_descr_from_group(&W_BUFFEREDRANDOM_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::TEXTIOWRAPPER_USER_TYPE)
    } {
        field_descr_from_group(&W_TEXTIOWRAPPER_USER_DESCR_GROUP, 1)
    } else if unsafe {
        pyre_object::py_type_check(obj, &pyre_interpreter::module::_io::FILEIO_USER_TYPE)
    } {
        field_descr_from_group(&W_FILEIO_USER_DESCR_GROUP, 1)
    } else if let Some(descr) = unsafe { winconsoleio_user_mapdict_descr(obj, 1) } {
        descr
    } else if let Some(descr) = unsafe { module_user_layout_mapdict_descr(obj, 1) } {
        descr
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_object::interp_exceptions::BASE_EXCEPTION_USER_TYPE,
        )
    } {
        exception_user_mapdict_descr(false, EXC_USER_STORAGE_OFFSET)
    } else if unsafe {
        pyre_object::py_type_check(
            obj,
            &pyre_object::interp_exceptions::EXCEPTION_EXTENDED_USER_TYPE,
        )
    } {
        exception_user_mapdict_descr(true, EXC_EXTENDED_USER_STORAGE_OFFSET)
    } else {
        object_storage_descr()
    }
}

/// The instance's own `PyObject.w_class` — the Python class `w_instance_new`
/// stamps into the header. Kept in the instance group (not the standalone
/// `w_class_descr`) so the instantiation emit's store is a virtual field of
/// the same size descr; see [`W_OBJECT_OBJECT_DESCR_GROUP`].
pub fn object_header_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_OBJECT_DESCR_GROUP, 2)
}

/// Size descriptor for an exact `object()` allocation via `NewWithVtable`
/// (vtable = `&INSTANCE_TYPE`); the header `w_class` and `map` are
/// `SetfieldGc`'d after, and `storage` stays at the allocator's zero (the
/// `_mapdict_init_empty` `storage = None` state).
pub fn w_object_object_size_descr() -> DescrRef {
    W_OBJECT_OBJECT_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for every other `W_ObjectObject` carrier
/// (`typedef.py` `_getusercls(W_ObjectObject)`). Field descrs are the
/// exact-object group's Arcs.
pub fn w_object_object_user_size_descr() -> DescrRef {
    W_OBJECT_OBJECT_USER_SIZE_DESCR.clone()
}

/// rlist.py:116 `l.length` — live length of a list under the Object
/// strategy. Under Integer/Float/Bytes strategies this field is 0 and
/// consumers must dispatch on `list.strategy` first.
pub fn list_length_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 0)
}

/// rlist.py:116 `l.items: Ptr(GcArray(OBJECTPTR))` — pointer to the
/// `ItemsBlock` GcArray body. Callers that need items[i] must combine
/// with the `PY_OBJECT_ARRAY` array descr (item_size=8, Ref,
/// base_size=`ITEMS_BLOCK_ITEMS_OFFSET`); callers that need capacity
/// must issue `ArraylenGc` against the same array descr.
pub fn list_items_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 1)
}

pub fn list_strategy_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 2)
}

pub fn list_int_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 3)
}

pub fn list_float_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 4)
}

/// `Ptr(GcArray(Signed))` — the `int_items` backing block (`erase([int])`).
/// Read as a Ref; combine with the GcArray(Signed) array descr
/// (`int_gcarray_descr`) for `GetarrayitemGcI` / `SetarrayitemGc`.
pub fn list_int_items_block_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 5)
}

/// `Ptr(GcArray(Float))` — the `float_items` backing block (`erase([float])`).
/// Read as a Ref; combine with the GcArray(Float) array descr
/// (`float_gcarray_descr`) for `GetarrayitemGcF` / `SetarrayitemGc`.
pub fn list_float_items_block_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 6)
}

pub fn list_bytes_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 9)
}

pub fn list_bytes_items_block_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 10)
}

pub fn list_ascii_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 11)
}

pub fn list_ascii_items_block_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 12)
}

pub fn list_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 7)
}

/// `Ptr(GcArray(OBJECTPTR))` — `wrappeditems` body per
/// `tupleobject.py` `_immutable_fields_ = ['wrappeditems[*]']`.
/// Immutable. Length comes from `arraylen_gc(items_block,
/// pyobject_gcarray_descr)` against the GcArray header — no
/// `tuple_length_descr` exists per upstream tupleobject.py:376-390
/// (`W_TupleObject` carries no separate length field).
pub fn tuple_wrappeditems_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 0)
}

pub fn tuple_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 1)
}

pub fn tuple_hash_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 2)
}

/// `W_SpecialisedTupleObject_ii.value0` — inline `i64` per
/// `specialisedtupleobject.py cls`. Immutable.
pub fn specialised_tuple_ii_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ii.value1` — inline `i64`. Immutable.
pub fn specialised_tuple_ii_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 1)
}

pub fn specialised_tuple_ii_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 2)
}

pub fn specialised_tuple_ii_hash_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 3)
}

/// `W_SpecialisedTupleObject_ff.value0` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ff.value1` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 1)
}

pub fn specialised_tuple_ff_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 2)
}

pub fn specialised_tuple_ff_hash_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 3)
}

/// `W_SpecialisedTupleObject_oo.value0` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_oo.value1` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 1)
}

pub fn specialised_tuple_oo_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 2)
}

pub fn specialised_tuple_oo_hash_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 3)
}

/// `ItemsBlock.capacity` — the GcArray length header at offset 0 of
/// an `ItemsBlock`, matching `rlist.py:84/251` `len(l.items)`
/// (allocated capacity, not live length). Immutable: once a block is
/// allocated the capacity is fixed; resize allocates a fresh block.
/// Callers combine `list_items_descr()` / `tuple_wrappeditems_descr()`
/// → `ItemsBlock*` with this descr to read the block's allocated size.
/// The group parent lets sub-walk reads pass the optimizer's
/// `ensure_ptr_info_arg0` parent lookup.
pub fn items_block_capacity_descr() -> DescrRef {
    field_descr_from_group(&ITEMS_BLOCK_DESCR_GROUP, 0)
}

pub fn int_intval_descr() -> DescrRef {
    field_descr_from_group(&W_INT_DESCR_GROUP, 0)
}

/// `W_IntObjectUser.intval`. Same byte offset as [`int_intval_descr`];
/// the descr identity is this group's field 2, which is what a virtual from
/// [`w_int_user_size_descr`] stores.
pub fn int_user_intval_descr() -> DescrRef {
    field_descr_from_group(&W_INT_USER_DESCR_GROUP, 2)
}

/// `W_IntObjectUser.map` — field 0 of [`W_INT_USER_DESCR_GROUP`].
pub fn int_user_map_descr() -> DescrRef {
    field_descr_from_group(&W_INT_USER_DESCR_GROUP, 0)
}

/// `W_IntObjectUser.storage` — field 1 of [`W_INT_USER_DESCR_GROUP`].
pub fn int_user_storage_descr() -> DescrRef {
    field_descr_from_group(&W_INT_USER_DESCR_GROUP, 1)
}

pub fn bool_intval_descr() -> DescrRef {
    field_descr_from_group(&W_BOOL_DESCR_GROUP, 0)
}

pub fn float_floatval_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_DESCR_GROUP, 0)
}

/// `W_FloatObjectUser.floatval`. Same byte offset as [`float_floatval_descr`];
/// the descr identity is this group's field 2, which is what a virtual from
/// [`w_float_user_size_descr`] stores.
pub fn float_user_floatval_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_USER_DESCR_GROUP, 2)
}

/// `W_FloatObjectUser.map` — field 0 of [`W_FLOAT_USER_DESCR_GROUP`].
pub fn float_user_map_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_USER_DESCR_GROUP, 0)
}

/// `W_FloatObjectUser.storage` — field 1 of [`W_FLOAT_USER_DESCR_GROUP`].
pub fn float_user_storage_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_USER_DESCR_GROUP, 1)
}

pub fn complex_real_descr() -> DescrRef {
    field_descr_from_group(&W_COMPLEX_DESCR_GROUP, 0)
}

pub fn complex_imag_descr() -> DescrRef {
    field_descr_from_group(&W_COMPLEX_DESCR_GROUP, 1)
}

/// FieldDescr for `W_LongObject.value` (the `*mut BigInt` gc-pointer), for the
/// inline-NEW boxing of a `jit_w_long_*_raw` result.
pub fn long_value_descr() -> DescrRef {
    field_descr_from_group(&W_LONG_DESCR_GROUP, 0)
}

/// Address a `FrameDebugData` field by its byte offset rather than by its
/// position in the group's list: a field added at the head re-points every
/// later index, and the reader that keeps a stale one then guards a
/// neighbouring slot instead of its own.
fn frame_debug_data_field_descr_at(offset: usize) -> DescrRef {
    let index = FRAME_DEBUG_DATA_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == offset)
        .expect("FrameDebugData descr group has no field at requested offset");
    field_descr_from_group(&*FRAME_DEBUG_DATA_DESCR_GROUP, index)
}

/// `FrameDebugData.w_locals` — the mapping `getorcreatedebug().w_locals`
/// reads at the head of `fast2locals` (pyframe.py).
pub fn frame_debug_data_w_locals_descr() -> DescrRef {
    frame_debug_data_field_descr_at(pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_LOCALS_OFFSET)
}

/// Rare per-frame globals override installed when one PyCode is executed in
/// a namespace other than its first-seen globals.
pub fn frame_debug_data_w_globals_descr() -> DescrRef {
    frame_debug_data_field_descr_at(pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_GLOBALS_OFFSET)
}

/// `FrameDebugData.w_extra_locals` — the dict `framelocalsproxy_setitem`
/// allocates for a key that names no writable fast local.  `fast2locals` never
/// touches it, but `frame_locals_proxy_snapshot` copies it into every mapping
/// `locals()` / `vars()` / `dir()` hand back, so a fold that rebuilds that
/// mapping from fastlocals alone has to see it.
pub fn frame_debug_data_w_extra_locals_descr() -> DescrRef {
    frame_debug_data_field_descr_at(
        pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_EXTRA_LOCALS_OFFSET,
    )
}

/// `FrameDebugData.w_f_trace` — the frame's own trace function, the slot
/// `pyframe.py get_w_f_trace` reads and `dispatch_bytecode`'s jitted arm tests
/// (`_d.w_f_trace is not None`).  Recorded at the portal merge point by
/// `record_portal_debugdata_guard` when the frame already carries a debug
/// block, which is the state every frame is in once a trace function has been
/// called on it.
pub fn frame_debug_data_w_f_trace_descr() -> DescrRef {
    frame_debug_data_field_descr_at(pyre_interpreter::pyframe::FRAME_DEBUG_DATA_W_F_TRACE_OFFSET)
}

/// `len(bytes)` returns the byte count.  `bytesobject.py` reads it as
/// `len(self._value)` on the RPython string; pyre precomputes it into
/// `W_BytesObject.len`, so the same answer is one immutable field read.
pub fn bytes_len_descr() -> DescrRef {
    field_descr_from_group(&W_BYTES_DESCR_GROUP, 1)
}

/// `W_BytearrayObject.length` — the count `bytearrayobject.py:_len` answers
/// with. Mutable; see [`W_BYTEARRAY_DESCR_GROUP`].
pub fn bytearray_length_descr() -> DescrRef {
    field_descr_from_group(&W_BYTEARRAY_DESCR_GROUP, 1)
}

/// `W_SetObject.len` — the count `setobject.py`'s `length` answers with,
/// reached there through the strategy. Mutable; see [`W_SET_DESCR_GROUP`].
pub fn set_len_descr() -> DescrRef {
    // `sstorage`, `strategy`, then `len`.
    field_descr_from_group(&W_SET_DESCR_GROUP, 2)
}

/// `W_SetObject.set_id`. Immutable; see [`W_SET_DESCR_GROUP`].
pub fn set_id_descr() -> DescrRef {
    field_descr_from_group(&W_SET_DESCR_GROUP, 4)
}

/// `W_SetObject.content_gen`. Mutable; see [`W_SET_DESCR_GROUP`].
pub fn set_content_gen_descr() -> DescrRef {
    field_descr_from_group(&W_SET_DESCR_GROUP, 5)
}

pub fn str_len_descr() -> DescrRef {
    // Python len(str) returns codepoint count.
    // unicodeobject.py W_UnicodeObject._len() → _length field.
    // `W_UnicodeObject.len` is a `usize`: 8 bytes on 64-bit, 4 on wasm32 — a
    // hardcoded 8 reads the adjacent field into the high half on a 32-bit
    // target (blackhole resume of `len(str)`).
    field_descr_from_group(&W_UNICODE_DESCR_GROUP, 2)
}

/// `W_UnicodeObject.value` — `_utf8`, the rstr `STR` payload
/// `descr_add` concatenates.
pub fn unicode_utf8_descr() -> DescrRef {
    field_descr_from_group(&W_UNICODE_DESCR_GROUP, 0)
}

/// `W_UnicodeObject.byte_len` — cached `len(_utf8)`, the rstr `STR` length.
pub fn unicode_byte_len_descr() -> DescrRef {
    field_descr_from_group(&W_UNICODE_DESCR_GROUP, 1)
}

/// `W_UnicodeObject.index_storage` — `_index_storage`, `rutf8.null_storage()`
/// until the first non-ASCII index.
pub fn unicode_index_storage_descr() -> DescrRef {
    field_descr_from_group(&W_UNICODE_DESCR_GROUP, 4)
}

// Object header and allocation descriptors.

/// `PyCode.code_ptr` — the host `CodeObject` every code-field getter resolves
/// through (`code_get_field` -> `require_code`).  Read only to prove it is
/// non-null, which is the check the getter would have run.
///
/// These four read-only fields form one positional group so every `getfield_gc`
/// can recover the owning SizeDescr at `optimizer.py:478`.  The unkeyed group
/// factory publishes only into the JIT descriptor snapshot; the collector's
/// `TypeInfo` table remains solely owned by `eval::initialize_gc`.  The parent
/// can therefore carry the real GC-managed PyCode type id, allowing
/// `StructPtrInfo.make_guards` to emit the correct `GUARD_GC_TYPE(code, 43)`,
/// without registering a second collector layout.
static PYCODE_DESCR_GROUP: LazyLock<majit_ir::descr::SimpleDescrGroup> = LazyLock::new(|| {
    use majit_ir::descr::{ArrayFlag, SimpleFieldDescrSpec};
    // `is_immutable` follows `pycode.py PyCode._immutable_fields_` per field.
    // `co_firstlineno` is listed there; `co_name` and `hidden_applevel` are
    // not. `code_ptr` has no upstream slot: `w_code_new_owned` stores the
    // permanently-live body once, before publication, so the read is pure
    // the way `co_code` is.
    let field = |field_key: &str,
                 offset: usize,
                 field_size: usize,
                 field_type: Type,
                 flag: ArrayFlag,
                 is_immutable: bool,
                 is_quasi_immutable: bool| SimpleFieldDescrSpec {
        index: stable_field_index(offset, field_size, field_type, flag == ArrayFlag::Signed),
        field_key: field_key.to_string(),
        name: field_key.to_string(),
        offset,
        field_size,
        field_type,
        is_immutable,
        is_quasi_immutable,
        flag,
        virtualizable: false,
        // The group lists only the four read-only payload fields; the
        // inherited `PyObject` header row is not among them, so none of them
        // is the class word and `class_word_field` answers `None` for this
        // layout. Testing `offset == W_CLASS_OFFSET` the way the object-group
        // factory does would read as if the header could appear here.
        is_class_word: Some(false),
        index_in_parent: 0,
    };
    let mut specs = vec![
        field(
            "code_ptr",
            pyre_interpreter::pycode::CODE_PTR_OFFSET,
            std::mem::size_of::<*const ()>(),
            Type::Int,
            ArrayFlag::Unsigned,
            true,
            false,
        ),
        // `pycode.py PyCode._immutable_fields_`: `w_globals?` is filled on
        // first execution and then promoted. A shared-code frame using a
        // different namespace keeps its override on FrameDebugData instead.
        field(
            "w_globals",
            pyre_interpreter::pycode::CODE_W_GLOBALS_OFFSET,
            std::mem::size_of::<pyre_object::PyObjectRef>(),
            Type::Ref,
            ArrayFlag::Pointer,
            false,
            true,
        ),
        // The slot is written once, by `box_code_object_with_firstlineno`,
        // onto the object `box_code_object` has just boxed out of a fresh
        // `Box` — no caching, so nothing can have read it first. `code.replace`
        // reads it and builds a new code object rather than writing this one.
        field(
            "co_firstlineno_raw",
            pyre_interpreter::pycode::CODE_CO_FIRSTLINENO_RAW_OFFSET,
            std::mem::size_of::<i32>(),
            Type::Int,
            ArrayFlag::Signed,
            true,
            false,
        ),
        field(
            "hidden_applevel",
            pyre_interpreter::pycode::CODE_HIDDEN_APPLEVEL_OFFSET,
            std::mem::size_of::<bool>(),
            Type::Int,
            ArrayFlag::Unsigned,
            false,
            false,
        ),
        // `co_name` is absent from `_immutable_fields_`, and this slot is
        // realized lazily besides: `w_code_name_obj` fills it on first demand,
        // so it goes null -> non-null after construction.
        field(
            "w_name",
            pyre_interpreter::pycode::CODE_W_NAME_OFFSET,
            std::mem::size_of::<pyre_object::PyObjectRef>(),
            Type::Ref,
            ArrayFlag::Pointer,
            false,
            false,
        ),
    ];
    specs.sort_by_key(|spec| spec.offset);
    for (index_in_parent, spec) in specs.iter_mut().enumerate() {
        spec.index_in_parent = index_in_parent;
    }
    majit_ir::descr::make_simple_descr_group_with_flags(
        u32::MAX,
        std::mem::size_of::<pyre_interpreter::pycode::PyCode>(),
        pyre_interpreter::pycode::W_CODE_GC_TYPE_ID,
        0,
        true,
        false,
        &specs,
    )
});

fn pycode_field_descr_at(offset: usize) -> DescrRef {
    let index = PYCODE_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == offset)
        .expect("PyCode descr group has no field at requested offset");
    field_descr_from_group(&*PYCODE_DESCR_GROUP, index)
}

pub fn pycode_code_ptr_descr() -> DescrRef {
    pycode_field_descr_at(pyre_interpreter::pycode::CODE_PTR_OFFSET)
}

pub fn pycode_w_globals_descr() -> DescrRef {
    pycode_field_descr_at(pyre_interpreter::pycode::CODE_W_GLOBALS_OFFSET)
}

/// `pycode.py` `"w_globals?"` — the globals `pyframe.py PyFrame.get_w_globals`
/// reads off `jit.promote(self.pycode)`.
///
/// Minted standalone rather than read out of [`PYCODE_DESCR_GROUP`], the way
/// [`STATICMETHOD_W_FUNCTION_QUASI_DESCR`] is: the group's row derives its
/// index from the layout, and this owner needs an index of its own for the
/// reason [`PYCODE_DESCR_TAG`] records. The group keeps its census row for
/// the field's ordinary reads.
static PYCODE_W_GLOBALS_QUASI_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    Arc::new(
        majit_ir::descr::SimpleFieldDescr::new_with_name(
            PYCODE_W_GLOBALS_INDEX,
            pyre_interpreter::pycode::CODE_W_GLOBALS_OFFSET,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            majit_ir::descr::ArrayFlag::Pointer,
            "PyCode.w_globals".to_string(),
            "w_globals".to_string(),
        )
        .with_quasi_immutable(true),
    )
});

pub fn pycode_w_globals_quasi_descr() -> DescrRef {
    PYCODE_W_GLOBALS_QUASI_DESCR.clone()
}

/// `PyCode.w_name` — the realized `co_name` string.  `w_code_name_obj` builds
/// it on first demand and retains it, so the slot IS the getter's value once
/// it is non-null; a null slot declines to the residual, which realizes it.
pub fn pycode_w_name_descr() -> DescrRef {
    pycode_field_descr_at(pyre_interpreter::pycode::CODE_W_NAME_OFFSET)
}

/// `PyCode.co_firstlineno_raw` — a signed 32-bit slot, because 3.14's
/// `CodeType` constructor accepts zero and negative first lines that
/// `CodeObject.first_line_number` cannot hold.
pub fn pycode_co_firstlineno_descr() -> DescrRef {
    pycode_field_descr_at(pyre_interpreter::pycode::CODE_CO_FIRSTLINENO_RAW_OFFSET)
}

/// `PyCode.hidden_applevel` — the frame-hidden flag read by
/// `PyFrame.hide()`.
pub fn pycode_hidden_applevel_descr() -> DescrRef {
    pycode_field_descr_at(pyre_interpreter::pycode::CODE_HIDDEN_APPLEVEL_OFFSET)
}

/// Size descriptor for W_IntObject allocation via NewWithVtable.
/// vtable = &INT_TYPE (ob_type for virtual materialization).
pub fn w_int_size_descr() -> DescrRef {
    W_INT_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_IntObjectUser` allocation via NewWithVtable.
/// vtable = `&INT_USER_TYPE`.
pub fn w_int_user_size_descr() -> DescrRef {
    W_INT_USER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_UnicodeObject` allocation via NewWithVtable.
/// vtable = &STR_TYPE (`unicodeobject.py W_UnicodeObject.__init__`).
pub fn w_unicode_size_descr() -> DescrRef {
    W_UNICODE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_BoolObject allocation via NewWithVtable.
/// vtable = &BOOL_TYPE; type_id = 0 (bool reuses the OBJECT root id).
pub fn w_bool_size_descr() -> DescrRef {
    W_BOOL_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_IntRangeIterator allocation via NewWithVtable.
/// vtable = &RANGE_ITER_TYPE; type_id = 0.
pub fn w_range_iter_size_descr() -> DescrRef {
    RANGE_ITER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_IntRangeStepOneIterator` allocation via
/// NewWithVtable.  vtable = &RANGE_ITER_STEP_ONE_TYPE.
pub fn w_range_iter_step_one_size_descr() -> DescrRef {
    RANGE_ITER_STEP_ONE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_IntRangeOneArgIterator` allocation via
/// NewWithVtable.  vtable = &RANGE_ITER_ONE_ARG_TYPE.
pub fn w_range_iter_one_arg_size_descr() -> DescrRef {
    RANGE_ITER_ONE_ARG_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_Range allocation via NewWithVtable.
/// vtable = &RANGE_TYPE.
pub fn w_range_size_descr() -> DescrRef {
    RANGE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_FloatObject allocation via NewWithVtable.
/// vtable = &FLOAT_TYPE (ob_type for virtual materialization).
pub fn w_float_size_descr() -> DescrRef {
    W_FLOAT_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_FloatObjectUser` allocation via NewWithVtable.
/// vtable = `&FLOAT_USER_TYPE`.
pub fn w_float_user_size_descr() -> DescrRef {
    W_FLOAT_USER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_ComplexObject` allocation via NewWithVtable.
/// vtable = &COMPLEX_TYPE (ob_type for virtual materialization).
pub fn w_complex_size_descr() -> DescrRef {
    W_COMPLEX_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_LongObject allocation via NewWithVtable (the inline
/// boxing of a bigint result). vtable = &LONG_TYPE (ob_type).
pub fn w_long_size_descr() -> DescrRef {
    W_LONG_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for canonical `W_TupleObject`.
pub fn w_tuple_size_descr() -> DescrRef {
    W_TUPLE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `iterobject.py W_TupleIterObject`.
pub fn tuple_iter_size_descr() -> DescrRef {
    TUPLE_ITER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `iterobject.py W_FastListIterObject`.
pub fn list_iter_size_descr() -> DescrRef {
    LIST_ITER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `functional.py W_Zip`.
pub fn w_zip_size_descr() -> DescrRef {
    W_ZIP_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_ii`.
pub fn specialised_tuple_ii_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_II_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_ff`.
pub fn specialised_tuple_ff_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_FF_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_oo`.
pub fn specialised_tuple_oo_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_OO_DESCR_GROUP.size_descr.clone()
}

/// SizeDescr + field descrs for exception allocation via NewWithVtable.
/// `user` selects the realbase struct or its `_getusercls` layout. Exact
/// realbases keep `exc_realbase_pytype` (the slim base is `EXCEPTION_TYPE`);
/// every other class shares the user PyType. Constructor-written fields
/// share offsets across both layouts. `w_context` is written separately
/// by the raise lowering; remaining pointer slots stay zeroed by GC
/// pointer clearing (PY_NULL), matching `w_exception_new_empty`.
///
/// Positional order is declaration order with the two `ob_header` words
/// dropped, matching `heaptracker.py all_fielddescrs` /
/// `get_fielddescr_index_in`.  The class word is appended last so every
/// real field keeps the index that walk numbers.
fn build_w_exception_group(kind: ExcKind, user: bool) -> PyreObjectDescrGroup {
    const SLIM_FIELDS: &[(&str, usize, usize, Type, bool, bool, bool)] = &[
        ("kind", EXC_KIND_OFFSET, 1, Type::Int, false, false, false),
        (
            "args_w",
            EXC_ARGS_W_OFFSET,
            WORD,
            Type::Ref,
            false,
            false,
            false,
        ),
        (
            "w_cause",
            EXC_W_CAUSE_OFFSET,
            WORD,
            Type::Ref,
            false,
            false,
            false,
        ),
        (
            "w_context",
            EXC_W_CONTEXT_OFFSET,
            WORD,
            Type::Ref,
            false,
            false,
            false,
        ),
        (
            "w_traceback",
            EXC_W_TRACEBACK_OFFSET,
            WORD,
            Type::Ref,
            false,
            false,
            false,
        ),
        (
            "suppress_context",
            EXC_SUPPRESS_CONTEXT_OFFSET,
            1,
            Type::Int,
            false,
            false,
            false,
        ),
        (
            "w_dict",
            EXC_W_DICT_OFFSET,
            WORD,
            Type::Ref,
            false,
            false,
            false,
        ),
        (
            "w_class",
            W_CLASS_OFFSET,
            WORD,
            Type::Ref,
            false,
            false,
            false,
        ),
    ];
    if !exc_kind_uses_extended_layout(kind) {
        return finish_exception_descr_group(kind, user, SLIM_FIELDS.to_vec());
    }
    finish_exception_descr_group(
        kind,
        user,
        vec![
            // Positional order is `W_BaseException` declaration order with the
            // two `ob_header` words dropped, matching `heaptracker.py
            // all_fielddescrs` / `get_fielddescr_index_in`.  The class word is
            // appended last so every real field keeps the index that walk
            // numbers; the walk never numbers a header word.
            // `kind` is a `u8` tag (1 byte, unsigned).
            ("kind", EXC_KIND_OFFSET, 1, Type::Int, false, false, false),
            (
                "args_w",
                EXC_ARGS_W_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_cause",
                EXC_W_CAUSE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            // `w_context` (`__context__`): a GC pointer slot.  Written by
            // the RAISE_VARARGS `__context__` chaining lowering
            // (`exc.w_context = ec.sys_exc_value`) so the optimizer can
            // track it on the virtual exception.
            (
                "w_context",
                EXC_W_CONTEXT_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_traceback",
                EXC_W_TRACEBACK_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            // This plain byte is not included in gc_fielddescrs, so a nursery
            // allocation must initialize it explicitly.
            (
                "suppress_context",
                EXC_SUPPRESS_CONTEXT_OFFSET,
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "w_object",
                EXC_W_OBJECT_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_start",
                EXC_W_START_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_end",
                EXC_W_END_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_reason",
                EXC_W_REASON_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_encoding",
                EXC_W_ENCODING_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_errno",
                EXC_W_ERRNO_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_winerror",
                EXC_W_WINERROR_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_strerror",
                EXC_W_STRERROR_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_filename",
                EXC_W_FILENAME_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_filename2",
                EXC_W_FILENAME2_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "written",
                EXC_WRITTEN_OFFSET,
                std::mem::size_of::<i64>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "w_code",
                EXC_W_CODE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_value",
                EXC_W_VALUE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_exc_name",
                EXC_W_NAME_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_attr_obj",
                EXC_W_ATTR_OBJ_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_import_path",
                EXC_W_IMPORT_PATH_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_import_name_from",
                EXC_W_IMPORT_NAME_FROM_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_import_msg",
                EXC_W_IMPORT_MSG_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_msg",
                EXC_W_SYNTAX_MSG_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_filename",
                EXC_W_SYNTAX_FILENAME_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_lineno",
                EXC_W_SYNTAX_LINENO_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_offset",
                EXC_W_SYNTAX_OFFSET_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_text",
                EXC_W_SYNTAX_TEXT_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_end_lineno",
                EXC_W_SYNTAX_END_LINENO_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_end_offset",
                EXC_W_SYNTAX_END_OFFSET_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_print_file_and_line",
                EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_syntax_metadata",
                EXC_W_SYNTAX_METADATA_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_group_message",
                EXC_W_GROUP_MESSAGE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_group_exceptions",
                EXC_W_GROUP_EXCEPTIONS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_group_exceptions_repr",
                EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_dict",
                EXC_W_DICT_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_class",
                W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
    )
}

/// Size, tid, vtable and STRUCT key for one exception layout.
///
/// `cache_key_name` is the full struct path so [`DECLARED_GROUPS`] hashes
/// the same identity `NewWithVtable` allocates. User layouts append `map`
/// and `storage` immediately before the trailing class word.
fn finish_exception_descr_group(
    kind: ExcKind,
    user: bool,
    mut fields: Vec<(&'static str, usize, usize, Type, bool, bool, bool)>,
) -> PyreObjectDescrGroup {
    let extended = exc_kind_uses_extended_layout(kind);
    if user {
        let class = fields
            .pop()
            .expect("exception descr group ends with w_class");
        if extended {
            fields.push((
                "map",
                EXC_EXTENDED_USER_MAP_OFFSET,
                WORD,
                Type::Int,
                false,
                false,
                false,
            ));
            fields.push((
                "storage",
                EXC_EXTENDED_USER_STORAGE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ));
        } else {
            fields.push((
                "map",
                EXC_USER_MAP_OFFSET,
                WORD,
                Type::Int,
                false,
                false,
                false,
            ));
            fields.push((
                "storage",
                EXC_USER_STORAGE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ));
        }
        fields.push(class);
    }
    let (size, tid, vtable, simple_name, def_path) = match (extended, user) {
        (false, false) => (
            W_BASE_EXCEPTION_SIZE,
            W_BASE_EXCEPTION_GC_TYPE_ID,
            &pyre_object::interp_exceptions::EXCEPTION_TYPE as *const _ as usize,
            "W_BaseException",
            "interp_exceptions::W_BaseException",
        ),
        (false, true) => (
            W_BASE_EXCEPTION_USER_SIZE,
            W_BASE_EXCEPTION_USER_GC_TYPE_ID,
            &pyre_object::interp_exceptions::BASE_EXCEPTION_USER_TYPE as *const _ as usize,
            "W_BaseExceptionUser",
            "interp_exceptions::W_BaseExceptionUser",
        ),
        (true, false) => (
            W_EXCEPTION_EXTENDED_SIZE,
            W_EXCEPTION_EXTENDED_GC_TYPE_ID,
            exc_realbase_pytype(kind) as *const _ as usize,
            "W_ExceptionExtended",
            "interp_exceptions::W_ExceptionExtended",
        ),
        (true, true) => (
            W_EXCEPTION_EXTENDED_USER_SIZE,
            W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID,
            &pyre_object::interp_exceptions::EXCEPTION_EXTENDED_USER_TYPE as *const _ as usize,
            "W_ExceptionExtendedUser",
            "interp_exceptions::W_ExceptionExtendedUser",
        ),
    };
    build_object_descr_group_with_extra_gc_edges(
        size,
        tid,
        vtable,
        &fields,
        simple_name,
        def_path,
        &[],
        &[],
        def_path,
        false,
    )
}

static W_BASE_EXCEPTION_DESCR_CACHE: LazyLock<Mutex<Vec<Option<PyreObjectDescrGroup>>>> =
    LazyLock::new(|| Mutex::new((0..EXC_KIND_COUNT).map(|_| None).collect()));

/// `_getusercls` groups, indexed by the same `ExcKind` as the base cache.
/// Two `Vec`s, not a side table: base versus user is one bit.
static W_EXCEPTION_USER_DESCR_CACHE: LazyLock<Mutex<Vec<Option<PyreObjectDescrGroup>>>> =
    LazyLock::new(|| Mutex::new((0..EXC_KIND_COUNT).map(|_| None).collect()));

fn with_w_exception_group_for<R>(
    kind: ExcKind,
    user: bool,
    f: impl FnOnce(&PyreObjectDescrGroup) -> R,
) -> R {
    let idx = kind as u8 as usize;
    let cache = if user {
        &W_EXCEPTION_USER_DESCR_CACHE
    } else {
        &W_BASE_EXCEPTION_DESCR_CACHE
    };
    let mut cache = cache.lock();
    if cache[idx].is_none() {
        cache[idx] = Some(build_w_exception_group(kind, user));
    }
    f(cache[idx].as_ref().unwrap())
}

/// Map or storage field of a `_getusercls` exception, looked up by offset on
/// the user SizeDescr (`ValueError` for the slim layout, `FileNotFoundError`
/// for the extended one). Both share one struct identity per layout.
fn exception_user_mapdict_descr(extended: bool, offset: usize) -> DescrRef {
    let kind = if extended {
        ExcKind::FileNotFoundError
    } else {
        ExcKind::ValueError
    };
    with_w_exception_group_for(kind, true, |group| w_exception_field_at(group, offset))
}

fn w_exception_field_at(group: &PyreObjectDescrGroup, offset: usize) -> DescrRef {
    let field = group
        .field_descrs
        .iter()
        .position(|d| d.offset() == offset)
        .unwrap_or_else(|| panic!("exception descr group has no field at offset {offset}"));
    field_descr_from_group(group, field)
}

/// Locate a field of the per-kind exception group by offset.  See
/// [`w_exception_dict_descr`] for why offset lookup is the right idiom.
fn w_exception_field_descr_by_offset_for(kind: ExcKind, user: bool, offset: usize) -> DescrRef {
    with_w_exception_group_for(kind, user, |group| w_exception_field_at(group, offset))
}

fn w_exception_field_descr_by_offset(kind: ExcKind, offset: usize) -> DescrRef {
    w_exception_field_descr_by_offset_for(kind, exc_kind_canonical_is_user_layout(kind), offset)
}

/// Field descrs for the exception construction emit: `(size, kind,
/// w_class, args_w)`.  Built and cached per `ExcKind` on first use.
pub fn w_exception_descrs_for(
    kind: ExcKind,
    user: bool,
) -> (DescrRef, DescrRef, DescrRef, DescrRef) {
    with_w_exception_group_for(kind, user, |group| {
        (
            group.size_descr.clone() as DescrRef,
            w_exception_field_at(group, EXC_KIND_OFFSET),
            w_exception_field_at(group, W_CLASS_OFFSET),
            w_exception_field_at(group, EXC_ARGS_W_OFFSET),
        )
    })
}

pub fn w_exception_descrs(kind: ExcKind) -> (DescrRef, DescrRef, DescrRef, DescrRef) {
    w_exception_descrs_for(kind, exc_kind_canonical_is_user_layout(kind))
}

/// Field descr for `W_BaseException.w_context` (the `__context__`
/// slot).  Used by the RAISE_VARARGS `__context__` chaining lowering;
/// shares the same parent `SizeDescr` as the `NewWithVtable` emit so the
/// optimizer recognises the store as a field of the virtual exception.
pub fn w_exception_context_descr_for(kind: ExcKind, user: bool) -> DescrRef {
    w_exception_field_descr_by_offset_for(kind, user, EXC_W_CONTEXT_OFFSET)
}

pub fn w_exception_context_descr(kind: ExcKind) -> DescrRef {
    w_exception_context_descr_for(kind, exc_kind_canonical_is_user_layout(kind))
}

/// Field descr for `W_BaseException.w_dict` (the lazily allocated instance
/// dictionary).  The LOAD_METHOD method-cache fold reads it to pin the
/// receiver at "carries no instance dictionary", which is what makes the
/// folded descriptor safe: a later `e.<name> = ...` allocates the dictionary
/// and side-exits.
///
/// Located by offset rather than by a hand-counted position.  The field
/// list is edited by hand, and naming the wrong index does not fail to
/// compile: it silently reads a neighbouring slot that stays null for an
/// ordinary subclass, which turns the shadowing guard below into a no-op
/// and lets compiled code keep calling a method an instance attribute has
/// already shadowed.
pub fn w_exception_dict_descr_for(kind: ExcKind, user: bool) -> DescrRef {
    w_exception_field_descr_by_offset_for(kind, user, EXC_W_DICT_OFFSET)
}

pub fn w_exception_dict_descr(kind: ExcKind) -> DescrRef {
    w_exception_dict_descr_for(kind, exc_kind_canonical_is_user_layout(kind))
}

/// Field descr for the plain `W_BaseException.suppress_context` byte.
pub fn w_exception_suppress_context_descr_for(kind: ExcKind, user: bool) -> DescrRef {
    w_exception_field_descr_by_offset_for(kind, user, EXC_SUPPRESS_CONTEXT_OFFSET)
}

pub fn w_exception_suppress_context_descr(kind: ExcKind) -> DescrRef {
    w_exception_suppress_context_descr_for(kind, exc_kind_canonical_is_user_layout(kind))
}

/// Field descriptor for `W_BaseException.w_traceback`, sharing the
/// per-kind exception allocation descriptor with the other exception slots.
pub fn w_exception_traceback_descr_for(kind: ExcKind, user: bool) -> DescrRef {
    w_exception_field_descr_by_offset_for(kind, user, EXC_W_TRACEBACK_OFFSET)
}

pub fn w_exception_traceback_descr(kind: ExcKind) -> DescrRef {
    w_exception_traceback_descr_for(kind, exc_kind_canonical_is_user_layout(kind))
}

static PYTRACEBACK_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_interpreter::pytraceback::{
        PYTRACEBACK_FRAME_OFFSET, PYTRACEBACK_GC_TYPE_ID, PYTRACEBACK_LASTI_OFFSET,
        PYTRACEBACK_LINENO_OFFSET, PYTRACEBACK_OBJECT_SIZE, PYTRACEBACK_TYPE,
        PYTRACEBACK_W_CODE_OFFSET, PYTRACEBACK_W_NEXT_OFFSET,
    };

    build_object_descr_group_with_extra_gc_edges(
        PYTRACEBACK_OBJECT_SIZE,
        PYTRACEBACK_GC_TYPE_ID,
        &PYTRACEBACK_TYPE as *const _ as usize,
        &[
            // Positional order is `PyTraceback` declaration order with the
            // two `ob_header` words dropped, matching `heaptracker.py
            // all_fielddescrs` / `get_fielddescr_index_in`.  The class word is
            // appended last so every real field keeps the index that walk
            // numbers; the walk never numbers a header word.
            (
                "frame",
                PYTRACEBACK_FRAME_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "lasti",
                PYTRACEBACK_LASTI_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "w_next",
                PYTRACEBACK_W_NEXT_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "lineno",
                PYTRACEBACK_LINENO_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "w_code",
                PYTRACEBACK_W_CODE_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_class",
                W_CLASS_OFFSET,
                WORD,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "PyTraceback",
        "pytraceback::PyTraceback",
        &[],
        &[],
        // `_cache_size` STRUCT identity has to be its own: sharing the
        // no-identity key hands `PYTRACEBACK_GC_TYPE_ID` to every keyless
        // struct header write.
        "PyTraceback",
        false,
    )
});

pub fn pytraceback_size_descr() -> DescrRef {
    PYTRACEBACK_DESCR_GROUP.size_descr.clone()
}

pub fn pytraceback_field_descr(index: usize) -> DescrRef {
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Field descriptor for `PyTraceback.frame`, the node's own frame slot read by
/// `descr_get_tb_frame`.
pub fn pytraceback_frame_descr() -> DescrRef {
    let index = PYTRACEBACK_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_interpreter::pytraceback::PYTRACEBACK_FRAME_OFFSET)
        .expect("PyTraceback descr group has no frame field");
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Field descriptor for `PyTraceback.w_next`, the chain link `descr_get_next`
/// reads.  Located by offset rather than by a hand-counted position, so a
/// later edit to the field list above cannot silently repoint this at a
/// neighbouring slot.
pub fn pytraceback_w_next_descr() -> DescrRef {
    let index = PYTRACEBACK_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_interpreter::pytraceback::PYTRACEBACK_W_NEXT_OFFSET)
        .expect("PyTraceback descr group has no w_next field");
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Field descriptor for `PyTraceback.lineno`, the line `descr_get_tb_lineno`
/// reports.  Located by offset for the same reason as
/// [`pytraceback_w_next_descr`].
pub fn pytraceback_lineno_descr() -> DescrRef {
    let index = PYTRACEBACK_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_interpreter::pytraceback::PYTRACEBACK_LINENO_OFFSET)
        .expect("PyTraceback descr group has no lineno field");
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Cached field descriptor for a raw reference slot selected by the
/// exception attribute fold.  The field is located by its `W_BaseException`
/// offset, so a slot added to `build_w_exception_group` cannot silently
/// renumber the ones after it; no parallel descriptor is constructed.
pub fn w_exception_attr_slot_descr(
    kind: ExcKind,
    slot: pyre_interpreter::baseobjspace::ExceptionAttrSlot,
) -> DescrRef {
    use pyre_interpreter::baseobjspace::ExceptionAttrSlot as Slot;
    let offset = match slot {
        Slot::Args => EXC_ARGS_W_OFFSET,
        Slot::Context => EXC_W_CONTEXT_OFFSET,
        Slot::Cause => EXC_W_CAUSE_OFFSET,
        Slot::Errno => EXC_W_ERRNO_OFFSET,
        Slot::Strerror => EXC_W_STRERROR_OFFSET,
        Slot::Filename => EXC_W_FILENAME_OFFSET,
        Slot::Filename2 => EXC_W_FILENAME2_OFFSET,
        Slot::Code => EXC_W_CODE_OFFSET,
        Slot::Traceback => EXC_W_TRACEBACK_OFFSET,
        Slot::UnicodeObject => EXC_W_OBJECT_OFFSET,
        Slot::UnicodeStart => EXC_W_START_OFFSET,
        Slot::UnicodeEnd => EXC_W_END_OFFSET,
        Slot::UnicodeReason => EXC_W_REASON_OFFSET,
        Slot::UnicodeEncoding => EXC_W_ENCODING_OFFSET,
        Slot::Name => EXC_W_NAME_OFFSET,
        Slot::AttrObj => EXC_W_ATTR_OBJ_OFFSET,
    };
    w_exception_field_descr_by_offset(kind, offset)
}

pub fn w_exception_attr_slot_descr_for(
    kind: ExcKind,
    slot: pyre_interpreter::baseobjspace::ExceptionAttrSlot,
    user: bool,
) -> DescrRef {
    use pyre_interpreter::baseobjspace::ExceptionAttrSlot as Slot;
    let offset = match slot {
        Slot::Args => EXC_ARGS_W_OFFSET,
        Slot::Context => EXC_W_CONTEXT_OFFSET,
        Slot::Cause => EXC_W_CAUSE_OFFSET,
        Slot::Errno => EXC_W_ERRNO_OFFSET,
        Slot::Strerror => EXC_W_STRERROR_OFFSET,
        Slot::Filename => EXC_W_FILENAME_OFFSET,
        Slot::Filename2 => EXC_W_FILENAME2_OFFSET,
        Slot::Code => EXC_W_CODE_OFFSET,
        Slot::Traceback => EXC_W_TRACEBACK_OFFSET,
        Slot::UnicodeObject => EXC_W_OBJECT_OFFSET,
        Slot::UnicodeStart => EXC_W_START_OFFSET,
        Slot::UnicodeEnd => EXC_W_END_OFFSET,
        Slot::UnicodeReason => EXC_W_REASON_OFFSET,
        Slot::UnicodeEncoding => EXC_W_ENCODING_OFFSET,
        Slot::Name => EXC_W_NAME_OFFSET,
        Slot::AttrObj => EXC_W_ATTR_OBJ_OFFSET,
    };
    w_exception_field_descr_by_offset_for(kind, user, offset)
}

/// Cached field descriptor for a flattened `W_BaseException` slot selected
/// by byte offset.  Returns `None` when the per-kind group does not carry the
/// requested offset.
pub fn w_exception_slot_descr_for(kind: ExcKind, offset: usize, user: bool) -> Option<DescrRef> {
    with_w_exception_group_for(kind, user, |group| {
        group
            .field_descrs
            .iter()
            .position(|d| d.offset() == offset)
            .map(|field| field_descr_from_group(group, field))
    })
}

pub fn w_exception_slot_descr(kind: ExcKind, offset: usize) -> Option<DescrRef> {
    w_exception_slot_descr_for(kind, offset, exc_kind_canonical_is_user_layout(kind))
}

/// Field descr for `ExecutionContext::sys_exc_value`, used by the JIT
/// lowering of PUSH_EXC_INFO / POP_EXCEPT to GETFIELD_GC_R / SETFIELD_GC.
///
/// `field_type = Ref` so the optimizer tracks the value as a GC
/// reference (virtual-defer + correct resume), but the `flag` is
/// deliberately NON-pointer so `is_pointer_field()` is false and the GC
/// rewrite emits NO write barrier (`rewrite.rs handle_write_barrier_setfield`
/// gates the barrier on `is_pointer_field() && val_is_ref`).  That is
/// correct here: the EC is a non-GC `Rc`-owned struct whose
/// `sys_exc_value` slot is forwarded directly as a GC root every
/// collection (`eval::walk_pyframe_roots`), so the generational
/// remembered-set barrier is unnecessary.  A single cached Arc gives the
/// PUSH and POP ops the same `descr_identity`, so the heap optimizer
/// dead-store-eliminates a balanced save/restore (and the stored
/// exception, if it never otherwise escapes, stays virtual and DCEs).
pub fn ec_sys_exc_value_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_SYS_EXC_VALUE_OFFSET)
}

/// Field descr for `ExecutionContext::current_gen_or_coroutine`, the slot
/// `executioncontext.py sys_exc_info` reads after a null `sys_exc_operror`.
/// Same group as [`ec_sys_exc_value_descr`] so a POP_EXCEPT restore of the
/// exception slot and this generator-head pin share one struct identity.
pub fn ec_current_gen_or_coroutine_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_CURRENT_GEN_OR_COROUTINE_OFFSET)
}

/// Field descr for `ExecutionContext::topframeref`, used by the JIT lowering
/// of `executioncontext.py enter` / `:96-97 leave` at an inlined call:
/// `frame.f_backref = self.topframeref` reads it and
/// `self.topframeref = jit.virtual_ref(frame)` writes it back.
///
/// Same group as [`ec_sys_exc_value_descr`] — one struct, one identity, so the
/// heap optimizer can forward an `enter` store to the matching `leave` read
/// and dead-store-eliminate a balanced pair whose frame never escaped.
pub fn ec_topframeref_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_TOPFRAMEREF_OFFSET)
}

/// Field descr for `ExecutionContext::w_tracefunc`, the slot
/// `executioncontext.py gettrace` reads (`jit.promote(self.w_tracefunc)`).
///
/// Marked `w_tracefunc?` so the portal pin is a `QUASIIMMUT_FIELD` marker
/// plus `GUARD_NOT_INVALIDATED`; `settrace` invalidates the watchers.
/// The index is [`EC_W_TRACEFUNC_INDEX`], not the group's ordinal `2`:
/// `quasi_immut_descr` casts from that number alone.
/// Recorded at the portal merge point by `record_portal_tracefunc_guard`
/// (`jitcode_dispatch/mod.rs`).  Same group as the two accessors above, so
/// the read shares their struct identity and the heapcache collapses a run
/// of merge points to one.
pub fn ec_w_tracefunc_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_W_TRACEFUNC_OFFSET)
}

/// Field descr for `ExecutionContext::profilefunc`, the slot
/// `executioncontext.py` lists as `profilefunc?` next to `w_tracefunc?`.
///
/// The compiled loop pins the empty function-pointer with `GetfieldGcI` +
/// `IntIsZero` + `GuardTrue`, the same shape PyPy's opt log shows for
/// `inst_profilefunc`.  `setllprofile` invalidates the watchers.
pub fn ec_profilefunc_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_PROFILEFUNC_OFFSET)
}

/// Resolve one `EC_DESCR_GROUP` field by byte offset.  The group stamps
/// `index_in_parent` as the field's rank by offset, so the positional order of
/// `field_descrs` follows the offsets rather than the declaration order; look
/// the field up by offset so the two accessors above cannot swap.
fn ec_field_descr(offset: usize) -> DescrRef {
    let parent = EC_DESCR_GROUP.size_descr.clone() as DescrRef;
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

/// Field descr for `ExecutionContext::py_recursion_depth`, the slot the
/// activation seam charges and gives back around a minted callee activation.
/// `pyframe.py` (`execute_frame.insert_stack_check_here`) puts the matching
/// test at the same place in the plain evaluator.
pub fn ec_py_recursion_depth_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_PY_RECURSION_DEPTH_OFFSET)
}

/// Field descr for `ExecutionContext::accounted_activation`, read and restored
/// by the same seam so a frame that reaches a portal door after the charge
/// does not pay a second unit.
pub fn ec_accounted_activation_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_ACCOUNTED_ACTIVATION_OFFSET)
}

/// The `ExecutionContext` field group.  `type_id 0 + vtable 0` →
/// `SimpleSizeDescr::is_object() == false`, so the optimizer builds a
/// StructPtrInfo for the (non-GC) EC pointer.  The object slots are Ref-typed
/// (ref value tracking) but Unsigned-flagged (`is_pointer_field()` is false →
/// no write barrier), which is correct because each is forwarded directly as
/// a GC root every collection (`eval::walk_pyframe_roots` for the running
/// context, `module::thread`'s per-context walk for the rest), so the
/// generational remembered-set barrier is unnecessary.
///
/// The two activation-accounting slots are Int-typed instead.  That walk
/// visits the context's object slots by name and never reaches them, which is
/// what makes an integer sound here: `accounted_activation` is a bare
/// identity token that may name an already-freed frame, so forwarding it as a
/// root would be wrong rather than merely unnecessary.
///
/// The group is minted with `is_gc_managed = false`: `ExecutionContext` is a
/// plain Rust struct reached through a raw pointer (`EC_SIZE` is
/// `size_of::<ExecutionContext>()`, the offsets are `offset_of!`), so it has no
/// type-id word at `ref - GcHeader::SIZE`.  `StructPtrInfo.make_guards` gates
/// `GUARD_GC_TYPE` on this flag, and the default GC-managed shape made it emit
/// `GUARD_GC_TYPE(ec, 0)` — a guard that reads the memory preceding the EC
/// allocation and compares it against the `type_id 0` above.
static EC_DESCR_GROUP: LazyLock<majit_ir::descr::SimpleDescrGroup> = LazyLock::new(|| {
    use majit_ir::descr::{ArrayFlag, SimpleFieldDescrSpec};
    let field = |index: u32, field_key: &str, offset: usize| SimpleFieldDescrSpec {
        index,
        // `descr.py get_field_descr` keys its cache on the bare fieldname;
        // `name` is the qualified display form.
        field_key: field_key.to_string(),
        name: format!("ExecutionContext.{field_key}"),
        offset,
        field_size: std::mem::size_of::<pyre_object::PyObjectRef>(),
        field_type: Type::Ref,
        is_immutable: false,
        is_quasi_immutable: false,
        flag: ArrayFlag::Unsigned,
        virtualizable: false,
        // `ExecutionContext` is not a `PyObject` layout; it carries no class
        // word. Its offsets are `EC_*` and share no origin with
        // `W_CLASS_OFFSET`, so matching on offset here would be a coincidence,
        // not an invariant.
        is_class_word: Some(false),
        // Stamped below, once the specs are in offset order.
        index_in_parent: 0,
    };
    // The activation counters.  `field_size` is derived from the field's own
    // type rather than spelled, so the 32-bit target reads the same four bytes
    // the struct actually stores there.
    let int_field = |index: u32, field_key: &str, offset: usize| SimpleFieldDescrSpec {
        index,
        field_key: field_key.to_string(),
        name: format!("ExecutionContext.{field_key}"),
        offset,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Int,
        is_immutable: false,
        is_quasi_immutable: false,
        flag: ArrayFlag::Unsigned,
        virtualizable: false,
        is_class_word: Some(false),
        index_in_parent: 0,
    };
    let mut w_tracefunc = field(2, "w_tracefunc", pyre_interpreter::EC_W_TRACEFUNC_OFFSET);
    // `executioncontext.py _immutable_fields_ = ['w_tracefunc?']`:
    // `get_field_descr` treats a `?` entry as pure (`!= False`) and quasi.
    // The reserved index is the owner identity `quasi_immut_descr` casts on;
    // the ordinal `2` is not one.
    w_tracefunc.index = EC_W_TRACEFUNC_INDEX;
    w_tracefunc.is_immutable = true;
    w_tracefunc.is_quasi_immutable = true;
    let mut profilefunc = int_field(5, "profilefunc", pyre_interpreter::EC_PROFILEFUNC_OFFSET);
    // `executioncontext.py _immutable_fields_ = ['profilefunc?']`.
    profilefunc.index = EC_PROFILEFUNC_INDEX;
    profilefunc.is_immutable = true;
    profilefunc.is_quasi_immutable = true;
    let mut specs = vec![
        field(
            0,
            "sys_exc_value",
            pyre_interpreter::EC_SYS_EXC_VALUE_OFFSET,
        ),
        field(1, "topframeref", pyre_interpreter::EC_TOPFRAMEREF_OFFSET),
        w_tracefunc,
        int_field(
            3,
            "py_recursion_depth",
            pyre_interpreter::EC_PY_RECURSION_DEPTH_OFFSET,
        ),
        int_field(
            4,
            "accounted_activation",
            pyre_interpreter::EC_ACCOUNTED_ACTIVATION_OFFSET,
        ),
        profilefunc,
        field(
            6,
            "current_gen_or_coroutine",
            pyre_interpreter::EC_CURRENT_GEN_OR_COROUTINE_OFFSET,
        ),
    ];
    // `index_in_parent` is the field's rank by byte offset
    // (`jitcode/assembler.rs`'s `field_specs_from_layout`), and once a parent
    // SizeDescr is bound it is also the `PtrInfo._fields` slot key the heap
    // optimizer reads
    // (`optimizeopt/heap.rs` `field_slot_index`), so the two fields must not
    // share a rank — a shared rank makes a read of one field resolve to the
    // cached value of the other.  `ExecutionContext` is `repr(Rust)`, so rank by
    // the actual offsets instead of the declaration order; sorting the specs
    // first keeps the rank, the spec order and the `all_fielddescrs` positional
    // order one numbering, as `.enumerate()` gives
    // `build_object_descr_group_with_def_path`.
    specs.sort_by_key(|spec| spec.offset);
    for (index_in_parent, spec) in specs.iter_mut().enumerate() {
        spec.index_in_parent = index_in_parent;
    }
    majit_ir::descr::make_simple_descr_group_with_flags(
        u32::MAX,
        pyre_interpreter::EC_SIZE,
        0,
        0,
        // is_gc_managed: no GC header, so no runtime type-id to guard on.
        false,
        // headerless: the JIT never allocates an `ExecutionContext`, so the
        // `NEW` rewrite shape this flag selects is not reachable.
        false,
        &specs,
    )
});

/// `gil.py` `GILThreadLocals.gil_ready` (`_immutable_fields_ = ['gil_ready?']`).
///
/// One process-wide `i64` at offset 0 of a non-GC `static mut`. Both
/// `is_immutable` and `is_quasi_immutable` are set so `SimpleFieldDescr::is_always_pure`
/// stays false: an always-pure read of a constant receiver folds the zero
/// with no `QuasiimmutField` and no invalidation. `is_gc_managed` is false
/// so the read does not emit `GUARD_GC_TYPE` against the bytes before the
/// static.
static GIL_READY_DESCR_GROUP: LazyLock<majit_ir::descr::SimpleDescrGroup> = LazyLock::new(|| {
    use majit_ir::descr::{ArrayFlag, SimpleFieldDescrSpec};
    let spec = SimpleFieldDescrSpec {
        index: GIL_READY_INDEX,
        field_key: "gil_ready".to_string(),
        name: "GilReadyState.gil_ready".to_string(),
        offset: 0,
        field_size: 8,
        field_type: Type::Int,
        is_immutable: true,
        is_quasi_immutable: true,
        flag: ArrayFlag::Signed,
        virtualizable: false,
        index_in_parent: 0,
        is_class_word: Some(false),
    };
    majit_ir::descr::make_simple_descr_group_with_flags(u32::MAX, 8, 0, 0, false, false, &[spec])
});

pub fn gil_ready_descr() -> DescrRef {
    GIL_READY_DESCR_GROUP.field_descrs[0].clone() as DescrRef
}

/// Size descriptor for W_SliceObject allocation via NewWithVtable.
/// vtable = &SLICE_TYPE (ob_type for virtual materialization).
/// Mirrors `pypy/objspace/std/objspace.py` `space.newslice` →
/// `W_SliceObject(w_start, w_end, w_step)` allocation shape.
pub fn w_slice_size_descr() -> DescrRef {
    W_SLICE_DESCR_GROUP.size_descr.clone()
}

/// `W_SliceObject.w_start` — `Ptr(W_Root)` per
/// `sliceobject.py:13` `_immutable_fields_ = ['w_start', ...]`. Immutable.
pub fn slice_w_start_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 0)
}

/// `W_SliceObject.w_stop` — `Ptr(W_Root)`. Immutable.
pub fn slice_w_stop_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 1)
}

/// `W_SliceObject.w_step` — `Ptr(W_Root)`. Immutable.
pub fn slice_w_step_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 2)
}

/// Cached SizeDescr for the host PyFrame virtualizable.
///
/// RPython's `GcCache.get_size_descr()` returns a stable descriptor
/// object for a given struct. Pyre keeps the PyFrame descriptors in the
/// `PYFRAME_DESCR_GROUP` singleton, so callers that need the parent
/// SizeDescr for `VirtualizableInfo::finalize_arc` must reuse that
/// cached Arc instead of allocating a fresh ephemeral `SizeDescr`.
pub fn pyframe_size_descr() -> DescrRef {
    PYFRAME_DESCR_GROUP.size_descr.clone()
}

pub fn pyframe_locals_cells_stack_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 0)
}

pub fn pyframe_stack_depth_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 1)
}

pub fn pyframe_next_instr_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 2)
}

pub fn pyframe_code_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 3)
}

/// rewrite.py `handle_call_assembler` scalar field read for the
/// `debugdata` slot of the virtualizable expansion (Phase D-1 prereq).
pub fn pyframe_debugdata_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 4)
}

pub fn pyframe_f_generator_wref_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 5)
}

pub fn pyframe_w_yielding_from_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 6)
}

pub fn pyframe_f_backref_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 7)
}

/// `pyframe.py PyFrame.get_builtin` — the per-frame builtin module selected
/// when the frame is created.  IMPORT_NAME traces this ordinary field read
/// before looking up `__import__` in the module dict.
pub fn pyframe_w_builtin_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 8)
}

/// `PyFrame.flags` — the byte carrying `FLAG_ESCAPED`.  Read-or-written by the
/// traced `tb_frame` fold to reproduce the getter's `mark_as_escaped()`.
/// Located by offset, so appending another field cannot repoint it.
pub fn pyframe_flags_descr() -> DescrRef {
    let index = PYFRAME_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == crate::frame_layout::PYFRAME_FLAGS_OFFSET)
        .expect("PyFrame descr group has no flags field");
    field_descr_from_group(&PYFRAME_DESCR_GROUP, index)
}

/// `PyFrame.failed_attr_cleanup` — the byte carrying the failed-attribute
/// finalizer state.  Its "run the deferred cleanup at the next opcode" value is
/// `u8::MAX`, so a `NewWithVtable` that leaves the slot holding recycled
/// nursery bytes can read as an armed request.  Located by offset, like
/// [`pyframe_flags_descr`], so appending another field cannot repoint it.
pub fn pyframe_failed_attr_cleanup_descr() -> DescrRef {
    let index = PYFRAME_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == crate::frame_layout::PYFRAME_FAILED_ATTR_CLEANUP_OFFSET)
        .expect("PyFrame descr group has no failed_attr_cleanup field");
    field_descr_from_group(&PYFRAME_DESCR_GROUP, index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A JIT-built inline callee frame is a `NewWithVtable` over recycled
    /// nursery bytes: `Nursery::reset` leaves them as they were
    /// (`malloc_zero_filled = False`) and the GC rewriter's `clear_gc_fields`
    /// initialises only the GC-reference slots.  Every non-`Type::Ref` field of
    /// `PyFrame` therefore owes the emitter an explicit store, and one added to
    /// the group without it is born holding whatever the previous tenant of
    /// those bytes left.  `failed_attr_cleanup` was that field: the value it
    /// reads as "run the deferred cleanup before the next opcode" is `u8::MAX`,
    /// so a recycled `0xff` made every dispatch on the frame run a full
    /// non-moving old-gen collection.
    #[test]
    fn inline_frame_emitters_store_every_scalar_pyframe_field() {
        fn setfield_offsets(
            emit: impl FnOnce(&mut majit_metainterp::TraceCtx) -> Option<majit_ir::OpRef>,
        ) -> std::collections::BTreeSet<usize> {
            let mut ctx = majit_metainterp::TraceCtx::for_test(0);
            emit(&mut ctx).expect("the emitter declines only on a namespace mismatch");
            ctx.into_recorder()
                .ops()
                .iter()
                .filter(|op| op.opcode == majit_ir::OpCode::SetfieldGc)
                .filter_map(|op| {
                    let descr = op.descr.borrow();
                    descr.as_ref()?.as_field_descr().map(FieldDescr::offset)
                })
                .collect()
        }

        let scalars: Vec<(&str, usize)> = PYFRAME_DESCR_GROUP
            .field_descrs
            .iter()
            .filter(|d| d.field_type() != Type::Ref)
            .map(|d| (d.field_name(), d.offset()))
            .collect();
        // `valuestackdepth`, `last_instr`, `flags`, `failed_attr_cleanup`: a
        // census that found fewer is reading the wrong group.
        assert!(scalars.len() >= 4, "scalar field census is vacuous");

        let with_params = setfield_offsets(|ctx| {
            let pycode = ctx.const_ref(0);
            let w_globals = ctx.const_ref(0);
            crate::helpers::emit_new_pyframe_inline_with_params(
                ctx,
                &[],
                &[],
                &[],
                0,
                1,
                0,
                pycode,
                w_globals,
            )
        });
        let self_recursive = setfield_offsets(|ctx| {
            let arg_box = ctx.const_ref(0);
            let pycode = ctx.const_ref(0);
            let w_globals = ctx.const_ref(0);
            crate::helpers::emit_new_pyframe_inline_self_recursive(
                ctx, arg_box, 1, 0, pycode, w_globals,
            )
        });

        for (emitter, stored) in [
            ("emit_new_pyframe_inline_with_params", &with_params),
            ("emit_new_pyframe_inline_self_recursive", &self_recursive),
        ] {
            for (name, offset) in &scalars {
                assert!(
                    stored.contains(offset),
                    "{emitter} stores no {name} (offset {offset}), so the field is \
                     born holding recycled nursery bytes",
                );
            }
        }
    }

    /// The same recycled-nursery-bytes rule, for the cell the inline frame
    /// builder allocates per pure cellvar: `family` is a `Type::Int` slot
    /// `clear_gc_fields` does not reach, and a cell born with a recycled word
    /// there hands `try_walker_specialize_load_deref` a `CellFamily` address
    /// to dereference that no `CellFamily` ever occupied.
    #[test]
    fn inline_cell_emitter_stores_every_scalar_cell_field() {
        let mut ctx = majit_metainterp::TraceCtx::for_test(0);
        let family = ctx.const_int(0);
        let header_w_class = ctx.const_ref(0);
        crate::helpers::emit_new_cell_inline(&mut ctx, family, header_w_class);
        let stored: std::collections::BTreeSet<usize> = ctx
            .into_recorder()
            .ops()
            .iter()
            .filter(|op| op.opcode == majit_ir::OpCode::SetfieldGc)
            .filter_map(|op| {
                let descr = op.descr.borrow();
                descr.as_ref()?.as_field_descr().map(FieldDescr::offset)
            })
            .collect();
        let scalars: Vec<(&str, usize)> = W_CELL_DESCR_GROUP
            .field_descrs
            .iter()
            .filter(|d| d.field_type() != Type::Ref)
            .map(|d| (d.field_name(), d.offset()))
            .collect();
        // `family`: a census that found none is reading the wrong group.
        assert!(!scalars.is_empty(), "scalar field census is vacuous");
        for (name, offset) in &scalars {
            assert!(
                stored.contains(offset),
                "emit_new_cell_inline stores no {name} (offset {offset}), so the field \
                 is born holding recycled nursery bytes",
            );
        }
    }

    /// Every group that stays out of both name registries still needs a
    /// `gc_cache._cache_size` STRUCT identity.  Sharing the no-identity key
    /// makes whichever group lands there the header tid for every keyless
    /// struct the blackhole and resume paths allocate, so a 24-byte block gets
    /// stamped `W_BASE_EXCEPTION_GC_TYPE_ID` and the collector then reads 36
    /// GC offsets across its neighbours.
    #[test]
    fn name_registry_less_groups_still_carry_a_cache_size_identity() {
        let (exception, ..) = w_exception_descrs(ExcKind::BaseException);
        let exception = exception.as_size_descr().expect("exception SizeDescr");
        assert_ne!(exception.cache_key(), 0);
        assert_eq!(
            majit_ir::descr::gc_cache()
                .lock()
                .resolve_struct_tid(exception.cache_key()),
            Some(W_BASE_EXCEPTION_GC_TYPE_ID)
        );
        let (value_error, ..) = w_exception_descrs(ExcKind::ValueError);
        let value_error = value_error
            .as_size_descr()
            .expect("ValueError user SizeDescr");
        assert_ne!(value_error.cache_key(), exception.cache_key());
        assert_eq!(
            majit_ir::descr::gc_cache()
                .lock()
                .resolve_struct_tid(value_error.cache_key()),
            Some(W_BASE_EXCEPTION_USER_GC_TYPE_ID)
        );

        let traceback = pytraceback_size_descr();
        let traceback = traceback.as_size_descr().expect("PyTraceback SizeDescr");
        assert_ne!(traceback.cache_key(), 0);
        assert_ne!(traceback.cache_key(), exception.cache_key());

        // And neither of them may be reachable through the sentinel: that is
        // the slot a serialized `BhDescr` with no key lands on.
        let sentinel = majit_ir::descr::gc_cache().lock().resolve_struct_tid(0);
        assert_ne!(sentinel, Some(W_BASE_EXCEPTION_GC_TYPE_ID));
        assert_ne!(
            sentinel,
            Some(pyre_interpreter::pytraceback::PYTRACEBACK_GC_TYPE_ID)
        );
    }

    /// A gc-managed declaration that leaves `type_id` at 0 does not thereby opt
    /// out of `GUARD_GC_TYPE`. `StructPtrInfo::make_guards`
    /// (`optimizeopt/info.rs`) emits one for any gc-managed, non-headerless
    /// descr, and it takes the id from whatever the `GcCache` slot for the key
    /// holds by then — a second producer's collector identity, which the
    /// runtime header can never match. `headerless` is the flag that declines
    /// the guard, so a struct that cannot name its runtime tid has to say so.
    #[test]
    fn a_gc_managed_group_that_names_no_type_id_declares_itself_headerless() {
        for (_, force) in DECLARED_GROUPS {
            force();
        }
        std::sync::LazyLock::force(&MODULE_USER_LAYOUT_DESCR_GROUPS);
        let gc = majit_ir::descr::gc_cache().lock();
        let is_unstamped = |def_path: &str| {
            let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(def_path));
            gc._cache_size
                .get(&key)
                .and_then(|descr| descr.as_size_descr())
                .is_some_and(|sd| sd.is_gc_managed() && !sd.headerless() && sd.type_id() == 0)
        };
        let mut unstamped: Vec<&str> = DECLARED_GROUPS
            .iter()
            .filter(|(def_path, _)| is_unstamped(def_path))
            .map(|(def_path, _)| *def_path)
            .collect();
        for entry in MODULE_USER_LAYOUT_DESCR_GROUPS.iter() {
            if is_unstamped(entry.def_path) {
                unstamped.push(entry.def_path);
            }
        }
        assert!(
            unstamped.is_empty(),
            "gc-managed groups with no type id and no headerless declaration: {unstamped:?}",
        );
    }

    /// `W_CType` is a pyre-module `#[pyre_class]` with no `DECLARED_GROUPS`
    /// row. Publishing the class census must make a `vtable=0` mint of
    /// its parent SizeDescr `is_object()`, so a speculative `kind` read
    /// takes `descr.py is_valid_class_for` instead of comparing the
    /// analyzer tid to the collector tid.
    #[cfg(feature = "pyre-module")]
    #[test]
    fn pyre_class_vtable_publish_makes_w_ctype_parent_an_object() {
        publish_pyre_class_vtables();
        let mut found = None;
        pyre_object::lltype::for_each_class_descriptor(|descr| {
            if descr
                .struct_path
                .ends_with("::_cffi_backend::ctypeobj::W_CType")
                || descr.struct_path.ends_with("::ctypeobj::W_CType")
            {
                found = Some(descr);
            }
        });
        let descr = found.expect("W_CType pyre_class descriptor");
        let stripped = match descr.struct_path.split_once("::") {
            Some((_, rest)) => rest,
            None => descr.struct_path,
        };
        assert_eq!(stripped, "module::_cffi_backend::ctypeobj::W_CType");
        let key = majit_ir::descr::path_hash(stripped);
        let mut gc = majit_ir::descr::gc_cache().lock();
        assert_eq!(
            gc._cache_gcstruct2vtable.get(&key).map(|(vt, _)| *vt),
            Some(descr.pytype_ptr as usize),
        );
        let parent = gc.get_size_descr(
            majit_ir::descr::LLType::Struct(key),
            descr.object_size,
            0,
            false,
        );
        let sd = parent.as_size_descr().expect("SizeDescr");
        assert!(
            sd.is_object(),
            "W_CType parent must carry the pytype vtable"
        );
        assert_eq!(sd.vtable(), descr.pytype_ptr as usize);
        assert!(
            !gc._cache_gcstruct2vtable
                .contains_key(&majit_ir::descr::path_hash("W_CType")),
            "leaf name is not a STRUCT key"
        );
    }

    /// Kind-0 packed parents of unpublished `#[pyre_class]` STRUCTs mint
    /// through `publish_borrowed_struct_layout` with the serialized
    /// `vtable=0`. That path must inject `_cache_gcstruct2vtable` the way
    /// `get_size_descr` does, so a speculative `kind` read takes
    /// `descr.py is_valid_class_for`.
    #[cfg(feature = "pyre-module")]
    #[test]
    fn pyre_class_vtable_is_injected_into_a_borrowed_kind0_parent() {
        publish_pyre_class_vtables();
        let mut found = None;
        pyre_object::lltype::for_each_class_descriptor(|descr| {
            if descr
                .struct_path
                .ends_with("::_cffi_backend::ctypeobj::W_CType")
                || descr.struct_path.ends_with("::ctypeobj::W_CType")
            {
                found = Some(descr);
            }
        });
        let descr = found.expect("W_CType pyre_class descriptor");
        let stripped = match descr.struct_path.split_once("::") {
            Some((_, rest)) => rest,
            None => descr.struct_path,
        };
        let key = majit_ir::descr::path_hash(stripped);
        let group = majit_ir::descr::publish_borrowed_struct_layout(
            u32::MAX,
            descr.object_size,
            0,
            key,
            0,
            true,
            false,
            &[],
            vec![majit_ir::descr::BorrowedField {
                index: 0,
                name: Cow::Borrowed("W_CType.kind"),
                field_key: "kind",
                offset: 64,
                field_size: 8,
                field_type: Type::Int,
                flag: majit_ir::descr::ArrayFlag::Signed,
                is_immutable: true,
                is_quasi_immutable: false,
                index_in_parent: 0,
                is_class_word: Some(false),
            }],
        );
        assert!(
            group.size_descr.is_object(),
            "borrowed W_CType parent must carry the pytype vtable"
        );
        assert_eq!(group.size_descr.vtable(), descr.pytype_ptr as usize);
        let parent_descr = group.field_descrs[0]
            .get_parent_descr()
            .expect("parent_descr");
        let parent = parent_descr.as_size_descr().expect("SizeDescr");
        assert!(parent.is_object());
        assert_eq!(parent.vtable(), descr.pytype_ptr as usize);
    }

    /// `_cache_gcstruct2vtable` is keyed by the STRUCT
    /// (`heaptracker.setup_cache_gcstruct2vtable`), never a short name.
    /// `W_Compress` is zlib and itertools; a leaf row would inject the
    /// first writer's vtable into the other class.
    #[cfg(feature = "pyre-module")]
    #[test]
    fn pyre_class_vtable_publish_keys_the_struct_not_the_leaf() {
        publish_pyre_class_vtables();
        let mut by_leaf: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        let mut full_paths: Vec<&str> = Vec::new();
        pyre_object::lltype::for_each_class_descriptor(|descr| {
            if descr.pytype_ptr.is_null() {
                return;
            }
            let stripped = match descr.struct_path.split_once("::") {
                Some((_, rest)) => rest,
                None => descr.struct_path,
            };
            full_paths.push(stripped);
            if let Some((_, leaf)) = stripped.rsplit_once("::") {
                by_leaf.entry(leaf).or_default().push(stripped);
            }
        });
        let gc = majit_ir::descr::gc_cache().lock();
        for path in &full_paths {
            let key = majit_ir::descr::path_hash(path);
            assert!(
                gc._cache_gcstruct2vtable.contains_key(&key),
                "missing vtable row for {path}"
            );
        }
        let mut shared_leaves = 0usize;
        for (leaf, paths) in &by_leaf {
            if paths.len() > 1 {
                shared_leaves += 1;
            }
            let leaf_is_a_full_path = full_paths.iter().any(|p| *p == *leaf);
            if leaf_is_a_full_path {
                continue;
            }
            let leaf_key = majit_ir::descr::path_hash(leaf);
            assert!(
                !gc._cache_gcstruct2vtable.contains_key(&leaf_key),
                "leaf {leaf} must not be a vtable key (paths {paths:?})"
            );
        }
        assert!(
            shared_leaves >= 1,
            "expected W_Compress (zlib vs itertools) to share a leaf"
        );
        let compress = by_leaf.get("W_Compress").expect("W_Compress leaf");
        assert!(
            compress.len() > 1,
            "W_Compress must be shared: {compress:?}"
        );
    }

    #[test]
    fn native_user_mapdict_fields_replace_prepass_placeholder_indices() {
        assert_eq!(W_INT_USER_DESCR_GROUP.field_descrs[0].index(), 0x6100_0000);
        assert_eq!(W_INT_USER_DESCR_GROUP.field_descrs[1].index(), 0x6100_0001);
        assert_eq!(W_INT_USER_DESCR_GROUP.field_descrs[2].index(), 0x6100_0002);
        assert_eq!(W_INT_USER_DESCR_GROUP.field_descrs[3].index(), 0x6100_0003);
        // The nested header leaf is `all_fielddescrs[0]`. The spec's own
        // class word stays `field_descrs[3]`.
        assert_eq!(
            W_INT_USER_DESCR_GROUP
                .size_descr
                .as_size_descr()
                .and_then(|sd| sd.class_word_index_in_parent()),
            Some(0)
        );
        assert_eq!(
            W_UNICODE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0010
        );
        assert_eq!(
            W_TUPLE_USER_DESCR_GROUP.field_descrs[1].index(),
            0x6100_0021
        );
        assert_eq!(
            W_FLOAT_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0030
        );
        assert_eq!(
            W_FLOAT_USER_DESCR_GROUP.field_descrs[1].index(),
            0x6100_0031
        );
        assert_eq!(
            W_FLOAT_USER_DESCR_GROUP.field_descrs[2].index(),
            0x6100_0032
        );
        assert_eq!(
            W_FLOAT_USER_DESCR_GROUP.field_descrs[3].index(),
            0x6100_0033
        );
        assert_eq!(
            W_FLOAT_USER_DESCR_GROUP
                .size_descr
                .as_size_descr()
                .and_then(|sd| sd.class_word_index_in_parent()),
            Some(0)
        );
        assert_eq!(
            W_COMPLEX_USER_DESCR_GROUP.field_descrs[1].index(),
            0x6100_0041
        );
        assert_eq!(
            W_BYTES_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0050
        );
        assert_eq!(
            W_BYTEARRAY_USER_DESCR_GROUP.field_descrs[1].index(),
            0x6100_0061
        );
        assert_eq!(W_LIST_USER_DESCR_GROUP.field_descrs[0].index(), 0x6100_0070);
        assert_eq!(W_SET_USER_DESCR_GROUP.field_descrs[0].index(), 0x6100_0080);
        assert_eq!(
            W_ARRAY_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0090
        );
        assert_eq!(
            W_WEAKREF_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_00A0
        );
        assert_eq!(
            W_ENUMERATE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_00B0
        );
        assert_eq!(W_MAP_USER_DESCR_GROUP.field_descrs[0].index(), 0x6100_00C0);
        assert_eq!(
            W_FILTER_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_00D0
        );
        assert_eq!(W_ZIP_USER_DESCR_GROUP.field_descrs[0].index(), 0x6100_00E0);
        assert_eq!(
            W_REVERSED_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_00F0
        );
        assert_eq!(
            W_SUPER_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0100
        );
        assert_eq!(
            W_PROPERTY_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0110
        );
        assert_eq!(
            W_COUNT_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0120
        );
        assert_eq!(
            W_REPEAT_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0130
        );
        assert_eq!(
            W_TAKEWHILE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0140
        );
        assert_eq!(
            W_DROPWHILE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0150
        );
        assert_eq!(
            W_FILTERFALSE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0160
        );
        assert_eq!(
            W_ISLICE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0170
        );
        assert_eq!(
            W_BATCHED_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0180
        );
        assert_eq!(
            W_PRODUCT_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0190
        );
        assert_eq!(
            W_COMBINATIONS_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_01A0
        );
        assert_eq!(
            W_COMBINATIONS_WITH_REPLACEMENT_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_01B0
        );
        assert_eq!(
            W_PERMUTATIONS_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_01C0
        );
        assert_eq!(
            W_GROUPBY_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_01D0
        );
        assert_eq!(
            W_COMPRESS_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_01E0
        );
        assert_eq!(
            W_STARMAP_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_01F0
        );
        assert_eq!(
            W_ACCUMULATE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0200
        );
        assert_eq!(
            W_ZIP_LONGEST_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0210
        );
        assert_eq!(
            W_PAIRWISE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0220
        );
        assert_eq!(
            W_CYCLE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0230
        );
        assert_eq!(
            W_CHAIN_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0240
        );
        assert_eq!(
            W_BUFFERABLE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0250
        );
        assert_eq!(
            W_DEQUE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F00
        );
        assert_eq!(
            W_STRUCT_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F10
        );
        assert_eq!(
            W_GENERIC_ALIAS_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F20
        );
        assert_eq!(W_LONG_USER_DESCR_GROUP.field_descrs[0].index(), 0x6100_0F30);
        assert_eq!(
            W_STATICMETHOD_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F40
        );
        assert_eq!(
            W_CLASSMETHOD_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F50
        );
        assert_eq!(
            W_MODULE_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F60
        );
        assert_eq!(
            W_LOCAL_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F70
        );
        assert_eq!(
            W_BYTESIO_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F80
        );
        assert_eq!(
            W_STRINGIO_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0F90
        );
        assert_eq!(
            W_BUFFEREDREADER_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0FA0
        );
        assert_eq!(
            W_BUFFEREDWRITER_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0FB0
        );
        assert_eq!(
            W_BUFFEREDRWPAIR_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0FC0
        );
        assert_eq!(
            W_BUFFEREDRANDOM_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0FD0
        );
        assert_eq!(
            W_TEXTIOWRAPPER_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0FE0
        );
        assert_eq!(
            W_FILEIO_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_1000
        );
        #[cfg(all(windows, not(feature = "sandbox")))]
        assert_eq!(
            W_WINCONSOLEIO_USER_DESCR_GROUP.field_descrs[0].index(),
            0x6100_0FF0
        );
        assert_eq!(
            W_INT_USER_DESCR_GROUP.field_descrs[1].field_type(),
            Type::Ref
        );
        assert_eq!(
            W_INT_USER_DESCR_GROUP.field_descrs[0].field_size(),
            core::mem::size_of::<usize>()
        );
        assert_eq!(
            W_UNICODE_USER_DESCR_GROUP.field_descrs[0].field_size(),
            core::mem::size_of::<usize>()
        );
        assert_eq!(
            W_TUPLE_USER_DESCR_GROUP.field_descrs[0].field_size(),
            core::mem::size_of::<usize>()
        );
    }

    #[test]
    fn str_len_descr_keeps_its_unicode_parent_alive() {
        let descr = str_len_descr();
        let field = descr.as_field_descr().expect("str len FieldDescr");
        assert_eq!(
            field.offset(),
            pyre_object::unicodeobject::UNICODE_LEN_OFFSET
        );
        assert!(field.is_immutable());

        let parent = field
            .get_parent_descr()
            .expect("W_UnicodeObject.len must retain its parent SizeDescr");
        let size = parent
            .as_size_descr()
            .expect("Unicode field parent must be a SizeDescr");
        assert_eq!(
            size.size(),
            pyre_object::unicodeobject::W_UNICODE_OBJECT_SIZE
        );
        assert_eq!(size.type_id(), W_UNICODE_GC_TYPE_ID);
        // `len` is spec index 2. The nested header leaf occupies
        // `all_fielddescrs[0]`, so the same descr sits at position 3.
        let parent_field = size.all_fielddescrs()[3].clone() as DescrRef;
        assert!(std::sync::Arc::ptr_eq(&parent_field, &descr));
    }

    #[test]
    fn pycode_field_descrs_share_parent_and_preserve_specs() {
        // The `always_pure` column is `pycode.py PyCode._immutable_fields_`
        // plus `code_ptr`. `co_firstlineno` is listed upstream, and
        // `code_ptr` is the permanently-live body pointer. Stating it per
        // field rather than asserting a blanket keeps the test able to fail
        // when a field's immutability moves in either direction.
        let expected = [
            (
                pycode_code_ptr_descr(),
                "code_ptr",
                pyre_interpreter::pycode::CODE_PTR_OFFSET,
                std::mem::size_of::<*const ()>(),
                Type::Int,
                false,
                0,
                true,
            ),
            (
                pycode_co_firstlineno_descr(),
                "co_firstlineno_raw",
                pyre_interpreter::pycode::CODE_CO_FIRSTLINENO_RAW_OFFSET,
                std::mem::size_of::<i32>(),
                Type::Int,
                true,
                1,
                true,
            ),
            (
                pycode_w_globals_descr(),
                "w_globals",
                pyre_interpreter::pycode::CODE_W_GLOBALS_OFFSET,
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                2,
                false,
            ),
            (
                pycode_hidden_applevel_descr(),
                "hidden_applevel",
                pyre_interpreter::pycode::CODE_HIDDEN_APPLEVEL_OFFSET,
                std::mem::size_of::<bool>(),
                Type::Int,
                false,
                3,
                false,
            ),
            (
                pycode_w_name_descr(),
                "w_name",
                pyre_interpreter::pycode::CODE_W_NAME_OFFSET,
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                4,
                false,
            ),
        ];
        let mut shared_parent = None;
        for (descr, name, offset, field_size, field_type, signed, index_in_parent, always_pure) in
            expected
        {
            let field = descr.as_field_descr().expect("PyCode FieldDescr");
            assert_eq!(field.field_name(), name);
            assert_eq!(field.offset(), offset);
            assert_eq!(field.field_size(), field_size);
            assert_eq!(field.field_type(), field_type);
            assert_eq!(field.is_field_signed(), signed);
            assert_eq!(descr.is_always_pure(), always_pure, "{name}");
            assert_eq!(descr.is_quasi_immutable(), name == "w_globals");
            assert_eq!(field.index_in_parent(), index_in_parent);
            let parent = field
                .get_parent_descr()
                .expect("PyCode fields must carry a parent_descr");
            if let Some(expected_parent) = &shared_parent {
                assert!(Arc::ptr_eq(expected_parent, &parent));
            } else {
                shared_parent = Some(parent);
            }
        }

        let parent = shared_parent.expect("PyCode group has fields");
        let size = parent
            .as_size_descr()
            .expect("PyCode field parent must be a SizeDescr");
        assert_eq!(
            size.size(),
            std::mem::size_of::<pyre_interpreter::pycode::PyCode>()
        );
        assert_eq!(size.type_id(), pyre_interpreter::pycode::W_CODE_GC_TYPE_ID);
        assert!(size.is_gc_managed());
        assert!(!size.headerless());
        assert_eq!(size.all_fielddescrs().len(), 5);
    }

    #[test]
    fn pyobject_size_descrs_include_inherited_w_class_gc_field() {
        let descr = w_int_size_descr();
        let size = descr.as_size_descr().expect("W_IntObject SizeDescr");
        assert!(
            size.gc_fielddescrs()
                .iter()
                .any(|fd| fd.offset() == W_CLASS_OFFSET),
            "malloc_zero_filled=False requires the inherited PyObject.w_class edge"
        );
    }

    /// Every layout's class word is the inherited `PyObject` header slot, and
    /// both accessors must name that one field.
    ///
    /// `SimpleFieldDescr` falls back to inferring `is_w_class` from the field
    /// name (`class_word_inferred_from_name`: `== "w_class"` or
    /// `ends_with(".w_class")`) when nothing declared, and
    /// `build_object_descr_group_with_extra_gc_edges` qualifies every name
    /// with the group's simple name.  So a group that declares a payload field
    /// literally called `w_class` publishes *two* descrs reporting
    /// `is_w_class()`, and both accessors take the first — which is the payload
    /// field, at the wrong offset and the wrong slot.
    ///
    /// Both halves are asserted because the two accessors read different lists:
    /// `class_word_field` answers a byte offset off `gc_fielddescrs`, and
    /// `class_word_index_in_parent` a slot position off `all_fielddescrs`.
    /// Each is asserted *per group*, `None` included, since a group that stops
    /// answering is invisible to any check phrased over the groups that still
    /// do.  The two lists differ in what an answer is owed for:
    /// `gc_fielddescrs` carries the shared header edge for every group, so
    /// `class_word_field` must always answer; `all_fielddescrs` holds the
    /// header row only for the groups that list it, so
    /// `class_word_index_in_parent` must answer exactly for those.
    #[test]
    fn every_groups_class_word_is_the_inherited_pyobject_header_slot() {
        let groups: &[(&str, DescrRef)] = &[
            ("Function", w_function_size_descr()),
            ("Method", w_method_size_descr()),
            ("W_ListObject", w_list_size_descr()),
            ("W_ObjectObject", w_object_object_size_descr()),
            ("W_ObjectObjectUser", w_object_object_user_size_descr()),
            ("W_IntObject", w_int_size_descr()),
            ("W_BoolObject", w_bool_size_descr()),
            ("W_RangeIterObject", w_range_iter_size_descr()),
            ("W_RangeObject", w_range_size_descr()),
            ("W_FloatObject", w_float_size_descr()),
            ("W_ComplexObject", w_complex_size_descr()),
            ("W_LongObject", w_long_size_descr()),
            ("W_TupleObject", w_tuple_size_descr()),
            ("W_TupleIter", tuple_iter_size_descr()),
            ("W_ListIter", list_iter_size_descr()),
            ("W_ZipObject", w_zip_size_descr()),
            ("SpecialisedTupleII", specialised_tuple_ii_size_descr()),
            ("SpecialisedTupleFF", specialised_tuple_ff_size_descr()),
            ("SpecialisedTupleOO", specialised_tuple_oo_size_descr()),
            ("PyTraceback", pytraceback_size_descr()),
            ("W_SliceObject", w_slice_size_descr()),
            ("PyFrame", pyframe_size_descr()),
        ];

        // Violations are accumulated rather than asserted in place: a test that
        // stops at the first bad group reports which one comes *first*, not
        // which ones are wrong, and the population is the finding here.
        // Every group is checked for an answer, not just for a wrong answer,
        // because counting the groups that *did* answer cannot see a group
        // that stopped: one answering group satisfies a population count while
        // every other group is dark.  What "must answer" means differs per
        // accessor and is spelled out at each one below.
        let mut violations: Vec<String> = Vec::new();
        for (label, descr) in groups {
            let size = descr
                .as_size_descr()
                .unwrap_or_else(|| panic!("{label} must be a SizeDescr"));

            // (a) the byte-offset answer names the header slot
            match size.class_word_field() {
                None => violations.push(format!(
                    "{label}: class_word_field() = None — a PyObject layout \
                     whose header row declares nothing"
                )),
                Some(fd) if fd.offset() != W_CLASS_OFFSET => violations.push(format!(
                    "{label}: class_word_field() -> `{}` at offset {} (want the \
                     inherited PyObject header at {W_CLASS_OFFSET})",
                    fd.field_name(),
                    fd.offset(),
                )),
                Some(_) => {}
            }

            // (b) the positional answer indexes that same field, and answers
            // at all exactly when this group's own positional list holds the
            // header row.  Ten of the groups below reach the header only as a
            // gc-only edge, which `with_extra_gc_fielddescr` keeps out of
            // `all_fielddescrs()` precisely so positional indexing is
            // unaffected — for those `None` is the required answer, and a
            // `Some` would mean some *payload* field claimed the class word.
            let header_is_positional = size
                .all_fielddescrs()
                .iter()
                .any(|fd| fd.offset() == W_CLASS_OFFSET);
            match (size.class_word_index_in_parent(), header_is_positional) {
                (None, true) => violations.push(format!(
                    "{label}: class_word_index_in_parent() = None while \
                     all_fielddescrs() does carry the header at \
                     {W_CLASS_OFFSET} — the slot lost its declaration and \
                     OptVirtualize's fold stops firing for this layout"
                )),
                (Some(idx), false) => violations.push(format!(
                    "{label}: class_word_index_in_parent() = {idx} while \
                     all_fielddescrs() carries no field at {W_CLASS_OFFSET} — \
                     a payload field is claiming a header row this list does \
                     not hold"
                )),
                (None, false) => {}
                (Some(idx), true) => match size.all_fielddescrs().get(idx) {
                    None => violations.push(format!(
                        "{label}: class_word_index_in_parent() = {idx} is out of \
                         range for all_fielddescrs() (len {})",
                        size.all_fielddescrs().len(),
                    )),
                    Some(fd) if fd.offset() != W_CLASS_OFFSET => violations.push(format!(
                        "{label}: class_word_index_in_parent() = {idx} names \
                             `{}` at offset {} (want the inherited PyObject \
                             header at {W_CLASS_OFFSET})",
                        fd.field_name(),
                        fd.offset(),
                    )),
                    Some(_) => {}
                },
            }
        }

        // This held two `Method` exceptions until the host gained a declaration
        // channel.  `Method` carries a payload field literally called `w_class`
        // — the class the bound function was found on, which
        // `method_w_class_descr` documents as "distinct from the inherited
        // `PyObject.w_class` naming the Method's own type" — and the group
        // factory qualifies it to `Method.w_class`, which the name fallback
        // accepts on the `.w_class` suffix.  It precedes the header row, so
        // `find()` stopped on it in both lists.
        //
        // Now `build_object_descr_group_with_extra_gc_edges` declares
        // `is_class_word: Some(offset == W_CLASS_OFFSET)`, a layout invariant a
        // name rule cannot express, and the exceptions retired.  The
        // declaration also outranks the guess on a `get_field_descr` cache hit
        // and rides `BhFieldSpec` through serialization, so neither a
        // name-only producer arriving first nor a blackhole round trip can put
        // the payload field back.
        assert!(
            violations.is_empty(),
            "{} of {} groups resolve a class word that is not the inherited \
             PyObject header — a payload field is shadowing its layout's \
             header row:\n  {}",
            violations.len(),
            groups.len(),
            violations.join("\n  "),
        );
    }

    #[test]
    fn jit_emitted_tracebacks_are_movable_but_raw_pointer_objects_are_not() {
        let traceback_descr = pytraceback_size_descr();
        let traceback_size = traceback_descr
            .as_size_descr()
            .expect("PyTraceback SizeDescr");
        assert!(
            !traceback_size.non_moving(),
            "JIT traceback refs are GC-map roots or traced object fields and must use the nursery"
        );

        let instance_descr = w_object_object_size_descr();
        let instance_size = instance_descr
            .as_size_descr()
            .expect("W_ObjectObject SizeDescr");
        assert!(
            instance_size.non_moving(),
            "raw instance pointers can survive across allocation without being rooted"
        );
        let user_instance = w_object_object_user_size_descr()
            .as_size_descr()
            .expect("W_ObjectObjectUser SizeDescr")
            .non_moving();
        assert!(
            user_instance,
            "a _getusercls carrier is the same non-moving W_ObjectObject allocation"
        );

        let storage_descr = crate::state::mapdict_storage_gcarray_descr();
        let storage_array = storage_descr
            .as_array_descr()
            .expect("mapdict storage ArrayDescr");
        assert!(
            storage_array.non_moving(),
            "the mapdict custom tracer marks raw storage pointers but cannot rewrite them"
        );
    }

    /// `W_ObjectObjectUser` is the same payload as `W_ObjectObject` with a
    /// different vtable (`typedef.py` `_getusercls`). The field Arcs stay
    /// one index space; the STRUCT key and GC tid do not.
    #[test]
    fn object_user_size_descr_shares_w_object_object_fields() {
        use std::sync::Arc;

        let exact = w_object_object_size_descr();
        let user = w_object_object_user_size_descr();
        let exact_sd = exact.as_size_descr().expect("W_ObjectObject SizeDescr");
        let user_sd = user.as_size_descr().expect("W_ObjectObjectUser SizeDescr");
        assert_eq!(
            user_sd.type_id(),
            pyre_object::objectobject::W_OBJECT_OBJECT_USER_GC_TYPE_ID
        );
        assert_eq!(
            user_sd.vtable(),
            &pyre_object::pyobject::INSTANCE_USER_TYPE as *const _ as usize
        );
        assert_eq!(
            exact_sd.type_id(),
            pyre_object::objectobject::W_OBJECT_OBJECT_GC_TYPE_ID
        );
        assert_eq!(user_sd.size(), exact_sd.size());
        assert!(user_sd.non_moving());
        assert_eq!(user.index(), super::W_OBJECT_OBJECT_USER_SIZE_INDEX);
        assert_ne!(user.index(), exact.index());
        assert_ne!(user_sd.cache_key(), 0);
        assert_ne!(user_sd.cache_key(), exact_sd.cache_key());
        assert_eq!(
            majit_ir::descr::gc_cache()
                .lock()
                .resolve_struct_tid(user_sd.cache_key()),
            Some(pyre_object::objectobject::W_OBJECT_OBJECT_USER_GC_TYPE_ID)
        );
        // map, storage, the spec class word, plus the prepended header leaf.
        assert_eq!(exact_sd.all_fielddescrs().len(), 4);
        assert_eq!(
            exact_sd.all_fielddescrs().len(),
            user_sd.all_fielddescrs().len()
        );
        for (exact_field, user_field) in exact_sd
            .all_fielddescrs()
            .iter()
            .zip(user_sd.all_fielddescrs().iter())
        {
            assert!(Arc::ptr_eq(exact_field, user_field));
        }
        assert_eq!(user_sd.class_word_index_in_parent(), Some(0));
    }

    /// The named `pyframe_*_descr` accessors pick their descr by position in
    /// `PYFRAME_DESCR_GROUP`, so removing a field from the group shifts every
    /// later position.  A stale index hands out a neighbouring field's descr:
    /// after `lastblock` left the group, `pyframe_f_backref_descr` answered
    /// `w_builtin`, and every JIT-built frame lost its caller chain and its
    /// builtins.
    #[test]
    fn pyframe_field_descr_accessors_name_their_own_offsets() {
        use crate::frame_layout as fl;
        let offset = |descr: DescrRef| descr.as_field_descr().expect("PyFrame FieldDescr").offset();
        let cases: [(&str, DescrRef, usize); 11] = [
            (
                "locals_cells_stack_w",
                pyframe_locals_cells_stack_descr(),
                fl::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
            ),
            (
                "valuestackdepth",
                pyframe_stack_depth_descr(),
                fl::PYFRAME_VALUESTACKDEPTH_OFFSET,
            ),
            (
                "last_instr",
                pyframe_next_instr_descr(),
                fl::PYFRAME_LAST_INSTR_OFFSET,
            ),
            ("pycode", pyframe_code_descr(), fl::PYFRAME_PYCODE_OFFSET),
            (
                "debugdata",
                pyframe_debugdata_descr(),
                fl::PYFRAME_DEBUGDATA_OFFSET,
            ),
            (
                "f_generator_wref",
                pyframe_f_generator_wref_descr(),
                fl::PYFRAME_F_GENERATOR_WREF_OFFSET,
            ),
            (
                "w_yielding_from",
                pyframe_w_yielding_from_descr(),
                fl::PYFRAME_W_YIELDING_FROM_OFFSET,
            ),
            (
                "f_backref",
                pyframe_f_backref_descr(),
                fl::PYFRAME_F_BACKREF_OFFSET,
            ),
            (
                "w_builtin",
                pyframe_w_builtin_descr(),
                fl::PYFRAME_W_BUILTIN_OFFSET,
            ),
            ("flags", pyframe_flags_descr(), fl::PYFRAME_FLAGS_OFFSET),
            (
                "failed_attr_cleanup",
                pyframe_failed_attr_cleanup_descr(),
                fl::PYFRAME_FAILED_ATTR_CLEANUP_OFFSET,
            ),
        ];
        for (name, descr, want) in cases {
            assert_eq!(offset(descr), want, "pyframe descr accessor for {name}");
        }
    }

    #[test]
    fn pyframe_size_descr_clears_the_vable_token_slot() {
        let descr = pyframe_size_descr();
        let size = descr.as_size_descr().expect("PyFrame SizeDescr");
        assert!(
            size.gc_fielddescrs()
                .iter()
                .any(|fd| fd.offset() == crate::frame_layout::PYFRAME_VABLE_TOKEN_OFFSET),
            "emit_force_virtualizable reads vable_token with GETFIELD_GC_R, so \
             clear_gc_fields must zero it on a JIT-inlined frame allocation"
        );
    }

    #[test]
    fn exception_size_descr_clears_every_runtime_traced_gc_field() {
        let (descr, _, _, _) = w_exception_descrs(ExcKind::BaseException);
        let size = descr.as_size_descr().expect("W_BaseException SizeDescr");
        assert_eq!(size.size(), W_BASE_EXCEPTION_SIZE);
        let mut actual: Vec<usize> = size.gc_fielddescrs().iter().map(|fd| fd.offset()).collect();
        actual.sort_unstable();
        actual.dedup();

        let mut expected = W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec();
        expected.push(W_CLASS_OFFSET);
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(actual, expected);

        let (user, _, _, _) = w_exception_descrs(ExcKind::ValueError);
        let user = user.as_size_descr().expect("W_BaseExceptionUser SizeDescr");
        assert_eq!(user.size(), W_BASE_EXCEPTION_USER_SIZE);
        let mut user_actual: Vec<usize> =
            user.gc_fielddescrs().iter().map(|fd| fd.offset()).collect();
        user_actual.sort_unstable();
        user_actual.dedup();
        let mut user_expected = W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec();
        user_expected.push(EXC_USER_STORAGE_OFFSET);
        user_expected.push(W_CLASS_OFFSET);
        user_expected.sort_unstable();
        user_expected.dedup();
        assert_eq!(user_actual, user_expected);
    }

    /// `heaptracker.py all_fielddescrs` skips `typeptr` and recurses into the
    /// inlined header, so the nested `PyObject.w_class` leaf is
    /// `all_fielddescrs[0]` and `all_fielddescrs(S)[i].get_index() == i`.
    /// The spec list still ends with its own class word.
    #[test]
    fn w_base_exception_field_indices_match_all_fielddescrs_order() {
        let (descr, _, _, _) = w_exception_descrs(ExcKind::BaseException);
        let fields = descr
            .as_size_descr()
            .expect("W_BaseException SizeDescr")
            .all_fielddescrs();
        // Leading nested `PyObject.w_class`, then seven declaration-order
        // fields, then the spec's own class word.
        assert_eq!(fields.len(), 9);
        for (i, field) in fields.iter().enumerate() {
            assert_eq!(
                field.index_in_parent(),
                i,
                "{} at offset {} has index_in_parent {}, not {i}",
                field.field_name(),
                field.offset(),
                field.index_in_parent(),
            );
        }
        // The trailing entry is the spec class word at `W_CLASS_OFFSET`.
        // The leading entry is that same offset, already below `kind`, so
        // only the span that excludes the trailing word has to increase.
        let last_increasing = fields.len() - 2;
        for i in 0..last_increasing {
            assert!(
                fields[i].offset() < fields[i + 1].offset(),
                "offsets 0..={last_increasing} must be strictly increasing: {}@{} then {}@{}",
                fields[i].field_name(),
                fields[i].offset(),
                fields[i + 1].field_name(),
                fields[i + 1].offset(),
            );
        }
        let class_word = fields.last().expect("group is non-empty");
        assert_eq!(class_word.offset(), W_CLASS_OFFSET);
        assert!(class_word.is_w_class());

        let (user, _, _, _) = w_exception_descrs(ExcKind::ValueError);
        let user_size = user.as_size_descr().expect("W_BaseExceptionUser SizeDescr");
        assert_eq!(user_size.size(), W_BASE_EXCEPTION_USER_SIZE);
        assert_eq!(user_size.type_id(), W_BASE_EXCEPTION_USER_GC_TYPE_ID);
        let user_fields = user_size.all_fielddescrs();
        // Leading class word, then the slim payload, then `map` and
        // `storage`, then the spec class word.
        assert_eq!(user_fields.len(), 11);
        assert_eq!(user_fields[8].offset(), EXC_USER_MAP_OFFSET);
        // Display name is `STRUCT._name + '.' + fieldname` (`get_field_descr`).
        assert_eq!(user_fields[8].field_name(), "W_BaseExceptionUser.map");
        assert_eq!(user_fields[9].offset(), EXC_USER_STORAGE_OFFSET);
        assert_eq!(user_fields[9].field_name(), "W_BaseExceptionUser.storage");
        assert_ne!(user_fields[8].index(), user_fields[9].index());
        let user_class = user_fields.last().expect("user group is non-empty");
        assert_eq!(user_class.offset(), W_CLASS_OFFSET);
        assert!(user_class.is_w_class());
    }

    #[test]
    fn oserror_size_descr_uses_the_extended_layout() {
        let (descr, _, _, _) = w_exception_descrs(ExcKind::OSError);
        let size = descr
            .as_size_descr()
            .expect("W_ExceptionExtended SizeDescr");
        assert_eq!(size.size(), W_EXCEPTION_EXTENDED_SIZE);
        assert_eq!(size.type_id(), W_EXCEPTION_EXTENDED_GC_TYPE_ID);
        assert!(size.size() > W_BASE_EXCEPTION_SIZE);
        let (user, _, _, _) = w_exception_descrs_for(ExcKind::FileNotFoundError, true);
        let user = user
            .as_size_descr()
            .expect("W_ExceptionExtendedUser SizeDescr");
        assert_eq!(user.size(), W_EXCEPTION_EXTENDED_USER_SIZE);
        assert_eq!(user.type_id(), W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID);
    }

    #[test]
    fn test_field_descr_indices_are_stable_and_distinct() {
        let a = make_field_descr(8, 8, Type::Int, false);
        let b = make_field_descr(8, 8, Type::Int, false);
        let c = make_field_descr(16, 8, Type::Int, false);

        assert_eq!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
    }

    #[test]
    fn ec_w_tracefunc_quasi_index_is_not_a_function_slot() {
        let idx = ec_w_tracefunc_descr().index();
        assert_eq!(idx, EC_W_TRACEFUNC_INDEX);
        assert_ne!(idx, 2);
        assert_ne!(idx, function_code_descr().index());
        assert!(function_quasi_immut_slot(idx).is_none());
        assert!(function_quasi_immut_slot(function_code_descr().index()).is_some());
    }

    #[test]
    fn ec_profilefunc_quasi_index_is_not_a_function_slot() {
        let idx = ec_profilefunc_descr().index();
        assert_eq!(idx, EC_PROFILEFUNC_INDEX);
        assert_ne!(idx, EC_W_TRACEFUNC_INDEX);
        assert!(function_quasi_immut_slot(idx).is_none());
    }

    #[test]
    fn test_array_descr_indices_are_distinct_per_call() {
        // PyPy `descr.py:350-351 cache[ARRAY_OR_STRUCT]` keys on ARRAY
        // object identity; `make_array_descr` callers without an
        // identity carrier (`array_type_id = None`) each produce a
        // distinct ARRAY → distinct `descr_id`.  Singleton semantics
        // require routing through `make_array_descr_with_full_id` with
        // a stable identity string instead.
        let a = make_array_descr(0, 8, None, Type::Int, false);
        let b = make_array_descr(0, 8, None, Type::Int, false);
        let c = make_array_descr(0, 8, None, Type::Ref, false);

        assert_ne!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
        assert_ne!(b.index(), c.index());
    }

    #[test]
    fn test_array_descr_with_full_id_singleton_per_identity() {
        // `descr.py get_array_descr` cache hit on
        // `LLType::Array(path_hash(atid))` returns the existing Arc
        // — `make_array_descr_with_full_id` with the same identity
        // string is a singleton.
        let a = crate::descr::make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Int,
            false,
            Some("pyre::test_singleton_id".to_string()),
        );
        let b = crate::descr::make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Int,
            false,
            Some("pyre::test_singleton_id".to_string()),
        );
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "same identity carrier must collapse to the same Arc"
        );
    }

    #[test]
    fn make_call_descr_from_bh_round_trips_most_general_effectinfo() {
        use majit_ir::EffectInfo;
        use majit_jitcode::jitcode::BhCallDescr;

        let bh = BhCallDescr::from_arg_classes("r".to_string(), 'r', EffectInfo::MOST_GENERAL);

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Ref]);
        assert_eq!(call.result_type(), Type::Ref);
        assert_eq!(call.result_size(), 8);
        assert!(!call.is_result_signed());
        assert_eq!(call.get_extra_info(), &EffectInfo::MOST_GENERAL);
        assert!(call.get_extra_info().check_can_raise(false));
    }

    #[test]
    fn make_call_descr_from_bh_round_trips_cannot_raise_effectinfo() {
        use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex};
        use majit_jitcode::jitcode::BhCallDescr;

        let extra_info = EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);
        let bh = BhCallDescr::from_arg_classes("ir".to_string(), 'v', extra_info.clone());

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Int, Type::Ref]);
        assert_eq!(call.result_type(), Type::Void);
        assert_eq!(call.result_size(), 0);
        assert_eq!(call.get_extra_info(), &extra_info);
        assert!(!call.get_extra_info().check_can_raise(false));
    }

    #[test]
    fn make_call_descr_from_bh_preserves_singlefloat_result_layout() {
        use majit_ir::EffectInfo;
        use majit_jitcode::jitcode::{BhCallDescr, CallResultErasedKey};

        let bh = BhCallDescr::from_arg_classes("S".to_string(), 'S', EffectInfo::MOST_GENERAL);

        assert_eq!(bh.arg_classes, "S");
        assert_eq!(bh.result_type, 'S');
        assert_eq!(bh.result_size, 4);
        assert!(!bh.result_signed);
        assert_eq!(bh.result_erased, CallResultErasedKey::SingleFloat);

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Int]);
        assert_eq!(call.result_type(), Type::Int);
        // descr.py `get_result_type()` parity — the raw 'S' char
        // must survive the BhCallDescr -> CallDescr conversion, so
        // downstream consumers can distinguish singlefloat from a real
        // int result.  pyre's `result_class()` returns the raw char
        // (matches `descr.py get_result_type()`); the normalized
        // form per descr.py (collapsing 'S' → 'i') is not yet
        // exposed as a separate method but the underlying `result_type`
        // is already `Type::Int`, which is the normalized view.
        assert_eq!(call.result_class(), 'S');
        assert_eq!(call.result_size(), 4);
        assert!(!call.is_result_signed());
    }

    /// A class-word declaration must survive serialization into a blackhole
    /// spec and back.
    ///
    /// `BhFieldSpec` is the only channel between a layout producer and the
    /// descriptors rebuilt from `descrs.bin`, so a flag it does not carry is a
    /// flag the far side has to guess — and the guess
    /// (`class_word_inferred_from_name`) reads a name, which a *payload* field
    /// can spell exactly like its layout's header row. The synthetic layout
    /// below is that shape: the payload at offset 24 is the one the name rule
    /// picks, and the header at offset 8 is the one the layout declares.
    #[test]
    fn a_class_word_declaration_survives_the_blackhole_round_trip() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::{BhFieldSpec, BhSizeSpec};

        let spec = |field_key: &str,
                    name: &str,
                    offset: usize,
                    index_in_parent: usize,
                    is_class_word: bool| {
            majit_ir::descr::SimpleFieldDescrSpec {
                index: index_in_parent as u32,
                field_key: field_key.to_string(),
                name: name.to_string(),
                offset,
                field_size: 8,
                field_type: Type::Ref,
                is_immutable: false,
                is_quasi_immutable: false,
                flag: ArrayFlag::Pointer,
                virtualizable: false,
                index_in_parent,
                is_class_word: Some(is_class_word),
            }
        };
        let group = majit_ir::descr::make_simple_descr_group(
            u32::MAX,
            32,
            0,
            0,
            &[
                spec("w_class", "Synth.w_class", 24, 0, false),
                spec("header", "Synth.header", 8, 1, true),
            ],
        );
        assert_eq!(
            group.size_descr.class_word_field().map(|fd| fd.offset()),
            Some(8),
            "the declaration outranks the name on the producing side"
        );

        let wire: Vec<BhFieldSpec> = group
            .size_descr
            .all_fielddescrs()
            .iter()
            .map(|fd| BhFieldSpec::from_field_descr(fd.as_ref()))
            .collect();
        assert_eq!(
            (wire[0].is_class_word, wire[1].is_class_word),
            (Some(false), Some(true)),
            "the wire must carry the declaration, not the name's answer"
        );

        let bytes = bincode::serialize(&wire).expect("BhFieldSpec is a wire record");
        let wire: Vec<BhFieldSpec> = bincode::deserialize(&bytes).expect("round trip");

        // A `type_id` used by no other test: `simple_descr_group_from_bh_size`
        // publishes into the process-global `gc_cache`, and an existing slot
        // would answer from the producing side and prove nothing.
        let rebuilt = simple_descr_group_from_bh_size(&BhSizeSpec {
            size: 32,
            type_id: 0xc1a5_0000_57ea_d001,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: wire,
        });
        assert_eq!(
            rebuilt.size_descr.class_word_field().map(|fd| fd.offset()),
            Some(8),
            "a rebuilt descr that re-guessed from the name would answer the \
             payload field at offset 24"
        );
        assert_eq!(
            rebuilt.size_descr.class_word_index_in_parent(),
            Some(1),
            "the positional answer must index the declared header row"
        );
    }

    #[test]
    fn make_descr_from_bh_field_preserves_parent_name_index() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::{BhDescr, BhFieldSpec, BhSizeSpec};

        let parent = BhSizeSpec {
            size: 24,
            type_id: 7,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![
                BhFieldSpec {
                    index: 0,
                    field_key: "next".into(),
                    name: "Cell.next".into(),
                    offset: 8,
                    field_size: 8,
                    field_type: Type::Ref,
                    field_flag: ArrayFlag::Pointer,
                    is_field_signed: false,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                    is_class_word: None,
                },
                BhFieldSpec {
                    index: 1,
                    field_key: "value".into(),
                    name: "Cell.value".into(),
                    offset: 16,
                    field_size: 8,
                    field_type: Type::Int,
                    field_flag: ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    index_in_parent: 1,
                    is_class_word: None,
                },
            ],
        };

        let descr = make_descr_from_bh(&BhDescr::Field {
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            field_flag: ArrayFlag::Signed,
            is_field_signed: true,
            is_immutable: true,
            is_quasi_immutable: false,
            index_in_parent: Some(1),
            parent: Some(std::sync::Arc::new(parent)),
            name: "value".into(),
            owner: "Cell".into(),
        });
        let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");

        assert_eq!(field.field_name(), "Cell.value");
        assert_eq!(field.index_in_parent(), 1);
        assert_eq!(field.offset(), 16);
        assert!(field.is_immutable());
        let parent = field
            .get_parent_descr()
            .expect("FieldDescr.parent_descr must be preserved");
        let size = parent
            .as_size_descr()
            .expect("parent_descr must be a SizeDescr");
        assert_eq!(size.size(), 24);
        assert_eq!(size.type_id(), 7);
        assert_eq!(size.all_fielddescrs().len(), 2);
        assert_eq!(size.all_fielddescrs()[1].field_name(), "Cell.value");
    }

    use majit_ir::descr::ArrayFlag;
    use majit_jitcode::jitcode::{BhDescr, BhFieldSpec, BhSizeSpec};

    fn fat_parent(type_id: u64, field_size: usize) -> std::sync::Arc<BhSizeSpec> {
        std::sync::Arc::new(BhSizeSpec {
            size: 232,
            type_id,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![BhFieldSpec {
                index: 0,
                field_key: "localspluskinds".into(),
                name: "CodeObject.localspluskinds".into(),
                offset: 216,
                field_size,
                field_type: Type::Ref,
                field_flag: ArrayFlag::Pointer,
                is_field_signed: false,
                is_immutable: true,
                is_quasi_immutable: false,
                index_in_parent: 0,
                is_class_word: None,
            }],
        })
    }

    fn bh_field(
        parent: &std::sync::Arc<BhSizeSpec>,
        name: &str,
        offset: usize,
        field_size: usize,
        field_type: Type,
        field_flag: ArrayFlag,
        index_in_parent: Option<usize>,
    ) -> DescrRef {
        make_descr_from_bh(&BhDescr::Field {
            offset,
            field_size,
            field_type,
            field_flag,
            is_field_signed: false,
            is_immutable: true,
            is_quasi_immutable: false,
            index_in_parent,
            parent: Some(std::sync::Arc::clone(parent)),
            name: name.into(),
            owner: "CodeObject".into(),
        })
    }

    /// `fielddescrof` stamps one parent slot on both fat-pointer words.
    /// `HeapCache` keys by `Descr::index()`, so the words need different keys
    /// when that slot does not describe both accesses.
    #[test]
    fn make_descr_from_bh_fat_parts_do_not_share_heapcache_index() {
        let wide = fat_parent(0xFA70_DA7A_0000_0001, 16);
        let data = bh_field(
            &wide,
            "localspluskinds.data",
            216,
            8,
            Type::Ref,
            ArrayFlag::Pointer,
            Some(0),
        );
        let len = bh_field(
            &wide,
            "localspluskinds.len",
            224,
            8,
            Type::Int,
            ArrayFlag::Unsigned,
            Some(0),
        );
        assert_ne!(data.index(), len.index());
        assert_ne!(data.index(), 0);
        assert_ne!(len.index(), 0);
        assert_eq!(data.as_field_descr().unwrap().offset(), 216);
        assert_eq!(len.as_field_descr().unwrap().offset(), 224);

        // The data word is the parent field itself. It keeps the slot so a
        // read of the field and a read of `.data` stay one cache entry.
        let exact = fat_parent(0xFA70_DA7A_0000_0002, 8);
        let data = bh_field(
            &exact,
            "localspluskinds.data",
            216,
            8,
            Type::Ref,
            ArrayFlag::Pointer,
            Some(0),
        );
        let len = bh_field(
            &exact,
            "localspluskinds.len",
            224,
            8,
            Type::Int,
            ArrayFlag::Unsigned,
            Some(0),
        );
        assert_eq!(data.index(), 0);
        assert_ne!(len.index(), 0);
        assert_ne!(data.index(), len.index());

        // A stale slot on a field that is not a split part keeps the position.
        // Only the suffixed vec/fat accesses move.
        let stale_parent = std::sync::Arc::new(BhSizeSpec {
            size: 32,
            type_id: 0xFA70_DA7A_0000_0003,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![BhFieldSpec {
                index: 0,
                field_key: "buffer".into(),
                name: "Owner.buffer".into(),
                offset: 16,
                field_size: 8,
                field_type: Type::Ref,
                field_flag: ArrayFlag::Pointer,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
                is_class_word: None,
            }],
        });
        let count = bh_field(
            &stale_parent,
            "count",
            24,
            8,
            Type::Int,
            ArrayFlag::Unsigned,
            Some(0),
        );
        assert_eq!(count.index(), 0);

        // `&dyn` data word `imp` and vtable word `imp.vtable` share one slot.
        // The vtable read must not reuse the data word's heapcache key.
        let dyn_parent = std::sync::Arc::new(BhSizeSpec {
            size: 32,
            type_id: 0xFA70_DA7A_0000_0006,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![BhFieldSpec {
                index: 1,
                field_key: "imp".into(),
                name: "DictStrategyRef.imp".into(),
                offset: 8,
                field_size: 8,
                field_type: Type::Ref,
                field_flag: ArrayFlag::Pointer,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
                is_class_word: None,
            }],
        });
        let imp = bh_field(
            &dyn_parent,
            "imp",
            8,
            8,
            Type::Ref,
            ArrayFlag::Pointer,
            Some(1),
        );
        let vtable = bh_field(
            &dyn_parent,
            "imp.vtable",
            16,
            8,
            Type::Ref,
            ArrayFlag::Pointer,
            Some(1),
        );
        assert_eq!(imp.index(), 1);
        assert_ne!(vtable.index(), imp.index());
        assert_eq!(vtable.as_field_descr().unwrap().offset(), 16);
    }

    /// The tracing pool resolves a field through `field_descr_ref_from_bh`,
    /// not `make_descr_from_bh`.  The same two fat-pointer words have to
    /// leave that path with different `Descr::index()` values, because that
    /// is the key `HeapCache` consults while the walk records the read.
    #[test]
    fn field_descr_ref_from_bh_fat_parts_do_not_share_heapcache_index() {
        fn walk(
            parent: &std::sync::Arc<BhSizeSpec>,
            name: &str,
            offset: usize,
            field_size: usize,
            field_type: Type,
            field_flag: ArrayFlag,
        ) -> DescrRef {
            majit_metainterp::field_descr_ref_from_bh(&BhDescr::Field {
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed: false,
                is_immutable: true,
                is_quasi_immutable: false,
                index_in_parent: Some(0),
                parent: Some(std::sync::Arc::clone(parent)),
                name: name.into(),
                owner: "CodeObject".into(),
            })
            .1
        }

        let wide = fat_parent(0xFA70_DA7A_0000_0004, 16);
        let data = walk(
            &wide,
            "localspluskinds.data",
            216,
            8,
            Type::Ref,
            ArrayFlag::Pointer,
        );
        let len = walk(
            &wide,
            "localspluskinds.len",
            224,
            8,
            Type::Int,
            ArrayFlag::Unsigned,
        );
        assert_ne!(data.index(), len.index());
        assert_ne!(data.index(), 0);
        assert_ne!(len.index(), 0);

        let exact = fat_parent(0xFA70_DA7A_0000_0005, 8);
        let data = walk(
            &exact,
            "localspluskinds.data",
            216,
            8,
            Type::Ref,
            ArrayFlag::Pointer,
        );
        let len = walk(
            &exact,
            "localspluskinds.len",
            224,
            8,
            Type::Int,
            ArrayFlag::Unsigned,
        );
        assert_eq!(data.index(), 0);
        assert_ne!(len.index(), data.index());

        let dyn_parent = std::sync::Arc::new(BhSizeSpec {
            size: 32,
            type_id: 0xFA70_DA7A_0000_0007,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![BhFieldSpec {
                index: 1,
                field_key: "imp".into(),
                name: "DictStrategyRef.imp".into(),
                offset: 8,
                field_size: 8,
                field_type: Type::Ref,
                field_flag: ArrayFlag::Pointer,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
                is_class_word: None,
            }],
        });
        let imp = majit_metainterp::field_descr_ref_from_bh(&BhDescr::Field {
            offset: 8,
            field_size: 8,
            field_type: Type::Ref,
            field_flag: ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(1),
            parent: Some(std::sync::Arc::clone(&dyn_parent)),
            name: "imp".into(),
            owner: "DictStrategyRef".into(),
        })
        .1;
        let vtable = majit_metainterp::field_descr_ref_from_bh(&BhDescr::Field {
            offset: 16,
            field_size: 8,
            field_type: Type::Ref,
            field_flag: ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(1),
            parent: Some(dyn_parent),
            name: "imp.vtable".into(),
            owner: "DictStrategyRef".into(),
        })
        .1;
        assert_eq!(imp.index(), 1);
        assert_ne!(vtable.index(), imp.index());
        assert_eq!(vtable.as_field_descr().unwrap().offset(), 16);
    }

    /// A build-time `setarrayitem_gc` on a list's int block and the walker's
    /// own `getarrayitem_gc_i` name one ArrayDescr, or the heap cache keeps
    /// the pre-store read alive across the store.
    #[test]
    fn make_descr_from_bh_list_int_block_is_the_runtime_gcarray_descr() {
        use majit_jitcode::jitcode::BhDescr;
        let token = &pyre_object::TYPED_ITEMS_BLOCK_INT_TOKEN;
        let atid = majit_jitcode::codewriter::jtransform::LIST_INT_ITEMS_ARRAY;
        let bridged = make_descr_from_bh(&BhDescr::Array {
            base_size: token.base_size,
            itemsize: token.item_size,
            len_offset: Some(token.len_offset),
            type_id: majit_ir::descr::path_hash(atid),
            gc_type_id: 0,
            item_type: Type::Int,
            is_array_of_pointers: false,
            is_array_of_structs: false,
            is_item_signed: true,
            is_gc_managed: true,
            ei_index: u32::MAX,
            array_type_id: Some(atid.to_string()),
            interior_fields: Vec::new(),
        });
        let runtime = crate::state::int_gcarray_descr();
        assert!(std::sync::Arc::ptr_eq(&bridged, &runtime));
        // `BhDescr::from_array_descr` drops the spelling; `cache_key` alone
        // still finds the published descr.
        let round_trip = make_descr_from_bh(&BhDescr::from_array_descr(
            runtime.as_array_descr().expect("ArrayDescr"),
        ));
        assert!(std::sync::Arc::ptr_eq(&round_trip, &runtime));
        assert_ne!(runtime.as_array_descr().unwrap().type_id(), 0);
        // `opimpl_new_array` allocates from `BhDescr::resolve_gc_tid` with
        // `gc_type_id: 0`; that lookup must recover the runtime tid, not
        // a truncated `path_hash`.
        assert_eq!(
            BhDescr::Array {
                base_size: token.base_size,
                itemsize: token.item_size,
                len_offset: Some(token.len_offset),
                type_id: majit_ir::descr::path_hash(atid),
                gc_type_id: 0,
                item_type: Type::Int,
                is_array_of_pointers: false,
                is_array_of_structs: false,
                is_item_signed: true,
                is_gc_managed: true,
                ei_index: u32::MAX,
                array_type_id: Some(atid.to_string()),
                interior_fields: Vec::new(),
            }
            .resolve_gc_tid(),
            runtime.as_array_descr().unwrap().type_id(),
        );
    }

    /// A traced integer-dict `NEW_ARRAY_CLEAR` must keep the collector
    /// tid `register_dict_entries` published, not the sequential id
    /// `init_array_descr` mints for an unknown ARRAY.
    ///
    /// `BhDescr.gc_type_id` is 0 (legacy serialized rows). The runtime
    /// ARRAY slot is consulted first, so the bridged descr is that Arc
    /// and carries whatever tid `i64_pyobject_entries_gc_type_id` has
    /// already published. Do not overwrite the process-global atomic:
    /// `runtime_gcarray_descr` stamps `_cache_array` in place.
    #[test]
    fn make_descr_from_bh_i64_entries_uses_registered_gc_tid() {
        use majit_jitcode::jitcode::BhDescr;
        type K = i64;
        type V = pyre_object::PyObjectRef;
        let atid = "GcArray<pyre_object::rordereddict_entries::Entry<i64,*mut PyObject>>";
        let runtime = crate::state::runtime_gcarray_descr(atid).expect("DICTENTRYARRAY descr");
        let bridged = make_descr_from_bh(&BhDescr::Array {
            base_size: std::mem::offset_of!(pyre_object::rordereddict::GcEntries<K, V>, items),
            itemsize: std::mem::size_of::<pyre_object::rordereddict::Entry<K, V>>(),
            len_offset: Some(std::mem::offset_of!(
                pyre_object::rordereddict::GcEntries<K, V>,
                length
            )),
            type_id: majit_ir::descr::path_hash(atid),
            gc_type_id: 0,
            item_type: Type::Ref,
            is_array_of_pointers: false,
            is_array_of_structs: true,
            is_item_signed: false,
            is_gc_managed: true,
            ei_index: u32::MAX,
            array_type_id: Some(atid.to_string()),
            interior_fields: Vec::new(),
        });
        assert!(std::sync::Arc::ptr_eq(&bridged, &runtime));
        let array = bridged.as_array_descr().expect("ArrayDescr");
        let published = pyre_object::rordereddict::i64_pyobject_entries_gc_type_id();
        if published != 0 {
            assert_eq!(array.type_id(), published);
        }
        assert!(array.is_array_of_structs());
    }

    /// An entry-struct array whose parent `SizeDescr` was minted before its
    /// fields existed still has to publish every interior slot
    /// `opcode_published_descr_members` will stamp.
    #[test]
    fn empty_owner_interior_fields_publish_the_effectinfo_slot() {
        use majit_ir::descr::{LLType, gc_cache};
        use majit_ir::effectinfo::DescrSetMember;
        use majit_jitcode::jitcode::{BhDescr, BhFieldSpec, BhInteriorFieldSpec, BhSizeSpec};

        let owner_id =
            majit_ir::descr::path_hash("pyre_jit_trace::descr::tests::empty_owner_interior::Entry");
        let array_id = majit_ir::descr::path_hash(
            "pyre_jit_trace::descr::tests::empty_owner_interior::GcArray",
        );
        assert_ne!(owner_id, 0);
        assert_ne!(array_id, 0);
        let owner = BhSizeSpec {
            owner: String::new(),
            size: 32,
            type_id: owner_id,
            vtable: 0,
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: Vec::new(),
        };
        let spec =
            |index_in_parent: usize, field_key: &str, offset: usize, field_type, field_flag| {
                BhFieldSpec {
                    index: index_in_parent as u32,
                    field_key: field_key.into(),
                    name: format!("T{owner_id}.{field_key}"),
                    offset,
                    field_size: 8,
                    field_type,
                    field_flag,
                    is_field_signed: false,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent,
                    is_class_word: None,
                }
            };
        let key = spec(0, "key", 0, Type::Ref, majit_ir::descr::ArrayFlag::Pointer);
        let hash = spec(
            3,
            "f_hash",
            24,
            Type::Int,
            majit_ir::descr::ArrayFlag::Unsigned,
        );
        let interior = |index, field: BhFieldSpec| BhInteriorFieldSpec {
            index,
            field,
            owner: owner.clone(),
        };
        let _restored = make_descr_from_bh(&BhDescr::InteriorField {
            array: Box::new(BhDescr::Array {
                base_size: 8,
                itemsize: 32,
                len_offset: Some(0),
                type_id: array_id,
                gc_type_id: 0,
                item_type: Type::Ref,
                is_array_of_pointers: false,
                is_array_of_structs: true,
                is_item_signed: false,
                is_gc_managed: true,
                ei_index: u32::MAX,
                array_type_id: None,
                interior_fields: vec![interior(0, key), interior(3, hash)],
            }),
            field: Box::new(BhDescr::Field {
                offset: 8,
                field_size: 1,
                field_type: Type::Int,
                field_flag: majit_ir::descr::ArrayFlag::Unsigned,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: Some(1),
                parent: None,
                name: "f_valid".into(),
                owner: String::new(),
            }),
        });

        let stamp = |name: &str, ei_index: u32| {
            super::stamp_effect_info_descr(
                &DescrSetMember::InteriorField {
                    array_id,
                    name: name.to_string(),
                },
                ei_index,
            );
        };
        stamp("f_hash", 11);
        stamp("key", 12);

        let gc = gc_cache().lock();
        assert!(
            !gc._cache_size.contains_key(&LLType::Struct(owner_id)),
            "an empty owner list must not register a field-less size",
        );
        let stored = |name: &str| {
            gc._cache_interiorfield
                .get(&(LLType::Array(array_id), name.to_string(), String::new()))
                .unwrap_or_else(|| panic!("{name} interior slot missing"))
                .clone()
        };
        let hash_descr = stored("f_hash");
        let key_descr = stored("key");
        drop(gc);
        assert_eq!(hash_descr.get_ei_index(), 11);
        assert_eq!(key_descr.get_ei_index(), 12);
        let hash_field = hash_descr
            .as_interior_field_descr()
            .expect("f_hash InteriorFieldDescr")
            .field_descr();
        assert_eq!(hash_field.offset(), 24);
        assert_eq!(hash_field.index_in_parent(), 3);
        let key_field = key_descr
            .as_interior_field_descr()
            .expect("key InteriorFieldDescr")
            .field_descr();
        assert_eq!(key_field.offset(), 0);
        assert_eq!(key_field.index_in_parent(), 0);
    }

    #[test]
    fn make_descr_from_bh_items_block_capacity_with_parent_is_canonical() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::{BhDescr, BhFieldSpec, BhSizeSpec};

        let capacity = BhFieldSpec {
            index: 0,
            field_key: "capacity".into(),
            name: "ItemsBlock.capacity".into(),
            offset: 0,
            field_size: std::mem::size_of::<usize>(),
            field_type: Type::Int,
            field_flag: ArrayFlag::Unsigned,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            is_class_word: None,
        };
        let parent = BhSizeSpec {
            size: std::mem::size_of::<usize>(),
            type_id: 3938139489201595032,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![capacity.clone()],
        };
        let descr = make_descr_from_bh(&BhDescr::Field {
            offset: capacity.offset,
            field_size: capacity.field_size,
            field_type: capacity.field_type,
            field_flag: capacity.field_flag,
            is_field_signed: capacity.is_field_signed,
            is_immutable: capacity.is_immutable,
            is_quasi_immutable: capacity.is_quasi_immutable,
            index_in_parent: Some(capacity.index_in_parent),
            parent: Some(std::sync::Arc::new(parent)),
            name: "capacity".into(),
            owner: "ItemsBlock".into(),
        });
        let canonical = items_block_capacity_descr();

        assert!(
            std::sync::Arc::ptr_eq(&descr, &canonical),
            "the build-time parent must not mint a second capacity FieldDescr",
        );
        assert_eq!(descr.index(), canonical.index());
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_strategy_items_leaves_to_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        // Shape `_handle_list_call` emits and the codewriter assembler
        // round-trips: dotted nested name, owner `W_ListObject`, offset 0
        // (unresolved, because `W_ListObject` is not in the codewriter
        // struct layouts), no parent. Without the bridge these mint a
        // SimpleFieldDescr at offset 0 (the list header).
        for (name, expected, ty) in [
            ("int_items.len", list_int_items_len_descr(), Type::Int),
            ("int_items.block", list_int_items_block_descr(), Type::Ref),
            ("float_items.len", list_float_items_len_descr(), Type::Int),
            (
                "float_items.block",
                list_float_items_block_descr(),
                Type::Ref,
            ),
            ("bytes_items.len", list_bytes_items_len_descr(), Type::Int),
            (
                "bytes_items.block",
                list_bytes_items_block_descr(),
                Type::Ref,
            ),
            ("ascii_items.len", list_ascii_items_len_descr(), Type::Int),
            (
                "ascii_items.block",
                list_ascii_items_block_descr(),
                Type::Ref,
            ),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: ty,
                field_flag: ArrayFlag::Signed,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: Some(0),
                parent: None,
                name: name.into(),
                owner: "W_ListObject".into(),
            });
            let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");
            assert_ne!(field.offset(), 0, "{name} bridged to the list header");
            assert_eq!(
                field.offset(),
                expected.as_field_descr().unwrap().offset(),
                "{name} offset must match the W_LIST_DESCR_GROUP entry",
            );
        }
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_bare_items_aliases_to_block_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        // A bare `int_items` / `float_items` / `bytes_items` read (the
        // `w_list_append` body
        // reads the typed-storage struct base before reaching `.ptr`/`.len`)
        // must bridge to the same canonical `.block` group entry as the dotted
        // `.block` leaf — a populated parent_descr and the `.block` offset, not
        // the offset-0 list header.
        for (name, expected) in [
            ("int_items", list_int_items_block_descr()),
            ("float_items", list_float_items_block_descr()),
            ("bytes_items", list_bytes_items_block_descr()),
            ("ascii_items", list_ascii_items_block_descr()),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: Type::Ref,
                field_flag: ArrayFlag::Signed,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: Some(0),
                parent: None,
                name: name.into(),
                owner: "W_ListObject".into(),
            });
            let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");
            assert!(
                field.get_parent_descr().is_some(),
                "{name} must carry a parent_descr after the bridge",
            );
            assert_eq!(
                field.offset(),
                expected.as_field_descr().unwrap().offset(),
                "{name} offset must match the `.block` W_LIST_DESCR_GROUP entry",
            );
        }
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_list_header_fields_to_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        // The `w_list_append` body reads `list.{strategy,length,items}`
        // directly. The codewriter mints these as a `SimpleFieldDescr` with no
        // parent_descr backreference, which `ensure_ptr_info_arg0` rejects.
        // The bridge must route them to the canonical group entries so
        // `get_parent_descr()` is populated (and the offset matches).
        for (name, expected, ty) in [
            ("strategy", list_strategy_descr(), Type::Int),
            ("length", list_length_descr(), Type::Int),
            ("items", list_items_descr(), Type::Ref),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: ty,
                field_flag: ArrayFlag::Signed,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: Some(0),
                parent: None,
                name: name.into(),
                owner: "W_ListObject".into(),
            });
            let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");
            assert!(
                field.get_parent_descr().is_some(),
                "{name} must carry a parent_descr after the bridge",
            );
            assert_eq!(
                field.offset(),
                expected.as_field_descr().unwrap().offset(),
                "{name} offset must match the W_LIST_DESCR_GROUP entry",
            );
        }
    }

    #[test]
    fn list_subclass_storage_is_cleared_by_allocation_descrs() {
        let list_size = w_list_size_descr();
        let list_gc_offsets: Vec<_> = list_size
            .as_size_descr()
            .unwrap()
            .gc_fielddescrs()
            .iter()
            .map(|field| field.offset())
            .collect();
        let user_size = W_LIST_USER_DESCR_GROUP.size_descr.clone();
        let user_gc_offsets: Vec<_> = user_size
            .as_size_descr()
            .unwrap()
            .gc_fielddescrs()
            .iter()
            .map(|field| field.offset())
            .collect();
        assert!(user_gc_offsets.contains(&std::mem::offset_of!(
            pyre_object::listobject::W_ListObjectUser,
            storage
        )));
        assert!(list_gc_offsets.contains(
            &(std::mem::offset_of!(W_ListObject, bytes_items)
                + pyre_object::bytes_array::BYTES_ARRAY_BLOCK_OFFSET)
        ));
        assert!(list_gc_offsets.contains(
            &(std::mem::offset_of!(W_ListObject, ascii_items)
                + pyre_object::unicode_array::UNICODE_ARRAY_BLOCK_OFFSET)
        ));
    }

    #[test]
    fn bare_gcstruct_descr_reconciles_at_path_hash() {
        let cases: &[(
            &str,
            u32,
            fn(u32),
            fn() -> DescrRef,
            fn() -> u32,
            usize,
            &[usize],
        )] = &[
            (
                "stringbuilder",
                0x00AB_CDEF,
                pyre_object::rbuilder::set_stringbuilder_gc_type_id,
                stringbuilder_size_descr,
                pyre_object::rbuilder::stringbuilder_gc_type_id,
                40,
                &[0, 32],
            ),
            (
                "stringpiece",
                0x00BE_EF01,
                pyre_object::rbuilder::set_stringpiece_gc_type_id,
                stringpiece_size_descr,
                pyre_object::rbuilder::stringpiece_gc_type_id,
                16,
                &[0, 8],
            ),
        ];
        for (name, seed, set_tid, size_descr, get_tid, expect_size, expect_offsets) in cases {
            set_tid(*seed);
            let size = size_descr();
            let sd = size
                .as_size_descr()
                .unwrap_or_else(|| panic!("case {name} is a SizeDescr"));
            let published_tid = get_tid();
            assert_eq!(sd.size(), *expect_size, "case {name} body size");
            assert!(!sd.headerless(), "case {name} must be a headered GcStruct");
            assert!(sd.is_gc_managed(), "case {name} is GC-managed");
            let traced: Vec<usize> = sd.gc_fielddescrs().iter().map(|f| f.offset()).collect();
            assert_eq!(&traced, expect_offsets, "case {name} traced offsets");
            assert_eq!(sd.type_id(), published_tid, "case {name} published tid");
            assert_eq!(
                sd.cache_key(),
                majit_ir::descr::path_hash(name),
                "case {name} path hash",
            );
            let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(name));
            let resolved = majit_ir::descr::gc_cache()
                .lock()
                ._cache_size
                .get(&key)
                .cloned()
                .unwrap_or_else(|| panic!("case {name} descr registered under its path_hash key"));
            let resolved = resolved
                .as_size_descr()
                .unwrap_or_else(|| panic!("case {name} resolved entry is a SizeDescr"));
            assert_eq!(resolved.size(), *expect_size, "case {name} resolved size");
            assert_eq!(
                resolved.type_id(),
                published_tid,
                "case {name} resolved tid"
            );
            let resolved_traced: Vec<usize> = resolved
                .gc_fielddescrs()
                .iter()
                .map(|f| f.offset())
                .collect();
            assert_eq!(
                &resolved_traced, expect_offsets,
                "case {name} resolved offsets"
            );
        }
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_box_payload_fields_to_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        // A codewriter-lowered body reads a box payload through the producer's
        // struct-layout descr: `W_IntObject` is modeled with a header, so its
        // `intval` lands at `index_in_parent` = 2.  The walker materializes the
        // same box through the single-field `W_*_DESCR_GROUP` (`index` = 0).
        // The bridge must return the group entry so the optimizer's
        // identity-keyed virtual-field cache matches the box-creation descr.
        for (owner, name, expected, ty) in [
            ("W_IntObject", "intval", int_intval_descr(), Type::Int),
            (
                "pyre_object::intobject::W_IntObject",
                "intval",
                int_intval_descr(),
                Type::Int,
            ),
            ("W_BoolObject", "intval", bool_intval_descr(), Type::Int),
            (
                "W_FloatObject",
                "floatval",
                float_floatval_descr(),
                Type::Float,
            ),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 16,
                field_size: 8,
                field_type: ty,
                field_flag: ArrayFlag::Signed,
                is_field_signed: true,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: Some(2),
                parent: None,
                name: name.into(),
                owner: owner.into(),
            });
            assert!(
                std::sync::Arc::ptr_eq(&descr, &expected),
                "{owner}.{name} must bridge to the canonical group entry Arc",
            );
        }
    }

    /// A `PyObject.w_class` `BhDescr` describing the same access as the
    /// canonical header descr, for both owner spellings the codewriter emits.
    fn w_class_bh(owner: &str, field_size: usize) -> majit_jitcode::jitcode::BhDescr {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        BhDescr::Field {
            offset: pyre_object::pyobject::W_CLASS_OFFSET,
            field_size,
            field_type: Type::Ref,
            field_flag: ArrayFlag::Signed,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: true,
            // slot 0 is `ob_type`; `w_class` is slot 1 of the header.
            index_in_parent: Some(1),
            parent: None,
            name: "w_class".into(),
            owner: owner.into(),
        }
    }

    /// The shared `PyObject` header's `w_class` bridges to the same descr the
    /// walker pins a value's class through, so a codewriter-lowered subclass
    /// test (`is_plain_int1`) reads the header the walker already guarded
    /// instead of emitting a second, uncacheable read of the same offset.
    #[test]
    fn w_class_descr_is_quasi_immutable_with_reserved_index() {
        let descr = w_class_descr();
        assert!(descr.is_quasi_immutable());
        assert_eq!(descr.index(), OBJECT_W_CLASS_INDEX);
    }

    #[test]
    fn make_descr_from_bh_size_and_kind0_parent_layout() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        {
            // make_descr_from_bh_size_reuses_declared_int_group
            let bh = BhDescr::Size {
                size: std::mem::size_of::<pyre_object::intobject::W_IntObject>(),
                type_id: majit_ir::descr::path_hash("intobject::W_IntObject"),
                vtable: 1,
                owner: "intobject::W_IntObject".into(),
                is_gc_managed: true,
                all_fielddescrs: vec![majit_jitcode::jitcode::BhFieldSpec {
                    index: 0,
                    field_key: "intval".into(),
                    name: "W_IntObject.intval".into(),
                    offset: 16,
                    field_size: 8,
                    field_type: Type::Int,
                    field_flag: ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                    is_class_word: None,
                }],
            };
            let descr = make_descr_from_bh(&bh);
            assert!(std::sync::Arc::ptr_eq(&descr, &w_int_size_descr()));
        }
        {
            // publish_kind0_parent_layout_inserts_an_undeclared_field_once
            use majit_jitcode::jitcode::BhFieldSpec;

            let type_id = majit_ir::descr::path_hash("startup_probe::NotARealStruct");
            let spec = majit_jitcode::jitcode::BhSizeSpec {
                size: 16,
                type_id,
                vtable: 1,
                owner: String::new(),
                is_gc_managed: true,
                headerless: false,
                all_fielddescrs: vec![BhFieldSpec {
                    index: 0,
                    field_key: "probe".into(),
                    name: "NotARealStruct.probe".into(),
                    offset: 8,
                    field_size: 8,
                    field_type: Type::Int,
                    field_flag: ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                    is_class_word: None,
                }],
            };
            publish_kind0_parent_layout(spec.clone());
            let key = majit_ir::descr::LLType::Struct(type_id);
            let first = majit_ir::descr::gc_cache()
                .lock()
                ._cache_field
                .get(&key)
                .and_then(|fields| fields.get("probe"))
                .cloned()
                .expect("undeclared parent layout publishes its field");
            publish_kind0_parent_layout(spec.clone());
            let second = majit_ir::descr::gc_cache()
                .lock()
                ._cache_field
                .get(&key)
                .and_then(|fields| fields.get("probe"))
                .cloned()
                .expect("second publish keeps the field");
            assert!(std::sync::Arc::ptr_eq(&first, &second));
            assert!(kind0_size_already_published(type_id));
            assert!(!kind0_size_already_published(0));
            // Same field count as the cached size: the packed names are the
            // row `get_field_descr` already stored.
            assert!(kind0_layout_already_published(type_id, 1));
            assert!(!kind0_layout_already_published(type_id, 2));
            assert!(!kind0_layout_already_published(0, 1));
            // The miss path forces module user layouts once (`descr.py`
            // `get_size_descr`). Their size rows are already cached.
            for entry in MODULE_USER_LAYOUT_DESCR_GROUPS.iter() {
                let laid_out =
                    majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(entry.def_path));
                assert!(
                    majit_ir::descr::gc_cache()
                        .lock()
                        ._cache_size
                        .contains_key(&laid_out),
                    "{}",
                    entry.def_path
                );
            }
        }
    }

    #[test]
    fn make_descr_from_bh_bridges_pyobject_w_class_to_the_walker_descr() {
        let canonical = w_class_descr();
        let width = W_CLASS_FIELD_DESCR.field_size();

        for owner in ["PyObject", "pyre_object::pyobject::PyObject"] {
            let descr = make_descr_from_bh(&w_class_bh(owner, width));
            assert!(
                std::sync::Arc::ptr_eq(&descr, &canonical),
                "{owner}.w_class must bridge to the walker's w_class descr Arc",
            );
        }
    }

    /// …but only when the two spellings describe the same access. The canonical
    /// descr hardcodes an 8-byte width while the codewriter sizes a pointer by
    /// `target_word_size()`, so on a 32-bit target the incoming descr is a
    /// narrower load at the same offset. Bridging there would widen the read
    /// over the adjacent payload, so the mismatch declines instead.
    #[test]
    fn make_descr_from_bh_declines_w_class_bridge_on_a_width_mismatch() {
        let canonical = w_class_descr();
        let narrower = W_CLASS_FIELD_DESCR.field_size() / 2;

        for owner in ["PyObject", "pyre_object::pyobject::PyObject"] {
            let descr = make_descr_from_bh(&w_class_bh(owner, narrower));
            assert!(
                !std::sync::Arc::ptr_eq(&descr, &canonical),
                "{owner}.w_class must not bridge to a descr of a different width",
            );
            assert_eq!(
                descr.as_field_descr().map(|f| f.field_size()),
                Some(narrower),
                "the declined descr must keep the width the codewriter asked for",
            );
        }
    }

    #[test]
    fn gil_ready_descr_is_quasi_immutable_not_always_pure() {
        let descr = gil_ready_descr();
        assert!(descr.is_quasi_immutable());
        assert!(!descr.is_always_pure());
        assert_eq!(descr.index(), GIL_READY_INDEX);
        let field = descr.as_field_descr().expect("field descr");
        assert_eq!(field.offset(), 0);
        assert_eq!(field.field_size(), 8);
        assert_eq!(field.field_type(), Type::Int);
    }

    #[test]
    fn make_descr_from_bh_bridges_gil_ready_to_the_quasi_descr() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        let canonical = gil_ready_descr();
        for owner in [
            "GilReadyState",
            "gil_ready::GilReadyState",
            "pyre_object::gil_ready::GilReadyState",
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                field_flag: ArrayFlag::Signed,
                is_field_signed: true,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: Some(0),
                parent: None,
                name: "gil_ready".into(),
                owner: owner.into(),
            });
            assert!(
                std::sync::Arc::ptr_eq(&descr, &canonical),
                "{owner}.gil_ready must bridge to the quasi descr",
            );
        }

        let narrow = make_descr_from_bh(&BhDescr::Field {
            offset: 0,
            field_size: 4,
            field_type: Type::Int,
            field_flag: ArrayFlag::Signed,
            is_field_signed: true,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(0),
            parent: None,
            name: "gil_ready".into(),
            owner: "GilReadyState".into(),
        });
        assert!(
            !std::sync::Arc::ptr_eq(&narrow, &canonical),
            "a 4-byte spelling must not bridge onto the 8-byte cell",
        );
    }

    #[test]
    fn descr_from_set_member_bridges_gil_ready() {
        let canonical = gil_ready_descr();
        let lookup = descr_from_set_member(&majit_ir::effectinfo::DescrSetMember::Field {
            struct_id: majit_ir::descr::path_hash("pyre_object::gil_ready::GilReadyState"),
            field_name: "gil_ready".into(),
        });
        let SetMemberLookup::Resolved(descr) = lookup else {
            panic!("gil_ready Field member must resolve");
        };
        assert!(
            std::sync::Arc::ptr_eq(&descr, &canonical),
            "gil_ready Field member must bridge to the quasi descr",
        );
    }

    /// The hand-lowered handler ops and translated execution-context helpers
    /// address the same three words through the same descriptor objects.
    /// `pypy/jit/backend/llsupport/descr.py GcCache.get_field_descr` has no
    /// second identity for a field merely because a different graph read it.
    #[test]
    fn make_descr_from_bh_bridges_execution_context_fields_to_declared_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::BhDescr;

        for (name, offset, canonical) in [
            (
                "sys_exc_value",
                pyre_interpreter::EC_SYS_EXC_VALUE_OFFSET,
                ec_sys_exc_value_descr(),
            ),
            (
                "topframeref",
                pyre_interpreter::EC_TOPFRAMEREF_OFFSET,
                ec_topframeref_descr(),
            ),
            (
                "w_tracefunc",
                pyre_interpreter::EC_W_TRACEFUNC_OFFSET,
                ec_w_tracefunc_descr(),
            ),
            (
                "profilefunc",
                pyre_interpreter::EC_PROFILEFUNC_OFFSET,
                ec_profilefunc_descr(),
            ),
            (
                "current_gen_or_coroutine",
                pyre_interpreter::EC_CURRENT_GEN_OR_COROUTINE_OFFSET,
                ec_current_gen_or_coroutine_descr(),
            ),
        ] {
            let (field_size, field_type) = if name == "profilefunc" {
                (std::mem::size_of::<usize>(), Type::Int)
            } else {
                (std::mem::size_of::<pyre_object::PyObjectRef>(), Type::Ref)
            };
            for owner in ["ExecutionContext", "executioncontext::ExecutionContext"] {
                let descr = make_descr_from_bh(&BhDescr::Field {
                    offset,
                    field_size,
                    field_type,
                    field_flag: ArrayFlag::Pointer,
                    is_field_signed: false,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: None,
                    parent: None,
                    name: name.into(),
                    owner: owner.into(),
                });
                assert!(
                    std::sync::Arc::ptr_eq(&descr, &canonical),
                    "{owner}.{name} must use the declared EC field Arc",
                );
            }
        }
    }

    #[test]
    fn make_descr_from_bh_struct_array_preserves_type_and_interior_fields() {
        use majit_ir::descr::ArrayFlag;
        use majit_jitcode::jitcode::{BhDescr, BhFieldSpec, BhInteriorFieldSpec, BhSizeSpec};

        let fields = vec![
            BhFieldSpec {
                index: 0,
                field_key: "x".into(),
                name: "Point.x".into(),
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                field_flag: ArrayFlag::Signed,
                is_field_signed: true,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
                is_class_word: None,
            },
            BhFieldSpec {
                index: 1,
                field_key: "y".into(),
                name: "Point.y".into(),
                offset: 8,
                field_size: 8,
                field_type: Type::Float,
                field_flag: ArrayFlag::Float,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
                is_class_word: None,
            },
        ];
        let owner = BhSizeSpec {
            size: 16,
            type_id: 11,
            vtable: 0,
            owner: String::new(),
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: fields.clone(),
        };
        let interior_fields = vec![
            BhInteriorFieldSpec {
                index: 0,
                field: fields[0].clone(),
                owner: owner.clone(),
            },
            BhInteriorFieldSpec {
                index: 1,
                field: fields[1].clone(),
                owner,
            },
        ];

        let descr = make_descr_from_bh(&BhDescr::Array {
            base_size: 8,
            itemsize: 16,
            len_offset: Some(0),
            type_id: 42,
            gc_type_id: 0,
            item_type: Type::Ref,
            is_array_of_pointers: false,
            is_array_of_structs: true,
            is_item_signed: false,
            // GcArray of Point structs (header-carrying), so the descr keeps
            // the GUARD_GC_TYPE gate.
            is_gc_managed: true,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields,
        });
        let array = descr.as_array_descr().expect("Array BhDescr -> ArrayDescr");

        assert!(array.is_array_of_structs());
        // Before the freeze-close hook, `init_array_descr` leaves
        // `UNSET_GC_TYPE_ID`. After it, the tid is the collector id
        // (`gc.py` `init_array_descr` / `TypeLayoutBuilder.get_type_id`).
        // Tid 0 is reserved (`gctypelayout.py`). Identity is `cache_key`.
        assert_ne!(array.type_id(), 0);
        assert_eq!(array.cache_key(), 42);
        assert_eq!(array.item_type(), Type::Ref);
        let interior = array
            .get_all_interiorfielddescrs()
            .expect("struct array must preserve interior field descrs");
        assert_eq!(interior.len(), 2);
        let second = interior[1]
            .as_interior_field_descr()
            .expect("interior field descr shape");
        assert_eq!(second.field_descr().field_name(), "Point.y");
        let parent = second
            .field_descr()
            .get_parent_descr()
            .expect("interior field parent_descr must be preserved");
        assert_eq!(parent.as_size_descr().unwrap().size(), 16);
    }

    /// `pyobject_gcarray_descr` is `cpu.arraydescrof(GcArray(OBJECTPTR))`:
    /// the runtime singleton must occupy the same `cache[ARRAY]` slot the
    /// codewriter names (`OBJECT_REF_GCARRAY_TYPE_ID`) so short-preamble
    /// `resolve_array_tid` recovers `PY_OBJECT_ARRAY_GC_TYPE_ID`.
    #[test]
    fn pyobject_gcarray_descr_publishes_under_the_codewriter_array_identity() {
        let descr = crate::state::pyobject_gcarray_descr();
        let array = descr
            .as_array_descr()
            .expect("pyobject_gcarray_descr must be an ArrayDescr");
        let atid = majit_jitcode::codewriter::jtransform::OBJECT_REF_GCARRAY_TYPE_ID;
        let cache_key = majit_ir::descr::path_hash(atid);
        assert_eq!(array.type_id(), PY_OBJECT_ARRAY_GC_TYPE_ID);
        assert_eq!(
            majit_ir::descr::gc_cache()
                .lock()
                .resolve_array_tid(cache_key),
            Some(PY_OBJECT_ARRAY_GC_TYPE_ID),
        );
    }

    /// `int_gcarray_descr` / `float_gcarray_descr` occupy the same
    /// `cache[ARRAY]` slots the codewriter names (`LIST_INT_ITEMS_ARRAY` /
    /// `LIST_FLOAT_ITEMS_ARRAY`) so `BhDescr::resolve_gc_tid` recovers the
    /// collector tid `GcLLDescr_framework.init_array_descr` wrote.
    #[test]
    fn typed_items_gcarray_descrs_publish_under_the_codewriter_array_identity() {
        crate::state::publish_typed_items_gcarray_descrs();
        for (descr, atid) in [
            (
                crate::state::int_gcarray_descr(),
                majit_jitcode::codewriter::jtransform::LIST_INT_ITEMS_ARRAY,
            ),
            (
                crate::state::float_gcarray_descr(),
                majit_jitcode::codewriter::jtransform::LIST_FLOAT_ITEMS_ARRAY,
            ),
        ] {
            let array = descr
                .as_array_descr()
                .expect("typed items gcarray descr must be an ArrayDescr");
            let cache_key = majit_ir::descr::path_hash(atid);
            assert_eq!(
                majit_ir::descr::gc_cache()
                    .lock()
                    .resolve_array_tid(cache_key),
                Some(array.type_id()),
            );
        }
    }
}

/// CallDescr for `pyre_object::longobject::jit_w_long_fits_int(obj) -> i64`.
/// `rbigint.fits_int()` is not annotated `@jit.elidable` upstream; it is only
/// used as a cannot-raise runtime guard before the elidable `toint()` call.
pub fn make_jit_w_long_fits_int_calldescr() -> DescrRef {
    majit_ir::make_call_descr(
        vec![Type::Ref],
        Type::Int,
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    )
}

/// CallDescr for `pyre_object::longobject::jit_w_long_toint(obj) -> i64`.
/// `W_LongObject.toint()` (longobject.py) → `rbigint.toint()`
/// (rbigint.py) — `EF_ELIDABLE_CANNOT_RAISE` because the caller
/// emits a fits_int GUARD_TRUE before invoking; OverflowError is
/// statically unreachable post-guard.
pub fn make_jit_w_long_toint_calldescr() -> DescrRef {
    majit_ir::make_call_descr(
        vec![Type::Ref],
        Type::Int,
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::ElidableCannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    )
}

fn simple_field_spec_from_bh(
    spec: &majit_jitcode::jitcode::BhFieldSpec,
) -> majit_ir::descr::SimpleFieldDescrSpec {
    majit_ir::descr::SimpleFieldDescrSpec {
        index: spec.index,
        field_key: spec.field_key().to_string(),
        name: spec.name.clone(),
        offset: spec.offset,
        field_size: spec.field_size,
        field_type: spec.field_type,
        is_immutable: spec.is_immutable,
        is_quasi_immutable: spec.is_quasi_immutable,
        flag: spec.field_flag,
        virtualizable: false,
        // The producer's declaration, carried straight through — `None`
        // included, so a spec that declared nothing leaves the question to the
        // name fallback instead of arriving as a declaration of a guess.
        is_class_word: spec.is_class_word,
        index_in_parent: spec.index_in_parent,
    }
}

/// `descr.py get_size_descr` cache parity.
///
/// PyPy `gc_cache._cache_size[STRUCT]` keys on the **STRUCT object
/// identity**, not on its layout — two distinct RPython STRUCTs that
/// share `(size, vtable, fieldlist)` get distinct `SizeDescr` Arcs.
/// Pyre's analogue of "STRUCT identity" is `BhSizeSpec.type_id`
/// (`jit_struct.rs:__majit_type_id` → `path_hash(module_path::TypeName)`):
/// every struct type has a unique `type_id`, and two RPython STRUCTs
/// with coincidentally-identical layout end up with distinct
/// `type_id`s.  Keying the cache on `type_id` alone matches PyPy's
/// per-type identity, where structural-equality keying (the prior
/// `BhSizeSpec`-by-value variant) would have collapsed identity for
/// layout-coincident-but-logically-distinct structs.
///
/// `spec.type_id == 0` is the legacy fallback path
/// (`assembler.rs`'s `bh_size_spec_from_callcontrol` stamps zero
/// when the analyzer-time callcontrol has no host-type carrier).
/// Without a STRUCT-identity carrier we MUST NOT key the cache by
/// the zero sentinel — different STRUCTs with `type_id == 0` would
/// alias onto the first one inserted (`or_insert` "first wins"),
/// silently mixing their field tables.  PyPy's `_cache_size[STRUCT]`
/// never aliases distinct STRUCTs; absent a real identity carrier,
/// the closest orthodox behaviour is "each call is a distinct
/// STRUCT" — mint fresh per call.
/// Every STRUCT this module declares a descriptor group for, by the def path
/// the group is registered under.
///
/// `build_object_descr_group_with_def_path` keys a group as
/// `path_hash(def_path)`, which is the `GcCache` `(STRUCT, fieldname)`
/// namespace a serialized `BhDescr` resolves into as well.  The two producers
/// do not carry the same information: a group spells out `descr.py
/// STRUCT._immutable_field(fieldname)` for each field, while the codewriter's
/// `immutable_fields_by_struct` is empty for the whole LLBC pipeline, so every
/// field a `BhDescr` describes reads as plain mutable.  `GcCache` hands the
/// slot to whoever asks first and returns it unchanged after that, so without
/// forcing the group here the rank a field ends up carrying is decided by
/// which of the two reached it first -- and a `BhDescr` that got there first
/// takes the declaration off every field of the STRUCT at once.
///
/// A group that does not reach that namespace is absent from the list:
/// `PYCODE` and `EC` mint through the unkeyed factory, and `PYTRACEBACK` is
/// registered under an empty def path, so its key is `path_hash("")` --
/// none of the three names a STRUCT a `BhDescr` can carry.
static PYERROR_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_interpreter::error::PyErrorObject;
    let field = |key, offset| {
        (
            key,
            offset,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            false,
            false,
        )
    };
    build_object_descr_group_with_def_path(
        std::mem::size_of::<PyErrorObject>(),
        <PyErrorObject as pyre_object::lltype::GcType>::type_id(),
        &pyre_interpreter::error::PYERROR_TYPE as *const _ as usize,
        &[
            field("w_type", pyre_interpreter::error::PYERROR_W_TYPE_OFFSET),
            field("message", pyre_interpreter::error::PYERROR_MESSAGE_OFFSET),
            field(
                "exc_object",
                pyre_interpreter::error::PYERROR_EXC_OBJECT_OFFSET,
            ),
            field(
                "w_name_context",
                pyre_interpreter::error::PYERROR_W_NAME_CONTEXT_OFFSET,
            ),
            field(
                "w_obj_context",
                pyre_interpreter::error::PYERROR_W_OBJ_CONTEXT_OFFSET,
            ),
            field(
                "w_class",
                std::mem::offset_of!(PyErrorObject, ob_header.w_class),
            ),
        ],
        "PyErrorObject",
        "error::PyErrorObject",
    )
});

static DECLARED_GROUPS: &[(&str, fn())] = &[
    ("intobject::W_IntObject", || {
        LazyLock::force(&W_INT_DESCR_GROUP);
    }),
    ("intobject::W_IntObjectUser", || {
        LazyLock::force(&W_INT_USER_DESCR_GROUP);
    }),
    ("floatobject::W_FloatObject", || {
        LazyLock::force(&W_FLOAT_DESCR_GROUP);
    }),
    ("floatobject::W_FloatObjectUser", || {
        LazyLock::force(&W_FLOAT_USER_DESCR_GROUP);
    }),
    ("complexobject::W_ComplexObject", || {
        LazyLock::force(&W_COMPLEX_DESCR_GROUP);
    }),
    ("error::PyErrorObject", || {
        LazyLock::force(&PYERROR_DESCR_GROUP);
    }),
    ("complexobject::W_ComplexObjectUser", || {
        LazyLock::force(&W_COMPLEX_USER_DESCR_GROUP);
    }),
    ("longobject::W_LongObject", || {
        LazyLock::force(&W_LONG_DESCR_GROUP);
    }),
    ("boolobject::W_BoolObject", || {
        LazyLock::force(&W_BOOL_DESCR_GROUP);
    }),
    ("unicodeobject::W_UnicodeObject", || {
        LazyLock::force(&W_UNICODE_DESCR_GROUP);
    }),
    ("unicodeobject::W_UnicodeObjectUser", || {
        LazyLock::force(&W_UNICODE_USER_DESCR_GROUP);
    }),
    ("bytesobject::W_BytesObject", || {
        LazyLock::force(&W_BYTES_DESCR_GROUP);
    }),
    ("bytesobject::W_BytesObjectUser", || {
        LazyLock::force(&W_BYTES_USER_DESCR_GROUP);
    }),
    ("bytearrayobject::W_BytearrayObject", || {
        LazyLock::force(&W_BYTEARRAY_DESCR_GROUP);
    }),
    ("bytearrayobject::W_BytearrayObjectUser", || {
        LazyLock::force(&W_BYTEARRAY_USER_DESCR_GROUP);
    }),
    ("setobject::W_SetObject", || {
        LazyLock::force(&W_SET_DESCR_GROUP);
    }),
    ("functional::W_IntRangeIterator", || {
        LazyLock::force(&RANGE_ITER_DESCR_GROUP);
    }),
    ("functional::W_IntRangeStepOneIterator", || {
        LazyLock::force(&RANGE_ITER_STEP_ONE_DESCR_GROUP);
    }),
    ("functional::W_IntRangeOneArgIterator", || {
        LazyLock::force(&RANGE_ITER_ONE_ARG_DESCR_GROUP);
    }),
    ("operation::_CallableIterator", || {
        LazyLock::force(&CALLABLE_ITERATOR_DESCR_GROUP);
    }),
    ("functional::W_Range", || {
        LazyLock::force(&RANGE_DESCR_GROUP);
    }),
    ("functional::W_Zip", || {
        LazyLock::force(&W_ZIP_DESCR_GROUP);
    }),
    ("iterobject::W_SeqIterObject", || {
        LazyLock::force(&SEQ_ITER_DESCR_GROUP);
    }),
    ("iterobject::W_TupleIterObject", || {
        LazyLock::force(&TUPLE_ITER_DESCR_GROUP);
    }),
    ("iterobject::W_ListIterObject", || {
        LazyLock::force(&LIST_ITER_DESCR_GROUP);
    }),
    ("function::Function", || {
        LazyLock::force(&FUNCTION_DESCR_GROUP);
    }),
    ("function::Method", || {
        LazyLock::force(&W_METHOD_DESCR_GROUP);
    }),
    ("function::StaticMethod", || {
        LazyLock::force(&W_STATICMETHOD_DESCR_GROUP);
    }),
    ("function::ClassMethod", || {
        LazyLock::force(&W_CLASSMETHOD_DESCR_GROUP);
    }),
    ("dictmultiobject::W_DictObject", || {
        LazyLock::force(&W_DICT_DESCR_GROUP);
        LazyLock::force(&W_DICT_LOOKUP_NS_DESCR);
        LazyLock::force(&W_DICT_STRATEGY_WORD_DESCR);
    }),
    ("celldict::ObjectMutableCell", || {
        LazyLock::force(&W_OBJECT_MUTABLE_CELL_DESCR_GROUP);
    }),
    ("nestedscope::Cell", || {
        LazyLock::force(&W_CELL_DESCR_GROUP);
    }),
    ("descriptor::W_Super", || {
        LazyLock::force(&W_SUPER_DESCR_GROUP);
    }),
    ("listobject::W_ListObject", || {
        LazyLock::force(&W_LIST_DESCR_GROUP);
    }),
    ("listobject::W_ListObjectUser", || {
        LazyLock::force(&W_LIST_USER_DESCR_GROUP);
    }),
    ("setobject::W_SetObjectUser", || {
        LazyLock::force(&W_SET_USER_DESCR_GROUP);
    }),
    ("interp_array::W_ArrayUser", || {
        LazyLock::force(&W_ARRAY_USER_DESCR_GROUP);
    }),
    ("weakref::W_WeakrefUser", || {
        LazyLock::force(&W_WEAKREF_USER_DESCR_GROUP);
    }),
    ("functional::W_EnumerateUser", || {
        LazyLock::force(&W_ENUMERATE_USER_DESCR_GROUP);
    }),
    ("functional::W_MapUser", || {
        LazyLock::force(&W_MAP_USER_DESCR_GROUP);
    }),
    ("functional::W_FilterUser", || {
        LazyLock::force(&W_FILTER_USER_DESCR_GROUP);
    }),
    ("functional::W_ZipUser", || {
        LazyLock::force(&W_ZIP_USER_DESCR_GROUP);
    }),
    ("functional::W_ReversedIteratorUser", || {
        LazyLock::force(&W_REVERSED_USER_DESCR_GROUP);
    }),
    ("descriptor::W_SuperUser", || {
        LazyLock::force(&W_SUPER_USER_DESCR_GROUP);
    }),
    ("descriptor::W_PropertyUser", || {
        LazyLock::force(&W_PROPERTY_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_CountUser", || {
        LazyLock::force(&W_COUNT_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_RepeatUser", || {
        LazyLock::force(&W_REPEAT_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_TakeWhileUser", || {
        LazyLock::force(&W_TAKEWHILE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_DropWhileUser", || {
        LazyLock::force(&W_DROPWHILE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_FilterFalseUser", || {
        LazyLock::force(&W_FILTERFALSE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_ISliceUser", || {
        LazyLock::force(&W_ISLICE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_BatchedUser", || {
        LazyLock::force(&W_BATCHED_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_ProductUser", || {
        LazyLock::force(&W_PRODUCT_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_CombinationsUser", || {
        LazyLock::force(&W_COMBINATIONS_USER_DESCR_GROUP);
    }),
    (
        "interp_itertools::W_CombinationsWithReplacementUser",
        || {
            LazyLock::force(&W_COMBINATIONS_WITH_REPLACEMENT_USER_DESCR_GROUP);
        },
    ),
    ("interp_itertools::W_PermutationsUser", || {
        LazyLock::force(&W_PERMUTATIONS_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_GroupByUser", || {
        LazyLock::force(&W_GROUPBY_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_CompressUser", || {
        LazyLock::force(&W_COMPRESS_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_StarMapUser", || {
        LazyLock::force(&W_STARMAP_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_AccumulateUser", || {
        LazyLock::force(&W_ACCUMULATE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_ZipLongestUser", || {
        LazyLock::force(&W_ZIP_LONGEST_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_PairwiseUser", || {
        LazyLock::force(&W_PAIRWISE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_CycleUser", || {
        LazyLock::force(&W_CYCLE_USER_DESCR_GROUP);
    }),
    ("interp_itertools::W_ChainUser", || {
        LazyLock::force(&W_CHAIN_USER_DESCR_GROUP);
    }),
    (
        "module::__pypy__::interp_buffer::bufferable_impl::W_BufferableUser",
        || {
            LazyLock::force(&W_BUFFERABLE_USER_DESCR_GROUP);
        },
    ),
    ("module::_collections::W_DequeUser", || {
        LazyLock::force(&W_DEQUE_USER_DESCR_GROUP);
    }),
    ("module::struct::W_StructUser", || {
        LazyLock::force(&W_STRUCT_USER_DESCR_GROUP);
    }),
    ("_pypy_generic_alias::GenericAliasUser", || {
        LazyLock::force(&W_GENERIC_ALIAS_USER_DESCR_GROUP);
    }),
    ("longobject::W_LongObjectUser", || {
        LazyLock::force(&W_LONG_USER_DESCR_GROUP);
    }),
    ("function::StaticMethodUser", || {
        LazyLock::force(&W_STATICMETHOD_USER_DESCR_GROUP);
    }),
    ("function::ClassMethodUser", || {
        LazyLock::force(&W_CLASSMETHOD_USER_DESCR_GROUP);
    }),
    ("module::ModuleUser", || {
        LazyLock::force(&W_MODULE_USER_DESCR_GROUP);
    }),
    ("module::thread::W_LocalUser", || {
        LazyLock::force(&W_LOCAL_USER_DESCR_GROUP);
    }),
    ("module::_io::W_BytesIOUser", || {
        LazyLock::force(&W_BYTESIO_USER_DESCR_GROUP);
    }),
    ("module::_io::W_StringIOUser", || {
        LazyLock::force(&W_STRINGIO_USER_DESCR_GROUP);
    }),
    ("module::_io::W_BufferedReaderUser", || {
        LazyLock::force(&W_BUFFEREDREADER_USER_DESCR_GROUP);
    }),
    ("module::_io::W_BufferedWriterUser", || {
        LazyLock::force(&W_BUFFEREDWRITER_USER_DESCR_GROUP);
    }),
    ("module::_io::W_BufferedRWPairUser", || {
        LazyLock::force(&W_BUFFEREDRWPAIR_USER_DESCR_GROUP);
    }),
    ("module::_io::W_BufferedRandomUser", || {
        LazyLock::force(&W_BUFFEREDRANDOM_USER_DESCR_GROUP);
    }),
    ("module::_io::W_TextIOWrapperUser", || {
        LazyLock::force(&W_TEXTIOWRAPPER_USER_DESCR_GROUP);
    }),
    ("module::_io::W_FileIOUser", || {
        LazyLock::force(&W_FILEIO_USER_DESCR_GROUP);
    }),
    #[cfg(all(windows, not(feature = "sandbox")))]
    ("module::_io::W_WinConsoleIOUser", || {
        LazyLock::force(&W_WINCONSOLEIO_USER_DESCR_GROUP);
    }),
    ("tupleobject::W_TupleObject", || {
        LazyLock::force(&W_TUPLE_DESCR_GROUP);
    }),
    ("tupleobject::W_TupleObjectUser", || {
        LazyLock::force(&W_TUPLE_USER_DESCR_GROUP);
    }),
    (
        "specialisedtupleobject::W_SpecialisedTupleObject_ii",
        || {
            LazyLock::force(&SPECIALISED_TUPLE_II_DESCR_GROUP);
        },
    ),
    (
        "specialisedtupleobject::W_SpecialisedTupleObject_ff",
        || {
            LazyLock::force(&SPECIALISED_TUPLE_FF_DESCR_GROUP);
        },
    ),
    (
        "specialisedtupleobject::W_SpecialisedTupleObject_oo",
        || {
            LazyLock::force(&SPECIALISED_TUPLE_OO_DESCR_GROUP);
        },
    ),
    ("object_array::ItemsBlock", || {
        LazyLock::force(&ITEMS_BLOCK_DESCR_GROUP);
    }),
    ("rbigint::RBigIntPair", || {
        LazyLock::force(&RBIGINT_PAIR_DESCR_GROUP);
    }),
    ("pyframe::FrameDebugData", || {
        LazyLock::force(&FRAME_DEBUG_DATA_DESCR_GROUP);
    }),
    ("pyframe::PyFrame", || {
        LazyLock::force(&PYFRAME_DESCR_GROUP);
    }),
    ("pyframe::frame_locals_proxy::FrameLocalsProxy", || {
        LazyLock::force(&FRAME_LOCALS_PROXY_DESCR_GROUP);
    }),
    ("sliceobject::W_SliceObject", || {
        LazyLock::force(&W_SLICE_DESCR_GROUP);
    }),
    ("objectobject::W_ObjectObject", || {
        LazyLock::force(&W_OBJECT_OBJECT_DESCR_GROUP);
    }),
    ("objectobject::W_ObjectObjectUser", || {
        LazyLock::force(&W_OBJECT_OBJECT_USER_SIZE_DESCR);
    }),
    ("interp_exceptions::W_BaseException", || {
        let _ = w_exception_descrs_for(ExcKind::BaseException, false);
    }),
    ("interp_exceptions::W_BaseExceptionUser", || {
        let _ = w_exception_descrs_for(ExcKind::ValueError, true);
    }),
    ("interp_exceptions::W_ExceptionExtended", || {
        let _ = w_exception_descrs_for(ExcKind::OSError, false);
    }),
    ("interp_exceptions::W_ExceptionExtendedUser", || {
        let _ = w_exception_descrs_for(ExcKind::FileNotFoundError, true);
    }),
    ("gil_ready::GilReadyState", || {
        LazyLock::force(&GIL_READY_DESCR_GROUP);
    }),
    ("pyre_object::gil_ready::GilReadyState", || {
        LazyLock::force(&GIL_READY_DESCR_GROUP);
    }),
];

/// The [`DECLARED_GROUPS`] rows by the `GcCache` key each one owns.
static DECLARED_GROUP_BY_KEY: LazyLock<std::collections::HashMap<u64, fn()>> =
    LazyLock::new(|| {
        DECLARED_GROUPS
            .iter()
            .map(|(def_path, force)| (majit_ir::descr::path_hash(def_path), *force))
            .collect()
    });

/// `heaptracker.py` `setup_cache_gcstruct2vtable`.
///
/// Walk every `#[pyre_class]` descriptor and register its `PyType` as
/// the STRUCT's vtable under the crate-stripped `path_hash`, matching
/// `heaptracker.py` `_cache_gcstruct2vtable[GCSTRUCT]` (the STRUCT
/// object, never a short name). Analyzer `get_size_descr(..., vtable=0)`
/// then mints an `is_object()` SizeDescr. `DECLARED_GROUPS` already does
/// this for the core types it names; pyre-module classes such as
/// `W_CType` have no row, and without this census their field descrs
/// fail `protect_speculative_field`'s non-object typeid equality.
///
/// Does not mint a SizeDescr: a fieldless shell would steal
/// `all_fielddescrs` from the later layout producer.
pub(crate) fn publish_pyre_class_vtables() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(publish_pyre_class_vtables_inner);
}

/// Second pass after the collector has stamped `TypeIdCell`s.
pub(crate) fn refresh_pyre_class_vtable_tids() {
    publish_pyre_class_vtables_inner();
}

fn publish_pyre_class_vtables_inner() {
    let mut gc = majit_ir::descr::gc_cache().lock();
    pyre_object::lltype::for_each_class_descriptor(|descr| {
        let vtable = descr.pytype_ptr as usize;
        if vtable == 0 {
            return;
        }
        let tid = descr.gc_type_id.get();
        let tid = if tid == pyre_object::lltype::TypeIdCell::UNASSIGNED {
            0
        } else {
            tid
        };
        let stripped = match descr.struct_path.split_once("::") {
            Some((_, rest)) => rest,
            None => descr.struct_path,
        };
        gc.register_gcstruct_vtable(majit_ir::descr::path_hash(stripped), vtable, tid);
    });
}

/// Force this module's own group for `cache_key`, if it declares one, so the
/// declaration takes the STRUCT's `(STRUCT, fieldname)` slots before a
/// serialized `BhDescr` can.
///
/// Idempotent and cheap after the first call for a STRUCT: the group is a
/// `LazyLock` and the second force is a load. A key this module does not
/// declare still forces [`MODULE_USER_LAYOUT_DESCR_GROUPS`] once. Its
/// constructor publishes every `def_path` (`descr.py` `get_size_descr`).
/// Matching `path_hash` on each miss does not select a row.
fn force_declared_group(cache_key: u64) {
    publish_pyre_class_vtables();
    if let Some(force) = DECLARED_GROUP_BY_KEY.get(&cache_key) {
        force();
        return;
    }
    std::sync::LazyLock::force(&MODULE_USER_LAYOUT_DESCR_GROUPS);
}

/// `descr.py` `get_size_descr` reads a cache translation already filled.
/// `pyjitpl.py` `finish_setup_descrs` does not hash a side table once per row.
pub(crate) fn force_all_declared_groups() {
    for (_, force) in DECLARED_GROUPS {
        force();
    }
    std::sync::LazyLock::force(&MODULE_USER_LAYOUT_DESCR_GROUPS);
}

/// `descr.py` `get_size_descr` / `get_field_descr` return the cached descr.
/// `force_declared_group` already published this STRUCT, so rebuilding it
/// from the serialized spec would clone every field name and mint a second
/// size descr that `register_keyed_size` then discards (the runtime group
/// owns the vtable). A vtable-less shell still falls through: that path is
/// the one `register_keyed_size` may upgrade.
fn published_declared_group(
    spec: &majit_jitcode::jitcode::BhSizeSpec,
) -> Option<majit_ir::descr::SimpleDescrGroup> {
    if spec.type_id == 0 {
        return None;
    }
    let key = majit_ir::descr::LLType::Struct(spec.type_id);
    let gc = majit_ir::descr::gc_cache().lock();
    let size_ref = gc._cache_size.get(&key)?.clone();
    let size_descr =
        majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(size_ref).ok()?;
    if size_descr.vtable() == 0 {
        return None;
    }
    let fields = gc._cache_field.get(&key)?;
    let mut field_descrs = Vec::with_capacity(spec.all_fielddescrs.len());
    for field in &spec.all_fielddescrs {
        let descr = fields.get(field.field_key())?.clone();
        if let Some(is_class_word) = field.is_class_word {
            descr.declare_class_word(is_class_word);
        }
        field_descrs.push(descr);
    }
    Some(majit_ir::descr::SimpleDescrGroup {
        size_descr,
        field_descrs,
    })
}

fn simple_descr_group_from_bh_size(
    spec: &majit_jitcode::jitcode::BhSizeSpec,
) -> majit_ir::descr::SimpleDescrGroup {
    if spec.type_id != 0 {
        force_declared_group(spec.type_id);
        if let Some(group) = published_declared_group(spec) {
            return group;
        }
    }
    let field_specs: Vec<_> = spec
        .all_fielddescrs
        .iter()
        .map(simple_field_spec_from_bh)
        .collect();

    if spec.type_id == 0 {
        // No STRUCT-identity carrier — mint fresh per call so distinct
        // type_id-less STRUCTs don't collapse onto the first-inserted
        // descr group.  Per-STRUCT caching kicks in only when callers
        // route through a real `type_id` source.
        //
        // The missing key says nothing about the STRUCT's shape, so carry
        // `is_gc_managed`/`headerless` here exactly as the keyed arm below
        // does.  `bh_size_spec_from_descr` (`majit-translate`) reads both
        // straight off a descr while taking `type_id` from `cache_key()`, so
        // a raw or header-less parent with no key reaches this arm; the
        // defaulting factory would give it back a GC header it does not have.
        return majit_ir::descr::make_simple_descr_group_with_flags(
            u32::MAX,
            spec.size,
            spec.type_id as u32,
            spec.vtable as usize,
            spec.is_gc_managed,
            spec.headerless,
            &field_specs,
        );
    }
    // The spec describes every field as plain mutable: the codewriter's
    // `immutable_fields_by_struct` is empty for the whole LLBC pipeline, so a
    // serialized `BhDescr` carries no `descr.py:229
    // STRUCT._immutable_field(fieldname)` rank.  `force_declared_group` above
    // asked first; this mint runs only when that publication did not already
    // cover every field (`published_declared_group`).
    // `descr.py get_size_descr` + `:218-239 get_field_descr`
    // keyed publish: GcCache is the sole owner/cache for this STRUCT.
    majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
        u32::MAX,
        spec.size,
        spec.type_id as u32,
        spec.type_id,
        spec.vtable as usize,
        spec.is_gc_managed,
        spec.headerless,
        &field_specs,
        &[],
    )
}

/// `descr.py` `GcCache.setup_descrs` numbers the rows `get_size_descr` /
/// `get_field_descr` already stored. A parented kind-0 field is one of those
/// rows: the parent layout's `all_fielddescrs` is the list. Publishing that
/// layout once fills `_cache_field`. Decoding the field slot again would
/// deserialize the same names and drop the `DescrRef`.
fn field_spec_from_owned_bh(
    spec: majit_jitcode::jitcode::BhFieldSpec,
) -> majit_ir::descr::SimpleFieldDescrSpec {
    // `descr.py` `get_field_descr` keeps one display name. An empty field key
    // is that name; moving it avoids the clone `simple_field_spec_from_bh`
    // does for a borrowed spec.
    let field_key = if spec.field_key.is_empty() {
        spec.name.clone()
    } else {
        spec.field_key
    };
    majit_ir::descr::SimpleFieldDescrSpec {
        index: spec.index,
        field_key,
        name: spec.name,
        offset: spec.offset,
        field_size: spec.field_size,
        field_type: spec.field_type,
        is_immutable: spec.is_immutable,
        is_quasi_immutable: spec.is_quasi_immutable,
        flag: spec.field_flag,
        virtualizable: false,
        is_class_word: spec.is_class_word,
        index_in_parent: spec.index_in_parent,
    }
}

/// Field counts for STRUCT rows that already carry a vtable.
///
/// `descr.py` `get_size_descr` returns the cached row. `pyjitpl.py`
/// `finish_setup_descrs` does not lock `GcCache` again for every packed
/// record to notice that. One pass fills this table; a later record compares
/// `type_id` and the field count.
pub(crate) fn kind0_vtable_field_counts() -> Vec<(u64, u32)> {
    let gc = majit_ir::descr::gc_cache().lock();
    let mut rows = Vec::new();
    for (key, size_ref) in &gc._cache_size {
        let majit_ir::descr::LLType::Struct(type_id) = *key else {
            continue;
        };
        let Ok(size) =
            majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(size_ref.clone())
        else {
            continue;
        };
        if size.vtable() == 0 {
            continue;
        }
        let Ok(nfields) =
            u32::try_from(majit_ir::descr::SizeDescr::all_fielddescrs(size.as_ref()).len())
        else {
            continue;
        };
        rows.push((type_id, nfields));
    }
    rows.sort_unstable();
    rows
}

/// `descr.py` `get_size_descr` / `get_field_descr` — the STRUCT row is
/// already in `GcCache`, carries a vtable, and lists the same fields the
/// packed record would rebuild. A count mismatch still mints: a declared
/// group can carry gc edges the packed list does not.
pub(crate) fn kind0_layout_already_published(type_id: u64, nfields: u32) -> bool {
    if type_id == 0 {
        return false;
    }
    let key = majit_ir::descr::LLType::Struct(type_id);
    let gc = majit_ir::descr::gc_cache().lock();
    let Some(size_ref) = gc._cache_size.get(&key).cloned() else {
        return false;
    };
    let Ok(size) = majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(size_ref)
    else {
        return false;
    };
    if size.vtable() == 0 {
        return false;
    }
    majit_ir::descr::SizeDescr::all_fielddescrs(size.as_ref()).len() == nfields as usize
}

/// `descr.py` `get_size_descr` has no row for this STRUCT yet.
pub(crate) fn kind0_struct_absent(type_id: u64) -> bool {
    if type_id == 0 {
        return false;
    }
    let key = majit_ir::descr::LLType::Struct(type_id);
    let gc = majit_ir::descr::gc_cache().lock();
    !gc._cache_size.contains_key(&key)
        && gc
            ._cache_field
            .get(&key)
            .is_none_or(|fields| fields.is_empty())
}

/// STRUCT keys `kind0_struct_absent` would see, taken in one `GcCache` lock.
///
/// `descr.py` `get_size_descr` consults the cache once per STRUCT.
/// `finish_setup_descrs` does not lock again for every parentless slot.
/// A nonempty `_cache_field` row counts: `kind0_struct_absent` treats that
/// as already published even when `_cache_size` has no entry.
pub(crate) fn kind0_present_struct_ids() -> Vec<u64> {
    let gc = majit_ir::descr::gc_cache().lock();
    let mut ids = Vec::new();
    for key in gc._cache_size.keys() {
        let majit_ir::descr::LLType::Struct(type_id) = *key else {
            continue;
        };
        if type_id != 0 {
            ids.push(type_id);
        }
    }
    for (key, fields) in &gc._cache_field {
        if fields.is_empty() {
            continue;
        }
        let majit_ir::descr::LLType::Struct(type_id) = *key else {
            continue;
        };
        if type_id != 0 {
            ids.push(type_id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Type ids `descr.py` `get_size_descr` already holds.
///
/// `pyjitpl.py` `finish_setup_descrs` does not lock `GcCache` again for every
/// parentless size slot. One pass records a row with a vtable or a field
/// list. A vtable-less shell with no fields stays out, so a later slot can
/// still upgrade it through `register_keyed_size`.
pub(crate) fn kind0_published_size_ids() -> Vec<u64> {
    let gc = majit_ir::descr::gc_cache().lock();
    let mut ids = Vec::new();
    for (key, size_ref) in &gc._cache_size {
        let majit_ir::descr::LLType::Struct(type_id) = *key else {
            continue;
        };
        if type_id == 0 {
            continue;
        }
        let Ok(size) =
            majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(size_ref.clone())
        else {
            continue;
        };
        if size.vtable() == 0
            && majit_ir::descr::SizeDescr::all_fielddescrs(size.as_ref()).is_empty()
        {
            continue;
        }
        ids.push(type_id);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// `descr.py` `get_size_descr` — the STRUCT row is already in `GcCache`.
/// A later size slot is that descr: do not bincode its field list again.
/// A vtable-less shell with no fields still decodes, so `register_keyed_size`
/// can upgrade it.
pub(crate) fn kind0_size_already_published(type_id: u64) -> bool {
    if type_id == 0 {
        return false;
    }
    let key = majit_ir::descr::LLType::Struct(type_id);
    let gc = majit_ir::descr::gc_cache().lock();
    let Some(size_ref) = gc._cache_size.get(&key).cloned() else {
        return false;
    };
    majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(size_ref).is_ok_and(
        |size| {
            size.vtable() != 0
                || !majit_ir::descr::SizeDescr::all_fielddescrs(size.as_ref()).is_empty()
        },
    )
}

pub(crate) fn publish_kind0_parent_layout(spec: majit_jitcode::jitcode::BhSizeSpec) {
    if spec.type_id == 0 {
        return;
    }
    // `descr.py` `get_field_descr` returns the cached row. A declared group
    // already filled it; moving the serialized names into a second mint would
    // allocate those strings again.
    force_declared_group(spec.type_id);
    if published_declared_group(&spec).is_some() {
        return;
    }
    let field_specs: Vec<_> = spec
        .all_fielddescrs
        .into_iter()
        .map(field_spec_from_owned_bh)
        .collect();
    let _group = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
        u32::MAX,
        spec.size,
        spec.type_id as u32,
        spec.type_id,
        spec.vtable as usize,
        spec.is_gc_managed,
        spec.headerless,
        &field_specs,
        &[],
    );
}

/// `descr.py` `get_field_descr` for a STRUCT that is not cached yet.
/// Field names stay the packed record's bytes.
pub(crate) fn publish_borrowed_parent_layout(layout: majit_jitcode::jitcode::StaticParentLayout) {
    if layout.type_id == 0 {
        return;
    }
    let fields = layout
        .fields
        .into_iter()
        .map(|field| majit_ir::descr::BorrowedField {
            index: field.index,
            name: Cow::Borrowed(field.name),
            field_key: field.field_key,
            offset: field.offset,
            field_size: field.field_size,
            field_type: field.field_type,
            flag: field.field_flag,
            is_immutable: field.is_immutable,
            is_quasi_immutable: field.is_quasi_immutable,
            index_in_parent: field.index_in_parent,
            is_class_word: field.is_class_word,
        })
        .collect();
    let _group = majit_ir::descr::publish_borrowed_struct_layout(
        u32::MAX,
        layout.size,
        layout.type_id as u32,
        layout.type_id,
        layout.vtable as usize,
        layout.is_gc_managed,
        layout.headerless,
        &[],
        fields,
    );
}

/// `claimed_index` is the descr's own `index_in_parent` claim, passed beside
/// `field` rather than read off it: `BhFieldSpec` is the shape of a PARENT's
/// field-list entry, where every index is a real position, so its field is a
/// plain `usize` and cannot carry the "never resolved a slot" state that a
/// standalone `BhDescr::Field` can be in. Flattening it here is what made the
/// unresolved-mint table unsplittable.
fn field_descr_from_bh_field(
    field: &majit_jitcode::jitcode::BhFieldSpec,
    parent: Option<&majit_jitcode::jitcode::BhSizeSpec>,
    claimed_index: Option<usize>,
) -> DescrRef {
    if let Some(parent) = parent {
        // `descr.py get_field_descr` cache-hit: when the parent
        // STRUCT is published in `_cache_size`, walk its
        // `all_fielddescrs()` directly so the returned Arc is the
        // runtime `PyreFieldDescr` (or analyzer-published
        // `SimpleFieldDescr`).  Both back-reference the same parent
        // SizeDescr via `parent_descr` (descr.py), so the
        // an adapter wrapper is unnecessary on this path — analyzer
        // raw-set Arcs and runtime allocator descrs share
        // one identity slot.
        if parent.type_id != 0 {
            let key = majit_ir::descr::LLType::Struct(parent.type_id);
            let field_key = field.field_key();
            let mut gc = majit_ir::descr::gc_cache().lock();
            // `descr.py cache[STRUCT][fieldname]` hit.
            if let Some(fd) = gc
                ._cache_field
                .get(&key)
                .and_then(|inner| inner.get(field_key))
            {
                return fd.clone() as DescrRef;
            }
            // Miss with the parent STRUCT published: mint through
            // `descr.py get_field_descr` so this resolution and the
            // walker's (`pyjitpl/dispatch.rs field_descr_ref_from_bh`) share
            // one Arc.  `descr.py parent_descr = get_size_descr(STRUCT)`
            // needs the `_cache_size` slot, so only take this route when it
            // is populated.
            if gc._cache_size.contains_key(&key) {
                let fd = gc.get_field_descr_declaring(
                    key,
                    &field_key,
                    Some(field.name.as_str()),
                    field.offset,
                    field.field_size,
                    field.field_type,
                    field.is_immutable,
                    field.is_quasi_immutable,
                    field.field_flag,
                    field.index,
                    false,
                    claimed_index,
                    // This is the route that lets a serialized spec populate
                    // the `(STRUCT, fieldname)` slot a layout producer will
                    // later find cached, so it carries the declaration rather
                    // than leaving the name fallback to seed it.
                    field.is_class_word,
                );
                return fd as DescrRef;
            }
        }
        // Cache miss / non-keyed parent — fall back to the descr group
        // field itself; the keyed path minted it through get_field_descr.
        let group = simple_descr_group_from_bh_size(parent);
        if let Some((pos, _)) =
            parent.all_fielddescrs.iter().enumerate().find(|(_, spec)| {
                spec.offset == field.offset && spec.field_key() == field.field_key()
            })
        {
            if let Some(descr) = group.field_descrs.get(pos) {
                return descr.clone() as DescrRef;
            }
        }
    }

    let mut descr = majit_ir::descr::SimpleFieldDescr::new_with_name(
        field.index,
        field.offset,
        field.field_size,
        field.field_type,
        field.is_immutable,
        field.field_flag,
        field.name.clone(),
        field.field_key().to_string(),
    )
    .with_quasi_immutable(field.is_quasi_immutable);
    // `new_with_name` seeded the answer from the display name; a spec that
    // declared one replaces that guess. A spec that declared nothing leaves it.
    if let Some(is_class_word) = field.is_class_word {
        descr = descr.with_class_word(is_class_word);
    }
    let arc: DescrRef = Arc::new(descr);
    // descr.py `get_field_descr` cache-miss path — register the
    // freshly-minted field descr so `compute_bitstrings` enumerates it.
    majit_ir::descr_registry::register_field(arc.clone());
    arc
}

/// Heapcache identity for one `BhDescr::Field` access.
///
/// `heaptracker.get_fielddescr_index_in` numbers a field by its slot, and
/// `HeapCache` keys a read by `Descr::index()`.  `fielddescrof` keeps that
/// slot on a `Vec` or fat-pointer part, then moves the offset and suffixes
/// the name (`.data`, `.len`, `.buf`, `.vtable`).  `.vtable` is the vtable word
/// of `&dyn`.  The two words then share one index, so a later read of the
/// vtable returns the data word.
///
/// A suffixed part whose parent slot does not describe this access takes
/// `stable_field_index` instead.  A part that still describes the slot keeps
/// the position, and so does every access that is not one of those parts.
fn heapcache_index_for_field_access(
    index_in_parent: Option<usize>,
    parent: Option<&majit_jitcode::jitcode::BhSizeSpec>,
    name: &str,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_field_signed: bool,
) -> u32 {
    let slot_describes_access = match (index_in_parent, parent) {
        (Some(index), Some(parent_spec)) => {
            parent_spec.all_fielddescrs.get(index).is_some_and(|spec| {
                spec.offset == offset
                    && spec.field_size == field_size
                    && spec.field_type == field_type
            })
        }
        _ => false,
    };
    let split_part = name.ends_with(".data")
        || name.ends_with(".len")
        || name.ends_with(".buf")
        || name.ends_with(".vtable");
    if parent.is_some() && index_in_parent.is_some() && split_part && !slot_describes_access {
        return stable_field_index(offset, field_size, field_type, is_field_signed);
    }
    match index_in_parent {
        Some(index) => index as u32,
        None => stable_field_index(offset, field_size, field_type, is_field_signed),
    }
}

pub(crate) fn bh_field_cache_key<'a>(owner: &str, name: &'a str) -> &'a str {
    // `descr.py` keys `cache[STRUCT][fieldname]` by the field name the
    // STRUCT already stores. The owner prefix is only a printable
    // qualification on the wire, so strip it in place instead of
    // allocating `format!("{owner}.")` on every kind-0 field.
    if owner.is_empty() {
        return name;
    }
    name.strip_prefix(owner)
        .and_then(|rest| rest.strip_prefix('.'))
        .unwrap_or(name)
}

/// Field half of an interior descr whose owner list has no row for it.
///
/// `get_size_descr` mints the parent before any field is attached, and
/// `bh_interior_field_specs_from_array_descr` copies that empty
/// `all_fielddescrs`. Running `simple_descr_group_from_bh_size` on the
/// empty list registers the empty size and leaves
/// `_cache_interiorfield` without the member
/// `opcode_published_descr_members` names by `BhFieldSpec::field_key`.
/// The spec already carries the offset, so the caller publishes that
/// field through `get_interiorfield_descr` under `field_key`.
fn field_descr_from_unlisted_interior_spec(
    field: &majit_jitcode::jitcode::BhFieldSpec,
) -> Arc<dyn majit_ir::descr::FieldDescr> {
    let mut descr = majit_ir::descr::SimpleFieldDescr::new_with_name(
        field.index,
        field.offset,
        field.field_size,
        field.field_type,
        field.is_immutable,
        field.field_flag,
        field.name.clone(),
        field.field_key().to_string(),
    )
    .with_quasi_immutable(field.is_quasi_immutable)
    .with_index_in_parent(field.index_in_parent);
    if let Some(is_class_word) = field.is_class_word {
        descr = descr.with_class_word(is_class_word);
    }
    Arc::new(descr)
}

/// Keyed sibling: accepts the u64 `cache_key` (= `path_hash(array_type_id)`)
/// so the freshly-minted `SimpleArrayDescr` lands in
/// `gc_cache._cache_array[LLType::Array(cache_key)]` in addition to
/// the snapshot order Vec.  Mirrors PyPy `cpu.arraydescrof(ARRAY)`
/// per-ARRAY cache identity (`descr.py get_array_descr`).  `cache_key == 0`
/// is the no-identity sentinel — registers via the non-keyed path.
pub fn make_struct_array_descr_full_keyed(
    descr_index: u32,
    base_size: usize,
    item_size: usize,
    len_offset: Option<usize>,
    type_id: u32,
    cache_key: u64,
    item_type: Type,
    interior_fields: &[majit_jitcode::jitcode::BhInteriorFieldSpec],
) -> DescrRef {
    use majit_ir::descr::{ArrayFlag, LLType, SimpleArrayDescr, gc_cache, try_downcast_arc};
    // `descr.py get_array_descr(gccache, ARRAY)` cache-or-mint:
    // an `LLType::Array(cache_key)` cache hit returns the existing Arc
    // (the `SimpleArrayDescr` in the slot — from a prior analyzer call
    // or from a runtime mint via `make_array_descr_with_full_id`); only
    // a miss mints a fresh
    // descr.  Matches PyPy `cpu.arraydescrof(ARRAY)` per-ARRAY object
    // identity — both pyre runtime mint sites and analyzer share a
    // single Arc per cache key.  `cache_key == 0` is the no-identity
    // sentinel (legacy non-keyed callers) — mint locally, no cache
    // publication.
    let array_descr_dyn: DescrRef = if cache_key != 0 {
        let array_key = LLType::Array(cache_key);
        let cached = gc_cache().lock().get_array_descr(
            array_key.clone(),
            base_size,
            item_size,
            ArrayFlag::Struct,
            item_type,
            len_offset.is_none(),
            len_offset.unwrap_or(0),
            false,
            '\x00',
        );
        // PyPy `gc.py init_array_descr` stamps `descr.tid`
        // from `layoutbuilder.get_type_id(A)` — a dense sequential
        // GC type id.  Pyre does not yet port the layoutbuilder
        // analog, so the cache-hit branch only
        // updates the per-trace `descr_index` and leaves
        // `SimpleArrayDescr.type_id` at its mint default (0, set in
        // `get_array_descr`).  The `BhDescr::Array.type_id` payload threaded
        // through this helper is the producer-side `path_hash(array_type_id)`
        // and already lands in `SimpleArrayDescr.cache_key` via the
        // `get_array_descr` cache-miss-mint stamp — structural identity (`cache_key`) is decoupled from GC tid
        // (`type_id`) per the `ArrayDescr::cache_key` trait doc.  Runtime
        // registrations (`SimpleArrayDescr` from the runtime mint
        // factories) carry their real GC tid at mint and win the cache
        // slot.
        cached.set_index(descr_index);
        cached
    } else {
        // No cache identity — local mint.  Two `cache_key == 0`
        // entries are intentionally distinct STRUCTs sharing the
        // no-identity sentinel; per-`make_array_descr` legacy callers
        // rely on this.
        let mut raw_array_descr = SimpleArrayDescr::with_flag(
            descr_index,
            base_size,
            item_size,
            type_id,
            item_type,
            ArrayFlag::Struct,
        );
        raw_array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
        let arc: DescrRef = Arc::new(raw_array_descr);
        majit_ir::descr_registry::register_array(arc.clone());
        arc
    };
    if interior_fields.is_empty() {
        return array_descr_dyn;
    }

    // Upcast the cached array descr to `Arc<dyn ArrayDescr>` for
    // `SimpleInteriorFieldDescr.array_descr` storage.  The cache slot
    // always holds a concrete `SimpleArrayDescr` (analyzer mint,
    // gc_cache internal mint, or runtime mint); downcast to the
    // concrete Arc type, then upcast.
    let array_descr_for_interior: Arc<dyn majit_ir::descr::ArrayDescr> =
        try_downcast_arc::<SimpleArrayDescr>(array_descr_dyn.clone())
            .expect("array descriptor cache must hold SimpleArrayDescr for struct arrays");

    let mut descrs: Vec<DescrRef> = Vec::new();
    for interior in interior_fields {
        let field = &interior.field;
        // A parent list that already holds this row keeps that group's
        // Arc. An empty list (or a list with no matching row) is minted
        // from the spec: `simple_descr_group_from_bh_size` on an empty
        // owner would register a field-less size and skip the slot.
        let (field_dyn, cache_name): (Arc<dyn majit_ir::descr::FieldDescr>, String) =
            if interior.owner.all_fielddescrs.is_empty() {
                (
                    field_descr_from_unlisted_interior_spec(field),
                    field.field_key().to_string(),
                )
            } else {
                let owner_group = simple_descr_group_from_bh_size(&interior.owner);
                let field_pos = interior
                    .owner
                    .all_fielddescrs
                    .iter()
                    .position(|candidate| {
                        candidate.index_in_parent == field.index_in_parent
                            && candidate.name == field.name
                    })
                    .unwrap_or(field.index_in_parent);
                if let Some(field_descr) = owner_group.field_descrs.get(field_pos) {
                    // Bare interior field name (`spec.name`) is the cache
                    // key per `descr.py cache[STRUCT][fieldname]` shape.
                    let bare_name = field
                        .name
                        .rsplit_once('.')
                        .map(|(_, name)| name.to_string())
                        .unwrap_or_else(|| field.name.clone());
                    (field_descr.clone(), bare_name)
                } else {
                    (
                        field_descr_from_unlisted_interior_spec(field),
                        field.field_key().to_string(),
                    )
                }
            };
        // `descr.py get_interiorfield_descr` cache-or-mint is keyed on
        // the outer ARRAY's lltype identity.  When the outer array
        // carries `cache_key != 0`, route through
        // `_cache_interiorfield[(LLType::Array(cache_key), name, "")]`
        // so both analyzer and runtime share one Arc per `(ARRAY, name)`
        // tuple.  With `cache_key == 0` (no-identity outer array) PyPy
        // has no "merge several ARRAYs' interiors into one slot"
        // behavior — local mint a fresh `SimpleInteriorFieldDescr` per
        // call so distinct no-identity arrays do not alias on their
        // interior field descrs.
        let ifd: DescrRef = if cache_key != 0 {
            gc_cache().lock().get_interiorfield_descr(
                LLType::Array(cache_key),
                cache_name,
                String::new(),
                array_descr_for_interior.clone(),
                field_dyn,
            )
        } else {
            Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
                u32::MAX,
                array_descr_for_interior.clone(),
                field_dyn,
            )) as DescrRef
        };
        // Per-trace `interior.index` stamp matches the analyzer's
        // `cc.interiorfielddescrof` codewriter idx convention.
        ifd.set_index(interior.index);
        descrs.push(ifd);
    }

    // `descr.py arraydescr.all_interiorfielddescrs = descrs`
    // set-once via OnceLock.  Cache-hit case: a prior populator already
    // set the list; our set is a no-op which is the desired semantic.
    array_descr_for_interior.set_all_interiorfielddescrs(descrs);
    array_descr_dyn
}

/// Concrete `JitCodeDescr` adapter for `inline_call_*` opcodes.
///
/// RPython parity: `JitCode(AbstractDescr)` carries `fnaddr` +
/// `calldescr` + the callee's bytecode body and is emitted directly as
/// the descr operand of `inline_call_*`. The codewriter side surfaces
/// this as `BhDescr::JitCode { jitcode_index, fnaddr, calldescr }`
/// (`majit-jitcode/src/codewriter/jitcode.rs`); the trace-side
/// walker (`jitcode_dispatch.rs::WalkContext`) consumes
/// `&[Arc<dyn Descr>]` and queries `as_jitcode_descr()` /
/// `jitcode_index()`.
///
/// `PyreJitCodeDescr` bridges those two layers: production callers
/// build a `Vec<DescrRef>` from the codewriter's `BhDescr` pool via
/// [`make_descr_from_bh`] (each `BhDescr::JitCode` wraps in this
/// struct so the walker's `as_jitcode_descr() -> Some(&self)` cast
/// succeeds; Field/Array/Size become `PyreFieldDescr` /
/// `SimpleArrayDescr` / `PyreSizeDescr`; `Call` becomes a
/// `MetaCallDescr` carrying the codewriter's `EffectInfo`).
///
/// Tests in `jitcode_dispatch.rs` previously used a `TestJitCodeDescr`
/// duplicate of this shape — production code now goes through the same
/// type so the test fixture can be progressively replaced without
/// behaviour drift.
#[derive(Debug)]
pub struct PyreJitCodeDescr {
    jitcode_index: usize,
}

impl PyreJitCodeDescr {
    /// Build a `PyreJitCodeDescr` with the given runtime jitcode index.
    /// `jitcode_index` indexes into the runtime's all-jitcodes table
    /// (`pyre-jit-trace/src/jitcode_runtime.rs::ALL_JITCODES`); the
    /// walker's `sub_jitcode_lookup` resolves it to the callee's body.
    pub fn new(jitcode_index: usize) -> Self {
        Self { jitcode_index }
    }
}

impl Descr for PyreJitCodeDescr {
    fn as_jitcode_descr(&self) -> Option<&dyn JitCodeDescr> {
        Some(self)
    }
}

impl JitCodeDescr for PyreJitCodeDescr {
    fn jitcode_index(&self) -> usize {
        self.jitcode_index
    }
}

/// Build a `DescrRef` carrying a `PyreJitCodeDescr`. Production callers
/// use this when materializing the descr pool from a codewriter
/// `&[BhDescr]` (`BhDescr::JitCode { jitcode_index, .. }` → this
/// adapter).
pub fn make_jitcode_descr(jitcode_index: usize) -> DescrRef {
    Arc::new(PyreJitCodeDescr::new(jitcode_index))
}

/// Trace-side `SwitchDictDescr` adapter. The bytecode blackhole keeps
/// `BhDescr::Switch` directly; the MIFrame walker needs an `Arc<dyn Descr>`
/// slot for the same `Assembler.descrs` index.
#[derive(Debug)]
pub struct PyreSwitchDescr {
    dict: std::collections::HashMap<i64, usize>,
    const_keys_in_order: Vec<i64>,
}

impl PyreSwitchDescr {
    pub fn new(dict: std::collections::HashMap<i64, usize>) -> Self {
        let mut const_keys_in_order: Vec<i64> = dict.keys().copied().collect();
        const_keys_in_order.sort_unstable();
        Self {
            dict,
            const_keys_in_order,
        }
    }
}

impl Descr for PyreSwitchDescr {
    fn repr(&self) -> String {
        let entries = self
            .const_keys_in_order
            .iter()
            .map(|key| {
                let target = self
                    .dict
                    .get(key)
                    .expect("const_keys_in_order must mirror SwitchDictDescr.dict");
                format!("{key}: {target}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("<SwitchDictDescr {{{entries}}}>")
    }

    fn as_switch_descr(&self) -> Option<&dyn SwitchDescr> {
        Some(self)
    }
}

impl SwitchDescr for PyreSwitchDescr {
    fn lookup(&self, value: i64) -> Option<usize> {
        self.dict.get(&value).copied()
    }

    fn const_keys_in_order(&self) -> &[i64] {
        &self.const_keys_in_order
    }
}

#[cfg(test)]
mod switch_descr_tests {
    use super::*;

    #[test]
    fn pyre_switch_descr_repr_matches_rpython_switchdictdescr() {
        let descr = PyreSwitchDescr::new(std::collections::HashMap::from([(9, 23), (5, 17)]));

        assert_eq!(
            <PyreSwitchDescr as Descr>::repr(&descr),
            "<SwitchDictDescr {5: 17, 9: 23}>"
        );
        assert_eq!(descr.const_keys_in_order(), &[5, 9]);
    }
}

/// Trace-side adapter for pyre's Rust-vtable method descriptor.
#[derive(Debug)]
pub struct PyreVtableMethodDescr {
    trait_root: String,
    method_name: String,
}

impl PyreVtableMethodDescr {
    pub fn new(trait_root: String, method_name: String) -> Self {
        Self {
            trait_root,
            method_name,
        }
    }
}

impl Descr for PyreVtableMethodDescr {
    fn repr(&self) -> String {
        format!(
            "VtableMethodDescr({}::{})",
            self.trait_root, self.method_name
        )
    }
}

/// `assembler.py Assembler.descrs` parity adapter — translate one
/// codewriter-side `BhDescr` slot (`majit-jitcode/src/codewriter/jitcode.rs`)
/// into the matching trace-side `Arc<dyn Descr>` so trace ops emitted
/// by the walker (`crate::jitcode_dispatch::dispatch_via_miframe`) can carry
/// real-content descrs instead of `make_fail_descr` placeholders.
///
/// RPython parity: in upstream the metainterp + blackhole interpreter
/// share one `metainterp_sd.all_descrs` list — the same Python object
/// is the field/array/call descr regardless of which path is reading
/// it. pyre carries the codewriter-side typed list (`BhDescr`) and the
/// trait-side `Arc<dyn Descr>` view (`DescrRef`) as separate Rust
/// types because `Arc<dyn Descr>` cannot be downcast safely; this
/// adapter is the single point that bridges them.
///
/// Every branch builds the same descriptor kind carried by the
/// codewriter-side `BhDescr`:
/// * `Field` — `offset`, `field_size`, `field_type`, signedness, and
///   immutable/quasi-immutable flags are preserved.
/// * `Array` — `base_size`, `itemsize`, `type_id`, item type, signedness,
///   and array-of-structs classification are preserved.
/// * `Size` — `size`, `type_id`, and `vtable` are preserved.
/// * `Call` — `BhCallDescr.arg_classes` (e.g. `"iR"`) maps to
///   `Vec<Type>` per char (`i`->Int, `r`->Ref, `f`->Float; `R`/`I`/`F`
///   var-list markers split into the per-arg base type), and
///   `result_type` (one of `'i','r','f','v'`) maps to the `Type` of
///   the call result. `extra_info` is threaded into
///   `make_call_descr_with_effect`, preserving RPython `call.py:320`
///   effectinfo_from_writeanalyze parity for descr cache keys and
///   residual-call classification.
/// * `Switch` / `VableField` / `VableArray` / `VtableMethod` — trace-side
///   adapters preserve the descriptor slot instead of substituting a
///   fail-descr placeholder.
pub fn make_descr_from_bh(bh: &majit_jitcode::jitcode::BhDescr) -> DescrRef {
    use majit_jitcode::jitcode::BhDescr;
    match bh {
        BhDescr::Field {
            offset,
            field_size,
            field_type,
            field_flag,
            is_field_signed,
            is_immutable,
            is_quasi_immutable,
            index_in_parent,
            parent,
            name,
            owner,
            ..
        } => {
            let field_key = bh_field_cache_key(owner, name);
            // list-append codewriter descr-bridge: `_handle_list_call`
            // (codewriter/jtransform.rs) lowers Integer-strategy list
            // ops to fields on the dotted nested names
            // `int_items.{len,block}` (owner `W_ListObject`).
            // `bh_field_name` treats the dotted name as already-qualified,
            // and `W_ListObject` is a runtime Rust type absent from the
            // codewriter struct layouts, so the assembler's `fielddescrof`
            // leaves these at offset 0 (the list header). Map the leaves to
            // the canonical `W_LIST_DESCR_GROUP` entries the walker-native
            // list specializations already use so an assembled codewriter
            // list body addresses `IntArray.{len,block}` rather
            // than the header.
            //
            // This runs BEFORE the parent-group lookup below, not after: when
            // the codewriter DOES model the parent struct, that lookup answers
            // with the parent group's own entry for the same offset, and the
            // field ends up carrying two descrs — the parent group's for a
            // codewriter-lowered body, `W_LIST_DESCR_GROUP`'s for the
            // walker-native specializations.  The heapcache and the optimizer's
            // heap pass both key on descr identity, so the split silently
            // breaks aliasing: the `w_list_append` sub-walk's
            // `SetfieldGc(int_items.len)` does not invalidate the `len(xs)`
            // read that follows it, which then folds to the pre-append length
            // (one skipped `list.pop(0)` per compiled loop entry). One field is
            // one descr — `metainterp_sd.all_descrs` has no second entry for a
            // field just because a different interpreter reached it.
            // Same split, one struct up: the shared `PyObject` header's
            // `w_class`. The walker pins a value's Python-level class by
            // reading offset 8 through `w_class_descr()`; a codewriter-lowered
            // body testing the same header — `is_plain_int1` reading
            // `value.w_class` (listobject.rs) — reaches it through the modelled
            // `PyObject` parent, whose group entry is a different identity for
            // the same field. The pinned constant then never reached the
            // second read, so the strict subclass test stayed symbolic and
            // re-emitted the load plus its null and equality tests.
            //
            // Like the leaves below this runs BEFORE the parent-group lookup,
            // which would otherwise answer with the parent's own entry.
            //
            // Only when the two spellings describe the same memory access.
            // Both universes size a pointer field from the target
            // (`new_w_class_field_descr` uses `WORD`, the codewriter
            // `layout::target_word_size()`), so the widths agree on every
            // target; the check stays as the guard that keeps a
            // differently-sized spelling from being widened onto the
            // canonical descr.
            if name.as_str() == "w_class"
                && matches!(
                    owner.as_str(),
                    "PyObject" | "pyre_object::pyobject::PyObject"
                )
            {
                let canonical = &*W_CLASS_FIELD_DESCR;
                if canonical.offset() == *offset
                    && canonical.field_size() == *field_size
                    && canonical.field_type() == *field_type
                {
                    return w_class_descr();
                }
            }
            // `ExecutionContext` is the second red portal input and is not a
            // GC object. PUSH_EXC_INFO / POP_EXCEPT and portal guards use the
            // declared `EC_DESCR_GROUP`, while a translated helper body names
            // the same Rust fields through Charon's qualified owner. PyPy's
            // `GcCache.get_field_descr` returns one descriptor object for one
            // `(STRUCT, fieldname)`; keeping a second Arc here makes the heap
            // optimizer miss the intervening `sys_exc_value` read and erase a
            // live handler store. Bridge every EC field the walker itself
            // emits before the generic parent-group lookup, as the shared
            // `PyObject.w_class` bridge above already does.
            if matches!(
                owner.as_str(),
                "ExecutionContext" | "executioncontext::ExecutionContext"
            ) {
                let canonical = match name.as_str() {
                    "sys_exc_value" => Some(ec_sys_exc_value_descr()),
                    "topframeref" => Some(ec_topframeref_descr()),
                    "w_tracefunc" => Some(ec_w_tracefunc_descr()),
                    "profilefunc" => Some(ec_profilefunc_descr()),
                    "current_gen_or_coroutine" => Some(ec_current_gen_or_coroutine_descr()),
                    _ => None,
                };
                if let Some(canonical) = canonical
                    && let Some(field) = canonical.as_field_descr()
                    && field.offset() == *offset
                    && field.field_size() == *field_size
                    && field.field_type() == *field_type
                {
                    return canonical;
                }
            }
            // `GilReadyState.gil_ready` is the process-wide `gil_ready?`
            // word. The codewriter has no `?` attribute on it (that would
            // emit `record_quasiimmut_field`, which the production
            // blackhole builder does not implement) and names the owner by
            // leaf or by module path. Bridge to the one quasi descr before
            // the generic parent lookup mints a mutable twin.
            if (owner.as_str() == "GilReadyState" || owner.as_str().ends_with("::GilReadyState"))
                && name.as_str() == "gil_ready"
            {
                let canonical = gil_ready_descr();
                if let Some(field) = canonical.as_field_descr()
                    && field.offset() == *offset
                    && field.field_size() == *field_size
                    && field.field_type() == *field_type
                {
                    return canonical;
                }
            }
            if owner.as_str() == "W_ListObject" {
                match name.as_str() {
                    "int_items.len" => return list_int_items_len_descr(),
                    "int_items.block" => return list_int_items_block_descr(),
                    "float_items.len" => return list_float_items_len_descr(),
                    "float_items.block" => return list_float_items_block_descr(),
                    "bytes_items.len" => return list_bytes_items_len_descr(),
                    "bytes_items.block" => return list_bytes_items_block_descr(),
                    "ascii_items.len" => return list_ascii_items_len_descr(),
                    "ascii_items.block" => return list_ascii_items_block_descr(),
                    // A bare `int_items` / `float_items` read addresses the
                    // typed-storage struct base, which is its first field
                    // (`block`, `INT_ARRAY_BLOCK_OFFSET == 0`) — the same
                    // offset + `Ref` type as the `.block` leaf above. The
                    // `w_list_append` body reads `int_items` directly before
                    // reaching `.len`; bridge it to the canonical
                    // `.block` group entry so the read resolves a parent_descr.
                    "int_items" => return list_int_items_block_descr(),
                    "float_items" => return list_float_items_block_descr(),
                    "bytes_items" => return list_bytes_items_block_descr(),
                    "ascii_items" => return list_ascii_items_block_descr(),
                    // The `w_list_append` body's `match list.strategy` reads the
                    // header `strategy` field directly.  The codewriter resolves
                    // its offset but produces a `SimpleFieldDescr` with no
                    // `parent_descr` backreference, which the optimizer's
                    // `ensure_ptr_info_arg0` rejects.  Bridge it to the canonical
                    // group entry (correct offset + parent_descr) like the
                    // `int_items.*` leaves above.
                    "strategy" => return list_strategy_descr(),
                    "length" => return list_length_descr(),
                    "items" => return list_items_descr(),
                    _ => {}
                }
            }
            // list-append object-strategy capacity read: `list.obj_capacity` lowers
            // to getfield_gc_r(items) + getfield_gc_i(block.capacity). The
            // block's offset-0 GcArray length header IS the allocated
            // capacity (immutable for the block's lifetime).
            //
            // This must precede the generic parent-group lookup. Charon gives
            // `ItemsBlock.capacity` a real parent, whose local field index is
            // zero; rebuilding that group would therefore return a distinct
            // index-0 descriptor. Walker-native list promotion seeds the
            // capacity under the canonical descriptor's stable index, and
            // RPython has only one FieldDescr identity for this field.
            if owner.as_str() == "ItemsBlock" && name.as_str() == "capacity" {
                return items_block_capacity_descr();
            }
            if let Some(parent) = parent {
                if parent.type_id != 0 {
                    // Before the lookup, not after it: a declared group that
                    // has not been forced yet leaves the slot empty, and the
                    // rebuild below would then fill it from the spec.
                    force_declared_group(parent.type_id);
                    let key = majit_ir::descr::LLType::Struct(parent.type_id);
                    let cached = {
                        let gc = majit_ir::descr::gc_cache().lock();
                        // The first field of a STRUCT publishes every row under
                        // `BhFieldSpec::field_key`. A later row whose wire name
                        // strips to a different spelling still names that row.
                        gc._cache_field.get(&key).and_then(|inner| {
                            if let Some(fd) = inner.get(field_key) {
                                return Some(fd.clone());
                            }
                            let spec = parent.all_fielddescrs.iter().find(|spec| {
                                spec.offset == *offset && spec.field_size == *field_size
                            })?;
                            inner.get(spec.field_key()).cloned()
                        })
                    };
                    if let Some(fd) = cached {
                        return fd as DescrRef;
                    }
                    let group = simple_descr_group_from_bh_size(parent);
                    if let Some((pos, _)) =
                        parent.all_fielddescrs.iter().enumerate().find(|(_, spec)| {
                            spec.offset == *offset && spec.field_key() == field_key
                        })
                    {
                        // `pos` is the slot this reader will index; the descr's
                        // own `index_in_parent` (`descr.py:228`) is what every
                        // later consumer indexes by
                        // (`optimizeopt/info.rs force_box`).  They are two
                        // producers' answers to one question and nothing else
                        // compares them: the cache-hit return above and
                        // `get_field_descr`'s own `derive_index_in_parent` both
                        // resolve by NAME, so a stale rank is silently replaced
                        // rather than reported.  Count it where it is still
                        // visible.
                        majit_ir::descr::census_attached_index(pos, *index_in_parent);
                        if let Some(descr) = group.field_descrs.get(pos) {
                            return descr.clone() as DescrRef;
                        }
                    }
                }
            }
            // list-append codewriter descr-bridge: a codewriter-lowered body reads a
            // box payload (`W_IntObject.intval` / `W_BoolObject.intval` /
            // `W_FloatObject.floatval`) through the producer's struct-layout
            // `SimpleFieldDescr` (header modeled, `index_in_parent` = 2).  The
            // walker materializes those same boxes through the single-field
            // `W_*_DESCR_GROUP` (`index_in_parent` = 0).  Both address the same
            // offset, but the optimizer's virtual-field cache matches on descr
            // identity, so a sub-walk read with the producer descr misses the
            // virtual built with the group descr — the box is forced and the
            // field read folds to uninitialized storage.  Return the canonical
            // group entry so the read matches the box-creation descr and folds
            // to the carried unboxed value (the payload is genuinely immutable
            // post-construction, matching the group entry's flag).
            match (owner.as_str(), name.as_str()) {
                ("W_IntObject" | "pyre_object::intobject::W_IntObject", "intval") => {
                    return int_intval_descr();
                }
                ("W_BoolObject" | "pyre_object::boolobject::W_BoolObject", "intval") => {
                    return bool_intval_descr();
                }
                ("W_FloatObject" | "pyre_object::floatobject::W_FloatObject", "floatval") => {
                    return float_floatval_descr();
                }
                ("W_ComplexObject" | "pyre_object::complexobject::W_ComplexObject", "real") => {
                    return complex_real_descr();
                }
                ("W_ComplexObject" | "pyre_object::complexobject::W_ComplexObject", "imag") => {
                    return complex_imag_descr();
                }
                _ => {}
            }
            let full_name = if owner.is_empty() || name.contains('.') {
                name.clone()
            } else {
                format!("{owner}.{name}")
            };
            // RPython `descr.py FieldDescr.get_index()` returns
            // the value `heaptracker.get_fielddescr_index_in(STRUCT,
            // name)` recorded into `FieldDescr.index` at construction
            // time (`descr.py`).  Pyre's `BhDescr::Field` carries
            // that as `index_in_parent`; thread it through as
            // `BhFieldSpec.index` so the `parent` matching fallback
            // produces a `SimpleFieldDescr` whose `index()` matches the
            // upstream value rather than a `u32::MAX` sentinel.
            let field = majit_jitcode::jitcode::BhFieldSpec {
                // `index()` is the key the `HeapCache` files a field read
                // under, so it has to separate every field one object can be
                // read through.  A numbered field answers with its position;
                // an unnumbered one used to answer `0`, which is a position
                // too — the one held by the first own field of whatever struct
                // is read off that same object.  `PyObject.ob_type` carries no
                // number (the positional census skips the header words), so it
                // filed its read under slot 0 and `W_StringIO.buffer` — offset
                // 16, position 0 — hit that entry and was handed the type
                // pointer, whose `+24` word then reached `arraylen_gc` as an
                // array.  Number the unnumbered ones off the payload instead,
                // the scheme the runtime-minted field descrs already use: its
                // `FIELD_DESCR_TAG` is disjoint from the positional numbers,
                // and two fields of one object cannot share an offset.
                //
                // A fat-pointer part is numbered and still not that position:
                // `fielddescrof` copies the parent slot, then reads a different
                // word.  `heapcache_index_for_field_access` keeps the position
                // only when the slot describes the access.
                index: heapcache_index_for_field_access(
                    *index_in_parent,
                    parent.as_deref(),
                    name,
                    *offset,
                    *field_size,
                    *field_type,
                    *is_field_signed,
                ),
                field_key: field_key.to_string(),
                name: full_name,
                offset: *offset,
                field_size: *field_size,
                field_type: *field_type,
                field_flag: *field_flag,
                is_field_signed: *is_field_signed,
                is_immutable: *is_immutable,
                is_quasi_immutable: *is_quasi_immutable,
                index_in_parent: index_in_parent.unwrap_or(0),
                // A standalone `BhDescr::Field` carries no layout, so nothing
                // here declares and the name is what the rebuild reads.
                is_class_word: None,
            };
            // The claim itself goes beside the spec, unflattened.
            field_descr_from_bh_field(
                &field,
                parent.as_ref().map(std::sync::Arc::as_ref),
                *index_in_parent,
            )
        }
        BhDescr::Array {
            base_size,
            itemsize,
            len_offset,
            type_id,
            item_type,
            is_array_of_structs,
            is_item_signed,
            ei_index,
            array_type_id,
            interior_fields,
            ..
        } => {
            // Old serialized BhDescr values default `gc_type_id` to zero.
            // Resolve that legacy structural identity through the GC cache
            // before publishing a runtime descriptor, exactly as the
            // blackhole/materialization paths do.
            let resolved_gc_type_id = bh.resolve_gc_tid();
            let descr = if *is_array_of_structs {
                // Stamp the runtime collector tid into `_cache_array`
                // before the keyed factory cache-hits, so a
                // `DICTENTRYARRAY` `NEW_ARRAY_CLEAR` keeps
                // `i64_pyobject_entries_gc_type_id` rather than the
                // analyzer mint `init_array_descr` assigned.
                // `descr.py get_array_descr` / `cpu.arraydescrof(ARRAY)`.
                if let Some(atid) = array_type_id.as_deref() {
                    let _ = crate::state::runtime_gcarray_descr(atid);
                }
                // `descr.py get_array_descr(gccache, ARRAY)`:
                // the u64 `*type_id` from `BhDescr::Array` is the cache
                // key (`path_hash` of the producer-side `array_type_id`,
                // see `BhSizeSpec.type_id` doc); thread it into the
                // keyed factory so `gc_cache._cache_array[LLType::Array(
                // cache_key)]` is populated and subsequent lookups
                // resolve to the same Arc.  The u32 truncation for the
                // SimpleArrayDescr gc tid is a TODO
                // (gc tid should come from `init_array_descr`
                // sequential allocation).
                make_struct_array_descr_full_keyed(
                    u32::MAX,
                    *base_size,
                    *itemsize,
                    *len_offset,
                    resolved_gc_type_id,
                    *type_id,
                    *item_type,
                    interior_fields,
                )
            } else if let Some(runtime) = array_type_id
                .as_deref()
                .and_then(crate::state::runtime_gcarray_descr)
                .or_else(|| {
                    // `descr.py get_array_descr` cache hit: `type_id` is the
                    // `cache_key` the producer read off a published descr
                    // (`BhDescr::from_array_descr` carries no spelling).
                    (*type_id != 0)
                        .then(|| {
                            majit_ir::descr::gc_cache()
                                .lock()
                                ._cache_array
                                .get(&majit_ir::descr::LLType::Array(*type_id))
                                .cloned()
                        })
                        .flatten()
                })
            {
                // `cpu.arraydescrof(ARRAY)` is one descr per ARRAY: a list
                // backing block the runtime already describes keeps that
                // descr, with its GC tid, instead of a second one the heap
                // cache would treat as a different array.
                let ad = runtime
                    .as_array_descr()
                    .expect("runtime gcarray descr is an ArrayDescr");
                assert!(
                    ad.base_size() == *base_size && ad.item_size() == *itemsize,
                    "build-time layout of {array_type_id:?} disagrees with the runtime block",
                );
                runtime
            } else {
                // `descr.py get_array_descr gccache._cache_array[ARRAY_OR_STRUCT]`
                // is keyed on lltype object identity; thread the
                // codewriter `array_type_id` across the BhDescr
                // boundary into the runtime `ArrayDescrKey` so two
                // BhDescr::Array entries that disagree only on the
                // Rust type spelling don't collapse to the same
                // registry slot (`set_ei_index` clobber).
                make_array_descr_with_full_id(
                    *base_size,
                    *itemsize,
                    resolved_gc_type_id,
                    *len_offset,
                    *item_type,
                    *is_item_signed,
                    array_type_id.clone(),
                )
            };
            // `effectinfo.py compute_bitstrings` republish: the
            // codewriter-side `array_index` carried across the BhDescr
            // boundary lands on the runtime descr so heap.rs
            // `force_from_effectinfo` (`heap.py`) reads the
            // same bitstring slot the producer wrote.
            if *ei_index != u32::MAX {
                descr.set_ei_index(*ei_index);
            }
            descr
        }
        BhDescr::Size {
            size,
            type_id,
            vtable,
            owner,
            all_fielddescrs,
            is_gc_managed,
            ..
        } => {
            // Publish this module's group before the lookup. A `BhDescr`
            // that arrives first would keep a vtable captured in the
            // producing process and a field list with no `w_class` edge.
            force_declared_group(*type_id);
            // `descr.py get_size_descr` cache-hit semantics:
            // when the producer's `type_id` matches a runtime publish
            // (`build_object_descr_group_with_def_path` →
            // `register_keyed_size`), the published `Arc<PyreSizeDescr>`
            // is the canonical object identity for that STRUCT.  Return
            // it directly so analyzer side-tables, runtime allocations,
            // and BhDescr round-trip all share one `Arc<dyn Descr>` —
            // matching PyPy `cache[STRUCT]` per-tuple identity.  Falls
            // through to the legacy mint path on cache miss (transient
            // empty-field `bh_new` descrs from `pyre-jit/src/eval.rs`,
            // test fixtures, etc.).
            if *type_id != 0 {
                let key = majit_ir::descr::LLType::Struct(*type_id);
                let hit = majit_ir::descr::gc_cache()
                    .lock()
                    ._cache_size
                    .get(&key)
                    .cloned();
                if let Some(descr) = hit {
                    return descr;
                }
            }
            // RPython `descr.py get_size_descr` → `:188 init_size_descr`
            // populates `SizeDescr.all_fielddescrs` (and the
            // `gc_fielddescrs` subset) from
            // `heaptracker.all_fielddescrs(STRUCT)` so consumers like
            // `info.py init_fields` (`ptr_info.rs`'s `init_fields`)
            // see the full struct field list off the descr without a
            // round-trip through the codewriter.  When the producer
            // shipped a non-empty `all_fielddescrs`, build the parent
            // `SimpleSizeDescr` via the cyclic `make_simple_descr_group`
            // path so `Arc<SimpleFieldDescr>` parents back-reference
            // the same `SimpleSizeDescr` (`descr.py` parent slot).
            // The transient short-lived `BhDescr::Size` constructed in
            // `pyre-jit/src/eval.rs` (`bh_new` / `bh_new_with_vtable`
            // dispatch) carries an empty list and falls through to the
            // bare ctor, which is the existing test-helper shape.
            let headerless = owner == HEADERLESS_SIZE_OWNER_MARKER;
            if all_fielddescrs.is_empty() && !headerless {
                // TODO: `make_size_descr_with_type_and_vtable`
                // takes the u32 gc tid; `*type_id` is the u64 cache key.
                // Truncate `as u32` until gc_cache routing.
                make_size_descr_with_type_and_vtable(*size, *type_id as u32, *vtable as usize)
            } else {
                let spec = majit_jitcode::jitcode::BhSizeSpec {
                    size: *size,
                    type_id: *type_id,
                    vtable: *vtable,
                    owner: owner.clone(),
                    is_gc_managed: *is_gc_managed,
                    headerless,
                    all_fielddescrs: all_fielddescrs.clone(),
                };
                simple_descr_group_from_bh_size(&spec).size_descr.clone()
            }
        }
        BhDescr::InteriorField { array, field } => {
            // `descr.py InteriorFieldDescr(arraydescr, fielddescr)`:
            // recompose the array + field sub-descrs.  The nested
            // BhDescrs are always `BhDescr::Array` / `BhDescr::Field`
            // (`BhDescr::from_interior_field_descr`); a short stream is an
            // encoder bug surfaced here rather than silently mis-typed.
            let (base_size, itemsize, len_offset, type_id, ei_index, item_type, interior_fields) =
                match array.as_ref() {
                    BhDescr::Array {
                        base_size,
                        itemsize,
                        len_offset,
                        type_id,
                        ei_index,
                        item_type,
                        interior_fields,
                        ..
                    } => (
                        *base_size,
                        *itemsize,
                        *len_offset,
                        *type_id,
                        *ei_index,
                        *item_type,
                        interior_fields,
                    ),
                    other => panic!(
                        "BhDescr::InteriorField array slot must be BhDescr::Array, got {other:?}"
                    ),
                };
            // `descr.py` `get_interiorfield_descr` builds the interior
            // descr from `get_array_descr(gc_ll_descr, ARRAY)` — the SAME
            // cached arraydescr the `BhDescr::Array` arm restores.  Route the
            // outer struct array through the identical keyed factory so the
            // rebuilt `InteriorFieldDescr.arraydescr` shares the
            // `gc_cache._cache_array[LLType::Array(cache_key)]` identity rather
            // than a fresh local mint, and re-attaches `all_interiorfielddescrs`
            // (`descr.py`).  `type_id` is the producer-side cache key
            // (`ArrayDescr.cache_key` = `path_hash(array_type_id)`); the u32
            // gc-tid slot is ignored on the cache path (`init_array_descr`
            // stamps the real tid), keeping ARRAY identity cache key separate
            // from GC layout tid.  On a cache MISS the factory requests
            // `ArrayFlag::Struct`, so the minted arraydescr is FLAG_STRUCT; on a
            // cache HIT the existing slot is reused verbatim, so the resolved
            // arraydescr is re-checked against `descr.py`'s
            // `assert arraydescr.flag == FLAG_STRUCT` below.
            let array_dyn = make_struct_array_descr_full_keyed(
                u32::MAX,
                base_size,
                itemsize,
                len_offset,
                type_id as u32,
                type_id,
                item_type,
                interior_fields,
            );
            // The cache slot always holds a concrete `SimpleArrayDescr`
            // (analyzer / gc_cache / runtime mint).
            let array_descr: Arc<dyn majit_ir::descr::ArrayDescr> =
                match majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleArrayDescr>(
                    array_dyn,
                ) {
                    Ok(simple) => simple,
                    Err(other) => panic!(
                        "BhDescr::InteriorField array resolved to an unknown ArrayDescr type: {other:?}"
                    ),
                };
            // `descr.py InteriorFieldDescr.__init__`:
            //   assert arraydescr.flag == FLAG_STRUCT
            // The factory requests `ArrayFlag::Struct` on a cache MISS, but a
            // cache HIT returns the existing `LLType::Array(cache_key)` slot
            // verbatim — which may be a non-struct array (a contaminated cache
            // key).  Reject those here rather than silently rebuilding the
            // InteriorFieldDescr around a non-struct array.
            assert!(
                array_descr.is_array_of_structs(),
                "BhDescr::InteriorField arraydescr must be FLAG_STRUCT (descr.py:389)"
            );
            // `effectinfo.py compute_bitstrings` republish, as the
            // `BhDescr::Array` arm does: the struct array is the descr the
            // frozen array bitstrings name.
            if ei_index != u32::MAX {
                array_descr.set_ei_index(ei_index);
            }
            let (offset, field_size, field_type, field_flag, index_in_parent, name) =
                match field.as_ref() {
                    BhDescr::Field {
                        offset,
                        field_size,
                        field_type,
                        field_flag,
                        index_in_parent,
                        name,
                        ..
                    } => (
                        *offset,
                        *field_size,
                        *field_type,
                        *field_flag,
                        *index_in_parent,
                        name.clone(),
                    ),
                    other => panic!(
                        "BhDescr::InteriorField field slot must be BhDescr::Field, got {other:?}"
                    ),
                };
            // Bare interior field name (`descr.py cache[STRUCT][fieldname]`
            // shape) — the `get_interiorfield_descr` cache key the
            // `make_struct_array_descr_full_keyed` interior loop used.
            let bare_name = name
                .rsplit_once('.')
                .map(|(_, n)| n.to_string())
                .unwrap_or_else(|| name.clone());
            // `descr.py` `InteriorFieldDescr.get_index()` delegates to
            // `fielddescr.get_index()` (= `index_in_parent`), read back by
            // `info.py` getinteriorfield/setinteriorfield_virtual.
            let field_descr: Arc<dyn majit_ir::descr::FieldDescr> = Arc::new(
                majit_ir::descr::SimpleFieldDescr::new_with_name(
                    u32::MAX,
                    offset,
                    field_size,
                    field_type,
                    false,
                    field_flag,
                    name.clone(),
                    name,
                )
                .with_index_in_parent(index_in_parent.unwrap_or(0)),
            );
            // `descr.py get_interiorfield_descr` cache-or-mint keyed on
            // the outer ARRAY identity, so the analyzer's
            // `cc.interiorfielddescrof`, the array's `all_interiorfielddescrs`,
            // and this restore share one `Arc` per `(ARRAY, name)` tuple — the
            // `make_struct_array_descr_full_keyed` interior loop above already
            // populated `_cache_interiorfield` (with the per-trace
            // `Descr::index()`) for `cache_key != 0`, so this hits and returns
            // that same descr.  `cache_key == 0` is the legacy no-identity
            // sentinel — local mint a fresh `SimpleInteriorFieldDescr` (the
            // `u32::MAX` slot is the "no setup_descrs index assigned" default;
            // `get_index()` comes from `field_descr.index_in_parent` above).
            if type_id != 0 {
                majit_ir::descr::gc_cache().lock().get_interiorfield_descr(
                    majit_ir::descr::LLType::Array(type_id),
                    bare_name,
                    String::new(),
                    array_descr,
                    field_descr,
                )
            } else {
                Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
                    u32::MAX,
                    array_descr,
                    field_descr,
                ))
            }
        }
        BhDescr::Call { calldescr } => make_call_descr_from_bh(calldescr),
        BhDescr::JitCode { jitcode_index, .. } => make_jitcode_descr(*jitcode_index),
        BhDescr::Switch { dict, .. } => Arc::new(PyreSwitchDescr::new(dict.clone())),
        BhDescr::VableField { index } => {
            crate::frame_layout::pyframe_static_field_descr(*index as u16)
        }
        BhDescr::VableArray { index } => {
            crate::frame_layout::pyframe_array_field_descr(*index as u16)
        }
        BhDescr::VtableMethod {
            trait_root,
            method_name,
        } => Arc::new(PyreVtableMethodDescr::new(
            trait_root.clone(),
            method_name.clone(),
        )),
    }
}

/// Look a serialized raw-set member back up in the process-global gccache.
///
/// **Lookup only — never mint.** `descr.py get_field_descr` is
/// cache-or-mint because upstream calls it from `heaptracker.all_fielddescrs
/// (STRUCT)`, which always has the whole `STRUCT` in hand and therefore mints
/// the parent `SizeDescr` with its complete `all_fielddescrs` list
/// (`descr.py init_size_descr`). A serialized member carries only its own
/// slot, so minting through it would publish a parent with an empty field
/// list, win `_cache_field` by first-write, and leave
/// `SizeDescr.all_fielddescrs` and `_cache_field` describing different
/// objects — breaking the positional invariant `heaptracker.py
/// get_fielddescr_index_in` establishes and `optimizeopt/info.rs force_box`
/// asserts.
///
/// Outcome of looking one serialized raw-set member back up.
enum SetMemberLookup {
    /// The gccache holds the slot the analyzer minted through.
    Resolved(majit_ir::DescrRef),
    /// The *container* (`STRUCT` / `ARRAY`) is absent from this process's
    /// descr universe entirely, so the member is dropped from the raw set.
    ///
    /// **Upstream cannot reach this.** `cpu.*descrof` and `compute_bitstrings`
    /// share one gccache in one process, so `setup_descrs` (`descr.py`)
    /// snapshots the very cache the raw-set descrs were minted into and
    /// `effectinfo.py` only ever unions descrs that are already there.
    /// A member that fails to resolve is therefore not a legitimate projection
    /// but a divergence, and its count is a badness counter held at zero
    /// (`descr_set_absent` on the `[jit-stats]` line) — pyre's form of the bare
    /// `assert`s upstream states this class of condition with.
    ///
    /// Two exact-key publishes hold it at zero inside the `Once` in
    /// `jitcode_runtime::rehydrate_build_descr_raw_sets`: `make_descr_from_bh`
    /// forces the matching module-scope group before each opcode layout, and
    /// `publish_effect_info_descr_mints` does the same for slots only a raw set
    /// names. `descrs.bin` — upstream's `opcode_descrs`, not its `all_descrs`
    /// — does not carry the latter.
    ///
    /// The arm stays because the answer is still reachable in principle, and a
    /// dropped member is a real hazard when it is: "absent" is evaluated once,
    /// when the `BhCallDescr` is materialised
    /// (`jitcode_runtime.rs rehydrated_call_descr_ref`), so a container
    /// registered afterwards would leave that EI permanently claiming "not
    /// written" for a field the callee does write, and the heapcache would not
    /// invalidate a read across the call. [`stale_absent_containers`] re-asks
    /// the question at exit and gates that residue at zero too.
    AbsentContainer,
    /// The container IS published but not under this member's key.  A real
    /// runtime descr for the field may exist under a different spelling, so
    /// dropping the member would answer "not written" for a field that is;
    /// the caller must fall back to the wildcard instead.
    Ambiguous,
}

/// Look a serialized raw-set member back up in the process-global gccache.
///
/// **Lookup only — never mint.** `descr.py get_field_descr` is
/// cache-or-mint because upstream calls it from `heaptracker.all_fielddescrs
/// (STRUCT)`, which always has the whole `STRUCT` in hand and therefore mints
/// the parent `SizeDescr` with its complete `all_fielddescrs` list
/// (`descr.py init_size_descr`). A serialized member carries only its own
/// slot, so minting through it would publish a parent with an empty field
/// list, win `_cache_field` by first-write, and leave
/// `SizeDescr.all_fielddescrs` and `_cache_field` describing different
/// objects — breaking the positional invariant `heaptracker.py
/// get_fielddescr_index_in` establishes and `optimizeopt/info.rs force_box`
/// asserts.
/// Ledger of every serialized raw-set member that came back
/// [`SetMemberLookup::Ambiguous`] — its container *is* published in this
/// process's descr universe, just not under the key the analyzer minted
/// through.
///
/// `Ambiguous` is the two-spellings defect by construction: both sides named
/// the same container and disagreed on how to spell it.  Its consumer degrades
/// the whole `EffectInfo` to `EF_RANDOM_EFFECTS`, so every entry below is a
/// residual call that stopped invalidating precisely and became a whole-heap
/// barrier.  `AbsentContainer` is the other non-answer and is equally a
/// divergence rather than a projection — see [`SetMemberLookup`] — but it is
/// the *missing table* defect, not the *two spellings* one, so the two are
/// counted apart.
///
/// This is the gate for struct-identity work; a converged-field-descr count is
/// not.  A field-descr census only counts the descrs a particular run happens
/// to mint, so deleting a call site greens it without joining anything,
/// whereas the members here are fixed by `descrs.bin` and stay `Ambiguous`
/// until the two spellings actually collapse onto one key.
/// Fill the gccache slots that only an `EffectInfo` raw set names, from the
/// mint arguments the analyzer recorded at its own `get_*_descr` call.
///
/// This is the far half of `descr.py`'s mint-or-hit: `get_field_descr`
/// (`:218-239`), `get_array_descr` (`:348-378`) and `get_interiorfield_descr`
/// (`:404-437`) all *construct* on a cache miss, and upstream reaches every one
/// of these slots because `compute_bitstrings` (`effectinfo.py`) unions
/// descr objects out of the same gccache `setup_descrs` (`descr.py`) then
/// snapshots.  Pyre carries only the assembler's opcode table across the
/// build/runtime split (`pyjitpl.py setup_descrs(asm.descrs)`), so a descr
/// minted purely to fill a raw set — named by no opcode — arrives with nothing
/// to resolve against.
///
/// Each entry is published under **the member's own key**, never a key
/// re-derived from a struct name: the analyzer prefers a source-attached
/// identity token over `path_hash(canonical_struct_name(..))` when it has one,
/// so re-deriving would land on a different slot than the one the member
/// names.
///
/// Runs **last**, after both the runtime groups and the `make_descr_from_bh`
/// loop (`jitcode_runtime.rs rehydrate_build_descr_raw_sets`), and writes only
/// into slots those left empty.  Going first would pre-fill a slot the loop
/// then asks for, with the analyzer's `index_in_parent` numbering rather than
/// the established producer's — the two disagree on any `#[pyre_class]`
/// struct, by the two injected header words.  Publishing the leftovers keeps
/// the established answers untouched.
///
/// The slots left empty are the ones no opcode names, which is why
/// `descrs.bin` — RPython's `opcode_descrs`, not its `all_descrs` — does not
/// carry them.  Upstream has that population too: `effectinfo.py`
/// builds every raw set through `cpu.fielddescrof` / `arraydescrof` /
/// `interiorfielddescrof`, a cache-or-mint into the gccache that runs whether
/// or not any operation names the field, and `descr.py setup_descrs`
/// then snapshots that gccache into `all_descrs`.  So a raw-set-only descr
/// referenced by no trace op is normal, not a sign of a mis-keyed publish.
pub(crate) fn publish_effect_info_descr_mints(entries: &[majit_ir::effectinfo::DescrMintEntry]) {
    use majit_ir::descr::LLType;
    use majit_ir::effectinfo::{DescrMintSpec, DescrSetMember};

    for entry in entries {
        let descr = match (&entry.member, &entry.spec) {
            (
                DescrSetMember::Field {
                    struct_id,
                    field_name,
                },
                spec,
            ) => mint_field(LLType::Struct(*struct_id), field_name, spec)
                .map(|descr| descr as majit_ir::DescrRef),
            (DescrSetMember::Array { array_id }, spec) => {
                mint_array(LLType::Array(*array_id), spec)
            }
            (
                DescrSetMember::InteriorField { array_id, name },
                DescrMintSpec::InteriorField {
                    array,
                    field_struct_id,
                    field_name,
                    field,
                },
            ) => {
                let array_key = LLType::Array(*array_id);
                let Some(array_descr) = mint_array(array_key.clone(), array)
                    .and_then(majit_ir::descr::descr_arc_as_array_descr)
                else {
                    continue;
                };
                // `descr.py fielddescr = get_field_descr(gc_ll_descr,
                // REALARRAY.OF, name)` — the element struct's own slot, then
                // `:436 InteriorFieldDescr(arraydescr, fielddescr)`.
                let Some(field_descr) =
                    mint_field(LLType::Struct(*field_struct_id), field_name, field)
                else {
                    continue;
                };
                Some(majit_ir::descr::gc_cache().lock().get_interiorfield_descr(
                    array_key,
                    name.clone(),
                    String::new(),
                    array_descr,
                    field_descr,
                ))
            }
            // A member paired with a spec of another shape cannot describe its
            // own slot; leaving it unpublished keeps it counted as absent
            // rather than filling the slot with the wrong layout.
            _ => None,
        };
        if let Some(descr) = descr
            && entry.ei_index != u32::MAX
        {
            descr.set_ei_index(entry.ei_index);
        }
    }
}

/// `descr.py:234-238` — bind the parent size descr, then mint the field.
fn mint_field(
    struct_key: majit_ir::descr::LLType,
    field_name: &str,
    spec: &majit_ir::effectinfo::DescrMintSpec,
) -> Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> {
    let majit_ir::effectinfo::DescrMintSpec::Field {
        struct_size,
        offset,
        field_size,
        field_type,
        flag,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
    } = spec
    else {
        return None;
    };
    if field_name == "gil_ready" {
        return Some(GIL_READY_DESCR_GROUP.field_descrs[0].clone()
            as std::sync::Arc<dyn majit_ir::descr::FieldDescr>);
    }
    // PyPy's single GcCache sees the runtime declaration at the same
    // `cpu.fielddescrof` request that creates this slot. At pyre's AOT
    // boundary, restore that order for this exact STRUCT only: its declaration
    // carries the vtable and immutable-field ranks that the analyzer mint
    // cannot reconstruct. This is deliberately per-key, matching
    // `descr.py:get_field_descr`, rather than forcing unrelated LazyLocks.
    if let majit_ir::descr::LLType::Struct(cache_key) = &struct_key {
        force_declared_group(*cache_key);
    }
    let mut gc = majit_ir::descr::gc_cache().lock();
    // Only an empty slot is this function's business: an unresolvable member is
    // by definition one nothing has filled, and a filled slot already belongs
    // to whoever published it. `get_field_descr` would return that owner's
    // descr anyway (`descr.py`), so taking the hit through it changes
    // nothing except to offer a second opinion on the layout — and the two
    // producers do not agree on one field of it. `index_in_parent` here is the
    // analyzer's `field_pos`, which counts the header words at offsets 0 and 8
    // that `heaptracker.py all_fielddescrs` skips, so it reads two higher
    // than the runtime publish's for the same field.
    if let Some(existing) = gc
        ._cache_field
        .get(&struct_key)
        .and_then(|inner| inner.get(field_name))
    {
        return Some(existing.clone() as std::sync::Arc<dyn majit_ir::descr::FieldDescr>);
    }
    // `descr.py fielddescr.parent_descr = get_size_descr(gccache, STRUCT,
    // vtable)`; the analyzer has no vtable surface, so 0 — a runtime publish
    // that carries one wins the slot either way.
    let _parent = gc.get_size_descr(struct_key.clone(), *struct_size, 0, false);
    Some(gc.get_field_descr(
        struct_key,
        field_name,
        None,
        *offset,
        *field_size,
        *field_type,
        *is_immutable,
        *is_quasi_immutable,
        *flag,
        u32::MAX,
        false,
        // The analyzer's `field_pos`, always computed — see the note above on
        // its two-word header offset against the runtime publish's numbering.
        Some(*index_in_parent),
    ))
}

/// `descr.py:353-370` — mint the array descr, including its lendescr.
fn mint_array(
    array_key: majit_ir::descr::LLType,
    spec: &majit_ir::effectinfo::DescrMintSpec,
) -> Option<majit_ir::DescrRef> {
    let majit_ir::effectinfo::DescrMintSpec::Array {
        base_size,
        item_size,
        flag,
        item_type,
        nolength,
        length_offset,
        is_pure,
        concrete_type,
    } = spec
    else {
        return None;
    };
    Some(majit_ir::descr::gc_cache().lock().get_array_descr(
        array_key,
        *base_size,
        *item_size,
        *flag,
        *item_type,
        *nolength,
        *length_offset,
        *is_pure,
        *concrete_type,
    ))
}

/// `ambiguous` empty alone is not enough to call the gate green: a member
/// whose container is not registered answers `AbsentContainer` instead, so a
/// run where `absent` dwarfs `resolved` has an empty `ambiguous` set
/// vacuously, because nothing was ever joined.  The ledger carries all three
/// outcomes so the denominator is visible next to the two failures.
#[derive(Default)]
pub struct SetMemberLedger {
    pub resolved: usize,
    /// Container keys no lookup could reach, by name.  Kept as a set rather
    /// than a count because a spelling miss hides here too: a container the
    /// runtime *did* publish, under a different key, is absent from both
    /// `_cache_field` and `_cache_size` at the analyzer's key and so reads
    /// exactly like a container this process never traced.
    pub absent: std::collections::BTreeSet<String>,
    pub ambiguous: std::collections::BTreeSet<String>,
    /// The dropped members themselves, so
    /// [`stale_absent_containers`] can ask the question again later.
    absent_members: Vec<majit_ir::effectinfo::DescrSetMember>,
}

static SET_MEMBER_LEDGER: LazyLock<Mutex<SetMemberLedger>> =
    LazyLock::new(|| Mutex::new(SetMemberLedger::default()));

fn set_member_label(m: &majit_ir::effectinfo::DescrSetMember) -> String {
    use majit_ir::effectinfo::DescrSetMember;
    match m {
        DescrSetMember::Field {
            struct_id,
            field_name,
        } => format!("Struct({struct_id:#018x}).{field_name}"),
        DescrSetMember::Array { array_id } => format!("Array({array_id:#018x})"),
        DescrSetMember::InteriorField { array_id, name } => {
            format!("Array({array_id:#018x}).{name}")
        }
    }
}

fn record_set_member_lookup(m: &majit_ir::effectinfo::DescrSetMember, out: &SetMemberLookup) {
    let mut ledger = SET_MEMBER_LEDGER.lock();
    match out {
        SetMemberLookup::Resolved(_) => ledger.resolved += 1,
        SetMemberLookup::AbsentContainer => {
            let label = set_member_label(m);
            if ledger.absent.insert(label) {
                ledger.absent_members.push(m.clone());
            }
        }
        SetMemberLookup::Ambiguous => {
            let label = set_member_label(m);
            ledger.ambiguous.insert(label);
        }
    }
}

/// Snapshot of the [`SET_MEMBER_LEDGER`].
pub fn set_member_ledger() -> SetMemberLedger {
    let ledger = SET_MEMBER_LEDGER.lock();
    SetMemberLedger {
        resolved: ledger.resolved,
        absent: ledger.absent.clone(),
        ambiguous: ledger.ambiguous.clone(),
        absent_members: Vec::new(),
    }
}

/// Re-ask the [`SetMemberLookup::AbsentContainer`] question against the
/// universe as it stands *now*, and return the members that have since become
/// reachable.
///
/// Every entry is a live instance of the gap documented on that variant: the
/// member was dropped from its raw set when the `BhCallDescr` was
/// materialised, the container was registered afterwards, and the frozen
/// `EffectInfo` still claims the callee does not touch it. Upstream cannot
/// reach this state — `effectinfo.py` builds its sets through
/// `cpu.fielddescrof` / `cpu.arraydescrof` in the same process that owns the
/// descr cache, so the container is always registered by construction.
///
/// An empty result is the evidence that the gap is not being hit; a non-empty
/// one names exactly which containers to publish eagerly next.
pub fn stale_absent_containers() -> Vec<String> {
    let members = SET_MEMBER_LEDGER.lock().absent_members.clone();
    members
        .iter()
        .filter(|m| !matches!(descr_from_set_member(m), SetMemberLookup::AbsentContainer))
        .map(set_member_label)
        .collect()
}

/// One struct's field map holds both spellings a field descriptor can carry:
/// the bare `field` and the `STRUCT.field` form `PyreFieldDescr` stores, which
/// `get_field_descr` composes, and a single map mixes them — `W_TupleObject`'s holds
/// `wrappeditems` and `hash` bare beside `PyObject.w_class`. Which one a field arrives under depends on the
/// producer that reached it first, so an analyzer member naming the field
/// bare cannot be resolved by exact match alone. `all_fielddescrs`'s
/// field-walk already joins the two forms this way.
///
/// A suffix hit still has to be unique. Two qualified keys can end in the
/// same field — an inherited name redeclared by a subclass — and that IS the
/// identity split the caller must not paper over, so it stays unresolved.
fn lookup_field_by_either_spelling<'m>(
    fields: &'m indexmap::IndexMap<String, std::sync::Arc<majit_ir::descr::SimpleFieldDescr>>,
    field_name: &str,
) -> Option<&'m std::sync::Arc<majit_ir::descr::SimpleFieldDescr>> {
    if let Some(fd) = fields.get(field_name) {
        return Some(fd);
    }
    let needle = format!(".{field_name}");
    // `register_keyed_field` is first-write. A later producer can store the
    // same bare name under another qualified key at a different offset; the
    // first descr whose own key is the bare name is the cached one.
    let mut exact = None;
    let mut suffix = None;
    let mut suffix_hits = 0usize;
    for (stored, fd) in fields {
        if fd.field_key() == field_name {
            exact.get_or_insert(fd);
        } else if stored.ends_with(&needle) {
            suffix_hits += 1;
            suffix.get_or_insert(fd);
        }
    }
    if exact.is_none() && suffix_hits > 1 {
        return None;
    }
    exact.or(suffix)
}

/// Two qualified keys ending in the same bare name. That is the identity
/// split [`lookup_field_by_either_spelling`] refuses to paper over.
fn field_map_spelling_splits(
    fields: &indexmap::IndexMap<String, std::sync::Arc<majit_ir::descr::SimpleFieldDescr>>,
    field_name: &str,
) -> bool {
    // A bare-key hit is resolvable above. This stays a split only when every
    // matching entry is a qualified spelling and those offsets disagree.
    if fields.values().any(|fd| fd.field_key() == field_name) {
        return false;
    }
    let needle = format!(".{field_name}");
    let mut offset = None;
    for (stored, fd) in fields {
        if stored == field_name || stored.ends_with(&needle) {
            match offset {
                None => offset = Some(fd.offset()),
                Some(prev) if prev != fd.offset() => return true,
                Some(_) => {}
            }
        }
    }
    false
}

/// `descr.py` `get_field_descr` returns the FieldDescr already stored on the
/// STRUCT. A borrowed layout publishes that object on the size and fills
/// `_cache_field` only on a later name lookup, so an EffectInfo member can
/// name `w_class` before that lookup. Adopt the existing arc. A second
/// FieldDescr would split the identity `heaptracker.py get_fielddescr_index_in`
/// indexes.
fn adopt_field_from_published_size(
    gc: &majit_ir::descr::GcCache,
    struct_key: &majit_ir::descr::LLType,
    field_name: &str,
) -> SetMemberLookup {
    let Some(size) = gc
        ._cache_size
        .get(struct_key)
        .and_then(|descr| descr.as_size_descr())
    else {
        return SetMemberLookup::AbsentContainer;
    };
    let needle = format!(".{field_name}");
    let mut exact: Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> = None;
    let mut suffix: Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> = None;
    let mut suffix_hits = 0usize;
    // `SizeDescr::class_word_field` reads `gc_fielddescrs` first. The
    // inherited `w_class` edge is published there and kept out of
    // `all_fielddescrs` (`with_extra_gc_fielddescr`).
    for field in size.gc_fielddescrs().iter().chain(size.all_fielddescrs()) {
        let key = field.field_key();
        let name = field.field_name();
        if key == field_name || name == field_name {
            exact.get_or_insert_with(|| field.clone());
        } else if key.ends_with(&needle) || name.ends_with(&needle) {
            suffix_hits += 1;
            suffix.get_or_insert_with(|| field.clone());
        }
    }
    let hit = if exact.is_none() && suffix_hits > 1 {
        None
    } else {
        exact.or(suffix)
    };
    match hit {
        Some(descr) => SetMemberLookup::Resolved(descr),
        None => SetMemberLookup::Ambiguous,
    }
}

/// The three spellings `make_descr_from_bh` and the descr pool use for
/// the shared `PyObject` header. `path_hash` is the `LLType::Struct` key.
fn is_shared_pyobject_header(struct_id: u64) -> bool {
    struct_id == majit_ir::descr::path_hash("pyobject::PyObject")
        || struct_id == majit_ir::descr::path_hash("PyObject")
        || struct_id == majit_ir::descr::path_hash("pyre_object::pyobject::PyObject")
}

/// Whether `struct_id` names `GilReadyState` under one of the owner spellings
/// `make_descr_from_bh` bridges.
fn is_gil_ready_struct(struct_id: u64) -> bool {
    [
        "GilReadyState",
        "gil_ready::GilReadyState",
        "pyre_object::gil_ready::GilReadyState",
    ]
    .into_iter()
    .any(|owner| struct_id == majit_ir::descr::path_hash(owner))
}

fn descr_from_set_member(m: &majit_ir::effectinfo::DescrSetMember) -> SetMemberLookup {
    use majit_ir::descr::{LLType, gc_cache};

    match m {
        majit_ir::effectinfo::DescrSetMember::Field {
            struct_id,
            field_name,
            ..
        } => {
            // `GilReadyState.gil_ready` lives in a non-GC static, so no
            // `gc_cache` slot names its struct.  A residual call that reads
            // the word publishes it in its read set; resolve it to the one
            // quasi descr, as `make_descr_from_bh` does for the opcode.
            if field_name.as_str() == "gil_ready" && is_gil_ready_struct(*struct_id) {
                return SetMemberLookup::Resolved(gil_ready_descr());
            }
            let struct_key = LLType::Struct(*struct_id);
            let gc = gc_cache().lock();
            if let Some(inner) = gc._cache_field.get(&struct_key)
                && let Some(fd) = lookup_field_by_either_spelling(inner, field_name)
            {
                return SetMemberLookup::Resolved(fd.clone() as majit_ir::DescrRef);
            }
            // The size's own field list, including a gc-only `w_class`,
            // is the object `descr.py` `get_field_descr` would return.
            // A split in the name map is only a defect when that list
            // does not already hold the member.
            let adopted = adopt_field_from_published_size(&gc, &struct_key, field_name);
            if matches!(adopted, SetMemberLookup::Resolved(_)) {
                return adopted;
            }
            // `make_descr_from_bh` returns `w_class_descr()` for the shared
            // `PyObject` header. The mint records that owner as
            // `pyobject::PyObject`, which the name bridge does not list, so
            // the size can be present while neither field list carries the
            // name. The stamp has to use the same descr the walker reads.
            if field_name == "w_class" && is_shared_pyobject_header(*struct_id) {
                return SetMemberLookup::Resolved(w_class_descr());
            }
            if gc
                ._cache_field
                .get(&struct_key)
                .is_some_and(|inner| field_map_spelling_splits(inner, field_name))
            {
                return SetMemberLookup::Ambiguous;
            }
            adopted
        }
        majit_ir::effectinfo::DescrSetMember::Array { array_id, .. } => {
            match gc_cache()
                .lock()
                ._cache_array
                .get(&LLType::Array(*array_id))
            {
                Some(ad) => SetMemberLookup::Resolved(ad.clone()),
                None => SetMemberLookup::AbsentContainer,
            }
        }
        majit_ir::effectinfo::DescrSetMember::InteriorField { array_id, name, .. } => {
            let gc = gc_cache().lock();
            let array_key = LLType::Array(*array_id);
            match gc
                ._cache_interiorfield
                .get(&(array_key.clone(), name.clone(), String::new()))
            {
                Some(d) => SetMemberLookup::Resolved(d.clone()),
                None if !gc._cache_array.contains_key(&array_key) => {
                    SetMemberLookup::AbsentContainer
                }
                None => SetMemberLookup::Ambiguous,
            }
        }
    }
}

/// Replay `EffectInfo.compute_bitstrings`'s `descr.ei_index = ...` mutation
/// for a member whose descriptor was already published through
/// `Assembler.descrs`. The caller deliberately carries no mint spec: reaching
/// a cache miss here means the build-time opcode/member join was wrong, and
/// minting a replacement would hide an identity split PyPy cannot represent.
/// A size is published, but this field is not on it. The caller can
/// still install the opcode `BhDescr::Field` that names the same struct
/// (`GcEntries.items` sits past the fixed `length` word).
pub(crate) fn ambiguous_field_struct(
    member: &majit_ir::effectinfo::DescrSetMember,
) -> Option<(u64, &str)> {
    match descr_from_set_member(member) {
        SetMemberLookup::Ambiguous => match member {
            majit_ir::effectinfo::DescrSetMember::Field {
                struct_id,
                field_name,
            } => Some((*struct_id, field_name.as_str())),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn stamp_effect_info_descr(
    member: &majit_ir::effectinfo::DescrSetMember,
    ei_index: u32,
) {
    match descr_from_set_member(member) {
        SetMemberLookup::Resolved(descr) => descr.set_ei_index(ei_index),
        SetMemberLookup::AbsentContainer => {
            panic!("opcode-published EffectInfo member has no runtime container: {member:?}")
        }
        SetMemberLookup::Ambiguous => {
            panic!("opcode-published EffectInfo member is ambiguous at runtime: {member:?}")
        }
    }
}

/// Fill the six `_*_descrs_*` raw sets from `descr_set_keys`, canonicalising
/// exactly as `effectinfo.py frozenset_or_none` /
/// `canonicalize_descr_set`.
///
/// All six sets or none of them: `effectinfo.py` makes them `None`
/// **iff** the EI is `EF_RANDOM_EFFECTS`, and `compute_bitstrings`
/// (`effectinfo.py`) asserts that biconditional before deciding
/// whether to clear the bitstrings.  A half-populated EI would clear them
/// while `extraeffect` still claims concrete effects, and the next
/// `check_readonly_descr_field` would then read a `None` bitstring.
/// `effectinfo.py` wildcard: the shape to fall back to whenever the
/// concrete sets cannot be rebuilt faithfully.  Conservative in the sound
/// direction — `has_random_effects()` makes every heap consumer assume the
/// call touched everything.
///
/// Both raw-set families have to be cleared, not just the compacted one.
/// `EffectInfo::default()` seeds every `_*_descrs_*` with `Some(vec![])` and
/// serde skips them, so a deserialized EI arrives carrying empty concrete sets;
/// leaving those behind states "writes nothing" — the bottom of the lattice —
/// next to an `extraeffect` saying "writes everything".  Readers that consult
/// the raw set directly (`writes_field_descr_by_identity`, which
/// `OptHeap::force_from_effectinfo` uses before `compute_bitstrings` has run)
/// would take the unsound half, and `compute_bitstrings` asserts the
/// biconditional outright (`effectinfo.py`).
fn degrade_to_random_effects(ei: &mut majit_ir::EffectInfo) {
    ei.extraeffect = majit_ir::ExtraEffect::RandomEffects;
    // effectinfo.py:364-365 — the wildcard forces can_collect.
    ei.can_collect = true;
    // `call.py:284-286` states it outright: "random_effects implies
    // can_invalidate".  `effectinfo.py MOST_GENERAL` is built with
    // `can_invalidate=True`, and so is pyre's own `EffectInfo::MOST_GENERAL`.
    // Without this a degraded EI is a shape upstream cannot construct —
    // random effects with `check_can_invalidate()` still false — and
    // `check_can_invalidate` is read on a path `has_random_effects()` does not
    // cover (`heap.py:457-459`; `_seen_guard_not_invalidated` is otherwise only
    // reset in `__init__`, `heap.py:341`), so a quasi-immutable guard would
    // survive a call the wildcard says invalidates everything.
    ei.can_invalidate = true;
    ei._readonly_descrs_fields = None;
    ei._write_descrs_fields = None;
    ei._readonly_descrs_arrays = None;
    ei._write_descrs_arrays = None;
    ei._readonly_descrs_interiorfields = None;
    ei._write_descrs_interiorfields = None;
    ei.readonly_descrs_fields = None;
    ei.write_descrs_fields = None;
    ei.readonly_descrs_arrays = None;
    ei.write_descrs_arrays = None;
    ei.readonly_descrs_interiorfields = None;
    ei.write_descrs_interiorfields = None;
    ei.single_write_descr_array = None;
}

pub fn rehydrate_effect_info(ei: &mut majit_ir::EffectInfo) {
    use degrade_to_random_effects as degrade;

    // `descr_set_keys` is the build-process -> runtime-process wire channel
    // for the six raw sets skipped by serde.  Consume it here: after the
    // canonical DescrRefs below have been installed, retaining the member
    // names duplicates the live object graph and has no RPython counterpart.
    let Some(keys) = ei.descr_set_keys.take() else {
        // No serialized key channel.  For a build-time EI that means the
        // codewriter already emitted the `EF_RANDOM_EFFECTS` wildcard, so
        // the six raw sets are `None` and there is nothing to rebuild.  Any
        // other shape lost its sets somewhere the invariant above does not
        // cover; restore the wildcard rather than leave a concrete
        // `extraeffect` pointing at absent sets.
        if ei.extraeffect != majit_ir::ExtraEffect::RandomEffects {
            degrade(ei);
        }
        return;
    };
    let resolve = |members: &[majit_ir::effectinfo::DescrSetMember]| {
        let mut out = Vec::with_capacity(members.len());
        // Walk the WHOLE set even once it is known to be incomplete: the
        // ledger behind `record_set_member_lookup` is what
        // `descr_set_absent` / `descr_set_ambiguous` report, and those are
        // gated in `check.py`.  Returning at the first non-answer would stop
        // counting the rest of the set, so the gate's own numbers would shrink
        // with the defect they exist to measure — by an order-dependent
        // amount, which is worse than useless.
        let mut complete = true;
        for m in members {
            let looked_up = descr_from_set_member(m);
            record_set_member_lookup(m, &looked_up);
            match looked_up {
                SetMemberLookup::Resolved(d) => out.push(d),
                // `effectinfo.py compute_bitstrings` unions EVERY
                // member of every raw set; a member there is a live descr
                // object, so upstream has no third answer.  Both of pyre's
                // non-answers therefore mean the same thing — this set
                // cannot be completed — and the only sound reply is the
                // `EF_RANDOM_EFFECTS` wildcard.  Keeping a *concrete* set
                // with the member omitted is what breaks the contract
                // `optimizeopt/heap.rs force_from_effectinfo` relies on:
                // out-of-range `bitcheck` returns false (`bitstring.py`
                // parity), so the omission reads as "the callee does not
                // write this field" and the optimizer keeps a stale heap
                // cache entry across a call that does write it.
                //
                // Absence is not even stable — the descr universe keeps
                // growing after the raw sets freeze, so a container absent
                // at rehydrate time can arrive later while the omission
                // lives on.  Closing the serialized universe (the
                // build-time `DescrMintSpec` channel) is what makes this
                // arm unreachable rather than merely rare.
                SetMemberLookup::AbsentContainer | SetMemberLookup::Ambiguous => complete = false,
            }
        }
        complete.then(|| majit_ir::effectinfo::canonicalize_descr_set(out))
    };
    let resolved = (
        resolve(&keys.readonly_fields),
        resolve(&keys.write_fields),
        resolve(&keys.readonly_arrays),
        resolve(&keys.write_arrays),
        resolve(&keys.readonly_interiorfields),
        resolve(&keys.write_interiorfields),
    );
    let (
        Some(readonly_fields),
        Some(write_fields),
        Some(readonly_arrays),
        Some(write_arrays),
        Some(readonly_interiorfields),
        Some(write_interiorfields),
    ) = resolved
    else {
        degrade(ei);
        return;
    };
    ei._readonly_descrs_fields = Some(readonly_fields);
    ei._write_descrs_fields = Some(write_fields);
    ei._readonly_descrs_arrays = Some(readonly_arrays);
    // effectinfo.py:201-206 single_write_descr_array — also `serde(skip)`,
    // and read in production by `heap.rs force_from_effectinfo`, so it is
    // re-derived from the set that just came back rather than left `None`.
    ei.single_write_descr_array = match write_arrays.as_slice() {
        [only] => Some(only.clone()),
        _ => None,
    };
    ei._write_descrs_arrays = Some(write_arrays);
    ei._readonly_descrs_interiorfields = Some(readonly_interiorfields);
    ei._write_descrs_interiorfields = Some(write_interiorfields);
}

/// Validate a translated EffectInfo's structural raw-set image without
/// retaining the rehydrated `Vec<DescrRef>` graph.
///
/// RPython discards the six raw frozensets after translation has run
/// `compute_bitstrings`; the translated object retains the compact bitstrings,
/// and each descriptor retains `ei_index`. Pyre's earlier runtime rehydration
/// rebuilt those translation-only frozensets solely because partitioning had
/// been delayed across the process boundary. Once the build artifact carries
/// both outputs, resolving every member here is an invariant check, not
/// runtime state construction. `single_write_descr_array` is the one raw
/// descriptor reference EffectInfo deliberately keeps after compaction
/// (`effectinfo.py`), so that singleton alone remains live.
pub fn prepare_frozen_effect_info(ei: &mut majit_ir::EffectInfo) {
    // Shared with `rehydrate_effect_info` rather than spelled out again: the
    // two had drifted, and this one was clearing only the compacted sets while
    // the six `_*_descrs_*` kept the `Some(vec![])` that `EffectInfo::default()`
    // seeds and serde skips.
    use degrade_to_random_effects as degrade;

    let Some(keys) = ei.descr_set_keys.take() else {
        if ei.extraeffect != majit_ir::ExtraEffect::RandomEffects {
            degrade(ei);
        }
        return;
    };
    let resolve = |members: &[majit_ir::effectinfo::DescrSetMember]| {
        let mut out = Vec::with_capacity(members.len());
        let mut complete = true;
        for member in members {
            let looked_up = descr_from_set_member(member);
            record_set_member_lookup(member, &looked_up);
            match looked_up {
                SetMemberLookup::Resolved(descr) => out.push(descr),
                SetMemberLookup::AbsentContainer | SetMemberLookup::Ambiguous => complete = false,
            }
        }
        complete.then_some(out)
    };
    let resolved = (
        resolve(&keys.readonly_fields),
        resolve(&keys.write_fields),
        resolve(&keys.readonly_arrays),
        resolve(&keys.write_arrays),
        resolve(&keys.readonly_interiorfields),
        resolve(&keys.write_interiorfields),
    );
    let (
        Some(_readonly_fields),
        Some(_write_fields),
        Some(_readonly_arrays),
        Some(write_arrays),
        Some(_readonly_interiorfields),
        Some(_write_interiorfields),
    ) = resolved
    else {
        degrade(ei);
        return;
    };
    ei.single_write_descr_array = match write_arrays.as_slice() {
        [only] => Some(only.clone()),
        _ => None,
    };
}

/// `BhCallDescr` -> `CallDescr` adapter. RPython parity: codewriter
/// `Assembler.descrs` carries the same `CallDescr` instance the
/// metainterp pulls during op recording. pyre keeps the codewriter-side
/// call descr as serializable fields and rebuilds a `MetaCallDescr` on
/// demand here, preserving the per-call-site `EffectInfo`.
///
/// `arg_classes` is RPython `CallDescr.arg_classes`: one char per non-void
/// function argument. Uppercase `I/R/F` are assembler list markers and must not
/// appear here.
pub fn make_call_descr_from_bh(bh: &majit_jitcode::jitcode::BhCallDescr) -> DescrRef {
    let arg_types: Vec<Type> = bh
        .arg_classes
        .chars()
        .filter_map(|c| match c {
            'i' | 'S' => Some(Type::Int),
            'r' => Some(Type::Ref),
            'f' | 'L' => Some(Type::Float),
            _ => None,
        })
        .collect();
    let result_type = match bh.result_type {
        'i' | 'S' => Type::Int,
        'r' => Type::Ref,
        'f' | 'L' => Type::Float,
        _ => Type::Void,
    };
    if bh.void_word_abi {
        assert_eq!(
            result_type,
            Type::Void,
            "void-word ABI requires a void BhCallDescr result type",
        );
    }
    let result_size = if bh.void_word_abi { 8 } else { bh.result_size };
    if let Some(translated_id) = bh.translated_effect_info_id {
        majit_metainterp::make_call_descr_sized_with_translated_effect(
            &bh.arg_classes,
            &arg_types,
            result_type,
            // `descr.py get_result_type()` — the raw char, same as the
            // non-translated arm below.  Dropping it here would collapse `'S'`
            // onto `'i'` and `'L'` onto `'f'` for every descr that carries a
            // translated EffectInfo id.
            bh.result_type,
            bh.result_signed,
            result_size,
            translated_id,
        )
    } else {
        majit_ir::descr::make_call_descr_full_with_classes(
            u32::MAX,
            bh.arg_classes.clone(),
            arg_types,
            result_type,
            bh.result_type,
            bh.result_signed,
            result_size,
            bh.extra_info.clone(),
        )
    }
}

/// descr.py InteriorFieldDescr for SETINTERIORFIELD_GC.
/// assert arraydescr.flag == FLAG_STRUCT.
/// llmodel.py bh_setinteriorfield_gc_i: bh_setinteriorfield_gc_{i,r,f} computes
/// offset = arraydescr.basesize + itemindex * itemsize + fielddescr.offset.
pub fn make_interior_field_descr(
    array_descr_index: u32,
    base_size: usize,
    item_size: usize,
    field_offset: usize,
    field_size: usize,
    field_type: u8, // 0=ref, 1=int, 2=float
    field_descr_index: u32,
) -> DescrRef {
    use majit_ir::descr::{
        ArrayFlag, SimpleArrayDescr, SimpleFieldDescr, SimpleInteriorFieldDescr,
    };
    let tp = match field_type {
        0 => Type::Ref,
        2 => Type::Float,
        _ => Type::Int,
    };
    // descr.py: assert arraydescr.flag == FLAG_STRUCT
    let array_descr = Arc::new(SimpleArrayDescr::with_flag(
        array_descr_index,
        base_size,
        item_size,
        0,
        Type::Void,
        ArrayFlag::Struct,
    ));
    majit_ir::descr_registry::register_array(array_descr.clone() as DescrRef);
    let field_descr = Arc::new(SimpleFieldDescr::new(
        field_descr_index,
        field_offset,
        field_size,
        tp,
        true, // immutable (struct fields in array-of-struct)
    ));
    majit_ir::descr_registry::register_field(field_descr.clone() as DescrRef);
    let interior: DescrRef = Arc::new(SimpleInteriorFieldDescr::new(
        field_descr_index,
        array_descr,
        field_descr,
    ));
    majit_ir::descr_registry::register_interior_field(interior.clone());
    interior
}

#[cfg(test)]
mod degrade_shape_tests {
    use super::*;

    /// `effectinfo.py:149-162` makes the six raw sets `None` **iff** the EI is
    /// `EF_RANDOM_EFFECTS`, and `compute_bitstrings` (`effectinfo.py`)
    /// asserts that biconditional.  Both degrade paths therefore have to clear
    /// the `_*_descrs_*` family as well as the compacted one — `serde(skip)`
    /// plus `EffectInfo::default()` means a deserialized EI arrives with
    /// `Some(vec![])` in all six, and an EI left holding those while
    /// `extraeffect` says random effects reads as "writes nothing" through
    /// `writes_field_descr_by_identity`, the unsound direction.
    #[test]
    fn both_degrade_paths_clear_the_raw_sets_too() {
        for (label, prepare) in [
            (
                "prepare_frozen_effect_info",
                prepare_frozen_effect_info as fn(&mut majit_ir::EffectInfo),
            ),
            (
                "rehydrate_effect_info",
                rehydrate_effect_info as fn(&mut majit_ir::EffectInfo),
            ),
        ] {
            // A concrete classification with no key channel — the shape whose
            // sets cannot be rebuilt, so it must fall back to the wildcard.
            let mut ei = majit_ir::EffectInfo {
                extraeffect: majit_ir::ExtraEffect::CanRaise,
                descr_set_keys: None,
                ..majit_ir::EffectInfo::default()
            };
            // The premise: `default()` really does seed the raw sets, so
            // clearing them is not vacuous.
            assert!(ei._write_descrs_fields.is_some(), "{label}: premise");

            prepare(&mut ei);

            assert_eq!(
                ei.extraeffect,
                majit_ir::ExtraEffect::RandomEffects,
                "{label}"
            );
            assert!(ei._readonly_descrs_fields.is_none(), "{label}");
            assert!(ei._write_descrs_fields.is_none(), "{label}");
            assert!(ei._readonly_descrs_arrays.is_none(), "{label}");
            assert!(ei._write_descrs_arrays.is_none(), "{label}");
            assert!(ei._readonly_descrs_interiorfields.is_none(), "{label}");
            assert!(ei._write_descrs_interiorfields.is_none(), "{label}");
            assert!(ei.write_descrs_fields.is_none(), "{label}");
            assert!(ei.single_write_descr_array.is_none(), "{label}");
            assert!(ei.can_collect, "{label}");
            assert!(ei.can_invalidate, "{label}");
        }
    }
}

#[cfg(test)]
mod set_member_lookup_tests {
    use super::*;
    use majit_ir::effectinfo::DescrSetMember;

    /// Distinguishing the two non-resolving outcomes is the whole gate:
    /// `AbsentContainer` is the projection onto a universe that never named the
    /// container, `Ambiguous` is the container being named under a different
    /// key. Only the second is a defect, and it is the one `check.py` gates at
    /// zero.
    ///
    /// Both assertions turn on one key each, so they hold whatever else the
    /// process-global gccache already carries.
    #[test]
    fn an_unnamed_container_is_absent_and_a_named_one_missing_the_field_is_ambiguous() {
        let published = 0x7e57_0000_0000_0001u64;
        let never_named = 0x7e57_0000_0000_0002u64;

        majit_ir::descr::gc_cache().lock().register_keyed_size(
            majit_ir::descr::LLType::Struct(published),
            Arc::new(majit_ir::descr::SimpleSizeDescr::with_vtable(
                u32::MAX,
                32,
                0,
                0,
            )) as DescrRef,
        );

        let member = |struct_id| DescrSetMember::Field {
            struct_id,
            field_name: "no_such_field".to_string(),
        };
        assert!(matches!(
            descr_from_set_member(&member(never_named)),
            SetMemberLookup::AbsentContainer
        ));
        assert!(matches!(
            descr_from_set_member(&member(published)),
            SetMemberLookup::Ambiguous
        ));
    }

    /// The EffectInfo member for the shared header is `pyobject::PyObject.w_class`.
    /// That spelling is not in the `make_descr_from_bh` name bridge, so the
    /// stamp answers with `w_class_descr` itself. `with_extra_gc_fielddescr`
    /// keeps the inherited header out of `all_fielddescrs`; EffectInfo still
    /// names that field `w_class`. A borrowed parent layout stores `w_class`
    /// on the size and `stamp_effect_info_descr` adopts that arc.
    #[test]
    fn w_class_is_the_shared_effectinfo_field() {
        {
            // pyobject_header_w_class_is_the_shared_descr
            let struct_id = majit_ir::descr::path_hash("pyobject::PyObject");
            let found = descr_from_set_member(&DescrSetMember::Field {
                struct_id,
                field_name: "w_class".to_string(),
            });
            let SetMemberLookup::Resolved(descr) = found else {
                panic!("pyobject::PyObject.w_class did not resolve");
            };
            // Another test in this process may have published the header's
            // own layout into the process-global `gc_cache`; the lookup then
            // answers with that size's field. Both name the same word.
            let published = majit_ir::descr::gc_cache()
                .lock()
                ._cache_size
                .contains_key(&majit_ir::descr::LLType::Struct(struct_id));
            if !published {
                assert!(std::sync::Arc::ptr_eq(&descr, &w_class_descr()));
            }
            let canonical = w_class_descr();
            assert_eq!(
                descr.as_field_descr().map(|f| f.offset()),
                canonical.as_field_descr().map(|f| f.offset())
            );
        }
        {
            // a_gc_only_w_class_on_the_size_is_the_effectinfo_field
            let struct_id = 0x7e57_0000_0000_0004u64;
            let field: std::sync::Arc<dyn majit_ir::descr::FieldDescr> =
                std::sync::Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
                    0,
                    8,
                    8,
                    majit_ir::Type::Ref,
                    false,
                    majit_ir::ArrayFlag::Pointer,
                    "PyObject.w_class".to_string(),
                    "w_class",
                ));
            let size = majit_ir::descr::SimpleSizeDescr::with_vtable(u32::MAX, 32, 0, 0x1000)
                .with_extra_gc_fielddescr(field.clone());
            majit_ir::descr::gc_cache().lock().register_keyed_size(
                majit_ir::descr::LLType::Struct(struct_id),
                std::sync::Arc::new(size) as DescrRef,
            );
            let found = descr_from_set_member(&DescrSetMember::Field {
                struct_id,
                field_name: "w_class".to_string(),
            });
            let SetMemberLookup::Resolved(descr) = found else {
                panic!("gc-only w_class was not adopted");
            };
            assert!(std::sync::Arc::ptr_eq(
                &descr,
                &(field as majit_ir::DescrRef)
            ));
        }
        {
            // stamp_adopts_w_class_already_stored_on_the_size
            use majit_ir::descr::{ArrayFlag, LLType, SimpleFieldDescr, SimpleSizeDescr};
            use majit_ir::value::Type;

            // Distinct from `a_gc_only_w_class_on_the_size_is_the_effectinfo_field`:
            // both tests publish into the process-global `gc_cache`.
            let struct_id = 0x7e57_0000_0000_0005u64;
            let key = "Owner.w_class";
            let field = Arc::new(SimpleFieldDescr::new_with_name(
                0,
                8,
                8,
                Type::Ref,
                false,
                ArrayFlag::Struct,
                key.to_string(),
                key,
            ));
            let size = SimpleSizeDescr::with_vtable(u32::MAX, 16, 0, 0).with_all_fielddescrs(vec![
                field.clone() as std::sync::Arc<dyn majit_ir::descr::FieldDescr>,
            ]);
            // Do not insert a `_cache_field` row. That is the borrowed publish.
            majit_ir::descr::gc_cache().lock().register_keyed_size(
                LLType::Struct(struct_id),
                Arc::new(size) as majit_ir::DescrRef,
            );

            let member = DescrSetMember::Field {
                struct_id,
                field_name: "w_class".to_string(),
            };
            let SetMemberLookup::Resolved(descr) = descr_from_set_member(&member) else {
                panic!("w_class on the size must resolve without a name-cache row");
            };
            let resolved = descr
                .as_field_descr()
                .expect("a Field member resolves to a FieldDescr");
            assert_eq!(resolved.offset(), 8);
            assert_eq!(resolved.field_name(), key);

            stamp_effect_info_descr(&member, 7);
            // The atomic lives on the size's FieldDescr. A minted replacement
            // would leave this one at u32::MAX.
            assert_eq!(field.get_ei_index(), 7);
        }
    }

    /// The label is what `PYRE_DESCR_SPELLING_GATE` prints, and it has to name
    /// the container in the same hex form `descrs.bin` keys on so an entry can
    /// be grepped straight back to a producer.
    #[test]
    fn labels_carry_the_container_key_and_member_name() {
        assert_eq!(
            set_member_label(&DescrSetMember::Field {
                struct_id: 0xa3af111df5325ac5,
                field_name: "items".to_string(),
            }),
            "Struct(0xa3af111df5325ac5).items"
        );
        assert_eq!(
            set_member_label(&DescrSetMember::Array {
                array_id: 0x0000000000000010
            }),
            "Array(0x0000000000000010)"
        );
    }

    /// The point of carrying the mint arguments across the build/runtime split:
    /// a container this process would never otherwise name resolves anyway,
    /// because the publish takes the same `descr.py` miss branch the
    /// analyzer took. Without it the member reads `AbsentContainer` — which is
    /// the pre-state this asserts first, on the same key.
    #[test]
    fn publishing_the_recorded_mint_turns_an_absent_member_into_a_resolved_one() {
        use majit_ir::effectinfo::{DescrMintEntry, DescrMintSpec};

        let struct_id = 0x7e57_0000_0000_0003u64;
        let member = DescrSetMember::Field {
            struct_id,
            field_name: "carried".to_string(),
        };
        assert!(
            matches!(
                descr_from_set_member(&member),
                SetMemberLookup::AbsentContainer
            ),
            "nothing has named this container yet"
        );

        publish_effect_info_descr_mints(&[DescrMintEntry {
            member: member.clone(),
            spec: DescrMintSpec::Field {
                struct_size: 24,
                offset: 16,
                field_size: 8,
                field_type: majit_ir::value::Type::Int,
                flag: majit_ir::descr::ArrayFlag::Signed,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
            },
            ei_index: u32::MAX,
        }]);

        let SetMemberLookup::Resolved(descr) = descr_from_set_member(&member) else {
            panic!("the published slot must resolve");
        };
        let field = descr
            .as_field_descr()
            .expect("a Field member resolves to a FieldDescr");
        assert_eq!(field.offset(), 16);
        assert_eq!(field.field_size(), 8);
        assert_eq!(field.index_in_parent(), 1);
    }
}

/// The one `EffectInfo` every `jit_ll_strconcat` calldescr is minted with:
/// `rstr.py ll_strconcat` is `@jit.elidable` under `oopspec
/// 'stroruni.concat(s1, s2)'`.  The `callinfocollection` row
/// `resume.py concat_strings` reads and the calls the walker records have to
/// name the same descr, so both spell the effect through here.
pub fn ll_strconcat_effectinfo() -> majit_ir::EffectInfo {
    majit_ir::EffectInfo::const_new(
        majit_ir::ExtraEffect::ElidableOrMemoryError,
        majit_ir::OopSpecIndex::StrConcat,
    )
}

/// `rstr.py LLHelpers.ll_str_mul` is `@jit.elidable`; overflow is MemoryError.
pub fn ll_str_mul_effectinfo() -> majit_ir::EffectInfo {
    majit_ir::EffectInfo::const_new(
        majit_ir::ExtraEffect::ElidableOrMemoryError,
        majit_ir::OopSpecIndex::None,
    )
}
