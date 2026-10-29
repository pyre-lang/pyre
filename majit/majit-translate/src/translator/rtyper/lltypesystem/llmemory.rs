//! `rpython/rtyper/lltypesystem/llmemory.py` — annotation types for
//! low-level memory addresses.
#![allow(non_camel_case_types, non_snake_case)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{LazyLock, OnceLock};

use crate::annotator::model::{KnownType, SomeObject, SomeObjectTrait, SomeValue};
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _address, _arraylenref, _endmarker, _ptr, _ptr_obj, _wref, Array, ArrayContainer, GCREF,
    GcKind, LowLevelType, NONGCREF, ParentIndex, Ptr, PtrTarget, WEAKREF_PTR, cast_int_to_ptr,
    cast_opaque_ptr, cast_pointer, container_value_as_ptr, direct_arrayitems, direct_fieldptr,
    direct_ptradd, nullptr, parentlink,
};

/// RPython `class fakeaddress(object)` (llmemory.py).
pub type fakeaddress = _address;

/// RPython `class fakeaddressEntry(ExtRegistryEntry)` (llmemory.py).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct fakeaddressEntry;

/// RPython `class NullAddressError(Exception)` (llmemory.py).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NullAddressError;

/// RPython `class DanglingPointerError(Exception)` (llmemory.py).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DanglingPointerError;

thread_local! {
    /// `_end_markers` (llmemory.py) — `<array of STRUCT>` identity →
    /// its `_endmarker`. `ItemOffset.ref` memoizes the sentinel per parent
    /// array so two references to one array's end share an identity.
    /// Upstream uses a `WeakKeyDictionary`; the translator never frees these
    /// containers, so a strong map keyed by identity is the consistent
    /// adaptation (as for `_subarray._cache` / `_arraylenref._cache`).
    static END_MARKERS: RefCell<HashMap<usize, _endmarker>> = RefCell::new(HashMap::new());
}

/// RPython `_fakeaccessor` (`llmemory.py`) and typed subclasses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _fakeaccessor {
    pub addr: _address,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _signed_fakeaccessor(pub _fakeaccessor);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _unsigned_fakeaccessor(pub _fakeaccessor);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _float_fakeaccessor(pub _fakeaccessor);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _char_fakeaccessor(pub _fakeaccessor);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _address_fakeaccessor(pub _fakeaccessor);

/// RPython `supported_access_types` (`llmemory.py`).
#[allow(non_upper_case_globals)]
pub static supported_access_types: LazyLock<HashMap<&'static str, LowLevelType>> =
    LazyLock::new(|| {
        HashMap::from([
            ("signed", LowLevelType::Signed),
            ("unsigned", LowLevelType::Unsigned),
            ("char", LowLevelType::Char),
            ("address", LowLevelType::Address),
            ("float", LowLevelType::Float),
        ])
    });

fn gcarray_of_ptr_type() -> LowLevelType {
    LowLevelType::Array(Box::new(Array::gc_with_hints(
        (*GCREF).clone(),
        vec![("placeholder".into(), ConstValue::Bool(true))],
    )))
}

/// RPython `gcarrayofptr_lengthoffset` (`llmemory.py`).
#[allow(non_upper_case_globals)]
pub static gcarrayofptr_lengthoffset: LazyLock<AddressOffset> =
    LazyLock::new(|| AddressOffset::ArrayLengthOffset(ArrayLengthOffset(gcarray_of_ptr_type())));

/// RPython `gcarrayofptr_itemsoffset` (`llmemory.py`).
#[allow(non_upper_case_globals)]
pub static gcarrayofptr_itemsoffset: LazyLock<AddressOffset> =
    LazyLock::new(|| AddressOffset::ArrayItemsOffset(ArrayItemsOffset(gcarray_of_ptr_type())));

/// RPython `gcarrayofptr_singleitemoffset` (`llmemory.py`).
#[allow(non_upper_case_globals)]
pub static gcarrayofptr_singleitemoffset: LazyLock<AddressOffset> = LazyLock::new(|| {
    AddressOffset::ItemOffset(ItemOffset {
        TYPE: (*GCREF).clone(),
        repeat: 1,
    })
});

/// RPython `RawMemmoveEntry(ExtRegistryEntry)` (`llmemory.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RawMemmoveEntry;

fn deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.llmemory.{name} — raw memory/address helper deferred"
    ))
}

pub fn ann_offsetof() -> Result<(), TyperError> {
    Err(deferred("ann_offsetof"))
}

pub fn ann_cast_ptr_to_adr() -> Result<(), TyperError> {
    Err(deferred("ann_cast_ptr_to_adr"))
}

pub fn ann_cast_adr_to_ptr() -> Result<(), TyperError> {
    Err(deferred("ann_cast_adr_to_ptr"))
}

pub fn ann_cast_adr_to_int() -> Result<(), TyperError> {
    Err(deferred("ann_cast_adr_to_int"))
}

pub fn ann_cast_int_to_adr() -> Result<(), TyperError> {
    Err(deferred("ann_cast_int_to_adr"))
}

pub fn ann_weakref_create() -> Result<(), TyperError> {
    Err(deferred("ann_weakref_create"))
}

pub fn ann_weakref_deref() -> Result<(), TyperError> {
    Err(deferred("ann_weakref_deref"))
}

pub fn cast_ptr_to_weakrefptr() -> Result<(), TyperError> {
    Err(deferred("cast_ptr_to_weakrefptr"))
}

pub fn llcast_ptr_to_weakrefptr() -> Result<(), TyperError> {
    Err(deferred("llcast_ptr_to_weakrefptr"))
}

pub fn cast_weakrefptr_to_ptr() -> Result<(), TyperError> {
    Err(deferred("cast_weakrefptr_to_ptr"))
}

pub fn llcast_weakrefptr_to_ptr() -> Result<(), TyperError> {
    Err(deferred("llcast_weakrefptr_to_ptr"))
}

pub fn raw_malloc() -> Result<(), TyperError> {
    Err(deferred("raw_malloc"))
}

pub fn ann_raw_malloc() -> Result<(), TyperError> {
    Err(deferred("ann_raw_malloc"))
}

pub fn raw_free() -> Result<(), TyperError> {
    Err(deferred("raw_free"))
}

pub fn ann_raw_free() -> Result<(), TyperError> {
    Err(deferred("ann_raw_free"))
}

pub fn raw_malloc_usage() -> Result<(), TyperError> {
    Err(deferred("raw_malloc_usage"))
}

pub fn ann_raw_malloc_usage() -> Result<(), TyperError> {
    Err(deferred("ann_raw_malloc_usage"))
}

pub fn raw_memclear() -> Result<(), TyperError> {
    Err(deferred("raw_memclear"))
}

pub fn ann_raw_memclear() -> Result<(), TyperError> {
    Err(deferred("ann_raw_memclear"))
}

pub fn raw_memcopy() -> Result<(), TyperError> {
    Err(deferred("raw_memcopy"))
}

pub fn ann_raw_memcopy() -> Result<(), TyperError> {
    Err(deferred("ann_raw_memcopy"))
}

pub fn raw_memmove() -> Result<(), TyperError> {
    Err(deferred("raw_memmove"))
}

pub fn raw_memmove_no_free() -> Result<(), TyperError> {
    Err(deferred("raw_memmove_no_free"))
}

fn _reccopy_lazy() -> Result<(), TyperError> {
    Err(deferred("_reccopy_lazy"))
}

fn _reccopy() -> Result<(), TyperError> {
    Err(deferred("_reccopy"))
}

/// `class SomeAddress(SomeObject)` (llmemory.py).
/// Annotation for low-level Address values. `immutable = True`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeAddress {
    pub base: SomeObject,
}

impl SomeAddress {
    pub fn new() -> Self {
        SomeAddress {
            base: SomeObject::new(KnownType::Address, true),
        }
    }

    /// `def is_null_address(self)` (llmemory.py).
    /// `return self.is_immutable_constant() and not self.const`
    /// — true when the annotation carries a constant that is a falsy
    /// address value (i.e. NULL / fakeaddress(None)).
    pub fn is_null_address(&self) -> bool {
        if !self.is_immutable_constant() {
            return false;
        }
        match &self.base.const_box {
            Some(c) => c.value.is_null_address(),
            None => false,
        }
    }

    /// `def getattr(self, s_attr)` (llmemory.py).
    /// Returns the annotation for `addr.<access_type>` — the intermediate
    /// value used in `addr.signed[offset]` patterns.
    pub fn annotation_getattr(attr: &str) -> Option<SomeTypedAddressAccess> {
        supported_access_type(attr).map(SomeTypedAddressAccess::new)
    }

    /// `def bool(self)` (llmemory.py).
    /// `return s_Bool`
    pub fn annotation_bool() -> SomeValue {
        SomeValue::Bool(crate::annotator::model::SomeBool::new())
    }
}

impl Default for SomeAddress {
    fn default() -> Self {
        SomeAddress::new()
    }
}

impl SomeObjectTrait for SomeAddress {
    fn knowntype(&self) -> KnownType {
        KnownType::Address
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// llmemory.py:730-735
fn supported_access_type(name: &str) -> Option<LowLevelType> {
    supported_access_types.get(name).cloned()
}

/// `class SomeTypedAddressAccess(SomeObject)` (llmemory.py).
/// Annotation for the intermediate value in `addr.signed[offset]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeTypedAddressAccess {
    pub access_type: LowLevelType,
    pub base: SomeObject,
}

impl SomeTypedAddressAccess {
    pub fn new(access_type: LowLevelType) -> Self {
        SomeTypedAddressAccess {
            access_type,
            base: SomeObject::new(KnownType::Object, false),
        }
    }
}

impl SomeObjectTrait for SomeTypedAddressAccess {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        false
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// `class Symbolic` (llmemory.py) → `class AddressOffset(Symbolic)`
/// (llmemory.py) and its subclasses. The runtime `ref` / `_raw_malloc`
/// / `raw_memcopy` methods operate on the `fakeaddress` simulator, which
/// pyre does not model; what flows through the annotator and rtyper is
/// the rtyping-level structure — variant identity, `known_nonneg`,
/// symbolic arithmetic, and `lltype() == Signed`.
///
/// `GCHeaderOffset` / `GCHeaderAntiOffset` (llmemory.py) are
/// omitted (blocker: GC transform not run). They carry a `gcheaderbuilder`
/// and are minted only by the GC transformer (`gctransform/`), which pyre
/// does not run; no annotator/rtyper path constructs one, so adding the
/// variants now would be unreachable dead code. Convergence path: port
/// them alongside the GC transform when/if pyre grows one — at that point
/// the `gcheaderbuilder` (`header_of_object` / `object_from_header`)
/// becomes the real dependency to port first.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemOffset {
    pub TYPE: LowLevelType,
    pub repeat: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldOffset {
    pub TYPE: LowLevelType,
    pub fldname: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompositeOffset {
    pub offsets: Vec<AddressOffset>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayItemsOffset(pub LowLevelType);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayLengthOffset(pub LowLevelType);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddressOffset {
    /// llmemory.py `class ItemOffset(AddressOffset)`.
    ItemOffset(ItemOffset),
    /// llmemory.py `class FieldOffset(AddressOffset)`.
    FieldOffset(FieldOffset),
    /// llmemory.py `class CompositeOffset(AddressOffset)`.
    CompositeOffset(CompositeOffset),
    /// llmemory.py `class ArrayItemsOffset(AddressOffset)`.
    ArrayItemsOffset(ArrayItemsOffset),
    /// llmemory.py `class ArrayLengthOffset(AddressOffset)`.
    ArrayLengthOffset(ArrayLengthOffset),
}

impl AddressOffset {
    /// llmemory.py `def lltype(self): return lltype.Signed`.
    pub fn lltype(&self) -> LowLevelType {
        LowLevelType::Signed
    }

    /// llmemory.py/77/195/255/286/333 `known_nonneg`.
    pub fn known_nonneg(&self) -> bool {
        match self {
            AddressOffset::ItemOffset(offset) => offset.repeat >= 0,
            AddressOffset::FieldOffset(_) => true,
            AddressOffset::ArrayItemsOffset(_) => true,
            AddressOffset::ArrayLengthOffset(_) => true,
            AddressOffset::CompositeOffset(offset) => {
                offset.offsets.iter().all(|o| o.known_nonneg())
            }
        }
    }

    /// llmemory.py `def __add__(self, other): return
    /// CompositeOffset(self, other)`.
    #[expect(
        clippy::should_implement_trait,
        reason = "This method is the direct llmemory.py __add__ operation used by the translator; AddressOffset's heterogeneous and fallible operations do not share Rust Add's output contract"
    )]
    pub fn add(self, other: AddressOffset) -> AddressOffset {
        AddressOffset::composite(vec![self, other])
    }

    /// llmemory.py `ItemOffset.__mul__` (`__rmul__ = __mul__`).
    /// Non-`ItemOffset` returns `NotImplemented` upstream → `None` here.
    #[expect(
        clippy::should_implement_trait,
        reason = "This preserves llmemory.py __mul__ returning NotImplemented for unsupported offset kinds, which cannot be represented by the standard Mul trait contract"
    )]
    pub fn mul(self, other: i64) -> Option<AddressOffset> {
        match self {
            AddressOffset::ItemOffset(offset) => Some(AddressOffset::ItemOffset(ItemOffset {
                TYPE: offset.TYPE,
                repeat: offset.repeat * other,
            })),
            _ => None,
        }
    }

    /// llmemory.py `ItemOffset.__neg__`; :250-253 `CompositeOffset
    /// .__neg__`. Only those two define `__neg__` upstream; for the other
    /// variants `-offset` raises `TypeError` (no `__neg__`), so `None`.
    #[expect(
        clippy::should_implement_trait,
        reason = "This preserves llmemory.py's partial __neg__ operation and its NotImplemented cases; std::ops::Neg requires an unconditional output"
    )]
    pub fn neg(self) -> Option<AddressOffset> {
        match self {
            AddressOffset::ItemOffset(offset) => Some(AddressOffset::ItemOffset(ItemOffset {
                TYPE: offset.TYPE,
                repeat: -offset.repeat,
            })),
            // llmemory.py __neg__ `ofs = [-item for item in self.offsets];
            // ofs.reverse(); return CompositeOffset(*ofs)`. The list
            // comprehension negates every element — if any `-item` raises
            // (FieldOffset/ArrayItemsOffset/ArrayLengthOffset have no
            // `__neg__`), the whole `__neg__` raises. `collect::<Option<_>>`
            // short-circuits to `None` so a non-negatable element is not
            // silently dropped.
            AddressOffset::CompositeOffset(offset) => {
                let mut ofs = offset
                    .offsets
                    .into_iter()
                    .map(|o| o.neg())
                    .collect::<Option<Vec<AddressOffset>>>()?;
                ofs.reverse();
                Some(AddressOffset::composite(ofs))
            }
            _ => None,
        }
    }

    /// llmemory.py `CompositeOffset.__new__` — flatten nested
    /// composites, merge adjacent same-`TYPE` `ItemOffset`s, and collapse
    /// a single-element list to its sole offset.
    pub fn composite(offsets: Vec<AddressOffset>) -> AddressOffset {
        let mut lst: Vec<AddressOffset> = Vec::new();
        for item in offsets {
            match item {
                AddressOffset::CompositeOffset(inner) => lst.extend(inner.offsets),
                other => lst.push(other),
            }
        }
        let mut i = lst.len().wrapping_sub(2);
        while (i as isize) >= 0 {
            if let (AddressOffset::ItemOffset(left), AddressOffset::ItemOffset(right)) =
                (&lst[i], &lst[i + 1])
                && left.TYPE == right.TYPE
            {
                let merged = AddressOffset::ItemOffset(ItemOffset {
                    TYPE: left.TYPE.clone(),
                    repeat: left.repeat + right.repeat,
                });
                lst.splice(i..i + 2, std::iter::once(merged));
            }
            i = i.wrapping_sub(1);
        }
        if lst.len() == 1 {
            lst.pop().unwrap()
        } else {
            AddressOffset::CompositeOffset(CompositeOffset { offsets: lst })
        }
    }

    /// Concrete byte size for code emission. pyre interprets / JITs rather
    /// than emitting C, so a symbolic offset that reaches the assembler is
    /// resolved to its concrete size here — the role RPython's
    /// `rpython/jit/backend/llsupport/symbolic.py` plays for the real
    /// backends (`get_field_token` / `get_size` / `get_array_token`).
    /// Struct field offsets and struct sizes come from `layout`, which the
    /// codewriter (the owner of struct layouts) supplies; primitive item
    /// sizes and the standard length-prefixed array tokens are computed
    /// directly.
    pub fn byte_size(&self, layout: &dyn OffsetLayout) -> Result<i64, String> {
        match self {
            AddressOffset::ItemOffset(offset) => {
                Ok(item_byte_size(&offset.TYPE, layout)? * offset.repeat)
            }
            // `symbolic.get_field_token(STRUCT, fldname)[0]`.
            AddressOffset::FieldOffset(offset) => {
                let LowLevelType::Struct(st) = &offset.TYPE else {
                    return Err(format!("FieldOffset on non-struct {:?}", offset.TYPE));
                };
                layout
                    .field_offset(&st._name, &offset.fldname)
                    .ok_or_else(|| {
                        format!(
                            "no field offset for {}.{} (get_field_token)",
                            st._name, offset.fldname
                        )
                    })
            }
            AddressOffset::CompositeOffset(offset) => {
                let mut total = 0;
                for o in &offset.offsets {
                    total += o.byte_size(layout)?;
                }
                Ok(total)
            }
            // `symbolic.get_array_token` basesize: the items start one word
            // after the length field for a standard length-prefixed array,
            // or at offset 0 for a `nolength` array (symbolic.py:39-42,
            // which sets `ofs_length = -1` and the items at the base).
            AddressOffset::ArrayItemsOffset(offset) => {
                if array_is_nolength(&offset.0) {
                    Ok(0)
                } else {
                    Ok(WORD)
                }
            }
            AddressOffset::ArrayLengthOffset(_) => Ok(0),
        }
    }

    /// `AddressOffset.ref(ptr)` (llmemory.py per variant: ItemOffset:79,
    /// FieldOffset:198, CompositeOffset:261, ArrayItemsOffset:289) — navigate
    /// `ptr` by this offset to the interior pointer it denotes.
    ///
    /// The container-element cases resolve via [`parentlink`] + container
    /// `getitem`/`_getattr` and hand back a `_container._as_ptr()`. The
    /// primitive/pointer-element cases (upstream `direct_ptradd`/
    /// `direct_fieldptr`/`direct_arrayitems`) build `_subarray` interior
    /// pointers, the array-end marker builds an `_endmarker_struct`, and
    /// `ArrayLengthOffset` builds an `_arraylenref`. A case still returns `Err`
    /// (so a constant fold over the offset declines) only when the base pointer
    /// cannot be navigated — a delayed/NULL `ptr` or an element kind the
    /// container does not model.
    pub fn r#ref(&self, ptr: &_ptr) -> Result<_ptr, String> {
        match self {
            AddressOffset::ItemOffset(offset) => item_offset_ref(&offset.TYPE, offset.repeat, ptr),
            AddressOffset::FieldOffset(offset) => {
                field_offset_ref(&offset.TYPE, &offset.fldname, ptr)
            }
            AddressOffset::ArrayItemsOffset(offset) => array_items_offset_ref(&offset.0, ptr),
            // llmemory.py `for item in self.offsets: ptr = item.ref(ptr)`.
            AddressOffset::CompositeOffset(offset) => {
                let mut p = ptr.clone();
                for o in &offset.offsets {
                    p = o.r#ref(&p)?;
                }
                Ok(p)
            }
            AddressOffset::ArrayLengthOffset(offset) => array_length_offset_ref(&offset.0, ptr),
        }
    }
}

/// `ItemOffset.ref(firstitemptr)` (llmemory.py), array-of-containers
/// arm: `parent, index = parentlink(firstitemptr._obj); index += repeat;
/// parent.getitem(index)._as_ptr()`.
fn item_offset_ref(ty: &LowLevelType, repeat: i64, firstitemptr: &_ptr) -> Result<_ptr, String> {
    let a: LowLevelType = firstitemptr._TYPE.TO.clone().into();
    if &a != ty {
        // `A` is a FixedSizeArray (or nolength Array) of primitives or pointers
        // whose item type matches `self.TYPE` → `direct_ptradd(firstitemptr,
        // repeat)` (llmemory.py:104-109). Any other `A` is a `TypeError`
        // (llmemory.py:110-111) and declines to fold.
        if primitive_array_matches_item(&a, ty) {
            return direct_ptradd(firstitemptr, repeat);
        }
        return Err(format!("ItemOffset::ref: got {a:?}, expected {ty:?}"));
    }
    let obj = firstitemptr
        ._obj()
        .map_err(|_| "ItemOffset::ref: delayed pointer".to_string())?;
    let (parent, index) = parentlink(&obj);
    let parent = parent.ok_or_else(|| format!("{firstitemptr:?} is not within a container"))?;
    let _ptr_obj::Array(arr) = &parent else {
        return Err(format!("{firstitemptr:?} is not within an array"));
    };
    // `if isinstance(index, str): assert index.startswith('item'); index = int(index[4:])`.
    let base: i64 = match index {
        Some(ParentIndex::Item(j)) => j,
        Some(ParentIndex::Field(name)) => name
            .strip_prefix("item")
            .and_then(|s| s.parse::<i64>().ok())
            .ok_or_else(|| format!("unexpected array parentindex {name:?}"))?,
        None => return Err("parent container has no index".into()),
    };
    let index = base + repeat;
    let len = arr.getlength() as i64;
    if index == len {
        // `for references exactly to the end of the array` →
        // `_end_markers[parent]` else `_endmarker_struct(A, parent, index)`,
        // then `._as_ptr()` (llmemory.py). `A == self.TYPE` here, so
        // `ty` is the array item struct type.
        let LowLevelType::Struct(item_struct) = ty else {
            return Err(format!(
                "ItemOffset::ref end marker on non-struct item {ty:?}"
            ));
        };
        // `_end_markers` is a `WeakKeyDictionary` keyed by the parent array
        // (llmemory.py). The keyed `Arc::as_ptr` address can be reused
        // after the array is dropped, so a cache hit is accepted only when the
        // stored end marker's weak `_wrparent` still upgrades to *this* array
        // (`parent_is`); a stale hit on a reused address recomputes, emulating
        // the auto-evicting dictionary.
        let array_id = arr.identity();
        let cached = END_MARKERS
            .with(|c| c.borrow().get(&array_id).cloned())
            .filter(|e| e.parent_is(&parent));
        let endmarker = cached.unwrap_or_else(|| {
            let e = _endmarker::new(item_struct, &parent, index as usize);
            END_MARKERS.with(|c| c.borrow_mut().insert(array_id, e.clone()));
            e
        });
        return Ok(endmarker._as_ptr(true));
    }
    // `parent.getitem(index)` (llmemory.py). A varsize `_array.getitem`
    // (lltype.py) is Python list indexing, so a negative index addresses
    // from the end; a fixed-size array `_fixedsizearray.getitem`
    // (lltype.py) asserts `0 <= index < length`, so a negative index is
    // rejected. Out-of-range declines either way.
    let actual = match &arr.TYPE {
        ArrayContainer::Array(_) if index < 0 => len + index,
        _ => index,
    };
    if actual < 0 || actual >= len {
        return Err(format!("ItemOffset::ref item {index} out of range"));
    }
    let item = arr
        .getitem(actual as usize)
        .ok_or_else(|| format!("ItemOffset::ref item {index} out of range"))?;
    container_value_as_ptr(&item, true)
        .ok_or_else(|| "ItemOffset::ref: item is not a container".into())
}

/// `FieldOffset.ref(struct)` (llmemory.py), container-field arm:
/// `substruct = struct._obj._getattr(fldname); substruct._as_ptr()`.
fn field_offset_ref(ty: &LowLevelType, fldname: &str, struct_ptr: &_ptr) -> Result<_ptr, String> {
    // `if lltype.typeOf(struct).TO != self.TYPE: struct = cast_pointer(
    //  Ptr(self.TYPE), struct)` (llmemory.py) — up-cast to the
    // offset's struct type when the pointer is to an inner sub-struct.
    let cur: LowLevelType = struct_ptr._TYPE.TO.clone().into();
    let casted;
    let struct_ptr: &_ptr = if &cur != ty {
        casted = cast_pointer(&Ptr::from_container_type(ty.clone())?, struct_ptr)?;
        &casted
    } else {
        struct_ptr
    };
    let LowLevelType::Struct(st) = ty else {
        return Err(format!("FieldOffset::ref on non-struct {ty:?}"));
    };
    let field_ty = st
        ._flds
        .get(fldname)
        .ok_or_else(|| format!("{} has no field {fldname:?}", st._name))?;
    if !field_ty.is_container_type() {
        // primitive field → `direct_fieldptr(struct, fldname)`
        // (llmemory.py:205-206).
        return direct_fieldptr(struct_ptr, fldname);
    }
    let obj = struct_ptr
        ._obj()
        .map_err(|_| "FieldOffset::ref: delayed pointer".to_string())?;
    let _ptr_obj::Struct(s) = &obj else {
        return Err(format!("FieldOffset::ref target is not a struct: {obj:?}"));
    };
    let sub = s
        ._getattr(fldname)
        .ok_or_else(|| format!("struct has no field {fldname:?}"))?;
    container_value_as_ptr(&sub, true)
        .ok_or_else(|| "FieldOffset::ref: field is not a container".into())
}

/// `ArrayItemsOffset.ref(arrayptr)` (llmemory.py), array-of-containers
/// arm: `arrayptr._obj.getitem(0)._as_ptr()`.
fn array_items_offset_ref(ty: &LowLevelType, arrayptr: &_ptr) -> Result<_ptr, String> {
    // `assert array_type_match(lltype.typeOf(arrayptr).TO, self.TYPE)`
    // (llmemory.py) — the pointer's array type must match the offset's,
    // otherwise the navigation is invalid and the fold declines.
    let a1: LowLevelType = arrayptr._TYPE.TO.clone().into();
    if !array_type_match(&a1, ty) {
        return Err(format!(
            "ArrayItemsOffset::ref: array type mismatch: {a1:?} vs {ty:?}"
        ));
    }
    let of = array_of_type(ty)?;
    if !of.is_container_type() {
        // primitive array → `direct_arrayitems(arrayptr)` (llmemory.py).
        return direct_arrayitems(arrayptr);
    }
    let obj = arrayptr
        ._obj()
        .map_err(|_| "ArrayItemsOffset::ref: delayed pointer".to_string())?;
    let _ptr_obj::Array(arr) = &obj else {
        return Err(format!(
            "ArrayItemsOffset::ref target is not an array: {obj:?}"
        ));
    };
    let item0 = arr
        .getitem(0)
        .ok_or_else(|| "ArrayItemsOffset::ref: empty array".to_string())?;
    container_value_as_ptr(&item0, true)
        .ok_or_else(|| "ArrayItemsOffset::ref: item is not a container".into())
}

/// `ArrayLengthOffset.ref(arrayptr)` (llmemory.py):
/// `_arraylenref._makeptr(arrayptr._obj, arrayptr._solid)` — a
/// `Ptr(FixedSizeArray(Signed, 1))` whose `getitem(0)` reads the array length.
fn array_length_offset_ref(ty: &LowLevelType, arrayptr: &_ptr) -> Result<_ptr, String> {
    // `assert array_type_match(lltype.typeOf(arrayptr).TO, self.TYPE)`
    // (llmemory.py).
    let a1: LowLevelType = arrayptr._TYPE.TO.clone().into();
    if !array_type_match(&a1, ty) {
        return Err(format!(
            "ArrayLengthOffset::ref: array type mismatch: {a1:?} vs {ty:?}"
        ));
    }
    let obj = arrayptr
        ._obj()
        .map_err(|_| "ArrayLengthOffset::ref: delayed pointer".to_string())?;
    let _ptr_obj::Array(arr) = obj else {
        return Err(format!(
            "ArrayLengthOffset::ref target is not an array: {obj:?}"
        ));
    };
    Ok(_arraylenref::_makeptr(Box::new(arr), arrayptr._solid))
}

/// `TYPE.OF` for an array lltype.
fn array_of_type(ty: &LowLevelType) -> Result<LowLevelType, String> {
    match ty {
        LowLevelType::Array(t) => Ok(t.OF.clone()),
        LowLevelType::FixedSizeArray(t) => Ok(t.OF.clone()),
        _ => Err(format!("expected an array type, got {ty:?}")),
    }
}

/// `(isinstance(A, FixedSizeArray) or (isinstance(A, Array) and
/// A._hints.get('nolength', False))) and array_item_type_match(A.OF,
/// self.TYPE)` (llmemory.py) — the predicate routing `ItemOffset.ref`
/// to `direct_ptradd`. A `direct_arrayitems`-derived pointer is always a
/// `FixedSizeArray(ITEM, 1)`, so that is the live arm; the nolength-`Array`
/// arm covers bare C-like arrays.
fn primitive_array_matches_item(a: &LowLevelType, ty: &LowLevelType) -> bool {
    match a {
        LowLevelType::FixedSizeArray(t) => array_item_type_match(&t.OF, ty),
        LowLevelType::Array(t) => {
            matches!(t._hints.get("nolength"), Some(ConstValue::Bool(true)))
                && array_item_type_match(&t.OF, ty)
        }
        _ => false,
    }
}

/// `array_item_type_match(T1, T2)` (llmemory.py): exact item-type
/// equality, or a concrete pointer item accepted by the generic `GCREF`
/// placeholder item type.
pub fn array_item_type_match(t1: &LowLevelType, t2: &LowLevelType) -> bool {
    t1 == t2 || (t2 == &*GCREF && matches!(t1, LowLevelType::Ptr(_)))
}

/// `array_type_match(A1, A2)` (llmemory.py): the offset's stored array
/// type `A2` must equal the pointer's actual array type `A1`, or `A2` is
/// exactly the `GCARRAY_OF_PTR` token and `A1` is a length-prefixed `GcArray`
/// of pointers. `GCARRAY_OF_PTR` (`GcArray(GCREF, hints={'placeholder': True})`)
/// is minted only by the GC transformer, which pyre does not run, so the
/// placeholder arm is unreachable here; it is modelled for completeness. `A2`
/// must match `GCARRAY_OF_PTR` exactly — a gc array of [`GCREF`] carrying the
/// `placeholder` hint — not merely any placeholder gc array of pointers.
pub fn array_type_match(a1: &LowLevelType, a2: &LowLevelType) -> bool {
    if a1 == a2 {
        return true;
    }
    let LowLevelType::Array(a2_arr) = a2 else {
        return false;
    };
    let a2_is_gcarray_of_ptr = a2_arr._gckind == GcKind::Gc
        && a2_arr.OF == *GCREF
        && matches!(
            a2_arr._hints.get("placeholder"),
            Some(ConstValue::Bool(true))
        );
    if !a2_is_gcarray_of_ptr {
        return false;
    }
    matches!(
        a1,
        LowLevelType::Array(a1_arr)
            if a1_arr._gckind == GcKind::Gc
                && matches!(a1_arr.OF, LowLevelType::Ptr(_))
                && !matches!(a1_arr._hints.get("nolength"), Some(ConstValue::Bool(true)))
    )
}

/// Word size (length-field width / pointer width).
const WORD: i64 = std::mem::size_of::<usize>() as i64;

/// C ABI `sizeof(wchar_t)` from a cargo target triple.
///
/// Windows is 2; Linux / macOS / wasm32 clang are 4 (`r_wchar_t.BITS/8`,
/// `ll2ctypes` `sizeof(c_wchar)`).
pub(crate) fn sizeof_wchar_t_for(os: &str, _arch: &str, _env: &str) -> i64 {
    if os == "windows" { 2 } else { 4 }
}

/// Target `wchar_t` width in bytes. Reads cargo's `CARGO_CFG_TARGET_OS`
/// through [`crate::layout::target_os`] (same channel as
/// `layout::target_word_size`); host `cfg!` when unset.
pub(crate) fn sizeof_wchar_t() -> i64 {
    static SIZE: OnceLock<i64> = OnceLock::new();
    *SIZE.get_or_init(|| {
        sizeof_wchar_t_for(
            crate::layout::target_os(),
            crate::layout::target_arch(),
            crate::layout::target_env(),
        )
    })
}

/// C ABI `sizeof(long double)` from a cargo target triple.
///
/// Upstream `rffi.py sizeof` / `sizeof_c_type("long double")` probes the
/// C compiler. This table stands in for that probe using clang TargetInfo
/// `LongDoubleWidth` (bytes). Unknown triples fall back to the host
/// `cfg!` width.
pub(crate) fn sizeof_longfloat_for(os: &str, arch: &str, env: &str) -> i64 {
    longfloat_width_bytes(os, arch, env).unwrap_or_else(|| {
        let host_env = if cfg!(target_env = "msvc") {
            "msvc"
        } else {
            ""
        };
        longfloat_width_bytes(std::env::consts::OS, std::env::consts::ARCH, host_env).unwrap_or(16)
    })
}

fn longfloat_width_bytes(os: &str, arch: &str, env: &str) -> Option<i64> {
    Some(match arch {
        "x86" => {
            if env == "msvc" {
                8
            } else if matches!(os, "macos" | "ios") {
                16
            } else if os == "android" {
                8
            } else {
                // Linux, BSDs, MinGW
                12
            }
        }
        "x86_64" => {
            if env == "msvc" {
                8
            } else {
                16
            }
        }
        "aarch64" => {
            if matches!(os, "macos" | "ios" | "tvos" | "watchos" | "windows") {
                8
            } else {
                16
            }
        }
        "arm" | "mips" => 8,
        "wasm32" | "wasm64" | "riscv32" | "riscv64" | "powerpc64" | "s390x" | "mips64"
        | "loongarch64" => 16,
        _ => return None,
    })
}

/// Target `long double` width in bytes. Reads cargo's
/// `CARGO_CFG_TARGET_{OS,ARCH,ENV}` through [`crate::layout::target_os`]
/// / `target_arch` / `target_env` (same channel as
/// `layout::target_word_size`); host `cfg!` when unset.
pub(crate) fn sizeof_longfloat() -> i64 {
    static SIZE: OnceLock<i64> = OnceLock::new();
    *SIZE.get_or_init(|| {
        sizeof_longfloat_for(
            crate::layout::target_os(),
            crate::layout::target_arch(),
            crate::layout::target_env(),
        )
    })
}

/// Resolver for the layout-dependent symbolic offsets, mirroring
/// `rpython/jit/backend/llsupport/symbolic.py`. The codewriter owns the
/// real struct layouts, so it implements this; defining the trait here
/// (rather than depending on the codewriter) keeps `llmemory` below the
/// `codewriter` layer.
pub trait OffsetLayout {
    /// `symbolic.get_field_token(STRUCT, fldname)[0]` — byte offset of the
    /// field within the struct.
    fn field_offset(&self, struct_name: &str, fldname: &str) -> Option<i64>;
    /// `symbolic.get_size(STRUCT)` — total struct byte size.
    fn struct_size(&self, struct_name: &str) -> Option<i64>;
    /// `symbolic.get_size(lltype.Signed)` — the target word
    /// (`layout::target_word_size`), not the host `usize`.
    /// `Signed` / `Unsigned` / `Address` item sizes read this. A layout
    /// aimed at another word (a 4-byte target while this process is
    /// 8-byte) overrides it; every other layout uses the target word.
    fn word_bytes(&self) -> i64 {
        crate::layout::target_word_size() as i64
    }
}

/// Byte size of a primitive `LowLevelType`. Mirrors
/// `symbolic.get_size(TYPE)` (symbolic.py:24), which resolves to
/// `ctypes.sizeof(ll2ctypes.get_ctypes_type(TYPE))`. `Signed` /
/// `Unsigned` / `Address` are the target word (`word`, from
/// [`OffsetLayout::word_bytes`]). `long long` and `double` stay 8 on
/// every target this codewriter emits; the host `usize` is not their
/// width. `UniChar` is [`sizeof_wchar_t`] (`r_wchar_t.BITS/8`);
/// `LongFloat` is the target-derived `long double` width
/// ([`sizeof_longfloat`]).
fn primitive_byte_size(ty: &LowLevelType, word: i64) -> Result<i64, String> {
    match ty {
        LowLevelType::Signed | LowLevelType::Unsigned | LowLevelType::Address => Ok(word),
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong | LowLevelType::Float => {
            Ok(8)
        }
        // `r_longlonglong` / `r_ulonglonglong` are 128-bit on every target
        // that defines them, so 16 is stable; `long double` is not.
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => Ok(16),
        LowLevelType::LongFloat => Ok(sizeof_longfloat()),
        LowLevelType::SingleFloat => Ok(4),
        LowLevelType::UniChar => Ok(sizeof_wchar_t()),
        LowLevelType::Bool | LowLevelType::Char => Ok(1),
        other => Err(format!("no primitive byte size for {other:?}")),
    }
}

/// `symbolic.get_size(TYPE)` for an `ItemOffset` element type — a
/// primitive width, a struct size from `layout`, or a `FixedSizeArray`
/// laid out as `length` inlined items of `OF`.
fn item_byte_size(ty: &LowLevelType, layout: &dyn OffsetLayout) -> Result<i64, String> {
    let word = layout.word_bytes();
    if let Ok(sz) = primitive_byte_size(ty, word) {
        return Ok(sz);
    }
    match ty {
        // A pointer item is one target word (`ctypes.c_void_p`).
        LowLevelType::Ptr(_) => Ok(word),
        LowLevelType::Struct(st) => layout
            .struct_size(&st._name)
            .ok_or_else(|| format!("no struct layout for {} (get_size)", st._name)),
        LowLevelType::FixedSizeArray(fa) => Ok(item_byte_size(&fa.OF, layout)? * fa.length as i64),
        other => Err(format!("no byte size for item type {other:?}")),
    }
}

/// True when `array_ty` is an `Array` carrying the `'nolength'` hint —
/// the items begin at the container base with no length prefix
/// (symbolic.py:39-42).
fn array_is_nolength(array_ty: &LowLevelType) -> bool {
    matches!(
        array_ty,
        LowLevelType::Array(arr)
            if matches!(arr._hints.get("nolength"), Some(ConstValue::Bool(true)))
    )
}

/// `llmemory.extra_item_after_alloc(ARRAY)` (llmemory.py) — the
/// `'extra_item_after_alloc'` array hint, `0` when absent.
pub fn extra_item_after_alloc(array_ty: &LowLevelType) -> i64 {
    match array_ty {
        LowLevelType::Array(arr) => match arr._hints.get("extra_item_after_alloc") {
            Some(ConstValue::Int(n)) => *n,
            _ => 0,
        },
        _ => 0,
    }
}

/// `TYPE.OF` for an `Array` (llmemory.py `ItemOffset(TYPE.OF)`).
fn array_of(array_ty: &LowLevelType) -> Result<LowLevelType, String> {
    match array_ty {
        LowLevelType::Array(arr) => Ok(arr.OF.clone()),
        other => Err(format!("itemoffsetof: {other:?} is not an Array")),
    }
}

/// `llmemory._sizeof_none(TYPE)` (llmemory.py) — `ItemOffset(TYPE)`,
/// asserting `not TYPE._is_varsize()` (so an `Array` or a varsize `Struct`
/// must be sized with an explicit `n`).
fn sizeof_none(ty: &LowLevelType) -> Result<AddressOffset, String> {
    if ty._is_varsize() {
        return Err(format!("sizeof: {ty:?} is varsize, pass n"));
    }
    Ok(AddressOffset::ItemOffset(ItemOffset {
        TYPE: ty.clone(),
        repeat: 1,
    }))
}

/// `llmemory.offsetof(TYPE, fldname)` (llmemory.py) —
/// `FieldOffset(TYPE, fldname)`, asserting the field exists.
pub fn offsetof(struct_ty: &LowLevelType, fldname: &str) -> Result<AddressOffset, String> {
    let LowLevelType::Struct(st) = struct_ty else {
        return Err(format!("offsetof: {struct_ty:?} is not a Struct"));
    };
    if st._flds.get(fldname).is_none() {
        return Err(format!("offsetof: {} has no field {fldname}", st._name));
    }
    Ok(AddressOffset::FieldOffset(FieldOffset {
        TYPE: struct_ty.clone(),
        fldname: fldname.to_string(),
    }))
}

/// `llmemory.itemoffsetof(TYPE, n=0)` (llmemory.py) —
/// `ArrayItemsOffset(TYPE)`, plus `ItemOffset(TYPE.OF) * n` when `n != 0`.
pub fn itemoffsetof(array_ty: &LowLevelType, n: i64) -> Result<AddressOffset, String> {
    let result = AddressOffset::ArrayItemsOffset(ArrayItemsOffset(array_ty.clone()));
    if n != 0 {
        let of = array_of(array_ty)?;
        let item = AddressOffset::ItemOffset(ItemOffset {
            TYPE: of,
            repeat: 1,
        })
        .mul(n)
        .expect("ItemOffset.mul is always Some");
        Ok(result.add(item))
    } else {
        Ok(result)
    }
}

/// `llmemory.arraylengthoffset(TYPE)` (llmemory.py) —
/// `ArrayLengthOffset(TYPE)`.
pub fn arraylengthoffset(array_ty: &LowLevelType) -> AddressOffset {
    AddressOffset::ArrayLengthOffset(ArrayLengthOffset(array_ty.clone()))
}

/// `llmemory._sizeof_int(TYPE, n)` (llmemory.py) — for a varsize
/// Struct, `offsetof(TYPE, arrayfld) + sizeof(ARRAY, n)`.
fn _internal_array_field(struct_ty: &LowLevelType) -> Result<(String, LowLevelType), String> {
    let LowLevelType::Struct(st) = struct_ty else {
        return Err(format!("don't know how to take the size of {struct_ty:?}"));
    };
    let fldname = st
        ._arrayfld
        .clone()
        .ok_or_else(|| format!("don't know how to take the size of {}", st._name))?;
    let array_ty = st
        ._flds
        .get(&fldname)
        .ok_or_else(|| format!("sizeof: {} missing array field {fldname}", st._name))?
        .clone();
    Ok((fldname, array_ty))
}

/// `llmemory._sizeof_int(TYPE, n)` (llmemory.py) — for a varsize
/// Struct, `offsetof(TYPE, arrayfld) + sizeof(ARRAY, n)`.
fn sizeof_int(struct_ty: &LowLevelType, n: i64) -> Result<AddressOffset, String> {
    let (fldname, array_ty) = _internal_array_field(struct_ty)?;
    Ok(offsetof(struct_ty, &fldname)?.add(sizeof_offset(&array_ty, Some(n))?))
}

/// `llmemory.sizeof(TYPE, n=None)` (llmemory.py). `n=None` sizes a
/// fixed (non-varsize) type; an `Array` is sized as
/// `itemoffsetof(TYPE) + _sizeof_none(TYPE.OF) * (n + extra_item_after_alloc)`;
/// a varsize `Struct` defers to [`sizeof_int`].
fn sizeof_offset(ty: &LowLevelType, n: Option<i64>) -> Result<AddressOffset, String> {
    match n {
        None => sizeof_none(ty),
        Some(n) => match ty {
            LowLevelType::Array(_) => {
                // `n += extra_item_after_alloc(TYPE)`
                let n = n + extra_item_after_alloc(ty);
                let of = array_of(ty)?;
                // `_sizeof_none(TYPE.OF) * n` — `_sizeof_none` asserts the
                // element type is not itself varsize.
                let item = sizeof_none(&of)?
                    .mul(n)
                    .expect("_sizeof_none yields an ItemOffset, mul is Some");
                Ok(itemoffsetof(ty, 0)?.add(item))
            }
            _ => sizeof_int(ty, n),
        },
    }
}

/// `llmemory.sizeof(TYPE, n=None)` (llmemory.py). `inputconst(Signed,
/// sizeof(TYPE))` accepts the result because `AddressOffset.lltype() ==
/// Signed` (matching RPython's `typeOf(Symbolic) -> val.lltype()`).
pub fn sizeof(ty: &LowLevelType, n: Option<i64>) -> Result<ConstValue, String> {
    Ok(ConstValue::AddressOffset(sizeof_offset(ty, n)?))
}

/// `llmemory.dead_wref` (llmemory.py) — `_wref(None)._as_ptr()`, the
/// single prebuilt pointer to a dead low-level weakref.
///
/// A process-wide singleton, matching upstream's module-level `dead_wref`
/// variable: `_ptr` equality respects container identity
/// (lltype.py), so every reference resolves to the same `_wref`
/// container and compares equal. `_ptr` is `Send + Sync`, so the value
/// lives in a `LazyLock` rather than a per-thread cell.
pub fn dead_wref() -> _ptr {
    static DEAD_WREF: LazyLock<_ptr> = LazyLock::new(|| _wref::new(None)._as_ptr());
    (*DEAD_WREF).clone()
}

/// `llmemory.weakref_create(ptarget)` (llmemory.py).
///
/// ```python
/// def weakref_create(ptarget):
///     PTRTYPE = lltype.typeOf(ptarget)
///     assert isinstance(PTRTYPE, lltype.Ptr)
///     assert PTRTYPE.TO._gckind == 'gc'
///     assert ptarget
///     return _wref(ptarget)._as_ptr()
/// ```
///
/// The `_ptr` argument type discharges `isinstance(PTRTYPE, Ptr)`. The
/// target is validated (gc + non-null), then held by a real `_wref`
/// container so [`weakref_deref`] can recover it.
pub fn weakref_create(ptarget: &_ptr) -> Result<_ptr, String> {
    if ptarget._togckind() != GcKind::Gc {
        return Err(format!(
            "weakref_create: target {:?} must be gc-kind",
            ptarget._TYPE
        ));
    }
    if !ptarget.nonzero() {
        return Err("weakref_create: target must be non-null".to_string());
    }
    Ok(_wref::new(Some(ptarget))._as_ptr())
}

/// `llmemory.weakref_deref(PTRTYPE, pwref)` (llmemory.py).
///
/// ```python
/// def weakref_deref(PTRTYPE, pwref):
///     assert isinstance(PTRTYPE, lltype.Ptr)
///     assert PTRTYPE.TO._gckind == 'gc'
///     assert lltype.typeOf(pwref) == WeakRefPtr
///     p = pwref._obj._dereference()
///     if p is None:
///         return lltype.nullptr(PTRTYPE.TO)
///     else:
///         return cast_any_ptr(PTRTYPE, p)
/// ```
///
/// `p = pwref._obj._dereference()` recovers the referent; a dead wref
/// yields `nullptr(PTRTYPE.TO)`, otherwise [`cast_any_ptr`] adapts the
/// concrete referent to the requested `PTRTYPE` (identity, opaque, or a
/// plain pointer cast).
pub fn weakref_deref(PTRTYPE: &LowLevelType, pwref: &_ptr) -> Result<_ptr, String> {
    let LowLevelType::Ptr(ptr_t) = PTRTYPE else {
        return Err(format!(
            "weakref_deref: arg 1 must be a Ptr type, got {PTRTYPE:?}"
        ));
    };
    if ptr_t._gckind() != GcKind::Gc {
        return Err(format!(
            "weakref_deref: arg 1 {PTRTYPE:?} must point to a gc container"
        ));
    }
    let _ptr_obj::Wref(wref) = pwref
        ._obj()
        .map_err(|_| "weakref_deref: arg 2 weakref is a delayed pointer".to_string())?
    else {
        return Err("weakref_deref: arg 2 must be a WeakRefPtr".to_string());
    };
    let deref = wref
        ._dereference()
        .map_err(|_| "weakref_deref: weakref referent is a delayed pointer".to_string())?;
    match deref {
        None => nullptr(LowLevelType::from(ptr_t.TO.clone())),
        Some(p) => cast_any_ptr(ptr_t, &p),
    }
}

/// RPython `cast_any_ptr(EXPECTED_TYPE, ptr)` (llmemory.py) — a
/// generalisation of the `cast_xxx_ptr` family that dispatches on whether
/// either side is `WeakRefPtr` or an `OpaqueType`:
///
/// ```python
/// def cast_any_ptr(EXPECTED_TYPE, ptr):
///     PTRTYPE = lltype.typeOf(ptr)
///     if PTRTYPE == EXPECTED_TYPE:                       return ptr
///     elif EXPECTED_TYPE == WeakRefPtr:                  return cast_ptr_to_weakrefptr(ptr)
///     elif PTRTYPE == WeakRefPtr:
///         ptr = cast_weakrefptr_to_ptr(None, ptr);       return cast_any_ptr(EXPECTED_TYPE, ptr)
///     elif (isinstance(EXPECTED_TYPE.TO, OpaqueType) or
///           isinstance(PTRTYPE.TO, OpaqueType)):         return cast_opaque_ptr(EXPECTED_TYPE, ptr)
///     else:                                              return cast_pointer(EXPECTED_TYPE, ptr)
/// ```
///
/// The two `WeakRefPtr` branches call `cast_ptr_to_weakrefptr` /
/// `cast_weakrefptr_to_ptr`, which upstream notes "exist only after the GC
/// transformation" (llmemory.py) via `_gctransformed_wref`. pyre
/// does not run the GC transformer, so a `WeakRefPtr` operand never reaches
/// here; the branches fail loud rather than fabricate a transformed
/// weakref.
pub fn cast_any_ptr(expected: &Ptr, ptr: &_ptr) -> Result<_ptr, String> {
    let ptrtype = &ptr._TYPE;
    if ptrtype == expected {
        return Ok(ptr.clone());
    }
    let is_weakref_ptr = |p: &Ptr| LowLevelType::Ptr(Box::new(p.clone())) == *WEAKREF_PTR;
    if is_weakref_ptr(expected) {
        return Err("cast_ptr_to_weakrefptr requires a configured GC transformer".to_string());
    }
    if is_weakref_ptr(ptrtype) {
        return Err("cast_weakrefptr_to_ptr requires a configured GC transformer".to_string());
    }
    if matches!(expected.TO, PtrTarget::Opaque(_)) || matches!(ptrtype.TO, PtrTarget::Opaque(_)) {
        return cast_opaque_ptr(expected, ptr);
    }
    cast_pointer(expected, ptr)
}

/// `cast_ptr_to_adr(obj)` (llmemory.py): wrap a low-level pointer in
/// a fake address, normalizing a null pointer to `NULL`.
pub fn cast_ptr_to_adr(obj: &_ptr) -> _address {
    if obj.nonzero() {
        _address::Fake(Box::new(obj.clone()))
    } else {
        _address::Null
    }
}

/// `cast_adr_to_int(adr, mode="emulated")` (llmemory.py).
///
/// `"emulated"`: `adr._cast_to_int(symbolic=False)` — normalized
/// container `_getid`. `"forced"`: the same integer `rffi.cast(Signed,
/// AddressAsInt(adr))` would yield (`fakeaddress._cast_to_int` then
/// `rffi.cast`). `"symbolic"` needs `AddressAsInt`, which is unported.
pub fn cast_adr_to_int(adr: &_address, mode: Option<&str>) -> Result<i64, String> {
    let mode = mode.unwrap_or("emulated");
    match mode {
        "emulated" | "forced" => adr._cast_to_int(mode != "emulated"),
        "symbolic" => Err("cast_adr_to_int symbolic mode needs AddressAsInt".into()),
        other => Err(format!("unsupported cast_adr_to_int mode {other:?}")),
    }
}

/// `cast_int_to_adr(int)` (llmemory.py):
///
/// ```python
/// def cast_int_to_adr(int):
///     if isinstance(int, AddressAsInt):
///         return int.adr
///     try:
///         ptr = lltype.cast_int_to_ptr(_NONGCREF, int)
///     except ValueError:
///         from rpython.rtyper.lltypesystem import ll2ctypes
///         ptr = ll2ctypes._int2obj[int]._as_ptr()
///     return cast_ptr_to_adr(ptr)
/// ```
///
/// The `AddressAsInt` branch is moot — pyre carries no `AddressAsInt`
/// constant. Folding the composition three ways:
/// - `int == 0`: `cast_int_to_ptr` returns `nullptr(_NONGCREF.TO)`, whose
///   `_obj0` is `None`, so `cast_ptr_to_adr` (`fakeaddress.__init__`,
///   llmemory.py) normalizes the null ptr to the NULL address.
/// - odd `int`: `cast_int_to_ptr` builds a tagged-integer `_NONGCREF` `_ptr`,
///   wrapped as the `fakeaddress` (`_address::Fake`).
/// - even non-zero `int`: `cast_int_to_ptr` raises `ValueError`
///   (lltype.py); upstream then resolves it through the runtime
///   `ll2ctypes._int2obj` table (llmemory.py), which has no
///   translation-time value — the fold declines.
pub fn cast_int_to_adr(int: i64) -> Option<_address> {
    if int == 0 {
        Some(_address::Null)
    } else if int & 1 == 1 {
        let ptr = cast_int_to_ptr(&NONGCREF, int).expect("odd int is a valid tagged pointer");
        Some(_address::Fake(Box::new(ptr)))
    } else {
        None
    }
}

/// `cast_adr_to_ptr(adr, EXPECTED_TYPE)` (llmemory.py) =
/// `adr._cast_to_ptr(EXPECTED_TYPE)` (llmemory.py). `_fixup()` is the
/// identity here (the llarena fake-arena rebind is unported). A live address
/// re-casts its pointer with [`cast_any_ptr`]; a NULL address yields
/// `nullptr(EXPECTED_TYPE.TO)` — so the `cast_int_to_adr(0)` round-trip lands
/// here through the `Null` arm. A tagged-integer address re-casts its
/// `_NONGCREF` pointer through [`cast_any_ptr`] like any other live address:
/// `EXPECTED == _NONGCREF` returns the tagged pointer unchanged, and every
/// other concrete/opaque `EXPECTED` fails through `cast_opaque_ptr` because the
/// bare-integer container has no `_obj.container` (upstream `InvalidCast`).
pub fn cast_adr_to_ptr(adr: &_address, expected: &Ptr) -> Result<_ptr, String> {
    match adr {
        _address::Fake(ptr) => cast_any_ptr(expected, ptr),
        _address::Null => nullptr(expected.TO.clone().into()),
        // A `ConstRefAddr` raw host pointer has no live container to re-cast at
        // fold time — upstream's rtyper would hold the prebuilt object's `_ptr`
        // (via `convert_const`); pyre cannot rebuild it from the bare integer.
        // It flows only as a `Ref` constant and never reaches here in practice.
        _address::IntCast(_) => Err(
            "cast_adr_to_ptr on a ConstRefAddr raw host pointer: no live container at fold time"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(ty: LowLevelType, repeat: i64) -> AddressOffset {
        AddressOffset::ItemOffset(ItemOffset { TYPE: ty, repeat })
    }

    fn field(ty: LowLevelType, fldname: &str) -> AddressOffset {
        AddressOffset::FieldOffset(FieldOffset {
            TYPE: ty,
            fldname: fldname.into(),
        })
    }

    fn composite(offsets: Vec<AddressOffset>) -> AddressOffset {
        AddressOffset::CompositeOffset(CompositeOffset { offsets })
    }

    fn array_items(ty: LowLevelType) -> AddressOffset {
        AddressOffset::ArrayItemsOffset(ArrayItemsOffset(ty))
    }

    fn array_length(ty: LowLevelType) -> AddressOffset {
        AddressOffset::ArrayLengthOffset(ArrayLengthOffset(ty))
    }

    #[test]
    fn module_level_parity_surfaces_keep_upstream_names() {
        assert_eq!(
            supported_access_types.get("signed"),
            Some(&LowLevelType::Signed)
        );
        assert_eq!(
            supported_access_types.get("address"),
            Some(&LowLevelType::Address)
        );
        assert_eq!(supported_access_type("float"), Some(LowLevelType::Float));

        assert!(matches!(
            &*gcarrayofptr_lengthoffset,
            AddressOffset::ArrayLengthOffset(_)
        ));
        assert!(matches!(
            &*gcarrayofptr_itemsoffset,
            AddressOffset::ArrayItemsOffset(_)
        ));
        assert_eq!(*gcarrayofptr_singleitemoffset, item((*GCREF).clone(), 1));

        let accessor = _fakeaccessor {
            addr: _address::Null,
        };
        let _signed = _signed_fakeaccessor(accessor.clone());
        let _unsigned = _unsigned_fakeaccessor(accessor.clone());
        let _float = _float_fakeaccessor(accessor.clone());
        let _char = _char_fakeaccessor(accessor.clone());
        let _address = _address_fakeaccessor(accessor);
        let _raw_entry = RawMemmoveEntry;
    }

    #[test]
    fn deferred_llmemory_helpers_name_the_original_surface() {
        let err = ann_offsetof().expect_err("offsetof analyzer is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ann_offsetof"));

        let err = raw_memmove().expect_err("raw memmove is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("raw_memmove"));

        let err = _reccopy().expect_err("recursive copy is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("_reccopy"));
    }

    #[test]
    fn cast_address_to_int_uses_existing_pointer_cast_rules() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            MallocFlavor, Struct, cast_int_to_ptr, cast_ptr_to_int as lltype_cast_ptr_to_int,
            malloc,
        };

        assert_eq!(cast_adr_to_int(&_address::Null, None), Ok(0));
        assert_eq!(
            cast_adr_to_int(&_address::IntCast(41), Some("forced")),
            Ok(41)
        );
        assert!(cast_adr_to_int(&_address::Null, Some("symbolic")).is_err());

        let s = Struct::new("thing", vec![("x".into(), LowLevelType::Signed)]);
        let live = malloc(
            LowLevelType::Struct(Box::new(s.clone())),
            None,
            MallocFlavor::Raw,
            false,
        )
        .unwrap();
        let adr = cast_ptr_to_adr(&live);
        let container_id = lltype_cast_ptr_to_int(&live).unwrap();
        assert_eq!(cast_adr_to_int(&adr, Some("emulated")), Ok(container_id));
        assert_eq!(cast_adr_to_int(&adr, Some("forced")), Ok(container_id));

        let live2 = _ptr::new_with_solid(live._TYPE.clone(), live._obj0_value(), live._solid);
        let adr2 = cast_ptr_to_adr(&live2);
        assert_ne!(live._hashable_identity(), live2._hashable_identity());
        assert_eq!(
            cast_adr_to_int(&adr, Some("forced")),
            cast_adr_to_int(&adr2, Some("forced"))
        );

        let ptr_t = Ptr {
            TO: PtrTarget::Struct(s),
        };
        let tagged = cast_int_to_ptr(&ptr_t, 19).unwrap();
        assert_eq!(cast_adr_to_int(&cast_ptr_to_adr(&tagged), None), Ok(19));
    }

    #[test]
    fn lltype_is_signed_for_every_variant() {
        // llmemory.py `def lltype(self): return lltype.Signed`.
        assert_eq!(item(LowLevelType::Signed, 1).lltype(), LowLevelType::Signed);
        assert_eq!(
            field(LowLevelType::Signed, "x").lltype(),
            LowLevelType::Signed
        );
    }

    #[test]
    fn item_offset_known_nonneg_tracks_repeat_sign() {
        // llmemory.py `return self.repeat >= 0`.
        assert!(item(LowLevelType::Signed, 3).known_nonneg());
        assert!(item(LowLevelType::Signed, 0).known_nonneg());
        assert!(!item(LowLevelType::Signed, -1).known_nonneg());
    }

    #[test]
    fn field_and_array_offsets_are_known_nonneg() {
        // llmemory.py/286/333 — FieldOffset/ArrayItemsOffset/
        // ArrayLengthOffset all `known_nonneg() -> True`.
        assert!(field(LowLevelType::Signed, "f").known_nonneg());
        assert!(array_items(LowLevelType::Signed).known_nonneg());
        assert!(array_length(LowLevelType::Signed).known_nonneg());
    }

    #[test]
    fn item_offset_mul_scales_repeat() {
        // llmemory.py `ItemOffset.__mul__`.
        assert_eq!(
            item(LowLevelType::Signed, 2).mul(3),
            Some(item(LowLevelType::Signed, 6))
        );
        // Non-ItemOffset `__mul__` returns NotImplemented upstream → None.
        assert_eq!(array_items(LowLevelType::Signed).mul(3), None);
    }

    #[test]
    fn item_offset_neg_negates_repeat() {
        // llmemory.py `ItemOffset.__neg__`.
        assert_eq!(
            item(LowLevelType::Signed, 4).neg(),
            Some(item(LowLevelType::Signed, -4))
        );
    }

    #[test]
    fn composite_flattens_nested_composites() {
        // llmemory.py — nested CompositeOffset is spliced inline.
        let inner = composite(vec![
            field(LowLevelType::Signed, "a"),
            array_items(LowLevelType::Char),
        ]);
        let outer = AddressOffset::composite(vec![inner, array_length(LowLevelType::Char)]);
        match outer {
            AddressOffset::CompositeOffset(offset) => assert_eq!(offset.offsets.len(), 3),
            other => panic!("expected CompositeOffset, got {other:?}"),
        }
    }

    #[test]
    fn composite_merges_adjacent_same_type_item_offsets() {
        // llmemory.py:234-239 — adjacent same-TYPE ItemOffsets merge; a
        // single resulting element collapses (llmemory.py:240-241).
        let merged = AddressOffset::composite(vec![
            item(LowLevelType::Signed, 2),
            item(LowLevelType::Signed, 3),
        ]);
        assert_eq!(merged, item(LowLevelType::Signed, 5));
    }

    #[test]
    fn composite_keeps_distinct_type_item_offsets_separate() {
        let offset = AddressOffset::composite(vec![
            item(LowLevelType::Signed, 2),
            item(LowLevelType::Char, 3),
        ]);
        assert_eq!(
            offset,
            composite(vec![
                item(LowLevelType::Signed, 2),
                item(LowLevelType::Char, 3),
            ])
        );
    }

    #[test]
    fn composite_neg_negates_and_reverses() {
        // llmemory.py `CompositeOffset.__neg__`.
        let offset = composite(vec![
            item(LowLevelType::Signed, 2),
            item(LowLevelType::Char, 3),
        ]);
        assert_eq!(
            offset.neg(),
            Some(composite(vec![
                item(LowLevelType::Char, -3),
                item(LowLevelType::Signed, -2),
            ]))
        );
    }

    #[test]
    fn composite_neg_fails_when_an_element_is_not_negatable() {
        // llmemory.py `[-item for item in self.offsets]` raises when an
        // element has no `__neg__` (FieldOffset here) — not silently dropped.
        let composite = composite(vec![
            item(LowLevelType::Signed, 2),
            field(LowLevelType::Signed, "f"),
        ]);
        assert_eq!(composite.neg(), None);
    }

    #[test]
    fn add_builds_composite_and_merges_when_compatible() {
        // llmemory.py `__add__ -> CompositeOffset(self, other)`.
        assert_eq!(
            item(LowLevelType::Signed, 2).add(item(LowLevelType::Signed, 5)),
            item(LowLevelType::Signed, 7)
        );
    }

    #[test]
    fn composite_known_nonneg_requires_all_parts() {
        // llmemory.py known_nonneg.
        assert!(
            composite(vec![
                item(LowLevelType::Signed, 1),
                field(LowLevelType::Signed, "f"),
            ])
            .known_nonneg()
        );
        assert!(
            !composite(vec![
                item(LowLevelType::Signed, 1),
                item(LowLevelType::Signed, -1),
            ])
            .known_nonneg()
        );
    }

    /// A layout source that knows no struct — used to exercise the
    /// layout-free offset kinds and the missing-layout error paths.
    struct NoLayout;
    impl OffsetLayout for NoLayout {
        fn field_offset(&self, _struct_name: &str, _fldname: &str) -> Option<i64> {
            None
        }
        fn struct_size(&self, _struct_name: &str) -> Option<i64> {
            None
        }
    }

    /// A layout source with one fixed answer for every query.
    struct FakeLayout {
        field: i64,
        size: i64,
    }
    impl OffsetLayout for FakeLayout {
        fn field_offset(&self, _struct_name: &str, _fldname: &str) -> Option<i64> {
            Some(self.field)
        }
        fn struct_size(&self, _struct_name: &str) -> Option<i64> {
            Some(self.size)
        }
    }

    fn struct_ty(name: &str) -> LowLevelType {
        use crate::translator::rtyper::lltypesystem::lltype::Struct;
        LowLevelType::Struct(Box::new(Struct::new(
            name,
            vec![("f".to_string(), LowLevelType::Signed)],
        )))
    }

    #[test]
    fn byte_size_resolves_primitives_and_sums_composites() {
        assert_eq!(item(LowLevelType::Signed, 3).byte_size(&NoLayout), Ok(24));
        assert_eq!(item(LowLevelType::Char, 4).byte_size(&NoLayout), Ok(4));
        assert_eq!(
            item(LowLevelType::UniChar, 1).byte_size(&NoLayout),
            Ok(sizeof_wchar_t())
        );
        let composite = composite(vec![
            item(LowLevelType::Signed, 1),
            item(LowLevelType::Char, 2),
        ]);
        assert_eq!(composite.byte_size(&NoLayout), Ok(10));
    }

    #[test]
    fn byte_size_resolves_array_tokens_without_layout() {
        // Standard length-prefixed array: items one word past the header,
        // length field at offset 0.
        assert_eq!(
            array_items(LowLevelType::Signed).byte_size(&NoLayout),
            Ok(WORD)
        );
        assert_eq!(
            array_length(LowLevelType::Signed).byte_size(&NoLayout),
            Ok(0)
        );
    }

    #[test]
    fn byte_size_field_and_struct_item_use_layout() {
        let s = struct_ty("S");
        let fo = field(s.clone(), "f");
        // No layout → get_field_token / get_size unavailable.
        assert!(fo.byte_size(&NoLayout).is_err());
        assert!(item(s.clone(), 2).byte_size(&NoLayout).is_err());
        // With a layout, FieldOffset reads the field offset and a struct
        // ItemOffset reads get_size * repeat.
        let layout = FakeLayout {
            field: 16,
            size: 40,
        };
        assert_eq!(fo.byte_size(&layout), Ok(16));
        assert_eq!(item(s, 2).byte_size(&layout), Ok(80));
    }

    #[test]
    fn byte_size_field_offset_on_non_struct_errors() {
        assert!(
            field(LowLevelType::Signed, "f")
                .byte_size(&NoLayout)
                .is_err()
        );
    }

    #[test]
    fn sizeof_primitive_returns_unit_item_offset() {
        // llmemory.py `sizeof(TYPE) -> ItemOffset(TYPE)`.
        assert_eq!(
            sizeof(&LowLevelType::Signed, None),
            Ok(ConstValue::AddressOffset(item(LowLevelType::Signed, 1)))
        );
    }

    #[test]
    fn sizeof_array_is_items_offset_plus_n_items() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, GcKind, frozendict};
        // llmemory.py `sizeof(ARRAY, n) -> itemoffsetof(ARRAY) +
        // sizeof(ARRAY.OF) * n`.
        let array_ty = LowLevelType::Array(Box::new(Array {
            OF: LowLevelType::Signed,
            _hints: frozendict::from(Vec::new()),
            _gckind: GcKind::Gc,
        }));
        let expected = composite(vec![
            array_items(array_ty.clone()),
            item(LowLevelType::Signed, 3),
        ]);
        assert_eq!(
            sizeof(&array_ty, Some(3)),
            Ok(ConstValue::AddressOffset(expected))
        );
    }

    fn gc_opaque(name: &str) -> _ptr {
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, opaqueptr};
        opaqueptr(LowLevelType::Opaque(Box::new(OpaqueType::gc(name))), "t").unwrap()
    }

    #[test]
    fn weakref_create_on_gc_target_yields_nonzero_gc_weakref() {
        // llmemory.py — `_wref(ptarget)._as_ptr()` for a gc target.
        let wref = weakref_create(&gc_opaque("GcThing")).unwrap();
        assert!(wref.nonzero());
        assert_eq!(wref._togckind(), GcKind::Gc);
    }

    #[test]
    fn weakref_create_rejects_null_target() {
        // llmemory.py `assert ptarget`.
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, nullptr};
        let null_gc = nullptr(LowLevelType::Opaque(Box::new(OpaqueType::gc("GcThing")))).unwrap();
        assert!(weakref_create(&null_gc).is_err());
    }

    #[test]
    fn dead_wref_is_a_single_shared_value() {
        // llmemory.py `dead_wref = _wref(None)._as_ptr()` — one prebuilt.
        let a = dead_wref();
        let b = dead_wref();
        assert!(a.nonzero());
        assert_eq!(a, b);
    }

    #[test]
    fn weakref_deref_recovers_the_created_target() {
        // llmemory.py — `weakref_deref(PTRTYPE, weakref_create(p))` is p.
        let target = gc_opaque("GcThing");
        let ptrtype = LowLevelType::Ptr(Box::new(target._TYPE.clone()));
        let wref = weakref_create(&target).unwrap();
        let got = weakref_deref(&ptrtype, &wref).unwrap();
        assert_eq!(got, target);
    }

    #[test]
    fn weakref_deref_of_dead_wref_is_null() {
        // llmemory.py:840-841 — a dead wref dereferences to `nullptr(PTRTYPE.TO)`.
        let ptrtype = LowLevelType::Ptr(Box::new(gc_opaque("GcThing")._TYPE.clone()));
        let got = weakref_deref(&ptrtype, &dead_wref()).unwrap();
        assert!(!got.nonzero());
    }

    fn array(of: LowLevelType, hints: Vec<(String, ConstValue)>) -> LowLevelType {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, GcKind, frozendict};
        LowLevelType::Array(Box::new(Array {
            OF: of,
            _hints: frozendict::from(hints),
            _gckind: GcKind::Gc,
        }))
    }

    #[test]
    fn sizeof_none_rejects_varsize_array() {
        // llmemory.py `_sizeof_none` asserts `not TYPE._is_varsize()`,
        // so an Array (always varsize) must be sized with an explicit n.
        assert!(sizeof(&array(LowLevelType::Signed, Vec::new()), None).is_err());
    }

    #[test]
    fn sizeof_array_adds_extra_item_after_alloc() {
        // llmemory.py `n += extra_item_after_alloc(TYPE)` — a STR-like
        // char array with the `extra_item_after_alloc=1` hint sizes n+1 items.
        let chars = array(
            LowLevelType::Char,
            vec![("extra_item_after_alloc".into(), ConstValue::Int(1))],
        );
        let expected = composite(vec![
            array_items(chars.clone()),
            item(LowLevelType::Char, 4),
        ]);
        assert_eq!(
            sizeof(&chars, Some(3)),
            Ok(ConstValue::AddressOffset(expected))
        );
    }

    #[test]
    fn array_items_offset_is_zero_for_nolength_array() {
        // symbolic.py:39-42 — a `nolength` array has no length prefix, so the
        // items begin at offset 0 instead of one word in.
        let nolength = array(
            LowLevelType::Signed,
            vec![("nolength".into(), ConstValue::Bool(true))],
        );
        assert_eq!(array_items(nolength).byte_size(&NoLayout), Ok(0));
        let plain = array(LowLevelType::Signed, Vec::new());
        assert_eq!(array_items(plain).byte_size(&NoLayout), Ok(WORD));
    }

    #[test]
    fn cast_any_ptr_concrete_to_opaque_hides_the_container() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            MallocFlavor, OpaqueType, Ptr, Struct, malloc,
        };
        // llmemory.py:1048 — a concrete gc referent cast to a gc opaque
        // (GCREF-style) PTRTYPE takes the `cast_opaque_ptr` concrete→opaque
        // branch and yields a non-null opaque pointer of the requested type.
        let st = LowLevelType::Struct(Box::new(Struct::gc_with_hints(
            "GcThing",
            vec![("x".into(), LowLevelType::Signed)],
            vec![],
        )));
        let target = malloc(st, None, MallocFlavor::Gc, false).unwrap();
        let wref = weakref_create(&target).unwrap();
        let gcref = LowLevelType::Ptr(Box::new(
            Ptr::from_container_type(LowLevelType::Opaque(Box::new(OpaqueType::gc("GCREF"))))
                .unwrap(),
        ));
        let got = weakref_deref(&gcref, &wref).unwrap();
        assert!(got.nonzero());
        assert_eq!(LowLevelType::Ptr(Box::new(got._TYPE.clone())), gcref);
    }

    #[test]
    fn array_length_offset_ref_reads_array_length() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelValue, MallocFlavor, malloc,
        };

        // `GcArray(Signed)` of length 3.
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(array_ty.clone(), Some(3), MallocFlavor::Gc, true).unwrap();

        // `ArrayLengthOffset(ARRAY).ref(arrayptr)` → a `_arraylenref` pointer
        // whose only item is the array length.
        let lenptr = array_length(array_ty).r#ref(&arrayptr).unwrap();
        let _ptr_obj::ArrayLenRef(lenref) = lenptr._obj().unwrap() else {
            panic!("ArrayLengthOffset::ref must yield an _arraylenref");
        };
        assert_eq!(lenref.getlength(), 1);
        assert_eq!(lenref.getitem(0), Some(LowLevelValue::Signed(3)));
    }

    #[test]
    fn array_items_offset_ref_rejects_mismatched_array_type() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, MallocFlavor, malloc};
        // `ArrayItemsOffset(A2).ref(arrayptr)` asserts `array_type_match(A1, A2)`
        // (llmemory.py) where `A1 = typeOf(arrayptr).TO`. A matching
        // element type folds; a mismatched one fails the assert (the over-fold
        // guard), so the fold declines instead of producing a wrong pointer.
        let signed_arr = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let arrayptr = malloc(signed_arr.clone(), Some(3), MallocFlavor::Gc, true).unwrap();
        assert!(array_items(signed_arr).r#ref(&arrayptr).is_ok());
        let float_arr = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Float)));
        assert!(array_items(float_arr).r#ref(&arrayptr).is_err());
    }

    #[test]
    fn array_item_type_match_accepts_gcref_placeholder_items() {
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, Ptr, PtrTarget};
        // llmemory.py: exact type equality, or `T2 == GCREF` and
        // `T1` is any lltype.Ptr.
        let concrete_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Opaque(OpaqueType::gc("Thing")),
        }));
        assert!(array_item_type_match(
            &LowLevelType::Signed,
            &LowLevelType::Signed
        ));
        assert!(array_item_type_match(&concrete_ptr, &GCREF));
        assert!(!array_item_type_match(&LowLevelType::Signed, &GCREF));
    }

    #[test]
    fn item_offset_ref_to_array_end_yields_endmarker() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, MallocFlavor, Struct, malloc,
        };

        // `GcArray(Struct('Item', ('x', Signed)))` of length 1 — an inlined
        // (non-gc) item struct, as required for an array of containers.
        let item_ty = LowLevelType::Struct(Box::new(Struct::new(
            "Item",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let array_ty = LowLevelType::Array(Box::new(Array::gc(item_ty.clone())));
        let arrayptr = malloc(array_ty.clone(), Some(1), MallocFlavor::Gc, true).unwrap();

        // `ArrayItemsOffset(ARRAY).ref` → item-0 struct pointer.
        let item0 = array_items(array_ty).r#ref(&arrayptr).unwrap();
        // `item0 + ItemOffset(Item, 1)` lands exactly at the array end (index
        // 1 == length 1) → the `_endmarker_struct` sentinel, typed `Ptr(Item)`.
        let endptr = item(item_ty.clone(), 1).r#ref(&item0).unwrap();
        let _ptr_obj::EndMarker(end) = endptr._obj().unwrap() else {
            panic!("an end-of-array reference must yield an _endmarker");
        };
        assert!(!end._was_freed());
        assert_eq!(LowLevelType::Struct(Box::new(end.TYPE.clone())), item_ty);
    }

    #[test]
    fn item_offset_ref_memoizes_end_marker_per_array() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, MallocFlavor, Struct, malloc,
        };

        // `_end_markers[parent]` (llmemory.py): two references exactly
        // to one array's end yield the same `_endmarker` container, so the two
        // fakeaddresses would compare equal.
        let item_ty = LowLevelType::Struct(Box::new(Struct::new(
            "Item",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let array_ty = LowLevelType::Array(Box::new(Array::gc(item_ty.clone())));
        let arrayptr = malloc(array_ty.clone(), Some(1), MallocFlavor::Gc, true).unwrap();

        let end_ptr = || {
            let item0 = array_items(array_ty.clone()).r#ref(&arrayptr).unwrap();
            item(item_ty.clone(), 1).r#ref(&item0).unwrap()
        };
        assert_eq!(end_ptr()._obj().unwrap(), end_ptr()._obj().unwrap());
    }

    #[test]
    fn sizeof_wchar_t_for_windows_is_2_else_4() {
        assert_eq!(sizeof_wchar_t_for("windows", "x86_64", "msvc"), 2);
        assert_eq!(sizeof_wchar_t_for("windows", "aarch64", "msvc"), 2);
        assert_eq!(sizeof_wchar_t_for("linux", "x86_64", "gnu"), 4);
        assert_eq!(sizeof_wchar_t_for("macos", "aarch64", ""), 4);
        assert_eq!(sizeof_wchar_t_for("unknown", "wasm32", ""), 4);
    }

    #[test]
    fn sizeof_longfloat_for_follows_c_abi_table() {
        // 8: MSVC; Apple aarch64; AAPCS32 arm; Android x86
        assert_eq!(sizeof_longfloat_for("windows", "x86_64", "msvc"), 8);
        assert_eq!(sizeof_longfloat_for("windows", "aarch64", "msvc"), 8);
        assert_eq!(sizeof_longfloat_for("macos", "aarch64", ""), 8);
        assert_eq!(sizeof_longfloat_for("ios", "aarch64", ""), 8);
        assert_eq!(sizeof_longfloat_for("linux", "arm", "gnu"), 8);
        assert_eq!(sizeof_longfloat_for("android", "x86", "gnu"), 8);
        // 12: SysV x86 (Linux, BSDs, MinGW)
        assert_eq!(sizeof_longfloat_for("linux", "x86", "gnu"), 12);
        assert_eq!(sizeof_longfloat_for("windows", "x86", "gnu"), 12);
        // 16: i386 Darwin; x86_64 non-msvc; aarch64 non-Apple/Windows; wasm; …
        assert_eq!(sizeof_longfloat_for("macos", "x86", ""), 16);
        assert_eq!(sizeof_longfloat_for("linux", "x86_64", "gnu"), 16);
        assert_eq!(sizeof_longfloat_for("windows", "x86_64", "gnu"), 16);
        assert_eq!(sizeof_longfloat_for("macos", "x86_64", ""), 16);
        assert_eq!(sizeof_longfloat_for("linux", "aarch64", "gnu"), 16);
        assert_eq!(sizeof_longfloat_for("unknown", "wasm32", ""), 16);
        assert_eq!(sizeof_longfloat_for("linux", "riscv64", "gnu"), 16);
        assert_eq!(sizeof_longfloat_for("linux", "powerpc64", "gnu"), 16);
        assert_eq!(sizeof_longfloat_for("linux", "s390x", "gnu"), 16);
    }

    #[test]
    fn sizeof_wchar_t_and_longfloat_match_target_helpers() {
        let os = crate::layout::target_os();
        let arch = crate::layout::target_arch();
        let env = crate::layout::target_env();
        assert_eq!(sizeof_wchar_t(), sizeof_wchar_t_for(os, arch, env));
        assert_eq!(sizeof_longfloat(), sizeof_longfloat_for(os, arch, env));
    }
}
