//! W_SetObject — Python `set` type.
//!
//! PyPy equivalent: pypy/objspace/std/setobject.py
//!
//! `setobject.py SetStrategy` is the dispatch. A fresh set is
//! `EmptySetStrategy` (`sstorage` is `erase(None)`). The first `add`
//! (`EmptySetStrategy.add`) installs `IntegerSetStrategy` for a plain int
//! (`is_plain_int1`), `BytesSetStrategy` for an exact `W_BytesObject`,
//! `AsciiSetStrategy` for an exact ASCII `W_UnicodeObject`,
//! `IdentitySetStrategy` when `W_TypeObject.compares_by_identity` holds,
//! and `ObjectSetStrategy` otherwise. `ObjectSetStrategy` stores `ObjectKey`
//! in an [`rordereddict`] and reuses the dict object strategy's hashing and
//! equality. `IdentitySetStrategy` stores the object itself
//! ([`IdentitySetKey`]), plus the `hash_w` digest from insertion.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::gc_hook::GCREF;
use crate::pyobject::*;
use pyre_macros::pyre_class;
use std::cell::UnsafeCell;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SET_TYPE: PyType = crate::pyobject::new_pytype_with_user_subclass_and_weakref(
    "set",
    &SET_USER_TYPE,
    std::mem::offset_of!(W_SetObject, lifeline),
);
/// `W_SetObjectUser` (`typedef.py` `_getusercls(W_SetObject)`).
pub static SET_USER_TYPE: PyType =
    crate::pyobject::new_user_pytype("set", &SET_TYPE, std::mem::offset_of!(W_SetObjectUser, map));
pub static FROZENSET_TYPE: PyType = crate::pyobject::new_pytype_with_user_subclass_and_weakref(
    "frozenset",
    &FROZENSET_USER_TYPE,
    std::mem::offset_of!(W_SetObject, lifeline),
);
/// `W_SetObjectUser` for `frozenset` (`typedef.py` `_getusercls`). The two
/// user typeptrs share one payload layout and one GC tid.
pub static FROZENSET_USER_TYPE: PyType = crate::pyobject::new_user_pytype(
    "frozenset",
    &FROZENSET_TYPE,
    std::mem::offset_of!(W_SetObjectUser, map),
);

/// setobject.py `W_SetIterObject`.  Unlike the old sequence-iterator
/// adapter this keeps the live set, so a size change is observed by next().
#[pyre_class("set_iterator", static_name = "SET_ITERATOR")]
pub struct W_SetIterObject {
    pub w_set: PyObjectRef,
    pub startlen: usize,
    /// Elements handed out so far, which `descr_reduce` needs and the slot
    /// cursor no longer counts once a hole opens.
    pub index: usize,
    /// Where the next element is read from.  `index` counted table positions
    /// while a delete renumbered them; a tombstoned table renumbers nothing,
    /// so the cursor is a slot and the two part company after a `discard`.
    pub slot: usize,
}

#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn w_set_iter_new(w_set: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_set = crate::gc_roots::pin_root(w_set);
    let startlen = unsafe { w_set_len(w_set) };
    W_SetIterObject::allocate_stable(W_SetIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_set,
        startlen,
        index: 0,
        slot: 0,
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_set_iterator(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && std::ptr::eq((*obj).ob_type, &SET_ITERATOR_TYPE)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_set(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SetIterObject)).w_set
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_set(obj: PyObjectRef, w_set: PyObjectRef) {
    (*(obj as *mut W_SetIterObject)).w_set = w_set;
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_startlen(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).startlen
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_startlen(obj: PyObjectRef, startlen: usize) {
    (*(obj as *mut W_SetIterObject)).startlen = startlen;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_index(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).index
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_index(obj: PyObjectRef, index: usize) {
    (*(obj as *mut W_SetIterObject)).index = index;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_slot(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).slot
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_slot(obj: PyObjectRef, slot: usize) {
    (*(obj as *mut W_SetIterObject)).slot = slot;
}

/// `setobject.py SetStrategy` class identity. A set-specific discriminant:
/// dict's `StrategyKind` names module, map, and kwargs strategies this file
/// does not have (`Ascii` is `AsciiSetStrategy`, not `UnicodeDictStrategy`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SetStrategyKind {
    Empty,
    Object,
    Bytes,
    Ascii,
    Int,
    Identity,
}

/// Failure modes of `ObjectSetStrategy.update`'s table merge.
pub enum SetUpdateError {
    /// An equality callback raised; its concrete exception is parked in the
    /// interpreter's pending dict-key error slot.
    Key(crate::dictmultiobject::DictKeyError),
    /// A callback changed one of the tables while it was being traversed.
    ChangedSize,
}

/// `setobject.py SetStrategy`. `EmptySetStrategy`, `BytesSetStrategy`,
/// `AsciiSetStrategy`, `IntegerSetStrategy`, `IdentitySetStrategy`, and
/// `ObjectSetStrategy` are the live kinds. `W_BaseSetObject` methods are
/// one-line `self.strategy.<op>(self, ...)` forwards: the public `w_set_*`
/// entry points take [`w_set_lock`] (or [`w_set_lock_pair`]), read `strategy`
/// under that same lock as `sstorage`, and call the method here.
/// [`w_set_object_storage`] is only valid for `SetStrategyKind::Object`.
pub trait SetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind;

    /// `SetStrategy.get_empty_storage`.
    fn get_empty_storage(&self) -> GCREF;

    /// `SetStrategy.length`. The published atomic `len` slot, the count
    /// [`w_set_len`] loads without the stripe.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`.
    unsafe fn length(&self, w_set: PyObjectRef) -> usize;

    /// `SetStrategy.add`. `key` already carries `space.hash_w`.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn add(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<(), SetUpdateError>;

    /// `SetStrategy.remove`. A removal that empties the live set calls
    /// `W_BaseSetObject.switch_to_empty_strategy` (`W_SetObject._discard_from_set`).
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn remove(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError>;

    /// `SetStrategy.has_key`.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    /// `key.hash` is the `space.hash_w` digest of `key.obj`.
    unsafe fn has_key(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError>;

    /// `SetStrategy.clear`.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn clear(&self, w_set: PyObjectRef);

    /// `SetStrategy.popitem`. `None` is the KeyError the caller raises.
    /// `AbstractUnwrappedSetStrategy.popitem` does not switch to empty.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn popitem(&self, w_set: PyObjectRef) -> Option<PyObjectRef>;

    /// `SetStrategy.get_storage_copy`, assigned onto `dst` the way
    /// `copy_real` / `EmptySetStrategy.update` install the erased copy.
    ///
    /// # Safety
    /// `src` and `dst` must point at valid `W_SetObject`s. Caller holds
    /// `w_set_lock_pair`.
    unsafe fn get_storage_copy(&self, src: PyObjectRef, dst: PyObjectRef);

    /// `SetStrategy.getkeys`.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn getkeys(&self, w_set: PyObjectRef) -> Vec<PyObjectRef>;

    /// Next live iteration slot at or after `from`
    /// (`EmptyIteratorImplementation.next_entry`,
    /// `IntegerIteratorImplementation`, the object table's `next_valid_slot`).
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn next_slot(&self, w_set: PyObjectRef, from: usize) -> Option<usize>;

    /// One `iterkeys_with_hash` step: the key stored at `slot`.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn key_at(
        &self,
        w_set: PyObjectRef,
        slot: usize,
    ) -> Option<crate::dictmultiobject::ObjectKey>;

    /// `num_ever_used_items` (`rordereddict.py` `_ll_dictnext`).
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn num_ever_used_items(&self, w_set: PyObjectRef) -> usize;

    /// Key half of `iterkeys_with_hash` at `index`. Null when the slot is dead.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn iterkey_at(&self, w_set: PyObjectRef, index: usize) -> *mut PyObject;

    /// Digest half of [`SetStrategy::iterkey_at`].
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn iterkey_hash_at(&self, w_set: PyObjectRef, index: usize) -> i64;

    /// Digests already stored on the elements, for `W_FrozensetObject.descr_hash`.
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
    unsafe fn stored_hashes(&self, w_set: PyObjectRef) -> Vec<i64>;

    /// `SetStrategy.update`.
    ///
    /// # Safety
    /// Both arguments must point at valid `W_SetObject`s. Caller holds
    /// `w_set_lock_pair`.
    unsafe fn update(&self, w_set: PyObjectRef, w_other: PyObjectRef)
    -> Result<(), SetUpdateError>;

    /// `SetStrategy.difference_update`.
    ///
    /// # Safety
    /// Both arguments must point at valid `W_SetObject`s. Caller holds
    /// `w_set_lock_pair`.
    unsafe fn difference_update(
        &self,
        w_set: PyObjectRef,
        w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError>;

    /// Element `PyObjectRef` slots. `EmptySetStrategy` and `IntegerSetStrategy`
    /// have none (`IntegerSetStrategy` keys are `i64`). `BytesSetStrategy`
    /// and `AsciiSetStrategy` visit the key block (`BytesKey` / `StrKey`).
    /// `IdentitySetStrategy` visits the object pointer on [`IdentitySetKey`].
    ///
    /// # Safety
    /// `w_set` must point at a valid `W_SetObject`.
    unsafe fn walk_gc_refs(&self, w_set: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef));
}

/// One-word strategy slot, the set-side [`crate::dictmultiobject::DictStrategyRef`].
///
/// `W_BaseSetObject.strategy` is one instance pointer. A `&dyn SetStrategy`
/// in the object would be a fat pointer, and the unit-struct singletons are
/// zero-sized, so this `#[repr(C)]` holder is what `strategy` stores.
#[repr(C)]
pub struct SetStrategyRef {
    /// Compared by field, the way `DictStrategyRef.kind` is, so a guard reads
    /// a `getfield` rather than a vtable call.
    pub kind: SetStrategyKind,
    pub imp: &'static dyn SetStrategy,
    /// `space.fromcache` singletons leave this null.
    pub owner: GCREF,
}

unsafe impl Sync for SetStrategyRef {}
unsafe impl Send for SetStrategyRef {}

impl std::ops::Deref for SetStrategyRef {
    type Target = dyn SetStrategy;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.imp
    }
}

/// `setobject.py ObjectSetStrategy`. The erased box is [`SetItemsStorage`]
/// (`newset` → `r_dict(eq_w, hash_w)`).
pub struct ObjectSetStrategy;

/// `setobject.py ObjectSetStrategy` process-wide singleton
/// (`space.fromcache(ObjectSetStrategy)`).
pub static OBJECT_SET_STRATEGY: ObjectSetStrategy = ObjectSetStrategy;

/// Holder `W_SetObject.strategy` points at.
pub static OBJECT_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Object,
    imp: &OBJECT_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py EmptySetStrategy`. `get_empty_storage` is `erase(None)`.
pub struct EmptySetStrategy;

/// `setobject.py EmptySetStrategy` process-wide singleton
/// (`space.fromcache(EmptySetStrategy)`).
pub static EMPTY_SET_STRATEGY: EmptySetStrategy = EmptySetStrategy;

/// Holder installed by `W_BaseSetObject.switch_to_empty_strategy`.
pub static EMPTY_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Empty,
    imp: &EMPTY_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py IntegerSetStrategy`. The erased box is [`IntSetStorage`]
/// (`erase({})` of plain ints). `is_correct_type` is `is_plain_int1`.
pub struct IntegerSetStrategy;

/// `setobject.py IntegerSetStrategy` process-wide singleton
/// (`space.fromcache(IntegerSetStrategy)`).
pub static INTEGER_SET_STRATEGY: IntegerSetStrategy = IntegerSetStrategy;

/// Holder installed by `EmptySetStrategy.add` for a plain int.
pub static INTEGER_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Int,
    imp: &INTEGER_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py BytesSetStrategy`. The erased box is [`BytesSetStorage`]
/// (`erase({})` of the `bytes` block). `is_correct_type` is
/// `type(w_key) is W_BytesObject`.
pub struct BytesSetStrategy;

/// `setobject.py BytesSetStrategy` process-wide singleton
/// (`space.fromcache(BytesSetStrategy)`).
pub static BYTES_SET_STRATEGY: BytesSetStrategy = BytesSetStrategy;

/// Holder installed by `EmptySetStrategy.add` for an exact `bytes`.
pub static BYTES_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Bytes,
    imp: &BYTES_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py AsciiSetStrategy`. The erased box is [`AsciiSetStorage`]
/// (`erase({})` of the utf8 text). `is_correct_type` is
/// `type(w_key) is W_UnicodeObject and w_key.is_ascii()`.
pub struct AsciiSetStrategy;

/// `setobject.py AsciiSetStrategy` process-wide singleton
/// (`space.fromcache(AsciiSetStrategy)`).
pub static ASCII_SET_STRATEGY: AsciiSetStrategy = AsciiSetStrategy;

/// Holder installed by `EmptySetStrategy.add` for an exact ASCII `str`.
pub static ASCII_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Ascii,
    imp: &ASCII_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py IdentitySetStrategy`. The erased box is [`IdentitySetStorage`]
/// (`erase({})` of the object itself). `is_correct_type` is
/// `W_TypeObject.compares_by_identity`.
pub struct IdentitySetStrategy;

/// `setobject.py IdentitySetStrategy` process-wide singleton
/// (`space.fromcache(IdentitySetStrategy)`).
pub static IDENTITY_SET_STRATEGY: IdentitySetStrategy = IdentitySetStrategy;

/// Holder installed by `EmptySetStrategy.add` when
/// `W_TypeObject.compares_by_identity` holds.
pub static IDENTITY_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Identity,
    imp: &IDENTITY_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// Python set object.
///
/// Layout: `[ob_header | sstorage | strategy | len | hash | set_id | content_gen | lifeline]`,
/// the `W_BaseSetObject` slots `sstorage` and `strategy` (`setobject.py`) plus
/// the atomic count, the frozenset hash cache, and the two words the integer
/// `add` hit guard reads. `sstorage` is the erased box;
/// [`SetItemsStorage`] (`ObjectSetStrategy.unerase`), [`IntSetStorage`]
/// (`IntegerSetStrategy.unerase`), [`BytesSetStorage`]
/// (`BytesSetStrategy.unerase`), [`AsciiSetStorage`]
/// (`AsciiSetStrategy.unerase`), or [`IdentitySetStorage`]
/// (`IdentitySetStrategy.unerase`).
#[repr(C)]
pub struct W_SetObject {
    pub ob_header: PyObject,
    /// `setobject.py W_BaseSetObject.sstorage`. The cast to [`GCREF`] is the
    /// erase; [`w_set_object_storage`] casts it back. Null is
    /// `EmptySetStrategy.get_empty_storage` (`erase(None)`).
    pub sstorage: GCREF,
    /// `setobject.py W_BaseSetObject.strategy`, one word. `w_set_new` stores
    /// [`EMPTY_SET_STRATEGY_REF`]; the first add stores
    /// [`INTEGER_SET_STRATEGY_REF`], [`BYTES_SET_STRATEGY_REF`],
    /// [`ASCII_SET_STRATEGY_REF`], [`IDENTITY_SET_STRATEGY_REF`], or
    /// [`OBJECT_SET_STRATEGY_REF`].
    pub strategy: &'static SetStrategyRef,
    /// Element count, read WITHOUT the stripe lock.
    ///
    /// `Objects/setobject.c set_len` answers `len(s)` as
    /// `FT_ATOMIC_LOAD_SSIZE_RELAXED(so->used)` — a relaxed atomic load and no
    /// critical section — so a reader is entitled to a value from either side
    /// of a concurrent mutation but never to a torn one.  The JIT's `len` fold
    /// lowers to exactly that load (`set_len_descr`), which is why the slot
    /// cannot be a plain `usize`: the mutators below write it while a compiled
    /// loop is reading, and only an atomic makes that pair defined.
    ///
    /// Same size and bit validity as `usize`, so the descriptor group keeps
    /// reading it as `Type::Int` at `offset_of!` — the shape
    /// `W_TupleObject.hash` already uses.
    pub len: AtomicUsize,
    /// setobject.py `W_FrozensetObject.hash = DEFAULT_HASH`.
    pub hash: i64,
    /// Identity assigned once by [`fresh_set_id`]. A nursery collection
    /// moves the object address; this word does not.
    pub set_id: usize,
    /// Bumped by [`W_SetObject::set_len_relaxed`] and [`set_write_barrier`]
    /// when membership or storage changes. A plain-int `add` that finds the
    /// key already stored does not reach either, so the word stays put.
    /// Relaxed, same as `len`.
    pub content_gen: AtomicUsize,
    /// `setobject.py W_BaseSetObject` `_lifeline_` methods. Exact `set` and
    /// `frozenset` instances store the lifeline here; a `_getusercls` subclass
    /// keeps this null and uses `MapdictWeakrefSupport`.
    pub lifeline: PyObjectRef,
}

impl W_SetObject {
    /// `FT_ATOMIC_LOAD_SSIZE_RELAXED(so->used)`.
    #[inline]
    pub fn len_relaxed(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// `FT_ATOMIC_STORE_SSIZE_RELAXED(so->used, n)`. Also bumps
    /// [`Self::content_gen`]: every length publish is a membership change.
    #[inline]
    pub fn set_len_relaxed(&self, n: usize) {
        self.len.store(n, Ordering::Relaxed);
        self.bump_content_gen();
    }

    /// Relaxed load of [`Self::content_gen`].
    #[inline]
    pub fn content_gen_relaxed(&self) -> usize {
        self.content_gen.load(Ordering::Relaxed)
    }

    /// Advance [`Self::content_gen`] by one. Load then store, not
    /// `fetch_add`: `len` already rtypes those two atomics. Under the stripe
    /// lock the pair is race-free for this set. Two racers that both store
    /// the same successor still move the word off the value a hit guard
    /// traced, so the guard fails closed.
    #[inline]
    fn bump_content_gen(&self) {
        let n = self.content_gen.load(Ordering::Relaxed);
        self.content_gen.store(n.wrapping_add(1), Ordering::Relaxed);
    }
}

/// The translated user-subclass layout selected by `typedef.py` `_getusercls`.
/// `W_SetObject` remains the base payload; `MapdictStorageMixin` contributes
/// its fields only to the generated user class. `set` and `frozenset`
/// subclass instances share this layout.
#[repr(C)]
pub struct W_SetObjectUser {
    pub base: W_SetObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_SetObjectUser, storage)
            == std::mem::offset_of!(W_SetObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// GC type id assigned to `W_SetObject` at JitDriver init time.
pub const W_SET_GC_TYPE_ID: u32 = 30;

/// GC-managed element table shared by `set` and `frozenset` bodies.
///
/// Keyed with [`ObjectKeyBuildHasher`](crate::dictmultiobject::ObjectKeyBuildHasher),
/// the same hasher `ObjectDictStorage` uses: `ObjectKey.hash` already *is*
/// `space.hash_w(obj)`, so the default `RandomState` would SipHash a digest
/// that is itself the hash. `rordereddict` feeds the cached integer straight
/// into the table; the multiply only spreads it into hashbrown's control bits.
pub type SetItemsStorage = crate::rordereddict::RDict<
    crate::dictmultiobject::ObjectKey,
    (),
    crate::dictmultiobject::ObjectKeyBuildHasher,
>;

/// `setobject.py IntegerSetStrategy.get_empty_dict` — `{}` of plain ints.
/// Keys are `i64` (`plain_int_w`); the table hash is [`IntKeyHash`]
/// (`ll_int_hash`), not a second digest of the key.
pub type IntSetStorage = crate::rordereddict::RDict<i64, (), crate::dictmultiobject::IntKeyHash>;

/// Runtime-assigned GC type id for the [`IntSetStorage`] entries array
/// (`GcArray` of `i64` keys and unit values; no `PyObjectRef`).
static INT_SET_ENTRIES_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IntSetStorage`] entries array.
pub fn set_int_set_entries_gc_type_id(id: u32) {
    INT_SET_ENTRIES_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IntSetStorage`] entries array.
#[majit_macros::dont_look_inside]
pub fn int_set_entries_gc_type_id() -> u32 {
    INT_SET_ENTRIES_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

impl crate::rordereddict::GcEntriesType for (i64, ()) {
    fn entries_gc_type_id() -> u32 {
        int_set_entries_gc_type_id()
    }
}

/// `setobject.py BytesSetStrategy.get_empty_dict` — `{}` of `bytes` blocks.
/// The key is [`BytesKey`](crate::dictmultiobject::BytesKey), the same block
/// `BytesDictStorage` stores; the table hash is [`BytesKeyHash`]
/// (`dictmultiobject.rs`), not `space.hash_w`.
pub type BytesSetStorage = crate::rordereddict::RDict<
    crate::dictmultiobject::BytesKey,
    (),
    crate::dictmultiobject::BytesKeyHash,
>;

/// Runtime-assigned GC type id for the [`BytesSetStorage`] entries array.
/// The registration traces the `BytesKey` block pointer
/// (`BytesKey::GC_REF_OFFSETS`).
static BYTES_SET_ENTRIES_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`BytesSetStorage`] entries array.
pub fn set_bytes_set_entries_gc_type_id(id: u32) {
    BYTES_SET_ENTRIES_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`BytesSetStorage`] entries array.
#[majit_macros::dont_look_inside]
pub fn bytes_set_entries_gc_type_id() -> u32 {
    BYTES_SET_ENTRIES_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

impl crate::rordereddict::GcEntriesType for (crate::dictmultiobject::BytesKey, ()) {
    fn entries_gc_type_id() -> u32 {
        bytes_set_entries_gc_type_id()
    }
}

/// `setobject.py AsciiSetStrategy.get_empty_dict` — `{}` of utf8 text.
/// The key is [`StrKey`](crate::celldict::StrKey) over an rstr `STR`
/// (`Utf8Str`); the table hash is [`StrKeyBuildHasher`]
/// (`celldict.rs` `ModuleDictEntries`), not `space.hash_w`.
pub type AsciiSetStorage = crate::rordereddict::RDict<
    crate::celldict::StrKey,
    (),
    crate::dictmultiobject::StrKeyBuildHasher,
>;

/// Runtime-assigned GC type id for the [`AsciiSetStorage`] entries array.
/// The registration traces the `StrKey` block pointer
/// (`StrKey::GC_REF_OFFSETS`).
static ASCII_SET_ENTRIES_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`AsciiSetStorage`] entries array.
pub fn set_ascii_set_entries_gc_type_id(id: u32) {
    ASCII_SET_ENTRIES_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`AsciiSetStorage`] entries array.
#[majit_macros::dont_look_inside]
pub fn ascii_set_entries_gc_type_id() -> u32 {
    ASCII_SET_ENTRIES_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

impl crate::rordereddict::GcEntriesType for (crate::celldict::StrKey, ()) {
    fn entries_gc_type_id() -> u32 {
        ascii_set_entries_gc_type_id()
    }
}

/// Identity-set key. Equality and the table hash are the object pointer,
/// the same contract as [`crate::identitydict::IdentityKey`]. `hash` is the
/// `hash_w` digest captured at insertion, the word [`crate::dictmultiobject::ObjectKey`]
/// already keeps, so [`SetStrategy::stored_hashes`] does not call `hash_w` again.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IdentitySetKey {
    pub obj: PyObjectRef,
    pub hash: i64,
}

impl std::hash::Hash for IdentitySetKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::gc_hook::gc_identity_hash(self.obj as usize).hash(state);
    }
}

impl PartialEq for IdentitySetKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.obj, other.obj)
    }
}

impl Eq for IdentitySetKey {}

impl crate::rordereddict::EntryDummy for IdentitySetKey {
    fn dummy() -> Self {
        Self {
            obj: std::ptr::null_mut(),
            hash: 0,
        }
    }
}

impl crate::rordereddict::GcRefOffsets for IdentitySetKey {
    const GC_REF_OFFSETS: &'static [usize] = &[std::mem::offset_of!(IdentitySetKey, obj)];
}

/// `setobject.py IdentitySetStrategy.get_empty_dict` — `{}` keyed by
/// object identity. The key is [`IdentitySetKey`]. The table hash is
/// `RandomState` over `gc_identity_hash`, the hasher `IdentityDictStorage`
/// uses. `(IdentityKey, PyObjectRef)` is the identity-dict entries id;
/// this table's value is `()`.
pub type IdentitySetStorage = crate::rordereddict::RDict<IdentitySetKey, ()>;

/// Runtime-assigned GC type id for the [`IdentitySetStorage`] entries array.
/// The registration traces the object pointer on [`IdentitySetKey`]
/// (`IdentitySetKey::GC_REF_OFFSETS`). The insertion digest is not a pointer.
static IDENTITY_SET_ENTRIES_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IdentitySetStorage`] entries array.
pub fn set_identity_set_entries_gc_type_id(id: u32) {
    IDENTITY_SET_ENTRIES_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IdentitySetStorage`] entries array.
#[majit_macros::dont_look_inside]
pub fn identity_set_entries_gc_type_id() -> u32 {
    IDENTITY_SET_ENTRIES_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

impl crate::rordereddict::GcEntriesType for (IdentitySetKey, ()) {
    fn entries_gc_type_id() -> u32 {
        identity_set_entries_gc_type_id()
    }
}

/// `setobject.py ObjectSetStrategy.unerase` — the erased `sstorage` word
/// cast back to [`SetItemsStorage`].
///
/// # Safety
/// `obj` must point to a valid `W_SetObject` whose strategy is
/// [`OBJECT_SET_STRATEGY`].
#[inline]
pub unsafe fn w_set_object_storage<'a>(obj: PyObjectRef) -> &'a SetItemsStorage {
    &*object_set_storage_ptr(&*(obj as *const W_SetObject))
}

/// Mutable [`w_set_object_storage`].
///
/// # Safety
/// Same as [`w_set_object_storage`].
#[inline]
pub unsafe fn w_set_object_storage_mut<'a>(obj: PyObjectRef) -> &'a mut SetItemsStorage {
    &mut *object_set_storage_ptr(&*(obj as *const W_SetObject))
}

/// Pointer form of [`w_set_object_storage`]. Debug-asserts `SetStrategyKind::Object`.
#[inline]
unsafe fn object_set_storage_ptr(set: &W_SetObject) -> *mut SetItemsStorage {
    debug_assert_eq!(set.strategy.kind, SetStrategyKind::Object);
    set.sstorage as *mut SetItemsStorage
}

/// True when `items` is still this set's live `ObjectSetStrategy` box.
///
/// `switch_to_empty_strategy` leaves `sstorage` null
/// (`EmptySetStrategy.get_empty_storage`). A probe that captured the old
/// box must not unerase the live word after that.
#[inline]
fn same_live_object_box(set: &W_SetObject, items: *mut SetItemsStorage) -> bool {
    set.strategy.kind == SetStrategyKind::Object && set.sstorage == items as GCREF
}

/// `BytesSetStrategy.is_correct_type` — `type(w_key) is W_BytesObject`.
#[inline]
unsafe fn is_exact_bytes_object(obj: PyObjectRef) -> bool {
    crate::is_exact_type(obj, &crate::BYTES_TYPE)
}

/// `AsciiSetStrategy.is_correct_type` — exact `W_UnicodeObject` and
/// `W_UnicodeObject.is_ascii`.
#[inline]
unsafe fn is_exact_ascii_str(obj: PyObjectRef) -> bool {
    crate::is_exact_type(obj, &crate::STR_TYPE) && crate::w_str_is_ascii(obj)
}

/// `setobject.py AbstractUnwrappedSetStrategy`.
///
/// One body for every unwrapped set. `IntegerSetStrategy`, `BytesSetStrategy`,
/// `AsciiSetStrategy`, and `IdentitySetStrategy` supply `unwrap` / `wrap` /
/// `is_correct_type` / `may_contain_equal_elements` and the erased storage
/// (`unerase` is [`AbstractUnwrappedSetStrategy::storage_ptr`], `get_empty_dict`
/// is the `RDict` behind [`AbstractUnwrappedSetStrategy::get_empty_storage`]).
pub trait AbstractUnwrappedSetStrategy: Sized {
    type Key: Copy
        + Eq
        + std::hash::Hash
        + crate::rordereddict::EntryDummy
        + crate::rordereddict::GcRefOffsets
        + 'static;
    type Hasher: std::hash::BuildHasher + Clone + Default + 'static;

    fn kind(&self) -> SetStrategyKind;
    fn strategy_ref(&self) -> &'static SetStrategyRef;
    fn storage_gc_type_id(&self) -> u32;

    /// `is_correct_type`.
    unsafe fn is_correct_type(&self, w_key: PyObjectRef) -> bool;
    /// `unwrap`.
    unsafe fn unwrap(&self, w_key: PyObjectRef) -> Self::Key;
    /// `unwrap`, keeping `hash` when the key stores an insertion digest.
    /// Int, bytes, and ascii ignore `hash`.
    unsafe fn unwrap_with_hash(&self, w_key: PyObjectRef, hash: i64) -> Self::Key {
        let _ = hash;
        unsafe { self.unwrap(w_key) }
    }
    /// The key stores the `hash_w` digest from insertion.
    /// Int, bytes, and ascii recompute; their digest is a function of the key.
    fn stores_insertion_hash(&self) -> bool {
        false
    }
    /// Insertion digest. Meaningful when [`Self::stores_insertion_hash`] is set.
    fn insertion_hash(&self, key: &Self::Key) -> i64 {
        let _ = key;
        0
    }
    /// Put `hash` back on a key rebuilt from a pinned object.
    fn key_with_insertion_hash(&self, key: Self::Key, hash: i64) -> Self::Key {
        let _ = hash;
        key
    }
    /// `wrap` (`newint` / `newbytes` / `newutf8`, or `IdentitySetStrategy.wrap`).
    unsafe fn wrap(&self, key: Self::Key) -> PyObjectRef;
    /// `may_contain_equal_elements`.
    fn may_contain_equal_elements(&self, other: SetStrategyKind) -> bool;

    /// `unerase`. The cast is the same for every unwrapped box.
    unsafe fn storage_ptr(
        &self,
        set: &W_SetObject,
    ) -> *mut crate::rordereddict::RDict<Self::Key, (), Self::Hasher> {
        debug_assert_eq!(set.strategy.kind, self.kind());
        set.sstorage as *mut crate::rordereddict::RDict<Self::Key, (), Self::Hasher>
    }

    /// `get_empty_storage` — `erase(get_empty_dict())`.
    fn get_empty_storage(&self) -> GCREF
    where
        (Self::Key, ()): crate::rordereddict::GcEntriesType,
    {
        crate::gc_storage::gc_alloc_storage_box(
            crate::rordereddict::RDict::<Self::Key, (), Self::Hasher>::new(),
            self.storage_gc_type_id(),
        ) as GCREF
    }

    /// True when `Key` is a GC block (`BytesKey`, `StrKey`, `IdentitySetKey`).
    /// `i64` is not.
    fn key_is_gc_ref(&self) -> bool {
        false
    }

    /// Root `key` when it is a GC block and return the forwarded key.
    /// A plain `i64` is unchanged. Pointer keys push exactly one shadow-stack
    /// slot (`pin_key`'s contract for [`snap_entries`]).
    fn pin_key(&self, key: Self::Key) -> Self::Key {
        key
    }

    /// Rebuild a GC key from the root [`pin_key`](Self::pin_key) published.
    fn key_from_pinned(&self, pinned: PyObjectRef) -> Self::Key {
        let _ = pinned;
        unreachable!("AbstractUnwrappedSetStrategy::key_from_pinned")
    }

    /// `walk_gc_refs` for one key. `i64` has nothing to visit.
    ///
    /// `iter_mut_for_trace` yields `&Key`. The visitor needs the key-block
    /// slot, so a pointer key casts that shared reference the way
    /// `BytesSetStrategy.walk_gc_refs` did.
    unsafe fn trace_key(&self, key: &Self::Key, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let _ = (key, visitor);
    }
}

/// `AsciiSetStrategy.wrap` — `space.newutf8` (`w_str_from_storage`).
///
/// `w_str_from_storage_and_length` builds the `W_UnicodeObject` before its
/// allocation, so a young `_utf8` forwarded by that allocation would be
/// stored stale. `w_bytes_from_block` pins the block and reloads it after
/// the body allocation; this does the same, then fills the fields
/// `w_str_from_storage_and_length` fills. ASCII means the code-point length
/// equals `len(_utf8)` (`w_str_from_storage`).
unsafe fn wrap_shared_utf8(block: *mut crate::unicodeobject::Utf8Str) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let block_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(block as PyObjectRef);
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(&STR_TYPE));
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        crate::W_UNICODE_GC_TYPE_ID,
        crate::W_UNICODE_OBJECT_SIZE,
    );
    let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut crate::unicodeobject::Utf8Str;
    let byte_len = if block.is_null() {
        0
    } else {
        unsafe { (*block).length }
    };
    let body = crate::W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        value: block,
        byte_len,
        len: byte_len,
        index_storage: std::ptr::null_mut(),
        hash: 0,
    };
    if raw.is_null() {
        crate::lltype::malloc_typed(body) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut crate::W_UnicodeObject, body) };
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    }
}

macro_rules! on_unwrapped {
    ($kind:expr, $s:ident => $body:expr) => {{
        match $kind {
            SetStrategyKind::Int => {
                let $s = &INTEGER_SET_STRATEGY;
                $body
            }
            SetStrategyKind::Bytes => {
                let $s = &BYTES_SET_STRATEGY;
                $body
            }
            SetStrategyKind::Ascii => {
                let $s = &ASCII_SET_STRATEGY;
                $body
            }
            SetStrategyKind::Identity => {
                let $s = &IDENTITY_SET_STRATEGY;
                $body
            }
            _ => unreachable!("AbstractUnwrappedSetStrategy"),
        }
    }};
}

struct KeySnap<K> {
    /// Parallel to `entry_slots`. `false` is a tombstone.
    live: Vec<bool>,
    /// Shadow-stack index of each live GC key, in slot order.
    pins: Vec<usize>,
    /// Live `i64` keys, in slot order. Empty when [`key_is_gc_ref`] is set.
    plain: Vec<K>,
    /// Insertion digests parallel to the live keys.
    /// Empty unless [`AbstractUnwrappedSetStrategy::stores_insertion_hash`].
    hashes: Vec<i64>,
    /// `RDict` probe-table length at the snapshot. Not a pointer.
    index_len: usize,
    nlive: usize,
}

unsafe fn key_at_slot<S: AbstractUnwrappedSetStrategy>(
    strategy: &S,
    set: &W_SetObject,
    slot: usize,
) -> Option<S::Key> {
    unsafe {
        (*strategy.storage_ptr(set))
            .get_slot(slot)
            .map(|(key, _)| *key)
    }
}

/// Snapshot live keys, rooting GC blocks first.
///
/// A nursery `STR` moves when `wrap` allocates. The entries array's trace
/// rewrites the table slot; a `Vec<StrKey>` copied earlier does not. GC keys
/// are therefore pinned, and later reads go through that pin. `i64` keys are
/// not pointers. `try_gc_alloc_stable_raw` is not used here, so this snapshot
/// is taken before any collecting `wrap`.
unsafe fn snap_entries<S: AbstractUnwrappedSetStrategy>(
    strategy: &S,
    set_slot: usize,
) -> KeySnap<S::Key> {
    let (n, index_len) = {
        let set = unsafe { &*(crate::gc_roots::shadow_stack_get(set_slot) as *const W_SetObject) };
        let storage = unsafe { &*strategy.storage_ptr(set) };
        (storage.entry_slots(), storage.index_len())
    };
    let mut live = vec![false; n];
    let mut pins = Vec::new();
    let mut plain = Vec::new();
    let mut hashes = Vec::new();
    for slot in 0..n {
        let set = unsafe { &*(crate::gc_roots::shadow_stack_get(set_slot) as *const W_SetObject) };
        let Some(key) = key_at_slot(strategy, set, slot) else {
            continue;
        };
        live[slot] = true;
        if strategy.stores_insertion_hash() {
            hashes.push(strategy.insertion_hash(&key));
        }
        if strategy.key_is_gc_ref() {
            let idx = crate::gc_roots::shadow_stack_len();
            let _ = strategy.pin_key(key);
            pins.push(idx);
        } else {
            plain.push(key);
        }
    }
    let nlive = if strategy.key_is_gc_ref() {
        pins.len()
    } else {
        plain.len()
    };
    KeySnap {
        live,
        pins,
        plain,
        hashes,
        index_len,
        nlive,
    }
}

unsafe fn snap_key<S: AbstractUnwrappedSetStrategy>(
    strategy: &S,
    snap: &KeySnap<S::Key>,
    live_index: usize,
) -> S::Key {
    let key = if strategy.key_is_gc_ref() {
        strategy.key_from_pinned(crate::gc_roots::shadow_stack_get(snap.pins[live_index]))
    } else {
        snap.plain[live_index]
    };
    if strategy.stores_insertion_hash() {
        strategy.key_with_insertion_hash(key, snap.hashes[live_index])
    } else {
        key
    }
}

unsafe fn publish_unwrapped_len<S>(strategy: &S, obj: PyObjectRef)
where
    S: AbstractUnwrappedSetStrategy,
{
    let set = unsafe { &mut *(obj as *mut W_SetObject) };
    if set.strategy.kind != strategy.kind() {
        return;
    }
    set.set_len_relaxed(unsafe { (*strategy.storage_ptr(set)).len() });
    set.hash = -1;
}

/// `W_SetObject._discard_from_set`: length 0 calls `switch_to_empty_strategy`.
unsafe fn publish_unwrapped_discard<S>(strategy: &S, obj: PyObjectRef)
where
    S: AbstractUnwrappedSetStrategy,
{
    let emptied = {
        let set = unsafe { &*(obj as *const W_SetObject) };
        set.strategy.kind == strategy.kind() && unsafe { (*strategy.storage_ptr(set)).is_empty() }
    };
    if emptied {
        unsafe { switch_to_empty_strategy(obj) };
    } else {
        publish_unwrapped_len(strategy, obj);
    }
}

/// `EmptySetStrategy.add` installs `strategy` and `get_empty_storage`.
///
/// Storage is published before the kind, so the word is never the new
/// strategy over null.
unsafe fn switch_empty_to<S>(strategy: &S, obj: PyObjectRef)
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let storage = AbstractUnwrappedSetStrategy::get_empty_storage(strategy);
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    {
        let set = unsafe { &mut *(obj as *mut W_SetObject) };
        set.sstorage = storage;
        set.strategy = strategy.strategy_ref();
    }
    set_write_barrier(obj);
}

/// `W_BaseSetObject.switch_to_object_strategy` for an unwrapped set.
///
/// `getdict_w` wraps each live key and `result[wrap(key)] = None` runs
/// `hash_w` before `switch_to_object_strategy` assigns the object
/// strategy. The slot image (live flags, probe-table length, pinned
/// keys) is taken first. The object table is built from that image
/// ([`crate::rordereddict::RDict::from_preserved_slots`]), and only then
/// are `sstorage` and the strategy published. A `hash_w` that `clear`s
/// the set replaces storage while the digests are still being computed;
/// the image is what gets installed, tombstones included, so slot numbers
/// stay those of the image. A `hash_w` that raises returns before that
/// publish, leaving the unwrapped strategy in place. The elements do not
/// change, so the frozenset hash cache is left alone.
unsafe fn unwrapped_switch_to_object<S>(
    strategy: &S,
    set_slot: usize,
) -> Result<(), crate::dictmultiobject::DictKeyError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let snap = snap_entries(strategy, set_slot);
    let n = snap.live.len();
    let index_len = snap.index_len;
    let mut hashes = Vec::with_capacity(snap.nlive);
    let wrap_base = crate::gc_roots::shadow_stack_len();
    for live_i in 0..snap.nlive {
        let key = snap_key(strategy, &snap, live_i);
        let wrapped = crate::gc_roots::pin_root(unsafe { strategy.wrap(key) });
        let keyed = unsafe { crate::dictmultiobject::object_key_for_checked(wrapped)? };
        hashes.push(keyed.hash);
    }
    let mut slot_keys = Vec::with_capacity(n);
    let mut live_i = 0usize;
    for slot in 0..n {
        if snap.live[slot] {
            slot_keys.push(crate::dictmultiobject::ObjectKey {
                hash: hashes[live_i],
                obj: crate::gc_roots::shadow_stack_get(wrap_base + live_i),
            });
            live_i += 1;
        } else {
            slot_keys.push(crate::dictmultiobject::ObjectKey {
                hash: 0,
                obj: std::ptr::null_mut(),
            });
        }
    }
    // Do not read `sstorage` here. `hash_w` above may have `clear`ed the set
    // and published a different box, or null.
    let mapped = crate::rordereddict::RDict::<
        crate::dictmultiobject::ObjectKey,
        (),
        crate::dictmultiobject::ObjectKeyBuildHasher,
    >::from_preserved_slots(&slot_keys, &snap.live, index_len);
    // `from_preserved_slots` pins the wrapped keys across `_ll_malloc_entries`.
    // The box malloc below is stable and does not collect; the keys stay on
    // the shadow stack until that box, which traces them, is installed.
    let storage = crate::gc_storage::gc_alloc_storage_box(mapped, set_items_gc_type_id());
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    {
        let set = unsafe { &mut *(obj as *mut W_SetObject) };
        let len = unsafe { (*storage).len() };
        set.sstorage = storage as GCREF;
        set.strategy = &OBJECT_SET_STRATEGY_REF;
        // `clear` during `hash_w` has already published length 0 on the empty
        // strategy. The installed table is the pre-callback image, so the
        // length has to come from that table. `hash` stays as it was: a
        // cached frozenset digest is left alone, and `clear` already stored
        // the uncomputed sentinel.
        set.set_len_relaxed(len);
    }
    set_write_barrier(obj);
    set_items_write_barrier(storage);
    Ok(())
}

/// `AbstractUnwrappedSetStrategy.add`.
unsafe fn unwrapped_add<S>(
    strategy: &S,
    w_set: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(key.obj);
    let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
    if unsafe { strategy.is_correct_type(key_obj) } {
        let unwrapped = unsafe { strategy.unwrap_with_hash(key_obj, key.hash) };
        let inserted = {
            let set =
                unsafe { &mut *(crate::gc_roots::shadow_stack_get(obj_slot) as *mut W_SetObject) };
            unsafe { (*strategy.storage_ptr(set)).insert(unwrapped, ()).is_none() }
        };
        if inserted {
            publish_unwrapped_len(strategy, crate::gc_roots::shadow_stack_get(obj_slot));
        }
        return Ok(());
    }
    // Wrong type: `switch_to_object_strategy`, then `w_set.add`.
    // `getdict_w` raises before the strategy word is published.
    unwrapped_switch_to_object(strategy, obj_slot).map_err(SetUpdateError::Key)?;
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::dictmultiobject::ObjectKey {
        hash: key.hash,
        obj: crate::gc_roots::shadow_stack_get(key_slot),
    };
    unsafe { (*(obj as *const W_SetObject)).strategy.add(obj, key) }
}

/// `AbstractUnwrappedSetStrategy.has_key`.
unsafe fn unwrapped_has_key<S>(
    strategy: &S,
    w_set: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(key.obj);
    let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
    if unsafe { strategy.is_correct_type(key_obj) } {
        let unwrapped = unsafe { strategy.unwrap(key_obj) };
        let set = unsafe { &*(crate::gc_roots::shadow_stack_get(obj_slot) as *const W_SetObject) };
        return Ok(unsafe { (*strategy.storage_ptr(set)).contains_key(&unwrapped) });
    }
    unwrapped_switch_to_object(strategy, obj_slot)?;
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::dictmultiobject::ObjectKey {
        hash: key.hash,
        obj: crate::gc_roots::shadow_stack_get(key_slot),
    };
    unsafe { (*(obj as *const W_SetObject)).strategy.has_key(obj, key) }
}

/// Delete one unwrapped key. `to_empty` is `W_SetObject._discard_from_set`.
/// `delitem_with_hash` passes `false` and leaves an empty dict in place.
unsafe fn unwrapped_delete<S>(
    strategy: &S,
    obj: PyObjectRef,
    w_key: PyObjectRef,
    to_empty: bool,
) -> bool
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let unwrapped = unsafe { strategy.unwrap(w_key) };
    let removed = {
        let set = unsafe { &mut *(obj as *mut W_SetObject) };
        unsafe { (*strategy.storage_ptr(set)).remove(&unwrapped).is_some() }
    };
    if removed {
        if to_empty {
            publish_unwrapped_discard(strategy, obj);
        } else {
            publish_unwrapped_len(strategy, obj);
        }
    }
    removed
}

/// `AbstractUnwrappedSetStrategy.remove`.
unsafe fn unwrapped_remove<S>(
    strategy: &S,
    w_set: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(key.obj);
    let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
    if unsafe { strategy.is_correct_type(key_obj) } {
        return Ok(unwrapped_delete(
            strategy,
            crate::gc_roots::shadow_stack_get(obj_slot),
            key_obj,
            true,
        ));
    }
    unwrapped_switch_to_object(strategy, obj_slot)?;
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::dictmultiobject::ObjectKey {
        hash: key.hash,
        obj: crate::gc_roots::shadow_stack_get(key_slot),
    };
    unsafe { (*(obj as *const W_SetObject)).strategy.remove(obj, key) }
}

/// `has_key` for `w_set_contains_key_for_update`. `None` means the set was
/// switched to object and the caller retries on that table. `delitem_with_hash`
/// is the remove twin (`to_empty` is false).
unsafe fn unwrapped_contains_or_switch<S>(
    strategy: &S,
    set_slot: usize,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<Option<bool>, crate::dictmultiobject::DictKeyError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    if unsafe { strategy.is_correct_type(key.obj) } {
        let unwrapped = unsafe { strategy.unwrap(key.obj) };
        let set = unsafe { &*(obj as *const W_SetObject) };
        Ok(Some(unsafe {
            (*strategy.storage_ptr(set)).contains_key(&unwrapped)
        }))
    } else {
        unwrapped_switch_to_object(strategy, set_slot)?;
        Ok(None)
    }
}

unsafe fn unwrapped_remove_or_switch<S>(
    strategy: &S,
    set_slot: usize,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<Option<()>, crate::dictmultiobject::DictKeyError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    if unsafe { strategy.is_correct_type(key.obj) } {
        unwrapped_delete(strategy, obj, key.obj, false);
        Ok(Some(()))
    } else {
        unwrapped_switch_to_object(strategy, set_slot)?;
        Ok(None)
    }
}

/// `AbstractUnwrappedSetStrategy.popitem`. The set stays on this strategy
/// when the dict underneath becomes empty.
unsafe fn unwrapped_popitem<S>(strategy: &S, w_set: PyObjectRef) -> Option<PyObjectRef>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let key = {
        let s = unsafe { &mut *(w_set as *mut W_SetObject) };
        let entries = unsafe { &mut *strategy.storage_ptr(s) };
        let (key, ()) = entries.pop()?;
        s.set_len_relaxed(s.len_relaxed() - 1);
        s.hash = -1;
        key
    };
    let _roots = crate::gc_roots::push_roots();
    let _ = crate::gc_roots::pin_root(w_set);
    // The popped block is no longer in the table. Pin it before `wrap`.
    let key = strategy.pin_key(key);
    Some(unsafe { strategy.wrap(key) })
}

/// `get_storage_copy` — `erase(d.copy())`.
///
/// `RDict::clone` reaches `_ll_malloc_entries` / `alloc_entries`, a
/// collection point. `src` and `dst` are nursery set objects; pin both
/// before the clone and write through the reloaded `dst`.
unsafe fn unwrapped_copy_storage<S>(strategy: &S, src: PyObjectRef, dst: PyObjectRef)
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::pin_roots(&[dst, src]);
    let src_slot = dst_slot + 1;
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let copied = unsafe { (*strategy.storage_ptr(&*(src as *const W_SetObject))).clone() };
    let len = copied.len();
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    {
        let d = unsafe { &mut *(dst as *mut W_SetObject) };
        d.sstorage =
            crate::gc_storage::gc_alloc_storage_box(copied, strategy.storage_gc_type_id()) as GCREF;
        d.strategy = strategy.strategy_ref();
        d.set_len_relaxed(len);
        d.hash = -1;
    }
    set_write_barrier(dst);
}

/// `getkeys` — `[wrap(key) for key in keys]`.
unsafe fn unwrapped_getkeys<S>(strategy: &S, w_set: PyObjectRef) -> Vec<PyObjectRef>
where
    S: AbstractUnwrappedSetStrategy,
{
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let snap = snap_entries(strategy, set_slot);
    let base = crate::gc_roots::shadow_stack_len();
    for live_i in 0..snap.nlive {
        let key = snap_key(strategy, &snap, live_i);
        let wrapped = unsafe { strategy.wrap(key) };
        let _ = crate::gc_roots::pin_root(wrapped);
    }
    (0..snap.nlive)
        .map(|i| crate::gc_roots::shadow_stack_get(base + i))
        .collect()
}

/// One `iterkeys_with_hash` step.
///
/// Int, bytes, and ascii take the digest from `hash_w` of `wrap(key)`.
/// An identity key already stores that digest, so this does not call `hash_w`.
unsafe fn unwrapped_key_object<S>(
    strategy: &S,
    w_set: PyObjectRef,
    slot: usize,
) -> Option<crate::dictmultiobject::ObjectKey>
where
    S: AbstractUnwrappedSetStrategy,
{
    let key = {
        let set = unsafe { &*(w_set as *const W_SetObject) };
        key_at_slot(strategy, set, slot)?
    };
    let _roots = crate::gc_roots::push_roots();
    let _ = crate::gc_roots::pin_root(w_set);
    if strategy.stores_insertion_hash() {
        let hash = strategy.insertion_hash(&key);
        let key = strategy.pin_key(key);
        let wrapped = crate::gc_roots::pin_root(unsafe { strategy.wrap(key) });
        return Some(crate::dictmultiobject::ObjectKey { hash, obj: wrapped });
    }
    let key = strategy.pin_key(key);
    let wrapped = crate::gc_roots::pin_root(unsafe { strategy.wrap(key) });
    Some(unsafe { crate::dictmultiobject::object_key_for(wrapped) })
}

unsafe fn unwrapped_stored_hashes<S>(strategy: &S, w_set: PyObjectRef) -> Vec<i64>
where
    S: AbstractUnwrappedSetStrategy,
{
    if strategy.stores_insertion_hash() {
        let set = unsafe { &*(w_set as *const W_SetObject) };
        let storage = unsafe { &*strategy.storage_ptr(set) };
        return storage
            .keys()
            .map(|key| strategy.insertion_hash(key))
            .collect();
    }
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let snap = snap_entries(strategy, set_slot);
    let mut hashes = Vec::with_capacity(snap.nlive);
    for live_i in 0..snap.nlive {
        let key = snap_key(strategy, &snap, live_i);
        let wrapped = crate::gc_roots::pin_root(unsafe { strategy.wrap(key) });
        let keyed = unsafe { crate::dictmultiobject::object_key_for(wrapped) };
        hashes.push(keyed.hash);
    }
    hashes
}

/// `AbstractUnwrappedSetStrategy.update` when both sets use this strategy:
/// `d_set.update(d_other)`.
unsafe fn unwrapped_update_same<S>(
    strategy: &S,
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    // Copied before any insert. `try_gc_alloc_stable_raw` does not collect,
    // so a `StrKey` / `BytesKey` in this `Vec` is still the block address
    // when `insert` stores it into the destination entries array.
    let keys = {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let storage = unsafe { strategy.storage_ptr(&*(src as *const W_SetObject)) };
        let mut keys = Vec::with_capacity(unsafe { (*storage).len() });
        for key in unsafe { (*storage).keys() } {
            keys.push(*key);
        }
        keys
    };
    let grew = {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let storage = unsafe { &mut *strategy.storage_ptr(&mut *(dst as *mut W_SetObject)) };
        let mut grew = false;
        for key in keys {
            if storage.insert(key, ()).is_none() {
                grew = true;
            }
        }
        grew
    };
    if grew {
        publish_unwrapped_len(strategy, crate::gc_roots::shadow_stack_get(dst_slot));
    }
    Ok(())
}

/// `_difference_update_unwrapped` for this strategy's key type.
unsafe fn unwrapped_difference_same<S>(
    strategy: &S,
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let dst_len = unsafe { w_set_len(crate::gc_roots::shadow_stack_get(dst_slot)) };
    let src_len = unsafe { w_set_len(crate::gc_roots::shadow_stack_get(src_slot)) };
    if dst_len < src_len {
        let mut keep = Vec::new();
        {
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let src = crate::gc_roots::shadow_stack_get(src_slot);
            let dst_storage = unsafe { strategy.storage_ptr(&*(dst as *const W_SetObject)) };
            let src_storage = unsafe { strategy.storage_ptr(&*(src as *const W_SetObject)) };
            let mut next = 0;
            while let Some(slot) = unsafe { (*dst_storage).next_valid_slot(next) } {
                let key = unsafe { *(*dst_storage).get_slot(slot).unwrap().0 };
                if unsafe { !(*src_storage).contains_key(&key) } {
                    keep.push(key);
                }
                next = slot + 1;
            }
        }
        let mut fresh = crate::rordereddict::RDict::<S::Key, (), S::Hasher>::new();
        for key in &keep {
            fresh.insert(*key, ());
        }
        let len = fresh.len();
        let storage = crate::gc_storage::gc_alloc_storage_box(fresh, strategy.storage_gc_type_id());
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        {
            let set = unsafe { &mut *(dst as *mut W_SetObject) };
            set.sstorage = storage as GCREF;
            set.strategy = strategy.strategy_ref();
            set.set_len_relaxed(len);
            set.hash = -1;
        }
        set_write_barrier(dst);
        return Ok(());
    }
    let keys = {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let src_storage = unsafe { strategy.storage_ptr(&*(src as *const W_SetObject)) };
        let mut keys = Vec::with_capacity(unsafe { (*src_storage).len() });
        for key in unsafe { (*src_storage).keys() } {
            keys.push(*key);
        }
        keys
    };
    let removed = {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let storage = unsafe { &mut *strategy.storage_ptr(&mut *(dst as *mut W_SetObject)) };
        let mut removed = false;
        for key in &keys {
            if storage.remove(key).is_some() {
                removed = true;
            }
        }
        removed
    };
    if removed {
        // `delitem_with_hash` does not call `switch_to_empty_strategy`.
        publish_unwrapped_len(strategy, crate::gc_roots::shadow_stack_get(dst_slot));
    }
    Ok(())
}

/// `_difference_wrapped`: keep unwrapped keys `w_other.has_key` misses.
unsafe fn unwrapped_difference_keep_missing<S>(
    strategy: &S,
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let snap = snap_entries(strategy, dst_slot);
    let mut keep_at = Vec::new();
    for live_i in 0..snap.nlive {
        let key = snap_key(strategy, &snap, live_i);
        let wrapped = crate::gc_roots::pin_root(unsafe { strategy.wrap(key) });
        let keyed = unsafe { crate::dictmultiobject::object_key_for(wrapped) };
        let _key_roots = crate::gc_roots::push_roots();
        let key_obj = crate::gc_roots::pin_root(keyed.obj);
        let present = w_set_contains_key_for_update(
            crate::gc_roots::shadow_stack_get(src_slot),
            crate::dictmultiobject::ObjectKey {
                hash: keyed.hash,
                obj: key_obj,
            },
        )?;
        if !present {
            keep_at.push(live_i);
        }
    }
    let mut fresh = crate::rordereddict::RDict::<S::Key, (), S::Hasher>::new();
    for live_i in keep_at {
        // Re-read the pin. A probe above may have moved a nursery block.
        let key = snap_key(strategy, &snap, live_i);
        fresh.insert(key, ());
    }
    let len = fresh.len();
    let storage = crate::gc_storage::gc_alloc_storage_box(fresh, strategy.storage_gc_type_id());
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    {
        let set = unsafe { &mut *(dst as *mut W_SetObject) };
        set.sstorage = storage as GCREF;
        set.strategy = strategy.strategy_ref();
        set.set_len_relaxed(len);
        set.hash = -1;
    }
    set_write_barrier(dst);
    Ok(())
}

/// `AbstractUnwrappedSetStrategy.update`.
unsafe fn unwrapped_update<S>(
    strategy: &S,
    w_set: PyObjectRef,
    w_other: PyObjectRef,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let src_kind = unsafe { (*(w_other as *const W_SetObject)).strategy.kind };
    if src_kind == SetStrategyKind::Empty {
        return Ok(());
    }
    if same_erased_storage(w_set, w_other) {
        return Ok(());
    }
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_other);
    if src_kind == strategy.kind() {
        return unwrapped_update_same(strategy, dst_slot, src_slot);
    }
    // A different strategy switches to object and retries. The kind is read
    // again because `switch_to_object_strategy` can run user code.
    unwrapped_switch_to_object(strategy, dst_slot).map_err(SetUpdateError::Key)?;
    update_object_from_other(dst_slot, src_slot)
}

/// `AbstractUnwrappedSetStrategy.difference_update`.
unsafe fn unwrapped_difference_update<S>(
    strategy: &S,
    w_set: PyObjectRef,
    w_other: PyObjectRef,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    let src_kind = unsafe { (*(w_other as *const W_SetObject)).strategy.kind };
    if src_kind == SetStrategyKind::Empty {
        return Ok(());
    }
    if same_erased_storage(w_set, w_other) {
        unsafe { w_set_clear(w_set) };
        set_write_barrier(w_set);
        return Ok(());
    }
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_set);
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_other);
    if src_kind == strategy.kind() {
        return unwrapped_difference_same(strategy, dst_slot, src_slot);
    }
    if !strategy.may_contain_equal_elements(src_kind) {
        return Ok(());
    }
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    if strategy.length_of(dst) < unsafe { (*(src as *const W_SetObject)).strategy.length(src) } {
        return unwrapped_difference_keep_missing(strategy, dst_slot, src_slot);
    }
    // A larger int self walks the object table (`difference_update_object_storage`).
    // Bytes, ascii, and identity walk wrapped keys (`_difference_update_wrapped`).
    if strategy.kind() == SetStrategyKind::Int {
        difference_update_object_storage(dst_slot, src_slot)
    } else {
        difference_remove_src_keys(dst_slot, src_slot)
    }
}

trait UnwrappedLen {
    unsafe fn length_of(&self, w_set: PyObjectRef) -> usize;
}

impl<S> UnwrappedLen for S
where
    S: AbstractUnwrappedSetStrategy,
{
    unsafe fn length_of(&self, w_set: PyObjectRef) -> usize {
        unsafe { (*(w_set as *const W_SetObject)).len_relaxed() }
    }
}

unsafe fn unwrapped_walk_gc_refs<S>(
    strategy: &S,
    w_set: PyObjectRef,
    visitor: &mut dyn FnMut(*mut PyObjectRef),
) where
    S: AbstractUnwrappedSetStrategy,
{
    let set = unsafe { &mut *(w_set as *mut W_SetObject) };
    if set.sstorage.is_null() {
        return;
    }
    let entries = unsafe { &mut *strategy.storage_ptr(set) };
    if strategy.key_is_gc_ref() {
        for (key, _) in entries.iter_mut_for_trace() {
            unsafe { strategy.trace_key(key, visitor) };
        }
        visitor(entries.entries_slot() as *mut PyObjectRef);
    }
    entries.visit_indexes(&mut |slot| visitor(slot as *mut PyObjectRef));
}

/// `ObjectSetStrategy.update` for an unwrapped operand: iterate `wrap` keys
/// into the object table. Does not switch this strategy.
unsafe fn object_update_from_unwrapped<S>(
    strategy: &S,
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError>
where
    S: AbstractUnwrappedSetStrategy,
{
    let snap = snap_entries(strategy, src_slot);
    for live_i in 0..snap.nlive {
        let key = snap_key(strategy, &snap, live_i);
        let wrapped = crate::gc_roots::pin_root(unsafe { strategy.wrap(key) });
        let keyed = unsafe { crate::dictmultiobject::object_key_for(wrapped) };
        let _key_roots = crate::gc_roots::push_roots();
        let key_obj = crate::gc_roots::pin_root(keyed.obj);
        w_set_insert_key_checked(
            crate::gc_roots::shadow_stack_get(dst_slot),
            crate::dictmultiobject::ObjectKey {
                hash: keyed.hash,
                obj: key_obj,
            },
        )?;
    }
    Ok(())
}

/// `ObjectSetStrategy.update` after `self` is on the object strategy.
/// Reads `src`'s kind again: `switch_to_object_strategy` can run user code.
unsafe fn update_object_from_other(dst_slot: usize, src_slot: usize) -> Result<(), SetUpdateError> {
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let kind = unsafe { (*(src as *const W_SetObject)).strategy.kind };
    match kind {
        SetStrategyKind::Empty => Ok(()),
        SetStrategyKind::Int
        | SetStrategyKind::Bytes
        | SetStrategyKind::Ascii
        | SetStrategyKind::Identity => {
            on_unwrapped!(kind, strategy => object_update_from_unwrapped(strategy, dst_slot, src_slot))
        }
        SetStrategyKind::Object => object_set_merge_captured(dst_slot, src),
    }
}

impl AbstractUnwrappedSetStrategy for IntegerSetStrategy {
    type Key = i64;
    type Hasher = crate::dictmultiobject::IntKeyHash;

    fn kind(&self) -> SetStrategyKind {
        SetStrategyKind::Int
    }
    fn strategy_ref(&self) -> &'static SetStrategyRef {
        &INTEGER_SET_STRATEGY_REF
    }
    fn storage_gc_type_id(&self) -> u32 {
        int_set_storage_gc_type_id()
    }
    unsafe fn is_correct_type(&self, w_key: PyObjectRef) -> bool {
        unsafe { crate::listobject::is_plain_int1(w_key) }
    }
    unsafe fn unwrap(&self, w_key: PyObjectRef) -> i64 {
        unsafe { crate::listobject::plain_int_w(w_key) }
    }
    unsafe fn wrap(&self, key: i64) -> PyObjectRef {
        crate::w_int_new(key)
    }
    fn may_contain_equal_elements(&self, other: SetStrategyKind) -> bool {
        // `IntegerSetStrategy.may_contain_equal_elements`.
        !matches!(
            other,
            SetStrategyKind::Bytes
                | SetStrategyKind::Ascii
                | SetStrategyKind::Empty
                | SetStrategyKind::Identity
        )
    }
}

impl AbstractUnwrappedSetStrategy for BytesSetStrategy {
    type Key = crate::dictmultiobject::BytesKey;
    type Hasher = crate::dictmultiobject::BytesKeyHash;

    fn kind(&self) -> SetStrategyKind {
        SetStrategyKind::Bytes
    }
    fn strategy_ref(&self) -> &'static SetStrategyRef {
        &BYTES_SET_STRATEGY_REF
    }
    fn storage_gc_type_id(&self) -> u32 {
        bytes_set_storage_gc_type_id()
    }
    unsafe fn is_correct_type(&self, w_key: PyObjectRef) -> bool {
        unsafe { is_exact_bytes_object(w_key) }
    }
    /// `BytesSetStrategy.unwrap` — the `bytes` block (`space.bytes_w`).
    unsafe fn unwrap(&self, w_key: PyObjectRef) -> Self::Key {
        crate::dictmultiobject::BytesKey(unsafe {
            crate::w_bytes_block(w_key) as *mut crate::bytesobject::BytesBlock
        })
    }
    /// `BytesSetStrategy.wrap` — `space.newbytes` (`w_bytes_from_block`).
    unsafe fn wrap(&self, key: Self::Key) -> PyObjectRef {
        crate::w_bytes_from_block(key.0)
    }
    fn may_contain_equal_elements(&self, other: SetStrategyKind) -> bool {
        // `BytesSetStrategy.may_contain_equal_elements`.
        !matches!(
            other,
            SetStrategyKind::Int | SetStrategyKind::Empty | SetStrategyKind::Identity
        )
    }
    fn key_is_gc_ref(&self) -> bool {
        true
    }
    fn pin_key(&self, key: Self::Key) -> Self::Key {
        crate::dictmultiobject::BytesKey(
            crate::gc_roots::pin_root(key.0 as PyObjectRef) as *mut crate::bytesobject::BytesBlock
        )
    }
    fn key_from_pinned(&self, pinned: PyObjectRef) -> Self::Key {
        crate::dictmultiobject::BytesKey(pinned as *mut crate::bytesobject::BytesBlock)
    }
    unsafe fn trace_key(&self, key: &Self::Key, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let key_ptr = key as *const Self::Key as *mut Self::Key;
        visitor(std::ptr::addr_of_mut!((*key_ptr).0) as *mut PyObjectRef);
    }
}

impl AbstractUnwrappedSetStrategy for AsciiSetStrategy {
    type Key = crate::celldict::StrKey;
    type Hasher = crate::dictmultiobject::StrKeyBuildHasher;

    fn kind(&self) -> SetStrategyKind {
        SetStrategyKind::Ascii
    }
    fn strategy_ref(&self) -> &'static SetStrategyRef {
        &ASCII_SET_STRATEGY_REF
    }
    fn storage_gc_type_id(&self) -> u32 {
        ascii_set_storage_gc_type_id()
    }
    unsafe fn is_correct_type(&self, w_key: PyObjectRef) -> bool {
        unsafe { is_exact_ascii_str(w_key) }
    }
    /// `AsciiSetStrategy.unwrap` — `space.utf8_w`, the str's `_utf8`.
    ///
    /// `module_dict_key_block` stores an interned str's block instead of
    /// copying the characters. The set key is that same word on the caller
    /// (`W_UnicodeObject.value`). The entries array traces `StrKey`, so a
    /// later nursery move rewrites the slot; an immortal block does not move
    /// or get swept. `insert` and `pop` allocate entries with
    /// `try_gc_alloc_stable_raw`, which does not collect, so the address is
    /// still the block when the slot is written. `pin_key` reloads it before
    /// `wrap`.
    unsafe fn unwrap(&self, w_key: PyObjectRef) -> Self::Key {
        crate::celldict::StrKey(unsafe {
            (*(w_key as *const crate::unicodeobject::W_UnicodeObject)).value
        })
    }
    unsafe fn wrap(&self, key: Self::Key) -> PyObjectRef {
        wrap_shared_utf8(key.0)
    }
    fn may_contain_equal_elements(&self, other: SetStrategyKind) -> bool {
        // `AsciiSetStrategy.may_contain_equal_elements` — the bytes exclusions.
        !matches!(
            other,
            SetStrategyKind::Int | SetStrategyKind::Empty | SetStrategyKind::Identity
        )
    }
    fn key_is_gc_ref(&self) -> bool {
        true
    }
    fn pin_key(&self, key: Self::Key) -> Self::Key {
        crate::celldict::StrKey(
            crate::gc_roots::pin_root(key.0 as PyObjectRef) as *mut crate::unicodeobject::Utf8Str
        )
    }
    fn key_from_pinned(&self, pinned: PyObjectRef) -> Self::Key {
        crate::celldict::StrKey(pinned as *mut crate::unicodeobject::Utf8Str)
    }
    unsafe fn trace_key(&self, key: &Self::Key, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let key_ptr = key as *const Self::Key as *mut Self::Key;
        visitor(std::ptr::addr_of_mut!((*key_ptr).0) as *mut PyObjectRef);
    }
}

impl AbstractUnwrappedSetStrategy for IdentitySetStrategy {
    type Key = IdentitySetKey;
    type Hasher = std::collections::hash_map::RandomState;

    fn kind(&self) -> SetStrategyKind {
        SetStrategyKind::Identity
    }
    fn strategy_ref(&self) -> &'static SetStrategyRef {
        &IDENTITY_SET_STRATEGY_REF
    }
    fn storage_gc_type_id(&self) -> u32 {
        identity_set_storage_gc_type_id()
    }
    /// `IdentitySetStrategy.is_correct_type` —
    /// `space.type(w_key).compares_by_identity()`. Same resolution as
    /// `IdentityDictStrategy.is_correct_type` /
    /// `EmptyDictStrategy.switch_to_correct_strategy`.
    unsafe fn is_correct_type(&self, w_key: PyObjectRef) -> bool {
        crate::dictmultiobject::key_compares_by_identity(w_key)
    }
    /// `IdentitySetStrategy.unwrap` — the object itself. Lookup keys do not
    /// carry a digest; [`Self::unwrap_with_hash`] is what `add` stores.
    unsafe fn unwrap(&self, w_key: PyObjectRef) -> Self::Key {
        IdentitySetKey {
            obj: w_key,
            hash: 0,
        }
    }
    unsafe fn unwrap_with_hash(&self, w_key: PyObjectRef, hash: i64) -> Self::Key {
        IdentitySetKey { obj: w_key, hash }
    }
    fn stores_insertion_hash(&self) -> bool {
        true
    }
    fn insertion_hash(&self, key: &Self::Key) -> i64 {
        key.hash
    }
    fn key_with_insertion_hash(&self, mut key: Self::Key, hash: i64) -> Self::Key {
        key.hash = hash;
        key
    }
    /// `IdentitySetStrategy.wrap`. `IdentityIteratorImplementation.next_entry`
    /// returns `w_key`.
    unsafe fn wrap(&self, key: Self::Key) -> PyObjectRef {
        key.obj
    }
    fn may_contain_equal_elements(&self, other: SetStrategyKind) -> bool {
        // `IdentitySetStrategy.may_contain_equal_elements`.
        !matches!(
            other,
            SetStrategyKind::Empty
                | SetStrategyKind::Int
                | SetStrategyKind::Bytes
                | SetStrategyKind::Ascii
        )
    }
    fn key_is_gc_ref(&self) -> bool {
        true
    }
    fn pin_key(&self, key: Self::Key) -> Self::Key {
        IdentitySetKey {
            obj: crate::gc_roots::pin_root(key.obj),
            hash: key.hash,
        }
    }
    fn key_from_pinned(&self, pinned: PyObjectRef) -> Self::Key {
        IdentitySetKey {
            obj: pinned,
            hash: 0,
        }
    }
    unsafe fn trace_key(&self, key: &Self::Key, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let key_ptr = key as *const Self::Key as *mut Self::Key;
        visitor(std::ptr::addr_of_mut!((*key_ptr).obj));
    }
}

impl<S> SetStrategy for S
where
    S: AbstractUnwrappedSetStrategy,
    (S::Key, ()): crate::rordereddict::GcEntriesType,
{
    fn strategy_kind(&self) -> SetStrategyKind {
        self.kind()
    }

    fn get_empty_storage(&self) -> GCREF {
        AbstractUnwrappedSetStrategy::get_empty_storage(self)
    }

    unsafe fn length(&self, w_set: PyObjectRef) -> usize {
        unsafe { (*(w_set as *const W_SetObject)).len_relaxed() }
    }

    unsafe fn add(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<(), SetUpdateError> {
        unwrapped_add(self, w_set, key)
    }

    unsafe fn remove(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError> {
        unwrapped_remove(self, w_set, key)
    }

    unsafe fn has_key(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError> {
        unwrapped_has_key(self, w_set, key)
    }

    unsafe fn clear(&self, w_set: PyObjectRef) {
        // `AbstractUnwrappedSetStrategy.clear` → `switch_to_empty_strategy`.
        unsafe { switch_to_empty_strategy(w_set) };
    }

    unsafe fn popitem(&self, w_set: PyObjectRef) -> Option<PyObjectRef> {
        unwrapped_popitem(self, w_set)
    }

    unsafe fn get_storage_copy(&self, src: PyObjectRef, dst: PyObjectRef) {
        unwrapped_copy_storage(self, src, dst);
    }

    unsafe fn getkeys(&self, w_set: PyObjectRef) -> Vec<PyObjectRef> {
        unwrapped_getkeys(self, w_set)
    }

    unsafe fn next_slot(&self, w_set: PyObjectRef, from: usize) -> Option<usize> {
        let set = unsafe { &*(w_set as *const W_SetObject) };
        unsafe { (*self.storage_ptr(set)).next_valid_slot(from) }
    }

    unsafe fn key_at(
        &self,
        w_set: PyObjectRef,
        slot: usize,
    ) -> Option<crate::dictmultiobject::ObjectKey> {
        unwrapped_key_object(self, w_set, slot)
    }

    unsafe fn num_ever_used_items(&self, w_set: PyObjectRef) -> usize {
        let set = unsafe { &*(w_set as *const W_SetObject) };
        unsafe { (*self.storage_ptr(set)).entry_slots() }
    }

    unsafe fn iterkey_at(&self, w_set: PyObjectRef, index: usize) -> *mut PyObject {
        match unwrapped_key_object(self, w_set, index) {
            Some(key) => key.obj,
            None => std::ptr::null_mut(),
        }
    }

    unsafe fn iterkey_hash_at(&self, w_set: PyObjectRef, index: usize) -> i64 {
        match unwrapped_key_object(self, w_set, index) {
            Some(key) => key.hash,
            None => 0,
        }
    }

    unsafe fn stored_hashes(&self, w_set: PyObjectRef) -> Vec<i64> {
        unwrapped_stored_hashes(self, w_set)
    }

    unsafe fn update(
        &self,
        w_set: PyObjectRef,
        w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError> {
        unwrapped_update(self, w_set, w_other)
    }

    unsafe fn difference_update(
        &self,
        w_set: PyObjectRef,
        w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError> {
        unwrapped_difference_update(self, w_set, w_other)
    }

    unsafe fn walk_gc_refs(&self, w_set: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        unwrapped_walk_gc_refs(self, w_set, visitor);
    }
}

/// Remove the entry occupying `slot` in O(1).
///
/// `_ll_dict_del_entry` marks the index slot [`DELETED`] and clears the entry
/// in place, so every surviving key keeps both its slot number and its
/// position in the walk order.
///
/// [`DELETED`]: crate::rordereddict::DELETED
unsafe fn set_remove_slot(items: *mut SetItemsStorage, slot: usize) {
    let _ = (*items).remove_slot(slot);
}

/// The slot holding the first element at or after `from`, or `None` past the
/// last one.
///
/// A walk counts in slots, not in positions: `_ll_dict_del_entry` leaves a
/// hole where it deleted, so slot numbers outlive their neighbours' removal
/// but stop being contiguous.  A caller re-reads its key with [`w_set_key_at`]
/// at the slot this hands back and resumes from `slot + 1`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_next_slot(obj: PyObjectRef, from: usize) -> Option<usize> {
    let _set_guard = w_set_lock(obj);
    let s = &*(obj as *const W_SetObject);
    // `W_BaseSetObject` iteration reads through `self.strategy`.
    s.strategy.next_slot(obj, from)
}

// PyPy serializes set strategy/storage operations with the GIL. Pyre is
// free-threaded, so use the same narrow address-striped reentrant-lock
// adaptation as listobject and dictmultiobject. Semantic state remains on
// W_SetObject; the stripes only restore the indivisible operation boundary
// supplied by PyPy's GIL. Stripes are reset in a fork child because a vanished
// thread may have held one at fork time.
struct ForkSetLock(UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkSetLock {}

impl ForkSetLock {
    fn new() -> Self {
        Self(UnsafeCell::new(parking_lot::ReentrantMutex::new(())))
    }

    fn get(&self) -> &parking_lot::ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(parking_lot::ReentrantMutex::new(())) };
    }
}

static SET_LOCKS: LazyLock<Vec<ForkSetLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkSetLock::new()).collect());

/// The set stripe is its lock word. `repr(transparent)` so a residual call
/// returns that integer and the guard's drop releases it, the same bracket
/// `ListGuard` uses for `rthread.py` `Lock.acquire` / `Lock.release`.
#[repr(transparent)]
struct SetGuard {
    lock: usize,
    not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for SetGuard {
    fn drop(&mut self) {
        // SAFETY: only the acquire helpers below construct this owner, from
        // one successful acquisition of a process-lifetime SET_LOCKS entry.
        unsafe { w_set_lock_release(self.lock) };
    }
}

#[inline]
fn set_lock_index(obj: PyObjectRef) -> usize {
    (obj as usize >> 4) & (SET_LOCKS.len() - 1)
}

/// Acquire a set's stripe without letting a contending mutator prevent a GC
/// stop-the-world. Only the acquire is opaque; the guarded set operation stays
/// visible to source translation, matching listobject/dictmultiobject.
#[majit_macros::dont_look_inside]
unsafe fn w_set_lock(obj: PyObjectRef) -> SetGuard {
    SetGuard {
        lock: w_set_lock_acquire(obj),
        not_send: std::marker::PhantomData,
    }
}

/// [`w_set_lock`] for the residual-call ABI: the guard's lock word.
/// Arguments and the result are descr words (`'r'`/`'i'` → i64;
/// `descr.py` `CallDescr.create_call_stub`, `callbuilder.py` `emit_raw_call`).
/// Conversion matches `helper_arg_from_i64` / `helper_return_to_i64` on the
/// macro trampoline (`emit_helper_fnaddr_registration`). The jitcode's drop
/// of the guard releases it through [`w_set_lock_release`].
pub extern "C" fn w_set_lock_jit_abi(obj: i64) -> i64 {
    let obj = obj as usize as PyObjectRef;
    std::mem::ManuallyDrop::new(unsafe { w_set_lock(obj) }).lock as i64
}

// The traced call names `w_set_lock`. Publish the word-returning entry under
// that path; `SetGuard` itself is not a residual result word.
#[cfg(not(target_arch = "wasm32"))]
#[majit_ir::linkme::distributed_slice(majit_ir::helper_fnaddr::HELPER_FNADDRS)]
#[linkme(crate = majit_ir::linkme)]
#[allow(non_upper_case_globals)]
static W_SET_LOCK_JIT_ABI: majit_ir::helper_fnaddr::HelperFnAddr =
    majit_ir::helper_fnaddr::HelperFnAddr::new(
        "pyre_object::setobject::w_set_lock",
        w_set_lock_jit_abi as *const (),
        1,
    );

#[cfg(target_arch = "wasm32")]
#[ctor::ctor(unsafe)]
fn register_w_set_lock_jit_abi() {
    majit_ir::helper_fnaddr::register(
        "pyre_object::setobject::w_set_lock",
        w_set_lock_jit_abi as *const (),
        1,
    );
}

/// Acquire one recursion level and return its opaque owner.
///
/// # Safety
/// `obj` must be a live set. The caller must release the returned handle
/// exactly once on the acquiring thread.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_lock_acquire(obj: PyObjectRef) -> usize {
    acquire_set_lock_handle(SET_LOCKS[set_lock_index(obj)].get())
}

fn acquire_set_lock_handle(lock: &'static parking_lot::ReentrantMutex<()>) -> usize {
    let guard = if let Some(guard) = lock.try_lock() {
        guard
    } else {
        let blocked = majit_gc::gc_sync::before_external_block();
        let guard = lock.lock();
        drop(blocked);
        guard
    };
    // `ReentrantMutex::force_unlock` supports a forgotten guard. The returned
    // handle carries that obligation.
    std::mem::forget(guard);
    lock as *const parking_lot::ReentrantMutex<()> as usize
}

/// Discharge one acquisition of a handle from [`w_set_lock_acquire`].
///
/// # Safety
/// `lock` must be an unreleased handle from [`w_set_lock_acquire`] on this thread.
#[majit_macros::dont_look_inside_cannot_raise]
pub unsafe fn w_set_lock_release(lock: usize) {
    unsafe { (&*(lock as *const parking_lot::ReentrantMutex<()>)).force_unlock() };
}

/// `try_lock` only. A contended stripe returns `None` so the caller can
/// leave the insert to [`w_set_lock`], which parks at a safepoint.
#[majit_macros::dont_look_inside]
fn try_w_set_lock(obj: PyObjectRef) -> Option<SetGuard> {
    let lock = SET_LOCKS[set_lock_index(obj)].get();
    let guard = lock.try_lock()?;
    std::mem::forget(guard);
    Some(SetGuard {
        lock: lock as *const parking_lot::ReentrantMutex<()> as usize,
        not_send: std::marker::PhantomData,
    })
}

/// Whether `value` is a plain int [`IntegerSetStrategy`] already stores.
///
/// [`AbstractUnwrappedSetStrategy::add`] inserts when the key is the wrong
/// type or absent. This answers the other half: `is_plain_int1` holds and
/// `contains_key` finds the unwrapped int. It does not hash, pin, allocate,
/// or wait on the stripe. A contended lock, a non-int, or any other strategy
/// answers false so the caller runs the real add.
#[majit_macros::dont_look_inside]
pub fn plain_int_already_in_int_set(set: PyObjectRef, value: PyObjectRef) -> bool {
    if set.is_null() || value.is_null() {
        return false;
    }
    unsafe {
        if !is_set(set) || !crate::listobject::is_plain_int1(value) {
            return false;
        }
        let n = crate::listobject::plain_int_w(value);
        let Some(_guard) = try_w_set_lock(set) else {
            return false;
        };
        let (kind, storage) = {
            let set_obj = &*(set as *const W_SetObject);
            (set_obj.strategy.kind, set_obj.sstorage)
        };
        if kind != SetStrategyKind::Int || storage.is_null() {
            return false;
        }
        (*(storage as *const IntSetStorage)).contains_key(&n)
    }
}

/// Lock two sets in stripe order. A shared stripe is acquired once.
#[majit_macros::dont_look_inside]
unsafe fn w_set_lock_pair(left: PyObjectRef, right: PyObjectRef) -> (SetGuard, Option<SetGuard>) {
    let left_index = set_lock_index(left);
    let right_index = set_lock_index(right);
    if left_index == right_index {
        return (w_set_lock(left), None);
    }
    if left_index < right_index {
        (w_set_lock(left), Some(w_set_lock(right)))
    } else {
        (w_set_lock(right), Some(w_set_lock(left)))
    }
}

pub fn set_locks_after_fork_child() {
    for lock in SET_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

/// Runtime-assigned GC type id for [`SetItemsStorage`]. Like the bigint
/// payload id, this is published by `pyre-jit::eval` after the fixed-constant
/// type registrations and is never embedded in a JIT allocation descriptor.
static SET_ITEMS_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`SetItemsStorage`].
pub fn set_set_items_gc_type_id(id: u32) {
    SET_ITEMS_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`SetItemsStorage`].
#[majit_macros::dont_look_inside]
pub fn set_items_gc_type_id() -> u32 {
    SET_ITEMS_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for the [`IntSetStorage`] box.
static INT_SET_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IntSetStorage`] box.
pub fn set_int_set_storage_gc_type_id(id: u32) {
    INT_SET_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IntSetStorage`] box.
#[majit_macros::dont_look_inside]
pub fn int_set_storage_gc_type_id() -> u32 {
    INT_SET_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for the [`BytesSetStorage`] box.
static BYTES_SET_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`BytesSetStorage`] box.
pub fn set_bytes_set_storage_gc_type_id(id: u32) {
    BYTES_SET_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`BytesSetStorage`] box.
#[majit_macros::dont_look_inside]
pub fn bytes_set_storage_gc_type_id() -> u32 {
    BYTES_SET_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for the [`AsciiSetStorage`] box.
static ASCII_SET_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`AsciiSetStorage`] box.
pub fn set_ascii_set_storage_gc_type_id(id: u32) {
    ASCII_SET_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`AsciiSetStorage`] box.
#[majit_macros::dont_look_inside]
pub fn ascii_set_storage_gc_type_id() -> u32 {
    ASCII_SET_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for the [`IdentitySetStorage`] box.
static IDENTITY_SET_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IdentitySetStorage`] box.
pub fn set_identity_set_storage_gc_type_id(id: u32) {
    IDENTITY_SET_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IdentitySetStorage`] box.
#[majit_macros::dont_look_inside]
pub fn identity_set_storage_gc_type_id() -> u32 {
    IDENTITY_SET_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fixed payload size (`framework.py:811`).
pub const W_SET_OBJECT_SIZE: usize = std::mem::size_of::<W_SetObject>();
/// User-subclass set layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (166) ahead of the
/// target-gated tail. `set` and `frozenset` share it.
pub const W_SET_USER_GC_TYPE_ID: u32 = 166;
pub const W_SET_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_SetObjectUser>();

impl crate::lltype::GcType for W_SetObject {
    fn type_id() -> u32 {
        W_SET_GC_TYPE_ID
    }
    const SIZE: usize = W_SET_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_SetObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_SET_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_SET_USER_OBJECT_SIZE;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_set(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SET_TYPE) || py_type_check(obj, &SET_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_frozenset(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FROZENSET_TYPE) || py_type_check(obj, &FROZENSET_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_set_or_frozenset(obj: PyObjectRef) -> bool {
    unsafe { is_set(obj) || is_frozenset(obj) }
}

/// Fire the write barrier on a set whose `items` field was just replaced.
///
/// `ll_dict_setitem` records the store on the `dicttable`
/// (`rordereddict.py` `GcStruct("dicttable")`), not on the set.
/// [`set_items_write_barrier`] is that store. This one is only the
/// `sstorage` assignment (`setobject.py` `W_BaseSetObject.clear`,
/// `get_storage_copy`). A no-GC-hook fallback allocation is not
/// collector-owned.
#[inline]
fn set_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    if obj.is_null() {
        return;
    }
    let set = unsafe { &*(obj as *const W_SetObject) };
    // Storage replacement is a membership change for every strategy,
    // including an integer set whose kind returns below.
    set.bump_content_gen();
    // `EmptySetStrategy.get_empty_storage` is null. Kind and storage are
    // read together; an empty set has nothing to remember.
    if set.strategy.kind != SetStrategyKind::Object {
        return;
    }
    let items = unsafe { object_set_storage_ptr(set) };
    if !items.is_null() && crate::gc_hook::try_gc_owns_object(items as crate::gc_hook::GCREF) {
        crate::gc_hook::try_gc_write_barrier(items as crate::gc_hook::GCREF);
    }
}

/// `setobject.py W_BaseSetObject.switch_to_empty_strategy`.
///
/// Strategy is published before the null storage word, so a reader still
/// observing `ObjectSetStrategy` still has the previous box.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
unsafe fn switch_to_empty_strategy(obj: PyObjectRef) {
    {
        let s = &mut *(obj as *mut W_SetObject);
        s.strategy = &EMPTY_SET_STRATEGY_REF;
        s.sstorage = EMPTY_SET_STRATEGY.get_empty_storage();
        s.set_len_relaxed(0);
        s.hash = -1;
    }
    set_write_barrier(obj);
}

/// `setobject.py EmptySetStrategy.add` — install `ObjectSetStrategy` and
/// its empty storage, then the caller performs the add. A plain int,
/// an exact `bytes`, an exact ASCII `str`, and a key whose type
/// `W_TypeObject.compares_by_identity` holds take [`switch_empty_to`]
/// with `IntegerSetStrategy`, `BytesSetStrategy`, `AsciiSetStrategy`, or
/// `IdentitySetStrategy`.
///
/// Storage is published before the kind, so the word is never
/// `ObjectSetStrategy` over null. `try_gc_alloc_stable_raw` does not collect;
/// the pin matches `install_empty_strategy`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` whose strategy is
/// `EmptySetStrategy`. Caller holds `w_set_lock`.
unsafe fn switch_empty_to_object_strategy(obj: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let storage = OBJECT_SET_STRATEGY.get_empty_storage();
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    {
        let set = &mut *(obj as *mut W_SetObject);
        set.sstorage = storage;
        set.strategy = &OBJECT_SET_STRATEGY_REF;
    }
    set_write_barrier(obj);
}

/// Publish `items`'s length when that box is still `dst`'s live storage.
///
/// A re-entrant `clear` has already stored length 0 on the empty strategy;
/// copying the orphan's length back would resurrect it.
///
/// # Safety
/// `dst` must point at a valid `W_SetObject`. `items` must be a live
/// [`SetItemsStorage`] box when it is still `dst`'s storage.
unsafe fn publish_len_if_live_box(dst: PyObjectRef, items: *mut SetItemsStorage) {
    let set = &mut *(dst as *mut W_SetObject);
    if same_live_object_box(set, items) {
        set.set_len_relaxed((*items).len());
        set.hash = -1;
    }
}

/// `setobject.py W_SetObject._discard_from_set`: a removal that leaves
/// `length() == 0` calls `switch_to_empty_strategy`. A removal against a box
/// `clear` already detached does not touch the live set.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject`. `items` must be a live
/// [`SetItemsStorage`] box when it is still `obj`'s storage. Caller holds
/// `w_set_lock`.
unsafe fn publish_discard_if_live_box(obj: PyObjectRef, items: *mut SetItemsStorage) {
    let emptied = {
        let set = &mut *(obj as *mut W_SetObject);
        if !same_live_object_box(set, items) {
            false
        } else {
            let new_len = (*items).len();
            if new_len == 0 {
                true
            } else {
                set.set_len_relaxed(new_len);
                set.hash = -1;
                false
            }
        }
    };
    if emptied {
        switch_to_empty_strategy(obj);
    }
}

/// Remember `items` after a store of a GC reference into that box.
///
/// The box is old (`gc_alloc_storage_box`) and `set_items_storage_custom_trace`
/// walks its `ObjectKey.obj` slots. A minor reaches an old box only from the
/// remembered set — `drag_out` does not trace an old root — including a box
/// `clear` has already detached (`d = self.unerase(w_set.sstorage)`,
/// `setobject.py`).
#[inline]
fn set_items_write_barrier(items: *mut SetItemsStorage) {
    crate::gc_hook::try_gc_write_barrier(items as crate::gc_hook::GCREF);
}

/// Process-wide source of [`W_SetObject::set_id`].
///
/// [`next_version_tag_serial`] is the same shape: a `static` atomic has no
/// llop, so the bump stays residual and the id is not a traced constant.
#[majit_macros::dont_look_inside]
fn fresh_set_id() -> usize {
    static NEXT_SET_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_SET_ID.fetch_add(1, Ordering::Relaxed)
}

/// Allocate an empty `set`.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `w_dict_new` twin: the body is a nursery allocation
/// (`try_gc_alloc_nursery_raw`). Residualising the constructor models it by
/// signature — a plain `PyObjectRef` GCREF. Storage is
/// `EmptySetStrategy.get_empty_storage` (`erase(None)`); the `SetItemsStorage`
/// box appears on the first add.
#[majit_macros::dont_look_inside]
pub fn w_set_new() -> PyObjectRef {
    alloc_set_object(&SET_TYPE)
}

/// Allocate an empty `frozenset`.
///
/// Same body as [`w_set_new`] with the constant `&FROZENSET_TYPE` baked
/// into `ob_type`. `#[dont_look_inside]` for the same nursery-allocation
/// reason as [`w_set_new`].
#[majit_macros::dont_look_inside]
pub fn w_frozenset_new() -> PyObjectRef {
    alloc_set_object(&FROZENSET_TYPE)
}

fn alloc_set_object(set_type: &'static PyType) -> PyObjectRef {
    // Allocate the body on the same nursery bump as `w_tuple_new`
    // (`malloc_fixedsize`). Falls back to `malloc_typed` when no GC hook is
    // installed (unit tests). `EmptySetStrategy.get_empty_storage` is
    // `erase(None)`, so there is no items box to pin. The class is reloaded
    // after the nursery allocation (`get_instantiate` / the body malloc can
    // collect).
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(set_type));
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_SET_GC_TYPE_ID, W_SET_OBJECT_SIZE);
    let body = W_SetObject {
        ob_header: PyObject {
            ob_type: set_type as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        sstorage: EMPTY_SET_STRATEGY.get_empty_storage(),
        strategy: &EMPTY_SET_STRATEGY_REF,
        len: crate::object_array::length_cell(0),
        hash: -1,
        set_id: fresh_set_id(),
        content_gen: crate::object_array::length_cell(0),
        lifeline: PY_NULL,
    };
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_SetObject, body);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(body) as PyObjectRef
    }
}

/// `allocate_instance(W_SetObjectUser, w_class)`: empty set or frozenset,
/// with `map`/`storage` at the `MapdictStorageMixin` initial state.
/// `frozen` selects `FROZENSET_USER_TYPE`.
///
/// `#[dont_look_inside]` for the same nursery-allocation reason as
/// [`w_set_new`]. The subclass word is pinned before the body malloc: a
/// user class is movable, and the allocation can collect.
#[majit_macros::dont_look_inside]
pub fn w_set_user_new_empty(w_class: PyObjectRef, frozen: bool) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let user_type: &'static PyType = if frozen {
        &FROZENSET_USER_TYPE
    } else {
        &SET_USER_TYPE
    };
    let raw =
        crate::gc_hook::try_gc_alloc_nursery_raw(W_SET_USER_GC_TYPE_ID, W_SET_USER_OBJECT_SIZE);
    let body = W_SetObjectUser {
        base: W_SetObject {
            ob_header: PyObject {
                ob_type: user_type as *const PyType,
                w_class: crate::gc_roots::shadow_stack_get(class_slot),
            },
            sstorage: EMPTY_SET_STRATEGY.get_empty_storage(),
            strategy: &EMPTY_SET_STRATEGY_REF,
            len: crate::object_array::length_cell(0),
            hash: -1,
            set_id: fresh_set_id(),
            content_gen: crate::object_array::length_cell(0),
            lifeline: PY_NULL,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_SetObjectUser, body);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(body) as PyObjectRef
    }
}

/// Allocate a populated set-like from a slice of elements (deduped).
///
/// A `&[PyObjectRef]` is not a root area, so the caller's words go stale at
/// the first insert: `w_set_add` hashes with `space.hash_w`, a collection
/// point that may run a user `__hash__`, and the set's own table can grow.
/// The whole slice is therefore published as one livevar set before the
/// constructor allocates, and both the set and each element are read back
/// out of their slots per insert.
fn set_from_items(new_set: impl FnOnce() -> PyObjectRef, items: &[PyObjectRef]) -> PyObjectRef {
    let roots = crate::gc_roots::push_roots();
    let base = roots.publish(items);
    roots.normalize(base, items.len());
    let set = base + items.len();
    let _ = roots.pin_root(new_set());
    for i in 0..items.len() {
        unsafe { w_set_add(roots.get(set), roots.get(base + i)) };
    }
    roots.get(set)
}

/// Allocate a populated set from a slice of elements (deduped).
pub fn w_set_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    set_from_items(w_set_new, items)
}

/// Allocate a populated frozenset from a slice of elements (deduped).
pub fn w_frozenset_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    set_from_items(w_frozenset_new, items)
}

/// Insert an element. No-op when already present.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add(obj: PyObjectRef, item: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for(crate::gc_roots::shadow_stack_get(item_slot));
    let key = crate::dictmultiobject::object_key_hashed(
        crate::gc_roots::shadow_stack_get(item_slot),
        key.hash,
    );
    let _ = w_set_insert_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key);
}

/// Insert an element keyed on a `space.hash_w` digest the caller already
/// holds, propagating an `eq_w` raise from the bucket probe.
///
/// `setobject.py newset` builds the backing `r_dict` with both
/// `space.eq_w` and `space.hash_w`, so one `add` hashes the element once and
/// compares it with `eq_w`, and either callback raising aborts the store.
/// A user `__hash__` is a collection point that can move both `obj` and
/// `item`, so the hash is taken by the caller while they are still rooted and
/// the digest handed down here; `hash` must be the `space.hash_w` result for
/// `item` (see [`object_key_hashed`](crate::dictmultiobject::object_key_hashed)).
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add_hashed_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
    hash: i64,
) -> Result<(), SetUpdateError> {
    w_set_insert_key_checked(obj, crate::dictmultiobject::object_key_hashed(item, hash))
}

/// Run a set operation with the callback-free guard raised, so
/// `dict_keys_equal` answers from its builtin type ladder and no user
/// `__eq__` observes or mutates the set mid-probe.
///
/// Returns `None` when a comparison escaped the ladder: `op` withholds its
/// mutation in that case, so the set is untouched and the caller re-runs the
/// operation by scanning entries without holding a table borrow across a
/// callback.
#[inline]
unsafe fn callback_free_set_op<T>(
    op: impl FnOnce() -> T,
) -> Option<Result<T, crate::dictmultiobject::DictKeyError>> {
    crate::dict_eq_hook::begin_callback_free_probe();
    let result = op();
    if crate::dict_eq_hook::end_callback_free_probe() {
        return None;
    }
    if crate::dictmultiobject::take_dict_key_error() {
        return Some(Err(crate::dictmultiobject::DictKeyError));
    }
    Some(Ok(result))
}

/// Find `key` by scanning same-hash entries of a *captured* storage box one at
/// a time.  `items` is the box the caller snapshotted at operation entry and
/// pinned, mirroring `AbstractUnwrappedSetStrategy.add`'s
/// `d = self.unerase(w_set.sstorage)`: the whole probe runs
/// against that box, so if a probing `__eq__` swaps the set's live storage
/// (`AbstractUnwrappedSetStrategy.clear` → `switch_to_empty_strategy`) the
/// scan completes against the now-orphaned snapshot. The live word is
/// `EmptySetStrategy.get_empty_storage` (`erase(None)`), not a new table.
///
/// The storage box has a stable address (`gc_alloc_storage_box` →
/// `try_gc_alloc_stable_raw`), so `items` never moves; the caller's pin only
/// keeps an orphaned box alive across the callbacks.  The table borrow ends
/// before equality can call user code; a callback that grows or reorders the
/// captured box restarts the scan (`ll_dict_lookup` paranoia,
/// `rordereddict.py`).  The generation counter is exact here, not a
/// proxy: it is bumped by every compaction and reindex, the only two things
/// that move an entry out from under a slot number.
unsafe fn scan_set_key_reentrant(
    items: *mut SetItemsStorage,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<(Option<usize>, crate::dictmultiobject::ObjectKey), crate::dictmultiobject::DictKeyError>
{
    'restart: loop {
        let generation = (*items).generation();
        let mut i = 0;
        loop {
            let Some((slot, stored_hash, stored_obj)) = (*items)
                .next_entry(i)
                .map(|(slot, stored, _)| (slot, stored.hash, stored.obj))
            else {
                return Ok((None, key));
            };

            if stored_hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored_obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);

                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(crate::dictmultiobject::DictKeyError);
                }
                // Validate the paranoia condition before acting on the result:
                // `ll_dict_lookup` restarts even when the comparison answered
                // `true`, because a callback that reallocated the buffer or moved
                // the candidate leaves the matched index stale
                // (`rordereddict.py`).
                let disturbed = (*items).generation() != generation
                    // `entries.valid(index) && entries[index].key == checkingkey`.
                    || !(*items).get_slot(slot).is_some_and(|(stored, _)| {
                        stored.hash == stored_hash && stored.obj == stored_obj
                    });
                if disturbed {
                    continue 'restart;
                }
                if equal {
                    return Ok((Some(slot), key));
                }
            }
            i = slot + 1;
        }
    }
}

/// Snapshot the set's storage box and pin it for a reentrant probe, mirroring
/// PyPy's capture-before-probe (`d = self.unerase(w_set.sstorage)`).  The pin
/// keeps the box alive even if a probing `__eq__` swaps the set's live storage
/// (`w_set_clear`); the returned pointer is used for the whole operation.
#[inline]
unsafe fn capture_set_items(obj: PyObjectRef) -> *mut SetItemsStorage {
    let items = object_set_storage_ptr(&*(obj as *const W_SetObject));
    let _ = crate::gc_roots::pin_root(items as PyObjectRef);
    items
}

/// Store a key that carries its own digest, propagating an `eq_w` raise from
/// the bucket probe.
///
/// `setobject.py _intersect_unwrapped` places a key it took from another
/// set with `setitem_with_hash(result, key, keyhash, None)`, i.e. under the
/// digest the key already carries rather than one taken afresh.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_insert_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _set_guard = w_set_lock(obj);
    // `W_BaseSetObject.add` → `self.strategy.add(self, w_key)`.
    (*(obj as *const W_SetObject)).strategy.add(obj, key)
}

/// Explicit `Result` shell the trace field-reads. `__discriminant` is an
/// `i64` at byte 0 (`0` = `Ok`, `1` = `Err`) and `__pos_0` is one word at
/// byte 8.
///
/// `Ok(())` and `Ok(false)` share the zero payload. `Ok(true)` stores 1.
/// `Err(DictKeyError)` stores the address of one static unit. `SetUpdateError`
/// is its `u8` tag (`Key` = 0, `ChangedSize` = 1).
#[repr(C)]
struct StrategyResultShell {
    discriminant: i64,
    payload: i64,
}

/// Same two words as [`StrategyResultShell`], with the payload spelled as a
/// pointer so the unit's address can live in a static.
#[repr(C)]
struct DictKeyResultShell {
    discriminant: i64,
    payload: &'static crate::dictmultiobject::DictKeyError,
}

const _: () = {
    assert!(std::mem::size_of::<StrategyResultShell>() == 16);
    assert!(std::mem::offset_of!(StrategyResultShell, discriminant) == 0);
    assert!(std::mem::offset_of!(StrategyResultShell, payload) == 8);
    assert!(std::mem::size_of::<DictKeyResultShell>() == 16);
    assert!(std::mem::offset_of!(DictKeyResultShell, discriminant) == 0);
    assert!(std::mem::offset_of!(DictKeyResultShell, payload) == 8);
};

static STRATEGY_RESULT_OK_ZERO: StrategyResultShell = StrategyResultShell {
    discriminant: 0,
    payload: 0,
};
static STRATEGY_RESULT_OK_TRUE: StrategyResultShell = StrategyResultShell {
    discriminant: 0,
    payload: 1,
};
static DICT_KEY_ERROR_UNIT: crate::dictmultiobject::DictKeyError =
    crate::dictmultiobject::DictKeyError;
static STRATEGY_RESULT_ERR_DICT_KEY: DictKeyResultShell = DictKeyResultShell {
    discriminant: 1,
    payload: &DICT_KEY_ERROR_UNIT,
};
static STRATEGY_RESULT_ERR_KEY: StrategyResultShell = StrategyResultShell {
    discriminant: 1,
    payload: 0,
};
static STRATEGY_RESULT_ERR_CHANGED_SIZE: StrategyResultShell = StrategyResultShell {
    discriminant: 1,
    payload: 1,
};

fn strategy_result_ptr<T>(shell: &'static T) -> crate::gc_hook::GCREF {
    std::ptr::from_ref(shell) as crate::gc_hook::GCREF
}

fn set_strategy_add_shell(result: Result<(), SetUpdateError>) -> crate::gc_hook::GCREF {
    let shell = match result {
        Ok(()) => &STRATEGY_RESULT_OK_ZERO,
        Err(SetUpdateError::Key(_)) => &STRATEGY_RESULT_ERR_KEY,
        Err(SetUpdateError::ChangedSize) => &STRATEGY_RESULT_ERR_CHANGED_SIZE,
    };
    strategy_result_ptr(shell)
}

fn set_strategy_bool_shell(
    result: Result<bool, crate::dictmultiobject::DictKeyError>,
) -> crate::gc_hook::GCREF {
    match result {
        Ok(true) => strategy_result_ptr(&STRATEGY_RESULT_OK_TRUE),
        Ok(false) => strategy_result_ptr(&STRATEGY_RESULT_OK_ZERO),
        Err(_) => strategy_result_ptr(&STRATEGY_RESULT_ERR_DICT_KEY),
    }
}

/// Residual word ABI for `SetStrategy::add`.
///
/// The traced call passes `(w_set, &ObjectKey)`. `&self` on the ZST strategy
/// is void in the flow graph, and `ObjectKey` stays one pointer. The vtable
/// method is `fn(&self, PyObjectRef, ObjectKey)` — hash and obj in registers —
/// so calling that pointer with the two residual words reads the key pointer
/// as the set. This trampoline is the operation the trace recorded.
///
/// The method returns a `Result` ADT (`setobject.py SetStrategy.add`); the
/// interned shell is that residual word, `llmemory.GCREF`.
#[majit_macros::dont_look_inside]
pub extern "C" fn set_strategy_add_key_ptr(
    w_set: PyObjectRef,
    key: *const crate::dictmultiobject::ObjectKey,
) -> crate::gc_hook::GCREF {
    let key = unsafe { std::ptr::read(key) };
    let result = unsafe { (*(w_set as *const W_SetObject)).strategy.add(w_set, key) };
    set_strategy_add_shell(result)
}

/// Residual word ABI for `SetStrategy::remove`. Same argument words as
/// [`set_strategy_add_key_ptr`].
#[majit_macros::dont_look_inside]
pub extern "C" fn set_strategy_remove_key_ptr(
    w_set: PyObjectRef,
    key: *const crate::dictmultiobject::ObjectKey,
) -> crate::gc_hook::GCREF {
    let key = unsafe { std::ptr::read(key) };
    let result = unsafe { (*(w_set as *const W_SetObject)).strategy.remove(w_set, key) };
    set_strategy_bool_shell(result)
}

/// Residual word ABI for `SetStrategy::has_key`. Same argument words as
/// [`set_strategy_add_key_ptr`].
#[majit_macros::dont_look_inside]
pub extern "C" fn set_strategy_has_key_ptr(
    w_set: PyObjectRef,
    key: *const crate::dictmultiobject::ObjectKey,
) -> crate::gc_hook::GCREF {
    let key = unsafe { std::ptr::read(key) };
    let result = unsafe {
        (*(w_set as *const W_SetObject))
            .strategy
            .has_key(w_set, key)
    };
    set_strategy_bool_shell(result)
}

/// Membership test for a key that carries its own digest, propagating an
/// `eq_w` raise from the bucket probe.
///
/// `setobject.py _intersect_unwrapped` probes the other side with
/// `contains_with_hash(d_other, key, keyhash)`, reusing the digest the key was
/// stored under instead of hashing it again.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_contains_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let _set_guard = w_set_lock(obj);
    // `W_BaseSetObject.has_key` → `self.strategy.has_key(self, w_key)`.
    (*(obj as *const W_SetObject)).strategy.has_key(obj, key)
}

/// Remove a key that carries its own digest, propagating an `eq_w` raise from
/// the bucket probe. Returns true when an element was removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_discard_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let _set_guard = w_set_lock(obj);
    // `W_BaseSetObject.remove` → `self.strategy.remove(self, w_item)`,
    // then `_discard_from_set` switches when the set is empty.
    (*(obj as *const W_SetObject)).strategy.remove(obj, key)
}

/// Membership test.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for(crate::gc_roots::shadow_stack_get(item_slot));
    w_set_contains_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key).unwrap_or(false)
}

/// Fallible variant of [`w_set_contains`].
///
/// `setobject.py EmptySetStrategy.has_key` — the element is hashed
/// even when the set is empty, so an unhashable element raises rather than
/// reading as absent.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for_checked(crate::gc_roots::shadow_stack_get(
        item_slot,
    ))?;
    w_set_contains_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key)
}

/// Remove an element if present. Returns true when removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard(obj: PyObjectRef, item: PyObjectRef) -> bool {
    // Hashing the element can run `__hash__` and collect: the set and the
    // element are livevars across it, as in `w_set_contains`.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for(crate::gc_roots::shadow_stack_get(item_slot));
    w_set_discard_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key).unwrap_or(false)
}

/// Fallible variant of [`w_set_discard`].
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    // Same livevar set as [`w_set_discard`].
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for_checked(crate::gc_roots::shadow_stack_get(
        item_slot,
    ))?;
    w_set_discard_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key)
}

/// Remove every element.
///
/// `setobject.py W_BaseSetObject.clear` → `AbstractUnwrappedSetStrategy.clear`
/// → `switch_to_empty_strategy`. `EmptySetStrategy.clear` is a no-op.
/// `get_empty_storage` is `erase(None)`, so the live word becomes null rather
/// than an emptied box. A probe that captured the old box
/// (`scan_set_key_reentrant`) keeps running against that orphan: a later
/// insert lands in the dropped box and is lost, and a membership test
/// completes against the snapshot.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_clear(obj: PyObjectRef) {
    let _set_guard = w_set_lock(obj);
    // `W_BaseSetObject.clear` → `self.strategy.clear(self)`.
    (*(obj as *const W_SetObject)).strategy.clear(obj);
}

/// Remove and return an arbitrary stored element without hashing it again.
///
/// `setobject.py ObjectSetStrategy.popitem` delegates to the backing
/// dictionary's `popitem`; the key already occupies a bucket, so no user
/// `__hash__` or `__eq__` callback occurs while it leaves the set.
///
/// # Safety
/// `obj` must point to a valid mutable `W_SetObject`.
pub unsafe fn w_set_popitem(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _set_guard = w_set_lock(obj);
    // `W_BaseSetObject.popitem` → `self.strategy.popitem(self)`.
    (*(obj as *const W_SetObject)).strategy.popitem(obj)
}

/// Take over a copy of another set's storage, keeping the digest each element
/// was stored under.
///
/// `EmptySetStrategy.update` assigns a fresh storage table to the GC pointer
/// field (`w_set.sstorage = w_other.get_storage_copy()`), while
/// `ObjectSetStrategy.get_storage_copy` creates that table with
/// `self.erase(d.copy())`. The underlying table is the GC-managed
/// `rordereddict.py` `GcStruct("dicttable")`. Do the same field reassignment here,
/// rather than overwriting the old table's pointee. Copying the buckets is
/// what makes the operand's elements reach the new set without handing them
/// to a user `__hash__` (or `__eq__`) a second time.
///
/// `RDict::clone` allocates a young `DICTENTRYARRAY` (`_ll_malloc_entries` /
/// `alloc_entries`), which is a collection point. `src` and `dst` are
/// nursery-born set objects, so they live on the shadow stack across that
/// malloc and `get_storage_copy` writes through the reloaded `dst`.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `w_set_new` twin: cloning `src`'s `SetItemsStorage` (`RDict<ObjectKey,
/// ()>`) and boxing it into `d.sstorage` is a foreign `RDict::clone` +
/// storage-box write. Tracing into it carries that host container op into the
/// caller and unifies the `sstorage` field as `Instance(RDict)`; residualising
/// the whole assignment keeps the box off the trace, modelling it as a void
/// effect on two GCREFs.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_copy_storage_from(dst: PyObjectRef, src: PyObjectRef) {
    let (_first_guard, _second_guard) = w_set_lock_pair(dst, src);
    // `w_other.get_storage_copy()` installed by `copy_real` /
    // `EmptySetStrategy.update`.
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::pin_roots(&[dst, src]);
    let src_slot = dst_slot + 1;
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src_set = &*(src as *const W_SetObject);
    src_set.strategy.get_storage_copy(src, dst);
}

/// `AbstractUnwrappedSetStrategy._difference_update_wrapped`: walk `src`
/// and `remove` each key from `dst`. `src` may be an int set; the walk
/// goes through [`w_set_key_at`] so the key is wrapped.
///
/// # Safety
/// Caller holds `w_set_lock_pair`. Both slots are pinned.
unsafe fn difference_remove_src_keys(
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError> {
    let mut i = 0;
    loop {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let Some(slot) = w_set_next_slot(src, i) else {
            break;
        };
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let Some(key) = w_set_key_at(src, slot) else {
            return Err(SetUpdateError::ChangedSize);
        };
        let src_set = &*(crate::gc_roots::shadow_stack_get(src_slot) as *const W_SetObject);
        let src_kind = src_set.strategy.kind;
        let src_storage = src_set.sstorage;
        let src_len = src_set.len_relaxed();
        let _key_roots = crate::gc_roots::push_roots();
        let key_obj = crate::gc_roots::pin_root(key.obj);
        w_set_remove_key_for_update(
            crate::gc_roots::shadow_stack_get(dst_slot),
            crate::dictmultiobject::ObjectKey {
                hash: key.hash,
                obj: key_obj,
            },
        )?;
        let src_set = &*(crate::gc_roots::shadow_stack_get(src_slot) as *const W_SetObject);
        if src_set.strategy.kind != src_kind
            || src_set.sstorage != src_storage
            || src_set.len_relaxed() != src_len
        {
            return Err(SetUpdateError::ChangedSize);
        }
        i = slot + 1;
    }
    Ok(())
}

/// `ObjectSetStrategy.update` when the other side is also an object set:
/// `d_obj.update(d_other)` on the two captured tables (`ll_dict_update`).
///
/// Both boxes are captured once. A callback that clears either set swaps its
/// live storage; the merge keeps reading the captured source and inserting
/// into the captured destination. Keys stay in the table so
/// `set_items_storage_custom_trace` can move them; a `Vec` lifted out would
/// not be walked.
///
/// # Safety
/// `dst` is on [`OBJECT_SET_STRATEGY`]. Caller holds `w_set_lock_pair`.
/// `dst_slot` is pinned. `src` is rooted for the capture.
unsafe fn object_set_merge_captured(
    dst_slot: usize,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let dst_items = capture_set_items(dst);
    let src_items = capture_set_items(src);
    let mut i = 0;
    while let Some((slot, &key, _)) = (*src_items).next_entry(i) {
        w_set_insert_key_into(crate::gc_roots::shadow_stack_get(dst_slot), dst_items, key)?;
        i = slot + 1;
    }
    Ok(())
}

/// `AbstractUnwrappedSetStrategy.difference_update` after the empty,
/// same-storage, and int-strategy shortcuts. Shared by `ObjectSetStrategy`
/// and by `IntegerSetStrategy` when `src` is an object set and self is at
/// least as large: the big-minus-small arm unerases `src` only.
///
/// # Safety
/// Caller holds `w_set_lock_pair`. `dst_slot` and `src_slot` are pinned.
/// Calling this with an int `dst` requires `w_set_len(dst) >= w_set_len(src)`,
/// so the small-minus-big arm (which unerases `dst` as [`SetItemsStorage`])
/// does not run.
unsafe fn difference_update_object_storage(
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError> {
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    // `SetStrategy.difference_update` — small_set -= big_set computes a fresh
    // difference by walking the smaller self storage, then replaces self's
    // storage wholesale. Besides the complexity bound, this preserves the
    // exact contains-with-hash callback direction of the upstream strategy.
    if w_set_len(dst) < w_set_len(src) {
        // The difference accumulator has no referrer until
        // `w_set_copy_storage_from` below: the `eq_w` a bucket probe runs is a
        // collection point that would otherwise sweep it, and the operand
        // bodies relocate, so each walk step reloads them.
        let result_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set_new());
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let dst_items = object_set_storage_ptr(&*(dst as *const W_SetObject));
        let dst_len = (*dst_items).len();
        let mut i = 0;
        while let Some(slot) = w_set_next_slot(crate::gc_roots::shadow_stack_get(dst_slot), i) {
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let Some(key) = w_set_key_at(dst, slot) else {
                return Err(SetUpdateError::ChangedSize);
            };
            if !w_set_contains_key_for_update(crate::gc_roots::shadow_stack_get(src_slot), key)? {
                let dst = crate::gc_roots::shadow_stack_get(dst_slot);
                if !same_live_object_box(&*(dst as *const W_SetObject), dst_items)
                    || (*dst_items).len() != dst_len
                {
                    return Err(SetUpdateError::ChangedSize);
                }
                // The comparison may clear or otherwise shorten `dst`.
                // The set probe restarts when `entry->key` changes;
                // once this live index disappeared there is no surviving
                // entry to copy into the difference result.
                let Some(key) = w_set_key_at(dst, slot) else {
                    return Err(SetUpdateError::ChangedSize);
                };
                w_set_insert_key_checked(crate::gc_roots::shadow_stack_get(result_slot), key)?;
            }
            i = slot + 1;
        }
        w_set_copy_storage_from(
            crate::gc_roots::shadow_stack_get(dst_slot),
            crate::gc_roots::shadow_stack_get(result_slot),
        );
        return Ok(());
    }
    // `src` is a distinct storage, so removing from `dst` cannot renumber it.
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let src_items = object_set_storage_ptr(&*(src as *const W_SetObject));
    let src_len = (*src_items).len();
    let mut i = 0;
    while let Some(slot) = w_set_next_slot(crate::gc_roots::shadow_stack_get(src_slot), i) {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let Some(key) = w_set_key_at(src, slot) else {
            return Err(SetUpdateError::ChangedSize);
        };
        w_set_remove_key_for_update(crate::gc_roots::shadow_stack_get(dst_slot), key)?;
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        if !same_live_object_box(&*(src as *const W_SetObject), src_items)
            || (*src_items).len() != src_len
        {
            return Err(SetUpdateError::ChangedSize);
        }
        i = slot + 1;
    }
    Ok(())
}

/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_difference_update_from_set(
    dst: PyObjectRef,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dst);
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(src);
    let (_first_guard, _second_guard) = w_set_lock_pair(
        crate::gc_roots::shadow_stack_get(dst_slot),
        crate::gc_roots::shadow_stack_get(src_slot),
    );
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    // `W_BaseSetObject.difference_update` → `self.strategy.difference_update`.
    (*(dst as *const W_SetObject))
        .strategy
        .difference_update(dst, src)
}

/// Merge a set operand's storage in, keeping the digests it holds.
///
/// `setobject.py ObjectSetStrategy.update` takes `d_obj.update(
/// d_other)` when the operand shares this strategy — labelled "optimization
/// only" upstream, but it is also what keeps a set operand's elements from
/// being handed to a user `__hash__` a second time. Elements equal across the
/// two sides still meet in a bucket, so `eq_w` runs and can raise.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_update_from_set(
    dst: PyObjectRef,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    // The bodies relocate: `w_set_insert_key_into` runs `eq_w`, and this
    // frame's `dst`/`src` locals are the only referrers for a temporary.
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dst);
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(src);
    let (_first_guard, _second_guard) = w_set_lock_pair(
        crate::gc_roots::shadow_stack_get(dst_slot),
        crate::gc_roots::shadow_stack_get(src_slot),
    );
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    // `W_BaseSetObject.update` → `self.strategy.update(self, w_other)`.
    (*(dst as *const W_SetObject)).strategy.update(dst, src)
}

/// Insert one cached-hash key without holding a table borrow across user
/// `eq_w`.
///
/// The set's storage box is captured and pinned once at entry
/// (`capture_set_items`), mirroring PyPy's `d = self.unerase(w_set.sstorage)`
/// (`AbstractUnwrappedSetStrategy.add`): the membership probe and the
/// follow-up insert both run against that box.  If a probing `__eq__` clears
/// the set, `switch_to_empty_strategy` nulls the live word while this insert
/// still targets the orphaned snapshot, so the element lands in the dropped
/// box and is lost — the behaviour PyPy exhibits when `add` captures its
/// storage before a re-entrant `clear`.  The scan drops the table borrow before each
/// comparison and inserts through an already-proven vacant raw entry, so
/// insertion performs no second user callback.
unsafe fn w_set_insert_key_reentrant(
    dst: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _roots = crate::gc_roots::push_roots();
    let items = capture_set_items(dst);
    w_set_insert_key_into(dst, items, key)
}

/// Probe-and-insert half of [`w_set_insert_key_reentrant`] against a storage
/// box the caller already captured and pinned.  `w_set_update_from_set` passes
/// the box it captured for the whole merge so every source key targets the same
/// (possibly orphaned) table, the way `ll_dict_update` keeps inserting into its
/// captured `dic1`.
unsafe fn w_set_insert_key_into(
    dst: PyObjectRef,
    items: *mut SetItemsStorage,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _dst_roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dst);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let mut key = key;
    key.obj = crate::gc_roots::pin_root(key.obj);
    // Single insert probe (matches `r_dict.setitem`'s one bucket scan), run
    // callback-free so no user `__eq__` mutates the set while the table
    // borrow is live.  When every same-hash comparison stays inside the
    // builtin ladder the probe appends in place; a pair it cannot decide
    // breaks the probe, withholds the store, and re-runs the operation over
    // `scan_set_key_reentrant` below.  Without this the membership half of
    // every `add` walks the table entry by entry, which is quadratic in the
    // set's size.
    if let Some(result) = callback_free_set_op(|| {
        let entries = &mut *items;
        let index = entries.index_of(&key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            return false;
        }
        if index.is_some() {
            return false;
        }
        // The probe above proved no bucket entry compares equal without
        // leaving the ladder, so this placement probe repeats those same
        // comparisons and cannot break either.
        let mut live_key = key;
        live_key.obj = crate::gc_roots::shadow_stack_get(key_slot);
        entries.insert(live_key, ());
        true
    }) {
        if result.map_err(SetUpdateError::Key)? {
            // The key landed in `items`, which a probing `clear` may already
            // have detached from `dst`. Barrier that box, not the set body.
            // The live length stays the 0 `clear` published.
            set_items_write_barrier(items);
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            publish_len_if_live_box(dst, items);
        }
        return Ok(());
    }

    key.obj = crate::gc_roots::shadow_stack_get(key_slot);
    let (found, key) = scan_set_key_reentrant(items, key).map_err(SetUpdateError::Key)?;
    if found.is_some() {
        return Ok(());
    }
    // The scan above proved the key absent, and repeating the probe here would
    // repeat its comparisons — the ones that can re-enter this table.  Place it
    // on the digest alone (`ll_call_insert_clean_function`).
    (*items).insert_known_absent(key, ());
    // Same box the scan captured. A `clear` inside that scan has already
    // nulled `dst.sstorage`; the new key lives here, and the live length
    // stays the 0 `clear` published.
    set_items_write_barrier(items);
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    publish_len_if_live_box(dst, items);
    Ok(())
}

/// Membership half of `contains_with_hash` for a mutation-sensitive set
/// operation.  As with `rordereddict.ll_dict_lookup`, no table borrow crosses
/// `eq_w`, and a callback that replaces an entry restarts the lookup.
unsafe fn w_set_contains_key_for_update(
    probe: PyObjectRef,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, SetUpdateError> {
    let _probe_roots = crate::gc_roots::push_roots();
    let probe_slot = crate::gc_roots::shadow_stack_len();
    let probe = crate::gc_roots::pin_root(probe);
    let key_root = crate::gc_roots::shadow_stack_len();
    key.obj = crate::gc_roots::pin_root(key.obj);
    // `EmptySetStrategy.has_key` is False once the key is hashed. The digest
    // is already on `key`.
    if (*(probe as *const W_SetObject)).strategy.kind == SetStrategyKind::Empty {
        return Ok(false);
    }
    // `AbstractUnwrappedSetStrategy.has_key`: a key of this strategy's type
    // probes the unwrapped table; any other key switches to object and retries.
    let kind = (*(probe as *const W_SetObject)).strategy.kind;
    if matches!(
        kind,
        SetStrategyKind::Int
            | SetStrategyKind::Bytes
            | SetStrategyKind::Ascii
            | SetStrategyKind::Identity
    ) {
        // `None`: the set switched to object and the scan below retries.
        let answered = on_unwrapped!(kind, strategy => {
            unwrapped_contains_or_switch(strategy, probe_slot, key)
        })
        .map_err(SetUpdateError::Key)?;
        if let Some(bit) = answered {
            return Ok(bit);
        }
    }
    let probe = crate::gc_roots::shadow_stack_get(probe_slot);
    key.obj = crate::gc_roots::shadow_stack_get(key_root);
    // Bucket probe first, as in `w_set_contains_key_checked`.  The walk below
    // is the reentrant fallback and visits every entry, so without this a
    // whole-set difference probes linearly per element and runs quadratic.
    if let Some(result) = callback_free_set_op(|| {
        let s = &*(probe as *const W_SetObject);
        if s.strategy.kind == SetStrategyKind::Empty {
            return false;
        }
        (*object_set_storage_ptr(s)).contains_key(&key)
    }) {
        return result.map_err(SetUpdateError::Key);
    }
    'restart: loop {
        let probe = crate::gc_roots::shadow_stack_get(probe_slot);
        key.obj = crate::gc_roots::shadow_stack_get(key_root);
        if (*(probe as *const W_SetObject)).strategy.kind == SetStrategyKind::Empty {
            return Ok(false);
        }
        let items = object_set_storage_ptr(&*(probe as *const W_SetObject));
        let len = (*items).len();
        let mut i = 0;
        loop {
            let Some((slot, &stored, _)) = (*items).next_entry(i) else {
                break;
            };
            if stored.hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored.obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);
                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(SetUpdateError::Key(crate::dictmultiobject::DictKeyError));
                }
                let probe = crate::gc_roots::shadow_stack_get(probe_slot);
                if !same_live_object_box(&*(probe as *const W_SetObject), items)
                    || (*items).len() != len
                    || !(*items).get_slot(slot).is_some_and(|(current, _)| {
                        current.hash == stored.hash && current.obj == stored_obj
                    })
                {
                    continue 'restart;
                }
                if equal {
                    return Ok(true);
                }
            }
            i = slot + 1;
        }
        return Ok(false);
    }
}

/// `delitem_with_hash` for a mutation-sensitive difference update.  Find the
/// matching bucket without lending the table across Python code, restart after
/// a hostile comparison like `ll_dict_lookup`, then delete the proven slot
/// without another equality callback.
unsafe fn w_set_remove_key_for_update(
    dst: PyObjectRef,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _dst_roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let dst = crate::gc_roots::pin_root(dst);
    let key_root = crate::gc_roots::shadow_stack_len();
    key.obj = crate::gc_roots::pin_root(key.obj);
    // An empty set has nothing to delete. `EmptySetStrategy.remove` is the
    // public discard path; this helper is `delitem_with_hash` and does not
    // switch strategy when the last element leaves.
    if (*(dst as *const W_SetObject)).strategy.kind == SetStrategyKind::Empty {
        return Ok(());
    }
    // `AbstractUnwrappedSetStrategy.remove` / `delitem_with_hash`. A key of
    // this strategy's type is deleted from the unwrapped table and the
    // strategy stays put even when the dict becomes empty. Any other key
    // switches, then the object path below deletes it.
    let kind = (*(dst as *const W_SetObject)).strategy.kind;
    if matches!(
        kind,
        SetStrategyKind::Int
            | SetStrategyKind::Bytes
            | SetStrategyKind::Ascii
            | SetStrategyKind::Identity
    ) {
        // `None`: the set switched to object and the scan below deletes it.
        // `Some`: `delitem_with_hash` already ran (`to_empty` is false).
        let answered = on_unwrapped!(kind, strategy => {
            unwrapped_remove_or_switch(strategy, dst_slot, key)
        })
        .map_err(SetUpdateError::Key)?;
        if answered.is_some() {
            return Ok(());
        }
    }
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    key.obj = crate::gc_roots::shadow_stack_get(key_root);
    // Locate the bucket callback-free before falling back to the entry walk,
    // which is linear in the set's size.  The index is resolved inside the
    // probe and the removal withheld when a comparison leaves the builtin
    // ladder, so the walk below can redo the whole operation.
    if let Some(result) = callback_free_set_op(|| {
        let set = &*(dst as *const W_SetObject);
        if set.strategy.kind == SetStrategyKind::Empty {
            return false;
        }
        let items = object_set_storage_ptr(set);
        let index = (*items).index_of(&key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            return false;
        }
        if let Some(index) = index {
            set_remove_slot(items, index);
            true
        } else {
            false
        }
    }) {
        if result.map_err(SetUpdateError::Key)? {
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let set = &mut *(dst as *mut W_SetObject);
            set.set_len_relaxed(set.len_relaxed() - 1);
            set.hash = -1;
        }
        return Ok(());
    }
    'restart: loop {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        key.obj = crate::gc_roots::shadow_stack_get(key_root);
        if (*(dst as *const W_SetObject)).strategy.kind == SetStrategyKind::Empty {
            return Ok(());
        }
        let items = object_set_storage_ptr(&*(dst as *const W_SetObject));
        let len = (*items).len();
        let mut found = None;
        let mut i = 0;
        loop {
            let Some((slot, &stored, _)) = (*items).next_entry(i) else {
                break;
            };
            if stored.hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored.obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);
                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(SetUpdateError::Key(crate::dictmultiobject::DictKeyError));
                }
                let dst = crate::gc_roots::shadow_stack_get(dst_slot);
                if !same_live_object_box(&*(dst as *const W_SetObject), items)
                    || (*items).len() != len
                    || !(*items).get_slot(slot).is_some_and(|(current, _)| {
                        current.hash == stored.hash && current.obj == stored_obj
                    })
                {
                    continue 'restart;
                }
                if equal {
                    found = Some(slot);
                    break;
                }
            }
            i = slot + 1;
        }
        if let Some(index) = found {
            set_remove_slot(items, index);
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let set = &mut *(dst as *mut W_SetObject);
            set.set_len_relaxed(set.len_relaxed() - 1);
            set.hash = -1;
        }
        return Ok(());
    }
}

/// Number of elements in the set.
///
/// Takes no stripe lock, matching `Objects/setobject.c set_len`:
/// `FT_ATOMIC_LOAD_SSIZE_RELAXED(so->used)`.  A lock here would be stricter
/// than the spec and would not buy anything — every mutator publishes the
/// count with a relaxed store while holding the stripe, so the value read is
/// from one side of a concurrent mutation either way, and the JIT's `len` fold
/// lowers to this same unlocked load.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetObject)).len_relaxed()
}

/// Cached frozenset hash; `-1` is the uncomputed sentinel.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_frozenset_cached_hash(obj: PyObjectRef) -> Option<i64> {
    let _set_guard = w_set_lock(obj);
    let hash = (*(obj as *const W_SetObject)).hash;
    (hash != -1).then_some(hash)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_frozenset_set_cached_hash(obj: PyObjectRef, hash: i64) {
    let _set_guard = w_set_lock(obj);
    (*(obj as *mut W_SetObject)).hash = hash;
}

/// Digests already carried by the r_dict keys. Python 3.14 frozenset hashing
/// consumes these instead of invoking each element's `__hash__` again.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_stored_hashes(obj: PyObjectRef) -> Vec<i64> {
    let _set_guard = w_set_lock(obj);
    // `EmptySetStrategy.iter` yields nothing. `IntegerSetStrategy` wraps
    // through `hash_w` (`intobject.py _hash_int`) inside `stored_hashes`.
    (*(obj as *const W_SetObject)).strategy.stored_hashes(obj)
}

/// The key in `slot`, carrying the digest it was stored under, or `None` when
/// nothing occupies it — past the last entry, or in the hole a delete left.
/// [`w_set_next_slot`] finds the slots that answer.
///
/// `setobject.py iterkeys_with_hash` walks a storage handing out
/// `(key, keyhash)` pairs so the walk's consumer can place or probe the key
/// without hashing it again. Reading one index at a time lets a caller whose
/// loop body reaches user code (an `eq_w` from a bucket probe) re-read the key
/// afterwards: the collector rewrites the `obj` slots inside the table in place
/// (`set_items_storage_custom_trace`), so a key read before that point can be
/// stale while the table itself stays correct.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_key_at(
    obj: PyObjectRef,
    slot: usize,
) -> Option<crate::dictmultiobject::ObjectKey> {
    let _set_guard = w_set_lock(obj);
    // `IntegerIteratorImplementation.next_entry` is `space.newint`.
    (*(obj as *const W_SetObject)).strategy.key_at(obj, slot)
}

/// `num_ever_used_items` (`rordereddict.py` `_ll_dictnext`): one past the
/// highest slot a walk may name. Dead slots below it read as null from
/// [`w_set_iterkey_at`].
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`) so the
/// entries-length read stays inside this residual and the caller's merge
/// loop only sees a word.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_num_ever_used_items(obj: *mut PyObject) -> usize {
    let _set_guard = w_set_lock(obj);
    (*(obj as *const W_SetObject))
        .strategy
        .num_ever_used_items(obj)
}

/// One step of `iterkeys_with_hash` (`rlib/objectmodel.py`): the key stored
/// at `index`, or null when that slot is dead or past the end.
///
/// The slot read (`RDict::get_slot`, the `get_index` lookup the merge used to
/// inline) is residual (`@jit.dont_look_inside`, `rlib/jit.py`) so a traced
/// merge loop does not lower it. The digest half is [`w_set_iterkey_hash_at`].
/// Parameters are `*mut PyObject` so the word-ABI trampoline is emitted; a
/// `PyObjectRef` alias is not that token.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_iterkey_at(obj: *mut PyObject, index: usize) -> *mut PyObject {
    let _set_guard = w_set_lock(obj);
    (*(obj as *const W_SetObject))
        .strategy
        .iterkey_at(obj, index)
}

/// Digest half of [`w_set_iterkey_at`]. Only meaningful when that call
/// returned a key. Same residual boundary.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `index` must name a live slot.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_iterkey_hash_at(obj: *mut PyObject, index: usize) -> i64 {
    let _set_guard = w_set_lock(obj);
    (*(obj as *const W_SetObject))
        .strategy
        .iterkey_hash_at(obj, index)
}

/// `contains_with_hash` (`rlib/objectmodel.py` / `ll_dict_contains_with_hash`).
///
/// Returns `1` when present, `0` when absent, and `-1` when user `__eq__`
/// raised. The concrete exception stays in the pending dict-key slot, the
/// same channel [`w_set_contains_key_checked`] uses. The reentrant scan lives
/// inside this residual so the caller's loop does not lower it.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `hash` must be the digest
/// `key` was stored under.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_contains_with_hash(obj: *mut PyObject, key: *mut PyObject, hash: i64) -> i64 {
    match w_set_contains_key_checked(obj, crate::dictmultiobject::ObjectKey { hash, obj: key }) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => -1,
    }
}

/// `setitem_with_hash` (`rlib/objectmodel.py` / `ll_dict_setitem_with_hash`)
/// placing `None` as the set value.
///
/// Returns `0` on success, `-1` when user `__eq__` raised (the exception is
/// in the pending dict-key slot), and `-2` when a callback resized the table
/// (`SetUpdateError::ChangedSize`). Absence was
/// already decided by [`w_set_contains_with_hash`]; insertion still probes,
/// because a callback may have changed the table, and that probe is
/// [`w_set_insert_key_checked`] (callback-free, then
/// [`scan_set_key_reentrant`]).
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `hash` must be the digest
/// `key` was stored under.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_setitem_with_hash(obj: *mut PyObject, key: *mut PyObject, hash: i64) -> i64 {
    match w_set_insert_key_checked(obj, crate::dictmultiobject::ObjectKey { hash, obj: key }) {
        Ok(()) => 0,
        Err(SetUpdateError::Key(_)) => -1,
        Err(SetUpdateError::ChangedSize) => -2,
    }
}

/// Snapshot the contained elements as a `Vec`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_items(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _set_guard = w_set_lock(obj);
    // `W_BaseSetObject.getkeys` → `self.strategy.getkeys(self)`.
    // A contended stripe parks in `before_external_block`, so the set is
    // reloaded after the acquire rather than kept in the pre-lock local.
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    (*(obj as *const W_SetObject)).strategy.getkeys(obj)
}

/// Walk, in place, every element `PyObjectRef` slot of a set for an
/// immortal owner.  Forwards each `ObjectKey.obj` slot: `ObjectKey.hash` is
/// identity-stable across a GC move, so writing the relocated pointer through
/// the key's `obj` slot keeps the bucket index valid.  Alloc-free — unlike
/// [`w_set_items`], which materialises a `Vec`.  A collector-owned set does
/// not use this walk: `set_items_storage_custom_trace` traces the storage
/// box, and the set body only forwards `items`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_walk_gc_refs(obj: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
    // No stripe: an immortal owner, the way `w_dict_walk_gc_refs` walks.
    // `EmptySetStrategy` / `IntegerSetStrategy` visit nothing.
    // `BytesSetStrategy` / `AsciiSetStrategy` visit the key block.
    // `IdentitySetStrategy` visits the object pointer on `IdentitySetKey`.
    (*(obj as *const W_SetObject))
        .strategy
        .walk_gc_refs(obj, visitor);
}

fn same_erased_storage(left: PyObjectRef, right: PyObjectRef) -> bool {
    let left = unsafe { &*(left as *const W_SetObject) };
    let right = unsafe { &*(right as *const W_SetObject) };
    left.strategy.kind == right.strategy.kind
        && !left.sstorage.is_null()
        && std::ptr::eq(left.sstorage, right.sstorage)
}

impl SetStrategy for EmptySetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind {
        SetStrategyKind::Empty
    }

    /// `EmptySetStrategy.get_empty_storage` — `erase(None)`.
    fn get_empty_storage(&self) -> GCREF {
        std::ptr::null_mut()
    }

    /// `EmptySetStrategy.length` is 0. The atomic slot is published as 0.
    unsafe fn length(&self, _w_set: PyObjectRef) -> usize {
        0
    }

    unsafe fn add(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<(), SetUpdateError> {
        // `EmptySetStrategy.add`: `is_plain_int1` → `IntegerSetStrategy`;
        // `type is W_BytesObject` → `BytesSetStrategy`; exact `W_UnicodeObject`
        // and `is_ascii` → `AsciiSetStrategy`;
        // `W_TypeObject.compares_by_identity` → `IdentitySetStrategy`;
        // else `ObjectSetStrategy`. Then `w_set.add`.
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set);
        let key_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(key.obj);
        let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
        if crate::listobject::is_plain_int1(key_obj) {
            switch_empty_to(
                &INTEGER_SET_STRATEGY,
                crate::gc_roots::shadow_stack_get(obj_slot),
            );
        } else if is_exact_bytes_object(key_obj) {
            switch_empty_to(
                &BYTES_SET_STRATEGY,
                crate::gc_roots::shadow_stack_get(obj_slot),
            );
        } else if is_exact_ascii_str(key_obj) {
            switch_empty_to(
                &ASCII_SET_STRATEGY,
                crate::gc_roots::shadow_stack_get(obj_slot),
            );
        } else if IDENTITY_SET_STRATEGY.is_correct_type(key_obj) {
            switch_empty_to(
                &IDENTITY_SET_STRATEGY,
                crate::gc_roots::shadow_stack_get(obj_slot),
            );
        } else {
            switch_empty_to_object_strategy(crate::gc_roots::shadow_stack_get(obj_slot));
        }
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let key = crate::dictmultiobject::ObjectKey {
            hash: key.hash,
            obj: crate::gc_roots::shadow_stack_get(key_slot),
        };
        (*(obj as *const W_SetObject)).strategy.add(obj, key)
    }

    /// `EmptySetStrategy.remove` returns False. The caller already hashed.
    unsafe fn remove(
        &self,
        _w_set: PyObjectRef,
        _key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError> {
        Ok(false)
    }

    /// `EmptySetStrategy.has_key` is False once `hash_w` has run.
    unsafe fn has_key(
        &self,
        _w_set: PyObjectRef,
        _key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError> {
        Ok(false)
    }

    /// `EmptySetStrategy.clear` is a no-op.
    unsafe fn clear(&self, _w_set: PyObjectRef) {}

    /// `EmptySetStrategy.popitem` raises KeyError.
    unsafe fn popitem(&self, _w_set: PyObjectRef) -> Option<PyObjectRef> {
        None
    }

    /// `EmptySetStrategy.get_storage_copy` is the null word (`erase(None)`).
    unsafe fn get_storage_copy(&self, _src: PyObjectRef, dst: PyObjectRef) {
        switch_to_empty_strategy(dst);
    }

    /// `EmptySetStrategy.getkeys` is `[]`.
    unsafe fn getkeys(&self, _w_set: PyObjectRef) -> Vec<PyObjectRef> {
        Vec::new()
    }

    /// `EmptyIteratorImplementation.next_entry` is always None.
    unsafe fn next_slot(&self, _w_set: PyObjectRef, _from: usize) -> Option<usize> {
        None
    }

    unsafe fn key_at(
        &self,
        _w_set: PyObjectRef,
        _slot: usize,
    ) -> Option<crate::dictmultiobject::ObjectKey> {
        None
    }

    unsafe fn num_ever_used_items(&self, _w_set: PyObjectRef) -> usize {
        0
    }

    unsafe fn iterkey_at(&self, _w_set: PyObjectRef, _index: usize) -> *mut PyObject {
        std::ptr::null_mut()
    }

    unsafe fn iterkey_hash_at(&self, _w_set: PyObjectRef, _index: usize) -> i64 {
        0
    }

    /// `EmptySetStrategy.iter` yields nothing.
    unsafe fn stored_hashes(&self, _w_set: PyObjectRef) -> Vec<i64> {
        Vec::new()
    }

    /// `EmptySetStrategy.update` steals the other's strategy and storage copy.
    unsafe fn update(
        &self,
        w_set: PyObjectRef,
        w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError> {
        if (*(w_other as *const W_SetObject)).strategy.kind != SetStrategyKind::Empty {
            w_set_copy_storage_from(w_set, w_other);
        }
        Ok(())
    }

    /// `EmptySetStrategy.difference_update` is a no-op.
    /// `may_contain_equal_elements(EmptySetStrategy)` is false, so an empty
    /// operand removes nothing; an empty self has nothing to remove.
    unsafe fn difference_update(
        &self,
        _w_set: PyObjectRef,
        _w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError> {
        Ok(())
    }

    unsafe fn walk_gc_refs(&self, _w_set: PyObjectRef, _visitor: &mut dyn FnMut(*mut PyObjectRef)) {
    }
}

impl SetStrategy for ObjectSetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind {
        SetStrategyKind::Object
    }

    /// `ObjectSetStrategy.get_empty_storage` — `erase(newset)`.
    fn get_empty_storage(&self) -> GCREF {
        crate::gc_storage::gc_alloc_storage_box(SetItemsStorage::default(), set_items_gc_type_id())
            as GCREF
    }

    unsafe fn length(&self, w_set: PyObjectRef) -> usize {
        (*(w_set as *const W_SetObject)).len_relaxed()
    }

    unsafe fn add(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<(), SetUpdateError> {
        // `ObjectSetStrategy` `is_correct_type` is always true, so `add`
        // stores into the captured box (`AbstractUnwrappedSetStrategy.add`).
        w_set_insert_key_reentrant(w_set, key)
    }

    unsafe fn remove(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError> {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set);
        let key_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(key.obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let key = crate::dictmultiobject::ObjectKey {
            hash: key.hash,
            obj: crate::gc_roots::shadow_stack_get(key_slot),
        };
        if let Some(result) = callback_free_set_op(|| {
            let items = {
                let s = &*(obj as *const W_SetObject);
                object_set_storage_ptr(s)
            };
            let index = (*items).index_of(&key);
            if crate::dict_eq_hook::callback_free_probe_broken() {
                return false;
            }
            let Some(index) = index else {
                return false;
            };
            set_remove_slot(items, index);
            publish_discard_if_live_box(obj, items);
            true
        }) {
            return result;
        }
        let items = capture_set_items(obj);
        let (found, _) = scan_set_key_reentrant(items, key)?;
        if let Some(index) = found {
            // Remove from the captured box; a `clear` during the probe orphans
            // it. A removal that empties the live box runs
            // `switch_to_empty_strategy` (`W_SetObject._discard_from_set`).
            set_remove_slot(items, index);
            let obj = crate::gc_roots::shadow_stack_get(obj_slot);
            publish_discard_if_live_box(obj, items);
            return Ok(true);
        }
        Ok(false)
    }

    unsafe fn has_key(
        &self,
        w_set: PyObjectRef,
        key: crate::dictmultiobject::ObjectKey,
    ) -> Result<bool, crate::dictmultiobject::DictKeyError> {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set);
        let key_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(key.obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let key = crate::dictmultiobject::ObjectKey {
            hash: key.hash,
            obj: crate::gc_roots::shadow_stack_get(key_slot),
        };
        if let Some(result) = callback_free_set_op(|| {
            let s = &*(obj as *const W_SetObject);
            (*object_set_storage_ptr(s)).contains_key(&key)
        }) {
            return result;
        }
        let items = capture_set_items(obj);
        let (found, _) = scan_set_key_reentrant(items, key)?;
        Ok(found.is_some())
    }

    /// `AbstractUnwrappedSetStrategy.clear` → `switch_to_empty_strategy`.
    unsafe fn clear(&self, w_set: PyObjectRef) {
        switch_to_empty_strategy(w_set);
    }

    unsafe fn popitem(&self, w_set: PyObjectRef) -> Option<PyObjectRef> {
        // `ObjectSetStrategy.popitem` delegates to the backing dict. The set
        // stays on this strategy when the dict underneath becomes empty.
        let s = &mut *(w_set as *mut W_SetObject);
        let entries = &mut *object_set_storage_ptr(s);
        let (key, ()) = entries.pop()?;
        s.set_len_relaxed(s.len_relaxed() - 1);
        s.hash = -1;
        Some(key.obj)
    }

    unsafe fn get_storage_copy(&self, src: PyObjectRef, dst: PyObjectRef) {
        let _roots = crate::gc_roots::push_roots();
        let dst_slot = crate::gc_roots::pin_roots(&[dst, src]);
        let src_slot = dst_slot + 1;
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let copied = (*object_set_storage_ptr(&*(src as *const W_SetObject))).clone();
        // `gc_alloc_storage_box` is a stable allocation and never collects.
        // `RDict::clone` above already did: write into the forwarded `dst`.
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        {
            let d = &mut *(dst as *mut W_SetObject);
            // Box first, then the kind, so the word is never Object over null.
            d.sstorage =
                crate::gc_storage::gc_alloc_storage_box(copied, set_items_gc_type_id()) as GCREF;
            d.strategy = &OBJECT_SET_STRATEGY_REF;
            d.set_len_relaxed((*object_set_storage_ptr(d)).len());
            d.hash = -1;
        }
        // `sstorage` assignment on the set, then the copied keys on the new
        // table. The clone filled a host `RDict` before the box existed, so
        // the element barrier did not run for those stores.
        set_write_barrier(dst);
        let d = &*(dst as *const W_SetObject);
        if !(*object_set_storage_ptr(d)).is_empty() {
            set_items_write_barrier(object_set_storage_ptr(d));
        }
    }

    unsafe fn getkeys(&self, w_set: PyObjectRef) -> Vec<PyObjectRef> {
        let s = &*(w_set as *const W_SetObject);
        let mut items = Vec::with_capacity((*object_set_storage_ptr(s)).len());
        for key in (*object_set_storage_ptr(s)).keys() {
            items.push(key.obj);
        }
        items
    }

    unsafe fn next_slot(&self, w_set: PyObjectRef, from: usize) -> Option<usize> {
        let s = &*(w_set as *const W_SetObject);
        (*object_set_storage_ptr(s)).next_valid_slot(from)
    }

    unsafe fn key_at(
        &self,
        w_set: PyObjectRef,
        slot: usize,
    ) -> Option<crate::dictmultiobject::ObjectKey> {
        let s = &*(w_set as *const W_SetObject);
        (*object_set_storage_ptr(s))
            .get_slot(slot)
            .map(|(&key, _)| key)
    }

    unsafe fn num_ever_used_items(&self, w_set: PyObjectRef) -> usize {
        let s = &*(w_set as *const W_SetObject);
        (*object_set_storage_ptr(s)).entry_slots()
    }

    unsafe fn iterkey_at(&self, w_set: PyObjectRef, index: usize) -> *mut PyObject {
        let s = &*(w_set as *const W_SetObject);
        match (*object_set_storage_ptr(s)).get_slot(index) {
            Some((key, _)) => key.obj,
            None => std::ptr::null_mut(),
        }
    }

    unsafe fn iterkey_hash_at(&self, w_set: PyObjectRef, index: usize) -> i64 {
        let s = &*(w_set as *const W_SetObject);
        match (*object_set_storage_ptr(s)).get_slot(index) {
            Some((key, _)) => key.hash,
            None => 0,
        }
    }

    unsafe fn stored_hashes(&self, w_set: PyObjectRef) -> Vec<i64> {
        let s = &*(w_set as *const W_SetObject);
        (*object_set_storage_ptr(s))
            .keys()
            .map(|key| key.hash)
            .collect()
    }

    unsafe fn update(
        &self,
        w_set: PyObjectRef,
        w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError> {
        // `ObjectSetStrategy.update`. An empty operand stops immediately.
        // The same strategy takes `d_obj.update(d_other)`; an int operand is
        // iterated as wrapped keys and does not switch this strategy.
        let src_kind = (*(w_other as *const W_SetObject)).strategy.kind;
        if src_kind == SetStrategyKind::Empty {
            return Ok(());
        }
        if same_erased_storage(w_set, w_other) {
            return Ok(());
        }
        let _roots = crate::gc_roots::push_roots();
        let dst_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set);
        let src_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_other);
        update_object_from_other(dst_slot, src_slot)
    }

    unsafe fn difference_update(
        &self,
        w_set: PyObjectRef,
        w_other: PyObjectRef,
    ) -> Result<(), SetUpdateError> {
        let src_kind = (*(w_other as *const W_SetObject)).strategy.kind;
        if src_kind == SetStrategyKind::Empty {
            return Ok(());
        }
        if same_erased_storage(w_set, w_other) {
            w_set_clear(w_set);
            set_write_barrier(w_set);
            return Ok(());
        }
        let _roots = crate::gc_roots::push_roots();
        let dst_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set);
        let src_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_other);
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        // `ObjectSetStrategy.may_contain_equal_elements` is true except for empty.
        // A larger object self removes the other's wrapped keys
        // (`_difference_update_wrapped`). A bytes, ascii, or identity operand
        // is not an object table, so it cannot take
        // `difference_update_object_storage`'s big-minus-small arm.
        if matches!(
            src_kind,
            SetStrategyKind::Int
                | SetStrategyKind::Bytes
                | SetStrategyKind::Ascii
                | SetStrategyKind::Identity
        ) && self.length(dst) >= (*(src as *const W_SetObject)).strategy.length(src)
        {
            return difference_remove_src_keys(dst_slot, src_slot);
        }
        difference_update_object_storage(dst_slot, src_slot)
    }

    unsafe fn walk_gc_refs(&self, w_set: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let set = &mut *(w_set as *mut W_SetObject);
        // `get_empty_storage` is null only on `EmptySetStrategy`. An object
        // set with a null word is not unerase'd.
        if set.sstorage.is_null() {
            return;
        }
        let entries = &mut *object_set_storage_ptr(set);
        for (key, _) in entries.iter_mut_for_trace() {
            let key_ptr = key as *const crate::dictmultiobject::ObjectKey
                as *mut crate::dictmultiobject::ObjectKey;
            visitor(std::ptr::addr_of_mut!((*key_ptr).obj) as *mut PyObjectRef);
        }
        visitor(entries.entries_slot() as *mut PyObjectRef);
        entries.visit_indexes(&mut |slot| visitor(slot as *mut PyObjectRef));
    }
}

/// `setobject.py UNROLL_CUTOFF`, the cutoff on
/// `get_storage_from_unwrapped_list`.
const SET_UNROLL_CUTOFF: usize = 5;

fn int_storage_from_unwrapped_iff(items: &[i64]) -> bool {
    majit_rlib::jit::loop_unrolling_heuristic(items, items.len(), SET_UNROLL_CUTOFF)
}

fn bytes_storage_from_unwrapped_iff(items: &[*const crate::bytesobject::BytesBlock]) -> bool {
    majit_rlib::jit::loop_unrolling_heuristic(items, items.len(), SET_UNROLL_CUTOFF)
}

fn ascii_storage_from_unwrapped_iff(
    items: &[*const crate::unicodeobject::UnicodeValueStorage],
) -> bool {
    majit_rlib::jit::loop_unrolling_heuristic(items, items.len(), SET_UNROLL_CUTOFF)
}

/// `AbstractUnwrappedSetStrategy.get_storage_from_unwrapped_list` for
/// plain ints. Duplicates collapse; the first insertion stays.
///
/// `get_empty_dict` then insert. `gc_alloc_storage_box` first so the
/// rdict is a GC object before the loop.
#[majit_macros::look_inside_iff(int_storage_from_unwrapped_iff)]
fn int_storage_from_unwrapped(items: &[i64]) -> (GCREF, usize) {
    let _roots = crate::gc_roots::push_roots();
    let storage =
        crate::gc_storage::gc_alloc_storage_box(IntSetStorage::new(), int_set_storage_gc_type_id());
    let _ = crate::gc_roots::pin_root(storage as PyObjectRef);
    let dict = unsafe { &mut *(storage as *mut IntSetStorage) };
    let idx_slot = dict.pin_indexes();
    for &item in items {
        dict.insert(item, ());
        dict.reload_indexes_root(idx_slot);
    }
    let len = dict.len();
    (storage as GCREF, len)
}

/// `get_storage_from_unwrapped_list` for `bytes` blocks. The key is the
/// block `BytesSetStrategy.unwrap` stores, not a copy of its characters.
///
/// `get_empty_dict` then insert through `gc_alloc_storage_box`.
#[majit_macros::look_inside_iff(bytes_storage_from_unwrapped_iff)]
fn bytes_storage_from_unwrapped(items: &[*const crate::bytesobject::BytesBlock]) -> (GCREF, usize) {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::shadow_stack_len();
    for &item in items {
        let _ = crate::gc_roots::pin_root(item as PyObjectRef);
    }
    let storage = crate::gc_storage::gc_alloc_storage_box(
        BytesSetStorage::new(),
        bytes_set_storage_gc_type_id(),
    );
    let _ = crate::gc_roots::pin_root(storage as PyObjectRef);
    let dict = unsafe { &mut *(storage as *mut BytesSetStorage) };
    let idx_slot = dict.pin_indexes();
    for index in 0..items.len() {
        let block =
            crate::gc_roots::shadow_stack_get(base + index) as *mut crate::bytesobject::BytesBlock;
        dict.insert(crate::dictmultiobject::BytesKey(block), ());
        dict.reload_indexes_root(idx_slot);
    }
    let len = dict.len();
    (storage as GCREF, len)
}

/// `get_storage_from_unwrapped_list` for ASCII rstrs.
///
/// `publish_roots` then one `normalize_roots`. Per-item `pin_root` would
/// query after the first rstr and leave the rest invisible. The pins stay
/// up across `gc_alloc_storage_box`. `get_empty_dict` then insert through
/// that box.
#[majit_macros::look_inside_iff(ascii_storage_from_unwrapped_iff)]
fn ascii_storage_from_unwrapped(
    items: &[*const crate::unicodeobject::UnicodeValueStorage],
) -> (GCREF, usize) {
    let _roots = crate::gc_roots::push_roots();
    let mut published = Vec::with_capacity(items.len());
    for &item in items {
        published.push(item as PyObjectRef);
    }
    let base = crate::gc_roots::publish_roots(&published);
    crate::gc_roots::normalize_roots(base, published.len());
    let storage = crate::gc_storage::gc_alloc_storage_box(
        AsciiSetStorage::new(),
        ascii_set_storage_gc_type_id(),
    );
    let _ = crate::gc_roots::pin_root(storage as PyObjectRef);
    let dict = unsafe { &mut *(storage as *mut AsciiSetStorage) };
    let idx_slot = dict.pin_indexes();
    for index in 0..items.len() {
        let block =
            crate::gc_roots::shadow_stack_get(base + index) as *mut crate::unicodeobject::Utf8Str;
        dict.insert(crate::celldict::StrKey(block), ());
        dict.reload_indexes_root(idx_slot);
    }
    let len = dict.len();
    (storage as GCREF, len)
}

/// Publish unwrapped storage. `sstorage` lands before the strategy, the
/// same order as `switch_empty_to`. The frozenset hash cache is left
/// alone: `set_strategy_and_setdata` assigns strategy and storage only.
///
/// # Safety
/// `obj` must be a live `W_SetObject`. Caller holds `w_set_lock`.
/// `storage` must be the box for `strategy_ref`.
unsafe fn publish_set_listview_storage(
    obj: PyObjectRef,
    storage: GCREF,
    strategy_ref: &'static SetStrategyRef,
    len: usize,
) {
    {
        let set = &mut *(obj as *mut W_SetObject);
        set.sstorage = storage;
        set.strategy = strategy_ref;
        set.set_len_relaxed(len);
    }
    set_write_barrier(obj);
}

/// Install `IntegerSetStrategy` from `listview_int`, including `[]`.
///
/// The box is pinned before `w_set_lock`. `gc_alloc_storage_box` is old-gen,
/// and a contended stripe parks in `before_external_block`; mark-sweep
/// reclaims a box that is not yet stored in `sstorage` (`IntArray::pin_block`
/// is the same bracket).
///
/// # Safety
/// `obj` must be a live `W_SetObject`.
pub unsafe fn w_set_install_int_items(obj: PyObjectRef, items: &[i64]) {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let (storage, len) = int_storage_from_unwrapped(items);
    let storage_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(storage as PyObjectRef);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let _guard = w_set_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let storage = crate::gc_roots::shadow_stack_get(storage_slot) as GCREF;
    publish_set_listview_storage(obj, storage, &INTEGER_SET_STRATEGY_REF, len);
}

/// Install `BytesSetStrategy` from `listview_bytes`, including `[]`.
///
/// # Safety
/// `obj` must be a live `W_SetObject`. Each pointer must be a live
/// `BytesBlock` or the slice must be empty.
pub unsafe fn w_set_install_bytes_items(
    obj: PyObjectRef,
    items: &[*const crate::bytesobject::BytesBlock],
) {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let (storage, len) = bytes_storage_from_unwrapped(items);
    let storage_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(storage as PyObjectRef);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let _guard = w_set_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let storage = crate::gc_roots::shadow_stack_get(storage_slot) as GCREF;
    publish_set_listview_storage(obj, storage, &BYTES_SET_STRATEGY_REF, len);
}

/// Install `AsciiSetStrategy` from `listview_ascii`, including `[]`.
///
/// The set and every rstr are one `publish_roots` before
/// `normalize_roots`. `listview_ascii` on a `str` returns fresh nursery
/// rstrs (`alloc_utf8_payload`); `pin_root` on the set alone would query
/// before those rstrs were roots. The box is pinned before `w_set_lock`.
///
/// # Safety
/// `obj` must be a live `W_SetObject`. Each pointer must be a live rstr
/// or the slice must be empty.
pub unsafe fn w_set_install_ascii_items(
    obj: PyObjectRef,
    items: &[*const crate::unicodeobject::UnicodeValueStorage],
) {
    let _roots = crate::gc_roots::push_roots();
    let mut published = Vec::with_capacity(1 + items.len());
    published.push(obj);
    for &item in items {
        published.push(item as PyObjectRef);
    }
    let base = crate::gc_roots::publish_roots(&published);
    crate::gc_roots::normalize_roots(base, published.len());
    let mut live = Vec::with_capacity(items.len());
    for index in 0..items.len() {
        live.push(crate::gc_roots::shadow_stack_get(base + 1 + index)
            as *const crate::unicodeobject::UnicodeValueStorage);
    }
    let (storage, len) = ascii_storage_from_unwrapped(&live);
    let storage_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(storage as PyObjectRef);
    let obj = crate::gc_roots::shadow_stack_get(base);
    let _guard = w_set_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(base);
    let storage = crate::gc_roots::shadow_stack_get(storage_slot) as GCREF;
    publish_set_listview_storage(obj, storage, &ASCII_SET_STRATEGY_REF, len);
}

/// `IntegerSetStrategy.listview_int`. `None` unless the strategy is
/// integer and the box is present. An empty integer set is `Some([])`.
///
/// # Safety
/// `obj` must be null or a live set or frozenset.
pub unsafe fn w_set_listview_int(obj: PyObjectRef) -> Option<Vec<i64>> {
    if obj.is_null() || !is_set_or_frozenset(obj) {
        return None;
    }
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _guard = w_set_lock(obj);
    let set = &*(crate::gc_roots::shadow_stack_get(obj_slot) as *const W_SetObject);
    if set.strategy.kind != SetStrategyKind::Int || set.sstorage.is_null() {
        return None;
    }
    let storage = &*(set.sstorage as *const IntSetStorage);
    Some(storage.keys().copied().collect())
}

/// `BytesSetStrategy.listview_bytes`.
///
/// # Safety
/// `obj` must be null or a live set or frozenset.
pub unsafe fn w_set_listview_bytes(
    obj: PyObjectRef,
) -> Option<Vec<*const crate::bytesobject::BytesBlock>> {
    if obj.is_null() || !is_set_or_frozenset(obj) {
        return None;
    }
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _guard = w_set_lock(obj);
    let set = &*(crate::gc_roots::shadow_stack_get(obj_slot) as *const W_SetObject);
    if set.strategy.kind != SetStrategyKind::Bytes || set.sstorage.is_null() {
        return None;
    }
    let storage = &*(set.sstorage as *const BytesSetStorage);
    Some(
        storage
            .keys()
            .map(|key| key.0 as *const crate::bytesobject::BytesBlock)
            .collect(),
    )
}

/// `AsciiSetStrategy.listview_ascii`.
///
/// # Safety
/// `obj` must be null or a live set or frozenset.
pub unsafe fn w_set_listview_ascii(
    obj: PyObjectRef,
) -> Option<Vec<*const crate::unicodeobject::UnicodeValueStorage>> {
    if obj.is_null() || !is_set_or_frozenset(obj) {
        return None;
    }
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _guard = w_set_lock(obj);
    let set = &*(crate::gc_roots::shadow_stack_get(obj_slot) as *const W_SetObject);
    if set.strategy.kind != SetStrategyKind::Ascii || set.sstorage.is_null() {
        return None;
    }
    let storage = &*(set.sstorage as *const AsciiSetStorage);
    Some(
        storage
            .keys()
            .map(|key| key.0 as *const crate::unicodeobject::UnicodeValueStorage)
            .collect(),
    )
}

/// `objspace.py listview_bytes` for an exact list or an exact dict.
/// Bytes objects are not a list of bytes (`listview_bytes` returns
/// `None` for `W_BytesObject`).
unsafe fn listview_bytes_of(
    obj: PyObjectRef,
) -> Option<Vec<*const crate::bytesobject::BytesBlock>> {
    if unsafe { crate::is_exact_list(obj) } {
        return unsafe { crate::listobject::w_list_getitems_bytes(obj) };
    }
    if unsafe { crate::is_exact_type(obj, &crate::DICT_TYPE) } {
        let dict = unsafe { &*(obj as *const crate::dictmultiobject::W_DictObject) };
        if dict.dstrategy.kind != crate::dictmultiobject::StrategyKind::Bytes
            || dict.dstorage.is_null()
        {
            return None;
        }
        let storage = unsafe { crate::dictmultiobject::w_dict_bytes_storage(obj) };
        return Some(
            storage
                .keys()
                .map(|key| key.0 as *const crate::bytesobject::BytesBlock)
                .collect(),
        );
    }
    None
}

/// `objspace.py listview_ascii` for an exact list, or an exact `str`.
/// `UnicodeDictStrategy` has no `listview_ascii`.
unsafe fn listview_ascii_of(
    obj: PyObjectRef,
) -> Option<Vec<*const crate::unicodeobject::UnicodeValueStorage>> {
    if unsafe { crate::is_exact_list(obj) } {
        return unsafe { crate::listobject::w_list_getitems_ascii(obj) };
    }
    if unsafe { crate::is_exact_type(obj, &crate::STR_TYPE) } {
        return unsafe { crate::w_unicode_listview_ascii(obj) };
    }
    None
}

/// `objspace.py listview_int` for an exact list, an exact dict, or an
/// exact `bytes` (`W_BytesObject.listview_int` / `_create_list_from_bytes`).
/// A bytes subclass is not this arm.
unsafe fn listview_int_of(obj: PyObjectRef) -> Option<Vec<i64>> {
    if unsafe { crate::is_exact_list(obj) } {
        return unsafe { crate::listobject::w_list_getitems_int(obj) };
    }
    if unsafe { crate::is_exact_type(obj, &crate::DICT_TYPE) } {
        let dict = unsafe { &*(obj as *const crate::dictmultiobject::W_DictObject) };
        if dict.dstrategy.kind != crate::dictmultiobject::StrategyKind::Int
            || dict.dstorage.is_null()
        {
            return None;
        }
        let storage = unsafe { crate::dictmultiobject::w_dict_int_storage(obj) };
        return Some(storage.keys().copied().collect());
    }
    if unsafe { crate::is_exact_type(obj, &crate::BYTES_TYPE) } {
        let data = unsafe { crate::w_bytes_data(obj) };
        return Some(data.iter().map(|&byte| i64::from(byte)).collect());
    }
    None
}

/// `set_strategy_and_setdata` for one exact list: `getitems_bytes`, then
/// `getitems_ascii`, then `getitems_int`. `Some([])` installs that
/// strategy. `None` from every probe leaves the set untouched.
///
/// # Safety
/// `w_set` must be a live set or frozenset. `w_list` must be a live list
/// (a subclass shares the `W_ListObject` prefix).
pub unsafe fn w_set_init_from_list_storage(w_set: PyObjectRef, w_list: PyObjectRef) -> bool {
    if w_set.is_null() || w_list.is_null() {
        return false;
    }
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_set, w_list]);
    if let Some(items) =
        crate::listobject::w_list_getitems_bytes(crate::gc_roots::shadow_stack_get(base + 1))
    {
        w_set_install_bytes_items(crate::gc_roots::shadow_stack_get(base), &items);
        return true;
    }
    if let Some(items) =
        crate::listobject::w_list_getitems_ascii(crate::gc_roots::shadow_stack_get(base + 1))
    {
        w_set_install_ascii_items(crate::gc_roots::shadow_stack_get(base), &items);
        return true;
    }
    if let Some(items) =
        crate::listobject::w_list_getitems_int(crate::gc_roots::shadow_stack_get(base + 1))
    {
        w_set_install_int_items(crate::gc_roots::shadow_stack_get(base), &items);
        return true;
    }
    false
}

/// `setobject.py set_strategy_and_setdata` listview arm.
///
/// Order is `listview_bytes`, `listview_ascii`, `listview_int`. An empty
/// view installs that strategy (`set(b"")` is an empty
/// `IntegerSetStrategy`, `set("")` is an empty `AsciiSetStrategy`).
/// Exact list, exact dict, exact bytes, and exact ASCII `str` are the
/// probes in `objspace.py`. A set operand is copied by the caller before
/// this runs. Returns false when every probe is `None`.
///
/// # Safety
/// `w_set` must be a live set or frozenset. `w_iterable` must be a live
/// object.
pub unsafe fn w_set_init_from_listview(w_set: PyObjectRef, w_iterable: PyObjectRef) -> bool {
    if w_set.is_null() || w_iterable.is_null() {
        return false;
    }
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[w_set, w_iterable]);
    if let Some(items) = listview_bytes_of(crate::gc_roots::shadow_stack_get(base + 1)) {
        w_set_install_bytes_items(crate::gc_roots::shadow_stack_get(base), &items);
        return true;
    }
    if let Some(items) = listview_ascii_of(crate::gc_roots::shadow_stack_get(base + 1)) {
        w_set_install_ascii_items(crate::gc_roots::shadow_stack_get(base), &items);
        return true;
    }
    if let Some(items) = listview_int_of(crate::gc_roots::shadow_stack_get(base + 1)) {
        w_set_install_int_items(crate::gc_roots::shadow_stack_get(base), &items);
        return true;
    }
    false
}

/// `EmptyListStrategy._extend_from_iterable` when the iterable is a
/// builtin set. `unpackiterable_int` installs only a non-empty int view
/// (`if lst`). `listview_bytes` and `listview_ascii` install an empty
/// view. A receiver that is no longer `Empty` or `Size`, and a set with
/// no view, return false so the caller keeps its snapshot.
///
/// The set lock inside the listview helper is released before the list
/// lock in the install.
///
/// # Safety
/// `list` must be a live list. `set` must be a live set or frozenset.
pub unsafe fn w_list_try_extend_empty_from_set(list: PyObjectRef, set: PyObjectRef) -> bool {
    if list.is_null() || set.is_null() {
        return false;
    }
    if !matches!(
        crate::listobject::w_list_strategy(list),
        crate::listobject::ListStrategy::Empty | crate::listobject::ListStrategy::Size
    ) {
        return false;
    }
    // Each listview helper takes the set stripe. A contended acquire parks
    // in `before_external_block`, so both operands are reloaded after the
    // view and before the matching install.
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[list, set]);
    if let Some(ints) = w_set_listview_int(crate::gc_roots::shadow_stack_get(base + 1))
        && !ints.is_empty()
        && crate::listobject::w_list_install_int_items(
            crate::gc_roots::shadow_stack_get(base),
            &ints,
        )
    {
        return true;
    }
    if let Some(blocks) = w_set_listview_bytes(crate::gc_roots::shadow_stack_get(base + 1))
        && crate::listobject::w_list_install_bytes_items(
            crate::gc_roots::shadow_stack_get(base),
            &blocks,
        )
    {
        return true;
    }
    if let Some(chars) = w_set_listview_ascii(crate::gc_roots::shadow_stack_get(base + 1))
        && crate::listobject::w_list_install_ascii_items(
            crate::gc_roots::shadow_stack_get(base),
            &chars,
        )
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictmultiobject::DictKeyError;
    use crate::intobject::w_int_new;

    fn shell_words(ptr: crate::gc_hook::GCREF) -> (i64, i64) {
        unsafe {
            let base = ptr as *const i64;
            (*base, *base.add(1))
        }
    }

    #[test]
    fn set_strategy_result_shells_keep_payload_words() {
        let ok_unit = super::set_strategy_add_shell(Ok(()));
        let ok_false = super::set_strategy_bool_shell(Ok(false));
        let ok_true = super::set_strategy_bool_shell(Ok(true));
        let err_key = super::set_strategy_add_shell(Err(SetUpdateError::Key(DictKeyError)));
        let err_size = super::set_strategy_add_shell(Err(SetUpdateError::ChangedSize));
        let err_dict = super::set_strategy_bool_shell(Err(DictKeyError));

        assert_eq!(ok_unit, ok_false);
        assert_ne!(ok_true, ok_false);
        assert_ne!(err_key, ok_unit);
        assert_ne!(err_size, err_key);
        assert_ne!(err_dict, err_key);
        assert_ne!(err_dict, ok_true);

        assert_eq!(shell_words(ok_unit), (0, 0));
        assert_eq!(shell_words(ok_true), (0, 1));
        assert_eq!(shell_words(err_key), (1, 0));
        assert_eq!(shell_words(err_size), (1, 1));
        let (discriminant, payload) = shell_words(err_dict);
        assert_eq!(discriminant, 1);
        assert_eq!(
            payload as usize as *const DictKeyError,
            std::ptr::from_ref(&super::DICT_KEY_ERROR_UNIT)
        );
    }

    fn install_test_hash_hook() {
        unsafe fn hash_int(obj: PyObjectRef) -> i64 {
            // Bool is not a plain int (`is_plain_int1`). A str must not be
            // read as `W_IntObject`. Exact ints hash to their value here;
            // production `hash_w` is `intobject.py _hash_int`. Bytes and str
            // hash by content, the way `hash_w` does, so a wrapped key's
            // digest can be compared with `object_key_for`.
            if crate::is_bool(obj) {
                return crate::w_bool_get_value(obj) as i64;
            }
            if crate::py_type_check(obj, &crate::INT_TYPE) {
                return crate::w_int_get_value(obj);
            }
            if crate::is_exact_type(obj, &crate::BYTES_TYPE) {
                return hash_test_bytes(crate::w_bytes_data(obj));
            }
            if crate::is_str(obj) {
                return hash_test_bytes(crate::w_str_get_wtf8(obj).as_bytes());
            }
            0
        }

        fn hash_test_bytes(bytes: &[u8]) -> i64 {
            let mut hash: i64 = 0x345678;
            for &byte in bytes {
                hash = hash.wrapping_mul(1_000_003).wrapping_add(byte as i64);
            }
            if hash == -1 { -2 } else { hash }
        }

        unsafe fn hash_str(_ptr: *const u8, _len: usize) -> i64 {
            0
        }

        crate::dict_eq_hook::register_hash_w_hook(hash_int);
        crate::dict_eq_hook::register_hash_str_hook(hash_str);
    }

    #[test]
    fn add_dedupes() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Empty
            );
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(w_set_contains(s, w_int_new(2)));
            assert!(!w_set_contains(s, w_int_new(3)));
        }
    }

    #[test]
    fn plain_int_hit_does_not_insert() {
        install_test_hash_hook();
        let s = w_set_new();
        let one = w_int_new(1);
        assert!(!plain_int_already_in_int_set(std::ptr::null_mut(), one));
        assert!(!plain_int_already_in_int_set(s, std::ptr::null_mut()));
        assert!(!plain_int_already_in_int_set(s, one));
        unsafe { w_set_add(s, one) };
        assert!(plain_int_already_in_int_set(s, one));
        assert!(!plain_int_already_in_int_set(s, w_int_new(2)));
        assert!(!plain_int_already_in_int_set(s, crate::w_bool_from(true)));
        assert_eq!(unsafe { w_set_len(s) }, 1);
        unsafe { w_set_add(s, crate::w_bool_from(true)) };
        assert_eq!(
            unsafe { (*(s as *const W_SetObject)).strategy.kind },
            SetStrategyKind::Object
        );
        assert!(!plain_int_already_in_int_set(s, one));
    }

    #[test]
    fn content_gen_is_stable_across_a_hit_add() {
        install_test_hash_hook();
        let a = w_set_new();
        let b = w_set_new();
        unsafe {
            let a_obj = &*(a as *const W_SetObject);
            let b_obj = &*(b as *const W_SetObject);
            assert_ne!(a_obj.set_id, 0);
            assert_ne!(a_obj.set_id, b_obj.set_id);
            let id = a_obj.set_id;
            let gen_empty = a_obj.content_gen_relaxed();

            w_set_add(a, w_int_new(1));
            let gen_inserted = (*(a as *const W_SetObject)).content_gen_relaxed();
            assert_ne!(gen_inserted, gen_empty);
            assert_eq!((*(a as *const W_SetObject)).set_id, id);

            w_set_add(a, w_int_new(1));
            assert_eq!(
                (*(a as *const W_SetObject)).content_gen_relaxed(),
                gen_inserted
            );
            assert!(plain_int_already_in_int_set(a, w_int_new(1)));

            w_set_discard(a, w_int_new(1));
            let gen_removed = (*(a as *const W_SetObject)).content_gen_relaxed();
            assert_ne!(gen_removed, gen_inserted);

            w_set_add(a, w_int_new(2));
            let gen_readded = (*(a as *const W_SetObject)).content_gen_relaxed();
            w_set_clear(a);
            assert_ne!(
                (*(a as *const W_SetObject)).content_gen_relaxed(),
                gen_readded
            );
            assert_eq!((*(a as *const W_SetObject)).set_id, id);

            w_set_add(a, crate::w_str_new("a"));
            let gen_str = (*(a as *const W_SetObject)).content_gen_relaxed();
            w_set_add(a, crate::w_str_new("a"));
            assert_eq!((*(a as *const W_SetObject)).content_gen_relaxed(), gen_str);
            assert_eq!(
                (*(a as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Ascii
            );
        }
    }

    #[test]
    fn discard_removes() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert!(w_set_discard(s, w_int_new(1)));
            assert!(!w_set_discard(s, w_int_new(99)));
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, w_int_new(2)));
        }
    }

    #[test]
    fn discard_from_the_front_keeps_later_members() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            for i in 0..64 {
                w_set_add(s, w_int_new(i));
            }
            for i in 0..32 {
                assert!(w_set_discard(s, w_int_new(i)));
            }
            assert_eq!(w_set_len(s), 32);
            for i in 0..32 {
                assert!(!w_set_contains(s, w_int_new(i)));
            }
            for i in 32..64 {
                assert!(w_set_contains(s, w_int_new(i)));
            }
        }
    }

    #[test]
    fn frozenset_distinct_type() {
        let s = w_set_new();
        let fs = w_frozenset_new();
        unsafe {
            assert!(is_set(s));
            assert!(!is_frozenset(s));
            assert!(is_frozenset(fs));
            assert!(!is_set(fs));
        }
    }

    /// `typedef.py` `_getusercls(W_SetObject)`: a subclass instance is
    /// `W_SetObjectUser` carrying `SET_USER_TYPE` or `FROZENSET_USER_TYPE`.
    #[test]
    fn set_subclass_instance_carries_user_typeptr() {
        assert_eq!(W_SET_USER_GC_TYPE_ID, 166);
        assert_eq!(
            W_SET_OBJECT_SIZE,
            std::mem::offset_of!(W_SetObject, lifeline) + std::mem::size_of::<PyObjectRef>()
        );
        assert_eq!(
            W_SET_USER_OBJECT_SIZE,
            W_SET_OBJECT_SIZE
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<*mut crate::object_array::ItemsBlock>()
        );
        let obj = w_set_new();
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &SET_TYPE));
            assert!(is_set(obj));
            assert!(!is_frozenset(obj));
        }
        let obj = w_frozenset_new();
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &FROZENSET_TYPE));
            assert!(is_frozenset(obj));
            assert!(!is_set(obj));
        }
        let obj = w_set_user_new_empty(get_instantiate(&SET_TYPE), false);
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &SET_USER_TYPE));
            assert!(is_set(obj));
            assert!(!is_frozenset(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &SET_TYPE));
            assert_eq!(w_set_len(obj), 0);
        }
        let obj = w_set_user_new_empty(get_instantiate(&FROZENSET_TYPE), true);
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &FROZENSET_USER_TYPE));
            assert!(is_frozenset(obj));
            assert!(!is_set(obj));
            assert!(!crate::pyobject::is_exact_type(obj, &FROZENSET_TYPE));
            assert_eq!(w_set_len(obj), 0);
        }
    }

    #[test]
    fn fresh_set_is_empty_with_null_storage() {
        let s = w_set_new();
        let fs = w_frozenset_new();
        unsafe {
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.strategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert_eq!(w_set_len(s), 0);
            let frozen = &*(fs as *const W_SetObject);
            assert_eq!(frozen.strategy.kind, SetStrategyKind::Empty);
            assert!(frozen.sstorage.is_null());
            assert_eq!(w_set_len(fs), 0);
        }
    }

    #[test]
    fn first_plain_int_add_promotes_to_int() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.strategy.kind, SetStrategyKind::Int);
            assert!(!set.sstorage.is_null());
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, w_int_new(1)));
        }
    }

    #[test]
    fn clear_returns_to_empty() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            w_set_clear(s);
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.strategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert_eq!(w_set_len(s), 0);
            assert!(!w_set_contains(s, w_int_new(1)));
            // `EmptySetStrategy.clear` is a no-op.
            w_set_clear(s);
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(s as *const W_SetObject)).sstorage.is_null());
        }
    }

    #[test]
    fn discard_of_last_element_returns_to_empty() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert!(w_set_discard(s, w_int_new(1)));
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );
            assert!(w_set_discard(s, w_int_new(2)));
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.strategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert_eq!(w_set_len(s), 0);
        }
    }

    #[test]
    fn contains_and_discard_on_empty_are_false() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            assert!(!w_set_contains(s, w_int_new(1)));
            assert!(!w_set_discard(s, w_int_new(1)));
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.strategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert!(w_set_popitem(s).is_none());
            assert!(w_set_next_slot(s, 0).is_none());
            assert!(w_set_key_at(s, 0).is_none());
            assert!(w_set_items(s).is_empty());
            assert!(w_set_stored_hashes(s).is_empty());
            assert_eq!(w_set_num_ever_used_items(s), 0);
            assert!(w_set_iterkey_at(s, 0).is_null());
            assert_eq!(w_set_iterkey_hash_at(s, 0), 0);
        }
    }

    #[test]
    fn empty_update_copy_and_difference_do_not_deref_null() {
        install_test_hash_hook();
        unsafe {
            let empty = w_set_new();
            let other = w_set_new();
            w_set_add(other, w_int_new(7));
            assert!(w_set_update_from_set(empty, other).is_ok());
            assert_eq!(
                (*(empty as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );
            assert_eq!(w_set_len(empty), 1);
            assert!(w_set_contains(empty, w_int_new(7)));

            let empty2 = w_set_new();
            assert!(w_set_update_from_set(other, empty2).is_ok());
            assert_eq!(w_set_len(other), 1);
            assert_eq!(
                (*(other as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );

            let dst = w_set_new();
            w_set_add(dst, w_int_new(1));
            w_set_copy_storage_from(dst, empty2);
            assert_eq!(
                (*(dst as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(dst as *const W_SetObject)).sstorage.is_null());
            assert_eq!(w_set_len(dst), 0);

            assert!(w_set_difference_update_from_set(other, empty2).is_ok());
            assert_eq!(w_set_len(other), 1);
            assert!(w_set_difference_update_from_set(empty2, other).is_ok());
            assert_eq!(
                (*(empty2 as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(empty2 as *const W_SetObject)).sstorage.is_null());

            // `AbstractUnwrappedSetStrategy.popitem` keeps the strategy when
            // the dict underneath becomes empty.
            let popped = w_set_popitem(other);
            assert!(popped.is_some());
            assert_eq!(w_set_len(other), 0);
            assert_eq!(
                (*(other as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );
            assert!(!(*(other as *const W_SetObject)).sstorage.is_null());
            // Same storage (`s -= s`) goes through `w_set_clear`.
            assert!(w_set_difference_update_from_set(other, other).is_ok());
            assert_eq!(
                (*(other as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(other as *const W_SetObject)).sstorage.is_null());
        }
    }

    #[test]
    fn integer_strategy_switches_to_object_without_renumbering_slots() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            w_set_add(s, w_int_new(1));
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );
            w_set_add(s, w_int_new(2));
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Int
            );
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(!w_set_contains(s, w_int_new(99)));
            w_set_add(s, w_int_new(3));
            assert!(w_set_discard(s, w_int_new(2)));
            let slot_a = w_set_next_slot(s, 0).unwrap();
            let key_a = w_set_key_at(s, slot_a).unwrap();
            assert_eq!(crate::w_int_get_value(key_a.obj), 1);
            assert_eq!(key_a.hash, 1);
            let slot_b = w_set_next_slot(s, slot_a + 1).unwrap();
            let key_b = w_set_key_at(s, slot_b).unwrap();
            assert_eq!(crate::w_int_get_value(key_b.obj), 3);
            let hole = slot_a + 1;
            if hole != slot_b {
                assert!(w_set_key_at(s, hole).is_none());
            }
            w_set_add(s, crate::w_str_new("x"));
            assert_eq!(
                (*(s as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Object
            );
            assert_eq!(w_set_len(s), 3);
            assert_eq!(
                crate::w_int_get_value(w_set_key_at(s, slot_a).unwrap().obj),
                1
            );
            assert_eq!(
                crate::w_int_get_value(w_set_key_at(s, slot_b).unwrap().obj),
                3
            );
            if hole != slot_b {
                assert!(w_set_key_at(s, hole).is_none());
            }
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(w_set_contains(s, w_int_new(3)));
            assert!(w_set_contains(s, crate::w_str_new("x")));
            assert!(!w_set_contains(s, w_int_new(2)));

            let t = w_set_new();
            w_set_add(t, w_int_new(5));
            assert!(w_set_discard(t, w_int_new(5)));
            assert_eq!(
                (*(t as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(t as *const W_SetObject)).sstorage.is_null());

            let p = w_set_new();
            w_set_add(p, w_int_new(9));
            let popped = w_set_popitem(p).unwrap();
            assert!(crate::is_int(popped));
            assert_eq!(crate::w_int_get_value(popped), 9);

            // `is_plain_int1` rejects bool, so `{True}` stays an object set.
            let b = w_set_new();
            w_set_add(b, crate::w_bool_from(true));
            assert_eq!(
                (*(b as *const W_SetObject)).strategy.kind,
                SetStrategyKind::Object
            );
        }
    }

    fn strategy_kind(obj: PyObjectRef) -> SetStrategyKind {
        unsafe { (*(obj as *const W_SetObject)).strategy.kind }
    }

    #[test]
    fn first_exact_bytes_add_installs_bytes() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            let first = crate::w_bytes_from_bytes(b"ab");
            w_set_add(s, first);
            w_set_add(s, crate::w_bytes_from_bytes(b"ab"));
            w_set_add(s, crate::w_bytes_from_bytes(b"cd"));
            assert_eq!(strategy_kind(s), SetStrategyKind::Bytes);
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, crate::w_bytes_from_bytes(b"ab")));
            assert!(w_set_contains(s, crate::w_bytes_from_bytes(b"cd")));
            assert!(!w_set_contains(s, crate::w_bytes_from_bytes(b"ef")));
            let slot = w_set_next_slot(s, 0).unwrap();
            let key = w_set_key_at(s, slot).unwrap();
            assert_eq!(crate::w_bytes_data(key.obj), b"ab");
            assert_eq!(key.hash, crate::dictmultiobject::object_key_for(first).hash);
            assert_eq!(
                key.hash,
                crate::dictmultiobject::object_key_for(key.obj).hash
            );
            assert!(w_set_discard(s, crate::w_bytes_from_bytes(b"cd")));
            assert_eq!(w_set_len(s), 1);
            assert!(!w_set_discard(s, crate::w_bytes_from_bytes(b"zz")));
            let popped = w_set_popitem(s).unwrap();
            assert_eq!(crate::w_bytes_data(popped), b"ab");
            assert_eq!(w_set_len(s), 0);
            // `AbstractUnwrappedSetStrategy.popitem` keeps the strategy.
            assert_eq!(strategy_kind(s), SetStrategyKind::Bytes);
        }
    }

    #[test]
    fn first_ascii_str_add_installs_ascii() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            let first = crate::w_str_new("ab");
            w_set_add(s, first);
            w_set_add(s, crate::w_str_new("ab"));
            w_set_add(s, crate::w_str_new("cd"));
            assert_eq!(strategy_kind(s), SetStrategyKind::Ascii);
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, crate::w_str_new("ab")));
            assert!(w_set_contains(s, crate::w_str_new("cd")));
            assert!(!w_set_contains(s, crate::w_str_new("ef")));
            let slot = w_set_next_slot(s, 0).unwrap();
            let key = w_set_key_at(s, slot).unwrap();
            assert_eq!(crate::w_str_get_wtf8(key.obj).as_bytes(), b"ab");
            assert_eq!(key.hash, crate::dictmultiobject::object_key_for(first).hash);
            assert_eq!(
                key.hash,
                crate::dictmultiobject::object_key_for(key.obj).hash
            );
            assert!(w_set_discard(s, crate::w_str_new("cd")));
            let popped = w_set_popitem(s).unwrap();
            assert_eq!(crate::w_str_get_wtf8(popped).as_bytes(), b"ab");
            assert_eq!(w_set_len(s), 0);
            assert_eq!(strategy_kind(s), SetStrategyKind::Ascii);
        }
    }

    #[test]
    fn first_non_ascii_str_installs_object() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            w_set_add(s, crate::w_str_new("\u{00e9}"));
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, crate::w_str_new("\u{00e9}")));
        }
    }

    #[test]
    fn discard_of_last_bytes_or_ascii_element_returns_to_empty() {
        install_test_hash_hook();
        unsafe {
            let bytes = w_set_new();
            w_set_add(bytes, crate::w_bytes_from_bytes(b"ab"));
            w_set_add(bytes, crate::w_bytes_from_bytes(b"cd"));
            assert!(w_set_discard(bytes, crate::w_bytes_from_bytes(b"ab")));
            assert_eq!(strategy_kind(bytes), SetStrategyKind::Bytes);
            assert!(w_set_discard(bytes, crate::w_bytes_from_bytes(b"cd")));
            assert_eq!(strategy_kind(bytes), SetStrategyKind::Empty);
            assert!((*(bytes as *const W_SetObject)).sstorage.is_null());
            assert_eq!(w_set_len(bytes), 0);

            let ascii = w_set_new();
            w_set_add(ascii, crate::w_str_new("ab"));
            assert!(w_set_discard(ascii, crate::w_str_new("ab")));
            assert_eq!(strategy_kind(ascii), SetStrategyKind::Empty);
            assert!((*(ascii as *const W_SetObject)).sstorage.is_null());
            assert_eq!(w_set_len(ascii), 0);
        }
    }

    #[test]
    fn bytes_strategy_switches_to_object_without_renumbering_slots() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            w_set_add(s, crate::w_bytes_from_bytes(b"a"));
            w_set_add(s, crate::w_bytes_from_bytes(b"b"));
            w_set_add(s, crate::w_bytes_from_bytes(b"c"));
            assert!(w_set_discard(s, crate::w_bytes_from_bytes(b"b")));
            assert_eq!(strategy_kind(s), SetStrategyKind::Bytes);
            let slot_a = w_set_next_slot(s, 0).unwrap();
            let key_a = w_set_key_at(s, slot_a).unwrap();
            assert_eq!(crate::w_bytes_data(key_a.obj), b"a");
            let slot_c = w_set_next_slot(s, slot_a + 1).unwrap();
            let key_c = w_set_key_at(s, slot_c).unwrap();
            assert_eq!(crate::w_bytes_data(key_c.obj), b"c");
            let hole = slot_a + 1;
            if hole != slot_c {
                assert!(w_set_key_at(s, hole).is_none());
            }
            w_set_add(s, w_int_new(7));
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert_eq!(w_set_len(s), 3);
            assert_eq!(
                crate::w_bytes_data(w_set_key_at(s, slot_a).unwrap().obj),
                b"a"
            );
            assert_eq!(
                crate::w_bytes_data(w_set_key_at(s, slot_c).unwrap().obj),
                b"c"
            );
            if hole != slot_c {
                assert!(w_set_key_at(s, hole).is_none());
            }
            assert!(w_set_contains(s, crate::w_bytes_from_bytes(b"a")));
            assert!(w_set_contains(s, crate::w_bytes_from_bytes(b"c")));
            assert!(w_set_contains(s, w_int_new(7)));
            assert!(!w_set_contains(s, crate::w_bytes_from_bytes(b"b")));
            let stored = w_set_key_at(s, slot_a).unwrap();
            assert_eq!(
                stored.hash,
                crate::dictmultiobject::object_key_for(crate::w_bytes_from_bytes(b"a")).hash
            );
        }
    }

    #[test]
    fn ascii_strategy_switches_to_object_without_renumbering_slots() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            w_set_add(s, crate::w_str_new("a"));
            w_set_add(s, crate::w_str_new("b"));
            w_set_add(s, crate::w_str_new("c"));
            assert!(w_set_discard(s, crate::w_str_new("b")));
            assert_eq!(strategy_kind(s), SetStrategyKind::Ascii);
            let slot_a = w_set_next_slot(s, 0).unwrap();
            assert_eq!(
                crate::w_str_get_wtf8(w_set_key_at(s, slot_a).unwrap().obj).as_bytes(),
                b"a"
            );
            let slot_c = w_set_next_slot(s, slot_a + 1).unwrap();
            assert_eq!(
                crate::w_str_get_wtf8(w_set_key_at(s, slot_c).unwrap().obj).as_bytes(),
                b"c"
            );
            let hole = slot_a + 1;
            if hole != slot_c {
                assert!(w_set_key_at(s, hole).is_none());
            }
            w_set_add(s, crate::w_bytes_from_bytes(b"z"));
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert_eq!(w_set_len(s), 3);
            assert_eq!(
                crate::w_str_get_wtf8(w_set_key_at(s, slot_a).unwrap().obj).as_bytes(),
                b"a"
            );
            assert_eq!(
                crate::w_str_get_wtf8(w_set_key_at(s, slot_c).unwrap().obj).as_bytes(),
                b"c"
            );
            if hole != slot_c {
                assert!(w_set_key_at(s, hole).is_none());
            }
            assert!(w_set_contains(s, crate::w_str_new("a")));
            assert!(w_set_contains(s, crate::w_str_new("c")));
            assert!(w_set_contains(s, crate::w_bytes_from_bytes(b"z")));
            assert!(!w_set_contains(s, crate::w_str_new("b")));
            let stored = w_set_key_at(s, slot_a).unwrap();
            assert_eq!(
                stored.hash,
                crate::dictmultiobject::object_key_for(crate::w_str_new("a")).hash
            );
        }
    }

    #[test]
    fn w_set_gc_type_id_matches_descr() {
        assert_eq!(W_SET_GC_TYPE_ID, 30);
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::type_id(),
            W_SET_GC_TYPE_ID
        );
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::SIZE,
            W_SET_OBJECT_SIZE
        );
    }

    struct ClearComparesByIdentityHook;

    impl Drop for ClearComparesByIdentityHook {
        fn drop(&mut self) {
            crate::dict_eq_hook::clear_compares_by_identity_hook();
        }
    }

    /// `ComparesByIdentityHookFn`: true only for `COMPARES_BY_IDENTITY_YES`.
    /// The guard clears the hook before the test returns.
    fn install_compares_by_identity_hook() -> ClearComparesByIdentityHook {
        unsafe fn hook(w_type: PyObjectRef) -> bool {
            crate::w_type_compares_by_identity_status(w_type) == crate::COMPARES_BY_IDENTITY_YES
        }
        crate::dict_eq_hook::register_compares_by_identity_hook(hook);
        ClearComparesByIdentityHook
    }

    fn new_ident_type(status: Option<u8>) -> PyObjectRef {
        let w_type = crate::w_type_new("Ident", crate::PY_NULL, std::ptr::null_mut());
        if let Some(status) = status {
            unsafe {
                crate::w_type_set_compares_by_identity_status(w_type, status);
            }
        }
        w_type
    }

    #[test]
    fn identity_may_contain_equal_elements_excludes_int_bytes_ascii_empty() {
        assert!(!IDENTITY_SET_STRATEGY.may_contain_equal_elements(SetStrategyKind::Empty));
        assert!(!IDENTITY_SET_STRATEGY.may_contain_equal_elements(SetStrategyKind::Int));
        assert!(!IDENTITY_SET_STRATEGY.may_contain_equal_elements(SetStrategyKind::Bytes));
        assert!(!IDENTITY_SET_STRATEGY.may_contain_equal_elements(SetStrategyKind::Ascii));
        assert!(IDENTITY_SET_STRATEGY.may_contain_equal_elements(SetStrategyKind::Object));
        assert!(IDENTITY_SET_STRATEGY.may_contain_equal_elements(SetStrategyKind::Identity));
    }

    #[test]
    fn fresh_set_stays_empty_and_identity_add_keeps_pointers() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let first = crate::w_instance_new(w_type);
        let second = crate::w_instance_new(w_type);
        let third = crate::w_instance_new(w_type);
        let s = w_set_new();
        unsafe {
            assert_eq!(strategy_kind(s), SetStrategyKind::Empty);
            w_set_add(s, first);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 1);
            w_set_add(s, first);
            assert_eq!(w_set_len(s), 1);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            w_set_add(s, second);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 2);
            w_set_add(s, first);
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, first));
            assert!(w_set_contains(s, second));
            assert!(!w_set_contains(s, third));
            let items = w_set_items(s);
            assert_eq!(items.len(), 2);
            assert!(items.contains(&first));
            assert!(items.contains(&second));
        }
    }

    #[test]
    fn identity_set_switches_to_object_for_int_or_str() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        unsafe {
            let by_int = w_set_new();
            let first = crate::w_instance_new(w_type);
            let second = crate::w_instance_new(w_type);
            w_set_add(by_int, first);
            w_set_add(by_int, second);
            assert_eq!(strategy_kind(by_int), SetStrategyKind::Identity);
            w_set_add(by_int, w_int_new(1));
            assert_eq!(strategy_kind(by_int), SetStrategyKind::Object);
            assert_eq!(w_set_len(by_int), 3);
            assert!(w_set_contains(by_int, first));
            assert!(w_set_contains(by_int, second));
            assert!(w_set_contains(by_int, w_int_new(1)));

            let by_str = w_set_new();
            let third = crate::w_instance_new(w_type);
            w_set_add(by_str, third);
            assert_eq!(strategy_kind(by_str), SetStrategyKind::Identity);
            w_set_add(by_str, crate::w_str_new("a"));
            assert_eq!(strategy_kind(by_str), SetStrategyKind::Object);
            assert_eq!(w_set_len(by_str), 2);
            assert!(w_set_contains(by_str, third));
            assert!(w_set_contains(by_str, crate::w_str_new("a")));
        }
    }

    #[test]
    fn compares_by_identity_no_or_unset_installs_object() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        unsafe {
            let unset = new_ident_type(None);
            let unset_inst = crate::w_instance_new(unset);
            let s = w_set_new();
            w_set_add(s, unset_inst);
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, unset_inst));

            let no = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_NO));
            let no_inst = crate::w_instance_new(no);
            let t = w_set_new();
            w_set_add(t, no_inst);
            assert_eq!(strategy_kind(t), SetStrategyKind::Object);
            assert!(w_set_contains(t, no_inst));
        }
    }

    #[test]
    fn no_compares_by_identity_hook_stays_on_object() {
        install_test_hash_hook();
        crate::dict_eq_hook::clear_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        unsafe {
            assert!(crate::dict_eq_hook::try_compares_by_identity(w_type).is_none());
            let inst = crate::w_instance_new(w_type);
            let s = w_set_new();
            assert_eq!(strategy_kind(s), SetStrategyKind::Empty);
            w_set_add(s, inst);
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, inst));
        }
    }

    #[test]
    fn identity_difference_against_int_set_stays_identity() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let first = crate::w_instance_new(w_type);
        let second = crate::w_instance_new(w_type);
        unsafe {
            let ident = w_set_new();
            w_set_add(ident, first);
            w_set_add(ident, second);
            let ints = w_set_new();
            w_set_add(ints, w_int_new(1));
            w_set_add(ints, w_int_new(2));
            assert!(w_set_difference_update_from_set(ident, ints).is_ok());
            assert_eq!(strategy_kind(ident), SetStrategyKind::Identity);
            assert_eq!(w_set_len(ident), 2);
            assert!(w_set_contains(ident, first));
            assert!(w_set_contains(ident, second));
        }
    }

    #[test]
    fn identity_update_from_identity_stays_identity() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let a = crate::w_instance_new(w_type);
        let b = crate::w_instance_new(w_type);
        let c = crate::w_instance_new(w_type);
        unsafe {
            let left = w_set_new();
            w_set_add(left, a);
            w_set_add(left, b);
            let right = w_set_new();
            w_set_add(right, b);
            w_set_add(right, c);
            assert!(w_set_update_from_set(left, right).is_ok());
            assert_eq!(strategy_kind(left), SetStrategyKind::Identity);
            assert_eq!(w_set_len(left), 3);
            assert!(w_set_contains(left, a));
            assert!(w_set_contains(left, b));
            assert!(w_set_contains(left, c));
        }
    }

    #[test]
    fn identity_status_flip_to_no_switches_on_next_add() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let inst = crate::w_instance_new(w_type);
        unsafe {
            let s = w_set_new();
            w_set_add(s, inst);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 1);
            crate::w_type_set_compares_by_identity_status(w_type, crate::COMPARES_BY_IDENTITY_NO);
            let extra = crate::w_instance_new(w_type);
            w_set_add(s, extra);
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert!(w_set_contains(s, inst));
            assert!(w_set_contains(s, extra));
            assert_eq!(w_set_len(s), 2);
        }
    }

    use std::cell::Cell;

    thread_local! {
        static INSTANCE_HASH: Cell<i64> = const { Cell::new(11) };
        static CLEAR_SET: Cell<PyObjectRef> = const { Cell::new(std::ptr::null_mut()) };
        static RAISE_INSTANCE_HASH: Cell<bool> = const { Cell::new(false) };
    }

    unsafe fn identity_test_hash(obj: PyObjectRef) -> i64 {
        if crate::is_bool(obj) {
            return crate::w_bool_get_value(obj) as i64;
        }
        if crate::py_type_check(obj, &crate::INT_TYPE) {
            return crate::w_int_get_value(obj);
        }
        if crate::is_exact_type(obj, &crate::BYTES_TYPE) || crate::is_str(obj) {
            return 2;
        }
        let clear = CLEAR_SET.with(|cell| cell.replace(std::ptr::null_mut()));
        if !clear.is_null() {
            w_set_clear(clear);
        }
        if RAISE_INSTANCE_HASH.with(|cell| cell.get()) {
            crate::dict_eq_hook::signal_hash_error(obj);
            return 0;
        }
        INSTANCE_HASH.with(|cell| cell.get())
    }

    unsafe fn identity_test_hash_str(_ptr: *const u8, _len: usize) -> i64 {
        0
    }

    fn install_identity_hash_hook() {
        INSTANCE_HASH.with(|cell| cell.set(11));
        CLEAR_SET.with(|cell| cell.set(std::ptr::null_mut()));
        RAISE_INSTANCE_HASH.with(|cell| cell.set(false));
        crate::dict_eq_hook::register_hash_w_hook(identity_test_hash);
        crate::dict_eq_hook::register_hash_str_hook(identity_test_hash_str);
    }

    #[test]
    fn identity_set_keeps_insertion_hash_after_hook_changes() {
        install_identity_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let inst = crate::w_instance_new(w_type);
        let extra = crate::w_instance_new(w_type);
        unsafe {
            let s = w_set_new();
            w_set_add(s, inst);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            let slot = w_set_next_slot(s, 0).unwrap();
            assert_eq!(w_set_key_at(s, slot).unwrap().hash, 11);
            assert_eq!(w_set_stored_hashes(s), vec![11]);

            INSTANCE_HASH.with(|cell| cell.set(99));
            assert_eq!(w_set_stored_hashes(s), vec![11]);
            assert_eq!(w_set_key_at(s, slot).unwrap().hash, 11);
            assert_eq!(w_set_iterkey_hash_at(s, slot), 11);
            assert!(w_set_contains(s, inst));

            let copied = w_set_new();
            w_set_copy_storage_from(copied, s);
            assert_eq!(strategy_kind(copied), SetStrategyKind::Identity);
            assert_eq!(w_set_stored_hashes(copied), vec![11]);

            let right = w_set_new();
            w_set_add(right, extra);
            assert_eq!(w_set_stored_hashes(right), vec![99]);
            assert!(w_set_update_from_set(s, right).is_ok());
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 2);
            let hashes = w_set_stored_hashes(s);
            assert_eq!(hashes, vec![11, 99]);

            let other = w_set_new();
            w_set_add(other, w_int_new(1));
            w_set_add(other, crate::w_str_new("a"));
            assert_eq!(strategy_kind(other), SetStrategyKind::Object);
            INSTANCE_HASH.with(|cell| cell.set(11));
            let kept = w_set_new();
            w_set_add(kept, inst);
            assert_eq!(w_set_stored_hashes(kept), vec![11]);
            INSTANCE_HASH.with(|cell| cell.set(99));
            assert!(w_set_difference_update_from_set(kept, other).is_ok());
            assert_eq!(strategy_kind(kept), SetStrategyKind::Identity);
            assert_eq!(w_set_len(kept), 1);
            assert_eq!(w_set_stored_hashes(kept), vec![11]);
            assert!(w_set_contains(kept, inst));
        }
    }

    #[test]
    fn identity_contains_survives_hash_that_clears_the_set() {
        install_identity_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let inst = crate::w_instance_new(w_type);
        unsafe {
            let s = w_set_new();
            w_set_add(s, inst);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 1);
            CLEAR_SET.with(|cell| cell.set(s));
            assert!(!w_set_contains(s, w_int_new(1)));
            assert_eq!(w_set_len(s), 1);
            assert_eq!(strategy_kind(s), SetStrategyKind::Object);
            assert!(w_set_contains(s, inst));
            assert!(!w_set_contains(s, w_int_new(1)));
            let items = w_set_items(s);
            assert_eq!(items, vec![inst]);
        }
    }

    #[test]
    fn null_w_class_uses_instantiate_for_identity() {
        install_test_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let yes = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let no = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_NO));
        let yes_tp = Box::leak(Box::new(crate::new_pytype("NullClassIdent")));
        let no_tp = Box::leak(Box::new(crate::new_pytype("NullClassValue")));
        let bare_tp = Box::leak(Box::new(crate::new_pytype("NullClassBare")));
        crate::set_instantiate(yes_tp, yes);
        crate::set_instantiate(no_tp, no);
        unsafe {
            let ident = crate::w_instance_new(yes);
            (*ident).w_class = std::ptr::null_mut();
            (*ident).ob_type = yes_tp;
            assert!(IDENTITY_SET_STRATEGY.is_correct_type(ident));
            let s = w_set_new();
            w_set_add(s, ident);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, ident));

            let valued = crate::w_instance_new(no);
            (*valued).w_class = std::ptr::null_mut();
            (*valued).ob_type = no_tp;
            assert!(!IDENTITY_SET_STRATEGY.is_correct_type(valued));
            let t = w_set_new();
            w_set_add(t, valued);
            assert_eq!(strategy_kind(t), SetStrategyKind::Object);

            let bare = crate::w_instance_new(yes);
            (*bare).w_class = std::ptr::null_mut();
            (*bare).ob_type = bare_tp;
            assert!(!IDENTITY_SET_STRATEGY.is_correct_type(bare));
            let u = w_set_new();
            w_set_add(u, bare);
            assert_eq!(strategy_kind(u), SetStrategyKind::Object);
        }
    }

    #[test]
    fn identity_switch_leaves_strategy_when_hash_raises() {
        install_identity_hash_hook();
        let _hook = install_compares_by_identity_hook();
        let w_type = new_ident_type(Some(crate::COMPARES_BY_IDENTITY_YES));
        let inst = crate::w_instance_new(w_type);
        struct ResetRaise;
        impl Drop for ResetRaise {
            fn drop(&mut self) {
                RAISE_INSTANCE_HASH.with(|cell| cell.set(false));
            }
        }
        let _reset = ResetRaise;
        unsafe {
            let s = w_set_new();
            w_set_add(s, inst);
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 1);
            let item = w_int_new(1);
            let hash = identity_test_hash(item);
            RAISE_INSTANCE_HASH.with(|cell| cell.set(true));
            let err = w_set_add_hashed_checked(s, item, hash);
            assert!(err.is_err());
            assert_eq!(strategy_kind(s), SetStrategyKind::Identity);
            assert_eq!(w_set_len(s), 1);
            assert_eq!(w_set_stored_hashes(s), vec![11]);
            assert!(w_set_contains(s, inst));
        }
    }

    fn set_hash(obj: PyObjectRef) -> i64 {
        unsafe { (*(obj as *const W_SetObject)).hash }
    }

    fn list_kind(obj: PyObjectRef) -> crate::ListStrategy {
        unsafe { crate::w_list_strategy(obj) }
    }

    fn set_int_values(obj: PyObjectRef) -> Vec<i64> {
        unsafe {
            w_set_items(obj)
                .into_iter()
                .map(|item| crate::w_int_get_value(item))
                .collect()
        }
    }

    fn set_bytes_values(obj: PyObjectRef) -> Vec<Vec<u8>> {
        unsafe {
            w_set_items(obj)
                .into_iter()
                .map(|item| crate::w_bytes_data(item).to_vec())
                .collect()
        }
    }

    fn set_str_bytes(obj: PyObjectRef) -> Vec<Vec<u8>> {
        unsafe {
            w_set_items(obj)
                .into_iter()
                .map(|item| crate::w_str_get_wtf8(item).as_bytes().to_vec())
                .collect()
        }
    }

    #[test]
    fn listview_int_list_dedupes_in_insertion_order() {
        install_test_hash_hook();
        unsafe {
            let items = crate::w_list_new(vec![w_int_new(1), w_int_new(1), w_int_new(2)]);
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, items));
            assert_eq!(strategy_kind(set), SetStrategyKind::Int);
            assert_eq!(w_set_len(set), 2);
            assert_eq!(set_int_values(set), vec![1, 2]);
            assert!(w_set_contains(set, w_int_new(1)));
            assert!(w_set_contains(set, w_int_new(2)));
            assert!(!w_set_contains(set, w_int_new(3)));
            assert_eq!(set_hash(set), -1);
        }
    }

    #[test]
    fn listview_bytes_list_shares_blocks() {
        install_test_hash_hook();
        unsafe {
            let first = crate::w_bytes_from_bytes(b"a");
            let items = crate::w_list_new(vec![
                first,
                crate::w_bytes_from_bytes(b"a"),
                crate::w_bytes_from_bytes(b"b"),
            ]);
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, items));
            assert_eq!(strategy_kind(set), SetStrategyKind::Bytes);
            assert_eq!(w_set_len(set), 2);
            assert_eq!(set_bytes_values(set), vec![b"a".to_vec(), b"b".to_vec()]);
            assert!(w_set_contains(set, crate::w_bytes_from_bytes(b"a")));
            assert!(w_set_contains(set, crate::w_bytes_from_bytes(b"b")));
            let stored = w_set_listview_bytes(set).unwrap();
            let listed = crate::w_list_getitems_bytes(items).unwrap();
            assert_eq!(stored[0], listed[0]);
            assert_eq!(stored[1], listed[2]);
        }
    }

    #[test]
    fn listview_ascii_list_installs_strings() {
        install_test_hash_hook();
        unsafe {
            let items = crate::w_list_new(vec![crate::w_str_new("a"), crate::w_str_new("b")]);
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, items));
            assert_eq!(strategy_kind(set), SetStrategyKind::Ascii);
            assert_eq!(set_str_bytes(set), vec![b"a".to_vec(), b"b".to_vec()]);
            assert!(w_set_contains(set, crate::w_str_new("a")));
            assert!(!w_set_contains(set, crate::w_str_new("ab")));
        }
    }

    #[test]
    fn listview_str_splits_chars_and_empty_ascii_switches_to_object() {
        install_test_hash_hook();
        unsafe {
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, crate::w_str_new("ab")));
            assert_eq!(strategy_kind(set), SetStrategyKind::Ascii);
            assert_eq!(w_set_len(set), 2);
            assert_eq!(set_str_bytes(set), vec![b"a".to_vec(), b"b".to_vec()]);
            assert!(w_set_contains(set, crate::w_str_new("a")));
            assert!(w_set_contains(set, crate::w_str_new("b")));
            assert!(!w_set_contains(set, crate::w_str_new("ab")));

            let dup = w_set_new();
            assert!(w_set_init_from_listview(dup, crate::w_str_new("aa")));
            assert_eq!(w_set_len(dup), 1);
            assert_eq!(set_str_bytes(dup), vec![b"a".to_vec()]);

            let empty = w_set_new();
            assert!(w_set_init_from_listview(empty, crate::w_str_new("")));
            assert_eq!(strategy_kind(empty), SetStrategyKind::Ascii);
            assert_eq!(w_set_len(empty), 0);
            assert_eq!(set_hash(empty), -1);
            w_set_add(empty, w_int_new(1));
            assert_eq!(strategy_kind(empty), SetStrategyKind::Object);

            let plain = w_set_new();
            w_set_add(plain, w_int_new(1));
            assert_eq!(strategy_kind(plain), SetStrategyKind::Int);
        }
    }

    #[test]
    fn listview_range_list_installs_ints() {
        install_test_hash_hook();
        unsafe {
            let items = crate::w_list_new_range(0, 1, 3);
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, items));
            assert_eq!(strategy_kind(set), SetStrategyKind::Int);
            assert_eq!(set_int_values(set), vec![0, 1, 2]);
        }
    }

    #[test]
    fn listview_mixed_list_and_non_ascii_stay_empty() {
        unsafe {
            let mixed = crate::w_list_new(vec![w_int_new(1), crate::w_str_new("a")]);
            let set = w_set_new();
            assert!(!w_set_init_from_listview(set, mixed));
            assert_eq!(strategy_kind(set), SetStrategyKind::Empty);

            let text = w_set_new();
            assert!(!w_set_init_from_listview(
                text,
                crate::w_str_new("\u{00e9}")
            ));
            assert_eq!(strategy_kind(text), SetStrategyKind::Empty);
        }
    }

    #[test]
    fn listview_bytes_object_installs_ords_and_empty_int_switches_to_object() {
        install_test_hash_hook();
        unsafe {
            let set = w_set_new();
            assert!(w_set_init_from_listview(
                set,
                crate::w_bytes_from_bytes(b"ab")
            ));
            assert_eq!(strategy_kind(set), SetStrategyKind::Int);
            assert_eq!(set_int_values(set), vec![97, 98]);
            assert!(w_set_contains(set, w_int_new(97)));
            assert!(w_set_contains(set, w_int_new(98)));

            let empty = w_set_new();
            assert!(w_set_init_from_listview(
                empty,
                crate::w_bytes_from_bytes(b"")
            ));
            assert_eq!(strategy_kind(empty), SetStrategyKind::Int);
            assert_eq!(w_set_len(empty), 0);
            assert_eq!(set_hash(empty), -1);
            w_set_add(empty, crate::w_bytes_from_bytes(b"a"));
            assert_eq!(strategy_kind(empty), SetStrategyKind::Object);

            let plain = w_set_new();
            w_set_add(plain, crate::w_bytes_from_bytes(b"a"));
            assert_eq!(strategy_kind(plain), SetStrategyKind::Bytes);
        }
    }

    #[test]
    fn listview_int_and_bytes_dicts_install_typed_sets() {
        install_test_hash_hook();
        unsafe {
            let ints = crate::w_dict_new();
            crate::w_dict_setitem(ints, 1, crate::w_none());
            crate::w_dict_setitem(ints, 1, crate::w_none());
            crate::w_dict_setitem(ints, 2, crate::w_none());
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, ints));
            assert_eq!(strategy_kind(set), SetStrategyKind::Int);
            assert_eq!(set_int_values(set), vec![1, 2]);

            let bytes = crate::w_dict_new();
            crate::w_dict_store(bytes, crate::w_bytes_from_bytes(b"a"), crate::w_none());
            crate::w_dict_store(bytes, crate::w_bytes_from_bytes(b"b"), crate::w_none());
            let set = w_set_new();
            assert!(w_set_init_from_listview(set, bytes));
            assert_eq!(strategy_kind(set), SetStrategyKind::Bytes);
            assert_eq!(set_bytes_values(set), vec![b"a".to_vec(), b"b".to_vec()]);
        }
    }

    #[test]
    fn listview_unicode_and_object_dicts_stay_empty() {
        install_test_hash_hook();
        unsafe {
            let text = crate::w_dict_new();
            crate::w_dict_store(text, crate::w_str_new("a"), crate::w_none());
            let set = w_set_new();
            assert!(!w_set_init_from_listview(set, text));
            assert_eq!(strategy_kind(set), SetStrategyKind::Empty);

            let obj = crate::w_dict_new();
            crate::w_dict_store(obj, crate::w_none(), crate::w_none());
            let set = w_set_new();
            assert!(!w_set_init_from_listview(set, obj));
            assert_eq!(strategy_kind(set), SetStrategyKind::Empty);
        }
    }

    #[test]
    fn listview_emptied_bytes_list_round_trips_into_empty_bytes_list() {
        unsafe {
            let items = crate::w_list_new(vec![crate::w_bytes_from_bytes(b"a")]);
            assert_eq!(list_kind(items), crate::ListStrategy::Bytes);
            assert!(crate::w_list_pop(items, 0).is_some());
            assert_eq!(list_kind(items), crate::ListStrategy::Bytes);
            assert_eq!(crate::w_list_len(items), 0);

            let set = w_set_new();
            assert!(w_set_init_from_listview(set, items));
            assert_eq!(strategy_kind(set), SetStrategyKind::Bytes);
            assert_eq!(w_set_len(set), 0);
            assert_eq!(set_hash(set), -1);

            let dest = crate::w_list_new(Vec::new());
            assert_eq!(list_kind(dest), crate::ListStrategy::Empty);
            assert!(w_list_try_extend_empty_from_set(dest, set));
            assert_eq!(list_kind(dest), crate::ListStrategy::Bytes);
            assert_eq!(crate::w_list_len(dest), 0);
        }
    }

    #[test]
    fn listview_empty_int_set_leaves_empty_list_empty() {
        unsafe {
            let set = w_set_new();
            assert!(w_set_init_from_listview(
                set,
                crate::w_bytes_from_bytes(b"")
            ));
            assert_eq!(strategy_kind(set), SetStrategyKind::Int);
            let dest = crate::w_list_new(Vec::new());
            assert!(!w_list_try_extend_empty_from_set(dest, set));
            assert_eq!(list_kind(dest), crate::ListStrategy::Empty);
            assert_eq!(crate::w_list_len(dest), 0);
        }
    }

    #[test]
    fn listview_size_list_takes_int_set_and_drops_hint() {
        unsafe {
            let dest = crate::w_list_new_with_sizehint(4);
            assert_eq!(list_kind(dest), crate::ListStrategy::Size);
            assert_eq!(crate::w_list_sizehint(dest), Some(4));
            let set = w_set_new();
            assert!(w_set_init_from_listview(
                set,
                crate::w_bytes_from_bytes(b"ab")
            ));
            assert!(w_list_try_extend_empty_from_set(dest, set));
            assert_eq!(list_kind(dest), crate::ListStrategy::Integer);
            assert_eq!(crate::w_list_len(dest), 2);
            assert_eq!(crate::w_list_getitems_int(dest), Some(vec![97, 98]));
            assert!(crate::w_list_sizehint(dest).is_none());
            // Two items from an empty receiver: `list_resize` allocates 8 slots.
            assert_eq!(crate::w_list_allocated(dest), 8);
        }
    }

    #[test]
    fn listview_nonempty_list_is_left_to_the_snapshot() {
        install_test_hash_hook();
        unsafe {
            let dest = crate::w_list_new(vec![w_int_new(7)]);
            let set = w_set_new();
            w_set_add(set, w_int_new(1));
            assert!(!w_list_try_extend_empty_from_set(dest, set));
            assert_eq!(list_kind(dest), crate::ListStrategy::Integer);
            assert_eq!(crate::w_list_len(dest), 1);
            assert_eq!(crate::w_list_getitems_int(dest), Some(vec![7]));
        }
    }

    #[test]
    fn listview_frozenset_from_int_list_keeps_uncomputed_hash() {
        unsafe {
            let items = crate::w_list_new(vec![w_int_new(1), w_int_new(2)]);
            let set = w_frozenset_new();
            assert_eq!(set_hash(set), -1);
            assert!(w_set_init_from_listview(set, items));
            assert_eq!(strategy_kind(set), SetStrategyKind::Int);
            assert_eq!(w_set_len(set), 2);
            assert_eq!(set_int_values(set), vec![1, 2]);
            assert_eq!(set_hash(set), -1);
        }
    }
}
