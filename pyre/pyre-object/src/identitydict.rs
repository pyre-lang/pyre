//! `pypy/objspace/std/identitydict.py` port — strategy for custom
//! instances which compare by identity (the default unless you
//! override `__hash__`, `__eq__` or `__cmp__`).
//!
//! Selected by `EmptyDictStrategy.switch_to_correct_strategy` per
//! `dictmultiobject.py switch_to_identity_strategy` when the
//! key's type satisfies `W_TypeObject.compares_by_identity()`
//! (`typeobject.py:353-371`).  The distinction from
//! `ObjectDictStrategy` is in the key comparison: raw PyObjectRef
//! identity (`is`) instead of `dict_keys_equal` (`space.eq_w` +
//! `space.hash_w`).

#![allow(unsafe_op_in_unsafe_fn)]

use crate::dictmultiobject::DictStrategy;
use crate::pyobject::PyObjectRef;

/// `identitydict.py IdentityDictStrategy` key type — identity
/// comparison + identity hash.  Stored in
/// `IndexMap<IdentityKey, PyObjectRef>` for O(1) lookup matching
/// PyPy's `mark_dict_non_null(d={})` (`:30-32`) — RPython resolves
/// `{}` keyed on instance identity to an order-preserving identity
/// hash table at translation time.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct IdentityKey(pub PyObjectRef);

impl std::hash::Hash for IdentityKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::gc_hook::gc_identity_hash(self.0 as usize).hash(state);
    }
}

impl PartialEq for IdentityKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for IdentityKey {}

impl crate::rordereddict::EntryDummy for IdentityKey {
    fn dummy() -> Self {
        Self(std::ptr::null_mut())
    }
}

/// `IdentityDictStrategy` backing — erased identity-keyed `{}`
/// (`identitydict.py`). GC-managed storage box (mirrors the other
/// dict strategies; see `dictmultiobject::ObjectDictStorage`).
pub type IdentityDictStorage = crate::rordereddict::RDict<IdentityKey, PyObjectRef>;

/// Runtime-assigned GC type id for the [`IdentityDictStorage`] box.
static IDENTITY_DICT_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IdentityDictStorage`] box.
pub fn set_identity_dict_storage_gc_type_id(id: u32) {
    IDENTITY_DICT_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IdentityDictStorage`] box.
#[majit_macros::dont_look_inside]
pub fn identity_dict_storage_gc_type_id() -> u32 {
    IDENTITY_DICT_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Typed read accessor — `dictmultiobject.py IdentityDictStrategy.unerase
/// (w_dict.dstorage)`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`IDENTITY_DICT_STRATEGY`].
#[inline]
unsafe fn identity_storage<'a>(obj: PyObjectRef) -> &'a IdentityDictStorage {
    let dict = &*(obj as *const crate::dictmultiobject::W_DictObject);
    &*(dict.dstorage as *const IdentityDictStorage)
}

/// Internal helper: `IdentityDictStrategy::getitem` body.  Caller must have
/// already verified `is_correct_type(w_key)`.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `ll_dict_lookup` (`rordereddict.py`).  [`IdentityKey`] hashes and
/// compares by address, so the probe runs no user Python code.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`IDENTITY_DICT_STRATEGY`].
fn w_dict_lookup_identity_strategy_iff(obj: PyObjectRef, key: PyObjectRef) -> bool {
    let storage = unsafe { identity_storage(obj) };
    majit_rlib::jit::isvirtual(storage) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(w_dict_lookup_identity_strategy_iff)]
pub unsafe fn w_dict_lookup_identity_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    identity_storage(obj).get(&IdentityKey(key)).copied()
}

/// `dictmultiobject.py delitem`'s identity-keyed remove — drop the
/// entry for `key` and report whether one was there.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(i))` on
/// `_ll_dict_del` (`rordereddict.py`).  The keys-version bump stays traced.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on [`IDENTITY_DICT_STRATEGY`].
fn w_dict_delete_identity_strategy_iff(obj: PyObjectRef, key: PyObjectRef) -> bool {
    let storage = unsafe { identity_storage(obj) };
    majit_rlib::jit::isvirtual(storage) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(w_dict_delete_identity_strategy_iff)]
pub unsafe fn w_dict_delete_identity_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    identity_storage_mut(obj)
        .remove(&IdentityKey(key))
        .is_some()
}

/// `dictmultiobject.py setitem`'s identity-keyed store — insert
/// `value` under the address-identity of `key`, reporting whether the slot was
/// newly filled so the caller bumps the keys-version only on a real insert.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `_ll_dict_setitem_lookup_done` (`rordereddict.py`).  [`IdentityKey`]
/// hashes and compares by address; the keys-version bump and write barrier
/// stay traced.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on [`IDENTITY_DICT_STRATEGY`].
fn w_dict_store_identity_strategy_iff(
    obj: PyObjectRef,
    key: PyObjectRef,
    _value: PyObjectRef,
) -> bool {
    let storage = unsafe { identity_storage(obj) };
    majit_rlib::jit::isvirtual(storage) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(w_dict_store_identity_strategy_iff)]
pub unsafe fn w_dict_store_identity_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> bool {
    identity_storage_mut(obj)
        .insert(IdentityKey(key), value)
        .is_none()
}

/// `dictmultiobject.py AbstractTypedStrategy.switch_to_object_strategy`
/// instantiation for IdentityDictStrategy — `wrap` is identity (`:26-27`), so
/// the migration ports each `IdentityKey(obj)` into
/// `ObjectKey { hash: hash_w(obj), obj }` without rewrapping keys.
///
/// Residualised (`@dont_look_inside`, `rlib/jit.py`) for the reason
/// `dictmultiobject::w_dict_switch_int_to_object_strategy` is: the body is
/// `IndexMap` construction and refill end to end, and the front end has no
/// lowering for it, so there is no modellable point inside to put the boundary
/// at.
///
/// # Safety
/// `w_dict` must be a valid `W_DictObject` on [`IDENTITY_DICT_STRATEGY`].
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_switch_identity_to_object_strategy(w_dict: PyObjectRef) {
    // Rooted exactly as `w_dict_switch_int_to_object_strategy` is: the
    // receiver moves, `object_key_for` hashes, and a pair copied into a
    // stack-local map would be pre-move by the next key's hash.
    let roots = crate::gc_roots::push_roots();
    let dict_slot = roots.base();
    let w_dict = roots.pin_root(w_dict);
    let old = &*((*(w_dict as *const crate::dictmultiobject::W_DictObject)).dstorage
        as *const IdentityDictStorage);
    let len = old.len();
    let mut hashes = Vec::with_capacity(len);
    let pairs_base = dict_slot + 1;
    let mut next = 0;
    while let Some(i) = old.next_valid_slot(next) {
        let k = old.get_slot(i).unwrap().0.0;
        let object_key = crate::dictmultiobject::object_key_for(k);
        let v = *old.get_slot(i).unwrap().1;
        hashes.push(object_key.hash);
        roots.publish(&[object_key.obj, v]);
        next = i + 1;
    }
    // Box the empty `RDict` before insert. A stack array would be unrooted
    // across the stores (`_ll_malloc_entries` runs inside `insert`).
    let new_storage = crate::gc_storage::gc_alloc_young_storage_box(
        crate::dictmultiobject::object_dict_storage_new(),
        crate::dictmultiobject::object_dict_storage_gc_type_id(),
    );
    let new_map = &mut *new_storage;
    for (i, &hash) in hashes.iter().enumerate() {
        let obj = roots.get(pairs_base + 2 * i);
        let value = roots.get(pairs_base + 2 * i + 1);
        new_map.insert(crate::dictmultiobject::ObjectKey { hash, obj }, value);
    }
    crate::dictmultiobject::install_object_dict_storage(roots.get(dict_slot), new_storage);
}

#[inline]
unsafe fn identity_storage_mut<'a>(obj: PyObjectRef) -> &'a mut IdentityDictStorage {
    let dict = &mut *(obj as *mut crate::dictmultiobject::W_DictObject);
    &mut *(dict.dstorage as *mut IdentityDictStorage)
}

/// `identitydict.py IdentityDictStrategy`.
///
/// ```python
/// class IdentityDictStrategy(AbstractTypedStrategy, DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("identitydict")
///
///     def wrap(self, unwrapped):
///         return unwrapped
///     def unwrap(self, wrapped):
///         return wrapped
///
///     def get_empty_storage(self):
///         d = {}
///         mark_dict_non_null(d)
///         return self.erase(d)
///
///     def is_correct_type(self, w_obj):
///         w_type = self.space.type(w_obj)
///         return w_type.compares_by_identity()
///
///     def _never_equal_to(self, w_lookup_type):
///         return False
///
///     def w_keys(self, w_dict):
///         return self.space.newlist(self.unerase(w_dict.dstorage).keys())
/// ```
pub struct IdentityDictStrategy;

/// `pypy/objspace/std/identitydict.py IdentityDictStrategy`
/// singleton — matches PyPy's `space.fromcache(IdentityDictStrategy)`.
pub static IDENTITY_DICT_STRATEGY: IdentityDictStrategy = IdentityDictStrategy;

/// The [`crate::dictmultiobject::DictStrategyRef`] holder a dict's `dstrategy`
/// slot points at.
pub static IDENTITY_DICT_STRATEGY_REF: crate::dictmultiobject::DictStrategyRef =
    crate::dictmultiobject::DictStrategyRef {
        kind: crate::dictmultiobject::StrategyKind::Identity,
        imp: &IDENTITY_DICT_STRATEGY,
        owner: std::ptr::null_mut(),
    };

impl IdentityDictStrategy {
    /// `identitydict.py IdentityDictStrategy.is_correct_type` —
    /// `self.space.type(w_obj).compares_by_identity()`.  Dispatch
    /// through the `dict_eq_hook::COMPARES_BY_IDENTITY_HOOK`
    /// trampoline (pyre-interpreter installs the MRO walker).
    #[inline]
    unsafe fn is_correct_type(w_key: PyObjectRef) -> bool {
        // `space.type(w_obj).compares_by_identity()`.
        crate::dictmultiobject::key_compares_by_identity(w_key)
    }
}

impl DictStrategy for IdentityDictStrategy {
    fn strategy_kind(&self) -> crate::dictmultiobject::StrategyKind {
        crate::dictmultiobject::StrategyKind::Identity
    }

    /// `dictmultiobject.py AbstractTypedStrategy.switch_to_object_strategy`
    /// instantiation for IdentityDictStrategy — `wrap` is identity
    /// (`:26-27`), so the migration ports each `IdentityKey(obj)` into
    /// `ObjectKey { hash: hash_w(obj), obj }` without rewrapping keys.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        w_dict_switch_identity_to_object_strategy(w_dict);
    }

    /// `identitydict.py get_empty_storage` — erased `{}` with
    /// non-null hint.  Pyre stores
    /// `IndexMap<IdentityKey, PyObjectRef>` — identity-keyed hash
    /// bucket for O(1) lookup + insertion-order preserving iteration.
    /// GC-managed box (`setfield_gc` on reassign).
    fn get_empty_storage(&self) -> crate::gc_hook::GCREF {
        // `identitydict.py get_empty_storage`: erased `{}` is a nursery
        // `dicttable`.
        crate::gc_storage::gc_alloc_young_storage_box(
            IdentityDictStorage::new(),
            identity_dict_storage_gc_type_id(),
        ) as crate::gc_hook::GCREF
    }

    /// `dictmultiobject.py AbstractTypedStrategy.getitem` —
    /// O(1) identity-keyed lookup.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if Self::is_correct_type(w_key) {
            return w_dict_lookup_identity_strategy(w_dict, w_key);
        }
        // `identitydict.py _never_equal_to` → always False, so
        // mismatched keys always promote and retry.
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key)
    }

    /// `dictmultiobject.py setitem` — identity-keyed insert;
    /// on mismatch, promote to Object.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if Self::is_correct_type(w_key) {
            crate::dictmultiobject::dict_write_barrier(w_dict);
            if w_dict_store_identity_strategy(w_dict, w_key, w_value) {
                crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
            }

            return;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py setitem_str` — IdentityDictStrategy
    /// promotes to Object on str setitem_str (str has its own
    /// non-identity `__eq__` / `__hash__`, so OVERRIDES_EQ_CMP_OR_HASH).
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_setitem_str(w_dict, key, w_value);
    }

    /// `dictmultiobject.py delitem` — identity-keyed remove;
    /// on mismatch, promote to Object.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if Self::is_correct_type(w_key) {
            let removed = w_dict_delete_identity_strategy(w_dict, w_key);
            if removed {
                crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
            }
            return removed;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        identity_storage(w_dict).len()
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        identity_storage(w_dict).keys().map(|k| k.0).collect()
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let mut out = Vec::new();
        for &v in identity_storage(w_dict).values() {
            out.push(v);
        }
        out
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let mut out = Vec::new();
        for (k, &v) in identity_storage(w_dict).iter() {
            out.push((k.0, v));
        }
        out
    }

    /// `dictmultiobject.py AbstractTypedStrategy.getitem` reads one
    /// entry off the typed storage.  The trait default instead rebuilds the
    /// whole `items()` per step, which makes one view walk quadratic in the
    /// entry count — and unlike the empty strategies it is written for, this
    /// one backs every dict keyed on ordinary instances.  `wrap` is identity
    /// here (`identitydict.py IdentityDictStrategy.wrap`), so the entry needs
    /// no allocation.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        identity_storage(w_dict)
            .get_slot(index)
            .map(|(k, &v)| (k.0, v))
    }

    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> Option<PyObjectRef> {
        identity_storage(w_dict).get_slot(index).map(|(_, &v)| v)
    }

    /// A delete leaves a tombstone rather than closing the hole, so the
    /// entries are not numbered densely and the default would walk positions.
    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        identity_storage(w_dict).next_valid_slot(from)
    }

    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        identity_storage(w_dict).prev_valid_slot(before)
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        let entries = identity_storage_mut(w_dict);
        if !entries.is_empty() {
            crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
        }
        entries.clear();
    }

    /// `dictmultiobject.py AbstractTypedStrategy.copy` — clone
    /// the typed `IndexMap<IdentityKey, _>` backing and wrap with the
    /// same IdentityDictStrategy.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let storage = identity_storage(w_dict);
        let new_storage = storage.clone_to_young_box(identity_dict_storage_gc_type_id());
        crate::dictmultiobject::w_dict_new_with(
            &IDENTITY_DICT_STRATEGY_REF,
            new_storage as crate::gc_hook::GCREF,
        )
    }

    /// `pypy.objspace.std.identitydict.IdentityDictStrategy` stores
    /// keys as `list[W_Root]` — the GC traces them as roots and follows
    /// the pointer.  Trace `key.0` through an unsafe pointer cast so a
    /// minor collection keeps each identity key's referent alive.
    ///
    /// Hash is now stable across GC moves via `gc_identity_hash`
    /// (shadow-based, minimark.py id_or_identityhash), so no rehash is needed.
    /// Trace key and value pointers so the GC updates them in place.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let entries = identity_storage_mut(w_dict);
        for (k, v) in entries.iter_mut_for_trace() {
            let key_ptr = k as *const IdentityKey as *mut IdentityKey;
            visitor(std::ptr::addr_of_mut!((*key_ptr).0));
            visitor(v as *mut PyObjectRef);
        }
        visitor(entries.entries_slot() as *mut PyObjectRef);
        entries.visit_indexes(&mut |slot| visitor(slot as *mut PyObjectRef));
    }
}
