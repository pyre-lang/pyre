//! W_DictMultiObject family — Python `dict` types.
//!
//! PyPy equivalent: `pypy/objspace/std/dictmultiobject.py`.
//!
//! Hosts the `W_DictMultiObject` Rust trait + concrete subclasses
//! `W_DictObject` (regular dict, `:313-325`) and `W_ModuleDictObject`
//! (module / globals dict backed by `ModuleDictStrategy` per
//! `pypy/objspace/std/celldict.py`).  The cutover lifted
//! pyre's `DictStorage`-based module-dict path out of
//! `pyre-interpreter/executioncontext.rs` and into a sibling
//! `celldict.rs` so the upstream strategy-per-W_Root model holds.
//!
//! Supports arbitrary PyObjectRef keys (int, str, etc.) with
//! equality comparison via `dict_keys_equal` (routed through the
//! `dict_eq_hook::EQ_W_HOOK` trampoline → `baseobjspace::eq_w`).
//!
//! ## Phase C-3 strategy dispatch status
//!
//! All eight W_DictMultiObject accessors route through
//! `w_dict_get_strategy(obj).method(obj, ...)` polymorphic dispatch
//! (PyPy's `w_dict.get_strategy().method(w_dict, ...)` shape):
//!
//! - `w_dict_setitem_str` → `dictmultiobject.py`
//! - `w_dict_getitem_str` → `:103-105`
//! - `w_dict_clear` → `:148-152` (`descr_clear`)
//! - `w_dict_items` → `:117-121`
//! - `w_dict_store` → `:97-99 setitem`
//! - `w_dict_lookup` → `:93-95 getitem`
//! - `w_dict_delitem` → `:101-102`
//! - `w_dict_len` → `:107-109 length`
//!
//! ## Field-deletion status (TODO queue)
//!
//! - `pyre-interpreter::DictStorage` struct: legacy str-keyed
//!   storage retained until its remaining translation funnels migrate.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::gc_hook::GCREF;
use crate::pyobject::*;
use crate::unicodeobject::Utf8Str;

/// `pypy/objspace/std/dictmultiobject.py ObjectDictStrategy`
/// — `r_dict(dict_keys_equal, hash_w)` key type.  The hash is cached
/// at insertion time so the `Hash` trait impl is infallible; equality
/// dispatches through `dict_keys_equal` so user-defined `__eq__`
/// resolves through the standard comparison protocol via the
/// `dict_eq_hook::EQ_W_HOOK` trampoline.
///
/// GC-move stability: `ObjectKey.hash` is the value of `space.hash_w(obj)`
/// at insertion time, which depends only on the object's *content* and
/// not on its memory address.  Python's data model requires hashable
/// types to have content-derived, identity-stable hashes
/// (`__hash__` must agree with `__eq__` and must not change for the
/// lifetime of the object); built-in immutables (int / str / tuple /
/// frozenset / bytes / bool) all satisfy this by construction, and
/// user-defined classes that override `__hash__` are contractually
/// required to do so.  Pyre's young-gen mark-and-copy collector may
/// move `key.obj` (the `*mut PyObject` payload) but the cached `hash`
/// remains valid because it is not a function of the address.  The
/// `walk_gc_refs` trace in `dictmultiobject.rs ObjectDictStrategy` visits
/// `key.obj` only and leaves `key.hash` untouched.  See
/// `identitydict.rs IdentityKey` for the contrasting *pointer-hashed*
/// case where this property does NOT hold.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ObjectKey {
    pub hash: i64,
    pub obj: PyObjectRef,
}

impl std::hash::Hash for ObjectKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_i64(self.hash);
    }
}

impl PartialEq for ObjectKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        // `IndexMap` compares `probe == stored`, so `self` is the incoming key
        // and `other` the one already in the bucket.  `ll_dict_lookup` runs
        // `keyeq(checkingkey, key)` with the *stored* key on the left
        // (`rordereddict.py`), which is what decides whose `__eq__` the
        // comparison protocol tries first; hand them over in that order.
        unsafe { dict_keys_equal(other.obj, self.obj) }
    }
}

impl Eq for ObjectKey {}

impl crate::rordereddict::EntryDummy for ObjectKey {
    fn dummy() -> Self {
        Self {
            hash: 0,
            obj: std::ptr::null_mut(),
        }
    }
}

/// `BytesDictStrategy` key: the `str` the `W_BytesObject` holds.
///
/// `unerase` yields that block; `wrap` builds a `bytes` object on it
/// (`w_bytes_from_block`) instead of copying the characters.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BytesKey(pub *mut crate::bytesobject::BytesBlock);

impl BytesKey {
    /// The block's `chars`, the same slice `[u8]` hashing and equality see.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { crate::bytesobject::bytes_block_chars(self.0) }
    }
}

impl std::hash::Hash for BytesKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // `BytesDictStrategy` keys are the STR (`rstr.py`); `ll_strhash`
        // reads `s.hash` and computes it only while the slot is 0.
        state.write_i64(crate::lowlevel_string::ll_strhash(
            self.0 as *mut crate::unicodeobject::Utf8Str,
        ) as i64);
    }
}

impl PartialEq for BytesKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for BytesKey {}

impl crate::rordereddict::EntryDummy for BytesKey {
    fn dummy() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl std::fmt::Debug for BytesKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_null() {
            return f.write_str("BytesKey(null)");
        }
        std::fmt::Debug::fmt(self.as_bytes(), f)
    }
}

/// Hash adapter for an `r_dict` key whose Python hash was already computed.
///
/// `rordereddict` feeds that cached integer directly into its table, and so
/// does this: the digest is already `space.hash_w(obj)`, and a second hash of
/// it would be pure cost.  Clustering on keys that share low bits is what
/// `ll_dict_lookup`'s perturb loop handles (`rordereddict.py`).
#[derive(Default)]
pub struct ObjectKeyHasher(u64);

impl std::hash::Hasher for ObjectKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for (shift, byte) in bytes.iter().copied().take(8).enumerate() {
            self.0 |= (byte as u64) << (shift * 8);
        }
    }

    #[inline]
    fn write_i64(&mut self, value: i64) {
        self.0 = value as u64;
    }
}

pub type ObjectKeyBuildHasher = std::hash::BuildHasherDefault<ObjectKeyHasher>;

/// The hasher for the `&str`-keyed entry tables (`celldict.py`'s module
/// storage).
///
/// RPython hashes a string with `ll_strhash`, a cheap multiply/xor chain whose
/// result is cached on the string object, so a dict probe on a name costs one
/// load upstream.  Rust's default `RandomState` is SipHash-1-3, which is a
/// keyed MAC: it exists to make hash-collision denial of service impractical
/// for tables fed untrusted keys, and these tables are fed program identifiers.
/// Every module-global read and write paid that MAC per probe.
///
/// FNV-1a over whole 8-byte words, so a name costs one or two rounds rather
/// than one per byte.  Streaming: `Hash for str` writes the bytes and then a
/// terminator, so the state carries across `write` calls.
pub struct StrKeyHasher(u64);

impl Default for StrKeyHasher {
    #[inline]
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl StrKeyHasher {
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    #[inline]
    fn mix(&mut self, word: u64) {
        self.0 = (self.0 ^ word).wrapping_mul(Self::PRIME);
    }
}

impl std::hash::Hasher for StrKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // One final avalanche so the low bits the table indexes with carry the
        // whole digest.
        let h = self.0;
        (h ^ (h >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9) >> 1
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.mix(bytes.len() as u64);
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.mix(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            // The zero-padded little-endian tail word, folded byte by byte: a
            // `copy_from_slice` of a variable length is an out-of-line
            // `memmove` on every global-name probe.
            let tail = rest
                .iter()
                .enumerate()
                .fold(0u64, |word, (i, &byte)| word | (byte as u64) << (8 * i));
            self.mix(tail);
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.mix(value as u64);
    }
}

pub type StrKeyBuildHasher = std::hash::BuildHasherDefault<StrKeyHasher>;

/// View a key as `&str` only when it is a str whose backing is valid UTF-8.
/// A lone-surrogate str returns `None` so the `&str`-keyed proxy fast paths
/// (which cannot represent it) fall through to the generic object-keyed path.
#[inline]
unsafe fn key_as_utf8(key: PyObjectRef) -> Option<&'static str> {
    if crate::is_str(key) {
        crate::w_str_get_value_opt(key)
    } else {
        None
    }
}

/// `dictmultiobject.py get_empty_storage r_dict(_, hash_w, ...)` —
/// `space.hash_w(obj)` precomputed for bucket placement.  There is a
/// single hashing path: the `dict_eq_hook::HASH_W_HOOK` trampoline,
/// installed at boot before any dict is built (production) or per test
/// thread (`#[cfg(test)]`).  A missing hook is a setup bug, not a
/// recoverable state, so [`missing_hash_hook`](crate::dict_eq_hook::missing_hash_hook)
/// fails loud rather than substituting a divergent structural hash.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn object_key_for(obj: PyObjectRef) -> ObjectKey {
    // `hash_w` runs the key's `__hash__` and is therefore a collection point,
    // while this key caches the pointer — the hazard [`object_key_hashed`]
    // documents. Publish `obj` across the call and build the key from the
    // reloaded address, or the entry lands in the `IndexMap` under the
    // pre-move pointer and every later probe dereferences reclaimed memory.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let hash = crate::dict_eq_hook::try_hash_w(obj)
        .unwrap_or_else(|| crate::dict_eq_hook::missing_hash_hook());
    if crate::dict_eq_hook::take_hash_error() {
        // Infallible path: swallow the error and use structural hash.
        // Checked callers should use `object_key_for_checked` instead.
    }
    // Clear any stale eq flag so the upcoming bucket probe starts clean;
    // an infallible probe's own error is then swallowed by the next key
    // construction, never leaking into a later checked op.
    crate::dict_eq_hook::take_eq_error();
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    ObjectKey { hash, obj }
}

/// Borrowed `&str` probe key for str-keyed dict GETs.  Hashes the WTF-8
/// bytes through [`dict_eq_hook::HASH_STR_HOOK`](crate::dict_eq_hook) so it
/// lands in the same `IndexMap` bucket an `object_key_for(w_str_new(key))`
/// would, and compares against a stored [`ObjectKey`] without materializing
/// a throwaway `W_UnicodeObject` — the per-lookup allocation that otherwise
/// leaks at every `getitem_str`.  PyPy's string-strategy `getitem_str`
/// (`dictmultiobject.py`) likewise probes from the raw str.
struct StrLookupKey<'a> {
    hash: i64,
    key: &'a str,
}

impl std::hash::Hash for StrLookupKey<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Identical to `ObjectKey::hash` (:78-80) so the borrow key hashes
        // into the same bucket as the stored object key.
        state.write_i64(self.hash);
    }
}

impl crate::rordereddict::Equivalent<ObjectKey> for StrLookupKey<'_> {
    #[inline]
    fn equivalent(&self, k: &ObjectKey) -> bool {
        // `ObjectKey::eq` gates on the cached hash before `dict_keys_equal`;
        // reproduce that so a hash mismatch never reads `k.obj`.
        if self.hash != k.hash {
            return false;
        }
        unsafe {
            if crate::is_exact_type(k.obj, &crate::STR_TYPE) {
                // Exact str: `__eq__` is value equality = WTF-8 byte equality
                // (`dict_keys_equal`'s str arm, :1503-1504).  A `&str` is valid
                // UTF-8, so a lone-surrogate stored key never matches.
                crate::w_str_get_wtf8(k.obj).as_bytes() == self.key.as_bytes()
            } else if crate::is_str(k.obj) {
                // str SUBCLASS keys may override `__eq__`/`__hash__`, so honour
                // the comparison protocol exactly as
                // `object_key_for(w_str_new(key))` → `ObjectKey::eq` →
                // `dict_keys_equal` does.  The materialized str is the rare-path
                // cost (a str-subclass dict key); the common exact-str arm
                // above allocates nothing.
                {
                    let _roots = crate::gc_roots::push_roots();
                    let w_key = crate::gc_roots::pin_root(crate::w_str_new_managed(self.key));
                    dict_keys_equal(k.obj, w_key)
                }
            } else {
                // ObjectDictStrategy uses the ordinary object equality hook
                // for every hash collision.  A non-str key is allowed to
                // compare equal to an exact str (and its `__eq__` may have
                // observable side effects), so the borrowed-str fast path
                // must not reject it by layout.  Materialise the query only
                // on this rare collision path, matching
                // `object_key_for(w_str_new(key))`.  Pass the stored key first,
                // the order `keyeq(checkingkey, key)` uses (`rordereddict.py:1055`).
                let _roots = crate::gc_roots::push_roots();
                let w_key = crate::gc_roots::pin_root(crate::w_str_new_managed(self.key));
                dict_keys_equal(k.obj, w_key)
            }
        }
    }
}

/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `ll_dict_lookup` (`rordereddict.py`).
fn dict_entries_probe_str_iff(entries: &ObjectDictStorage, _hash: i64, key: *mut Utf8Str) -> bool {
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
}

/// The rstr `STR` key (one word) probe [`dict_entries_get_str`] runs once the
/// str hash hook has produced `hash` — `celldict.py getitem_str`'s single
/// `self.unerase(w_dict.dstorage).get(key)`.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `ll_dict_lookup` (`rordereddict.py`).  The trampoline is the residual
/// arm when the dict is not virtual.
///
/// Returns `PY_NULL` for a miss: the residual result is one nullable word, as
/// `rordereddict.py ll_dict_lookup` returns one `Signed`.
///
/// # Safety
/// `entries` must be a live entry table whose keys are valid `ObjectKey`s.
#[majit_macros::look_inside_iff(dict_entries_probe_str_iff)]
pub unsafe fn dict_entries_probe_str(
    entries: &ObjectDictStorage,
    hash: i64,
    key: *mut Utf8Str,
) -> PyObjectRef {
    let key_str =
        unsafe { std::str::from_utf8_unchecked(crate::unicodeobject::utf8_payload_bytes(key)) };
    entries
        .get(&StrLookupKey { hash, key: key_str })
        .copied()
        .unwrap_or(PY_NULL)
}

/// The owned-key probe [`dict_entries_get_str`]'s no-hook arm runs — the same
/// `IndexMap::get` behind an [`object_key_for`] key instead of a borrowed one.
///
/// Residualised for the reason [`dict_entries_probe_str`] is: a body that
/// reaches either `get` stops there, so leaving one arm traced keeps the
/// enclosing graph blocked no matter what the other arm does.
///
/// # Safety
/// Same as [`dict_entries_probe_str`]; `key` must be a valid PyObjectRef.
fn dict_entries_probe_object_iff(entries: &ObjectDictStorage, key: PyObjectRef) -> bool {
    // rordereddict.py ll_dict_lookup: isvirtual(d) and isconstant(key)
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(dict_entries_probe_object_iff)]
pub unsafe fn dict_entries_probe_object(
    entries: &ObjectDictStorage,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    entries.get(&object_key_for(key)).copied()
}

/// The object-keyed remove side of the same table — `dictmultiobject.py delitem
/// delitem`'s `del self.unerase(w_dict.dstorage)[self.unwrap(w_key)]`.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(i))` on
/// `_ll_dict_del` (`rordereddict.py`).  The storage is an `RDict` now, so
/// the predicate is spelled; when it is false the trampoline is the same
/// residual as the former `dont_look_inside`.
///
/// # Safety
/// Same as [`dict_entries_probe_object`].
fn dict_entries_remove_object_iff(entries: &mut ObjectDictStorage, key: PyObjectRef) -> bool {
    // rordereddict.py _ll_dict_del: isvirtual(d) and isconstant(i)
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(dict_entries_remove_object_iff)]
pub unsafe fn dict_entries_remove_object(
    entries: &mut ObjectDictStorage,
    key: PyObjectRef,
) -> bool {
    entries.remove(&object_key_for(key)).is_some()
}

/// [`dict_entries_probe_object`] for a caller that already ran
/// [`object_key_for_checked`] — the checked paths, which must hash before the
/// probe so a raising `__hash__` is reported without touching the table.
///
/// The key arrives DECOMPOSED into its two words rather than as an
/// [`ObjectKey`]: a struct by value is not a residual-ABI argument shape, and
/// `hash` + `obj` are exactly what the caller holds anyway (the same split
/// [`dict_entries_probe_str`] takes).
///
/// # Safety
/// Same as [`dict_entries_probe_object`]; `hash` must be the digest
/// [`object_key_for_checked`] produced for `obj`.
fn dict_entries_probe_hashed_iff(
    entries: &ObjectDictStorage,
    _hash: i64,
    obj: PyObjectRef,
) -> bool {
    // rordereddict.py ll_dict_lookup: isvirtual(d) and isconstant(key)
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&obj)
}

#[majit_macros::look_inside_iff(dict_entries_probe_hashed_iff)]
pub unsafe fn dict_entries_probe_hashed(
    entries: &ObjectDictStorage,
    hash: i64,
    obj: PyObjectRef,
) -> Option<PyObjectRef> {
    entries.get(&ObjectKey { hash, obj }).copied()
}

/// The store side of [`dict_entries_probe_hashed`] —
/// `dictmultiobject.py setitem`'s
/// `self.unerase(w_dict.dstorage)[self.unwrap(w_key)] = w_value`, returning
/// the displaced value so the caller can tell an overwrite from an insert.
///
/// Residualised for [`dict_entries_probe_object`]'s reason: `IndexMap` has no
/// lowering, so the call is the last modellable point.  Upstream's
/// `_ll_dict_setitem_lookup_done` (`rordereddict.py`) is likewise
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))`, and
/// neither conjunct can hold for an `IndexMap` here.
///
/// # Safety
/// Same as [`dict_entries_probe_hashed`].
fn dict_entries_insert_hashed_iff(
    entries: &mut ObjectDictStorage,
    _hash: i64,
    obj: PyObjectRef,
    _value: PyObjectRef,
) -> bool {
    // rordereddict.py _ll_dict_setitem_lookup_done: isvirtual(d) and isconstant(key)
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&obj)
}

#[majit_macros::look_inside_iff(dict_entries_insert_hashed_iff)]
pub unsafe fn dict_entries_insert_hashed(
    entries: &mut ObjectDictStorage,
    hash: i64,
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Option<PyObjectRef> {
    entries.insert(ObjectKey { hash, obj }, value)
}

/// Read the value stored at an entry index the caller already settled on —
/// `rordereddict.py ll_dict_getitem`'s `d.entries[i].value` after
/// `ll_dict_lookup` returned the slot.
///
/// `d.entries[i].value` after `ll_dict_lookup` returned the slot
/// (`rordereddict.py ll_dict_getitem`).  A positional field read, so
/// the body stays look-inside.
///
/// # Safety
/// Same as [`dict_entries_probe_hashed`]; `index` must be a live entry index.
pub unsafe fn dict_entries_value_at(entries: &ObjectDictStorage, index: usize) -> PyObjectRef {
    *entries.get_slot(index).unwrap().1
}

/// Overwrite the value at an entry index the caller already settled on —
/// `dictmultiobject.py setitem_str`'s in-place update, which reuses
/// the stored key rather than re-inserting.
///
/// In-place update of `d.entries[i].value` (`dictmultiobject.py setitem_str`).
///
/// # Safety
/// Same as [`dict_entries_value_at`].
pub unsafe fn dict_entries_value_set_at(
    entries: &mut ObjectDictStorage,
    index: usize,
    value: PyObjectRef,
) {
    entries.set_slot_value(index, value);
}

/// The owned-key store — [`dict_entries_probe_object`]'s twin, hashing `key`
/// through [`object_key_for`] the way `dict_entries_remove_object` does.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `_ll_dict_setitem_lookup_done` (`rordereddict.py`).
///
/// # Safety
/// Same as [`dict_entries_probe_object`].
fn dict_entries_insert_object_iff(
    entries: &mut ObjectDictStorage,
    key: PyObjectRef,
    _value: PyObjectRef,
) -> bool {
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(dict_entries_insert_object_iff)]
pub unsafe fn dict_entries_insert_object(
    entries: &mut ObjectDictStorage,
    key: PyObjectRef,
    value: PyObjectRef,
) {
    // `object_key_for` hashes through the key's `__hash__`, so the caller's
    // `value` word is already handed over when the collection runs.  Publish it
    // here and insert the reloaded one.
    let roots = crate::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(value);
    let object_key = object_key_for(key);
    entries.insert(object_key, roots.get(value_slot));
}

/// The stored key's object word at an entry index, `None` once the walk has
/// passed the last entry — `rordereddict.py ll_dict_lookup`'s
/// `entries[i].key` plus the bound that ends its scan.
///
/// `entries[i].key` (`rordereddict.py ll_dict_lookup`).  A positional
/// field read, so the body stays look-inside.
///
/// # Safety
/// Same as [`dict_entries_value_at`].
pub unsafe fn dict_entries_key_obj_at(
    entries: &ObjectDictStorage,
    index: usize,
) -> Option<PyObjectRef> {
    entries.slot_key(index).map(|stored| stored.obj)
}

/// The stored key's cached digest at an entry index — `rordereddict.py:1053
/// entries.hash(i)`, the hash compared before `keyeq` is allowed to run.
///
/// Split from [`dict_entries_key_obj_at`] rather than returned beside it: an
/// `ObjectKey` by value is not a residual-ABI argument shape, and the caller
/// reads both words at the same index with no callback in between.
///
/// # Safety
/// Same as [`dict_entries_value_at`].
pub unsafe fn dict_entries_key_hash_at(entries: &ObjectDictStorage, index: usize) -> i64 {
    match entries.slot_key(index) {
        Some(stored) => stored.hash,
        None => 0,
    }
}

/// `d.num_ever_used_items` — the bound of a slot walk, and not the pair count
/// once a delete has left a tombstone behind (`rordereddict.py:1373`).
///
/// # Safety
/// Same as [`dict_entries_value_at`].
pub unsafe fn dict_entries_slot_count(entries: &ObjectDictStorage) -> usize {
    entries.entry_slots()
}

/// `entries.valid(index) && entries[index].key == checkingkey` — the second
/// half of `ll_dict_lookup`'s paranoia condition (`rordereddict.py`),
/// answered against the two key words the caller sampled before the
/// comparison it is validating.
///
/// # Safety
/// Same as [`dict_entries_value_at`].
pub unsafe fn dict_entries_key_is_at(
    entries: &ObjectDictStorage,
    index: usize,
    hash: i64,
    obj: PyObjectRef,
) -> bool {
    entries
        .get_slot(index)
        .is_some_and(|(stored, _)| stored.hash == hash && stored.obj == obj)
}

/// The table's identity, as `ll_dict_lookup`'s paranoia restart reads it
/// (`rordereddict.py entries != d.entries`): a stale slot number can no
/// longer be trusted once the entries have been renumbered.
///
/// A compaction is what renumbers them (`ll_dict_remove_deleted_items`,
/// `:803`), and it can run without changing the allocation's size, so the
/// counter the storage bumps on every reindex is the observable — a capacity
/// read would miss exactly the case this test exists for.
///
/// # Safety
/// Same as [`dict_entries_value_at`].
pub unsafe fn dict_entries_generation(entries: &ObjectDictStorage) -> u32 {
    entries.generation()
}

/// Drop the last entry — the undo the checked store runs when a user `__eq__`
/// raised mid-probe and `insert` therefore appended a spurious entry.
///
/// Undo a checked store that appended a spurious entry after a raising
/// `__eq__`.  A slot pop, so the body stays look-inside.
///
/// # Safety
/// Same as [`dict_entries_probe_hashed`].
pub unsafe fn dict_entries_pop_last(entries: &mut ObjectDictStorage) {
    entries.pop();
}

/// Str dict GET over the rstr `STR` key (one word): hash via the str hook and
/// probe `entries` without building a `W_UnicodeObject`.  Falls back to the
/// wrapping `object_key_for(w_str_from_storage(key))` path when no str hash
/// hook is installed (pyre-object lib tests without the str hook, pre-init
/// snapshot tools), preserving today's behavior there.
///
/// `hash` is `key`'s digest when the caller already holds it —
/// [`crate::unicodeobject::w_str_hash_memoized`] on the string object `key` was
/// borrowed from (`rstr.py ll_strhash`).  Zero means the caller has
/// none and the STR is hashed here.
#[inline]
unsafe fn dict_entries_get_str(
    entries: &ObjectDictStorage,
    key: *mut Utf8Str,
    hash: i64,
) -> Option<PyObjectRef> {
    if hash != 0 || crate::dict_eq_hook::has_hash_str_hook() {
        let hash = if hash != 0 {
            hash
        } else {
            crate::dict_eq_hook::hash_str_hooked_str(key)
        };
        // Clear any stale eq flag so a str-subclass comparison in the probe
        // starts clean, matching `object_key_for`'s pre-probe reset (:125).
        crate::dict_eq_hook::take_eq_error();
        nullable_to_option(dict_entries_probe_str(entries, hash, key))
    } else {
        dict_entries_probe_object(entries, crate::unicodeobject::w_str_from_storage(key))
    }
}

/// `key`'s digest: the one the caller memoized on the string object it borrowed
/// `key` from, or — when it holds none — a fresh hash of the STR key.
/// `None` when no str hash hook is installed, which puts the caller on its
/// allocating owned-key arm.
/// Membership probe returning the entry index, for str-keyed
/// `setitem_str` over the rstr `STR` key (one word): a re-store to an existing
/// name updates in place and reuses the stored key, so only a genuinely new
/// key allocates a persistent `W_UnicodeObject` (PyPy `setitem_str` keeps
/// `newtext(key)` only for the inserted key, not an overwrite —
/// `dictmultiobject.py`).  Falls back to the wrapping
/// `object_key_for(w_str_from_storage(key))` probe when no str hash hook is
/// installed.
///
/// `hash` carries a caller-held digest, as in [`dict_entries_get_str`].
#[inline(never)]
unsafe fn dict_entries_index_of_str_or_absent(
    entries: &ObjectDictStorage,
    key: *mut Utf8Str,
) -> isize {
    if crate::dict_eq_hook::has_hash_str_hook() {
        let hash = crate::dict_eq_hook::hash_str_hooked_str(key);
        crate::dict_eq_hook::take_eq_error();
        let key_str =
            unsafe { std::str::from_utf8_unchecked(crate::unicodeobject::utf8_payload_bytes(key)) };
        entries.index_or_absent(&StrLookupKey { hash, key: key_str })
    } else {
        entries.index_or_absent(&object_key_for(crate::unicodeobject::w_str_from_storage(
            key,
        )))
    }
}

unsafe fn dict_entries_index_of_str(
    entries: &ObjectDictStorage,
    key: *mut Utf8Str,
    hash: i64,
) -> Option<usize> {
    if hash != 0 || crate::dict_eq_hook::has_hash_str_hook() {
        let hash = if hash != 0 {
            hash
        } else {
            crate::dict_eq_hook::hash_str_hooked_str(key)
        };
        crate::dict_eq_hook::take_eq_error();
        dict_entries_index_of_str_hashed(entries, hash, key)
    } else {
        dict_entries_index_of_object(entries, crate::unicodeobject::w_str_from_storage(key))
    }
}

/// The rstr `STR` key (one word) membership probe [`dict_entries_index_of_str`]
/// runs once the str hash hook has produced `hash` — [`dict_entries_probe_str`]'s
/// twin, returning the entry index instead of the value.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `ll_dict_lookup` (`rordereddict.py`), the index-returning twin of
/// [`dict_entries_probe_str`].
///
/// # Safety
/// Same as [`dict_entries_probe_str`].
fn dict_entries_index_of_str_hashed_iff(
    entries: &ObjectDictStorage,
    _hash: i64,
    key: *mut Utf8Str,
) -> bool {
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(dict_entries_index_of_str_hashed_iff)]
pub unsafe fn dict_entries_index_of_str_hashed(
    entries: &ObjectDictStorage,
    hash: i64,
    key: *mut Utf8Str,
) -> Option<usize> {
    let key_str =
        unsafe { std::str::from_utf8_unchecked(crate::unicodeobject::utf8_payload_bytes(key)) };
    entries.index_of(&StrLookupKey { hash, key: key_str })
}

/// The owned-key membership probe [`dict_entries_index_of_str`]'s no-hook arm
/// runs — [`dict_entries_probe_object`]'s twin.
///
/// Same predicate as [`dict_entries_probe_object`].
///
/// # Safety
/// Same as [`dict_entries_probe_object`].
fn dict_entries_index_of_object_iff(entries: &ObjectDictStorage, key: PyObjectRef) -> bool {
    majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
}

#[majit_macros::look_inside_iff(dict_entries_index_of_object_iff)]
pub unsafe fn dict_entries_index_of_object(
    entries: &ObjectDictStorage,
    key: PyObjectRef,
) -> Option<usize> {
    entries.index_of(&object_key_for(key))
}

/// Build the persistent exact-str key for a new `setitem_str` entry.
/// When the raw-string hash hook is installed, use it directly instead of
/// re-entering Python-level `hash_w` on the freshly boxed string.  This is
/// required while bootstrapping `str.__hash__` itself: its defining type dict
/// is already a W_DictObject, so `hash_w(w_str)` would recursively look up
/// `str.__hash__` in the dict currently being filled.
///
/// `hash` is the caller's digest for `key`, or 0 when it holds none.  The
/// digest is memoized on the new string (`rstr.py ll_strhash`), so the
/// key this dict now owns never hashes its bytes again.
#[inline]
unsafe fn object_key_for_new_str(key: *mut Utf8Str, hash: i64) -> ObjectKey {
    let obj = crate::unicodeobject::w_str_from_storage(key);
    let hash = match hash {
        0 => match crate::dict_eq_hook::try_hash_str(key) {
            Some(hash) => {
                crate::dict_eq_hook::take_eq_error();
                hash
            }
            None => return object_key_for(obj),
        },
        memoized => memoized,
    };
    unsafe { crate::unicodeobject::w_str_set_hash(obj, hash) };
    ObjectKey { hash, obj }
}

/// Fallible variant of [`object_key_for`].  When the `hash_w` hook
/// signals an error (unhashable type, user `__hash__` raised), this
/// returns `Err(DictKeyError)`.  The caller retrieves the concrete
/// error from the interpreter-side error slot.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn object_key_for_checked(obj: PyObjectRef) -> Result<ObjectKey, DictKeyError> {
    // Same publication as [`object_key_for`]: `hash_w` can collect and this
    // key caches the pointer, so key on the reloaded address.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let hash = crate::dict_eq_hook::try_hash_w(obj)
        .unwrap_or_else(|| crate::dict_eq_hook::missing_hash_hook());
    if crate::dict_eq_hook::take_hash_error() {
        return Err(DictKeyError);
    }
    // Clean slate for the bucket probe that follows in the caller; its
    // eq error is read back via `take_dict_key_error` after the access.
    crate::dict_eq_hook::take_eq_error();
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    Ok(ObjectKey { hash, obj })
}

/// [`ObjectKey`] from a `space.hash_w` digest the caller already holds.
///
/// The pair to [`object_key_for_checked`] for callers that must hash at a
/// point where `obj` can still be rooted — a user `__hash__` is a collection
/// point, and the key caches `obj`, so a key built around the hash call
/// captures the pre-move pointer.  Hashing first and handing the digest down
/// here keys the element on the reloaded pointer and, because `hash_w` runs
/// once, keeps `__hash__` at the single call `r_dict` makes.
///
/// `hash` must come from the same `space.hash_w` [`object_key_for`] would
/// have invoked, or the key lands in a different bucket than an equal one
/// stored through that path.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn object_key_hashed(obj: PyObjectRef, hash: i64) -> ObjectKey {
    // Clean slate for the bucket probe that follows in the caller, as in
    // `object_key_for_checked`.
    crate::dict_eq_hook::take_eq_error();
    ObjectKey { hash, obj }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn hash_key_checked(obj: PyObjectRef) -> Result<(), DictKeyError> {
    let _ = crate::dict_eq_hook::try_hash_w(obj);
    if crate::dict_eq_hook::take_hash_error() {
        return Err(DictKeyError);
    }
    Ok(())
}

/// `dictmultiobject.py` `w_dict.strategy is SomeStrategy` — a field
/// compare on the holder, not a vtable call.  Upstream stores the
/// strategy instance and tests identity; the holder carries `kind`
/// so the real path sees a `getfield` the dual gate can type.
#[inline]
fn strategy_is(current: &DictStrategyRef, expected: StrategyKind) -> bool {
    current.kind == expected
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn key_compares_by_identity(key: PyObjectRef) -> bool {
    // A tagged immediate is an `int`, which compares by value, never by
    // identity; short-circuit before the `w_class` deref. Gated on
    // `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(key) {
        return false;
    }
    // `space.type` / `W_Root.getclass` / `BaseUserClassMapdict.getclass`.
    // Read-only singletons leave that word null; `gettypefor(ob_type)` is
    // `get_instantiate`.
    let mut w_type = crate::mapdict::instance_getclass(key) as PyObjectRef;
    if w_type.is_null() {
        let tp = (*key).ob_type;
        if tp.is_null() {
            return false;
        }
        w_type = get_instantiate(&*tp);
    }
    !w_type.is_null()
        && matches!(
            crate::dict_eq_hook::try_compares_by_identity(w_type),
            Some(true)
        )
}

#[inline]
unsafe fn _never_equal_to_int(key: PyObjectRef) -> bool {
    crate::is_none(key) || crate::is_bytes(key) || crate::is_str(key)
}

/// Marker error returned by checked dict operations when the key's
/// `__hash__` fails.  The concrete `PyError` is stored in a
/// thread-local on the `pyre-interpreter` side; the caller converts
/// this marker to a real exception via `take_pending_hash_error()`.
#[derive(Debug)]
pub struct DictKeyError;

/// `true` when the most recent checked dict op raised through either key
/// callback of the `r_dict(eq_w, hash_w)` pair: `space.hash_w` at key
/// construction or `space.eq_w` during the bucket probe.  Reading clears
/// both flags.  The concrete `PyError` rides the interpreter-side
/// pending slot, retrieved via `take_pending_hash_error`.
#[inline]
pub(crate) unsafe fn take_dict_key_error() -> bool {
    let hash = crate::dict_eq_hook::take_hash_error();
    let eq = crate::dict_eq_hook::take_eq_error();
    hash || eq
}

/// Test-only structural hash hook.  Walks the same built-in type ladder
/// as `dict_keys_equal` (`:1207-1260`) so equal keys land in the same
/// bucket, giving `#[cfg(test)]` dict tests a single deterministic hash
/// path without reaching into `pyre-interpreter`'s `space.hash_w` (which
/// lives in a crate above this one).  Installed via `register_hash_w_hook`
/// by the test harness; never a production code path — production hashes
/// exclusively through `space.hash_w` (`baseobjspace.py is_none`).
/// Shared str-key digest for the test hooks: the WTF-8 byte sequence hashed
/// with `DefaultHasher`.  Used by both [`builtin_structural_hash`]'s str arm
/// (object-keyed) and the str-keyed `hash_str` test hook so a key stored via
/// `hash_w` and probed via a borrowed `StrLookupKey` land in the same bucket.
#[cfg(test)]
fn structural_str_hash_bytes(bytes: &[u8]) -> i64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish() as i64
}

/// `HashStrHookFn` test trampoline pairing with [`builtin_structural_hash`].
#[cfg(test)]
unsafe fn builtin_structural_str_hash(ptr: *const u8, len: usize) -> i64 {
    structural_str_hash_bytes(unsafe { std::slice::from_raw_parts(ptr, len) })
}

#[cfg(test)]
unsafe fn builtin_structural_hash(obj: PyObjectRef) -> i64 {
    if obj.is_null() {
        return 0;
    }
    if crate::is_bool(obj) {
        return crate::w_bool_get_value(obj) as i64;
    }
    if crate::is_int(obj) {
        return crate::w_int_get_value(obj);
    }
    if crate::is_str(obj) {
        // Hash the WTF-8 bytes so a lone-surrogate key does not panic in
        // `w_str_get_value`; the byte sequence is the hashed identity.
        return structural_str_hash_bytes(crate::w_str_get_wtf8(obj).as_bytes());
    }
    if crate::bytesobject::is_bytes(obj) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        crate::bytesobject::w_bytes_data(obj).hash(&mut h);
        return h.finish() as i64;
    }
    if crate::is_tuple(obj) {
        let n = crate::w_tuple_len(obj);
        let mut acc: i64 = 0x345678;
        for i in 0..n {
            let e = crate::w_tuple_getitem(obj, i as i64).unwrap_or(std::ptr::null_mut());
            acc = acc
                .wrapping_mul(1000003)
                .wrapping_add(builtin_structural_hash(e));
        }
        return acc;
    }
    if crate::is_frozenset(obj) {
        // Order-independent (xor) so frozenset({a,b}) and frozenset({b,a})
        // hash equal, matching `dict_keys_equal`'s set-containment arm.
        let mut acc: i64 = 0;
        for &e in crate::w_set_items(obj).iter() {
            acc ^= builtin_structural_hash(e);
        }
        return acc;
    }
    obj as usize as i64
}

/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject(W_Root)`
/// abstract base — both `W_DictObject` (regular dicts) and
/// `W_ModuleDictObject` (module/global dicts) inherit from it.
///
/// ```python
/// class W_DictMultiObject(W_Root):
///     """ Abstract base class that does not store a strategy. """
///     __slots__ = ['space', 'dstorage']
///
///     def get_strategy(self):
///         raise NotImplementedError("abstract method")
///
///     def set_strategy(self, strategy):
///         raise NotImplementedError("abstract method")
/// ```
///
/// Pyre's `space` field is implicit (no `ObjSpace` shim per-dict);
/// `dstorage` lives on each concrete subclass as the appropriate type
/// (`*mut Vec<(PyObjectRef, PyObjectRef)>` for W_DictObject,
/// `*mut ModuleDictStorage` for W_ModuleDictObject).  The trait
/// surfaces the abstract `get_strategy` / `set_strategy` for static
/// trait-dispatch sites; runtime polymorphic dispatch (i.e. when only
/// a `PyObjectRef` is in hand) goes through the free function
/// [`w_dict_get_strategy`].
#[allow(non_camel_case_types)]
pub trait W_DictMultiObject {
    /// `dictmultiobject.py W_DictMultiObject.get_strategy`
    /// abstract method, overridden by `W_DictObject` (`:321-322`) and
    /// `W_ModuleDictObject` (`:338-339`).  Each concrete subclass
    /// returns its strategy slot.
    fn get_strategy(&self) -> &dyn crate::dictmultiobject::DictStrategy;

    /// `dictmultiobject.py W_DictMultiObject.set_strategy`
    /// abstract method, overridden by `W_DictObject` (`:324-325`) and
    /// `W_ModuleDictObject` (`:341-342`).  Pyre limits the setter to
    /// a `&'static DictStrategyRef` (the singleton dispatch surface);
    /// W_ModuleDictObject strategy promotion to ObjectDictStrategy
    /// continues to go through `w_module_dict_switch_to_object_strategy`
    /// per `celldict.py switch_to_object_strategy`.
    fn set_strategy(&mut self, strategy: &'static DictStrategyRef);
}

/// `pypy/objspace/std/dictmultiobject.py W_DictObject(W_DictMultiObject)`
/// — the regular-dict concrete subclass.  PyPy slots are
/// `['dstrategy']` on top of W_DictMultiObject's `['space', 'dstorage']`;
/// pyre carries the same logical slots.
///
/// Layout: `[ob_header | dstorage | dstrategy]`
///
/// Slots:
/// - `dstorage`: erased ObjectDictStrategy storage —
///   `Vec<(PyObjectRef, PyObjectRef)>` matches PyPy's
///   `r_dict(space.eq_w, space.hash_w)` storage (`:1209-1212`).  Keys
///   compared by `dict_keys_equal` which routes through the registered
///   `dict_eq_hook::EQ_W_HOOK` trampoline → `baseobjspace::eq_w`.
/// - `dstrategy`: PyPy `:315 __slots__ = ['dstrategy']`.  Reference
///   to the active strategy singleton; `space.fromcache(StrategyCls)`
///   returns the same instance per space, so pyre stores a `&'static
///   dyn DictStrategy` (fat pointer).  New dicts start in
///   `OBJECT_DICT_STRATEGY`; promotion is `w_dict_set_strategy`
///   per `:324-325 set_strategy`.
#[repr(C)]
pub struct W_DictObject {
    pub ob_header: PyObject,
    /// `dstorage` from `W_DictMultiObject.__slots__` (`dictmultiobject.py`).
    /// PyPy's `rerased`-erased storage; pyre uses [`GCREF`] for the
    /// same opacity contract (`llmemory.GCREF`).  Each strategy
    /// `unerase(dict.dstorage)` via a typed accessor.  The erased pointer is a
    /// GC-managed storage box (`gc_alloc_young_storage_box`, one per-strategy
    /// concrete type: [`ObjectDictStorage`] / [`IntDictStorage`] /
    /// [`BytesDictStorage`] / `identitydict::IdentityDictStorage` /
    /// `kwargsdict::KwargsDictStorage`), so `w_dict.dstorage = strategy.erase(new)`
    /// is a `setfield_gc` and the old box is reclaimed by the sweep — matching
    /// PyPy's `r_dict` = GcStruct("dicttable") (rdict.py).  A MapDictStrategy
    /// `dstorage` is instead the backing instance (a GC edge, mapdict.rs); the
    /// side-table holder ([`w_dict_new_unmanaged_side_table_value`]) keeps an
    /// off-GC `malloc_raw` box.
    pub dstorage: GCREF,
    /// `dstrategy` from `W_DictObject.__slots__` (`dictmultiobject.py`) —
    /// **one word**, like `W_ModuleDictObject.mstrategy` beside it and like the
    /// single instance pointer upstream stores.  The strategy singletons are
    /// unit structs, so a `&'static dyn DictStrategy` here would be a fat
    /// pointer carrying the vtable in the *reference*: no field for the JIT to
    /// read, and a data word that Rust may collapse to one address across
    /// distinct ZST statics.  [`DictStrategyRef`] holds the trait object
    /// instead and derefs to it, so this slot is a plain pointer that a
    /// `guard_class` can pin the way RPython pins an inlined virtual call's
    /// receiver.
    pub dstrategy: &'static DictStrategyRef,
    /// Mutation state carried implicitly by PyPy's live strategy iterator.
    /// `IndexMap::shift_remove` compacts its storage, so pyre records key-set
    /// changes explicitly; value-only overwrites leave this unchanged.
    pub keys_version: usize,
    /// Generation bumped whenever `clear` empties an object dict in place.
    /// `ll_dict_clear` reallocates `d.entries` to a fresh empty array
    /// (`rordereddict.py`), so a probe that cleared the dict through an
    /// `__eq__` callback must restart on the `entries != d.entries` arm
    /// (`:1058`).  `IndexMap::clear` retains its buffer, leaving the capacity
    /// and every stale index unchanged, so the reentrant scan watches this
    /// counter to reproduce that restart.  It bumps only on clear, never on a
    /// non-emptying insert or delete, so a callback that merely reshapes the
    /// dict does not trigger a spurious restart PyPy would not take.
    pub clear_gen: usize,
}

/// Typed accessor — `dictmultiobject.py ObjectDictStrategy.getitem`
/// (`self.unerase(w_dict.dstorage)`) returns the `r_dict(dict_keys_equal,
/// hash_w)` backing.  Pyre stores this as `IndexMap<ObjectKey,
/// PyObjectRef>`: a hash bucket for O(1) lookup that also preserves
/// insertion order (CPython 3.7+ / PyPy3 dict semantics).  Shared with
/// `UnicodeDictStrategy` per `dictmultiobject.py`'s str
/// fast-path delegation to the Object helpers.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// `ObjectDictStrategy` or `UnicodeDictStrategy`.
#[inline]
pub unsafe fn w_dict_object_storage<'a>(obj: PyObjectRef) -> &'a ObjectDictStorage {
    let dict = &*(obj as *const W_DictObject);
    &*(dict.dstorage as *const ObjectDictStorage)
}

/// Mutable typed accessor — write-side of [`w_dict_object_storage`].
///
/// # Safety
/// Same as [`w_dict_object_storage`].
#[inline]
pub unsafe fn w_dict_object_storage_mut<'a>(obj: PyObjectRef) -> &'a mut ObjectDictStorage {
    let dict = &mut *(obj as *mut W_DictObject);
    &mut *(dict.dstorage as *mut ObjectDictStorage)
}

/// Typed accessor for `IntDictStrategy.unerase(w_dict.dstorage)` —
/// `dictmultiobject.py IntDictStrategy.erase/unerase` pair
/// produced by `rerased.new_erasing_pair("integer")`.  Returns the
/// native `IndexMap<i64, PyObjectRef>` backing — insertion-ordered
/// hash bucket matching PyPy's `Dict[int, W_Root]` (RPython resolves
/// to an order-preserving hash table at translation time).
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` whose strategy is
/// [`crate::dictmultiobject::INT_DICT_STRATEGY`].
#[inline]
pub unsafe fn w_dict_int_storage<'a>(obj: PyObjectRef) -> &'a IntDictStorage {
    let dict = &*(obj as *const W_DictObject);
    &*(dict.dstorage as *const IntDictStorage)
}

/// Mutable counterpart of [`w_dict_int_storage`].
///
/// # Safety
/// Same as [`w_dict_int_storage`].
#[inline]
pub unsafe fn w_dict_int_storage_mut<'a>(obj: PyObjectRef) -> &'a mut IntDictStorage {
    let dict = &mut *(obj as *mut W_DictObject);
    &mut *(dict.dstorage as *mut IntDictStorage)
}

/// Typed accessor for `BytesDictStrategy.unerase(w_dict.dstorage)` —
/// `dictmultiobject.py BytesDictStrategy.erase/unerase` pair
/// produced by `rerased.new_erasing_pair("bytes")`.  Returns the
/// native `RDict<BytesKey, PyObjectRef>` backing — insertion-ordered
/// hash bucket matching PyPy `Dict[str, W_Root]` (RPython resolves to
/// an order-preserving hash table at translation time). The key is the
/// `BytesBlock` `unerase` returns, not a copy of its characters.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` whose strategy is
/// [`crate::dictmultiobject::BYTES_DICT_STRATEGY`].
#[inline]
pub unsafe fn w_dict_bytes_storage<'a>(obj: PyObjectRef) -> &'a BytesDictStorage {
    let dict = &*(obj as *const W_DictObject);
    &*(dict.dstorage as *const BytesDictStorage)
}

/// Mutable counterpart of [`w_dict_bytes_storage`].
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
#[inline]
pub unsafe fn w_dict_bytes_storage_mut<'a>(obj: PyObjectRef) -> &'a mut BytesDictStorage {
    let dict = &mut *(obj as *mut W_DictObject);
    &mut *(dict.dstorage as *mut BytesDictStorage)
}

/// GC type id assigned to `W_DictObject` at JitDriver init time.
pub const W_DICT_GC_TYPE_ID: u32 = 29;

// ── dstorage GC-managed storage boxes ───────────────────────────────
//
// PyPy's `W_DictMultiObject.dstorage` (`dictmultiobject.py`) erases an
// `r_dict` = `GcStruct("dicttable")` (rdict.py:210) — GC-managed, so
// `w_dict.dstorage = strategy.erase(new)` is a `setfield_gc` and the old
// table is reclaimed by the collector. pyre stores the same erased handle
// as a `*mut u8`, but each concrete strategy backs it with a native Rust
// container. Those containers now live in a GC-managed box born young
// (`gc_alloc_young_storage_box`, [`crate::gc_storage`]) tagged with a per-type
// runtime-assigned id — the mapdict `W_MAPDICT_STORAGE_GC_TYPE_ID` leaf
// precedent (`object_array.rs`). Each box that holds a GC reference has
// its own custom trace (`object_dict_storage_custom_trace` and the
// int/bytes/identity/kwargs twins). `dict_object_custom_trace` forwards
// the `dstorage` slot only. `DictStrategy::walk_gc_refs` remains the
// walker for an immortal or off-GC owner. Box reassignment on a strategy
// switch is a plain field store; the unreachable old box is reclaimed by
// the sweep, which runs the box tid's `storage_box_destructor` drop glue.
//
// Ids are auto-assigned (like the bigint box, `longobject.rs`) so they land
// at the registry tail and don't perturb the fixed-const tid chain.

/// `ObjectDictStrategy` / `UnicodeDictStrategy` backing — erased
/// `r_dict(dict_keys_equal, hash_w)` (`dictmultiobject.py`). Also the
/// universal `switch_to_object_strategy` sink and the identity-dict
/// promotion target.
pub type ObjectDictStorage =
    crate::rordereddict::RDict<ObjectKey, PyObjectRef, ObjectKeyBuildHasher>;

#[inline]
pub fn object_dict_storage_new() -> ObjectDictStorage {
    ObjectDictStorage::with_hasher(ObjectKeyBuildHasher::default())
}

#[inline]
pub fn object_dict_storage_with_capacity(capacity: usize) -> ObjectDictStorage {
    ObjectDictStorage::with_capacity_and_hasher(capacity, ObjectKeyBuildHasher::default())
}

/// `ll_int_hash` (`rpython/rtyper/lltypesystem/rint.py`) — an integer key
/// hashes to itself, and `rordereddict` feeds that straight into the table.
///
/// The clustering an identity hash invites on keys that share low bits is what
/// `ll_dict_lookup`'s perturb loop exists for: it folds the high bits back in
/// after the first probe (`rordereddict.py`).  A digest here would
/// only be paying for the mixing twice.
#[derive(Default)]
pub struct IntKeyHasher(u64);

impl std::hash::Hasher for IntKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for (shift, byte) in bytes.iter().copied().take(8).enumerate() {
            self.0 |= (byte as u64) << (shift * 8);
        }
    }

    #[inline]
    fn write_i64(&mut self, value: i64) {
        self.0 = value as u64;
    }
}

/// `IntDictStrategy` backing — erased `Dict[int, W_Root]`
/// (`dictmultiobject.py`).
pub type IntKeyHash = std::hash::BuildHasherDefault<IntKeyHasher>;
/// See [`BytesDictStorage`].
pub type BytesKeyHash = ObjectKeyBuildHasher;
pub type IntDictStorage = crate::rordereddict::RDict<i64, PyObjectRef, IntKeyHash>;
/// `BytesDictStrategy` backing — erased `Dict[str, W_Root]`
/// (`dictmultiobject.py`). `unerase` yields [`BytesKey`], the block the
/// wrapped `bytes` object holds.
///
/// [`BytesKey`] writes `ll_strhash`, the digest memoized in the STR header
/// (`rstr.py` `LLHelpers.ll_strhash` / `STR.hash`), and [`BytesKeyHash`]
/// hands it to the table unchanged, the way `ll_dict_lookup` takes
/// `d.keyhash(key)` (`rordereddict.py`).
pub type BytesDictStorage = crate::rordereddict::RDict<BytesKey, PyObjectRef, BytesKeyHash>;

macro_rules! dict_storage_gc_type_id {
    ($atomic:ident, $setter:ident, $getter:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        static $atomic: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        #[doc = concat!("Record the GC type id registered for the `", stringify!($ty), "` box.")]
        pub fn $setter(id: u32) {
            $atomic.store(id, std::sync::atomic::Ordering::Relaxed);
        }
        #[doc = concat!("Read the runtime-assigned GC type id for the `", stringify!($ty), "` box.")]
        #[majit_macros::dont_look_inside]
        pub fn $getter() -> u32 {
            $atomic.load(std::sync::atomic::Ordering::Relaxed)
        }
    };
}

dict_storage_gc_type_id!(
    OBJECT_DICT_STORAGE_GC_TYPE_ID,
    set_object_dict_storage_gc_type_id,
    object_dict_storage_gc_type_id,
    ObjectDictStorage,
    "Runtime-assigned GC type id for the `ObjectDictStorage` box."
);
dict_storage_gc_type_id!(
    INT_DICT_STORAGE_GC_TYPE_ID,
    set_int_dict_storage_gc_type_id,
    int_dict_storage_gc_type_id,
    IntDictStorage,
    "Runtime-assigned GC type id for the `IntDictStorage` box."
);
dict_storage_gc_type_id!(
    BYTES_DICT_STORAGE_GC_TYPE_ID,
    set_bytes_dict_storage_gc_type_id,
    bytes_dict_storage_gc_type_id,
    BytesDictStorage,
    "Runtime-assigned GC type id for the `BytesDictStorage` box."
);
dict_storage_gc_type_id!(
    DICT_STRATEGY_REF_GC_TYPE_ID,
    set_dict_strategy_ref_gc_type_id,
    dict_strategy_ref_gc_type_id,
    DictStrategyRef,
    "Runtime-assigned GC type id for a per-dict `DictStrategyRef` holder."
);

/// Reclaim an off-GC `ObjectDictStorage` box that is NOT collector-owned.
///
/// After the storage-box migration, a regular dict's `dstorage` is a
/// GC-managed box reclaimed by the sweep (its tid carries the drop glue), so
/// the sweep-time `dict_object_destructor` must not free it. The sole
/// off-GC exception is [`w_dict_new_unmanaged_side_table_value`], whose
/// holder is not GC-managed (root-walked, never custom-traced) and keeps a
/// `malloc_raw` `ObjectDictStorage`. `try_gc_owns_object` discriminates —
/// mirrors `object_array::dealloc_items_block`.
///
/// # Safety
/// `dstorage` must be null or a valid `ObjectDictStorage` pointer.
#[inline]
pub unsafe fn dealloc_offgc_object_dict_storage(dstorage: GCREF) {
    if dstorage.is_null() {
        return;
    }
    if !crate::gc_hook::try_gc_owns_object(dstorage) {
        drop(Box::from_raw(dstorage as *mut ObjectDictStorage));
    }
}

/// Fixed payload size (`framework.py:811`).
pub const W_DICT_OBJECT_SIZE: usize = std::mem::size_of::<W_DictObject>();

impl crate::lltype::GcType for W_DictObject {
    fn type_id() -> u32 {
        W_DICT_GC_TYPE_ID
    }
    const SIZE: usize = W_DICT_OBJECT_SIZE;
}

/// `pypy/objspace/std/dictmultiobject.py W_DictObject(W_DictMultiObject)`
/// inheritance — `get_strategy`/`set_strategy` overrides read/write
/// the `dstrategy` slot directly.
impl W_DictMultiObject for W_DictObject {
    #[inline]
    fn get_strategy(&self) -> &dyn crate::dictmultiobject::DictStrategy {
        self.dstrategy.imp
    }

    #[inline]
    fn set_strategy(&mut self, strategy: &'static DictStrategyRef) {
        self.dstrategy = strategy;
    }
}

/// Remember a dict whose body or storage box is about to receive a young
/// reference.
///
/// Residual (`@dont_look_inside`, `rlib/jit.py`), the shape of mapdict's
/// `instance_write_barrier`: the box's entries live in Rust `Vec`s that only
/// the box's custom trace walks, so a store into them is not a
/// `setfield_gc` / `setarrayitem_gc` the backend rewrite can put
/// `COND_CALL_GC_WB` in front of. The header-flag test below is GC code,
/// which jitcode never contains.
#[majit_macros::dont_look_inside]
#[inline]
pub(crate) fn dict_write_barrier(obj: PyObjectRef) {
    // `FixedObjectArray::set_ref`: inspect TRACK_YOUNG_PTRS inline and
    // enter the barrier only while the old holder still needs to be
    // remembered. The dict body is remembered for the `dstorage` field
    // store. Entry stores land in the storage box (`rordereddict.py`
    // `GcStruct("dicttable")`), which has its own trace, so that box is
    // remembered even when the body is young or already on the remembered
    // set. Map and Class `dstorage` is the instance or the type, not a box.
    if obj.is_null() {
        return;
    }
    let header = unsafe { majit_gc::header::header_of(obj as usize) };
    if unsafe { (*header).has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
        crate::gc_hook::try_gc_write_barrier_managed(obj as crate::gc_hook::GCREF);
    }
    if unsafe { is_any_dict(obj) } {
        let dict = unsafe { &*(obj as *const W_DictObject) };
        let kind = dict.dstrategy.kind;
        if kind != StrategyKind::Map && kind != StrategyKind::Class && !dict.dstorage.is_null() {
            crate::gc_hook::try_gc_write_barrier(dict.dstorage as crate::gc_hook::GCREF);
        }
    }
}

/// Point a regular dict at a freshly filled object-strategy storage box.
///
/// The box is born young (`gc_alloc_young_storage_box`) and the keys and
/// values just written into it may be nursery-born too — typed-strategy
/// switches mint `newint` / `newbytes` wrappers, and kwargs / identity
/// copies keep whatever age the previous table held. The dict's custom_trace
/// is the only walker of those interiors, so an already-old dict has to be
/// remembered before the next minor (`setfield_gc` on `dstorage`) or the box
/// and its contents are freed under it.
pub(crate) unsafe fn install_object_dict_storage(
    w_dict: PyObjectRef,
    new_storage: *mut ObjectDictStorage,
) {
    install_dict_storage(w_dict, new_storage as GCREF, &OBJECT_DICT_STRATEGY_REF);
}

/// `pypy/objspace/std/dictmultiobject.py W_DictObject.get_strategy`
/// (the regular-dict subclass returns its `dstrategy` slot directly).
///
/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.get_strategy`
/// abstract method, overridden by `W_DictObject` (`:321-322`) and
/// `W_ModuleDictObject` (`:338-339`).  Polymorphic dispatch: returns
/// the strategy live in the dict's slot — `dstrategy` for regular
/// dicts (currently always `OBJECT_DICT_STRATEGY`), or `mstrategy`
/// (the per-allocation `ModuleDictStrategy`) for module dicts.
///
/// The `&'static` lifetime is sound because pyre's strategy
/// allocations (`malloc_raw` + `Box::leak` pattern for
/// `ModuleDictStrategy`) live for the program duration, matching the
/// strategy-singleton model of PyPy `space.fromcache(StrategyCls)`.
///
/// # Safety
/// `obj` must be a valid PyObjectRef pointing at a `W_DictObject` or
/// `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_dict_get_strategy(
    obj: PyObjectRef,
) -> &'static dyn crate::dictmultiobject::DictStrategy {
    if is_module_dict(obj) {
        let strat_ptr = (*(obj as *const W_ModuleDictObject)).mstrategy;
        return (*strat_ptr).imp;
    }
    let dict = &*(obj as *const W_DictObject);
    dict.dstrategy.imp
}

/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.set_strategy`
/// abstract method, overridden by `W_DictObject` (`:324-325`) and
/// `W_ModuleDictObject` (`:341-342`).  Polymorphic dispatch: writes
/// the strategy slot of whichever subclass `obj` points to.
///
/// # Safety
/// `obj` must be a valid PyObjectRef pointing at a `W_DictObject` or
/// `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_dict_set_strategy(obj: PyObjectRef, strategy: &'static DictStrategyRef) {
    if is_module_dict(obj) {
        let md = &mut *(obj as *mut W_ModuleDictObject);
        md.jit_force_quasi_immutable("mutate_mstrategy");
        md.mstrategy = strategy as *const DictStrategyRef as *mut DictStrategyRef;
        dict_write_barrier(obj);
        return;
    }
    let dict = &mut *(obj as *mut W_DictObject);
    dict.dstrategy = strategy;
}

/// `celldict.py getdictvalue_no_unwrapping` slot lookup for the JIT
/// global cell fast path.  Returns the insertion-order index of `name`
/// inside the module dict's `ModuleDictStorage`, or `None` when `obj` is
/// not a `W_ModuleDictObject` in `ModuleDictStrategy` mode (after
/// `switch_to_object_strategy` the erased `dstorage` holds ObjectDictStorage and
/// the version-keyed cell path is permanently invalid).  The index is
/// stable across in-place cell mutation and value overwrite (`IndexMap`
/// keeps the slot position), so it serves as the elidable lookup key.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_cell_slot_of(obj: PyObjectRef, name: &str) -> Option<usize> {
    if obj.is_null() || !is_module_dict(obj) {
        return None;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if w_module_dict_is_object_strategy(obj) || md.dstorage.is_null() {
        return None;
    }
    (*(md.dstorage as *const crate::celldict::ModuleDictStorage)).index_of(name)
}

/// `celldict.py _getdictvalue_no_unwrapping_pure` — the raw stored
/// value-or-cell at `slot` (the result of `getdictvalue_no_unwrapping`,
/// _not_ unwrapped).  `None` when `obj` is not a module dict in
/// `ModuleDictStrategy` mode or `slot` is out of range.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_cell_at(obj: PyObjectRef, slot: usize) -> Option<PyObjectRef> {
    if obj.is_null() || !is_module_dict(obj) {
        return None;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if w_module_dict_is_object_strategy(obj) || md.dstorage.is_null() {
        return None;
    }
    (*(md.dstorage as *const crate::celldict::ModuleDictStorage))
        .get_slot(slot)
        .map(|(_, v)| *v)
        .filter(|p| !p.is_null())
}

/// Direct O(1) entry count of a module dict's `ModuleDictStorage`
/// (`dictmultiobject.py length` for the `ModuleDictStrategy`
/// case), bypassing the strategy vtable. `None` when `obj` is not a
/// `W_ModuleDictObject` in `ModuleDictStrategy` mode. Used by the JIT
/// frame-shape guard on the per-portal-entry hot path.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_storage_len(obj: PyObjectRef) -> Option<usize> {
    if obj.is_null() || !is_module_dict(obj) {
        return None;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if w_module_dict_is_object_strategy(obj) || md.dstorage.is_null() {
        return None;
    }
    Some((*(md.dstorage as *const crate::celldict::ModuleDictStorage)).len())
}

/// `quasiimmut.py get_current_qmut_instance` for a module dict
/// strategy's `version?` quasi-immutable field (`celldict.py
/// _immutable_fields_ = ["version?"]`).  The recording binds the returned
/// instance, and the loop compiled from it registers there, so a later
/// `mutated()` (new key, `del`, `switch_to_object_strategy`, or a write that
/// replaces a stored cell) flips the flag and fails the loop's
/// `GUARD_NOT_INVALIDATED`.
///
/// Takes the strategy rather than the dict because that is what the trace pins:
/// `version` lives on the strategy, so the `QUASIIMMUT_FIELD` names it directly.
///
/// # Safety
/// `strategy` must be null or point at a valid `ModuleDictStrategy`.
pub unsafe fn module_dict_strategy_current_version_qmut(
    strategy: *mut crate::celldict::ModuleDictStrategy,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if strategy.is_null() {
        return None;
    }
    Some((*strategy).current_version_qmut())
}

/// `pyjitpl.py MIFrame.opimpl_jit_force_quasi_immutable mutatebox.nonnull()` for this strategy's `version?`.
///
/// # Safety
/// `strategy` must be null or point at a valid `ModuleDictStrategy`.
pub unsafe fn module_dict_strategy_version_qmut_installed(
    strategy: *const crate::celldict::ModuleDictStrategy,
) -> bool {
    !strategy.is_null() && (*strategy).version_qmut_installed()
}

/// `quasiimmut.py do_force_quasi_immutable` for this strategy's
/// `version?` — the invalidation the tracer performs itself before aborting
/// (`pyjitpl.py:1113-1115`).
///
/// # Safety
/// `strategy` must be null or point at a valid `ModuleDictStrategy`.
pub unsafe fn module_dict_strategy_force_version_qmut(
    strategy: *const crate::celldict::ModuleDictStrategy,
) {
    if strategy.is_null() {
        return;
    }
    (*strategy).force_version_qmut();
}

/// Allocate a new empty dict per `dictmultiobject.py:67-69
/// allocate_and_init_instance`:
///
/// ```python
/// strategy = space.fromcache(EmptyDictStrategy)
/// storage = strategy.get_empty_storage()
/// W_DictObject.__init__(w_obj, space, strategy, storage)
/// ```
///
/// `EmptyDictStrategy.get_empty_storage` is `erase(None)`.  The first
/// mutating call promotes via `switch_to_correct_strategy` and installs
/// real storage, the same `w_dict_new_kwargs` path.
/// A traced call is [`newdict_empty`]: null `dstorage`, `EmptyDictStrategy`,
/// same shape as `ll_newdict`. `malloc_typed_managed` is the boxing primitive
/// `w_str_from_storage_and_length` / specialised-tuple constructors expose so
/// `fuse_boxing_alloc` emits `new_with_vtable`; `malloc_typed` leaks off-heap
/// (`alloc_with_gc_header`) and that pass treats it as non-movable.
#[inline(never)]
pub fn newdict_empty() -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        dstorage: std::ptr::null_mut(),
        dstrategy: &crate::dictmultiobject::EMPTY_DICT_STRATEGY_REF,
        keys_version: 0,
        clear_gen: 0,
    }) as PyObjectRef
}

/// Allocate a new empty dict per `dictmultiobject.py allocate_and_init_instance`.
///
/// One body: [`newdict_empty`] (`malloc_typed_managed` / `ll_newdict`).
/// `fuse_boxing_alloc` rewrites that cluster to `new_with_vtable`.
#[inline(never)]
pub fn w_dict_new() -> PyObjectRef {
    newdict_empty()
}

/// Empty `frozendict` — same `W_DictObject` payload as [`newdict_empty`],
/// allocated with [`crate::pyobject::FROZENDICT_TYPE`] the way
/// [`crate::setobject::w_frozenset_new`] allocates `W_SetObject` under
/// `FROZENSET_TYPE`.
#[inline(never)]
pub fn w_frozendict_new() -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_DictObject {
        ob_header: PyObject {
            ob_type: &crate::pyobject::FROZENDICT_TYPE as *const PyType,
            w_class: get_instantiate(&crate::pyobject::FROZENDICT_TYPE),
        },
        dstorage: std::ptr::null_mut(),
        dstrategy: &crate::dictmultiobject::EMPTY_DICT_STRATEGY_REF,
        keys_version: 0,
        clear_gen: 0,
    }) as PyObjectRef
}

/// Clone `obj`'s strategy + storage into a newly allocated exact `frozendict`.
///
/// `W_FrozensetObject.descr_new2` / `W_BaseSetObject.from_storage_and_strategy`
/// allocate a frozen instance then install the source's strategy and a copy
/// of its storage. Strategy `copy()` still mints a `dict`; adopt that
/// strategy/storage onto the new frozendict (`w_dict_adopt_regular_copy_for_empty_update`).
///
/// # Safety
/// `obj` must be a live `W_DictObject` (dict or frozendict storage).
pub unsafe fn w_frozendict_copy_from(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return w_frozendict_new();
    }
    let _roots = crate::gc_roots::push_roots();
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_frozendict_new());
    let copied = w_dict_copy(crate::gc_roots::shadow_stack_get(src_slot));
    let copied_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(copied);
    w_dict_adopt_regular_copy_for_empty_update(
        crate::gc_roots::shadow_stack_get(dst_slot),
        crate::gc_roots::shadow_stack_get(copied_slot),
    );
    crate::gc_roots::shadow_stack_get(dst_slot)
}

/// Type-namespace twin of [`w_dict_new`].
///
/// `new_typeobject_with_metatype_and_layout` hands this dict to an `init`
/// closure as a Rust `PyObjectRef`. Descriptor allocation inside that
/// closure collects (`instantiate(GetSetProperty)` is young-nonmoving), and
/// the translator is not there to reload the local. A moving nursery birth
/// leaves `init` holding the corpse; `w_dict_setitem_str` then reads
/// poison (`posix.DirEntry` under `PYPY_GC_NURSERY=1`). Young-nonmoving
/// keeps the header address stable the way [`w_getset_property_new`] does.
#[majit_macros::dont_look_inside]
pub fn w_dict_new_nonmoving() -> PyObjectRef {
    let value = W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        dstorage: std::ptr::null_mut(),
        dstrategy: &crate::dictmultiobject::EMPTY_DICT_STRATEGY_REF,
        keys_version: 0,
        clear_gen: 0,
    };
    let raw = crate::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(
        W_DICT_GC_TYPE_ID,
        W_DICT_OBJECT_SIZE,
    );
    if raw.is_null() {
        // Stable (old) header: the caller hands this pointer into an `init`
        // closure by value, so a moving nursery fallback would recreate the
        // DirEntry poison read. Address stability is the contract.
        return alloc_dict_object(value, true);
    }
    unsafe {
        std::ptr::write(raw as *mut W_DictObject, value);
    }
    crate::gc_hook::try_gc_write_barrier(raw);
    raw as PyObjectRef
}

pub type MakeInstanceDictHookFn = fn() -> PyObjectRef;

majit_gc::global_hook!(static MAKE_INSTANCE_DICT_HOOK: MakeInstanceDictHookFn);

pub fn register_make_instance_dict_hook(hook: MakeInstanceDictHookFn) {
    MAKE_INSTANCE_DICT_HOOK.set(Some(hook));
}

pub fn clear_make_instance_dict_hook() {
    MAKE_INSTANCE_DICT_HOOK.set(None);
}

/// `dictmultiobject.py:56-73` `space.newdict(instance=True)`.
///
/// The interpreter installs the mapdict factory once its hooks are available.
/// The plain-dict fallback keeps pyre-object-only users working before that
/// initialization.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the body
/// dispatches through the runtime `MAKE_INSTANCE_DICT_HOOK` fn-pointer cell (or
/// falls back to [`w_dict_new`]); both arms return a fresh dict GCREF, so
/// residualising the whole getter models it by signature — a plain
/// `PyObjectRef` — instead of tracing the hook-cell indirection.
#[majit_macros::dont_look_inside]
pub fn w_dict_new_instance() -> PyObjectRef {
    match MAKE_INSTANCE_DICT_HOOK.get() {
        Some(factory) => factory(),
        None => w_dict_new(),
    }
}

/// `dictmultiobject.py allocate_and_init_instance` kwargs
/// branch — `strategy = space.fromcache(EmptyKwargsDictStrategy)`.
/// Function-call sites that build a `**kwargs` dict route through
/// this allocator so the first unicode setitem promotes the dict
/// directly to `KwargsDictStrategy` (skipping the regular
/// `UnicodeDictStrategy` intermediate).
///
/// Own graph so `fuse_boxing_alloc` rewrites the young
/// `malloc_typed_managed` cluster to `new_with_vtable` + field stores
/// (`allocate_and_init_instance` traces as plain malloc + setfields).
/// `alloc_dict_object` stays `@dont_look_inside` for every other
/// caller.
#[inline(never)]
pub fn w_dict_new_kwargs() -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        dstorage: std::ptr::null_mut(),
        dstrategy: &crate::dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY_REF,
        keys_version: 0,
        clear_gen: 0,
    }) as PyObjectRef
}

/// `dictmultiobject.py W_DictObject(space, strategy, storage)` —
/// caller-chosen strategy + caller-owned dstorage.  Used by the
/// per-strategy `copy()` overrides (`:1152 AbstractTypedStrategy.copy`)
/// to allocate a fresh W_DictObject that preserves the source's
/// strategy + a freshly cloned typed storage box.  Length is computed
/// on demand by `strategy.length(self)` from the typed storage shape.
pub fn w_dict_new_with(strategy: &'static DictStrategyRef, dstorage: GCREF) -> PyObjectRef {
    // The clone filled the host table before the box existed, so no entry
    // store ran `dict_write_barrier`. Remember the box before
    // `alloc_dict_object`, which can collect.
    if !dstorage.is_null() {
        crate::gc_hook::try_gc_write_barrier(dstorage);
    }
    alloc_dict_object(
        W_DictObject {
            ob_header: PyObject {
                ob_type: &DICT_TYPE as *const PyType,
                w_class: get_instantiate(&DICT_TYPE),
            },
            dstorage,
            dstrategy: strategy,
            keys_version: 0,
            clear_gen: 0,
        },
        false,
    )
}

/// Allocate a dict for a pyre-side address-keyed side table.
///
/// These tables are not part of the translated object graph yet, so the dict
/// holder itself must keep a stable raw address. The table walker traces the
/// entries through [`w_dict_walk_entries_mut`] instead.
///
/// Unlike a regular dict, this holder is not GC-managed (it is `malloc_typed`'d
/// with an object-root id and root-walked, never `dict_object_custom_trace`d),
/// so its `dstorage` stays an off-GC `malloc_raw` box — the sole exception to
/// the S2 GC-managed storage-box migration. `dealloc_offgc_object_dict_storage`
/// frees it (its `try_gc_owns_object` guard is false for this box).
pub fn w_dict_new_unmanaged_side_table_value() -> PyObjectRef {
    let entries: *mut ObjectDictStorage = crate::lltype::malloc_raw(object_dict_storage_new());
    crate::lltype::malloc_typed(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        dstorage: entries as GCREF,
        dstrategy: &crate::dictmultiobject::OBJECT_DICT_STRATEGY_REF,
        keys_version: 0,
        clear_gen: 0,
    }) as PyObjectRef
}

/// Visit the raw `entries` vector's key/value slots with mutable access.
///
/// Used by pyre-side side table walkers whose dict object is not itself
/// GC-managed but whose contained PyObjectRef values still need relocation.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_walk_entries_mut(obj: PyObjectRef, mut visitor: impl FnMut(&mut PyObjectRef)) {
    let entries = w_dict_object_storage_mut(obj);
    for (key, value) in entries.iter_mut_for_trace() {
        // `ObjectKey.hash` is precomputed at insertion (`object_key_for`)
        // and stays valid across a GC move — Python's `__hash__` contract
        // is identity-stable for hashable (immutable) keys, which is the
        // only class of object permitted in this strategy.  Mutating the
        // embedded `obj` slot in place therefore does not invalidate
        // IndexMap's bucket placement.
        let key_ptr = key as *const ObjectKey as *mut ObjectKey;
        visitor(&mut (*key_ptr).obj);
        visitor(value);
    }
    let slot = entries.entries_slot() as *mut PyObjectRef;
    visitor(unsafe { &mut *slot });
    entries.visit_indexes(&mut |slot| visitor(unsafe { &mut *(slot as *mut PyObjectRef) }));
}

/// Visit every GC-reference slot in a dict's strategy-owned storage.
///
/// This is the host-side counterpart of the translated `rerased` trace
/// function. It is used when an immortal/prebuilt owner must descend through
/// an already-old dict during a minor collection; merely visiting the dict
/// object itself does not rescan that old object's storage.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_walk_gc_refs(obj: PyObjectRef, visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    if obj.is_null() {
        return;
    }
    // Grey the dicttable box (`W_DictMultiObject.dstorage`). Walking only
    // the entries interiors leaves a young-nonmoving box without
    // `VISITED_RMY`, and the next trace of `dstorage` reads a swept header.
    // EmptyDictStrategy / EmptyKwargsDictStrategy store `erased(None)`
    // (`get_empty_storage`); skip that null the way `dict_object_custom_trace`
    // does. Visiting it would walk a slot the strategy has no storage for.
    let dstorage_slot =
        std::ptr::addr_of_mut!((*(obj as *mut W_DictObject)).dstorage) as *mut PyObjectRef;
    if !(*dstorage_slot).is_null() {
        visitor(unsafe { &mut *dstorage_slot });
    }
    let strategy = unsafe { w_dict_get_strategy(obj) };
    let mut adapter = |slot: *mut PyObjectRef| {
        visitor(unsafe { &mut *slot });
    };
    unsafe { strategy.walk_gc_refs(obj, &mut adapter) };
}

/// Allocate and initialise a heap `W_DictObject`.
///
/// Dispatches the thread-local GC allocation hook (`try_gc_alloc` /
/// `try_gc_alloc_stable`) the tracer cannot model, falling back to
/// `lltype::malloc_typed` (`NewWithVtable`); the JIT residualises the
/// call instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py`), the `box_str_constant` / `try_gc_add_root` twin.
///
/// `stable` selects old-gen. Only callers that need a fixed address ask for
/// it: forcing it for every dict strands a compiled frame's ref slot across a
/// minor collection (a non-moving dict changes which loads the trace keeps
/// live, and `jf_frame[43]` is then left holding a pre-move nursery address).
/// Reproducer: `synth/mutate_then_raise_caught` under
/// `MAJIT_GC_NURSERY_POISON=1`.
#[majit_macros::dont_look_inside]
pub fn alloc_dict_object(value: W_DictObject, stable: bool) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket (`framework.py`), the `w_method_new`
    // twin: `dstorage` is a freshly allocated young storage box and the header
    // malloc below can collect, so pin it across the malloc and re-read the
    // relocated address afterwards.
    let mut value = value;
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(value.dstorage as PyObjectRef);
    let raw = if stable {
        crate::gc_hook::try_gc_alloc_stable_raw(W_DICT_GC_TYPE_ID, W_DICT_OBJECT_SIZE)
    } else {
        // The `malloc_typed` box below answers a missing hook. It cannot
        // answer a GC that owns the heap and refused: the dict's strategy and
        // storage edges would sit outside the object graph, behind a header
        // the collector never walks.
        crate::gc_hook::GcAllocOutcome::from_hook(crate::gc_hook::try_gc_alloc(
            W_DICT_GC_TYPE_ID,
            W_DICT_OBJECT_SIZE,
        ))
        .allocated_or_abort(W_DICT_OBJECT_SIZE)
        .unwrap_or(std::ptr::null_mut())
    };
    value.dstorage = crate::gc_roots::shadow_stack_get(save_point) as GCREF;
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_DictObject, value);
        }
        // A `stable` header is old-gen while `dstorage` is young, so the store
        // above is an old->young edge: remember it or the next minor never
        // traces the box and its entries dangle. No-op for a young header.
        crate::gc_hook::try_gc_write_barrier(raw);
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(value) as PyObjectRef
    }
}

// ── W_ModuleDictObject ──────────────────────────────────────────────
//
// `pypy/objspace/std/dictmultiobject.py W_ModuleDictObject`:
//
//     class W_ModuleDictObject(W_DictMultiObject):
//         """ a dict object for a module, that is not expected to
//         change. It stores the strategy as a quasi-immutable
//         field. """
//         __slots__ = ['mstrategy']
//         _immutable_fields_ = ['mstrategy?']
//
//         def __init__(self, space, strategy, storage):
//             W_DictMultiObject.__init__(self, space, storage)
//             self.mstrategy = strategy
//
//         def get_strategy(self):
//             return self.mstrategy
//
//         def set_strategy(self, strategy):
//             self.mstrategy = strategy
//
// Carries an owned `ModuleDictStorage` and a per-allocation
// `ModuleDictStrategy` (PyPy reuses one strategy instance per space;
// pyre allocates one strategy per W_ModuleDictObject for simplicity
// until the `space` shim grows a strategy cache).

/// Static `PyType` tag for `W_ModuleDictObject`.  Distinct from
/// `DICT_TYPE` so `py_type_check` can tell module dicts apart from
/// regular dicts inside the Rust runtime — but PyPy presents both as
/// `space.w_dict` to user code (`dictmultiobject.py
/// allocate_instance(W_ModuleDictObject, space.w_dict)`).  The
/// `TYPEOBJECT_CACHE` registration at
/// `pyre-interpreter/typedef.rs` maps `MODULE_DICT_TYPE` to
/// the same `dict` W_TypeObject as `DICT_TYPE`, so `type(g) is dict`
/// and `isinstance(g, dict)` hold on `W_ModuleDictObject` instances
/// even though they carry a different Rust layout / GC type id.
pub static MODULE_DICT_TYPE: PyType = new_pytype("dict");

#[repr(C)]
#[majit_macros::jit_immutable_fields("mstrategy?")]
pub struct W_ModuleDictObject {
    pub ob_header: PyObject,
    /// `dstorage` from `W_DictMultiObject.__slots__` (`dictmultiobject.py`).
    /// A GC-managed non-moving erased storage box (off-GC storage).
    /// Reassignment is a `setfield_gc`; its concrete type follows the live
    /// strategy exactly as `rerased` storage does upstream.
    pub dstorage: GCREF,
    /// `mstrategy` from `W_ModuleDictObject.__slots__` (`:331`).
    /// A GC-managed non-moving storage box (off-GC storage).
    /// Quasi-immutable (`mstrategy?`): every store goes through
    /// [`W_ModuleDictObject::jit_force_quasi_immutable`] first.
    pub mstrategy: *mut DictStrategyRef,
    /// Key-set mutation state corresponding to the live iterator held by
    /// PyPy's ModuleDictStrategy iterator implementation.
    pub keys_version: usize,
    /// The same iterator restart generation carried by `W_DictObject`.
    /// Keeping the concrete layouts prefix-identical lets the ordinary
    /// ObjectDictStrategy operate on a module dict after PyPy's literal
    /// `mstrategy`/`dstorage` swap.
    pub clear_gen: usize,
    /// Hidden `mutate_mstrategy` field `quasiimmut.py get_mutate_field_name`
    /// names and `rclass.py InstanceRepr._setup_repr` adds for
    /// `dictmultiobject.py W_ModuleDictObject._immutable_fields_ = ['mstrategy?']`.
    pub mutate_mstrategy: crate::quasiimmut::QuasiImmutField,
}

/// GC type id assigned to `W_ModuleDictObject`.  Lands at slot 48,
/// the first free id after the foreign-pytype loop in
/// `pyre/pyre-jit/src/eval.rs` registers NONE_TYPE (43),
/// NOTIMPLEMENTED_TYPE (44), ELLIPSIS_TYPE (45), CODE_TYPE (46)
/// and PYTRACEBACK_TYPE (47).
pub const W_MODULE_DICT_GC_TYPE_ID: u32 = 48;

/// Fixed payload size used by `gct_fv_gc_malloc`.
pub const W_MODULE_DICT_OBJECT_SIZE: usize = std::mem::size_of::<W_ModuleDictObject>();

impl crate::lltype::GcType for W_ModuleDictObject {
    fn type_id() -> u32 {
        W_MODULE_DICT_GC_TYPE_ID
    }
    const SIZE: usize = W_MODULE_DICT_OBJECT_SIZE;
}

impl W_ModuleDictObject {
    /// `rclass.py InstanceRepr.hook_setfield` `jit_force_quasi_immutable(vinst, 'mutate_mstrategy')`,
    /// the native shape `warmspot.py rewrite_force_quasi_immutable` leaves.
    #[majit_macros::jit_llop("jit_force_quasi_immutable")]
    #[inline(never)]
    pub fn jit_force_quasi_immutable(&self, mutatefieldname: &'static str) {
        match mutatefieldname {
            "mutate_mstrategy" => {
                if !self.mutate_mstrategy.is_installed() {
                    return;
                }
                unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.mutate_mstrategy) };
            }
            _ => panic!("unknown mutate field name {mutatefieldname:?}"),
        }
    }
}

/// `pypy/objspace/std/dictmultiobject.py W_ModuleDictObject(W_DictMultiObject)`
/// inheritance — `get_strategy`/`set_strategy` overrides read/write
/// the `mstrategy` slot (a per-allocation `ModuleDictStrategy`).
/// The `&'static` lifetime on the returned strategy reference is
/// sound because pyre's `ModuleDictStrategy` is malloc_raw'd and
/// lives for the W_ModuleDictObject's full lifetime (which itself
/// outlives any caller currently in scope).
impl W_DictMultiObject for W_ModuleDictObject {
    #[inline]
    fn get_strategy(&self) -> &dyn crate::dictmultiobject::DictStrategy {
        unsafe { (*self.mstrategy).imp }
    }

    /// `dictmultiobject.py W_ModuleDictObject.set_strategy` (`:341-342`) —
    /// the bare slot write.  `celldict.py switch_to_object_strategy`
    /// (`:173-186`) is what composes it with the matching `dstorage` swap.
    fn set_strategy(&mut self, strategy: &'static DictStrategyRef) {
        self.jit_force_quasi_immutable("mutate_mstrategy");
        self.mstrategy = strategy as *const DictStrategyRef as *mut DictStrategyRef;
    }
}

/// Allocate a fresh non-moving `W_ModuleDictObject` whose storage is empty
/// and whose strategy carries a fresh `VersionTag`. Module dicts are
/// long-lived, and their by-value handle is passed to initializers across
/// allocating stores, so the header must keep a stable address. Mirrors
/// `dictmultiobject.py allocate_and_init_instance(module=True)`
/// path:
///
/// ```python
/// if module:
///     strategy = ModuleDictStrategy(space)
///     storage = strategy.get_empty_storage()
///     w_obj = space.allocate_instance(W_ModuleDictObject, space.w_dict)
///     W_ModuleDictObject.__init__(w_obj, space, strategy, storage)
///     return w_obj
/// ```
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`):
/// the body performs a `lltype::malloc_typed_stable` allocation
/// (`W_ModuleDictObject`) that `fuse_boxing_alloc` leaves unfused, so
/// the JIT residualises the whole call to a stable runtime fnaddr
/// instead of tracing the allocation.  The `-> PyObjectRef` result is a
/// plain GCREF with no discriminant to erase.
#[majit_macros::dont_look_inside]
pub fn w_module_dict_new() -> PyObjectRef {
    let roots = crate::gc_roots::push_roots();
    let strategy = crate::gc_storage::gc_alloc_storage_box(
        crate::celldict::ModuleDictStrategy::new(),
        crate::celldict::module_dict_strategy_gc_type_id(),
    );
    let strategy_slot = roots.publish(&[strategy as PyObjectRef]);
    let strategy = roots.get(strategy_slot) as *mut crate::celldict::ModuleDictStrategy;
    // The strategy box is non-moving and lives as long as this holder.  The
    // holder's `owner` word is explicitly traced by the module-dict custom
    // trace, so extending the reference lifetime expresses that GC ownership
    // without leaking a process-global strategy.
    let imp: &'static dyn DictStrategy =
        unsafe { std::mem::transmute::<&dyn DictStrategy, &'static dyn DictStrategy>(&*strategy) };
    let strategy_ref = crate::gc_storage::gc_alloc_storage_box(
        DictStrategyRef {
            kind: StrategyKind::Module,
            imp,
            owner: strategy as GCREF,
        },
        dict_strategy_ref_gc_type_id(),
    );
    let strategy_ref_slot = roots.publish(&[strategy_ref as PyObjectRef]);
    let storage = unsafe {
        crate::gc_storage::gc_alloc_young_storage_box(
            (*strategy).get_empty_storage(),
            crate::celldict::module_dict_storage_gc_type_id(),
        )
    };
    let storage_slot = roots.publish(&[storage as PyObjectRef]);
    crate::lltype::malloc_typed_stable(W_ModuleDictObject {
        ob_header: PyObject {
            // `dictmultiobject.py:67 space.allocate_instance(...,
            // space.w_dict)` — module dicts present as `dict` to
            // user code (registered via TYPEOBJECT_CACHE on the
            // interpreter side); the Rust static tag distinguishes
            // the layout internally.
            ob_type: &MODULE_DICT_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_DICT_TYPE),
        },
        dstorage: roots.get(storage_slot) as GCREF,
        mstrategy: roots.get(strategy_ref_slot) as *mut DictStrategyRef,
        keys_version: 0,
        clear_gen: 0,
        mutate_mstrategy: crate::quasiimmut::QuasiImmutField::new(),
    }) as PyObjectRef
}

/// A PyPy-GIL-sized dict operation boundary.
///
/// `rgil` now holds the process GIL for every stretch of pyre code, exactly as
/// PyPy does. The old address-striped reentrant mutex predated that port and
/// duplicated the GIL on every dictionary operation. The root scope remains:
/// hashing or equality may collect, so callers still publish and reload every
/// live `PyObjectRef` across the operation.
pub struct DictOperationGuard {
    roots: crate::gc_roots::RootScope,
    root_base: usize,
}

impl DictOperationGuard {
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn new(obj: PyObjectRef, refs: &[PyObjectRef]) -> Self {
        let roots = crate::gc_roots::push_roots();
        // The dictionary and its operands become visible in one write phase
        // before any hashing/equality callback can collect. `refs` is a
        // caller-owned native slice no collection rewrites, so publish all
        // values before normalizing any one of them.
        //
        // Every slot access goes through the scope's cached root-stack cell:
        // a dict operation touches each of its operands again on the way out,
        // and the free `gc_roots` functions re-resolve the thread local on
        // every call — which on Darwin is an out-of-line `_tlv_get_addr`.
        let root_base = roots.publish(&[obj]);
        roots.publish(refs);
        roots.normalize(root_base, 1 + refs.len());
        Self { roots, root_base }
    }

    /// Re-read a rooted operand.
    ///
    /// The collector rewrites this scope's slots in place, so calling this
    /// again after a step that can collect is what keeps a raw local naming
    /// the dictionary's current address — a `W_DictObject` header comes from
    /// the movable `try_gc_alloc` hook.  Two steps in this file collect: a
    /// strategy promotion, which re-stores every key through a fresh box, and
    /// `object_key_for_checked`, which runs the key's own `__hash__`.
    #[inline]
    pub fn root(&self, index: usize) -> PyObjectRef {
        self.roots.get(self.root_base + index)
    }
}

macro_rules! lock_dict_refs {
    ($guard:ident, $obj:ident) => {
        let $guard = unsafe { DictOperationGuard::new($obj, &[]) };
        let $obj = $guard.root(0);
    };
    ($guard:ident, $obj:ident, $one:ident) => {
        let $guard = unsafe { DictOperationGuard::new($obj, &[$one]) };
        let $obj = $guard.root(0);
        let $one = $guard.root(1);
    };
    // Receiver deliberately left unbound: a body whose first touch of the
    // dictionary follows a step that collects reads it off `root(0)` there,
    // and a binding here would only offer the pre-collection word.
    (defer $guard:ident, $obj:ident, $one:ident) => {
        let $guard = unsafe { DictOperationGuard::new($obj, &[$one]) };
        let $one = $guard.root(1);
    };
    // Both words are re-read from the guard after a collecting step, so the
    // pre-collection bindings would be unused.
    (reroot $guard:ident, $obj:ident, $one:ident) => {
        let $guard = unsafe { DictOperationGuard::new($obj, &[$one]) };
    };
    ($guard:ident, $obj:ident, $one:ident, $two:ident) => {
        let $guard = unsafe { DictOperationGuard::new($obj, &[$one, $two]) };
        let $obj = $guard.root(0);
        let $one = $guard.root(1);
        let $two = $guard.root(2);
    };
    ($guard:ident, $obj:ident, $one:ident, $two:ident, $three:ident) => {
        let $guard = unsafe { DictOperationGuard::new($obj, &[$one, $two, $three]) };
        let $obj = $guard.root(0);
        let $one = $guard.root(1);
        let $two = $guard.root(2);
        let $three = $guard.root(3);
    };
}

/// Compatibility hook for the after-fork sequence. Dict operations are
/// serialized by `rgil`, whose own fork hook recreates the process GIL.
pub fn module_dict_locks_after_fork_child() {}

/// Predicate: dict is in ObjectDictStrategy mode (post-switch).  When
/// true, `dstorage` contains ObjectDictStorage. Mirrors
/// `W_DictMultiObject.get_strategy()` returning
/// `ObjectDictStrategy` vs `ModuleDictStrategy` (`dictmultiobject.py`).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_is_object_strategy(obj: PyObjectRef) -> bool {
    let strategy = (*(obj as *const W_ModuleDictObject)).mstrategy;
    !strategy.is_null() && (*strategy).kind == StrategyKind::Object
}

/// Read-only view of ObjectDictStrategy's erased storage; returns `None`
/// when the dict is still in ModuleDictStrategy mode.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_object_storage<'a>(obj: PyObjectRef) -> Option<&'a ObjectDictStorage> {
    let raw = &*(obj as *const W_ModuleDictObject);
    if !w_module_dict_is_object_strategy(obj) {
        None
    } else {
        Some(&*(raw.dstorage as *const ObjectDictStorage))
    }
}

/// Mutable view of ObjectDictStrategy's erased storage; requires the
/// dict to already be in object-strategy mode.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject` for which
/// `w_module_dict_is_object_strategy(obj)` holds.
#[inline]
pub unsafe fn w_module_dict_object_storage_mut<'a>(obj: PyObjectRef) -> &'a mut ObjectDictStorage {
    let raw = &mut *(obj as *mut W_ModuleDictObject);
    debug_assert!(w_module_dict_is_object_strategy(obj));
    &mut *(raw.dstorage as *mut ObjectDictStorage)
}

/// Mutable view of ObjectDictStrategy's erased storage when present;
/// returns `None` while the dict is still in ModuleDictStrategy mode.
/// Use this variant when the caller does not control the strategy
/// state at the call site (e.g. trait-dispatch entry points that
/// service both pre- and post-switch dicts).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_object_storage_mut_opt<'a>(
    obj: PyObjectRef,
) -> Option<&'a mut ObjectDictStorage> {
    let raw = &mut *(obj as *mut W_ModuleDictObject);
    if !w_module_dict_is_object_strategy(obj) {
        None
    } else {
        Some(&mut *(raw.dstorage as *mut ObjectDictStorage))
    }
}

/// Concrete strategy payload for `ModuleDictStrategy.unerase`.  The erased
/// dispatch holder is the live `mstrategy`; its traced `owner` carries the
/// per-module state (version and global caches) while that strategy is active.
#[inline]
pub unsafe fn w_module_dict_module_strategy_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut crate::celldict::ModuleDictStrategy {
    let holder = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
    debug_assert_eq!(holder.strategy_kind(), StrategyKind::Module);
    debug_assert!(!holder.owner.is_null());
    &mut *(holder.owner as *mut crate::celldict::ModuleDictStrategy)
}

/// Concrete storage behind `ModuleDictStrategy.unerase(w_dict.dstorage)`.
#[inline]
pub unsafe fn w_module_dict_module_storage_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut crate::celldict::ModuleDictStorage {
    debug_assert!(!w_module_dict_is_object_strategy(obj));
    &mut *((*(obj as *mut W_ModuleDictObject)).dstorage as *mut crate::celldict::ModuleDictStorage)
}

#[inline]
pub unsafe fn w_module_dict_module_storage<'a>(
    obj: PyObjectRef,
) -> &'a crate::celldict::ModuleDictStorage {
    debug_assert!(!w_module_dict_is_object_strategy(obj));
    &*((*(obj as *const W_ModuleDictObject)).dstorage as *const crate::celldict::ModuleDictStorage)
}

/// `pypy/objspace/std/celldict.py switch_to_object_strategy`:
///
/// ```python
/// def switch_to_object_strategy(self, w_dict):
///     space = self.space
///     d = self.unerase(w_dict.dstorage)
///     strategy = space.fromcache(ObjectDictStrategy)
///     d_new = strategy.unerase(strategy.get_empty_storage())
///     for key, cell in d.iteritems():
///         d_new[_wrapkey(space, key)] = unwrap_cell(self.space, cell)
///     if self.caches is not None:
///         for cache in self.caches.itervalues():
///             cache.cell = None
///             cache.valid = False
///         self.caches = None
///     w_dict.set_strategy(strategy)
///     w_dict.dstorage = strategy.erase(d_new)
/// ```
///
/// Copies all str entries from the old erased ModuleDictStorage into fresh
/// ObjectDictStorage, preserving insertion order, invalidates the displaced
/// module caches, then replaces both `mstrategy` and `dstorage`. This is the
/// literal shape of `celldict.py ModuleDictStrategy.switch_to_object_strategy`.
///
/// No-op when already in object-strategy mode.
///
/// Residualised for the reason [`w_dict_switch_int_to_object_strategy`] is —
/// the promotion is `IndexMap` construction and refill end to end, so there is
/// no modellable point inside it to put the boundary at.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_module_dict_switch_to_object_strategy(obj: PyObjectRef) {
    lock_dict_refs!(_module_guard, obj);
    if w_module_dict_is_object_strategy(obj) {
        return;
    }
    // The promotion mints young key objects into the fresh object storage
    // (prebuilt-family storage; see `w_module_dict_setitem_str_internal`).
    crate::gc_roots::mark_prebuilt_roots_dirty();
    let roots = crate::gc_roots::push_roots();
    let dict_slot = roots.base();
    let obj = roots.pin_root(obj);
    let raw = &mut *(obj as *mut W_ModuleDictObject);
    let holder = &mut *raw.mstrategy;
    debug_assert_eq!(holder.strategy_kind(), StrategyKind::Module);
    let strategy = &mut *(holder.owner as *mut crate::celldict::ModuleDictStrategy);
    let storage = &mut *(raw.dstorage as *mut crate::celldict::ModuleDictStorage);
    let mut hashes = Vec::new();
    let pairs_base = dict_slot + 1;
    for (k, v) in strategy
        .getiterkeys(storage)
        .zip(strategy.getitervalues(storage))
    {
        let key_obj = crate::celldict::_wrapkey(k);
        let object_key = object_key_for(key_obj);
        hashes.push(object_key.hash);
        roots.publish(&[object_key.obj, v]);
    }
    // Box the empty `RDict` before insert, so `_ll_malloc_entries` stores
    // the array into a collector-owned dict rather than a stack local.
    let new_storage = crate::gc_storage::gc_alloc_young_storage_box(
        object_dict_storage_new(),
        object_dict_storage_gc_type_id(),
    );
    let new_map = &mut *new_storage;
    for (i, &hash) in hashes.iter().enumerate() {
        new_map.insert(
            ObjectKey {
                hash,
                obj: roots.get(pairs_base + 2 * i),
            },
            roots.get(pairs_base + 2 * i + 1),
        );
    }
    // `celldict.py`: every live GlobalCache becomes invalid
    // because the strategy is being swapped out; the JIT must
    // recompile any trace keyed on the prior version.
    strategy.invalidate_caches();
    strategy.mutated();
    raw.jit_force_quasi_immutable("mutate_mstrategy");
    raw.mstrategy = &OBJECT_DICT_STRATEGY_REF as *const DictStrategyRef as *mut DictStrategyRef;
    raw.dstorage = new_storage as GCREF;
    dict_write_barrier(obj);
}

/// `dictmultiobject.py _never_equal_to_string`:
///
/// ```python
/// def _never_equal_to_string(space, w_lookup_type):
///     return (space.is_w(w_lookup_type, space.w_NoneType) or
///             space.is_w(w_lookup_type, space.w_int) or
///             space.is_w(w_lookup_type, space.w_bool) or
///             space.is_w(w_lookup_type, space.w_float))
/// ```
///
/// True when `key`'s type guarantees the key can never `==` any Python
/// string, so a still-ModuleDictStrategy dict can skip the strategy
/// switch and report absence directly.  For all other non-str types
/// (e.g. user-defined classes with custom `__eq__`/`__hash__`), the
/// caller must `switch_to_object_strategy` and re-dispatch the lookup
/// on the unified ObjectDictStrategy storage.
///
/// # Safety
/// `key` must be a valid PyObjectRef.
#[inline]
pub unsafe fn _never_equal_to_string(key: PyObjectRef) -> bool {
    crate::is_none(key) || crate::is_int(key) || crate::is_bool(key) || crate::is_float(key)
}

/// `is W_ModuleDictObject` predicate.  Disambiguates `W_ModuleDictObject`
/// from `W_DictObject` even though both surface as Python's `dict`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_module_dict(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MODULE_DICT_TYPE)
}

/// `dictmultiobject.py W_DictObject.get_strategy` / `:339
/// W_ModuleDictObject.get_strategy` — read the owning strategy.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_get_strategy(
    obj: PyObjectRef,
) -> *mut crate::celldict::ModuleDictStrategy {
    if w_module_dict_is_object_strategy(obj) {
        return std::ptr::null_mut();
    }
    (*(*(obj as *const W_ModuleDictObject)).mstrategy).owner
        as *mut crate::celldict::ModuleDictStrategy
}

/// [`w_module_dict_get_strategy`] for a caller that has not yet proven the
/// layout — null for anything that is not a `W_ModuleDictObject`.
///
/// `mstrategy` sits at the same offset as `W_DictObject`'s `dstrategy`, which
/// is a `&'static dyn DictStrategy` fat pointer, so the unchecked read on a
/// plain dict returns one half of that pair.  Every strategy singleton is a
/// zero-sized static, so the result is non-null and points into read-only
/// memory: a caller testing `.is_null()` to mean "not a module dict" gets
/// `false` and then reads — or atomically writes — past the static.
///
/// # Safety
/// `obj` must be null or a valid `PyObjectRef`.
#[inline]
pub unsafe fn w_module_dict_strategy_or_null(
    obj: PyObjectRef,
) -> *mut crate::celldict::ModuleDictStrategy {
    if obj.is_null() || !is_module_dict(obj) {
        return std::ptr::null_mut();
    }
    w_module_dict_get_strategy(obj)
}

/// Strategy identity stamp used by `_new_next` parity (`dictmultiobject
/// .py:829 self.strategy is self.w_dict.get_strategy()`).
///
/// Returns a `usize` that is stable across the dict's lifetime as long
/// as no strategy transition occurs, and changes whenever the strategy
/// transitions (e.g. `W_ModuleDictObject::switch_to_object_strategy`
/// replacing `mstrategy`). Distinct dicts
/// MAY share the same id when they share the strategy singleton
/// (e.g. two regular W_DictObject instances both on
/// `OBJECT_DICT_STRATEGY`); iterator parity is preserved because
/// the comparison is only performed against the SAME dict the iterator
/// was opened on.
///
/// # Safety
/// `obj` must point to a valid dict (W_DictObject or W_ModuleDictObject).
pub unsafe fn w_dict_strategy_id(obj: PyObjectRef) -> usize {
    if is_module_dict(obj) {
        let m = &*(obj as *const W_ModuleDictObject);
        return m.mstrategy as usize;
    }
    let d = &*(obj as *const W_DictObject);
    // The holder's own address, not `imp`'s data pointer.  Every strategy
    // singleton is a unit struct, so the data half of the `&'static dyn
    // DictStrategy` inside the holder is the address of a zero-sized static and
    // Rust does not guarantee those differ between distinct strategies — two
    // strategies could stamp the same id and a transition between them would be
    // invisible to the iterators that compare this.  `DictStrategyRef` is not
    // zero-sized, so one holder per singleton gives one distinct address per
    // strategy.
    d.dstrategy as *const DictStrategyRef as usize
}

/// `interp_magic.py strategy` — concrete dict strategy class name.
///
/// Module dictionaries currently retain their per-dict
/// `ModuleDictStrategy` allocation after promotion, so consult the live
/// storage mode before the trait object.  This reports the same strategy
/// transition PyPy exposes while the remaining single-`dstorage` structural
/// refactor is carried out.
///
/// # Safety
/// `obj` must point to a live `W_DictObject` or `W_ModuleDictObject`.
pub unsafe fn w_dict_strategy_name(obj: PyObjectRef) -> &'static str {
    if is_module_dict(obj) {
        return if w_module_dict_is_object_strategy(obj) {
            "ObjectDictStrategy"
        } else {
            "ModuleDictStrategy"
        };
    }
    match w_dict_get_strategy(obj).strategy_kind() {
        StrategyKind::Empty => "EmptyDictStrategy",
        StrategyKind::EmptyKwargs => "EmptyKwargsDictStrategy",
        StrategyKind::Object => "ObjectDictStrategy",
        StrategyKind::Bytes => "BytesDictStrategy",
        StrategyKind::Unicode => "UnicodeDictStrategy",
        StrategyKind::Int => "IntDictStrategy",
        StrategyKind::Identity => "IdentityDictStrategy",
        StrategyKind::Kwargs => "KwargsDictStrategy",
        StrategyKind::Module => "ModuleDictStrategy",
        StrategyKind::Map => "MapDictStrategy",
        StrategyKind::Class => "ClassDictStrategy",
    }
}

/// `getitem` for an exact `str` key, taken only where the probe cannot reach
/// Python, and `default` when the key is absent.
///
/// `UnicodeDictStrategy` stores `str` keys only, and its `getitem` sends an
/// exact `str` straight to the probe, so every comparison it can make is
/// builtin.  Every other strategy is refused: `ObjectDictStrategy` holds keys
/// of any type and can reach a stored key's `__eq__`, and `MapDictStrategy`
/// can materialise object storage before it looks anything up.
///
/// The strategy is read here rather than by the caller because the two have to
/// be one operation.  A separate check is a time-of-check window — a
/// concurrent store can devolve the dict in between, and an unchecked probe
/// then reports a raising `__eq__` as a miss.  `None` says the dict was not on
/// that strategy, and the caller owes the checked lookup.
///
/// # Safety
/// `obj` must point to a live dict and `key` to a live exact `str`.
pub unsafe fn w_dict_lookup_str_keyed(
    obj: PyObjectRef,
    key: PyObjectRef,
    default: PyObjectRef,
) -> Option<PyObjectRef> {
    if is_module_dict(obj) {
        return None;
    }
    lock_dict_refs!(_dict_guard, obj, key, default);
    let strategy = w_dict_get_strategy(obj);
    if strategy.strategy_kind() != StrategyKind::Unicode {
        return None;
    }
    let w = strategy.getitem(obj, key);
    Some(if w.is_null() { default } else { w })
}

/// Key-set mutation state captured by dict iterators.
///
/// PyPy's `BaseIteratorImplementation` owns a live iterator over the
/// strategy storage (`dictmultiobject.py`).  That native iterator
/// retains deletion/tombstone state even when the dictionary returns to the
/// same length.  Pyre's `IndexMap::shift_remove` compacts immediately, so the
/// equivalent state lives on the dictionary object itself.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_keys_version(obj: PyObjectRef) -> usize {
    if is_module_dict(obj) {
        return (*(obj as *const W_ModuleDictObject)).keys_version;
    }
    (*(obj as *const W_DictObject)).keys_version
}

/// Advance the live-iterator key state after a key is inserted or removed.
/// Value-only replacement must not call this: both PyPy and Python 3.14 allow
/// replacing a value while iterating over the dictionary.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_bump_keys_version(obj: PyObjectRef) {
    if is_module_dict(obj) {
        let dict = &mut *(obj as *mut W_ModuleDictObject);
        dict.keys_version = dict.keys_version.wrapping_add(1);
    } else {
        let dict = &mut *(obj as *mut W_DictObject);
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
}

/// `dictmultiobject.py W_DictMultiObject.setitem_str`
/// dispatched through ModuleDictStrategy (`celldict.py`):
///
/// ```python
/// def setitem_str(self, key, w_value):
///     self.get_strategy().setitem_str(self, key, w_value)
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_setitem_str(obj: PyObjectRef, key: &str, w_value: PyObjectRef) {
    w_module_dict_setitem_str_internal(obj, key, w_value);
}

/// Compatibility spelling retained until the remaining callers are collapsed.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_module_dict_setitem_str_no_proxy(
    obj: PyObjectRef,
    key: &str,
    w_value: PyObjectRef,
) {
    w_module_dict_setitem_str_internal(obj, key, w_value);
}

unsafe fn w_module_dict_setitem_str_internal(
    mut obj: PyObjectRef,
    key: &str,
    w_value: PyObjectRef,
) {
    // `ModuleDictStrategy.setitem_str` is a cell lookup plus `write_cell`.
    // That arm allocates only through `malloc_typed_stable` and never runs
    // user code, so it does not collect. The guard stays on the object-strategy
    // arm, which wraps a new key or runs a str subclass `__eq__`/`__hash__`.
    if w_module_dict_is_object_strategy(obj) {
        lock_dict_refs!(_module_guard, obj, w_value);
        // Module-dict storage is Box-immortal, reached only by the
        // prebuilt-family root walk; record the store (gc_roots.rs
        // prebuilt-root write tracking).
        crate::gc_roots::mark_prebuilt_roots_dirty();
        // Post-switch: ObjectDictStrategy storage = r_dict(space.eq_w,
        // space.hash_w) per `dictmultiobject.py`; pyre's
        // `dict_keys_equal` enforces the same bucket invariant
        // (Item 1.2).  An overwrite reuses the stored key and updates the
        // value in place; only a genuinely new key wraps a W_UnicodeObject
        // (`setitem_str` keeps `newtext(key)` for the inserted key only —
        // `dictmultiobject.py`).  A new-key wrap dispatches through
        // `dict_keys_equal` so str subclasses honour their `__eq__`/`__hash__`.
        // The receiver and its storage are non-moving, but a movable `w_value`
        // is only a by-value copy of the guard's slot, and both the probe (a
        // str-subclass `__eq__`) and the miss arm's key wrap collect before the
        // store — so hold the value in a slot of its own and read it back at
        // whichever arm writes it.
        let roots = crate::gc_roots::push_roots();
        let value_slot = roots.base();
        let _ = roots.pin_root(w_value);
        let entries = w_module_dict_object_storage_mut(obj);
        dict_write_barrier(obj);
        let key_str = crate::unicodeobject::prebuilt_str_storage(key);
        let idx = dict_entries_index_of_str_or_absent(entries, key_str);
        if idx < 0 {
            let w_key = crate::unicodeobject::w_str_from_storage(key_str);
            dict_entries_insert_object(entries, w_key, roots.get(value_slot));
            w_dict_bump_keys_version(obj);
        } else {
            dict_entries_value_set_at(entries, idx as usize, roots.get(value_slot));
        }
        return;
    }
    {
        let strategy = w_module_dict_module_strategy_mut(obj);
        // `_setitem_str_cell_known` barriers whichever object it stores
        // into: the cell for an in-place rewrite, the storage otherwise.
        let old_len = w_module_dict_module_storage(obj).len();
        crate::with_roots!(obj => strategy.setitem_str(obj, key, w_value));
        if w_module_dict_module_storage(obj).len() != old_len {
            w_dict_bump_keys_version(obj);
        }
    }
}

/// `celldict.py getitem_str`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_getitem_str(
    obj: PyObjectRef,
    key: *mut Utf8Str,
) -> Option<PyObjectRef> {
    // `ModuleDictStrategy.getitem_str` is `getdictvalue_no_unwrapping` plus
    // `unwrap_cell`. Neither collects. The guard stays on the object-strategy
    // arm, whose probe can run a str subclass `__eq__`.
    if w_module_dict_is_object_strategy(obj) {
        lock_dict_refs!(_module_guard, obj);
        if let Some(entries) = w_module_dict_object_storage(obj) {
            // Post-switch ObjectStrategy: route through `dict_keys_equal`
            // (`ObjectDictStrategy.get_empty_storage` r_dict(eq_w, hash_w)) instead of
            // raw String content equality so str-subclass keys with
            // overridden `__eq__`/`__hash__` are reachable from the
            // str-fast-path lookup.  The rstr `STR` key (one word) probe avoids
            // the per-lookup throwaway `W_UnicodeObject` (`getitem_str` parity).
            return dict_entries_get_str(entries, key, 0);
        }
    }
    let strategy = &*w_module_dict_get_strategy(obj);
    let key_str = std::str::from_utf8_unchecked(crate::unicodeobject::utf8_payload_bytes(key));
    strategy.getitem_str(obj, key_str)
}

/// GC root walk over a `W_ModuleDictObject`'s movable references.
///
/// A module dict's authoritative storage is reached only behind raw
/// pointers — the `dstorage` cell map, the post-`switch_to_object_strategy`
/// erased `dstorage` and the module strategy cache registry — neither is inline
/// `gc_ptr_offsets`. The non-moving W_ModuleDictObject's registered
/// `module_dict_object_custom_trace` invokes this walk; the explicit frame
/// root walker does too for raw globals/builtins carriers. Without the walk,
/// `LOAD_GLOBAL`
/// (`w_module_dict_getitem_str`, which reads the authoritative `dstorage`
/// cell map) would observe relocated values through never-forwarded slots.
/// `walk_pyframe_roots` calls this
/// for every live frame's globals and builtins so the real storage stays
/// forwarded across collection.  No-op for non-module dicts.
///
/// `visitor` receives each movable value slot. `walk_module_value_slot`
/// visits a collector-owned `MutableCell` as the slot (its `w_value` is
/// traced through the cell type's pointer offsets) and, for a cell outside
/// the heap, visits the inner `w_value` in place.
///
/// # Safety
/// `obj` must be a valid `PyObjectRef` (null tolerated).  `visitor` must
/// tolerate being called on every movable slot reachable here.
pub unsafe fn w_module_dict_walk_gc_cells(
    obj: PyObjectRef,
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    if obj.is_null() || !is_module_dict(obj) {
        return;
    }
    let md = &mut *(obj as *mut W_ModuleDictObject);
    if !md.dstorage.is_null() {
        visitor(unsafe { &mut *(std::ptr::addr_of_mut!(md.dstorage) as *mut PyObjectRef) });
    }
    if w_module_dict_is_object_strategy(obj) {
        let object_storage = &mut *(md.dstorage as *mut ObjectDictStorage);
        for (key, value) in object_storage.iter_mut_for_trace() {
            // ObjectKey.hash is precomputed and identity-stable across GC
            // moves, so writing through the raw obj slot does not desync
            // the IndexMap bucket index.
            let key_ptr = key as *const ObjectKey as *mut ObjectKey;
            visitor(&mut (*key_ptr).obj);
            crate::celldict::walk_module_value_slot(value, visitor);
        }
        visitor(unsafe { &mut *(object_storage.entries_slot() as *mut PyObjectRef) });
        object_storage
            .visit_indexes(&mut |slot| visitor(unsafe { &mut *(slot as *mut PyObjectRef) }));
    } else {
        let storage = &mut *(md.dstorage as *mut crate::celldict::ModuleDictStorage);
        for (key, value) in storage.iter_mut_for_trace() {
            // `ModuleDictStrategy` keys are the `str` block. The block does
            // not move; the visit keeps it alive.
            let key_ptr = key as *const crate::celldict::StrKey as *mut crate::celldict::StrKey;
            visitor(&mut *(std::ptr::addr_of_mut!((*key_ptr).0) as *mut PyObjectRef));
            crate::celldict::walk_module_value_slot(value, visitor);
        }
        visitor(unsafe { &mut *(storage.entries_slot() as *mut PyObjectRef) });
        storage.visit_indexes(&mut |slot| {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef) });
        });
        w_module_dict_module_strategy_mut(obj).walk_cache_cells(visitor);
    }
}

/// `celldict.py delitem` (str path).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_delitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    w_module_dict_delitem_str_internal(obj, key)
}

/// Compatibility spelling retained until the remaining callers are collapsed.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_module_dict_delitem_str_no_proxy(
    obj: PyObjectRef,
    key: &str,
) -> Option<PyObjectRef> {
    w_module_dict_delitem_str_internal(obj, key)
}

unsafe fn w_module_dict_delitem_str_internal(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    lock_dict_refs!(_module_guard, obj);
    if w_module_dict_is_object_strategy(obj) {
        // Post-switch ObjectStrategy: route through `dict_keys_equal`
        // for the same r_dict bucket reason as `w_module_dict_setitem_str`
        // / `w_module_dict_getitem_str` (Item 2.2).
        let w_key = crate::w_str_new(key);
        let entries = w_module_dict_object_storage_mut(obj);
        if let Some(removed) = entries.remove(&object_key_for(w_key)) {
            w_dict_bump_keys_version(obj);
            return Some(removed);
        }
        return None;
    }
    let removed = {
        let strategy = w_module_dict_module_strategy_mut(obj);
        strategy.delitem_str(obj, key)
    };
    if removed.is_some() {
        w_dict_bump_keys_version(obj);
        return removed;
    }
    None
}

/// `celldict.py length`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_length(obj: PyObjectRef) -> usize {
    lock_dict_refs!(_module_guard, obj);
    if let Some(entries) = w_module_dict_object_storage(obj) {
        entries.len()
    } else {
        let strategy = &*w_module_dict_get_strategy(obj);
        strategy.length(obj)
    }
}

/// True when [`dict_keys_equal`]'s builtin ladder settles `key`'s equality
/// on its own, without `space.eq_w`.
///
/// Restricted to the exact layouts the ladder reads: a subclass may override
/// `__eq__`, a BigInt-backed `int` does not carry the `intval` the int arm
/// reads, and a `float` has no arm at all (yet `1 == 1.0`), so all three are
/// rejected.  An exact tuple is included when every item is itself builtin.
/// `is_exact_tuple` is that exactness test.  `Cls_oo`
/// (`makespecialisedtuple2`) keeps `ob_type` of `SPECIALISED_TUPLE_OO_TYPE`
/// and `w_class` of the canonical tuple class; `is_exact_type` compares
/// `ob_type` while `w_class` is still null and therefore misses the pair
/// `wraptuple2` builds.  The ladder walks items with `w_tuple_getitem`, and
/// a specialised `_ii` / `_ff` boxes on that read, which would collect under
/// the callback-free table borrow.  `TUPLE_USER_TYPE` is rejected on
/// `ob_type` as well: before `init_typeobjects` its `w_class` and
/// `get_instantiate` are both null, and `is_exact_tuple` would accept it.
/// The predicate is one-sided — rejecting a key only costs the caller its
/// fast path.
#[inline]
unsafe fn key_equality_is_builtin(key: PyObjectRef) -> bool {
    if crate::is_long(key) {
        return false;
    }
    if (crate::is_exact_type(key, &crate::INT_TYPE) && crate::is_int(key))
        || (crate::is_exact_type(key, &crate::BOOL_TYPE) && crate::is_bool(key))
        || (crate::is_exact_type(key, &crate::STR_TYPE) && crate::is_str(key))
        || (crate::is_exact_type(key, &crate::bytesobject::BYTES_TYPE)
            && crate::bytesobject::is_bytes(key))
    {
        return true;
    }
    if crate::is_specialised_tuple_ii(key) || crate::is_specialised_tuple_ff(key) {
        return false;
    }
    // Subclass `__eq__` is not the tuple ladder.  Checked before
    // `is_exact_tuple` for the null-instantiate reason in the doc comment.
    if crate::py_type_check(key, &crate::TUPLE_USER_TYPE) || !crate::is_exact_tuple(key) {
        return false;
    }
    let n = crate::w_tuple_len(key);
    for i in 0..n {
        let Some(item) = crate::w_tuple_getitem(key, i as i64) else {
            return false;
        };
        if !key_equality_is_builtin(item) {
            return false;
        }
    }
    true
}

/// Compare two dict keys for equality.
///
/// `a` is the key already stored in the bucket and `b` the incoming one, the
/// order `ll_dict_lookup` passes to `keyeq` (`rordereddict.py`,
/// `keyeq(checkingkey, key)`).  The comparison protocol tries the left
/// operand's `__eq__` first, so the order is observable whenever a key's
/// `__eq__` is asymmetric or has side effects.
///
/// `pypy/objspace/std/dictmultiobject.py ObjectDictStrategy` —
/// the storage is `r_dict(space.eq_w, space.hash_w)` so every key
/// lookup routes through `space.eq_w` (`baseobjspace.py`),
/// which honours user-defined `__eq__`.  pyre-object cannot depend on
/// pyre-interpreter, so we go through the `dict_eq_hook::EQ_W_HOOK`
/// trampoline registered at `pyre-jit::eval` init.  When the hook is
/// not installed (pyre-object lib tests, snapshot tools) we fall back
/// to the limited-type builtin equality below — sufficient for the
/// hashable-builtin smoke tests but not for arbitrary user types.
pub(crate) unsafe fn dict_keys_equal(a: PyObjectRef, b: PyObjectRef) -> bool {
    // Once `space.eq_w` has raised earlier in this probe, nothing may match —
    // not by user comparison and not by identity.  The Rust `Eq` callback
    // cannot abort the `IndexMap` scan, so answering `false` for the rest of it
    // is how a raised probe is made to look like the `r_dict(space.eq_w,
    // space.hash_w)` lookup it stands for: `ll_dict_lookup` propagates at the
    // first comparison, so no later key — however it would have compared — can
    // still be found.  The flag is cleared per op in
    // `object_key_for(_checked)`, so this only fires after a raise within the
    // current probe.
    //
    // This has to sit above the identity shortcut, not below it.  A bucket can
    // hold both a key whose `__eq__` raises and the incoming key's own object;
    // if the raising one is probed first and identity still answered `true`
    // afterwards, `insert` would REPLACE rather than append, and the checked
    // store's `dict_entries_pop_last` undo would then drop an unrelated last
    // entry while leaving the replaced value overwritten.
    if crate::dict_eq_hook::eq_error_pending() {
        return false;
    }
    if std::ptr::eq(a, b) {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }
    if crate::dict_eq_hook::callback_free_probe_active() {
        // A callback-free probe promises that no user code runs while the
        // container is being read, so the ladder below must answer on its
        // own.  A pair it cannot decide breaks the probe: the caller
        // discards this result and re-runs without holding a table borrow
        // across the callback.
        if !(key_equality_is_builtin(a) && key_equality_is_builtin(b)) {
            crate::dict_eq_hook::break_callback_free_probe();
            return false;
        }
    } else if let Some(eq) = unsafe { crate::dict_eq_hook::try_eq_w(a, b) } {
        // `ObjectKey::eq` already gates on `self.hash != other.hash`
        // before calling `dict_keys_equal`, so the bucket invariant
        // (same hash_w → same bucket) is enforced by the cached hash
        // in `ObjectKey.hash`.  No need to re-hash here.
        return eq;
    }
    // Mixed numeric: bool ↔ int (Python: True == 1 and False == 0).
    let a_is_bool = crate::is_bool(a);
    let b_is_bool = crate::is_bool(b);
    let a_is_int = crate::is_int(a);
    let b_is_int = crate::is_int(b);
    if (a_is_int || a_is_bool) && (b_is_int || b_is_bool) {
        let av = if a_is_bool {
            crate::w_bool_get_value(a) as i64
        } else {
            crate::w_int_get_value(a)
        };
        let bv = if b_is_bool {
            crate::w_bool_get_value(b) as i64
        } else {
            crate::w_int_get_value(b)
        };
        return av == bv;
    }
    // Str keys — compare the WTF-8 bytes so lone-surrogate keys compare by
    // content instead of panicking in `w_str_get_value`.
    if crate::is_str(a) && crate::is_str(b) {
        return crate::w_str_get_wtf8(a).as_bytes() == crate::w_str_get_wtf8(b).as_bytes();
    }
    // Bytes keys — compare byte contents.
    if crate::bytesobject::is_bytes(a) && crate::bytesobject::is_bytes(b) {
        return crate::bytesobject::w_bytes_data(a) == crate::bytesobject::w_bytes_data(b);
    }
    // Tuple keys — element-wise compare via dict_keys_equal recursively.
    if crate::is_tuple(a) && crate::is_tuple(b) {
        let la = crate::w_tuple_len(a);
        let lb = crate::w_tuple_len(b);
        if la != lb {
            return false;
        }
        for i in 0..la {
            let ea = crate::w_tuple_getitem(a, i as i64).unwrap_or(std::ptr::null_mut());
            let eb = crate::w_tuple_getitem(b, i as i64).unwrap_or(std::ptr::null_mut());
            if !dict_keys_equal(ea, eb) {
                return false;
            }
        }
        return true;
    }
    // frozenset / set: element-wise containment via the same equality.
    if crate::is_frozenset(a) && crate::is_frozenset(b) {
        let ai = crate::w_set_items(a);
        let bi = crate::w_set_items(b);
        if ai.len() != bi.len() {
            return false;
        }
        return ai
            .iter()
            .all(|&x| bi.iter().any(|&y| dict_keys_equal(x, y)));
    }
    false
}

/// Get a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.getitem`
/// — `w_dict.get_strategy().getitem(w_dict, w_key)`.
pub unsafe fn w_dict_lookup(obj: PyObjectRef, key: PyObjectRef) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj, key);
    nullable_to_option(w_dict_get_strategy(obj).getitem(obj, key))
}

/// True when a regular dict is still on EmptyDictStrategy or
/// EmptyKwargsDictStrategy.  PyPy's `EmptyDictStrategy.pop` does not
/// hash the key: it returns the provided default, or raises KeyError
/// when no default was passed (`dictmultiobject.py`).
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_is_empty_strategy(obj: PyObjectRef) -> bool {
    if is_module_dict(obj) {
        return false;
    }
    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    strategy_is(dstrategy, StrategyKind::Empty) || strategy_is(dstrategy, StrategyKind::EmptyKwargs)
}

/// Fallible variant of [`w_dict_lookup`].  Propagates hash errors
/// via [`DictKeyError`]; the caller recovers the concrete exception
/// from the interpreter-side thread-local error slot.
///
/// `dictmultiobject.py W_DictMultiObject.getitem` — the
/// strategy's `getitem` calls `space.hash_w(w_key)` which may raise.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_lookup_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key);
    if is_module_dict(obj) {
        return w_module_dict_lookup_inner_checked(obj, key);
    }
    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    let strategy = dstrategy.imp;
    if strategy_is(dstrategy, StrategyKind::Empty)
        || strategy_is(dstrategy, StrategyKind::EmptyKwargs)
    {
        hash_key_checked(key)?;
        return Ok(None);
    }
    if strategy_is(dstrategy, StrategyKind::Object) {
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Bytes) {
        if crate::is_bytes(key) {
            return Ok(w_dict_lookup_bytes_strategy(obj, key));
        }
        if _never_equal_to_string(key) {
            return Ok(None);
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Unicode) {
        if crate::is_exact_type(key, &crate::STR_TYPE) {
            return w_dict_lookup_object_strategy_checked(obj, key);
        }
        if _never_equal_to_string(key) {
            return Ok(None);
        }
        w_dict_set_strategy(obj, &crate::dictmultiobject::OBJECT_DICT_STRATEGY_REF);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Int) {
        if crate::listobject::is_plain_int1(key) {
            return Ok(w_dict_lookup_int_strategy(obj, key));
        }
        if _never_equal_to_int(key) {
            return Ok(None);
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Identity) {
        if key_compares_by_identity(key) {
            return Ok(nullable_to_option(strategy.getitem(obj, key)));
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Kwargs) {
        if crate::is_exact_type(key, &crate::STR_TYPE) {
            return Ok(nullable_to_option(strategy.getitem(obj, key)));
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    let result = nullable_to_option(strategy.getitem(obj, key));
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(result)
}

/// Run an object-dict operation with the callback-free guard raised, so
/// `dict_keys_equal` answers from its builtin type ladder and no user
/// `__eq__` observes or mutates the dict mid-probe.
///
/// Yields `None` when a comparison escaped the ladder: the body withholds its
/// mutation in that case, so the dict is untouched and the caller re-runs the
/// operation by scanning entries without holding a table borrow across a
/// callback.
///
/// A macro rather than a function taking `impl FnOnce`: a closure parameter
/// puts an `FnOnce::call_once` in front of every probe and closures have no
/// lifted counterpart, so each dict graph that probes stops there.  Upstream
/// writes the attempt out at the call site (`ll_dict_lookup`'s fast pass
/// before the paranoia restart, `rordereddict.py`) rather than passing
/// it as a callable.
macro_rules! callback_free_dict_op {
    ($body:block) => {{
        crate::dict_eq_hook::begin_callback_free_probe();
        let result = $body;
        if crate::dict_eq_hook::end_callback_free_probe() {
            None
        } else if take_dict_key_error() {
            Some(Err(DictKeyError))
        } else {
            Some(Ok(result))
        }
    }};
}

/// The `ObjectKey`-keyed storage servicing the reentrant scan, for either a
/// regular `W_DictObject` in object strategy (`dstorage`) or a module dict
/// switched to object strategy (`object_storage`).  Both hold the same
/// `IndexMap<ObjectKey, PyObjectRef>` — celldict devolves to
/// `ObjectDictStrategy` (`celldict.py switch_to_object_strategy`), whose lookup is `ll_dict_lookup`
/// like the regular object strategy.
///
/// # Safety
/// `obj` must point to a valid object-strategy dict backing.  A module dict
/// must already be in object strategy (`object_storage` non-null); the caller
/// switches before the scan, and object strategy is terminal, so it stays
/// non-null for the whole scan.
#[inline]
unsafe fn scan_object_entries<'a>(obj: PyObjectRef) -> &'a ObjectDictStorage {
    if is_module_dict(obj) {
        w_module_dict_object_storage(obj)
            .expect("object-strategy module dict must have object_storage")
    } else {
        let dict = &*(obj as *const W_DictObject);
        &*(dict.dstorage as *const ObjectDictStorage)
    }
}

/// The disturbance generation `ll_dict_lookup` compares to detect a callback
/// that reallocated or reset the table in place (`rordereddict.py`):
/// `clear_gen` for a regular dict (bumped by an in-place `clear`),
/// `keys_version` for a module dict (bumped by every key-set mutation, so a
/// superset — a spurious restart on an unrelated mutation only re-scans).
/// `ll_dict_lookup` restarts only on the direct checks (`entries`/`indexes`
/// realloc, or the candidate entry changed) with no generation counter, so a
/// module-dict callback that mutates an unrelated key without a realloc makes
/// the restart — and thus the observable `__eq__` callback count — differ from
/// upstream.  Narrowing that to a clear-only generation needs a per-module-dict
/// counter distinct from `keys_version`, which reintroduces the module-dict
/// fat-pointer layout hazard this fallback was written to avoid; the extra
/// re-scan is safe (never a wrong result), so it stands.
///
/// # Safety
/// `obj` must point to a valid dict backing (`W_DictObject` or module dict).
#[inline]
unsafe fn scan_clear_generation(obj: PyObjectRef) -> usize {
    if is_module_dict(obj) {
        (*(obj as *const W_ModuleDictObject)).keys_version
    } else {
        (*(obj as *const W_DictObject)).clear_gen
    }
}

/// Find `key` in the object storage by scanning same-hash entries one at a
/// time.  The table borrow ends before equality can call user code; if that
/// callback changes the candidate entry, restart from the beginning like
/// `ll_dict_lookup`'s paranoia restart (`rordereddict.py`).  Services
/// both regular object-strategy dicts and object-strategy module dicts via
/// [`scan_object_entries`]/[`scan_clear_generation`].
unsafe fn scan_dict_key_reentrant(
    mut obj: PyObjectRef,
    mut key: ObjectKey,
) -> Result<(Option<usize>, ObjectKey, PyObjectRef), DictKeyError> {
    'restart: loop {
        // Pin the container for the whole attempt so a comparison that moves a
        // nursery dict does not strand the raw `obj` the scan re-derefs after
        // each callback (`w_dict_new` allocates through the movable
        // `try_gc_alloc` hook).  Snapshot the table generation: `ll_dict_lookup`
        // retries on `entries != d.entries` even when the candidate entry is
        // untouched (`rordereddict.py`); a compaction renumbers every
        // entry, so a stale slot would compare the wrong key.
        let _attempt = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        obj = crate::gc_roots::pin_root(obj);
        let (table_generation, clear_generation) = (
            dict_entries_generation(scan_object_entries(obj)),
            scan_clear_generation(obj),
        );
        let mut i = 0;
        loop {
            obj = crate::gc_roots::shadow_stack_get(obj_slot);
            let (stored_hash, stored_obj) = {
                let entries = scan_object_entries(obj);
                // `while i < num_ever_used_items: if entries.valid(i)`
                // (`rordereddict.py:1373`).  A deleted entry leaves a
                // tombstone, so the walk skips it rather than ending there.
                if i >= dict_entries_slot_count(entries) {
                    return Ok((None, key, obj));
                }
                let Some(stored_obj) = dict_entries_key_obj_at(entries, i) else {
                    i += 1;
                    continue;
                };
                (dict_entries_key_hash_at(entries, i), stored_obj)
            };

            if stored_hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored_obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);

                let equal = dict_keys_equal(stored_obj, key.obj);
                obj = crate::gc_roots::shadow_stack_get(obj_slot);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if take_dict_key_error() {
                    return Err(DictKeyError);
                }
                // Validate the paranoia condition before acting on the result:
                // `ll_dict_lookup` restarts even when the comparison answered
                // `true`, because a callback that reallocated the table or moved
                // the candidate leaves the matched index stale
                // (`rordereddict.py`).
                let disturbed = {
                    let entries = scan_object_entries(obj);
                    // `entries != d.entries`: a compaction renumbered the
                    // entries, or a `clear` reset the table in place
                    // (`clear_gen`) — either replaces `d.entries` in
                    // `ll_dict_lookup`'s eyes.
                    dict_entries_generation(entries) != table_generation
                        || scan_clear_generation(obj) != clear_generation
                        // `entries.valid(index) && entries[index].key == checkingkey`.
                        || !dict_entries_key_is_at(entries, i, stored_hash, stored_obj)
                };
                if disturbed {
                    continue 'restart;
                }
                if equal {
                    return Ok((Some(i), key, obj));
                }
            }
            i += 1;
        }
    }
}

/// Internal helper: `ObjectDictStrategy::getitem` body for pyre's
/// W_DictObject. Called only from the strategy trait impl to avoid
/// recursion through `w_dict_lookup`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_lookup_object_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj, key);
    w_dict_lookup_object_strategy_checked(obj, key).unwrap_or(None)
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_lookup_object_strategy_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    lock_dict_refs!(defer _dict_guard, obj, key);
    let object_key = object_key_for_checked(key)?;
    // `object_key_for_checked` runs the key's own `__hash__`, so the
    // dictionary read below is the first use after a collection point: a
    // `W_DictObject` header comes from the movable `try_gc_alloc` hook, and
    // the guard's slot is what the collector rewrites.
    let obj = _dict_guard.root(0);
    if let Some(result) = callback_free_dict_op!({
        let dict = &*(obj as *const W_DictObject);
        let entries = &*(dict.dstorage as *const ObjectDictStorage);
        dict_entries_probe_hashed(entries, object_key.hash, object_key.obj)
    }) {
        return result;
    }
    let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
    Ok(found.map(|i| {
        let dict = &*(obj as *const W_DictObject);
        let entries = &*(dict.dstorage as *const ObjectDictStorage);
        dict_entries_value_at(entries, i)
    }))
}

/// Internal helper: `ModuleDictStrategy::getitem` body for pyre's
/// W_ModuleDictObject — `celldict.py`:
///   * post-`switch_to_object_strategy` path: walk unified entries
///     Vec via `dict_keys_equal`.
///   * str fast path on fresh ModuleDictStrategy: route to
///     `w_module_dict_getitem_str`.
///   * non-str + never-eq-str: fast-return None.
///   * non-str otherwise: promote then walk entries.
///     Called only from the strategy trait impl to avoid recursion
///     through `w_dict_lookup`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_lookup_inner(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_module_guard, obj, key);
    if w_module_dict_object_storage(obj).is_some() {
        return w_module_dict_lookup_object_entries(obj, key);
    }
    if let Some(_ks) = key_as_utf8(key) {
        return w_module_dict_getitem_str(obj, crate::unicodeobject::w_str_storage(key));
    }
    if _never_equal_to_string(key) {
        return None;
    }
    // The strategy switch allocates: read the key back.
    w_module_dict_switch_to_object_strategy(obj);
    let obj = _module_guard.root(0);
    w_module_dict_lookup_object_entries(obj, _module_guard.root(1))
}

/// The unified-entries probe both arms of [`w_module_dict_lookup_inner`]
/// perform — `celldict.py:139 self.unerase(w_dict.dstorage).get(w_key)` once
/// the module dict has been promoted to object storage.
///
/// `@jit.look_inside_iff(jit.isvirtual(d) and jit.isconstant(key))` on
/// `ll_dict_lookup` (`rordereddict.py`).  Returns `None` when the dict is
/// not on object storage, matching the `?` the second call site used to spell.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
fn w_module_dict_lookup_object_entries_iff(obj: PyObjectRef, key: PyObjectRef) -> bool {
    // The storage box is what `ll_dict_lookup` sees as `d`.
    let entries = unsafe { w_module_dict_object_storage(obj) };
    entries.is_some_and(|entries| {
        majit_rlib::jit::isvirtual(entries) && majit_rlib::jit::isconstant(&key)
    })
}

#[majit_macros::look_inside_iff(w_module_dict_lookup_object_entries_iff)]
pub unsafe fn w_module_dict_lookup_object_entries(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    let entries = w_module_dict_object_storage(obj)?;
    entries.get(&object_key_for(key)).copied()
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_module_dict_lookup_inner_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    lock_dict_refs!(_module_guard, obj, key);
    if w_module_dict_object_storage(obj).is_some() {
        let object_key = object_key_for_checked(key)?;
        // The storage is interior to the dictionary, so it is derived from the
        // reloaded receiver rather than carried across the hash above.
        let entries = w_module_dict_object_storage(_module_guard.root(0)).ok_or(DictKeyError)?;
        let hit = dict_entries_probe_hashed(entries, object_key.hash, object_key.obj);
        if take_dict_key_error() {
            return Err(DictKeyError);
        }
        return Ok(hit);
    }
    if let Some(_ks) = key_as_utf8(key) {
        return Ok(w_module_dict_getitem_str(
            obj,
            crate::unicodeobject::w_str_storage(key),
        ));
    }
    if _never_equal_to_string(key) {
        return Ok(None);
    }
    w_module_dict_switch_to_object_strategy(obj);
    // The strategy switch allocates: hash the reloaded key.
    let object_key = object_key_for_checked(_module_guard.root(1))?;
    // Derived after the hash, for the reason the branch above gives.
    let entries = w_module_dict_object_storage(_module_guard.root(0)).ok_or(DictKeyError)?;
    let hit = dict_entries_probe_hashed(entries, object_key.hash, object_key.obj);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    if let Some(v) = hit {
        return Ok(Some(v));
    }
    Ok(None)
}

/// Set a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.setitem`
/// — `w_dict.get_strategy().setitem(w_dict, w_key, w_value)`.
/// Dispatches through the polymorphic strategy slot so module dicts
/// go through `ModuleDictStrategy::setitem` and regular dicts through
/// `ObjectDictStrategy::setitem`.
pub unsafe fn w_dict_store(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    // A stored value is always a live object reference — null is never a
    // dict value (deletes `shift_remove` the entry; there is no null
    // tombstone).  Enforcing it at the store boundary means every lookup
    // return is provably non-null.
    debug_assert!(!value.is_null(), "w_dict_store: null value");
    lock_dict_refs!(_dict_guard, obj, key, value);
    w_dict_get_strategy(obj).setitem(obj, key, value)
}

/// Fallible variant of [`w_dict_store`].
///
/// `dictmultiobject.py W_DictMultiObject.setitem` — the
/// strategy's `setitem` calls `space.hash_w(w_key)` which may raise.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_store_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key, value);
    w_dict_store_checked_inner(obj, key, value, 0, 0)
}

/// [`w_dict_store_checked`] keyed on a `space.hash_w` digest the caller
/// already holds — see [`object_key_hashed`] for why hashing moves out to the
/// caller.  Only the object strategy consults `hash`; the typed strategies key
/// on the unwrapped payload and never reach `space.hash_w`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_store_hashed_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
    hash: i64,
) -> Result<(), DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key, value);
    w_dict_store_checked_inner(obj, key, value, hash, 1)
}

unsafe fn w_dict_store_checked_inner(
    mut obj: PyObjectRef,
    mut key: PyObjectRef,
    mut value: PyObjectRef,
    keyhash: i64,
    hash_known: i64,
) -> Result<(), DictKeyError> {
    debug_assert!(!value.is_null(), "w_dict_store_checked: null value");
    if is_module_dict(obj) {
        return w_module_dict_store_inner_checked(obj, key, value);
    }
    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    let strategy = dstrategy.imp;
    if strategy_is(dstrategy, StrategyKind::Empty) {
        crate::with_roots!(obj, key, value => {
            crate::dictmultiobject::EMPTY_DICT_STRATEGY.switch_to_correct_strategy(obj, key)
        });
        return w_dict_store_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::EmptyKwargs) {
        crate::with_roots!(obj, key, value => {
            crate::dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY.switch_to_correct_strategy(obj, key)
        });
        return w_dict_store_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::Object) {
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::Bytes) {
        if crate::is_bytes(key) {
            w_dict_store_bytes_strategy(obj, key, value);
            return Ok(());
        }
        crate::with_roots!(obj, value => strategy.switch_to_object_strategy(obj));
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::Unicode) {
        if !crate::is_exact_type(key, &crate::STR_TYPE) {
            w_dict_set_strategy(obj, &crate::dictmultiobject::OBJECT_DICT_STRATEGY_REF);
        }
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::Int) {
        if crate::listobject::is_plain_int1(key) {
            w_dict_store_int_strategy(obj, key, value);
            return Ok(());
        }
        crate::with_roots!(obj, value => strategy.switch_to_object_strategy(obj));
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::Identity) {
        if key_compares_by_identity(key) {
            strategy.setitem(obj, key, value);
            return Ok(());
        }
        crate::with_roots!(obj, value => strategy.switch_to_object_strategy(obj));
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy_is(dstrategy, StrategyKind::Kwargs) {
        if crate::is_exact_type(key, &crate::STR_TYPE) {
            strategy.setitem(obj, key, value);
            return Ok(());
        }
        crate::with_roots!(obj, value => strategy.switch_to_object_strategy(obj));
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    if strategy.strategy_kind() == StrategyKind::Map {
        // mapdict.py MapDictStrategy.setitem — exact str keys stay
        // as map nodes; every other key first devolves the instance __dict__
        // to ObjectDictStrategy and then performs the ordinary fallible dict
        // insertion.  Calling the trait's infallible `setitem` here would make
        // that second leg use `object_key_for`, which deliberately consumes a
        // `__hash__` exception.  Keep the checked W_DictMultiObject.setitem
        // path checked across the strategy transition instead.
        if crate::is_exact_type(key, &crate::STR_TYPE) {
            strategy.setitem(obj, key, value);
            return Ok(());
        }
        crate::with_roots!(obj, value => strategy.switch_to_object_strategy(obj));
        return w_dict_store_object_strategy_checked_inner(obj, key, value, keyhash, hash_known);
    }
    strategy.setitem(obj, key, value);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(())
}

/// `dictmultiobject.py DictStrategy.setdefault` (base) +
/// `:749-753 EmptyDictStrategy.setdefault` +
/// `:1073-1079 AbstractTypedStrategy.setdefault` — strategy-dispatched
/// setdefault.
///
/// The Empty case routes through the checked store
/// (`EmptyDictStrategy.setdefault` → `w_dict.setitem`) so an unhashable
/// key raises TypeError before insertion.
///
/// # Safety
/// `obj`, `key`, `value` must be valid PyObjectRef.
pub unsafe fn w_dict_setdefault_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<PyObjectRef, DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key, value);
    if is_module_dict(obj) {
        // `celldict.py ModuleDictStrategy.setdefault`: for a str
        // key, grab the cell and return its value if set, else store the
        // default through the cell; non-str keys switch to object
        // strategy.  `w_module_dict_*_inner_checked` carry the
        // str-vs-object dispatch + cell write; pyre fetches the cell in
        // the lookup and again in the store rather than reusing one cell
        // object (`_setitem_str_cell_known`).
        if let Some(existing) = w_module_dict_lookup_inner_checked(obj, key)? {
            return Ok(existing);
        }
        // A non-str key sends the lookup above through the object strategy,
        // which hashes it, and a `__hash__` written in Python is a collection
        // point that moves the module dict.  Every word the store is handed
        // comes off its root slot, and so does the result: the by-value words
        // taken before the lookup name pre-move headers.
        w_module_dict_store_inner_checked(
            _dict_guard.root(0),
            _dict_guard.root(1),
            _dict_guard.root(2),
        )?;
        return Ok(_dict_guard.root(2));
    }
    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    let strategy = dstrategy.imp;
    // `dictmultiobject.py EmptyDictStrategy.setdefault`:
    //   self.switch_to_correct_strategy(w_dict, w_key)
    //   w_dict.setitem(w_key, w_default)
    //   return w_default
    // `w_dict.setitem` is the public CHECKED store, so an unhashable
    // key raises TypeError *before* insertion (parity with the
    // empty-dict setitem rejecting unhashable).  Route through
    // `w_dict_store_checked` and propagate its `Result` directly:
    // `object_key_for_checked` consumes the hook error slot, so a
    // post-hoc `take_hash_error()` would observe no pending error.
    if strategy_is(dstrategy, StrategyKind::Empty) {
        crate::dictmultiobject::EMPTY_DICT_STRATEGY.switch_to_correct_strategy(obj, key);
        w_dict_store_checked(
            _dict_guard.root(0),
            _dict_guard.root(1),
            _dict_guard.root(2),
        )?;
        return Ok(_dict_guard.root(2));
    }
    if strategy_is(dstrategy, StrategyKind::EmptyKwargs) {
        crate::dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY.switch_to_correct_strategy(obj, key);
        w_dict_store_checked(
            _dict_guard.root(0),
            _dict_guard.root(1),
            _dict_guard.root(2),
        )?;
        return Ok(_dict_guard.root(2));
    }
    if strategy_is(dstrategy, StrategyKind::Object) {
        // `AbstractTypedStrategy.setdefault` (`dictmultiobject.py`) runs
        // `r_dict.setdefault`, a single bucket probe that returns the occupied
        // value or fills the vacant slot.  Run that probe callback-free; a
        // comparison the builtin ladder cannot settle withholds the insert and
        // re-runs the probe over `scan_dict_key_reentrant`.
        let object_key = object_key_for_checked(key)?;
        let obj = _dict_guard.root(0);
        let value = _dict_guard.root(2);
        if let Some(result) = callback_free_dict_op!({
            let dict = &mut *(obj as *mut W_DictObject);
            let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
            let index = entries.index_of(&object_key);
            if crate::dict_eq_hook::callback_free_probe_broken() {
                None
            } else {
                match index {
                    Some(i) => Some(*entries.get_slot(i).unwrap().1),
                    None => {
                        dict_write_barrier(obj);
                        dict_entries_insert_hashed(entries, object_key.hash, object_key.obj, value);
                        w_dict_bump_keys_version(obj);
                        Some(value)
                    }
                }
            }
        }) {
            if let Ok(Some(existing)) = result {
                return Ok(existing);
            }
            result?;
            // `Ok(None)` only arises on a broken probe (which surfaces as the
            // outer `None`); fall through to the scan defensively.
        }
        // `value` is a native local held across the scan; a mid-scan GC in a
        // probing `__eq__` could move it before the insert/return below, so pin
        // and reload it alongside the container the scan returns.
        let _value_root = crate::gc_roots::push_roots();
        let value_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value);
        let (found, object_key, obj) = scan_dict_key_reentrant(obj, object_key)?;
        let value = crate::gc_roots::shadow_stack_get(value_slot);
        if let Some(i) = found {
            let dict = &*(obj as *const W_DictObject);
            let entries = &*(dict.dstorage as *const ObjectDictStorage);
            return Ok(*entries.get_slot(i).unwrap().1);
        }
        crate::dict_eq_hook::begin_callback_free_probe();
        let dict = &mut *(obj as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        dict_write_barrier(obj);
        dict_entries_insert_hashed(entries, object_key.hash, object_key.obj, value);
        let _ = crate::dict_eq_hook::end_callback_free_probe();
        w_dict_bump_keys_version(obj);
        return Ok(value);
    }
    if strategy_is(dstrategy, StrategyKind::Kwargs) {
        // `kwargsdict.py KwargsDictStrategy.setdefault`: exact text
        // keys stay in the parallel arrays; every other key first switches
        // to ObjectDictStrategy and then calls `w_dict.setdefault` again.
        // Preserve that re-dispatch through this checked entry point.  Calling
        // the trait method here would reach ObjectDictStrategy's infallible
        // `object_key_for` path after the switch and consume a failing
        // `__hash__` instead of propagating it.
        if crate::is_exact_type(key, &crate::STR_TYPE) {
            return Ok(strategy.setdefault(obj, key, value));
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_setdefault_checked(obj, key, value);
    }
    // `dictmultiobject.py DictStrategy.setdefault` is `getitem`
    // followed by `setitem`.  Both of those are the infallible spellings here,
    // and a typed strategy answering either for a key of another type reaches
    // the object strategy through them -- where a key that cannot be hashed
    // has nowhere to report itself, so the store went ahead on a key with no
    // hash.  Run the checked pair, which is the same two steps.  The lookup
    // can run a probing `__eq__`, so the three by-value words are slotted and
    // read back after it.
    let roots = crate::gc_roots::push_roots();
    let dict_slot = roots.publish(&[obj, key, value]);
    if let Some(existing) = w_dict_lookup_checked(obj, key)? {
        return Ok(existing);
    }
    let value = roots.get(dict_slot + 2);
    w_dict_store_checked(roots.get(dict_slot), roots.get(dict_slot + 1), value)?;
    Ok(value)
}

/// `dictmultiobject.py DictStrategy.pop` (base) +
/// `:783-787 EmptyDictStrategy.pop` +
/// Whether popping `key` moves the dictionary to the object strategy first —
/// the third arm of `AbstractTypedStrategy.pop`
/// (`dictmultiobject.py`), and the one that hashes the key.
///
/// # Safety
/// `key` must be a valid PyObjectRef.
unsafe fn pop_switches_to_object_strategy(strategy: &DictStrategyRef, key: PyObjectRef) -> bool {
    if strategy_is(strategy, StrategyKind::Bytes) {
        return !crate::is_bytes(key) && !_never_equal_to_string(key);
    }
    if strategy_is(strategy, StrategyKind::Unicode) {
        return !crate::is_exact_type(key, &crate::STR_TYPE) && !_never_equal_to_string(key);
    }
    if strategy_is(strategy, StrategyKind::Int) {
        return !crate::listobject::is_plain_int1(key) && !_never_equal_to_int(key);
    }
    false
}

/// `:1123-1138 AbstractTypedStrategy.pop` — strategy-dispatched pop.
///
/// `EmptyDictStrategy.pop` does NOT hash the key
/// (`:783-787` returns default / raises KeyError directly).
///
/// Returns `Ok(Some(value))` on hit, `Ok(None)` on miss (caller
/// handles default/KeyError), or `Err(DictKeyError)` on hash error.
///
/// # Safety
/// `obj`, `key` must be valid PyObjectRef.
pub unsafe fn w_dict_pop_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key);
    if is_module_dict(obj) {
        match w_module_dict_lookup_inner_checked(obj, key)? {
            Some(val) => {
                w_module_dict_delitem_inner_checked(obj, key)?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    } else {
        let mut dstrategy = (*(obj as *const W_DictObject)).dstrategy;
        let mut strategy = dstrategy.imp;
        // `AbstractTypedStrategy.pop` (`dictmultiobject.py`): a key
        // of the wrong type that could still compare equal moves the
        // dictionary to the object strategy and pops there, and that pop is
        // what hashes the key.  The typed strategies reach the same place
        // through their own `getitem`, which goes by way of the infallible
        // lookup and so swallows the hash error a caller of this is asking
        // for; making the move here keeps it.
        let _roots = crate::gc_roots::push_roots();
        let (mut obj, mut key) = (obj, key);
        if pop_switches_to_object_strategy(dstrategy, key) {
            let obj_slot = crate::gc_roots::pin_roots(&[obj, key]);
            strategy.switch_to_object_strategy(obj);
            obj = crate::gc_roots::shadow_stack_get(obj_slot);
            key = crate::gc_roots::shadow_stack_get(obj_slot + 1);
            dstrategy = (*(obj as *const W_DictObject)).dstrategy;
            strategy = dstrategy.imp;
        }
        if strategy_is(dstrategy, StrategyKind::Object) {
            // `AbstractTypedStrategy.pop` (`dictmultiobject.py`) performs
            // one r_dict lookup followed by removal.  Run that probe
            // callback-free; a comparison the builtin ladder cannot settle
            // withholds the removal and re-runs it over
            // `scan_dict_key_reentrant`.
            let object_key = object_key_for_checked(key)?;
            let obj = _dict_guard.root(0);
            if let Some(result) = callback_free_dict_op!({
                let dict = &mut *(obj as *mut W_DictObject);
                let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
                let index = entries.index_of(&object_key);
                if crate::dict_eq_hook::callback_free_probe_broken() {
                    None
                } else {
                    match index {
                        Some(i) => {
                            let removed = entries.remove_slot(i).unwrap().1;
                            w_dict_bump_keys_version(obj);
                            Some(removed)
                        }
                        None => None,
                    }
                }
            }) {
                return result;
            }
            let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
            if let Some(i) = found {
                let dict = &mut *(obj as *mut W_DictObject);
                let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
                let removed = entries.remove_slot(i).unwrap().1;
                w_dict_bump_keys_version(obj);
                return Ok(Some(removed));
            }
            return Ok(None);
        }
        match strategy.pop(obj, key, None) {
            Ok(val) => {
                if take_dict_key_error() {
                    return Err(DictKeyError);
                }
                Ok(Some(val))
            }
            Err(DictPopError) => {
                if take_dict_key_error() {
                    return Err(DictKeyError);
                }
                Ok(None)
            }
        }
    }
}

/// Internal helper: `ObjectDictStrategy::setitem` body for pyre's
/// W_DictObject — walks the dstorage Vec by `dict_keys_equal`,
/// updates the matching entry or pushes a new one, fires the GC
/// write barrier. Called only from the strategy
/// trait impl to avoid recursion through `w_dict_store`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_store_object_strategy(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj, key, value);
    let _ = w_dict_store_object_strategy_checked(obj, key, value);
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_store_object_strategy_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key, value);
    w_dict_store_object_strategy_checked_inner(obj, key, value, 0, 0)
}

/// Callback-free half of `ll_dict_setitem` (`rordereddict.py`
/// `_ll_dict_setitem_lookup_done`).
///
/// Returns `0` when the entry was written, `1` when a comparison left the
/// builtin ladder and the caller must rerun through
/// [`w_dict_object_setitem_reentrant`], and `-1` when `__hash__` or `__eq__`
/// raised. The concrete exception stays in the pending dict-key slot.
/// `hash_known == 0` hashes `key` once and writes that digest to `hash_out`
/// so the reentrant retry does not call `__hash__` again.
///
/// The table probe is residual (`@jit.dont_look_inside`, `rlib/jit.py`): the
/// entry array and the `ObjectKey` pair stay inside this leaf, and the
/// caller's store only sees a word. Parameters are `*mut PyObject` so the
/// word-ABI trampoline is emitted.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`. `hash_out` must be non-null.
/// When `hash_known != 0`, `hash` is the digest `key` was stored under.
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_object_setitem_callback_free(
    obj: *mut PyObject,
    key: *mut PyObject,
    hash: i64,
    hash_known: i64,
    value: *mut PyObject,
    hash_out: *mut i64,
) -> i64 {
    debug_assert!(!value.is_null(), "w_dict_store_object_strategy: null value");
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, value]);
    let value_slot = obj_slot + 1;
    let object_key = if hash_known != 0 {
        object_key_hashed(key, hash)
    } else {
        match object_key_for_checked(key) {
            Ok(object_key) => object_key,
            Err(_) => return -1,
        }
    };
    *hash_out = object_key.hash;
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let value = crate::gc_roots::shadow_stack_get(value_slot);
    if let Some(result) = callback_free_dict_op!({
        let dict = &mut *(obj as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        let index = entries.index_of(&object_key);
        if !crate::dict_eq_hook::callback_free_probe_broken() {
            dict_write_barrier(obj);
            match index {
                Some(i) => {
                    *entries.get_slot_mut(i).unwrap().1 = value;
                }
                None => {
                    dict_entries_insert_hashed(entries, object_key.hash, object_key.obj, value);
                    dict.keys_version = dict.keys_version.wrapping_add(1);
                }
            }
        }
    }) {
        return match result {
            Ok(()) => 0,
            Err(_) => -1,
        };
    }
    1
}

/// Reentrant half of `ll_dict_setitem`: [`scan_dict_key_reentrant`] then the
/// overwrite or the callback-free placement insert. Same status word as
/// [`w_dict_object_setitem_callback_free`], except `1` is not returned — the
/// scan already decided equality. The fat `(slot, ObjectKey)` pair never
/// leaves this residual.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`, and `hash` must be the digest
/// already computed for `key`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_object_setitem_reentrant(
    obj: *mut PyObject,
    key: *mut PyObject,
    hash: i64,
    value: *mut PyObject,
) -> i64 {
    let object_key = object_key_hashed(key, hash);
    let _value_root = crate::gc_roots::push_roots();
    let value_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(value);
    let (found, object_key, obj) = match scan_dict_key_reentrant(obj, object_key) {
        Ok(found) => found,
        Err(_) => return -1,
    };
    let value = crate::gc_roots::shadow_stack_get(value_slot);
    dict_write_barrier(obj);
    match found {
        Some(i) => {
            let dict = &mut *(obj as *mut W_DictObject);
            let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
            *entries.get_slot_mut(i).unwrap().1 = value;
        }
        None => {
            crate::dict_eq_hook::begin_callback_free_probe();
            let dict = &mut *(obj as *mut W_DictObject);
            let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
            dict_entries_insert_hashed(entries, object_key.hash, object_key.obj, value);
            let _ = crate::dict_eq_hook::end_callback_free_probe();
            dict.keys_version = dict.keys_version.wrapping_add(1);
        }
    }
    0
}

unsafe fn w_dict_store_object_strategy_checked_inner(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
    hash: i64,
    hash_known: i64,
) -> Result<(), DictKeyError> {
    debug_assert!(!value.is_null(), "w_dict_store_object_strategy: null value");
    // `ll_dict_setitem`: one callback-free bucket scan, then the paranoia
    // restart when a comparison leaves the builtin ladder. Both probes are
    // residual leaves; this function only branches on the status word.
    // The digest arrives as a word. Matching `Option<i64>` is a classdef-less
    // `__pos_0` read, which phase A cannot lower.
    let mut keyhash = hash;
    // `ll_dict_setitem` keeps `d`, `key` and `value` live across
    // `d.keyhash(key)`: the probe can run `__hash__` and collect, so the
    // reentrant step reads the words back from the roots.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, key, value]);
    let probed =
        w_dict_object_setitem_callback_free(obj, key, keyhash, hash_known, value, &mut keyhash);
    if probed < 0 {
        return Err(DictKeyError);
    }
    if probed == 0 {
        return Ok(());
    }
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::gc_roots::shadow_stack_get(obj_slot + 1);
    let value = crate::gc_roots::shadow_stack_get(obj_slot + 2);
    if w_dict_object_setitem_reentrant(obj, key, keyhash, value) < 0 {
        return Err(DictKeyError);
    }
    Ok(())
}

/// Internal helper: `ModuleDictStrategy::setitem` body for pyre's
/// W_ModuleDictObject — `celldict.py`:
///   * str keys on a fresh ModuleDictStrategy → `setitem_str`
///   * non-str keys OR already-promoted → `switch_to_object_strategy`
///     and write into the unified dstorage Vec.
///     Called only from the strategy trait impl to avoid recursion
///     through `w_dict_store`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_store_inner(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    lock_dict_refs!(_module_guard, obj, key, value);
    // Prebuilt-family store (see `w_module_dict_setitem_str_internal`).
    crate::gc_roots::mark_prebuilt_roots_dirty();
    if !w_module_dict_is_object_strategy(obj)
        && let Some(ks) = key_as_utf8(key)
    {
        return w_module_dict_setitem_str(obj, ks, value);
    }
    if !w_module_dict_is_object_strategy(obj) {
        w_module_dict_switch_to_object_strategy(obj);
    }
    // The strategy switch allocates and the key's `__hash__` runs Python:
    // hash the reloaded key first, then reload the dict and the value.
    let object_key = object_key_for(_module_guard.root(1));
    let obj = _module_guard.root(0);
    let value = _module_guard.root(2);
    let entries = w_module_dict_object_storage_mut(obj);
    let inserted =
        dict_entries_insert_hashed(entries, object_key.hash, object_key.obj, value).is_none();
    if inserted {
        w_dict_bump_keys_version(obj);
    }
    dict_write_barrier(obj);
}

/// Object-strategy `ll_dict_setitem` for a module dict that has already
/// switched. Returns `0` on success and `-1` when `__hash__` or `__eq__`
/// raised. A raising probe that appended a spurious entry is popped, matching
/// the previous inline body. The `ObjectKey` and the entry array stay inside
/// this residual.
///
/// # Safety
/// `obj` must point to a valid object-strategy `W_ModuleDictObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_module_dict_store_object_key(
    obj: *mut PyObject,
    key: *mut PyObject,
    value: *mut PyObject,
) -> i64 {
    lock_dict_refs!(reroot _leaf_guard, obj, value);
    let object_key = match object_key_for_checked(key) {
        Ok(object_key) => object_key,
        Err(_) => return -1,
    };
    // `__hash__` may have run a moving collection.
    let obj = _leaf_guard.root(0);
    let value = _leaf_guard.root(1);
    let entries = w_module_dict_object_storage_mut(obj);
    let previous = dict_entries_insert_hashed(entries, object_key.hash, object_key.obj, value);
    if take_dict_key_error() {
        dict_entries_pop_last(entries);
        return -1;
    }
    if previous.is_none() {
        w_dict_bump_keys_version(obj);
    }
    dict_write_barrier(obj);
    0
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_module_dict_store_inner_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), DictKeyError> {
    lock_dict_refs!(_module_guard, obj, key, value);
    // `celldict.py` `ModuleDictStrategy.setitem_str` stays in this graph.
    // A non-str key promotes, then one object-strategy `ll_dict_setitem`.
    if !w_module_dict_is_object_strategy(obj)
        && let Some(ks) = key_as_utf8(key)
    {
        w_module_dict_setitem_str(obj, ks, value);
        return Ok(());
    }
    if !w_module_dict_is_object_strategy(obj) {
        w_module_dict_switch_to_object_strategy(obj);
    }
    let obj = _module_guard.root(0);
    let key = _module_guard.root(1);
    let value = _module_guard.root(2);
    if w_module_dict_store_object_key(obj, key, value) < 0 {
        return Err(DictKeyError);
    }
    Ok(())
}

/// True only for a regular `W_DictObject` on EmptyDictStrategy.
/// `W_ModuleDictObject` also surfaces as a
/// Python `dict`, but it has a different Rust layout and must take the
/// polymorphic item-loop path.
///
/// Mirrors the destination test in `dictmultiobject.py update1_dict_dict`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_is_regular_empty(obj: PyObjectRef) -> bool {
    if is_module_dict(obj) {
        return false;
    }
    let dict = &*(obj as *const W_DictObject);
    dict.dstrategy.strategy_kind() == crate::dictmultiobject::StrategyKind::Empty
}

/// Adopt a freshly copied regular-dict storage into an empty regular
/// destination for `dict.update(dict)`'s PyPy fast path.
///
/// PyPy `update1_dict_dict` performs:
/// `w_copy = w_data.get_strategy().copy(w_data); w_dict.set_strategy(...);
/// w_dict.dstorage = w_copy.dstorage`.  Empty dest storage is
/// `erased(None)`; this helper installs the copy's strategy/storage
/// and fires the explicit GC write barrier that RPython field stores
/// would get from the GC.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_adopt_regular_copy_for_empty_update(dst: PyObjectRef, w_copy: PyObjectRef) {
    debug_assert!(!is_module_dict(dst));
    debug_assert!(!is_module_dict(w_copy));

    let dst_dict = &mut *(dst as *mut W_DictObject);
    let copy_dict = &*(w_copy as *const W_DictObject);
    let old_dstorage = dst_dict.dstorage;

    // PyPy aliases the GC storage pointer here:
    // `w_dict.dstorage = w_copy.dstorage`.  The storage box is an independent
    // GC object, not memory owned by `w_copy`, so both dicts must keep tracing
    // it until the consumed temporary is swept.  Clearing the temporary's
    // field would leave a live W_DictObject whose strategy cannot trace its
    // null storage during that same collection. `setfield_gc` records the
    // old-to-young edge on `dst`.
    install_dict_storage(dst, copy_dict.dstorage, copy_dict.dstrategy);

    // Empty dest storage is `erased(None)`.  The off-GC side-table case
    // is freed via the shared guarded helper.
    dealloc_offgc_object_dict_storage(old_dstorage);
}

/// Get a value by int key (convenience wrapper).  Wraps the raw i64
/// into a `W_IntObject` and dispatches through `w_dict_lookup` so the
/// strategy-slot path applies uniformly to W_DictObject and
/// W_ModuleDictObject.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_getitem(obj: PyObjectRef, key: i64) -> Option<PyObjectRef> {
    w_dict_lookup(obj, crate::w_int_new(key))
}

/// Set a value by int key (convenience wrapper).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_setitem(obj: PyObjectRef, key: i64, value: PyObjectRef) {
    w_dict_store(obj, crate::w_int_new(key), value)
}

/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.getitem_str`
/// — `w_dict.get_strategy().getitem_str(w_dict, key)`.  Dispatches
/// through the polymorphic strategy slot so module dicts go through
/// `ModuleDictStrategy::getitem_str` and regular dicts through
/// `ObjectDictStrategy::getitem_str`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_getitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    w_dict_getitem_str_hashed(obj, key, 0)
}

/// [`w_dict_getitem_str`] for a caller that already holds `key`'s digest —
/// [`crate::unicodeobject::w_str_hash_memoized`] on the string object `key` was
/// borrowed from (`rstr.py ll_strhash`).  Zero leaves the strategy to
/// hash the borrowed bytes, which is what every other caller gets.
///
/// # Safety
/// Same as [`w_dict_getitem_str`].
pub unsafe fn w_dict_getitem_str_hashed(
    obj: PyObjectRef,
    key: &str,
    hash: i64,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj);
    let key = crate::unicodeobject::prebuilt_str_storage(key);
    nullable_to_option(w_dict_get_strategy(obj).getitem_str_hashed(obj, key, hash))
}

/// Error-propagating sibling of [`w_dict_getitem_str`], the str-keyed
/// counterpart of [`w_dict_lookup_checked`].
///
/// A borrowed-`&str` probe still runs `dict_keys_equal` against whatever the
/// bucket holds, so a stored non-string key whose hash collides can reach a
/// user `__eq__`.  When that raises, the callback pair records the error and
/// reports "not equal", which `Option` alone renders as an ordinary miss —
/// a name lookup would then continue as though the name were absent.  Drain
/// the flag so the caller can surface the exception instead.
///
/// # Safety
/// `obj` must point to a valid dict.
pub unsafe fn w_dict_getitem_str_checked(
    obj: PyObjectRef,
    key: &str,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    w_dict_getitem_str_checked_hashed(obj, key, 0)
}

/// [`w_dict_getitem_str_checked`] for a caller that already holds `key`'s
/// digest, as [`w_dict_getitem_str_hashed`] takes it.
///
/// # Safety
/// `obj` must point to a valid dict.
pub unsafe fn w_dict_getitem_str_checked_hashed(
    obj: PyObjectRef,
    key: &str,
    hash: i64,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    let hit = w_dict_getitem_str_hashed(obj, key, hash);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(hit)
}

/// Internal helper for `ObjectDictStrategy::getitem_str`. Kept as a free
/// function so the strategy trait impl can share the rstr `STR` key lookup.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.  The sole caller is
/// `ObjectDictStrategy::getitem_str`, which only fires when the
/// receiver's `dstrategy` slot is `OBJECT_DICT_STRATEGY`; module
/// dicts route through `ModuleDictStrategy::getitem_str` →
/// `w_module_dict_getitem_str` instead.
pub unsafe fn w_dict_getitem_str_object_strategy(
    obj: PyObjectRef,
    key: *mut Utf8Str,
) -> Option<PyObjectRef> {
    w_dict_getitem_str_object_strategy_hashed(obj, key, 0)
}

/// [`w_dict_getitem_str_object_strategy`] for a caller that already holds
/// `key`'s digest (`rstr.py ll_strhash`).
///
/// # Safety
/// Same as [`w_dict_getitem_str_object_strategy`].
pub unsafe fn w_dict_getitem_str_object_strategy_hashed(
    obj: PyObjectRef,
    key: *mut Utf8Str,
    hash: i64,
) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const ObjectDictStorage);
    // Under the `dict_keys_equal` hash/eq pair, str-keyed lookups hash on the
    // str hash and compare on WTF-8 byte equality.  The rstr `STR` key (one
    // word) probe avoids the per-lookup throwaway `W_UnicodeObject`
    // (`getitem_str` parity).
    dict_entries_get_str(entries, key, hash)
}

/// `W_DictMultiObject.setitem_str` — `w_dict.get_strategy().setitem_str(w_dict, key, value)`.
/// The module-dict arm is looked inside through [`w_dict_setitem_str_hashed`]
/// into `ModuleDictStrategy.setitem_str`; the generic arm is the residual
/// [`w_dict_setitem_str_hashed_generic`].
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_setitem_str(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    w_dict_setitem_str_hashed(obj, key, 0, value)
}

/// [`w_dict_setitem_str`] for a caller that already holds `key`'s digest, as
/// [`w_dict_getitem_str_hashed`] takes it.
///
/// The module-dict arm (`is_module_dict && !object_strategy`) is looked
/// inside and reaches `ModuleDictStrategy.setitem_str` via
/// [`w_module_dict_setitem_str`]. `ModuleDictStrategy` has no
/// `setitem_str_hashed` override; the trait default forwards to
/// `setitem_str` and discards `hash`, which this arm matches. The generic
/// arm residualises through [`w_dict_setitem_str_hashed_generic`].
///
/// # Safety
/// Same as [`w_dict_setitem_str`].
pub unsafe fn w_dict_setitem_str_hashed(
    obj: PyObjectRef,
    key: &str,
    hash: i64,
    value: PyObjectRef,
) {
    // STORE_NAME / STORE_GLOBAL on a module dict still in
    // `ModuleDictStrategy`. That arm is `setitem_str` → `write_cell` and
    // does not collect, so the guard would root operands nothing moves.
    // Every other strategy, including a module dict after
    // `switch_to_object_strategy`, can run user code and keeps the guard.
    if is_module_dict(obj) && !w_module_dict_is_object_strategy(obj) {
        let _ = hash;
        w_module_dict_setitem_str(obj, key, value);
        return;
    }
    w_dict_setitem_str_hashed_generic(obj, key, hash, value)
}

/// Generic (non-module) arm of [`w_dict_setitem_str_hashed`].
///
/// The non-module strategies can run user `__hash__`/`__eq__` and mutate
/// storage under the dict lock.
#[majit_macros::dont_look_inside]
/// # Safety
/// Same as [`w_dict_setitem_str_hashed`].
unsafe fn w_dict_setitem_str_hashed_generic(
    obj: PyObjectRef,
    key: &str,
    hash: i64,
    value: PyObjectRef,
) {
    lock_dict_refs!(_dict_guard, obj, value);
    let key = crate::unicodeobject::prebuilt_str_storage(key);
    w_dict_get_strategy(obj).setitem_str_hashed(obj, key, hash, value)
}

/// Compatibility spelling retained until the remaining callers are collapsed.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_setitem_str_no_proxy(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    w_dict_setitem_str(obj, key, value);
}

/// Compatibility spelling retained until the remaining callers are collapsed.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_delitem_str_no_proxy(obj: PyObjectRef, key: &str) -> bool {
    w_dict_delitem_str(obj, key)
}

/// WTF-8 keyed equivalent of `space.getitem_str` — `getitem_str` is itself
/// a fast path of `space.getitem`, so a key that is valid UTF-8 takes the str
/// fast path and a lone-surrogate key wraps into a `W_UnicodeObject` and
/// routes through the general `w_dict_lookup` (`space.getitem`).
///
/// The lone-surrogate arm is a requirement, not a tuning choice: the str
/// probe keys an `IndexMap` on `try_hash_str` of the STR, and a name
/// carrying a lone surrogate has no `&str` view to hash — `w_str_get_value`
/// panics on one.  Stored keys are unaffected either way, since the probe
/// compares WTF-8 bytes (`dict_entries_probe_str`) and never takes a `&str`
/// view of an entry.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// Whether `key`'s WTF-8 bytes are valid UTF-8 (no lone surrogate).
///
/// Residualised (`dont_look_inside`) so the `Result<&str, Utf8Error>` that
/// `as_str` returns never crosses the trace boundary; the residual is a
/// single-word `bool`.  The wtf8-keyed dict adapters branch on this in place
/// of `match key.as_str()`.
#[majit_macros::dont_look_inside]
pub unsafe fn wtf8_key_is_utf8(key: &rustpython_wtf8::Wtf8) -> bool {
    key.as_str().is_ok()
}

/// Reinterpret `key`'s bytes as `&str` without re-validating.
///
/// Only ever called on the branch [`wtf8_key_is_utf8`] already validated, so
/// the reinterpret is infallible.  Folded to identity in
/// `front/mir.rs::wtf8_string_identity_alias` (`&Wtf8` and `&str` share the
/// one immutable string value), so the call disappears and this body is never
/// lifted.
pub unsafe fn wtf8_key_as_str_unchecked(key: &rustpython_wtf8::Wtf8) -> &str {
    core::str::from_utf8_unchecked(key.as_bytes())
}

/// Wrap a lone-surrogate `key` into a `W_UnicodeObject` for the object-keyed
/// slow path.
///
/// Residualised (`dont_look_inside`) as one objectptr call wrapping both
/// `to_wtf8_buf` (external, unliftable) and `w_str_from_wtf8`.  Deliberately
/// not a global `to_wtf8_buf` identity fold: other callers mutate the owned
/// `Wtf8Buf`.
#[majit_macros::dont_look_inside]
pub unsafe fn wtf8_surrogate_key_str_object(key: &rustpython_wtf8::Wtf8) -> PyObjectRef {
    crate::w_str_from_wtf8_managed(key.to_wtf8_buf())
}

pub unsafe fn w_dict_getitem_wtf8(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
) -> Option<PyObjectRef> {
    if wtf8_key_is_utf8(key) {
        w_dict_getitem_str(obj, wtf8_key_as_str_unchecked(key))
    } else {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(obj);
        let w_key = crate::gc_roots::pin_root(wtf8_surrogate_key_str_object(key));
        w_dict_lookup(crate::gc_roots::shadow_stack_get(obj_slot), w_key)
    }
}

/// Fallible sibling of [`w_dict_getitem_wtf8`].
///
/// The probe compares against whatever the bucket holds, so a stored
/// non-string key whose hash collides with `key` can reach a user `__eq__`
/// that raises.  The unchecked spelling reports that as a miss; this one
/// surfaces it, and the caller recovers the concrete exception from the
/// interpreter-side error slot.
///
/// Mirrors the unchecked spelling's dispatch so a valid-UTF-8 key keeps the
/// borrowed-`&str` probe and its `getitem_str` parity — routing every name
/// through [`w_dict_lookup_checked`] instead would wrap a throwaway
/// `W_UnicodeObject` per lookup, which is exactly what
/// `w_dict_getitem_str_object_strategy` exists to avoid.
///
/// The two arms need different checked leaves because only one of them
/// swallows: `w_dict_lookup_object_strategy` (:2668) is
/// `..._checked(..).unwrap_or(None)`, so draining the flag after the
/// unchecked `w_dict_lookup` always reads `false`, while
/// `dict_entries_get_str` (:451) leaves the probe's flag standing for
/// [`w_dict_getitem_str_checked`] to take.  A raising comparison is reachable
/// only from a bucket holding a non-string key, i.e. only under the object
/// strategy — whose `getitem_str` (:6219) is the borrowed probe — so the
/// `&str` arm loses no error.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_getitem_wtf8_checked(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    if wtf8_key_is_utf8(key) {
        w_dict_getitem_str_checked(obj, wtf8_key_as_str_unchecked(key))
    } else {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(obj);
        let w_key = crate::gc_roots::pin_root(wtf8_surrogate_key_str_object(key));
        w_dict_lookup_checked(crate::gc_roots::shadow_stack_get(obj_slot), w_key)
    }
}

/// WTF-8 keyed equivalent of `space.setitem_str` — `setitem_str` is itself
/// a fast path of `space.setitem`, so a key that is valid UTF-8 takes the
/// str fast path (keeping an ASCII/Unicode dict on its strategy) and a
/// lone-surrogate key wraps into a `W_UnicodeObject` and routes through the
/// general `w_dict_store` (`space.setitem`).
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_setitem_wtf8(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
    value: PyObjectRef,
) {
    if wtf8_key_is_utf8(key) {
        w_dict_setitem_str(obj, wtf8_key_as_str_unchecked(key), value);
    } else {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::pin_roots(&[obj, value]);
        let w_key = crate::gc_roots::pin_root(wtf8_surrogate_key_str_object(key));
        w_dict_store(
            crate::gc_roots::shadow_stack_get(obj_slot),
            w_key,
            crate::gc_roots::shadow_stack_get(obj_slot + 1),
        );
    }
}

/// Compatibility spelling retained until the remaining callers are collapsed.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` or `W_ModuleDictObject`.
pub unsafe fn w_dict_setitem_wtf8_no_proxy(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
    value: PyObjectRef,
) {
    w_dict_setitem_wtf8(obj, key, value);
}

/// WTF-8 keyed sibling of `w_dict_delitem_str_no_proxy`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` or `W_ModuleDictObject`.
pub unsafe fn w_dict_delitem_wtf8_no_proxy(obj: PyObjectRef, key: &rustpython_wtf8::Wtf8) -> bool {
    lock_dict_refs!(_dict_guard, obj);
    if is_module_dict(obj) {
        if let Ok(key) = key.as_str() {
            return w_module_dict_delitem_str_no_proxy(obj, key).is_some();
        }
        if !w_module_dict_is_object_strategy(obj) {
            w_module_dict_switch_to_object_strategy(obj);
        }
        let obj = _dict_guard.root(0);
        let entries = w_module_dict_object_storage_mut(obj);
        let _roots = crate::gc_roots::push_roots();
        let w_key = crate::gc_roots::pin_root(crate::w_str_from_wtf8_managed(key.to_wtf8_buf()));
        let removed = entries.remove(&object_key_for(w_key)).is_some();
        if removed {
            w_dict_bump_keys_version(obj);
        }
        return removed;
    }
    let dict = &mut *(obj as *mut W_DictObject);
    // Detect via `strategy_kind()`; the strategy singletons are
    // zero-sized statics whose addresses can coalesce, so `std::ptr::eq`
    // on them is unreliable (an `Object`-strategy dict could otherwise
    // be misread as `Empty` and skip the removal entirely).
    use crate::dictmultiobject::StrategyKind;
    match dict.dstrategy.strategy_kind() {
        StrategyKind::Empty => return false,
        StrategyKind::Object => {}
        StrategyKind::Map | StrategyKind::Class => {
            let _roots = crate::gc_roots::push_roots();
            let obj_slot = crate::gc_roots::shadow_stack_len();
            let _ = crate::gc_roots::pin_root(obj);
            let w_key =
                crate::gc_roots::pin_root(crate::w_str_from_wtf8_managed(key.to_wtf8_buf()));
            let obj = crate::gc_roots::shadow_stack_get(obj_slot);
            let dict = &mut *(obj as *mut W_DictObject);
            return dict.dstrategy.delitem(obj, w_key);
        }
        _ => dict.dstrategy.switch_to_object_strategy(obj),
    }
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let w_key = crate::gc_roots::pin_root(crate::w_str_from_wtf8_managed(key.to_wtf8_buf()));
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
    let removed = entries.remove(&object_key_for(w_key)).is_some();
    if removed {
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
    removed
}

/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.descr_delitem`
/// — `space.delitem(w_dict, w_key)` which calls
/// `w_dict.get_strategy().delitem(w_dict, w_key)`.  PyPy has no
/// `delitem_str` fast path on the trait (only `setitem_str` and
/// `getitem_str`); pyre's str-keyed convenience wrapper routes
/// through `w_dict_delitem(obj, w_str_new(key))` for parity.
///
/// Returns `true` when the key was present in the strategy storage.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_delitem_str(obj: PyObjectRef, key: &str) -> bool {
    let w_key = crate::w_str_new(key);
    w_dict_delitem(obj, w_key)
}

/// Remove an entry by arbitrary key (str or non-str).  Returns `true`
/// when the key was present.  Mirrors `pypy/objspace/std/dictmultiobject.py
/// W_DictMultiObject.delitem` — PyPy's flat strategy walks the entries
/// list, comparing each key by `space.eq_w` (the `dict_keys_equal`
/// helper here), so `del d[1]` on an int key removes the entry.  The
/// previous str-only `w_dict_delitem_str` left non-str entries
/// untouched, breaking `dict.pop(int_key)`'s after-pop deletion.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.delitem`
/// — `w_dict.get_strategy().delitem(w_dict, w_key)`.  Dispatches
/// through the polymorphic strategy slot so module dicts go through
/// `ModuleDictStrategy::delitem` and regular dicts through
/// `ObjectDictStrategy::delitem`.
pub unsafe fn w_dict_delitem(obj: PyObjectRef, key: PyObjectRef) -> bool {
    lock_dict_refs!(_dict_guard, obj, key);
    w_dict_get_strategy(obj).delitem(obj, key)
}

/// Fallible variant of [`w_dict_delitem`].
///
/// `dictmultiobject.py W_DictMultiObject.delitem` — the
/// strategy's `delitem` calls `space.hash_w(w_key)` which may raise.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_delitem_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<bool, DictKeyError> {
    lock_dict_refs!(_dict_guard, obj, key);
    if is_module_dict(obj) {
        return w_module_dict_delitem_inner_checked(obj, key);
    }
    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    let strategy = dstrategy.imp;
    if strategy_is(dstrategy, StrategyKind::Empty)
        || strategy_is(dstrategy, StrategyKind::EmptyKwargs)
    {
        hash_key_checked(key)?;
        return Ok(false);
    }
    if strategy_is(dstrategy, StrategyKind::Object) {
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Bytes) {
        if crate::is_bytes(key) {
            return Ok(w_dict_delitem_bytes_strategy(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Unicode) {
        if crate::is_exact_type(key, &crate::STR_TYPE) {
            return w_dict_delitem_object_strategy_checked(obj, key);
        }
        w_dict_set_strategy(obj, &crate::dictmultiobject::OBJECT_DICT_STRATEGY_REF);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Int) {
        if crate::listobject::is_plain_int1(key) {
            return Ok(w_dict_delitem_int_strategy(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Identity) {
        if key_compares_by_identity(key) {
            return Ok(strategy.delitem(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(dstrategy, StrategyKind::Kwargs) {
        strategy.switch_to_object_strategy(obj);
        let obj = _dict_guard.root(0);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    let removed = strategy.delitem(obj, key);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(removed)
}

/// `dictmultiobject.py W_DictMultiObject.nondescr_delitem_if_value_is`
/// / `rordereddict.py ll_dict_delitem_if_value_is`: perform one key probe
/// and remove the matched entry only when its current value is pointer-identical
/// to `value`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_delitem_if_value_is_checked(
    mut obj: PyObjectRef,
    key: PyObjectRef,
    mut value: PyObjectRef,
) -> Result<bool, DictKeyError> {
    if is_module_dict(obj) {
        if !w_module_dict_is_object_strategy(obj) {
            crate::with_roots!(obj, value => w_module_dict_switch_to_object_strategy(obj));
        }
        let object_key = crate::with_roots!(obj, value => object_key_for_checked(key))?;
        if let Some(result) = callback_free_dict_op!({
            let entries = w_module_dict_object_storage_mut(obj);
            match entries.index_of(&object_key) {
                Some(index)
                    if entries
                        .get_slot(index)
                        .is_none_or(|(_, stored)| *stored != value) =>
                {
                    false
                }
                Some(index) => {
                    entries.remove_slot(index);
                    w_dict_bump_keys_version(obj);
                    true
                }
                None => false,
            }
        }) {
            return result;
        }
        // A probing `__eq__`/`__hash__` escaped the callback-free ladder; pin
        // `value` across the reentrant scan (a mid-scan GC could move it before
        // the identity check), then remove in place on the module storage.
        let _value_root = crate::gc_roots::push_roots();
        let value_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value);
        let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
        let value = crate::gc_roots::shadow_stack_get(value_slot);
        let Some(index) = found else {
            return Ok(false);
        };
        let entries = w_module_dict_object_storage_mut(obj);
        if entries
            .get_slot(index)
            .is_none_or(|(_, stored)| *stored != value)
        {
            return Ok(false);
        }
        entries.remove_slot(index);
        w_dict_bump_keys_version(obj);
        return Ok(true);
    }

    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    let strategy = dstrategy.imp;
    if strategy.strategy_kind() != StrategyKind::Object {
        crate::with_roots!(obj, value => strategy.switch_to_object_strategy(obj));
    }
    let object_key = crate::with_roots!(obj, value => object_key_for_checked(key))?;
    if let Some(result) = callback_free_dict_op!({
        let dict = &mut *(obj as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        match entries.index_of(&object_key) {
            Some(index)
                if entries
                    .get_slot(index)
                    .is_none_or(|(_, stored)| *stored != value) =>
            {
                false
            }
            Some(index) => {
                entries.remove_slot(index);
                dict.keys_version = dict.keys_version.wrapping_add(1);
                true
            }
            None => false,
        }
    }) {
        return result;
    }
    // `value` is a native local held across the scan; a mid-scan GC in a
    // probing `__eq__` could move it before the identity check below, so pin
    // and reload it alongside the container the scan returns.
    let _value_root = crate::gc_roots::push_roots();
    let value_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(value);
    let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
    let value = crate::gc_roots::shadow_stack_get(value_slot);
    let Some(index) = found else {
        return Ok(false);
    };
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
    if entries
        .get_slot(index)
        .is_none_or(|(_, stored)| *stored != value)
    {
        return Ok(false);
    }
    entries.remove_slot(index);
    dict.keys_version = dict.keys_version.wrapping_add(1);
    Ok(true)
}

/// Reposition `k` to the back (`last`) or front of a native insertion-ordered
/// typed store, bumping `keys_version` only on a real move.  Returns whether the
/// key existed.  Shared by the Int/Bytes fast paths of
/// [`w_dict_move_to_end_checked`]; the caller has proven the strategy so the
/// `RDict<K, _, S>` cast matches `dstorage`.
///
/// # Safety
/// `dict` must point to a valid `W_DictObject` whose `dstorage` is an
/// `RDict<K, PyObjectRef, S>`.
unsafe fn typed_move_to_end<
    K: std::hash::Hash
        + Eq
        + Copy
        + crate::rordereddict::EntryDummy
        + crate::rordereddict::GcRefOffsets,
    S: std::hash::BuildHasher,
>(
    dict: *mut W_DictObject,
    k: &K,
    last: bool,
) -> bool
where
    (K, PyObjectRef): crate::rordereddict::GcEntriesType,
{
    let dict = &mut *dict;
    let entries = &mut *(dict.dstorage as *mut crate::rordereddict::RDict<K, PyObjectRef, S>);
    match entries.index_of(k) {
        Some(index) => {
            if entries.move_slot_to_end(index, last) {
                dict.keys_version = dict.keys_version.wrapping_add(1);
            }
            true
        }
        None => false,
    }
}

/// PyPy `W_DictMultiObject.nondescr_move_to_end` (`dictmultiobject.py`):
/// reposition an existing key to the back (`last`) or front of the insertion
/// order, bypassing any subclass hooks.  Locates the entry callback-free,
/// falling back to the reentrant scan when a probing `__eq__`/`__hash__`
/// escapes the builtin ladder, then shifts it with `IndexMap::move_index` (the
/// in-place equivalent of `objectmodel.move_to_end`, `:1175`).  Returns whether
/// the key existed; `false` maps to `KeyError` at the wrapper.  `keys_version`
/// is bumped: a reorder invalidates index-keyed caches (dict-view cursors,
/// module-dict global slots) and raises the mutation-during-iteration guard,
/// matching `OrderedDict` under CPython 3.14.
///
/// # Safety
/// `obj` must point to a valid dict backing (`W_DictObject` or module dict).
pub unsafe fn w_dict_move_to_end_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    last: bool,
) -> Result<bool, DictKeyError> {
    // `object_key_for_checked` hashes the key (`space.hash_w`, a collection
    // point), and the strategy switch below reallocates the backing store, so
    // pin and reload `obj`/`key` across the lock safepoint exactly as the other
    // checked dict primitives do — a stale `obj` or `ObjectKey.obj` after a
    // moving GC would corrupt the `IndexMap` access.
    lock_dict_refs!(_dict_guard, obj, key);
    if is_module_dict(obj) {
        if !w_module_dict_is_object_strategy(obj) {
            w_module_dict_switch_to_object_strategy(obj);
        }
        // The strategy switch allocates: hash the reloaded key.
        let object_key = object_key_for_checked(_dict_guard.root(1))?;
        let obj = _dict_guard.root(0);
        if let Some(result) = callback_free_dict_op!({
            let entries = w_module_dict_object_storage_mut(obj);
            match entries.index_of(&object_key) {
                Some(index) => {
                    // A key already at the wanted end does not move, and the
                    // move answers so; skip the version bump then, so a live
                    // iterator is not spuriously invalidated (as in
                    // w_dict_delitem_if_value_is_checked).
                    if entries.move_slot_to_end(index, last) {
                        w_dict_bump_keys_version(obj);
                    }
                    true
                }
                None => false,
            }
        }) {
            return result;
        }
        // A probing `__eq__`/`__hash__` escaped the callback-free ladder; the
        // reentrant scan (which services the module `object_storage`) locates
        // the entry outside the table borrow, then the reorder runs in place —
        // the object-strategy tail below, but on the module storage.
        let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
        let Some(index) = found else {
            return Ok(false);
        };
        let entries = w_module_dict_object_storage_mut(obj);
        if entries.move_slot_to_end(index, last) {
            w_dict_bump_keys_version(obj);
        }
        return Ok(true);
    }

    let dstrategy = (*(obj as *const W_DictObject)).dstrategy;
    let strategy = dstrategy.imp;
    // `EmptyDictStrategy.move_to_end` (dictmultiobject.py) raises
    // `KeyError` for the requested key without hashing it: an unhashable key
    // reports `KeyError`, not the object strategy's unhashable `TypeError`, and
    // no user `__hash__` runs.  Dispatch through the empty strategy before
    // switching, matching `w_dict_delitem_checked`'s empty-strategy arm (which,
    // being `del`, does hash).
    if strategy_is(dstrategy, StrategyKind::Empty)
        || strategy_is(dstrategy, StrategyKind::EmptyKwargs)
    {
        return Ok(false);
    }

    let kind = strategy.strategy_kind();

    // Typed stores reorder in place, preserving the strategy — the
    // `AbstractTypedStrategy.move_to_end` fast path (`dictmultiobject.py`).
    // Int and Bytes keep their native `RDict<i64>` / `RDict<BytesKey>`
    // storage; the lookup compares native keys, so it is callback-free and
    // needs no reentrant fallback.  A foreign-type key cannot be present in a
    // typed store, so it falls through to the object-strategy path below.
    if kind == StrategyKind::Int && crate::listobject::is_plain_int1(key) {
        let k = crate::listobject::plain_int_w(key);
        return Ok(typed_move_to_end::<i64, IntKeyHash>(
            obj as *mut W_DictObject,
            &k,
            last,
        ));
    }
    if kind == StrategyKind::Bytes && crate::is_exact_type(key, &crate::bytesobject::BYTES_TYPE) {
        let k = BytesKey(crate::w_bytes_block(key) as *mut crate::bytesobject::BytesBlock);
        return Ok(typed_move_to_end::<BytesKey, BytesKeyHash>(
            obj as *mut W_DictObject,
            &k,
            last,
        ));
    }

    // Object and Unicode share the `IndexMap<ObjectKey>` representation, so the
    // reorder below runs in place; only a genuinely foreign store (a typed
    // store reached with a non-native key, or a Kwargs/Map store) is converted;
    // the empty strategies are handled above.
    if kind != StrategyKind::Object && kind != StrategyKind::Unicode {
        strategy.switch_to_object_strategy(obj);
    }
    let object_key = object_key_for_checked(key)?;
    let obj = _dict_guard.root(0);
    if let Some(result) = callback_free_dict_op!({
        let dict = &mut *(obj as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        match entries.index_of(&object_key) {
            Some(index) => {
                if entries.move_slot_to_end(index, last) {
                    dict.keys_version = dict.keys_version.wrapping_add(1);
                }
                true
            }
            None => false,
        }
    }) {
        return result;
    }
    // A probing `__eq__` may have run and moved the container; the reentrant
    // scan re-pins it and returns the current index (no `value` to carry).
    let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
    let Some(index) = found else {
        return Ok(false);
    };
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
    if entries.move_slot_to_end(index, last) {
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
    Ok(true)
}

/// Internal helper: `ObjectDictStrategy::delitem` body for pyre's
/// W_DictObject — dstorage lookup + removal. Called only from the strategy trait impl to avoid
/// recursion through `w_dict_delitem`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_delitem_object_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    lock_dict_refs!(_dict_guard, obj, key);
    w_dict_delitem_object_strategy_checked(obj, key).unwrap_or(false)
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_delitem_object_strategy_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<bool, DictKeyError> {
    lock_dict_refs!(defer _dict_guard, obj, key);
    let object_key = object_key_for_checked(key)?;
    let obj = _dict_guard.root(0);
    // Single delitem probe, run callback-free so no user `__eq__` mutates the
    // dict while the `IndexMap` borrow is live.  A comparison the builtin
    // ladder cannot settle withholds the removal and re-runs the probe over
    // `scan_dict_key_reentrant`.
    if let Some(result) = callback_free_dict_op!({
        let dict = &mut *(obj as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        let index = entries.index_of(&object_key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            false
        } else {
            match index {
                Some(i) => {
                    entries.remove_slot(i);
                    dict.keys_version = dict.keys_version.wrapping_add(1);
                    true
                }
                None => false,
            }
        }
    }) {
        return result;
    }
    let (found, _, obj) = scan_dict_key_reentrant(obj, object_key)?;
    if let Some(i) = found {
        let dict = &mut *(obj as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        entries.remove_slot(i);
        dict.keys_version = dict.keys_version.wrapping_add(1);
        return Ok(true);
    }
    Ok(false)
}

/// Internal helper: `ModuleDictStrategy::delitem` body for pyre's
/// W_ModuleDictObject — `celldict.py`:
///   * post-`switch_to_object_strategy`: walk unified storage,
///     remove, and call mutated().
///   * str fast path on fresh ModuleDictStrategy → delitem_str.
///   * non-str + never-eq-str: fast-return false.
///   * non-str otherwise: switch_to_object_strategy then walk.
///     Called only from the strategy trait impl to avoid recursion
///     through `w_dict_delitem`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_delitem_inner(obj: PyObjectRef, key: PyObjectRef) -> bool {
    lock_dict_refs!(_module_guard, obj, key);
    if w_module_dict_is_object_strategy(obj) {
        let entries = w_module_dict_object_storage_mut(obj);
        if dict_entries_remove_object(entries, key) {
            w_dict_bump_keys_version(obj);
            return true;
        }
        return false;
    }
    if let Some(ks) = key_as_utf8(key) {
        return w_module_dict_delitem_str(obj, ks).is_some();
    }
    if _never_equal_to_string(key) {
        return false;
    }
    // The strategy switch allocates: read the key back.
    w_module_dict_switch_to_object_strategy(obj);
    let obj = _module_guard.root(0);
    let key = _module_guard.root(1);
    let entries = w_module_dict_object_storage_mut(obj);
    if dict_entries_remove_object(entries, key) {
        w_dict_bump_keys_version(obj);
        return true;
    }
    false
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_module_dict_delitem_inner_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<bool, DictKeyError> {
    lock_dict_refs!(_module_guard, obj, key);
    if w_module_dict_is_object_strategy(obj) {
        let object_key = object_key_for_checked(key)?;
        let obj = _module_guard.root(0);
        let entries = w_module_dict_object_storage_mut(obj);
        let removed = entries.remove(&object_key).is_some();
        if take_dict_key_error() {
            return Err(DictKeyError);
        }
        if removed {
            w_dict_bump_keys_version(obj);
            return Ok(true);
        }
        return Ok(false);
    }
    if let Some(ks) = key_as_utf8(key) {
        return Ok(w_module_dict_delitem_str(obj, ks).is_some());
    }
    if _never_equal_to_string(key) {
        return Ok(false);
    }
    w_module_dict_switch_to_object_strategy(obj);
    // The strategy switch allocates: hash the reloaded key.
    let object_key = object_key_for_checked(_module_guard.root(1))?;
    let obj = _module_guard.root(0);
    let entries = w_module_dict_object_storage_mut(obj);
    let removed = entries.remove(&object_key).is_some();
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    if removed {
        w_dict_bump_keys_version(obj);
        return Ok(true);
    }
    Ok(false)
}

/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.descr_clear`
/// — `w_dict.get_strategy().clear(w_dict)`.  Dispatches through the
/// polymorphic strategy slot.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_clear(obj: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj);
    w_dict_get_strategy(obj).clear(obj);
}

/// Internal helper: `ObjectDictStrategy::clear` body for pyre's
/// W_DictObject — truncates the dstorage Vec and resets the JIT
/// inline length cache.  Called only from the strategy trait impl
/// to avoid recursion through `w_dict_clear`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_clear_object_strategy(obj: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
    if !entries.is_empty() {
        dict.keys_version = dict.keys_version.wrapping_add(1);
        // Mirror `ll_dict_clear`'s `d.entries` realloc: an in-place
        // `IndexMap::clear` keeps the buffer, so a reentrant scan that cleared
        // the dict through `__eq__` would otherwise miss the restart when the
        // dict is refilled to the same capacity.
        dict.clear_gen = dict.clear_gen.wrapping_add(1);
    }
    entries.clear();
}

/// Internal helper: `ModuleDictStrategy::clear` body for pyre's
/// W_ModuleDictObject — branches on `is_object_strategy` and drains
/// whichever storage half is live.  Called only from the strategy
/// trait impl to avoid recursion through `w_dict_clear`.  The proxy
/// flush stays in the public `w_dict_clear` wrapper.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_clear_inner(obj: PyObjectRef) {
    lock_dict_refs!(_module_guard, obj);
    if w_module_dict_is_object_strategy(obj) {
        let entries = w_module_dict_object_storage_mut(obj);
        let changed = !entries.is_empty();
        entries.clear();
        if changed {
            w_dict_bump_keys_version(obj);
        }
    } else {
        let strategy = w_module_dict_module_strategy_mut(obj);
        let changed = !w_module_dict_module_storage(obj).is_empty();
        strategy.clear(obj);
        if changed {
            w_dict_bump_keys_version(obj);
        }
    }
}

/// `pypy/objspace/std/dictmultiobject.py W_DictMultiObject.length`
/// — `return self.get_strategy().length(self)`.  Dispatches through
/// the polymorphic strategy slot.
///
/// No root guard: `length` is `len(unerase(dstorage))` /
/// `d.num_live_items` (`rordereddict.py ll_dict_len`) and does not
/// collect, the same reason [`crate::listobject::w_list_len`] reads
/// `live_len` without `w_list_lock`.  The guard's `push_roots` is
/// `dont_look_inside`, so taking it residualized every traced `len(dict)`.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_len(obj: PyObjectRef) -> usize {
    // `W_DictMultiObject.length` → `self.get_strategy().length(self)`.
    // `length` does not hash or compare keys, so there is no
    // `DictOperationGuard` (`lock_dict_refs!`).
    w_dict_get_strategy(obj).length(obj)
}

/// Internal helper: `ObjectDictStrategy::length` body for pyre's
/// W_DictObject — `len(self.unerase(w_dict.dstorage))` per
/// `dictmultiobject.py length`. Called only from the strategy trait
/// impl to avoid recursion through `w_dict_len`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
#[inline]
pub unsafe fn w_dict_length_object_strategy(obj: PyObjectRef) -> usize {
    majit_ir::icf_identity!(concat!(
        module_path!(),
        "::",
        "w_dict_length_object_strategy"
    ));
    // `AbstractTypedStrategy.length`: `len(self.unerase(w_dict.dstorage))`.
    // `rordereddict.rtype_len` is `getfield num_live_items` (no custom eq/hash
    // on the typed tables; object-strategy `ll_dict_len` is the same field).
    // Two field reads, same shape as `w_list_len`: a getter call would
    // residualize and the walker would execute the 2-arg rustc vtable item.
    let dict = &*(obj as *const W_DictObject);
    (*(dict.dstorage as *const ObjectDictStorage)).num_live_items
}

/// Iterate over all (key, value) pairs without type assumptions.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_items(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    w_dict_get_strategy(obj).items(obj)
}

/// `dictmultiobject.py` `W_DictMultiObject.descr_popitem` delegates to
/// `W_DictMultiObject.popitem`, which dispatches through the live strategy.
///
/// Traced like the other strategy dispatchers: `w_dict_delitem` and
/// `w_dict_clear` mutate the same `IndexMap` and mapdict storages with no
/// barrier, and `dictmultiobject.py` carries no `@dont_look_inside` here.
///
/// # Safety
/// `obj` must be a valid PyObjectRef pointing at a `W_DictObject` or
/// `W_ModuleDictObject`.
pub unsafe fn w_dict_popitem(obj: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    w_dict_get_strategy(obj).popitem(obj)
}

/// `w_dict_get_strategy(obj).nth_item(obj, index)` — single-entry dispatch
/// for a slot the caller already holds.  A walk wants [`w_dict_next_item`],
/// which steps over the tombstones a delete leaves behind.
///
/// # Safety
/// `obj` must be a valid `W_DictObject` PyObjectRef.
pub unsafe fn w_dict_nth_item(
    obj: PyObjectRef,
    index: usize,
) -> Option<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    w_dict_get_strategy(obj).nth_item(obj, index)
}

/// The next `(slot, key, value)` at or after `from`, and the slot to resume
/// from — `_ll_dictnext` (`rordereddict.py`).
///
/// A dict view iterator holds this slot rather than a count of the pairs it
/// has seen, because a delete leaves a tombstone and the two stop agreeing.
///
/// # Safety
/// `obj` must be a valid `W_DictObject` PyObjectRef.
pub unsafe fn w_dict_next_item(
    obj: PyObjectRef,
    from: usize,
) -> Option<(usize, PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    let strategy = w_dict_get_strategy(obj);
    let slot = strategy.next_slot(obj, from)?;
    let (key, value) = strategy.nth_item(obj, slot)?;
    Some((slot, key, value))
}

/// The first live entry slot at or after `from`, for a walk that reads the
/// entry through some other accessor — `_ll_dictnext`'s scan alone
/// (`rordereddict.py`).
///
/// # Safety
/// `obj` must be a valid `W_DictObject` PyObjectRef.
pub unsafe fn w_dict_next_slot(obj: PyObjectRef, from: usize) -> Option<usize> {
    lock_dict_refs!(_dict_guard, obj);
    w_dict_get_strategy(obj).next_slot(obj, from)
}

/// [`w_dict_next_item`] descending, for `reversed(d)` —
/// `objectmodel.reversed_dict` (`dictmultiobject.py`).  `before` is
/// exclusive, so a fresh walk starts at `usize::MAX`.
///
/// # Safety
/// `obj` must be a valid `W_DictObject` PyObjectRef.
pub unsafe fn w_dict_prev_item(
    obj: PyObjectRef,
    before: usize,
) -> Option<(usize, PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    let strategy = w_dict_get_strategy(obj);
    let slot = strategy.prev_slot(obj, before)?;
    let (key, value) = strategy.nth_item(obj, slot)?;
    Some((slot, key, value))
}

/// Read only the value in slot `index`.
///
/// `dictmultiobject.py AbstractTypedStrategy.getitem` reads the typed storage
/// value without wrapping its key.
///
/// # Safety
/// `obj` must be a valid `W_DictObject` PyObjectRef.
pub unsafe fn w_dict_nth_value(obj: PyObjectRef, index: usize) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj);
    nullable_to_option(w_dict_get_strategy(obj).nth_value(obj, index))
}

/// Return the key in slot `index` together with the hash stored in an exact
/// dict's object-shaped table.
///
/// CPython 3.14's set fast path consumes exact-dict keys with their cached
/// hashes (`set_update_dict_lock_held`), which is observable for keys whose
/// `__hash__` has side effects.  PyPy exposes the corresponding cached-hash
/// iterator as `DictStrategy.getiteritems_with_hash`; pyre's Object and
/// Unicode strategies share the `IndexMap<ObjectKey, _>` representation, so
/// they can hand that same pair to the set implementation without hashing the
/// application-level key again.  Other typed strategies return `None` and use
/// their ordinary iterable path.
///
/// # Safety
/// `obj` must be a valid regular `W_DictObject` (not a module dict).
pub unsafe fn w_dict_nth_hashed_key(obj: PyObjectRef, index: usize) -> Option<ObjectKey> {
    lock_dict_refs!(_dict_guard, obj);
    match w_dict_get_strategy(obj).strategy_kind() {
        StrategyKind::Object | StrategyKind::Unicode => {
            let dict = &*(obj as *const W_DictObject);
            let entries = &*(dict.dstorage as *const ObjectDictStorage);
            entries.get_slot(index).map(|(key, _)| *key)
        }
        _ => None,
    }
}

/// `dictmultiobject.py W_DictMultiObject.descr_copy` —
/// `w_dict.copy()` delegates to `strategy.copy(w_dict)` so typed
/// strategies preserve their backing shape (`:1152
/// AbstractTypedStrategy.copy` → fresh W_DictObject with same strategy
/// + cloned typed dstorage).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_copy(obj: PyObjectRef) -> PyObjectRef {
    lock_dict_refs!(_dict_guard, obj);
    w_dict_get_strategy(obj).copy(obj)
}

/// `dictmultiobject.py AbstractTypedStrategy.copy` —
/// `W_DictObject(space, self, self.erase(dstorage.copy()))`.
///
/// Shared by `ObjectDictStrategy` and `UnicodeDictStrategy` the way
/// `w_dict_lookup_object_strategy` / `w_dict_items_object_strategy`
/// share the erased `ObjectDictStorage` backing. `strategy` is mixin
/// `self`: the dest keeps the source's typed strategy.
///
/// # Safety
/// `w_dict` must be a valid `W_DictObject` whose `dstorage` is
/// `ObjectDictStorage`.
pub unsafe fn w_dict_copy_object_strategy(
    w_dict: PyObjectRef,
    strategy: &'static DictStrategyRef,
) -> PyObjectRef {
    let dict = &*(w_dict as *const W_DictObject);
    let storage = &*(dict.dstorage as *const ObjectDictStorage);
    // `AbstractTypedStrategy.copy` erases a cloned `r_dict` (`dicttable`),
    // a nursery object. `type.__new__` copies the class namespace through
    // this path (`w_dict_copy`); a born-old clone kept discarded types
    // alive at every minor.
    let new_storage = storage.clone_to_young_box(object_dict_storage_gc_type_id());
    w_dict_new_with(strategy, new_storage as GCREF)
}

/// Internal helper: `IntDictStrategy::setitem` body.
///
/// look_inside_iff lives on `ll_dict_setitem_lookup_done`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictmultiobject::INT_DICT_STRATEGY`]; `key` must be a
/// plain `W_IntObject` (not bool).
pub unsafe fn w_dict_store_int_strategy(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj, key, value);
    dict_write_barrier(obj);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut IntDictStorage);
    let k = crate::listobject::plain_int_w(key);
    let fresh = entries.insert(k, value).is_none();
    if fresh {
        let dict = &mut *(_dict_guard.root(0) as *mut W_DictObject);
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
}

/// Internal helper: `IntDictStrategy::getitem` body —
/// `dictmultiobject.py self.unerase(w_dict.dstorage).get(self.unwrap(w_key), None)`.
/// Caller must have already verified `is_correct_type(w_key)`.
///
/// look_inside_iff lives on `ll_dict_lookup`.
///
/// # Safety
/// Same as [`w_dict_store_int_strategy`].
pub unsafe fn w_dict_lookup_int_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj, key);
    let entries = w_dict_int_storage(obj);
    let k = crate::listobject::plain_int_w(key);
    entries.get(&k).copied()
}

/// Return the insertion-order index selected by an int-strategy lookup.
///
/// # Safety
/// Same as [`w_dict_lookup_int_strategy`].
pub unsafe fn w_dict_index_of_int_strategy(obj: PyObjectRef, key: PyObjectRef) -> Option<usize> {
    lock_dict_refs!(_dict_guard, obj, key);
    w_dict_index_of_int_locked(obj, key)
}

/// [`w_dict_index_of_int_strategy`]'s body for a caller that already holds the
/// dict guard.
///
/// The probe unwraps a machine int and compares it — nothing here allocates, so
/// no collection can run between the caller's `lock_dict_refs!` and the answer,
/// and a second guard would republish the roots the caller is already holding.
///
/// # Safety
/// Same as [`w_dict_index_of_int_strategy`], and the caller holds a
/// `DictOperationGuard` over `obj` from which `obj` and `key` were read.
unsafe fn w_dict_index_of_int_locked(obj: PyObjectRef, key: PyObjectRef) -> Option<usize> {
    w_dict_int_storage(obj).index_of(&crate::listobject::plain_int_w(key))
}

/// Return the live value selected by an int-strategy lookup.
///
/// # Safety
/// Same as [`w_dict_lookup_int_strategy`].
pub unsafe fn w_dict_lookup_or_null_int_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj, key);
    let index = w_dict_index_of_int_locked(obj, key)?;
    w_dict_int_storage(obj)
        .get_slot(index)
        .map(|(_, &value)| value)
}

/// Return the insertion-order index selected by a Unicode-strategy lookup.
///
/// `dictmultiobject.py getitem_str` routes exact-str keys through the raw
/// string probe.  The trace-time caller declines when `w_str_get_value_opt`
/// or the raw hash hook is unavailable, so this helper has no allocating
/// object-key fallback.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictmultiobject::UNICODE_DICT_STRATEGY`].
pub unsafe fn w_dict_index_of_unicode_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<usize> {
    lock_dict_refs!(_dict_guard, obj, key);
    w_dict_index_of_unicode_locked(obj, key)
}

/// [`w_dict_index_of_unicode_strategy`]'s body for a caller that already holds
/// the dict guard.
///
/// The type test, the borrow of the key's bytes, the digest and the
/// `StrLookupKey` probe (a byte compare, never an app-level `__eq__`) all run
/// without allocating, so no collection can run between the caller's
/// `lock_dict_refs!` and the answer, and a second guard would republish the
/// roots the caller is already holding.
///
/// # Safety
/// Same as [`w_dict_index_of_unicode_strategy`], and the caller holds a
/// `DictOperationGuard` over `obj` from which `obj` and `key` were read.
unsafe fn w_dict_index_of_unicode_locked(obj: PyObjectRef, key: PyObjectRef) -> Option<usize> {
    if !crate::is_exact_type(key, &crate::STR_TYPE) {
        return None;
    }
    let key_str = crate::w_str_get_value_opt(key)?;
    let hash = w_str_memoized_hash(key, key_str)?;
    crate::dict_eq_hook::take_eq_error();
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const ObjectDictStorage);
    dict_entries_index_of_str_hashed(entries, hash, crate::unicodeobject::w_str_storage(key))
}

/// `rstr.py ll_strhash` — answer from the digest memoized on the string
/// and compute it only while the slot still reads zero, which upstream spells
/// as `jit.conditional_call_elidable(s.hash, ...)`.
///
/// The `HASH_STR_HOOK` trampoline takes `(ptr, len)` and so cannot reach the
/// memo, which left every str-keyed probe re-digesting the same bytes through
/// two thread-local reads and an indirect call.  The value is the one
/// `builtins::hash_value` publishes for the same string, so a memo written here
/// and a memo written there are interchangeable.
///
/// # Safety
/// `key` must be an exact `W_UnicodeObject` and `key_str` its own storage.
unsafe fn w_str_memoized_hash(key: PyObjectRef, _key_str: &str) -> Option<i64> {
    let memo = crate::w_str_get_hash(key);
    if memo != 0 {
        return Some(memo);
    }
    let hash = crate::dict_eq_hook::try_hash_str(crate::unicodeobject::w_str_storage(key))?;
    crate::w_str_set_hash(key, hash);
    Some(hash)
}

/// Return the live value selected by a Unicode-strategy lookup.
///
/// `dictmultiobject.py getitem_str` routes exact-str keys through the raw
/// string probe. The index lookup and value read share one dict lock so the
/// insertion-order index cannot be invalidated between the two operations —
/// this guard is that lock, and both steps run under it directly rather than
/// through helpers that take one each.
///
/// # Safety
/// Same as [`w_dict_index_of_unicode_strategy`].
pub unsafe fn w_dict_lookup_or_null_unicode_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj, key);
    let index = w_dict_index_of_unicode_locked(obj, key)?;
    // The Unicode and Object strategies share the `IndexMap<ObjectKey, _>`
    // representation, so the value comes out of the same table the index names.
    w_dict_object_storage(obj)
        .get_slot(index)
        .map(|(_, &value)| value)
}

/// Internal helper: `IntDictStrategy::delitem` body —
/// `dictmultiobject.py del self.unerase(w_dict.dstorage)[self.unwrap(w_key)]`.
/// Returns `true` if a key was removed.
///
/// look_inside_iff lives on `_ll_dict_del`.
///
/// # Safety
/// Same as [`w_dict_store_int_strategy`].
pub unsafe fn w_dict_delitem_int_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    lock_dict_refs!(_dict_guard, obj, key);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut IntDictStorage);
    let k = crate::listobject::plain_int_w(key);
    if entries.remove(&k).is_some() {
        dict.keys_version = dict.keys_version.wrapping_add(1);
        true
    } else {
        false
    }
}

/// Internal helper: `IntDictStrategy::length` body —
/// `dictmultiobject.py len(self.unerase(w_dict.dstorage))`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictmultiobject::INT_DICT_STRATEGY`].
#[inline]
pub unsafe fn w_dict_length_int_strategy(obj: PyObjectRef) -> usize {
    majit_ir::icf_identity!(concat!(module_path!(), "::", "w_dict_length_int_strategy"));
    // `AbstractTypedStrategy.length` / `IntDictStrategy`:
    // `len(self.unerase(w_dict.dstorage))` → `getfield num_live_items`.
    let dict = &*(obj as *const W_DictObject);
    (*(dict.dstorage as *const IntDictStorage)).num_live_items
}

/// Internal helper: `IntDictStrategy::items` body —
/// `dictmultiobject.py items` walks the typed storage and
/// wraps each i64 key via `self.wrap` (`:1342 wrap=newint`).
///
/// # Safety
/// Same as [`w_dict_length_int_strategy`].
pub unsafe fn w_dict_items_int_strategy(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    // The result is a Rust-heap Vec the collector never walks and one key is
    // wrapped per entry, so a pair pushed early would be pre-move by the next
    // wrap.  Accumulate the run in root slots — each value re-read from the
    // traced storage after its own iteration's wrap — and copy it out at the
    // end.
    let roots = crate::gc_roots::push_roots();
    let base = roots.base();
    let entries = w_dict_int_storage(obj);
    let len = entries.len();
    // A slot is not a position: a deleted entry leaves a tombstone the walk
    // steps over (`rordereddict.py:1373`).  `len` still counts the live pairs,
    // which is exactly how many root slots the loop publishes.
    let mut next = 0;
    while let Some(i) = entries.next_valid_slot(next) {
        let w_key = crate::w_int_new(*entries.get_slot(i).unwrap().0);
        roots.publish(&[w_key, *entries.get_slot(i).unwrap().1]);
        next = i + 1;
    }
    let mut items = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        items.push((roots.get(base + 2 * i), roots.get(base + 2 * i + 1)));
        i += 1;
    }
    items
}

/// Internal helper: single-entry accessor for `IntDictStrategy` — the
/// `(newint(key), value)` pair in slot `index`, read in O(1).  Backs the dict
/// view iterator's per-step fetch (`baseobjspace.rs` `next()`), replacing a
/// full `w_dict_items_int_strategy` re-materialisation per step.
///
/// # Safety
/// Same as [`w_dict_items_int_strategy`].
pub unsafe fn w_dict_nth_item_int_strategy(
    obj: PyObjectRef,
    index: usize,
) -> Option<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    let (k, v) = w_dict_int_storage(obj)
        .get_slot(index)
        .map(|(&k, &v)| (k, v))?;
    // Wrapping the key collects, and `v` is a bare local by then.
    let roots = crate::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(v);
    let w_key = crate::w_int_new(k);
    Some((w_key, roots.get(value_slot)))
}

/// Internal helper: `IntDictStrategy::clear` body —
/// `dictmultiobject.py self.unerase(w_dict.dstorage).clear()`.
///
/// # Safety
/// Same as [`w_dict_length_int_strategy`].
pub unsafe fn w_dict_clear_int_strategy(obj: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut IntDictStorage);
    if !entries.is_empty() {
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
    entries.clear();
}

/// `dictmultiobject.py AbstractTypedStrategy.switch_to_object_strategy`
/// for `IntDictStrategy` — rebuild the erased `{}` on the object shape, refill
/// it with `newint(key)` keys, then relabel the dict.
///
/// Residualised (`@dont_look_inside`, `rlib/jit.py`) even though upstream
/// carries no marker here: upstream can trace the body because every step is an
/// RPython dict primitive the JIT models (`ll_newdict`,
/// `ll_dict_setitem`'s `look_inside_iff(isvirtual)`), whereas the whole body is
/// `IndexMap` — an external crate with no lowering at all — so the last
/// modellable point is the call itself.  The `()` return crosses the residual
/// boundary the way `setobject::w_set_copy_storage_from` does.
///
/// # Safety
/// `w_dict` must be a valid `W_DictObject` on `INT_DICT_STRATEGY`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_switch_int_to_object_strategy(w_dict: PyObjectRef) {
    // Three separate hazards, one bracket.  The receiver moves, so the field
    // stores at the end must go through a reloaded word.  Each wrapped key is
    // minted inside the loop and is reachable from nothing the collector walks
    // until the new table is attached.  And the new table is a Rust-heap map:
    // a pair copied into it early would be pre-move by the next iteration's
    // mint.  So the run is accumulated in root slots and the table is filled
    // from those slots once every allocation is behind us.
    let roots = crate::gc_roots::push_roots();
    let dict_slot = roots.base();
    let w_dict = roots.pin_root(w_dict);
    // Borrow the old typed box (its field stays live, so it is traced
    // while the migration allocates keys); after the store the box is
    // unreachable and the sweep reclaims it.
    let old = &*((*(w_dict as *const W_DictObject)).dstorage as *const IntDictStorage);
    let len = old.len();
    let mut hashes = Vec::with_capacity(len);
    let pairs_base = dict_slot + 1;
    // A slot is not a position: a deleted entry leaves a tombstone the walk
    // steps over (`rordereddict.py:1373`).  `len` still counts the live pairs,
    // which is exactly how many the loop publishes.
    let mut next = 0;
    while let Some(i) = old.next_valid_slot(next) {
        let k = *old.get_slot(i).unwrap().0;
        let w_key = crate::w_int_new(k);
        let object_key = object_key_for(w_key);
        // Take the value only now: the old table is traced, so re-reading the
        // slot after this iteration's allocations yields the current word.
        let v = *old.get_slot(i).unwrap().1;
        hashes.push(object_key.hash);
        roots.publish(&[object_key.obj, v]);
        next = i + 1;
    }
    // The empty box is allocated before the fill so the last collection point
    // is behind the pairs being copied out of their slots.
    let new_storage = crate::gc_storage::gc_alloc_young_storage_box(
        object_dict_storage_with_capacity(len),
        object_dict_storage_gc_type_id(),
    );
    let new_map = &mut *new_storage;
    // `ll_getitem_fast` on the hash/pair arrays, not `Enumerate`.
    let hash_p = hashes.as_ptr();
    let mut i = 0usize;
    while i < hashes.len() {
        let obj = roots.get(pairs_base + 2 * i);
        let value = roots.get(pairs_base + 2 * i + 1);
        new_map.insert(
            ObjectKey {
                hash: *hash_p.add(i),
                obj,
            },
            value,
        );
        i += 1;
    }
    install_object_dict_storage(roots.get(dict_slot), new_storage);
}

/// Internal helper: `BytesDictStrategy::setitem` body —
/// `dictmultiobject.py` direct typed-storage write.  Caller
/// must have already verified `is_correct_type(w_key)`.
///
/// look_inside_iff lives on `ll_dict_setitem_lookup_done`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictmultiobject::BYTES_DICT_STRATEGY`]; `key` must be a
/// `W_BytesObject`.
pub unsafe fn w_dict_store_bytes_strategy(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj, key, value);
    dict_write_barrier(obj);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut BytesDictStorage);
    let k = BytesKey(crate::w_bytes_block(key) as *mut crate::bytesobject::BytesBlock);
    if entries.insert(k, value).is_none() {
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
}

/// Internal helper: `BytesDictStrategy::getitem` body —
/// `dictmultiobject.py`.  Caller must have already verified
/// `is_correct_type(w_key)`.
///
/// look_inside_iff lives on `ll_dict_lookup`.
///
/// # Safety
/// Same as [`w_dict_store_bytes_strategy`].
pub unsafe fn w_dict_lookup_bytes_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    lock_dict_refs!(_dict_guard, obj, key);
    let entries = w_dict_bytes_storage(obj);
    let k = BytesKey(crate::w_bytes_block(key) as *mut crate::bytesobject::BytesBlock);
    entries.get(&k).copied()
}

/// Internal helper: `BytesDictStrategy::delitem` body —
/// `dictmultiobject.py`.  Returns `true` if a key was removed.
///
/// # Safety
/// Same as [`w_dict_store_bytes_strategy`].
pub unsafe fn w_dict_delitem_bytes_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    lock_dict_refs!(_dict_guard, obj, key);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut BytesDictStorage);
    let k = BytesKey(crate::w_bytes_block(key) as *mut crate::bytesobject::BytesBlock);
    if entries.remove(&k).is_some() {
        dict.keys_version = dict.keys_version.wrapping_add(1);
        true
    } else {
        false
    }
}

/// Internal helper: `BytesDictStrategy::length` body.
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
#[inline]
pub unsafe fn w_dict_length_bytes_strategy(obj: PyObjectRef) -> usize {
    majit_ir::icf_identity!(concat!(
        module_path!(),
        "::",
        "w_dict_length_bytes_strategy"
    ));
    // `AbstractTypedStrategy.length` / `BytesDictStrategy`:
    // `len(self.unerase(w_dict.dstorage))` → `getfield num_live_items`.
    let dict = &*(obj as *const W_DictObject);
    (*(dict.dstorage as *const BytesDictStorage)).num_live_items
}

/// Internal helper: `BytesDictStrategy::items` body —
/// `dictmultiobject.py` with `wrap = newbytes` per
/// `:1234-1235`.
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
pub unsafe fn w_dict_items_bytes_strategy(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    // Accumulated in root slots for [`w_dict_items_int_strategy`]'s reason.
    let roots = crate::gc_roots::push_roots();
    let base = roots.base();
    let entries = w_dict_bytes_storage(obj);
    let len = entries.len();
    // A slot is not a position: a deleted entry leaves a tombstone the walk
    // steps over (`rordereddict.py:1373`).  `len` still counts the live pairs,
    // which is exactly how many root slots the loop publishes.
    let mut next = 0;
    while let Some(i) = entries.next_valid_slot(next) {
        let w_key = crate::w_bytes_from_block(entries.get_slot(i).unwrap().0.0);
        roots.publish(&[w_key, *entries.get_slot(i).unwrap().1]);
        next = i + 1;
    }
    let mut items = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        items.push((roots.get(base + 2 * i), roots.get(base + 2 * i + 1)));
        i += 1;
    }
    items
}

/// Internal helper: single-entry accessor for `BytesDictStrategy`.
///
/// # Safety
/// Same as [`w_dict_items_bytes_strategy`].
pub unsafe fn w_dict_nth_item_bytes_strategy(
    obj: PyObjectRef,
    index: usize,
) -> Option<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    let entries = w_dict_bytes_storage(obj);
    let (k, v) = entries.get_slot(index)?;
    let block = k.0;
    // `wrap` allocates, and `value` is a bare local by then.
    let roots = crate::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(*v);
    let w_key = crate::w_bytes_from_block(block);
    Some((w_key, roots.get(value_slot)))
}

/// Internal helper: `BytesDictStrategy::clear` body.
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
pub unsafe fn w_dict_clear_bytes_strategy(obj: PyObjectRef) {
    lock_dict_refs!(_dict_guard, obj);
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut BytesDictStorage);
    if !entries.is_empty() {
        dict.keys_version = dict.keys_version.wrapping_add(1);
    }
    entries.clear();
}

/// [`w_dict_switch_int_to_object_strategy`] for `BytesDictStrategy`, refilling
/// with `newbytes(key)` keys.  Residualised for the same reason.
///
/// # Safety
/// `w_dict` must be a valid `W_DictObject` on `BYTES_DICT_STRATEGY`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_switch_bytes_to_object_strategy(w_dict: PyObjectRef) {
    // Rooted exactly as [`w_dict_switch_int_to_object_strategy`] is, for the
    // same three reasons.
    let roots = crate::gc_roots::push_roots();
    let dict_slot = roots.base();
    let w_dict = roots.pin_root(w_dict);
    let old = &*((*(w_dict as *const W_DictObject)).dstorage as *const BytesDictStorage);
    let len = old.len();
    let mut hashes = Vec::with_capacity(len);
    let pairs_base = dict_slot + 1;
    // A slot is not a position, as in [`w_dict_switch_int_to_object_strategy`].
    let mut next = 0;
    while let Some(i) = old.next_valid_slot(next) {
        let w_key = crate::w_bytes_from_block(old.get_slot(i).unwrap().0.0);
        let object_key = object_key_for(w_key);
        let v = *old.get_slot(i).unwrap().1;
        hashes.push(object_key.hash);
        roots.publish(&[object_key.obj, v]);
        next = i + 1;
    }
    let new_storage = crate::gc_storage::gc_alloc_young_storage_box(
        object_dict_storage_with_capacity(len),
        object_dict_storage_gc_type_id(),
    );
    let new_map = &mut *new_storage;
    let hash_p = hashes.as_ptr();
    let mut i = 0usize;
    while i < hashes.len() {
        let obj = roots.get(pairs_base + 2 * i);
        let value = roots.get(pairs_base + 2 * i + 1);
        new_map.insert(
            ObjectKey {
                hash: *hash_p.add(i),
                obj,
            },
            value,
        );
        i += 1;
    }
    install_object_dict_storage(roots.get(dict_slot), new_storage);
}

/// Internal helper: `ObjectDictStrategy::items` body for pyre's
/// W_DictObject — clones the authoritative storage. Called only from the
/// strategy trait impl to avoid recursion through `w_dict_items`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_items_object_strategy(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const ObjectDictStorage);
    let mut items = Vec::new();
    let mut slot = 0;
    while let Some(i) = entries.next_valid_slot(slot) {
        let (k, v) = entries.get_slot(i).unwrap();
        items.push((k.obj, *v));
        slot = i + 1;
    }
    items
}

/// Internal helper: single-entry accessor for `ObjectDictStrategy` /
/// `UnicodeDictStrategy`, reading the slot directly.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_nth_item_object_strategy(
    obj: PyObjectRef,
    index: usize,
) -> Option<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_dict_guard, obj);
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const ObjectDictStorage);
    entries.get_slot(index).map(|(k, &v)| (k.obj, v))
}

/// `rordereddict.py ll_call_lookup_function` — the `dict.lookup` oopspec
/// residual, answering the entry index for `w_key` (or -1 for a miss) under a
/// hash the caller already computed.
///
/// `flag` is the `FLAG_LOOKUP` / `FLAG_STORE` / `FLAG_DELETE` selector
/// `ll_dict_lookup` takes (`rordereddict.py`), and **only
/// `FLAG_LOOKUP` is implemented here**.  The argument exists because the
/// optimizer reads it off `arg(4)` (`heap.py`) and requires it to be a
/// constant; the recorder only ever emits 0.
///
/// The other two are not "the same answer" — they are not expressible.  On a
/// miss `FLAG_STORE` answers the free slot the caller is to write into, and
/// `FLAG_DELETE` walks to the tombstone, both of which read the free-slot
/// bookkeeping an `IndexMap` does not keep.  This helper answers -1 for every
/// miss, so a store or delete lookup would be wrong rather than merely
/// conservative.
///
/// Only a Unicode-strategy dict reaches here, so every stored key is an exact
/// `str` and the probe's comparisons are WTF-8 byte equality.  The
/// callback-free probe bracket enforces that: a pair that would need user code
/// breaks the probe and reports a miss, which side-exits the compiled trace to
/// the generic residual rather than running `__eq__` inside a call the recorder
/// promised cannot raise.
///
/// The object parameters are spelled `*mut PyObject` rather than
/// [`PyObjectRef`]: `dont_look_inside` decides syntactically whether it can
/// emit the `__majit_call_target_*` word-ABI trampoline, and an alias path is
/// not a raw-pointer token, so the alias spelling silently yields no
/// trampoline — which is what the wasm backend's `call_indirect` needs.
///
/// The table read holds the dict's own lock, like every other reader of
/// `dstorage`, so a concurrent `setitem` cannot resize the `IndexMap` under the
/// scan.  Holding it across the probe cannot deadlock: `dict_keys_equal` runs no
/// user code while the bracket is active — a pair the builtin ladder cannot
/// decide breaks the probe instead — so nothing under the lock can re-enter a
/// dict operation.
///
/// # Safety
/// `w_dict` must be a live `W_DictObject` on a strategy whose storage is an
/// `ObjectDictStorage`, and `hash` the digest `object_key_for` produces for
/// `w_key`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_unicode_lookup_index(
    w_dict: *mut PyObject,
    w_key: *mut PyObject,
    hash: i64,
    flag: i64,
) -> i64 {
    debug_assert_eq!(flag, 0, "only FLAG_LOOKUP is implemented");
    lock_dict_refs!(_dict_guard, w_dict, w_key);
    let dict = &*(w_dict as *const W_DictObject);
    let entries = &*(dict.dstorage as *const ObjectDictStorage);
    crate::dict_eq_hook::take_eq_error();
    crate::dict_eq_hook::begin_callback_free_probe();
    let found = entries.index_of(&ObjectKey { hash, obj: w_key });
    let probe_needed_user_code = crate::dict_eq_hook::end_callback_free_probe();
    match found {
        Some(i) if !probe_needed_user_code => i as i64,
        _ => -1,
    }
}

/// `rstr.ll_strhash` — the digest [`object_key_for`] caches for an exact `str`
/// key, isolated so a recorder that has already guarded the key's exact type
/// can emit it as one elidable call and let the optimizer hoist it out of a
/// loop.  Memoized on the string object, content-derived and identity-stable,
/// which is what `@jit.elidable` asserts.
///
/// `w_key` is spelled `*mut PyObject` for [`w_dict_unicode_lookup_index`]'s
/// reason.
///
/// # Safety
/// `w_key` must be a live exact `str`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_unicode_key_hash(w_key: *mut PyObject) -> i64 {
    crate::dict_eq_hook::try_hash_w(w_key)
        .unwrap_or_else(|| crate::dict_eq_hook::missing_hash_hook())
}

/// `rordereddict.py ll_dict_getitem` — the value half of the lookup, read
/// at the entry index [`w_dict_unicode_lookup_index`] settled on, and answering
/// null for anything the index no longer describes.
///
/// The index is re-validated against the key that produced it, because the two
/// halves are two residuals and the lock is not held across the gap.  Upstream
/// runs them as one translated sequence under the GIL and can index straight
/// in; here another thread's `del` compacts the table (`IndexMap::shift_remove`
/// closes the hole), so an index that was in range and correct at lookup time
/// can name a *different* live key by the time it is read.  A bounds check does
/// not see that — the index is still in range — and the answer would be a wrong
/// value rather than a crash.  Re-validating also makes the optimizer's own CSE
/// safe: `_optimize_call_dict_lookup` may delete a repeat of the same
/// `(dict, key)` probe, so the index arriving here can have been produced by an
/// earlier iteration.
///
/// The strategy is re-checked under the lock for the same reason: the trace
/// guarded it before the call, and only the Unicode and Object strategies share
/// this `IndexMap<ObjectKey, _>` storage shape.
///
/// A null answer fails the caller's `GuardNonnull` and side-exits to the generic
/// residual, which redoes the whole subscript.
///
/// # Safety
/// `obj` must be a live regular `W_DictObject` (not a module dict), and `hash`
/// the digest `object_key_for` produces for `w_key`.
pub unsafe fn w_dict_unicode_value_at_checked(
    obj: PyObjectRef,
    index: usize,
    w_key: PyObjectRef,
    hash: i64,
) -> PyObjectRef {
    lock_dict_refs!(_dict_guard, obj, w_key);
    if !matches!(
        w_dict_get_strategy(obj).strategy_kind(),
        StrategyKind::Unicode | StrategyKind::Object
    ) {
        return std::ptr::null_mut();
    }
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const ObjectDictStorage);
    let Some((entry_key, &value)) = entries.get_slot(index) else {
        return std::ptr::null_mut();
    };
    if entry_key.hash != hash {
        return std::ptr::null_mut();
    }
    crate::dict_eq_hook::take_eq_error();
    crate::dict_eq_hook::begin_callback_free_probe();
    let same_key = dict_keys_equal(entry_key.obj, w_key);
    let probe_needed_user_code = crate::dict_eq_hook::end_callback_free_probe();
    if same_key && !probe_needed_user_code {
        value
    } else {
        std::ptr::null_mut()
    }
}

/// Overwrite the value already stored at `index` when that slot still holds
/// `w_key`.
///
/// `typedef.py` `TypeCache.build` assigns the copied `GetSetProperty` back
/// onto the key the raw dict already contains. The hit arm of
/// Internal helper: `ModuleDictStrategy::items` body for pyre's
/// W_ModuleDictObject — branches on `is_object_strategy` and emits
/// from whichever storage half is live.  Called only from the
/// strategy trait impl to avoid recursion through `w_dict_items`.
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_items_inner(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_module_guard, obj);
    if let Some(entries) = w_module_dict_object_storage(obj) {
        let mut items = Vec::new();
        let mut slot = 0;
        while let Some(i) = entries.next_valid_slot(slot) {
            let (k, v) = entries.get_slot(i).unwrap();
            items.push((k.obj, *v));
            slot = i + 1;
        }
        items
    } else {
        let strategy = &*w_module_dict_get_strategy(obj);
        let storage = w_module_dict_module_storage(obj);
        // Wrapping a name the intern table has not seen allocates, so a value
        // zipped in before a later wrap would be pre-move in the Rust-heap
        // result.  Wrap every name into a root slot first, then walk the
        // values — which the cell storage keeps traced — with no allocation
        // left to run.
        let roots = crate::gc_roots::push_roots();
        let keys_base = roots.base();
        let mut n = 0usize;
        let mut slot = 0usize;
        while let Some(i) = strategy.next_entry_slot(storage, slot) {
            let name = strategy.nth_key(storage, i).unwrap();
            let _ = roots.pin_root(crate::celldict::_wrapkey(name));
            n += 1;
            slot = i + 1;
        }
        let mut items = Vec::with_capacity(n);
        slot = 0;
        let mut out = 0usize;
        while let Some(i) = strategy.next_entry_slot(storage, slot) {
            let value = strategy.nth_unwrapped_value(storage, i).unwrap();
            items.push((roots.get(keys_base + out), value));
            out += 1;
            slot = i + 1;
        }
        items
    }
}

/// [`w_module_dict_items_inner`] for one iteration position.
///
/// The dict view's integer cursor asks for a single entry, and both storage
/// halves are `IndexMap`s, so this reads that entry rather than building the
/// whole list. Only the one name is wrapped.
///
/// `celldict.py getiterkeys`/`getitervalues` are lazy iterators
/// upstream; the cursor stands in for them here because the GC-object layout
/// cannot hold a live iterator ([`DictStrategy::nth_item`]).  Serving that
/// cursor from the default `nth_item` — `self.items(w_dict).into_iter()
/// .nth(index)` — makes the stand-in quadratic: [`w_module_dict_items_inner`]
/// wraps EVERY name, so an `n`-entry dict costs `n` wraps per step: measured, a
/// 350-name module dict walked ten times grew peak RSS by 156.8 MB where
/// CPython grew none, and removing it took 10.3 MB off interpreter startup —
/// `dir(module)` inside `importlib._bootstrap` pays this walk.  The wrap
/// itself is [`crate::celldict::_wrapkey`], which interns rather than minting;
/// what it costs on a repeat is a table hit, and the per-step count is still
/// what this cursor exists to hold at one.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_nth_item_inner(
    obj: PyObjectRef,
    index: usize,
) -> Option<(PyObjectRef, PyObjectRef)> {
    lock_dict_refs!(_module_guard, obj);
    if let Some(entries) = w_module_dict_object_storage(obj) {
        return entries.get_slot(index).map(|(k, &v)| (k.obj, v));
    }
    let strategy = &*w_module_dict_get_strategy(obj);
    let key = strategy.nth_key(w_module_dict_module_storage(obj), index)?;
    // Wrapping the name allocates, so root the wrapper and read the value
    // back afterwards — the cell storage keeps it traced.
    let roots = crate::gc_roots::push_roots();
    let key_slot = roots.base();
    let _ = roots.pin_root(crate::celldict::_wrapkey(key));
    let value = strategy.nth_unwrapped_value(w_module_dict_module_storage(obj), index)?;
    Some((roots.get(key_slot), value))
}

/// [`w_module_dict_nth_item_inner`]'s value half, which wraps no name —
/// `dictmultiobject.py AbstractTypedStrategy.getitem` reads the typed storage
/// value without
/// wrapping its key, and here that skips the whole allocation.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_nth_value_inner(obj: PyObjectRef, index: usize) -> Option<PyObjectRef> {
    lock_dict_refs!(_module_guard, obj);
    if let Some(entries) = w_module_dict_object_storage(obj) {
        return entries.get_slot(index).map(|(_, &v)| v);
    }
    let strategy = &*w_module_dict_get_strategy(obj);
    strategy.nth_unwrapped_value(w_module_dict_module_storage(obj), index)
}

/// `celldict.py values` — the cells, with no name wrapped.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_values_inner(obj: PyObjectRef) -> Vec<PyObjectRef> {
    lock_dict_refs!(_module_guard, obj);
    if let Some(entries) = w_module_dict_object_storage(obj) {
        return entries.values().copied().collect();
    }
    let strategy = &*w_module_dict_get_strategy(obj);
    strategy
        .getitervalues(w_module_dict_module_storage(obj))
        .collect()
}

/// Iterate over (key_str, value) pairs. Keys must be str objects.
///
/// Pyre-only convenience wrapper around `w_dict_items` that drops
/// non-str entries and unwraps the str keys.  Dispatches through
/// the strategy slot via `w_dict_items`, so W_DictObject and
/// W_ModuleDictObject both use their authoritative storage.
///
/// Keys that carry a lone surrogate (not valid UTF-8) are skipped:
/// the remaining `&str`-keyed consumers (dict_storage_store, module
/// `__dir__`, builtins-module iteration) cannot yet represent a
/// surrogate key, so skipping them here avoids a UTF-8 view of WTF-8
/// storage.  The keyword-argument ABI no longer uses this helper — it
/// threads the byte-ish key through [`w_dict_str_entries_wtf8`].
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_str_entries(obj: PyObjectRef) -> Vec<(String, PyObjectRef)> {
    let source = w_dict_items(obj);
    let mut entries = Vec::with_capacity(source.len());
    for (key, value) in source {
        if crate::is_str(key)
            && let Some(key) = crate::w_str_get_value_opt(key)
        {
            entries.push((key.to_string(), value));
        }
    }
    entries
}

/// Iterate over (key_wtf8, value) pairs, preserving lone-surrogate keys.
///
/// The surrogate-preserving counterpart of [`w_dict_str_entries`], used
/// by the keyword-argument ABI (`call_with_kwargs`, `parse_builtin_signature`).
/// `Arguments` keeps keyword names as byte-ish `str` (`keywords: [str]`,
/// `argument.py`), so a `**{'\udc80': v}` key survives as a `Wtf8Buf`
/// rather than being dropped.
///
/// Non-str keys are dropped here, which is correct for the name-enumeration
/// callers (`dir()`, dict merge).  It is NOT, however, the `**kwargs`-unpack
/// contract: `argument.py` `_do_combine_starstarargs_wrapped` RAISES
/// `TypeError("keywords must be strings, not '%T'")` on a non-str key rather
/// than skipping it, so a caller on that path (`CALL_FUNCTION_EX`) must enforce
/// the TypeError itself — this helper does not.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_str_entries_wtf8(
    obj: PyObjectRef,
) -> Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> {
    let source = w_dict_items(obj);
    let mut entries = Vec::with_capacity(source.len());
    for (key, value) in source {
        if crate::is_str(key) {
            entries.push((crate::w_str_get_wtf8(key).to_owned(), value));
        }
    }
    entries
}

// ____________________________________________________________
// Iteration and views (`dictmultiobject.py W_BaseDictMultiIterObject+`, `:1603+`).

/// `dictmultiobject.py` — three sibling view classes share the
/// `W_DictViewObject` base. Pyre folds them into one struct + tag
/// because the body is otherwise identical (only the iteration / repr
/// shape differs).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictViewKind {
    Keys = 0,
    Values = 1,
    Items = 2,
}

// `W_BaseDictMultiIterObject`
//
// `pypy/objspace/std/dictmultiobject.py` `W_BaseDictMultiIterObject`
// (and the concrete kind subclasses `W_DictMultiIterKeysObject` /
// `W_DictMultiIterValuesObject` / `W_DictMultiIterItemsObject`) line-by-line
// port. PyPy's iterator captures the source dict + a strategy-specific
// iterator into the entries; mutation tracking happens via the
// `len(w_dict) != self.startlen` check inside `next_w` per
// `:1727-1733`, which raises `RuntimeError("dictionary changed size
// during iteration")`.
//
// Pyre's flat entries Vec lets us index directly; the parity-correct
// detection compares `dict.version` against the version captured at
// iter() time, matching PyPy's `dictversion` check in
// `W_BaseDictMultiIterObject`.

pub static DICT_KEYITERATOR_TYPE: PyType = crate::pyobject::new_pytype("dict_keyiterator");
pub static DICT_VALUEITERATOR_TYPE: PyType = crate::pyobject::new_pytype("dict_valueiterator");
pub static DICT_ITEMITERATOR_TYPE: PyType = crate::pyobject::new_pytype("dict_itemiterator");
pub static DICT_REVERSEKEYITERATOR_TYPE: PyType =
    crate::pyobject::new_pytype("dict_reversekeyiterator");
pub static DICT_REVERSEVALUEITERATOR_TYPE: PyType =
    crate::pyobject::new_pytype("dict_reversevalueiterator");
pub static DICT_REVERSEITEMITERATOR_TYPE: PyType =
    crate::pyobject::new_pytype("dict_reverseitemiterator");

/// `dictmultiobject.py _new_next` — captures both the source
/// dict's `len` and the active strategy at iter() time; `next()`
/// performs two parity-mandated checks per `:821-822` and `:829`:
///   * `self.len != self.w_dict.length()` -> RuntimeError
///     ("dictionary changed size during iteration"). Re-assigning
///     an existing key inside a loop (`for k in d: d[k] = new`) is
///     permitted because `length()` is unchanged.
///   * `self.strategy is not self.w_dict.get_strategy()` -> strategy
///     transition (e.g. `switch_to_object_strategy` mid-iteration).
///     The (key, value) pair in `result` may be out-of-date; PyPy
///     re-looks up the key in the dict and raises "dictionary
///     changed during iteration" if the key was removed (`:837-841`).
///     Keys/values iterators accept the stale result.
#[repr(C)]
pub struct W_BaseDictMultiIterObject {
    pub ob_header: PyObject,
    /// `:1707 self.w_dict` — back-pointer to the live source dict.
    pub w_dict: PyObjectRef,
    /// `:809 self.len = w_dict.length()` parity — captures len at
    /// iter() time; iterator compares against `w_dict.length()` on
    /// each `next()` step.
    pub startlen: usize,
    /// Key-set state of the live strategy iterator at construction time.
    /// This is the explicit counterpart of the mutation state retained by
    /// PyPy's underlying RPython dict iterator.
    pub start_keys_version: usize,
    /// The number of entries already consumed — what `__length_hint__` and
    /// `__reduce__` report, and the bound `startlen` is compared against.
    pub index: usize,
    /// The cursor into the source dict's entry slots.
    ///
    /// Separate from [`Self::index`] because the two stop agreeing the moment
    /// a delete leaves a tombstone: `index` counts pairs, this names a slot.
    /// It stands in for the live RPython dict iterator PyPy's
    /// `BaseIteratorImplementation` owns (`dictmultiobject.py`), which
    /// a GC object layout here cannot hold.  A reverse walk starts one past
    /// the end, at `usize::MAX`.
    pub slot: usize,
    /// `DictViewKind` repurposed for the three concrete iterator
    /// kinds (`W_DictMultiIterKeysObject` / `ValuesObject` /
    /// `ItemsObject`).
    pub kind: DictViewKind,
    /// Whether the strategy iterator walks insertion order backwards.
    /// PyPy represents this in the concrete `iterreversed()` implementation;
    /// the cursor position remains the number of already-consumed entries.
    pub reverse: bool,
    /// `:807 self.strategy = strategy` — strategy identity at iter()
    /// time, stored as the strategy pointer cast to `usize` for
    /// identity comparison (PyPy's `self.strategy is
    /// self.w_dict.get_strategy()` at `:829`). Strategy singletons
    /// are `'static` so the cast is stable across the iterator's
    /// lifetime.
    pub start_strategy_id: usize,
}

pub const DICT_VIEW_ITER_W_DICT_OFFSET: usize =
    std::mem::offset_of!(W_BaseDictMultiIterObject, w_dict);

/// Runtime-assigned GC type id shared by the three concrete iterator types.
/// PyPy's `W_BaseDictMultiIterObject` is a GC object whose `w_dict` edge is
/// traced.  Keep this auto-numbered because the fixed registration chain has
/// grown past the historical slot 42.
pub static W_DICT_VIEW_ITERATOR_GC_TYPE_ID: crate::lltype::TypeIdCell =
    crate::lltype::TypeIdCell::auto();
pub const W_DICT_VIEW_ITERATOR_OBJECT_SIZE: usize =
    std::mem::size_of::<W_BaseDictMultiIterObject>();

pub const W_DICT_VIEW_ITERATOR_GC_PTR_OFFSETS: [usize; 1] = [DICT_VIEW_ITER_W_DICT_OFFSET];

/// Reads the runtime-assigned dict-view iterator type id (stamped once at
/// init by the JIT driver's GC registration); the value is not a build-time
/// constant, so the JIT residualises the read instead of tracing into it
/// (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub fn dict_view_iterator_gc_type_id() -> u32 {
    W_DICT_VIEW_ITERATOR_GC_TYPE_ID.get()
}

impl crate::lltype::GcType for W_BaseDictMultiIterObject {
    fn type_id() -> u32 {
        dict_view_iterator_gc_type_id()
    }
    const SIZE: usize = W_DICT_VIEW_ITERATOR_OBJECT_SIZE;
}

/// Pick the Python-visible iterator type for a given view kind so
/// `type(iter(d.keys())) is dict_keyiterator` (etc.).
pub fn dict_view_iterator_type_for_kind(kind: DictViewKind, reverse: bool) -> &'static PyType {
    match (kind, reverse) {
        (DictViewKind::Keys, false) => &DICT_KEYITERATOR_TYPE,
        (DictViewKind::Values, false) => &DICT_VALUEITERATOR_TYPE,
        (DictViewKind::Items, false) => &DICT_ITEMITERATOR_TYPE,
        (DictViewKind::Keys, true) => &DICT_REVERSEKEYITERATOR_TYPE,
        (DictViewKind::Values, true) => &DICT_REVERSEVALUEITERATOR_TYPE,
        (DictViewKind::Items, true) => &DICT_REVERSEITEMITERATOR_TYPE,
    }
}

/// Allocate a fresh dict iterator capturing `w_dict`'s current length
/// and active strategy. Mirrors `dictmultiobject.py
/// W_BaseIteratorImplementation.__init__`:
///
/// ```python
/// self.strategy = strategy
/// self.len = w_dict.length()
/// ```
pub fn w_dict_view_iterator_new(w_dict: PyObjectRef, kind: DictViewKind) -> PyObjectRef {
    w_dict_view_iterator_new_direction(w_dict, kind, false)
}

/// Allocate the reversed concrete iterator corresponding to `kind`.
pub fn w_dict_view_reverse_iterator_new(w_dict: PyObjectRef, kind: DictViewKind) -> PyObjectRef {
    w_dict_view_iterator_new_direction(w_dict, kind, true)
}

/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `w_dict_new` twin: the body picks the iterator's type word at runtime
/// (`dict_view_iterator_type_for_kind(kind, reverse)`), and a cluster whose
/// type word is not a constant address does not lower — the word rides on the
/// allocation or not at all, since a `setfield_gc` whose descr `is_typeptr()`
/// is removed downstream. The primary allocation is hand-rolled through
/// `try_gc_alloc_nursery_raw` besides, which is not one of the
/// `lltype::malloc*` spellings `fuse_boxing_alloc` recognises. Residualise the
/// whole constructor — the JIT models it by signature as a plain
/// `PyObjectRef` GCREF and emits a residual call.
#[majit_macros::dont_look_inside]
fn w_dict_view_iterator_new_direction(
    w_dict: PyObjectRef,
    kind: DictViewKind,
    reverse: bool,
) -> PyObjectRef {
    let tp = dict_view_iterator_type_for_kind(kind, reverse);
    let _roots = crate::gc_roots::push_roots();
    let dict_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_dict);
    let startlen = unsafe { w_dict_len(crate::gc_roots::shadow_stack_get(dict_slot)) };
    let start_keys_version =
        unsafe { w_dict_keys_version(crate::gc_roots::shadow_stack_get(dict_slot)) };
    let start_strategy_id =
        unsafe { w_dict_strategy_id(crate::gc_roots::shadow_stack_get(dict_slot)) };
    let value = W_BaseDictMultiIterObject {
        ob_header: PyObject {
            ob_type: tp as *const PyType,
            w_class: get_instantiate(tp),
        },
        w_dict: crate::gc_roots::shadow_stack_get(dict_slot),
        startlen,
        start_keys_version,
        index: 0,
        slot: if reverse { usize::MAX } else { 0 },
        kind,
        reverse,
        start_strategy_id,
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        dict_view_iterator_gc_type_id(),
        W_DICT_VIEW_ITERATOR_OBJECT_SIZE,
    );
    if raw.is_null() {
        crate::lltype::malloc_typed(value) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut W_BaseDictMultiIterObject, value) };
        // The shell is a livevar across the barrier: that call is a `gc_op`,
        // which leaves RUNNING before taking `gc_mutex` (`gc_sync.rs`) and
        // roots only its own copy, so a foreign collector can move it there.
        // Publish it first and return the address the slot carries afterwards.
        let shell_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(raw as PyObjectRef);
        crate::gc_hook::try_gc_write_barrier(raw);
        crate::gc_roots::shadow_stack_get(shell_slot)
    }
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_view_iterator(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &DICT_KEYITERATOR_TYPE)
            || py_type_check(obj, &DICT_VALUEITERATOR_TYPE)
            || py_type_check(obj, &DICT_ITEMITERATOR_TYPE)
            || py_type_check(obj, &DICT_REVERSEKEYITERATOR_TYPE)
            || py_type_check(obj, &DICT_REVERSEVALUEITERATOR_TYPE)
            || py_type_check(obj, &DICT_REVERSEITEMITERATOR_TYPE)
    }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_dict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).w_dict }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_kind(obj: PyObjectRef) -> DictViewKind {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).kind }
}

/// Direction selected by the concrete iterator class.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_view_iterator_get_reverse(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).reverse }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_startlen(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).startlen }
}

/// Key-set mutation state captured with the source strategy iterator.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_dict_view_iterator_get_start_keys_version(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).start_keys_version }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_index(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).index }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_set_index(obj: PyObjectRef, value: usize) {
    unsafe {
        (*(obj as *mut W_BaseDictMultiIterObject)).index = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_slot(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).slot }
}

/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_set_slot(obj: PyObjectRef, value: usize) {
    unsafe {
        (*(obj as *mut W_BaseDictMultiIterObject)).slot = value;
    }
}

/// `:807 self.strategy = strategy` — strategy id captured at iter()
/// creation. Consumers compare it against the dict's current
/// `w_dict_strategy_id` to detect strategy transitions
/// (`dictmultiobject.py:829`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseDictMultiIterObject`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_start_strategy_id(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BaseDictMultiIterObject)).start_strategy_id }
}

// `W_DictViewObject`

pub static DICT_KEYS_TYPE: PyType = crate::pyobject::new_pytype("dict_keys");
pub static DICT_VALUES_TYPE: PyType = crate::pyobject::new_pytype("dict_values");
pub static DICT_ITEMS_TYPE: PyType = crate::pyobject::new_pytype("dict_items");

/// Layout: `[ob_header | kind: DictViewKind | w_dict: PyObjectRef]`.
///
/// `w_dict` is the live `W_DictObject` the view is attached to; PyPy's
/// `W_DictViewObject.w_dict` plays the same role. Mutations on the
/// source dict are visible through the view because every reader
/// (iter / len / contains) goes through `w_dict` rather than caching
/// a snapshot.
#[repr(C)]
pub struct W_DictViewObject {
    pub ob_header: PyObject,
    pub kind: DictViewKind,
    pub w_dict: PyObjectRef,
}

pub const DICT_VIEW_KIND_OFFSET: usize = std::mem::offset_of!(W_DictViewObject, kind);
pub const DICT_VIEW_W_DICT_OFFSET: usize = std::mem::offset_of!(W_DictViewObject, w_dict);

/// GC type id assigned to `W_DictViewObject` at JitDriver init time.
/// 32 is taken by `W_GENERATOR_GC_TYPE_ID`; the next free slot is 39
/// (one past `W_DICT_PROXY_GC_TYPE_ID = 38`).
pub const W_DICT_VIEW_GC_TYPE_ID: u32 = 39;

pub const W_DICT_VIEW_OBJECT_SIZE: usize = std::mem::size_of::<W_DictViewObject>();

/// Single inline `PyObjectRef`-shaped field — the back-pointer to the
/// source dict.
pub const W_DICT_VIEW_GC_PTR_OFFSETS: [usize; 1] = [DICT_VIEW_W_DICT_OFFSET];

impl crate::lltype::GcType for W_DictViewObject {
    fn type_id() -> u32 {
        W_DICT_VIEW_GC_TYPE_ID
    }
    const SIZE: usize = W_DICT_VIEW_OBJECT_SIZE;
}

/// Pick the Python-visible type for a given view kind. Mirrors
/// PyPy's three distinct W_TypeObject identities so
/// `type(d.keys()) is dict_keys`, `type(d.values()) is dict_values`,
/// `type(d.items()) is dict_items` all hold.
#[inline]
pub fn dict_view_type_for_kind(kind: DictViewKind) -> &'static PyType {
    match kind {
        DictViewKind::Keys => &DICT_KEYS_TYPE,
        DictViewKind::Values => &DICT_VALUES_TYPE,
        DictViewKind::Items => &DICT_ITEMS_TYPE,
    }
}

/// Allocate a `dict_keys` view.  `W_DictViewKeysObject.__init__`
/// (`dictmultiobject.py`) is a two-field store of the source dict onto a
/// class whose type is known at the constructor; `descr_keys` returns
/// `W_DictViewKeysObject(self)`.  The type pointer is the `DICT_KEYS_TYPE`
/// singleton so `fuse_boxing_alloc` can rewrite `malloc_typed_managed` to
/// `new_with_vtable` — a runtime `dict_view_type_for_kind` match leaves
/// `ob_type` non-constant and the cluster unfused.
#[inline(always)]
pub fn w_dict_keys_new(w_dict: PyObjectRef) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_DictViewObject {
        ob_header: PyObject {
            ob_type: &DICT_KEYS_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_KEYS_TYPE),
        },
        kind: DictViewKind::Keys,
        w_dict,
    }) as PyObjectRef
}

/// Allocate a `dict_values` view.  `W_DictViewValuesObject.__init__`.
#[inline(always)]
pub fn w_dict_values_new(w_dict: PyObjectRef) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_DictViewObject {
        ob_header: PyObject {
            ob_type: &DICT_VALUES_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_VALUES_TYPE),
        },
        kind: DictViewKind::Values,
        w_dict,
    }) as PyObjectRef
}

/// Allocate a `dict_items` view.  `W_DictViewItemsObject.__init__`.
#[inline(always)]
pub fn w_dict_items_new(w_dict: PyObjectRef) -> PyObjectRef {
    crate::lltype::malloc_typed_managed(W_DictViewObject {
        ob_header: PyObject {
            ob_type: &DICT_ITEMS_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_ITEMS_TYPE),
        },
        kind: DictViewKind::Items,
        w_dict,
    }) as PyObjectRef
}

/// Dispatch to the type-specific constructor.  Residual / non-wrapper
/// callers that only have a `DictViewKind` still go through a fused
/// `new_with_vtable` body.
#[inline(always)]
pub fn w_dict_view_new(w_dict: PyObjectRef, kind: DictViewKind) -> PyObjectRef {
    match kind {
        DictViewKind::Keys => w_dict_keys_new(w_dict),
        DictViewKind::Values => w_dict_values_new(w_dict),
        DictViewKind::Items => w_dict_items_new(w_dict),
    }
}

/// Test whether `obj` is any of the three view types.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_view(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &DICT_KEYS_TYPE)
            || py_type_check(obj, &DICT_VALUES_TYPE)
            || py_type_check(obj, &DICT_ITEMS_TYPE)
    }
}

/// Test whether `obj` is the keys view.
///
/// `dictmultiobject.py _is_set_like` — `isinstance(w_other,
/// W_DictViewKeysObject)`.  The keys view is set-like: it defines `__eq__`
/// and so is unhashable.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_view_keys(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DICT_KEYS_TYPE) }
}

/// Test whether `obj` is the items view.
///
/// `dictmultiobject.py _is_set_like` — `isinstance(w_other,
/// W_DictViewItemsObject)`.  The items view is set-like: it defines `__eq__`
/// and so is unhashable.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_view_items(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DICT_ITEMS_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewObject`.
#[inline]
pub unsafe fn w_dict_view_get_kind(obj: PyObjectRef) -> DictViewKind {
    unsafe { (*(obj as *const W_DictViewObject)).kind }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewObject`.
#[inline]
pub unsafe fn w_dict_view_get_dict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_DictViewObject)).w_dict }
}

// -- DictStrategy family
// PyPy keeps DictStrategy and the built-in concrete strategies in
// pypy/objspace/std/dictmultiobject.py.  Keep them in this Rust module too so
// paths and generated JIT symbol names line up with the source port.

/// `dictmultiobject.py DictStrategy` — abstract base.
///
/// Concrete strategies implement the required-no-default methods
/// (`get_empty_storage`, `getitem`, `setitem`, `delitem`, `length`,
/// iterator producers).  Default implementations cover the
/// derived APIs PyPy provides as overridable fallbacks
/// (`getitem_str`, `setitem_str`, `setdefault`, `w_keys`, `values`,
/// `items`, `popitem`, `clear`, `listview_*`, `view_as_kwargs`).
///
/// The PyPy `space` argument is omitted: pyre's `pyre-object` crate
/// has no `ObjSpace` shim, so callers that need str-wrapping
/// (`getitem_str`'s default → `getitem(space.newtext(key))`) call
/// `crate::unicodeobject::w_str_from_storage` directly.
/// Identifies a concrete `DictStrategy` impl.  Used by
/// `dictmultiobject::strategy_is` to discriminate strategies that share
/// the same data pointer because each `*_DICT_STRATEGY` static is a
/// unit-struct ZST (Rust collapses ZST static addresses, so
/// pointer/`std::ptr::eq` checks cannot tell them apart reliably).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrategyKind {
    Empty,
    EmptyKwargs,
    Object,
    Bytes,
    Unicode,
    Int,
    Identity,
    Kwargs,
    Module,
    /// `pypy/objspace/std/mapdict.py MapDictStrategy` — the strategy a
    /// user instance's `__dict__` adopts, routing get/set/del through the
    /// instance's mapdict map+storage.  Implemented in pyre-interpreter
    /// (`objspace::std::mapdict::MapDictStrategy`) because the map-node layer
    /// lives there; this variant lets `strategy_is` discriminate it.
    Map,
    /// `pypy/objspace/std/classdict.py ClassDictStrategy` — exposes a
    /// `W_TypeObject` namespace. `dstorage` erases the type; implemented in
    /// pyre-interpreter (`objspace::std::classdict::ClassDictStrategy`).
    Class,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DictPopError;

pub trait DictStrategy {
    /// Discriminate strategies by concrete impl — see [`StrategyKind`]
    /// for the rationale.  Required because pointer comparison on the
    /// `&'static dyn DictStrategy` slot is unreliable for ZST strategies.
    fn strategy_kind(&self) -> StrategyKind;

    /// `dictmultiobject.py get_empty_storage` — return a
    /// freshly-allocated erased storage for this strategy.  Required.
    fn get_empty_storage(&self) -> GCREF;

    /// `dictmultiobject.py getitem` — required. Returns `PY_NULL` on a miss.
    ///
    /// Returns `PY_NULL` for a miss: the slot result is one nullable
    /// word, as `dictmultiobject.py getitem` returns `None`.
    ///
    /// # Safety
    /// `w_dict` and `w_key` must be valid PyObjectRef.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef;

    /// `dictmultiobject.py getitem_str` — default falls
    /// through to `getitem(w_dict, space.newtext(key))`.
    /// Returns `PY_NULL` on a miss.
    ///
    /// `key` is the rstr `STR` of the text (`W_UnicodeObject.value`), one word, as `dictmultiobject.py getitem_str` takes an RPython str.
    ///
    /// Returns `PY_NULL` for a miss: the slot result is one nullable
    /// word, as `dictmultiobject.py getitem` returns `None`.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str) -> PyObjectRef {
        let w_key = crate::unicodeobject::w_str_from_storage(key);
        self.getitem(w_dict, w_key)
    }

    /// [`getitem_str`](Self::getitem_str) told `key`'s digest, so a strategy
    /// whose table is keyed on `space.hash_w` can probe without re-hashing the
    /// bytes.  The digest is the memo `rstr.py ll_strhash` keeps in the
    /// string object the caller borrowed `key` from; zero means it holds none.
    /// Only the strategies backed by that table override this — the default
    /// discards the digest and re-hashes.
    /// Returns `PY_NULL` on a miss.
    ///
    /// `key` is the rstr `STR` of the text (`W_UnicodeObject.value`), one word, as `dictmultiobject.py getitem_str` takes an RPython str.
    ///
    /// Returns `PY_NULL` for a miss: the slot result is one nullable
    /// word, as `dictmultiobject.py getitem` returns `None`.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn getitem_str_hashed(
        &self,
        w_dict: PyObjectRef,
        key: *mut Utf8Str,
        hash: i64,
    ) -> PyObjectRef {
        let _ = hash;
        self.getitem_str(w_dict, key)
    }

    /// `dictmultiobject.py setitem` — required.
    ///
    /// # Safety
    /// `w_dict`, `w_key`, and `w_value` must be valid PyObjectRef.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef);

    /// `dictmultiobject.py setitem_str` — default falls
    /// through to `setitem(w_dict, space.newtext(key), w_value)`.
    ///
    /// `key` is the rstr `STR` of the text (`W_UnicodeObject.value`), one word, as `dictmultiobject.py getitem_str` takes an RPython str.
    ///
    /// # Safety
    /// `w_dict` and `w_value` must be valid PyObjectRef.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str, w_value: PyObjectRef) {
        let w_key = crate::unicodeobject::w_str_from_storage(key);
        self.setitem(w_dict, w_key, w_value);
    }

    /// [`setitem_str`](Self::setitem_str)'s counterpart to
    /// [`getitem_str_hashed`](Self::getitem_str_hashed), taking `key`'s digest
    /// the same way.
    ///
    /// `key` is the rstr `STR` of the text (`W_UnicodeObject.value`), one word, as `dictmultiobject.py getitem_str` takes an RPython str.
    ///
    /// # Safety
    /// `w_dict` and `w_value` must be valid PyObjectRef.
    unsafe fn setitem_str_hashed(
        &self,
        w_dict: PyObjectRef,
        key: *mut Utf8Str,
        hash: i64,
        w_value: PyObjectRef,
    ) {
        let _ = hash;
        self.setitem_str(w_dict, key, w_value);
    }

    /// `dictmultiobject.py delitem` — required.
    /// Returns `true` if a key was removed.
    ///
    /// # Safety
    /// `w_dict` and `w_key` must be valid PyObjectRef.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool;

    /// `dictmultiobject.py length` — required.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize;

    /// `dictmultiobject.py setdefault` — slow default
    /// implementation; concrete strategies override.
    ///
    /// # Safety
    /// `w_dict`, `w_key`, and `w_value` must be valid PyObjectRef.
    unsafe fn setdefault(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
        w_value: PyObjectRef,
    ) -> PyObjectRef {
        // `getitem` hashes the key, so all three by-value words are stale by
        // the time the store runs — and the default is handed back after the
        // store, which allocates again.  Slot them and read back at each use.
        let roots = crate::gc_roots::push_roots();
        let dict_slot = roots.publish(&[w_dict, w_key, w_value]);
        let w_result = self.getitem(w_dict, w_key);
        if !w_result.is_null() {
            return w_result;
        }
        self.setitem(
            roots.get(dict_slot),
            roots.get(dict_slot + 1),
            roots.get(dict_slot + 2),
        );
        roots.get(dict_slot + 2)
    }

    /// `dictmultiobject.py w_keys` — collect all keys.
    /// Default builds from the strategy's iterator API.  Concrete
    /// strategies that can short-circuit (e.g. cloning an internal
    /// `Vec`) override.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef>;

    /// `dictmultiobject.py values` — collect all values.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef>;

    /// `dictmultiobject.py items` — collect (key, value) pairs.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)>;

    /// Fetch the `(key, value)` entry in slot `index`, wrapping only that one
    /// key — O(1) per step for the typed strategies.  Answers `None` for a
    /// slot past the end and for a tombstone, which is `entries.valid(i)`
    /// (`rordereddict.py`).
    ///
    /// `index` is a slot, not the n-th live pair; [`Self::next_slot`] is what
    /// walks from one to the next.  Stands in for PyPy's `next_*_entry`
    /// pulling one entry from the strategy cursor
    /// (`dictmultiobject.py`); the dict view iterator's integer field
    /// replaces the live iterator the GC-object layout cannot hold.  The
    /// default materialises the full list (fine for the tiny empty
    /// strategies); typed strategies override for O(1).
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        self.items(w_dict).into_iter().nth(index)
    }

    /// Read the value at iteration position `index` without requiring a
    /// separately materialised key
    /// (`dictmultiobject.py AbstractTypedStrategy.getitem`).
    /// Returns `PY_NULL` on a miss.
    ///
    /// Returns `PY_NULL` for a miss: the slot result is one nullable
    /// word, as `dictmultiobject.py getitem` returns `None`.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> PyObjectRef {
        match self.nth_item(w_dict, index) {
            Some((_, value)) => value,
            None => PY_NULL,
        }
    }

    /// The first live slot at or after `from` — `_ll_dictnext`'s scan,
    /// "while i < num_ever_used_items: if entries.valid(i)"
    /// (`rordereddict.py`).
    ///
    /// A delete leaves a tombstone rather than closing the hole, so a slot
    /// number is not an iteration position and a walk has to skip.  The
    /// default numbers its entries densely, which is true of the storage-free
    /// strategies.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        (from < self.length(w_dict)).then_some(from)
    }

    /// [`Self::next_slot`] descending, for `reversed(d)`: PyPy's
    /// `getiterreversed` is `objectmodel.reversed_dict`
    /// (`dictmultiobject.py`), itself a descending slot walk with the
    /// same `valid()` skip.  `before` is exclusive, so a fresh reverse walk
    /// starts at `usize::MAX`.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        before.min(self.length(w_dict)).checked_sub(1)
    }

    /// `dictmultiobject.py DictStrategy.pop` — remove and
    /// return the value for `w_key`.  Returns `Ok(value)` on hit,
    /// `Ok(w_default)` on miss when a default is provided, or
    /// `Err(())` on miss without default (caller raises KeyError).
    ///
    /// # Safety
    /// `w_dict` and `w_key` must be valid PyObjectRef.
    unsafe fn pop(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
        w_default: Option<PyObjectRef>,
    ) -> Result<PyObjectRef, DictPopError> {
        // dictmultiobject.py pop
        // `getitem` hashes the key and `delitem` allocates, so the receiver,
        // the key, the default and the fetched value are each stale at their
        // next use.  A `None` default slots a null word, which the root stack
        // tolerates.
        let roots = crate::gc_roots::push_roots();
        let dict_slot = roots.publish(&[w_dict, w_key, w_default.unwrap_or(std::ptr::null_mut())]);
        let value_slot = dict_slot + 3;
        let val = self.getitem(w_dict, w_key);
        if !val.is_null() {
            let _ = roots.pin_root(val);
            if !self.delitem(roots.get(dict_slot), roots.get(dict_slot + 1)) {
                return Err(DictPopError);
            }
            Ok(roots.get(value_slot))
        } else if w_default.is_some() {
            Ok(roots.get(dict_slot + 2))
        } else {
            Err(DictPopError)
        }
    }

    /// `dictmultiobject.py popitem` — remove and return the
    /// most recently inserted (key, value) pair.  Python 3.7+ `popitem`
    /// is LIFO (`pypy/objspace/std/dictmultiobject.py
    /// descr_popitem`); the default routes through the strategy's
    /// `items()` and pops the tail.  Concrete strategies override for
    /// O(1) backings (e.g. `ModuleDictStrategy` uses
    /// `IndexMap::pop`).
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        // `items` and `delitem` both allocate, so the receiver is stale before
        // the removal and the returned pair is stale after it.
        let roots = crate::gc_roots::push_roots();
        let dict_slot = roots.base();
        let w_dict = roots.pin_root(w_dict);
        let mut items = self.items(w_dict);
        let (w_key, w_value) = items.pop()?;
        let pair_base = roots.publish(&[w_key, w_value]);
        self.delitem(roots.get(dict_slot), roots.get(pair_base));
        Some((roots.get(pair_base), roots.get(pair_base + 1)))
    }

    /// `dictmultiobject.py getiterreversed` — iterate
    /// (key, value) pairs in reverse insertion order (used by
    /// `reversed(dict)` per `pypy/objspace/std/dictmultiobject.py
    /// W_DictMultiObject.descr_reversed`).  Default reverses the
    /// strategy's materialised `items()`; ordered backings override
    /// for streaming reverse iteration.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let mut items = self.items(w_dict);
        items.reverse();
        items
    }

    /// `dictmultiobject.py W_DictMultiObject.descr_copy` —
    /// strategy-side hook for `dict.copy()`.  Default allocates a
    /// fresh W_DictObject and copies via `items()`; ordered backings
    /// (e.g. `ModuleDictStrategy`) override to unwrap cell-wrapped
    /// values during the copy per `celldict.py copy`.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        // The materialised pairs sit in a Rust-heap Vec the collector never
        // walks and the fresh dict moves under every store, so the whole run
        // is published before the dict is minted and each word is read back at
        // the store that consumes it.
        let roots = crate::gc_roots::push_roots();
        let pairs_base = roots.base();
        let items = self.items(w_dict);
        for &(k, v) in &items {
            roots.publish(&[k, v]);
        }
        let new_slot = roots.publish(&[crate::dictmultiobject::w_dict_new()]);
        for i in 0..items.len() {
            crate::dictmultiobject::w_dict_store(
                roots.get(new_slot),
                roots.get(pairs_base + 2 * i),
                roots.get(pairs_base + 2 * i + 1),
            );
        }
        roots.get(new_slot)
    }

    /// `dictmultiobject.py clear` — reset to EmptyDictStrategy.
    /// Concrete strategies override to swap the W_DictObject's
    /// `dstrategy` and `dstorage`.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn clear(&self, w_dict: PyObjectRef);

    /// `dictmultiobject.py AbstractTypedStrategy.switch_to_object_strategy`
    /// (plus `:732 EmptyDictStrategy.switch_to_object_strategy`,
    /// `:1195+ ObjectDictStrategy` no-op).  Polymorphic dispatch lets
    /// callers that hold only `&dyn DictStrategy` (e.g. the storage
    /// back-mirror `*_no_proxy` helpers) demote typed storage to
    /// ObjectDictStrategy without inspecting strategy identity.
    ///
    /// Default just relabels the strategy pointer: correct for
    /// `ObjectDictStrategy` (already object) and `UnicodeDictStrategy`
    /// (storage shape matches `Vec<(PyObjectRef, PyObjectRef)>`).
    /// Strategies with typed storage layouts override to rebuild the
    /// object Vec before swapping the pointer.
    ///
    /// # Safety
    /// `w_dict` must point at a valid `W_DictObject` whose
    /// `dstrategy` is `self`.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_set_strategy(w_dict, &OBJECT_DICT_STRATEGY_REF);
    }

    /// `dictmultiobject.py listview_bytes` — default returns
    /// `None`; bytes/str strategies override to expose backing list.
    fn listview_bytes(&self, _w_dict: PyObjectRef) -> Option<Vec<Vec<u8>>> {
        None
    }

    /// `dictmultiobject.py listview_ascii` — default returns
    /// `None`.
    fn listview_ascii(&self, _w_dict: PyObjectRef) -> Option<Vec<String>> {
        None
    }

    /// `dictmultiobject.py listview_int` — default returns
    /// `None`.
    fn listview_int(&self, _w_dict: PyObjectRef) -> Option<Vec<i64>> {
        None
    }

    /// `dictmultiobject.py DictStrategy.view_as_kwargs` —
    /// abstract base returns `([], [])` in PyPy's EmptyDictStrategy
    /// override and the upstream default `(None, None)` for non-
    /// kwarg/non-unicode strategies (`:568-569`).  Concrete strategies
    /// override: KwargsDictStrategy returns the parallel arrays
    /// (`kwargsdict.py`), UnicodeDictStrategy mints arrays
    /// from its r_dict storage (`dictmultiobject.py`).
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef.
    unsafe fn view_as_kwargs(
        &self,
        _w_dict: PyObjectRef,
    ) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
        (None, None)
    }

    /// Strategy-side GC trace dispatch for `W_DictObject.dstorage`.
    ///
    /// TODO: RPython's translator generates a
    /// per-`rerased`-pair GC trace function from the
    /// `new_erasing_pair("name")` call (`rpython/rlib/rerased.py`
    /// `new_erasing_pair`), so each PyPy strategy's storage layout
    /// (`r_dict`, `Dict[str, ...]`, `Dict[int, ...]`, parallel arrays)
    /// is traced through its compile-time-known type.  Pyre's
    /// `W_DictObject.dstorage: GCREF` is erased at runtime, so the
    /// per-strategy trace must dispatch through this trait method.
    ///
    /// Default walks the uniform `Vec<(PyObjectRef, PyObjectRef)>`
    /// shape — every strategy shares that layout until Slices D3/D4
    /// migrate Int/Bytes/Unicode/Kwargs to native typed storages,
    /// where they override `walk_gc_refs` to walk i64-keyed pairs
    /// (skipping the i64 half), `BytesKey` pairs (the block and the
    /// value), or parallel `keys`/`values` arrays.
    ///
    /// # Safety
    /// `w_dict` must be a valid PyObjectRef pointing at a W_DictObject
    /// whose strategy is `self`.  `visitor` may mutate the visited
    /// PyObjectRef slot to relocate the referenced object during GC.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let entries = crate::dictmultiobject::w_dict_object_storage_mut(w_dict);
        for (key, value) in entries.iter_mut_for_trace() {
            // See `w_dict_walk_entries_mut` — ObjectKey.hash is precomputed
            // and identity-stable across GC moves, so writing through the
            // raw obj slot does not desync the IndexMap bucket index.
            let key_ptr = key as *const crate::dictmultiobject::ObjectKey
                as *mut crate::dictmultiobject::ObjectKey;
            visitor(std::ptr::addr_of_mut!((*key_ptr).obj));
            visitor(value as *mut PyObjectRef);
        }
        let slot = entries.entries_slot() as *mut PyObjectRef;
        visitor(slot);
        entries.visit_indexes(&mut |slot| visitor(slot as *mut PyObjectRef));
    }
}

/// Convert a dict-strategy miss (`PY_NULL`) back to `Option`.
#[inline]
pub fn nullable_to_option(w: PyObjectRef) -> Option<PyObjectRef> {
    if w.is_null() { None } else { Some(w) }
}

/// The one-word strategy slot a dict stores, holding the trait object that
/// does the dispatch.
///
/// PyPy's `dstrategy` is one instance pointer and the strategy's *type* lives
/// in the instance header, which is what the JIT's `guard_class` reads when it
/// inlines `self.get_strategy().getitem(...)`.  A Rust `&dyn` inverts that —
/// the vtable rides in the reference — so a dict holding one directly has no
/// field to guard, and its data word is the address of a zero-sized static,
/// which Rust is free to make identical across distinct strategies.  Boxing
/// the trait object in this `#[repr(C)]` holder restores both properties: the
/// dict slot is a single pointer, and each strategy's holder is a distinct,
/// non-zero-sized address.
///
/// It derefs to `dyn DictStrategy`, so `dict.dstrategy.getitem(..)` and every
/// other call through the slot reads exactly as before.
#[repr(C)]
pub struct DictStrategyRef {
    /// `DictStrategy` class identity as a stored field (`convert_const`
    /// of a prebuilt).  Compared by [`strategy_is`] instead of a
    /// vtable `strategy_kind` call, so the real path types a `getfield`.
    pub kind: StrategyKind,
    pub imp: &'static dyn crate::dictmultiobject::DictStrategy,
    /// GC owner of a per-dict strategy instance.  Stateless
    /// `space.fromcache(...)` holders leave this null; a module dict's holder
    /// points at its `ModuleDictStrategy`, matching PyPy's `mstrategy` instance
    /// while keeping the Rust trait object behind a one-word dict slot.
    pub owner: GCREF,
}

/// MiniMark-allocated by `gc_alloc_storage_box` with the tid from `dict_strategy_ref_gc_type_id`;
/// this impl declares the gckind only (allocation keeps that tid).
impl crate::lltype::GcType for DictStrategyRef {
    fn type_id() -> u32 {
        u32::MAX
    }
    const SIZE: usize = 0;
}

/// The holders below wrap only the stateless unit-struct singletons, which are
/// shared process-wide exactly as `space.fromcache(StrategyCls)` shares them
/// upstream.  `DictStrategy` itself cannot carry a `Sync` bound —
/// `ModuleDictStrategy`'s `GlobalCache` holds `*mut PyObject` — and a module
/// dict never uses a holder anyway: its strategy lives behind
/// `W_ModuleDictObject.mstrategy`.
unsafe impl Sync for DictStrategyRef {}
unsafe impl Send for DictStrategyRef {}

impl std::ops::Deref for DictStrategyRef {
    type Target = dyn crate::dictmultiobject::DictStrategy;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.imp
    }
}

/// `pypy/objspace/std/dictmultiobject.py+ ObjectDictStrategy`
/// process-wide singleton.  PyPy's `space.fromcache(ObjectDictStrategy)`
/// returns the same instance for every space, and `W_DictObject`'s
/// `dstrategy` slot points at it; pyre's `ObjectDictStrategy` is a
/// zero-sized type so a `&'static` reference suffices.  Use this
/// constant whenever the line-by-line port calls
/// `space.fromcache(ObjectDictStrategy)`.
pub static OBJECT_DICT_STRATEGY: ObjectDictStrategy = ObjectDictStrategy;

/// `pypy/objspace/std/dictmultiobject.py EmptyDictStrategy`
/// singleton — same `space.fromcache` semantics as
/// `OBJECT_DICT_STRATEGY`.
pub static EMPTY_DICT_STRATEGY: EmptyDictStrategy = EmptyDictStrategy;

/// `pypy/objspace/std/kwargsdict.py EmptyKwargsDictStrategy`
/// singleton.  Subclass of `EmptyDictStrategy` that promotes
/// directly to `KwargsDictStrategy` on the first unicode setitem,
/// skipping the regular `UnicodeDictStrategy` step.  Selected by
/// `dictmultiobject.py allocate_and_init_instance` when the
/// `kwargs=True` flag is set — pyre's `w_dict_new_kwargs()`
/// allocator wires this singleton in for function-call kwarg dicts.
pub static EMPTY_KWARGS_DICT_STRATEGY: EmptyKwargsDictStrategy = EmptyKwargsDictStrategy;

/// `pypy/objspace/std/dictmultiobject.py BytesDictStrategy`
/// singleton.
pub static BYTES_DICT_STRATEGY: BytesDictStrategy = BytesDictStrategy;

/// `pypy/objspace/std/dictmultiobject.py UnicodeDictStrategy`
/// singleton.
pub static UNICODE_DICT_STRATEGY: UnicodeDictStrategy = UnicodeDictStrategy;

/// `pypy/objspace/std/dictmultiobject.py IntDictStrategy`
/// singleton.
pub static INT_DICT_STRATEGY: IntDictStrategy = IntDictStrategy;

/// The [`DictStrategyRef`] holders a dict's `dstrategy` slot points at — one
/// per singleton above.  Each is a distinct address; the singletons themselves
/// are zero-sized and need not be.
pub static OBJECT_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: StrategyKind::Object,
    imp: &OBJECT_DICT_STRATEGY,
    owner: std::ptr::null_mut(),
};
pub static EMPTY_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: StrategyKind::Empty,
    imp: &EMPTY_DICT_STRATEGY,
    owner: std::ptr::null_mut(),
};
pub static EMPTY_KWARGS_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: StrategyKind::EmptyKwargs,
    imp: &EMPTY_KWARGS_DICT_STRATEGY,
    owner: std::ptr::null_mut(),
};
pub static BYTES_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: StrategyKind::Bytes,
    imp: &BYTES_DICT_STRATEGY,
    owner: std::ptr::null_mut(),
};
pub static UNICODE_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: StrategyKind::Unicode,
    imp: &UNICODE_DICT_STRATEGY,
    owner: std::ptr::null_mut(),
};
pub static INT_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: StrategyKind::Int,
    imp: &INT_DICT_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `dictmultiobject.py EmptyDictStrategy`.
///
/// ```python
/// class EmptyDictStrategy(DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("empty")
///
///     def get_empty_storage(self):
///         return self.erase(None)
///
///     def switch_to_correct_strategy(self, w_dict, w_key):
///         if type(w_key) is self.space.StringObjectCls:
///             self.switch_to_bytes_strategy(w_dict)
///             return
///         if type(w_key) is self.space.UnicodeObjectCls:
///             self.switch_to_unicode_strategy(w_dict)
///             return
///         w_type = self.space.type(w_key)
///         if self.space.is_w(w_type, self.space.w_int):
///             self.switch_to_int_strategy(w_dict)
///         elif w_type.compares_by_identity():
///             self.switch_to_identity_strategy(w_dict)
///         else:
///             self.switch_to_object_strategy(w_dict)
///
///     def getitem(self, w_dict, w_key):
///         self.space.hash(w_key)
///         return None
/// ```
///
/// `switch_to_correct_strategy` discriminates the key type:
/// `is_bytes` → Bytes, exact str → Unicode, plain `is_int`
/// (excluding bool) → Int.  IdentityDictStrategy is selected by
/// the `compares_by_identity` MRO walker (Slice D5) routed through
/// `dict_eq_hook::COMPARES_BY_IDENTITY_HOOK`; everything else falls
/// into the Object fallback.
pub struct EmptyDictStrategy;

/// `w_dict.dstorage = strategy.erase(storage)` / `w_dict.set_strategy`:
/// one `setfield_gc` pair. The GC transform records the old-to-young
/// edge; [`dict_write_barrier`] is that record.
pub(crate) unsafe fn install_dict_storage(
    w_dict: PyObjectRef,
    storage: GCREF,
    strategy: &'static DictStrategyRef,
) {
    let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
    dict.dstorage = storage;
    dict.dstrategy = strategy;
    dict_write_barrier(w_dict);
}

/// Install a freshly allocated strategy storage on an empty dict.
///
/// RPython's `w_dict.dstorage = strategy.erase(storage)` is a `setfield_gc`:
/// the GC transform roots and reloads `w_dict` across the storage allocation,
/// then records the old-to-young edge.  Do the same explicitly here.
unsafe fn install_empty_strategy(w_dict: PyObjectRef, strategy: &'static DictStrategyRef) {
    let _roots = crate::gc_roots::push_roots();
    let dict_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_dict);
    let storage = strategy.get_empty_storage();
    let _ = crate::gc_roots::pin_root(storage as crate::PyObjectRef);
    let w_dict = crate::gc_roots::shadow_stack_get(dict_slot);
    install_dict_storage(w_dict, storage, strategy);
}

impl EmptyDictStrategy {
    /// `dictmultiobject.py switch_to_correct_strategy`.
    ///
    /// # Safety
    /// `w_dict` and `w_key` must be valid PyObjectRef.
    pub(crate) unsafe fn switch_to_correct_strategy(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
    ) {
        // `:693-695 type(w_key) is self.space.StringObjectCls`
        // (Python 2 str / Python 3 bytes).
        if crate::is_bytes(w_key) {
            self.switch_to_bytes_strategy(w_dict);
            return;
        }
        // `:696-698 type(w_key) is self.space.UnicodeObjectCls`
        // (Python 2 unicode / Python 3 str).
        if crate::is_exact_type(w_key, &crate::STR_TYPE) {
            install_empty_strategy(w_dict, &UNICODE_DICT_STRATEGY_REF);
            return;
        }
        // `:700-701 is_w(w_type, self.space.w_int)` — plain int only;
        // bool inherits from int in Python 3 but PyPy's
        // `is_w(type(b), w_int)` is False because `type(True) is bool`.
        if crate::listobject::is_plain_int1(w_key) {
            self.switch_to_int_strategy(w_dict);
            return;
        }
        // `:702-705 elif w_type.compares_by_identity():
        //     self.switch_to_identity_strategy(w_dict)`.
        // `w_type` is `space.type(w_key)`. The hook is
        // `dict_eq_hook::COMPARES_BY_IDENTITY_HOOK` (pyre-interpreter
        // installs the MRO walker at startup; pyre-object snapshot/lib
        // tests return `None` and fall through to the Object strategy).
        if key_compares_by_identity(w_key) {
            self.switch_to_identity_strategy(w_dict);
            return;
        }
        self.switch_to_object_strategy(w_dict);
    }

    /// `dictmultiobject.py switch_to_int_strategy`:
    /// ```python
    /// def switch_to_int_strategy(self, w_dict):
    ///     strategy = self.space.fromcache(IntDictStrategy)
    ///     storage = strategy.get_empty_storage()
    ///     w_dict.set_strategy(strategy)
    ///     w_dict.dstorage = storage
    /// ```
    ///
    /// Empty dest storage is `erased(None)`; overwriting `dstorage`
    /// with the fresh `IntDictStrategy::get_empty_storage` box matches
    /// `w_dict.dstorage = storage`.
    ///
    /// # Safety
    /// `w_dict` must point at a valid `W_DictObject` whose strategy
    /// is currently `EmptyDictStrategy`.
    unsafe fn switch_to_int_strategy(&self, w_dict: PyObjectRef) {
        install_empty_strategy(w_dict, &INT_DICT_STRATEGY_REF);
    }

    /// `dictmultiobject.py switch_to_bytes_strategy`:
    /// ```python
    /// def switch_to_bytes_strategy(self, w_dict):
    ///     strategy = self.space.fromcache(BytesDictStrategy)
    ///     storage = strategy.get_empty_storage()
    ///     w_dict.set_strategy(strategy)
    ///     w_dict.dstorage = storage
    /// ```
    ///
    /// The field overwrite is a `setfield_gc`; same lifetime contract
    /// as `switch_to_int_strategy`.
    ///
    /// # Safety
    /// Same as [`switch_to_int_strategy`].
    unsafe fn switch_to_bytes_strategy(&self, w_dict: PyObjectRef) {
        install_empty_strategy(w_dict, &BYTES_DICT_STRATEGY_REF);
    }

    /// `dictmultiobject.py switch_to_identity_strategy`:
    /// ```python
    /// def switch_to_identity_strategy(self, w_dict):
    ///     from pypy.objspace.std.identitydict import IdentityDictStrategy
    ///     strategy = self.space.fromcache(IdentityDictStrategy)
    ///     storage = strategy.get_empty_storage()
    ///     w_dict.set_strategy(strategy)
    ///     w_dict.dstorage = storage
    /// ```
    ///
    /// IdentityDictStrategy storage shape matches ObjectDictStrategy
    /// (`Vec<(PyObjectRef, PyObjectRef)>`) — distinction is in the
    /// lookup comparison (raw `==` instead of `dict_keys_equal`).  We
    /// still allocate a fresh box per PyPy's set_strategy + dstorage
    /// reset contract.
    ///
    /// # Safety
    /// Same as [`switch_to_int_strategy`].
    unsafe fn switch_to_identity_strategy(&self, w_dict: PyObjectRef) {
        install_empty_strategy(w_dict, &crate::identitydict::IDENTITY_DICT_STRATEGY_REF);
    }
}

/// `kwargsdict.py EmptyKwargsDictStrategy(EmptyDictStrategy)`.
///
/// ```python
/// class EmptyKwargsDictStrategy(EmptyDictStrategy):
///     def switch_to_unicode_strategy(self, w_dict):
///         strategy = self.space.fromcache(KwargsDictStrategy)
///         storage = strategy.get_empty_storage()
///         w_dict.set_strategy(strategy)
///         w_dict.dstorage = storage
/// ```
///
/// Rust has no inheritance: the override is expressed as a
/// separate struct whose `setitem` / `setitem_str` route the
/// unicode-key branch into `KwargsDictStrategy` instead of
/// `UnicodeDictStrategy`.  Non-unicode key branches and every
/// other DictStrategy hook delegate to `EMPTY_DICT_STRATEGY`
/// because PyPy's subclass inherits those methods unchanged.
pub struct EmptyKwargsDictStrategy;

impl EmptyKwargsDictStrategy {
    /// `kwargsdict.py switch_to_unicode_strategy` — the
    /// subclass override; promotes the W_DictObject straight to
    /// `KWARGS_DICT_STRATEGY` instead of `UNICODE_DICT_STRATEGY`.
    ///
    /// # Safety
    /// `w_dict` must be a W_DictObject whose strategy is
    /// `EMPTY_KWARGS_DICT_STRATEGY`.
    unsafe fn switch_to_kwargs_strategy(&self, w_dict: PyObjectRef) {
        install_empty_strategy(w_dict, &crate::kwargsdict::KWARGS_DICT_STRATEGY_REF);
    }

    /// `dictmultiobject.py switch_to_correct_strategy`
    /// duplicated with the unicode branch redirected per PyPy's
    /// subclass MRO dispatch (kwargsdict.py switch_to_unicode_strategy).
    ///
    /// # Safety
    /// `w_dict` and `w_key` must be valid PyObjectRef.
    pub(crate) unsafe fn switch_to_correct_strategy(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
    ) {
        if crate::is_bytes(w_key) {
            EMPTY_DICT_STRATEGY.switch_to_bytes_strategy(w_dict);
            return;
        }
        if crate::is_exact_type(w_key, &crate::STR_TYPE) {
            self.switch_to_kwargs_strategy(w_dict);
            return;
        }
        if crate::listobject::is_plain_int1(w_key) {
            EMPTY_DICT_STRATEGY.switch_to_int_strategy(w_dict);
            return;
        }
        if key_compares_by_identity(w_key) {
            EMPTY_DICT_STRATEGY.switch_to_identity_strategy(w_dict);
            return;
        }
        // `kwargsdict.py EmptyKwargsDictStrategy` inherits the parent's
        // `switch_to_object_strategy` (`dictmultiobject.py`),
        // which allocates ObjectDictStrategy's empty Vec. Routing
        // through the parent avoids leaving `w_dict_new_kwargs`'s
        // null `dstorage` in place when the first key isn't unicode.
        EMPTY_DICT_STRATEGY.switch_to_object_strategy(w_dict);
    }
}

crate::register_dyn_impl_vtable!(
    DYN_IMPL_EmptyKwargsDictStrategy,
    EmptyKwargsDictStrategy,
    DictStrategy
);
impl DictStrategy for EmptyKwargsDictStrategy {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::EmptyKwargs
    }

    /// `kwargsdict.py` inherits `EmptyDictStrategy.get_empty_storage`.
    fn get_empty_storage(&self) -> GCREF {
        EMPTY_DICT_STRATEGY.get_empty_storage()
    }

    /// An empty kwargs dict has the `erased(None)` null `dstorage` from
    /// `w_dict_new_kwargs` and holds no entries — `setitem`/`setitem_str`
    /// switch to a concrete strategy before storing — so there is nothing
    /// to trace.  Same walk as `EmptyDictStrategy`.
    unsafe fn walk_gc_refs(
        &self,
        _w_dict: PyObjectRef,
        _visitor: &mut dyn FnMut(*mut PyObjectRef),
    ) {
    }

    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef {
        EMPTY_DICT_STRATEGY.getitem(w_dict, w_key)
    }

    unsafe fn setitem(
        &self,
        mut w_dict: PyObjectRef,
        mut w_key: PyObjectRef,
        mut w_value: PyObjectRef,
    ) {
        // `dictmultiobject.py setitem` — promote via the subclass's
        // `switch_to_correct_strategy`, then setitem on the new
        // strategy.  The kwargs override redirects the unicode
        // branch to KwargsDictStrategy.
        crate::with_roots!(w_dict, w_key, w_value => {
            self.switch_to_correct_strategy(w_dict, w_key)
        });
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    unsafe fn setitem_str(
        &self,
        mut w_dict: PyObjectRef,
        key: *mut Utf8Str,
        mut w_value: PyObjectRef,
    ) {
        // `dictmultiobject.py` setitem_str — caller already
        // chose the str-keyed path, so promote directly to
        // KwargsDictStrategy via the subclass override.
        let key_bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key) };
        let key = unsafe { std::str::from_utf8_unchecked(key_bytes) };
        crate::with_roots!(w_dict, w_value => self.switch_to_kwargs_strategy(w_dict));
        crate::dictmultiobject::w_dict_setitem_str(w_dict, key, w_value);
    }

    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        EMPTY_DICT_STRATEGY.delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        EMPTY_DICT_STRATEGY.length(w_dict)
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        EMPTY_DICT_STRATEGY.w_keys(w_dict)
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        EMPTY_DICT_STRATEGY.values(w_dict)
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        EMPTY_DICT_STRATEGY.items(w_dict)
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        EMPTY_DICT_STRATEGY.clear(w_dict);
    }

    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        EMPTY_DICT_STRATEGY.popitem(w_dict)
    }

    unsafe fn view_as_kwargs(
        &self,
        w_dict: PyObjectRef,
    ) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
        EMPTY_DICT_STRATEGY.view_as_kwargs(w_dict)
    }

    /// `kwargsdict.py EmptyKwargsDictStrategy` inherits the parent's
    /// `switch_to_object_strategy` (`dictmultiobject.py`).
    /// Delegate to `EMPTY_DICT_STRATEGY` so the null `dstorage` from
    /// `w_dict_new_kwargs` is replaced by ObjectDictStrategy's empty
    /// Vec — without this, the typed parent's relabel-only default
    /// would leave the null pointer in place.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        EMPTY_DICT_STRATEGY.switch_to_object_strategy(w_dict);
    }

    /// `kwargsdict.py EmptyKwargsDictStrategy` inherits empty-dict behavior: copying an
    /// empty kwargs dict returns a fresh empty kwargs dict on the
    /// EmptyKwargsDictStrategy so the first unicode setitem still
    /// promotes directly to KwargsDictStrategy (skipping the regular
    /// EmptyDictStrategy intermediate that goes through
    /// UnicodeDictStrategy).
    unsafe fn copy(&self, _w_dict: PyObjectRef) -> PyObjectRef {
        crate::dictmultiobject::w_dict_new_kwargs()
    }
}

crate::register_dyn_impl_vtable!(DYN_IMPL_EmptyDictStrategy, EmptyDictStrategy, DictStrategy);
impl DictStrategy for EmptyDictStrategy {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::Empty
    }

    fn get_empty_storage(&self) -> GCREF {
        // `erased(None)` — null is the only inhabitant of "empty
        // storage" before a switch installs a real backing.
        std::ptr::null_mut()
    }

    /// An empty dict holds `erased(None)` and no entries.  Walking it as
    /// `ObjectDictStorage` would dereference null, the same kwargs empty
    /// case (`EmptyKwargsDictStrategy.walk_gc_refs`).
    unsafe fn walk_gc_refs(
        &self,
        _w_dict: PyObjectRef,
        _visitor: &mut dyn FnMut(*mut PyObjectRef),
    ) {
    }

    /// `dictmultiobject.py EmptyDictStrategy.switch_to_object_strategy`
    /// — `storage = strategy.get_empty_storage(); w_dict.set_strategy(strategy);
    /// w_dict.dstorage = storage`.  Allocates a fresh Object-shape box
    /// so subclasses whose `dstorage` is null (`w_dict_new_kwargs`)
    /// don't end up with an OBJECT_DICT_STRATEGY label over a null
    /// pointer.  The field overwrite is a `setfield_gc`.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        install_empty_strategy(w_dict, &OBJECT_DICT_STRATEGY_REF);
    }

    unsafe fn getitem(&self, _w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef {
        // `dictmultiobject.py EmptyDictStrategy.getitem`:
        //   # in case the key is unhashable, try to hash it
        //   self.space.hash(w_key)
        //   # return None anyway
        //   return None
        // Force a hash dispatch so user-defined `__hash__` side effects
        // fire and unhashable types surface a TypeError up through the
        // hook trampoline.  Partial parity: `dict_eq_hook::try_hash_w`
        // currently swallows hash errors (returns `0` for unhashable),
        // so the TypeError surfaces only if the hook is installed and
        // routed through a future Result-aware variant.  Tracked in
        // MEMORY as the "hash hook error propagation" epic.
        let _ = crate::dict_eq_hook::try_hash_w(w_key);
        PY_NULL
    }

    // dictmultiobject.py setdefault
    unsafe fn setdefault(
        &self,
        mut w_dict: PyObjectRef,
        mut w_key: PyObjectRef,
        mut w_value: PyObjectRef,
    ) -> PyObjectRef {
        crate::with_roots!(w_dict, w_key, w_value => {
            self.switch_to_correct_strategy(w_dict, w_key)
        });
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
        w_value
    }

    // dictmultiobject.py pop
    unsafe fn pop(
        &self,
        _w_dict: PyObjectRef,
        _w_key: PyObjectRef,
        w_default: Option<PyObjectRef>,
    ) -> Result<PyObjectRef, DictPopError> {
        if let Some(d) = w_default {
            Ok(d)
        } else {
            Err(DictPopError)
        }
    }

    unsafe fn setitem(
        &self,
        mut w_dict: PyObjectRef,
        mut w_key: PyObjectRef,
        mut w_value: PyObjectRef,
    ) {
        // `dictmultiobject.py setitem`:
        //   self.switch_to_correct_strategy(w_dict, w_key)
        //   w_dict.setitem(w_key, w_value)
        crate::with_roots!(w_dict, w_key, w_value => {
            self.switch_to_correct_strategy(w_dict, w_key)
        });
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    unsafe fn setitem_str(
        &self,
        mut w_dict: PyObjectRef,
        key: *mut Utf8Str,
        mut w_value: PyObjectRef,
    ) {
        // `dictmultiobject.py setitem_str`:
        //   self.switch_to_unicode_strategy(w_dict)
        //   w_dict.setitem_str(key, w_value)
        // Unicode-strategy promotion is direct since the caller has
        // already chosen the str-keyed path.
        let key_bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key) };
        let key = unsafe { std::str::from_utf8_unchecked(key_bytes) };
        crate::with_roots!(w_dict, w_value => {
            install_empty_strategy(w_dict, &UNICODE_DICT_STRATEGY_REF)
        });
        crate::dictmultiobject::w_dict_setitem_str(w_dict, key, w_value);
    }

    unsafe fn delitem(&self, _w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        // `dictmultiobject.py EmptyDictStrategy.delitem`:
        //   # in case the key is unhashable, try to hash it
        //   self.space.hash(w_key)
        //   raise KeyError
        // Pyre returns false; the caller raises KeyError.  Same hash
        // dispatch ordering as `getitem` above so unhashable types
        // surface TypeError before KeyError once the hash hook
        // propagates errors (see "hash hook error propagation" epic).
        let _ = crate::dict_eq_hook::try_hash_w(w_key);
        false
    }

    unsafe fn length(&self, _w_dict: PyObjectRef) -> usize {
        0
    }

    unsafe fn w_keys(&self, _w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        Vec::new()
    }

    unsafe fn values(&self, _w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        Vec::new()
    }

    unsafe fn items(&self, _w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        Vec::new()
    }

    unsafe fn clear(&self, _w_dict: PyObjectRef) {
        // Already empty; no-op.
    }

    unsafe fn popitem(&self, _w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        // `dictmultiobject.py popitem` — KeyError.
        None
    }

    /// `dictmultiobject.py EmptyDictStrategy.view_as_kwargs`:
    /// ```python
    /// def view_as_kwargs(self, w_dict):
    ///     return ([], [])
    /// ```
    /// Empty kwarg fast path succeeds with zero entries.
    unsafe fn view_as_kwargs(
        &self,
        _w_dict: PyObjectRef,
    ) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
        (Some(Vec::new()), Some(Vec::new()))
    }

    /// Copying an empty dict yields a fresh empty dict on the same
    /// `EmptyDictStrategy` so the first setitem still triggers
    /// `switch_to_correct_strategy` per `dictmultiobject.py`.
    unsafe fn copy(&self, _w_dict: PyObjectRef) -> PyObjectRef {
        crate::dictmultiobject::w_dict_new()
    }
}

/// `dictmultiobject.py ObjectDictStrategy`.
///
/// ```python
/// class ObjectDictStrategy(AbstractTypedStrategy, DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("object")
///
///     def wrap(self, unwrapped): return unwrapped
///     def unwrap(self, wrapped): return wrapped
///     def is_correct_type(self, w_obj): return True
///
///     def get_empty_storage(self):
///         new_dict = r_dict(self.space.eq_w, self.space.hash_w, force_non_null=True)
///         return self.erase(new_dict)
///
///     def w_keys(self, w_dict):
///         return self.space.newlist(self.unerase(w_dict.dstorage).keys())
///
///     def setitem_str(self, w_dict, s, w_value):
///         self.setitem(w_dict, self.space.newtext(s), w_value)
///
///     def switch_to_object_strategy(self, w_dict):
///         assert 0, "should be unreachable"
/// ```
///
/// The fallback "any-key" strategy: `is_correct_type` always returns
/// `True`, so any incoming key lands here.  Keys compare via
/// `space.eq_w` / `space.hash_w` — pyre's stub uses
/// [`crate::pyobject::dict_keys_equal`] from
/// [`crate::dictmultiobject`] for parity.
///
/// Skeleton implementation pending the `W_DictObject.dstorage`
/// migration; methods route through pyre's existing
/// `Vec<(PyObjectRef, PyObjectRef)>` so the strategy can be
/// switched on without disturbing call sites.
pub struct ObjectDictStrategy;

crate::register_dyn_impl_vtable!(
    DYN_IMPL_ObjectDictStrategy,
    ObjectDictStrategy,
    DictStrategy
);
impl DictStrategy for ObjectDictStrategy {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::Object
    }

    fn get_empty_storage(&self) -> GCREF {
        // `dictmultiobject.py get_empty_storage`: erased `r_dict(dict_keys_equal,
        // hash_w)` = `GcStruct("dicttable")` (rdict.py), a nursery object.
        // `kwargsdict.py get_empty_storage` already uses the young-nonmoving
        // box; a born-old table stays on `old_objects_pointing_to_young` and
        // keeps a discarded type's GetSets alive at every minor.
        crate::gc_storage::gc_alloc_young_storage_box(
            crate::dictmultiobject::object_dict_storage_new(),
            crate::dictmultiobject::object_dict_storage_gc_type_id(),
        ) as GCREF
    }

    /// `dictmultiobject.py getitem` — `self.unerase
    /// (w_dict.dstorage).get(w_key)`. Body in
    /// `w_dict_lookup_object_strategy` to avoid recursing through
    /// `w_dict_lookup`.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef {
        crate::dictmultiobject::w_dict_lookup_object_strategy(w_dict, w_key).unwrap_or(PY_NULL)
    }

    /// `dictmultiobject.py ObjectDictStrategy.getitem_str` —
    /// upstream just delegates to `getitem` after wrapping the key.
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str) -> PyObjectRef {
        self.getitem_str_hashed(w_dict, key, 0)
    }

    unsafe fn getitem_str_hashed(
        &self,
        w_dict: PyObjectRef,
        key: *mut Utf8Str,
        hash: i64,
    ) -> PyObjectRef {
        crate::dictmultiobject::w_dict_getitem_str_object_strategy_hashed(w_dict, key, hash)
            .unwrap_or(PY_NULL)
    }

    /// `dictmultiobject.py setitem_str` — `ObjectDictStrategy`
    /// overrides to wrap the str then dispatch to `setitem`.  The
    /// default-trait setitem_str does the same path; keep parity
    /// override here so reverse readers can match.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str, w_value: PyObjectRef) {
        let w_key = crate::unicodeobject::w_str_from_storage(key);
        self.setitem(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py setitem` — `self.unerase
    /// (w_dict.dstorage)[w_key] = w_value`; body in
    /// `w_dict_store_object_strategy` to avoid recursing through
    /// `w_dict_store`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        crate::dictmultiobject::w_dict_store_object_strategy(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py delitem` — `del self.unerase
    /// (w_dict.dstorage)[w_key]`; body in `w_dict_delitem_object_strategy` to avoid
    /// recursing through `w_dict_delitem`.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        crate::dictmultiobject::w_dict_delitem_object_strategy(w_dict, w_key)
    }

    /// `dictmultiobject.py length` — `len(self.unerase
    /// (w_dict.dstorage))`; body in `w_dict_length_object_strategy` to
    /// avoid recursing through `w_dict_len`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        crate::dictmultiobject::w_dict_length_object_strategy(w_dict)
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items(w_dict);
        let mut keys = Vec::with_capacity(items.len());
        for (k, _) in items {
            keys.push(k);
        }
        keys
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items(w_dict);
        let mut values = Vec::with_capacity(items.len());
        for (_, v) in items {
            values.push(v);
        }
        values
    }

    /// `dictmultiobject.py items` — `self.unerase
    /// (w_dict.dstorage).iteritems()` materialised.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_items_object_strategy(w_dict)
    }

    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_nth_item_object_strategy(w_dict, index)
    }

    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_object_storage(w_dict).next_valid_slot(from)
    }

    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_object_storage(w_dict).prev_valid_slot(before)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.popitem`.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let dict = &mut *(w_dict as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        let (key, value) = entries.pop()?;
        dict.keys_version = dict.keys_version.wrapping_add(1);
        Some((key.obj, value))
    }

    /// `dictmultiobject.py clear` — `self.unerase
    /// (w_dict.dstorage).clear()`. Direct dstorage truncation +
    /// JIT length-cache reset.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_clear_object_strategy(w_dict);
    }

    /// `dictmultiobject.py AbstractTypedStrategy.copy` —
    /// `W_DictObject(space, self, self.erase(dstorage.copy()))`.
    /// Proxy-attached W_DictObjects bypass this override in
    /// `w_dict_copy` so str-key entries that live only in the proxy
    /// survive.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        crate::dictmultiobject::w_dict_copy_object_strategy(w_dict, &OBJECT_DICT_STRATEGY_REF)
    }
}

/// `dictmultiobject.py BytesDictStrategy`.
///
/// ```python
/// class BytesDictStrategy(AbstractTypedStrategy, DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("bytes")
///
///     def wrap(self, unwrapped):
///         return self.space.newbytes(unwrapped)
///     def unwrap(self, wrapped):
///         return self.space.bytes_w(wrapped)
///     def is_correct_type(self, w_obj):
///         return space.is_w(space.type(w_obj), space.w_bytes)
///
///     def get_empty_storage(self):
///         res = {}
///         mark_dict_non_null(res)
///         return self.erase(res)
///
///     def _never_equal_to(self, w_lookup_type):
///         return _never_equal_to_string(self.space, w_lookup_type)
///
///     def listview_bytes(self, w_dict):
///         return self.unerase(w_dict.dstorage).keys()
///
///     def w_keys(self, w_dict):
///         return self.space.newlist_bytes(self.listview_bytes(w_dict))
///
///     def wrapkey(space, key):
///         return space.newbytes(key)
/// ```
///
/// Bytes-keyed dict storage — `is_correct_type` returns true only for
/// W_BytesObject keys; mixed keys force `switch_to_object_strategy`
/// per `dictmultiobject.py`.  Native `RDict<BytesKey, PyObjectRef>`
/// backing — `unerase` is the block, `wrap` shares it.
pub struct BytesDictStrategy;

crate::register_dyn_impl_vtable!(DYN_IMPL_BytesDictStrategy, BytesDictStrategy, DictStrategy);
impl DictStrategy for BytesDictStrategy {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::Bytes
    }

    /// `dictmultiobject.py AbstractTypedStrategy.switch_to_object_strategy`
    /// instantiation for BytesDictStrategy — `wrap` shares the unwrapped
    /// `str` (`w_bytes_from_block`). Walks the typed `RDict<BytesKey, _>`,
    /// rebuilds `RDict<ObjectKey, _>`, drops the typed box.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_switch_bytes_to_object_strategy(w_dict);
    }

    /// `dictmultiobject.py get_empty_storage` — erased `{}`
    /// with `mark_dict_non_null` hint.  Pyre stores the typed map as
    /// `RDict<BytesKey, PyObjectRef>`: a hash bucket for O(1) lookup
    /// that also preserves insertion order.  GC-managed box
    /// (`setfield_gc` on reassign). The key is the block `unwrap` reads
    /// off the `bytes` object.
    fn get_empty_storage(&self) -> GCREF {
        // `dictmultiobject.py get_empty_storage`: erased `{}` is a nursery
        // `dicttable`. `ll_dict_grow`'s `d.entries = newitems` write barrier
        // can remember a young `DICTENTRYARRAY` only from a MiniMark header.
        crate::gc_storage::gc_alloc_young_storage_box(
            crate::dictmultiobject::BytesDictStorage::new(),
            crate::dictmultiobject::bytes_dict_storage_gc_type_id(),
        ) as GCREF
    }

    /// `dictmultiobject.py AbstractTypedStrategy.getitem`.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef {
        if crate::is_bytes(w_key) {
            return crate::dictmultiobject::w_dict_lookup_bytes_strategy(w_dict, w_key)
                .unwrap_or(PY_NULL);
        }
        // `:1099-1100 _never_equal_to(space.type(w_key))` —
        // `_never_equal_to_string` (`:21-31`) for str-keyed strategies.
        if crate::dictmultiobject::_never_equal_to_string(w_key) {
            return PY_NULL;
        }
        // `:1101-1103` switch + re-dispatch.
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key).unwrap_or(PY_NULL)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.setitem`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if crate::is_bytes(w_key) {
            crate::dictmultiobject::w_dict_store_bytes_strategy(w_dict, w_key, w_value);
            return;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py AbstractTypedStrategy.setitem_str`
    /// — BytesDictStrategy promotes to object on str setitem_str
    /// because str ≠ bytes.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str, w_value: PyObjectRef) {
        let key_bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key) };
        let key = unsafe { std::str::from_utf8_unchecked(key_bytes) };
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_setitem_str(w_dict, key, w_value);
    }

    /// `dictmultiobject.py AbstractTypedStrategy.delitem`.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if crate::is_bytes(w_key) {
            return crate::dictmultiobject::w_dict_delitem_bytes_strategy(w_dict, w_key);
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        crate::dictmultiobject::w_dict_length_bytes_strategy(w_dict)
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items_bytes_strategy(w_dict);
        let mut keys = Vec::with_capacity(items.len());
        for (k, _) in items {
            keys.push(k);
        }
        keys
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items_bytes_strategy(w_dict);
        let mut values = Vec::with_capacity(items.len());
        for (_, v) in items {
            values.push(v);
        }
        values
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_items_bytes_strategy(w_dict)
    }

    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_nth_item_bytes_strategy(w_dict, index)
    }

    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_bytes_storage(w_dict).next_valid_slot(from)
    }

    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_bytes_storage(w_dict).prev_valid_slot(before)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.popitem`.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let (key, value) = {
            let dict = &mut *(w_dict as *mut W_DictObject);
            let entries = &mut *(dict.dstorage as *mut BytesDictStorage);
            let (key, value) = entries.pop()?;
            dict.keys_version = dict.keys_version.wrapping_add(1);
            (key, value)
        };

        let _roots = crate::gc_roots::push_roots();
        let value_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value);
        let w_key = crate::w_bytes_from_block(key.0);
        let value = crate::gc_roots::shadow_stack_get(value_slot);
        Some((w_key, value))
    }

    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> PyObjectRef {
        lock_dict_refs!(_dict_guard, w_dict);
        match crate::dictmultiobject::w_dict_bytes_storage(w_dict).get_slot(index) {
            Some((_, &value)) => value,
            None => PY_NULL,
        }
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_clear_bytes_strategy(w_dict);
    }

    /// `dictmultiobject.py listview_bytes` — `self.unerase
    /// (w_dict.dstorage).keys()`.  Copies each block's characters out;
    /// the stored key stays the block `unerase` holds.
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
    )]
    fn listview_bytes(&self, w_dict: PyObjectRef) -> Option<Vec<Vec<u8>>> {
        let entries = unsafe { crate::dictmultiobject::w_dict_bytes_storage(w_dict) };
        Some(entries.keys().map(|key| key.as_bytes().to_vec()).collect())
    }

    /// `BytesDictStrategy` storage is `Dict[str, W_Root]`. `unerase`
    /// yields the block the key `str` is; both that block and the value
    /// are GC refs. The block does not move, and the visit keeps it alive.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let entries = crate::dictmultiobject::w_dict_bytes_storage_mut(w_dict);
        for (key, value) in entries.iter_mut_for_trace() {
            let key_ptr = key as *const crate::dictmultiobject::BytesKey
                as *mut crate::dictmultiobject::BytesKey;
            visitor(std::ptr::addr_of_mut!((*key_ptr).0) as *mut PyObjectRef);
            visitor(value as *mut PyObjectRef);
        }
        visitor(entries.entries_slot() as *mut PyObjectRef);
        entries.visit_indexes(&mut |slot| visitor(slot as *mut PyObjectRef));
    }

    /// `dictmultiobject.py AbstractTypedStrategy.copy` — clone
    /// the typed `RDict<BytesKey, PyObjectRef>` backing and wrap
    /// with the same BytesDictStrategy. Cloning copies the block
    /// pointer (`wrap` shares the `str`).
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let dict = &*(w_dict as *const crate::dictmultiobject::W_DictObject);
        let storage = &*(dict.dstorage as *const crate::dictmultiobject::BytesDictStorage);
        let new_storage =
            storage.clone_to_young_box(crate::dictmultiobject::bytes_dict_storage_gc_type_id());
        crate::dictmultiobject::w_dict_new_with(&BYTES_DICT_STRATEGY_REF, new_storage as GCREF)
    }
}

/// `dictmultiobject.py UnicodeDictStrategy`.
///
/// ```python
/// class UnicodeDictStrategy(AbstractTypedStrategy, DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("unicode")
///
///     def wrap(self, unwrapped):
///         return unwrapped
///     def unwrap(self, wrapped):
///         assert type(wrapped) is self.space.UnicodeObjectCls
///         return wrapped
///     def is_correct_type(self, w_obj):
///         return type(w_obj) is space.UnicodeObjectCls
///
///     def get_empty_storage(self):
///         res = create_empty_unicode_key_dict(self.space)
///         return self.erase(res)
///
///     def setitem_str(self, w_dict, key, w_value):
///         self.setitem(w_dict, self.space.newtext(key), w_value)
///
///     def getitem_str(self, w_dict, key):
///         assert key is not None
///         return self.getitem(w_dict, self.space.newtext(key))
///
///     def wrapkey(space, key):
///         return key
/// ```
///
/// Unicode-keyed dict storage (Py3's str).  Unlike BytesDictStrategy,
/// `wrap`/`unwrap` are identity functions because keys are already
/// PyObjectRef.  Storage stays on the unified
/// `Vec<(PyObjectRef, PyObjectRef)>` shape because PyPy's
/// `r_dict(unicode_eq, unicode_hash)` (`:1280-1304`) stores
/// W_UnicodeObject keys directly — the unified Vec layout is
/// structurally identical, no native typed migration needed for
/// parity.
pub struct UnicodeDictStrategy;

crate::register_dyn_impl_vtable!(
    DYN_IMPL_UnicodeDictStrategy,
    UnicodeDictStrategy,
    DictStrategy
);
impl DictStrategy for UnicodeDictStrategy {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::Unicode
    }

    fn get_empty_storage(&self) -> GCREF {
        // `dictmultiobject.py create_empty_unicode_key_dict`
        // returns an empty `r_dict(unicode_eq, unicode_hash)`, a nursery
        // `dicttable`. Type namespaces (`type(name, (), {})`) land here
        // on the first str store; a born-old box promoted the type at
        // every minor through the remembered set.
        crate::gc_storage::gc_alloc_young_storage_box(
            crate::dictmultiobject::object_dict_storage_new(),
            crate::dictmultiobject::object_dict_storage_gc_type_id(),
        ) as GCREF
    }

    /// `dictmultiobject.py AbstractTypedStrategy.getitem`.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef {
        if crate::is_exact_type(w_key, &crate::STR_TYPE) {
            return crate::dictmultiobject::w_dict_lookup_object_strategy(w_dict, w_key)
                .unwrap_or(PY_NULL);
        }
        if crate::dictmultiobject::_never_equal_to_string(w_key) {
            return PY_NULL;
        }
        crate::dictmultiobject::w_dict_set_strategy(w_dict, &OBJECT_DICT_STRATEGY_REF);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key).unwrap_or(PY_NULL)
    }

    /// `dictmultiobject.py setitem_str` override — wraps the
    /// key then dispatches to `setitem`.  UnicodeDictStrategy keeps
    /// str keys on the fast path; no promotion.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str, w_value: PyObjectRef) {
        self.setitem_str_hashed(w_dict, key, 0, w_value);
    }

    unsafe fn setitem_str_hashed(
        &self,
        w_dict: PyObjectRef,
        key: *mut Utf8Str,
        hash: i64,
        w_value: PyObjectRef,
    ) {
        // This is the one `setitem_str` that both collects and stores itself:
        // the membership probe can run a str-subclass `__eq__` (and its no-hook
        // arm wraps the key), and a miss mints the persistent key.  A dict
        // header moves, and these parameters are by-value copies of the caller's
        // words, so the root bracket the entry point holds does not update them
        // — publish the receiver and the value here and read both slots back
        // after the probe and after the mint.
        let roots = crate::gc_roots::push_roots();
        let dict_slot = roots.publish(&[w_dict, w_value]);
        let value_slot = dict_slot + 1;
        let found = dict_entries_index_of_str(w_dict_object_storage(w_dict), key, hash);
        match found {
            Some(idx) => {
                let w_value = roots.get(value_slot);
                let entries = w_dict_object_storage_mut(roots.get(dict_slot));
                *entries.get_slot_mut(idx).unwrap().1 = w_value;
            }
            None => {
                // Mint before reading either slot: call arguments evaluate left
                // to right, so a slot read written inline with the mint would be
                // taken before the collection that invalidates it.
                let stored_key = object_key_for_new_str(key, hash);
                let w_value = roots.get(value_slot);
                let dict = &mut *(roots.get(dict_slot) as *mut W_DictObject);
                let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
                dict_entries_insert_hashed(entries, stored_key.hash, stored_key.obj, w_value);
                dict.keys_version = dict.keys_version.wrapping_add(1);
            }
        };
        dict_write_barrier(roots.get(dict_slot));
    }

    /// `dictmultiobject.py getitem_str` override.
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str) -> PyObjectRef {
        self.getitem_str_hashed(w_dict, key, 0)
    }

    unsafe fn getitem_str_hashed(
        &self,
        w_dict: PyObjectRef,
        key: *mut Utf8Str,
        hash: i64,
    ) -> PyObjectRef {
        crate::dictmultiobject::w_dict_getitem_str_object_strategy_hashed(w_dict, key, hash)
            .unwrap_or(PY_NULL)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.setitem`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if crate::is_exact_type(w_key, &crate::STR_TYPE) {
            crate::dictmultiobject::w_dict_store_object_strategy(w_dict, w_key, w_value);
            return;
        }
        crate::dictmultiobject::w_dict_set_strategy(w_dict, &OBJECT_DICT_STRATEGY_REF);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py AbstractTypedStrategy.delitem`.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if crate::is_exact_type(w_key, &crate::STR_TYPE) {
            return crate::dictmultiobject::w_dict_delitem_object_strategy(w_dict, w_key);
        }
        crate::dictmultiobject::w_dict_set_strategy(w_dict, &OBJECT_DICT_STRATEGY_REF);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        crate::dictmultiobject::w_dict_length_object_strategy(w_dict)
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items_object_strategy(w_dict);
        let mut keys = Vec::with_capacity(items.len());
        for (k, _) in items {
            keys.push(k);
        }
        keys
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items_object_strategy(w_dict);
        let mut values = Vec::with_capacity(items.len());
        for (_, v) in items {
            values.push(v);
        }
        values
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_items_object_strategy(w_dict)
    }

    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_nth_item_object_strategy(w_dict, index)
    }

    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_object_storage(w_dict).next_valid_slot(from)
    }

    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_object_storage(w_dict).prev_valid_slot(before)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.popitem`.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let dict = &mut *(w_dict as *mut W_DictObject);
        let entries = &mut *(dict.dstorage as *mut ObjectDictStorage);
        let (key, value) = entries.pop()?;
        dict.keys_version = dict.keys_version.wrapping_add(1);
        Some((key.obj, value))
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_clear_object_strategy(w_dict);
    }

    /// `dictmultiobject.py view_as_kwargs` override.
    ///
    /// ```python
    /// @jit.look_inside_iff(lambda self, w_dict: w_dict._unrolling_heuristic())
    /// def view_as_kwargs(self, w_dict):
    ///     d = self.unerase(w_dict.dstorage)
    ///     l = len(d)
    ///     keys, values = [None] * l, [None] * l
    ///     i = 0
    ///     for w_key, val in d.iteritems():
    ///         keys[i] = w_key
    ///         values[i] = val
    ///         i += 1
    ///     return keys, values
    /// ```
    ///
    /// All keys on a UnicodeDictStrategy dict are W_UnicodeObjects, so the
    /// parallel arrays go straight to `argument.py:109-119`'s kwargs
    /// fast path without re-checking types.
    unsafe fn view_as_kwargs(
        &self,
        w_dict: PyObjectRef,
    ) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
        let items = crate::dictmultiobject::w_dict_items_object_strategy(w_dict);
        let mut keys = Vec::with_capacity(items.len());
        let mut values = Vec::with_capacity(items.len());
        for (w_key, w_val) in items {
            keys.push(w_key);
            values.push(w_val);
        }
        (Some(keys), Some(values))
    }

    /// `dictmultiobject.py AbstractTypedStrategy.copy` — clone
    /// the IndexMap backing and wrap with the same UnicodeDictStrategy
    /// (shares Object's `IndexMap<ObjectKey, _>` shape — str fast-path
    /// helpers route through the Object backing per
    /// `dictmultiobject.py`).  Proxy-attached W_DictObjects
    /// route through `w_dict_copy`'s union-walk fallback.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        crate::dictmultiobject::w_dict_copy_object_strategy(w_dict, &UNICODE_DICT_STRATEGY_REF)
    }
}

/// `dictmultiobject.py-... IntDictStrategy`.
///
/// ```python
/// class IntDictStrategy(AbstractTypedStrategy, DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("int")
///
///     def wrap(self, unwrapped):
///         return self.space.newint(unwrapped)
///     def unwrap(self, wrapped):
///         from pypy.objspace.std.listobject import plain_int_w
///         return plain_int_w(self.space, wrapped)
///     def is_correct_type(self, w_obj):
///         space = self.space
///         return space.type(w_obj).is_plain_int1()
/// ```
///
/// Int-keyed dict storage.  `is_correct_type` accepts plain
/// `W_IntObject` (not bool, which has its own correctness check at
/// the `listobject.is_plain_int1` predicate per
/// `pypy/objspace/std/listobject.py`).  Native
/// `Vec<(i64, PyObjectRef)>` backing (Slice D3) per
/// `dictmultiobject.py`; `wrap = newint`, `unwrap =
/// plain_int_w`.
pub struct IntDictStrategy;

impl IntDictStrategy {
    /// `dictmultiobject.py IntDictStrategy.is_correct_type`
    /// — `space.type(w_obj).is_plain_int1()`.  Plain int (not bool)
    /// per `listobject.py:is_plain_int1`.
    #[inline]
    fn is_correct_type(w_key: PyObjectRef) -> bool {
        unsafe { crate::listobject::is_plain_int1(w_key) }
    }

    /// `dictmultiobject.py _never_equal_to` for int — never
    /// equal to None / bytes / unicode lookup types.  Pyre's bytes/str
    /// are distinct types from int so the cheap pre-check is the same
    /// as `_never_equal_to_string` minus the bool wrinkle (bool == int
    /// in equality, but is_correct_type fences bool out).
    #[inline]
    unsafe fn never_equal_to(w_key: PyObjectRef) -> bool {
        // `space.is_w(w_lookup_type, space.w_NoneType)`:
        if w_key == crate::w_none() {
            return true;
        }
        // bytes / unicode never equal int.
        crate::is_bytes(w_key) || crate::is_str(w_key)
    }
}

crate::register_dyn_impl_vtable!(DYN_IMPL_IntDictStrategy, IntDictStrategy, DictStrategy);
impl DictStrategy for IntDictStrategy {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::Int
    }

    /// `dictmultiobject.py AbstractTypedStrategy.switch_to_object_strategy`:
    /// wraps each i64 key via `wrap = newint` (`:1342`), produces a
    /// fresh `IndexMap<ObjectKey, PyObjectRef>` and drops the old typed
    /// `IndexMap<i64, PyObjectRef>` box.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_switch_int_to_object_strategy(w_dict);
    }

    /// `dictmultiobject.py IntDictStrategy.get_empty_storage`
    /// — `erase({})` (typed `Dict[int, W_Root]` in RPython).  Pyre
    /// stores the typed map as `IndexMap<i64, PyObjectRef>`: a hash
    /// bucket for O(1) lookup that also preserves insertion order
    /// (CPython 3.7+ / PyPy3 dict semantics).  GC-managed box
    /// (`setfield_gc` on reassign).
    fn get_empty_storage(&self) -> GCREF {
        // `dictmultiobject.py IntDictStrategy.get_empty_storage`: erased `{}`
        // is a nursery `dicttable`.
        crate::gc_storage::gc_alloc_young_storage_box(
            crate::dictmultiobject::IntDictStorage::new(),
            crate::dictmultiobject::int_dict_storage_gc_type_id(),
        ) as GCREF
    }

    /// `dictmultiobject.py AbstractTypedStrategy.getitem`.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> PyObjectRef {
        if Self::is_correct_type(w_key) {
            return crate::dictmultiobject::w_dict_lookup_int_strategy(w_dict, w_key)
                .unwrap_or(PY_NULL);
        }
        if Self::never_equal_to(w_key) {
            return PY_NULL;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key).unwrap_or(PY_NULL)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.setitem`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if Self::is_correct_type(w_key) {
            crate::dictmultiobject::w_dict_store_int_strategy(w_dict, w_key, w_value);
            return;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py setitem_str` — int strategy
    /// promotes on str setitem_str.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: *mut Utf8Str, w_value: PyObjectRef) {
        let key_bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key) };
        let key = unsafe { std::str::from_utf8_unchecked(key_bytes) };
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_setitem_str(w_dict, key, w_value);
    }

    /// `dictmultiobject.py AbstractTypedStrategy.delitem`.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if Self::is_correct_type(w_key) {
            return crate::dictmultiobject::w_dict_delitem_int_strategy(w_dict, w_key);
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        crate::dictmultiobject::w_dict_length_int_strategy(w_dict)
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items_int_strategy(w_dict);
        let mut keys = Vec::with_capacity(items.len());
        for (k, _) in items {
            keys.push(k);
        }
        keys
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let items = crate::dictmultiobject::w_dict_items_int_strategy(w_dict);
        let mut values = Vec::with_capacity(items.len());
        for (_, v) in items {
            values.push(v);
        }
        values
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_items_int_strategy(w_dict)
    }

    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_dict_nth_item_int_strategy(w_dict, index)
    }

    unsafe fn next_slot(&self, w_dict: PyObjectRef, from: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_int_storage(w_dict).next_valid_slot(from)
    }

    unsafe fn prev_slot(&self, w_dict: PyObjectRef, before: usize) -> Option<usize> {
        crate::dictmultiobject::w_dict_int_storage(w_dict).prev_valid_slot(before)
    }

    /// `dictmultiobject.py AbstractTypedStrategy.popitem`.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let (key, value) = {
            let dict = &mut *(w_dict as *mut W_DictObject);
            let entries = &mut *(dict.dstorage as *mut IntDictStorage);
            let (key, value) = entries.pop()?;
            dict.keys_version = dict.keys_version.wrapping_add(1);
            (key, value)
        };

        let _roots = crate::gc_roots::push_roots();
        let value_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(value);
        let w_key = crate::w_int_new(key);
        let value = crate::gc_roots::shadow_stack_get(value_slot);
        Some((w_key, value))
    }

    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> PyObjectRef {
        lock_dict_refs!(_dict_guard, w_dict);
        match crate::dictmultiobject::w_dict_int_storage(w_dict).get_slot(index) {
            Some((_, &value)) => value,
            None => PY_NULL,
        }
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_dict_clear_int_strategy(w_dict);
    }

    /// `dictmultiobject.py IntDictStrategy.listview_int` —
    /// `self.unerase(w_dict.dstorage).keys()`.  Returns the native
    /// `Vec<i64>` of keys directly from the typed storage.
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
    )]
    fn listview_int(&self, w_dict: PyObjectRef) -> Option<Vec<i64>> {
        let entries = unsafe { crate::dictmultiobject::w_dict_int_storage(w_dict) };
        Some(entries.keys().copied().collect())
    }

    /// PyPy traces `Dict[int, W_Root]` only over the value side at
    /// translation time (`rerased.new_erasing_pair("integer")` +
    /// auto-generated GC walker); pyre's runtime dispatch mirrors
    /// that by skipping the i64 key half.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let entries = crate::dictmultiobject::w_dict_int_storage_mut(w_dict);
        for value in entries.values_mut_for_trace() {
            visitor(value as *mut PyObjectRef);
        }
        visitor(entries.entries_slot() as *mut PyObjectRef);
        entries.visit_indexes(&mut |slot| visitor(slot as *mut PyObjectRef));
    }

    /// `dictmultiobject.py AbstractTypedStrategy.copy` — clone
    /// the typed `IndexMap<i64, PyObjectRef>` backing and wrap with
    /// the same IntDictStrategy.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let dict = &*(w_dict as *const crate::dictmultiobject::W_DictObject);
        let storage = &*(dict.dstorage as *const crate::dictmultiobject::IntDictStorage);
        let new_storage =
            storage.clone_to_young_box(crate::dictmultiobject::int_dict_storage_gc_type_id());
        crate::dictmultiobject::w_dict_new_with(&INT_DICT_STRATEGY_REF, new_storage as GCREF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::{w_int_get_value, w_int_new};
    use crate::w_str_new;

    #[test]
    fn object_dict_table_takes_the_cached_python_hash_directly() {
        let mut hasher = ObjectKeyHasher::default();
        std::hash::Hasher::write_i64(&mut hasher, 0x1234_5678);
        assert_eq!(std::hash::Hasher::finish(&hasher), 0x1234_5678);
    }

    /// Install the single hash path on this test thread.  pyre-object
    /// cannot reach `pyre-interpreter`'s `space.hash_w`, so its own dict
    /// tests register the deterministic builtin structural hash; the
    /// `register_hash_w_hook` cell is thread-local and libtest spawns a
    /// fresh thread per `#[test]`, so each dict-building test installs it.
    fn install_test_hash_hook() {
        crate::dict_eq_hook::register_hash_w_hook(builtin_structural_hash);
        crate::dict_eq_hook::register_hash_str_hook(builtin_structural_str_hash);
    }

    #[test]
    fn empty_dict_walk_gc_refs_skips_null_dstorage() {
        unsafe {
            let dict = w_dict_new();
            assert_eq!(w_dict_strategy_name(dict), "EmptyDictStrategy");
            assert!((*(dict as *const W_DictObject)).dstorage.is_null());
            w_dict_walk_gc_refs(dict, &mut |_| {
                panic!("EmptyDictStrategy has no storage slots")
            });

            let kwargs = w_dict_new_kwargs();
            assert_eq!(w_dict_strategy_name(kwargs), "EmptyKwargsDictStrategy");
            assert!((*(kwargs as *const W_DictObject)).dstorage.is_null());
            w_dict_walk_gc_refs(kwargs, &mut |_| {
                panic!("EmptyKwargsDictStrategy has no storage slots")
            });
        }
    }

    #[test]
    fn strategy_class_name_tracks_regular_and_module_transitions() {
        install_test_hash_hook();
        let dict = w_dict_new();
        let module = w_module_dict_new();
        unsafe {
            assert_eq!(w_dict_strategy_name(dict), "EmptyDictStrategy");
            w_dict_setitem(dict, 1, crate::w_none());
            assert_eq!(w_dict_strategy_name(dict), "IntDictStrategy");

            assert_eq!(w_dict_strategy_name(module), "ModuleDictStrategy");
            w_dict_setitem(module, 1, crate::w_none());
            assert_eq!(w_dict_strategy_name(module), "ObjectDictStrategy");
        }
    }

    #[test]
    fn test_dict_int_key() {
        let dict = w_dict_new();
        unsafe {
            assert!(is_dict(dict));
            w_dict_setitem(dict, 1, w_int_new(100));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 100);
        }
    }

    #[test]
    fn test_dict_str_key() {
        install_test_hash_hook();
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem_str(dict, "hello", w_int_new(42));
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(dict, "hello").unwrap()),
                42
            );
            assert!(w_dict_getitem_str(dict, "world").is_none());
        }
    }

    #[test]
    fn getitem_str_slot_takes_one_word_str() {
        install_test_hash_hook();
        let d = w_dict_new_with(
            &OBJECT_DICT_STRATEGY_REF,
            OBJECT_DICT_STRATEGY.get_empty_storage(),
        );
        unsafe {
            w_dict_setitem_str(d, "abc", w_int_new(42));
            let strat: &dyn DictStrategy = w_dict_get_strategy(d);
            let hit = strat.getitem_str(d, crate::unicodeobject::prebuilt_str_storage("abc"));
            assert_eq!(w_int_get_value(hit), 42);
            assert_eq!(
                strat.getitem_str(d, crate::unicodeobject::prebuilt_str_storage("zz")),
                crate::PY_NULL
            );

            let w_key = crate::w_str_new("abc");
            let hit = strat.getitem(d, w_key);
            assert_eq!(w_int_get_value(hit), 42);
            let w_miss = crate::w_str_new("zz");
            assert_eq!(strat.getitem(d, w_miss), crate::PY_NULL);
        }
    }

    #[test]
    fn str_probe_helpers_take_one_word_str() {
        install_test_hash_hook();
        let d = w_dict_new_with(
            &OBJECT_DICT_STRATEGY_REF,
            OBJECT_DICT_STRATEGY.get_empty_storage(),
        );
        unsafe {
            w_dict_setitem_str(d, "abc", w_int_new(42));
            let entries = w_dict_object_storage(d);
            let hit = dict_entries_get_str(
                entries,
                crate::unicodeobject::prebuilt_str_storage("abc"),
                0,
            );
            assert_eq!(w_int_get_value(hit.unwrap()), 42);
            assert!(
                dict_entries_get_str(entries, crate::unicodeobject::prebuilt_str_storage("zz"), 0)
                    .is_none()
            );

            crate::dict_eq_hook::clear_hash_str_hook();
            let hit = dict_entries_get_str(
                entries,
                crate::unicodeobject::prebuilt_str_storage("abc"),
                0,
            );
            assert_eq!(w_int_get_value(hit.unwrap()), 42);
            assert!(
                dict_entries_get_str(entries, crate::unicodeobject::prebuilt_str_storage("zz"), 0)
                    .is_none()
            );
        }
    }

    #[test]
    fn test_w_dict_unicode_items_and_str_entries() {
        install_test_hash_hook();
        let dict = w_dict_new();
        unsafe {
            assert!(w_dict_is_empty_strategy(dict));

            w_dict_setitem_str(dict, "alpha", w_int_new(1));
            w_dict_setitem_str(dict, "beta", w_int_new(2));

            assert_eq!(
                w_dict_get_strategy(dict).strategy_kind(),
                StrategyKind::Unicode,
            );

            let items = w_dict_items(dict);
            assert_eq!(items.len(), 2);
            assert!(items.iter().any(|&(key, value)| {
                crate::w_str_get_wtf8(key) == "alpha" && w_int_get_value(value) == 1
            }));
            assert!(items.iter().any(|&(key, value)| {
                crate::w_str_get_wtf8(key) == "beta" && w_int_get_value(value) == 2
            }));

            let mut str_entries = w_dict_str_entries(dict);
            str_entries.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(str_entries.len(), 2);
            assert_eq!(str_entries[0].0, "alpha");
            assert_eq!(w_int_get_value(str_entries[0].1), 1);
            assert_eq!(str_entries[1].0, "beta");
            assert_eq!(w_int_get_value(str_entries[1].1), 2);
        }
    }

    #[test]
    fn test_w_dict_unicode_lookup_index_helpers() {
        install_test_hash_hook();
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem_str(dict, "alpha", w_int_new(1));
            w_dict_setitem_str(dict, "beta", w_int_new(2));

            let alpha = w_str_new("alpha");
            let beta = w_str_new("beta");
            let gamma = w_str_new("gamma");
            assert_eq!(
                w_dict_unicode_lookup_index(dict, alpha, w_dict_unicode_key_hash(alpha), 0),
                0,
            );
            assert_eq!(
                w_dict_unicode_lookup_index(dict, beta, w_dict_unicode_key_hash(beta), 0),
                1,
            );
            assert_eq!(
                w_dict_unicode_lookup_index(dict, gamma, w_dict_unicode_key_hash(gamma), 0),
                -1,
            );

            // The value read answers for the key that produced the index, not
            // for whatever now sits at it.
            w_dict_setitem_str(dict, "gamma", w_int_new(3));
            let beta_hash = w_dict_unicode_key_hash(beta);
            let beta_value = w_dict_getitem_str(dict, "beta").unwrap();
            assert_eq!(w_dict_unicode_lookup_index(dict, beta, beta_hash, 0), 1);
            assert_eq!(
                w_dict_unicode_value_at_checked(dict, 1, beta, beta_hash),
                beta_value,
            );
            // Slot 0 is IN RANGE and holds a live entry, so a bounds check
            // passes and would answer alpha's value for beta's index — only the
            // key re-validation catches it.
            assert!(w_dict_unicode_value_at_checked(dict, 0, beta, beta_hash).is_null());

            // `_ll_dict_del_entry` tombstones the slot it empties rather than
            // renumbering the ones after it, so beta keeps the index it was
            // found under and slot 0 becomes a hole that answers for nobody.
            w_dict_delitem_str(dict, "alpha");
            assert_eq!(w_dict_unicode_lookup_index(dict, beta, beta_hash, 0), 1);
            assert_eq!(
                w_dict_unicode_value_at_checked(dict, 1, beta, beta_hash),
                beta_value,
            );
            assert!(w_dict_unicode_value_at_checked(dict, 0, beta, beta_hash).is_null());
        }
    }

    #[test]
    fn test_dict_pyobj_key() {
        install_test_hash_hook();
        let dict = w_dict_new();
        let key = crate::w_str_new("test");
        unsafe {
            w_dict_store(dict, key, w_int_new(99));
            assert_eq!(w_int_get_value(w_dict_lookup(dict, key).unwrap()), 99);
        }
    }

    /// Force every key into one bucket so a store probes the existing
    /// entry by equality.
    unsafe fn constant_collision_hash(_obj: PyObjectRef) -> i64 {
        42
    }

    use std::sync::atomic::{AtomicU32, Ordering};
    static RAISING_EQ_CALLS: AtomicU32 = AtomicU32::new(0);

    /// An `eq_w` hook that always raises: it records the call, signals the
    /// error flag, and returns `false` (the Rust `Eq` callback cannot raise).
    unsafe fn raising_eq(a: PyObjectRef, _b: PyObjectRef) -> bool {
        RAISING_EQ_CALLS.fetch_add(1, Ordering::Relaxed);
        crate::dict_eq_hook::signal_eq_error(a);
        false
    }

    /// A user `__eq__` that raises during a colliding store must leave the
    /// dict unchanged and surface the first error, matching r_dict raising
    /// mid-probe without completing setitem.  Verifies: the store reports an
    /// error, the withheld store never inserts (k1 intact, k2 never stored),
    /// and `__eq__` runs exactly once (the scan short-circuits after the
    /// first raise).
    ///
    /// The keys are out-of-`int`-range longs so `key_equality_is_builtin` is
    /// false and the comparison actually routes through the `eq_w` hook —
    /// exact `str`/`int` keys, and exact tuples of those, settle on the
    /// builtin ladder and never reach a user `__eq__`.
    #[test]
    fn test_dict_store_raising_eq_leaves_dict_unchanged() {
        use crate::longobject::w_long_new;
        use majit_rlib::rbigint::RBigInt as BigInt;
        install_test_hash_hook();
        unsafe {
            crate::dict_eq_hook::register_hash_w_hook(constant_collision_hash);
            let dict = w_dict_new();
            let huge = BigInt::from(i64::MAX) * BigInt::from(2);
            let k1 = w_long_new(huge.clone());
            let k2 = w_long_new(huge + BigInt::from(1));
            // Seed one entry while the bucket is empty (no comparison runs).
            w_dict_store(dict, k1, w_int_new(1));

            RAISING_EQ_CALLS.store(0, Ordering::Relaxed);
            crate::dict_eq_hook::register_eq_w_hook(raising_eq);
            let result = w_dict_store_checked(dict, k2, w_int_new(2));
            crate::dict_eq_hook::clear_eq_w_hook();

            assert!(result.is_err());
            // The probe stopped at the first raising comparison.
            assert_eq!(RAISING_EQ_CALLS.load(Ordering::Relaxed), 1);
            // The dict is unchanged: k1 intact, k2 never inserted.
            assert_eq!(w_int_get_value(w_dict_lookup(dict, k1).unwrap()), 1);
            assert!(w_dict_lookup(dict, k2).is_none());
            // The error flag was consumed by the store, not left dangling.
            assert!(!crate::dict_eq_hook::take_eq_error());
        }
    }

    static TUPLE_LADDER_EQ_CALLS: AtomicU32 = AtomicU32::new(0);

    /// Counts `eq_w` and answers unequal, so a ladder hit is a successful
    /// lookup with a zero count, and a probe that broke is a miss plus a
    /// nonzero count.
    unsafe fn tuple_ladder_eq_false(_a: PyObjectRef, _b: PyObjectRef) -> bool {
        TUPLE_LADDER_EQ_CALLS.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Exact tuples of exact `str` compare on the callback-free ladder.
    /// Length 2 is `Cls_oo`; length 3 is the array-backed `W_TupleObject`.
    /// `Cls_ii` boxes on `w_tuple_getitem`, and a tuple subclass may override
    /// `__eq__`, so both break the probe and reach `eq_w`.
    #[test]
    fn tuple_of_exact_str_probe_uses_builtin_ladder() {
        install_test_hash_hook();
        unsafe {
            crate::dict_eq_hook::register_eq_w_hook(tuple_ladder_eq_false);
            let dict = w_dict_new();
            let stored = crate::w_tuple_new(vec![w_str_new("i"), w_str_new("marker")]);
            assert!(crate::is_specialised_tuple_oo(stored));
            w_dict_store(dict, stored, w_int_new(7));
            let lookup = crate::w_tuple_new(vec![w_str_new("i"), w_str_new("marker")]);
            TUPLE_LADDER_EQ_CALLS.store(0, Ordering::Relaxed);
            let found = w_dict_lookup(dict, lookup);
            assert_eq!(TUPLE_LADDER_EQ_CALLS.load(Ordering::Relaxed), 0);
            assert_eq!(w_int_get_value(found.unwrap()), 7);

            let stored3 = crate::w_tuple_new(vec![
                w_str_new("i"),
                w_str_new("marker"),
                w_str_new("odd_only"),
            ]);
            assert!(crate::py_type_check(stored3, &crate::TUPLE_TYPE));
            assert!(!crate::is_specialised_tuple_oo(stored3));
            w_dict_store(dict, stored3, w_int_new(8));
            let lookup3 = crate::w_tuple_new(vec![
                w_str_new("i"),
                w_str_new("marker"),
                w_str_new("odd_only"),
            ]);
            TUPLE_LADDER_EQ_CALLS.store(0, Ordering::Relaxed);
            let found3 = w_dict_lookup(dict, lookup3);
            assert_eq!(TUPLE_LADDER_EQ_CALLS.load(Ordering::Relaxed), 0);
            assert_eq!(w_int_get_value(found3.unwrap()), 8);

            // `_ii` hashes by the unboxed ints, so the second pair shares a
            // bucket.  The predicate still refuses it.
            let ii_stored = crate::w_tuple_new(vec![w_int_new(1), w_int_new(2)]);
            assert!(crate::is_specialised_tuple_ii(ii_stored));
            w_dict_store(dict, ii_stored, w_int_new(9));
            let ii_lookup = crate::w_tuple_new(vec![w_int_new(1), w_int_new(2)]);
            TUPLE_LADDER_EQ_CALLS.store(0, Ordering::Relaxed);
            assert!(w_dict_lookup(dict, ii_lookup).is_none());
            assert!(TUPLE_LADDER_EQ_CALLS.load(Ordering::Relaxed) >= 1);

            let w_class = crate::w_type_new("TupleSub", crate::PY_NULL, std::ptr::null_mut());
            let sub_stored = crate::w_tuple_subclass_new_array_backed(
                vec![w_str_new("i"), w_str_new("marker")],
                w_class,
            );
            w_dict_store(dict, sub_stored, w_int_new(11));
            let sub_lookup = crate::w_tuple_subclass_new_array_backed(
                vec![w_str_new("i"), w_str_new("marker")],
                w_class,
            );
            TUPLE_LADDER_EQ_CALLS.store(0, Ordering::Relaxed);
            assert!(w_dict_lookup(dict, sub_lookup).is_none());
            assert!(TUPLE_LADDER_EQ_CALLS.load(Ordering::Relaxed) >= 1);
            crate::dict_eq_hook::clear_eq_w_hook();
        }
    }

    #[test]
    fn test_dict_overwrite() {
        let dict = w_dict_new();
        unsafe {
            let start_version = w_dict_keys_version(dict);
            w_dict_setitem(dict, 1, w_int_new(10));
            let inserted_version = w_dict_keys_version(dict);
            assert_ne!(inserted_version, start_version);
            w_dict_setitem(dict, 1, w_int_new(20));
            assert_eq!(w_dict_len(dict), 1);
            assert_eq!(w_dict_keys_version(dict), inserted_version);
            assert!(w_dict_delitem(dict, w_int_new(1)));
            assert_ne!(w_dict_keys_version(dict), inserted_version);
        }
    }

    #[test]
    fn w_dict_gc_type_id_matches_descr() {
        assert_eq!(W_DICT_GC_TYPE_ID, 29);
        assert_eq!(
            <W_DictObject as crate::lltype::GcType>::type_id(),
            W_DICT_GC_TYPE_ID
        );
        assert_eq!(
            <W_DictObject as crate::lltype::GcType>::SIZE,
            W_DICT_OBJECT_SIZE
        );
    }

    #[test]
    fn module_dict_basic_roundtrip() {
        let md = w_module_dict_new();
        unsafe {
            assert!(is_module_dict(md));
            // PyPy surfaces both as `dict` (`dictmultiobject.py:67
            // allocate_instance(..., space.w_dict)`), so the
            // user-visible `is_dict` covers the module-dict layout.
            assert!(is_dict(md));
            w_module_dict_setitem_str(md, "x", w_int_new(1));
            w_module_dict_setitem_str(md, "y", w_int_new(2));
            assert_eq!(w_module_dict_length(md), 2);
            assert_eq!(
                w_int_get_value(
                    w_module_dict_getitem_str(md, crate::unicodeobject::prebuilt_str_storage("x"))
                        .unwrap()
                ),
                1,
            );
            assert_eq!(
                w_int_get_value(
                    w_module_dict_getitem_str(md, crate::unicodeobject::prebuilt_str_storage("y"))
                        .unwrap()
                ),
                2,
            );
            let removed = w_module_dict_delitem_str(md, "x").unwrap();
            assert_eq!(w_int_get_value(removed), 1);
            assert_eq!(w_module_dict_length(md), 1);
            assert!(
                w_module_dict_getitem_str(md, crate::unicodeobject::prebuilt_str_storage("x"))
                    .is_none()
            );
        }
    }

    #[test]
    fn module_dict_routes_through_w_dict_dispatch() {
        // Confirms `w_dict_*` public ops dispatch through the
        // `W_ModuleDictObject` branch when given a module dict, so
        // existing callers that take `PyObjectRef` continue to work
        // without knowing which layout backs the dict.
        let md = w_module_dict_new();
        unsafe {
            w_dict_setitem_str(md, "alpha", w_int_new(11));
            w_dict_setitem_str(md, "beta", w_int_new(22));
            assert_eq!(w_dict_len(md), 2);
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(md, "alpha").unwrap()),
                11,
            );
            assert_eq!(
                w_int_get_value(w_dict_lookup(md, w_str_new("beta")).unwrap()),
                22,
            );
            assert!(w_dict_delitem_str(md, "alpha"));
            assert!(w_dict_getitem_str(md, "alpha").is_none());
            assert_eq!(w_dict_len(md), 1);
        }
    }

    #[test]
    fn module_dict_strategy_switch_replaces_both_erased_slots() {
        install_test_hash_hook();
        let md = w_module_dict_new();
        unsafe {
            w_dict_setitem_str(md, "alpha", w_int_new(11));
            let raw = &*(md as *const W_ModuleDictObject);
            let module_holder = raw.mstrategy;
            let module_storage = raw.dstorage;

            w_dict_setitem(md, 7, w_int_new(22));

            let raw = &*(md as *const W_ModuleDictObject);
            assert!(w_module_dict_is_object_strategy(md));
            assert_ne!(raw.mstrategy, module_holder);
            assert_ne!(raw.dstorage, module_storage);
            assert!(std::ptr::eq(
                raw.mstrategy,
                &OBJECT_DICT_STRATEGY_REF as *const DictStrategyRef as *mut DictStrategyRef,
            ));
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(md, "alpha").unwrap()),
                11
            );
            assert_eq!(
                w_int_get_value(w_dict_lookup(md, w_int_new(7)).unwrap()),
                22,
            );
        }
    }

    #[test]
    fn mstrategy_store_invalidates_quasi_immutable_watchers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let md = w_module_dict_new();
        unsafe {
            let flag = Arc::new(AtomicBool::new(false));
            let token = crate::quasiimmut::test_loop_token(&flag);
            (*(md as *const W_ModuleDictObject))
                .mutate_mstrategy
                .get_current_qmut_instance()
                .register_loop_token(&token);
            assert!(!flag.load(Ordering::Acquire));
            w_module_dict_switch_to_object_strategy(md);
            assert!(flag.load(Ordering::Acquire));
        }
    }

    #[test]
    fn strategy_holder_kind_matches_the_singleton() {
        assert_eq!(OBJECT_DICT_STRATEGY_REF.kind, StrategyKind::Object);
        assert_eq!(EMPTY_DICT_STRATEGY_REF.kind, StrategyKind::Empty);
        assert_eq!(
            EMPTY_KWARGS_DICT_STRATEGY_REF.kind,
            StrategyKind::EmptyKwargs
        );
        assert_eq!(BYTES_DICT_STRATEGY_REF.kind, StrategyKind::Bytes);
        assert_eq!(UNICODE_DICT_STRATEGY_REF.kind, StrategyKind::Unicode);
        assert_eq!(INT_DICT_STRATEGY_REF.kind, StrategyKind::Int);
        assert_eq!(
            crate::identitydict::IDENTITY_DICT_STRATEGY_REF.kind,
            StrategyKind::Identity
        );
        assert_eq!(
            crate::kwargsdict::KWARGS_DICT_STRATEGY_REF.kind,
            StrategyKind::Kwargs
        );
        assert!(strategy_is(&OBJECT_DICT_STRATEGY_REF, StrategyKind::Object));
        assert!(!strategy_is(&OBJECT_DICT_STRATEGY_REF, StrategyKind::Empty));
    }

    #[test]
    fn module_dict_gc_type_id_matches_descr() {
        // Trip-wire mirroring the W_CELL / W_INT / W_FLOAT guards —
        // panics on startup if the constant here drifts from the id
        // that `pyre/pyre-jit/src/eval.rs` asserts at JitDriver init.
        assert_eq!(W_MODULE_DICT_GC_TYPE_ID, 48);
        assert_eq!(
            <W_ModuleDictObject as crate::lltype::GcType>::type_id(),
            W_MODULE_DICT_GC_TYPE_ID
        );
        assert_eq!(
            <W_ModuleDictObject as crate::lltype::GcType>::SIZE,
            W_MODULE_DICT_OBJECT_SIZE
        );
    }
}
