//! `pypy/objspace/std/kwargsdict.py` port — dict implementation
//! specialized for keyword argument dicts.
//!
//! Based on two parallel lists `(keys_w, values_w)` of `PyObjectRef`.
//! Optimized for the common `**kwargs` shape: a small number of
//! distinct string keys with O(n) linear-scan lookup that the JIT
//! constant-folds when the dict size and lookup key are both
//! constant.
//!
//! `EmptyKwargsDictStrategy` (`kwargsdict.py`) is selected by
//! `w_dict_new_kwargs`; function-call `**kwargs` collectors allocate through
//! that entry point and the first unicode store promotes directly to this
//! parallel-array strategy.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::dictmultiobject::DictStrategy;
use crate::pyobject::PyObjectRef;

/// `kwargsdict.py KwargsDictStrategy`.
///
/// ```python
/// class KwargsDictStrategy(DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("kwargsdict")
///
///     def get_empty_storage(self):
///         d = ([], [])
///         return self.erase(d)
///
///     def is_correct_type(self, w_obj):
///         space = self.space
///         return space.is_w(space.type(w_obj), space.w_text)
///
///     def setitem(self, w_dict, w_key, w_value):
///         if self.is_correct_type(w_key):
///             self.setitem_correct(w_dict, w_key, w_value)
///             return
///         else:
///             self.switch_to_object_strategy(w_dict)
///             w_dict.setitem(w_key, w_value)
/// ```
///
/// Two-list backing chosen because:
/// - Function-call sites always create small kwarg dicts.
/// - The JIT can fold the entire lookup loop when both size and key
///   are constant via `jit.look_inside_iff`.
/// - At size ≥ 16 entries (`kwargsdict.py`) the strategy
///   auto-promotes to `UnicodeDictStrategy` to avoid degenerate O(n).
pub struct KwargsDictStrategy;

/// `ll_getitem_fast` on the parallel `keys_w` / `values_w` lists
/// (`kwargsdict.py` `keys_w[i]` / `values_w[i]`).
fn kwargs_at(items: &[PyObjectRef], i: usize) -> PyObjectRef {
    // Scalar slice index, not `as_ptr().add`: the front-end already
    // lowers `core::slice::index::<Impl>::index` to `getitem`
    // (`rlist.py` `ll_getitem_fast`).  A raw-pointer walk types the
    // load as Integer and `__cast_instance_intrinsic` to PyObject fails.
    items[i]
}

/// `pypy/objspace/std/kwargsdict.py KwargsDictStrategy`
/// singleton — matches PyPy's `space.fromcache(KwargsDictStrategy)`.
pub static KWARGS_DICT_STRATEGY: KwargsDictStrategy = KwargsDictStrategy;

/// The [`crate::dictmultiobject::DictStrategyRef`] holder a dict's `dstrategy`
/// slot points at.
pub static KWARGS_DICT_STRATEGY_REF: crate::dictmultiobject::DictStrategyRef =
    crate::dictmultiobject::DictStrategyRef {
        kind: crate::dictmultiobject::StrategyKind::Kwargs,
        imp: &KWARGS_DICT_STRATEGY,
        owner: std::ptr::null_mut(),
    };

/// `KwargsDictStrategy` backing — erased `([], [])` parallel arrays
/// (`kwargsdict.py`). GC-managed storage box (mirrors the other
/// dict strategies; see `dictmultiobject::ObjectDictStorage`).
pub type KwargsDictStorage = (Vec<PyObjectRef>, Vec<PyObjectRef>);

/// Runtime-assigned GC type id for the [`KwargsDictStorage`] box.
static KWARGS_DICT_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`KwargsDictStorage`] box.
pub fn set_kwargs_dict_storage_gc_type_id(id: u32) {
    KWARGS_DICT_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`KwargsDictStorage`] box.
#[majit_macros::dont_look_inside]
pub fn kwargs_dict_storage_gc_type_id() -> u32 {
    KWARGS_DICT_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// `kwargsdict.py switch_to_object_strategy` — walk the parallel
/// arrays, rebuild `IndexMap<ObjectKey, PyObjectRef>`, retire the typed
/// parallel-array box.
///
/// Residualised (`@dont_look_inside`, `rlib/jit.py`) for the reason
/// `dictmultiobject::w_dict_switch_int_to_object_strategy` is: upstream traces
/// the same loop because every step is an RPython dict primitive the JIT
/// models, whereas this body is `IndexMap` end to end and the front end has no
/// lowering for it, so the last modellable point is the call itself.
///
/// # Safety
/// `w_dict` must be a valid `W_DictObject` on [`KWARGS_DICT_STRATEGY`].
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_switch_kwargs_to_object_strategy(w_dict: PyObjectRef) {
    // Same three hazards as `w_dict_switch_int_to_object_strategy`: the
    // receiver moves, `object_key_for` hashes (a collection point) while
    // the value is a bare local, and a pair copied into a stack-local map
    // would be pre-move by the next key's hash. Accumulate in root slots
    // and fill the box once every allocation is behind us.
    let roots = crate::gc_roots::push_roots();
    let dict_slot = roots.base();
    let w_dict = roots.pin_root(w_dict);
    let len = kwargs_storage(w_dict).0.len();
    let mut hashes = Vec::with_capacity(len);
    let pairs_base = dict_slot + 1;
    for i in 0..len {
        let object_key =
            crate::dictmultiobject::object_key_for(kwargs_storage(roots.get(dict_slot)).0[i]);
        // Take the value only now: the old table is traced through the
        // pinned dict, so re-reading the slot after this iteration's hash
        // yields the current word.
        let v = kwargs_storage(roots.get(dict_slot)).1[i];
        hashes.push(object_key.hash);
        roots.publish(&[object_key.obj, v]);
    }
    let new_storage = crate::gc_storage::gc_alloc_young_storage_box(
        crate::dictmultiobject::object_dict_storage_with_capacity(len),
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

/// `kwargsdict.py:62` size threshold past which the strategy
/// promotes itself to UnicodeDictStrategy to avoid O(n) lookup
/// degeneracy on too-large kwarg dicts.
const KWARGS_PROMOTE_THRESHOLD: usize = 16;

/// Typed accessor for `KwargsDictStrategy.unerase(w_dict.dstorage)` —
/// `kwargsdict.py` parallel-array shape.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` whose strategy is
/// [`KWARGS_DICT_STRATEGY`].
#[inline]
unsafe fn kwargs_storage<'a>(obj: PyObjectRef) -> &'a (Vec<PyObjectRef>, Vec<PyObjectRef>) {
    let dict = &*(obj as *const crate::dictmultiobject::W_DictObject);
    &*(dict.dstorage as *const (Vec<PyObjectRef>, Vec<PyObjectRef>))
}

#[inline]
unsafe fn kwargs_storage_mut<'a>(obj: PyObjectRef) -> &'a mut (Vec<PyObjectRef>, Vec<PyObjectRef>) {
    let dict = &mut *(obj as *mut crate::dictmultiobject::W_DictObject);
    &mut *(dict.dstorage as *mut (Vec<PyObjectRef>, Vec<PyObjectRef>))
}

impl KwargsDictStrategy {
    /// `kwargsdict.py is_correct_type` — `space.is_w
    /// (space.type(w_obj), space.w_text)`.  Plain str (Py3 unicode).
    #[inline]
    unsafe fn is_correct_type(w_key: PyObjectRef) -> bool {
        crate::is_exact_type(w_key, &crate::STR_TYPE)
    }

    /// `kwargsdict.py switch_to_unicode_strategy` —
    /// promote to UnicodeDictStrategy when size hits the threshold.
    /// PyPy walks the parallel arrays and re-inserts each entry via
    /// `w_dict.setitem`; pyre does the same so any non-ASCII keys
    /// further promote to ObjectDictStrategy.
    unsafe fn switch_to_unicode_strategy(&self, w_dict: PyObjectRef) {
        // kwargsdict.py: keys_w, values_w = self.unerase(w_dict.dstorage);
        // then replace dstorage and re-insert. RPython lists are their own
        // GC objects, so the unerase locals keep them alive; pin each
        // entry here before get_empty_storage / setitem can collect, while
        // the old tuple box is still reachable through the pinned dict.
        let roots = crate::gc_roots::push_roots();
        let dict_slot = roots.base();
        let _ = roots.pin_root(w_dict);
        let len = kwargs_storage(roots.get(dict_slot)).0.len();
        let pairs_base = dict_slot + 1;
        for i in 0..len {
            // Reload after each publish: incr_stack may grow and collect,
            // forwarding the tuple box through the pinned dict.
            let old = kwargs_storage(roots.get(dict_slot));
            roots.publish(&[kwargs_at(&old.0, i), kwargs_at(&old.1, i)]);
        }
        let fresh = crate::dictmultiobject::UNICODE_DICT_STRATEGY.get_empty_storage();
        let w_dict = roots.get(dict_slot);
        let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
        dict.dstorage = fresh;
        dict.dstrategy = &crate::dictmultiobject::UNICODE_DICT_STRATEGY_REF;
        crate::dictmultiobject::dict_write_barrier(w_dict);
        for i in 0..len {
            // `w_dict.setitem` (`kwargsdict.py switch_to_unicode_strategy`).
            // NB: this can turn the dict into an object strategy, if a
            // key is not ASCII.
            crate::dictmultiobject::w_dict_store(
                roots.get(dict_slot),
                roots.get(pairs_base + 2 * i),
                roots.get(pairs_base + 2 * i + 1),
            );
        }
    }
}

/// `kwargsdict.py KwargsDictStrategy._setitem_correct_indirection`
/// `@jit.look_inside_iff(lambda self, w_dict, w_key, w_value:
/// jit.isconstant(self.length(w_dict)) and jit.isconstant(w_key))`.
fn kwargs_setitem_correct_indirection_iff(
    w_dict: PyObjectRef,
    w_key: PyObjectRef,
    _w_value: PyObjectRef,
) -> bool {
    let length = unsafe { KWARGS_DICT_STRATEGY.length(w_dict) };
    majit_rlib::jit::isconstant(&length) && majit_rlib::jit::isconstant(&w_key)
}

/// `kwargsdict.py _setitem_correct_indirection` — linear scan, overwrite
/// on hit, else append or promote at 16.
#[majit_macros::look_inside_iff(kwargs_setitem_correct_indirection_iff)]
unsafe fn kwargs_setitem_correct_indirection(
    w_dict: PyObjectRef,
    w_key: PyObjectRef,
    w_value: PyObjectRef,
) {
    // kwargsdict.py: keys_w, values_w = self.unerase(w_dict.dstorage)
    let storage = kwargs_storage_mut(w_dict);
    crate::dictmultiobject::dict_write_barrier(w_dict);
    let keys_w = &mut storage.0;
    let values_w = &mut storage.1;
    for i in 0..keys_w.len() {
        // keys_w[i].eq_w(w_key) — W_UnicodeObject.eq_w; is_correct_type
        // already proved w_key is exact str, and this strategy only stores
        // those.
        if crate::w_str_eq_w(kwargs_at(keys_w, i), w_key) {
            // Direct element store in this body. A helper that
            // returns `&mut items[i]` leaves an opaque `index_mut`
            // whose destination is not a single deref here.
            values_w[i] = w_value;
            return;
        }
    }
    if keys_w.len() >= KWARGS_PROMOTE_THRESHOLD {
        KWARGS_DICT_STRATEGY.switch_to_unicode_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
        return;
    }
    keys_w.push(w_key);
    values_w.push(w_value);
    crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
}

/// `kwargsdict.py KwargsDictStrategy._getitem_correct_indirection`
/// `@jit.look_inside_iff(lambda self, w_dict, w_key:
/// jit.isconstant(self.length(w_dict)) and jit.isconstant(w_key))`.
fn kwargs_getitem_correct_indirection_iff(w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
    let length = unsafe { KWARGS_DICT_STRATEGY.length(w_dict) };
    majit_rlib::jit::isconstant(&length) && majit_rlib::jit::isconstant(&w_key)
}

/// `kwargsdict.py _getitem_correct_indirection` — linear scan of
/// `keys_w` / `values_w`.
#[majit_macros::look_inside_iff(kwargs_getitem_correct_indirection_iff)]
unsafe fn kwargs_getitem_correct_indirection(
    w_dict: PyObjectRef,
    w_key: PyObjectRef,
) -> Option<PyObjectRef> {
    // kwargsdict.py: keys_w, values_w = self.unerase(w_dict.dstorage)
    let (keys_w, values_w) = kwargs_storage(w_dict);
    for i in 0..keys_w.len() {
        // keys_w[i].eq_w(w_key) — W_UnicodeObject.eq_w.
        if crate::w_str_eq_w(kwargs_at(keys_w, i), w_key) {
            return Some(kwargs_at(values_w, i));
        }
    }
    None
}

impl DictStrategy for KwargsDictStrategy {
    fn strategy_kind(&self) -> crate::dictmultiobject::StrategyKind {
        crate::dictmultiobject::StrategyKind::Kwargs
    }

    /// `kwargsdict.py get_empty_storage` — erased `([], [])`, born young
    /// like the nursery tuple upstream erases.  GC-managed box (`setfield_gc`
    /// on reassign).
    fn get_empty_storage(&self) -> crate::gc_hook::GCREF {
        crate::gc_storage::gc_alloc_young_storage_box(
            KwargsDictStorage::default(),
            kwargs_dict_storage_gc_type_id(),
        ) as crate::gc_hook::GCREF
    }

    /// `kwargsdict.py switch_to_object_strategy` — walk
    /// parallel arrays, rebuild `IndexMap<ObjectKey, PyObjectRef>`,
    /// retire the typed parallel-array box.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        w_dict_switch_kwargs_to_object_strategy(w_dict);
    }

    /// `kwargsdict.py getitem` — `is_correct_type` →
    /// `getitem_correct` / `_getitem_correct_indirection`, else
    /// `_never_equal_to` short-circuit or promote.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if Self::is_correct_type(w_key) {
            return kwargs_getitem_correct_indirection(w_dict, w_key);
        }
        // `kwargsdict.py _never_equal_to` returns False — no
        // short-circuit; always promote and retry.
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key)
    }

    /// `kwargsdict.py setdefault` — keep an exact unicode key on the
    /// parallel-array strategy, otherwise switch first and re-dispatch on the
    /// new strategy.  The latter detail matters structurally: after the swap
    /// PyPy calls `w_dict.setdefault`, it does not keep invoking methods on
    /// the stale `KwargsDictStrategy` instance.
    unsafe fn setdefault(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
        w_default: PyObjectRef,
    ) -> PyObjectRef {
        if Self::is_correct_type(w_key) {
            if let Some(w_result) = self.getitem(w_dict, w_key) {
                return w_result;
            }
            self.setitem(w_dict, w_key, w_default);
            return w_default;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_get_strategy(w_dict).setdefault(w_dict, w_key, w_default)
    }

    /// `kwargsdict.py setitem` — `is_correct_type` → `setitem_correct` /
    /// `_setitem_correct_indirection`, else promote to object strategy.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if Self::is_correct_type(w_key) {
            kwargs_setitem_correct_indirection(w_dict, w_key, w_value);
            return;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `kwargsdict.py delitem` — switches to object strategy
    /// first (XXX comment: "could do better but is it worth it?").
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    /// `kwargsdict.py length`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        kwargs_storage(w_dict).0.len()
    }

    /// `kwargsdict.py w_keys` — returns a copy of `keys_w`.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        kwargs_storage(w_dict).0.clone()
    }

    /// `kwargsdict.py values`.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        kwargs_storage(w_dict).1.clone()
    }

    /// `kwargsdict.py KwargsDictStrategy.items` — the list comprehension
    /// reads both arrays at each index in range(len(keys_w)). Keep that
    /// index-driven flow rather than zip's shortest-input truncation.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        let mut items = Vec::new();
        for i in 0..keys_w.len() {
            items.push((kwargs_at(keys_w, i), kwargs_at(values_w, i)));
        }
        items
    }

    /// `create_iterator_classes(KwargsDictStrategy)` reads the two backing
    /// lists at the same cursor position.  Supplying the cursor operation
    /// directly avoids the trait fallback's `items().into_iter().nth(index)`,
    /// which rebuilt the complete kwargs list at every iterator step.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        if index < keys_w.len() && index < values_w.len() {
            Some((kwargs_at(keys_w, index), kwargs_at(values_w, index)))
        } else {
            None
        }
    }

    /// Value-iterator twin of [`Self::nth_item`], matching
    /// `kwargsdict.py itervalues`' direct values-list cursor.
    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> Option<PyObjectRef> {
        let values_w = &kwargs_storage(w_dict).1;
        if index < values_w.len() {
            Some(kwargs_at(values_w, index))
        } else {
            None
        }
    }

    /// `kwargsdict.py popitem` — pop from both arrays in lock-step.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let storage = kwargs_storage_mut(w_dict);
        let w_key = storage.0.pop()?;
        let w_value = storage.1.pop()?;
        crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
        Some((w_key, w_value))
    }

    /// `kwargsdict.py getiterreversed` — copy/reverse the key list in
    /// PyPy.  Pyre's iterator carrier consumes key/value pairs, so walk both
    /// parallel lists from the tail without first materialising `items()`.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        keys_w
            .iter()
            .copied()
            .zip(values_w.iter().copied())
            .rev()
            .collect()
    }

    /// `kwargsdict.py clear` — `w_dict.dstorage =
    /// self.get_empty_storage()`.  The field overwrite is a `setfield_gc`,
    /// barrier included: the new box is young; the unreachable old
    /// parallel-array box is reclaimed by the collector.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        // `get_empty_storage` can collect. Pin the dict, allocate, then
        // reload the forwarded address before writing `dstorage`.
        let _roots = crate::gc_roots::push_roots();
        let dict_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_dict);
        let w_dict = crate::gc_roots::shadow_stack_get(dict_slot);
        let nonempty = {
            let dict = &*(w_dict as *mut crate::dictmultiobject::W_DictObject);
            let storage = &*(dict.dstorage as *const (Vec<PyObjectRef>, Vec<PyObjectRef>));
            !storage.0.is_empty()
        };
        let fresh = self.get_empty_storage();
        let w_dict = crate::gc_roots::shadow_stack_get(dict_slot);
        let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
        if nonempty {
            dict.keys_version = dict.keys_version.wrapping_add(1);
        }
        dict.dstorage = fresh;
        crate::dictmultiobject::dict_write_barrier(w_dict);
    }

    /// `kwargsdict.py view_as_kwargs` — copy parallel arrays
    /// to non-resizable slices for the `**kwargs` fast unpack.
    unsafe fn view_as_kwargs(
        &self,
        w_dict: PyObjectRef,
    ) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        (Some(keys_w.clone()), Some(values_w.clone()))
    }

    /// `kwargsdict.py` traces both `keys_w` and `values_w` as
    /// `list[W_Root]` — every entry on both sides is PyObjectRef.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let storage = kwargs_storage_mut(w_dict);
        for k in storage.0.iter_mut() {
            visitor(k as *mut PyObjectRef);
        }
        for v in storage.1.iter_mut() {
            visitor(v as *mut PyObjectRef);
        }
    }

    /// `dictmultiobject.py AbstractTypedStrategy.copy` — clone
    /// the parallel `(keys_w, values_w)` arrays and wrap with the
    /// same KwargsDictStrategy.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let storage = kwargs_storage(w_dict);
        let new_storage = crate::gc_storage::gc_alloc_young_storage_box(
            storage.clone(),
            kwargs_dict_storage_gc_type_id(),
        );
        crate::dictmultiobject::w_dict_new_with(
            &KWARGS_DICT_STRATEGY_REF,
            new_storage as crate::gc_hook::GCREF,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_test_hash_hooks() {
        unsafe fn hash_object(obj: PyObjectRef) -> i64 {
            if crate::is_int(obj) {
                crate::w_int_get_value(obj)
            } else {
                0
            }
        }

        unsafe fn hash_str(_ptr: *const u8, _len: usize) -> i64 {
            0
        }

        crate::dict_eq_hook::register_hash_w_hook(hash_object);
        crate::dict_eq_hook::register_hash_str_hook(hash_str);
    }

    unsafe fn kwargs_with(entries: &[(&str, i64)]) -> PyObjectRef {
        let w_dict = crate::dictmultiobject::w_dict_new_kwargs();
        for &(key, value) in entries {
            crate::dictmultiobject::w_dict_setitem_str(w_dict, key, crate::w_int_new(value));
        }
        assert_eq!(
            crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
            crate::dictmultiobject::StrategyKind::Kwargs
        );
        w_dict
    }

    #[test]
    fn kwargs_items_copy_preserves_parallel_array_order_and_identity() {
        unsafe {
            let empty = kwargs_with(&[("temporary", 0)]);
            KWARGS_DICT_STRATEGY.clear(empty);
            assert!(KWARGS_DICT_STRATEGY.items(empty).is_empty());

            let w_dict = kwargs_with(&[("first", 1), ("second", 2)]);
            let mut items = KWARGS_DICT_STRATEGY.items(w_dict);
            let (keys_w, values_w) = kwargs_storage(w_dict);
            assert_eq!(items.len(), keys_w.len());
            for i in 0..keys_w.len() {
                assert_eq!(items[i], (kwargs_at(keys_w, i), kwargs_at(values_w, i)));
            }
            items.reverse();
            let fresh = KWARGS_DICT_STRATEGY.items(w_dict);
            assert_eq!(crate::w_str_get_wtf8(fresh[0].0), "first");
            assert_eq!(crate::w_int_get_value(fresh[0].1), 1);
            assert_eq!(crate::w_str_get_wtf8(fresh[1].0), "second");
            assert_eq!(crate::w_int_get_value(fresh[1].1), 2);
        }
    }

    #[test]
    fn kwargs_cursor_reads_parallel_arrays_without_materialising_items() {
        unsafe {
            let w_dict = kwargs_with(&[("a", 10), ("b", 20), ("c", 30)]);
            let strategy = crate::dictmultiobject::w_dict_get_strategy(w_dict);

            for (index, (key, value)) in [("a", 10), ("b", 20), ("c", 30)].into_iter().enumerate() {
                let (w_key, w_value) = strategy.nth_item(w_dict, index).unwrap();
                assert_eq!(crate::w_str_get_wtf8(w_key), key);
                assert_eq!(crate::w_int_get_value(w_value), value);
                assert_eq!(
                    crate::w_int_get_value(strategy.nth_value(w_dict, index).unwrap()),
                    value
                );
            }
            assert!(strategy.nth_item(w_dict, 3).is_none());
            assert!(strategy.nth_value(w_dict, 3).is_none());

            let reversed = strategy.getiterreversed(w_dict);
            let keys: Vec<_> = reversed
                .iter()
                .map(|&(key, _)| crate::w_str_get_wtf8(key).as_str().unwrap())
                .collect();
            assert_eq!(keys, ["c", "b", "a"]);
        }
    }

    #[test]
    fn kwargs_setitem_overwrites_an_existing_parallel_value() {
        unsafe {
            let w_dict = kwargs_with(&[("a", 1), ("b", 2)]);
            crate::dictmultiobject::w_dict_setitem_str(w_dict, "a", crate::w_int_new(9));
            let strategy = crate::dictmultiobject::w_dict_get_strategy(w_dict);
            assert_eq!(
                strategy.strategy_kind(),
                crate::dictmultiobject::StrategyKind::Kwargs
            );
            assert_eq!(strategy.length(w_dict), 2);
            let (w_key, w_value) = strategy.nth_item(w_dict, 0).unwrap();
            assert_eq!(crate::w_str_get_wtf8(w_key), "a");
            assert_eq!(crate::w_int_get_value(w_value), 9);
            assert_eq!(
                crate::w_int_get_value(strategy.nth_value(w_dict, 1).unwrap()),
                2
            );
        }
    }

    #[test]
    fn kwargs_setdefault_keeps_string_strategy_and_redispatches_other_keys() {
        unsafe {
            install_test_hash_hooks();
            let w_dict = kwargs_with(&[("a", 10)]);
            let strategy = crate::dictmultiobject::w_dict_get_strategy(w_dict);
            let w_a = crate::w_str_new("a");
            let existing = strategy.setdefault(w_dict, w_a, crate::w_int_new(99));
            assert_eq!(crate::w_int_get_value(existing), 10);

            let w_b = crate::w_str_new("b");
            let inserted = strategy.setdefault(w_dict, w_b, crate::w_int_new(20));
            assert_eq!(crate::w_int_get_value(inserted), 20);
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Kwargs
            );

            let w_int_key = crate::w_int_new(1);
            let inserted = crate::dictmultiobject::w_dict_setdefault_checked(
                w_dict,
                w_int_key,
                crate::w_int_new(30),
            )
            .unwrap();
            assert_eq!(crate::w_int_get_value(inserted), 30);
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Object
            );
        }
    }

    /// `newdict(kwargs=True)` plus `setitem` of a str subclass promotes to
    /// the object strategy and stores that object. An exact str stays on
    /// KwargsDictStrategy. Storing a subclass into a kwargs dict that
    /// already holds an exact str promotes and keeps both pointers.
    #[test]
    fn kwargs_str_subclass_promotes_to_object_and_keeps_both_pointers() {
        unsafe {
            install_test_hash_hooks();
            let w_class = crate::w_type_new("StrSub", crate::PY_NULL, std::ptr::null_mut());
            let subclass =
                crate::w_str_subclass_from_wtf8(rustpython_wtf8::Wtf8Buf::from("a"), w_class);
            let w_dict = crate::dictmultiobject::w_dict_new_kwargs();
            crate::dictmultiobject::w_dict_store(w_dict, subclass, crate::w_int_new(1));
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Object
            );
            let items = crate::dictmultiobject::w_dict_items(w_dict);
            assert_eq!(items.len(), 1);
            assert!(std::ptr::eq(items[0].0, subclass));
            assert_eq!(crate::w_int_get_value(items[0].1), 1);

            let exact = crate::dictmultiobject::w_dict_new_kwargs();
            let exact_key = crate::w_str_new("a");
            crate::dictmultiobject::w_dict_store(exact, exact_key, crate::w_int_new(1));
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(exact).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Kwargs
            );

            let mixed = crate::dictmultiobject::w_dict_new_kwargs();
            let mixed_exact = crate::w_str_new("a");
            crate::dictmultiobject::w_dict_store(mixed, mixed_exact, crate::w_int_new(1));
            let mixed_sub =
                crate::w_str_subclass_from_wtf8(rustpython_wtf8::Wtf8Buf::from("b"), w_class);
            crate::dictmultiobject::w_dict_store(mixed, mixed_sub, crate::w_int_new(2));
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(mixed).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Object
            );
            let items = crate::dictmultiobject::w_dict_items(mixed);
            assert_eq!(items.len(), 2);
            assert!(items.iter().any(|&(key, value)| {
                std::ptr::eq(key, mixed_exact) && crate::w_int_get_value(value) == 1
            }));
            assert!(items.iter().any(|&(key, value)| {
                std::ptr::eq(key, mixed_sub) && crate::w_int_get_value(value) == 2
            }));
        }
    }
}
