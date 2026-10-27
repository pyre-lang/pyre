//! Dict definitions — key / value carriers mirroring the
//! identity-based merge semantics of [`super::listdef`].
//!
//! RPython upstream: `rpython/annotator/dictdef.py` (117 LOC).
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Upstream `class DictKey(ListItem)` / `class DictValue(ListItem)`
//!   extend `ListItem`. Rust has no single inheritance, so the
//!   subclass-only fields (`custom_eq_hash`, `s_rdict_eqfn`,
//!   `s_rdict_hashfn`) live flattened on [`super::listdef::ListItem`]
//!   itself — see the TODO doc on that struct.
//!   `DictKey` / `DictValue` here become zero-sized namespaces for the
//!   subclass's `__init__` / `merge` associated functions, matching
//!   the upstream class shape at the call-site level. The `patch()`
//!   overrides (dictdef.py, 70-72) retarget the appropriate slot
//!   on the containing `DictDef` via
//!   [`super::listdef::ItemOwner::DictKey`] /
//!   [`super::listdef::ItemOwner::DictValue`] variants consulted by
//!   the shared `ListItem::merge` patch loop.
//!
//! * `DictKey.emulate_rdict_calls` (dictdef.py) dispatches
//!   `eq` + `hash` as `bookkeeper.emulate_pbc_call` pairs and
//!   validates return annotations against `SomeBool` / `SomeInteger`.

use std::cell::RefCell;
use std::rc::Rc;

use super::bookkeeper::{Bookkeeper, EmulatedPbcCallKey, PositionKey};
use super::listdef::{ItemOwner, ListItem, journal_listitem_mutation};
use super::model::{SomeBool, SomeInteger, SomeValue, UnionError, union};

/// RPython `class DictKey(ListItem)` (dictdef.py). Zero-sized
/// namespace for the subclass constructor + `merge` override. The
/// subclass-specific `custom_eq_hash` / `s_rdict_eqfn` /
/// `s_rdict_hashfn` fields live on [`ListItem`] itself (see that
/// struct's TODO doc).
#[derive(Debug)]
pub struct DictKey;

impl DictKey {
    /// RPython `DictKey.__init__(bookkeeper, s_value, is_r_dict=False)`
    /// (dictdef.py). Returns the underlying `ListItem` wrapped in
    /// a shared cell, ready for storage in `DictDef.dictkey`.
    #[expect(
        clippy::new_ret_no_self,
        reason = "This is the direct DictKey.__init__ namespace port from rpython/annotator/dictdef.py; Rust represents the inherited ListItem instance as Rc<RefCell<ListItem>>"
    )]
    pub fn new(
        bookkeeper: Option<Rc<Bookkeeper>>,
        s_value: SomeValue,
        is_r_dict: bool,
    ) -> Rc<RefCell<ListItem>> {
        let mut item = ListItem::new(bookkeeper, s_value);
        // upstream: `self.custom_eq_hash = is_r_dict` (dictdef.py).
        item.custom_eq_hash = is_r_dict;
        Rc::new(RefCell::new(item))
    }

    /// RPython `DictKey.merge(self, other)` (dictdef.py).
    ///
    /// Wraps [`ListItem::merge`] with the `custom_eq_hash` mixing-guard
    /// from upstream line 21-22 and propagates `update_rdict_annotations`
    /// from upstream line 24-27.
    pub fn merge(
        self_li: &Rc<RefCell<ListItem>>,
        other_li: &Rc<RefCell<ListItem>>,
    ) -> Result<Rc<RefCell<ListItem>>, UnionError> {
        // upstream: `if self is not other:`.
        if !Rc::ptr_eq(self_li, other_li) {
            // upstream: `assert self.custom_eq_hash == other.custom_eq_hash,
            //            "mixing plain dictionaries with r_dict()"`.
            let (self_ceh, other_ceh) = (
                self_li.borrow().custom_eq_hash,
                other_li.borrow().custom_eq_hash,
            );
            if self_ceh != other_ceh {
                return Err(UnionError {
                    lhs: Box::new(self_li.borrow().s_value.clone()),
                    rhs: Box::new(other_li.borrow().s_value.clone()),
                    msg: "mixing plain dictionaries with r_dict()".into(),
                });
            }
        }

        // Snapshot other's rdict annotations before ListItem::merge
        // potentially retargets the cell.
        let (other_eqfn, other_hashfn) = {
            let b = other_li.borrow();
            (b.s_rdict_eqfn.clone(), b.s_rdict_hashfn.clone())
        };

        let merged = ListItem::merge(self_li, other_li)?;

        // upstream: `if self.custom_eq_hash: self.update_rdict_annotations(
        //     other.s_rdict_eqfn, other.s_rdict_hashfn, other=other)`.
        if merged.borrow().custom_eq_hash {
            Self::update_rdict_annotations(&merged, other_eqfn, other_hashfn)?;
        }
        Ok(merged)
    }

    /// RPython `DictKey.update_rdict_annotations(s_eqfn, s_hashfn,
    /// other=None)` (dictdef.py).
    pub fn update_rdict_annotations(
        self_li: &Rc<RefCell<ListItem>>,
        s_eqfn: SomeValue,
        s_hashfn: SomeValue,
    ) -> Result<(), UnionError> {
        // upstream: `assert self.custom_eq_hash`.
        debug_assert!(self_li.borrow().custom_eq_hash);
        let (cur_eqfn, cur_hashfn) = {
            let b = self_li.borrow();
            (b.s_rdict_eqfn.clone(), b.s_rdict_hashfn.clone())
        };
        // upstream: `s_eqfn = union(s_eqfn, self.s_rdict_eqfn)`.
        let new_eqfn = union(&s_eqfn, &cur_eqfn)?;
        let new_hashfn = union(&s_hashfn, &cur_hashfn)?;
        // `ListItem.merge` on a ptr-equal cell returns before journaling.
        // This write still has to be undone with that failed scope.
        journal_listitem_mutation(self_li);
        {
            let mut b = self_li.borrow_mut();
            b.s_rdict_eqfn = new_eqfn;
            b.s_rdict_hashfn = new_hashfn;
        }
        // upstream: `self.emulate_rdict_calls(other=other)` (dictdef.py
        // tail). Performs two `bookkeeper.emulate_pbc_call` dispatches —
        // one for eq, one for hash — and validates the return annotations
        // against `s_Bool` / `SomeInteger` (dictdef.py), raising
        // `AnnotatorError` on mismatch.
        Self::emulate_rdict_calls(self_li)
    }

    /// RPython `DictKey.generalize(s_other_value)` (dictdef.py).
    ///
    /// Wraps [`ListItem::generalize`] with the rdict eq/hash
    /// re-validation that upstream runs when the key lattice widens
    /// under a `custom_eq_hash` (= r_dict) key cell. This is the
    /// method that `DictDef.generalize_key` must route through — the
    /// subclass `generalize` override is load-bearing (without it, r_
    /// dict eq/hash constraints drift silently as keys generalize).
    pub fn generalize(
        self_li: &Rc<RefCell<ListItem>>,
        s_other_value: &SomeValue,
    ) -> Result<bool, UnionError> {
        // upstream: `updated = ListItem.generalize(self, s_other_value)`.
        journal_listitem_mutation(self_li);
        let updated = {
            let mut b = self_li.borrow_mut();
            b.generalize(s_other_value)?
        };
        // upstream: `if updated and self.custom_eq_hash:
        //                self.emulate_rdict_calls()`.
        if updated && self_li.borrow().custom_eq_hash {
            Self::emulate_rdict_calls(self_li)?;
        }
        Ok(updated)
    }

    /// RPython `DictKey.emulate_rdict_calls(other=None)` (dictdef.py).
    ///
    fn emulate_rdict_calls(self_li: &Rc<RefCell<ListItem>>) -> Result<(), UnionError> {
        let (bookkeeper, s_key, s_eqfn, s_hashfn) = {
            let b = self_li.borrow();
            (
                b.bookkeeper.clone().ok_or_else(|| UnionError {
                    lhs: Box::new(b.s_value.clone()),
                    rhs: Box::new(b.s_value.clone()),
                    msg: "r_dict key has no bookkeeper".into(),
                })?,
                b.s_value.clone(),
                b.s_rdict_eqfn.clone(),
                b.s_rdict_hashfn.clone(),
            )
        };
        let item_id = Rc::as_ptr(self_li) as usize;
        let s_eq = bookkeeper
            .emulate_pbc_call(
                EmulatedPbcCallKey::RDictCall {
                    item_id,
                    role: "eq",
                },
                &s_eqfn,
                &[s_key.clone(), s_key.clone()],
                &[],
                None,
            )
            .map_err(|e| UnionError {
                lhs: Box::new(s_eqfn.clone()),
                rhs: Box::new(s_key.clone()),
                msg: e.msg.unwrap_or_else(|| "emulate_pbc_call failed".into()),
            })?;
        if !SomeValue::Bool(SomeBool::new()).contains(&s_eq) {
            return Err(UnionError {
                lhs: Box::new(s_eq),
                rhs: Box::new(s_key.clone()),
                msg: "the custom eq function of an r_dict must return a boolean".into(),
            });
        }

        let s_hash = bookkeeper
            .emulate_pbc_call(
                EmulatedPbcCallKey::RDictCall {
                    item_id,
                    role: "hash",
                },
                &s_hashfn,
                std::slice::from_ref(&s_key),
                &[],
                None,
            )
            .map_err(|e| UnionError {
                lhs: Box::new(s_hashfn.clone()),
                rhs: Box::new(s_key.clone()),
                msg: e.msg.unwrap_or_else(|| "emulate_pbc_call failed".into()),
            })?;
        if !SomeValue::Integer(SomeInteger::default()).contains(&s_hash) {
            return Err(UnionError {
                lhs: Box::new(s_hash),
                rhs: Box::new(s_key),
                msg: "the custom hash function of an r_dict must return an integer".into(),
            });
        }
        Ok(())
    }
}

/// RPython `class DictValue(ListItem)` (dictdef.py). Zero-sized
/// namespace for the subclass's `patch()` override; retargeting
/// lands through [`ItemOwner::DictValue`].
#[derive(Debug)]
pub struct DictValue;

impl DictValue {
    /// Upstream has no explicit `DictValue.__init__` — it inherits
    /// [`ListItem::__init__`] verbatim (dictdef.py). Keep the
    /// factory on the subclass namespace so `DictDef::new` mirrors the
    /// upstream `DictValue(bookkeeper, s_value)` call shape.
    #[expect(
        clippy::new_ret_no_self,
        reason = "This preserves the inherited DictValue/ListItem constructor call shape from rpython/annotator/dictdef.py while Rust stores the concrete ListItem cell"
    )]
    pub fn new(bookkeeper: Option<Rc<Bookkeeper>>, s_value: SomeValue) -> Rc<RefCell<ListItem>> {
        Rc::new(RefCell::new(ListItem::new(bookkeeper, s_value)))
    }

    /// Upstream has no `DictValue.merge` — it inherits
    /// [`ListItem::merge`]. Thin forwarding wrapper preserves the
    /// call-site parity of `self.dictvalue.merge(other.dictvalue)`
    /// (dictdef.py).
    pub fn merge(
        self_li: &Rc<RefCell<ListItem>>,
        other_li: &Rc<RefCell<ListItem>>,
    ) -> Result<Rc<RefCell<ListItem>>, UnionError> {
        ListItem::merge(self_li, other_li)
    }
}

/// Inner cell of a [`DictDef`].
#[derive(Debug)]
pub(crate) struct DictDefInner {
    /// Current key cell; retargetable via shared ref.
    pub(crate) dictkey: RefCell<Rc<RefCell<ListItem>>>,
    /// Current value cell; retargetable via shared ref.
    pub(crate) dictvalue: RefCell<Rc<RefCell<ListItem>>>,
    /// upstream `self.force_non_null` (dictdef.py).
    pub force_non_null: bool,
    /// upstream `self.simple_hash_eq` (dictdef.py).
    pub simple_hash_eq: bool,
}

/// RPython `class DictDef` (dictdef.py).
#[derive(Clone)]
pub struct DictDef {
    pub(crate) inner: Rc<DictDefInner>,
}

impl std::fmt::Debug for DictDef {
    /// Parity with `DictDef.__repr__` (dictdef.py):
    /// `'<{%r: %r}>'`, recursion-guarded (shared [`ReprGuard`]) so a
    /// self-referential key/value type elides instead of overflowing.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = Rc::as_ptr(&self.inner) as usize;
        let Some(_guard) = super::repr_guard::ReprGuard::enter(id) else {
            return f.write_str("<{...}>");
        };
        let Ok(k) = self.inner.dictkey.try_borrow() else {
            return f.write_str("<{...}>");
        };
        let Ok(key) = k.try_borrow() else {
            return f.write_str("<{...}>");
        };
        let Ok(v) = self.inner.dictvalue.try_borrow() else {
            return f.write_str("<{...}>");
        };
        let Ok(val) = v.try_borrow() else {
            return f.write_str("<{...}>");
        };
        write!(f, "<{{{:?}: {:?}}}>", key.s_value, val.s_value)
    }
}

impl DictDef {
    /// RPython `DictDef.__init__(bookkeeper=None,
    /// s_key=s_ImpossibleValue, s_value=s_ImpossibleValue,
    /// is_r_dict=False, force_non_null=False, simple_hash_eq=False)`
    /// (dictdef.py).
    pub fn new(
        bookkeeper: Option<Rc<Bookkeeper>>,
        s_key: SomeValue,
        s_value: SomeValue,
        is_r_dict: bool,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> Self {
        let key_li = DictKey::new(bookkeeper.clone(), s_key, is_r_dict);
        let value_li = DictValue::new(bookkeeper, s_value);
        let inner = Rc::new(DictDefInner {
            dictkey: RefCell::new(key_li.clone()),
            dictvalue: RefCell::new(value_li.clone()),
            force_non_null,
            simple_hash_eq,
        });
        // upstream: `self.dictkey.itemof[self] = True` (dictdef.py).
        key_li
            .borrow_mut()
            .itemof
            .push(ItemOwner::DictKey(Rc::downgrade(&inner)));
        // upstream: `self.dictvalue.itemof[self] = True` (dictdef.py).
        value_li
            .borrow_mut()
            .itemof
            .push(ItemOwner::DictValue(Rc::downgrade(&inner)));
        DictDef { inner }
    }

    /// RPython `DictDef.same_as(other)` (dictdef.py): identity
    /// on BOTH the key cell AND the value cell.
    pub fn same_as(&self, other: &DictDef) -> bool {
        let ak = self.inner.dictkey.borrow();
        let bk = other.inner.dictkey.borrow();
        let av = self.inner.dictvalue.borrow();
        let bv = other.inner.dictvalue.borrow();
        Rc::ptr_eq(&*ak, &*bk) && Rc::ptr_eq(&*av, &*bv)
    }

    /// RPython `DictDef.union(other)` (dictdef.py).
    ///
    /// Dispatches key / value merges through [`DictKey::merge`] and
    /// [`DictValue::merge`] so the `custom_eq_hash` mixing guard and
    /// `update_rdict_annotations` propagation from dictdef.py
    /// fire at the correct override layer.
    pub fn union_with(&self, other: &DictDef) -> Result<(), UnionError> {
        let self_k = self.inner.dictkey.borrow().clone();
        let other_k = other.inner.dictkey.borrow().clone();
        DictKey::merge(&self_k, &other_k)?;

        let self_v = self.inner.dictvalue.borrow().clone();
        let other_v = other.inner.dictvalue.borrow().clone();
        DictValue::merge(&self_v, &other_v)?;
        Ok(())
    }

    pub fn s_key(&self) -> SomeValue {
        self.inner.dictkey.borrow().borrow().s_value.clone()
    }

    /// RPython `dictdef.dictkey.custom_eq_hash` read access
    /// (binaryop.py:528 / 533). Returns `True` for an r_dict, `False`
    /// otherwise.
    pub fn custom_eq_hash(&self) -> bool {
        self.inner.dictkey.borrow().borrow().custom_eq_hash
    }

    pub fn s_value(&self) -> SomeValue {
        self.inner.dictvalue.borrow().borrow().s_value.clone()
    }

    /// RPython `DictDef.dictkey` accessor — shared Rc for identity-based
    /// walkers (`check_no_flags_on_instances`).
    pub fn dictkey_rc(&self) -> std::rc::Rc<std::cell::RefCell<super::listdef::ListItem>> {
        self.inner.dictkey.borrow().clone()
    }

    /// RPython `DictDef.dictvalue` accessor.
    pub fn dictvalue_rc(&self) -> std::rc::Rc<std::cell::RefCell<super::listdef::ListItem>> {
        self.inner.dictvalue.borrow().clone()
    }

    /// RPython `DictDef.read_key(position_key)` (dictdef.py).
    ///
    /// `position_key` is `Option` to preserve upstream's "no reflow
    /// frame active → `None` position" flow; `None` is just dropped
    /// from the read-locations set (Rust `HashSet` cannot carry it
    /// directly). See [`super::listdef::ListDef::read_item`] for the
    /// same shape.
    pub(crate) fn read_key(&self, position_key: Option<PositionKey>) -> SomeValue {
        let li = self.inner.dictkey.borrow().clone();
        let mut li_mut = li.borrow_mut();
        if let Some(pk) = position_key {
            li_mut.read_locations.insert(pk);
        }
        li_mut.s_value.clone()
    }

    /// RPython `DictDef.read_value(position_key)` (dictdef.py).
    pub(crate) fn read_value(&self, position_key: Option<PositionKey>) -> SomeValue {
        let li = self.inner.dictvalue.borrow().clone();
        let mut li_mut = li.borrow_mut();
        if let Some(pk) = position_key {
            li_mut.read_locations.insert(pk);
        }
        li_mut.s_value.clone()
    }

    /// RPython `DictDef.generalize_key(s_key)` (dictdef.py).
    ///
    /// Routes through [`DictKey::generalize`] so the subclass override
    /// (dictdef.py) fires: when widening a r_dict key cell
    /// actually updates it, the eq/hash emulate-pbc callback must run
    /// to re-validate custom eq/hash annotations. Calling
    /// `ListItem::generalize` directly would bypass that path.
    pub fn generalize_key(&self, s_key: &SomeValue) -> Result<bool, UnionError> {
        // `RDict::get<Q>` (`rordereddict.rs`) takes `Q: Equivalent<K>`.
        // RPython `getitem_str` wraps to the stored key
        // (`dictmultiobject.py` `ObjectDictStrategy.getitem_str`) so
        // `generalize_key` never sees a second key type. A borrowed
        // probe must not enter the key cell.
        if is_equivalent_lookup_probe(s_key) {
            return Ok(false);
        }
        let li = self.inner.dictkey.borrow().clone();
        // `BytesKey` is `rstr.py` `STR`. A `[u8]` probe is `STR.chars`
        // (`Array(Char)` / `rlist.py`), the same Equivalent slice
        // `Borrow<[u8]>` uses (`dictmultiobject.rs` `BytesKey`). Keep
        // the stored cell as STR.
        let current = li.borrow().s_value.clone();
        if matches!(current, SomeValue::String(_)) && is_rstr_chars_list(s_key) {
            return Ok(false);
        }
        if matches!(s_key, SomeValue::String(_)) && is_rstr_chars_list(&current) {
            journal_listitem_mutation(&li);
            let mut li_mut = li.borrow_mut();
            if li_mut.dont_change_any_more {
                return Err(UnionError {
                    lhs: Box::new(current),
                    rhs: Box::new(s_key.clone()),
                    msg: "TooLateForChange on generalize()".into(),
                });
            }
            li_mut.s_value = s_key.clone();
            li_mut.notify_update();
            return Ok(true);
        }
        DictKey::generalize(&li, s_key)
    }

    /// RPython `DictDef.generalize_value(s_value)` (dictdef.py).
    pub fn generalize_value(&self, s_value: &SomeValue) -> Result<bool, UnionError> {
        let li = self.inner.dictvalue.borrow().clone();
        journal_listitem_mutation(&li);
        let mut li_mut = li.borrow_mut();
        li_mut.generalize(s_value)
    }
}

/// `StrLookupKey` is `Equivalent<ObjectKey>` (`rordereddict.rs`
/// `RDict::get`). The stored key cell stays `K`.
fn is_equivalent_lookup_probe(s_key: &SomeValue) -> bool {
    let SomeValue::Instance(inst) = s_key else {
        return false;
    };
    let Some(classdef) = inst.classdef.as_ref() else {
        return false;
    };
    let name = &classdef.borrow().name;
    let leaf = name.rsplit("::").next().unwrap_or(name);
    let leaf = leaf.rsplit('.').next().unwrap_or(leaf);
    leaf.ends_with("LookupKey")
}

/// `[u8]` / `GcArray(Char)` as a dict key: unsigned items, the
/// `rstr.py` `STR.chars` view (`rlist.py` `Array(Char)`).
fn is_rstr_chars_list(s: &SomeValue) -> bool {
    let SomeValue::List(list) = s else {
        return false;
    };
    let item = list.listdef.listitem_rc();
    matches!(item.borrow().s_value, SomeValue::Integer(ref i) if i.unsigned)
}

impl PartialEq for DictDef {
    fn eq(&self, other: &Self) -> bool {
        self.same_as(other)
    }
}

impl Eq for DictDef {}

impl std::hash::Hash for DictDef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let k = self.inner.dictkey.borrow();
        let v = self.inner.dictvalue.borrow();
        let kp: *const RefCell<ListItem> = Rc::as_ptr(&*k);
        let vp: *const RefCell<ListItem> = Rc::as_ptr(&*v);
        (kp as usize).hash(state);
        (vp as usize).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{SomeInteger, SomeString, SomeValue};
    use super::*;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn same_as_across_clone() {
        let a = DictDef::new(
            None,
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let b = a.clone();
        assert!(a.same_as(&b));
    }

    #[test]
    fn distinct_dictdefs_are_not_same_as() {
        let a = DictDef::new(
            None,
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let b = DictDef::new(
            None,
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        assert!(!a.same_as(&b));
    }

    #[test]
    fn generalize_key_routes_through_dictkey_override_on_rdict() {
        // upstream dictdef.py generalize — on an r_dict (custom_eq_hash=true),
        // when generalize actually widens the key type, emulate_rdict_
        // calls must fire. The test passes a SomeInteger as s_eqfn which
        // `bookkeeper.emulate_pbc_call` rejects ("expects SomePBC"),
        // confirming the dispatch reaches the bookkeeper entry-point
        // rather than silently succeeding on the ListItem::generalize
        // path.
        let dd = DictDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::default()),
            /* is_r_dict = */ true,
            false,
            false,
        );
        let err = dd
            .generalize_key(&SomeValue::Integer(SomeInteger::new(false, false)))
            .expect_err("rdict key widen must reach emulate_pbc_call");
        assert!(err.msg.contains("emulate_pbc_call"));
    }

    #[test]
    fn generalize_key_skips_equivalent_lookup_probe() {
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;

        let object_key = ClassDef::new_standalone("dictmultiobject::ObjectKey", None);
        let probe = ClassDef::new_standalone("dictmultiobject::StrLookupKey", None);
        let dd = DictDef::new(
            None,
            SomeValue::Instance(SomeInstance::new(
                Some(object_key.clone()),
                false,
                Default::default(),
            )),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let updated = dd
            .generalize_key(&SomeValue::Instance(SomeInstance::new(
                Some(probe),
                false,
                Default::default(),
            )))
            .expect("Equivalent probe must not union into the stored key cell");
        assert!(!updated);
        match dd.read_key(None) {
            SomeValue::Instance(inst) => {
                let name = inst
                    .classdef
                    .expect("stored key keeps ObjectKey")
                    .borrow()
                    .name
                    .clone();
                assert!(
                    name.ends_with("ObjectKey"),
                    "stored key must stay ObjectKey, got {name}"
                );
            }
            other => panic!("stored key must stay ObjectKey instance, got {other:?}"),
        }
    }

    #[test]
    fn generalize_key_keeps_str_cell_for_char_array_probe() {
        use super::super::listdef::ListDef;
        use super::super::model::SomeList;

        let dd = DictDef::new(
            None,
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let probe = SomeValue::List(SomeList::new(ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(true, true)),
            false,
            false,
        )));
        let updated = dd
            .generalize_key(&probe)
            .expect("STR.chars probe must not union into the STR key cell");
        assert!(!updated);
        assert!(matches!(dd.read_key(None), SomeValue::String(_)));
    }

    #[test]
    fn generalize_key_upgrades_char_array_cell_to_str() {
        use super::super::listdef::ListDef;
        use super::super::model::SomeList;

        let bookkeeper = bk();
        let dd = DictDef::new(
            Some(bookkeeper.clone()),
            SomeValue::List(SomeList::new(ListDef::new(
                Some(bookkeeper),
                SomeValue::Integer(SomeInteger::new(true, true)),
                false,
                false,
            ))),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let updated = dd
            .generalize_key(&SomeValue::String(SomeString::default()))
            .expect("STR must replace the Array(Char) key cell");
        assert!(updated);
        assert!(matches!(dd.read_key(None), SomeValue::String(_)));
    }

    #[test]
    fn generalize_key_on_plain_dict_does_not_trip_rdict_stub() {
        // upstream dictdef.py generalize — custom_eq_hash=false path skips
        // emulate_rdict_calls entirely. Same widening on a plain dict
        // succeeds.
        let dd = DictDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::default()),
            /* is_r_dict = */ false,
            false,
            false,
        );
        let updated = dd
            .generalize_key(&SomeValue::Integer(SomeInteger::new(false, false)))
            .expect("plain-dict key widen must not hit rdict stub");
        assert!(updated);
    }

    #[test]
    fn union_merges_key_and_value_types() {
        let a = DictDef::new(
            Some(bk()),
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
            false,
        );
        let b = DictDef::new(
            Some(bk()),
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
            false,
        );
        a.union_with(&b).expect("merge must succeed");
        // Value type widens to signed.
        if let SomeValue::Integer(v) = a.s_value() {
            assert!(!v.nonneg);
        } else {
            panic!("expected SomeInteger value type");
        }
        // Post-merge identity propagates.
        assert!(a.same_as(&b));
    }
}
