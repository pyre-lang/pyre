//! List definitions — the identity-based element-type carrier used by
//! `SomeList`.
//!
//! RPython upstream: `rpython/annotator/listdef.py` (207 LOC).
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Upstream identity: `ListDef.same_as(other)` reduces to
//!   `self.listitem is other.listitem`. In Python, `is` compares
//!   object identity; two independently-constructed `ListDef`s never
//!   share a `ListItem` unless the bookkeeper merged them.
//!   Rust equivalent: `Rc::ptr_eq(&self.listitem, &other.listitem)`.
//!
//! * Upstream mutation: `ListItem.merge(other)` mutates `self.s_value`
//!   in place then calls `self.patch()`, which walks `self.itemof`
//!   (the set of `ListDef`s currently using this `ListItem`) and
//!   rewrites each `listdef.listitem = self`. After merge both
//!   ListDefs' `same_as` returns True. The Rust port reproduces this
//!   via:
//!     - [`ListDef`] wraps an `Rc<ListDefInner>`; every `ListDef`
//!       clone shares the same inner cell.
//!     - [`ListDefInner::listitem`] is an interior-mutable slot
//!       (`RefCell<Rc<RefCell<ListItem>>>`) so `patch()` can retarget
//!       it through a shared reference.
//!     - [`ListItem::itemof`] stores [`ItemOwner`] weak backrefs so
//!       both `ListDef` owners AND `DictDef` owners can be patched
//!       through the same list. DictKey/DictValue upstream patch via
//!       separate `patch()` overrides on the subclasses; Rust
//!       composition flattens the distinction into an enum.
//!
//! * `TLS.no_side_effects_in_union` (model.py) is replaced by
//!   a thread-local counter with an RAII guard. The guard name is
//!   Rust-native — upstream uses a bare `try/finally` on the global —
//!   but the semantics match byte-for-byte.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::{IndexMap, IndexSet};

use super::repr_guard::ReprGuard;

use super::bookkeeper::{Bookkeeper, PositionKey};
use super::model::{AnnotatorError, SomeList, SomeValue, UnionError};

// Exists to localise prepass nondeterminism (gh#1139).
static REFLOW_FROM_LISTITEM: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn reflow_from_listitem_count() -> u64 {
    REFLOW_FROM_LISTITEM.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
static LISTITEM_WIDEN: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn listitem_widen_count() -> u64 {
    LISTITEM_WIDEN.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
static LISTITEM_NOTIFY_UPDATE: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn listitem_notify_update_count() -> u64 {
    LISTITEM_NOTIFY_UPDATE.load(Ordering::Relaxed)
}

/// RPython `class TooLateForChange(AnnotatorError)` (listdef.py).
/// Raised when mutation is attempted on a `dont_change_any_more`
/// listitem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TooLateForChange;

impl std::fmt::Display for TooLateForChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TooLateForChange")
    }
}

impl std::error::Error for TooLateForChange {}

impl From<TooLateForChange> for AnnotatorError {
    fn from(_: TooLateForChange) -> Self {
        AnnotatorError::new("TooLateForChange")
    }
}

/// RPython `class ListChangeUnallowed(AnnotatorError)` (listdef.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListChangeUnallowed(pub String);

impl std::fmt::Display for ListChangeUnallowed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListChangeUnallowed: {}", self.0)
    }
}

impl std::error::Error for ListChangeUnallowed {}

thread_local! {
    /// Rust-side mirror of upstream `TLS.no_side_effects_in_union`
    /// (model.py). `union()` increments this counter before
    /// dispatching to `pair(s1, s2).union()`; `ListItem.merge` /
    /// `DictKey.merge` consult it to refuse mutation and raise
    /// `UnionError` instead. Thread-local so parallel tests observe
    /// independent state.
    static NO_SIDE_EFFECTS_IN_UNION: Cell<usize> = const { Cell::new(0) };
}

/// RAII wrapper around the `TLS.no_side_effects_in_union += 1` /
/// `-= 1` pattern at `model.py` — upstream guards the
/// increment with a bare `try/finally`; Rust wraps it in a `Drop`
/// impl so callers cannot accidentally leak the state.
///
/// This type has no upstream name — the Rust port introduces it only
/// because `finally` does not exist as a language construct. The guard
/// is `!Send` / `!Sync` so the increment never escapes the thread that
/// took it.
pub(crate) struct SideEffectFreeGuard {
    _not_send: std::marker::PhantomData<*mut ()>,
}

impl SideEffectFreeGuard {
    pub(crate) fn enter() -> Self {
        NO_SIDE_EFFECTS_IN_UNION.with(|c| c.set(c.get() + 1));
        SideEffectFreeGuard {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for SideEffectFreeGuard {
    fn drop(&mut self) {
        NO_SIDE_EFFECTS_IN_UNION.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Mirror of upstream `getattr(TLS, 'no_side_effects_in_union', 0)`
/// (listdef.py:60) — True when the counter is non-zero.
pub(crate) fn in_side_effect_free_union() -> bool {
    NO_SIDE_EFFECTS_IN_UNION.with(|c| c.get() > 0)
}

/// Backref slot used by [`ListItem::itemof`] — enumerates every kind
/// of owner that can hold the listitem so `patch()` retargets the
/// correct cell.
///
/// Upstream (listdef.py) `ListItem.patch()` walks
/// `self.itemof` and sets `listdef.listitem = self`. DictKey /
/// DictValue override `patch()` (dictdef.py, 70-72) to retarget
/// `dictdef.dictkey` / `dictdef.dictvalue` instead. Rust has no
/// subclass dispatch for the `patch` override, so the enum captures
/// the upstream subclass identity directly.
#[derive(Clone, Debug)]
pub(crate) enum ItemOwner {
    /// Owner slot for `ListDef.listitem`.
    ListDef(Weak<ListDefInner>),
    /// Owner slot for `DictDef.dictkey`.
    DictKey(Weak<super::dictdef::DictDefInner>),
    /// Owner slot for `DictDef.dictvalue`.
    DictValue(Weak<super::dictdef::DictDefInner>),
}

/// RPython `class ListItem` (listdef.py).
///
/// The identity-carrying element-type state of a list annotation. Wrap
/// in `Rc<RefCell<ListItem>>` for sharing — sister `ListDef`s clone
/// the Rc so their identity comparisons hold.
///
/// ## TODO: DictKey/DictValue field flattening
///
/// Upstream has `class DictKey(ListItem)` (dictdef.py) and
/// `class DictValue(ListItem)` (dictdef.py). The subclass
/// instances share a `ListItem`-shaped cell with `DictDef` via
/// `dictdef.dictkey` / `dictdef.dictvalue`, and merge operations walk
/// across both base and subclass slots (`custom_eq_hash`,
/// `s_rdict_eqfn`, `s_rdict_hashfn`). Rust has no single inheritance,
/// and [`ItemOwner`] backrefs already store a bare
/// `Rc<RefCell<ListItem>>`, so we flatten the three `DictKey`
/// subclass fields onto this base struct. For non-DictKey uses the
/// fields stay at their default (`custom_eq_hash = false`,
/// `s_rdict_eqfn = s_rdict_hashfn = SomeValue::Impossible`) exactly
/// as `class DictKey(ListItem)` declares in dictdef.py. This is
/// the minimum-deviation collapse of subclass → flattened field
/// (parity rule #1).
#[derive(Debug)]
pub struct ListItem {
    /// RPython `self.s_value` (listdef.py:30).
    pub s_value: SomeValue,
    /// RPython `self.bookkeeper` (listdef.py:31). `None` matches
    /// upstream's `bookkeeper=None` path (listdef.py:34-35) and sets
    /// [`Self::dont_change_any_more`] at construction.
    pub bookkeeper: Option<Rc<Bookkeeper>>,
    /// RPython `self.mutated` (listdef.py:13).
    pub mutated: bool,
    /// RPython `self.resized` (listdef.py:14).
    pub resized: bool,
    /// RPython `self.range_step` (listdef.py:15). Upstream stores the
    /// step as either `None` (list not from a range()) or an integer;
    /// `0` means "variable step" — the sentinel value produced by
    /// merging two constant-step lists with different step values.
    pub range_step: Option<i64>,
    /// RPython `self.dont_change_any_more` (listdef.py:16).
    pub dont_change_any_more: bool,
    /// RPython `self.immutable` (listdef.py:17).
    pub immutable: bool,
    /// RPython `self.must_not_resize` (listdef.py:18).
    pub must_not_resize: bool,
    /// RPython `self.itemof = {}` (listdef.py). Weak backrefs to
    /// every owner currently using this `ListItem`.
    pub(crate) itemof: Vec<ItemOwner>,
    /// RPython `self.read_locations = set()` (listdef.py).
    /// The ordered container is required because the key hashes on a
    /// pointer and the loop over the members produces a work order.
    pub(crate) read_locations: IndexSet<PositionKey>,
    /// Flattened `DictKey.custom_eq_hash` (dictdef.py). `false` for
    /// every non-DictKey ListItem.
    pub custom_eq_hash: bool,
    /// Flattened `DictKey.s_rdict_eqfn` (dictdef.py). Defaults to
    /// `SomeValue::Impossible` (= upstream `s_ImpossibleValue`).
    pub s_rdict_eqfn: SomeValue,
    /// Flattened `DictKey.s_rdict_hashfn` (dictdef.py).
    pub s_rdict_hashfn: SomeValue,
}

impl ListItem {
    /// RPython `ListItem.__init__(bookkeeper, s_value)` (listdef.py).
    ///
    /// Sets `dont_change_any_more = True` when `bookkeeper is None`.
    pub fn new(bookkeeper: Option<Rc<Bookkeeper>>, s_value: SomeValue) -> Self {
        let dont_change_any_more = bookkeeper.is_none();
        ListItem {
            s_value,
            bookkeeper,
            mutated: false,
            resized: false,
            range_step: None,
            dont_change_any_more,
            immutable: false,
            must_not_resize: false,
            itemof: Vec::new(),
            read_locations: IndexSet::new(),
            // Flattened DictKey defaults (dictdef.py, 13).
            custom_eq_hash: false,
            s_rdict_eqfn: SomeValue::Impossible,
            s_rdict_hashfn: SomeValue::Impossible,
        }
    }

    /// RPython `ListItem.notify_update()` (listdef.py).
    ///
    /// ```python
    /// def notify_update(self):
    ///     '''Reflow from all reading points'''
    ///     for position_key in self.read_locations:
    ///         self.bookkeeper.annotator.reflowfromposition(position_key)
    /// ```
    ///
    /// `self.bookkeeper` is optional on the Rust port (matches upstream's
    /// `bookkeeper=None` path at listdef.py which sets
    /// `dont_change_any_more=True`). When the bookkeeper / annotator
    /// backlink is absent, the loop is structurally preserved but no
    /// reflow fires — the only way to reach notify_update from that
    /// state is a guarded path that already errors via
    /// [`TooLateForChange`].
    pub fn notify_update(&self) {
        LISTITEM_NOTIFY_UPDATE.fetch_add(1, Ordering::Relaxed);
        let Some(bk) = self.bookkeeper.as_ref() else {
            return;
        };
        let Some(ann) = bk.annotator.borrow().upgrade() else {
            return;
        };
        for position_key in &self.read_locations {
            REFLOW_FROM_LISTITEM.fetch_add(1, Ordering::Relaxed);
            ann.reflowfromposition(position_key);
        }
    }

    /// RPython `ListItem.generalize(s_other_value)` (listdef.py).
    ///
    /// Widens `self.s_value` with `s_other_value` via `unionof`, then
    /// (if widened) notifies reflow readers. Returns `true` when the
    /// type actually widened.
    pub fn generalize(&mut self, s_other_value: &SomeValue) -> Result<bool, UnionError> {
        let s_new_value = super::model::union(&self.s_value, s_other_value)?;
        let updated = s_new_value != self.s_value;
        if updated {
            if self.dont_change_any_more {
                return Err(UnionError {
                    lhs: Box::new(self.s_value.clone()),
                    rhs: Box::new(s_other_value.clone()),
                    msg: "TooLateForChange on generalize()".into(),
                });
            }
            self.s_value = s_new_value;
            LISTITEM_WIDEN.fetch_add(1, Ordering::Relaxed);
            self.notify_update();
        }
        Ok(updated)
    }

    /// RPython `ListItem.mutate()` (listdef.py).
    pub fn mutate(&mut self) -> Result<(), TooLateForChange> {
        if !self.mutated {
            if self.dont_change_any_more {
                return Err(TooLateForChange);
            }
            self.immutable = false;
            self.mutated = true;
        }
        Ok(())
    }

    /// RPython `ListItem.resize()` (listdef.py).
    pub fn resize(&mut self) -> Result<(), AnnotatorError> {
        if !self.resized {
            if self.dont_change_any_more {
                return Err(AnnotatorError::new("TooLateForChange"));
            }
            if self.must_not_resize {
                return Err(AnnotatorError::new("ListChangeUnallowed: resizing list"));
            }
            self.resized = true;
        }
        Ok(())
    }

    /// RPython `ListItem.setrangestep(step)` (listdef.py).
    pub fn setrangestep(&mut self, step: Option<i64>) -> Result<(), TooLateForChange> {
        if step != self.range_step {
            if self.dont_change_any_more {
                return Err(TooLateForChange);
            }
            self.range_step = step;
        }
        Ok(())
    }

    /// RPython `ListItem.merge(other)` (listdef.py).
    ///
    /// Takes two `Rc<RefCell<ListItem>>` associated-function style
    /// rather than `&mut self` / `&mut other` because the borrow
    /// checker refuses two mutable borrows of the same `Vec<ItemOwner>`
    /// slot when the caller happens to pass the same Rc twice (the
    /// `Rc::ptr_eq` shortcut below exits before any borrow). The
    /// upstream method name is preserved.
    pub fn merge(
        self_li: &Rc<RefCell<ListItem>>,
        other_li: &Rc<RefCell<ListItem>>,
    ) -> Result<Rc<RefCell<ListItem>>, UnionError> {
        // upstream: `if self is not other:`.
        if Rc::ptr_eq(self_li, other_li) {
            return Ok(self_li.clone());
        }

        // upstream: `if getattr(TLS, 'no_side_effects_in_union', 0):
        //                raise UnionError(self, other)`.
        if in_side_effect_free_union() {
            let a = self_li.borrow();
            let b = other_li.borrow();
            return Err(UnionError {
                lhs: Box::new(a.s_value.clone()),
                rhs: Box::new(b.s_value.clone()),
                msg: "ListItem.merge during side-effect-free union".into(),
            });
        }

        // upstream: `if other.dont_change_any_more: if
        // self.dont_change_any_more: raise TooLateForChange;
        // else: self, other = other, self` (listdef.py).
        let (driver_li, folded_li) = {
            let self_b = self_li.borrow();
            let other_b = other_li.borrow();
            if other_b.dont_change_any_more {
                if self_b.dont_change_any_more {
                    return Err(UnionError {
                        lhs: Box::new(self_b.s_value.clone()),
                        rhs: Box::new(other_b.s_value.clone()),
                        msg: "TooLateForChange".into(),
                    });
                }
                (other_li, self_li)
            } else {
                (self_li, other_li)
            }
        };

        // Snapshot folded side (everything upstream reads as `other.X`
        // after the swap). Having a single snapshot prevents intermixed
        // borrows with driver_mut below.
        let (folded_s_value, folded_itemof, folded_read_locations, folded_flags) = {
            let folded_b = folded_li.borrow();
            (
                folded_b.s_value.clone(),
                folded_b.itemof.clone(),
                folded_b.read_locations.clone(),
                (
                    folded_b.mutated,
                    folded_b.resized,
                    folded_b.immutable,
                    folded_b.must_not_resize,
                    folded_b.range_step,
                ),
            )
        };

        // Checkpoint both items before any flag / itemof / s_value write.
        // `AddedBlocksGuard` restores op results and cancels a pending
        // reflow, but not this in-place `ListItem.merge`.
        journal_merge_items(driver_li, folded_li);

        // upstream lines 73-85: flag merges. Order preserved exactly.
        {
            let mut driver_mut = driver_li.borrow_mut();
            driver_mut.immutable &= folded_flags.2;
            if folded_flags.3 {
                if driver_mut.resized {
                    return Err(UnionError {
                        lhs: Box::new(driver_mut.s_value.clone()),
                        rhs: Box::new(folded_s_value.clone()),
                        msg: "ListChangeUnallowed: list merge with a resized".into(),
                    });
                }
                driver_mut.must_not_resize = true;
            }
        }
        if folded_flags.0 {
            // upstream: `self.mutate()` — propagates TooLateForChange.
            driver_li.borrow_mut().mutate().map_err(|_| UnionError {
                lhs: Box::new(driver_li.borrow().s_value.clone()),
                rhs: Box::new(folded_s_value.clone()),
                msg: "TooLateForChange on mutate() during merge".into(),
            })?;
        }
        if folded_flags.1 {
            // upstream: `self.resize()` — propagates TooLateForChange /
            // ListChangeUnallowed.
            driver_li.borrow_mut().resize().map_err(|e| UnionError {
                lhs: Box::new(driver_li.borrow().s_value.clone()),
                rhs: Box::new(folded_s_value.clone()),
                msg: e.msg.unwrap_or_else(|| "resize() failed".into()),
            })?;
        }
        let driver_range_step = driver_li.borrow().range_step;
        if folded_flags.4 != driver_range_step {
            // upstream: `self.setrangestep(self._step_map[...])`.
            let new_step = merge_range_step(driver_range_step, folded_flags.4);
            driver_li
                .borrow_mut()
                .setrangestep(new_step)
                .map_err(|_| UnionError {
                    lhs: Box::new(driver_li.borrow().s_value.clone()),
                    rhs: Box::new(folded_s_value.clone()),
                    msg: "TooLateForChange on setrangestep() during merge".into(),
                })?;
        }

        // upstream: `self.itemof.update(other.itemof)` (listdef.py).
        driver_li
            .borrow_mut()
            .itemof
            .extend(folded_itemof.iter().cloned());

        // upstream lines 86-91.
        let driver_s_value_pre = driver_li.borrow().s_value.clone();
        let new_s_value = super::model::union(&driver_s_value_pre, &folded_s_value)?;
        let widens_driver = new_s_value != driver_s_value_pre;
        let widens_folded = new_s_value != folded_s_value;
        if widens_driver && driver_li.borrow().dont_change_any_more {
            return Err(UnionError {
                lhs: Box::new(driver_s_value_pre),
                rhs: Box::new(folded_s_value),
                msg: "TooLateForChange on dont_change_any_more ListItem".into(),
            });
        }

        // upstream: `self.patch()` (listdef.py, 100-103). After the
        // itemof.update above, driver.itemof holds every owner; retarget
        // them all to driver.
        let patch_list = driver_li.borrow().itemof.clone();
        for owner in &patch_list {
            retarget_owner(owner, driver_li);
        }

        // upstream lines 93-98: conditional s_value update + notify +
        // read_locations merge.
        if widens_driver {
            driver_li.borrow_mut().s_value = new_s_value;
            driver_li.borrow().notify_update();
        }
        if widens_folded {
            // upstream: `other.notify_update()`. folded_li still holds
            // the old read_locations snapshot before we overwrite
            // driver.read_locations below.
            folded_li.borrow().notify_update();
        }
        // upstream: `self.read_locations |= other.read_locations`.
        driver_li
            .borrow_mut()
            .read_locations
            .extend(folded_read_locations);

        Ok(driver_li.clone())
    }
}

/// First-touch record of one `ListItem` mutated inside an added-blocks
/// scope. `AddedBlocksGuard` puts variable bindings back and drops a
/// reflow that has not run; `ListItem.merge` still widens the shared
/// item in place. `SomeList` equality is listitem identity, so
/// `mergeinputargs` (annrpython.py) does not re-enter the reader.
/// Leaving the widened item next to the restored getitem result makes
/// `rtype_getitem`'s lltype disagree with `hop.r_result`. RPython has
/// no per-subject rollback.
struct ListItemCheckpoint {
    s_value: SomeValue,
    mutated: bool,
    resized: bool,
    immutable: bool,
    must_not_resize: bool,
    dont_change_any_more: bool,
    range_step: Option<i64>,
    read_locations: IndexSet<PositionKey>,
    /// `itemof` only grows (`ListItem.merge`). Restore truncates.
    itemof_len: usize,
    custom_eq_hash: bool,
    s_rdict_eqfn: SomeValue,
    s_rdict_hashfn: SomeValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum OwnerSlot {
    ListDef,
    DictKey,
    DictValue,
}

/// Mutations recorded while `RPythonAnnotator::listitem_journal` is
/// `Some`. Direct field writes on rollback: no `generalize`, no
/// `notify_update`, no reflow. A reflow from `Drop` would re-seed the
/// failed subject's evicted blocks.
pub(crate) struct ListItemJournal {
    items: IndexMap<usize, (Rc<RefCell<ListItem>>, ListItemCheckpoint)>,
    /// First owner retarget, in the order `patch` applied them.
    /// `DictKey` and `DictValue` are both `Weak<DictDefInner>`, so the
    /// slot tag keeps those addresses apart.
    owners: IndexMap<(OwnerSlot, usize), (ItemOwner, Rc<RefCell<ListItem>>)>,
}

impl ListItemJournal {
    pub(crate) fn new() -> Self {
        ListItemJournal {
            items: IndexMap::new(),
            owners: IndexMap::new(),
        }
    }

    pub(crate) fn note_item(&mut self, li: &Rc<RefCell<ListItem>>) {
        let key = Rc::as_ptr(li) as usize;
        if self.items.contains_key(&key) {
            return;
        }
        let Ok(item) = li.try_borrow() else {
            return;
        };
        let checkpoint = ListItemCheckpoint {
            s_value: item.s_value.clone(),
            mutated: item.mutated,
            resized: item.resized,
            immutable: item.immutable,
            must_not_resize: item.must_not_resize,
            dont_change_any_more: item.dont_change_any_more,
            range_step: item.range_step,
            read_locations: item.read_locations.clone(),
            itemof_len: item.itemof.len(),
            custom_eq_hash: item.custom_eq_hash,
            s_rdict_eqfn: item.s_rdict_eqfn.clone(),
            s_rdict_hashfn: item.s_rdict_hashfn.clone(),
        };
        drop(item);
        self.items.insert(key, (Rc::clone(li), checkpoint));
    }

    pub(crate) fn note_owner(&mut self, owner: &ItemOwner, previous: Rc<RefCell<ListItem>>) {
        let key = owner_slot_key(owner);
        self.owners
            .entry(key)
            .or_insert_with(|| (owner.clone(), previous));
    }

    /// A committed inner scope's writes stay when the outer scope rolls
    /// back. Fields the inner scope changed are copied into this
    /// checkpoint so `rollback` writes the current value back. Fields
    /// only this scope changed keep the original checkpoint.
    pub(crate) fn keep_committed_helper(&mut self, helper: &ListItemJournal) {
        for (key, (li, helper_saved)) in &helper.items {
            let Some((_, outer_saved)) = self.items.get_mut(key) else {
                continue;
            };
            let Ok(current) = li.try_borrow() else {
                continue;
            };
            if current.s_value != helper_saved.s_value {
                outer_saved.s_value = current.s_value.clone();
            }
            if current.mutated != helper_saved.mutated {
                outer_saved.mutated = current.mutated;
            }
            if current.resized != helper_saved.resized {
                outer_saved.resized = current.resized;
            }
            if current.immutable != helper_saved.immutable {
                outer_saved.immutable = current.immutable;
            }
            if current.must_not_resize != helper_saved.must_not_resize {
                outer_saved.must_not_resize = current.must_not_resize;
            }
            if current.dont_change_any_more != helper_saved.dont_change_any_more {
                outer_saved.dont_change_any_more = current.dont_change_any_more;
            }
            if current.range_step != helper_saved.range_step {
                outer_saved.range_step = current.range_step;
            }
            if current.read_locations != helper_saved.read_locations {
                outer_saved.read_locations = current.read_locations.clone();
            }
            if current.itemof.len() != helper_saved.itemof_len {
                outer_saved.itemof_len = current.itemof.len();
            }
            if current.custom_eq_hash != helper_saved.custom_eq_hash {
                outer_saved.custom_eq_hash = current.custom_eq_hash;
            }
            if current.s_rdict_eqfn != helper_saved.s_rdict_eqfn {
                outer_saved.s_rdict_eqfn = current.s_rdict_eqfn.clone();
            }
            if current.s_rdict_hashfn != helper_saved.s_rdict_hashfn {
                outer_saved.s_rdict_hashfn = current.s_rdict_hashfn.clone();
            }
        }
        for (key, (owner, helper_previous)) in &helper.owners {
            let Some(current) = owner_slot_rc(owner) else {
                continue;
            };
            if Rc::ptr_eq(&current, helper_previous) {
                continue;
            }
            if let Some((_, previous)) = self.owners.get_mut(key) {
                *previous = current;
            }
        }
    }

    /// Put owner slots back, then item fields. `mem::take` first so a
    /// write during restore cannot re-journal the restored state.
    pub(crate) fn rollback(&mut self) {
        let owners = std::mem::take(&mut self.owners);
        let items = std::mem::take(&mut self.items);
        for (_, (owner, previous)) in owners.into_iter().rev() {
            restore_owner_slot(&owner, &previous);
        }
        for (_, (li, checkpoint)) in items {
            let Ok(mut item) = li.try_borrow_mut() else {
                continue;
            };
            item.s_value = checkpoint.s_value;
            item.mutated = checkpoint.mutated;
            item.resized = checkpoint.resized;
            item.immutable = checkpoint.immutable;
            item.must_not_resize = checkpoint.must_not_resize;
            item.dont_change_any_more = checkpoint.dont_change_any_more;
            item.range_step = checkpoint.range_step;
            item.read_locations = checkpoint.read_locations;
            item.itemof.truncate(checkpoint.itemof_len);
            item.custom_eq_hash = checkpoint.custom_eq_hash;
            item.s_rdict_eqfn = checkpoint.s_rdict_eqfn;
            item.s_rdict_hashfn = checkpoint.s_rdict_hashfn;
        }
    }
}

fn owner_slot_key(owner: &ItemOwner) -> (OwnerSlot, usize) {
    match owner {
        ItemOwner::ListDef(weak) => (OwnerSlot::ListDef, weak.as_ptr() as usize),
        ItemOwner::DictKey(weak) => (OwnerSlot::DictKey, weak.as_ptr() as usize),
        ItemOwner::DictValue(weak) => (OwnerSlot::DictValue, weak.as_ptr() as usize),
    }
}

fn annotator_of(li: &Rc<RefCell<ListItem>>) -> Option<Rc<super::annrpython::RPythonAnnotator>> {
    let item = li.try_borrow().ok()?;
    let bookkeeper = item.bookkeeper.as_ref()?;
    bookkeeper.try_annotator()
}

/// Record `li` if an added-blocks scope is open. No-op when the item
/// has no annotator, or when the journal `RefCell` is already borrowed
/// (`Drop` must not panic).
pub(crate) fn journal_listitem_mutation(li: &Rc<RefCell<ListItem>>) {
    let Some(ann) = annotator_of(li) else {
        return;
    };
    ann.note_listitem_mutation(li);
}

fn journal_merge_items(driver: &Rc<RefCell<ListItem>>, folded: &Rc<RefCell<ListItem>>) {
    let ann_driver = annotator_of(driver);
    let ann_folded = annotator_of(folded);
    if let Some(ann) = ann_driver.as_ref().or(ann_folded.as_ref()) {
        ann.note_listitem_mutation(driver);
    }
    if let Some(ann) = ann_folded.as_ref().or(ann_driver.as_ref()) {
        ann.note_listitem_mutation(folded);
    }
}

fn journal_owner_retarget(
    driver: &Rc<RefCell<ListItem>>,
    owner: &ItemOwner,
    previous: &Rc<RefCell<ListItem>>,
) {
    let ann = annotator_of(driver).or_else(|| annotator_of(previous));
    let Some(ann) = ann else {
        return;
    };
    ann.note_owner_retarget(owner, Rc::clone(previous));
}

fn with_owner_slot<R>(
    owner: &ItemOwner,
    f: impl FnOnce(&RefCell<Rc<RefCell<ListItem>>>) -> R,
) -> Option<R> {
    match owner {
        ItemOwner::ListDef(weak) => weak.upgrade().map(|inner| f(&inner.listitem)),
        ItemOwner::DictKey(weak) => weak.upgrade().map(|inner| f(&inner.dictkey)),
        ItemOwner::DictValue(weak) => weak.upgrade().map(|inner| f(&inner.dictvalue)),
    }
}

fn owner_slot_rc(owner: &ItemOwner) -> Option<Rc<RefCell<ListItem>>> {
    with_owner_slot(owner, |slot| {
        slot.try_borrow().ok().map(|current| Rc::clone(&current))
    })
    .flatten()
}

fn assign_owner_slot(owner: &ItemOwner, driver: &Rc<RefCell<ListItem>>) {
    with_owner_slot(owner, |slot| {
        *slot.borrow_mut() = Rc::clone(driver);
    });
}

fn restore_owner_slot(owner: &ItemOwner, previous: &Rc<RefCell<ListItem>>) {
    with_owner_slot(owner, |slot| {
        if let Ok(mut current) = slot.try_borrow_mut() {
            *current = Rc::clone(previous);
        }
    });
}

/// `ListItem.patch`: point `owner`'s cell at `driver`. The previous
/// cell is journaled so an uncommitted scope can put it back.
fn retarget_owner(owner: &ItemOwner, driver: &Rc<RefCell<ListItem>>) {
    let Some(previous) = owner_slot_rc(owner) else {
        return;
    };
    if !Rc::ptr_eq(&previous, driver) {
        journal_owner_retarget(driver, owner, &previous);
    }
    assign_owner_slot(owner, driver);
}

/// RPython `ListItem._step_map[type(self.range_step),
/// type(other.range_step)]` (listdef.py). Upstream keys the dict
/// on `(type(None), int)` / `(int, type(None))` / `(int, int)`.
fn merge_range_step(self_step: Option<i64>, other_step: Option<i64>) -> Option<i64> {
    match (self_step, other_step) {
        // `(NoneType, int)` / `(int, NoneType)` → None.
        (None, Some(_)) | (Some(_), None) => None,
        // `(int, int)` with different values → 0 (variable step).
        (Some(a), Some(b)) if a != b => Some(0),
        // Same-value (int, int) or (None, None) — upstream never
        // invokes the map on equality, so the branch just returns
        // self.
        _ => self_step,
    }
}

/// Inner cell of a [`ListDef`].
///
/// `listitem` lives inside an interior-mutable slot so
/// [`ListItem::merge`] can retarget it through an [`ItemOwner`].
#[derive(Debug)]
pub(crate) struct ListDefInner {
    pub(crate) listitem: RefCell<Rc<RefCell<ListItem>>>,
}

/// RPython `class ListDef` (listdef.py).
#[derive(Clone)]
pub struct ListDef {
    pub(crate) inner: Rc<ListDefInner>,
}

impl fmt::Debug for ListDef {
    /// Parity with `ListDef.__repr__` (listdef.py):
    /// `'<[%r]%s%s%s%s>'`, recursion-guarded so a self-referential
    /// element type elides instead of overflowing the stack.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id = Rc::as_ptr(&self.inner) as usize;
        let Some(_guard) = ReprGuard::enter(id) else {
            return f.write_str("<[...]>");
        };
        let Ok(li) = self.inner.listitem.try_borrow() else {
            return f.write_str("<[...]>");
        };
        let Ok(item) = li.try_borrow() else {
            return f.write_str("<[...]>");
        };
        write!(
            f,
            "<[{:?}]{}{}{}{}>",
            item.s_value,
            if item.mutated { "m" } else { "" },
            if item.resized { "r" } else { "" },
            if item.immutable { "I" } else { "" },
            if item.must_not_resize { "!R" } else { "" },
        )
    }
}

impl ListDef {
    /// RPython `ListDef.__init__(bookkeeper, s_item=s_ImpossibleValue,
    /// mutated=False, resized=False)` (listdef.py).
    pub fn new(
        bookkeeper: Option<Rc<Bookkeeper>>,
        s_item: SomeValue,
        mutated: bool,
        resized: bool,
    ) -> Self {
        let mut item = ListItem::new(bookkeeper, s_item);
        // upstream: `self.listitem.mutated = mutated | resized;
        //            self.listitem.resized = resized`.
        item.mutated = mutated || resized;
        item.resized = resized;
        let li = Rc::new(RefCell::new(item));
        let inner = Rc::new(ListDefInner {
            listitem: RefCell::new(li.clone()),
        });
        // upstream: `self.listitem.itemof[self] = True`.
        li.borrow_mut()
            .itemof
            .push(ItemOwner::ListDef(Rc::downgrade(&inner)));
        ListDef { inner }
    }

    /// RPython `ListDef.same_as(other)` (listdef.py).
    pub fn same_as(&self, other: &ListDef) -> bool {
        let a = self.inner.listitem.borrow();
        let b = other.inner.listitem.borrow();
        Rc::ptr_eq(&*a, &*b)
    }

    /// RPython `ListDef.union(other)` (listdef.py).
    pub fn union_with(&self, other: &ListDef) -> Result<(), UnionError> {
        let self_li = self.inner.listitem.borrow().clone();
        let other_li = other.inner.listitem.borrow().clone();
        let _ = ListItem::merge(&self_li, &other_li)?;
        Ok(())
    }

    /// RPython `ListDef.listitem.s_value` accessor — shortcut used by
    /// call sites that only need the element annotation without
    /// borrowing the listitem directly.
    pub fn s_value(&self) -> SomeValue {
        self.inner.listitem.borrow().borrow().s_value.clone()
    }

    /// RPython `ListDef.listitem` accessor — returns the shared
    /// `Rc<RefCell<ListItem>>` so `check_no_flags_on_instances` and
    /// other walkers can inspect identity.
    pub fn listitem_rc(&self) -> Rc<RefCell<ListItem>> {
        self.inner.listitem.borrow().clone()
    }

    /// RPython `ListDef.mutate()` (listdef.py).
    pub fn mutate(&self) -> Result<(), TooLateForChange> {
        let li = self.inner.listitem.borrow().clone();
        journal_listitem_mutation(&li);
        let mut li_mut = li.borrow_mut();
        li_mut.mutate()
    }

    /// RPython `ListDef.resize()` (listdef.py).
    ///
    /// ```python
    /// def resize(self):
    ///     self.listitem.mutate()
    ///     self.listitem.resize()
    /// ```
    pub fn resize(&self) -> Result<(), AnnotatorError> {
        let li = self.inner.listitem.borrow().clone();
        journal_listitem_mutation(&li);
        let mut li_mut = li.borrow_mut();
        li_mut
            .mutate()
            .map_err(|_| AnnotatorError::new("TooLateForChange"))?;
        li_mut.resize()
    }

    /// RPython `ListDef.read_item(position_key)` (listdef.py).
    ///
    /// Records a read location for eventual `notify_update()` reflow,
    /// then returns the current element annotation. `position_key` is
    /// `Option` because some callers (list builtins) have no reflow
    /// position and pass `None`. Dropping that `None` is faithful, not a
    /// divergence: `read_locations` is consumed by `reflowfromposition`
    /// (annrpython.py), which unpacks `graph, block, index =
    /// position_key` — a `None` member would crash the reflow loop, so
    /// upstream's set only ever holds real positions. The Rust port's
    /// `HashSet<PositionKey>` encodes that invariant in the type.
    pub(crate) fn read_item(&self, position_key: Option<PositionKey>) -> SomeValue {
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        if let Some(pk) = position_key {
            li_mut.read_locations.insert(pk);
        }
        li_mut.s_value.clone()
    }

    /// RPython `ListDef.generalize(s_value)` (listdef.py).
    pub fn generalize(&self, s_value: &SomeValue) -> Result<bool, UnionError> {
        let li = self.inner.listitem.borrow().clone();
        journal_listitem_mutation(&li);
        let mut li_mut = li.borrow_mut();
        li_mut.generalize(s_value)
    }

    /// RPython `ListDef.never_resize(self)` (listdef.py).
    ///
    /// ```python
    /// def never_resize(self):
    ///     if self.listitem.resized:
    ///         raise ListChangeUnallowed("list already resized")
    ///     self.listitem.must_not_resize = True
    /// ```
    pub fn never_resize(&self) -> Result<(), ListChangeUnallowed> {
        let li = self.inner.listitem.borrow().clone();
        if li.borrow().resized {
            return Err(ListChangeUnallowed("list already resized".to_string()));
        }
        // `mark_as_immutable` sets `immutable` after this returns, so the
        // first-touch checkpoint must be the state before either write.
        journal_listitem_mutation(&li);
        li.borrow_mut().must_not_resize = true;
        Ok(())
    }

    /// RPython `ListDef.mark_as_immutable(self)` (listdef.py).
    ///
    /// ```python
    /// def mark_as_immutable(self):
    ///     self.never_resize()
    ///     if not self.listitem.mutated:
    ///         self.listitem.immutable = True
    /// ```
    pub fn mark_as_immutable(&self) -> Result<(), ListChangeUnallowed> {
        self.never_resize()?;
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        if !li_mut.mutated {
            li_mut.immutable = true;
        }
        // else: the list was already mutated; leave immutable=false. The
        // upstream comment at listdef.py:203-204 notes this is expected
        // and matches test_can_merge_immutable_list_with_regular_list.
        Ok(())
    }

    /// RPython `ListDef.offspring(bookkeeper, *others)` (listdef.py).
    ///
    /// ```python
    /// def offspring(self, bookkeeper, *others):
    ///     position = bookkeeper.position_key
    ///     s_self_value = self.read_item(position)
    ///     s_other_values = []
    ///     for other in others:
    ///         s_other_values.append(other.read_item(position))
    ///     s_newlst = bookkeeper.newlist(s_self_value, *s_other_values)
    ///     s_newvalue = s_newlst.listdef.read_item(position)
    ///     self.generalize(s_newvalue)
    ///     for other in others:
    ///         other.generalize(s_newvalue)
    ///     return s_newlst
    /// ```
    pub fn offspring(
        &self,
        bookkeeper: &Rc<Bookkeeper>,
        others: &[&ListDef],
    ) -> Result<SomeList, AnnotatorError> {
        // upstream: `position = bookkeeper.position_key`. Outside of a
        // reflow frame this is `None`; Python dict/set keys accept it,
        // so the Rust port passes the Option through.
        let position = bookkeeper.current_position_key();
        let s_self_value = self.read_item(position.clone());
        let mut s_other_values: Vec<SomeValue> = Vec::with_capacity(others.len());
        for other in others {
            s_other_values.push(other.read_item(position.clone()));
        }
        // upstream: `bookkeeper.newlist(s_self_value, *s_other_values)`.
        let mut all_values = Vec::with_capacity(1 + s_other_values.len());
        all_values.push(s_self_value);
        all_values.extend(s_other_values);
        let s_newlst = bookkeeper.newlist(&all_values, None)?;
        let s_newvalue = s_newlst.listdef.read_item(position);
        self.generalize(&s_newvalue)
            .map_err(|e| AnnotatorError::new(e.msg))?;
        for other in others {
            other
                .generalize(&s_newvalue)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(s_newlst)
    }

    /// RPython `ListDef.generalize_range_step(range_step)`
    /// (listdef.py).
    ///
    /// Creates a fresh ListItem carrying the candidate `range_step`,
    /// then merges it into `self.listitem` so `_step_map` collapses
    /// the two step values (matching upstream lines 82-85).
    pub fn generalize_range_step(&self, range_step: Option<i64>) -> Result<(), UnionError> {
        let bookkeeper = {
            let li = self.inner.listitem.borrow().clone();
            li.borrow().bookkeeper.clone()
        };
        let mut new_item = ListItem::new(bookkeeper, SomeValue::Impossible);
        new_item.range_step = range_step;
        let new_li = Rc::new(RefCell::new(new_item));
        let self_li = self.inner.listitem.borrow().clone();
        let _ = ListItem::merge(&self_li, &new_li)?;
        Ok(())
    }

    /// RPython `ListDef.agree(bookkeeper, other)` (listdef.py).
    ///
    /// Bidirectionally generalises both sides against each other at
    /// the bookkeeper's current position, then reconciles `range_step`
    /// if either side is range-derived. `position_key` is passed as
    /// `Option` so the upstream None-key caching path (no reflow
    /// frame active) flows through unchanged.
    pub fn agree(&self, bookkeeper: &Bookkeeper, other: &ListDef) -> Result<(), UnionError> {
        let position = bookkeeper.current_position_key();
        let s_self_value = self.read_item(position.clone());
        let s_other_value = other.read_item(position);
        self.generalize(&s_other_value)?;
        other.generalize(&s_self_value)?;
        let (self_step, other_step) = {
            let a = self.inner.listitem.borrow().clone();
            let b = other.inner.listitem.borrow().clone();
            (a.borrow().range_step, b.borrow().range_step)
        };
        if self_step.is_some() {
            self.generalize_range_step(other_step)?;
        }
        if other_step.is_some() {
            other.generalize_range_step(self_step)?;
        }
        Ok(())
    }
}

impl PartialEq for ListDef {
    /// RPython `SomeList.__eq__` (model.py:339-348) uses
    /// `listdef.same_as`; mirror the identity-only semantics here so
    /// wrapping structs picking up `derive(PartialEq)` inherit it.
    fn eq(&self, other: &Self) -> bool {
        self.same_as(other)
    }
}

impl Eq for ListDef {}

/// Stable hash matching the identity-based equality above.
impl std::hash::Hash for ListDef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let li = self.inner.listitem.borrow();
        let raw: *const RefCell<ListItem> = Rc::as_ptr(&*li);
        (raw as usize).hash(state);
    }
}

/// RPython `s_list_of_strings = SomeList(ListDef(None,
/// SomeString(no_nul=True), resized=True))` (listdef.py).
pub(crate) fn s_list_of_strings() -> super::model::SomeList {
    super::model::SomeList::new(ListDef::new(
        None,
        super::model::SomeValue::String(super::model::SomeString::new(false, true)),
        false,
        true,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::model::{SomeInteger, SomeValue};
    use super::*;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn debug_of_self_referential_list_terminates() {
        use super::super::model::SomeList;
        // A list whose element type is itself (RPython `l = []; l.append(l)`)
        // is a legal, self-referential annotation. Formatting it must break
        // the cycle instead of recursing forever, mirroring the reprdict
        // recursion guard in `SomeObject.__repr__` (model.py:68).
        let ld = ListDef::new(None, SomeValue::Impossible, false, false);
        // Close the cycle: listitem.s_value := SomeList(ld).
        ld.listitem_rc().borrow_mut().s_value = SomeValue::List(SomeList::new(ld.clone()));
        // Must terminate (no stack overflow) and mark the elided cycle.
        let rendered = format!("{:?}", SomeValue::List(SomeList::new(ld.clone())));
        assert!(rendered.contains("..."), "cycle not elided: {rendered}");
    }

    #[test]
    fn same_as_identity_preserved_across_clone() {
        let a = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        let b = a.clone();
        assert!(a.same_as(&b));
    }

    #[test]
    fn distinct_listdefs_are_not_same_as() {
        let a = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        let b = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        assert!(!a.same_as(&b));
    }

    #[test]
    fn union_with_shares_listitem_after_merge() {
        // bookkeeper=Some(...) → dont_change_any_more stays False, so
        // merge proceeds without TooLateForChange.
        let a = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        assert!(!a.same_as(&b));
        a.union_with(&b).expect("merge must succeed");
        assert!(a.same_as(&b));
    }

    #[test]
    fn merge_refuses_under_side_effect_free_guard() {
        let a = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );

        let _guard = SideEffectFreeGuard::enter();
        assert!(a.union_with(&b).is_err());
        assert!(!a.same_as(&b));
    }

    #[test]
    fn dont_change_any_more_merge_widening_errors() {
        // upstream: merging two final listitems with different element
        // types triggers TooLateForChange.
        let a = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let err = a
            .union_with(&b)
            .expect_err("widening final list must error");
        assert!(err.msg.contains("TooLateForChange"));
    }

    #[test]
    fn merge_range_step_matches_upstream_table() {
        // upstream `_step_map`:
        //   (NoneType, int) → None
        //   (int, NoneType) → None
        //   (int, int)      → 0
        assert_eq!(merge_range_step(None, Some(2)), None);
        assert_eq!(merge_range_step(Some(2), None), None);
        assert_eq!(merge_range_step(Some(2), Some(3)), Some(0));
        // Same-value / double-None fall back to self.
        assert_eq!(merge_range_step(None, None), None);
    }

    #[test]
    fn notify_update_skips_when_annotator_unwired() {
        // Upstream: reflow is predicated on `self.bookkeeper.annotator`
        // being a live reference. The Rust port holds it as
        // `Weak<RPythonAnnotator>` so tests with a bare `Bookkeeper::new`
        // (no annotator) must silently skip — not panic, not hang.
        let mut item = ListItem::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
        );
        let pk = super::super::bookkeeper::PositionKey::new(1, 2, 3);
        item.read_locations.insert(pk);
        item.notify_update();
    }

    #[test]
    fn notify_update_reflows_position_keys_through_annotator_backlink() {
        // Build an RPythonAnnotator so bk.annotator upgrades to a live
        // Rc. Seed a block into annotated (as Some(None) = "awaiting
        // flowin"), register it in all_blocks, then call notify_update
        // on a ListItem whose read_locations points at that block's
        // PositionKey. Verify reflowpendingblock queued the block.
        use super::super::annrpython::RPythonAnnotator;
        use super::super::bookkeeper::PositionKey;
        use crate::flowspace::model::{Block, BlockKey, FunctionGraph};
        use std::cell::RefCell;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Rc::new(RefCell::new(Block::new(vec![])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", block.clone())));
        let bkey = BlockKey::of(&block);
        // Pre-seed annotator tables so reflowpendingblock's asserts pass.
        ann.annotated.borrow_mut().insert(bkey.clone(), None);
        ann.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), block.clone());
        let pk = PositionKey::from_refs(&graph, &block, 0);

        let mut item = ListItem::new(
            Some(ann.bookkeeper.clone()),
            SomeValue::Integer(SomeInteger::new(true, false)),
        );
        item.read_locations.insert(pk);
        item.notify_update();

        // reflowpendingblock -> schedulependingblock inserts into
        // genpendingblocks[generation]. Verify the block landed there.
        let pending = ann.genpendingblocks.borrow();
        assert!(pending[0].contains_key(&bkey));
    }
}
