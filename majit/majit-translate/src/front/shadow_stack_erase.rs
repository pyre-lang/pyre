//! Take the shadow-stack root bracket out of a body before it is lowered.
//!
//! Upstream never puts a root bracket in a jitcode.  `gc_push_roots` /
//! `gc_pop_roots` are genop'd by `ShadowStackFrameworkGCTransformer`
//! (`memory/gctransform/shadowstack.py` `push_roots` / `pop_roots`), and that
//! transformer runs out of the C backend's database long after
//! `warmspot.py` `make_jitcodes` has read the graphs.  pyre spells the bracket
//! in interpreter source, so this pass removes it from what the codewriter
//! reads, the way the gctransformer's absence does upstream.  The native build
//! keeps executing the source bracket unchanged.
//!
//! The bracket is scalar-replaced: every slot a body publishes becomes one
//! synthetic MIR local.  A pin writes the slot local and answers the value it
//! was handed, a read-back answers the slot local, a `set` rewrites it, and the
//! opener, `base()`, `shadow_stack_len()`, the normalize calls and the close
//! all disappear.  The flow-graph builder then carries each slot through the
//! body's blocks like any other local, so a merge or a loop needs nothing of
//! its own.  The references the slots held stay rooted by the jitcode's own
//! root sets (the jitframe gcmap, the blackhole's `registers_r`, the recorder).
//!
//! Scalar-replaced slots need a compile-time offset from the depth the body
//! was entered at, or from a merge whose predecessors disagreed on that
//! offset (`IdxFact::AtMerge`).  `pin_roots` / `publish` of a slice the body
//! already holds — an incoming `&[T]` parameter, or a slice local it built
//! — is the same shape with that list as the source: `get(base + k)` is
//! `getarrayitem` of the list (`cdataobj.py descr_call` / `W_CData.call`
//! passes `args_w` through; `pyframe.py Frame.__init__` livevars are that
//! list).  The publish dest is rewritten to `0` so later `base + k`
//! arithmetic names the `k`-th element, and scalar slot depth is unchanged
//! because those elements were never interned.  The walk is an abstract
//! interpretation of the stack depth first.  A callee is assumed to leave the depth where it
//! found it: a body whose own depth at `Return` is not zero is refused, so the
//! only functions that could break that assumption are the ones this pass never
//! rewrites.  A callee that *calls* such a function but still returns at
//! depth 0 (its own `RootScope` Close rewinds) is depth-neutral: it does not
//! read the caller's slots by index, and a collection inside it roots the
//! caller's replaced slots through the jitframe gcmap, the blackhole's
//! `registers_r`, and the recorder.  Observes at entry depth is a refusal
//! (every live slot is the caller's). Inside this body's Open it is
//! allowed for neutrality and for erasure (`eval_slice_index` pin then
//! `getindex_w`): the callee reads our pins, which become slot locals,
//! and it does not leave extra slots. A LeavesAbove or ParamSlots
//! callee is the same split for neutrality, but erasure still refuses
//! — after the rewrite nothing rewinds a leftover
//! (`_fix_graph_after_inlining`). A callee that *returns* the leftover
//! inside an owned guard (`body_returns_owned_scope`; the dest is the
//! nested `RootScope`, as `RootedItems::new` packages one) is the
//! nested-Open shape: the leftover is that dest, the caller drops it,
//! and ParamSlots methods of the dest read the nested bracket the
//! callee kept rather than this body's slot locals. Anything the
//! interpretation cannot model refuses the whole body, which then
//! lowers exactly as it did before.
//!
//! Unwind edges are not followed: the flow-graph builder does not lower an
//! `on_unwind` cleanup chain either, and one chain is shared by calls made at
//! different depths.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, FunDecl, Operand, Place, PlaceKind, ProjectionElem, RegularCall, Rvalue, SpanRef,
    StmtKind, SwitchTargets, TermKind, TyRef, Unstructured, UnwindKind,
};
use serde_json::{Value, json};

/// One shadow-stack operation, keyed by the callee's path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    Open,
    Close,
    Base,
    Len,
    Pin,
    Publish,
    Normalize,
    NormalizeMoved,
    Get,
    Set,
    ReloadTop,
    /// `shadow_stack_copy_range` / `_into_vec`: a Get of argument 0 for
    /// the walk, Unmodeled for erase.
    CopyRange,
    /// `RootedItems::new` / `Default`: a second Open.
    ItemsOpen,
    /// `RootedItems::push`.
    ItemsPush,
    /// `RootedItems::assert_owns_the_top`: a void debug check.
    ItemsNop,
    /// `RootedItems::len`: the modeled container length (`rlist.py` `ll_length`).
    ItemsLen,
    /// `RootedItems::is_empty`: `len == 0` (`RootedItems::is_empty`).
    ItemsIsEmpty,
    /// `RootedItems::get`: own-slot read, offset from the set's open depth.
    /// `take` is the same for the walk; erase refuses it (no Vec constructor).
    ItemsGet,
    /// `RootedItems` drop / `drop_in_place`.
    ItemsClose,
    /// Touches the stack in a way this pass does not model.
    Unmodeled,
}

/// Classify a `gc_roots` callee, and say whether it takes the guard as its
/// first argument.  `None` for a callee outside the module.
///
/// Charon names an inherent method `gc_roots::<Impl>::leaf`, so the type a
/// method belongs to is read off the call: the receiver's type for a method,
/// the destination's for the guard's constructor.
fn classify_call(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
) -> Option<(Leaf, bool)> {
    let path = callee_path(call, llbc)?;
    let segments: Vec<&str> = path.split("::").collect();
    if !segments.contains(&super::ROOT_SCOPE_MODULE) {
        return None;
    }
    let leaf = *segments.last()?;
    let names_type = |ty: &majit_charon_reader::ullbc::TyRef, name: &str| {
        super::tyref_class_root(ty, llbc).is_some_and(|root| root.rsplit("::").next() == Some(name))
    };
    let receiver_ty = match call.args.first() {
        Some(Operand::Copy(p) | Operand::Move(p)) => Some(&p.ty),
        _ => None,
    };
    let is_method = segments.contains(&super::ROOT_SCOPE_TYPE)
        || segments.contains(&super::ROOTED_ITEMS_TYPE)
        || segments.iter().any(|s| s.starts_with('<'));
    if is_method {
        if names_type(&call.dest.ty, super::ROOT_SCOPE_TYPE) && leaf == "new" {
            return Some((Leaf::Open, false));
        }
        if names_type(&call.dest.ty, "RootedItems") && matches!(leaf, "new" | "default") {
            return Some((Leaf::ItemsOpen, false));
        }
        if names_type(&call.dest.ty, "RootedOnceRef")
            || receiver_ty.is_some_and(|ty| names_type(ty, "RootedOnceRef"))
        {
            // Process-global MiniMark slot, not the thread shadow stack.
            return None;
        }
        if receiver_ty.is_some_and(|ty| names_type(ty, "RootedItems")) {
            let kind = match leaf {
                "push" => Leaf::ItemsPush,
                "len" => Leaf::ItemsLen,
                "is_empty" => Leaf::ItemsIsEmpty,
                "assert_owns_the_top" => Leaf::ItemsNop,
                "get" | "take" => Leaf::ItemsGet,
                "drop" | "drop_in_place" => Leaf::ItemsClose,
                _ => Leaf::Unmodeled,
            };
            return Some((kind, true));
        }
        if !receiver_ty.is_some_and(|ty| names_type(ty, super::ROOT_SCOPE_TYPE)) {
            // `RootStack` and the walkers: the stack itself.
            return Some((Leaf::Unmodeled, false));
        }
        let kind = match leaf {
            "base" => Leaf::Base,
            "pin_root" => Leaf::Pin,
            "get" => Leaf::Get,
            "publish" | "pin_roots" => Leaf::Publish,
            "normalize" => Leaf::Normalize,
            "normalize_moved" => Leaf::NormalizeMoved,
            "set" => Leaf::Set,
            "drop" | "drop_in_place" => Leaf::Close,
            _ => Leaf::Unmodeled,
        };
        return Some((kind, true));
    }
    let kind = match leaf {
        "push_roots" => Leaf::Open,
        "root_scope_close" => return Some((Leaf::Close, true)),
        "pin_root" => Leaf::Pin,
        "pin_roots" | "publish_roots" => Leaf::Publish,
        "normalize_roots" => Leaf::Normalize,
        "shadow_stack_len" => Leaf::Len,
        "shadow_stack_get" => Leaf::Get,
        "shadow_stack_set" => Leaf::Set,
        "reload_top_root" | "reload_top_root_at" => Leaf::ReloadTop,
        "shadow_stack_copy_range" | "shadow_stack_copy_range_into_vec" => Leaf::CopyRange,
        "mark_prebuilt_roots_dirty"
        | "prebuilt_roots_dirty"
        | "clear_prebuilt_roots_dirty"
        | "increase_root_stack_depth"
        | "root_stack_depth" => return None,
        // Everything else in the module reads or rewrites the stack itself.
        _ => Leaf::Unmodeled,
    };
    Some((kind, false))
}

/// Shadow-stack depth, or a pin/get index, relative to entry or to a merge.
///
/// `Const(d)` is today's case: `d` slots above the body's entry.
/// `AtMerge { block, k }` is the depth on entry to merge `block`, plus `k`.
/// A merge whose predecessors carry different `Const` depths takes
/// `AtMerge { block, 0 }` instead of refusing (`depth-disagrees-at-merge`).
/// Two `AtMerge` from different merges meeting again, or an `AtMerge`
/// meeting a `Const` at a block that is not that merge, is still a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum IdxFact {
    Const(usize),
    AtMerge { block: usize, k: usize },
}

impl IdxFact {
    fn add(self, n: usize) -> Self {
        match self {
            IdxFact::Const(d) => IdxFact::Const(d + n),
            IdxFact::AtMerge { block, k } => IdxFact::AtMerge { block, k: k + n },
        }
    }

    fn checked_add(self, n: usize) -> Option<Self> {
        match self {
            IdxFact::Const(d) => d.checked_add(n).map(IdxFact::Const),
            IdxFact::AtMerge { block, k } => {
                k.checked_add(n).map(|k| IdxFact::AtMerge { block, k })
            }
        }
    }

    fn checked_sub(self, n: usize) -> Option<Self> {
        match self {
            IdxFact::Const(d) => d.checked_sub(n).map(IdxFact::Const),
            IdxFact::AtMerge { block, k } => {
                k.checked_sub(n).map(|k| IdxFact::AtMerge { block, k })
            }
        }
    }

    fn as_const(self) -> Option<usize> {
        match self {
            IdxFact::Const(d) => Some(d),
            IdxFact::AtMerge { .. } => None,
        }
    }

    fn is_entry(self) -> bool {
        matches!(self, IdxFact::Const(0))
    }

    /// True when `self` is a strictly outer Open relative to `closed`.
    fn strictly_outside(self, closed: IdxFact) -> bool {
        match (self, closed) {
            (IdxFact::Const(a), IdxFact::Const(b)) => a < b,
            (IdxFact::Const(_), IdxFact::AtMerge { .. }) => true,
            (IdxFact::AtMerge { .. }, IdxFact::Const(_)) => false,
            (IdxFact::AtMerge { block: b1, k: k1 }, IdxFact::AtMerge { block: b2, k: k2 })
                if b1 == b2 =>
            {
                k1 < k2
            }
            (IdxFact::AtMerge { .. }, IdxFact::AtMerge { .. }) => false,
        }
    }

    fn close_exceeds(self, current: IdxFact) -> bool {
        match (self, current) {
            (IdxFact::Const(g), IdxFact::Const(d)) => g > d,
            (IdxFact::Const(_), IdxFact::AtMerge { .. }) => false,
            (IdxFact::AtMerge { block: b1, k: gk }, IdxFact::AtMerge { block: b2, k: dk })
                if b1 == b2 =>
            {
                gk > dk
            }
            (IdxFact::AtMerge { .. }, _) => true,
        }
    }
}

/// Pins published through one guard. `RootedItems::len` / `is_empty` read
/// this the way `rlist.py` `ll_length` reads `l.length`: only an exact count
/// is a value; a joined lower bound is not statically represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuardPins {
    Exact(usize),
    Min(usize),
}

impl GuardPins {
    fn exact(self) -> Option<usize> {
        match self {
            GuardPins::Exact(n) => Some(n),
            GuardPins::Min(_) => None,
        }
    }

    fn count(self) -> usize {
        match self {
            GuardPins::Exact(n) | GuardPins::Min(n) => n,
        }
    }

    fn plus_one(self) -> Self {
        match self {
            GuardPins::Exact(n) => GuardPins::Exact(n + 1),
            GuardPins::Min(n) => GuardPins::Min(n + 1),
        }
    }

    fn join(self, other: Self) -> Self {
        match (self, other) {
            (GuardPins::Exact(a), GuardPins::Exact(b)) if a == b => GuardPins::Exact(a),
            (a, b) => GuardPins::Min(a.count().min(b.count())),
        }
    }
}

/// One scalar-replaced slot: an absolute offset from entry, or a slot
/// local to merge `block` at relative offset `k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Slot {
    Abs(usize),
    Rel { block: usize, k: usize },
}

/// What a special local stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// A `RootScope` guard opened at this depth.
    Guard(IdxFact),
    /// A borrow of a guard.
    Alias(usize),
    /// A slot index: `Const` from entry, or `AtMerge` from a merge.
    Index(IdxFact),
    /// A checked-add pair whose `.0` is this index.
    IndexPair(IdxFact),
    /// `pin_roots` dest of an incoming `&[PyObjectRef]`
    /// (`pin_roots_incoming_slice`). Not a scalar slot: `Leaf::Len` and
    /// merge depth ignore it. `get(base + k)` is that list at `k`.
    SliceBase { slice_local: usize },
    /// `base + k` of a [`Special::SliceBase`].
    SliceIndex { slice_local: usize, k: usize },
    /// Checked-add pair whose `.0` is a [`Special::SliceIndex`].
    SliceIndexPair { slice_local: usize, k: usize },
}

fn is_slice_special(special: Special) -> bool {
    matches!(
        special,
        Special::SliceBase { .. } | Special::SliceIndex { .. } | Special::SliceIndexPair { .. }
    )
}

fn slice_of_special(special: Special) -> Option<usize> {
    match special {
        Special::SliceBase { slice_local }
        | Special::SliceIndex { slice_local, .. }
        | Special::SliceIndexPair { slice_local, .. } => Some(slice_local),
        _ => None,
    }
}

/// A call or assert this pass rewrites to a `Goto`, with the statements that
/// replace it.
struct TermRewrite {
    target: u64,
    stmts: Vec<Value>,
}

struct Plan {
    specials: HashMap<usize, Special>,
    /// Statements to drop, per block.
    removed: HashMap<usize, Vec<usize>>,
    /// Statements to insert after a statement, per block.
    inserted_after: HashMap<(usize, usize), Vec<Value>>,
    terms: HashMap<usize, TermRewrite>,
    slot_count: usize,
    slot_ty: Option<Value>,
    /// Pins published through each `RootedItems` / `RootScope` guard.
    pin_count: HashMap<usize, GuardPins>,
    /// Dense ids for [`Slot`] locals, assigned in first-seen order.
    slot_intern: HashMap<Slot, usize>,
    /// Blocks no path from the entry reaches once the rewritten calls lose
    /// their unwind edges.
    unreachable: Vec<usize>,
    /// Slice local -> (local ty, `*local` ty) for a [`Special::SliceBase`].
    slice_src: HashMap<usize, (Value, Value)>,
}

impl Plan {
    /// Drop pin-count keys for a guard closed at `closed` and every set
    /// opened inside it. Nested exclusive RootScopes (`zip_two_tuple_next`)
    /// close before they merge with a sibling that never opened them;
    /// leftover keys would refuse that join (`pin-count-disagrees-at-merge`).
    fn forget_closed_guards(&mut self, closed: IdxFact) {
        self.pin_count
            .retain(|other, _| match self.specials.get(other) {
                Some(Special::Guard(gd)) => gd.strictly_outside(closed),
                _ => false,
            });
    }

    fn forget_closed_split_pins(&self, splits: &mut [SplitPred], closed: IdxFact) {
        for s in splits {
            s.pins.retain(|other, _| match self.specials.get(other) {
                Some(Special::Guard(gd)) => gd.strictly_outside(closed),
                _ => false,
            });
        }
    }

    fn intern_slot(&mut self, n_locals: usize, slot: Slot) -> usize {
        let next = self.slot_intern.len();
        n_locals + *self.slot_intern.entry(slot).or_insert(next)
    }
}

enum WalkErr {
    Refusal(Refusal),
    NeedMerge(usize),
}

impl From<Refusal> for WalkErr {
    fn from(r: Refusal) -> Self {
        WalkErr::Refusal(r)
    }
}

fn note_merge_pred_depth(
    merge_min: &mut HashMap<usize, usize>,
    merge_max: &mut HashMap<usize, usize>,
    block: usize,
    d: usize,
) {
    merge_min
        .entry(block)
        .and_modify(|m| *m = (*m).min(d))
        .or_insert(d);
    merge_max
        .entry(block)
        .and_modify(|m| *m = (*m).max(d))
        .or_insert(d);
}

/// Known constant of a switch scrutinee. After `exitswitch vexit`,
/// `constfold.constant_diffuse` replaces `vexit` with the arm constant
/// on that link; `rewire_link_for_known_exitswitch` then follows only
/// the matching exit. This pass keeps the same fact so a later switch
/// on that local does not walk infeasible arms. Facts are keyed by the
/// canonical local, or by `SliceLen` of the slice `args.len()` /
/// `PtrMetadata` / `Len` read — two copies of the same length are one
/// key. `gctransform` `push_roots` / `pop_roots` never see the bracket
/// the JIT reads; a refused erasure here is a lowering gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FactKey {
    Local(usize),
    SliceLen(usize),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ScrutineeFacts {
    vals: BTreeMap<FactKey, u64>,
}

impl ScrutineeFacts {
    fn get(&self, key: FactKey) -> Option<u64> {
        self.vals.get(&key).copied()
    }

    fn insert(&mut self, key: FactKey, value: u64) {
        self.vals.insert(key, value);
    }

    fn remove(&mut self, key: FactKey) {
        self.vals.remove(&key);
    }

    fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }

    fn intersect(&self, other: &Self) -> Self {
        Self {
            vals: self
                .vals
                .iter()
                .filter(|(k, v)| other.vals.get(k) == Some(*v))
                .map(|(k, v)| (*k, *v))
                .collect(),
        }
    }
}

/// One predecessor's Const depth at an `AtMerge`, distinguished by its
/// scrutinee facts. Bounded: at most one entry per fact set, and a hard
/// cap (`MAX_SCRUTINEE_SPLITS`) when a new set is added. Restored at a
/// later switch on the same key so `get` sees the arm's own depth
/// (`ll_getitem_nonneg` against that arm's length).
#[derive(Clone, Debug, PartialEq, Eq)]
struct SplitPred {
    facts: ScrutineeFacts,
    depth: IdxFact,
    open: usize,
    pins: HashMap<usize, GuardPins>,
}

/// Hard cap on distinct fact sets at one merge. `funccallunrolling` is
/// `range(4)` plus the zero arm — five predecessors. Refuse rather than
/// split without bound.
const MAX_SCRUTINEE_SPLITS: usize = 8;

fn ptr_metadata_base(place: &Place) -> Option<usize> {
    let PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) = &place.kind else {
        return None;
    };
    (elem == "PtrMetadata")
        .then(|| place_local(inner).or_else(|| deref_of_local(inner)))
        .flatten()
}

fn slice_len_key(base: usize, copies: &BTreeMap<usize, FactKey>) -> FactKey {
    match copies.get(&base) {
        Some(FactKey::Local(l)) => FactKey::SliceLen(*l),
        Some(FactKey::SliceLen(s)) => FactKey::SliceLen(*s),
        None => FactKey::SliceLen(base),
    }
}

fn fact_key_of_place(place: &Place, copies: &BTreeMap<usize, FactKey>) -> Option<FactKey> {
    if let Some(base) = ptr_metadata_base(place) {
        return Some(slice_len_key(base, copies));
    }
    let l = place_local(place)?;
    Some(copies.get(&l).copied().unwrap_or(FactKey::Local(l)))
}

fn operand_fact_key(op: &Operand, copies: &BTreeMap<usize, FactKey>) -> Option<FactKey> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => fact_key_of_place(p, copies),
        Operand::Const(_) => None,
    }
}

fn rvalue_fact_key(rvalue: &Rvalue, copies: &BTreeMap<usize, FactKey>) -> Option<FactKey> {
    match rvalue {
        Rvalue::Use(op, _) => operand_fact_key(op, copies),
        Rvalue::Len(place) => {
            let base = place_local(place).or_else(|| deref_of_local(place))?;
            Some(slice_len_key(base, copies))
        }
        _ => None,
    }
}

fn place_root_local(place: &Place) -> Option<usize> {
    place_local(place)
        .or_else(|| deref_of_local(place))
        .or_else(|| tuple_field_of_local(place).map(|(l, _)| l))
}

fn prune_splits(splits: &mut Vec<SplitPred>) {
    splits.retain(|s| !s.facts.is_empty());
    if splits.len() >= 2 {
        let first = splits[0].facts.clone();
        if splits.iter().all(|s| s.facts == first) {
            splits.clear();
        }
    }
}

fn kill_local_facts(
    facts: &mut ScrutineeFacts,
    splits: &mut Vec<SplitPred>,
    copies: &mut BTreeMap<usize, FactKey>,
    local: usize,
) {
    let key = copies.remove(&local).unwrap_or(FactKey::Local(local));
    facts.remove(key);
    facts.remove(FactKey::Local(local));
    facts.remove(FactKey::SliceLen(local));
    for s in splits.iter_mut() {
        s.facts.remove(key);
        s.facts.remove(FactKey::Local(local));
        s.facts.remove(FactKey::SliceLen(local));
    }
    prune_splits(splits);
}

fn add_to_depth(depth: &mut IdxFact, splits: &mut [SplitPred], n: usize) {
    *depth = depth.add(n);
    for s in splits {
        s.depth = s.depth.add(n);
    }
}

fn rewind_depth(depth: &mut IdxFact, splits: &mut [SplitPred], d: IdxFact) {
    // Nested Open does not change scalar depth, so Close of that guard
    // must not replace a Const split snapshot with the walk's AtMerge
    // (`rewind` of the live depth only). Subtract the same slot delta
    // from each Const snapshot when the walk actually drops slots.
    let delta = match (*depth, d) {
        (IdxFact::Const(a), IdxFact::Const(b)) if a >= b => a - b,
        (
            IdxFact::AtMerge {
                block: b1, k: k1, ..
            },
            IdxFact::AtMerge {
                block: b2, k: k2, ..
            },
        ) if b1 == b2 && k1 >= k2 => k1 - k2,
        (IdxFact::AtMerge { k, .. }, IdxFact::Const(_)) => k,
        _ => 0,
    };
    *depth = d;
    if delta == 0 {
        return;
    }
    for s in splits {
        if let Some(nd) = s.depth.checked_sub(delta) {
            s.depth = nd;
        }
    }
}

fn add_split_pred(splits: &mut Vec<SplitPred>, neu: SplitPred) -> Result<(), WalkErr> {
    // One state per Const depth at the merge: extra switch facts that
    // do not change the pin count (an assert If, a bounds-check) are
    // intersected away so the product of unrelated keys cannot exceed
    // the predecessor cap (`constfold.constant_diffuse` keeps one
    // constant per link; depths are the distinguishing piece here).
    if let Some(old) = splits.iter_mut().find(|s| s.depth == neu.depth) {
        old.facts = old.facts.intersect(&neu.facts);
        return Ok(());
    }
    if splits.iter().any(|s| s.facts == neu.facts) {
        return Err(WalkErr::Refusal("depth-disagrees-at-merge"));
    }
    if splits.len() >= MAX_SCRUTINEE_SPLITS {
        return Err(WalkErr::Refusal("scrutinee-split-unbounded"));
    }
    splits.push(neu);
    Ok(())
}

fn union_splits(
    mut acc: Vec<SplitPred>,
    incoming: &[SplitPred],
) -> Result<Vec<SplitPred>, WalkErr> {
    for neu in incoming {
        add_split_pred(&mut acc, neu.clone())?;
    }
    Ok(acc)
}

fn intersect_copies(
    a: &BTreeMap<usize, FactKey>,
    b: &BTreeMap<usize, FactKey>,
) -> BTreeMap<usize, FactKey> {
    a.iter()
        .filter(|(k, v)| b.get(k) == Some(*v))
        .map(|(k, v)| (*k, *v))
        .collect()
}

fn switch_arm_value(v: &Value) -> Option<u64> {
    if let Some(b) = v.get("Bool").and_then(Value::as_bool) {
        return Some(u64::from(b));
    }
    let inner = v.get("Scalar").unwrap_or(v);
    let inner = inner.get("Integer").unwrap_or(inner);
    let lit = inner.get("Unsigned").or_else(|| inner.get("Signed"))?;
    let s = lit.as_array()?.get(1)?.as_str()?;
    s.parse().ok()
}

fn switch_known_target(targets: &SwitchTargets, value: u64) -> u64 {
    match targets {
        SwitchTargets::If(then_bb, else_bb) => {
            if value != 0 {
                *then_bb
            } else {
                *else_bb
            }
        }
        SwitchTargets::SwitchInt(_, arms, default) => {
            for (scalar, t) in arms {
                if switch_arm_value(scalar) == Some(value) {
                    return *t;
                }
            }
            *default
        }
    }
}

struct SwitchSuccessor {
    target: u64,
    value: Option<u64>,
}

fn switch_successors(targets: &SwitchTargets) -> Vec<SwitchSuccessor> {
    match targets {
        SwitchTargets::If(then_bb, else_bb) => vec![
            SwitchSuccessor {
                target: *then_bb,
                value: Some(1),
            },
            SwitchSuccessor {
                target: *else_bb,
                value: Some(0),
            },
        ],
        SwitchTargets::SwitchInt(_, arms, default) => {
            let mut out: Vec<SwitchSuccessor> = arms
                .iter()
                .map(|(scalar, t)| SwitchSuccessor {
                    target: *t,
                    value: switch_arm_value(scalar),
                })
                .collect();
            // A default that aliases a valued arm is the same path; walking
            // it again with no fact would intersect the arm's fact to empty
            // (`constant_diffuse` keeps one constant per link).
            if !arms.iter().any(|(_, t)| t == default) {
                out.push(SwitchSuccessor {
                    target: *default,
                    value: None,
                });
            }
            out
        }
    }
}

fn term_forward_targets(term: &TermKind) -> Vec<u64> {
    match term {
        TermKind::Call { target, .. }
        | TermKind::Drop { target, .. }
        | TermKind::Assert { target, .. }
        | TermKind::Goto { target } => vec![*target],
        TermKind::Switch { targets, .. } => switch_successors(targets)
            .into_iter()
            .map(|s| s.target)
            .collect(),
        _ => Vec::new(),
    }
}

fn forward_pred_counts(body: &Unstructured, llbc: &Llbc, forward_blocks: &[bool]) -> Vec<usize> {
    let n = body.body.len();
    let mut counts = vec![0usize; n];
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        let Ok(term) = block.term_ref(llbc) else {
            continue;
        };
        for t in term_forward_targets(&term) {
            if let Some(c) = counts.get_mut(t as usize) {
                *c = c.saturating_add(1);
            }
        }
    }
    counts
}

/// Same-key pin maps at an `AtMerge` join keep the per-guard minimum.
/// Agreeing `Exact` counts stay exact; disagreeing counts become `Min`.
/// Different keys still refuse (`pin-count-disagrees-at-merge`).
fn join_pin_counts_min(
    a: &HashMap<usize, GuardPins>,
    b: &HashMap<usize, GuardPins>,
) -> Option<HashMap<usize, GuardPins>> {
    if a == b {
        return Some(a.clone());
    }
    if a.len() != b.len() {
        return None;
    }
    let mut out = HashMap::with_capacity(a.len());
    for (k, va) in a {
        let vb = b.get(k)?;
        out.insert(*k, va.join(*vb));
    }
    Some(out)
}

/// Slot a Get/Set of `index` reads at `depth`.
///
/// An absolute `Const(j)` at an `AtMerge` is `Slot::Abs(j)` only when
/// `j < merge_min[block]`. `ll_getitem_nonneg` compares against the actual
/// list length (`index >= l.ll_length()` raises); the length proven on
/// every predecessor is that minimum. `min <= j` is a slot some
/// predecessor never wrote (`read-not-defined-on-all-predecessors`).
/// Relative slots (`AtMerge` index below the current merge depth) are
/// created after the merge and do not read that path-dependent suffix.
fn slot_for_get(
    index: IdxFact,
    depth: IdxFact,
    merge_min: &HashMap<usize, usize>,
) -> Result<Slot, Refusal> {
    match index {
        IdxFact::Const(j) => match depth {
            IdxFact::Const(d) if j < d => Ok(Slot::Abs(j)),
            IdxFact::Const(_) => Err("get-above-depth"),
            IdxFact::AtMerge { block, .. } => {
                let min = *merge_min.get(&block).ok_or("get-above-depth")?;
                if j < min {
                    Ok(Slot::Abs(j))
                } else {
                    Err("read-not-defined-on-all-predecessors")
                }
            }
        },
        IdxFact::AtMerge { block: ib, k: ik } => match depth {
            IdxFact::AtMerge { block: db, k: dk } if ib == db && ik < dk => {
                Ok(Slot::Rel { block: ib, k: ik })
            }
            _ => Err("get-above-depth"),
        },
    }
}

fn slots_from_depth(depth: IdxFact, n: usize) -> impl Iterator<Item = Slot> {
    (0..n).map(move |i| match depth {
        IdxFact::Const(d) => Slot::Abs(d + i),
        IdxFact::AtMerge { block, k } => Slot::Rel { block, k: k + i },
    })
}

type Refusal = &'static str;

fn place_local(place: &Place) -> Option<usize> {
    match place.kind {
        PlaceKind::Local(l) => Some(l as usize),
        _ => None,
    }
}

fn operand_local(op: &Operand) -> Option<usize> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => place_local(p),
        Operand::Const(_) => None,
    }
}

/// The base local of `*local`.
fn deref_of_local(place: &Place) -> Option<usize> {
    let PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) = &place.kind else {
        return None;
    };
    (elem == "Deref").then(|| place_local(inner)).flatten()
}

/// `local.0` / `local.1` of a tuple, or `(*local).0` of a borrow.
fn tuple_field_of_local(place: &Place) -> Option<(usize, u64)> {
    let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind else {
        return None;
    };
    let field = elem.get("Field")?.as_array()?;
    let index = field.get(1)?.as_u64()?;
    let base = place_local(inner).or_else(|| deref_of_local(inner))?;
    Some((base, index))
}

/// Whether a `Ref` / `RawPtr` kind writes through the borrowed place.
fn ref_kind_mutates(kind: &Value) -> bool {
    match kind {
        Value::String(s) => s.contains("Mut"),
        Value::Object(m) => m.keys().any(|k| k.contains("Mut")),
        _ => false,
    }
}

/// Assignments of each local: `Assign` destinations, call destinations,
/// and `&mut` / `*mut` borrows of a local. `StorageLive` / `StorageDead`
/// are not definitions. Only blocks in `forward_blocks` are counted: an
/// `on_unwind` cleanup block is still in the body.
fn local_def_counts(body: &Unstructured, llbc: &Llbc, forward_blocks: &[bool]) -> Vec<u32> {
    let n = body.locals.locals.len();
    let mut counts = vec![0u32; n];
    let bump = |counts: &mut [u32], l: Option<usize>| {
        if let Some(l) = l
            && let Some(c) = counts.get_mut(l)
        {
            *c = c.saturating_add(1);
        }
    };
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        for stmt in &block.statements {
            let Ok(kind) = stmt.stmt_kind_ref() else {
                continue;
            };
            match kind {
                StmtKind::Assign(place, rvalue) => {
                    bump(&mut counts, place_local(place));
                    match rvalue {
                        Rvalue::Ref {
                            place: src, kind, ..
                        }
                        | Rvalue::RawPtr {
                            place: src, kind, ..
                        } if ref_kind_mutates(kind) => {
                            bump(
                                &mut counts,
                                place_local(src).or_else(|| deref_of_local(src)),
                            );
                        }
                        _ => {}
                    }
                }
                StmtKind::StorageLive(_) | StmtKind::StorageDead(_) => {}
                _ => {}
            }
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc) {
            bump(&mut counts, place_local(&call.dest));
        }
    }
    counts
}

/// An unsigned scalar constant operand.
fn const_usize(op: &Operand, llbc: &Llbc) -> Option<usize> {
    let Operand::Const(v) = op else {
        return None;
    };
    let lit = llbc.const_expr_literal(v)?;
    let scalar = lit.get("Scalar")?;
    let lit = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
    lit.as_array()?.get(1)?.as_str()?.parse().ok()
}

/// Whether `base + x` of a published slice can use the add as a runtime
/// element index (`Get` of leftover-slice `k == 0`). A `const` operand, a
/// `PlaceKind::Global` (`FrameBox::new` `inputs + FRAME_BOX_DEBUG_INPUT`),
/// or a local this body assigned. A parameter / return place stays
/// `index-not-static`.
fn slice_addend_is_computed(op: &Operand, stmt_assigned: &HashMap<usize, usize>) -> bool {
    match op {
        Operand::Const(_) => true,
        Operand::Copy(place) | Operand::Move(place) => match place.kind {
            PlaceKind::Local(l) => stmt_assigned.get(&(l as usize)).copied().unwrap_or(0) >= 1,
            PlaceKind::Global { .. } => true,
            _ => false,
        },
    }
}

/// A compile-time slot-index `+ k` or `- k`. `"*Checked"` yields a pair;
/// `"Add"` / `"Sub"` / `{"Add": "Wrap"}` / `{"Sub": "Wrap"}` a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexArith {
    Add { checked: bool },
    Sub { checked: bool },
}

fn binop_is_index_arith(op: &Value) -> Option<IndexArith> {
    match op {
        Value::String(s) if s == "AddChecked" => Some(IndexArith::Add { checked: true }),
        Value::String(s) if s == "Add" => Some(IndexArith::Add { checked: false }),
        Value::Object(m) if m.contains_key("Add") => Some(IndexArith::Add { checked: false }),
        Value::String(s) if s == "SubChecked" => Some(IndexArith::Sub { checked: true }),
        Value::String(s) if s == "Sub" => Some(IndexArith::Sub { checked: false }),
        Value::Object(m) if m.contains_key("Sub") => Some(IndexArith::Sub { checked: false }),
        _ => None,
    }
}

fn copy_operand_json(op: &Value) -> Value {
    match op.as_object() {
        Some(m) if m.contains_key("Move") => json!({ "Copy": m["Move"].clone() }),
        _ => op.clone(),
    }
}

fn local_place_json(local: usize, ty: &Value) -> Value {
    json!({ "kind": { "Local": local }, "ty": ty })
}

fn assign_json(dest: Value, rvalue: Value) -> Value {
    json!({ "Assign": [dest, rvalue] })
}

/// `(*slice)[index]` — `get(base + k)` of a published incoming slice.
fn slice_index_use(
    slice: usize,
    slice_ty: &Value,
    deref_ty: &Value,
    index_op: Value,
    elem_ty: &Value,
) -> Value {
    json!({
        "Use": [{
            "Copy": {
                "kind": {"Projection": [
                    {
                        "kind": {"Projection": [
                            {"kind": {"Local": slice}, "ty": slice_ty.clone()},
                            "Deref"
                        ]},
                        "ty": deref_ty.clone()
                    },
                    {"Index": {"offset": index_op, "from_end": false}}
                ]},
                "ty": elem_ty.clone()
            }
        }, "Yes"]
    })
}

/// Like [`super::mentions_local`], but skip an assertion's `check_kind`.
/// That field is panic-message metadata; the flow-graph builder reads
/// only `cond`.
fn mentions_local_skipping_check_kind(v: &Value, wanted: &bit_set::BitSet) -> bool {
    match v {
        Value::Object(map) => {
            if let Some(local) = map.get("Local").and_then(Value::as_u64)
                && wanted.contains(local as usize)
            {
                return true;
            }
            map.iter()
                .filter(|(key, _)| key.as_str() != "check_kind")
                .any(|(_, nested)| mentions_local_skipping_check_kind(nested, wanted))
        }
        Value::Array(items) => items
            .iter()
            .any(|nested| mentions_local_skipping_check_kind(nested, wanted)),
        _ => false,
    }
}

fn specials_in_json(v: &Value, specials: &HashMap<usize, Special>) -> Vec<Special> {
    let mut out = Vec::new();
    fn walk(v: &Value, specials: &HashMap<usize, Special>, out: &mut Vec<Special>) {
        match v {
            Value::Object(map) => {
                if let Some(local) = map.get("Local").and_then(Value::as_u64)
                    && let Some(s) = specials.get(&(local as usize))
                {
                    out.push(*s);
                }
                for nested in map.values() {
                    walk(nested, specials, out);
                }
            }
            Value::Array(items) => {
                for nested in items {
                    walk(nested, specials, out);
                }
            }
            _ => {}
        }
    }
    walk(v, specials, &mut out);
    out
}

/// Overflow bit of an erased `AddChecked` (`IndexPair.1`). The add is a
/// compile-time slot offset, so the overflow flag is false by construction.
fn assert_is_erased_index_overflow(
    assert: &majit_charon_reader::ullbc::AssertStmt,
    specials: &HashMap<usize, Special>,
) -> bool {
    let pair = match &assert.cond {
        Operand::Copy(p) | Operand::Move(p) => tuple_field_of_local(p),
        Operand::Const(_) => None,
    };
    matches!(
        pair.and_then(|(pair, field)| (field == 1).then_some(specials.get(&pair))),
        Some(Some(Special::IndexPair(_) | Special::SliceIndexPair { .. }))
    )
}

/// `BoundsCheck { len, index }` in Charon's `AssertKind` payload.
fn bounds_check_len_and_index(check_kind: &Value) -> Option<(&Value, &Value)> {
    let bc = check_kind.get("BoundsCheck")?;
    if let Some(map) = bc.as_object() {
        return Some((map.get("len")?, map.get("index")?));
    }
    let arr = bc.as_array()?;
    (arr.len() >= 2).then_some((&arr[0], &arr[1]))
}

/// Compile-time slot index named by an operand JSON node.
fn json_index_fact(v: &Value, specials: &HashMap<usize, Special>) -> Option<IdxFact> {
    match specials_in_json(v, specials).as_slice() {
        [Special::Index(k)] | [Special::IndexPair(k)] => Some(*k),
        [Special::SliceBase { .. }] => Some(IdxFact::Const(0)),
        [Special::SliceIndex { k, .. } | Special::SliceIndexPair { k, .. }] => {
            Some(IdxFact::Const(*k))
        }
        _ => None,
    }
}

/// Compile-time slot offset named by an operand JSON node.
fn json_index_offset(v: &Value, specials: &HashMap<usize, Special>) -> Option<usize> {
    json_index_fact(v, specials).and_then(IdxFact::as_const)
}

/// Length of the erased slot array: a constant, or `shadow_stack_len`
/// (an `Index` special holding the current depth).
fn json_slot_array_len(
    v: &Value,
    specials: &HashMap<usize, Special>,
    const_locals: &HashMap<usize, usize>,
    llbc: &Llbc,
) -> Option<IdxFact> {
    if let Some(n) = json_const_usize(v, llbc) {
        return Some(IdxFact::Const(n));
    }
    if let Some(local) = json_operand_local(v)
        && let Some(n) = const_locals.get(&local)
    {
        return Some(IdxFact::Const(*n));
    }
    match specials_in_json(v, specials).as_slice() {
        [Special::Index(d)] | [Special::IndexPair(d)] => Some(*d),
        _ => None,
    }
}

fn index_fits_len(index: IdxFact, len: IdxFact, merge_min: &HashMap<usize, usize>) -> bool {
    match (index, len) {
        (IdxFact::Const(j), IdxFact::Const(n)) => j < n,
        (IdxFact::AtMerge { block: b1, k: ik }, IdxFact::AtMerge { block: b2, k: lk })
            if b1 == b2 =>
        {
            ik < lk
        }
        (IdxFact::Const(j), IdxFact::AtMerge { block, k: lk }) => {
            let min = merge_min.get(&block).copied().unwrap_or(0);
            j < min.saturating_add(lk)
        }
        (IdxFact::AtMerge { k: ik, .. }, IdxFact::Const(n)) => ik < n,
        _ => false,
    }
}

fn json_operand_local(v: &Value) -> Option<usize> {
    v.get("Copy")
        .or_else(|| v.get("Move"))
        .and_then(|p| p.get("kind"))
        .and_then(|k| k.get("Local"))
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

fn json_const_usize(v: &Value, llbc: &Llbc) -> Option<usize> {
    let lit = llbc.const_expr_literal(v.get("Const").unwrap_or(v))?;
    let scalar = lit.get("Scalar")?;
    let lit = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
    lit.as_array()?.get(1)?.as_str()?.parse().ok()
}

/// Assert that is true by construction of the erased bracket: the overflow
/// bit of `len + k`, or a `BoundsCheck` whose index is an erased slot
/// offset and whose length is that same slot array's length (`index < len`).
/// Anything else keeps the bracket.
fn assert_is_erased_index_add_overflow(
    assert: &majit_charon_reader::ullbc::AssertStmt,
    specials: &HashMap<usize, Special>,
) -> bool {
    if assert.expected {
        return false;
    }
    let Some(arr) = assert.check_kind.get("Overflow").and_then(Value::as_array) else {
        return false;
    };
    let is_index_arith = match arr.first() {
        Some(Value::String(s)) => {
            s == "Add" || s == "AddChecked" || s == "Sub" || s == "SubChecked"
        }
        Some(Value::Object(m)) => m.contains_key("Add") || m.contains_key("Sub"),
        _ => false,
    };
    if !is_index_arith || arr.len() < 3 {
        return false;
    }
    // A compile-time slot offset plus or minus a small usize cannot
    // overflow a machine word: the offsets this pass binds are the
    // published slot count, and a `len - k` it accepted already
    // proved `k` fits.
    json_index_offset(&arr[1], specials).is_some() || json_index_offset(&arr[2], specials).is_some()
}

fn assert_is_slot_index_bounds_check(
    assert: &majit_charon_reader::ullbc::AssertStmt,
    specials: &HashMap<usize, Special>,
    const_locals: &HashMap<usize, usize>,
    llbc: &Llbc,
    merge_min: &HashMap<usize, usize>,
) -> bool {
    if assert_is_erased_index_overflow(assert, specials)
        || assert_is_erased_index_add_overflow(assert, specials)
    {
        return true;
    }
    if !assert.expected {
        return false;
    }
    let Some((len, index)) = bounds_check_len_and_index(&assert.check_kind) else {
        return false;
    };
    let Some(k) = json_index_fact(index, specials) else {
        return false;
    };
    json_slot_array_len(len, specials, const_locals, llbc)
        .is_some_and(|n| index_fits_len(k, n, merge_min))
}

/// How a closure body names the slot it reads.
///
/// `Capture` / `CapturePlusArg` are [`super::function_is_root_slot_getter`].
/// `CapturePlusConst` is `|| shadow_stack_get(captured + k)`: the addend is a
/// literal in the body, which that helper traces as opaque.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotGetter {
    Capture,
    CapturePlusArg,
    CapturePlusConst(u64),
}

/// Classify a callee as a root-slot getter.  Reuses
/// [`super::function_is_root_slot_getter`] for the two kinds `RootBracketPlan`
/// already answers, then ports `classify_root_slot_getter_body` for a
/// `captured + const` body.
fn slot_getter_of(
    llbc: &Llbc,
    fun_id: u64,
    cache: &mut HashMap<u64, Option<SlotGetter>>,
) -> Option<SlotGetter> {
    if let Some(&cached) = cache.get(&fun_id) {
        return cached;
    }
    let name_of = |reg: &RegularCall| super::regular_call_name_path(reg, llbc);
    let got = match super::function_is_root_slot_getter(llbc, fun_id, &name_of) {
        Some(super::RootSlotGetter::Capture) => Some(SlotGetter::Capture),
        Some(super::RootSlotGetter::CapturePlusArg) => Some(SlotGetter::CapturePlusArg),
        None => classify_capture_plus_const(llbc, fun_id, &name_of),
    };
    cache.insert(fun_id, got);
    got
}

/// `classify_root_slot_getter_body` with a `Const` usize treated as an addend
/// rather than an opaque leaf, so `|| shadow_stack_get(base + 1)` is a Get of
/// slot `k + 1`.
fn classify_capture_plus_const(
    llbc: &Llbc,
    fun_id: u64,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<SlotGetter> {
    let fd = llbc.fn_by_id(fun_id)?;
    let raw = fd.body.as_ref()?;
    if raw.get().len() > 16 * 1024 {
        return None;
    }
    let body = fd.unstructured()?;
    if body.body.len() > 8 || body.locals.arg_count != 2 {
        return None;
    }
    let mut defs: HashMap<usize, Rvalue> = HashMap::new();
    let mut getter_arg: Option<Operand> = None;
    let mut getter_call: Option<(usize, usize)> = None;
    let mut calls = 0usize;
    let forward_blocks = super::forward_reachable_mask(llbc, &body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref().ok()? {
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::Borrowck(_)
                | StmtKind::PlaceMention(_)
                | StmtKind::Assert(_) => {}
                StmtKind::Assign(place, value) => {
                    let PlaceKind::Local(dest) = place.kind else {
                        return None;
                    };
                    let dest = dest as usize;
                    if dest <= 2 || defs.insert(dest, value.clone()).is_some() {
                        return None;
                    }
                    if !super::root_slot_rvalue_is_traceable(value) {
                        return None;
                    }
                }
                StmtKind::Unknown => return None,
            }
        }
        let term = serde_json::from_str::<TermKind>(bb.terminator.kind.get()).ok()?;
        match term {
            TermKind::Assert { .. }
            | TermKind::Goto { .. }
            | TermKind::Return
            | TermKind::UnwindResume
            | TermKind::UnwindTerminate => {}
            TermKind::Call { call, target, .. } => {
                calls += 1;
                if calls > 1 {
                    return None;
                }
                let CallFunc::Regular(reg) = &call.func else {
                    return None;
                };
                let path = name_of(reg)?;
                let leaf = path.rsplit("::").next().unwrap_or("");
                if leaf != "shadow_stack_get"
                    || !path
                        .split("::")
                        .any(|segment| segment == super::ROOT_SCOPE_MODULE)
                    || call.args.len() != 1
                    || !matches!(call.dest.kind, PlaceKind::Local(0))
                {
                    return None;
                }
                getter_call = Some((bb_idx, target as usize));
                getter_arg = call.args.into_iter().next();
            }
            TermKind::Drop { .. }
            | TermKind::Switch { .. }
            | TermKind::Abort(_)
            | TermKind::Panic { .. }
            | TermKind::UndefinedBehavior
            | TermKind::Unknown => return None,
        }
    }
    let (getter_bb, normal_target) = getter_call?;
    if !super::getter_returns_follow_call(llbc, &body, getter_bb, normal_target) {
        return None;
    }
    let trace = const_slot_trace_operand(
        llbc,
        &defs,
        getter_arg.as_ref()?,
        &mut bit_set::BitSet::new(),
    );
    if trace.opaque || trace.env != 1 || trace.arg != 0 || trace.adds != 1 {
        return None;
    }
    Some(SlotGetter::CapturePlusConst(trace.const_sum))
}

#[derive(Clone, Copy)]
struct ConstSlotTrace {
    env: u32,
    arg: u32,
    adds: u32,
    const_sum: u64,
    opaque: bool,
}

impl ConstSlotTrace {
    fn env_leaf() -> Self {
        Self {
            env: 1,
            arg: 0,
            adds: 0,
            const_sum: 0,
            opaque: false,
        }
    }

    fn arg_leaf() -> Self {
        Self {
            env: 0,
            arg: 1,
            adds: 0,
            const_sum: 0,
            opaque: false,
        }
    }

    fn const_leaf(n: u64) -> Self {
        Self {
            env: 0,
            arg: 0,
            adds: 0,
            const_sum: n,
            opaque: false,
        }
    }

    fn opaque() -> Self {
        Self {
            env: 0,
            arg: 0,
            adds: 0,
            const_sum: 0,
            opaque: true,
        }
    }

    fn join_add(self, other: Self) -> Self {
        Self {
            env: self.env.saturating_add(other.env),
            arg: self.arg.saturating_add(other.arg),
            adds: self.adds.saturating_add(other.adds).saturating_add(1),
            const_sum: self.const_sum.saturating_add(other.const_sum),
            opaque: self.opaque || other.opaque,
        }
    }
}

fn const_slot_trace_operand(
    llbc: &Llbc,
    defs: &HashMap<usize, Rvalue>,
    op: &Operand,
    visiting: &mut bit_set::BitSet,
) -> ConstSlotTrace {
    match op {
        Operand::Copy(place) | Operand::Move(place) => {
            const_slot_trace_place(llbc, defs, place, visiting)
        }
        Operand::Const(_) => match const_usize(op, llbc) {
            Some(n) => ConstSlotTrace::const_leaf(n as u64),
            None => ConstSlotTrace::opaque(),
        },
    }
}

fn const_slot_trace_place(
    llbc: &Llbc,
    defs: &HashMap<usize, Rvalue>,
    place: &Place,
    visiting: &mut bit_set::BitSet,
) -> ConstSlotTrace {
    let Some(root) = super::place_root_through_deref_or_field(place) else {
        return ConstSlotTrace::opaque();
    };
    const_slot_trace_local(llbc, defs, root, visiting)
}

fn const_slot_trace_local(
    llbc: &Llbc,
    defs: &HashMap<usize, Rvalue>,
    local: usize,
    visiting: &mut bit_set::BitSet,
) -> ConstSlotTrace {
    if local == 1 {
        return ConstSlotTrace::env_leaf();
    }
    if local == 2 {
        return ConstSlotTrace::arg_leaf();
    }
    if !visiting.insert(local) {
        return ConstSlotTrace::opaque();
    }
    let Some(value) = defs.get(&local) else {
        visiting.remove(local);
        return ConstSlotTrace::opaque();
    };
    let trace = match value {
        Rvalue::Use(op, _) => const_slot_trace_operand(llbc, defs, op, visiting),
        Rvalue::BinaryOp(op, lhs, rhs) if super::root_slot_sum_is_checked(op).is_some() => {
            const_slot_trace_operand(llbc, defs, lhs, visiting)
                .join_add(const_slot_trace_operand(llbc, defs, rhs, visiting))
        }
        _ => ConstSlotTrace::opaque(),
    };
    visiting.remove(local);
    trace
}

/// What a rewritten getter call answers.
enum GetterAnswer {
    /// `|| shadow_stack_get(captured)` of a scalar slot.
    Scalar(IdxFact),
    /// `|| shadow_stack_get(captured + k)` of an incoming slice
    /// (`pin_roots_incoming_slice`).
    Slice { slice_local: usize, k: usize },
}

/// Peel a closure call of a [`SlotGetter`] down to the static slot index and
/// the temporaries that only build the capture (`peel_closure_capture`,
/// `peel_captured_index`, `peel_getter_addend`).
fn try_slot_getter_call(
    body: &Unstructured,
    llbc: &Llbc,
    call: &majit_charon_reader::ullbc::CallPayload,
    specials: &HashMap<usize, Special>,
    assigned: &HashMap<usize, usize>,
    call_dests: &bit_set::BitSet,
    forward_blocks: &[bool],
    cache: &mut HashMap<u64, Option<SlotGetter>>,
) -> Option<(GetterAnswer, Vec<usize>)> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let fun_id = super::regular_call_fun_decl_id(&reg.kind)?;
    let kind = slot_getter_of(llbc, fun_id, cache)?;
    let receiver = call.args.first()?;
    let mut bases = HashMap::new();
    let mut candidates = bit_set::BitSet::new();
    for (local, spec) in specials {
        if matches!(
            spec,
            Special::Index(_)
                | Special::IndexPair(_)
                | Special::SliceBase { .. }
                | Special::SliceIndex { .. }
                | Special::SliceIndexPair { .. }
        ) {
            bases.insert(*local, 0);
            candidates.insert(0);
        }
    }
    if bases.is_empty() {
        return None;
    }
    let offsets: HashMap<usize, super::RootSlotSum> = HashMap::new();
    let len_index: HashMap<usize, (usize, usize)> = HashMap::new();
    let aliases: HashMap<usize, usize> = HashMap::new();
    let index = super::RootGetterIndex {
        assigned,
        bases: &bases,
        offsets: &offsets,
        len_index: &len_index,
        candidates: &candidates,
        aliases: &aliases,
        call_dests,
        forward_blocks,
    };
    let peeled = match super::peel_closure_capture(llbc, body, &index, receiver) {
        super::CapturePeel::Ready(peeled) => peeled,
        _ => return None,
    };
    let (index_local, _, index_carriers) =
        super::peel_captured_index(llbc, body, &index, &peeled.capture)?;
    let peel_kind = match kind {
        SlotGetter::Capture | SlotGetter::CapturePlusConst(_) => super::RootSlotGetter::Capture,
        SlotGetter::CapturePlusArg => super::RootSlotGetter::CapturePlusArg,
    };
    let addend = super::peel_getter_addend(llbc, body, &index, peel_kind, &call.args)?;
    let body_addend = match kind {
        SlotGetter::CapturePlusConst(n) => n,
        _ => 0,
    };
    let extra = usize::try_from(body_addend)
        .ok()?
        .checked_add(usize::try_from(addend.addend).ok()?)?;
    let answer = match specials.get(&index_local) {
        Some(Special::Index(k) | Special::IndexPair(k)) => {
            GetterAnswer::Scalar(k.checked_add(extra)?)
        }
        Some(Special::SliceBase { slice_local }) => GetterAnswer::Slice {
            slice_local: *slice_local,
            k: extra,
        },
        Some(
            Special::SliceIndex { slice_local, k } | Special::SliceIndexPair { slice_local, k },
        ) => GetterAnswer::Slice {
            slice_local: *slice_local,
            k: k.checked_add(extra)?,
        },
        _ => return None,
    };
    let mut carriers = peeled.carriers;
    carriers.extend(index_carriers);
    carriers.extend(addend.carriers);
    if carriers
        .iter()
        .any(|local| *local == index_local || specials.contains_key(local))
    {
        return None;
    }
    Some((answer, carriers))
}

/// Build the rewrite, or say why this body keeps its bracket.
fn analyze(body: &Unstructured, llbc: &Llbc) -> Result<Option<Plan>, Refusal> {
    let mut merge_blocks = HashSet::new();
    loop {
        match analyze_with_merges(body, llbc, &merge_blocks) {
            Err(WalkErr::NeedMerge(t)) => {
                if !merge_blocks.insert(t) {
                    return Err("depth-disagrees-at-merge");
                }
            }
            Err(WalkErr::Refusal(r)) => return Err(r),
            Ok(plan) => return Ok(plan),
        }
    }
}

fn analyze_with_merges(
    body: &Unstructured,
    llbc: &Llbc,
    merge_blocks: &HashSet<usize>,
) -> Result<Option<Plan>, WalkErr> {
    let n_blocks = body.body.len();
    let n_locals = body.locals.locals.len();
    let mut classified: HashMap<usize, (Leaf, bool)> = HashMap::new();
    let forward_blocks = super::forward_reachable_mask(llbc, body);
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && let Some(class) = classify_call(call, llbc)
        {
            classified.insert(bb, class);
        }
    }
    if classified.is_empty() {
        return Ok(None);
    }
    if !llbc.stack_sensitive_fns_complete() {
        return Err(WalkErr::Refusal("callee-stack-effects-unknown"));
    }
    // A callee that leaves slots, indexes through a parameter or returns
    // an index refuses here, whether or not it also reads below entry:
    // after the rewrite nothing rewinds a leftover
    // (`_fix_graph_after_inlining`). A callee that only reads below entry
    // is checked against the entry depth in the walk below.
    //
    // A returned owned guard is that leftover packaged in the dest the
    // caller drops, so it is not this body's slots. ParamSlots methods
    // of that dest read the nested bracket.
    let owned_scope_locals = owned_scope_receiver_locals(body, llbc, &forward_blocks);
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && callee_path(call, llbc)
                .as_deref()
                .is_some_and(|path| callee_effect_of(llbc, path).refuses_erase())
            && !callee_leftover_is_owned_scope(call, llbc, &owned_scope_locals)
        {
            return Err(WalkErr::Refusal("calls-stack-sensitive-fn"));
        }
    }
    if classified
        .values()
        .any(|(leaf, _)| leaf_unmodeled_for_erase(*leaf))
    {
        return Err(WalkErr::Refusal("unmodeled-stack-op"));
    }

    let mut plan = Plan {
        specials: HashMap::new(),
        removed: HashMap::new(),
        inserted_after: HashMap::new(),
        terms: HashMap::new(),
        slot_count: 0,
        slot_ty: None,
        pin_count: HashMap::new(),
        slot_intern: HashMap::new(),
        unreachable: Vec::new(),
        slice_src: HashMap::new(),
    };
    // `RootedItems::get(i)` takes a compile-time offset from the set's
    // open depth, often as `_t = const i`.  Follow those locals.
    let bind =
        |specials: &mut HashMap<usize, Special>, local: usize, special: Special| match specials
            .insert(local, special)
        {
            Some(previous) if previous != special => Err("special-rebound"),
            _ => Ok(()),
        };

    let def_counts = local_def_counts(body, llbc, &forward_blocks);
    let single_def = |dest: usize| def_counts.get(dest).copied() == Some(1);
    let mut stmt_assigned: HashMap<usize, usize> = HashMap::new();
    let mut call_dests = bit_set::BitSet::new();
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        for stmt in &block.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && let Some(dest) = place_local(place)
            {
                *stmt_assigned.entry(dest).or_default() += 1;
            }
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && let Some(dest) = place_local(&call.dest)
        {
            call_dests.insert(dest);
        }
    }
    let mut const_locals: HashMap<usize, usize> = HashMap::new();
    let mut array_lens: HashMap<usize, usize> = HashMap::new();
    let mut getter_cache: HashMap<u64, Option<SlotGetter>> = HashMap::new();
    let mut getter_bbs = bit_set::BitSet::new();
    let mut capture_temps = bit_set::BitSet::new();
    let mut depth_in: Vec<Option<IdxFact>> = vec![None; n_blocks];
    // Guards currently open. Incoming-slice Gets are allowed only while
    // this is non-zero: Open does not change scalar depth, so Close of a
    // slice-only bracket is still depth 0.
    let mut open_in: Vec<Option<usize>> = vec![None; n_blocks];
    let mut pin_count_in: Vec<Option<HashMap<usize, GuardPins>>> = vec![None; n_blocks];
    let mut facts_in: Vec<Option<ScrutineeFacts>> = vec![None; n_blocks];
    let mut copies_in: Vec<Option<BTreeMap<usize, FactKey>>> = vec![None; n_blocks];
    let mut splits_in: Vec<Option<Vec<SplitPred>>> = vec![None; n_blocks];
    let pred_counts = forward_pred_counts(body, llbc, &forward_blocks);
    let mut merge_min: HashMap<usize, usize> = HashMap::new();
    let mut merge_max: HashMap<usize, usize> = HashMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    if n_blocks == 0 {
        return Ok(None);
    }
    depth_in[0] = Some(IdxFact::Const(0));
    open_in[0] = Some(0);
    pin_count_in[0] = Some(HashMap::new());
    facts_in[0] = Some(ScrutineeFacts::default());
    copies_in[0] = Some(BTreeMap::new());
    splits_in[0] = Some(Vec::new());
    queue.push_back(0);
    let mut visited = vec![false; n_blocks];
    let reach = |depth_in: &mut Vec<Option<IdxFact>>,
                 open_in: &mut Vec<Option<usize>>,
                 pin_count_in: &mut Vec<Option<HashMap<usize, GuardPins>>>,
                 facts_in: &mut Vec<Option<ScrutineeFacts>>,
                 copies_in: &mut Vec<Option<BTreeMap<usize, FactKey>>>,
                 splits_in: &mut Vec<Option<Vec<SplitPred>>>,
                 queue: &mut VecDeque<usize>,
                 specials: &HashMap<usize, Special>,
                 visited: &mut [bool],
                 merge_min: &mut HashMap<usize, usize>,
                 merge_max: &mut HashMap<usize, usize>,
                 target: u64,
                 depth: IdxFact,
                 open: usize,
                 pins: &HashMap<usize, GuardPins>,
                 facts: &ScrutineeFacts,
                 copies: &BTreeMap<usize, FactKey>,
                 splits: &[SplitPred]|
     -> Result<(), WalkErr> {
        let target = target as usize;
        if target >= n_blocks {
            return Ok(());
        }
        // Abort / panic defaults are shared by many switches and carry
        // no slots; two depths meeting there is not a merge.
        if block_is_sink(body, llbc, target) {
            return Ok(());
        }
        // Scalar depth only: an incoming slice is not a slot
        // (`pin_roots_incoming_slice`). A Drop of the guard rewinds to
        // the open depth, so a slice-only early return may join a
        // `pin_root` arm there (`__majit_wrap_dict_descr_keys`).
        let rewinds = block_rewinds_scope(body, llbc, &classified, specials, target);
        let mut incoming_splits = splits.to_vec();
        let mut join_facts = facts.clone();
        let incoming = if merge_blocks.contains(&target) {
            match depth {
                IdxFact::Const(d) => {
                    note_merge_pred_depth(merge_min, merge_max, target, d);
                    // Depths disagree and the preds carry different
                    // scrutinee facts: keep a Const snapshot per fact
                    // set so a later switch on that key can restore the
                    // arm's own depth. Slot intern on this block still
                    // uses `AtMerge` (relative pins after the join).
                    // Joined facts stay empty: the distinguishing values
                    // live in `incoming_splits`.
                    if !facts.is_empty() {
                        add_split_pred(
                            &mut incoming_splits,
                            SplitPred {
                                facts: facts.clone(),
                                depth,
                                open,
                                pins: pins.clone(),
                            },
                        )?;
                        join_facts = ScrutineeFacts::default();
                    }
                    IdxFact::AtMerge {
                        block: target,
                        k: 0,
                    }
                }
                IdxFact::AtMerge { block, k: 0 } if block == target => depth,
                _ => return Err(WalkErr::Refusal("depth-disagrees-at-merge")),
            }
        } else {
            depth
        };
        let cap = pred_counts[target].max(MAX_SCRUTINEE_SPLITS);
        if incoming_splits.len() > cap {
            return Err(WalkErr::Refusal("scrutinee-split-unbounded"));
        }
        let mut requeue = false;
        match depth_in[target] {
            Some(old) if old == incoming => {}
            // A Drop rewinds, so two Const depths may join there
            // (slice-only early return vs `pin_root`). An AtMerge
            // successor meeting a Const is still a disagreement.
            Some(IdxFact::Const(_)) if rewinds && matches!(incoming, IdxFact::Const(_)) => {}
            // A unique arm first reached via the AtMerge walk, then
            // via a split restore: keep the Const. The reverse refines
            // below. Merge blocks stay AtMerge.
            Some(IdxFact::Const(_))
                if pred_counts[target] <= 1 && matches!(incoming, IdxFact::AtMerge { .. }) => {}
            Some(IdxFact::Const(_)) => match incoming {
                IdxFact::Const(_) => return Err(WalkErr::NeedMerge(target)),
                _ => return Err(WalkErr::Refusal("depth-disagrees-at-merge")),
            },
            Some(IdxFact::AtMerge { .. })
                if pred_counts[target] <= 1 && matches!(incoming, IdxFact::Const(_)) =>
            {
                depth_in[target] = Some(incoming);
                open_in[target] = Some(open);
                pin_count_in[target] = Some(pins.clone());
                requeue = true;
            }
            Some(IdxFact::AtMerge { .. }) => {
                return Err(WalkErr::Refusal("depth-disagrees-at-merge"));
            }
            None => {
                depth_in[target] = Some(incoming);
                open_in[target] = Some(open);
                queue.push_back(target);
            }
        }
        match &facts_in[target] {
            Some(prev) => {
                let joined = prev.intersect(&join_facts);
                if &joined != prev {
                    facts_in[target] = Some(joined);
                    requeue = true;
                }
            }
            None => facts_in[target] = Some(join_facts),
        }
        match &copies_in[target] {
            Some(prev) => {
                let joined = intersect_copies(prev, copies);
                if &joined != prev {
                    copies_in[target] = Some(joined);
                    requeue = true;
                }
            }
            None => copies_in[target] = Some(copies.clone()),
        }
        match &splits_in[target] {
            Some(prev) => {
                let joined = union_splits(prev.clone(), &incoming_splits)?;
                if &joined != prev {
                    splits_in[target] = Some(joined);
                    requeue = true;
                }
            }
            None => splits_in[target] = Some(incoming_splits),
        }
        if requeue && visited[target] {
            visited[target] = false;
            queue.push_back(target);
        }
        match &pin_count_in[target] {
            Some(prev) if prev != pins => {
                if rewinds {
                    Ok(())
                } else if merge_blocks.contains(&target) {
                    let Some(joined) = join_pin_counts_min(prev, pins) else {
                        return Err(WalkErr::Refusal("pin-count-disagrees-at-merge"));
                    };
                    if &joined != prev {
                        pin_count_in[target] = Some(joined);
                        if visited[target] {
                            visited[target] = false;
                            queue.push_back(target);
                        }
                    }
                    Ok(())
                } else if pred_counts[target] <= 1
                    && matches!(depth_in[target], Some(IdxFact::Const(_)))
                    && matches!(incoming, IdxFact::AtMerge { .. })
                {
                    Ok(())
                } else {
                    Err(WalkErr::Refusal("pin-count-disagrees-at-merge"))
                }
            }
            Some(_) => Ok(()),
            None => {
                pin_count_in[target] = Some(pins.clone());
                Ok(())
            }
        }
    };

    while let Some(bb) = queue.pop_front() {
        if visited[bb] {
            continue;
        }
        visited[bb] = true;
        plan.removed.remove(&bb);
        plan.terms.remove(&bb);
        plan.inserted_after.retain(|k, _| k.0 != bb);
        let mut depth = depth_in[bb].expect("queued blocks have a depth");
        let mut open = open_in[bb].expect("queued blocks have an open count");
        plan.pin_count = pin_count_in[bb]
            .clone()
            .expect("queued blocks have a pin_count");
        let mut facts = facts_in[bb]
            .clone()
            .expect("queued blocks have scrutinee facts");
        let mut copies = copies_in[bb]
            .clone()
            .expect("queued blocks have copy facts");
        let mut splits = splits_in[bb]
            .clone()
            .expect("queued blocks have split preds");
        let block = &body.body[bb];
        // A `usize` local is only a compile-time index inside this block.
        // Crossing a join with a stale constant would rewrite `get(i)` to the
        // wrong slot; a later non-const store of the same local must drop it.
        const_locals.clear();
        // Statements: the index, pair and alias definitions.
        for (si, stmt) in block.statements.iter().enumerate() {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            if let Some(base) = place_root_local(place) {
                kill_local_facts(&mut facts, &mut splits, &mut copies, base);
            }
            if let Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } = rvalue
                && let Some(base) = place_root_local(src)
            {
                kill_local_facts(&mut facts, &mut splits, &mut copies, base);
            }
            let Some(dest) = place_local(place) else {
                continue;
            };
            if let Some(key) = rvalue_fact_key(rvalue, &copies) {
                copies.insert(dest, key);
            }
            const_locals.remove(&dest);
            match rvalue {
                Rvalue::Aggregate(kind, operands) if kind.get("Array").is_some() => {
                    if single_def(dest) {
                        array_lens.insert(dest, operands.len());
                    }
                }
                Rvalue::Len(src) => {
                    let base = place_local(src).or_else(|| deref_of_local(src));
                    if single_def(dest)
                        && let Some(n) = base.and_then(|l| array_lens.get(&l).copied())
                    {
                        const_locals.insert(dest, n);
                    }
                }
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let base = place_local(src).or_else(|| deref_of_local(src));
                    if single_def(dest)
                        && let Some(n) = base.and_then(|l| array_lens.get(&l).copied())
                    {
                        array_lens.insert(dest, n);
                    }
                }
                // Cast / UnaryOp may change the value (narrowing, negation,
                // bit-not). `constfold.fold_op_list` evaluates them; this
                // pass does not, so the fact is dropped.
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                    if single_def(dest)
                        && let Some((pair, 1)) = tuple_field_of_local(src)
                        && let Some(n) = array_lens.get(&pair).copied()
                    {
                        const_locals.insert(dest, n);
                    }
                }
                _ => {}
            }
            let special = match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let guard = place_local(src)
                        .and_then(|l| match plan.specials.get(&l) {
                            Some(Special::Guard(_)) => Some(l),
                            _ => None,
                        })
                        .or_else(|| {
                            deref_of_local(src).and_then(|l| match plan.specials.get(&l) {
                                Some(Special::Alias(g)) => Some(*g),
                                _ => None,
                            })
                        });
                    guard.map(Special::Alias)
                }
                Rvalue::Use(op, _) => match op {
                    Operand::Copy(src) | Operand::Move(src) => {
                        if let Some(l) = place_local(src) {
                            if single_def(dest) {
                                if let Some(n) = const_locals.get(&l).copied() {
                                    const_locals.insert(dest, n);
                                }
                                if let Some(n) = array_lens.get(&l).copied() {
                                    array_lens.insert(dest, n);
                                }
                            }
                            match plan.specials.get(&l) {
                                Some(Special::Index(k)) => Some(Special::Index(*k)),
                                Some(Special::Alias(g)) => Some(Special::Alias(*g)),
                                Some(Special::Guard(d)) => Some(Special::Guard(*d)),
                                Some(Special::SliceBase { slice_local }) => {
                                    Some(Special::SliceBase {
                                        slice_local: *slice_local,
                                    })
                                }
                                Some(Special::SliceIndex { slice_local, k }) => {
                                    Some(Special::SliceIndex {
                                        slice_local: *slice_local,
                                        k: *k,
                                    })
                                }
                                Some(Special::SliceIndexPair { slice_local, k }) => {
                                    Some(Special::SliceIndexPair {
                                        slice_local: *slice_local,
                                        k: *k,
                                    })
                                }
                                _ => None,
                            }
                        } else if let Some((pair, 0)) = tuple_field_of_local(src) {
                            match plan.specials.get(&pair) {
                                Some(Special::IndexPair(k)) => Some(Special::Index(*k)),
                                Some(Special::SliceIndexPair { slice_local, k }) => {
                                    Some(Special::SliceIndex {
                                        slice_local: *slice_local,
                                        k: *k,
                                    })
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    Operand::Const(_) => {
                        if single_def(dest)
                            && let Some(n) = const_usize(op, llbc)
                        {
                            const_locals.insert(dest, n);
                        }
                        None
                    }
                },
                Rvalue::BinaryOp(op, lhs, rhs) => match binop_is_index_arith(op) {
                    Some(arith) => {
                        let spec_of = |o: &Operand| {
                            operand_local(o).and_then(|l| plan.specials.get(&l).copied())
                        };
                        let slice_add = |spec: Option<Special>, other: &Operand, checked: bool| {
                            let c = const_usize(other, llbc)?;
                            match spec {
                                Some(Special::SliceBase { slice_local }) => Some(if checked {
                                    Special::SliceIndexPair { slice_local, k: c }
                                } else {
                                    Special::SliceIndex { slice_local, k: c }
                                }),
                                Some(Special::SliceIndex { slice_local, k }) => {
                                    let k = k.checked_add(c)?;
                                    Some(if checked {
                                        Special::SliceIndexPair { slice_local, k }
                                    } else {
                                        Special::SliceIndex { slice_local, k }
                                    })
                                }
                                _ => None,
                            }
                        };
                        if let Some(special) = match arith {
                            IndexArith::Add { checked } => slice_add(spec_of(lhs), rhs, checked)
                                .or_else(|| slice_add(spec_of(rhs), lhs, checked)),
                            IndexArith::Sub { .. } => None,
                        } {
                            Some(special)
                        } else if matches!(arith, IndexArith::Add { .. })
                            && let Some(slice_local) = spec_of(lhs)
                                .and_then(slice_of_special)
                                .or_else(|| spec_of(rhs).and_then(slice_of_special))
                        {
                            // Non-const `base + i` of a leftover slice
                            // (`do_call` n>1, `try_new` live-tail) when `i`
                            // is a local this body assigned. An unassigned
                            // addend (return place, parameter) stays
                            // `index-not-static`. A `const` item rustc
                            // spells as `Copy(Global)` is the same shape
                            // (`FrameBox::new` `inputs + FRAME_BOX_DEBUG_INPUT`).
                            let other = if spec_of(lhs).and_then(slice_of_special).is_some() {
                                rhs
                            } else {
                                lhs
                            };
                            let computed = slice_addend_is_computed(other, &stmt_assigned);
                            if computed {
                                Some(if matches!(arith, IndexArith::Add { checked: true }) {
                                    Special::SliceIndexPair { slice_local, k: 0 }
                                } else {
                                    Special::SliceIndex { slice_local, k: 0 }
                                })
                            } else {
                                None
                            }
                        } else {
                            let index_of = |o: &Operand| match spec_of(o) {
                                Some(Special::Index(k)) => Some(k),
                                _ => None,
                            };
                            // `len + k` and `len - k` are compile-time offsets
                            // from the depth at the Len/Base. `zip_two_tuple_next`
                            // names a just-pinned slot as `shadow_stack_len() - 1`.
                            let offset = match (arith, index_of(lhs), index_of(rhs)) {
                                (IndexArith::Add { checked }, Some(k), None) => {
                                    const_usize(rhs, llbc).map(|c| (k.add(c), checked))
                                }
                                (IndexArith::Add { checked }, None, Some(k)) => {
                                    const_usize(lhs, llbc).map(|c| (k.add(c), checked))
                                }
                                (IndexArith::Sub { checked }, Some(k), None) => {
                                    const_usize(rhs, llbc)
                                        .and_then(|c| k.checked_sub(c).map(|n| (n, checked)))
                                }
                                _ => None,
                            };
                            offset.map(|(k, checked)| {
                                if checked {
                                    Special::IndexPair(k)
                                } else {
                                    Special::Index(k)
                                }
                            })
                        }
                    }
                    None => None,
                },
                _ => None,
            };
            if let Some(special) = special {
                bind(&mut plan.specials, dest, special)?;
                if !is_slice_special(special) {
                    plan.removed.entry(bb).or_default().push(si);
                }
            }
        }
        let term_json = block.terminator.kind_value();
        match block.term_ref(llbc) {
            Ok(TermKind::Call { call, target, .. }) => {
                if let Some(d) = place_root_local(&call.dest) {
                    kill_local_facts(&mut facts, &mut splits, &mut copies, d);
                }
                // A call that receives the local by `&mut` / `*mut` mutates
                // it (`local_def_counts` `ref_kind_mutates`). The `&mut`
                // itself is a prior Ref assign (already killed); a `*p`
                // argument is the same mutation through the pointer.
                for arg in &call.args {
                    if let Operand::Copy(p) | Operand::Move(p) = arg
                        && let Some(base) = deref_of_local(p)
                    {
                        kill_local_facts(&mut facts, &mut splits, &mut copies, base);
                    }
                }
                let Some(&(leaf, method)) = classified.get(&bb) else {
                    // A closure whose body is `shadow_stack_get` of a captured
                    // index (or `captured + const`) is a Get of that slot.
                    // `RootBracketPlan.capture_temps` already answers this at
                    // lowering; the codewriter must not see the bracket
                    // (`gctransform` `push_roots` runs after `make_jitcodes`).
                    if let Some((answer, carriers)) = try_slot_getter_call(
                        body,
                        llbc,
                        call,
                        &plan.specials,
                        &stmt_assigned,
                        &call_dests,
                        &forward_blocks,
                        &mut getter_cache,
                    ) {
                        let dest_json = &term_json["Call"]["call"]["dest"];
                        let stmts = match answer {
                            GetterAnswer::Scalar(k) => {
                                let slot = slot_for_get(k, depth, &merge_min)?;
                                let ty = plan.slot_ty.clone().ok_or("get-before-any-pin")?;
                                vec![assign_json(
                                    dest_json.clone(),
                                    json!({ "Use": [{ "Copy": local_place_json(plan.intern_slot(n_locals, slot), &ty) }, "Yes"] }),
                                )]
                            }
                            GetterAnswer::Slice { slice_local, k } => {
                                if open == 0 {
                                    return Err(WalkErr::Refusal("get-above-depth"));
                                }
                                vec![assign_json(
                                    dest_json.clone(),
                                    slice_elem_use(body, slice_local, dest_json, k)?,
                                )]
                            }
                        };
                        plan.terms.insert(
                            bb,
                            TermRewrite {
                                target: *target,
                                stmts,
                            },
                        );
                        getter_bbs.insert(bb);
                        for local in carriers {
                            capture_temps.insert(local);
                        }
                        reach(
                            &mut depth_in,
                            &mut open_in,
                            &mut pin_count_in,
                            &mut facts_in,
                            &mut copies_in,
                            &mut splits_in,
                            &mut queue,
                            &plan.specials,
                            &mut visited,
                            &mut merge_min,
                            &mut merge_max,
                            *target,
                            depth,
                            open,
                            &plan.pin_count,
                            &facts,
                            &copies,
                            &splits,
                        )?;
                        continue;
                    }
                    // rustc spells `drop(guard)` as
                    // `_t = Move(guard); core::mem::drop(Move(_t))`. The
                    // Move copy is already bound as Guard and stripped;
                    // the Call is Close of that Guard, the way
                    // `TermKind::Drop` of a Guard is. Census walks the
                    // raw body (`ShadowStackFrameworkGCTransformer.push_roots`
                    // / `pop_roots` run after `warmspot.py make_jitcodes`).
                    if let Some(d) = mem_drop_guard_depth(call, llbc, &plan.specials) {
                        if d.close_exceeds(depth) {
                            return Err(WalkErr::Refusal("close-above-depth"));
                        }
                        plan.forget_closed_guards(d);
                        plan.forget_closed_split_pins(&mut splits, d);
                        rewind_depth(&mut depth, &mut splits, d);
                        open = open.saturating_sub(1);
                        for s in splits.iter_mut() {
                            s.open = s.open.saturating_sub(1);
                        }
                        plan.terms.insert(
                            bb,
                            TermRewrite {
                                target: *target,
                                stmts: Vec::new(),
                            },
                        );
                        reach(
                            &mut depth_in,
                            &mut open_in,
                            &mut pin_count_in,
                            &mut facts_in,
                            &mut copies_in,
                            &mut splits_in,
                            &mut queue,
                            &plan.specials,
                            &mut visited,
                            &mut merge_min,
                            &mut merge_max,
                            *target,
                            depth,
                            open,
                            &plan.pin_count,
                            &facts,
                            &copies,
                            &splits,
                        )?;
                        continue;
                    }
                    // Observes at Known(0) reads the caller's slots.
                    // Inside our Open the callee reads our pins, which
                    // become slot locals (`eval_slice_index` / `getindex_w`).
                    if callee_path(call, llbc)
                        .as_deref()
                        .is_some_and(|path| callee_effect_of(llbc, path).observes)
                        && depth.is_entry()
                    {
                        return Err(WalkErr::Refusal("calls-stack-sensitive-fn"));
                    }
                    reach(
                        &mut depth_in,
                        &mut open_in,
                        &mut pin_count_in,
                        &mut facts_in,
                        &mut copies_in,
                        &mut splits_in,
                        &mut queue,
                        &plan.specials,
                        &mut visited,
                        &mut merge_min,
                        &mut merge_max,
                        *target,
                        depth,
                        open,
                        &plan.pin_count,
                        &facts,
                        &copies,
                        &splits,
                    )?;
                    continue;
                };
                let args_json = &term_json["Call"]["call"]["args"];
                let dest_json = &term_json["Call"]["call"]["dest"];
                let dest = place_local(&call.dest);
                let arg0 = usize::from(method);
                let guard = if method {
                    let receiver = call.args.first().and_then(operand_local);
                    match receiver.and_then(|l| plan.specials.get(&l)) {
                        Some(Special::Alias(g)) => Some(*g),
                        _ => return Err(WalkErr::Refusal("receiver-not-a-guard-borrow")),
                    }
                } else {
                    None
                };
                let guard_depth = |g: usize| match plan.specials.get(&g) {
                    Some(Special::Guard(d)) => Ok(*d),
                    _ => Err("guard-not-opened"),
                };
                let index_arg = |i: usize| -> Result<IdxFact, Refusal> {
                    match call
                        .args
                        .get(i)
                        .and_then(operand_local)
                        .and_then(|l| plan.specials.get(&l))
                    {
                        Some(Special::Index(k)) => Ok(*k),
                        _ => Err("index-not-static"),
                    }
                };
                let mut stmts = Vec::new();
                match leaf {
                    Leaf::Open | Leaf::ItemsOpen => {
                        let dest = dest.ok_or("guard-dest-projected")?;
                        bind(&mut plan.specials, dest, Special::Guard(depth))?;
                        open += 1;
                        for s in splits.iter_mut() {
                            s.open += 1;
                            s.pins.insert(dest, GuardPins::Exact(0));
                        }
                        plan.pin_count.insert(dest, GuardPins::Exact(0));
                    }
                    Leaf::Close | Leaf::ItemsClose => {
                        let d = guard_depth(guard.ok_or("close-without-guard")?)?;
                        if d.close_exceeds(depth) {
                            return Err(WalkErr::Refusal("close-above-depth"));
                        }
                        plan.forget_closed_guards(d);
                        plan.forget_closed_split_pins(&mut splits, d);
                        rewind_depth(&mut depth, &mut splits, d);
                        open = open.saturating_sub(1);
                        for s in splits.iter_mut() {
                            s.open = s.open.saturating_sub(1);
                        }
                    }
                    Leaf::Base => {
                        let d = guard_depth(guard.ok_or("base-without-guard")?)?;
                        bind(
                            &mut plan.specials,
                            dest.ok_or("index-dest-projected")?,
                            Special::Index(d),
                        )?;
                    }
                    Leaf::Len => {
                        bind(
                            &mut plan.specials,
                            dest.ok_or("index-dest-projected")?,
                            Special::Index(depth),
                        )?;
                    }
                    Leaf::Pin | Leaf::ItemsPush => {
                        let value = &args_json[arg0];
                        let ty = value
                            .as_object()
                            .and_then(|m| m.values().next())
                            .and_then(|p| p.get("ty"))
                            .cloned()
                            .ok_or("pin-value-untyped")?;
                        plan.slot_ty.get_or_insert(ty.clone());
                        let copied = copy_operand_json(value);
                        let slot = slots_from_depth(depth, 1)
                            .next()
                            .expect("n=1 yields one slot");
                        stmts.push(assign_json(
                            local_place_json(plan.intern_slot(n_locals, slot), &ty),
                            json!({ "Use": [copied.clone(), "Yes"] }),
                        ));
                        // `pin_root` answers the value; `RootedItems::push`
                        // answers `()`.  Copy dest only when it is the value.
                        if dest_json.get("ty") == Some(&ty) {
                            stmts.push(assign_json(
                                dest_json.clone(),
                                json!({ "Use": [copied, "Yes"] }),
                            ));
                        }
                        add_to_depth(&mut depth, &mut splits, 1);
                        if let Some(g) = guard {
                            let e = plan.pin_count.entry(g).or_insert(GuardPins::Exact(0));
                            *e = e.plus_one();
                            for s in splits.iter_mut() {
                                let e = s.pins.entry(g).or_insert(GuardPins::Exact(0));
                                *e = e.plus_one();
                            }
                        }
                    }
                    Leaf::Publish => {
                        let slice = call.args.get(arg0).and_then(operand_local);
                        // Same-block array first; then an aggregate in a
                        // dominating predecessor (`pin_roots_slice_values`
                        // shape); then an incoming `&[PyObjectRef]`
                        // (`pin_roots_incoming_slice`). `gctransform`
                        // `push_roots` runs after `make_jitcodes`.
                        match resolve_published(body, block, slice, llbc) {
                            Some(Published::Array {
                                stmt_index: agg_si,
                                operands,
                            }) => {
                                let mut inserted = Vec::new();
                                for (slot, element) in
                                    slots_from_depth(depth, operands.len()).zip(operands.iter())
                                {
                                    let ty = element
                                        .as_object()
                                        .and_then(|m| m.values().next())
                                        .and_then(|p| p.get("ty"))
                                        .cloned()
                                        .ok_or("publish-element-untyped")?;
                                    plan.slot_ty.get_or_insert(ty.clone());
                                    inserted.push(assign_json(
                                        local_place_json(plan.intern_slot(n_locals, slot), &ty),
                                        json!({ "Use": [copy_operand_json(element), "Yes"] }),
                                    ));
                                }
                                plan.inserted_after
                                    .entry((bb, agg_si))
                                    .or_default()
                                    .extend(inserted);
                                if let Some(dest) = dest {
                                    bind(&mut plan.specials, dest, Special::Index(depth))?;
                                }
                                add_to_depth(&mut depth, &mut splits, operands.len());
                            }
                            Some(Published::Slice {
                                local: slice_local,
                                ty,
                                deref_ty,
                            }) => {
                                plan.slice_src.insert(slice_local, (ty, deref_ty));
                                if let Some(d) = dest {
                                    bind(
                                        &mut plan.specials,
                                        d,
                                        Special::SliceBase { slice_local },
                                    )?;
                                }
                                // Slice-relative base is 0 so later
                                // `base + k` names element `k`.
                                stmts.push(assign_json(
                                    dest_json.clone(),
                                    json!({ "Use": [{ "Const": [{ "Integer": { "Unsigned": ["Usize", "0"] } }, dest_json["ty"].clone()] }, "Yes"] }),
                                ));
                            }
                            None => {
                                if let Some((agg_bb, agg_si, operands)) =
                                    resolve_published_array_in_body(
                                        body,
                                        slice,
                                        &stmt_assigned,
                                        &forward_blocks,
                                    )
                                {
                                    let mut inserted = Vec::new();
                                    for (slot, element) in
                                        slots_from_depth(depth, operands.len()).zip(operands.iter())
                                    {
                                        let ty = element
                                            .as_object()
                                            .and_then(|m| m.values().next())
                                            .and_then(|p| p.get("ty"))
                                            .cloned()
                                            .ok_or("publish-element-untyped")?;
                                        plan.slot_ty.get_or_insert(ty.clone());
                                        inserted.push(assign_json(
                                            local_place_json(plan.intern_slot(n_locals, slot), &ty),
                                            json!({ "Use": [copy_operand_json(element), "Yes"] }),
                                        ));
                                    }
                                    plan.inserted_after
                                        .entry((agg_bb, agg_si))
                                        .or_default()
                                        .extend(inserted);
                                    if let Some(dest) = dest {
                                        bind(&mut plan.specials, dest, Special::Index(depth))?;
                                    }
                                    add_to_depth(&mut depth, &mut splits, operands.len());
                                } else if let Some((slice_local, _)) = slice.and_then(|s| {
                                    super::pin_roots_incoming_slice(
                                        body,
                                        s,
                                        &stmt_assigned,
                                        &forward_blocks,
                                    )
                                }) {
                                    let loc = body
                                        .locals
                                        .locals
                                        .get(slice_local)
                                        .ok_or("slice-untyped")?;
                                    let ty = ty_ref_json(&loc.ty);
                                    plan.slice_src
                                        .entry(slice_local)
                                        .or_insert((ty.clone(), ty));
                                    if let Some(d) = dest {
                                        bind(
                                            &mut plan.specials,
                                            d,
                                            Special::SliceBase { slice_local },
                                        )?;
                                    }
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        json!({ "Use": [{ "Const": [{ "Integer": { "Unsigned": ["Usize", "0"] } }, dest_json["ty"].clone()] }, "Yes"] }),
                                    ));
                                } else {
                                    return Err(WalkErr::Refusal("publish-array-unresolved"));
                                }
                            }
                        }
                    }
                    Leaf::ItemsLen | Leaf::ItemsIsEmpty => {
                        let g = guard.ok_or("items-count-without-guard")?;
                        let n = plan
                            .pin_count
                            .get(&g)
                            .copied()
                            .and_then(GuardPins::exact)
                            .ok_or("items-count-not-static")?;
                        dest.ok_or("index-dest-projected")?;
                        let ty = dest_json["ty"].clone();
                        let lit = if matches!(leaf, Leaf::ItemsLen) {
                            json!({ "Integer": { "Unsigned": ["Usize", n.to_string()] } })
                        } else {
                            json!({ "Bool": n == 0 })
                        };
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Const": [lit, ty] }, "Yes"] }),
                        ));
                    }
                    Leaf::Normalize | Leaf::ItemsNop => {}
                    Leaf::NormalizeMoved => {
                        let ty = dest_json["ty"].clone();
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Const": [{ "Bool": false }, ty] }, "Yes"] }),
                        ));
                    }
                    Leaf::Get => {
                        let index_local = call.args.get(arg0).and_then(operand_local);
                        match index_local.and_then(|l| plan.specials.get(&l).copied()) {
                            Some(Special::Index(k)) => {
                                let slot = slot_for_get(k, depth, &merge_min)?;
                                let ty = plan.slot_ty.clone().ok_or("get-before-any-pin")?;
                                stmts.push(assign_json(
                                    dest_json.clone(),
                                    json!({ "Use": [{ "Copy": local_place_json(plan.intern_slot(n_locals, slot), &ty) }, "Yes"] }),
                                ));
                            }
                            Some(Special::SliceBase { slice_local }) => {
                                if open == 0 {
                                    return Err(WalkErr::Refusal("get-above-depth"));
                                }
                                stmts.push(assign_json(
                                    dest_json.clone(),
                                    slice_elem_use(body, slice_local, dest_json, 0)?,
                                ));
                            }
                            Some(
                                Special::SliceIndex { slice_local, k }
                                | Special::SliceIndexPair { slice_local, k },
                            ) => {
                                if open == 0 {
                                    return Err(WalkErr::Refusal("get-above-depth"));
                                }
                                if k == 0 {
                                    // Dummy k from leftover-slice `base + i`,
                                    // or compile-time `base + 0`. Publish dest
                                    // is 0, so this operand is the element
                                    // index.
                                    let (slice_ty, deref_ty) = plan
                                        .slice_src
                                        .get(&slice_local)
                                        .cloned()
                                        .or_else(|| {
                                            body.locals.locals.get(slice_local).map(|loc| {
                                                let ty = ty_ref_json(&loc.ty);
                                                (ty.clone(), ty)
                                            })
                                        })
                                        .ok_or("slice-src-missing")?;
                                    let elem_ty =
                                        dest_json.get("ty").cloned().ok_or("get-dest-untyped")?;
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        slice_index_use(
                                            slice_local,
                                            &slice_ty,
                                            &deref_ty,
                                            copy_operand_json(&args_json[arg0]),
                                            &elem_ty,
                                        ),
                                    ));
                                } else {
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        slice_elem_use(body, slice_local, dest_json, k)?,
                                    ));
                                }
                            }
                            _ => return Err(WalkErr::Refusal("index-not-static")),
                        }
                    }
                    Leaf::ItemsGet => {
                        // `take` has no index; the rewrite has no Vec constructor.
                        let Some(index_op) = call.args.get(arg0) else {
                            return Err(WalkErr::Refusal("unmodeled-stack-op"));
                        };
                        // `RootedItems::get(i)`: `i` is an offset from the
                        // set's open depth (`gc_roots.rs` `RootedItems::get`).
                        let rel = const_usize(index_op, llbc)
                            .or_else(|| {
                                operand_local(index_op).and_then(|l| const_locals.get(&l).copied())
                            })
                            .ok_or("index-not-static")?;
                        let g = guard.ok_or("get-without-guard")?;
                        let d = guard_depth(g)?;
                        let pins = plan.pin_count.get(&g).map(|p| p.count()).unwrap_or(0);
                        if rel >= pins {
                            return Err(WalkErr::Refusal("get-out-of-set"));
                        }
                        let k = d.add(rel);
                        let slot = slot_for_get(k, depth, &merge_min)?;
                        let ty = plan.slot_ty.clone().ok_or("get-before-any-pin")?;
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Copy": local_place_json(plan.intern_slot(n_locals, slot), &ty) }, "Yes"] }),
                        ));
                    }
                    Leaf::Set => {
                        let k = index_arg(arg0)?;
                        let slot = slot_for_get(k, depth, &merge_min).map_err(|r| {
                            if r == "get-above-depth" {
                                "set-above-depth"
                            } else {
                                r
                            }
                        })?;
                        let ty = plan.slot_ty.clone().ok_or("set-before-any-pin")?;
                        stmts.push(assign_json(
                            local_place_json(plan.intern_slot(n_locals, slot), &ty),
                            json!({ "Use": [copy_operand_json(&args_json[arg0 + 1]), "Yes"] }),
                        ));
                    }
                    Leaf::ReloadTop => {
                        let slot = match depth {
                            IdxFact::Const(0) | IdxFact::AtMerge { k: 0, .. } => {
                                return Err(WalkErr::Refusal("reload-empty-stack"));
                            }
                            IdxFact::Const(d) => Slot::Abs(d - 1),
                            IdxFact::AtMerge { block, k } => Slot::Rel { block, k: k - 1 },
                        };
                        let ty = plan.slot_ty.clone().ok_or("reload-before-any-pin")?;
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Copy": local_place_json(plan.intern_slot(n_locals, slot), &ty) }, "Yes"] }),
                        ));
                    }
                    Leaf::Unmodeled | Leaf::CopyRange => unreachable!("refused above"),
                }
                plan.terms.insert(
                    bb,
                    TermRewrite {
                        target: *target,
                        stmts,
                    },
                );
                reach(
                    &mut depth_in,
                    &mut open_in,
                    &mut pin_count_in,
                    &mut facts_in,
                    &mut copies_in,
                    &mut splits_in,
                    &mut queue,
                    &plan.specials,
                    &mut visited,
                    &mut merge_min,
                    &mut merge_max,
                    *target,
                    depth,
                    open,
                    &plan.pin_count,
                    &facts,
                    &copies,
                    &splits,
                )?;
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                let dropped = place_local(place);
                if let Some(d) = dropped.and_then(|l| match plan.specials.get(&l) {
                    Some(Special::Guard(d)) => Some(*d),
                    _ => None,
                }) {
                    // A drop flag the artefact does not carry can leave the
                    // guard unopened on this path; the depth says so.
                    if d.close_exceeds(depth) {
                        return Err(WalkErr::Refusal("close-above-depth"));
                    }
                    plan.forget_closed_guards(d);
                    plan.forget_closed_split_pins(&mut splits, d);
                    rewind_depth(&mut depth, &mut splits, d);
                    open = open.saturating_sub(1);
                    for s in splits.iter_mut() {
                        s.open = s.open.saturating_sub(1);
                    }
                    plan.terms.insert(
                        bb,
                        TermRewrite {
                            target: *target,
                            stmts: Vec::new(),
                        },
                    );
                    reach(
                        &mut depth_in,
                        &mut open_in,
                        &mut pin_count_in,
                        &mut facts_in,
                        &mut copies_in,
                        &mut splits_in,
                        &mut queue,
                        &plan.specials,
                        &mut visited,
                        &mut merge_min,
                        &mut merge_max,
                        *target,
                        depth,
                        open,
                        &plan.pin_count,
                        &facts,
                        &copies,
                        &splits,
                    )?;
                } else if dropped.is_some_and(|l| is_root_scope_local(body, llbc, l)) {
                    return Err(WalkErr::Refusal("drop-of-unopened-guard"));
                } else {
                    reach(
                        &mut depth_in,
                        &mut open_in,
                        &mut pin_count_in,
                        &mut facts_in,
                        &mut copies_in,
                        &mut splits_in,
                        &mut queue,
                        &plan.specials,
                        &mut visited,
                        &mut merge_min,
                        &mut merge_max,
                        *target,
                        depth,
                        open,
                        &plan.pin_count,
                        &facts,
                        &copies,
                        &splits,
                    )?;
                }
            }
            Ok(TermKind::Assert { assert, target, .. }) => {
                if assert_is_slot_index_bounds_check(
                    assert,
                    &plan.specials,
                    &const_locals,
                    llbc,
                    &merge_min,
                ) {
                    plan.terms.insert(
                        bb,
                        TermRewrite {
                            target: *target,
                            stmts: Vec::new(),
                        },
                    );
                }
                reach(
                    &mut depth_in,
                    &mut open_in,
                    &mut pin_count_in,
                    &mut facts_in,
                    &mut copies_in,
                    &mut splits_in,
                    &mut queue,
                    &plan.specials,
                    &mut visited,
                    &mut merge_min,
                    &mut merge_max,
                    *target,
                    depth,
                    open,
                    &plan.pin_count,
                    &facts,
                    &copies,
                    &splits,
                )?;
            }
            Ok(TermKind::Goto { target }) => reach(
                &mut depth_in,
                &mut open_in,
                &mut pin_count_in,
                &mut facts_in,
                &mut copies_in,
                &mut splits_in,
                &mut queue,
                &plan.specials,
                &mut visited,
                &mut merge_min,
                &mut merge_max,
                *target,
                depth,
                open,
                &plan.pin_count,
                &facts,
                &copies,
                &splits,
            )?,
            Ok(TermKind::Switch { discr, targets }) => {
                // `constfold.constant_diffuse` after `exitswitch vexit`;
                // `rewire_link_for_known_exitswitch` on a later switch.
                let key = operand_fact_key(discr, &copies);
                let restore: Vec<SplitPred> = if let Some(key) = key {
                    splits
                        .iter()
                        .filter(|s| s.facts.get(key).is_some())
                        .cloned()
                        .collect()
                } else {
                    Vec::new()
                };
                let use_splits =
                    key.is_some() && !restore.is_empty() && restore.len() == splits.len();
                if use_splits {
                    let key = key.expect("filtered on key");
                    for s in restore {
                        let v = s.facts.get(key).expect("restore filtered on key");
                        let target = switch_known_target(targets, v);
                        // Depth is the arm's own Const snapshot; open and
                        // pins follow this block (nested Open/Close on the
                        // shared suffix already updated `plan.pin_count`).
                        reach(
                            &mut depth_in,
                            &mut open_in,
                            &mut pin_count_in,
                            &mut facts_in,
                            &mut copies_in,
                            &mut splits_in,
                            &mut queue,
                            &plan.specials,
                            &mut visited,
                            &mut merge_min,
                            &mut merge_max,
                            target,
                            s.depth,
                            open,
                            &plan.pin_count,
                            &s.facts,
                            &copies,
                            &[],
                        )?;
                    }
                } else if let Some(v) = key.and_then(|k| facts.get(k)) {
                    let target = switch_known_target(targets, v);
                    reach(
                        &mut depth_in,
                        &mut open_in,
                        &mut pin_count_in,
                        &mut facts_in,
                        &mut copies_in,
                        &mut splits_in,
                        &mut queue,
                        &plan.specials,
                        &mut visited,
                        &mut merge_min,
                        &mut merge_max,
                        target,
                        depth,
                        open,
                        &plan.pin_count,
                        &facts,
                        &copies,
                        &splits,
                    )?;
                } else {
                    for arm in switch_successors(targets) {
                        let mut arm_facts = facts.clone();
                        if let (Some(key), Some(v)) = (key, arm.value) {
                            arm_facts.insert(key, v);
                        }
                        reach(
                            &mut depth_in,
                            &mut open_in,
                            &mut pin_count_in,
                            &mut facts_in,
                            &mut copies_in,
                            &mut splits_in,
                            &mut queue,
                            &plan.specials,
                            &mut visited,
                            &mut merge_min,
                            &mut merge_max,
                            arm.target,
                            depth,
                            open,
                            &plan.pin_count,
                            &arm_facts,
                            &copies,
                            &splits,
                        )?;
                    }
                }
            }
            Ok(TermKind::Return) => {
                if !depth.is_entry() {
                    // 10.04 `Drop` of the guard lives in an `is_cleanup`
                    // block. `FunDecl::unstructured` keeps those blocks, so
                    // `UnwindResume` is among them and a later
                    // `UnwindTerminate` can be last. Return is then the
                    // normal closer.
                    if body
                        .body
                        .iter()
                        .any(|bb| matches!(bb.term_ref(llbc), Ok(TermKind::UnwindResume)))
                    {
                        depth = IdxFact::Const(0);
                    } else {
                        return Err(WalkErr::Refusal("returns-with-published-slots"));
                    }
                }
            }
            Ok(
                TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::Abort(_)
                | TermKind::Panic { .. }
                | TermKind::UndefinedBehavior,
            ) => {}
            _ => return Err(WalkErr::Refusal("unknown-terminator")),
        }
    }

    // Local 0 is the return place. An Index bound there is a helper that
    // answers a slot (`pin_self`: pin, then `shadow_stack_len() - 1`).
    // The Sub/Len statements that defined it are stripped as specials, and
    // `Return` does not mention local 0, so the rewritten body would return
    // an unbound local. Callers keep that index (`from_slot`) and lower the
    // call as `getattr(recv, method)` (`CallTarget::Method`); a broken
    // callee graph blocks that getattr. Refuse: the Pin lives in the
    // caller's RootScope and is only erasable if the helper is inlined.
    if matches!(
        plan.specials.get(&0),
        Some(Special::Index(_) | Special::IndexPair(_))
    ) {
        return Err(WalkErr::Refusal("returns-slot-index"));
    }

    // Temporaries that only build a getter capture (borrow of the index, the
    // closure aggregate, the receiver borrow, the argument tuple) bind
    // nothing once the call is a slot copy.  Another use — passing the
    // closure, storing it — would still run the real `shadow_stack_get`.
    if !capture_temps.is_empty() {
        if !super::capture_carriers_stay_inside(body, &capture_temps, &getter_bbs, &forward_blocks)
        {
            return Err(WalkErr::Refusal("capture-escapes"));
        }
        for (bb, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb] || !visited[bb] {
                continue;
            }
            for (si, stmt) in block.statements.iter().enumerate() {
                if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                    && place_local(place).is_some_and(|dest| capture_temps.contains(dest))
                {
                    plan.removed.entry(bb).or_default().push(si);
                }
            }
        }
    }

    // Every surviving mention of a guard, borrow or index has to be one of the
    // shapes rewritten above; anything else would read a local no block binds.
    let mut watched = bit_set::BitSet::new();
    for (local, special) in &plan.specials {
        // Slice bases stay assigned (`dest = 0`) and their `base + k`
        // arithmetic is kept, so a surviving mention is a use of a
        // bound local. Guard / Alias / Index defs are stripped.
        if !is_slice_special(*special) {
            watched.insert(*local);
        }
    }
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        if !visited[bb] {
            continue;
        }
        let removed = plan.removed.get(&bb);
        for (si, stmt) in block.statements.iter().enumerate() {
            if removed.is_some_and(|r| r.contains(&si)) {
                continue;
            }
            if matches!(
                stmt.stmt_kind_ref(),
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Borrowck(_))
            ) {
                continue;
            }
            if super::mentions_local(stmt.kind_value(), &watched) {
                return Err(WalkErr::Refusal("unmodeled-use-in-statement"));
            }
        }
        if plan.terms.contains_key(&bb) {
            continue;
        }
        // `check_kind` is panic-message metadata; lowering reads only
        // `cond` (`mentions_local_outside_raw_array_items`). A later
        // `args[i]` BoundsCheck may still name a guard-borrow local in
        // that metadata after Charon reused the id.
        if mentions_local_skipping_check_kind(block.terminator.kind_value(), &watched) {
            // BoundsCheck / overflow asserts on erased slot-index
            // arithmetic (`shadow_stack_len` + `+ k`, or the array
            // index of `publish_roots(&[...])`). Failure is the
            // panic edge; the offsets are compile-time so the check
            // is statically true. Guard/Alias mentions stay: rewriting
            // those Goto'd past Drop of a surviving RootScope.
            let target = match block.term_ref(llbc) {
                Ok(TermKind::Assert { assert, target, .. })
                    if assert_is_slot_index_bounds_check(
                        assert,
                        &plan.specials,
                        &const_locals,
                        llbc,
                        &merge_min,
                    ) =>
                {
                    *target
                }
                _ => return Err(WalkErr::Refusal("unmodeled-use-in-terminator")),
            };
            plan.terms.insert(
                bb,
                TermRewrite {
                    target,
                    stmts: Vec::new(),
                },
            );
        }
    }
    plan.unreachable = (0..n_blocks).filter(|bb| !visited[*bb]).collect();
    plan.slot_count = plan.slot_intern.len();
    Ok(Some(plan))
}

fn leaf_unmodeled_for_erase(leaf: Leaf) -> bool {
    matches!(leaf, Leaf::Unmodeled | Leaf::CopyRange)
}

fn is_rooted_items_local(body: &Unstructured, llbc: &Llbc, local: usize) -> bool {
    let Some(decl) = body.locals.locals.get(local) else {
        return false;
    };
    super::output_adt_def_id_free(&decl.ty, llbc)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|t| {
            let path = t.item_meta.name_path();
            path.rsplit("::").next() == Some("RootedItems")
                && path.split("::").any(|s| s == super::ROOT_SCOPE_MODULE)
        })
}

pub(super) fn is_root_scope_local(body: &Unstructured, llbc: &Llbc, local: usize) -> bool {
    let Some(decl) = body.locals.locals.get(local) else {
        return false;
    };
    super::output_adt_def_id_free(&decl.ty, llbc)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|t| {
            let path = t.item_meta.name_path();
            super::gc_root_scope_type_path(&path) || super::gc_rooted_items_type_path(&path)
        })
}

/// Drop of the guard / `root_scope_close` rewinds to the open depth, so
/// predecessors may hold different scalar depths. Slots are locals; the
/// close does not read them (`__majit_wrap_dict_descr_keys` early return
/// vs `pin_root`).
fn block_is_sink(body: &Unstructured, llbc: &Llbc, bb: usize) -> bool {
    matches!(
        body.body.get(bb).and_then(|b| b.term_ref(llbc).ok()),
        Some(
            TermKind::Abort(_)
                | TermKind::Panic { .. }
                | TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::UndefinedBehavior
        )
    )
}

fn block_rewinds_scope(
    body: &Unstructured,
    llbc: &Llbc,
    classified: &HashMap<usize, (Leaf, bool)>,
    specials: &HashMap<usize, Special>,
    bb: usize,
) -> bool {
    if matches!(
        classified.get(&bb),
        Some((Leaf::Close | Leaf::ItemsClose, _))
    ) {
        return true;
    }
    match body.body.get(bb).and_then(|b| b.term_ref(llbc).ok()) {
        Some(TermKind::Drop { place, .. }) => {
            matches!(
                place_local(place).and_then(|l| specials.get(&l)),
                Some(Special::Guard(_))
            ) || place_local(place).is_some_and(|l| is_root_scope_local(body, llbc, l))
        }
        _ => false,
    }
}

/// `(*slice)[k]` as a `Use` rvalue. Lowering's `pair_getitem` /
/// `guard_item_index` is the bounds check rustc would emit for that
/// Index; a separate Assert terminator would need extra blocks.
fn slice_elem_use(
    body: &Unstructured,
    slice_local: usize,
    dest_place: &Value,
    k: usize,
) -> Result<Value, Refusal> {
    let slice_ty = body
        .locals
        .locals
        .get(slice_local)
        .map(|loc| ty_ref_json(&loc.ty))
        .ok_or("slice-untyped")?;
    let elem_ty = dest_place.get("ty").cloned().ok_or("get-dest-untyped")?;
    let local_place = json!({ "kind": { "Local": slice_local }, "ty": slice_ty });
    let deref = json!({ "kind": { "Projection": [local_place, "Deref"] }, "ty": slice_ty });
    let offset = json!({
        "Const": [
            { "Integer": { "Unsigned": ["Usize", k.to_string()] } },
            { "Deduplicated": 0 }
        ]
    });
    let index_place = json!({
        "kind": {
            "Projection": [deref, { "Index": { "offset": offset, "from_end": false } }]
        },
        "ty": elem_ty
    });
    Ok(json!({ "Use": [{ "Copy": index_place }, "Yes"] }))
}

/// Unique forward `Assign` of `local`, with its block and statement index.
fn unique_assign_stmt<'a>(
    body: &'a Unstructured,
    local: usize,
    assigned: &HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, usize, &'a majit_charon_reader::ullbc::Statement)> {
    if assigned.get(&local) != Some(&1) {
        return None;
    }
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks.get(bb).copied().unwrap_or(false) {
            continue;
        }
        for (si, stmt) in block.statements.iter().enumerate() {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && place_local(place) == Some(local)
            {
                return Some((bb, si, stmt));
            }
        }
    }
    None
}

/// Same walk as [`resolve_published`], across forward-reachable
/// single-def locals, so an aggregate built in a predecessor is resolved.
fn resolve_published_array_in_body(
    body: &Unstructured,
    slice: Option<usize>,
    assigned: &HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, usize, Vec<Value>)> {
    let mut local = slice?;
    for _ in 0..8 {
        let assigns = assigned.get(&local).copied().unwrap_or(0);
        if assigns != 1 {
            return None;
        }
        let (bb, si, stmt) = unique_assign_stmt(body, local, assigned, forward_blocks)?;
        let Ok(StmtKind::Assign(_, rvalue)) = stmt.stmt_kind_ref() else {
            return None;
        };
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                local = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                local = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, _) if kind.get("Array").is_some() => {
                let operands = stmt.kind_value()["Assign"][1]["Aggregate"][1]
                    .as_array()?
                    .clone();
                return Some((bb, si, operands));
            }
            _ => return None,
        }
    }
    None
}

/// A `pin_roots` / `publish` argument this pass can answer `get` from.
enum Published {
    /// Same-block `[x, y, ..]` aggregate: scalar-replaced into slot locals.
    Array {
        stmt_index: usize,
        operands: Vec<Value>,
    },
    /// A slice local the body already holds. `get(base + k)` is
    /// `getarrayitem` of that list (`pin_roots_incoming_slice` /
    /// `cdataobj.py descr_call`).
    Slice {
        local: usize,
        ty: Value,
        deref_ty: Value,
    },
}

/// Follow a published slice back to the array literal it borrows, in the same
/// block: `_a = [x, y]; _r = &_a; _s = &*_r; _p = _s as &[_]`.  A leftover
/// local after that walk is the list itself — an incoming `&[T]` or a slice
/// the body built — which is the `pin_roots_incoming_slice` shape.
///
/// The array's statement index is this block's own. Cleanup blocks stay in
/// the body at their extracted indices and are not renumbered out of the way.
fn resolve_published(
    body: &Unstructured,
    block: &majit_charon_reader::ullbc::BasicBlock,
    slice: Option<usize>,
    llbc: &Llbc,
) -> Option<Published> {
    let mut want = slice?;
    let mut last_deref_ty: Option<Value> = None;
    for (si, stmt) in block.statements.iter().enumerate().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if place_local(place) != Some(want) {
            continue;
        }
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                want = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } => {
                if let Some(d) = deref_of_local(src) {
                    last_deref_ty = Some(ty_ref_json(&src.ty));
                    want = d;
                } else {
                    want = place_local(src)?;
                }
            }
            Rvalue::Aggregate(kind, _) if kind.get("Array").is_some() => {
                let operands = stmt.kind_value()["Assign"][1]["Aggregate"][1]
                    .as_array()?
                    .clone();
                return Some(Published::Array {
                    stmt_index: si,
                    operands,
                });
            }
            _ => return None,
        }
    }
    if is_root_scope_local(body, llbc, want) {
        return None;
    }
    let loc = body.locals.locals.get(want)?;
    let ty = ty_ref_json(&loc.ty);
    let deref_ty = last_deref_ty.unwrap_or_else(|| ty.clone());
    Some(Published::Slice {
        local: want,
        ty,
        deref_ty,
    })
}

/// A value of `ty` that is live at entry: the first argument of that type.
/// Interned `Abs` slots used after a later switch restore are live through
/// the join on every predecessor (`thread_undefined_op_operands` /
/// `edge_args`). The seed fills short arms so those locals are assigned;
/// `slot_for_get` still refuses an `AtMerge` read of `j >= merge_min`
/// (`ll_getitem_nonneg`). A pin on the long arm overwrites the seed, and
/// `constfold.rewire_link_for_known_exitswitch` restores that arm's Const
/// so the get copies the pin.
fn slot_seed_rvalue(u: &Value, ty: &Value) -> Option<Value> {
    let locals = u["locals"]["locals"].as_array()?;
    let arg_count = u["locals"]["arg_count"].as_u64()? as usize;
    (1..=arg_count).find_map(|i| {
        (locals.get(i)?.get("ty") == Some(ty))
            .then(|| json!({ "Use": [{ "Copy": local_place_json(i, ty) }, "Yes"] }))
    })
}

/// Rewrite `fd`'s body with its root brackets scalar-replaced.
///
/// `Ok(None)`: the body has no bracket.  `Err`: it has one this pass refuses;
/// the reason names the shape.
pub(super) fn erase_shadow_stack(
    fd: &FunDecl,
    body: &Unstructured,
    llbc: &Llbc,
) -> Result<Option<Unstructured>, Refusal> {
    if fd.body.is_none() {
        return Ok(None);
    }
    // The root-stack runtime itself implements the leaves this pass removes;
    // its bodies stay as written for the callers that keep their brackets.
    if fd.item_meta.name_path().contains("::gc_roots::") {
        return Ok(None);
    }
    // `body` keeps every extracted block at its original index (promoted
    // constants already spliced). The walk records only blocks reached from
    // bb0; `on_unwind` is not a successor, so a cleanup block stays in the
    // body and is not entered. Rewriting against any other numbering is how
    // a same-block array publish was refused as unresolved.
    //
    // The JSON rebuild parses every terminator kind. A body with no bracket
    // never reads that tree (`analyze` returns `None`), so build it only
    // once a plan exists.
    let plan = match analyze(body, llbc)? {
        Some(plan) => plan,
        None => return Ok(None),
    };
    let mut u = unstructured_json_from_body(body);
    let span = u["span"].clone();
    // Interned slot locals. Seed them in bb0 so a merge whose predecessors
    // pinned different counts still has every `Abs(j)` assigned on the short
    // arms: those locals are live through the shared suffix. An `AtMerge`
    // get of `j >= merge_min` is still refused (`ll_getitem_nonneg`); a
    // later switch that restores Const reads the arm's own pin, which
    // overwrote the seed (`constfold.constant_diffuse`).
    let slot_seed = if plan.slot_count > 0 {
        let ty = plan.slot_ty.clone().ok_or("slot-untyped")?;
        let seed = slot_seed_rvalue(&u, &ty);
        let locals = u["locals"]["locals"].as_array_mut().ok_or("body-json")?;
        let base = locals.len();
        for k in 0..plan.slot_count {
            locals.push(json!({
                "index": base + k,
                "name": null,
                "span": span,
                "ty": ty.clone(),
            }));
        }
        Some((base, ty, seed))
    } else {
        None
    };
    let blocks = u["body"].as_array_mut().ok_or("body-json")?;
    for (bb, block) in blocks.iter_mut().enumerate() {
        // An unreachable block may still name a guard or an index whose
        // definition is gone; nothing runs it.
        if plan.unreachable.contains(&bb) {
            block["statements"] = Value::Array(Vec::new());
            block["terminator"]["kind"] = json!({ "Abort": "UnwindTerminate" });
            continue;
        }
        let removed = plan.removed.get(&bb);
        let term = plan.terms.get(&bb);
        let has_inserts = plan.inserted_after.keys().any(|(b, _)| *b == bb);
        let seed_here = bb == 0 && slot_seed.as_ref().is_some_and(|(_, _, s)| s.is_some());
        if removed.is_none() && term.is_none() && !has_inserts && !seed_here {
            continue;
        }
        let stmt_span = block["terminator"]["span"].clone();
        let make_stmt = |kind: Value| {
            json!({
                "span": stmt_span,
                "kind": kind,
                "comments_before": [],
            })
        };
        let old = block["statements"].as_array().cloned().unwrap_or_default();
        let mut new = Vec::with_capacity(old.len() + plan.slot_count);
        if seed_here && let Some((base, ty, Some(rv))) = &slot_seed {
            for k in 0..plan.slot_count {
                new.push(make_stmt(assign_json(
                    local_place_json(*base + k, ty),
                    rv.clone(),
                )));
            }
        }
        for (si, stmt) in old.into_iter().enumerate() {
            if !removed.is_some_and(|r| r.contains(&si)) {
                new.push(stmt);
            }
            if let Some(extra) = plan.inserted_after.get(&(bb, si)) {
                new.extend(extra.iter().cloned().map(make_stmt));
            }
        }
        if let Some(term) = term {
            new.extend(term.stmts.iter().cloned().map(make_stmt));
            block["terminator"]["kind"] = json!({ "Goto": { "target": term.target } });
        }
        block["statements"] = Value::Array(new);
    }
    serde_json::from_value::<Unstructured>(u)
        .map(Some)
        .map_err(|_| "rewritten-body-unparsable")
}

fn span_ref_json(span: &SpanRef) -> Value {
    match span {
        SpanRef::Inline(data) => json!({
            "data": {
                "file_id": data.file_id,
                "beg": { "line": data.beg.line, "col": data.beg.col },
                "end": { "line": data.end.line, "col": data.end.col },
            }
        }),
        SpanRef::Deduplicated(id) => json!({ "Deduplicated": id }),
    }
}

fn ty_ref_json(ty: &TyRef) -> Value {
    match ty {
        TyRef::Dedup { id } => json!({ "Deduplicated": id }),
        TyRef::Inline { value: (id, v) } => json!({ "Value": [*id, &**v] }),
        TyRef::Other(v) => (*v.0).clone(),
    }
}

/// JSON for `body` as the rewrite sees it: same blocks, same statement
/// indices, same terminator targets as the typed CFG.
fn unstructured_json_from_body(body: &Unstructured) -> Value {
    let locals: Vec<Value> = body
        .locals
        .locals
        .iter()
        .map(|loc| {
            json!({
                "index": loc.index,
                "name": loc.name,
                "span": span_ref_json(&loc.span),
                "ty": ty_ref_json(&loc.ty),
            })
        })
        .collect();
    let blocks: Vec<Value> = body
        .body
        .iter()
        .map(|bb| {
            let statements: Vec<Value> = bb
                .statements
                .iter()
                .map(|st| {
                    json!({
                        "span": span_ref_json(&st.span),
                        "kind": st.kind_value().clone(),
                    })
                })
                .collect();
            json!({
                "statements": statements,
                "terminator": {
                    "span": bb.terminator.span.as_ref().map(span_ref_json),
                    "kind": bb.terminator.kind_value().clone(),
                },
                "kind": match bb.kind {
                    UnwindKind::Regular => "Regular",
                    UnwindKind::Cleanup => "Cleanup",
                    UnwindKind::Terminate => "Terminate",
                },
            })
        })
        .collect();
    json!({
        "span": span_ref_json(&body.span),
        "locals": {
            "arg_count": body.locals.arg_count,
            "locals": locals,
        },
        "body": blocks,
    })
}

/// [`erase_shadow_stack`], falling back to the body as extracted.  A refusal is
/// counted under its reason.
pub(super) fn erase_or_keep(fd: &FunDecl, body: Unstructured, llbc: &Llbc) -> Unstructured {
    match erase_shadow_stack(fd, &body, llbc) {
        Ok(Some(erased)) => erased,
        Ok(None) => body,
        Err(reason) => {
            crate::decline::record(
                crate::decline::gate::SHADOW_STACK_ERASE,
                reason,
                format_args!("{}", fd.item_meta.name_path()),
            );
            body
        }
    }
}

/// Census-path only: every local body's walk under the registered
/// effects must equal the registered effect. A registered Observes
/// whose own walk is clean (`why=-`) is the sticky-fixpoint bug.
fn debug_check_registered_effects_match_rewalk(llbc: &Llbc) {
    let effect = |path: &str| callee_effect_of(llbc, path);
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let name = fd.item_meta.name_path();
        let registered = effect(&name);
        let walked = stack_walk(&body, llbc, &effect).effect();
        if registered != walked {
            panic!(
                "stack-effect invariant: {name} registered {} rewalk {} why={}",
                registered.as_str(),
                walked.as_str(),
                stack_walk(&body, llbc, &effect)
                    .why
                    .as_deref()
                    .unwrap_or("-")
            );
        }
    }
}

/// Every local body with a root bracket, bucketed by what the scalar
/// replacement did with it: `"erased"`, or the refusal reason.  The subject
/// list names the bodies in each bucket.
pub fn census(llbc: &Llbc) -> std::collections::BTreeMap<&'static str, Vec<String>> {
    ensure_stack_sensitive_fns(llbc);
    debug_check_registered_effects_match_rewalk(llbc);
    let detail = std::env::var("CENSUS_DETAIL").is_ok();
    let mut out: std::collections::BTreeMap<&'static str, Vec<String>> = Default::default();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let bucket = match erase_shadow_stack(fd, &body, llbc) {
            Ok(Some(_)) => "erased",
            Ok(None) => continue,
            Err(reason) => reason,
        };
        let name = fd.item_meta.name_path();
        if detail {
            let walk = stack_walk(&body, llbc, &|path| callee_effect_of(llbc, path));
            let callee = first_non_none_callee(&body, llbc);
            eprintln!(
                "[census-detail] {name}:{bucket} why={} callee={} effect={}",
                walk.why.as_deref().unwrap_or("-"),
                callee.as_ref().map(|(p, _)| p.as_str()).unwrap_or("-"),
                callee
                    .as_ref()
                    .map(|(_, effect)| effect.as_str())
                    .unwrap_or("-"),
            );
        }
        out.entry(bucket).or_default().push(name);
    }
    out
}

// ---------------------------------------------------------------------------
// Stack effects a caller cannot see past
// ---------------------------------------------------------------------------

/// Stack depth relative to a body's entry.
/// `Known(d)` is exact. `AtLeast(d)` is a lower bound: a publish of
/// unknown length or a LeavesAbove callee ran at that depth.
/// `Unknown` is no bound (a Close of an unknown guard, or a join of
/// incompatible depths).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    Known(usize),
    /// Lower bound relative to entry. `Above` was `AtLeast(1)`.
    AtLeast(usize),
    Unknown,
}

impl Depth {
    /// The body holds at least one slot of its own.
    fn above_entry(self) -> bool {
        matches!(self, Depth::Known(1..) | Depth::AtLeast(1..))
    }

    fn lower_bound(self) -> Option<usize> {
        match self {
            Depth::Known(d) | Depth::AtLeast(d) => Some(d),
            Depth::Unknown => None,
        }
    }

    fn join(self, other: Depth) -> Depth {
        if self == other {
            self
        } else {
            match (self.lower_bound(), other.lower_bound()) {
                (Some(a), Some(b)) => {
                    let m = a.min(b);
                    if self == Depth::Known(m) && other == Depth::Known(m) {
                        Depth::Known(m)
                    } else {
                        Depth::AtLeast(m)
                    }
                }
                _ => Depth::Unknown,
            }
        }
    }

    fn add(self, n: usize) -> Depth {
        match self {
            Depth::Known(d) => Depth::Known(d + n),
            Depth::AtLeast(d) => Depth::AtLeast(d + n),
            Depth::Unknown => Depth::Unknown,
        }
    }

    fn checked_sub(self, n: usize) -> Option<Depth> {
        match self {
            Depth::Known(d) => d.checked_sub(n).map(Depth::Known),
            Depth::AtLeast(_) | Depth::Unknown => Some(Depth::Unknown),
        }
    }

    /// The depth after a callee that may leave slots behind, or a
    /// publish of unknown length: `Known(d)` becomes `AtLeast(d)`.
    fn after_unknown_push(self) -> Depth {
        match self {
            Depth::Known(d) | Depth::AtLeast(d) => Depth::AtLeast(d),
            Depth::Unknown => Depth::Unknown,
        }
    }
}

/// Effect of a callee on the caller's shadow stack.
/// `observes` and `leaves_above` are independent: a body that reads
/// below entry can also return with slots still published, and a caller
/// whose bracket is erased has no Close left to rewind them.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct CalleeEffect {
    observes: bool,
    leaves_above: bool,
    param_slots: Vec<u8>,
    /// Every reachable `Return` yields an own index (>= entry).
    returns_index: bool,
}

impl CalleeEffect {
    fn none() -> Self {
        Self::default()
    }

    /// Reads below entry and returns at entry depth on every path.
    #[cfg(test)]
    fn observes() -> Self {
        Self {
            observes: true,
            leaves_above: false,
            param_slots: Vec::new(),
            returns_index: false,
        }
    }

    /// Reads below entry and may end at any depth.
    fn worst() -> Self {
        Self {
            observes: true,
            leaves_above: true,
            param_slots: Vec::new(),
            returns_index: false,
        }
    }

    /// The erased caller cannot keep this call: something the callee
    /// publishes or indexes outlives it, and nothing rewinds a leftover
    /// once the caller's Open / Close are gone.
    fn refuses_erase(&self) -> bool {
        self.leaves_above || !self.param_slots.is_empty() || self.returns_index
    }

    #[cfg(test)]
    fn leaves_above() -> Self {
        Self {
            observes: false,
            leaves_above: true,
            param_slots: Vec::new(),
            returns_index: false,
        }
    }

    fn is_none(&self) -> bool {
        !self.observes && !self.leaves_above && self.param_slots.is_empty() && !self.returns_index
    }

    fn as_str(&self) -> &'static str {
        if self.observes && self.leaves_above {
            "ObservesLeaves"
        } else if self.observes {
            "Observes"
        } else if !self.param_slots.is_empty() {
            "ParamSlots"
        } else if self.leaves_above {
            "LeavesAbove"
        } else if self.returns_index {
            "ReturnsIndex"
        } else {
            "None"
        }
    }
}

/// Registered effect of `path`: `None` if not sensitive or proven
/// neutral; otherwise the union of the sets it was registered in. A
/// sensitive path in none of them comes from an artefact that did not
/// classify it, and is taken at its worst.
fn callee_effect_of(llbc: &Llbc, path: &str) -> CalleeEffect {
    if llbc.is_stack_depth_neutral_fn(path) {
        return CalleeEffect::none();
    }
    let param_slots = llbc.stack_param_slots(path).unwrap_or_default();
    let leaves = llbc.is_stack_leaves_above_fn(path);
    let returns_index = llbc.is_stack_returns_index_fn(path);
    let observes = llbc.is_stack_observes_fn(path);
    let effect = CalleeEffect {
        observes,
        leaves_above: leaves,
        param_slots,
        returns_index,
    };
    if effect.is_none() && llbc.is_stack_sensitive_fn(path) {
        return CalleeEffect::worst();
    }
    effect
}

fn first_non_none_callee(body: &Unstructured, llbc: &Llbc) -> Option<(String, CalleeEffect)> {
    let forward_blocks = super::forward_reachable_mask(llbc, body);
    for (bb_idx, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && let Some(path) = callee_path(call, llbc)
        {
            let effect = callee_effect_of(llbc, &path);
            if !effect.is_none() {
                return Some((path, effect));
            }
        }
    }
    None
}

/// The callee path a call terminator names, for a statically resolved callee.
fn callee_path(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> Option<String> {
    let majit_charon_reader::ullbc::CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let id = super::regular_call_fun_decl_id(&reg.kind)?;
    llbc.fn_by_id(id).map(|fd| fd.item_meta.name_path())
}

/// `core::mem::drop` / `std::mem::drop`, the Call spelling of `drop(x)`.
fn is_mem_drop_path(path: &str) -> bool {
    matches!(path, "core::mem::drop" | "std::mem::drop")
}

/// Depth of a watched Guard that `core::mem::drop` closes, if this Call is
/// that drop of the Guard itself or of a same-block `_t = Move(guard)` copy.
fn mem_drop_guard_depth(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
    specials: &HashMap<usize, Special>,
) -> Option<IdxFact> {
    if !callee_path(call, llbc)
        .as_deref()
        .is_some_and(is_mem_drop_path)
        || call.args.len() != 1
    {
        return None;
    }
    let local = operand_local(call.args.first()?)?;
    match specials.get(&local) {
        Some(Special::Guard(d)) => Some(*d),
        _ => None,
    }
}

fn call_fun_decl_id(call: &majit_charon_reader::ullbc::CallPayload) -> Option<u64> {
    let majit_charon_reader::ullbc::CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    super::regular_call_fun_decl_id(&reg.kind)
}

fn call_returns_owned_scope(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> bool {
    call_fun_decl_id(call).is_some_and(|id| super::fn_returns_owned_scope(llbc, id))
}

/// Locals that hold a nested owned guard this body received, plus the
/// borrows and copies the artefact passes as the method receiver.
fn owned_scope_receiver_locals(
    body: &Unstructured,
    llbc: &Llbc,
    forward_blocks: &[bool],
) -> HashSet<usize> {
    let mut owned = HashSet::new();
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && callee_path(call, llbc)
                .as_deref()
                .is_some_and(|path| callee_effect_of(llbc, path).refuses_erase())
            && call_returns_owned_scope(call, llbc)
            && let Some(dest) = place_local(&call.dest)
        {
            owned.insert(dest);
        }
    }
    if owned.is_empty() {
        return owned;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (bb, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let Some(dest) = place_local(place) else {
                    continue;
                };
                if owned.contains(&dest) {
                    continue;
                }
                let src = match rvalue {
                    Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                        place_local(src).or_else(|| deref_of_local(src))
                    }
                    Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                        place_local(src).or_else(|| deref_of_local(src))
                    }
                    _ => None,
                };
                if src.is_some_and(|l| owned.contains(&l)) {
                    owned.insert(dest);
                    changed = true;
                }
            }
        }
    }
    owned
}

/// The leftover this call publishes is a nested owned guard the caller
/// holds, or a ParamSlots read of that guard, not this body's slot locals.
fn callee_leftover_is_owned_scope(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
    owned: &HashSet<usize>,
) -> bool {
    let effect = callee_path(call, llbc)
        .as_deref()
        .map(|path| callee_effect_of(llbc, path))
        .unwrap_or_else(CalleeEffect::none);
    if effect.observes {
        return false;
    }
    if call_returns_owned_scope(call, llbc) {
        return true;
    }
    !effect.param_slots.is_empty()
        && !effect.leaves_above
        && !effect.returns_index
        && call
            .args
            .first()
            .and_then(operand_local)
            .is_some_and(|l| owned.contains(&l))
}

/// Result of the per-path shadow-stack walk.
struct StackWalk {
    /// `None` when every reachable return / unwind restores the entry
    /// depth and no instruction reads or writes a slot below it.
    why: Option<String>,
    /// A reachable block opened a `RootScope`. Unreachable open/close
    /// pairs do not set this.
    saw_open: bool,
    /// Some instruction reads or writes a slot below entry.
    reads_below: bool,
    /// Some reachable return / unwind is `Known(d>=1)` or `AtLeast(_)`.
    leaves_above: bool,
    /// Some reachable return / unwind is `Unknown`, or the walk bailed
    /// because depth grew in a loop.
    unknown_exit: bool,
    /// Parameter positions this body uses as shadow-stack indices.
    param_slots: Vec<u8>,
    /// Every reachable `Return` yields a local-0 own index (the value
    /// itself or a struct field) with a lower bound relative to entry.
    returns_index: bool,
}

impl StackWalk {
    fn effect(&self) -> CalleeEffect {
        // An exit the walk could not bound may sit below entry (the
        // caller's slots are gone) or above it (slots are left), so it
        // sets both.
        let observes = self.reads_below || self.unknown_exit;
        CalleeEffect {
            observes,
            leaves_above: self.leaves_above || self.unknown_exit,
            param_slots: if observes {
                Vec::new()
            } else {
                self.param_slots.clone()
            },
            returns_index: self.returns_index && !observes,
        }
    }
}

/// Depth-neutral: every reachable path restores the entry depth, the body
/// never reads a caller-owned slot, Observes callees are refused only at
/// entry depth (inside our Open they see our pins; Close rewinds),
/// LeavesAbove is allowed, no parameter-indexed slot access, no returned
/// own-index summary, and a `RootScope` opens on some reachable path.
fn body_is_depth_neutral(
    body: &Unstructured,
    llbc: &Llbc,
    effect: &dyn Fn(&str) -> CalleeEffect,
) -> bool {
    let walk = stack_walk(body, llbc, effect);
    walk.why.is_none() && walk.saw_open && walk.param_slots.is_empty() && !walk.returns_index
}

fn clear_tracked(
    dest: usize,
    params: &mut HashMap<usize, u8>,
    param_pairs: &mut HashMap<usize, u8>,
    indices: &mut HashMap<usize, Depth>,
    pairs: &mut HashMap<usize, Depth>,
    field_indices: &mut HashMap<(usize, u64), Depth>,
    field_params: &mut HashMap<(usize, u64), u8>,
) {
    params.remove(&dest);
    param_pairs.remove(&dest);
    indices.remove(&dest);
    pairs.remove(&dest);
    field_indices.retain(|&(l, _), _| l != dest);
    field_params.retain(|&(l, _), _| l != dest);
}

fn copy_field_maps(
    src: usize,
    dest: usize,
    field_indices: &mut HashMap<(usize, u64), Depth>,
    field_params: &mut HashMap<(usize, u64), u8>,
) {
    let idx: Vec<(u64, Depth)> = field_indices
        .iter()
        .filter(|((l, _), _)| *l == src)
        .map(|((_, f), k)| (*f, *k))
        .collect();
    for (f, k) in idx {
        field_indices.insert((dest, f), k);
    }
    let ps: Vec<(u64, u8)> = field_params
        .iter()
        .filter(|((l, _), _)| *l == src)
        .map(|((_, f), p)| (*f, *p))
        .collect();
    for (f, p) in ps {
        field_params.insert((dest, f), p);
    }
}

fn stack_walk(
    body: &Unstructured,
    llbc: &Llbc,
    effect: &dyn Fn(&str) -> CalleeEffect,
) -> StackWalk {
    let n_blocks = body.body.len();
    if n_blocks == 0 {
        return StackWalk {
            why: None,
            saw_open: false,
            reads_below: false,
            leaves_above: false,
            unknown_exit: false,
            param_slots: Vec::new(),
            returns_index: false,
        };
    }
    let mut why = String::new();
    let mut saw_open = false;
    let mut depth_in: Vec<Option<Depth>> = vec![None; n_blocks];
    let mut guards: HashMap<usize, Depth> = HashMap::new();
    let mut aliases: HashMap<usize, usize> = HashMap::new();
    let mut indices: HashMap<usize, Depth> = HashMap::new();
    let mut pairs: HashMap<usize, Depth> = HashMap::new();
    let mut params: HashMap<usize, u8> = HashMap::new();
    let mut param_pairs: HashMap<usize, u8> = HashMap::new();
    let mut field_indices: HashMap<(usize, u64), Depth> = HashMap::new();
    let mut field_params: HashMap<(usize, u64), u8> = HashMap::new();
    let arg_count = body.locals.arg_count as usize;
    for i in 1..=arg_count {
        if let Ok(p) = u8::try_from(i - 1) {
            params.insert(i, p);
        }
    }
    let mut param_slots: Vec<u8> = Vec::new();
    depth_in[0] = Some(Depth::Known(0));
    let mut queue: VecDeque<usize> = VecDeque::from([0]);
    let mut reads_below = false;
    // The first read below entry. The walk goes on past it so the exits
    // are still judged: a reader that also leaves slots is a different
    // callee from one that returns at its entry depth.
    let mut read_why: Option<String> = None;
    let mut leaves_above = false;
    let mut unknown_exit = false;
    let mut returns_index: Option<bool> = None;
    let mut rounds = 0usize;
    while let Some(bb) = queue.pop_front() {
        rounds += 1;
        if rounds > n_blocks * 8 + 64 {
            // A depth that keeps growing round a loop: slots pile up.
            return StackWalk {
                why: Some(read_why.unwrap_or_else(|| "depth-grows-in-loop".into())),
                saw_open,
                reads_below,
                leaves_above,
                unknown_exit: true,
                param_slots,
                returns_index: false,
            };
        }
        let mut depth = depth_in[bb].expect("queued blocks have a depth");
        let block = &body.body[bb];
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            let Some(dest) = place_local(place) else {
                continue;
            };
            match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    let guard = place_local(src)
                        .filter(|l| guards.contains_key(l))
                        .or_else(|| deref_of_local(src).and_then(|l| aliases.get(&l).copied()));
                    if let Some(g) = guard {
                        aliases.insert(dest, g);
                    }
                    // A borrow of an own index (or of a struct that
                    // carries one) still names that index, so a method
                    // call on `storage` can pass the `&mut` to a
                    // ParamSlots callee.
                    if let Some(l) = place_local(src).or_else(|| deref_of_local(src)) {
                        if let Some(&k) = indices.get(&l) {
                            indices.insert(dest, k);
                        }
                        if let Some(&p) = params.get(&l) {
                            params.insert(dest, p);
                        }
                        copy_field_maps(l, dest, &mut field_indices, &mut field_params);
                    }
                }
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                    let mut param_from = None;
                    let mut index_from = None;
                    let mut fields_from = None;
                    if let Some(l) = place_local(src) {
                        index_from = indices.get(&l).copied();
                        if let Some(g) = aliases.get(&l).copied() {
                            aliases.insert(dest, g);
                        }
                        if let Some(d) = guards.get(&l).copied() {
                            let entry = guards.entry(dest).or_insert(d);
                            *entry = entry.join(d);
                        }
                        param_from = params.get(&l).copied();
                        fields_from = Some(l);
                    } else if let Some((base, field)) = tuple_field_of_local(src) {
                        // Checked-add pairs: `.0` is the sum. Unary ADT
                        // wrappers (Option::Some of an own index): the
                        // payload field is the same usize, so it stays
                        // >= entry. A usize field of a parameter is a
                        // value the caller supplied, so it records that
                        // parameter in param_slots when used as a Get.
                        if field == 0 {
                            index_from = pairs.get(&base).copied();
                            param_from = param_pairs.get(&base).copied();
                        }
                        if index_from.is_none() {
                            index_from = field_indices
                                .get(&(base, field))
                                .copied()
                                .or_else(|| indices.get(&base).copied());
                        }
                        if param_from.is_none() {
                            param_from = field_params
                                .get(&(base, field))
                                .copied()
                                .or_else(|| params.get(&base).copied());
                        }
                    }
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    if let Some(k) = index_from {
                        indices.insert(dest, k);
                    }
                    if let Some(p) = param_from {
                        params.insert(dest, p);
                    }
                    if let Some(src_local) = fields_from {
                        copy_field_maps(src_local, dest, &mut field_indices, &mut field_params);
                    }
                }
                Rvalue::Aggregate(_, operands) => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    // Each operand that is an own index stays one in that
                    // field: the struct stores the same usize. A unary wrap
                    // (Option::Some) is the same fact on the dest itself.
                    for (i, op) in operands.iter().enumerate() {
                        let Some(l) = operand_local(op) else {
                            continue;
                        };
                        if let Some(&k) = indices.get(&l) {
                            field_indices.insert((dest, i as u64), k);
                        }
                        if let Some(&p) = params.get(&l) {
                            field_params.insert((dest, i as u64), p);
                        }
                    }
                    if let [op] = operands.as_slice() {
                        if let Some(k) = operand_local(op).and_then(|l| indices.get(&l).copied()) {
                            indices.insert(dest, k);
                        } else if let Some(p) =
                            operand_local(op).and_then(|l| params.get(&l).copied())
                        {
                            params.insert(dest, p);
                        }
                    }
                }
                Rvalue::BinaryOp(op, lhs, rhs) => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    if let Some(arith) = binop_is_index_arith(op) {
                        let base =
                            |o: &Operand| operand_local(o).and_then(|l| indices.get(&l).copied());
                        // Unsigned add cannot produce a smaller index:
                        // Known(k)+x stays AtLeast(k), param p+x stays param p.
                        let offset = match (arith, base(lhs), base(rhs)) {
                            (IndexArith::Add { checked }, Some(k), _) => {
                                Some((add_unsigned_index(k, rhs, llbc), checked))
                            }
                            (IndexArith::Add { checked }, None, Some(k)) => {
                                Some((add_unsigned_index(k, lhs, llbc), checked))
                            }
                            (IndexArith::Sub { checked }, Some(k), None) => const_usize(rhs, llbc)
                                .and_then(|c| k.checked_sub(c).map(|n| (n, checked))),
                            _ => None,
                        };
                        if let Some((k, checked)) = offset {
                            if checked {
                                pairs.insert(dest, k);
                            } else {
                                indices.insert(dest, k);
                            }
                        } else {
                            let param_of = |o: &Operand| {
                                operand_local(o).and_then(|l| params.get(&l).copied())
                            };
                            let kept = match (arith, param_of(lhs), param_of(rhs)) {
                                (IndexArith::Add { checked }, Some(p), _) => Some((p, checked)),
                                (IndexArith::Add { checked }, None, Some(p)) => Some((p, checked)),
                                _ => None,
                            };
                            if let Some((p, checked)) = kept {
                                if checked {
                                    param_pairs.insert(dest, p);
                                } else {
                                    params.insert(dest, p);
                                }
                            }
                        }
                    }
                }
                _ => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                }
            }
        }
        // `None`: the block ends the walk.  Otherwise the successors and the
        // depth they receive.
        let mut successors: Vec<u64> = Vec::new();
        match block.term_ref(llbc) {
            Ok(TermKind::Call { call, target, .. }) => {
                successors.push(*target);
                if let Some(dest) = place_local(&call.dest) {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                }
                let path = callee_path(call, llbc);
                match classify_call(call, llbc) {
                    Some((leaf, method)) => {
                        let arg0 = usize::from(method);
                        let guard = call
                            .args
                            .first()
                            .and_then(operand_local)
                            .and_then(|l| aliases.get(&l).copied());
                        let items_guard = call.args.first().and_then(operand_local).and_then(|l| {
                            aliases
                                .get(&l)
                                .copied()
                                .or_else(|| guards.contains_key(&l).then_some(l))
                        });
                        match leaf {
                            Leaf::Open | Leaf::ItemsOpen => {
                                saw_open = true;
                                if let Some(dest) = place_local(&call.dest) {
                                    let entry = guards.entry(dest).or_insert(depth);
                                    *entry = entry.join(depth);
                                }
                            }
                            Leaf::Close | Leaf::ItemsClose => {
                                let g = if matches!(leaf, Leaf::ItemsClose) {
                                    items_guard
                                } else {
                                    guard
                                };
                                depth = g
                                    .and_then(|g| guards.get(&g).copied())
                                    .unwrap_or(Depth::Unknown);
                            }
                            Leaf::Base => {
                                if let (Some(dest), Some(g)) = (place_local(&call.dest), guard) {
                                    indices.insert(
                                        dest,
                                        guards.get(&g).copied().unwrap_or(Depth::Unknown),
                                    );
                                }
                            }
                            Leaf::Len => {
                                if let Some(dest) = place_local(&call.dest) {
                                    indices.insert(dest, depth);
                                }
                            }
                            Leaf::Pin | Leaf::ItemsPush => depth = depth.add(1),
                            Leaf::Publish => {
                                let slice = call.args.get(arg0).and_then(operand_local);
                                if let Some(dest) = place_local(&call.dest) {
                                    indices.insert(dest, depth);
                                }
                                match resolve_published_array_len(block, slice) {
                                    Some(n) => depth = depth.add(n),
                                    None => depth = depth.after_unknown_push(),
                                }
                            }
                            Leaf::Normalize
                            | Leaf::NormalizeMoved
                            | Leaf::ItemsNop
                            | Leaf::ItemsLen
                            | Leaf::ItemsIsEmpty => {}
                            Leaf::Get | Leaf::Set | Leaf::CopyRange => {
                                if !call
                                    .args
                                    .get(arg0)
                                    .and_then(operand_local)
                                    .is_some_and(|l| {
                                        accept_index(l, &indices, &params, &mut param_slots)
                                    })
                                {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                }
                            }
                            Leaf::ItemsGet => {
                                if items_guard.and_then(|g| guards.get(&g).copied()).is_none() {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                }
                            }
                            Leaf::ReloadTop => {
                                if !matches!(depth, Depth::Known(d) if d > 0) {
                                    reads_below = true;
                                    why = "reload-top-below".into();
                                }
                            }
                            Leaf::Unmodeled => {
                                reads_below = true;
                                depth = Depth::Unknown;
                                why = format!("unmodeled {}", path.as_deref().unwrap_or(""));
                            }
                        }
                    }
                    None => {
                        let callee = path
                            .as_deref()
                            .map(effect)
                            .unwrap_or_else(CalleeEffect::none);
                        if callee.observes {
                            // Unproven stack-sensitive callee. At
                            // Known(0) every live slot is the caller's,
                            // so the body is not neutral. Inside our
                            // own Open (`eval_slice_index` pin then
                            // `getindex_w`) only the depth is unknown:
                            // Close rewinds our guard, and
                            // `is_stack_depth_neutral_fn` still holds
                            // so BINARY_SLICE can erase its bracket.
                            if !depth.above_entry() {
                                reads_below = true;
                                why = format!(
                                    "unproven-stack-sensitive {}",
                                    path.as_deref().unwrap_or("")
                                );
                            }
                            // A reader that may also leave slots, or end
                            // below its entry, leaves no bound at all.
                            depth = if callee.leaves_above {
                                Depth::Unknown
                            } else {
                                depth
                            };
                        } else {
                            // pin() returns shadow_stack_len() captured at
                            // callee entry, an own index >= entry; the
                            // caller dest is the depth at the call.
                            let depth_at_call = depth;
                            if callee.leaves_above {
                                depth = depth.after_unknown_push();
                            }
                            for &p in &callee.param_slots {
                                if !call
                                    .args
                                    .get(p as usize)
                                    .and_then(operand_local)
                                    .is_some_and(|l| {
                                        accept_index(l, &indices, &params, &mut param_slots)
                                    })
                                {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                    break;
                                }
                            }
                            if callee.returns_index
                                && let Some(dest) = place_local(&call.dest)
                                && depth_at_call.lower_bound().is_some()
                            {
                                // pin() returns the entry Len; a struct
                                // whose field is an own index is the same
                                // bound at the call-site depth.
                                indices.insert(dest, depth_at_call);
                            }
                        }
                    }
                }
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                successors.push(*target);
                if let Some(g) = place_local(place).filter(|l| guards.contains_key(l)) {
                    depth = guards[&g];
                } else if place_local(place).is_some_and(|l| {
                    is_root_scope_local(body, llbc, l) || is_rooted_items_local(body, llbc, l)
                }) {
                    depth = Depth::Unknown;
                }
            }
            Ok(TermKind::Assert { target, .. }) => successors.push(*target),
            Ok(TermKind::Goto { target }) => successors.push(*target),
            Ok(TermKind::Switch { targets, .. }) => match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(a, b) => {
                    successors.extend([*a, *b]);
                }
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                    successors.extend(arms.iter().map(|(_, t)| *t));
                    successors.push(*default);
                }
            },
            Ok(TermKind::Return) => {
                let this_returns_index = indices.get(&0).is_some_and(|d| d.lower_bound().is_some())
                    || field_indices
                        .iter()
                        .any(|((l, _), d)| *l == 0 && d.lower_bound().is_some());
                returns_index = Some(returns_index.unwrap_or(true) && this_returns_index);
                note_exit(
                    depth,
                    "returns-at",
                    &mut why,
                    &mut leaves_above,
                    &mut unknown_exit,
                );
            }
            Ok(TermKind::UnwindResume | TermKind::UnwindTerminate) => note_exit(
                depth,
                "unwinds-at",
                &mut why,
                &mut leaves_above,
                &mut unknown_exit,
            ),
            _ => {}
        }
        if reads_below && read_why.is_none() {
            read_why = Some(why.clone());
        }
        for target in successors {
            let target = target as usize;
            if target >= n_blocks {
                continue;
            }
            let next = match depth_in[target] {
                None => depth,
                Some(old) => old.join(depth),
            };
            if depth_in[target] != Some(next) {
                depth_in[target] = Some(next);
                queue.push_back(target);
            }
        }
    }
    let why = if reads_below {
        read_why
    } else if unknown_exit || leaves_above {
        Some(why)
    } else {
        None
    };
    // A reader's parameter-indexed accesses add nothing to "reads below".
    if reads_below {
        param_slots.clear();
    }
    param_slots.sort_unstable();
    param_slots.dedup();
    StackWalk {
        why,
        saw_open,
        reads_below,
        leaves_above,
        unknown_exit,
        param_slots,
        returns_index: returns_index == Some(true),
    }
}

/// `Known(k) + c` stays `Known(k+c)` when `c` is a constant; any other
/// unsigned addend drops to `AtLeast(k)`. Adding cannot yield a smaller
/// unsigned index than the tracked lower bound.
fn add_unsigned_index(k: Depth, other: &Operand, llbc: &Llbc) -> Depth {
    match const_usize(other, llbc) {
        Some(c) => k.add(c),
        None => match k {
            Depth::Known(d) | Depth::AtLeast(d) => Depth::AtLeast(d),
            Depth::Unknown => Depth::Unknown,
        },
    }
}

fn accept_index(
    local: usize,
    indices: &HashMap<usize, Depth>,
    params: &HashMap<usize, u8>,
    param_slots: &mut Vec<u8>,
) -> bool {
    if let Some(d) = indices.get(&local) {
        return matches!(d, Depth::Known(_) | Depth::AtLeast(_));
    }
    if let Some(&p) = params.get(&local) {
        if !param_slots.contains(&p) {
            param_slots.push(p);
        }
        return true;
    }
    false
}

fn note_exit(
    depth: Depth,
    kind: &str,
    why: &mut String,
    leaves_above: &mut bool,
    unknown_exit: &mut bool,
) {
    match depth {
        Depth::Known(0) => {}
        Depth::Unknown => {
            *unknown_exit = true;
            *why = format!("{kind} {depth:?}");
        }
        Depth::Known(_) | Depth::AtLeast(_) => {
            *leaves_above = true;
            if !*unknown_exit && why.is_empty() {
                *why = format!("{kind} {depth:?}");
            }
        }
    }
}

/// [`resolve_published_array`] without the JSON: only the literal's length.
fn resolve_published_array_len(
    block: &majit_charon_reader::ullbc::BasicBlock,
    slice: Option<usize>,
) -> Option<usize> {
    let mut want = slice?;
    for stmt in block.statements.iter().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if place_local(place) != Some(want) {
            continue;
        }
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                want = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } => {
                want = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, operands) if kind.get("Array").is_some() => {
                return Some(operands.len());
            }
            _ => return None,
        }
    }
    None
}

/// The functions of `llbc` whose shadow-stack effect a caller cannot see
/// past.  Callees from artefacts linked earlier are answered by what they
/// registered on `llbc` ([`Llbc::is_stack_sensitive_fn`],
/// [`Llbc::is_stack_leaves_above_fn`]).
pub fn discover_stack_sensitive_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_fn_effects(llbc).0
}

/// Local bodies that never read below entry but can return with slots
/// still published above it. Harvested in link order like
/// [`discover_stack_sensitive_fns`].
pub fn discover_stack_leaves_above_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_fn_effects(llbc).1
}

/// Local bodies that index the shadow stack through a parameter.
pub fn discover_stack_param_slots_fns(llbc: &Llbc) -> Vec<(String, Vec<u8>)> {
    discover_stack_fn_effects(llbc).2
}

/// Local bodies whose every reachable return is an own index relative
/// to entry. Harvested in link order like
/// [`discover_stack_sensitive_fns`].
pub fn discover_stack_returns_index_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_fn_effects(llbc).3
}

/// Sensitive, leaves-above, param-slots, returns-index and observes
/// sets from one fixpoint, so harvest does not recompute the body walks.
pub fn discover_stack_fn_effects(
    llbc: &Llbc,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<(String, Vec<u8>)>,
    Vec<String>,
    Vec<String>,
) {
    discover_stack_effects(llbc)
}

/// Least consistent assignment of per-body [`stack_walk`] over this artefact.
///
/// Seed every local body against all-`None` (own-body effects only). Then
/// Gauss-Seidel: recompute a body from scratch against the live classes
/// (previous crates through [`callee_effect_of`]). Jacobi rounds oscillate
/// on a caller that Observes only because a `returns_index` callee is not
/// yet in the previous map; processing Observes first lets that caller
/// drop once the callee's summary is live. A revisit cap freezes still-
/// changing bodies as Observes.
fn discover_stack_effects(
    llbc: &Llbc,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<(String, Vec<u8>)>,
    Vec<String>,
    Vec<String>,
) {
    let fds: Vec<&FunDecl> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .collect();
    let names: Vec<String> = fds.iter().map(|fd| fd.item_meta.name_path()).collect();
    let local: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
    let mut callers: HashMap<String, Vec<usize>> = HashMap::new();
    let mut class: HashMap<String, CalleeEffect> = HashMap::new();
    let mut last_walk: Vec<Option<(Vec<(String, CalleeEffect)>, CalleeEffect)>> =
        vec![None; fds.len()];
    let none_effect = |path: &str| {
        if local.contains(path) {
            CalleeEffect::none()
        } else {
            callee_effect_of(llbc, path)
        }
    };
    let mut queued = vec![false; fds.len()];
    let mut observes_ids: Vec<usize> = Vec::new();
    let mut other_ids: Vec<usize> = Vec::new();
    for (i, fd) in fds.iter().enumerate() {
        let Some(body) = fd.unstructured_shared() else {
            continue;
        };
        let forward_blocks = super::forward_reachable_mask(llbc, &body);
        for (bb_idx, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
                && let Some(path) = callee_path(call, llbc)
            {
                callers.entry(path).or_default().push(i);
            }
        }
        let queried = std::cell::RefCell::new(Vec::new());
        let effect = |path: &str| {
            let answer = none_effect(path);
            if local.contains(path) {
                queried
                    .borrow_mut()
                    .push((path.to_string(), answer.clone()));
            }
            answer
        };
        let new = stack_walk(&body, llbc, &effect).effect();
        last_walk[i] = Some((queried.into_inner(), new.clone()));
        if new.observes {
            observes_ids.push(i);
        } else {
            other_ids.push(i);
        }
        if !new.is_none() {
            class.insert(names[i].clone(), new);
        }
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in observes_ids.into_iter().chain(other_ids) {
        queued[i] = true;
        queue.push_back(i);
    }
    let mut visits = vec![0u8; fds.len()];
    const VISIT_CAP: u8 = 8;
    while let Some(i) = queue.pop_front() {
        queued[i] = false;
        if visits[i] >= VISIT_CAP {
            let old = class
                .get(&names[i])
                .cloned()
                .unwrap_or_else(CalleeEffect::none);
            class.insert(names[i].clone(), CalleeEffect::worst());
            if old != CalleeEffect::worst() {
                for &caller in callers.get(&names[i]).into_iter().flatten() {
                    if queued[caller] {
                        continue;
                    }
                    queued[caller] = true;
                    queue.push_back(caller);
                }
            }
            continue;
        }
        visits[i] = visits[i].saturating_add(1);
        let new = if let Some((queried, prev)) = &last_walk[i]
            && queried.iter().all(|(path, answer)| {
                class.get(path).cloned().unwrap_or_else(CalleeEffect::none) == *answer
            }) {
            prev.clone()
        } else {
            let Some(body) = fds[i].unstructured_shared() else {
                continue;
            };
            let queried = std::cell::RefCell::new(Vec::new());
            let effect = |path: &str| {
                if local.contains(path) {
                    let answer = class.get(path).cloned().unwrap_or_else(CalleeEffect::none);
                    queried
                        .borrow_mut()
                        .push((path.to_string(), answer.clone()));
                    answer
                } else {
                    callee_effect_of(llbc, path)
                }
            };
            let new = stack_walk(&body, llbc, &effect).effect();
            last_walk[i] = Some((queried.into_inner(), new.clone()));
            new
        };
        let old = class
            .get(&names[i])
            .cloned()
            .unwrap_or_else(CalleeEffect::none);
        if new == old {
            continue;
        }
        if new.is_none() {
            class.remove(&names[i]);
        } else {
            class.insert(names[i].clone(), new.clone());
        }
        for &caller in callers.get(&names[i]).into_iter().flatten() {
            if queued[caller] {
                continue;
            }
            queued[caller] = true;
            if new.observes {
                queue.push_back(caller);
            } else {
                // Callee dropped Observes or gained a summary: re-walk
                // Observes callers first so they can drop.
                queue.push_front(caller);
            }
        }
    }
    let mut sensitive: Vec<String> = Vec::new();
    let mut leaves: Vec<String> = Vec::new();
    let mut param_slots: Vec<(String, Vec<u8>)> = Vec::new();
    let mut returns_index: Vec<String> = Vec::new();
    let mut observes: Vec<String> = Vec::new();
    for (name, effect) in class {
        if effect.is_none() {
            continue;
        }
        sensitive.push(name.clone());
        if effect.leaves_above {
            leaves.push(name.clone());
        }
        if effect.observes {
            observes.push(name.clone());
        }
        if !effect.param_slots.is_empty() {
            param_slots.push((name.clone(), effect.param_slots));
        }
        if effect.returns_index {
            returns_index.push(name);
        }
    }
    sensitive.sort();
    observes.sort();
    leaves.sort();
    param_slots.sort_by(|a, b| a.0.cmp(&b.0));
    returns_index.sort();
    (sensitive, leaves, param_slots, returns_index, observes)
}

/// Local bodies proven depth-neutral: every path restores the entry
/// depth, no caller-owned slot is read, Observes callees are refused,
/// and LeavesAbove callees are allowed (fixpoint, pessimistic start).
/// Harvested in link order like [`discover_stack_sensitive_fns`].
/// A body whose last failed check queried no newly proven path is not re-checked.
pub fn discover_depth_neutral_fns(llbc: &Llbc) -> Vec<String> {
    let fds: Vec<&FunDecl> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .collect();
    // Proven name -> how many names were proven up to and including it.
    let mut proven: HashMap<String, usize> = HashMap::new();
    // Per body: the proven count at its last failed check and the callee paths that check
    // queried. Only a newly proven queried path can change the answer.
    let mut last_check: Vec<Option<(usize, Vec<String>)>> = vec![None; fds.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (i, fd) in fds.iter().enumerate() {
            let name = fd.item_meta.name_path();
            if proven.contains_key(&name) {
                continue;
            }
            if let Some((at, queried)) = &last_check[i]
                && queried
                    .iter()
                    .all(|path| proven.get(path).is_none_or(|&n| n <= *at))
            {
                continue;
            }
            let Some(body) = fd.unstructured_shared() else {
                last_check[i] = Some((proven.len(), Vec::new()));
                continue;
            };
            let queried = std::cell::RefCell::new(Vec::new());
            let effect = |path: &str| {
                queried.borrow_mut().push(path.to_string());
                if proven.contains_key(path) {
                    CalleeEffect::none()
                } else {
                    callee_effect_of(llbc, path)
                }
            };
            if body_is_depth_neutral(&body, llbc, &effect) {
                let n = proven.len() + 1;
                proven.insert(name, n);
                changed = true;
            } else {
                last_check[i] = Some((proven.len(), queried.into_inner()));
            }
        }
    }
    let mut out: Vec<String> = proven.into_keys().collect();
    out.sort();
    out
}

/// Classify `llbc`'s own functions and mark its set complete, for a caller
/// that lowers one artefact on its own.  Callees from other artefacts count
/// as insensitive here; the linked translation registers them first.
pub fn ensure_stack_sensitive_fns(llbc: &Llbc) {
    if llbc.stack_sensitive_fns_complete() {
        return;
    }
    let (found, leaves, params, ret_idx, observes) = discover_stack_effects(llbc);
    llbc.register_stack_sensitive_fns(found);
    llbc.register_stack_observes_fns(observes);
    llbc.register_stack_leaves_above_fns(leaves);
    llbc.register_stack_param_slots_fns(params);
    llbc.register_stack_returns_index_fns(ret_idx);
    llbc.register_stack_depth_neutral_fns(discover_depth_neutral_fns(llbc));
    llbc.mark_stack_sensitive_fns_complete();
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::Llbc;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    fn span() -> Value {
        json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    }

    fn fixture_llbc() -> Llbc {
        let ident = |s: &str| json!({"Ident": [s, 0]});
        let push_roots = json!({
            "def_id": 1,
            "item_meta": {
                "name": [
                    ident("pyre_object"),
                    ident("gc_roots"),
                    ident("push_roots")
                ],
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Deduplicated": 0}
            },
            "body": "Opaque"
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [null, push_roots]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn analyze_open_then(sink: Value) -> Result<Option<Plan>, Refusal> {
        let ty = json!({"Deduplicated": 0});
        let place = |i: u64| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64| json!({"index": i, "name": null, "span": span(), "ty": ty});
        let bb = |kind: Value| {
            json!({
                "statements": [],
                "terminator": {"span": span(), "kind": kind},
                "kind": "Regular"
            })
        };
        let open = json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": 1}, "generics": {}}},
                "args": [],
                "dest": place(1)
            },
            "target": 1,
            "on_unwind": 2
        }});
        let raw = json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1)]},
            "body": [
                bb(open),
                bb(sink),
                bb(json!("UnwindResume"))
            ]
        });
        let body: Unstructured = serde_json::from_value(raw.clone()).expect("fixture body");
        analyze(&body, &fixture_llbc())
    }

    fn assert_erase_accepted(sink: Value, label: &str) {
        match analyze_open_then(sink) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("{label}: expected accepted erase, got no bracket"),
            Err(reason) => panic!("{label}: expected accepted erase, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_undefined_behavior_sink_like_abort() {
        assert_erase_accepted(json!({"Abort": "UndefinedBehavior"}), "Abort");
        assert_erase_accepted(json!("UndefinedBehavior"), "UndefinedBehavior");
    }

    #[test]
    fn analyze_accepts_unwind_terminate_sink_like_abort() {
        assert_erase_accepted(json!({"Abort": "UnwindTerminate"}), "Abort");
        assert_erase_accepted(json!("UnwindTerminate"), "UnwindTerminate");
    }

    fn opaque_fun(def_id: u64, name: &[&str], body: Value) -> Value {
        let ident = |s: &str| json!({"Ident": [s, 0]});
        json!({
            "def_id": def_id,
            "item_meta": {
                "name": name.iter().map(|s| ident(s)).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Deduplicated": 0}
            },
            "body": body
        })
    }

    /// A `pin_roots` of a same-block array, with a cleanup block in front of
    /// that array block. The body keeps every extracted block at its original
    /// index; erase still accepts the publish.
    #[test]
    fn erase_accepts_publish_when_cleanup_blocks_precede_the_array() {
        let ty = json!({"Deduplicated": 0});
        let place = |i: u64| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64| json!({"index": i, "name": null, "span": span(), "ty": ty});
        let stmt = |kind: Value| json!({"span": span(), "kind": kind});
        let bb = |statements: Vec<Value>, kind: Value, is_cleanup: bool| {
            let unwind = if is_cleanup { "Cleanup" } else { "Regular" };
            json!({
                "statements": statements,
                "terminator": {"span": span(), "kind": kind},
                "kind": unwind
            })
        };
        let call = |fun: u64, args: Vec<Value>, dest: u64, target: u64, on_unwind: u64| {
            json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": args,
                    "dest": place(dest)
                },
                "target": target,
                "on_unwind": on_unwind
            }})
        };
        let array = json!({"Assign": [
            place(2),
            {"Aggregate": [
                {"Array": [ty, {"Deduplicated": 1}, null]},
                [
                    {"Move": place(3)},
                    {"Move": place(4)}
                ]
            ]}
        ]});
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": (0..6).map(local).collect::<Vec<_>>()
            },
            "body": [
                bb(vec![], call(1, vec![], 1, 2, 1), false),
                bb(vec![], json!("UnwindResume"), true),
                bb(
                    vec![stmt(array)],
                    call(2, vec![json!({"Move": place(2)})], 5, 3, 1),
                    false
                ),
                bb(vec![], json!("UndefinedBehavior"), false)
            ]
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque")),
                    opaque_fun(
                        3,
                        &["pyre_object", "celldict", "write_cell"],
                        json!({ "Unstructured": unstructured })
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let fd = llbc.fn_by_id(3).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        assert_eq!(
            body.body.len(),
            4,
            "extracted blocks stay at their original indices"
        );
        assert!(
            body.body[1].is_cleanup
                && matches!(body.body[1].term_ref(&llbc), Ok(TermKind::UnwindResume)),
            "block 1 is the cleanup UnwindResume"
        );
        match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// rustc spells `drop(g)` as `_t = Move(g); core::mem::drop(Move(_t))`.
    /// Census calls `erase_shadow_stack` on the raw body, so that Call is
    /// Close of the watched Guard (`TermKind::Drop` of a Guard already is).
    #[test]
    fn erase_accepts_mem_drop_of_a_moved_guard() {
        let ty = json!({"Deduplicated": 0});
        let place = |i: u64| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64| json!({"index": i, "name": null, "span": span(), "ty": ty});
        let stmt = |kind: Value| json!({"span": span(), "kind": kind});
        let bb = |statements: Vec<Value>, kind: Value| {
            json!({
                "statements": statements,
                "terminator": {"span": span(), "kind": kind},
                "kind": "Regular"
            })
        };
        let call = |fun: u64, args: Vec<Value>, dest: u64, target: u64, on_unwind: u64| {
            json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": args,
                    "dest": place(dest)
                },
                "target": target,
                "on_unwind": on_unwind
            }})
        };
        let move_guard = json!({"Assign": [
            place(2),
            {"Use": [{"Move": place(1)}, "No"]}
        ]});
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": (0..4).map(local).collect::<Vec<_>>()
            },
            "body": [
                bb(vec![], call(1, vec![], 1, 1, 3)),
                bb(
                    vec![stmt(move_guard)],
                    call(2, vec![json!({"Move": place(2)})], 3, 2, 3)
                ),
                bb(vec![], json!("Return")),
                bb(vec![], json!("UnwindResume"))
            ]
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["core", "mem", "drop"], json!("Opaque")),
                    opaque_fun(
                        3,
                        &["pyre_object", "fixture", "drop_moved_guard"],
                        json!({ "Unstructured": unstructured })
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let fd = llbc.fn_by_id(3).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    fn stack_ops_llbc() -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn ty() -> Value {
        json!({"Deduplicated": 0})
    }

    fn place(i: u64) -> Value {
        json!({"kind": {"Local": i}, "ty": ty()})
    }

    fn local_decl(i: u64) -> Value {
        json!({"index": i, "name": null, "span": span(), "ty": ty()})
    }

    fn term_bb(kind: Value) -> Value {
        json!({
            "statements": [],
            "terminator": {"span": span(), "kind": kind},
            "kind": "Regular"
        })
    }

    fn bb(kind: Value) -> Value {
        term_bb(kind)
    }

    fn bb_cleanup(kind: Value) -> Value {
        json!({
            "statements": [],
            "terminator": {"span": span(), "kind": kind},
            "kind": "Cleanup"
        })
    }

    fn call_fun(fun: u64, args: Vec<Value>, dest: u64, target: u64) -> Value {
        json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    }

    fn drop_local(local: u64, target: u64) -> Value {
        json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    }

    fn body_of(n_locals: u64, blocks: Vec<Value>) -> Unstructured {
        body_of_args(0, n_locals, blocks)
    }

    fn body_of_args(arg_count: u64, n_locals: u64, blocks: Vec<Value>) -> Unstructured {
        let raw = json!({
            "span": span(),
            "locals": {
                "arg_count": arg_count,
                "locals": (0..n_locals).map(local_decl).collect::<Vec<_>>()
            },
            "body": blocks
        });
        serde_json::from_value(raw).expect("fixture body")
    }

    fn is_neutral(body: &Unstructured, llbc: &Llbc) -> bool {
        body_is_depth_neutral(body, llbc, &|_| CalleeEffect::none())
    }

    #[test]
    fn depth_neutral_refuses_scope_leaked_on_one_branch() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            3,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [3, 4]}
                }})),
                term_bb(drop_local(1, 5)),
                term_bb(json!({"Goto": {"target": 5}})),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            !is_neutral(&body, &llbc),
            "a branch that skips Close must not be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_refuses_balanced_scope_plus_caller_slot_read() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(3, vec![], 2, 2)),
                // Local 0 is not a Len/Base index: untracked, so index-not-own.
                term_bb(call_fun(2, vec![json!({"Copy": place(0)})], 3, 3)),
                term_bb(drop_local(1, 4)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            !is_neutral(&body, &llbc),
            "Get of an untracked index must not be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_refuses_open_and_close_in_unreachable_blocks() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            2,
            vec![
                term_bb(json!("Return")),
                term_bb(call_fun(1, vec![], 1, 2)),
                term_bb(drop_local(1, 3)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            !is_neutral(&body, &llbc),
            "open/close only in unreachable blocks must not be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_accepts_eval_slice_index_style_balanced_body() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            3,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(drop_local(1, 3)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            is_neutral(&body, &llbc),
            "open, pin, close on every path must be depth-neutral"
        );
    }

    fn stack_ops_llbc_with_callee() -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["other", "unproven_callee"], json!("Opaque")),
                    opaque_fun(6, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque")),
                    opaque_fun(7, &["other", "leaves_above_callee"], json!("Opaque")),
                    opaque_fun(8, &["other", "reload_slot"], json!("Opaque")),
                    opaque_fun(9, &["other", "pin_helper"], json!("Opaque")),
                    opaque_fun(10, &["other", "storage_helper"], json!("Opaque")),
                    opaque_fun(11, &["other", "observes_and_leaves"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.register_stack_sensitive_fns([
            "other::unproven_callee".into(),
            "other::leaves_above_callee".into(),
            "other::reload_slot".into(),
            "other::pin_helper".into(),
            "other::storage_helper".into(),
            "other::observes_and_leaves".into(),
        ]);
        llbc.register_stack_observes_fns([
            "other::unproven_callee".into(),
            "other::observes_and_leaves".into(),
        ]);
        llbc.register_stack_leaves_above_fns([
            "other::leaves_above_callee".into(),
            "other::pin_helper".into(),
            "other::observes_and_leaves".into(),
        ]);
        llbc.register_stack_param_slots_fns([("other::reload_slot".into(), vec![0])]);
        llbc.register_stack_returns_index_fns([
            "other::pin_helper".into(),
            "other::storage_helper".into(),
        ]);
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn bb_stmts(statements: Vec<Value>, kind: Value) -> Value {
        json!({
            "statements": statements,
            "terminator": {"span": span(), "kind": kind},
            "kind": "Regular"
        })
    }

    fn assign_local(dest: u64, rvalue: Value) -> Value {
        json!({
            "kind": {"Assign": [place(dest), rvalue]},
            "comments_before": [],
            "span": span()
        })
    }

    fn usize_lit(k: u64) -> Value {
        json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]})
    }

    /// A cleanup block that reassigns a local is not a second definition.
    #[test]
    fn local_def_counts_skips_cleanup_reassign() {
        let llbc = fixture_llbc();
        let use_copy0 = json!({"Use": [{"Copy": place(0)}, "Yes"]});
        let body = body_of(
            2,
            vec![
                bb_stmts(vec![assign_local(1, use_copy0.clone())], json!("Return")),
                json!({
                    "statements": [assign_local(1, use_copy0)],
                    "terminator": {"span": span(), "kind": "UnwindResume"},
                    "kind": "Cleanup"
                }),
            ],
        );
        let forward_blocks = super::super::forward_reachable_mask(&llbc, &body);
        let counts = local_def_counts(&body, &llbc, &forward_blocks);
        assert_eq!(counts.get(1).copied(), Some(1));
    }

    /// `zip_two_tuple_next` / `tuple_iter_descr_next`: pin, then name the
    /// slot as `shadow_stack_len() - 1`, then get it.
    fn len_minus_one_body(sub_op: Value) -> Unstructured {
        let copy = |i| json!({"Copy": place(i)});
        body_of(
            6,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": [sub_op, copy(3), usize_lit(1)]}),
                    )],
                    call_fun(2, vec![copy(4)], 5, 4),
                ),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        )
    }

    /// `zip_two_tuple_next`: an inner RootScope opens on one arm and closes
    /// before the join with the arm that never opened it. Closing must drop
    /// that set's pin-count key so both sides arrive with the outer map.
    #[test]
    fn analyze_accepts_nested_scope_closed_before_merge() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 3, 2)),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [3, 5]}
                }})),
                term_bb(call_fun(1, vec![], 2, 4)),
                term_bb(drop_local(2, 5)),
                term_bb(drop_local(1, 6)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// `match nargs` pins a different count per arm, then joins before
    /// Close (`funccall_flat_from_args`). The join is `AtMerge`, not a
    /// refusal.
    #[test]
    fn analyze_accepts_merge_of_different_const_pin_counts() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            5,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [3, 4]}
                }})),
                term_bb(json!({"Goto": {"target": 5}})),
                term_bb(call_fun(4, vec![copy(0)], 3, 5)),
                term_bb(drop_local(1, 6)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected erasure of nargs-pin merge"),
            Err(reason) => panic!("expected erasure of nargs-pin merge, got {reason}"),
        }
    }

    /// `pin_roots(&*args)` of an incoming `&[T]` (`cdataobj.py descr_call`,
    /// `pyframe.py Frame.__init__` livevars). `get(base)` is `getarrayitem`
    /// of that list; the publish dest is 0.
    #[test]
    fn analyze_accepts_publish_of_incoming_slice() {
        let llbc = stack_ops_llbc();
        let mv = |i| json!({"Move": place(i)});
        let copy = |i| json!({"Copy": place(i)});
        let deref_ref = json!({"Ref": {
            "place": {
                "kind": {"Projection": [place(1), "Deref"]},
                "ty": ty()
            },
            "kind": "Shared",
            "ptr_metadata": null
        }});
        let body = body_of_args(
            1,
            6,
            vec![
                term_bb(call_fun(1, vec![], 2, 1)),
                bb_stmts(
                    vec![assign_local(3, deref_ref)],
                    call_fun(5, vec![mv(3)], 4, 2),
                ),
                term_bb(call_fun(2, vec![copy(4)], 5, 3)),
                term_bb(drop_local(2, 4)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected erasure of incoming-slice publish"),
            Err(reason) => panic!("expected erasure of incoming-slice publish, got {reason}"),
        }
    }

    /// After that merge, `get` of an index below the minimum predecessor
    /// depth reads the absolute slot. A pin at the merge writes a
    /// relative slot; `len - 1` names it for the subsequent `get`.
    #[test]
    fn analyze_accepts_get_below_merge_min_and_pin_after_merge() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            10,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                // Index(Const(0)) at the Open depth, below every arm's pins.
                term_bb(call_fun(3, vec![], 2, 2)),
                term_bb(call_fun(4, vec![copy(0)], 3, 3)),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [4, 5]}
                }})),
                term_bb(json!({"Goto": {"target": 6}})),
                term_bb(call_fun(4, vec![copy(0)], 4, 6)),
                term_bb(call_fun(2, vec![copy(2)], 5, 7)),
                term_bb(call_fun(4, vec![copy(0)], 6, 8)),
                term_bb(call_fun(3, vec![], 7, 9)),
                bb_stmts(
                    vec![assign_local(
                        8,
                        json!({"BinaryOp": [json!("Sub"), copy(7), usize_lit(1)]}),
                    )],
                    call_fun(2, vec![copy(8)], 9, 10),
                ),
                term_bb(drop_local(1, 11)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected erasure of get-below-min plus post-merge pin"),
            Err(reason) => {
                panic!("expected erasure of get-below-min plus post-merge pin, got {reason}")
            }
        }
    }

    /// An `AtMerge` successor meeting a `Const` predecessor at a later
    /// join is still `depth-disagrees-at-merge`.
    #[test]
    fn analyze_refuses_atmerge_meeting_const() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            5,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [2, 6]}
                }})),
                term_bb(call_fun(4, vec![copy(0)], 2, 3)),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(4)},
                    "targets": {"If": [4, 5]}
                }})),
                term_bb(call_fun(4, vec![copy(0)], 3, 5)),
                term_bb(json!({"Goto": {"target": 7}})),
                term_bb(call_fun(4, vec![copy(0)], 4, 7)),
                term_bb(drop_local(1, 8)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("depth-disagrees-at-merge") => {}
            Ok(Some(_)) => panic!("expected depth-disagrees-at-merge, got accepted"),
            Ok(None) => panic!("expected depth-disagrees-at-merge, got no bracket"),
            Err(reason) => panic!("expected depth-disagrees-at-merge, got {reason}"),
        }
    }

    /// Absolute `get(Const(j))` after a merge with `j >= max` predecessor
    /// depth names a slot no arm defined.
    #[test]
    fn analyze_refuses_read_above_merge_max() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            7,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(3, vec![], 2, 2)),
                bb_stmts(
                    vec![assign_local(
                        3,
                        json!({"BinaryOp": [json!("Add"), copy(2), usize_lit(2)]}),
                    )],
                    json!({"Switch": {
                        "discr": {"Copy": place(0)},
                        "targets": {"If": [3, 4]}
                    }}),
                ),
                term_bb(call_fun(4, vec![copy(0)], 4, 5)),
                term_bb(call_fun(4, vec![copy(0)], 4, 6)),
                term_bb(json!({"Goto": {"target": 7}})),
                term_bb(call_fun(4, vec![copy(0)], 5, 7)),
                term_bb(call_fun(2, vec![copy(3)], 6, 8)),
                term_bb(drop_local(1, 9)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("read-not-defined-on-all-predecessors") => {}
            Ok(Some(_)) => panic!("expected read-not-defined-on-all-predecessors, got accepted"),
            Ok(None) => panic!("expected read-not-defined-on-all-predecessors, got no bracket"),
            Err(reason) => panic!("expected read-not-defined-on-all-predecessors, got {reason}"),
        }
    }

    /// One/two-pin diamond then `get(1)`: slot 1 exists on the long arm
    /// only. `ll_getitem_nonneg` would raise on the short arm; erasure
    /// refuses rather than invent a seed (`read-not-defined-on-all-predecessors`).
    #[test]
    fn analyze_refuses_get_of_slot_absent_on_one_predecessor() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            7,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(3, vec![], 2, 2)),
                bb_stmts(
                    vec![assign_local(
                        3,
                        json!({"BinaryOp": [json!("Add"), copy(2), usize_lit(1)]}),
                    )],
                    json!({"Switch": {
                        "discr": {"Copy": place(0)},
                        "targets": {"If": [3, 4]}
                    }}),
                ),
                term_bb(call_fun(4, vec![copy(0)], 4, 5)),
                term_bb(call_fun(4, vec![copy(0)], 4, 6)),
                term_bb(json!({"Goto": {"target": 7}})),
                term_bb(call_fun(4, vec![copy(0)], 5, 7)),
                term_bb(call_fun(2, vec![copy(3)], 6, 8)),
                term_bb(drop_local(1, 9)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("read-not-defined-on-all-predecessors") => {}
            Ok(Some(_)) => panic!("expected read-not-defined-on-all-predecessors, got accepted"),
            Ok(None) => panic!("expected read-not-defined-on-all-predecessors, got no bracket"),
            Err(reason) => panic!("expected read-not-defined-on-all-predecessors, got {reason}"),
        }
    }

    fn switch_int(discr: u64, arms: &[(u64, u64)], default: u64) -> Value {
        json!({"Switch": {
            "discr": {"Copy": place(discr)},
            "targets": {"SwitchInt": [
                null,
                arms.iter().map(|(v, t)| json!([
                    {"Scalar": {"Unsigned": ["Usize", v.to_string()]}},
                    t
                ])).collect::<Vec<_>>(),
                default
            ]}
        }})
    }

    fn json_has_local(v: &Value, local: usize) -> bool {
        match v {
            Value::Object(m) => {
                m.get("Local").and_then(Value::as_u64) == Some(local as u64)
                    || m.values().any(|c| json_has_local(c, local))
            }
            Value::Array(a) => a.iter().any(|c| json_has_local(c, local)),
            _ => false,
        }
    }

    fn rewrite_mentions_slot(plan: &Plan, bb: usize, slot: Slot, n_locals: usize) -> bool {
        let Some(&idx) = plan.slot_intern.get(&slot) else {
            return false;
        };
        let local = n_locals + idx;
        let Some(term) = plan.terms.get(&bb) else {
            return false;
        };
        term.stmts.iter().any(|stmt| json_has_local(stmt, local))
    }

    /// Two switches on the same unmodified local: pin count per arm of
    /// the first matches get count per arm of the second. Each get
    /// resolves to the slot that arm pinned (`constfold.constant_diffuse`
    /// after `exitswitch`; `rewire_link_for_known_exitswitch`).
    #[test]
    fn analyze_erases_repeated_switch_on_same_scrutinee() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let abort = json!({"Abort": "UndefinedBehavior"});
        let body = body_of_args(
            1,
            12,
            vec![
                term_bb(call_fun(1, vec![], 2, 1)),
                term_bb(call_fun(3, vec![], 3, 2)),
                term_bb(switch_int(1, &[(0, 3), (1, 4)], 13)),
                term_bb(call_fun(4, vec![copy(0)], 4, 6)),
                term_bb(call_fun(4, vec![copy(0)], 5, 5)),
                term_bb(call_fun(4, vec![copy(1)], 6, 6)),
                term_bb(json!({"Goto": {"target": 7}})),
                term_bb(switch_int(1, &[(0, 8), (1, 9)], 13)),
                term_bb(call_fun(2, vec![copy(3)], 8, 11)),
                term_bb(call_fun(2, vec![copy(3)], 9, 10)),
                bb_stmts(
                    vec![assign_local(
                        9,
                        json!({"BinaryOp": [json!("Add"), copy(3), usize_lit(1)]}),
                    )],
                    call_fun(2, vec![copy(9)], 10, 12),
                ),
                term_bb(drop_local(2, 14)),
                term_bb(drop_local(2, 14)),
                term_bb(abort),
                term_bb(json!("Return")),
            ],
        );
        let n_locals = body.locals.locals.len();
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected erasure of correlated pin/get switches"),
            Err(reason) => panic!("expected erasure of correlated pin/get switches, got {reason}"),
        };
        assert!(
            rewrite_mentions_slot(&plan, 8, Slot::Abs(0), n_locals),
            "arm 0 get must copy the slot that arm pinned"
        );
        assert!(
            rewrite_mentions_slot(&plan, 9, Slot::Abs(0), n_locals),
            "arm 1 first get must copy slot 0"
        );
        assert!(
            rewrite_mentions_slot(&plan, 10, Slot::Abs(1), n_locals),
            "arm 1 second get must copy the second pin of that arm"
        );
    }

    /// Same shape as the correlated switches, but the scrutinee is
    /// assigned between them, so the fact is killed and the suffix get
    /// is again `read-not-defined-on-all-predecessors`.
    #[test]
    fn analyze_refuses_repeated_switch_after_scrutinee_reassign() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let abort = json!({"Abort": "UndefinedBehavior"});
        let body = body_of_args(
            1,
            12,
            vec![
                term_bb(call_fun(1, vec![], 2, 1)),
                term_bb(call_fun(3, vec![], 3, 2)),
                term_bb(switch_int(1, &[(0, 3), (1, 4)], 14)),
                term_bb(call_fun(4, vec![copy(0)], 4, 6)),
                term_bb(call_fun(4, vec![copy(0)], 5, 5)),
                term_bb(call_fun(4, vec![copy(1)], 6, 6)),
                term_bb(json!({"Goto": {"target": 7}})),
                bb_stmts(
                    vec![assign_local(1, json!({"Use": [usize_lit(99), "Yes"]}))],
                    json!({"Goto": {"target": 8}}),
                ),
                term_bb(switch_int(1, &[(0, 9), (1, 10)], 14)),
                term_bb(call_fun(2, vec![copy(3)], 8, 12)),
                term_bb(call_fun(2, vec![copy(3)], 9, 11)),
                bb_stmts(
                    vec![assign_local(
                        9,
                        json!({"BinaryOp": [json!("Add"), copy(3), usize_lit(1)]}),
                    )],
                    call_fun(2, vec![copy(9)], 10, 13),
                ),
                term_bb(drop_local(2, 15)),
                term_bb(drop_local(2, 15)),
                term_bb(abort),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("read-not-defined-on-all-predecessors") => {}
            Ok(Some(_)) => panic!("expected refusal after scrutinee reassign, got accepted"),
            Ok(None) => panic!("expected refusal after scrutinee reassign, got no bracket"),
            Err(reason) => panic!("expected read-not-defined-on-all-predecessors, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_len_minus_one_after_pin() {
        let llbc = stack_ops_llbc();
        let body = len_minus_one_body(json!("Sub"));
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
        assert!(
            is_neutral(&body, &llbc),
            "len-1 of a just-pinned slot must be depth-neutral"
        );
    }

    /// `Drop` of the guard lives in an `is_cleanup` block. `unstructured`
    /// keeps those blocks, so a later `UnwindTerminate` can be last while
    /// `UnwindResume` remains in the body. Return with published slots is
    /// accepted when a retained block is `UnwindResume`.
    #[test]
    fn analyze_accepts_return_with_slots_when_unwind_resume_is_not_last() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            3,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(json!("Return")),
                bb_cleanup(json!("UnwindResume")),
                bb_cleanup(json!("UnwindTerminate")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// `pin_self`: pin, `len - 1`, return that index. The return place is
    /// the Index special; erase would strip its assignment. Last block is
    /// `UnwindResume` so the cleanup-drop depth hack does not refuse first.
    #[test]
    fn analyze_refuses_returning_the_len_minus_one_slot_index() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            4,
            vec![
                term_bb(call_fun(4, vec![copy(1)], 2, 1)),
                term_bb(call_fun(3, vec![], 3, 2)),
                bb_stmts(
                    vec![assign_local(
                        0,
                        json!({"BinaryOp": [json!("Sub"), copy(3), usize_lit(1)]}),
                    )],
                    json!("Return"),
                ),
                term_bb(json!("UnwindResume")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("returns-slot-index") => {}
            Ok(Some(_)) => panic!("expected returns-slot-index, got accepted"),
            Ok(None) => panic!("expected returns-slot-index, got no bracket"),
            Err(reason) => panic!("expected returns-slot-index, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_len_minus_one_wrap_after_pin() {
        let llbc = stack_ops_llbc();
        let body = len_minus_one_body(json!({"Sub": "Wrap"}));
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_len_minus_one_subchecked_after_pin() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let field0 = json!({
            "kind": {"Projection": [place(4), {"Field": [null, 0]}]},
            "ty": ty()
        });
        let body = body_of(
            7,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![
                        assign_local(
                            4,
                            json!({"BinaryOp": ["SubChecked", copy(3), usize_lit(1)]}),
                        ),
                        assign_local(5, json!({"Use": [{"Move": field0}, "No"]})),
                    ],
                    call_fun(2, vec![copy(5)], 6, 4),
                ),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
        assert!(
            is_neutral(&body, &llbc),
            "SubChecked len-1 of a just-pinned slot must be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_refuses_unproven_stack_sensitive_callee() {
        let llbc = stack_ops_llbc_with_callee();
        let observes = |path: &str| {
            if path.ends_with("unproven_callee") {
                CalleeEffect::observes()
            } else {
                CalleeEffect::none()
            }
        };
        // After Close the depth is Known(0): every remaining slot is the
        // caller's, so an Observes callee is not neutral.
        let after_close = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(drop_local(1, 3)),
                term_bb(call_fun(5, vec![], 4, 4)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            !body_is_depth_neutral(&after_close, &llbc, &observes),
            "Observes callee at Known(0) must not be depth-neutral"
        );
        assert!(
            body_is_depth_neutral(&after_close, &llbc, &|_| CalleeEffect::none()),
            "the same body is depth-neutral once the callee is proven"
        );
        // Unproven callees inside our Open/Close are depth-neutral:
        // they see our pins, and Close rewinds the guard.
        let inside_scope = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(5, vec![], 3, 3)),
                term_bb(call_fun(5, vec![], 3, 4)),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&inside_scope, &llbc, &observes),
            "unproven callees inside our Open/Close are depth-neutral"
        );
    }

    #[test]
    fn open_pin_callee_close_observes_vs_leaves_above() {
        let llbc = stack_ops_llbc_with_callee();
        let effect = |path: &str| callee_effect_of(&llbc, path);
        let observes_inside = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(5, vec![], 3, 3)),
                term_bb(drop_local(1, 4)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&observes_inside, &llbc, &effect),
            "Open -> Pin -> Observes -> Close is depth-neutral"
        );
        match analyze(&observes_inside, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected erasure of Open/Pin/Observes/Close"),
            Err(reason) => panic!("expected erasure of Open/Pin/Observes/Close, got {reason}"),
        }
        let leaves_inside = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(7, vec![], 3, 3)),
                term_bb(drop_local(1, 4)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&leaves_inside, &llbc, &effect),
            "Open -> Pin -> LeavesAbove -> Close is depth-neutral"
        );
        match analyze(&leaves_inside, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        // A callee that reads below entry and also leaves slots: Close
        // still rewinds it, so the body stays neutral, but the erased
        // body would have no Close left.
        let observes_and_leaves_inside = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(11, vec![], 3, 3)),
                term_bb(drop_local(1, 4)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&observes_and_leaves_inside, &llbc, &effect),
            "Open -> Pin -> Observes+LeavesAbove -> Close is depth-neutral"
        );
        match analyze(&observes_and_leaves_inside, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        let only_observes_and_leaves = body_of(
            2,
            vec![
                term_bb(call_fun(11, vec![], 1, 1)),
                term_bb(json!("Return")),
            ],
        );
        assert_eq!(
            stack_walk(&only_observes_and_leaves, &llbc, &effect).effect(),
            CalleeEffect::worst(),
            "a reader's leftover reaches its caller's summary"
        );
        let only_leaves = body_of(
            2,
            vec![term_bb(call_fun(7, vec![], 1, 1)), term_bb(json!("Return"))],
        );
        let walk = stack_walk(&only_leaves, &llbc, &effect);
        assert_eq!(
            walk.effect(),
            CalleeEffect::leaves_above(),
            "a body that only calls LeavesAbove and returns is LeavesAbove"
        );
    }

    #[test]
    fn param_slots_callee_own_index_is_neutral_untracked_is_observes() {
        let llbc = stack_ops_llbc_with_callee();
        let effect = |path: &str| callee_effect_of(&llbc, path);
        let callee = body_of_args(
            1,
            3,
            vec![
                term_bb(call_fun(2, vec![json!({"Copy": place(1)})], 2, 1)),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&callee, &llbc, &effect);
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of a parameter index is ParamSlots, not Observes: {:?}",
            walk.effect()
        );
        assert!(
            !body_is_depth_neutral(&callee, &llbc, &effect),
            "a param_slots body is not depth-neutral"
        );
        let own_index = body_of(
            5,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(3, vec![], 3, 3)),
                term_bb(call_fun(8, vec![json!({"Copy": place(3)})], 4, 4)),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&own_index, &llbc, &effect),
            "caller passing its own Len index to a param_slots callee is depth-neutral"
        );
        match analyze(&own_index, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        let untracked = body_of(
            4,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(8, vec![json!({"Copy": place(0)})], 3, 3)),
                term_bb(drop_local(1, 4)),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&untracked, &llbc, &effect);
        assert!(
            walk.effect().observes,
            "caller passing an untracked index to a param_slots callee is Observes: why={:?}",
            walk.why
        );
        match analyze(&untracked, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        // B.1: param p + any unsigned operand stays param p.
        let param_plus = body_of_args(
            1,
            4,
            vec![
                bb_stmts(
                    vec![assign_local(
                        2,
                        json!({"BinaryOp": ["Add", json!({"Copy": place(1)}), json!({"Copy": place(3)})]}),
                    )],
                    call_fun(2, vec![json!({"Copy": place(2)})], 3, 1),
                ),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&param_plus, &llbc, &effect);
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of param+unsigned is ParamSlots, not Observes: {:?}",
            walk.effect()
        );
        // B.1: Known(k) + unsigned local is AtLeast(k), still own.
        let known_plus = body_of(
            6,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                term_bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": ["Add", json!({"Copy": place(3)}), json!({"Copy": place(5)})]}),
                    )],
                    call_fun(2, vec![json!({"Copy": place(4)})], 5, 4),
                ),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&known_plus, &llbc, &effect),
            "Get of Len+unsigned must be own (AtLeast of the Len bound)"
        );
        // B.2: pin-style helper returns the entry-depth index; caller Get of dest is own.
        let pin_then_get = body_of(
            5,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(9, vec![], 3, 2)),
                term_bb(call_fun(2, vec![json!({"Copy": place(3)})], 4, 3)),
                term_bb(drop_local(1, 4)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&pin_then_get, &llbc, &effect),
            "Get of a returns_index dest (depth at the call) must be own"
        );
    }

    #[test]
    fn pin_style_body_returns_own_index_at_entry_depth() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        // Len into the return place, then Pin: every Return yields Known(0).
        let pin_style = body_of(
            3,
            vec![
                term_bb(call_fun(3, vec![], 0, 1)),
                term_bb(call_fun(4, vec![copy(1)], 2, 2)),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&pin_style, &llbc, &|_| CalleeEffect::none());
        assert!(
            walk.returns_index && walk.effect().leaves_above && !walk.effect().observes,
            "Len then Pin then Return of the Len dest is LeavesAbove+returns_index: {:?}",
            walk.effect()
        );
        let misses = body_of(
            3,
            vec![
                term_bb(call_fun(3, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(1)], 2, 2)),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&misses, &llbc, &|_| CalleeEffect::none());
        assert!(
            !walk.returns_index,
            "Return of unbound local 0 is not returns_index: {:?}",
            walk.effect()
        );
    }

    #[test]
    fn option_wrap_of_own_index_stays_own() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let some = json!({"Adt": [
            {"builtin": null, "generics": {
                "const_generics": [], "regions": [], "trait_refs": [], "types": [ty()]
            }, "id": 1},
            1,
            null
        ]});
        let payload = json!({
            "kind": {"Projection": [place(4), {"Field": [1, 0]}]},
            "ty": ty()
        });
        // Open, Pin, Len, Some(len), extract payload, Get: the Option
        // wrap is identity on the usize, so the Get is still own.
        let body = body_of(
            7,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![
                        assign_local(4, json!({"Aggregate": [some, [copy(3)]]})),
                        assign_local(5, json!({"Use": [{"Copy": payload}, "No"]})),
                    ],
                    call_fun(2, vec![copy(5)], 6, 4),
                ),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            is_neutral(&body, &llbc),
            "Get of Option::Some(Len) payload must be own"
        );
    }

    #[test]
    fn field_of_param_used_as_index_is_param_slots() {
        let llbc = stack_ops_llbc();
        let field0 = json!({
            "kind": {"Projection": [place(1), {"Field": [null, 0]}]},
            "ty": ty()
        });
        // Get of a usize field of argument 0: the caller supplied that
        // value, so the body is ParamSlots not Observes.
        let callee = body_of_args(
            1,
            4,
            vec![
                bb_stmts(
                    vec![assign_local(2, json!({"Use": [{"Copy": field0}, "No"]}))],
                    call_fun(2, vec![json!({"Copy": place(2)})], 3, 1),
                ),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&callee, &llbc, &|_| CalleeEffect::none());
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of param.field is ParamSlots: {:?}",
            walk.effect()
        );
        let deref_field = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(1), "Deref"]}, "ty": ty()},
                {"Field": [null, 0]}
            ]},
            "ty": ty()
        });
        let via_ref = body_of_args(
            1,
            4,
            vec![
                bb_stmts(
                    vec![assign_local(
                        2,
                        json!({"Use": [{"Copy": deref_field}, "No"]}),
                    )],
                    call_fun(2, vec![json!({"Copy": place(2)})], 3, 1),
                ),
                term_bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&via_ref, &llbc, &|_| CalleeEffect::none());
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of (*param).field is ParamSlots: {:?}",
            walk.effect()
        );
    }

    #[test]
    fn struct_return_with_index_field_is_returns_index() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let adt = json!({"Adt": [
            {"builtin": null, "generics": {
                "const_generics": [], "regions": [], "trait_refs": [], "types": []
            }, "id": 1},
            0,
            null
        ]});
        // Len into a field of the return struct: the body returns an
        // own-index carrier.
        let helper = body_of(
            3,
            vec![
                term_bb(call_fun(3, vec![], 1, 1)),
                bb_stmts(
                    vec![assign_local(
                        0,
                        json!({"Aggregate": [adt, [copy(1), copy(2)]]}),
                    )],
                    json!("Return"),
                ),
            ],
        );
        let walk = stack_walk(&helper, &llbc, &|_| CalleeEffect::none());
        assert!(
            walk.returns_index && !walk.effect().observes,
            "Return of a struct holding a Len dest is returns_index: {:?}",
            walk.effect()
        );
        let llbc = stack_ops_llbc_with_callee();
        let effect = |path: &str| callee_effect_of(&llbc, path);
        // Caller treats the dest as an own-index carrier and passes it
        // to a ParamSlots callee.
        let caller = body_of(
            5,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(call_fun(10, vec![], 3, 3)),
                term_bb(call_fun(8, vec![copy(3)], 4, 4)),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&caller, &llbc, &effect),
            "ParamSlots of a returns_index struct dest must be own"
        );
        let ref_of = json!({
            "kind": {"Local": 3},
            "ty": ty()
        });
        let via_ref = body_of(
            6,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(call_fun(10, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"Ref": {
                            "kind": "Mut",
                            "place": ref_of,
                            "ptr_metadata": {"Const": null}
                        }}),
                    )],
                    call_fun(8, vec![copy(4)], 5, 4),
                ),
                term_bb(drop_local(1, 5)),
                term_bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&via_ref, &llbc, &effect),
            "ParamSlots of a &mut returns_index dest must be own"
        );
    }

    #[test]
    fn bounds_assert_not_erased_when_length_local_is_reassigned() {
        let llbc = stack_ops_llbc_with_callee();
        let array = json!({"Assign": [
            place(2),
            {"Aggregate": [
                {"Array": [ty(), {"Deduplicated": 1}, null]},
                [
                    {"Move": place(7)},
                    {"Move": place(8)},
                    {"Move": place(9)}
                ]
            ]}
        ]});
        let len_of_array = json!({"Assign": [
            place(5),
            {"Len": place(2)}
        ]});
        let unknown = json!({"Assign": [
            place(5),
            {"Use": [{"Copy": place(6)}, "Yes"]}
        ]});
        let assert_term = json!({"Assert": {
            "assert": {
                "cond": {"Copy": place(3)},
                "expected": true,
                "check_kind": {"BoundsCheck": {
                    "len": {"Copy": place(5)},
                    "index": {"Copy": place(3)}
                }}
            },
            "target": 6,
            "on_unwind": 99
        }});
        let body = body_of(
            10,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                bb_stmts(
                    vec![stmt(array)],
                    call_fun(6, vec![json!({"Move": place(2)})], 3, 2),
                ),
                term_bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [3, 4]}
                }})),
                bb_stmts(vec![stmt(len_of_array)], json!({"Goto": {"target": 5}})),
                bb_stmts(vec![stmt(unknown)], json!({"Goto": {"target": 5}})),
                bb_stmts(vec![], assert_term),
                term_bb(drop_local(1, 7)),
                term_bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(plan)) => assert!(
                !plan.terms.contains_key(&5),
                "bounds assert must not be erased when the length local is assigned on two paths"
            ),
            Err(reason) => assert_eq!(
                reason, "unmodeled-use-in-terminator",
                "keeping the bracket is the sound answer when the length local disagrees across paths"
            ),
            Ok(None) => panic!("body has a root bracket"),
        }
    }

    #[test]
    fn bounds_check_against_unrelated_length_is_refused() {
        let specials = HashMap::from([(5usize, Special::Index(IdxFact::Const(0)))]);
        let assert: majit_charon_reader::ullbc::AssertStmt = serde_json::from_value(json!({
            "cond": {"Copy": place(5)},
            "expected": true,
            "check_kind": {"BoundsCheck": {
                "len": {"Copy": place(9)},
                "index": {"Copy": place(5)}
            }}
        }))
        .expect("assert");
        assert!(
            !assert_is_slot_index_bounds_check(
                &assert,
                &specials,
                &HashMap::new(),
                &fixture_llbc(),
                &HashMap::new(),
            ),
            "BoundsCheck against a length that is not the erased slot array must refuse"
        );
    }

    #[test]
    fn bounds_check_of_erased_slot_index_against_slot_len_is_accepted() {
        let specials = HashMap::from([
            (5usize, Special::Index(IdxFact::Const(0))),
            (6usize, Special::Index(IdxFact::Const(3))),
        ]);
        let assert: majit_charon_reader::ullbc::AssertStmt = serde_json::from_value(json!({
            "cond": {"Copy": place(5)},
            "expected": true,
            "check_kind": {"BoundsCheck": {
                "len": {"Copy": place(6)},
                "index": {"Copy": place(5)}
            }}
        }))
        .expect("assert");
        assert!(
            assert_is_slot_index_bounds_check(
                &assert,
                &specials,
                &HashMap::new(),
                &fixture_llbc(),
                &HashMap::new(),
            ),
            "index 0 of a 3-slot erased array is true by construction"
        );
    }

    fn getter_llbc() -> Llbc {
        let capture_body = json!({
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 2,
                    "locals": (0..4).map(local_decl).collect::<Vec<_>>()
                },
                "body": [
                    bb(call_fun(3, vec![json!({"Copy": place(1)})], 0, 1)),
                    bb(json!("Return"))
                ]
            }
        });
        let field0 = json!({
            "kind": {"Projection": [place(3), {"Field": [null, 0]}]},
            "ty": ty()
        });
        let plus_const_body = json!({
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 2,
                    "locals": (0..5).map(local_decl).collect::<Vec<_>>()
                },
                "body": [
                    bb_stmts(
                        vec![assign_local(
                            3,
                            json!({"BinaryOp": [
                                "AddChecked",
                                {"Copy": place(1)},
                                usize_lit(1)
                            ]}),
                        )],
                        json!({"Assert": {
                            "assert": {
                                "cond": {"Move": {
                                    "kind": {"Projection": [place(3), {"Field": [null, 1]}]},
                                    "ty": ty()
                                }},
                                "expected": false,
                                "check_kind": {"Overflow": [
                                    {"Add": "Wrap"},
                                    {"Copy": place(1)},
                                    usize_lit(1)
                                ]}
                            },
                            "target": 1,
                            "on_unwind": 99
                        }}),
                    ),
                    bb_stmts(
                        vec![assign_local(4, json!({"Use": [{"Move": field0}, "Yes"]}))],
                        call_fun(3, vec![json!({"Move": place(4)})], 0, 2),
                    ),
                    bb(json!("Return"))
                ]
            }
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["fixture", "capture", "call"], capture_body),
                    opaque_fun(6, &["fixture", "plus_const", "call"], plus_const_body),
                    opaque_fun(7, &["fixture", "other"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn copy_local(i: u64) -> Value {
        json!({"Copy": place(i)})
    }

    fn move_local(i: u64) -> Value {
        json!({"Move": place(i)})
    }

    fn borrow_local(dest: u64, src: u64) -> Value {
        assign_local(
            dest,
            json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    }

    fn closure_aggregate(dest: u64, capture: u64) -> Value {
        assign_local(
            dest,
            json!({"Aggregate": [{"Adt": [3026, null]}, [move_local(capture)]]}),
        )
    }

    fn empty_unit(dest: u64) -> Value {
        assign_local(dest, json!({"Aggregate": [{"Adt": [0, null]}, []]}))
    }

    fn pin_two_array_stmts() -> Vec<Value> {
        vec![
            assign_local(8, json!({"Use": [copy_local(0), "Yes"]})),
            assign_local(9, json!({"Use": [copy_local(0), "Yes"]})),
            assign_local(
                6,
                json!({"Aggregate": [
                    {"Array": [ty(), {"Deduplicated": 1}, null]},
                    [move_local(8), move_local(9)]
                ]}),
            ),
            borrow_local(7, 6),
        ]
    }

    fn capture_of_base(borrow: u64, closure: u64, recv: u64, unit: u64) -> Vec<Value> {
        vec![
            borrow_local(borrow, 4),
            closure_aggregate(closure, borrow),
            borrow_local(recv, closure),
            empty_unit(unit),
        ]
    }

    /// `long_rshift` shape: Open, `pin_roots(&[a, b])`, closures over the
    /// base, `shadow_stack_get` of the captured index and of `captured + 1`.
    fn long_rshift_shape_body(call_after_close: bool) -> Unstructured {
        let mut blocks = vec![
            bb(call_fun(1, vec![], 1, 1)),
            bb_stmts(
                pin_two_array_stmts(),
                call_fun(2, vec![move_local(7)], 4, 2),
            ),
        ];
        let capture_a = capture_of_base(12, 11, 18, 19);
        if call_after_close {
            blocks.push(bb_stmts(capture_a, drop_local(1, 3)));
            blocks.push(bb(call_fun(5, vec![copy_local(18), copy_local(19)], 30, 4)));
            blocks.push(bb(json!("Return")));
        } else {
            blocks.push(bb_stmts(
                capture_a,
                call_fun(5, vec![copy_local(18), copy_local(19)], 30, 3),
            ));
            blocks.push(bb_stmts(
                capture_of_base(14, 13, 20, 21),
                call_fun(6, vec![copy_local(20), copy_local(21)], 31, 4),
            ));
            blocks.push(bb(drop_local(1, 5)));
            blocks.push(bb(json!("Return")));
        }
        body_of(32, blocks)
    }

    #[test]
    fn analyze_erases_long_rshift_closure_capture() {
        let llbc = getter_llbc();
        let body = long_rshift_shape_body(false);
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        let n_locals = 32;
        let slot0 = local_place_json(n_locals, &ty());
        let slot1 = local_place_json(n_locals + 1, &ty());
        let rewrite0 = plan.terms.get(&2).expect("capture get rewritten");
        assert_eq!(
            rewrite0.stmts,
            vec![json!({"Assign": [
                place(30),
                {"Use": [{"Copy": slot0}, "Yes"]}
            ]})],
            "|| shadow_stack_get(base) is a copy of slot 0"
        );
        let rewrite1 = plan.terms.get(&3).expect("plus-const get rewritten");
        assert_eq!(
            rewrite1.stmts,
            vec![json!({"Assign": [
                place(31),
                {"Use": [{"Copy": slot1}, "Yes"]}
            ]})],
            "|| shadow_stack_get(base + 1) is a copy of slot 1"
        );
    }

    #[test]
    fn analyze_refuses_capture_escapes_when_closure_is_passed() {
        let llbc = getter_llbc();
        let blocks = vec![
            bb(call_fun(1, vec![], 1, 1)),
            bb_stmts(
                pin_two_array_stmts(),
                call_fun(2, vec![move_local(7)], 4, 2),
            ),
            bb_stmts(
                capture_of_base(12, 11, 18, 19),
                call_fun(5, vec![copy_local(18), copy_local(19)], 30, 3),
            ),
            bb(call_fun(7, vec![copy_local(11)], 31, 4)),
            bb(drop_local(1, 5)),
            bb(json!("Return")),
        ];
        let body = body_of(32, blocks);
        match analyze(&body, &llbc) {
            Err("capture-escapes") => {}
            Ok(Some(_)) => panic!("expected capture-escapes, got accepted"),
            Ok(None) => panic!("expected capture-escapes, got no bracket"),
            Err(reason) => panic!("expected capture-escapes, got {reason}"),
        }
    }

    #[test]
    fn analyze_refuses_closure_get_after_close() {
        let llbc = getter_llbc();
        let body = long_rshift_shape_body(true);
        match analyze(&body, &llbc) {
            Err("get-above-depth") => {}
            Ok(Some(_)) => panic!("expected get-above-depth, got accepted"),
            Ok(None) => panic!("expected get-above-depth, got no bracket"),
            Err(reason) => panic!("expected get-above-depth, got {reason}"),
        }
    }

    #[test]
    fn analyze_getter_body_still_refuses_index_not_static() {
        let llbc = getter_llbc();
        let fd = llbc.fn_by_id(5).expect("capture getter");
        let body = fd.unstructured().expect("getter body");
        match analyze(&body, &llbc) {
            Err("index-not-static") => {}
            Ok(Some(_)) => panic!("expected index-not-static, got accepted"),
            Ok(None) => panic!("expected index-not-static, got no bracket"),
            Err(reason) => panic!("expected index-not-static, got {reason}"),
        }
    }

    fn incoming_slice_llbc() -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn slice_index_use_json(slice_local: u64, dest: u64, k: u64) -> Value {
        let deref = json!({
            "kind": {"Projection": [place(slice_local), "Deref"]},
            "ty": ty()
        });
        json!({"Assign": [
            place(dest),
            {"Use": [{"Copy": {
                "kind": {"Projection": [deref, {"Index": {
                    "offset": {"Const": [
                        {"Integer": {"Unsigned": ["Usize", k.to_string()]}},
                        {"Deduplicated": 0}
                    ]},
                    "from_end": false
                }}]},
                "ty": ty()
            }}, "Yes"]}
        ]})
    }

    /// `__majit_wrap_dict_descr_keys` shape: incoming `&[PyObjectRef]`,
    /// `get(base)`, then `len` / `pin_root` / `get(len)`.
    fn dict_descr_keys_shape_body(get_after_close: bool) -> Unstructured {
        let mut blocks = vec![
            bb(call_fun(1, vec![], 2, 1)),
            bb(call_fun(5, vec![copy_local(1)], 3, 2)),
        ];
        if get_after_close {
            blocks.push(bb(drop_local(2, 3)));
            blocks.push(bb(call_fun(2, vec![copy_local(3)], 4, 4)));
            blocks.push(bb(json!("Return")));
        } else {
            blocks.push(bb(call_fun(2, vec![copy_local(3)], 4, 3)));
            blocks.push(bb(call_fun(3, vec![], 5, 4)));
            blocks.push(bb(call_fun(4, vec![copy_local(4)], 6, 5)));
            blocks.push(bb(call_fun(2, vec![copy_local(5)], 7, 6)));
            blocks.push(bb(drop_local(2, 7)));
            blocks.push(bb(json!("Return")));
        }
        body_of_args(1, 8, blocks)
    }

    #[test]
    fn analyze_erases_incoming_slice_then_scalar_pin() {
        let llbc = incoming_slice_llbc();
        let body = dict_descr_keys_shape_body(false);
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        let rewrite_base = plan.terms.get(&2).expect("get(base) rewritten");
        assert_eq!(
            rewrite_base.stmts,
            vec![slice_index_use_json(1, 4, 0)],
            "get(base) of an incoming slice is (*args)[0]"
        );
        let n_locals = 8;
        let slot0 = local_place_json(n_locals, &ty());
        let rewrite_len = plan.terms.get(&5).expect("get(len) rewritten");
        assert_eq!(
            rewrite_len.stmts,
            vec![json!({"Assign": [
                place(7),
                {"Use": [{"Copy": slot0}, "Yes"]}
            ]})],
            "get(len) after pin_root is scalar slot 0; the slice adds no depth"
        );
        assert_eq!(plan.slot_count, 1, "incoming slice is not a scalar slot");
    }

    #[test]
    fn analyze_erases_array_aggregate_in_predecessor_block() {
        let llbc = incoming_slice_llbc();
        let array = json!({"Aggregate": [
            {"Array": [ty(), {"Deduplicated": 1}, null]},
            [move_local(3), move_local(4)]
        ]});
        let body = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb_stmts(vec![assign_local(2, array)], json!({"Goto": {"target": 2}})),
                bb(call_fun(5, vec![move_local(2)], 5, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    #[test]
    fn analyze_refuses_incoming_slice_get_with_runtime_index() {
        let llbc = incoming_slice_llbc();
        let body = body_of_args(
            1,
            6,
            vec![
                bb(call_fun(1, vec![], 2, 1)),
                bb(call_fun(5, vec![copy_local(1)], 3, 2)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": ["Add", copy_local(3), copy_local(0)]}),
                    )],
                    call_fun(2, vec![copy_local(4)], 5, 3),
                ),
                bb(drop_local(2, 4)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("index-not-static") => {}
            Ok(Some(_)) => panic!("expected index-not-static, got accepted"),
            Ok(None) => panic!("expected index-not-static, got no bracket"),
            Err(reason) => panic!("expected index-not-static, got {reason}"),
        }
    }

    #[test]
    fn analyze_refuses_incoming_slice_get_after_close() {
        let llbc = incoming_slice_llbc();
        let body = dict_descr_keys_shape_body(true);
        match analyze(&body, &llbc) {
            Err("get-above-depth") => {}
            Ok(Some(_)) => panic!("expected get-above-depth, got accepted"),
            Ok(None) => panic!("expected get-above-depth, got no bracket"),
            Err(reason) => panic!("expected get-above-depth, got {reason}"),
        }
    }

    /// Early return at slice-only depth 0 joins `pin_root` (depth 1) at
    /// the guard's Drop; the Drop rewinds, so the join is accepted.
    #[test]
    fn analyze_erases_incoming_slice_early_return_join_at_drop() {
        let llbc = incoming_slice_llbc();
        let body = body_of_args(
            1,
            8,
            vec![
                bb(call_fun(1, vec![], 2, 1)),
                bb(call_fun(5, vec![copy_local(1)], 3, 2)),
                bb(call_fun(2, vec![copy_local(3)], 4, 3)),
                bb(json!({"Switch": {
                    "discr": {"Copy": place(4)},
                    "targets": {"If": [4, 5]}
                }})),
                bb(json!({"Goto": {"target": 8}})),
                bb(call_fun(3, vec![], 5, 6)),
                bb(call_fun(4, vec![copy_local(4)], 6, 7)),
                bb(call_fun(2, vec![copy_local(5)], 7, 8)),
                bb(drop_local(2, 9)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(plan)) => {
                assert_eq!(plan.slot_count, 1, "only the scalar pin is a slot");
            }
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    fn type_struct(def_id: u64, name: &[&str]) -> Value {
        json!({
            "def_id": def_id,
            "item_meta": {
                "name": name.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            },
            "kind": {"Struct": []}
        })
    }

    fn rooted_items_adt() -> Value {
        json!({
            "Adt": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    }

    fn rooted_items_ref() -> Value {
        json!({"Ref": ["Erased", rooted_items_adt(), "Mut"]})
    }

    fn value_ty() -> Value {
        json!({"Scalar": {"Integer": {"Unsigned": "U64"}}})
    }

    fn unit_ty() -> Value {
        json!({"Tuple": []})
    }

    fn usize_ty() -> Value {
        json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
    }

    fn bool_ty() -> Value {
        json!({"Scalar": "Bool"})
    }

    fn usize_const(k: u64) -> Value {
        json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, usize_ty()]})
    }

    fn place_ty(i: u64, ty: &Value) -> Value {
        json!({"kind": {"Local": i}, "ty": ty})
    }

    fn local_ty(i: u64, ty: Value) -> Value {
        json!({"index": i, "name": null, "span": span(), "ty": ty})
    }

    fn stmt(kind: Value) -> Value {
        json!({"span": span(), "kind": kind})
    }

    fn call_term(fun: u64, args: Vec<Value>, dest: Value, target: u64, on_unwind: u64) -> Value {
        json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                "args": args,
                "dest": dest
            },
            "target": target,
            "on_unwind": on_unwind
        }})
    }

    fn drop_term(local: u64, ty: &Value, target: u64, on_unwind: u64) -> Value {
        json!({"Drop": {
            "place": place_ty(local, ty),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": on_unwind
        }})
    }

    fn rooted_items_llbc(unstructured: Value) -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "type_decls": [
                    type_struct(0, &["pyre_object", "gc_roots", "RootedItems"])
                ],
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "RootedItems", "new"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "RootedItems", "push"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "RootedItems", "get"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "RootedItems", "take"], json!("Opaque")),
                    opaque_fun(
                        5,
                        &["pyre_object", "eval", "subject"],
                        json!({ "Unstructured": unstructured })
                    ),
                    opaque_fun(6, &["pyre_object", "gc_roots", "RootedItems", "len"], json!("Opaque")),
                    opaque_fun(7, &["pyre_object", "gc_roots", "RootedItems", "is_empty"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn assign_copies_local(body: &Unstructured, dest: u64, src: u64) -> bool {
        body.body.iter().any(|bb| {
            bb.statements.iter().any(|st| {
                let assign = &st.kind_value()["Assign"];
                assign[0]["kind"].get("Local") == Some(&json!(dest))
                    && assign[1]["Use"][0]
                        .get("Copy")
                        .and_then(|p| p["kind"].get("Local"))
                        == Some(&json!(src))
            })
        })
    }

    fn assign_uses_const_lit(body: &Unstructured, dest: u64, lit: &Value) -> bool {
        body.body.iter().any(|bb| {
            bb.statements.iter().any(|st| {
                let assign = &st.kind_value()["Assign"];
                assign[0]["kind"].get("Local") == Some(&json!(dest))
                    && assign[1]["Use"][0].get("Const").and_then(|c| c.get(0)) == Some(lit)
            })
        })
    }

    fn rooted_items_call_paths(body: &Unstructured, llbc: &Llbc) -> Vec<String> {
        body.body
            .iter()
            .filter_map(|bb| match bb.term_ref(llbc) {
                Ok(TermKind::Call { call, .. }) => {
                    super::callee_path(call, llbc).filter(|path| path.contains("RootedItems"))
                }
                _ => None,
            })
            .collect()
    }

    /// `RootedItems::new` → `push(a)` → `push(b)` → `get(0)` / `get(_t)`
    /// (`_t = const 1`) → `drop`. The rewrite removes every `RootedItems::*`
    /// call and both gets read the pinned values.
    #[test]
    fn rooted_items_bracket_erases() {
        let items = rooted_items_adt();
        let items_ref = rooted_items_ref();
        let value = value_ty();
        let unit = unit_ty();
        let usize = usize_ty();
        let copy = |i: u64, ty: &Value| json!({"Copy": place_ty(i, ty)});
        let mv = |i: u64, ty: &Value| json!({"Move": place_ty(i, ty)});
        let borrow = stmt(json!({
            "Assign": [
                place_ty(2, &items_ref),
                {"Ref": {"place": place_ty(1, &items), "kind": "Mut", "ptr_metadata": null}}
            ]
        }));
        let t_const = stmt(json!({
            "Assign": [place_ty(7, &usize), {"Use": [usize_const(1), "Yes"]}]
        }));
        // bb0 new → bb1 push(a) → bb2 push(b) → bb3 get(0)
        // → bb4 _t = 1; get(_t) → bb5 drop → bb6 return; bb7 unwind.
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    local_ty(0, unit.clone()),
                    local_ty(1, items.clone()),
                    local_ty(2, items_ref.clone()),
                    local_ty(3, value.clone()),
                    local_ty(4, value.clone()),
                    local_ty(5, value.clone()),
                    local_ty(6, value.clone()),
                    local_ty(7, usize.clone()),
                    local_ty(8, unit.clone())
                ]
            },
            "body": [
                bb_stmts(vec![], call_term(1, vec![], place_ty(1, &items), 1, 7)),
                bb_stmts(
                    vec![borrow],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(3, &value)],
                        place_ty(8, &unit),
                        2,
                        7
                    )
                ),
                bb_stmts(
                    vec![],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(4, &value)],
                        place_ty(8, &unit),
                        3,
                        7
                    )
                ),
                bb_stmts(
                    vec![],
                    call_term(
                        3,
                        vec![copy(2, &items_ref), usize_const(0)],
                        place_ty(5, &value),
                        4,
                        7
                    )
                ),
                bb_stmts(
                    vec![t_const],
                    call_term(
                        3,
                        vec![copy(2, &items_ref), copy(7, &usize)],
                        place_ty(6, &value),
                        5,
                        7
                    )
                ),
                bb_stmts(vec![], drop_term(1, &items, 6, 7)),
                bb_stmts(vec![], json!("Return")),
                bb_stmts(vec![], json!("UnwindResume"))
            ]
        });
        let llbc = rooted_items_llbc(unstructured);
        let fd = llbc.fn_by_id(5).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        let erased = match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(erased)) => erased,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        assert!(
            rooted_items_call_paths(&erased, &llbc).is_empty(),
            "erased body still calls RootedItems::*: {:?}",
            rooted_items_call_paths(&erased, &llbc)
        );
        // 9 original locals; slot 0 is local 9 (from `a`), slot 1 is local 10.
        assert!(
            assign_copies_local(&erased, 9, 3),
            "push(a) should write slot 0 from a"
        );
        assert!(
            assign_copies_local(&erased, 10, 4),
            "push(b) should write slot 1 from b"
        );
        assert!(
            assign_copies_local(&erased, 5, 9),
            "get(0) should read the value pinned at slot 0"
        );
        assert!(
            assign_copies_local(&erased, 6, 10),
            "get(_t) with _t = const 1 should read the value pinned at slot 1"
        );
    }

    /// `RootedItems::new` → `push(a)` → `len` → `get(0)` → `drop`.
    /// `len` is `rlist.py` `ll_length`: dest is the pin count, not a deleted call.
    #[test]
    fn rooted_items_len_assigns_pin_count() {
        let items = rooted_items_adt();
        let items_ref = rooted_items_ref();
        let value = value_ty();
        let unit = unit_ty();
        let usize = usize_ty();
        let copy = |i: u64, ty: &Value| json!({"Copy": place_ty(i, ty)});
        let mv = |i: u64, ty: &Value| json!({"Move": place_ty(i, ty)});
        let borrow = stmt(json!({
            "Assign": [
                place_ty(2, &items_ref),
                {"Ref": {"place": place_ty(1, &items), "kind": "Mut", "ptr_metadata": null}}
            ]
        }));
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    local_ty(0, unit.clone()),
                    local_ty(1, items.clone()),
                    local_ty(2, items_ref.clone()),
                    local_ty(3, value.clone()),
                    local_ty(4, usize.clone()),
                    local_ty(5, value.clone()),
                    local_ty(6, unit.clone())
                ]
            },
            "body": [
                bb_stmts(vec![], call_term(1, vec![], place_ty(1, &items), 1, 6)),
                bb_stmts(
                    vec![borrow],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(3, &value)],
                        place_ty(6, &unit),
                        2,
                        6
                    )
                ),
                bb_stmts(
                    vec![],
                    call_term(6, vec![copy(2, &items_ref)], place_ty(4, &usize), 3, 6)
                ),
                bb_stmts(
                    vec![],
                    call_term(
                        3,
                        vec![copy(2, &items_ref), usize_const(0)],
                        place_ty(5, &value),
                        4,
                        6
                    )
                ),
                bb_stmts(vec![], drop_term(1, &items, 5, 6)),
                bb_stmts(vec![], json!("Return")),
                bb_stmts(vec![], json!("UnwindResume"))
            ]
        });
        let llbc = rooted_items_llbc(unstructured);
        let fd = llbc.fn_by_id(5).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        let erased = match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(erased)) => erased,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        assert!(
            rooted_items_call_paths(&erased, &llbc).is_empty(),
            "erased body still calls RootedItems::*: {:?}",
            rooted_items_call_paths(&erased, &llbc)
        );
        assert!(
            assign_uses_const_lit(
                &erased,
                4,
                &json!({ "Integer": { "Unsigned": ["Usize", "1"] } })
            ),
            "len dest should be the pin count 1"
        );
        assert!(
            assign_copies_local(&erased, 7, 3),
            "push(a) should write slot 0 from a"
        );
        assert!(
            assign_copies_local(&erased, 5, 7),
            "get(0) should read the value pinned at slot 0"
        );
    }

    /// `RootedItems::new` → `is_empty` → branch → `drop`.
    /// Empty set: dest is `true` (`count == 0`), and the If still branches on it.
    #[test]
    fn rooted_items_is_empty_assigns_zero_count() {
        let items = rooted_items_adt();
        let items_ref = rooted_items_ref();
        let unit = unit_ty();
        let flag_ty = bool_ty();
        let copy = |i: u64, ty: &Value| json!({"Copy": place_ty(i, ty)});
        let borrow = stmt(json!({
            "Assign": [
                place_ty(2, &items_ref),
                {"Ref": {"place": place_ty(1, &items), "kind": "Mut", "ptr_metadata": null}}
            ]
        }));
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    local_ty(0, unit.clone()),
                    local_ty(1, items.clone()),
                    local_ty(2, items_ref.clone()),
                    local_ty(3, flag_ty.clone())
                ]
            },
            "body": [
                bb_stmts(vec![], call_term(1, vec![], place_ty(1, &items), 1, 7)),
                bb_stmts(
                    vec![borrow],
                    call_term(7, vec![copy(2, &items_ref)], place_ty(3, &flag_ty), 2, 7)
                ),
                bb_stmts(
                    vec![],
                    json!({"Switch": {
                        "discr": {"Copy": place_ty(3, &flag_ty)},
                        "targets": {"If": [3, 4]}
                    }})
                ),
                bb_stmts(vec![], json!({"Goto": {"target": 5}})),
                bb_stmts(vec![], json!({"Goto": {"target": 5}})),
                bb_stmts(vec![], drop_term(1, &items, 6, 7)),
                bb_stmts(vec![], json!("Return")),
                bb_stmts(vec![], json!("UnwindResume"))
            ]
        });
        let llbc = rooted_items_llbc(unstructured);
        let fd = llbc.fn_by_id(5).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        let erased = match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(erased)) => erased,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        assert!(
            rooted_items_call_paths(&erased, &llbc).is_empty(),
            "erased body still calls RootedItems::*: {:?}",
            rooted_items_call_paths(&erased, &llbc)
        );
        assert!(
            assign_uses_const_lit(&erased, 3, &json!({ "Bool": true })),
            "is_empty dest should be true when the pin count is 0"
        );
        let switches_on_flag = erased.body.iter().any(|bb| {
            let kind = bb.terminator.kind_value();
            kind.get("Switch")
                .and_then(|s| s.get("discr"))
                .and_then(|d| d.get("Copy"))
                .and_then(|p| p["kind"].get("Local"))
                == Some(&json!(3))
        });
        assert!(
            switches_on_flag,
            "zero-count is_empty should still feed the If"
        );
    }

    /// `_t = 0; _t = 1; get(_t)` is not a single-definition constant:
    /// keep the bracket rather than rewrite `get(1)`.
    #[test]
    fn rooted_items_get_follows_reassigned_const() {
        let items = rooted_items_adt();
        let items_ref = rooted_items_ref();
        let value = value_ty();
        let unit = unit_ty();
        let usize = usize_ty();
        let copy = |i: u64, ty: &Value| json!({"Copy": place_ty(i, ty)});
        let mv = |i: u64, ty: &Value| json!({"Move": place_ty(i, ty)});
        let borrow = stmt(json!({
            "Assign": [
                place_ty(2, &items_ref),
                {"Ref": {"place": place_ty(1, &items), "kind": "Mut", "ptr_metadata": null}}
            ]
        }));
        let t_zero = stmt(json!({
            "Assign": [place_ty(7, &usize), {"Use": [usize_const(0), "Yes"]}]
        }));
        let t_one = stmt(json!({
            "Assign": [place_ty(7, &usize), {"Use": [usize_const(1), "Yes"]}]
        }));
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    local_ty(0, unit.clone()),
                    local_ty(1, items.clone()),
                    local_ty(2, items_ref.clone()),
                    local_ty(3, value.clone()),
                    local_ty(4, value.clone()),
                    local_ty(5, value.clone()),
                    local_ty(6, unit.clone()),
                    local_ty(7, usize.clone())
                ]
            },
            "body": [
                bb_stmts(vec![], call_term(1, vec![], place_ty(1, &items), 1, 6)),
                bb_stmts(
                    vec![borrow],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(3, &value)],
                        place_ty(6, &unit),
                        2,
                        6
                    )
                ),
                bb_stmts(
                    vec![],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(4, &value)],
                        place_ty(6, &unit),
                        3,
                        6
                    )
                ),
                bb_stmts(
                    vec![t_zero, t_one],
                    call_term(
                        3,
                        vec![copy(2, &items_ref), copy(7, &usize)],
                        place_ty(5, &value),
                        4,
                        6
                    )
                ),
                bb_stmts(vec![], drop_term(1, &items, 5, 6)),
                bb_stmts(vec![], json!("Return")),
                bb_stmts(vec![], json!("UnwindResume"))
            ]
        });
        let llbc = rooted_items_llbc(unstructured);
        let fd = llbc.fn_by_id(5).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        match erase_shadow_stack(fd, &body, &llbc) {
            Err("index-not-static") => {}
            Ok(Some(_)) => panic!("reassigned const index should refuse the rewrite, not erase"),
            Ok(None) => {
                panic!("reassigned const index should refuse the rewrite, not skip as no bracket")
            }
            Err(reason) => panic!("expected index-not-static, got {reason}"),
        }
    }

    /// `_t = const 0` then `_t = copy(dynamic)` then `get(_t)` is not a
    /// static index: keep the bracket rather than rewrite `get(0)`.
    #[test]
    fn rooted_items_get_after_dynamic_index_keeps_the_bracket() {
        let items = rooted_items_adt();
        let items_ref = rooted_items_ref();
        let value = value_ty();
        let unit = unit_ty();
        let usize = usize_ty();
        let copy = |i: u64, ty: &Value| json!({"Copy": place_ty(i, ty)});
        let mv = |i: u64, ty: &Value| json!({"Move": place_ty(i, ty)});
        let borrow = stmt(json!({
            "Assign": [
                place_ty(2, &items_ref),
                {"Ref": {"place": place_ty(1, &items), "kind": "Mut", "ptr_metadata": null}}
            ]
        }));
        let t_const = stmt(json!({
            "Assign": [place_ty(7, &usize), {"Use": [usize_const(0), "Yes"]}]
        }));
        let t_dyn = stmt(json!({
            "Assign": [place_ty(7, &usize), {"Use": [copy(8, &usize), "Yes"]}]
        }));
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    local_ty(0, unit.clone()),
                    local_ty(1, items.clone()),
                    local_ty(2, items_ref.clone()),
                    local_ty(3, value.clone()),
                    local_ty(4, value.clone()),
                    local_ty(5, value.clone()),
                    local_ty(6, unit.clone()),
                    local_ty(7, usize.clone()),
                    local_ty(8, usize.clone())
                ]
            },
            "body": [
                bb_stmts(vec![], call_term(1, vec![], place_ty(1, &items), 1, 6)),
                bb_stmts(
                    vec![borrow],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(3, &value)],
                        place_ty(6, &unit),
                        2,
                        6
                    )
                ),
                bb_stmts(
                    vec![],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(4, &value)],
                        place_ty(6, &unit),
                        3,
                        6
                    )
                ),
                bb_stmts(
                    vec![t_const, t_dyn],
                    call_term(
                        3,
                        vec![copy(2, &items_ref), copy(7, &usize)],
                        place_ty(5, &value),
                        4,
                        6
                    )
                ),
                bb_stmts(vec![], drop_term(1, &items, 5, 6)),
                bb_stmts(vec![], json!("Return")),
                bb_stmts(vec![], json!("UnwindResume"))
            ]
        });
        let llbc = rooted_items_llbc(unstructured);
        let fd = llbc.fn_by_id(5).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        match erase_shadow_stack(fd, &body, &llbc) {
            Err("index-not-static") => {}
            Ok(Some(_)) => panic!("dynamic index should refuse the rewrite, not erase"),
            Ok(None) => panic!("dynamic index should refuse the rewrite, not skip as no bracket"),
            Err(reason) => panic!("expected index-not-static, got {reason}"),
        }
    }

    /// `RootedItems::take` is Unmodeled: new → push → take → drop keeps the
    /// bracket.
    #[test]
    fn rooted_items_take_keeps_the_bracket() {
        let items = rooted_items_adt();
        let items_ref = rooted_items_ref();
        let value = value_ty();
        let unit = unit_ty();
        let copy = |i: u64, ty: &Value| json!({"Copy": place_ty(i, ty)});
        let mv = |i: u64, ty: &Value| json!({"Move": place_ty(i, ty)});
        let borrow = stmt(json!({
            "Assign": [
                place_ty(2, &items_ref),
                {"Ref": {"place": place_ty(1, &items), "kind": "Mut", "ptr_metadata": null}}
            ]
        }));
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    local_ty(0, unit.clone()),
                    local_ty(1, items.clone()),
                    local_ty(2, items_ref.clone()),
                    local_ty(3, value.clone()),
                    local_ty(4, unit.clone()),
                    local_ty(5, unit.clone())
                ]
            },
            "body": [
                bb_stmts(vec![], call_term(1, vec![], place_ty(1, &items), 1, 5)),
                bb_stmts(
                    vec![borrow],
                    call_term(
                        2,
                        vec![copy(2, &items_ref), mv(3, &value)],
                        place_ty(4, &unit),
                        2,
                        5
                    )
                ),
                bb_stmts(
                    vec![],
                    call_term(
                        4,
                        vec![copy(2, &items_ref)],
                        place_ty(5, &unit),
                        3,
                        5
                    )
                ),
                bb_stmts(vec![], drop_term(1, &items, 4, 5)),
                bb_stmts(vec![], json!("Return")),
                bb_stmts(vec![], json!("UnwindResume"))
            ]
        });
        let llbc = rooted_items_llbc(unstructured);
        let fd = llbc.fn_by_id(5).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        match erase_shadow_stack(fd, &body, &llbc) {
            Err("unmodeled-stack-op") => {}
            Ok(Some(_)) => panic!("take should refuse the rewrite, not erase the bracket"),
            Ok(None) => panic!("take should refuse the rewrite, not skip it as no bracket"),
            Err(reason) => panic!("expected unmodeled-stack-op, got {reason}"),
        }
        let kept = erase_or_keep(fd, body, &llbc);
        let paths = rooted_items_call_paths(&kept, &llbc);
        assert!(
            paths.iter().any(|p| p.ends_with("::take")),
            "unerased body should still call RootedItems::take, got {paths:?}"
        );
        assert!(
            paths.iter().any(|p| p.ends_with("::new")),
            "unerased body should still call RootedItems::new, got {paths:?}"
        );
    }

    /// A callee that opens a nested `RootScope` and returns it in the dest,
    /// then ParamSlots methods of that dest. The leftover is the dest the
    /// caller drops, not this body's slots.
    fn owned_scope_llbc() -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["other", "owned_scope_ctor"], json!("Opaque")),
                    opaque_fun(6, &["other", "owned_scope_get"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.fn_by_id(5)
            .expect("owned_scope_ctor")
            .returns_owned_scope
            .store(true, Ordering::Relaxed);
        llbc.register_stack_sensitive_fns([
            "other::owned_scope_ctor".into(),
            "other::owned_scope_get".into(),
        ]);
        llbc.register_stack_returns_index_fns(["other::owned_scope_ctor".into()]);
        llbc.register_stack_param_slots_fns([("other::owned_scope_get".into(), vec![0])]);
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn returned_owned_scope_body() -> Unstructured {
        let copy = |i| json!({"Copy": place(i)});
        body_of(
            5,
            vec![
                term_bb(call_fun(1, vec![], 1, 1)),
                term_bb(call_fun(4, vec![copy(0)], 2, 2)),
                term_bb(call_fun(5, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"Ref": {
                            "place": place(3),
                            "kind": "Shared",
                            "ptr_metadata": null
                        }}),
                    )],
                    call_fun(6, vec![copy(4)], 2, 4),
                ),
                term_bb(drop_local(3, 5)),
                term_bb(drop_local(1, 6)),
                term_bb(json!("Return")),
            ],
        )
    }

    #[test]
    fn analyze_accepts_returned_owned_scope_inside_open() {
        let llbc = owned_scope_llbc();
        match analyze(&returned_owned_scope_body(), &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!(
                "expected erasure of Open/Pin/owned-scope dest/ParamSlots/Close, got {reason}"
            ),
        }
    }

    #[test]
    fn analyze_refuses_returns_index_that_is_not_an_owned_scope() {
        let llbc = owned_scope_llbc();
        llbc.fn_by_id(5)
            .expect("owned_scope_ctor")
            .returns_owned_scope
            .store(false, Ordering::Relaxed);
        match analyze(&returned_owned_scope_body(), &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
    }
}
