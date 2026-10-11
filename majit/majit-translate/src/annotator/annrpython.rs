//! RPython `rpython/annotator/annrpython.py` — `RPythonAnnotator` driver.
//!
//! Holds the `RPythonAnnotator` public surface that the `binaryop` /
//! `unaryop` dispatchers consume, together with the driver itself
//! (`build_types`, `complete`, `processblock`, `consider_op`, `flowin`,
//! …).
//!
//! Fields and method signatures mirror upstream line-by-line; method
//! bodies that still require un-ported machinery (pendingblocks queue,
//! TranslationContext, policy specialize, …) carry the upstream
//! comment verbatim and a `todo!()` stub so every stub surfaces at
//! runtime rather than silently no-op'ing.

use indexmap::{IndexMap, IndexSet};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::flowspace::model::{
    BlockKey, BlockRef, GraphKey, GraphRef, Hlvalue, LinkKey, LinkRef, Variable, checkgraph,
};
use super::bookkeeper::{Bookkeeper, PositionKey};
use super::classdesc::{ClassAttrJournal, ClassDef};
use super::listdef::{ItemOwner, ListItem, ListItemJournal};
use super::model::{SomeValue, TLS, UnionError, unionof};
use super::policy::{AnnotatorPolicy, PolicyHandle};
use crate::tool::ansi_print::AnsiLogger;
use crate::translator::translator::TranslationContext;

/// RPython `annrpython.py log = py.log.Producer("annrpython")`.
/// Upstream uses `py.log` which Pyre approximates with [`AnsiLogger`]
/// (`rpython/tool/ansi_print.py` parity port); both unconditionally
/// route output to stderr.
pub static LOG: AnsiLogger = AnsiLogger::new("annrpython");

// Exists to localise prepass nondeterminism (gh#1139).
static REFLOW_COUNT: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn reflow_count() -> u64 {
    REFLOW_COUNT.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
static PROCESSBLOCK_COUNT: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn processblock_count() -> u64 {
    PROCESSBLOCK_COUNT.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
static REFLOW_FROM_NOTIFY: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn reflow_from_notify_count() -> u64 {
    REFLOW_FROM_NOTIFY.load(Ordering::Relaxed)
}

// Exists to localise prepass nondeterminism (gh#1139).
static NOTIFY_HIT_ON_REUSED: AtomicU64 = AtomicU64::new(0);

// Exists to localise prepass nondeterminism (gh#1139).
pub fn notify_hit_on_reused_count() -> u64 {
    NOTIFY_HIT_ON_REUSED.load(Ordering::Relaxed)
}

/// One entry of [`RPythonAnnotator::notify`].
pub(crate) struct NotifyEntry {
    /// Retains the block so its address cannot be recycled by a later
    /// block — the identity the key encodes must stay unique for as
    /// long as the entry lives.
    block: BlockRef,
    positions: IndexSet<PositionKey>,
}

/// RPython `class RPythonAnnotator(object)` (annrpython.py).
///
/// "Block annotator for RPython."
pub struct RPythonAnnotator {
    /// RPython `self.translator` (annrpython.py:30-35):
    ///
    /// ```python
    /// if translator is None:
    ///     from rpython.translator.translator import TranslationContext
    ///     translator = TranslationContext()
    ///     translator.annotator = self
    /// self.translator = translator
    /// ```
    ///
    /// Always populated; `new()` defaults it when `None` is supplied.
    /// Rust keeps a strong reference here and mirrors upstream's
    /// `translator.annotator = self` only on the implicit-translator
    /// path.
    pub translator: Rc<TranslationContext>,
    /// RPython `self.genpendingblocks = [{}]` (annrpython.py:36).
    /// Per-generation list of `{block: graph-containing-it}`. The
    /// inner map pairs identity-keyed blocks with the owning graph
    /// reference so `processblock(graph, block)` can recover the graph
    /// later in `complete_pending_blocks()`.
    pub genpendingblocks: RefCell<Vec<IndexMap<BlockKey, (BlockRef, GraphRef)>>>,
    /// RPython `self.annotated = {}` (annrpython.py).
    ///
    /// Upstream stores `dict[Block, bool | FunctionGraph]`:
    ///   * `block not in self.annotated`: never seen.
    ///   * `self.annotated[block] == False`: inputs bound, awaiting
    ///     `flowin()`.
    ///   * `self.annotated[block] == graph`: done.
    ///     Rust port models the two non-absent states as
    ///     `Option<GraphRef>`: `None = False`, `Some = graph`.
    pub annotated: RefCell<IndexMap<BlockKey, Option<GraphRef>>>,
    /// Rust-side identity→BlockRef index. Upstream's `self.annotated`
    /// dict keys ARE the Python `Block` objects, so iterating the dict
    /// yields block references directly. The Rust `BlockKey` identity
    /// indirection loses that, so we maintain this parallel map —
    /// populated in lockstep with every `annotated.insert` and
    /// `blocked_blocks.insert` — to let `call_sites()` walk "all seen
    /// blocks" without hunting through auxiliary tables.
    pub all_blocks: RefCell<IndexMap<BlockKey, BlockRef>>,
    /// RPython `self.added_blocks = None` (annrpython.py).
    ///
    /// Upstream sentinel: `None = track nothing`, `{} = start
    /// tracking`, filled by `processblock`. Rust port stores
    /// identity-keyed blocks; the value side is `()` (we only use it
    /// as a set).
    pub added_blocks: RefCell<Option<IndexMap<BlockKey, BlockRef>>>,
    /// Pyre's per-subject dual-gate continues after annotation failures,
    /// unlike upstream's single fatal Translator run. While an
    /// `added_blocks` scope is active, this transaction log captures only
    /// previously-seen blocks before their input bindings are widened or
    /// reflowed, so an uncommitted scope can restore them without cloning the
    /// entire monotonically-growing annotator session for every subject.
    subject_annotation_snapshots: RefCell<Option<IndexMap<BlockKey, BlockAnnotationSnapshot>>>,
    /// In-place `ListItem` mutations made while an added-blocks scope is
    /// active. Uncommitted `AddedBlocksGuard` restores op-result bindings
    /// but `ListItem.merge` widens the shared item outside that snapshot;
    /// `mergeinputargs` (annrpython.py) does not see it because `SomeList`
    /// equality is listitem identity. Undo the item with the bindings or
    /// `rtype_getitem`'s lltype no longer matches `hop.r_result`. `None`
    /// outside a scope: ordinary annotation does not record.
    listitem_journal: RefCell<Option<ListItemJournal>>,
    /// `ClassDef.attrs` replacements made while an added-blocks scope is
    /// active. `ListItem.merge` restores the item, but
    /// `ClassDef.generalize_attr` leaves the slot on the failed
    /// subject's `ListDef`. `setbinding` then sees two lists that do
    /// not `contains` each other. `None` outside a scope.
    class_attr_journal: RefCell<Option<ClassAttrJournal>>,
    /// RPython `self.links_followed = {}` (annrpython.py).
    /// Retains each link because the key is its address: a link removed from a
    /// block's `exits` would otherwise be freed, and a later link allocated at
    /// that address would read back as already followed.
    pub links_followed: RefCell<IndexMap<LinkKey, LinkRef>>,
    /// RPython `self.notify = {}` (annrpython.py).
    /// The ordered container is required because the key hashes on a
    /// pointer and the loop over the values produces a work order.
    pub(crate) notify: RefCell<IndexMap<BlockKey, NotifyEntry>>,
    /// RPython `self.fixed_graphs = {}` (annrpython.py). Graphs
    /// that have already been rtyped — `addpendingblock` rejects new
    /// pending entries against these.
    pub fixed_graphs: RefCell<IndexMap<GraphKey, GraphRef>>,
    /// RPython `self.blocked_blocks = {}` (annrpython.py).
    /// `{blocked_block: (graph, opindex)}`. `opindex == None` upstream
    /// → `None` here (block is blocked at entry, not mid-flow).
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub blocked_blocks: RefCell<IndexMap<BlockKey, (BlockRef, GraphRef, Option<usize>)>>,
    /// RPython `self.blocked_graphs = {}` (annrpython.py). Records
    /// graphs that have at least one blocked block; the `bool` value
    /// tracks the `blocked_graphs[graph] = True` flip in `complete()`.
    pub blocked_graphs: RefCell<IndexMap<GraphKey, (GraphRef, bool)>>,
    /// RPython `self.frozen = False` (annrpython.py).
    pub frozen: RefCell<bool>,
    /// RPython `self.policy` (annrpython.py:47-51).
    pub policy: RefCell<PolicyHandle>,
    /// RPython `self.bookkeeper` (annrpython.py:52-54).
    pub bookkeeper: Rc<Bookkeeper>,
    /// RPython `self.keepgoing` (annrpython.py:55).
    pub keepgoing: bool,
    /// RPython `self.failed_blocks = set()` (annrpython.py).
    pub failed_blocks: RefCell<IndexMap<BlockKey, BlockRef>>,
    /// RPython `self.errors = []` (annrpython.py).
    pub errors: RefCell<Vec<String>>,
}

/// RPython `class BlockedInference(Exception)` (annrpython.py).
///
/// Thrown from `consider_op` / `flowin` to signal "the situation is
/// currently blocked, try other blocks and come back later". The
/// driver catches it in `flowin` / `processblock` and reflects the
/// block into `blocked_blocks` for a later retry.
///
/// Parity deviation (Rust adaptation): upstream uses `raise` /
/// `except BlockedInference`. Rust returns `Result<_, BlockedInference>`
/// from `consider_op`, and `flowin` converts back into the upstream
/// exception flow via match. The struct-level fields match upstream's
/// public attribute names.
pub struct BlockedInference {
    /// RPython `self.op` — the operation that tripped the block (may
    /// be the same raising op at the block tail). Stored as
    /// `HLOperation` because upstream consumers (`flowin`) inspect
    /// `self.op.opname`, which Rust recovers via `op.kind.opname()`.
    pub op: super::super::flowspace::operation::HLOperation,
    /// RPython `self.opindex` — the operation's position in
    /// `block.operations`. `None` for the `-1` sentinel upstream uses
    /// when the op came from a direct `consider_op` call.
    pub opindex: Option<usize>,
    /// RPython `self.break_at` — the bookkeeper's position at the
    /// moment of the block, or `None` if no reflow frame is active.
    pub(crate) break_at: Option<PositionKey>,
}

impl BlockedInference {
    /// RPython `BlockedInference.__init__(self, annotator, op, opindex)`
    /// (annrpython.py).
    pub fn new(
        ann: &RPythonAnnotator,
        op: super::super::flowspace::operation::HLOperation,
        opindex: Option<usize>,
    ) -> Self {
        BlockedInference {
            op,
            opindex,
            break_at: ann.bookkeeper.current_position_key(),
        }
    }
}

impl std::fmt::Debug for BlockedInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockedInference({:?})", self.op.kind)
    }
}

impl std::fmt::Display for BlockedInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.break_at {
            Some(pk) => write!(
                f,
                "<BlockedInference break_at graph={} block={} op={} [{:?}]>",
                pk.graph_id, pk.block_id, pk.op_index, self.op.kind
            ),
            None => write!(f, "<BlockedInference break_at ? [{:?}]>", self.op.kind),
        }
    }
}

/// Outcome of [`RPythonAnnotator::flowin_op_loop`] — decides which
/// exits-side dispatch path runs.
///
/// Upstream encodes this via try/except control flow; the Rust port
/// explicits the four outcomes so the exits-side logic can run after
/// all three "except" clauses have agreed on a final `exits` list.
enum OpLoopOutcome {
    /// upstream `else:` — op loop finished without blocks; dead-exit
    /// pruning applies before the exits-side dispatch runs.
    Normal,
    /// upstream `if e.op is block.raising_op:` — exits-side dispatch
    /// runs with `exits = [l for l in block.exits if l.exitcase is not None]`.
    BlockedOnRaisingOp,
    /// upstream `elif e.op.opname in ('simple_call', 'call_args', 'next'):
    /// return` — the block is harmlessly blocked and flowin should
    /// silently return without touching exits/notify.
    HarmlesslySwallowed,
    /// upstream `else: raise` — processblock records the block in
    /// `blocked_blocks`.
    Blocked(BlockedInference),
}

/// The three exception classes `flowin` (annrpython.py)
/// catches during the op loop, ported as a Rust `Result` error type
/// so the loop can dispatch without `panic!` / `catch_unwind`.
///
/// Upstream structure:
/// ```python
/// try:
///     ...
/// except BlockedInference as e:
///     ...
/// except HarmlesslyBlocked:
///     return
/// except AnnotatorError as e:
///     e.source = gather_error(self, graph, block, i)
///     if self.keepgoing:
///         self.errors.append(e)
///         self.failed_blocks.add(block)
///         return
///     raise
/// ```
enum FlowinError {
    /// upstream `BlockedInference` — transient block, retry later.
    Blocked(BlockedInference),
    /// upstream `annmodel.HarmlesslyBlocked` — swallow, return.
    Harmless(crate::annotator::model::HarmlesslyBlocked),
    /// upstream `annmodel.AnnotatorError` (incl. `UnionError`) —
    /// route through gather_error + keepgoing.
    Annotator(crate::annotator::model::AnnotatorError),
}

impl From<BlockedInference> for FlowinError {
    fn from(e: BlockedInference) -> Self {
        FlowinError::Blocked(e)
    }
}

impl From<crate::annotator::model::HarmlesslyBlocked> for FlowinError {
    fn from(e: crate::annotator::model::HarmlesslyBlocked) -> Self {
        FlowinError::Harmless(e)
    }
}

impl From<crate::annotator::model::AnnotatorError> for FlowinError {
    fn from(e: crate::annotator::model::AnnotatorError) -> Self {
        FlowinError::Annotator(e)
    }
}

impl From<crate::annotator::model::AnnotatorException> for FlowinError {
    fn from(e: crate::annotator::model::AnnotatorException) -> Self {
        use crate::annotator::model::AnnotatorException;
        match e {
            AnnotatorException::Annotator(e) => FlowinError::Annotator(e),
            AnnotatorException::Harmless(e) => FlowinError::Harmless(e),
        }
    }
}

/// RPython `gather_error(annotator, graph, block, operindex)`
/// (tool/error.py) — thin wrapper that binds the
/// [`crate::tool::error::gather_error`] port into annrpython's legacy
/// call shape. The operindex is `Option<usize>` upstream (`None`
/// means "no op, block-level"); callers passing a concrete index wrap
/// it in `Some`.
fn gather_error(ann: &RPythonAnnotator, graph: &GraphRef, block: &BlockRef, i: usize) -> String {
    crate::tool::error::gather_error(ann, graph, block, Some(i))
}

/// RAII guard returned by [`RPythonAnnotator::using_policy`] — mirrors
/// upstream's `@contextmanager` so callers use `let _g =
/// ann.using_policy(...);` in the same shape as `with
/// self.using_policy(policy):`.
pub(crate) struct PolicyGuard<'a> {
    ann: &'a RPythonAnnotator,
    saved: Option<PolicyHandle>,
}

impl<'a> Drop for PolicyGuard<'a> {
    fn drop(&mut self) {
        if let Some(saved) = self.saved.take() {
            *self.ann.policy.borrow_mut() = saved;
        }
    }
}

/// Revert an evicted block to a pristine "never annotated" state by
/// clearing the annotation on every Variable it DEFINES (inputargs +
/// operation results).  Both dual-gate eviction paths
/// ([`AddedBlocksGuard::drop`] and
/// [`RPythonAnnotator::raise_if_subject_blocked`]) must pair every
/// `annotated.shift_remove` with this reset: dropping a block from
/// `annotated` while its Variables retain annotations leaves a
/// half-state where a later subject reaching the same (session-shared)
/// block sees `seen_before == false` in `addpendingblock`, takes the
/// `bindinputargs` initial-seed path, and calls `setbinding` with the
/// caller's (possibly narrower) cell over the retained annotation —
/// tripping the monotonicity assert `setbinding: new value does not
/// contain old` (e.g. two callers of one helper passing different
/// constant static addresses).
///
/// Best-effort under `try_borrow`: a panic that unwinds while a cell
/// is borrowed degrades to a leak rather than an abort-during-unwind
/// double-borrow.
fn clear_block_bindings(block: &BlockRef) {
    if let Ok(blk) = block.try_borrow() {
        let vars = blk
            .inputargs
            .iter()
            .chain(blk.operations.iter().map(|op| &op.result));
        for v in vars {
            if let Hlvalue::Variable(var) = v
                && let Ok(mut slot) = var.annotation.try_borrow_mut()
            {
                *slot = None;
            }
        }
    }
}

struct BlockAnnotationSnapshot {
    block: BlockRef,
    graph: Option<GraphRef>,
    definitions: Vec<(Variable, Option<Rc<SomeValue>>)>,
}

/// RAII guard for `self.added_blocks = {}; ...; finally:
/// self.added_blocks = saved` in `complete_helpers()`.
///
/// In the dual-gate per-subject driver the guard additionally enforces
/// session isolation: a subject that fails (and is Skipped to the legacy
/// walker) must leave the shared annotator exactly as it found it.
/// `annotated_at_entry` snapshots the `annotated` keyset at scope open;
/// unless [`commit`](Self::commit) marks the drive successful, the drop
/// evicts every block this subject added from `annotated` / `all_blocks`
/// / `blocked_blocks` and prunes the matching `genpendingblocks` entries.
/// Without this, a half-annotated leftover poisons every later subject's
/// session-global `specialize_more_blocks` walk: a `None` (False)
/// sentinel crashes `specialize_block` (in `rtyper.rs`), and a processed
/// block carrying an unbound op result raises `KeyError: no binding for
/// arg`.  `raise_if_subject_blocked` only covers the blocked-on-drain
/// path; this covers the `compute_at_fixpoint` / late-step / panic-unwind
/// failure paths that bypass it.
pub(crate) struct AddedBlocksGuard<'a> {
    ann: &'a RPythonAnnotator,
    saved: Option<Option<IndexMap<BlockKey, BlockRef>>>,
    saved_annotation_snapshots: Option<Option<IndexMap<BlockKey, BlockAnnotationSnapshot>>>,
    saved_listitem_journal: Option<Option<ListItemJournal>>,
    saved_class_attr_journal: Option<Option<ClassAttrJournal>>,
    annotated_at_entry: std::collections::HashSet<BlockKey>,
    /// `links_followed` at scope entry. `flowin` records a link only
    /// after the block's ops bind (`annrpython.py` `flowin`: a
    /// `BlockedInference` on `simple_call` returns before
    /// `follow_link`). A failed subject's reflow can follow the link
    /// and then restore the unbound `simple_call` result; without
    /// restoring this map, Phase B `bindingrepr`s that result.
    links_followed_at_entry: IndexMap<LinkKey, LinkRef>,
    committed: std::cell::Cell<bool>,
}

impl<'a> AddedBlocksGuard<'a> {
    /// Mark this subject's annotator drive successful so its blocks
    /// persist (they are `already_seen` after specialize and may back
    /// later subjects' cross-graph calls). Call right before returning a
    /// successful result; any earlier `?`/panic exit leaves the guard
    /// uncommitted and rolls the subject's blocks back out.
    pub(crate) fn commit(&self) {
        self.committed.set(true);
    }
}

impl<'a> Drop for AddedBlocksGuard<'a> {
    fn drop(&mut self) {
        if self.committed.get() {
            // The inner journal is about to be discarded. Fold the writes
            // it committed into the outer checkpoint first, or the outer
            // scope's uncommitted drop restores the pre-helper item while
            // helper reflow bindings stay at the widened type.
            if let Some(outer_slot) = self.saved_listitem_journal.as_mut()
                && let Some(outer) = outer_slot.as_mut()
                && let Ok(inner_slot) = self.ann.listitem_journal.try_borrow()
                && let Some(inner) = inner_slot.as_ref()
            {
                outer.keep_committed_helper(inner);
            }
            if let Some(outer_slot) = self.saved_class_attr_journal.as_mut()
                && let Some(outer) = outer_slot.as_mut()
                && let Ok(inner_slot) = self.ann.class_attr_journal.try_borrow()
                && let Some(inner) = inner_slot.as_ref()
            {
                outer.keep_committed_helper(inner);
            }
        } else {
            // Roll the shared list item back before restoring op results.
            // A reflow that already ran saved the pre-widen result in the
            // snapshot below; a reflow that was only scheduled is removed
            // from `genpendingblocks` and never applies the widen.
            if let Ok(mut journal) = self.ann.listitem_journal.try_borrow_mut()
                && let Some(journal) = journal.as_mut()
            {
                journal.rollback();
            }
            // `ClassDef.generalize_attr` replaced the slot after
            // `ListItem.merge`. Put the slot back once the item's owner
            // points at the pre-merge list again.
            if let Ok(mut journal) = self.ann.class_attr_journal.try_borrow_mut()
                && let Some(journal) = journal.as_mut()
            {
                journal.rollback();
            }
            // `flowin` does not follow a link out of a `simple_call` that
            // raised `BlockedInference`. A reflow inside this scope may
            // have recorded the link, then the snapshot below puts the
            // result annotation back to unbound. Drop the link record
            // with the annotation or `specialize_block` bindingreprs it.
            *self.ann.links_followed.borrow_mut() = self.links_followed_at_entry.clone();
            // Restore only previously-seen blocks that this subject actually
            // touched. `snapshot_block_if_tracking` records the cells before
            // mergeinputargs/reflow mutates them.
            if let Some(snapshots) = self.ann.subject_annotation_snapshots.borrow().as_ref() {
                for (bkey, saved) in snapshots {
                    self.ann
                        .annotated
                        .borrow_mut()
                        .insert(bkey.clone(), saved.graph.clone());
                    self.ann
                        .all_blocks
                        .borrow_mut()
                        .insert(bkey.clone(), saved.block.clone());
                    self.ann.blocked_blocks.borrow_mut().shift_remove(bkey);
                    for (var, annotation) in &saved.definitions {
                        *var.annotation.borrow_mut() = annotation.clone();
                    }
                }
                if !snapshots.is_empty() {
                    for generation in self.ann.genpendingblocks.borrow_mut().iter_mut() {
                        generation.retain(|bkey, _| !snapshots.contains_key(bkey));
                    }
                }
            }
            // Best-effort eviction under `try_borrow_mut`: a panic that
            // unwinds while one of these maps is borrowed degrades to a
            // leak rather than an abort-during-unwind double-borrow.
            if let (Ok(mut annotated), Ok(mut all_blocks), Ok(mut blocked_blocks)) = (
                self.ann.annotated.try_borrow_mut(),
                self.ann.all_blocks.try_borrow_mut(),
                self.ann.blocked_blocks.try_borrow_mut(),
            ) {
                let new_keys: Vec<BlockKey> = annotated
                    .keys()
                    .filter(|k| !self.annotated_at_entry.contains(*k))
                    .cloned()
                    .collect();
                let fixed_graphs = self.ann.fixed_graphs.try_borrow();
                for k in &new_keys {
                    // A block of a fixed (already-rtyped) graph is
                    // evicted from the maps like any other — leaving it
                    // in `annotated` would re-feed the session-global
                    // specialize walk with a block whose rtyping
                    // already failed, echoing one failure into every
                    // later subject.  Only its Variable BINDINGS are
                    // kept: the `addpendingblock` fixed-graph arm
                    // validates later callers against them and returns
                    // before the `bindinputargs` re-seed, so the
                    // bindings-without-`annotated`-entry state is
                    // harmless for fixed graphs.
                    let is_fixed = match fixed_graphs.as_deref() {
                        Ok(fixed) => annotated
                            .get(k)
                            .cloned()
                            .flatten()
                            .or_else(|| {
                                blocked_blocks.get(k).map(|(_, g, _)| std::rc::Rc::clone(g))
                            })
                            .is_some_and(|g| fixed.contains_key(&GraphKey::of(&g))),
                        Err(_) => false,
                    };
                    annotated.shift_remove(k);
                    blocked_blocks.shift_remove(k);
                    // See `clear_block_bindings` — every eviction from
                    // `annotated` must also reset the block's defined
                    // Variables (unless the graph is fixed).  Mirrors
                    // the dual-gate Skip arm's graph-wide
                    // `var.annotation = None` reset
                    // (codewriter.rs::dual_gate_type_state).
                    if let Some(block) = all_blocks.shift_remove(k)
                        && !is_fixed
                    {
                        clear_block_bindings(&block);
                    }
                }
                if !new_keys.is_empty()
                    && let Ok(mut pending) = self.ann.genpendingblocks.try_borrow_mut()
                {
                    for g in pending.iter_mut() {
                        g.retain(|k, _| self.annotated_at_entry.contains(k));
                    }
                }
            }
        }
        if let Some(saved) = self.saved.take() {
            *self.ann.added_blocks.borrow_mut() = saved;
        }
        if let Some(saved) = self.saved_annotation_snapshots.take() {
            *self.ann.subject_annotation_snapshots.borrow_mut() = saved;
        }
        if let Some(saved) = self.saved_listitem_journal.take()
            && let Ok(mut journal) = self.ann.listitem_journal.try_borrow_mut()
        {
            *journal = saved;
        }
        if let Some(saved) = self.saved_class_attr_journal.take()
            && let Ok(mut journal) = self.ann.class_attr_journal.try_borrow_mut()
        {
            *journal = saved;
        }
    }
}

impl RPythonAnnotator {
    /// RPython `RPythonAnnotator.__init__(self, translator=None,
    /// policy=None, bookkeeper=None, keepgoing=False)`
    /// (annrpython.py).
    ///
    /// Follows upstream: if `bookkeeper` is `None`, construct a fresh
    /// `Bookkeeper(self)`; if `policy` is `None`, install a default
    /// `AnnotatorPolicy()`. The `TLS.bookkeeper = self.bookkeeper`
    /// assignment at upstream line ~81 is folded into
    /// [`Bookkeeper::enter`] — no-op here because we haven't entered a
    /// reflow frame yet.
    pub fn new(
        translator: Option<TranslationContext>,
        policy: Option<AnnotatorPolicy>,
        bookkeeper: Option<Rc<Bookkeeper>>,
        keepgoing: bool,
    ) -> Rc<Self> {
        Self::new_with_translator_mode(translator.map(Rc::new), policy, bookkeeper, keepgoing)
    }

    /// Shared-identity constructor used by
    /// `TranslationContext.buildannotator()`.
    pub fn new_with_translator(
        translator: Option<Rc<TranslationContext>>,
        policy: Option<AnnotatorPolicy>,
        bookkeeper: Option<Rc<Bookkeeper>>,
        keepgoing: bool,
    ) -> Rc<Self> {
        Self::new_with_translator_mode(translator, policy, bookkeeper, keepgoing)
    }

    fn new_with_translator_mode(
        translator: Option<Rc<TranslationContext>>,
        policy: Option<AnnotatorPolicy>,
        bookkeeper: Option<Rc<Bookkeeper>>,
        keepgoing: bool,
    ) -> Rc<Self> {
        // annrpython.py:30 imports rpython.rtyper.extfuncregistry for
        // side effects before seeding the annotator fields.
        crate::translator::rtyper::extfuncregistry::register_external_functions()
            .expect("register extfuncregistry");

        let translator_was_provided = translator.is_some();
        let policy = policy.unwrap_or_default();
        let annotator_policy = PolicyHandle::from(policy.clone());
        let bookkeeper =
            bookkeeper.unwrap_or_else(|| Rc::new(Bookkeeper::new_with_policy(policy.clone())));
        // upstream annrpython.py:30-34 — default to a fresh
        // TranslationContext when caller passes None.
        let translator = translator.unwrap_or_else(|| Rc::new(TranslationContext::new()));
        // Upstream `Bookkeeper.__init__(self, annotator)` stores
        // `self.annotator = annotator`. The translator backlink is
        // only installed for the implicit-translator path in
        // annrpython.py.
        let ann = Rc::new_cyclic(|weak: &std::rc::Weak<Self>| {
            bookkeeper.set_annotator(weak.clone());
            RPythonAnnotator {
                translator: Rc::clone(&translator),
                genpendingblocks: RefCell::new(vec![IndexMap::new()]),
                annotated: RefCell::new(IndexMap::new()),
                all_blocks: RefCell::new(IndexMap::new()),
                added_blocks: RefCell::new(None),
                subject_annotation_snapshots: RefCell::new(None),
                listitem_journal: RefCell::new(None),
                class_attr_journal: RefCell::new(None),
                links_followed: RefCell::new(IndexMap::new()),
                notify: RefCell::new(IndexMap::new()),
                fixed_graphs: RefCell::new(IndexMap::new()),
                blocked_blocks: RefCell::new(IndexMap::new()),
                blocked_graphs: RefCell::new(IndexMap::new()),
                frozen: RefCell::new(false),
                policy: RefCell::new(annotator_policy),
                bookkeeper: Rc::clone(&bookkeeper),
                keepgoing,
                failed_blocks: RefCell::new(IndexMap::new()),
                errors: RefCell::new(Vec::new()),
            }
        });
        if !translator_was_provided {
            translator.set_annotator(Rc::downgrade(&ann));
        }
        ann
    }

    /// RPython `annotation(self, arg)` (annrpython.py).
    ///
    /// ```python
    /// def annotation(self, arg):
    ///     "Gives the SomeValue corresponding to the given Variable or Constant."
    ///     if isinstance(arg, Variable):
    ///         return arg.annotation
    ///     elif isinstance(arg, Constant):
    ///         return self.bookkeeper.immutablevalue(arg.value)
    ///     else:
    ///         raise TypeError('Variable or Constant expected, got %r' % (arg,))
    /// ```
    ///
    /// `None` means the variable has not been bound yet (upstream
    /// `arg.annotation` returns `None` on an unseen variable); a
    /// constant whose value cannot be lifted via `immutablevalue`
    /// is an upstream exception — Rust mirrors that by panicking
    /// (callers can always wrap in `std::panic::catch_unwind` when
    /// emulating the upstream `try/except KeyError` style).
    pub fn annotation(&self, arg: &Hlvalue) -> Option<SomeValue> {
        match arg {
            Hlvalue::Variable(v) => v.annotation.borrow().as_ref().map(|rc| (**rc).clone()),
            Hlvalue::Constant(c) => match self.bookkeeper.immutableconstant(c) {
                Ok(sv) => Some(sv),
                Err(e) => panic!(
                    "AnnotatorError: immutablevalue({:?}) failed — {}",
                    c.value, e
                ),
            },
        }
    }

    /// RPython `binding(self, arg)` (annrpython.py).
    ///
    /// ```python
    /// def binding(self, arg):
    ///     s_arg = self.annotation(arg)
    ///     if s_arg is None:
    ///         raise KeyError
    ///     return s_arg
    /// ```
    pub fn binding(&self, arg: &Hlvalue) -> SomeValue {
        // Required substring `"KeyError: no binding for arg"` is
        // preserved verbatim for `cutover.rs`'s
        // `is_known_unported`; the trailing `{arg:?}` adds the
        // failing Hlvalue (typically a Variable name like `v11` or a
        // Constant payload) so per-graph diagnosis can locate the
        // unbound operand without a separate verbose pass.
        self.annotation(arg)
            .unwrap_or_else(|| panic!("KeyError: no binding for arg {arg:?}"))
    }

    /// RPython `typeannotation(self, t)` (annrpython.py).
    pub fn typeannotation(
        &self,
        spec: &super::signature::AnnotationSpec,
    ) -> Result<SomeValue, super::signature::SignatureError> {
        super::signature::annotation(spec, Some(&self.bookkeeper))
    }

    /// RPython `setbinding(self, arg, s_value)` (annrpython.py).
    ///
    /// ```python
    /// def setbinding(self, arg, s_value):
    ///     s_old = arg.annotation
    ///     if s_old is not None:
    ///         if not s_value.contains(s_old):
    ///             log.WARNING(...)
    ///             assert False
    ///     arg.annotation = s_value
    /// ```
    ///
    /// Rust requires `&mut Variable` to mutate `annotation`. The
    /// driver passes owned `Variable` references while processing a
    /// block, so this is called with `&mut v` there.
    pub fn setbinding(&self, arg: &mut Variable, s_value: SomeValue) {
        let stored = {
            let annotation_ref = arg.annotation.borrow();
            if let Some(s_old) = annotation_ref.as_ref()
                && !s_value.contains(s_old)
            {
                // A pyre-only discriminant refinement can name two variant
                // classes that do not contain each other (`Result::Ok`
                // versus `Result<T,E>::Ok`).  Their integer union drops the
                // arm whose payloads do not meet, and that join contains
                // the old binding.  Store the join.  A union that fails, or
                // a join that still does not contain the old value, is the
                // upstream assert.
                let _guard = super::listdef::SideEffectFreeGuard::enter();
                let widened = super::model::union(&s_value, s_old)
                    .ok()
                    .filter(|u| u.contains(s_old));
                drop(_guard);
                if let Some(widened) = widened {
                    widened
                } else {
                    let where_ = self.bookkeeper.current_position_key().map(|pk| {
                        let gname = pk
                            .graph()
                            .map(|g| g.borrow().name.clone())
                            .unwrap_or_default();
                        format!(" graph={gname} op_index={}", pk.op_index)
                    });
                    panic!(
                        "setbinding: new value does not contain old ({:?} ⊄ {:?}) var={}{}",
                        s_value,
                        **s_old,
                        arg.name(),
                        where_.unwrap_or_default()
                    );
                }
            } else {
                s_value
            }
        };
        *arg.annotation.borrow_mut() = Some(Rc::new(stored));
    }

    /// RPython `warning(self, msg, pos=None)` (annrpython.py-...).
    ///
    /// Driver-level logging.
    pub fn warning(&self, msg: &str) {
        // RPython `annrpython.py log.WARNING("%s/ %s" % (pos, msg))`.
        // `log` is `py.log.Producer("annrpython")`, an unconditional
        // diagnostic channel — not the PYPYLOG-gated `debug_print`.
        // Pyre routes through the `AnsiLogger("annrpython")` port so the
        // output shape `[annrpython:WARNING] msg` matches upstream's
        // `_make_method(':WARNING', ...)` emission.
        LOG.WARNING(msg);
    }

    /// Install `self.bookkeeper` into `TLS.bookkeeper` — mirrors
    /// upstream annrpython.py:~81 side-effect so `getbookkeeper()`
    /// works during calls that don't go through
    /// `Bookkeeper::at_position`.
    pub fn install_tls_bookkeeper(&self) {
        TLS.with(|state| state.borrow_mut().bookkeeper = Some(Rc::clone(&self.bookkeeper)));
    }

    // RPython `convenience high-level interface` — diagnostics + validation
    // helpers used by the driver/caller side.

    // RPython `convenience high-level interface` (annrpython.py) —
    // build_types / build_graph_types / annotate_helper entrypoints.

    /// RPython `build_types(self, function, input_arg_types,
    /// complete_now=True, main_entry_point=False)` (annrpython.py).
    ///
    /// ```python
    /// def build_types(self, function, input_arg_types, complete_now=True,
    ///                 main_entry_point=False):
    ///     """Recursively build annotations about the specific entry point."""
    ///     assert isinstance(function, types.FunctionType), "fix that!"
    ///
    ///     from rpython.annotator.policy import AnnotatorPolicy
    ///     policy = AnnotatorPolicy()
    ///     # make input arguments and set their type
    ///     args_s = [self.typeannotation(t) for t in input_arg_types]
    ///
    ///     # XXX hack
    ///     annmodel.TLS.check_str_without_nul = (
    ///         self.translator.config.translation.check_str_without_nul)
    ///
    ///     with self.using_policy(policy):
    ///         flowgraph, inputs_s = self.get_call_parameters(function, args_s)
    ///
    ///     if main_entry_point:
    ///         self.translator.entry_point_graph = flowgraph
    ///     return self.build_graph_types(flowgraph, inputs_s, complete_now=complete_now)
    /// ```
    ///
    /// Rust uses `HostObject::is_user_function()` for the same
    /// precondition, but preserves the upstream failure mode by
    /// panicking with the same `"fix that!"` message when the caller
    /// passes a non-function.
    pub fn build_types(
        self: &Rc<Self>,
        function: &crate::flowspace::model::HostObject,
        input_arg_types: &[super::signature::AnnotationSpec],
        complete_now: bool,
        main_entry_point: bool,
    ) -> Result<Option<SomeValue>, super::model::AnnotatorError> {
        // upstream: `assert isinstance(function, types.FunctionType), "fix that!"`.
        assert!(function.is_user_function(), "fix that!");

        // upstream: `from rpython.annotator.policy import AnnotatorPolicy`
        //           `policy = AnnotatorPolicy()`.
        let policy = PolicyHandle::from(AnnotatorPolicy::new());

        // upstream: `args_s = [self.typeannotation(t) for t in input_arg_types]`.
        let mut args_s: Vec<SomeValue> = Vec::with_capacity(input_arg_types.len());
        for spec in input_arg_types {
            let sv = self
                .typeannotation(spec)
                .map_err(|e| -> super::model::AnnotatorError { e.into() })?;
            args_s.push(sv);
        }

        // upstream: `# XXX hack`
        //           `annmodel.TLS.check_str_without_nul = (
        //               self.translator.config.translation.check_str_without_nul)`.
        let check_str_without_nul = self.translator.config.translation.check_str_without_nul;
        TLS.with(|state| state.borrow_mut().check_str_without_nul = check_str_without_nul);

        // upstream: `with self.using_policy(policy):
        //               flowgraph, inputs_s = self.get_call_parameters(function, args_s)`.
        let (flowgraph, inputs_s) = {
            let _guard = self.using_policy(policy);
            self.get_call_parameters(function, args_s)?
        };

        // upstream: `if main_entry_point: self.translator.entry_point_graph = flowgraph`.
        if main_entry_point {
            *self.translator.entry_point_graph.borrow_mut() = Some(flowgraph.graph.clone());
        }

        // upstream: `return self.build_graph_types(flowgraph, inputs_s,
        //                                         complete_now=complete_now)`.
        // get_call_parameters now returns Vec<Option<SomeValue>> via
        // the Option-aware specialize chain (description.py
        // pycall propagation parity).
        self.build_graph_types(&flowgraph.graph, &inputs_s, complete_now)
    }

    /// RPython `get_call_parameters(self, function, args_s)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def get_call_parameters(self, function, args_s):
    ///     with self.bookkeeper.at_position(None):
    ///         desc = self.bookkeeper.getdesc(function)
    ///         return desc.get_call_parameters(args_s)
    /// ```
    pub fn get_call_parameters(
        self: &Rc<Self>,
        function: &crate::flowspace::model::HostObject,
        args_s: Vec<SomeValue>,
    ) -> Result<
        (
            Rc<crate::flowspace::pygraph::PyGraph>,
            Vec<Option<SomeValue>>,
        ),
        super::model::AnnotatorError,
    > {
        // upstream: `with self.bookkeeper.at_position(None):`.
        let _guard = self.bookkeeper.at_position(None);
        // upstream: `desc = self.bookkeeper.getdesc(function)`.
        let desc = self.bookkeeper.getdesc(function)?;
        // upstream: `return desc.get_call_parameters(args_s)`.
        desc.get_call_parameters(args_s)
    }

    /// RPython `annotate_helper(self, function, args_s, policy=None)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def annotate_helper(self, function, args_s, policy=None):
    ///     if policy is None:
    ///         from rpython.annotator.policy import AnnotatorPolicy
    ///         policy = AnnotatorPolicy()
    ///         # XXX hack
    ///         annmodel.TLS.check_str_without_nul = (
    ///             self.translator.config.translation.check_str_without_nul)
    ///     with self.using_policy(policy):
    ///         graph, inputcells = self.get_call_parameters(function, args_s)
    ///         self.build_graph_types(graph, inputcells, complete_now=False)
    ///         self.complete_helpers()
    ///     return graph
    /// ```
    ///
    /// Used by rtyper-time helper injection (e.g. `annlowlevel.py`);
    /// the `None → default policy + TLS-hack` branch matches upstream's
    /// short-circuit for un-configured callers.
    pub fn annotate_helper(
        self: &Rc<Self>,
        function: &crate::flowspace::model::HostObject,
        args_s: Vec<SomeValue>,
        policy: Option<PolicyHandle>,
    ) -> Result<Rc<crate::flowspace::pygraph::PyGraph>, super::model::AnnotatorError> {
        // upstream: `if policy is None: ...`.
        let policy = match policy {
            Some(p) => p,
            None => {
                let p = PolicyHandle::from(AnnotatorPolicy::new());
                // upstream: `annmodel.TLS.check_str_without_nul = (
                //     self.translator.config.translation.check_str_without_nul)`.
                let check_str_without_nul =
                    self.translator.config.translation.check_str_without_nul;
                TLS.with(|state| state.borrow_mut().check_str_without_nul = check_str_without_nul);
                p
            }
        };
        // upstream: `with self.using_policy(policy): ...`.
        let _guard = self.using_policy(policy);
        // upstream: `graph, inputcells = self.get_call_parameters(function, args_s)`.
        let (graph, inputcells) = self.get_call_parameters(function, args_s)?;
        // upstream: `self.build_graph_types(graph, inputcells, complete_now=False)`.
        self.build_graph_types(&graph.graph, &inputcells, false)?;
        // upstream: `self.complete_helpers()`.
        self.complete_helpers()?;
        // upstream: `return graph`.
        Ok(graph)
    }

    /// RPython `build_graph_types(self, flowgraph, inputcells,
    /// complete_now=True)` (annrpython.py).
    ///
    /// ```python
    /// def build_graph_types(self, flowgraph, inputcells, complete_now=True):
    ///     checkgraph(flowgraph)
    ///     nbarg = len(flowgraph.getargs())
    ///     assert len(inputcells) == nbarg
    ///     self.addpendinggraph(flowgraph, inputcells)
    ///     if complete_now:
    ///         self.complete()
    ///     return self.annotation(flowgraph.getreturnvar())
    /// ```
    pub fn build_graph_types(
        &self,
        graph: &GraphRef,
        inputcells: &[Option<SomeValue>],
        complete_now: bool,
    ) -> Result<Option<SomeValue>, crate::annotator::model::AnnotatorError> {
        checkgraph(&graph.borrow());
        let nbarg = graph.borrow().getargs().len();
        assert_eq!(
            nbarg,
            inputcells.len(),
            "build_graph_types: wrong number of input cells ({} args vs {} cells)",
            nbarg,
            inputcells.len()
        );
        self.addpendinggraph(graph, inputcells);
        if complete_now {
            self.complete()?;
        }
        let returnvar = graph.borrow().getreturnvar();
        Ok(self.annotation(&returnvar))
    }

    /// RPython `complete_helpers(self)` (annrpython.py).
    ///
    /// Drives `complete()` inside a scoped `added_blocks` tracker so
    /// the caller can learn which blocks were added for the helper.
    ///
    /// ```python
    /// def complete_helpers(self):
    ///     saved = self.added_blocks
    ///     self.added_blocks = {}
    ///     try:
    ///         self.complete()
    ///         self.simplify(block_subset=self.added_blocks)
    ///     finally:
    ///         self.added_blocks = saved
    /// ```
    pub fn complete_helpers(&self) -> Result<(), crate::annotator::model::AnnotatorError> {
        let saved = self.added_blocks.borrow_mut().replace(IndexMap::new());
        let saved_annotation_snapshots = self
            .subject_annotation_snapshots
            .borrow_mut()
            .replace(IndexMap::new());
        // Helper mutations commit with this drive. Swap the journal out
        // so they are not charged to an outer subject scope, whose
        // uncommitted drop would otherwise undo them.
        let saved_listitem_journal = self
            .listitem_journal
            .borrow_mut()
            .replace(ListItemJournal::new());
        let saved_class_attr_journal = self
            .class_attr_journal
            .borrow_mut()
            .replace(ClassAttrJournal::new());
        // `complete_helpers` is the RPython helper-graph drive: its
        // `finally` only restores `added_blocks` (annrpython.py),
        // never evicting blocks.  Pre-commit so the guard's
        // subject-isolation eviction (used by the dual-gate driver) stays
        // inert here.
        let _guard = AddedBlocksGuard {
            ann: self,
            saved: Some(saved),
            saved_annotation_snapshots: Some(saved_annotation_snapshots),
            saved_listitem_journal: Some(saved_listitem_journal),
            saved_class_attr_journal: Some(saved_class_attr_journal),
            annotated_at_entry: std::collections::HashSet::new(),
            links_followed_at_entry: self.links_followed.borrow().clone(),
            committed: std::cell::Cell::new(true),
        };
        self.complete()?;
        // upstream annrpython.py passes `block_subset=self.added_blocks`
        // unconditionally (the dict we seeded at line 117), so simplify
        // only touches graphs reachable from blocks added during this
        // `complete()`. Passing `Some(&[])` (rather than `None`) when
        // the set is empty matches upstream's empty-dict semantics —
        // simplify then iterates zero graphs and skips
        // `perform_normalizations`.
        let subset_blocks: Vec<BlockRef> = self
            .added_blocks
            .borrow()
            .as_ref()
            .expect("complete_helpers: added_blocks tracker disappeared")
            .values()
            .cloned()
            .collect();
        self.simplify(Some(subset_blocks.as_slice()), None);
        Ok(())
    }

    /// RPython `using_policy(self, policy)` (annrpython.py).
    ///
    /// A context manager that temporarily swaps `self.policy`. The
    /// Rust port returns a RAII guard that restores the saved policy
    /// on drop, matching the upstream `try ... finally` contract.
    pub(crate) fn using_policy<P>(&self, policy: P) -> PolicyGuard<'_>
    where
        P: Into<PolicyHandle>,
    {
        let saved = std::mem::replace(&mut *self.policy.borrow_mut(), policy.into());
        PolicyGuard {
            ann: self,
            saved: Some(saved),
        }
    }

    // RPython `interface for annotator.bookkeeper` (annrpython.py).

    /// RPython `recursivecall(self, graph, whence, inputcells)`
    /// (annrpython.py).
    ///
    /// Used by `bookkeeper.pbc_call` to bring a callee graph into the
    /// pending queue, wire the call site into `self.notify`, and
    /// return the callee's current return annotation. `whence` is the
    /// caller's `(parent_graph, parent_block, parent_index)` position
    /// — passing `None` marks an entry-point call.
    pub fn recursivecall(
        &self,
        graph: &GraphRef,
        whence: Option<(GraphRef, BlockRef, usize)>,
        inputcells: &[Option<SomeValue>],
    ) -> SomeValue {
        use super::model::s_impossible_value;
        if let Some((parent_graph, parent_block, parent_index)) = whence.clone() {
            // upstream: `self.translator.update_call_graph(parent_graph, graph, tag)`.
            let tag = (BlockKey::of(&parent_block), parent_index);
            self.translator.update_call_graph(&parent_graph, graph, tag);
            // upstream: `returnpositions = self.notify.setdefault(graph.returnblock, set())`
            let returnblock = graph.borrow().returnblock.clone();
            // upstream: `position_key = (parent_graph, parent_block,
            // parent_index)`. Use `from_refs` so the weak Graph/Block
            // refs are retained for downstream consumers (reflowfromposition).
            let pk = super::bookkeeper::PositionKey::from_refs(
                &parent_graph,
                &parent_block,
                parent_index,
            );
            self.notify
                .borrow_mut()
                .entry(BlockKey::of(&returnblock))
                .or_insert_with(|| NotifyEntry {
                    block: Rc::clone(&returnblock),
                    positions: IndexSet::new(),
                })
                .positions
                .insert(pk);
        }

        // upstream: `self.addpendingblock(graph, graph.startblock, inputcells)`.
        let startblock = graph.borrow().startblock.clone();
        self.addpendingblock(graph, &startblock, inputcells);

        // upstream: `v = graph.getreturnvar(); return self.binding(v)` or
        // s_ImpossibleValue on KeyError.
        let v = graph.borrow().getreturnvar();
        self.annotation(&v).unwrap_or_else(s_impossible_value)
    }

    /// RPython `gettype(self, variable)` (annrpython.py).
    ///
    /// ```python
    /// def gettype(self, variable):
    ///     if isinstance(variable, Constant):
    ///         return type(variable.value)
    ///     elif isinstance(variable, Variable):
    ///         s_variable = variable.annotation
    ///         if s_variable:
    ///             return s_variable.knowntype
    ///         else:
    ///             return object
    ///     else:
    ///         raise TypeError(...)
    /// ```
    ///
    /// Rust port surfaces the result as [`super::model::KnownType`];
    /// `KnownType::Object` stands in for upstream's `object` fallback.
    /// Constants route through the bookkeeper's `immutablevalue` to
    /// recover the KnownType that the value would receive post-
    /// annotation.
    pub fn gettype(&self, arg: &Hlvalue) -> super::model::KnownType {
        use super::model::{KnownType, SomeObjectTrait};
        match arg {
            Hlvalue::Variable(v) => match v.annotation.borrow().as_ref() {
                Some(rc) => (**rc).knowntype(),
                None => KnownType::Object,
            },
            Hlvalue::Constant(c) => self
                .bookkeeper
                .immutableconstant(c)
                .map(|sv| sv.knowntype())
                .unwrap_or(KnownType::Object),
        }
    }

    /// RPython `getuserclassdefinitions(self)` (annrpython.py):
    /// `return self.bookkeeper.classdefs`.
    pub fn getuserclassdefinitions(
        &self,
    ) -> Vec<Rc<std::cell::RefCell<super::classdesc::ClassDef>>> {
        self.bookkeeper.classdefs.borrow().clone()
    }

    /// RPython `validate(self)` (annrpython.py):
    /// `self.bookkeeper.check_no_flags_on_instances()`.
    pub fn validate(&self) {
        self.bookkeeper.check_no_flags_on_instances();
    }

    /// Internal helper behind [`Self::call_sites`] that also preserves
    /// the upstream `(graph, block, opindex)` identity as a
    /// [`PositionKey`].
    pub(crate) fn call_sites_with_positions(
        &self,
    ) -> Vec<(
        super::super::flowspace::model::SpaceOperation,
        super::bookkeeper::PositionKey,
    )> {
        let mut out = Vec::new();
        let added = self.added_blocks.borrow();
        let blocks: Vec<(GraphRef, BlockRef)> = match added.as_ref() {
            // upstream: `newblocks = self.added_blocks` (dict/set of Blocks).
            Some(set) => {
                let annotated = self.annotated.borrow();
                set.values()
                    .filter_map(|block| {
                        let graph = annotated
                            .get(&BlockKey::of(block))
                            .and_then(|g| g.clone())?;
                        Some((graph, block.clone()))
                    })
                    .collect()
            }
            // upstream: `newblocks = self.annotated  # all of them`.
            // Iterating a Python dict yields keys (Block objects). The
            // Rust port rebuilds those keys through `all_blocks` and
            // keeps the associated graph alongside them so
            // `PositionKey::from_refs` can reconstruct the real call
            // site identity.
            None => {
                let annotated = self.annotated.borrow();
                let all_blocks = self.all_blocks.borrow();
                annotated
                    .iter()
                    .filter_map(|(block_key, graph)| {
                        let graph = graph.clone()?;
                        let block = all_blocks.get(block_key)?.clone();
                        Some((graph, block))
                    })
                    .collect()
            }
        };
        drop(added);
        for (graph, block) in blocks {
            let blk = block.borrow();
            for (i, op) in blk.operations.iter().enumerate() {
                if op.opname == "simple_call" || op.opname == "call_args" {
                    out.push((
                        op.clone(),
                        super::bookkeeper::PositionKey::from_refs(&graph, &block, i),
                    ));
                }
                // upstream: `if op.result.annotation is None: break`.
                if let super::super::flowspace::model::Hlvalue::Variable(v) = &op.result
                    && v.annotation.borrow().is_none()
                {
                    break;
                }
            }
        }
        out
    }

    /// RPython `call_sites(self)` (annrpython.py).
    ///
    /// Yields every `simple_call` / `call_args` operation over the
    /// driver's tracked blocks — `added_blocks` if set, else
    /// everything in `annotated`. Stops early on a block once an
    /// operation with an unannotated result surfaces (matches the
    /// upstream `break` on a partially-annotated block).
    pub fn call_sites(&self) -> Vec<super::super::flowspace::model::SpaceOperation> {
        self.call_sites_with_positions()
            .into_iter()
            .map(|(op, _)| op)
            .collect()
    }

    /// RPython `simplify(self, block_subset=None, extra_passes=None)`
    /// (annrpython.py).
    ///
    /// Upstream body:
    ///
    /// ```python
    /// transform.transform_graph(self, block_subset=block_subset,
    ///                           extra_passes=extra_passes)
    /// if block_subset is None:
    ///     graphs = self.translator.graphs
    /// else:
    ///     graphs = {}
    ///     for block in block_subset:
    ///         graph = self.annotated.get(block)
    ///         if graph:
    ///             graphs[graph] = True
    /// for graph in graphs:
    ///     simplify.eliminate_empty_blocks(graph)
    /// self.bookkeeper.compute_at_fixpoint()
    /// if block_subset is None:
    ///     perform_normalizations(self)
    /// ```
    ///
    /// The Rust port routes the fixpoint call through the bookkeeper's
    /// `compute_at_fixpoint`, invokes `eliminate_empty_blocks` over the
    /// translator's graph set, and calls the normalizecalls driver on
    /// the full-graph path.
    pub(crate) fn simplify(
        &self,
        block_subset: Option<&[BlockRef]>,
        extra_passes: Option<&[super::super::translator::transform::TransformPass]>,
    ) {
        // upstream: `transform.transform_graph(self, block_subset=..., extra_passes=...)`.
        super::super::translator::transform::transform_graph(self, extra_passes, block_subset);
        // upstream:
        //   if block_subset is None:
        //       graphs = self.translator.graphs
        //   else:
        //       graphs = {}
        //       for block in block_subset:
        //           graph = self.annotated.get(block)
        //           if graph:
        //               graphs[graph] = True
        let graphs: Vec<GraphRef> = match block_subset {
            None => self.translator.graphs.borrow().clone(),
            Some(blocks) => {
                // The ordered container is required because the key hashes
                // on a pointer and the loop over the values produces a work
                // order.
                let mut seen: IndexMap<GraphKey, GraphRef> = IndexMap::new();
                let annotated = self.annotated.borrow();
                for block in blocks {
                    let key = BlockKey::of(block);
                    if let Some(Some(graph)) = annotated.get(&key) {
                        seen.entry(GraphKey::of(graph)).or_insert(graph.clone());
                    }
                }
                seen.into_values().collect()
            }
        };
        // upstream: `for graph in graphs: simplify.eliminate_empty_blocks(graph)`.
        for graph_ref in graphs {
            let fg = graph_ref.borrow();
            super::super::translator::simplify::eliminate_empty_blocks(&fg);
        }
        // upstream: `self.bookkeeper.compute_at_fixpoint()`.
        // Upstream lets the call propagate any raised exception out
        // of `simplify` (see `annrpython.py:simplify` —
        // `bookkeeper.compute_at_fixpoint()` runs without a
        // `try`/`except`).  Pyre's `simplify` returns `()` so we
        // panic on uncaught errors, structurally matching upstream's
        // "uncaught Python exception terminates the annotator" path.
        // Re-shaping `simplify` to `Result<(), AnnotatorError>` is
        // the eventual port; until that lands, panic preserves the
        // upstream "the annotator stops" semantics.
        self.bookkeeper
            .compute_at_fixpoint()
            .unwrap_or_else(|err| std::panic::panic_any(err.in_stage("compute_at_fixpoint")));
        // upstream: `if block_subset is None: perform_normalizations(self)`
        if block_subset.is_none() {
            super::super::translator::rtyper::normalizecalls::perform_normalizations(self)
                .unwrap_or_else(|err| {
                    std::panic::panic_any(err.in_stage("perform_normalizations"))
                });
        }
    }

    /// RPython `apply_renaming(self, s_out, renaming)`
    /// (annrpython.py).
    ///
    /// Rewrites the `is_type_of` list of a `SomeTypeOf` and the
    /// `knowntypedata` table of a `SomeBool` under the provided
    /// variable-renaming. `renaming` maps each old variable to a
    /// (possibly empty) list of new variables — the annotation is
    /// either rebuilt against the renamed variables or stripped if
    /// no renaming entry exists.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn apply_renaming(
        &self,
        s_out: SomeValue,
        renaming: &HashMap<Rc<Variable>, Vec<Rc<Variable>>>,
    ) -> SomeValue {
        use super::model::{
            KnownTypeData, SomeBool, SomeTypeOf, SomeValue as SV, add_knowntypedata, r#typeof,
        };
        // Rebuild SomeTypeOf.is_type_of against the renamed variables.
        // upstream: `for v in s_out.is_type_of: renamed += renaming[v]`.
        let s_out = match s_out {
            SV::TypeOf(t) => {
                let mut renamed_is_type_of: Vec<Rc<Variable>> = Vec::new();
                for v in &t.is_type_of {
                    if let Some(new_vs) = renaming.get(v) {
                        renamed_is_type_of.extend(new_vs.iter().map(Rc::clone));
                    }
                }
                let newcell = r#typeof(&renamed_is_type_of);
                match newcell {
                    SV::TypeOf(mut nc) => {
                        if t.base.const_box.is_some() {
                            nc.base.const_box = t.base.const_box.clone();
                        }
                        SV::TypeOf(nc)
                    }
                    // typeof returns SomeType when args_v is empty.
                    other => other,
                }
            }
            other => other,
        };
        // Rewrite SomeBool.knowntypedata under the renaming.
        if let SV::Bool(b) = &s_out
            && let Some(ktd) = &b.knowntypedata
        {
            let mut renamed: KnownTypeData = HashMap::new();
            for (value, constraints) in ktd {
                let mut new_inner: HashMap<Rc<Variable>, SomeValue> = HashMap::new();
                for (v, s) in constraints {
                    if let Some(new_vs) = renaming.get(v) {
                        for new_v in new_vs {
                            new_inner.insert(Rc::clone(new_v), s.clone());
                        }
                    }
                }
                if !new_inner.is_empty() {
                    renamed.insert(*value, new_inner);
                }
            }
            // upstream: `assert isinstance(s_out, SomeBool)` + rebuild.
            let mut newcell = SomeBool::new();
            // SomeObject.is_immutable_constant (model.py):
            // `self.is_constant() and self.immutable`. SomeBool
            // carries `immutable = True`, so the precondition here
            // is just "const_box populated".
            if b.base.const_box.is_some() {
                newcell.base.const_box = b.base.const_box.clone();
            }
            newcell.set_knowntypedata(renamed);
            // rebuild add_knowntypedata side-effect: not needed —
            // set_knowntypedata already stores the rebuilt map.
            let _ = add_knowntypedata; // keep the import alive.
            let _ = SomeTypeOf::new(vec![]); // keep the import alive.
            return SV::Bool(newcell);
        }
        s_out
    }

    /// RPython `whereami(self, position_key)` (annrpython.py).
    ///
    /// ```python
    /// graph, block, i = position_key
    /// ...
    /// return repr(graph) + blk + opid
    /// ```
    ///
    /// Rust `PositionKey` carries the u64-encoded (graph, block, index)
    /// tuple. For log messages we stringify each component.
    #[cfg(test)]
    pub(crate) fn whereami(&self, pk: PositionKey) -> String {
        let opid = if pk.op_index > 0 {
            format!(" op={}", pk.op_index)
        } else {
            String::new()
        };
        format!("graph={} block={}{}", pk.graph_id, pk.block_id, opid)
    }

    /// RPython `complete(self)` (annrpython.py).
    ///
    /// Drains every pending-block generation until fixpoint, then
    /// validates that every tracked graph has an annotated return
    /// variable. Blocked blocks / failed blocks return an
    /// `AnnotatorError`, matching upstream's `raise`.
    pub fn complete(&self) -> Result<(), crate::annotator::model::AnnotatorError> {
        use super::model::SomeObjectTrait;
        loop {
            self.complete_pending_blocks()?;
            // upstream: `self.policy.no_more_blocks_to_annotate(self)`.
            self.policy.borrow().no_more_blocks_to_annotate(self)?;
            let any_pending = self.genpendingblocks.borrow().iter().any(|d| !d.is_empty());
            if !any_pending {
                break;
            }
        }

        // Collect the set of graphs whose return annotation must be
        // forced after the fixpoint loop.
        let added = self.added_blocks.borrow();
        let (newgraphs, got_blocked) = match added.as_ref() {
            Some(added_set) => {
                // Walk added_set's blocks, resolve annotated[block] →
                // owning graph; treat `None` (= False) entries as
                // "blocked".
                let annotated = self.annotated.borrow();
                // The ordered container is required because the key
                // hashes on a pointer and the loop over the values
                // produces a work order.
                let mut graphs: IndexMap<GraphKey, GraphRef> = IndexMap::new();
                let mut got_blocked = false;
                for bkey in added_set.keys() {
                    match annotated
                        .get(bkey)
                        .expect("complete: added_blocks entry missing from annotated")
                    {
                        Some(graph) => {
                            graphs.insert(GraphKey::of(graph), Rc::clone(graph));
                        }
                        None => got_blocked = true,
                    }
                }
                (graphs.values().cloned().collect::<Vec<_>>(), got_blocked)
            }
            None => {
                let got_blocked = self.annotated.borrow().values().any(|g| g.is_none());
                let graphs: Vec<GraphRef> = self.translator.graphs.borrow().clone();
                (graphs, got_blocked)
            }
        };
        drop(added);

        if !self.failed_blocks.borrow().is_empty() {
            let errors = self.errors.borrow();
            let text = if errors.is_empty() {
                format!("Annotation failed, {} errors were recorded.", errors.len())
            } else {
                format!(
                    "Annotation failed, {} errors were recorded:\n{}",
                    errors.len(),
                    errors.join("\n-----")
                )
            };
            return Err(crate::annotator::model::AnnotatorError::new(text));
        }

        if got_blocked {
            return Err(self.blocked_annotation_error());
        }

        // Force every return-var annotation to exist.
        for graph in newgraphs {
            self.force_return_var_annotation(&graph);
            // Upstream: `v = graph.exceptblock.inputargs[1]; if
            // v.annotation is not None and v.annotation.can_be_none():
            //     raise AnnotatorError(...)`.
            let excvar = {
                let g = graph.borrow();
                let eb = g.exceptblock.borrow();
                eb.inputargs[1].clone()
            };
            if let Hlvalue::Variable(v) = excvar {
                let annotation = v.annotation.borrow();
                if let Some(rc) = annotation.as_ref()
                    && rc.can_be_none()
                {
                    return Err(crate::annotator::model::AnnotatorError::new(format!(
                        "{:?} is found by annotation to possibly raise None, but the None \
                             was not suppressed by the flow space",
                        graph.borrow().name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Bind a graph's return var to `s_ImpossibleValue` when the fixpoint
    /// left it unannotated (annrpython.py:259-261).
    ///
    /// A graph all of whose reachable paths terminate at the exceptblock
    /// leaves the returnblock unreachable, so its return var never earns
    /// a binding. Seeding it to the bottom type lets the rtyper's
    /// `setconcretetype(getreturnvar())` resolve to the void/impossible
    /// repr rather than crash on a missing binding.
    pub(crate) fn force_return_var_annotation(&self, graph: &GraphRef) {
        use super::model::s_impossible_value;
        let returnvar = graph.borrow().getreturnvar();
        if let Hlvalue::Variable(v) = &returnvar
            && v.annotation.borrow().is_none()
        {
            let sv = s_impossible_value();
            let graph_mut = graph.borrow_mut();
            let mut rb = graph_mut.returnblock.borrow_mut();
            if let Hlvalue::Variable(vm) = &mut rb.inputargs[0] {
                self.setbinding(vm, sv);
            }
        }
    }

    /// Run the return-var seeding tail of `complete()`
    /// (annrpython.py) over every annotated graph.
    ///
    /// The two-phase rtyper prepass drives annotation per subject and
    /// defers the once-at-end `complete()`; that deferral drops the
    /// return-var seeding, so an always-raising callee graph reaches
    /// Phase B with an unannotated return var and crashes
    /// `setconcretetype`. Running the seeding explicitly before Phase B
    /// restores the upstream guarantee.
    pub(crate) fn seed_all_annotated_return_vars(&self) {
        let graphs: Vec<GraphRef> = {
            let annotated = self.annotated.borrow();
            // The ordered container is required because the key hashes
            // on a pointer and the loop over the values produces a work
            // order.
            let mut seen: IndexMap<GraphKey, GraphRef> = IndexMap::new();
            for g in annotated.values().flatten() {
                seen.entry(GraphKey::of(g)).or_insert_with(|| Rc::clone(g));
            }
            seen.into_values().collect()
        };
        for graph in &graphs {
            self.force_return_var_annotation(graph);
        }
    }

    /// Construct the orthodox `complete()` blocked-annotation error
    /// (annrpython.py): flip every blocked graph's flag and
    /// render the multi-line "Blocked block" report through
    /// `format_blocked_annotation_error`.
    fn blocked_annotation_error(&self) -> crate::annotator::model::AnnotatorError {
        let mut bg = self.blocked_graphs.borrow_mut();
        for (_k, (_g, flag)) in bg.iter_mut() {
            *flag = true;
        }
        drop(bg);
        let blocked_blocks = self.blocked_blocks.borrow();
        let text = crate::tool::error::format_blocked_annotation_error(self, &blocked_blocks);
        crate::annotator::model::AnnotatorError::new(text)
    }

    /// Open a scoped `added_blocks` tracker around an external drive of
    /// the pending queue, returning the RAII guard that restores the
    /// previous tracker on drop. Mirrors the `saved = self.added_blocks;
    /// self.added_blocks = {}` prologue of `complete_helpers`
    /// (annrpython.py) for callers — e.g. the dual-gate
    /// per-subject driver — that drain via `complete_pending_blocks`
    /// directly rather than through `complete()`.
    fn snapshot_block_if_tracking(&self, bkey: &BlockKey, block: &BlockRef) {
        {
            let snapshots = self.subject_annotation_snapshots.borrow();
            let Some(snapshots) = snapshots.as_ref() else {
                return;
            };
            if snapshots.contains_key(bkey) {
                return;
            }
        }
        let Some(graph) = self.annotated.borrow().get(bkey).cloned() else {
            // A block first discovered in this scope is evicted by the
            // ordinary added-block rollback and has no prior state to save.
            return;
        };
        let definitions = {
            let block = block.borrow();
            block
                .inputargs
                .iter()
                .chain(block.operations.iter().map(|op| &op.result))
                .filter_map(|value| match value {
                    Hlvalue::Variable(var) => Some((var.clone(), var.annotation.borrow().clone())),
                    _ => None,
                })
                .collect()
        };
        self.subject_annotation_snapshots
            .borrow_mut()
            .as_mut()
            .expect("annotation snapshot scope disappeared")
            .insert(
                bkey.clone(),
                BlockAnnotationSnapshot {
                    block: block.clone(),
                    graph,
                    definitions,
                },
            );
    }

    pub(crate) fn enter_added_blocks_scope(&self) -> AddedBlocksGuard<'_> {
        let saved = self.added_blocks.borrow_mut().replace(IndexMap::new());
        let saved_annotation_snapshots = self
            .subject_annotation_snapshots
            .borrow_mut()
            .replace(IndexMap::new());
        let saved_listitem_journal = self
            .listitem_journal
            .borrow_mut()
            .replace(ListItemJournal::new());
        let saved_class_attr_journal = self
            .class_attr_journal
            .borrow_mut()
            .replace(ClassAttrJournal::new());
        let annotated_at_entry: std::collections::HashSet<BlockKey> =
            self.annotated.borrow().keys().cloned().collect();
        let links_followed_at_entry = self.links_followed.borrow().clone();
        AddedBlocksGuard {
            ann: self,
            saved: Some(saved),
            saved_annotation_snapshots: Some(saved_annotation_snapshots),
            saved_listitem_journal: Some(saved_listitem_journal),
            saved_class_attr_journal: Some(saved_class_attr_journal),
            annotated_at_entry,
            links_followed_at_entry,
            committed: std::cell::Cell::new(false),
        }
    }

    pub(crate) fn note_listitem_mutation(&self, li: &Rc<RefCell<ListItem>>) {
        let Ok(mut journal) = self.listitem_journal.try_borrow_mut() else {
            return;
        };
        let Some(journal) = journal.as_mut() else {
            return;
        };
        journal.note_item(li);
    }

    pub(crate) fn note_owner_retarget(&self, owner: &ItemOwner, previous: Rc<RefCell<ListItem>>) {
        let Ok(mut journal) = self.listitem_journal.try_borrow_mut() else {
            return;
        };
        let Some(journal) = journal.as_mut() else {
            return;
        };
        journal.note_owner(owner, previous);
    }

    pub(crate) fn note_class_attr_slots(&self, classdefs: &[Rc<RefCell<ClassDef>>], name: &str) {
        let Ok(mut journal) = self.class_attr_journal.try_borrow_mut() else {
            return;
        };
        let Some(journal) = journal.as_mut() else {
            return;
        };
        for classdef in classdefs {
            journal.note(classdef, name);
        }
    }

    /// Replicate `complete()`'s blocked-annotation guard
    /// (annrpython.py) for callers that drain the pending queue
    /// via `complete_pending_blocks` instead of `complete()`. When any
    /// block added in the current `added_blocks` scope stayed in the
    /// `None` (False) sentinel state — a permanently `BlockedInference`
    /// block — raise the blocked-annotation error.
    ///
    /// Upstream lets the error abort the whole translation; the pyre
    /// dual-gate instead catches it and skips the subject to the legacy
    /// walker, then reuses the session-shared annotator for the next
    /// subject. The subject's blocks are therefore rolled back out of
    /// the shared maps — otherwise the surviving `None` sentinel crashes
    /// a later `specialize_block` (in `rtyper.rs`), and a surviving
    /// `Some` block could insert a link conversion against an evicted
    /// sentinel successor.
    pub(crate) fn raise_if_subject_blocked(
        &self,
    ) -> Result<(), crate::annotator::model::AnnotatorError> {
        let (added_keys, any_blocked): (Vec<BlockKey>, bool) = {
            let added = self.added_blocks.borrow();
            let Some(added_set) = added.as_ref() else {
                return Ok(());
            };
            let annotated = self.annotated.borrow();
            let any_blocked = added_set
                .keys()
                .any(|bkey| matches!(annotated.get(bkey), Some(None)));
            (added_set.keys().cloned().collect(), any_blocked)
        };
        if !any_blocked {
            return Ok(());
        }
        let err = self.blocked_annotation_error();
        let mut annotated = self.annotated.borrow_mut();
        let mut all_blocks = self.all_blocks.borrow_mut();
        let mut blocked_blocks = self.blocked_blocks.borrow_mut();
        let fixed_graphs = self.fixed_graphs.borrow();
        for bkey in &added_keys {
            // Evict from the maps unconditionally — a leftover entry
            // would re-feed the session-global specialize walk with a
            // block whose processing already failed.  A block of a
            // fixed (already-rtyped) graph keeps its Variable BINDINGS
            // though: the `addpendingblock` fixed-graph arm validates
            // later callers against them and returns before the
            // `bindinputargs` re-seed, so the
            // bindings-without-`annotated`-entry state is harmless for
            // fixed graphs (and clearing them would strand every later
            // caller on "fixed graph's inputarg lacks annotation").
            let is_fixed = annotated
                .get(bkey)
                .cloned()
                .flatten()
                .or_else(|| {
                    blocked_blocks
                        .get(bkey)
                        .map(|(_, g, _)| std::rc::Rc::clone(g))
                })
                .is_some_and(|g| fixed_graphs.contains_key(&GraphKey::of(&g)));
            annotated.shift_remove(bkey);
            // See `clear_block_bindings` — evicting from `annotated`
            // without resetting the block's defined Variables leaves a
            // half-state that trips the `setbinding` monotonicity
            // assert when the next subject re-seeds the same
            // session-shared callee block (the `AddedBlocksGuard` drop
            // cannot compensate: by then these keys are already gone
            // from `annotated`, so its `new_keys` sweep never sees
            // them).
            if let Some(block) = all_blocks.shift_remove(bkey)
                && !is_fixed
            {
                clear_block_bindings(&block);
            }
            blocked_blocks.shift_remove(bkey);
        }
        Err(err)
    }

    // RPython `medium-level interface` (annrpython.py) — block
    // scheduling and pending-blocks queue.

    /// RPython `addpendinggraph(self, flowgraph, inputcells)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def addpendinggraph(self, flowgraph, inputcells):
    ///     self.addpendingblock(flowgraph, flowgraph.startblock, inputcells)
    /// ```
    pub fn addpendinggraph(&self, graph: &GraphRef, inputcells: &[Option<SomeValue>]) {
        let startblock = graph.borrow().startblock.clone();
        self.addpendingblock(graph, &startblock, inputcells);
    }

    /// RPython `addpendingblock(self, graph, block, cells)`
    /// (annrpython.py).
    ///
    /// Registers an entry point into `block` with the given input
    /// cells. If `graph` is in `fixed_graphs`, the pass is a
    /// safety-check only: the new annotations must not widen the
    /// existing inputs. Otherwise the block is seeded via
    /// `bindinputargs` (fresh) or `mergeinputargs` (already seen), and
    /// then queued via `schedulependingblock` if the block still needs
    /// flowin work.
    pub fn addpendingblock(&self, graph: &GraphRef, block: &BlockRef, cells: &[Option<SomeValue>]) {
        let gkey = GraphKey::of(graph);
        if self.fixed_graphs.borrow().contains_key(&gkey) {
            // Upstream: safety-check path for graphs that have already
            // been rtyped. The new annotations must fit within the
            // existing ones (`unionof(old, new) == old`). We validate
            // positionally against `block.inputargs`.
            let blk = block.borrow();
            for (a, s_newarg_opt) in blk.inputargs.iter().zip(cells) {
                let Hlvalue::Variable(v) = a else { continue };
                let s_old = v
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|rc| (**rc).clone())
                    .unwrap_or_else(|| {
                        panic!(
                            "addpendingblock: fixed graph's inputarg lacks annotation \
                             (graph {}, inputarg {})",
                            graph.borrow().name,
                            v.name(),
                        )
                    });
                // Upstream `cells` carries SomeObject — None never
                // reaches the fixed-graph safety check.  Panic loud
                // rather than silently ignoring late-stage unbound
                // input.
                let s_newarg = s_newarg_opt.as_ref().unwrap_or_else(|| {
                    panic!(
                        "addpendingblock: fixed graph received None caller cell \
                         for inputarg {}",
                        v.name()
                    )
                });
                let merged = unionof([&s_old, s_newarg]).unwrap_or_else(|_| {
                    panic!(
                        "AnnotatorError: Late-stage annotation is not allowed to modify the \
                         existing annotation for variable {} ({:?})",
                        v.name(),
                        s_old
                    )
                });
                if merged != s_old {
                    panic!(
                        "AnnotatorError: Late-stage annotation is not allowed to modify the \
                         existing annotation for variable {} ({:?})",
                        v.name(),
                        s_old
                    );
                }
            }
            return;
        }

        // Upstream: `assert not self.frozen`.
        assert!(
            !*self.frozen.borrow(),
            "addpendingblock: annotator is frozen"
        );

        let bkey = BlockKey::of(block);
        let seen_before = self.annotated.borrow().contains_key(&bkey);
        if seen_before {
            self.snapshot_block_if_tracking(&bkey, block);
        }
        if !seen_before {
            self.bindinputargs(graph, block, cells);
        } else {
            self.mergeinputargs(graph, block, cells);
        }
        // After binding/merging, `annotated[block]` is either
        // `Some(False)` (needs flowin) or `Some(graph)` (already
        // completed). Upstream: `if not self.annotated[block]:
        // self.schedulependingblock(graph, block)`.
        let needs_flow = matches!(self.annotated.borrow().get(&bkey), Some(None));
        if needs_flow {
            self.schedulependingblock(graph, block);
        }
    }

    /// RPython `schedulependingblock(self, graph, block)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def schedulependingblock(self, graph, block):
    ///     generation = getattr(block, 'generation', 0)
    ///     self.genpendingblocks[generation][block] = graph
    /// ```
    pub fn schedulependingblock(&self, graph: &GraphRef, block: &BlockRef) {
        let generation = block.borrow().generation.unwrap_or(0) as usize;
        let mut pending = self.genpendingblocks.borrow_mut();
        while pending.len() <= generation {
            pending.push(IndexMap::new());
        }
        pending[generation].insert(BlockKey::of(block), (Rc::clone(block), Rc::clone(graph)));
    }

    /// RPython `complete_pending_blocks(self)` (annrpython.py).
    ///
    /// Drains every pending-block generation, bumping each block's
    /// `generation` and calling `processblock(graph, block)`. Because
    /// `processblock` can re-queue blocks into a later generation via
    /// `addpendingblock → schedulependingblock`, the outer loop keeps
    /// scanning until every dict is empty.
    pub fn complete_pending_blocks(&self) -> Result<(), crate::annotator::model::AnnotatorError> {
        loop {
            // Find the first non-empty generation dict.
            let mut generation: usize = 0;
            {
                let pending = self.genpendingblocks.borrow();
                while generation < pending.len() && pending[generation].is_empty() {
                    generation += 1;
                }
                if generation == pending.len() {
                    return Ok(()); // all empty ⇒ done
                }
            }

            // Upstream: `gen += 1; if len(self.genpendingblocks) == gen:
            //             self.genpendingblocks.append({})`.
            let next_generation_u32 = (generation + 1) as u32;
            {
                let mut pending = self.genpendingblocks.borrow_mut();
                if pending.len() == generation + 1 {
                    pending.push(IndexMap::new());
                }
            }

            // Drain the current generation's pending blocks.
            loop {
                let entry = {
                    let mut pending = self.genpendingblocks.borrow_mut();
                    match pending[generation].keys().next().cloned() {
                        Some(k) => pending[generation].shift_remove(&k),
                        None => None,
                    }
                };
                let Some((block, graph)) = entry else { break };
                block.borrow_mut().generation = Some(next_generation_u32);
                self.processblock(&graph, &block)?;
            }
        }
    }

    /// RPython `reflowpendingblock(self, graph, block)`
    /// (annrpython.py).
    ///
    /// Re-queues an already-annotated block for a fresh flowin pass.
    /// Used when a dependency tracked in `notify` fires.
    pub fn reflowpendingblock(&self, graph: &GraphRef, block: &BlockRef) {
        assert!(
            !*self.frozen.borrow(),
            "reflowpendingblock: annotator is frozen"
        );
        assert!(
            !self
                .fixed_graphs
                .borrow()
                .contains_key(&GraphKey::of(graph)),
            "reflowpendingblock: graph is fixed"
        );
        let bkey = BlockKey::of(block);
        self.snapshot_block_if_tracking(&bkey, block);
        self.schedulependingblock(graph, block);
        assert!(
            self.annotated.borrow().contains_key(&bkey),
            "reflowpendingblock: block not yet annotated"
        );
        self.annotated.borrow_mut().insert(bkey.clone(), None);
        self.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), Rc::clone(block));
        self.blocked_blocks
            .borrow_mut()
            .insert(bkey, (Rc::clone(block), Rc::clone(graph), None));
    }

    /// RPython `follow_link(self, graph, link, constraints)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def follow_link(self, graph, link, constraints):
    ///     assert not (isinstance(link.exitcase, (types.ClassType, type)) and
    ///             issubclass(link.exitcase, BaseException))
    ///
    ///     ignore_link = False
    ///     inputs_s = []
    ///     renaming = defaultdict(list)
    ///     for v_out, v_input in zip(link.args, link.target.inputargs):
    ///         renaming[v_out].append(v_input)
    ///
    ///     for v_out in link.args:
    ///         s_out = self.annotation(v_out)
    ///         if v_out in constraints:
    ///             s_constraint = constraints[v_out]
    ///             s_out = pair(s_out, s_constraint).improve()
    ///             # ignore links that try to pass impossible values
    ///             if s_out == s_ImpossibleValue:
    ///                 ignore_link = True
    ///         s_out = self.apply_renaming(s_out, renaming)
    ///         inputs_s.append(s_out)
    ///     if ignore_link:
    ///         return
    ///
    ///     self.links_followed[link] = True
    ///     self.addpendingblock(graph, link.target, inputs_s)
    /// ```
    ///
    /// `constraints` maps variables in `link.args` to the `SomeValue`
    /// they are *known* to carry when the link is taken — the caller
    /// extracts this from the source block's `exitswitch.annotation`
    /// `knowntypedata`.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn follow_link(
        &self,
        graph: &GraphRef,
        link: &LinkRef,
        constraints: &HashMap<Rc<Variable>, SomeValue>,
    ) {
        use super::binaryop::improve;

        let link_borrow = link.borrow();
        let target_rc = link_borrow
            .target
            .clone()
            .expect("follow_link: link.target must be present");

        // upstream assert: exitcase must not be an exception class.
        // (Exception exits go through follow_raise_link instead.) In the
        // Rust port exception exitcases surface as Hlvalue::Constant
        // with a class HostObject; the caller is responsible for not
        // routing those here.

        // Build renaming map: link.args (source) → link.target.inputargs (dest).
        let target_inputargs = target_rc.borrow().inputargs.clone();
        let mut renaming: HashMap<Rc<Variable>, Vec<Rc<Variable>>> = HashMap::new();
        for (v_out_opt, v_input) in link_borrow.args.iter().zip(&target_inputargs) {
            let Some(Hlvalue::Variable(vo)) = v_out_opt else {
                continue;
            };
            let Hlvalue::Variable(vi) = v_input else {
                continue;
            };
            renaming
                .entry(Rc::new(vo.clone()))
                .or_default()
                .push(Rc::new(vi.clone()));
        }

        let mut ignore_link = false;
        let mut inputs_s: Vec<SomeValue> = Vec::new();
        for v_out_opt in &link_borrow.args {
            let Some(v_out) = v_out_opt else {
                // Transient None in link.args (merge-link undefined-local)
                // — upstream treats absence as impossible; seed the
                // target inputarg with the bottom annotation.
                inputs_s.push(SomeValue::Impossible);
                continue;
            };
            // `flowin` returns before `follow_link` when `simple_call`
            // raises `BlockedInference` (`annrpython.py` `flowin`), so
            // the unbound result is never a followed link. Recording it
            // here leaves Phase B to `bindingrepr` a `None` annotation.
            let mut s_out = match self.annotation(v_out) {
                Some(s) => s,
                None => {
                    ignore_link = true;
                    SomeValue::Impossible
                }
            };
            // upstream: `if v_out in constraints: s_out = pair(s_out, s_c).improve()`.
            if let Hlvalue::Variable(v) = v_out {
                let key = Rc::new(v.clone());
                if let Some(s_constraint) = constraints.get(&key) {
                    s_out = improve(&s_out, s_constraint);
                    if matches!(s_out, SomeValue::Impossible) {
                        ignore_link = true;
                    }
                }
            }
            s_out = self.apply_renaming(s_out, &renaming);
            inputs_s.push(s_out);
        }
        if ignore_link {
            return;
        }

        let lkey = LinkKey::of(link);
        drop(link_borrow);
        self.links_followed.borrow_mut().insert(lkey, link.clone());
        // Internal flowin produces concrete SomeValue per link arg (None
        // is upstream's "unannotated caller arg" signal that originates
        // from the call-site interface, not from intra-graph flow).
        let inputs_s_opt: Vec<Option<SomeValue>> = inputs_s.into_iter().map(Some).collect();
        self.addpendingblock(graph, &target_rc, &inputs_s_opt);
    }

    /// RPython `follow_raise_link(self, graph, link, s_last_exc_value)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def follow_raise_link(self, graph, link, s_last_exc_value):
    ///     v_last_exc_type = link.last_exception
    ///     v_last_exc_value = link.last_exc_value
    ///
    ///     assert (isinstance(link.exitcase, (types.ClassType, type)) and
    ///             issubclass(link.exitcase, BaseException))
    ///
    ///     assert v_last_exc_type and v_last_exc_value
    ///
    ///     if isinstance(v_last_exc_value, Variable):
    ///         self.setbinding(v_last_exc_value, s_last_exc_value)
    ///
    ///     if isinstance(v_last_exc_type, Variable):
    ///         self.setbinding(v_last_exc_type, typeof([v_last_exc_value]))
    ///
    ///     inputs_s = []
    ///     renaming = defaultdict(list)
    ///     for v_out, v_input in zip(link.args, link.target.inputargs):
    ///         renaming[v_out].append(v_input)
    ///
    ///     for v_out, v_input in zip(link.args, link.target.inputargs):
    ///         if v_out == v_last_exc_type:
    ///             s_out = typeof(renaming[v_last_exc_value])
    ///             if isinstance(v_last_exc_type, Constant):
    ///                 s_out.const = v_last_exc_type.value
    ///             elif v_last_exc_type.annotation.is_constant():
    ///                 s_out.const = v_last_exc_type.annotation.const
    ///             inputs_s.append(s_out)
    ///         else:
    ///             s_out = self.annotation(v_out)
    ///             s_out = self.apply_renaming(s_out, renaming)
    ///             inputs_s.append(s_out)
    ///
    ///     self.links_followed[link] = True
    ///     self.addpendingblock(graph, link.target, inputs_s)
    /// ```
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn follow_raise_link(&self, graph: &GraphRef, link: &LinkRef, s_last_exc_value: SomeValue) {
        use super::model::r#typeof;

        // Phase 1: mutate link's extravars in place (setbinding). We
        // need &mut access to the Variable slots inside the Link.
        #[expect(
            clippy::type_complexity,
            reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
        )]
        let (target_rc, args_clone, v_last_exc_value_opt, v_last_exc_type_opt): (
            BlockRef,
            Vec<Option<Hlvalue>>,
            Option<Rc<Variable>>,
            Option<Hlvalue>,
        ) = {
            let mut link_mut = link.borrow_mut();
            let target_rc = link_mut
                .target
                .clone()
                .expect("follow_raise_link: link.target must be present");

            // upstream: `assert v_last_exc_type and v_last_exc_value`.
            assert!(
                link_mut.last_exception.is_some(),
                "follow_raise_link: link.last_exception is required"
            );
            assert!(
                link_mut.last_exc_value.is_some(),
                "follow_raise_link: link.last_exc_value is required"
            );

            // if isinstance(v_last_exc_value, Variable): self.setbinding(v_last_exc_value, s_last_exc_value)
            if let Some(Hlvalue::Variable(v)) = link_mut.last_exc_value.as_mut() {
                self.setbinding(v, s_last_exc_value.clone());
            }

            // Capture a snapshot of v_last_exc_value as Rc<Variable>
            // (only the Variable case — the Constant case doesn't enter
            // typeof). typeof needs the updated v (post-setbinding)
            // so we sample after the mutation above.
            let v_last_exc_value_rc: Option<Rc<Variable>> = match link_mut.last_exc_value.as_ref() {
                Some(Hlvalue::Variable(v)) => Some(Rc::new(v.clone())),
                _ => None,
            };

            // if isinstance(v_last_exc_type, Variable): self.setbinding(v_last_exc_type, typeof([v_last_exc_value]))
            if let Some(Hlvalue::Variable(v)) = link_mut.last_exception.as_mut() {
                let args: Vec<Rc<Variable>> = v_last_exc_value_rc
                    .as_ref()
                    .map(|r| vec![Rc::clone(r)])
                    .unwrap_or_default();
                let s_type = r#typeof(&args);
                self.setbinding(v, s_type);
            }

            // Capture the post-mutation last_exception (used for the
            // v_out == v_last_exc_type equality in phase 2).
            let v_last_exc_type_clone = link_mut.last_exception.clone();

            (
                target_rc,
                link_mut.args.clone(),
                v_last_exc_value_rc,
                v_last_exc_type_clone,
            )
        };

        // Phase 2: build renaming + inputs_s, then addpendingblock.
        let target_inputargs = target_rc.borrow().inputargs.clone();
        let mut renaming: HashMap<Rc<Variable>, Vec<Rc<Variable>>> = HashMap::new();
        for (v_out_opt, v_input) in args_clone.iter().zip(&target_inputargs) {
            let Some(Hlvalue::Variable(vo)) = v_out_opt else {
                continue;
            };
            let Hlvalue::Variable(vi) = v_input else {
                continue;
            };
            renaming
                .entry(Rc::new(vo.clone()))
                .or_default()
                .push(Rc::new(vi.clone()));
        }

        // The exc_type identity for the `v_out == v_last_exc_type` test
        // — only meaningful when v_last_exception is a Variable. Rust's
        // Variable PartialEq is id-based, so direct `==` comparison
        // matches upstream's Python object-identity semantics.
        let exc_type_var: Option<Variable> = v_last_exc_type_opt.as_ref().and_then(|h| match h {
            Hlvalue::Variable(v) => Some(v.clone()),
            _ => None,
        });

        let mut inputs_s: Vec<SomeValue> = Vec::new();
        for v_out_opt in &args_clone {
            let Some(v_out) = v_out_opt else {
                inputs_s.push(SomeValue::Impossible);
                continue;
            };

            // upstream: `if v_out == v_last_exc_type:` — id-based
            // Variable equality (see Variable::PartialEq).
            let is_exc_type = match (v_out, exc_type_var.as_ref()) {
                (Hlvalue::Variable(v), Some(et)) => v == et,
                _ => false,
            };

            if is_exc_type {
                // upstream: s_out = typeof(renaming[v_last_exc_value])
                let renamed: Vec<Rc<Variable>> = v_last_exc_value_opt
                    .as_ref()
                    .and_then(|k| renaming.get(k).cloned())
                    .unwrap_or_default();
                let mut s_out = r#typeof(&renamed);

                // upstream: `.const` override — Constant case or
                // constant-annotation case.
                match v_last_exc_type_opt.as_ref() {
                    Some(Hlvalue::Constant(c)) => {
                        // `s_out.const = v_last_exc_type.value` — only
                        // meaningful if s_out is a SomeTypeOf.
                        if let SomeValue::TypeOf(t) = &mut s_out {
                            t.base.const_box = Some(c.clone());
                        }
                    }
                    Some(Hlvalue::Variable(v)) => {
                        // `elif v_last_exc_type.annotation.is_constant(): s_out.const = v_last_exc_type.annotation.const`.
                        let annotation = v.annotation.borrow();
                        if let Some(ann) = annotation.as_ref()
                            && let SomeValue::TypeOf(t_prev) = ann.as_ref()
                            && let Some(c) = t_prev.base.const_box.as_ref()
                            && let SomeValue::TypeOf(t) = &mut s_out
                        {
                            t.base.const_box = Some(c.clone());
                        }
                    }
                    None => {}
                }
                inputs_s.push(s_out);
            } else {
                // Same `flowin` rule as `follow_link`: an unbound arg
                // means this raise-link was not followed.
                // `follow_raise_link` (annrpython.py) uses
                // `annotation(v_out)` for the handler input. The
                // carrier (`OperationError` / `error::PyErrorObject`)
                // subclasses `Exception`; it is never a `W_Root`.
                let Some(s_out) = self.annotation(v_out) else {
                    return;
                };
                let s_out = self.apply_renaming(s_out, &renaming);
                inputs_s.push(s_out);
            }
        }

        let lkey = LinkKey::of(link);
        self.links_followed.borrow_mut().insert(lkey, link.clone());
        let inputs_s_opt: Vec<Option<SomeValue>> = inputs_s.into_iter().map(Some).collect();
        self.addpendingblock(graph, &target_rc, &inputs_s_opt);
    }

    /// RPython `reflowfromposition(self, position_key)`
    /// (annrpython.py).
    ///
    /// ```python
    /// def reflowfromposition(self, position_key):
    ///     graph, block, index = position_key
    ///     self.reflowpendingblock(graph, block)
    /// ```
    ///
    /// `PositionKey` now carries `Weak<FunctionGraph>` + `Weak<Block>`;
    /// if either has been dropped since the position was recorded the
    /// reflow is silently skipped (the block's owner graph is already
    /// gone). The reflow is likewise skipped when the block upgrades but
    /// is no longer in `annotated`: `raise_if_subject_blocked` (q.v.)
    /// evicts a Skipped subject's blocks from `annotated` without pruning
    /// the shared `notify` map, leaving a position whose `Weak<Block>`
    /// still upgrades against a session-retained `Rc`.
    pub(crate) fn reflowfromposition(&self, position_key: &PositionKey) {
        REFLOW_COUNT.fetch_add(1, Ordering::Relaxed);
        // upstream: `graph, block, index = position_key`
        let Some(graph) = position_key.graph() else {
            return;
        };
        let Some(block) = position_key.block() else {
            return;
        };
        // A `notify` position may target a block that left `annotated`
        // via the `raise_if_subject_blocked` rollback. Such a target is
        // no longer scheduled, so skip the reflow — restoring the
        // invariant (annrpython.py:392/418) that a reflow target is
        // present in `annotated`. `reflowpendingblock`'s assert is
        // intentionally left intact for genuine in-session ordering
        // violations.
        if !self.annotated.borrow().contains_key(&BlockKey::of(&block)) {
            return;
        }
        self.reflowpendingblock(&graph, &block);
    }

    /// RPython `bindinputargs(self, graph, block, inputcells)`
    /// (annrpython.py).
    ///
    /// Creates the initial bindings for the input args of a block:
    /// every `block.inputargs[i]` gets `setbinding(..., inputcells[i])`.
    /// The block is then marked as `annotated[block] = False`
    /// (= awaiting flowin) and registered in `blocked_blocks`.
    pub fn bindinputargs(
        &self,
        graph: &GraphRef,
        block: &BlockRef,
        inputcells: &[Option<SomeValue>],
    ) {
        {
            let mut blk = block.borrow_mut();
            assert_eq!(
                blk.inputargs.len(),
                inputcells.len(),
                "bindinputargs: inputargs/cells arity mismatch"
            );
            for (a, cell) in blk.inputargs.iter_mut().zip(inputcells.iter()) {
                if let Hlvalue::Variable(v) = a {
                    // `annrpython.py` calls `setbinding(a,
                    // cell)` unconditionally.  When `cell is None`
                    // and `arg.annotation` is already set, upstream
                    // crashes with `AttributeError: 'NoneType' object
                    // has no attribute 'contains'` inside
                    // `setbinding` — Python's lattice contract that
                    // an existing binding can only widen.  Match that
                    // fail-loud: an existing annotation cannot be
                    // bound to None.  When the existing slot is
                    // already None we leave it alone (no-op
                    // `arg.annotation = None`).
                    match cell {
                        Some(cell) => self.setbinding(v, cell.clone()),
                        None => {
                            let annot = v.annotation.borrow();
                            assert!(
                                annot.is_none(),
                                "bindinputargs: None cell overrides existing \
                                 annotation {:?} (annrpython.py:425 contract)",
                                annot.as_ref().map(|rc| (**rc).clone())
                            );
                        }
                    }
                }
            }
        }
        let bkey = BlockKey::of(block);
        self.annotated.borrow_mut().insert(bkey.clone(), None);
        self.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), Rc::clone(block));
        self.blocked_blocks
            .borrow_mut()
            .insert(bkey, (Rc::clone(block), Rc::clone(graph), None));
    }

    /// RPython `mergeinputargs(self, graph, block, inputcells)`
    /// (annrpython.py).
    ///
    /// Widens each of the block's existing input annotations via
    /// `unionof(old, new)`. If any merged cell differs from its old
    /// counterpart, the block is re-seeded via `bindinputargs` so the
    /// widened input triggers a fresh flow pass.
    pub fn mergeinputargs(
        &self,
        graph: &GraphRef,
        block: &BlockRef,
        inputcells: &[Option<SomeValue>],
    ) {
        let blk = block.borrow();
        let oldcells: Vec<SomeValue> = blk
            .inputargs
            .iter()
            .enumerate()
            .map(|(i, a)| {
                self.annotation(a).unwrap_or_else(|| {
                    panic!(
                        "KeyError: no binding for arg [mergeinputargs slot={i}/{n} \
                         block={block_id:?} graph_inputargs_len={ga_len}] {a:?}",
                        n = blk.inputargs.len(),
                        block_id = std::ptr::addr_of!(*blk),
                        ga_len = graph.borrow().getargs().len(),
                    )
                })
            })
            .collect();
        drop(blk);

        // `annrpython.py:432` calls `unionof(c1, c2)` directly —
        // every link.arg's `binding(a)` is concrete by construction
        // (an unbound `Variable.annotation` would raise
        // AttributeError before the unionof runs).  Pyre's
        // `Vec<Option<SomeValue>>` cascade guarantees `Some(_)` from
        // intra-graph emission (`follow_link` lifts unbound Variables
        // to `Impossible` first, then wraps as `Some`); a `None`
        // reaching here means a call-site interface (recursivecall /
        // addpendinggraph) leaked an unannotated arg through —
        // fail-loud matches the upstream AttributeError.
        let unions: Result<Vec<SomeValue>, (usize, UnionError)> = oldcells
            .iter()
            .zip(inputcells.iter())
            .enumerate()
            .map(|(i, (c1, c2))| {
                let s2 = c2.as_ref().unwrap_or_else(|| {
                    panic!(
                        "mergeinputargs: inputcells[{i}] is None — caller leaked \
                         an unannotated arg (annrpython.py:432 unionof would have \
                         raised AttributeError)"
                    )
                });
                unionof([c1, s2]).map_err(|e| (i, e))
            })
            .collect();

        let unions = match unions {
            Ok(u) => u,
            Err((slot, e)) => {
                // One body is every monomorphization. A payload or pointer
                // class that does not meet the class already bound here is
                // a different instantiation, not a generalization. Keep the
                // bound class; replacing it with a classdef-less word fails
                // `setbinding` against a string already stored in the slot.
                let gname = graph.borrow().name.clone();
                let per_type_body = gname.ends_with("gc_alloc_storage_box")
                    || gname == "copy_nonoverlapping"
                    || gname.ends_with("::copy_nonoverlapping");
                if per_type_body {
                    let _ = (slot, e);
                    oldcells.clone()
                } else {
                    // `mergeinputargs` attaches the offending source to the
                    // UnionError before it is recorded or re-raised. `UnionError`
                    // renders only the two annotations, which on its own does not
                    // say which merge produced them — and the merging block is
                    // routinely an inlined callee, so the graph the caller was
                    // annotating is not the graph that failed. Carry the same
                    // `source_lines` context here, plus the input slot the two
                    // annotations belong to.
                    let source = crate::tool::error::source_lines(
                        graph,
                        Some(block),
                        None,
                        None,
                        true,
                        crate::tool::error::SHOW_DEFAULT_LINES_OF_CODE,
                    )
                    .join("\n");
                    // `source_lines1` answers `no source!` for every graph lowered
                    // from LLBC rather than from Python (`graph.source` is absent),
                    // which is all of them here. Quote the block's operations
                    // instead — the same "show the offending block" role upstream's
                    // source-line range plays, rendered the way `gather_error`
                    // already renders an operation (`tool/error.rs`'s `gather_error`).
                    let ops = {
                        let b = block.borrow();
                        if b.operations.is_empty() {
                            "    <no operations>".to_string()
                        } else {
                            b.operations
                                .iter()
                                .map(|op| format!("    {op}"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };
                    let e = format!("{e}\n\n[mergeinputargs slot={slot}]\n{source}\n{ops}");
                    // Upstream keeps going when `self.keepgoing` is set;
                    // otherwise re-raises.
                    if self.keepgoing {
                        self.errors.borrow_mut().push(e);
                        self.failed_blocks
                            .borrow_mut()
                            .insert(BlockKey::of(block), Rc::clone(block));
                        return;
                    }
                    panic!("UnionError in mergeinputargs: {e}");
                }
            }
        };

        if unions != oldcells {
            // mergeinputargs's `unions` are concrete `SomeValue` (no
            // None contribution survives the unionof reduction); wrap
            // back as `Some(_)` for the Option-aware bindinputargs.
            let unions_opt: Vec<Option<SomeValue>> = unions.into_iter().map(Some).collect();
            self.bindinputargs(graph, block, &unions_opt);
        }
    }

    // RPython `flowing annotations in blocks` (annrpython.py).

    /// RPython `consider_op(self, op)` (annrpython.py).
    ///
    /// ```python
    /// def consider_op(self, op):
    ///     for arg in op.args:
    ///         if isinstance(self.annotation(arg), SomeImpossibleValue):
    ///             raise BlockedInference(self, op, -1)
    ///     resultcell = op.consider(self)
    ///     if resultcell is None:
    ///         resultcell = s_ImpossibleValue
    ///     elif resultcell == s_ImpossibleValue:
    ///         raise BlockedInference(self, op, -1)
    ///     assert isinstance(resultcell, annmodel.SomeObject)
    ///     assert isinstance(op.result, Variable)
    ///     self.setbinding(op.result, resultcell)
    /// ```
    ///
    /// Rust returns the result cell (or [`FlowinError`] on block /
    /// harmless swallow / annotator error). The caller owns the block
    /// borrow and is responsible for writing the binding into
    /// `op.result` (which lives behind a `RefCell` on the block).
    fn consider_op(
        &self,
        hlop: &super::super::flowspace::operation::HLOperation,
    ) -> Result<SomeValue, FlowinError> {
        // upstream: any arg whose annotation is SomeImpossibleValue
        // means the op cannot run — block.
        for arg in &hlop.args {
            if matches!(self.annotation(arg), Some(SomeValue::Impossible)) {
                return Err(BlockedInference::new(self, hlop.clone(), None).into());
            }
        }
        // upstream `resultcell = op.consider(self)`; AnnotatorError
        // raised inside get_specialization / registered handlers
        // propagates as `FlowinError::Annotator` via the `?`.
        //
        // upstream annrpython.py:654-657 distinguishes a handler that
        // returns Python `None` (a void op such as `setattr`) from one
        // that returns `s_ImpossibleValue`:
        //   if resultcell is None:                # void → bind impossible, no block
        //       resultcell = s_ImpossibleValue
        //   elif resultcell == s_ImpossibleValue: # the operation cannot succeed
        //       raise BlockedInference(self, op, -1)
        let resultcell = match hlop.consider(self)? {
            None => SomeValue::Impossible,
            Some(SomeValue::Impossible) => {
                return Err(BlockedInference::new(self, hlop.clone(), None).into());
            }
            Some(cell) => cell,
        };
        Ok(resultcell)
    }

    /// RPython `get_exception(self, operation)` (annrpython.py).
    ///
    /// ```python
    /// def get_exception(self, operation):
    ///     """
    ///     Return the annotation for all exceptions that `operation` may raise.
    ///     """
    ///     can_only_throw = operation.get_can_only_throw(self)
    ///     if can_only_throw is None:
    ///         return SomeInstance(self.bookkeeper.getuniqueclassdef(Exception))
    ///     else:
    ///         return self.bookkeeper.new_exception(can_only_throw)
    /// ```
    pub fn get_exception(
        &self,
        hlop: &super::super::flowspace::operation::HLOperation,
    ) -> SomeValue {
        use super::super::flowspace::model::HOST_ENV;
        use super::model::{SomeException, SomeInstance};
        let can_only_throw = hlop.get_can_only_throw(self);
        match can_only_throw {
            // upstream `None` → `SomeInstance(getuniqueclassdef(Exception))`.
            None => {
                let exc_class = HOST_ENV
                    .lookup_exception_class("Exception")
                    .expect("HOST_ENV missing builtin Exception");
                let cdef = self
                    .bookkeeper
                    .getuniqueclassdef(&exc_class)
                    .expect("getuniqueclassdef(Exception) failed");
                SomeValue::Instance(SomeInstance::new(
                    Some(cdef),
                    false,
                    std::collections::BTreeMap::new(),
                ))
            }
            // upstream `self.bookkeeper.new_exception(xs)`.
            Some(xs) => {
                let classes: Vec<_> = xs
                    .iter()
                    .map(|b| {
                        HOST_ENV
                            .lookup_exception_class(b.host_name())
                            .unwrap_or_else(|| panic!("HOST_ENV missing builtin {}", b.host_name()))
                    })
                    .collect();
                let s_exc = self
                    .bookkeeper
                    .new_exception(&classes)
                    .expect("new_exception failed");
                // `SomeException` path — no coercion to SomeInstance;
                // flowin drives intersection/difference on this type.
                let _ = SomeException::new(vec![]); // keep import alive
                SomeValue::Exception(s_exc)
            }
        }
    }

    /// Outcome of the flowin operations-side loop — determines which
    /// exits-side path runs next.
    fn flowin_op_loop(
        &self,
        graph: &GraphRef,
        block: &BlockRef,
    ) -> Result<OpLoopOutcome, FlowinError> {
        let mut i = 0usize;
        // Re-read the length each iteration because upstream's
        // `new_ops = op.transform(self); block.operations[i:i+1] = new_ops`
        // mutates the list under the loop pointer.
        while i < block.borrow().operations.len() {
            // upstream: `with self.bookkeeper.at_position((graph, block, i)):`
            // Upstream's 3-tuple carries the real graph/block refs; use
            // `from_refs` so `PositionKey::graph()` / `block()` can
            // upgrade them back (reflowfromposition consumes these).
            let pk = super::bookkeeper::PositionKey::from_refs(graph, block, i);
            let _pg = self.bookkeeper.at_position(Some(pk));

            // `same_as` — pyre's front-end cast lowering
            // (`front/ast.rs::lower_value_cast`) emits the rtyper
            // renaming op (`rtyper.py:478-481`) directly into the flow
            // graph to carry an identity / source-type-unknown
            // `Expr::Cast`'s target `result_ty` down to
            // `rbuiltin::rtype_same_as`.  RPython's `operation.py` never
            // registers `same_as` (it is rtyper-generated), so
            // `OpKind::from_opname` declines it and the canonical
            // dispatch below would fall into the "Unknown op → skip"
            // branch, leaving the result Variable unbound — the next
            // operation that reads it then raises an unbound-argument
            // `AnnotatorError`.  At the annotation level the cast is a
            // pure identity: the operand's `SomeValue` is unchanged (the
            // lltype retype is applied later by `rtype_same_as`), so bind
            // the result to the operand's binding, matching upstream
            // `setbinding(op.result, binding(op.args[0]))`.
            {
                let is_same_as = {
                    let blk = block.borrow();
                    let sp = &blk.operations[i];
                    sp.opname == "same_as" && sp.args.len() == 1
                };
                if is_same_as {
                    let arg = block.borrow().operations[i].args[0].clone();
                    // `binding` (not `annotation`): an unbound operand
                    // mid-block is upstream's `KeyError` raise
                    // (`binding(op.args[0])`) — completing the op with
                    // the result unbound would only move the same
                    // failure onto the next reader of the result.
                    let s_arg = self.binding(&arg);
                    {
                        let mut blk = block.borrow_mut();
                        if let Hlvalue::Variable(v) = &mut blk.operations[i].result {
                            self.setbinding(v, s_arg);
                        }
                    }
                    i += 1;
                    continue;
                }
            }

            // `uint_mul_high` / `uint_lt` — low-level unsigned ops the
            // front-end (`front::checked_arith_uint`) emits directly for the
            // native unsigned-overflow tests (`checked_mul`/`checked_add`
            // lowering).  RPython's flowspace registers neither (they are
            // rtyper-level ops, not `add_operator` operators), so
            // `OpKind::from_opname` declines them and the canonical dispatch
            // below would skip them, leaving the result Variable unbound — the
            // next reader (`eq(result, 0)`) then raises an unbound-argument
            // `AnnotatorError`.  Bind the result at the annotation level:
            // `uint_mul_high` yields an unsigned integer (the widening
            // product's high word), `uint_lt` a bool (the carry / order
            // predicate).  The lltype retype is applied later at rtype.
            {
                let uint_result = {
                    let blk = block.borrow();
                    let sp = &blk.operations[i];
                    match sp.opname.as_str() {
                        "uint_mul" | "uint_mul_high" if sp.args.len() == 2 => {
                            // `SomeInteger { unsigned: true }`, matching how
                            // `ValueType::Unsigned` shells elsewhere
                            // (`codewriter::annotation_state`).
                            Some(SomeValue::Integer(super::model::SomeInteger::new(
                                false, true,
                            )))
                        }
                        "uint_lt" | "uint_le" | "uint_gt" | "uint_ge" if sp.args.len() == 2 => {
                            Some(SomeValue::Bool(super::model::SomeBool::new()))
                        }
                        // `int_between(n, m, p)` is `n <= m < p`, a bool.
                        // The front emits it for `(lo..=hi).contains`;
                        // flowspace has no operator of that name.
                        "int_between" if sp.args.len() == 3 => {
                            Some(SomeValue::Bool(super::model::SomeBool::new()))
                        }
                        // `llop.raw_load` / `llop.raw_store`.  The adapter
                        // seeds the result with `lltype_to_annotation` of
                        // the item descr (`_LLOpEntry.compute_result_annotation`);
                        // take ownership via `setbinding`.  `raw_store`
                        // returns Void (`s_None`).
                        "raw_load" if sp.args.len() == 2 => match &sp.result {
                            Hlvalue::Variable(v) => {
                                v.annotation.borrow().as_ref().map(|rc| (**rc).clone())
                            }
                            _ => None,
                        },
                        "raw_store" if sp.args.len() == 3 => Some(
                            crate::translator::rtyper::llannotation::lltype_to_annotation(
                                crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
                            ),
                        ),
                        // `jtransform.py` `_int_to_int_cast` emits
                        // `int_signext(v, nbytes)`. The front spells that
                        // leaf `signext`. Flowspace has no operator of that
                        // name, so an unbound result stays `Impossible` /
                        // `Void` and rtype then disagrees with `Signed`.
                        "signext" if sp.args.len() == 2 => Some(SomeValue::Integer(
                            super::model::SomeInteger::new(false, false),
                        )),
                        _ => None,
                    }
                };
                if let Some(s_result) = uint_result {
                    let mut blk = block.borrow_mut();
                    if let Hlvalue::Variable(v) = &mut blk.operations[i].result {
                        self.setbinding(v, s_result);
                    }
                    i += 1;
                    continue;
                }
            }

            // Reify `SpaceOperation` (Block storage) into an
            // `HLOperation` the dispatcher can consume. Upstream
            // `Block.operations` carries HLOperations directly; the
            // Rust port's flowspace stores the lowered SpaceOperation
            // shape, so we rebuild the HLOperation for the dispatcher
            // at each visit.
            let hlop = {
                let blk = block.borrow();
                let sp = &blk.operations[i];
                let Some(kind) =
                    super::super::flowspace::operation::OpKind::from_opname(&sp.opname)
                else {
                    // Unknown op → skip (upstream never reaches this
                    // branch; the RPython flow object space only emits
                    // ops whose opname is registered).
                    i += 1;
                    continue;
                };
                let args = sp.args.clone();
                let result_var = match &sp.result {
                    Hlvalue::Variable(v) => v.clone(),
                    Hlvalue::Constant(_) => {
                        // upstream: `assert isinstance(op.result, Variable)`.
                        panic!(
                            "flowin: op.result must be Variable (got Constant) for {:?}",
                            sp.opname
                        );
                    }
                };
                super::super::flowspace::operation::HLOperation {
                    kind,
                    args,
                    result: result_var,
                    offset: sp.offset,
                }
            };

            // upstream: `new_ops = op.transform(self)` + block rewrite.
            //
            // ```python
            // new_ops = op.transform(self)
            // if new_ops is not None:
            //     block.operations[i:i+1] = new_ops
            //     if not new_ops:
            //         continue
            //     new_ops[-1].result = op.result
            //     op = new_ops[0]
            // ```
            //
            // `HLOperation.transform` currently returns `None` for every
            // op (empty transform registry) so the branch below is dead
            // until optimizer registrations land — but the control-flow
            // mirror is in place so behaviour tracks upstream once they
            // do.
            let hlop = match hlop.transform(self) {
                None => hlop,
                Some(new_ops) => {
                    let op_result = hlop.result.clone();
                    let new_sps: Vec<super::super::flowspace::model::SpaceOperation> = new_ops
                        .iter()
                        .map(|new_hlop| {
                            super::super::flowspace::model::SpaceOperation::with_offset(
                                new_hlop.kind.opname(),
                                new_hlop.args.clone(),
                                Hlvalue::Variable(new_hlop.result.clone()),
                                new_hlop.offset,
                            )
                        })
                        .collect();
                    {
                        let mut blk = block.borrow_mut();
                        blk.operations.splice(i..=i, new_sps);
                    }
                    // `if not new_ops: continue` — empty replacement
                    // consumes the op.
                    if new_ops.is_empty() {
                        continue;
                    }
                    // `new_ops[-1].result = op.result` — only the last
                    // replacement inherits the original result.
                    {
                        let mut blk = block.borrow_mut();
                        let last_idx = i + new_ops.len() - 1;
                        blk.operations[last_idx].result = Hlvalue::Variable(op_result);
                    }
                    // `op = new_ops[0]`.
                    new_ops[0].clone()
                }
            };

            // Hand off to consider_op.
            let result_cell = match self.consider_op(&hlop) {
                Ok(sv) => sv,
                Err(FlowinError::Blocked(mut e)) => {
                    // upstream flowin:504-526 — BlockedInference handling.
                    // Decide whether the blocked op is the block's
                    // raising_op (in which case exits-side proceeds with
                    // only exception exits) or a harmless call-family
                    // op (swallow) or else propagate.
                    let is_raising_op = {
                        let blk = block.borrow();
                        blk.canraise() && i + 1 == blk.operations.len()
                    };
                    if is_raising_op {
                        e.opindex = Some(i);
                        return Ok(OpLoopOutcome::BlockedOnRaisingOp);
                    }
                    match hlop.kind {
                        super::super::flowspace::operation::OpKind::SimpleCall
                        | super::super::flowspace::operation::OpKind::CallArgs
                        | super::super::flowspace::operation::OpKind::Next => {
                            return Ok(OpLoopOutcome::HarmlesslySwallowed);
                        }
                        _ => {
                            e.opindex = Some(i);
                            return Ok(OpLoopOutcome::Blocked(e));
                        }
                    }
                }
                Err(FlowinError::Harmless(_)) => {
                    // upstream: `except annmodel.HarmlesslyBlocked: return`.
                    return Ok(OpLoopOutcome::HarmlesslySwallowed);
                }
                Err(FlowinError::Annotator(mut e)) => {
                    // upstream annrpython.py:531-537:
                    //     except annmodel.AnnotatorError as e:
                    //         e.source = gather_error(self, graph, block, i)
                    //         if self.keepgoing:
                    //             self.errors.append(e)
                    //             self.failed_blocks.add(block)
                    //             return
                    //         raise
                    e.source = Some(gather_error(self, graph, block, i));
                    if self.keepgoing {
                        self.errors.borrow_mut().push(format!("{e}"));
                        self.failed_blocks
                            .borrow_mut()
                            .insert(BlockKey::of(block), Rc::clone(block));
                        return Ok(OpLoopOutcome::HarmlesslySwallowed);
                    }
                    return Err(e.into());
                }
            };

            // upstream: `self.setbinding(op.result, resultcell)`. The
            // Variable we captured above is a snapshot; mutate the
            // live slot inside block.operations[i].result.
            {
                let mut blk = block.borrow_mut();
                if let Hlvalue::Variable(v) = &mut blk.operations[i].result {
                    self.setbinding(v, result_cell);
                }
            }
            i += 1;
        }

        Ok(OpLoopOutcome::Normal)
    }

    /// RPython `flowin(self, graph, block)` (annrpython.py).
    ///
    /// Port structure:
    /// 1. Run the op loop via [`flowin_op_loop`]. Its outcome decides
    ///    which `exits` list the exits-side dispatch uses.
    /// 2. Exits-side dispatch (annrpython.py):
    ///    - `block.canraise` → `get_exception` + per-link intersection /
    ///      difference with `follow_raise_link` for matches and the
    ///      `None` exitcase routed through `follow_link`.
    ///    - Otherwise → knowntypedata-driven `follow_link` per exit.
    /// 3. Notify reflow (annrpython.py): any position subscribed
    ///    to this block's updates is re-queued.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    fn flowin(&self, graph: &GraphRef, block: &BlockRef) -> Result<(), FlowinError> {
        use super::super::flowspace::model::LinkRef;
        use super::model::{
            ExitCaseKey, KnownTypeData, SomeException, SomeInstance, SomeObjectTrait, difference,
            intersection, s_impossible_value,
        };

        // Phase 1 — op loop.
        let outcome = self.flowin_op_loop(graph, block)?;

        // Phase 2 — determine `exits`.
        let exits: Vec<LinkRef> = match outcome {
            OpLoopOutcome::Normal => {
                // upstream else: dead-branch pruning.
                let mut exits = block.borrow().exits.clone();
                // `if isinstance(block.exitswitch, Variable):`
                let exitswitch = block.borrow().exitswitch.clone();
                if let Some(Hlvalue::Variable(_)) = &exitswitch {
                    let s_exitswitch = self.binding(exitswitch.as_ref().unwrap());
                    if s_exitswitch.is_constant() {
                        // upstream: filter by `link.exitcase == s.const`.
                        let s_const = s_exitswitch
                            .const_()
                            .expect("is_constant() but const_() empty")
                            .clone();
                        exits.retain(|link| {
                            let link_b = link.borrow();
                            match &link_b.exitcase {
                                Some(Hlvalue::Constant(c)) => c.value == s_const,
                                Some(Hlvalue::Variable(_)) => false,
                                None => false,
                            }
                        });
                    }
                }
                exits
            }
            OpLoopOutcome::BlockedOnRaisingOp => {
                // upstream: `exits = [link for link in block.exits
                //                     if link.exitcase is not None]`.
                let all = block.borrow().exits.clone();
                all.into_iter()
                    .filter(|l| l.borrow().exitcase.is_some())
                    .collect()
            }
            OpLoopOutcome::HarmlesslySwallowed => {
                // upstream `return` — skip exits & notify reflow.
                return Ok(());
            }
            OpLoopOutcome::Blocked(e) => {
                return Err(e.into());
            }
        };

        // Phase 3 — exits-side dispatch.
        let canraise = block.borrow().canraise();
        if canraise {
            // upstream: `op = block.raising_op; s_exception = self.get_exception(op)`.
            // Rebuild the raising_op HLOperation from the last
            // SpaceOperation.
            let hlop = {
                let blk = block.borrow();
                let raising = blk
                    .raising_op()
                    .expect("canraise block missing raising_op")
                    .clone();
                let Some(kind) =
                    super::super::flowspace::operation::OpKind::from_opname(&raising.opname)
                else {
                    panic!("flowin: raising_op opname unknown: {:?}", raising.opname);
                };
                let result_var = match &raising.result {
                    Hlvalue::Variable(v) => v.clone(),
                    Hlvalue::Constant(_) => panic!("flowin: raising_op result must be Variable"),
                };
                super::super::flowspace::operation::HLOperation {
                    kind,
                    args: raising.args.clone(),
                    result: result_var,
                    offset: raising.offset,
                }
            };
            let mut s_exception = self.get_exception(&hlop);

            for link in &exits {
                let exitcase = link.borrow().exitcase.clone();
                match exitcase {
                    None => {
                        self.follow_link(graph, link, &HashMap::new());
                    }
                    Some(case_hlv) => {
                        // upstream: `if s_exception == s_ImpossibleValue: break`.
                        if s_exception == s_impossible_value() {
                            break;
                        }
                        // upstream: `s_case = SomeInstance(self.bookkeeper.getuniqueclassdef(case))`.
                        let case_host = match &case_hlv {
                            Hlvalue::Constant(c) => match &c.value {
                                super::super::flowspace::model::ConstValue::HostObject(obj) => {
                                    obj.clone()
                                }
                                other => panic!(
                                    "flowin: canraise exitcase must be a HostObject class \
                                         Constant, got {:?}",
                                    other
                                ),
                            },
                            Hlvalue::Variable(_) => {
                                panic!("flowin: canraise link.exitcase must be a Constant")
                            }
                        };
                        let case_cdef = self
                            .bookkeeper
                            .getuniqueclassdef(&case_host)
                            .expect("getuniqueclassdef for exit case failed");
                        let s_case = SomeValue::Instance(SomeInstance::new(
                            Some(case_cdef),
                            false,
                            std::collections::BTreeMap::new(),
                        ));
                        // upstream: `s_matching_exc = intersection(s_exception, s_case)`.
                        let s_matching_exc = intersection(&s_exception, &s_case);
                        if s_matching_exc != s_impossible_value() {
                            self.follow_raise_link(graph, link, s_matching_exc);
                        }
                        // upstream: `s_exception = difference(s_exception, s_case)`.
                        s_exception = difference(&s_exception, &s_case);
                    }
                }
            }
            // keep imports alive (SomeException used indirectly in
            // get_exception).
            let _ = SomeException::new(vec![]);
        } else {
            // upstream: knowntypedata-driven constraints per exit.
            let exitswitch = block.borrow().exitswitch.clone();
            // `getattr(block.exitswitch.annotation, "knowntypedata", {})`
            // (annrpython.py) — read generically off the exitswitch
            // binding; SomeBool and an integer-discriminant SomeInteger
            // both carry it, every other annotation defaults to `{}`.
            // The binding's TYPE also fixes the exitcase-key flavour: a
            // bool exitswitch keys by truth, an integer `SwitchInt`
            // (enum discriminant) keys by the case constant.
            let (knowntypedata, int_keyed): (KnownTypeData, bool) = match exitswitch {
                Some(Hlvalue::Variable(ref v)) => {
                    let ann = v.annotation.borrow();
                    let rc = ann.as_ref();
                    let ktd = rc
                        .and_then(|rc| rc.knowntypedata().cloned())
                        .unwrap_or_default();
                    let int_keyed = matches!(rc.map(|r| r.as_ref()), Some(SomeValue::Integer(_)));
                    (ktd, int_keyed)
                }
                _ => (HashMap::new(), false),
            };
            for link in &exits {
                let exitcase = link.borrow().exitcase.clone();
                // `knowntypedata.get(link.exitcase, {})` — key by the
                // exitcase value. A bool exitswitch keys `True`/`False`;
                // pyre's MIR also encodes a bool `SwitchInt` arm as
                // `Int(0/1)` (in `front/mir.rs`), so a bool binding collapses
                // the integer back to a truth value. An integer
                // discriminant keys directly by the `Int` case constant.
                let case_key: Option<ExitCaseKey> = match exitcase {
                    Some(Hlvalue::Constant(c)) => match &c.value {
                        super::super::flowspace::model::ConstValue::Bool(b) => {
                            Some(ExitCaseKey::Bool(*b))
                        }
                        super::super::flowspace::model::ConstValue::Int(i) if int_keyed => {
                            Some(ExitCaseKey::Int(*i))
                        }
                        super::super::flowspace::model::ConstValue::Int(i) => {
                            Some(ExitCaseKey::Bool(*i != 0))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let constraints = case_key
                    .and_then(|k| knowntypedata.get(&k).cloned())
                    .unwrap_or_default();
                self.follow_link(graph, link, &constraints);
            }
        }

        // Phase 4 — notify reflow.
        // upstream: `if block in self.notify: for position in
        // self.notify[block]: self.reflowfromposition(position)`.
        let positions: Vec<PositionKey> = {
            let bkey = BlockKey::of(block);
            let notify = self.notify.borrow();
            match notify.get(&bkey) {
                Some(entry) => {
                    debug_assert_eq!(BlockKey::of(&entry.block), bkey);
                    if crate::determinism_trace_enabled()
                        && crate::flowspace::model::block_address_was_reused(block)
                    {
                        NOTIFY_HIT_ON_REUSED.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[DTRACE-REUSE] notify hit on recycled block addr={} npos={}",
                            bkey.as_usize(),
                            entry.positions.len()
                        );
                    }
                    entry.positions.iter().cloned().collect()
                }
                None => Vec::new(),
            }
        };
        for position in positions {
            // upstream: `self.reflowfromposition(position)`
            REFLOW_FROM_NOTIFY.fetch_add(1, Ordering::Relaxed);
            self.reflowfromposition(&position);
        }

        Ok(())
    }

    /// RPython `processblock(self, graph, block)` (annrpython.py).
    ///
    /// ```python
    /// self.annotated[block] = graph
    /// if block in self.failed_blocks:
    ///     return
    /// if block in self.blocked_blocks:
    ///     del self.blocked_blocks[block]
    /// try:
    ///     self.flowin(graph, block)
    /// except BlockedInference as e:
    ///     self.annotated[block] = False
    ///     self.blocked_blocks[block] = (graph, e.opindex)
    /// except Exception as e:
    ///     ...
    ///     raise
    /// if self.added_blocks is not None:
    ///     self.added_blocks[block] = True
    /// ```
    ///
    /// The body runs `flowin(graph, block)` and performs the
    /// bookkeeping around it (annotated flip, blocked cleanup,
    /// added_blocks tracking). `BlockedInference` raised by `flowin`
    /// reflects the block back into `blocked_blocks` for a later retry.
    pub fn processblock(
        &self,
        graph: &GraphRef,
        block: &BlockRef,
    ) -> Result<(), crate::annotator::model::AnnotatorError> {
        PROCESSBLOCK_COUNT.fetch_add(1, Ordering::Relaxed);
        let bkey = BlockKey::of(block);
        // upstream: `self.annotated[block] = graph`.
        self.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(Rc::clone(graph)));
        self.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), Rc::clone(block));
        // upstream: `if block in self.failed_blocks: return`.
        if self.failed_blocks.borrow().contains_key(&bkey) {
            return Ok(());
        }
        // upstream: `if block in self.blocked_blocks: del ...`.
        self.blocked_blocks.borrow_mut().shift_remove(&bkey);

        // upstream annrpython.py:397-406:
        //     try:
        //         self.flowin(graph, block)
        //     except BlockedInference as e:
        //         self.annotated[block] = False
        //         self.blocked_blocks[block] = (graph, e.opindex)
        //     except Exception as e:
        //         # hack for debug tools only
        //         if not hasattr(e, '__annotator_block'):
        //             setattr(e, '__annotator_block', block)
        //         raise
        match self.flowin(graph, block) {
            Ok(()) => {}
            Err(FlowinError::Blocked(e)) => {
                self.annotated.borrow_mut().insert(bkey.clone(), None);
                self.blocked_blocks.borrow_mut().insert(
                    bkey.clone(),
                    (Rc::clone(block), Rc::clone(graph), e.opindex),
                );
            }
            Err(FlowinError::Annotator(mut e)) => {
                e.set_annotator_block(block);
                return Err(e);
            }
            Err(FlowinError::Harmless(_)) => {
                unreachable!("flowin() should swallow HarmlesslyBlocked before processblock()")
            }
        }

        // upstream: `if self.added_blocks is not None:
        //              self.added_blocks[block] = True`.
        if let Some(added) = self.added_blocks.borrow_mut().as_mut() {
            added.insert(bkey, Rc::clone(block));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::model::{Block, FunctionGraph};
    use super::super::model::SomeInteger;
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[test]
    fn failed_subject_restores_a_shared_listitem_merge() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = Rc::clone(&ann.bookkeeper);
        let a = crate::annotator::listdef::ListDef::new(
            Some(Rc::clone(&bk)),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = crate::annotator::listdef::ListDef::new(
            Some(bk),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let before = a.listitem_rc().borrow().s_value.clone();
        {
            let _guard = ann.enter_added_blocks_scope();
            a.union_with(&b).expect("merge");
            assert_ne!(a.listitem_rc().borrow().s_value, before);
        }
        assert_eq!(a.listitem_rc().borrow().s_value, before);
    }

    #[test]
    fn failed_subject_restores_rdict_callbacks() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let dd = crate::annotator::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::default()),
            true,
            false,
            false,
        );
        let key = dd.dictkey_rc();
        let before = key.borrow().s_rdict_eqfn.clone();
        {
            let _guard = ann.enter_added_blocks_scope();
            let err = crate::annotator::dictdef::DictKey::update_rdict_annotations(
                &key,
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect_err("non-pbc callbacks fail emulate");
            assert!(err.msg.contains("emulate_pbc_call"));
            assert_ne!(key.borrow().s_rdict_eqfn, before);
        }
        assert_eq!(key.borrow().s_rdict_eqfn, before);
        assert_eq!(key.borrow().s_rdict_hashfn, SomeValue::Impossible);
    }

    fn mk_graph(name: &str, n_args: usize) -> GraphRef {
        let inputs: Vec<Hlvalue> = (0..n_args)
            .map(|i| Hlvalue::Variable(Variable::named(format!("a{i}"))))
            .collect();
        let startblock = Block::shared(inputs);
        Rc::new(RefCell::new(FunctionGraph::new(name, startblock)))
    }

    #[test]
    fn addpendingblock_seeds_annotated_and_queues_block() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("f", 1);
        let startblock = graph.borrow().startblock.clone();
        let s_int = SomeValue::Integer(SomeInteger::default());
        ann.addpendinggraph(&graph, &[Some(s_int.clone())]);

        // bindinputargs side effect: block appears in `annotated` as
        // `Some(None)` (= False, awaiting flowin).
        let bkey = BlockKey::of(&startblock);
        let annmap = ann.annotated.borrow();
        assert!(annmap.contains_key(&bkey));
        assert!(matches!(annmap.get(&bkey), Some(None)));
        drop(annmap);

        // schedulependingblock side effect: generation-0 dict now
        // carries (block, graph).
        let pending = ann.genpendingblocks.borrow();
        assert_eq!(pending[0].len(), 1);
        assert!(pending[0].contains_key(&bkey));

        // setbinding side effect: the startblock inputarg now has the
        // SomeInteger annotation.
        let bound = {
            let blk = startblock.borrow();
            match &blk.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.borrow().as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        assert_eq!(bound, Some(s_int));
    }

    #[test]
    fn mergeinputargs_rebinds_when_union_widens() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("g", 1);
        let startblock = graph.borrow().startblock.clone();

        ann.addpendinggraph(&graph, &[Some(SomeValue::Integer(SomeInteger::default()))]);
        // Re-seed with SomeFloat — unionof(Int, Float) = Float, which
        // differs from the old cell, so bindinputargs must re-run.
        ann.addpendingblock(
            &graph,
            &startblock,
            &[Some(
                SomeValue::Float(super::super::model::SomeFloat::new()),
            )],
        );

        let bound = {
            let blk = startblock.borrow();
            match &blk.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.borrow().as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        assert!(matches!(bound, Some(SomeValue::Float(_))));
    }

    #[test]
    fn setbinding_widens_incompatible_discriminant_arms() {
        use super::super::model::{ExitCaseKey, SomeInstance, add_knowntypedata};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let left = bk
            .getuniqueclassdef(&bk.intern_class_by_qualname("Left"))
            .expect("Left");
        let right = bk
            .getuniqueclassdef(&bk.intern_class_by_qualname("Right"))
            .expect("Right");
        let recv = Rc::new(Variable::named("recv"));
        let arm = |classdef| {
            let mut ktd = super::super::model::KnownTypeData::new();
            add_knowntypedata(
                &mut ktd,
                ExitCaseKey::Int(0),
                std::slice::from_ref(&recv),
                SomeValue::Instance(SomeInstance::new(
                    Some(classdef),
                    false,
                    std::collections::BTreeMap::new(),
                )),
            );
            let mut si = SomeInteger::new(true, false);
            si.set_knowntypedata(ktd);
            SomeValue::Integer(si)
        };
        let mut v = Variable::named("disc");
        ann.setbinding(&mut v, arm(left));
        ann.setbinding(&mut v, arm(right));
        let bound = v.annotation.borrow().as_ref().map(|rc| (**rc).clone());
        match bound {
            Some(SomeValue::Integer(si)) => {
                assert!(
                    si.knowntypedata.is_none(),
                    "incompatible variant arms drop out of the join"
                );
            }
            other => panic!("expected an integer binding, got {other:?}"),
        }
    }

    #[test]
    fn setbinding_keeps_instantiated_variant_over_template() {
        use super::super::model::{ExitCaseKey, SomeInstance, add_knowntypedata};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let template = bk
            .getuniqueclassdef(&bk.intern_class_by_qualname("Result::Ok"))
            .expect("template");
        let inst = bk
            .getuniqueclassdef(
                &bk.intern_class_by_qualname("result::Result<StepResult,PyError>::Ok"),
            )
            .expect("inst");
        let recv = Rc::new(Variable::named("recv"));
        let arm = |classdef| {
            let mut ktd = super::super::model::KnownTypeData::new();
            add_knowntypedata(
                &mut ktd,
                ExitCaseKey::Int(0),
                std::slice::from_ref(&recv),
                SomeValue::Instance(SomeInstance::new(
                    Some(classdef),
                    false,
                    std::collections::BTreeMap::new(),
                )),
            );
            let mut si = SomeInteger::new(true, false);
            si.set_knowntypedata(ktd);
            SomeValue::Integer(si)
        };
        let mut v = Variable::named("disc");
        ann.setbinding(&mut v, arm(inst.clone()));
        ann.setbinding(&mut v, arm(template));
        let name = {
            let bound = v.annotation.borrow();
            let Some(SomeValue::Integer(si)) = bound.as_ref().map(|rc| rc.as_ref()) else {
                panic!("expected integer");
            };
            let ktd = si.knowntypedata.as_ref().expect("refinement kept");
            let inner = ktd.get(&ExitCaseKey::Int(0)).expect("case 0");
            let s = inner.get(&recv).expect("receiver");
            match s {
                SomeValue::Instance(inst) => {
                    inst.classdef.as_ref().expect("class").borrow().name.clone()
                }
                other => panic!("expected instance, got {other:?}"),
            }
        };
        assert!(
            name.contains('<'),
            "join keeps the instantiated variant, got {name}"
        );
    }

    #[test]
    #[should_panic(expected = "setbinding: new value does not contain old")]
    fn setbinding_still_rejects_a_real_type_clash() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let mut v = Variable::named("v");
        ann.setbinding(&mut v, SomeValue::Integer(SomeInteger::default()));
        ann.setbinding(&mut v, super::super::model::s_str0());
    }

    #[test]
    fn gettype_variable_returns_knowntype_or_object() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let mut v = Variable::named("x");
        assert_eq!(
            ann.gettype(&Hlvalue::Variable(v.clone())),
            super::super::model::KnownType::Object
        );
        ann.setbinding(&mut v, SomeValue::Integer(SomeInteger::default()));
        assert_eq!(
            ann.gettype(&Hlvalue::Variable(v)),
            super::super::model::KnownType::Int
        );
    }

    #[test]
    fn validate_calls_bookkeeper_hook() {
        // No-op stub — just verify it's wired up and callable.
        let ann = RPythonAnnotator::new(None, None, None, false);
        ann.validate();
    }

    #[test]
    fn simplify_invokes_bookkeeper_fixpoint() {
        // No-op stub — exercise the call site with the upstream
        // `block_subset=None, extra_passes=None` shape.
        let ann = RPythonAnnotator::new(None, None, None, false);
        ann.simplify(None, None);
    }

    #[test]
    fn whereami_formats_position_key() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let pk = PositionKey::new(42, 17, 3);
        let s = ann.whereami(pk);
        assert!(s.contains("graph=42"));
        assert!(s.contains("block=17"));
        assert!(s.contains("op=3"));
    }

    #[test]
    fn using_policy_restores_on_drop() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        // Baseline: a default policy. Swap in an alternate policy
        // inside a scope; after the scope ends the default is back.
        let alt = AnnotatorPolicy::new();
        {
            let _g = ann.using_policy(alt);
            // no way to compare policies directly — exercise the Drop.
        }
        // Sanity — another policy swap still works after the first
        // guard has been dropped.
        let _g2 = ann.using_policy(AnnotatorPolicy::new());
        drop(_g2);
    }

    #[test]
    fn complete_helpers_restores_added_blocks_on_success() {
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let graph = mk_graph("saved", 0);
        let block = graph.borrow().startblock.clone();
        ann.added_blocks
            .borrow_mut()
            .replace(IndexMap::from([(BlockKey::of(&block), block.clone())]));

        ann.complete_helpers().unwrap();

        let restored = ann.added_blocks.borrow();
        let restored = restored.as_ref().expect("added_blocks should be restored");
        assert!(restored.contains_key(&BlockKey::of(&block)));
    }

    #[test]
    fn complete_helpers_restores_added_blocks_on_panic() {
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let graph = mk_graph("saved", 0);
        let block = graph.borrow().startblock.clone();
        ann.added_blocks
            .borrow_mut()
            .replace(IndexMap::from([(BlockKey::of(&block), block.clone())]));
        ann.bookkeeper
            .pending_specializations
            .borrow_mut()
            .push(Box::new(|| panic!("boom from pending_specializations")));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ann.complete_helpers().unwrap();
        }));
        assert!(result.is_err(), "expected complete_helpers to panic");

        let restored = ann.added_blocks.borrow();
        let restored = restored.as_ref().expect("added_blocks should be restored");
        assert!(restored.contains_key(&BlockKey::of(&block)));
    }

    #[test]
    fn build_graph_types_zero_arg_calls_complete() {
        // Minimum valid graph (mirrors upstream test_mingraph): no
        // inputs, startblock closes directly to returnblock with
        // `Constant(1)`. `complete()` binds `returnvar` to
        // `SomeValue::Integer` — this test just verifies the
        // build_graph_types → checkgraph → complete wiring.
        use super::super::super::flowspace::model::{BlockRefExt, ConstValue, Constant, Link};
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let graph = mk_graph("f0", 0);
        {
            let g = graph.borrow();
            let link = Rc::new(RefCell::new(Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
                Some(g.returnblock.clone()),
                None,
            )));
            g.startblock.closeblock(vec![link]);
        }
        translator.graphs.borrow_mut().push(Rc::clone(&graph));
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let r = ann.build_graph_types(&graph, &[], true).unwrap();
        assert!(matches!(r, Some(SomeValue::Integer(_))), "got {:?}", r);
    }

    #[test]
    fn flowin_empty_block_is_noop() {
        // No operations → flowin returns Ok immediately; processblock
        // flips annotated[block] to Some(graph).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("empty", 0);
        let startblock = graph.borrow().startblock.clone();
        ann.processblock(&graph, &startblock).unwrap();
        assert!(matches!(
            ann.annotated.borrow().get(&BlockKey::of(&startblock)),
            Some(Some(_))
        ));
    }

    #[test]
    fn flowin_consider_op_binds_result() {
        // A single Pos op on a bound Integer should produce a
        // SomeInteger binding on the op.result.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("posgraph", 1);
        let startblock = graph.borrow().startblock.clone();

        // Seed the input variable with SomeInteger.
        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
            // Construct `pos(a)` — result variable captures the op output.
            let arg = blk.inputargs[0].clone();
            let result = Hlvalue::Variable(Variable::named("r"));
            blk.operations
                .push(super::super::super::flowspace::model::SpaceOperation::new(
                    "pos",
                    vec![arg],
                    result,
                ));
        }

        ann.processblock(&graph, &startblock).unwrap();
        // After flowin, block.operations[0].result should carry an
        // Integer binding (SomeInteger.pos on SomeInteger = SomeInteger).
        let blk = startblock.borrow();
        if let Hlvalue::Variable(v) = &blk.operations[0].result {
            let sv = v.annotation.borrow().as_ref().map(|rc| (**rc).clone());
            assert!(matches!(sv, Some(SomeValue::Integer(_))), "got {:?}", sv);
        } else {
            panic!("op.result is not Variable");
        }
        // annotated[block] == Some(graph).
        assert!(matches!(
            ann.annotated.borrow().get(&BlockKey::of(&startblock)),
            Some(Some(_))
        ));
        // Not queued as blocked.
        assert!(
            !ann.blocked_blocks
                .borrow()
                .contains_key(&BlockKey::of(&startblock))
        );
    }

    #[test]
    fn processblock_propagates_annotator_error_when_keepgoing_is_off() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("hash_object", 1);
        let startblock = graph.borrow().startblock.clone();

        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::object());
            }
            let arg = blk.inputargs[0].clone();
            let result = Hlvalue::Variable(Variable::named("r"));
            blk.operations
                .push(super::super::super::flowspace::model::SpaceOperation::new(
                    "hash",
                    vec![arg],
                    result,
                ));
        }

        let err = ann
            .processblock(&graph, &startblock)
            .expect_err("keepgoing=false must propagate AnnotatorError");
        let msg = err.msg.as_deref().unwrap_or("");
        assert!(msg.contains("cannot use hash() in RPython"));
        let source = err.source.as_deref().unwrap_or("");
        assert!(source.contains("hash_object"));
    }

    #[test]
    fn follow_link_seeds_target_block_with_renamed_args() {
        // Build: source block ---link(v0 -> u0)---> target block.
        // v0 bound SomeInteger; after follow_link, u0 (target's
        // inputarg) should carry SomeInteger via addpendingblock.
        use super::super::super::flowspace::model::{Block, Link};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("link_test", 1);
        // Create a target block with one inputarg u0.
        let u0 = Hlvalue::Variable(Variable::named("u0"));
        let target = Block::shared(vec![u0.clone()]);

        // The source block has the existing startblock with inputarg
        // a0 = Variable v0; bind it to SomeInteger.
        let source = graph.borrow().startblock.clone();
        {
            let mut src = source.borrow_mut();
            if let Hlvalue::Variable(v) = &mut src.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
        }
        let source_a0 = source.borrow().inputargs[0].clone();

        // Construct the link: source.exits = [link(args=[a0], target)].
        let link = Rc::new(RefCell::new(Link::new(
            vec![source_a0],
            Some(target.clone()),
            None,
        )));

        // Drive follow_link with empty constraints.
        ann.follow_link(&graph, &link, &HashMap::new());

        // links_followed should record this link.
        assert!(
            ann.links_followed
                .borrow()
                .contains_key(&LinkKey::of(&link))
        );
        // target.inputargs[0].annotation should be SomeInteger.
        let bound = {
            let t = target.borrow();
            match &t.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.borrow().as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        assert!(
            matches!(bound, Some(SomeValue::Integer(_))),
            "got {:?}",
            bound
        );
    }

    #[test]
    fn follow_link_does_not_record_unbound_simple_call_result() {
        // `flowin` (`annrpython.py`) swallows `BlockedInference` on
        // `simple_call` and returns before `follow_link`, so the
        // unbound result is not a followed link. Recording it makes
        // Phase B `bindingrepr` raise `KeyError: no binding for arg`.
        use super::super::super::flowspace::model::{Block, Link};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("unbound_call", 0);
        let unbound = Hlvalue::Variable(Variable::named("call_result"));
        let target = Block::shared(vec![Hlvalue::Variable(Variable::named("u0"))]);
        let link = Rc::new(RefCell::new(Link::new(vec![unbound], Some(target), None)));
        ann.follow_link(&graph, &link, &HashMap::new());
        assert!(
            !ann.links_followed
                .borrow()
                .contains_key(&LinkKey::of(&link)),
            "unbound simple_call result must not be links_followed"
        );
    }

    #[test]
    fn uncommitted_scope_drops_links_followed_by_reflow() {
        // A failed per-subject reflow restores the pre-reflow
        // annotation (still unbound) and must restore `links_followed`
        // with it. `flowin` would not have recorded the link.
        use super::super::super::flowspace::model::{Block, Link};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("reflow_rollback", 1);
        let source = graph.borrow().startblock.clone();
        {
            let mut src = source.borrow_mut();
            if let Hlvalue::Variable(v) = &mut src.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
        }
        let source_a0 = source.borrow().inputargs[0].clone();
        let target = Block::shared(vec![Hlvalue::Variable(Variable::named("u0"))]);
        let link = Rc::new(RefCell::new(Link::new(vec![source_a0], Some(target), None)));
        let scope = ann.enter_added_blocks_scope();
        ann.follow_link(&graph, &link, &HashMap::new());
        assert!(
            ann.links_followed
                .borrow()
                .contains_key(&LinkKey::of(&link))
        );
        drop(scope);
        assert!(
            !ann.links_followed
                .borrow()
                .contains_key(&LinkKey::of(&link)),
            "uncommitted reflow must not leave links_followed set"
        );
    }

    fn list_item_state(listdef: &super::super::listdef::ListDef) -> (bool, bool, bool) {
        let li = listdef.listitem_rc();
        let item = li.borrow();
        let nonneg = match &item.s_value {
            SomeValue::Integer(i) => i.nonneg,
            other => panic!("list item is {other:?}"),
        };
        (nonneg, item.mutated, item.resized)
    }

    fn getitem_result_nonneg(block: &BlockRef) -> bool {
        let blk = block.borrow();
        let Hlvalue::Variable(var) = &blk.operations[0].result else {
            panic!("getitem result is not a variable");
        };
        let bound = var.annotation.borrow();
        let Some(sv) = bound.as_ref() else {
            panic!("getitem result unbound");
        };
        match sv.as_ref() {
            SomeValue::Integer(i) => i.nonneg,
            other => panic!("getitem result is {other:?}"),
        }
    }

    /// Annotate `getitem(list, 0)` outside a scope so the read position
    /// is committed and the result is bound to the nonneg item.
    fn annotate_nonneg_list_getitem(
        ann: &RPythonAnnotator,
        name: &str,
    ) -> (super::super::listdef::ListDef, BlockRef) {
        use super::super::super::flowspace::model::{ConstValue, Constant, SpaceOperation};
        use super::super::listdef::ListDef;
        use super::super::model::SomeList;

        let listdef = ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let graph = mk_graph(name, 1);
        let startblock = graph.borrow().startblock.clone();
        {
            let mut blk = startblock.borrow_mut();
            let arg = blk.inputargs[0].clone();
            blk.operations.push(SpaceOperation::new(
                "getitem",
                vec![arg, Hlvalue::Constant(Constant::new(ConstValue::Int(0)))],
                Hlvalue::Variable(Variable::named("item")),
            ));
        }
        ann.addpendinggraph(
            &graph,
            &[Some(SomeValue::List(SomeList::new(listdef.clone())))],
        );
        ann.complete_pending_blocks().expect("annotate getitem");
        let (nonneg, mutated, resized) = list_item_state(&listdef);
        assert!(nonneg && !mutated && !resized);
        assert!(getitem_result_nonneg(&startblock));
        (listdef, startblock)
    }

    fn signed_integer() -> SomeValue {
        SomeValue::Integer(SomeInteger::new(false, false))
    }

    #[test]
    fn failed_subject_scope_restores_listitem_widened_by_reflow() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let (listdef, block) = annotate_nonneg_list_getitem(&ann, "list_reflow_rollback");
        let scope = ann.enter_added_blocks_scope();
        listdef
            .generalize(&signed_integer())
            .expect("generalize signed");
        ann.complete_pending_blocks().expect("reflow getitem");
        assert!(
            !getitem_result_nonneg(&block),
            "reflow inside the scope must widen the getitem result"
        );
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(!nonneg, "generalize must widen the list item");
        drop(scope);
        assert!(
            getitem_result_nonneg(&block),
            "uncommitted scope must restore the getitem result"
        );
        let (nonneg, mutated, resized) = list_item_state(&listdef);
        assert!(
            nonneg && !mutated && !resized,
            "uncommitted scope must restore the list item"
        );
    }

    #[test]
    fn failed_scope_undoes_listitem_before_reflow_runs() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let (listdef, block) = annotate_nonneg_list_getitem(&ann, "list_reflow_scheduled");
        let scope = ann.enter_added_blocks_scope();
        listdef
            .generalize(&signed_integer())
            .expect("generalize signed");
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(!nonneg);
        assert!(getitem_result_nonneg(&block));
        drop(scope);
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(nonneg);
        assert!(getitem_result_nonneg(&block));
        ann.complete_pending_blocks().expect("no leftover reflow");
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(nonneg);
        assert!(getitem_result_nonneg(&block));
    }

    #[test]
    fn committed_subject_scope_keeps_listitem_widen() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let (listdef, block) = annotate_nonneg_list_getitem(&ann, "list_reflow_commit");
        let scope = ann.enter_added_blocks_scope();
        listdef
            .generalize(&signed_integer())
            .expect("generalize signed");
        ann.complete_pending_blocks().expect("reflow getitem");
        scope.commit();
        drop(scope);
        assert!(!getitem_result_nonneg(&block));
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(!nonneg);
    }

    #[test]
    fn failed_subject_scope_restores_merged_listitem() {
        use super::super::listdef::ListDef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let (list_a, block) = annotate_nonneg_list_getitem(&ann, "list_merge_rollback");
        let list_b = ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            signed_integer(),
            true,
            true,
        );
        let scope = ann.enter_added_blocks_scope();
        list_a.union_with(&list_b).expect("union");
        ann.complete_pending_blocks()
            .expect("reflow merged getitem");
        assert!(list_a.same_as(&list_b));
        let (nonneg, _, resized) = list_item_state(&list_a);
        assert!(!nonneg && resized);
        assert!(!getitem_result_nonneg(&block));
        drop(scope);
        assert!(!list_a.same_as(&list_b), "owner retarget must be undone");
        let (nonneg, mutated, resized) = list_item_state(&list_a);
        assert!(nonneg && !mutated && !resized);
        let (b_nonneg, b_mutated, b_resized) = list_item_state(&list_b);
        assert!(!b_nonneg && b_mutated && b_resized);
        assert!(getitem_result_nonneg(&block));
    }

    #[test]
    fn nested_listitem_journal_respects_scope_commit() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let (listdef, block) = annotate_nonneg_list_getitem(&ann, "list_inner_rollback");
        let outer = ann.enter_added_blocks_scope();
        {
            let inner = ann.enter_added_blocks_scope();
            listdef
                .generalize(&signed_integer())
                .expect("generalize signed");
            ann.complete_pending_blocks().expect("inner reflow");
            assert!(!getitem_result_nonneg(&block));
            drop(inner);
        }
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(nonneg);
        assert!(getitem_result_nonneg(&block));
        drop(outer);
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(nonneg);

        let ann = RPythonAnnotator::new(None, None, None, false);
        let (listdef, block) = annotate_nonneg_list_getitem(&ann, "list_inner_commit");
        let outer = ann.enter_added_blocks_scope();
        {
            let inner = ann.enter_added_blocks_scope();
            listdef
                .generalize(&signed_integer())
                .expect("generalize signed");
            ann.complete_pending_blocks().expect("inner reflow");
            inner.commit();
        }
        drop(outer);
        assert!(!getitem_result_nonneg(&block));
        let (nonneg, _, _) = list_item_state(&listdef);
        assert!(
            !nonneg,
            "outer rollback must not undo a committed inner widen"
        );
    }

    #[test]
    fn committed_inner_resize_survives_outer_s_value_rollback() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let (listdef, block) = annotate_nonneg_list_getitem(&ann, "list_helper_resize");
        let outer = ann.enter_added_blocks_scope();
        listdef
            .generalize(&signed_integer())
            .expect("outer generalize");
        {
            let inner = ann.enter_added_blocks_scope();
            listdef.resize().expect("inner resize");
            inner.commit();
        }
        drop(outer);
        let (nonneg, mutated, resized) = list_item_state(&listdef);
        assert!(nonneg, "outer item widen rolls back, got signed");
        assert!(
            mutated && resized,
            "committed inner resize stays, got mutated={mutated} resized={resized}"
        );
        assert!(getitem_result_nonneg(&block));
    }

    fn classdef_named(
        ann: &RPythonAnnotator,
        name: &str,
    ) -> Rc<RefCell<super::super::classdesc::ClassDef>> {
        use super::super::super::flowspace::model::HostObject;
        use super::super::classdesc::ClassDesc;
        let pyobj = HostObject::new_class(name, vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            pyobj,
            name.into(),
        )));
        super::super::classdesc::ClassDef::new(&ann.bookkeeper, &desc)
    }

    fn unsigned_list(ann: &RPythonAnnotator) -> super::super::listdef::ListDef {
        use super::super::listdef::ListDef;
        ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::new(true, true)),
            false,
            false,
        )
    }

    fn impossible_list(ann: &RPythonAnnotator) -> super::super::listdef::ListDef {
        use super::super::listdef::ListDef;
        ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Impossible,
            false,
            false,
        )
    }

    fn install_list_attr(
        classdef: &Rc<RefCell<super::super::classdesc::ClassDef>>,
        name: &str,
        listdef: &super::super::listdef::ListDef,
    ) {
        use super::super::classdesc::Attribute;
        use super::super::model::SomeList;
        let mut attr = Attribute::new(name);
        attr.s_value = SomeValue::List(SomeList::new(listdef.clone()));
        classdef.borrow_mut().attrs.insert(name.into(), attr);
    }

    fn attr_list_same(
        classdef: &Rc<RefCell<super::super::classdesc::ClassDef>>,
        name: &str,
        listdef: &super::super::listdef::ListDef,
    ) -> bool {
        let borrowed = classdef.borrow();
        let Some(attr) = borrowed.attrs.get(name) else {
            return false;
        };
        match &attr.s_value {
            SomeValue::List(list) => list.listdef.same_as(listdef),
            _ => false,
        }
    }

    #[test]
    fn failed_scope_restores_class_attr_list() {
        use super::super::model::SomeList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let cd = classdef_named(&ann, "rordereddict.RDict");
        let field = unsigned_list(&ann);
        let fresh = impossible_list(&ann);
        install_list_attr(&cd, "indexes", &field);
        let scope = ann.enter_added_blocks_scope();
        super::super::classdesc::ClassDef::generalize_attr(
            &cd,
            "indexes",
            Some(SomeValue::List(SomeList::new(fresh.clone()))),
        )
        .expect("generalize_attr");
        assert!(
            fresh.same_as(&field),
            "Attribute.merge unifies the list items"
        );
        assert!(attr_list_same(&cd, "indexes", &fresh));
        drop(scope);
        assert!(
            attr_list_same(&cd, "indexes", &field),
            "uncommitted ClassDef.generalize_attr must restore the slot"
        );
        assert!(
            !fresh.same_as(&field),
            "ListItem.merge owner retarget is undone"
        );
        assert!(
            matches!(fresh.s_value(), SomeValue::Impossible),
            "fresh item rolls back, got {:?}",
            fresh.s_value()
        );
        assert!(
            matches!(field.s_value(), SomeValue::Integer(i) if i.unsigned && i.nonneg),
            "original item stays unsigned, got {:?}",
            field.s_value()
        );
        let restored = cd.borrow().attrs.get("indexes").unwrap().s_value.clone();
        let mut var = Variable::named("indexes");
        ann.setbinding(&mut var, SomeValue::List(SomeList::new(field.clone())));
        ann.setbinding(&mut var, restored);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ann.setbinding(&mut var, SomeValue::List(SomeList::new(fresh.clone())));
        }));
        assert!(
            refused.is_err(),
            "setbinding the rolled-back list over the original must refuse"
        );
    }

    #[test]
    fn failed_scope_restores_hoisted_subclass_attr() {
        use super::super::super::flowspace::model::HostObject;
        use super::super::classdesc::ClassDesc;
        use super::super::model::SomeList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let parent_py = HostObject::new_class("rordereddict.RDict", vec![]);
        let parent_desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            parent_py.clone(),
            "rordereddict.RDict".into(),
        )));
        let parent = super::super::classdesc::ClassDef::new(&ann.bookkeeper, &parent_desc);
        let child_py = HostObject::new_class("rordereddict.RDictSub", vec![parent_py]);
        let mut child_shell =
            ClassDesc::new_shell(&ann.bookkeeper, child_py, "rordereddict.RDictSub".into());
        child_shell.basedesc = Some(Rc::clone(&parent_desc));
        let child_desc = Rc::new(RefCell::new(child_shell));
        let child = super::super::classdesc::ClassDef::new(&ann.bookkeeper, &child_desc);
        let field = unsigned_list(&ann);
        let fresh = impossible_list(&ann);
        install_list_attr(&child, "indexes", &field);
        let scope = ann.enter_added_blocks_scope();
        super::super::classdesc::ClassDef::generalize_attr(
            &parent,
            "indexes",
            Some(SomeValue::List(SomeList::new(fresh.clone()))),
        )
        .expect("hoist");
        assert!(attr_list_same(&parent, "indexes", &fresh));
        assert!(child.borrow().attrs.get("indexes").is_none());
        drop(scope);
        assert!(
            parent.borrow().attrs.get("indexes").is_none(),
            "parent slot was absent before the failed hoist"
        );
        assert!(
            attr_list_same(&child, "indexes", &field),
            "subclass slot must come back"
        );
        assert!(!fresh.same_as(&field));
        assert!(matches!(fresh.s_value(), SomeValue::Impossible));
    }

    #[test]
    fn committed_scope_keeps_generalized_class_attr() {
        use super::super::model::SomeList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let cd = classdef_named(&ann, "rordereddict.RDict");
        let field = unsigned_list(&ann);
        let fresh = impossible_list(&ann);
        install_list_attr(&cd, "indexes", &field);
        let scope = ann.enter_added_blocks_scope();
        super::super::classdesc::ClassDef::generalize_attr(
            &cd,
            "indexes",
            Some(SomeValue::List(SomeList::new(fresh.clone()))),
        )
        .expect("generalize_attr");
        scope.commit();
        drop(scope);
        assert!(fresh.same_as(&field));
        assert!(attr_list_same(&cd, "indexes", &fresh));
        assert!(matches!(
            fresh.s_value(),
            SomeValue::Integer(i) if i.unsigned && i.nonneg
        ));
    }

    #[test]
    fn committed_inner_generalize_attr_survives_outer_rollback() {
        use super::super::model::SomeList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let cd = classdef_named(&ann, "rordereddict.RDict");
        let field = unsigned_list(&ann);
        let fresh = impossible_list(&ann);
        install_list_attr(&cd, "indexes", &field);
        let outer = ann.enter_added_blocks_scope();
        {
            let inner = ann.enter_added_blocks_scope();
            super::super::classdesc::ClassDef::generalize_attr(
                &cd,
                "indexes",
                Some(SomeValue::List(SomeList::new(fresh.clone()))),
            )
            .expect("inner generalize_attr");
            inner.commit();
        }
        drop(outer);
        assert!(
            fresh.same_as(&field),
            "outer rollback must not split a committed inner merge"
        );
        assert!(attr_list_same(&cd, "indexes", &fresh));
        assert!(
            !matches!(fresh.s_value(), SomeValue::Impossible),
            "committed inner list must stay widened, got {:?}",
            fresh.s_value()
        );
    }

    #[test]
    fn follow_link_applies_bool_true_constraint() {
        // source's a0 bound to SomeBool with an unknown value.
        // constraint map says "if this link is taken, a0 is True".
        // improve(SomeBool, s_True) returns s_True (contained). Target
        // inputarg receives the refined SomeBool(const=True) binding.
        use super::super::super::flowspace::model::{Block, ConstValue, Constant, Link};
        use super::super::model::SomeBool;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("bool_cstrt", 1);
        let u0 = Hlvalue::Variable(Variable::named("u0"));
        let target = Block::shared(vec![u0]);

        let source = graph.borrow().startblock.clone();
        {
            let mut src = source.borrow_mut();
            if let Hlvalue::Variable(v) = &mut src.inputargs[0] {
                ann.setbinding(v, SomeValue::Bool(SomeBool::new()));
            }
        }
        let source_a0 = source.borrow().inputargs[0].clone();
        let link = Rc::new(RefCell::new(Link::new(
            vec![source_a0.clone()],
            Some(target.clone()),
            None,
        )));

        let mut constraints: HashMap<Rc<Variable>, SomeValue> = HashMap::new();
        if let Hlvalue::Variable(v) = &source_a0 {
            let mut s_true = SomeBool::new();
            s_true.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
            constraints.insert(Rc::new(v.clone()), SomeValue::Bool(s_true));
        }
        ann.follow_link(&graph, &link, &constraints);

        // target.inputargs[0] should carry the refined bool (const=True).
        let bound = {
            let t = target.borrow();
            match &t.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.borrow().as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        match bound {
            Some(SomeValue::Bool(b)) => {
                assert!(b.base.const_box.is_some(), "expected constant bool");
            }
            other => panic!("expected SomeBool(const=True), got {:?}", other),
        }
    }

    #[test]
    fn recursivecall_queues_callee_and_wires_notify() {
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let caller = mk_graph("caller", 0);
        let callee = mk_graph("callee", 1);
        let caller_startblock = caller.borrow().startblock.clone();
        let whence = Some((Rc::clone(&caller), Rc::clone(&caller_startblock), 0));

        let r = ann.recursivecall(
            &callee,
            whence,
            &[Some(SomeValue::Integer(SomeInteger::default()))],
        );
        // The callee's return var has no binding yet → s_ImpossibleValue.
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
        // notify[callee.returnblock] should carry the caller's
        // position key.
        let returnblock = callee.borrow().returnblock.clone();
        let notify = ann.notify.borrow();
        let entry = notify
            .get(&BlockKey::of(&returnblock))
            .expect("notify entry missing");
        assert!(Rc::ptr_eq(&entry.block, &returnblock));
        assert_eq!(entry.positions.len(), 1);
    }

    #[test]
    fn complete_on_empty_schedule_is_noop() {
        // No pending blocks, no translator — complete() should fall
        // through the "all empty" branch and exit cleanly.
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        ann.complete().unwrap();
    }

    #[test]
    fn complete_rejects_added_blocks_entries_missing_from_annotated() {
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let graph = mk_graph("broken", 0);
        let block = graph.borrow().startblock.clone();

        ann.added_blocks
            .borrow_mut()
            .replace(IndexMap::from([(BlockKey::of(&block), block)]));

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ann.complete()))
            .expect_err("missing annotated entry should still panic");
        let text = if let Some(text) = err.downcast_ref::<String>() {
            text.as_str()
        } else if let Some(text) = err.downcast_ref::<&'static str>() {
            text
        } else {
            panic!("unexpected panic payload");
        };
        assert!(text.contains("complete: added_blocks entry missing from annotated"));
    }

    #[test]
    fn reflowpendingblock_marks_block_as_needing_reflow() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("h", 1);
        let startblock = graph.borrow().startblock.clone();
        ann.addpendinggraph(&graph, &[Some(SomeValue::Integer(SomeInteger::default()))]);

        // Simulate "block was annotated by processblock" by setting
        // annotated[block] = Some(graph).
        let bkey = BlockKey::of(&startblock);
        ann.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(Rc::clone(&graph)));

        ann.reflowpendingblock(&graph, &startblock);

        // Now annotated[block] is None (= False) and the block is
        // re-queued in genpendingblocks and blocked_blocks.
        let annmap = ann.annotated.borrow();
        assert!(matches!(annmap.get(&bkey), Some(None)));
        drop(annmap);
        let pending = ann.genpendingblocks.borrow();
        assert!(pending[0].contains_key(&bkey));
        drop(pending);
        let blocked = ann.blocked_blocks.borrow();
        assert!(blocked.contains_key(&bkey));
    }

    #[test]
    fn reflowfromposition_skips_block_evicted_from_annotated() {
        // `raise_if_subject_blocked` evicts a Skipped subject's blocks
        // from `annotated` but leaves the shared `notify` map untouched;
        // the block's session-retained `Rc` keeps the position's
        // `Weak<Block>` upgradeable. A later notify reflow must silently
        // skip such a dangling position instead of asserting in
        // `reflowpendingblock`.
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let caller = mk_graph("caller", 0);
        let callee = mk_graph("callee", 1);
        let caller_startblock = caller.borrow().startblock.clone();
        let whence = Some((Rc::clone(&caller), Rc::clone(&caller_startblock), 0));
        ann.recursivecall(
            &callee,
            whence,
            &[Some(SomeValue::Integer(SomeInteger::default()))],
        );

        // Annotate the caller block, then evict it (mirroring the
        // rollback in `raise_if_subject_blocked`) while notify still
        // points at it.
        let bkey = BlockKey::of(&caller_startblock);
        ann.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(Rc::clone(&caller)));
        ann.annotated.borrow_mut().shift_remove(&bkey);
        assert!(!ann.annotated.borrow().contains_key(&bkey));

        // Fire the stale notify position — must not panic.
        let returnblock = callee.borrow().returnblock.clone();
        let positions: Vec<_> = ann
            .notify
            .borrow()
            .get(&BlockKey::of(&returnblock))
            .map(|entry| entry.positions.iter().cloned().collect())
            .unwrap_or_default();
        assert_eq!(positions.len(), 1);
        for position in &positions {
            ann.reflowfromposition(position);
        }

        // The evicted block was neither re-scheduled nor re-blocked.
        assert!(!ann.annotated.borrow().contains_key(&bkey));
        assert!(!ann.blocked_blocks.borrow().contains_key(&bkey));
    }

    // Dispatch::None consider() — upstream operation.py.

    #[test]
    fn consider_newtuple_builds_sometuple_from_args() {
        // NewTuple.consider returns SomeTuple(items=[annotator.annotation(a)
        // for a in args]).  Bind two args to SomeInteger / s_str0 and
        // confirm the Tuple carries both.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        use super::super::model::s_str0;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("nt", 2);
        let startblock = graph.borrow().startblock.clone();
        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
            if let Hlvalue::Variable(v) = &mut blk.inputargs[1] {
                ann.setbinding(v, s_str0());
            }
        }
        let blk = startblock.borrow();
        let args = blk.inputargs.clone();
        drop(blk);
        let hlop = HLOperation::new(OpKind::NewTuple, args);
        let got = hlop.consider(&ann).unwrap().unwrap();
        if let SomeValue::Tuple(t) = got {
            assert_eq!(t.items.len(), 2);
            assert!(matches!(t.items[0], SomeValue::Integer(_)));
            assert!(matches!(t.items[1], SomeValue::String(_)));
        } else {
            panic!("expected SomeValue::Tuple, got {:?}", got);
        }
    }

    #[test]
    fn consider_newlist_routes_through_bookkeeper_newlist() {
        // NewList.consider defers to bookkeeper.newlist(*args_s); with
        // a single SomeInteger arg the returned ListDef.s_value must be
        // SomeInteger.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("nl", 1);
        let startblock = graph.borrow().startblock.clone();
        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
        }
        let args = startblock.borrow().inputargs.clone();
        let hlop = HLOperation::new(OpKind::NewList, args);
        let got = hlop.consider(&ann).unwrap().unwrap();
        if let SomeValue::List(list) = got {
            assert!(matches!(list.listdef.s_value(), SomeValue::Integer(_)));
        } else {
            panic!("expected SomeValue::List, got {:?}", got);
        }
    }

    #[test]
    fn consider_newdict_returns_empty_somedict() {
        // NewDict.consider returns bookkeeper.newdict(), an empty dict
        // whose s_key/s_value start as Impossible (DictDef default).
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let hlop = HLOperation::new(OpKind::NewDict, vec![]);
        let got = hlop.consider(&ann).unwrap().unwrap();
        assert!(matches!(got, SomeValue::Dict(_)));
    }

    #[test]
    fn consider_newslice_raises_annotator_error() {
        // NewSlice.consider always raises AnnotatorError.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let hlop = HLOperation::new(OpKind::NewSlice, vec![]);
        let err = hlop.consider(&ann).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("extended slicing"));
    }

    // gather_error / keepgoing — upstream annrpython.py:531-537.

    #[test]
    fn gather_error_returns_location_string() {
        // gather_error formats graph/block/op provenance per upstream
        // error.py:67-82. The Rust wrapper now follows upstream's
        // valid-index contract; when the graph has no source metadata,
        // the formatter must still include the graph provenance and
        // the `no source!` marker.
        use super::super::super::flowspace::model::SpaceOperation;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("gather", 0);
        let startblock = graph.borrow().startblock.clone();
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "newtuple",
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        ));
        let msg = gather_error(&ann, &graph, &startblock, 0);
        assert!(
            msg.contains("In <FunctionGraph of gather at 0x"),
            "got {msg:?}"
        );
        assert!(msg.contains("no source!"), "got {msg:?}");
    }

    #[test]
    fn position_key_from_refs_upgrades_graph_and_block() {
        // PositionKey carries Weak refs to FunctionGraph / Block so
        // `reflowfromposition` can recover them. The synthetic
        // `PositionKey::new` path leaves the refs dangling.
        let graph = mk_graph("pkrefs", 0);
        let startblock = graph.borrow().startblock.clone();
        let pk = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 7);
        let g = pk.graph().expect("graph weak ref should upgrade");
        assert!(Rc::ptr_eq(&g, &graph));
        let b = pk.block().expect("block weak ref should upgrade");
        assert!(Rc::ptr_eq(&b, &startblock));
        assert_eq!(pk.op_index, 7);

        // Synthetic constructor leaves refs dangling.
        let pk2 = super::super::bookkeeper::PositionKey::new(1, 2, 3);
        assert!(pk2.graph().is_none());
        assert!(pk2.block().is_none());
    }

    #[test]
    fn position_key_equality_on_identity_hashes() {
        // Two PositionKey values from the same Rc<FunctionGraph> /
        // Rc<Block> must compare equal regardless of how they were
        // constructed — matches Python tuple identity via `id()`.
        let graph = mk_graph("eq", 0);
        let startblock = graph.borrow().startblock.clone();
        let pk_a = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 0);
        let pk_b = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 0);
        assert_eq!(pk_a, pk_b);
        // Different op_index → not equal.
        let pk_c = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 1);
        assert_ne!(pk_a, pk_c);
    }

    #[test]
    fn flowin_error_variants_convert_from() {
        // From impls on FlowinError matter because consider_op uses
        // `?` on all three upstream exception classes.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        use super::super::model::{AnnotatorError, HarmlesslyBlocked};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let hlop = HLOperation::new(OpKind::NewDict, vec![]);
        let blocked: FlowinError = BlockedInference::new(&ann, hlop, None).into();
        assert!(matches!(blocked, FlowinError::Blocked(_)));
        let harmless: FlowinError = HarmlesslyBlocked.into();
        assert!(matches!(harmless, FlowinError::Harmless(_)));
        let annerr: FlowinError = AnnotatorError::new("oops").into();
        assert!(matches!(annerr, FlowinError::Annotator(_)));
    }

    #[test]
    fn build_types_rejects_non_user_function() {
        // Upstream: `assert isinstance(function, types.FunctionType),
        // "fix that!"` (annrpython.py:76). Passing a class object is
        // the most common accidental mis-call.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let cls =
            super::super::super::flowspace::model::HostObject::new_class("pkg.Foo", Vec::new());
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ann.build_types(&cls, &[], true, false);
        }));
        assert!(err.is_err());
    }

    #[test]
    fn translation_config_check_str_without_nul_round_trips_through_translator() {
        // Minimal scaffolding test: the `TranslationConfig.translation`
        // subfield added to back `build_types` annrpython.py must
        // be settable at construction and readable through
        // `self.translator.config.translation.*`.
        use crate::translator::translator::TranslationContext;
        let mut translator = TranslationContext::new();
        translator.config.translation.check_str_without_nul = true;
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        assert!(ann.translator.config.translation.check_str_without_nul);
    }

    #[test]
    fn explicit_translator_does_not_install_translator_annotator_backref() {
        use crate::translator::translator::TranslationContext;
        let translator = TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        assert!(ann.translator.annotator().is_none());
    }

    #[test]
    fn implicit_translator_installs_translator_annotator_backref() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator_ann = ann
            .translator
            .annotator()
            .expect("implicit translator should point back to annotator");
        assert!(Rc::ptr_eq(&translator_ann, &ann));
    }

    #[test]
    fn get_call_parameters_rejects_non_function_desc() {
        // Upstream calls `desc.get_call_parameters(args_s)` directly.
        // Passing a class HostObject routes through `getdesc -> ClassDesc`,
        // which does not define that method.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let cls =
            super::super::super::flowspace::model::HostObject::new_class("pkg.Foo", Vec::new());
        let err = ann.get_call_parameters(&cls, Vec::new()).unwrap_err();
        assert!(
            err.msg
                .as_ref()
                .is_some_and(|m| m.contains("has no get_call_parameters")),
            "expected missing `get_call_parameters`, got {:?}",
            err.msg
        );
    }

    fn classdef_name(s: &SomeValue) -> String {
        let classdef = match s {
            SomeValue::Instance(inst) => inst.classdef.clone(),
            SomeValue::Exception(exc) => exc.classdefs.first().cloned(),
            SomeValue::TypeOf(_) => return "typeof".to_string(),
            other => return format!("{other:?}"),
        };
        classdef
            .map(|cd| cd.borrow().name.clone())
            .unwrap_or_else(|| "<none>".to_string())
    }

    fn exception_instance(ann: &RPythonAnnotator, name: &str) -> SomeValue {
        use super::super::model::SomeInstance;
        use crate::flowspace::model::HOST_ENV;
        let host = HOST_ENV
            .lookup_exception_class(name)
            .unwrap_or_else(|| panic!("HOST_ENV missing {name}"));
        let classdef = ann
            .bookkeeper
            .getuniqueclassdef(&host)
            .unwrap_or_else(|e| panic!("getuniqueclassdef({name}): {e}"));
        SomeValue::Instance(SomeInstance::new(
            Some(classdef),
            false,
            std::collections::BTreeMap::new(),
        ))
    }

    /// Follow one raise link. Returns the extravar binding, the
    /// non-value block input that was pre-bound to `Exception`, and
    /// the `last_exc_value` block input.
    fn follow_one_raise(
        ann: &RPythonAnnotator,
        s_exc: SomeValue,
    ) -> (SomeValue, SomeValue, SomeValue, SomeValue) {
        use crate::flowspace::model::{Block, Link};
        let mut payload = Variable::named("payload");
        ann.setbinding(&mut payload, SomeValue::Integer(SomeInteger::default()));
        let mut also_exc = Variable::named("also_exc");
        ann.setbinding(&mut also_exc, exception_instance(ann, "Exception"));
        let exc_type = Variable::named("last_exception");
        let exc_value = Variable::named("last_exc_value");
        let target = Block::shared(vec![
            Hlvalue::Variable(Variable::named("in_payload")),
            Hlvalue::Variable(Variable::named("in_also")),
            Hlvalue::Variable(Variable::named("in_etype")),
            Hlvalue::Variable(Variable::named("in_evalue")),
        ]);
        let mut link = Link::new(
            vec![
                Hlvalue::Variable(payload),
                Hlvalue::Variable(also_exc),
                Hlvalue::Variable(exc_type.clone()),
                Hlvalue::Variable(exc_value.clone()),
            ],
            Some(target.clone()),
            None,
        );
        link.extravars(
            Some(Hlvalue::Variable(exc_type)),
            Some(Hlvalue::Variable(exc_value)),
        );
        let link = link.into_ref();
        let graph = mk_graph("raise_input", 0);
        ann.follow_raise_link(&graph, &link, s_exc);
        let extravar = {
            let link = link.borrow();
            let Hlvalue::Variable(v) = link.last_exc_value.as_ref().unwrap() else {
                panic!("last_exc_value must be a variable");
            };
            v.annotation
                .borrow()
                .as_ref()
                .map(|rc| (**rc).clone())
                .expect("extravar binding")
        };
        let type_extravar = {
            let link = link.borrow();
            let Hlvalue::Variable(v) = link.last_exception.as_ref().unwrap() else {
                panic!("last_exception must be a variable");
            };
            v.annotation
                .borrow()
                .as_ref()
                .map(|rc| (**rc).clone())
                .expect("type extravar binding")
        };
        let inputs = {
            let block = target.borrow();
            block
                .inputargs
                .iter()
                .map(|arg| {
                    let Hlvalue::Variable(v) = arg else {
                        panic!("inputarg");
                    };
                    v.annotation
                        .borrow()
                        .as_ref()
                        .map(|rc| (**rc).clone())
                        .expect("input binding")
                })
                .collect::<Vec<_>>()
        };
        (
            extravar,
            type_extravar,
            inputs[1].clone(),
            inputs[3].clone(),
        )
    }

    fn assert_class_ptr_eq(
        s: &SomeValue,
        expected: &Rc<RefCell<super::super::classdesc::ClassDef>>,
        what: &str,
    ) {
        let SomeValue::Instance(inst) = s else {
            panic!("{what}: expected instance, got {}", classdef_name(s));
        };
        let got = inst.classdef.as_ref().expect("classdef");
        assert!(
            Rc::ptr_eq(got, expected),
            "{what}: class {} is not {}",
            got.borrow().name,
            expected.borrow().name
        );
    }

    #[test]
    fn bare_exception_block_input_stays_exception() {
        use super::super::model::SomeException;
        use crate::front::StructFieldRegistry;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyobject::PyObject".to_string(),
            vec![("ob_type".to_string(), "usize".to_string())],
        );
        ann.bookkeeper.set_struct_fields(Rc::new(reg));
        let exception = {
            let s = exception_instance(&ann, "Exception");
            let SomeValue::Instance(inst) = &s else {
                panic!("Exception instance");
            };
            inst.classdef.clone().unwrap()
        };
        let index = {
            let s = exception_instance(&ann, "IndexError");
            let SomeValue::Instance(inst) = &s else {
                panic!("IndexError instance");
            };
            inst.classdef.clone().unwrap()
        };

        let (extravar, type_extravar, also, value) =
            follow_one_raise(&ann, exception_instance(&ann, "Exception"));
        assert_class_ptr_eq(&extravar, &exception, "bare extravar");
        assert!(
            matches!(type_extravar, SomeValue::TypeOf(_)),
            "last_exception stays typeof, got {}",
            classdef_name(&type_extravar)
        );
        assert_class_ptr_eq(&also, &exception, "non-value Exception arg");
        // `follow_raise_link` (annrpython.py) binds the handler input
        // from `annotation(v_out)`. OperationError subclasses Exception
        // and is never a W_Root, so the input stays Exception.
        assert_class_ptr_eq(&value, &exception, "bare Exception block input");

        let (extravar, _, _, value) =
            follow_one_raise(&ann, exception_instance(&ann, "IndexError"));
        assert_class_ptr_eq(&extravar, &index, "IndexError extravar");
        assert_class_ptr_eq(&value, &index, "lone IndexError block input");

        let overflow = exception_instance(&ann, "OverflowError");
        let (_, _, _, value) = follow_one_raise(&ann, overflow);
        let overflow_cd = {
            let s = exception_instance(&ann, "OverflowError");
            let SomeValue::Instance(inst) = s else {
                panic!("OverflowError");
            };
            inst.classdef.unwrap()
        };
        assert_class_ptr_eq(&value, &overflow_cd, "OverflowError block input");

        let key = {
            let s = exception_instance(&ann, "KeyError");
            let SomeValue::Instance(inst) = s else {
                panic!("KeyError");
            };
            inst.classdef.unwrap()
        };
        let catch_all = SomeValue::Exception(SomeException::new(vec![
            Rc::clone(&index),
            key,
            Rc::clone(&exception),
        ]));
        let (extravar, _, _, value) = follow_one_raise(&ann, catch_all.clone());
        assert!(
            matches!(extravar, SomeValue::Exception(_)),
            "catch-all extravar stays SomeException, got {}",
            classdef_name(&extravar)
        );
        assert!(
            matches!(value, SomeValue::Exception(_)),
            "catch-all block input stays SomeException, got {}",
            classdef_name(&value)
        );

        let lone = SomeValue::Exception(SomeException::new(vec![Rc::clone(&index)]));
        let (extravar, _, _, value) = follow_one_raise(&ann, lone);
        let SomeValue::Exception(exc) = &value else {
            panic!(
                "lone IndexError SomeException stays SomeException, got {}",
                classdef_name(&value)
            );
        };
        assert!(
            exc.classdefs.len() == 1 && Rc::ptr_eq(&exc.classdefs[0], &index),
            "lone IndexError SomeException must not widen to the PyObject root"
        );
        let SomeValue::Exception(exc_var) = &extravar else {
            panic!(
                "lone IndexError extravar stays SomeException, got {}",
                classdef_name(&extravar)
            );
        };
        assert!(Rc::ptr_eq(&exc_var.classdefs[0], &index));
    }

    #[test]
    fn pyerror_block_input_is_deref_target() {
        use super::super::model::SomeInstance;
        use crate::front::StructFieldRegistry;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyobject::PyObject".to_string(),
            vec![("ob_type".to_string(), "usize".to_string())],
        );
        reg.fields.insert(
            "error::PyErrorObject".to_string(),
            vec![("ob_header".to_string(), "pyobject::PyObject".to_string())],
        );
        ann.bookkeeper.set_struct_fields(Rc::new(reg));
        ann.bookkeeper.set_exception_carrier("error::PyErrorObject");
        let host = ann
            .bookkeeper
            .exception_carrier_class()
            .expect("carrier class");
        let carrier = ann
            .bookkeeper
            .getuniqueclassdef(&host)
            .expect("PyErrorObject classdef");
        let s_exc = SomeValue::Instance(SomeInstance::new(
            Some(carrier.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let (extravar, _, _, value) = follow_one_raise(&ann, s_exc);
        assert_class_ptr_eq(&extravar, &carrier, "extravar stays the carrier");
        assert_eq!(
            classdef_name(&value),
            "error::PyErrorObject",
            "block input is the handle's Deref target"
        );
    }
}
