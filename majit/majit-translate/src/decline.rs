//! Decline census: make a silently-refused lowering countable.
//!
//! Every lowering gate in this pipeline refuses with a bare `continue`,
//! `return 0`, or `false`.  None of them raises, none of them warns, and
//! none of them counts — so when a probe does not move after a change,
//! nothing in the tree can say whether the gate the change targeted was
//! consulted and said no, or was never reached at all.  Reading a jitcode
//! list by hand is currently the only way to tell those apart.
//!
//! Upstream does not stop translation either, but it carries the reason as
//! a value: `_handle_list_call` raises `NotSupported(prefix + oopspec_name)`
//! (`rpython/jit/codewriter/jtransform.py`), naming the shape it
//! refused, and `rewrite_op_direct_call` catches it and falls through to a
//! residual call (`jtransform.py`).  The refusal is equally silent
//! at the end — but at the point of refusal the reason existed.  This module
//! does NOT turn any decline into an error — every gate keeps its exact
//! current control flow — it only records the reason where it still exists,
//! so it survives the fall through to a residual.
//!
//! # Contract
//!
//! - **Nothing reads these counters to decide anything.**  The only
//!   consumers are [`snapshot`] and [`dump_to_stderr`], both of which
//!   write to stderr / return data to a test.  No gate branches on a
//!   count, so an instrumented build takes exactly the lowering decisions
//!   an uninstrumented one takes.
//! - **Off by default.**  With neither switch set, every recorder returns
//!   before touching the map or formatting anything; [`record`] takes its
//!   `subject` as `fmt::Arguments`, so the caller's `format_args!`
//!   allocates nothing on the disabled path.
//! - **A count names its gate AND its reason.**  A bare "something
//!   declined" count would reproduce the problem this module exists to
//!   fix, one level up.  Where a gate classifies its own refusal — the
//!   dual gate's `cutover::unported_category` — the count key is the ARM
//!   that matched, since which arm dominates is what decides the next
//!   stretch of work.
//! - **Events and subjects are counted separately.**  [`record_named`]
//!   and [`record_reason`] also record the distinct subject, so a row
//!   reports "N events over M distinct graphs". Reading an event count as
//!   a graph count is wrong wherever a gate can revisit one graph.
//!
//! # Switches
//!
//! `MAJIT_DECLINE_LOG` is the switch; `MAJIT_MIR_FRONTEND_DEBUG` (the
//! existing front-end debug switch, `front/checked_arith.rs` et al.) is
//! accepted as an alias at level 1 so a reader who already knows this
//! codebase's debug channel does not have to learn a second one.
//!
//! | value | effect |
//! |---|---|
//! | unset | disabled — no counting, no output, no formatting |
//! | `1` (or `MAJIT_MIR_FRONTEND_DEBUG` set) | count per (gate, reason); [`record_reason`] also prints its runtime reason |
//! | `2` | the above, plus one stderr line per individual decline |
//!
//! Level 1 does not print per-event lines because the instrumented gates
//! include per-op ones (`guess_call_kind` runs once per call operation in
//! every graph); a whole-program run would emit tens of thousands of
//! lines and bury the summary.  [`record_reason`] is the exception: its
//! callers are per-graph, and its reason is a runtime string that the
//! bounded count key cannot carry.
//!
//! # Reading a decline count
//!
//! A gate that reports zero declines is *not* the same as a gate that
//! accepted everything: it can equally mean the gate was never reached.
//! Where that distinction matters the gate also records the reached-but-
//! declined case under a distinct reason, so a missing row and a zero row
//! stay different observations.
//!
//! Population filters are deliberately NOT recorded.  `fuse_boxing_alloc`
//! inspects every operation in a graph and skips the ones that are not
//! `malloc_typed` calls at all; counting those would make the instrument
//! part of the population it measures — the count would be dominated by
//! operations that were never candidates.  Recording starts once a site
//! has been identified as the kind of thing the gate exists to lower.

use parking_lot::Mutex;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Arguments;
use std::sync::{LazyLock, OnceLock};

/// Every instrumented gate's census name, in one place.
///
/// These live here, beside the recorder, rather than next to each gate.
/// A name defined at its call site can be referenced while this module
/// is absent — which is exactly how a half-applied edit produced a tree
/// that named `model::fuse_boxing_alloc`'s gates and could not build.
/// Declared here, a gate name cannot exist without the module that
/// consumes it, so the two cannot go out of sync.
pub mod gate {
    /// `model::fuse_boxing_alloc`'s site loop — a `malloc_typed` call
    /// the pass identified but did not rewrite.
    pub const FUSE_BOXING_ALLOC: &str = "model::fuse_boxing_alloc";
    /// The `resolve_header_plan` closure inside it, whose `None` the site
    /// loop can only report as `vtable-unresolved`.
    pub const RESOLVE_HEADER_PLAN: &str = "model::resolve_header_plan";
    /// The `Result<T, PyError>` callee rule.
    pub const RESULT_EXC_CALLEE: &str = "result_exc::lower_result_exc_returns";
    /// The `?`-site caller rule.
    pub const RESULT_EXC_CALLER: &str = "result_exc::rewire_result_exc_call_sites";
    /// The builtin gateway PBC family that seeds graph discovery.
    pub const WRAPPER_FAMILY: &str = "call::compute_builtin_wrapper_indirect_graphs";
    /// Registry population — a callable skipped here resolves as a host
    /// builtin or residual stub rather than as a user graph.
    pub const CALL_REGISTRY: &str = "cutover::populate_call_registry_from_call_graphs";
    /// The real-rtyper-versus-legacy-walker fork.
    pub const DUAL_GATE: &str = "codewriter::dual_gate_publish_concretetypes";
    /// Graph discovery: which callees can become a `JitCode` at all.
    pub const FIND_ALL_GRAPHS: &str = "call::find_all_graphs_bfs";
    /// Per-call-site emission: `residual_call_*` versus an inlined entry.
    pub const GUESS_CALL_KIND: &str = "call::guess_call_kind";
    /// The policy clause behind a discovery refusal.
    pub const LOOK_INSIDE_GRAPH: &str = "policy::look_inside_graph";
    /// A body whose shadow-stack root bracket keeps its calls, by the shape
    /// the scalar replacement refused.
    pub const SHADOW_STACK_ERASE: &str = "mir::erase_shadow_stack";
    /// The `<[T]>::get(slice, i)` bounds-checked-diamond recognizer.  Its
    /// three conjuncts answer different questions, so each records its own
    /// reason: which one dominates decides whether the next stretch of work
    /// is a `SliceIndex` widening or a narrow-root one.
    pub const SLICE_GET_SITE: &str = "mir::recognize_slice_get_site";
    /// The `PlaceKind::Global` lane chain in `resolve_place`.  A static
    /// whose value none of the lanes can produce is synthesized as a
    /// nullary call to its own path, which surfaces two stages later as
    /// `not registered in CallRegistry` — naming the static, not the
    /// lane failure that produced it.  Recording here is what keeps the
    /// two apart in a census.
    pub const GLOBAL_PLACE: &str = "mir::resolve_place::Global";
    /// The per-function builder loop in `build_semantic_program_*`.  A
    /// declaration dropped here never becomes a `SemanticFunction`, so it
    /// never reaches `function_graphs` and every callsite resolves it as
    /// an unregistered path.
    pub const SEMANTIC_FN_LOOP: &str = "mir::build_semantic_program";
    /// A lowered graph passed a pointer to `OwnerRootGuard`, or to an
    /// aggregate that contains one, into a callee that stays residual.
    pub const OWNER_ROOT_GUARD_ESCAPE: &str = "mir::owner_root_guard";
    /// A lowered graph passed a pointer to `FrameAnchor`, or to an
    /// aggregate that contains one, into a callee that stays residual.
    pub const FRAME_ANCHOR_ESCAPE: &str = "mir::frame_anchor";
    /// A lowered graph passed `RBigIntGcRoot`, or an aggregate that
    /// contains one, by value or through a pointer, into a callee that
    /// stays residual, or received one as the call destination.
    pub const RBIGINT_GC_ROOT_ESCAPE: &str = "mir::rbigint_gc_root";
    /// A handle is in the graph, but the erased `RBigInt` type cannot be
    /// resolved from the handle's fields or from the `RBigInt` declaration
    /// path. Value lowering would drop `new` / `deref` while type lowering
    /// kept the handle.
    pub const RBIGINT_GC_ROOT_UNRESOLVED: &str = "mir::rbigint_gc_root_unresolved";
    /// `front::iter_adapter` `map(it, f).collect()` / `from_iter` rewrite.
    /// A structural miss leaves the residual `Range::map` FunctionPath,
    /// which then fails phaseA as an unregistered CallRegistry path.
    pub const ITER_ADAPTER: &str = "front::iter_adapter";
    /// Drop of a one-word `Vec` whose header this graph did not allocate.
    /// `rewrite_op_free` still wants a free; omitting it is a leak when
    /// the header was allocated here and only a phi hid the producer.
    pub const RUST_VEC_DROP: &str = "rust-vec-drop";
    /// Cross-crate `FunctionDesc.cachedgraph`: an Opaque-imported generic
    /// whose defining FunDecl could not be specialized (no unique body,
    /// or the call-site generics could not be rewritten into that crate's
    /// tables). The call keeps the unspecialized path.
    pub const CROSS_CRATE_SPEC: &str = "clause_spec::cross_crate_spec";
}

/// Verbosity of the decline census.  Resolved once, from the environment.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    /// No counting and no output.  The default.
    Off,
    /// Count per (gate, reason); print only runtime reason strings.
    Counters,
    /// Also print one line per individual decline.
    Events,
}

fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var_os("MAJIT_DECLINE_LOG") {
        Some(value) if value == "2" => Level::Events,
        Some(value) if value == "0" || value.is_empty() => Level::Off,
        Some(_) => Level::Counters,
        // Alias: the front end's established debug switch turns the
        // census on at counter level, so a reader who already reaches
        // for `MAJIT_MIR_FRONTEND_DEBUG` gets the decline rows too
        // rather than finding a second, parallel channel for the same
        // job.
        None if std::env::var_os("MAJIT_MIR_FRONTEND_DEBUG").is_some() => Level::Counters,
        None => Level::Off,
    })
}

/// Whether the census is on.
///
/// Call sites that would have to compute a reason (walk a call target,
/// re-derive which of three conditions refused) guard that work on this
/// so the disabled path stays free of it.
#[inline]
pub fn enabled() -> bool {
    level() != Level::Off
}

/// `(gate, reason) -> count`, ordered so a dump is stable across runs.
///
/// `Cow` because most reasons are `&'static str` tags fixed by the call
/// site, while a few gates classify a runtime message into a bounded set
/// of owned strings.  Unbounded keys (a graph name, a full panic payload)
/// belong in `subject`, never here.
static COUNTS: LazyLock<Mutex<BTreeMap<(&'static str, Cow<'static, str>), Row>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// One `(gate, reason)` row.
///
/// `events` and `subjects` can differ and the difference is the point:
/// a gate called once per graph can decline the same graph repeatedly
/// (`dual_gate_publish_concretetypes` runs per graph but a shared callee
/// is re-visited), so an event count alone cannot be read as "how many
/// graphs". A per-graph figure needs a denominator anyone can
/// reproduce, which means counting distinct subjects, not calls.
#[derive(Default)]
struct Row {
    events: u64,
    /// Distinct subjects, when the call site named one.  Empty for the
    /// gates whose subject is an operation rather than a nameable graph.
    subjects: BTreeSet<String>,
}

fn bump(gate: &'static str, reason: Cow<'static, str>, subject: Option<&str>) {
    // `parking_lot` carries no poison flag, so a thread that panicked
    // mid-record leaves the next caller an ordinary guard: the census cannot
    // turn someone else's failure into a second one, which is what an
    // instrument measuring a run must never do.
    let mut counts = COUNTS.lock();
    let row = counts.entry((gate, reason)).or_default();
    row.events += 1;
    if let Some(subject) = subject {
        row.subjects.insert(subject.to_string());
    }
}

/// Record one decline whose reason is fixed by the call site.
///
/// `gate` names the function that refused, `reason` names which of its
/// refusal paths ran, and `subject` names what was refused (a graph name,
/// a call path).  Only `(gate, reason)` is counted; `subject` is printed
/// at [`Level::Events`] and otherwise never formatted.
#[inline]
pub fn record(gate: &'static str, reason: &'static str, subject: Arguments<'_>) {
    let level = level();
    if level == Level::Off {
        return;
    }
    bump(gate, Cow::Borrowed(reason), None);
    if level == Level::Events {
        eprintln!("[decline] {gate} {reason}: {subject}");
    }
}

/// Record one decline against a NAMED subject, so the row carries a
/// distinct-subject count alongside its event count.
///
/// Use this wherever the subject is a graph (or anything else with a
/// stable identity a reader can count): it is what makes a figure like
/// "88 of 95 graphs" reproducible from the instrument rather than from
/// one person's notes. `subject` is formatted on every call once the
/// census is on, so keep it to an identity, not a dump.
#[inline]
pub fn record_named(gate: &'static str, reason: &'static str, subject: &str) {
    let level = level();
    if level == Level::Off {
        return;
    }
    bump(gate, Cow::Borrowed(reason), Some(subject));
    if level == Level::Events {
        eprintln!("[decline] {gate} {reason}: {subject}");
    }
}

/// Record one decline that already carries a formatted reason string,
/// against a named subject.
///
/// `class` is the bounded count key; `reason` is the gate's own message, which
/// is printed at [`Level::Counters`] — unconditionally, i.e. without consulting
/// any narrower switch the gate's own logging sits behind. Use this only for
/// per-graph gates: it formats on every call once the census is on.
#[inline]
pub fn record_reason(gate: &'static str, class: &'static str, reason: &str, subject: &str) {
    let level = level();
    if level == Level::Off {
        return;
    }
    bump(gate, Cow::Borrowed(class), Some(subject));
    eprintln!("[decline] {gate} {class} {subject}: {reason}");
}

/// Record a gate's ACCEPT arm, so its rows sum to a denominator.
///
/// Only for a gate where the accept/decline ratio is the finding — the
/// dual gate, where "N graphs Skipped" is meaningless without "out of
/// how many". Named `observe` rather than `record` because it is not a
/// decline, and the dump labels it so a reader cannot add it into one.
#[inline]
pub fn observe_accept(gate: &'static str, class: &'static str, subject: &str) {
    let level = level();
    if level == Level::Off {
        return;
    }
    bump(gate, Cow::Borrowed(class), Some(subject));
    if level == Level::Events {
        // `[accept]`, not `[decline]`: the per-event lines are what a reader
        // greps, and a shared prefix would put accepts into a decline count
        // the summary rows deliberately keep apart.
        eprintln!("[accept] {gate} {class}: {subject}");
    }
}

/// Every `(gate, reason, events, distinct_subjects)` recorded so far,
/// gate-then-reason ordered.
///
/// `distinct_subjects` is 0 where the call site named no subject; it is
/// NOT a claim that one subject was involved.
pub fn snapshot() -> Vec<(&'static str, String, u64, usize)> {
    let counts = COUNTS.lock();
    counts
        .iter()
        .map(|((gate, reason), row)| (*gate, reason.to_string(), row.events, row.subjects.len()))
        .collect()
}

/// The distinct subjects recorded under one `(gate, reason)` row.
///
/// The list, not the count — for the reader who needs to know WHICH
/// graphs, not how many. Empty for rows whose call site named none.
pub fn subjects_of(gate: &str, reason: &str) -> Vec<String> {
    let counts = COUNTS.lock();
    counts
        .iter()
        .filter(|((g, r), _)| *g == gate && r == reason)
        .flat_map(|(_, row)| row.subjects.iter().cloned())
        .collect()
}

/// Print the census to stderr.  A no-op when the census is off, so a
/// caller can wire this in unconditionally.
///
/// `label` identifies the run, since a test binary may census more than
/// one pipeline. Counters are cumulative across the whole process — the
/// map is never cleared, because clearing it would let one run's dump
/// silently omit declines that a previous run in the same binary had
/// already recorded.
pub fn dump_to_stderr(label: &str) {
    if !enabled() {
        return;
    }
    let rows = snapshot();
    let total: u64 = rows.iter().map(|(_, _, n, _)| n).sum();
    eprintln!(
        "=== majit decline census [{label}]: {total} events, {rows_len} (gate, reason) rows ===",
        rows_len = rows.len()
    );
    eprintln!(
        "    events  subjects  reason        (subjects = DISTINCT named subjects; '-' = call site named none)"
    );
    if rows.is_empty() {
        // Distinguish "no gate declined" from "the census was on but no
        // instrumented gate ran": both print this line, and neither is
        // evidence that a lowering succeeded.
        eprintln!("  (no instrumented gate recorded a decline in this process)");
        return;
    }
    let mut current = "";
    for (gate, reason, events, subjects) in rows {
        if gate != current {
            eprintln!("  {gate}");
            current = gate;
        }
        // A row whose subjects were never named prints `-` rather than
        // `0`: zero distinct subjects and "this gate does not name its
        // subject" are different facts and must not share a spelling.
        let subjects = if subjects == 0 {
            "-".to_string()
        } else {
            subjects.to_string()
        };
        eprintln!("    {events:6}  {subjects:>8}  {reason}");
    }
    // rust-vec-drop names the graphs, so a leak census can say which
    // headers a Drop could not prove this graph allocated.
    for reason in [
        "header-not-allocated-in-graph",
        "header-not-live-at-drop",
        "maybe-initialised",
        "vec-drop-unknown-item-kind",
    ] {
        let names = subjects_of(gate::RUST_VEC_DROP, reason);
        if names.is_empty() {
            continue;
        }
        eprintln!("  rust-vec-drop {reason} subjects ({n}):", n = names.len());
        for name in names {
            eprintln!("    {name}");
        }
    }
}

/// Dump the census when this value is dropped, including during an
/// unwind.
///
/// A translation run that panics is exactly the run whose refusals matter
/// most — the pipeline is designed to fail loud on a shape it cannot
/// digest — so the dump must not be an ordinary statement at the end of
/// the happy path, where a panic would skip it.
pub struct CensusScope {
    label: &'static str,
}

impl CensusScope {
    /// `label` is `&'static str` so an off census pays nothing to name a
    /// run it will not print.
    pub fn new(label: &'static str) -> Self {
        Self { label }
    }
}

impl Drop for CensusScope {
    fn drop(&mut self) {
        dump_to_stderr(self.label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The disabled path must not touch the map.  This is the property
    /// that keeps the instrument out of the decisions it measures: with
    /// the switch off there is no state for anything to read.
    ///
    /// The test binary inherits whatever the environment set, and
    /// `level()` is resolved once per process, so assert the invariant
    /// that holds either way: recording bumps this gate's own row exactly
    /// when the census is on.  The gate name is private to this test so
    /// the count is unaffected by whatever the rest of the test binary is
    /// declining in parallel.
    #[test]
    fn record_touches_the_map_exactly_when_enabled() {
        const GATE: &str = "decline::tests::record_touches_the_map_exactly_when_enabled";
        let row = |rows: Vec<(&'static str, String, u64, usize)>| -> u64 {
            rows.iter()
                .filter(|(gate, reason, _, _)| *gate == GATE && reason == "probe")
                .map(|(_, _, n, _)| *n)
                .sum()
        };
        let before = row(snapshot());
        record(GATE, "probe", format_args!("subject"));
        assert_eq!(row(snapshot()), before + u64::from(enabled()));
    }
}
