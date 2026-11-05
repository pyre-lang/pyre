//! Debug-gated detector for distinct `OpRef` variants sharing one `raw()` key.
//!
//! `OpRef::raw()` omits the enum tag, so values from different input, operation,
//! or constant namespaces can map to the same integer. A collision matters when
//! two numbering scopes feed one raw-keyed table; payload changes within the
//! same variant are valid rewrites and are not collisions.
//!
//! Constants are skipped because they have no raw key. Callers must therefore
//! instrument before any transformation that replaces a register with a
//! constant. This audit reports contested lookup keys; the bank check in
//! `translate_trace_iter_opref` separately validates the resolved value.
//!
//! Build with `jit-audits` to compile the instrumentation sites. In that
//! build, `MAJIT_OPREF_VARIANT_AUDIT=1` reports and `=abort` panics on the
//! first collision; with the environment gate off, `note` is one thread-local
//! read and a return. Without the Cargo feature, the module and its call sites
//! are absent.
//!
//! State and mode are thread-local because each trace is processed on one
//! thread and parallel tests must not share diagnostic observations.
//!
//! Retires when the two namespaces are made structurally uncollidable: at that
//! point a collision is unrepresentable rather than merely unobserved, and
//! this instrument has nothing left to detect.

use crate::resoperation::OpRef;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::{Discriminant, discriminant};

const OFF: u8 = 1;
const REPORT: u8 = 2;
const ABORT: u8 = 3;

/// `(label, scope, key)` — one slot of one table.
type Key = (&'static str, usize, u32);

/// One distinct collision, plus how often it recurred.
struct Collision {
    label: &'static str,
    scope: usize,
    key: u32,
    first: OpRef,
    now: OpRef,
    occurrences: usize,
}

#[derive(Default)]
struct Audit {
    /// `None` until this thread has read the environment.
    mode: Option<u8>,
    /// The OpRef that first claimed each key.
    ///
    /// `scope` is caller-supplied and identifies the *table*, not the process:
    /// two different caches may legitimately assign different variants to the
    /// same raw, and only a collision *within one table* is a defect. Callers
    /// pass something stable for the table's lifetime, e.g. `slice.as_ptr()`.
    seen: HashMap<Key, OpRef>,
    /// Keyed by slot *and* by the ordered pair of variants, so one slot that
    /// collides two different ways stays two findings.
    collisions: HashMap<(Key, Discriminant<OpRef>, Discriminant<OpRef>), Collision>,
    collision_occurrences: usize,
    /// Every non-const note this thread accepted while enabled. This separates
    /// a clean result from an instrumented path that never ran.
    notes: usize,
    /// Notes whose key was already claimed and whose OpRef has the SAME
    /// variant as the stored entry — payload ignored, so a legitimate rewrite
    /// (`InputArgRef(0)` answered by `InputArgRef(4824)`) counts here, and so
    /// does re-noting a byte-identical box.
    ///
    revisits_same_variant: usize,
}

thread_local! {
    static AUDIT: RefCell<Audit> = RefCell::new(Audit::default());
}

fn resolve_mode() -> u8 {
    match std::env::var("MAJIT_OPREF_VARIANT_AUDIT") {
        Ok(v) if v == "abort" => ABORT,
        Ok(v) if v != "0" && !v.is_empty() => REPORT,
        _ => OFF,
    }
}

/// Whether the audit is collecting on this thread. Callers may use this to
/// skip computing a scope value; `note` re-checks, so it is an optimization.
#[inline]
pub fn enabled() -> bool {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        *audit.mode.get_or_insert_with(resolve_mode) >= REPORT
    })
}

/// Collision *occurrences* on this thread since start or since [`reset`] —
/// every firing, including repeats of one already reported.
///
/// Pair this with [`notes`], because zero also results when the instrumented
/// path never ran.
pub fn collisions() -> usize {
    AUDIT.with(|a| a.borrow().collision_occurrences)
}

/// Distinct `(slot, variant-pair)` collisions — how many places are affected,
/// as opposed to how often they fire ([`collisions`]).
pub fn distinct_collisions() -> usize {
    AUDIT.with(|a| a.borrow().collisions.len())
}

/// Notes that landed on an already-claimed key carrying the SAME variant.
///
/// See the field's doc for the exact definition. In short: a revisit, payload
/// ignored — the healthy translation-cache answer and the byte-identical
/// re-note both count here, and neither is a collision.
pub fn revisits_same_variant() -> usize {
    AUDIT.with(|a| a.borrow().revisits_same_variant)
}

/// Notes accepted on this thread while enabled, collisions included.
///
/// This is the reached-ness half of any verdict: `collisions() == 0` means
/// "no collision" only when `notes() > 0`. Zero here means the instrumented
/// code did not run — a dead instrument, not a clean one.
pub fn notes() -> usize {
    AUDIT.with(|a| a.borrow().notes)
}

/// Distinct `(label, scope, key)` triples seen on this thread.
pub fn keys_seen() -> usize {
    AUDIT.with(|a| a.borrow().seen.len())
}

/// Forget every key and zero this thread's counters.
pub fn reset() {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        audit.seen.clear();
        audit.collisions.clear();
        audit.collision_occurrences = 0;
        audit.notes = 0;
        audit.revisits_same_variant = 0;
    });
}

/// Print this thread's totals and one line per distinct collision.
///
/// Also runs from `Audit`'s destructor at thread exit, so a gated run gets a
/// total without the subject having to call anything. That is best-effort —
/// a thread-local destructor is not guaranteed to run on the main thread — so
/// a caller that needs the summary should call this itself rather than assume
/// teardown produced it.
pub fn report_summary() {
    AUDIT.with(|a| print_summary(&a.borrow()));
}

fn print_summary(audit: &Audit) {
    if audit.mode.unwrap_or(OFF) < REPORT {
        return;
    }
    eprintln!(
        "[opref-audit] SUMMARY notes={} keys={} revisits_same_variant={} \
         distinct_collisions={} collision_occurrences={}",
        audit.notes,
        audit.seen.len(),
        audit.revisits_same_variant,
        audit.collisions.len(),
        audit.collision_occurrences,
    );
    let mut rows: Vec<&Collision> = audit.collisions.values().collect();
    rows.sort_by_key(|l| std::cmp::Reverse(l.occurrences));
    for row in rows {
        eprintln!(
            "[opref-audit]   x{} {}: key={} (scope={:#x}) first {:?}, now {:?}",
            row.occurrences, row.label, row.key, row.scope, row.first, row.now,
        );
    }
}

impl Drop for Audit {
    fn drop(&mut self) {
        print_summary(self);
    }
}

/// Force this thread's mode, bypassing the environment, so a test can exercise
/// the detector regardless of how the run was invoked.
pub fn set_mode_for_test(on: bool) {
    AUDIT.with(|a| a.borrow_mut().mode = Some(if on { REPORT } else { OFF }));
}

/// Record that `opref` was used as a `raw()` key in `label`'s table `scope`.
///
/// Fires when an `OpRef` of a different variant already claimed that key in
/// the same table.
///
/// Inline-`Const` variants are skipped: they carry their value inline and have
/// no `raw()` encoding at all (`raw()` panics on them by design).
#[inline]
pub fn note(label: &'static str, scope: usize, opref: OpRef) {
    if opref.is_none() || opref.is_constant() {
        return;
    }
    note_key(label, scope, opref.raw(), opref);
}

/// Record that `opref` claims the key `key` in `label`'s table `scope`, where
/// `key` is supplied rather than taken from `opref.raw()`.
///
/// Use this for a lookup and the entry it resolves to: both must be recorded
/// under the table slot even when their own raw payloads differ. Only the enum
/// variant is compared.
#[inline]
pub fn note_key(label: &'static str, scope: usize, key: u32, opref: OpRef) {
    if opref.is_none() || opref.is_constant() {
        return;
    }
    let report = AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        if *audit.mode.get_or_insert_with(resolve_mode) < REPORT {
            return None;
        }
        audit.notes += 1;
        let abort = audit.mode == Some(ABORT);
        let slot = (label, scope, key);
        let Some(&first) = audit.seen.get(&slot) else {
            audit.seen.insert(slot, opref);
            return None;
        };
        if discriminant(&first) == discriminant(&opref) {
            audit.revisits_same_variant += 1;
            return None;
        }
        audit.collision_occurrences += 1;
        let occurrences = {
            let row = audit
                .collisions
                .entry((slot, discriminant(&first), discriminant(&opref)))
                .or_insert(Collision {
                    label,
                    scope,
                    key,
                    first,
                    now: opref,
                    occurrences: 0,
                });
            row.occurrences += 1;
            row.occurrences
        };
        // Print once per distinct collision. A hot slot would otherwise emit
        // one line per firing and bury every other finding in the run.
        if occurrences > 1 {
            return None;
        }
        Some((
            first,
            abort,
            audit.collisions.len(),
            audit.collision_occurrences,
        ))
    });
    let Some((first, abort, distinct, total)) = report else {
        return;
    };
    let message = format!(
        "[opref-audit] {label}: key={key} claimed by two OpRef variants in one table \
         (scope={scope:#x}): first {first:?}, now {opref:?} \
         [distinct #{distinct}, occurrences so far {total}]"
    );
    eprintln!("{message}");
    assert!(!abort, "{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// POSITIVE CONTROL. Two variants sharing a payload must fire.
    ///
    /// Without this the detector's silence is worthless: an assertion that
    /// only ever sees singletons reads exactly like one that never runs.
    #[test]
    fn two_variants_on_one_key_fire() {
        set_mode_for_test(true);
        reset();
        note("t", 0, OpRef::input_arg_int(7));
        assert_eq!(
            collisions(),
            0,
            "the first claim on a key is never a collision"
        );
        note("t", 0, OpRef::int_op(7));
        assert_eq!(
            collisions(),
            1,
            "InputArgInt(7) and IntOp(7) share raw=7 and must be reported"
        );
    }

    /// The collision is between a lookup and the entry it lands on,
    /// not between two lookups: the query `InputArgInt(0)` reaches slot 0,
    /// which the recorder filled for `InputArgRef(4824)`. The two carry
    /// different raws, so noting each under its own `raw()` files them under
    /// different keys and nothing is compared — the detector reads zero and is
    /// blind, not clean. Both must be noted under the SLOT.
    #[test]
    fn a_lookup_colliding_with_its_stored_entry_fires() {
        set_mode_for_test(true);
        reset();
        let slot = 0;
        note_key("cache", 0, slot, OpRef::input_arg_int(0));
        note_key("cache", 0, slot, OpRef::input_arg_ref(4824));
        assert_eq!(
            collisions(),
            1,
            "the query's bank and the stored entry's bank differ at slot 0"
        );

        // And the wiring that would have missed it: keying each by its own
        // raw() puts them in different buckets and reports nothing.
        reset();
        note("cache", 0, OpRef::input_arg_int(0));
        note("cache", 0, OpRef::input_arg_ref(4824));
        assert_eq!(
            collisions(),
            0,
            "documents the blind wiring — raw 0 and raw 4824 are different keys"
        );
    }

    /// A translation-cache rewrite may change the payload while retaining the
    /// same variant; that is not a namespace collision.
    #[test]
    fn a_rewrite_that_keeps_the_variant_is_silent() {
        set_mode_for_test(true);
        reset();
        let slot = 0;
        note_key("cache", 0, slot, OpRef::input_arg_ref(0));
        note_key("cache", 0, slot, OpRef::input_arg_ref(4824));
        assert_eq!(
            collisions(),
            0,
            "same bank, new position is what a translation cache is FOR"
        );
        assert_eq!(
            revisits_same_variant(),
            1,
            "and it must be counted, not merely ignored"
        );
    }

    /// The counters must partition the notes, so no note can be silently
    /// dropped into a category nobody reads. Any future early return that
    /// forgets to classify its note breaks this identity.
    #[test]
    fn every_note_lands_in_exactly_one_category() {
        set_mode_for_test(true);
        reset();
        note_key("t", 0, 3, OpRef::input_arg_int(3)); // new key
        note_key("t", 0, 3, OpRef::input_arg_int(99)); // revisit, same variant
        note_key("t", 0, 3, OpRef::int_op(3)); // collision
        note_key("t", 0, 4, OpRef::int_op(4)); // new key
        assert_eq!(notes(), 4);
        assert_eq!(
            keys_seen() + revisits_same_variant() + collisions(),
            notes(),
            "first claims + same-variant revisits + collisions must be every note"
        );
    }

    /// Dedupe and count are separate facts and the detector owes both: one
    /// line per distinct collision, and a tally of how hot each one is.
    #[test]
    fn a_recurring_collision_is_one_finding_with_many_occurrences() {
        set_mode_for_test(true);
        reset();
        note_key("t", 0, 5, OpRef::input_arg_int(5));
        for _ in 0..7 {
            note_key("t", 0, 5, OpRef::int_op(5));
        }
        assert_eq!(distinct_collisions(), 1, "one slot, one variant pair");
        assert_eq!(collisions(), 7, "and the firing rate is not thrown away");
    }

    /// CONTROL FOR THE LIVENESS COUNTER ITSELF.
    ///
    /// `notes()` exists so a reported `collisions() == 0` can be told apart
    /// from an instrument that never ran. A counter that is itself never
    /// incremented would reintroduce the same defect one level down, so it is
    /// asserted to move — and to move on the *silent* path, which is exactly
    /// the case a zero-collision verdict rests on.
    #[test]
    fn notes_counts_every_accepted_note() {
        set_mode_for_test(true);
        reset();
        assert_eq!(notes(), 0, "reset must zero the liveness counter");
        note("t", 0, OpRef::int_op(1));
        note("t", 0, OpRef::int_op(2));
        note("t", 0, OpRef::int_op(1));
        assert_eq!(notes(), 3, "every accepted note counts, repeats included");
        assert_eq!(keys_seen(), 2, "two distinct keys");
        assert_eq!(
            collisions(),
            0,
            "the reached-ness proof must hold on the silent path"
        );

        // Constants carry their value inline and have no raw() encoding, so
        // they are skipped entirely and must not inflate the reached count.
        note("t", 0, OpRef::const_int(9));
        assert_eq!(notes(), 3);
    }

    /// Gated off, the liveness counter must stay at zero too — otherwise a
    /// disabled run would look reached.
    #[test]
    fn disabled_records_no_notes() {
        set_mode_for_test(false);
        reset();
        note("t", 0, OpRef::int_op(1));
        assert_eq!(notes(), 0);
        assert_eq!(collisions(), 0);
    }

    /// NEGATIVE CONTROL. The same box seen repeatedly is the common case and
    /// must stay silent, or the detector drowns its own signal.
    #[test]
    fn one_variant_seen_repeatedly_is_silent() {
        set_mode_for_test(true);
        reset();
        for _ in 0..8 {
            note("t", 0, OpRef::int_op(7));
        }
        assert_eq!(collisions(), 0);
    }

    /// The same raw in two different tables is legal — a per-table numbering
    /// may assign the same position to different variants. Scoping is what
    /// keeps the detector from reporting correct behaviour as a defect.
    #[test]
    fn same_raw_in_two_scopes_is_silent() {
        set_mode_for_test(true);
        reset();
        note("t", 0x1000, OpRef::input_arg_int(7));
        note("t", 0x2000, OpRef::int_op(7));
        assert_eq!(collisions(), 0);
    }

    /// Distinct raws never collide however many variants are involved.
    #[test]
    fn distinct_raws_are_silent() {
        set_mode_for_test(true);
        reset();
        note("t", 0, OpRef::input_arg_int(0));
        note("t", 0, OpRef::int_op(1));
        note("t", 0, OpRef::input_arg_ref(2));
        assert_eq!(collisions(), 0);
    }

    /// Gated off, the detector must not report — otherwise every host pays for
    /// an instrument they did not ask for.
    #[test]
    fn disabled_records_nothing() {
        set_mode_for_test(false);
        reset();
        note("t", 0, OpRef::input_arg_int(7));
        note("t", 0, OpRef::int_op(7));
        assert_eq!(collisions(), 0);
    }
}
