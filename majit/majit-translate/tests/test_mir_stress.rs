//! Stress test the MIR-driven lowering driver against the full
//! extracted `pyre-interpreter.ullbc` snapshot.
//!
//! Skipped by default: the snapshot is 133 MB and not checked into git
//! (regenerable via `scripts/extract-llbc.py`). Set
//! `MAJIT_MIR_STRESS_LLBC=path/to/file.ullbc` to enable, or use the
//! default path the extractor writes to.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    BasicBlock, PlaceKind, Rvalue, StmtKind, SwitchTargets, TermKind, Unstructured,
};
// `CallPayload`'s `dest` is the binding site for a Call-terminator local,
// which the lowering driver writes into `local_var[dest_local]` — so the
// classifier must treat Call-`dest` blocks as Assign-equivalent.
use majit_translate::front::mir::{LowerContext, LowerError, lower_fun_decl};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn stress_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MAJIT_MIR_STRESS_LLBC") {
        return Some(PathBuf::from(p));
    }
    let default = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    ));
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

/// Successors of a basic block, used by the RPO / back-edge classifier
/// `classify_uninitialised_local_rpo_vs_loop_carried`.  Mirrors the
/// `lower_terminator` edges: the normal target *and* `on_unwind` for
/// `Call`/`Assert`/`Drop`; both arms of an `If`; every `SwitchInt` arm
/// plus its default.
fn block_succs(llbc: &Llbc, blk: &BasicBlock) -> Vec<usize> {
    let ts: Vec<u64> = match blk.term(llbc) {
        Ok(TermKind::Goto { target }) => vec![target],
        Ok(TermKind::Call {
            target, on_unwind, ..
        })
        | Ok(TermKind::Assert {
            target, on_unwind, ..
        })
        | Ok(TermKind::Drop {
            target, on_unwind, ..
        }) => vec![target, on_unwind],
        Ok(TermKind::Switch { targets, .. }) => match targets {
            SwitchTargets::If(a, b) => vec![a, b],
            SwitchTargets::SwitchInt(_, arms, default) => {
                let mut v: Vec<u64> = arms.iter().map(|(_, bb)| *bb).collect();
                v.push(default);
                v
            }
        },
        _ => vec![],
    };
    ts.into_iter().map(|t| t as usize).collect()
}

/// Peel a `PlaceKind` down to the outermost MIR `Local(N)` it ultimately
/// addresses. `Local(N)` -> `Some(N)`; `Projection(inner, _)` recurses
/// into `inner` (so `*local`, `local.field`, `local[i]` all report `N`);
/// `Global`/`Unknown` -> `None`.
fn place_base_local(kind: &PlaceKind) -> Option<u64> {
    match kind {
        PlaceKind::Local(i) => Some(*i),
        PlaceKind::Projection(inner, _) => place_base_local(&inner.kind),
        _ => None,
    }
}

/// Compute reverse-postorder of the CFG rooted at block 0, *and* the set
/// of back-edges (child appears on the DFS recursion stack when the edge
/// is traversed). Returns `(rpo_index, back_edges)` where
/// `rpo_index[b]` is `b`'s position in reverse-postorder (lower = earlier)
/// or `usize::MAX` if `b` is unreachable from block 0.
fn rpo_and_back_edges(
    llbc: &Llbc,
    blocks: &[BasicBlock],
) -> (Vec<usize>, BTreeSet<(usize, usize)>) {
    let n = blocks.len();
    // Iterative DFS that records postorder and back-edges. `state`:
    // 0 = white (unvisited), 1 = grey (on stack), 2 = black (done).
    let mut state = vec![0u8; n];
    let mut postorder: Vec<usize> = Vec::with_capacity(n);
    let mut back_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    // Stack of (node, next-successor-index-to-process).
    let mut stack: Vec<(usize, usize)> = Vec::new();
    if n == 0 {
        return (vec![], back_edges);
    }
    state[0] = 1;
    stack.push((0, 0));
    while let Some(&(node, idx)) = stack.last() {
        let succs = block_succs(llbc, &blocks[node]);
        if idx < succs.len() {
            stack.last_mut().unwrap().1 += 1;
            let s = succs[idx];
            if s >= n {
                continue; // out-of-range successor, ignore
            }
            match state[s] {
                0 => {
                    state[s] = 1;
                    stack.push((s, 0));
                }
                1 => {
                    // Edge to a node currently on the recursion stack:
                    // a genuine back-edge.
                    back_edges.insert((node, s));
                }
                _ => {}
            }
        } else {
            state[node] = 2;
            postorder.push(node);
            stack.pop();
        }
    }
    // reverse-postorder index: reverse of postorder.
    let mut rpo_index = vec![usize::MAX; n];
    for (rank, &b) in postorder.iter().rev().enumerate() {
        rpo_index[b] = rank;
    }
    (rpo_index, back_edges)
}

/// Can `from` reach `to` using only forward edges (i.e. never traversing
/// a member of `back_edges`)? Plain BFS over the back-edge-pruned CFG.
fn reaches_without_backedge(
    llbc: &Llbc,
    blocks: &[BasicBlock],
    from: usize,
    to: usize,
    back_edges: &BTreeSet<(usize, usize)>,
) -> bool {
    if from == to {
        return true;
    }
    let n = blocks.len();
    let mut seen = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    seen[from] = true;
    queue.push_back(from);
    while let Some(b) = queue.pop_front() {
        for s in block_succs(llbc, &blocks[b]) {
            if s >= n || back_edges.contains(&(b, s)) {
                continue;
            }
            if s == to {
                return true;
            }
            if !seen[s] {
                seen[s] = true;
                queue.push_back(s);
            }
        }
    }
    false
}

/// Parse `read_bb` and `failing_local` out of the fail-loud message
/// `bb{R}: read of MIR local {N} before any Assign — …`.
fn parse_read_bb_and_local(msg: &str) -> Option<(usize, u64)> {
    // "bb{R}: read of MIR local {N} before any Assign"
    let rest = msg.strip_prefix("bb")?;
    let (r_str, after) = rest.split_once(':')?;
    let r: usize = r_str.trim().parse().ok()?;
    let marker = "read of MIR local ";
    let idx = after.find(marker)? + marker.len();
    let tail = &after[idx..];
    let end = tail.find(|c: char| !c.is_ascii_digit())?;
    let n: u64 = tail[..end].parse().ok()?;
    Some((r, n))
}

/// The census's finder predicate, and the one duplicated literal here.
///
/// This spelling is a copy. `resolve_place` in `majit_translate`'s
/// `front::mir` builds the failure message, and `is_known_lowering_gap`
/// tests for this same substring to decide whether to re-lower the body in
/// reverse-postorder. Neither is reachable from an integration test, so
/// nothing in this file can arm the literal against its producer: if the
/// driver's wording moves, this census reads 0 and that zero is a dead
/// selector wearing the shape of a clean corpus. Stated rather than
/// guarded — a guard here would only assert the copy against itself.
const FINDER: &str = "uninitialised local";

/// Which bucket one uninitialised-local read falls into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum UninitClass {
    ForwardRef,
    LoopCarried,
    Unknown,
}

impl UninitClass {
    fn label(self) -> &'static str {
        match self {
            UninitClass::ForwardRef => "forward-ref",
            UninitClass::LoopCarried => "loop-carried",
            UninitClass::Unknown => "unknown",
        }
    }
}

/// Bucket one `(read_bb, local_n)` uninitialised-local read against a
/// body's CFG, returning the class and the human detail line.
///
/// Extracted from the census loop so that the census and its positive
/// control run the SAME classifier: a control that re-implements the
/// classification grades a second copy and says nothing about the one that
/// produced the tally.
fn classify_uninit_read(
    llbc: &Llbc,
    blocks: &[BasicBlock],
    read_bb: usize,
    local_n: u64,
) -> (UninitClass, String) {
    // (1) Every block that BINDS local N, i.e. seeds `local_var[N]`.
    // The driver seeds the slot in two places:
    //   * `lower_assign` on a *direct* `PlaceKind::Local(N)` Assign
    //     statement, and
    //   * the Call-terminator destination `CallPayload.dest` when it is a
    //     direct `Local(N)`.
    // A *projection* write (`(*N).field = …`) presupposes N already bound
    // and never seeds the slot — we record those separately for the detail
    // string only.
    let mut direct_assign_blocks: Vec<usize> = Vec::new();
    let mut proj_assign_blocks: Vec<usize> = Vec::new();
    for (bb_idx, blk) in blocks.iter().enumerate() {
        let mut seeds = false;
        let mut proj = false;
        for st in &blk.statements {
            if let Ok(StmtKind::Assign(place, _)) = st.stmt_kind() {
                match &place.kind {
                    PlaceKind::Local(i) if *i == local_n => seeds = true,
                    PlaceKind::Projection(..) if place_base_local(&place.kind) == Some(local_n) => {
                        proj = true
                    }
                    _ => {}
                }
            }
        }
        // Call-terminator destination — the dominant binding site for
        // these failures (the local is the result of a fn call).
        if let Ok(TermKind::Call { call, .. }) = blk.term(llbc) {
            match &call.dest.kind {
                PlaceKind::Local(i) if *i == local_n => seeds = true,
                PlaceKind::Projection(..) if place_base_local(&call.dest.kind) == Some(local_n) => {
                    proj = true
                }
                _ => {}
            }
        }
        if seeds {
            direct_assign_blocks.push(bb_idx);
        }
        if proj {
            proj_assign_blocks.push(bb_idx);
        }
    }
    direct_assign_blocks.dedup();
    proj_assign_blocks.dedup();

    // (2)+(3) CFG: reverse-postorder + back-edge set from block 0.
    let (rpo_index, back_edges) = rpo_and_back_edges(llbc, blocks);

    // Classify against the *binding* (direct) assign blocks.
    let read_rpo = rpo_index.get(read_bb).copied().unwrap_or(usize::MAX);

    // Forward-ref signal A: at least one direct-assign block precedes the
    // read block in reverse-postorder.
    let mut rpo_precedes = false;
    // Forward-ref signal B (stronger): at least one direct-assign block
    // reaches the read block over forward edges only (never crossing a
    // back-edge), i.e. on a non-loop path.
    let mut forward_reaches = false;
    for &ab in &direct_assign_blocks {
        let ab_rpo = rpo_index.get(ab).copied().unwrap_or(usize::MAX);
        if ab_rpo != usize::MAX && read_rpo != usize::MAX && ab_rpo < read_rpo {
            rpo_precedes = true;
        }
        if reaches_without_backedge(llbc, blocks, ab, read_bb, &back_edges) {
            forward_reaches = true;
        }
    }

    if direct_assign_blocks.is_empty() {
        // No direct binding site at all (only projection writes, or none).
        // RPO ordering cannot help; this is not a plain forward-ref.
        if proj_assign_blocks.is_empty() {
            (
                UninitClass::Unknown,
                format!(
                    "local {local_n} read at bb{read_bb} has NO Assign anywhere \
                     (direct or projection); cannot classify"
                ),
            )
        } else {
            (
                UninitClass::LoopCarried,
                format!(
                    "local {local_n} read at bb{read_bb} has only projection-base \
                     writes at {proj_assign_blocks:?} (no direct Local({local_n}) \
                     Assign to seed local_var); RPO ordering cannot bind it"
                ),
            )
        }
    } else if rpo_precedes || forward_reaches {
        (
            UninitClass::ForwardRef,
            format!(
                "direct Assign(Local({local_n})) at blocks {direct_assign_blocks:?}; \
                 read at bb{read_bb}; RPO read-rank={read_rpo}; \
                 rpo_precedes={rpo_precedes} forward_reaches={forward_reaches} \
                 (an assign-block precedes/forward-reaches the read => RPO traversal \
                 alone binds it, no phi){}",
                if proj_assign_blocks.is_empty() {
                    String::new()
                } else {
                    format!("; proj-writes at {proj_assign_blocks:?}")
                }
            ),
        )
    } else {
        (
            UninitClass::LoopCarried,
            format!(
                "direct Assign(Local({local_n})) at blocks {direct_assign_blocks:?}; \
                 read at bb{read_bb}; RPO read-rank={read_rpo}; back-edges={:?}; \
                 every assign-block reaches the read ONLY through a back-edge \
                 (none precede it in RPO, none forward-reach it) => genuine \
                 loop-carried value needing a phi/block-inputarg{}",
                back_edges,
                if proj_assign_blocks.is_empty() {
                    String::new()
                } else {
                    format!("; proj-writes at {proj_assign_blocks:?}")
                }
            ),
        )
    }
}

/// Collapse an `Unsupported` reason to a stable family key: every run of
/// ASCII digits becomes `#`, and the result is bounded. `bb12: read of MIR
/// local 7 …` and `bb3: read of MIR local 40 …` therefore share one row,
/// so the reason table counts SHAPES rather than instances.
fn reason_family(msg: &str) -> String {
    let mut out = String::new();
    let mut last_digit = false;
    for c in msg.chars() {
        if out.len() >= 72 {
            out.push('…');
            break;
        }
        if c.is_ascii_digit() {
            if !last_digit {
                out.push('#');
            }
            last_digit = true;
        } else {
            last_digit = false;
            out.push(c);
        }
    }
    out
}

/// Positive control for the census below: demonstrate that the classifier
/// and its message parser CAN produce a non-zero bucket, on this corpus,
/// through the same code the census runs.
///
/// A denominator alone does not make the tally readable — a walk that
/// examines N functions and hands 0 to the classifier is still ambiguous
/// unless something shows the classifier is capable of bucketing at all.
/// This refuses (panics) rather than returning an uninterpretable zero:
/// if no body in the corpus can arm it, the tally must not be reported.
///
/// It grades the INSTRUMENT, never the corpus: the subjects are synthesised
/// `(read_bb, local)` pairs over a real body, so a green here says nothing
/// about how many genuine uninitialised-local failures exist.
fn arm_classifier(llbc: &Llbc) -> String {
    // (1) The message parser, both directions, against literals.
    let probe = "bb7: read of MIR local 42 before any Assign — \
                 uninitialised local, not yet supported";
    let unrelated = "bb7: a failure of some entirely different shape";
    assert_eq!(
        parse_read_bb_and_local(probe),
        Some((7, 42)),
        "control: the parser did not read (bb, local) out of a \
         producer-shaped message — the census can classify nothing"
    );
    assert_eq!(
        parse_read_bb_and_local(unrelated),
        None,
        "control: the parser accepted a message carrying no local read — \
         its Some() would not be evidence"
    );

    // (2) The finder predicate, both directions, against the same pair.
    // See FINDER: this arms the predicate against a literal of the
    // producer's shape, NOT against the producer.
    assert!(
        probe.contains(FINDER),
        "control: FINDER misses a producer-shaped message"
    );
    assert!(
        !unrelated.contains(FINDER),
        "control: FINDER matches an unrelated message"
    );

    // (3)+(4) The classifier itself, on a REAL body from this corpus.
    for fd in llbc.iter_local_fns() {
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let blocks = &body.body;
        if blocks.len() < 2 {
            continue;
        }
        let (rpo_index, _back_edges) = rpo_and_back_edges(llbc, blocks);

        // Discovery only: find a block that directly binds SOME local, and
        // a block that follows it in reverse-postorder. The bucket itself
        // is decided by `classify_uninit_read`, not here.
        let mut found: Option<(usize, usize, u64)> = None;
        'outer: for (a, blk) in blocks.iter().enumerate() {
            if rpo_index[a] == usize::MAX {
                continue;
            }
            let mut bound: Option<u64> = None;
            for st in &blk.statements {
                if let Ok(StmtKind::Assign(place, _)) = st.stmt_kind() {
                    if let PlaceKind::Local(i) = &place.kind {
                        bound = Some(*i);
                        break;
                    }
                }
            }
            if bound.is_none() {
                if let Ok(TermKind::Call { call, .. }) = blk.term(llbc) {
                    if let PlaceKind::Local(i) = &call.dest.kind {
                        bound = Some(*i);
                    }
                }
            }
            let Some(n) = bound else { continue };
            for r in 0..blocks.len() {
                if rpo_index[r] != usize::MAX && rpo_index[r] > rpo_index[a] {
                    found = Some((a, r, n));
                    break 'outer;
                }
            }
        }
        let Some((assign_bb, read_bb, local_n)) = found else {
            continue;
        };
        let name = fd.item_meta.name_path();

        // POSITIVE arm: a read placed after its binding block in RPO must
        // bucket as forward-ref.
        let (cls, detail) = classify_uninit_read(llbc, blocks, read_bb, local_n);
        assert_eq!(
            cls,
            UninitClass::ForwardRef,
            "control: a read of local {local_n} at bb{read_bb}, bound at \
             bb{assign_bb} which precedes it in RPO, did not bucket as \
             forward-ref in {name} | {detail}"
        );

        // NEGATIVE arm: a local nothing binds must NOT bucket as either
        // named class, or the positive arm above is not discriminating.
        let (absent_cls, _) = classify_uninit_read(llbc, blocks, read_bb, u64::MAX);
        assert_eq!(
            absent_cls,
            UninitClass::Unknown,
            "control: a local with no Assign anywhere bucketed as \
             {} in {name} — the classifier is not discriminating",
            absent_cls.label()
        );

        return format!(
            "control ARMED on {name}: local {local_n} bound at bb{assign_bb}, \
             read at bb{read_bb} => forward-ref; the same read of an unbound \
             local => unknown"
        );
    }

    panic!(
        "control UNARMED: no body in this corpus offers a direct local \
         binding with an RPO-later block, so nothing demonstrates the \
         classifier can bucket a subject — refusing to report a tally \
         nobody can read"
    );
}

/// Diagnostic: for each uninitialised-local
/// lowering failure, classify whether processing blocks in
/// reverse-postorder would bind the failing local before its read
/// (forward-ref, fixable by traversal order alone, no phi) or whether the
/// only `Assign` to the local reaches the reading block exclusively via a
/// back-edge (genuine loop-carried value, needs a phi / block-inputarg).
///
/// The subject set is post-retry, which changes what the buckets mean.
/// `lower_fun_decl` lowers linearly, and on a `LowerError::Unsupported`
/// whose message satisfies `is_known_lowering_gap` — which is satisfied by
/// exactly this census's [`FINDER`] substring — it re-lowers the whole body
/// in `BlockOrder::ReversePostorder`. The retry's `?` propagates whatever
/// that second attempt fails with, so a message reaching this census has
/// already failed under BOTH block orders. Consequently a non-zero
/// `forward_ref` is a DISAGREEMENT between this classifier and the driver
/// (the classifier says RPO ordering binds the read; the driver ran RPO and
/// it did not), never a subject waiting for an RPO fix.
///
/// The tally is therefore printed with its denominator: the filter ladder
/// from local functions walked down to messages the finder matched, plus
/// the family distribution of every `Unsupported` reason seen. Three zeros
/// on a live `Err(Unsupported)` population and three zeros on an empty one
/// are different readings, and the tally alone cannot tell them apart.
#[test]
#[ignore = "diagnostic; set MAJIT_MIR_STRESS_LLBC"]
fn classify_uninitialised_local_rpo_vs_loop_carried() {
    let Some(path) = stress_path() else {
        eprintln!("skip: set MAJIT_MIR_STRESS_LLBC");
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");

    // Arm the instrument BEFORE reporting anything it produces. This
    // panics rather than returning if the classifier cannot be shown to
    // bucket a subject on this corpus.
    eprintln!("{}", arm_classifier(&llbc));

    let mut forward_ref = 0usize;
    let mut loop_carried = 0usize;
    let mut other = 0usize;

    // The filter ladder, so the three buckets sit on a stated population.
    let mut walked = 0usize;
    let mut skipped_global_init = 0usize;
    let mut skipped_no_body = 0usize;
    let mut lowered_ok = 0usize;
    let mut err_not_found = 0usize;
    let mut err_schema = 0usize;
    let mut err_unsupported = 0usize;
    let mut finder_matched = 0usize;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let context = LowerContext::new(&llbc);

    for fd in llbc.iter_local_fns() {
        walked += 1;
        if fd.is_global_initializer().is_some() {
            skipped_global_init += 1;
            continue;
        }
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            skipped_no_body += 1;
            continue;
        };
        let msg = match lower_fun_decl(&context, fd) {
            Ok(_) => {
                lowered_ok += 1;
                continue;
            }
            Err(LowerError::FunctionNotFound(_)) => {
                err_not_found += 1;
                continue;
            }
            Err(LowerError::Schema(_)) => {
                err_schema += 1;
                continue;
            }
            Err(LowerError::Unsupported(msg)) => {
                err_unsupported += 1;
                *reasons.entry(reason_family(&msg)).or_insert(0) += 1;
                msg
            }
        };
        if !msg.contains(FINDER) {
            continue;
        }
        finder_matched += 1;
        let name = fd.item_meta.name_path();
        let Some((read_bb, local_n)) = parse_read_bb_and_local(&msg) else {
            eprintln!("PARSE-FAIL {name} | {msg}");
            other += 1;
            continue;
        };

        let (class, detail) = classify_uninit_read(&llbc, &body.body, read_bb, local_n);
        match class {
            UninitClass::ForwardRef => forward_ref += 1,
            UninitClass::LoopCarried => loop_carried += 1,
            UninitClass::Unknown => other += 1,
        }
        eprintln!("CLASSIFY [{}] {name} | {detail}", class.label());
    }

    let bucketed = forward_ref + loop_carried + other;

    eprintln!("\n=== uninitialised-local classification tally ===");
    eprintln!("forward_ref  (classifier says RPO order binds it): {forward_ref}");
    eprintln!("loop_carried (classifier says needs phi/inputarg): {loop_carried}");
    eprintln!("other/unknown:                                     {other}");
    eprintln!("--- bucketed total:                                {bucketed}");

    eprintln!("\n=== denominator: the ladder those buckets sit on ===");
    eprintln!("local fns walked:                {walked}");
    eprintln!("  skipped, global initializer:   {skipped_global_init}");
    eprintln!("  skipped, no Unstructured body: {skipped_no_body}");
    eprintln!("  lowered Ok:                    {lowered_ok}");
    eprintln!("  Err(FunctionNotFound):         {err_not_found}");
    eprintln!("  Err(Schema):                   {err_schema}");
    eprintln!("  Err(Unsupported):              {err_unsupported}");
    eprintln!("    of which matched {:?}: {finder_matched}", FINDER);

    let mut families: Vec<(&String, &usize)> = reasons.iter().collect();
    families.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    const TOP: usize = 20;
    eprintln!(
        "\n=== Err(Unsupported) reason families \
         (probe-set total {err_unsupported} over {} distinct families) ===",
        families.len()
    );
    for (family, n) in families.iter().take(TOP) {
        eprintln!("  {n:>6}  {family}");
    }
    if families.len() > TOP {
        let elided: usize = families.iter().skip(TOP).map(|(_, n)| **n).sum();
        eprintln!(
            "  ({} further families elided, {elided} occurrence(s))",
            families.len() - TOP
        );
    }

    // Partition identities. Neither says anything about the corpus — both
    // hold by construction — but they make a future `continue` added
    // without a counter, or a matched message that falls out of the
    // classifier entirely, loud rather than a silently shrunk denominator.
    assert_eq!(
        bucketed, finder_matched,
        "every message the finder matched must land in exactly one bucket"
    );
    assert_eq!(
        walked,
        skipped_global_init
            + skipped_no_body
            + lowered_ok
            + err_not_found
            + err_schema
            + err_unsupported,
        "every walked function must land on exactly one ladder rung"
    );
}

/// Classification of where a Call/Assert/Drop `on_unwind` target leads,
/// after following any `Goto` chain to its eventual terminator.
#[derive(Default)]
struct UnwindTally {
    /// Total Call/Assert/Drop terminators inspected.
    total_call_terms: usize,
    /// Per terminator-kind ("Call"/"Assert"/"Drop") count.
    by_term_kind: BTreeMap<&'static str, usize>,
    /// Eventual-terminator taxonomy after following the Goto chain.
    eventual: BTreeMap<&'static str, usize>,
    /// How many `on_unwind` targets' goto-chains execute *any*
    /// non-trivial statement before reaching UnwindResume/Abort.
    real_work: usize,
    /// How many of those non-trivial chains are destructor cleanup: the
    /// source terminator is a `Drop` and the chain does not end at a
    /// `Call` or a `Return`.
    real_work_destructor_cleanup: usize,
    /// fn::bb examples of real-work unwind chains (capped).
    examples: Vec<String>,
    /// Goto-chain length histogram (0 = on_unwind target is itself the
    /// terminating block, 1 = one Goto hop, …).
    chain_len_hist: BTreeMap<usize, usize>,
}

/// Does this statement do non-trivial (catch-like) work? Storage
/// markers, place mentions, and inline overflow-`Assert`s are trivial
/// bookkeeping; an `Assign` whose rvalue is anything other than a plain
/// `Use` (move/copy/const) or a `Ref`/`RawPtr` is treated as real work,
/// and any unknown statement (e.g. `SetDiscriminant`, `Deinit`) is real
/// work too.
fn stmt_is_real_work(stmt: &majit_charon_reader::Statement) -> bool {
    match stmt.stmt_kind() {
        Ok(StmtKind::StorageLive(_))
        | Ok(StmtKind::StorageDead(_))
        | Ok(StmtKind::PlaceMention(_))
        | Ok(StmtKind::Assert(_))
        | Ok(StmtKind::Borrowck(_)) => false,
        Ok(StmtKind::Assign(_, rv)) => !matches!(
            rv,
            Rvalue::Use(_, _) | Rvalue::Ref { .. } | Rvalue::RawPtr { .. }
        ),
        // Unknown statement kind (SetDiscriminant, Deinit, …) — treat
        // as real work so we never under-count.
        Ok(StmtKind::Unknown) => true,
        // Failed to project — be conservative, count as real work so it
        // shows up as an example to inspect.
        Err(_) => true,
    }
}

/// Follow the `on_unwind` goto-chain from `start_bb`, classifying the
/// eventual terminator and whether any block on the chain does real
/// work. Returns `(eventual_kind, did_real_work, drop_in_chain,
/// chain_len)`.
fn classify_unwind_chain(
    llbc: &Llbc,
    blocks: &[BasicBlock],
    start_bb: usize,
) -> (&'static str, bool, bool, usize) {
    let mut cur = start_bb;
    let mut did_real_work = false;
    // NOTE: this flag is only observable on the arms that keep WALKING.
    // The `Call`, `Return` and `Switch` arms return the instant they are
    // reached, so they hand back whatever it held before the walk stopped
    // — structurally `false` for a zero-hop chain, whatever the source
    // terminator was. Do not use it as an exemption predicate; the caller
    // keys the exemption on the source terminator and the eventual, both
    // of which it already knows.
    let mut drop_in_chain = false;
    let mut hops = 0usize;
    // Bound the walk so a malformed/cyclic chain can't hang the test.
    for _ in 0..64 {
        let Some(block) = blocks.get(cur) else {
            return ("oob-block", did_real_work, drop_in_chain, hops);
        };
        // Any non-trivial statement in this block is catch-like work.
        if block.statements.iter().any(stmt_is_real_work) {
            did_real_work = true;
        }
        match block.term(llbc) {
            Ok(TermKind::UnwindResume) => {
                return ("UnwindResume", did_real_work, drop_in_chain, hops);
            }
            Ok(TermKind::UnwindTerminate) => {
                return ("UnwindTerminate", did_real_work, drop_in_chain, hops);
            }
            Ok(TermKind::Abort(_)) | Ok(TermKind::Panic { .. }) => {
                return ("Abort", did_real_work, drop_in_chain, hops);
            }
            Ok(TermKind::UndefinedBehavior) => {
                return ("UndefinedBehavior", did_real_work, drop_in_chain, hops);
            }
            Ok(TermKind::Return) => {
                // An on_unwind path that *returns* would be genuine
                // cleanup-then-return; flag it as work.
                return ("Return", true, drop_in_chain, hops);
            }
            Ok(TermKind::Goto { target }) => {
                cur = target as usize;
                hops += 1;
                continue;
            }
            Ok(TermKind::Drop { target, .. }) => {
                // A Drop on the unwind path is a destructor call — the
                // chain continues to its own `target` (the next cleanup
                // step). This is "work" only in the destructor sense.
                drop_in_chain = true;
                cur = target as usize;
                hops += 1;
                continue;
            }
            Ok(TermKind::Call { .. }) => {
                // A real Call on the unwind path = catch-like work
                // (e.g. a cleanup routine that itself calls a fn).
                return ("Call", true, drop_in_chain, hops);
            }
            Ok(TermKind::Assert { target, .. }) => {
                // Inline assert on the unwind path; keep following.
                cur = target as usize;
                hops += 1;
                continue;
            }
            Ok(TermKind::Switch { .. }) => {
                // A branch on the unwind path is genuine control flow
                // (cleanup that inspects state) — flag as work.
                return ("Switch", true, drop_in_chain, hops);
            }
            Ok(TermKind::Unknown) | Err(_) => {
                return ("other/unknown-term", did_real_work, drop_in_chain, hops);
            }
        }
    }
    ("chain-too-long", true, drop_in_chain, hops)
}

#[test]
#[ignore = "requires the 205MB pyre-interpreter.ullbc snapshot; \
            set MAJIT_MIR_STRESS_LLBC"]
fn mir_on_unwind_target_taxonomy() {
    let Some(path) = stress_path() else {
        eprintln!(
            "skip: set MAJIT_MIR_STRESS_LLBC or run scripts/extract-llbc.py to make \
             target/llbc/pyre-interpreter.ullbc available"
        );
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");

    let mut tally = UnwindTally::default();

    for fd in llbc.iter_local_fns() {
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let blocks = &body.body;
        for (bb_idx, block) in blocks.iter().enumerate() {
            let (term_kind_label, on_unwind): (&'static str, u64) = match block.term(&llbc) {
                Ok(TermKind::Call { on_unwind, .. }) => ("Call", on_unwind),
                Ok(TermKind::Assert { on_unwind, .. }) => ("Assert", on_unwind),
                Ok(TermKind::Drop { on_unwind, .. }) => ("Drop", on_unwind),
                _ => continue,
            };
            tally.total_call_terms += 1;
            *tally.by_term_kind.entry(term_kind_label).or_default() += 1;

            let (eventual, did_work, drop_in_chain, chain_len) =
                classify_unwind_chain(&llbc, blocks, on_unwind as usize);
            *tally.eventual.entry(eventual).or_default() += 1;
            *tally.chain_len_hist.entry(chain_len).or_default() += 1;
            // A `Drop` terminator's unwind edge IS the destructor-cleanup
            // continuation — precisely what this exemption exists for —
            // while a chain ending in a `Call` or a `Return` may be a
            // handler, which is the thing the assertion below exists to
            // catch. So exempt on the SOURCE terminator plus the eventual,
            // never on `drop_in_chain`: that flag is written only while
            // walking and is unreachable at the arms that return at once.
            //
            // `Switch` is the third case and it splits. `classify_unwind_chain`
            // calls it "cleanup that inspects state", and inside drop glue that
            // is what it is: dropping an array branches over its elements, and
            // this corpus holds exactly one such chain
            // (`<array>::<Impl>::drop_in_place`, measured). Outside drop glue a
            // branch on the unwind path is the catch-like control flow the
            // assertion below exists to catch, so the exemption is withheld
            // there. Keying on the drop-glue symbol is a proxy; it is the one
            // the corpus actually distinguishes, and a `Switch` appearing under
            // any other name is reported rather than absorbed.
            let in_drop_glue = fd.item_meta.name_path().contains("drop_in_place");
            let destructor_cleanup = term_kind_label == "Drop"
                && !matches!(eventual, "Call" | "Return")
                && (eventual != "Switch" || in_drop_glue);
            if did_work {
                tally.real_work += 1;
                if destructor_cleanup {
                    tally.real_work_destructor_cleanup += 1;
                }
                if tally.examples.len() < 40 {
                    tally.examples.push(format!(
                        "{}::bb{bb_idx} [{term_kind_label}] on_unwind=bb{on_unwind} \
                         -> {eventual} (chain_len={chain_len}, drop_in_chain={drop_in_chain})",
                        fd.item_meta.name_path()
                    ));
                }
            }
        }
    }

    eprintln!("\n=== on_unwind target taxonomy (whole interpreter) ===");
    eprintln!("path: {}", path.display());
    eprintln!(
        "total Call/Assert/Drop terminators inspected: {}",
        tally.total_call_terms
    );
    eprintln!("\nby terminator kind:");
    for (k, n) in &tally.by_term_kind {
        eprintln!("  {n:>8}  {k}");
    }
    eprintln!("\neventual terminator of on_unwind goto-chain:");
    let mut ev: Vec<_> = tally.eventual.iter().collect();
    ev.sort_by(|a, b| b.1.cmp(a.1));
    for (k, n) in ev {
        eprintln!("  {n:>8}  {k}");
    }
    eprintln!("\ngoto-chain length histogram (hops to terminator):");
    for (len, n) in &tally.chain_len_hist {
        eprintln!("  len={len:>2}  {n:>8}");
    }
    eprintln!("\nany_handler_does_real_work: {}", tally.real_work > 0);
    eprintln!(
        "  chains doing real (non-trivial) work: {}",
        tally.real_work
    );
    eprintln!(
        "    of which are destructor cleanup (Drop source, non-Call/Return \
         eventual): {}",
        tally.real_work_destructor_cleanup
    );
    eprintln!(
        "    non-destructor real-work chains (genuine catch suspects): {}",
        tally.real_work - tally.real_work_destructor_cleanup
    );
    if !tally.examples.is_empty() {
        eprintln!("\nreal-work examples (capped at 40):");
        for ex in &tally.examples {
            eprintln!("  {ex}");
        }
    }

    // The decisive number to read is `non-destructor real-work chains` —
    // if that is 0, every on_unwind path is a bare panic-propagation
    // (UnwindResume/Abort) or pure destructor cleanup, and dropping it
    // loses no try/except.
    assert!(
        tally.total_call_terms > 0,
        "expected at least one Call/Assert/Drop terminator in the snapshot"
    );
    // The decisive invariant — no on_unwind path does catch-like work (a
    // Call/Return, a Switch outside drop glue, or a non-trivial statement)
    // other than pure destructor drop-glue. If this ever trips, the
    // corpus grew a Rust catch/cleanup that the front-graph driver would
    // silently drop, and the "drop on_unwind" adaptation must be revisited.
    let non_destructor_real_work = tally.real_work - tally.real_work_destructor_cleanup;
    assert_eq!(
        non_destructor_real_work, 0,
        "found {non_destructor_real_work} on_unwind chain(s) doing non-destructor \
         catch-like work; dropping on_unwind would lose semantics — see \
         examples above"
    );
}

#[test]
#[ignore = "requires the 205MB pyre-interpreter.ullbc snapshot; \
            set MAJIT_MIR_STRESS_LLBC"]
fn coverage_gate_accepts_the_real_snapshot() {
    // The fail-loud coverage gate in `build_semantic_program_from_llbc`
    // must return `Ok` over the real snapshot: every current lowering
    // skip is the tracked uninitialised-local gap, so none is classified
    // a regression. If a future change makes a body fail to lower with an
    // unrecognised error, this builder returns `Err` and the build
    // (and this test) fails loudly instead of silently dropping the fn.
    let Some(path) = stress_path() else {
        eprintln!("skip: set MAJIT_MIR_STRESS_LLBC");
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");
    majit_translate::front::mir::build_semantic_program_from_llbc(&llbc).expect(
        "coverage gate must accept the real snapshot — all skips are the \
         tracked uninitialised-local gap",
    );
}
