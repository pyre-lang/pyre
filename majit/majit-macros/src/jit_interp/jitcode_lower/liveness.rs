use super::*;

/// Per-marker live set produced by `compute_per_marker_liveness`.
/// Index aligns with the order in which `LiveMarker` ops appear in
/// `op_metadata`. Each entry is a single `BTreeSet<Register>` matching
/// RPython `liveness.py`'s `set()` of `Register` objects — bank info
/// rides on `Register.kind` and the encoder splits at emit time per
/// `assembler.py get_liveness_info(args, kind)`.
#[allow(dead_code)]
type LiveMarkerLiveSets = Vec<BTreeSet<Register>>;
/// Compute the live register set captured at every `LiveMarker` op in
/// `op_metadata`, mirroring RPython
/// `rpython/jit/codewriter/liveness.py:33-79
/// _compute_liveness_must_continue`.
///
/// The walk is backward (def `discard`, use `add`); branch ops fold in
/// the destination label's accumulated alive set; label definitions
/// store the current alive set for forward jumps to consume on the
/// next iteration. Iterations continue until no label or marker entry
/// changes (fixed-point), matching RPython's `must_continue` loop.
///
/// Returned `Vec<BTreeSet<Register>>` is indexed in `LiveMarker`
/// encounter order, so callers can pair entries with their
/// `live_placeholder()` emit sites.
#[allow(dead_code)]
pub(super) fn compute_per_marker_liveness(op_metadata: &[OpMeta]) -> LiveMarkerLiveSets {
    let marker_indices: Vec<usize> = op_metadata
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.control, ControlFlowClass::LiveMarker))
        .map(|(i, _)| i)
        .collect();

    let mut label_alive: HashMap<String, BTreeSet<Register>> = HashMap::new();
    let mut live_at_marker: HashMap<usize, BTreeSet<Register>> = HashMap::new();

    loop {
        let mut changed = false;
        let mut alive: BTreeSet<Register> = BTreeSet::new();

        for i in (0..op_metadata.len()).rev() {
            let op = &op_metadata[i];
            match op.control {
                ControlFlowClass::LiveMarker => {
                    // RPython liveness.py:44-53 — `-live-` first folds in
                    // any explicit force-alive register args and any
                    // TLabel target's accumulated alive set, then records
                    // the resulting alive at this marker. The mutation
                    // also propagates upstream so the registers / labels
                    // the marker keeps alive stay alive in earlier ops.
                    for target in &op.live_target_labels {
                        let name = target.to_string();
                        if let Some(s) = label_alive.get(&name) {
                            alive.extend(s.iter().copied());
                        }
                    }
                    alive.extend(op.reads.iter().copied());
                    let prev = live_at_marker.get(&i);
                    if prev.is_none() || prev.unwrap() != &alive {
                        live_at_marker.insert(i, alive.clone());
                        changed = true;
                    }
                }
                ControlFlowClass::LabelDef => {
                    // RPython liveness.py:36-42 — record alive against
                    // the label name (union with prior iterations).
                    let name = op
                        .target_label
                        .as_ref()
                        .expect("label_def needs target")
                        .to_string();
                    let entry = label_alive.entry(name).or_default();
                    let before = entry.len();
                    entry.extend(alive.iter().copied());
                    if entry.len() != before {
                        changed = true;
                    }
                }
                ControlFlowClass::UnconditionalJump => {
                    // RPython follow_label (liveness.py) — `alive`
                    // becomes the label's accumulated set (overwrite,
                    // not union, since fall-through past `jump` is
                    // unreachable).
                    let name = op
                        .target_label
                        .as_ref()
                        .expect("jump needs target")
                        .to_string();
                    alive = label_alive.get(&name).cloned().unwrap_or_default();
                }
                ControlFlowClass::ConditionalGuard => {
                    // Def-first (`alive.discard`), matching Linear: a fused
                    // branch-op that also writes a result (e.g.
                    // `int_*_jump_if_ovf`, whose `dst` is produced on the
                    // fall-through) must kill that def so the not-yet-written
                    // register does not survive into the preceding `-live-`
                    // guard snapshot.  Pure `goto_if_not_*` guards carry no
                    // writes, so this is a no-op for them.
                    for w in &op.writes {
                        alive.remove(w);
                    }
                    // Fold the branch target's alive set into the
                    // fall-through alive set, then add the cond_reg(s)
                    // as uses. RPython treats `goto_if_not` as a
                    // normal op whose TLabel arg triggers
                    // follow_label (alive update) and whose register
                    // args (cond) become uses.
                    if let Some(target) = op.target_label.as_ref() {
                        let name = target.to_string();
                        if let Some(s) = label_alive.get(&name) {
                            alive.extend(s.iter().copied());
                        }
                    }
                    for r in &op.reads {
                        alive.insert(*r);
                    }
                }
                ControlFlowClass::Linear => {
                    // RPython liveness.py:60-69 — def first
                    // (`alive.discard(reg)`) then uses (`alive.add(x)`).
                    for w in &op.writes {
                        alive.remove(w);
                    }
                    for r in &op.reads {
                        alive.insert(*r);
                    }
                }
                ControlFlowClass::Terminal => {
                    // `*_return` — no successor, no fall-through. Reset
                    // the alive set (nothing is alive past a return) and
                    // then add this op's own reads so the returned value
                    // stays alive upstream.
                    alive.clear();
                    for r in &op.reads {
                        alive.insert(*r);
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    marker_indices
        .iter()
        .map(|i| live_at_marker.remove(i).unwrap_or_default())
        .collect()
}

/// Encode-time bank split, mirroring RPython
/// `rpython/jit/codewriter/assembler.py get_liveness_info(args,
/// kind)`. Walks a marker's accumulated alive set and projects out the
/// indices belonging to a single bank, producing the per-bank u8 vector
/// the BC_LIVE encoder consumes (`assembler.py:147-157` writes the
/// `(live_i, live_r, live_f)` triple as three sorted bitsets).
///
/// The walker (`compute_per_marker_liveness`) keeps a single
/// `BTreeSet<Register>` per marker so that the analysis stays
/// structurally identical to RPython's `set()` of `Register` objects;
/// the bank split is deferred to this helper at emit time.
///
/// `BTreeSet<Register>` already iterates in `(kind, index)` order due
/// to `Register`'s derived `Ord`, so the resulting `Vec<u8>` is sorted
/// — matching `assembler.py:148 live = sorted(live)`.
#[allow(dead_code)]
pub(super) fn get_liveness_info(set: &BTreeSet<Register>, kind: BindingKind) -> Vec<u8> {
    set.iter()
        .filter(|r| r.kind == kind)
        .map(|r| r.index)
        .collect()
}

/// Convenience: return the `(live_i, live_r, live_f)` triple sourced
/// from `set`. Used by `maybe_dump_liveness` and by the BC_LIVE
/// per-marker patcher (`live_placeholder_with_triple` consumers added
/// at emit time).
#[allow(dead_code)]
pub(super) fn liveness_triple(set: &BTreeSet<Register>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        get_liveness_info(set, BindingKind::Int),
        get_liveness_info(set, BindingKind::Ref),
        get_liveness_info(set, BindingKind::Float),
    )
}

/// Same as [`liveness_triple`] but consuming a typed register slice
/// (post-`annotate_live_markers_with_liveness` `LiveMarker.reads`).
/// Mirrors RPython `assembler.py get_liveness_info(args, kind)`
/// applied to the marker's args directly, which by then are the full
/// alive set per `liveness.py:52`.
#[allow(dead_code)]
pub(super) fn liveness_triple_from_reads(reads: &[Register]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut live_i = Vec::new();
    let mut live_r = Vec::new();
    let mut live_f = Vec::new();
    for reg in reads {
        match reg.kind {
            BindingKind::Int => live_i.push(reg.index),
            BindingKind::Ref => live_r.push(reg.index),
            BindingKind::Float => live_f.push(reg.index),
        }
    }
    (live_i, live_r, live_f)
}

/// RPython `compute_liveness(ssarepr)` mutates each `-live-` instruction
/// (`liveness.py ssarepr.insns[i] = insn[:1] + tuple(alive) + tuple(labels)`)
/// before `remove_repeated_live(ssarepr)` runs. Mirror that order by
/// materialising the fixed-point alive set back onto each `LiveMarker`'s
/// `reads` operand; the repeated-live pass and the emit-time triple
/// rewrite both consume the ssarepr-mutated shape directly.
pub(super) fn annotate_live_markers_with_liveness(op_metadata: &mut [OpMeta]) {
    let live_sets = compute_per_marker_liveness(op_metadata);
    let mut next_marker = 0usize;
    for meta in op_metadata.iter_mut() {
        if !matches!(meta.control, ControlFlowClass::LiveMarker) {
            continue;
        }
        meta.reads = live_sets[next_marker].iter().copied().collect();
        next_marker += 1;
    }
    debug_assert_eq!(
        next_marker,
        live_sets.len(),
        "compute_per_marker_liveness output count must match LiveMarker op_metadata entries"
    );
}

/// Generate the per-marker liveness prebuild tokens that
/// `__prebuild_jitcode_liveness_*` (codegen_trace.rs) replays into the
/// driver-shared `Assembler` at install time. Each `LiveMarker` op
/// emits an `__asm._register_liveness_offset(&[live_i], &[live_r],
/// &[live_f])` call so RPython `pyjitpl.py finish_setup` order is
/// preserved: every per-marker triple lands in `asm.all_liveness`
/// before `metainterp_sd.liveness_info` snapshots it. Trace-time
/// `JitCodeBuilder::finalize_liveness` then only dedups against the
/// pre-registered offsets, never grows the table past the snapshot.
///
/// `inline_prebuild` carries any nested-helper prebuild tokens that
/// were aggregated during lowering.  Emit the current body's own
/// `-live-` triples first, then nested helper prebuilds: RPython
/// `codewriter.py make_jitcodes` assembles the caller graph that discovered
/// an inline callee before draining the pending callee graph queued by
/// `call.py get_jitcode`.
pub(super) fn liveness_prebuild_tokens(
    op_metadata: &[OpMeta],
    inline_prebuild: &[TokenStream],
) -> TokenStream {
    let live_regs = op_metadata.iter().filter_map(|m| {
        if !matches!(m.control, ControlFlowClass::LiveMarker) {
            return None;
        }
        let (live_i, live_r, live_f) = liveness_triple_from_reads(&m.reads);
        let live_i_tokens = live_i_tokens(&live_i);
        let register = quote! {
            let _ = __asm._register_liveness_offset(
                #live_i_tokens,
                &[#(#live_r),*],
                &[#(#live_f),*],
            );
        };
        Some(if let Some(condition) = m.live_condition.as_ref() {
            let condition = condition.emit.clone();
            quote! {
                if #condition {
                    #register
                }
            }
        } else {
            register
        })
    });
    quote! {
        #(#live_regs)*
        #(#inline_prebuild)*
    }
}

/// Collapse runs of consecutive `LiveMarker` ops (and any intervening
/// `LabelDef` ops) into a single `LiveMarker`, mirroring RPython
/// `rpython/jit/codewriter/liveness.py remove_repeated_live`.
///
/// The lowerer currently never emits markers in succession (each
/// `live_placeholder()` site sits in front of a guard / call op so a
/// non-marker non-label always intervenes), making this function a
/// structural no-op for present `#[jit_interp]` consumers. It still
/// runs end-to-end so future lowerers (or post-processing passes that
/// inject extra markers around inline-call boundaries) inherit the
/// RPython collapse semantics for free.
///
/// `op_metadata` and `statements` must stay index-aligned; both vectors
/// are mutated in lockstep.
#[allow(dead_code)]
pub(super) fn remove_repeated_live(
    op_metadata: &mut Vec<OpMeta>,
    statements: &mut Vec<TokenStream>,
) {
    debug_assert_eq!(op_metadata.len(), statements.len());
    let mut new_meta: Vec<OpMeta> = Vec::with_capacity(op_metadata.len());
    let mut new_stmts: Vec<TokenStream> = Vec::with_capacity(statements.len());
    let mut i = 0;
    while i < op_metadata.len() {
        if !matches!(op_metadata[i].control, ControlFlowClass::LiveMarker) {
            new_meta.push(op_metadata[i].clone());
            new_stmts.push(statements[i].clone());
            i += 1;
            continue;
        }
        // Collect the run of consecutive markers (separated by label
        // definitions only).
        let first_marker_idx = i;
        let mut markers: Vec<usize> = vec![i];
        let mut interleaved_labels: Vec<usize> = Vec::new();
        i += 1;
        while i < op_metadata.len() {
            match op_metadata[i].control {
                ControlFlowClass::LiveMarker => {
                    markers.push(i);
                    i += 1;
                }
                ControlFlowClass::LabelDef => {
                    interleaved_labels.push(i);
                    i += 1;
                }
                _ => break,
            }
        }
        if markers.len() == 1 {
            // RPython `remove_repeated_live`: labels collected after a lone
            // `-live-` still move in front of it.  A jump to one of those
            // labels must execute the liveness snapshot too; retaining the
            // original `live, label` order lets that entry path skip it.
            for li in &interleaved_labels {
                new_meta.push(op_metadata[*li].clone());
                new_stmts.push(statements[*li].clone());
            }
            new_meta.push(op_metadata[first_marker_idx].clone());
            new_stmts.push(statements[first_marker_idx].clone());
            continue;
        }
        // TODO: `liveness.py remove_repeated_live`
        // unions the `reads` of every marker in the run because every
        // upstream marker actually fires (RPython has no conditional
        // emission).  pyre's `live_marker_if` markers exist or not at
        // runtime depending on the helper-policy byte
        // (`__majit_call_policy_<name>()`), so unioning their reads here
        // would over-capture: when only one condition holds at runtime,
        // the merged BC_LIVE would still pin the union of the
        // would-have-fired siblings' alive sets.  When the run contains
        // any unconditional marker the merged BC_LIVE is guaranteed to
        // fire (PyPy parity), so unioning is safe and the merged marker
        // becomes unconditional.  When every marker is conditional, fall
        // back to keeping them unmerged — each emits its own BC_LIVE
        // only when its own condition holds, matching PyPy's per-site
        // alive-set capture (at the cost of skipping `liveness.py`'s
        // dedup, which `production` doesn't trigger anyway because the
        // lowerer emits at most one marker per call/guard site).
        if markers
            .iter()
            .all(|mi| op_metadata[*mi].live_condition.is_some())
        {
            for idx in first_marker_idx..i {
                new_meta.push(op_metadata[idx].clone());
                new_stmts.push(statements[idx].clone());
            }
            continue;
        }
        // Multiple markers with at least one unconditional: union their
        // `reads` registers per RPython `liveness.py:111-115
        // liveset.update(live[1:])`.  Union typed Register reads as a
        // single bag (Ord = (kind, index)) and union
        // `live_target_labels` separately.  Result is unconditional —
        // the unconditional sibling forces emit, so the conditional
        // siblings' reads fold in at this fully-fired position.
        let mut merged_reads: Vec<Register> = Vec::new();
        let mut merged_labels: Vec<Ident> = Vec::new();
        for mi in &markers {
            let m = &op_metadata[*mi];
            merged_reads.extend(m.reads.iter().copied());
            merged_labels.extend(m.live_target_labels.iter().cloned());
        }
        merged_reads.sort();
        merged_reads.dedup();
        merged_labels.sort_by_key(|label| label.to_string());
        merged_labels.dedup_by_key(|label| label.to_string());
        let merged_marker = OpMeta::live_marker_with(merged_reads, merged_labels);
        for li in &interleaved_labels {
            new_meta.push(op_metadata[*li].clone());
            new_stmts.push(statements[*li].clone());
        }
        new_meta.push(merged_marker);
        // Reuse the first marker's statement token (a single
        // `live_placeholder()` call); the duplicated runs don't survive
        // the collapse since RPython prints just one `-live-` for the
        // whole run.  `rewrite_live_marker_statements_with_triples`
        // (later pass) overwrites the body — the merged marker's
        // `live_condition` is `None` so the rewrite emits an
        // unconditional `live_placeholder_with_triple(...)`.
        new_stmts.push(statements[first_marker_idx].clone());
    }
    *op_metadata = new_meta;
    *statements = new_stmts;
}

/// Emit-time bridge: replace each `LiveMarker`
/// statement's `live_placeholder()` call with the triple-aware
/// `live_placeholder_with_triple(&[live_i...], &[live_r...], &[live_f...])`
/// shape, sourcing the per-marker triples from
/// [`compute_per_marker_liveness`] split per bank by [`liveness_triple`]
/// (mirrors `assembler.py get_liveness_info(args, kind)`).
///
/// Runs after [`remove_repeated_live`] so the marker count seen by the
/// walker matches the number of statements that actually survive into
/// the lowered output.
///
/// The runtime effect is no-op until the dispatch JitCode builder
/// calls `JitCodeBuilder::finalize_liveness(&mut asm)` — until then,
/// `pending_live_triples` accumulates per-builder records but the
/// emitted `live/<00 00>` slot stays at offset 0, identical to the
/// `live_placeholder()` shape it replaces.  `finalize_liveness` runs
/// against the driver-shared `Arc<Mutex<Assembler>>` snapshotted by
/// `install_canonical_liveness` at install time.
///
/// Each register index must fit in `u8` per RPython
/// `rpython/jit/codewriter/assembler.py` — the bitset encoder
/// only addresses 0..=255 (8 register-bytes × 8 bits). The typed
/// `Register::new` constructor asserts this bound at every
/// emit site, so by the time the walker hands us a `BTreeSet<Register>`
/// the indices are guaranteed `u8`-clean.
fn live_i_tokens(live_i: &[u8]) -> TokenStream {
    quote! { &[#(#live_i),*] }
}

pub(super) fn rewrite_live_marker_statements_with_triples(
    op_metadata: &[OpMeta],
    statements: &mut [TokenStream],
) {
    debug_assert_eq!(op_metadata.len(), statements.len());
    let live_sets = compute_per_marker_liveness(op_metadata);
    let mut next_marker = 0usize;
    for (i, m) in op_metadata.iter().enumerate() {
        if !matches!(m.control, ControlFlowClass::LiveMarker) {
            continue;
        }
        let (live_i, live_r, live_f) = liveness_triple(&live_sets[next_marker]);
        next_marker += 1;
        let live_i_tokens = live_i_tokens(&live_i);
        let live_stmt = quote! {
            let _ = __builder.live_placeholder_with_triple(
                #live_i_tokens,
                &[#(#live_r),*],
                &[#(#live_f),*],
            );
        };
        statements[i] = if let Some(condition) = m.live_condition.as_ref() {
            let condition = condition.emit.clone();
            quote! {
                if #condition {
                    #live_stmt
                }
            }
        } else {
            live_stmt
        };
    }
    debug_assert_eq!(
        next_marker,
        live_sets.len(),
        "compute_per_marker_liveness output count must match LiveMarker op_metadata entries"
    );
}

/// Print per-marker live sets to stderr when `MAJIT_DUMP_LIVENESS` is
/// set in the proc-macro build environment. `label` is the lowerer
/// scope being dumped (e.g. helper name) so concurrent expansions are
/// distinguishable.
pub(super) fn maybe_dump_liveness(label: &str, op_metadata: &[OpMeta]) {
    if std::env::var("MAJIT_DUMP_LIVENESS").is_err() {
        return;
    }
    let live_sets = compute_per_marker_liveness(op_metadata);
    let marker_count = op_metadata
        .iter()
        .filter(|m| matches!(m.control, ControlFlowClass::LiveMarker))
        .count();
    eprintln!(
        "=== majit liveness dump [{}] op_metadata={} markers={} ===",
        label,
        op_metadata.len(),
        marker_count
    );
    for (idx, set) in live_sets.iter().enumerate() {
        let (live_i, live_r, live_f) = liveness_triple(set);
        eprintln!(
            "  marker[{}] live_i={:?} live_r={:?} live_f={:?}",
            idx, live_i, live_r, live_f,
        );
    }
}
