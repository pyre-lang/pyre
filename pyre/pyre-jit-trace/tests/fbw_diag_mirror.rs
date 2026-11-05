//! Keeps `pyre-wasm-runner`'s positional fbw label array synchronized with
//! `pyre_jit_trace::trace::fbw_diag::LABELS`.
//!
//! The runner links no pyre crate — it is a wasmtime host that reaches the
//! counters through the `pyre_fbw_diag` wasm export — so it cannot import the
//! authority and instead restates it as a positional array. Each side sizes
//! its own value array from its own labels, which catches a slot added
//! without a name; neither compiler sees the SPELLINGS, so a rename drifts
//! silently and every tally from the divergence onward is printed under the
//! wrong key. Hence this test: the authority is LINKED (`fbw_diag::LABELS` is
//! in scope), the mirror is read out of the runner's source as text, the same
//! way `majit-metainterp/tests/runner_label_mirrors.rs` does for `MC_DIAG_LABELS`.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // `<root>/pyre/pyre-jit-trace` -> `<root>`
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("CARGO_MANIFEST_DIR has fewer than two ancestors")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    // A missing file must FAIL rather than skip. This test's whole job is to
    // compare two files; one that cannot find them and passes anyway is the
    // failure mode it exists to prevent, one level up.
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {} — {e}", path.display()))
}

/// The string literals of the first array literal following `anchor`.
///
/// A declaration can carry a `[` in its TYPE, so scanning for the first `[`
/// after the name may land there and yield zero entries. Anchor on the name,
/// then step to the `= [` that opens the value.
fn array_after(text: &str, anchor: &str, what: &str) -> Vec<String> {
    let at = text
        .find(anchor)
        .unwrap_or_else(|| panic!("{what}: anchor {anchor:?} not found"));
    let eq = text[at..]
        .find("= [")
        .unwrap_or_else(|| panic!("{what}: no `= [` after anchor {anchor:?}"))
        + at;
    let open = eq + text[eq..].find('[').expect("`= [` contains a `[`");

    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut end = open;
    for (i, b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(depth == 0 && end > open, "{what}: unbalanced array literal");

    let mut out = Vec::new();
    for line in text[open + 1..end].lines() {
        // Whole-line comments are dropped so a `"` inside prose cannot be
        // mistaken for an entry. Entries are one per line in both files.
        let line = line.trim();
        if line.starts_with("//") {
            continue;
        }
        let mut rest = line;
        while let Some(s) = rest.find('"') {
            let after = &rest[s + 1..];
            let Some(e) = after.find('"') else { break };
            out.push(after[..e].to_string());
            rest = &after[e + 1..];
        }
    }
    out
}

fn runner_source() -> String {
    read(&repo_root().join("pyre/pyre-wasm-runner/src/main.rs"))
}

/// The linked authority and the runner's textual mirror of it.
///
/// The authority is not parsed: this test links `pyre-jit-trace`, so
/// `LABELS` itself is in scope and no anchor can select the wrong array on
/// that side. Only the runner — which this crate cannot import — is read as
/// text.
fn extract(runner_src: &str) -> (&'static [&'static str], Vec<String>) {
    (
        pyre_jit_trace::trace::fbw_diag::LABELS,
        array_after(runner_src, "let fbw_labels", "runner"),
    )
}

/// The integer a declaration evaluates to: decimal, `0x` hex, or `N << M`.
///
/// Hex is accepted because the bit-layout half of the mirror is only readable
/// as masks — a `FIELD_MASK` spelled `65535` to keep a decimal-only parser
/// happy would trade the thing being guarded for the guard. `N << M` is the
/// spelling `bitflags!` uses for a single bit, so a flag written that way
/// does not need a second decimal alias for this test to see it.
fn declared_int(src: &str, anchor: &str, what: &str) -> u64 {
    let at = src
        .find(anchor)
        .unwrap_or_else(|| panic!("{what}: {anchor:?} declaration not found"));
    let tail = src[at + anchor.len()..].trim_start();
    let expr: String = tail
        .chars()
        .take_while(|c| !matches!(*c, ';' | ',' | '\n' | '}'))
        .collect();
    parse_u64_expr(expr.trim())
        .unwrap_or_else(|| panic!("{what}: {anchor:?} is not a number or `N << M` ({expr:?})"))
}

fn parse_u64_expr(expr: &str) -> Option<u64> {
    if let Some((left, right)) = expr.split_once("<<") {
        let left = parse_u64_literal(left.trim())?;
        let right = parse_u64_literal(right.trim())?;
        return left.checked_shl(u32::try_from(right).ok()?);
    }
    parse_u64_literal(expr)
}

#[test]
fn declared_int_reads_shift_and_hex() {
    assert_eq!(
        declared_int("const X: u64 = 20;", "const X: u64 = ", "t"),
        20
    );
    assert_eq!(
        declared_int("const X: u64 = 0xffff;", "const X: u64 = ", "t"),
        0xffff
    );
    assert_eq!(
        declared_int("const VALID = 1 << 0;", "const VALID = ", "t"),
        1
    );
    assert_eq!(
        declared_int("const COMMITTED = 1 << 1;", "const COMMITTED = ", "t"),
        2
    );
    assert_eq!(
        declared_int("const BRIDGE = 1 << 2;", "const BRIDGE = ", "t"),
        4
    );
}

fn parse_u64_literal(s: &str) -> Option<u64> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

/// A `bitflags!` member under `type_name`, e.g. `FbwRingFlags` / `VALID`.
fn declared_bitflag(src: &str, type_name: &str, flag: &str, what: &str) -> u64 {
    let type_at = src
        .find(type_name)
        .unwrap_or_else(|| panic!("{what}: bitflags type {type_name:?} not found"));
    declared_int(&src[type_at..], &format!("const {flag} = "), what)
}

/// The slots whose two spellings disagree, over the shorter of the two.
fn divergences(authority: &[&str], runner: &[String]) -> Vec<String> {
    (0..authority.len().min(runner.len()))
        .filter(|&i| authority[i] != runner[i].as_str())
        .map(|i| {
            format!(
                "  slot {i}: fbw_diag={:?} runner={:?}",
                authority[i], runner[i]
            )
        })
        .collect()
}

#[test]
fn the_runner_mirror_matches_fbw_diag_labels() {
    let runner_src = runner_source();
    let (authority, runner) = extract(&runner_src);
    let slots = authority.len();

    assert_eq!(
        runner.len(),
        slots,
        "pyre-wasm-runner's fbw label array has {} entries but fbw_diag::LABELS \
         has {slots}. A tally slot was added to fbw_diag::LABELS without being \
         appended to the mirror in pyre-wasm-runner/src/main.rs (or vice versa). \
         APPEND it — inserting in the middle silently renames every tally after \
         the insertion point.\n\
         BUT FIRST: a count that is far off rather than off by a few means the \
         `let fbw_labels` anchor matched a different array, and the extraction is \
         at fault, not the mirror.",
        runner.len(),
    );

    let divergent = divergences(authority, &runner);
    assert!(
        divergent.is_empty(),
        "the pyre-wasm-runner fbw mirror no longer matches \
         pyre_jit_trace::trace::fbw_diag::LABELS.\n{}\n\
         The runner zips these labels against `pyre_fbw_diag(i)` BY INDEX, so \
         every tally from the first divergent slot onward is printed under the \
         wrong name — and check.py merges every `[jit-stats]` line into one flat \
         key->value map, so a wrong name is compared against the wrong baseline \
         rather than reported as missing. Fix the mirror in \
         pyre-wasm-runner/src/main.rs to match pyre-jit-trace, which is the \
         authority: `RING_BASE` — the offset the counter array itself is laid \
         out against — is that array's own length.",
        divergent.join("\n"),
    );
}

/// The text comparison above proves the two ARRAYS agree; it cannot prove
/// either one agrees with the slot CONSTANTS, which is what the bump sites
/// actually index by. Both arrays staying in step while `LABELS` shifts
/// against `WALKS`/`MIDBODY_LATCH`/... would rename every tally on both
/// backends at once, which no diff between them can see. Bind each label to
/// its own constant.
///
/// Two entries deliberately break the mechanical `ESCAPE_*` -> `fbw_escape_*`
/// reading — `ESCAPE_FORCE_BY_PORTAL` is `fbw_force_by_portal`, not
/// `fbw_escape_force_by_portal` — which is why this is written out rather than
/// derived from the constant names.
#[test]
fn each_label_sits_at_its_own_slot_constant() {
    use pyre_jit_trace::trace::fbw_diag as d;

    let bindings = [
        (d::WALKS, "fbw_walks"),
        (d::ROLLED_BACK_WITH_EFFECTS, "fbw_rolled_back_with_effects"),
        (d::MIDBODY_LATCH, "fbw_midbody_latch"),
        (
            d::MIDBODY_LATCH_NEW_UNJOURNALED,
            "fbw_midbody_latch_new_unjournaled",
        ),
        (d::ESCAPE_PLAIN_FALLBACK, "fbw_escape_plain_fallback"),
        (
            d::ESCAPE_PLAIN_FALLBACK_UNCLEAN,
            "fbw_escape_plain_fallback_unclean",
        ),
        (d::ESCAPE_PORTAL_ONLY, "fbw_escape_portal_only"),
        (
            d::ESCAPE_PUBLISHED_CALLEE_ONLY,
            "fbw_escape_published_callee_only",
        ),
        (
            d::ESCAPE_PORTAL_AND_PUBLISHED_CALLEE,
            "fbw_escape_portal_and_published_callee",
        ),
        (d::ESCAPE_FORCE_BY_PORTAL, "fbw_force_by_portal"),
        (d::ESCAPE_FORCE_BY_CALLEE_ONLY, "fbw_force_by_callee_only"),
        (
            d::STORE_JOURNAL_ROLLBACK_FAILED,
            "fbw_store_journal_rollback_failed",
        ),
        (
            d::BLACKHOLE_ADOPTED_SINGLE_FRAME,
            "fbw_blackhole_adopted_single_frame",
        ),
        (
            d::BLACKHOLE_ADOPTED_MULTI_FRAME,
            "fbw_blackhole_adopted_multi_frame",
        ),
        (d::GATE_DECLINED_SHAPE, "gate_declined_shape"),
        (d::BRIDGE_EC_FROM_PORTAL_RED, "bridge_ec_from_portal_red"),
        (d::BRIDGE_EC_MISSING, "bridge_ec_missing"),
        (d::FORITER_ITEM_DROPPED, "fbw_foriter_item_dropped"),
    ];

    // Exhaustiveness: a slot added to `fbw_diag` grows `LABELS` (rustc enforces
    // that) and the runner mirror (the length check above enforces that), but
    // neither would notice that its constant went unbound here. Require the
    // bound slots to be exactly `0..RING_BASE`.
    let mut bound: Vec<usize> = bindings.iter().map(|&(slot, _)| slot).collect();
    bound.sort_unstable();
    assert_eq!(
        bound,
        (0..d::RING_BASE).collect::<Vec<_>>(),
        "the slot constants bound below are not exactly 0..RING_BASE — a new \
         fbw_diag slot was added without binding its constant to its label here",
    );

    for (slot, expected) in bindings {
        assert_eq!(
            d::LABELS[slot],
            expected,
            "slot {slot} is named {:?} but the constant that indexes it means \
             {expected:?}. A slot constant was inserted or renumbered without \
             moving its entry in LABELS, so every bump from this slot on is \
             reported under another slot's key — on BOTH backends at once, \
             which is why the runner-mirror diff cannot see it.",
            d::LABELS[slot],
        );
    }
}

/// POSITIVE CONTROL — proves the checks above can fail.
///
/// A guard that has never been observed failing asserts nothing. Both drift
/// shapes are injected into the REAL runner source, in memory, and the checks
/// are required to catch them. Perturbing the real text rather than a
/// hand-written fixture is what makes this a control over the ACTUAL
/// extraction: a fixture would only exercise `divergences`, leaving the
/// anchors — the part that is easiest to get silently wrong — untested.
#[test]
fn the_mirror_check_catches_injected_drift() {
    let runner_src = runner_source();
    let (authority, runner) = extract(&runner_src);
    let slots = authority.len();
    assert_eq!(runner.len(), slots, "control needs a green starting tree");

    // Both injections edit only the text FROM the anchor onward, so a future
    // comment that happens to quote one of these keys ahead of the array
    // cannot absorb the perturbation and turn this control silently green.
    let anchor_at = runner_src
        .find("let fbw_labels")
        .expect("control: runner anchor already validated above");
    let (head, tail) = runner_src.split_at(anchor_at);
    let inject = |from: &str, to: &str| format!("{head}{}", tail.replacen(from, to, 1));

    // Shape 1 — a rename. Repoint one slot at a name from a different slot so
    // the entry stays a plausible fbw key and only the ORDER is wrong.
    let renamed = inject(
        &format!("\"{}\",", runner[1]),
        &format!("\"{}\",", runner[2]),
    );
    assert_ne!(renamed, runner_src, "control: rename injection was a no-op");
    let drifted = array_after(&renamed, "let fbw_labels", "runner");
    assert!(
        !divergences(authority, &drifted).is_empty(),
        "a renamed slot went undetected — `divergences` or the runner anchor is \
         broken, and this file is guarding nothing",
    );

    // Shape 2 — a dropped slot. The length check, not `divergences`, is what
    // has to catch this: removing the LAST entry leaves every surviving slot
    // correctly named.
    let dropped = inject(&format!("\"{}\",\n", runner[slots - 1]), "");
    assert_ne!(dropped, runner_src, "control: drop injection was a no-op");
    let short = array_after(&dropped, "let fbw_labels", "runner");
    assert_eq!(
        short.len(),
        slots - 1,
        "the drop injection did not shorten the extracted array, so the length \
         assertion in the mirror test is untested",
    );
    assert!(
        divergences(authority, &short).is_empty(),
        "control assumption broken: dropping the LAST slot should leave every \
         remaining slot correctly named, so only the length check can catch it",
    );
}

/// The ring GEOMETRY and BIT LAYOUT are mirrored too. Geometry and shift
/// widths are bare integers with nothing to derive them from on the runner's
/// side; the `FLAG_*` bits live on `FbwRingFlags` / `RingFlags`.
///
/// `RING_BASE` is `LABELS.len()` here, so a tally added on the authority side
/// moves the ring's start — and the runner, which decodes the ring by
/// arithmetic on its own copies, would keep reading from the old offset and
/// print the tail of the tallies as if it were a walk's outcome name. Renumber
/// a `SHIFT_*` instead and every census line reports one field's value under
/// another field's name. The label comparison above cannot see either: the
/// runner's LABEL array and its layout constants are independent
/// declarations.
#[test]
fn the_runner_mirrors_the_ring_layout() {
    use pyre_jit_trace::trace::fbw_diag as d;

    let runner_src = runner_source();
    for (anchor, authority, name) in [
        ("const RING_BASE: u32 = ", d::RING_BASE as u64, "RING_BASE"),
        (
            "const RING_ENTRIES: u32 = ",
            d::RING_ENTRIES as u64,
            "RING_ENTRIES",
        ),
        (
            "const RING_STRIDE: u32 = ",
            d::RING_STRIDE as u64,
            "RING_STRIDE",
        ),
        (
            "const NAME_SLOTS: u32 = ",
            d::NAME_SLOTS as u64,
            "NAME_SLOTS",
        ),
        (
            "const SHIFT_EFFECTS: u32 = ",
            d::SHIFT_EFFECTS as u64,
            "SHIFT_EFFECTS",
        ),
        (
            "const SHIFT_JOURNAL: u32 = ",
            d::SHIFT_JOURNAL as u64,
            "SHIFT_JOURNAL",
        ),
        (
            "const SHIFT_EXEC_MF: u32 = ",
            d::SHIFT_EXEC_MF as u64,
            "SHIFT_EXEC_MF",
        ),
        ("const SHIFT_LEG: u32 = ", d::SHIFT_LEG as u64, "SHIFT_LEG"),
        ("const FIELD_MASK: u64 = ", d::FIELD_MASK, "FIELD_MASK"),
    ] {
        let mirrored = declared_int(&runner_src, anchor, "runner");
        assert_eq!(
            mirrored, authority,
            "pyre-wasm-runner declares {name} = {mirrored} but \
             pyre_jit_trace::trace::fbw_diag::{name} is {authority}. The runner \
             indexes the `pyre_fbw_diag` export with `RING_BASE + entry * \
             RING_STRIDE` and unpacks the counter slot with the `SHIFT_*` / \
             `RingFlags` set, so a stale copy decodes the wrong words and prints \
             them as a walk outcome rather than failing.",
        );
    }
    for (flag, authority) in [
        ("VALID", d::RingFlags::VALID.bits()),
        ("COMMITTED", d::RingFlags::COMMITTED.bits()),
        ("BRIDGE", d::RingFlags::BRIDGE.bits()),
    ] {
        let mirrored = declared_bitflag(&runner_src, "FbwRingFlags", flag, "runner");
        assert_eq!(
            mirrored, authority,
            "pyre-wasm-runner FbwRingFlags::{flag} = {mirrored} but \
             pyre_jit_trace::trace::fbw_diag::RingFlags::{flag} is {authority}. \
             The runner unpacks the counter slot with these bits, so a stale \
             copy decodes the wrong words and prints them as a walk outcome \
             rather than failing.",
        );
    }
}
