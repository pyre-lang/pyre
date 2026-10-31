//! The two tracers' jitcode-opcode coverage, pinned.
//!
//! One jitcode encoding is consumed by two independent tracers:
//!
//! * majit's own, `majit-metainterp/src/pyjitpl/opimpl.rs`
//!   `OPCODE_IMPLEMENTATIONS` (plus `run_one_step`'s inline `BC_LIVE` /
//!   `BC_JUMP` in `dispatch.rs`), which keys on the `insns::BC_*` byte
//!   constants; and
//! * pyre's full-body walker, `pyre-jit-trace/src/jitcode_dispatch/`, which
//!   keys on the `opname/argcodes` string the same table names the byte by.
//!
//! Production runs the second one — `jitdriver.rs` says so at the segmenting
//! check it had to duplicate: "`pyjitpl/dispatch.rs` carries a second copy on
//! `MIFrame::run_one_step`, which production tracing does not go through." The
//! first is reachable only from inside majit (`mod pyjitpl` is private), so its
//! pressure comes from the crate's own tests and the `#[jit_interp]` machines
//! in `majit/examples/`, never from the flagship's corpus. Nothing compared the
//! two before this file: an arm added to one and not the other, or removed from
//! one and not the other, moved silently.
//!
//! This is a coverage gate, not a behaviour gate. It answers "which of the two
//! decodes this opcode at all", which is the cheapest question that can be
//! asked mechanically; it says nothing about whether two arms for the same
//! opcode agree. Equal coverage is also not the goal — the two consume
//! different jitcode populations, so a documented asymmetry is the normal
//! state and the snapshots below are what make an UNdocumented one visible.
//!
//! Updating: move the key between the lists here in the same commit that moves
//! the arm, and say in the commit message which walker gained or lost it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // <root>/pyre/pyre-jit-trace/ -> <root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pyre-jit-trace manifest sits two levels below the repo root")
        .to_path_buf()
}

fn read(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel))
        .unwrap_or_else(|err| panic!("{rel} must be readable from the repo root: {err}"))
}

/// Drop a `//` line comment, so a name that only appears in prose does not read
/// as an implementation.
fn code_of(line: &str) -> &str {
    match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    }
}

/// Every `"opname/argcodes"` string literal on one line.
fn quoted(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('"') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else { break };
        out.push(after[..close].to_string());
        rest = &after[close + 1..];
    }
    out
}

/// The `opname/argcodes` → `BC_*` table, in the order `insns.rs` writes it.
///
/// This is the encoding's own key universe, and it is deliberately read from
/// source rather than from `insns_opname_to_byte()`: the runtime table overlays
/// what THIS build's assembler emitted, so it varies with the analyzed source
/// set, while a coverage snapshot has to compare two static arm lists.
fn insns_table(root: &Path) -> Vec<(String, String)> {
    let src = read(root, "majit/majit-jitcode/src/codewriter/insns.rs");
    // Drop `//` comments so a name that only appears in prose does not
    // read as an implementation, then scan the joined remainder so a
    // rustfmt-wrapped `m.insert(\n    "key",\n    BC_X,\n)` still
    // counts as one encoding entry.
    let mut stripped = String::new();
    for line in src.lines() {
        stripped.push_str(code_of(line));
        stripped.push('\n');
    }
    let mut out = Vec::new();
    let mut rest = stripped.as_str();
    while let Some(at) = rest.find("m.insert(") {
        let after = &rest[at + "m.insert(".len()..];
        let Some(key) = quoted(after).into_iter().next() else {
            rest = &after[1..];
            continue;
        };
        let Some(comma) = after.find(',') else {
            rest = &after[1..];
            continue;
        };
        let bc: String = after[comma + 1..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if bc.starts_with("BC_") {
            out.push((key, bc));
        }
        rest = &after[1..];
    }
    assert!(
        out.len() > 150,
        "the insns table parser found only {} entries; `m.insert(\"key\", BC_X)` \
         is no longer how the table is written",
        out.len(),
    );
    out
}

/// Collect every `insns::BC_*` name in `body` that is not inside a comment.
fn collect_insns_bc(out: &mut BTreeSet<String>, body: &str) {
    for line in body.lines() {
        let code = code_of(line);
        let mut rest = code;
        while let Some(at) = rest.find("insns::BC_") {
            let after = &rest[at + "insns::".len()..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            out.insert(name);
            rest = &after[1..];
        }
    }
}

/// The `BC_*` constants majit's tracer names outside a comment.
///
/// The old source was `execute_one_instruction`'s `match bytecode`. That
/// match split into `opimpl_*` handlers keyed by `OPCODE_IMPLEMENTATIONS`
/// (`pyjitpl.py opcode_implementations`). `run_one_step` still handles
/// `BC_LIVE` and `BC_JUMP` inline before the table; the old match named
/// both, so they stay in this set even if a later edit drops the table row.
fn majit_walker_bytecodes(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let src = read(root, "majit/majit-metainterp/src/pyjitpl/opimpl.rs");
    let at = src
        .find("const OPCODE_IMPLEMENTATIONS")
        .expect("OPCODE_IMPLEMENTATIONS must exist");
    collect_insns_bc(&mut out, braced_body(&src, at));

    let dispatch = read(root, "majit/majit-metainterp/src/pyjitpl/dispatch.rs");
    let at = dispatch
        .find("pub fn run_one_step")
        .expect("run_one_step must exist");
    collect_insns_bc(&mut out, braced_body(&dispatch, at));

    assert!(
        out.len() > 100,
        "majit's tracer named only {} opcode constants; OPCODE_IMPLEMENTATIONS \
         no longer spells `insns::BC_*`",
        out.len(),
    );
    out
}

/// The text between the first `{` at or after `open` and its matching `}`.
///
/// Each scan wants a construct's own body and nothing after it: the
/// `OPCODE_IMPLEMENTATIONS` table, `run_one_step`'s inline LIVE/JUMP skip,
/// `handle`'s `match op.key`, and the arithmetic `regular_record_table!`.
/// Reading to end-of-file instead would admit lines that are not entries
/// at all.
fn braced_body(src: &str, open: usize) -> &str {
    let start = src[open..].find('{').expect("the construct opens a block") + open + 1;
    let mut depth = 1usize;
    for (offset, ch) in src[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[start..start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after byte {open}");
}

/// The keys pyre's walker answers: the arms of `handle`'s `match op.key`, plus
/// the `regular_record_table!` entries that stand in for the arithmetic arms.
fn pyre_walker_keys(root: &Path) -> BTreeSet<String> {
    let src = read(root, "pyre/pyre-jit-trace/src/jitcode_dispatch/mod.rs");
    let at = src
        .find("fn handle<Sym: WalkSym>(")
        .expect("the walker's per-opcode `handle` must exist");
    let open = src[at..]
        .find("match op.key {")
        .map(|rel| at + rel)
        .expect("`handle` must dispatch on `op.key`");
    let mut out = BTreeSet::new();
    for line in braced_body(&src, open).lines() {
        // An arm pattern sits at the match's own indent and carries its `=>`
        // on the same line; every arm in `handle` is written that way, and the
        // length assertion below is what notices if one stops being.
        let Some(fat) = line.find("=>") else { continue };
        if !line.starts_with("        \"") {
            continue;
        }
        out.extend(quoted(&line[..fat]));
    }
    let arith = read(root, "pyre/pyre-jit-trace/src/jitcode_dispatch/arith.rs");
    let table = arith
        .find("regular_record_table! {")
        .expect("the arithmetic arms live in `regular_record_table!`");
    // Only the invocation's own body. Scanning to end-of-file would let any
    // later `"opname/argcodes"` string in `arith.rs` — a doc example, a test
    // fixture — enter this set without a generated arm behind it, and a key
    // that enters unearned is a missing walker arm this gate stops noticing.
    for line in braced_body(&arith, table).lines() {
        for key in quoted(code_of(line)) {
            if key.contains('/') {
                out.insert(key);
            }
        }
    }
    assert!(
        out.len() > 150,
        "pyre's walker answered only {} keys; `handle`'s arms are no longer \
         one-per-line at the match indent",
        out.len(),
    );
    out
}

/// Keys majit's tracer decodes and pyre's walker does not.
const MAJIT_ONLY: &[&str] = &[
    // The `#[jit_interp]` state-machine accessors. A machine declares its own
    // `state = T`; a Python portal jitcode declares none.
    "load_state_field/di",
    "load_state_field_float/df",
    "load_state_field_ref/dr",
    "store_state_field/di",
    "store_state_field_float/df",
    "store_state_field_ref/dr",
    // The `/P` flat-payload adapters. `blackhole.rs` carries a
    // `handler_*_ext` for each; the walker has no arm.
    "call_assembler_float_ext/P",
    "call_assembler_int_ext/P",
    "call_assembler_ref_ext/P",
    "call_assembler_void_ext/P",
    "cond_call_value_int_ext/P",
    "cond_call_value_ref_ext/P",
    "cond_call_void_ext/P",
    "inline_call_nested_ext/P",
    "record_known_result_int_ext/P",
    "record_known_result_ref_ext/P",
    // Singles.  `goto_if_not_int_is_true/iL` left this list when #1679
    // re-landed the fusion with the walker arm it needs, which is the move
    // this file exists to make visible. The three raw-array keys
    // (`getarrayitem_raw_{i,f}`, `setarrayitem_raw_f`) left when the
    // walker gained the arms `pyjitpl.py` `opimpl_getarrayitem_raw_{i,f}`
    // / `opimpl_setarrayitem_raw_f` already named. `record_exact_value_i/ii`
    // / `record_exact_value_r/rr` left when the walker gained
    // `pyjitpl.py opimpl_record_exact_value_{i,r}` / `_record_exact_value`.
    "arraybase_vable/rdd>i",
    "goto_if_exception_mismatch/iL",
    "newlist_clear/idddd>r",
    "rvmprof_code/ii",
    // `opimpl_unreachable` aborts the walk instead of panicking.
    "unreachable/",
];

/// Keys pyre's walker answers and majit's tracer does not.
const PYRE_ONLY: &[&str] = &[
    // `abort/>i` (`insns.rs` `BC_ABORT_RESULT_I`). The encoding names it
    // and pyre's walker shares the `abort/>r` arm. Majit's tracer names
    // `BC_ABORT` and `BC_ABORT_RESULT_R` only (`dispatch.rs`).
    "abort/>i",
    // The float array reads. `getarrayitem_gc_r_pure/rid>r` left this list
    // when majit's tracer gained the pure array-read arms.
    // `getarrayitem_gc_f/rid>f` and `getarrayitem_gc_f_pure/rid>f` left this
    // list when majit's tracer gained the float getarrayitem arms.
    // Constant-length form. The register-length `new_array_clear/id>r`
    // is decoded by both tracers.
    "new_array_clear/cd>r",
    "setarrayitem_gc_i/ricd",
    "setarrayitem_gc_r/rcrd",
    // The push/pop bank transfers.
    "float_pop/>f",
    "float_push/f",
    "int_pop/>i",
    "int_push/i",
    "ref_pop/>r",
    "ref_push/r",
    // Raw float store.
    "raw_store_f/iifd",
    // `pyjitpl.py opimpl_gc_load_indexed_i`. Majit's tracer has no arm.
    "gc_load_indexed_i/riiii>i",
    // `pyjitpl.py` `_opimpl_getinteriorfield_gc_any` /
    // `execute_setinteriorfield_gc`. Majit's tracer has no arm.
    "getinteriorfield_gc_f/rid>f",
    "getinteriorfield_gc_i/rid>i",
    "getinteriorfield_gc_r/rid>r",
    "setinteriorfield_gc_f/rifd",
    "setinteriorfield_gc_i/riid",
    "setinteriorfield_gc_r/rird",
    // Constant-index string read (`pyjitpl.py opimpl_strgetitem`). Majit's
    // `BC_STRLEN` / `BC_STRGETITEM` arms decode the register-index shapes
    // only.
    "strgetitem/rc>i",
];

/// Keys the encoding names that neither tracer decodes.
///
/// Not the same statement as `PYRE_ONLY` / `MAJIT_ONLY`: a key here is one the
/// assembler could emit into a jitcode that no tracer could then walk, so a key
/// ARRIVING here is the move worth reviewing.
/// Keys the encoding names that neither tracer walks.
///
/// Measured once, so that a key ARRIVING here is the thing to look at rather
/// than the list itself: the registration-gap snapshot in
/// `jitcode_runtime.rs` pins most of them against `build_emitted_insns()`,
/// the opnames the build actually emitted.
///
/// An unhandled key is a clean `DispatchError::UnsupportedOpname` decline,
/// so a new arrival costs compilation rather than correctness.
const NEITHER: &[&str] = &[
    "check_neg_index/rid>i",
    "gc_load_indexed_f/riiii>f",
    "getlistitem_gc_f/ridd>f",
    "getlistitem_gc_i/ridd>i",
    "getlistitem_gc_r/ridd>r",
    "newlist/idddd>r",
    "newlist_hint/idddd>r",
];

fn expected(list: &[&str]) -> BTreeSet<String> {
    let set: BTreeSet<String> = list.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(set.len(), list.len(), "the expectation list repeats a key");
    set
}

#[test]
fn the_two_tracers_opcode_coverage_matches_its_snapshot() {
    let root = repo_root();
    let table = insns_table(&root);
    let majit_bc = majit_walker_bytecodes(&root);
    let pyre_keys = pyre_walker_keys(&root);

    let (mut majit_only, mut pyre_only, mut neither) =
        (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let mut both = 0usize;
    for (key, bc) in &table {
        match (majit_bc.contains(bc), pyre_keys.contains(key)) {
            (true, true) => both += 1,
            (true, false) => {
                majit_only.insert(key.clone());
            }
            (false, true) => {
                pyre_only.insert(key.clone());
            }
            (false, false) => {
                neither.insert(key.clone());
            }
        }
    }

    let advice = "A key moved between the two tracers' coverage sets. Move it \
                  between the lists in this file in the same commit, and say \
                  which tracer gained or lost the arm. A key that arrives in \
                  NEITHER is the one to look at hardest: the assembler can emit \
                  it and no tracer can then walk it.";
    assert_eq!(majit_only, expected(MAJIT_ONLY), "MAJIT_ONLY: {advice}");
    assert_eq!(pyre_only, expected(PYRE_ONLY), "PYRE_ONLY: {advice}");
    assert_eq!(neither, expected(NEITHER), "NEITHER: {advice}");

    // The double-implemented population, which is what a behaviour comparison
    // would have to cover. Asserted as a floor rather than a number so that
    // adding an arm to both tracers costs no edit here, while losing one from
    // either is caught by the three sets above.
    assert!(
        both >= 150,
        "the two tracers answer only {both} keys in common; the earlier \
         measurement was 150 of {}",
        table.len(),
    );
}

/// Keys pyre's walker answers that the encoding does not name.
///
/// `int_same_as/i>i` is unreachable from any assembled jitcode and says so
/// at its own definition: `jtransform.py rewrite_op_same_as` removes
/// `same_as` before assembly. `abort/>i` left this list when `insns.rs`
/// named `BC_ABORT_RESULT_I`; the walker already shared that arm with
/// `abort/>r`, and majit's tracer still does not decode the byte, so the
/// key sits in `PYRE_ONLY`.
///
/// `newstr`, `strsetitem` and `copystrcontent` are translator-only keys: the
/// codewriter numbers them through the `assembler.py setdefault` dynamic
/// allocator (`insn_byte_opt` answers `None`), as the `StringBuilderRepr`
/// helper graphs (`rbuilder.py ll_append` / `ll_grow_by` / `ll_build`) emit
/// them.  No walked body builds through those graphs yet.
///
/// `getfield_raw_r/id>r` is numbered the same way. `jtransform.py
/// rewrite_op_getfield` emits the pure form to load an immutable field off
/// a raw struct (map terminator `w_cls`); `blackhole.py bhimpl_getfield_raw_r`
/// / `wire_bhimpl_handlers` bind the handler. `insns.rs` has no `BC_*`.
///
/// `record_quasiimmut_field/idd` is numbered the same way. A headerless
/// owner (`CellFamily`, `AuditHolder`, `GilReadyState`) puts the raw
/// pointer in the int bank, so `jtransform.py rewrite_op_getfield` keys
/// `record_quasiimmut_field/idd`; `blackhole.rs wire_bhimpl_handlers`
/// binds `handler_record_quasiimmut_field` next to `/rdd` (`blackhole.py
/// bhimpl_record_quasiimmut_field`). `insns.rs` has no `BC_*`.
#[test]
fn the_walker_answers_no_key_the_encoding_cannot_name() {
    let root = repo_root();
    let named: BTreeSet<String> = insns_table(&root).into_iter().map(|(key, _)| key).collect();
    let mut unnamed: Vec<String> = pyre_walker_keys(&root)
        .into_iter()
        .filter(|key| !named.contains(key))
        .collect();
    unnamed.sort();
    let mut expected: Vec<String> = [
        "getfield_raw_r/id>r",
        "record_quasiimmut_field/idd",
        "int_same_as/i>i",
        "newstr/i>r",
        "newstr/c>r",
        "strsetitem/rii",
        "strsetitem/rci",
        "strsetitem/ric",
        "strsetitem/rcc",
        "copystrcontent/rriii",
        "copystrcontent/rrcii",
        "copystrcontent/rrici",
        "copystrcontent/rriic",
        "copystrcontent/rrcci",
        "copystrcontent/rrcic",
        "copystrcontent/rricc",
        "copystrcontent/rrccc",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    expected.sort();
    assert_eq!(
        unnamed, expected,
        "the walker's set of arms with no `insns` entry drifted; a new one is \
         either dead or an entry the table is missing",
    );
}
