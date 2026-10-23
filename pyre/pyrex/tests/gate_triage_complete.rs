//! Checks that each environment-gate namespace and its triage document contain
//! the same live names. The scan covers Rust workspace members and project-owned
//! Python under `pyre/` and `scripts/`.
//!
//! Gate names in comments are not readers. Build-script
//! `rerun-if-env-changed` declarations are readers because they affect Cargo's
//! dependency tracking even when the environment lookup is indirect.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // <root>/pyre/pyrex/ -> <root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pyrex manifest sits two levels below the repo root")
        .to_path_buf()
}

/// Workspace member directories, read from the root `Cargo.toml`.
///
/// Deriving the search roots from the member list rather than walking the whole
/// repository is what keeps untracked scratch copies of source files out: a
/// `.rs` file only reaches a compiler if it belongs to a member crate, so only
/// those can hold a live gate. It also needs no maintenance — a new crate has to
/// join `members` to build at all.
fn workspace_member_dirs(root: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read root Cargo.toml");
    // Anchored at line start: `default-members = [` sits above `members = [`
    // and an unanchored search finds that one instead.
    let after = manifest
        .split_once("\nmembers = [")
        .expect("root Cargo.toml has a members list")
        .1;
    let list = after.split_once(']').expect("members list is closed").0;
    let dirs: Vec<PathBuf> = list
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix('"')?;
            Some(root.join(line.split_once('"')?.0))
        })
        .collect();
    assert!(
        dirs.len() > 10,
        "parsed only {} workspace members — the Cargo.toml parse is wrong",
        dirs.len()
    );
    dirs
}

/// Where this project's own Python lives.
///
/// An allow-list rather than a tree walk, for two reasons the Rust side does not
/// have. The repository vendors the CPython and PyPy sources — `lib-python/`,
/// `pypy/`, `rpython/`, `lib_pypy/`, some 4600 `.py` files that are upstream's
/// and not ours — and untracked scratch directories sit at the repository root.
/// Neither is under these two.
const PYTHON_ROOTS: [&str; 2] = ["pyre", "scripts"];

fn collect_sources(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // A member crate's own `target/` from a standalone build.
            if entry.file_name() != "target" {
                collect_sources(&path, ext, out);
            }
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push(path);
        }
    }
}

/// The function names that read an environment variable in this tree.
///
/// `env::var` (with its `_os` suffix) is the common one, but a search for it
/// alone misses the host seam: `host_os::var` on the sandbox path and
/// `host_seam::getenv`, which takes a *byte* string. `PYRE_STDLIB` is read
/// through both seam forms in `pyre-interpreter/src/importing.rs` and stays
/// visible to a `std::env` search only because unrelated `env::var` readers
/// exist in `pyre-wasm-runner`. A wasm- or sandbox-only gate would have no such
/// cover and would bypass the brake entirely.
///
/// `environ.get` and `getenv` are the Python side. Only these two, because they
/// read unambiguously: the harness also writes gates into a child's environment
/// (`env["PYRE_PROBE"] = …`, `env.pop(…)`), and a subscript form cannot be told
/// from a read without parsing. Writing a gate for a child is not owning it —
/// the child's read is what this file asks about, and that read is in Rust.
///
/// A form belonging to the other language costs nothing: `env::var(` cannot
/// occur in Python, nor `environ.get(` in Rust.
///
/// `read_uint_from_env` is one name-taking helper in the tree that a gate
/// literal reaches: the collector's GC knobs pass the name to it and it forwards
/// to `env::var` with a *variable*, so the literal never sits beside a form
/// above and three live `MAJIT_GC_BH_PROBE_*` sub-knobs were invisible to both
/// brakes at once — read but reported as unread, documented but reported as
/// undocumented. Its sibling `read_float_from_env` is not here because no gate
/// name reaches it; add it the day one does.
///
/// `env_var` is the other, and it covers its `_os` suffix the way `env::var`
/// does. `jit_env::{env_var, env_var_os}` is the seam the JIT knobs read
/// through so a wasm guest -- whose `std::env` is permanently empty -- resolves
/// them against an environment its host supplies. The literal sits beside the
/// helper, not beside `env::var`, and the six `PYRE_JD1*` / `PYRE_JIT` /
/// `PYRE_NO_JIT` gates went unread by this brake the moment they moved.
/// Written bare, because the call sites spell it both ways: imported
/// (`env_var("PYRE_JIT")`) and qualified (`crate::jit_env::env_var_os(..)`).
///
/// Every form is matched on an identifier boundary: the character before it
/// must not continue a name, so `my_read_uint_from_env` is not the helper and
/// `pyre_env::var` is not `std::env::var`. The qualifiers the real read sites
/// carry — `::` before `env::var`, `getenv` and `host_os::var`, `.` before
/// `getenv` and `environ.get` — all clear that boundary.
const READ_FORMS: [&str; 7] = [
    "env::var",
    "env_var",
    "host_os::var",
    "getenv",
    "environ.get",
    "read_uint_from_env",
    "env_is_set",
];

/// Cargo's build-script declaration that the build depends on a gate:
/// `println!("cargo::rerun-if-env-changed=NAME")`.
///
/// This counts as a read because it controls when Cargo reruns the build script.
/// It also covers gates passed through an indirect environment-read helper.
const CARGO_ENV_DEP: &str = "rerun-if-env-changed=";

/// A gate namespace: the prefix its names carry, and the document that owns them.
///
/// Each namespace owns a document so separately versioned components carry the
/// retirement information for their own gates.
struct GateNamespace {
    prefix: &'static str,
    doc: &'static str,
}

/// Adding a row arms both checks for the entire namespace. Add its triage
/// document in the same change; otherwise every live name in the namespace is
/// reported as undocumented.
///
/// The checks compare names; they do not validate a row's prose.
const NAMESPACES: [GateNamespace; 2] = [
    GateNamespace {
        prefix: "PYRE_",
        doc: "pyre/gate-triage.md",
    },
    GateNamespace {
        prefix: "MAJIT_",
        doc: "majit/gate-triage.md",
    },
];

/// The namespace owning this gate name, if any.
fn namespace_of(name: &str) -> Option<&'static GateNamespace> {
    NAMESPACES.iter().find(|ns| name.starts_with(ns.prefix))
}

/// Gate names this text reads from the environment.
///
/// Matches the read forms above rather than every mention of the name, so a gate
/// discussed in a comment or held in a Rust const does not count as live. That
/// is the same distinction `gate-triage.md` §2 draws by hand.
/// Whether the character before `at` continues an identifier, which makes the
/// form a tail of a longer name rather than the name itself.
fn continues_an_identifier(text: &str, at: usize) -> bool {
    text[..at]
        .chars()
        .next_back()
        .is_some_and(|c| c.is_alphanumeric() || c == '_')
}

fn gates_read_by(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    for form in READ_FORMS {
        for (at, _) in text.match_indices(form) {
            if continues_an_identifier(text, at) {
                continue;
            }
            let rest = &text[at + form.len()..];
            // `env::var_os` is the same read wearing a suffix.
            let rest = rest.strip_prefix("_os").unwrap_or(rest);
            let Some(rest) = rest.strip_prefix('(') else {
                continue;
            };
            let rest = rest.trim_start();
            // `host_seam::getenv` names its gate with a byte string.
            let rest = rest.strip_prefix('b').unwrap_or(rest);
            let Some(rest) = rest.strip_prefix('"') else {
                continue;
            };
            let Some(end) = rest.find('"') else { continue };
            let name = &rest[..end];
            if namespace_of(name).is_some() {
                found.push(name);
            }
        }
    }
    // The build-script dependency declaration, which carries its name bare
    // rather than as a quoted call argument.
    for (at, _) in text.match_indices(CARGO_ENV_DEP) {
        let rest = &text[at + CARGO_ENV_DEP.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if namespace_of(name).is_some() {
            found.push(name);
        }
    }
    found
}

#[test]
fn gates_read_by_matches_the_read_forms_and_nothing_else() {
    let sample = r#"
        std::env::var("PYRE_A").is_ok();
        env::var_os("PYRE_B").is_none();
        std::env::var(
            "PYRE_C",
        );
        host_os::var("PYRE_D").ok();
        crate::host_seam::getenv(b"PYRE_E");
        os.environ.get("PYRE_F")
        os.getenv("PYRE_G")
        println!("cargo::rerun-if-env-changed=PYRE_H");
        read_uint_from_env("PYRE_I").unwrap_or(0);
        env_var("PYRE_J");
        crate::jit_env::env_var_os("PYRE_K");
        // a longer name ending in a read form is not that read form
        my_read_uint_from_env("PYRE_NOT_THE_HELPER");
        supplied_env_var("PYRE_NOT_THE_SEAM");
        pyre_env::var("PYRE_NOT_STD_ENV");
        # a gate written into a child's environment is not a read of ours
        env["PYRE_NOT_A_READ_EITHER"] = "1"
        // PYRE_MENTIONED_IN_A_COMMENT
        const PYRE_CONST: &str = "PYRE_NOT_A_READ";
        other::var("PYRE_NOT_ENV");
        env::var("HOME");
    "#;
    let mut got = gates_read_by(sample);
    // Sorted: the scan groups by read form, so the order carries no meaning.
    got.sort_unstable();
    assert_eq!(
        got,
        vec![
            "PYRE_A", "PYRE_B", "PYRE_C", "PYRE_D", "PYRE_E", "PYRE_F", "PYRE_G", "PYRE_H",
            "PYRE_I", "PYRE_J", "PYRE_K"
        ]
    );
}

/// Does this `##` heading introduce a section that records history?
///
/// §2 lists names that were never env vars; a section of retired gates or of
/// names with no read site records gates that are *gone*. A name in either is
/// the record of something that does not gate anything today, so it must not
/// satisfy the brake — otherwise re-introducing a reader lands green on the
/// strength of an obituary, with neither its new polarity nor a retirement plan
/// written down.
///
/// `gate-triage.md` carries no retirement sections today (they were removed on
/// 2026-08-20 in favour of the file's history), so only the "not gates" arm
/// matches. The other two stay because either document may grow such a section
/// again, and the per-row check below is what covers a retirement note written
/// under a heading that reads live.
fn is_history_heading(heading: &str) -> bool {
    let lower = heading.to_ascii_lowercase();
    // "retired", not "retire": §4 is a *live* section whose heading says when
    // its gates will go ("retire when the epic closes").
    lower.contains("retired") || lower.contains("dead") || lower.contains("not gates")
}

/// The names carrying `prefix` that this line spells out, in the order it
/// spells them.
///
/// Tokenized rather than substring-searched: `contains("PYRE_A")` is satisfied
/// by a documented `PYRE_ANCHOR_STRICT`, so a new gate whose name is a prefix of
/// a listed one would slip through the brake unnoticed.
///
/// The same tokenization keeps the two namespaces apart where they overlap in
/// text: `PYRE_MAJIT_STATS_ANCESTOR` contains `MAJIT_`, and the preceding-
/// character guard is what stops it being read as a majit gate.
fn gate_names_in<'a>(line: &'a str, prefix: &str) -> Vec<&'a str> {
    let mut names = Vec::new();
    for (at, _) in line.match_indices(prefix) {
        // A name preceded by a name character is the tail of a longer token.
        if line[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        let end = line[at..]
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .map_or(line.len(), |off| at + off);
        names.push(&line[at..end]);
    }
    names
}

/// Every `PYRE_*` name the triage document lists as a live gate.
///
/// Scoped to the live sections for the reason in `is_history_heading` — `###`
/// subsections inherit their `##` parent, which is what keeps §6a–§6c live under
/// §6 — and then filtered by what the document records about each name.
///
/// **Retirement is a property of the gate, not of the line.** A retirement line
/// retires its *subject*, and a name stays retired wherever else it is written:
/// a gate recorded retired in one table, then named again by live prose
/// discussing the seam it used to guard, would otherwise look live on the
/// strength of those mentions while having no reader at all.
///
/// **Subject, not mention**: only the *first* name on a retirement line is
/// retired by it. A row retiring `PYRE_AUTHORITATIVE` may end
/// "`PYRE_PROBE_AUTHORITATIVE` is separate and remains live" — the whole point
/// of that clause is that the second name is not the one being retired.
///
/// "retired", not "retire": §4 is a live section whose gates are the ones that
/// "retire when the epic closes".
///
/// A line that *negates* the word is refused rather than read either way — see
/// `negates_retirement`.
fn gates_documented_in<'a>(triage: &'a str, prefix: &str) -> BTreeSet<&'a str> {
    let mut live_names: BTreeSet<&str> = BTreeSet::new();
    let mut retired_subjects: BTreeSet<&str> = BTreeSet::new();
    let mut live = true;
    for (at, line) in triage.lines().enumerate() {
        if let Some(heading) = line.strip_prefix("## ") {
            live = !is_history_heading(heading);
        }
        let names = gate_names_in(line, prefix);
        let lower = line.to_ascii_lowercase();
        if lower.contains("retired") {
            if let Some(&subject) = names.first() {
                assert!(
                    !negates_retirement(&lower),
                    "line {} of the {prefix}* triage document names {subject} beside a \
                     negated `retired`, so the scan cannot tell whether the line records \
                     a retirement:\n  {line}\n\n\
                     The scan reads `retired` as a verdict on the line's first name and \
                     would drop {subject} from the live set wherever else the document \
                     writes it — which a line saying it is *not* retired does not ask \
                     for. Either state the retirement plainly, or write the note without \
                     the word.",
                    at + 1
                );
                retired_subjects.insert(subject);
            }
        } else if live {
            live_names.extend(names);
        }
    }
    live_names.retain(|name| !retired_subjects.contains(name));
    live_names
}

/// Does this line say a gate is *not* retired?
///
/// A negated retirement beside a gate name is ambiguous to the line-oriented
/// parser, so it is rejected instead of silently dropping a live entry.
fn negates_retirement(lowercased_line: &str) -> bool {
    ["not retired", "never retired", "not yet retired"]
        .iter()
        .any(|negation| lowercased_line.contains(negation))
}

/// A documented token that is a wildcard's stem rather than a gate.
///
/// This file writes `PYRE_*` and `PYRE_FBW_*` when it means a family, and the
/// token scan stops at the `*`, leaving a name ending in `_`. No gate is named
/// that way, so the trailing underscore is the whole test.
fn is_wildcard_stem(name: &str) -> bool {
    name.ends_with('_')
}

/// Every source file a gate can be read from.
fn source_files(root: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for member in workspace_member_dirs(root) {
        collect_sources(&member, "rs", &mut sources);
    }
    assert!(
        sources.len() > 100,
        "found only {} .rs files across the workspace members — the walk is not \
         reaching the tree",
        sources.len()
    );

    let rust_files = sources.len();
    for py_root in PYTHON_ROOTS {
        let dir = root.join(py_root);
        assert!(
            dir.is_dir(),
            "PYTHON_ROOTS names {py_root}, which is not a directory under {}",
            root.display()
        );
        collect_sources(&dir, "py", &mut sources);
    }
    assert!(
        sources.len() - rust_files > 100,
        "found only {} .py files under {PYTHON_ROOTS:?} — the walk is not \
         reaching the harness",
        sources.len() - rust_files
    );
    sources
}

/// `(gate name, repo-relative file)` for every environment read in the tree.
fn read_sites(root: &Path) -> BTreeSet<(String, String)> {
    // This file's own fixture spells out `env::var("PYRE_A")` and friends, which
    // are test data rather than gates.
    let self_path = root.join(file!());
    let mut sites = BTreeSet::new();
    for path in source_files(root) {
        if path == self_path {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for name in gates_read_by(&text) {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            sites.insert((name.to_string(), rel.display().to_string()));
        }
    }
    sites
}

/// The triage document owning `ns`, read from disk.
fn triage_text(root: &Path, ns: &GateNamespace) -> String {
    let path = root.join(ns.doc);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn namespace(prefix: &str) -> &'static GateNamespace {
    NAMESPACES
        .iter()
        .find(|ns| ns.prefix == prefix)
        .expect("NAMESPACES names this prefix")
}

/// The live/history split in `gate-triage.md` is load-bearing for both brakes,
/// and a heading reword would otherwise change what they accept without failing.
/// Anchored on each side: a retired name must not count, a listed live name must.
#[test]
fn pyre_triage_live_history_split_holds_at_both_anchors() {
    let root = repo_root();
    let ns = namespace("PYRE_");
    // Bound, not inlined: the returned set borrows the document text.
    let triage = triage_text(&root, ns);
    let documented = gates_documented_in(&triage, ns.prefix);

    assert!(
        !documented.contains("PYRE_STR_DESCR"),
        "PYRE_STR_DESCR is a field-descriptor const named only in §2, the \
         not-gates section, and nothing reads it from the environment — so it \
         must not count as documented. is_history_heading no longer matches \
         gate-triage.md's headings"
    );
    assert!(
        documented.contains("PYRE_WALKABORT_OFF"),
        "PYRE_WALKABORT_OFF is listed live in §6a but did not count as \
         documented — is_history_heading is excluding a live section"
    );
    // §6a2 is a sibling subsection and inherits §6 the same way, so it needs
    // its own anchor: a heading filter that swallowed only §6a2 would leave
    // the §6a probe above green.
    assert!(
        documented.contains("PYRE_JD1"),
        "PYRE_JD1 is listed live in §6a2 but did not count as documented — \
         is_history_heading is excluding a live section"
    );
}

#[test]
fn every_live_gate_has_a_triage_entry() {
    let root = repo_root();
    let sites = read_sites(&root);

    for ns in &NAMESPACES {
        let triage = triage_text(&root, ns);
        let documented = gates_documented_in(&triage, ns.prefix);
        let missing: Vec<&(String, String)> = sites
            .iter()
            .filter(|(name, _)| name.starts_with(ns.prefix) && !documented.contains(name.as_str()))
            .collect();

        assert!(
            missing.is_empty(),
            "{} {}* gate(s) are read from the environment but have no entry in \
             {}:\n{}\n\n\
             Add a row for each: what it gates, its default polarity, and — for a \
             default-ON experiment — the epic whose close retires it.",
            missing.len(),
            ns.prefix,
            ns.doc,
            missing
                .iter()
                .map(|(name, file)| format!("  {name}  ({file})"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

#[test]
fn every_live_triage_entry_still_has_a_reader() {
    let root = repo_root();
    let read: BTreeSet<String> = read_sites(&root)
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    for ns in &NAMESPACES {
        let triage = triage_text(&root, ns);
        let stale: Vec<&str> = gates_documented_in(&triage, ns.prefix)
            .into_iter()
            .filter(|name| !is_wildcard_stem(name) && !read.contains(*name))
            .collect();

        assert!(
            stale.is_empty(),
            "{} name(s) are listed live in {} but nothing in the tree reads \
             them:\n{}\n\n\
             Delete the row in the same change that deleted the reader: this file \
             records what is live, and the document's own history is where a \
             retired gate's polarity and epic are read from.",
            stale.len(),
            ns.doc,
            stale
                .iter()
                .map(|name| format!("  {name}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// Prove that each namespace is checked against its own document, rather than a
/// union that could hide a name recorded in the wrong document.
#[test]
fn both_brakes_are_parameterized_over_the_namespace() {
    let ex_doc = "## §1 live\n`EX_KNOWN` gates a thing.\n";
    let syn_doc = "## §1 live\n`SYN_KNOWN` gates a thing.\n`EX_ELSEWHERE` is named here.\n";

    assert!(
        gates_documented_in(ex_doc, "EX_").contains("EX_KNOWN"),
        "prefix scan missed its own name"
    );
    assert!(
        gates_documented_in(syn_doc, "SYN_").contains("SYN_KNOWN"),
        "prefix scan missed its own name"
    );

    // Positive control: the name is findable by this scan, in the document that
    // spells it.
    assert!(
        gates_documented_in(syn_doc, "EX_").contains("EX_ELSEWHERE"),
        "the control failed — the scan cannot find EX_ELSEWHERE anywhere, so the \
         assertion below would pass for the wrong reason"
    );
    // The check: it is not documented for `EX_`, because `EX_`'s document does
    // not spell it. A scan over the union of all documents would pass the
    // control and fail here — which is the failure that makes one shared
    // document unsafe, and the reason `MAJIT_` cannot simply be pointed at
    // `pyre/gate-triage.md`.
    assert!(
        !gates_documented_in(ex_doc, "EX_").contains("EX_ELSEWHERE"),
        "a name absent from its own namespace's document counted as documented — \
         the lookup is not per document"
    );

    // Retirement is per document too: the same name may be live in one and
    // retired in the other, and each document governs only its own namespace.
    let retired = gates_documented_in("## §1 live\n`EX_KNOWN` retired.\n", "EX_");
    assert!(
        !retired.contains("EX_KNOWN"),
        "a retirement line in the namespace's own document did not retire its subject"
    );
}

/// `PYRE_MAJIT_STATS_ANCESTOR` and friends spell `MAJIT_` inside a `PYRE_` name.
///
/// The preceding-character guard in `gate_names_in` is the only thing keeping
/// them out of a future `MAJIT_` namespace, and it is one `continue` in a loop —
/// cheap to lose in an edit, and its loss would silently move four documented
/// pyre gates into majit's ledger.
#[test]
fn a_prefix_appearing_inside_a_longer_name_is_not_a_gate_of_that_namespace() {
    let line = "`PYRE_MAJIT_STATS_ANCESTOR`, `PYRE_MAJIT_STATS_ROOT_ONLY`, `MAJIT_LOG`";
    assert_eq!(
        gate_names_in(line, "MAJIT_"),
        vec!["MAJIT_LOG"],
        "a `MAJIT_` embedded in a `PYRE_MAJIT_*` name was read as a majit gate"
    );
    assert_eq!(
        gate_names_in(line, "PYRE_"),
        vec!["PYRE_MAJIT_STATS_ANCESTOR", "PYRE_MAJIT_STATS_ROOT_ONLY"]
    );
}
