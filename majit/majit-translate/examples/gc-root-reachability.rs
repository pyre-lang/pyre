//! Answer `framework.py`'s question over a charon artefact: which functions can
//! reach application-level Python, and therefore a collection.
//!
//! Usage: `cargo run -p majit-translate --release --example gc-root-reachability -- <file.ullbc>...`
//!
//! The donor set is not a matter of taste; see `GC_JOIN_WITH` below.  For the
//! interpreter the answer is
//!
//! ```text
//! GC_JOIN_WITH=build/llbc/pyre-jit.ullbc,build/llbc/pyre-object.ullbc,build/llbc/majit-rlib.ullbc \
//!   ./target/release/examples/gc-root-reachability build/llbc/pyre-interpreter.ullbc
//! ```

use majit_translate::memory::gctransform::{framework, liveness};

const PYTHON_DISPATCH_SEEDS_REF: &[&str] = framework::PYTHON_DISPATCH_SEEDS;

/// Write one artefact's rows out, truncating the file on the first artefact
/// of the run and appending for every one after it.
///
/// The scan takes a list of artefacts and each has its own rows, so a plain
/// write leaves only the last artefact's -- and leaves it looking complete.
/// `opened` is what tells the first artefact from the rest.
fn write_rows(
    path: &str,
    rows: &str,
    opened: &mut std::collections::HashSet<String>,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    if opened.insert(path.to_string()) {
        opts.write(true).create(true).truncate(true);
    } else {
        opts.append(true).create(true);
    }
    opts.open(path)?.write_all(rows.as_bytes())
}

/// Resident set size in MB, or `None` where the platform has no cheap answer.
///
/// A phase trace, not a budget: this run holds a multi-hundred-megabyte
/// artefact and builds several indexes over it, and which of those the peak
/// belongs to is not inferable from the total.  Off unless `GC_RSS_TRACE` is
/// set, and written to stderr, so the gate's stdout stays exactly what it was.
fn rss_mb() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb / 1024)
}

fn mark(label: &str) {
    if std::env::var_os("GC_RSS_TRACE").is_none() {
        return;
    }
    match rss_mb() {
        Some(mb) => eprintln!("[rss] {mb:6} MB  {label}"),
        None => eprintln!("[rss]      ?     {label}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    mark("start");
    // A run that says it wrote and did not must not exit 0: the file it
    // named is what a consumer reads, and an absent one reads as no findings.
    let mut write_failed = false;
    let mut opened: std::collections::HashSet<String> = Default::default();
    if args.is_empty() {
        eprintln!("usage: gc-root-reachability <file.ullbc>...");
        std::process::exit(2);
    }
    // Artefacts whose call graph is joined in but whose bodies are not scanned.
    // One artefact is not one program: `pyre-jit.ullbc` carries the portal and
    // the blackhole but hardly any of the interpreter they hand control to, so
    // asked alone it answers that nothing reaches a collection.  Naming
    // `pyre-interpreter.ullbc` here is what makes that answer mean anything.
    // Donors are loaded once and their `Llbc` dropped: only the graph is kept,
    // because holding two multi-hundred-megabyte artefacts open at once is what
    // the memory goes to.
    //
    // `pyre-object.ullbc` is not optional either, and leaving it out fails
    // quietly rather than loudly.  `gc_hook::try_gc_alloc_collecting_rooted` --
    // the hook every host-side collecting allocation goes through -- occurs 16
    // times in `pyre-object.ullbc` and **zero** times in
    // `pyre-interpreter.ullbc` (a nonsense pattern scores zero in both, so the
    // search is not the thing that is broken).  Charon inlines only the callees
    // a crate reaches, and 1359 `pyre_object` bodies is not the whole crate.
    // Joined with `pyre-jit.ullbc` alone the run therefore reports that seed as
    // UNMATCHED and then answers, measured 2026-08-24 over the same artefacts:
    //
    // | | jit only | + pyre-object | + majit-rlib |
    // |---|---|---|---|
    // | can reach a collection | 5365 | 6856 | 6899 |
    // | brackets reaching NO collection | 262 | 3 | 3 |
    // | unbracketed collecting calls, live ref | 1331 | 2066 | 2066 |
    //
    // The middle row is the one that misleads: 262 `push_roots` brackets read
    // as protecting nothing, and every one of them was a false report of a
    // pointless bracket.  Only the two `standalone_*` seeds stay unmatched with
    // all three donors, and those are the no-GC-box test configuration, so
    // their absence from a production artefact is the correct answer.
    //
    // What did NOT move across the three runs: tier 1 (10 calls in 8 fns),
    // tier 1.5 (0), and frames carried across a collecting call (0).  The
    // move-hazard verdict is donor-independent; only the liveness backlog is
    // not.
    let donors: Vec<String> = std::env::var("GC_JOIN_WITH")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut donor_llbcs: Vec<majit_charon_reader::Llbc> = Vec::with_capacity(donors.len());
    let donor_graphs: Vec<(String, framework::CallGraph)> = donors
        .iter()
        .map(|path| {
            let llbc = match majit_charon_reader::Llbc::load(path) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{path}: {e:?}");
                    std::process::exit(1);
                }
            };
            mark(&format!("donor loaded  {path}"));
            let g = framework::build(&llbc);
            mark(&format!("donor graphed {path}"));
            donor_llbcs.push(llbc);
            (path.clone(), g)
        })
        .collect();
    let donor_pairs: Vec<(&majit_charon_reader::Llbc, &framework::CallGraph)> = donor_llbcs
        .iter()
        .zip(donor_graphs.iter())
        .map(|(llbc, (_, graph))| (llbc, graph))
        .collect();
    mark("all donors graphed");

    for path in &args {
        let llbc = match majit_charon_reader::Llbc::load(path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{path}: {e:?}");
                std::process::exit(1);
            }
        };
        mark(&format!("subject loaded  {path}"));
        let cg = framework::build(&llbc);
        mark(&format!("subject graphed {path}"));
        let total = cg.names.len();
        println!("== {} ({} fun_decls) ==", llbc.crate_name(), total);
        // Which crates this artefact actually carries bodies for.  Charon
        // inlines the dependency graph, so one artefact is usually enough for
        // a cross-crate answer — but that has to be shown, not assumed.
        let mut per_crate: std::collections::BTreeMap<&str, usize> = Default::default();
        for name in cg.names.values() {
            per_crate
                .entry(name.split("::").next().unwrap_or("?"))
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
        let mut roots: Vec<(&&str, &usize)> = per_crate.iter().collect();
        roots.sort_by(|a, b| b.1.cmp(a.1));
        let shown: Vec<String> = roots
            .iter()
            .take(6)
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        println!(
            "   crates with bodies   : {} ({}{})",
            per_crate.len(),
            shown.join(" "),
            if per_crate.len() > 6 { " …" } else { "" }
        );
        // Every question below — the seeds, the closure, the opaque taint — is
        // asked of the joined graph and the answer projected back onto this
        // artefact's ids, because the liveness scan walks *these* bodies.  With
        // no donors the join is a renumbering of `cg` and nothing changes.
        let mut parts: Vec<&framework::CallGraph> = vec![&cg];
        parts.extend(donor_graphs.iter().map(|(_, g)| g));
        let joined = framework::Joined::build(&parts);
        mark("joined graph built");
        for (donor, g) in &donor_graphs {
            println!(
                "   joined with          : {} ({} fun_decls)",
                donor,
                g.names.len()
            );
        }
        if !donor_graphs.is_empty() {
            println!(
                "   joined graph          : {} nodes; {} spelling(s) merged across artefacts, \
                 {} held apart as ambiguous",
                joined.graph.names.len(),
                joined.joined_names,
                joined.ambiguous_names
            );
        }
        // A seed that matches nothing empties half the closure silently, and
        // the name table is the only place to see why.  Substring, not the
        // anchored form the seeds use, so a near-miss spelling still shows up.
        if let Ok(pat) = std::env::var("GC_NAME_GREP") {
            let mut hits: Vec<&String> = joined
                .graph
                .names
                .values()
                .filter(|n| n.contains(&pat))
                .collect();
            hits.sort();
            println!("   names containing {pat:?}: {}", hits.len());
            for n in hits.iter().take(60) {
                println!("       {n}");
            }
        }
        // "Can this particular helper collect?" is the question every
        // adjudication of an opaque finding turns on, and a reachability bit
        // is worthless without the chain behind it.
        if let Ok(pat) = std::env::var("GC_PATH_FROM") {
            let (py0, _) = joined.graph.seeds_for(PYTHON_DISPATCH_SEEDS_REF);
            let (col0, _) = joined.graph.seeds_for(framework::COLLECTING_SEEDS);
            let mut sd = py0;
            sd.extend(col0);
            let mut hits: Vec<(&u64, &String)> = joined
                .graph
                .names
                .iter()
                .filter(|(_, n)| n.contains(&pat))
                .collect();
            hits.sort_by_key(|(_, n)| n.as_str());
            for (id, n) in hits.iter().take(20) {
                match joined.graph.path_to(**id, &sd) {
                    Some(chain) => println!("   {n}\n       REACHES: {}", chain.join(" -> ")),
                    None => println!("   {n}\n       reaches no seed"),
                }
            }
        }
        println!(
            "   python-dispatch seeds : {}",
            joined.graph.seed_report(framework::PYTHON_DISPATCH_SEEDS)
        );
        println!(
            "   collecting-alloc seeds: {}",
            joined.graph.seed_report(framework::COLLECTING_SEEDS)
        );
        println!(
            "   safepoint seeds       : {}",
            joined.graph.seed_report(framework::SAFEPOINT_SEEDS)
        );
        let (jpy, _) = joined.graph.seeds_for(framework::PYTHON_DISPATCH_SEEDS);
        let (jcol, _) = joined.graph.seeds_for(framework::COLLECTING_SEEDS);
        let mut joined_seeds = jpy;
        joined_seeds.extend(jcol);
        let reach = joined.project(0, &joined.graph.reaching(&joined_seeds));
        // A wait outside the RUNNING census lets another mutator collect.
        // That justifies a bracket already open across the wait. It does not
        // enter `reach`: callers that hold a word across the wait stay out of
        // the unbracketed ratchet (`SAFEPOINT_SEEDS`).
        let (jsp, _) = joined.graph.seeds_for(framework::SAFEPOINT_SEEDS);
        let safepoint = joined.project(0, &joined.graph.reaching(&jsp));
        mark("reachability closure done");
        // The seeds as *this* artefact spells them.  The tiering below compares
        // a finding's `callee_id`, which is an id in this artefact, so it needs
        // the local set; the closure above needs the joined one.
        let (py, _) = cg.seeds_for(framework::PYTHON_DISPATCH_SEEDS);
        let (col, _) = cg.seeds_for(framework::COLLECTING_SEEDS);
        let mut seeds = py;
        seeds.extend(col);
        println!(
            "   can reach a collection: {} / {} ({}%)",
            reach.len(),
            total,
            if total == 0 {
                0
            } else {
                reach.len() * 100 / total
            }
        );
        // What the join bought, so a run without donors is never mistaken for
        // the whole-program answer this one is.
        if !donor_graphs.is_empty() {
            let alone = cg.reaching(&seeds).len();
            println!(
                "       without the join  : {alone} / {total} — the join admits {} more",
                reach.len().saturating_sub(alone)
            );
        }
        println!(
            "   unresolved-callee fns : {} (call through fn-ptr / dyn / unresolved trait)",
            cg.indirect.len()
        );

        // Judge the hand-written brackets: a root scope can only be protecting
        // against a collection if its own function can reach one.  This is a
        // *necessary* condition, so a function that fails it holds a bracket
        // that nothing in this crate can justify.
        //
        // `push_roots` opens the native source-level bracket. The translator's
        // separate `gc_push_roots` graph marker is later reduced to one
        // coloured frame per function, but that graph operation is not a Rust
        // callee in the extracted ULLBC and therefore does not belong here.
        let pin_ids: Vec<u64> = cg
            .names
            .iter()
            .filter(|(_, n)| n.ends_with("gc_roots::push_roots"))
            .map(|(&id, _)| id)
            .collect();
        if pin_ids.is_empty() {
            // Not a reason to stop: an artefact that brackets nothing is the
            // one most likely to hold defects, and the liveness scan below is
            // exactly what answers for it.
            println!("   (no gc_roots root-scope opener in this artefact's name table)");
        }
        let mut bracketed: Vec<u64> = cg
            .callees
            .iter()
            .filter(|(_, cs)| pin_ids.iter().any(|p| cs.contains(p)))
            .map(|(&id, _)| id)
            .collect();
        bracketed.sort_unstable();
        // An unresolved callee is undecidable *transitively*: a helper that
        // dispatches through a `global_hook!` function pointer makes every one
        // of its callers undecidable too, not just itself.  Without this the
        // "cannot reach a collection" bucket silently absorbs them.
        let opaque = joined.project(0, &joined.graph.reaching(&joined.graph.indirect));
        let (mut justified, mut undecidable, mut unjustified) = (0usize, 0usize, Vec::new());
        for id in &bracketed {
            if reach.contains(id) || safepoint.contains(id) {
                justified += 1;
            } else if opaque.contains(id) {
                undecidable += 1;
            } else {
                unjustified.push(*id);
            }
        }
        let mut traits: Vec<(&String, &usize)> = cg.opaque.dyn_trait.iter().collect();
        traits.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let top: Vec<String> = traits
            .iter()
            .take(8)
            .map(|(n, c)| format!("{}={c}", n.rsplit("::").next().unwrap_or(n)))
            .collect();
        println!(
            "   opaque call census    : dyn-trait={} sites over {} traits, fn-value={}, unknown={}",
            traits.iter().map(|(_, c)| **c).sum::<usize>(),
            traits.len(),
            cg.opaque.fn_value,
            cg.opaque.unknown
        );
        let mut variants: Vec<(&&str, &usize)> = cg.opaque.by_variant.iter().collect();
        variants.sort_by(|a, b| b.1.cmp(a.1));
        println!(
            "       by charon spelling: {}",
            variants
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!("       top traits: {}", top.join(" "));
        if std::env::var("GC_OPAQUE_SOURCES").is_ok() {
            let src = framework::dynamic_call_sources(&llbc);
            let mut rows: Vec<(&String, &usize)> = src.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            println!("       fn-value callee sources:");
            for (label, count) in rows.iter().take(20) {
                println!("           {count:5}  {label}");
            }
        }
        println!(
            "   fns tainted by an unresolved callee (transitive): {} / {}",
            opaque.len(),
            total
        );
        println!(
            "   functions holding a push_roots bracket: {}",
            bracketed.len()
        );
        println!("       can reach a collection (justified) : {justified}");
        println!("       hold an unresolved call (undecided): {undecidable}");
        println!(
            "       cannot reach any collection        : {}",
            unjustified.len()
        );
        for id in unjustified.iter().take(40) {
            println!(
                "           {}",
                cg.names.get(id).map_or("?", String::as_str)
            );
        }
        if unjustified.len() > 40 {
            println!("           … and {} more", unjustified.len() - 40);
        }

        // The other direction, and the one that finds defects rather than
        // overhead: a call that can collect, a GC pointer live across it, and
        // no bracket anywhere in the function.
        // How far past a named callee the movable ranking looks.  0 is the
        // shipped behaviour and the one the gate's `tier 1.5` invariant was
        // recorded against; raising it is a measurement, not a new default.
        let movable_hops: u32 = std::env::var("GC_MOVABLE_HOPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        // Asked of the JOINED graph and projected back, the same way `reach` is:
        // `w_tuple_getitem` and most of the rest of the markers are declared in
        // this artefact but bodied in a donor, so resolving them against `cg`
        // alone answers with whatever subset happens to carry a body here.
        let movable_callees = joined.project(
            0,
            &liveness::movable_callee_ids(
                &joined.graph,
                liveness::MOVABLE_GC_MARKERS,
                movable_hops,
            ),
        );
        println!(
            "   movable-addressing callees at {movable_hops} hop(s): {} \
             ({} before the join)",
            movable_callees.len(),
            liveness::movable_callee_ids(&cg, liveness::MOVABLE_GC_MARKERS, movable_hops).len()
        );
        let mut gc_tys = liveness::gc_ptr_type_ids(&llbc);
        if gc_tys.is_empty() {
            println!("   (no PyObjectRef type id found — liveness scan skipped)");
            continue;
        }
        // `Option<PyObjectRef>` locals are the same word behind a tag.  Behind
        // an env var because the gate parses this output and its baseline was
        // recorded over bare `PyObjectRef` locals only.
        if std::env::var("GC_OPTION_REFS").is_ok() {
            let opt = liveness::gc_option_type_ids(&llbc, &gc_tys);
            println!("   Option<PyObjectRef> type ids: {}", opt.len());
            gc_tys.extend(opt);
        }
        if std::env::var("GC_SLICE_ARGS").is_ok() {
            let slices = liveness::gc_slice_type_ids(&llbc, &gc_tys);
            println!("   &[PyObjectRef] type ids: {}", slices.len());
            gc_tys.extend(slices);
        }
        let push_root_ids: std::collections::HashSet<u64> = pin_ids.iter().copied().collect();
        let (found, stats) = liveness::scan(
            &llbc,
            &cg,
            &reach,
            &push_root_ids,
            &gc_tys,
            &movable_callees,
            &donor_pairs,
        );
        mark("scan 1/4 (resolved, gc locals)");
        println!(
            "   liveness scan: {} bodies; {} with a terminator this reader could not parse; \
             {} call(s) withheld as dominated by a push_roots",
            stats.bodies_scanned, stats.unparsed_terminator_bodies, stats.withheld_under_a_bracket
        );
        println!(
            "       and {} body/bodies with a statement this reader could not parse",
            stats.unparsed_statement_bodies
        );
        // What the `movable` columns below had to work with.  A zero says the
        // ranking never had an input, which is not the same answer as a clean
        // corpus and must not be read as one.
        println!(
            "       movable-argument supply: {} bodies hand {} GC pointer(s) to one",
            stats.bodies_with_movable_args, stats.movable_arg_locals
        );
        // The withheld figure above says a bracket dominates the call.  It does
        // not say the root the call needed is in that bracket, and nothing has
        // ever asked: `postprocess_double_check` asserts exactly this upstream,
        // after `shadowcolor.py` has run, whereas every bracket in this
        // artefact is hand-written and ungraded.  A bracket read as coverage
        // while pinning the wrong set is worse than no bracket, because it also
        // removes the call from the finding count above.
        println!(
            "       of those withheld: {} pin every live pointer, {} are SHORT a root, \
             {} could not be read",
            stats.withheld_bracket_covers,
            stats.withheld_bracket_short,
            stats.withheld_contents_opaque
        );
        // Overlapping root scopes are read rather than withheld, so this is a
        // measurement of how much of the unread set happens to be nested, not
        // of what nesting costs.  Behind an env var because the gate parses
        // this output by regex and every pattern there must match exactly once.
        if std::env::var("GC_NESTED_CENSUS").is_ok() {
            println!(
                "           nested-scope census: {} of {} bodies hold two live root \
                 scopes, and {} of the {} unread calls come from one",
                stats.bodies_with_nested_scopes,
                stats.bodies_scanned,
                stats.withheld_opaque_from_nested,
                stats.withheld_contents_opaque
            );
        }
        println!(
            "           of the SHORT, {} miss a root the body produced itself \
             (no caller's bracket can be covering those)",
            stats.withheld_bracket_short_body_local
        );
        println!(
            "           of the SHORT, {} miss a root this body later addresses \
             as a list/dict (a caller's pin cannot rescue those)",
            stats.withheld_bracket_short_movable
        );
        for sb in stats.short_brackets.iter().take(20) {
            println!(
                "           SHORT {}:{} {} -> {}  missing [{}]  body-local [{}]  movable [{}]  pinned [{}]",
                sb.file,
                sb.line,
                sb.func_name,
                sb.callee_name,
                sb.missing.join(", "),
                sb.missing_local.join(", "),
                sb.missing_movable.join(", "),
                sb.pinned.join(", ")
            );
        }
        if stats.short_brackets.len() > 20 {
            println!(
                "           ... and {} more (set GC_SHORT_BRACKETS_JSON to read them all)",
                stats.short_brackets.len() - 20
            );
        }
        // Item 11 of the GC advisory: a site that pins and then goes on using
        // the local it passed in, rather than the word the pin handed back.
        // `let _ = pin_root(x)` is the sanctioned spelling for a liveness-only
        // pin, and writing it asserts that x's kind never moves; the `movable`
        // column is that assertion checked.
        let by_pin: std::collections::BTreeMap<&str, usize> =
            stats
                .stale_pin_reads
                .iter()
                .fold(Default::default(), |mut m, r| {
                    *m.entry(r.pin_name.rsplit("::").next().unwrap_or("?"))
                        .or_default() += 1;
                    m
                });
        println!(
            "   pins whose argument the body still reads afterwards: {} ({} reading a local \
             later addressed as a list/dict)",
            stats.pin_arg_read_after, stats.pin_arg_read_after_movable
        );
        println!(
            "       by pin: {}",
            by_pin
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for r in stats
            .stale_pin_reads
            .iter()
            .filter(|r| !r.movable.is_empty())
            .take(20)
        {
            println!(
                "           STALE-PIN {}:{} {} [{}]  movable [{}]  via {}",
                r.file,
                r.line,
                r.func_name,
                r.locals.join(", "),
                r.movable.join(", "),
                r.pin_name.rsplit("::").next().unwrap_or("?")
            );
        }
        // Two filters sit between the count above and the rows just printed:
        // only a non-empty `movable` column prints, and only the first 20 of
        // those. A run whose stale pins are all non-movable prints the count
        // and no rows at all, so say where the rest are — the short-bracket
        // block above owes and prints the same hint.
        let stale_pins_shown = stats
            .stale_pin_reads
            .iter()
            .filter(|r| !r.movable.is_empty())
            .count()
            .min(20);
        if stats.stale_pin_reads.len() > stale_pins_shown {
            println!(
                "           ... and {} more (set GC_STALE_PIN_JSON to read them all)",
                stats.stale_pin_reads.len() - stale_pins_shown
            );
        }
        if let Ok(path) = std::env::var("GC_STALE_PIN_JSON") {
            let mut out = String::new();
            for r in &stats.stale_pin_reads {
                let row = serde_json::json!({
                    "file": r.file, "line": r.line, "func": r.func_name,
                    "pin": r.pin_name, "locals": r.locals, "movable": r.movable,
                });
                out.push_str(&row.to_string());
                out.push('\n');
            }
            match write_rows(&path, &out, &mut opened) {
                Ok(()) => println!(
                    "       wrote {} stale-pin read(s) to {path}",
                    stats.stale_pin_reads.len()
                ),
                Err(e) => {
                    println!("       FAILED to write {path}: {e}");
                    write_failed = true;
                }
            }
        }
        // Why the unread brackets went unread.  Behind an env var for the same
        // reason as the nested-scope census: the gate matches this output by
        // regex.
        if let Ok(path) = std::env::var("GC_OPAQUE_JSON") {
            for (reason, n) in &stats.opaque_by_reason {
                println!("           unread because {reason}: {n} call(s)");
            }
            let mut out = String::new();
            for (func, file, reason, calls) in &stats.opaque_bodies {
                let row = serde_json::json!({
                    "func": func, "file": file, "reason": reason, "calls": calls,
                });
                out.push_str(&row.to_string());
                out.push('\n');
            }
            match write_rows(&path, &out, &mut opened) {
                Ok(()) => println!(
                    "       wrote {} unread bracket body/bodies to {path}",
                    stats.opaque_bodies.len()
                ),
                Err(e) => {
                    println!("       FAILED to write {path}: {e}");
                    write_failed = true;
                }
            }
        }
        if let Ok(path) = std::env::var("GC_SHORT_BRACKETS_JSON") {
            let mut out = String::new();
            for sb in &stats.short_brackets {
                let row = serde_json::json!({
                    "file": sb.file, "line": sb.line, "func": sb.func_name,
                    "callee": sb.callee_name,
                    "missing": sb.missing,
                    "missing_local": sb.missing_local,
                    "missing_movable": sb.missing_movable,
                    "pinned": sb.pinned,
                });
                out.push_str(&row.to_string());
                out.push('\n');
            }
            match write_rows(&path, &out, &mut opened) {
                Ok(()) => println!(
                    "       wrote {} short bracket(s) to {path}",
                    stats.short_brackets.len()
                ),
                Err(e) => {
                    println!("       FAILED to write {path}: {e}");
                    write_failed = true;
                }
            }
        }
        // The resolved graph is an *under*-approximation of what collects: a
        // call whose dispatch edge is unresolved is excluded, so a clean
        // resolved census is not a clean census.  Re-run with the opaque set
        // folded in and report both, rather than let the difference go unsaid.
        let mut conservative = reach.clone();
        conservative.extend(opaque.iter().copied());
        let (found_conservative, stats_conservative) = liveness::scan(
            &llbc,
            &cg,
            &conservative,
            &push_root_ids,
            &gc_tys,
            &movable_callees,
            &donor_pairs,
        );
        mark("scan 2/4 (opaque-folded, gc locals)");
        if let Ok(path) = std::env::var("GC_CONSERVATIVE_JSON") {
            let mut out = String::new();
            for f in &found_conservative {
                let row = serde_json::json!({
                    "file": f.file, "line": f.line, "func": f.func_name,
                    "callee": f.callee_name,
                    "movable": f.movable_use,
                    "live_non_arg": f.live_non_arg,
                    "live_arg": f.live_arg,
                });
                out.push_str(&row.to_string());
                out.push('\n');
            }
            match write_rows(&path, &out, &mut opened) {
                Ok(()) => println!(
                    "       wrote {} conservative finding(s) to {path}",
                    found_conservative.len()
                ),
                Err(e) => {
                    println!("       FAILED to write {path}: {e}");
                    write_failed = true;
                }
            }
        }
        let conservative_fns: std::collections::BTreeSet<&str> = found_conservative
            .iter()
            .map(|f| f.func_name.as_str())
            .collect();
        let mut by_fn: std::collections::BTreeMap<&str, Vec<&liveness::Finding>> =
            Default::default();
        for f in &found {
            by_fn.entry(f.func_name.as_str()).or_default().push(f);
        }
        let non_arg_fns: Vec<&&str> = by_fn
            .iter()
            .filter(|(_, v)| v.iter().any(|f| !f.live_non_arg.is_empty()))
            .map(|(k, _)| k)
            .collect();
        println!(
            "   unbracketed calls that can collect with a live PyObjectRef: {} in {} fn(s)",
            found.len(),
            by_fn.len()
        );
        println!(
            "       counting unresolved dispatch as collecting too: {} in {} fn(s)",
            found_conservative.len(),
            conservative_fns.len()
        );
        // The conservative scan covers a superset of bodies, so its own
        // withheld and unparsed figures are the ones its finding count has to
        // be read against; quoting the resolved scan's line next to it would
        // pair a count with another scan's accounting.
        println!(
            "           over {} bodies; {} with an unparsable terminator, {} with an \
             unparsable statement; {} call(s) withheld under a push_roots",
            stats_conservative.bodies_scanned,
            stats_conservative.unparsed_terminator_bodies,
            stats_conservative.unparsed_statement_bodies,
            stats_conservative.withheld_under_a_bracket
        );
        println!(
            "       of which hold a NON-ARGUMENT live pointer: {} fn(s)",
            non_arg_fns.len()
        );
        // The findings as data rather than as a report.  The printed lines
        // above are shaped for a reader and are parsed positionally by
        // `scripts/check-gc-root-brackets.py`; a rewriter needs the site
        // itself -- which file, which line, which callee, and the names to
        // bracket -- and needs it to survive a change to the prose.
        if let Ok(path) = std::env::var("GC_FINDINGS_JSON") {
            let mut out = String::new();
            for f in &found {
                let row = serde_json::json!({
                    "file": f.file,
                    "line": f.line,
                    "func": f.func_name,
                    "callee": f.callee_name,
                    // The two live columns stay apart: upstream's
                    // `get_livevars_for_roots` drops the call's own arguments
                    // for a moving GC, and whether pyre may do the same is the
                    // divergence this module's header records.
                    "live_non_arg": f.live_non_arg,
                    "live_arg": f.live_arg,
                    "movable_use": f.movable_use,
                });
                out.push_str(&row.to_string());
                out.push('\n');
            }
            match write_rows(&path, &out, &mut opened) {
                Ok(()) => println!("       wrote {} finding(s) to {path}", found.len()),
                // A report that says it wrote and did not is worse than one
                // that fails, so this is loud even though the scan itself
                // succeeded, and the run's exit status carries it too.
                Err(e) => {
                    println!("       FAILED to write {path}: {e}");
                    write_failed = true;
                }
            }
        }
        // Tier 1: the callee is *itself* a dispatch seed, so "this call runs
        // Python" needs no transitive argument.  Tier 2 reaches Python only
        // through further calls and is far likelier to be a false positive.
        let mut tier1: Vec<&liveness::Finding> = found
            .iter()
            .filter(|f| seeds.contains(&f.callee_id) && !f.live_non_arg.is_empty())
            .collect();
        tier1.sort_by(|a, b| a.func_name.cmp(&b.func_name).then(a.line.cmp(&b.line)));
        let t1_fns: std::collections::BTreeSet<&str> =
            tier1.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "       tier 1 (callee IS a dispatch seed): {} call(s) in {} fn(s)",
            tier1.len(),
            t1_fns.len()
        );
        if std::env::var("GC_LIVENESS_TIER1").is_ok() {
            for f in &tier1 {
                println!(
                    "           {}:{}  across {}  live: {:?}",
                    f.func_name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.live_non_arg
                );
            }
        }
        // Tier 1.5: not a direct dispatch seed, but a live pointer that this
        // body later hands to a `list`/`dict` accessor — the shape both
        // confirmed defects in `list.index` and `mul` had.
        let mut tier15: Vec<&liveness::Finding> = found
            .iter()
            .filter(|f| !seeds.contains(&f.callee_id) && !f.movable_use.is_empty())
            .collect();
        tier15.sort_by(|a, b| a.func_name.cmp(&b.func_name).then(a.line.cmp(&b.line)));
        let t15_fns: std::collections::BTreeSet<&str> =
            tier15.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "       tier 1.5 (live ptr later addressed as list/dict): {} call(s) in {} fn(s)",
            tier15.len(),
            t15_fns.len()
        );
        let t15_conservative = found_conservative
            .iter()
            .filter(|f| !seeds.contains(&f.callee_id) && !f.movable_use.is_empty())
            .count();
        println!("       tier 1.5 counting unresolved dispatch too: {t15_conservative} call(s)");
        if std::env::var("GC_LIVENESS_TIER15").is_ok() {
            for f in &tier15 {
                println!(
                    "           {}:{}  across {}  movable-use: {:?}",
                    f.func_name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.movable_use
                );
                match cg.path_to(f.callee_id, &seeds) {
                    Some(chain) => println!("               path: {}", chain.join(" -> ")),
                    None => println!("               path: (none — reached only via a seed alias)"),
                }
            }
        }
        let show: usize = std::env::var("GC_LIVENESS_SHOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        for name in non_arg_fns.iter().take(show) {
            // Every such call in the body, not the first one found.  A single
            // row per function hides a second unrooted use further down the
            // same body — and that is precisely the one a repair pass, working
            // from this list, then leaves behind.
            for f in by_fn[**name].iter().filter(|f| !f.live_non_arg.is_empty()) {
                println!(
                    "           {}:{}  across {}  live: {:?}",
                    name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.live_non_arg
                );
            }
        }

        // The same question asked of the running frame.  A `PyObjectRef` has a
        // root stack to be pinned on; the frame is carried as a bare
        // `&mut PyFrame` no walker reaches, so the only repair is to re-read it
        // out of a `FrameAnchor` after the call.  That reload kills the stale
        // local at the call, so no bracket set is passed — a body that reloads
        // correctly simply has nothing live across the call to report.
        let frame_tys = liveness::frame_ptr_type_ids(&llbc);
        if frame_tys.is_empty() {
            println!("   (no PyFrame pointer type id found — frame scan skipped)");
            continue;
        }
        let no_bracket: std::collections::HashSet<u64> = Default::default();
        // The frame has no list/dict-addressing call to rank by: a stale frame
        // is read back through `FrameAnchor`, not dereferenced as a container.
        let no_movable: std::collections::HashSet<u64> = Default::default();
        let (frames, frame_stats) = liveness::scan(
            &llbc,
            &cg,
            &reach,
            &no_bracket,
            &frame_tys,
            &no_movable,
            &donor_pairs,
        );
        mark("scan 3/4 (resolved, frame locals)");
        let (frames_conservative, _) = liveness::scan(
            &llbc,
            &cg,
            &conservative,
            &no_bracket,
            &frame_tys,
            &no_movable,
            &donor_pairs,
        );
        mark("scan 4/4 (opaque-folded, frame locals)");
        let frame_fns: std::collections::BTreeSet<&str> =
            frames.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "   frame carried across a call that can collect: {} in {} fn(s)",
            frames.len(),
            frame_fns.len()
        );
        println!(
            "       counting unresolved dispatch as collecting too: {} call(s)",
            frames_conservative.len()
        );
        println!(
            "           over {} bodies; {} with an unparsable terminator, {} with an \
             unparsable statement",
            frame_stats.bodies_scanned,
            frame_stats.unparsed_terminator_bodies,
            frame_stats.unparsed_statement_bodies
        );
        // The frame reaches the callee as an argument in most of these, and an
        // argument goes just as stale as any other local here (the module
        // header records why upstream can drop them and pyre cannot).  Both
        // columns are therefore hazards; they are split only because a frame
        // the call does not even see is the more obviously wrong shape.
        let mut frame_tier1: Vec<&liveness::Finding> = frames
            .iter()
            .filter(|f| seeds.contains(&f.callee_id))
            .collect();
        frame_tier1.sort_by(|a, b| a.func_name.cmp(&b.func_name).then(a.line.cmp(&b.line)));
        let ft1_fns: std::collections::BTreeSet<&str> =
            frame_tier1.iter().map(|f| f.func_name.as_str()).collect();
        println!(
            "       tier 1 (callee IS a dispatch seed): {} call(s) in {} fn(s)",
            frame_tier1.len(),
            ft1_fns.len()
        );
        if let Ok(filter) = std::env::var("GC_FRAME_SCAN") {
            // A whole-crate list is not a work item; the filter is how a
            // surface (`OpcodeStepExecutor`, say) is cut out of it and read.
            // The resolved rows are marked, so the ones only the conservative
            // scan finds — a body whose dispatch this reader cannot follow —
            // are adjudicated as the weaker claim they are rather than read
            // beside the others.
            let resolved: std::collections::HashSet<(u64, u64, u64)> = frames
                .iter()
                .map(|f| (f.func, f.line, f.callee_id))
                .collect();
            for f in &frames_conservative {
                if !filter.is_empty() && !f.func_name.contains(&filter) {
                    continue;
                }
                let mark = if resolved.contains(&(f.func, f.line, f.callee_id)) {
                    "resolved"
                } else {
                    "opaque  "
                };
                println!(
                    "           [{mark}] {}:{}  across {}  live: {:?} arg: {:?}",
                    f.func_name,
                    f.line,
                    f.callee_name.rsplit("::").next().unwrap_or(""),
                    f.live_non_arg,
                    f.live_arg
                );
            }
        }
    }
    if write_failed {
        std::process::exit(1);
    }
}
