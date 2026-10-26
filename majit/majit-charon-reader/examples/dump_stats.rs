//! Diagnostic tool — load a `.ullbc` file and dump summary statistics.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example dump_stats -p majit-charon-reader -- target/llbc/pyre-object.ullbc
//! ```
//!
//! Prints crate name, function counts (decoded / error / opaque), and
//! a per-call-class breakdown for all `Call` terminators in the
//! extracted bodies. Useful for sanity-checking a fresh extraction
//! and for spotting regressions when bumping the Charon pin.

use majit_charon_reader::{
    Llbc,
    ullbc::{CallClass, StmtKind, TermKind},
};
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dump_stats <file.ullbc>");
        std::process::exit(2);
    });

    let llbc = Llbc::load(&path).unwrap_or_else(|e| {
        eprintln!("error: cannot load {path}: {e}");
        std::process::exit(1);
    });

    println!("file:            {path}");
    println!("crate:           {}", llbc.crate_name());
    println!("charon_version:  {}", llbc.file.charon_version);
    println!("has_errors:      {}", llbc.file.has_errors);

    let mut bodies = 0usize;
    let mut errors = 0usize;
    let mut opaque = 0usize;
    let mut stmt_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut term_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut call_classes: BTreeMap<&'static str, usize> = BTreeMap::new();

    let mut total_fns = 0usize;
    for fd in llbc.iter_local_fns() {
        total_fns += 1;
        if fd.error_message().is_some() {
            errors += 1;
            continue;
        }
        let Some(u) = fd.unstructured() else {
            opaque += 1;
            continue;
        };
        bodies += 1;
        for bb in &u.body {
            for st in &bb.statements {
                let label = match st.stmt_kind() {
                    Ok(StmtKind::StorageLive(_)) => "StorageLive",
                    Ok(StmtKind::StorageDead(_)) => "StorageDead",
                    Ok(StmtKind::Assign(..)) => "Assign",
                    Ok(StmtKind::Assert(..)) => "Assert (stmt)",
                    Ok(StmtKind::PlaceMention(_)) => "PlaceMention",
                    Ok(StmtKind::Borrowck(_)) => "Borrowck",
                    Ok(StmtKind::Unknown) => "Unknown",
                    Err(_) => "DecodeError",
                };
                *stmt_kinds.entry(label).or_default() += 1;
            }
            let term_label = match bb.term(&llbc) {
                Ok(TermKind::Return) => "Return",
                Ok(TermKind::UnwindResume) => "UnwindResume",
                Ok(TermKind::UnwindTerminate) => "UnwindTerminate",
                Ok(TermKind::Abort(_)) => "Abort",
                Ok(TermKind::Panic { .. }) => "Panic",
                Ok(TermKind::UndefinedBehavior) => "UndefinedBehavior",
                Ok(TermKind::Goto { .. }) => "Goto",
                Ok(TermKind::Switch { .. }) => "Switch",
                Ok(TermKind::Call { call, .. }) => {
                    let cls = match call.func.classify() {
                        CallClass::Direct => "direct",
                        CallClass::Trait => "trait",
                        CallClass::Dynamic => "dynamic",
                        CallClass::Ptr => "ptr",
                        CallClass::Unknown => "unknown",
                    };
                    *call_classes.entry(cls).or_default() += 1;
                    "Call"
                }
                Ok(TermKind::Assert { .. }) => "Assert (term)",
                Ok(TermKind::Drop { .. }) => "Drop",
                Ok(TermKind::Unknown) => "Unknown",
                Err(_) => "DecodeError",
            };
            *term_kinds.entry(term_label).or_default() += 1;
        }
    }

    println!();
    println!("functions:       {total_fns} total");
    println!("                  - {bodies} decoded bodies");
    println!("                  - {errors} translation errors (e.g. thread_local!)");
    println!("                  - {opaque} opaque / structured-only");

    println!();
    println!("statement kinds:");
    for (k, c) in &stmt_kinds {
        println!("  {k:18} {c}");
    }
    println!();
    println!("terminator kinds:");
    for (k, c) in &term_kinds {
        println!("  {k:18} {c}");
    }
    println!();
    println!("call classes (of `Call` terminators):");
    for (k, c) in &call_classes {
        println!("  {k:18} {c}");
    }
}
