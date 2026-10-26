//! Diagnostic tool — find how a root function reaches a target callee.
//!
//! ```sh
//! cargo run --release --example who_calls -p majit-charon-reader -- \
//!     baseobjspace::isinstance vec::Vec::with_capacity \
//!     target/llbc/pyre-interpreter.ullbc target/llbc/pyre-object.ullbc
//! ```
//!
//! Loads every `.ullbc` given, builds the direct-call graph over their merged
//! function tables (matching cross-crate edges by flattened name), and prints
//! a shortest call path from each function whose name contains `<root>` to the
//! first callee whose name contains `<target>`.
use majit_charon_reader::{
    Llbc,
    ullbc::{CallFunc, CallKind, FunId, TermKind},
};
use std::collections::{HashMap, HashSet, VecDeque};

fn main() {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .expect("usage: who_calls <root> <target> <file.ullbc>...");
    let target = args.next().expect("target");
    let paths: Vec<String> = args.collect();
    assert!(!paths.is_empty(), "at least one .ullbc is required");

    // name -> direct callee names, merged across crates.
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();
    for path in &paths {
        let llbc = Llbc::load(path).unwrap();
        let name_of = |func: &CallFunc| -> Option<String> {
            match func {
                CallFunc::Regular(r) => match &r.kind {
                    CallKind::Fun(FunId::Regular { id }) => {
                        llbc.fn_by_id(*id).map(|f| f.item_meta.name_path())
                    }
                    _ => None,
                },
                _ => None,
            }
        };
        for fd in llbc.iter_local_fns() {
            let Some(u) = fd.unstructured() else { continue };
            let entry = graph.entry(fd.item_meta.name_path()).or_default();
            for bb in &u.body {
                if let Ok(TermKind::Call { call, .. }) = bb.term(&llbc)
                    && let Some(callee) = name_of(&call.func)
                {
                    entry.push(callee);
                }
            }
        }
        eprintln!("loaded {path}: {} fns in graph", graph.len());
    }

    let roots: Vec<String> = graph
        .keys()
        .filter(|k| k.contains(&root))
        .cloned()
        .collect();
    for start in roots {
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut seen: HashSet<String> = HashSet::from([start.clone()]);
        let mut queue = VecDeque::from([start.clone()]);
        let mut hit = None;
        while let Some(cur) = queue.pop_front() {
            for callee in graph.get(&cur).map(Vec::as_slice).unwrap_or(&[]) {
                if !seen.insert(callee.clone()) {
                    continue;
                }
                prev.insert(callee.clone(), cur.clone());
                if callee.contains(&target) {
                    hit = Some(callee.clone());
                    break;
                }
                queue.push_back(callee.clone());
            }
            if hit.is_some() {
                break;
            }
        }
        match hit {
            None => println!("── {start}\n   (no path to {target})"),
            Some(end) => {
                let mut chain = vec![end.clone()];
                let mut cur = end;
                while let Some(p) = prev.get(&cur) {
                    chain.push(p.clone());
                    cur = p.clone();
                }
                chain.reverse();
                println!("── {start}");
                for (i, step) in chain.iter().enumerate() {
                    println!("   {}{}", "  ".repeat(i), step);
                }
            }
        }
    }
}
