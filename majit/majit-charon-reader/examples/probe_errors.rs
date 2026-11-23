//! Probe `.ullbc` for schema variants we haven't typed yet.
use majit_charon_reader::Llbc;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: probe_errors <file.ullbc>");
        std::process::exit(2);
    });
    let llbc = Llbc::load(&path).unwrap();

    let mut errors: BTreeMap<String, usize> = BTreeMap::new();
    let mut samples: BTreeMap<String, String> = BTreeMap::new();
    for fd in llbc.iter_local_fns() {
        let Some(u) = fd.unstructured() else { continue };
        for bb in u.body() {
            for st in &bb.statements {
                if let Err(e) = st.stmt_kind() {
                    let outer = outer_key(st.kind_value());
                    let key = format!("[stmt:{outer}] {}", msg(&e));
                    *errors.entry(key.clone()).or_default() += 1;
                    samples
                        .entry(key)
                        .or_insert_with(|| short_sample(st.kind_value()));
                }
            }
            if let Err(e) = bb.term(&llbc) {
                let raw = bb.terminator.kind_value();
                let outer = outer_key(raw);
                let key = format!("[term:{outer}] {}", msg(&e));
                *errors.entry(key.clone()).or_default() += 1;
                samples.entry(key).or_insert_with(|| short_sample(raw));
            }
        }
    }
    println!("error tally:");
    for (k, c) in &errors {
        println!("  {c:5}  {k}");
        if let Some(s) = samples.get(k) {
            println!("        sample: {s}");
        }
    }
}

fn outer_key(v: &serde_json::Value) -> String {
    if let Some(s) = v.as_str() {
        return format!("\"{s}\"");
    }
    if let Some(obj) = v.as_object()
        && let Some(k) = obj.keys().next()
    {
        return k.clone();
    }
    "?".into()
}

fn msg(e: &str) -> &str {
    e.split(';').next().unwrap_or(e).trim()
}

fn short_sample(v: &serde_json::Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_default();
    // Truncate on a char boundary: `&s[..240]` would panic if byte 240
    // splits a multibyte UTF-8 scalar (serde_json emits raw UTF-8).
    if s.chars().count() > 240 {
        let prefix: String = s.chars().take(240).collect();
        format!("{prefix}…")
    } else {
        s
    }
}
