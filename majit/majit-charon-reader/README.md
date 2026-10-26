# majit-charon-reader

Stable-Rust parser for Charon `.llbc` / `.ullbc` JSON artefacts.

Consumed by `majit-translate`'s MIR-driven flowspace driver. Produced
by `scripts/extract-llbc.py` running the pinned Charon binary that
`scripts/install-charon.py` installs.

## Usage

```rust
use majit_charon_reader::{Llbc, ullbc::{CallClass, TermKind}};

let llbc = Llbc::load("target/llbc/pyre-object.ullbc")?;
println!("crate = {}", llbc.crate_name());

for fd in llbc.iter_local_fns() {
    let Some(u) = fd.unstructured() else { continue };
    for bb in &u.body {
        if let Ok(TermKind::Call { call, .. }) = bb.term() {
            match call.func.classify() {
                CallClass::Direct  => { /* monomorphized fn call */ }
                CallClass::Trait   => { /* trait-bound generic */ }
                CallClass::Dynamic => { /* dyn Trait virtual call */ }
                CallClass::Ptr     => { /* function-pointer call */ }
                CallClass::Unknown => { /* fail-loud */ }
            }
        }
    }
}
```

## Schema policy

The reader covers the subset of Charon ULLBC that the flowspace driver
actually consumes. Fields the driver does not read remain
`serde_json::Value` so that newer Charon versions stay loadable
without code changes. Variants we typed are listed by name; every
enum has a `#[serde(other)] Unknown` arm so unrecognised future
variants surface as fail-loud `Unknown` instead of being silently
dropped.

Run the diagnostic example to spot a fresh schema gap:

```sh
cargo run --example dump_stats -p majit-charon-reader -- \
    target/llbc/pyre-object.ullbc
cargo run --example probe_errors -p majit-charon-reader -- \
    target/llbc/pyre-object.ullbc        # error tally per outer variant
```

Add a new typed variant + a regression test under `tests/corpus.rs`
whenever a new construct appears.

## Validation against real crates

The reader is validated against the corpus snapshot (checked in at
`majit/charon-corpus/corpus.ullbc`) via `tests/corpus.rs`, and against
the full extracted `pyre-object.ullbc` (23 MB) / `pyre-interpreter.ullbc`
(133 MB) snapshots not committed to git (regenerable via
`scripts/extract-llbc.py`). Every statement and terminator in every
extracted body decodes:

| crate            | bodies | stmt decode errors | term decode errors |
|------------------|-------:|-------------------:|--------------------:|
| corpus           |      5 |                  0 |                   0 |
| pyre-object      |  1718  |                  0 |                   0 |
| pyre-interpreter |  5435  |                  0 |                   0 |

The pyre-interpreter `Call` terminators classify as:

```text
  direct    25502   (Fun.Regular: monomorphized direct call)
  trait       382   (Fun.Trait: trait-bound generic, statically resolved)
  dynamic      36   (Dynamic: dyn Trait virtual call)
```

The 36 `dynamic` count is the audited `dyn Trait` virtual-call total
across the JIT-consumed crates (`pyre-object` / `pyre-interpreter` /
`pyre-module`); see the dyn-Trait classification in
`majit-translate/src/front/mod.rs`.
