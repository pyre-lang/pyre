# Charon Corpus

Regression fixtures for the Charon ULLBC reader and MIR frontend work from
[issue #97](https://github.com/youknowone/pyre/issues/97).

The checked-in corpus records representative Charon output so reader and
frontend changes can be tested without re-extracting the full pyre crates.

## Layout

```text
charon-corpus/
├── README.md              # this file
├── Cargo.toml             # the micro-crate (representative shapes)
├── src/lib.rs
├── corpus.ullbc           # extracted ULLBC for the crate (checked in)
└── inspect_llbc.py        # JSON dumper used to produce the findings below
```

## Setup notes (Charon install)

Use the canonical fetcher:

```sh
scripts/install-charon.py
../.pyre-build/charon/<platform>/<version>/charon toolchain-path  # one-time nightly install (~1 min)
```

The script pins `CHARON_VERSION_DEFAULT="nightly-2026.10.25"` and installs
to a shared cache at `../.pyre-build/charon/<platform>/<version>` by default,
so sibling worktrees on the same pin reuse the binary and an older pin keeps
its own directory. Override with `PYRE_SHARED_BUILD=/path/to/cache` or
`CHARON_DEST=/path/to/bin`; use
`CHARON_VERSION=nightly-YYYY.MM.DD scripts/install-charon.py` to bump.

Charon itself follows that nightly tag. It internally pins Rust toolchain
`nightly-2026-09-17` (`rustc-dev`, `llvm-tools-preview`, `rust-src`,
`miri` components); `charon toolchain-path` auto-installs the
toolchain on first run via rustup. There is no formal stable release
tag — every release is a nightly tag.

The extraction toolchain is **separate from** the toolchain our codebase
compiles with. The downstream stable-Rust consumer never needs the pinned
nightly; it only needs the `.llbc` JSON.

## Reproducing

```sh
# 0. install Charon (once)
../../scripts/install-charon.py

# 1. regenerate the checked-in fixture, in place (run from this directory).
#    <charon> is ../../../.pyre-build/charon/<platform>/<version>/charon by default —
#    the location `install-charon.py` writes to.
<charon> cargo --ullbc --reconstruct-panic-calls --inline-anon-consts --dest-file "$PWD/corpus.ullbc"

# (`../../scripts/extract-llbc.py corpus` runs the same extraction through the
#  fingerprinting driver, but its artefact lands in `target/llbc/corpus.ullbc`,
#  not here; the checked-in fixture is only updated by writing this path.)

# 2. inspect
python3 inspect_llbc.py corpus.ullbc                  # summary
python3 inspect_llbc.py corpus.ullbc desugar_mix      # detailed BB dump

# 3. run the reader/frontend regression tests that consume corpus.ullbc
cargo test -p majit-charon-reader
cargo test -p majit-translate --test test_mir_frontend
```

## Corpus

Four functions from `src/lib.rs`, chosen to cover the shapes called
out in issue #97:

| Function            | Shape                                       | ULLBC BBs |
|---------------------|---------------------------------------------|-----------|
| `straight_line_add` | straight-line arithmetic                    | 7         |
| `branch_loop_sum`   | `for` loop + `if/else` branch               | 16        |
| `strategy_len`      | enum `match` (dict-strategy stand-in)       | 5         |
| `parse_one`         | `match` with guards, internal helper for #4 | 8         |
| `desugar_mix`       | `?` + `for` + `match` + `break`             | 29        |

The corpus also includes a header-first object model. These functions pin
lowering decisions that otherwise fail by leaving a residual call or an
untyped field access rather than raising an error, so each is pinned by an
assertion in `majit-translate/tests/test_mir_frontend.rs`:

| Function              | Premise it pins                                       |
|-----------------------|-------------------------------------------------------|
| `w_object_type`       | `(*w).ob_type` narrows to a *typed* `FieldRead`        |
| `w_new_int`           | the boxing cluster fuses to one `NewWithVtable`        |
| `w_new_type_only_int` | so does a cluster whose header declares no class word  |
| `w_number_add`        | a narrowing-chain arm lowers to a direct `FunctionPath` call |

`ObjectHeader` has both `ob_type` and a per-instance `w_class`.
`TypeOnlyHeader` has only `ob_type`, matching RPython's root `OBJECT`, whose
only data field is `typeptr`. The two-word allocation can fuse when its
`w_class` is derived from the same class object as `ob_type`; the one-word
allocation can fuse from its declared layout because no class word exists to
disagree with the type pointer.

Both shapes are carried because `resolve_header_plan` has an arm for each, and
a corpus holding only the two-word shape can never reach the second: the match
lands on the stored `w_class` before `header_declares_no_class_word` is
consulted.

The fixture's `object_model::get_instantiate` helper pins the single-argument
class-lookup contract recognized by `model.rs`. The one-word header does not
use this helper. `_immutable_fields_W_IntObject` preserves the marker shape
harvested by `front::llbc_hints`; `lower_function` runs upstream of that
consumer, so no assertion in `test_mir_frontend.rs` covers it. Every other
identity in the corpus is named after no host, so a fixture cannot tell a
general recogniser apart from one conformed to a particular host.

`corpus.ullbc` is regenerated in place whenever this fixture is edited, and it
embeds the source verbatim, so counts taken against an older artefact — op
censuses, lowered-graph totals, local-fn counts — are not comparable across a
regeneration. Re-measure rather than subtract.

## Findings

### 1. `.llbc` top-level shape

```json
{
  "charon_version": "0.1.196",
  "has_errors": false,
  "translated": {
    "crate_name": "...",
    "options": {...},
    "target_information": {...},
    "files": [...],
    "item_names": [...],
    "type_decls":  [...],
    "fun_decls":   [...],
    "global_decls":[...],
    "trait_decls": [...],
    "trait_impls": [...],
    "ordered_decls": {...}
  }
}
```

- Types are **deduplicated globally** via `{"Deduplicated": <id>}` references
  into a parallel pool. Inline forms appear as
  `{"HashConsedValue": [<id>, <ty>]}` for the first occurrence.
- `fun_decls` includes **opaque references** to functions defined in other
  crates (their bodies are `null`). In the corpus crate, **5 of 213
  `fun_decls` carry our local bodies**; the rest are pulled in by name from
  `core` because they appear in call positions (iterator/Try/etc.).

### 2. Per-function body — ULLBC (basic-block CFG)

`charon cargo --ullbc` produces the unstructured form, which is the
direct analog of MIR and the right shape for issue #97's BFS driver. The
default `charon cargo` output reconstructs control-flow into a structured
nested AST and is **not** what we want.

ULLBC body shape:

```json
"body": {
  "Unstructured": {
    "span":             { "data": { "file_id": ..., "beg": {...}, "end": {...}}},
    "bound_body_regions": 0,
    "locals": {
      "arg_count": <n>,
      "locals": [{ "index": i, "name": null|"s", "span": {...}, "ty": ... }]
    },
    "body": [<BasicBlock>...]
  }
}
```

`BasicBlock` shape:

```json
{
  "statements":  [<Statement>...],
  "terminator":  { "span": ..., "kind": <TermKind>, "comments_before": [...] },
  "span":        { ... }
}
```

#### Locals

- Slot `0` = return value place; slots `1..arg_count` = arguments;
  the rest = compiler-introduced temporaries.
- Each local carries: `index`, optional `name` (preserved for user-named
  bindings, `null` for compiler-introduced temps), a precise source `span`,
  and a (deduplicated) `ty`.
- This means the lowering driver can recover **user-meaningful names** for
  bindings directly from the locals table.
- It also means **rustc-introduced temporaries are visible**, including the
  tuple-typed temp used by `AddChecked` (see §4 below).

#### Statements

Observed statement `kind` variants in the corpus:

| Kind                           | What it is                                              |
|--------------------------------|----------------------------------------------------------|
| `StorageLive(local_idx)`       | mark local slot alive                                   |
| `StorageDead(local_idx)`       | mark local slot dead                                    |
| `Assign(Place, Rvalue)`        | the main building block                                 |
| `Assert{...}`                  | inline assertion (e.g. overflow check)                  |

`Place` is `{ "kind": Local(i) | Projection(Place, ProjectionElem), "ty": ... }`
where `ProjectionElem` includes `Field(VariantId, FieldId)`, deref, indexing,
etc., all pre-resolved.

`Rvalue` observed forms include `Use(Operand)`, `BinaryOp(Op, Operand, Operand)`,
`UnaryOp`, `Aggregate`, `Ref`, `Cast`, `Discriminant(Place)`, `Len(Place)`,
etc. `Operand` is `Copy(Place) | Move(Place) | Constant(ConstantExpr)`.

#### Terminators

Observed terminator `kind` variants in the corpus:

| Kind                                       | Successors             |
|--------------------------------------------|-------------------------|
| `Return`                                   | (none)                  |
| `UnwindResume`                             | (none, ABI exit)        |
| `Abort`                                    | (none)                  |
| `Goto { target }`                          | `target`                |
| `Switch { discr, targets: If(then, else) }`| `then`, `else`          |
| `Switch { discr, targets: SwitchInt(ty, cases: [(scalar, bb)], default) }` | each case + default |
| `Call { call, target, on_unwind }`         | `target` (success), `on_unwind` (panic edge) |
| `Assert { assert, target, on_unwind }`     | `target`, `on_unwind`   |
| `Drop { ..., target, on_unwind }`          | `target`, `on_unwind`   |

Every fallible terminator (`Call`, `Assert`, `Drop`) carries an
**explicit `on_unwind` edge** to the unwind landing pad. This is where the
ULLBC carries panic/unwind information directly.

### 3. What Charon resolves ahead of the lowering driver

| Information                | Charon ULLBC                                |
|----------------------------|-----------------------------------------------|
| Concrete `Ty` per local    | already attached to every local + place      |
| Trait method resolution    | `Call.func.Regular.kind.Fun.Regular(<id>)` is a direct `fun_decls` index; `trait_refs` carries instantiation |
| `?` desugaring             | rewritten into `Try::branch` + `FromResidual::from_residual` calls + Switch on `ControlFlow` |
| `for` desugaring           | rewritten into `IntoIterator::into_iter` + loop over `Iterator::next` + Switch on `Option` |
| Generic instantiation      | full, with `generics: { regions, types, const_generics, trait_refs }` |
| Source spans               | from rustc, file_id + beg/end line/col       |
| Overflow checks            | `BinaryOp("AddChecked", ...)` + paired `Assert.check_kind.Overflow(...)` (see §4) |
| Drop / unwind edges        | `Drop` and unwind terminators are first-class |

### 4. Surprises and pitfalls

- **Overflow-checked arithmetic is in the ULLBC.** `a + b` in debug mode
  lowers to `Assign(t, BinaryOp("AddChecked", a, b))` followed by
  `Assert(!t.1, on_failure: Panic, check_kind: Overflow(Add Wrap, a, b))`
  and a `Assign(result, t.0)`. Our lowering driver has to either honor
  these (emit overflow checks) or strip them (treating `AddChecked` as
  `Add Wrap` and removing the paired `Assert`). For interpreter-style
  hot paths the latter is almost certainly what we want, but the
  decision must be explicit.
- **`Structured` vs `Unstructured`.** Charon's default (`charon cargo`)
  reconstructs the CFG into a nested-AST `Structured` form. That form
  has the advantage that loops and `if/else` are explicit syntactic
  constructs (closer to `syn`), but it is **not** what mirrors
  `flowcontext.py`'s BFS over basic blocks. For this project we should
  always use `--ullbc`. See `corpus.ullbc` for the artifact.
- **Cargo features matter.** Charon invokes `cargo build` under the
  hood, so feature flags affect what compiles. `pyre-interpreter`
  pulls in `majit-metainterp`, which has an explicit `compile_error!`
  unless `cranelift` or `dynasm` is enabled. The right invocation is
  `charon cargo --ullbc -- --features cranelift` (or `dynasm`).
- **`thread_local!` accessors translate as `Error` bodies.** Charon
  emits a per-statement error (`"charon does not support thread local
  references"`) but the surrounding function still extracts. In
  `pyre-interpreter` this produced 70 `Error`-bodied accessor
  closures out of 8,940 with-body decls; none of the interpreter's
  own functions failed. Downstream consumers must treat
  thread-local reads as opaque ops.
- **`has_errors: true` is set whenever any item failed**, even if it
  was a generated accessor. So `has_errors` alone isn't a hard gate —
  the per-item `body: { "Error": {...} }` walk is what matters.

### 5. Real-crate extraction smoke tests

These were **not** checked into the repo (the `.ullbc` files are too
large to commit), but the runs are reproducible from the commands
below.

#### `pyre-object`

```
charon cargo --ullbc --dest-file /tmp/pyre-object.ullbc
```

Result: **22.7 MB ULLBC. Compiled cleanly.**
- 3,733 `fun_decls` total; 2,422 with body slot; 28 `Error` bodies
  (all `thread_local!` accessors).
- 1,861 `pyre_object::*` functions with bodies extracted.
- 500 `core`/`std`/`alloc` bodies pulled in by reference.

#### `pyre-interpreter`

```
charon cargo --ullbc --dest-file /tmp/pyre-interpreter.ullbc \
  -- --features cranelift
```

Result: **133 MB ULLBC. Compiled cleanly with `cranelift` feature.**
- 11,728 `fun_decls` total; 8,940 with body slot; 70 `Error` bodies
  (all `thread_local!`).
- 6,448 `pyre_interpreter::*` functions extracted.
- Pulled-in deps with bodies: `core` 1,314, `pyre_object` 371,
  `std` 267, `alloc` 165, `rustpython_compiler_core` 93,
  `malachite_bigint` 90, `pymath` 54, `rustpython_sre_engine` 54,
  `libc` 33, others ≤ 20.
- BB-count distribution (local crate only):
  median 5, p90 36, p99 136, **max 2,225**
  (`pyre_interpreter::pyopcode::execute_opcode_step`).
- Terminator distribution (all extracted fns):
  Call 25,920, Drop 14,894, Abort 9,417, Goto 8,043, Return 7,169,
  Switch 6,218, UnwindResume 5,749, Assert 1,666.
- Only **7 `type_decls` carry `DynTrait`** — much lower than the raw
  grep count in the issue, because most `dyn Trait` raw lines appear
  inside comments or in setup/builder API surface that Charon either
  skips or that does not require lowering.

The high `Drop` and `UnwindResume` counts confirm that handling Rust's
drop / unwind semantics is a real concern for the lowering driver, not
something we can wave away.

### 6. Open questions / next steps

- **Charon version pinning.** The releases are all `prerelease: true`
  nightlies, so pinning a stable release means picking a specific
  nightly tag and committing to it. `nightly-2026.05.24` is a
  reasonable starting point.
- **`dyn Trait` classification.** The type-level count is low (7), but
  classifying the hot-path call sites (especially `DictStrategy` and
  friends) still requires walking them individually. That belongs in
  its own task.
- **`charon-lib` dependency from stable Rust.** A downstream consumer
  that uses `serde`/`charon-lib` needs `charon-lib` to build on the
  stable Rust toolchain (not the pinned Charon nightly). The crate is
  published on crates.io but its feature set / MSRV needs confirmation.
- **Caching strategy.** A 133 MB ULLBC per interpreter rebuild is too
  much to regenerate per developer build. The options are to check in
  extracted artifacts, cache in CI, or both.

### 7. What this corpus covers

| Coverage                                                                   | Where |
|----------------------------------------------------------------------------|--------|
| Representative corpus                                                      | `src/lib.rs` |
| Charon run over the corpus                                                | `corpus.ullbc` |
| `.llbc` for locals/places/constants/calls/trait resolution/terminators/discriminants/spans/unwind | §2–§4 |
| MIR → `FunctionGraph` lowering                                             | `majit-translate::front::mir` |
| Lowering regression against the corpus                                    | `majit-charon-reader::tests::corpus`, `majit-translate::tests::test_mir_frontend` |
| Checked-in notes / fixtures                                                | this README + `src/lib.rs` + `corpus.ullbc` + `inspect_llbc.py` |
