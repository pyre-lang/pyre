# AGENTS.md

## Primary porting boundary: PyPy's module surface

**PyPy's implementation shape is the default and module presence follows
PyPy.** Before implementing any module, first verify that the real `pypy3`
oracle imports it and identify its owner in `pypy/`, `rpython/`, or
`lib_pypy/`. If PyPy has no such module, it is not a pyre porting target unless
the user explicitly expands the scope. A skipped CPython test does not override
this boundary; fix PyPy's real public-module owner or fallback instead. See
“Module presence follows PyPy” below for the full rule and examples.

**Hard stop:** if neither the real `pypy3` import nor an upstream PyPy owner can
be verified, do not add the module or keep it in the implementation backlog.

**Known exclusions:** `_testlimitedcapi` is a CPython-only test helper, and
PyPy has no `_datetime` extension module. Do not implement either one; preserve
and repair PyPy's pure-Python `datetime` path when the public module is broken.

## The JIT is generated from the interpreter source

pyre is structured like PyPy: `pyre-interpreter` is the RPython-interpreter
analog, and **the JIT is not hand-written**. `majit-translate` reads the
interpreter's Rust source and generates it — `front/ast.rs` (parse) →
`flowspace/` (`flowcontext.py`/`framestate.py`) → `annotator/` (`annrpython.py`)
→ `rtyper/` → `codewriter/` (`jtransform.py`/`codewriter.py`, emits JitCode) —
the same pipeline RPython's translator runs over PyPy.

**So "Rust can't be meta-traced" is never a valid excuse for a deviation.**
Whatever the interpreter source expresses is what the generated JIT must
preserve; a JIT that diverges from the interpreter has a generation defect to
fix. Never justify a mismatch by appeal to the implementation language.

### One red frame per frame

PyPy keeps one frame object per inlined Python call (`MIFrame` tracing,
`BlackholeInterpreter` resuming), each with its own
`jitcode → pycode → w_globals → locals`. `LOAD_GLOBAL` reads it off the *live*
frame (`pyframe.py` `get_w_globals`); resume rebuilds one frame per encoded
jitcode header (`resume.py` `rebuild_from_resumedata`). No shared frame slot
exists, so namespace confusion is impossible.

The frame is the loop's single **red** input, `pycode` the **green**. Thread
that red frame through **every** frame, inlined non-portal callees included.
Collapsing callees onto one anchor (a single `portal_frame_reg`, a single
bridge-resume root frame) drops the callee's own pycode/globals/locals — the one
root cause behind the pycode-`names` miscompile, the LOAD_GLOBAL namespace
mismatch, bridge-resume inline-frame globals and vable-resident root locals. Fix
it by restoring the per-frame red frame, never by baking an anchor's value as a
constant.

### No new hand folds

`pyjitpl.py` has no hand-written specializations. Every fast path PyPy has
comes from tracing the interpreter's own body (`space.add`, `int_w`,
`w_float_new` as `new_with_vtable` + `setfield`) plus interpreter-side hints
(`@jit.elidable`, `@jit.look_inside_iff`, `promote`, `@jit.unroll_safe`) and
the optimizer (`intbounds`, `heap`, `virtualize`, `unroll`). pyre's
`try_walker_specialize_*` functions and `spec_folds!` rows are the manual
traces the descent work exists to retire, and their count is monotonic:

- **The census must not go up.** Rows:
  `sed -n '/^spec_folds! {/,/^}/p' pyre/pyre-jit-trace/src/jitcode_dispatch/diag.rs | rg -cF '=> ("'`.
  Functions: `rg -o 'fn try_walker_specialize_' pyre/ majit/ -g '*.rs' | wc -l`.
  A change that raises either number is rejected in review; record both
  numbers before and after in the commit message when a change touches them.
- **A descent wall is a lowering gap, not a reason to fold.** When descending
  a helper does not finish or aborts (an unlowered constructor such as
  `w_float_new`/`w_complex_new`, a guard with a null class pointer, a trace
  length or recursion limit), fix the codewriter, walker or optimizer so the
  body records, and cite the upstream shape that records it. Emitting the
  helper's result by hand, or residualizing the helper to make the walk stop
  early, re-creates the manual trace under a new name.
- **A renamed fold is still a fold.** A `try_emit_exact_*` or similar
  "exact operand" shortcut consulted before a descent is a hand fold gated
  under a descent row and counts toward the census. Do not hide one under a
  `_descent` row.
- **Speed is recovered upstream's way.** Where a descent is slower than the
  fold it replaced, the missing piece is an interpreter hint or an optimizer
  pass the fold was standing in for; port that. Reinstating the fold,
  raising a `pyre-check` ratio gate, or re-recording `.jitstats` without a
  root cause are not fixes.

## Charon LLBC extraction — the prepass input

The annotator/rtyper prepass and the `PYRE_RTYPER_VERBOSE` census read
**pre-extracted `.ullbc` under `target/llbc/`, not the Rust source**. A change to
`pyre-interpreter` / `pyre-object` / `pyre-module` / `pyre-jit` is invisible until re-extraction;
`majit-translate` changes take effect immediately, because the translator runs
live over the frozen bodies.

Charon is shared and pre-installed at `../.pyre-build/charon/<platform>/<version>/charon`
(pinned `nightly-2026.10.25`), **not on `PATH`** — `which charon` finding nothing
does not mean it is missing.

```bash
python3 scripts/install-charon.py    # idempotent, usually a no-op
python3 scripts/extract-llbc.py      # pyre-object pyre-interpreter pyre-module pyre-jit
touch pyre/pyre-jit-trace/build.rs
PYRE_RTYPER_VERBOSE=1 cargo build --release -p pyre-jit-trace   # runs the prepass
# census: target/release/build/pyre-jit-trace-*/stderr, rg -c 'PREPASS phaseA fail'
```

- `extract-llbc.py` **skips** a crate whose source fingerprint is unchanged;
  `--force` / `LLBC_FORCE_REEXTRACT=1` overrides. `pyre-interpreter.ullbc` is
  ~300 MB and takes minutes.
- Worktrees share a content-addressed LLBC store (`<shared>/llbc-store`, or
  `LLBC_STORE`) and import a matching artefact via clone/reflink instead of
  re-running Charon. `LLBC_STORE=off` disables it; `--prune-store` drops unused
  entries.
- `cargo clean` drops `target/llbc`; the next extract re-imports it from the
  LLBC store. The prepass cache lives in `<shared>/jit-trace-cache`.
- **A Linux container never extracts into the host's `target/llbc`.** Mount a
  per-container directory over `/workspace/pyre/target/llbc` at `container run`
  (see `/apple-container`, step 2).
- **On Windows, keep `CHARON_TARGET_DIR` short.** cargo passes one
  `-L dependency=...` per dependency build dir, so under a long target path
  `pyre-interpreter`'s rustc line passes the 32767-character limit; cargo then
  hands charon-driver an `@argfile`, charon-driver reads `--target` from its
  unexpanded argv, does not find it, and compiles the crate against the host
  sysroot, failing every `--extern` with `E0463: can't find crate`. The wall of
  missing crates is that, not a bad install or a stale pin. CI's Windows job
  sets `CHARON_TARGET_DIR=D:\ct`; a local path of that length works, while
  `.pyre-build/charon-target-<worktree>` under a home directory does not.
- A nonzero `extract-llbc.py` does **not** stop a shell gate on its own, and the
  `cargo test` after it then runs against stale LLBC. Abort the script on it.
- It internally sets `MAJIT_LLBC_EXTRACTION=1` to break the
  `pyre-jit → pyre-jit-trace → pyre-jit.ullbc` bootstrap cycle, which makes
  `build.rs` emit placeholder artifacts. Never set it for an ordinary build.
- Interpreter-source work ⇒ re-extract, then rebuild the prepass.
  Translator-only work ⇒ prepass rebuild alone.
- `rg` note: a stray `--replace` in user config mangles long lines; pass
  `rg --no-config` when bucketing.

## The wasm gate runs on every OS

Its only prerequisite beyond the native backends is the
`wasm32-unknown-unknown` target — wasmtime is linked into `pyre-wasm-runner`,
and the guest builds on stable with no `-Z build-std`. Once the target is
installed `DEFAULT_BACKENDS` **adds wasm by itself**, so a bare
`python3 pyre/check.py` runs three backends and `--backend dynasm,cranelift`
*narrows* it.

CI installs that target on the ubuntu leg alone because wasm output is
platform-independent — **a cost decision, not a capability limit.** Never defer
a wasm-only failure to CI on the grounds that the host cannot reproduce it.

## Data structure parity with RPython/PyPy

majit and pyre are line-by-line ports, so the container choice is part of the
port. **Treat every `HashMap` and every `thread_local!` as suspicious**: find the
upstream owner and storage shape first.

1. **Look up the upstream attribute before choosing a container.** A Rust
   `HashMap` is not the default translation of a Python `dict` — over a
   small/dense key space, or where insertion order or index lookup suffices,
   `VecMap` / `IndexMap` / `Vec` is closer. Prove the semantics from upstream.
2. **Side-tables are usually wrong.** RPython stores per-box information *on the
   box* (`box._forwarded`, `PtrInfo`, `IntBound`, descr attributes). Reaching for
   `HashMap<OpRef, _>` means you skipped that machinery; route through
   `OptContext::with_intbound_mut` / `set_ptr_info` instead.
3. **A borrow-checker workaround** is acceptable only when every alternative was
   tried, the deviation is minimal, and a comment cites the RPython original.
4. **Do not delete an RPython method to "simplify".** If `optimizer.py` has
   `ensure_ptr_info_arg0`, the port has it — the shortcut diverges, and the next
   porter's `heap.py` port stops compiling for no visible reason.
5. **TLS is almost never the right owner.** Type objects, module state,
   registries and semantic caches are process-global or interpreter-owned in
   PyPy and stay shared here; never duplicate them per thread to satisfy `Sync`.
   TLS is right only where upstream makes the state itself thread-specific
   (current thread/execution context, errno) or for a disposable cache that
   cannot affect identity, semantics, lifetime or GC reachability. Anything else
   needs an upstream citation in the code.

The measured cascade this prevents: deleting `ensure_ptr_info_arg0` for a
side-table `OptHeap.array_min_lengths` left `postprocess_arraylen_gc` unable to
read it, so it was crippled to a hardcoded `IntBound::nonnegative()`, which then
forced a parallel `ExportedValueInfo::int_lower_bound`. One non-orthodox
`HashMap`, four files of divergence.

## The PyPy oracle: run it before arguing about orthodoxy

`rpython/` says what upstream *claims*; a real `pypy3` shows what it *does*. For
"is this orthodox or our deviation?", run the oracle **first** — before reading
source, forming a theory, or recording a verdict.

```bash
PYPYLOG=jit-summary:- pypy3 pyre/bench/synth/<fixture>.py    # vs MAJIT_STATS=1
```

Most `pyre/bench/synth` fixtures need no stdlib, so this costs one command. Read
`Total # of loops`/`bridges`, `forcings`, `virtualizables forced`, every
`abort: *`, `nvirtuals`. **A differing counter is a pointer, not the answer** —
find the upstream line behind it, usually a single JIT hint
(`@jit.look_inside_iff`, `dont_look_inside`, `elidable`, `unroll_safe`), and cite
it. `PYPYLOG=jit-log-opt:FILE` dumps the optimized trace when the summary is too
coarse. `pypy3` is 3.11, so trim newer syntax out of the fixture.

This overturned a weeks-old "unfixable by construction" verdict on
`getframe_inline_subwalk_multiframe` in one command: the oracle reported
`forcings: 0`, and the `@jit.look_inside_iff` on `getframe`
(`pypy/module/sys/vm.py`) named the reason.

## Spec follows CPython 3.15t; engineering follows PyPy

**Pursue PyPy parity to the extreme as engineering, and CPython 3.15t as spec;
where they collide, take the 3.15t behaviour and engineer it the way PyPy
would.** Neither goal yields wholesale — the axis decides which one governs the
line in front of you.

The `t` is the **free-threaded** build: "CPython does X" is an answer only once X
holds without the GIL. Correct-because-a-global-lock-serialises-it is not
on-spec.

### Module presence follows PyPy

**Do not implement a module that PyPy does not have merely to unskip a CPython
test.** Whether a module exists, and which layer owns it, is part of the PyPy
implementation shape. Before adding any builtin, extension, compatibility, or
test-support module, verify that the real `pypy3` oracle imports it and locate
its owner in `pypy/`, `rpython/`, or `lib_pypy/`. If neither exists, the module
is not a pyre porting target unless the user explicitly expands the scope.
This applies to builtin-type and CPython-test-suite sweeps too: exclude an
absent module from the backlog instead of treating its skipped tests as missing
implementation work. The goal is to complete PyPy's module/type surface in the
PyPy way, not to reproduce CPython's private extension-module inventory.

In particular, CPython-only test helpers such as `_testlimitedcapi` are not
product modules. PyPy also has no `_datetime` extension module: it provides
`datetime` through its pure-Python implementation, so pyre must preserve the
working pure-Python path rather than invent an `_datetime` stub or accelerator.
A CPython test that directly imports one of these absent private modules may
skip or remain blocked; that is not by itself an implementation gap. If the
public stdlib feature is broken, port PyPy's actual owner/fallback and fix that
root cause. Never add an empty or partial module just to make the import
succeed, because that suppresses the intended fallback and turns a clean skip
into misleading failures.

**The spec governs only what a caller can observe** — return value, exception
type/message/attributes, identity, encoding-and-errors contract, accepted
argument shapes. Everything else follows PyPy **unconditionally**: names, module
paths, control-flow order, data structures, storage owner, JIT hints. A
structural divergence does not become a spec fix by sitting next to one.

**It is not a 3.11-vs-3.15 question** — six of seven adjudicated cases had no
version delta at all (`sched_setscheduler` since 3.3, `PyUnicode_FSConverter`
since 3.3, PEP 529, PEP 471). "3.15" pins *which* CPython you read
(`lib-python/stdlib-version.txt`); a missing delta is not grounds to refuse the
exception, and a real one earns nothing by itself.

**Six tests, in order; stop at the first leaf.** The full procedure is in
`/parity` under "SPEC-DEVIATION" — do not invoke this ruling without reading it.

1. Can a Python snippet print a difference? No → ordinary parity finding.
2. Do you hold an admissible 3.15 artefact — an in-tree `lib-python/3/…`
   assertion, a measured run at the pinned version, or C source read at that tag?
   Prose is not admissible; a comment in pyre's own source is never the artefact.
3. Do the two upstreams actually disagree? If they agree and pyre differs from
   both, it is a plain regression that no spec reasoning rescues.
4. Is PyPy's shape load-bearing for a mechanism pyre also has? Search the whole
   definition, decorators included, in `rpython/` and `pypy/` for `@jit.*`,
   `_immutable_*`, `_attrs_`, `unrolling_iterable`, `make_sure_not_resized`,
   `rgc.*`. A hint governing the value you are changing → **STOP, follow PyPy**.
   Record the negative search too.
5. Per-site artefact plus a blast-radius census: every departing site needs the
   artefact forcing *that* site ("consistency with a sibling" is not one), then
   `rg` pyre's own readers, `pyre-jit*` and `majit*` included.
6. Does pyre land on 3.15t across the whole decision? Landing where **neither**
   upstream sits is a defect however well one axis matched.

File the result under the review's `## 4. Structural adaptations` as
`[3.15-spec] <ours> ↔ <pypy> — <observable>; evidence: <route + cite>`, and
comment at the site citing both sides.

## Porting discipline

- Strict line-by-line structural parity. No shortcuts, no reimplementation from
  scratch, no declaring a phase complete without the literal refactor.
- If a parity fix regresses, find the root cause before reverting. Structural
  alignment skipped is not success, even with green benchmarks.
- Cite upstream by **symbol**, not `file:line`. Numbers rot silently and a
  rotted citation still reads as authoritative; a symbol stays checkable with
  `rg`. Use a line number only where no symbol pins the claim, and name the
  enclosing symbol beside it.
- Confirm the worktree (`git rev-parse --show-toplevel`) before editing and
  before staging — dozens of sibling worktrees share one `.git`.

## Build and CI polling

- Never check a running local build or CI job more often than once every five
  minutes. While it runs, continue useful independent work when any is
  available; otherwise wait for the five-minute boundary. Do not spend tokens
  on repetitive status-only updates.
- CPython reference results for `pyre/check.py` synthetic fixtures and
  `parity_tests/run.py` are cached at `<shared>/oracle-cache` (override with
  `PYRE_ORACLE_CACHE`; `PYRE_ORACLE_CACHE=off` disables). `--no-oracle-cache`
  refreshes entries without reading them.

## Iterate on the core; gate on the product

The product binary links `pyre-module`, the crate that owns the builtin
modules under `pyre/pyre-module/src/module/` (`math`, `_json`, `_ssl`,
`_cffi_backend`, `unicodedata`, `zlib`, …). Work on majit, the JIT, the GC or
the interpreter core does not need them, so the edit–build–test loop leaves
them out:

```bash
export CHARON_TARGET_DIR=$HOME/Projects/.pyre-build/charon-target-<worktree>
pyre/scripts/build-jit-core.sh           # extract the 4 core crates, pyre-dynasm without pyre-module (`jit-core` profile)
pyre/scripts/build-jit-core.sh --check   # same extraction, `cargo check` only
python3 pyre/check.py --backend cranelift   # check.py's cranelift leg is the core build
cargo test -p pyre-jit --no-default-features --features dynasm,prepass
```

- **Core-green is not product-green.** The core compiles nothing behind
  `cfg(feature = "pyre-module")`, including the JIT registrations that name
  those modules. A change that touches `pyre-module` or those registrations
  needs the product build before it is judged.
- **A fixture that imports a `pyre-module` module names it in its header**
  (`# pyre-check: requires-modules=math`). `check.py` reads each binary's
  `sys.builtin_module_names` and prints `skip (no math)` on a binary built
  without it, whatever its backend. Never exempt a backend
  (`skip-backends=`) for a module it lacks: whether the module is there is a
  property of the build, not of the backend.
- **`build-jit-core.sh` writes `target/jit-core/pyre-dynasm`** under the
  `jit-core` cargo profile (release optimisation, no LTO, 16 incremental
  codegen units), so its timings are not the product's. `python3
  pyre/check.py --build no --backend dynasm target/jit-core/pyre-dynasm`
  measures it, with the same `requires-modules` skips as the cranelift leg.
  JIT counters (`.jitstats`) are comparable; wall-clock ratios are not.
- The core loop does not replace the full gate below. When the full gate is
  due (see "How much to verify"), it is the product `cargo test` and a bare
  `python3 pyre/check.py`, not their core variants.
- **Which crate owns a module has three criteria, and PyPy's
  `default_modules` tier is not one of them.** A module belongs in
  `pyre-interpreter` when the interpreter reaches it by name -- `import`,
  `absolute_import`, `import_module`, or upstream's `space.getbuiltinmodule`,
  so `bootstrap_importlib_modules` -> `_imp`, `eval.rs build_template_op` ->
  `_template` and `importing.py read_compiled_module` -> `marshal` all count.
  The last shows why a pyre-side string search is not the whole test: pyre
  reaches `marshal` from `interp_imp.rs` as a direct Rust call, so only
  upstream names it -- read the upstream owner before concluding a module is
  unreached. A module belongs there too when PyPy's `essential_modules` names it
  (`_opcode`, `__pypy__`), or when CPython's `Modules/Setup.bootstrap` does
  (`_abc`, `_functools`, `_stat`, `_symtable`, `_types`, `faulthandler`, `pwd`,
  `time`). Everything else stays in `pyre-module`, and the test runs in both
  directions: a module in `pyre-interpreter` that answers none of the three
  belongs in `pyre-module`. Note that `Modules/Setup.bootstrap` is narrower than
  CPython's always-built-in table in `config.c.in` -- `_tokenize` is in the
  latter only and is not a criterion hit. A `default_modules` hit is **not** a
  reason to move: `pypyoption.py` gives every module a
  `BoolOption(modname, default=modname in default_modules)`, so that tier means
  "on by default, switchable off" -- which is what `pyrex`'s
  `default = [..., "pyre-module"]` already expresses. `math` and `cmath` are
  `default_modules` and stay optional; a fixture that needs one takes the skip
  marker above instead.

## How much to verify

The full gate (`cargo test --all` plus a bare `pyre/check.py`) costs over an
hour. It is not the default for every commit. Scale verification to what
changed since the last green signal, local or CI:

- **Rebase, cherry-pick or history rewrite of a tree that was green.** Do not
  rerun the tests. Confirm the rewrite kept the work: the commit list, and the
  patch-ids of the commits that did not conflict. Run `cargo check` on the
  crates whose conflicts you resolved by hand. A conflict resolved by blending
  logic from both sides also gets the tests that exercise that region. CI
  judges the rest.
- **New code.** Run the tests that exercise it: the touched crate's tests, the
  relevant `check.py --synthetic-pattern` fixtures, and the regression test for
  the bug. Widen to the full gate when the change is cross-cutting (translator
  front end, codewriter, GC rooting protocol, backend ABI, a shared runtime
  helper), and once before opening a PR.
- **The last CI run on this branch was red.** Before the next push, run every
  failed job's command locally and see it pass. Use `/apple-container` for an
  ubuntu-only red. A push that only hopes the red is fixed spends another CI
  run.
- **Before every push**, which takes seconds: `cargo fmt --all -- --check` and
  codespell. The `pre-commit` job gates every other CI job. Also make sure the
  tip contains `origin/main` (`git merge-base --is-ancestor origin/main HEAD`).
  A PR that is behind main gets no workflow run at all.

## The full gate

- `cargo test --all --no-default-features --features dynasm,pyre-module`. All
  three parts matter. `--features dynasm` selects a backend;
  `--no-default-features` is what keeps the selection consistent across crates.
  With default features on, the workspace's example interpreters default to
  `cranelift` and supply it to the one shared `majit-metainterp`, while
  `pyre-jit` compiles in its dynasm registrations — so the binary dispatches
  through one backend and holds the other's hooks. Nothing emits an error; the
  tests simply run against a mismatched pair. `pyre-module` names the crate that
  owns the optional builtin modules; it is a `pyrex` default, and because the
  dependency is optional, `--no-default-features` leaves the crate out of the
  launcher the integration tests run. CI's cargo test steps pass it, and the
  tests that need those modules are gated on the feature, so leaving it out
  here does not fail — it silently runs fewer tests than CI does.
- `python3 pyre/check.py` — every backend the host can build. A perf regression
  is a finding to explain, not an automatic veto: if the slower code is the
  line-by-line port and the faster was a shortcut, **the port stands** — record
  it and name the upstream optimization that would recover it (`/parity`
  Principle 4). Revert only when the regression has no such explanation. This
  file is loaded every session and Principle 4 is not, so do not restate that
  rule here in a form that contradicts it.
- Re-record a `.jitstats` baseline only when the new number is the one that
  should hold. "The recorded number no longer matches" is never on its own a
  reason: a gate is a target to reach, not a figure to refit.
- When rebasing or cherry-picking, check the fix isn't already on main
  (`git log main --grep=…`).

## Debugging discipline

- Before running the test, verify the traced path is actually reached — check
  gating and feature flags.
- Fix the interpreter/JIT root cause. Do not build workarounds (fallback
  modules, special cases) around it.
