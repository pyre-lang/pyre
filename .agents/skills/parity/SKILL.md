---
name: parity
description: Audit and fix majit/pyre changes for strict line-by-line RPython/PyPy structural parity. Use for $parity, legacy /parity requests, upstream-source comparisons, and parity-directed porting or debugging in this repository.
---

# RPython/PyPy Line-by-line Parity

## Codex operation and repository rules

Use `$parity <task>`; recognize `/parity <task>` as legacy wording with the
same intent. Read the current worktree's `AGENTS.md` first. Confirm
`git rev-parse --show-toplevel` before editing and again before staging.
Explicit user scope, including an earlier audit-only instruction, takes
precedence over this skill's default fixes. Importing or editing this skill
is not an instruction to run an audit.

All repository paths below are relative to that verified root. Use
`rg --no-config` for searches. Cite upstream by file and **symbol**; add
current line numbers for review navigation, never as a substitute for the
symbol. Historical examples in references must be checked against today's
source and oracle before reuse.

For an orthodoxy diagnosis, run the real `pypy3` fixture **before** reading
source or forming a theory; then locate the upstream decision and JIT hints.
Before adding any module, verify both a real `pypy3` import and its owner in
`pypy/`, `rpython/`, or `lib_pypy/`. An absent module is not a porting target
unless the user explicitly expands the scope. In particular, do not add
`_testlimitedcapi` or `_datetime`; repair PyPy's public fallback owner.

The JIT is generated from interpreter source. Preserve one red frame per
Python frame, including inlined callees. Trace/interpreter divergence is a
generation defect; follow `AGENTS.md` for frame ownership, upstream container
storage, and TLS discipline. Do not excuse a mismatch by the Rust language.

Use a local Markdown report for persistent findings and follow-up work, such
as `scratchpad/parity-report.md` (create its parent directory when needed).
Record each deferred item with a stable local ID, citation, blocker, and
convergence path. Do not assume Claude task or memory tools exist. If a user
names an unavailable task ID, look for its brief in local artifacts; request
its contents only if the task cannot otherwise be identified.

## First principles

Apply these to every decision made while this skill is active, within the user's authorized task and constraints.

1. **Structural equivalence, not functional equivalence.** Match module paths, module names, class/type names, function names, variable names, even down to `_` prefixes and `self`-passing conventions. Functional-only equivalence silently accumulates divergence and causes bugs that never occur in RPython.

2. **PyPy is the engineering reference; CPython 3.15t governs observable behaviour.** The local PyPy tree at `rpython/` and `pypy/` is the authoritative reference. Before writing any non-trivial change, open the RPython counterpart and read it. Line-by-line porting usually has the answer; trust upstream over clever adaptations.

3. **Every deviation must be classified and justified.** Four classes:
   - **NEW-DEVIATION** — introduced by us, in this diff or recent history, with no RPython backing. These must be removed. Replace with the correct RPython code, not with another adaptation.
   - **PRE-EXISTING-ADAPTATION** — a deviation documented with an RPython reference and an explanation of why the mechanical line-by-line port is blocked. **PRE-EXISTING-ADAPTATION is a temporary marker, not a permanent justification.** It is the label on a piece of technical debt: the code admits it diverges from RPython and commits to converging back. Its presence is not a reason to leave the code alone — it is a reason to fix it now if the blocker is no longer real, or to add a concrete "convergence path" comment if the blocker still holds. Every PRE-EXISTING-ADAPTATION encountered during a `$parity` pass must be re-evaluated: (a) is the blocker still real? (b) can it be ported right now, in this session? (c) if not, what specific dependency (helper, pass, data layout) would need to land first? Never use "it's marked PRE-EXISTING-ADAPTATION" as a reason to skip the fix. If the mechanical port is obvious and doesn't exceed session scope, do it now.
   - **PARITY** — matches RPython structure line-by-line. No action.
   - **SPEC-DEVIATION** — a deviation from PyPy that makes pyre match the observable behaviour of the CPython pinned in `lib-python/stdlib-version.txt`, where PyPy and that CPython genuinely differ, adjudicated under AGENTS.md "Spec follows CPython 3.15t; engineering follows PyPy". This is **not** a NEW-DEVIATION and is **not** auto-fixed under Principle 6 — reverting one to PyPy's shape re-introduces a known bug. Recognise it by the site comment citing both sides and the `§4 [3.15-spec]` entry in the review report. Audit it instead: does the comment carry an artefact (a `lib-python/3` `file:line`, a measured run at the pinned version, or C source read at that tag — never a pyre comment, never a PEP)? Is it free of a PyPy-side hint governing the changed value (`@jit.*`, `_immutable_*`, `_attrs_`, `make_sure_not_resized`, read across the whole definition including decorators, in `rpython/` too)? Does pyre now match PyPy on every adjacent observable that reads the state it changed? If any check fails, the exemption is unproven: run the full six-test procedure and classify the finding from the evidence before changing code. Structure — names, control flow, data structures, storage owner — is never covered by this class and is audited as usual. The full six-test procedure is in "SPEC-DEVIATION: the six tests" in [references/spec-deviation.md](references/spec-deviation.md); read it in full before applying or rejecting this exemption.

4. **Performance can temporarily regress.** If a benchmark slows down because parity-correct code replaced a clever local shortcut, accept it. Performance is recovered by further line-by-line porting of the upstream optimization, not by reintroducing the shortcut. Record the regression and name the upstream optimization that would recover it. Explicit user instructions still govern task scope.

5. **Don't stop at the first dependency.** If line-by-line porting is impossible because a dependency (helper, pass, opcode, descriptor) has not been ported yet, port that dependency first in the same RPython-parity style. Then come back.

6. **Parity regressions are fixed by default, and take priority over the follow-up task.** A **parity regression** is a NEW-DEVIATION introduced by the diff under audit — i.e., absent at the audit base, present at HEAD. Regressions found in audit are **fixed in-session by default** — no per-item user confirmation is required — and the follow-up task proceeds afterward. The user opts out by directing so explicitly in the session (e.g. `$parity only audit`, `$parity don't change <file>`, `$parity skip the regression and just do <X>`). "The user might not want this fixed" is not a reason to skip — they would have said so.

Pre-existing NEW-DEVIATIONs (already at the audit base, simply visible because they sit in a modified file) are still violations per Principle 3 and the rule on PRE-EXISTING-ADAPTATION (Principle 7), but they are **not** auto-fixed by this default. Flag them in the audit so the user can decide whether to widen scope; do not silently expand the diff to clean unrelated old debt. Auto-fix discipline is scoped to what this diff broke, not to opportunistic cleanup of the surrounding code.

7. **PRE-EXISTING-ADAPTATION is a fix queue, not an absolution.** When a PRE-EXISTING-ADAPTATION appears in the code you are touching — or in code adjacent to your task — treat it like a NEW-DEVIATION with a longer history. The default action is to port it back to canonical RPython shape in the same session. Only defer when there is a specific, cited, still-real blocker (a dependency not yet ported, a layout change that cascades across more files than the session can safely touch, a benchmark regression whose root cause is another unported optimization). "It works today" is never a sufficient reason to leave a PRE-EXISTING-ADAPTATION in place. If this feels aggressive, remember that adaptations accumulate silently: every PRE-EXISTING-ADAPTATION left untouched adds a permanent divergence surface that future ports must work around. Paying the conversion cost now, in the smallest feasible slice, is how the codebase actually converges on RPython.

## Behavior under `$parity`

When this skill is invoked — either by the user typing `$parity ...` or by Codex recognizing a parity-related ask — do the following in order:

### Step 1: Parse invocation

Separate the invocation into:
- The `$parity` flag itself (already consumed).
- The **follow-up task** — everything the user wrote after `$parity`. This is the actual work to be done.

If there is no follow-up task (just bare `$parity`), default to "run a full parity audit of the current diff, **auto-fix every parity regression** (NEW-DEVIATION introduced by the diff relative to the audit base), and report what was changed". This applies whenever the user has not explicitly opted out in the session (`$parity only audit`, `$parity report and stop`, `$parity skip <file>`). No per-finding confirmation is required — the invocation itself is the authorization. Pre-existing NEW-DEVIATIONs (visible because they live in a touched file but absent from the diff itself) are listed in the audit for awareness only — see Principle 6.

Defer a regression only when a concrete blocker prevents completing it within the authorized scope. The following can require broader work, but are not automatic deferrals or permission gates:

- changes that touch cross-crate API shape;
- changes that regenerate lockfiles or move files between modules;
- fixes that require porting a missing upstream dependency whose own port would cascade beyond a single session (port the dependency first per Principle 5; if that's not possible in scope, surface).

Self-contained fixes below that bar — renaming a local to match upstream, reordering arms to mirror upstream control flow, replacing a side-table with the upstream field, extracting a helper that already exists upstream, deleting an undocumented cache — are auto-fixed by default whether or not they are "trivial" in the strict semantic-preserving sense. Parity-correct code that changes observable behavior in the direction of upstream is still an auto-fix; the regression is the *deviation*, not the change that removes it.

When the follow-up task is present, the same default applies to regressions found in audit: they are fixed in-session (in priority order per Principle 6), then the follow-up task runs.

### Step 2: Parity audit

Before touching code, verify `upstream/main` exists. Do not auto-fetch or substitute local `main`/`origin/main`; if it is absent, report that the requested audit base is unavailable and what remote information is missing. Then inspect the current state:

```bash
git diff upstream/main...HEAD         # committed changes on current branch (base = upstream/main, the remote base, NOT local main/origin/main)
git diff HEAD                         # working tree (unstaged + staged)
git status --short                    # untracked files worth noting
```

Use the merge base for committed-change provenance and inspect staged/unstaged changes as well. A changed file alone does not prove a finding was introduced by the patch: compare the relevant base body. Include task-related untracked source explicitly; do not sweep unrelated user scratch files into the audit. When called from `$codex-review`, use its `upstream/main` comparison and authoritative changed-file list consistently.

For each changed source file, find the RPython/PyPy counterpart **mechanically** (see next section). Read the upstream file and compare.

Report findings as a short **Parity audit** section before executing the follow-up task:

- Files modified with their mechanically-derived RPython counterpart (one line each).
- For each hunk that contains a candidate NEW-DEVIATION: cite the file:line in the diff, cite the RPython file and symbol it should have mirrored (with a current line number), quote the deviation concisely.
- Files modified where the mechanical rule does not produce a valid counterpart — treat these as structural deviations in themselves and flag to the user.

If the audit turns up **regressions** (NEW-DEVIATIONs introduced by the diff vs the audit base), list them in the report and fix them in Step 3 (regressions first, then the follow-up task, per Principle 6). Surface-and-defer only regressions with a concrete blocker under Step 1; the rest are fixed without asking. Pre-existing NEW-DEVIATIONs seen in modified files (already present at the audit base) are listed for awareness but **not** auto-fixed — that's a separate scope decision for the user.

### Step 3: Execute the follow-up task under parity

Run the follow-up task with these constraints:

- Read the relevant RPython source before writing code. Cite the upstream file and symbol being ported in the code comment or in the summary to the user.
- When removing pre-existing code, confirm with the RPython upstream that the removal agrees with the upstream. If RPython has the code in a different form, **replace** rather than **delete**.
- When introducing anything new, check whether RPython has a counterpart. If so, port it. If not, either (a) port the RPython dependency that provides it, or (b) document the unavoidable structural adaptation with the upstream decision point, concrete blocker, and convergence path. Do not label a new deviation pre-existing to exempt it, and do not use this route to add a module absent from PyPy.
- If the follow-up task's natural implementation would introduce a NEW-DEVIATION, push back: state the tradeoff, propose the RPython-aligned alternative, and let the user choose. Do not silently introduce new deviations.

### Step 4: Verify

Before testing, confirm the traced path is reached under the selected features
and gates. Interpreter/object/JIT source changes require LLBC re-extraction
before rebuilding the prepass; translator-only changes require the prepass
rebuild alone. Follow the commands and bootstrap restrictions in `AGENTS.md`.

After runtime changes, and before committing them, run:

- `cargo test --all --no-default-features --features dynasm`.
- `cargo test --all --no-default-features --features cranelift` if those paths changed.
- `python3 pyre/check.py` for every backend the host can build, including wasm
  when `wasm32-unknown-unknown` is installed. Do not narrow the gate silently.

For documentation/skill-only changes, validate the edited artifacts instead
of rebuilding the runtime. Report failures or checks that could not run;
do not claim the affected findings are closed without the relevant checks.
Accept explained parity-correct performance regressions, record them in the
local report, and name the missing upstream optimization. Do not reintroduce
shortcuts or re-record `.jitstats` solely to fit the latest measurement.

## Finding the counterpart: mechanical rule first

The goal is that **any** file in majit/pyre has its RPython/PyPy counterpart discoverable from the path alone. When this works, no mapping table is needed. When it doesn't work, the failure is itself a parity signal — the path structure diverged from upstream and that divergence should be on the audit list.

### Crate-level roots

Match the most specific prefix first; `translator/rtyper/` takes precedence over `translator/`. Everything below is mechanical.

| majit/pyre crate root | Upstream root |
|---|---|
| `majit/majit-metainterp/src/` | `rpython/jit/metainterp/` |
| `majit/majit-translate/src/codewriter/` | `rpython/jit/codewriter/` |
| `majit/majit-translate/src/flowspace/` | `rpython/flowspace/` |
| `majit/majit-translate/src/annotator/` | `rpython/annotator/` |
| `majit/majit-translate/src/translator/rtyper/` | `rpython/rtyper/` |
| `majit/majit-translate/src/translator/` | `rpython/translator/` |
| `majit/majit-backend-dynasm/src/x86/` | `rpython/jit/backend/x86/` |
| `majit/majit-backend-dynasm/src/aarch64/` | `rpython/jit/backend/aarch64/` |
| `majit/majit-backend-cranelift/src/` | `rpython/jit/backend/llsupport/` (Cranelift plays the role of LLSupport) |
| `pyre/pyre-interpreter/src/` | `pypy/interpreter/` + `pypy/objspace/std/` + `pypy/module/` |
| `pyre/pyre-object/src/` | `pypy/objspace/std/` (object layouts) |

The `majit-` / `pyre-` prefix is a Cargo workspace namespace, not a claim
that the crate lives under `rpython/jit/` or `pypy/<anything>/`. Each row
is an independent mapping; add new rows rather than deriving from the
prefix. `rpython/` ↔ `majit/`, `pypy/` ↔ `pyre/` at the package-root
level — crates under `majit/` can correspond to any `rpython/<package>/`,
not only `rpython/jit/`.

**Crate boundary invariant**: `majit/*` crates MUST NOT depend on any
`pyre/*` crate, mirroring upstream's `rpython/` ⊥ `pypy/` separation.
External third-party crates (e.g. `rustpython-compiler-core` for CPython
3.15 bytecode tables) are allowed as they play the role of RPython's
host-stdlib imports (e.g. `from opcode import ...`).

The following crates carry architectural divergences from upstream and their roots are PRE-EXISTING-ADAPTATIONs by design. Audit individual files against RPython as if the root were `rpython/jit/metainterp/`, and classify mismatches per the rules below.

- `majit/majit-ir/` — extracted-out IR / OpCode / Descr layer. In RPython these live inside `rpython/jit/metainterp/resoperation.py` + `history.py` + scattered descr files in `rpython/jit/backend/`. The crate split itself is a Rust adaptation; the file-level names inside still need to line up.
- `pyre/pyre-jit-trace/` — pyre-specific layer for tracing Python bytecode. RPython's register-machine jitcode path lives in `rpython/jit/metainterp/pyjitpl.py` (opimpl_*), `blackhole.py`, and `codewriter/`. Auditors: the *logic* here must match `pyjitpl.py` opimpls file-by-file even though the directory is different.
- `pyre/pyre-jit/` — pyre's warm-entry / portal-runner / resume glue. RPython's counterparts are `rpython/jit/metainterp/warmstate.py`, `warmspot.py`, `compile.py` (loop/bridge creation).

### Within-root mechanical transform

For any file under a known crate root, compute the counterpart by:

1. Strip the crate root prefix.
2. Replace the trailing `.rs` with `.py`. For `mod.rs`, treat it as `__init__.py` (usually empty; the real content is in siblings).
3. Prepend the upstream root.

Examples (all should resolve to an **existing** file on disk):

| pyre/majit path | Expected upstream path |
|---|---|
| `majit/majit-metainterp/src/pyjitpl.rs` | `rpython/jit/metainterp/pyjitpl.py` |
| `majit/majit-metainterp/src/optimizeopt/unroll.rs` | `rpython/jit/metainterp/optimizeopt/unroll.py` |
| `majit/majit-metainterp/src/optimizeopt/heap.rs` | `rpython/jit/metainterp/optimizeopt/heap.py` |
| `majit/majit-metainterp/src/blackhole.rs` | `rpython/jit/metainterp/blackhole.py` |
| `majit/majit-translate/src/codewriter/jtransform.rs` | `rpython/jit/codewriter/jtransform.py` |
| `majit/majit-backend-dynasm/src/x86/regalloc.rs` | `rpython/jit/backend/x86/regalloc.py` |
| `pyre/pyre-interpreter/src/baseobjspace.rs` | `pypy/interpreter/baseobjspace.py` |

Verify the counterpart exists before proceeding:

```bash
test -f rpython/jit/metainterp/optimizeopt/unroll.py && echo OK
```

### When the mechanical transform fails

If the derived path does not exist, **that is itself a parity finding**. Do not silently fall back to a multi-file "whichever upstream file has similar content" search without naming the problem. Classify the mismatch:

1. **File renamed in majit/pyre vs upstream.** e.g. `majit-metainterp/src/heap.rs` vs `rpython/jit/metainterp/heapcache.py`. The rename is a structural deviation; rename back unless there is a cited reason. If the majit file genuinely combines multiple upstream files, see #2.

2. **One majit/pyre file fuses multiple upstream files.** e.g. `pyre-interpreter/src/baseobjspace.rs` containing both `baseobjspace.py` and `abstractinst.py` content. Fused files are structural deviations: upstream's module boundaries carry semantic meaning (separate import graph, separate test units). Split as two files unless merging was explicit user intent. Even without splitting, annotate each function with the upstream file and symbol it was copied from.

3. **majit/pyre file at a different directory than upstream.** e.g. `majit-metainterp/src/jitcode/` whose upstream lives in `rpython/jit/codewriter/`. The crate boundary is wrong. Flag as structural deviation. When porting, keep the function-level parity to `rpython/jit/codewriter/` even if the directory cannot be moved in this change.

4. **Upstream has no counterpart.** Could be (a) a Rust-specific adaptation that deserves PRE-EXISTING-ADAPTATION status (document which upstream decision it encodes), or (b) a NEW-DEVIATION that should never have been created. Determine which. A file named with pyre-specific domain vocabulary (e.g. `pyre_sym.rs`, `jit_state.rs`, `constant_pool.rs`) that has no upstream parallel is high-risk — those are often where NEW-DEVIATIONs live.

5. **Auto-generated files.** `target/release/build/*/out/*.rs`, proc-macro expansions, `.template.rs` files. Skip these in the audit; follow the template source instead.

Report every mechanical-transform failure in the audit, even if you then successfully locate the logical counterpart manually.

### Within a file: mechanical name match

Within a matched file, the structural expectation continues:

- **Type names** should match upstream class names 1:1 (Rust CamelCase ↔ Python CamelCase, trivial).
- **Function / method names** should match upstream, modulo Rust snake_case (`optimize_INT_ADD` → `optimize_int_add`).
- **Field / variable names** should match upstream identifiers literally (`box`, `opnum`, `orgpc`, `resumepc`, `postponed_op`, `truthy_values`, …).
- **Control flow order** should match upstream — don't reorder if/elif chains or loop bodies for "clarity".

Any local name that doesn't appear anywhere in upstream (`rg --no-config` in `rpython/` and `pypy/` for the string) is a candidate NEW-DEVIATION. Examples seen in this repo:

- `pending_branch_other_target` — no upstream equivalent
- `last_comparison_*` cache — no upstream equivalent
- `pre_opcode_*` stack snapshot — no upstream equivalent (RPython uses per-PC liveness, not per-opcode state capture)
- `other_target` resume adaptation — no upstream equivalent

## Signals of NEW-DEVIATION

Treat these patterns as high-likelihood NEW-DEVIATION. Verify against upstream before removing, but flag in every audit.

- **Undocumented side tables / HashMap caches** in structs that RPython does not have. RPython's Box-identity model (Python object `is`) removes the need for most side tables; if pyre has one, it's often a flat-OpRef compensation that should either be lifted or removed.
- **`take()` / `Option::take` / `std::mem::replace` used to temporarily swap state for a snapshot**, where RPython would write and read a single `frame.pc` field. If the save/restore is over fields with no RPython counterpart (`pre_opcode_*`, `pending_branch_*`, `last_comparison_*`), the fields themselves may be the deviation.
- **Function names that don't appear in RPython**: `rg --no-config` in `rpython/jit/` and `pypy/` for the name. If nothing matches, the function is a pyre-only helper. Check whether an RPython counterpart exists under a different name before declaring it NEW.
- **Comments like** `// pyre-only`, `// TODO remove`, `// workaround`, `// adaptation`, `// fallback`, `// temporary`, `// hack` without a cited RPython file and symbol. Honest self-reports of deviation.
- **`is_*_classlike` / `is_*_like` / `_maybe_*` helpers that reinterpret a raw pointer** as a different struct and peek at offsets. Type-confusion bombs — RPython has proper `isinstance` checks.
- **Feature flags / environment-variable switches** that gate behavior differences from RPython. RPython doesn't use env vars to switch semantics; if pyre does, both code paths are deviations in different directions.
- **"Simplified" resume / guard paths** (other_target resume, post-pop snapshot, …). RPython's resume is one path, not several.

None of these are automatically NEW-DEVIATION — check upstream first. But they warrant scrutiny.

## Output shape

When the skill is active and a response is being written:

1. **Parity audit** (short section, file:line references):
   - Keep the user-facing audit concise; put a large complete census in the local report.
   - List each modified file with its mechanically-derived counterpart (or a "❌ mechanical transform failed: <reason>" note).
   - Cite each candidate NEW-DEVIATION as `<majit path>:<line> ↔ <rpython path>:<line> — <deviation summary>`.
   - Mark each finding with its disposition: `[auto-fix]` for regressions being fixed in this response (the default for regressions), `[deferred: <reason>]` for regressions that exceed session scope per Step 1's blocker criteria, `[pre-existing]` for NEW-DEVIATIONs already present at the audit base (flagged for user awareness only, not auto-fixed).
   - If there is nothing to report, still include the section with `Clean — no new deviations in current diff`.

2. **Follow-up task** (the bulk of the response):
   - Executed under the principles above.
   - Reading RPython source is expected. Cite it.
   - New deviations proposed by the natural solution must be flagged to the user with an RPython-aligned alternative before committing to code.

3. **Verification** (if code was changed):
   - Which test commands were run and their outcomes.
   - Any temporary performance regression noted.

4. **Summary** (one or two sentences):
   - What changed, what was left for follow-up.

Keep the tone direct and the citations concrete. No vague "I'll make sure this matches PyPy" language — always point at an upstream file and symbol, with current line numbers when useful.

## Interaction with other mindsets

- This skill is the strict enforced form of the "First principles" above: under `$parity`, there is no wiggle room for "Rust language adaptations" except where the RPython line is explicitly cited.
- If committing is requested, complete parity auditing and the repository checks before the commit; no separate commit skill is required.
- If the user invokes `$parity` inside a larger plan document (e.g. `jtransform_optimize_goto_if_not_port.md`), use the plan to identify the authorized scope and audit base. Re-evaluate adjacent PRE-EXISTING-ADAPTATION blockers under Principle 7; an old annotation is not an exemption.


## SPEC-DEVIATION: the six tests

Read [references/spec-deviation.md](references/spec-deviation.md) in full before
adjudicating any CPython-observable exception. It preserves the six tests and
worked examples from the original parity skill. Apply the current `AGENTS.md`
module boundary and CPython **3.15t** requirement throughout.
