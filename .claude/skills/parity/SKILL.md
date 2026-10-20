---
name: parity
description: Enforce strict line-by-line RPython/PyPy structural parity for changes in the majit/pyre codebase. Invoked via `/parity`, usually combined with a follow-up task (e.g. `/parity continue #task 15`, `/parity fix the optimizer`, `/parity why is nested_loop slow`). Use this skill whenever the user types `/parity`, or when they ask for RPython-parity verification, line-by-line checking against upstream, structural-equivalence review, or pypy-source comparison. The skill puts Claude into "parity mode" — the rest of the user's message is executed under the strict parity principles below, and Claude must read the local PyPy/RPython source at `{rpython,pypy}/` before making non-trivial changes.
---

# RPython/PyPy Line-by-line Parity

## First principles

Apply these to every decision made while this skill is active, even if the user's follow-up task seems to point elsewhere.

1. **Structural equivalence, not functional equivalence.** Match module paths, module names, class/type names, function names, variable names, even down to `_` prefixes and `self`-passing conventions. Functional-only equivalence silently accumulates divergence and causes bugs that never occur in RPython.

2. **Upstream source is the spec.** The local PyPy tree at `rpython/` and `pypy/` is the authoritative reference. Before writing any non-trivial change, open the RPython counterpart and read it. Line-by-line porting usually has the answer; trust upstream over clever adaptations.

3. **Every deviation must be classified and justified.** Four classes:
   - **NEW-DEVIATION** — introduced by us, in this diff or recent history, with no RPython backing. These must be removed. Replace with the correct RPython code, not with another adaptation.
   - **PRE-EXISTING-ADAPTATION** — a deviation documented with an RPython reference and an explanation of why the mechanical line-by-line port is blocked. **PRE-EXISTING-ADAPTATION is a temporary marker, not a permanent justification.** It is the label on a piece of technical debt: the code admits it diverges from RPython and commits to converging back. Its presence is not a reason to leave the code alone — it is a reason to fix it now if the blocker is no longer real, or to add a concrete "convergence path" comment if the blocker still holds. Every PRE-EXISTING-ADAPTATION encountered during a `/parity` pass must be re-evaluated: (a) is the blocker still real? (b) can it be ported right now, in this session? (c) if not, what specific dependency (helper, pass, data layout) would need to land first? Never use "it's marked PRE-EXISTING-ADAPTATION" as a reason to skip the fix. If the mechanical port is obvious and doesn't exceed session scope, do it now.
   - **PARITY** — matches RPython structure line-by-line. No action.
   - **SPEC-DEVIATION** — a deviation from PyPy that makes pyre match the observable behaviour of the CPython pinned in `lib-python/stdlib-version.txt`, where PyPy and that CPython genuinely differ, adjudicated under AGENTS.md "Spec follows CPython 3.15; implementation follows PyPy". This is **not** a NEW-DEVIATION and is **not** auto-fixed under Principle 6 — reverting one to PyPy's shape re-introduces a known bug. Recognise it by the site comment citing both sides and the `§4 [3.15-spec]` entry in the review report. Audit it instead: does the comment carry an artefact (a `lib-python/3` `file:line`, a measured run at the pinned version, or C source read at that tag — never a pyre comment, never a PEP)? Does any PyPy-side hint govern the value it changes (`@jit.*`, `_immutable_*`, `_attrs_`, `make_sure_not_resized`, read across the whole definition including decorators, in `rpython/` too)? Does pyre now match PyPy on every adjacent observable that reads the state it changed? Any "no" makes it an ordinary NEW-DEVIATION again — report it, and restore PyPy's shape. Structure — names, control flow, data structures, storage owner — is never covered by this class and is audited as usual. The full six-test procedure is in "SPEC-DEVIATION: the six tests" at the end of this file.

4. **Performance can temporarily regress.** If a benchmark slows down because parity-correct code replaced a clever local shortcut, accept it. Performance is recovered by further line-by-line porting of the upstream optimization, not by reintroducing the shortcut. Neither the user nor the benchmark suite overrides the parity principle.

5. **Don't stop at the first dependency.** If line-by-line porting is impossible because a dependency (helper, pass, opcode, descriptor) has not been ported yet, port that dependency first in the same RPython-parity style. Then come back.

6. **Parity regressions are fixed by default, and take priority over the follow-up task.** A **parity regression** is a NEW-DEVIATION introduced by the diff under audit — i.e., absent at the audit base, present at HEAD. Regressions found in audit are **fixed in-session by default** — no per-item user confirmation is required — and the follow-up task proceeds afterward. The user opts out only by directing so explicitly in the same invocation (e.g. `/parity only audit`, `/parity don't change <file>`, `/parity skip the regression and just do <X>`). "The user might not want this fixed" is not a reason to skip — they would have said so.

Pre-existing NEW-DEVIATIONs (already at the audit base, simply visible because they sit in a modified file) are still violations per Principle 3 and the rule on PRE-EXISTING-ADAPTATION (Principle 7), but they are **not** auto-fixed by this default. Flag them in the audit so the user can decide whether to widen scope; do not silently expand the diff to clean unrelated old debt. Auto-fix discipline is scoped to what this diff broke, not to opportunistic cleanup of the surrounding code.

7. **PRE-EXISTING-ADAPTATION is a fix queue, not an absolution.** When a PRE-EXISTING-ADAPTATION appears in the code you are touching — or in code adjacent to your task — treat it like a NEW-DEVIATION with a longer history. The default action is to port it back to canonical RPython shape in the same session. Only defer when there is a specific, cited, still-real blocker (a dependency not yet ported, a layout change that cascades across more files than the session can safely touch, a benchmark regression whose root cause is another unported optimization). "It works today" is never a sufficient reason to leave a PRE-EXISTING-ADAPTATION in place. If this feels aggressive, remember that adaptations accumulate silently: every PRE-EXISTING-ADAPTATION left untouched adds a permanent divergence surface that future ports must work around. Paying the conversion cost now, in the smallest feasible slice, is how the codebase actually converges on RPython.

## Behavior under `/parity`

When this skill is invoked — either by the user typing `/parity ...` or by Claude recognizing a parity-related ask — do the following in order:

### Step 1: Parse invocation

Separate the invocation into:
- The `/parity` flag itself (already consumed).
- The **follow-up task** — everything the user wrote after `/parity`. This is the actual work to be done.

If there is no follow-up task (just bare `/parity`), default to "run a full parity audit of the current diff, **auto-fix every parity regression** (NEW-DEVIATION introduced by the diff relative to the audit base), and report what was changed". This applies whenever the user has not explicitly opted out in the invocation (`/parity only audit`, `/parity report and stop`, `/parity skip <file>`). No per-finding confirmation is required — the invocation itself is the authorization. Pre-existing NEW-DEVIATIONs (visible because they live in a touched file but absent from the diff itself) are listed in the audit for awareness only — see Principle 6.

The only NEW-DEVIATIONs that get surfaced-and-deferred instead of fixed are those that exceed safe session scope:

- changes that touch cross-crate API shape;
- changes that regenerate lockfiles or move files between modules;
- fixes that require porting a missing upstream dependency whose own port would cascade beyond a single session (port the dependency first per Principle 5; if that's not possible in scope, surface).

Self-contained fixes below that bar — renaming a local to match upstream, reordering arms to mirror upstream control flow, replacing a side-table with the upstream field, extracting a helper that already exists upstream, deleting an undocumented cache — are auto-fixed by default whether or not they are "trivial" in the strict semantic-preserving sense. Parity-correct code that changes observable behavior in the direction of upstream is still an auto-fix; the regression is the *deviation*, not the change that removes it.

When the follow-up task is present, the same default applies to regressions found in audit: they are fixed in-session (in priority order per Principle 6), then the follow-up task runs.

### Step 2: Parity audit

Before touching any code, run a quick audit of the current state:

```bash
git diff upstream/main...HEAD         # committed changes on current branch (base = upstream/main, the remote base, NOT local main/origin/main)
git diff HEAD                         # working tree (unstaged + staged)
git status --short                    # untracked files worth noting
```

For each changed file, find the RPython/PyPy counterpart **mechanically** (see next section). Read the upstream file and compare.

Report findings as a short **Parity audit** section before executing the follow-up task:

- Files modified with their mechanically-derived RPython counterpart (one line each).
- For each hunk that contains a candidate NEW-DEVIATION: cite the file:line in the diff, cite the RPython file:line it should have mirrored, quote the deviation concisely.
- Files modified where the mechanical rule does not produce a valid counterpart — treat these as structural deviations in themselves and flag to the user.

If the audit turns up **regressions** (NEW-DEVIATIONs introduced by the diff vs the audit base), list them in the report and fix them in Step 3 (regressions first, then the follow-up task, per Principle 6). Surface-and-defer only the regressions that fall under Step 1's session-scope opt-out list; the rest are fixed without asking. Pre-existing NEW-DEVIATIONs seen in modified files (already present at the audit base) are listed for awareness but **not** auto-fixed — that's a separate scope decision for the user.

### Step 3: Execute the follow-up task under parity

Run the follow-up task with these constraints:

- Read the relevant RPython source before writing code. Cite the file:line being ported in the code comment or in the summary to the user.
- When removing pre-existing code, confirm with the RPython upstream that the removal agrees with the upstream. If RPython has the code in a different form, **replace** rather than **delete**.
- When introducing anything new, check whether RPython has a counterpart. If so, port it. If not, either (a) port the RPython dependency that provides it, or (b) mark it clearly as a documented PRE-EXISTING-ADAPTATION with a comment citing the RPython decision point.
- If the follow-up task's natural implementation would introduce a NEW-DEVIATION, push back: state the tradeoff, propose the RPython-aligned alternative, and let the user choose. Do not silently introduce new deviations.

### Step 4: Verify

After making changes, run the checks that would catch regressions. Scale them per AGENTS.md "How much to verify":

- Start with the touched crate's tests and the `check.py` fixtures that exercise the changed path.
- A metainterp or optimizer parity change is cross-cutting: its regressions often surface as unrelated test failures. Run `cargo test --all --no-default-features --features dynasm,pyre-module`, and add the `cranelift` variant if cranelift paths are touched.
- Run the full `python3 pyre/check.py` when the change is cross-cutting, and before opening a PR.

Accept temporary performance regressions. Do NOT re-introduce shortcuts to recover perf. Record the regression in MEMORY and move on.

## Finding the counterpart: mechanical rule first

The goal is that **any** file in majit/pyre has its RPython/PyPy counterpart discoverable from the path alone. When this works, no mapping table is needed. When it doesn't work, the failure is itself a parity signal — the path structure diverged from upstream and that divergence should be on the audit list.

### Crate-level roots

Only the crate-level roots need memorizing. Everything below is mechanical.

| majit/pyre crate root | Upstream root |
|---|---|
| `majit/majit-metainterp/src/` | `rpython/jit/metainterp/` |
| `majit/majit-translate/src/jit_codewriter/` | `rpython/jit/codewriter/` |
| `majit/majit-translate/src/flowspace/` | `rpython/flowspace/` |
| `majit/majit-translate/src/annotator/` (future) | `rpython/annotator/` |
| `majit/majit-translate/src/rtyper/` (future) | `rpython/rtyper/` |
| `majit/majit-translate/src/translator/` (future) | `rpython/translator/` |
| `majit/majit-translate/src/translate_legacy/` | pre-roadmap ad-hoc — deleted at P8.11, no upstream |
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
| `majit/majit-codewriter/src/jtransform.rs` | `rpython/jit/codewriter/jtransform.py` |
| `majit/majit-backend-dynasm/src/x86/regalloc.rs` | `rpython/jit/backend/x86/regalloc.py` |
| `pyre/pyre-interpreter/src/baseobjspace.rs` | `pypy/interpreter/baseobjspace.py` |

Verify the counterpart exists before proceeding:

```bash
test -f rpython/jit/metainterp/optimizeopt/unroll.py && echo OK
```

### When the mechanical transform fails

If the derived path does not exist, **that is itself a parity finding**. Do not silently fall back to a multi-file "whichever upstream file has similar content" search without naming the problem. Classify the mismatch:

1. **File renamed in majit/pyre vs upstream.** e.g. `majit-metainterp/src/heap.rs` vs `rpython/jit/metainterp/heapcache.py`. The rename is a structural deviation; rename back unless there is a cited reason. If the majit file genuinely combines multiple upstream files, see #2.

2. **One majit/pyre file fuses multiple upstream files.** e.g. `pyre-interpreter/src/baseobjspace.rs` containing both `baseobjspace.py` and `abstractinst.py` content. Fused files are structural deviations: upstream's module boundaries carry semantic meaning (separate import graph, separate test units). Split as two files unless merging was explicit user intent. Even without splitting, annotate each function with the upstream file:line it was copied from.

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

Any local name that doesn't appear anywhere in upstream (grep `rpython/` and `pypy/` for the string) is a candidate NEW-DEVIATION. Examples seen in this repo:

- `pending_branch_other_target` — no upstream equivalent
- `last_comparison_*` cache — no upstream equivalent
- `pre_opcode_*` stack snapshot — no upstream equivalent (RPython uses per-PC liveness, not per-opcode state capture)
- `other_target` resume adaptation — no upstream equivalent

## Signals of NEW-DEVIATION

Treat these patterns as high-likelihood NEW-DEVIATION. Verify against upstream before removing, but flag in every audit.

- **Undocumented side tables / HashMap caches** in structs that RPython does not have. RPython's Box-identity model (Python object `is`) removes the need for most side tables; if pyre has one, it's often a flat-OpRef compensation that should either be lifted or removed.
- **`take()` / `Option::take` / `std::mem::replace` used to temporarily swap state for a snapshot**, where RPython would write and read a single `frame.pc` field. If the save/restore is over fields with no RPython counterpart (`pre_opcode_*`, `pending_branch_*`, `last_comparison_*`), the fields themselves may be the deviation.
- **Function names that don't appear in RPython**: grep `rpython/jit/` and `pypy/` for the name. If nothing matches, the function is a pyre-only helper. Check whether an RPython counterpart exists under a different name before declaring it NEW.
- **Comments like** `// pyre-only`, `// TODO remove`, `// workaround`, `// adaptation`, `// fallback`, `// temporary`, `// hack` without a cited RPython file:line. Honest self-reports of deviation.
- **`is_*_classlike` / `is_*_like` / `_maybe_*` helpers that reinterpret a raw pointer** as a different struct and peek at offsets. Type-confusion bombs — RPython has proper `isinstance` checks.
- **Feature flags / environment-variable switches** that gate behavior differences from RPython. RPython doesn't use env vars to switch semantics; if pyre does, both code paths are deviations in different directions.
- **"Simplified" resume / guard paths** (other_target resume, post-pop snapshot, …). RPython's resume is one path, not several.

None of these are automatically NEW-DEVIATION — check upstream first. But they warrant scrutiny.

## Output shape

When the skill is active and a response is being written:

1. **Parity audit** (short section, file:line references):
   - 3–10 lines max.
   - List each modified file with its mechanically-derived counterpart (or a "❌ mechanical transform failed: <reason>" note).
   - Cite each candidate NEW-DEVIATION as `<majit path>:<line> ↔ <rpython path>:<line> — <deviation summary>`.
   - Mark each finding with its disposition: `[auto-fix]` for regressions being fixed in this response (the default for regressions), `[deferred: <reason>]` for regressions that exceed session scope per Step 1's opt-out list, `[pre-existing]` for NEW-DEVIATIONs already present at the audit base (flagged for user awareness only, not auto-fixed).
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

Keep the tone direct and the citations concrete. No vague "I'll make sure this matches PyPy" language — always point at a file:line.

## Interaction with other mindsets

- This skill is the strict enforced form of the "First principles" above: under `/parity`, there is no wiggle room for "Rust language adaptations" except where the RPython line is explicitly cited.
- `/commit` (the commit skill) is compatible — parity auditing should happen before committing, not after.
- If the user invokes `/parity` inside a larger plan document (e.g. `jtransform_optimize_goto_if_not_port.md`), the plan's existing PRE-EXISTING-ADAPTATION annotations are respected; the audit focuses only on changes introduced since the plan was written.

## SPEC-DEVIATION: the six tests

Standing ruling: **pyre's *implementation* is a port of PyPy; pyre's *spec* —
what a Python program can observe — is CPython 3.15.** A behavioural difference
from PyPy is a parity regression **unless** a CPython 3.15 artefact shows PyPy is
wrong about what the caller observes. Then it is a spec fix, and PyPy's shape
still governs every other line on the way there.

**This is not a 3.11-vs-3.15 question, and reading it as one is why this cluster
gets re-filed every cycle.** Of seven adjudicated cases, six have no version
delta at all: `sched_setscheduler` has returned None since 3.3,
`PyUnicode_FSConverter` has accepted bytes since 3.3, PEP 529 surrogatepass is
3.6, `DirEntry` has cached its `stat_result` since PEP 471, audit hooks have
suppressed everything under `Exception` since at least 3.9, and the `'%U'`
attribute-error message is byte-identical in 3.13. These are standing
PyPy-vs-CPython divergences, not PyPy lagging a release. "3.15" pins *which*
CPython you read — `lib-python/stdlib-version.txt`, currently `v3.15.0` — it does
not narrow the rule to version lag, and the absence of a delta is not grounds to
refuse the exception. Conversely a real delta earns nothing on its own:
`DirEntry.stat()` object caching predates 3.11 **and** still follows PyPy, for
the reason in test 4.

#### What the spec governs

Only what a caller can observe: a return value, an exception's type / message /
attributes, object identity, an encoding-and-errors contract, and which argument
shapes are accepted or rejected.

Everything else follows PyPy **unconditionally** — names, module paths,
control-flow order, data structures, storage owner, JIT hints. See "Data
structure parity with RPython/PyPy" above. A structural divergence does not
become a spec fix by sitting next to one. If the only thing wrong is *how* pyre
reaches an answer PyPy also reaches, restore PyPy's shape.

#### The test — run in order, stop at the first leaf

**1. Can you write a Python snippet whose printed output differs?**
No → **STOP, this section does not apply.** Judge it as an ordinary parity
finding.

**2. Do you hold an admissible artefact for the 3.15 side?** One of three
routes, and say which:

  a. **In-tree pin** — a `lib-python/3/` test that asserts it, or stdlib code
     that depends on it, at `file:line`, quoted. Strongest, no network.
  b. **Measured** — a run on an interpreter whose version equals
     `lib-python/stdlib-version.txt`, one fresh process per case, reading the
     observable directly rather than inferring it from a return value.
  c. **C source at the pinned tag in a named checkout**, quoted —
     `git -C ~/Projects/cpython show v3.15.0:Modules/posixmodule.c`.
     `Modules/`, `Objects/` and `Python/` are **not in this tree** (`Include/`
     holds only a README); a claim about them names the checkout and the tag, or
     it is memory.

  **Prose is not admissible as evidence of the observable.** The docs and PEP 578
  both say an audit hook's error must derive from `RuntimeError`; the
  implementation has swallowed everything under `Exception` for many releases,
  and `pypy/module/sys/vm.py:496-498` coded to the prose. (Prose *is* evidence of
  PyPy's own intent — `interp_encoding.py:10`'s `# PEP 529` is what makes the
  surrogatepass declaration PyPy's position. That belongs to test 3, not here.)

  **A pyre in-tree comment is never the artefact.** Two of these deviations were
  justified by comments a *later* PR wrote. §4 records the decision, not the
  proof.

  **Platform clause.** When the behaviour is `#[cfg(windows)]`- or Linux-gated
  and neither oracle can execute it on this host, route (b) is unavailable — use
  (a) or (c) and record the platform in the finding *and* the code comment. An
  unqualified "measured on python3.15" on a Windows-only path names a run that
  could not have happened.

No artefact → **STOP. You may not invoke this section.**

**3. Do the two upstreams actually disagree?** Read the PyPy side at the line
that *decides*, and run `pypy3` when a fixture allows (see "The PyPy oracle").
The binary is corroboration, not the authority: in-tree PyPy is
`PYPY_VERSION = (7, 3, 24, "alpha", 0)` (`pypy/module/sys/version.py:18`) while
the installed `pypy3` is 7.3.22, so an `AttributeError` from the binary is not
evidence that the checkout lacks the function.

  They agree → not a spec conflict. If pyre differs from both, that is a plain
  **regression (§1/§2)** and no spec reasoning rescues it. The first question any
  of these findings must survive is "is pyre wrong against *both*?"

  **PyPy disagrees with itself → follow PyPy's own declaration, and this section
  does not open.** On Windows `pypy/module/sys/interp_encoding.py:9-11` declares
  `surrogatepass` per PEP 529 — and `getfilesystemencodeerrors` returns it, so
  PyPy's own shipped `os.fsencode` (`lib-python/3/os.py:847-874 _fscodec`) uses
  it — while `pypy/interpreter/unicodehelper.py:70-72` converts interpreter-level
  paths with `surrogateescape`. Honouring the declaration against the conversion
  path *increases* PyPy fidelity: file it under §4 with both lines cited, claim
  no spec authority for it, and do not let it carry an unrelated restructure.

**4. Is PyPy's shape load-bearing for a mechanism pyre also has?** Name the
mechanism PyPy's shape serves, **whether or not PyPy states one**. A stated
reason is sufficient evidence, not necessary — a shape with no English rationale
is not a shape with no reason. Search, and record the search, over:

  - the **whole definition** — decorator lines included — of the PyPy function
    that produces the divergent value, and of every helper on the path that
    produces it, in `rpython/` as well as `pypy/`;
  - any class- or module-level binding those bodies read: `_immutable_`,
    `_immutable_fields_`, `_attrs_`, `unrolling_iterable`, a module-level table;
  - triggers: `@jit.elidable`, `@jit.unroll_safe`, `@jit.dont_look_inside`,
    `@jit.look_inside_iff`, `@specialize`, `jit.promote`, `jit.hint(`,
    `we_are_translated`, `rgc.`, `make_sure_not_resized`, or a per-call rebuild
    that feeds one.

  Cite the trigger's `file:line` **and name the value it protects**: a hint that
  does not govern the value you are changing is not a trigger — `_immutable_ =
  True` at `interp_bytesio.py:16`/`:54` sits on `BytesIOBuffer`/`BytesIOView` and
  governs nothing `close_w` does. Found nothing? Record the negative search at
  `file:line`, the same discipline the next section imposes on "PyPy has no
  counterpart".

  Trigger present and pyre has the mechanism — the tracing JIT and its virtuals,
  the GC, the annotator, the RPython-level representation → **STOP. Follow
  PyPy.** That is implementation, which this ruling assigns to PyPy. Overriding
  it needs an explicit ruling from the repo owner: raise it and leave the finding
  standing until they rule.

  *Deliberateness decides nothing in either direction.* PyPy's own shipped copy
  of `test_memoryio.py` gates the `BytesIO.close()` `BufferError` assertion
  behind `check_impl_detail(pypy=False)` with "PyPy export buffers differently"
  — knowing non-conformance that still loses, because that constraint is one
  PyPy has and pyre does not. An accidental omission is no freer: it still has to
  clear 1, 2, 3, 5 and 6. And an omission is distinguishable from a design — PyPy
  already has `_check_exports` and calls it at `interp_bytesio.py:79`/`:120`/
  `:131` while skipping `:194`. Machinery that exists and is wired at the
  siblings is not a decision.

**5. Per-site artefact, and a blast-radius census.** For **every** pyre
`file:line` where you depart from PyPy, name the artefact that forces *that
site*. "Consistency with a sibling" is not an artefact. Then `rg` pyre's own
readers of the shape you are deleting — including `pyre-jit*` and `majit*` — and
record what you found. If a reader keys on the shape, the shape is
implementation: restore it there and bring it to the user.

  This is the check reviewers already do by hand: `error_is_exception` has
  exactly one caller (`vm.rs:2869`), so the audit-hook change cannot reach past
  `addaudithook`. Contrast `is_w`: `pyre/pyre-jit-trace/src/jitcode_dispatch/
  specialize.rs:5260 is_w_compares_by_value` lists exactly the seven types whose
  `is_w` compares by value and makes `:5343` decline the `IS_OP` fold, so
  "`is` is pointer identity in CPython" is not a spec fix — it is a JIT change
  with an unmeasured blast radius.

**6. Does pyre land on 3.15 across the whole decision, and on PyPy everywhere
else?** Adjacency is defined by **what reads the state you changed**, not by "the
same function". A change that lands pyre where **neither** upstream sits is a
defect regardless of which axis matched 3.15 — strictly worse than following
PyPy, and filed under §1. `DirEntry` again: the entry's stat cache is read by
`check_mode`/`is_dir` at `interp_scandir.py:317`, and `posixmodule.c` also
aliases lstat into the stat slot for a non-symlink and seeds the cache from
`is_dir`/`is_file`; PyPy reproduces both. Caching the object while doing neither
invents a third behaviour.

Reaching here: **not a parity regression.** File under `## 4. Structural
adaptations` as
`[3.15-spec] our_file.rs:line ↔ pypy_file.py:line — <observable>; evidence: <route + cite>`
so the next cycle sees it adjudicated instead of re-deriving it. That records the
decision; it does not close it — per the codex-review skill, §4 is a
classification, not a verdict.

#### "PyPy has no counterpart" is a search, not a default

If PyPy genuinely has no counterpart, tests 4 and 5 have nothing to evaluate and
**this section is not what licenses the code** — write it in the shape of the
nearest PyPy sibling, not in the shape of the C module, and say which sibling.
Record the exact search and its scope (`rg -n 'if_nametoindex' pypy/ rpython/`),
and re-run it at review time. `socket.if_nametoindex` shipped with the recorded
belief "PyPy has no `if_nametoindex`, so there is no `unwrap_spec` to port" —
false since PyPy `4faf5831374` (2023-12), registered at
`pypy/module/_socket/interp_socket.py:1315-1321` and `moduledef.py:19`. The
behaviour survived on other evidence; the reasoning had to be deleted (#1089). A
counterpart found later **re-opens** the deviation: re-justify it against all six
tests, or revert to PyPy's shape.

#### What you actually do

1. **Find the line in `pypy/` where the decision is made** — the converter in the
   `unwrap_spec`, the `return`, the `raise`, the format string, the missing call.
   Not the caller, not a wrapper, not a post-hoc fixup.
2. **Change that, and only as far as the observable reaches.** Keep PyPy's helper
   when PyPy has one; the check goes exactly where PyPy's sibling sites put
   theirs.
3. **If PyPy's shape genuinely cannot produce the observable, the computation
   changes too — but state what PyPy's shape cannot produce.** PyPy's
   `if_nametoindex` answers a miss by scanning `rsocket.if_nameindex()` and
   raising a one-argument `OSError`, which has no `errno` to report and so cannot
   satisfy `test_socket.py:1227`; calling libc `if_nametoindex(3)` is the minimum
   that can. Say that in the comment.
4. **Take the family the observable defines, and no more.** Every site in the
   family needs its own artefact under test 5. `if_nameindex`/`if_indextoname`
   came with `if_nametoindex` because the same converter contract governs them;
   that is not licence to re-route every `path_or_fd_w` caller off one finding.
   A one-site fix that leaves siblings inconsistent is a new mismatch; a
   forty-site sweep off one artefact is a bigger one.
5. **Comment at the site, citing both sides.** State the observable, the PyPy
   `file:line` whose decision you replaced and what it produces there, the
   evidence route with its cite, and the platform if the code is gated. A "do not
   restore this" instruction carries its evidence or it is worthless. Naming the
   other implementation is necessary here, which is rare: name the *symbol*
   (`posixmodule.c path_converter`, `sysmodule.c sys_addaudithook_impl`), never
   `CPython:` as a prefix and never "CPython's X" — the comment guideline holds.

```rust
// Both setters answer None. `interp_posix.py:3100`/`:3133` hand back the raw
// `handle_posix_error` result instead, which is 0 on every success and which
// `os.sched_setparam` does not publish. `posixmodule.c
// os_sched_setscheduler_impl` ends `Py_RETURN_NONE` (read at v3.14.6 in
// ~/Projects/cpython; identical at v3.11.0, clinic output=cde27faa55dc993e).
```

#### Why

PyPy is an implementation of CPython, not a competing specification. In each of
these cases a user program can tell the two apart: `os.sched_setparam` hands back
`0` instead of None; `sys.addaudithook` lets a `ValueError` escape instead of
swallowing it and dropping the hook; `BytesIO.close()` yanks the storage from
under a live `getbuffer()` view instead of raising `BufferError`; `str(OSError)`
says `Windows Error 3765269347` where `test_exceptions.py:432-438` demands
`Windows Error 0xe06d7363`; a failed `getattr` reports `'\udcfe'` escaped to six
characters instead of the code point. Shipping PyPy's answer there ships a bug —
often one PyPy itself concedes. But the same rule read one step too wide deletes
a JIT design, which is why tests 4 and 5 exist and why they are the ones that
actually get skipped.

Worked example (2026-08-13). `BytesIO.close()` with a live export.
`lib-python/3/test/test_memoryio.py:458-461` asserts `BufferError` from `write`,
`truncate` **and** `close`, then `assertFalse(memio.closed)`, and
`CBytesIOTest(PyBytesIOTest)` at `:836` inherits it unmodified for the C class
(route a). An A/B on two real interpreters (route b): CPython 3.14.6 raises and
stays open under both `io.BytesIO` and `_pyio.BytesIO`; PyPy 3.11.15 closes
successfully and the still-held memoryview degrades to `len(b) == 0`. PyPy's own
shipped copy of that test wraps the whole block in
`if support.check_impl_detail(pypy=False):` — it *skips* the assertion rather
than claiming the spec changed, which is a concession of non-conformance, not a
version gap. Test 4: `_immutable_` at `interp_bytesio.py:16`/`:54` governs
`BytesIOBuffer`/`BytesIOView`, not what `close_w` does; no other hint in
`close_w`, `close`, or `rStringIO.close`. Test 5: the check has one reader,
`bytesio.rs` itself; `bytearray_check_exports` (`builtins.rs:148`) reads the real
exporter counter fed by `getbuffer`, so it is neither vacuous nor JIT-visible.
The fix is one line — PyPy already has `_check_exports`
(`interp_bytesio.py:91-94`) with the exact `BufferError` text and already calls
it from `descr_init`, `write_w` and `truncate_w` (`:79`, `:120`, `:131`); the
only thing missing is the call in `close_w` (`:194-195`). `bytesio.rs:466` places
that same check before the store, so the object stays open on failure. Nothing
else moved. An op-by-op sweep confirms PyPy's own gate is now over-broad: `write`
and `truncate` *do* raise on PyPy today; only `close` still does not.
