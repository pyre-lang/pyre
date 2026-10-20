---
name: codex-review
description: Run a Codex-CLI static-analysis parity review of the current diff against the local RPython/PyPy sources, then act on it — fix the regressions and new mismatches in-session, and file the rest as follow-up tasks. Invoked via `/codex-review`. Use this skill at the END of a work cycle, or whenever the user asks to "run codex review", "codex 리뷰 돌려줘", "리뷰 사이클 돌려줘", "parity review with codex", "check this diff against RPython with codex", or any request to have Codex audit the current changes for RPython/PyPy porting fidelity. Reach for it any time the user wants a Codex-driven parity pass over the working diff, even if they don't say the exact word "codex-review".
---

# Codex parity review cycle

Each development cycle in this repo closes with a Codex-CLI review: Codex
statically compares the working diff (`git diff upstream/main`) against the local
RPython/PyPy sources and reports porting divergences in four sections. This
skill runs that review and then **acts on the report** so the cycle actually
converges instead of just producing a wall of text.

The split is deliberate:

- **Sections 1 & 2** are problems *this patch* introduced (parity regressions,
  and other new mismatches). They are the cost of the work just done, so they
  are fixed **now**, in this session, before the cycle is considered closed.
- **Sections 3 & 4** are not this patch's fault — section 3 is pre-existing
  debt and section 4 is intended structural adaptation. But "pre-existing" and
  "adaptation" are *classifications, not verdicts*: the reason to surface them
  is to **decide** — per finding — whether to fix now, fix later, or
  consciously leave in place with a cited justification. Each one gets a
  reasoned disposition (Step 4), never an automatic skip. The common outcome is
  a follow-up task, because chasing unrelated old debt would balloon the
  current diff and blur what this cycle changed — but that is the *result of a
  judgment*, not a reflex, and it is overridden when the fix is small, adjacent
  to the work just done, or removes a latent bug this patch can now reach.

## Step 1 — Run the Codex review

The exact review prompt is checked into the repo so the skill and the CI
workflow (`.github/workflows/codex-review.yml`) stay in sync. Read it from
`.github/codex-review-prompt.md` and pass it to Codex. Run from the repo root
so Codex sees the diff and the `rpython/`/`pypy/` trees:

```bash
PROMPT="$(cat .github/codex-review-prompt.md)"
codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-6.1-sol \
  -c 'model_reasoning_effort="medium"' \
  -C "$(git rev-parse --show-toplevel)" \
  --output-last-message .claude/codex-review-report.md \
  "$PROMPT" </dev/null
```

`--output-last-message` writes just the final four-section report to the file
(stdout still streams Codex's progress). Read `.claude/codex-review-report.md` for
the report to parse.

Notes:

- `--dangerously-bypass-approvals-and-sandbox` and `</dev/null` are required —
  without them `codex exec` blocks on an interactive approval prompt and hangs.
  See the `/codex` command for the same pattern.
- The prompt is read-only by construction (it asks Codex to *report*, not edit).
  Do not let Codex modify files in this step.
- `-m gpt-6.1-sol` is the default; honor a `--model <name>` the user passes in their
  invocation.
- Keep reasoning effort at `medium` for every review invocation, including
  retries; do not inherit or raise it from local settings.
- The diff base is `upstream/main` (the remote base, NOT local `main` or
  `origin/main`). This skill does **not** auto-fetch — `upstream/main` is
  whatever the user last fetched (they sync the `upstream` remote manually). If
  `upstream/main` is missing, stop and ask the user to add the remote
  (`git remote add upstream <URL>`) and fetch it; do not fall back to local `main`.

If `codex` is missing, unauthenticated, or exits non-zero, stop and report that
plainly — do not fabricate a review.

## Step 2 — Parse the four sections

The report uses these verbatim headings (guaranteed by the shared prompt):

```
## 1. Regressions to PyPy parity introduced by this patch
## 2. Other mismatches introduced by this patch
## 3. Pre-existing mismatches (already present before this patch)
## 4. Structural adaptations
```

Split the report on those headings. A section whose body is just `None.` is
empty — skip it. Every finding should carry an
`our_file.rs:line ↔ upstream.py:line` citation; if one is vague, open the cited
files and pin it down before acting.

## Step 3 — Fix sections 1 & 2 in-session (always)

These are regressions and new mismatches this patch introduced, so fix them
now. This is exactly the job `/parity` exists for, so run it under parity
discipline:

- For each finding, open the cited `our_*.rs` and the upstream
  `rpython/`/`pypy/` counterpart and confirm the divergence is real (Codex can
  be wrong — verify against the source, don't fix on faith).
- Port back to the upstream structure rather than inventing a new adaptation.
  Section 1 (parity regressions) is highest priority; section 2 next.
- A finding that turns out to be a false positive, or a genuine structural
  adaptation Codex mis-sorted into 1/2, gets reclassified — drop it from the
  fix list and note why. Don't fix something that isn't actually wrong.
- After the fixes, verify as AGENTS.md "How much to verify" scales it. Run the
  tests covering the fixed code. Run the full gate
  (`cargo test --all --no-default-features --features dynasm,pyre-module` and
  `python3 pyre/check.py`) when the fixes are cross-cutting. Don't claim the
  section is closed until the checks you ran pass.

If a section-1/2 fix is genuinely too large to land in this session (it needs an
unported upstream dependency, or cascades across many files), say so explicitly
and move that single item to a follow-up task (Step 4) with the blocker named —
but the default is to fix it here.

## Step 4 — Adjudicate sections 3 & 4 (decide; never reflex-defer)

Sections 3 (pre-existing mismatches) and 4 (structural adaptations) are not
this patch's fault, but that classification is **not a verdict**. The point of
surfacing them is to reach a reasoned *disposition* on each one. "It's
pre-existing" and "it's an adaptation" are never, by themselves, a reason to
skip — that is exactly the reflex `/parity` Principle 7 forbids
("PRE-EXISTING-ADAPTATION is a fix queue, not an absolution; 'it works today'
is never a sufficient reason").

For **each** finding (verify the citation against the upstream source first —
Codex can misclassify, and a real regression sometimes lands in section 3/4),
assign one disposition:

- **fix-now** — port it back in this session. This is the *default* under
  `/parity` Principle 7 whenever the fix is self-contained and the original
  blocker is gone. Choose it especially when the fix is small, sits in code
  this patch already touches/depends on, or closes a latent bug the current
  work can now reach. A fixed section-3/4 item moves into this cycle's diff
  exactly like a section-1/2 fix; verify the same way (Step 3).
- **won't-fix (documented)** — the divergence is *correct as-is*: a deliberate,
  still-valid structural adaptation (RPython↔Rust language gap, 3.11↔3.15
  opcode difference, GIL/free-threading, no-filesystem/no-libc on wasm, …).
  Confirm the upstream decision point it encodes and ensure an in-code comment
  cites it; if the comment is missing, that documentation *is* the fix and is
  made now. Record the justification in the report.
- **defer (blocked)** — fixing it is right but out of safe session scope: a
  specific, cited, still-real blocker (an unported upstream dependency, a layout
  change cascading across more files than this session can touch, a regression
  rooted in another unported optimization), or it is unrelated old debt large
  enough that pulling it in would balloon and blur this cycle's diff. Only then
  register a follow-up task with `TaskCreate` — one per finding or tight
  cluster, titled with the file and gist, body carrying the full Codex
  citation, the named blocker, and the convergence path. These deferred tasks
  are not worked now.

The bias for unrelated section-3/4 debt is still toward a follow-up task rather
than ballooning the diff — but reaching that outcome requires the judgment
above, not a blanket "section 3/4 ⇒ defer". Stating only "it's pre-existing"
as the reason is a process error.

## Step 5 — Report

Close with a short summary:

- Counts per section, each broken down by disposition — fixed-now /
  won't-fix(documented) / deferred(blocked) / reclassified (e.g. `1: 2 fixed,
  2: 1 fixed + 1 reclassified, 3: 1 fixed + 1 won't-fix, 4: 1 won't-fix + 1
  deferred`). A bare "3: 2 deferred" with no per-item reasoning is a process
  error (Step 4).
- What was changed in-session and the verification result.
- The follow-up tasks that were filed (ids/titles).
- The raw report path (`.claude/codex-review-report.md`) for reference.

## Relationship to CI

`.github/workflows/codex-review.yml` runs the **same** prompt on every PR push
and posts the four-section report as a sticky PR comment. CI only *reports*;
this skill is the side that *acts*. Keep the prompt change in lockstep: edit
`.github/codex-review-prompt.md` once and both consumers pick it up.
