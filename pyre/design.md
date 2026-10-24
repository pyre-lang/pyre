# pyre Design Charter

**Status**: normative. This document states the design axioms, the layer
architecture, the adaptation decisions, and the macro-strategy that govern
pyre and majit development. It is grounded in three sources, in order of
authority:

1. **The current PyPy/RPython source tree** (`pypy/`, `rpython/` in this
   repository) — the living, corrected result of twenty years of evolution.
   For any mechanism, this is ground truth for *what* to build.
2. **The PyPy EU reports (2004–2007)** — digested in `../eu_report.md`,
   judged in `../eu_report_assessment.md`. The reports are the canonical
   statement of *why* the architecture is shaped this way: the founders'
   reasoning, their measurements, and — read against later history — a
   catalog of dead ends that must not be re-explored. Code tells us what
   PyPy is; the reports tell us what PyPy meant.
3. **pyre's own measured results** — memory files, benchmark history,
   check.py.

When these conflict: current PyPy code beats the EU reports on mechanism;
the EU reports beat folklore on rationale; pyre's own measurements beat both
on what works *in Rust*.

---

## 1. Mission and layer correspondence

pyre's goal, stated once: **put an RPython-equivalent layer (majit) on top of
Rust, and a PyPy-equivalent (pyre) on top of that.**

| PyPy world | pyre world | Notes |
|---|---|---|
| RPython the language | **Rust** | The host language is no longer a Python subset; it is a real language with a real type system. See §3.1. |
| RPython translator (flowspace → annotator → rtyper) | **majit-translate** (`front/ast` → `flowspace/` → `annotator/` → `rtyper/`) over **Charon LLBC** artifacts | Same pipeline, same module names, run at `cargo build` time over extracted `.ullbc` instead of live bytecode. |
| `jtransform`/codewriter → JitCode | **majit-translate `codewriter/`** → JitCode | Same role, at `cargo build` time. pyre additionally runs a *second*, hand-written codewriter over user `CodeObject`s at runtime; see §3.7. |
| `warmspot` — translation-time portal generator | **split**: build-time derivation in majit-translate, hand-written warm entry in pyre-jit | Not a port. `apply_jit` is unwired and `warmspot.rs` is a `pub use` namespace; see §3.7. |
| metainterp, optimizer, resume, blackhole | **majit-metainterp / majit-trace** | Line-by-line port of the *tracing* JIT (pyjitpl5 lineage), not the 2007 PE JIT. The history, optimizer, resume and blackhole halves are what production runs; the `pyjitpl` MIFrame *tracer* is not — `mod pyjitpl` is private, `cpu.rs` calls that tracer retired, and pyre records through the hand-written full-body walker in `pyre-jit-trace/src/jitcode_dispatch/`. That walker is A1 debt of the same class §3.7 tracks. |
| x86/ARM/… hand-written backends (~300k LOC) | **majit-backend-dynasm / -cranelift / -wasm** | Three thin backends behind one trait, current primary dynasm; see §3.4. |
| incminimark GC | **majit-gc** (nursery + oldgen + incremental + card marking) | Port of the winner, not of Boehm/refcount/mark-sweep. |
| sandbox transform | **rsandbox** | Compile-time sandbox aspect. |
| `pypy/interpreter/` + `pypy/objspace/std/` | **pyre-interpreter + pyre-object** | Structural port, same names, same relative locations. |
| CPython 2.7/3.10 compat | **CPython 3.14** compat | Bit-exact; RustPython compiler front-end supplies bytecode. |
| GIL | **no GIL** | See §3.3. |

majit is a general framework: pyre is its primary consumer but majit must
never depend on pyre. Secondary consumers (aheui-mjit, toy interpreters,
wasmi experiments) exist deliberately — they are the generality proof, the
role PyPy's Prolog/JavaScript interpreters played for RPython.

---

## 2. Axioms

These are the EU-era strategic claims that survived twenty years of
evidence (assessment §7). They are not up for casual renegotiation.

**A1 — Single executable specification.** The interpreter source *is* the
semantics of pyre. The JIT is generated from it; it is never written to have
semantics of its own. Any behavioral divergence between compiled traces and
the interpreter is a **generation defect to fix in majit**, never an accepted
limitation, and never justified by "the interpreter is Rust". (This is the
meta-tracing principle; AGENTS.md states the enforcement details, including
frame identity.)

**A2 — Low-level policy is woven, not written.** Memory management, sandbox,
concurrency machinery, code generation, representation tricks live in the
translation/build layer or in majit, not in interpreter source. The EU
reports' strongest empirical finding stands: encoding a low-level decision
throughout an interpreter's source (CPython's refcounting) makes it
effectively unchangeable. pyre-interpreter must stay readable as "a
straightforward Rust program that executes Python bytecode".

**A3 — Runtime information is the optimization engine.** Static analysis
(annotator/rtyper prepass) exists to *generate the machinery*; the machinery
optimizes with runtime facts — tracing, promotion/guards, virtuals,
virtualizables, quasi-immutability. PyPy's own static-PE JIT losing to
tracing is the controlling precedent. Corollary: when choosing between a
smarter compile-time analysis and a runtime guard, default to the guard.

**A4 — Fall back, never restrict.** Full CPython 3.14 semantics by
construction: traces deopt to the interpreter (blackhole/resume) rather than
the language being restricted to what compiles. `sys._getframe`-class
introspection must work. Correctness is bit-exact — no float tolerances, no
"close enough" (root-cause the divergent operation instead).

**A5 — Tests over proofs; measurement over theory.** Every feature and every
bug fix carries a test. Default-flips of optimizations require the full
benchmark suite; negative and null results get recorded, not forgotten.
Expect D06.1's usual outcome — "no clear tendency" — and keep every
experimental mechanism behind a kill switch until evidence flips it.

**A6 — Port the winner, at parity.** The porting target is *modern* PyPy
source, line-by-line, with data-structure parity (no invented side tables,
no "simplified" shapes — AGENTS.md rules). The EU reports document many
mechanisms (multimethods, PE JIT, ootype, CPS stackless, refcounting) that
modern PyPy deleted; those are rationale history, not porting targets.

**A7 — Deletion is part of the method.** PyPy's team deleted its own
flagship implementations when evidence turned, usually for reasons they had
already written down as open issues. pyre inherits this: gated experiments
are cheap to delete; long-lived gates that never flip are debt; a mechanism
kept "because we built it" is a bug in the process.

---

## 3. Adaptations — where pyre deliberately diverges from the reports

The EU insights cannot be applied verbatim; the substrate changed. Each
adaptation below names what is kept and what is replaced.

### 3.1 "Analyse live programs" → analyse extracted artifacts

The reports' core front-end move — run full Python as a *preprocessor*, then
analyse the live image with bounded dynamism — solved a problem Rust does not
have: recovering static structure from an unspecified dynamic language.
pyre replaces it with:

- **Rust's type system** does what the annotator's type recovery did, at
  zero project cost, with real diagnostics. The chronic RPython pains the
  reports admit (no specification, whole-program-or-nothing, first-error-only,
  cryptic messages, RPylint as a band-aid) are structurally absent.
- **Charon `.ullbc` extraction** plays the role of the frozen live image:
  a whole-program low-level view of the interpreter for majit-translate.
  The price is the same one PyPy paid for image freezing — staleness: source
  changes are invisible until re-extraction. This is a permanent operational
  discipline (fingerprint skipping, forced re-extract), not a temporary bug.
- **Bootstrap dynamism** (RPython's metaclass tricks, generated gateway
  classes, memo functions) maps to proc macros and `build.rs` codegen.
  The annotator that remains in majit is the one RPython's *JIT* needed:
  binding-time and representation analysis over low-level bodies — greens vs
  reds, what is promotable, what is elidable — not type inference.

What is *kept* from the front-end story: the fall-back principle. RPython let
un-analysable code fall back to interpretation; majit lets un-lowerable
interpreter constructs fall back to residual calls / `dont_look_inside`.
The norm (A1) is that each such fallback is a tracked gap with a census, not
a silent permanent hole — the whole point of the prepass census workflow.

### 3.2 Hints: need-oriented, few, and load-bearing

The reports' hint philosophy transfers intact and is worth restating because
it constrains API design for majit forever: hints are **need-oriented** —
placed at the few points where runtime constancy is valuable (`promote`,
green fields, `elidable`, merge points, virtualizables) — and an
unsatisfiable requirement must be a loud error, not silent
de-specialization. majit expresses these as proc-macro attributes with the
exact RPython names. Adding a new kind of hint to work around a translator
weakness is the wrong direction; fix the translator (A1).

### 3.3 Threading: the aspect argument, with no-GIL as the target

D05.4 treats the GIL as one pluggable concurrency model among several and
keeps all concurrency policy out of interpreter semantics. pyre accepts the
framing, and **free-threaded is the target, not the present mechanism**. The
same separation argument is what makes that target reachable — because PyPy's
interpreter source never encoded GIL assumptions into *semantics*, a no-GIL
port is even expressible — but what ships today is a port of RPython's own GIL
(`thread_gil.c` → `majit-gc/src/rgil.rs`); see "GC under free threading"
below for why. Consequences that are already policy:

- GIL-dependent RPython machinery is ported, not merely named: `rgil` has
  production call sites, and a thread holds the GIL for as long as it runs
  pyre code. The pieces that remain name-only (heapcache reset on GIL
  release, `release_gil` effect info) are the JIT-side ones.
- Where RPython used ambient singletons justified by the GIL, pyre must
  justify each adaptation explicitly (TLS with a documented PyPy audit —
  the BACK_EDGE_BH_BUILDER precedent), never silently.
- The GC and object model must be designed for concurrent mutators *as an
  aspect-layer concern* (majit-gc), not by sprinkling atomics through
  pyre-object.

**GC under free threading (gh#396).** The core is an unchanged incminimark;
what supplies its exclusive-heap-access requirement has changed once.

- *Settled 2026-07:* a **stop-the-world safepoint harness** — the HotSpot/SGen
  shape, explicitly not PEP 703's non-moving route.
- *Superseded 2026-08:* pyre supplies exclusive heap access with **a port of
  RPython's own GIL** (`thread_gil.c` → `majit-gc/src/rgil.rs`). The harness's
  per-`gc_op` gate could not be made cheap on its own terms — its entry needs
  per-thread state, which on Mach-O is a `_tlv_get_addr` call on every
  allocation — and upstream has no such gate at all (`rg threadlocalref_addr
  rpython/memory/gctransform/framework.py rpython/memory/gc/incminimark.py`
  has no hits: the allocation path reads no thread identity). Holding the GIL
  across pyre code instead measured **−15.0%** on
  `synth/int_mul_ovf_bignum_promote`, 12/12 rounds faster, and makes `gc_op` a
  bare borrow of the singleton. The safepoint harness is still in the tree
  behind the GIL; retiring it is tracked separately. TLABs and per-thread
  store buffers were never built.
- *Contract restatement.* What incminimark actually relies on is (a)
  exclusive heap access during each collection **step**, (b) enumerability
  of all roots at that instant, (c) all inter-step mutations caught by the
  write barrier. Only (a) is what the two mechanisms above disagree about;
  (b) and (c) are the same either way. The audit that this holds for the
  *incremental* major is source-verified: major slices already execute
  at minor-collection time (`minor_collection_with_major_progress`,
  incminimark.py:824), hence inside STW windows; the barrier is deliberately
  newvalue-agnostic ("the incremental GC nowadays relies on this fact",
  incminimark.py:1516-1518) and its records are consumed at the next minor
  (VISITED clear + `more_objects_to_trace` re-append, incminimark.py:
  2079-2083). Concurrency therefore lives entirely in whatever supplies (a),
  and the ported collection algorithms are not rewritten either way.
- *Moving nursery is kept.* JIT inline nursery allocation is a performance
  pillar; embedder-boundary pointer stability is solved by pinning (upstream
  already has it: `GCFLAG_PINNED`, incminimark.py:148) or oldgen promotion,
  not by adopting a non-moving allocator.
- *Deviation zones are exactly four*, and everything else remains parity
  (N2): (1) allocation front-end, single bump pointer → per-thread TLAB
  chunks carved from the one nursery region (per-chunk pinning walls);
  (2) the barrier **producer** side — atomic header-flag RMW, per-thread
  SSBs flushing into the canonical `old_objects_pointing_to_young`, so the
  consumer code stays a line-by-line port; (3) codegen addressing:
  `nursery_free` baked static address → thread-context-relative, with the
  inline fast-path shape unchanged; (4) the safepoint subsystem itself,
  which has no upstream counterpart (and therefore no merge surface).
- *Non-options, with their failure mode*:
  per-access locks on the GC handle (fixes the data race, not the
  moving-collector root-visibility race — demonstrated by the gh#396
  cargo-test heap corruption); interior mutability without synchronization
  (hides the UB); per-thread heaps (breaks shared-heap semantics, and
  process-global caches already made it flaky in practice); a concurrent
  non-STW collector (requires rewriting incminimark's tracing/evacuation).
- *TLS discipline refined*: thread-local state is legitimate exactly where
  the design is per-thread (TLABs, mutator contexts); what is forbidden is
  a TLS raw pointer into unsynchronized shared mutable state — the gh#396
  defect. Staged execution plan (P0 soundness core → P1 TLAB/SSB/back-edge
  polls → P2 `_thread` + object-model epic) is recorded on gh#396.

### 3.4 Backends: three thin backends, not six hand-written

D08.2 closed with a research question: can the meta-JIT sit on a lower-level
code generator that owns regalloc/instruction selection? RPython answered
"no" by default and paid ~300k LOC of hand-written backends. pyre's answer is
to keep **three thin backends behind one trait** rather than owning six full
instruction sets. The current primary is **dynasm** — direct machine-code
emission via dynasm-rs, favored for compile latency and fine control;
**Cranelift** is the portable option that does take regalloc/instruction
selection downward (the literal "yes" to D08.2's question); **wasm** is a
target of its own. The trade is explicit: a less specialized lowest layer in
exchange for not owning six instruction sets. When a backend is the
bottleneck (compile latency, missing patterns), the recourse ladder is: fix
usage → dynasm fast path → upstream Cranelift work — not a hand-written pyre
backend.

### 3.5 What is explicitly *not* on the map (anti-roadmap)

Each item below was built by PyPy, measured, and deleted or abandoned —
the reports plus later history are the evidence (assessment §§2–3). Do not
rebuild them in pyre/majit without new, written justification that addresses
why the original failed:

- **Pluggable object spaces** (thunk/taint/logic/proxy spaces). The objspace
  *interface* stays; runtime-swappable semantics died — JIT-hostile,
  dispatch-costly. Security = rsandbox at the translation layer instead.
- **Multimethod dispatch + table slicing** in the object space.
- **Two-modules-per-type** layout; EU-era std-objspace shapes generally.
- **High-level backends / ootype** (JVM/CLI/JS). pyre-wasm is not this: it
  is a low-level target through the same lltype-like pipeline.
- **CPS/graph-transform stackless.** If deep coroutine support is ever
  needed, follow modern PyPy (stacklets/continulets — small per-platform
  assembly), not D07.1's transform. The "views" composability analysis
  remains valid input for any future design.
- **rctypes-style FFI** ("write extensions in RPython/Rust and translate
  them everywhere"). The C-extension story must follow the approaches that
  actually worked (cpyext-class emulation / HPy / cffi-class FFI) — §5,
  Phase C.
- **Exotic default representations** (ropes, string slices, prebuilt int
  boxes, optimal arrays). Adaptive **storage strategies** — the shape that
  shipped — are the porting target.
- **Syntax extensibility** as a goal. pyre tracks CPython exactly; the
  compiler front-end is not an extension point.
- **Naive refcounting**, conservative GC as default, or any GC not derived
  from the incminimark lineage.

**Anti-roadmap vs. provisional alternatives — a distinction.** The
exclusions above concern *re-adopting a mechanism PyPy abandoned* (ropes,
multimethods, ootype…) as a default; reversing one requires the §6
justification. They say nothing about pyre's own **provisional
implementations** of the *winning* mechanisms. Where the faithful port is not
yet in place, a simpler alternative is an acceptable **interim** — provided
the orthodox port stays the target (A6/N2), the deviation is tracked
(A7, §3.1), and it remains **reversible to the canonical implementation**.
Shipping an alternative now never forecloses correcting to the orthodox one
when measurement or need calls for it; that correction is expected, not a
policy change.

### 3.6 Configuration: small supported matrix, build-time resolution

D13.1's mechanism (options resolved at translation time, dependency-checked,
write-once) maps to Cargo features + build-time codegen; its *lesson* maps to
policy: the 1.0 compatibility matrix showed "any combination of aspects"
failing in practice, and post-EU PyPy pruned to essentially one supported
configuration. pyre keeps:

- **One blessed default configuration** that is always green and always
  benchmarked (this is what CI and check.py gate).
- Experimental behavior behind `PYRE_*` / `MAJIT_*` env gates or features.
  Default-off is the norm; a gate may be default-**ON** where its reason for
  existing is to keep the switched-off arm reachable as a one-binary A/B, and
  then its registry row has to name the trigger that retires it. Either
  polarity carries an owner-issue and an intended flip-or-delete decision. A
  gate is a staging area, not a home. The registries are `pyre/gate-triage.md`
  and `majit/gate-triage.md`; a gate read from the environment with no row
  there fails `gate_triage_complete`.
- Aspect combinations (e.g. wasm × JIT × GC modes) are individually declared
  supported or unsupported; silence means unsupported.

---

### 3.7 The portal boundary: warmspot split in two

Upstream mints the portal at translation time. `warmspot.apply_jit` — the body
of the translator task literally named "JIT compiler generation"
(`task_pyjitpl_lltype`) — derives each driver's green/red specification
(`make_args_specification`), rewrites the `jit_merge_point` and `can_enter_jit`
markers into calls (`rewrite_jit_merge_point`, `rewrite_can_enter_jits`), and
fills the fields `JitDriverStaticData` declares but never computes. Upstream's
`jitdriver.py` is an attribute container with two executable statements
precisely because warmspot writes the rest. pyre splits that pipeline across two
layers, unevenly:

- **The derivation and the marker erasure do run at build time**, over Charon
  LLBC, in majit-translate's `jtransform` and `CallControl::setup_jitdriver`,
  driven from `pyre-jit-trace/build.rs`. The derived green/red layout is
  asserted against the real MIR operands and a mismatch fails the build. This
  half is at the right layer and is not debt.
- **`apply_jit` itself is unported**, but not for the reason first recorded
  here. `task_pyjitpl_lltype` assembles every upstream-shaped argument and then
  returns `TaskError`; majit-metainterp's `warmspot.rs` is a `pub use`
  namespace, not an implementation. The blocker was written down as
  "majit-translate does not depend on majit-metainterp", which reads as though
  adding that edge is the fix. It is not: majit-metainterp already depends on
  majit-translate, so the reverse edge is a direct cycle and Cargo rejects it.
  Upstream has the same cycle — `warmspot` imports `removenoops` and
  `call_final_function` from the translator at module level — and breaks it by
  importing `apply_jit` *inside* `task_pyjitpl_lltype` rather than at module
  level. pyre's crate split is a faithful split of a cycle Python papers over,
  so the seam is the port, not a workaround: `VirtualizableInfoHandle` and
  `VirtualRefInfoHandle` already invert exactly this direction for the two
  objects `apply_jit` builds out of metainterp, and a third seam for
  `MetaInterpStaticData` needs no new edge. Seven `missing_task_leaf` sites
  exist across that driver and only two are JIT-related.

  What is genuinely missing is narrower than the stub suggests. The heaviest
  part of `apply_jit` — the generated `portal_runner` / `ll_portal_runner` — is
  already ported, by hand and with per-line citations, into pyre-jit's
  `call_jit.rs`; `make_args_specifications`, `make_jitcodes`,
  `make_virtualizable_infos`, `build_meta_interp` and `finish_setup_descrs` all
  have covered counterparts. The gap is the graph-rewrite family
  (`rewrite_can_enter_jits`, `rewrite_jitcell_accesses`,
  `rewrite_set_param_and_get_stats`, `rewrite_force_virtual`,
  `rewrite_force_quasi_immutable`, `add_finish`, `make_driverhook_graphs`,
  `create_jit_entry_points`) together with `inline_inlineable_portals` and
  `prejit_optimizations`. Mutable graph rewriting now exists in
  `majit-translate::register_configured_jitdrivers`: with `PYRE_PORTAL_SPLIT=1`, it copies
  and splits the interpreter graph before its merge point, following
  `WarmRunnerDesc.split_graph_and_record_jitdriver`. This does not yet supply
  `rewrite_jit_merge_point`'s original-portal/runner rewrite or move production
  tracing onto the split graph. The remaining boundary is the consumers of
  those graphs and the outstanding rewrite passes, not crate layering or a
  complete absence of mutable graphs. `warmspot-portal.plan.md` records the
  subsequent split-portal probes and their remaining runtime dependencies.
- **A second codewriter runs at runtime.** majit-translate's
  `transform_graph_to_jitcode` consumes a `FunctionGraph` once per build;
  pyre-jit's `transform_graph_to_jitcode` consumes a user `CodeObject`, is
  fallible, and runs unboundedly. Upstream has one, over the interpreter's own
  graphs. This is the A1 debt in this area — it is written, it carries Python
  opcode semantics, and it has already produced a wrong answer of exactly the
  class N3 names: in a chained blackhole resume `portal_frame_reg` aliased the
  caller frame, so an inlined callee's `LOAD_GLOBAL` indexed the caller's
  `names` table. A1 is **not** weakened to accommodate it; it stands as a
  tracked generation defect whose convergence target is majit-translate's
  codewriter.

**Measured cost, re-measured 2026-08-26** (dynasm release binary, best-of-3
`time.process_time()`, one million iterations, both loop shapes run against
CPython 3.14 and PyPy 7.3.20 on the same machine). The cliff moved rather than
vanished. On a **call-free** hot loop pyre now pays nothing for either hook —
0.0042 s profiled against 0.0042 s bare. On a loop that **calls** a small
function every iteration it still pays two orders of magnitude: 272–328× for
`sys.setprofile` and 178–215× for `sys.settrace` across runs, where CPython
pays 2.5–3.3× and PyPy about 1× and stays compiled. The charge is therefore
per *call event*, not per profiled frame, and the earlier 1168–2836× figure is
superseded — it was taken before the bracket below landed.

Event counts still match, which is what licenses the comparison: at a million
iterations all three runtimes report the same `call` / `return` / `c_call`
multiset. pyre additionally reports 28 `importlib._bootstrap` weakref-callback
frames that reference counting had already reclaimed before the profiled
window opened — a collector-timing difference outside the loop body, 0.003% of
the events, and not JIT-attributable, since it survives `PYRE_NO_JIT=1`.

**The missing bracket is restored, and the falsification below came back
clean.** Upstream brackets the portal itself: `PyFrame.execute_frame` wraps
`dispatch` — the function that carries the merge point and nothing else — in
`ExecutionContext.enter`, `call_trace`, then `return_trace` and `leave` in
`finally` clauses. pyre had put that bracket *inside* the plain dispatch body
(`eval_frame_plain_with_resume`) and left the JIT dispatch body bare. It no
longer does: `eval_with_jit_inner` now calls `call_trace` and `return_trace`
around the portal, and `frame_tracing_active` — the gate that had been sending
every traced or profiled frame down `execute_frame_plain` — is narrowed to the
per-frame `!frame.get_w_f_trace().is_null()`. The global `profilefunc` and
tracefunc disjuncts are gone and no events went missing, which is exactly what
this entry predicted. **The gate is no longer the implementation of frame
events for JIT-eligible frames**, and a profiled call-free loop now measures
the JIT.

**The two hooks failed for different reasons, and the obvious repair of the
second one is unsound.** Every number in the next three paragraphs is the
diagnosis as it stood before the recorder landed; what replaced it is measured
at the end of this section. Under `MAJIT_STATS=1` on the call-bearing loop,
`sys.settrace` keeps its compilation — `loops_compiled=4`, `loops_aborted=0`,
`guard_failures=3`, a summary line byte-identical to the bare arm's — so its
whole cost is executed per-call dispatch inside compiled code. `sys.setprofile`
instead *loses* the compilation: `loops_compiled` falls 4 → 1 and
`loops_aborted` rises 0 → 5, and after the fifth the green key is banned
(`abort_ceiling_banned=1`), so the next 174,776 attempts are refused outright
and the loop runs interpreted for the rest of the run.

Those five are reported as `abrt_bridge` with all three `giveup_*` splits
reading 0, i.e. they take the reason ladder's fourth rung — nothing staged,
fall back to `AbortReason::Generic`, whose integer *is* `ABORT_BRIDGE`. That
rung has no upstream counterpart: every `SwitchToBlackhole` upstream takes its
`Counters.ABORT_*` as a constructor argument, so a reason cannot be absent
there. `PYRE_FBW_DEBUG_ABORT` supplied the name the counters cannot:
`DispatchError::ProfiledResidualCall`.

The walker declined a residual call made from a profiled frame because a
builtin callee owes `c_call` / `c_return`, and the walker *decides* that call —
folding or residualising it — rather than tracing through the arm that reports
it, so a trace taken there would run its tail silently. That was right for a
builtin, and it was applied to every callee; the site's own comment recorded the
widening as known: a `CallFn` naming a Python callee "owes no `c_call` either,
but the fold gives the walker no way to tell".

**Narrowing it to upstream's line was tried on 2026-08-26 and reverted.**
`call_valuestack` diverts only `is_builtin_code(w_func)`, the plain `CallFn`
shape carries the callable as operand 0, and a `GuardValue` pins the
recording-time answer so that a later builtin at that site side-exits. All of
that works: with it, the three arms compile identically and the fixture's own
event multiset is unchanged. It is still wrong, and
`profile_hook_armed_before_a_hot_loop` fails on all three backends because of
it. Admitting the trace for the Python call also admits every **folded**
builtin in the same loop body, and a fold never reaches that dispatcher at
all — so `len`'s `c_call` / `c_return` stop firing while `callee`'s `call` /
`return` keep coming. The decline on the Python call was what protected the
folded builtin. That fixture says so in advance, and it is right: the gate can
only narrow "with the reporting to back it up".

So the ordering was fixed, and it is the opposite of the tempting one: the
reporting has to be **recorded into the trace** before the decline can narrow.

**That is what landed, and the reporting turned out not to need a new trace
op.** Upstream's own `test_cprofile_builtin` unpacks exactly one loop out of a
profiled `lst.append` / `lst.pop` pair, so "record, not decline" is upstream's
answer and not an invention. The invention is only the mechanism, and the shape
is forced by `resume_snapshot.rs`: pyre's blackhole re-enters Python bytecode
only, so any walker-emitted guard resumes at the enclosing opcode boundary and
a three-op bracket — recorded `c_call`, fold, recorded `c_return` — cannot
exist. Resuming *at* the CALL fires `c_call` twice; resuming *past* it skips
the builtin. The bracket has to be one residual whose leaf is
`baseobjspace.py call_args_and_c_profile`, which is also upstream's line.

pyre already had that leaf, reached through the `profile_frame` parameter
`call_valuestack` threads down to `call_function_carrier_with_mode`. What it
did not have was any residual that passed a frame: `flatten.rs
lower_simple_call_hlop_to_insn` collapses the whole Python CALL to a frameless
`RuntimeHelperKind::CallFn` residual, so the reporting arm existed in no
jitcode the walker walks. So the three CALL-family helpers now take
`call_valuestack`'s own diversion themselves
(`call_jit.rs residual_call_c_profile_frame`): a residual of one of them *is*
the bytecode's dispatch, which settles by construction the distinction
`c_profile_frame` draws between the bytecode's call and one made inside
`descr_call`, and the executing frame is the context's top frame. The gate is
upstream's, `frame.get_is_being_profiled() and is_builtin_code(w_func)`, asked
on the raw callable before the `_Method` unwrap, exactly where `eval.rs`'s own
CALL asks it. One helper covers tracing, compiled code and the blackhole
alike, because all three call `bh_call_fn_N`.

The walker's remaining job is the small half: keep those calls reachable.
`walker_foldable_runtime_helper` answers `RuntimeHelperKind::None` for the
three while a profiler is installed, so every fold, every descent into a
builtin's jitcode and the `CALL_ASSEMBLER` door decline exactly as they do for
a shape they cannot handle, and the uniform decline path is the generic
residual. The two doors into a *Python* callee were already closed by
`ec_hook_installed`, which is why the frame-level `call` / `return` never
needed this. The `iIRd` dispatcher carries the same binding: it has call folds
of its own and never had a profiled decline, which cost nothing only for as
long as `iRd`'s decline aborted the whole trace ahead of it, and would have
been a live hole the moment that decline went.

One claim examined along the way did not survive: that the method-form
dispatcher carries no profiled decline and so already drops events for a bound
builtin. Measured on `lst.append` / `lst.pop` in a hot profiled loop, pyre
reports 5000 of each, and so does CPython, and both
`profile_hook_armed_before_a_hot_loop` and
`profile_hook_c_call_is_bytecode_level_only` pass.

Two smaller defects fell out of the same investigation and are fixed.
`function.py`'s `is_builtin_code` — unwrap a `_Method`, take the function's
code, ask whether that code is a `BuiltinCode` — had been ported as a
forwarder to the gateway predicate, which answers a different question about a
different object; it was dead code, and reviving it as the real port is what
lets any of the tests above be written correctly. And `PyFrame::call` gated
its whole `_flat_pycall`-shaped fast path on `!get_is_being_profiled()`, where
`call_valuestack` diverts only `is_builtin_code(w_func)` and lets a `Function`
keep `funccall_valuestack`; that conjunct is now upstream's. Narrowing it
changed nothing about the cliff — `loops_compiled=1 loops_aborted=5` identical
before and after — so it is recorded as parity, and as a refuted hypothesis
rather than a fix.

**What upstream does not do.** It does not fold the tracing state away.
`ExecutionContext` declares `_immutable_fields_` with `profilefunc?` and
`w_tracefunc?`, yet the recorded traces read both as ordinary fields and guard
them, and the comment directly above that declaration says so: the fields
"should be known to a constant … but they're not". They are cheap because
they sit on the entry bridge, once per frame activation — not because they
disappear. Nor would the declaration help here: `quasi_immut_descr` requires a
constant struct operand, and pyre's `ec` is a portal red (`PYPYJIT_RED_VARS`),
so it is never one. The per-opcode half is a different mechanism again —
`dispatch_bytecode`'s explicit `we_are_jitted()` arm tests the *per-frame*
`w_f_trace` through the virtualizable `debugdata`, not the global tracefunc.

**The green's two halves land differently here, and pyre already holds the one
that survives.** `is_being_profiled` is a portal green
(`pypyjit_driver_layout.rs`) and `driver.rs make_green_key(code_ptr, pc,
is_being_profiled)` mints from it, so the profiled state does get its own cell,
counter and procedure token: `profile_hook_c_call_is_bytecode_level_only`
compiles 14 loops for its 7 arms, two keys each, which is that separation
counted rather than assumed.

The other half — folding `get_is_being_profiled()` out of the unprofiled trace
— has nothing to fold. Upstream's branch is foldable because the JIT records
through `call_valuestack` itself, so the test is an operation in the trace.
pyre residualises the call, and the branch lives inside `call_jit.rs
bh_call_fn_impl` where no recorded operation corresponds to it. Every read of
the flag in `pyre-jit` and `pyre-jit-trace` sits at trace time, at green-key
mint, or at portal entry — none per call in a compiled body — and
`funccall_valuestack`, whose `PyFrame::call` conjunct is discussed above, is
reached only from `pyre-interpreter`'s own CALL. This is §3.8's theme arriving
at the cost model: the opaque objspace has already hidden the branch the fold
was for, so the green cannot be where the remaining cost goes.

**Falsification, run 2026-08-26 — passed.** The prediction was that restoring
the activation bracket would leave event counts unchanged with the gate still
in place, and would then let the gate's `profilefunc`/global-tracefunc
disjuncts be dropped without losing events. Both held: the gate now reads only
the per-frame `w_f_trace` and the `call`/`return`/`c_call` multisets are
unchanged, so the bracket was what the gate was standing in for.

**The recorder landed, and a built control says it paid.** The decline is
gone: `residual_call_c_profile_frame` gives the three CALL-family residual
helpers `call_valuestack`'s C-profile diversion, and
`walker_foldable_runtime_helper` withholds only the *fold identity* of those
three helpers from a profiled walk, so the call is recorded as an opaque
residual instead of aborting the trace. `DispatchError::ProfiledResidualCall`
no longer exists.

Measured against a binary built at this commit's own parent — the only valid
control, since `PYRE_JIT=0` moves the bare arm as well as the profiled one and
so is not one — on a hot profiled loop, five interleaved repetitions on a quiet
machine, reported as ns/event because the bare arms are ~1 ms and divide out:

| shape | control | with the recorder | |
|---|---|---|---|
| builtin callee (`len`) | 999–1019 | 788–826 | **1.27x** |
| Python callee | 1305–1328 | 1136–1183 | **1.17x** |

`MAJIT_STATS=1` on the same script gives the mechanism rather than the effect.
The control reads `loops_compiled=3 loops_aborted=10`, with all ten on the
reason ladder's fourth rung (`abrt_bridge=10`, `abrt_unclassified_default=10`)
and two green keys banned outright (`abort_ceiling_banned=2`,
`abort_ceiling_refused=2`). This tree reads `loops_compiled=5
loops_aborted=0`, every abort counter at zero, and `caro_backedge` falls 12 → 4
because a banned key keeps re-entering the merge point it can never compile.

The successor claim this entry now stands or falls by: **the remaining cost is
neither the bracket, the recorder, nor the green.** The bracket is landed and
falsified-as-predicted; the recorder is landed and measured above; and the
green cannot be it, for the reason given two paragraphs up — pyre has the half
that separates cells and there is no per-call branch left to fold. Yet a
profiled call-bearing loop still runs ~1.1 µs/event against pypy's 0.4 ns and
CPython 3.14's 48.5, so the third mechanism the previous claim named as a
falsifier IS in play, and naming it is the next step rather than a contingency.

The first place to look is what a profiled compiled caller does with a callee
it may not inline. `inline_call.rs` declines on `ec_hook_installed()` for any
installed hook — correctly, since the walker has no route to record
`executioncontext.py`'s `_trace`, which calls back into app-level Python — so
the callee becomes a residual reaching `call_user_function_with_ctx`, where the
plain interpreter's CALL would have taken `funccall_valuestack`'s
`_flat_pycall`-shaped fast path. That is a hypothesis, not a measurement: the
control above says compiling is better than not compiling, not that the
compiled path is close to what it could be.


### 3.8 The fold layer: hand-written compensation for an opaque objspace

pyre records traces through 63 `try_walker_specialize_*` functions — 59 in
`jitcode_dispatch/specialize.rs`, one in `residual_call.rs`
(`load_deref`) and three in `inline_call.rs` (`instance_iter`,
`instance_next`, `generator_next`) — described by the 70 rows of
`SPEC_FOLD_ROWS` (one fold can back several rows, and row-less folds exist).
Four of those rows are not folds at all: every label ending in `_descent` —
`subscr_tuple_descent`, `binary_op_descent`, `compare_op_descent` and
`builtin_len_descent` — carry a row only so they can be suppressed and A/B'd
like the fold they replaced. Three of the four (`subscr_tuple_descent`,
`binary_op_descent`, `compare_op_descent`) are orthodox sub-walks through a
`try_walker_orthodox_*` entry, while `builtin_len_descent` gates the generated
gateway shortcut in `inline_call.rs` instead. Counting them as debt
overstates it by four; the fold count is 66. The three unary descent rows
retired when `flatten` began emitting canonical codewriter `inline_call_r_r`
operations to `descroperation::pos`, `neg` and `invert`. `pos` and `invert`
enter those bodies with no residual-call gate. The `unary_neg` and `unary_not`
rows were the residual fallback for a build whose bodies are unbound; those
folds are retired too, and the sites fall through to the canonical inline-call.
Nothing in this charter named that layer before 2026-08-26, which is itself
the finding: it is the largest single adaptation in the tree.

Re-derive every number here before citing it; this section has published
two miscounts, and both survived because the recipe beside them did not run.
Every command below is quoted as it must be typed.

* Rows — select the complete `spec_folds!` invocation by its symbol boundaries:
  `sed -n '/^spec_folds! {/,/^}/p' pyre/pyre-jit-trace/src/jitcode_dispatch/diag.rs | rg -cF '=> ("'`
  answers 70; replacing the final matcher with `rg -cF '_descent"'` answers
  the 4 descent rows. A fixed line range is invalid here: the previous range
  ended before the macro did and published a stale count.
  `-F` is load-bearing: without it the `(` is an unclosed regex group and
  `rg` exits 2 rather than counting.
* Definitions — `rg -c` reports one count *per file*, so it answers 59/1/3
  rather than 63. Sum the matches instead:
  `rg -o 'fn try_walker_specialize_' pyre/ majit/ -g '*.rs' | wc -l`.
  The descent entries are a separate population:
  `rg -o 'fn try_walker_orthodox_' pyre/ majit/ -g '*.rs' | wc -l` answers 18.
* Corpus — `ls pyre/bench/synth/*.py | wc -l`. Non-recursive **on purpose**:
  it answers 560. A recursive walk also sweeps archived subdirectories and
  over-counts the active corpus. Do not "fix" this to a recursive `find`.

`specialize.rs`'s own line count moved seven times in seven commits and is
not a usable identifier for a tree.

**Why it exists.** PyPy's objspace is RPython, so the tracer walks into it and
`optimizeopt/` only ever sees ordinary recorded operations. pyre's objspace is
compiled Rust reached through an opaque residual call, so every fold's first
job is to *recognise* that residual and answer in its place. Upstream has no
equivalent job, which means the obvious convergence — "retire the folds in
favour of the ported optimizer" — is not available as stated. Group by group:

| group | nearest upstream |
|---|---|
| unbox → raw int/float/bigint arithmetic, compare, truth, cast | `OptIntBounds`, `OptRewrite.optimize_INT_IS_TRUE`, `OptPure` — cleanup only |
| opaque builtin call → direct (mostly pure elidable) call | `OptPure.optimize_CALL_PURE_I`; recognition is `jtransform._handle_math_sqrt_call` and `@jit.elidable`, not a pass |
| residual → `new_with_vtable` / `new_array` so it stays virtual | `OptVirtualize` removes such ops; the emitter is `MIFrame.opimpl_newlist` |
| guarded heap field / array / mapdict read and write | `OptHeap` CSEs them; the emitter is traced `LOAD_ATTR_caching` |
| type-identity shortcut | retired 2026-08-24; was `OptRewrite._optimize_oois_ooisnot` plus `Optimizer.constant_fold` — a real pass |
| frame / execution-context introspection | none at any layer; PyPy forces the virtualizable instead |
| function-object construction | none |
| callee inlining (`instance_next`, `kwonly_defaults_inline`) | none as a pass; `MIFrame.opimpl_inline_call` reaches the callee by tracing into it |
| `_descent` rows — not folds | the descent itself (3 orthodox sub-walks) or the generated gateway shortcut (`builtin_len_descent`); the count is the 4 `_descent` rows above |

The groups are explanatory, not a second manually maintained census. Their
boundaries contain judgement calls (`set_add_method` may be read as a call or
a heap mutation, and the `super` rows mix virtual construction with frame
access), so attaching an independently edited `n` column made the table look
exact while letting it disagree with `SPEC_FOLD_ROWS`. The reproducible split
is 66 hand-written rows plus 4 `_descent` rows (3 orthodox sub-walks and one
gateway shortcut), re-derived on 2026-09-27.

Four groups have only downstream cleanup upstream, three have nothing at all,
and exactly one has a counterpart that is a pass rather than a consumer. So
the convergence target for this layer is **descent reach into the
interpreter** — making the objspace walkable — and not the porting of an
optimizer pass. A1 is not weakened by this; the layer stays hand-written
generation debt, but its stated repair is now the right one.

**What was tried.** A gateway-wrapper pilot gave `math.sqrt` its own jitcode
and a published `fnaddr`; the descent still declined on 433 transitive
blockers and retired zero folds. On 2026-09-25 the same shape retired every
`math` fold but `math_frexp`: each builtin is an interp2app gateway whose fast
arm pins its domain before the C call and then calls one unboxed leaf, so the
generic builtin descent walks it. A fast arm may not test the C result and
fall back to the slow path, because the slow path has no executable address
once that call has run. `frexp` followed on 2026-10-20: its gateway roots the
mantissa box across the exponent box with `with_roots!` and packs them with
`wraptuple2`, the shape `integer_divmod_pair` already descends. On 2026-09-26 the `str`
`startswith`/`endswith` folds followed `find`/`rfind`/`count`: each
gateway's fast arm now runs before the arity and keyword checks, whose kwargs
scan did not lower and had kept the descent declining, and calls one elidable
prefix leaf. The `builtin_fold1` / `builtin_fold2` rows (`abs`, `hash`,
`ord`, `min`, `max`) followed the same day: each builtin is a gateway whose
fast arm pins an exact int, float or `str` operand and either computes inline
or calls one `elidable_cannot_raise` leaf the optimizer can hoist. A census over the whole corpus as it stood
on 2026-08-31 — 521 synthetic fixtures then, every row observed — found
no fold with `consulted=0`, so the layer is not merely carrying dead arms.
`load_deref` alone never fires, and naming each of its early returns shows
why: all 38 declines across the 359 fixtures holding a nested function report
the same first guard, `OpRef::is_constant`, so its cell always arrives red.
That is a missing capability rather than dead weight — closure-callee
inlining needs the fold, the fold needs constant cells, and constant cells
need the inlining.

**What does not hold it in place.** Fixture coverage of a fold is
`# pyre-check: trace-shape=`: a code-object name and the operations the
optimized loop must contain or omit. No fixture carries `spec-folds=`. Three folds
have no trace-shape row because no loop compiles both with and without them
and differs by an operation: `store_attr_direct`, `load_type_attr`,
`frame_lasti`. Retirement is blocked by reach, not by headers.

**The bounded first step has been taken, and it does not settle the
question.** The type-identity group — `builtin_type`, `builtin_isinstance`,
`builtin_issubclass`, plus `builtin_hasattr` — left `SPEC_FOLD_ROWS` in
`4953fb0edf8`, on the evidence that suppressing each one moved no stdout, no
`[jit-stats]` counter and no wall clock. That is gate neutrality, not a
descent: nothing in that change demonstrates the trace now carries the
interpreter's own `abstract_isinstance_w` shape, and `isinstance` is still
recognised outside the layer by the `observed_replay_safe_isinstance` test in
`try_execute_residual_call_via_executor`, where it gates replay-safety, carries
no row, and the census cannot see it. Read that
commit's message with care — it claims five retirements including
`load_type_name_attr`, but its diff never touches that identifier and the
fold remains live in `try_walker_specialize_load_type_name_attr`.

**Falsification.** A retirement counts against this entry only if the trace
it leaves is the interpreter's own shape. `MAJIT_LOG=1`'s
`--- trace (before opt) ---` must show the residual replaced by ordinary
recorded operations, matching `PYPYLOG=jit-log-opt:FILE pypy3` on the same
fixture. A retirement that only shows unchanged counters — as the four above
do — proves the fold was not load-bearing, not that reach arrived. If reach
does arrive for a group and its folds still cannot be retired, then reach is
not the binding constraint and this entry is wrong: the layer would be
compensating for something the descent cannot reach in principle, and the
right response is to say what that is rather than to widen the descent again.

---

## 4. Norms (operating rules)

**N1 — Layering.** majit never depends on pyre. pyre-interpreter stays
traceable "straightforward Rust"; when majit-translate cannot lower an
interpreter construct, the default resolution is a majit improvement, the
fallback is a *tracked* residual call, and the forbidden resolution is
contorting interpreter semantics to please the translator. "Rust can't be
meta-traced" is never a valid excuse (AGENTS.md).

**N2 — Parity.** Line-by-line structural parity with modern PyPy/RPython:
same modules, names, data structures. No Rust-native collection where
RPython used an attribute/forwarded slot; borrow-checker workarounds minimal
and documented with the RPython original cited. Do not delete RPython
methods to "simplify". (Full rules: AGENTS.md; they are part of this
charter.)

**N3 — Frame identity.** One red frame per interpreted frame, everywhere:
tracing (MIFrame), resume, blackhole, bridges. Collapsing inlined callees
onto shared anchors is the known root cause of a whole bug class
(LOAD_GLOBAL namespace confusion, pycode miscompiles). The reports' own
virtualizable design assumed per-frame identity; RPython's 1-red-arg frame
shape is the convergence target.

**N4 — Correctness gates.** `cargo check` + `cargo test` (both feature
configs) green before commit; full benchmark suite (all 8) after JIT
changes with no regressions; bit-exact CPython 3.14 parity for observable
behavior; compliance-suite pass rate only moves up. Root-cause fixes only —
no workaround modules, no tolerances.

**N5 — Empirical flips.** An optimization becomes default-ON only with:
benchmark evidence on the suite (not one kernel — the 1.47×-gcc lesson),
green tests, and a kill switch that stays for one stabilization period.
Record refutations in memory/docs; D06.1-style negative results are a
deliverable, not an embarrassment.

**N6 — Translation latency is a first-class cost.** Whole-program
translation friction taxed PyPy for two decades (2h builds in D09.1; "the
attention of another dev for the whole sprint" culture). Guard pyre's loop:
incremental Rust builds, LLBC fingerprint skipping, prepass performance,
and no O(n²) in trace/compile paths (the #345 class). Wall-clock of the
edit-test cycle is reviewed like a benchmark.

**N7 — Documentation of thinking, not just code.** The EU reports' lasting
value is that the *reasoning* was written down, which is what made later
deletion rational rather than amnesiac. pyre's analogs — issue epics, memory
files, this charter — must record why, what was measured, and what would
falsify the decision. A major mechanism landing without a written rationale
is incomplete.

---

## 5. Macro-strategy: the ten-year arc

The EU project's honest final ledger — research vision delivered, product
usability consciously deferred, and the deferral costing a decade — defines
pyre's sequencing. The phases overlap; the ordering states *priority under
contention*, not a waterfall.

**Phase A — The JIT spine (now).** Make meta-tracing-by-translation
boringly correct and PyPy-fast on the benchmark suite. Concretely: eliminate
compilation cliffs (unported opcodes/`abort_permanent`, no-token loops,
recursion walls), converge frame/resume machinery on RPython shapes (frame
identity epics, pc_map retirement, resume rebuild parity), and close the
fib_recursive-class call-frame gap. Exit criterion: parity-class performance
with PyPy on the suite with the JIT never producing wrong answers. This
phase outranks everything, because it is the part the reports proved *can*
fail structurally (the 2007 JIT) and the part every later phase builds on.

**Phase B — Language and stdlib completeness.** CPython 3.14 compliance ramp
(regrtest enablement, test-infra rounds, GC-root classes of bugs), full
built-in coverage, generators/async, memory-model soundness under the
compliance suites. This is the "engineering after research" that PyPy
deferred; pyre schedules it as a standing workstream that grows as Phase A
stabilizes, not as an afterthought.

**Phase C — Adoption surfaces.** The two things history says decide real
use: **C-extension compatibility** (choose and execute an HPy/cpyext-class
strategy — with the rctypes failure and the cpyext cost curve as priors;
target a decision document, then years of grind) and **no-GIL parallelism
actually delivered** (thread scheduling, concurrent GC hardening, and the
free-threading ecosystem story that CPython 3.13+ opened). no-GIL is pyre's
principal differentiator against both CPython and PyPy; it must land as an
aspect-layer property (§3.3), never as interpreter-source complexity.

**Phase D — Platform axes.** The validated axes of l×o×p, modernized:
**wasm** as the p-axis (browser/edge/embedded — D11.1's reopened questions:
size budgets, reduced builds, sandbox-by-substrate), **rsandbox** as the
o-axis security aspect, cross-platform backend maturity (dynasm/AArch64/x86
via cranelift). Each platform combination enters the supported matrix
explicitly (§3.6) with its own CI, or stays experimental.

**Phase E — majit as a public framework.** The l-axis: majit as *the* way to
give a Rust interpreter a tracing JIT — the role RPython proved with
Pyrolog/Topaz/HippyVM. Requires: API stability, documentation of the hint
vocabulary, at least two non-pyre consumers in CI (aheui-mjit today), and
the discipline that pyre-specific needs land as general mechanisms or not at
all. This is last not because it matters least but because a framework
extracted from one working product beats a framework designed for
hypothetical ones — also an EU-report lesson ("RPython was 'just' the
implementation language" until it was proven).

**Standing constraints across all phases**: N1–N7; the anti-roadmap (§3.5);
and the A7 duty to delete. Every phase inherits the reports' deepest single
lesson: *the architecture survives because the interpreter stays a clean
executable specification and everything else is generated, woven, measured —
and replaceable.*

---

## 6. Amending this charter

This charter changes by evidence, in writing: a proposal must cite the
measurement or upstream-PyPy precedent that motivates it, state which axiom
or norm it modifies, and record what was tried before. Additions to the
anti-roadmap require the failure evidence; removals from it require new
justification addressing the original failure. The document history is part
of the document.
