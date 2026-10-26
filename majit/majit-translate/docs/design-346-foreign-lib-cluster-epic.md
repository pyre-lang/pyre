# gh#346 — Foreign-std-library cluster epic (vec! / malachite / String / Wtf8 / IndexMap)

## Why this epic exists

The measurements in this document have the following provenance. They are not
three readings of one unchanged input set:

- **268 total phaseA failures:** commit `eca75827fe43618b24328653c150751f9b5399e8`,
  release profile, default features, the then-current frozen `target/llbc`
  snapshot (not re-extracted), and `MAJIT_RTYPER_VERBOSE=1`. The exact command
  was:

  ```text
  touch pyre/pyre-jit-trace/build.rs
  MAJIT_RTYPER_VERBOSE=1 cargo build --release -p pyre-jit-trace
  stderr=$(ls -t target/release/build/pyre-jit-trace-*/stderr | head -1)
  rg --no-config 'PREPASS phaseA fail' "$stderr" | sort -u | wc -l
  ```

- **16 `try_from`-headed phaseA failures:** commit
  `ccdc1a52be2237365154976149e7b573db811d82`, release profile, default
  features, that worktree's frozen `target/llbc` snapshot, and
  `MAJIT_RTYPER_VERBOSE=1`. This is a filtered diagnostic count, not a total:

  ```text
  touch pyre/pyre-jit-trace/build.rs
  MAJIT_RTYPER_VERBOSE=1 cargo build --release -p pyre-jit-trace
  stderr=$(ls -t target/release/build/pyre-jit-trace-*/stderr | head -1)
  rg --no-config 'PREPASS phaseA fail' "$stderr" | rg --no-config 'try_from' | sort -u | wc -l
  ```

- **276 total post-Slice-A phaseA failures** (the *post-Slice-A baseline
  commit*, named that way everywhere below): release profile, default
  features, and `MAJIT_RTYPER_VERBOSE=1`.

  That commit existed only on a feature branch and was later replaced by the
  squash merge in PR #606 (`7944966b4dd`, "jit: retire foreign-lib cluster
  walls (gh#346 Slice A + B1a + B2 + B3a)"). The squash tree contains the
  whole PR, not the exact tree measured here, so the 276 reading cannot be
  reproduced from that merge. Re-run the command below at a named commit
  before comparing a new result to 276. Slice A changed `pyre-object`, so this
  run first re-extracted the default LLBC set with the repository's pinned
  Charon, then rebuilt and counted:

  ```text
  python3 scripts/extract-llbc.py
  touch pyre/pyre-jit-trace/build.rs
  MAJIT_RTYPER_VERBOSE=1 cargo build --release -p pyre-jit-trace
  stderr=$(ls -t target/release/build/pyre-jit-trace-*/stderr | head -1)
  rg --no-config 'PREPASS phaseA fail' "$stderr" | sort -u | wc -l
  ```

The 276 run is **not** a direct 268→276 Slice-A delta: the post-Slice-A
baseline commit descends
from `eca75827fe4` through intervening changes and uses re-extracted LLBC, while
268 used the older frozen snapshot. The 268 figure is therefore the historical
pre-Slice-A planning baseline only; the separately configured 276 run is the
post-Slice-A baseline for Slice-B scoping.

A root-cause analysis (deepest-root, per-record all-blockers) ranked the leaves by "sole-unblock"
leverage. The top three (`box_assume_init` / vec!, `malachite ::try_from`, and the
`Wtf8`/`Map::collect`/`IndexMap` group) looked independently high-value, but a **peel-and-recensus**
of each proved otherwise:

- vec!/NewList recognizer re-add → census only 268→266 (net −2), and it **breaks the build**
  (`newlist/r>r` unwired-snapshot).
- `malachite try_from` via `plain_int_w` `#[dont_look_inside]` → **net ZERO** (and the attribute
  didn't even take: `plain_int_w` is inlined by rustc/Charon before the JIT sees it).
- Combined vec!+try_from → 268→265, only `getitem_list` lifts.

**Root insight:** every hot list/dict/str *mutation* graph (`setitem_list`, `getitem_list`,
`w_list_setitem`, `setitem_bytearray`, `plain_int_w`) is guarded by a **stack** of foreign-std walls.
Peeling one exposes the next:

```text
box_assume_init (vec!) → malachite ::try_from → String::new → core::slice::get / join
  → Wtf8Buf::with_capacity → Enumerate::next → IndexMap::insert / get
```

No *single* leaf yields standalone census lift because the leaves **co-block the same graphs**. The
only way to lift the cluster's hot graphs is to close the whole wall-stack **together**. This is the
"census-line depth is a LIAR" phenomenon (documented in the rtyper-legacy rebase memory) at cluster
scale — the census reports only the first wall per graph.

## Slice ORDER — corrected 7/16 (the recognizer is a CAPSTONE, not a prerequisite)

The original plan put vec!/NewList first because `box_assume_init` is the *census-visible frontier*
wall. That was wrong, and it was proven wrong empirically (see "Verdict" below). The front/mir vec!
recognizer is a **shared front-end rewrite**: it plants `OpKind::NewList` into a graph regardless of
whether that graph will lift through the two-phase prepass or drop to the legacy walker. The legacy
walker cannot lower `NewList` (it never runs `rtype_newlist`), so its raw op reaches the assembler's
default arm and emits `newlist/r>r` — an opname with no blackhole handler — which breaks the build via
`default_bh_builder_unwired_set_matches_task_85_snapshot`. Therefore the recognizer is only safe once
**no** vec!-bearing graph can still drop to legacy, i.e. after every co-blocking wall closes.

**Corrected order: Slice A (try_from) → Slice B (String/Wtf8/IndexMap/slice/iter) → Slice C
(vec!/NewList recognizer, LAST).**

### Slice A — malachite `try_from` (task #21, FIRST, no deps)

**Census (filtered run at `ccdc1a52be2237365154976149e7b573db811d82`; exact
configuration and command above):** 16 phaseA graphs have a `try_from` first wall, split into TWO
unrelated families:

- **13 graphs = malachite** (`["malachite_bigint","bigint","<Impl>","try_from"]`) — the Slice-A target.
- **3 graphs = int-to-int** (`["core","convert","num","<Impl>","try_from"]`: `int_pow`, `pow`,
  `opcode_get_iter`) — NOT malachite. `int_pow` is `u32::try_from(vb)` with a genuine
  `Err(_)=>return Err(memory_error)` branch (descroperation.rs:860). Out of Slice A's scope; defer.

**Reachability of the 13 malachite graphs** (traced through the census onion): every one reaches the
wall through exactly one of three functions — `setitem_list` (baseobjspace.rs:2359), `getitem_list`
(baseobjspace.rs:1268), or `plain_int_w` (listobject.rs:362, reached via `w_list_setitem` /
`w_list_append` / directly).

**The crux — the narrowing is NOT uniform (verified by direct `rg` census of every
`i64::try_from(w_long_get_value(...))` site):**
- **bucket (a) panic** — `plain_int_w` (listobject.rs:366) is `.unwrap_or_else(|_| panic!())`,
  semantically identical to `jit_bigint_to_i64_value` (longobject.rs:364). A pure target-swap is
  bit-exact HERE.
- **bucket (c) genuinely-fallible** — `getitem_list` (baseobjspace.rs:1303) and `setitem_list`
  (baseobjspace.rs:2372) — and 6 more `getindex_w`-inlined siblings (baseobjspace.rs:1362, 1427,
  1493, 2523, 2547, 2584; 8 total) — are `match i64::try_from(...) { Ok(i)=>i, Err(_)=>return
  Err(IndexError/ValueError) }`. Overflow throws a **Python exception**, NOT a panic. A panic-residual
  swap would be a CORRECTNESS REGRESSION (bit-exact violation). The coercion is deliberately inlined
  per-callsite ("the same rtyper reason as getitem_list") — no shared choke point.

So Slice A is NOT the "pure target-swap" the original note assumed. Two distinct lowerings:

1. **bucket (a) `plain_int_w`**: pure target-swap of `i64::try_from(<opaque BigInt>)` →
   `jit_bigint_to_i64_value` (both `#[dont_look_inside]` residuals ALREADY registered,
   jit_fnaddr.rs:911-919), guarded on `tyref_is_opaque_bigint` (mir.rs:11226). Mirror
   `bigint_binop_residual_path` (mir.rs:6457). But the enclosing `Result` local — `plain_int_w`'s
   `try_from` result is immediately `.unwrap_or_else`d, so the MIR may already fold it; verify.
2. **bucket (c) `match i64::try_from(<opaque BigInt>)`** (8 sites, ALL verified by an exhaustive
   census: `getitem_list` baseobjspace.rs:1303, `getitem_str` :1362, `getitem_bytes_like` :1427 +
   :1493, `setitem_list` :2372, `setitem_bytearray` :2523 + :2547, `byte_w` :2584 — each
   `Ok=>i, Err(_)=>return Err(IndexError/ValueError)`): lower to the runtime-discriminant `Result`
   aggregate, using `try_lower_checked_neg` (mir.rs:8610) as the EXACT template (it already builds a
   runtime-disc `Result`/`Option` via `emit_tagged_pair_aggregate`, mir.rs:9339). Emit:
   `fits = jit_bigint_to_i64_fits(bigint)` (residual, returns 1 when it fits), `disc =
   BinOp("eq", fits, 0)` (Result convention: **Ok=0, Err=1** — mir.rs:8259, so disc=0 when it fits),
   a `payload`, then `emit_tagged_pair_aggregate(disc, payload)` with `Ok`=tag 0. The
   `Err(_)=>return Err(IndexError)` arm survives UNTOUCHED as real user code reached via the disc==1
   switch branch. This is the substantive part of Slice A.

   **⚠️ EAGER-PAYLOAD PANIC HAZARD (the one non-obvious constraint):** `emit_tagged_pair_aggregate`
   writes `__pos_0 = payload` UNCONDITIONALLY in the call's block (mir.rs:8682-8689 pushes the payload
   op before the goto), *before* the consumer's discriminant switch runs in a successor block. For
   `checked_neg` the payload op is `neg` (total, never traps), so eager eval is safe. But
   `jit_bigint_to_i64_value` **PANICS on overflow** (longobject.rs:366). If `payload =
   jit_bigint_to_i64_value(bigint)` is computed eagerly and the BigInt does NOT fit, the residual
   panics in the walker/blackhole graph BEFORE the switch can route to the `Err` arm — turning
   `lst[2**100]` into a panic instead of the correct `IndexError`. **A bit-exact regression.**
   Resolution: `__pos_0` is only READ on the Ok path (disc==0 guarantees fits), so its value on the
   Err path is dead — the payload op only needs to be TOTAL (non-trapping), not correct-on-overflow.
   Use a NON-PANICKING total residual for the payload — add `jit_bigint_to_i64_value_or_zero`
   (`i64::try_from(num).unwrap_or(0)`, mirrors the existing bucket-(d) idiom at listobject.rs:1317)
   next to the fits/value pair in longobject.rs and register it in jit_fnaddr.rs. On the Ok path it
   equals `jit_bigint_to_i64_value`; on the (dead) Err path it returns 0 instead of panicking. This is
   RPython-faithful: upstream `toint`-after-`fits_int`-guard is elidable *because* the guard proves it
   fits; with no guard in the interpreter graph, the total form is the honest lowering.

   For **bucket (a) `plain_int_w`**: the same aggregate lowering serves it — its `.unwrap_or_else(|_|
   panic!)` reads `__pos_0` on the (always-taken, precondition-guaranteed) Ok path. Prefer UNIFYING
   both buckets on one recognizer for `<opaque BigInt>::try_from` rather than a separate target-swap.

Fail-safe by construction (a non-BigInt / unlisted target leaves the residual `<Impl>` call the census
Skips). Verify: `cargo test -p majit-translate` + census set-diff (expect the 13 malachite graphs to
drop OR expose their NEXT co-blocking wall) + `default_bh_builder_unwired_set_matches_task_85_snapshot`
green + `check.py` bit-exact 3-backend (list/dict subscript with huge int indices exercises the
Err arm — MUST still raise IndexError/ValueError, not panic).

### Slice B — String / Wtf8 / IndexMap / slice / iter-adapter residuals (task #22, after A)

Scoped 7/17 on the separate post-Slice-A census run
(the post-Slice-A baseline commit, 276 total phaseA; exact
configuration and command above): **46 distinct unregistered
residual paths**, saved to `/tmp/sliceB_residual_ranking.txt` (de-escape `\"`→`"` before counting).
The three walls that gate the hot dispatcher heads (innermost per record):

- `iter::adapters::map::Map::collect` (25 hits) → `setitem_slot`, `w_list_append`, `setitem_list`,
  `w_list_setitem`.
- `core::str::<Impl>::as_bytes` (7) → `getitem_slot`, `dict_entries_get_str`.
- `sync::atomic::AtomicBool::store` (15) → `object_setattr` (via `w_type_set_abstract`).

The 46 paths split into four orthodoxy buckets — **do NOT residualize a bucket-(N) core op** (that is
the non-orthodox band-aid: a silent perf regression no correctness test catches):

- **(F) foreign-opaque residual** — `Map::collect`, `Wtf8::*`, `IndexMap::{get,insert,get_index,
  with_capacity,get_index_mut}`, `AtomicBool::store`, `BigInt::{sign,to_u32}`, `fmt::rt::Argument::
  new_debug`. NOT auto-collected because the owners are EXTERNAL crates (`indexmap`/`wtf8`/`core`/`std`)
  → `iter_local_fns` (charon-reader:208) never sees them, and `collect_foreign_opaque_method_externals`
  (mir.rs:10681) only walks LOCAL opaque ADTs with a self-receiver and a modelable result
  (`foreign_opaque_method_result_valuetype` mir.rs:10773 declines `Option`/enum/tuple/ref). `BigInt`
  `sign`/`to_u32` owner IS opaque+local but the enum/`Option` result is declined. Fix = wrap the
  **pyre-side caller** in `#[dont_look_inside]` + `push_alias_pair`. Template already shipped:
  `w_dict_{store,lookup}_int_strategy` (jit_fnaddr.rs:557-580) residualize their internal
  `IndexMap::{insert,get}` wholesale.
- **(N) native-lowerable core op** — `core::slice::{get,first,index,as_ptr,chunks_exact}`,
  `from_raw_parts`, `f64::abs` (rtyper has `rtype_abs` rfloat.rs:216 → `float_abs`, method callsite
  unwired), `num::checked_div` (→ runtime-disc `Result`, template `try_lower_checked_neg` mir.rs:8610),
  `convert::{from,num::try_from}` (int-to-int `try_from` = the Slice-A-deferred `int_pow`/`pow`/
  `opcode_get_iter`), `RangeInclusive::new` (→ `rtype_builtin_range` rrange.rs:160), `Rev::next`,
  `mut_ptr::add`, `Vec::{index,index_mut}`. Real rtype/recognizer, never a residual.
- **(C) vec!/box/alloc cluster** — deferred to Slice C (capstone).
- **(P) pyre-internal accessor** — `set_async`, `EVAL_NESTING`, `EXC_CLASS_REGISTRY`,
  `subclass_range_read`, `GcType::type_id`, `Constants::{deref,index}`. `push_alias_pair` siblings of
  the jit_fnaddr.rs:697-758 runtime-state accessors.

**Sub-slice order (cheapest-per-leverage first; census depth LIES — head movement only when B1+B2
co-land):**

- **B1a (FIRST, done)** — the `str::as_bytes` / `Wtf8::as_str` identity-fold gap. Root cause:
  `is_string_as_bytes_identity` (mir.rs:7434) and `is_string_to_str_identity` (mir.rs:7405) both gate on
  `deref_impl_owner_leaf(fd)` matching an owner leaf, but `deref_impl_owner_leaf` (mir.rs:10530) resolves
  the owner through `resolve_impl_owner_adt_def_id_free`, which needs an ADT def-id. The primitive `str`
  is a `{Builtin:"Str"}` node with no def-id, so the literal `"str"` arm is dead; `Wtf8::as_str` fails a
  different way (`is_string_to_str_identity` gates owner `== "String"` only). Fix: gate both folds on the
  **receiver** being a string value via `tyref_is_string_value` (mir.rs:11539, which already handles
  `Builtin("Str")` + `String` + `Wtf8`/`Wtf8Buf` uniformly) rather than the impl-owner leaf. Do NOT
  touch `deref_impl_owner_leaf` (it also drives the `cast_pointer` thin-pointer rewrite).
- **B1b** — the (P) register-only accessors + `fmt::Argument::new_debug` fold into the existing fmt
  family (mir.rs:1757/7500).
- **B2** — the (F) foreign residuals (`Map::collect`, `AtomicBool::store`, `IndexMap` family, `BigInt`
  `sign`/`to_u32`). The str-dict path is STRUCTURALLY DIFFERENT from int/bytes: it uses the shared
  `dict_entries_get_str` helper (dictmultiobject.rs:179, not `#[dont_look_inside]`), not a dedicated
  strategy leaf — add a `w_dict_lookup_str_strategy`-style residual leaf or mark the shared helper.
- **B3 (LAST in B, real rtyper work)** — the (N) native lowerings.

### Slice C — vec!/NewList recognizer + repr-generic rtype_newlist (task #20, LAST, capstone)

- **Ca** Re-add the front/mir recognizer (verbatim from the reverted work in PR #310): match
  `box_assume_init_into_vec_unsafe(box [e0..eN])` → `OpKind::NewList{args}`. Helpers
  `read_array_literal_elements` (mir.rs:13581) + `fmt_path_ends_with` (mir.rs:13673) still in tree.
- **Cb** `remove_dead_aggregates` (model.rs:2469) already sweeps the dead `Box::new_uninit`. No work.
- **Cc** repr-generic `rtype_newlist` (rlist.rs:395): accept BOTH `ListRepr` (Resized) AND
  `FixedSizeListRepr` (Fixed) — a never-mutated vec! annotates NON-resized → `FixedSizeListRepr`
  (rmodel.rs:3208). Fixed arm builds via `build_ll_newlist_helper_graph(ListLayout::Fixed)` +
  `build_ll_fixed_setitem_fast_helper_graph`. RPython-faithful (rlist.py:338-344 is repr-generic).
  **This code was written and verified to compile + pass `cargo test -p majit-translate`; it is
  correct but INSUFFICIENT alone — it only lifts graphs that fully rtype.** Recover it from this
  session's transcript / the reverted diff when Slice C lands.
- **Cd** Only after Slices A+B close every co-blocking wall does the recognizer become safe: with no
  vec!-graph dropping to legacy, `rtype_newlist` runs on every one, decomposing `NewList` to
  `ll_newlist` + `ll_fixed_setitem_fast` residual calls before assembly → no raw `newlist` reaches
  the default arm → `newlist/r>r` snapshot stays green. **Gate every Slice-C attempt on
  `default_bh_builder_unwired_set_matches_task_85_snapshot` (reads `insns.bin`, NOT census stderr).**

Verify each slice: `cargo test -p majit-translate` + census phaseA set-diff (count distinct
`[PREPASS phaseA fail]`) + `default_bh_builder_unwired_set_matches_task_85_snapshot` green +
`check.py` bit-exact 3-backend.

## Metric

The metric is the distinct `[PREPASS phaseA fail]` count produced by the exact
commands and configurations recorded above. Use set-diff only between runs made
from the same source commit ancestry and the same extracted LLBC snapshot. In
particular, 268 at `eca75827fe4` is the historical pre-Slice-A planning baseline,
16 at `ccdc1a52be2` is only the filtered `try_from` subset, and 276 at the
post-Slice-A baseline commit is separately configured; do not treat
268→276 as a Slice-A regression. GOTCHA: the census **stderr** logs only phaseA
*reasons* — the emitted `newlist/r>r`
opname lives in `insns.bin` (build OUT_DIR `target/debug/build/pyre-jit-trace-<hash>/out/insns.bin`;
`strings insns.bin | rg newlist`), not stderr. The cluster's hot graphs (`setitem_list`,
`getitem_list`, `w_list_setitem`, `setitem_bytearray`) only lift once ALL their stacked walls close —
expect most of the census movement on the LAST slice.

## Superseding census — 2026-09-02

The Slice A/B/C item lists above are historical. They were derived from a
268/276-record census, and the two-phase prepass has since grown its subject
set by an order of magnitude, so their named subjects no longer exist in the
census. In particular `int_pow`, `pow` and `opcode_get_iter` (Slice A's
deferral note) name no record now; re-derive a population from a fresh census
before working any item list in this document.

**Provenance.** Branch `rtyper2` with the batch below applied: release
profile, default features, `target/llbc` re-extracted for `majit-rlib
pyre-object pyre-interpreter pyre-jit` at each tree, `PYRE_RTYPER_VERBOSE=1
cargo build --release -p pyre-jit-trace`, graded by the build's own exit status
and by the stderr file's mtime against the build start, never by `ls -t` alone.

The baseline and the probe of each pair below were taken on one tree, and the
branch has been rebased past that tree since, so those commits are no longer
reachable by hash. What survives a rebase is the pair — a movement measured
against its own baseline — not the absolute, which the section after next
re-measures on a later base and finds different.

A census is comparable only to another census taken on the same tree with the
same artefacts. Both halves matter: a rebase that touches any of the four
extracted crates restales `target/llbc`, and the chain order is therefore
extract → test → census, never test → census on artefacts from an older tree.

**Phase A = 1883 records, one record per subject.** Reading a record: split on
`(?=\[PREPASS )`; a trailing `thread '<unnamed>' … panicked` block belongs to
the NEXT record; the subject is the record head, and the nested chain inside
one record is a free victim map. The text is JSON-escaped and quotes
identifiers in backticks, so a needle must be copied out of the file, never
retyped.

| family | records | share |
|---|---|---|
| unregistered CallRegistry path | 1203 | 63.9% |
| Blocked block | 202 | 10.7% |
| classdef-less `getattr` | 171 | 9.1% |
| UnionError | 147 | 7.8% |
| no analyser registered | 44 | 2.3% |
| undefined operand | 27 | 1.4% |
| other | 89 | 4.7% |

### The frontier is flat, and the census total is not the metric

`MAJIT_RTYPER_FRONTIER=1` resumes past every unresolvable call instead of
stopping at the first, so each subject reports its whole call-wall set rather
than the wall it happened to reach first. On this tree 629 subjects carry wall
rows; 397 of them have a single distinct wall, and **the largest single wall
serves 25 subjects** (`undefined operand`), the next 8, then 7, 7, 6, 6. No
registry closure on this frontier buys more than ~1.3% of phase A.

That flatness is why a wall's record count going to zero says nothing on its
own. The `IndirectCall` wall (253 records, then the single largest leaf) was
closed at `9bfbafe3f80`: the wall went 253 → 0 and **not one graph lifted** —
all 253 subjects relocated, to Blocked block (60), `iter::sources::once::once`
(52), `Chain::next` (22), UnionError (15) and eleven smaller heads. Grade a
wall by the per-subject victim map — for each subject that named X in the
baseline, what is its head cause in the probe — not by the total.

Two co-blocking stacks are already measured and should be assumed, not
rediscovered:

- `getattr_str_impl` (baseobjspace.rs:5791) holds `w_method_new` at :5928 and,
  through `object_getattr_miss` at :6345 → `getdictvalue` → `has_mapdict_layout`,
  `W_INT_USER_GC_TYPE_ID` at mapdict.rs:644. Source order decides which is
  reported. The signature is visible in the census as a near-even split inside
  one class family (`_io::buffered_random` 9/9, `_io::buffered` 8/9,
  `_io::textio` 13/7).
- Foreign `core::iter` adapters chain among themselves: registering
  `iter::sources::once::once` moves its victims to `Chain::chain`, then to
  `Chain::next`, which is an adapter state machine with no RPython counterpart.
  The cure for that family is source-side (do not write the adapter chain), not
  a registry row.

### What the first graded batch moved, and where the frontier went

Three changes measured together against the 1883-record baseline:
`Constant(None, Void)` for a Void operand, the `SomeChar`/`SomeString` MRO
edge, and dropping the `chain(once(..))` adapter out of `w_method_new`.
Phase A 1883 → 1848; 48 subjects left the fail list, 13 entered it, no graph
that previously cleared phase A stopped clearing it.

The 13 entrants are not regressions. Each appears nowhere in the baseline
census — not as a failure and not as a skip — because its caller blocked
before it could be reached. They fail at a merge (`Method ∪ W_FloatObject` and
kin), which is the wall behind `w_method_new`, one level deeper than the wall
that was closed.

The frontier also moved from phase A to phase B for the first time on this
family. Phase B had **zero** failures in the baseline; it now has 25, and all
25 carry one message: `pair(rtype_add) not implemented for (StringRepr,
CharRepr)`. That is the rtyper-side twin of the annotator MRO edge, closed in
the same commit; it is recorded here because the shape recurs. A rule ported
at the annotator alone will surface its rtyper half as a phase-B wall, and
phase-B counts are not in the phase-A histogram — read both.

### Refuted: lowering `Box::new_uninit` to a zero-arg synthetic ctor

A no-arg `boxed::Box::new_uninit()` (32 records / 14 graphs) looks like it
should lower to `CallTarget::synthetic_transparent_ctor("Box")` with no args,
the spelling `Rvalue::ShallowInitBox` already uses. It should not, as written:

- A zero-arg `SyntheticTransparentCtor` is first read as a unit-variant
  singleton (`flowspace_adapter.rs`, the `args.is_empty()` arm). `"Box"` is not
  on `is_synthetic_unit_variant_path`, so it falls through.
- The general arm then interns by qualname only for the fixed placeholder tags
  `Tuple | Array | Closure`. `synthetic_transparent_ctor` leaves `owner_path`
  empty, so `"Box"` takes the `HostObject::new_class` branch and mints a fresh
  `ClassDesc` **per site** — the exact split the same arm's comment warns about,
  whose symptom is a UnionError at a join with no common base.

Either add `Box` to the interned placeholder set (it is likewise one universal
tag with no enum ambiguity, so the stated justification for the other three
covers it) or give the ctor an owner path. Both change the existing
`ShallowInitBox` lowering, so neither ships without its own census.

### What a base move does to these numbers

Re-taken with the same batch applied on a base 238 files further on, and
`target/llbc` re-extracted for all four crates, phase A reads 1889 and phase B
reads 5. Neither is comparable to the 1848 and 25 above — that is the rule at
the top of this section, applied to itself. What carries over is the
composition.

Phase B's two shapes at that base:

- `pyre_object::interp_exceptions::is_exception` — `don't know how to convert
  from <InstanceRepr Ptr GcStruct pyobject::PyType> to <RootClassRepr Ptr
  Struct object_vtable>`. There is no such conversion upstream to port:
  `convert_from_to` between instance reprs is `pairtype(InstanceRepr,
  InstanceRepr)`, the only pairtype naming a class repr is
  `pairtype(ClassesPBCRepr, ClassRepr)`, and an instance's vtable is reached
  by an *operation* — `InstanceRepr.rtype_type`, which reads the `typeptr`
  field or calls `ll_inst_type`. So the site to change is the one asking the
  rtyper to convert, not the pairtype table.
- Four graphs — `topframe_for_locals` and three
  `__majit_wrap_descr_typecheck_*` getters — all failing on the same op,
  `simple_call(jit_force_virtualizable, frame)`, whose result variable carries
  `SomeNone` and no concretetype, which is what the rtyper's `the annotator
  doesn't agree that 'simple_call' has no return value` reports. The same
  graphs were `MAJIT_RTYPER skip` rows on the earlier base; the base moved the
  force gateways, and the graphs now reach phase B to fail there.

### The one slice the census names: a dead vtable read

Every `Blocked block` record is a `getattr` (189 of them at that base), and 68
name a vtable slot on a receiver whose `SomeInstance` classdef is the empty
`{vtable}` class: `__cast_instance_intrinsic(recv, "{vtable}")`, then
`getattr(.., "method_strategy_kind")`. 58 of the 68 were `IndirectCall`
records before that wall was closed, so this is where most of that wall's
relocated victims went — the victim map the grading rule above asks for, one
slice later.

The cause is on the `OpKind::IndirectCall` arm. It drops the `funcptr`
*operand*, because the `getattr` re-derives it from the receiver — but the ops
that *defined* that operand, the vtable cast and the slot read, are already in
the graph, and the annotator still annotates them, against a class with no
fields. The front end cannot simply stop emitting them: the residual path
still calls through that pointer for real (`codewriter/call.rs`'s
`IndirectCall { graphs: None, funcptr, .. }`, and its `inline.rs` and
`result_exc.rs` peers). So the repair is a dead-op sweep over the flowspace
graph in the prepass, which is a design rather than a registration, and it
needs its own census.
