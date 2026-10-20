# SPEC-DEVIATION: the six tests

Read the current repository `AGENTS.md` first. CPython 3.15 means **3.15t**
(the free-threaded build) throughout this reference. Historical versions,
measurements, and line numbers below are examples from the source skill,
not fresh evidence: read the current pin, rerun applicable oracles, and cite
upstream symbols as well as current navigation lines. For orthodoxy questions,
run the available `pypy3` fixture before source inspection or diagnosis.


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
control-flow order, data structures, storage owner, JIT hints. See `AGENTS.md` "Data structure parity with RPython/PyPy". A structural divergence does not
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
     `lib-python/stdlib-version.txt`, using the free-threaded build, one fresh process per case, reading the
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

This does not authorize adding an absent module. First verify a real `pypy3`
import and an upstream owner in `pypy/`, `rpython/`, or `lib_pypy/`; otherwise
exclude the module unless the user explicitly expands the scope.

For a missing function within a verified PyPy module, if PyPy genuinely has no counterpart, tests 4 and 5 have nothing to evaluate and
**this section is not what licenses the code** — write it in the shape of the
nearest PyPy sibling, not in the shape of the C module, and say which sibling.
Record the exact search and its scope (`rg --no-config -n 'if_nametoindex' pypy/ rpython/`),
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
