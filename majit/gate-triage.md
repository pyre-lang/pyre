# majit environment gate triage

This catalog must contain exactly the live `MAJIT_*` environment gates read by workspace Rust and Python sources. `pyre/pyrex/tests/gate_triage_complete.rs` checks both directions.

Each entry records its reader, purpose, and retirement condition. `UNRECORDED` marks information that must not be guessed.

## Live gates

### Translation, backend, and runtime diagnostics

These controls moved with the engine facilities they govern. Unless stated
otherwise they are disabled when unset and can be removed when ordinary tests
cover the condition they diagnose.

| gate | default | purpose and retirement condition |
|---|---|---|
| `MAJIT_CALLEE_CENSUS` | OFF | Reports resolved and unresolved translation callees; remove when every supported callee is classified. |
| `MAJIT_CALLEE_CENSUS_ROWS` | value | Limits rows printed by `MAJIT_CALLEE_CENSUS`; remove with that census. |
| `MAJIT_CALLEE_RCA` | OFF | Reports metainterpreter callee-resolution decisions; remove when those decisions are covered by focused tests. |
| `MAJIT_CL_NO_CLOSING_JUMP` | OFF | Disables Cranelift's in-code closing jump to exercise external jump dispatch; remove when that fallback no longer needs comparison coverage. |
| `MAJIT_CL_SIGN_RETURN` | OFF | Restores the aarch64 return-address signing `cranelift_native` turns on for Apple hosts, which brackets each compiled body with `pacibz`/`retabz`; remove when that pair no longer needs a runtime comparison path. |
| `MAJIT_DESCR_POOL_CENSUS` | OFF | Reports descriptor interning and duplication; remove when descriptor identity is covered by ordinary tests. |
| `MAJIT_DETERMINISM_TRACE` | OFF | Prints inputs used to diagnose nondeterministic translation output; remove when deterministic output is enforced structurally. |
| `MAJIT_DTRACE_CONST_BT` | OFF | Adds a backtrace to constant-propagation diagnostics; remove with those diagnostics. |
| `MAJIT_DTRACE_CONST_FROM` | value | Filters constant-propagation diagnostics by source value; remove with those diagnostics. |
| `MAJIT_DTRACE_CONST_TO` | value | Filters constant-propagation diagnostics by destination value; remove with those diagnostics. |
| `MAJIT_DYNASM_EXEC_DIAG` | OFF | Reports dynasm trace execution; remove when trace entry is covered by ordinary telemetry. |
| `MAJIT_FNPTR_INDIRECT` | OFF | Enables indirect function-pointer lowering; retain while that lowering remains a build configuration. |
| `MAJIT_GC_FREELIST_DIAG` | OFF | Reports GC freelist allocation and reuse; remove when freelist accounting has sufficient invariant tests. |
| `MAJIT_HELPER_FNADDR_SKIP` | OFF | Counts each `#[dont_look_inside]` / `#[elidable]`-family function that got no residual trampoline, by reason (`generic`, `fat-pointer arg`, `Result return`, `method receiver`, `other`); remove when every annotated residual is either published or covered by an ordinary skip test. |
| `MAJIT_GC_ITEMSBLOCK` | ON | Selects GC-managed list item blocks; `0`, `off`, or `false` restores the fallback, which can be removed after deleting the alternate representation. |
| `MAJIT_GC_ROOT_CENSUS` | OFF | Tallies per root walker what one major cycle re-seeded, since `collect_roots` costs one `is_managed_heap_object` probe per root and the root count, not the live-object count, is what a major on a small heap spends that phase on; remove when the off-GC immortal type and module dicts are ordinary GC objects, which is what makes their slots traceable children instead of registered roots. |
| `MAJIT_JTRANSFORM_SHADOW` | OFF | Compares shadow and primary jtransform results; remove after deleting the shadow implementation. |
| `MAJIT_MIR_FRAMESTATE` | ON | Selects framestate-threaded MIR lowering; `0` or `false` restores the older lowering, and the escape hatch retires with that path. |
| `MAJIT_MIR_FRAMESTATE_DEBUG` | OFF | Prints framestate merge diagnostics; remove when merge failures are covered by focused tests. |
| `MAJIT_MIR_FRAMESTATE_STRICT` | OFF | Turns framestate fallback into a hard failure; remove after deleting the fallback. |
| `MAJIT_MIR_FRONTEND_DEBUG` | OFF | Prints MIR frontend lowering diagnostics; remove when unsupported shapes are fully classified. |
| `MAJIT_MIR_FRONTEND_LLBC` | path list | Supplies the complete LLBC input set to the translator; retain as consumer configuration. |
| `MAJIT_MIR_STRESS_LLBC` | path | Supplies the LLBC snapshot for ignored stress tests; retain while those tests are external-fixture tests. |
| `MAJIT_NBODY_DEBUG` | OFF | Reports nested-body tracing decisions; remove when those decisions have focused coverage. |
| `MAJIT_NO_UNROLL` | OFF | Skips the unroll optimizer for diagnosis; remove when the optimizer no longer needs a runtime comparison path. |
| `MAJIT_PORTAL_RCA` | OFF | Reports portal-selection decisions; remove when portal selection is covered by focused tests. |
| `MAJIT_PROBE_SUBSCR` | OFF | Reports subscription tracing and dispatch decisions; remove when the relevant opcode paths have focused coverage. |
| `MAJIT_PROFILE_DRAIN` | OFF | Profiles codewriter queue draining; remove when pipeline performance no longer needs phase attribution. |
| `MAJIT_PROFILE_PIPELINE` | OFF | Reports translation phase time and memory; retain while translation performance needs phase attribution. |
| `MAJIT_ROOT_BRACKET_ERASE` | ON | Keeps a root bracket out of the jitcode when its read-backs are answerable without the shadow stack; `0` or `false` restores the residual bracket, and the escape hatch retires when the erasure covers every bracket shape. |
| `MAJIT_RTYPER_FRONTIER` | OFF | Makes the flowspace adapter resume past an unresolvable call and report the subject's whole call-wall set instead of the first; remove when the census no longer needs sole-blocker ranking. |
| `MAJIT_RTYPER_VERBOSE` | OFF | Emits per-graph rtyper failure census rows; remove when all supported graphs translate or fail through stable classifications. |
| `MAJIT_S9_PROBE` | OFF | Records optimizer stage-nine diagnostics; remove when that stage has focused invariant coverage. |
| `MAJIT_SIZE_SHELL_OWNERS` | OFF | Reports owners of size-descriptor shells; remove when shell identity is enforced structurally. |
| `MAJIT_TRACE_CALL_DIAG` | OFF | Reports calls emitted by the native backend; remove when call lowering has sufficient trace-level coverage. |
| `MAJIT_TRACE_OPS_DIAG` | OFF | Reports operations executed by the native backend; remove when operation lowering has sufficient trace-level coverage. |
| `MAJIT_VABLE_IDX_PROBE` | OFF | Reports whether virtualizable array indices are constant; remove when supported index shapes are covered by ordinary tests. |
| `MAJIT_VABLE_READ_PROBE` | OFF | Prints every exit of the virtualizable ref read, with the vable identity and the reading register, because a `getarrayitem_vable_r` that answers `(opref, None)` panics later in `read_ref_reg` naming none of the four exits that produce it; also names which cause made a vable descriptor lookup decline, since "no vinfo bound" and "index past the declared list" abort the walk identically and call for opposite fixes; remove when the concrete-less exits are separated by focused tests. |

### `MAJIT_BH_DEBUG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `bh_debug_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BH_ENTRY_EXC`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `bh_entry_stale_exc_report()`, called from both single- and multi-frame drivers
- What it does: `MAJIT_BH_ENTRY_EXC`: report a `BH_LAST_EXC_VALUE` that survived into a blackhole drive the caller did not describe as raising.  The clear-before / check-after protocol brackets every call-shaped opcode inside a drive, so the channel cannot go stale within one; what no bracket covers is the window between phases, where a residual raise's leftover value is still in TLS when the next drive starts and a drive entered with `raising_exception == false` seeds `exception_last_value` from its own argument instead.  Clearing the channel on entry is not the fix — `call_jit.rs` deliberately keeps a pending value across its decline path — so this reports the disagreement rather than repairing it.
- Retirement condition: Remove once the inter-phase window is either measured to be unobservable, or closed by a bracket that spans it, so the disagreement has no producer left.

### `MAJIT_BH_NULL_ARG`

- Read sites: 1 — `majit/majit-metainterp/src/blackhole.rs`
- Accessor: `bh_null_arg_report()`
- What it does: `MAJIT_BH_NULL_ARG`: report a null ref argument about to be handed to a residual call, with the jitcode coordinate, before the callee can dereference it.  Some ABIs pass a legitimate null sentinel (e.g. the CallFn `null_or_self` slot), so this reports rather than aborts.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_BAIL`

- Read sites: 1 — `pyre/pyre-jit/src/call_jit.rs`
- Accessor: `bridge_bail_stage()`, consulted at three points of `trace_and_compile_from_bridge()`
- What it does: `MAJIT_BRIDGE_BAIL=<stage>`: return `ResumeBlackhole` from `trace_and_compile_from_bridge` at a chosen point inside the guard-failure entry, so a wrong answer that only appears with bridges on can be attributed to one prefix of what the attempt does.  `MAJIT_NO_BRIDGE` and `MAJIT_MAX_BRIDGES` name WHICH bridge; this names which STEP of it.  1 = before `decode_and_restore_guard_failure`; 2 = after it; 3 = after the frame's `last_instr` is repointed.  Stage 1 is the control and has to reproduce `MAJIT_NO_BRIDGE=1` exactly.  Off by default.
- Retirement condition: Remove when a bridge attempt cannot leave a heap effect behind the rollback it falls into, so a bisect over the attempt's steps has nothing left to find.

### `MAJIT_BRIDGE_DEBUG`

- Read sites: 5 — `majit/majit-macros/src/jit_interp/codegen_state.rs`, `majit/majit-metainterp/src/lib.rs`
- Accessor: `ref_identity_slots_end()`; also read inline in `setup_bridge_sym()` and `bridge_debug_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_DIAG`

- Read sites: 2 — `majit/majit-macros/src/jit_interp/codegen_state.rs`, `majit/majit-metainterp/src/resume_box_reader.rs`
- Accessor: `setup_bridge_sym()`; also read inline in `replay_pending_fields()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_FUEL_LOG`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: read inline in `bridge_fuel_take()`
- What it does: Prints the sequence number of each bridge compilation that consumes bridge fuel.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_CLOSEDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `closedbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_CL_GCSTORE_LOG`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_COVERAGE_AUDIT`

- Read sites: 1 — `majit/majit-translate/src/codewriter/assembler.rs`
- Accessor: `assemble_with_callcontrol()`
- What it does: Lists every SSA variable without a register-allocation color, grouped by graph. This complements the panic-on-first-gap mode.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_COVERAGE_PANIC`

- Read sites: 1 — `majit/majit-translate/src/codewriter/assembler.rs`
- Accessor: read inline in `assemble_with_callcontrol()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DEBUG_DECLARES`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DECLINE_LOG`

- Read sites: 1 — `majit/majit-translate/src/decline.rs`
- Accessor: `level()`
- What it does: Census of the lowering gates' silent declines. Unset, `0`, or empty disables it; any other value counts declines per (gate, reason) and prints runtime reasons; `2` additionally prints one line per decline event. `MAJIT_MIR_FRONTEND_DEBUG` is accepted as an alias at the counter level.
- Retirement condition: retire when the decline counts are no longer needed to steer cel lowering coverage.

### `MAJIT_DIAG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `diag_enabled()`
- What it does: Enables the metainterpreter's diagnostic counters. Bridge-close counters distinguish declined compilation attempts from closes for which compilation was not attempted.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP`

- Read sites: 1 — `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_dump_enabled()`
- What it does: Whether `MAJIT_DUMP` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_BYTECODE`

- Read sites: 1 — `pyre/pyre-jit/src/eval.rs`
- Accessor: `dump_bytecode_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_CLIF`

- Read sites: 4 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`, at all four sites — two for the trace body, two for the host-callable entry wrapper
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_LIVENESS`

- Read sites: 1 — `majit/majit-macros/src/jit_interp/jitcode_lower/liveness.rs`
- Accessor: `maybe_dump_liveness()`
- What it does: Print per-marker live sets to stderr when `MAJIT_DUMP_LIVENESS` is set in the proc-macro build environment. `label` is the lowerer scope being dumped (e.g. helper name) so concurrent expansions are distinguishable.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_SSAREPR`

- Read sites: 1 — `pyre/pyre-jit/src/jit/assembler.rs`
- Accessor: `dump_assembled_ssarepr()`
- What it does: Print the assembled instruction stream, byte position first, for graphs whose name matches `MAJIT_DUMP_SSAREPR`. A blackhole failure reports a raw `(jitcode, position)` pair; without the stream there is no way back from that byte offset to the op that wrote it or to the register operands it reads.  The env lookup is cached because `try_assemble` runs per graph on the tracing path.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_FAILVALS`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `failvals_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_FIELD_MINT_TRACE`

- Read sites: 2 — `majit/majit-ir/src/descr.rs`, `pyre/pyre-jit-trace/build.rs`
- Accessor: `field_mint_trace_enabled()`; the build script also declares it as a rerun input and bypasses its code-generation cache while enabled
- What it does: Setting it to `1` prints field-descriptor mint disagreements (`cache_hit_disagree`, `ei_descr_mint_disagree`) and keeps the analyzer live so those diagnostics cannot be hidden by a restored artifact cache. Unset is inert.
- Retirement condition: Remove when field and size descriptor identity no longer has fallback or disagreement paths to diagnose.

### `MAJIT_FIELD_POS_UNRESOLVED`

- Read sites: 1 — `majit/majit-ir/src/descr.rs`
- Accessor: `field_position_unresolved_limit()`
- What it does: Dumps unresolved field-position rows. A positive integer limits the number of rows; `1` or a non-numeric value dumps the whole table; unset disables the dump.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_BH_PROBE`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `bh_probe_enabled()`
- What it does: Whether the blackhole-object probe is enabled.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_BH_PROBE_CLASSES`

- Read sites: 1 — `majit/majit-gc/src/collector.rs`
- Accessor: `read_uint_from_env("MAJIT_GC_BH_PROBE_CLASSES")`, default 10
- What it does: How many distinct classes the probe above reports.
- Retirement condition: with `MAJIT_GC_BH_PROBE`, whose report it shapes.

### `MAJIT_GC_BH_PROBE_MINOR`

- Read sites: 1 — `majit/majit-gc/src/collector.rs`
- Accessor: `read_uint_from_env("MAJIT_GC_BH_PROBE_MINOR")`, default 120
- What it does: Which minor collection the probe reports at. The interpreter fills the old generation long before the JIT compiles anything, so reporting at minor #1 can only show pre-JIT allocations.
- Retirement condition: with `MAJIT_GC_BH_PROBE`.

### `MAJIT_GC_BH_PROBE_FROM`

- Read sites: 2 — `majit/majit-gc/src/collector.rs`
- Accessor: `read_uint_from_env("MAJIT_GC_BH_PROBE_FROM")`, default 0
- What it does: Skips the minors before the one being investigated. The probe's reachability walk is a whole-heap traversal per minor, too slow to leave on for a whole run.
- Retirement condition: with `MAJIT_GC_BH_PROBE`.

### `MAJIT_GC_DRAIN_CENSUS`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `drain_census_dump_interval()`
- What it does: Set `MAJIT_GC_DRAIN_CENSUS` to a positive integer to also dump the running summary every that many collections. The end-of-run summary is unreachable for the runs this census is most needed on — a collection storm that has to be killed rather than waited out — so those need the periodic line.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_LIFETIME_LOG`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `gc_lifetime_log_enabled()`
- What it does: `MAJIT_GC_LIFETIME_LOG` — trace remembered-set adds and old-gen frees. Read once.  The gate sits in the write barrier and the old-gen sweep, and `std::env::var_os` takes the environment lock and scans it linearly on every call, so asking per event costs whether or not the variable is set.  Same shape as `majit_metainterp::majit_log_enabled`.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_NURSERY_POISON`

- Read sites: 3 — `majit/majit-gc/src/nursery.rs`, `majit/majit-gc/src/oldgen.rs`, `majit/majit-gc/src/shadow_stack.rs`
- Accessor: `gc_nursery_poison_enabled()`
- What it does: fill recycled nursery and old-gen memory with a poison word instead of zeroes, so an allocation path that relies on its memory arriving zeroed fails where it reads rather than later. Two reads initialise `poison_on_reset` (`nursery.rs`) and `poison_on_alloc` (`oldgen.rs`); the third gates the pre-registration poison scan in `push_resume_ref_roots`, which asks per call, so the accessor caches the lookup. The nursery half of this is upstream's `gc_nursery_debug` (`PYPY_GC_NURSERY_DEBUG`), which selects `arena_reset` mode 3; that name is read separately and additively, so either spelling turns the fill on and neither turns the other off.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_STRESS`

- Read sites: 1 — `majit/majit-gc/src/collector.rs`
- Accessor: `env_is_set("MAJIT_GC_STRESS")` in `with_config()`, behind `#[cfg(feature = "gc_stress")]`. Presence, not a parse: empty is on. Falls back to the embedder table (`GC_ENV_NAMES` / `set_supplied_env`) so a wasm32 guest with no process environment still opts in through `pyre_set_gc_env`.
- What it does: with the `gc_stress` feature compiled in, force a full collection at the start of every `alloc_with_type` so a live object that is not reachable from a registered root or custom-trace path is moved or swept on the next allocation. Off by default even when the feature is compiled (`stress_collect` starts false unless the name is set). Test/diagnostic only.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_STRESS_TRACE_ALLOC`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `gc_stress_trace_alloc_enabled()`. Presence, not a parse: empty is on. Read once and cached, for the reason `MAJIT_GC_LIFETIME_LOG` records.
- What it does: off by default. When set, run a minor collection before Trace-pool mallocs (`opencoder.py` `Trace._ops` and the other pools in `trace_bufs`) and before `bh_alloc_struct` in the dynasm runner, so a young address copied into a Rust local and used after a Trace-pool append dies at that use. `malloc_fast` (`framework.py`, `inline=True`) does not take the call: rbigint digit arrays share that entry. Diagnostic only; it is not the `gc_stress` feature path.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_YOUNG_RAWMALLOC`

- Read sites: 2 — `majit/majit-gc/src/collector.rs`, `majit/majit-gc/src/rewrite.rs`
- Accessor: `young_rawmalloc_enabled()`
- What it does: `MAJIT_GC_YOUNG_RAWMALLOC` — whether an oversized allocation from a *collecting* entry point may be born YOUNG and non-moving (incminimark.py `external_malloc(..., alloc_young=True)`) instead of straight into the old generation. On by default, because born-old is the deviation: upstream gives every oversized `malloc_fixedsize`/`malloc_varsize` result `alloc_young=True`, while a born-old block can only be reclaimed by a major. `=0` restores the born-old behaviour without a rebuild, which is what makes a defect that appears only once these objects can die at a minor bisectable against a single binary. Read once and cached, for the reason `MAJIT_GC_LIFETIME_LOG` records.
- Second read site, and why it must exist: `rewrite::GcRewriterImpl::gen_malloc_fixedsize` applies rewrite.py's `remember_write_barrier` to the `malloc_big_fixedsize` result, which is sound only while that helper births young. Turning the gate off births it old, where an elided barrier would lose the first young pointer stored into it — so the same gate must suppress the stamp. The gate being process-wide and read once is what makes the compile-time answer valid for every trace the process compiles; a per-allocation switch could not be consulted here at all.
- Measured reach, so the gate is not read as bisecting more than it can: `MAJIT_GC_LIFETIME_LOG=1` records **zero** `kind=raw-young` births for `bytes`, `bytearray`, `tuple`, `list`, `str`, `dict` and `int` at sizes well past `large_object_threshold`, and for a JIT-compiled loop allocating a 20000-element list 300 times. The oversized populations the host allocates do not reach the collecting entry at all: a list's items block takes `try_gc_alloc` -> `alloc_with_type_no_collect`, an rbigint's digit block takes `alloc_fast_nursery_typed` (`majit-rlib`'s `try_alloc_typed_items_block_nursery`, whose comment records the same no-collect reason), and a `str`'s bytes are a Rust-owned buffer outside the GC heap entirely. The host callers of the collecting entry (`listobject.rs`, `unicodeobject.rs`, `weakref.rs`) all pass a fixed object-header size that is never oversized. Compiled code's `malloc_array`/`malloc_big_fixedsize` is the door that does reach it. `collector.rs::the_no_collect_entry_keeps_its_born_old_arm_when_oversized` pins the split.
- Retirement condition: when a released cycle has shipped with the young path on and no report traces to it, delete the gate and the born-old fallback with it. It exists for the bisection, not as a supported mode. Grade that condition against the reach above -- with no interpreter allocation reaching the young arm, a quiet cycle is not yet evidence, and the gate should outlive the first change that gives it a production population.

### `MAJIT_GUARDLOG`

- Read sites: 2 — `majit/majit-metainterp/src/jitdriver.rs`, `majit/majit-metainterp/src/pyjitpl.rs`
- Accessor: `guardlog_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GUARD_CENSUS`

- Read sites: 2 — `majit/majit-metainterp/src/lib.rs`, `pyre/pyrex/src/lib.rs`
- Accessor: `guard_census_enabled()`; also read inline in `maybe_print_jit_stats()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GUARD_RESUME_REFUSE_PENDING_FIELDS`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `guard_resume_refuse_pending_fields()`
- What it does: Restores `guard_carries_pending_fields`, the refusal to build a guard-resume bridge for a guard whose deferred writes span more than one virtual. Off by default: upstream has no such refusal, and the double-application it once guarded against is answered by placing the decision ahead of `start_bridge_tracing` rather than by declining. Kept so the two answers stay comparable inside one binary, which a second build cannot do.
- Retirement condition: Remove once no consumer needs the refusal restored to attribute a result.

### `MAJIT_HEAPDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `heapdbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_J2PLAN_LOG`

- Read sites: 2 — `majit/majit-backend-dynasm/src/aarch64/assembler.rs`, `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_j2plan_log_enabled()`; also read inline in `_assemble()`
- What it does: Whether `MAJIT_J2PLAN_LOG` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_JITFRAME_POOL`

- Read sites: 1 — `majit/majit-backend/src/deadframe.rs`
- Accessor: `seed_jitframe_pool_arm()`, behind `jitframe_pool_enabled()`
- What it does: Selects which arm allocates the jitframe a compiled entry runs on, for backends that build frames out of the Rust heap rather than the GC nursery. `0` selects the owned arm, one `calloc`/`free` pair per entry; anything else, including leaving it unset, selects the pooled per-thread free list. Read once and latched, so it names a strategy for the process rather than a per-entry state; `set_jitframe_pool` overrides it for a harness that can call in.
- Retirement condition: when the owned arm is retired — it exists to be differenced against the pooled one, and a build with no second arm has nothing to select.

### `MAJIT_PROBE_EXTRA`

- Read sites: 1 — `majit/majit-backend/src/deadframe.rs`
- Accessor: `probe_extra_stage()`, consulted only under `__execute-stage-probe`
- What it does: Names one fixed step of a compiled entry — `guard`, `heap`, `flags`, `attach`, `descr` or `arc` — that the cranelift entry repeats `frame_build_repeats()` times beside the frame build, so the entry probe's frame-build column reads that step's per-entry cost as a delta against a run with the variable unset. Unset or any other value repeats nothing. Read once and latched. Off the probe feature the accessor is never called.
- Retirement condition: with the probe feature — it is an instrument on a feature that ships nothing.

### `MAJIT_LLBC_EXTRACTION`

- Read sites: 2 — `pyre/pyre-jit-trace/build.rs`
- Accessor: `main()` — one `cargo::rerun-if-env-changed=` declaration and one `env::var_os` read, on adjacent lines
- What it does: **UNRECORDED** — no doc comment describes the gate; the one above `main()` describes the build script. Read off the site, not quoted: `=1` calls `emit_llbc_extraction_placeholders()` and returns early, so the extraction does not run.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG`

- Read sites: 19 — `majit/majit-backend-cranelift/src/compiler.rs`, `majit/majit-backend-dynasm/src/lib.rs`, `majit/majit-gc/src/lib.rs`, `majit/majit-gc/src/rewrite.rs`, `majit/majit-ir/src/debug.rs`, `majit/majit-metainterp/src/lib.rs`, `majit/majit-trace/src/logger.rs`
- Accessor: no single accessor — `majit_log_enabled()` is defined once per crate and several sites read the environment inline; relocate with `rg -w MAJIT_LOG <file>`.
- What it does: Whether `MAJIT_LOG` is set, cached at first access.  Mirrors PyPy's `PYPYLOG` env-var check (`rpython/rlib/debug.py:31-38`).
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_JTET`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `log_jtet_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_OPT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `log_opt_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_OPS`

- Read sites: 1 — `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_ops_log_enabled()`
- What it does: Logs backend emission events (`[dynasm] emit/discard/guard`) without enabling per-execution `MAJIT_LOG` diagnostics. The version marker lets census consumers reject an older binary rather than treating a missing option as zero ops.
- Retirement condition: Remove when ordinary tests cover emission-shape checks without an env-gated log.

### `MAJIT_LOG_TEST_CHILD`

- Read sites: 1 — `majit/majit-backend-dynasm/tests/ops_log.rs`
- Accessor: read inline in `log_probe` / `emission_logging_is_independent_and_announces_its_version_once`
- What it does: Marks the child process that `ops_log.rs` respawns so the probe can set `MAJIT_LOG_OPS` / `MAJIT_LOG` without the parent test inheriting them.
- Retirement condition: Remove with the `ops_log` isolation test.

### `MAJIT_MACRO_DEBUG`

- Read sites: 8 — `majit/majit-macros/src/jit_interp/jitcode_lower/dispatch.rs`, `majit/majit-macros/src/jit_interp/jitcode_lower/lower_stmt.rs`
- Accessor: `try_inline_dispatch_arm()`; also read inline in `lower_dispatch_chain()`, `lower_return_stmt()` and `lower_stmt_fallback()`
- What it does: **UNRECORDED** — no doc comment describes the gate. Read off the sites, not quoted: every read guards an `eprintln!` and nothing else. The prose around them documents the lowering decisions being printed, not what the gate is for.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MAX_BRIDGES`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `bridge_fuel_take()`
- What it does: Allows the first N bridge compilations, then suppresses further bridges. Fuel is consumed only after the other `should_bridge` conditions pass.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MINT_INDEX_CENSUS`

- Read sites: 1 — `pyre/pyre-jit-trace/build.rs`
- Accessor: read inline in `real_main()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MPTRACE`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `mptrace_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_NO_BRIDGE`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `no_bridge_enabled()`
- What it does: `MAJIT_NO_BRIDGE`: suppress bridge recording so every guard failure resumes through the blackhole.  Public because a frontend that owns its own guard-failure entry point has to honour it there too — gating only the jitdriver-internal paths leaves the variable set but inert, which reads as "bridges are off" while they keep recording.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_NO_GUARD_RESUME_BRIDGE`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `no_guard_resume_bridge_enabled()`
- What it does: `MAJIT_NO_GUARD_RESUME_BRIDGE`: decline every bridge entry taken directly from a guard failure, so each one resumes through the blackhole and the bridge is grown from the next merge point instead.  Narrower than `MAJIT_NO_BRIDGE`, which suppresses bridge recording outright: this leaves the merge-point-grown bridges in place, so a wrong answer that survives `MAJIT_NO_BRIDGE` and disappears here is attributable to what the walk rebuilds from resume data rather than to bridges as such.  Costly — the blackhole arm reaches the merge point by interpreting — so it is a bisection tool, not a policy.  Off by default.
- Retirement condition: Remove when a bridge entered from a guard failure records the same answer as the blackhole arm reaching the same merge point, so the two arms have nothing left to separate.

### `MAJIT_OPREF_VARIANT_AUDIT`

- Read sites: 1 — `majit/majit-ir/src/opref_audit.rs`
- Accessor: read inline in `resolve_mode()`
- What it does: available only in a `jit-audits` build; `=1` reports and `=abort` panics on the first collision of two `OpRef` variants on one `raw()` key. The ordinary build contains neither the state nor its call sites. In an audit build with the environment gate off, each site performs one thread-local read and returns. State and mode are thread-local so two tests in parallel cannot read each other's collisions or silence one another.
- Retirement condition: Remove when the two `OpRef` namespaces are structurally unable to collide.

### `MAJIT_OPTRACE`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `optrace_enabled()`
- What it does: Per-op trace of `run_to_end`'s dispatch loop (frame depth, pc, raw opcode). Diagnostic for pinpointing the op that faults a hardware-signal crash (SIGBUS/SIGSEGV) which `catch_unwind` cannot capture.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PCSEQ`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `pcseq_enabled()`
- What it does: Prints the control-flow decisions a walk makes: the interpreter pc every `jit_merge_point` visit carries, and every `switch`, `goto_if_not` and `loop_header`, each tagged with the jitcode that owns it and its frame depth (`d=`). A walk that records without ever closing gives the same reading at the top (`seen_loop_header_for_jdindex` stays -1 at every merge point) whether the `loop_header` op was never reached or was reached and declined, and the two call for opposite fixes; the branch lines name which edge diverted the walk, at whatever depth it happened.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PROBE_LIVENESS`

- Read sites: 1 — `pyre/pyre-jit/src/call_jit.rs`
- Accessor: `majit_probe_liveness_enabled()`
- What it does: Whether `MAJIT_PROBE_LIVENESS` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_REGALLOC_DEBUG`

- Read sites: 2 — `majit/majit-translate/src/tool/algo/regalloc.rs`
- Accessor: `std::env::var_os("MAJIT_REGALLOC_DEBUG")` in `perform_register_allocation` and `try_coalesce`
- What it does: prints portal Ref colour occupancy, protected-sharing, and the names that share a merge-point red. Used to see whether a later Scope joined the reserved vm colour. The ordinary build still runs the reservation; this only opens the log.
- Default polarity: **OFF**; unset disables the log.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_REG_WRITE_AUDIT`

- Read sites: 1 — `majit/majit-ir/src/reg_write_audit.rs`
- Accessor: read inline in `resolve()`, cached per thread by `enabled()`
- What it does: available only in a `jit-audits` build; records the source location of the latest write to each integer register, so a later read can report its writer. The ordinary build contains neither the state nor its call sites. Diagnostic state is thread-local to keep parallel traces independent.
- Default polarity: **OFF**; unset, empty, and `0` disable it.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_SKIP_BRIDGES`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `bridge_fuel_skip_set()`, read by `bridge_fuel_skipped()` and by `bridge_fuel_take()`
- What it does: `MAJIT_SKIP_BRIDGES=a,b,...`: decline exactly the listed bridge sequence numbers and take every other one, so a bisect can ask whether one numbered bridge corrupts on its own or whether the corruption is cumulative. `MAJIT_MAX_BRIDGES` cannot answer that, because declining number N also declines everything after it. The two share one sequence — the n-th bridge `should_bridge` otherwise reaches — and either one alone starts the count, so this gate needs no fuel limit beside it.
- Retirement condition: Remove with `MAJIT_MAX_BRIDGES`, once a compiled bridge cannot produce a value the same guard's blackhole resume would not.

### `MAJIT_SMALLIR`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `smallir_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_SPDIAG`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `spdiag_enabled()`
- What it does: Enables stack-position diagnostics on hot back-edge and guard-failure paths. The value is cached to avoid repeated environment reads.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STATS`

- Read sites: 3 — `majit/majit-trace/src/logger.rs`, `pyre/pyre-wasm-runner/src/main.rs`, `pyre/pyrex/src/lib.rs`
- Accessor: `stats_enabled()`; also read inline in `run()` and `maybe_print_jit_stats()`
- What it does: Whether JIT statistics collection is enabled. Checks MAJIT_STATS=1 or MAJIT_LOG=1.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STRICT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `jit_strict_mode()`
- What it does: Strict JIT mode: a non-`InvalidLoop` panic during compilation is a bug and must fail loudly rather than silently degrade to the interpreter and mask the bug behind correct output. Enabled in debug builds (`cargo test`) and whenever `MAJIT_STRICT` is set (release benches / CI); off in plain release so production keeps graceful degradation. Cached like `majit_log_enabled`.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STRUCT_LAYOUT_CENSUS`

- Read sites: 2 — `majit/majit-translate/src/lib.rs`, `pyre/pyre-jit-trace/build.rs`
- Accessor: `struct_layout_census_enabled()`; the build script also declares it as a rerun input and bypasses its code-generation cache while enabled
- What it does: Setting it to `1` reports structure IDs that resolve to multiple spellings or conflicting concrete layouts during translation, as `conflict`, `variant` and `summary` lines. Unset is inert.
- Retirement condition: Remove when one structure ID cannot collect conflicting layouts by construction.

### `MAJIT_TEST_SHELL_REGISTRY_CHILD`

- Read sites: 1 — `majit/majit-metainterp/src/optimizeopt/virtualize.rs`, test-only and native-only
- Accessor: `explicit_sum_inheritance_rejects_a_registered_lookalike()`
- What it does: Marks the isolated child process that runs the fixture replacing the process-global type registry. The parent sets it only for its exact-test child, preventing recursive child spawning without making production registry identity thread-local.
- Default polarity: **OFF**; unset launches the isolated child, any present value runs the fixture in that child. It is not a runtime optimization switch.
- Retirement condition: Remove when this fixture has a dedicated test executable or another process-isolation mechanism that does not need a child marker.

### `MAJIT_TLDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `tldbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_TRACE_ENTRY_CENSUS`

- Read sites: 1 — `majit/majit-backend-wasm/src/lib.rs`
- Accessor: `trace_entry_census_enabled()`; the wasm guest has no environment, so a host arms the same facility through `trace_entry_census_force()`
- What it does: Counts entries into each emitted trace module per resume key, so a steady state can be attributed to the module and dispatch key it re-enters.
- Retirement condition: Remove when the wasm trace-crossing epic closes and per-key entry counts are no longer the way that budget is attributed.

### `MAJIT_VERIFY`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: `majit_verify_enabled()`
- What it does: Whether `MAJIT_VERIFY` is set, cached at first access.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_X2_PROBE`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: `drop()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.
