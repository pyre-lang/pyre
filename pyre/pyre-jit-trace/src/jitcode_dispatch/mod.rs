//! Trace-side jitcode walker.
//!
//! RPython parity: this is the trace-side counterpart of
//! `BlackholeInterpBuilder.dispatch_loop` (`blackhole.py`). The
//! blackhole loop *executes* each `bhimpl_*` in turn; the tracing-side
//! analogue lives in `pyjitpl.py:opimpl_*` where each opcode becomes
//! a `MetaInterp.execute_and_record` call (RPython
//! `pyjitpl.py`). This module is the sole production tracer:
//! it consumes the codewriter-emitted jitcode bytes directly, executing
//! as it records (`is_authoritative_executor`). The trait-driven
//! `MIFrame::execute_opcode_step` interpret loop is retired.
//!
//! Module layout vs. RPython: because this FBW walker has no single
//! `rpython/jit/metainterp/` file counterpart (the file-for-file parity
//! mirror is the `majit-metainterp` crate — `pyjitpl.rs`, `executor.rs`,
//! `heapcache.rs`, `resume.rs`, `virtualizable.rs`, `blackhole.rs`, ...),
//! the submodules here are a pyre-local navigability split. Each submodule
//! header names the PyPy concept it is the trace-side counterpart of, or
//! declares itself pyre-specific where there is no upstream analogue.
//! Behavioural parity is enforced per opcode arm, not per file.
//!
//! Opcode coverage:
//!
//! | opname              | parity status | behaviour |
//! |---------------------|---------------|-----------|
//! | `live/`             | PARITY        | skip OFFSET_SIZE, continue (RPython tracing does not record `live/` either) |
//! | `goto/L`            | PARITY        | jump to 2-byte LE target, continue |
//! | `catch_exception/L` | PARITY        | skip 2-byte target on normal flow (`pyjitpl.py` records nothing); the target is consumed by `inline_call`'s SubRaise arm via `try_catch_exception_at` (`pyjitpl.py`) |
//! | `switch/id`         | STRUCTURAL ADAPTATION | RPython `opimpl_switch` shape: read int box, lookup `SwitchDictDescr.dict`, emit `GUARD_VALUE` on hit or `INT_EQ` + `GUARD_FALSE` chain on miss. Concrete branch value comes from `TraceCtx::concrete_of_opref`; non-concrete symbolic OpRefs surface `SwitchValueNotConcrete` instead of guessing a branch. |
//! | `ref_return/r`      | PARITY        | top-level: record `Finish(reg) descr=done_with_this_frame_descr_ref` + terminate (`pyjitpl.py:opimpl_ref_return → compile_done_with_this_frame`); sub-walk: surface `SubReturn{Some(value)}` to caller (`pyjitpl.py finishframe`) |
//! | `int_return/i`      | PARITY        | int-bank counterpart of `ref_return/r` — top-level records `Finish(reg) descr=done_with_this_frame_descr_int` (`pyjitpl.py`), sub-walk surfaces `SubReturn{Some(value)}`. RPython `pyjitpl.py opimpl_int_return = _opimpl_any_return`. |
//! | `float_return/f`    | PARITY        | float-bank counterpart — top-level records `Finish(reg) descr=done_with_this_frame_descr_float` (`pyjitpl.py`), sub-walk surfaces `SubReturn{Some(value)}`. RPython `pyjitpl.py opimpl_float_return = _opimpl_any_return`. |
//! | `void_return/`      | PARITY        | void return — top-level records `Finish([]) descr=done_with_this_frame_descr_void` (`pyjitpl.py`, `exits = []` branch), sub-walk surfaces `SubReturn{None}`. RPython `pyjitpl.py opimpl_void_return → finishframe(None)`. |
//! | `inline_call_r_r/dR>r` | PARITY (per-frame catch) | recurses into sub-jitcode via `JitCodeDescr::jitcode_index()`, populates callee `registers_r` (`setup_call_r`, OOR surfaces `InlineCallArityMismatch`), clears `last_exc_value` before writing `SubReturn{value}` into its Ref dst (`pyjitpl.py finishframe`), and scans caller's `op.next_pc` for `live/` + `catch_exception/L` on `SubRaise` (`pyjitpl.py finishframe_exception`). Sub-walk reaching `Terminate` is unexpected (top-level should never fire from a sub-walk); `SubReturn{None}` clears before surfacing `UnexpectedVoidSubReturn`. |
//! | `inline_call_r_i/dR>i` | PARITY        | int-result sibling of `inline_call_r_r/dR>r`. Same recursion, arglist, normal-return clear, and raise routing; only the dst bank changes (`registers_i[dst] = subreturn_value`). RPython `pyjitpl.py _opimpl_inline_call*` is generated through `_opimpl_any_inline_call` decorator that varies on the result type — pyre's walker shares the body via `dispatch_inline_call_dr_kind(dst_bank)`. |
//! | `inline_call_ir_r/dIR>r`, `inline_call_ir_i/dIR>i` | PARITY | extended-arglist siblings — descr + I-list + R-list + dst. RPython `setup_call(argboxes_i, argboxes_r, argboxes_f)` (pyjitpl.py) populates the callee's int + ref banks from the two lists. Walker uses `dispatch_inline_call_dir_kind(dst_bank)` which reads `read_int_var_list` then `read_ref_var_list`, clears the caller exception slot on normal return, and surfaces per-bank arity overflow as `InlineCallIntArityMismatch` / `InlineCallArityMismatch`. |
//! | `inline_call_irf_r/dIRF>r`, `inline_call_irf_f/dIRF>f` | PARITY | full-arglist variants — descr + I-list + R-list + F-list + dst. RPython same `setup_call` distribution; walker uses `dispatch_inline_call_dirf_kind(dst_bank)` extending the dIR helper with `read_float_var_list` + float-bank arg setup and the same normal-return clear. Float arity overflow surfaces `InlineCallFloatArityMismatch`. |
//! | `int_copy/i>i`      | PARITY        | `registers_i[dst] = registers_i[src]` SSA rename, no IR op emitted (`pyjitpl.py _opimpl_any_copy + >i` decorator) |
//! | `ref_copy/r>r`      | PARITY        | Ref-bank sibling — `registers_r[dst] = registers_r[src]` SSA rename, no IR op. Const-source variants (codewriter `emit_ref_copy!` with `ConstRef`) resolve through the constants window of `registers_r` (pre-populated by `setposition` in [`num_regs_r, num_regs_and_consts_r)`). |
//! | `int_<binop>/ii>i`  | PARITY        | int_add/int_sub/int_mul/int_and/int_signext/int_or/int_xor/int_lshift/int_rshift + comparisons int_eq/int_ne/int_lt/int_le/int_gt/int_ge (15 ops). Reads two `i`-coded regs, records `OpCode::Int<Binop>` with `[a, b]`, writes recorder result into dst (`pyjitpl.py`). Mixed shapes such as `int_lshift/ri>i` stay unwired: those are kind-flow kind-flow bugs and must stay unsupported. |
//! | `float_<binop>/ff>f` + `float_neg/f>f` | PARITY | float_add/float_sub/float_truediv binops + float_neg unary (4 ops total — float_mul, float comparisons, float_abs all absent from codewriter today, would land mechanically when emitted). Read on `registers_f` bank, record `OpCode::Float<Binop>`, write dst (`pyjitpl.py`). |
//! | `int_neg/i>i`, `int_invert/i>i` | PARITY | unary i→i ops via `unop_int_record`. RPython `pyjitpl.py` exec-generated unary opimpls. `int_same_as/i>i` has a dormant walker arm for forward-prep, but the generated table should not contain it because RPython `jtransform.py rewrite_op_same_as` removes `same_as` before assembly. |
//! | `cast_int_to_float/i>f` | PARITY | i-bank read, record `CastIntToFloat`, f-bank write. RPython `pyjitpl.py cast_int_to_float` (same exec-generated unary opimpl loop). |
//! | `ptr_eq/rr>i`, `ptr_ne/rr>i` | PARITY | r-bank pair → record PtrEq/PtrNe → i-bank dst via `binop_ref_to_int_record`. RPython `pyjitpl.py` exec-generated comparison opimpls. The `if b1 is b2: return <const>` fast path is wired: `binop_ref_to_int_record` answers an identical operand pair from `fastpath_same_boxes` without recording, as `binop_int_record` does for the int compares. |
//! | `getfield_gc_i/rd>i`, `getfield_gc_r/rd>r` | PARITY (heapcache-aware) | r-bank obj + descr → heapcache lookup. Cache hit returns cached OpRef without recording; cache miss records `OpCode::GetfieldGc<I,R>` + `getfield_now_known` writeback. RPython `pyjitpl.py + 929-950 _opimpl_getfield_gc_any_pureornot`. ConstPtr fast-path (`pyjitpl.py`) deferred — pyre walker doesn't track ConstPtr identity (optimizer's job post-trace). The pyre-specific `id>X` shape (int source — kind-flow kind-flow) stays unsupported. |
//! | `setfield_gc_i/rid`, `setfield_gc_r/rrd` | PARITY (heapcache-aware, alias-clearing) | r-bank box + (i\|r)-bank valuebox + descr. If `getfield_cached(obj,descr) == Some(valuebox)` skip recording (RPython `if upd.currfieldbox is valuebox: return`); otherwise record `OpCode::SetfieldGc(obj, valuebox)` + `setfield_cached` write-through. Aliasing semantics: `CacheEntry.do_write_with_aliasing` (heapcache.py) routes through `_clear_cache_on_write(seen_alloc)` — always wipes `cache_anything`, additionally wipes `cache_seen_allocation` when the write target itself isn't seen-allocated. RPython `pyjitpl.py _opimpl_setfield_gc_any`. The disabled is_unescaped branch (`pyjitpl.py`) is intentionally not ported — RPython itself has it commented out. `iid` / `ird` (int box) shapes stay unsupported (kind-flow territory). |
//! | `getarrayitem_gc_r/rid>r` | PARITY (heapcache-aware) | r-bank array + i-bank index + descr → heapcache `getarrayitem` lookup. Cache hit returns cached OpRef without IR; cache miss records `OpCode::GetarrayitemGcR(array, index)` + `getarrayitem_now_known` writeback. RPython `pyjitpl.py _do_getarrayitem_gc_any`. All three `_i` / `_r` / `_f` result shapes are wired to this same heapcache body (kind-keyed dst bank, dispatch arms below) and registered in `wellknown_bh_insns()` (`insns.rs`) for blackhole execution + codewriter emission. |
//! | `setarrayitem_gc_r/rird`, `setarrayitem_gc_r/rcrd` | PARITY (heapcache-aware) | r-bank array + i-bank index + r-bank value + descr. Always records `OpCode::SetarrayitemGc(array, index, value)` + `heapcache.setarrayitem(...)` write. RPython `pyjitpl.py _opimpl_setarrayitem_gc_any` — no skip-on-redundant short-circuit because `setarrayitem` does aliasing-aware invalidation. The `rcrd` `c`-argcode form (USE_C_FORM `assembler.py/312`) decodes the index as one inline signed byte → ConstInt; same recording body otherwise. `rrid` / `rrrd` / `rrfd` (Ref index) shapes stay unsupported (kind-flow). |
//! | `strlen/r>i`, `strgetitem/ri>i`, `strgetitem/rc>i` | PARITY | `pyjitpl.py opimpl_strlen` / `opimpl_strgetitem` — `self.execute(rop.STRLEN/STRGETITEM, ...)`. Live result stamped via `cpu.bh_strlen` / `bh_strgetitem` so a later `goto_if_not` can take the observed branch. |
//! | `newstr/{i,c}>r`, `strsetitem/r{i,c}{i,c}`, `copystrcontent/rr{i,c}{i,c}{i,c}` | PARITY | `pyjitpl.py opimpl_newstr` / `opimpl_strsetitem` / `opimpl_copystrcontent` — `self.execute(rop.NEWSTR/STRSETITEM/COPYSTRCONTENT, ...)`. `cpu.bh_newstr` allocates the live STR; a store or copy runs when every operand is live and in bounds. |
//! | `residual_call_r_r/iRd>r` | TODO (`direct_assembler_call` + `capture_resumedata` not yet wired) | classifies the call by `EffectInfo`. Wired sub-cases: (1) libffi via [`direct_libffi_call`] — checked first, on `OopSpecIndex::LibffiCall`; parses the `CIF_DESCRIPTION_P` through [`libffi_call_plan`] and records one `GETARRAYITEM_RAW` per argument slot before the `CALL_RELEASE_GIL_*`, declining to the `resbox is None` fallthrough on an unsupported kind; (2) release-gil via [`direct_call_release_gil`] — `CallReleaseGilI` + arglist `[savebox, funcbox] + argboxes[1:]` reshape per `pyjitpl.py`, plus the outer forces-branch `GUARD_NOT_FORCED` (`:2079`) + `GUARD_NO_EXCEPTION` (`:2082`); (3) loop-invariant heapcache via [`loopinvariant_lookup`] / [`loopinvariant_now_known`] per `pyjitpl.py + 2109`; (4) vable IR bookkeeping (`pyjitpl.py`) via [`maybe_walker_vable_and_vrefs_before_residual_call`] — emits FORCE_TOKEN + SETFIELD_GC only; the runtime heap halves of the token protocol (`vinfo.tracing_before_residual_call` / `vrefinfo.tracing_before_residual_call` and the after-call `vinfo.tracing_after_residual_call`, `pyjitpl.py`) are bracketed around the concrete callee execution by [`try_execute_residual_call_via_executor`], which arms TOKEN_TRACING_RESCALL before the call and probe-and-clears it after, surfacing [`DispatchError::VableEscapedDuringResidualCall`] on a detected force (`pyjitpl.py` ABORT_ESCAPE parity). The vref halves of the bracket ARE called by the walker, under the same `is_may_force` gate, and their loops are NOT empty: [`walker_ec_enter`] takes a vref of every seeded callee frame through `TraceCtx::opimpl_virtual_ref` (`ExecutionContext.enter`'s vref), so `virtualref_boxes` carries real pairs. Measured over 431 synth + 93 parity fixtures: 5487 bracket entries, of which 686 (12.5%) saw at least one pair and 66 of 316 emitting fixtures reached a nonzero count, up to 7 pairs. The remaining branches go through [`select_residual_call_opcode`]: `CallMayForce*` + `GuardNotForced` on the rest of the forces-virtual path (`pyjitpl.py`), `CallLoopinvariant*` on `EF_LOOPINVARIANT` (`pyjitpl.py`), `CallPure*` on elidable, otherwise `Call*`. `GuardNoException` follows whenever `effectinfo.check_can_raise(False)` is true (`pyjitpl.py handle_possible_exception`). `heapcache.invalidate_caches_varargs(call_opcode, ei, allboxes)` (`pyjitpl.py + 2659`) is wired around every recorded call op. `OS_NOT_IN_TRACE` is fail-loud-guarded up front via [`do_not_in_trace_call_result`] — `effect_info_for_call_flavor` stub never sets the index today (pyre-jit's `flatten.rs`), making it dead until producers land. Same fail-loud treatment via [`do_jit_force_virtual_guard`] for `OS_JIT_FORCE_VIRTUAL` (stricter-than-PyPy — needs OpRef→concrete-pointer resolver). Still deferred (each blocked on infrastructure absent from pyre-jit-trace): `direct_assembler_call` specialization (`pyjitpl.py` — assembler_call paths route through `inline_call_*/dR>X` instead), KEEPALIVE for vablebox (only fires when `direct_assembler_call` returns a vablebox), and `num_live`-aware `capture_resumedata(after_residual_call=True)` on the guards (`pyjitpl.py → 2586`). |
//! | `residual_call_r_i/iRd>i` | PARITY (kind sibling of `_r_r`) | same EffectInfo classification + guard emission as `_r_r` — `select_residual_call_opcode('i', ...)` returns the int-typed `Call*` family (`CallReleaseGilI` / `CallMayForceI` / `CallLoopinvariantI` / `CallPureI` / `CallI`); only the dst writeback bank (`registers_i`) differs. RPython parity: `pyjitpl.py opimpl_residual_call_r_i = _opimpl_residual_call1`; `do_residual_call`'s `descr.get_normalized_result_type()` dispatch (pyjitpl.py) selects the int-result CALL op. Argboxes pass through [`build_allboxes`] same as `_r_r` (R-list-only argboxes → identity permutation when arg_types is ref-only). |
//! | `residual_call_ir_r/iIRd>r` | PARITY (shape sibling of `_r_r`) | adds an i-bank list between funcptr and the R-list. RPython parity: `pyjitpl.py opimpl_residual_call_ir_r = _opimpl_residual_call2`; `boxes2` argcode (`pyjitpl.py`) decodes the two count-prefixed lists into `argboxes = [i_args..., r_args...]`. Walker passes that flat list through [`build_allboxes`] (line-by-line port of `pyjitpl.py _build_allboxes`) which permutes argboxes by `descr.get_arg_types()` so the recorded `Call*` arglist matches the callee's actual ABI even for mixed orderings like `[REF, INT, REF, INT]`. Same EffectInfo classification + guard emission as `_r_r` via [`select_residual_call_opcode`]. |
//! | `raise/r`           | PARITY (`GUARD_CLASS`) | sets the session's `last_exc_value` (`pyjitpl.py`); top-level records `Finish(exc) descr=exit_frame_with_exception_descr_ref` (`pyjitpl.py compile_exit_frame_with_exception`); sub-walk surfaces `SubRaise{exc}`. Caller-side handler scan (`finishframe_exception`) lives on `inline_call`'s SubRaise arm (above). RPython `pyjitpl.py` also emits `GUARD_CLASS(exc, cls_of_box(exc))` when `heapcache.is_class_known(exc) == false`; the retired trait-side path read `concrete_exc.ob_header.ob_type` from the concrete frame snapshot and emitted the orthodox `GuardClass(exc_box, cls_const)` per the heapcache `is_class_known` gate. |
//! | `reraise/`          | PARITY        | reads `ctx.last_exc_value()` (asserts via `ReraiseWithoutLastExcValue` matching `pyjitpl.py assert`); same dual top-level/sub-walk routing as `raise/r` (`pyjitpl.py popframe + finishframe_exception`). |
//! | `last_exc_value/>r` | PARITY        | reads `ctx.last_exc_value()`, writes the OpRef into `registers_r[dst]` — pure SSA rename, no IR op recorded. RPython `pyjitpl.py opimpl_last_exc_value` returns `self.metainterp.last_exc_box` after asserting `last_exc_value` is non-null; missing slot surfaces `LastExcValueWithoutActiveException` (codewriter invariant: only emits inside `catch_exception/L` body). |
//!
//! Covers: decode walker, `WalkContext { registers_r, trace_ctx }` +
//! `ref_return/r` recording, `goto/L`, `catch_exception/L`,
//! `reraise/`, `int_copy/i>i`, `residual_call_r_r/iRd>r`,
//! `inline_call_r_r/dR>r` recursion, caller-frame `catch_exception`
//! scan, `last_exc_value` field, `reraise` finishframe routing, typed
//! arity / shape / no-active-exception errors, production
//! `PyreJitCodeDescr` adapter.
//!
//! This module replaced the trait dispatch in
//! `MIFrame::execute_opcode_step` (now retired). The entry point is
//! [`walk`], driven from `trace_bytecode` (`trace.rs`) with the
//! appropriate context.
//!
//! Production fidelity gaps (ranked by priority for follow-on work):
//!
//! 1. `residual_call_r_r/iRd>r` `do_residual_call` port
//!    (`pyjitpl.py`). Walker selects the IR opcode via
//!    [`select_residual_call_opcode`] (`CallReleaseGil*` /
//!    `CallMayForce*` / `CallLoopinvariant*` / `CallPure*` / `Call*`),
//!    unconditionally emits `GuardNotForced` on the forces and
//!    release-gil branches, and emits `GuardNoException` whenever
//!    `effectinfo.check_can_raise(False)` is true. Items now wired (was
//!    deferred in earlier audits):
//!    - `vable_and_vrefs_before_residual_call` IR portion (FORCE_TOKEN +
//!      SETFIELD_GC `vable_token_descr`) via
//!      [`walker_vable_and_vrefs_before_residual_call`].
//!    - `direct_call_release_gil` (`pyjitpl.py`) via
//!      [`direct_call_release_gil`].
//!    - `loopinvariant_lookup` / `loopinvariant_now_known`
//!      (`pyjitpl.py + 2109`).
//!    - `heapcache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
//!      (`pyjitpl.py + 2072`) wired around every recorded call op.
//!    - `OS_NOT_IN_TRACE` fail-loud guard via
//!      [`do_not_in_trace_call_result`] (`pyjitpl.py`) —
//!      `effect_info_for_call_flavor` stub never sets the index today
//!      (`flatten.rs::effect_info_for_call_flavor` audit table), making
//!      it dead until the codewriter analyzer trio (annotator/rtyper/
//!      translator) lands.
//!    Items still deferred (each on infrastructure outside walker
//!    scope):
//!    a. **Now walker-bracketed (formerly trait-leg-only)**:
//!       `vable_after_residual_call` (`pyjitpl.py`) observes a runtime
//!       force via a heap-token read. The authoritative walk executes
//!       the callee concretely in
//!       [`try_execute_residual_call_via_executor`], which arms the
//!       token before the call and probe-and-clears it after, surfacing
//!       [`DispatchError::VableEscapedDuringResidualCall`] on a force
//!       (`pyjitpl.py` ABORT_ESCAPE parity).  The vref halves
//!       (`vrefs_before_residual_call` / `vrefs_after_residual_call` →
//!       `stop_tracking_virtualref`) are bracketed at the same two
//!       points, over the vrefs an inlined call's `enter` pushed
//!       (`inline_call.rs::walker_ec_enter`).  They still iterate zero
//!       times for a callee the walker inlines without seeding a frame:
//!       such a level has no frame object to take a `jit.virtual_ref`
//!       of.  Upstream always has one because it inlines the
//!       interpreter's own frame construction as traced code, so the
//!       app-level frame is an ordinary virtual the optimizer keeps —
//!       and `ExecutionContext.enter`
//!       (`executioncontext.py`) takes the vref of THAT.  Not of
//!       the `MIFrame` `perform_call` → `newframe` pushes
//!       (`pyjitpl.py, 1862-1874`), which is the tracer's own
//!       register frame and never carries a vref.
//!    b. **Codewriter-side**: `direct_assembler_call` + KEEPALIVE on
//!       vablebox (`pyjitpl.py + 2080-2081`). Walker's
//!       residual_call dispatchers never receive `assembler_call=True`
//!       — the parallel `inline_call_*/dR>X` opcode family
//!       ([`dispatch_inline_call_dr_kind`]) routes that case.
//!    c. **Cross-leg, not yet implemented**: `_do_jit_force_virtual`
//!       PTR_EQ + GUARD_VALUE prelude (`pyjitpl.py → 2153-2172`).
//!       Walker fail-louds via [`do_jit_force_virtual_guard`]
//!       (stricter-than-PyPy: typed error rather than divergent IR);
//!       full body needs an OpRef → concrete-pointer resolver (Task
//!       #45) before it can return `Some(vref_opref)` /
//!       `Some(standard_opref)` / `None`. Production reach today: 0 —
//!       `OopSpecIndex::JitForceVirtual` is set only by
//!       `jtransform.rs`'s `_handle_jit_call` `jit.force_virtual` arm,
//!       which our benchmarks don't reach. Metainterp orthodox port at
//!       `majit-metainterp`'s `_do_jit_force_virtual` is tests-only.
//!    d. **Implemented**: `direct_libffi_call` (`pyjitpl.py`) — this
//!       module's [`direct_libffi_call`] parses `CIF_DESCRIPTION_P`
//!       through [`libffi_call_plan`] and records the per-argument
//!       `GETARRAYITEM_RAW` reads plus the `CALL_RELEASE_GIL_*`.  It is
//!       reached from the `_cffi_backend` `jit_ffi_call_impl_*` helpers,
//!       whose `libffi_call` oopspec sets `OopSpecIndex::LibffiCall`;
//!       `libffi_call_plan` is a `None`-returning stub under `sandbox`
//!       and `wasm32`, where that module is absent.
//!       `MetaInterp::direct_libffi_call` is the leg that still declines
//!       universally, needing the dynamic calldescr builder pyre lacks,
//!       so it falls through to `direct_call_release_gil` /
//!       `direct_call_may_force`.
//!    e. Guard recording uses `ctx.trace_ctx.record_guard(..., 0)`
//!       followed by `walker_capture_snapshot_for_last_guard`
//!       (`jitcode_dispatch.rs:walker_capture_snapshot_for_last_guard`)
//!       — the walker-side port of `capture_resumedata(
//!       after_residual_call=True)` (`pyjitpl.py`).  Each
//!       residual_call guard (`GuardNotForced`, `GuardNoException`)
//!       carries a single-frame snapshot keyed by
//!       `ctx.outer_jitcode_index` so the optimizer's
//!       `store_final_boxes_in_guard` finds populated resume data.
//!       Active-box narrowing via per-PC liveness is a future
//!       follow-up; today's helper conservatively snapshots every
//!       non-`OpRef::NONE` register across all three banks.
//!    (`_build_allboxes` ABI re-ordering is wired — see
//!    [`build_allboxes`].)
//! 2. `raise/r`'s `GUARD_CLASS` (`pyjitpl.py opimpl_raise`)
//!    is emitted by reading `concrete_exc.ob_header.ob_type`, checking
//!    `heap_cache.is_class_known(exc_box)`, and emits
//!    `GuardClass(exc_box, cls_const)` when needed — the orthodox
//!    `pyjitpl.py` flow.
//! 3. End-to-end portal-closure helper tests (`walk_return_value_helper_*`,
//!    `walk_pop_top_helper_*`) stay `#[ignore]` until handlers exist for
//!    every opname the codewriter-emitted callee bodies use (e.g.
//!    `getfield_vable_i/rd>i`).
//! 4. (External) `build_default_bh_builder_with_unwired_report` is a
//!    transitional helper for kind-flow (6 unwired opnames:
//!    `int_ge/ir>i`, `int_mul/ir>i`, `int_ne/fr>i`, `int_xor/ri>i`,
//!    `setarrayitem_gc_f/rrfd`, `setarrayitem_gc_i/rrid` — kind-flow
//!    bug in assembler emitting mixed-kind operand types). RPython
//!    upstream has no non-strict builder. Removed when kind-flow
//!    closes; not blocking dispatcher work.
//! 5. Concrete-truth-dependent branch opnames (`goto_if_not/iL`,
//!    `goto_if_exception_mismatch/iL`) and the non-constant side of
//!    `switch/id`. RPython
//!    `pyjitpl.py opimpl_goto_if_not`: `switchcase = box.getint()`
//!    branches on the runtime concrete value — `if switchcase: opnum =
//!    GUARD_TRUE; promoted_box = CONST_1` else `opnum = GUARD_FALSE`,
//!    then `metainterp.generate_guard(opnum, box, resumepc=orgpc)`.
//!    `WalkContext` carries `OpRef`s rather than concrete
//!    `box.getint()` values, so these opnames are resolved against the
//!    per-step concrete frame snapshot: the walker reads the runtime
//!    truth and records the corresponding branch guard directly.
//!    Same handling for `goto_if_exception_mismatch/iL`
//!    (`pyjitpl.py` — `last_exc_value`/llexitcase comparison).
//! 6. Class-introspection opname `last_exception/>i`. RPython
//!    `pyjitpl.py opimpl_last_exception`: returns
//!    `ConstInt(ptr2int(rclass.ll_cast_to_object(exc_value).typeptr))` —
//!    the class pointer of the standing exception. Resolving the class
//!    needs `concrete_exc.ob_header.ob_type`, read from the concrete frame
//!    snapshot, the same source used for items 2 and 5.

use crate::jitcode_runtime::{DecodedOp, decode_op_at};
use crate::state::{ConcreteValue, MIFrame, WalkSym};
use majit_ir::{DescrRef, OopSpecIndex, OpCode, OpRef, Type, Value};

/// Descriptor accessor carried by an active MIFrame walk.
///
/// Per-function and test pools remain ordinary slices; the build-time global
/// pool implements the same interface with one memoized entry per index.
pub trait DescrRefTable {
    fn at(&self, index: usize) -> Option<DescrRef>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: AsRef<[DescrRef]> + ?Sized> DescrRefTable for T {
    fn at(&self, index: usize) -> Option<DescrRef> {
        self.as_ref().get(index).cloned()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}
use majit_metainterp::{TraceCtx, VableArrayStore, default_effect_info};

// jitcode_dispatch submodules (extracted from this file). Their `pub`
// items are re-exported so `crate::jitcode_dispatch::` paths stay stable.
// __SUBMODULES__
mod specialize;
pub use specialize::*;
mod inline_call;
pub use inline_call::*;
mod residual_call;

pub fn finalize_published_last_instr(frame: usize, last_instr: isize) {
    residual_call::finalize_published_last_instr(frame, last_instr);
}
pub use residual_call::*;
mod fbw_state;
mod symbolic_fold;
pub use fbw_state::*;
mod resume_snapshot;
pub use resume_snapshot::*;
mod branch;
pub use branch::*;
mod vstack_mirror;
pub use vstack_mirror::*;
mod vable_ops;

mod frame_state;
mod register_bank;
mod register_list;
pub(crate) use frame_state::WalkFrameStateRoot;
pub use frame_state::{WalkFrameState, WalkFrameStateData};
pub use register_bank::RegisterBank;
pub(crate) use register_bank::RegisterBankRoot;
use register_bank::RegisterValues;
pub use register_list::RegisterList;
pub(crate) use register_list::RegisterListRoot;
pub use vable_ops::*;
mod heapcache_ops;
pub use heapcache_ops::*;
mod bridge_subwalk;
pub use bridge_subwalk::*;
mod arith;
pub use arith::*;
mod diag;
pub use diag::*;

/// Body of a callee jitcode that the walker needs to recurse into.
/// RPython parity: when `inline_call_r_r/dR>r` fires, the metainterp
/// reads the descr's `JitCode` body (`pyjitpl.py
/// _opimpl_inline_call*`). Walker consumes the same minimal subset:
/// the bytecode bytes + register-bank sizes for the fresh callee
/// frame.
///
/// Body is always `'static` — production wires the lookup to the
/// per-index runtime cache whose `Arc<JitCode>` entries live inside a
/// leaked cell table; tests
/// either use static byte arrays or `Box::leak` to surface
/// `'static`. Constraining the body's lifetime simplifies
/// `WalkContext`'s lifetime parameters — otherwise the closure's
/// covariance would force register-bank borrows to extend to the
/// lookup's lifetime.
#[derive(Debug, Clone)]
pub struct SubJitCodeBody {
    /// Callee's jitcode bytes (RPython `JitCode.code`).
    pub code: &'static [u8],
    /// Number of Ref-bank registers the callee declares
    /// (`JitCode.num_regs_r`). The walker allocates a fresh
    /// `Vec<OpRef>` of this size for the recursive frame.
    pub num_regs_r: usize,
    /// Number of Int-bank registers (`JitCode.num_regs_i`).
    pub num_regs_i: usize,
    /// Number of Float-bank registers (`JitCode.num_regs_f`).
    pub num_regs_f: usize,
    /// Callee's Int-bank constant pool (`JitCode.constants_i`).
    /// The callee bytecode references constant slots via register
    /// indices `[num_regs_i, num_regs_i + constants_i.len())`;
    /// `setposition` (RPython `pyjitpl.py copy_constants`)
    /// pre-populates those slots with `ConstClass(constants_i[i])`.
    pub constants_i: &'static [i64],
    /// Callee's Ref-bank constant pool (`JitCode.constants_r`). Each slot
    /// holds, behind `ConstSlotR::get`, the erased `PyObjectRef` of a const
    /// object resolved at codewriter time. The slot is interior-mutable
    /// because the GC walker forwards it in place once a collection moves the
    /// object it names, and by then the pool is reachable only through an
    /// `Arc`.
    pub constants_r: &'static [majit_jitcode::codewriter::jitcode::ConstSlotR],
    /// Callee's Float-bank constant pool (`JitCode.constants_f`).
    pub constants_f: &'static [i64],
}

/// Caller-provided sub-jitcode lookup. RPython equivalent: descr
/// resolution within the metainterp loop reads `BhDescr::JitCode {
/// jitcode_index, .. }` and looks up `ALL_JITCODES[idx]`. Walker
/// inverts the dependency: the caller supplies the lookup so the
/// walker stays decoupled from the runtime's all-jitcodes table
/// (production passes the lazy per-index runtime lookup,
/// tests pass synthetic closures over a local fixture map).
pub type SubJitCodeLookup = dyn Fn(usize) -> Option<SubJitCodeBody>;

/// Build a [`SubJitCodeBody`] view over the build-time `ALL_JITCODES[idx]`
/// entry. Returns `None` for an
/// out-of-range index.
///
/// The per-index cell table is `Box::leak`'d at initialization, so the borrowed `code` /
/// `constants_*` slices are `'static` as [`SubJitCodeBody`] requires.
///
/// This is the production sub-jitcode lookup shape — the shadow walker,
/// the per-opcode arm entry, and trace-time list-helper specializations
/// that descend into a charon body (e.g. `w_list_append`) all resolve a
/// callee body through it. RPython parity: a `BhDescr::JitCode {
/// jitcode_index }` operand resolves to `ALL_JITCODES[jitcode_index]`.
pub fn sub_jitcode_body_by_index(idx: usize) -> Option<SubJitCodeBody> {
    crate::jitcode_runtime::get_jitcode_ref_by_index(idx).map(|jc| SubJitCodeBody {
        code: jc.code.as_slice(),
        num_regs_r: jc.num_regs_r(),
        num_regs_i: jc.num_regs_i(),
        num_regs_f: jc.num_regs_f(),
        constants_i: jc.constants_i.as_slice(),
        constants_r: jc.constants_r.as_slice(),
        constants_f: jc.constants_f.as_slice(),
    })
}

/// State the walker reads from / writes to while stepping. RPython
/// equivalent: `MetaInterp` itself — the trace recorder, the symbolic
/// register banks (`registers_i`, `registers_r`, `registers_f`), and
/// the metainterp static data are all reachable from `self` in
/// `pyjitpl.py:opimpl_*`. Pyre passes them via this struct so the
/// walker can be tested without standing up a full `MIFrame`.
///
/// Field roster:
///
/// * `registers_r`: Ref bank for `r`-coded operands.
/// * `registers_i`: Int bank for `i`-coded operands.
///   `registers_f` (Float bank) lands when float opnames join the
///   handler table.
/// * `descr_refs`: descr pool for `d`-coded operands.
///   Mirrors RPython `Assembler.descrs` (`assembler.py`); each
///   2-byte LE descr index in the jitcode bytes resolves through this
///   table.
/// * `trace_ctx`: live trace recorder.
///
/// Register banks are *mutable* — `int_copy/i>i` and
/// `residual_call_r_r/iRd>r` write their dst slot inline (RPython parity:
/// `pyjitpl.py _opimpl_any_copy` returns the box, the
/// `@arguments("box")` + `>X` decorator pair writes it into the result
/// slot; `pyjitpl.py _opimpl_residual_call*` returns the
/// recorder OpRef which the `>X` slot consumes). `inline_call_r_r/dR>r`
/// also *would* write dst (after sub-jitcode recursion) but stays
/// deferred — see the per-handler comments + module-level "Production
/// fidelity gaps" below.
///
/// Raw `BhDescr` pool selector for the `(VableArray, Array)` recognition
/// in vable-array ops (`vable_array_descrs_from_jitcode`).
///
/// RPython `MIFrame.vable_array_index_pair_at` (`blackhole.rs`) reads
/// `self.descrs[idx]` and asserts `isinstance(BhDescr_VableArray)` to
/// recover the array `index`.  Pyre's single per-walk descr table is
/// either the shared global pool (build-time canonical jitcodes
/// resolve through `ALL_DESCRS`) or the
/// per-`CodeObject` body JitCode's own `exec.descrs`
/// (`jitcode/mod.rs`) — runtime per-frame jitcodes have no global
/// allocation index, so they carry their own pool.  This selector keeps
/// the recognition logic byte-identical and only switches the pool
/// source; it is NOT a swap of the adapted [`WalkContext::descr_refs`]
/// (those still resolve `d`-coded operands index-parallel to whichever
/// raw pool is selected).
#[derive(Clone, Copy)]
pub enum RawDescrPool<'a> {
    /// Shared global `ALL_DESCRS` (`jitcode_runtime::all_descrs`).
    /// Build-time canonical jitcodes (e.g. `w_list_append`, inlined by
    /// the full-body walker's specialization sub-walks) and tests use
    /// this — their `d`/`j` operands index the global pool.
    Global,
    /// Per-`CodeObject` body pool (`JitCode.exec.descrs`).  Full-body
    /// walks (the walker-as-tracer path) resolve `d`/`j` operands through
    /// the body JitCode's own pool.
    PerFn(&'a [majit_metainterp::jitcode::RuntimeBhDescr]),
}

impl<'a> RawDescrPool<'a> {
    /// Resolve the raw `BhDescr` at `idx`, mirroring RPython
    /// `self.descrs[idx]`.  `None` for an out-of-range index or a
    /// per-fn slot whose `RuntimeBhDescr` is not an ordinary `Descr`
    /// (a `JitCode` / `Call` / `AssemblerToken` slot — never read as a
    /// vable-array descr operand).
    fn bh_descr_at(self, idx: usize) -> Option<&'a majit_jitcode::jitcode::BhDescr> {
        match self {
            Self::Global => crate::jitcode_runtime::get_descr_by_index(idx),
            Self::PerFn(descrs) => descrs.get(idx).and_then(|d| d.as_bh_descr()),
        }
    }

    fn len(self) -> usize {
        match self {
            Self::Global => crate::jitcode_runtime::descr_table().len(),
            Self::PerFn(descrs) => descrs.len(),
        }
    }

    /// Name of the jitcode an `inline_call` operand's `jitcode_index()`
    /// names.  A per-fn pool numbers its own slots, so the index is an
    /// `ALL_JITCODES` index only under the global pool; reading a per-fn
    /// slot number through `get_jitcode_ref_by_index` names an unrelated
    /// jitcode.
    pub(crate) fn inline_callee_name(self, jitcode_index: usize) -> Option<&'a str> {
        match self {
            Self::Global => crate::jitcode_runtime::get_jitcode_ref_by_index(jitcode_index)
                .map(|jc| jc.name.as_str()),
            Self::PerFn(descrs) => descrs
                .get(jitcode_index)
                .and_then(|descr| descr.as_jitcode())
                .map(|jc| jc.name()),
        }
    }

    /// The callee body `inline_callee_name` names: same index convention,
    /// the `JitCode` itself.  Global slots are `ALL_JITCODES` indices;
    /// a per-fn slot is that pool's `RuntimeBhDescr::JitCode`.
    pub(crate) fn inline_callee_jitcode(
        self,
        jitcode_index: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        match self {
            Self::Global => crate::jitcode_runtime::get_runtime_jitcode_by_index(jitcode_index),
            Self::PerFn(descrs) => descrs
                .get(jitcode_index)
                .and_then(|descr| descr.as_jitcode_owned()),
        }
    }

    fn runtime_jitcode_at(
        self,
        idx: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        match self {
            // A canonical helper's `inline_call` operand is a global-pool
            // descr index.  `BhDescr::JitCode` stores the `all_jitcodes`
            // index (`BhDescr::as_jitcode_index`); answering `None` here
            // aborts the helper descent at that call (`__import__`'s
            // package-fromlist arm) and the outer walk residualizes the
            // whole builtin.
            Self::Global => {
                let descr = crate::jitcode_runtime::get_descr_by_index(idx)?;
                let majit_jitcode::jitcode::BhDescr::JitCode { jitcode_index, .. } = descr else {
                    return None;
                };
                crate::jitcode_runtime::get_runtime_jitcode_by_index(*jitcode_index)
            }
            Self::PerFn(descrs) => descrs.get(idx).and_then(|descr| descr.as_jitcode_owned()),
        }
    }
}

/// A callee local slot's recording-time concrete, tagged with the frame
/// register it was written through. Own-frame reads (`getarrayitem_vable`) and
/// the mid-body live-value recovery only honor an entry whose `frame_reg`
/// matches the frame they resolve against, so a value stored to another frame's
/// same-indexed slot never leaks across a frame identity.
#[derive(Clone, Copy)]
pub struct CalleeLocalConcrete {
    pub frame_reg: u16,
    pub value: majit_ir::Value,
}

/// Per-inline-level callee locals shadow owned by the walking frame
/// (`MIFrame.registers_i/r/f`, `pyjitpl.py`).
///
/// The maps are keyed by `localsplus` slot rather than register color because
/// pyre lowers callee `LOAD_FAST`/`STORE_FAST` to
/// `getarrayitem_vable`/`setarrayitem_vable(frame, slot)`, while the
/// `WalkContext` register banks are post-regalloc color-indexed. At top level
/// the slot concrete comes from the seeded `virtualizable_boxes`; a fresh
/// inlined callee has no seeded frame, so it owns this slot-indexed shadow.
/// The concrete values survive may-force heapcache clears and only specialize
/// recording; runtime guards still re-check them.
pub struct CalleeLocalsShadow {
    /// SSA value held by each fresh-frame local slot. Own-frame vable accesses
    /// fold through this map without emitting GC ops: `fresh_virtualizable`
    /// makes `is_virtualizable_getset` return false
    /// (`rpython/jit/codewriter/jtransform.py`; the frame is marked by
    /// `pypy/interpreter/pycode.py`).
    pub opref: std::collections::HashMap<i64, OpRef>,
    /// Recording-time concrete held by each local slot, including across
    /// may-force operations that clear the heapcache. Each entry records the
    /// frame register it was written through so a slot read resolves against
    /// its own frame only.
    pub concrete: std::collections::HashMap<i64, CalleeLocalConcrete>,
    /// Portal frame register used to resolve own-frame vable operations.
    /// `u16::MAX` means that the strict fresh-frame fold is inactive.
    pub fold_frame_reg: u16,
    /// Concrete `PyFrame*` backing the folded own-frame writes.
    pub concrete_frame: usize,
    /// Box seeded into `fold_frame_reg`; `OpRef::NONE` means the frame register
    /// was not seeded, so the fold must not be disarmed into nonstandard ops.
    pub frame_box: OpRef,
    /// The seed block emitted a real callee `PyFrame` for this inline level, so
    /// the fold is writing away from a frame that has a heap locals array
    /// behind it.  Its LOCAL region is then observable — through a traceback,
    /// `f_locals` or `sys._getframe` — and must not be folded away; the operand
    /// -stack region above `nlocals` is not reachable that way and still folds.
    pub frame_materialized: bool,
    /// `CodeObject` of THIS inline level's frame.
    ///
    /// A `getarrayitem_vable_*` / `setarrayitem_vable_*` slot indexes the frame
    /// the op names, and a Python pc reached inside a sub-walk indexes that
    /// frame's bytecode — both are the callee's, not the outer portal frame the
    /// `fbw_mode.snapshot_sym` describes.  Null when the level did not resolve
    /// a callee code object.
    pub code_ptr: *const pyre_interpreter::CodeObject,
}

impl Default for CalleeLocalsShadow {
    fn default() -> Self {
        Self {
            opref: Default::default(),
            concrete: Default::default(),
            fold_frame_reg: u16::MAX,
            concrete_frame: 0,
            frame_box: OpRef::NONE,
            frame_materialized: false,
            code_ptr: std::ptr::null(),
        }
    }
}

impl CalleeLocalsShadow {
    /// A `NONE` value clears the slot.
    pub(crate) fn set_opref(&mut self, slot: i64, value: OpRef) {
        if value.is_none() {
            self.opref.remove(&slot);
        } else {
            self.opref.insert(slot, value);
        }
    }

    /// A `Void` value clears the slot so a later read does not resurrect a
    /// stale concrete. `frame_reg` is the frame register the write targeted.
    fn set_concrete(&mut self, frame_reg: u16, slot: i64, value: majit_ir::Value) {
        if matches!(value, majit_ir::Value::Void) {
            self.concrete.remove(&slot);
        } else {
            self.concrete
                .insert(slot, CalleeLocalConcrete { frame_reg, value });
        }
    }
}

/// One inlined-callee level of the walk's framestack.
pub struct InlineFrame {
    /// `true` for the portal `MIFrame` at `framestack[0]`.
    /// `pyjitpl.py initialize_state_from_start` / `newframe(mainjitcode)`.
    pub is_portal: bool,
    /// Callee `w_code`, used by the recursion-depth scan. Once the same code
    /// reaches the live `max_unroll_recursion`, the call folds to a residual
    /// instead of unrolling its call tree (`pyjitpl.py`).
    pub w_code: usize,
    /// Whether this frame was entered with the portal greenkey that
    /// `_opimpl_recursive_call` passed to `MetaInterp.perform_call`.
    ///
    /// `resume.rebuild_from_resumedata` reconstructs frames with
    /// `MetaInterp.newframe(jitcode)`, leaving `MIFrame.greenkey` as `None`;
    /// those frames remain live inline levels but do not contribute to the
    /// recursive-unroll count. Fresh forward-inline portal calls carry it.
    pub recursion_greenkey: bool,
    /// `MetaInterp.call_ids[-1]` for this portal frame
    /// (`MetaInterp.newframe`).  `DEBUG_MERGE_POINT` carries the id so two
    /// separate invocations at the same inline depth remain distinguishable.
    /// Transparent helper levels (`w_code == 0`) are not portal frames and
    /// inherit the nearest Python frame's id instead of consuming one.
    pub call_id: u64,
    /// Last Python opcode for which this frame emitted
    /// `DEBUG_MERGE_POINT`. A Python opcode can occupy several disjoint
    /// JitCode emission runs; the marker belongs to the opcode execution, not
    /// to each run.
    pub debug_merge_point_py_pc: Option<u32>,
    /// Paused levels sitting between this callee and the next one out,
    /// OUTERMOST-FIRST; empty preserves the straight-line single-frame
    /// collapse while retaining the callee level.
    ///
    /// Usually one — the caller paused at its CALL.  A constructor adds a
    /// second: `typeobject.py descr_call` runs between the caller and
    /// `__init__`, and upstream that is simply another `MIFrame` on
    /// `metainterp.framestack`, which `capture_resumedata` hands over whole
    /// and `convert_and_run_from_pyjitpl` chains in order.  Holding the chain
    /// per level is what keeps such a level at its own depth however deep the
    /// callee chain below it goes.
    pub parents: Vec<InlineParentFrame>,
    /// [`FBW_EXECUTED_EFFECT_COUNT`] when this level was entered, i.e. at the
    /// CALL its `parent` pauses on.  An abort-point flush that resumes the
    /// caller AT that CALL re-executes the call, discarding everything this
    /// level and its descendants did, so it is only sound while the odometer
    /// has not moved since.
    pub entry_executed_effects: usize,
    /// This level's own `MIFrame.registers_*` while a descendant (a helper
    /// sub-walk) is running. `None` until a child pauses this frame.
    live: Option<LiveFrameRegs>,
}

impl InlineFrame {
    /// Portal frame `pyjitpl.py newframe(mainjitcode)` leaves at
    /// `framestack[0]`.
    pub(crate) fn portal() -> Self {
        Self {
            is_portal: true,
            w_code: 0,
            recursion_greenkey: false,
            call_id: 0,
            debug_merge_point_py_pc: None,
            parents: Vec::new(),
            entry_executed_effects: 0,
            live: None,
        }
    }

    /// Register banks saved when a descendant paused this Python frame.
    pub(crate) fn paused_live(&self) -> Option<LiveFrameRegs> {
        self.live.clone()
    }
}

/// The live register banks of one paused `MIFrame`.
///
/// `MetaInterp.replace_box` writes `framestack` frames in place. The portal
/// is `framestack[0]`; inlined callees follow.
#[derive(Clone)]
pub(crate) struct LiveFrameRegs {
    pub(crate) registers_r: RegisterBank,
    pub(crate) registers_i: RegisterBank,
    pub(crate) registers_f: RegisterBank,
    frame_state: WalkFrameState,
}

impl LiveFrameRegs {
    fn new(
        registers_r: &RegisterBank,
        registers_i: &RegisterBank,
        registers_f: &RegisterBank,
        frame_state: &WalkFrameState,
    ) -> Self {
        Self {
            registers_r: registers_r.clone(),
            registers_i: registers_i.clone(),
            registers_f: registers_f.clone(),
            frame_state: frame_state.clone(),
        }
    }

    pub(crate) fn frame_state(&self) -> &WalkFrameState {
        &self.frame_state
    }

    fn replace_active_box(&self, oldbox: OpRef, newbox: OpRef) {
        let bank = match oldbox.ty() {
            Some(majit_ir::Type::Int) => &self.registers_i,
            Some(majit_ir::Type::Ref) => &self.registers_r,
            Some(majit_ir::Type::Float) => &self.registers_f,
            _ => return,
        };
        bank.replace_active_box(oldbox, newbox);
        self.frame_state.replace_active_box(oldbox, newbox);
    }
}

/// Per-trace-attempt walk session, owned by the walk driver and threaded
/// through [`WalkContext`] — `MetaInterp.framestack` (`pyjitpl.py`,
/// `:2487`; depth scan `:1390`). Innermost level last.
/// Carrier-boundary raise (`finishframe_exception` at the bridge carrier).
///
/// Set on the drain session and copied onto the root walk's new session
/// before [`dispatch_via_miframe`] runs. Lives on [`WalkSession`] rather
/// than TLS because a residual executed while tracing may start another
/// walk on the same OS thread — the same reason
/// [`WalkSession::open_inline_activations`] is per session.
#[derive(Clone, Copy)]
pub(crate) struct CarrierRaiseSeed {
    pub exc: OpRef,
    pub exc_concrete: crate::state::ConcreteValue,
    pub catch_target: Option<usize>,
}

pub struct WalkSession {
    /// Live frame owners waiting to apply `MetaInterp.replace_box` to their
    /// borrowed register banks. Resume snapshots are updated synchronously.

    /// Paused transparent-helper `SubWalkFrame` banks. Helpers are not
    /// Python `MIFrame`s and must not sit on `framestack`.
    pub(crate) helper_live: Vec<LiveFrameRegs>,
    /// The root frame's `is_being_profiled` portal green for this walk.
    pub is_being_profiled: bool,
    /// Inlined callee levels. Parent snapshots are outermost-first, matching
    /// `Snapshot.frames`; a caller is pushed at its inline CALL and popped
    /// when the callee sub-walk returns.
    pub framestack: Vec<InlineFrame>,
    /// `MetaInterp.current_call_id`: the root portal owns id 0 and each
    /// subsequently entered Python portal frame consumes the next id.
    pub next_call_id: u64,
    /// Seeded Python activations currently open in this walk attempt.
    ///
    /// This is the activation-width counterpart of `MetaInterp.framestack`:
    /// a `CALL_ASSEMBLER` that resumes after several inlined entries must
    /// charge every entry it skips. It is per session rather than TLS because
    /// executing a residual while tracing may start another walk on the same
    /// OS thread; that nested MetaInterp has no ownership of the outer walk's
    /// frames.
    pub open_inline_activations: u32,
    /// Root-frame counterpart of [`InlineFrame::debug_merge_point_py_pc`].
    pub root_debug_merge_point_py_pc: Option<u32>,
    /// Whether an abort fired inside an inline sub-walk. Its `op.pc` is then a
    /// callee coordinate with no meaning in an enclosing frame's JitCode, so
    /// neither abort-point flushing nor a later caller-level blackhole latch
    /// may consume it after the sub-walk unwinds.
    pub abort_in_subwalk: bool,
    /// A `DispatchError` propagated out of an inline sub-walk. `stop_pc`
    /// stays in the callee jitcode while the enclosing walk's position is
    /// the call. `convert_and_run_from_pyjitpl` resumes every frame
    /// `aborted_tracing` captured, so a single-frame adopt of the caller
    /// declines into entry replay.
    pub crossed_inline_subwalk: bool,
    /// `DispatchError::TraceTooLong` was raised. A bridge-carrier drain
    /// drops the `Result` (`P2Drain::MiddleDriveErr`, `p2_drain_abort`)
    /// and would otherwise count `AbortReason::Generic`. The raise already
    /// ran `note_root_trace_too_long` (`blackhole_if_trace_too_long`);
    /// the drain still has to return `SwitchToBlackhole(ABORT_TOO_LONG)`.
    pub trace_too_long: bool,
    /// `MetaInterp.last_exc_value` (`pyjitpl.py`): the standing
    /// exception the walk is unwinding, as one slot for the whole walk
    /// rather than one per frame.
    ///
    /// Set by `raise/r` (`pyjitpl.py opimpl_raise`) and by the
    /// residual-call arms that observe a raise (`execute_ll_raised`,
    /// `:2775`); cleared by a normal return before the frame is popped
    /// (`finishframe`, `:2505`) and by the residual success arm
    /// (`clear_exception`, `:2782`). `newframe` never touches it, which is
    /// why it lives here and not on [`WalkContext`]: an inline sub-walk
    /// pushes a frame, and the handler that finally reads the slot may sit
    /// any number of levels above the frame that raised.
    ///
    /// The readers are the handler-body opcodes — `last_exception`,
    /// `last_exc_value/>r`, `reraise/` — which `catch_exception/L` is the
    /// only codewriter route to.
    pub last_exc_value: Option<OpRef>,
    /// Concrete shadow companion to [`WalkSession::last_exc_value`].
    ///
    /// Holds the live `PyObjectRef` of the standing exception so
    /// `last_exc_value/>r` can propagate the concrete into the
    /// destination's `concrete_registers_r` slot, and so a follow-on
    /// `raise/r` reading that destination finds a non-Null concrete and
    /// emits the correct GUARD_CLASS.
    ///
    /// `ConcreteValue::Null` means "no active exception concrete known" —
    /// it matches `last_exc_value == None` for the common case, or means
    /// an adapter caller seeded only the symbolic OpRef without a concrete
    /// (e.g. a synthetic test fixture).
    pub last_exc_value_concrete: ConcreteValue,
    /// Handoff from the carrier drain into the root walk that
    /// [`dispatch_via_miframe`] consumes once. `None` on every walk that
    /// is not that root `finishframe_exception` continuation.
    pub(crate) carrier_raise_seed: Option<CarrierRaiseSeed>,
    /// Blackhole `tmpreg_r`/`tmpreg_i`/`tmpreg_f` (`blackhole.py`):
    /// the single-slot scratch that `insert_renamings` (`flatten.py`)
    /// routes a cyclic parallel move through via `*_push`/`*_pop` pairs.
    /// `ref_push/r` writes the source Ref (+ its concrete shadow) here;
    /// `ref_pop/>r` reads it back into a dst register.  Persisted on the
    /// per-walk session because a push and its matching pop straddle
    /// intervening `*_copy` ops within one trampoline.
    ///
    /// The Ref and Int banks carry their concrete shadow alongside the OpRef,
    /// because those shadows live in the `concrete_registers_{r,i}` side
    /// tables: moving the OpRef alone would leave the destination slot's
    /// shadow describing whatever value the move overwrote.  The Float bank
    /// resolves concretes from the OpRef itself, so it needs no companion.
    pub tmpreg_r: OpRef,
    pub tmpreg_r_concrete: ConcreteValue,
    pub tmpreg_i: OpRef,
    pub tmpreg_i_concrete: ConcreteValue,
    pub tmpreg_f: OpRef,
    /// Concrete root frame and jitcode identity for application traceback
    /// recording. Inlined callees carry their own identity separately in
    /// `InlineCalleeConsts`.
    pub recording_frame_ptr: usize,
    pub recording_jitcode_index: i32,
    /// Last opcode executed by the root recording frame.
    pub recording_opcode_position: usize,
    /// How many `BINARY_OP` / `COMPARE_OP` dunder inlines admitted on the
    /// entry's rewind are currently descending.
    ///
    /// Non-zero means the walk is inside a region whose only exit is the
    /// record-time cut, so no unjournaled commit may run — see
    /// [`fbw_binop_rewind_refuse_commit`].
    ///
    /// It lives on the session for the reason [`WalkSession::last_exc_value`]
    /// does: the entry that raises the region and the residual dispatch that
    /// has to refuse sit any number of inline levels apart, and neither owns
    /// the other's [`WalkContext`]. The session is also the boundary that
    /// makes the state mean what it says — it is created per walk attempt, so
    /// a second walk reaching the same dispatch on the same OS thread starts
    /// outside every region rather than inheriting one.
    pub binop_rewind_depth: usize,
    /// Set when a commit was refused inside the region above, so the entry
    /// that raised it knows the descent stopped for that reason and may take
    /// the cut.
    pub binop_rewind_refused: bool,
    /// The recorder position the region above was raised at.
    ///
    /// A store into an object the region itself allocated is not a commit the
    /// cut has to undo: the cut discards the operations that built the object,
    /// and the concrete object the walker allocated alongside them is
    /// unreachable from anything the re-execution can name — the same reading
    /// [`inline_call::try_walker_inline_type_call`] already states for its own
    /// rewind ("the orphaned instance is unreachable and `__init__` has not
    /// run").  Every other route out of the region — a store into a
    /// pre-existing object, an unjournaled residual — is still refused, so an
    /// object that predates the region cannot escape while it stands.
    ///
    /// Which of the two an object is stays on the box, where the porting rules
    /// ask for it: `new_object` raises `HF_SEEN_ALLOCATION` / `HF_IS_UNESCAPED`
    /// on every allocation the walk records, and `escape_box` and every reset
    /// of the cache lower them, so an object since handed to something outside
    /// the region stops answering on its own. This position is only what tells
    /// an allocation of this region from one the walk made before it:
    /// value-producing ops are numbered by opencoder `_index` (`OpRef::RefOp`
    /// `raw()`), so an `OpRef` at or above this `_index` was minted inside the
    /// region. `_count` also counts voids and is not that sequence. A
    /// rewind taken inside the region cuts to a position the region itself
    /// took and resets the heap cache with it, so a reused number answers as an
    /// allocation again only once something re-allocates there.
    ///
    /// `None` outside a region: set by the outermost
    /// [`fbw_state::BinopRewindInlineGuard`] and cleared when it unwinds, so it
    /// never carries a position from a region that has already ended.
    pub binop_rewind_fresh_from: Option<u32>,
    /// FINISH operand of a top-level portal exit (`N_aryOp._args`).
    ///
    /// The same pair rides [`DispatchOutcome::Terminate`] back to
    /// `full_body_walk_trace`. This field is the copy `walk_session_roots`
    /// forwards when the operand is a `ConstPtr`, so a collection between
    /// the return and the compile `FINISH` does not leave the outcome's
    /// address stale.
    pub finish_payload: Option<(OpRef, majit_ir::Type)>,
}

impl Default for WalkSession {
    fn default() -> Self {
        Self {
            is_being_profiled: false,

            helper_live: Vec::new(),
            framestack: Vec::new(),
            next_call_id: 1,
            open_inline_activations: 0,
            root_debug_merge_point_py_pc: None,
            abort_in_subwalk: false,
            crossed_inline_subwalk: false,
            trace_too_long: false,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            carrier_raise_seed: None,
            tmpreg_r: OpRef::NONE,
            tmpreg_r_concrete: ConcreteValue::Null,
            tmpreg_i: OpRef::NONE,
            tmpreg_i_concrete: ConcreteValue::Null,
            tmpreg_f: OpRef::NONE,
            recording_frame_ptr: 0,
            recording_jitcode_index: -1,
            recording_opcode_position: 0,
            binop_rewind_depth: 0,
            binop_rewind_refused: false,
            binop_rewind_fresh_from: None,
            finish_payload: None,
        }
    }
}

impl WalkSession {
    /// `pyjitpl.py` portal is `framestack[0]`; inlined callees follow.
    pub(crate) fn at_portal(&self) -> bool {
        self.framestack.iter().all(|frame| frame.is_portal)
    }

    /// Inlined-callee depth, excluding the portal at `framestack[0]`.
    pub(crate) fn inline_depth(&self) -> usize {
        self.framestack
            .iter()
            .filter(|frame| !frame.is_portal)
            .count()
    }

    pub(crate) fn last_inline(&self) -> Option<&InlineFrame> {
        self.framestack.iter().rev().find(|frame| !frame.is_portal)
    }

    pub(crate) fn first_inline(&self) -> Option<&InlineFrame> {
        self.framestack.iter().find(|frame| !frame.is_portal)
    }

    /// Claim an abort coordinate for the frame whose `walk()` observed it.
    ///
    /// The first inline sub-walk to see the error owns its `pc` and may build
    /// a multi-frame blackhole image from that coordinate. If that build
    /// declines, enclosing `walk()` calls see the same `DispatchError`, but
    /// its `pc` is still in the innermost JitCode. Returning `false` there
    /// prevents a caller from pairing the callee coordinate with its own
    /// frame — RPython keeps both on the same `MIFrame` throughout unwind.
    pub(crate) fn claim_abort_coordinate(&mut self, inline_subwalk: bool) -> bool {
        let claimed_by_deeper_frame = self.abort_in_subwalk;
        if inline_subwalk {
            self.abort_in_subwalk = true;
        }
        !claimed_by_deeper_frame
    }
}

/// Apply a walk-time concrete `record_application_traceback` with the undo log
/// armed.
///
/// The authoritative walk attaches this iteration's node to the LIVE exception
/// while it records, so a walk that does not commit has left a node the
/// interpreter's replay then records again.  On a freshly raised exception that
/// is invisible — the replay builds a new object and the discarded one is
/// unreachable — but a raise that re-uses one exception object accumulates both
/// deliveries, and the handler reads the whole chain twice over.
/// `fbw_store_journal_rollback` splices the journaled link back out; the commit
/// path keeps it, because a committed walk's node IS this iteration's.
/// `attach` receives the shadow-stack slot of `exc_ptr`. A concrete
/// `record_application_traceback` (`w_pytraceback_new`) collects, and the
/// slot — not the `PyObjectRef` copy — is what the collector rewrites.
/// Re-read that slot after every allocating step inside `attach`.
fn journaled_concrete_traceback_attach(
    exc_ptr: &mut pyre_object::PyObjectRef,
    attach: impl FnOnce(usize),
) {
    let _roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(*exc_ptr);
    let live = pyre_object::gc_roots::shadow_stack_get(slot);
    let previous_head = crate::jitcode_dispatch::fbw_traceback_journal_head(live);
    // The previous head is a nursery traceback node. Pin it too: the attach
    // moves it, and the journal's "head changed" compare needs the forwarded
    // address (`fbw_traceback_journal_push_if_attached`).
    let prev_slot = match previous_head {
        Some(head) if !head.is_null() => {
            let prev_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(head);
            Some(prev_slot)
        }
        _ => None,
    };
    attach(slot);
    let live = pyre_object::gc_roots::shadow_stack_get(slot);
    let previous_head = match previous_head {
        Some(_) => Some(match prev_slot {
            Some(prev_slot) => pyre_object::gc_roots::shadow_stack_get(prev_slot),
            None => pyre_object::PY_NULL,
        }),
        None => None,
    };
    crate::jitcode_dispatch::fbw_traceback_journal_push_if_attached(live, previous_head);
    *exc_ptr = pyre_object::gc_roots::shadow_stack_get(slot);
}

fn note_forwarded_exc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    exc_concrete: &mut ConcreteValue,
    old: pyre_object::PyObjectRef,
    live: pyre_object::PyObjectRef,
) {
    *exc_concrete = ConcreteValue::Ref(live);
    // `WalkSession::last_exc_value_concrete` is the copy `SubRaise` was
    // built from. It is not an `ExceptionRoot` (`TraceRoots` walks
    // `PyreSym::last_exc_value` / `current_exc_value` only).
    if ctx.last_exc_value_concrete() == ConcreteValue::Ref(old) {
        ctx.set_last_exc_value_concrete(ConcreteValue::Ref(live));
    }
}

/// Record the implicit `__context__` of an exception this trace catches itself.
///
/// `error.py record_context` runs inside the interpreter's exception
/// dispatch.  A raise whose handler is part of the trace routes straight to
/// that handler, so no interpreter dispatch ever sees the error and the chain
/// is simply never derived — the same gap the traceback recorders above close,
/// and the reason a hot loop raising under a live `except` reads
/// `__context__ is None` on exactly its compiled iterations.
///
/// The store is not journaled the way the traceback node is.  It only fills a
/// `__context__` the resolver reported as still unset, and an aborted walk is
/// replayed by the interpreter, which derives the context from the same
/// execution context and so writes the same object; a leftover store is
/// therefore the answer the replay reaches anyway.
///
/// The hook performs the chaining and *answers* the resulting value, and the
/// trace stores that answer itself.  The store cannot be left to the hook
/// alone: `make_call_descr_with_effect` rejects a non-trivial raw descr set
/// minted after `compute_bitstrings` (`effectinfo.py`), so a call
/// cannot name `w_context` as its write set here, and the only alternative
/// that admits an unnamed write to it is `default_effect_info`
/// (`EffectInfo::MOST_GENERAL`) — "can raise, can force, writes anything",
/// which costs a `GuardNoException`, a `GuardNotForced` and a full heapcache
/// flush at every raise.  `cannot_raise_effect_info()` covers what the hook
/// really does: it cannot raise, and it is a heap mutator because breaking a
/// context cycle nulls the `__context__` of an exception other than this one.
/// The explicit `SetfieldGc` then gives the optimizer the one write it can
/// model.  `can_collect` is cleared on that effect: `chain_context` only
/// stores pointers, `w_exception_set_context` takes `exception_write_barrier`,
/// and `get_sys_exception` reads `getexecutioncontext`.  None of those collect,
/// so the call is not a safepoint.
fn record_inline_exception_context(ctx: &mut TraceCtx, exc: OpRef, exc_concrete: ConcreteValue) {
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return;
    };
    if exc_ptr.is_null() || !unsafe { pyre_object::is_exception(exc_ptr) } {
        return;
    }
    // An exception a raise lowering already chained is not this compensation's
    // to touch: that lowering emits `exc.w_context = ec.sys_exc_value` as IR on
    // the still-virtual object and applies the same write to its concrete, so
    // the store costs nothing and folds away with the exception when the
    // handler never looks at it.  Stamping it again here would duplicate the
    // store and, worse, hand the exception to a call, which forces the
    // allocation the optimizer had removed.  `raise_catch_loop.py` and
    // `synth/raise_bare_class_slot_kinds.py` both catch without touching the
    // object, so that forcing alone doubled their execution time.
    if fbw_context_chained_contains(exc) {
        return;
    }
    // Do not consult recording-time `chain_context_would_write` here.
    // `handle_bytecode` / `handle_exception_with_context` already ran
    // `OperationError.record_context` on a residual or inlined callee's
    // concrete exception, so that predicate sees a stamped slot and would
    // drop the residual the compiled iterations still need: a jitted callee
    // raises without going through that interpreter dispatch, and this
    // catch-side store is the only write.  A same-frame bare reraise
    // (`RAISE_VARARGS 0` / `RERAISE`, `RaiseWithExplicitTraceback`) is
    // skipped at the `SubRaise` catch site instead, matching
    // `pyopcode.py handle_bytecode` routing that exception around
    // `record_context`.
    // The hook chains through `chain_context`, so calling it here both applies
    // the effect to this authoritative walk's concrete exception and reaches
    // the value the compiled iterations will store.
    majit_metainterp::resolve_exception_context_for_recording(exc_ptr as usize as i64);
    let hook = majit_metainterp::resolve_exception_context_hook_address();
    if !hook.is_null() && !exc.is_none() {
        // `w_context` sits at one offset for every kind, so the recording
        // iteration's kind names the right bytes even if a later one differs;
        // the descr must still name the layout `allocate_instance` stamped,
        // because a realbase and its `_getusercls` do not share a SizeDescr.
        let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc_ptr) };
        let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc_ptr) };
        let context_descr = crate::descr::w_exception_context_descr_for(kind, user);
        let mut effect = majit_metainterp::cannot_raise_effect_info();
        effect.can_collect = false;
        let w_context = ctx.call_ref_typed_with_effect(hook, &[exc], &[Type::Ref], effect);
        let context_idx = context_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[exc, w_context], context_descr);
        ctx.heapcache_setfield_cached(exc, context_idx, w_context);
    }
}

fn record_top_level_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: &mut ConcreteValue,
    opcode_position: usize,
    execute_concrete: bool,
    emit_runtime: bool,
    node: Option<OpRef>,
) {
    if !ctx.is_top_level {
        return;
    }
    let ConcreteValue::Ref(mut exc_ptr) = *exc_concrete else {
        return;
    };
    if exc_ptr.is_null() {
        return;
    }
    let (frame_ptr, jitcode_index) = {
        let session = ctx.session.borrow();
        (session.recording_frame_ptr, session.recording_jitcode_index)
    };
    if execute_concrete {
        // Attaching the live frame to a traceback escapes the virtualizable, so
        // publish the walk's locals first.  `virtualizable.py write_boxes`
        // makes that write unconditional, and `pyopcode.py` performs it
        // before attaching the application traceback.  Both that flush and
        // `w_pytraceback_new` collect; the exception word stays on the
        // shadow stack `journaled_concrete_traceback_attach` publishes.
        let old = exc_ptr;
        journaled_concrete_traceback_attach(&mut exc_ptr, |slot| {
            crate::state::flush_locals_region_to_frame(ctx.trace_ctx, frame_ptr);
            let live = pyre_object::gc_roots::shadow_stack_get(slot);
            majit_metainterp::record_application_traceback_for_recording(
                live as usize as i64,
                frame_ptr as i64,
                jitcode_index,
                opcode_position as i32,
            );
        });
        note_forwarded_exc(ctx, exc_concrete, old, exc_ptr);
        stamp_traceback_node_concrete(
            ctx,
            node,
            exc_ptr,
            frame_ptr as *mut pyre_interpreter::PyFrame,
        );
    }
    let hook = majit_metainterp::record_application_traceback_hook_address();
    let frame = crate::state::pyjitcode_for_jitcode_index(jitcode_index)
        .and_then(|jitcode| {
            ctx.registers_r
                .get(jitcode.metadata.portal_frame_reg as usize)
        })
        .unwrap_or(OpRef::NONE);
    if emit_runtime && !hook.is_null() && !exc.is_none() && !frame.is_none() {
        let jitcode = ctx.trace_ctx.const_int(i64::from(jitcode_index));
        let opcode = ctx.trace_ctx.const_int(opcode_position as i64);
        ctx.trace_ctx.call_void_typed_with_effect(
            hook,
            &[exc, frame, jitcode, opcode],
            &[Type::Ref, Type::Ref, Type::Int, Type::Int],
            default_effect_info(),
        );
    }
}

/// One node per inlined level the exception-edge bridge resumed PAST, for the
/// frames `set_exc_edge_discarded_levels` parked before the walk began.
///
/// `pyopcode.py pytraceback.record_application_traceback` runs before the
/// `:152` exception-table lookup, so a frame the unwind only PASSES THROUGH
/// contributes a node just like the one that catches it.  Upstream never has to
/// say so: it resumes onto a rebuilt MIFrame stack and the unwind is traced code
/// that runs each level's own recorder.  The exc-edge route exists precisely
/// because pyre discards those levels instead, which leaves this the only place
/// their nodes can come from.
///
/// Innermost-first, because `record_application_traceback` PREPENDS: the deepest
/// discarded frame must go on while the node the residual raise already attached
/// is still the head.  The caller then records the catching frame last, so the
/// chain reads outermost-first exactly as the interpreter builds it.
///
/// Emitted as IR, not merely executed: `trace_and_compile_from_bridge` runs once,
/// and every later guard failure of this class enters the compiled bridge
/// directly.
fn record_exc_edge_discarded_tracebacks<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: &mut ConcreteValue,
) {
    let levels = take_exc_edge_discarded_levels();
    if levels.is_empty() {
        return;
    }
    let ConcreteValue::Ref(mut exc_ptr) = *exc_concrete else {
        return;
    };
    if exc_ptr.is_null() {
        return;
    }
    let hook = majit_metainterp::record_discarded_level_traceback_hook_address();
    for &(w_code, py_pc) in levels.iter().rev() {
        let old = exc_ptr;
        journaled_concrete_traceback_attach(&mut exc_ptr, |slot| {
            let live = pyre_object::gc_roots::shadow_stack_get(slot);
            majit_metainterp::record_discarded_level_traceback_for_recording(
                live as usize as i64,
                w_code as i64,
                py_pc as i64,
            );
        });
        note_forwarded_exc(ctx, exc_concrete, old, exc_ptr);
        if hook.is_null() || exc.is_none() {
            continue;
        }
        let w_code_op = ctx.trace_ctx.const_ref(w_code as i64);
        let py_pc_op = ctx.trace_ctx.const_int(py_pc as i64);
        ctx.trace_ctx.call_void_typed_with_effect(
            hook,
            &[exc, w_code_op, py_pc_op],
            &[Type::Ref, Type::Ref, Type::Int],
            default_effect_info(),
        );
    }
}

/// Give the node box its object.  `execute_new_with_vtable` is record plus
/// allocate in one step (history.py `RefFrontendOp`: the box holds the
/// object), and pyre splits the two: the recorded `NewWithVtable` above, the
/// allocation in the concrete attach.  The head the attach leaves on the
/// exception is that object, provided it names the attaching frame (the
/// attach dedups against a node already there).
fn stamp_traceback_node_concrete<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    node: Option<OpRef>,
    exc_ptr: pyre_object::PyObjectRef,
    frame_ptr: *mut pyre_interpreter::PyFrame,
) {
    let Some(node) = node else {
        return;
    };
    if exc_ptr.is_null() || frame_ptr.is_null() {
        return;
    }
    let head = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc_ptr) };
    if head.is_null()
        || unsafe { !pyre_interpreter::pytraceback::is_pytraceback(head) }
        || unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(head) } != frame_ptr
    {
        return;
    }
    ctx.trace_ctx
        .set_opref_concrete(node, Value::Ref(majit_ir::GcRef(head as usize)));
}

fn record_inline_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: &mut ConcreteValue,
    opcode_position: usize,
    execute_concrete: bool,
    emit_runtime: bool,
    node: Option<OpRef>,
) {
    if ctx.is_top_level {
        return;
    }
    let Some(consts) = ctx.inline_callee_consts else {
        return;
    };
    // A non-standard virtualizable frame in a bridge sub-walk carries the
    // `GcRef(usize::MAX)` sentinel (or null) as `w_code` instead of a real
    // `PyCode` — a synthetic frame with no Python code to anchor a node on.
    // The host adapter (`record_inline_traceback_for_recording`) dereferences
    // `w_code` through `createframe_obj`, so a sentinel / garbage pointer would
    // SIGSEGV.  Skip null / sentinel / non-code; the null + sentinel checks run
    // before `is_code`, whose `py_type_check` would deref the raw sentinel
    // (`CAN_BE_TAGGED` is off).
    let w_code = ctx.inline_w_code();
    if w_code == 0 || w_code == usize::MAX {
        return;
    }
    if !unsafe { pyre_interpreter::pycode::is_code(w_code as pyre_object::PyObjectRef) } {
        return;
    }
    let ConcreteValue::Ref(mut exc_ptr) = *exc_concrete else {
        return;
    };
    if exc_ptr.is_null() {
        return;
    }
    if execute_concrete {
        // `pytraceback.py record_application_traceback(space, operror,
        // frame, last_instruction)` anchors the node on the frame that is
        // executing. Every admitted inline level has one: the sub-walk runs
        // the callee on it, and the emitted node names the very same
        // object, since [`traceback_node_site`] resolves its frame operand from
        // this register and the seed stamps the concrete frame onto that
        // operand's box (`inline_call.rs`).
        //
        let node_frame = crate::py_coord::containing_py_pc_for_jitcode_pc_public(
            consts.jitcode_index,
            opcode_position as i32,
        )
        .map(|py_pc| {
            let frame_reg = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)
                .expect("an inlined level must have JitCode metadata")
                .metadata
                .portal_frame_reg;
            let frame = concrete_portal_frame(ctx, consts.jitcode_index)
                .expect("an inlined MIFrame must carry its own concrete PyFrame");
            (frame, py_pc, frame_reg)
        });
        let old = exc_ptr;
        journaled_concrete_traceback_attach(&mut exc_ptr, |slot| {
            if let Some((frame_ptr, _py_pc, frame_reg)) = node_frame {
                // `dispatch_bytecode` publishes the executing instruction through
                // the frame field before handling the exception.  Keep all three
                // views in step here as well: emitted IR, the recording-time frame,
                // and heapcache knowledge used by a following `f_lasti` read.
                // Attaching the live frame to a traceback escapes the
                // virtualizable, so publish this callee's walk-time locals
                // first.  `virtualizable.py write_boxes` makes that
                // write unconditional before `pyopcode.py` attaches the
                // application traceback.  The flush boxes locals and can
                // collect; the exception stays in `slot` and the frame
                // stays on this bracket.
                let _frame_roots = pyre_object::gc_roots::push_roots();
                let frame_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(frame_ptr as pyre_object::PyObjectRef);
                residual_call::record_and_publish_inline_callee_last_instr(ctx, opcode_position);
                flush_callee_locals_region_to_frame(
                    ctx,
                    pyre_object::gc_roots::shadow_stack_get(frame_slot)
                        as *mut pyre_interpreter::PyFrame,
                    frame_reg,
                );
                let frame_ptr = pyre_object::gc_roots::shadow_stack_get(frame_slot)
                    as *mut pyre_interpreter::PyFrame;
                let live = pyre_object::gc_roots::shadow_stack_get(slot);
                majit_metainterp::record_application_traceback_for_recording(
                    live as usize as i64,
                    frame_ptr as i64,
                    consts.jitcode_index,
                    opcode_position as i32,
                )
            }
        });
        note_forwarded_exc(ctx, exc_concrete, old, exc_ptr);
        if let Some((frame_ptr, _py_pc, _frame_reg)) = node_frame {
            stamp_traceback_node_concrete(ctx, node, exc_ptr, frame_ptr);
        }
    }
    let frame = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)
        .and_then(|jitcode| {
            ctx.registers_r
                .get(jitcode.metadata.portal_frame_reg as usize)
        })
        .unwrap_or(OpRef::NONE);
    let frame_hook = majit_metainterp::record_application_traceback_hook_address();
    if emit_runtime && !frame_hook.is_null() && !exc.is_none() {
        assert!(
            !frame.is_none(),
            "an inlined MIFrame must carry its own red PyFrame"
        );
        let jitcode = ctx.trace_ctx.const_int(i64::from(consts.jitcode_index));
        let opcode = ctx.trace_ctx.const_int(opcode_position as i64);
        ctx.trace_ctx.call_void_typed_with_effect(
            frame_hook,
            &[exc, frame, jitcode, opcode],
            &[Type::Ref, Type::Ref, Type::Int, Type::Int],
            default_effect_info(),
        );
    }
}

/// The concrete `PyFrame` this level's portal frame register holds, or `None`
/// when the level is not a materialized inline MIFrame.
fn concrete_portal_frame<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode_index: i32,
) -> Option<*mut pyre_interpreter::PyFrame> {
    let jitcode = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    let ConcreteValue::Ref(frame) = ctx
        .frame_state
        .borrow()
        .concrete_registers_r
        .get(jitcode.metadata.portal_frame_reg as usize)
        .copied()?
    else {
        return None;
    };
    (!frame.is_null()).then_some(frame as *mut pyre_interpreter::PyFrame)
}

/// Whether `shadow` can reproduce every locals slot of a frame reached through
/// `frame_reg`.
///
/// `virtualizable.py write_boxes` writes every locals-array element
/// because `virtualizable_boxes` is total. This shadow is sparse instead, so an
/// absent slot carries two meanings and only one of them may be skipped:
///
/// * absent from BOTH maps — this walk never wrote the slot, so the frame still
///   holds the value `write_boxes` would have written back.
/// * present in `opref` but not in `concrete` — the walk DID write the slot with
///   no concrete half ([`CalleeLocalsShadow::set_concrete`] drops a
///   `Value::Void`), so the frame's value is stale and unreconstructable.
///
/// `Value::Void` and `GcRef::NO_CONCRETE` are the "no concrete half" sentinels,
/// not a NULL local: writing one back would store `PY_NULL` or an untraceable
/// word over the slot the walk holds in a register. Only entries recorded
/// through `frame_reg` describe this frame.
fn callee_locals_region_is_publishable(
    shadow: &CalleeLocalsShadow,
    nlocals: usize,
    frame_reg: u16,
) -> bool {
    (0..nlocals as i64).all(|slot| match shadow.concrete.get(&slot) {
        Some(concrete) => {
            concrete.frame_reg == frame_reg
                && !matches!(concrete.value, Value::Void)
                && !matches!(concrete.value, Value::Ref(r) if r == majit_ir::GcRef::NO_CONCRETE)
        }
        None => !shadow.opref.contains_key(&slot),
    })
}

/// [`crate::state::flush_locals_region_to_frame`]'s twin for an inlined level.
///
/// The top-level frame's slots come from the seeded `virtualizable_boxes`; a
/// fresh inlined callee has no seeded frame and owns [`CalleeLocalsShadow`]
/// instead, so the escape write reads the locals region out of that shadow.
/// `frame_reg` is the level's portal frame register, and only entries written
/// through it belong to `frame`.
fn flush_callee_locals_region_to_frame<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    frame: *mut pyre_interpreter::PyFrame,
    frame_reg: u16,
) -> bool {
    flush_callee_locals_region(&ctx.frame_state, frame, frame_reg)
}

/// [`flush_callee_locals_region_to_frame`] against a shadow reached without a
/// `WalkContext`.  The escape force runs from `force_pyframe`, which only has
/// the `TraceCtx`, so the level's shadow is published for the duration of the
/// residual instead (`residual_call.rs`).
pub(crate) fn flush_callee_locals_region(
    state: &WalkFrameState,
    frame: *mut pyre_interpreter::PyFrame,
    frame_reg: u16,
) -> bool {
    if frame.is_null() {
        return false;
    }
    let Some(nlocals) = crate::state::concrete_nlocals(frame as usize) else {
        return false;
    };
    // Validation pass first: it allocates nothing, so a decline leaves the
    // frame untouched (same all-or-nothing discipline as the top-level twin).
    let frame_ptr = frame as *const u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < nlocals {
        return false;
    }
    // `virtualizable.py write_boxes` keeps the source boxes and destination
    // array live throughout writeback. Native boxing can collect: retain roots
    // for both destinations and read each source from its shared owner after
    // the preceding allocation, with no frame-state borrow across boxing.
    let frame_root = majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(frame as usize));
    let array_root = majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(arr_ptr as usize));
    state.write_callee_locals(nlocals, frame_reg, |abs, value| {
        let boxed = crate::state::boxed_slot_value_for_type(Type::Ref, &value);
        let arr_ptr = array_root.get().0 as *mut pyre_object::FixedObjectArray;
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = boxed;
        }
        // Boxing an Int/Float slot allocates, and each minor collection
        // consumes the array's remembered-set entry, so re-arm per store.
        crate::state::frame_array_write_barrier(frame_root.get().0 as *mut u8, arr_ptr);
    })
}

/// Apply `PyFrame.frame_finished_execution = True` on the concrete frame a
/// `*_return` just finished.
///
/// `pyopcode.py` `RETURN_VALUE` stores the bit before returning, and the
/// codewriter emits that read-or-write of `PyFrame.flags` immediately before
/// `*_return`. The walker records it by executing that jitcode. The recorded
/// `setfield_gc` stays record-only for a frame the walk did not allocate, and
/// a no-replay portal exit never re-executes the trace, so the concrete bit
/// still has to land on that pre-existing frame here. `YIELD_VALUE` does not
/// reach this function: its frame exit is the `abort_permanent` arm, which
/// finishes the trace without the store. `generator.py` `send_ex` reads the
/// bit to tell the two apart.
fn finish_current_frame_execution<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) {
    // `dispatch_bytecode` publishes `last_instr` before every opcode.  The
    // walker bypasses that loop, so an escaped inline MIFrame needs the return
    // coordinate explicitly; otherwise a frame returned by
    // `sys._getframe(0)` remains at its constructor sentinel after the callee
    // has finished.  The helper is a no-op for the top-level frame, whose
    // finish path uses `fbw_publish_exit_last_instr` below.
    if !ctx.is_top_level {
        residual_call::record_and_publish_inline_callee_last_instr(ctx, opcode_position);
    }
    let concrete_frame = if ctx.is_top_level {
        // `fbw_store_token_in_vable` captures a guard snapshot and that
        // capture returns `Err`. Setting the bit before the `?` would leave
        // it on a frame whose walk was declined. `commit_top_level_frame_finished`
        // runs only after `fbw_terminate_with_finish` returns `Ok`.
        return;
    } else {
        let jitcode_index = ctx
            .inline_callee_consts
            .map_or(-1, |consts| consts.jitcode_index);
        let mut concrete = std::ptr::null_mut();
        if let Some(jitcode) = crate::state::pyjitcode_for_jitcode_index(jitcode_index) {
            let frame_reg = jitcode.metadata.portal_frame_reg as usize;
            concrete = match ctx
                .frame_state
                .borrow()
                .concrete_registers_r
                .get(frame_reg)
                .copied()
            {
                Some(ConcreteValue::Ref(frame_ptr)) => frame_ptr as *mut pyre_interpreter::PyFrame,
                _ => std::ptr::null_mut(),
            };
        }
        // `sys._getframe` marks the inlined callee's own red frame escaped
        // (`CalleeLocalsShadow`).  After the walk rewrites that frame as
        // `NewWithVtable`, `portal_frame_reg` can name a different box, and
        // finishing that other box leaves the returned object with only
        // `FLAG_ESCAPED`.  `frame.clear()` then refuses a frame that has
        // already returned. A frame the walk allocated already received the
        // jitcode store; this covers the pre-existing escaped object the
        // record-only `setfield_gc` did not write.
        if let Some(shadow) = ctx.frame_state.borrow().callee_shadow.as_ref() {
            let shadow_concrete = if shadow.concrete_frame == 0 {
                std::ptr::null_mut()
            } else {
                shadow.concrete_frame as *mut pyre_interpreter::PyFrame
            };
            if !shadow_concrete.is_null() && unsafe { (*shadow_concrete).escaped() } {
                concrete = shadow_concrete;
            }
        }
        if concrete.is_null() || unsafe { !(*concrete).escaped() } {
            // An unescaped inline frame has no observer after return.
            // optimizeopt.virtualize removes that virtual MIFrame.
            return;
        }
        concrete
    };
    if !concrete_frame.is_null() {
        unsafe { (*concrete_frame).set_frame_finished_execution(true) };
    }
}

/// `PyFrame.frame_finished_execution = True` on the live portal frame, after
/// `fbw_terminate_with_finish` has accepted the exit. See
/// [`finish_current_frame_execution`].
fn commit_top_level_frame_finished<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) {
    let frame = ctx.session.borrow().recording_frame_ptr;
    if frame == 0 {
        return;
    }
    unsafe {
        (*(frame as *mut pyre_interpreter::PyFrame)).set_frame_finished_execution(true);
    }
}

fn recording_raise_keeps_existing_traceback<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) -> bool {
    let jitcode_index = if ctx.is_top_level {
        ctx.session.borrow().recording_jitcode_index
    } else {
        ctx.inline_callee_consts
            .map_or(-1, |consts| consts.jitcode_index)
    };
    crate::state::jitcode_pc_raise_keeps_existing_traceback(jitcode_index, opcode_position as i32)
}

/// The frame identity, code object and instruction coordinate one
/// `PyTraceback` node needs, resolved from the walk's current frame.
struct TracebackNodeSite {
    /// `PyTraceback.frame`.  `OpRef::NONE` when the walk has no materialized
    /// frame for this level — a branchless leaf inlined without one leaves
    /// the color unseeded.  Recording NONE as a `SetfieldGc` operand would
    /// put a bogus box in the trace, so every caller has to DECLINE on it and
    /// leave the node to the frame-fabricating hook.
    frame: OpRef,
    w_code: usize,
    last_instruction: i32,
}

/// Resolve the `PyTraceback` fields of the frame the walk is currently in.
///
/// `None` means no node can be built here: `w_code` is missing, is the
/// `GcRef(usize::MAX)` sentinel a non-standard virtualizable sub-walk
/// carries, is not a code object, or is `hidden_applevel`
/// (`pytraceback.py record_application_traceback`'s first line — such a
/// frame contributes no node at all).  The null / sentinel tests run before
/// `is_code`, whose `py_type_check` would dereference the raw sentinel.
fn traceback_node_site<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) -> Option<TracebackNodeSite> {
    let (jitcode_index, w_code) = if ctx.is_top_level {
        let session = ctx.session.borrow();
        let frame = session.recording_frame_ptr as *const pyre_interpreter::PyFrame;
        let w_code = if frame.is_null() {
            0
        } else {
            unsafe { (*frame).pycode as usize }
        };
        (session.recording_jitcode_index, w_code)
    } else {
        ctx.inline_callee_consts.map_or((-1, 0), |consts| {
            (consts.jitcode_index, ctx.inline_w_code())
        })
    };
    if w_code == 0 || w_code == usize::MAX {
        return None;
    }
    if !unsafe { pyre_interpreter::pycode::is_code(w_code as pyre_object::PyObjectRef) } {
        return None;
    }
    if unsafe {
        pyre_interpreter::pycode::w_code_hidden_applevel(w_code as pyre_object::PyObjectRef)
    } {
        return None;
    }
    let jitcode = crate::state::pyjitcode_for_jitcode_index(jitcode_index)?;
    let last_instruction = crate::py_coord::containing_py_pc_for_jitcode_pc_public(
        jitcode_index,
        opcode_position as i32,
    )?;
    let frame = ctx
        .registers_r
        .get(jitcode.metadata.portal_frame_reg as usize)
        .unwrap_or(OpRef::NONE);
    // Storing this box into the node's `frame` field lets the traceback
    // outlive the frame, so folded inline-callee locals have to reach the
    // array first.  A `STORE_FAST` on the callee's own fresh frame is folded
    // to an SSA register and emits no array store; without replaying that fold,
    // the escaping frame carries only its seeded parameters and every
    // `tb_frame.f_locals` consumer loses the callee's non-parameter locals.
    // Upstream needs no replay because `STORE_FAST` writes
    // `locals_cells_stack_w` itself and virtualization removes the store while
    // the frame stays virtual (`pypy/interpreter/pyframe.py`,
    // `rpython/jit/metainterp/virtualizable.py`).  Declining on `Err` leaves
    // the node to the opaque fabricating hook, the same disposition both
    // callers already take for an unresolved frame.
    if !frame.is_none()
        && residual_call::disarm_folded_inline_callee_after_escape(ctx, opcode_position).is_err()
    {
        return None;
    }
    Some(TracebackNodeSite {
        frame,
        w_code,
        last_instruction,
    })
}

/// Emit the `NewWithVtable` + field stores that build one `PyTraceback`
/// node, and stamp it as `exc`'s traceback.  `w_next` is the node's tail.
fn emit_traceback_node<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    kind: pyre_object::interp_exceptions::ExcKind,
    user: bool,
    site: &TracebackNodeSite,
    w_next: OpRef,
    opcode_position: usize,
) -> Result<OpRef, DispatchError> {
    let traceback = ctx
        .trace_ctx
        .execute_new_with_vtable(crate::descr::pytraceback_size_descr());
    let fields = [
        (site.frame, 0),
        // Field 1 is `lasti`, which the slot holds in bytes.
        (
            ctx.trace_ctx
                .const_int(i64::from(site.last_instruction) * 2),
            1,
        ),
        (w_next, 2),
        // `record_application_traceback` leaves `lineno` at
        // `LINENO_NOT_COMPUTED`; the `tb_lineno` getter resolves it.
        (
            ctx.trace_ctx
                .const_int(pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED),
            3,
        ),
        (ctx.trace_ctx.const_ref(site.w_code as i64), 4),
        (
            ctx.trace_ctx
                .const_ref(pyre_object::pyobject::get_instantiate(
                    &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE,
                ) as i64),
            5,
        ),
    ];
    for (value, index) in fields {
        let descr = crate::descr::pytraceback_field_descr(index);
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[traceback, value], descr.clone());
        ctx.trace_ctx
            .heapcache_setfield_cached(traceback, descr.index(), value);
    }

    // `f_lineno` resolves through `offset2lineno(pycode, last_instr)` on every
    // read, so the frame itself has to carry the coordinate — the node's own
    // `tb_lasti` answers a different question and is frozen.  Preserve the
    // exact operation PyPy traces from `pyopcode.py`'s per-opcode
    // `self.last_instr = ...`: `pyjitpl.py:1188-1199
    // _opimpl_setfield_vable` updates the shadow for the standard frame and
    // falls through to SETFIELD_GC only for a nonstandard/inlined frame.  A raw
    // SETFIELD_GC here wrote the root frame on every caught-exception bridge,
    // even though `last_instr` is declared virtualizable by
    // `interp_jit.py:25-30`.
    let last_instr_value = ctx.trace_ctx.const_int(i64::from(site.last_instruction));
    let unavailable = || DispatchError::TracebackNodeVableFieldUnavailable {
        pc: opcode_position,
        field: "last_instr",
    };
    let vinfo = ctx.trace_ctx.virtualizable_info().ok_or_else(unavailable)?;
    let last_instr_index = vinfo
        .static_field_index_by_name("last_instr")
        .ok_or_else(unavailable)?;
    let last_instr_descr = vinfo.static_field_descr(last_instr_index);
    // A traceback node names an inlined callee's frame as often as the walk's
    // own. Promote the `_nonstandard_virtualizable` `isstandard` PTR_EQ
    // through the walker `implement_guard_value` so the snapshot is complete
    // when recorded.
    let write = vable_ops::with_replace_frames(ctx, |ctx| {
        let nonstandard = vable_ops::walker_nonstandard_virtualizable(
            ctx,
            opcode_position,
            site.frame,
            &last_instr_descr,
        )?;
        Ok(ctx.trace_ctx.vable_setfield_checked(
            nonstandard,
            site.frame,
            last_instr_descr.clone(),
            last_instr_value,
            Some(Value::Int(i64::from(site.last_instruction))),
        ))
    })?;
    let _ = write;

    let traceback_descr = crate::descr::w_exception_traceback_descr_for(kind, user);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[exc, traceback],
        traceback_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(exc, traceback_descr.index(), traceback);
    Ok(traceback)
}

/// IR-virtual PREPEND of one `PyTraceback` node — the general-case sibling
/// of [`record_fresh_application_traceback`].
///
/// `pytraceback.py record_application_traceback`:
///
/// ```text
///     tb = operror.get_traceback()
///     tb = PyTraceback(space, frame, last_instruction, tb)
///     operror.set_traceback(tb)
/// ```
///
/// The fresh variant hard-codes `w_next = NULL`, sound only for an exception
/// the walk just built.  Here `get_traceback()` is emitted as a real
/// `GETFIELD_GC_R` on the per-kind `w_traceback` descr and threaded into
/// `w_next`, so the node prepends instead of truncating the chain a callee
/// already contributed to.
///
/// Every op is optimizer-visible, so escape analysis deletes the node when
/// the exception never escapes.  The opaque
/// `record_{,inline_,top_level_}application_traceback` hooks cannot be: they
/// carry `default_effect_info()` — `EffectInfo::MOST_GENERAL`, i.e. random
/// effects — so the optimizer escapes every argument, forces all lazy sets
/// and drops the whole heapcache, and the top-level one additionally passes
/// the virtualizable frame box, forcing the vable.
///
/// Returns `false` when nothing was emitted; the caller then keeps the
/// opaque hook.
fn record_prepend_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    opcode_position: usize,
) -> Result<Option<OpRef>, DispatchError> {
    if exc.is_none() || exc.is_constant() {
        // A `Const` exception box freezes the RECORDING iteration's address
        // (`walker_record_guard_exception` pins every raise after the first
        // one in a walk), so reading and writing `w_traceback` through it
        // would chain onto a stale object on every later iteration.  The
        // opaque hook takes the live exception as an argument instead.
        return Ok(None);
    }
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return Ok(None);
    };
    if exc_ptr.is_null() || unsafe { !pyre_object::is_exception(exc_ptr) } {
        return Ok(None);
    }
    let Some(site) = traceback_node_site(ctx, opcode_position) else {
        return Ok(None);
    };
    if site.frame.is_none() {
        // No materialized frame for this level.  The opaque inline hook
        // fabricates one from the promoted callee metadata
        // (`record_inline_traceback_for_recording`), so `tb_frame` keeps
        // answering a real frame there; a null here would lose it.
        //
        // Top level has no such hook fallback: `record_top_level_application_
        // traceback` resolves its frame operand from the same unseeded color
        // and declines too, so that level contributes no node at all.  A
        // shorter traceback than `record_application_traceback` would build is
        // the deliberate trade — the frame box a node needs is exactly what
        // this walk lacks, and the vable box is not a substitute (a traceback
        // outlives the frame, so storing it demands the escape marking this
        // path does not perform).
        return Ok(None);
    }
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc_ptr) };
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc_ptr) };
    // `tb = operror.get_traceback()`.  MUST precede the node's own store, or
    // the heapcache answers with the node being built and `w_next` self-links.
    let traceback_descr = crate::descr::w_exception_traceback_descr_for(kind, user);
    let w_next = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, exc, traceback_descr);
    // `record_application_traceback` reads the chain through
    // `operror.get_traceback()`, which marks the node's frame escaped
    // (error.py `OperationError.get_traceback`). The concrete attach that
    // follows this emission performs that write, so the trace records it.
    emit_previous_traceback_frame_escape(ctx, w_next);
    let node = emit_traceback_node(ctx, exc, kind, user, &site, w_next, opcode_position)?;
    Ok(Some(node))
}

/// error.py `OperationError.get_traceback`: `tb.frame.mark_as_escaped()` when
/// the exception already carries a `PyTraceback`.
///
/// Emitted only for a node this walk knows: one it built (a class-known
/// box), whose `frame` the heapcache answers with the frame box the node
/// was built from.  Tracing the `if tb is not None` on a node loaded at run
/// time would need a guard, and a guard here has no position the blackhole
/// can resume from — the read sits inside the synthesized exception
/// dispatch, not in the frame's body.  That frame belongs to a level that
/// ran as real frames and left with the exception, so it is forced already;
/// the mark on it changes no later read (`tb_frame` marks on access).
fn emit_previous_traceback_frame_escape<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    w_next: OpRef,
) {
    if w_next.is_none() || w_next.is_constant() {
        return;
    }
    let Some(Value::Ref(prev_tb)) = ctx.trace_ctx.concrete_of_opref(w_next) else {
        return;
    };
    let prev_tb = prev_tb.0 as pyre_object::PyObjectRef;
    if prev_tb.is_null()
        || unsafe { !pyre_interpreter::pytraceback::is_pytraceback(prev_tb) }
        || !ctx.trace_ctx.heap_cache().is_class_known(w_next)
    {
        return;
    }
    let tb_frame = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        w_next,
        crate::descr::pytraceback_frame_descr(),
    );
    // pyframe.py `mark_as_escaped`: `self.escaped = True`, one bit of the
    // packed `flags` byte here.
    let flags_descr = crate::descr::pyframe_flags_descr();
    let live_flags =
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, tb_frame, flags_descr.clone());
    let escaped_bit = ctx
        .trace_ctx
        .const_int(i64::from(pyre_interpreter::PyFrame::FLAG_ESCAPED));
    let new_flags = ctx
        .trace_ctx
        .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[tb_frame, new_flags],
        flags_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(tb_frame, flags_descr.index(), new_flags);
}

/// IR-virtual traceback record for an exception the walk itself built: the
/// node starts a FRESH chain, so `w_next` is NULL rather than the
/// exception's current traceback.  Sound only where the caller has proven
/// the exception carries no prior node (`fbw_built_exc_take`); the general
/// case is [`record_prepend_application_traceback`].
///
/// Returns `false` when nothing was emitted; the caller then keeps the
/// opaque hook.
fn record_fresh_application_traceback<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    opcode_position: usize,
) -> Result<Option<OpRef>, DispatchError> {
    let ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return Ok(None);
    };
    if exc_ptr.is_null() || unsafe { !pyre_object::is_exception(exc_ptr) } {
        return Ok(None);
    }
    let Some(site) = traceback_node_site(ctx, opcode_position) else {
        return Ok(None);
    };
    if site.frame.is_none() {
        // Same disposition as the prepend sibling, for the same reason: the
        // opaque inline hook fabricates a frame from the promoted callee
        // metadata, while a null one answers None and breaks every consumer
        // that follows `tb_frame.f_code` — `traceback.print_exc` among them.
        return Ok(None);
    }
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc_ptr) };
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc_ptr) };
    let w_next = ctx.trace_ctx.const_ref(0);
    let node = emit_traceback_node(ctx, exc, kind, user, &site, w_next, opcode_position)?;
    Ok(Some(node))
}

/// Compile-time-constant frame fields of an inlined callee.
///
/// The callee's globals and `W_Code` are GCREFs, so they live on the walked
/// `WalkFrameStateData` (`inline_w_globals` / `inline_w_code`) and are read
/// through [`WalkContext::inline_w_globals`] / [`WalkContext::inline_w_code`].
#[derive(Clone, Copy)]
pub struct InlineCalleeConsts {
    /// Jitcode identity used to translate a callee opcode to its Python
    /// instruction coordinate when constructing `PyTraceback` metadata.
    jitcode_index: i32,
}

/// Walk-scoped FBW modes inherited parent-to-child across sub-walk
/// constructions. Formerly three thread-locals.
pub struct FbwWalkMode<Sym: WalkSym> {
    /// Full-body snapshot root used to map a guard's jitcode `op_pc` through
    /// the outer jitcode and read the live walk banks. Null for per-opcode
    /// walks, where `walker_capture_snapshot_for_last_guard` keeps its legacy
    /// single-coordinate behavior.
    pub snapshot_sym: *const Sym,
    /// Guards emitted in an inline sub-walk resume at the caller's CALL
    /// boundary rather than mapping the callee `op_pc` through the outer
    /// jitcode in `walker_capture_snapshot_for_last_guard`.
    pub inline_subwalk: bool,
    /// Python pc of the caller CALL instruction an inline sub-walk is
    /// executing under.  Used only for the temporary live-frame coordinate a
    /// residual frame reader observes; it is not a walk-end resume claim.
    /// Nested inlines inherit the outermost portal CALL so a residual can
    /// still publish `last_instr` onto the portal frame.
    pub inline_caller_py_pc: Option<u32>,
    /// Python pc of the CALL that entered THIS level, in the immediate
    /// caller's own code.  Nested inlines do not inherit: `leaf` reading
    /// `_getframe(1).f_lasti` on `mid` owes `mid`'s CALL, not the portal's.
    pub immediate_inline_caller_py_pc: Option<u32>,
    /// The current sub-walk is a translated builtin gateway helper.  The
    /// helper is not a Python frame, but it is an RPython `MIFrame` and must be
    /// retained as the innermost blackhole frame so a post-residual guard can
    /// continue after the residual instead of re-running the Python CALL.
    pub transparent_helper_subwalk: bool,
    /// Absolute build-time `MetaInterpStaticData.jitcodes` index of that
    /// helper. `None` outside a translated helper sub-walk.
    pub transparent_helper_jitcode_index: Option<usize>,
    /// Set only while walking the generated `write_cell` helper.  Inner
    /// ops are `helper=None` Void writes the StoreName loop-var exemption
    /// cannot see; the caller journals them (`FBW_CELL_STORE_JOURNAL`).
    pub cell_store_helper_subwalk: bool,
    /// A bridge-carrier resume folds nested self-recursive calls directly to
    /// `CALL_ASSEMBLER` (`opimpl_recursive_call_assembler`) rather than
    /// re-unrolling the call tree to the multi-frame depth cap.
    pub carrier_resume: bool,

    /// Whether [`WalkFrameStateData::current_exception_seed`] names a value this walk
    /// stored into `ExecutionContext.sys_exc_value`, as opposed to the
    /// exception the bridge is resuming with.  Only the former answers "what
    /// does that field hold": the bridge's in-flight exception is the one a
    /// handler's `PUSH_EXC_INFO` is about to publish, so saving it as that
    /// opcode's `prev` makes the matching `POP_EXCEPT` reinstate the exception
    /// the handler just finished with.
    pub current_exception_seed_from_walk_store: bool,
    /// Walker-carried mirror of
    /// `MetaInterp.class_of_last_exc_is_const` (`pyjitpl.py`).
    /// This is shared logically across recursive MIFrame walks; catch routing
    /// writes the proven-class state back into the caller's copy.
    pub class_of_last_exc_is_const: bool,
    /// Python-pc of a guard-failure bridge walk's own resume coordinate.
    /// `Some` only on the top-level walk of a bridge trace, `None` otherwise
    /// (loop compiles and sub-walks).
    ///
    /// `generate_guard(resumepc=orgpc)` (`pyjitpl.py`) places a
    /// guard's resume coordinate INSIDE the guarded opcode's implementation,
    /// strictly past the dispatch-top `jit_merge_point`, so an RPython MIFrame
    /// resumed from a guard never re-crosses the loop-header merge point at
    /// position zero. Pyre's bridge walker instead resumes at the opcode
    /// BOUNDARY and would re-cross the header immediately with an empty body,
    /// closing a 0-progress no-op bridge. The merge-point arm consumes this
    /// (via `take()`) to skip exactly the first crossing that lands on the
    /// bridge's own resume coordinate, restoring the RPython positional
    /// semantics.
    pub bridge_entry_merge_pc: Option<usize>,
    /// FOR_ITER key inherited only by the user-instance `__next__` sub-walk.
    /// Every guard it emits is tagged so failure resumes in the blackhole
    /// instead of compiling a bridge from the middle of the callee.
    pub instance_next_foriter_green_key: Option<u64>,
    /// Census marker paired with `instance_next_foriter_green_key`.  Keeping
    /// the route identity separate from the optional descr key lets the
    /// specialization census prove that every captured callee guard was
    /// actually stamped.
    pub instance_next_foriter_census_active: bool,
    /// Set for the sub-walk that resumes a suspended generator body in place of
    /// the `jit_next` residual a `FOR_ITER` over it would otherwise leave.
    /// `None` for every other walk, which is what keeps the yield interception
    /// and the durable-frame journal off every path but this one.
    pub generator_resume: Option<GeneratorResumeSubwalk>,
}

/// The suspended generator a resume sub-walk is running on.
///
/// Two coordinates and one address, all fixed before the walk starts: the
/// frame the body executes on, the marker the body's single `yield` lowered
/// to, and the Python pc that marker stands at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratorResumeSubwalk {
    /// The suspended `PyFrame` — `gen.frame`, the frame the generator will be
    /// resumed from again after this yield.  Durable, so every walk-time write
    /// to it is journalled (`fbw_arm_durable_frame_undo`).
    pub frame: usize,
    /// The OpRef that names that frame in the trace — a `GetfieldGc` off the
    /// generator, not a baked address — so suspension stores land on the
    /// same box the compiled loop re-reads.
    pub frame_box: OpRef,
    /// JitCode offset of the `abort_permanent` the body's `yield` lowered to.
    /// Exact rather than "any `abort_permanent`": a body can carry markers for
    /// unported opcodes too, and those still abort.
    pub yield_marker_jit_pc: usize,
    /// Python pc of that `yield`.  The coordinate the frame is left suspended
    /// at, so the next resume continues after it.
    pub yield_py_pc: u32,
}

impl<Sym: WalkSym> Clone for FbwWalkMode<Sym> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Sym: WalkSym> Copy for FbwWalkMode<Sym> {}

/// The outer snapshot's Python-PC coordinate. Non-root producers preserve the
/// raw JitCode offset that produced the Python word, postponing the exact
/// `trivia_normalized_py_pc_for_jitcode_pc` lookup until a consumer needs it.
/// Root entries and test fixtures have no such native coordinate and retain
/// their Python value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EntryPyPc {
    Py(u32),
    Jit(usize),
}

impl EntryPyPc {
    #[allow(dead_code)]
    fn audit_variant(self) -> &'static str {
        match self {
            Self::Py(_) => "py",
            Self::Jit(_) => "jit",
        }
    }
}

impl<Sym: WalkSym> Default for FbwWalkMode<Sym> {
    fn default() -> Self {
        Self {
            snapshot_sym: std::ptr::null(),
            inline_subwalk: false,
            inline_caller_py_pc: None,
            immediate_inline_caller_py_pc: None,
            transparent_helper_subwalk: false,
            transparent_helper_jitcode_index: None,
            cell_store_helper_subwalk: false,
            carrier_resume: false,
            current_exception_seed_from_walk_store: false,
            class_of_last_exc_is_const: false,
            bridge_entry_merge_pc: None,
            instance_next_foriter_green_key: None,
            instance_next_foriter_census_active: false,
            generator_resume: None,
        }
    }
}

/// `WalkContext` carries two lifetimes:
/// * `'frame` — the inner-frame lifetime: register banks + trace
///   recorder. Sub-walk recursion (`inline_call_r_r/dR>r`) allocates
///   fresh register banks scoped to the sub-walk's block, so
///   `'frame` must be allowed to *shrink* on recursion.
/// * `'static_a` — the outer lifetime: descr pool + sub-jitcode
///   lookup. These flow unchanged from caller into callee, so they
///   keep their original (longer) lifetime.
pub struct WalkContext<'frame, 'static_a: 'frame, Sym: WalkSym> {
    /// The active and paused adapters share the same frame-owned slots.
    pub frame_state: WalkFrameState,
    /// Compile-time-constant frame fields of this inlined callee's own
    /// unseeded portal frame. This is the walk-time equivalent of the
    /// codewriter's non-portal branch (`codewriter.rs`),
    /// where `pycode` and `w_globals` are constants rather than
    /// reads that alias the caller's frame. `None` when this is not an
    /// inlined-callee sub-walk.
    pub inline_callee_consts: Option<InlineCalleeConsts>,
    /// FBW walk modes inherited by nested sub-walk contexts.
    pub fbw_mode: FbwWalkMode<Sym>,
    /// Caller-owned state shared by every frame in this walk attempt.
    pub session: &'static_a std::cell::RefCell<WalkSession>,
    /// Symbolic Ref-bank register file. Indexing matches RPython
    /// `MIFrame.registers_r` (`pyjitpl.py`); the byte after a
    /// `r`-coded operand opcode indexes directly into this slice.
    /// Shared frame-owned slots permit result stores and GC forwarding without
    /// an exclusive slice surviving a descendant call.
    pub registers_r: &'frame RegisterBank,
    /// Symbolic Int-bank register file. Indexing matches RPython
    /// `MIFrame.registers_i` (`pyjitpl.py`). Result stores update shared
    /// frame-owned slots, just as `int_copy/i>i` writes the MIFrame list.
    pub registers_i: &'frame RegisterBank,
    /// Symbolic Float-bank register file. Indexing matches RPython
    /// `MIFrame.registers_f` (`pyjitpl.py`). Mutable so
    /// `float_<binop>/ff>f` and `float_neg/f>f` can land their dst.
    pub registers_f: &'frame RegisterBank,

    /// Concrete shadow mirror for `registers_i`.  Color-indexed (not
    /// semantic-slot indexed like `concrete_registers_r`) because pyre's
    /// Int bank has no "semantic-slot" abstraction — Int registers are
    /// post-regalloc colors directly.  Length equals `registers_i.len()`
    /// on production entries; the test fixtures pass `&mut []`.
    ///
    /// Invariant: every walker handler that writes `registers_i[dst]`
    /// MUST also write `concrete_registers_i[dst]` in lock-step,
    /// mirroring the Ref-bank contract above.  `write_int_reg` enforces
    /// it — it sanitizes the value and writes the shadow slot with the
    /// symbolic one.
    pub concrete_registers_i: &'frame mut [ConcreteValue],
    /// Descr pool for `d`-coded operands. Each `d` argcode in the
    /// jitcode bytes resolves to `descr_refs[2-byte LE index]`.
    /// RPython `Assembler.descrs` (`assembler.py`) +
    /// `BlackholeInterpBuilder.setup_descrs` (`blackhole.py`)
    /// — production callers pass the codewriter-emitted descr table.
    pub descr_refs: &'static_a dyn DescrRefTable,
    /// Raw `BhDescr` pool source for vable-array `(VableArray, Array)`
    /// recognition (`vable_array_descrs_from_jitcode`).  [`RawDescrPool::
    /// Global`] for build-time canonical jitcodes + tests (their operands
    /// index the shared `ALL_DESCRS`); [`RawDescrPool::PerFn`] for full-body
    /// walks (the per-`CodeObject` body resolves through its own
    /// `exec.descrs`).  Index-parallel to [`Self::descr_refs`].
    pub raw_descrs: RawDescrPool<'static_a>,
    /// Whether this walk is the SOLE concrete-execution leg.
    ///
    /// `false` (shadow validation, the diagnostic full-body probe,
    /// tests): a separate concrete interpreter (the Python interpreter,
    /// or none for the discard-the-trace probe) is
    /// authoritative, so the walker must NOT re-execute may-force
    /// residual calls — doing so would double their side effects /
    /// corrupt the live heap (`cut_trace` rolls back only the IR
    /// recorder, not heap/iterator state).
    ///
    /// `true` (the production full-body walk and its inline
    /// sub-walks): the walker is the only thing executing the
    /// JitCode body — `eval_loop_jit` skips `execute_opcode_step` for
    /// walker-handled opcodes — so
    /// [`try_execute_residual_call_via_executor`] runs residual calls
    /// concretely.  RPython parity: the metainterp executes EVERY
    /// residual_call during tracing (`do_residual_call` →
    /// `executor.execute_varargs`, pyjitpl.py), pure or not, so a
    /// downstream `goto_if_not` reads a concrete result.
    pub is_authoritative_executor: bool,
    /// Live trace recorder. `record_finish` / `record_op` /
    /// `record_op_with_descr` go through this.
    pub trace_ctx: &'frame mut TraceCtx,
    /// Whether this `WalkContext` is the outermost trace frame
    /// (`true`) or a nested sub-jitcode frame entered through
    /// `inline_call_r_r/dR>r` recursion (`false`). The flag
    /// disambiguates dual-behaviour terminators:
    ///
    /// * `ref_return/r` at top-level stashes the finish payload +
    ///   Terminate; inside a sub-walk it returns `SubReturn { result }`
    ///   so the caller's `inline_call_*` handler can write the dst
    ///   register.
    /// * `raise/r` propagates `SubRaise { exc }` either way; only at
    ///   top-level, with no handler anywhere in the framestack, does the
    ///   post-op scan turn it into an `is_exception` finish payload.  In
    ///   a sub-walk the caller's `inline_call_*` handler may catch via
    ///   `catch_exception` metadata or bubble up further.
    ///
    /// RPython parity: pyre flattens the framestack-driven
    /// `metainterp.popframe()` + `finishframe[_exception]` flow
    /// (`pyjitpl.py`) into this Rust-level outcome.
    pub is_top_level: bool,
    /// Caller-provided callback resolving a `jitcode_index` to a
    /// `SubJitCodeBody`. Invoked when `inline_call_r_r/dR>r` fires
    /// and needs to recurse into the callee's bytecode body.
    pub sub_jitcode_lookup: &'static_a SubJitCodeLookup,
    /// Outer snapshot coordinate. Root entries keep `MIFrame.orgpc`; every
    /// coordinate-native producer stores its raw JitCode offset and resolves
    /// the matching Python PC lazily at the consumer boundary.
    ///
    /// Read by [`walker_capture_snapshot_for_last_guard`] to stamp the
    /// snapshot frame's Python PC.
    pub entry_py_pc: EntryPyPc,
    /// Codewrite-time resume-marker twin for the outer snapshot coordinate
    /// (`entry_py_pc`), carried by the arm-path snapshot word. `None` when the
    /// creation site has no
    /// jitcode-native outer coordinate.
    pub outer_resume_marker_jit_pc: Option<usize>,
    /// JitCode index of the **outer** `PyJitCode.jitcode` — the Python
    /// bytecode jitcode whose Python opcode is currently being
    /// dispatched.  Pyre's blackhole resume only re-enters Python-
    /// bytecode jitcodes, so guard snapshots must reference the outer
    /// pyjitcode regardless of how deep the walker's sub-walk nesting
    /// is.
    ///
    /// The retired per-opcode arm entry read this from
    /// `(*sym.jitcode).index()`; inline sub-walks seed it from the
    /// CALL-site capture (sub-walks don't change the outer Python
    /// opcode).  Test fixtures + [`dispatch_via_miframe`] default to
    /// `0` — the full-body guard capture reads `sym.jitcode` directly
    /// instead of this field.
    pub outer_jitcode_index: u32,

    /// Snapshot-capture failure latched by the `WalkerFrameOps`
    /// `generate_guard` impl, whose `()` trait signature (shared with
    /// `MIFrame`) has no error channel.  The residual-call dispatcher
    /// drains the latch and surfaces the `DispatchError` so a guard
    /// recorded without a resume snapshot aborts the walk instead of
    /// compiling.
    pub pending_guard_snapshot_error: Option<DispatchError>,

    /// #73: the absolute operand-stack depth `vstack_boxes` currently
    /// reflects — the depth ON ENTRY to the Python opcode at
    /// `vstack_cur_pypc` (i.e. AFTER the previous opcode's stack effect
    /// was reconciled).
    pub vstack_depth: usize,
    /// #73: the Python pc of the opcode currently being walked.  A change
    /// in `containing_py_pc_for_jitcode_pc(jit_pc)` from this value marks a
    /// Python-opcode boundary, where the previous opcode's stack effect is
    /// reconciled into `vstack_boxes` (see [`reconcile_vstack_at_boundary`]).
    pub vstack_cur_pypc: u32,
    /// #73: whether `vstack_boxes` is a trustworthy mirror of the live
    /// operand stack.  Set `false` at walk entry until seeded, and latched
    /// `false` permanently on the first unmodeled stack effect so the
    /// `stack_sync` overlay declines to use it.
    pub vstack_valid: bool,

    /// #389(b): the py_pc the walk backed off FROM when it entered the
    /// codewriter's out-of-order FOR_ITER-entry permutation lowering
    /// (`SWAP`/`BUILD_LIST`/`SWAP` before a `FOR_ITER`, lowered non-monotonically
    /// in jitcode).  `u32::MAX` when not inside such a region.  While set, the
    /// per-op reconcile is invalid (the walk re-visits already-passed py_pcs out
    /// of order), so `reconcile_vstack_at_boundary` reseeds the whole mirror from
    /// the virtualizable shadow instead of replaying stack effects.  Cleared once
    /// the walk advances past this ceiling (py-pc order is monotonic again).
    pub vstack_reorder_ceiling: u32,

    /// The py_pc the walk's floor lookup reports while it is inside an
    /// out-of-line exception LANDING block — the unwind bookkeeping the
    /// codewriter emits per catch site, after the whole body, which then jumps
    /// to the handler proper.  `None` when the catch target is in line with its
    /// handler (nothing to skip).
    ///
    /// Those bytes lower no Python opcode, but they are laid out inside some
    /// other opcode's floor segment, so the walk reports that opcode as a
    /// boundary and the mirror would replay a stack effect from a block it
    /// never ran — and, py order having gone backwards, arm the reorder region
    /// over the whole handler body.  Hold the handler-entry coordinate until
    /// the walk leaves those bytes instead.
    pub vstack_handler_landing_py: Option<u32>,
    /// #73: the jitcode offset of the `-live-` byte that
    /// precedes the CURRENT opcode's guard resume point — the `-live-`
    /// BEFORE (`pyjitpl.py`, normal guard resume reads at
    /// `self.pc - SIZE_LIVE_OP`).  Maintained purely from `DecodedOp` by the
    /// walk; `usize::MAX` until the first `-live-` is seen.  Side-data only.
    pub live_before_jit_pc: usize,
    /// #73: the jitcode offset of the `-live-` byte that
    /// trails a residual-call opcode — the `-live-` AFTER (`pyjitpl.py`,
    /// residual-call guard resume reads at `self.pc`).  Maintained purely
    /// from `DecodedOp` by the walk; `usize::MAX` until set.  Side-data only.
    pub live_after_jit_pc: usize,
}

/// Locals of one frame that is live on this trace.
pub(crate) enum ReceiverTraceLocals {
    /// The portal virtualizable's `locals_cells_stack_w` boxes.
    Portal,
    /// One inline frame's own slot map. Missing slots are not listed.
    Inline(Vec<(i64, OpRef, majit_ir::Value)>),
}

fn inline_shadow_slots(shadow: &CalleeLocalsShadow) -> Vec<(i64, OpRef, majit_ir::Value)> {
    let mut slots: Vec<(i64, OpRef, majit_ir::Value)> = shadow
        .opref
        .iter()
        .filter(|(_, opref)| !opref.is_none())
        .map(|(&slot, &opref)| {
            let value = shadow
                .concrete
                .get(&slot)
                .map(|entry| entry.value)
                .unwrap_or(majit_ir::Value::Void);
            (slot, opref, value)
        })
        .collect();
    slots.sort_by_key(|(slot, _, _)| *slot);
    slots
}

fn inline_shadow_matches(shadow: &CalleeLocalsShadow, obj: OpRef, concrete: usize) -> bool {
    (shadow.frame_box != OpRef::NONE && shadow.frame_box == obj)
        || (concrete != 0 && shadow.concrete_frame == concrete)
}

impl<Sym: WalkSym> WalkContext<'_, '_, Sym> {
    /// The frame on this trace — portal virtualizable or any inline level —
    /// whose box or concrete address is `obj` / `concrete`.
    pub(crate) fn receiver_trace_locals(
        &self,
        obj: OpRef,
        concrete: usize,
    ) -> Option<ReceiverTraceLocals> {
        let hit = |shadow: &CalleeLocalsShadow| -> Option<ReceiverTraceLocals> {
            inline_shadow_matches(shadow, obj, concrete)
                .then(|| ReceiverTraceLocals::Inline(inline_shadow_slots(shadow)))
        };
        if let Some(found) = self
            .frame_state
            .borrow()
            .callee_shadow
            .as_ref()
            .and_then(&hit)
        {
            return Some(found);
        }
        let session = self.session.borrow();
        for frame in &session.framestack {
            if let Some(live) = frame.live.as_ref()
                && let Some(found) = live
                    .frame_state
                    .borrow()
                    .callee_shadow
                    .as_ref()
                    .and_then(&hit)
            {
                return Some(found);
            }
            for parent in &frame.parents {
                let Some(state) = parent.frame_state.as_ref() else {
                    continue;
                };
                if let Some(found) = state.borrow().callee_shadow.as_ref().and_then(&hit) {
                    return Some(found);
                }
            }
        }
        for helper in &session.helper_live {
            if let Some(found) = helper
                .frame_state
                .borrow()
                .callee_shadow
                .as_ref()
                .and_then(&hit)
            {
                return Some(found);
            }
        }
        drop(session);
        let vable_box = self.trace_ctx.standard_virtualizable_box();
        let vable_ptr = self.trace_ctx.standard_virtualizable_ptr();
        let heap_ptr = self
            .trace_ctx
            .virtualizable_heap_ptr()
            .map(|ptr| ptr as usize);
        let vref_box = (concrete != 0)
            .then(|| self.trace_ctx.virtualref_virtual_for_object_ptr(concrete))
            .flatten();
        let portal = vable_box == Some(obj)
            || vable_ptr == Some(concrete)
            || heap_ptr == Some(concrete)
            || vref_box.is_some_and(|red| vable_box == Some(red) || red == obj);
        if portal {
            return Some(ReceiverTraceLocals::Portal);
        }
        // The executing inline frame's `f_back` is still on this trace. When
        // that caller is the loop portal, its locals are the virtualizable
        // boxes even if the traceback names the live object and the vable
        // cell names the snapshot copy.
        let inline = current_inline_concrete_frame();
        if inline != 0 && concrete != 0 {
            let raw = unsafe { (*(inline as *const pyre_interpreter::PyFrame)).f_backref };
            if !raw.is_null() {
                let caller = if unsafe {
                    majit_metainterp::virtualref::ptr_is_virtual_ref(raw as *const u8)
                } {
                    unsafe { majit_metainterp::virtualref::vref_forced(raw as *const u8) as usize }
                } else {
                    raw as usize
                };
                if caller == concrete && (vable_box.is_some() || vable_ptr.is_some()) {
                    return Some(ReceiverTraceLocals::Portal);
                }
            }
        }
        None
    }

    /// The standing exception, read from the one session-wide slot
    /// (`metainterp.last_exc_value`, `pyjitpl.py opimpl_last_exc_value`).
    fn last_exc_value(&self) -> Option<OpRef> {
        self.session.borrow().last_exc_value
    }

    /// Concrete shadow of [`WalkContext::last_exc_value`].
    fn last_exc_value_concrete(&self) -> ConcreteValue {
        self.session.borrow().last_exc_value_concrete
    }

    /// `metainterp.last_exc_value = exc` with its concrete shadow
    /// (`pyjitpl.py`, `:2775 execute_ll_raised`).
    fn set_last_exc_value(&self, exc: OpRef, concrete: ConcreteValue) {
        let mut sess = self.session.borrow_mut();
        sess.last_exc_value = Some(exc);
        sess.last_exc_value_concrete = concrete;
    }

    /// Rewrite only the symbolic half, for the arm that has already
    /// established the concrete one — `handle_possible_exception` pins the
    /// class first and then swaps in the `GUARD_EXCEPTION` result box
    /// (`pyjitpl.py`).
    fn set_last_exc_value_op(&self, exc: OpRef) {
        self.session.borrow_mut().last_exc_value = Some(exc);
    }

    /// Rewrite only the concrete half, for the arm that already carries the
    /// symbolic OpRef and is refining the live `PyObjectRef` behind it.
    fn set_last_exc_value_concrete(&self, concrete: ConcreteValue) {
        self.session.borrow_mut().last_exc_value_concrete = concrete;
    }

    /// `metainterp.clear_exception()` (`pyjitpl.py`), which
    /// `finishframe` also runs before popping the frame (`:2505`).
    fn clear_last_exc_value(&self) {
        let mut sess = self.session.borrow_mut();
        sess.last_exc_value = None;
        sess.last_exc_value_concrete = ConcreteValue::Null;
    }

    /// Restore both halves of the slot, including back to "no exception".
    ///
    /// The residual-call arms that save the slot across a nested walk need
    /// the `None` case too, which [`WalkContext::set_last_exc_value`] cannot
    /// express.
    fn restore_last_exc_value(&self, exc: Option<OpRef>, concrete: ConcreteValue) {
        let mut sess = self.session.borrow_mut();
        sess.last_exc_value = exc;
        sess.last_exc_value_concrete = concrete;
    }

    /// Resolve the outer snapshot coordinate at the exact consumer boundary.
    fn entry_py_pc(&self) -> u32 {
        match self.entry_py_pc {
            EntryPyPc::Py(py_pc) => py_pc,
            EntryPyPc::Jit(jitcode_pc) => crate::py_coord::resume_py_pc_for_jitcode_word(
                self.outer_jitcode_index as i32,
                jitcode_pc as i32,
            ) as u32,
        }
    }

    /// Live `Function.w_func_globals_obj` of this inlined callee, from the
    /// walked `WalkFrameState.inline_w_globals` slot.
    fn inline_w_globals(&self) -> usize {
        self.frame_state.borrow().inline_w_globals
    }

    /// Live `W_Code` of this inlined callee, from the walked
    /// `WalkFrameState.inline_w_code` slot.
    fn inline_w_code(&self) -> usize {
        self.frame_state.borrow().inline_w_code
    }
}

/// Outcome of dispatching one opcode. The walker uses this to decide
/// whether to continue stepping or terminate.
///
/// RPython parity: `pyjitpl.py:opimpl_*` returns through Python's
/// generator/exception flow — opcodes that end a trace raise
/// `DoneWithThisFrameRef`/`SwitchToBlackhole`/`ChangeFrame`. Pyre
/// flattens that into an explicit enum because Rust has no analogous
/// non-local exit and we want the walker to stay in plain Result form.
///
/// Not `Copy`: the `CloseLoop` variant carries a `Vec<OpRef>` of merge-point
/// jump args, mirroring `TraceAction::CloseLoopWithArgs` (`lib.rs`)
/// which is likewise non-`Copy`.
#[derive(Debug, Clone)]
pub enum DispatchOutcome {
    /// Step succeeded, continue with the next opcode at the returned pc.
    Continue,
    /// Trace ends here. The arm produced a final `ref_return`/`raise`
    /// equivalent at the top-level frame and no further bytes should
    /// be walked.
    ///
    /// `finish_arg` / `finish_arg_type` are the FINISH operand
    /// (`N_aryOp._args`). `Type::Void` is a `void_return/` with no
    /// operand (`OpRef::NONE`). A raised exit is the same `Type::Ref`
    /// operand distinguished by [`fbw_finish_is_exception`].
    Terminate {
        finish_arg: OpRef,
        finish_arg_type: Type,
    },
    /// Sub-walk frame returned with a result OpRef (Some) or void
    /// (None — no `>X` slot in the callee's `*_return` op). Surfaced
    /// only when `WalkContext::is_top_level == false`. The caller's
    /// `inline_call_r_r/dR>r` handler consumes this to write the dst
    /// register and continue stepping its own jitcode.
    ///
    /// RPython parity: `metainterp.popframe()` after an `opimpl_*_return`
    /// (`pyjitpl.py`) — the callee frame ends, control returns
    /// to the caller's metainterp loop with the resbox in hand.
    SubReturn { result: Option<OpRef> },
    /// Sub-walk frame raised. RPython
    /// `metainterp.popframe() + finishframe_exception()` walks up the
    /// framestack scanning each parent's exceptiontable; pyre's walker
    /// surfaces the outcome to the caller's `inline_call_*` handler,
    /// which today bubbles it up further (no per-handler
    /// exceptiontable scan yet — that lives behind the
    /// `catch_exception/L` metadata pipe and is deferred until the
    /// per-PC exceptiontable plumb-through lands).
    ///
    /// `exc_concrete` carries the callee's `last_exc_value_concrete`
    /// across the frame boundary so the caller's `inline_call` SubRaise
    /// arm can seed its own `last_exc_value_concrete` and a downstream
    /// `raise/r` / `reraise/` reads the right concrete for GUARD_CLASS
    /// emission. Empty when the callee itself didn't track a concrete
    /// (e.g. shadow gap or `Null`-seeded raise).
    SubRaise {
        exc: OpRef,
        exc_concrete: ConcreteValue,
    },
    /// Trace recording must abort and resume in blackhole mode.
    ///
    /// RPython parity: `pyjitpl.py` routes
    /// `OS_NOT_IN_TRACE` residual calls through
    /// `do_not_in_trace_call`; `pyjitpl.py` raises
    /// `SwitchToBlackhole(ABORT_ESCAPE)` if the concrete call raises.
    /// The trace walker cannot execute the callee concretely yet, so a
    /// reached `OS_NOT_IN_TRACE` site is surfaced as this non-local
    /// outcome instead of recording a call that upstream would omit.
    SwitchToBlackhole {
        reason: i32,
        raising_exception: bool,
    },
    /// `jit_merge_point/cIRFIRF` reached a loop header that was already
    /// visited with a matching red-bank shape — close the trace as a
    /// loop. `jump_args` are the merge point's red boxes (the live loop
    /// args, in `[int.., ref.., float..]` order); `loop_header_pc` is the
    /// Python pc of the merge point (decoded from the op's `next_instr`
    /// green).
    ///
    /// RPython parity: `reached_loop_header` "found"
    /// branch → `compile_trace`/close-loop. The production driver maps
    /// this to `TraceAction::CloseLoopWithArgs { jump_args, loop_header_pc }`
    /// (`lib.rs`); the retired trait-path counterpart was
    /// `close_loop_args`'s `Ok(Some(live_args))`.
    CloseLoop {
        jump_args: Vec<OpRef>,
        loop_header_pc: usize,
        /// Codewrite-time resume-marker twin at the merge point op's jitcode
        /// offset, carried into the loop-close guards' snapshot words.
        loop_header_marker_jit_pc: Option<usize>,
        /// Python instruction containing the explicit `loop_header` op that
        /// preceded this merge-point crossing.  The codewriter emits that op in
        /// the `JUMP_BACKWARD` block, so this is the owning back-edge opcode.
        back_edge_pc: Option<usize>,
        /// Resume-marker twin for [`Self::CloseLoop::back_edge_pc`].
        back_edge_marker_jit_pc: Option<usize>,
    },
    /// `jit_merge_point/cIRFIRF` reached a loop header that already has
    /// compiled targets, and the in-walk `compile_trace` attempt
    /// succeeded: the trace-so-far was compiled as a bridge / entry
    /// bridge ending in a JUMP into the existing loop
    /// (`reached_loop_header` →
    /// `self.compile_trace(live_arg_boxes, ptoken)`; "raises in case it
    /// works"). The tracing session was consumed by `compile_trace`;
    /// the walk must stop without compiling or aborting again. The
    /// driver maps this to `TraceAction::CompileTrace`.
    ///
    /// `loop_header_pc` is the merge point's python pc (the
    /// `jit_merge_point` `next_instr`), carried so the walk driver can
    /// flush the walk-end frame state and resume the interpreter AT the
    /// header instead of replaying the walked region
    /// (`flush_walk_end_state_to_frame`).
    CompileTracePending { loop_header_pc: usize },
    /// A multi-frame inlined callee sub-walk reached the callee's OWN
    /// `jit_merge_point` (its loop back-edge / header) while a compiled
    /// loop token for that green key already exists. Instead of declining
    /// the whole enclosing trace (the old `JitMergePointGreenKeyUnresolved`
    /// path → trait leg), the sub-walk surfaces this so the caller's
    /// `try_walker_inline_user_call` return site emits a
    /// `CALL_ASSEMBLER` into the callee loop token — the walker mirror of
    /// `opimpl_recursive_call_assembler` (`metainterp.rs`). The callee
    /// frame is the seeded virtual `PyFrame` the inline already populated
    /// via `setarrayitem_vable` during the prologue walk; the caller sets
    /// `last_instr = target_pc - 1` on it and passes it as the
    /// CALL_ASSEMBLER `[frame, ec]` red arg (forcing the virtual
    /// materializes the locals). `token` is `Some` when
    /// `get_or_make_portal_assembler_token_arc` already has a loop token
    /// (`direct_assembler_call`). A bridge carrier whose lookup returns
    /// none surfaces `None` so the drain records the portal runner via
    /// `direct_call_may_force` instead of walking the callee's loop body
    /// (`do_recursive_call`, `assembler_call=False`).
    SubLoopCalleeCallAssembler {
        token: Option<std::sync::Arc<majit_backend::JitCellToken>>,
        target_pc: usize,
    },
    /// `jit_merge_point` was crossed by a trace that already carries
    /// `force_finish_trace` and has passed 0.8x the trace limit
    /// (`pyjitpl.py`, the tail of `MIFrame.debug_merge_point`).
    /// The trace is close enough to the limit that aborting it would waste
    /// the whole recording, so it is terminated here instead: the walker has
    /// already recorded a `GUARD_ALWAYS_FAILS` at this merge point, which
    /// takes every execution back to the interpreter.
    ///
    /// The compile half needs the `MetaInterp` the walker does not hold, so
    /// it runs in the driver, in one of the two arms upstream branches to at
    /// `pyjitpl.py`: `is_loop` selects
    /// [`majit_metainterp::TraceAction::SegmentedLoop`] (`compile_simple_loop`
    /// plus `attach_procedure_to_interp`) over
    /// [`majit_metainterp::TraceAction::SegmentedBridge`] (`compile_trace`
    /// with the resumekey). `exception_box` is the operand of the unreachable
    /// FINISH behind the guard; the loop arm's FINISH is already recorded, the
    /// bridge arm's is built by `compile_finish_from_active_session`.
    SegmentTrace { is_loop: bool, exception_box: OpRef },
}

/// `pyjitpl.py _create_segmented_trace_and_blackhole`.
///
/// The trace is close enough to `trace_limit` that aborting it would waste the
/// whole recording, so it is terminated here instead: an always-failing guard
/// takes every execution back to the interpreter, and the FINISH behind it
/// exists only to give the segment a terminator.
///
/// `mp_opcode_pc` is the merge-point op's jitcode pc (the guard's resume
/// coordinate); `mp_green_pc` is the Python pc it names.
fn create_segmented_trace<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    mp_opcode_pc: usize,
    mp_green_pc: usize,
    synthesized_python_opcode_boundary: bool,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    // Upstream's cut cannot fail to reach the blackhole: the
    // `SwitchToBlackhole` it raises is caught by
    // `run_blackhole_interp_to_cancel_tracing`, whose conversion has no
    // failure arm.  Single-pass tracing reaches the same guarantee from the
    // other side -- by cutting only once the forward image exists -- because
    // the walk has already RUN everything it recorded, so the alternative to
    // a forward resume is executing that region a second time.  A refusal
    // leaves `force_finish_trace` armed and the walk recording, so the next
    // merge point or the too-long check answers instead.
    //
    // Before the guard: a refusal must leave no always-fails guard and no
    // FINISH behind, and the latch records nothing into the trace.
    //
    // `step_vstack_mirror` leaves a block-head marker on the opcode that
    // entered it.  A JUMP_BACKWARD cut then publishes that STORE's entry
    // value, and `bhimpl_jit_merge_point`'s ContinueRunningNormally resumes
    // the loop header one slot deeper than `depth_at_py_pc` there.  Same
    // post-step reconcile as `blackhole_if_trace_too_long`.
    reconcile_vstack_to_resume_pc(ctx, mp_opcode_pc);
    if !latch_abort_blackhole(ctx, mp_opcode_pc, "segment-cut") {
        census_record("SegmentTrace::LatchRefused");
        return Ok(None);
    }
    // The latch answers for the snapshot-array stack source; this leg publishes
    // from the walker mirror, whose own height check runs only in the adopter.
    // Ask it here, while a refusal is still free.
    //
    // The refusal has to unstage what the latch just wrote, and from BOTH slots:
    // `latch_abort_blackhole` routes an inline sub-walk to the multi-frame one,
    // which this predicate cannot see and this leg has no adopt for.  Left
    // staged, that image answers `abort_blackhole_latched()` for a later abort,
    // which then records none of its own and adopts this merge point instead of
    // its own stop — dropping everything executed in between.
    if !latched_single_frame_mirror_publishable() {
        reset_single_frame_blackhole();
        census_record("SegmentTrace::MirrorStackRefused");
        return Ok(None);
    }
    // pyjitpl.py `generate_guard(rop.GUARD_ALWAYS_FAILS)`. The resume
    // position is the merge-point op, whose preceding `-live-` marker names the
    // boxes the blackhole resumes with.
    ctx.trace_ctx.record_guard(OpCode::GuardAlwaysFails, &[], 0);
    let snapshot = if synthesized_python_opcode_boundary {
        walker_capture_segmented_python_opcode_guard(ctx, mp_opcode_pc)
    } else {
        walker_capture_snapshot_for_last_guard(ctx, mp_opcode_pc)
    };
    if let Err(error) = snapshot {
        let cause = error.variant_name();
        census_record(cause);
        return Err(DispatchError::SegmentTraceSnapshotUnavailable {
            pc: mp_opcode_pc,
            cause,
        });
    }
    // pyjitpl.py:1633-1636 `exception_box = ConstInt(ptr2int(
    // llexception.typeptr))` — the AssertionError type pointer the unreachable
    // FINISH escapes with. The op sits behind a guard that always fails, so the
    // placeholder `ConstInt(0)` stands in for a value nothing reads. Both arms
    // below take this same box, as upstream does.
    let exception_box = ctx.trace_ctx.const_int(0);
    // pyjitpl.py:1639-1640 `if (metainterp.current_merge_points and
    // isinstance(metainterp.resumekey, compile.ResumeFromInterpDescr)):` — a
    // trace that owns a merge point and entered from the interpreter becomes a
    // segmented loop; anything else (a guard-origin bridge) takes the else-arm.
    let is_loop = ctx
        .trace_ctx
        .current_merge_points_first_greenkey()
        .is_some()
        && ctx.trace_ctx.resumekey_original_loop_token().is_none();
    // pyjitpl.py:1637 `history.record1(rop.FINISH, exception_box, None,
    // descr=token)`. The loop arm keeps it here; the bridge arm leaves it to
    // `compile_finish_from_active_session`, the port of pyjitpl.py:1666
    // `compile_trace(metainterp, resumekey, [exception_box])`, which records the
    // same FINISH carrying `exit_frame_with_exception_descr_ref`. Recording it
    // here as well would give that trace two terminators.
    if is_loop {
        ctx.trace_ctx.record_finish(exception_box, Type::Int);
    }
    // pyjitpl.py:1671-1673: "we now need to blackhole back to the interpreter
    // instead of jumping to some existing code, because we are at a really
    // arbitrary place here." Under single-pass tracing the walk already RAN
    // everything it recorded, so the interpreter must resume at this merge
    // point's own green pc rather than at the one it was left holding —
    // the same handoff the abort path publishes. Without it the walked
    // iterations are executed a second time.
    ctx.trace_ctx.walk_final_pc = Some(mp_green_pc);
    ctx.trace_ctx.walk_final_reds = Vec::new();
    // pyjitpl.py:1673 `raise SwitchToBlackhole(ABORT_SEGMENTED_TRACE)`.
    Ok(Some(DispatchOutcome::SegmentTrace {
        is_loop,
        exception_box,
    }))
}

impl PartialEq for DispatchOutcome {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Continue, Self::Continue) => true,
            (
                Self::Terminate {
                    finish_arg: a_arg,
                    finish_arg_type: a_ty,
                },
                Self::Terminate {
                    finish_arg: b_arg,
                    finish_arg_type: b_ty,
                },
            ) => a_arg == b_arg && a_ty == b_ty,
            (Self::SubReturn { result: a }, Self::SubReturn { result: b }) => a == b,
            (
                Self::SubRaise {
                    exc: a_exc,
                    exc_concrete: a_concrete,
                },
                Self::SubRaise {
                    exc: b_exc,
                    exc_concrete: b_concrete,
                },
            ) => a_exc == b_exc && a_concrete == b_concrete,
            (
                Self::SwitchToBlackhole {
                    reason: a_reason,
                    raising_exception: a_raising,
                },
                Self::SwitchToBlackhole {
                    reason: b_reason,
                    raising_exception: b_raising,
                },
            ) => a_reason == b_reason && a_raising == b_raising,
            (
                Self::CloseLoop {
                    jump_args: a_args,
                    loop_header_pc: a_pc,
                    loop_header_marker_jit_pc: a_marker,
                    back_edge_pc: a_back_edge_pc,
                    back_edge_marker_jit_pc: a_back_edge_marker,
                },
                Self::CloseLoop {
                    jump_args: b_args,
                    loop_header_pc: b_pc,
                    loop_header_marker_jit_pc: b_marker,
                    back_edge_pc: b_back_edge_pc,
                    back_edge_marker_jit_pc: b_back_edge_marker,
                },
            ) => {
                a_args == b_args
                    && a_pc == b_pc
                    && a_marker == b_marker
                    && a_back_edge_pc == b_back_edge_pc
                    && a_back_edge_marker == b_back_edge_marker
            }
            (
                Self::SegmentTrace {
                    is_loop: a_is_loop,
                    exception_box: a_exc,
                },
                Self::SegmentTrace {
                    is_loop: b_is_loop,
                    exception_box: b_exc,
                },
            ) => a_is_loop == b_is_loop && a_exc == b_exc,
            (
                Self::CompileTracePending {
                    loop_header_pc: a_pc,
                },
                Self::CompileTracePending {
                    loop_header_pc: b_pc,
                },
            ) => a_pc == b_pc,
            (
                Self::SubLoopCalleeCallAssembler {
                    token: a_token,
                    target_pc: a_pc,
                },
                Self::SubLoopCalleeCallAssembler {
                    token: b_token,
                    target_pc: b_pc,
                },
            ) => {
                a_token.as_ref().map(|token| token.number)
                    == b_token.as_ref().map(|token| token.number)
                    && a_pc == b_pc
            }
            _ => false,
        }
    }
}

/// `opimpl_jit_merge_point`'s non-zero `portal_call_depth` arm surfaces the
/// recursive portal call. A carrier resume (`rebuild_from_resumedata`) has no
/// caller `op.next_pc`; the guard pc is the merge point itself (`op.pc`).
fn surface_carrier_or_inline_subloop(
    token: Option<std::sync::Arc<majit_backend::JitCellToken>>,
    target_pc: usize,
    carrier_resume: bool,
    op_pc: usize,
    op_next_pc: usize,
) -> (DispatchOutcome, usize) {
    (
        DispatchOutcome::SubLoopCalleeCallAssembler { token, target_pc },
        if carrier_resume { op_pc } else { op_next_pc },
    )
}

fn trace_too_long_blackhole_snapshot_safe(outcome: &DispatchOutcome) -> bool {
    matches!(outcome, DispatchOutcome::Continue)
}

fn trace_too_long_abort_safe(
    outcome: &DispatchOutcome,
    blackhole_latched: bool,
    executed_effects: usize,
) -> bool {
    trace_too_long_blackhole_snapshot_safe(outcome) && (blackhole_latched || executed_effects == 0)
}

/// Errors surfaced by the trace-side walker.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DispatchError {
    /// The opcode byte at `pc` is not present in the `insns` table or
    /// the instruction's operand bytes overflowed the code slice. This
    /// is the same `decode_op_at -> None` path surfaced as a typed
    /// error.
    UndecodableOpcode { pc: usize },
    /// The opcode is decodable but the dispatcher has no handler for
    /// it yet. Carries the `opname/argcodes` key so callers can
    /// identify what blocked the walk.
    UnsupportedOpname { pc: usize, key: &'static str },
    /// A register operand byte indexed past the symbolic register file.
    /// `len` is the slice length the walker was handed in the
    /// `WalkContext`; `reg` is the byte the codewriter emitted. Surfaces
    /// either an assembler-pass bug (out-of-range register) or a
    /// caller mismatch between the symbolic register layout and the
    /// arm's expected number of registers.
    RegisterOutOfRange {
        pc: usize,
        reg: usize,
        len: usize,
        bank: &'static str,
    },
    /// A register operand byte indexed a slot the walk never wrote.
    /// `MIFrame.registers_*` start as `[None] * num_regs`
    /// (`pyjitpl.py`) and a jitcode only names a register the
    /// codewriter's liveness proves is assigned on every path reaching the
    /// read, so upstream never observes the hole.  Pyre's codewriter can
    /// still emit one — an exception edge whose merge block reads a
    /// value-stack slot none of its predecessors renames into — and the
    /// `OpRef::NONE` that read yields is not a box: recording it produces an
    /// op argument no backend can bind (`regalloc.py` `env[box]`
    /// KeyError; dynasm `RegisterManager.loc`, cranelift `resolve_opref`).
    /// Decline the walk at the read instead of carrying the hole into the
    /// trace.
    RegisterReadUnbound {
        pc: usize,
        reg: usize,
        bank: &'static str,
    },
    /// A `d`-coded descr index resolved past the descr pool. Surfaces
    /// either an assembler-pass bug (descr index out of range) or a
    /// caller mismatch between the codewriter's descr table size and
    /// the table the walker was handed in `WalkContext::descr_refs`.
    DescrIndexOutOfRange { pc: usize, index: usize, len: usize },
    /// `inline_call_*` resolved a descr that does not implement
    /// `JitCodeDescr`. Surfaces either a codewriter bug (an
    /// `inline_call_*` opnum emitted with a non-jitcode descr index)
    /// or a caller mismatch (the descr pool wasn't built from the
    /// codewriter's descr table). `descr_index` is the 2-byte LE
    /// index the walker decoded.
    ExpectedJitCodeDescr { pc: usize, descr_index: usize },
    /// `inline_call_*`'s descr resolved to a `jitcode_index`, but the
    /// caller's `sub_jitcode_lookup` returned `None`. Production wires
    /// the lookup to the lazy runtime jitcode cache; tests
    /// build synthetic maps. A `None` return means the codewriter
    /// emitted an index past the runtime's jitcode table.
    SubJitCodeNotFound { pc: usize, jitcode_index: usize },
    /// `inline_call_*` provided more Ref args in its R-list than the
    /// callee declared `num_regs_r` slots. RPython parity: `pyjitpl.py
    /// MIFrame.setup_call(argboxes)` distributes argboxes into the
    /// callee's typed register banks; the JitCode-level shape contract
    /// (`assembler.py:write_call`) requires `len(argboxes) <=
    /// num_regs_r` for the `_r_r` variant. Excess args are a
    /// codewriter-emitted shape mismatch.
    InlineCallArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_r: usize,
    },
    /// `inline_call_*` provided more Int args in its I-list than the
    /// callee declared `num_regs_i` slots. Same shape contract as the
    /// Ref variant — `pyjitpl.py setup_call` populates each
    /// kind-bank from its respective list and asserts capacity.
    InlineCallIntArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_i: usize,
    },
    /// `inline_call_*` provided more Float args in its F-list than the
    /// callee declared `num_regs_f` slots.
    InlineCallFloatArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_f: usize,
    },
    /// Internal yield from one heap-owned sub-walk frame to the explicit
    /// sub-walk driver.  It is consumed before ordinary abort handling and can
    /// never escape [`run_sub_jitcode_walk`].  RPython's `_interpret` gets the
    /// same transition by changing `framestack[-1]` in its single driver loop.
    SubWalkSuspended { pc: usize },
    /// `inline_call_r_r/dR>r`'s callee surfaced
    /// `SubReturn { result: None }`. RPython parity: the `_r_r` variant
    /// is wired (in `assembler.py:gen_inline_call`) to a callee whose
    /// `*_return` op carries a Ref; reaching it without a result means
    /// the callee body executed `void_return/` (or an analogue) instead
    /// of `ref_return/r`, which is a codewriter shape mismatch — the
    /// caller has nowhere to land the missing value.
    UnexpectedVoidSubReturn { pc: usize },
    /// `inline_call_*_v/d{R,IR,IRF}`'s callee surfaced
    /// `SubReturn { result: Some(_) }`. RPython parity: the `_v` variant
    /// (`bhimpl_inline_call_*_v`, `blackhole.py`) is
    /// wired to a callee whose `*_return` op is `void_return/`; reaching
    /// it with a typed-return result means the callee body executed
    /// `int_return/i` / `ref_return/r` / `float_return/f` instead of
    /// `void_return/`, which is a codewriter shape mismatch — the
    /// caller has no `>X` slot to land the surplus value.
    UnexpectedNonVoidSubReturn { pc: usize },
    /// `reraise/` fired but `WalkContext::last_exc_value` was `None`.
    /// RPython parity: `pyjitpl.py
    /// opimpl_reraise: assert self.metainterp.last_exc_value` —
    /// reaching `reraise` without an active exception is a codewriter
    /// invariant violation (`raise` or a catch-handler entry must have
    /// set `last_exc_value` first).
    ReraiseWithoutLastExcValue { pc: usize },
    /// `last_exc_value/>r` fired but `WalkContext::last_exc_value` was
    /// `None`. RPython parity: `pyjitpl.py opimpl_last_exc_value`:
    ///
    ///   exc_value = self.metainterp.last_exc_value
    ///   assert exc_value
    ///   return self.metainterp.last_exc_box
    ///
    /// Same codewriter invariant as `reraise/`: this opname only emits
    /// inside a `catch_exception` body where the unwinder has already
    /// stored the in-flight exception. Reaching it without an active
    /// exception is a flatten/codewriter shape mismatch.
    LastExcValueWithoutActiveException { pc: usize },
    /// `catch_exception/L` was reached on the normal fall-through path
    /// (no `SubRaise` routing) but `WalkContext::last_exc_value` was
    /// non-`None`. RPython parity: `pyjitpl.py opimpl_catch_exception`:
    ///
    ///   assert not self.metainterp.last_exc_value
    ///
    /// On the normal path the previous instruction did NOT raise — if
    /// it had, `finishframe_exception` would have routed control past
    /// the catch_exception/L (or to its target if matched), never
    /// running the catch_exception/L instruction itself. Reaching it
    /// with an active exception means the codewriter mis-emitted a
    /// catch_exception/L outside an exception-table position, OR a
    /// previous handler forgot to clear `last_exc_value` after handling.
    CatchExceptionWithActiveException { pc: usize },

    /// `residual_call_*` decoded a descr that does not implement
    /// `CallDescr`. RPython parity: `pyjitpl.py
    /// do_residual_call` always receives a `calldescr` from the
    /// codewriter — there is no fallback path. The walker mirrors that
    /// invariant by surfacing a typed error when the descr_pool entry
    /// at the operand-encoded index lacks a CallDescr downcast. In
    /// production the codewriter never emits a non-CallDescr; this
    /// variant fires only when test fixtures (or future deviations)
    /// route a non-CallDescr into a residual_call slot.
    ResidualCallDescrNotCallDescr { pc: usize, descr_index: usize },
    /// A residual_call argument resolved to `OpRef::NONE` — an unbound
    /// register.  RPython's `do_residual_call` reads every argbox out of
    /// `env[box]`, so a missing binding is a `KeyError` there; the
    /// cranelift/dynasm backends mirror that as a hard `resolve_opref`
    /// assert when `OpRef::NONE` reaches arg-resolution.  A per-opcode
    /// arm whose body forwards an unseeded dispatcher parameter (e.g. a
    /// seam wrapper that residualizes `execute_<op>(executor, code,
    /// instruction, op_arg)` — only `executor`=r0 is seeded at the arm
    /// entry, so `code`/`instruction`/`op_arg` stay `OpRef::NONE`) would
    /// record such a call.  Surfacing it here turns the would-be backend
    /// crash into a graceful trace abort, matching the pre-seam inline
    /// arm whose payload read aborted with `GotoIfNotValueNotConcrete`.
    ResidualCallArgUnbound { pc: usize, arg_index: usize },
    /// A `getfield_gc_*` cache-miss would record an op whose `FieldDescr`
    /// carries no `get_parent_descr()` backreference.  The optimizer's
    /// `ensure_ptr_info_arg0` (`optimizer.py`) panics rather than
    /// install a malformed PtrInfo for such a descr, so — like
    /// `ResidualCallArgUnbound` — surface it here to turn the would-be
    /// optimizer crash into a graceful trace abort.  Production sub-walks
    /// never reach this (they would already panic); it fires only on the
    /// orthodox `w_list_append` descent over a descr (e.g.
    /// `W_ListObject.strategy`) not yet resolved to its parent descr group.
    FieldDescrMissingParentDescr { pc: usize },
    /// The orthodox `w_list_append` body sub-walk descended the full append
    /// (past the strategy switch + `is_plain_int1`, descr-pool resolved) and
    /// the trace compiles, but the guards it records resume through side-exit
    /// bridges whose reconstruction is wrong — executing the compiled loop
    /// jumps to garbage.  This is the sub-walk guard resume/faillocs epic
    /// (#62/#73/#34) the hand-rolled fold sidesteps with explicit
    /// `walker_capture_snapshot_for_last_guard` snapshots.  Surfaced here to
    /// abort the trace gracefully (interpreter fallback) instead of committing
    /// a trace that SIGSEGVs on its first side exit.  Default OFF.
    OrthodoxSubWalkTraceUnsupported { pc: usize, symbolic: i64 },
    /// A `new` descriptor cannot resolve to a type id present in the active
    /// collector's registered type table. The walker must stop before either
    /// executing or recording the allocation: both paths would stamp the same
    /// invalid header. This error deliberately keeps legacy entry replay;
    /// blackhole resume at this opcode would use the same descriptor fallback.
    UnregisteredNewGcType { pc: usize, type_id: u64 },
    /// The LIST_APPEND opcode's void `jit_list_append` residual
    /// (`ListAppendValue`) reached the authoritative full-body walker but the
    /// orthodox `w_list_append` fold declined (the list needs a resize, or the
    /// strategy/value is unsupported — `orthodox_list_append_recognize`
    /// returned `None`) or the fold is disabled.  Unlike the fold's
    /// `orthodox_list_append_commit`, the generic residual dispatcher
    /// concrete-executes `jit_list_append` WITHOUT `fbw_list_journal_push_append`,
    /// so `fbw_store_journal_rollback` cannot rewind it; a later trace abort +
    /// interpreter replay would then apply the SAME append twice. Decline the
    /// trace to interpretation instead of falling through, mirroring the
    /// pre-#171 `emit_abort_permanent` LIST_APPEND lowering — the common
    /// int/float/object append still folds; only the decline shapes abort, and
    /// they aborted before #171 too.
    UnfoldableListAppendResidualUnsupported { pc: usize },
    /// `switch/id` decoded a descr that does not implement
    /// `SwitchDescr`. RPython parity: `pyjitpl.py` asserts
    /// `isinstance(switchdict, SwitchDictDescr)`.
    ExpectedSwitchDescr { pc: usize },
    /// `switch/id` needs RPython's `valuebox.getint()` at trace time.
    /// The symbolic walker can obtain that only when `TraceCtx` can
    /// reconstruct an Int concrete for the OpRef today;
    /// choosing a branch without a concrete value would record the wrong
    /// guard chain, so surface the missing concrete value explicitly.
    SwitchValueNotConcrete { pc: usize, value: OpRef },
    /// `goto_if_not/iL` needs RPython's `box.getint()` at trace time
    /// (`pyjitpl.py opimpl_goto_if_not`).  Without the
    /// concrete value the walker can't pick GUARD_TRUE vs GUARD_FALSE
    /// or decide whether to jump to the label target, so surface the
    /// missing concrete explicitly instead of guessing.
    GotoIfNotValueNotConcrete { pc: usize, value: OpRef },
    /// `OS_NOT_IN_TRACE` must run the callee concretely and record no
    /// IR on the normal path (`pyjitpl.py`). The standalone
    /// symbolic walker has no concrete executor, so it must stop here
    /// instead of faking either the normal return or
    /// `SwitchToBlackhole`.
    NotInTraceRequiresConcreteExecution { pc: usize },
    /// `OS_JIT_FORCE_VIRTUAL` would short-circuit `do_residual_call`
    /// before recording `CALL_MAY_FORCE_*` (`pyjitpl.py →
    /// 2153-2172 _do_jit_force_virtual`). The short-circuit needs a
    /// concrete `vref_ptr` for arbitrary Ref OpRefs to determine
    /// `isstandard_int` at trace time — walker only knows
    /// `concrete_vable_ptr`, not the concrete value behind every Ref
    /// OpRef. Surfacing this as an error prevents silently recording
    /// `CALL_MAY_FORCE_*` for an op the live tracer would have folded
    /// to `vref_box` / `standard_box` / None. Production reach today:
    /// `OopSpecIndex::JitForceVirtual` is set only by
    /// `jtransform.rs jit.force_virtual` lowering, which our
    /// benchmarks don't trigger; this guard is fail-loud against future
    /// silent TODOs.
    JitForceVirtualRequiresConcreteResolver { pc: usize },
    /// `{get,set}field_vable_{i,r,f}` read its box operand from a Ref
    /// register holding `OpRef::None`. RPython parity: `pyjitpl.py
    /// opimpl_getfield_vable_{i,r,f}` / `_opimpl_setfield_vable(box, ...,
    /// pc)` always receive a live `box` from the register file — the
    /// virtualizable frame box. Pyre's walker initializes Ref register
    /// slots to `OpRef::None`; an inlined callee frame may leave the vable
    /// register unseeded (documented walker arg-seeding gap). Both vable
    /// accessors route through `begin_nonstandard_virtualizable` →
    /// `heapcache.nonstandard_virtualizables_now_known(box)`, which would
    /// feed `OpRef::None` (`raw() == u32::MAX`) into the dense heapcache
    /// flag `Vec<u32>`, resizing it to `u32::MAX + 1` (16 GiB). Surface as
    /// a trace abort so production resumes in the interpreter rather than
    /// allocating; no compiled trace is produced.
    VableBoxNotSeeded { pc: usize },
    /// `{get,set}arrayitem_vable_*` / `arraylen_vable` decoded its
    /// `(VableArray, Array)` descr-pool pair but one of the two slots
    /// was missing or held the wrong `BhDescr` variant. RPython parity:
    /// `MIFrame.vable_array_index_pair_at` (`blackhole.rs`)
    /// asserts the exact `(VableArray, Array)` pair and panics otherwise
    /// — a malformed pair is a flatten/codewriter shape bug. The walker
    /// surfaces it as a fail-loud abort instead of crashing the tracer.
    VableArrayDescrMalformed {
        pc: usize,
        field_idx: usize,
        array_idx: usize,
    },
    /// `{get,set}arrayitem_vable_*` / `arraylen_vable` reached the vable
    /// resolution path but `TraceCtx::virtualizable_info()` was `None`.
    /// RPython parity: these opnames only emit for the jitdriver's
    /// virtualizable (`pyjitpl.py`); a missing
    /// `virtualizable_info` means the descr pair pointed at an array
    /// field of a struct that is not the registered virtualizable, which
    /// is a codewriter shape mismatch.
    VableArrayMissingVirtualizableInfo { pc: usize },
    /// The `VableArray.index` decoded from the descr pair indexed past
    /// `VirtualizableInfo::array_field_descrs` / `array_descrs`. Surfaces
    /// a codewriter/virtualizable-layout mismatch between the emitted
    /// array-field index and the registered virtualizable's array fields.
    VableArrayIndexOutOfRange { pc: usize, index: usize },
    /// `{get,set}arrayitem_vable_*` needs RPython's `indexbox.getint()`
    /// at trace time (`pyjitpl.py _get_arrayitem_vable_index`
    /// calls `implement_guard_value(indexbox, pc)` then `indexbox.getint()`).
    /// The symbolic walker can resolve that only when the index OpRef has
    /// a concrete Int; without it the array slot can't be chosen, so
    /// surface the missing concrete explicitly instead of guessing slot 0.
    VableArrayIndexNotConcrete { pc: usize, value: OpRef },
    /// `abort/` (BC_ABORT) reached. The front-end emits this marker for a
    /// graph node with no dedicated `OpKind`; reaching it during a body
    /// walk means the trace contains an untranslatable op and cannot be
    /// recorded. Blackhole counterpart `handler_abort_marker`
    /// (`blackhole.rs`) sets `aborted = true` + `LeaveFrame`; the
    /// walker surfaces it as a typed abort that the production driver
    /// maps to `TraceAction::Abort` (recoverable — may retry later).
    AbortMarkerReached { pc: usize },
    /// An inlined callee's keyword-only default mapping moved between the
    /// record-time resolve and the marker that watches it.
    /// `kwonly_defaults_for_inline` reads each entry's cell up front, and the
    /// emit installs `record_quasiimmut_field` on the strategy version only
    /// once the inline is committed to; a store from another thread in that
    /// window bumps a version no watcher is registered against yet, so the
    /// seeding would bake the pre-store cell and never be told.  The emit is
    /// past the point it can decline from -- the guards for this inline are
    /// already recorded -- so the walk gives the trace up instead.  The
    /// sibling `emit_namespace_cell_fold` answers the same window with a
    /// plain decline, which is why only this site needs a class of its own.
    KwonlyDefaultsMappingRacedRecord { pc: usize },
    /// Record-time construction of a concrete shadow object failed after the
    /// specialization had emitted IR. Falling through to the generic residual
    /// would retain that partial prologue, so abort the whole walk.
    ConcreteShadowAllocationFailed { pc: usize },
    /// `abort_permanent/` (BC_ABORT_PERMANENT) reached. pyre's codegen
    /// emits this for fail-paths that must always terminate the frame
    /// (e.g. BigInt-overflow / unported-op fallbacks). Blackhole
    /// counterpart `bhimpl_abort_permanent` (`blackhole.rs`) routes a
    /// TLS-stashed exception or sets `aborted`; during a trace walk no
    /// blackhole TLS exists, so the walker surfaces a typed permanent
    /// abort that the production driver maps to `TraceAction::AbortPermanent`
    /// (never trace this location again).
    AbortPermanentMarkerReached { pc: usize },
    /// A result-bearing may-force CALL (`CallMayForce{R,I,F}`, a Python
    /// function-entry call) was recorded with a concrete-NULL Ref
    /// argument. This is the specialized direct-call shape: the callee
    /// was folded to its entry address and the `PUSH_NULL` self-slot is
    /// baked as `ptr(0x0)`. `try_execute_residual_call_via_executor` skips
    /// such a call (its NULL-receiver SEGV guard), so the result stays
    /// unresolved, and the baked NULL arg makes the compiled call pass
    /// NULL where the entry needs the callee's globals/closure — yielding
    /// a NULL result at runtime (closures / functions bound to a local
    /// and called in a loop). The walker surfaces a typed abort so the
    /// driver maps it to `TraceAction::Abort` and the loop falls back to
    /// the interpreter instead of compiling a wrong trace.
    MayForceNullRefArgUnsupported { pc: usize },
    /// The virtualizable escaped during a concrete-executed may-force
    /// residual call: `vinfo.tracing_after_residual_call(virtualizable)`
    /// found the token cleared by the callee's force path
    /// (pyjitpl.py `vable_after_residual_call` →
    /// `SwitchToBlackhole(Counters.ABORT_ESCAPE, raising_exception=True)`).
    /// The trace can no longer treat the frame as a virtualizable, so the
    /// walk aborts and the interpreter resumes from the (now heap-
    /// authoritative) frame. Soft abort — escape is data-dependent, the
    /// same location may trace cleanly later.
    VableEscapedDuringResidualCall { pc: usize },
    /// A walker-emitted guard needs a resume snapshot, but the live
    /// virtualizable box list carries an untyped entry (typically the
    /// identity box `[-1]` of a deeper inlined / recursive frame). Building
    /// the snapshot would trip the `build_vable_snapshot_boxes`
    /// `OpRef::ty()` invariant. The full-body walk surfaces a typed abort so
    /// the driver maps it to `TraceAction::Abort` and interpretation instead
    /// of panicking the tracer. Resuming such a guard needs the multi-frame
    /// vable snapshot machinery.
    GuardSnapshotVableUntyped { pc: usize },
    /// A guard capture had no decodable carried JitCode coordinate. Publishing
    /// a Python pc here would make a later side exit resume at a guessed
    /// block head, so the walker declines before the trace is installed.
    GuardResumeCoordinateUnavailable { pc: usize },
    /// `last_exception/>i` fired but no concrete standing exception was
    /// available. RPython parity: `pyjitpl.py opimpl_last_exception`:
    ///
    ///   exc_value = self.metainterp.last_exc_value
    ///   assert exc_value
    ///   assert self.metainterp.class_of_last_exc_is_const
    ///
    /// The class pointer is read from the concrete exception's
    /// `ob_header.ob_type`, so a missing `last_exc_value` OpRef or a
    /// `Null`/non-Ref `last_exc_value_concrete` violates the same
    /// codewriter invariant as [`LastExcValueWithoutActiveException`]:
    /// this opname only emits inside a `catch_exception` body where the
    /// unwinder has already stored the in-flight exception.
    LastExceptionWithoutActiveException { pc: usize },
    /// `jit_merge_point/cIRFIRF` could not derive its green key from the
    /// op's green operands. pyre's portal jitdriver greens =
    /// `[next_instr, is_being_profiled, pycode]` (`eval.rs`), so the
    /// green key is `make_green_key` at `(concrete(gr[0]=pycode),
    /// concrete(gi[0]=next_instr))`, with the walk's own
    /// `is_being_profiled`. This fires when the int/ref green
    /// list is empty or its leading element has no concrete Int/Ref —
    /// either a codewriter shape mismatch or (pre-Phase-5) the greens
    /// weren't seeded with concretes. RPython parity:
    /// `pyjitpl.py get_procedure_token(greenboxes)` always receives
    /// concrete greens at a reached merge point.
    JitMergePointGreenKeyUnresolved { pc: usize },
    /// The portal frame reached its merge point with a tracer already armed
    /// (`debugdata.w_f_trace` non-NULL). `dispatch_bytecode`'s jitted arm
    /// (pyopcode.py) answers that state with `ec.bytecode_only_trace(self)`
    /// on every opcode; the walker has no route to record that call, so a
    /// trace taken here would compile a loop that owes `line` events and
    /// cannot fire them. Decline and leave the frame to the interpreter.
    PortalFrameTracerArmed { pc: usize },
    /// `loop_header/i` or `jit_merge_point/iIRFIRF` could not resolve its
    /// jdindex operand to a concrete Int. The assembler encodes the jdindex as a
    /// populated int-constant-pool slot (`assembler.rs loop_header`:
    /// `add_const_i` + patch at `finish()`), so an unresolved slot is a
    /// structural encoding bug, mirroring the `expect` on
    /// `frame.getint(ctx, slot)` in majit's `BC_LOOP_HEADER` arm
    /// (`pyjitpl/dispatch.rs`).
    LoopHeaderJdIndexUnresolved { pc: usize },
    /// An `inline_call_*` sub-walk surfaced `DispatchOutcome::CloseLoop`.
    /// `jit_merge_point` is the portal loop header; an inlined (non-portal)
    /// callee body should never reach one and close a loop. RPython parity:
    /// `_opimpl_inline_call*` (`pyjitpl.py`) inlines the callee
    /// into the SAME trace — recursive portal re-entry takes the
    /// `recursive_call` path, not `inline_call`. Reaching a loop-close
    /// inside an inline sub-walk is therefore a codewriter/flatten shape
    /// mismatch.
    SubWalkClosedLoop { pc: usize },
    /// A `goto_if_not` branch guard resumes at a target that still carries
    /// two or more live operand-stack temps (resume-target stack depth > 1)
    /// and the not-taken edge's kept-value recovery is incomplete.
    /// This is the multi-kept-temp short-circuit shape — `(x and y) or z`,
    /// chained comparison `a < b < c` — where CPython keeps more than one
    /// tested value on the value stack across the branch (`COPY` / `TO_BOOL`
    /// / `POP_JUMP_IF_*`). The full-body walk's single-frame guard snapshot
    /// rebuilds locals + the post-opcode operand stack from the live register
    /// banks; a single depth-1 kept temp is recovered from the walk-level box
    /// mirror (`vstack`) in `collect_outer_active_boxes`.  Depth > 1 is
    /// supported only when the not-taken edge's decoded `ref_copy` moves
    /// (`#420`) cover every distinct kept resume color AND each move's source
    /// resolves to a live register value; that resolved set then drives the
    /// snapshot encoder.  This abort fires only when that recovery is
    /// incomplete — a kept slot the edge does not rename ("live-across"), a
    /// cyclic `*_push`/`*_pop` move, a truncated/under-sized color map, or a
    /// source that resolves to `OpRef::NONE` — i.e. when the deopt re-entry
    /// would otherwise restore a wrong value into a loop-carried slot
    /// (per-PC resume-value precision).  Plain `while` / `if`
    /// branches resume at depth 0 and are unaffected.  The walker surfaces a
    /// typed abort so the driver maps it to `TraceAction::Abort` →
    /// interpreter fallback (correct, untraced) instead of compiling a trace
    /// whose guard-failure path corrupts the frame.
    BranchGuardKeptStackUnsupported { pc: usize },
    /// A branch guard's snapshot has a live Ref color whose operand-stack slot
    /// has no per-slot source: the walk mirror holds `OpRef::NONE` there and
    /// the decoded edge-move recovery carries no entry for that color.
    /// The remaining fallback reads
    /// the resume merge color out of the guard-pc register file, where it is
    /// unwritten at the guard point or reused by the regalloc — the #424
    /// merge-color staleness the per-slot mirror replaced.  The gate that lets
    /// a valid mirror through justified itself by that recovery, so where the
    /// recovery does not in fact cover the slot the capture declines instead.
    BranchGuardKeptSlotUnsourced { pc: usize },
    /// A kept-stack branch guard's not-taken (resume) arm READS a regular
    /// Ref register (`< num_regs_r`) that is neither snapshot-live nor
    /// produced inside the arm, so the blackhole resumes it as NULL and
    /// feeds NULL into the consuming op — the boxed-int short-circuit /
    /// conditional-expression resume miscompile (a heap `ConstPtr`, an int
    /// outside the 1-byte immediate range `[0, 256)`, parked in a register
    /// live-across the guard).  The driver maps this to
    /// `TraceAction::AbortPermanent` → `DONT_TRACE_HERE`.  Demoting it to a
    /// plain [`TraceAction::Abort`] would still reach the same terminal state
    /// through pyre's abort ceiling, so the permanent mapping records the
    /// structural nature of this abort without changing runtime behavior.
    BranchGuardUnrestorableKeptStackPermanent { pc: usize },
    /// `opimpl_jit_force_quasi_immutable`: the watcher was installed when the
    /// walk reached a write that invalidates it, so the tracer forces the
    /// watcher and raises `SwitchToBlackhole(ABORT_FORCE_QUASIIMMUT)`.
    ///
    /// `pc` is the next operation to execute after that synthetic marker.
    ForceQuasiImmutable { pc: usize },
    /// A residual call to a pure-Python callee that is inline-eligible
    /// (plain, exact-positional, closure-free, not recursion-bound) but
    /// whose body is NOT a straight-line leaf — it carries an internal loop
    /// or branch (`goto_if_not` / `switch`) or a non-static `vable` op, so
    /// the fast-path register-seeding inline (`try_walker_inline_user_call`)
    /// declines.  Emitting the residual leaves the callee re-interpreted per
    /// iteration (and its short inner loops compile + deopt-storm), strictly
    /// slower than interpreting. The retired MIFrame tracer inlined such
    /// callees via `push_inline_frame` + the `recursive-call-assembler` loop back-edge
    /// (`pyjitpl.rs` opimpl_recursive_call_assembler).  Surface a typed
    /// abort so the key interprets without JIT (`FBW_DECLINED_KEYS`) until
    /// the walker itself covers loop-callee inlining.
    LoopBearingCalleeInlineUnsupported {
        pc: usize,
        /// Whether the aborting MIFrame has applied an effect since its caller
        /// entered it.  A clean attempted frame can be discarded and re-entered
        /// from the CALL; an applied frame must be preserved and run forward.
        blackhole_required: bool,
    },
    /// An in-flight FOR_ITER body executed a non-journalable
    /// in-place builtin-container mutation (`acc += delta` for an object-/float-
    /// strategy list, `bytearray`, `set`, `dict`, …).  The abort rollback cannot
    /// rewind it and a deliver re-run would double it, so the walk declines BEFORE
    /// the commit and the location interprets permanently (`AbortPermanent`).
    InplaceContainerMutationUnsupported { pc: usize },
    /// Exception-edge bridge: `call_jit` routed the exc edge (published the
    /// standing exception and declined to hand the guard to the blackhole), but
    /// this frame carries no `catch_exception` covering the raise — a state the
    /// routing precondition is supposed to exclude, since a routed walk must
    /// resume at a handler rather than fall through to the NULL raised-call
    /// result.  Abort so the guard failure resumes via the blackhole instead.
    /// Where the handler LEADS does not reach this error: a handler that returns
    /// out of the frame routes like any other (`pyjitpl.py`
    /// `finishframe_exception` jumps to the catch unconditionally, and the
    /// return is `finishframe`'s ordinary case, `pyjitpl.py`).
    ExcEdgeNoInFrameCatch { pc: usize },
    /// The recorded trace passed `warmstate.trace_limit` (`rlib/jit.py:592`
    /// default 6000). RPython checks this after every traced step —
    /// `pyjitpl.py _interpret` calls `blackhole_if_trace_too_long()` right
    /// after `run_one_step()` — and raises `SwitchToBlackhole(ABORT_TOO_LONG)`.
    TraceTooLong { pc: usize, ops: usize },
    /// A segmented-trace cut had already latched its complete forward
    /// blackhole image, but the always-failing guard's resume snapshot could
    /// not be represented. The incomplete segment must be discarded, while
    /// the latched image must still run forward: replaying from the trace entry
    /// would repeat the concrete effects the walk already executed. `cause`
    /// preserves the snapshot builder's original decline class for diagnosis.
    SegmentTraceSnapshotUnavailable { pc: usize, cause: &'static str },
    /// The traceback node's `last_instr` store could not resolve the
    /// virtualizable static field named `field`: either the trace carries no
    /// `VirtualizableInfo` at all, or the registered one declares no such
    /// field.  `interp_jit.py:25-30` declares `last_instr` virtualizable and
    /// `pyjitpl.py _opimpl_setfield_vable` routes the write through
    /// it, so an unresolvable descr means the walk is recording against a
    /// frame layout the jitdriver never registered.  Abort rather than fall
    /// back to a raw SETFIELD_GC, which writes the root frame's shadow on
    /// every caught-exception bridge.
    TracebackNodeVableFieldUnavailable { pc: usize, field: &'static str },
}

impl DispatchError {
    /// The variant's identifier as a `'static` string, for the
    /// decline-class census (see [`census_record`]).  One arm per
    /// variant so a new variant fails to compile until it is named here.
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            Self::UndecodableOpcode { .. } => "UndecodableOpcode",
            Self::UnsupportedOpname { .. } => "UnsupportedOpname",
            Self::RegisterOutOfRange { .. } => "RegisterOutOfRange",
            Self::RegisterReadUnbound { .. } => "RegisterReadUnbound",
            Self::DescrIndexOutOfRange { .. } => "DescrIndexOutOfRange",
            Self::ExpectedJitCodeDescr { .. } => "ExpectedJitCodeDescr",
            Self::SubJitCodeNotFound { .. } => "SubJitCodeNotFound",
            Self::InlineCallArityMismatch { .. } => "InlineCallArityMismatch",
            Self::InlineCallIntArityMismatch { .. } => "InlineCallIntArityMismatch",
            Self::InlineCallFloatArityMismatch { .. } => "InlineCallFloatArityMismatch",
            Self::SubWalkSuspended { .. } => "SubWalkSuspended",
            Self::UnexpectedVoidSubReturn { .. } => "UnexpectedVoidSubReturn",
            Self::UnexpectedNonVoidSubReturn { .. } => "UnexpectedNonVoidSubReturn",
            Self::ReraiseWithoutLastExcValue { .. } => "ReraiseWithoutLastExcValue",
            Self::LastExcValueWithoutActiveException { .. } => "LastExcValueWithoutActiveException",
            Self::CatchExceptionWithActiveException { .. } => "CatchExceptionWithActiveException",
            Self::ResidualCallDescrNotCallDescr { .. } => "ResidualCallDescrNotCallDescr",
            Self::ResidualCallArgUnbound { .. } => "ResidualCallArgUnbound",
            Self::ExpectedSwitchDescr { .. } => "ExpectedSwitchDescr",
            Self::SwitchValueNotConcrete { .. } => "SwitchValueNotConcrete",
            Self::GotoIfNotValueNotConcrete { .. } => "GotoIfNotValueNotConcrete",
            Self::NotInTraceRequiresConcreteExecution { .. } => {
                "NotInTraceRequiresConcreteExecution"
            }
            Self::JitForceVirtualRequiresConcreteResolver { .. } => {
                "JitForceVirtualRequiresConcreteResolver"
            }
            Self::VableBoxNotSeeded { .. } => "VableBoxNotSeeded",
            Self::VableArrayDescrMalformed { .. } => "VableArrayDescrMalformed",
            Self::VableArrayMissingVirtualizableInfo { .. } => "VableArrayMissingVirtualizableInfo",
            Self::VableArrayIndexOutOfRange { .. } => "VableArrayIndexOutOfRange",
            Self::VableArrayIndexNotConcrete { .. } => "VableArrayIndexNotConcrete",
            Self::AbortMarkerReached { .. } => "AbortMarkerReached",
            Self::KwonlyDefaultsMappingRacedRecord { .. } => "KwonlyDefaultsMappingRacedRecord",
            Self::ConcreteShadowAllocationFailed { .. } => "ConcreteShadowAllocationFailed",
            Self::AbortPermanentMarkerReached { .. } => "AbortPermanentMarkerReached",
            Self::MayForceNullRefArgUnsupported { .. } => "MayForceNullRefArgUnsupported",
            Self::VableEscapedDuringResidualCall { .. } => "VableEscapedDuringResidualCall",
            Self::GuardSnapshotVableUntyped { .. } => "GuardSnapshotVableUntyped",
            Self::GuardResumeCoordinateUnavailable { .. } => "GuardResumeCoordinateUnavailable",
            Self::LastExceptionWithoutActiveException { .. } => {
                "LastExceptionWithoutActiveException"
            }
            Self::JitMergePointGreenKeyUnresolved { .. } => "JitMergePointGreenKeyUnresolved",
            Self::PortalFrameTracerArmed { .. } => "PortalFrameTracerArmed",
            Self::LoopHeaderJdIndexUnresolved { .. } => "LoopHeaderJdIndexUnresolved",
            Self::SubWalkClosedLoop { .. } => "SubWalkClosedLoop",
            Self::BranchGuardKeptStackUnsupported { .. } => "BranchGuardKeptStackUnsupported",
            Self::BranchGuardKeptSlotUnsourced { .. } => "BranchGuardKeptSlotUnsourced",
            Self::ForceQuasiImmutable { .. } => "ForceQuasiImmutable",
            // Split on the discriminant: `true` converts through WalkAbort
            // when frames are materialized; `false` without a seeded frame
            // is residual-not-abort.  One key cannot answer which fired.
            Self::LoopBearingCalleeInlineUnsupported {
                blackhole_required: true,
                ..
            } => "LoopBearingCalleeInlineUnsupported/blackhole",
            Self::LoopBearingCalleeInlineUnsupported { .. } => {
                "LoopBearingCalleeInlineUnsupported/carrier"
            }
            Self::FieldDescrMissingParentDescr { .. } => "FieldDescrMissingParentDescr",
            Self::OrthodoxSubWalkTraceUnsupported { .. } => "OrthodoxSubWalkTraceUnsupported",
            Self::UnregisteredNewGcType { .. } => "UnregisteredNewGcType",
            Self::UnfoldableListAppendResidualUnsupported { .. } => {
                "UnfoldableListAppendResidualUnsupported"
            }
            Self::BranchGuardUnrestorableKeptStackPermanent { .. } => {
                "BranchGuardUnrestorableKeptStackPermanent"
            }
            Self::InplaceContainerMutationUnsupported { .. } => {
                "InplaceContainerMutationUnsupported"
            }
            Self::ExcEdgeNoInFrameCatch { .. } => "ExcEdgeNoInFrameCatch",
            Self::TraceTooLong { .. } => "TraceTooLong",
            Self::SegmentTraceSnapshotUnavailable { .. } => "SegmentTraceSnapshotUnavailable",
            Self::TracebackNodeVableFieldUnavailable { .. } => "TracebackNodeVableFieldUnavailable",
        }
    }

    /// The jitcode coordinate the walk stopped at.  Every variant records the
    /// `pc` of the instruction that could not be walked, and none of them ran
    /// that instruction's arm, so this doubles as the resume coordinate a
    /// blackhole conversion has to `setposition` to (`blackhole.py`
    /// `copy_data_from_miframe` reads each level's `frame.pc` the same way).
    /// One arm per variant so a new variant fails to compile until it says
    /// where it stopped.
    pub(crate) fn stop_pc(&self) -> usize {
        match self {
            Self::UndecodableOpcode { pc, .. }
            | Self::UnsupportedOpname { pc, .. }
            | Self::RegisterOutOfRange { pc, .. }
            | Self::RegisterReadUnbound { pc, .. }
            | Self::DescrIndexOutOfRange { pc, .. }
            | Self::ExpectedJitCodeDescr { pc, .. }
            | Self::SubJitCodeNotFound { pc, .. }
            | Self::InlineCallArityMismatch { pc, .. }
            | Self::InlineCallIntArityMismatch { pc, .. }
            | Self::InlineCallFloatArityMismatch { pc, .. }
            | Self::SubWalkSuspended { pc, .. }
            | Self::UnexpectedVoidSubReturn { pc, .. }
            | Self::UnexpectedNonVoidSubReturn { pc, .. }
            | Self::ReraiseWithoutLastExcValue { pc, .. }
            | Self::LastExcValueWithoutActiveException { pc, .. }
            | Self::CatchExceptionWithActiveException { pc, .. }
            | Self::ResidualCallDescrNotCallDescr { pc, .. }
            | Self::ResidualCallArgUnbound { pc, .. }
            | Self::ExpectedSwitchDescr { pc, .. }
            | Self::SwitchValueNotConcrete { pc, .. }
            | Self::GotoIfNotValueNotConcrete { pc, .. }
            | Self::NotInTraceRequiresConcreteExecution { pc, .. }
            | Self::JitForceVirtualRequiresConcreteResolver { pc, .. }
            | Self::VableBoxNotSeeded { pc, .. }
            | Self::VableArrayDescrMalformed { pc, .. }
            | Self::VableArrayMissingVirtualizableInfo { pc, .. }
            | Self::VableArrayIndexOutOfRange { pc, .. }
            | Self::VableArrayIndexNotConcrete { pc, .. }
            | Self::AbortMarkerReached { pc, .. }
            | Self::ConcreteShadowAllocationFailed { pc, .. }
            | Self::AbortPermanentMarkerReached { pc, .. }
            | Self::KwonlyDefaultsMappingRacedRecord { pc, .. }
            | Self::MayForceNullRefArgUnsupported { pc, .. }
            | Self::VableEscapedDuringResidualCall { pc, .. }
            | Self::GuardSnapshotVableUntyped { pc, .. }
            | Self::GuardResumeCoordinateUnavailable { pc, .. }
            | Self::LastExceptionWithoutActiveException { pc, .. }
            | Self::JitMergePointGreenKeyUnresolved { pc, .. }
            | Self::PortalFrameTracerArmed { pc, .. }
            | Self::LoopHeaderJdIndexUnresolved { pc, .. }
            | Self::SubWalkClosedLoop { pc, .. }
            | Self::BranchGuardKeptStackUnsupported { pc, .. }
            | Self::BranchGuardKeptSlotUnsourced { pc, .. }
            | Self::ForceQuasiImmutable { pc, .. }
            | Self::LoopBearingCalleeInlineUnsupported { pc, .. }
            | Self::FieldDescrMissingParentDescr { pc, .. }
            | Self::OrthodoxSubWalkTraceUnsupported { pc, .. }
            | Self::UnregisteredNewGcType { pc, .. }
            | Self::UnfoldableListAppendResidualUnsupported { pc, .. }
            | Self::BranchGuardUnrestorableKeptStackPermanent { pc, .. }
            | Self::InplaceContainerMutationUnsupported { pc, .. }
            | Self::ExcEdgeNoInFrameCatch { pc, .. }
            | Self::TraceTooLong { pc, .. }
            | Self::SegmentTraceSnapshotUnavailable { pc, .. }
            | Self::TracebackNodeVableFieldUnavailable { pc, .. } => *pc,
        }
    }

    /// Whether this abort leaves the walk's MIFrame image COMPLETE — the walker
    /// stopped because it cannot MODEL the next operation, not because a value
    /// it needs is missing.
    ///
    /// Only the first family has an upstream counterpart.  Every RPython
    /// `SwitchToBlackhole` is a stop-tracing DECISION taken with the registers
    /// bound (`ABORT_TOO_LONG`, `ABORT_BRIDGE`, …), which is why
    /// `_copy_data_from_miframe` (`blackhole.py`) copies the banks
    /// unconditionally and has no failing path.  Pyre's walker resolves
    /// registers, descrs and concretes lazily against the live trace context,
    /// so a second family exists here that upstream has no analog for: the
    /// abort IS the report that a value is unavailable
    /// (`RegisterReadUnbound`, `SwitchValueNotConcrete`, `*ArgUnbound`, a
    /// malformed descr).  Converting one of those to a blackhole resumes
    /// execution on exactly the hole the walker refused to read, so those
    /// keep the legacy entry replay.
    ///
    /// An allow-list, not a deny-list: a class left out only forgoes the
    /// handoff, while a class wrongly let in resumes on missing data.
    pub(crate) fn leaves_complete_image(&self) -> bool {
        matches!(
            self,
            Self::UndecodableOpcode { .. }
                | Self::UnsupportedOpname { .. }
                | Self::OrthodoxSubWalkTraceUnsupported { .. }
                | Self::UnfoldableListAppendResidualUnsupported { .. }
                | Self::MayForceNullRefArgUnsupported { .. }
                | Self::InplaceContainerMutationUnsupported { .. }
                | Self::InlineCallArityMismatch { .. }
                | Self::InlineCallIntArityMismatch { .. }
                | Self::InlineCallFloatArityMismatch { .. }
                | Self::LoopBearingCalleeInlineUnsupported {
                    blackhole_required: true,
                    ..
                }
                // Forcing the watcher leaves every register bound.
                | Self::ForceQuasiImmutable { .. }
                // Emitted only after both segment-cut blackhole preflights
                // succeed. The missing guard snapshot affects compilation of
                // the discarded trace, not the complete forward image.
                | Self::SegmentTraceSnapshotUnavailable { .. }
        )
    }

    /// Modeling-gap abort that still leaves a complete MIFrame image when the
    /// inlined callee's `PyFrame` is already seeded.
    ///
    /// [`Self::leaves_complete_image`] is the allow-list the root walk uses:
    /// a class left out only forgoes the handoff.  Inside a seeded inline the
    /// frames exist (`InlineConcreteFrameGuard`), so the same stop-tracing
    /// decision can convert through `convert_and_run_from_pyjitpl`
    /// (`blackhole.py`) instead of falling to entry replay.  Missing-value
    /// classes stay false — converting those resumes on the hole the walker
    /// refused to read.
    ///
    /// `SubWalkClosedLoop`, `UnexpectedVoidSubReturn` and
    /// `UnexpectedNonVoidSubReturn` fire in the caller's `inline_call`
    /// handler after the callee residuals have already run. The latch
    /// coordinate is that CALL, so converting would re-enter the body.
    pub(crate) fn can_convert_seeded_inline(&self) -> bool {
        if self.leaves_complete_image() {
            return true;
        }
        matches!(
            self,
            Self::AbortMarkerReached { .. }
                | Self::AbortPermanentMarkerReached { .. }
                | Self::PortalFrameTracerArmed { .. }
                | Self::NotInTraceRequiresConcreteExecution { .. }
                | Self::KwonlyDefaultsMappingRacedRecord { .. }
                | Self::ConcreteShadowAllocationFailed { .. }
                | Self::BranchGuardKeptStackUnsupported { .. }
                | Self::BranchGuardKeptSlotUnsourced { .. }
                | Self::BranchGuardUnrestorableKeptStackPermanent { .. }
                | Self::ExcEdgeNoInFrameCatch { .. }
                // In-walk refuse of a loop-bearing callee (`fbw_decline_inline_callee`).
                // With a seeded inline the frames are already an MIFrame stack, so
                // the refuse converts in place (`run_blackhole_interp_to_cancel_tracing`)
                // instead. Unseeded remains residual-not-abort: `carrier_owned` when
                // `!frames_materialized`.
                | Self::LoopBearingCalleeInlineUnsupported {
                    blackhole_required: false,
                    ..
                }
        )
    }

    /// Construct the callee-inline decline.  The
    /// `LoopBearingCalleeInlineUnsupported` variant is emitted from ~20 sites
    /// (multi-frame seed preconditions, snapshot capture, hazard scan), so
    /// route them through one constructor that records the source location
    /// under `PYRE_LB_SITE` to tell the decline reasons apart in a census.
    #[track_caller]
    #[inline]
    pub(crate) fn callee_inline_unsupported(pc: usize) -> Self {
        if std::env::var_os("PYRE_LB_SITE").is_some() {
            let loc = std::panic::Location::caller();
            eprintln!("[lb-site] {}:{} pc={pc}", loc.file(), loc.line());
        }
        // Most call sites have no MIFrame session, so they cannot see the
        // per-frame effect delta `fbw_decline_inline_callee` reads.  An
        // in-flight consume those sites still have to honor: a generator,
        // map, dict/set iterator, itertools iterator, or user `__next__`
        // has no cursor to roll back, and re-entering the CALL runs
        // `__next__` again.  `convert_and_run_from_pyjitpl` continues the
        // framestack instead.  A journaled cursor stays replayable here
        // unless a body effect already stands.
        let blackhole_required = fbw_state::fbw_foriter_unjournaled_consume()
            || (fbw_state::fbw_foriter_inflight_active()
                && fbw_state::fbw_foriter_any_body_effect_signal());
        Self::callee_inline_abort(pc, blackhole_required)
    }

    /// Classify a nested residual decline by whether the aborting MIFrame has
    /// already applied an effect.  A clean attempted frame can be discarded
    /// and re-entered from its caller's CALL; an applied frame must be retained
    /// and run forward with the rest of the live stack, as
    /// `convert_and_run_from_pyjitpl` does (`blackhole.py`).
    pub(crate) fn callee_inline_abort(pc: usize, blackhole_required: bool) -> Self {
        Self::LoopBearingCalleeInlineUnsupported {
            pc,
            blackhole_required,
        }
    }
}

/// Per-process census of full-body-walk decline classes, keyed by
/// [`DispatchError::variant_name`] (or a synthetic key for the non-`Err`
/// abort outcomes).  Populated only on the cold abort path via
/// [`census_record`]; off the hot trace path entirely.  The dump
/// (`census_dump`) is gated on [`fbw_debug_abort_enabled`], so with the
/// `PYRE_FBW_DEBUG_ABORT` flag unset the only cost is one `BTreeMap`
/// insert per declined walk (already a rare, slow event).
///
/// Process-wide, not per thread: `_thread` gives Python real OS threads, each
/// of which traces on its own, so a thread-local map would have the dump report
/// only the thread that happened to print it and silently drop every decline
/// the others took.  A `Mutex` is free at this rate — the map is touched only
/// on the cold decline path, never on the hot trace path.
static FBW_DECLINE_CENSUS: parking_lot::Mutex<std::collections::BTreeMap<&'static str, usize>> =
    parking_lot::Mutex::new(std::collections::BTreeMap::new());

/// Lock the census, recovering from poisoning: a `BTreeMap` of counters has no
/// invariant a panicking writer can leave broken, and a diagnostic that goes
/// silent after an unrelated panic is worse than one that keeps counting.
fn census_map() -> parking_lot::MutexGuard<'static, std::collections::BTreeMap<&'static str, usize>>
{
    FBW_DECLINE_CENSUS.lock()
}

/// Bump the decline count for `name` (a [`DispatchError::variant_name`] or
/// a synthetic outcome key).  Called from the trace-side decline routing.
/// When [`fbw_debug_abort_enabled`] is set, also emits the running census
/// so a corpus run prints `[fbw-census]` lines (the last block printed is
/// the final tally) without needing a process-exit hook.
pub(crate) fn census_record(name: &'static str) {
    *census_map().entry(name).or_insert(0) += 1;
    census_dump();
}

thread_local! {
    /// Code objects already counted by a pre-trace frame admission diagnostic,
    /// so a hot declined helper records exactly one census entry instead of one
    /// per call. Keyed by the stable `CodeObject` pointer.
    static FRAME_SHAPE_DECLINE_SEEN: std::cell::RefCell<std::collections::BTreeSet<usize>> =
        const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
}

/// Record — once per distinct code object — the frame-shape decline of a frame
/// that `unsupported_jit_shape` keeps out of the tracer entirely: a
/// `CurrentFrameOnly` FOR_ITER frame (a body with a non-journalable mutator —
/// the #57 gate — or a `finally`-duplicated loop), a `NestedBreakBridgeResume`
/// frame, or a frame whose constant pool overruns the single-byte index
/// encoding.  The tracer never runs for such a frame, so
/// its decline would otherwise leave no census entry and read as a silent
/// no-token gap; recording it here lets the `PYRE_FBW_DEBUG_ABORT` corpus
/// attribute the pre-trace frame-shape decline alongside the traced declines.
/// `code_ptr` is the `CodeObject` pointer, used only to dedup repeated entries
/// of the same declined frame; `kind` names the shape for the census line.
pub fn census_record_frame_shape_decline(code_ptr: usize, kind: &'static str) {
    let first = FRAME_SHAPE_DECLINE_SEEN.with(|s| s.borrow_mut().insert(code_ptr));
    if first {
        census_record(kind);
    }
}

/// What became of one in-flight FOR_ITER continuation, from the walk that
/// stashed it to the interpreter re-entry that either resumed its body or did
/// not.  The three refusal columns are three different defects and only one of
/// them is about effects, so a single `refused` total cannot be read: an
/// effect that must not be re-run, a frame parked where the push would corrupt
/// the operand stack, and an entry no frame was ever offered.  Each one costs
/// the iteration its stashed item belonged to.
#[derive(Clone, Copy, Default)]
pub(crate) struct ForiterInflightCensusCounts {
    /// The item was pushed and the frame repositioned to its body pc.
    pub(crate) delivered: usize,
    /// R1: a body effect committed since the consume, so re-running the body
    /// would double it.
    pub(crate) refused_effect: usize,
    /// The live frame was not parked at the header this body pc names, so the
    /// push would have landed on the wrong operand stack.
    pub(crate) refused_header: usize,
    /// Destroyed by the take's single-delivery clear: a different entry was
    /// the one the parked frame could accept.
    pub(crate) dropped_unmatched: usize,
}

/// Which column [`census_record_foriter_inflight`] bumps.
#[derive(Clone, Copy)]
pub(crate) enum ForiterInflightOutcome {
    Delivered,
    RefusedEffect,
    RefusedHeader,
    DroppedUnmatched,
}

thread_local! {
    /// In-flight FOR_ITER delivery outcomes, keyed by the frame's stable raw
    /// `CodeObject` identity and the Python body pc. The map is touched only
    /// when its diagnostic is enabled, so ordinary execution does not
    /// allocate or count on the abort path.
    static FORITER_INFLIGHT_CENSUS: std::cell::RefCell<
        std::collections::BTreeMap<(usize, usize), ForiterInflightCensusCounts>
    > = const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// Whether the in-flight FOR_ITER delivery census collects. Callers that must
/// compute a census key before recording share this gate, so resolving the key
/// costs nothing when the diagnostic is off.
pub(crate) fn foriter_inflight_census_enabled() -> bool {
    // Always collect under `cfg(test)`: the unit tests drive each outcome
    // directly, and they are the only control this census has -- no corpus
    // fixture reaches a refusal, so a run that reads zero everywhere cannot
    // tell a quiet census from a broken one.
    if cfg!(test) {
        return true;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("PYRE_FORITER_INFLIGHT_CENSUS").is_some()
            || std::env::var_os("PYRE_FBW_DEBUG_ABORT").is_some()
    })
}

/// The census row for one `(code, body pc)` key, for a test that just drove an
/// outcome through [`fbw_foriter_inflight_take`] and the deliver site.
#[cfg(test)]
pub(crate) fn census_foriter_inflight_row(
    code_ptr: usize,
    body_pc: usize,
) -> ForiterInflightCensusCounts {
    FORITER_INFLIGHT_CENSUS
        .with(|c| c.borrow().get(&(code_ptr, body_pc)).copied())
        .unwrap_or_default()
}

/// Drop every accumulated row, so one test's outcomes cannot be read by the
/// next one on the same thread.
#[cfg(test)]
pub(crate) fn census_foriter_inflight_reset() {
    FORITER_INFLIGHT_CENSUS.with(|c| c.borrow_mut().clear());
}

/// Record one legacy in-flight FOR_ITER delivery decision. The collection
/// gate precedes the thread-local lookup, so the census costs nothing when
/// diagnostics are off.
pub(crate) fn census_record_foriter_inflight(
    code_ptr: usize,
    body_pc: usize,
    outcome: ForiterInflightOutcome,
) {
    if !foriter_inflight_census_enabled() {
        return;
    }
    FORITER_INFLIGHT_CENSUS.with(|c| {
        let mut map = c.borrow_mut();
        let counts = map.entry((code_ptr, body_pc)).or_default();
        *match outcome {
            ForiterInflightOutcome::Delivered => &mut counts.delivered,
            ForiterInflightOutcome::RefusedEffect => &mut counts.refused_effect,
            ForiterInflightOutcome::RefusedHeader => &mut counts.refused_header,
            ForiterInflightOutcome::DroppedUnmatched => &mut counts.dropped_unmatched,
        } += 1;
    });
    census_dump_foriter_inflight();
}

/// Print the accumulated in-flight delivery table after each observation, so
/// the last block in a process log is its final tally without an exit hook.
fn census_dump_foriter_inflight() {
    // Only the env-var reader wants the running table on stderr; under
    // `cfg(test)` the census collects for the assertions alone.
    if !(std::env::var_os("PYRE_FORITER_INFLIGHT_CENSUS").is_some()
        || std::env::var_os("PYRE_FBW_DEBUG_ABORT").is_some())
    {
        return;
    }
    FORITER_INFLIGHT_CENSUS.with(|c| {
        for (&(code_ptr, body_pc), counts) in c.borrow().iter() {
            let (name, source) = if code_ptr == 0 {
                ("<unknown>", "<unknown>")
            } else {
                // The code object is owned by the live frame/JitCode for the
                // duration of the process; the census only reads its labels.
                let code = unsafe { &*(code_ptr as *const pyre_interpreter::CodeObject) };
                (code.qualname.as_str(), code.source_path.as_str())
            };
            eprintln!(
                "[fbw-foriter-census] code=0x{code_ptr:x} name={name:?} source={source:?} \
                 body_pc={body_pc} DELIVERED={} REFUSED_EFFECT={} REFUSED_HEADER={} \
                 DROPPED_UNMATCHED={}",
                counts.delivered,
                counts.refused_effect,
                counts.refused_header,
                counts.dropped_unmatched
            );
        }
    });
}

/// The accumulated decline census as `(variant name, count)` pairs, sorted by
/// name — the same content [`census_dump`] prints, for a reader that has no
/// stderr to print it to.  A wasm guest is exactly that: `eprintln!` goes
/// nowhere and `PYRE_FBW_DEBUG_ABORT` never reaches it (the runner has no
/// environment to pass), so the `[fbw-census]` attribution of every
/// `loops_aborted` is unreachable there unless it can be read back as data.
/// `pyre-wasm` exports it per index; the host runner joins the pairs.
pub fn census_entries() -> Vec<(&'static str, usize)> {
    census_map().iter().map(|(n, &v)| (*n, v)).collect()
}

/// Print the accumulated decline census as `[fbw-census] <name>: <count>`
/// lines, sorted by name.  No-op unless [`fbw_debug_abort_enabled`].
/// Safe to call repeatedly (a diagnostic dump, not a reset).
pub(crate) fn census_dump() {
    if !fbw_debug_abort_enabled() {
        return;
    }
    let map = census_map();
    if map.is_empty() {
        eprintln!("[fbw-census] (no declines recorded)");
        return;
    }
    for (name, count) in map.iter() {
        eprintln!("[fbw-census] {name}: {count}");
    }
}

/// The same tally [`census_dump`] prints, as text, and without the
/// [`fbw_debug_abort_enabled`] gate.
///
/// wasm32 has no descriptors, so the guest's `eprintln!` reaches nothing and
/// `std::env::var_os` is permanently empty there — the dump above is doubly
/// unreachable on that target. The counters themselves are always accumulated
/// (`census_record` bumps the map unconditionally), so handing them back as
/// text lets `pyre-wasm` export them and the runner print them host-side, the
/// same way the jit-stats counters cross the boundary.
///
/// Reading, not draining: safe to call more than once.
pub fn census_report() -> String {
    census_map()
        .iter()
        .map(|(name, count)| format!("[fbw-census] {name}: {count}\n"))
        .collect()
}

thread_local! {
    /// The inlined levels an exception-edge bridge resumes PAST, outermost-first
    /// and excluding the live frame, as `(w_code, python pc)`.
    ///
    /// `call_jit.rs trace_and_compile_from_bridge` routes the exc edge only for
    /// a raise that unwinds clear out of every inlined callee, and answers that
    /// question from the per-level `resume_coords` it decoded.  Those coordinates
    /// are the last description of the discarded levels anywhere in the process —
    /// the walk that follows starts flat — so they are parked here for the walk's
    /// handler entry to turn into traceback nodes.
    static FBW_EXC_EDGE_DISCARDED_LEVELS: std::cell::RefCell<Vec<(usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Publish the levels an exc-edge route is about to discard.  Always called on
/// the routing path, with an empty slice for a single-frame resume, so a
/// previous bridge's levels can never leak into this one.
pub fn set_exc_edge_discarded_levels(levels: &[(usize, usize)]) {
    FBW_EXC_EDGE_DISCARDED_LEVELS.with(|c| {
        let mut slot = c.borrow_mut();
        slot.clear();
        slot.extend_from_slice(levels);
    });
}

pub(crate) fn take_exc_edge_discarded_levels() -> Vec<(usize, usize)> {
    FBW_EXC_EDGE_DISCARDED_LEVELS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Kind bank a `GreenType` / `Type` reads. `Void` consumes nothing
/// (`history.py getkind` drops void arguments).
fn recursive_call_bank(ty: majit_ir::Type) -> Option<char> {
    match ty {
        majit_ir::Type::Int => Some('i'),
        majit_ir::Type::Ref => Some('r'),
        majit_ir::Type::Float => Some('f'),
        majit_ir::Type::Void => None,
    }
}

fn take_banked(
    bank: char,
    i: &mut usize,
    r: &mut usize,
    f: &mut usize,
    ii: &[OpRef],
    rr: &[OpRef],
    ff: &[OpRef],
) -> Option<OpRef> {
    let slot = match bank {
        'i' => {
            let op = *ii.get(*i)?;
            *i += 1;
            op
        }
        'r' => {
            let op = *rr.get(*r)?;
            *r += 1;
            op
        }
        'f' => {
            let op = *ff.get(*f)?;
            *f += 1;
            op
        }
        _ => return None,
    };
    Some(slot)
}

/// The word a green's box carries, or `None` when the walk has no value
/// for it: `_opimpl_recursive_call` reads its greens off `Const` boxes
/// (`verify_green_args`), so a valueless green is not a key.
fn opref_known_i64<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>, op: OpRef) -> Option<i64> {
    match ctx.trace_ctx.concrete_of_opref(op) {
        Some(majit_ir::Value::Int(n)) => Some(n),
        Some(majit_ir::Value::Ref(g)) => Some(g.0 as i64),
        Some(majit_ir::Value::Float(n)) => Some(n.to_bits() as i64),
        _ => None,
    }
}

pub(crate) fn opref_concrete(ctx_value: Option<majit_ir::Value>) -> crate::state::ConcreteValue {
    match ctx_value {
        Some(majit_ir::Value::Int(n)) => crate::state::ConcreteValue::Int(n),
        Some(majit_ir::Value::Float(n)) => crate::state::ConcreteValue::Float(n),
        // `GcRef::NO_CONCRETE` is "no runtime value", the same answer
        // `concrete_from_recorded_opref` already gives. A live object
        // word here would be the sentinel's bit pattern.
        Some(majit_ir::Value::Ref(g)) if g != majit_ir::GcRef::NO_CONCRETE => {
            crate::state::ConcreteValue::Ref(g.0 as pyre_object::PyObjectRef)
        }
        _ => crate::state::ConcreteValue::Null,
    }
}

fn portal_mainjitcode_index(jd_index: usize) -> Option<usize> {
    let key = match jd_index {
        0 => {
            return crate::jitcode_runtime::portal_jitcode().map(|jc| jc.index());
        }
        1 => "baseobjspace::unpackiterable_portal",
        2 => crate::genentry_state::GENENTRY_PORTAL_KEY,
        _ => return None,
    };
    crate::jitcode_runtime::portal_jitcode_for_key(key).map(|jc| jc.index())
}

/// `pyjitpl.py _opimpl_recursive_call` inline / assembler arms.
///
/// `None` means the residual `do_recursive_call(assembler_call=False)`
/// arm still has to run (`inlining` false is not the production case;
/// a missing token or portal body falls through to it).
fn recursive_call_inline_or_assembler<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    jd_index: usize,
    dst_bank: char,
    dst: usize,
    call_opcode: OpCode,
    greens_i: &[OpRef],
    greens_r: &[OpRef],
    greens_f: &[OpRef],
    reds_i: &[OpRef],
    reds_r: &[OpRef],
    reds_f: &[OpRef],
    adr: i64,
    descr: &majit_ir::DescrRef,
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    let Some((driver, _)) = crate::driver::try_driver_pair() else {
        return Ok(None);
    };
    let (green_spec, red_types) = {
        let Some(jd) = driver.meta_interp().staticdata.jitdrivers_sd.get(jd_index) else {
            return Ok(None);
        };
        (jd.green_args_spec(), jd.red_arg_types_as_ir_types())
    };
    let mut gi = 0usize;
    let mut gr = 0usize;
    let mut gf = 0usize;
    let mut green_ops = Vec::with_capacity(green_spec.len());
    let mut green_values = Vec::with_capacity(green_spec.len());
    let mut green_types = Vec::with_capacity(green_spec.len());
    for spec in &green_spec {
        let Some(bank) = recursive_call_bank(majit_ir::green_type_to_ir(*spec)) else {
            continue;
        };
        let Some(opref) = take_banked(
            bank, &mut gi, &mut gr, &mut gf, greens_i, greens_r, greens_f,
        ) else {
            return Ok(None);
        };
        let Some(value) = opref_known_i64(ctx, opref) else {
            return Ok(None);
        };
        green_values.push(value);
        green_types.push(*spec);
        green_ops.push(opref);
    }
    let mut ri = 0usize;
    let mut rr = 0usize;
    let mut rf = 0usize;
    let mut red_ops = Vec::with_capacity(red_types.len());
    for ty in &red_types {
        let Some(bank) = recursive_call_bank(*ty) else {
            continue;
        };
        let Some(opref) = take_banked(bank, &mut ri, &mut rr, &mut rf, reds_i, reds_r, reds_f)
        else {
            return Ok(None);
        };
        red_ops.push(opref);
    }
    let green_key =
        majit_ir::GreenKey::with_types(green_values.clone(), green_types.iter().copied());
    // `_opimpl_recursive_call` reads `targetjitdriver_sd.warmstate`, not the
    // driver that is currently tracing.
    if !driver
        .meta_interp_mut()
        .warm_state_for_driver(jd_index)
        .inlining()
    {
        return Ok(None);
    }
    // `warmrunnerdesc.memory_manager.max_unroll_recursion`: the one manager.
    let max_unroll = driver.meta_interp().memory_manager.max_unroll_recursion() as usize;
    let can_inline = driver
        .meta_interp_mut()
        .warm_state_for_driver(jd_index)
        .can_inline_callable_for_key(&green_key);
    let portal_index = portal_mainjitcode_index(jd_index);
    // `for f in self.metainterp.framestack`: count the frames running the
    // target driver's `mainjitcode` whose `greenkey` equals these greens,
    // green by green (`same_constant`).
    let count = portal_index.map_or(0, |index| {
        driver
            .meta_interp()
            .framestack
            .frames
            .iter()
            .filter(|f| f.jitcode.index() == index)
            .filter(|f| {
                f.greenkey
                    .as_ref()
                    .and_then(|(_, key)| key.as_ref())
                    .is_some_and(|key| *key == green_key)
            })
            .count()
    });
    // `count >= memmgr.max_unroll_recursion` → `dont_trace_here`.
    if can_inline && count >= max_unroll {
        driver
            .meta_interp_mut()
            .warm_state_for_driver(jd_index)
            .disable_noninlinable_function_for_key(&green_key);
    }
    let inline = can_inline && count < max_unroll;
    if inline {
        if let Some(index) = portal_index {
            if let Some(body) = sub_jitcode_body_by_index(index) {
                let int_args: Vec<OpRef> = greens_i
                    .iter()
                    .copied()
                    .chain(reds_i.iter().copied())
                    .collect();
                let ref_args: Vec<OpRef> = greens_r
                    .iter()
                    .copied()
                    .chain(reds_r.iter().copied())
                    .collect();
                let float_args: Vec<OpRef> = greens_f
                    .iter()
                    .copied()
                    .chain(reds_f.iter().copied())
                    .collect();
                let int_concretes: Vec<crate::state::ConcreteValue> = int_args
                    .iter()
                    .map(|op| opref_concrete(ctx.trace_ctx.concrete_of_opref(*op)))
                    .collect();
                let ref_concretes: Vec<crate::state::ConcreteValue> = ref_args
                    .iter()
                    .map(|op| opref_concrete(ctx.trace_ctx.concrete_of_opref(*op)))
                    .collect();
                // `pyjitpl.py newframe`: portal_call_depth, call_ids, ENTER_PORTAL_FRAME.
                let subwalk_jd = crate::state::note_inline_subwalk_start(
                    (green_key.get_uhash(), Some(green_key.clone())),
                    Some(index),
                );
                let walked = match inline_call::run_sub_jitcode_walk(
                    ctx,
                    op.pc,
                    &body,
                    &int_args,
                    &int_concretes,
                    &ref_args,
                    &ref_concretes,
                    &float_args,
                ) {
                    // `popframe` records `LEAVE_PORTAL_FRAME`. The raise has
                    // already left `_interpret`, so the close must not run.
                    Err(error @ DispatchError::TraceTooLong { .. }) => return Err(error),
                    other => other,
                };
                if subwalk_jd.is_some() {
                    crate::state::note_inline_subwalk_end();
                }
                let walked = walked?;
                return Ok(Some(finish_recursive_inline(
                    ctx, op, dst, dst_bank, walked,
                )?));
            }
        }
    }
    // `do_recursive_call(assembler_call=True)` → `direct_assembler_call`.
    // "verify that we have all green args, needed to make sure that
    // assembler that we call is still correct"
    {
        let meta = driver.meta_interp();
        let jd = &meta.staticdata.jitdrivers_sd[jd_index];
        majit_metainterp::MIFrame::verify_green_args(jd, &green_ops);
    }
    let greenboxes: Vec<majit_ir::Value> = green_values
        .iter()
        .zip(green_types.iter())
        .map(|(value, spec)| match majit_ir::green_type_to_ir(*spec) {
            majit_ir::Type::Int => majit_ir::Value::Int(*value),
            majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(*value as u64)),
            majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(*value as usize)),
            majit_ir::Type::Void => majit_ir::Value::Void,
        })
        .collect();
    let Some(token) = driver
        .meta_interp_mut()
        .get_or_make_jitdriver_assembler_token_arc(
            jd_index,
            green_key.get_uhash(),
            &greenboxes,
            &red_types,
        )
    else {
        return Ok(None);
    };
    let funcbox = ctx.trace_ctx.const_int(adr);
    let allboxes = recursive_call_allboxes(
        funcbox, call_descr, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
    );
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
    // Concrete portal runner first (`do_recursive_call`), then the
    // recorded op is `CALL_ASSEMBLER` (`direct_assembler_call`).
    // `OpRef::NONE` lets `vrefs_after_residual_call` finish before the
    // assembler op is recorded (`try_execute_residual_call_via_executor`).
    let exec = try_execute_residual_call_via_executor(
        ctx,
        call_opcode,
        &allboxes,
        call_descr,
        OpRef::NONE,
        op.pc,
        None,
        false,
    )?;
    let (concrete, raised) = match exec {
        ResidualExecOutcome::Executed(Ok(result)) => (result, 0i64),
        ResidualExecOutcome::Executed(Err(exc)) => (0, exc),
        ResidualExecOutcome::Declined(cause) => {
            fbw_state::fbw_mark_unjournaled_effect(cause);
            let recorded =
                ctx.trace_ctx
                    .record_op_with_descr(call_opcode, &allboxes, descr.clone());
            // `do_residual_call` step 5: invalidate the heapcache on the
            // CALL_MAY_FORCE just recorded.
            ctx.trace_ctx.heapcache_invalidate_caches_varargs(
                call_opcode,
                Some(call_descr.get_extra_info()),
                &allboxes,
            );
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            return Ok(Some((DispatchOutcome::Continue, op.next_pc)));
        }
    };
    let recorded = match dst_bank {
        'r' => ctx
            .trace_ctx
            .call_assembler_ref_arc_typed(token, &red_ops, &red_types),
        'i' => ctx
            .trace_ctx
            .call_assembler_int_arc_typed(token, &red_ops, &red_types),
        'f' => ctx
            .trace_ctx
            .call_assembler_float_arc_typed(token, &red_ops, &red_types),
        'v' => {
            ctx.trace_ctx
                .call_assembler_void_arc_typed(token, &red_ops, &red_types);
            OpRef::NONE
        }
        _ => unreachable!("dst_bank matched above"),
    };
    // `pyjitpl.py do_residual_call` step 5: invalidate on the
    // CALL_MAY_FORCE executed in step 2, not the recorded CALL_ASSEMBLER.
    ctx.trace_ctx.heapcache_invalidate_caches_varargs(
        call_opcode,
        Some(call_descr.get_extra_info()),
        &allboxes,
    );
    // `make_result_of_lastop(resbox)`: the executed result is the op's
    // value whatever its bits (zero, +0.0 and null included).
    if recorded != OpRef::NONE && raised == 0 {
        let value = match dst_bank {
            'i' => majit_ir::Value::Int(concrete),
            'f' => majit_ir::Value::Float(f64::from_bits(concrete as u64)),
            _ => majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
        };
        ctx.trace_ctx.set_opref_concrete(recorded, value);
    }
    if recorded != OpRef::NONE {
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    // `direct_assembler_call` returns the target driver's virtualizable red
    // box; `do_residual_call` records `KEEPALIVE` on it after
    // `GUARD_NOT_FORCED`.
    let vablebox = {
        let meta = driver.meta_interp();
        let jd = &meta.staticdata.jitdrivers_sd[jd_index];
        usize::try_from(jd.index_of_virtualizable)
            .ok()
            .and_then(|index| red_ops.get(index).copied())
    };
    if let Some(vablebox) = vablebox {
        ctx.trace_ctx.record_op(OpCode::Keepalive, &[vablebox]);
    }
    if raised != 0 {
        let exc = ctx.trace_ctx.const_ref(raised);
        let exc_concrete = crate::state::ConcreteValue::Ref(raised as pyre_object::PyObjectRef);
        ctx.set_last_exc_value(exc, exc_concrete);
        walker_record_guard_exception(ctx, op.pc)?;
        return Ok(Some((
            DispatchOutcome::SubRaise { exc, exc_concrete },
            op.next_pc,
        )));
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// `do_residual_call`'s `_build_allboxes`: the portal runner's arguments in
/// `descr.get_arg_types()` order, demuxed out of the three register banks.
#[allow(clippy::too_many_arguments)]
fn recursive_call_allboxes(
    funcbox: OpRef,
    call_descr: &dyn majit_ir::descr::CallDescr,
    greens_i: &[OpRef],
    greens_r: &[OpRef],
    greens_f: &[OpRef],
    reds_i: &[OpRef],
    reds_r: &[OpRef],
    reds_f: &[OpRef],
) -> Vec<OpRef> {
    let banks = [
        (greens_i, Type::Int),
        (greens_r, Type::Ref),
        (greens_f, Type::Float),
        (reds_i, Type::Int),
        (reds_r, Type::Ref),
        (reds_f, Type::Float),
    ];
    let argboxes: Vec<OpRef> = banks
        .iter()
        .flat_map(|(ops, _)| ops.iter().copied())
        .collect();
    let argbox_types: Vec<Type> = banks
        .iter()
        .flat_map(|(ops, ty)| std::iter::repeat_n(*ty, ops.len()))
        .collect();
    build_allboxes(
        funcbox,
        &argboxes,
        &argbox_types,
        call_descr.arg_types(),
        None,
    )
}

fn finish_recursive_inline<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    dst: usize,
    dst_bank: char,
    walked: DispatchOutcome,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    match walked {
        DispatchOutcome::SubReturn { result } => {
            if let Some(result) = result {
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, result)?;
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            ctx.set_last_exc_value(exc, exc_concrete);
            Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
        }
        other => Ok((other, op.next_pc)),
    }
}

/// `pyjitpl.py _opimpl_recursive_call` / `do_recursive_call`.
///
/// The original portal graph ends in this op (`warmspot.py
/// rewrite_jit_merge_point`). The call target is `portal_runner_adr`,
/// not the portal jitcode, so the walk records the runner call instead
/// of the evaluator body.
fn dispatch_recursive_call<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let jd_box = read_int_reg(code, op, 0, ctx)?;
    let jd_index = match ctx.trace_ctx.concrete_of_opref(jd_box) {
        Some(Value::Int(index)) if index >= 0 => index as usize,
        _ => {
            return Err(DispatchError::UnsupportedOpname {
                pc: op.pc,
                key: op.key,
            });
        }
    };
    let mut offset = 1usize;
    let (greens_i, width) = read_int_var_list(code, op, offset, ctx)?;
    offset += width;
    let (greens_r, width) = read_ref_var_list(code, op, offset, ctx)?;
    offset += width;
    let (greens_f, width) = read_float_var_list(code, op, offset, ctx)?;
    offset += width;
    let (reds_i, width) = read_int_var_list(code, op, offset, ctx)?;
    offset += width;
    let (reds_r, width) = read_ref_var_list(code, op, offset, ctx)?;
    offset += width;
    let (reds_f, width) = read_float_var_list(code, op, offset, ctx)?;
    let _ = (offset, width);
    let dst_bank = match op.opname {
        "recursive_call_i" => 'i',
        "recursive_call_r" => 'r',
        "recursive_call_f" => 'f',
        "recursive_call_v" => 'v',
        _ => {
            return Err(DispatchError::UnsupportedOpname {
                pc: op.pc,
                key: op.key,
            });
        }
    };
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.next_pc - 1] as usize
    };
    let (adr, descr) = {
        let (driver, _) = crate::driver::driver_pair();
        let Some(jd) = driver.meta_interp().staticdata.jitdrivers_sd.get(jd_index) else {
            return Err(DispatchError::UnsupportedOpname {
                pc: op.pc,
                key: op.key,
            });
        };
        let Some(descr) = jd.portal_calldescr.clone() else {
            return Err(DispatchError::UnsupportedOpname {
                pc: op.pc,
                key: op.key,
            });
        };
        if jd.portal_runner_adr == 0 {
            return Err(DispatchError::UnsupportedOpname {
                pc: op.pc,
                key: op.key,
            });
        }
        (jd.portal_runner_adr, descr)
    };
    let descr_for_record = descr.clone();
    let Some(call_descr) = descr.as_call_descr() else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: op.key,
        });
    };
    let call_opcode = match dst_bank {
        'r' => OpCode::CallMayForceR,
        'i' => OpCode::CallMayForceI,
        'f' => OpCode::CallMayForceF,
        'v' => OpCode::CallMayForceN,
        _ => unreachable!("dst_bank matched above"),
    };
    // `pyjitpl.py _opimpl_recursive_call`: `can_inline_callable` then
    // `perform_call`, otherwise `do_recursive_call(assembler_call=True)`.
    if let Some(done) = recursive_call_inline_or_assembler(
        ctx,
        op,
        jd_index,
        dst_bank,
        dst,
        call_opcode,
        &greens_i,
        &greens_r,
        &greens_f,
        &reds_i,
        &reds_r,
        &reds_f,
        adr,
        &descr,
        call_descr,
    )? {
        return Ok(done);
    }
    let funcbox = ctx.trace_ctx.const_int(adr);
    let allboxes = recursive_call_allboxes(
        funcbox, call_descr, &greens_i, &greens_r, &greens_f, &reds_i, &reds_r, &reds_f,
    );
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
    let recorded = ctx
        .trace_ctx
        .record_op_with_descr(call_opcode, &allboxes, descr_for_record);
    let exec = try_execute_residual_call_via_executor(
        ctx,
        call_opcode,
        &allboxes,
        call_descr,
        recorded,
        op.pc,
        Some((op.next_pc, dst_bank, dst)),
        false,
    )?;
    // `do_residual_call` step 5: invalidate the heapcache on the
    // CALL_MAY_FORCE executed above.
    ctx.trace_ctx.heapcache_invalidate_caches_varargs(
        call_opcode,
        Some(call_descr.get_extra_info()),
        &allboxes,
    );
    let raised = match exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_state::fbw_mark_unjournaled_effect(cause);
            false
        }
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    if raised {
        walker_record_guard_exception(ctx, op.pc)?;
        let exc = ctx
            .last_exc_value()
            .expect("recursive_call raise seeds last_exc_value");
        let exc_concrete = ctx.last_exc_value_concrete();
        return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Walk one opcode at `pc` and return the dispatch outcome plus the
/// next pc. Side effects reach `ctx.trace_ctx` only for opnames whose
/// handler explicitly records (e.g. `ref_return/r` calls
/// `record_finish`).
///
/// The returned `next_pc` is normally `op.next_pc` (linear advance
/// past the operand bytes); branch handlers (`goto/L` etc.) override
/// this with their target.
pub fn step<Sym: WalkSym>(
    code: &[u8],
    pc: usize,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let op: DecodedOp = decode_op_at(code, pc).ok_or(DispatchError::UndecodableOpcode { pc })?;
    inline_call::note_subwalk_driver_step::<Sym>(op.opname, pc, ctx.trace_ctx.get_trace_position());
    // The walker mixes translated vable operations (which update the shadow)
    // with concrete interpreter steps (which update the heap PyFrame).  Pull
    // those concrete writes into `virtualizable_boxes` before any handler can
    // read or synchronize it, matching the invariant documented by
    // `TraceCtx::refresh_virtualizable_shadow_from_heap`.
    ctx.trace_ctx.refresh_virtualizable_shadow_from_heap();
    if ctx.is_top_level {
        ctx.session.borrow_mut().recording_opcode_position = op.pc;
    }
    // Per-CodeObject portal JitCodes contain the lowered Python opcode body,
    // but not the dispatch loop's `jit_merge_point` marker that preceded that
    // body in `interp_jit.py`. Restore `MIFrame.debug_merge_point` at the
    // exact Python-opcode emission boundary before executing the body. A
    // JitCode that really carries `jit_merge_point` records its marker in that
    // handler below instead, so it is not doubled here.
    // `-live-` is codewriter resume metadata, not an interpreter opcode.
    // Consume it first so the marker on the following real op captures this
    // opcode's own live anchor rather than the preceding one's.
    if op.opname != "live" && !op.key.starts_with("jit_merge_point") {
        if let Some(outcome) = record_python_debug_merge_point(ctx, code, op.pc)? {
            return Ok((outcome, op.next_pc));
        }
    }
    // #73: maintain the `-live-` BEFORE anchor.  Every
    // `-live-` byte the walk decodes becomes the resume point preceding the
    // NEXT guard (`pyjitpl.py`, normal guard resume reads at
    // `self.pc - SIZE_LIVE_OP`).  `op.pc` is the genuine jitcode offset of
    // the `-live-` byte.  Side-data only — read under a debug audit gate.
    if op.opname == "live" {
        ctx.live_before_jit_pc = op.pc;
    }
    // #73: maintain the walk-level operand-stack box
    // mirror.  Detects a Python-opcode boundary at this jitcode pc and
    // reconciles the previous opcode's stack effect into `ctx.vstack_boxes`
    // BEFORE this op runs.  Writes ONLY the new `vstack_*` fields — never
    // the existing registers / snapshot / control flow — so the mirror is
    // pure side-data until a later slice makes the read authoritative.  No-op
    // unless the full-body walk owns the virtualizable shadow and the
    // mirror is still valid.
    step_vstack_mirror(ctx, pc);
    let effects_before = fbw_executed_effect_count();
    let result = handle(&op, code, ctx);
    if matches!(
        result,
        Err(DispatchError::LoopBearingCalleeInlineUnsupported { .. })
    ) {
        FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS.with(|c| {
            c.set(Some((
                op.pc,
                fbw_executed_effect_count().saturating_sub(effects_before),
            )))
        });
    }
    result
}

/// Walk the code from `start_pc` until a terminating opcode fires.
/// Returns the terminating outcome plus the pc immediately after the
/// terminator. Top-level callers expect `DispatchOutcome::Terminate`
/// (other variants appear only inside a sub-walk frame entered via
/// `inline_call_r_r/dR>r` — `ref_return/r` and `raise/r` produce
/// `SubReturn` / `SubRaise` there).
///
/// **Top-level uncaught SubRaise**: when an inline_call
/// SubRaise bubbles up through every parent frame without a
/// `catch_exception/L` handler match and reaches the outermost
/// `walk()` invocation, RPython `pyjitpl.py
/// finishframe_exception` records `compile_exit_frame_with_exception(
/// last_exc_box)` — i.e. `FINISH(exc, exit_frame_with_exception_descr_ref)`
/// + raise `ExitFrameWithExceptionRef`. The walker mirrors this on
/// exit: if the loop terminates with `SubRaise` AND `ctx.is_top_level
/// == true`, record the FINISH and convert the outcome to `Terminate`
/// before returning. Sub-walk frames keep returning `SubRaise` to
/// their callers (the unwind continues until either a handler
/// matches or the outermost walker handles it).
pub fn walk<Sym: WalkSym>(
    code: &[u8],
    start_pc: usize,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let mut pc = start_pc;
    loop {
        // A sub-walk error recovered inside an earlier opcode must not make a
        // later abort of this frame decline. The flag is raised again only
        // when this step's own sub-walk propagates.
        ctx.session.borrow_mut().crossed_inline_subwalk = false;
        let opcode_position = pc;
        let (outcome, next_pc) = match step(code, pc, ctx) {
            Ok(stepped) => stepped,
            // Not an abort: a nested inline_call asked the heap-owned
            // sub-walk driver to push its callee.  Preserve this frame at the
            // CALL coordinate and let the driver resume it after the callee
            // produces SubReturn/SubRaise, matching PyPy's one `_interpret`
            // loop over `metainterp.framestack`.
            Err(error @ DispatchError::SubWalkSuspended { .. }) => return Err(error),
            Err(error) => {
                // `pyjitpl.py run_blackhole_interp_to_cancel_tracing`
                // parity.  Upstream has ONE abort path: every abort inside
                // `_interpret` raises `SwitchToBlackhole`, and
                // `blackhole.py convert_and_run_from_pyjitpl` FINISHES the
                // frames the walk reached — it never returns to the caller and
                // never replays the aborted region.  The walk executes
                // residuals concretely, so replaying from the trace entry
                // re-applies every effect it already committed.
                //
                // Only the `is_too_long()` arm below (and the bridge sub-walk,
                // `bridge_subwalk.rs`) used to latch that image, which left
                // every other `DispatchError` — the whole capability-gap family
                // — aborting with no handoff at all.  Latch here instead, at
                // the step that failed, so the class of the abort stops
                // deciding whether the recovery exists.
                //
                // `latch_abort_blackhole` is fully gated: a non-authoritative
                // walk, an incomplete register image, or an unsupported frame
                // shape all decline and leave the legacy entry replay in place,
                // so an unlatched abort is never worse than before.
                //
                // Kept in lockstep with the epilogue's own exclusions
                // (`run_perfn_walk`, `WalkEndCommitLeg::WalkAbort`).
                // `VableEscapedDuringResidualCall` needs no exclusion —
                // it latches its narrower resume-marker image at force time,
                // before the error unwinds here, so the guard below defers to it.
                // Do not consume the session's one-shot coordinate claim for
                // a transparent helper: its opcode coordinate is not a Python
                // resume coordinate, and the enclosing Python walk must still
                // be able to claim and latch the propagated abort.
                let coordinate_belongs_to_this_frame = !ctx.fbw_mode.transparent_helper_subwalk
                    && ctx
                        .session
                        .borrow_mut()
                        .claim_abort_coordinate(ctx.fbw_mode.inline_subwalk);
                if fbw_state::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[walk-abort] err={:?} stop_pc={} opcode_position={} \
                         helper={} inline={} claimed={} callee_jc={:?}",
                        error,
                        error.stop_pc(),
                        opcode_position,
                        ctx.fbw_mode.transparent_helper_subwalk,
                        ctx.fbw_mode.inline_subwalk,
                        coordinate_belongs_to_this_frame,
                        ctx.inline_callee_consts.map(|c| c.jitcode_index),
                    );
                }
                // `LoopBearingCalleeInlineUnsupported { blackhole_required: false }`
                // without a seeded callee is residual-not-abort: the enclosing
                // CALL is not entered, so there is no MIFrame to convert.
                // A seeded inline already has the callee `PyFrame` and converts
                // in place (`run_blackhole_interp_to_cancel_tracing`).
                // `AbortPermanentMarkerReached` at the root still owns the
                // AbortPc flush.
                let frames_materialized = ctx.fbw_mode.inline_subwalk
                    && residual_call::current_inline_concrete_frame() != 0;
                let carrier_owned = !frames_materialized
                    && matches!(
                        error,
                        DispatchError::LoopBearingCalleeInlineUnsupported {
                            blackhole_required: false,
                            ..
                        } | DispatchError::AbortPermanentMarkerReached { .. }
                    );
                // Kept in lockstep with the epilogue's own gate: an abort that
                // reports a MISSING value cannot be converted, because the
                // image it would hand the blackhole is the one that just
                // failed to resolve ([`DispatchError::leaves_complete_image`]).
                //
                // A translated builtin gateway is not an MIFrame.  Let its
                // error unwind to the enclosing Python walk, where the same
                // latch is built from that frame's own register banks.  If we
                // latch here, `build_multi_frame_miframe` appends the helper's
                // r0 args-slice as though it were a Python red frame and the
                // blackhole resume loses per-frame identity.  This is the
                // abort counterpart of the transparent-helper guard snapshot
                // and trace-limit handling below.
                //
                // A seeded inline whose abort is a modeling gap rather than a
                // missing value latches even when the root allow-list has not
                // named the class: `run_blackhole_interp_to_cancel_tracing`
                // (`pyjitpl.py`) converts whatever `framestack` is live.
                if coordinate_belongs_to_this_frame
                    && (error.leaves_complete_image()
                        || (frames_materialized && error.can_convert_seeded_inline()))
                    && !carrier_owned
                    && !abort_blackhole_latched()
                {
                    // The image belongs to THIS WalkContext's MIFrame, so its
                    // resume coordinate must be this frame's opcode boundary.
                    // A canonical helper is transparent: an error raised at
                    // helper pc N propagates through `step`, but N indexes the
                    // helper JitCode and is not a coordinate in this Python
                    // frame's JitCode.  RPython keeps those as distinct
                    // MIFrames; pairing the helper pc with the enclosing frame
                    // lands `setposition` inside an operand payload.  Resume
                    // the enclosing frame at the opcode whose step propagated
                    // the abort instead.
                    let _ = latch_abort_blackhole(ctx, opcode_position, "mod2730");
                }
                return Err(error);
            }
        };
        pc = next_pc;
        // pyjitpl.py `_interpret`: `blackhole_if_trace_too_long()` runs
        // after every `run_one_step()`. This loop is that loop's counterpart —
        // `step` is `run_one_step` — and it is the only place the check runs:
        // the concrete bytecode dispatch loop it replaced deliberately dropped
        // it (`pyre-jit/src/eval.rs`, whose comment records why), so nothing
        // else bounds a walk that keeps recording without reaching a close.
        // `TraceCtx::is_too_long` is the `history.length() >
        // warmrunnerstate.trace_limit` test and `num_ops` is a `Vec::len`, so
        // this is one comparison per step.
        //
        // The walker layer holds `&mut TraceCtx` and cannot reach
        // `MetaInterp::blackhole_if_trace_too_long` for the full bookkeeping;
        // `note_root_trace_too_long` performs it instead — the
        // `find_biggest_function` → `disable_noninlinable_function` half as well
        // as the loop and bridge arms `prepare_trace_segmenting` (pyjitpl.py)
        // keeps apart. `DispatchError::TraceTooLong` becomes the explicit
        // `TraceAction::SwitchToBlackhole(ABORT_TOO_LONG)` unwind, so the catch
        // consumes the reason without running this decision again.
        //
        // `blackhole_if_trace_too_long` raises AFTER `run_one_step`, so the
        // forward image must carry `pc`, the already-advanced `next_pc`, rather
        // than `opcode_position`.  `latch_abort_blackhole` copies the
        // live MIFrame registers while this WalkContext still owns them; the
        // run-per-fn epilogue drives that image forward exactly like RPython's
        // `run_blackhole_interp_to_cancel_tracing`.
        //
        // A complete image makes the abort safe regardless of effects.  If the
        // image cannot be built, a zero-effect walk may still take the legacy
        // replay; an effectful walk must retain the former keep-recording
        // fallback because replay would apply an irreversible effect twice.
        // The remaining overshoot is tallied so an unsupported multi-frame
        // shape cannot silently become unbounded.
        // pyjitpl.py raises out of `debug_merge_point`, i.e. from
        // INSIDE the step, so `blackhole_if_trace_too_long` (pyjitpl.py)
        // never sees a segmented trace. When a walk crosses a merge point
        // already past 1.0x the limit both conditions hold, and upstream
        // segments rather than aborts because its cut raises first. Return
        // ahead of the too-long check for the same reason — turning the cut
        // back into `TraceTooLong` here would re-abort exactly the key the
        // segmenting signal was set to stop aborting.
        if let DispatchOutcome::SegmentTrace { .. } = outcome {
            return Ok((outcome, pc));
        }
        // A translated builtin gateway is transparent to the Python MIFrame
        // stack and has no blackhole entry point of its own. Its bytecode is
        // one implementation detail of the enclosing Python CALL step, so do
        // not split the trace in the middle of it: doing so would have to pair
        // helper registers with a Python JitCode's red-frame register. Finish
        // the helper and let the enclosing Python `walk()` perform this same
        // post-step limit check at its real per-frame coordinate, matching
        // RPython's one-red-frame ownership.
        //
        // The exemption defers the abort, it does not drop it. `is_too_long`
        // is a `num_ops > trace_limit` comparison over the shared `TraceCtx`,
        // so it still holds when the helper returns and the enclosing frame
        // runs this check. Aborting inside the helper is what turns one
        // overflow into a retrace storm: the helper has no merge point that
        // can close a segment, and the tokenless cell keeps counting.
        if !ctx.fbw_mode.transparent_helper_subwalk && ctx.trace_ctx.is_too_long() {
            // `step` has advanced the register banks for `Continue`. The
            // other outcomes still need the match below to perform their
            // frame transition: in particular, `SubRaise` may enter this
            // frame's handler and `SubReturn` is delivered to its caller by
            // the inline-call boundary. RPython checks the length only after
            // those transitions have happened, so never abort from the
            // pre-transition callee image here — even a zero-effect entry
            // replay would resume the caller without delivering the return or
            // raise that this step produced.
            let snapshot_safe = trace_too_long_blackhole_snapshot_safe(&outcome);
            // Reconcile the operand-stack mirror to the post-step resume
            // coordinate before latching.  A block-head merge point is
            // otherwise left on the completed opcode's on-entry stack
            // (`reconcile_vstack_to_resume_pc`).
            if snapshot_safe {
                reconcile_vstack_to_resume_pc(ctx, pc);
            }
            let mut blackhole_latched = snapshot_safe && latch_abort_blackhole(ctx, pc, "mod2786");
            // Root-walk too-long publishes the walker mirror (`_copy_data_from_miframe`
            // equivalent).  Ask the height check here, while a refusal can still
            // unstage: a preflighted adopt treats a later decline as an assert.
            if blackhole_latched
                && !ctx.fbw_mode.inline_subwalk
                && !latched_single_frame_mirror_publishable()
            {
                reset_single_frame_blackhole();
                blackhole_latched = false;
            }
            if trace_too_long_abort_safe(&outcome, blackhole_latched, fbw_executed_effect_count()) {
                let ops = ctx.trace_ctx.num_recorded_ops();
                crate::state::note_root_trace_too_long(
                    ctx.trace_ctx.current_merge_points_first_green_key_pair(),
                    ctx.trace_ctx.resumekey_original_loop_token().cloned(),
                );
                ctx.session.borrow_mut().trace_too_long = true;
                return Err(DispatchError::TraceTooLong {
                    pc: opcode_position,
                    ops,
                });
            } else {
                majit_metainterp::mc_diag_bump(26);
            }
        }
        match outcome {
            DispatchOutcome::Continue => {}
            DispatchOutcome::Terminate { .. }
            | DispatchOutcome::SubReturn { .. }
            | DispatchOutcome::SwitchToBlackhole { .. }
            | DispatchOutcome::CloseLoop { .. }
            | DispatchOutcome::CompileTracePending { .. }
            // A multi-frame inlined callee reached its own loop
            // header; propagate up to `try_walker_inline_user_call`, which
            // emits the recursive CALL_ASSEMBLER at the call boundary.
            | DispatchOutcome::SubLoopCalleeCallAssembler { .. }
            // Returned above, ahead of the too-long check; covered here so the
            // match stays exhaustive.
            | DispatchOutcome::SegmentTrace { .. } => {
                return Ok((outcome, pc));
            }
            DispatchOutcome::SubRaise { exc, mut exc_concrete } => {
                // RPython `finishframe_exception`: before
                // unwinding to the caller, scan THIS frame for a matching
                // `catch_exception/L` handler at the post-op position. A
                // residual call that raised inside a try-block resumes its own
                // except body here; only when the current frame has no handler
                // does the exception unwind further. The `inline_call` SubRaise
                // arm performs the symmetric scan for caller frames, so a raise
                // routes to the nearest enclosing handler regardless of whether
                // it sits in the raising frame or an inlining ancestor. Without
                // this current-frame scan, a top-level walk whose raising op
                // sits in a try-block (e.g. a standalone-traced callee whose
                // getitem raises) skipped its own handler and recorded
                // `exit_frame_with_exception`, letting the exception escape a
                // frame that actually catches it.
                if let Some(target) = try_catch_exception_at(code, pc) {
                    // `op.key` carries the full `opname/argcodes` spelling;
                    // `op.opname` is only the part before the `/`.
                    let raised_in_this_frame = decode_op_at(code, opcode_position)
                        .is_some_and(|op| matches!(op.key, "raise/r" | "reraise/"));
                    // finishframe_exception propagates into this frame
                    // before catch_exception/L enters its handler. Record
                    // that frame-boundary step once; handler entry itself
                    // does not add a traceback node.
                    //
                    // The node has to exist on the compiled run too: the
                    // handler is part of the trace, so this frame never
                    // surfaces an error the interpreter's `handle_exception`
                    // could record from.  A raise the `raise/r` / `reraise/`
                    // arm already routed here is the exception — that arm
                    // emits the runtime record itself, so emitting again
                    // would give the frame two nodes.
                    let recording_opcode_position = ctx.session.borrow().recording_opcode_position;
                    let node_position = if ctx.is_top_level {
                        recording_opcode_position
                    } else {
                        opcode_position
                    };
                    let node = if raised_in_this_frame {
                        None
                    } else {
                        record_prepend_application_traceback(ctx, exc, exc_concrete, node_position)?
                    };
                    let emit_runtime = !raised_in_this_frame && node.is_none();
                    // `RaiseWithExplicitTraceback` (`RAISE_VARARGS 0`, `RERAISE`)
                    // re-raises the handled instance and does not write
                    // `__context__` (`pyopcode.py handle_bytecode` sends it
                    // around `OperationError.record_context`).  The catch
                    // compensation's residual still takes `exc` as an argument,
                    // which forces a pendingfields reconstruction of a virtual
                    // exception the handler never reads.  PyPy's reraise
                    // bridge DCEs that object; skip the call here so the heap
                    // optimizer can too.
                    // `recording_raise_keeps_existing_traceback` is also true
                    // for `FOR_ITER`, which keeps the iterator's traceback
                    // but still writes `__context__` for a non-StopIteration.
                    // Gate on `raised_in_this_frame` (`raise/r` / `reraise/`)
                    // so this skip is exactly the bare-reraise opcode case.
                    if raised_in_this_frame
                        && recording_raise_keeps_existing_traceback(ctx, opcode_position)
                    {
                        fbw_context_chained_insert(exc);
                    } else {
                        record_inline_exception_context(ctx.trace_ctx, exc, exc_concrete);
                    }
                    record_inline_application_traceback(
                        ctx,
                        exc,
                        &mut exc_concrete,
                        opcode_position,
                        true,
                        emit_runtime,
node,
                    );
                    record_top_level_application_traceback(
                        ctx,
                        exc,
                        &mut exc_concrete,
                        recording_opcode_position,
                        true,
                        emit_runtime,
node,
                    );
                    ctx.set_last_exc_value(exc, exc_concrete);
                    // pyjitpl.py `finishframe_exception` only
                    // unwinds frames and selects the handler.  The shared
                    // MetaInterp exception-class state was established by
                    // `execute_ll_raised` / `handle_possible_exception` or
                    // `opimpl_raise` and must survive the frame transition.
                    // The exception is now caught by this frame's handler, so
                    // drain the standing residual-call exception flag the
                    // raising helper published (`try_execute_residual_call_via_
                    // executor` Err arm restores `BH_LAST_EXC_VALUE` so the
                    // interpreter's walker-skip path can re-raise an *uncaught*
                    // exception). Leaving it set would make that path spuriously
                    // re-raise the now-handled exception after the walk ends.
                    // Mirrors `blackhole.rs route_to_catch`'s `BH_LAST_EXC_VALUE
                    // = 0` on handler entry.
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
                    // #370: re-seed the operand-stack mirror at the handler
                    // entry so a kept-stack guard inside the handler compiles
                    // from the mirror instead of declining.  The unwind
                    // boundary is otherwise unmodeled, so without this the
                    // mirror latches invalid and every in-handler kept-stack
                    // guard declines.  `vstack_enter_exception_handler` falls
                    // back safely (per-slot NONE hole / `vstack_valid = false`)
                    // for the coordinates it cannot reconstruct.
                    vstack_enter_exception_handler(ctx, target, exc);
                    pc = target;
                    continue;
                }
                if ctx.is_top_level {
                    let recording_opcode_position =
                        ctx.session.borrow().recording_opcode_position;
                    // The node reads the raise coordinate out of
                    // `frame.last_instr`, on both routes — the interpreter's
                    // recorder and the emitted one, which falls back to that
                    // field.  Compiled code never wrote it, so publish it here.
                    fbw_publish_exit_last_instr(ctx, recording_opcode_position);
                    // `pyopcode.py handle_operation_error`: the no-handler
                    // path stores `frame_finished_execution = True` before
                    // `raise operr` leaves the portal, so the store precedes
                    // the exit's token protocol; the inlined-callee sibling is
                    // `walker_ec_leave`.
                    fbw_record_top_level_frame_finished(ctx);
                    // `pyjitpl.py compile_exit_frame_with_exception` opens
                    // with `store_token_in_vable()`, exactly as
                    // `compile_done_with_this_frame` does. Keep this exit lazy
                    // too; a traceback reader forces it through the armed
                    // deadframe only if redirected fields are observed.
                    fbw_store_token_in_vable(ctx, recording_opcode_position)?;
                    // Runtime node after the vable token, same order as
                    // `compile_exit_frame_with_exception` (`pyjitpl.py`) plus
                    // `record_application_traceback` (`pytraceback.py`):
                    // store_token, then attach.  Prefer the IR-virtual
                    // `NewWithVtable` node (`record_prepend_application_traceback`)
                    // so a function-entry raise (`thrower` / `raise KeyError(i)`)
                    // keeps the exception and frame virtual through
                    // `Finish(ExitFrameWithExceptionDescrRef)`.  The opaque
                    // hook is `EffectInfo::MOST_GENERAL` and forces both.
                    // Same fallback the in-frame catch arm above already uses.
                    if !recording_raise_keeps_existing_traceback(ctx, opcode_position) {
                        // The prepend reads `get_traceback()` off the live
                        // exception, so it runs BEFORE the concrete attach
                        // of this frame's node: attached first, the read
                        // answers this frame's own node, and the guard the
                        // escape mark derives from it fails on every
                        // compiled run, whose exception carries no such
                        // node yet.
                        //
                        // Emit at runtime too when the prepend declines, not
                        // only for the recording pass.  Leaving the node to
                        // the interpreter holds only for a trace the
                        // interpreter entered: `CALL_ASSEMBLER` enters this
                        // trace from another trace's compiled code, and the
                        // `exit_frame_with_exception` it finishes with
                        // dispatches to `handle_fail_exit_frame_with_
                        // exception`, which republishes the value and
                        // returns into the caller's machine code.  No
                        // interpreter sees the error, so this frame
                        // contributes no node and the traceback is one frame
                        // short per `CALL_ASSEMBLER` entry.  The
                        // interpreter-entry case stays single-node: the
                        // second record is screened by
                        // `screen_frame_already_recorded` clearing
                        // `attach_tb` for a frame the chain head names.
                        let node = record_prepend_application_traceback(
                            ctx,
                            exc,
                            exc_concrete,
                            recording_opcode_position,
                        )?;
                        let emit_runtime = node.is_none();
                        record_top_level_application_traceback(
                            ctx,
                            exc,
                            &mut exc_concrete,
                            recording_opcode_position,
                            true,
                            emit_runtime,
node,
                        );
                    }
                    // The concrete bit lands only once `fbw_store_token_in_vable`
                    // above has accepted the exit (see
                    // `finish_current_frame_execution`); the recorded store is
                    // already in the trace.
                    commit_top_level_frame_finished(ctx);
                    // RPython parity: framestack exhausted with no handler
                    // match → `compile_exit_frame_with_exception(last_exc_box)`.
                    // Stash the exception the same way the value-return arms
                    // stash their result (payload + no inline FINISH); the
                    // Terminate arm builds `TraceAction::Finish {
                    // exit_with_exception: true }` so the compile consumer
                    // records the FINISH once against
                    // `exit_frame_with_exception_descr`.  Recording it here too
                    // would double it.
                    let (finish_arg, finish_arg_type) =
                        fbw_terminate_with_raise(ctx, exc, exc_concrete);
                    return Ok((
                        DispatchOutcome::Terminate {
                            finish_arg,
                            finish_arg_type,
                        },
                        pc,
                    ));
                } else {
                    if !recording_raise_keeps_existing_traceback(ctx, opcode_position) {
                        // Emit the node at runtime as well as applying it for
                        // the recording pass, for the same reason the top-level
                        // arm above does — here because an inlined callee has
                        // no frame to return to at all: the walk pops it
                        // symbolically, so unless the trace carries the record
                        // itself the callee contributes no node once the trace
                        // runs compiled, and the exception reaches its handler
                        // one frame short.
                        let node = record_prepend_application_traceback(
                            ctx,
                            exc,
                            exc_concrete,
                            opcode_position,
                        )?;
                        let emit_runtime = node.is_none();
                        record_inline_application_traceback(
                            ctx,
                            exc,
                            &mut exc_concrete,
                            opcode_position,
                            true,
                            emit_runtime,
node,
                        );
                    }
                    return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, pc));
                }
            }
        }
    }
}

/// Read a Ref-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_r[code[pc+1]]` for an `r`-coded operand.
fn read_ref_reg<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let reg = code[op.pc + 1 + operand_offset] as usize;
    let value = read_ref_reg_raw(code, op, operand_offset, ctx)?;
    // `[None] * num_regs` initial state (`pyjitpl.py:190-197`): a slot the
    // walk never wrote holds no box, and `OpRef::NONE` is not one.  See
    // [`DispatchError::RegisterReadUnbound`].
    if value.is_none() {
        return Err(DispatchError::RegisterReadUnbound {
            pc: op.pc,
            reg,
            bank: "r",
        });
    }
    Ok(value)
}

/// [`read_ref_reg`] without the unbound-slot check, for the handful of arms
/// that give an unwritten slot their own, more specific decline — the
/// `*_vable_*` family reads its object operand this way so an unseeded
/// virtualizable register still surfaces [`DispatchError::VableBoxNotSeeded`]
/// instead of the generic [`DispatchError::RegisterReadUnbound`].
fn read_ref_reg_raw<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_r
        .get(reg)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_r.len(),
            bank: "r",
        })
}

/// Read an Int-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_i[code[pc+1]]` for an `i`-coded operand.
fn read_int_reg<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_i
        .get(reg)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_i.len(),
            bank: "i",
        })
}

/// Read a Float-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_f[code[pc+1]]` for an `f`-coded operand.
fn read_float_reg<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_f
        .get(reg)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_f.len(),
            bank: "f",
        })
}

/// Read a 2-byte little-endian label operand at `pc + 1 +
/// operand_offset`. RPython encoding: `assembler.py:write_label`
/// writes the resolved target as `chr(target & 0xFF)` +
/// `chr((target >> 8) & 0xFF)`, matching `bhimpl_goto`'s
/// `code[pc] | (code[pc+1] << 8)` decode.
pub(crate) fn read_label(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// Outcome of probing the per-frame raise-bubbling lookahead at
/// `position` (the pc just after a raising op).
///
/// RPython parity: `finishframe_exception` walks
/// through three mutually-exclusive cases after skipping a leading
/// `live/`:
///
///   1. Next op is `catch_exception/L` → jump to the handler target,
///      `raise ChangeFrame`. (Handler matched.)
///   2. Next op is `rvmprof_code/ii` → call `cintf.jit_rvmprof_code(arg1,
///      arg2)` for instrumentation, then fall through to `popframe()`
///      (continue unwinding).
///   3. Otherwise → `popframe()` (continue unwinding).
///
/// Cases 2 and 3 both unwind, but case 2 also fires the rvmprof side
/// effect. RPython at line 2531 invokes `cintf.jit_rvmprof_code(arg1,
/// arg2)` directly during tracing — RPython does NOT record this as
/// an IR op, but the side effect IS observable (it advances the
/// rvmprof profiler state). The helper surfaces the matched register
/// pair via [`FinishframeLookahead::RvmprofCode`] so a future port
/// can invoke pyre's `bh.handle_rvmprof_enter`-equivalent
/// (`pyre-jit/src/call_jit.rs`); the walker records IR directly
/// and the rvmprof side effect is not yet ported, so the caller drops
/// it here — TODO,
/// scoped to the rvmprof profiler instrumentation only (no trace IR
/// effect).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FinishframeLookahead {
    /// Handler match: caller-frame jump target (2-byte LE label after
    /// `catch_exception/L`). Caller sets `last_exc_value` and resumes
    /// at `target`.
    CatchTarget(usize),
    /// `rvmprof_code/ii` lies on the unwind path. Caller continues
    /// unwinding (no handler match); the runtime `cintf.jit_rvmprof_code`
    /// side effect (`pyjitpl.py`, mirrored by
    /// `blackhole.rs bhimpl_rvmprof_code`) is dropped for now. The op is
    /// only emitted when the interpreter main loop is rvmprof-instrumented,
    /// and the call is itself a no-op unless a vmprof profiler HOOK is
    /// installed, so the drop is inert for ordinary execution.
    ///
    /// Deliberately not fired here: this helper is a lookahead that pops
    /// no frame, and `MetaInterp::finishframe_exception` — the loop that
    /// does pop them — already makes the call, so firing here would
    /// double the enter/leave marks.  `arg1_reg` / `arg2_reg` stay
    /// surfaced for the decode.
    #[allow(dead_code)]
    RvmprofCode { arg1_reg: u8, arg2_reg: u8 },
    /// Neither match — unwinding continues with no side effect.
    NoMatch,
}

/// Probe the per-frame raise-bubbling lookahead. RPython parity:
/// `finishframe_exception` line-by-line —
/// `live/` skip then sequential `catch_exception` / `rvmprof_code` /
/// fall-through arms.
fn finishframe_lookahead_at(code: &[u8], position: usize) -> FinishframeLookahead {
    let mut pos = position;
    let Some(op) = decode_op_at(code, pos) else {
        return FinishframeLookahead::NoMatch;
    };
    // RPython `if opcode == op_live: position += SIZE_LIVE_OP`.
    if op.key == "live/" {
        pos = op.next_pc;
    }
    let Some(next) = decode_op_at(code, pos) else {
        return FinishframeLookahead::NoMatch;
    };
    if next.key == "catch_exception/L" {
        let lo = code[next.pc + 1] as usize;
        let hi = code[next.pc + 2] as usize;
        return FinishframeLookahead::CatchTarget(lo | (hi << 8));
    }
    if next.key == "rvmprof_code/ii" {
        // RPython `pyjitpl.py`:
        //   arg1 = frame.registers_i[ord(code[position + 1])].getint()
        //   arg2 = frame.registers_i[ord(code[position + 2])].getint()
        //   assert arg1 == 1
        //   cintf.jit_rvmprof_code(arg1, arg2)
        // Walker surfaces the operand byte indices for the caller to
        // decide whether to symbolically record. The `cintf` call itself
        // belongs to the frame-popping loop, which is
        // `MetaInterp::finishframe_exception` and does perform it; firing
        // it from here would mark an enter/leave on a lookahead that pops
        // nothing.
        let arg1_reg = code[next.pc + 1];
        let arg2_reg = code[next.pc + 2];
        return FinishframeLookahead::RvmprofCode { arg1_reg, arg2_reg };
    }
    FinishframeLookahead::NoMatch
}

/// Convenience wrapper preserving the legacy
/// `try_catch_exception_at(...) -> Option<target>` shape used by
/// existing callers. Returns `Some(target)` only on the
/// `CatchTarget` arm; `RvmprofCode` and `NoMatch` collapse to `None`
/// (both cases continue unwinding from the caller's POV). This is a
/// lookahead predicate, not a frame-popping loop, so it carries no
/// `jit_rvmprof_code` side effect — `MetaInterp::finishframe_exception`
/// is the loop that decodes `rvmprof_code` and calls it.
pub(crate) fn try_catch_exception_at(code: &[u8], position: usize) -> Option<usize> {
    match finishframe_lookahead_at(code, position) {
        FinishframeLookahead::CatchTarget(target) => Some(target),
        FinishframeLookahead::RvmprofCode { .. } | FinishframeLookahead::NoMatch => None,
    }
}

/// Mirror of `blackhole.rs BlackholeInterpreter::handle_exception_in_frame`
/// for the walker: locate the `catch_exception/L` that owns an exception-guard
/// resume position, forward case first, then the backward scan.
///
/// Forward case (blackhole.py): the `catch_exception` sits directly after
/// the resume `-live-` — explicit `raise` sites, and residual ops whose resume
/// coordinate is their own post-call `-live-` (a mid-Python-op residual such as
/// a binary-op helper resumes at its OWN op's `-live-`, with its catch
/// following it).  Backward case: the after-residual-call layout where the
/// catch sits BEHIND the resume `-live-` (`find_catch_before_resume_live`).
/// Returns the handler target (2-byte LE label), or `None` when the raising op
/// sits outside any in-frame try (propagate).
pub(crate) fn find_catch_for_exc_resume(code: &[u8], resume_live_pos: usize) -> Option<usize> {
    let mut position = resume_live_pos;
    if let Some(op) = decode_op_at(code, position) {
        if op.key == "live/" {
            position = op.next_pc;
        }
        if let Some(op) = decode_op_at(code, position) {
            if op.key == "catch_exception/L" {
                let lo = code[position + 1] as usize;
                let hi = code[position + 2] as usize;
                return Some(lo | (hi << 8));
            }
        }
    }
    find_catch_before_resume_live(code, resume_live_pos)
}

/// Mirror of `blackhole.rs BlackholeInterpreter::find_catch_before_resume_live`
/// for the walker.  An after-residual-call exception guard resumes at the
/// no-exception fallthrough `-live-` (the next opcode after the call); the
/// `catch_exception/L` that belongs to the just-executed raising op sits BEHIND
/// that resume `-live-`.  The caught can-raise op's expansion is `[op, -live-,
/// catch_exception, -live-(block entry), vable stores…]` (the guessexception
/// split closes the block at the op's trailing `-live-`, so the successor block
/// opens with its own `-live-` before the moved stores).  Scan op boundaries
/// backward from `resume_live_pos`, newest first, hopping over exactly one
/// `-live-` (the successor's block-entry marker); accept the `catch_exception`
/// only when it sits immediately before it — any other op there is the raising
/// op itself (uncaught), and a second `-live-` means the raising op sits
/// outside any in-frame try.
/// Returns the handler target (2-byte LE label after `catch_exception/L`), or
/// `None` when the raising op sits outside any in-frame try (propagate).
/// The `catch_exception/L` target of the can-raise op whose OWN trailing
/// `-live-` sits at `resume_live_pos`.
///
/// A caught can-raise op expands to `[op, -live-, catch_exception,
/// -live-(block entry), vable stores…]`.  [`find_catch_before_resume_live`]
/// keys off the SUCCESSOR block-entry `-live-` and reads backward from it; a
/// caller holding the op's own trailing `-live-` instead is one op short of
/// that coordinate, so the same scan walks straight past the catch. Read
/// forward for those callers: when `resume_live_pos` is a `-live-` whose very
/// next op is a `catch_exception`, that catch is the one guarding the op.
pub(crate) fn catch_target_after_resume_live(code: &[u8], resume_live_pos: usize) -> Option<usize> {
    let live = decode_op_at(code, resume_live_pos)?;
    if live.key != "live/" {
        return None;
    }
    let catch = decode_op_at(code, live.next_pc)?;
    if catch.key != "catch_exception/L" {
        return None;
    }
    let lo = *code.get(catch.pc + 1)? as usize;
    let hi = *code.get(catch.pc + 2)? as usize;
    Some(lo | (hi << 8))
}

pub(crate) fn find_catch_before_resume_live(code: &[u8], resume_live_pos: usize) -> Option<usize> {
    let mut pcs: Vec<usize> = crate::jitcode_runtime::decoded_ops(code)
        .map(|op| op.pc)
        .filter(|&pc| pc < resume_live_pos)
        .collect();
    pcs.sort_unstable_by(|a, b| b.cmp(a));
    let mut crossed_block_entry_live = false;
    for pc in pcs {
        let op = decode_op_at(code, pc)?;
        if op.key == "catch_exception/L" {
            let lo = code[pc + 1] as usize;
            let hi = code[pc + 2] as usize;
            return Some(lo | (hi << 8));
        }
        if op.key == "live/" {
            if crossed_block_entry_live {
                // Second `-live-`: the raising op has no catch.
                return None;
            }
            crossed_block_entry_live = true;
            continue;
        }
        if crossed_block_entry_live {
            // The op immediately before the block-entry `-live-` is not a
            // `catch_exception` — it is the raising op itself (uncaught).
            return None;
        }
    }
    None
}

/// Byte offset of the 2-byte label operand inside an op's operand block,
/// derived from the operand signature in `key` (the part after `/`, up to the
/// `>` that introduces the result). Register operands (`i` / `r` / `f`) are one
/// byte each, so the offset is the number of them preceding the `L`.
///
/// `None` when the op carries no label, or when an operand whose width this
/// decode does not model (a var-list, a descr) sits in front of it.
fn label_operand_offset(key: &str) -> Option<usize> {
    let signature = key.split('/').nth(1)?.split('>').next()?;
    let mut offset = 0usize;
    for operand in signature.chars() {
        match operand {
            'L' => return Some(offset),
            'i' | 'r' | 'f' => offset += 1,
            _ => return None,
        }
    }
    None
}

/// How the `except` handler at a `catch_exception` target leaves the frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExcHandlerShape {
    /// Reaches `jit_merge_point`.
    Rejoins,
    /// Every scanned path returns (`*_return`); none reraise.
    Returns,
    /// Clears state and `reraise`s, and no path returns.
    Reraise,
    /// `except E as e: return` — the match path returns, the CHECK_EXC_MATCH
    /// miss reraises. `perform_call` (`pyjitpl.py`) traces the taken return;
    /// the miss is a guard side-exit, not a reason to residualize the callee.
    ExceptAsReturn,
    /// Switch, budget, or a scan that proved neither.
    Unproven,
}

/// Bounded forward reachability from `catch_target`.
pub(crate) fn exc_handler_shape(code: &[u8], catch_target: usize) -> ExcHandlerShape {
    let mut visited = std::collections::HashSet::new();
    let mut work = vec![catch_target];
    let mut budget = 4096usize;
    let mut saw_return = false;
    let mut saw_reraise = false;
    while let Some(pc) = work.pop() {
        if budget == 0 {
            return ExcHandlerShape::Unproven;
        }
        budget -= 1;
        if !visited.insert(pc) {
            continue;
        }
        let Some(op) = decode_op_at(code, pc) else {
            continue;
        };
        if op.key.starts_with("jit_merge_point") {
            return ExcHandlerShape::Rejoins;
        }
        if matches!(
            op.key,
            "ref_return/r" | "int_return/i" | "float_return/f" | "void_return/"
        ) {
            saw_return = true;
            continue;
        }
        // `raise/r` is how a bare `RERAISE` leaves the frame. It does not
        // fall through: the bytes after it are the next block, and walking
        // into them reports a `ref_return` the handler never reaches.
        if op.key.starts_with("reraise") || op.key.starts_with("raise") {
            saw_reraise = true;
            continue;
        }
        // Follow every jitcode branch that carries a target, the same
        // edges [`control_successors`] already enumerates from the
        // opcode table: `goto/L`, the `goto_if_not*` family,
        // `goto_if_exception_mismatch/iL` (`pyjitpl.py`
        // `opimpl_goto_if_exception_mismatch` jumps to `next_exc_target`),
        // and `*_jump_if_ovf`. A mismatch arm that reraises must be
        // seen, or a matching `ref_return` classifies the handler
        // `Returns`. Mixed return+reraise is `ExceptAsReturn`.
        // `switch` and an unmodelled label are Unproven.
        match control_successors(code, &op) {
            None => return ExcHandlerShape::Unproven,
            Some(succs) => work.extend(succs),
        }
    }
    if saw_return && !saw_reraise {
        ExcHandlerShape::Returns
    } else if saw_reraise && !saw_return {
        ExcHandlerShape::Reraise
    } else if saw_return && saw_reraise {
        ExcHandlerShape::ExceptAsReturn
    } else {
        ExcHandlerShape::Unproven
    }
}

fn control_successors(code: &[u8], op: &DecodedOp) -> Option<Vec<usize>> {
    let key = op.key;
    if key.starts_with("raise")
        || key.starts_with("reraise")
        || key == "unreachable/"
        || key == "void_return/"
        || key.starts_with("ref_return/")
        || key.starts_with("int_return/")
        || key.starts_with("float_return/")
    {
        return Some(Vec::new());
    }
    if key == "goto/L" {
        return Some(succ_in_code(code, read_label(code, op, 0)));
    }
    if key == "catch_exception/L" {
        return Some(succ_in_code(code, op.next_pc));
    }
    if key.starts_with("switch") {
        return None;
    }
    let labeled = label_operand_offset(key).is_some()
        || key.starts_with("goto_if")
        || key.contains("jump_if_ovf");
    if labeled {
        let offset = label_operand_offset(key)?;
        let mut succs = succ_in_code(code, read_label(code, op, offset));
        succs.extend(succ_in_code(code, op.next_pc));
        return Some(succs);
    }
    Some(succ_in_code(code, op.next_pc))
}

fn succ_in_code(code: &[u8], pc: usize) -> Vec<usize> {
    if pc < code.len() {
        vec![pc]
    } else {
        Vec::new()
    }
}

/// True when a path reachable from `position` reads the walker's active
/// exception (`last_exception/>i` / `last_exc_value/>r`) before reaching the
/// `catch_exception/L` that would replace it.
///
/// The jitcode is one flat byte stream whose blocks are stitched together by
/// `goto/L` labels, so the block that physically follows an op is frequently
/// not its successor. Advancing by `op.next_pc` alone therefore walks out of
/// the current block and answers from whatever block the assembler happened to
/// place next — reporting that block's `catch_exception/L` while the real
/// successor reads the exception. The residual-call sites act on that answer by
/// clearing `last_exc_value`, so the read then finds no active exception and
/// the walk aborts (`LastExcValueWithoutActiveException`) for the whole run.
///
/// Labels are followed instead: `goto/L` continues at its target, a
/// conditional branch queues its taken arm and continues on the fall-through,
/// and ops that end a path stop it — `catch_exception/L` because the handler it
/// introduces installs a fresh exception, `raise/r` because it replaces the
/// current one, `unreachable/` because it has no successor, and the `*_return`
/// family because the frame is gone. Targets are visited once, which bounds the
/// walk across back edges.
///
/// The answer is an over-approximation of the reads: whenever the successor set
/// is not decodable — `switch/id`, whose targets live in a descr this scan
/// cannot read, or any other label-carrying op whose shape is not modelled —
/// the exception is reported as read, as it is for an op the decode cannot read
/// at all. Over-reporting only leaves a recorded exception standing for the
/// caller, whereas under-reporting drops one a later read still needs.
fn reads_last_exc_before_next_catch(code: &[u8], position: usize) -> bool {
    // Arms queued by a conditional branch, and the jump targets already
    // started from. Both stay empty on a straight-line answer.
    let mut pending: Vec<usize> = Vec::new();
    let mut started: Vec<usize> = Vec::new();
    let mut pc = position;
    loop {
        let path_continues = match decode_op_at(code, pc) {
            // The decode fails on an opcode byte outside the instruction table
            // and on a payload the stream is too short to hold, not only at the
            // end of the code — the ops beyond it are unknown either way.
            None => return true,
            Some(op) => match op.key {
                // `opimpl_reraise` re-raises `last_exc_value`, and
                // `opimpl_goto_if_exception_mismatch` asserts it before testing
                // the class, so both read the exception exactly like
                // `last_exc_value/>r`.
                "last_exception/>i"
                | "last_exc_value/>r"
                | "reraise/"
                | "goto_if_exception_mismatch/iL" => return true,
                "catch_exception/L" | "raise/r" | "unreachable/" | "int_return/i"
                | "int_return/c" | "ref_return/r" | "float_return/f" | "void_return/" => false,
                "goto/L" => {
                    let target = read_label(code, &op, 0);
                    let fresh = !started.contains(&target);
                    if fresh {
                        started.push(target);
                        pc = target;
                    }
                    fresh
                }
                key if key.starts_with("goto_if") || key.contains("jump_if_ovf") => {
                    let Some(offset) = label_operand_offset(key) else {
                        return true;
                    };
                    let target = read_label(code, &op, offset);
                    if !started.contains(&target) {
                        started.push(target);
                        pending.push(target);
                    }
                    pc = op.next_pc;
                    true
                }
                key => {
                    if key == "switch/id" || label_operand_offset(key).is_some() {
                        return true;
                    }
                    pc = op.next_pc;
                    true
                }
            },
        };
        if !path_continues {
            match pending.pop() {
                Some(next) => pc = next,
                None => return false,
            }
        }
    }
}

/// Read a 2-byte little-endian descr index operand and resolve to
/// the descr from [`WalkContext::descr_refs`]. RPython equivalent:
/// `BlackholeInterpreter.descrs[code[pc] | (code[pc+1] << 8)]`
/// (`blackhole.py` setup + per-`bhimpl_*` site).
fn read_descr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<DescrRef, DispatchError> {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    let index = lo | (hi << 8);
    crate::jitcode_runtime::record_descr_demand(index);
    ctx.descr_refs
        .at(index)
        .ok_or(DispatchError::DescrIndexOutOfRange {
            pc: op.pc,
            index,
            len: ctx.descr_refs.len(),
        })
}

fn concrete_int_for_switch<Sym: WalkSym>(
    op: &DecodedOp,
    value: OpRef,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<i64, DispatchError> {
    match ctx.trace_ctx.concrete_of_opref(value) {
        Some(Value::Int(v)) => Ok(v),
        _ => Err(DispatchError::SwitchValueNotConcrete { pc: op.pc, value }),
    }
}

// Walker guard recording (`GuardNoException`, `GuardNotForced` after
// residual_call) pairs every guard with
// `walker_capture_snapshot_for_last_guard`, the walker-side port of
// RPython's `capture_resumedata(after_residual_call=True)`
// (`pyjitpl.py`).  RPython `pyjitpl.py
// generate_guard` walks `metainterp.framestack` and consults per-opcode
// liveness (`pyjitpl.py get_list_of_active_boxes`) to encode the
// live `i`/`r`/`f` registers in i→r→f order plus virtualizable / vref
// boxes.  Walker's helper today omits per-PC liveness narrowing (future
// follow-up: thread the `op_live` byte table through `SubJitCodeBody`)
// and conservatively snapshots every non-`OpRef::NONE` register —
// over-capture is correctness-preserving because the optimizer's
// `store_final_boxes_in_guard` (`optimizeopt/mod.rs`) derives
// `op.fail_args` from the snapshot via `store_final_boxes(liveboxes)`,
// so dead registers are dropped before they reach the backend.  Walker
// IR is no longer rolled back via `cut_trace` for the production
// dispatch; the snapshot must therefore be RPython-orthodox.

/// Read a Ref-bank variadic operand list (`R` argcode): 1 length byte
/// followed by `len` register bytes. Returns the resolved [`OpRef`]s
/// in jitcode order plus the total operand byte width (so callers can
/// skip past or compute downstream operand offsets).
///
/// RPython parity: `assembler.py:write_varlist` emits exactly this
/// shape — `chr(len(args))` followed by one byte per arg register.
fn read_ref_var_list<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_r
            .get(reg)
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_r.len(),
                bank: "r",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Read a Ref-bank register operand's concrete shadow value.
/// Mirrors [`read_ref_reg`] but indexes into
/// `ctx.concrete_registers_r`. Returns `ConcreteValue::Null` when the
/// register is out of range — symmetric with `concrete_value_at`'s
/// fallback at `state.rs`. Out-of-range OpRef reads still surface
/// `RegisterOutOfRange` via [`read_ref_reg`]; this helper assumes the
/// OpRef read succeeded, so a missing concrete slot is "stack tail not
/// yet seeded" not "register byte out of range".
fn read_ref_reg_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.frame_state
        .borrow()
        .concrete_registers_r
        .get(reg)
        .copied()
        .unwrap_or(ConcreteValue::Null)
}

/// Write a Ref-bank register and its concrete shadow in lock-step.
/// Replaces the inlined
/// `registers_r.get_mut(dst).ok_or(...)?; *slot = value` pattern at
/// every walker handler that writes `registers_r[dst]`.  The concrete
/// shadow update is the WHOLE POINT of this helper: the shadow MUST
/// stay in sync with the symbolic side or
/// downstream consumers (`raise/r` GUARD_CLASS, future
/// `getfield_gc_r` cache lookups) will silently mis-fire.
///
/// `concrete` semantics:
/// * `ConcreteValue::Ref(ptr)` — the handler knows the concrete result
///   (e.g. `ref_copy/r>r` propagating from the source slot's shadow,
///   `raise/r` setting the just-raised exception's concrete).
/// * `ConcreteValue::Null` — the handler doesn't know (most recorded
///   ops: field reads, residual calls, …).  Downstream GUARD_CLASS
///   gates treat Null as "skip the guard", same as slots the snapshot
///   never populated.
fn write_ref_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let len = ctx.registers_r.len();
    let _ = ctx
        .registers_r
        .get(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc,
            reg: dst,
            len,
            bank: "r",
        })?;
    ctx.registers_r.set(dst, value);
    // Snapshot is sized to `registers_r.len()` at dispatch entry, so
    // a dst-in-bounds OpRef write implies in-bounds for the shadow.
    // `get_mut` defensively to tolerate sub-walk shadows that lag the
    // OpRef bank if a future caller mis-sizes them.
    //
    // collapse non-Ref ConcreteValue (Int / Float)
    // to Null before storing into the Ref shadow.  `concrete_from_
    // recorded_opref` returns whatever kind the per-OpRef concrete
    // table holds; a kind mismatch (e.g. boxed Int returned through a
    // Ref result slot) would otherwise leak Int/Float bits into
    // `concrete_registers_r`, breaking ref-only downstream consumers
    // (`getfield_gc_r` sanity loads, `raise/r` GUARD_CLASS) that
    // expect `ConcreteValue::Ref(_)` or `Null`.
    let sanitized = match concrete {
        ConcreteValue::Ref(_) | ConcreteValue::Null => concrete,
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => {
            ConcreteValue::Null
        }
    };
    if let Some(c_slot) = ctx
        .frame_state
        .borrow_mut()
        .concrete_registers_r
        .get_mut(dst)
    {
        *c_slot = sanitized;
    }
    // #73: record the box just written as the candidate
    // operand-stack TOS for the current Python opcode.  A value-producing
    // opcode (LOAD_*, BINARY_OP, COPY, …) lands its result on the stack
    // TOS; this is the last Ref it writes, so capturing it here lets
    // `reconcile_vstack_at_boundary` reconstruct the new TOS without a
    // per-opcode hook.  Cheap unconditional write to a new side-field;
    // only consumed when the mirror is valid (never alters existing state).
    ctx.frame_state.borrow_mut().vstack_last_ref = value;
    Ok(())
}

/// Write a Ref register for a link renaming (`flatten.py insert_renamings`:
/// `ref_copy` / `ref_pop`) without stamping operand TOS.
///
/// A renaming moves a value that is already live into the target block's
/// register; it is not the result of the Python opcode it happens to be
/// flattened into.  The opcode's own push stamped `vstack_last_ref` at its
/// `setarrayitem_vable_r` or residual result, and a following renaming of an
/// unrelated local (a loop-carried constant, say) must not replace it.
fn write_renaming_ref_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let saved = ctx.frame_state.borrow().vstack_last_ref;
    write_ref_reg(ctx, pc, dst, value, concrete)?;
    ctx.frame_state.borrow_mut().vstack_last_ref = saved;
    Ok(())
}

/// Write a pyre scalar virtualizable Ref field without stamping operand TOS.
///
/// Pyre's scalar virtualizable fields are `last_instr(0)`, `pycode(1)`,
/// `valuestackdepth(2)` and `debugdata(3)` (`virtualizable_gen.rs`,
/// `NUM_VABLE_SCALARS = 4`). They are frame
/// bookkeeping; the Python operand stack lives in the separate
/// `locals_cells_stack_w` array (`virtualizable_gen.rs`,
/// `pyre-interpreter/src/pyframe.rs`).  PyPy's `interp_jit.py`
/// grounds the same scalar-vs-array split, and `pyjitpl.py
/// get_list_of_active_boxes` sources liveness-indexed register boxes, not
/// scalar virtualizable fields.
fn write_vable_field_ref_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let saved = ctx.frame_state.borrow().vstack_last_ref;
    write_ref_reg(ctx, pc, dst, value, concrete)?;
    ctx.frame_state.borrow_mut().vstack_last_ref = saved;
    Ok(())
}

/// One raw integer element read of `itemsize` bytes at `addr`, widened to
/// `i64` by the descr's signedness — the `bhimpl_raw_load_i` width dispatch.
///
/// # Safety
/// `addr` must name readable memory of at least `itemsize` bytes.
unsafe fn raw_load_int(addr: *const u8, itemsize: usize, signed: bool) -> i64 {
    unsafe {
        match (itemsize, signed) {
            (1, true) => i64::from(addr.cast::<i8>().read_unaligned()),
            (1, false) => i64::from(addr.read_unaligned()),
            (2, true) => i64::from(addr.cast::<i16>().read_unaligned()),
            (2, false) => i64::from(addr.cast::<u16>().read_unaligned()),
            (4, true) => i64::from(addr.cast::<i32>().read_unaligned()),
            (4, false) => i64::from(addr.cast::<u32>().read_unaligned()),
            (8, _) => addr.cast::<i64>().read_unaligned(),
            _ => panic!("raw_load descr itemsize {itemsize} is not a minted width"),
        }
    }
}

/// Store counterpart of [`raw_load_int`] — `bhimpl_raw_store_i`.
///
/// # Safety
/// `addr` must name writable memory of at least `itemsize` bytes.
unsafe fn raw_store_int(addr: *mut u8, itemsize: usize, value: i64) {
    unsafe {
        match itemsize {
            1 => addr.write_unaligned(value as u8),
            2 => addr.cast::<u16>().write_unaligned(value as u16),
            4 => addr.cast::<u32>().write_unaligned(value as u32),
            8 => addr.cast::<i64>().write_unaligned(value),
            _ => panic!("raw_store descr itemsize {itemsize} is not a minted width"),
        }
    }
}

/// Int-bank twin of [`read_ref_reg_concrete`] (Int-bank concrete shadow).
/// Reads the Int-bank slot at the operand index from
/// `ctx.concrete_registers_i`.  Returns `ConcreteValue::Null` for
/// out-of-range reads — the only legal time the slice is shorter than
/// `registers_i` is at test fixtures that pass `&mut []`, and those
/// don't trigger `goto_if_not/iL` / `switch/id` paths.
fn read_int_reg_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.concrete_registers_i
        .get(reg)
        .copied()
        .unwrap_or(ConcreteValue::Null)
}

/// Float-bank twin of [`read_int_reg_concrete`]. Reads the Float-bank slot at
/// the operand index and resolves its concrete value from the OpRef's value
/// carrier. Returns `ConcreteValue::Null` when the register is out of range or
/// has no concrete Float value.
pub(crate) fn read_float_reg_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_f
        .get(reg)
        .and_then(|value| ctx.trace_ctx.concrete_of_opref(value))
        .map_or(ConcreteValue::Null, |value| match value {
            Value::Float(v) => ConcreteValue::Float(v),
            _ => ConcreteValue::Null,
        })
}

/// Float-valued store counterpart of [`raw_store_int`] —
/// `bhimpl_raw_store_f`.
///
/// # Safety
/// `addr` must name writable memory of at least `itemsize` bytes.
unsafe fn raw_store_float(addr: *mut u8, itemsize: usize, value: f64) {
    unsafe {
        match itemsize {
            8 => addr.cast::<f64>().write_unaligned(value),
            // A single-precision element is not a Float-bank value:
            // `jit_libffi.py types` keeps the `'S'` kind residual for that
            // reason, and `_handle_raw_access_call` mints no float width but
            // `f64`.
            _ => panic!("raw_store_f descr itemsize {itemsize} is not a minted width"),
        }
    }
}

/// Int-bank twin of [`write_ref_reg`] (Int-bank concrete shadow).  Writes an Int
/// register and its concrete shadow in lock-step.  Mirrors the
/// Ref-bank contract: every walker handler that writes
/// `registers_i[dst]` MUST also write `concrete_registers_i[dst]` so
/// downstream `goto_if_not/iL` / `switch/id` can fold the branch.
///
/// `concrete` semantics:
/// * `ConcreteValue::Int(v)` — the handler knows the concrete result
///   (e.g. `int_copy/i>i` propagating from the source slot's shadow,
///   an `int_<binop>` fold of two concrete inputs).
/// * `ConcreteValue::Null` — the handler doesn't know (e.g. residual
///   `Call*I`, `getfield_gc_i` cache miss).  Downstream consumers
///   surface `GotoIfNotValueNotConcrete` for unknown branch inputs.
fn write_int_reg<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let len = ctx.registers_i.len();
    let _ = ctx
        .registers_i
        .get(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc,
            reg: dst,
            len,
            bank: "i",
        })?;
    ctx.registers_i.set(dst, value);
    // Mirror `write_ref_reg`'s defensive get_mut.  Test fixtures pass
    // an empty `concrete_registers_i` slice; production callers
    // (Concrete shadow seeding) size it to `registers_i.len()` at dispatch entry.
    //
    // Symmetry with `write_ref_reg`: collapse
    // non-Int ConcreteValue to Null before storing into the Int
    // shadow so a kind-mismatched stamp can't leak Ref/Float bits
    // into `concrete_registers_i`.
    let sanitized = match concrete {
        ConcreteValue::Int(_) | ConcreteValue::Null => concrete,
        // `is_int(bool_obj)` is true and `ConcreteValue::Bool::getint()`
        // coerces to `bool as i64`, so booleans flow safely through the
        // Int shadow.
        ConcreteValue::Bool(v) => ConcreteValue::Int(v as i64),
        ConcreteValue::Ref(_) | ConcreteValue::Float(_) => ConcreteValue::Null,
    };
    if let Some(c_slot) = ctx.concrete_registers_i.get_mut(dst) {
        *c_slot = sanitized;
    }
    Ok(())
}

/// `pyjitpl.py` `_opimpl_isconstant` / `_opimpl_isvirtual`: `ConstInt`, no IR.
fn write_hint_bool<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    predicate: bool,
) -> Result<(), DispatchError> {
    let value = i64::from(predicate);
    let result = ctx.trace_ctx.const_int(value);
    write_int_reg(ctx, pc, dst, result, ConcreteValue::Int(value))
}

/// Derive a `ConcreteValue` for shadow write-back from a freshly
/// recorded `OpRef` via `concrete_of_opref` (concrete_of_opref derivation).
///
/// RPython parity: `pyjitpl.py:execute_with_descr` /
/// `rpython/jit/metainterp/executor.py` stamps `box.value` on every
/// executed op result through the per-opcode LLOp executor — `Box.value`
/// IS the load-bearing concrete channel.  Pyre's `concrete_of_opref`
/// table-lookup is the orthodox shadow of that channel: constant pool
/// (`history.py`), virtualizable boxes
/// (`pyjitpl.py`), `set_opref_concrete` stamps from
/// `binop_int_record` / `unop_int_record`, and standard virtualizable
/// box hits all surface here.
///
/// Returns `ConcreteValue::Null` when `concrete_of_opref` yields `None`
/// (no concrete known) — the caller's downstream `goto_if_not/iL` /
/// GUARD_CLASS dispatch treats Null as "skip the fold / skip the guard".
/// A residual `Value::Ref(GcRef(usize::MAX))` storage placeholder
/// (`vable_setfield`) is likewise mapped to Null since it signals "no
/// concrete known" rather than an actual pointer.
#[inline]
fn concrete_from_recorded_opref<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    opref: OpRef,
) -> ConcreteValue {
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Int(v)) => ConcreteValue::Int(v),
        Some(Value::Float(v)) => ConcreteValue::Float(v),
        Some(Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
            ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef)
        }
        _ => ConcreteValue::Null,
    }
}

/// The `operand_offset` of this op's Ref var-list (`R`), for a reader that
/// holds only the decoded op and must find that list itself.
///
/// A dispatcher that decoded the whole op passes the offset it already
/// computed; anything reached later — an abort leg, a specializer entered
/// after resolution — has no such value and must not guess one.  `R` sits at
/// offset 1 for the Ref-only residual shape (`iRd>r`) but NOT for the mixed
/// one (`iIRd>r`, `riIRd>r`, `iiIRd>r`), whose leading Int list is itself
/// variable-width: reading offset 1 there lands on the Int list's length byte
/// and takes its register indices into the Ref bank, yielding unrelated
/// objects with a plausible length.
///
/// Walks the argcode widths of `blackhole.py:112-157`, the same walk
/// [`decode_op_at`] performs.  `None` — an op declaring no Ref list, or one
/// whose earlier operands cannot be width-counted — leaves the caller to
/// decline.
///
/// [`decode_op_at`]: crate::jitcode_runtime::decode_op_at
#[cfg(test)]
fn ref_var_list_operand_offset(code: &[u8], op: &DecodedOp) -> Option<usize> {
    let first_operand_pc = op.pc + 1;
    let mut cursor = first_operand_pc;
    let mut chars = op.argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'R' => return Some(cursor - first_operand_pc),
            'i' | 'c' | 'r' | 'f' => cursor += 1,
            'L' | 'd' | 'j' => cursor += 2,
            'I' | 'F' => cursor += 1 + *code.get(cursor)? as usize,
            '>' => {
                chars.next()?;
                cursor += 1;
            }
            _ => return None,
        }
    }
    None
}

/// Read concrete shadow values for a Ref-bank variadic operand list.
/// Parallels [`read_ref_var_list`] — reads the
/// same byte indices but resolves through `ctx.concrete_registers_r`.
/// Used by `inline_call_*` to propagate per-arg concrete shadow into
/// the callee's fresh shadow Vec.
fn read_ref_var_list_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Vec<ConcreteValue> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    (0..len)
        .map(|i| {
            let reg = code[len_pc + 1 + i] as usize;
            ctx.frame_state
                .borrow()
                .concrete_registers_r
                .get(reg)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        })
        .collect()
}

/// Read concrete shadow values for an Int-bank variadic operand list.
/// Int twin of [`read_ref_var_list_concrete`] — same byte indices,
/// resolved through `ctx.concrete_registers_i`.  Used by `inline_call_*`
/// to propagate each int arg's concrete shadow into the callee's fresh
/// shadow Vec (`setup_call` int-bank parity), so a callee body can fold
/// a `goto_if_not/iL` / `switch/id` over a concrete primitive-int arg.
fn read_int_var_list_concrete<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Vec<ConcreteValue> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    (0..len)
        .map(|i| {
            let reg = code[len_pc + 1 + i] as usize;
            ctx.concrete_registers_i
                .get(reg)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        })
        .collect()
}

/// Read an Int-bank variadic operand list (`I` argcode). Same shape as
/// [`read_ref_var_list`] but indexes into `registers_i`. RPython
/// `assembler.py:write_varlist` emits a single shape regardless of
/// kind; the kind letter (`I` / `R` / `F`) only steers which register
/// file the bytes index into.
fn read_int_var_list<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_i
            .get(reg)
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_i.len(),
                bank: "i",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Read a Float-bank variadic operand list (`F` argcode). Mirror of
/// [`read_int_var_list`] for the float bank.
fn read_float_var_list<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_f
            .get(reg)
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_f.len(),
                bank: "f",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// RPython `pyjitpl.py opimpl_switch`:
///
/// * read the traced value box and concrete `valuebox.getint()`
/// * on hit, `implement_guard_value(valuebox, orgpc)` and jump target
/// * on miss, emit `INT_EQ(valuebox, ConstInt(key))` plus `GUARD_FALSE`
///   for every `switchdict.const_keys_in_order`, then fall through
///
/// Each `GUARD_VALUE` / `GUARD_FALSE` attaches a production resume snapshot
/// via `walker_capture_snapshot_for_last_guard`, mirroring `pyjitpl.py
/// opimpl_switch`'s `generate_guard(..., resumepc=orgpc) →
/// capture_resumedata(orgpc)`.
fn dispatch_switch_id<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let valuebox = read_int_reg(code, op, 0, ctx)?;
    let descr = read_descr(code, op, 1, ctx)?;
    let switchdict = descr
        .as_switch_descr()
        .ok_or(DispatchError::ExpectedSwitchDescr { pc: op.pc })?;
    let search_value = concrete_int_for_switch(op, valuebox, ctx)?;

    if let Some(target) = switchdict.lookup(search_value) {
        if !valuebox.is_constant() {
            let expected = ctx.trace_ctx.const_int(search_value);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[valuebox, expected], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            vable_ops::walker_replace_box(ctx, valuebox, expected);
        }
        return Ok((DispatchOutcome::Continue, target));
    }

    if !valuebox.is_constant() {
        // pyjitpl.py opimpl_switch miss path — emit IntEq +
        // GuardFalse for every key in switchdict (the trace bails out
        // if any subsequent execution lands on a missed key).
        // Box(value) parity: stamp each IntEq result with the
        // (concrete_value == key) bool when valuebox's Box.value
        // resolves an Int.
        let valuebox_concrete = match ctx.trace_ctx.box_value(valuebox) {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        for &key in switchdict.const_keys_in_order() {
            let keybox = ctx.trace_ctx.const_int(key);
            // pyjitpl.py `opimpl_switch` miss uses
            // `execute(INT_EQ, ...)` for each key.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::IntEq, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::IntEq, majit_metainterp::counters::RECORDED_OPS);
            let eqbox = ctx.trace_ctx.record_op(OpCode::IntEq, &[valuebox, keybox]);
            if let Some(v) = valuebox_concrete {
                ctx.trace_ctx
                    .set_opref_concrete(eqbox, majit_ir::Value::Int((v == key) as i64));
            }
            ctx.trace_ctx.record_guard(OpCode::GuardFalse, &[eqbox], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Bridge an `MIFrame`'s register banks +
/// trace recorder + last-exc state into a `WalkContext` and run
/// `walk()` against the supplied jitcode body.
///
/// RPython parity context: in RPython the metainterp loop iterates
/// over `metainterp.framestack[-1].pc` calling `bytecode_step` which
/// dispatches to the right `opimpl_*`. There's no separate "walker
/// entry" because the metainterp loop *is* the walker. This entry
/// point is pyre's equivalent: it drives the walker against the
/// supplied `MIFrame` state and is the sole production tracing path.
///
/// Field plumbing:
/// * `registers_r/i/f` — allocated fresh per call sized to
///   `top_num_regs_* + top_constants_*.len()`, then populated by
///   the inline `setup_call` from `argboxes_*` and constant slots.
///   PyPy parity: `pyjitpl.py MIFrame.__init__` allocates
///   the bank vectors at frame construction; `:188 setup_call`
///   populates slots `[0..argboxes.len())` from the caller's
///   argboxes. Walker handlers writing dst slots (`int_copy`,
///   `binop_int_record`, etc.) mutate them in place; the banks are
///   dropped when this function returns (matching PyPy's per-frame
///   lifetime).
/// * `trace_ctx` — borrowed mutably from `miframe.ctx`'s
///   `TraceCtx`. Recording (`record_op`, `finish`, etc.) goes
///   through this.
/// * `last_exc_value` — reads `sym.last_exc_box` as the initial
///   value (`OpRef::NONE` collapses to `None`). On exit the
///   walker's final `last_exc_value` is mirrored back if non-None,
///   so a `raise/r` -> `catch_exception/L` -> handler trace
///   leaves `sym.last_exc_box` pointing at the in-flight exc OpRef
///   (parity with RPython metainterp.last_exc_value).
/// * `descr_refs`, `sub_jitcode_lookup` — caller-provided, same
///   contract as direct `walk()` callers. Production callers wire
///   `crate::jitcode_runtime::all_descrs()` + a JitCode-resolving
///   closure over the lazy runtime jitcode cache.
///
/// `is_top_level` selects the outer-frame semantic:
///
/// * `true` — outermost trace entry. `*_return/*` arms record
///   `Finish(value, done_with_this_frame_descr_<kind>)` and a `raise/r`
///   that is never caught records
///   `Finish(exc, exit_frame_with_exception_descr_ref)`.
/// * `false` — sub-frame entry: `*_return/*` arms surface
///   `SubReturn { result }` and uncaught `raise/r` arms surface
///   `SubRaise { exc }` to the caller. The shadow validator (Phase
///   D-3) drives this for per-Python-opcode arms — a Python-opcode arm
///   compiled by the codewriter ends with `*_return/*` (since each arm
///   is a self-contained sub-jitcode invoked from the outer dispatcher
///   via `inline_call_r_r/dR>r`), so shadow mode must NOT emit a FINISH
///   per Python opcode.
///
/// Sub-walks driven by `inline_call_r_r/dR>r` recursion always set
/// `is_top_level=false` regardless of this caller-side flag (the
/// recursion constructs its own `WalkContext`).
///
/// **Production wiring**: the full-body-walk walker
/// (`full_body_walk_trace`) is the caller, dispatching each JitCode
/// opcode through this entry as it walks the body.
///
/// Bridge exception seeding follows `pyjitpl.py`: a
/// `ResumeGuardExcDescr`/`ResumeGuardCopiedExcDescr` source guard carries
/// exception state into bridge tracing. A non-exception-guard bridge whose
/// entry path reads `last_exc_value` before the next `catch_exception/L`
/// (the `except E as name:` cleanup tail) seeds from the frame's current
/// exception (`ExecutionContext.sys_exc_value`) instead — the same slot the
/// original trace already reads. Operand-stack values are never scanned to
/// infer a standing exception.
fn seed_standing_exception_for_walk<Sym: WalkSym>(
    sym: &mut Sym,
    trace_ctx: &mut TraceCtx,
    code: &[u8],
    position: usize,
) {
    if trace_ctx.is_bridge_trace
        && !trace_ctx.bridge_source_is_exception_guard()
        && !reads_last_exc_before_next_catch(code, position)
    {
        return;
    }

    // `_prepare_exception_resumption` (pyjitpl.py) reads the
    // exception off THIS failure's deadframe (`cpu.grab_exc_value`), and every
    // bridge compile runs on a fresh `MetaInterp`, so a previous compile's
    // `last_exc_value` can never leak into a new bridge trace.  Pyre's sym
    // persists across walks; the routed publish (`BH_LAST_EXC_VALUE`, set from
    // the same grab) is the fresh-failure signal and must OVERWRITE any
    // exception state a previous walk left on the sym.  A preseeded sym is
    // kept only when no fresh signal exists — the multi-frame carrier walk
    // re-seeds per frame after the first frame drained the cell.
    // For the walks that reach here this cell is the delivery path for
    // `_prepare_exception_resumption`: `call_jit.rs` publishes
    // `cpu.grab_exc_value`'s result before bridge tracing starts and zeroes
    // it when the guard carried no exception.  On an exception-guard
    // bridge one of the two arms below always returns, so a pre-seed
    // placed on the sym in `setup_bridge_sym` could only rewrite the
    // pointer this read applies anyway.  The multi-frame carrier walk
    // seeds the same cell in `drive_bridge_carrier_walk`.
    let bh_exc = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
    if bh_exc != 0 {
        let exc = bh_exc as pyre_object::PyObjectRef;
        if !exc.is_null() && unsafe { pyre_object::is_exception(exc) } {
            let exc_box = trace_ctx.const_ref(exc as i64);
            sym.set_current_exc_value(exc);
            sym.set_current_exc_box(exc_box);
            sym.set_last_exc_value(exc);
            sym.set_last_exc_box(exc_box);
            sym.set_class_of_last_exc_is_const(true);
            return;
        }
    }

    if trace_ctx.is_bridge_trace && trace_ctx.bridge_source_is_exception_guard() {
        // The carrier already copied this failure's grab onto the sym.
        // A residual between that copy and this seed clears the cell, so
        // an empty cell is not "the deadframe published nothing".
        if trace_ctx.bridge_grab_seeded() && !sym.last_exc_value().is_null() {
            return;
        }
        // Null arm (`pyjitpl.py:3152-3154`): the deadframe published NO
        // exception, and for an exception-guard bridge that publish is the
        // sole authority — `clear_exception()` on the fresh MetaInterp.
        // Keeping a preseeded sym here would walk the exception-flavor
        // continuation for a no-exception failure: a loop whose exception
        // guard was recorded through a raising iteration then compiles the
        // HANDLER as its no-exception bridge, and every non-raising
        // iteration runs the handler body.
        sym.set_current_exc_value(pyre_object::PY_NULL);
        sym.set_current_exc_box(OpRef::NONE);
        sym.set_last_exc_value(pyre_object::PY_NULL);
        sym.set_last_exc_box(OpRef::NONE);
        sym.set_class_of_last_exc_is_const(false);
        return;
    }

    if !sym.last_exc_box().is_none() {
        return;
    }

    let current = pyre_interpreter::eval::get_current_exception();
    if !current.is_null() && unsafe { pyre_object::is_exception(current) } {
        let exc_box = standing_exc_box_from_current(sym, trace_ctx, current);
        sym.set_current_exc_value(current);
        sym.set_current_exc_box(exc_box);
        sym.set_last_exc_value(current);
        sym.set_last_exc_box(exc_box);
        // `execute_ll_raised`: the class is a Const only after
        // GUARD_EXCEPTION or GUARD_CLASS. A live `GetfieldGcR` of
        // `sys_exc_value` has no such guard, so the flag stays false and
        // `handle_possible_exception` keeps the guard op as `last_exc_box`
        // instead of a `ConstPtr` of the recording-time object.
        // `pin_nonconst_standing_exception_class` records the GUARD_CLASS
        // (`opimpl_raise`) once the walk can capture its resume snapshot.
        sym.set_class_of_last_exc_is_const(exc_box.is_constant());
    }
}

/// `opimpl_raise` / `handle_possible_exception`: a seeded exception whose
/// box is not a constant gets `GUARD_CLASS` before
/// `class_of_last_exc_is_const` becomes true. Consumers
/// (`opimpl_last_exception`, `opimpl_goto_if_exception_mismatch`) then
/// fold the class off a pinned value.
pub(crate) fn pin_nonconst_standing_exception_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if ctx.fbw_mode.class_of_last_exc_is_const {
        return Ok(());
    }
    let Some(exc) = ctx.last_exc_value() else {
        return Ok(());
    };
    if exc.is_constant() {
        ctx.fbw_mode.class_of_last_exc_is_const = true;
        return Ok(());
    }
    if ctx.trace_ctx.heap_cache().is_class_known(exc) {
        ctx.fbw_mode.class_of_last_exc_is_const = true;
        return Ok(());
    }
    let ConcreteValue::Ref(exc_ptr) = ctx.last_exc_value_concrete() else {
        return Ok(());
    };
    if exc_ptr.is_null() || !unsafe { pyre_object::is_exception(exc_ptr) } {
        return Ok(());
    }
    let exc_class_ptr = unsafe {
        (*(exc_ptr as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type
    };
    let cls_const = ctx.trace_ctx.const_int(exc_class_ptr as usize as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardClass, &[exc, cls_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    ctx.trace_ctx.heap_cache_mut().class_now_known(exc);
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    Ok(())
}

/// Box for a current-exception seed: a live `GetfieldGcR` of
/// `ExecutionContext.sys_exc_value` on a bridge that has the portal EC red,
/// otherwise a const-ref of the recording-time pointer.
fn standing_exc_box_from_current<Sym: WalkSym>(
    sym: &Sym,
    trace_ctx: &mut TraceCtx,
    current: pyre_object::PyObjectRef,
) -> OpRef {
    let ec = sym.execution_context();
    if trace_ctx.is_bridge_trace && !ec.is_none() {
        let exc_box = trace_ctx.record_op_with_descr(
            OpCode::GetfieldGcR,
            &[ec],
            crate::descr::ec_sys_exc_value_descr(),
        );
        trace_ctx.set_opref_concrete(exc_box, Value::Ref(majit_ir::GcRef(current as usize)));
        exc_box
    } else {
        trace_ctx.const_ref(current as i64)
    }
}

/// Resolve the top-level walk's live frame code when the promoted `pycode`
/// green has no concrete shadow.
///
/// RPython keeps `MIFrame.jitcode` on each live frame and
/// `opimpl_jit_merge_point` reads the greens from that frame.  The full-body
/// walker has the same per-frame identity in `snapshot_sym.jitcode`; using it
/// here restores the missing concrete shadow without borrowing a caller or
/// portal-global anchor.  Inline sub-walks deliberately do not use this
/// fallback: their code comes from their own `InlineCalleeConsts`, and the
/// existing loop-callee handling below decides whether they may cross a merge
/// point.
fn top_level_live_code<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> Option<*const ()> {
    if !ctx.is_top_level || ctx.fbw_mode.snapshot_sym.is_null() {
        return None;
    }
    let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
    if sym.jitcode().is_null() {
        return None;
    }
    let raw = unsafe { (&(*sym.jitcode()).payload).code_ptr };
    if raw.is_null() {
        None
    } else {
        Some(pyre_interpreter::live_code_wrapper(raw as *const ()) as *const ())
    }
}

fn guard_current_frame_globals_identity<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    expected_globals: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    if expected_globals.is_null() {
        return Ok(false);
    }
    // `pyframe.py:LOAD_GLOBAL` reads `self.get_w_globals()` from the live
    // MIFrame.  A non-portal inline level has its own green pycode and thus
    // its own constant `pycode.w_globals`; the metainterp-wide
    // `virtualizable_boxes` still describes the portal/root frame and must not
    // be used to guard this callee's namespace.  `inline_callee_consts` is the
    // current inline frame's counterpart (also used by
    // `try_resolve_inline_callee_static_field` above).  Checking that frame's
    // identity here prevents a bridge-resumed callee from emitting
    // `GUARD_VALUE(root.w_globals, callee.w_globals)`.
    if ctx.inline_callee_consts.is_some() {
        return Ok(ctx.inline_w_globals() == expected_globals as usize);
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(false);
    }
    let w_globals_op = unsafe { (&*sym_ptr).frame_w_globals() };
    if w_globals_op.is_none() {
        return Ok(false);
    }
    let expected = ctx.trace_ctx.const_ref(expected_globals as i64);
    if w_globals_op.is_constant() {
        if ctx.trace_ctx.const_value(w_globals_op) != Some(expected_globals as i64) {
            return Ok(false);
        }
        // The constant is the RECORDING frame's namespace, published from
        // `pycode.w_globals`, which is green because `pycode.py
        // frame_stores_global` keeps the FIRST namespace a code object runs
        // under.  It therefore cannot tell that frame apart from one running
        // the same code object under a second `exec(code, ns)`: that frame
        // carries its own namespace in `debugdata` and still enters this loop
        // under the same `(code, pc)` warm key.  AGENTS.md "One red frame per
        // frame" — read the live red frame instead of trusting the anchor.
        //
        // `pyframe.py get_w_globals` is `debugdata.w_globals` when the frame
        // carries a payload and `promote(pycode).w_globals` when it does not,
        // so emit that shape.  `debugdata` is a virtualizable field, so the
        // read answers from `virtualizable_boxes` and records no op — the same
        // route `fast2locals`' fold takes for the neighbouring `w_locals`.
        let Some((debugdata_op, majit_ir::Value::Ref(recorded))) = ctx
            .trace_ctx
            .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
        else {
            return Ok(false);
        };
        if recorded == majit_ir::GcRef::NO_CONCRETE {
            // "no recorded word" is not "recorded a null word", and it is also
            // not an address to read a payload out of.
            return Ok(false);
        }
        // `createframe_obj` allocates the payload only when
        // `frame_stores_global` reports the frame's namespace differs from the
        // published one, so the recorded direction is what separates the two
        // arms — and each is pinned so a frame on the other one side-exits.
        let present = recorded.as_usize() != 0;
        if present {
            // Read the override through the SHADOW's payload, which is what the
            // emitted `getfield_gc_r` reads, and decline before emitting
            // anything if it is not the namespace this fold answers with.  A
            // tracing shadow hands out a clone of the payload around the same
            // fields, so the value is the invariant to check — never the
            // holder's address.
            let shadow = recorded.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
            if unsafe { (*shadow).w_globals } != expected_globals {
                return Ok(false);
            }
        }
        // Every LOAD_GLOBAL in the frame reaches here, and the emitter's
        // `_establish_nullity` consult is what keeps the second one from
        // spending `trace_limit` on a guard the first already carries.
        let opcode = if present {
            OpCode::GuardNonnull
        } else {
            OpCode::GuardIsnull
        };
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, opcode, &[debugdata_op])?;
        if !present {
            // No payload, so the namespace IS `pycode.w_globals` — the constant
            // already compared above.
            return Ok(true);
        }
        // The payload's `w_globals` is the authority, and it is a mutable
        // field, so the read stays a real load off the live frame's payload.
        // The payload address itself is per-frame — every call allocates one —
        // so only the value it holds can be guarded.
        let live = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            debugdata_op,
            crate::descr::frame_debug_data_w_globals_descr(),
        );
        if !live.is_constant() {
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[live, expected],
            )?;
            // The guard proved the read; hand the heapcache the constant so the
            // next fold site reads it back instead of guarding again.
            ctx.trace_ctx.heap_cache_mut().replace_box(live, expected);
        }
        return Ok(true);
    }
    // pypy/objspace/std/celldict.py `elidable_promote('0,1,2')` promotes
    // `w_dict`; mirror that by pinning the runtime frame's w_globals identity.
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[w_globals_op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    vable_ops::walker_replace_box(ctx, w_globals_op, expected);
    Ok(true)
}

fn replace_movable_load_global_namespace_with_frame_globals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    runtime_helper: majit_ir::RuntimeHelperKind,
    r_args: &mut [OpRef],
) {
    if runtime_helper != majit_ir::RuntimeHelperKind::LoadGlobal {
        return;
    }
    let Some(ns_box) = r_args.first_mut() else {
        return;
    };

    // `pyframe.py get_w_globals LOAD_GLOBAL` asks the current MIFrame for
    // `self.get_w_globals()`.  An inline sub-walk therefore has to source the
    // namespace through that level's own frame red, just as `setup_call`
    // installs one frame per inlined call in `pyjitpl.py`.  The
    // metainterp-wide virtualizable boxes below belong only to the portal/root
    // frame; substituting their namespace here collapses caller and callee
    // identity and makes a cross-module callee execute LOAD_GLOBAL against the
    // caller's globals.
    if let Some(consts) = ctx.inline_callee_consts {
        let Some(jitcode) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
            return;
        };
        let Some(frame) = ctx
            .registers_r
            .get(jitcode.metadata.portal_frame_reg as usize)
            .filter(|frame| !frame.is_none())
        else {
            // The old single-frame inline path has no callee red to read.  Do
            // not borrow the root frame — that collapses caller and callee
            // identity — but the callee's OWN `__globals__` is recorded right
            // here, and `try_walker_load_global_cell_fold`'s builtins leg is
            // written for exactly this case (`frame_ptr == 0`, deriving the
            // builtin module from the namespace's `__builtins__` cell).  Naming
            // it directly is what lets that leg run; leaving the codewriter's
            // null placeholder standing declines the fold and then aborts the
            // trace on the residual's unbound callee-frame argument.
            //
            // Name the callee's own `__globals__` as a `ConstPtr`.  That word
            // is forwarded in `WalkFrameState.inline_w_globals` and, once
            // recorded, by `walk_const_ptr_refs` / the gcref table.
            let w_globals = ctx.inline_w_globals();
            if w_globals != 0 {
                *ns_box = ctx.trace_ctx.const_ref(w_globals as i64);
            }
            return;
        };
        let w_globals = crate::state::frame_get_globals_obj(ctx.trace_ctx, frame);
        // Recording-time residual execution needs the same concrete shadow;
        // the compiled value remains the GETFIELD_GC_R result above and is
        // therefore still sourced from the live callee frame.
        ctx.trace_ctx.try_set_opref_concrete(
            w_globals,
            Value::Ref(majit_ir::GcRef(ctx.inline_w_globals())),
        );
        *ns_box = w_globals;
        return;
    }

    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return;
    }
    let w_globals_op = unsafe { (&*sym_ptr).frame_w_globals() };
    if !w_globals_op.is_none() {
        *ns_box = w_globals_op;
    }
}

/// Line-by-line port of `pyjitpl.py MetaInterp._build_allboxes`.
/// Permutes a flat `argboxes` array (concat of i-list ++ r-list ++ f-list
/// in source order) so positions match the callee's `descr.get_arg_types()`
/// ABI ordering. Returns `[funcbox, ...permuted_argboxes]`.
///
/// RPython operates on a flat `argboxes` of typed `Box` objects + reads
/// `box.type`. The walker has only `OpRef`s, so the type is supplied
/// out-of-band via the `argbox_types` parallel array. By construction
/// the operand decoders (`read_int_var_list` / `read_ref_var_list` /
/// `read_float_var_list`) tag each entry with its bank, so the parallel
/// array is correct without needing a runtime type query.
///
/// `pyjitpl.py _build_allboxes(..., prepend_box=)`. Residual calls pass
/// `None`; `opimpl_conditional_call*` prepends the condition/value box.
fn build_allboxes(
    funcbox: OpRef,
    argboxes: &[OpRef],
    argbox_types: &[Type],
    arg_types: &[Type],
    prepend_box: Option<OpRef>,
) -> Vec<OpRef> {
    debug_assert_eq!(
        argboxes.len(),
        argbox_types.len(),
        "argboxes and argbox_types must align",
    );
    // RPython line 1961: `allboxes = [None] * (len(argboxes)+1 + …)`.
    let extra = usize::from(prepend_box.is_some());
    let total = arg_types.len() + 1 + extra;
    let mut allboxes: Vec<OpRef> = Vec::with_capacity(total);
    if let Some(pre) = prepend_box {
        allboxes.push(pre);
    }
    // RPython line 1966: `allboxes[i] = funcbox`.
    allboxes.push(funcbox);
    // RPython line 1968: `src_i = src_r = src_f = 0`.
    let mut src_i = 0usize;
    let mut src_r = 0usize;
    let mut src_f = 0usize;
    // RPython line 1969-1989: outer `for kind in descr.get_arg_types()`
    // with one type-filter `while True` loop per kind.
    for &kind in arg_types {
        let box_oref = match kind {
            Type::Int => loop {
                // RPython line 1971-1975: advance src_i past non-INT
                // entries until an INT box is found.
                let b = argboxes[src_i];
                let bt = argbox_types[src_i];
                src_i += 1;
                if bt == Type::Int {
                    break b;
                }
            },
            Type::Ref => loop {
                // RPython line 1977-1981.
                let b = argboxes[src_r];
                let bt = argbox_types[src_r];
                src_r += 1;
                if bt == Type::Ref {
                    break b;
                }
            },
            Type::Float => loop {
                // RPython line 1983-1987 (kind == 'L' long-long path
                // not separately modeled — pyre's Type::Float covers
                // both).
                let b = argboxes[src_f];
                let bt = argbox_types[src_f];
                src_f += 1;
                if bt == Type::Float {
                    break b;
                }
            },
            // RPython line 1988-1989: `else: raise AssertionError`.
            // Type::Void in arg_types is an internal invariant violation.
            Type::Void => panic!("_build_allboxes: arg_types must not contain Void"),
        };
        allboxes.push(box_oref);
    }
    debug_assert_eq!(allboxes.len(), total, "allboxes shape post-condition");
    allboxes
}

/// Decode the descr index from a 2-byte LE operand. Companion to
/// [`read_descr`] for callers that need the raw index for error
/// reporting (e.g. `ResidualCallDescrNotCallDescr`).
fn decode_descr_index(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// `pyjitpl.py heapcache.call_loopinvariant_known_result`
/// short-circuit: when the EffectInfo's extraeffect is `EF_LOOPINVARIANT`
/// AND the heapcache has a cached result for `(descr_index, allboxes[0])`,
/// the trace skips re-recording the `CALL_LOOPINVARIANT_*` op and the
/// caller reuses the cached OpRef.  Returns `None` for non-loopinvariant
/// EI or a cache miss; the caller then falls through to the normal record
/// path and follows up with [`loopinvariant_now_known`] to populate the
/// cache for subsequent matching calls.
///
/// RPython upstream (`heapcache.py`) keys the lookup by descr
/// **identity** and `allboxes[0].getint()`.  Upstream's
/// `do_residual_or_indirect_call` (`pyjitpl.py`) reaches
/// `do_residual_call` for **both** non-`Const` `funcbox` and `Const`
/// funcboxes whose address has no registered jitcode — the
/// `isinstance(funcbox, Const)` guard only short-circuits to
/// `perform_call` when `bytecode_for_address` resolves a jitcode.
/// In the residual path `allboxes[0].getint()` is well-defined
/// regardless of `Const`-ness because every Box subclass exposes
/// `getint()` over its runtime int (`history.BoxInt._value` /
/// `ConstInt.value`).
///
/// pyre's [`TraceCtx::concrete_of_opref`]
/// (`majit-metainterp/src/trace_ctx.rs`) reconstructs the
/// concrete int from the per-trace constant pool only for constant
/// OpRefs; non-constant OpRefs are symbolic at the dispatcher
/// layer and carry no runtime int the trace-time walker can read.
/// When `funcptr` is non-constant we skip the cache entirely —
/// using `funcptr.0` as a sentinel would key on symbolic identity
/// rather than the concrete callee, risking false hits across two
/// different non-const funcptrs that share an OpRef after IR
/// renaming, and false misses across two different OpRefs aliasing
/// the same concrete callee.  Returning `None` is the conservative
/// choice (the caller falls through to record the call); the cost
/// is a missed cache hit on non-const funcptrs that upstream would
/// have caught.  Convergence with upstream's full coverage requires
/// threading concrete-int shadow alongside OpRef for non-const ints
/// — significant work remaining.
///
/// `descr_key` is the descriptor's stable identity key (`Descr::index()`),
/// matching upstream's identity comparison on `descr` more closely than
/// the operand-encoded descr-table slot.
#[inline]
fn loopinvariant_lookup<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    descr_key: u32,
    funcptr: OpRef,
) -> Option<OpRef> {
    if ei.extraeffect != majit_ir::ExtraEffect::LoopInvariant {
        return None;
    }
    let arg0_int = funcptr_concrete_int(ctx, funcptr)?;
    ctx.trace_ctx
        .heap_cache()
        .call_loopinvariant_known_result(descr_key, arg0_int)
        .map(|(opref, _resvalue)| opref)
}

/// `pyjitpl.py heapcache.call_loopinvariant_now_known`: after
/// recording a fresh `CALL_LOOPINVARIANT_*` op, remember the
/// `(descr_index, allboxes[0].getint()) -> result` mapping so the
/// next matching call short-circuits via [`loopinvariant_lookup`].
/// No-op for non-loopinvariant EI, and no-op when `funcptr` is
/// non-constant (no concrete int key — see [`loopinvariant_lookup`]).
///
/// `resvalue` is stored as `0`.  RPython's upstream caller
/// (`pyjitpl.py`) stores `res` — the concrete value returned
/// by `execute_varargs` after actually running the callee.  pyre-
/// jit-trace records symbolically without executing, so no concrete
/// result exists at this point: `record_op_with_descr` returns a
/// freshly-minted SSA OpRef whose runtime value is only known when
/// the compiled trace later executes.  The cached `_resvalue` is
/// unused by [`loopinvariant_lookup`]'s consumer (only the symbolic
/// OpRef is read for register writeback), so the `0` placeholder is
/// observationally equivalent.  Convergence with upstream requires
/// either threading the concrete result up from the executing trace
/// (concrete shadow tracking) or dropping the field from
/// the cache shape entirely (separate cleanup).
#[inline]
fn loopinvariant_now_known<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    descr_key: u32,
    funcptr: OpRef,
    result: OpRef,
) {
    if ei.extraeffect != majit_ir::ExtraEffect::LoopInvariant {
        return;
    }
    let Some(arg0_int) = funcptr_concrete_int(ctx, funcptr) else {
        return;
    };
    ctx.trace_ctx
        .heap_cache_mut()
        .call_loopinvariant_now_known(descr_key, arg0_int, result, 0);
}

#[allow(dead_code)] // codewriter.py exception-handler scan
fn walk_body_has_exception_handler<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    code: &[u8],
) -> bool {
    let jitcode_index = if ctx.is_top_level {
        ctx.session.borrow().recording_jitcode_index
    } else {
        ctx.inline_callee_consts
            .map_or(-1, |consts| consts.jitcode_index)
    };
    crate::state::jitcode_source_has_exception_handler(jitcode_index).unwrap_or_else(|| {
        crate::jitcode_runtime::decoded_ops(code).any(|op| op.opname == "catch_exception")
    })
}

/// Resolve a residual-call funcptr OpRef to the concrete function
/// pointer integer that RPython's heapcache keys on
/// (`heapcache.py` calls `allboxes[0].getint()`).
///
/// Returns `Some(int)` when `funcptr` is a constant int OpRef whose
/// value lives in pyre's per-trace constant pool. For the RPython-legal
/// `EF_LOOPINVARIANT` direct-call producer, `call.py` asserts
/// no runtime args and emits a constant function box; that is the path
/// this cache is meant to mirror. General residual calls can arrive
/// from indirect calls with non-constant funcboxes
/// (`pyjitpl.py`), so `None` means "skip the loop-invariant
/// cache" rather than inventing an alias-prone sentinel key.
#[inline]
fn funcptr_concrete_int<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    funcptr: OpRef,
) -> Option<i64> {
    if !funcptr.is_constant() {
        return None;
    }
    match ctx.trace_ctx.concrete_of_opref(funcptr) {
        Some(majit_ir::Value::Int(v)) => Some(v),
        _ => None,
    }
}

/// PyPy `_opimpl_residual_call{1,2,3}` (pyjitpl.py) port for residual calls
/// not completed by [`try_fold_pure_call_via_executor`].  That fast path
/// handles cannot-raise elidable calls; can-raise elidable calls continue here
/// so their exception state is observable.
///
/// Upstream calls `executor.execute_varargs(opnum, argboxes, descr,
/// exc=can_raise, pure=is_elidable)` which dispatches through
/// `cpu.bh_call_*` and clears/records BH exception state via
/// `metainterp.clear_exception()` + `metainterp.execute_raised(...)`.
/// This walker-friendly counterpart returns `Result<i64, i64>` directly
/// (via [`majit_metainterp::executor::execute_residual_call`]) so the
/// caller can wire the BH exception into `WalkContext.last_exc_value`
/// without dragging in a `MetaInterp` seam.
///
/// **Why this exists**: the walk is the sole execution leg for the
/// traced iteration — the synchronous marker walk returns its terminal
/// outcome without re-running `execute_opcode_step`. For opcodes whose body contains a
/// non-elidable `residual_call_*` (`store_subscr_fn` /
/// `set_current_exception` / etc.), the helper is never invoked → heap
/// mutation never happens → next read derefs stale container → SIGBUS
/// (5 STORE_SUBSCR-hot benches).  The
/// orthodox fix is to widen this function across `Call*` /
/// `CallLoopinvariant*` / `CallMayForce*` shapes, mirroring PyPy's
/// `_opimpl_residual_call*` which concrete-executes *every* residual
/// call regardless of EI.
///
/// **Caller contract**:
/// * `call_opcode` must be one of the residual call shapes:
///   `Call{I,R,F,N}` (the default arm), `CallLoopinvariant{I,R,F,N}`
///   (the `EF_LOOPINVARIANT` arm), or `CallMayForce{I,R,F,N}` (the
///   `forces_virtual_or_virtualizable` arm — only when no active
///   virtualizable is present, see force-virtual gate below).
///   * can-raise `CallPure*` is included; cannot-raise `CallPure*` has already
///     been handled by [`try_fold_pure_call_via_executor`].
///   * `CallReleaseGil*` / `CallAssembler*` are excluded from THIS
///     function's direct opcode set, but for distinct reasons:
///     - `CallReleaseGil*` IS concrete-executed — `do_residual_call`
///       (`pyjitpl.py`) runs step 2 with `opnum1 =
///       CALL_MAY_FORCE_*` for the *whole* forces branch, and release-gil
///       is a sub-case inside it.  [`direct_call_release_gil`] records the
///       `CALL_RELEASE_GIL_*` op then re-enters this function with a
///       `CallMayForce*` opcode and the original `allboxes`, so the GIL
///       wrapper runs once during tracing exactly like a may-force call
///       (the GIL transition is inert single-threaded during recording).
///     - `CallAssembler*` is correctly NOT executed here: an assembler
///       token re-enters the JIT, it is not a plain C func address, so the
///       raw-`func_ptr` executor would fault.  Recursive `CALL_ASSEMBLER`
///       is folded by [`try_walker_call_assembler_self_recursive`], which
///       concrete-executes the callee through its own `CallMayForceR` path
///       on the real callable (token lives only in the recorded op).
///   * **`CallMayForce*` vable token protocol**: PyPy `do_residual_call`
///     (`pyjitpl.py`) concrete-executes every `CallMayForce*`
///     via `executor.execute_varargs`, bracketed by the heap halves of
///     the token protocol: `vinfo.tracing_before_residual_call(virtualizable)`
///     (pyjitpl.py, sets `TOKEN_TRACING_RESCALL`) before the
///     call and `vinfo.tracing_after_residual_call(virtualizable)`
///     (pyjitpl.py) after it.  This function mirrors both
///     halves around `execute_residual_call` whenever the jitdriver has
///     an active `standard_virtualizable_box()` with a live heap pointer
///     — the bracket `vable_and_vrefs_before_residual_call` /
///     `vable_after_residual_call` describes upstream.  A
///     cleared token after the call means the callee forced the
///     virtualizable: surface [`DispatchError::VableEscapedDuringResidualCall`]
///     (`SwitchToBlackhole(ABORT_ESCAPE)` parity, pyjitpl.py).
///     With no active vable the bracket is skipped — nothing to force.
///     The IR half (`FORCE_TOKEN` + `SETFIELD_GC(vable_token_descr)`)
///     stays at the dispatcher's
///     [`maybe_walker_vable_and_vrefs_before_residual_call`] call site.
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// **Authoritative-executor gate**: fires ONLY when the walk is the
/// sole concrete-execution leg
/// ([`WalkContext::is_authoritative_executor`]) — the production
/// full-body walk and its inline sub-walks qualify
/// (`eval_loop_jit` skips `execute_opcode_step` for walker-handled
/// opcodes).  In shadow / diagnostic-probe mode the flag is `false`, so
/// the call is recorded symbolically only — re-executing there would
/// double the side effects (the concrete interpreter already ran it)
/// or corrupt the live heap under the discard-the-trace probe.
///
/// **Return value**:
/// * `Ok(ResidualExecOutcome::Executed(Ok(_)))` — helper executed normally, `recorded` OpRef
///   stamped with the concrete result.
/// * `Ok(ResidualExecOutcome::Executed(Err(bh_exc)))` — helper raised; `bh_exc` is the wrapped
///   `PyError` pointer (from `BH_LAST_EXC_VALUE`).  Caller is
///   responsible for routing into `WalkContext.last_exc_value` so the
///   downstream `GuardNoException` walker handler picks it up; the
///   `recorded` OpRef is NOT stamped (no concrete result).
/// * `Ok(ResidualExecOutcome::Declined(_))` — fold declined (preconditions not met: not the
///   authoritative executor, opcode out of set, funcbox non-const, arity
///   exceeds [`MAX_HOST_CALL_ARITY`], any operand lacks a concrete
///   `box_value`, or any Ref arg is NULL), or the call's result is void
///   (recorded symbolically; the compiled loop applies the side effect —
///   see the void branch below).  The trace still has the recorded call
///   op for the optimizer to consume later; walker falls through as if
///   this function did not exist.
/// * `Err(VableEscapedDuringResidualCall)` — the callee forced the
///   active virtualizable during a `CallMayForce*`; the walk must abort
///   (pyjitpl.py ABORT_ESCAPE parity, see the token-protocol bullet
///   above).
///
/// **Wire status**: invoked from all three
/// dispatch entry points (`dispatch_residual_call_iRd_kind`,
/// `dispatch_residual_call_iIRd_kind`, `dispatch_residual_call_iIRFd_kind`)
/// alongside [`try_fold_pure_call_via_executor`].  The may-force /
/// can-raise path wires the `Err` exception through
/// `WalkContext.last_exc_value`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidualDecline {
    ValueUnavailable,
    Symbolic,
    /// A CANNOT-raise `CallPure*` the fold could not answer, because some
    /// argbox is still symbolic.  Recording it and moving on IS the tracing
    /// behaviour — an elidable call that cannot raise has neither an effect for
    /// anything to apply later nor an exception to miss — so unlike the other
    /// two this decline leaves no obligation behind.  An `EF_ELIDABLE_CAN_RAISE`
    /// callee wears the same opcode but is declined as `Symbolic`, because the
    /// raise it may owe is an observable a no-replay walk-end road would drop.
    PureUnfolded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidualExecOutcome {
    Executed(Result<i64, i64>),
    Declined(ResidualDecline),
}

/// `pyjitpl.py` short-circuit guard.  RPython `do_residual_call`
/// runs `_do_jit_force_virtual` (`pyjitpl.py`) when
/// `effectinfo.oopspecindex == OS_JIT_FORCE_VIRTUAL`:
///
/// ```text
/// def _do_jit_force_virtual(self, allboxes):
///     if (self.jitdriver_sd.virtualizable_info is None and
///         self.jitdriver_sd.greenfield_info is None):
///         return None
///     if len(allboxes) == 2:
///         [vrefbox] = allboxes[1:]
///         standard_box = self.virtualizable_boxes[-1].getref_base()
///         if standard_box != vrefbox.getref_base():    # concrete pointer compare
///             return None
///         return self.virtualizable_boxes[-1]
///     ...
/// ```
///
/// PyPy returns one of `vref_box` / `standard_box` / `None`; on the
/// `None` fall-through it records the normal `CALL_MAY_FORCE_*`.  The
/// walker CANNOT reproduce this faithfully because the comparison
/// `standard_box.getref_base() != vrefbox.getref_base()` requires a
/// concrete `*mut PyObject` for the `vrefbox` Ref OpRef; pyre's
/// symbolic walker only carries `concrete_vable_ptr` for the active
/// virtualizable, not a per-OpRef → concrete-pointer map.
///
/// Choices considered:
///  1. Silent fall-through (always record `CALL_MAY_FORCE_*`) — would
///     silently emit IR on a path PyPy folds away whenever the vref
///     IS the standard virtualizable.  Trace divergence with no diff
///     report; rejected.
///  2. OpRef-equality short-circuit (`if vrefbox_opref == standard_box_opref:
///     short-circuit; else fall through`) — sufficient for the same-OpRef
///     case but UNSOUND for the different-OpRef-same-concrete-pointer case
///     (walker would record `CALL_MAY_FORCE_*` while PyPy folds; mismatch
///     vs. live tracer's IR).  Rejected.
///  3. Fail-loud (current).  STRICTER than PyPy: the walker stops with a
///     typed error rather than emit divergent IR, which immediately
///     flags any producer that starts emitting
///     `OopSpecIndex::JitForceVirtual`.
///
/// Convergence path back to 1:1 PyPy parity: an OpRef → concrete-pointer
/// resolver.  When that lands the guard becomes a real
/// `_do_jit_force_virtual()` body that returns `Some(vref_opref)` /
/// `Some(standard_opref)` / `None` and threads through the dispatcher
/// like the release-gil short-circuit does today.
///
/// Production reach today: zero — `OopSpecIndex::JitForceVirtual` is
/// set only by `jtransform.rs jit.force_virtual` lowering, which
/// our benchmarks don't reach.  The guard is fail-loud futureproofing
/// for the day a producer (e.g. an explicit `jit.force_virtual` callee
/// from `pyre_interpreter`) lights up the path.
#[inline]
fn do_jit_force_virtual_guard(ei: &majit_ir::EffectInfo, pc: usize) -> Result<(), DispatchError> {
    if ei.oopspecindex == OopSpecIndex::JitForceVirtual {
        return Err(DispatchError::JitForceVirtualRequiresConcreteResolver { pc });
    }
    Ok(())
}

/// `pyjitpl.py _get_list_of_active_boxes` parity for the
/// walker-emitted snapshot: read each live register from its
/// kind-specific bank in (int, ref, float) order, dropping non-live
/// slots regardless of whether the OpRef happens to be set.  The
/// liveness lookup matches the decoder side
/// (`state::frame_value_count_at` /
/// `frame_liveness_reg_indices_by_bank_at`), so encoder and decoder
/// agree byte-for-byte on the snapshot shape consumed at resume.
///
/// Returns an empty vector when no liveness is registered for the
/// `(jitcode_index, pc)` pair (skeleton payload or out-of-range PC);
/// the downstream optimizer surfaces the empty snapshot as a no-op.
/// `true` when `op` is a `ConstPtr` carrying the NULL gcref (`GcRef(0)`) —
/// the encoding an unset vable shadow slot decodes to.  Distinct from
/// `OpRef::NONE` (no value at all).
fn opref_is_null_const_ptr(op: OpRef) -> bool {
    op.as_const_ptr().is_some_and(|g| g.is_null())
}

/// The source selected for `collect_outer_active_boxes` entry metadata.  The
/// audit compares the JitCode-PC variants to the legacy Python-PC tables.
#[derive(Clone, Copy)]
enum OuterActiveBoxesEntryTwin {
    Plain,
    Trivia,
}

impl OuterActiveBoxesEntryTwin {
    #[allow(dead_code)]
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Trivia => "trivia",
        }
    }
}

fn snapshot_diag_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PYRE_SNAPSHOT_DIAG").is_some())
}

fn collect_outer_active_boxes<Sym: WalkSym>(
    sym: &Sym,
    trace_ctx: &mut TraceCtx,
    regs_i: &(impl RegisterValues + ?Sized),
    regs_r: &(impl RegisterValues + ?Sized),
    regs_f: &(impl RegisterValues + ?Sized),
    outer_jitcode_index: u32,
    guard_present: bool,
    // The resume-carried coordinate remains the bank-liveness source. Entry
    // metadata can be derived from an earlier raw operation coordinate, so it
    // must travel independently rather than inheriting that marker word.
    carried_jitcode_pc: i32,
    entry_jitcode_pc: i32,
    entry_twin: OuterActiveBoxesEntryTwin,
    entry_caller: &'static str,
    vstack: Option<&[OpRef]>,
    kept_recovered: &[(u16, OpRef)],
    // Set to the first live Ref color whose operand-stack slot has NO per-slot
    // source (mirror hole + no edge-move recovery), so the branch-guard caller
    // can decline instead of encoding the stale merge-color read.  `None` for
    // the callers that are not reconstructing a branch guard.
    mut unrecovered_kept: Option<&mut Option<u32>>,
) -> Vec<OpRef> {
    // `#124` Approach B: resolve the base live-box set through the carried
    // JitCode coordinate so the encoder's color set matches the decoder's
    // (`setup_bridge_sym` / `rebuild_inline_callee`), which read the same
    // carried word.  With the flag off `carried_jitcode_pc` is ignored and
    // this is the plain py_pc→jitcode-translation query.
    let banks = crate::state::frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
        outer_jitcode_index as i32,
        carried_jitcode_pc,
    );
    let mut active = Vec::with_capacity(banks.int.len() + banks.ref_.len() + banks.float.len());
    // RPython `pyjitpl.py _get_list_of_active_boxes` reads
    // `self.registers_X[index]` directly per liveness index.  Pyre
    // diverges on the Ref bank for portal-owner frames: the
    // `registers_r` semantic-mirror write in
    // `write_stack_slot` (trace_opcode.rs) is retired so stack-slot colors
    // sit at `OpRef::NONE` in `sym.registers_r`; the authoritative
    // shadow lives in `trace_ctx.virtualizable_boxes`.  Codewriter
    // liveness also force-alives `portal_frame_reg` / `portal_ec_reg`
    // (codewriter.rs `filter_liveness_in_place`) — these are
    // scratch colors past `nlocals + max_stackdepth` that have no
    // semantic frame slot, sourced instead from `sym.frame` /
    // `sym.execution_context` (`interp_jit.py reds = ['frame', 'ec']`).
    //
    // Mirror the retired MIFrame snapshot materializer
    // (trace_opcode.rs `get_list_of_active_boxes`): map each
    // live Ref color to its semantic index via
    // `semantic_ref_slot_for_reg_color`, read the vable shadow for
    // portal-owner frames, and route the two portal red regs through
    // `sym.frame` / `sym.execution_context` directly.
    let (nlocals, valid_stack_only, owns_vable, portal_frame_reg, portal_ec_reg) =
        if sym.jitcode().is_null() {
            (0usize, 0usize, false, u16::MAX, u16::MAX)
        } else {
            unsafe {
                let jc = &*sym.jitcode();
                let payload = &jc.payload;
                // Operand-stack depth at the snapshot coordinate. The liveness
                // banks (`frame_liveness_reg_indices_by_bank_at`) are read at
                // that coordinate too, so the per-PC color→slot window
                // (`pcdep_opt`'s stack clamp below) must use its depth — NOT
                // `sym.valuestackdepth` (the walker's *current* position). For
                // the per-opcode entry caller the two coincide, but a guard
                // resuming at a not-taken branch target with a kept operand-stack
                // temp (conditional expr / short-circuit / chained compare,
                // #124/#281) resumes at a depth `> 0` while the walker stands at
                // depth 0 — using the current depth there drops the kept temp's
                // semantic slot, corrupting the frame.
                let stack_depth_at_pc = if payload.code_ptr.is_null() {
                    0usize
                } else {
                    match (entry_twin, entry_jitcode_pc >= 0) {
                        (OuterActiveBoxesEntryTwin::Plain, true) => payload
                            .depth_for_jitcode_pc_pred(entry_jitcode_pc as usize)
                            .unwrap_or(0)
                            as usize,
                        (OuterActiveBoxesEntryTwin::Trivia, true) => payload
                            .depth_trivia_for_jitcode_pc(entry_jitcode_pc as usize)
                            .unwrap_or(0)
                            as usize,
                        // Defensive default for a genuinely absent entry
                        // coordinate (marker miss); 0-fire across the corpus.
                        _ => 0,
                    }
                };
                (
                    sym.nlocals(),
                    stack_depth_at_pc,
                    sym.owns_virtualizable_shadow(),
                    payload.metadata.portal_frame_reg,
                    payload.metadata.portal_ec_reg,
                )
            }
        };
    // #348: per-snapshot-coordinate color→slot entries — the color→slot source
    // the inversions below consult for every drained (non-portal) jitcode, the
    // per-program-point color space the `-live-` markers carry. Branch-guard
    // resumes (`guard_present`) are fully covered here; a per-opcode-entry
    // resume whose live Ref colors are all constants/leaked carries no entry (the
    // resume snapshot records Variables only), so `semantic_ref_slot_for_reg_color`
    // returns `None` and the live color falls to the `regs_r[color]` walk-bank
    // read below. Portal-bridge frames carry no `pcdep` entry (empty by install),
    // so for them `pcdep_opt` is `None`, `semantic_ref_slot_for_reg_color`
    // returns `None`, and every live color falls to the `regs_r[color]`
    // walk-bank read.
    let pcdep_entries: Vec<(u8, u16, u16)> = if sym.jitcode().is_null() {
        Vec::new()
    } else {
        unsafe {
            let jc = &*sym.jitcode();
            match (entry_twin, entry_jitcode_pc >= 0) {
                (OuterActiveBoxesEntryTwin::Plain, true) => jc
                    .payload
                    .pcdep_for_jitcode_pc(entry_jitcode_pc as usize)
                    .unwrap_or_default(),
                (OuterActiveBoxesEntryTwin::Trivia, true) => jc
                    .payload
                    .pcdep_trivia_for_jitcode_pc(entry_jitcode_pc as usize)
                    .map(ToOwned::to_owned)
                    .unwrap_or_default(),
                // Defensive default for a genuinely absent entry coordinate
                // (marker miss); 0-fire across the corpus.
                _ => Vec::new(),
            }
        }
    };
    let pcdep_opt: Option<&[(u8, u16, u16)]> =
        (!pcdep_entries.is_empty()).then(|| pcdep_entries.as_slice());
    let (guard_pcdep_entries, guard_stack_only) = if guard_present {
        if sym.jitcode().is_null() {
            (Vec::new(), 0usize)
        } else {
            unsafe {
                let jc = &*sym.jitcode();
                let cjc = carried_jitcode_pc;
                let entries = if cjc >= 0 {
                    jc.payload
                        .pcdep_for_jitcode_pc(cjc as usize)
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let depth = if jc.payload.code_ptr.is_null() {
                    0usize
                } else if cjc >= 0 {
                    jc.payload
                        .depth_for_jitcode_pc_pred(cjc as usize)
                        .unwrap_or(0) as usize
                } else {
                    0
                };
                (entries, depth)
            }
        }
    } else {
        (Vec::new(), 0usize)
    };
    // Int / Float bank diagnostic panic: pyre's banks are sized to the
    // jitcode's `num_regs_X`, which the codewriter co-publishes with the
    // liveness side-table, so every liveness index is in range by
    // construction.  A miss here is a tracer-side invariant violation
    // (size mismatch) — panic loudly so the bug surfaces at the encode
    // site instead of bleeding NONE values into `encode_snapshot_boxes`
    // where `get_opref_type(NONE)` panics with no breadcrumb pointing at
    // the source.
    let live_i = banks.int.clone();
    let live_r = banks.ref_.clone();
    let live_f = banks.float.clone();
    let (ni, nr, nf) = (regs_i.len(), regs_r.len(), regs_f.len());
    let vable_len = trace_ctx.virtualizable_boxes_len().unwrap_or(0);
    // Int / Float bank candidates: pyre's vable static fields decode as
    // Int (last_instr, valuestackdepth, etc.); if liveness expects an Int
    // register that the register banks did not fill, the candidate fill is
    // one of these sym shadow OpRefs.  Including them in the diagnostic
    // lets fallback work jump straight to the right source.
    let vable_vsd = sym.vable_valuestackdepth();
    let vable_last_instr = sym.vable_last_instr();
    let dump_ctx = |bank: &'static str, reg_idx: u32| -> String {
        let int_hint = if bank == "int" {
            format!(
                ", sym.vable_valuestackdepth={vable_vsd:?}, sym.vable_last_instr={vable_last_instr:?}"
            )
        } else {
            String::new()
        };
        format!(
            "collect_outer_active_boxes: liveness-active {bank} \
             register {reg_idx} holds OpRef::NONE \
             (outer_jitcode_index={outer_jitcode_index}, entry_jitcode_pc={entry_jitcode_pc}, \
              nlocals={nlocals}, owns_vable={owns_vable}, \
              vable_len={vable_len}, \
              num_regs_i={ni}, num_regs_r={nr}, num_regs_f={nf}, \
              live_banks_i={live_i:?}, live_banks_r={live_r:?}, live_banks_f={live_f:?}\
              {int_hint})",
        )
    };
    for &idx in &banks.int {
        let v = regs_i
            .get_box(idx as usize)
            .unwrap_or_else(|| panic!("{}", dump_ctx("int", idx)));
        if v == OpRef::NONE {
            panic!("{}", dump_ctx("int", idx));
        }
        active.push(v);
    }
    // #420: exact not-taken edge-move recovery, keyed by resume-merge color.
    // The guard trampoline's `ref_copy(dst <- src)` gave this kept slot's live
    // guard-pc source value directly, exact for any kept-stack depth.
    let kept_recovered: std::collections::HashMap<u32, OpRef> = kept_recovered
        .iter()
        .map(|&(dst, v)| (dst as u32, v))
        .collect();
    // Mirror-sourced operand-stack slots: the walk-level box mirror
    // (`ctx.vstack_boxes`, indexed by absolute operand-stack depth) is the
    // analog of `MIFrame.registers_r` for the valuestack.  A branch guard
    // uses it for the not-taken arm's kept bottom; a CALL-site helper
    // snapshot uses it for the live receiver / argument boxes so a guard
    // inside the helper resumes those same OpRefs rather than a peel
    // ConstPtr.  The merge-color heuristics below remain the fallback for
    // slots the mirror does not cover.
    let vstack_mirror = vstack;
    for &idx in &banks.ref_ {
        let color = idx as usize;
        // The semantic slot this color owns AT THE GUARD PC.  Where it equals
        // the slot being sourced, `regs_r[color]` is upstream's
        // `registers_r[index]` (`get_list_of_active_boxes`, `pyjitpl.py`) for
        // that slot rather than a merge color the regalloc may have reused;
        // both the mirror-hole report and the operand-stack value choice below
        // turn on that identity.
        let guard_owned_slot = guard_present
            .then(|| {
                crate::state::semantic_ref_slot_for_reg_color(
                    nlocals,
                    guard_stack_only,
                    &guard_pcdep_entries,
                    color,
                )
            })
            .flatten();
        if let Some(mirror) = vstack_mirror {
            // Resume operand-stack slot for this live Ref color; the mirror
            // box for that slot is the kept value (bottom-anchored: resume
            // slot `s` == `vstack_boxes[s]`, the same mapping `mirror_covers_
            // kept` validates).
            if let Some(sem) = crate::state::semantic_ref_slot_for_reg_color(
                nlocals,
                valid_stack_only,
                pcdep_opt.unwrap_or(&[]),
                color,
            ) {
                if sem >= nlocals {
                    let m = mirror.get(sem - nlocals).copied().unwrap_or(OpRef::NONE);
                    // The walk-level mirror is a sparse, per-opcode value
                    // source.  A present CONST_NULL is therefore an actual
                    // PUSH_NULL/self_or_null operand, not the dense vable
                    // shadow's "slot was never written" encoding.  PyPy's
                    // MIFrame registers preserve CONST_NULL in snapshots;
                    // keep it here as well.  Only OpRef::NONE is a hole.
                    if m != OpRef::NONE {
                        // A CALL-site helper snapshot uses the same mirror
                        // for the live receiver.  Its PUSH_NULL/self_or_null
                        // slot is a typed ConstPtr(NULL); overlaying that
                        // through a reused color would plant NULL on a live
                        // Ref (pickle async-forcing then reads an unwritten
                        // result home).  Branch-guard reconstruction still
                        // needs the NULL: that is the kept PUSH_NULL.
                        if guard_present || !opref_is_null_const_ptr(m) {
                            active.push(m);
                            continue;
                        }
                    }
                    // The mirror did not supply this operand-stack slot.  Two
                    // other per-slot sources remain, and only the absence of
                    // BOTH is a hole: the decoded edge-move recovery
                    // (`kept_recovered`, tested below) and the virtualizable
                    // shadow, which the resolution further down already treats
                    // as authoritative for a slot the guard pc's color map does
                    // not claim (`shadow_is_real` → `vbox`).  Where neither
                    // answers, the fallback reads the resume merge color out of
                    // the guard-pc register file, unwritten at the guard point
                    // or reused by the regalloc — the #424 staleness the
                    // per-slot mirror replaced; report that case so the caller
                    // declines rather than encode the stale read.  Declining
                    // where the shadow does answer is not free: the capture
                    // point sits after a residual has run, so it lands as a
                    // `fbw_rolled_back_with_effects` double-apply
                    // (`raise_reg_unbound_jitstress`, mirror at depth 1 against
                    // a guard resuming at depth 3).
                    //
                    // Restricted to a NONE hole, where there is demonstrably no
                    // source at all.  A NULL const-ptr is NOT declined, and that
                    // is a measured decision rather than an oversight: a census
                    // over `test_re`, `test_strftime` and all 393 synth programs
                    // found 3 mirror-missed slots, every one a NULL const-ptr
                    // with `recovered=false` — so the gate's justification is
                    // indeed false there — but all 3 are in one PASSING program
                    // (`synth/list_append_write_barrier_gc`), and declining them
                    // regressed it (loops 12→11, bridges 5→3, aborts 1→4) with
                    // no demonstration that the value encoded today is wrong.
                    //
                    // A hole the guard pc's OWN color→slot map resolves is not
                    // reported at all.  `get_list_of_active_boxes`
                    // (`pyjitpl.py`) snapshots `self.registers_r[index]` and
                    // never fails, because the register file IS the source; the
                    // stale-read hazard this report exists for is the case
                    // where that identity does not hold — the resume merge
                    // color is unwritten at the guard point, or the regalloc
                    // reused it for an unrelated SSA temp.  When
                    // `pcdep_color_slots` AT THE GUARD PC maps this color to
                    // this very semantic slot, the two coincide: the walk
                    // register read below means exactly `registers_r[index]`,
                    // and the stack-slot arm consumes it through the same
                    // `guard_pc_proves_slot` test.  That also settles what the
                    // mirror cannot: a slot whose live value is a NULL — an
                    // already-cleared operand, a `self_or_null` — is a hole in
                    // the mirror by construction, because
                    // `reseed_vstack_from_shadow` reads a dense array where an
                    // absent slot and a written NULL are the same word and so
                    // refuses to source either.  The register file draws that
                    // distinction, and upstream keeps the NULL: PyPy's MIFrame
                    // registers preserve CONST_NULL in snapshots.
                    //
                    // The shadow is the other source the resolution below
                    // already consults: for an operand-stack slot the guard
                    // pc's color map does not claim, it reads
                    // `virtualizable_box_at(NUM_VABLE_SCALARS + sem)`.  A hole
                    // that arm resolves must not be reported either, or the
                    // decline lands as `fbw_rolled_back_with_effects` at a
                    // capture point a residual has already run past — on
                    // `raise_reg_unbound_jitstress` the walk mirror stands at
                    // depth 1 while the branch guard resumes at depth 3, and
                    // both kept slots read NONE from the mirror and a live
                    // `RefOp` from the shadow.
                    if m == OpRef::NONE
                        && !kept_recovered.contains_key(&idx)
                        && guard_owned_slot != Some(sem)
                    {
                        let shadow_sources_slot = trace_ctx
                            .virtualizable_box_at(crate::virtualizable_gen::NUM_VABLE_SCALARS + sem)
                            .is_some_and(|b| {
                                b != OpRef::NONE
                                    && !opref_is_null_const_ptr(b)
                                    && b.ty() == Some(majit_ir::Type::Ref)
                            });
                        if !shadow_sources_slot {
                            // The operand-stack mirror has no box, but the
                            // register for this live color still holds one
                            // (the value `LOAD_FAST` left there, for
                            // instance). `_get_list_of_active_boxes` reads
                            // `registers_r[index]` for every live color with
                            // no further test, so encode that register.
                            //
                            // A bridge inputarg left in the color after the
                            // virtualizable slot moved on is not that value.
                            // The merge point publishes the shadow box
                            // (`reached_loop_header` `live_arg_boxes +=
                            // virtualizable_boxes`). Naming the stale
                            // inputarg makes the loop cut decline a box the
                            // entry contract does not carry. Fall through
                            // and let the shadow arm below choose.
                            let reg = regs_r.get_box(color).unwrap_or(OpRef::NONE);
                            if reg != OpRef::NONE && !reg.is_input_arg() {
                                active.push(reg);
                                continue;
                            }
                            if let Some(first) = unrecovered_kept.as_deref_mut() {
                                first.get_or_insert(idx);
                            }
                        }
                    }
                }
            }
        }
        if let Some(&rv) = kept_recovered.get(&idx) {
            // Edge-move-resolved kept operand; overrides the unwritten
            // merge-color read for this not-taken-arm operand-stack slot.
            active.push(rv);
            continue;
        }
        let fallback = || {
            regs_r
                .get_box(color)
                .unwrap_or_else(|| panic!("{}", dump_ctx("ref", idx)))
        };
        let semantic_idx = crate::state::semantic_ref_slot_for_reg_color(
            nlocals,
            valid_stack_only,
            pcdep_opt.unwrap_or(&[]),
            color,
        );
        // Portal-red routing applies only to the force-alived SCRATCH case
        // (`filter_liveness_in_place` keeps `portal_frame_reg`/`portal_ec_reg`
        // in every `-live-` R-bank even where the color names no frame slot).
        // If the walk's live register bank already carries a real box, snapshot
        // that box first: it is the direct `get_list_of_active_boxes`
        // (`pyjitpl.py`) source for this live color, and bridge resume uses
        // the same color-indexed bank to recover the EC red (`state.rs`).
        // Fall back to the named red field only when the bank is empty.  If both
        // are empty, the force-alived red is dead at this capture point, so encode
        // `CONST_NULL` (history.py), matching the union-liveness arm below.
        //
        // The jitcode register allocator ALSO assigns these colors to real frame
        // slots at other PCs (e.g. a call-result register live across a later
        // call), where `pcdep_color_slots` maps the color to a semantic slot.
        // Routing such a colliding color to `sym.frame`/`sym.execution_context`
        // encodes the wrong box; a color that names a live semantic slot therefore
        // takes the normal slot-value paths below.
        let is_portal_red_scratch = semantic_idx.is_none()
            && ((color as u16 == portal_frame_reg && portal_frame_reg != u16::MAX)
                || (color as u16 == portal_ec_reg && portal_ec_reg != u16::MAX));
        let value = if is_portal_red_scratch {
            let live_reg = regs_r
                .get_box(color)
                .filter(|&v| v != OpRef::NONE && !opref_is_null_const_ptr(v));
            let red_field = if color as u16 == portal_frame_reg {
                sym.frame()
            } else {
                sym.execution_context()
            };
            live_reg
                .or_else(|| {
                    (red_field != OpRef::NONE && !opref_is_null_const_ptr(red_field))
                        .then_some(red_field)
                })
                .unwrap_or_else(|| OpRef::const_ptr(majit_ir::GcRef(0)))
        } else if owns_vable {
            match semantic_idx {
                Some(s_idx) if s_idx < nlocals + valid_stack_only => {
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    let vbox = trace_ctx.virtualizable_box_at(nvs + s_idx);
                    let walk_box = regs_r.get_box(color);
                    if s_idx >= nlocals {
                        // Operand-stack slot.  `pyjitpl.py` snapshots
                        // `self.registers_r[index]`, but pyre's stack
                        // `write_ref_reg` mirror was retired: outside a
                        // branch guard's own per-PC color map, the walk
                        // register may hold a stale value from a prior SSA def
                        // that shared the color.  Prefer the live register
                        // (the upstream source) when the guard PC's
                        // `pcdep_color_slots` proves this color owns the same
                        // stack slot at the guard capture point — there the
                        // register read means exactly `registers_r[index]`;
                        // where ownership is unprovable the virtualizable
                        // shadow remains authoritative.
                        //
                        // "Authoritative" deliberately does NOT extend to a NULL
                        // shadow slot.  Upstream reads the operand stack out of
                        // one list — `create_top_snapshot(frame, vable_boxes,
                        // ...)` (`opencoder.py`) and `live_arg_boxes +=
                        // self.virtualizable_boxes` (`pyjitpl.py:2987`) are the
                        // same object — so it is tempting to let the shadow
                        // answer here too and keep a post-header snapshot inside
                        // the merge point's published set.  Measured: it is
                        // wrong.  Letting the NULL answer makes
                        // `synth/nested_break_not_hot` raise `TypeError:
                        // comparison on null operand` in `plain_break`, because
                        // the traced arm has already popped a slot the guard's
                        // resume target still reads.  The shadow under-reports
                        // that slot; the walk register carries its live box.
                        let shadow_is_real = vbox.is_some_and(|b| !opref_is_null_const_ptr(b));
                        let walk_real =
                            walk_box.filter(|&v| v != OpRef::NONE && !opref_is_null_const_ptr(v));
                        let shadow_real =
                            vbox.filter(|&b| b != OpRef::NONE && !opref_is_null_const_ptr(b));
                        let guard_pc_proves_slot = guard_owned_slot == Some(s_idx);
                        // A bridge inputarg left in the color is not the
                        // stack temp. The merge point's loop inputargs are
                        // the shadow boxes; naming the inputarg makes the
                        // loop cut decline it. A NULL shadow still loses to
                        // the register (`nested_break_not_hot`).
                        if let (Some(w), Some(s)) = (walk_real, shadow_real) {
                            if w != s && w.is_input_arg() {
                                s
                            } else if guard_pc_proves_slot {
                                w
                            } else {
                                s
                            }
                        } else if guard_pc_proves_slot {
                            walk_real.or(vbox).unwrap_or_else(fallback)
                        } else if shadow_is_real {
                            vbox.unwrap_or_else(fallback)
                        } else {
                            walk_real.unwrap_or_else(|| vbox.unwrap_or_else(fallback))
                        }
                    } else {
                        // `pyjitpl.py get_list_of_active_boxes` reads
                        // `self.registers_r[index]`, and a local's register is
                        // written in lock-step with the virtualizable array, so
                        // the live register is the box — the shadow is only the
                        // fallback for colors the retired mirror left empty.
                        //
                        // The color→slot map that named this a local is a
                        // per-program-point label, not a binding: the regalloc
                        // freely reuses a local's color for an unrelated SSA
                        // temp (a residual-call result, a raise operand), and
                        // the map still reports the slot.  Preferring the
                        // shadow there substituted the local's value for the
                        // temp — a `raise` whose operand color doubled as
                        // `self` then resumed publishing `self` as the
                        // exception.
                        //
                        // A bridge inputarg left in the register after the
                        // virtualizable slot moved on is not that temp. The
                        // merge point publishes the shadow box
                        // (`reached_loop_header` `live_arg_boxes +=
                        // virtualizable_boxes`), and `CutTrace` treats that
                        // box as a loop inputarg. Naming the stale inputarg
                        // instead makes the loop cut decline a box the entry
                        // contract does not carry.
                        let walk_real =
                            walk_box.filter(|&b| b != OpRef::NONE && !opref_is_null_const_ptr(b));
                        let shadow_real =
                            vbox.filter(|&b| b != OpRef::NONE && !opref_is_null_const_ptr(b));
                        match (walk_real, shadow_real) {
                            (Some(w), Some(s)) if w != s && w.is_input_arg() => s,
                            _ => walk_real.or(vbox).unwrap_or_else(fallback),
                        }
                    }
                }
                // `semantic_idx` is `None`: this Ref color names no live
                // frame slot at the resume PC.  Under the canonical splice a
                // single `-live-` marker is SHARED across a range of Python
                // PCs at different stack depths, and its liveness banks carry
                // the UNION of those PCs' live colors.  Resuming at a
                // shallower PC therefore sees colors that are dead here:
                //   - a mapped frame stack slot BEYOND this resume's live
                //     window (`semantic_ref_slot_for_reg_color` is `None`
                //     because the slot sits past `valid_stack_only`), or
                //   - under free (non-identity) coloring, a NON-frame SSA
                //     temp (in neither the local nor the stack color map)
                //     that is live only at another PC the marker spans.
                // Both are dead at this PC: the trace produced no box for a
                // dead value, so `registers_r[color]` is NONE.  Both decode
                // paths already tolerate this — the blackhole rebuild
                // (`state.rs` ref-bank loop) and the bridge decoder drop a
                // Ref color whose semantic slot is `None`.  Keep encode
                // symmetric: substitute `CONST_NULL` (history.py) so the
                // positional snapshot/liveness count stays aligned rather
                // than failing loud on the unsourceable dead slot.  A color
                // the trace DID produce here (e.g. an op hoisted ahead of its
                // consumer block's marker so it is live-in at the resume
                // marker) carries its real `registers_r` box — `regs_r[color]`
                // is non-NONE precisely when the value is genuinely live, so
                // this read is the same `get_list_of_active_boxes` parity the
                // mapped arm above uses.  Under the walker (gate-off) each PC
                // owns a depth-narrowed marker, so this arm never fires.
                None => match regs_r.get_box(color) {
                    // A produced box is live at this marker even though the
                    // color names no frame slot.
                    Some(v) if v != OpRef::NONE && !v.is_input_arg() => v,
                    // A bridge inputarg that the virtualizable shadow still
                    // holds is one of the merge point's loop inputargs.
                    // One the shadow has replaced is a stale color: the
                    // decoder drops a slot-less color, and leaving the old
                    // inputarg in the snapshot makes the loop cut decline
                    // a box the entry contract does not carry.
                    Some(v)
                        if v.is_input_arg()
                            && (0..trace_ctx.virtualizable_boxes_len().unwrap_or(0))
                                .any(|i| trace_ctx.virtualizable_box_at(i) == Some(v)) =>
                    {
                        v
                    }
                    _ => OpRef::const_ptr(majit_ir::GcRef(0)),
                },
                _ => fallback(),
            }
        } else {
            fallback()
        };
        if value == OpRef::NONE {
            panic!("{}", dump_ctx("ref", idx));
        }
        // `PYRE_SNAPSHOT_DIAG=1`: report every live Ref color whose snapshot
        // box did NOT come from the walk register bank.  `pyjitpl.py get_list_of_active_boxes
        // get_list_of_active_boxes` reads `registers_r[index]` unconditionally,
        // so each of these is a place where pyre's virtualizable-shadow
        // preference decided otherwise — the resume value and the register the
        // resumed bytecode reads then disagree.
        if snapshot_diag_enabled() {
            let walk = regs_r.get_box(color).unwrap_or(OpRef::NONE);
            if walk != OpRef::NONE && walk != value {
                eprintln!(
                    "[snap] jc={outer_jitcode_index} carried={carried_jitcode_pc} \
                     entry={entry_jitcode_pc} color={color} semantic_idx={semantic_idx:?} \
                     guard_present={guard_present} nlocals={nlocals} \
                     stack_only={valid_stack_only} walk={walk:?} chose={value:?}"
                );
            }
        }
        active.push(value);
    }
    for &idx in &banks.float {
        let v = regs_f
            .get_box(idx as usize)
            .unwrap_or_else(|| panic!("{}", dump_ctx("float", idx)));
        if v == OpRef::NONE {
            panic!("{}", dump_ctx("float", idx));
        }
        active.push(v);
    }
    active
}

/// Sync the symbolic vable `last_instr` at an intermediate `jit_merge_point`
/// reached mid-trace (the `jit_merge_point` REGISTER branch — a bridge
/// re-entering the inner loop, returning `Continue`): override `last_instr`
/// to `merge_pc - 1` (a resume into the target loop must re-enter at the
/// header opcode) and mirror it into the `virtualizable_boxes` shadow that
/// `close_loop_args_at`'s JUMP-arg derivation reads.
///
/// Without the override the JUMP into the existing loop carries the LAST
/// GUARD's published `last_instr` (e.g. 104 instead of header-1=86 on
/// fannkuch), so a vable sync inside the target loop would resume the
/// interpreter at the wrong bytecode (permutation state never reaches its
/// exit condition → non-crashing infinite loop).
///
/// `interp_jit.py` `jump_absolute` writes `frame.last_instr` before
/// `can_enter_jit`. Snapshot capture save/restores the boxes; this merge
/// write is the live-frame header pc.
fn sync_intermediate_merge_point_last_instr(ctx: &mut TraceCtx, merge_pc: usize) {
    if ctx.standard_virtualizable_box().is_none() {
        return;
    }
    let last_instr_value = merge_pc as i64 - 1;
    let opref = ctx.const_int(last_instr_value);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx,
        "last_instr",
        opref,
        Value::Int(last_instr_value),
    );
}

/// Call-scoped inputs for one guard-snapshot capture — the explicit-parameter
/// port of `capture_resumedata`'s keyword arguments (pyjitpl.py),
/// alongside the already-explicit `after_residual_call`.
#[derive(Clone, Copy, Default)]
pub(crate) struct GuardCaptureScope<'a> {
    /// Request that a residual call's after-call guard route its resume
    /// through the call's OWN post-call `catch_exception` instead of the
    /// generic post-call fallthrough. This is derived automatically from the
    /// guard opcode; scoped callers may also request it explicitly. The
    /// snapshot helper only acts on this when the call's CALL pc is directly
    /// covered by an enclosing
    /// exception-table handler: it then carries the CALL jitcode offset so a
    /// deopt resumes at the call's own catch. The blackhole's
    /// `handle_exception_in_frame` routes the raise to the enclosing handler
    /// instead of escaping the frame. Without this, a
    /// residual resumes at the NEXT opcode, whose own catch receives only a
    /// raise from that opcode, not from the call itself; a residual whose CALL
    /// pc sits directly under a try (its fallthrough may leave the covered
    /// region — e.g. a FOR_ITER-next fallthrough is the continue-arm body,
    /// reached only on a non-null item, which carries no catch for the call's
    /// own raise) needs its own catch to receive the raise. Uncovered
    /// residuals fall back to the fallthrough resume even when this is set.
    pub residual_call_catch_resume: bool,

    /// Bridge-entry flavor guard (`_prepare_exception_resumption`,
    /// pyjitpl.py): the walk-entry `position` IS the resume
    /// coordinate — the failing source guard's own carried word, already a
    /// decodable `-live-` startpoint the runtime just resolved.  Carry it
    /// verbatim and key the resume py off its forward twin.  The default
    /// twin lookups compensate op-START keys (the after-residual advance,
    /// the block-head re-key) and would move an already-advanced coordinate
    /// a second time — on a call inside a try-block that lands on the
    /// physically-following `except` handler block, so the entry guard's
    /// own bridge resumes INSIDE the handler.  None outside the bridge-entry
    /// captures.
    pub carried_resume_jit_pc: Option<usize>,

    /// The branch guard's own jitcode `op.pc` for a kept-stack branch guard
    /// (#124). The snapshot helper is invoked with the *resume* coordinate
    /// (`other_target`, the not-taken arm), so the guard's own coordinate is
    /// otherwise unavailable to the encoder. `collect_outer_active_boxes`
    /// reads this to recover the kept operand-stack value(s): a branch guard's
    /// not-taken arm resumes at a merge point whose live Ref color was minted
    /// by the not-taken edge's register move and is therefore unwritten in the
    /// walk's register file at the guard point; the value lives instead in the
    /// guard-pc-only color the edge move reads from. None outside the gated
    /// kept-stack path.
    pub branch_guard_jitcode_pc: Option<usize>,

    /// `#420` not-taken edge-move recovery: the decoded `ref_copy`
    /// parallel-move list of a kept-stack branch guard's trampoline, as
    /// `(resume-merge color, guard-state kept value)` pairs. The branch handler
    /// reads each `ref_copy(dst <- src)` of the not-taken edge and supplies
    /// `(dst, registers_r[src])`; `collect_outer_active_boxes` and the
    /// `stack_sync` vable overlay read it so every kept operand-stack slot
    /// resumes with the exact value the edge move would produce, instead of the
    /// stale merge-color read. This replaces the positional depth-1 heuristic
    /// with the exact, depth-independent edge resolution. Empty outside the
    /// gated kept-stack path.
    pub branch_guard_kept_recovered: &'a [(u16, OpRef)],

    /// Select which already-recorded guard op receives the resume position.
    /// Needed whenever the guard being captured is not the last thing recorded:
    /// `_nonstandard_virtualizable` (pyjitpl.py) emits its promote
    /// `GUARD_VALUE` and then `emit_force_virtualizable` records GETFIELD_GC /
    /// PTR_NE / COND_CALL on top of it, so the default last-op stamp lands on
    /// the COND_CALL and leaves the guard holding the recorder's placeholder
    /// resume position.
    pub guard_stamp: GuardStampTarget,
}

/// Which already-recorded guard op a capture stamps its resume position on.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) enum GuardStampTarget {
    /// The last recorded op — the guard IS that op.
    #[default]
    LastOp,
}

/// Fallback for `rlib/jit.py` `max_unroll_recursion` when a skeleton walk has
/// no live driver/warmstate to consult. Production must read the parameter
/// from warmstate: `pypyjit.set_param(max_unroll_recursion=...)` updates it at
/// runtime, and `_opimpl_recursive_call` (`pyjitpl.py`) compares against that
/// live memory-manager value rather than the default.
const FBW_DEFAULT_MAX_INLINE_RECURSION: usize = 7;

/// Upper bound on the parameter count the self-recursive `CALL_ASSEMBLER` fold
/// accepts. Accumulator/linear recursion is low-arity; the cap keeps a
/// pathological signature off the fold path.
const FBW_REC_CA_MAX_PARAMS: usize = 8;

/// One paused caller frame the multi-frame inline snapshot must carry
/// (#68/#124).  Computed at the inline CALL site — where the caller's live
/// register banks are still in scope — and read back at guard-capture time
/// (where the walk context is the callee's, so the caller banks are gone).
#[derive(Clone)]
struct InlineParentFrame {
    /// The caller's jitcode index (`(*JitCode).index`).
    jitcode_index: u32,
    /// The caller CALL opcode's JitCode pc. A nested inline-capture abort
    /// carries this native coordinate to the interpreter-flush boundary.
    call_jitcode_pc: Option<usize>,
    /// Concrete operand-stack slots at the caller CALL, keyed by absolute
    /// `locals_cells_stack_w` index. Used only by the abort-point flush.
    call_stack_overrides: Vec<(usize, pyre_object::PyObjectRef)>,
    /// Complete concrete image needed to turn this paused caller into an
    /// `MIFrame` if a force aborts in a descendant residual call.  Captured
    /// while the caller's register banks are still live; `None` is a
    /// best-effort decline that leaves the existing guard snapshot usable.
    blackhole: Option<InlineParentBlackhole>,
    /// Authoritative derivation of the caller's Python resume pc.  The
    /// Python-native consumers resolve this at their boundary; a CALL inside
    /// a try-block (catch marker) still declines multi-frame.
    resume_coord: ParentResumeCoord,
    /// Codewrite-time jitcode-keyed marker for the CALL fallthrough resume
    /// point, queried at the CALL site.  Bridge-root frames have no CALL
    /// offset in hand and carry `None`.
    resume_marker_jit_pc: Option<usize>,
    /// The caller's `in_a_call` active boxes at its resolved resume pc — the liveness
    /// at the return point with the not-yet-produced call-result slot nulled
    /// (`get_list_of_active_boxes(in_a_call=true)` parity, trace_opcode.rs).
    boxes: Vec<OpRef>,
    /// Live `MIFrame.registers_{r,i,f}` of the paused caller, plus the
    /// walker extras on `WalkFrameState`. `MetaInterp.replace_box` writes
    /// these in place (`pyjitpl.py replace_box` → `replace_active_box_in_frame`).
    /// `None` on a reconstructed resume image that has no live walk
    /// (bridge reconstruct, ctor continuation).
    registers_r: Option<RegisterBank>,
    registers_i: Option<RegisterBank>,
    registers_f: Option<RegisterBank>,
    frame_state: Option<WalkFrameState>,
    /// CALL in this paused caller that entered the child. `f_lasti` /
    /// `f_lineno` for a getframe landing on this caller. `None` when the
    /// parent is a reconstructed image with no live CALL.
    caller_py_pc: Option<u32>,
}

impl InlineParentFrame {
    fn attach_live_caller(
        mut self,
        registers_r: &RegisterBank,
        registers_i: &RegisterBank,
        registers_f: &RegisterBank,
        frame_state: &WalkFrameState,
    ) -> Self {
        self.registers_r = Some(registers_r.clone());
        self.registers_i = Some(registers_i.clone());
        self.registers_f = Some(registers_f.clone());
        self.frame_state = Some(frame_state.clone());
        self
    }

    fn with_caller_py_pc(mut self, jitcode_index: u32, call_jit_pc: usize) -> Self {
        self.caller_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc_public(
            jitcode_index as i32,
            call_jit_pc as i32,
        )
        .map(|py| py as u32);
        self
    }

    fn paused_concrete_frame(&self) -> Option<usize> {
        let state = self.frame_state.as_ref()?;
        let borrowed = state.borrow();
        let ptr = borrowed.callee_shadow.as_ref()?.concrete_frame;
        (ptr != 0).then_some(ptr)
    }
}

#[derive(Clone)]
struct InlineParentBlackhole {
    /// Position immediately after the CALL.  The callee blackhole writes its
    /// return value through `call_result_reg()` before this caller runs.
    resume_pc: usize,
    int_values: Vec<(usize, i64)>,
    /// `(register color, value)`, read out of the color-indexed
    /// `concrete_registers_r` shadow and restored into the MIFrame's
    /// color-indexed Ref bank.
    ref_values: Vec<(usize, pyre_object::PyObjectRef)>,
    /// Float values have no concrete shadow bank; retain their OpRefs and
    /// resolve them at force time while the trace context is still live.
    float_values: Vec<(usize, OpRef)>,
}

/// The derivation flavor of a paused caller frame's Python resume pc.
#[derive(Clone, Copy)]
enum ParentResumeCoord {
    /// `resume_py_pc = trivia_normalized_py_pc_for_jitcode_pc(jitcode_index, jitcode_pc)`. Used by
    /// both bridge-root and reconstructed-recipe parent frames.
    Backxlat(usize),
    /// `resume_py_pc = semantic_fallthrough_pc(code,
    /// containing_py_pc_for_jitcode_pc(metadata, call_jitcode_pc))`.
    CallFallthrough(usize),
}

/// `typeobject.py descr_call` paused between the caller's CALL and `__init__`.
///
/// Upstream this level costs nothing to build — `descr_call` is an ordinary
/// graph, so `perform_call` already pushed its `MIFrame` and
/// `capture_resumedata` hands it over with the rest of the framestack. Pyre's
/// walker synthesises the type call instead of entering that graph
/// (`try_walker_inline_type_call`), so the level has to be spelled out; the
/// jitcode it names is `crate::ctor_continuation`'s.
///
/// `boxes` is the instance alone: that is the tail's whole live state, and its
/// `ref_return` is what fills the caller's pending call-result slot on the way
/// out. `blackhole` is `None` because the level owns no register banks to
/// force, and `call_stack_overrides` is empty because it has no operand stack.
pub(crate) fn ctor_continuation_parent_frame(instance: OpRef) -> Option<InlineParentFrame> {
    let jitcode_index = crate::ctor_continuation::jitcode_index()?;
    let resume_pc = crate::ctor_continuation::resume_pc()?;
    Some(InlineParentFrame {
        jitcode_index: jitcode_index as u32,
        call_jitcode_pc: None,
        call_stack_overrides: Vec::new(),
        blackhole: None,
        // The tail has no Python code object and so no Python pc. `Backxlat`
        // over a codeless jitcode is what already answers `u32::MAX` for such
        // a frame, which is the existing spelling for "no Python coordinate".
        resume_coord: ParentResumeCoord::Backxlat(resume_pc),
        resume_marker_jit_pc: Some(resume_pc),
        boxes: vec![instance],
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
    })
}

/// An operator's own tail, paused between the caller's opcode and the
/// dunder inlined under it.
///
/// [`ctor_continuation_parent_frame`]'s sibling for every operator whose
/// result is not the callee's return value; the jitcode it names is
/// `crate::operator_continuation`'s, and that module's doc carries why the
/// level has to exist at all.
///
/// `boxes` is empty because the level's whole state is the pending
/// call-result slot the dunder is about to fill, and a resume section never
/// carries one (`get_list_of_active_boxes(in_a_call=True)`). `blackhole` is
/// `None` because the level owns no register banks to force, and
/// `call_stack_overrides` is empty because it has no operand stack.
pub(crate) fn operator_continuation_parent_frame(
    tail: crate::operator_continuation::OperatorTail,
) -> Option<InlineParentFrame> {
    let jitcode_index = crate::operator_continuation::jitcode_index(tail)?;
    let resume_pc = crate::operator_continuation::resume_pc(tail)?;
    Some(InlineParentFrame {
        jitcode_index: jitcode_index as u32,
        call_jitcode_pc: None,
        call_stack_overrides: Vec::new(),
        blackhole: None,
        // The tail has no Python code object and so no Python pc. `Backxlat`
        // over a codeless jitcode is what already answers `u32::MAX` for such
        // a frame, which is the existing spelling for "no Python coordinate".
        resume_coord: ParentResumeCoord::Backxlat(resume_pc),
        resume_marker_jit_pc: Some(resume_pc),
        boxes: Vec::new(),
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
    })
}

/// RAII guard for one framestack level. Pop on drop so `?` and nested
/// sub-walks unwind to the caller's level.
struct InlineFrameGuard<'a>(&'a std::cell::RefCell<WalkSession>);

/// The translated stack root of this attempt's MetaInterp/framestack
/// (warmstate.py WarmEnterState.make_entry_point.bound_reached). Borrow the
/// stable RefCell for the whole attempt, including empty-stack intervals and
/// post-walk snapshots. Frame guards may be stored in a Vec and retired in
/// insertion order; none of them owns this publication's lifetime.
pub(crate) struct WalkSessionRoots<'a> {
    _area: majit_gc::shadow_stack::MutatorExtraAreaGuard,
    _session: &'a std::cell::RefCell<WalkSession>,
}

impl<'a> WalkSessionRoots<'a> {
    pub(crate) fn new(session: &'a std::cell::RefCell<WalkSession>) -> Self {
        let area = unsafe {
            // The returned guard's borrow prevents moving/dropping the cell
            // before retirement. Collectors visit it only under mutator STW
            // or synchronously on its owner; no session borrow spans a
            // collecting call. The checked callback enforces that last rule.
            majit_gc::shadow_stack::MutatorExtraAreaGuard::new(
                walk_session_roots,
                (session as *const std::cell::RefCell<WalkSession>).cast(),
                "walk_session",
            )
        };
        Self {
            _area: area,
            _session: session,
        }
    }
}

unsafe fn walk_session_roots(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let cell = unsafe { &*(data as *const std::cell::RefCell<WalkSession>) };
    let mut session = cell
        .try_borrow_mut()
        .expect("WalkSession borrow held across collection");
    let walk_ptr = |ptr: &mut pyre_object::PyObjectRef,
                    visitor: &mut dyn FnMut(&mut majit_ir::GcRef)| {
        let mut root = majit_ir::GcRef(*ptr as usize);
        visitor(&mut root);
        *ptr = root.0 as pyre_object::PyObjectRef;
    };
    // MetaInterp.last_exc_value and blackhole.py tmpreg_r are owner fields,
    // not merely roots when an inline frame happens to be on the stack.
    if let Some(value) = session.last_exc_value.as_mut() {
        value.walk_const_ptr_refs_mut(visitor);
    }
    if let ConcreteValue::Ref(value) = &mut session.last_exc_value_concrete {
        walk_ptr(value, visitor);
        unsafe { pyre_interpreter::eval::walk_raw_exception_roots(*value, visitor) };
    }
    if let Some(seed) = session.carrier_raise_seed.as_mut() {
        seed.exc.walk_const_ptr_refs_mut(visitor);
        if let ConcreteValue::Ref(value) = &mut seed.exc_concrete {
            walk_ptr(value, visitor);
            unsafe { pyre_interpreter::eval::walk_raw_exception_roots(*value, visitor) };
        }
    }
    session.tmpreg_r.walk_const_ptr_refs_mut(visitor);
    if let Some((value, _)) = session.finish_payload.as_mut() {
        value.walk_const_ptr_refs_mut(visitor);
    }
    if let ConcreteValue::Ref(value) = &mut session.tmpreg_r_concrete {
        walk_ptr(value, visitor);
    }
    for frame in session.framestack.iter_mut() {
        for parent in frame.parents.iter_mut() {
            // history.py ConstPtr.value remains reachable through each
            // paused frame's active boxes, including constructor tails.
            for value in &mut parent.boxes {
                value.walk_const_ptr_refs_mut(visitor);
            }
            for (_, value) in &mut parent.call_stack_overrides {
                walk_ptr(value, visitor);
            }
            if let Some(blackhole) = parent.blackhole.as_mut() {
                for (_, value) in &mut blackhole.ref_values {
                    walk_ptr(value, visitor);
                }
            }
        }
    }
}

impl<'a> InlineFrameGuard<'a> {
    fn enter(
        session: &'a std::cell::RefCell<WalkSession>,
        w_code: usize,
        recursion_greenkey: bool,
        parents: Vec<InlineParentFrame>,
    ) -> Self {
        let mut walk = session.borrow_mut();
        let call_id = if w_code == 0 {
            walk.framestack
                .iter()
                .rev()
                .find(|frame| frame.w_code != 0)
                .map(|frame| frame.call_id)
                .unwrap_or(0)
        } else {
            let call_id = walk.next_call_id;
            walk.next_call_id = walk.next_call_id.saturating_add(1);
            call_id
        };
        walk.framestack.push(InlineFrame {
            is_portal: false,
            w_code,
            recursion_greenkey,
            call_id,
            debug_merge_point_py_pc: None,
            parents,
            entry_executed_effects: fbw_executed_effect_count(),
            live: None,
        });
        drop(walk);
        if fbw_depth_census_enabled() {
            fbw_depth_census_record(&session.borrow().framestack);
        }
        InlineFrameGuard(session)
    }
}

impl Drop for InlineFrameGuard<'_> {
    fn drop(&mut self) {
        let mut session = self.0.borrow_mut();
        session.framestack.pop();
    }
}

/// Terminal disposition of a walk kept for the no-replay exit:
/// the top-level frame's concrete return value, or the concrete
/// exception object of the uncaught raise that ended the walk
/// (`opimpl_raise` → `finishframe_exception` →
/// `compile_exit_frame_with_exception`, pyjitpl.py +
/// 3238-3242; the caller consumes it as `ExitFrameWithExceptionRef`,
/// jitexc.py).
#[derive(Clone, Copy)]
pub enum FinishConcrete {
    Return(ConcreteValue),
    Raise(ConcreteValue),
}

/// # Safety
/// `data` must come from [`capture_fbw_finish_concrete_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn fbw_finish_concrete_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let cell = unsafe { &*(data as *const std::cell::Cell<Option<FinishConcrete>>) };
    let Some(finish) = cell.get() else {
        return;
    };
    let (value, is_raise) = match finish {
        FinishConcrete::Return(value) => (value, false),
        FinishConcrete::Raise(value) => (value, true),
    };
    if let ConcreteValue::Ref(ptr) = value {
        let mut gcref = majit_ir::GcRef(ptr as usize);
        visitor(&mut gcref);
        // The visitor may forward (relocate) the ref; write it back so
        // the stashed value points at the moved object.
        let value = ConcreteValue::Ref(gcref.0 as pyre_object::PyObjectRef);
        cell.set(Some(if is_raise {
            FinishConcrete::Raise(value)
        } else {
            FinishConcrete::Return(value)
        }));
    }
}

/// One in-flight FOR_ITER continuation entry (#57 Option C): the item the
/// FOR_ITER `for_iter_next` residual consumed on the authoritative walk, the
/// body coordinate to resume at (the FOR_ITER `py_pc + 1` continue-arm
/// fallthrough, resolved only at a match point),
/// and whether a body effect committed since THIS consume.
///
/// `body_effect_since_consume` is the R1 double-apply guard, per entry: an
/// irreversible heap mutation that succeeded after this consume (a body the
/// store/append journals do NOT cover).  Re-running this iteration's body on
/// delivery would re-apply it, so [`fbw_foriter_inflight_take`] refuses
/// delivery when set.  A mutation committed while several FOR_ITER items are
/// in flight is "after" every one of them (re-running ANY of their bodies
/// re-applies it), so the executor marks the flag on EVERY active entry; a
/// fresh consume's own entry starts clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InflightForiterBody {
    /// Legacy entry-PC fallback for a per-opcode walk or fixture with no
    /// full-body JitCode coordinate.
    Py(usize),
    /// The `for_iter_next` residual's own JitCode pc plus the identity of the
    /// JitCode that pc indexes — the inlined callee's inside a sub-walk, the
    /// outer portal's otherwise.  The body fallthrough is derived only at a
    /// match point.  Negative when the identity is unresolvable, which
    /// `inflight_foriter_body_pc` reports as no coordinate at all.
    Jit { jitcode_index: i32, op_pc: usize },
}

#[derive(Clone, Copy)]
struct InflightForiter {
    item: pyre_object::PyObjectRef,
    body: InflightForiterBody,
    body_effect_since_consume: bool,
    /// [`fbw_bridge_iter_journal_capture`] pushed a cursor for this consume.
    /// False for a generator, `map`, dict/set iterator, itertools, or user
    /// `__next__`: entry replay would call `__next__` again.
    consume_journaled: bool,
    /// The walk re-reached this FOR_ITER's consume after the item's body ran
    /// (a NEW `for_iter_next` attempt was dispatched for the same body).
    /// A completed entry must never be re-delivered — its body already ran
    /// during the walk — but a flush may still adopt the walk end state at
    /// the header WITHOUT delivery (the interpreter re-attempts the consume).
    body_completed: bool,
}

/// Which `(seq, index)` pair a [`BridgeIterJournalEntry::Cursor`] restores.
#[derive(Clone, Copy)]
enum BridgeIterKind {
    List,
    ListReverse,
    Tuple,
}

/// Concrete iterator state to restore when a bridge walk is abandoned.
#[derive(Clone, Copy)]
enum BridgeIterJournalEntry {
    Range {
        iter: pyre_object::PyObjectRef,
        pre_current: i64,
        pre_remaining: i64,
    },
    /// The three iterators whose payload is a source reference plus a cursor,
    /// and whose successful advance moves only the cursor.
    Cursor {
        kind: BridgeIterKind,
        iter: pyre_object::PyObjectRef,
        pre_seq: pyre_object::PyObjectRef,
        pre_index: isize,
    },
}

thread_local! {
    /// Undo log for the walked region's eagerly executed list stores:
    /// `(list, key, displaced_value)` triples pushed by the `STORE_SUBSCR`
    /// specializations before they mutate the list.  Upstream executes
    /// every traced operation concretely (pyjitpl.py
    /// execute_and_record) and never re-runs the traced region, so the
    /// walker applies the store at trace time too.  Pyre's transitional
    /// non-commit paths instead RE-RUN the region (the legacy
    /// replay-from-snapshot), which would re-apply the store against the
    /// already-mutated heap (a swap re-reads its own output and swaps
    /// back) — so a walk that does not commit its end state
    /// (`flush_walk_end_state_to_frame`) rolls these entries back in
    /// reverse order, restoring the pre-walk heap the replay expects.  A
    /// committing walk drops the log (the mutation is already applied,
    /// exactly once).  Dies with the replay paths.  Entries are GC roots
    /// via [`fbw_store_journal_root_walker`].
    static FBW_STORE_JOURNAL: std::cell::RefCell<Vec<[pyre_object::PyObjectRef; 3]>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Ordered undo log for eagerly executed list effects. Same transitional
    /// rationale as [`FBW_STORE_JOURNAL`]: a committing walk drops the log;
    /// a non-commit walk replays it in reverse push order so interleaved
    /// appends and end-pops restore the exact pre-walk list state. Dies with
    /// the legacy replay paths. Entries' list refs are GC roots via
    /// [`fbw_store_journal_root_walker`]. `PopEnd::w_item` is boxed so a
    /// strategy change can restore it into Object storage, and is visited
    /// because the journal holds it live across safepoints.
    static FBW_LIST_EFFECT_JOURNAL: std::cell::RefCell<Vec<FbwListEffect>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for Empty-list first-append promotion during pyre's
    /// speculative full-body walk.  Entries ride the same commit/rollback
    /// lifecycle as [`FBW_LIST_EFFECT_JOURNAL`]: a committed walk keeps the typed
    /// strategy, while a replayed walk restores the list to Empty so replay
    /// can execute the append from the original shape.  This exists only for
    /// pyre's speculative-replay walk and can be removed when
    /// single-executor tracing lands (gh#73/#34).  Entries are GC roots via
    /// [`fbw_store_journal_root_walker`].
    static FBW_APPEND_PROMOTE_JOURNAL: std::cell::RefCell<Vec<pyre_object::PyObjectRef>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for the walked region's eagerly executed module-global
    /// `IntMutableCell` stores: `(cell, intvalue_before)` pairs pushed by the
    /// StoreName/StoreGlobal cell fold before it writes `cell.intvalue` in
    /// place.  Same rationale as [`FBW_STORE_JOURNAL`] — the walker is the
    /// authoritative executor, so a folded store must apply its concrete
    /// effect at walk time (the residual it replaces would have run
    /// `write_cell` via `try_execute_residual_call_via_executor`); a
    /// non-commit walk restores each cell's prior `intvalue` in reverse push
    /// order so the legacy replay re-applies the store against the pre-walk
    /// heap.  The cell is a nursery object (`ObjectMutableCell.__init__`),
    /// so each entry is a GC root via [`fbw_store_journal_root_walker`].
    static FBW_CELL_STORE_JOURNAL: std::cell::RefCell<Vec<FbwCellStore>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for the walked region's eagerly executed namespace bindings.
    /// `STORE_NAME` / `STORE_GLOBAL` / `DELETE_NAME` / `DELETE_GLOBAL` reach the
    /// walker as residuals that write the frame's namespace dict in place, and
    /// no other journal records what they displaced.  A walk that does not
    /// commit hands its region to a replay that re-executes it FROM THE WALK
    /// ENTRY, so every statement between the entry and the store reads the
    /// walk's binding while the rest of the frame still holds its pre-walk
    /// value: a bridge walk that runs off the tail of one `for` iteration into
    /// the head of the next leaves the loop variable — and everything derived
    /// from it — advanced, and the replayed tail reports the previous
    /// iteration's data against the next iteration's counters.  Entries restore
    /// the displaced value (or unbind a name the walk created) in reverse push
    /// order, the way [`FBW_STORE_JOURNAL`] restores list elements.  The
    /// displaced value can have lost its only other reference to the store, so
    /// entries are GC roots via [`fbw_store_journal_root_walker`].
    static FBW_NAMESPACE_STORE_JOURNAL: std::cell::RefCell<Vec<FbwNamespaceStore>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Set when the rollback above undid a binding that is NOT the in-flight
    /// FOR_ITER's loop-variable store.  An in-flight delivery repositions the
    /// frame to the FOR_ITER body instead of replaying from the walk entry, so
    /// a binding written before that FOR_ITER would never be re-applied; the
    /// delivery refuses while this stands and keeps the conservative drop.
    static FBW_NAMESPACE_STORE_ROLLED_BACK: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    /// Undo log for the walked region's eagerly executed
    /// `set_current_exception` stores against the LIVE per-thread
    /// `ExecutionContext.sys_exc_value`.  The authoritative walk lowers
    /// PUSH_EXC_INFO / POP_EXCEPT to `set_current_exception` residuals and
    /// applies their concrete effect at walk time
    /// ([`try_walker_lower_exc_info_residual`]) so a following
    /// `get_current_exception` reads the right value.  Same rationale as
    /// [`FBW_STORE_JOURNAL`]: on a non-commit exit the legacy replay re-runs
    /// the walked region and must find the pre-walk `sys_exc_value`.  Without
    /// this journal an exception that propagates OUT of an except-handler
    /// (the handler body itself raises) aborts the walk BEFORE its paired
    /// POP_EXCEPT restore runs, leaving the live EC holding the caught
    /// exception; the next frame reads it as the active exception and chains
    /// it as a spurious `__context__`.  Each entry is the displaced prior
    /// value (`get_current_exception()` read just before the eager store);
    /// the rollback re-applies them in reverse push order so the final write
    /// restores the walk-entry value.  Entries are GC roots via
    /// [`fbw_store_journal_root_walker`].
    static FBW_SYS_EXC_JOURNAL: std::cell::RefCell<Vec<pyre_object::PyObjectRef>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo entries for the concrete traceback-head stores performed while
    /// bridge-entry exception arms are recorded: `(exception, attached_node)`.
    /// The two arms are mutually exclusive per invocation, but a carrier
    /// sub-walk that continues its root under [`WalkJournals::Keep`] runs both
    /// inside one journal lifecycle (the callee handler entry, then the root's
    /// carrier-raise seed), so the log holds every attach.  It shares the store
    /// journal's lifecycle: a committing walk keeps the nodes and clears the
    /// log, while a discarded walk splices each node back out in reverse push
    /// order.  Both refs of every entry are GC roots via
    /// [`fbw_store_journal_root_walker`].
    ///
    /// TLS owner: the log belongs to the in-progress walk, and a walk runs to
    /// its commit-or-discard point on the thread that started it — the same
    /// ownership the sibling `FBW_*_JOURNAL` logs above use.  The entries never
    /// outlive that walk and are never read from another thread; GC
    /// reachability is supplied explicitly by the root walker rather than by
    /// the storage.
    static FBW_TRACEBACK_STORE_JOURNAL:
        std::cell::RefCell<Vec<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// In-flight FOR_ITER continuation (#57 Option C): `(consumed_item,
    /// body coordinate)` stashed when the FOR_ITER `for_iter_next` residual
    /// ([`RuntimeHelperKind::ForIterNext`]) runs concretely on the
    /// authoritative walk and advances the real shared heap iterator.  The
    /// advance is an irreversible side effect with no journal undo
    /// (`functional.rs` `current += step`); on a successful CloseLoop commit
    /// the walk-end flush adopts the advanced state, so the stash is dropped
    /// ([`fbw_store_journal_commit`]).  On a trace ABORT the walk discards
    /// its recording but the iterator stays advanced — so the stash is
    /// DELIVERED to the live frame (the consumed item pushed, the frame
    /// repositioned at its resolved body pc) instead of dropping the iteration, the
    /// `_copy_data_from_miframe` continue-forward analog (blackhole.py).
    /// The body coordinate represents the FOR_ITER continue-arm fallthrough
    /// (`py_pc + 1`, codewriter.rs continue arm). The item ref is a GC root via
    /// [`fbw_store_journal_root_walker`].  Cleared at walk start
    /// ([`fbw_store_journal_reset`]).
    ///
    /// A LIFO stack, not a single slot: a walk that descends into a NESTED
    /// FOR_ITER has BOTH the outer loop's consumed item and the inner loop's
    /// consumed item in flight at once.  Each [`InflightForiter`] is keyed by
    /// its exact JitCode + native opcode coordinate (or by the legacy Python
    /// pc when no JitCode exists), so equally numbered caller/callee FOR_ITERs
    /// remain distinct. A re-consume of the SAME FOR_ITER (the prior
    /// iteration's body completed) replaces that loop's entry while a consume
    /// of a DIFFERENT (nested) FOR_ITER pushes a new entry — the outer entry is
    /// no longer destroyed by the inner consume.  The abort delivery
    /// ([`fbw_foriter_inflight_take`]) still consumes only the most-recent
    /// (top) entry, matching the single-slot behaviour; preserving the
    /// remaining entries is the representation S2 needs to deliver each at its
    /// true frame slot.
    static FBW_FORITER_INFLIGHT: std::cell::RefCell<Vec<InflightForiter>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Undo log for a bridge/retrace recording walk's eager iterator cursor
    /// mutation.  The main walk leaves the mutation unjournaled and relies on
    /// in-flight FOR_ITER forward-delivery to recover the consumed item on
    /// abort; the bridge/retrace abort path has no such delivery, so a bridge
    /// walk restores each entry when it does NOT commit — leaving the recording
    /// side-effect neutral so interpreter resume re-consumes the item exactly
    /// once.
    /// Only populated while `is_bridge_trace`; empty (no-op) on the main walk.
    static FBW_BRIDGE_ITER_JOURNAL: std::cell::RefCell<Vec<BridgeIterJournalEntry>> =
        const { std::cell::RefCell::new(Vec::new()) };

    static FBW_UNJOURNALED_VALUE_UNAVAILABLE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    static FBW_UNJOURNALED_SYMBOLIC: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    static FBW_EXECUTED_RESIDUAL_VOID: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FBW_EXECUTED_RESIDUAL_MAYFORCE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FBW_EXECUTED_RESIDUAL_PLAIN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };

    /// gh#467 executed-effect odometer: bumped only when the walk concretely
    /// applies an effect — a store/append/cell journal push, a Void / mutator-
    /// tagged residual executed by [`try_execute_residual_call_via_executor`],
    /// or a residual whose concrete run entered a user Python frame (a value-
    /// returning dunder that may mutate).  A declined residual recorded only
    /// symbolically sets an unjournaled flag ([`FBW_UNJOURNALED_VALUE_UNAVAILABLE`]
    /// / [`FBW_UNJOURNALED_SYMBOLIC`]) but does NOT move this
    /// counter: no effect executed.  The inline abort-forward-flush gate
    /// (`try_walker_inline_user_call` / gh#467) snapshots both signals at the
    /// CALL.  A nonzero count delta means the callee attempt cannot be
    /// discarded and re-executed without risking a double.
    static FBW_EXECUTED_EFFECT_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };

    /// `ABORT_FORCE_QUASIIMMUT` resume carrier: `(py_pc, operand-stack OpRef
    /// mirror)` taken at the abort, for the flush leg that resumes the
    /// interpreter at that opcode.
    ///
    /// The mirror is needed because the abort lands MID-EXPRESSION: the vable
    /// shadow's stack region reads NULL outside a merge point, so the plain
    /// flush declines on the first operand slot and the walk falls back to the
    /// replay this leg exists to avoid.  Holds symbolic `OpRef`s only — the
    /// concrete refs are resolved and rooted inside the flush.
    static FBW_QMUT_ABORT_STACK: std::cell::RefCell<Option<(usize, Vec<OpRef>)>> =
        const { std::cell::RefCell::new(None) };

    /// Kept-stack branch-abort resume carrier: `(py_pc, complete operand-stack
    /// OpRef mirror)` captured from the live MIFrame-equivalent walk context
    /// before it is dropped.
    ///
    /// Like the quasiimmutable carrier above, this is needed only for a
    /// mid-expression abort: the virtualizable array is authoritative at merge
    /// points, while RPython resumes a blackhole from the MIFrame register
    /// image at the exact abort point (`blackhole.py _copy_data_from_miframe`).  The trace
    /// epilogue accepts this carrier only when its Python coordinate exactly
    /// matches the decoded abort resume pc.
    static FBW_BRANCH_ABORT_STACK: std::cell::RefCell<Option<(usize, Vec<OpRef>)>> =
        const { std::cell::RefCell::new(None) };

    /// [`FBW_EXECUTED_EFFECT_COUNT`] sampled as the walk ENTERED the Python
    /// opcode it is currently inside, as `(py_pc, odometer)`.
    ///
    /// A leg that resumes the interpreter at a Python opcode re-executes that
    /// whole opcode — `flush_walk_end_state_to_frame` sets `last_instr = pc - 1`
    /// — so it owes a proof that the opcode has applied nothing yet.  This is
    /// that proof's sample point, and it must be the OPCODE boundary: a
    /// per-`-live-` or per-jitcode-op sample would read zero for an opcode whose
    /// earlier ops already ran effects.
    ///
    /// Recorded only on the top-level walk (`reconcile_vstack_at_boundary`, the
    /// boundary detector).  A callee sub-walk's `py_pc` names a different code
    /// object, so leaving the sample absent there makes the gate decline rather
    /// than compare coordinates from two universes.
    static FBW_OPCODE_ENTRY_EFFECTS: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };

    /// Effect-count delta of the exact JitCode opcode that most recently
    /// returned `LoopBearingCalleeInlineUnsupported`.  The structural
    /// mid-body carrier accepts only a zero delta: pyre re-runs the enclosing
    /// Python opcode boundary, unlike byte-exact `blackhole_from_resumedata`
    /// (`pyjitpl.py`), so any already-applied effect in that opcode must
    /// keep the legacy path.
    static FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };

    /// Inline-abort MidBody carrier: latched by
    /// [`try_walker_inline_user_call`] when a supported abort fires inside a
    /// TOP-level inline sub-walk.  Holds the rebuilt-callee image so the walk
    /// driver can resume inside the callee; the live framestack converts
    /// through `convert_and_run_from_pyjitpl` (`blackhole.py`).  The
    /// `PyObjectRef`s are rooted by [`fbw_store_journal_root_walker`] across
    /// the abort unwind's allocations.
    static FBW_ABORT_CALL_RESUME: std::cell::RefCell<Option<InlineAbortCarrier>> =
        const { std::cell::RefCell::new(None) };

    /// The set of OpRefs the walker built inline via
    /// [`try_walker_trace_exception_new`] (the virtualizable `NewWithVtable`
    /// exception). The immediately-following `RaiseVarargs` residual consults
    /// it to take the instance fast path — skipping the
    /// residual `normalize_raise_varargs_jit` publish + `GUARD_EXCEPTION`
    /// and emitting `__context__` as a `SetfieldGc` on the (still virtual)
    /// exception.  Reset at each FBW walk entry via `fbw_store_journal_reset`
    /// so a stale OpRef key from a prior recorder cannot leak across walks.
    static FBW_BUILT_EXC: std::cell::RefCell<std::collections::HashSet<OpRef>> =
        std::cell::RefCell::new(std::collections::HashSet::new());

    /// The OpRefs whose `__context__` a raise lowering already emitted as
    /// `SetfieldGc(exc, GetfieldGcR(ec, sys_exc_value))`.  Distinct from
    /// [`FBW_BUILT_EXC`], which `RaiseVarargs` *consumes* — by the time the
    /// exception reaches the handler that catches it in this same trace, that
    /// set no longer names it, and the in-trace catch compensation would
    /// stamp the field a second time.  Reset with the other walk state.
    static FBW_CONTEXT_CHAINED: std::cell::RefCell<std::collections::HashSet<OpRef>> =
        std::cell::RefCell::new(std::collections::HashSet::new());

    /// LIFO stack of the previous-exception slot value
    /// saved by each lowered `PUSH_EXC_INFO` (`get_current_exception` arm),
    /// paired with its live concrete.  `POP_EXCEPT`'s restore consumes the top
    /// entry, so the slot is set back to the TRUE saved prev (None for an outer
    /// handler, the outer exception for a nested one) instead of the operand-
    /// stack value the codewriter threads into the `set_current_exception`
    /// residual — which the walker resolves to the just-caught exception, not
    /// the saved prev.  Restoring the saved prev is what lets the balanced
    /// save/restore cancel (the locally-caught exception de-escapes and DCEs)
    /// and keeps `sys.exc_info()` correct after the handler unwinds.  Reset per
    /// walk via `fbw_store_journal_reset`.
    static FBW_EXC_PREV: std::cell::RefCell<Vec<(OpRef, pyre_object::PyObjectRef)>> =
        std::cell::RefCell::new(Vec::new());

    /// Set when a `PUSH_EXC_INFO` prev save (`get_current_exception`) is
    /// lowered; the immediately-following `set_current_exception` is then that
    /// PUSH's slot store (it stores the EXC and leaves the saved prev on
    /// [`FBW_EXC_PREV`]) rather than a `POP_EXCEPT` restore (which pops it).
    /// The two `SetCurrentException` shapes are otherwise identical.
    static FBW_EXC_PENDING_PUSH_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

enum FbwListEffect {
    Append {
        list: pyre_object::PyObjectRef,
        length_before: usize,
        /// `PyListObject.allocated` as the specialization read it before the
        /// append.  The length rewind alone would leave the over-allocation
        /// the eager append computed, so the undo restores the field the same
        /// way it restores the length.
        allocated_before: isize,
    },
    PopEnd {
        list: pyre_object::PyObjectRef,
        length_before: usize,
        w_item: pyre_object::PyObjectRef,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum InlineAbortCarrier {
    MidBody(MidBodyPayload),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MidBodyAbortKind {
    Marker,
    Structural,
}

#[derive(Clone, Debug)]
pub(crate) struct MidBodyPayload {
    pub abort_kind: MidBodyAbortKind,
    /// The outer CALL's native coordinate, resolved at the interpreter flush
    /// boundary.
    pub outer_jitcode_index: u32,
    pub call_jitcode_pc: usize,
    pub call_stack_len: usize,
    /// The rebuilt callee's native abort coordinate, resolved at the
    /// interpreter flush boundary.
    pub callee_jitcode_index: u32,
    pub abort_jitcode_pc: usize,
    /// Forward-carried Python instruction PC for `abort_jitcode_pc`, stamped
    /// once at capture so the flush reads a scalar instead of re-inverting the
    /// coordinate through the jitcode->py tables.
    pub callee_py_pc: usize,
    pub w_code: pyre_object::PyObjectRef,
    pub w_globals: pyre_object::PyObjectRef,
    pub live_locals: Vec<Option<ConcreteValue>>,
    pub live_stack: Vec<ConcreteValue>,
    pub return_value: pyre_object::PyObjectRef,
    /// The instance `__new__` produced when the rebuilt callee is a
    /// constructor's `__init__`, `PY_NULL` otherwise.
    ///
    /// This leg resumes the callee directly on the caller and hands the caller
    /// whatever the callee returns.  For a constructor that is `__init__`'s
    /// `None`, and the caller's CALL result has to be the instance:
    /// `type_descr_call_impl` sits between the two frames and its tail is what
    /// discards `__init__`'s result and returns `w_newobject`.  The rebuild
    /// cannot reconstruct that native level, so the flush plays the tail
    /// itself off this field.  [`crate::ctor_continuation`] plays the same
    /// tail for the blackhole resume.
    pub constructor_instance: pyre_object::PyObjectRef,
}

/// Eager in-place cell write the StoreName/StoreGlobal fold applied.
/// `write_cell` mutates `IntMutableCell.intvalue` or
/// `ObjectMutableCell.w_value` without bumping `version?`; the walk
/// must restore the prior payload if it does not commit.
#[derive(Clone, Copy)]
enum FbwCellStore {
    Int {
        cell: pyre_object::PyObjectRef,
        before: i64,
    },
    Obj {
        cell: pyre_object::PyObjectRef,
        before: pyre_object::PyObjectRef,
    },
    /// A walked `setfield_gc` / `setarrayitem_gc` into an object the walk did
    /// not allocate, executed as it was recorded (`pyjitpl.py
    /// execute_and_record`).  `offset` / `size` locate the stored word inside
    /// `obj` — a field offset, or `base_size + index * item_size` — and
    /// `before` is what it held.  `managed` is whether the collector owns
    /// `obj`: only then is it forwarded as a root and re-barriered on undo.
    Gc {
        obj: pyre_object::PyObjectRef,
        offset: usize,
        size: usize,
        before: Value,
        managed: bool,
        /// `TracePosition::_count` once the store's op was recorded.  A cut
        /// back past it discards the op, and with it this store
        /// ([`fbw_gc_store_journal_cut`]).
        op_count: u32,
    },
}

/// One [`FBW_NAMESPACE_STORE_JOURNAL`] entry: the namespace dict a `*_NAME` /
/// `*_GLOBAL` residual wrote, the name it bound, and the value that name held
/// before the write — null when the name was unbound, which the rollback undoes
/// with a delete.  `loop_var` marks the FOR_ITER body's own loop-variable
/// store, the one binding an in-flight delivery re-executes for itself.
#[derive(Clone, Copy)]
struct FbwNamespaceStore {
    namespace: pyre_object::PyObjectRef,
    name: pyre_object::PyObjectRef,
    displaced: pyre_object::PyObjectRef,
    loop_var: bool,
    /// `Some` for a `STORE_SUBSCR` into a dict another thread can write.
    /// Rollback restores only while the entry is still this exact object.
    /// The root walker forwards it: a nursery value moves, and a stale
    /// pointer would look like a concurrent replacement.
    written: Option<pyre_object::PyObjectRef>,
}

struct FbwStoreJournalRootArea {
    stores: *const std::cell::RefCell<Vec<[pyre_object::PyObjectRef; 3]>>,
    list_effects: *const std::cell::RefCell<Vec<FbwListEffect>>,
    append_promote: *const std::cell::RefCell<Vec<pyre_object::PyObjectRef>>,
    cell_stores: *const std::cell::RefCell<Vec<FbwCellStore>>,
    namespace_stores: *const std::cell::RefCell<Vec<FbwNamespaceStore>>,
    sys_exc: *const std::cell::RefCell<Vec<pyre_object::PyObjectRef>>,
    traceback_store:
        *const std::cell::RefCell<Vec<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)>>,
    foriter: *const std::cell::RefCell<Vec<InflightForiter>>,
    bridge_iter: *const std::cell::RefCell<Vec<BridgeIterJournalEntry>>,
    abort_resume: *const std::cell::RefCell<Option<InlineAbortCarrier>>,
    escape_flush_undo: *const std::cell::RefCell<Option<EscapeFlushUndo>>,
    locals_mirror_undo: *const std::cell::RefCell<Vec<FbwLocalsMirrorUndo>>,
    single_frame_blackhole: *const std::cell::RefCell<Option<LatchedSingleFrameBlackhole>>,
    multi_frame_blackhole: *const std::cell::RefCell<Option<LatchedMultiFrameBlackhole>>,
}

thread_local! {
    static FBW_STORE_JOURNAL_ROOT_AREA: FbwStoreJournalRootArea = FbwStoreJournalRootArea {
        stores: FBW_STORE_JOURNAL.with(|value| value as *const _),
        list_effects: FBW_LIST_EFFECT_JOURNAL.with(|value| value as *const _),
        append_promote: FBW_APPEND_PROMOTE_JOURNAL.with(|value| value as *const _),
        cell_stores: FBW_CELL_STORE_JOURNAL.with(|value| value as *const _),
        namespace_stores: FBW_NAMESPACE_STORE_JOURNAL.with(|value| value as *const _),
        sys_exc: FBW_SYS_EXC_JOURNAL.with(|value| value as *const _),
        traceback_store: FBW_TRACEBACK_STORE_JOURNAL.with(|value| value as *const _),
        foriter: FBW_FORITER_INFLIGHT.with(|value| value as *const _),
        bridge_iter: FBW_BRIDGE_ITER_JOURNAL.with(|value| value as *const _),
        abort_resume: FBW_ABORT_CALL_RESUME.with(|value| value as *const _),
        escape_flush_undo: escape_flush_undo_cell_ptr(),
        locals_mirror_undo: locals_mirror_undo_cell_ptr(),
        single_frame_blackhole: single_frame_blackhole_cell_ptr(),
        multi_frame_blackhole: multi_frame_blackhole_cell_ptr(),
    };
}

/// Whether the opcode at `resume_py_pc` in the live frame's code is a
/// FOR_ITER — a non-FOR_ITER resume pc that merely happens to satisfy
/// `body_pc == some entry's resolved body pc` is Shape B.
fn foriter_header_at(frame: usize, resume_py_pc: usize) -> bool {
    let Some(raw_code) = frame_raw_code(frame) else {
        return false;
    };
    matches!(
        pyre_interpreter::decode_instruction_at(unsafe { &*raw_code }, resume_py_pc),
        Some((pyre_interpreter::Instruction::ForIter { .. }, _))
    )
}

/// Whether an in-flight FOR_ITER continuation belongs to `frame` at this
/// exact body coordinate.  PyPy's in-flight iterator state lives on the
/// corresponding MIFrame; matching only the Python pc collapses equally
/// numbered caller/callee FOR_ITERs onto one shared slot.
fn foriter_body_matches_frame(body: InflightForiterBody, frame: usize, body_pc: usize) -> bool {
    if inflight_foriter_body_pc(body) != Some(body_pc) {
        return false;
    }
    match body {
        // The legacy fallback has no JitCode identity to compare.  It is used
        // only when the walk itself cannot provide one, so retain its former
        // single-frame behaviour.
        InflightForiterBody::Py(_) => true,
        InflightForiterBody::Jit { jitcode_index, .. } => {
            let Some(frame_code) = frame_raw_code(frame) else {
                return false;
            };
            crate::state::raw_code_for_jitcode_index(jitcode_index)
                .is_some_and(|inflight_code| std::ptr::eq(inflight_code, frame_code))
        }
    }
}

fn frame_raw_code(frame: usize) -> Option<*const pyre_interpreter::CodeObject> {
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return None;
    }
    Some(unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    })
}

fn exact_floor_segment_anchor(
    metadata: &crate::PyJitCodeMetadata,
    py_pc: usize,
    jit_pc: usize,
) -> bool {
    crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
        .is_some_and(|(start, py)| start == jit_pc && py as usize == py_pc)
}

/// The jitcode ops the callee-rebuild leg would re-run if it resumed at the
/// Python pc owning `jit_pc` — everything from that pc's floor-segment start up
/// to `jit_pc`.  Empty exactly when [`exact_floor_segment_anchor`] holds, so a
/// non-empty list names what makes the anchor inexact.
pub(crate) fn floor_segment_ops_before(
    metadata: &crate::PyJitCodeMetadata,
    code: &[u8],
    jit_pc: usize,
) -> Vec<String> {
    let Some((mut pc, _py)) =
        crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
    else {
        return vec!["<no floor segment>".to_string()];
    };
    let mut keys = Vec::new();
    while pc < jit_pc {
        let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) else {
            keys.push("<undecodable>".to_string());
            break;
        };
        keys.push(format!(
            "{}@f{}",
            op.key,
            code.get(op.pc + 1).copied().unwrap_or(u8::MAX)
        ));
        pc = op.next_pc;
    }
    keys
}

/// Admit an abort pc at the semantic head of a portal-shaped Python opcode —
/// that is, one whose whole jitcode prefix inside its own floor segment writes
/// nothing but THIS frame's own virtualizable: operand-stack slots
/// (`setarrayitem_vable_r`) and the two int bookkeeping fields,
/// `last_instr`/`valuestackdepth` (`setfield_vable_i`).
///
/// The callee-rebuild leg reconstructs the frame wholesale — a fresh `PyFrame`
/// whose locals, stack area, `valuestackdepth` and `last_instr` all come from
/// the carried payload — so every one of those writes is erased before the
/// rebuilt frame runs, and the resume state at `py_pc` is unaffected by how far
/// into the opcode's spill the walk got.  `live_stack`'s depth is
/// `depth_at_py_pc[py_pc]` (`depth_for_jitcode_pc_pred` is keyed per Python pc),
/// i.e. the depth BEFORE the opcode, which is what resuming at `py_pc` needs.
///
/// `getarrayitem_vable_r` is NOT admitted even though it too only reads the
/// vable: it writes a jitcode REGISTER, and the payload sources `live_stack`
/// and `live_locals` out of `concrete_registers_r` by color.  A clobbered color
/// would be read back as a live value.
///
/// A registered trace-entry marker may sit inside the preceding floor segment:
/// `merge_entry_by_green` deliberately points at the block-head marker, while
/// `py_floor_by_jit_pc` continues to describe the containing opcode.  Admit
/// that seam only for an exact `(py_pc, jit_pc)` trace entry and only when the
/// whole prefix still passes the own-frame bookkeeping whitelist below.  The
/// rebuilt frame discards those stores just like same-floor spills.
///
/// Anything else, or a write aimed at another frame, declines.
fn portal_vable_bookkeeping_anchor(
    metadata: &crate::PyJitCodeMetadata,
    built_as_portal: bool,
    portal_frame_reg: u16,
    perfn_descrs: &[majit_metainterp::jitcode::RuntimeBhDescr],
    code: &[u8],
    py_pc: usize,
    jit_pc: usize,
    mut python_pc_for_op: impl FnMut(usize) -> usize,
) -> bool {
    if exact_floor_segment_anchor(metadata, py_pc, jit_pc) {
        return true;
    }
    let is_trace_entry_marker = metadata
        .merge_entry_by_green
        .binary_search_by_key(&(py_pc as u32), |&(py, _)| py)
        .ok()
        .is_some_and(|i| metadata.merge_entry_by_green[i].1 as usize == jit_pc);
    if !built_as_portal || portal_frame_reg > u8::MAX as u16 {
        return false;
    }
    let Some((mut pc, floor_py_pc)) =
        crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
    else {
        return false;
    };
    if (floor_py_pc as usize != py_pc && !is_trace_entry_marker) || pc >= jit_pc {
        return false;
    }
    while pc < jit_pc {
        let Some(op) = crate::jitcode_runtime::decode_op_at(code, pc) else {
            return false;
        };
        if op.next_pc > jit_pc
            || (!is_trace_entry_marker && python_pc_for_op(op.pc) != py_pc)
            || code.get(op.pc + 1).copied() != Some(portal_frame_reg as u8)
        {
            return false;
        }
        match op.key {
            // Which stack slot a spill addresses does not matter: the whole
            // stack area is rewritten from `live_stack`.
            "setarrayitem_vable_r/rirdd" => {}
            // `rid`: opcode, frame reg, value reg, little-endian descr index.
            // The int vable fields are `last_instr` (0) and `valuestackdepth`
            // (2), both reassigned by the rebuild; a non-`VableField` descr is
            // not frame bookkeeping.
            "setfield_vable_i_imm/rddd" => {
                // opcode, frame reg, u32 immediate, little-endian descr index.
                let Some((&lo, &hi)) = code.get(op.pc + 6).zip(code.get(op.pc + 7)) else {
                    return false;
                };
                let descr_index = lo as usize | ((hi as usize) << 8);
                if !matches!(
                    perfn_descrs.get(descr_index),
                    Some(majit_metainterp::jitcode::RuntimeBhDescr::Descr(descr))
                        if matches!(
                            descr.as_ref(),
                            majit_jitcode::jitcode::BhDescr::VableField { .. }
                        )
                ) {
                    return false;
                }
            }
            "setfield_vable_i/rid" => {
                let Some((&lo, &hi)) = code.get(op.pc + 3).zip(code.get(op.pc + 4)) else {
                    return false;
                };
                let descr_index = lo as usize | ((hi as usize) << 8);
                if !matches!(
                    perfn_descrs.get(descr_index),
                    Some(majit_metainterp::jitcode::RuntimeBhDescr::Descr(descr))
                        if matches!(
                            descr.as_ref(),
                            majit_jitcode::jitcode::BhDescr::VableField { .. }
                        )
                ) {
                    return false;
                }
            }
            // A pure vable read writes no frame state — only a jitcode
            // register, and the rebuild discards those: it reconstructs a
            // PyFrame, sets `last_instr = callee_py_pc - 1` and resumes the
            // PLAIN interpreter (`frame.execute_frame`), which never observes a
            // jitcode register bank.  So whether this op ran before the abort
            // makes no difference to the frame the rebuild hands over.  That is
            // a weaker requirement than the spill stores above, which are
            // admitted only because the rebuild rewrites the areas they touch.
            "getarrayitem_vable_r/ridd>r" => {}
            _ => return false,
        }
        pc = op.next_pc;
    }
    pc == jit_pc
}

/// Read one exact Ref slot from a callee sub-walk's fresh-frame virtualizable
/// shadow. The shadow's `fold_frame_reg` witnesses the strict-fold shape; an
/// entry whose recorded `frame_reg` matches witnesses the live multi-frame
/// inline shape. A foreign frame, missing, non-Ref, or null value is ambiguous
/// and declines.
fn callee_vable_ref_at(
    shadow: Option<&CalleeLocalsShadow>,
    frame_reg: u16,
    slot: usize,
) -> Option<ConcreteValue> {
    if frame_reg == u16::MAX {
        return None;
    }
    let shadow = shadow?;
    let entry = shadow.concrete.get(&(slot as i64)).copied()?;
    let strict_fold_witness = shadow.fold_frame_reg == frame_reg;
    let inline_scope_witness = entry.frame_reg == frame_reg;
    if !strict_fold_witness && !inline_scope_witness {
        return None;
    }
    match entry.value {
        Value::Ref(r) if r.as_usize() != 0 => {
            Some(ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef))
        }
        Value::Int(_) | Value::Float(_) | Value::Ref(_) | Value::Void => None,
    }
}

/// Scoped setter for [`SELFREC_CA_FOLD_ACTIVE`]. Restores the prior value so
/// nested folds and early `?` unwinds cannot strand the exemption enabled.
struct SelfRecCaFoldGuard {
    prior: bool,
}

impl SelfRecCaFoldGuard {
    fn enter() -> Self {
        let prior = SELFREC_CA_FOLD_ACTIVE.with(|c| {
            let prior = c.get();
            c.set(true);
            prior
        });
        Self { prior }
    }
}

impl Drop for SelfRecCaFoldGuard {
    fn drop(&mut self) {
        SELFREC_CA_FOLD_ACTIVE.with(|c| c.set(self.prior));
    }
}

struct ExceptionStringInlineGuard {
    prior: bool,
}

impl ExceptionStringInlineGuard {
    fn enter() -> Self {
        let prior = EXCEPTION_STRING_INLINE_ACTIVE.with(|c| {
            let prior = c.get();
            c.set(true);
            prior
        });
        Self { prior }
    }
}

impl Drop for ExceptionStringInlineGuard {
    fn drop(&mut self) {
        EXCEPTION_STRING_INLINE_ACTIVE.with(|c| c.set(self.prior));
    }
}

/// # Safety
/// `data` must come from [`capture_fbw_store_journal_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn fbw_store_journal_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let area = unsafe { &*(data as *const FbwStoreJournalRootArea) };
    let stores = unsafe { &mut *(*area.stores).as_ptr() };
    for triple in stores.iter_mut() {
        for slot in triple.iter_mut() {
            visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
        }
    }
    let list_effects = unsafe { &mut *(*area.list_effects).as_ptr() };
    for effect in list_effects.iter_mut() {
        match effect {
            FbwListEffect::Append { list, .. } => {
                visitor(unsafe { &mut *(list as *mut pyre_object::PyObjectRef).cast() });
            }
            FbwListEffect::PopEnd { list, w_item, .. } => {
                visitor(unsafe { &mut *(list as *mut pyre_object::PyObjectRef).cast() });
                visitor(unsafe { &mut *(w_item as *mut pyre_object::PyObjectRef).cast() });
            }
        }
    }
    // SAFETY: promoted-list refs can be nursery-resident across the rest of
    // the walk; a minor collection may move them before rollback restores
    // Empty, so each journal slot is a root.
    let append_promote = unsafe { &mut *(*area.append_promote).as_ptr() };
    for list in append_promote.iter_mut() {
        visitor(unsafe { &mut *(list as *mut pyre_object::PyObjectRef).cast() });
    }
    // Cell-store journal: the cell is nursery-allocated and a mid-walk
    // rebind can drop the module dict's only reference.  Forwarding the
    // slot keeps the rollback's restore on the live object.
    let cell_stores = unsafe { &mut *(*area.cell_stores).as_ptr() };
    for entry in cell_stores.iter_mut() {
        match entry {
            FbwCellStore::Int { cell, .. } => {
                visitor(unsafe { &mut *(cell as *mut pyre_object::PyObjectRef).cast() });
            }
            FbwCellStore::Obj { cell, before } => {
                visitor(unsafe { &mut *(cell as *mut pyre_object::PyObjectRef).cast() });
                if !before.is_null() {
                    visitor(unsafe { &mut *(before as *mut pyre_object::PyObjectRef).cast() });
                }
            }
            FbwCellStore::Gc {
                obj,
                before,
                managed,
                ..
            } => {
                if *managed {
                    visitor(unsafe { &mut *(obj as *mut pyre_object::PyObjectRef).cast() });
                }
                if let Value::Ref(before) = before
                    && before.0 != 0
                {
                    visitor(before);
                }
            }
        }
    }
    // Namespace-store journal: the store replaced the displaced binding in its
    // dict slot, so the entry can be that value's only remaining owner, and the
    // rollback reads the dict and the name back to put it there — forward all
    // three.  A null `displaced` is the unbound case and names no object.
    // `written` is the object a `STORE_SUBSCR` rollback compares by identity;
    // it has to move with the dict entry or the comparison misses.
    let namespace_stores = unsafe { &mut *(*area.namespace_stores).as_ptr() };
    for entry in namespace_stores.iter_mut() {
        for slot in [
            &mut entry.namespace as *mut pyre_object::PyObjectRef,
            &mut entry.name as *mut pyre_object::PyObjectRef,
            &mut entry.displaced as *mut pyre_object::PyObjectRef,
        ] {
            if unsafe { *slot }.is_null() {
                continue;
            }
            visitor(unsafe { &mut *slot.cast() });
        }
        if let Some(written) = entry.written.as_mut() {
            if !written.is_null() {
                visitor(unsafe { &mut *(written as *mut pyre_object::PyObjectRef).cast() });
            }
        }
    }
    // The displaced `sys_exc_value` entries are exception objects that may be
    // nursery-resident and no longer referenced elsewhere once the eager store
    // overwrote the EC slot; forward each so a minor collection during the rest
    // of the walk cannot free/move the value the rollback restores.
    let sys_exc = unsafe { &mut *(*area.sys_exc).as_ptr() };
    for displaced in sys_exc.iter_mut() {
        // SAFETY: `PyObjectRef` and `GcRef` share the usize repr; the
        // borrowed area keeps the Vec storage alive for the visit.
        visitor(unsafe { &mut *(displaced as *mut pyre_object::PyObjectRef).cast() });
    }
    // The exception and freshly attached traceback node remain live until the
    // walk commits or rolls back.  The entry may be their only scanned owner
    // after later concrete execution changes the exception's head.
    let traceback_store = unsafe { &mut *(*area.traceback_store).as_ptr() };
    for (exception, node) in traceback_store.iter_mut() {
        visitor(unsafe { &mut *(exception as *mut pyre_object::PyObjectRef).cast() });
        visitor(unsafe { &mut *(node as *mut pyre_object::PyObjectRef).cast() });
    }
    // #57 Option C: each captured in-flight FOR_ITER item is nursery-resident
    // across the rest of the walk (subsequent residual calls allocate and a
    // minor collection moves nursery objects), so forward every entry's item
    // as a root.
    let foriter = unsafe { &mut *(*area.foriter).as_ptr() };
    for entry in foriter.iter_mut() {
        // SAFETY: as above — only the `PyObjectRef` slot is a root; the
        // `usize` body pc and the bool flag are plain scalars.
        visitor(unsafe { &mut *(&mut entry.item as *mut pyre_object::PyObjectRef).cast() });
    }
    // The escape-flush undo snapshot stays armed from force time until the
    // abort epilogue consumes it, and its slots can be the ONLY reference to
    // pre-walk locals the flush displaced (the live-frame slots now hold
    // walk-end values) — forward each so a minor collection on the abort
    // unwind cannot move/free what `restore_escape_flush_undo` writes back.
    let escape_undo = unsafe { &mut *(*area.escape_flush_undo).as_ptr() };
    if let Some(undo) = escape_undo.as_mut() {
        for slot in undo.slots.iter_mut() {
            visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
        }
        // The post-flush image is compared against the live slots, so it has to
        // be forwarded on the same window: a stale address would read as "the
        // residual wrote this" for every moved value and suppress the undo.
        for slot in undo.flush_image.iter_mut() {
            visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
        }
    }
    // The `f_locals` fold's pre-mirror images are armed the same way and live
    // just as long — until the walk's commit or rollback leg consumes them —
    // and the live-frame slots they shadow now hold the mirrored values, so
    // each displaced entry needs forwarding too.
    let locals_mirror = unsafe { &mut *(*area.locals_mirror_undo).as_ptr() };
    for entry in locals_mirror.iter_mut() {
        for slot in entry.slots.iter_mut() {
            visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
        }
    }
    // The vable-force and trace-too-long MIFrame images survive the dispatch
    // unwind in TLS. Their Option<i64> Ref banks are not otherwise visible to
    // the collector, so forward every populated color until the walk-end
    // handler takes the latch. The adopters bridge the pre-drive publication
    // window and the blackhole drivers then install their own packed roots.
    //
    // A `MIFrame` stores each Ref register as the box itself. Once the
    // frames are latched here, `MetaInterp::walk_active_trace_refs` no
    // longer reaches them, so ConstPtr gcrefs are forwarded below.
    let single_frame_blackhole = unsafe { &mut *(*area.single_frame_blackhole).as_ptr() };
    if let Some(latched) = single_frame_blackhole.as_mut() {
        for slot in latched.miframe.ref_regs.iter_mut() {
            if let Some(majit_ir::OpRef::ConstPtr(gcref)) = slot.as_mut() {
                visitor(unsafe { &mut *(&mut gcref.0 as *mut usize).cast() });
            }
        }
        if latched.last_exc_value != 0 {
            visitor(unsafe { &mut *(&mut latched.last_exc_value as *mut i64).cast() });
        }
        // The operand-stack image is resolved to concrete refs at latch time and
        // then held plainly until the adopter publishes it, which is the same
        // invisible-to-the-collector window as the Ref bank above.  The escape
        // flush avoids the question by rooting its stack for the duration of the
        // flush call (`push_resume_ref_roots`); this one outlives a single call,
        // so it is forwarded here instead.
        if let Some(mirror) = latched.mirror_stack.as_mut() {
            for slot in mirror.slots.iter_mut() {
                visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
            }
        }
    }
    let multi_frame_blackhole = unsafe { &mut *(*area.multi_frame_blackhole).as_ptr() };
    if let Some(latched) = multi_frame_blackhole.as_mut() {
        for frame in latched.framestack.frames.iter_mut() {
            for slot in frame.ref_regs.iter_mut() {
                if let Some(majit_ir::OpRef::ConstPtr(gcref)) = slot.as_mut() {
                    visitor(unsafe { &mut *(&mut gcref.0 as *mut usize).cast() });
                }
            }
        }
        if latched.last_exc_value != 0 {
            visitor(unsafe { &mut *(&mut latched.last_exc_value as *mut i64).cast() });
        }
        // Same window as the single-frame arm above.
        if let Some(mirror) = latched.mirror_stack.as_mut() {
            for slot in mirror.slots.iter_mut() {
                visitor(unsafe { &mut *(slot as *mut pyre_object::PyObjectRef).cast() });
            }
        }
    }
    // Forward journal refs so rollback remains valid across moving GC.
    let bridge_iter = unsafe { &mut *(*area.bridge_iter).as_ptr() };
    for entry in bridge_iter.iter_mut() {
        match entry {
            BridgeIterJournalEntry::Range { iter, .. } => {
                visitor(unsafe { &mut *(iter as *mut pyre_object::PyObjectRef).cast() });
            }
            BridgeIterJournalEntry::Cursor { iter, pre_seq, .. } => {
                visitor(unsafe { &mut *(iter as *mut pyre_object::PyObjectRef).cast() });
                visitor(unsafe { &mut *(pre_seq as *mut pyre_object::PyObjectRef).cast() });
            }
        }
    }
    // The latched MidBody carrier is nursery-resident across the abort unwind
    // — the flush boxes Int/Float locals, which can trigger a minor collection
    // that moves these refs before they are written into the frame array — so
    // forward every slot as a root.
    let abort_resume = unsafe { &mut *(*area.abort_resume).as_ptr() };
    if let Some(carrier) = abort_resume.as_mut() {
        match carrier {
            InlineAbortCarrier::MidBody(payload) => {
                visitor(unsafe {
                    &mut *(&mut payload.w_code as *mut pyre_object::PyObjectRef).cast()
                });
                visitor(unsafe {
                    &mut *(&mut payload.w_globals as *mut pyre_object::PyObjectRef).cast()
                });
                if !payload.return_value.is_null() {
                    visitor(unsafe {
                        &mut *(&mut payload.return_value as *mut pyre_object::PyObjectRef).cast()
                    });
                }
                if !payload.constructor_instance.is_null() {
                    visitor(unsafe {
                        &mut *(&mut payload.constructor_instance as *mut pyre_object::PyObjectRef)
                            .cast()
                    });
                }
                for slot in payload.live_locals.iter_mut().flatten() {
                    if let ConcreteValue::Ref(value) = slot {
                        visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
                    }
                }
                for slot in &mut payload.live_stack {
                    if let ConcreteValue::Ref(value) = slot {
                        visitor(unsafe { &mut *(value as *mut pyre_object::PyObjectRef).cast() });
                    }
                }
            }
        }
    }
}

/// #73: classification of a Python opcode's effect on the walk-level
/// symbolic operand stack ([`WalkFrameStateData::vstack_boxes`]), used by
/// [`reconcile_vstack_at_boundary`] to update `vstack_boxes` at an
/// opcode boundary.  The depth delta is already known from
/// `depth_at_py_pc`; this only decides WHERE the new boxes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VstackOpClass {
    /// A single net-result lands on the new TOS, and its box is the last
    /// Ref written via [`write_ref_reg`] during this opcode
    /// (`vstack_last_ref`).  Truncate to the new depth, then overwrite the
    /// new TOS with `vstack_last_ref`.  Covers value producers whose
    /// result is the topmost stack slot: LOAD_FAST / LOAD_CONST /
    /// LOAD_GLOBAL(result) / LOAD_NAME / LOAD_ATTR / BINARY_OP /
    /// BINARY_SUBSCR / COMPARE_OP / unary ops / CALL /
    /// IS_OP / CONTAINS_OP / single-result BUILD_*.
    ResultToTos,
    /// Pyre's method-form `LOAD_GLOBAL` pushes the callable and then a NULL
    /// `self_or_null` sentinel.  The callable is recovered from the
    /// virtualizable shadow; the top slot is an explicit Ref constant so a
    /// guard inside the following CALL can snapshot the live NULL instead of
    /// inheriting an older value from that frame slot.
    LoadGlobalMethod,
    /// The opcode only pops (and/or stores to a local/global/attr/subscr,
    /// or is an unconditional control transfer).  Truncate to the new
    /// depth WITHOUT touching the surviving TOS — the box already in that
    /// slot is the live value.  Covers POP_TOP / POP_JUMP_IF_* /
    /// STORE_FAST / STORE_GLOBAL / STORE_NAME / STORE_ATTR /
    /// STORE_SUBSCR / STORE_SLICE / JUMP_* / RETURN_* / DELETE_*.
    PopOnlyOrSideStore,
    /// `SWAP(i)` permutes the operand stack: it exchanges the TOS box with
    /// the box `i` positions below it (the `localsplus[depth-1]` /
    /// `localsplus[depth-i]` exchange in `swap_values`).  Net depth is
    /// unchanged; the carried `usize` is the decoded `i`.  Reconcile applies
    /// the same exchange to `vstack_boxes`, so a value kept across a SWAP
    /// (e.g. `a + ((i & 1) or 5)` reorders its operands with SWAP before the
    /// short-circuit guard) lands on the right slot instead of latching the
    /// mirror invalid.
    Swap(usize),
    /// `COPY(i)` duplicates the box `i` positions from the top onto the new
    /// TOS (`opcode_copy_value` = `push(peek_at(i-1))`).  Net depth +1; the
    /// new TOS box is `vstack_boxes[depth-1-i]` (the COPIED slot), NOT the
    /// last Ref written — only `COPY 1` (dup of TOS, the and/or/condexpr and
    /// chained-comparison truthiness dup) coincides with `vstack_last_ref`.
    /// Reconcile sources the duplicated slot directly so `COPY i>1`
    /// (duplicating a deeper operand) is faithful.  The carried `usize` is
    /// the decoded `i`.
    Copy(usize),
    /// An exception-machinery opcode (PUSH_EXC_INFO / CHECK_EXC_MATCH /
    /// RERAISE / WITH_EXCEPT_START / RAISE_VARARGS) reached INSIDE an
    /// exception handler.  Its operand-stack effect is not a simple
    /// producer/pop — the unwinder and the exc-info machinery rewrite the
    /// stack — but the lowering writes every resulting operand slot through
    /// `setarrayitem_vable_r`, so the virtualizable shadow holds the correct
    /// post-opcode operand stack.  Reconcile truncates to the new depth and
    /// reseeds the slots from the shadow (`reseed_vstack_from_shadow`); a
    /// slot the shadow cannot source (a genuine NULL exc-info slot, an
    /// Int/Float temp) stays a NONE hole and `mirror_covers_kept` declines
    /// for it (the conservative fallback).  Reached with a valid mirror only
    /// via [`vstack_enter_exception_handler`]; without that handler-entry
    /// reseed the mirror is already invalid at the unwind boundary, so this
    /// arm is inert on the non-exception path.
    ShadowReseed,
    /// `UNPACK_SEQUENCE` / `UNPACK_EX`: pop one sequence and push its
    /// elements (net push > 1), each written through `setarrayitem_vable_r`.
    /// A single `vstack_last_ref` write cannot reconstruct the whole pushed
    /// group, but every pushed element IS in the virtualizable shadow, so
    /// reconcile clears the affected range `[pop_point .. new_depth)` (from
    /// the popped sequence slot upward) to NONE holes and the general
    /// hole-fill sources them from the shadow.  Slots BELOW the popped
    /// sequence keep their mirror-tracked boxes (the hole-fill never
    /// overwrites a non-NONE slot).  Never latches the mirror invalid.
    MultiResultFromShadow,
    /// `LOAD_SPECIAL` (`__enter__` / `__exit__`): pop the context manager and
    /// push the resolved special method followed by the call's `self_or_null`
    /// slot.  Same pop-then-push-a-group shape as
    /// [`VstackOpClass::MultiResultFromShadow`], except that the top slot is a
    /// live NULL the shadow cannot return — its source is a dense array where
    /// an absent slot and a written NULL are the same word — so reconcile
    /// stamps the NULL constant there instead, as [`Self::LoadGlobalMethod`]
    /// does for the other callable/NULL pair.
    LoadSpecialMethod,
    /// Anything that does not fit the shapes above — FOR_ITER or any opcode
    /// this classifier does not recognise.  Latches `vstack_valid = false`
    /// so the overlay omits those slots, which resume re-materializes (zero
    /// regression).
    Unmodeled,
}

/// The frame whose JitCode byte offsets the branch-resume gate readers
/// ([`branch_resume_target_stack_depth`] and the kept-slot hazard checks)
/// resolve through.  Holds the frame's `PyJitCode` payload — its `metadata`
/// (the tables for the jitcode-pc → Python-pc inversion) and `code_ptr` (the
/// liveness key).
///
/// Two constructors mark the frame model: the gate must read the frame whose
/// jitcode the `target` offset indexes, NOT a single global.
/// * [`outer`](Self::outer) — the outermost portal/main frame held by
///   `fbw_mode.snapshot_sym`.
/// * [`current`](Self::current) — the innermost inlined callee being
///   sub-walked (framestack top), or the portal frame when no
///   sub-walk is active.  Mirrors the callee derivation in
///   `walker_capture_multi_frame_inline_snapshot` (the framestack
///   → `ensure_jitcode_index` → `pyjitcode_for_jitcode_index` chain) so the
///   gate and the snapshot encoder consult one consistent active frame.
struct ActiveResumeFrame(std::sync::Arc<crate::PyJitCode>);

impl ActiveResumeFrame {
    /// The outermost portal/main frame (`fbw_mode.snapshot_sym`).  `None`
    /// outside a full-body walk (tests or diagnostic callers).
    fn outer<Sym: WalkSym>(snapshot_sym: *const Sym) -> Option<Self> {
        let full_body_sym = snapshot_sym;
        if full_body_sym.is_null() {
            return None;
        }
        // SAFETY: identical contract to the gate readers — the pointer is set
        // only for the lifetime of the full-body `dispatch_via_miframe`, and
        // only the immutable `payload` Arc is cloned.
        let sym = unsafe { &*full_body_sym };
        if sym.jitcode().is_null() {
            return None;
        }
        let jc = unsafe { &*sym.jitcode() };
        Some(ActiveResumeFrame(jc.payload.clone()))
    }

    /// The active frame at the current walk point: the innermost inlined
    /// callee when a sub-walk is in progress, else the portal frame.
    fn current<Sym: WalkSym>(
        session: &std::cell::RefCell<WalkSession>,
        snapshot_sym: *const Sym,
    ) -> Option<Self> {
        let current_code = session.borrow().last_inline().map(|frame| frame.w_code);
        match current_code {
            Some(callee_w_code) => {
                let idx = crate::state::ensure_jitcode_index(callee_w_code as *const ())?;
                let pjc = crate::state::pyjitcode_for_jitcode_index(idx)?;
                Some(ActiveResumeFrame(pjc))
            }
            None => Self::outer(snapshot_sym),
        }
    }

    fn vstack_coordinate_for_jitcode_pc(
        &self,
        jit_pc: usize,
    ) -> Option<(u32, *const pyre_interpreter::CodeObject, usize)> {
        let pjc = &self.0;
        if pjc.code_ptr.is_null() {
            return None;
        }
        let py_pc = vstack_containing_py_pc(&pjc.metadata, jit_pc);
        // Raw py_pc-keyed static-liveness read: the unpopulated-twin fallback
        // (skeleton / fixture) and the audit oracle.
        let raw_depth = || {
            crate::liveness::liveness_for(pjc.code_ptr)
                .depth_at_py_pc()
                .get(py_pc as usize)
                .copied()
                .unwrap_or(0) as usize
        };
        let depth = if pjc.depth_containing_populated() {
            let depth = pjc.depth_containing_for_jitcode_pc(jit_pc).unwrap_or(0) as usize;
            if pcmap_containing_audit_enabled() {
                assert_eq!(
                    depth,
                    raw_depth(),
                    "PYRE_PCMAP_CONTAINING_AUDIT: resume-frame containing-depth twin diverged at jit_pc {jit_pc} (py {py_pc})"
                );
            }
            depth
        } else {
            raw_depth()
        };
        Some((py_pc, pjc.code_ptr, depth))
    }

    /// Coordinate used while stepping an already-seeded operand-stack
    /// mirror.  A block-head metadata entry is JitCode control-flow trivia,
    /// not a Python opcode boundary: defer reconciliation until the first
    /// real operation in that block, whose containing coordinate is exact.
    fn vstack_step_coordinate_for_jitcode_pc(
        &self,
        jit_pc: usize,
        current_py_pc: u32,
    ) -> Option<(u32, *const pyre_interpreter::CodeObject, usize)> {
        let code_ptr = self.0.code_ptr;
        if code_ptr.is_null() {
            return None;
        }
        let py_pc = vstack_step_py_pc(&self.0.metadata, jit_pc, current_py_pc);
        let depth = vstack_step_depth(&self.0, jit_pc, py_pc);
        Some((py_pc, code_ptr, depth))
    }

    fn body_matches(&self, sub_body: &SubJitCodeBody) -> bool {
        let code = self.0.jitcode.code.as_slice();
        code.len() == sub_body.code.len() && code.as_ptr() == sub_body.code.as_ptr()
    }

    /// Recursive closure calls need their own operand-stack mirror even
    /// while the wider callee-mirror experiment remains opt-in.  RPython
    /// pushes one MIFrame per recursive call, so a conditional expression in
    /// the callee keeps its pre-branch operands on that frame.  The legacy
    /// collapsed sub-walk cannot resume such a kept stack after an earlier
    /// closure-set mutation: replaying from entry sees the mutation and skips
    /// the rest of the call.  Restrict automatic admission to handler-free
    /// closure callees whose code identity matches the live root frame;
    /// exception-bearing callees still use the established path until their
    /// cross-frame handler resume is complete.
    #[allow(dead_code)]
    fn is_handler_free_closure(&self) -> bool {
        let code = self.0.code_ptr;
        if code.is_null() {
            return false;
        }
        let code = unsafe { &*code };
        !code.freevars.is_empty() && code.exceptiontable.is_empty()
    }
}

/// Walker counterpart of `MIFrame.debug_merge_point` for per-CodeObject
/// JitCodes whose source `interp_jit.py` dispatch marker is outside the body
/// the codewriter drains.
///
/// `PyJitCodeMetadata::py_floor_by_jit_pc` is built from
/// `SSARepr::pc_first_insn_pos`, so an exact key is the one place the Python
/// dispatch loop entered that opcode.  Do not use `py_exact_by_jit_pc` here:
/// it also reopens the opcode's owner around out-of-line exception/cleanup
/// blocks.  Those blocks run *inside* `PyFrame.handle_bytecode`, after the
/// `pypyjitdriver.jit_merge_point` in `interp_jit.py`; treating one as a new
/// dispatch boundary can segment while `MetaInterp.last_exc_value` is still
/// transient, then resume without the caught exception.
fn record_python_debug_merge_point<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    jit_pc: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    let Some(active) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) else {
        return Ok(None);
    };
    if active.0.code_ptr.is_null() || !active.0.metadata.built_as_portal {
        return Ok(None);
    }
    // `jit_pc` indexes the body being walked, and `py_floor_by_jit_pc`
    // indexes the active frame's.  A helper sub-walk pushes no Python frame,
    // so the active frame stays the caller's while the offsets restart from
    // zero inside the helper, and a small offset that happens to land on one
    // of the caller's opcode boundaries reads back as a dispatch of the
    // caller's first instructions.  `debug_merge_point` belongs to
    // `opimpl_jit_merge_point` (`pyjitpl.py`), which only the portal reaches,
    // so require the walked body to be the frame's own.
    if !std::ptr::eq(code.as_ptr(), active.0.jitcode.code.as_slice().as_ptr()) {
        return Ok(None);
    }
    let Ok(boundary_index) = active
        .0
        .metadata
        .py_floor_by_jit_pc
        .binary_search_by_key(&(jit_pc as u32), |&(off, _)| off)
    else {
        return Ok(None);
    };
    let py_pc = active.0.metadata.py_floor_by_jit_pc[boundary_index].1;
    let w_code = pyre_interpreter::live_code_wrapper(active.0.code_ptr.cast()) as usize;
    if w_code == 0 {
        return Ok(None);
    }

    let (portal_call_depth, current_call_id, is_being_profiled, already_recorded) = {
        let mut session = ctx.session.borrow_mut();
        let is_being_profiled = session.is_being_profiled;
        let mut active_frame_index = None;
        for (index, frame) in session.framestack.iter().enumerate() {
            if frame.w_code != 0 {
                active_frame_index = Some(index);
            }
        }
        let slot = match active_frame_index {
            Some(index) => &mut session.framestack[index].debug_merge_point_py_pc,
            None => &mut session.root_debug_merge_point_py_pc,
        };
        let already_recorded = *slot == Some(py_pc);
        *slot = Some(py_pc);
        let depth = ctx
            .trace_ctx
            .portal_call_depth_fn
            .as_ref()
            .map(|f| f() as usize)
            .unwrap_or(0);
        let call_id = ctx
            .trace_ctx
            .current_call_id_fn
            .as_ref()
            .map(|f| f())
            .unwrap_or(0);
        (depth, call_id, is_being_profiled, already_recorded)
    };
    if already_recorded {
        return Ok(None);
    }

    // pyjitpl.py `MIFrame.debug_merge_point`:
    // `[jd_index, portal_call_depth, current_call_id] + greenkey`, where the
    // pypyjit greenkey is `(next_instr, is_being_profiled, pycode)`.
    let debug_args = [
        ctx.trace_ctx.const_int(0),
        ctx.trace_ctx.const_int(portal_call_depth as i64),
        ctx.trace_ctx.const_int(current_call_id as i64),
        ctx.trace_ctx.const_int(py_pc as i64),
        ctx.trace_ctx.const_int(is_being_profiled as i64),
        ctx.trace_ctx.const_ref(w_code as i64),
    ];
    ctx.trace_ctx
        .record_op(OpCode::DebugMergePoint, &debug_args);

    // `MIFrame.debug_merge_point` performs this check immediately after
    // recording the marker. A location that already overflowed has
    // `force_finish_trace` set by `prepare_trace_segmenting`; close it at the
    // next opcode boundary once it passes 80% of `trace_limit`, before the
    // `_interpret` loop can turn the same attempt into another 100% abort.
    //
    // As in the explicit `jit_merge_point` handler, surface the compile action
    // only from the top-level walk. An inline sub-walk has no driver arm that
    // can consume a segmented trace independently of its enclosing call.
    if ctx.is_top_level
        && ctx.trace_ctx.force_finish_trace()
        && ctx.trace_ctx.num_ops() > ctx.trace_ctx.trace_limit() * 4 / 5
        // PyPy reaches `_create_segmented_trace_and_blackhole` from
        // `MIFrame.debug_merge_point`, then `convert_and_run_from_pyjitpl`
        // transfers `MetaInterp.last_exc_value` into the first blackhole
        // frame.  Pyre's per-CodeObject marker is synthesized by the walker,
        // whose segmented-trace handoff has no blackhole frame on which to
        // preserve that transient slot.  Defer the cut until the handler has
        // consumed/cleared it; the DEBUG_MERGE_POINT itself still remains at
        // the orthodox opcode boundary, and the next safe marker performs the
        // same 0.8x check.
        && ctx.session.borrow().last_exc_value.is_none()
    {
        return create_segmented_trace(ctx, jit_pc, py_pc as usize, true);
    }
    Ok(None)
}

/// Walker-side port of `pyjitpl.py handle_possible_exception`'s
/// exception branch.
///
/// Emit `GuardException(exc_type_const)` + snapshot, reading the
/// exception's `ob_header.ob_type` from
/// `WalkContext::last_exc_value_concrete` to pin the class.  Mirrors
/// RPython's `class_of_box_known` shape: the guard recovery has the
/// exact subclass available at replay, matching `opimpl_raise`'s
/// `GUARD_CLASS(exc, cls_of_box(exc))` pattern (`pyjitpl.py`).
///
/// Falls back to `GuardNoException` when the concrete exception
/// pointer is unavailable (sub-walk shadow gap or non-Ref concrete) —
/// the residual call op stays in the trace and the optimizer's
/// per-call guard-emission still catches exception divergence at
/// replay, just without the class pin.
pub(crate) fn walker_record_guard_exception<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    let exc_obj = match ctx.last_exc_value_concrete() {
        ConcreteValue::Ref(p) if !p.is_null() => p,
        _ => {
            ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, pc)?;
            return Ok(());
        }
    };
    let class_of_last_exc_is_const = ctx.fbw_mode.class_of_last_exc_is_const;
    // `pyjitpl.py` / `pyjitpl.rs`: always emit
    // `GuardException` with a const class pin, but keep its live result box
    // unless the exception class was already proven constant.
    // `ob_header.ob_type` is the layout vtable `allocate_instance` stamps.
    // `_getusercls` shares one vtable across every `_new_exception` class of
    // a realbase, so the guard pins the layout, not the Python class.
    let exc_type_ptr = unsafe {
        (*(exc_obj as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type as i64
    };
    let exc_type_const = ctx.trace_ctx.const_int(exc_type_ptr);
    let guard_op = ctx
        .trace_ctx
        .record_guard(OpCode::GuardException, &[exc_type_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    // `op.setref_base(val)` supplies the recording-time shadow without
    // changing the guard result's live replay identity.
    ctx.trace_ctx.set_opref_concrete(
        guard_op,
        majit_ir::Value::Ref(majit_ir::GcRef(exc_obj as usize)),
    );
    let exc_box = if class_of_last_exc_is_const {
        ctx.trace_ctx.const_ref(exc_obj as usize as i64)
    } else {
        guard_op
    };
    ctx.set_last_exc_value_op(exc_box);
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    Ok(())
}

fn clear_walk_exception<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>) {
    ctx.clear_last_exc_value();
}

/// `pyjitpl.py MetaInterp.direct_libffi_call` — the `OS_LIBFFI_CALL`
/// sub-case of the forces-virtual-or-virtualizable branch, ahead of the
/// release-gil and may-force ones.
///
/// ```python
/// def direct_libffi_call(self, argboxes, valueconst, orig_calldescr):
///     assert self.staticdata.has_libffi_call
///     box_cif_description = argboxes[1]
///     if not isinstance(box_cif_description, ConstInt):
///         return None     # cannot be handled by direct_libffi_call()
///     cif_description = box_cif_description.getint()
///     ...
///     calldescr = self.cpu.calldescrof_dynamic(cif_description, extrainfo)
///     if calldescr is None:
///         return None     # cannot be handled by direct_libffi_call()
///     box_exchange_buffer = argboxes[3]
///     for i in range(cif_description.nargs):
///         kind, descr, itemsize = get_arg_descr(self.cpu, cif_description.atypes[i])
///         ofs = cif_description.exchange_args[i]
///         assert ofs % itemsize == 0
///         ...record GETARRAYITEM_RAW_I / _F...
///     c_saveall = ConstInt(rffi.RFFI_ERR_ALL | rffi.RFFI_ALT_ERRNO)
///     opnum = rop.call_release_gil_for_descr(orig_calldescr)
///     return self.history.record_nospec(opnum,
///                                       [c_saveall, argboxes[2]] + arg_boxes,
///                                       valueconst, calldescr)
/// ```
///
/// The recorded call takes the argument *values* instead of the buffer, so
/// the buffer itself never reaches a call and can stay virtual; the result
/// is written back into it by the `raw_store` that follows the call in
/// `jit_libffi.rs`'s `_do_ffi_call_*`.
///
/// Every "cannot be handled" answer is [`LibffiCallOutcome::Declined`],
/// which puts the caller back on the release-gil / may-force selection —
/// the same fallthrough upstream's `if resbox is None:` takes.  Beyond that the shape is
/// [`direct_call_release_gil`]'s: the helper still runs concretely on the
/// original `allboxes`, the heapcache is still invalidated on the
/// corresponding `CALL_MAY_FORCE_*`, and the same two guards close it.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn direct_libffi_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    pc: usize,
    caller: &'static str,
) -> Result<LibffiCallOutcome, DispatchError> {
    // pyjitpl.py: box_cif_description = argboxes[1]; if not ConstInt: return None
    let Some(cif_box) = allboxes.get(1).copied() else {
        return Ok(LibffiCallOutcome::Declined);
    };
    let Some(majit_ir::Value::Int(cif_description)) = cif_box.inline_const_to_value() else {
        return Ok(LibffiCallOutcome::Declined);
    };
    if cif_description <= 0 {
        return Ok(LibffiCallOutcome::Declined);
    }
    let cif_description = cif_description as usize;
    // pyjitpl.py: box_exchange_buffer = argboxes[3]
    let (Some(func_addr_box), Some(exchange_buffer_box)) =
        (allboxes.get(2).copied(), allboxes.get(3).copied())
    else {
        return Ok(LibffiCallOutcome::Declined);
    };
    // `cpu.calldescrof_dynamic(cif_description, extrainfo)` returns None on
    // any kind the backend cannot carry (`ffisupport.py get_call_descr_dynamic`
    // catching `UnsupportedKind`), and so does this.
    let Some(plan) = libffi_call_plan(cif_description, dst_bank) else {
        return Ok(LibffiCallOutcome::Declined);
    };

    // pyjitpl.py `vable_and_vrefs_before_residual_call` — libffi is
    // unconditionally a forces sub-case, so no gate; see
    // [`direct_call_release_gil`] for the IR-vs-heap split.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, pc);

    // pyjitpl.py: one GETARRAYITEM_RAW per non-void argument, reading the
    // slot the argument conversion already stored into.
    let mut arg_boxes: Vec<OpRef> = Vec::with_capacity(plan.args.len());
    for arg in &plan.args {
        let index = ctx.trace_ctx.const_int(arg.index);
        let arg_descr = crate::descr::raw_carray_descr(arg.item_type, arg.item_size, arg.signed);
        let opcode = match arg.item_type {
            Type::Float => OpCode::GetarrayitemRawF,
            _ => OpCode::GetarrayitemRawI,
        };
        arg_boxes.push(ctx.trace_ctx.record_op_with_descr(
            opcode,
            &[exchange_buffer_box, index],
            arg_descr,
        ));
    }

    // pyjitpl.py: "for now, any call via libffi saves and restores
    // everything (that is, errno and SetLastError/GetLastError on Windows).
    // Note these flags match the ones in clibffi.ll_callback".
    let c_saveall = ctx
        .trace_ctx
        .const_int(majit_jitcode::rffi::RFFI_ERR_ALL | majit_jitcode::rffi::RFFI_ALT_ERRNO);
    // pyjitpl.py: opnum = rop.call_release_gil_for_descr(orig_calldescr),
    // asserted equal to the one the dynamic descr selects.
    let opcode = match dst_bank {
        'i' => OpCode::CallReleaseGilI,
        'f' => OpCode::CallReleaseGilF,
        'v' => OpCode::CallReleaseGilN,
        _ => unreachable!(
            "{caller}: dst_bank '{dst_bank}' reached direct_libffi_call, but \
             libffi_call_plan admits only the 'i' / 'f' / 'v' result kinds \
             resoperation.py:1240-1248 gives a CALL_RELEASE_GIL opcode"
        ),
    };
    let calldescr = majit_metainterp::make_call_descr_sized_with_effect(
        &plan.arg_types,
        plan.result_type,
        plan.result_signed,
        plan.result_size,
        ei.clone(),
    );
    let mut new_args = Vec::with_capacity(arg_boxes.len() + 2);
    new_args.push(c_saveall);
    new_args.push(func_addr_box);
    new_args.extend_from_slice(&arg_boxes);
    // Record-then-execute, as in `direct_call_release_gil`: the FBW walker
    // records before it runs the helper, and the result back-patches
    // `recorded`.
    let recorded = ctx
        .trace_ctx
        .record_op_with_descr(opcode, &new_args, calldescr);

    // pyjitpl.py step 2 / step 5 both key on the corresponding
    // `CALL_MAY_FORCE_*`, on the ORIGINAL `allboxes` and the ORIGINAL descr:
    // what actually runs is the `jit_ffi_call_impl_*` helper the jitcode
    // named, not the C function the recorded op names.
    let mayforce_opnum = match dst_bank {
        'i' => OpCode::CallMayForceI,
        'f' => OpCode::CallMayForceF,
        'v' => OpCode::CallMayForceN,
        _ => unreachable!("dst_bank validated above"),
    };
    let resid_exec = try_execute_residual_call_via_executor(
        ctx,
        mayforce_opnum,
        allboxes,
        call_descr,
        recorded,
        pc,
        None,
        false,
    )?;
    let resid_raised = match resid_exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_abort_nested_unjournaled_residual(ctx, pc, Some(cause))?;
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };
    debug_assert!(
        !resid_raised || ei.check_can_raise(false),
        "{caller}: libffi helper raised on a `!can_raise` EI — \
         EffectInfo claim/reality mismatch"
    );
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(mayforce_opnum, Some(ei), allboxes);
    if dst_bank != 'v' {
        write_residual_call_result_to_dst(ctx, pc, dst, dst_bank, recorded)?;
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    if ei.check_can_raise(false) {
        if resid_raised {
            walker_record_guard_exception(ctx, pc)?;
            let exc = ctx
                .last_exc_value()
                .expect("resid_raised implies last_exc_value seeded by the Err branch");
            let exc_concrete = ctx.last_exc_value_concrete();
            return Ok(LibffiCallOutcome::Raised(DispatchOutcome::SubRaise {
                exc,
                exc_concrete,
            }));
        }
        ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    Ok(LibffiCallOutcome::Recorded)
}

/// What [`direct_libffi_call`] did, in the three shapes the caller has to
/// tell apart: upstream's `resbox is None` fallthrough, its recorded
/// specialization, and the raising arm that stops the walk.
pub(crate) enum LibffiCallOutcome {
    /// `return None  # cannot be handled by direct_libffi_call()`.  The
    /// caller goes on to the release-gil / may-force selection.
    Declined,
    /// The `CALL_RELEASE_GIL_*` and both of the forces branch's guards were
    /// recorded; the caller has nothing left to do for this residual.
    Recorded,
    /// The concretely-executed helper raised, so nothing after the call is
    /// recorded onto this arm.
    Raised(DispatchOutcome),
}

/// One argument of a libffi call as the trace records it:
/// `ffisupport.py get_arg_descr`'s `(kind, descr, itemsize)` triple plus the
/// index into the exchange buffer that `descr` scales.
struct LibffiArg {
    item_type: Type,
    item_size: usize,
    signed: bool,
    index: i64,
}

/// What `cpu.calldescrof_dynamic` needs off a `CIF_DESCRIPTION`, and what
/// the per-argument `GETARRAYITEM_RAW` records need.
struct LibffiCallPlan {
    args: Vec<LibffiArg>,
    arg_types: Vec<Type>,
    result_type: Type,
    result_signed: bool,
    result_size: usize,
}

/// `ffisupport.py get_call_descr_dynamic` + `get_arg_descr`, read straight
/// off the `CIF_DESCRIPTION` block at `cif_description`.
///
/// `None` is `UnsupportedKind`: a struct passed by value, a `long double`, a
/// single-float, and a `long long` wider than a machine word have no
/// argument slot a `CALL_RELEASE_GIL` can carry here, and a result kind that
/// disagrees with the residual's own destination bank means the jitcode and
/// the cif describe different calls.
///
/// `pyjitpl.py`'s `assert self.staticdata.has_libffi_call` "constant-folds
/// away the rest of this function if the codewriter didn't produce any
/// OS_LIBFFI_CALL at all".  `_cffi_backend` is the only producer of that
/// oopspec and the only builder of the block, so a build without it reads no
/// `CIF_DESCRIPTION` and declines to the same `resbox is None` fallthrough an
/// unsupported kind takes.
fn libffi_call_plan(cif_description: usize, dst_bank: char) -> Option<LibffiCallPlan> {
    let hooks = pyre_interpreter::importing::optional_module_hooks()?;
    // `ffi_result = cif_description.rtype; reskind = get_ffi_type_kind(...)`.
    let cif = unsafe { (hooks.libffi_cif_shape)(cif_description) }?;
    let reskind = cif.rtype.kind;
    // `if reskind == 'v': result_size = 0 else: intmask(ffi_result.c_size)`;
    // `is_ffi_type_signed` answers `kind != 'u'`.
    let (result_type, result_signed) = match reskind {
        b'v' if dst_bank == 'v' => (Type::Void, false),
        b'i' if dst_bank == 'i' => (Type::Int, true),
        b'u' if dst_bank == 'i' => (Type::Int, false),
        b'f' if dst_bank == 'f' => (Type::Float, true),
        _ => return None,
    };
    let result_size = if reskind == b'v' { 0 } else { cif.rtype.size };
    let mut args = Vec::with_capacity(cif.args.len());
    let mut arg_types = Vec::with_capacity(cif.args.len());
    for (atype, ofs) in cif.args {
        // `_get_ffi2descr_dict`: an integer argument keys on its own size, a
        // float on `('f', 0)`.  `('v', 0)` is the one entry that carries no
        // descr — the argument occupies no slot and is skipped.
        let (item_type, item_size, signed) = match atype.kind {
            b'v' => continue,
            b'i' => (Type::Int, atype.size, true),
            b'u' => (Type::Int, atype.size, false),
            b'f' => (Type::Float, 8, false),
            _ => return None,
        };
        if item_size == 0 {
            return None;
        }
        // `ofs = cif_description.exchange_args[i]; assert ofs % itemsize == 0`
        // — the index a `GETARRAYITEM_RAW` carries is that offset in items,
        // so an offset the item size does not divide has no such spelling.
        if ofs % item_size != 0 {
            return None;
        }
        args.push(LibffiArg {
            item_type,
            item_size,
            signed,
            index: i64::try_from(ofs / item_size).ok()?,
        });
        arg_types.push(item_type);
    }
    Some(LibffiCallPlan {
        args,
        arg_types,
        result_type,
        result_signed,
        result_size,
    })
}

fn direct_call_release_gil<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ei: &majit_ir::EffectInfo,
    allboxes: &[OpRef],
    descr: DescrRef,
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    pc: usize,
    caller: &'static str,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    // pyjitpl.py `vable_and_vrefs_before_residual_call` —
    // release-gil is unconditionally a forces sub-case
    // (`pyjitpl.py` sits inside the forces-virtual-or-virtualizable
    // branch), so no `emit_guard_not_forced` gate is needed here.
    // RPython's pre-call vrefs heap mutation
    // (`pyjitpl.py vrefs_before_residual_call`) and the
    // after-call helpers (`pyjitpl.py`) are handled by the
    // residual-call execution path — see
    // [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
    // split rationale.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, pc);
    // pyjitpl.py: realfuncaddr, saveerr = effectinfo.call_release_gil_target
    let (realfuncaddr, saveerr) = ei.call_release_gil_target;
    // pyjitpl.py: funcbox/savebox ConstInt
    let savebox = ctx.trace_ctx.const_int(saveerr as i64);
    let funcbox_real = ctx.trace_ctx.const_int(realfuncaddr as i64);
    // pyjitpl.py: opnum = rop.call_release_gil_for_descr(calldescr).
    // resoperation.py maps the descr's normalized result
    // type to {CALL_RELEASE_GIL_I, CALL_RELEASE_GIL_F, CALL_RELEASE_GIL_N};
    // 'r' is explicitly skipped (`# no such thing`).
    let opcode = match dst_bank {
        'i' => OpCode::CallReleaseGilI,
        'f' => OpCode::CallReleaseGilF,
        'v' => OpCode::CallReleaseGilN,
        'r' => panic!(
            "{caller}: CALL_RELEASE_GIL_R has no upstream counterpart \
             (resoperation.py:1243-1244 `# no such thing`); a 'r'-result \
             release-gil callee cannot be lowered to an IR op the \
             optimizer/backend can consume."
        ),
        _ => unreachable!(
            "{caller}: dst_bank '{dst_bank}' not supported by direct_call_release_gil \
             (callers must pass 'i' / 'f' / 'v' per resoperation.py:1240-1248)"
        ),
    };
    // pyjitpl.py: history.record_nospec(opnum,
    //                          [savebox, funcbox] + argboxes[1:], ..., calldescr)
    let mut new_args = Vec::with_capacity(allboxes.len() + 1);
    new_args.push(savebox);
    new_args.push(funcbox_real);
    if allboxes.len() > 1 {
        new_args.extend_from_slice(&allboxes[1..]);
    }
    // Ordering note: the walker records the op here and concrete-executes
    // below, whereas `do_residual_call` executes the `CALL_MAY_FORCE_*`
    // first and only records (`direct_call_release_gil`, pyjitpl.py)
    // after `vrefs_after_residual_call()`. This record-then-execute order
    // is a structural property of the FBW residual-call path shared by the
    // generic walker — not specific to release-gil — so the result still
    // back-patches `recorded`; a 1:1 execute-then-record reorder would
    // require restructuring the whole walker residual-call flow.
    let recorded = ctx.trace_ctx.record_op_with_descr(opcode, &new_args, descr);
    // The forces-branch concrete-execute and the heapcache invalidation
    // both key on the corresponding `CALL_MAY_FORCE_*` (`pyjitpl.py-
    // 2072 opnum1`), NOT `CALL_RELEASE_GIL_*`.
    let mayforce_opnum = match dst_bank {
        'i' => OpCode::CallMayForceI,
        'f' => OpCode::CallMayForceF,
        'v' => OpCode::CallMayForceN,
        _ => unreachable!("dst_bank validated above"),
    };
    // pyjitpl.py `do_residual_call` step 2: the forces branch
    // concrete-executes the helper via `executor.execute_varargs(opnum1,
    // allboxes, descr)` BEFORE step 4 selects which CALL_* op to record.
    // The release-gil sub-case (`pyjitpl.py`) sits *inside* that
    // branch, so it is executed identically to a `CALL_MAY_FORCE_*` — on
    // the **original** `allboxes` (`allboxes[0]` is the wrapper funcbox;
    // the recorded op above used the re-shaped `[savebox, funcbox_real,
    // …]`).  This removes the asymmetry where release-gil was the only
    // forces sub-case that recorded without executing — the same
    // un-executed-side-effect SIGBUS class already closed for the
    // may-force branch.
    // `pyjitpl.py` `do_residual_call` / `execute_and_record_varargs` runs
    // `call_external_function` (`ccall_*`). Only the recorded op is
    // `CALL_RELEASE_GIL` (`direct_call_release_gil`).
    // `try_execute_residual_call_via_executor` self-gates (authoritative-
    // executor flag, const-funcbox, symbolic-fnaddr-tag sanity) and degrades to
    // recording-only on decline; on success it stamps `recorded` with the
    // concrete result and brackets the active virtualizable token for the
    // duration of the call (its may-force arm).
    let resid_exec = try_execute_residual_call_via_executor(
        ctx,
        mayforce_opnum,
        allboxes,
        call_descr,
        recorded,
        pc,
        None,
        false,
    )?;
    // A decline leaves the call recorded symbolically WITHOUT running it, so
    // the walk-end no-replay commit must stay off for this trace.
    let resid_raised = match resid_exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_abort_nested_unjournaled_residual(ctx, pc, Some(cause))?;
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };
    debug_assert!(
        !resid_raised || ei.check_can_raise(false),
        "{caller}: release-gil helper raised on a `!can_raise` EI — \
         EffectInfo claim/reality mismatch"
    );
    // pyjitpl.py `heapcache.invalidate_caches_varargs(opnum1, descr,
    // allboxes)` — forces-branch invalidation uses `opnum1`
    // (`CALL_MAY_FORCE_*`) on the **original** `allboxes` so heapcache's
    // `mark_escaped_varargs` sees the same operand identities upstream does.
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(mayforce_opnum, Some(ei), allboxes);
    // pyjitpl.py execute_varargs: `make_result_of_lastop(op)` runs
    // BEFORE `handle_possible_exception()` precisely "because we need the box
    // to show up in get_list_of_active_boxes()".  Write the recorded OpRef
    // into `registers_*[dst]` for every non-void result REGARDLESS of whether
    // the helper raised — the GUARD_NOT_FORCED resume snapshot captured below
    // must see the result box in its slot, exactly as upstream's
    // unconditional (non-void) make_result_of_lastop does.  `'v'` (void) is a
    // no-op in `write_residual_call_result_to_dst`; on a raised call the OpRef
    // carries a Null concrete shadow, never read on the exception path.
    if dst_bank != 'v' {
        write_residual_call_result_to_dst(ctx, pc, dst, dst_bank, recorded)?;
    }
    // pyjitpl.py GUARD_NOT_FORCED — unconditional on the outer
    // forces-virtual-or-virtualizable branch (the release-gil sub-case is
    // inside that branch).  A force inside the concrete-executed helper is
    // observed by the executor's vable token bracket above, surfacing as
    // `VableEscapedDuringResidualCall` (aborting the walk) before reaching
    // this guard.
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    // pyjitpl.py handle_possible_exception — emits GUARD_EXCEPTION
    // when the recording-time helper raised (then finishframe_exception:
    // surface `SubRaise` so the dead arm tail is not recorded onto the
    // exception path), else GUARD_NO_EXCEPTION.  The capture ports
    // `capture_resumedata(after_residual_call=True)` so the optimizer's
    // `store_final_boxes_in_guard` finds a populated `rd_resume_position`.
    if ei.check_can_raise(false) {
        if resid_raised {
            walker_record_guard_exception(ctx, pc)?;
            let exc = ctx
                .last_exc_value()
                .expect("resid_raised implies last_exc_value seeded by the Err branch");
            let exc_concrete = ctx.last_exc_value_concrete();
            return Ok(Some(DispatchOutcome::SubRaise { exc, exc_concrete }));
        }
        ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    Ok(None)
}

/// Resolve a concrete callable pointer to `(w_code, arg_count, has_closure)`,
/// validating the `Function -> PyCode -> CodeObject` type chain at every
/// hop.  Returns `None` — decline to the orthodox residual call — when any link
/// fails to type-check.
///
/// The `callable` comes from `ctx.concrete_registers_r`, a best-effort shadow
/// that is NOT a GC root: a collection during the walk can leave the shadow
/// pointing at freed/relocated memory whose first word still happens to read
/// `FUNCTION_TYPE`.  Reading `(*callable).code` then yields a non-`PyCode`
/// (a host-allocated code wrapper never lives in the GC heap), so the
/// `CODE_TYPE` tag check rejects the stale shadow before `code_ptr` is read —
/// degrading to a residual call instead of dereferencing garbage.  Both walker
/// call levers (inline-at-residual and self-recursive `CALL_ASSEMBLER`) share
/// this resolver so the staleness guard is applied uniformly.
///
/// # Safety
/// `callable` must be a non-null pointer obtained from a `ConcreteValue::Ref`.
pub(crate) unsafe fn resolve_inlinable_callee(
    callable: pyre_object::PyObjectRef,
) -> Option<(*const (), usize, bool)> {
    unsafe {
        // `is_function` also admits METHOD_DESCRIPTOR_TYPE.  Inlinable
        // callees are `function.py Function` and the MixedModule-wrapped
        // `BuiltinFunction` that still carries PyCode (`_load_lazily`).
        // Interp-level builtins have BuiltinCode and fail the CODE_TYPE
        // check below.  Slot wrappers / method descriptors stay residual.
        if !pyre_interpreter::is_function(callable) {
            return None;
        }
        let ob_type = (*callable).ob_type as *const () as usize;
        let function_type_addr = &pyre_interpreter::FUNCTION_TYPE as *const _ as usize;
        let builtin_function_type_addr =
            &pyre_interpreter::BUILTIN_FUNCTION_TYPE as *const _ as usize;
        if ob_type != function_type_addr && ob_type != builtin_function_type_addr {
            return None;
        }
        let w_code = pyre_interpreter::function_get_code(callable);
        if w_code.is_null() {
            return None;
        }
        let code_type_addr = &pyre_interpreter::pycode::CODE_TYPE as *const _ as usize;
        if (*(w_code as *const pyre_object::pyobject::PyObject)).ob_type as *const () as usize
            != code_type_addr
        {
            return None;
        }
        let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject;
        if raw.is_null() {
            return None;
        }
        // Calling a generator / coroutine / async-generator function does not
        // run its body — it builds a suspended frame object and returns it.
        // Its jitcode therefore starts at `RETURN_GENERATOR`, which the
        // codewriter lowers to an `abort_permanent` marker: inlining the body
        // at the call site walks code the call never reaches, so the attempt
        // can only end in an abort that discards the whole trace.  Decline to
        // the residual call instead — the creation is an ordinary allocating
        // call the walk steps over, and the body's suspension never enters the
        // trace.  `generator.py` `should_not_inline` (consumed at
        // `:63`) is upstream's decision point for the same boundary; it governs
        // the *body* at `send_ex`, which is a separate question from the
        // creation call.
        if pyre_interpreter::pyframe::code_flags_make_generator((*raw).flags) {
            return None;
        }
        let closure = pyre_interpreter::function_get_closure(callable);
        Some((w_code, (*raw).arg_count as usize, !closure.is_null()))
    }
}

/// `space.lookup(w_obj, '__call__')` for a callable that is neither a
/// `Function` nor a `Method` — `descroperation.py
/// DescrOperation.call_args`'s third arm.
///
/// Returns the resolved `__call__` alongside the class it was found through and
/// that class's version tag, which is what pins the lookup: the walker's
/// counterpart of `space.lookup` promoting the type.  What kind of callable the
/// answer is — an app-level function to inline, a builtin gateway to descend
/// into — is the caller's question; a lookup this cannot pin (an unversioned
/// class, a class whose `__call__` is `type.__call__`) declines here.
///
/// # Safety
/// `callable` must be a valid object.
unsafe fn lookup_instance_dunder_call(
    callable: pyre_object::PyObjectRef,
) -> Option<(pyre_object::PyObjectRef, pyre_object::PyObjectRef, u64)> {
    if callable.is_null() {
        return None;
    }
    // A class reaches `type.__call__`, which instantiates rather than calls a
    // `__call__` found on the metaclass; `try_walker_inline_type_call` owns
    // that shape and resolves `__new__` / `__init__` for it.
    if unsafe { pyre_object::is_type(callable) } {
        return None;
    }
    let w_class = unsafe { (*callable).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        return None;
    }
    // A version tag of 0 is a type whose dict changes are not tracked, so the
    // `__call__` answer below cannot be pinned.
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        return None;
    }
    let method = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, "__call__") }?;
    Some((method, w_class, version_tag))
}

/// `typeobject.py descr_call` for a class whose `__new__` is a builtin gateway.
///
/// When `__init__` is `object`'s, the whole of `type.__call__` is the `__new__`
/// call, which `try_walker_inline_builtin_call` enters with the class as its
/// receiver.  `object.__init__` rejects surplus arguments only when `__new__`
/// is `object`'s or `__init__` is overridden, and neither holds here, so that
/// half of `descr_call` has no effect to reproduce.
///
/// That is true of the class named at the call; `descr_call` looks `__init__`
/// up on the type of what `__new__` returned.  A builtin `__new__` handed the
/// exact class returns another type only through a conversion dunder of an
/// argument (`__int__`, `__str__`, ...), so every argument is required to be
/// an instance of a builtin type, which the caller pins — but only in the
/// `__init__`-is-`object`'s case, where skipping `__init__` is valid only if
/// the result type cannot change.
///
/// When `__init__` is itself a builtin gateway (`list.__init__` calling
/// `extend`), `descr_call` runs both halves and the result type is the class
/// being called.  The conversion-dunder pin does not apply: `list(x)` iterates
/// `x` rather than converting it.  The second return is that `__init__`; the
/// caller walks it after `__new__` and keeps the instance.
///
/// # Safety
/// `callable` and every entry of `args` must be valid objects.
fn exact_complex_type(w_type: pyre_object::PyObjectRef) -> bool {
    let complex_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::COMPLEX_TYPE);
    !complex_type.is_null() && std::ptr::eq(w_type, complex_type)
}

unsafe fn resolve_type_call_builtin_new(
    callable: pyre_object::PyObjectRef,
    args: &[ConcreteValue],
) -> Option<(pyre_object::PyObjectRef, Option<pyre_object::PyObjectRef>)> {
    if callable.is_null() || !unsafe { pyre_object::is_type(callable) } {
        return None;
    }
    let w_metatype = pyre_interpreter::typedef::w_type();
    let w_object = pyre_interpreter::typedef::w_object();
    if w_object.is_null() || std::ptr::eq(callable, w_metatype) {
        return None;
    }
    // A metaclass may supply its own `__call__`; only `type`'s is this shape.
    if !std::ptr::eq(unsafe { (*callable).w_class }, w_metatype) {
        return None;
    }
    if unsafe { pyre_object::typeobject::w_type_get_version_tag(callable) } == 0 {
        return None;
    }
    if unsafe {
        pyre_object::w_type_disallows_instantiation(callable)
            || pyre_object::w_type_is_abstract(callable)
            || pyre_object::typeobject::w_type_get_hasuserdel(callable)
            || pyre_object::typeobject::w_type_has_vectorcall(callable)
    } {
        return None;
    }
    // Both answers below are baked by identity under `callable`'s version tag,
    // which an in-place `write_cell` store does not move.
    if unsafe { type_attr_is_cell_backed(callable, "__new__") }
        || unsafe { type_attr_is_cell_backed(callable, "__init__") }
    {
        return None;
    }
    let tp_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(callable, "__new__") }?;
    let obj_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__new__") };
    if Some(tp_new) == obj_new || !unsafe { pyre_interpreter::is_function_carrier(tp_new) } {
        return None;
    }
    let tp_init = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(callable, "__init__") };
    let obj_init = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__init__") };
    let builtin_init = if tp_init == obj_init {
        None
    } else {
        let init = tp_init?;
        if !unsafe { pyre_interpreter::is_function_carrier(init) } {
            return None;
        }
        Some(init)
    };
    for arg in args {
        let ConcreteValue::Ref(arg) = arg else {
            return None;
        };
        if arg.is_null()
            || *arg == pyre_object::PY_NULL
            || (pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(*arg))
        {
            return None;
        }
        let w_class = unsafe { (**arg).w_class };
        if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
            return None;
        }
        // Skipping `__init__` is valid only when `__new__` cannot return
        // another type through a conversion dunder.  Running a builtin
        // `__init__` is the `descr_call` shape for `list(x)` / similar, and
        // the result type is the class being called regardless of whether
        // `x` is a heaptype (`FrameLocalsProxy` is one).
        //
        // `complexobject.py descr__new__` of the exact `complex` type returns
        // an exact `complex` (or the same exact complex it was given).
        // `objectobject.py descr__init__` is then a no-op: `__init__` is
        // `object`'s and `__new__` is not.  A heaptype argument (`Idx()`)
        // does not change that result type, so `descr_call` still enters
        // `__new__`.
        if builtin_init.is_none()
            && unsafe { pyre_object::typeobject::w_type_is_heaptype(w_class) }
            && !exact_complex_type(callable)
        {
            return None;
        }
    }
    Some((tp_new, builtin_init))
}

/// [`lookup_instance_dunder_call`] narrowed to an app-level `__call__` the
/// walker can inline.  A `__call__` that is not a plain inlinable function — a
/// `classmethod`, another callable object — has no body to walk and declines
/// here, exactly as `resolve_inlinable_callee` declines a callee that is
/// neither `Function` nor a PyCode-carrying `BuiltinFunction`; a builtin
/// gateway `__call__` is `try_walker_inline_builtin_call`'s.
///
/// # Safety
/// `callable` must be a valid object.
unsafe fn resolve_instance_dunder_call(
    callable: pyre_object::PyObjectRef,
) -> Option<(
    pyre_object::PyObjectRef,
    pyre_object::PyObjectRef,
    u64,
    *const (),
    usize,
    bool,
)> {
    let (method, w_class, version_tag) = unsafe { lookup_instance_dunder_call(callable) }?;
    let (w_code, nparams, has_closure) = unsafe { resolve_inlinable_callee(method) }?;
    Some((method, w_class, version_tag, w_code, nparams, has_closure))
}

/// The `attr_cell` half of [`ExceptionInlineReceiverGuard`] for a callee the
/// walker resolved with `space.lookup(w_class, name)`.
///
/// `lookup` unwraps a `MutableCell`, so the resolved callee alone does not say
/// whether the namespace entry can still change under the pinned
/// `_version_tag`.  Ask the raw entry instead: an `ObjectMutableCell` is one
/// `write_cell` updates in place, which moves no tag, so an inline -- an
/// identity decision, the body is baked -- owes the getfield and the promote
/// [`walker_promote_object_mutable_cell`] emits.
///
/// # Safety
/// `w_class` must be a live type object.
/// A name whose raw class-namespace entry is an `ObjectMutableCell` cannot be
/// baked on `_version_tag` alone: `write_cell` updates an installed cell in
/// place and moves no tag.  An emit with nowhere to hang the promote declines
/// on this, the way `super_attr_suffix_lookup`'s own-dict read does.  The
/// convergence is the getfield-and-promote pair [`inline_attr_cell_guard`]
/// carries, not this decline.
///
/// # Safety
/// `w_type` must be a live type object.
unsafe fn type_attr_is_cell_backed(w_type: pyre_object::PyObjectRef, name: &str) -> bool {
    !unsafe {
        pyre_interpreter::baseobjspace::type_attr_object_cell(
            w_type,
            rustpython_wtf8::Wtf8::new(name),
        )
    }
    .is_null()
}

unsafe fn inline_attr_cell_guard(
    w_class: pyre_object::PyObjectRef,
    name: &str,
    method: pyre_object::PyObjectRef,
) -> Option<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)> {
    let cell = unsafe {
        pyre_interpreter::baseobjspace::type_attr_object_cell(
            w_class,
            rustpython_wtf8::Wtf8::new(name),
        )
    };
    (!cell.is_null()).then_some((cell, method))
}

/// `(receiver, concrete_receiver, w_class, version_tag, attr_cell)` for an
/// inlined call whose callee the receiver's class resolved.  `attr_cell` is
/// `(cell, payload)` when the namespace entry that produced the callee is an
/// `ObjectMutableCell`: the tag pin does not cover an in-place write to one, so
/// the payload is re-read and promoted rather than baked on the tag alone.
type ExceptionInlineReceiverGuard = (
    OpRef,
    pyre_object::PyObjectRef,
    pyre_object::PyObjectRef,
    u64,
    Option<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)>,
);

/// `(arg, concrete_arg, w_type)` for an inlined operator's non-receiver
/// argument: the operand op to guard, its record-time concrete object (for the
/// physical `ob_type`), and the Python `W_TypeObject` the reflected-op decline
/// was computed against (pinned via the live `w_class`).
type ArgClassGuard = (OpRef, pyre_object::PyObjectRef, pyre_object::PyObjectRef);

/// `residual_call` shape `iIRd>X` dispatcher — `_ir_*` arglist with
/// both an int-bank list and a ref-bank list before the descr. RPython
/// parity: `pyjitpl.py _opimpl_residual_call2` (`@arguments`
/// argspec `"box", "boxes2", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1`. The `boxes2`
/// argcode (`pyjitpl.py`) decodes two adjacent
/// count-prefixed lists into a single concatenated `argboxes` array
/// `[i_args..., r_args...]`. `_build_allboxes` (`pyjitpl.py`,
/// ported to [`build_allboxes`]) then permutes those to match
/// `descr.get_arg_types()` ABI ordering, so a callee whose `arg_types`
/// is `[REF, INT, REF, INT]` ends up with allboxes
/// `[funcbox, r_args[0], i_args[0], r_args[1], i_args[1]]`.
///
/// Operand layout `iIRd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 2B descr + 1B `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iRd_kind` via [`select_residual_call_opcode`],
/// and the same release-gil ([`direct_call_release_gil`]) +
/// loop-invariant heapcache ([`loopinvariant_lookup`] /
/// [`loopinvariant_now_known`]) sub-cases route through dedicated
/// helpers ahead of the selector.
///
/// Heapcache invalidation matches `iRd_kind`:
/// `invalidate_caches_varargs(call_opcode, ei, allboxes)` after every
/// recorded call op (`pyjitpl.py _record_helper_varargs`); the
/// release-gil helper invalidates with `CALL_MAY_FORCE_*` per
/// `pyjitpl.py`. `OS_NOT_IN_TRACE` is fail-loud-guarded up front
/// via [`do_not_in_trace_call_result`] (matches `iRd_kind`). Pre-call
/// vable IR bookkeeping (`vable_and_vrefs_before_residual_call`
/// IR-only portion at `pyjitpl.py`: FORCE_TOKEN +
/// SETFIELD_GC) is wired identically to `iRd_kind` via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`]; the runtime
/// heap mutations and the after-call helpers are handled by the
/// residual-call execution path.
///
/// Still missing relative to upstream — same set as `iRd_kind` and
/// blocked on the same infrastructure: `OS_JIT_FORCE_VIRTUAL`
/// short-circuit, `direct_libffi_call` / `direct_assembler_call`
/// specialization, KEEPALIVE for vablebox, `num_live`-aware
/// `capture_resumedata(after_residual_call=True)` on the guards. See
/// `dispatch_residual_call_iRd_kind`'s docstring for the per-item
/// blocking rationale.
/// Shared #57 gate for the BINARY_OP / COMPARE_OP int specialization.
/// Validates that both ref-list operands are concrete `W_IntObject` and
/// obtains the authentic boxed result via the same `execute_residual_call`
/// path the generic leg uses (so the concrete shadow holds the runtime
/// object incl. small-int caching / identity).
///
/// Returns `Some((lhs, rhs, lhs_val, rhs_val, boxed_result_i64))` when the
/// specialization may proceed, or `None` to fall through to the generic
/// `CallMayForce` record (non-int operands, non-const funcptr/args, or a
/// helper that raised).
/// Shared tail of the int/float specialization gates: pull the helper
/// `func_ptr` (the `allboxes[0]` constant) and the concrete arg words out
/// of `allboxes`, then run the helper concretely via
/// `execute_residual_call`.  Returns the authentic boxed-result pointer,
/// or `None` when an arg is non-concrete (vable sentinel / unstamped) or
/// the helper raises — both gates then defer to the generic record so the
/// Python-level `__op__` semantics are preserved.
fn walker_execute_may_force_boxed<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Option<i64> {
    match walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)? {
        Ok(result) if result != 0 => Some(result),
        _ => None,
    }
}

/// Execute the authentic boxed helper at record time, retaining its raising
/// result for specializations that trace through an exception arm.  The
/// ordinary result-only gate above deliberately maps a raise back to `None`;
/// this lower-level form lets a caller replace the opaque call with guarded
/// allocation IR while keeping the interpreter-produced exception as its
/// concrete shadow.
fn walker_execute_may_force_boxed_outcome<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Option<Result<i64, i64>> {
    if allboxes.is_empty() || !allboxes[0].is_constant() {
        return None;
    }
    let func_ptr = match ctx.trace_ctx.box_value(allboxes[0]) {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return None,
    };
    if func_ptr == 0 || majit_jitcode::codewriter::call::is_symbolic_fnaddr(func_ptr) {
        return None;
    }
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for &arg in &allboxes[1..] {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                if r == majit_ir::GcRef::NO_CONCRETE {
                    return None;
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => return None,
        };
        args.push(v);
    }
    Some(majit_metainterp::executor::execute_residual_call(
        call_descr, func_ptr, &args,
    ))
}

/// Resolve the concrete `PyObjectRef` carried by a Ref-bank operand's
/// recorded concrete, or `None` when it is the vable sentinel / null.
fn walker_concrete_ref_object<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opref: OpRef,
) -> Option<pyre_object::PyObjectRef> {
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
            let obj = r.as_usize() as pyre_object::PyObjectRef;
            // A bit-cast of a small int is a non-null `Ref` and not an
            // object. `is_method` loads `ob_type` from it.
            let aligned = (obj as usize) % std::mem::align_of::<usize>() == 0;
            if obj.is_null() || !aligned {
                None
            } else {
                Some(obj)
            }
        }
        _ => None,
    }
}

/// Resolve the walker's execution-context OpRef from the outer portal's
/// separately carried second red (`interp_jit.py reds = ['frame', 'ec']`).
///
/// `None` outside a production full-body walk (no materialized portal sym)
/// or when the portal frame OpRef is unset — the PUSH_EXC_INFO / POP_EXCEPT
/// exc-info lowering then declines to the residual (SAFE).
fn walker_ensure_execution_context<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Option<OpRef> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    let sym = unsafe { &*sym_ptr };
    (!sym.execution_context().is_none()).then(|| sym.execution_context())
}

/// Recover the portal EC red before the full-body walk records its first guard.
///
/// The portal `[frame, ec]` reds (`interp_jit.py reds = ['frame', 'ec']`)
/// are force-alived in every `-live-` op's R-bank, so every guard's resume
/// snapshot lists the dedicated EC color. Loop/function-entry syms seed
/// `InputArgRef(1)` and bridge setup decodes the failing guard's corresponding
/// inputarg into the same field — except where the guard's resume data named
/// no value at that color, which leaves the sym with none.
///
/// The walker's snapshot-capture path
/// (`walker_capture_snapshot_for_last_guard` → `collect_outer_active_boxes`)
/// runs AFTER the guard, so recording the recovery there would place the read
/// after the guard that references it (a use-before-def; the resume
/// position would also stamp onto the read rather than the guard, leaving
/// the guard with `resume_pos = -1`).  Recover here instead — at walk entry,
/// before any opcode is dispatched and thus before any guard. When the frame
/// itself is unset, this is a no-op: a frameless sym's first `ec` consumer
/// goes through `MIFrame::ensure_execution_context`, which records the same
/// read at an ordinary dispatch position. When the EC box is already seeded
/// the read is not repeated. Either way the live context is stamped onto
/// that box. `pyjitpl.py opimpl_record_quasiimmut_field` reads
/// `box.getref_base()`; the IR operand stays the red input.
pub(crate) fn seed_execution_context_for_walk<Sym: WalkSym>(
    sym: &mut Sym,
    trace_ctx: &mut TraceCtx,
) {
    if sym.execution_context().is_none() && !sym.frame().is_none() {
        let ec = crate::helpers::emit_current_execution_context(
            trace_ctx,
            "ExecutionContext::WalkEntry",
        );
        sym.set_execution_context(ec);
    }
    stamp_live_execution_context(trace_ctx, sym.execution_context());
}

/// Store the live `ExecutionContext` pointer as `ec_box`'s trace-time concrete.
///
/// `pyjitpl.py opimpl_record_quasiimmut_field` reads `box.getref_base()`.
/// The portal red is an input; this is that recording-time value. The IR
/// operand stays the input.
pub(crate) fn stamp_live_execution_context(trace_ctx: &mut TraceCtx, ec_box: OpRef) {
    if ec_box.is_none() {
        return;
    }
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return;
    }
    let concrete = Value::Ref(majit_ir::GcRef(ec as usize));
    trace_ctx.try_set_opref_concrete(ec_box, concrete);
}

/// Walker-native unbox of a boxed `W_IntObject` operand: `GUARD_CLASS`
/// (with the walker snapshot) when the operand's class is not yet known,
/// then the ctx-only `trace_unbox_int` getfield.  Mirrors
/// `trace_unbox_int_with_resume` (`state.rs`) with the guard emitted
/// walker-native (`record_guard` + `walker_capture_snapshot_for_last_guard`)
/// instead of via `MIFrame::generate_guard` — the full-body walk has
/// decomposed `MIFrame` into the reborrowed sym slices held by
/// `WalkContext`, so reconstructing an `MIFrame` here would alias them.
fn walker_unbox_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    int_type_addr: i64,
) -> Result<OpRef, DispatchError> {
    walker_unbox_int_typed(
        ctx,
        op_pc,
        obj,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )
}

/// Pin exact `w_class` first, then unbox. Exact class implies the layout
/// vtable, so [`walker_unbox_int_typed`] sees `is_class_known` and skips
/// the redundant `GuardClass`.
fn walker_unbox_int_exact<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
    expected_class: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    walker_guard_exact_w_class(ctx, op_pc, obj, expected_class)?;
    walker_unbox_int_typed(ctx, op_pc, obj, type_addr, intval_descr)
}

/// True when `obj` is an InputArg the concrete boundary GUARANTEES was
/// converted from a tagged immediate to a heap `W_IntObject` — i.e. a
/// frame LOCALS-region array-item InputArg (`raw() - vable_array_base <
/// nlocals`) of the active root/loop trace.
///
/// The boundary only converts the locals region: `untag_tagged_frame_locals`
/// (`eval.rs`) loops `0..frame.nlocals()`, and the record seed (`state.rs`
/// `init_symbolic` array-item pass) converts only `item_idx < nlocals`. The
/// `locals_cells_stack_w` array laid out behind those InputArgs also holds the
/// VALUE-STACK tail (`nlocals..live_prefix`), which the boundary does NOT
/// touch — so a value-stack-slot InputArg can still arrive tagged on a later
/// entry and MUST keep its defensive tag guard.
///
/// Returns `false` for:
///   * a scalar InputArg (`frame`/`ec`/`last_instr`/... at `raw() <
///     vable_array_base`) — never a boxed int operand, but classified
///     conservatively;
///   * a value-stack-slot array-item InputArg (`slot >= nlocals`);
///   * a bridge trace's InputArg (no `vable_array_base`), whose boxes come
///     from the guard-fail resume stream, not the converted portal frame;
///   * a non-InputArg (op result / const / temp).
///
/// Reads the active portal sym via `fbw_mode.snapshot_sym` (the same
/// source `reseed_vstack_from_shadow` uses for `nlocals`); `false` when the
/// pointer is null (no materialized portal, as in tests), which
/// keeps the guard (correct-but-conservative).
fn walker_inputarg_is_converted_local<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    obj: OpRef,
) -> bool {
    if !obj.is_input_arg() {
        return false;
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return false;
    }
    // SAFETY: pointer live for the full-body walk; read-only nlocals /
    // vable_array_base.
    let (nlocals, base) = unsafe { ((*sym_ptr).nlocals(), (*sym_ptr).vable_array_base()) };
    let Some(base) = base else {
        return false;
    };
    let raw = obj.raw();
    raw >= base && ((raw - base) as usize) < nlocals
}

/// [`walker_unbox_int`] with an explicit `intval` descr so a `bool` operand
/// can guard its own `&BOOL_TYPE` and read through `bool_intval_descr`
/// (`bool` shares `W_IntObject`'s `intval` field).
fn walker_unbox_int_typed<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> Result<OpRef, DispatchError> {
    // A known-class operand is provably a heap `W_IntObject` — a prior
    // `GuardClass` derefed its `ob_type`, impossible on a tagged immediate — so
    // it needs neither the tag test nor a repeated `GuardClass`.  Skipping the
    // tag block for it is also what keeps a JIT-made heap box (e.g. a `wrapint`
    // result) from emitting a `CastPtrToInt` lowbit test that would force the
    // box out of virtuality in the loop.
    if pyre_object::tagged_int::CAN_BE_TAGGED && !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        if let Some(o) = walker_concrete_ref_object(ctx, obj) {
            if pyre_object::tagged_int::is_tagged_int(o) {
                let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, obj, true);
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[lowbit])?;
                return Ok(crate::helpers::emit_untag_int(
                    ctx.trace_ctx,
                    obj,
                    pyre_object::tagged_int::untag_int(o),
                ));
            } else if !walker_inputarg_is_converted_local(ctx, obj) {
                // Emit the defensive `GuardFalse(lowbit)` for every heap-concrete
                // operand EXCEPT a frame LOCALS-region InputArg (handled below):
                //   * a NON-InputArg (a residual-call result box) can still
                //     receive a tagged immediate on a future arrival;
                //   * a value-stack-slot InputArg (`slot >= nlocals`) and a
                //     bridge InputArg (no vable layout) are NOT covered by the
                //     concrete boundary, which converts only the LOCALS region
                //     (`untag_tagged_frame_locals` loops `0..frame.nlocals()`;
                //     the record seed converts `item_idx < nlocals`), so a later
                //     entry could deliver a tagged immediate to that slot.
                // With the guard the boxed leg then deopts (not faults) on a
                // tagged arrival.
                let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, obj, false);
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[lowbit])?;
            }
            // A heap-concrete LOCALS-region InputArg is an entry/loop-carried
            // frame local that the concrete boundary
            // (`untag_tagged_frame_locals` at compiled-loop entry + the
            // trace-record seed) already converted to a heap `W_IntObject`, and
            // re-converts on every subsequent entry, so no tagged immediate ever
            // reaches it. Emit NO tag test for it: the `CastPtrToInt`+`IntAnd`+
            // `GuardFalse` does not forward through the loop-close `SetfieldGc`,
            // so it would survive the unroll and pin a per-iteration rebox in the
            // steady loop. Skipping it yields the flag-false `GuardClass`+
            // immutable `GetfieldGc` shape (raw carry).
        }
    }
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let already_this_class = obj.is_constant()
            && walker_concrete_ref_object(ctx, obj).is_some_and(|o| {
                !o.is_null()
                    && !(pyre_object::tagged_int::CAN_BE_TAGGED
                        && pyre_object::tagged_int::is_tagged_int(o))
                    && std::ptr::eq(
                        unsafe { (*o).ob_type },
                        type_addr as *const pyre_object::PyType,
                    )
            });
        if already_this_class {
            ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
        } else {
            let type_const = ctx.trace_ctx.const_int(type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
            ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
        }
    }
    Ok(crate::trace_unbox_int(
        ctx.trace_ctx,
        obj,
        type_addr,
        intval_descr,
    ))
}

/// Walker-native unbox of a fits-int `W_LongObject` operand into a raw i64 —
/// the long analogue of [`walker_unbox_int`], emitting the walker
/// snapshot guards.  `is_plain_int1` accepts a `W_LongObject` whose BigInt
/// fits a machine word (`type(w_obj) is W_LongObject and w_obj._fits_int()`,
/// listobject.py), and `plain_int_w` unwraps it via `_int_w`:
///
/// 1. `GUARD_CLASS(obj, LONG_TYPE)` when the class is not already known.
/// 2. `residual_call(jit_w_long_fits_int, obj)` + `GUARD_TRUE` — a later
///    execution may see the BigInt grown out of i64 range, so re-probe and
///    deopt to the residual on a non-fitting arrival.
/// 3. `residual_call(jit_w_long_toint, obj)` — `rbigint.toint()`, statically
///    non-raising after the fits guard.
///
/// A `W_LongObject` is a distinct 8-aligned heap struct and is never a tagged
/// immediate, so the `GUARD_CLASS` deref needs no lowbit pre-test.
fn walker_unbox_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    long_type_addr: i64,
) -> Result<OpRef, DispatchError> {
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(long_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    }
    let obj_concrete = walker_concrete_ref_object(ctx, obj);
    let fits_fn = pyre_object::longobject::jit_w_long_fits_int as *const ();
    let fits = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        fits_fn,
        &[obj],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    if let Some(o) = obj_concrete {
        let fits_concrete = pyre_object::longobject::jit_w_long_fits_int(o);
        ctx.trace_ctx
            .set_opref_concrete(fits, majit_ir::Value::Int(fits_concrete));
    }
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[fits])?;
    let toint_fn = pyre_object::longobject::jit_w_long_toint as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        toint_fn,
        &[obj],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    if let Some(o) = obj_concrete {
        let v = pyre_object::longobject::jit_w_long_toint(o);
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(v));
    }
    Ok(raw)
}

/// Walker-native unbox of a boxed `W_FloatObject` operand: the float
/// analogue of [`walker_unbox_int`].  `GUARD_CLASS` (with the walker
/// snapshot) when the operand's class is not yet known, then the ctx-only
/// `trace_unbox_float` getfield (its own `is_class_known` check is then a
/// no-op because `class_now_known` was just set).
fn walker_unbox_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    float_type_addr: i64,
) -> Result<OpRef, DispatchError> {
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(float_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    }
    Ok(crate::trace_unbox_float(
        ctx.trace_ctx,
        obj,
        float_type_addr,
        crate::descr::float_floatval_descr(),
    ))
}

/// floatobject.py `int_between(-1, i2 >> 48, 1)` — true while a double
/// represents `value` exactly, doubles carrying at least 48 bits of precision.
/// A wider int must be compared through its bigint, not through a rounded
/// double.
pub(crate) fn int_is_exact_as_float(value: i64) -> bool {
    (-1..1).contains(&(value >> 48))
}

/// Emit [`int_is_exact_as_float`] as a trace precondition, lowered exactly as
/// `int_between` is (`INT_SUB` + `UINT_LT`), so a re-entered trace carrying a
/// wider int bails out to the exact comparison instead of comparing rounded
/// doubles.
fn walker_guard_int_exact_as_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw_int: OpRef,
    value: i64,
) -> Result<(), DispatchError> {
    let shift = ctx.trace_ctx.const_int(48);
    let top = ctx
        .trace_ctx
        .record_op(OpCode::IntRshift, &[raw_int, shift]);
    ctx.trace_ctx
        .set_opref_concrete(top, majit_ir::Value::Int(value >> 48));
    let low = ctx.trace_ctx.const_int(-1);
    let offset = ctx.trace_ctx.record_op(OpCode::IntSub, &[top, low]);
    ctx.trace_ctx
        .set_opref_concrete(offset, majit_ir::Value::Int((value >> 48) + 1));
    let width = ctx.trace_ctx.const_int(2);
    let fits = ctx.trace_ctx.record_op(OpCode::UintLt, &[offset, width]);
    ctx.trace_ctx
        .set_opref_concrete(fits, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[fits])
}

/// Coerce a boxed operand to a raw `f64` OpRef for a float op: float →
/// `walker_unbox_float`; int → `walker_unbox_int` then `cast_int_to_float`
/// (`space.float_w` dispatches int through int2float).  Stamps the result
/// with the already-known concrete `val` so downstream `box_value` sees it.
/// Shared by the float-binary and float-compare specializations.
///
/// `exact_int` additionally guards that an int operand converts losslessly.
/// A float arithmetic operand legitimately rounds (`space.float_w`), but a
/// comparison is decided against the int's exact value (`_compare`), so only
/// the compare specialization sets it.
///
/// The int arm pins `w_class` here rather than leaving it to the caller.  The
/// unbox proves `ob_type`, which an `int` subclass shares with `int`, while
/// every gate that admits one of these folds read `w_class`; without the pin a
/// trace recorded on an exact int keeps answering for a subclass whose
/// `__float__` returns something else.  Making it part of the coercion is what
/// keeps the obligation from being restated — and forgotten — per arm.
/// The float arm carries no pin: reading a `float` subclass's payload is what
/// `try_get_double` does too, so the coerced value is the one the builtin would
/// have seen.  An operation that instead *dispatches* on the operand's class
/// (`space.add`, `space.lt`) needs the float arm pinned as well and takes
/// [`walker_coerce_dispatching_operand_to_float`].
fn walker_coerce_operand_to_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    is_int: bool,
    val: f64,
    exact_int: bool,
) -> Result<OpRef, DispatchError> {
    let raw = if is_int {
        // bool shares int's `intval`; guard its own &BOOL_TYPE before the cast.
        let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_obj);
        let raw_int = walker_unbox_int_typed(ctx, op_pc, obj, type_addr, descr)?;
        if exact_int {
            let value = unsafe { pyre_object::w_int_get_value(concrete_obj) };
            walker_guard_int_exact_as_float(ctx, op_pc, raw_int, value)?;
        }
        ctx.trace_ctx.record_op(OpCode::CastIntToFloat, &[raw_int])
    } else {
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        walker_unbox_float(ctx, op_pc, obj, float_type_addr)?
    };
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(val));
    if is_int {
        walker_guard_exact_w_class(ctx, op_pc, obj, walker_numeric_builtin_class(concrete_obj))?;
    }
    Ok(raw)
}

/// [`walker_coerce_operand_to_float`] for an operation that dispatches on the
/// operand's Python-level class instead of only reading its payload.  A `float`
/// subclass overriding `__add__` / `__lt__` reaches a different implementation
/// than the one recorded, and the unbox proves only `ob_type`, so pin `w_class`
/// on that arm too.
fn walker_coerce_dispatching_operand_to_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    is_int: bool,
    val: f64,
    exact_int: bool,
) -> Result<OpRef, DispatchError> {
    let raw =
        walker_coerce_operand_to_float(ctx, op_pc, obj, concrete_obj, is_int, val, exact_int)?;
    if !is_int {
        walker_guard_exact_w_class(ctx, op_pc, obj, walker_numeric_builtin_class(concrete_obj))?;
    }
    Ok(raw)
}

/// Emit a walker-native guard (`record_guard` + the walker snapshot for
/// the just-recorded guard).  Mirrors `MIFrame::generate_guard` for the
/// full-body walk.
///
/// A guard whose subject is a constant is not recorded, matching
/// `pyjitpl.py generate_guard`'s `if isinstance(box, Const): return`: the
/// property the guard would test is already decided by the constant, so the
/// op is dead weight the trace still pays for against `trace_limit`.  The
/// first argument of a data guard is its subject; a control-flow guard
/// (`GUARD_NOT_INVALIDATED`, ...) passes none and is always recorded, which is
/// the `box=None` arm of the same test.
fn walker_emit_guard_with_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
) -> Result<(), DispatchError> {
    let subject = args.first().copied();
    if subject.is_some_and(|arg| arg.is_constant()) {
        return Ok(());
    }
    // `pyjitpl.py _establish_nullity`: a box's nullity is immutable, so the
    // first guard on it answers for every later site, and `heapcache.py
    // is_nullity_known` is what the second site asks.  A fold reaches a
    // receiver the surrounding opcode has already proved non-null often
    // enough that the repeat is a whole operand's worth of `trace_limit`.
    let nullity = match opcode {
        OpCode::GuardNonnull => Some(true),
        OpCode::GuardIsnull => Some(false),
        _ => None,
    };
    if let (Some(subject), Some(is_nonnull)) = (subject, nullity)
        && ctx.trace_ctx.heapcache_nullity_known(subject) == Some(is_nonnull)
    {
        return Ok(());
    }
    stamp_guard_value_concrete(ctx.trace_ctx, opcode, args);
    ctx.trace_ctx.record_guard(opcode, args, 0);
    if let (Some(subject), Some(is_nonnull)) = (subject, nullity) {
        ctx.trace_ctx.heap_cache_mut().nullity_now_known(subject);
    }
    walker_capture_snapshot_for_last_guard(ctx, op_pc)
}

/// A recording-path `GUARD_VALUE(box, const)` has already observed equality.
/// Mirror `FrontendOp.value` by attaching that constant to the guarded box so
/// bridge recipe construction sees the same fact as the runtime guard.
fn stamp_guard_value_concrete(trace_ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
    if opcode != OpCode::GuardValue || args.len() < 2 {
        return;
    }
    if let Some(expected) = trace_ctx.concrete_of_opref(args[1]) {
        trace_ctx.set_opref_concrete(args[0], expected);
    }
}

/// [`walker_emit_guard_with_snapshot`] named for a hand-fold's guards.
///
/// A fold reaches this from both the per-opcode arm entry, where the outer
/// active-box set is `ctx.outer_active_boxes`, and the main full-body walk,
/// where that set is empty and the snapshot helper re-derives `py_pc` from
/// `op_pc` and recomputes the active boxes per guard.  Both are the shared
/// emitter's business, so the two spellings record the same thing.
fn walker_emit_fold_guard_with_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
) -> Result<(), DispatchError> {
    walker_emit_guard_with_snapshot(ctx, op_pc, opcode, args)
}

/// [`walker_emit_fold_guard_with_snapshot`] for a fold inside a Python opcode
/// that cannot be re-entered at its block head.
///
/// A guard ordinarily resumes at the block-head `-live-` of its Python
/// opcode.  For an exception-table target that marker precedes the handler
/// landing's `last_exception` / `last_exc_value` pair, and a guard failure
/// carries no exception for those to read (`bhimpl_last_exc_value` asserts
/// non-null).  Carry the walk's own `-live-` BEFORE anchor instead: it sits
/// after the landing, on the operations the opcode has left to run, and is a
/// decodable startpoint by construction (`pyjitpl.py` reads the normal guard
/// resume at `self.pc - SIZE_LIVE_OP`).
///
/// `false` when the walk has no stepped anchor to carry, leaving the caller
/// to decline its fold.
fn walker_emit_anchored_fold_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
) -> Result<bool, DispatchError> {
    let subject = args.first().copied();
    // `pyjitpl.py _establish_nullity` / `heapcache.py is_nullity_known`, as
    // in [`walker_emit_guard_with_snapshot`]: an answered question needs no
    // guard, and the fold stands either way.
    let nullity = match opcode {
        OpCode::GuardNonnull => Some(true),
        OpCode::GuardIsnull => Some(false),
        _ => None,
    };
    if subject.is_some_and(|arg| arg.is_constant()) {
        return Ok(true);
    }
    if let (Some(subject), Some(is_nonnull)) = (subject, nullity)
        && ctx.trace_ctx.heapcache_nullity_known(subject) == Some(is_nonnull)
    {
        return Ok(true);
    }
    // The anchor answers only when the walk stepped past the block head: the
    // `-live-` it names must be a LATER one, or it is the block head itself
    // (or a stale word from a walk that entered mid-opcode) and carrying it
    // resumes on the landing this exists to skip.  An inline sub-walk's
    // `op_pc` is a callee coordinate the outer jitcode's tables do not hold,
    // and its capture resumes at the CALL site instead, so the anchor is not
    // this guard's to carry there.
    let anchor = ctx.live_before_jit_pc;
    if anchor == usize::MAX || ctx.fbw_mode.inline_subwalk {
        return Ok(false);
    }
    let block_head = {
        let sym = ctx.fbw_mode.snapshot_sym;
        if sym.is_null() {
            return Ok(false);
        }
        let jitcode = unsafe { (&*sym).jitcode() };
        if jitcode.is_null() {
            return Ok(false);
        }
        unsafe { (&*jitcode).payload.resume_marker_for_jitcode_pc(op_pc) }
    };
    let Some(block_head) = block_head else {
        return Ok(false);
    };
    if anchor <= block_head {
        return Ok(false);
    }
    stamp_guard_value_concrete(ctx.trace_ctx, opcode, args);
    ctx.trace_ctx.record_guard(opcode, args, 0);
    if let (Some(subject), Some(is_nonnull)) = (subject, nullity) {
        ctx.trace_ctx.heap_cache_mut().nullity_now_known(subject);
    }
    walker_capture_snapshot_for_last_guard_scoped(
        ctx,
        op_pc,
        GuardCaptureScope {
            carried_resume_jit_pc: Some(anchor),
            ..GuardCaptureScope::default()
        },
    )?;
    Ok(true)
}

fn walker_flush_guard_not_invalidated<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.pending_guard_not_invalidated_pc().is_some() {
        ctx.trace_ctx.set_pending_guard_not_invalidated(None);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNotInvalidated, &[])?;
    }
    Ok(())
}

/// Record `int_eq(raw, const k)` and stamp its already-known concrete
/// truth.  Used to build the div/mod precondition guards walker-native.
fn walker_int_eq_const<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    raw: OpRef,
    k: i64,
    concrete_truth: i64,
) -> OpRef {
    let k_const = ctx.trace_ctx.const_int(k);
    let r = ctx.trace_ctx.record_op(OpCode::IntEq, &[raw, k_const]);
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_truth));
    r
}

/// Record a float comparison with its already-known concrete truth, then
/// pin the observed direction with `GuardTrue`/`GuardFalse` (walker
/// snapshot at `op_pc`).
fn walker_float_cmp_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    args: &[OpRef],
    truth: bool,
) -> Result<(), DispatchError> {
    let c = ctx.trace_ctx.record_op(opcode, args);
    ctx.trace_ctx
        .set_opref_concrete(c, majit_ir::Value::Int(truth as i64));
    let guard = if truth {
        OpCode::GuardTrue
    } else {
        OpCode::GuardFalse
    };
    walker_emit_guard_with_snapshot(ctx, op_pc, guard, &[c])
}

/// Guard the non-NaN precondition of raw-f64 storage. A class guard alone
/// admits NaNs; `raw != raw` sends them back to the boxed interpreter path.
fn walker_guard_float_not_nan<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
) -> Result<(), DispatchError> {
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatNe, &[raw, raw], false)
}

/// Box a raw i64 int result. Under `CAN_BE_TAGGED`, when the recorded
/// concrete `value` fits the tagged range, emit an immediate
/// `(value<<1)|1` (rtagged.py ll_int_to_unboxed) guarded by
/// `IntAddOvf(raw,raw)` + `GuardNoOverflow` — the doubling overflow IS the
/// fits check, and a runtime value that does not fit deopts instead of
/// producing a wrong box. Otherwise (or flag off) fall back to the heap
/// `wrapint` box. The caller stamps the Ref concrete.
fn walker_box_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
    value: i64,
) -> Result<OpRef, DispatchError> {
    let _ = (op_pc, value);
    // intobject.py `wrapint` (withprebuiltint=False): every boxing
    // allocates a fresh `W_IntObject`. Interning a constant raw here
    // is WITHPREBUILTINT for the residual `box_int_fn` path. Only
    // `LOAD_SMALL_INT` / `getconstant_w` intern, via the codewriter
    // frontend.
    Ok(crate::state::wrapint(ctx.trace_ctx, raw))
}

/// Float twin of [`walker_box_int`]: records the `NEW_WITH_VTABLE +
/// SETFIELD_GC` pair for a `W_FloatObject` so the box stays virtualizable.
/// There is no tagged float representation, so the boxed concrete is just the
/// heap pointer and no counterpart to [`box_int_concrete`] is needed.
fn walker_box_float<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>, raw: OpRef) -> OpRef {
    crate::state::wrapfloat(ctx.trace_ctx, raw)
}

/// Concrete counterpart to [`walker_box_int`]. The op is now a heap
/// `NewWithVtable`, so the stamped concrete must ALSO be a heap ptr. When the
/// residual result is tagged (flag-true), materialize a real heap
/// `W_IntObject` via `w_int_new_unique` for the concrete so
/// `op(heap) == concrete(heap)`.
fn box_int_concrete(value: i64, runtime_ptr: i64) -> majit_ir::Value {
    let ptr = if pyre_object::tagged_int::CAN_BE_TAGGED
        && pyre_object::tagged_int::is_tagged_int(runtime_ptr as pyre_object::PyObjectRef)
    {
        pyre_object::intobject::w_int_new_unique(value) as i64
    } else {
        runtime_ptr
    };
    majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize))
}

/// Emit a walker-native `GUARD_CLASS(obj, type_addr)` (with the walker
/// snapshot) when `obj`'s class is not yet known, then record it as known.
/// The guard-only counterpart of [`walker_unbox_int_typed`] for operands
/// that are passed to a residual call by reference rather than unboxed.
fn walker_guard_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    // `heapcache.py is_class_known` is only `HF_KNOWN_CLASS`, not
    // `isinstance`.  `W_IntObject.intval` and `W_LongObject.value` sit at
    // the same offset, so skipping `GuardClass(LONG)` because some other
    // class is already known lets a box recorded as `INT` take the long
    // payload path: `jit_bigint_int_eq` then treats the intval as a
    // `*const BigInt`.  Skip only when the known class is this one, the
    // same compare `walker_guard_mapdict_instance_shape` already does.
    let known_cls = walker_concrete_ref_object(ctx, obj).map(|o| unsafe { (*o).ob_type as i64 });
    if !(ctx.trace_ctx.heap_cache().is_class_known(obj) && known_cls == Some(type_addr)) {
        // `GuardClass` reads `ob_type` off `obj` (rpython/jit/backend/x86/
        // assembler.py `_cmp_guard_class` derefs the pointer with no tag
        // test), so the frontend must not hand it a tagged immediate. `obj`
        // here is a concrete heap box at record time (callers gate on
        // `is_long`), but the trace is reused for operands that arrive tagged
        // on a later entry. Emit the low-bit `GuardFalse` first — mirroring
        // `walker_unbox_int_typed`'s boxed leg — so a tagged arrival deopts
        // instead of faulting on the class deref.
        if pyre_object::tagged_int::CAN_BE_TAGGED
            && walker_concrete_ref_object(ctx, obj)
                .is_some_and(|o| !pyre_object::tagged_int::is_tagged_int(o))
        {
            let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, obj, false);
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[lowbit])?;
        }
        let type_const = ctx.trace_ctx.const_int(type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    }
    Ok(())
}

/// Guard the fixed-layout mapdict-carrier representation, the receiver type's
/// live Python class, live version tag, and the exact map shape used by the
/// mapdict attribute folds.  The layout guard proves that a later receiver
/// still has the wide user layout before either mapdict field is read, and
/// the Python class guard which subclass it is. The promoted map identity then
/// pins its storage coordinates (mapdict.py).
fn walker_guard_mapdict_instance_shape<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
    version_tag: u64,
    map: pyre_interpreter::objspace::std::mapdict::MapRef,
) -> Result<(), DispatchError> {
    // `load/store_attr_*_fast_path` accepted this receiver only after
    // `has_mapdict_storage` proved its prefix. Preserve the concrete layout
    // tag here: claiming every carrier is `INSTANCE_TYPE` poisons the heap
    // cache for native-layout subclasses and lets later folds use unrelated
    // field descriptors on the same box.
    //
    // A cached class that DISAGREES with the layout tag is that poisoning,
    // and gating on `is_class_known` alone would then skip the guard and let
    // the storage coordinates below run against an unguarded layout. Compare
    // the recorded class instead and guard whenever it is absent or differs;
    // a contradictory pair is folded out by the optimizer's constant-class
    // handling, which discards the trace rather than reading a wild slot.
    let layout_type_addr = unsafe { (*concrete_obj).ob_type as i64 };
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(layout_type_addr);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    }

    // `typedef.py _getusercls` gives a builtin user subclass a generated payload
    // containing MapdictStorageMixin, and the GuardClass above already
    // separates it from the shorter builtin payload. Every Python subclass of
    // one builtin shares that generated class, so pin `w_class` before loading
    // `map`; a different subclass exits first.
    walker_guard_exact_w_class(ctx, op_pc, obj, w_type)?;

    // The instance map pins the storage layout, but class mutation can change
    // lookup precedence without changing that map. Pin the receiver type's
    // version tag so such a mutation revokes the loop before the folded
    // storage access is reused.
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    // guard_value(getfield_gc_i(obj, map), C_map): `jit.promote(self.map)`
    // (`mapdict.py`).  The map nodes are interned + immortal, so the
    // pointer is a stable identity guarded as an opaque word (object_map_descr
    // is Int-typed).
    //
    // The guard may only be elided when the map read is ALREADY a compile-time
    // constant — i.e. a prior promotion in this trace pinned it via
    // `replace_box`.  It must NOT be elided merely because `box_value(map_op)`
    // reports the concrete map: every traced getfield op carries its live value
    // (`opimpl_getfield_gc_i` -> `set_opref_concrete`, `history.py`
    // FrontendOp), so `box_value == map` holds for the very first read and
    // would drop the guard on the trace's entry.  A trace whose map guard is
    // dropped reads `storage[storageindex]` off any same-class receiver whose
    // map differs (GuardClass alone does not pin the layout), returning a wild
    // slot value.  Pin the map with `replace_box` after guarding so a later
    // fold on the same receiver correctly elides (matching the trait
    // `implement_guard_value`).
    let map_op = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, unsafe {
        crate::descr::mapdict_map_descr(concrete_obj)
    });
    if !map_op.is_constant() {
        let map_const = ctx.trace_ctx.const_int(map as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[map_op, map_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(map_op, map_const);
    }
    Ok(())
}

/// True when the payload `ob_type` already determines the Python-visible
/// class, so pinning `w_class` against `w_type` is a tautology.
///
/// pyre carries the Python-visible class in `w_class` beside the payload
/// `ob_type` that `GuardClass` reads, which is the whole reason a second pin
/// exists: a builtin subclass shares the layout and retags `w_class`.  Three
/// properties together remove that freedom.  `w_type` is the canonical class
/// of the payload; it refuses to be a base class, so no subclass can share the
/// layout; and it is not a heap type, so `__class__` assignment cannot retag an
/// instance onto or off it — `objectobject.py descr_set___class__` requires
/// both ends to be mutable heap types.  `frame`, `code` and `traceback` are all
/// three, which is what makes the pin removable on a traceback walk.
///
/// This is the argument [`walker_numeric_builtin_class`] already makes for
/// `bool`, stated for a receiver whose class is not known at codegen time.
///
/// A builtin with a `_getusercls` class (`PyType.user_subclass`) needs none of
/// the three: its subclass instances carry that class as their typeptr, so the
/// builtin's own typeptr already names the canonical class.
///
/// # Safety
/// `ob_type` must be null or a live `PyType`, and `w_type` a live type object.
unsafe fn walker_payload_determines_w_class(
    ob_type: *const pyre_object::PyType,
    w_type: pyre_object::PyObjectRef,
) -> bool {
    if ob_type.is_null() {
        return false;
    }
    unsafe {
        std::ptr::eq(pyre_object::pyobject::get_instantiate(&*ob_type), w_type)
            && (!(*ob_type).user_subclass.is_null()
                || (!pyre_object::typeobject::w_type_get_acceptable_as_base_class(w_type)
                    && !pyre_object::typeobject::w_type_is_cpython_heaptype(w_type)))
    }
}

/// Pin every branch input preceding a raw typed exception-slot arm.
/// `GuardClass` fixes the kind-specific `W_BaseException` layout and kind tag;
/// `GuardValue(getfield(w_class))` distinguishes heap subclasses sharing that
/// layout; and the class `version_tag` guard pins the preceding type-dict miss.
/// Exception `w_dict` lookup follows these arms in `baseobjspace.rs`, so it is
/// intentionally not part of the guard set.  This is the same promoted-class
/// lookup shape used by the mapdict folds (`mapdict.py`).
fn walker_guard_exception_attr_slot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
    version_tag: u64,
) -> Result<(), DispatchError> {
    let ob_type = unsafe { (*concrete_obj).ob_type };
    let physical_type = ob_type as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(physical_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    }
    // The `GuardClass` above pins the payload, and for a receiver whose payload
    // determines its class there is nothing left for the `w_class` pin to
    // exclude.  It is not free: the pin does not dedupe the way `GuardClass`
    // does through `is_class_known`, so a traceback walk pays a load and a
    // guard on every hop of every node.
    if !unsafe { walker_payload_determines_w_class(ob_type, w_type) } {
        let live_w_class =
            crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, crate::descr::w_class_descr());
        let w_class_const = ctx.trace_ctx.const_ref(w_type as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[live_w_class, w_class_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(live_w_class, w_class_const);
    }

    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_pin_type_version_tag(ctx, op_pc, type_const)?;
    Ok(())
}

fn walker_load_name_from_code(w_code_ptr: usize, name_idx: usize) -> Option<String> {
    if w_code_ptr == 0 {
        return None;
    }
    unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return None;
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        pyre_interpreter::pyframe::load_name_from_code(code, name_idx).map(ToString::to_string)
    }
}

fn walker_record_getfield_gc_i_uncached<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj: OpRef,
    descr: DescrRef,
) -> OpRef {
    // Plain GETFIELD_GC_I regardless of purity; OptHeap re-derives
    // purity from the descr. There is no pure getfield opnum.
    ctx.trace_ctx
        .record_op_with_descr(OpCode::GetfieldGcI, &[obj], descr)
}

/// Pin a type's method-cache version without reading it, per
/// `typeobject.py:177 _immutable_fields_ = ['_version_tag?']`.
///
/// `promote(self.version_tag())` (typeobject.py) needs the tag green so the
/// `_pure_lookup_where_with_method_cache` fold can run. On a quasi-immutable
/// field that costs a `QUASIIMMUT_FIELD` marker plus one
/// `GUARD_NOT_INVALIDATED` per trace, not a load and a `GUARD_VALUE` per read:
/// `mutated()` revokes the loop instead of the trace re-proving the tag. The
/// difference is load-bearing inside a body with un-inlined calls, since a
/// residual `CALL_MAY_FORCE` flushes a mutable field's cache but cannot flush a
/// quasi-immutable one.
///
/// `w_type_const` must already be pinned to the receiver's type — the caller
/// guards `w_class` first, so the marker attaches to a constant the optimizer
/// can hand to `register_quasi_immutable_deps` as the watched object.
fn walker_pin_type_version_tag<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    w_type_const: OpRef,
) -> Result<(), DispatchError> {
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        w_type_const,
        crate::descr::type_version_tag_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

/// Make a class-namespace `ObjectMutableCell`'s payload green again, so a fold
/// that bakes the resolved method survives an in-place rebind.
///
/// [`walker_pin_type_version_tag`] covers a REPLACING store: `write_cell`
/// builds a fresh cell and `mutated()` revokes the tag here and in every
/// subclass.  It covers nothing about an in-place write, which returns `None`
/// from `write_cell` and moves no tag at all -- so the payload has to be read.
/// The pin is what holds the cell pointer still, which is why the load is a
/// `getfield` off a `ConstPtr` and not a second lookup, and the `guard_value`
/// is what hands the fold back the constant it resolved on.
fn walker_promote_object_mutable_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    cell: pyre_object::PyObjectRef,
    expected: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    let cell_const = ctx.trace_ctx.const_ref(cell as i64);
    let value = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        cell_const,
        crate::descr::object_mutable_cell_value_descr(),
    );
    let expected_const = ctx.trace_ctx.const_ref(expected as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[value, expected_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(value, expected_const);
    Ok(())
}

/// The `descriptor.py:175 _immutable_fields_ = ["w_fget?", "w_fset?",
/// "w_fdel?"]` / `function.py:673 ['w_function?']` twin of
/// [`walker_pin_type_version_tag`]: pin the slot a descriptor fold is about to
/// bake, whether that is a property's accessor or a `staticmethod` /
/// `classmethod`'s wrapped callable.
///
/// The receiver pins the folds already hold — class, `w_class`, and the type's
/// `_version_tag?` — make the DESCRIPTOR a compile-time answer, and stop
/// there: `property.__init__` on an installed descriptor replaces `fget`/
/// `fset` in place and bumps no type's version, so without this the trace kept
/// calling the previous getter.  Upstream covers exactly that gap with the `?`
/// on the slots themselves.
///
/// A marker, never a load: the descriptor is a baked `ConstPtr`, and reading a
/// field through one is the hazard `try_walker_inline_resolved_user_call`'s
/// `guards_the_callee_function` gate exists to avoid.  `record_quasiimmut_field`
/// dereferences the owner only at record and compile time, and a property is
/// allocated non-moving, so both reads see the object where the constant says
/// it is.
fn walker_pin_descriptor_slot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    w_descr: pyre_object::PyObjectRef,
    field: majit_ir::DescrRef,
) -> Result<(), DispatchError> {
    // `rewrite.py` `_gcref_index` / `quasiimmut.py`: put the descriptor
    // in the ConstPtr gcrefs table so a nursery move updates the constant.
    // There is no `can_move` gate on this pin (`rpython/jit` uses
    // `can_move` only in backend `convert_to_imm`).
    let descr_const = ctx.trace_ctx.const_ref(w_descr as i64);
    crate::state::record_quasiimmut_field(ctx.trace_ctx, descr_const, field);
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

/// Pin whatever the class-MRO value's `space.get` arm depends on beyond the
/// receiver type's version tag, as [`pyre_interpreter::type_attr_value_fast_path`]
/// reported it.
fn walker_pin_type_attr_binding<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    binding: pyre_interpreter::TypeAttrBinding,
) -> Result<(), DispatchError> {
    match binding {
        // Neither arm reads a field of the value, so the version tag that
        // covers rebinding the name is already the whole precondition.
        pyre_interpreter::TypeAttrBinding::Unbound
        | pyre_interpreter::TypeAttrBinding::Function => Ok(()),
        pyre_interpreter::TypeAttrBinding::UnboundHeap { w_value } => {
            // `descr_getattribute` → `space.get` looks up `__get__` on
            // `type(w_value)`.  The payload type's `_version_tag?` is what
            // `mutated` moves when that class gains `__get__`.
            // `descr_set___class__` writes `w_class?` through
            // `notify_w_class_mutated_then` and leaves every type version
            // alone, so the class word is pinned on the value itself.
            // `type_attr_value_fast_path` only reports this arm when the
            // payload type exists and its version tag is nonzero.
            let Some(value_type) = pyre_interpreter::typedef::r#type(w_value) else {
                return Ok(());
            };
            let type_const = ctx.trace_ctx.const_ref(value_type.as_ptr() as i64);
            crate::state::record_quasiimmut_field(
                ctx.trace_ctx,
                type_const,
                crate::descr::type_version_tag_descr(),
            );
            let value_const = ctx.trace_ctx.const_ref(w_value as i64);
            crate::state::record_quasiimmut_field(
                ctx.trace_ctx,
                value_const,
                crate::descr::w_class_descr(),
            );
            walker_flush_guard_not_invalidated(ctx, op_pc)
        }
        pyre_interpreter::TypeAttrBinding::StaticMethod { w_wrapper } => {
            walker_pin_descriptor_slot(
                ctx,
                op_pc,
                w_wrapper,
                crate::descr::staticmethod_w_function_quasi_descr(),
            )
        }
    }
}

/// The `function.py:47 _immutable_fields_ = ['code?', 'w_func_globals?',
/// 'closure?[*]', 'defs_w?[*]']` twin of [`walker_pin_descriptor_slot`]: pin
/// the callee's code slot when the inline lever cannot re-prove it per
/// iteration.
///
/// The inline bakes `code` by choosing which callee jitcode to walk into, so
/// the value ends up spread across the inlined body rather than in one box a
/// `GUARD_VALUE` could re-check.  Where the caller pins the callee function
/// itself, that guard still runs and is the cheaper answer for a callee whose
/// identity changes every iteration; everywhere else — a constant callable, or
/// a specializer that dispatched on some other object — this marker is what
/// makes `f.__code__ = g.__code__` revoke the loop.
///
/// A marker, never a load: the owner is dereferenced only at record and
/// compile time, which is why this arm is clear of the baked-`ConstPtr` hazard
/// that keeps the guard arm off a constant callable.  Both reads resolve the
/// owner by raw address, so the caller refuses a callee the collector can
/// relocate.
fn walker_pin_function_code<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    callable: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    walker_pin_function_quasi_field(ctx, op_pc, callable, crate::descr::function_code_descr())
}

/// `function.py:34 _immutable_fields_ = ['code?', 'w_func_globals?', ...]`
/// marker on one `?` slot of a baked callee.
fn walker_pin_function_quasi_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    callable: pyre_object::PyObjectRef,
    descr: majit_ir::DescrRef,
) -> Result<(), DispatchError> {
    let callable_const = ctx.trace_ctx.const_ref(callable as i64);
    crate::state::record_quasiimmut_field(ctx.trace_ctx, callable_const, descr);
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

/// The `celldict.py _immutable_fields_ = ["version?"]` twin of
/// [`walker_pin_type_version_tag`]: pin the module namespace's strategy version
/// so the folds that bake a slot's stored cell (or the absence of a name) are
/// revoked by `mutated()` instead of re-reading the dict each iteration.
///
/// The marker attaches to the strategy box, not the dict object, because
/// `version` lives on the strategy — `getdictvalue_no_unwrapping`
/// (celldict.py) promotes `self` and then `self.version` for the same
/// reason. One marker covers every fold against one namespace, since they all
/// depend on that single field.
///
/// `Ok(false)` when the dict has no strategy box to pin, which declines the
/// fold rather than baking a constant nothing watches.
fn walker_pin_namespace_version<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    ns: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    let strategy = unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(ns) };
    if strategy.is_null() {
        return Ok(false);
    }
    let strategy_const = ctx.trace_ctx.const_ref(strategy as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        strategy_const,
        crate::descr::module_dict_version_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    Ok(true)
}

/// Pin mapdict's quasi-immutable fields with `const_ref`, never `const_int`:
/// `quasi_immut_descr`, `current_quasiimmut_field_value`, and optimizer
/// dependency collection require `Value::Ref`. The raw non-GC pointer cannot
/// reach the gcref table because `OptHeap` consumes `QUASIIMMUT_FIELD` before
/// the GC rewriter runs, exactly as `walker_pin_namespace_version` relies on.
///
/// The owner is the enum variant payload, never `MapNode`: the enum has no
/// `repr`, while the descr offsets are relative to `Terminator` and
/// `PlainAttribute`. `CachedAttributeHolder` is already standalone.
///
/// `walker_flush_guard_not_invalidated` drains a pending flag, so consecutive
/// markers at one fold site emit exactly one `GUARD_NOT_INVALIDATED`.
fn walker_pin_terminator_allow_unboxing<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    term: *const pyre_interpreter::objspace::std::mapdict::Terminator,
) -> Result<(), DispatchError> {
    if term.is_null() {
        return Ok(());
    }
    let owner = ctx.trace_ctx.const_ref(term as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::terminator_allow_unboxing_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

/// Pin `vm.py AuditHolder._immutable_fields_ = ['hooks_w?[:]']`, the read
/// `vm.py`'s `holder.hooks_w is None` early-out makes before `audit`
/// returns without calling anything.
///
/// A fold that elides an audit event has to depend on that read, or a later
/// `addaudithook` would silently stop being able to observe the event.  On a
/// quasi-immutable field the dependency costs a marker plus one
/// `GUARD_NOT_INVALIDATED` per trace, and `addaudithook` revokes the loop.
///
/// The caller passes the holder it has already resolved: there is none before
/// `sys` is registered, and a fold that cannot pin this field must decline
/// rather than elide an event nothing watches.
fn walker_pin_audit_hooks<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    holder: *const pyre_interpreter::module::sys::vm::AuditHolder,
) -> Result<(), DispatchError> {
    if holder.is_null() {
        return Ok(());
    }
    let owner = ctx.trace_ctx.const_ref(holder as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::audit_holder_hooks_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

fn walker_pin_holder_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    holder: *const pyre_interpreter::objspace::std::mapdict::CachedAttributeHolder,
) -> Result<(), DispatchError> {
    if holder.is_null() {
        return Ok(());
    }
    let owner = ctx.trace_ctx.const_ref(holder as i64);
    crate::state::record_quasiimmut_field(ctx.trace_ctx, owner, crate::descr::holder_attr_descr());
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

fn walker_pin_holder_typ<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    holder: *const pyre_interpreter::objspace::std::mapdict::CachedAttributeHolder,
) -> Result<(), DispatchError> {
    if holder.is_null() {
        return Ok(());
    }
    let owner = ctx.trace_ctx.const_ref(holder as i64);
    crate::state::record_quasiimmut_field(ctx.trace_ctx, owner, crate::descr::holder_typ_descr());
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

fn walker_record_getfield_gc_r_uncached<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj: OpRef,
    descr: DescrRef,
) -> OpRef {
    // Plain GETFIELD_GC_R regardless of purity; OptHeap re-derives
    // purity from the descr. There is no pure getfield opnum.
    ctx.trace_ctx
        .record_op_with_descr(OpCode::GetfieldGcR, &[obj], descr)
}

/// True only when this `LoadAttr` residual is immediately consumed by the
/// paired `load_method_self(obj, attr, code, name_idx)` residual emitted for
/// `LOAD_METHOD`.  Plain `LOAD_ATTR` of a function descriptor must keep the
/// normal bound-method semantics and cannot be rewritten to the raw function.
fn next_op_is_load_method_self_for_attr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
    attr_dst_reg: usize,
) -> bool {
    let Some(mut next) = crate::jitcode_runtime::decode_op_at(code, op.next_pc) else {
        return false;
    };
    while next.opname == "live"
        || next.opname.starts_with("setarrayitem_vable")
        || next.opname.starts_with("setfield_vable")
    {
        let Some(after_live) = crate::jitcode_runtime::decode_op_at(code, next.next_pc) else {
            return false;
        };
        next = after_live;
    }
    if next.key != "residual_call_ir_r/iIRd>r" {
        return false;
    }
    let i_len_pc = next.pc + 2;
    let Some(&i_len_byte) = code.get(i_len_pc) else {
        return false;
    };
    let i_width = 1 + i_len_byte as usize;
    let r_len_pc = next.pc + 1 + 1 + i_width;
    let Some(&r_len_byte) = code.get(r_len_pc) else {
        return false;
    };
    let r_len = r_len_byte as usize;
    if r_len < 2 {
        return false;
    }
    let Some(&attr_reg_byte) = code.get(r_len_pc + 1 + 1) else {
        return false;
    };
    if attr_reg_byte as usize != attr_dst_reg {
        return false;
    }
    let r_width = 1 + r_len;
    let descr_offset = 1 + i_width + r_width;
    let descr_index = decode_descr_index(code, &next, descr_offset);
    crate::jitcode_runtime::record_descr_demand(descr_index);
    ctx.descr_refs.at(descr_index).is_some_and(|descr| {
        descr.as_call_descr().is_some_and(|cd| {
            cd.get_extra_info().runtime_helper == majit_ir::RuntimeHelperKind::LoadMethodSelf
        })
    })
}

/// The canonical Python class of an exact builtin operand, or `None` when it
/// is a subclass instance or carries no `w_class` to pin.
///
/// A specialization that bypasses special-method lookup needs this at RECORD
/// time to pick the class object its replay guard compares against.  PyPy has
/// no equivalent step: `space.allocate_instance` gives an `int` subclass its
/// own RPython class, so one `guard_class` already separates it from
/// `W_IntObject`.  pyre shares the `ob_type` layout between the two and carries
/// the Python-visible class in `w_class`, so `GuardClass` alone leaves a
/// subclass free to reuse the trace and skip its `__floordiv__` / `__pow__` /
/// `__lt__` override.
///
/// `is_exact_builtin_instance` also accepts a NULL `w_class`; that operand
/// cannot be pinned by [`walker_guard_exact_w_class`], so it declines here
/// rather than emitting a guard that would fail on its own recorded operand.
///
/// # Safety
/// `obj` must be a non-null, untagged heap object.
unsafe fn walker_exact_builtin_class(
    obj: pyre_object::PyObjectRef,
) -> Option<pyre_object::PyObjectRef> {
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() {
            return None;
        }
        let canonical = pyre_object::pyobject::get_instantiate(&*(*obj).ob_type);
        std::ptr::eq(w_class, canonical).then_some(canonical)
    }
}

/// Canonical Python class for an exact int/float specialization operand.
/// Tagged ints have no object header and bool cannot be subclassed, so a null
/// result tells `walker_guard_exact_w_class` that the existing shape guard is
/// already sufficient. Heap operands were admitted by
/// `is_exact_builtin_instance` in the shared operand gate.
fn walker_numeric_builtin_class(obj: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    // A walker-made wrapint has no concrete object; it is always `int`
    // (`emit_box_int_inline` / `note_class_word_after_new`).
    if obj.is_null() {
        return pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        pyre_object::PY_NULL
    } else if unsafe { pyre_object::is_bool(obj) } {
        pyre_object::PY_NULL
    } else if unsafe { pyre_object::is_float(obj) } {
        pyre_object::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE)
    } else {
        pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE)
    }
}

/// Whether `op` is already the constant `expected`.
///
/// A freshly recorded GETFIELD carries the live class as its concrete
/// snapshot (`field_sanity_load`), so `box_value` matching is not a proof
/// — that is exactly the case that still needs `GUARD_VALUE`.  After
/// `implement_guard_value` the heapcache must serve the *constant* box
/// (`is_constant()`), and only then is a second pin a tautology.
fn walker_ref_box_is<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op: OpRef,
    expected: pyre_object::PyObjectRef,
) -> bool {
    op.is_constant()
        && matches!(
            ctx.trace_ctx.box_value(op),
            Some(majit_ir::Value::Ref(r)) if r.0 == expected as usize
        )
}

/// Read `obj.w_class` through the heapcache and pin it to `expected_typeobj`.
///
/// `pyjitpl.py` `_opimpl_getfield_gc_any_pureornot` records the load once
/// (`upd.getfield_now_known`), then `implement_guard_value` makes that box
/// the constant.  A later reader of the same field must reuse the cached
/// box and emit no second GETFIELD / GUARD_VALUE.  The uncached
/// `GETFIELD_GC_R` spelling used to skip that cache, so
/// `walker_emit_super_proxy` and `walker_emit_super_attr_lookup_guards`
/// each recorded the same receiver-class proof — the name-bound
/// `s = super(C, self); s.val` spelling paid it twice per iteration.
fn walker_pin_instance_w_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    expected_typeobj: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let descr = crate::descr::w_class_descr();
    let actual = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, descr);
    let expected = ctx.trace_ctx.const_ref(expected_typeobj as i64);
    if walker_ref_box_is(ctx, actual, expected_typeobj) {
        return Ok(expected);
    }
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[actual, expected])?;
    // `pyjitpl.py` `implement_guard_value`: generate_guard then replace_box.
    // heapcache.replace_box marks the GETFIELD FrontendOp replaced-with-const;
    // the next `opimpl_getfield_gc_r` returns that Const via
    // `maybe_replace_with_const`.  It does not write the Const back with
    // `getfield_now_known`.
    ctx.trace_ctx.heap_cache_mut().replace_box(actual, expected);
    Ok(expected)
}

/// Walker-native mirror of the trait `trace_guard_exact_w_class`
/// (`trace_opcode.rs`): emit `getfield_gc_r(w_class)` →
/// `guard_value(expected)` so the spec_ii fast path only stays live for an
/// element whose Python-level `w_class` is the canonical type object — a later
/// subclass sharing the payload `ob_type` side-exits rather than being
/// silently rewrapped as a plain int.  Skipped for an unescaped element (a
/// fresh `wrapint` box is provably plain, and reading its `w_class` would
/// force the box OptVirtualize removes).
///
/// Spelled `guard_value` rather than `ptr_eq` + `guard_true` because only
/// `guard_value` teaches the optimizer the equality: `rewrite.py
/// postprocess_GUARD_VALUE` runs `make_constant(arg0, arg1)`, so the read
/// becomes a constant and exports LEVEL_CONSTANT at the loop jump instead of
/// being carried as a label argument.  `optimize_GUARD_TRUE` learns nothing
/// about the box inside a `ptr_eq`, which leaves a loop-invariant class word
/// to be re-read and re-compared once per iteration.
fn walker_guard_exact_w_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    expected_typeobj: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    if expected_typeobj.is_null() || ctx.trace_ctx.heap_cache().is_unescaped(obj) {
        return Ok(());
    }
    // A ConstPtr interned / co_consts / builtin-method box already is the
    // exact builtin. Recording GETFIELD_GC_R(w_class)+GUARD_VALUE every
    // iteration doubled `list_ops` guard failures. Heap instances handed
    // through `inline_call` can have `__class__` reassigned; those are not
    // `is_constant()` interned boxes with a matching live `w_class`.
    if obj.is_constant()
        && let Some(concrete) = walker_concrete_ref_object(ctx, obj)
        && !concrete.is_null()
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(concrete))
        && std::ptr::eq(unsafe { (*concrete).w_class }, expected_typeobj)
    {
        return Ok(());
    }
    // A builtin whose user subclass instances carry their own typeptr
    // (`typedef.py _getusercls`) is proven exact by its typeptr: `GuardClass`
    // is the whole proof and the class word is never read.
    if let Some(concrete) = walker_concrete_ref_object(ctx, obj)
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(concrete))
    {
        let ob_type = unsafe { (*concrete).ob_type };
        if !unsafe { (*ob_type).user_subclass }.is_null()
            && std::ptr::eq(
                pyre_object::pyobject::get_instantiate(unsafe { &*ob_type }),
                expected_typeobj,
            )
        {
            return walker_guard_class(ctx, op_pc, obj, ob_type as i64);
        }
    }
    // Every predicate that admits one of these folds — `is_exact_builtin_instance`,
    // `is_plain_int1`, [`walker_exact_builtin_class`] — treats a null `w_class` as a
    // second spelling of "exact builtin", while the guard below reads the slot and
    // pins a single value.  Pinning the canonical against an operand that carries
    // the null spelling emits a guard the recorded operand itself fails, and nothing
    // writes the slot afterwards, so it fails on every execution without ever
    // converging — one bridge per `trace_eagerness` bucket.  Establishing that the
    // operand carries what is pinned is the caller's, which is what this checks.
    debug_assert!(
        walker_concrete_ref_object(ctx, obj).is_none_or(|concrete| {
            (pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(concrete))
                || std::ptr::eq(unsafe { (*concrete).w_class }, expected_typeobj)
        }),
        "guard_exact_w_class at pc={op_pc} would pin a `w_class` its recorded operand does not carry",
    );
    walker_pin_instance_w_class(ctx, op_pc, obj, expected_typeobj)?;
    // Do not stamp `class_now_known` here. Exact `w_class` is a stronger
    // proof, but `optimize_GETFIELD` / peel import can fold the
    // Getfield+GuardValue away on a LABEL input. `walker_guard_class`
    // must still emit `GuardClass` so a later entry whose box is an
    // int cannot take the long `GetfieldGcR(value)` path and pass
    // intval to `jit_bigint_int_mul` (`selfrec_bridge_nontail_promote`).
    Ok(())
}

/// Global descr-pool sub-jitcode lookup (resolves a global jitcode index
/// through `ALL_JITCODES`, mirroring the shadow walker's lookup).  A
/// build-time canonical sub-body (`w_list_append`) carries no per-fn descr
/// pool (`JitCodeBody` has no `descrs` field), so it resolves its `d`/`j`
/// descr operands through the global pool (`all_descr_refs()` /
/// `RawDescrPool::Global`), and its inline-call descrs through this lookup.
static GLOBAL_SUB_JITCODE_LOOKUP_FN: fn(usize) -> Option<SubJitCodeBody> =
    sub_jitcode_body_by_index;

/// Return the driver green key for this top-level FOR_ITER residual.
///
/// `op_pc` belongs to the current per-CodeObject JitCode, while the driver
/// key is `(W_Code, python FOR_ITER pc)`.  The full-body snapshot root owns
/// the outer JitCode metadata needed for that inversion.  Inlined sub-walks
/// have a distinct callee JitCode but deliberately share the root snapshot;
/// decline their range fold rather than associating a callee guard with the
/// caller's key.
fn walker_foriter_green_key<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Option<u64> {
    // `op_pc` indexes the JitCode the walk is CURRENTLY executing, so the code
    // object paired with it has to be that JitCode's.  Inside an inline
    // sub-walk that is the callee's, resolved through
    // `InlineCalleeConsts` exactly as `fbw_foriter_body_from_op_pc` resolves
    // the matching body coordinate — `fbw_mode.snapshot_sym` still names the
    // outer portal, and mapping a callee offset through the caller's pc tables
    // answers with a Python pc that belongs to neither loop.
    //
    // Reading the caller's tables was why this returned `None` for every
    // sub-walk, and the FOR_ITER specializations that key their demotion descr
    // off this green key declined with it: a `for` loop inside an inlined
    // callee kept the opaque `ForIterNext` residual, which runs the iterator's
    // `__next__` as a real frame in the plain interpreter.
    if let Some(consts) = ctx.inline_callee_consts {
        let w_code = ctx.inline_w_code() as *const ();
        if w_code.is_null() {
            return None;
        }
        let jitcode = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)?;
        let foriter_start_pc =
            crate::py_coord::containing_py_pc_for_jitcode_pc(&jitcode.metadata, op_pc) as usize;
        return Some(crate::driver::make_green_key(
            w_code,
            foriter_start_pc,
            ctx.session.borrow().is_being_profiled,
        ));
    }
    if ctx.fbw_mode.snapshot_sym.is_null() {
        return None;
    }
    // SAFETY: snapshot-root mode keeps the outer `PyreSym` live throughout
    // the full-body walk.  This reads immutable code metadata only.
    let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
    if sym.jitcode().is_null() {
        return None;
    }
    let jitcode = unsafe { &*sym.jitcode() };
    let raw_code = unsafe { jitcode.raw_code() };
    let w_code = pyre_interpreter::live_code_wrapper(raw_code as *const ()) as *const ();
    if w_code.is_null() {
        return None;
    }
    let foriter_start_pc =
        crate::py_coord::containing_py_pc_for_jitcode_pc(&jitcode.payload.metadata, op_pc) as usize;
    Some(crate::driver::make_green_key(
        w_code,
        foriter_start_pc,
        ctx.session.borrow().is_being_profiled,
    ))
}

fn mark_trace_reads_module_global(
    tc: &mut TraceCtx,
    w_globals: pyre_object::PyObjectRef,
    name: &str,
) {
    if !w_globals.is_null() && crate::state::module_dict_cell_slot_direct(w_globals, name).is_some()
    {
        tc.reads_module_global = true;
    }
}

fn mark_trace_reads_module_global_from_code(
    tc: &mut TraceCtx,
    w_globals: pyre_object::PyObjectRef,
    w_code_ptr: usize,
    name_idx: usize,
) -> bool {
    // A null globals dict cannot be probed for membership; report unresolved so
    // the caller takes the conservative fallback, matching
    // `mark_trace_reads_module_global_from_frame_name`.
    if w_globals.is_null() || w_code_ptr == 0 {
        return false;
    }
    let Some(name) = (unsafe {
        let raw = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject;
        if raw.is_null() {
            None
        } else {
            pyre_interpreter::pyframe::load_name_from_code(&*raw, name_idx).map(|n| n.to_string())
        }
    }) else {
        return false;
    };
    mark_trace_reads_module_global(tc, w_globals, &name);
    true
}

fn mark_trace_reads_module_global_from_frame_name(
    tc: &mut TraceCtx,
    frame_ptr: usize,
    w_name_ptr: usize,
) -> bool {
    if frame_ptr == 0 || w_name_ptr == 0 {
        return false;
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return false;
    }
    let Some(name) = (unsafe {
        pyre_object::unicodeobject::w_str_get_value_opt(w_name_ptr as pyre_object::PyObjectRef)
    }) else {
        return false;
    };
    mark_trace_reads_module_global(tc, w_globals, name);
    true
}

/// Shared module-dict cell fast path for the LOAD_GLOBAL / LOAD_NAME folds:
/// const-fold a module-global name read to a `QuasiimmutField` version guard
/// + elidable `jit_namespace_cell_lookup`, reading an `ObjectMutableCell`'s
/// `w_value` live.  Returns `false` (fall through to the residual) for a
/// missing / `IntMutableCell` slot.
fn emit_module_dict_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    w_globals: pyre_object::PyObjectRef,
    name: &str,
) -> Result<bool, DispatchError> {
    // Cell fast path applies only to a module dict still in strategy mode
    // whose slot holds a raw value, an `ObjectMutableCell`, or an
    // `IntMutableCell`; null/absent keys fall through to the residual.
    if let Some(slot) = crate::state::module_dict_cell_slot_direct(w_globals, name) {
        if let Some(stored) = crate::state::module_dict_cell_value_direct(w_globals, slot) {
            if !stored.is_null() {
                // `celldict.py getdictvalue_no_unwrapping` promotes `self`
                // and reads `version?` on every lookup, cell or raw.
                // `_setitem_str_cell_known` and `delitem` call `mutated()`.
                // A watcher already installed makes
                // `pyjitpl.py opimpl_jit_force_quasi_immutable` abort the trace.
                return emit_namespace_cell_fold(
                    ctx, op_pc, dst, dst_bank, w_globals, slot, stored, true, true,
                );
            }
        }
        // The name is present in the module dict but unfoldable
        // (null / strategy-switched); keep the residual
        // rather than misreaching into the builtins fallback (the residual
        // reads the live globals slot, which is correct).
        return Ok(false);
    }
    // Name absent from this module dict.
    Ok(false)
}

/// Emit the `QUASIIMMUT_FIELD(ns, slot)` + elidable `jit_namespace_cell_lookup`
/// + (for a mutable cell) live field read that the LOAD_GLOBAL cell fold lowers
/// to, seeding the dst's concrete with the unwrapped value.  `stored` is the
/// raw value-or-cell at `slot` of `ns` (a module dict in strategy mode); the
/// caller has already proven it foldable and non-null.  An
/// `ObjectMutableCell` reads `cell.w_value` (`getfield_gc_r`); an
/// `IntMutableCell` reads `cell.intvalue` (`getfield_gc_i`) and re-boxes it
/// (the box is elided by the optimizer when the sole consumer unboxes).
fn emit_namespace_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    ns: pyre_object::PyObjectRef,
    slot: usize,
    stored: pyre_object::PyObjectRef,
    guard_frame_globals: bool,
    pin_version: bool,
) -> Result<bool, DispatchError> {
    if guard_frame_globals && !guard_current_frame_globals_identity(ctx, op_pc, ns)? {
        return Ok(false);
    }
    // `celldict.py getdictvalue_no_unwrapping` is `@elidable_promote` on
    // `version?` for every lookup: present cell, raw value, and miss.
    // `setitem_str` does the same lookup before `write_cell`.  A later
    // insert or `delitem` (`mutated()`) fails GUARD_NOT_INVALIDATED.
    if pin_version {
        if !walker_pin_namespace_version(ctx, op_pc, ns)? {
            return Ok(false);
        }
        if crate::state::module_dict_cell_value_direct(ns, slot) != Some(stored) {
            return Ok(false);
        }
    }
    let Some((result_opref, _)) = emit_namespace_cell_value(ctx, op_pc, stored)? else {
        return Ok(false);
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result_opref)?;
    // `pyjitpl.py _opimpl_residual_call*` finishes its no-raise
    // tail with `metainterp.clear_exception()`.  The fold replaces a
    // SUCCESSFUL (non-raising) `load_global_fn` residual, so it must mirror
    // that clear: in a handler body the except-side `LOAD_GLOBAL` runs with a
    // standing `last_exc_value` (the just-raised exception being matched), and
    // the residual success arm (`ctx.last_exc_value = None` at the
    // `exec_result` Ok leg) is what drains it before the handler's trailing
    // `catch_exception/L`.  Without this clear the elided residual leaves
    // `last_exc_value` set and the walk aborts `CatchExceptionWithActiveException`.
    ctx.clear_last_exc_value();
    Ok(true)
}

/// The value half of [`emit_namespace_cell_fold`], for callers with no jitcode
/// destination register to write: bake `stored` as a `ConstPtr`, read the live
/// field behind a mutable cell, and hand back the operand together with the
/// object it stands for.
///
/// The caller has already proven `stored` foldable and non-null AND pinned the
/// namespace version — that marker is what revokes the constant this emits.
/// Writing a destination is deliberately left out: `write_ref_reg` also stamps
/// `ctx.vstack_last_ref`, so a caller that is not standing in for a residual
/// must not be made to go through it.
fn emit_namespace_cell_value<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    stored: pyre_object::PyObjectRef,
) -> Result<Option<(OpRef, pyre_object::PyObjectRef)>, DispatchError> {
    // The value half is `typeobject.py unwrap_cell`, generated.  A
    // hand-written getfield/box here was the fold this descent replaces.
    specialize::try_walker_orthodox_unwrap_cell(ctx, op_pc, stored)
}

/// pyjitpl.py `reached_loop_header`: `live_arg_boxes += virtualizable_boxes;
/// live_arg_boxes.pop()`. Every declared slot is a box
/// (`virtualizable.py read_boxes`); a missing slot is a producer bug.
pub(crate) fn append_virtualizable_boxes(ctx: &TraceCtx, mut reds: Vec<OpRef>) -> Vec<OpRef> {
    reds.extend(ctx.virtualizable_data_boxes());
    reds
}

fn write_branch_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    result_write: Option<(usize, OpRef)>,
) -> Result<(), DispatchError> {
    if let Some((dst, resbox)) = result_write {
        let concrete = match ctx.trace_ctx.concrete_of_opref(resbox) {
            Some(Value::Int(value)) => ConcreteValue::Int(value),
            _ => ConcreteValue::Null,
        };
        write_int_reg(ctx, op_pc, dst, resbox, concrete)?;
    }
    Ok(())
}

fn branch_without_guard<Sym: WalkSym>(
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    taken_pc: usize,
    result_write: Option<(usize, OpRef)>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    write_branch_result(ctx, op.pc, result_write)?;
    Ok((DispatchOutcome::Continue, taken_pc))
}

fn guarded_branch_core<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    guard_opcode: OpCode,
    guard_operands: &[OpRef],
    taken_pc: usize,
    other_pc: usize,
    result_write: Option<(usize, OpRef)>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Resume-data capture (`capture_resumedata(resumepc=orgpc)` at
    // `pyjitpl.py`) is threaded via
    // `walker_capture_snapshot_for_last_guard(other_target)`.
    // Branch guards resume at the runtime jump destination — the
    // branch NOT taken in the trace — not the `goto_if_not` opcode
    // itself (`trace_opcode.rs`, `resume_pc =
    // other_target`).  The trace took the `switchcase` direction;
    // a guard failure flips to the other arm, where the Python
    // interpreter has already popped the comparison truth, so the
    // blackhole must re-enter past `POP_JUMP_IF_*`.  GuardTrue
    // (trace fell through to `op.next_pc`) → resume at `target`;
    // GuardFalse (trace jumped to `target`) → resume at `op.next_pc`.
    let other_target = resolve_branch_target_through_trampoline(code, other_pc);
    {
        // #124/#281: a branch guard whose resume target still holds a
        // live operand-stack temp (short-circuit `and`/`or`, the
        // conditional expression, chained comparison) keeps that temp
        // on the not-taken arm the single-frame snapshot does not model
        // by itself.  The kept temp(s) are recovered from the not-taken
        // edge's `ref_copy` parallel-move trampoline (decoded below),
        // exact for any kept-stack depth.  Plain `while` / `if`
        // branches resume at depth 0 and carry no kept temp.
        //
        // #171 sub-walk single-frame collapse: inside an inlined-callee
        // sub-walk that resumes at the caller's CALL boundary,
        // `other_target` is a *callee* coordinate absent from the outer
        // jitcode's py_pc→jitcode tables, so `branch_resume_target_stack_depth`
        // (which maps through `fbw_mode.snapshot_sym`) would read a
        // meaningless outer depth at the coincidental offset.  The
        // collapse guard does not resume at a callee coordinate at all:
        // like every other single-frame sub-walk guard it collapses to
        // the caller's CALL boundary (`entry_py_pc` / `outer_active_boxes`,
        // `walker_capture_snapshot_for_last_guard_impl`), re-executing
        // the whole call on deopt — so there is no callee kept-stack
        // slot to recover.  Treat it as depth 0.
        //
        // Scope this to the collapse case ONLY: the #68 multiframe
        // inline path (`n_parents == n_callees`, both > 0) resumes the callee at its
        // OWN pc through `GuardCaptureScope::branch_guard_jitcode_pc`
        // (`walker_capture_multi_frame_inline_snapshot`), so its
        // kept-stack branches still need the real depth/recovery.
        //
        // The non-collapse read resolves `other_target` through an
        // `ActiveResumeFrame`.  `current()` selects the innermost inlined
        // callee when a sub-walk is active (else the portal frame), so a
        // callee branch guard's `other_target` is inverted through the
        // callee's own tables rather than the outer frame's — the frame
        // whose box collection (`collect_callee_active_boxes`) already
        // keys off the same framestack top. The #68
        // multiframe path reaches this `else`; the single-frame collapse
        // case short-circuits to `None` above.
        let single_frame_collapse = ctx.fbw_mode.inline_subwalk && {
            let session = ctx.session.borrow();
            let n_parents = session
                .framestack
                .iter()
                .filter(|frame| !frame.parents.is_empty())
                .count();
            let n_callees = session.inline_depth();
            !(n_parents > 0 && n_parents == n_callees)
        };
        // A canonical helper body (`run_sub_jitcode_walk`) walks its OWN
        // jitcode over its OWN register bank and pushes no `InlineFrame`, so
        // `current()` resolves to the Python frame BELOW it instead -- the
        // innermost inlined callee, or the portal.  Every read taken through
        // that frame is then cross-bank: `other_target` is a helper offset,
        // and `depth_trivia` / `pcdep_trivia` answer a foreign offset through
        // a predecessor scan rather than failing, so the depth and the color
        // list come back plausible and wrong.  `kept_stack_has_boxed_int_hazard`
        // finishes the sequence by dereferencing whatever word the foreign
        // color lands on in `ctx.concrete_registers_r` as a `PyObjectRef`,
        // which is an unmapped address rather than a wrong answer.
        //
        // Depth 0 is the right answer, not merely the safe one: a helper frame
        // owns no Python operand stack, and its guards resume at the paused
        // parents' CALL coordinates
        // (`walker_capture_transparent_helper_snapshot`), re-executing the call
        // -- exactly the reasoning `single_frame_collapse` above already
        // applies to the helper walk whose framestack is empty.  This closes
        // the same case when it is not.
        //
        // `transparent_helper_subwalk` and not `inline_subwalk`: the latter is
        // also set for a Python-callee sub-walk, where the bank and the target
        // DO belong to the frame `current()` returns and the gate must keep
        // working.
        let gate_frame = if ctx.fbw_mode.transparent_helper_subwalk {
            None
        } else {
            ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym)
        };
        let resume_depth = if single_frame_collapse {
            None
        } else {
            gate_frame
                .as_ref()
                .and_then(|f| branch_resume_target_stack_depth(f, other_target))
        };
        let kept_stack = resume_depth.is_some_and(|d| d > 0);
        let depth_gt_1 = resume_depth.is_some_and(|d| d > 1);
        // Mirror-sourced kept-stack compile: a kept-stack guard
        // COMPILES whenever the walk-level operand-stack mirror
        // (`ctx.vstack_boxes`) covers EVERY kept resume slot
        // `0..resume_depth` with a non-NONE box — i.e. the
        // snapshot is 100% mirror-sourced with no legacy fallback.  This
        // bypasses the kept-stack declines below (whose purpose is
        // precisely the unreliable legacy resume the mirror replaces).
        // When the mirror does NOT cover a kept slot,
        // `mirror_covers_kept` is `false` and the hazard checks below
        // still run — but with the flat maps deleted they decline only
        // for an unrestorable-Ref arm (Hazard 1) or an INVALID mirror
        // (undermodeled walk); a VALID mirror with a NONE hole (an
        // edge-materialized merge temp) compiles, its slot sourced from
        // the decoded trampoline recovery (`resolved_recovered`).  That
        // last claim is not assumed: an uncovered slot with no recovery
        // entry has no per-slot source at all, and
        // `collect_outer_active_boxes` reports it so the snapshot capture
        // declines (`BranchGuardKeptSlotUnsourced`) rather than falling
        // through to the stale merge-color read.
        let mirror_covers_kept = ctx.vstack_valid
            && resume_depth.is_some_and(|d| {
                (0..d).all(|s| {
                    ctx.frame_state
                        .borrow()
                        .vstack_boxes
                        .get(s as usize)
                        .copied()
                        .is_some_and(|b| b != OpRef::NONE)
                })
            });
        // `branch_resume_target_stack_depth` reads
        // `fbw_mode.snapshot_sym`. Probe the jitcode-store depth for
        // the unrestorable-arm decline below as well.
        // `kept_stack_any_leg` and the kept-stack hazard checks below
        // all read `fbw_mode.snapshot_sym`, which models the top-level
        // traced jitcode's register file. In an inlined-callee sub-walk
        // (`is_top_level == false`) the current `concrete_registers_r`
        // is the callee's, so an outer stack-slot color indexes a
        // foreign callee register — `kept_boxed_int` below would then
        // dereference an unrelated `Ref`, a dangling pointer
        // (KERN_INVALID_ADDRESS / SIGSEGV). A callee branch collapses to
        // the caller's CALL boundary on deopt (the #171 single-frame
        // collapse), so it has no top-level kept-stack slot to recover;
        // gate on `is_top_level` and treat it as no kept stack, matching
        // the `resume_depth` collapse handling above.
        //
        // The depth lookup also keys off `ctx.outer_jitcode_index`,
        // which is the FBW sym's `(*sym.jitcode).index` and uniformly 0
        // (the canonical core's `index` is never stamped with the
        // runtime `MetaInterpStaticData.jitcodes` position), so for the
        // second and later distinct functions compiled in a program it
        // resolves to `jitcodes[0]` — the FIRST function — and reads its
        // metadata at this function's jitcode pc, yielding a wrong depth.
        // The FBW-leg `kept_stack` reads the depth through `gate_frame`
        // (`ActiveResumeFrame`, resolved per-function via
        // `ensure_jitcode_index`), so it is correct for every function;
        // `kept_stack_any_leg` is retained only as the fallback for the
        // `gate_frame == None` case.
        let kept_stack_any_leg = ctx.is_top_level
            && branch_resume_target_stack_depth_any_leg(other_target, ctx.outer_jitcode_index)
                .is_some_and(|d| d > 0);
        // A kept-stack guard's not-taken arm keeps one or more
        // operand-stack temps live across the guard.  The not-taken
        // edge resolves the merge through inline `ref_copy(dst <- src)`
        // moves (`flatten.rs insert_renamings`): the live kept
        // value sits at the guard-pc color `src`, which the walk has
        // written, while the resume merge color `dst` is unwritten at
        // the guard point.  Decode that trampoline into `(dst, src)`
        // pairs (`#420`); the snapshot / vable recovery then reads
        // `registers_r[src]` for each kept slot — exact for any kept-
        // stack depth, superseding the positional depth-1 heuristic.
        let raw_branch_target = other_pc;
        let kept_recovered = if kept_stack {
            decode_branch_trampoline_ref_moves(code, raw_branch_target)
        } else {
            Some(Vec::new())
        };
        if std::env::var_os("PYRE_DIAG124C").is_some() && kept_stack {
            eprintln!(
                "[edge124] pc={} depth={resume_depth:?} raw_target={raw_branch_target} \
                 moves(dst<-src)={kept_recovered:?}",
                op.pc,
            );
        }
        // Resolve each `(dst, src)` move to `(dst, live guard value)`
        // against the guard-state register file NOW, before recording
        // the guard.  A move whose `src` is out of range or holds
        // `OpRef::NONE` recovers no live kept value and is dropped so the
        // snapshot encoder never records a dead kept slot.  A const-source
        // `ref_copy` patches `src` into the constants window of
        // `registers_r`, so this one read covers register and const
        // sources alike.  This resolved set is the single source of truth
        // the snapshot encoder reads
        // (`GuardCaptureScope::branch_guard_kept_recovered` below);
        // `record_guard` records into the trace history only and
        // does not mutate `registers_r`, so reading it here is identical
        // to reading it post-guard.
        let resolved_recovered: Option<Vec<(u16, OpRef)>> = kept_recovered.as_ref().map(|mv| {
            mv.iter()
                .filter_map(|&(dst, src)| {
                    let v = ctx.registers_r.get(src as usize)?;
                    (v != OpRef::NONE).then_some((dst, v))
                })
                .collect()
        });
        // PARITY DEVIATION (converges at the symbolic-valuestack
        // capture, #73/#423): `opimpl_goto_if_not` (pyjitpl.py)
        // always records GUARD_TRUE/FALSE for a non-constant condition
        // and captures resumedata (pyjitpl.py), never declining —
        // its resume reconstruction is complete.  pyre's kept-stack
        // resume reconstruction is NOT yet complete (the walker
        // snapshot is partial), so recording the guard and
        // resuming would rebuild a kept slot as NULL / a wrong value:
        // the #416/#420 boxed-int short-circuit / conditional-expression
        // SIGSEGV + silent miscompile.  Until that capture lands pyre
        // deviates by declining here.  A kept-stack guard's not-taken
        // arm is only safe to compile when the blackhole can reconstruct
        // every value the arm reads on resume.  Three resume hazards make
        // a kept-stack arm unsafe; each is described at its check below.
        // Decline → interpreter (correct).  Applies to depth-1 and
        // depth > 1.
        //
        // Gate on `kept_stack` (per-function-correct via `gate_frame`)
        // OR the jitcode-store `kept_stack_any_leg` fallback.
        // `kept_stack_any_leg` alone is unsound for the second and later
        // distinct functions in a program: its `outer_jitcode_index`
        // resolves to `jitcodes[0]` (the first function) and reports a
        // wrong depth, so a kept-stack arm in a later function would skip
        // the decline and silently miscompile.  `kept_stack` reads the
        // correct per-function depth and closes that gap.
        if (kept_stack || kept_stack_any_leg) && !mirror_covers_kept {
            let liveness = branch_arm_resume_ref_liveness(ctx.fbw_mode, other_target);
            // Hazard (1): the not-taken arm reads a regular Ref register
            // the blackhole resumes as NULL (the conditional-expression
            // boxed-int NULL-deref crash).
            //
            // Scoped to the undermodeled invalid-mirror walk
            // (`!ctx.vstack_valid`), exactly like Hazards (2)/(3) below.
            // A VALID walk mirror sources every on-stack kept slot from
            // `ctx.vstack_boxes` and every kept local from the vable
            // shadow (`collect_outer_active_boxes`), so on resume the
            // not-taken arm's Ref reads are reconstructed per-slot even
            // when the guard's snapshot liveness (`live_ref`) does not
            // name that register color — the same per-slot recovery that
            // makes the short-circuit / conditional-expression kept-stack
            // reads (#416/#420) restorable for Hazards (2)/(3).  The
            // register-file scan only proves a hazard for the INVALID
            // mirror, where those per-slot sources are unavailable.
            let reads_null_ref = !ctx.vstack_valid
                && match &liveness {
                    Some((live_ref, num_regs_r)) => {
                        branch_arm_reads_unrestorable_ref(code, other_target, live_ref, *num_regs_r)
                    }
                    // Liveness unavailable (`fbw_mode.snapshot_sym` is
                    // null, or the coordinate is unresolved) — cannot
                    // prove restorable, so decline.
                    None => true,
                };
            // Hazard (2): the not-taken edge carries `ref_copy` renames
            // (`kept_recovered` non-empty) — the #416/#420 short-circuit
            // / chained-comparison kept-stack recovery.  Historically the
            // recovery read the flat merge-color register file, which
            // could return a stale reused box for a hoisted heap-int
            // constant (the `((i & 1) and 1000000)` silent miscompile).
            // With the flat maps deleted the capture is per-slot: a
            // VALID walk mirror sources every on-stack kept slot, and an
            // edge-materialized merge slot resolves through the decoded
            // `(dst, src)` trampoline against the guard-pc register file
            // (`resolved_recovered`) — verified byte-exact against the
            // declined-interpreter oracle across the 599-program
            // adversarial corpus (kept-stack census, incl. the
            // heap-int short-circuit / conditional-expression repros).
            // The hazard remains only for an UNDERMODELED walk (invalid
            // mirror: inline sub-walk / Unmodeled opcode), where those
            // per-slot sources are unavailable and forcing the recovery
            // through miscompiles (nested and/or under an inline
            // sub-walk).
            let uses_edge_recovery = !ctx.vstack_valid
                && kept_recovered
                    .as_deref()
                    .is_some_and(|moves| !moves.is_empty());
            // Hazard (3): a kept operand-stack slot itself holds a heap
            // int outside the 1-byte immediate range `[0, 256)` (the
            // accumulator in `acc += (x if c else y)`).  Historically the
            // guard's resume snapshot rebuilt such a slot through the
            // flat merge-color maps and could deliver a WRONG / NULL box
            // (the conditional-expression boxed-int crash).  As with
            // Hazard (2), the per-slot capture that replaced the flat
            // maps restores boxed-int kept slots faithfully whenever the
            // walk mirror is VALID (corpus-verified against the declined
            // oracle), so the hazard is scoped to the undermodeled
            // invalid-mirror walk.
            let kept_boxed_int = !ctx.vstack_valid
                && gate_frame.as_ref().is_some_and(|f| {
                    kept_stack_has_boxed_int_hazard(
                        f,
                        other_target,
                        &ctx.frame_state.borrow().concrete_registers_r,
                    )
                });
            // A not-taken arm resuming at an exception-handler-protected
            // PC carries the kept exception operand (`PUSH_EXC_INFO`'s
            // Ref) on its operand stack; the handler-entry mirror reseed
            // reconstructs it directly (so `mirror_covers_kept` gates this
            // whole block out), and where the mirror still does not cover,
            // that kept Ref always also trips Hazard (1)/(2)/(3) — so the
            // exc-region case needs no decline of its own.
            if reads_null_ref || uses_edge_recovery || kept_boxed_int {
                // Attribute the kept-stack decline to the hazard that
                // fired and the mirror state behind it, so a corpus run
                // (`PYRE_FBW_DEBUG_ABORT`) can separate the distinct
                // decline causes without re-instrumenting: an inline
                // sub-walk (`subwalk`/`!vstack_valid`), a genuinely
                // unrestorable regular Ref (`reads_null_ref` with a
                // valid mirror), or an undermodeled-mirror boxed-int /
                // edge-recovery slot.
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[decline-why] PERMANENT pc={} other_target={} vstack_valid={} \
                         subwalk={} mirror_covers_kept={} depth_gt_1={} kept_stack={} \
                         kept_stack_any_leg={} reads_null_ref={} uses_edge_recovery={} \
                         kept_boxed_int={} \
                         kept_recovered_nonempty={}",
                        op.pc,
                        other_target,
                        ctx.vstack_valid,
                        ctx.fbw_mode.inline_subwalk,
                        mirror_covers_kept,
                        depth_gt_1,
                        kept_stack,
                        kept_stack_any_leg,
                        reads_null_ref,
                        uses_edge_recovery,
                        kept_boxed_int,
                        kept_recovered.as_deref().is_some_and(|m| !m.is_empty()),
                    );
                }
                // Stamp the abort coordinate at the raise point so the
                // driver gate cannot observe an unrelated prior abort.
                //
                // `MIFrame.run_one_step` has already advanced `frame.pc`, and
                // `opimpl_goto_if_not` has already selected its link, when a
                // tracing abort converts the frame to a blackhole
                // (`pyjitpl.py`, `blackhole.py:1800-1810`).  Preserve  allow-line-citation
                // that post-branch Python continuation and its popped stack,
                // rather than making the interpreter re-run the guard opcode.
                // The latter needs the transient truth operand which JitCode
                // keeps only in an Int register, and replaying from the trace
                // entry when it is absent double-applies prior residuals.
                latch_taken_python_branch_abort_stack(ctx, gate_frame.as_ref(), guard_opcode);
                // Stamp only at the root: a sub-walk raise must leave
                // `abort_in_subwalk` clear so `claim_abort_coordinate` can
                // own this pc. Overwrite a leftover true from a recovered
                // nested walk when this is the portal.
                if !ctx.fbw_mode.inline_subwalk {
                    ctx.session.borrow_mut().abort_in_subwalk = false;
                }
                return Err(DispatchError::BranchGuardUnrestorableKeptStackPermanent { pc: op.pc });
            }
        }
        // Depth > 1 with an unseeded overlay used to decline
        // (`BranchGuardKeptStackUnsupported`).  `generate_guard` has no
        // such gate: it snapshots the `-live-` int/ref/float set at this
        // pc, which is how a raw-address walk (`getfield_raw_i` /
        // `int_add` substruct) keeps an int-bank address live across a
        // `goto_if_not` whose Python operand stack is also live (the
        // nested FOR_ITER iterators in `searches()`).  Capture below
        // reads those banks; `BranchGuardKeptSlotUnsourced` still
        // declines a kept Ref with no per-slot source.
        ctx.trace_ctx.record_guard(guard_opcode, guard_operands, 0);
        // Publish the guard's own jitcode coordinate ONLY for the
        // kept-stack case so the snapshot encoder recovers the kept
        // operand-stack values from the guard-pc register file (the
        // resume coordinate `other_target` names a merge point whose
        // live colors the walk has not written at the guard point).
        // A depth-0 branch resumes losslessly at `other_target` via
        // the baseline `py_pc → jitcode` resume-translation path;
        // routing it through the
        // guard-pc carrier would resume one opcode early (re-running
        // `goto_if_not`) and desync the decoded box layout.
        // Feed the snapshot encoder the SAME resolved set the gate
        // checked (resolved above, before `record_guard`): each
        // `(dst, live guard value)`, sources already filtered to live
        // in-range values. Only a kept-stack guard publishes these
        // inputs.
        let kept = if kept_stack {
            resolved_recovered.unwrap_or_default()
        } else {
            Vec::new()
        };
        walker_capture_snapshot_for_last_guard_scoped(
            ctx,
            other_target,
            GuardCaptureScope {
                branch_guard_jitcode_pc: kept_stack.then_some(op.pc),
                branch_guard_kept_recovered: &kept,
                ..GuardCaptureScope::default()
            },
        )?;
    }
    write_branch_result(ctx, op.pc, result_write)?;
    Ok((DispatchOutcome::Continue, taken_pc))
}

/// Preserve the already-selected Python successor for a kept-stack branch
/// abort, matching the forward `MIFrame.pc` copied by
/// `convert_and_run_from_pyjitpl`.
fn latch_taken_python_branch_abort_stack<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    frame: Option<&ActiveResumeFrame>,
    guard_opcode: OpCode,
) {
    let Some(frame) = frame else {
        return;
    };
    let py_pc = ctx.vstack_cur_pypc as usize;
    let pjc = &frame.0;
    if pjc.code_ptr.is_null() {
        return;
    }
    let code = unsafe { &*pjc.code_ptr };
    let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, py_pc) else {
        return;
    };
    let truth = match guard_opcode {
        OpCode::GuardTrue => true,
        OpCode::GuardFalse => false,
        _ => return,
    };
    let jump = match instr {
        pyre_interpreter::Instruction::PopJumpIfTrue { .. } => truth,
        pyre_interpreter::Instruction::PopJumpIfFalse { .. } => !truth,
        _ => return,
    };
    let successor = if jump {
        crate::liveness::target_pc(code, &instr, py_pc, op_arg)
    } else {
        Some(crate::pyjitpl::semantic_fallthrough_pc(code, py_pc))
    };
    let Some(successor) = successor else {
        return;
    };
    let Some(depth) = crate::liveness::liveness_for(pjc.code_ptr)
        .depth_at_py_pc()
        .get(successor)
        .copied()
        .map(usize::from)
    else {
        return;
    };
    if !ctx.vstack_valid
        || depth > ctx.vstack_depth
        || depth > ctx.frame_state.borrow().vstack_boxes.len()
    {
        return;
    }
    fbw_branch_abort_stack_latch(
        successor,
        ctx.frame_state.borrow().vstack_boxes[..depth].to_vec(),
    );
}

fn goto_if_not_branch_on<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    condbox: OpRef,
    switchcase: i64,
    target: usize,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // pyjitpl.py `opimpl_goto_if_not`: `switchcase = box.getint()` then
    // `if switchcase: assert switchcase == 1`.  A non-0/1 value means the
    // box is not a Bool; abort the walk instead of killing the process.
    if switchcase != 0 && switchcase != 1 {
        return Err(DispatchError::GotoIfNotValueNotConcrete {
            pc: op.pc,
            value: condbox,
        });
    }
    let (guard_opcode, taken_pc, other_pc) = if switchcase != 0 {
        (OpCode::GuardTrue, op.next_pc, target)
    } else {
        (OpCode::GuardFalse, target, op.next_pc)
    };

    // `generate_guard` in `pyjitpl.py opimpl_goto_if_not` skips Const boxes.
    // No register replacement occurs here; fused comparisons pass
    // `replace=False`, preserving loop-variant conditions.
    if condbox.is_constant() {
        branch_without_guard(op, ctx, taken_pc, None)
    } else {
        guarded_branch_core(
            code,
            op,
            ctx,
            guard_opcode,
            &[condbox],
            taken_pc,
            other_pc,
            None,
        )
    }
}

fn int_ovf_jump<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let target = read_label(code, op, 0);
    let b1 = read_int_reg(code, op, 2, ctx)?;
    let b2 = read_int_reg(code, op, 3, ctx)?;
    let dst = code[op.pc + 5] as usize;
    let known = match (
        read_int_reg_concrete(code, op, 2, ctx),
        read_int_reg_concrete(code, op, 3, ctx),
    ) {
        (ConcreteValue::Int(lhs), ConcreteValue::Int(rhs)) => Some((lhs, rhs)),
        _ => None,
    };
    let (resbox, overflow) = record_int_ovf(ctx, op.pc, opcode, b1, b2, known)?;

    // `pyjitpl.py opimpl_int_add_jump_if_ovf` and
    // `pyjitpl.py handle_possible_overflow_error`: Const operands branch
    // without a guard; symbolic operands emit an operand-less overflow guard.
    let (guard_opcode, taken_pc, result_write) = if overflow {
        (OpCode::GuardOverflow, target, None)
    } else {
        (OpCode::GuardNoOverflow, op.next_pc, Some((dst, resbox)))
    };
    if resbox.is_constant() {
        branch_without_guard(op, ctx, taken_pc, result_write)
    } else {
        ctx.trace_ctx.record_guard(guard_opcode, &[], 0);
        // `handle_possible_overflow_error` resumes at `orgpc`: unlike
        // `goto_if_not`, blackhole must re-execute the arithmetic opcode so
        // the no-overflow path writes its result register.
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        write_branch_result(ctx, op.pc, result_write)?;
        Ok((DispatchOutcome::Continue, taken_pc))
    }
}

fn fused_goto_if_not_int<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let b = read_int_reg(code, op, 1, ctx)?;
    let condbox = record_int_cmp(ctx, opcode, a, b);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 2);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

/// The portal frame's `debugdata` read from `PyFrame.dispatch_bytecode`'s
/// `we_are_jitted()` arm (pyopcode.py):
///
/// ```text
/// while True:
///     assert next_instr & 1 == 0
///     self.last_instr = intmask(next_instr)
///     if jit.we_are_jitted():
///         _d = self.debugdata
///         if ec.space.reverse_debugging or (
///                 _d is not None and _d.w_f_trace is not None):
///             ec.bytecode_only_trace(self)
///             next_instr = r_uint(self.last_instr)
/// ```
///
/// `debugdata` is one of `PyFrame._virtualizable_` (interp_jit.py), so the
/// read resolves against `virtualizable_boxes` and folds to the value the
/// frame carried when the trace started.  A trace that folds a virtualizable
/// field without validating it compiles a loop that cannot observe that field
/// changing.  Upstream validates it, and its own optimized trace says how:
/// `patch_new_loop_to_load_virtualizable_fields` (compile.py) loads every
/// virtualizable field off the frame at loop entry, and virtual-state matching
/// then pins the one the body was specialized on —
///
/// ```text
/// +280: p5 = getfield_gc_r(p0, descr=<FieldP ...PyFrame.inst_debugdata 16>)
/// +316: label(p0, p1, i2, p3, i4, p5, ...)
/// +352: guard_isnull(p5, ...)
/// ```
///
/// so arming `f_trace` on a running frame fails that guard and the interpreter
/// resumes firing `line` events.  pyre loads the field at entry the same way
/// but pins nothing, so the compiled loop ran the whole tail blind.
///
/// Recorded at the portal merge point — the trace's counterpart of the top of
/// `dispatch_bytecode`'s loop.  The read is registered with the heapcache, so
/// a run of merge points with no intervening call collapses to one read and
/// one guard.  A call does not collapse, so a loop body carrying residual
/// calls records one pair per call-free window — measured, three for a loop
/// whose body makes one residual Python call.  Upstream carries exactly one
/// because its guard comes from virtual-state matching at the label rather
/// than from the recorded read; matching that would mean pinning the vable
/// field in the virtual state, which pyre does not do for `debugdata` today.
fn record_portal_debugdata_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    // An inlined callee's own header is not the portal loop the compiled code
    // re-enters, and the sub-walk's boxes describe the callee frame.
    if ctx.fbw_mode.inline_subwalk {
        return Ok(());
    }
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return Ok(());
    };
    let Some(field_index) = info.static_field_index_by_name("debugdata") else {
        return Ok(());
    };
    let Some(frame_box) = ctx.trace_ctx.standard_virtualizable_box() else {
        return Ok(());
    };
    // The value the trace is folding, read from `virtualizable_boxes` rather
    // than from the live frame: the guard exists to validate exactly that fold,
    // so a live frame that has since diverged from it must still be guarded
    // against, not silently agreed with.
    let Some((_, Value::Ref(debugdata))) = ctx.trace_ctx.virtualizable_entry_at(field_index) else {
        return Ok(());
    };
    if debugdata.as_usize() != 0 {
        let debugdata = debugdata.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
        // `_d is not None and _d.w_f_trace is not None`.  Upstream answers
        // that state with `ec.bytecode_only_trace(self)` on every opcode and
        // records the call, so its bridge out of the failed `guard_isnull`
        // keeps firing events from compiled code.  pyre's walker has no route
        // to record that call, so anything it compiles here runs the tail with
        // no way to fire the events the frame is owed.  Decline and leave the
        // frame to the interpreter, which fires them.
        //
        // Every walk that reaches a merge point in this state is declined, not
        // only one that has recorded nothing yet.  The walk that would
        // otherwise lose the tail is the BRIDGE out of the failed guard, and
        // that one resumes inside the loop body: it reaches its merge point
        // with the body already recorded, so a decline conditioned on an empty
        // recording never fires for it — measured, 401 of 1999 `line` events
        // on `settrace_f_trace_armed_mid_loop`, the rest swallowed by a bridge
        // that compiled at the 400th guard failure.
        //
        // `getorcreatedebug` is the single creation point for the block, and
        // reading `f_lineno`, `f_locals`, `locals()`, `f_lasti` or `f_back`
        // does not reach it.  Two things do: `executioncontext.py _trace`,
        // which takes `getorcreatedebug(init_lineno=...)` before it calls the
        // callback, so every frame that has reported once carries the block
        // with `w_f_trace` still null; and `setprofile` / `force_all_frames`,
        // which mint it as `getorcreatedebug().is_being_profiled = ...` with
        // no tracer installed at all.
        //
        // So this arm is not the rare one it reads as — with the portal
        // serving traced frames it is the common one, and leaving it unguarded
        // left a loop that had already reported to a global hook blind to
        // `f_trace` being armed on it afterwards (measured: 6 `line` events
        // for a 10 000-iteration tail).  Guard the slot itself.  The profiling
        // route is why that guard is gated on `ec.w_tracefunc` below rather
        // than emitted here.  What bounds profiling's own event count is not
        // this arm, and is no longer open.  Three mechanisms bound it:
        // `is_being_profiled` is a portal green (`interp_jit.py greens`), so a
        // profiled activation names a different cell from the one an
        // unprofiled warm-up filled; `walker_foldable_runtime_helper`
        // (`residual_call.rs`) answers `RuntimeHelperKind::None` under that
        // green, so every CALL-family helper stays a residual instead of
        // folding past its report; and `residual_call_c_profile_frame`
        // (`call_jit.rs`) reads the PER-FRAME flag live, so a frame
        // `force_all_frames` marks mid-loop reports from its next residual on
        // rather than waiting for a re-entry.
        //
        // `a_profiler_installed_from_a_call_event_keeps_c_events`
        // (`pyre/bench/synth`) pins the result as an EXACT per-tail count, not
        // a floor: `c_call` and `c_return` must each equal the tail and must
        // grow by the whole difference between two tails, which is what
        // separates a live bound from a green read once at entry.  Nine
        // further `setprofile` fixtures beside it cover the frame-identity,
        // inlined-callee, portal-resume, force-all-frames and call-free-loop
        // shapes.
        //
        // The non-null half stays a decline rather than a guard, and it is
        // reached: `PYRE_FBW_DEBUG_ABORT` names four `PortalFrameTracerArmed`
        // aborts in that arm, the same count as when `debugdata` starts null.
        // Such a frame reports its whole tail either way — 99999 of 99999
        // `call` events at a 100000 tail, against 1042 with the decline
        // disabled — because `settrace` forces every frame, so the exit is
        // `GuardNotForced` where the guard below would otherwise be it, and
        // the decline then refuses to recompile.
        if !unsafe { (*debugdata).w_f_trace }.is_null() {
            census_record("PortalFrameTracerArmed::Decline");
            return Err(DispatchError::PortalFrameTracerArmed { pc: op_pc });
        }
        // Only a trace recorded while a global trace function is live needs the
        // guard below.  Without one, `record_portal_tracefunc_guard` pins
        // `ec.w_tracefunc` NULL at this same merge point, and installing a
        // trace function is what makes an `f_trace` fire at all
        // (`eval_loop_jit` gates `bytecode_trace` on that slot), so that guard
        // is already what leaves compiled code.  Emitting this one
        // unconditionally moved 36 fixtures — `guard_failures 4 -> 204` and a
        // bridge where there was none on `type_name_attr_fold` — for a state
        // they never enter: a debug block exists on any frame that has been
        // handed out, walked for `f_lineno` or asked for `locals()`.
        let ec = pyre_interpreter::call::getexecutioncontext();
        if ec.is_null() || unsafe { (*ec).w_tracefunc }.is_null() {
            return Ok(());
        }
        let read = read_portal_debugdata(
            ctx,
            op_pc,
            frame_box,
            &info,
            field_index,
            debugdata as usize,
        )?;
        let trace_descr = crate::descr::frame_debug_data_w_f_trace_descr();
        let trace_descr_index = trace_descr.index();
        if ctx
            .trace_ctx
            .heapcache_getfield_cached(read, trace_descr_index)
            .is_some()
        {
            return Ok(());
        }
        let armed = ctx
            .trace_ctx
            .record_op_with_descr(OpCode::GetfieldGcR, &[read], trace_descr);
        ctx.trace_ctx
            .heapcache_getfield_now_known(read, trace_descr_index, armed);
        ctx.trace_ctx
            .set_opref_concrete(armed, Value::Ref(majit_ir::GcRef(0)));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardIsnull, &[armed])?;
        ctx.trace_ctx.heap_cache_mut().nullity_now_known(armed);
        return Ok(());
    }
    read_portal_debugdata(ctx, op_pc, frame_box, &info, field_index, 0)?;
    Ok(())
}

/// Record the portal frame's `debugdata` read once and pin what it read: NULL
/// when the frame carries no debug block, non-null when it does.  Both arms of
/// `record_portal_debugdata_guard` need the read, and both need it guarded —
/// the value is the one the trace folded, so a frame whose block appears (or
/// disappears) under the same green key has to leave compiled code rather than
/// be agreed with.
fn read_portal_debugdata<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    frame_box: OpRef,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    field_index: usize,
    concrete: usize,
) -> Result<OpRef, DispatchError> {
    // Resolved here rather than at the call sites: minting a struct descr
    // registers it, so resolving it on a path that then declines is an
    // observable side effect on every later descr index.
    let descr = info.static_field_struct_descr(field_index);
    let descr_index = descr.index();
    if let Some(read) = ctx
        .trace_ctx
        .heapcache_getfield_cached(frame_box, descr_index)
    {
        return Ok(read);
    }
    let read = ctx
        .trace_ctx
        .record_op_with_descr(OpCode::GetfieldGcR, &[frame_box], descr);
    ctx.trace_ctx
        .heapcache_getfield_now_known(frame_box, descr_index, read);
    ctx.trace_ctx
        .set_opref_concrete(read, Value::Ref(majit_ir::GcRef(concrete)));
    let opcode = if concrete == 0 {
        OpCode::GuardIsnull
    } else {
        OpCode::GuardNonnull
    };
    walker_emit_guard_with_snapshot(ctx, op_pc, opcode, &[read])?;
    ctx.trace_ctx.heap_cache_mut().nullity_now_known(read);
    Ok(read)
}

/// True while a trace function or a profiler is installed on the running
/// ExecutionContext — `executioncontext.py call_trace`'s own test,
/// `self.gettrace() is not None or self.profilefunc is not None`.
///
/// Every door this walker has into a Python callee — the inline, the direct
/// `CALL_ASSEMBLER` into the callee's own compiled trace — enters the callee
/// past `pyframe.py execute_frame`, and `execute_frame` is what runs
/// `ec.call_trace` / `ec.return_trace` around the eval loop.  So while this
/// holds, none of those doors can report what the callee owes and the call has
/// to stay a residual, which reaches `execute_frame` and runs both.  Upstream
/// needs no such test: it traces THROUGH `execute_frame`, so the reads become
/// guards in the compiled loop and the events are recorded with the callee.
///
/// Read live off the running context rather than from a recorded box.  The
/// answer only decides whether to record a door at all, and every caller bails
/// before recording any IR; the other half — a hook installed after the trace
/// was recorded — is `record_portal_tracefunc_guard` for a trace function and
/// `record_portal_profilefunc_guard` (`profilefunc?`) for a profiler.
pub(crate) fn ec_hook_installed() -> bool {
    let ec = pyre_interpreter::call::getexecutioncontext();
    !ec.is_null() && unsafe { !(*ec).w_tracefunc.is_null() || (*ec).profilefunc.is_some() }
}

/// The global trace function `executioncontext.py gettrace` reads —
/// `jit.promote(self.w_tracefunc)` — pinned NULL at the portal merge point.
///
/// The events an inlined callee owes a trace function are `call_trace` /
/// `return_trace`, which `pyframe.py execute_frame` runs around the eval loop.
/// Upstream traces through both, so whatever inlines the callee inlines their
/// `gettrace()` reads with it and the compiled loop carries a guard; this
/// walker synthesizes the call instead (`walker_ec_enter`, `inline_call.rs`)
/// and records neither, so without a guard a loop compiled with no tracer goes
/// on inlining — and silencing — its callees after one is installed.  Measured
/// over a 100 000-iteration tail with `sys.settrace` armed from inside the
/// loop: 1 043 `call` events for a callee the walker inlines, against 100 000
/// on cpython 3.14.6, on pypy3 7.3.22 and for a callee left residual.
///
/// `w_tracefunc` is an ExecutionContext field rather than a frame one, so the
/// read starts from the portal's own `ec` red (`interp_jit.py reds =
/// ['frame', 'ec']`) rather than a frame field; only a bridge whose red has
/// not been seeded needs `walker_ensure_execution_context` to recover it from
/// the frame.  The slot is `w_tracefunc?`, so the pin is a `QUASIIMMUT_FIELD`
/// marker plus `GUARD_NOT_INVALIDATED`; `settrace` invalidates the watchers.
/// The `GuardIsnull` is the `promote(None)` half.  The read is registered
/// with the heapcache, so a run of merge points with no intervening call
/// collapses to one loop-invariant read.
///
/// A trace recorded while the slot is ALREADY non-NULL records nothing: there
/// is no fold to validate, and `try_walker_inline_resolved_user_call_inner`
/// declines every Python-callee inline in that state, so the events come from
/// the interpreter's own `execute_frame`.  `sys.setprofile` is the
/// sibling `profilefunc?` pin (`record_portal_profilefunc_guard`), not
/// this slot.
fn record_portal_tracefunc_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    // An inlined callee's own header is not the portal loop the compiled code
    // re-enters, and the sub-walk's boxes describe the callee frame.
    if ctx.fbw_mode.inline_subwalk {
        return Ok(());
    }
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return Ok(());
    }
    // A hook installed after the loop was recorded fails `GuardIsnull`
    // (or `GUARD_NOT_INVALIDATED`).  The bridge out of that guard
    // resumes in the body with the hook already live; this walker
    // cannot record `ec.call_trace`, so compiling that bridge would
    // silence the tail.  Decline, same as `w_f_trace` above, and
    // leave the events to the interpreter.
    if !unsafe { (*ec).w_tracefunc }.is_null() {
        return Err(DispatchError::PortalFrameTracerArmed { pc: op_pc });
    }
    let Some(ec_box) = walker_ensure_execution_context(ctx) else {
        return Ok(());
    };
    // The red stays an input. The concrete is what
    // `opimpl_record_quasiimmut_field` reads with `box.getref_base()`.
    stamp_live_execution_context(ctx.trace_ctx, ec_box);
    let descr = crate::descr::ec_w_tracefunc_descr();
    let descr_index = descr.index();
    // Consult the cache before the marker. The getfield below publishes
    // the loaded constant; the marker is only the invalidation half.
    let already = ctx
        .trace_ctx
        .heapcache_getfield_cached(ec_box, descr_index)
        .is_some();
    // `_immutable_fields_ = ['w_tracefunc?']`: `gettrace` is
    // `jit.promote(self.w_tracefunc)`.  The `?` is the invalidation
    // half — `settrace` notifies, `GUARD_NOT_INVALIDATED` fails, and
    // the loop is not re-entered with the folded NULL.
    crate::state::record_quasiimmut_field(ctx.trace_ctx, ec_box, descr.clone());
    if already {
        return Ok(());
    }
    // `executioncontext.py gettrace`: `return jit.promote(self.w_tracefunc)`
    // on a `w_tracefunc?` slot.  The marker plus `GUARD_NOT_INVALIDATED`
    // is what `?` costs; `settrace` invalidates the watchers.  The
    // `GuardIsnull` is the `promote(None)` half this portal records.
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    let read = ctx
        .trace_ctx
        .record_op_with_descr(OpCode::GetfieldGcR, &[ec_box], descr);
    ctx.trace_ctx
        .heapcache_getfield_now_known(ec_box, descr_index, read);
    ctx.trace_ctx
        .set_opref_concrete(read, Value::Ref(majit_ir::GcRef(0)));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardIsnull, &[read])?;
    ctx.trace_ctx.heap_cache_mut().nullity_now_known(read);
    Ok(())
}

/// `executioncontext.py` `profilefunc?`.  We do not trace through
/// `execute_frame`, so PyPy's `getfield + int_is_zero + guard_true` is
/// not a compiled-loop read we have to replay.  The `?` watcher is what
/// `setllprofile` needs: a compiled loop jumps to itself and never
/// re-reads the `is_being_profiled` green, and GNI fails when the slot
/// is written.  Recording the promote half here ate five ops of a
/// `trace_limit=8` walk and fragmented `trace_too_long_effect_replay`
/// (aborts 17→46, guard_failures 1247→7746).  Marker only; the pending
/// GNI flushes on the next real guard, coalesced with `w_tracefunc?`.
fn record_portal_profilefunc_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    _op_pc: usize,
) -> Result<(), DispatchError> {
    if ctx.fbw_mode.inline_subwalk {
        return Ok(());
    }
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() || unsafe { (*ec).profilefunc.is_some() } {
        return Ok(());
    }
    let Some(ec_box) = walker_ensure_execution_context(ctx) else {
        return Ok(());
    };
    stamp_live_execution_context(ctx.trace_ctx, ec_box);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        ec_box,
        crate::descr::ec_profilefunc_descr(),
    );
    Ok(())
}

/// RPython `pyjitpl.py _establish_nullity(box, orgpc)` — the shared body
/// behind `opimpl_goto_if_not_ptr_nonzero` / `opimpl_goto_if_not_ptr_iszero`:
///
///   value = box.nonnull()
///   if heapcache.is_nullity_known(box):
///       profiler.count_ops(rop.GUARD_NONNULL, HEAPCACHED_OPS)
///       return value
///   if value:
///       if not heapcache.is_class_known(box):
///           generate_guard(rop.GUARD_NONNULL, box, resumepc=orgpc)
///   else:
///       if not isinstance(box, Const):
///           generate_guard(rop.GUARD_ISNULL, box, resumepc=orgpc)
///           promoted_box = executor.constant_from_op(box)
///           replace_box(box, promoted_box)
///   heapcache.nullity_now_known(box)
///   return value
///
/// Unlike the compare-fused gotos this records no condition and emits no
/// GuardTrue/GuardFalse: the nullity guard *is* the guard, and the branch
/// direction comes from the pointer the walk observed. The guard emission
/// stays walker-side because resume snapshots are —
/// `walker_emit_guard_with_snapshot` is the walk's `generate_guard`.
///
/// The heapcache check runs before `box.nonnull()`: upstream reads the
/// concrete pointer unconditionally, but the walker knows it only through
/// the concrete shadow, which may be absent even when a prior guard already
/// proved the box's nullity. Consulting the cache first keeps the known
/// answer from being lost to a missing shadow (the value is read lazily only
/// on a cache miss).
///
/// Returns `box.nonnull()`.
fn establish_nullity<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    boxref: OpRef,
) -> Result<bool, DispatchError> {
    // Pyre's `is_nullity_known` splits upstream's boolean into which side is
    // known (`Some(true)` non-null, `Some(false)` null, `None` unknown). When
    // the nullity is already known, the stored side *is* `box.nonnull()` — an
    // SSA box's nullity is immutable — so the heapcache answers the branch
    // without reading the pointer. This is consulted before the concrete
    // shadow so a box whose nullity a prior guard proved still resolves even
    // when its Ref shadow is absent.
    let known = ctx.trace_ctx.heapcache_nullity_known(boxref);
    if let Some(value) = known {
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GuardNonnull,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return Ok(value);
    }
    // Nullity not yet known: `box.nonnull()` reads the pointer itself. The
    // walker knows it only through the concrete shadow, and an absent shadow
    // is not evidence of either nullity, so decline rather than guess the
    // branch direction.
    let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: op.key,
        });
    };
    let value = !ptr.is_null();
    if value {
        // A known class already implies non-null, so the guard is redundant.
        if !ctx.trace_ctx.heap_cache().is_class_known(boxref) {
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[boxref])?;
        }
    } else if !boxref.is_constant() {
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardIsnull, &[boxref])?;
        // `constant_from_op` of a proven-null ref is the null constant.
        let promoted = ctx.trace_ctx.const_ref(0);
        vable_ops::walker_replace_box(ctx, boxref, promoted);
    }
    ctx.trace_ctx.heap_cache_mut().nullity_now_known(boxref);
    Ok(value)
}

/// The unary members of the fused-goto family. `pyjitpl.py`
/// `opimpl_goto_if_not_int_is_zero` and `opimpl_goto_if_not_int_is_true` record
/// the condition and hand it to `opimpl_goto_if_not` with `replace=False`:
///
///   condbox = self.execute(rop.INT_IS_ZERO, box)
///   self.opimpl_goto_if_not(condbox, target, orgpc, replace=False)
///
/// `replace=False` is why the condition is recorded and then dropped: it names
/// no register, so there is nothing for a later op to read it back from.
///
/// Operand layout `iL`: 1B int reg + 2B label.
fn fused_goto_if_not_int_unary<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let condbox = record_int_unary(ctx, opcode, a);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 1);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

fn fused_goto_if_not_float<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let b = read_float_reg(code, op, 1, ctx)?;
    let condbox = record_float_cmp(ctx, opcode, a, b);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 2);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

fn fused_goto_if_not_ptr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_ref_reg(code, op, 0, ctx)?;
    let b = read_ref_reg(code, op, 1, ctx)?;
    let a_concrete = read_ref_reg_concrete(code, op, 0, ctx);
    let b_concrete = read_ref_reg_concrete(code, op, 1, ctx);
    let condbox = record_ptr_cmp(ctx, opcode, a, b, a_concrete, b_concrete);
    let switchcase = match ctx.trace_ctx.concrete_of_opref(condbox) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::GotoIfNotValueNotConcrete {
                pc: op.pc,
                value: condbox,
            });
        }
    };
    let target = read_label(code, op, 2);
    goto_if_not_branch_on(code, op, ctx, condbox, switchcase, target)
}

/// Per-opname dispatch table. Returning `(outcome, next_pc)` lets
/// branching handlers (`goto/L`) override the linear `op.next_pc`
/// advance; non-branching handlers return `op.next_pc` unchanged.
fn handle<Sym: WalkSym>(
    op: &DecodedOp,
    code: &[u8],
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // The `int_*` / `float_*` / `ptr_*` families whose arm is a uniform
    // `HELPER(code, op, ctx, OpCode::VARIANT)` call are dispatched from the
    // `regular_record_table!`-generated table (see `arith.rs`); a `Some`
    // there stands in for the arm that used to live in this `match`.
    if let Some(res) = dispatch_regular_record(op, code, ctx) {
        return res;
    }
    match op.key {
        "live/" => Ok((DispatchOutcome::Continue, op.next_pc)),
        "loop_header/i" => {
            // pyjitpl.py `opimpl_loop_header(jdindex, orgpc)`:
            // pure flag setter — stamps `seen_loop_header_for_jdindex` so
            // the following `jit_merge_point` treats the arrival as a
            // loop crossing (the lowered `can_enter_jit` at a backward
            // jump, jtransform.py). The close/register decision
            // happens at the merge point (pyjitpl.py). Mirrors
            // majit's `BC_LOOP_HEADER` arm (`pyjitpl/dispatch.rs`).
            let jd_opref = read_int_reg(code, op, 0, ctx)?;
            let jdindex = match ctx.trace_ctx.concrete_of_opref(jd_opref) {
                Some(Value::Int(v)) => v,
                _ => return Err(DispatchError::LoopHeaderJdIndexUnresolved { pc: op.pc }),
            };
            ctx.trace_ctx.seen_loop_header_for_jdindex = jdindex as i32;
            ctx.trace_ctx.seen_loop_header_jit_pc = Some(op.pc);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython parity: `pyjitpl.py _opimpl_inline_call*`
        // pushes a fresh `MIFrame(jitcode)` populated with caller args,
        // raises `ChangeFrame()` so the metainterp loop dispatches the
        // next op on the new frame, and on `*_return` pops back via
        // `metainterp.finishframe(value)` — writing `value` into the
        // caller's `>X` slot. Walker simulates the same shape with
        // synchronous recursion through `dispatch_inline_call_dr_kind`.
        //
        // The `_r_r` (Ref result) and `_r_i` (Int result) variants share
        // the same `dR` arglist shape; only the dst bank differs.
        "inline_call_r_r/dR>r" => dispatch_inline_call_dr_kind(code, op, ctx, 'r'),
        "inline_call_r_i/dR>i" => dispatch_inline_call_dr_kind(code, op, ctx, 'i'),
        // `_r_v/dR` — void-return variant per `bhimpl_inline_call_r_v`
        // (`blackhole.py`).  Same recursion + arglist as `_r_*`;
        // callee exits via `void_return/`, no SubReturn writeback.
        "inline_call_r_v/dR" => dispatch_inline_call_dr_kind(code, op, ctx, 'v'),
        // `_ir_*` variants extend the arglist to a (I-list, R-list) pair.
        // RPython's `setup_call(argboxes_i, argboxes_r, argboxes_f)` populates
        // both kind banks. The dst bank still selects the SubReturn write
        // target (Ref bank for `_ir_r/dIR>r`, Int bank for `_ir_i/dIR>i`,
        // void no-write for `_ir_v/dIR`).
        "inline_call_ir_r/dIR>r" => dispatch_inline_call_dir_kind(code, op, ctx, 'r'),
        "inline_call_ir_i/dIR>i" => dispatch_inline_call_dir_kind(code, op, ctx, 'i'),
        "inline_call_ir_v/dIR" => dispatch_inline_call_dir_kind(code, op, ctx, 'v'),
        // `_irf_*` variants extend the arglist with a float list (I-list,
        // R-list, F-list). Same `setup_call(argboxes_i, argboxes_r,
        // argboxes_f)` distribution; dst bank chooses Int / Ref / Float
        // for the SubReturn writeback or void no-write for `_irf_v/dIRF`.
        "inline_call_irf_i/dIRF>i" => dispatch_inline_call_dirf_kind(code, op, ctx, 'i'),
        "inline_call_irf_r/dIRF>r" => dispatch_inline_call_dirf_kind(code, op, ctx, 'r'),
        "inline_call_irf_f/dIRF>f" => dispatch_inline_call_dirf_kind(code, op, ctx, 'f'),
        "inline_call_irf_v/dIRF" => dispatch_inline_call_dirf_kind(code, op, ctx, 'v'),
        "goto/L" => {
            // RPython `blackhole.py bhimpl_goto(target): return
            // target`. The 2-byte LE label was resolved by
            // `assembler.fix_labels` to a direct pc; pyre + RPython
            // agree that goto records nothing (pure control flow).
            let target = read_label(code, op, 0);
            Ok((DispatchOutcome::Continue, target))
        }
        "goto_if_not/iL" => {
            // RPython `pyjitpl.py opimpl_goto_if_not`:
            //
            //   @arguments("box", "label", "orgpc")
            //   def opimpl_goto_if_not(self, box, target, orgpc, replace=True):
            //       switchcase = box.getint()
            //       if switchcase:
            //           assert switchcase == 1
            //           opnum = rop.GUARD_TRUE
            //           promoted_box = CONST_1
            //       else:
            //           opnum = rop.GUARD_FALSE
            //           promoted_box = CONST_0
            //       self.metainterp.generate_guard(opnum, box, resumepc=orgpc)
            //       if not switchcase:
            //           self.pc = target
            //       if isinstance(box, Const):
            //           return
            //       if replace:
            //           self.metainterp.replace_box(box, promoted_box)
            //
            // Operand layout `iL`: 1B Int register + 2B LE label.
            // Concrete branch value comes from `TraceCtx::concrete_of_opref`
            // (same path `switch/id` uses); non-concrete OpRefs surface
            // `GotoIfNotValueNotConcrete` rather than guess a direction.
            let valuebox = read_int_reg(code, op, 0, ctx)?;
            let target = read_label(code, op, 1);
            let switchcase = match ctx.trace_ctx.concrete_of_opref(valuebox) {
                Some(Value::Int(v)) => v,
                _ => {
                    if std::env::var("PYRE_DIAG_GIN").is_ok() {
                        let tb = read_int_reg_concrete(code, op, 0, ctx);
                        let reg = code[op.pc + 1] as usize;
                        eprintln!(
                            "[diag-gin] pc={} valuebox={:?} reg={} concrete_of_opref=NON-INT typed_bank={:?}",
                            op.pc, valuebox, reg, tb
                        );
                    }
                    return Err(DispatchError::GotoIfNotValueNotConcrete {
                        pc: op.pc,
                        value: valuebox,
                    });
                }
            };
            goto_if_not_branch_on(code, op, ctx, valuebox, switchcase, target)
        }
        // `pyjitpl.py`'s exec-generated `opimpl_goto_if_not_<cmp>` loop:
        // `condbox = self.execute(rop.<CMP>, b1, b2); self.opimpl_goto_if_not(
        // condbox, target, orgpc, replace=False)`. The fused arm records the
        // same compare and reuses `goto_if_not_branch_on`.
        // The pointer pair reads its branch direction straight out of
        // `_establish_nullity` — no condition op, no GuardTrue/GuardFalse:
        //
        //   def opimpl_goto_if_not_ptr_nonzero(self, box, target, orgpc):
        //       if not self._establish_nullity(box, orgpc):
        //           self.pc = target
        //
        //   def opimpl_goto_if_not_ptr_iszero(self, box, target, orgpc):
        //       if self._establish_nullity(box, orgpc):
        //           self.pc = target
        //
        // Operand layout `rL`: 1B ref reg + 2B label.
        "goto_if_not_ptr_nonzero/rL" | "goto_if_not_ptr_iszero/rL" => {
            let boxref = read_ref_reg(code, op, 0, ctx)?;
            let nonnull = establish_nullity(code, op, ctx, boxref)?;
            let jump = if op.key == "goto_if_not_ptr_nonzero/rL" {
                !nonnull
            } else {
                nonnull
            };
            let target = read_label(code, op, 1);
            Ok((
                DispatchOutcome::Continue,
                if jump { target } else { op.next_pc },
            ))
        }
        // Same shape with one operand.  `pyjitpl.py`
        // `opimpl_goto_if_not_int_is_true` and `opimpl_goto_if_not_int_is_zero`
        // each `self.execute` their unary and hand the fresh condbox to
        // `opimpl_goto_if_not(..., replace=False)`, so each records its own op
        // instead of branching on the value.
        //
        // Only `blackhole.py` aliases the pair
        // (`bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`).  The walker
        // is the metainterp side, where the difference is observable twice
        // over: the plain arm records no `int_is_true`, and its `replace`
        // rewrites the *value* box to a constant -- for the fused form that box
        // is `x`, not the boolean the branch tested.
        //
        // Both spellings reach the stream: `jtransform::optimize_goto_if_not`
        // folds `int_is_true` and `int_is_zero` alike, and the assembler keys
        // each on its own byte.
        "goto_if_not_int_is_true/iL" => {
            fused_goto_if_not_int_unary(code, op, ctx, OpCode::IntIsTrue)
        }
        "goto_if_not_int_is_zero/iL" => {
            fused_goto_if_not_int_unary(code, op, ctx, OpCode::IntIsZero)
        }
        "goto_if_not_int_lt/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntLt),
        "goto_if_not_int_le/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntLe),
        "goto_if_not_int_eq/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntEq),
        "goto_if_not_int_ne/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntNe),
        "goto_if_not_int_gt/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntGt),
        "goto_if_not_int_ge/iiL" => fused_goto_if_not_int(code, op, ctx, OpCode::IntGe),
        "goto_if_not_float_lt/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatLt),
        "goto_if_not_float_le/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatLe),
        "goto_if_not_float_eq/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatEq),
        "goto_if_not_float_ne/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatNe),
        "goto_if_not_float_gt/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatGt),
        "goto_if_not_float_ge/ffL" => fused_goto_if_not_float(code, op, ctx, OpCode::FloatGe),
        "goto_if_not_ptr_eq/rrL" => fused_goto_if_not_ptr(code, op, ctx, OpCode::PtrEq),
        "goto_if_not_ptr_ne/rrL" => fused_goto_if_not_ptr(code, op, ctx, OpCode::PtrNe),
        "int_add_jump_if_ovf/Lii>i" => int_ovf_jump(code, op, ctx, OpCode::IntAddOvf),
        "int_sub_jump_if_ovf/Lii>i" => int_ovf_jump(code, op, ctx, OpCode::IntSubOvf),
        "int_mul_jump_if_ovf/Lii>i" => int_ovf_jump(code, op, ctx, OpCode::IntMulOvf),
        "catch_exception/L" => {
            // RPython `blackhole.py bhimpl_catch_exception(target)` —
            // "no-op when run normally" — and `pyjitpl.py
            // opimpl_catch_exception`:
            //
            //   def opimpl_catch_exception(self, target):
            //       """This is a no-op when run normally.  We can check that
            //       last_exc_value is a null ptr; it should have been set to None
            //       by the previous instruction.  If the previous instruction
            //       raised instead, finishframe_exception() should have been
            //       called and we would not be there."""
            //       assert not self.metainterp.last_exc_value
            //
            // The 2-byte target is metadata: when a `raise` fires on the
            // previous instruction, `handle_exception_in_frame`
            // (`blackhole.py`) reads it to redirect the unwinder
            // (consumed by `try_catch_exception_at` from the inline_call
            // SubRaise arm). Linear walk advances past the operand
            // without using the target.
            //
            // The RPython assert turns into a typed error here:
            // reaching `catch_exception/L` with `ctx.last_exc_value =
            // Some(_)` means either (a) the codewriter emitted a
            // catch_exception/L outside an exception-table position,
            // or (b) a previous catch handler didn't clear
            // last_exc_value after handling the raise. Either is a
            // codewriter-pass invariant violation.
            if ctx.last_exc_value().is_some() {
                return Err(DispatchError::CatchExceptionWithActiveException { pc: op.pc });
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "switch/id" => dispatch_switch_id(code, op, ctx),
        "residual_call_r_r/iRd>r" => dispatch_residual_call_iRd_kind(code, op, ctx, 'r'),
        // `_r_i/iRd>i` mirrors `_r_r/iRd>r` with the dst kind flipped to
        // int. RPython `pyjitpl.py _opimpl_residual_call1` is
        // exec-generated for `_callR` and `_callI` (Type::Ref vs
        // Type::Int return) — see resoperation.py `Type::Int =>
        // CallI`. EffectInfo classification + GUARD_NO_EXCEPTION
        // emission stay identical; only the result OpCode (CallI /
        // CallPureI) and dst writeback bank (registers_i) differ.
        "residual_call_r_i/iRd>i" => dispatch_residual_call_iRd_kind(code, op, ctx, 'i'),
        // `_ir_*/iIRd>X` extends the arglist with an i-bank list. RPython
        // `pyjitpl.py:_opimpl_residual_call*` exec-generates this for
        // callees taking both int + ref args (setup_call distributes
        // (argboxes_i, argboxes_r, argboxes_f=[])). Same EffectInfo
        // classification path; the only operand-shape change is the
        // I-list prefix between funcptr and the R-list.
        "residual_call_ir_r/iIRd>r" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'r'),
        "residual_call_ir_i/iIRd>i" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'i'),
        // `_irf_*/iIRFd>X` extends `_ir_*` with an f-bank list before the
        // descr (`pyjitpl.py _opimpl_residual_call3`, `boxes3`
        // argcode `pyjitpl.py`). EffectInfo classification +
        // guard emission identical; only the operand layout adds the F
        // suffix list.
        "residual_call_irf_r/iIRFd>r" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'r'),
        "residual_call_irf_i/iIRFd>i" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'i'),
        "residual_call_irf_f/iIRFd>f" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'f'),
        // `_*_v/iRd|iIRd|iIRFd` void variants — `_opimpl_residual_call*`
        // bodies discard the call result for void return kinds
        // (`pyjitpl.py opimpl_residual_call_r_v = _opimpl_residual_call1`,
        // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
        // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`;
        // `blackhole.py bhimpl_residual_call_*_v`).
        // EffectInfo classification + guard emission match the result-typed
        // siblings; only the operand layout drops the `>X` dst byte and
        // the writeback no-ops via `write_residual_call_result_to_dst`'s
        // `'v'` arm. `select_residual_call_opcode`'s `'v'` arm maps to the
        // `CallN` / `CallPureN` / `CallMayForceN` / `CallLoopinvariantN`
        // family per `resoperation.py Type::Void => CallN`.
        "residual_call_r_v/iRd" => dispatch_residual_call_iRd_kind(code, op, ctx, 'v'),
        "residual_call_ir_v/iIRd" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'v'),
        // `pyjitpl.py opimpl_conditional_call_ir_v`: condition + func + I/R.
        "conditional_call_ir_v/iiIRd" => dispatch_conditional_call_ir_v(code, op, ctx),
        // `pyjitpl.py _opimpl_conditional_call_value` / `do_conditional_call(is_value=True)`.
        "conditional_call_value_ir_i/iiIRd>i" => {
            dispatch_conditional_call_value_ir_i(code, op, ctx)
        }
        "conditional_call_value_ir_r/riIRd>r" => {
            dispatch_conditional_call_value_ir_r(code, op, ctx)
        }
        // `pyjitpl.py opimpl_record_known_result_i_ir_v` records only.
        "record_known_result_i_ir_v/iiIRd" => dispatch_record_known_result_i_ir_v(code, op, ctx),
        "record_known_result_r_ir_v/riIRd" => dispatch_record_known_result_r_ir_v(code, op, ctx),
        "residual_call_irf_v/iIRFd" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'v'),
        // The `int_*` / `float_*` / `ptr_*` record families are routed
        // through `dispatch_regular_record` (see `arith.rs`) before this
        // match, so their arms no longer appear here.
        // `int_between/iii>i` decomposes a 3-arg range check at record
        // time per `pyjitpl.py opimpl_int_between` into
        // `INT_SUB + (INT_EQ on ConstInt(1) fast path | INT_SUB +
        // UINT_LT generic)`. Surfaces in `make_ll_isinstance`'s
        // range-covering branch (`rclass.py`).
        "int_between/iii>i" => int_between_record(code, op, ctx),
        // `int_floordiv/ii>i` and `int_mod/ii>i` intentionally absent:
        // `jtransform.py` rewrites both to
        // `direct_call(ll_int_py_*)` before jitcode emission.  The
        // trace-front lowering at `majit-translate/src/codegen.rs`
        // mirrors that rewrite for code reaching the JIT trace, so
        // this walker is never asked to dispatch the bare ops on a
        // traceable path.  Build-time helper graphs that still emit
        // the bare ops (e.g. `pyre/pyre-interpreter/src/baseobjspace.rs`
        // long_mod / long_div until the build-pipeline jtransform
        // port lands) get a `setdefault`-allocated dynamic byte and
        // resolve through BH dispatch only.
        // `cast_float_to_int` / `cast_int_to_float` / `cast_int_to_ptr` /
        // `cast_ptr_to_int`
        // route through `dispatch_regular_record` (see `arith.rs`
        // `unop_cast_record`) — part of the `pyjitpl.py`
        // exec-generated unary family.
        "ptr_nonzero/r>i" => ptr_nullity_record(code, op, ctx, true),
        "ptr_iszero/r>i" => ptr_nullity_record(code, op, ctx, false),
        "guard_class/r>i" => guard_class_record(code, op, ctx, 'i'),
        "guard_class/r>r" => guard_class_record(code, op, ctx, 'r'),
        "int_guard_value/i" => guard_value_record(code, op, ctx, GuardValueBank::Int),
        "ref_guard_value/r" => guard_value_record(code, op, ctx, GuardValueBank::Ref),
        "float_guard_value/f" => guard_value_record(code, op, ctx, GuardValueBank::Float),
        "abort/>r" | "abort/>i" => {
            // `Assembler::encode_op` retains the result byte when an
            // untranslatable operation still owns an SSA result.  It remains
            // an abort: advancing would leave that destination unwritten and
            // let the trace consume a stale box.  Match the blackhole result
            // handlers and the result-less `abort/` arm below.
            Err(DispatchError::AbortMarkerReached { pc: op.pc })
        }
        "abort/" => {
            // pyre-only `BC_ABORT` marker (`handler_abort_marker`,
            // `blackhole.rs`): the front-end emitted a graph node
            // with no dedicated `OpKind`, so reaching it means the body
            // carries an untranslatable op. The trace cannot be recorded;
            // surface a recoverable abort (production driver →
            // `TraceAction::Abort`).
            Err(DispatchError::AbortMarkerReached { pc: op.pc })
        }
        "abort_permanent/" => {
            // A generator-resume sub-walk arriving at the marker its own
            // `yield` lowered to is not aborting: that marker is the
            // suspension, and the value it suspends with is the FOR_ITER item
            // the resume was walked for.
            if let Some(item) = generator_resume_yield(ctx, op.pc)? {
                return Ok((
                    DispatchOutcome::SubReturn { result: Some(item) },
                    op.next_pc,
                ));
            }
            // `YIELD_VALUE` on any walk except the generator-resume sub-walk
            // is a frame exit (`dispatch`'s `except Yield`), the same FINISH
            // `ref_return/r` records. The frame stays suspended: do not set
            // `FLAG_FRAME_FINISHED`.
            if let Some((yielded, yield_py_pc)) = portal_yield_frame_exit(ctx, op.pc) {
                if !yielded.is_constant() {
                    if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) =
                        ctx.trace_ctx.concrete_of_opref(yielded)
                    {
                        if ptr != 0 {
                            ctx.trace_ctx.set_opref_concrete(
                                yielded,
                                majit_ir::Value::Ref(majit_ir::GcRef(ptr)),
                            );
                        }
                    }
                }
                if ctx.is_top_level {
                    if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) =
                        ctx.trace_ctx.concrete_of_opref(yielded)
                    {
                        if ptr != 0 {
                            fbw_finish_concrete_set(ConcreteValue::Ref(
                                ptr as pyre_object::PyObjectRef,
                            ));
                        }
                    }
                    let (finish_arg, finish_arg_type) =
                        fbw_terminate_with_finish(ctx, yielded, op.pc)?;
                    // After the finish's own `last_instr` store, so the hint
                    // flushes the yield coordinate and the popped depth.
                    flush_yield_exit(ctx, yield_py_pc);
                    return Ok((
                        DispatchOutcome::Terminate {
                            finish_arg,
                            finish_arg_type,
                        },
                        op.next_pc,
                    ));
                }
                return Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(yielded),
                    },
                    op.next_pc,
                ));
            }
            // pyre-only `BC_ABORT_PERMANENT` fail-path
            // (`bhimpl_abort_permanent`, `blackhole.rs`): emitted for
            // paths that must always terminate the frame (BigInt-overflow
            // / unported-op fallbacks). Surface a permanent abort
            // (production driver → `TraceAction::AbortPermanent`) so the
            // location is never traced again.
            //
            // Capture the inline-sub-walk state here: an `op.pc` reached
            // inside an inlined callee is a callee coordinate the outer
            // walk's py_pc→jitcode tables cannot resolve, so the abort-point
            // flush must
            // decline. A sub-walk raise leaves `abort_in_subwalk` for
            // `claim_abort_coordinate`; the root overwrites a leftover true
            // so the driver cannot inherit a nested walk's flag.
            if !ctx.fbw_mode.inline_subwalk {
                ctx.session.borrow_mut().abort_in_subwalk = false;
            }
            Err(DispatchError::AbortPermanentMarkerReached { pc: op.pc })
        }
        // `pyjitpl.py opimpl_getfield_raw_i`: `execute_with_descr(
        // GETFIELD_RAW_I, fielddescr, box)`. Operand layout `id>i`:
        // 1B int base + 2B descr + 1B dest. The base is a raw address
        // in the int bank (`blackhole.py bhimpl_getfield_raw_i`).
        "getfield_raw_i/id>i" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let descr = read_descr(code, op, 1, ctx)?;
            let concrete = match (
                read_int_reg_concrete(code, op, 0, ctx),
                descr.as_field_descr(),
            ) {
                (ConcreteValue::Int(ptr), Some(_)) if ptr != 0 => ctx
                    .trace_ctx
                    .field_sanity_load(ptr, &descr, majit_ir::Type::Int)
                    .and_then(|v| match v {
                        majit_ir::Value::Int(n) => Some(ConcreteValue::Int(n)),
                        _ => None,
                    })
                    .unwrap_or(ConcreteValue::Null),
                _ => ConcreteValue::Null,
            };
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::GetfieldRawI, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::GetfieldRawI,
                majit_metainterp::counters::RECORDED_OPS,
            );
            let result = ctx
                .trace_ctx
                .record_op_with_descr(OpCode::GetfieldRawI, &[base], descr);
            if let ConcreteValue::Int(value) = concrete {
                ctx.trace_ctx.set_opref_concrete(result, Value::Int(value));
            }
            let dst = code[op.pc + 4] as usize;
            write_int_reg(ctx, op.pc, dst, result, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // pyjitpl.py `opimpl_getfield_raw_f`: `execute_with_descr(
        // GETFIELD_RAW_F, fielddescr, box)`. Operand layout `id>f`.
        "getfield_raw_f/id>f" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let descr = read_descr(code, op, 1, ctx)?;
            let concrete = match (
                read_int_reg_concrete(code, op, 0, ctx),
                descr.as_field_descr(),
            ) {
                (ConcreteValue::Int(ptr), Some(_)) if ptr != 0 => ctx
                    .trace_ctx
                    .field_sanity_load(ptr, &descr, majit_ir::Type::Float)
                    .and_then(|v| match v {
                        majit_ir::Value::Float(n) => Some(ConcreteValue::Float(n)),
                        _ => None,
                    })
                    .unwrap_or(ConcreteValue::Null),
                _ => ConcreteValue::Null,
            };
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::GetfieldRawF, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::GetfieldRawF,
                majit_metainterp::counters::RECORDED_OPS,
            );
            let result = ctx
                .trace_ctx
                .record_op_with_descr(OpCode::GetfieldRawF, &[base], descr);
            if let ConcreteValue::Float(value) = concrete {
                ctx.trace_ctx
                    .set_opref_concrete(result, Value::Float(value));
            }
            let dst = code[op.pc + 4] as usize;
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // pyjitpl.py `_opimpl_setfield_raw_any` / `opimpl_setfield_raw_i`:
        // `execute_with_descr(rop.SETFIELD_RAW, fielddescr, box, valuebox)`.
        // `blackhole.py bhimpl_setfield_raw_i` (`iid`). The walk is the
        // execution, same as `raw_store_i`: a non-concrete operand declines
        // rather than continue past a lost write. No heapcache.
        "setfield_raw_i/iid" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let value = read_int_reg(code, op, 1, ctx)?;
            let descr = read_descr(code, op, 2, ctx)?;
            match (
                read_int_reg_concrete(code, op, 0, ctx),
                read_int_reg_concrete(code, op, 1, ctx),
                descr.as_field_descr(),
            ) {
                (ConcreteValue::Int(ptr), ConcreteValue::Int(v), Some(fd)) if ptr != 0 => {
                    let addr = (ptr as usize).wrapping_add(fd.offset()) as *mut u8;
                    // SAFETY: the walk executed the residuals that produced
                    // `ptr`; `fd.offset()` / `fd.field_size()` are the field
                    // descr (`bhimpl_setfield_raw_i`).
                    unsafe { raw_store_int(addr, fd.field_size(), v) };
                }
                (ConcreteValue::Int(0), _, Some(_)) => {}
                _ => {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: "setfield_raw_i/iid non-concrete operand",
                    });
                }
            }
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::SetfieldRaw, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::SetfieldRaw,
                majit_metainterp::counters::RECORDED_OPS,
            );
            ctx.trace_ctx.heapcache_invalidate_caches_varargs(
                OpCode::SetfieldRaw,
                None,
                &[base, value],
            );
            ctx.trace_ctx
                .record_op_with_descr(OpCode::SetfieldRaw, &[base, value], descr);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // pyjitpl.py `opimpl_setfield_raw_f` / `blackhole.py
        // bhimpl_setfield_raw_f` (`ifd`).
        "setfield_raw_f/ifd" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let value = read_float_reg(code, op, 1, ctx)?;
            let descr = read_descr(code, op, 2, ctx)?;
            match (
                read_int_reg_concrete(code, op, 0, ctx),
                read_float_reg_concrete(code, op, 1, ctx),
                descr.as_field_descr(),
            ) {
                (ConcreteValue::Int(ptr), ConcreteValue::Float(v), Some(fd)) if ptr != 0 => {
                    let addr = (ptr as usize).wrapping_add(fd.offset()) as *mut u8;
                    unsafe { raw_store_float(addr, fd.field_size(), v) };
                }
                (ConcreteValue::Int(0), _, Some(_)) => {}
                _ => {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: "setfield_raw_f/ifd non-concrete operand",
                    });
                }
            }
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::SetfieldRaw, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::SetfieldRaw,
                majit_metainterp::counters::RECORDED_OPS,
            );
            ctx.trace_ctx.heapcache_invalidate_caches_varargs(
                OpCode::SetfieldRaw,
                None,
                &[base, value],
            );
            ctx.trace_ctx
                .record_op_with_descr(OpCode::SetfieldRaw, &[base, value], descr);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Heapcache-aware getfield reads. RPython
        // `pyjitpl.py opimpl_getfield_gc_<i|r|f>` →
        // `_opimpl_getfield_gc_any_pureornot` (`pyjitpl.py`)
        // dispatches the same way through `heapcache.get_field_updater`.
        // Walker handles the canonical `rd>X` shapes (Ref source);
        // pyre-specific `id>X` variants where the source is an int
        // register holding an unwrapped pointer are kind-flow kind-flow
        // territory and stay unsupported here.
        "getfield_gc_i/rd>i" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcI, 'i'),
        "getfield_gc_r/rd>r" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcR, 'r'),
        "getfield_gc_f/rd>f" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcF, 'f'),
        // RPython aliases the `_pure` jitcode spelling to the plain
        // handler (`pyjitpl.py opimpl_getfield_gc_i_pure =
        // opimpl_getfield_gc_i`), so the recorded opcode stays plain
        // and purity is re-derived from `descr.is_always_pure()` by
        // the optimizer (`heap.py:641` const fold + invalidation-exempt
        // field cache).
        "getfield_gc_i_pure/rd>i" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcI, 'i')
        }
        "getfield_gc_r_pure/rd>r" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcR, 'r')
        }
        "getfield_gc_f_pure/rd>f" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcF, 'f')
        }
        // Virtualizable getfield reads. RPython
        // `pyjitpl.py opimpl_getfield_vable_{i,r,f}` —
        // walker delegates to `TraceCtx::vable_getfield_{int,ref,float}`
        // which already implements `_nonstandard_virtualizable` fallback
        // to GETFIELD_GC + standard-vable `virtualizable_boxes[index]`
        // cache read (`majit-metainterp/src/trace_ctx.rs`).
        // Same `rd>X` operand shape as `getfield_gc_*`; only the
        // semantic handler routes through the vable mirror.
        "getfield_vable_i/rd>i" => getfield_vable_via_metainterp(code, op, ctx, 'i'),
        "getfield_vable_r/rd>r" => getfield_vable_via_metainterp(code, op, ctx, 'r'),
        "getfield_vable_f/rd>f" => getfield_vable_via_metainterp(code, op, ctx, 'f'),
        // Virtualizable setfield writes. RPython
        // `pyjitpl.py _opimpl_setfield_vable` — walker
        // delegates to `TraceCtx::vable_setfield`
        // (`majit-metainterp/src/trace_ctx.rs`) which handles the
        // `_nonstandard_virtualizable` fallback to SETFIELD_GC + the
        // standard-vable `virtualizable_boxes[index] = valuebox` +
        // `synchronize_virtualizable` mirror.  Operand shapes:
        // `setfield_vable_i/rid`, `setfield_vable_r/rrd`,
        // `setfield_vable_f/rfd` — value bank differs, no dst byte.
        "setfield_vable_i/rid" => setfield_vable_via_metainterp(code, op, ctx, 'i'),
        "setfield_vable_i_imm/rddd" => setfield_vable_int_imm(code, op, ctx),
        "setfield_vable_r/rrd" => setfield_vable_via_metainterp(code, op, ctx, 'r'),
        "setfield_vable_f/rfd" => setfield_vable_via_metainterp(code, op, ctx, 'f'),
        // Virtualizable array reads/writes + length. RPython
        // `pyjitpl.py _opimpl_{get,set}arrayitem_vable` /
        // `opimpl_arraylen_vable` — walker delegates to the
        // `TraceCtx::vable_{get,set}arrayitem_*` / `vable_arraylen_vable`
        // ports which already implement the `_nonstandard_virtualizable`
        // GC fallback and the standard-vable `virtualizable_boxes[index]`
        // cache path.  The `(VableArray, Array)` descr pair is resolved to
        // the vinfo's identity-keyed `(fdescr, adescr)` via
        // `vable_array_descrs_from_jitcode`.
        "getarrayitem_vable_i/ridd>i" => getarrayitem_vable_via_metainterp(code, op, ctx, 'i'),
        "getarrayitem_vable_r/ridd>r" => getarrayitem_vable_via_metainterp(code, op, ctx, 'r'),
        "getarrayitem_vable_f/ridd>f" => getarrayitem_vable_via_metainterp(code, op, ctx, 'f'),
        "setarrayitem_vable_i/riidd" => setarrayitem_vable_via_metainterp(code, op, ctx, 'i'),
        "setarrayitem_vable_r/rirdd" => setarrayitem_vable_via_metainterp(code, op, ctx, 'r'),
        "setarrayitem_vable_f/rifdd" => setarrayitem_vable_via_metainterp(code, op, ctx, 'f'),
        "arraylen_vable/rdd>i" => arraylen_vable_via_metainterp(code, op, ctx),
        // RPython `pyjitpl.py opimpl_hint_force_virtualizable(box)` is a thin
        // forward:
        //
        //   self.metainterp.gen_store_back_in_vable(box)
        //
        // `TraceCtx::gen_store_back_in_vable` hosts the whole body, including
        // the nonstandard-virtualizable and already-forced gates, so the arm
        // reads its one operand and hands it over. Operand layout `r`.
        "hint_force_virtualizable/r" => {
            let vable = read_ref_reg(code, op, 0, ctx)?;
            ctx.trace_ctx.gen_store_back_in_vable(vable);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `jtransform.py rewrite_op_getfield` emits this ahead of a
        // quasi-immutable getfield. Operand layout `rdd`: the struct, the
        // field descr, and the mutate descr. The mutate descr is unused;
        // `quasi_immut_descr` resolves the watcher from the field descr.
        // The opcode has no result register.
        "record_quasiimmut_field/rdd" => {
            let obj = read_ref_reg(code, op, 0, ctx)?;
            let descr = read_descr(code, op, 1, ctx)?;
            let _ = crate::state::record_quasiimmut_field(ctx.trace_ctx, obj, descr);
            walker_flush_guard_not_invalidated(ctx, op.pc)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // setfield_gc canonical shapes. `iid` / `ird` (int box)
        // shapes are pyre kind-flow kind-flow territory and stay
        // unsupported.
        "setfield_gc_i/rid" => setfield_gc_via_heapcache(code, op, ctx, 'i'),
        // USE_C_FORM short value (`assembler.py`): the stored
        // int is an inline signed byte read as a `ConstInt` box.
        "setfield_gc_i/rcd" => setfield_gc_via_heapcache(code, op, ctx, 'c'),
        "setfield_gc_r/rrd" => setfield_gc_via_heapcache(code, op, ctx, 'r'),
        "setfield_gc_f/rfd" => setfield_gc_via_heapcache(code, op, ctx, 'f'),
        // Raw memory carries no heapcache bookkeeping: where the `_gc_`
        // family consults and updates the cache, `pyjitpl.py`'s raw pair
        // only records into the cache-free path.
        //
        //   def opimpl_raw_load_i(self, addrbox, offsetbox, arraydescr):
        //       return self.execute_with_descr(rop.RAW_LOAD_I, arraydescr,
        //                                      addrbox, offsetbox)
        //
        // `execute_with_descr` executes the load as well as recording it,
        // and the walk is the execution: the address was produced by
        // residual calls the walk ran natively, so when both operands carry
        // concretes the load is performed against the live memory and the
        // result shadow gets the loaded word.  Without concretes the result
        // stays `Null` and readers decline instead of consuming a value the
        // walker did not observe.
        // Operand layout `iid>i`: 1B base + 1B offset + 2B descr + 1B dst.
        "raw_load_i/iid>i" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let offset = read_int_reg(code, op, 1, ctx)?;
            let descr = read_descr(code, op, 2, ctx)?;
            let concrete = match (
                read_int_reg_concrete(code, op, 0, ctx),
                read_int_reg_concrete(code, op, 1, ctx),
                descr.as_array_descr(),
            ) {
                (ConcreteValue::Int(b), ConcreteValue::Int(o), Some(ad)) => {
                    let addr = b.wrapping_add(o) as usize as *const u8;
                    // SAFETY: the walk already executed the residuals that
                    // produced `addr`, so it names live raw memory of at
                    // least `item_size` bytes.
                    let value = unsafe { raw_load_int(addr, ad.item_size(), ad.is_item_signed()) };
                    ConcreteValue::Int(value)
                }
                _ => ConcreteValue::Null,
            };
            // pyjitpl.py `opimpl_raw_load_i`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawLoadI, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawLoadI, majit_metainterp::counters::RECORDED_OPS);
            let result =
                ctx.trace_ctx
                    .record_op_with_descr(OpCode::RawLoadI, &[base, offset], descr);
            // `pyjitpl.py MIFrame.execute_and_record` / `_record_helper`
            // pass the executed `resvalue` into `history.record2`, so the
            // result box carries `box.value`. Stamp the same channel:
            // `uint_ge` / `goto_if_not` read `concrete_of_opref`, not only
            // the Int-bank slot shadow.
            if let ConcreteValue::Int(value) = concrete {
                ctx.trace_ctx.set_opref_concrete(result, Value::Int(value));
            }
            let dst = code[op.pc + 5] as usize;
            write_int_reg(ctx, op.pc, dst, result, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Float twin: `opimpl_raw_load_f`.  The Float bank carries no
        // concrete shadow, so this arm records without executing; a float
        // consumer that needs the concrete value declines downstream.
        // Operand layout `iid>f`: 1B base + 1B offset + 2B descr + 1B dst.
        "raw_load_f/iid>f" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let offset = read_int_reg(code, op, 1, ctx)?;
            let descr = read_descr(code, op, 2, ctx)?;
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawLoadF, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawLoadF, majit_metainterp::counters::RECORDED_OPS);
            let result =
                ctx.trace_ctx
                    .record_op_with_descr(OpCode::RawLoadF, &[base, offset], descr);
            let dst = code[op.pc + 5] as usize;
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `_opimpl_raw_store` delegates to `execute_raw_store`:
        //
        //   self.execute_and_record(rop.RAW_STORE, arraydescr,
        //                           addrbox, offsetbox, valuebox)
        //
        // `execute_and_record` performs the store as well as recording it,
        // and the walk is the execution: skipping the store would hand the
        // rest of the walked body a buffer the program never filled.  A
        // non-concrete operand therefore declines the walk instead of
        // continuing past a lost store.  No cache update, no result
        // register.
        // Operand layout `iiid`: 1B base + 1B offset + 1B value + 2B descr.
        "raw_store_i/iiid" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let offset = read_int_reg(code, op, 1, ctx)?;
            let value = read_int_reg(code, op, 2, ctx)?;
            let descr = read_descr(code, op, 3, ctx)?;
            match (
                read_int_reg_concrete(code, op, 0, ctx),
                read_int_reg_concrete(code, op, 1, ctx),
                read_int_reg_concrete(code, op, 2, ctx),
                descr.as_array_descr(),
            ) {
                (ConcreteValue::Int(b), ConcreteValue::Int(o), ConcreteValue::Int(v), Some(ad)) => {
                    let addr = b.wrapping_add(o) as usize as *mut u8;
                    // SAFETY: same contract as the raw_load arm — the walk
                    // executed the residuals that produced `addr`.
                    unsafe { raw_store_int(addr, ad.item_size(), v) };
                }
                _ => {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: "raw_store_i/iiid non-concrete operand",
                    });
                }
            }
            // pyjitpl.py `_opimpl_raw_store`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawStore, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawStore, majit_metainterp::counters::RECORDED_OPS);
            ctx.trace_ctx
                .record_op_with_descr(OpCode::RawStore, &[base, offset, value], descr);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Float twin of `raw_store_i`, under the same execute-as-you-walk
        // contract: the walk performs the store, so a non-concrete operand
        // declines rather than continue past a lost write.  The value's
        // concrete f64 comes from the Float bank's OpRef carrier.
        // Operand layout `iifd`: 1B base + 1B offset + 1B value + 2B descr.
        "raw_store_f/iifd" => {
            let base = read_int_reg(code, op, 0, ctx)?;
            let offset = read_int_reg(code, op, 1, ctx)?;
            let value = read_float_reg(code, op, 2, ctx)?;
            let descr = read_descr(code, op, 3, ctx)?;
            match (
                read_int_reg_concrete(code, op, 0, ctx),
                read_int_reg_concrete(code, op, 1, ctx),
                read_float_reg_concrete(code, op, 2, ctx),
                descr.as_array_descr(),
            ) {
                (
                    ConcreteValue::Int(b),
                    ConcreteValue::Int(o),
                    ConcreteValue::Float(v),
                    Some(ad),
                ) => {
                    let addr = b.wrapping_add(o) as usize as *mut u8;
                    // SAFETY: same contract as the raw_store_i arm — the walk
                    // executed the residuals that produced `addr`.
                    unsafe { raw_store_float(addr, ad.item_size(), v) };
                }
                _ => {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: "raw_store_f/iifd non-concrete operand",
                    });
                }
            }
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawStore, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::RawStore, majit_metainterp::counters::RECORDED_OPS);
            ctx.trace_ctx
                .record_op_with_descr(OpCode::RawStore, &[base, offset, value], descr);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Heapcache-aware array reads/writes (canonical `rid>X` /
        // `ri{i,r,f}d` shapes).  Array indices are always int-classified,
        // so the index operand is always decoded from the `i` register
        // bank; non-canonical Ref-index shapes (`rrd>r`/`rrrd`/`rrfd`)
        // never arise.
        "getarrayitem_gc_i/rid>i" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcI, 'i')
        }
        "getarrayitem_gc_r/rid>r" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcR, 'r')
        }
        "getarrayitem_gc_f/rid>f" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcF, 'f')
        }
        // RPython `opimpl_getarrayitem_gc_{i,f,r}_pure`
        // — distinct opimpls (NOT aliased to the non-pure form,
        // unlike `getfield_gc_*_pure`).
        // Records `rop.GETARRAYITEM_GC_PURE_{I,F,R}` directly through
        // `_do_getarrayitem_gc_any(rop.GETARRAYITEM_GC_PURE_*, ...)`.
        // The ConstPtr+ConstInt constant-fold fast path requires BOTH the
        // array box and the index box to be `Const`; on the walker they
        // are generally runtime OpRefs, so the fold precondition is unmet
        // and the miss branch records the Pure rop.
        "getarrayitem_gc_i_pure/rid>i" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureI, 'i')
        }
        "getarrayitem_gc_r_pure/rid>r" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureR, 'r')
        }
        "getarrayitem_gc_f_pure/rid>f" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureF, 'f')
        }
        "setarrayitem_gc_i/riid" => setarrayitem_gc_via_heapcache(code, op, ctx, 'i'),
        // const-VALUE `c`-argcode form (USE_C_FORM `assembler.py`):
        // the store value is one inline signed byte → ConstInt, decoded
        // inside the handler.  Mirror of `setfield_gc_i/rcd`.
        "setarrayitem_gc_i/ricd" => setarrayitem_gc_via_heapcache(code, op, ctx, 'c'),
        "setarrayitem_gc_r/rird" => setarrayitem_gc_via_heapcache(code, op, ctx, 'r'),
        // `c`-argcode form (`assembler.py emit_const(allow_short
        // =True)`, USE_C_FORM `assembler.py`): the index is one
        // inline signed byte → ConstInt.  Same body as
        // [`setarrayitem_gc_via_heapcache`] with the index read from
        // the bytecode instead of an i-bank register.
        "setarrayitem_gc_r/rcrd" => {
            let array = read_ref_reg(code, op, 0, ctx)?;
            let index = OpRef::ConstInt(code[op.pc + 2] as i8 as i64);
            let value = read_ref_reg(code, op, 2, ctx)?;
            let descr = read_descr(code, op, 3, ctx)?;
            let descr_index = descr.index();
            // pyjitpl.py `_opimpl_setarrayitem_gc_any`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::SetarrayitemGc,
                majit_metainterp::counters::RECORDED_OPS,
            );
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetarrayitemGc,
                &[array, index, value],
                descr.clone(),
            );
            ctx.trace_ctx
                .heapcache_setarrayitem(array, index, descr_index, value);
            walker_execute_setarrayitem_gc(code, op, ctx, 'r', array, index, value, &descr)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "setarrayitem_gc_f/rifd" => setarrayitem_gc_via_heapcache(code, op, ctx, 'f'),
        // pyjitpl.py `_opimpl_setarrayitem_raw_any` /
        // `opimpl_setarrayitem_raw_i`:
        // `execute_with_descr(rop.SETARRAYITEM_RAW, arraydescr, arraybox,
        // indexbox, itembox)`. `blackhole.py bhimpl_setarrayitem_raw_i`
        // (`iiid`); `llmodel.py bh_setarrayitem_raw_i = bh_setarrayitem_gc_i`.
        // `setarrayitem_raw_i` is not in `USE_C_FORM` (`assembler.py`), so
        // the encoding is always registers. The walk is the execution,
        // same as `raw_store_i`: a non-concrete operand declines rather
        // than continue past a lost write. No heapcache.setarrayitem —
        // `_record_helper` `invalidate_caches` is a no-op via
        // `heapcache.py clear_caches_not_necessary` for SETARRAYITEM_RAW.
        "setarrayitem_raw_i/iiid" => {
            let array = read_int_reg(code, op, 0, ctx)?;
            let index = read_int_reg(code, op, 1, ctx)?;
            let value = read_int_reg(code, op, 2, ctx)?;
            let descr = read_descr(code, op, 3, ctx)?;
            match (
                read_int_reg_concrete(code, op, 0, ctx),
                read_int_reg_concrete(code, op, 1, ctx),
                read_int_reg_concrete(code, op, 2, ctx),
                descr.as_array_descr(),
            ) {
                (
                    ConcreteValue::Int(array_ptr),
                    ConcreteValue::Int(idx),
                    ConcreteValue::Int(v),
                    Some(ad),
                ) => {
                    let item_addr = (array_ptr as usize)
                        .wrapping_add(ad.base_size())
                        .wrapping_add((idx as usize).wrapping_mul(ad.item_size()));
                    // SAFETY: same contract as the raw_store arm — the walk
                    // executed the residuals that produced `array_ptr`;
                    // `base_size` / `item_size` are the array descr
                    // (`bhimpl_setarrayitem_raw_i` /
                    // `bh_setarrayitem_gc_i`).
                    unsafe { raw_store_int(item_addr as *mut u8, ad.item_size(), v) };
                }
                _ => {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: "setarrayitem_raw_i/iiid non-concrete operand",
                    });
                }
            }
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::SetarrayitemRaw, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::SetarrayitemRaw,
                majit_metainterp::counters::RECORDED_OPS,
            );
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetarrayitemRaw,
                &[array, index, value],
                descr,
            );
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `pyjitpl.py opimpl_getinteriorfield_gc_{i,r,f}` /
        // `execute_setinteriorfield_gc`. Same operand layouts as the
        // array-item arms; the descr is the interior field.
        "getinteriorfield_gc_i/rid>i" => {
            getinteriorfield_gc_via_heapcache(code, op, ctx, OpCode::GetinteriorfieldGcI, 'i')
        }
        "getinteriorfield_gc_r/rid>r" => {
            getinteriorfield_gc_via_heapcache(code, op, ctx, OpCode::GetinteriorfieldGcR, 'r')
        }
        "getinteriorfield_gc_f/rid>f" => {
            getinteriorfield_gc_via_heapcache(code, op, ctx, OpCode::GetinteriorfieldGcF, 'f')
        }
        "setinteriorfield_gc_i/riid" => setinteriorfield_gc_via_heapcache(code, op, ctx, 'i'),
        "setinteriorfield_gc_r/rird" => setinteriorfield_gc_via_heapcache(code, op, ctx, 'r'),
        "setinteriorfield_gc_f/rifd" => setinteriorfield_gc_via_heapcache(code, op, ctx, 'f'),
        // `arraylen_gc` — `blackhole.py bhimpl_arraylen_gc`
        // (@arguments("cpu","r","d", returns="i")).  Operand layout `rd>i`:
        // 1B r-reg(array) + 2B descr + 1B i-reg(dst).  Delegates to
        // `state::opimpl_arraylen_gc` (heapcache-aware length tracking,
        // pyjitpl.py).
        "arraylen_gc/rd>i" => {
            let array = read_ref_reg(code, op, 0, ctx)?;
            let descr = read_descr(code, op, 1, ctx)?;
            let result = crate::state::opimpl_arraylen_gc(&mut ctx.trace_ctx, array, descr);
            let dst = code[op.pc + 4] as usize;
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `pyjitpl.py opimpl_strlen` / `opimpl_strgetitem` —
        // `return self.execute(rop.STRLEN, strbox)` /
        // `return self.execute(rop.STRGETITEM, strbox, indexbox)`.
        // Operand layouts match `blackhole.py bhimpl_strlen` /
        // `bhimpl_strgetitem` (`r>i` / `ri>i` / `rc>i`).
        "strlen/r>i" => opimpl_strlen(code, op, ctx),
        "strgetitem/ri>i" | "strgetitem/rc>i" => opimpl_strgetitem(code, op, ctx),
        // `pyjitpl.py opimpl_newstr` / `opimpl_strsetitem` /
        // `opimpl_copystrcontent` — `self.execute(rop.NEWSTR / STRSETITEM /
        // COPYSTRCONTENT, ...)`, with the `blackhole.py` operand layouts.
        "newstr/i>r" | "newstr/c>r" => opimpl_newstr(code, op, ctx),
        "strsetitem/rii" | "strsetitem/rci" | "strsetitem/ric" | "strsetitem/rcc" => {
            opimpl_strsetitem(code, op, ctx)
        }
        "copystrcontent/rriii" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rrcii" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rrici" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rriic" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rrcci" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rrcic" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rricc" => opimpl_copystrcontent(code, op, ctx),
        "copystrcontent/rrccc" => opimpl_copystrcontent(code, op, ctx),
        // RPython `pyjitpl.py opimpl_assert_not_none` and
        // `opimpl_record_exact_class` are heapcache hints: consult the
        // cache, record once, then stamp what the record proved. Both
        // bodies already live on `TraceCtx` — the walker reads the
        // operands and delegates, so the two tracers cannot drift.
        //
        // Operand layout `r`: 1B ref reg.
        "assert_not_none/r" => {
            let opref = read_ref_reg(code, op, 0, ctx)?;
            let known_nonnull = ctx.trace_ctx.heapcache_nullity_known(opref) == Some(true);
            let concrete = if known_nonnull {
                0
            } else {
                // `trace_assert_not_none` reaches `executor.py
                // do_assert_not_none` on a cache miss, so only that path needs
                // the pointer value. An absent shadow cannot prove non-nullness.
                let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) else {
                    return Err(DispatchError::UnsupportedOpname {
                        pc: op.pc,
                        key: op.key,
                    });
                };
                ptr as i64
            };
            ctx.trace_ctx.trace_assert_not_none(opref, concrete);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Operand layout `ri`: 1B ref reg + 1B int reg holding the class
        // vtable address. A non-constant class operand is skipped inside
        // the helper, matching the `isinstance(clsbox, Const)` gate.
        "record_exact_class/ri" => {
            let opref = read_ref_reg(code, op, 0, ctx)?;
            let cls = read_int_reg(code, op, 1, ctx)?;
            ctx.trace_ctx.trace_record_exact_class(opref, cls);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `pyjitpl.py` `_opimpl_isconstant` / `_opimpl_isvirtual`: write
        // `ConstInt(...)` and record no IR.  Operand layout `[src][dst]`.
        "int_isconstant/i>i" => {
            let src = read_int_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            write_hint_bool(ctx, op.pc, dst, src.is_constant())?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_isconstant/r>i" => {
            let src = read_ref_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            write_hint_bool(ctx, op.pc, dst, src.is_constant())?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_isvirtual/r>i" => {
            let src = read_ref_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            write_hint_bool(ctx, op.pc, dst, ctx.trace_ctx.is_likely_virtual(src))?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_new` delegates to `execute_new`:
        //
        //   resbox = self.execute_and_record(rop.NEW, typedescr)
        //   self.heapcache.new(resbox)
        //   return resbox
        //
        // Same `d>r` layout and same posture as `new_with_vtable` below,
        // minus the class stamp — a plain struct allocation carries no
        // vtable word, so nothing is known about its class.
        "new/d>r" => {
            let descr = read_descr(code, op, 0, ctx)?;
            // Every header-writing allocation must carry a tid in the active
            // collector's type table; tracing a fabricated header records
            // garbage that the concrete and compiled allocation paths share.
            if !ctx.trace_ctx.new_allocation_tid_is_sound(&descr) {
                return Err(DispatchError::UnregisteredNewGcType {
                    pc: op.pc,
                    type_id: descr
                        .as_size_descr()
                        .map_or(0, |size| size.type_id() as u64),
                });
            }
            let concrete = ctx.trace_ctx.execute_new_allocation(&descr, false);
            // The `set_opref_concrete` below is what roots this object: it
            // stamps the allocation onto the recorded op's `value` cell, which
            // `MetaInterp::walk_active_trace_refs` forwards.  Nothing between
            // here and there allocates from the GC heap, so no collection can
            // observe the object before it is reachable from that root.
            // pyjitpl.py `execute_new`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::New, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::New, majit_metainterp::counters::RECORDED_OPS);
            let resbox = ctx.trace_ctx.record_op_with_descr(OpCode::New, &[], descr);
            ctx.trace_ctx.heap_cache_mut().new_object(resbox);
            let dst = code[op.pc + 3] as usize;
            if let Some(value) = concrete {
                ctx.trace_ctx.set_opref_concrete(resbox, value);
            }
            let concrete = match concrete {
                Some(Value::Ref(majit_ir::GcRef(ptr))) => {
                    ConcreteValue::Ref(ptr as pyre_object::PyObjectRef)
                }
                _ => ConcreteValue::Null,
            };
            write_ref_reg(ctx, op.pc, dst, resbox, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_new_with_vtable` delegates straight to
        // `execute_new_with_vtable`:
        //
        //   resbox = self.execute_and_record(rop.NEW_WITH_VTABLE, descr)
        //   self.heapcache.new(resbox)
        //   self.heapcache.class_now_known(resbox)
        //   return resbox
        //
        // Operand layout `d>r`: 2B descr + 1B dst (`assembler.rs`
        // `new_with_vtable` pushes the descr u16 then the result reg).
        "new_with_vtable/d>r" => {
            let descr = read_descr(code, op, 0, ctx)?;
            if !ctx.trace_ctx.new_allocation_tid_is_sound(&descr) {
                return Err(DispatchError::UnregisteredNewGcType {
                    pc: op.pc,
                    type_id: descr
                        .as_size_descr()
                        .map_or(0, |size| size.type_id() as u64),
                });
            }
            let concrete = ctx.trace_ctx.execute_new_allocation(&descr, true);
            // Rooted by the `set_opref_concrete` stamp below, as in `new/d>r`.
            if let Some(Value::Ref(majit_ir::GcRef(ptr))) = concrete
                && let Some(w_class) = descr.as_size_descr().and_then(|size| size.w_class_obj())
            {
                unsafe {
                    (*(ptr as *mut pyre_object::PyObject)).w_class =
                        w_class as pyre_object::PyObjectRef;
                }
            }
            // pyjitpl.py `execute_new_with_vtable`.
            ctx.trace_ctx
                .profiler()
                .count_ops(OpCode::NewWithVtable, majit_metainterp::counters::OPS);
            ctx.trace_ctx.profiler().count_ops(
                OpCode::NewWithVtable,
                majit_metainterp::counters::RECORDED_OPS,
            );
            let resbox =
                ctx.trace_ctx
                    .record_op_with_descr(OpCode::NewWithVtable, &[], descr.clone());
            ctx.trace_ctx.heap_cache_mut().new_object(resbox);
            crate::helpers::note_class_word_after_new(ctx.trace_ctx, resbox, &descr);
            let dst = code[op.pc + 3] as usize;
            if let Some(value) = concrete {
                ctx.trace_ctx.set_opref_concrete(resbox, value);
            }
            let concrete = match concrete {
                Some(Value::Ref(majit_ir::GcRef(ptr))) => {
                    ConcreteValue::Ref(ptr as pyre_object::PyObjectRef)
                }
                _ => ConcreteValue::Null,
            };
            write_ref_reg(ctx, op.pc, dst, resbox, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // RPython `pyjitpl.py opimpl_new_array` /
        // `opimpl_new_array_clear` — both are
        // `_opimpl_new_array(rop.NEW_ARRAY{,_CLEAR}, lengthbox,
        // arraydescr)`, which records that opnum and seeds
        // `heapcache.new_array(resbox, lengthbox)`. Operand layout
        // per `bhimpl_new_array{,_clear} @arguments("cpu","i","d",
        // returns="r")`: 1B i-reg(length) + 2B descr + 1B r-reg(dst).
        // The `cd>r` arm is the `c`-argcode form (inline signed-byte
        // length, USE_C_FORM): identical byte layout, length decoded
        // as ConstInt. `new_array/cd>r` is not in the encoding table.
        "new_array/id>r" | "new_array_clear/id>r" | "new_array_clear/cd>r" => {
            let clear = op.key.starts_with("new_array_clear/");
            let opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            let length = if op.key == "new_array_clear/cd>r" {
                OpRef::ConstInt(code[op.pc + 1] as i8 as i64)
            } else {
                read_int_reg(code, op, 0, ctx)?
            };
            let descr = read_descr(code, op, 1, ctx)?;
            let is_ref_array = descr
                .as_array_descr()
                .is_some_and(|a| a.is_array_of_pointers() && !a.is_array_of_structs());
            let is_float_array = descr
                .as_array_descr()
                .is_some_and(|a| a.is_array_of_floats());
            // `ordereddict.malloc_i64_entries` → `new_array_clear` of
            // `GcArray(Entry)`. `_overallocate_entries_len(0)` is a
            // computed register (`baselen + (baselen >> 3) + 8`), so the
            // length is known from `Box.value` without being a jitcode
            // const. `heapcache.new_array` below still virtualizes only
            // when `length.is_constant()`. `tid` is `ArrayDescr.tid`
            // (`gc.py` `init_array_descr` / `cpu.arraydescrof`).
            let cleared_struct_array = if clear {
                descr.as_array_descr().and_then(|array| {
                    let item_size = array.item_size();
                    let items_base = array.base_size();
                    let tid = array.type_id();
                    let matches_header = array.is_array_of_structs() && array.len_descr().is_some();
                    matches_header.then_some((tid, item_size, items_base))
                })
            } else {
                None
            };
            // pyjitpl.py `_opimpl_new_array`.
            ctx.trace_ctx
                .profiler()
                .count_ops(opcode, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(opcode, majit_metainterp::counters::RECORDED_OPS);
            let resbox = ctx
                .trace_ctx
                .record_op_with_descr(opcode, &[length], descr.clone());
            // heapcache.py `new_array(box, lengthbox)` adds
            // the virtual/unescaped flags only when `lengthbox` is a
            // Const ("only constant-length arrays are virtuals").
            ctx.trace_ctx
                .heap_cache_mut()
                .new_array(resbox, length, length.is_constant());
            let dst = code[op.pc + 4] as usize;
            // Walker virtual-force (module-global fresh-container off-by-one
            // fix): for a ref array whose live length is known, eagerly
            // allocate a real GC-traced block and stamp it as the
            // recording-time concrete VALUE (`Op.value` — GC-rooted via
            // `walk_op_const_ptr_refs`, NOT `make_constant`, so the compiled
            // trace still allocates fresh per iteration; same posture as an
            // executed residual's observed result). A later BUILD_LIST /
            // BUILD_TUPLE residual, or a void `OS_ARRAYCOPY` into this dest
            // (`tuple_ll_listslice_startstop`), then resolves a concrete
            // array arg and runs during the walk instead of aborting
            // `ResidualCallArgUnbound`. heapcache still virtualises only a
            // Const length ("only constant-length arrays are virtuals");
            // the stamp follows the live `box_value`, so a slicelength
            // that is a known Int from `slice_adjust_indices` but not a
            // Const still binds the dest register. The companion
            // `walker_fill_materialized_array` fills slots at `setarrayitem_gc`
            // on that stamped NewArray / NewArrayClear block, including when
            // the length is a known Int that is not a Const (heapcache has
            // then marked the array escaped); if any element/index is
            // non-concrete it reverts the array to the no-concrete sentinel
            // so the residual declines (abort, as before).
            // An unknown length keeps the Null posture. Signed/Float
            // arrays stamp a typed items block so `OS_ARRAYCOPY` can run;
            // `NEW_ARRAY_CLEAR` zeroes it (`bh_new_array_clear`) so
            // `getarrayitem_gc_i` reads `FREE` (0).
            let mut concrete = ConcreteValue::Null;
            if let Some((tid, item_size, items_base)) = cleared_struct_array {
                // `cd>r` encodes the length in the opcode byte. Reading
                // that byte as a register index aliases another slot.
                let from_register = op.key != "new_array_clear/cd>r";
                if let Some(n) =
                    heapcache_ops::known_int_operand(code, op, 0, length, ctx, from_register)
                        .filter(|&n| majit_ir::ptr_info::reasonable_array_index(n))
                    && let Some(block) = heapcache_ops::materialize_cleared_struct_gcarray(
                        n as usize, tid, item_size, items_base,
                    )
                    && ctx.trace_ctx.try_set_opref_concrete(
                        resbox,
                        majit_ir::Value::Ref(majit_ir::GcRef(block as usize)),
                    )
                {
                    concrete = ConcreteValue::Ref(block as pyre_object::PyObjectRef);
                }
            } else if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(length) {
                if let Ok(cap) = usize::try_from(n) {
                    // Ref arrays: BUILD_LIST / tuple slice. Signed/Float
                    // arrays: IntegerListStrategy `newlist` (`int_ll_newlist`
                    // → `new_array`). A later void `OS_ARRAYCOPY` into that
                    // dest (`ll_listslice_new_int_list`) needs a concrete
                    // block or the walk aborts `ResidualCallArgUnbound`.
                    // `bh_new_array` (`llmodel.py`) passes the live
                    // `arraydescr` to `gc_malloc_array`. Item width and
                    // type id come from that descr. A layout this path
                    // cannot represent leaves the result unbound (decline
                    // the concrete step) rather than allocating a wrong
                    // integer block. Zero length is a valid empty array.
                    let block: Option<*mut u8> = if is_ref_array {
                        unsafe {
                            pyre_object::object_array::alloc_cleared_ref_items_block_gc(cap)
                                .map(|p| p as *mut u8)
                        }
                    } else if let Some(ad) = descr.as_array_descr() {
                        if ad.is_array_of_structs() || ad.item_size() != std::mem::size_of::<u64>()
                        {
                            None
                        } else {
                            // Analyzer `get_array_descr` leaves
                            // `UNSET_GC_TYPE_ID` until
                            // `register_unresolved_array_tids` (`gc.py`
                            // `init_array_descr`). Fall back to the host
                            // `GcArray(Signed)` / `GcArray(Float)` tid,
                            // the same way mint `0` does.
                            let raw = ad.type_id();
                            let tid = if raw != 0 && !majit_ir::descr::array_tid_is_unresolved(raw)
                            {
                                raw
                            } else if is_float_array {
                                pyre_object::object_array::gc_float_array_gc_type_id()
                            } else {
                                pyre_object::object_array::gc_int_array_gc_type_id()
                            };
                            if tid == 0 || majit_ir::descr::array_tid_is_unresolved(tid) {
                                None
                            } else {
                                let p = unsafe {
                                    pyre_object::object_array::alloc_typed_items_block_nursery(
                                        cap, tid,
                                    )
                                };
                                if p.is_null() {
                                    None
                                } else {
                                    // `alloc_typed_items_block_nursery` stores
                                    // `cap.max(1)` (other callers keep that).
                                    // `bh_new_array` / `execute_new_array_clear`
                                    // use the requested length; 0 stays 0.
                                    // `ll_alloc_and_set` zeros via
                                    // `typed_items_block_clear`.
                                    unsafe {
                                        if clear {
                                            pyre_object::object_array::typed_items_block_clear(p);
                                        }
                                        (*p).capacity = cap;
                                    }
                                    Some(p as *mut u8)
                                }
                            }
                        }
                    } else {
                        None
                    };
                    if let Some(block) = block {
                        if ctx.trace_ctx.try_set_opref_concrete(
                            resbox,
                            majit_ir::Value::Ref(majit_ir::GcRef(block as usize)),
                        ) {
                            concrete = ConcreteValue::Ref(block as pyre_object::PyObjectRef);
                        }
                    }
                }
            }
            write_ref_reg(ctx, op.pc, dst, resbox, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_copy/i>i" => {
            // RPython `pyjitpl.py _opimpl_any_copy(self, box) → box`
            // + `@arguments("box")` + `>i` result coding: read src
            // register, write the same OpRef into the dst slot. Pypy
            // records *no* IR op for a copy — pure SSA-level rename.
            // Operand layout `i>i`: 1B src + 1B dst.
            //
            // Int-bank concrete shadow: propagate the source slot's Int-bank concrete
            // shadow alongside the symbolic OpRef, mirroring the
            // `ref_copy/r>r` concrete-shadow chain.  Without this, a
            // `goto_if_not/iL` reading the dst slot wouldn't see the
            // concrete and would surface `GotoIfNotValueNotConcrete`
            // even when the source had a known concrete (e.g. a
            // constant Int seeded by `allocate_callee_register_banks`).
            let src_val = read_int_reg(code, op, 0, ctx)?;
            let src_concrete = read_int_reg_concrete(code, op, 0, ctx);
            let dst = code[op.pc + 2] as usize;
            write_int_reg(ctx, op.pc, dst, src_val, src_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_copy/c>i" => {
            // `int_copy/c>i` — USE_C_FORM short source (`assembler.py`):
            // the small ConstInt is one inline signed byte (`signedord`,
            // `blackhole.py`), not a `registers_i` slot. Like `int_copy/i>i`
            // this records no IR op (pure SSA copy); the dst is seeded with the
            // constant box plus its concrete shadow so a downstream
            // `goto_if_not/iL` / `switch/id` can fold. Operand layout `c>i`:
            // 1B signed const + 1B dst.
            let value = code[op.pc + 1] as i8 as i64;
            let dst = code[op.pc + 2] as usize;
            write_int_reg(
                ctx,
                op.pc,
                dst,
                OpRef::ConstInt(value),
                ConcreteValue::Int(value),
            )?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_copy/f>f" => {
            // Float-bank sibling of `int_copy/i>i` — pure SSA-level
            // rename, no IR op recorded. Operand layout `f>f`: 1B src
            // + 1B dst.
            let src_val = read_float_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, src_val);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_push/r" => {
            // Blackhole `bhimpl_ref_push` (`blackhole.py`):
            // `self.tmpreg_r = a`.  `insert_renamings` (`flatten.py`)
            // emits `*_push`/`*_pop` around a cyclic parallel move — the
            // swap `r_a <-> r_b` lowers to `push r_b; copy r_b<-r_a;
            // pop r_a` so the overwritten value survives in the tmpreg.
            // Pure SSA-level scratch move, no IR op recorded.  Operand
            // layout `r`: 1B src.
            let src_val = read_ref_reg(code, op, 0, ctx)?;
            let src_concrete = read_ref_reg_concrete(code, op, 0, ctx);
            {
                let mut sess = ctx.session.borrow_mut();
                sess.tmpreg_r = src_val;
                sess.tmpreg_r_concrete = src_concrete;
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_pop/>r" => {
            // Blackhole `bhimpl_ref_pop` (`blackhole.py`):
            // `return self.get_tmpreg_r()`.  Reads the value stashed by
            // the matching `ref_push/r` back into a dst register, in
            // lock-step with its concrete shadow.  Operand layout `>r`:
            // 1B dst.
            let (val, concrete) = {
                let mut sess = ctx.session.borrow_mut();
                let val = sess.tmpreg_r;
                let concrete = sess.tmpreg_r_concrete;
                sess.tmpreg_r = OpRef::NONE;
                sess.tmpreg_r_concrete = ConcreteValue::Null;
                (val, concrete)
            };
            let dst = code[op.pc + 1] as usize;
            write_renaming_ref_reg(ctx, op.pc, dst, val, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_push/i" => {
            // Blackhole `bhimpl_int_push` (`blackhole.py`):
            // `self.tmpreg_i = a`.  Int-bank sibling of `ref_push/r`,
            // stashing the concrete shadow alongside the OpRef.
            // Operand layout `i`: 1B src.
            let src_val = read_int_reg(code, op, 0, ctx)?;
            let src_concrete = read_int_reg_concrete(code, op, 0, ctx);
            {
                let mut sess = ctx.session.borrow_mut();
                sess.tmpreg_i = src_val;
                sess.tmpreg_i_concrete = src_concrete;
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_pop/>i" => {
            // Blackhole `bhimpl_int_pop` (`blackhole.py`).  Writes through
            // `write_int_reg` so `concrete_registers_i[dst]` tracks the value
            // the move restores instead of the one it overwrote.
            // Operand layout `>i`: 1B dst.
            let (val, concrete) = {
                let sess = ctx.session.borrow();
                (sess.tmpreg_i, sess.tmpreg_i_concrete)
            };
            let dst = code[op.pc + 1] as usize;
            write_int_reg(ctx, op.pc, dst, val, concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_push/f" => {
            // Blackhole `bhimpl_float_push` (`blackhole.py`).
            // Operand layout `f`: 1B src.
            let src_val = read_float_reg(code, op, 0, ctx)?;
            ctx.session.borrow_mut().tmpreg_f = src_val;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_pop/>f" => {
            // Blackhole `bhimpl_float_pop` (`blackhole.py`).  The Float bank
            // resolves concretes from the OpRef itself
            // (`read_float_reg_concrete`) rather than a side table, so moving
            // the OpRef carries the concrete with it.
            // Operand layout `>f`: 1B dst.
            let val = ctx.session.borrow().tmpreg_f;
            let dst = code[op.pc + 1] as usize;
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, val);
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_copy/r>r" => {
            // Ref-bank sibling of `int_copy/i>i`. Same RPython
            // `_opimpl_any_copy` body — the `>r` suffix only changes
            // which register bank the writeback lands in. Const-source
            // variants (codewriter `emit_ref_copy!` with `ConstRef`)
            // resolve via the constants window of `registers_r`: the
            // assembler's `load_const_r` patches the src operand to a
            // constants-pool register index in `[num_regs_r,
            // num_regs_and_consts_r)`, which `setposition` (RPython
            // `pyjitpl.py`) pre-populates with the const OpRef.
            // No IR op recorded.
            let src_val = read_ref_reg(code, op, 0, ctx)?;
            // Propagate the source slot's concrete
            // shadow alongside the symbolic OpRef.  This is the
            // critical chain: catch_exception → seeds last_exc_value
            // / concrete → `last_exc_value/>r` writes both into
            // `registers_r[X]` and `concrete_registers_r[X]` →
            // `ref_copy/r>r` copies X to Y → a follow-on `raise/r`
            // reads Y and finds the correct concrete to emit
            // GUARD_CLASS against.  Without this propagation the
            // copy chain wipes the concrete and silently disables
            // the guard.
            let src_concrete = read_ref_reg_concrete(code, op, 0, ctx);
            let dst = code[op.pc + 2] as usize;
            // A copy is a link renaming (`flatten.py insert_renamings`,
            // `same_as`). Restoring TOS after the write is what keeps
            // GET_ITER's iterator from being replaced by a later
            // `output = ''` block-exit copy.
            write_renaming_ref_reg(ctx, op.pc, dst, src_val, src_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_return/r" => {
            // RPython `pyjitpl.py:opimpl_ref_return(self, value)` calls
            // `metainterp.finishframe(value)`. Two branches by frame depth:
            //
            //   * Outermost frame → `compile_done_with_this_frame` (pyjitpl.py)
            //     records `rop.FINISH(value)` with
            //     `done_with_this_frame_descr_ref`. Trace ends.
            //   * Nested frame → `metainterp.popframe()` returns control to
            //     the caller's metainterp loop with `value` in hand; the
            //     caller's `_opimpl_inline_call*` lands `value` in its
            //     `>r` slot via `make_result_of_lastop`.
            //
            // Walker selects between the two via `ctx.is_top_level`.
            let result = read_ref_reg(code, op, 0, ctx)?;
            finish_current_frame_execution(ctx, op.pc);
            // PyPy `box.value = result` parity at the frame boundary:
            // the callee's slot-keyed concrete shadow (`concrete_registers_r`)
            // carries the live PyObject pointer; mirror it onto the
            // OpRef-keyed `opref_concrete` channel so the caller's
            // `concrete_from_recorded_opref` (in `dispatch_inline_call_*_kind`)
            // sees the stamped Box.value.  Skips constants — `TraceCtx::constants
            // .get_value` is the authoritative shadow for those.
            if !result.is_constant() {
                if let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) {
                    if !ptr.is_null() {
                        ctx.trace_ctx.set_opref_concrete(
                            result,
                            majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)),
                        );
                    }
                }
            }
            if ctx.is_top_level {
                // Slice b: route the loop-free portal exit through
                // `TraceAction::Finish` so the compile pipeline records
                // the FINISH from `finish_args`.  Re-box to Type::Ref +
                // store_token_in_vable, then stash the payload; do NOT
                // call `ctx.trace_ctx.finish()` here (would double-record).
                //
                // No-replay portal exit: also stash the CONCRETE return
                // so `eval.rs` returns the walk's result directly instead
                // of re-running the compiled trace against the already
                // side-effected heap (the walk consumed it).  A null /
                // unknown concrete leaves the cell `None` → degrade.
                if let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) {
                    if !ptr.is_null() {
                        fbw_finish_concrete_set(ConcreteValue::Ref(ptr));
                    }
                }
                let (finish_arg, finish_arg_type) = fbw_terminate_with_finish(ctx, result, op.pc)?;
                commit_top_level_frame_finished(ctx);
                Ok((
                    DispatchOutcome::Terminate {
                        finish_arg,
                        finish_arg_type,
                    },
                    op.next_pc,
                ))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "int_return/i" => {
            // RPython `pyjitpl.py opimpl_int_return = _opimpl_any_return`
            // (pyjitpl.py `_opimpl_any_return: self.metainterp.finishframe(box)`).
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py)
            // records `FINISH([value], descr=done_with_this_frame_descr_int)`.
            // Sub-walk: `SubReturn { Some(value) }` — caller's
            // `inline_call_*_i` would land the int OpRef in its `>i` slot.
            // Operand layout `i`: 1B int register at op.pc+1.
            let result = read_int_reg(code, op, 0, ctx)?;
            finish_current_frame_execution(ctx, op.pc);
            // PyPy `box.value = result` parity at the frame boundary —
            // see `ref_return/r` comment above for rationale.
            if !result.is_constant() {
                if let ConcreteValue::Int(v) = read_int_reg_concrete(code, op, 0, ctx) {
                    ctx.trace_ctx
                        .set_opref_concrete(result, majit_ir::Value::Int(v));
                }
            }
            if ctx.is_top_level {
                // Slice b: portal-exit FINISH carries Type::Ref even for
                // an int return (the eval_loop_jit result_type is REF),
                // so `fbw_ensure_boxed_for_ca` re-boxes via wrapint.
                //
                // No-replay portal exit: stash the concrete int so
                // `eval.rs` returns the walk's result directly (re-boxed
                // via `ConcreteValue::to_pyobj`).
                if let ConcreteValue::Int(v) = read_int_reg_concrete(code, op, 0, ctx) {
                    fbw_finish_concrete_set(ConcreteValue::Int(v));
                }
                let (finish_arg, finish_arg_type) = fbw_terminate_with_finish(ctx, result, op.pc)?;
                commit_top_level_frame_finished(ctx);
                Ok((
                    DispatchOutcome::Terminate {
                        finish_arg,
                        finish_arg_type,
                    },
                    op.next_pc,
                ))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "int_return/c" => {
            // USE_C_FORM short source (`assembler.py`): the return value
            // is one inline signed byte (`signedord`, `blackhole.py`),
            // not a `registers_i` slot — so `result` is a `ConstInt` that
            // already carries its value (no `set_opref_concrete` needed).
            // Otherwise identical to `int_return/i`. Operand layout `c`:
            // 1B signed const at op.pc+1.
            let value = code[op.pc + 1] as i8 as i64;
            let result = OpRef::ConstInt(value);
            finish_current_frame_execution(ctx, op.pc);
            if ctx.is_top_level {
                fbw_finish_concrete_set(ConcreteValue::Int(value));
                let (finish_arg, finish_arg_type) = fbw_terminate_with_finish(ctx, result, op.pc)?;
                commit_top_level_frame_finished(ctx);
                Ok((
                    DispatchOutcome::Terminate {
                        finish_arg,
                        finish_arg_type,
                    },
                    op.next_pc,
                ))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "float_return/f" => {
            // RPython `pyjitpl.py opimpl_float_return = _opimpl_any_return`.
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py)
            // records `FINISH([value], descr=done_with_this_frame_descr_float)`.
            // Sub-walk: `SubReturn { Some(value) }` carrying the float
            // OpRef — same enum variant as int/ref because the OpRef is
            // bank-agnostic; the caller's inline_call variant decides
            // which bank to write into.
            // Operand layout `f`: 1B float register at op.pc+1.
            let result = read_float_reg(code, op, 0, ctx)?;
            finish_current_frame_execution(ctx, op.pc);
            if ctx.is_top_level {
                // Slice b: portal-exit FINISH carries Type::Ref;
                // `fbw_ensure_boxed_for_ca` re-boxes the float via
                // wrapfloat.
                if let Some(majit_ir::Value::Float(v)) = ctx.trace_ctx.box_value(result) {
                    fbw_finish_concrete_set(ConcreteValue::Float(v));
                }
                let (finish_arg, finish_arg_type) = fbw_terminate_with_finish(ctx, result, op.pc)?;
                commit_top_level_frame_finished(ctx);
                Ok((
                    DispatchOutcome::Terminate {
                        finish_arg,
                        finish_arg_type,
                    },
                    op.next_pc,
                ))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "void_return/" => {
            // RPython `pyjitpl.py opimpl_void_return`:
            //
            //   @arguments()
            //   def opimpl_void_return(self):
            //       self.metainterp.finishframe(None)
            //
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py)
            // takes the `result_type == VOID` branch — `exits = []`,
            // `token = sd.done_with_this_frame_descr_void`. The FINISH
            // carries no value.
            // Sub-walk: `SubReturn { None }` — RPython's
            // `_opimpl_inline_call_*_v` variants don't write a dst
            // register on the caller side (the codewriter emits no `>X`
            // marker for void calls).
            // No operand bytes (the `/` argcodes is empty).
            finish_current_frame_execution(ctx, op.pc);
            if ctx.is_top_level {
                // Slice b: route the void portal exit through
                // `TraceAction::Finish` (empty args) so the compile
                // pipeline records the FINISH(void) from `finish_args`,
                // mirroring the three value-returning arms.  Store the
                // assembler token in the vable + GUARD_NOT_FORCED_2 like
                // those arms, then stash a void-marked payload; do NOT
                // call `ctx.trace_ctx.finish()` here (would double-record).
                //
                // No-replay portal exit: stash `Null` (= void → None at
                // the consume site) so a side-effecting void function
                // returns directly instead of re-running its already
                // applied effects.
                fbw_finish_concrete_set(ConcreteValue::Null);
                let (finish_arg, finish_arg_type) = fbw_terminate_void_with_finish(ctx, op.pc)?;
                commit_top_level_frame_finished(ctx);
                Ok((
                    DispatchOutcome::Terminate {
                        finish_arg,
                        finish_arg_type,
                    },
                    op.next_pc,
                ))
            } else {
                Ok((DispatchOutcome::SubReturn { result: None }, op.next_pc))
            }
        }
        "raise/r" => {
            // RPython `pyjitpl.py opimpl_raise(exc_value_box, orgpc)`:
            //   if not self.metainterp.heapcache.is_class_known(exc_value_box):
            //       clsbox = self.cls_of_box(exc_value_box)
            //       self.metainterp.generate_guard(rop.GUARD_CLASS, exc_value_box,
            //                                      clsbox, resumepc=orgpc)
            //   self.metainterp.class_of_last_exc_is_const = True
            //   self.metainterp.last_exc_value = exc_value_box.getref(...)
            //   self.metainterp.last_exc_box = exc_value_box
            //   self.metainterp.popframe()
            //   self.metainterp.finishframe_exception()
            //
            // Walker dual behaviour:
            //   * `is_top_level` → outermost FINISH (above).
            //   * sub-walk frame → propagate `SubRaise { exc }` to the
            //     caller's `inline_call_*` handler.
            //
            // GUARD_CLASS emission is enabled: `concrete_registers_r`
            // is `&mut` and every walker write goes through
            // [`write_ref_reg`], so the concrete tracks the symbolic
            // in lock-step.  Were the shadow an immutable dispatch-entry
            // snapshot, sibling handlers could rewrite `registers_r[dst]`
            // without updating it, so this read would find a stale
            // concrete and silently skip the guard.  Read-after-write
            // now returns the right concrete (or `Null` if the handler
            // didn't know, in which case the guard skips — same
            // semantics as the snapshot's tail).
            //
            // Mirrors the retired trait-side raise path.  The read at
            // `ob_header.ob_type` is the layout vtable `allocate_instance`
            // stamps. `_getusercls` shares one vtable across `_new_exception`
            // classes, so `GuardClass` pins that layout.
            // Stashes the concrete into `ctx.last_exc_value_concrete()`
            // so a downstream
            // `last_exc_value/>r` can propagate it into its dst slot.
            let exc = read_ref_reg(code, op, 0, ctx)?;
            let mut concrete_exc = read_ref_reg_concrete(code, op, 0, ctx);
            if matches!(concrete_exc, ConcreteValue::Ref(p) if p.is_null())
                && let Some(Value::Ref(gc_ref)) = ctx.trace_ctx.box_value(exc)
                && gc_ref != majit_ir::GcRef::NO_CONCRETE
            {
                let ptr = gc_ref.as_usize() as pyre_object::PyObjectRef;
                if !ptr.is_null() && unsafe { pyre_object::is_exception(ptr) } {
                    concrete_exc = ConcreteValue::Ref(ptr);
                }
            }
            // `pyjitpl.py opimpl_raise` calls
            // `generate_guard(GUARD_CLASS, exc_value_box, clsbox,
            // resumepc=orgpc)`; the first line of `generate_guard`
            // (`pyjitpl.py`) is `if isinstance(box, Const):
            // return`. Const exception boxes already pin the class so
            // no guard is needed. Where the guard IS emitted below, it
            // captures its resume data at this opcode's pc through
            // `walker_capture_snapshot_for_last_guard`, matching
            // `generate_guard`'s `resumepc=orgpc`.
            if !exc.is_constant() {
                if let ConcreteValue::Ref(exc_ptr) = concrete_exc {
                    if !exc_ptr.is_null() && !ctx.trace_ctx.heap_cache().is_class_known(exc) {
                        let exc_class_ptr = unsafe {
                            (*(exc_ptr as *const pyre_object::interp_exceptions::W_BaseException))
                                .ob_header
                                .ob_type
                        };
                        let cls_const = ctx.trace_ctx.const_int(exc_class_ptr as usize as i64);
                        ctx.trace_ctx
                            .record_guard(OpCode::GuardClass, &[exc, cls_const], 0);
                        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
                        ctx.trace_ctx.heap_cache_mut().class_now_known(exc);
                    }
                }
            }
            ctx.set_last_exc_value(exc, concrete_exc);
            ctx.fbw_mode.class_of_last_exc_is_const = true;
            let freshly_normalized = fbw_built_exc_take(exc);
            if !recording_raise_keeps_existing_traceback(ctx, op.pc) {
                let caught_in_frame = try_catch_exception_at(code, op.next_pc).is_some();
                if caught_in_frame {
                    // Unless this walk built the exception it may already
                    // carry callee nodes, so the general case prepends onto
                    // its chain rather than starting a fresh one.  Either
                    // recorder declining leaves the node to the opaque hook,
                    // which builds a frame from the callee's code + globals
                    // for the level the walk has no box for.  That frame
                    // carries no locals and its own identity, so it is a
                    // weaker node than the live one — still a node whose
                    // `tb_frame.f_code` answers the right code object.
                    let node = if freshly_normalized {
                        record_fresh_application_traceback(ctx, exc, concrete_exc, op.pc)?
                    } else {
                        record_prepend_application_traceback(ctx, exc, concrete_exc, op.pc)?
                    };
                    let emit_runtime = node.is_none();
                    record_inline_application_traceback(
                        ctx,
                        exc,
                        &mut concrete_exc,
                        op.pc,
                        false,
                        emit_runtime,
                        node,
                    );
                    record_top_level_application_traceback(
                        ctx,
                        exc,
                        &mut concrete_exc,
                        op.pc,
                        false,
                        emit_runtime,
                        node,
                    );
                }
            }
            // Route the top-level raise through `SubRaise` so walk()'s
            // SubRaise arm runs the in-frame
            // `catch_exception/L` lookahead (`finishframe_lookahead_at`) and
            // jumps into the handler instead of recording the top-level
            // exit-frame finish (which escapes a try/except as a no-payload
            // Terminate abort).
            Ok((
                DispatchOutcome::SubRaise {
                    exc,
                    exc_concrete: concrete_exc,
                },
                op.next_pc,
            ))
        }
        "last_exc_value/>r" => {
            // RPython parity: `pyjitpl.py opimpl_last_exc_value`:
            //
            //   @arguments()
            //   def opimpl_last_exc_value(self):
            //       exc_value = self.metainterp.last_exc_value
            //       assert exc_value
            //       return self.metainterp.last_exc_box
            //
            // Reads no operand; the `>r` decorator writes the result into
            // `registers_r[dst]`. No IR op recorded — the standing
            // `metainterp.last_exc_box` (mirrored here as
            // `ctx.last_exc_value()`) is already a recorder OpRef from when
            // `raise/r` set it. This is a pure SSA-rename of the
            // exception slot into a Ref-bank dst, mirroring how
            // `int_copy/i>i` and `_opimpl_any_copy` collapse to a
            // register move without recording.
            //
            // Operand layout `>r`: 1B dst register only (the `>r` arg is
            // the writeback marker, not a separate operand byte; the dst
            // byte sits at op.pc+1).
            //
            // Forward-prep status: the opname is registered in
            // `wire_handler("last_exc_value/>r", handler_last_exc_value)`
            // (`blackhole.rs`) and `m.insert("last_exc_value/>r",
            // BC_LAST_EXC_VALUE)` (`jitcode/mod.rs`), but pyre's
            // codewriter does not currently emit `FlatOp::LastExcValue`
            // for any traced Python arm — `dump_unsupported_opnames_in_insns_table`
            // confirms the opname is absent from `OUT_DIR/insns.bin`.
            // The handler matches RPython's unconditional `setup_insns`
            // registration so it's ready when an except-handler arm
            // (e.g. `BC_LAST_EXC_VALUE` consumer in CPython 3.14
            // `LOAD_SPECIAL`/`CHECK_EXC_MATCH` lowering) lands.
            let exc = ctx
                .last_exc_value()
                .ok_or(DispatchError::LastExcValueWithoutActiveException { pc: op.pc })?;
            let dst = code[op.pc + 1] as usize;
            // Propagate the standing exception's
            // concrete shadow into the dst slot.  `ctx.last_exc_value_
            // concrete` is the live `PyObjectRef` (seeded by either an
            // adapter caller or an earlier walker `raise/r`). This lets a follow-on `raise/r` reading
            // `registers_r[dst]` find a non-Null concrete and emit the
            // correct GUARD_CLASS.
            let exc_concrete = ctx.last_exc_value_concrete();
            write_ref_reg(ctx, op.pc, dst, exc, exc_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "last_exception/>i" => {
            // RPython parity: `pyjitpl.py opimpl_last_exception`:
            //
            //   @arguments()
            //   def opimpl_last_exception(self):
            //       exc_value = self.metainterp.last_exc_value
            //       assert exc_value
            //       assert self.metainterp.class_of_last_exc_is_const
            //       exc_cls = rclass.ll_cast_to_object(exc_value).typeptr
            //       return ConstInt(ptr2int(exc_cls))
            //
            // The class pointer is the standing exception's `ob_type`.
            // `read_typeptr_from_exception` (`dispatch.rs`) routes
            // through `cpu.cls_of_box`. Walker reads the live concrete
            // exception's `W_BaseException.ob_header.ob_type` directly. The result is
            // a `ConstInt(typeptr)` — no IR op recorded, matching
            // RPython's `return ConstInt(...)`.
            //
            // Operand layout `>i`: 1B dst register only; the dst byte sits
            // at `op.pc + 1`.
            let exc = ctx
                .last_exc_value()
                .ok_or(DispatchError::LastExceptionWithoutActiveException { pc: op.pc })?;
            let exc_ptr = match ctx.last_exc_value_concrete() {
                ConcreteValue::Ref(p) if !p.is_null() => p,
                _ => {
                    return Err(DispatchError::LastExceptionWithoutActiveException { pc: op.pc });
                }
            };
            let typeptr = unsafe {
                (*(exc_ptr as *const pyre_object::interp_exceptions::W_BaseException))
                    .ob_header
                    .ob_type as i64
            };
            let dst = code[op.pc + 1] as usize;
            let cls_const = ctx.trace_ctx.const_int(typeptr);
            write_int_reg(ctx, op.pc, dst, cls_const, ConcreteValue::Int(typeptr))?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "reraise/" => {
            // RPython parity: `pyjitpl.py opimpl_reraise(self)` —
            //
            //   assert self.metainterp.last_exc_value
            //   self.metainterp.popframe()
            //   self.metainterp.finishframe_exception()
            //
            // Reads no operand; uses the standing `metainterp.last_exc_value`
            // which was set either by an earlier `raise/r` in this frame
            // or — when the unwinder routed into a `catch_exception`
            // handler — by the inline_call SubRaise arm just before
            // jumping to the handler PC.
            //
            // Walker behaviour mirrors `raise/r`'s dual-frame routing:
            //   * top-level → outermost FINISH(last_exc_value,
            //     exit_frame_with_exception_descr_ref).
            //   * sub-walk → SubRaise{exc=last_exc_value}, bubbling
            //     through the parent's inline_call handler (which may
            //     itself catch via `catch_exception/L` lookahead).
            //
            // `last_exc_value == None` violates the RPython assert and
            // surfaces as `ReraiseWithoutLastExcValue` (codewriter
            // invariant: `reraise` only emits inside a `catch_exception`
            // body or after an explicit `raise`).
            let exc = ctx
                .last_exc_value()
                .ok_or(DispatchError::ReraiseWithoutLastExcValue { pc: op.pc })?;
            // Symmetric with `raise/r`.
            Ok((
                DispatchOutcome::SubRaise {
                    exc,
                    exc_concrete: ctx.last_exc_value_concrete(),
                },
                op.next_pc,
            ))
        }
        // The `i` spelling is what the assembler emits once the jitdriver
        // index outstrips a signed byte. Its leading byte names an Int-bank
        // slot containing the jdindex; the `c` spelling stores the jdindex
        // directly in that byte.
        "jit_merge_point/cIRFIRF" | "jit_merge_point/iIRFIRF" => {
            // Copy the walk green before any mutable session borrow in this
            // handler; key construction and greenboxes must use one value.
            let is_being_profiled = ctx.session.borrow().is_being_profiled;
            // RPython parity: `opimpl_jit_merge_point` →
            // `reached_loop_header`. pyre's retired
            // trait mirror was `close_loop_args`.
            //
            // The JitCode merge point carries its greens + reds inline
            // (`blackhole.rs bhimpl_jit_merge_point` decodes the same
            // six typed lists). Walking JitCode, the walker reads them
            // directly rather than recomputing live args via liveness.
            //
            // Operand layout `cIRFIRF`: jdindex (`c`, 1 signed byte) +
            // greens (`I`=gi, `R`=gr, `F`=gf) + reds (`I`=ri, `R`=rr,
            // `F`=rf). pyre's portal jitdriver greens =
            // `[next_instr, is_being_profiled, pycode]` (`eval.rs`),
            // so gi[0]=next_instr (the Python pc) and gr[0]=pycode
            // (PyCode). The green key is derivable from the op's own
            // greens, and `next_instr` is the SAME Python-pc coordinate the
            // trace-start seed uses (`trace.rs add_merge_point(make_green_key(
            // w_code, start_pc))`) — no jitcode-pc/python-pc mismatch.
            let jdindex = if op.key == "jit_merge_point/iIRFIRF" {
                let jd_opref = read_int_reg(code, op, 0, ctx)?;
                match ctx.trace_ctx.concrete_of_opref(jd_opref) {
                    Some(Value::Int(value)) => value as usize,
                    _ => return Err(DispatchError::LoopHeaderJdIndexUnresolved { pc: op.pc }),
                }
            } else {
                code[op.pc + 1] as i8 as usize
            };
            let mut off = 1usize; // skip the jdindex operand
            let (gi, n) = read_int_var_list(code, op, off, ctx)?;
            off += n;
            let (gr, n) = read_ref_var_list(code, op, off, ctx)?;
            off += n;
            let (gf, n) = read_float_var_list(code, op, off, ctx)?;
            off += n;
            let (ri, n) = read_int_var_list(code, op, off, ctx)?;
            off += n;
            let (rr, n) = read_ref_var_list(code, op, off, ctx)?;
            off += n;
            let (rf, _n) = read_float_var_list(code, op, off, ctx)?;

            // Green key from (pycode, next_instr) = (gr[0], gi[0]) concretes.
            let (Some(&pc_green), Some(&code_green)) = (gi.first(), gr.first()) else {
                return Err(DispatchError::JitMergePointGreenKeyUnresolved { pc: op.pc });
            };
            // `next_instr` is a Python bytecode position. A negative green
            // is not one; `as usize` would wrap it and file the loop under
            // a different header.
            let next_instr = match ctx.trace_ctx.concrete_of_opref(pc_green) {
                Some(Value::Int(v)) => usize::try_from(v).ok(),
                _ => None,
            };
            let Some(next_instr) = next_instr else {
                return Err(DispatchError::JitMergePointGreenKeyUnresolved { pc: op.pc });
            };

            // `MIFrame.debug_merge_point` records a real
            // `DEBUG_MERGE_POINT` operation before any loop-header decision.
            // It is debugging-only after optimization, but it remains part of
            // `history.length()` and therefore of `trace_limit`.  Omitting it
            // makes recursive traces systematically shorter than the same
            // opcode walk in PyPy and can change which trace reaches the
            // configured limit.
            //
            // RPython operands are
            // `[jd_index, portal_call_depth, current_call_id] + greenkey`.
            // The FBW walker keeps portal frames on `WalkSession::framestack`;
            // canonical helper levels have `w_code == 0` and do not contribute
            // to either depth or call identity.
            let portal_call_depth = ctx
                .trace_ctx
                .portal_call_depth_fn
                .as_ref()
                .map(|f| f() as usize)
                .unwrap_or(0);
            let current_call_id = ctx
                .trace_ctx
                .current_call_id_fn
                .as_ref()
                .map(|f| f())
                .unwrap_or(0);
            let mut debug_args = Vec::with_capacity(3 + gi.len() + gr.len() + gf.len());
            debug_args.push(ctx.trace_ctx.const_int(jdindex as i64));
            debug_args.push(ctx.trace_ctx.const_int(portal_call_depth as i64));
            debug_args.push(ctx.trace_ctx.const_int(current_call_id as i64));
            debug_args.extend_from_slice(&gi);
            debug_args.extend_from_slice(&gr);
            debug_args.extend_from_slice(&gf);
            ctx.trace_ctx
                .record_op(OpCode::DebugMergePoint, &debug_args);

            // pyjitpl.py, the tail of `MIFrame.debug_merge_point`, which
            // `opimpl_jit_merge_point` calls ahead of every early return the
            // loop-header protocol below makes.  The check follows the
            // DEBUG_MERGE_POINT record directly, so `history.length()` counts
            // nothing the opcode body records after the merge point:
            //
            //     if (metainterp.force_finish_trace and
            //             (metainterp.history.length() >
            //              warmrunnerstate.trace_limit * 0.8)):
            //         self._create_segmented_trace_and_blackhole()
            //
            // A key that already overflowed once carries the segmenting signal
            // — `JC_FORCE_FINISH` on the cell for a loop, `FORCE_BRIDGE_SEGMENTING`
            // on the source token for a bridge — and both reach the walk as
            // `TraceCtx::force_finish`. Cutting the trace here, strictly before
            // the 1.0x `blackhole_if_trace_too_long` check the step loop runs
            // (`pyjitpl.py`), is what stops that key from overflowing
            // and aborting forever.
            //
            // The cut belongs at a merge point and nowhere else: the guard it
            // records resumes through the `-live-` marker that precedes every
            // `jit_merge_point` op, which an arbitrary mid-walk position has no
            // counterpart for.
            //
            // Top-level only. Upstream reaches this from an inlined frame too,
            // but a sub-walk's outcome is consumed by
            // `try_walker_inline_user_call`, which has no route to the driver
            // arm that owns the compile half — surfacing it from there would
            // abort the enclosing trace instead of segmenting it.
            if ctx.is_top_level
                && ctx.trace_ctx.force_finish_trace()
                && ctx.trace_ctx.num_ops() > ctx.trace_ctx.trace_limit() * 4 / 5
            {
                if let Some(outcome) = create_segmented_trace(ctx, op.pc, next_instr, false)? {
                    return Ok((outcome, op.next_pc));
                }
            }

            // `dispatch_bytecode`'s `we_are_jitted()` arm reads the portal
            // frame's `debugdata` at the top of every opcode (pyopcode.py).
            // The merge point is the trace's counterpart of that loop top.
            record_portal_debugdata_guard(ctx, op.pc)?;
            // `execute_frame`'s `ec.call_trace` / `ec.return_trace`
            // (pyframe.py) read the global trace function on every call the
            // loop makes.  The walker records neither for an inlined callee,
            // so the loop pins the slot instead.
            record_portal_tracefunc_guard(ctx, op.pc)?;
            record_portal_profilefunc_guard(ctx, op.pc)?;

            // An inlined callee's own loop
            // header routes to a `CALL_ASSEMBLER` into its already-compiled loop
            // token EVEN WHEN its pycode green resolves.  nbody's `advance` has a
            // const-Ref code_green (resolves) plus an existing loop token, so the
            // recovery in the `code_green unresolved` arm below never fires and
            // the walk falls through to the normal loop-crossing path — which
            // closes a degenerate module-loop iteration at the inner header or
            // walks into the callee body and aborts
            // `LoopBearingCalleeInlineUnsupported`.  Firing here (only inside a
            // sub-walk — the framestack is non-empty — and only when a token
            // exists) routes the inlined loop-bearing callee to its own compiled
            // loop, the trait-parity `LoopTargetDescr`/`CALL_ASSEMBLER` shape.
            //
            // "An inlined callee's own loop" is whose loop this header is, and
            // that is a FRAME question, not a code one: `opimpl_jit_merge_point`
            // (pyjitpl.py) closes the loop only under `if not
            // self.metainterp.portal_call_depth:` and otherwise finishes the
            // frame and takes `do_recursive_call(..., assembler_call=True)`.
            // A sub-walk is exactly a portal frame below the trace root — every
            // one runs with `is_top_level == false` — while a bridge's
            // outer-frame continuation runs the ROOT forward at the top level
            // even with reconstructed parent frames still on the framestack, so
            // it keeps closing with a JUMP back into its own loop rather than a
            // CALL_ASSEMBLER request no caller consumes (`OuterNonTerminate`).
            //
            // A CODE-identity test cannot stand in for that: a self-recursive
            // function inlined into itself has the callee's `w_code` EQUAL to the
            // trace root's, so it read as "the root's own loop" and closed the
            // loop against the caller's frame — publishing the header pc over the
            // caller's post-loop operand stack, which `FOR_ITER` then advances as
            // if it were the iterator (`loop_callee_for_header_resume`).
            //
            // A transparent helper walk is NOT such a portal frame: it is a
            // Rust helper body descended into, and `framestack.last()` there
            // describes the helper's own entry (or, for a root helper walk that
            // pushes none, an unrelated caller). The arm is reachable from one:
            // `_unpackiterable_unknown_length` is a second jit driver and the
            // only helper jitcode in the build that carries a
            // `jit_merge_point` -- 2 of 2939 do, and the other is the portal
            // `eval_loop_jit` itself. So exclude it explicitly rather than
            // relying on the token lookup to miss.
            //
            // No corpus program reaches the arm this way today: over the 491
            // synthetic fixtures the print below never fires while its control
            // in `inline_call.rs` does, on one fixture and one helper. The
            // exclusion is the invariant, not a repair of an observed answer.
            if fbw_debug_abort_enabled() && ctx.fbw_mode.transparent_helper_jitcode_index.is_some()
            {
                eprintln!(
                    "[fbw-mergepoint-in-helper] jitcode_index={:?} is_top_level={} \
                     framestack_top_w_code={:?}",
                    ctx.fbw_mode.transparent_helper_jitcode_index,
                    ctx.is_top_level,
                    ctx.session
                        .borrow()
                        .framestack
                        .last()
                        .map(|frame| frame.w_code),
                );
            }
            // `opimpl_jit_merge_point` (`pyjitpl.py`), the `else` taken when
            // `metainterp.portal_call_depth` is non-zero: this header is not
            // the traced loop's own close. Finish the callee and
            // `do_recursive_call(..., assembler_call=True)`. A frame rebuilt
            // by `rebuild_from_resumedata` is an ordinary MIFrame, so a bridge
            // that resumes inside a callee and later reaches that callee's
            // header takes the same arm. `drive_bridge_frame_subwalk` consumes
            // the outcome: a token records CALL_ASSEMBLER
            // (`direct_assembler_call`); no token records the portal runner
            // (`direct_call_may_force`) instead of walking the loop body.
            // Nested inlines inherit `carrier_resume` and are consumed by
            // `try_walker_inline_resolved_user_call`. A missing token there
            // residualizes the original CALL.
            let carrier_resume = ctx.fbw_mode.carrier_resume;
            let callee_code = (!ctx.is_top_level
                && ctx.fbw_mode.transparent_helper_jitcode_index.is_none())
            .then(|| {
                ctx.session
                    .borrow()
                    .framestack
                    .last()
                    .map(|frame| frame.w_code)
            })
            .flatten();
            if let Some(callee_code) = callee_code {
                let callee_key = crate::driver::make_green_key_typed(
                    callee_code as *const (),
                    next_instr,
                    is_being_profiled,
                );
                let (driver, _) = crate::driver::driver_pair();
                let greenboxes = [
                    Value::Int(next_instr as i64),
                    Value::Int(is_being_profiled as i64),
                    Value::Ref(majit_ir::GcRef(callee_code)),
                ];
                let red_types = [Type::Ref, Type::Ref];
                let token = driver.get_or_make_portal_assembler_token_arc(
                    &callee_key,
                    &greenboxes,
                    &red_types,
                );
                if token.is_some() || carrier_resume {
                    return Ok(surface_carrier_or_inline_subloop(
                        token,
                        next_instr,
                        carrier_resume,
                        op.pc,
                        op.next_pc,
                    ));
                }
            }
            let code_ptr = match ctx.trace_ctx.concrete_of_opref(code_green) {
                Some(Value::Ref(gcref)) if gcref.0 != 0 => gcref.0 as *const (),
                _ => {
                    // Inside a multi-frame inline sub-walk the callee's own
                    // `jit_merge_point` (its loop header) carries a pycode green
                    // with no live Ref shadow, so this resolution fails and the
                    // enclosing trace would decline. Recover
                    // the callee code from the FBW inline stack; if a compiled
                    // loop token already exists for (callee_code, next_instr),
                    // surface a recursive CALL_ASSEMBLER request to the caller's
                    // inline return site (mirror `opimpl_recursive_call_
                    // assembler`, metainterp.rs).
                    // Same carrier rule as the arm above: a resume with no
                    // token still leaves the loop body via `direct_call_may_force`.
                    let carrier_resume = ctx.fbw_mode.carrier_resume;
                    let callee_code = ctx
                        .session
                        .borrow()
                        .framestack
                        .last()
                        .map(|frame| frame.w_code);
                    if let Some(callee_code) = callee_code {
                        let callee_key = crate::driver::make_green_key_typed(
                            callee_code as *const (),
                            next_instr,
                            is_being_profiled,
                        );
                        let (driver, _) = crate::driver::driver_pair();
                        let greenboxes = [
                            Value::Int(next_instr as i64),
                            Value::Int(is_being_profiled as i64),
                            Value::Ref(majit_ir::GcRef(callee_code)),
                        ];
                        let red_types = [Type::Ref, Type::Ref];
                        let token = driver.get_or_make_portal_assembler_token_arc(
                            &callee_key,
                            &greenboxes,
                            &red_types,
                        );
                        if token.is_some() || carrier_resume {
                            return Ok(surface_carrier_or_inline_subloop(
                                token,
                                next_instr,
                                carrier_resume,
                                op.pc,
                                op.next_pc,
                            ));
                        }
                    }
                    top_level_live_code(ctx)
                        .ok_or(DispatchError::JitMergePointGreenKeyUnresolved { pc: op.pc })?
                }
            };
            let key = crate::driver::make_green_key(code_ptr, next_instr, is_being_profiled);

            // pyjitpl.py: a jit_merge_point is a loop CROSSING
            // only when a `loop_header` op (the lowered `can_enter_jit`
            // at a backward-jump site) stamped the per-trace flag just
            // before — arrival by straight-line fall-through (e.g. the
            // first check of a fresh inner `while`) records nothing and
            // walks on. Without this gate the walker closes a degenerate
            // "outer-iteration" loop at the inner header (exhausted-check
            // → outer increment → new-iterator → re-arrival), occupying
            // the inner loop's green key so its specialized retrace can
            // never compile. Mirrors majit's `BC_JIT_MERGE_POINT` auto
            // loop-header + close protocol (`pyjitpl/dispatch.rs`).
            // The `c` form's jdindex is the leading literal byte; the `i`
            // form was resolved through its Int-bank operand above.
            // pyjitpl.py capture_resumedata: a guard's resume coordinate
            // (`resumepc=orgpc`) lies INSIDE the guarded opcode's
            // implementation, past the dispatch-top `jit_merge_point`, so an
            // RPython MIFrame resumed from a guard never re-crosses the
            // loop-header merge point at position zero. The walker resumes a
            // bridge at the opcode BOUNDARY, so its first crossing at the
            // resume coordinate is the same op it is resuming INTO — not a
            // loop crossing. Skip exactly once. `take()` clears on the first
            // crossing regardless of pc, so a mid-body-resume bridge whose
            // first crossing is a DIFFERENT header is unaffected.
            // pyjitpl.py `_handle_guard_failure` pre-arms the flag
            // when the source guard is a `ResumeAtPositionDescr` — the descr
            // `inline_short_preamble` stamps onto the guards it replays
            // (unroll.py:337 / :409). Those guards sit at the target loop's
            // entry, so the bridge grown from one has to close at its very
            // first merge point rather than record another iteration first.
            // Arming here rather than at the bridge setup keeps the flag equal
            // to the `jdindex` the assert below compares against.
            if ctx.trace_ctx.seen_loop_header_for_jdindex < 0
                && std::mem::take(&mut ctx.trace_ctx.bridge_resume_at_position)
            {
                ctx.trace_ctx.seen_loop_header_for_jdindex = jdindex as i32;
                ctx.trace_ctx.seen_loop_header_jit_pc = None;
            }
            if ctx.is_top_level
                && ctx.trace_ctx.seen_loop_header_for_jdindex < 0
                && ctx.fbw_mode.bridge_entry_merge_pc.take() == Some(next_instr)
            {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            if ctx.trace_ctx.seen_loop_header_for_jdindex < 0 {
                // pyjitpl.py `if not any_operation: return`.
                if ctx.trace_ctx.num_ops() == 0 {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // pyjitpl.py `if not jitdriver_sd.no_loop_header:`
                let no_loop_header = ctx
                    .trace_ctx
                    .metainterp_sd()
                    .jitdrivers_sd
                    .get(jdindex)
                    .map(|jd| jd.no_loop_header)
                    .unwrap_or(false);
                if !no_loop_header {
                    // pyjitpl.py `if self.metainterp.portal_call_depth:
                    // return` — nested portal call waits for an explicit
                    // loop_header.
                    let depth_zero = ctx
                        .trace_ctx
                        .portal_call_depth_fn
                        .as_ref()
                        .map(|f| f() == 0)
                        .unwrap_or(false);
                    if !depth_zero || !ctx.is_top_level {
                        return Ok((DispatchOutcome::Continue, op.next_pc));
                    }
                    // pyjitpl.py: fall-through arrival counts as
                    // an automatic loop_header only when compiled targets
                    // already exist for the crossed green key.
                    let has_targets = ctx
                        .trace_ctx
                        .has_compiled_targets_fn
                        .as_ref()
                        .map(|f| f(key))
                        .unwrap_or(false);
                    if !has_targets {
                        return Ok((DispatchOutcome::Continue, op.next_pc));
                    }
                }
                // pyjitpl.py: automatically add a loop_header.
                ctx.trace_ctx.seen_loop_header_for_jdindex = jdindex as i32;
                ctx.trace_ctx.seen_loop_header_jit_pc = None;
            }
            // pyjitpl.py.
            assert!(
                ctx.trace_ctx.seen_loop_header_for_jdindex == jdindex as i32,
                "found a loop_header for a JitDriver that does not match \
                 the following jit_merge_point's"
            );
            ctx.trace_ctx.seen_loop_header_for_jdindex = -1;
            let back_edge_jit_pc = ctx.trace_ctx.seen_loop_header_jit_pc.take();

            // pyjitpl.py self.heapcache.reset()
            ctx.trace_ctx.heap_cache_mut().reset();

            // `close_loop_args_at` (trace_opcode.rs) runs the merge-point
            // vable `last_instr` sync BEFORE building the jump args: the
            // scalar is overridden to `merge_pc - 1` (a resume into the
            // target loop must re-enter at the header opcode). The walker
            // must do the same before `append_virtualizable_boxes` below —
            // otherwise the compile_trace arm's JUMP into the existing loop
            // carries the LAST GUARD's published `last_instr` (e.g. 104
            // instead of header-1=86 on fannkuch), resuming the interpreter
            // at the wrong bytecode (fannkuch permutation state never
            // reaches its exit condition → non-crashing infinite loop).
            sync_intermediate_merge_point_last_instr(ctx.trace_ctx, next_instr);

            // Reds = the live loop args, in bytecode bank order
            // [int.., ref.., float..]. For pyre's portal jitdriver the
            // reds are `[frame, ec]` (both Ref → rr), matching the
            // reds-only LABEL inputargs after
            // `patch_new_loop_to_load_virtualizable_fields`.
            let mut live_args: Vec<OpRef> = Vec::with_capacity(ri.len() + rr.len() + rf.len());
            live_args.extend(ri.iter().copied());
            live_args.extend(rr.iter().copied());
            live_args.extend(rf.iter().copied());
            // The loop-close path (`run_perfn_walk` CloseLoop post-processing,
            // trace.rs) rebuilds the jump args via `close_loop_args_at`, which
            // sources the reds from `sym.frame` / `sym.execution_context` — not
            // from the walk register file read above (the register slot may
            // hold a const-folded alias of the same value). The merge-point
            // registration must use the SAME box identities: `history.cut`
            // (cross-loop cut, pyjitpl.py) takes the registered
            // green_boxes as the new loop's inputargs, and a close-side red
            // absent from them escapes into an extra appended inputarg —
            // producing an entry layout `patch_new_loop_to_load_virtualizable_
            // fields` cannot reduce, so every interpreter entry aborts
            // (`extend_compiled_live_values` count mismatch).
            {
                let sym_ptr = ctx.fbw_mode.snapshot_sym;
                if !sym_ptr.is_null() && ri.is_empty() && rf.is_empty() && rr.len() == 2 {
                    let sym = unsafe { &*sym_ptr };
                    live_args[0] = sym.frame();
                    if !sym.execution_context().is_none() {
                        live_args[1] = sym.execution_context();
                    }
                }
            }
            // A crossed `compile_trace` walk keeps the origin frame's
            // `virtualizable_boxes`. Dest LABEL arity is dest's
            // `locals_cells_stack_w` length (`virtualizable.py
            // get_array_length`). Re-derive dest boxes with recorded
            // GETFIELD_GC / GETARRAYITEM_GC (`initialize_virtualizable`
            // `read_boxes` as IR, `patch_new_loop_to_load_virtualizable_fields`)
            // so the JUMP args are boxes, not constants, and match dest.
            if let Some(&frame) = live_args.first() {
                ctx.trace_ctx.gen_load_from_other_virtualizable(frame);
                // Re-sync replaces the cache, including last_instr. Pin
                // dest's slot to merge_pc-1 after that write.
                sync_intermediate_merge_point_last_instr(ctx.trace_ctx, next_instr);
            }
            // Array index `>= valuestackdepth` is the dead stack tail
            // (`pyframe.py popvalue_maybe_none`). majit's fill is generic
            // (`virtualizable.py VirtualizableInfo`); supply the boundary
            // from the concrete frame, falling back to the symbolic mirror
            // the same way `close_loop_args_at` does.
            let dead_array_tail_from = (!ctx.fbw_mode.snapshot_sym.is_null()).then(|| {
                let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
                crate::state::concrete_stack_depth(sym.tracing_vable_frame_addr())
                    .unwrap_or_else(|| sym.valuestackdepth())
            });
            ctx.trace_ctx
                .fill_virtualizable_boxes_to_declared_layout(dead_array_tail_from);
            live_args = append_virtualizable_boxes(ctx.trace_ctx, live_args);

            // pyjitpl.py remove_consts_and_duplicates over the
            // reds + virtualizable_boxes[:-1]: every live arg must be a
            // distinct non-const box before it is registered as a merge
            // point or matched for loop closure. A const or duplicate is
            // replaced with a fresh `same_as` op, written back into the
            // virtualizable shadow so subsequent reads/snapshots use the
            // new identity (the in-place boxes[i] mutation upstream).
            // Without this, the registered green_boxes carry the SAME
            // OpRef at two positions and `cut_trace_from`'s remap maps
            // both to the LAST position — every body/snapshot reference
            // to the duplicated box then reads the wrong loop-carried
            // slot (e.g. a local aliased by a dead stack entry reads the
            // entry's NULL at every deopt).
            {
                use std::collections::HashSet;
                let mut duplicates: HashSet<OpRef> = HashSet::new();
                for i in 0..live_args.len() {
                    let opref = live_args[i];
                    if opref.is_constant() || !duplicates.insert(opref) {
                        let tp = ctx
                            .trace_ctx
                            .get_opref_type(opref)
                            .unwrap_or(majit_ir::Type::Ref);
                        let new_opref = ctx.trace_ctx.record_same_as(opref, tp);
                        live_args[i] = new_opref;
                        // live_args = [frame, ec, vable_boxes[0..len-1]];
                        // frame/ec are first-occurrence runtime boxes and
                        // never wrap, so only the vable payload mirrors
                        // back (virtualizable_boxes[i] = op upstream).
                        if i >= 2 {
                            ctx.trace_ctx.set_virtualizable_box_at(i - 2, new_opref);
                        }
                    }
                }
            }

            if std::env::var("PYRE_DIAG_51C").is_ok() {
                eprintln!(
                    "[51c-redclose] pc={} ri_regs={:?} rr_regs={:?} rf_regs={:?}",
                    op.pc, ri, rr, rf
                );
                for (idx, &arg) in live_args.iter().enumerate() {
                    eprintln!(
                        "[51c-redclose]   live_arg[{idx}] {arg:?} concrete={:?}",
                        ctx.trace_ctx.concrete_of_opref(arg)
                    );
                }
            }

            // pyjitpl.py compile_trace attempt (retired trait
            // mirror `close_loop_args`): when the
            // crossed green key already has compiled targets and no
            // retrace is in progress, close the trace-so-far as a bridge
            // (guard origin) / entry bridge (interp origin,
            // `compile_trace_from_interp`) ending in a JUMP into the
            // existing loop.  Without this a func-entry trace that walks
            // the prologue and reaches the already-hot inner loop header
            // falls through to compile_loop → has_compiled_targets →
            // SwitchToBlackhole(ABORT_BAD_LOOP), so every portal call
            // re-runs the prologue interpreted and re-aborts a trace —
            // overhead scaling with call count.
            if ctx.is_top_level && ctx.is_authoritative_executor {
                let (driver, _) = crate::driver::driver_pair();
                let has_partial = driver.meta_interp().partial_trace().is_some();
                let bridge_origin = driver
                    .meta_interp()
                    .bridge_info()
                    .map(|b| (b.trace_id, b.fail_index));
                let has_targets = driver.meta_interp().has_compiled_targets(key);
                // pyjitpl.py `if not self.partial_trace:` is the only
                // compile_trace gate. Upstream retries on every header visit.
                if !has_partial && has_targets {
                    // pyjitpl.py reached_loop_header:
                    //
                    //     # generate a dummy guard just before the JUMP so
                    //     # that unroll can use it when it's creating
                    //     # artificial guards.
                    //     self.generate_guard(rop.GUARD_FUTURE_CONDITION)
                    //
                    // Upstream emits it once in `reached_loop_header`, ahead
                    // of BOTH the `get_procedure_token` / `compile_trace`
                    // pair that closes into an ALREADY compiled loop
                    // and the merge-point scan that closes a
                    // loop of the trace's own. pyre splits those two
                    // outcomes across different returns and had the guard on
                    // only one of them: the own-loop leg gets it from
                    // `close_loop_args_at`, which runs off
                    // `DispatchOutcome::CloseLoop`, while this leg returns
                    // `CompileTracePending` and never reached it. So every
                    // bridge arrived at the optimizer carrying no
                    // GUARD_FUTURE_CONDITION — measured across
                    // `pyre/bench/synth`: 672 bridge compilations, 0 with
                    // one. `Optimizer.patchguardop` was left to a stand-in
                    // synthesized from one of the bridge's own body guards,
                    // whose resume coordinate is mid-trace where upstream's
                    // is the merge point's, and a bridge with no such guard
                    // got none at all.
                    //
                    // `_jump_to_existing_trace` dereferences that
                    // patchguardop for the extra virtual-state guards
                    // (`unroll.py` `_jump_to_existing_trace`) and hands it to
                    // `inline_short_preamble`, which stamps it onto every
                    // replayed short-preamble guard.
                    //
                    // Emitted on this leg only, so the own-loop leg is not
                    // double-guarded. The capture coordinate is the
                    // `jit_merge_point` op's OWN JitCode pc — the walk is
                    // standing on the loop header, so "where we are" and
                    // "where the synthesized JUMP goes" are the same
                    // program point. `next_instr` is the green Python pc
                    // (`make_green_key` above); feeding it to a parameter
                    // read as a JitCode offset resolves an unrelated
                    // `-live-` marker, and the snapshot then describes a
                    // program point the walk registers hold nothing for.
                    ctx.trace_ctx
                        .record_guard(OpCode::GuardFutureCondition, &[], 0);
                    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
                    let outcome = match bridge_origin {
                        // Guard-origin: existing bridge path.
                        Some(_) => {
                            driver
                                .meta_interp_mut()
                                .compile_trace(key, &live_args, bridge_origin)
                        }
                        // pyjitpl.py interp-origin: a
                        // function-entry trace (ResumeFromInterpDescr)
                        // closes as an entry bridge jumping into the
                        // already-compiled hot loop (compile.py);
                        // a trace rooted at a *loop header* falls back to
                        // the plain bridge shape.
                        None => match driver.compile_trace_entry_data() {
                            Some((original_green_key, mut entry_meta)) => {
                                // `compile_trace_entry_data` clones the active
                                // trace metadata, whose `namespace_dependent` is
                                // only finalized by `finish_trace_namespace_dependency`
                                // after the walk returns. An entry bridge is
                                // compiled mid-walk, before that finalize, so a
                                // trace that has already read a module global
                                // would otherwise install the bridge with a stale
                                // `namespace_dependent = false` and let it be
                                // re-entered after later namespace growth. Fold in
                                // the live per-trace flag so the bridge keeps the
                                // conservative namespace gate.
                                entry_meta.namespace_dependent |= ctx.trace_ctx.reads_module_global;
                                driver.meta_interp_mut().compile_trace_from_interp(
                                    key,
                                    &live_args,
                                    original_green_key,
                                    entry_meta,
                                )
                            }
                            // compile.py compile_trace — an interp-origin close is a
                            // `ResumeFromInterpDescr` entry bridge, and with no
                            // entry data there is no original green key to attach
                            // it to. `compile_trace(key, args, None)` cannot serve
                            // that state: its `None`-origin arm demands the
                            // entry-bridge payload and answers `Cancelled` for
                            // every input, after deep-cloning the whole
                            // trace-so-far and lowering its snapshot pool. Decline
                            // here instead, so the give-up reads as one.
                            // `compile_trace_entry_data` reads
                            // `trace_ctx().root_green_key()` and `trace_meta()`,
                            // and every `JitDriver` trace start installs both
                            // halves, so this is unreachable from a jd0 walk; only
                            // `MetaInterp::force_start_tracing` opens a tracer with
                            // no session envelope.
                            None => majit_metainterp::CompileOutcome::Cancelled,
                        },
                    };
                    // pyjitpl.py `classify_compile_outcome` is the shared
                    // reading of a close's outcome; the sibling close in
                    // `JitDriver::merge_point` already goes through it.
                    match driver.meta_interp().classify_compile_outcome(outcome) {
                        majit_metainterp::BridgeCompileResult::Compiled => {
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][walker-reached-loop-header] compile_trace success: \
                                 key={} pc={} bridge={:?}",
                                    key, next_instr, bridge_origin
                                );
                            }
                            // pyjitpl.py raise_if_successful() — the
                            // successful compile_trace ends tracing; surface
                            // the dedicated outcome so the driver maps it to
                            // `TraceAction::CompileTrace` (no further compile
                            // or abort on this session).
                            driver.note_compile_trace_success();
                            return Ok((
                                DispatchOutcome::CompileTracePending {
                                    loop_header_pc: next_instr,
                                },
                                op.next_pc,
                            ));
                        }
                        // pyjitpl.py raise_if_successful does not raise
                        // on None. Fall through to the merge-point scan;
                        // the next header visit retries compile_trace.
                        majit_metainterp::BridgeCompileResult::RetraceNeeded
                        | majit_metainterp::BridgeCompileResult::Declined
                        | majit_metainterp::BridgeCompileResult::Failed => {}
                    }
                    // The jump did not take (`compile.compile_trace` returns
                    // None when none of the existing loop tokens match). Fall
                    // through to the merge-point scan below, exactly as
                    // `reached_loop_header` does after its own
                    // `self.compile_trace(...)` call returns (pyjitpl.py):
                    // the scan closes at the first same-greenkey merge point, and
                    // `compile_loop` gives that trace up at its own
                    // `has_compiled_targets` (pyjitpl.py); a first visit
                    // registers a merge point (pyjitpl.py) and keeps
                    // tracing.
                    majit_metainterp::mc_diag_bump(27);
                }
            }

            // pyjitpl.py: a matching merge point (same green key
            // + red-bank shape) closes the loop; first visit registers and
            // continues to unroll.
            let live_typed =
                crate::driver::make_green_key_typed(code_ptr, next_instr, is_being_profiled);
            if ctx
                .trace_ctx
                .has_merge_point_same_greenkey(key, Some(&live_typed), live_args.len())
            {
                // The matched merge point need not be the one tracing started
                // from: `reached_loop_header` scans every registered merge
                // point in reverse and closes at the first same_greenkey hit,
                // whichever loop that is, and `compile_loop` re-derives the
                // green key from the merge point it closed at.
                let (loop_header_marker_jit_pc, back_edge_pc, back_edge_marker_jit_pc) = {
                    let sym_ptr = ctx.fbw_mode.snapshot_sym;
                    if sym_ptr.is_null() {
                        (None, None, None)
                    } else {
                        let sym = unsafe { &*sym_ptr };
                        if sym.jitcode().is_null() {
                            (None, None, None)
                        } else {
                            let payload = unsafe { &(&*sym.jitcode()).payload };
                            let back_edge_pc = back_edge_jit_pc.map(|jit_pc| {
                                crate::py_coord::containing_py_pc_for_jitcode_pc(
                                    &payload.metadata,
                                    jit_pc,
                                ) as usize
                            });
                            (
                                payload.resume_marker_for_jitcode_pc(op.pc),
                                back_edge_pc,
                                back_edge_jit_pc.and_then(|jit_pc| {
                                    payload.resume_marker_for_jitcode_pc(jit_pc)
                                }),
                            )
                        }
                    }
                };
                Ok((
                    DispatchOutcome::CloseLoop {
                        jump_args: live_args,
                        loop_header_pc: next_instr,
                        loop_header_marker_jit_pc,
                        back_edge_pc,
                        back_edge_marker_jit_pc,
                    },
                    op.next_pc,
                ))
            } else {
                // This merge point registers (does not close) — no already
                // registered merge point carries this green key and red-bank
                // shape, so this is the first arrival at the loop whose header
                // is `next_instr`.  The intermediate merge-point
                // vable→heap writeback (#62 / #67-remaining) already ran
                // above, before the live-args build, mirroring
                // `close_loop_args_at`'s ordering.
                let green_boxes: Vec<majit_metainterp::GreenBox> = live_args
                    .iter()
                    .map(|opref| {
                        let ty = ctx.trace_ctx.get_opref_type(*opref).unwrap_or_else(|| {
                            panic!(
                                "jit_merge_point live arg {opref:?} has no type in \
                                 OptContext; RPython Box always carries its type"
                            )
                        });
                        majit_metainterp::GreenBox::new(*opref, ty)
                    })
                    .collect();
                ctx.trace_ctx.add_merge_point_with_key(
                    key,
                    Some(crate::driver::make_green_key_typed(
                        code_ptr,
                        next_instr,
                        is_being_profiled,
                    )),
                    green_boxes,
                    next_instr,
                );
                Ok((DispatchOutcome::Continue, op.next_pc))
            }
        }
        key if key.starts_with("recursive_call_") => dispatch_recursive_call(code, op, ctx),
        other => Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: other,
        }),
    }
}

#[cfg(test)]
mod tests;
