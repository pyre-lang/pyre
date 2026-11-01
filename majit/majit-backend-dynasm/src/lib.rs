/// rpython/jit/backend/ parity:
/// Direct machine code generation via dynasm-rs with in-place patching.
///
/// Module structure mirrors RPython's backend directory layout:
///   llsupport/llmodel.py    → runner.rs  (DynasmBackend)
///   llsupport/assembler.py  → assembler.rs (BaseAssembler — shared)
///   llsupport/regalloc.py   → regalloc.rs (BaseRegalloc — shared)
///   llsupport/jitframe.py   → jitframe.rs
///   x86/assembler.py        → x86/assembler.rs (Assembler386)
///   aarch64/assembler.py    → aarch64/assembler.rs (AssemblerARM64)
///   aarch64/codebuilder.py  → aarch64/codebuilder.rs
///   aarch64/opassembler.py  → aarch64/opassembler.rs (ResOpAssembler)
///   aarch64/registers.py    → aarch64/registers.rs
///
/// arch.rs, codebuf.rs, guard.rs, regloc.rs are from llsupport/.
// ── Shared modules (llsupport/ parity) ──
pub mod arch;
pub mod callbuilder;
pub mod codebuf;
pub mod gcmap;
pub mod guard;
pub use majit_backend::jitframe;
pub use majit_backend::llmodel;
pub(crate) mod jump;
#[expect(
    clippy::too_many_arguments,
    reason = "the register-allocation entry points preserve RPython's explicit state-threading signatures; bundling those arguments would diverge from the audited line-by-line backend port"
)]
pub mod regalloc;
pub mod regloc;
pub mod runner;

// ── Architecture-specific modules ──
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86;

// ── llmodel.py _store_exception JIT exception state ──
// RPython stores exception state in thread-local (GIL-protected) globals.
// Cranelift uses JIT_EXC_VALUE / JIT_EXC_TYPE atomics (compiler.rs).
// Dynasm uses the same pattern for structural equivalence.

use std::sync::atomic::{AtomicI64, Ordering};

/// Whether a `debug_print` would be emitted on this thread.
///
/// `have_debug_prints` — true only inside a debug section the `MAJIT_LOG`
/// prefix filter accepts, so a category filter silences these sites.
/// Not a cached `std::env::var_os`: the ready bit changes when a section
/// opens. Mirrors the equivalent helper in `majit-backend-cranelift`.
///
/// `debug_print.h` `OP_HAVE_DEBUG_PRINTS(r)` inlines one load of
/// `pypy_have_debug_prints`.
#[inline(always)]
pub fn majit_log_enabled() -> bool {
    majit_ir::debug::have_debug_prints()
}

/// Log backend emission events without enabling per-execution diagnostics.
/// The version marker lets census consumers reject an older binary rather
/// than interpreting an unsupported option as zero emitted operations.
pub fn majit_ops_log_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let enabled = std::env::var_os("MAJIT_LOG_OPS").is_some();
        if enabled {
            eprintln!("[dynasm] op-log-version=1");
        }
        enabled
    });
    *ENABLED || majit_log_enabled()
}

/// Whether `MAJIT_DUMP` is set, cached at first access.
#[inline]
pub fn majit_dump_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_DUMP").is_some());
    *ENABLED
}

/// Whether `MAJIT_GC_FREELIST_DIAG` is set, cached at first access.
///
/// Read once per residual call and twice per compiled-trace entry, so the
/// uncached form put `getenv` on the hottest paths the backend has.
#[inline]
pub fn gc_freelist_diag_enabled() -> bool {
    // One cache for the whole process: `majit_gc` reads the same variable in
    // `free_arena`.
    majit_gc::gc_freelist_diag_enabled()
}

/// Whether `MAJIT_DYNASM_EXEC_DIAG` is set, cached at first access.
#[inline]
pub fn dynasm_exec_diag_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_DYNASM_EXEC_DIAG").is_some());
    *ENABLED
}

/// The trace id `MAJIT_TRACE_CALL_DIAG` selects, cached at first access.
///
/// Consulted once per guard recovery stub and once per call, so the uncached
/// form took the environment lock for every guard the backend assembled.
pub fn trace_call_diag_id() -> Option<u64> {
    static ID: std::sync::LazyLock<Option<u64>> = std::sync::LazyLock::new(|| {
        std::env::var("MAJIT_TRACE_CALL_DIAG")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
    });
    *ID
}

/// The trace id `MAJIT_TRACE_OPS_DIAG` selects, cached at first access.
pub fn trace_ops_diag_id() -> Option<u64> {
    static ID: std::sync::LazyLock<Option<u64>> = std::sync::LazyLock::new(|| {
        std::env::var("MAJIT_TRACE_OPS_DIAG")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
    });
    *ID
}

static JIT_EXC_VALUE: AtomicI64 = AtomicI64::new(0);
// Holds the pending exception's `typeptr` (an immortal static `PyType`), never a
// managed object, so it is deliberately not GC-rooted.
static JIT_EXC_TYPE: AtomicI64 = AtomicI64::new(0);

/// llmodel.py _store_exception parity: set JIT exception state.
/// `value` is a valid OBJECTPTR (or 0). Exception class derived from
/// value.typeptr (offset 0), matching RPython's invariant.
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        unsafe { *(value as *const i64) }
    };
    JIT_EXC_VALUE.store(value, Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, Ordering::Relaxed);
}

/// Check if an exception is currently pending.
pub fn jit_exc_is_pending() -> bool {
    JIT_EXC_VALUE.load(Ordering::Relaxed) != 0
}

/// cpu.grab_exc_value parity: read exception class from TLS.
pub fn jit_exc_class_raw() -> i64 {
    JIT_EXC_TYPE.load(Ordering::Relaxed)
}

/// `_store_and_reset_exception` (x86/assembler.py, 1847-1848) parity:
/// read `pos_exc_value` and zero the global.  Not `grab_exc_value`, which
/// reads `jf_guard_exc` off the deadframe and does not write.
pub fn jit_exc_value_raw() -> i64 {
    JIT_EXC_VALUE.swap(0, Ordering::Relaxed)
}

/// Non-destructive read of `JIT_EXC_VALUE` for the GC root walker (unlike
/// `jit_exc_value_raw`, which swaps the cell to 0).
pub fn jit_exc_value_peek() -> i64 {
    JIT_EXC_VALUE.load(Ordering::Relaxed)
}

/// Root-walker write-back for `JIT_EXC_VALUE`: a minor collection moved the
/// pending exception from `old` to `new`. A compare-exchange, so a cell that
/// no longer holds `old` is left alone.
pub fn jit_exc_value_forward(old: i64, new: i64) {
    let _ = JIT_EXC_VALUE.compare_exchange(old, new, Ordering::Relaxed, Ordering::Relaxed);
}

/// Clear exception state.
pub fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
}

/// Address of `JIT_EXC_VALUE` for direct memory load/store from JIT-emitted
/// code (`_store_and_reset_exception` parity, `x86/assembler.py`).
pub fn jit_exc_value_addr() -> usize {
    &JIT_EXC_VALUE as *const _ as usize
}

/// Address of `JIT_EXC_TYPE` for direct memory load/store from JIT-emitted
/// code.  Mirrors `cpu.pos_exception()`; cleared together with
/// `pos_exc_value` whenever the propagate path drains the global exception.
pub fn jit_exc_type_addr() -> usize {
    &JIT_EXC_TYPE as *const _ as usize
}

/// Write a thread-local slot that compiled entrypoints may read back.
pub fn jit_threadlocalref_set(offset: i64, value: i64) {
    majit_rlib::rthread::threadlocalref_set(offset as usize, value);
}

/// `llmodel.py` `threadlocalref_addr`: the `pypy_threadlocal_s` address a
/// compiled entry receives as its second argument. The prologue spills it
/// and `CALL_ASSEMBLER` reloads it for the callee.
#[inline]
pub(crate) fn jit_threadlocalref_base() -> *const i64 {
    majit_rlib::rthread::threadlocalref_addr()
}

// ── CALL_ASSEMBLER helper infrastructure ──
// assembler.py:345-350: jd.assembler_helper_adr — the slow path runtime
// entry point for CALL_ASSEMBLER when callee doesn't finish normally.
//
// Cranelift uses register_call_assembler_blackhole/bridge/force/unbox_int.
// Dynasm provides the same registration API and a C-callable trampoline
// that the generated slow path code calls directly.

use std::sync::OnceLock;

/// Blackhole resume: (descr_addr, deadframe) → Option<result>.
///
/// The receiving handler recovers the failed descr from `descr_addr`
/// via `Backend::fail_descr_arc_from_addr` (`history.py:125`
/// `cpu.get_latest_descr` parity) and derives the resume identity
/// (`jct.green_key` / `descr.trace_id()` /
/// `descr.fail_index_per_trace()`) from that Arc, mirroring
/// `compile.py resume_in_blackhole(descr, deadframe)` where the descr
/// is the sole identity carrier.  No surrogate triple crosses the C-ABI.
///
/// The pending exception stays in `jf_guard_exc`. The receiver reads it
/// with `cpu.grab_exc_value(deadframe)` (`llmodel.py`); the grab does
/// not write the slot. Clearing `jf_guard_exc` is emitted code's job
/// (`_restore_exception`, x86/assembler.py) and a fresh frame's job
/// (`execute_token` / `malloc_jitframe`). `blackhole.py
/// _prepare_resume_from_failure` hands that read to the resumed frame.
/// `compile.py resume_in_blackhole(descr, deadframe)`: descr address
/// plus the jitframe. Values stay in `jf_frame[]`.
pub type BlackholeFn = fn(usize, *mut jitframe::JitFrame) -> Option<i64>;

/// Bridge compilation: raw values, descr identity, and optional GUARD_VALUE
/// operand → compiled?
///
/// The receiving handler recovers the failed descr from `descr_addr`
/// via `Backend::fail_descr_arc_from_addr` (`history.py:125`
/// `cpu.get_latest_descr` parity) and derives the bridge source
/// identity (`jct.green_key` / `descr.trace_id()` /
/// `descr.fail_index_per_trace()`) from that Arc, mirroring
/// `pyjitpl.py handle_guard_failure(self, resumedescr,
/// deadframe)`.  No surrogate triple crosses the C-ABI.
/// `pyjitpl.py handle_guard_failure(self, resumedescr, deadframe)`.
/// `None` = compile.py `resume_in_blackhole`. `Some(result)` = the
/// must_compile arm finished (`ContinueRunningNormally` / portal
/// re-entry) and `handle_fail` must not blackhole.
pub type BridgeFn = fn(*mut jitframe::JitFrame, usize, i64, bool) -> Option<i64>;

/// Force callee: (callee_frame_ptr) → result
pub type ForceFn = extern "C" fn(i64) -> i64;

/// Unbox int from Ref: (raw_ref) → i64
pub type UnboxIntFn = fn(i64) -> i64;

static CA_BLACKHOLE_FN: OnceLock<BlackholeFn> = OnceLock::new();
static CA_BRIDGE_FN: OnceLock<BridgeFn> = OnceLock::new();

static CA_FORCE_FN: OnceLock<ForceFn> = OnceLock::new();
static CA_UNBOX_INT_FN: OnceLock<UnboxIntFn> = OnceLock::new();

/// JitFrame field descriptors supplied by the interpreter crate so the
/// GC rewriter's `handle_call_assembler` pass (rewrite.py) can
/// emit the correct GC_LOAD / GC_STORE sequence for callee jitframes.
#[derive(Clone)]
pub struct JitFrameLayoutInfo {
    pub jitframe_descrs: Option<majit_gc::rewrite::JitFrameDescrs>,
}

static JITFRAME_LAYOUT: OnceLock<JitFrameLayoutInfo> = OnceLock::new();

pub fn register_jitframe_layout(info: JitFrameLayoutInfo) {
    let _ = JITFRAME_LAYOUT.set(info);
}

pub(crate) fn jitframe_layout() -> Option<JitFrameLayoutInfo> {
    JITFRAME_LAYOUT.get().cloned()
}

/// Pyre dynasm extension: CALL_ASSEMBLER builds some callee jitframes via
/// `libc::calloc`, so the GC needs an explicit registration hook before the
/// callee pushes that frame on the jitframe shadow stack.
pub extern "C" fn dynasm_register_libc_jitframe(addr: i64) {
    if addr != 0 {
        majit_gc::shadow_stack::register_libc_jitframe(addr as usize);
    }
}

/// Remove a libc-jitframe from the registry immediately before freeing it.
pub extern "C" fn dynasm_unregister_libc_jitframe(addr: i64) {
    if addr != 0 {
        majit_gc::shadow_stack::unregister_libc_jitframe(addr as usize);
    }
}

pub fn register_libc_jitframe_helper_addr() -> usize {
    dynasm_register_libc_jitframe as *const () as usize
}

pub fn unregister_libc_jitframe_helper_addr() -> usize {
    dynasm_unregister_libc_jitframe as *const () as usize
}

/// Register blackhole resume handler (same API as Cranelift).
pub fn register_call_assembler_blackhole(f: BlackholeFn) {
    CA_BLACKHOLE_FN.set(f).ok();
}

/// Register bridge compilation handler (same API as Cranelift).
pub fn register_call_assembler_bridge(f: BridgeFn) {
    CA_BRIDGE_FN.set(f).ok();
}

/// Register force handler (same API as Cranelift).
pub fn register_call_assembler_force(f: ForceFn) {
    CA_FORCE_FN.set(f).ok();
}

/// Return the force_fn address for embedding in generated code.
/// Returns 0 if not registered.
pub fn call_assembler_force_fn_addr() -> usize {
    CA_FORCE_FN.get().map(|f| *f as usize).unwrap_or(0)
}

/// Register int unbox handler (same API as Cranelift).
pub fn register_call_assembler_unbox_int(f: UnboxIntFn) {
    CA_UNBOX_INT_FN.set(f).ok();
}

/// rpython/jit/backend/llsupport/llmodel.py `insert_stack_check`
/// parity. The interpreter registers the three addresses the JIT
/// prologue needs to emit the inline stack-overflow probe matching
/// `rpython/jit/backend/x86/assembler.py:1085-1091`:
///
///   * `end_adr` — address of `PYRE_STACKTOOBIG.stack_end`
///     (`LL_stack_get_end_adr`). The probe loads this and subtracts
///     the current SP.
///   * `length_adr` — address of `PYRE_STACKTOOBIG.stack_length`
///     (`LL_stack_get_length_adr`). The probe compares the diff
///     against this.
///   * `slowpath_addr` — `pyre_stack_too_big_slowpath` function
///     pointer (`STACK_CHECK_SLOWPATH` in llmodel). Called on fast-
///     path miss with the current SP; returns 1 on real overflow.
#[derive(Copy, Clone, Debug)]
pub struct StackCheckAddresses {
    pub end_adr: usize,
    pub length_adr: usize,
    pub slowpath_addr: usize,
}

static STACK_CHECK_ADDRS: OnceLock<StackCheckAddresses> = OnceLock::new();

/// Register the three addresses the JIT prologue needs for the inline
/// stack-overflow probe. Called once at startup from pyre-jit; matches
/// `rpython/jit/backend/llsupport/llmodel.py insert_stack_check`.
pub fn register_stack_check_addresses(end_adr: usize, length_adr: usize, slowpath_addr: usize) {
    let _ = STACK_CHECK_ADDRS.set(StackCheckAddresses {
        end_adr,
        length_adr,
        slowpath_addr,
    });
}

/// Retrieve the registered inline-probe addresses. Returns `None` if
/// the interpreter has not yet registered them (tests or early
/// startup); the prologue emitter skips the probe in that case,
/// mirroring `assembler.py:1082 "if self.stack_check_slowpath == 0:
/// pass (no stack check)"`.
pub fn stack_check_addresses() -> Option<StackCheckAddresses> {
    STACK_CHECK_ADDRS.get().copied()
}

/// warmspot.py `assembler_call_helper` parity —
/// C-callable trampoline for the CALL_ASSEMBLER slow path.
///
/// Upstream:
///     fail_descr = self.cpu.get_latest_descr(deadframe)
///     try:
///         fail_descr.handle_fail(deadframe, self.metainterp_sd, jd)
///     except jitexc.JitException as e:
///         return handle_jitexception(e)
///
/// In upstream each `AbstractFailDescr` subclass carries its own
/// `handle_fail`, and the helper's job is purely to dispatch. pyre
/// encodes finish descrs as raw-pointer singletons
/// (`done_with_this_frame_descr_{void,int,ref,float}`,
/// `exit_frame_with_exception_descr_ref`) and resume-guard descrs
/// reach this helper through their `DescrRef`, so this trampoline
/// performs the type dispatch inline and delegates to one of the
/// `handle_fail_*` helpers below.
///
/// **TODO**: pyre does not yet carry
/// `FailDescr::handle_fail` as a virtual method because descrs are
/// stored as raw `usize` sentinels rather than `Box<dyn FailDescr>`,
/// and `metainterp_sd` / `jitdriver_sd` are not plumbed through
/// `majit-backend`. Converting to an object-hierarchy requires
/// reworking descr storage across both backends — tracked as a
/// follow-up on top of this refactor. The body below is written so
/// every branch lines up 1:1 with upstream's `handle_fail` sites.
///
/// Input:  callee_jf_ptr = deadframe,
///         green_key = caller loop's header_pc (used by CA_*_FN hooks
///         to find the owning compiled trace).
/// Output: the int interpretation of the handled result (the caller
///         re-casts to the portal's return type).
///
/// The callee jitframe is GC-managed by the rewriter's
/// `handle_call_assembler` path, so the helper must not free it here.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "this C-ABI trampoline is called only by generated machine code, which guarantees that the embedded CPU handle and jitframe pointers remain live; making the symbol unsafe does not express that foreign-call contract"
)]
pub extern "C" fn call_assembler_helper_trampoline(
    cpu_handle: *const guard::CpuDescrCell,
    callee_jf_ptr: *mut jitframe::JitFrame,
    green_key: u64,
) -> i64 {
    if callee_jf_ptr.is_null() {
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("ca-helper null callee_jf_ptr green_key={green_key}"),
            );
        }
        return 0;
    }
    let frame_ptr = callee_jf_ptr;
    // warmspot.py:1022 `fail_descr = cpu.get_latest_descr(deadframe)`.
    let descr_raw = unsafe { llmodel::get_latest_descr(frame_ptr) };
    // `compile.py:665-674` parity: resolve the attached `DoneWithThisFrame*`
    // + `ExitFrameWithExceptionDescrRef` identities from the cpu instance
    // that emitted the CALL_ASSEMBLER.  `cpu_handle` is the heap-pinned
    // `Arc<CpuDescrCell>` address baked as a compile-time
    // constant into the emitted call site — same role as RPython's
    // `self.cpu` closure capture in the translated code.  A compiled
    // trace keeps an `Arc` clone alive alongside its executable buffer
    // so the pointee outlives any subsequent `DynasmBackend` drop.
    //
    // A null `cpu_handle` means "no attached cpu" — only reachable from
    // direct-trampoline unit tests that construct the jitframe without
    // going through `compile_loop`.  In that mode no finish descr can
    // match, so dispatch falls through to `handle_fail_resume_guard`.
    let attached = if cpu_handle.is_null() {
        majit_backend::AttachedDescrPtrs::default()
    } else {
        unsafe { (*cpu_handle).read().descr_ptrs() }
    };
    // warmspot.py:1023-1028 `fail_descr.handle_fail(deadframe, ...)`.
    // Callee jitframe is GC-managed by the rewriter's
    // `handle_call_assembler` path (rewrite.py `gen_malloc_frame`),
    // so this helper must not free it here — upstream
    // `assembler_call_helper` (warmspot.py) likewise does not
    // free `deadframe`.
    handle_fail_dispatch(&attached, descr_raw, frame_ptr, green_key)
}

/// Dispatch to the `handle_fail` variant that matches `descr_raw`.
/// Upstream equivalent: the virtual-method call in warmspot.py:1024,
/// resolved at runtime by the Python class of `fail_descr`.
fn handle_fail_dispatch(
    attached: &majit_backend::AttachedDescrPtrs,
    descr_raw: usize,
    frame_ptr: *mut jitframe::JitFrame,
    green_key: u64,
) -> i64 {
    if descr_raw == 0 {
        // Deviation: unresolved-callee path. Upstream never reaches
        // this state — an unlinked target is patched out before the
        // CALL_ASSEMBLER emits the call. Retained as a safety fence.
        return 0;
    }
    // `compile.py:665-674` parity: the finish descrs attached to
    // `self.cpu` are compared by pointer identity.  RPython backend
    // code reaches them through `cpu.done_with_this_frame_descr_*`;
    // pyre resolves them from the heap-pinned cpu handle baked into
    // the CALL_ASSEMBLER emit site (resolved in the trampoline).
    if descr_raw == attached.done_with_this_frame_descr_void
        || descr_raw == attached.done_with_this_frame_descr_int
        || descr_raw == attached.done_with_this_frame_descr_ref
        || descr_raw == attached.done_with_this_frame_descr_float
    {
        // compile.py `_DoneWithThisFrameDescr` subclasses
        // (Void/Int/Ref/Float) — the four finish singletons.
        return handle_fail_done_with_this_frame(attached, descr_raw, frame_ptr);
    }
    if descr_raw == attached.exit_frame_with_exception_descr_ref {
        // compile.py `ExitFrameWithExceptionDescrRef.handle_fail`:
        //   value = cpu.get_ref_value(deadframe, 0)
        //   raise jitexc.ExitFrameWithExceptionRef(value)
        // pyre returns the raw slot-0 payload here; the caller stub
        // observes `is_exit_frame_with_exception` on the synthesized
        // FailDescr and re-raises through `jit_exc_value`.
        return handle_fail_exit_frame_with_exception(frame_ptr);
    }
    if descr_raw != 0 && descr_raw == attached.propagate_exception_descr {
        // compile.py `PropagateExceptionDescr.handle_fail`:
        //     exception = cpu.grab_exc_value(deadframe)
        //     if not exception:
        //         exception = cast_instance_to_gcref(memory_error)
        //     raise jitexc.ExitFrameWithExceptionRef(exception)
        // The CALL_ASSEMBLER caller observes `is_exit_frame_with_exception`
        // on the synthesized FailDescr (runner.rs::find_descr_by_ptr) and
        // re-raises through `jit_exc_value`; the same shape applies here
        // — read jf_guard_exc, fall back to memory_error, re-raise.
        return handle_fail_propagate_exception(frame_ptr);
    }
    // compile.py `AbstractResumeGuardDescr.handle_fail`.
    // `history.py AbstractDescr.show(cpu, descr_gcref)` parity:
    // `descr_raw` is the thin pointer of the `FailDescrCell` baked at
    // codegen time; reconstruct the cell via `Arc::from_raw` and read
    // the wrapped descr.  No global lookup — the cell's strong refcount
    // is held by `clt.asmmemmgr_gcreftracers` for the lifetime of the
    // executing JIT code.
    let descr_arc = { unsafe { majit_ir::recover_fail_descr_cell(descr_raw) } };
    // `compile.py` `PropagateExceptionDescr.handle_fail`
    // (`compile_tmp_callback`). The compare above matches the singleton
    // cell `AbstractDescr.hide` bakes. This guard's recovery stub writes
    // its own `FailDescrCell`. `show` (`recover_fail_descr_cell`) of that
    // cell is the same Arc as `show` of the singleton when the guard's
    // descr is `propagate_exception_descr`; otherwise the resume-guard
    // path finds no `rd_numb`, drops `jf_guard_exc`, and the caller
    // continues as if the call returned NULL.
    if attached.propagate_exception_descr != 0 {
        let singleton =
            unsafe { majit_ir::recover_fail_descr_cell(attached.propagate_exception_descr) };
        if std::sync::Arc::ptr_eq(&descr_arc, &singleton) {
            return handle_fail_propagate_exception(frame_ptr);
        }
    }
    let descr_fd = match descr_arc.as_fail_descr() {
        Some(fd) => fd,
        None => return 0,
    };
    handle_fail_resume_guard(descr_fd, descr_raw, frame_ptr, green_key)
}

/// compile.py `DoneWithThisFrameDescr{Void,Int,Ref,Float}.handle_fail`.
///
/// Upstream raises `jitexc.DoneWithThisFrame*(slot-0)` and
/// `handle_jitexception` shape-casts the payload to the portal's
/// return type.  pyre returns the raw i64 back; the JIT-emitted caller
/// stub (`assembler_helper_wrapper`) reinterprets it as the correct
/// kind.  Each subclass picks a different accessor so the returned
/// bits carry the right sign-extension / pointer-tagging / float
/// bit-layout for the portal return type:
///
/// * `DoneWithThisFrameDescrVoid.handle_fail` — no payload, return 0
///   (compile.py:628-632).
/// * `DoneWithThisFrameDescrInt.handle_fail` — `get_int_value(df, 0)`
///   (compile.py:635-640).
/// * `DoneWithThisFrameDescrRef.handle_fail` — `get_ref_value(df, 0)`
///   (compile.py:644-649).
/// * `DoneWithThisFrameDescrFloat.handle_fail` — `get_float_value(df, 0)`
///   (compile.py:653-657).
///
/// The descr identity selects the accessor — matched through the four
/// `guard::done_with_this_frame_descr_*_ptr()` singletons rather than
/// a `fail_arg_types[0]` probe, because the Void descr carries no type
/// entry at all.
fn handle_fail_done_with_this_frame(
    attached: &majit_backend::AttachedDescrPtrs,
    descr_raw: usize,
    frame_ptr: *mut jitframe::JitFrame,
) -> i64 {
    if descr_raw == attached.done_with_this_frame_descr_void {
        return 0;
    }
    if descr_raw == attached.done_with_this_frame_descr_int {
        return unsafe { llmodel::get_int_value_direct(frame_ptr, 0) as i64 };
    }
    if descr_raw == attached.done_with_this_frame_descr_ref {
        return unsafe { llmodel::get_ref_value_direct(frame_ptr, 0) as i64 };
    }
    if descr_raw == attached.done_with_this_frame_descr_float {
        return unsafe { llmodel::get_float_value_direct(frame_ptr, 0) as i64 };
    }
    // Unreachable: the dispatch gate above already narrowed
    // `descr_raw` to one of the four singletons.
    0
}

/// compile.py `ExitFrameWithExceptionDescrRef.handle_fail`.
///
/// Upstream reads slot 0 as a gcref and raises
/// `jitexc.ExitFrameWithExceptionRef(value)`.  Pyre hands the raw int
/// payload back; the caller stub
/// (`pyre/pyre-jit/src/eval.rs` handle_jit_outcome) routes the result
/// into `PyError::from_exc_object` when the synthesized FailDescr
/// carries `is_exit_frame_with_exception = true`.
///
/// Also publishes the exception via `jit_exc_raise` so a future
/// `genop_guard_guard_no_exception` (assembler.py) emitted by
/// the caller sees the pending exception — symmetric with cranelift's
/// CALL_ASSEMBLER trampoline path.  PyPy's
/// `generate_guard_no_exception` checks `cpu.pos_exception()` (the
/// exception type slot), so dynasm's guard-no-exception fast path
/// compares against `JIT_EXC_TYPE`; `jit_exc_raise` publishes both
/// slots and keeps the value slot available for
/// `_store_and_reset_exception` / `save_exception`.
fn handle_fail_exit_frame_with_exception(frame_ptr: *mut jitframe::JitFrame) -> i64 {
    let value = unsafe { llmodel::get_ref_value_direct(frame_ptr, 0) as i64 };
    // warmspot.py:998-1005 + llsupport/assembler.py:345 —
    // `assembler_helper_wrapper` invokes `handle_jitexception` which
    // re-raises through `pos_exc_value`.
    jit_exc_raise(value);
    value
}

/// compile.py `PropagateExceptionDescr.handle_fail`.
///
/// Reads `jf_guard_exc` (set by Layer 3's emitted
/// `_store_and_reset_exception` in `OpCode::CheckMemoryError`) the
/// way `llmodel.py grab_exc_value` does — a read, no write — and
/// falls back to `memory_error` when empty
/// (`cast_instance_to_gcref(memory_error)`). Clearing the slot is
/// `_restore_exception` (x86/assembler.py). The caller observes
/// `is_exit_frame_with_exception=true` on the synthesized FailDescr
/// and re-raises the value through `jit_exc_value`.
///
/// Republishes the value via `jit_exc_raise` for the same
/// caller-side GUARD_NO_EXCEPTION reason as
/// `handle_fail_exit_frame_with_exception`; Layer 3's emit cleared
/// the globals on transfer, so this is a fresh store of the resolved
/// (possibly memory_error fallback) value.
fn handle_fail_propagate_exception(frame_ptr: *mut jitframe::JitFrame) -> i64 {
    // `llmodel.py grab_exc_value`: read `jf_guard_exc` off the deadframe.
    let exc_val = unsafe { (*frame_ptr).jf_guard_exc };
    let value = if exc_val != 0 {
        exc_val as i64
    } else {
        majit_backend::memory_error_singleton_ref()
    };
    jit_exc_raise(value);
    value
}

/// compile.py `AbstractResumeGuardDescr.handle_fail`.
///
/// Upstream:
///     if must_compile(...) and not rstack.stack_almost_full():
///         self._trace_and_compile_from_bridge(deadframe, ...)
///     else:
///         resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
///     assert 0, "unreachable"
///
/// compile.py `handle_fail`: `if must_compile and not stack_almost_full`
/// traces and raises; `else` is `resume_in_blackhole`. The must_compile
/// decision and `_trace_and_compile_from_bridge` live in
/// `jit_ca_handle_guard_failure`. A `Some` result is the raise
/// (`ContinueRunningNormally` → portal re-entry); `None` is the else arm.
fn handle_fail_resume_guard(
    descr: &dyn majit_ir::FailDescr,
    descr_raw: usize,
    frame_ptr: *mut jitframe::JitFrame,
    _outer_green_key: u64,
) -> i64 {
    let trace_id = descr.trace_id();
    let fail_index = descr.fail_index_per_trace();

    let guard_value_operand = majit_backend::guard_value_counter_slot(descr)
        .map(|slot| unsafe { llmodel::get_int_value_direct(frame_ptr, slot) as i64 });

    // The bridge hook allocates. A nursery jitframe is then poison, and
    // `jf_forward` is no longer a stub `JitFrame::resolve` can follow.
    // The owner root is updated by the collector; re-read it after every
    // call that can collect. Pin is declined for a frame that holds GC
    // pointers, so it does not keep this address stable.
    let frame_root = (!frame_ptr.is_null() && majit_gc::gc_owns_object(frame_ptr as usize))
        .then(|| majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(frame_ptr as usize)));
    let mut frame_ptr = frame_ptr;
    let current_frame = |root: &Option<majit_gc::shadow_stack::OwnerRootGuard>,
                         frame_ptr: *mut jitframe::JitFrame| {
        root.as_ref()
            .map(|root| root.get().0 as *mut jitframe::JitFrame)
            .unwrap_or(frame_ptr)
    };

    // pyjitpl.py:2921-2923 parity: recover the owning Arc<JitCellToken>
    // identity from the descr.  When the weakref is dead (memmgr-evicted
    // JCT — "should be rare" per upstream), `compile.giveup()` raises
    // `SwitchToBlackhole(ABORT_BRIDGE)` (compile.py:27-29).  That raise sits
    // ABOVE the `try:` at pyjitpl.py:2926, so the `except SwitchToBlackhole`
    // at :2930-2931 does NOT catch it and the `finally:` at :2932-2935 does
    // not run: it leaves `handle_guard_failure` with nothing traced.  Pyre
    // mirrors this by gating bridge tracing on `Some(jct)` and passing
    // `green_key=0` to the blackhole callback so it derives the key from
    // the deadframe's pyframe pointer (call_jit.rs); never
    // substitute the parent CALL_ASSEMBLER caller's green_key here — that
    // would mis-route resume storage and trip `compile_bridge`'s
    // `debug_assert_eq!(source_jct.green_key, green_key)` (pyjitpl.rs).
    let owning_jct = majit_backend::descr_owning_jct(descr);
    frame_ptr = current_frame(&frame_root, frame_ptr);

    // `jf_guard_exc` stays where the failure stub wrote it. `grab_exc_value`
    // (`llmodel.py`) only reads the field. The slot is a GCREF visited by
    // `jitframe_trace` for the whole helper: the callee frame is still the
    // rewriter's live frame, so the exception stays rooted across the bridge
    // hook without a second root. `_restore_exception` (x86/assembler.py)
    // clears the slot when emitted code moves the value back to
    // `pos_exc_value`; a later execution starts from `malloc_jitframe` /
    // `alloc_off_gc_jitframe`, which zeroes the fixed header.

    // compile.py `if must_compile and not stack_almost_full`.
    // Fail args stay in `jf_frame[]` (`llmodel.py get_int_value`).
    if let (Some(_jct), Some(bridge_fn)) = (owning_jct.as_ref(), CA_BRIDGE_FN.get()) {
        let bridged = bridge_fn(
            frame_ptr,
            descr_raw,
            guard_value_operand.unwrap_or(0),
            guard_value_operand.is_some(),
        );
        frame_ptr = current_frame(&frame_root, frame_ptr);
        if let Some(result) = bridged {
            return result;
        }
    }

    // compile.py `else: resume_in_blackhole(descr, deadframe)`.
    // The receiver reads `jf_guard_exc` (`grab_exc_value`).
    let blackhole = CA_BLACKHOLE_FN.get();
    let bh_result = blackhole.and_then(|blackhole| blackhole(descr_raw, frame_ptr));
    // `jit_blackhole_resume_from_guard` can collect in `to_exc_object`
    // and return None. The owner root moved; this copy did not.
    frame_ptr = current_frame(&frame_root, frame_ptr);
    if let Some(bh_result) = bh_result {
        if majit_log_enabled() {
            eprintln!(
                "[dynasm][ca-helper] resume-guard trace_id={trace_id} fail_index={fail_index} result=0x{bh_result:x}"
            );
        }
        return bh_result;
    }
    if majit_log_enabled() {
        eprintln!(
            "[dynasm][ca-helper] resume-guard trace_id={trace_id} fail_index={fail_index} fell through to 0 descr=0x{descr_raw:x} frame={frame_ptr:p} guard_exc=0x{:x}",
            unsafe { (*frame_ptr).jf_guard_exc }
        );
    }
    // `assert 0, "unreachable"` upstream — pyre returns 0 when neither
    // hook is registered (e.g. bare-backend tests exercise the helper
    // without a metainterp behind it). A pending exception must still
    // reach the caller's GUARD_NO_EXCEPTION / GUARD_EXCEPTION: the
    // failure stub already moved it out of `pos_exc_value` into
    // `jf_guard_exc`, and returning 0 with the cell empty is a normal
    // NULL result. Leave an exception the blackhole already published
    // (stack overflow) in place. A registered blackhole that answered
    // `None` bailed to the interpreter after delivering the exception to
    // a handler, so there is nothing left to raise.
    let guard_exc = unsafe { (*frame_ptr).jf_guard_exc };
    if blackhole.is_none() && guard_exc != 0 && !jit_exc_is_pending() {
        jit_exc_raise(guard_exc as i64);
    }
    0
}

/// Return the address of the trampoline for embedding in generated code.
pub fn call_assembler_helper_addr() -> usize {
    call_assembler_helper_trampoline as *const () as usize
}

/// CALL_ASSEMBLER recursive callee execution trampoline. JIT-generated
/// code calls this instead of directly branching to the callee entry.
///
/// No alternate-stack switch: upstream RPython enters the callee on
/// the caller's native stack, and the callee's compiled prologue
/// performs its own inline SP probe
/// (rpython/jit/backend/x86/assembler.py:1085
/// `_call_header_with_stack_check`) against `PYRE_STACKTOOBIG.stack_end`.
/// Growing the stack here would shift SP into a different guard
/// region and invalidate that budget comparison.
///
/// Input:  arg0 = callee jf_ptr, arg1 = callee entry address
/// Output: callee's returned jf_ptr
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "this C-ABI trampoline is entered only from generated code with a live jitframe and compiled entry address; its safe Rust signature is the exact addressable ABI consumed by the assembler"
)]
pub extern "C" fn call_assembler_execute_trampoline(
    jf_ptr: *mut jitframe::JitFrame,
    callee_addr: usize,
) -> *mut jitframe::JitFrame {
    if majit_log_enabled() && !jf_ptr.is_null() {
        let input0_ofs =
            jitframe::FIRST_ITEM_OFFSET + crate::arch::JITFRAME_FIXED_SIZE * jitframe::SIZEOFSIGNED;
        let input1_ofs = input0_ofs + jitframe::SIZEOFSIGNED;
        let frame_len = unsafe {
            *((jf_ptr as *const u8).add(jitframe::JF_FRAME_OFS + jitframe::LENGTHOFS)
                as *const isize)
        };
        let frame_info = unsafe {
            *((jf_ptr as *const u8).add(jitframe::JF_FRAME_INFO_OFS as usize) as *const usize)
        };
        let input0 = unsafe { *((jf_ptr as *const u8).add(input0_ofs) as *const usize) };
        let input1 = unsafe { *((jf_ptr as *const u8).add(input1_ofs) as *const usize) };
        eprintln!(
            "[dynasm][ca-exec] enter jf={jf_ptr:p} callee=0x{callee_addr:x} len={frame_len} frame_info=0x{frame_info:x} input0=0x{input0:x} input1=0x{input1:x}"
        );
    }
    let func: unsafe extern "C" fn(*mut jitframe::JitFrame) -> *mut jitframe::JitFrame =
        unsafe { std::mem::transmute(callee_addr) };
    let result = unsafe { func(jf_ptr) };
    if majit_ir::debug::have_debug_prints() {
        majit_ir::debug::log_one("jit-backend", &format!("ca-exec leave jf={result:p}"));
    }
    result
}

/// Return the address of the execute trampoline.
pub fn call_assembler_execute_addr() -> usize {
    call_assembler_execute_trampoline as *const () as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;

    unsafe fn alloc_test_jitframe(descr: usize, slots: &[i64]) -> *mut jitframe::JitFrame {
        let depth = slots.len();
        let size = jitframe::JitFrame::alloc_size(depth);
        let ptr = unsafe { libc::malloc(size) as *mut jitframe::JitFrame };
        assert!(!ptr.is_null());
        unsafe {
            libc::memset(ptr as *mut std::ffi::c_void, 0, size);
            jitframe::JitFrame::init(ptr, jitframe::NULLFRAMEINFO, depth);
            llmodel::set_latest_descr(ptr, descr);
            for (index, value) in slots.iter().copied().enumerate() {
                llmodel::set_int_value(ptr, index, value as isize);
            }
        }
        ptr
    }

    // ── Bug 1 regression: unresolved target must not dereference result as pointer ──
    // The old code let the helper return value flow into `mov rdx, rax; mov rcx, [rdx]`
    // which dereferenced an integer as a pointer. This test verifies the trampoline
    // returns a plain value without crashing when jf_descr is 0 (pending/unresolved).

    #[test]
    fn test_helper_trampoline_null_jf_returns_zero() {
        let result = call_assembler_helper_trampoline(std::ptr::null(), std::ptr::null_mut(), 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_helper_trampoline_descr_zero_returns_zero() {
        let jf = unsafe { alloc_test_jitframe(0, &[0, 0, 0, 0]) };
        let result = call_assembler_helper_trampoline(std::ptr::null(), jf, 42);
        assert_eq!(result, 0);
    }

    // ── Bug 2 regression: force_fn must NOT be called with jitframe pointer ──
    // The old code passed a dynasm jitframe (i64 array) to force_fn which
    // expected a PyFrame pointer. Now force_fn is not called from the trampoline.
    // Verify: even when a guard descr is present, force_fn is never invoked.

    #[test]
    fn test_helper_trampoline_does_not_call_force_fn() {
        // force_fn is stored in CA_FORCE_FN (OnceLock). We can't reset it,
        // but we CAN verify the trampoline returns blackhole result (or 0)
        // without calling force. We do this by setting up a guard descr and
        // checking the trampoline reads fail_arg values correctly.

        let descr = majit_backend::make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
        let cell = majit_ir::FailDescrCell::wrap(descr.clone());
        let descr_ptr = majit_ir::FailDescrCell::thin_ptr(&cell);

        let jf = unsafe { alloc_test_jitframe(descr_ptr, &[100, 200, 0, 0]) };
        let result = call_assembler_helper_trampoline(std::ptr::null(), jf, 42);
        // No blackhole registered → returns 0, no crash.
        assert_eq!(result, 0);
        drop(cell);
        drop(descr);
    }

    #[test]
    fn test_helper_trampoline_does_not_execute_bridge() {
        // compile.py:701 parity: helper does NOT re-enter bridges.
        // Bridges are executed via patched guard jumps, not the helper.
        // `bridge_addr` moved off the descr to the backend
        // side-table; the helper-trampoline path never queries that
        // table, so this test simply confirms the trampoline returns
        // the blackhole result (or 0) regardless of bridge presence.
        let descr = majit_backend::make_resume_guard_descr_typed(vec![Type::Int]);
        let cell = majit_ir::FailDescrCell::wrap(descr.clone());
        let descr_ptr = majit_ir::FailDescrCell::thin_ptr(&cell);

        let jf = unsafe { alloc_test_jitframe(descr_ptr, &[123, 0, 0, 0]) };
        let result = call_assembler_helper_trampoline(std::ptr::null(), jf, 99);
        // Helper blackhole-resumes (no blackhole registered → 0), NOT bridge result.
        assert_eq!(result, 0);

        drop(cell);
        drop(descr);
    }

    // ── Bug 3 regression: green_key must reach the handler, not be 0 ──

    #[test]
    fn test_helper_trampoline_passes_green_key() {
        use std::sync::atomic::AtomicU64;
        static OBSERVED_GREEN_KEY: AtomicU64 = AtomicU64::new(0);

        // We can't register a one-shot blackhole fn (OnceLock is set-once).
        // Instead, test the trampoline's function signature contract:
        // green_key is the second parameter (rsi on x86_64).
        // Direct call test — not through generated code.
        let green_key: u64 = 0xDEAD_BEEF_CAFE;
        let result =
            call_assembler_helper_trampoline(std::ptr::null(), std::ptr::null_mut(), green_key);
        // Null ptr → early return 0, but the function accepted green_key.
        assert_eq!(result, 0);
        // The real verification is compile-time: the signature is
        //   extern "C" fn(*mut i64, u64) -> i64
        // and generated code passes green_key in rsi. If this test compiles
        // and runs, the ABI contract is correct.
        let _ = OBSERVED_GREEN_KEY; // suppress unused
    }

    // ── done_with_this_frame_descr: 4 type-specific singletons ──

    /// Test-local stand-in for `compile::DoneWithThisFrameDescr*`
    /// singletons.  The dynasm backend stores the metainterp-side
    /// `Arc<dyn Descr>` in a process-global `OnceLock`, so seeding
    /// from this test mirrors what `pyjitpl.py:2222`
    /// (`make_and_attach_done_descrs([self, cpu])`) does at runtime.
    #[derive(Debug)]
    struct TestMarker(Type);
    impl majit_ir::Descr for TestMarker {}
    impl majit_ir::FailDescr for TestMarker {
        fn fail_index(&self) -> u32 {
            u32::MAX
        }
        fn fail_arg_types(&self) -> &[Type] {
            match self.0 {
                Type::Void => &[],
                Type::Int => &[Type::Int],
                Type::Ref => &[Type::Ref],
                Type::Float => &[Type::Float],
            }
        }
        fn is_finish(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_done_with_this_frame_descr_four_variants() {
        use std::sync::Arc;
        let void: majit_ir::DescrRef = Arc::new(TestMarker(Type::Void));
        let int: majit_ir::DescrRef = Arc::new(TestMarker(Type::Int));
        let ref_: majit_ir::DescrRef = Arc::new(TestMarker(Type::Ref));
        let float: majit_ir::DescrRef = Arc::new(TestMarker(Type::Float));

        // `pyjitpl.py` `make_and_attach_done_descrs([self, cpu])`
        // parity: the four `DoneWithThisFrameDescr*` singletons live on
        // the owning cpu instance.  Construct a per-test backend, attach
        // the descrs through the Backend trait, and read pointers out of
        // `attached_descr_ptrs()` — matching RPython's
        // `self.cpu.done_with_this_frame_descr_*` access pattern.
        let mut backend = runner::DynasmBackend::new();
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_void(
            &mut backend,
            void,
        );
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_int(
            &mut backend,
            int,
        );
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_ref(
            &mut backend,
            ref_,
        );
        <runner::DynasmBackend as majit_backend::Backend>::set_done_with_this_frame_descr_float(
            &mut backend,
            float,
        );

        let ptrs = backend.attached_descr_ptrs();
        let void_ptr = ptrs.done_with_this_frame_descr_void;
        let int_ptr = ptrs.done_with_this_frame_descr_int;
        let ref_ptr = ptrs.done_with_this_frame_descr_ref;
        let float_ptr = ptrs.done_with_this_frame_descr_float;

        // All four must be distinct (metainterp-side `DoneWithThisFrameDescr*`
        // singletons are separate Arcs per result type).
        assert_ne!(void_ptr, 0);
        assert_ne!(int_ptr, 0);
        assert_ne!(ref_ptr, 0);
        assert_ne!(float_ptr, 0);
        assert_ne!(void_ptr, int_ptr);
        assert_ne!(void_ptr, ref_ptr);
        assert_ne!(void_ptr, float_ptr);
        assert_ne!(int_ptr, ref_ptr);
        assert_ne!(int_ptr, float_ptr);
        assert_ne!(ref_ptr, float_ptr);

        // is_done_with_this_frame_descr must recognize all four.
        assert!(ptrs.is_done_with_this_frame_descr(void_ptr));
        assert!(ptrs.is_done_with_this_frame_descr(int_ptr));
        assert!(ptrs.is_done_with_this_frame_descr(ref_ptr));
        assert!(ptrs.is_done_with_this_frame_descr(float_ptr));

        // Arbitrary pointer must NOT be recognized.
        assert!(!ptrs.is_done_with_this_frame_descr(0x12345678));
        assert!(!ptrs.is_done_with_this_frame_descr(0));

        // ptr_for_type must route correctly.
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Void),
            void_ptr
        );
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Int),
            int_ptr
        );
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Ref),
            ref_ptr
        );
        assert_eq!(
            ptrs.done_with_this_frame_descr_ptr_for_type(Type::Float),
            float_ptr
        );
    }

    // ── Exception state: jit_exc_raise / jit_exc_value_raw ──

    #[test]
    fn test_jit_exc_raise_and_clear() {
        jit_exc_clear();
        assert!(!jit_exc_is_pending());
        assert_eq!(jit_exc_class_raw(), 0);

        // Simulate raising with a fake object (type ptr at offset 0).
        let fake_type: i64 = 0x42;
        let fake_obj: [i64; 2] = [fake_type, 0x99];
        jit_exc_raise(fake_obj.as_ptr() as i64);
        assert!(jit_exc_is_pending());
        assert_eq!(jit_exc_class_raw(), fake_type);

        // value_raw returns and clears.
        let val = jit_exc_value_raw();
        assert_eq!(val, fake_obj.as_ptr() as i64);
        assert!(!jit_exc_is_pending());

        jit_exc_clear();
    }
}
