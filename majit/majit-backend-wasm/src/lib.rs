/// WebAssembly backend for majit.
///
/// Generates wasm bytecodes via wasm-encoder. On wasm32 targets,
/// instantiates the emitted trace modules through a host binding (see
/// `glue`): the `web` feature uses the browser `WebAssembly` API via
/// wasm-bindgen, the `host-import` feature uses plain wasm imports that a
/// native embedder (wasmi / wasmtime) supplies. On native targets,
/// compile_loop succeeds but execute_token requires a wasm host
/// (unreachable natively).
///
/// Which binding a build has is a feature, not a target OS: `host-import` is
/// satisfied by an embedder, and an embedder is exactly what a WASI command
/// runs under. So the split below is wasm32 against native, and a wasm32 build
/// with no binding selected keeps `glue`'s stubs rather than being routed to
/// the native arm.
pub mod codegen;
pub mod failguard;
mod func_sig;
mod release;

pub use func_sig::{FuncSigVal, WasmSig, decode_func_sig, encode_func_sig};

/// The wasm host compiles and resumes on the thread that ran the
/// compiled frame (`eval.rs` post-`run_compiled`). cargo's default
/// harness is N threads against one process-global cpu
/// (finish cells, ExtraHeap, `cpu.gc_ll_descr`). PyPy never
/// interleaves those. Cargo.toml has no per-package `test-threads`;
/// `#[serial]` only serializes bodies (TLS teardown still races).
/// This constructor runs before libtest's `main` reads
/// `RUST_TEST_THREADS`, so this crate's test binary is one thread
/// without the caller passing `--test-threads`. Other crates stay
/// parallel (`cargo test --all` is one process per crate).
#[cfg(all(test, not(target_arch = "wasm32")))]
mod serial_cpu_tests {
    extern "C" fn set_one_test_thread() {
        // SAFETY: constructor runs before `main`, single-threaded.
        // libtest reads `RUST_TEST_THREADS` in `main`.
        unsafe { std::env::set_var("RUST_TEST_THREADS", "1") };
    }

    #[used]
    #[cfg_attr(
        any(target_os = "macos", target_os = "ios"),
        unsafe(link_section = "__DATA,__mod_init_func")
    )]
    #[cfg_attr(
        any(target_os = "linux", target_os = "android", target_os = "freebsd"),
        unsafe(link_section = ".init_array")
    )]
    static SET_ONE_TEST_THREAD: extern "C" fn() = set_one_test_thread;
}

#[cfg(target_arch = "wasm32")]
mod glue;

use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

/// Diagnostic-only `compile_bridge` outcome tallies, read out via the
/// `pyre_jit_bridge_diag` guest export (the runner prints them at
/// `PYRE_WASM_JIT_STATS` time). A static counter — NOT a host import, which
/// would shift the wasm function-index space and break the JIT's baked
/// `fn as usize` table indices. Index legend: 0 = compile_bridge entered,
/// 1 = declined CALL_ASSEMBLER, 2 = declined multi-label peeled,
/// 3 = declined not-a-direct-loop-guard, 4 = declined ref-home overflow,
/// 5 = bridge compiled (chained in-module), 6 = loop-closing shape seen,
/// 7 = source loop has a preamble. Sub-breakdown of the index-2 multi-label
/// decline (TEMP, for the resume-at-last-label measurement): 8 = JUMP descr
/// did not resolve (target_ord None), 9 = target_ord Some but != last label,
/// 10 = arity mismatch, 11 = loop-closing bridge advances no loop-carried value
/// (guard side-trace that would livelock the chained loop), 14 = accepted
/// CALL_ASSEMBLER trace, 15 = unused (the trampoline-decline tally nothing
/// bumps; scratch left the frame, so a movable callee is not declined).
/// Index 16 records the dormant
/// forced-terminal-decline runtime regression hook. Sub-breakdown of the
/// index-8 unresolved-target decline: 17 = the terminal JUMP carries no descr
/// at all, 18 = the descr is present but `ll_loop_code` is 0.
/// Publish-side counterpart, so an unresolved lookup can be told from a label
/// that was never offered: 19 = labels published off a peeled trace, 20 =
/// published off a non-peeled trace, 21 = a non-peeled trace's first label left
/// unpublished (no descr, or its arity is not the inputarg count), 22 = a
/// dropped loop or a retired widened bridge retracted a published entry.
/// 19-21 count loops and
/// LABEL-bearing bridges alike, since both go through the same publish step
/// (`x86/assembler.py fixup_target_tokens` runs on either path); a bridge
/// with no LABEL is not tallied by 21. `compile_loop`'s own outcome
/// split — every `Err` it returns is a `loops_aborted` bump in the metainterp,
/// and the reason string never reaches the host from inside the guest, so the
/// classification has to be a counter: 23 = compile_loop entered, 24 =
/// compile_loop returned a token, 25 = declined by
/// `wasm_unsupported_trace_reason` (the #62 loop-callee CALL_ASSEMBLER gap),
/// 26 = the wasm host rejected the emitted module (`func_handle == 0`).
/// Indices 2 and 4 double as compile_loop's other two declines.
///
/// 27-28 ask 12-13's question of EVERY accepted bridge rather than only the
/// CALL_ASSEMBLER ones: 27 = the source guard's dispatch cell was written, so
/// the loop epilogue now tail-calls this bridge in-module; 28 = it was not,
/// because the owning trace reserved no cell array, so the guard keeps
/// round-tripping to the host and the bridge is compiled but unreachable.
/// `BRIDGE_OK` (5) only says the backend accepted a bridge, which is strictly
/// weaker than "the guard can reach it"; without this split the two are
/// indistinguishable from outside the guest.
///
/// 29 = the cell written at 27 was ALREADY non-zero, i.e. this guard had a
/// reachable bridge and kept failing anyway. One of those is ordinary (a guard
/// re-bridged after its first bridge was outgrown); a count that tracks
/// `BRIDGE_OK` says the epilogue dispatch is not taking the cell at all and
/// every bridge after the first is dead weight.
/// 30 = a re-emitted module was rejected by the host, or there is no host
/// replacement binding (`classify_inline_install_error`);
/// 31 = re-emission succeeded and the rebuilt module is installed in the loop's
/// original table slot. 31 is the only positive evidence that a re-emission
/// ran at all: a re-emission that silently never fires is indistinguishable
/// from one that fires and changes nothing.
/// 32 = a loop-closing bridge region was inlined; 33 = inlining declined
/// because the source guard belongs to an already chained trace; 34 = the
/// bridge is not loop-closing; 35 = the owner has no retained module inputs;
/// 36 = that guard already owns a region; 37 = the merged stream exceeds the
/// owner's frozen frame geometry; 38 = unused since the non-header gate was
/// removed; 39 = the merged stream has no local loop LABEL for the wasm back
/// edge. 40-43 split a rejected inline trial into value-layout,
/// Ref-home-layout, missing-local-label, and other backend errors. 44 = a
/// bridge compiled with a parameter entry; 45 = parameter entry declined
/// because the source module has frame-only dispatch; 46 = the bridge entry
/// cannot name the source guard's fail arguments — a parameter entry whose
/// arity disagrees with the guard's live count, or a frame entry whose
/// positional slots are not where that guard spilled them (the two arms are
/// mutually exclusive: `bridge_param_dispatch_for` selects one for the
/// module from its guard count); 47 =
/// LABEL publication suppressed because the bridge entry has nonzero parameters.
/// 48 = an inline trial's LABEL-resume storage exceeds the frozen frame; 49 =
/// the region carries a CALL_ASSEMBLER the owner build emits no arm for; 50 =
/// the owner is already invalidated, so a merged region would inherit its set
/// flag instead of starting valid; 51 = a region retained for a deferred merge
/// whose closing JUMP names a LABEL published by another module, so it keeps
/// the cross-module tail call and merges only its entry side; 52 = the
/// region's source guard is in the peeled preamble, outside the `loop` its
/// block is opened in; 53 = eligible but no trip callback is published to
/// defer to; 54 = eligible, merge deferred until the bridge standing in for it
/// has been entered `INLINE_TRIP_THRESHOLD` times; 55 = that trip fired and the
/// merge was attempted; 56 = reserved (the obsolete per-bridge invalidation
/// dependency refusal; dependencies now invalidate the whole token).
///
/// 57-63 split slot 1, which says only that some CALL_ASSEMBLER target did not
/// resolve and leaves the trace unsupported. Each answers one of the questions
/// `general_call_assembler_target` asks before it admits a target: 57 is the
/// historical result-kind decline slot; 58 = it carries no call descr, or no
/// target token; 59 = a descriptor/opcode result mismatch or invalid argument
/// kind; 60 = the target token is not in the registry at all;
/// 61 = a registered target's deferred module failed to materialize a func
/// handle; 62 = the registered geometry disagrees with the operation (input
/// types, zero callee frame bytes, absent gcmap, absent compiled loop); 63 =
/// the target compiled once and has since declined terminally. A decline that
/// falls in 60-63 names a target that exists, which is the half of slot 1 a
/// retrace could plausibly resolve; 57-59 name the operation itself.
///
/// 64-65 = reserved (cross-module / large-owner eager refusals). These regions
/// now defer until their entry count earns the owner re-emission.
///
/// 66 = `compile_loop` entered (`cl_entered`) but `build_wasm_module` returned
/// `Unsupported`. Slots 25/26 only name the pre-codegen CALL_ASSEMBLER filter
/// and a host module reject; every other `Result` return on that stretch was
/// silent, so a `cl_entered` without `cl_ok` could not say which opcode or
/// frame-layout check declined. The last reason string is also kept for the
/// host (`compile_loop_last_error`).
pub static BRIDGE_DIAG: [AtomicU64; BRIDGE_DIAG_LABELS.len()] =
    [const { AtomicU64::new(0) }; BRIDGE_DIAG_LABELS.len()];

/// Short key per [`BRIDGE_DIAG`] slot, in index order, spelling the legend
/// above as something a reader can join against.
///
/// This array is the slot count — [`BRIDGE_DIAG`] takes its length from it —
/// so a tally cannot be added without naming it. The wasm host mirrors these
/// keys positionally (it links no majit crate) and prints them under
/// `[jit-stats] bridge_diag`; without a declaration to compare against, a slot
/// bumped here and unnamed there is simply never reported.
///
/// A few slots are legend entries that no site bumps today (a decline that was
/// split finer, and the two `ml_*` sub-breakdowns). They keep their names so
/// the indices below them do not move.
pub const BRIDGE_DIAG_LABELS: &[&str] = &[
    "entered",
    "decl_callasm",
    "decl_multipeel",
    "decl_notdirect",
    "decl_refhome",
    "BRIDGE_OK",
    "loopclosing",
    "src_preamble",
    "ml_descr_none",
    "ml_unsafe_label",
    "ml_arity_mismatch",
    "decl_noadvance",
    "ca_cell_set",
    "ca_cells_zero",
    "accepted_ca",
    "unused_ca_trampoline",
    "forced_ca_terminal_decline",
    "ml_no_descr",
    "ml_unpublished",
    "pub_peeled",
    "pub_flat",
    "pub_flat_skipped",
    "label_retracted",
    "cl_entered",
    "cl_ok",
    "cl_decl_unsupported",
    "cl_decl_host_reject",
    "cell_set",
    "cell_missing",
    "cell_rebridge",
    "reemit_failed",
    "reemit_ok",
    "inline_ok",
    "inline_decl_not_direct",
    "inline_decl_not_loop_closing",
    "inline_decl_not_reemittable",
    "inline_decl_already_owned",
    "inline_decl_frame",
    "inline_decl_not_header",
    "inline_decl_no_loop_label",
    "inline_decl_value_layout",
    "inline_decl_ref_layout",
    "inline_decl_missing_label",
    "inline_decl_other",
    "bridge_param_ok",
    "bridge_param_decl_source_frame",
    "bridge_param_decl_arity",
    "bridge_param_label_suppressed",
    "inline_decl_label_resume_layout",
    "inline_decl_call_assembler",
    "inline_decl_owner_invalidated",
    "inline_foreign_jump",
    "inline_ok_outside_loop",
    "inline_decl_no_trip_helper",
    "inline_deferred",
    "inline_trip_fired",
    "inline_decl_defer_invalidation_guard",
    "ca_decl_opcode",
    "ca_decl_descr",
    "ca_decl_types",
    "ca_decl_unregistered",
    "ca_decl_materialize",
    "ca_decl_geometry",
    "ca_decl_terminal",
    "inline_decl_foreign_eager",
    "inline_decl_eager_too_large",
    "cl_decl_codegen",
];

#[repr(u8)]
#[derive(Clone, Copy)]
pub(crate) enum FrameShortageKind {
    FrameValueSlots = 1,
    OrdinaryRefHomes = 2,
    LabelResumeRefSlots = 3,
    LabelResumeCaptureSlots = 4,
}

#[derive(Clone, Copy)]
pub(crate) struct FrameShortage {
    pub(crate) kind: FrameShortageKind,
    pub(crate) needed: usize,
    pub(crate) available: usize,
}

impl FrameShortage {
    pub(crate) const fn new(kind: FrameShortageKind, needed: usize, available: usize) -> Self {
        Self {
            kind,
            needed,
            available,
        }
    }
}

/// The first three inline geometry failures, packed as
/// `(kind: u8, needed: u24, available: u24)`. They expose a frozen-layout
/// shortage without changing the compile result.
static INLINE_GEOMETRY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static INLINE_GEOMETRY_COUNT: AtomicU64 = AtomicU64::new(0);
/// The first three reasons an inline-bridge install was refused, verbatim.
/// The names carry "trial" because they are a guest export the runner looks up
/// by string; the errors themselves come from the install itself, which is the
/// only build there is.
static INLINE_TRIAL_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Why each loop-closing bridge was refused a merge into its owner, capped so
/// a long run cannot grow the log without bound. `bridge_diag`'s counters say
/// how many declines each reason took; these records carry the keys that say
/// which ones matter — `(slot, key)` joins a record against the trace-entry
/// census, whose `entries` count is how often that crossing actually ran.
static INLINE_DECLINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
const INLINE_DECLINE_LOG_CAP: usize = 64;
/// Last `compile_loop` decline reason. Guest `eprintln` never reaches the
/// host; this string is packed out through `pyre_jit_compile_loop_last_error`.
static COMPILE_LOOP_LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn record_compile_loop_error(error: &BackendError) {
    *COMPILE_LOOP_LAST_ERROR.lock() = error.to_string();
}

fn decline_compile_loop<T>(error: BackendError) -> Result<T, BackendError> {
    record_compile_loop_error(&error);
    Err(error)
}

/// Last `compile_loop` `Unsupported` reason, or empty if none declined.
pub fn compile_loop_last_error() -> String {
    COMPILE_LOOP_LAST_ERROR.lock().clone()
}

pub(crate) fn record_inline_geometry(kind: FrameShortageKind, needed: usize, available: usize) {
    const FIELD_MASK: u64 = (1 << 24) - 1;

    let index = INLINE_GEOMETRY_COUNT.fetch_add(1, Ordering::Relaxed) as usize;
    if let Some(slot) = INLINE_GEOMETRY.get(index) {
        slot.store(
            ((kind as u64) << 48)
                | ((needed as u64).min(FIELD_MASK) << 24)
                | (available as u64).min(FIELD_MASK),
            Ordering::Relaxed,
        );
    }
}

/// Read a packed `(kind, needed, available)` inline geometry failure.
pub fn inline_geometry_diag(index: usize) -> u64 {
    INLINE_GEOMETRY
        .get(index)
        .map_or(0, |slot| slot.load(Ordering::Relaxed))
}

/// Number of inline geometry failures, including records beyond the three
/// diagnostics retained in [`INLINE_GEOMETRY`].
pub fn inline_geometry_count() -> u64 {
    INLINE_GEOMETRY_COUNT.load(Ordering::Relaxed)
}

pub fn inline_trial_errors() -> String {
    INLINE_TRIAL_ERRORS.lock().join(" | ")
}

pub(crate) fn record_inline_decline(record: String) {
    let mut log = INLINE_DECLINES.lock();
    if log.len() < INLINE_DECLINE_LOG_CAP {
        log.push(record);
    }
}

pub fn inline_declines() -> String {
    INLINE_DECLINES.lock().join(" | ")
}

fn record_inline_trial_error(error: &BackendError) {
    let mut errors = INLINE_TRIAL_ERRORS.lock();
    if errors.len() < 3 {
        errors.push(error.to_string());
    }
}

/// Sort a refused inline install into the decline tallies the host prints.
/// `replace_module` rejecting the bytes, or a build with no host binding to
/// replace them through, is a re-emission outcome and stays on its own counter;
/// every other reason is the merged module declining to emit, which is what the
/// per-shortage buckets are for.
fn classify_inline_install_error(error: &BackendError) {
    let BackendError::Unsupported(reason) = error else {
        diag_bump(37);
        diag_bump(43);
        return;
    };
    if reason.contains("wasm host rejected the re-emitted trace module")
        || reason.contains("no host replacement binding")
    {
        diag_bump(30);
        return;
    }
    diag_bump(37);
    if reason.contains("frame value slots exceed frozen frame layout") {
        diag_bump(40);
    } else if reason.contains("ordinary ref homes") {
        diag_bump(41);
    } else if reason.contains("label resume layout") {
        diag_bump(48);
    } else if reason.contains("no CALL_ASSEMBLER arm for") {
        diag_bump(49);
    } else if reason.contains("inlined bridge stream has no local loop LABEL") {
        diag_bump(42);
    } else {
        diag_bump(43);
    }
}

/// Above this many exits, duplicating a parameter bridge arm at every guard is
/// larger and slower to compile than the shared frame-entry epilogue.
///
/// The parameter arm is the right shape for the small hot loops it was added
/// for: once a bridge exists it tail-calls it without spilling through the
/// frame.  Large application-level traces are different.  Each guard carries
/// a cell load, an indirect-call arm containing every live fail argument, and
/// the ordinary spill fallback; the two ~850-guard modules in
/// `pickle_terminal_raise_resume` grew by 78KB and made host Cranelift spend
/// about 100ms more compiling paths which acquired no bridge. Keep the fast
/// arm for ordinary traces and use the one shared frame dispatch once that
/// replication is no longer bounded.
const MAX_BRIDGE_PARAM_GUARDS: usize = 256;
static TRACE_ENTRY_CENSUS_FORCED: AtomicBool = AtomicBool::new(false);

/// One compiled trace's guest-memory entry counters.  The generated module
/// updates `counts[key]` directly, so this owner must outlive every module
/// that bakes its base address.
struct TraceEntryCensus {
    trace_id: u64,
    counts: Box<[u64]>,
}

/// The census deliberately has no per-entry Rust callback: a module writes
/// this guest-memory storage itself.  The runner reads it only after Python
/// exits, when no trace is executing.
static TRACE_ENTRY_CENSUS: Mutex<Vec<TraceEntryCensus>> = Mutex::new(Vec::new());

/// Baked into an armed module. `trace_id` is the backend's monotonic trace id,
/// which stays attached to a loop when its module is re-emitted.
#[derive(Clone, Copy)]
pub struct TraceEntryCensusStorage {
    pub trace_id: u64,
    pub base: u32,
    pub key_count: u32,
}

/// Arm trace-entry instrumentation before the guest starts compiling traces.
/// `MAJIT_TRACE_ENTRY_CENSUS` selects the same facility wherever the guest has
/// an environment to read it from; a guest that has none is armed by its host
/// through this function instead.
pub fn trace_entry_census_enable() {
    TRACE_ENTRY_CENSUS_FORCED.store(true, Ordering::Relaxed);
}

fn trace_entry_census_enabled() -> bool {
    if TRACE_ENTRY_CENSUS_FORCED.load(Ordering::Relaxed) {
        return true;
    }
    // Read on every target. Whether a wasm guest has an environment is a
    // property of its embedder, not of the architecture: one launched as a
    // WASI command inherits the variables its host passes it, and one with no
    // environment reads an absent variable rather than failing to compile.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAJIT_TRACE_ENTRY_CENSUS").is_some())
}

/// Allocate the one counter array that an armed physical trace module uses.
/// Re-emission clones the stored descriptor, preserving both the trace id and
/// the counters rather than assigning the replacement a second identity.
fn alloc_trace_entry_census(trace_id: u64, key_count: usize) -> Option<TraceEntryCensusStorage> {
    if !trace_entry_census_enabled() {
        return None;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut counts = vec![0u64; key_count].into_boxed_slice();
        let base = counts.as_mut_ptr() as usize as u32;
        TRACE_ENTRY_CENSUS
            .lock()
            .push(TraceEntryCensus { trace_id, counts });
        Some(TraceEntryCensusStorage {
            trace_id,
            base,
            key_count: key_count as u32,
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (trace_id, key_count);
        None
    }
}

/// Greppable, stable host readout of the guest-written entry counters.
pub fn trace_entry_census_summary() -> String {
    let census = TRACE_ENTRY_CENSUS.lock();
    let mut total = 0u64;
    let mut report = String::new();
    for trace in census.iter() {
        for (key, count) in trace.counts.iter().enumerate() {
            // Trace modules update this memory directly, outside Rust's alias
            // analysis; volatile makes the post-run host read explicit.
            let count = unsafe { core::ptr::read_volatile(count) };
            if count != 0 {
                total = total.saturating_add(count);
                report.push_str(&format!(
                    "[trace-entry-census] trace_id={} key={key} entries={count}\n",
                    trace.trace_id
                ));
            }
        }
    }
    report.push_str(&format!("[trace-entry-census] total={total}\n"));
    report
}

/// The wasm loop `token` was last compiled as, when it has one. Both merge
/// arms price themselves off its `module_bytes`.
fn compiled_wasm_loop(token: &JitCellToken) -> Option<&CompiledWasmLoop> {
    token
        .compiled
        .get()
        .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
}

/// Live deferred merges on `owner`'s `asmmemmgr_blocks`.
pub fn pending_inline_count(owner: &JitCellToken) -> usize {
    owner_pending_slot_addrs(owner)
        .into_iter()
        .filter(|&addr| {
            // SAFETY: `addr` was just read out of this owner's `pending_inlines`.
            unsafe { pending_slot_at(addr) }.entry.borrow().is_some()
        })
        .count()
}

/// Entries the bridge standing in for a merge must be entered before the merge
/// is taken, for an owner whose last emission was `owner_module_bytes` long.
///
/// A merge re-emits the whole owner, so its cost scales with the owner's size
/// rather than the region's: cranelift charges about 0.65 ms per KB of module,
/// while a cross-module crossing the merge removes is about 2.5 ns. Those two
/// rates are what [`DEFAULT_INLINE_TRIP_BYTES_FACTOR`] converts between; a
/// bridge's entry count is a floor on the crossings removed, so the price is
/// a floor too.
///
/// [`INLINE_TRIP_THRESHOLD`] stays as the lower bound, because a fixture whose
/// entire crossing budget is under a millisecond cannot pay back any rebuild.
///
/// Above the price at which a merge is refused outright sits a band where the
/// price only postpones a merge that is taken anyway, and every crossing in
/// that window is paid for nothing. The default sits below that band.
fn inline_trip_threshold_for(owner_module_bytes: u32) -> u64 {
    let priced = DEFAULT_INLINE_TRIP_BYTES_FACTOR.saturating_mul(owner_module_bytes as u64);
    priced.max(INLINE_TRIP_THRESHOLD)
}

fn bridge_param_dispatch_profitable(guard_count: usize) -> bool {
    guard_count <= MAX_BRIDGE_PARAM_GUARDS
}

fn bridge_param_dispatch_for(guard_count: usize) -> bool {
    bridge_param_dispatch_profitable(guard_count)
}

/// Read a `BRIDGE_DIAG` tally (saturating index). Surfaced to the host through
/// the `pyre_jit_bridge_diag` export in the `pyre-wasm` crate.
pub fn bridge_diag(i: usize) -> u64 {
    BRIDGE_DIAG
        .get(i)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0)
}

/// Last `compile_loop` `Err` reason. The guest has no stderr, so the
/// `BackendError` string otherwise dies inside `catch_unwind` / `?` and the
/// host only sees `cl_entered` without `cl_ok` / `cl_decl_*`.
static LAST_COMPILE_ERR: Mutex<String> = Mutex::new(String::new());

fn record_last_compile_err(err: &majit_backend::BackendError) {
    *LAST_COMPILE_ERR.lock() = err.to_string();
}

/// These declines are a property of the ops in this trace. A later
/// trace of the same guard can omit them, so the bridge decline stays
/// retryable. Other `Unsupported` shapes stay terminal.
fn bridge_codegen_error(err: majit_backend::BackendError) -> majit_backend::BackendError {
    match err {
        majit_backend::BackendError::RetryableUnsupported(reason) => {
            majit_backend::BackendError::CompilationFailed(reason)
        }
        majit_backend::BackendError::Unsupported(reason)
            if bridge_unsupported_is_retryable(&reason) =>
        {
            majit_backend::BackendError::CompilationFailed(reason)
        }
        other => other,
    }
}

fn bridge_unsupported_is_retryable(reason: &str) -> bool {
    (reason.contains("COND_CALL")
        && (reason.contains("no direct residual signature")
            || reason.contains("no web trampoline")))
        || reason.contains("residual call has")
        || reason.contains("CallMallocNurseryVarsizeHeaderless")
        || reason.contains("read with no producing op")
        || reason.contains("GuardFutureCondition")
        || reason.contains("VirtualRef")
        || reason.contains("RawLoad")
        || reason.contains("RawStore")
        || reason.contains("GuardAlwaysFails")
        || reason.contains("GetinteriorfieldGc")
        || reason.contains("SetinteriorfieldRaw")
        || reason.contains("Strlen")
        || reason.contains("Unicodelen")
        || reason.contains("Strgetitem")
        || reason.contains("Unicodegetitem")
        || reason.contains("Strsetitem")
        || reason.contains("Unicodesetitem")
        || reason.contains("Strhash")
        || reason.contains("Unicodehash")
        || reason.contains("Copystrcontent")
        || reason.contains("Copyunicodecontent")
}

// Snapshot of `last_compile_err` for the host's byte-at-index read.
// `last_compile_err_len` refreshes it so each byte load does not re-lock
// and re-clone the live string.
thread_local! {
    static LAST_COMPILE_ERR_SNAP: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Host-visible last `compile_loop` decline. Empty when every compile
/// returned a token.
pub fn last_compile_err() -> String {
    LAST_COMPILE_ERR.lock().clone()
}

/// Refresh the snapshot and return its length.
pub fn last_compile_err_len() -> u32 {
    let bytes = last_compile_err().into_bytes();
    let len = bytes.len() as u32;
    LAST_COMPILE_ERR_SNAP.with(|snap| *snap.borrow_mut() = bytes);
    len
}

/// Byte `i` of the snapshot [`last_compile_err_len`] last built. 0 if
/// `i` is out of range.
pub fn last_compile_err_byte(i: u32) -> u32 {
    LAST_COMPILE_ERR_SNAP.with(|snap| snap.borrow().get(i as usize).copied().unwrap_or(0) as u32)
}

/// Number of JIT trace entries made from the guest.
#[cfg(target_arch = "wasm32")]
pub fn jit_execute_count() -> u64 {
    glue::jit_execute_count()
}

/// Number of host modules materialized after the lazy-install gate.
#[cfg(target_arch = "wasm32")]
pub fn jit_compile_count() -> u64 {
    glue::jit_compile_count()
}

/// Number of materializations served by the byte-identical module cache.
#[cfg(target_arch = "wasm32")]
pub fn jit_compile_cache_hits() -> u64 {
    glue::jit_compile_cache_hits()
}

#[inline]
fn diag_bump(i: usize) {
    BRIDGE_DIAG[i].fetch_add(1, Ordering::Relaxed);
}

// A fresh loop frame is that trace's own spill count. A bridge or entry
// bridge that needs more keeps those offsets and appends a tail
// (`FrameGeometry::extend`); its prologue grows the live frame
// (`emit_check_frame_depth`, `wasm_realloc_frame`; upstream
// `assembler.py` `_check_frame_depth`, `llmodel.py` `realloc_frame`).
const FROZEN_CHAIN_LABEL_REF_SLOTS: usize = 2;
/// Slots a frozen layout is rounded up to.
///
/// Sibling loops through the same interpreter differ by a slot or two.
/// Rounding lands those siblings on one layout so a chained bridge and its
/// target share spill and home offsets. A layout that still differs is
/// entered at the target's offsets (`remap_frame_layout`) after
/// `_check_frame_depth` grows the frame. The cost is the rounded-away
/// slots, bounded by this constant per compiled token.
const FROZEN_CHAIN_SLOT_GRANULARITY: usize = 16;

/// `n` rounded up to a whole number of [`FROZEN_CHAIN_SLOT_GRANULARITY`] slots.
fn frozen_slot_count(n: usize) -> usize {
    n.div_ceil(FROZEN_CHAIN_SLOT_GRANULARITY) * FROZEN_CHAIN_SLOT_GRANULARITY
}

/// Words a label-parameter entry accepts after `frame_ptr`. One value for
/// every such entry in the process: a loop-closing JUMP reaches its target
/// with `return_call_indirect` on the shared table, and that type-checks the
/// callee against the *calling* module's type index, so two modules cannot
/// each pick their own width.
///
/// A peeled preamble's LABEL is the header live set plus the prefix's extra
/// values, so the JUMP that resumes there is that wide. 16 left a 17-arg JUMP
/// on the frame-slot path; 24 covers it and still rounds `ARITY + 1` to the
/// same 32-slot frozen granule (`frozen_slot_count`).
pub const FROZEN_LABEL_PARAM_ARITY: usize = 24;

/// An op whose result advances loop-carried state. A value produced inside the
/// re-running region by arithmetic or by a heap load is fresh on each pass, so
/// a JUMP carrying it advances the loop. Copies (`SameAs*`), casts,
/// comparisons, and allocations do not.
fn advances_loop_state(opcode: majit_ir::OpCode) -> bool {
    use majit_ir::OpCode::*;
    matches!(
        opcode,
        IntAdd
            | IntSub
            | IntMul
            | UintMulHigh
            | IntFloorDiv
            | IntMod
            | IntAnd
            | IntOr
            | IntXor
            | IntRshift
            | IntLshift
            | UintRshift
            | IntSignext
            | FloatAdd
            | FloatSub
            | FloatMul
            | FloatTrueDiv
            | FloatFloorDiv
            | FloatMod
            | FloatNeg
            | FloatAbs
            | IntNeg
            | IntInvert
            | IntAddOvf
            | IntSubOvf
            | IntMulOvf
            | GetfieldGcR
            | GetfieldGcI
            | GetfieldGcF
            | GetfieldRawI
            | GetfieldRawR
            | GetfieldRawF
            | GetarrayitemGcR
            | GetarrayitemGcI
            | GetarrayitemGcF
            | GetarrayitemRawI
            | GetarrayitemRawR
            | GetarrayitemRawF
            | GcLoadI
            | GcLoadR
            | GcLoadF
            | GcLoadIndexedI
            | GcLoadIndexedR
            | GcLoadIndexedF
            | RawLoadI
            | RawLoadF
    )
}

/// Per-guard (per-trace order), per-fail-arg: whether the value was produced
/// by loop-state-advancing arithmetic or a heap load in the part of the trace
/// that re-runs on every pass — the ops after the loop-header (last) LABEL, or
/// the WHOLE trace when it has no LABEL (a bridge, or a Label-less recursion
/// loop, whose body runs in full each pass). Such a fail arg is fresh in the
/// failing iteration, so a loop-closing bridge that JUMPs it verbatim still
/// advances the chained loop⇄bridge cycle (`compile_bridge`'s livelock check).
fn guard_fail_args_advanced(
    ops: &[majit_ir::Op],
    guard_exits: &[codegen::GuardExit],
) -> Vec<Vec<bool>> {
    let start = ops
        .iter()
        .rposition(|op| op.opcode == majit_ir::OpCode::Label)
        .map_or(0, |p| p + 1);
    let advanced_ids: std::collections::HashSet<u32> = ops[start..]
        .iter()
        .filter(|op| advances_loop_state(op.opcode))
        .map(|op| op.pos().get())
        .filter(|r| *r != majit_ir::OpRef::NONE && !r.is_constant())
        .map(|r| r.raw())
        .collect();
    guard_exits
        .iter()
        .map(|g| {
            let mask =
                crate::codegen::live_fail_arg_mask(g.meta_descr.as_ref(), g.fail_arg_refs.len());
            g.fail_arg_refs
                .iter()
                .zip(mask)
                .filter(|(_, live)| *live)
                .map(|(r, _)| !r.is_constant() && advanced_ids.contains(&r.raw()))
                .collect()
        })
        .collect()
}

use failguard::{
    CallAssemblerTarget, CompiledWasmLoop, LabelTarget, WasmFailDescr, WasmFrameData, descr_at,
    ensure_ca_cell, fill_exit_cell, label_target, mark_gnf2_token, publish_label_target,
    publish_token_target, target_from_token,
};
use majit_backend::{AsmInfo, BackendError, DeadFrame, JitCellToken};
use majit_gc::GcAllocator;
use majit_ir::{FailDescr, GcRef, InputArgRc, Op, OpRc, Value};

/// `x86/assembler.py` `patch_pending_failure_recoveries`: once the module is
/// accepted, each guard's `faildescr.adr_jump_offset` names the cell its exit
/// dispatches through. Called only after the last decline, because a declined
/// build frees the cell array the address points into.
fn patch_pending_failure_recoveries(guard_exits: &[codegen::GuardExit]) {
    for g in guard_exits {
        if g.bridge_cell == 0 {
            continue;
        }
        let Some(fd) = g.meta_descr.as_ref().and_then(|meta| meta.as_fail_descr()) else {
            continue;
        };
        if fd.is_finish() {
            continue;
        }
        let stamp = std::panic::AssertUnwindSafe(|| fd.set_adr_jump_offset(g.bridge_cell as usize));
        let _ = std::panic::catch_unwind(stamp);
    }
}

/// `x86/assembler.py fixup_target_tokens`, called from BOTH `assemble_loop`
/// (:612) and `assemble_bridge` (:706) — a LABEL assembled inside a bridge is a
/// jump target for later traces exactly like a loop's, and `compile_retrace`
/// (compile.py) reaches the backend through `send_bridge_to_backend`,
/// so a retrace IS a bridge that defines its own LABEL.
///
/// Returns `(label_descrs, published_descrs)`: the descr identity of every
/// LABEL in ordinal order, and the subset whose `LabelTarget` box was
/// stored in `resources` with its address written to
/// `LoopTargetDescr::ll_loop_code`. `compile_loop` keeps the first for
/// its own JUMP resolution; `compile_bridge` hands the second to the source
/// loop so `Drop` retracts them.
fn stamp_and_publish_label_targets(
    resources: &mut release::LoopAsmResources,
    func_handle: u32,
    frame: codegen::FrameGeometry,
    inputargs: &[InputArgRc],
    ops: &[Op],
    bridge_entry_arity: Option<usize>,
    owner_token: usize,
) -> (Vec<usize>, Vec<majit_ir::DescrRef>) {
    // Stamp each LABEL's loop-target descr with its ordinal (0, 1, 2, …) so a
    // loop-closing bridge can recover which label its terminal JUMP targets:
    // the JUMP and the LABEL share the descr by Arc identity, so the ordinal
    // written here is readable from the bridge's JUMP in `compile_bridge`.
    // Pure metadata — emits no wasm bytes, so the module shape is unchanged.
    // Skip a LABEL whose descr is not loop-target-backed (`set_label_block_id`
    // would panic on a non-`AtomicU32` slot).
    let mut label_block_id: u32 = 0;
    let mut label_descrs: Vec<usize> = Vec::new();
    let mut label_refs: Vec<Option<majit_ir::DescrRef>> = Vec::new();
    for op in ops.iter() {
        if op.opcode != majit_ir::OpCode::Label {
            continue;
        }
        label_refs.push(op.getdescr());
        // Descr identity of each label, in ordinal order, so
        // `compile_bridge` can resolve which of THIS loop's labels a
        // closing JUMP targets by Arc identity (the JUMP and the LABEL
        // share the descr). The stamped `label_block_id` alone cannot: a
        // loop retraced into several specializations re-stamps a shared
        // descr, and every specialization's start label carries ordinal
        // 0 — a bridge targeting ANOTHER specialization's label would
        // otherwise be mis-chained into this one.
        label_descrs.push(
            op.getdescr()
                .map(|d| std::sync::Arc::as_ptr(&d) as *const () as usize)
                .unwrap_or(0),
        );
        if let Some(descr) = op.getdescr()
            && let Some(target) = descr.as_loop_target_descr()
        {
            target.set_label_block_id(label_block_id);
        }
        label_block_id += 1;
    }
    // Per-label resume metadata (ordinal order) for `compile_bridge`'s
    // accept condition: a loop-closing bridge may resume at ANY label via
    // the entry `br_table`, provided its JUMP arity matches that label's
    // arg count and the label's args are the complete live set of the
    // trace remainder.
    let label_num_args = codegen::label_arg_counts(ops);
    let label_resume_info = codegen::label_resume_info(inputargs, ops, frame);
    // The wide entry occupies the slot the host appended right after this
    // module's narrow one, so it exists only once a handle does. Where
    // `func_handle` is 0 — a native build, which has no host at all —
    // `func_handle + 1` would name slot 1, another trace's entry rather than
    // an absent one, so both fields share the 0-means-absent encoding.
    let wide_slot = if func_handle != 0
        && codegen::has_label_param_entry(inputargs, ops, frame, bridge_entry_arity)
    {
        func_handle + 1
    } else {
        0
    };
    let mut published_descrs: Vec<majit_ir::DescrRef> = Vec::new();
    // A parameter entry with no fail values remains structurally `(i32) ->
    // i32`, so type-0 indirect calls may enter it. Only a nonzero parameter
    // entry is incompatible with published LABEL targets.
    let suppress_publication = matches!(bridge_entry_arity, Some(arity) if arity > 0);

    // Publish this loop's enterable labels so a loop-closing bridge from
    // ANY loop can chain into them in-module (jump-to-existing-trace). A
    // peeled loop's labels are each enterable through the entry br_table
    // (key = ordinal + 1). A non-peeled loop has no dispatch: only its
    // FIRST label is enterable — through the plain entry (key 0), whose
    // input loader reads `num_inputs` positional slots — and only when
    // the label's arity equals that (the standard loop shape, whose
    // first label's args ARE the inputargs).
    if codegen::is_resumable_peeled(ops) {
        // Only labels at or before the loop header have a resume loader
        // (`codegen::resumable_label_count`); the header is the last of
        // them, and a bridge landing there re-runs no advancing segment.
        let resumable = codegen::resumable_label_count(ops);
        let header = resumable.saturating_sub(1);
        for (j, &id) in label_descrs.iter().enumerate().take(resumable) {
            if id == 0 {
                continue;
            }
            let Some(descr) = label_refs[j].clone() else {
                continue;
            };
            if suppress_publication {
                diag_bump(47);
            } else {
                diag_bump(19);
                publish_label_target(
                    resources,
                    &descr,
                    LabelTarget {
                        func_handle,
                        wide_slot,
                        key: j as u32 + 1,
                        num_args: label_num_args[j],
                        resume_safe: label_resume_info[j].0,
                        requires_own_frame: label_resume_info[j].1,
                        is_last_label: j == header,
                        frame,
                        owner_token,
                    },
                );
                published_descrs.push(descr);
            }
        }
    } else {
        // A LABEL with real work before it is not reachable through the plain
        // entry: key 0 runs the function from its first op, so a bridge chaining
        // there would re-run that work. Before the descr-strict dispatch every
        // such trace was `is_resumable_peeled` and never reached this branch; a
        // `jump_to_preamble` retrace (own LABEL, foreign closing JUMP) is not, so
        // state the assumption the `is_last_label: true` publication relies on.
        let first_label_at_entry = ops
            .iter()
            .position(|op| op.opcode == majit_ir::OpCode::Label)
            == Some(0);
        let publishable = first_label_at_entry
            && label_descrs.first().is_some_and(|&id| id != 0)
            && label_num_args.first() == Some(&inputargs.len());
        // Counter 21 answers "this trace HAS a first label and it was left
        // unpublished". A trace with no LABEL at all — every ordinary bridge —
        // has nothing to publish and nothing withheld, so it is not a tally.
        if !publishable && !label_descrs.is_empty() {
            diag_bump(21);
        }
        if publishable && suppress_publication {
            diag_bump(47);
        } else if publishable {
            let descr = label_refs[0]
                .clone()
                .expect("publishable label has a descr");
            diag_bump(20);
            publish_label_target(
                resources,
                &descr,
                LabelTarget {
                    func_handle,
                    wide_slot,
                    key: 0,
                    num_args: inputargs.len(),
                    resume_safe: true,
                    requires_own_frame: false,
                    // No real ops precede a non-peeled loop's header, so
                    // an entry re-run lands at the header without any
                    // advancing segment — the livelock check applies.
                    is_last_label: true,
                    frame,
                    owner_token,
                },
            );
            published_descrs.push(descr);
        }
    }

    (label_descrs, published_descrs)
}

/// Process-wide pending-exception pair. `llmodel.py` `pos_exception` /
/// `pos_exc_value`: the interpreter's current exception, which a residual
/// raise publishes and the trace's `GuardNoException` / `GuardException`
/// load by absolute address. A `must_save_exception` exit copies
/// `pos_exc_value` into the frame's `jf_guard_exc` and clears both cells;
/// `grab_exc_value` reads the frame, not this pair.
static JIT_EXC_VALUE: AtomicI64 = AtomicI64::new(0);
static JIT_EXC_TYPE: AtomicI64 = AtomicI64::new(0);

/// Residual-call scratch shared by emitted wasm and the host trampoline.
/// Trampoline use is strictly LIFO: the host materialises every argument
/// before invoking the callee, and the guest loads the result immediately on
/// return, so a nested guest trampoline call cannot observe an outer call's
/// live data.
static JIT_CALL_AREA: [AtomicI64; codegen::FrameGeometry::CALL_AREA_SLOTS] =
    [const { AtomicI64::new(0) }; codegen::FrameGeometry::CALL_AREA_SLOTS];

/// llmodel.py _store_exception parity: set JIT exception state.
/// `value` is a valid OBJECTPTR (or 0); the exception class is read from
/// `value.typeptr` (offset 0).
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        // `typeptr` is a machine pointer (32-bit on wasm32); read it at
        // pointer width and zero-extend, so the high bits stay clear and
        // `GuardException`'s type comparison matches the baked class pointer.
        unsafe { *(value as *const usize) as i64 }
    };
    JIT_EXC_VALUE.store(value, Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, Ordering::Relaxed);
}

/// grab_exc_value parity: read the pending exception value and clear both
/// slots. Called host-side after a trace returns through a guard exit.
pub fn jit_exc_take() -> i64 {
    let value = JIT_EXC_VALUE.swap(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
    value
}

/// Non-destructive read of `JIT_EXC_VALUE` for the GC root walker (unlike
/// `jit_exc_take`, which swaps the cell to 0).
pub fn jit_exc_value_peek() -> i64 {
    JIT_EXC_VALUE.load(Ordering::Relaxed)
}

/// Root-walker write-back for `JIT_EXC_VALUE`: a minor collection moved the
/// pending exception from `old` to `new`. A compare-exchange, so a cell that
/// no longer holds `old` is left alone.
pub fn jit_exc_value_forward(old: i64, new: i64) {
    let _ = JIT_EXC_VALUE.compare_exchange(old, new, Ordering::Relaxed, Ordering::Relaxed);
}

/// Clear both exception slots without reading the value.
pub fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
}

/// Address of `JIT_EXC_VALUE`, embedded as an immediate in JIT-emitted wasm
/// so the trace can load/store it over the shared linear memory
/// (`_store_and_reset_exception` parity).
pub fn jit_exc_value_addr() -> usize {
    core::ptr::addr_of!(JIT_EXC_VALUE) as usize
}

/// Address of `JIT_EXC_TYPE`, embedded as an immediate in JIT-emitted wasm.
pub fn jit_exc_type_addr() -> usize {
    core::ptr::addr_of!(JIT_EXC_TYPE) as usize
}

/// Address of `JIT_CALL_AREA`, embedded as an immediate in JIT-emitted wasm.
pub fn jit_call_area_addr() -> usize {
    core::ptr::addr_of!(JIT_CALL_AREA) as usize
}

/// `llop.threadlocalref_get`: the word at byte offset `offset` of this
/// thread's `pypy_threadlocal_s`.
pub extern "C" fn wasm_jit_threadlocalref_get(offset: i64) -> i64 {
    majit_rlib::rthread::threadlocalref_get(offset as usize)
}

/// Write a thread-local slot that compiled traces may read back.
pub fn jit_threadlocalref_set(offset: i64, value: i64) {
    majit_rlib::rthread::threadlocalref_set(offset as usize, value);
}

/// callbuilder.py `write_real_errno`, run just before a CALL_RELEASE_GIL's
/// raw call. The result is a dummy that keeps the `(i64)->i64` helper type.
extern "C" fn wasm_jit_write_real_errno(save_err: i64) -> i64 {
    majit_rlib::rposix::_errno_before(save_err);
    0
}

/// callbuilder.py `read_real_errno`, run just after a CALL_RELEASE_GIL's
/// raw call.
extern "C" fn wasm_jit_read_real_errno(save_err: i64) -> i64 {
    majit_rlib::rposix::_errno_after(save_err);
    0
}

/// The per-thread GC box, and the accessors every trampoline reaches it through.
///
/// `gc.py:30` `GcLLDescription.__init__` holds `self.gcdescr` as a plain field
/// on the backend descriptor — there is no per-thread allocator upstream — so
/// this cell is scaffolding, not a ported structure. Only `install_gc_box`
/// fills it and only tests reach that; the production build goes through
/// `install_gc_standalone` and allocates from the `gc_sync` singleton.
///
/// Every accessor opens with `majit_gc::gc_box_installed()`, which without
/// `majit-gc/gc_box` is a constant `false` — so in a production build each one
/// folds to `None`, the thread-local becomes unreachable, and the trampolines
/// call `gc_sync` directly. The gate lives in `majit-gc` because a Cargo
/// feature is per-crate: this crate cannot `#[cfg]` on a feature of its
/// dependency, so the box is eliminated by the optimizer rather than by
/// conditional compilation. Mirrors `majit-backend-dynasm/src/runner.rs`'s
/// `gc_box`.
pub(crate) mod gc_box {
    use super::{GcAllocator, Ordering, RefCell};
    use std::cell::Cell;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    static NEXT_GC_BOX_GEN: AtomicU64 = AtomicU64::new(1);

    /// TLS payload whose destructor forgets the MiniMark.
    /// Thread teardown must not free the nursery: a dropped nursery's pages
    /// return to the OS and are reused while still referenced.
    struct LeakingNursery(Option<Box<dyn GcAllocator>>);

    impl Drop for LeakingNursery {
        fn drop(&mut self) {
            if let Some(gc) = self.0.take() {
                std::mem::forget(gc);
            }
        }
    }

    thread_local! {
        /// llmodel.py self.gc_ll_descr — owned by the active wasm backend on
        /// this thread. Stored as a thread-local so the backend-agnostic
        /// `majit_gc::ActiveGcGuardHooks` shims can reach the live allocator
        /// without taking a wasm dependency. RPython's `cpu.gc_ll_descr`
        /// parity, single-slot per thread.
        static WASM_ACTIVE_GC: RefCell<LeakingNursery> =
            const { RefCell::new(LeakingNursery(None)) };
        /// Read-only mirror of the box address: the interpreter-safepoint major
        /// holds the mutable borrow while extra-root walkers ask whether a slot
        /// is GC-managed, so that query routes through the raw pointer instead
        /// of taking a second borrow.
        static WASM_ACTIVE_GC_RAW: std::cell::Cell<Option<*mut dyn GcAllocator>> =
            const { std::cell::Cell::new(None) };
        /// Installation id of the live box. An [`ActiveGcBox`] only
        /// uninstalls when this still matches, so dropping an older
        /// backend cannot clear a newer one on the same thread.
        static WASM_ACTIVE_GC_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        /// Clone of the boxed MiniMark's `finalizer_queues`, stored next to
        /// (not inside the borrow of) `WASM_ACTIVE_GC` so
        /// `get_finalizer_queue_index` / `ll_next_dead` need no `RefCell`
        /// borrow. Follows this thread's boxed collector the same way
        /// `WASM_ACTIVE_GC` does.
        static WASM_ACTIVE_FINALIZER_QUEUES: RefCell<Option<Arc<majit_gc::FinalizerQueueTable>>> =
            const { RefCell::new(None) };
        static WASM_ACTIVE_FINALIZER_QUEUES_PTR: Cell<*const majit_gc::FinalizerQueueTable> =
            const { Cell::new(std::ptr::null()) };
    }

    /// `&mut` access to this thread's GC box, for allocation, write barriers
    /// and collection. `None` means there is no box, and the caller runs its
    /// `gc_sync` path instead.
    pub(super) fn with_mut<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        // `try_with`: `WasmFrameData::drop` and hook trampolines can run
        // while this thread's locals are already being destroyed. cranelift
        // `gc_box::clear` / `CA_DISPATCH_TABLE` use the same seam.
        WASM_ACTIVE_GC
            .try_with(|cell| {
                let mut guard = cell.borrow_mut();
                let raw: *mut dyn GcAllocator = guard.0.as_deref_mut()?;
                // SAFETY: `guard` holds the borrow for the whole `f` call and
                // these are non-reentrant top-level trampolines, so the reborrow
                // is exclusive and outlives `f`.
                Some(f(unsafe { &mut *raw }))
            })
            .ok()
            .flatten()
    }

    /// Read-only access that tolerates being reached from inside a collection:
    /// when an in-progress mutation already holds the mutable borrow, read the
    /// same allocator through the raw mirror rather than taking a second one.
    pub(super) fn with_reentrant_ref<R>(f: impl FnOnce(&dyn GcAllocator) -> R) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        match WASM_ACTIVE_GC.try_with(|cell| match cell.try_borrow() {
            Ok(guard) => guard.0.as_deref().map(f),
            // SAFETY: the mirror is published and cleared under the same
            // borrow as the box itself, so a non-null value points at the
            // live allocator, and this query only reads it.
            Err(_) => WASM_ACTIVE_GC_RAW
                .try_with(|raw| raw.get().map(|p| f(unsafe { &*p })))
                .ok()
                .flatten(),
        }) {
            Ok(r) => r,
            Err(_) => None,
        }
    }

    /// Identity hash for this thread's box.
    ///
    /// A free borrow allocates the shadow. `with_mut` already holds the
    /// `RefCell` across a collection, and that walk calls
    /// `Trace::refresh_from_gc` without entering `gc_op`, so the busy
    /// arm reads `id_or_identityhash_reentrant` through the raw mirror.
    /// `None` means this thread has no box.
    pub(super) fn id_or_identityhash(addr: usize) -> Option<usize> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        WASM_ACTIVE_GC
            .try_with(|cell| match cell.try_borrow_mut() {
                Ok(mut guard) => {
                    let raw: *mut dyn GcAllocator = guard.0.as_deref_mut()?;
                    // SAFETY: `guard` holds the borrow for this call.
                    Some(unsafe { &mut *raw }.id_or_identityhash(addr))
                }
                Err(_) => WASM_ACTIVE_GC_RAW
                    .try_with(|raw| {
                        raw.get().map(|p| {
                            // SAFETY: the mirror is published under the borrow
                            // the caller still holds, and this query only reads.
                            unsafe { &*p }.id_or_identityhash_reentrant(addr)
                        })
                    })
                    .ok()
                    .flatten(),
            })
            .ok()
            .flatten()
    }

    /// Whether this thread holds a box at all.
    pub(super) fn present() -> bool {
        majit_gc::gc_box_installed()
            && WASM_ACTIVE_GC
                .try_with(|cell| cell.borrow().0.is_some())
                .unwrap_or(false)
    }

    /// Store `gc` as this thread's box, publishing the raw mirror with it.
    /// Returns `(generation, installed)`: the installation id the matching
    /// [`ActiveGcBox`] must present to uninstall, and whether the TLS slot
    /// was empty (a new live box, not a replacement). A previous box is
    /// forgotten, not dropped: a dropped nursery's pages return to the OS
    /// and are reused while still referenced.
    pub(super) fn store(gc: Box<dyn majit_gc::GcAllocator>) -> (u64, bool) {
        let table = gc.finalizer_queue_table();
        let ptr = table
            .as_ref()
            .map(|a| Arc::as_ptr(a))
            .unwrap_or(std::ptr::null());
        let generation = NEXT_GC_BOX_GEN.fetch_add(1, Ordering::Relaxed);
        let installed = WASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let installed = guard.0.is_none();
            if let Some(old) = guard.0.take() {
                std::mem::forget(old);
            }
            guard.0 = Some(gc);
            let raw = guard.0.as_deref_mut().map(|gc| gc as *mut dyn GcAllocator);
            WASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(raw));
            WASM_ACTIVE_GC_GEN.with(|slot| slot.set(generation));
            installed
        });
        WASM_ACTIVE_FINALIZER_QUEUES.with(|cell| {
            *cell.borrow_mut() = table;
        });
        WASM_ACTIVE_FINALIZER_QUEUES_PTR.with(|cell| cell.set(ptr));
        (generation, installed)
    }

    /// Uninstall this thread's box without freeing its nursery.
    ///
    /// The raw mirror is cleared first so a reentrant query during
    /// uninstall does not observe a dangling pointer. The MiniMark
    /// itself is leaked: a dropped nursery's pages return to the OS and
    /// are reused while still referenced. ExtraHeap / InputArg slabs
    /// reuse those pages and smash their mutex words.
    /// Returns `true` when a box was actually removed.
    pub(crate) fn clear() -> bool {
        WASM_ACTIVE_GC_GEN.with(|slot| slot.set(0));
        WASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(None));
        let removed = WASM_ACTIVE_GC.with(|cell| {
            if let Some(gc) = cell.borrow_mut().0.take() {
                std::mem::forget(gc);
                true
            } else {
                false
            }
        });
        WASM_ACTIVE_FINALIZER_QUEUES_PTR.with(|cell| cell.set(std::ptr::null()));
        WASM_ACTIVE_FINALIZER_QUEUES.with(|cell| {
            *cell.borrow_mut() = None;
        });
        removed
    }

    /// Pointer to this thread's boxed collector table, or null when there is
    /// no box. Does not borrow `WASM_ACTIVE_GC`.
    pub(super) fn finalizer_queues_ptr() -> *const majit_gc::FinalizerQueueTable {
        WASM_ACTIVE_FINALIZER_QUEUES_PTR.with(|cell| cell.get())
    }

    /// [`clear`] only when `generation` is still the live installation.
    pub(crate) fn clear_if_generation(generation: u64) -> bool {
        let live = WASM_ACTIVE_GC_GEN.with(|slot| slot.get());
        if live == generation && generation != 0 {
            clear()
        } else {
            false
        }
    }
}

/// Read-only GC query for the guard hooks and codegen helpers. The box arm is
/// reentrancy-tolerant because these can fire during a collection's extra-root
/// walk, which is also why the singleton arm is the reentrant read.
fn with_wasm_active_gc<R>(f: impl Fn(&dyn GcAllocator) -> R) -> Option<R> {
    if let Some(r) = gc_box::with_reentrant_ref(&f) {
        return Some(r);
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_query_reentrant(|gc| f(gc)));
    }
    None
}

/// `&mut` counterpart of `with_wasm_active_gc` for GC mutations
/// (allocation, write barriers, collection). Test box → box; production
/// (no box, `gc_sync` initialized) → `gc_sync::gc_op`; no GC at all →
/// `None` so callers keep their non-GC fallback. Top-level mutator/
/// blackhole trampolines, never inside a collection, so `gc_op` is correct.
fn with_wasm_active_gc_mut<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> Option<R> {
    if gc_box::present() {
        return gc_box::with_mut(f);
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_op(|gc| f(gc)));
    }
    None
}

/// `transform.py _ll_malloc_fixedsize` OOM: `jit_exc_raise` of the
/// prebuilt MemoryError singleton, then the allocator returns NULL.
/// A zero singleton means the runtime cannot raise (`debug.py fatalerror`).
fn raise_memory_error() {
    let v = majit_backend::memory_error_singleton_ref();
    if v == 0 {
        eprintln!("fatal error: MemoryError raised with no runtime installed");
        std::process::abort();
    }
    jit_exc_raise(v);
}

/// Register all backend-agnostic `majit_gc::set_active_*` hooks to the
/// wasm trampolines. Shared by `install_gc_box` (test path: also stores a
/// box in TLS) and `install_gc_standalone` (production: hooks only, no box
/// — the trampolines then route to the `gc_sync` singleton).
fn register_active_hooks(supports_guard_gc_type: bool) {
    majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
        check_is_object: Some(wasm_check_is_object),
        is_tagged_immediate: Some(wasm_is_tagged_immediate),
        get_actual_typeid: Some(wasm_get_actual_typeid),
        subclass_range: Some(wasm_subclass_range),
        typeid_subclass_range: Some(wasm_typeid_subclass_range),
        typeid_is_object: Some(wasm_typeid_is_object),
        is_registered_type_id: Some(wasm_is_registered_type_id),
        can_move: Some(wasm_can_move),
        pin: Some(wasm_pin),
        unpin: Some(wasm_unpin),
        is_pinned: Some(wasm_is_pinned),
        supports_guard_gc_type,
    });
    majit_gc::set_active_alloc_nursery_typed(Some(wasm_alloc_nursery_typed));
    majit_gc::set_active_thread_has_gc_box(Some(gc_box::present));
    majit_gc::set_active_raise_memory_error(Some(raise_memory_error));
    majit_gc::set_active_alloc_nursery_headerless_no_collect(Some(
        wasm_alloc_nursery_headerless_no_collect,
    ));
    majit_gc::set_active_alloc_nursery_typed_with_placement(Some(
        wasm_alloc_nursery_typed_with_placement,
    ));
    majit_gc::set_active_alloc_nursery_collecting_typed(Some(wasm_alloc_nursery_collecting_typed));
    majit_gc::set_active_alloc_nursery_collecting_typed_rooted(Some(
        wasm_alloc_nursery_collecting_typed_rooted,
    ));
    majit_gc::set_active_alloc_nursery_collecting_typed_roots(Some(
        wasm_alloc_nursery_collecting_typed_roots,
    ));
    majit_gc::set_active_alloc_oldgen_typed(Some(wasm_alloc_oldgen_typed));
    majit_gc::set_active_alloc_young_nonmoving_typed(Some(wasm_alloc_young_nonmoving_typed));
    majit_gc::set_active_alloc_young_nonmoving_typed_no_collect(Some(
        wasm_alloc_young_nonmoving_typed_no_collect,
    ));
    majit_gc::set_active_root_hooks(Some(wasm_gc_add_root), Some(wasm_gc_remove_root));
    majit_gc::set_active_gc_owns_object(Some(wasm_gc_owns_object));
    majit_gc::set_active_gc_shrink_array(Some(wasm_gc_shrink_array));
    majit_gc::set_active_gc_varsize_layout(Some(wasm_gc_varsize_layout));
    majit_gc::set_active_gc_id_or_identityhash(Some(wasm_id_or_identityhash));
    majit_gc::set_active_write_barrier(Some(wasm_active_gc_write_barrier));
    majit_gc::set_active_write_barrier_before_move(Some(wasm_active_gc_write_barrier_before_move));
    majit_gc::set_active_write_barrier_from_array(Some(wasm_active_gc_write_barrier_from_array));
    majit_gc::set_active_writebarrier_before_copy(Some(wasm_active_gc_writebarrier_before_copy));
    majit_gc::set_active_write_barrier_managed(Some(wasm_active_gc_write_barrier_managed));
    majit_gc::set_active_get_objects(Some(wasm_get_objects));
    majit_gc::set_active_get_referents(Some(wasm_get_referents));
    majit_gc::set_active_subgraph_has_pending_finalizer(Some(wasm_subgraph_has_pending_finalizer));
    majit_gc::set_active_is_tracked(Some(wasm_is_tracked));
    majit_gc::set_active_gcflag_hooks(
        Some(wasm_get_gcflag_extra),
        Some(wasm_toggle_gcflag_extra),
        Some(wasm_get_gcflag_dummy),
    );
    majit_gc::set_active_get_rpy_memory_usage(Some(wasm_get_rpy_memory_usage));
    majit_gc::set_active_get_rpy_type_index(Some(wasm_get_rpy_type_index));
    majit_gc::set_active_get_rpy_roots(Some(wasm_get_rpy_roots));
    majit_gc::set_active_get_rpy_referents(Some(wasm_get_rpy_referents));
    majit_gc::set_active_is_app_level_object(Some(wasm_is_app_level_object));
    majit_gc::set_active_dump_rpy_heap(Some(wasm_dump_rpy_heap));
    majit_gc::set_active_get_typeids_text(Some(wasm_get_typeids_text));
    majit_gc::set_active_get_typeids_list(Some(wasm_get_typeids_list));
    majit_gc::set_active_add_memory_pressure(Some(wasm_add_memory_pressure));
    majit_gc::set_active_maybe_collect_for_external_malloc(Some(
        wasm_maybe_collect_for_external_malloc,
    ));
    majit_gc::set_active_total_memory_pressure(Some(wasm_total_memory_pressure));
    majit_gc::set_active_collect_generation(Some(wasm_collect_generation));
    majit_gc::set_active_collect_step(Some(wasm_collect_step));
    majit_gc::set_active_collect_oldgen(Some(wasm_collect_oldgen_nonmoving));
    majit_gc::set_active_heap_stats(Some(active_gc_heap_stats));
    majit_gc::set_active_gc_memory_stats(Some(active_gc_memory_stats));
    majit_gc::set_active_major_threshold_reached(Some(active_gc_major_threshold_reached));
    majit_gc::set_active_minor_collections_since_major(Some(
        active_gc_minor_collections_since_major,
    ));
    majit_gc::set_active_finalizer_hooks(Some(wasm_register_finalizer));
    majit_gc::set_active_finalizer_queue_table(Some(wasm_finalizer_queues));
}

fn wasm_finalizer_queues() -> *const majit_gc::FinalizerQueueTable {
    gc_box::finalizer_queues_ptr()
}

/// Live per-thread wasm GC boxes. Root hooks are process-global, so they
/// stay installed until the last box is dropped — clearing one thread must
/// not unhook another thread's still-active heap.
static WASM_GC_BOXES: AtomicUsize = AtomicUsize::new(0);

/// Owns the TLS GC box installed by [`install_gc_box`]. Dropping it
/// uninstalls the box on this thread (`llmodel.py` `cpu.gc_ll_descr`
/// dies with the cpu) only if this guard still owns the slot.
pub(crate) struct ActiveGcBox {
    generation: u64,
}

impl Drop for ActiveGcBox {
    fn drop(&mut self) {
        if gc_box::clear_if_generation(self.generation) {
            withdraw_root_hooks_if_last_box();
        }
    }
}

/// Store a GC allocator in the wasm backend thread-local and register
/// the `majit_gc::set_active_*` function-pointer hooks, without
/// requiring a `WasmBackend` instance.
/// Install a GC box into TLS and register all `set_active_*` hooks. Test
/// path only — `set_gc_allocator` hands ownership of a real allocator to
/// the backend thread. Production uses [`install_gc_standalone`], which
/// registers the same hooks WITHOUT a box so the trampolines fall through
/// to `gc_sync`.
///
/// Two live boxes at once is not a supported configuration: `ConstPtr`
/// `MAJOR_LIVE` and `TRACE_OPS_GC_TYPE_ID` are process-global indexes
/// into one collector.
fn install_gc_box(gc: Box<dyn majit_gc::GcAllocator>) -> ActiveGcBox {
    // Per-thread allocator: its nursery is not the singleton's, so the
    // process-wide published range can no longer answer `is_nursery_object`.
    majit_gc::disarm_published_nursery();
    majit_gc::note_gc_box_installed(gc.has_gcrootmap());
    let supports_guard_gc_type = gc.supports_guard_gc_type();
    let (generation, installed) = gc_box::store(gc);
    if installed {
        WASM_GC_BOXES.fetch_add(1, Ordering::Release);
    }
    register_active_hooks(supports_guard_gc_type);
    ActiveGcBox { generation }
}

/// Drop the active wasm GC box. Callers must go through this helper rather
/// than reaching the thread-local directly, otherwise the raw mirror used by
/// `wasm_gc_owns_object`'s reentrant fallback would be left pointing at
/// freed memory. Matches dynasm/cranelift `clear_gc_allocator`.
///
/// Root hooks are withdrawn too: a later `WasmFrameData` drop with a
/// leftover MiniMark would otherwise treat a test `GcRef` token as a heap
/// pointer. `install_gc_box` reinstalls the hooks.
pub fn clear_gc_allocator() {
    if !gc_box::clear() {
        return;
    }
    withdraw_root_hooks_if_last_box();
}

fn withdraw_root_hooks_if_last_box() {
    // Withdraw the process-global hooks only when this was the last box.
    // A leftover MiniMark on a later `WasmFrameData` drop must not see a
    // test `GcRef` token as a heap pointer, but another thread's box still
    // needs the hooks.
    if WASM_GC_BOXES
        .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
        .ok()
        == Some(1)
    {
        majit_gc::set_active_root_hooks(None, None);
    }
}

/// Production path: register all `set_active_*` hooks WITHOUT storing a
/// box. `WASM_ACTIVE_GC` stays `None`, so every trampoline routes to the
/// process-global `gc_sync` singleton (the per-thread GC box is the
/// free-threading gap R4 removes).
///
/// The type table stays open. `gctypelayout.py encode_type_shapes_now`
/// closes it at translation; pyre closes it before the first reader so
/// JIT-only types can still be registered after startup.
pub fn install_gc_standalone() {
    let supports_guard_gc_type = majit_gc::gc_sync::gc_query(|gc| gc.supports_guard_gc_type());
    register_active_hooks(supports_guard_gc_type);
}

/// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)` of the GC owned
/// by this thread's wasm backend, or `(0, 0)` if none is installed. Lets a host
/// runner split GC-retained memory from host-heap growth.
pub fn active_gc_heap_stats() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.heap_byte_stats()).unwrap_or((0, 0))
}

pub fn active_gc_memory_stats() -> majit_gc::GcMemoryStats {
    with_wasm_active_gc(|gc| gc.gc_memory_stats()).unwrap_or_default()
}

/// Whether the GC owned by this thread's wasm backend wants a major collection
/// (incminimark.py `threshold_reached`). Drives the interpreter GC
/// safepoint, which is on by default on wasm.
pub fn active_gc_major_threshold_reached() -> bool {
    with_wasm_active_gc(|gc| gc.major_threshold_reached()).unwrap_or(false)
}

/// Minor collections the active GC has run since its last major, or `0` when
/// none is installed.
pub fn active_gc_minor_collections_since_major() -> usize {
    with_wasm_active_gc(|gc| gc.minor_collections_since_major()).unwrap_or(0)
}

/// Diagnostic: `(minor_collections, major_collections)` of the active GC, or
/// `(0, 0)` when none is installed. Companion to [`active_gc_heap_stats`].
pub fn active_gc_collection_counts() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.collection_counts()).unwrap_or((0, 0))
}

/// Assemble the inline nursery-bump parameters for this trace's
/// `CallMallocNursery*` ops (rewrite.py malloc-fast-path eligibility over
/// the gc.py:525-531 nursery address surface), or `None` when no GC is
/// active, the `gc_stress` feature is compiled in (the fast path would
/// bypass its per-allocation stress collections), or no allocation op
/// qualifies.
fn nursery_alloc_params(ops: &[Op]) -> Option<codegen::NurseryAllocParams> {
    if majit_gc::gc_stress_enabled() {
        return None;
    }
    let tids: std::collections::HashSet<u32> = ops
        .iter()
        .filter_map(|op| match op.opcode {
            majit_ir::OpCode::New | majit_ir::OpCode::NewWithVtable => {
                Some(op.getdescr()?.as_size_descr()?.type_id())
            }
            majit_ir::OpCode::NewArray | majit_ir::OpCode::NewArrayClear => {
                Some(op.getdescr()?.as_array_descr()?.type_id())
            }
            _ => None,
        })
        .collect();
    let has_rewritten_malloc = ops.iter().any(|op| {
        matches!(
            op.opcode,
            majit_ir::OpCode::CallMallocNursery
                | majit_ir::OpCode::CallMallocNurseryHeaderless
                | majit_ir::OpCode::CallMallocNurseryVarsize
                | majit_ir::OpCode::CallMallocNurseryVarsizeHeaderless
                | majit_ir::OpCode::CallMallocNurseryVarsizeFrame
        )
    });
    if tids.is_empty() && !has_rewritten_malloc {
        return None;
    }
    with_wasm_active_gc(|gc| {
        let free_addr = gc.nursery_free_addr();
        let top_addr = gc.nursery_top_addr();
        if free_addr == 0 || top_addr == 0 {
            return None;
        }
        let plain_tids: std::collections::HashSet<u32> = tids
            .iter()
            .copied()
            .filter(|&t| gc.type_alloc_is_plain(t))
            .collect();
        if plain_tids.is_empty() && !has_rewritten_malloc {
            return None;
        }
        Some(codegen::NurseryAllocParams {
            free_addr: free_addr as u32,
            top_addr: top_addr as u32,
            large_threshold: gc.max_nursery_object_size(),
            plain_tids,
        })
    })?
}

/// Assemble the direct CA arm's fixed-size nursery/frame parameters. This is
/// deliberately separate from ordinary `CallMallocNursery*` eligibility: a CA frame needs
/// both the nursery words and the JitFrame shadow-stack top/limit cells.
/// Missing active GC (or gc_stress) leaves the pre-existing helper path intact.
fn ca_inline_params(frame_bytes: u32) -> Option<codegen::CaInlineParams> {
    if majit_gc::gc_stress_enabled() {
        return None;
    }
    let jitframe_tid = wasm_jitframe_tid();
    let depth = frame_bytes as usize / std::mem::size_of::<isize>();
    let total = ((majit_gc::header::GcHeader::SIZE
        + majit_backend::jitframe::JitFrame::alloc_size(depth))
    .max(majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE)
        + 7)
        & !7;
    with_wasm_active_gc(|gc| {
        assert_ne!(
            jitframe_tid, 0,
            "wasm CA inline frame path requires the registered JitFrame type id"
        );
        if total >= gc.max_nursery_object_size() || !gc.type_alloc_is_plain(jitframe_tid) {
            return None;
        }
        let nursery_free_addr = gc.nursery_free_addr();
        let nursery_top_addr = gc.nursery_top_addr();
        let jf_top_addr = majit_gc::shadow_stack::get_root_stack_top_addr();
        let jf_limit_addr = majit_gc::shadow_stack::get_root_stack_limit_addr();
        (nursery_free_addr != 0 && nursery_top_addr != 0 && jf_top_addr != 0 && jf_limit_addr != 0)
            .then_some(codegen::CaInlineParams {
                nursery_free_addr: nursery_free_addr as u32,
                nursery_top_addr: nursery_top_addr as u32,
                jf_top_addr: jf_top_addr as u32,
                jf_limit_addr: jf_limit_addr as u32,
                jitframe_tid,
                large_threshold: gc.max_nursery_object_size(),
            })
    })?
}

/// Whether the host entry runs a trace on a `JitFrame` it pushed onto the
/// jitframe shadow stack.
///
/// `execute_token` allocates that frame only once a `JitFrame` type id has been
/// registered; with none it runs the trace on a plain host buffer, which no
/// collection moves and which the shadow stack never describes. Every frame
/// reload a trace body emits answers out of that shadow stack, so an embedder
/// that registered no type id must get no reloads at all — a reload there would
/// replace the running frame pointer with whatever root happens to sit on top.
fn host_entry_frame_is_jitframe() -> bool {
    cfg!(target_arch = "wasm32") || wasm_jitframe_tid() != 0
}

/// Address of the active jitframe shadow-stack top cell for ordinary trace
/// body reloads. This does not depend on nursery fast-path eligibility: the
/// reload is valid whenever a GC is active at compilation time *and* the host
/// entry runs its traces on a pushed `JitFrame`.
fn jf_top_addr() -> Option<u32> {
    if !host_entry_frame_is_jitframe() {
        return None;
    }
    with_wasm_active_gc(|_| majit_gc::shadow_stack::get_root_stack_top_addr())
        .and_then(|addr| u32::try_from(addr).ok())
        .filter(|&addr| addr != 0)
}

/// Table slot of the frame-reload helper, or `0` when the running frame is not
/// one the shadow stack describes. Zero reaches codegen as "this trace needs no
/// reload", which is what a frame the host never pushed — and never moves —
/// requires.
fn body_reload_fn_ptr() -> i64 {
    if !host_entry_frame_is_jitframe() {
        return 0;
    }
    wasm_jit_ca_reload_frame as *const () as usize as i64
}

/// `majit_gc::CollectGenerationFn` installed by `register_active_hooks`. Drives
/// `gc.collect(n)` (`interp_gc.py`) through the active GC. Without it
/// `majit_gc::collect_generation` has no hook to dispatch to and silently
/// returns,
/// so no major cycle ever runs on this backend and
/// `deal_with_objects_with_finalizers` — which lives inside the major — never
/// executes: no `__del__`, no generator `finally`, not even under an explicit
/// `gc.collect()`. Mirrors dynasm's `dynasm_collect_generation` and cranelift's
/// `collect_generation_via_active_runtime`.
fn wasm_collect_generation(generation: i64) {
    with_wasm_active_gc_mut(|gc| gc.collect_generation(generation));
}

fn wasm_collect_step() -> majit_gc::GcStepTransition {
    with_wasm_active_gc_mut(|gc| gc.collect_step()).unwrap_or(majit_gc::GcStepTransition {
        // `rgc.py collect_step`: SCANNING on both sides would never report completion.
        old_state: majit_gc::GcStepTransition::STATE_MARKING,
        new_state: majit_gc::GcStepTransition::STATE_SCANNING,
    })
}

/// `majit_gc::CollectOldgenFn` installed by `set_gc_allocator`. Drives the
/// interpreter-safepoint non-moving old-gen major (`gc_interp::safepoint`,
/// default-on on wasm) through the active GC. Needs mutable access, so it
/// routes via `with_wasm_active_gc_mut` (test box → box; production → the
/// `gc_sync` singleton). Mirrors dynasm's `dynasm_collect_oldgen_nonmoving`
/// and cranelift's `collect_oldgen_nonmoving_via_active_runtime`.
fn wasm_collect_oldgen_nonmoving() {
    with_wasm_active_gc_mut(|gc| gc.collect_oldgen_nonmoving());
}

fn wasm_get_objects(generation: i8, visitor: majit_gc::GetObjectsVisitorFn) {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_objects(generation, &mut visit));
}

fn wasm_get_referents(obj: GcRef, visitor: majit_gc::GetObjectsVisitorFn) {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_referents(obj, &mut visit));
}

fn wasm_subgraph_has_pending_finalizer(roots: &[GcRef]) -> bool {
    with_wasm_active_gc_mut(|gc| gc.subgraph_has_pending_finalizer(roots)).unwrap_or(false)
}

fn wasm_is_tracked(obj: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.is_tracked(obj)).unwrap_or(false)
}

fn wasm_get_gcflag_extra(obj: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.get_gcflag_extra(obj)).unwrap_or(false)
}

fn wasm_toggle_gcflag_extra(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.toggle_gcflag_extra(obj));
}

fn wasm_get_gcflag_dummy(obj: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.get_gcflag_dummy(obj)).unwrap_or(false)
}

fn wasm_get_rpy_memory_usage(obj: GcRef) -> Option<usize> {
    with_wasm_active_gc_mut(|gc| gc.get_rpy_memory_usage(obj)).flatten()
}

fn wasm_get_rpy_type_index(obj: GcRef) -> Option<usize> {
    with_wasm_active_gc_mut(|gc| gc.get_rpy_type_index(obj)).flatten()
}

fn wasm_get_rpy_roots(visitor: majit_gc::GetObjectsVisitorFn) -> bool {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_rpy_roots(&mut visit)).unwrap_or(false)
}

fn wasm_get_rpy_referents(obj: GcRef, visitor: majit_gc::GetObjectsVisitorFn) -> bool {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_rpy_referents(obj, &mut visit)).unwrap_or(false)
}

fn wasm_is_app_level_object(obj: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.is_app_level_object(obj)).unwrap_or(false)
}

fn wasm_dump_rpy_heap(fd: i32) -> Result<bool, i32> {
    with_wasm_active_gc_mut(|gc| gc.dump_rpy_heap(fd)).unwrap_or(Ok(false))
}

fn wasm_get_typeids_text() -> Option<Vec<u8>> {
    with_wasm_active_gc(|gc| gc.get_typeids_text()).flatten()
}

fn wasm_get_typeids_list() -> Option<Vec<usize>> {
    with_wasm_active_gc(|gc| gc.get_typeids_list()).flatten()
}

fn wasm_add_memory_pressure(size: isize, object: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.add_memory_pressure(size, object));
}

fn wasm_maybe_collect_for_external_malloc(totalsize: usize) -> bool {
    with_wasm_active_gc_mut(|gc| gc.maybe_collect_for_external_malloc(totalsize)).unwrap_or(false)
}

fn wasm_total_memory_pressure() -> isize {
    with_wasm_active_gc_mut(|gc| gc.total_memory_pressure()).unwrap_or(0)
}

/// `minimark.py id_or_identityhash` trampoline. The collector
/// records a move-stable hash in its side table before the object can be
/// relocated; the unhooked `majit_gc::gc_id_or_identityhash` fallback returns
/// the raw address instead, which changes under the object when a minor
/// collection moves it out of the nursery. Mirrors dynasm's
/// `dynasm_id_or_identityhash`.
fn wasm_id_or_identityhash(addr: usize) -> usize {
    // `boehm.py` `ll_identityhash`: `h = ~cast_adr_to_int(addr)`.
    // `GcLLDescr_boehm.gcrootmap` is `None`, which is `!collector_installed`.
    if !majit_gc::collector_installed() {
        return !addr;
    }
    // `Trace::refresh_from_gc` runs from `MetaInterp::walk_active_trace_refs`
    // during a collection. The singleton arm is already inside `gc_op`.
    // A per-thread box holds `gc_box::with_mut` instead and never sets that
    // flag; `gc_box::id_or_identityhash` answers from the box either way.
    if majit_gc::gc_sync::in_gc_op() {
        return majit_gc::gc_sync::gc_query_reentrant(|g| g.id_or_identityhash_reentrant(addr));
    }
    if let Some(id) = gc_box::id_or_identityhash(addr) {
        return id;
    }
    if majit_gc::gc_sync::is_initialized() {
        return majit_gc::gc_sync::gc_op(|g| g.id_or_identityhash(addr));
    }
    addr
}

fn wasm_register_finalizer(fq_index: usize, obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.register_finalizer(fq_index, obj));
}

/// `majit_gc::CheckIsObjectFn` installed by `set_gc_allocator`.
/// Mirrors cranelift's `check_is_object_via_active_runtime`: dispatches
/// through the wasm-thread-local GC allocator.
fn wasm_check_is_object(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn wasm_is_tagged_immediate(addr: usize) -> bool {
    with_wasm_active_gc(|gc| gc.is_tagged_immediate(addr)).unwrap_or(false)
}

fn wasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_wasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn wasm_can_move(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.can_move(gcref)).unwrap_or(false)
}

fn wasm_pin(gcref: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.pin(gcref)).unwrap_or(false)
}

fn wasm_unpin(gcref: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.unpin(gcref)).expect("missing active GC runtime");
}

fn wasm_is_pinned(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.is_pinned(gcref)).unwrap_or(false)
}

fn wasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn wasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

fn wasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_wasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

fn wasm_is_registered_type_id(typeid: u32) -> bool {
    with_wasm_active_gc(|gc| (typeid as usize) < gc.type_count()).unwrap_or(false)
}

/// Host-side nursery allocation trampoline. Published via
/// `majit_gc::set_active_alloc_nursery_typed` so backend-agnostic
/// callers (pyre-object `w_int_new`, …) can route through the
/// wasm-owned GC.
fn wasm_alloc_nursery_typed(type_id: u32, size: usize) -> GcRef {
    // See cranelift/dynasm counterparts: host-side allocation must not
    // trigger collection because the caller holds a raw pointer that
    // is not a registered GC root.
    with_wasm_active_gc_mut(|gc| gc.try_alloc_nursery_no_collect_typed(type_id, size))
        .unwrap_or(GcRef(0))
}

/// `majit_gc::AllocNurseryHeaderlessNoCollectFn`. The metainterp's jitcode
/// tracer allocates a `NEW` on a `headerless` descr through here so the object
/// lands in the interpreter's own collected pool rather than the host heap,
/// where its collector could not see it. Returns `GcRef(0)` when no GC is
/// bound, leaving the caller on its own path.
fn wasm_alloc_nursery_headerless_no_collect(size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_headerless_no_collect(size)).unwrap_or(GcRef(0))
}

/// Placement-reporting companion of [`wasm_alloc_nursery_typed`].
///
/// # Safety
/// `needs_write_barrier` must remain a valid mutable `bool` slot until this
/// call returns.
unsafe fn wasm_alloc_nursery_typed_with_placement(
    type_id: u32,
    size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    with_wasm_active_gc_mut(|gc| unsafe {
        gc.try_alloc_nursery_no_collect_typed_with_placement(type_id, size, needs_write_barrier)
    })
    .unwrap_or(GcRef(0))
}

/// Host-side collecting nursery allocation used by elidable bigint payload
/// helpers. This is the wasm twin of the dynasm/cranelift hooks: the active
/// backend must replace every process-global allocation hook as one unit so a
/// previously-installed native backend cannot receive wasm allocations.
fn wasm_alloc_nursery_collecting_typed(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// Rooted collecting companion for a result whose GC child exists only in a
/// native Rust slot while the parent allocation may collect.
///
/// # Safety
/// `root` and `needs_write_barrier` must remain valid mutable slots until this
/// call returns.
unsafe fn wasm_alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    with_wasm_active_gc_mut(|gc| unsafe {
        gc.alloc_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
    })
    .unwrap_or(GcRef(0))
}

unsafe fn wasm_alloc_nursery_collecting_typed_roots(
    type_id: u32,
    size: usize,
    roots: *mut GcRef,
    root_count: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    with_wasm_active_gc_mut(|gc| unsafe {
        gc.alloc_fast_nursery_collecting_typed_roots(
            type_id,
            size,
            roots,
            root_count,
            needs_write_barrier,
        )
    })
    .unwrap_or(GcRef(0))
}

/// Host-side old-gen allocation trampoline. Stable
/// across minor/major collections — see dynasm counterpart.
fn wasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_oldgen_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// `external_malloc(..., alloc_young=True)` on the wasm-owned GC: a stable
/// address that the next minor frees unless something reaches it. May itself
/// run that minor (`threshold_reached` then
/// `minor_collection_with_major_progress`).
fn wasm_alloc_young_nonmoving_typed(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_young_nonmoving_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// [`wasm_alloc_young_nonmoving_typed`] without the collection in front of
/// the birth (`GcAllocator::alloc_young_nonmoving_typed_no_collect`).
fn wasm_alloc_young_nonmoving_typed_no_collect(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_young_nonmoving_typed_no_collect(type_id, size))
        .unwrap_or(GcRef(0))
}

/// Allocate the block a blackhole `bh_new*` descr describes, in the non-moving
/// old generation as the dynasm and cranelift runners do: resume
/// materialization keeps raw pointers to these across the forward blackhole
/// run, and a nursery block would be relocated out from under them at the next
/// minor collection.
fn wasm_bh_alloc(type_id: u32, payload_size: usize) -> i64 {
    let gc_ptr = if type_id != 0 {
        wasm_alloc_oldgen_typed(type_id, payload_size).0
    } else {
        0
    };
    if gc_ptr != 0 {
        // A blackhole-materialized object is born into the old generation, so the
        // collector reaches it only through the remembered set. It is not on the
        // frame chain `walk_pyframe_roots` walks, and the resume fills it through
        // stores that are barrier-free precisely because that chain is a root set
        // (`ExecutionContext::enter`), so nothing would put it there. A nursery
        // object needed none of this — being young was enough to have its fields
        // traced. Remember it at birth to restore that, before any field is
        // written; `TRACK_YOUNG_PTRS` is set by `finish_alloc_in_oldgen`, so this
        // is the ordinary barrier, not a new mechanism.
        //
        // The raw fallback below is deliberately outside it: that block is plain
        // malloc, not a collector-owned object, so it has no header to remember.
        majit_gc::gc_write_barrier(GcRef(gc_ptr));
        return gc_ptr as i64;
    }
    if type_id != 0 {
        // `GcLLDescr_framework._bh_malloc` returns NULL on failure so the
        // blackhole wrapper can raise `MemoryError`.  A raw fallback for a
        // typed descr drops the GC header and its tracing layout.
        return 0;
    }
    wasm_bh_alloc_raw(payload_size)
}

/// Non-GC descrs (`type_id == 0`, raw buffers) keep the plain zeroed malloc the
/// dynasm runner uses for the same descr shape.
fn wasm_bh_alloc_raw(size: usize) -> i64 {
    let Ok(layout) = std::alloc::Layout::from_size_align(size.max(1), 8) else {
        return 0;
    };
    unsafe { std::alloc::alloc_zeroed(layout) as i64 }
}

/// Allocate the struct a blackhole `bh_new` / `bh_new_with_vtable` describes,
/// mirroring the dynasm runner's `bh_alloc_struct`.
///
/// A headerless descr names a struct from the interpreter's own
/// `headerless_structs` pool, which carries no type word at `ref - 8`; a
/// header-writing allocator returns `base + GcHeader::SIZE` and puts a block
/// the interpreter owns onto the collector's lists.
///
/// `resolve_gc_tid` routes the serialized `path_hash` cache key back to the
/// dense GC tid (`gc.py:536-542`); a raw cache key read as a tid indexes past
/// the type table on the first collection that traces the block.
fn wasm_bh_alloc_struct(sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
    let size = sizedescr.as_size();
    if sizedescr.is_headerless() {
        let gc_ptr = wasm_alloc_nursery_headerless_no_collect(size).0;
        if gc_ptr != 0 {
            return gc_ptr as i64;
        }
        return wasm_bh_alloc_raw(size);
    }
    wasm_bh_alloc(sizedescr.resolve_gc_tid(), size)
}

/// `gc.py` malloc-helper OOM signalling, the twin of the dynasm runner's
/// `oom_signal_if_zero`: translated `do_malloc_fixedsize_clear` raises
/// `MemoryError`, which lowers to "store the singleton in `pos_exc_value`,
/// return NULL". These trampolines return 0 directly, so the store belongs
/// here — the emitted memory-error check (`codegen.rs`
/// `emit_memory_error_check`) moves the value out of the cell and into
/// `jf_guard_exc`. `PropagateExceptionDescr.handle_fail` reads it back.
#[inline]
fn oom_signal_if_zero(result: i64) -> i64 {
    if result == 0 {
        let value = majit_backend::memory_error_singleton_ref();
        if value != 0 {
            jit_exc_raise(value);
        }
    }
    result
}

/// JIT-trace allocation trampoline target for `New` / `NewWithVtable`.
///
/// A compiled trace cannot allocate directly (the GC lives behind the
/// `WASM_ACTIVE_GC` thread-local), so the `New` codegen routes through the
/// host `jit_call` trampoline, which resolves this function via the module's
/// `__indirect_function_table` (its address is taken in `compile_loop`, so it
/// lands in the table) and invokes it with `(type_id, size)`. Returns the new
/// object pointer, or 0 when no GC is installed. The `ob_type` field for
/// `NewWithVtable` is written inline by codegen at `vtable_offset`.
///
/// Unlike the general [`wasm_alloc_nursery_typed`] host hook (which must not
/// collect — its callers hold unrooted raw pointers), this JIT-trace path is
/// safe to collect: the trace registers every live Ref's frame home slot as a
/// GC root and reloads its locals from the (forwarded) homes after each
/// allocation. So it uses the *collecting* `alloc_nursery_typed`, which
/// triggers a minor collection on nursery-full instead of leaking to old-gen.
pub extern "C" fn wasm_jit_alloc(type_id: i64, size: i64) -> i64 {
    let obj = with_wasm_active_gc_mut(|gc| {
        gc.alloc_nursery_typed(type_id as u32, size as usize).0 as i64
    })
    .unwrap_or(0);
    // IncrementalMiniMark `malloc_zero_filled = False`. rewrite.py
    // `clear_gc_fields` NULLs leftover GC-pointer fields; the helper does
    // not fill the payload.
    oom_signal_if_zero(obj)
}

/// Headerless nursery overflow helper. Returns the raw allocation base with
/// no GC header, matching cranelift's `gc_alloc_nursery_headerless_shim`.
pub extern "C" fn wasm_jit_alloc_headerless(size: i64) -> i64 {
    let Ok(size) = usize::try_from(size) else {
        return oom_signal_if_zero(0);
    };
    if size == 0 {
        return oom_signal_if_zero(0);
    }
    let size = size.saturating_add(7) & !7;
    let obj = with_wasm_active_gc_mut(|gc| {
        if let Some(base) = try_headerless_nursery_bump(gc, size) {
            return base as i64;
        }
        gc.collect_nursery();
        if let Some(base) = try_headerless_nursery_bump(gc, size) {
            return base as i64;
        }
        0
    })
    .unwrap_or(0);
    // IncrementalMiniMark `malloc_zero_filled = False`. rewrite.py
    // does not fill a headerless bump; leftover GC-pointer fields are
    // the delayed-zero stores, not this helper.
    oom_signal_if_zero(obj)
}

fn try_headerless_nursery_bump(gc: &mut dyn majit_gc::GcAllocator, size: usize) -> Option<usize> {
    let nf_addr = gc.nursery_free_addr();
    let nt_addr = gc.nursery_top_addr();
    if nf_addr == 0 || nt_addr == 0 {
        return None;
    }
    unsafe {
        let nf = nf_addr as *mut usize;
        let nt = nt_addr as *const usize;
        let free = nf.read();
        let top = nt.read();
        let new_free = free.checked_add(size)?;
        if new_free > top {
            return None;
        }
        nf.write(new_free);
        Some(free)
    }
}

/// JIT-trace variable-size allocation trampoline target for `NewArray` /
/// `NewArrayClear`. Allocates `length` items and writes the length field at
/// `len_offset`, mirroring [`WasmBackend::bh_new_array`].
pub extern "C" fn wasm_jit_alloc_array(
    type_id: i64,
    base_size: i64,
    item_size: i64,
    length: i64,
    len_offset: i64,
) -> i64 {
    // `incminimark.py external_malloc` refuses a negative length by raising
    // `MemoryError`, so this edge signals like an exhausted heap.
    let Ok(length) = usize::try_from(length) else {
        return oom_signal_if_zero(0);
    };
    let obj = with_wasm_active_gc_mut(|gc| {
        let obj = gc.alloc_varsize_typed(
            type_id as u32,
            base_size as usize,
            item_size as usize,
            length,
        );
        if obj.is_null() { 0 } else { obj.0 as i64 }
    })
    .unwrap_or(0);
    if obj != 0 {
        // IncrementalMiniMark does not zero-fill. rewrite.py emits
        // ZERO_ARRAY only for NEW_ARRAY_CLEAR; wasm codegen does the
        // same after this helper returns.
        unsafe {
            *((obj as *mut u8).add(len_offset as usize) as *mut usize) = length;
        }
    }
    oom_signal_if_zero(obj)
}

/// Old-generation twin of [`wasm_jit_alloc`], selected by the `New` /
/// `NewWithVtable` codegen for a `non_moving` size descr. Same signature, so it
/// shares the call shape; only the generation differs.
///
/// A descr marked `non_moving` is one whose object is reached through a raw
/// pointer nothing forwards — the interpreter holds it across an allocation, or
/// another object's field stores it outside the collector's view. Placing such
/// an object in the movable nursery leaves those pointers aimed at the pre-move
/// copy. The native backends honour the flag in the GC rewrite pass
/// (`rewrite.rs` `handle_new`); wasm lowers `New` itself, so it must apply the
/// same policy here.
pub extern "C" fn wasm_jit_alloc_oldgen(type_id: i64, size: i64) -> i64 {
    let obj =
        with_wasm_active_gc_mut(|gc| gc.alloc_oldgen_typed(type_id as u32, size as usize).0 as i64)
            .unwrap_or(0);
    oom_signal_if_zero(obj)
}

/// Old-generation twin of [`wasm_jit_alloc_array`], selected by the `NewArray` /
/// `NewArrayClear` codegen for a `non_moving` array descr. Same signature and
/// the same length stamp; see [`wasm_jit_alloc_oldgen`] for why the generation
/// is part of the descr's contract.
pub extern "C" fn wasm_jit_alloc_array_oldgen(
    type_id: i64,
    base_size: i64,
    item_size: i64,
    length: i64,
    len_offset: i64,
) -> i64 {
    let Ok(length) = usize::try_from(length) else {
        return oom_signal_if_zero(0);
    };
    // Overflowing the payload size is MemoryError, same as a negative length.
    let Some(payload_size) = (item_size as usize)
        .checked_mul(length)
        .and_then(|var_size| (base_size as usize).checked_add(var_size))
    else {
        return oom_signal_if_zero(0);
    };
    let obj = with_wasm_active_gc_mut(|gc| {
        let obj = gc.alloc_oldgen_typed(type_id as u32, payload_size);
        if obj.is_null() {
            0
        } else {
            unsafe {
                *((obj.0 as *mut u8).add(len_offset as usize) as *mut usize) = length;
            }
            obj.0 as i64
        }
    })
    .unwrap_or(0);
    oom_signal_if_zero(obj)
}

/// rewrite.py `gen_malloc_array` standard arm:
/// `CALL_R(malloc_array_fn, itemsize, typeid, length)`.
/// [`wasm_jit_alloc_array`] is the NewArray trampoline ABI
/// `(type_id, base_size, item_size, length, len_offset)`.
pub extern "C" fn wasm_malloc_array(item_size: i64, type_id: i64, num_elem: i64) -> i64 {
    wasm_jit_alloc_array(
        type_id,
        std::mem::size_of::<usize>() as i64,
        item_size,
        num_elem,
        0,
    )
}

/// rewrite.py `gen_malloc_array` nonstandard arm:
/// `CALL_R(fn, basesize, itemsize, lengthofs, typeid, length)`.
pub extern "C" fn wasm_malloc_array_nonstandard(
    base_size: i64,
    item_size: i64,
    length_ofs: i64,
    type_id: i64,
    num_elem: i64,
) -> i64 {
    wasm_jit_alloc_array(type_id, base_size, item_size, num_elem, length_ofs)
}

/// Old-generation twin of [`wasm_malloc_array`], selected by
/// `gen_malloc_array` for a `non_moving` array descr.
pub extern "C" fn wasm_malloc_array_oldgen(item_size: i64, type_id: i64, num_elem: i64) -> i64 {
    wasm_jit_alloc_array_oldgen(
        type_id,
        std::mem::size_of::<usize>() as i64,
        item_size,
        num_elem,
        0,
    )
}

/// Old-generation twin of [`wasm_malloc_array_nonstandard`].
pub extern "C" fn wasm_malloc_array_nonstandard_oldgen(
    base_size: i64,
    item_size: i64,
    length_ofs: i64,
    type_id: i64,
    num_elem: i64,
) -> i64 {
    wasm_jit_alloc_array_oldgen(type_id, base_size, item_size, num_elem, length_ofs)
}

/// rewrite.py `gen_malloc_str` (rewrite.py): `CALL_R(malloc_str_fn, length)`.
/// This helper also takes `type_id` because `rewrite.rs` `gen_malloc_str`
/// emits `CALL_R(malloc_str_fn, type_id, length)`.
/// Layout matches `codegen::BUILTIN_STR_TOKEN_BASE_SIZE` /
/// `codegen::BUILTIN_STRING_LEN_OFFSET`.
pub extern "C" fn wasm_malloc_str(type_id: i64, length: i64) -> i64 {
    wasm_jit_alloc_array(
        type_id,
        codegen::BUILTIN_STR_TOKEN_BASE_SIZE as i64,
        1,
        length,
        codegen::BUILTIN_STRING_LEN_OFFSET as i64,
    )
}

/// rewrite.py `gen_malloc_unicode` (rewrite.py):
/// `CALL_R(malloc_unicode_fn, length)`.
/// This helper also takes `type_id` because `rewrite.rs` `gen_malloc_unicode`
/// emits `CALL_R(malloc_unicode_fn, type_id, length)`.
/// Layout matches `codegen::BUILTIN_UNICODE_TOKEN_BASE_SIZE` /
/// `codegen::BUILTIN_STRING_LEN_OFFSET`.
pub extern "C" fn wasm_malloc_unicode(type_id: i64, length: i64) -> i64 {
    wasm_jit_alloc_array(
        type_id,
        codegen::BUILTIN_UNICODE_TOKEN_BASE_SIZE as i64,
        4,
        length,
        codegen::BUILTIN_STRING_LEN_OFFSET as i64,
    )
}

fn ca_locs_from_token(token: &JitCellToken) -> Option<majit_gc::rewrite::CallAssemblerCalleeLocs> {
    let clt = token.compiled_loop_token()?;
    let frame_info_ptr = {
        let info = clt.frame_info.lock();
        &*info as *const majit_backend::JitFrameInfo as usize
    };
    let frame_depth = clt.frame_info.lock().depth() as usize;
    let ll_initial_locs = clt._ll_initial_locs.lock().clone();
    Some(majit_gc::rewrite::CallAssemblerCalleeLocs {
        _ll_initial_locs: ll_initial_locs,
        frame_depth,
        frame_info_ptr,
        index_of_virtualizable: token.virtualizable_arg_index().map_or(-1, |i| i as i32),
    })
}

/// `call_jit.rs` `jitframe_layout_descrs`. Offsets are from the object base
/// past the GC header, which is what `CallMallocNurseryVarsizeFrame` returns.
fn wasm_jitframe_descrs() -> majit_gc::rewrite::JitFrameDescrs {
    use majit_backend::jitframe::*;
    majit_gc::rewrite::JitFrameDescrs {
        jitframe_tid: wasm_jitframe_tid(),
        jitframe_fixed_size: JITFRAME_FIXED_SIZE,
        jf_frame_info_ofs: JF_FRAME_INFO_OFS,
        jf_descr_ofs: JF_DESCR_OFS,
        jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
        jf_savedata_ofs: JF_SAVEDATA_OFS,
        jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
        jf_forward_ofs: JF_FORWARD_OFS,
        jf_frame_ofs: JF_FRAME_OFS,
        jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
        jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
        sign_size: SIGN_SIZE,
        // The entry reads each input as an i64 from its slot.
        jf_frame_itemsize: codegen::SLOT_SIZE as usize,
    }
}

/// dynasm `register_call_assembler_target` / `BaseRegalloc._set_initial_bindings`.
///
/// The wasm entry reads input `k` from `spill_slot_ofs(k)` off the items
/// base (`FIRST_ITEM_OFFSET`). `handle_call_assembler` adds
/// `jf_frame_baseitemofs`, so each loc is that same byte offset.
fn publish_ca_initial_locs(
    token: &majit_backend::JitCellToken,
    frame: codegen::FrameGeometry,
    n_inputs: usize,
) {
    let Some(clt) = token.compiled_loop_token() else {
        return;
    };
    let locs: Vec<i32> = (0..n_inputs)
        .map(|i| frame.spill_slot_ofs(i as u64) as i32)
        .collect();
    *clt._ll_initial_locs.lock() = locs;
    ensure_ca_cell(token);
}

/// Force-tail words carried on `WasmFailDescr` so `force_arg_word` reconstructs
/// `FrameGeometry::force_slot_ofs` from the physical value index in `fail_locs`.
fn force_tail_fields(frame: codegen::FrameGeometry) -> (u32, u32, u32) {
    if !frame.has_tail() {
        return (0, 0, 0);
    }
    (
        frame.force_tail_base() as u32,
        frame.prefix_value_slots as u32,
        frame.value_tail_index() as u32,
    )
}

fn call_assembler_token_from_descr(
    descr: &majit_ir::DescrRef,
) -> Option<&std::sync::Arc<JitCellToken>> {
    descr
        .as_loop_token_descr()?
        .token_handle_any()?
        .downcast_ref::<std::sync::Arc<JitCellToken>>()
}

/// A `CALL_ASSEMBLER` whose callee has not published locs cannot be rewritten.
/// The message matches `wasm_unsupported_trace_reason` so the decline tally
/// stays on the same string.
fn missing_call_assembler_locs(ops: &[Op]) -> Option<String> {
    for op in ops.iter().filter(|op| op.opcode.is_call_assembler()) {
        let ready = op.getdescr().is_some_and(|d| {
            call_assembler_token_from_descr(&d)
                .and_then(|tok| ca_locs_from_token(tok))
                .is_some()
        });
        if !ready {
            return Some(format!(
                "wasm backend: {:?} (loop-callee inline)",
                op.opcode
            ));
        }
    }
    None
}

/// Production GC rewriter used by `compile_loop` / `compile_bridge`.
///
/// `llsupport/gc.py` `get_ll_description` + `rewrite.py`
/// `GcRewriterAssembler`. Native backends run this before assemble.
/// `CALL_ASSEMBLER` goes through `handle_call_assembler`. The callee's
/// `_ll_initial_locs` are published by `publish_ca_initial_locs` before
/// this rewriter runs. malloc / zero / barrier / `GC_LOAD` are the same
/// shared rewrite.
#[doc(hidden)]
pub fn gc_rewriter() -> majit_gc::rewrite::GcRewriterImpl {
    let collector = with_wasm_active_gc(|gc| {
        (
            gc.nursery_free_addr(),
            gc.nursery_top_addr(),
            gc.max_nursery_object_size(),
            gc.get_write_barrier_descr(),
        )
    });
    let is_boehm = collector.is_none();
    let (nursery_free_addr, nursery_top_addr, max_nursery_size, wb_descr) =
        collector.unwrap_or((0, 0, 0, None));
    majit_gc::rewrite::GcRewriterImpl {
        nursery_free_addr,
        nursery_top_addr,
        max_nursery_size,
        wb_descr,
        jitframe_info: Some(wasm_jitframe_descrs()),
        call_assembler_callee_locs: Some(Box::new(|_ltd| None)),
        load_supported_factors: &[1],
        supports_load_effective_address: true,
        malloc_zero_filled: is_boehm,
        memcpy_fn: majit_ir::memcpy_fn_addr(),
        memcpy_descr: majit_ir::make_memcpy_calldescr(),
        str_descr: codegen::builtin_string_array_descr(majit_ir::OpCode::Newstr)
            .expect("Newstr must produce a str ArrayDescr"),
        unicode_descr: codegen::builtin_string_array_descr(majit_ir::OpCode::Newunicode)
            .expect("Newunicode must produce a unicode ArrayDescr"),
        str_hash_descr: codegen::builtin_string_hash_field_descr(majit_ir::OpCode::Strhash)
            .expect("Strhash must produce a str hash FieldDescr"),
        unicode_hash_descr: codegen::builtin_string_hash_field_descr(majit_ir::OpCode::Unicodehash)
            .expect("Unicodehash must produce a unicode hash FieldDescr"),
        fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
        fielddescr_tid: (!is_boehm).then(majit_ir::make_tid_field_descr),
        malloc_array_fn: wasm_malloc_array as *const () as i64,
        malloc_array_nonstandard_fn: wasm_malloc_array_nonstandard as *const () as i64,
        malloc_array_oldgen_fn: wasm_malloc_array_oldgen as *const () as i64,
        malloc_array_nonstandard_oldgen_fn: wasm_malloc_array_nonstandard_oldgen as *const ()
            as i64,
        malloc_str_fn: wasm_malloc_str as *const () as i64,
        malloc_unicode_fn: wasm_malloc_unicode as *const () as i64,
        malloc_big_fixedsize_fn: wasm_malloc_big_fixedsize as *const () as i64,
        malloc_big_fixedsize_oldgen_fn: wasm_malloc_big_fixedsize_oldgen as *const () as i64,
        malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
        malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
        malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
        malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
        malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
        standard_array_basesize: std::mem::size_of::<usize>(),
        standard_array_length_ofs: 0,
        headerless_fixedsize: false,
    }
}

/// Same GC rewrite `compile_loop` / `compile_bridge` run before `build_wasm_module`.
#[doc(hidden)]
pub fn rewrite_ops_for_gc(
    ops: Vec<Op>,
    constants: &indexmap::IndexMap<u32, i64>,
) -> (
    Vec<Op>,
    indexmap::IndexMap<u32, i64>,
    Option<Arc<majit_gc::GcTable>>,
) {
    let mut rewriter = gc_rewriter();
    rewriter.call_assembler_callee_locs = Some(Box::new(|ltd| {
        ltd.token_handle_any()
            .and_then(|any| any.downcast_ref::<std::sync::Arc<JitCellToken>>())
            .and_then(|tok| ca_locs_from_token(tok))
    }));
    rewrite_ops_for_gc_with(&rewriter, ops, constants)
}

/// [`rewrite_ops_for_gc`] with an explicit rewriter (tests that need
/// IncrementalMiniMark `malloc_zero_filled=false` when no collector is bound).
#[doc(hidden)]
pub fn rewrite_ops_for_gc_with(
    rewriter: &majit_gc::rewrite::GcRewriterImpl,
    ops: Vec<Op>,
    constants: &indexmap::IndexMap<u32, i64>,
) -> (
    Vec<Op>,
    indexmap::IndexMap<u32, i64>,
    Option<Arc<majit_gc::GcTable>>,
) {
    use majit_gc::GcRewriter;
    // rewrite.py `gen_malloc_str` parity: inject str_descr/unicode_descr for
    // NEWSTR/NEWUNICODE. The STRLEN/STRGETITEM/STRHASH arms read the length
    // and hash offsets off that descr, so the stream has to carry it before
    // the rewrite, not after — dynasm `assemble_loop` and cranelift
    // `rewrite_ops` inject at the same point.
    codegen::inject_builtin_string_descrs(&ops);
    let boxed: Vec<majit_ir::OpRc> = ops.into_iter().map(majit_ir::OpRc::new).collect();
    let mut const_map = majit_ir::ConstMap::default();
    for (&k, &v) in constants {
        const_map.insert(k, majit_ir::Const::from_raw_i64(v, majit_ir::Type::Int));
    }
    let (rewritten, gcrefs) = rewriter.rewrite_for_gc_with_constants(&boxed, &const_map);
    // `resolve_constant` only reads the pool, so the caller's i64 map is
    // the map the next pass should see.
    let out_constants = constants.clone();
    let ops: Vec<Op> = rewritten.iter().map(|rc| (**rc).clone()).collect();
    let table = (!gcrefs.is_empty()).then(|| majit_gc::GcTable::from_gcrefs(&gcrefs));
    (ops, out_constants, table)
}

/// `GcTable::compile_key` list, slot order. Empty when this compile has no table.
fn gc_const_keys_of(table: Option<&majit_gc::GcTable>) -> Vec<usize> {
    let Some(table) = table else {
        return Vec::new();
    };
    (0..table.len()).map(|i| table.compile_key(i)).collect()
}

/// rewrite.py `gen_malloc_fixedsize` / gc.py `malloc_big_fixedsize(size, tid)`.
/// The CALL_R size is `payload + GcHeader::SIZE`; [`wasm_jit_alloc`] takes
/// `(type_id, payload_size)`.
pub extern "C" fn wasm_malloc_big_fixedsize(size: i64, type_id: i64) -> i64 {
    let payload = (size as usize).saturating_sub(majit_gc::header::GcHeader::SIZE);
    wasm_jit_alloc(type_id, payload as i64)
}

/// `malloc_big_fixedsize` old-generation twin for a `non_moving` size descr.
pub extern "C" fn wasm_malloc_big_fixedsize_oldgen(size: i64, type_id: i64) -> i64 {
    let payload = (size as usize).saturating_sub(majit_gc::header::GcHeader::SIZE);
    wasm_jit_alloc_oldgen(type_id, payload as i64)
}
/// Exact guest-side implementation for the JIT IR's `FloatMod`. Keeping this
/// in the interpreter module avoids both an incorrect arithmetic expansion
/// (wasm has no remainder instruction) and a guest→host→guest call.
extern "C" fn wasm_jit_fmod(a: f64, b: f64) -> f64 {
    a % b
}

/// Table indices of runtime trampolines used by trace codegen. Taking each
/// address here is what keeps the function in the module's
/// `__indirect_function_table`, so a trace can `call_indirect` it.
fn alloc_helpers() -> codegen::AllocHelpers {
    codegen::AllocHelpers {
        new_fn_ptr: wasm_jit_alloc as *const () as usize as i64,
        new_array_fn_ptr: wasm_jit_alloc_array as *const () as usize as i64,
        headerless_fn_ptr: wasm_jit_alloc_headerless as *const () as usize as i64,
        threadlocal_fn_ptr: wasm_jit_threadlocalref_get as *const () as usize as i64,
        fmod_fn_ptr: wasm_jit_fmod as *const () as usize as i64,
        write_real_errno_fn_ptr: wasm_jit_write_real_errno as *const () as usize as i64,
        read_real_errno_fn_ptr: wasm_jit_read_real_errno as *const () as usize as i64,
    }
}

/// JIT-trace write-barrier trampoline for `CondCallGcWb`. Invokes the
/// active GC's `write_barrier`, which adds an old object that may now
/// hold a young reference to the remembered set (and clears
/// TRACK_YOUNG_PTRS). A young base (no flag) or a null base is a no-op.
/// Returns 0 — the store codegen ignores it.
pub extern "C" fn wasm_jit_write_barrier(obj: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.write_barrier(GcRef(obj as usize)));
    0
}

/// Array-store counterpart of [`wasm_jit_write_barrier`].
///
/// `incminimark.py jit_remember_young_pointer_from_array` / dynasm
/// `dynasm_write_barrier_from_array`: the JIT has already seen
/// TRACK_YOUNG_PTRS set and CARDS_SET clear. If the object has cards,
/// arm CARDS_SET; otherwise fall back to the generic remembered-set
/// path. The caller then marks the card inline when CARDS_SET is set.
pub extern "C" fn wasm_jit_write_barrier_from_array(obj: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.jit_remember_young_pointer_from_array(GcRef(obj as usize)));
    0
}

/// The write-barrier geometry and helper addresses the emitted barrier reads.
///
/// `gc.py` sets `write_barrier_descr` from the collector, and the backends read
/// it back rather than assuming a layout: `jit_wb_cards_set` is zero for a
/// collector configured without cards, and `jit_wb_card_page_shift` is that
/// collector's own shift. Falling back to the current header layout when no GC
/// is live mirrors `dynasm_write_barrier_descr`.
fn wasm_write_barrier_helpers() -> codegen::WriteBarrierHelpers {
    // `fn as usize` is the `__indirect_function_table` index on wasm32; taking
    // it here keeps the function in the table.
    let fn_ptr = wasm_jit_write_barrier as *const () as usize as i64;
    let array_fn_ptr = wasm_jit_write_barrier_from_array as *const () as usize as i64;
    match with_wasm_active_gc(|gc| gc.get_write_barrier_descr()).flatten() {
        Some(descr) => codegen::WriteBarrierHelpers::new(fn_ptr, array_fn_ptr, &descr),
        None => codegen::WriteBarrierHelpers::for_current_gc(fn_ptr, array_fn_ptr),
    }
}

/// `_call_header_shadowstack` when `jf_top` is not in linear memory.
///
/// `handle_call_assembler` already emitted `CallMallocNurseryVarsizeFrame`
/// for `frame_ptr`. The guest emits the two stores itself when
/// `CaInlineParams` publishes `jf_top` / `jf_limit`. This helper is the path
/// where the shadow stack stays in host TLS (`shadow_stack.rs` `JF_ROOT_STACK`),
/// which a wasm module cannot address.
pub extern "C" fn wasm_jit_ca_push_frame(frame_ptr: i64) -> i64 {
    if frame_ptr == 0 {
        return 0;
    }
    let jf_ref = GcRef(frame_ptr as usize);
    majit_gc::shadow_stack::push_jf(jf_ref);
    frame_ptr
}

/// Pop the top jitframe shadow-stack
/// entry on CA-arm exit. The CA recursion is strict LIFO — each level pushes
/// one frame before its `call_indirect` and pops after, and a deopt resume runs
/// on the host's own shadow stack — so removing the top entry releases exactly
/// this callee's frame.
///
/// `_call_footer_shadowstack` for a module whose shadow-stack cells are not
/// published; the inline footer is the same `SUB`. No barrier: an old
/// callee frame was remembered by its own `_reload_frame_if_necessary`
/// after each collecting call.
pub extern "C" fn wasm_jit_ca_pop_frame(_items_base: i64) -> i64 {
    if majit_gc::shadow_stack::jf_top_ptr().is_null() {
        return 0;
    }
    majit_gc::shadow_stack::pop_jf_top();
    0
}

/// Reload the current CA callee frame pointer after a recursive call. The GC
/// may have moved the callee frame during the recursive call; `jf_top_ptr()`
/// reads the forwarded base from the jitframe shadow-stack slot. At this point
/// this recursion level's frame is the top — deeper levels have already popped.
/// Analog of `_reload_frame_if_necessary`; returns the ITEMS base held in the
/// CA arm's `ca_cfp_local`.
pub extern "C" fn wasm_jit_ca_reload_frame() -> i64 {
    let jf = majit_gc::shadow_stack::jf_top_ptr().0 as *mut majit_backend::jitframe::JitFrame;
    // `_check_frame_depth` may have threaded `jf_forward` without the
    // shadow slot moving yet. `jitframe_resolve` returns the frame the
    // callee actually ran on.
    let jf = if jf.is_null() {
        jf
    } else {
        unsafe { majit_backend::jitframe::JitFrame::resolve(jf) }
    };
    jf as i64 + majit_backend::jitframe::FIRST_ITEM_OFFSET as i64
}

/// Reload the CA caller's frame pointer after the callee-frame allocation.
/// The allocation occurs before the callee is pushed, so while the callee is
/// live the caller remains one entry below the shadow-stack top. Returns that
/// caller's ITEMS base for local 0.
pub extern "C" fn wasm_jit_ca_reload_caller_frame() -> i64 {
    let jf = majit_gc::shadow_stack::jf_under_top_ptr().0 as *mut majit_backend::jitframe::JitFrame;
    let jf = if jf.is_null() {
        jf
    } else {
        unsafe { majit_backend::jitframe::JitFrame::resolve(jf) }
    };
    jf as i64 + majit_backend::jitframe::FIRST_ITEM_OFFSET as i64
}

/// `llmodel.py realloc_frame`, called from `_check_frame_depth` when the
/// running frame's `jf_frame` length is below this module's item count.
///
/// `items_base` is wasm local 0 (the `jf_frame` items). The returned items
/// base belongs to the new frame, or 0 when allocation fails. The guest then
/// keeps local 0 on the old frame and takes the MemoryError exit.
/// `malloc_jitframe_no_collect` places a registered jitframe type id
/// (`jitframe_prefer_oldgen` is false, so the nursery, unless the request
/// exceeds it). No type id uses `alloc_off_gc_jitframe`.
/// `jf_frame_info` is the token's `CompiledLoopToken.frame_info` (`JitFrame::init`
/// on the host entry, `rewrite.py` `gen_malloc_frame` on a CA callee).
///
/// # Safety
/// `items_base` must be the items pointer of a live `JitFrame` whose
/// `jf_frame_info` is non-null.
pub unsafe extern "C" fn wasm_realloc_frame(items_base: i64, depth: i64) -> i64 {
    use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame, JitFrameInfo};
    if items_base == 0 || depth <= 0 {
        return items_base;
    }
    let old_jf = (items_base as usize - FIRST_ITEM_OFFSET) as *mut JitFrame;
    let base_ofs = (majit_gc::header::GcHeader::SIZE + FIRST_ITEM_OFFSET) as isize;
    let tid = wasm_jitframe_tid();
    // `realloc_frame` initializes whatever `alloc` returns. Size the frame
    // the same way first and refuse a null before that write.
    let prepared = unsafe {
        let fi = (*old_jf).jf_frame_info as *mut JitFrameInfo;
        if depth as isize > (*fi).depth() {
            (*fi).update_frame_depth(base_ofs as i64, depth);
        }
        // `jfi_frame_size` counts the GC header (`base_ofs` above, the
        // `CallMallocNurseryVarsizeFrame` total); both allocators take the
        // payload after it.
        let alloc_bytes = (*fi).size() as usize - majit_gc::header::GcHeader::SIZE;
        if tid != 0 {
            wasm_malloc_jitframe_no_collect(tid, alloc_bytes)
        } else {
            let new_jf = majit_backend::jitframe::alloc_off_gc_jitframe(alloc_bytes);
            if !new_jf.is_null() {
                majit_gc::shadow_stack::register_libc_jitframe(new_jf as usize);
            }
            new_jf
        }
    };
    if prepared.is_null() {
        return 0;
    }
    let new_jf = unsafe {
        majit_backend::jitframe::realloc_frame(
            old_jf,
            depth as isize,
            base_ofs,
            |_size_bytes| prepared,
            |new_jf| {
                if tid != 0 {
                    wasm_jit_write_barrier(new_jf as i64);
                }
            },
        )
    };
    // `frame.jf_forward = new_frame` (`llmodel.py`). `realloc_frame` writes
    // it raw; a GC frame barriers the old object the way `dynasm_realloc_frame` does.
    if tid != 0 {
        wasm_jit_write_barrier(old_jf as i64);
    }
    // assembler.py `_build_frame_realloc_slowpath` stores the new frame at
    // `[root_stack_top - WORD]` unconditionally.
    retarget_shadow_jf(old_jf, new_jf);
    (new_jf as usize + FIRST_ITEM_OFFSET) as i64
}

/// `malloc_jitframe_no_collect` without `zeroed_gc_frame`'s assert.
///
/// `jitframe_prefer_oldgen` selects the generation. A null result is the
/// guest's MemoryError (`wasm_realloc_frame` returns 0).
fn wasm_malloc_jitframe_no_collect(
    type_id: u32,
    size_bytes: usize,
) -> *mut majit_backend::jitframe::JitFrame {
    with_wasm_active_gc_mut(|gc| {
        let gcref = if majit_backend::jitframe::jitframe_prefer_oldgen() {
            gc.alloc_oldgen_typed(type_id, size_bytes)
        } else {
            gc.alloc_nursery_no_collect_typed(type_id, size_bytes)
        };
        if gcref.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            std::ptr::write_bytes(gcref.0 as *mut u8, 0, size_bytes);
        }
        gcref.0 as *mut majit_backend::jitframe::JitFrame
    })
    .unwrap_or(std::ptr::null_mut())
}

/// `assembler.py` `_build_frame_realloc_slowpath`: the top shadow slot is this
/// frame, and the slowpath overwrites it with the grown one.
fn retarget_shadow_jf(
    old: *mut majit_backend::jitframe::JitFrame,
    new: *mut majit_backend::jitframe::JitFrame,
) {
    let top_addr = majit_gc::shadow_stack::get_root_stack_top_addr();
    let word = std::mem::size_of::<usize>();
    unsafe {
        let top = *(top_addr as *const usize);
        debug_assert!(top >= word);
        let slot = (top - word) as *mut usize;
        debug_assert_eq!(*slot, old as usize);
        *slot = new as usize;
    }
}

/// Host-side root-register trampoline.
///
/// # Safety
/// Caller must keep `slot` valid until [`wasm_gc_remove_root`] is
/// called with the same pointer.
pub(crate) unsafe fn wasm_gc_add_root(slot: *mut GcRef) {
    with_wasm_active_gc_mut(|gc| unsafe { gc.add_root(slot) });
}

/// Batched [`wasm_gc_add_root`] for one stack-shaped root bracket.
///
/// # Safety
/// Every slot must remain valid until removed with [`wasm_gc_remove_roots`].
#[allow(dead_code)] // batched wasm_gc_add_root / gc.add_root
pub(crate) unsafe fn wasm_gc_add_roots(slots: &[usize]) {
    if slots.is_empty() {
        return;
    }
    with_wasm_active_gc_mut(|gc| {
        for &slot in slots {
            unsafe { gc.add_root(slot as *mut GcRef) };
        }
    });
}

/// Companion to [`wasm_gc_add_root`].
pub(crate) fn wasm_gc_remove_root(slot: *mut GcRef) {
    with_wasm_active_gc_mut(|gc| gc.remove_root(slot));
}

/// Batched [`wasm_gc_remove_root`] for one stack-shaped root bracket.
#[allow(dead_code)] // batched wasm_gc_remove_root / gc.remove_root
pub(crate) fn wasm_gc_remove_roots(slots: impl Iterator<Item = usize>) {
    with_wasm_active_gc_mut(|gc| {
        for slot in slots {
            gc.remove_root(slot as *mut GcRef);
        }
    });
}

/// Host-side write-barrier trampoline for the interpreter (mapdict / list /
/// set / dict stores route through `majit_gc::gc_write_barrier`). Mirrors
/// `dynasm_gc_write_barrier`; without it every interpreter ref-store is a
/// silent no-op, so a collecting nursery loses old→young pointers.
fn wasm_active_gc_write_barrier_before_move(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.writebarrier_before_move(obj));
}

fn wasm_active_gc_write_barrier_from_array(obj: GcRef, index: usize) {
    with_wasm_active_gc_mut(|gc| gc.write_barrier_from_array(obj, index));
}

fn wasm_active_gc_writebarrier_before_copy(
    source: GcRef,
    dest: GcRef,
    source_start: usize,
    dest_start: usize,
    length: usize,
) -> bool {
    with_wasm_active_gc_mut(|gc| {
        gc.writebarrier_before_copy(source, dest, source_start, dest_start, length)
    })
    .unwrap_or(true)
}

pub(crate) fn wasm_active_gc_write_barrier(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.write_barrier(obj));
}

fn wasm_active_gc_write_barrier_managed(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.write_barrier_managed(obj));
}

/// Host-side `is_managed_heap_object` trampoline.
///
/// This query can fire reentrantly from an extra-root walker mid-collection
/// (the interpreter-safepoint major holds the box's mutable borrow while
/// asking whether a slot is GC-managed), so both arms are read-only.
pub(crate) fn wasm_gc_owns_object(addr: usize) -> bool {
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.is_managed_heap_object(addr)) {
        return r;
    }
    majit_gc::gc_sync::is_initialized()
        && majit_gc::gc_sync::gc_query_reentrant(|g| g.is_managed_heap_object(addr))
}

/// `llop.shrink_array`.  This changes a GC-owned object's length word and must
/// therefore take the exclusive collector path; `gc_query_reentrant` is for
/// read-only queries made while a collection may already hold `&mut`.
fn wasm_gc_shrink_array(addr: usize, smaller_length: usize) -> bool {
    with_wasm_active_gc_mut(|gc| gc.shrink_array(addr, smaller_length)).unwrap_or(false)
}

fn wasm_gc_varsize_layout(addr: usize) -> Option<majit_gc::GcVarSizeLayout> {
    let obj = GcRef(addr);
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.varsize_layout(obj)) {
        return r;
    }
    if majit_gc::gc_sync::is_initialized() {
        majit_gc::gc_sync::gc_query_reentrant(|g| g.varsize_layout(obj))
    } else {
        None
    }
}

pub struct WasmBackend {
    /// `rpython/jit/backend/model.py __init__ self.tracker =
    /// CPUTotalTracker()` parity — per-instance `cpu.tracker`
    /// exposed via [`majit_backend::Backend::cpu_tracker`].
    cpu_tracker: std::sync::Arc<majit_backend::CpuTotalTracker>,
    /// `asmmemmgr.py` `AsmMemoryManager` parity — what
    /// `jit_hooks.stats_asmmemmgr_{allocated,used}` reads. The emitted trace is
    /// a wasm module handed to the host compiler, so there is no arena of ours
    /// to size: `allocated` and `used` are both the module's byte length, which
    /// is the figure `asmmemmgr.py` counts for a block a `materialize`
    /// handed out.
    asm_memory_stats: std::sync::Arc<majit_backend::AsmMemoryManagerStats>,
    /// Lifetime tokens for the blocks recorded above. The host keeps every
    /// instantiated module for as long as this backend can enter it, so the
    /// tokens are held for the backend's life and give `used` back with it.
    asm_memory_blocks: std::cell::RefCell<Vec<majit_backend::AsmMemoryBlock>>,
    trace_counter: u64,
    /// One-shot header PC the metainterp publishes before `compile_loop`.
    next_header_pc: u64,
    /// Optimizer constant pool (constant-namespace OpRef → i64 value).
    constants: indexmap::IndexMap<u32, i64>,
    /// llmodel.py:64-69 self.vtable_offset.
    vtable_offset: Option<usize>,
    /// Byte offset of the class word beside the type word; see
    /// `Backend::set_w_class_offset`.
    w_class_offset: Option<usize>,
    /// Test-path `gc_ll_descr`. Dropping the backend uninstalls the TLS
    /// box so a cargo worker thread does not run MiniMark `Drop` at
    /// pthread TLS teardown.
    gc_box: Option<ActiveGcBox>,
    /// `compile.py` `make_and_attach_done_descrs` and `pyjitpl.py`
    /// `propagate_exception_descr`. Heap-pinned so a moved `WasmBackend`
    /// keeps the `jf_descr` immediates compiled modules already baked.
    pub(crate) exit_cells: std::sync::Arc<failguard::CpuExitCells>,
    /// Trips recorded while a bridge was running: the owner's `Weak` plus the
    /// slot address. A later drain upgrades the owner and checks the address
    /// is still in that owner's `asmmemmgr_blocks` before dereferencing it.
    tripped_inlines: RefCell<Vec<(Weak<JitCellToken>, usize)>>,
}

/// GC type id of the `JitFrame`. The single registration authority is `eval.rs`
/// (the type is registered there alongside the rest of the heap types, before
/// `freeze_types`); it pushes the id here through `set_wasm_jitframe_tid`,
/// mirroring how it feeds `majit_backend_{cranelift,dynasm}::set_jitframe_gc_type_id`.
/// The orthodox (`PYRE_WASM_CA`) frame path allocates the host-entry frame as a
/// real GC-managed `JitFrame` of this type so the collector forwards its Ref item
/// slots through the `jf_gcmap` custom trace. 0 = not yet pushed (the orthodox
/// path stays disabled until then).
static WASM_JITFRAME_TID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Host entry point used by `eval.rs` to publish the registered `JitFrame` type
/// id (counterpart to `set_jitframe_gc_type_id` on the native backends).
pub fn set_wasm_jitframe_tid(id: u32) {
    WASM_JITFRAME_TID.store(id, std::sync::atomic::Ordering::Relaxed);
    majit_gc::bh_probe_ignore_tid(id);
}

// Only read on the wasm32 execute_token path and by CA callee-frame allocation.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn wasm_jitframe_tid() -> u32 {
    WASM_JITFRAME_TID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Item bytes for one host entry. `frame_info.depth` grows in
/// `wasm_realloc_frame` via `update_frame_depth`, so a later entry allocates
/// the enlarged frame up front. A loop that never grew keeps `frame_bytes`.
fn entry_frame_bytes(token: &JitCellToken, frame_bytes: u32) -> usize {
    let sign = std::mem::size_of::<isize>();
    let from_info = token
        .compiled_loop_token()
        .map(|clt| clt.frame_info.lock().depth().max(0) as usize * sign)
        .unwrap_or(0);
    (frame_bytes as usize).max(from_info)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn frame_info_ptr(token: &JitCellToken) -> *const majit_backend::jitframe::JitFrameInfo {
    token
        .compiled_loop_token()
        .map(|clt| clt.frame_info.data_ptr() as *const majit_backend::jitframe::JitFrameInfo)
        .unwrap_or(std::ptr::null())
}

/// Build a `jf_gcmap` bitmap marking the surviving Ref-home region as the
/// frame's traced GC roots, in the `JitFrame`'s Signed-granular item indexing.
///
/// On wasm32 `isize` is 4 bytes, so `jf_frame` items are 4-byte Signed slots and
/// each 8-byte data slot spans two items — the orthodox PyPy 32-bit layout where
/// a one-word value (a `GcRef`) occupies a single item and a two-word value
/// (i64) occupies a pair. A Ref home written as an i64 keeps the guest pointer in
/// its LOW word (little-endian), at Signed item index `(HOME_SLOT_BASE + h *
/// 8) / sign`. `jitframe_trace` strides items by `sign` and forwards one word per
/// marked bit, so marking those indices exposes each home's `GcRef` (the high
/// word stays unmarked). Returns `[data_word_count, word0, ...]` in `usize`
/// words (GCMAP array layout: `gcmap[0]` = number of data words).
fn leak_home_gcmap(
    own: &mut release::LoopAsmResources,
    frame: codegen::FrameGeometry,
    used_ordinary: usize,
    used_labels: usize,
) -> usize {
    own.park_gcmap(codegen::build_home_gcmap(frame, used_ordinary, used_labels))
}

/// Bitwise union of two GCMAP arrays (`[n, word0, ...]`).
///
/// Ordinary homes and LABEL captures grow independently, so two live maps
/// can be incomparable: a retained bridge may mark more LABEL bits while a
/// re-emitted owner marks more ordinary bits. Returning either pointer
/// unmarks the other region. `None` of either argument is treated as empty.
/// When one already covers the other the covered pointer is returned so a
/// comparable publish does not leak.
pub extern "C" fn wasm_jit_union_gcmap(old: i64, new: i64) -> i64 {
    let old_ptr = old as usize as *const usize;
    let new_ptr = new as usize as *const usize;
    if old_ptr.is_null() {
        return new;
    }
    if new_ptr.is_null() {
        return old;
    }
    unsafe {
        let n_old = *old_ptr;
        let n_new = *new_ptr;
        let n_overlap = n_old.min(n_new);
        let mut old_extra = false;
        let mut new_extra = false;
        for i in 0..n_overlap {
            let o = *old_ptr.add(1 + i);
            let n = *new_ptr.add(1 + i);
            old_extra |= o & !n != 0;
            new_extra |= n & !o != 0;
        }
        for i in n_overlap..n_old {
            old_extra |= *old_ptr.add(1 + i) != 0;
        }
        for i in n_overlap..n_new {
            new_extra |= *new_ptr.add(1 + i) != 0;
        }
        if !old_extra {
            return new;
        }
        if !new_extra {
            return old;
        }
        let n = n_old.max(n_new);
        let mut buf = vec![0usize; 1 + n];
        buf[0] = n;
        for i in 0..n_old {
            buf[1 + i] |= *old_ptr.add(1 + i);
        }
        for i in 0..n_new {
            buf[1 + i] |= *new_ptr.add(1 + i);
        }
        // Runtime union. The two inputs are compile-time maps owned by the
        // loop's asm blocks. This result is stored into `jf_gcmap` while the
        // trace is running; `allocate_gcmap` likewise hands the caller a
        // leaked pointer with no second owner.
        Box::into_raw(buf.into_boxed_slice()) as *mut usize as i64
    }
}

/// `__indirect_function_table` slot of `call_jit::wasm_ca_resume_deopt`,
/// published by pyre-jit at boot (`init_jit_hooks`). When an in-guest
/// self-recursive CALL_ASSEMBLER callee leaves its trace through a guard with no
/// bridge — a deopt the in-guest fast path cannot finish — the CA arm
/// `call_indirect`s this slot to blackhole-resume that callee on the host (no
/// re-execution of its pre-guard work) and read back its result. `0` (unset)
/// makes `compile_bridge` decline the CA lift, since the arm would have no way
/// to complete a deopt. Stored as `u64` to reuse the imported atomics.
static CA_DEOPT_HELPER_SLOT: AtomicU64 = AtomicU64::new(0);
/// Dormant runtime-regression selector. The wasm runner writes this through a
/// guest export before executing a test program; zero keeps production runs
/// unchanged. `1` selects the first admitted target, otherwise the value is a
/// `JitCellToken` number.
static FORCE_CA_TERMINAL_DECLINE: AtomicU64 = AtomicU64::new(0);

/// `__indirect_function_table` index of the deferred-merge trip callback,
/// published from pyre-jit the way [`CA_DEOPT_HELPER_SLOT`] is. Zero keeps
/// every merge deferred forever, which is what a host without the callback
/// wants: the bridge stays out of line and correct.
static INLINE_TRIP_HELPER_SLOT: AtomicU64 = AtomicU64::new(0);

/// Entries into an out-of-line bridge before its merge into the owner is
/// installed.
///
/// Merging costs the owner a re-emission — a whole module rebuild, tens of ms
/// of compile time — and buys 7.78ns per cross-module crossing it removes.
/// Nothing at admission time predicts how many that will be: every bridge is
/// compiled at the same guard-failure threshold, and the crossings that decide
/// the answer all happen afterwards and stay in-guest. So the bridge counts its
/// own entries and asks for the merge here.
///
/// The count is calibrated against the two populations the corpus actually
/// holds, because a bridge's entry count is a floor on the crossings its merge
/// removes rather than an estimate of them — merging also brings the region's
/// own external jumps in-module, which on `short_circuit_value_kept_stack` made
/// three merges worth ten times their bridges' entries. Below this, a fixture's
/// ENTIRE crossing budget is under a millisecond and no rebuild can pay back:
/// `polymorphic_slot_retype` crosses 88.4k times in its whole run and
/// `short_circuit_side_effects` 37.7k, and eagerly merging cost them 1.09x
/// each. Above it, the hot bridges of `short_circuit_value_kept_stack` (106k to
/// 532k entries) are worth 0.82x, and `str_getitem_len_hot`'s single region is
/// worth 0.75x on 72.0M crossings removed.
const INLINE_TRIP_THRESHOLD: u64 = 100_000;

/// The per-byte half of the same price, in entries per byte of the module the
/// merge re-emits — see [`inline_trip_threshold_for`], which takes the larger
/// of the two.
///
/// A merge charges cranelift for the whole owner while the crossings it removes
/// answer only to the bridge, so a fixture whose owner is large enough loses on
/// a merge the entry count alone would have taken. At this value `fannkuch`
/// keeps 8 of its 10 merges and `nbody` 3 of its 4, dropping 30KB and 18KB of
/// emitted module, while the merges of the four fixtures that never lose one
/// are postponed by an amount too small to charge them anything. Twice this is
/// already inside the band where postponement dominates.
const DEFAULT_INLINE_TRIP_BYTES_FACTOR: u64 = 40;

/// Bytes of owner module above which a header merge waits for entry evidence
/// through [`inline_trip_threshold_for`], rather than re-emitting immediately.
/// Quasi-immutable dependencies are token-owned, so deferred installation does
/// not weaken invalidation. Keep the existing eager cost boundary; a large
/// owner is not grounds for permanently losing a profitable merge.
const DEFAULT_INLINE_EAGER_MAX_BYTES: u32 = 4096;

/// A merge that passed every inline check and is waiting on
/// [`INLINE_TRIP_THRESHOLD`] entries into the bridge compiled in its place.
///
/// Installation replaces the retained wasm module after compiled execution
/// returns. Until that swap the guard keeps dispatching to the attached
/// bridge, the same window `patch_jump_for_descr` leaves closed. Invalidation
/// remains owned by the loop token across module replacement.
pub(crate) struct PendingInline {
    /// The loop this region merges into. Weak so a leftover retry
    /// cannot keep an otherwise unreachable owner (and its module)
    /// alive for the rest of the thread.
    owner: Weak<JitCellToken>,
    region: codegen::InlinedBridge,
    /// When the source guard lived on a standalone parent bridge, remap
    /// `(parent_trace_id, parent_local_fail_index)` at install once that
    /// parent is in the owner's merged stream. `None` is the ordinary
    /// owner-stream index already stored on `region`.
    remap: Option<(u64, u32)>,
    /// Set when a one-shot trip leftover is waiting on a sibling peel
    /// (`uninitialized_label`). Ordinary deferred entries stay false so a
    /// hot sibling cannot pull them in before their own threshold.
    retry_on_sibling: bool,
}

impl PendingInline {
    fn owner(&self) -> Option<Arc<JitCellToken>> {
        self.owner.upgrade()
    }

    fn same_owner(&self, owner: &Arc<JitCellToken>) -> bool {
        self.owner().is_some_and(|o| Arc::ptr_eq(&o, owner))
    }
}

/// `addr` is the box address baked into the bridge (`alloc_pending_inline`).
///
/// # Safety
/// `addr` must still be a `PendingInlineSlot` box inside some live
/// `LoopAsmResources::pending_inlines`. `free_loop_and_bridges` drops that box.
unsafe fn pending_slot_at(addr: usize) -> &'static release::PendingInlineSlot {
    unsafe { &*(addr as *const release::PendingInlineSlot) }
}

fn pending_slot_is_member(owner: &JitCellToken, addr: usize) -> bool {
    owner_pending_slot_addrs(owner).contains(&addr)
}

/// Pending-inline slots on `owner`, in `asmmemmgr_blocks` order.
/// `push_resources` runs at the end of each `compile_bridge`, which is the
/// order the old process-wide ids were allocated, so this replaces the id sort.
/// The lock is not held after this returns: a later re-emission pushes blocks.
fn owner_pending_slot_addrs(owner: &JitCellToken) -> Vec<usize> {
    let Some(clt) = owner.compiled_loop_token() else {
        return Vec::new();
    };
    let blocks = clt.asmmemmgr_blocks.lock();
    let mut addrs = Vec::new();
    for block in blocks.iter() {
        let Some(resources) = block.downcast_ref::<release::LoopAsmResources>() else {
            continue;
        };
        for slot in &resources.pending_inlines {
            addrs.push(&**slot as *const release::PendingInlineSlot as usize);
        }
    }
    addrs
}

/// Drop every live deferred merge of an invalidated `owner` and point its
/// guard cells back at the attached bridges. A dead owner's slots are freed
/// with its `asmmemmgr_blocks`.
fn discard_invalidated_owner_inlines(owner: &Arc<JitCellToken>) {
    for addr in owner_pending_slot_addrs(owner) {
        // SAFETY: `addr` was just read out of this owner's `pending_inlines`.
        let Some(item) = unsafe { pending_slot_at(addr) }.entry.borrow_mut().take() else {
            continue;
        };
        if item.remap.is_none() {
            WasmBackend::restore_dispatch_cell(owner, item.region.source_fail_index);
        }
    }
}

/// The merged-stream exit ordinal of a guard belonging to a region already
/// merged into this loop, or `None` when `trace_id` names no such region.
///
/// `InlinedBridge::source_fail_index` indexes the merged stream — the owner's
/// own ops followed by every region's, in attach order — whose exits are
/// numbered across the whole of it. A guard in the owner itself is already at
/// its own ordinal; a region's guards start past the owner's and past every
/// region attached before it. The answer stays valid however long a deferred
/// merge waits, because merging only ever appends.
fn merged_region_fail_index(
    inputs: &codegen::ModuleBuildInputs,
    trace_id: u64,
    region_fail_index: u32,
) -> Option<u32> {
    let mut ordinal = codegen::guard_exit_count(&inputs.inputargs, &inputs.ops);
    for region in &inputs.inlined_bridges {
        let count = codegen::guard_exit_count(&region.inputargs, &region.ops);
        if region.trace_id == trace_id {
            return ((region_fail_index as usize) < count)
                .then(|| (ordinal + region_fail_index as usize) as u32);
        }
        ordinal += count;
    }
    None
}

/// Note that a bridge has counted its way to the threshold. Called from
/// compiled code, inside the bridge module. Install now: the parent is not
/// re-entered, and its next back-edge loads the resume cell. The slot stays
/// on the queue so `execute_assembler` can retry after the trace returns.
pub fn record_inline_trip(pending_slot: i64) {
    let backend = EXECUTING_WASM_BACKEND.load(std::sync::atomic::Ordering::Relaxed);
    if backend.is_null() {
        debug_assert!(false, "inline trip with no executing wasm backend");
        return;
    }
    let slot = pending_slot as usize;
    // SAFETY: the probe runs inside this bridge, so its slot box is still
    // on the bridge's `LoopAsmResources`.
    let owner = unsafe { pending_slot_at(slot) }
        .entry
        .borrow()
        .as_ref()
        .map(|item| item.owner.clone());
    let Some(owner) = owner else {
        return;
    };
    // The pointer is the backend whose `execute_token` is inside
    // `glue::execute` on this thread. Install only writes a cell and
    // replaces a module; it does not call the running function.
    // Membership is not re-checked: this call is still inside the bridge,
    // so the slot cannot have been freed since the read above.
    unsafe {
        (*backend).tripped_inlines.borrow_mut().push((owner, slot));
        (*backend).install_live_pending_inline(slot);
    }
}

/// Backend currently inside `glue::execute`. Not a loop table: one call,
/// cleared when that call returns. The wasm host is single-threaded.
static EXECUTING_WASM_BACKEND: std::sync::atomic::AtomicPtr<WasmBackend> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
struct ExecutingBackendGuard;

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl ExecutingBackendGuard {
    fn enter(backend: &WasmBackend) -> Self {
        EXECUTING_WASM_BACKEND.store(
            backend as *const WasmBackend as *mut WasmBackend,
            std::sync::atomic::Ordering::Relaxed,
        );
        Self
    }
}

impl Drop for ExecutingBackendGuard {
    fn drop(&mut self) {
        EXECUTING_WASM_BACKEND.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Record a deferred merge and describe the probe the bridge standing in for
/// it carries. The slot lives in `asm_resources`, dropped with a rejected
/// build and freed with the bridge's `asmmemmgr_blocks` once published.
fn register_pending_inline(
    asm_resources: &mut release::LoopAsmResources,
    owner: Arc<JitCellToken>,
    region: codegen::InlinedBridge,
    owner_module_bytes: u32,
    remap: Option<(u64, u32)>,
) -> codegen::InlineTripProbe {
    // A region placed outside the loop re-enters its prefix every iteration
    // (`jump_to_preamble`), so the owner-size term would postpone the merge
    // while those crossings keep accumulating. Header-resuming regions still
    // price the re-emission against owner bytes (`inline_trip_threshold_for`).
    let threshold = if region.outside_loop {
        INLINE_TRIP_THRESHOLD
    } else {
        inline_trip_threshold_for(owner_module_bytes)
    };
    let (pending_slot, counter_addr) = asm_resources.alloc_pending_inline(PendingInline {
        owner: Arc::downgrade(&owner),
        region,
        remap,
        retry_on_sibling: false,
    });
    codegen::InlineTripProbe {
        counter_addr,
        threshold,
        trip_fn_ptr: inline_trip_helper_slot() as i64,
        pending_slot: pending_slot as i64,
    }
}

/// Host entry point publishing [`INLINE_TRIP_HELPER_SLOT`] (called from
/// pyre-jit's `init_jit_hooks` with `wasm_jit_inline_trip as *const () as
/// usize`, which on wasm32 is the function's table index).
pub fn set_inline_trip_helper_slot(slot: u32) {
    INLINE_TRIP_HELPER_SLOT.store(slot as u64, Ordering::Relaxed);
}

/// Current trip-callback table slot (0 = unset).
fn inline_trip_helper_slot() -> u32 {
    INLINE_TRIP_HELPER_SLOT.load(Ordering::Relaxed) as u32
}

/// Host entry point publishing [`CA_DEOPT_HELPER_SLOT`] (called from pyre-jit's
/// `init_jit_hooks` with `wasm_ca_resume_deopt as *const () as usize`, which on
/// wasm32 is the function's table index).
pub fn set_ca_deopt_helper_slot(slot: u32) {
    CA_DEOPT_HELPER_SLOT.store(slot as u64, Ordering::Relaxed);
}

/// Current CA deopt-helper table slot (0 = unset).
pub fn ca_deopt_helper_slot() -> u32 {
    CA_DEOPT_HELPER_SLOT.load(Ordering::Relaxed) as u32
}

#[cfg(any(test, not(target_arch = "wasm32")))]
thread_local! {
    static TEST_RESIDUAL_TARGET_SIGS: std::cell::RefCell<HashMap<i64, i64>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Install a `jit_func_sig` encoding for host-side tests. Production reads
/// the function table directly and does not keep a map.
#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn set_test_residual_target_sig(addr: i64, encoded: i64) {
    TEST_RESIDUAL_TARGET_SIGS.with(|map| {
        map.borrow_mut().insert(addr, encoded);
    });
}

/// Drop every injected encoding.
#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn clear_test_residual_target_sigs() {
    TEST_RESIDUAL_TARGET_SIGS.with(|map| map.borrow_mut().clear());
}

/// Declared wasm type of table slot `addr`, or `None` when the slot is not a
/// function, the type is not a single result (`encode_func_sig` = 0:
/// multi-value, sret / aggregate), or the address is a host import outside
/// the guest table. No address-keyed cache: each lookup reads the table (or
/// the test injection).
pub fn residual_target_sig(addr: i64) -> Option<WasmSig> {
    decode_func_sig(query_residual_target_sig(addr))
}

fn query_residual_target_sig(addr: i64) -> i64 {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        if let Some(encoded) =
            TEST_RESIDUAL_TARGET_SIGS.with(|map| map.borrow().get(&addr).copied())
        {
            return encoded;
        }
    }
    #[cfg(all(target_arch = "wasm32", feature = "host-import"))]
    {
        return unsafe { jit_func_sig(addr as i32) };
    }
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    {
        return jit_func_sig_web::jit_func_sig(addr as i32);
    }
    #[cfg(not(all(target_arch = "wasm32", any(feature = "host-import", feature = "web"))))]
    {
        let _ = addr;
        0
    }
}

/// `None` (encoding 0) when the slot is not one wasm function type: multi-value
/// result, sret / aggregate, or a host import outside the guest table.
#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn jit_func_sig(slot: i32) -> i64;
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
mod jit_func_sig_web {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(raw_module = "./jit_glue.js")]
    unsafe extern "C" {
        pub fn jit_func_sig(slot: i32) -> i64;
    }
}

/// Install the host fallback for a residual callee wasm cannot name as one
/// function type, or whose address is a host import outside the guest table.
///
/// `descr.py` `CallDescr.create_call_stub` calls `lltype.Ptr(FUNC)` built
/// from the same FUNC as the calldescr. Published guests match that type and
/// take the word stub; this hook is the remaining case.
#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
pub fn install_residual_host_call() {
    majit_backend::call_stub::set_residual_host_call(Some(residual_host_call));
}

/// Call a residual target whose wasm type is not a single function type the
/// descr can name, or whose address is a host import outside the guest table.
///
/// `descr.py` `CallDescr.create_call_stub` calls `lltype.Ptr(FUNC)` built from
/// the same FUNC as the calldescr. Published guests match that type, and
/// `stub_matches_table` then returns `None` so the word stub runs. This hook
/// remains for a callee `encode_func_sig` cannot describe (multi-value result,
/// sret / aggregate such as `Vec` or `Result`, encoding 0) and for host
/// imports that are not in the guest table.
#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
pub fn residual_host_call(
    func_ptr: usize,
    args: &[i64],
    classes: &[majit_backend::call_stub::ArgClass],
    result: char,
    result_signed: bool,
    result_size: usize,
) -> Option<i64> {
    use codegen::{
        CALL_ARGS_OFS, CALL_FUNC_OFS, CALL_NARGS_OFS, CALL_RESULT_OFS, CALL_RESULT_SIZE_OFS,
        MAX_CALL_ARGS,
    };

    if stub_matches_table(func_ptr, classes, result) {
        return None;
    }
    assert!(
        args.len() <= MAX_CALL_ARGS,
        "residual_host_call: arity {} exceeds {MAX_CALL_ARGS}",
        args.len()
    );
    let base = RESIDUAL_CALL_SCRATCH.0.get() as *mut u8;
    unsafe {
        (base.add(CALL_FUNC_OFS as usize) as *mut i64).write_unaligned(func_ptr as i64);
        (base.add(CALL_NARGS_OFS as usize) as *mut i64).write_unaligned(args.len() as i64);
        for (i, &arg) in args.iter().enumerate() {
            (base.add(CALL_ARGS_OFS as usize + i * 8) as *mut i64).write_unaligned(arg);
        }
        (base.add(CALL_RESULT_SIZE_OFS as usize) as *mut i64).write_unaligned(result_size as i64);
        jit_call_host(base as u32);
        let raw = (base.add(CALL_RESULT_OFS as usize) as *const i64).read_unaligned();
        Some(widen_reflected_result(
            func_ptr,
            raw,
            result,
            result_signed,
            result_size,
        ))
    }
}

/// `callbuilder.py` `load_result` for a reflected wasm result. An i32 whose
/// descr is a signed int narrower than a word sign-extends. An f32 whose
/// descr is a float promotes. Singlefloat bits and pointer words stay.
pub fn widen_reflected_result(
    func_ptr: usize,
    value: i64,
    result: char,
    result_signed: bool,
    result_size: usize,
) -> i64 {
    use func_sig::FuncSigVal;
    let Some(sig) = residual_target_sig(func_ptr as i64) else {
        return value;
    };
    match (result, sig.result) {
        ('f', Some(FuncSigVal::F32)) => f64::from(f32::from_bits(value as u32)).to_bits() as i64,
        ('i', Some(FuncSigVal::I32)) if result_signed && result_size < 8 => match result_size {
            1 => value as i8 as i64,
            2 => value as i16 as i64,
            _ => value as i32 as i64,
        },
        _ => value,
    }
}

/// True when the guest table type is the descr's FUNC (`'i'`/`'r'` → i64,
/// `'f'` → f64, `'S'` → f32, `'v'` → no result). A match falls through to the
/// word stub. `None` from `residual_target_sig` (no table type: multi-value,
/// sret/aggregate, or a host import) is not a match.
#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
fn stub_matches_table(
    func_ptr: usize,
    classes: &[majit_backend::call_stub::ArgClass],
    result: char,
) -> bool {
    use func_sig::FuncSigVal;
    use majit_backend::call_stub::ArgClass;

    if !majit_backend::call_stub::call_stub_arm_exists(classes) {
        return false;
    }
    let Some(sig) = residual_target_sig(func_ptr as i64) else {
        return false;
    };
    if sig.params.len() != classes.len() {
        return false;
    }
    for (param, class) in sig.params.iter().zip(classes) {
        let expect = match class {
            ArgClass::Int => FuncSigVal::I64,
            ArgClass::Float => FuncSigVal::F64,
            ArgClass::Single => FuncSigVal::F32,
        };
        if *param != expect {
            return false;
        }
    }
    // `'L'` returns `i64` and `bh_call_f` would call the `f64` stub. The host
    // returns those bits and the float bank stores them.
    let expect_result = match result {
        'v' => None,
        'r' | 'i' => Some(FuncSigVal::I64),
        'f' => Some(FuncSigVal::F64),
        'S' => Some(FuncSigVal::F32),
        _ => return false,
    };
    sig.result == expect_result
}

/// Host import used when the callee has no single wasm function type or lives
/// outside the guest table.
#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
#[link(wasm_import_module = "majit_host")]
unsafe extern "C" {
    fn jit_call_host(frame_ptr: u32);
}

#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
struct ResidualCallScratch(core::cell::UnsafeCell<[u8; codegen::MIN_FRAME_BYTES]>);

// A wasm module instance is single-threaded. Nested calls synchronously consume
// their arguments before reusing the buffer and write their result before the
// outer call reads its own result.
#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
unsafe impl Sync for ResidualCallScratch {}

#[cfg(all(target_arch = "wasm32", feature = "host-import"))]
static RESIDUAL_CALL_SCRATCH: ResidualCallScratch =
    ResidualCallScratch(core::cell::UnsafeCell::new([0; codegen::MIN_FRAME_BYTES]));

/// Configure the dormant terminal-decline regression hook.
pub fn set_force_ca_terminal_decline(selector: u64) {
    FORCE_CA_TERMINAL_DECLINE.store(selector, Ordering::Relaxed);
}

/// A legacy pool-indexed const (`ConstInt(u32)` etc.) reached the wasm backend
/// without a value in the constants pool. `set_constants_pool` runs before
/// `assemble`, so every legitimate legacy const is already present; an arg
/// landing here means the optimizer producer failed to seed it. RPython
/// `ConstInt.value` (history.py) is always present, so never register a
/// placeholder `0` — that would emit the constant as zero. Panic at the parity
/// hole, matching the dynasm/cranelift backends.
fn missing_legacy_const(arg: majit_ir::OpRef) -> ! {
    panic!(
        "wasm collect_constants_from_ops: legacy pool-indexed const OpRef \
         (raw={}) is absent from the constants pool — the optimizer producer \
         must seed it (or mint an inline Const) instead of registering 0.",
        arg.raw()
    );
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmBackend {
    pub fn new() -> Self {
        WasmBackend {
            cpu_tracker: std::sync::Arc::new(majit_backend::CpuTotalTracker::default()),
            asm_memory_stats: std::sync::Arc::new(majit_backend::AsmMemoryManagerStats::default()),
            asm_memory_blocks: std::cell::RefCell::new(Vec::new()),
            trace_counter: 0,
            next_header_pc: 0,
            constants: indexmap::IndexMap::new(),
            vtable_offset: None,
            w_class_offset: None,
            gc_box: None,
            exit_cells: std::sync::Arc::new(failguard::CpuExitCells::new()),
            tripped_inlines: RefCell::new(Vec::new()),
        }
    }

    /// Active vtable_offset for wasm codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    // `set_constants_pool`, `set_next_trace_id`, and `set_next_header_pc`
    // are provided via the `Backend` trait impl below.

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Mirrors `CraneliftBackend::set_gc_allocator`: stores the box in
    /// the wasm thread-local seam and publishes the same five
    /// `ActiveGcGuardHooks` so the backend-agnostic optimizer /
    /// blackhole executor reach the live allocator without taking a
    /// wasm dependency.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        gc.freeze_types();
        // Drop the previous `ActiveGcBox` first. Assignment would install
        // the replacement and then run the old guard's `Drop`, which
        // `gc_box::clear`s the just-installed allocator.
        drop(self.gc_box.take());
        self.gc_box = Some(install_gc_box(gc));
    }

    /// No-op: present for API parity with the dynasm backend so
    /// backend-agnostic consumers can call it uniformly. The wasm `New`
    /// allocation path is out of scope for the GC-routed-New opt-in.
    pub fn set_new_via_gc(&mut self, _enabled: bool) {}

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr.
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_wasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)).flatten()
    }

    /// Resolve the vtable integer carried by GuardClass /
    /// GuardNonnullClass / GuardSubclass `arg(1)`.
    ///
    /// RPython represents these class operands as `ConstInt` vtable
    /// addresses: `model.py cls_of_box()` returns
    /// `ConstInt(ptr2int(obj.typeptr))`, `virtualstate.py:748` builds
    /// `ConstInt(descr.get_vtable())`, and backends read
    /// `op.getarg(1).getint()` (aarch64/regalloc.py:829). Inline ConstInt
    /// carries the value directly (history.py `ConstInt.value`).
    fn const_class_vtable(&self, arg: majit_ir::OpRef) -> Option<i64> {
        arg.const_int_value()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. wasm codegen runs without a
    /// borrow of `self`, so we materialize the resolver as a HashMap.
    /// Only GuardClass / GuardNonnullClass need this table — GuardGcType
    /// already carries an immediate typeid (assembler.py:1919-1922) and
    /// GUARD_IS_OBJECT / GUARD_SUBCLASS use a different lookup path.
    fn collect_classptr_typeid_table(&self, ops: &[Op]) -> HashMap<i64, u32> {
        let mut table = HashMap::new();
        if self.vtable_offset.is_some() {
            return table;
        }
        if with_wasm_active_gc(|_| ()).is_none() {
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.num_args() >= 2
                && let Some(classptr) = self.const_class_vtable(op.arg(1).to_opref())
                && let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize)
            {
                table.insert(classptr, tid);
            }
        }
        table
    }

    /// Pre-fetch `GuardGcTypeInfo` from the installed `gc_ll_descr`.
    ///
    /// Mirrors the `self.cpu.gc_ll_descr.get_translated_info_*` /
    /// `cpu.subclassrange_min_offset` lookups that RPython's
    /// `genop_guard_guard_is_object` (x86/assembler.py) and
    /// `genop_guard_guard_subclass` (x86/assembler.py) do at
    /// codegen time. The returned struct is handed to
    /// `codegen::build_wasm_module`; the codegen arms assert
    /// `supports_guard_gc_type` before reading any other field.
    ///
    /// Also pre-computes `(subclassrange_min, subclassrange_max)` for
    /// every constant classptr argument of a `GuardSubclass` op
    /// (assembler.py:1971-1974 reads these bounds at codegen time).
    fn collect_guard_gc_type_info(&self, ops: &[Op]) -> codegen::GuardGcTypeInfo {
        with_wasm_active_gc(|gc| {
            let mut info = codegen::GuardGcTypeInfo {
                supports_guard_gc_type: gc.supports_guard_gc_type(),
                ..codegen::GuardGcTypeInfo::default()
            };
            if !info.supports_guard_gc_type {
                return info;
            }
            // assembler.py:1934-1937: gc_ll_descr lookups.
            let (base, shift, sizeof_ti) = gc.get_translated_info_for_typeinfo();
            info.base_type_info = base;
            info.shift_by = shift;
            info.sizeof_ti = sizeof_ti;
            let (infobits_off, is_object_flag) = gc.get_translated_info_for_guard_is_object();
            info.infobits_offset = infobits_off;
            info.is_object_flag = is_object_flag;
            // assembler.py:1951: cpu.subclassrange_min_offset.
            info.subclassrange_min_offset = gc.subclassrange_min_offset();
            // assembler.py:1971-1974: (subclassrange_min, subclassrange_max)
            // for every constant GuardSubclass arg1.
            for op in ops {
                if op.opcode == majit_ir::OpCode::GuardSubclass
                    && op.num_args() >= 2
                    && let Some(classptr) = self.const_class_vtable(op.arg(1).to_opref())
                    && let Some(range) = gc.subclass_range(classptr as usize)
                {
                    info.subclass_ranges.insert(classptr, range);
                }
            }
            info
        })
        .unwrap_or_default()
    }

    /// Pull every reference constant out of `ops` into a per-loop `GcTable`
    /// and replace it with a `LoadFromGcTable` of its slot
    /// (`majit_gc::rewrite::remove_ref_constants`, rewrite.py).
    ///
    /// A `GcRef` baked as a code immediate is invisible to the moving
    /// collector: the first minor collection that promotes the referenced
    /// object out of the nursery leaves the immediate pointing into nursery
    /// space that is later reused or zeroed by `reset`. The table slot is a
    /// GC root the collector forwards in place, so the emitted load always
    /// reads the object at its current address. Returns `None` for a trace
    /// with no reference constant, leaving the module byte-identical.
    #[allow(dead_code)] // constptr-only subset; production uses `rewrite_ops_for_gc`
    fn intern_ref_constants(
        inputargs: &[InputArgRc],
        ops: Vec<Op>,
    ) -> (Vec<Op>, Option<Arc<majit_gc::GcTable>>) {
        let next_pos = codegen::next_value_pos(inputargs, &ops);
        let input_indices: Vec<u32> = inputargs.iter().map(|ia| ia.index).collect();
        let (ops, gcrefs) =
            majit_gc::rewrite::remove_ref_constants_for_inputs(&ops, next_pos, &input_indices);
        let table = (!gcrefs.is_empty()).then(|| majit_gc::GcTable::from_gcrefs(&gcrefs));
        (ops, table)
    }

    /// Run `rewrite.py` then intern the gcref table. Replaces
    /// [`Self::intern_ref_constants`] on the production compile path.
    fn rewrite_ops_for_gc(&mut self, ops: Vec<Op>) -> (Vec<Op>, Option<Arc<majit_gc::GcTable>>) {
        let (ops, new_constants, table) = crate::rewrite_ops_for_gc(ops, &self.constants);
        self.constants = new_constants;
        (ops, table)
    }
    /// `x86/assembler.py` `gcreftracers.append(tracer)` — keep the
    /// per-loop table alive for as long as the compiled trace that bakes its
    /// base address. `LIVE_GC_TABLES` holds only a `Weak`, so this strong
    /// reference is what keeps the slots rooted and forwardable.
    /// `assembler.py gcreftracers.append(tracer)`: pin the metainterp descr
    /// Arcs this compiled code's guards dereference onto the owning
    /// `CompiledLoopToken.asmmemmgr_gcreftracers` (`model.py`), the same
    /// `Vec<DescrRef>` tracer shape cranelift registers.  The frontend
    /// keeps no per-bridge record (`compile.py send_bridge_to_backend`), so
    /// this tracer is where the GC root walker reaches a bridge guard's
    /// `rd_consts`, and what keeps the descr alive for the token's lifetime.
    fn register_meta_descrs(token: &JitCellToken, descrs: &[Arc<WasmFailDescr>]) {
        if let Some(clt) = token.compiled_loop_token() {
            let meta: Vec<majit_ir::descr::DescrRef> =
                descrs.iter().filter_map(|d| d.meta_descr.clone()).collect();
            let tracer: Arc<dyn std::any::Any + Send + Sync> = Arc::new(meta);
            clt.asmmemmgr_gcreftracers.lock().push(tracer);
        }
    }

    fn register_gc_table(token: &JitCellToken, table: Arc<majit_gc::GcTable>) {
        if let Some(clt) = token.compiled_loop_token() {
            let tracer: Arc<dyn std::any::Any + Send + Sync> = table.clone();
            clt.asmmemmgr_gcreftracers.lock().push(tracer);
        }
        // `gcreftracer.py` `llop.gc_writebarrier(tr)`: the table enters
        // this MiniMark's remembered set for one minor.
        let _ = with_wasm_active_gc_mut(|gc| gc.remember_gc_table(&table));
    }

    /// Validate that every constant OpRef appearing as an arg is resolvable.
    ///
    /// Inline-Const variants (`ConstInt`/`ConstFloat`/
    /// `ConstPtr`) carry `.value` on the OpRef itself (history.py:
    /// 227/268/314), so they need no `self.constants` side-table entry and
    /// are skipped. A legacy idx-keyed `ConstInt(u32)` / `ConstFloat(u32)` /
    /// `ConstPtr(u32)` must have been seeded by `set_constants_pool`; one that
    /// is missing is a producer gap and panics rather than defaulting to 0.
    fn collect_constants_from_ops(&mut self, ops: &[Op]) {
        for op in ops {
            for arg in op.getarglist().iter() {
                let arg = arg.to_opref();
                if arg.is_constant()
                    && arg.inline_const_bits().is_none()
                    && !self.constants.contains_key(&arg.raw())
                {
                    missing_legacy_const(arg);
                }
            }
            if let Some(fail_args) = op.getfailargs() {
                for arg in fail_args.iter() {
                    let arg = arg.to_opref();
                    if arg.is_constant()
                        && arg.inline_const_bits().is_none()
                        && !self.constants.contains_key(&arg.raw())
                    {
                        missing_legacy_const(arg);
                    }
                }
            }
        }
    }

    /// Take the merges whose bridges have tripped since the last call, for a
    /// caller with no compiled trace left on the stack.
    pub fn take_tripped_inlines(&self) -> Vec<(Weak<JitCellToken>, usize)> {
        std::mem::take(&mut *self.tripped_inlines.borrow_mut())
    }

    /// Merge a deferred region into its owner, for a bridge that has been
    /// entered [`INLINE_TRIP_THRESHOLD`] times.
    ///
    /// The trip itself only queued the slot ([`record_inline_trip`]); this runs
    /// from the host once the trace has returned, and also from the probe while
    /// the bridge is still on the stack.
    ///
    /// A candidate that no longer qualifies — an invalidated owner, a loop that
    /// has since taken a region for the same guard — is dropped rather than
    /// retried: the bridge is already installed and correct, so the only thing
    /// lost is the merge.
    ///
    /// `tripped` is `(owner weak, slot address)`. The address is dereferenced
    /// only after the owner upgrades and the address is still one of that
    /// owner's `pending_inlines`.
    pub fn install_pending_inline(&self, tripped: (Weak<JitCellToken>, usize)) {
        let (owner_weak, slot_addr) = tripped;
        let Some(owner) = owner_weak.upgrade() else {
            return;
        };
        if !pending_slot_is_member(&owner, slot_addr) {
            return;
        }
        self.install_live_pending_inline(slot_addr);
    }

    /// Install `slot_addr`, which the caller has already shown is live.
    fn install_live_pending_inline(&self, slot_addr: usize) {
        // SAFETY: `record_inline_trip` calls this while the bridge is still
        // on the stack. `install_pending_inline` calls it only after
        // `pending_slot_is_member`.
        let owner = {
            let borrowed = unsafe { pending_slot_at(slot_addr) }.entry.borrow();
            borrowed.as_ref().and_then(|item| item.owner())
        };
        let Some(owner) = owner else {
            // SAFETY: same live-slot guarantee as the read above.
            unsafe { pending_slot_at(slot_addr) }
                .entry
                .borrow_mut()
                .take();
            return;
        };
        // `sweep_dead_pending` only ran from here. A dead owner is freed with
        // its blocks; an invalidated one still has them, so restore its cells.
        if owner.is_invalidated() {
            discard_invalidated_owner_inlines(&owner);
            return;
        }
        // SAFETY: same live-slot guarantee as the owner read above.
        let Some(pending) = unsafe { pending_slot_at(slot_addr) }
            .entry
            .borrow_mut()
            .take()
        else {
            return;
        };
        // The driver has already classified the exit, and no compiled frame
        // remains. Re-emission swaps the owner's module here.
        diag_bump(55);
        let mut work = vec![(slot_addr, pending.region, pending.remap)];
        // Other trips for this owner would each re-emit the whole module.
        // Fold them into this rebuild so one Cranelift compile covers them.
        // Children compiled as `not_direct` wait with a remap.
        // An `uninitialized_label` trip can also fire before the sibling
        // peel that publishes its JUMP target; fold those too so the
        // one-shot probe is not the only retry.
        // Block order is attach order (`push_resources` at the end of each
        // `compile_bridge`), the same order the old ids were allocated in.
        let sibling_addrs: Vec<usize> = owner_pending_slot_addrs(&owner)
            .into_iter()
            .filter(|&addr| {
                // SAFETY: `addr` was just read out of this owner's `pending_inlines`.
                unsafe { pending_slot_at(addr) }
                    .entry
                    .borrow()
                    .as_ref()
                    .is_some_and(|item| {
                        item.same_owner(&owner) && (item.remap.is_some() || item.retry_on_sibling)
                    })
            })
            .collect();
        for addr in sibling_addrs {
            // SAFETY: `addr` was just read out of this owner's `pending_inlines`.
            if let Some(item) = unsafe { pending_slot_at(addr) }.entry.borrow_mut().take() {
                work.push((addr, item.region, item.remap));
            }
        }
        let queued = std::mem::take(&mut *self.tripped_inlines.borrow_mut());
        let mut keep = Vec::new();
        let mut extra_addrs = Vec::new();
        for (weak, addr) in queued {
            let Some(queued_owner) = weak.upgrade() else {
                continue;
            };
            if !pending_slot_is_member(&queued_owner, addr) {
                continue;
            }
            // SAFETY: `pending_slot_is_member` just found `addr` in this owner's blocks.
            let same = unsafe { pending_slot_at(addr) }
                .entry
                .borrow()
                .as_ref()
                .is_some_and(|item| item.same_owner(&owner));
            if same {
                extra_addrs.push(addr);
            } else {
                keep.push((weak, addr));
            }
        }
        self.tripped_inlines.borrow_mut().extend(keep);
        for addr in extra_addrs {
            diag_bump(55);
            // SAFETY: `addr` passed `pending_slot_is_member` above, and this
            // install has not freed the owner's blocks.
            if let Some(item) = unsafe { pending_slot_at(addr) }.entry.borrow_mut().take() {
                // Remapped children still name a parent-local fail index;
                // the owner's descr array does not hold that guard.
                work.push((addr, item.region, item.remap));
            }
        }
        let fail_indices: Vec<u32> = work
            .iter()
            .filter(|(_, _, remap)| remap.is_none())
            .map(|(_, r, _)| r.source_fail_index)
            .collect();
        // The compiled probe still names the slot it was registered under.
        // Leftover remaps must go back into that same slot; a fresh one
        // would leave the already-emitted trip calling a hole.
        let remap_pending_ids: HashMap<(u64, u32), usize> = work
            .iter()
            .filter_map(|(addr, _, remap)| remap.map(|key| (key, *addr)))
            .collect();
        let fail_pending_ids: HashMap<u32, usize> = work
            .iter()
            .filter(|(_, _, remap)| remap.is_none())
            .map(|(addr, region, _)| (region.source_fail_index, *addr))
            .collect();
        let trigger_id = slot_addr;
        let (mut leftover, mut terminal) = self.install_inline_region_batch(
            &owner,
            work.into_iter().map(|(_, r, remap)| (r, remap)).collect(),
        );
        let put_back = |addr: usize, region: codegen::InlinedBridge, remap: Option<(u64, u32)>| {
            // SAFETY: `addr` is a slot box still stored in this owner's
            // `pending_inlines`. This install takes the `Option` and does not
            // drop the box; re-emission only pushes new blocks.
            *unsafe { pending_slot_at(addr) }.entry.borrow_mut() = Some(PendingInline {
                owner: Arc::downgrade(&owner),
                region,
                remap,
                retry_on_sibling: remap.is_none(),
            });
        };
        // Optional remaps / leftover labels must not make the newly
        // tripped region fail the whole rebuild.
        if terminal && leftover.len() > 1 && !owner.is_invalidated() {
            let (trigger_left, optional_left): (Vec<_>, Vec<_>) =
                leftover
                    .into_iter()
                    .partition(|(region, remap)| match remap {
                        Some(key) => remap_pending_ids.get(key) == Some(&trigger_id),
                        None => {
                            fail_pending_ids.get(&region.source_fail_index) == Some(&trigger_id)
                        }
                    });
            if !trigger_left.is_empty() {
                (leftover, terminal) = self.install_inline_region_batch(&owner, trigger_left);
                if !terminal && !owner.is_invalidated() {
                    for (region, remap) in optional_left {
                        let addr = match remap {
                            Some(key) => remap_pending_ids.get(&key).copied(),
                            None => fail_pending_ids.get(&region.source_fail_index).copied(),
                        };
                        let Some(addr) = addr else {
                            continue;
                        };
                        put_back(addr, region, remap);
                    }
                }
            } else {
                leftover = optional_left;
            }
        }
        if leftover.iter().any(|(_, remap)| remap.is_none()) {
            for source_fail_index in fail_indices {
                Self::restore_dispatch_cell(&owner, source_fail_index);
            }
        }
        if owner.is_invalidated() || terminal {
            return;
        }
        // Leftovers whose parent or sibling is not in the owner yet stay
        // pending so a later install can pick them up. The compiled probe
        // still names this slot.
        for (region, remap) in leftover {
            let addr = match remap {
                Some(key) => remap_pending_ids.get(&key).copied(),
                None => fail_pending_ids.get(&region.source_fail_index).copied(),
            };
            let Some(addr) = addr else {
                continue;
            };
            put_back(addr, region, remap);
        }
    }

    fn cell_for_meta(
        descrs: &[std::sync::Arc<WasmFailDescr>],
        meta: &majit_ir::DescrRef,
    ) -> Option<u32> {
        descrs.iter().find_map(|descr| {
            descr
                .meta_descr
                .as_ref()
                .is_some_and(|existing| std::sync::Arc::ptr_eq(existing, meta))
                .then_some(descr.bridge_cell)
                .filter(|&addr| addr != 0)
        })
    }

    /// Cell address of each guard and `Finish`, in collect order.
    ///
    /// `region_trace` is set for an inlined bridge. `owner_base` is the loop's
    /// original cell array; only the owner's own exits are indexed into it.
    /// A failing guard that still has no cell gets one here. That is the
    /// descr's cell from this compile onward.
    fn reemit_guard_cells(
        source_loop: &CompiledWasmLoop,
        ops: &[Op],
        region_trace: Option<u64>,
        owner_base: Option<u32>,
        owner_limit: usize,
        fresh: &mut Vec<Box<[u32]>>,
    ) -> Vec<u32> {
        let descrs = source_loop.fail_descrs.borrow();
        let mut ordinal = 0u32;
        let mut addrs = Vec::new();
        for op in ops
            .iter()
            .filter(|op| op.opcode.is_guard() || op.opcode == majit_ir::OpCode::Finish)
        {
            let meta = op.getdescr();
            let mut addr = meta
                .as_ref()
                .and_then(|meta| Self::cell_for_meta(&descrs, meta))
                .unwrap_or(0);
            if addr == 0 {
                if let Some(trace_id) = region_trace {
                    if let Some(found) = descrs.iter().find_map(|descr| {
                        (descr.trace_id == trace_id
                            && descr.fail_index == ordinal
                            && descr.bridge_cell != 0)
                            .then_some(descr.bridge_cell)
                    }) {
                        addr = found;
                    }
                } else if let Some(base) = owner_base.filter(|&base| base != 0) {
                    if (ordinal as usize) < owner_limit {
                        addr = base + ordinal * std::mem::size_of::<u32>() as u32;
                    }
                }
            }
            let finish = meta
                .as_ref()
                .and_then(|meta| meta.as_fail_descr())
                .is_some_and(|fd| fd.is_finish());
            if addr == 0 && op.opcode.is_guard() && !finish {
                let (cell, owner) = codegen::alloc_bridge_cells(1);
                if let Some(owner) = owner {
                    fresh.push(owner);
                }
                addr = cell;
            }
            ordinal += 1;
            addrs.push(addr);
        }
        addrs
    }

    fn owner_guard_descr(
        source_loop: &CompiledWasmLoop,
        fail_index: u32,
    ) -> Option<Arc<WasmFailDescr>> {
        source_loop.fail_descrs.borrow().iter().find_map(|descr| {
            (descr.trace_id == source_loop.trace_id && descr.fail_index == fail_index)
                .then(|| Arc::clone(descr))
        })
    }

    /// Put each guard's stable cell address back on `adr_jump_offset` so a
    /// rebuild bakes that address. `compile_bridge` clears the slot after
    /// patching; `WasmFailDescr.bridge_cell` keeps it.
    fn restore_guard_cell_offsets(source_loop: &CompiledWasmLoop, ops: &[Op]) {
        let descrs = source_loop.fail_descrs.borrow();
        for op in ops.iter().filter(|op| op.opcode.is_guard()) {
            let Some(meta) = op.getdescr() else {
                continue;
            };
            let Some(fd) = meta.as_fail_descr() else {
                continue;
            };
            if fd.is_finish() || fd.adr_jump_offset() != 0 {
                continue;
            }
            let Some(addr) = Self::cell_for_meta(&descrs, &meta).filter(|&addr| addr != 0) else {
                continue;
            };
            let stamp = std::panic::AssertUnwindSafe(|| fd.set_adr_jump_offset(addr as usize));
            let _ = std::panic::catch_unwind(stamp);
        }
    }

    /// Re-point a guard's dispatch cell at the bridge slot stored on its descr.
    fn restore_dispatch_cell(owner: &JitCellToken, source_fail_index: u32) {
        let Some(source_loop) = owner
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return;
        };
        let Some(descr) = Self::owner_guard_descr(source_loop, source_fail_index) else {
            return;
        };
        crate::failguard::write_guard_cell(
            descr.bridge_cell,
            descr.bridge_slot.load(std::sync::atomic::Ordering::Relaxed),
        );
    }

    /// Rebuild `owner` with `region` merged into it. `false` leaves the owner
    /// exactly as it was, for a caller that still has an out-of-line bridge to
    /// fall back on.
    fn install_inline_region(&self, owner: &JitCellToken, region: codegen::InlinedBridge) -> bool {
        self.install_inline_region_batch(owner, vec![(region, None)])
            .0
            .is_empty()
    }

    fn install_inline_region_batch(
        &self,
        owner: &JitCellToken,
        regions: Vec<(codegen::InlinedBridge, Option<(u64, u32)>)>,
    ) -> (Vec<(codegen::InlinedBridge, Option<(u64, u32)>)>, bool) {
        if regions.is_empty() {
            return (Vec::new(), false);
        }
        if owner.is_invalidated() {
            diag_bump(50);
            return (regions, true);
        }
        let Some(source_loop) = owner
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return (regions, true);
        };
        let Some(mut candidate) = source_loop.reemit.borrow().as_ref().cloned() else {
            diag_bump(35);
            return (regions, true);
        };
        let mut attached = 0usize;
        let mut attached_pairs = Vec::new();
        let mut leftover = regions;
        // A compile-time `uninitialized_label` may become legal after a
        // sibling peel is attached. Retry leftovers against the growing
        // candidate; still-doomed regions stay out of line.
        loop {
            let mut progressed = false;
            let mut still = Vec::new();
            for (mut region, remap) in leftover {
                if let Some((parent_trace_id, parent_fail_index)) = remap {
                    let Some(idx) =
                        merged_region_fail_index(&candidate, parent_trace_id, parent_fail_index)
                    else {
                        still.push((region, remap));
                        continue;
                    };
                    region.source_fail_index = idx;
                }
                let source_fail_index = region.source_fail_index;
                if candidate
                    .inlined_bridges
                    .iter()
                    .any(|r| r.source_fail_index == source_fail_index)
                {
                    diag_bump(36);
                    continue;
                }
                // The out-of-line bridge grows the frame to its JUMP target's
                // depth in its prologue (`assembler.py` `_assemble` sizes
                // `_check_frame_depth` by the target's `jfi_frame_depth`). A
                // merged region has no prologue of its own, so a region storing
                // at another layout's offsets past the owner's live depth
                // stays out of line.
                let owner_live_items = candidate.frame.ca_frame_depth();
                if region.external_jump.as_ref().is_some_and(|ext| {
                    ext.frame != candidate.frame && ext.frame.signed_item_count() > owner_live_items
                }) {
                    still.push((region, remap));
                    continue;
                }
                region.outside_loop = region.outside_loop
                    || codegen::source_guard_precedes_loop_label(&candidate.ops, source_fail_index)
                    || candidate.inlined_bridges.iter().any(|r| r.outside_loop);
                if region.outside_loop {
                    // A foreign JUMP has no owner LABEL to have crossed;
                    // the capture-loader check applies only to in-module
                    // preamble peels.
                    if region.external_jump.is_none() {
                        let mut owner_ops = candidate.ops.clone();
                        for existing in &candidate.inlined_bridges {
                            owner_ops.extend(existing.ops.iter().cloned());
                        }
                        if !codegen::outside_region_labels_initialized(
                            &owner_ops,
                            source_fail_index,
                            &region.ops,
                        ) {
                            still.push((region, remap));
                            continue;
                        }
                    }
                    diag_bump(52);
                }
                candidate.inlined_bridges.push(region.clone());
                attached_pairs.push((region, remap));
                attached += 1;
                progressed = true;
            }
            leftover = still;
            if !progressed || leftover.is_empty() {
                break;
            }
        }
        if attached == 0 {
            return (leftover, false);
        }
        let mut merged_ops = candidate.ops.clone();
        for region in &candidate.inlined_bridges {
            merged_ops.extend(region.ops.iter().cloned());
        }
        // `candidate.constants` keeps the pool of the compile that recorded
        // these ops -- the owner's. The pool is keyed by value position, and
        // the merge rebases every region off the owner's ids, so a region's
        // own entries reach the build through `InlinedBridge::constants` and
        // are replayed at the rebase offset. Assigning the merging bridge's
        // pool here instead would leave the owner's window described by
        // another trace's keys: an owner-only folded value loses its entry,
        // and one the bridge happens to number the same way is seeded with the
        // wrong bits.
        candidate.classptr_to_typeid = self.collect_classptr_typeid_table(&merged_ops);
        candidate.guard_gc_type_info = self.collect_guard_gc_type_info(&merged_ops);
        candidate.nursery = nursery_alloc_params(&merged_ops);
        // The merged regions supersede their own dispatch cells. Remove
        // them before reemit so the fresh array cannot replay a contradictory
        // slot — the bridge on the stack right now finishes its pass either
        // way, and nothing enters it again.
        //
        // A remapped child compiled as `not_direct` stores a merged-stream
        // ordinal. That index is in the live owner array only after its
        // parent region is already installed. The same-batch case — parent
        // and child folded into this rebuild — lands past
        // `num_guard_cells`, which is the live array. `register_pending_inline`
        // already refuses to aim the trip probe at an owner cell then;
        // writing one here would store past the array into the guest heap.
        let live_cell_count = source_loop.num_guard_cells.get();
        let attached_fail_indices: Vec<u32> = candidate.inlined_bridges
            [candidate.inlined_bridges.len() - attached..]
            .iter()
            .map(|region| region.source_fail_index)
            .collect();
        let mut zeroed_cells = Vec::new();
        for &source_fail_index in &attached_fail_indices {
            if (source_fail_index as usize) >= live_cell_count {
                continue;
            }
            let Some(descr) = Self::owner_guard_descr(source_loop, source_fail_index) else {
                continue;
            };
            if descr.bridge_cell == 0 {
                continue;
            }
            zeroed_cells.push(Arc::clone(&descr));
            crate::failguard::write_guard_cell(descr.bridge_cell, 0);
        }
        // Eligibility IS the emission: `reemit_loop` runs the same
        // `build_wasm_module` over the same candidate, and nothing it does
        // before that call mutates state a failure would have to unwind — it
        // reads the fail-index base and allocates a cell array that is dropped
        // on the error path. So install directly and let the build answer,
        // instead of asking it once as a trial and once for real.
        let old_inputs = source_loop.reemit.replace(Some(candidate));
        match self.reemit_loop(owner) {
            Ok(()) => {
                diag_bump(31);
                for _ in 0..attached {
                    diag_bump(32);
                }
                // The guard cell stays zero so the inlined region is not
                // also dispatched. Leave the label descr's target in place:
                // inbound JUMPs still enter the old module.
                return (leftover, false);
            }
            Err(error) => {
                source_loop.reemit.replace(old_inputs);
                for descr in zeroed_cells {
                    crate::failguard::write_guard_cell(
                        descr.bridge_cell,
                        descr.bridge_slot.load(std::sync::atomic::Ordering::Relaxed),
                    );
                }
                record_inline_trial_error(&error);
                classify_inline_install_error(&error);
                leftover.splice(0..0, attached_pairs);
            }
        }
        (leftover, true)
    }

    /// Rebuild a loop module and install it into its original shared-table
    /// slot. The retained inputs are post-intern, so this does not allocate a
    /// second GC reference table or change any reference-constant immediate.
    #[allow(unreachable_code, unused_variables)]
    pub fn reemit_loop(&self, token: &JitCellToken) -> Result<(), BackendError> {
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            .ok_or_else(|| {
                BackendError::Unsupported("wasm backend: no compiled loop to re-emit".into())
            })?;
        let Some(mut inputs) = compiled.reemit.borrow().as_ref().cloned() else {
            return Err(BackendError::Unsupported(
                "wasm backend: entry bridge is not re-emittable".into(),
            ));
        };
        let old_handle = compiled.eager_func_handle();
        if old_handle == 0 {
            return Err(BackendError::Unsupported(
                "wasm backend: unmaterialized loop is not re-emittable".into(),
            ));
        }

        // `guard_exit_count` walks the whole op list it is handed, so each
        // count is taken once here: the exit loop below needs the per-region
        // counts for every one of its exits, and re-deriving them there would
        // walk every appended region once per guard.
        let own_guard_count = codegen::guard_exit_count(&inputs.inputargs, &inputs.ops);
        let region_guard_counts: Vec<usize> = inputs
            .inlined_bridges
            .iter()
            .map(|region| codegen::guard_exit_count(&region.inputargs, &region.ops))
            .collect();
        let merged_guard_count = own_guard_count + region_guard_counts.iter().sum::<usize>();
        if inputs.bridge_param_dispatch && !bridge_param_dispatch_profitable(merged_guard_count) {
            // compile_bridge has already published functions with the source
            // guards' parameter ABI. Flipping this flag would call those
            // functions with the frame-only type and trap. Retain the existing
            // owner and out-of-line bridge instead of growing past the bound.
            return Err(BackendError::Unsupported(
                "wasm backend: merged owner exceeds parameter-bridge guard limit".into(),
            ));
        }
        inputs.fail_index_base = 0;
        // Every guard the rebuilt module can fail through names the cell its
        // descr already owns. A guard that was inlined before it had a module
        // of its own has no cell yet; allocate that cell now, before emit, so
        // the descr and the module share it.
        Self::restore_guard_cell_offsets(compiled, &inputs.ops);
        let mut fresh_cells = Vec::new();
        let mut guard_cell_addrs = Self::reemit_guard_cells(
            compiled,
            &inputs.ops,
            None,
            Some(compiled.bridge_cells_base.get()),
            own_guard_count,
            &mut fresh_cells,
        );
        for region in &inputs.inlined_bridges {
            Self::restore_guard_cell_offsets(compiled, &region.ops);
            guard_cell_addrs.extend(Self::reemit_guard_cells(
                compiled,
                &region.ops,
                Some(region.trace_id),
                None,
                0,
                &mut fresh_cells,
            ));
        }
        inputs.guard_cell_addrs = guard_cell_addrs;
        inputs.bridge_cells_base = 0;
        let _fresh_cells = fresh_cells;
        inputs.ca.compute_home_gcmap = true;
        inputs.ca.home_gcmap_has_prior = true;
        // Do not null grown LABEL homes on every keyed entry. A later
        // loop-closing bridge writes those captures and tail-calls back;
        // zeroing them here would restore nulls. Pre-growth owner
        // attachments are dropped below when the LABEL tail grows.
        // Key-0 still clears the full used-label range.
        inputs.ca.home_gcmap_min_ordinary = compiled.num_ref_homes.get();
        inputs.ca.home_gcmap_min_labels = compiled.used_label_homes.get();
        let mut asm_resources = release::LoopAsmResources::default();
        asm_resources.exit_cells = Some(std::sync::Arc::clone(&self.exit_cells));
        asm_resources.bridge_cells.extend(_fresh_cells);
        inputs.ca.exit_table_base = asm_resources.alloc_exit_table(merged_guard_count) as u32;
        inputs.ca.gcmap_sink = &mut asm_resources as *mut _ as usize;
        // The module still running has the back-edge check. This
        // replacement is what that check tail-calls, so its jump is a
        // plain `br`, matching the jump left by `patch_jump_for_descr`.
        // A later replace is entered by the bridge's closing
        // `return_call_indirect` of this same slot.
        inputs.ca.resume_entry_addr = 0;
        inputs.ca.resume_generation = 0;
        let resume_generation = {
            #[cfg(target_arch = "wasm32")]
            {
                compiled
                    .resume_entry
                    .generation
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .wrapping_add(1)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                0u32
            }
        };
        // `assemble_bridge` raises the loop's `frame_info` depth to what its
        // bridges need, and every bridge prologue grows a shorter live frame
        // (`_check_frame_depth`). A merged region is that bridge inside the
        // owner, so the owner takes the region's depth: extend the layout by
        // the shortage (the prefix, and so every published offset, is kept)
        // and grow the frame at entry. The replacement is tail-called with
        // the live jitframe, so the prologue always checks.
        inputs.ca.realloc_fn_ptr = wasm_realloc_frame as *const () as usize as i64;
        let (wasm_bytes, guard_exits, merged_ref_homes, merged_labels) = loop {
            let mut shortage = None;
            match codegen::build_wasm_module_reporting_shortage(&inputs, &mut shortage) {
                Ok(built) => break built,
                Err(error) => {
                    let frame = inputs.frame;
                    let grown = match shortage.map(|s| (s.kind, s.needed)) {
                        Some((FrameShortageKind::FrameValueSlots, needed)) => {
                            frame.extend(needed, frame.addressable_ordinary_homes())
                        }
                        Some((FrameShortageKind::OrdinaryRefHomes, needed)) => {
                            frame.extend(frame.value_slots, needed)
                        }
                        _ => frame,
                    };
                    if grown == frame {
                        return Err(error);
                    }
                    inputs.frame = grown;
                    inputs.ca.realloc_fn_ptr = wasm_realloc_frame as *const () as usize as i64;
                }
            }
        };
        let code_size = wasm_bytes.len();
        let descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .enumerate()
            .map(|(index, g)| {
                let mut region_start = own_guard_count;
                let trace_id = inputs
                    .inlined_bridges
                    .iter()
                    .zip(&region_guard_counts)
                    .find_map(|(region, &count)| {
                        let contains = (region_start..region_start + count).contains(&index);
                        region_start += count;
                        contains.then_some(region.trace_id)
                    })
                    .unwrap_or(compiled.trace_id);
                let previous = compiled.fail_descrs.borrow().iter().find_map(|descr| {
                    descr
                        .meta_descr
                        .as_ref()
                        .zip(g.meta_descr.as_ref())
                        .is_some_and(|(existing, meta)| std::sync::Arc::ptr_eq(existing, meta))
                        .then(|| Arc::clone(descr))
                });
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    fail_locs: g.fail_locs.clone(),
                    is_finish: g.is_finish,
                    force_args_offset: inputs.frame.force_slot_base as u32,
                    force_tail_base: force_tail_fields(inputs.frame).0,
                    force_prefix_slots: force_tail_fields(inputs.frame).1,
                    value_tail_index: force_tail_fields(inputs.frame).2,
                    force_gcmap_ptr: g.exit_gcmap_ptr,
                    bridge_cell: g.bridge_cell,
                    fail_arg_advanced: previous
                        .as_ref()
                        .map(|descr| descr.fail_arg_advanced.clone())
                        .filter(|advanced| !advanced.is_empty())
                        .unwrap_or_else(|| {
                            let live = codegen::live_fail_arg_count(
                                g.meta_descr.as_ref(),
                                g.fail_arg_refs.len(),
                            );
                            vec![false; live]
                        }),
                    trace_ref_homes: previous
                        .as_ref()
                        .map(|descr| descr.trace_ref_homes)
                        .unwrap_or(0)
                        .max(compiled.num_ref_homes.get()),
                    trace_label_homes: previous
                        .as_ref()
                        .map(|descr| descr.trace_label_homes)
                        .unwrap_or(0)
                        .max(compiled.used_label_homes.get()),
                    param_dispatch: previous
                        .as_ref()
                        .map(|descr| descr.param_dispatch)
                        .unwrap_or(inputs.bridge_param_dispatch),
                    bridge_slot: std::sync::atomic::AtomicU32::new(
                        previous
                            .as_ref()
                            .map(|descr| {
                                descr.bridge_slot.load(std::sync::atomic::Ordering::Relaxed)
                            })
                            .unwrap_or(0),
                    ),
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();

        // When the LABEL-capture tail grows, do not replace the live
        // handle in place. Cross-loop and entry bridges on other tokens
        // already baked `return_call_indirect` against that slot; a
        // replacement would publish the wider map and restore the new
        // captures from uninitialized words. Install the grown module
        // at a new slot. Already-baked inbound JUMPs keep the old
        // module; `stamp_and_publish_label_targets` and CA dispatch
        // below name the new slot for later compiles.
        // `assembler.py` `patch_jump_for_descr` rewrites those jumps
        // in place; wasm cannot, so the old slot stays the destination.
        let old_labels_before = compiled.used_label_homes.get();
        let labels_grew = old_labels_before < merged_labels.max(old_labels_before);
        #[cfg(target_arch = "wasm32")]
        let install_handle = if labels_grew {
            let new_handle = glue::compile_module_cached(&wasm_bytes);
            if new_handle == 0 {
                return Err(BackendError::Unsupported(
                    "wasm host rejected the re-emitted trace module".into(),
                ));
            }
            compiled.func_handle.set(new_handle);
            new_handle
        } else {
            // A slot shared through the module cache comes back as a fresh
            // one, installed like the grown-labels arm above.
            let handle = glue::replace_module(old_handle, &wasm_bytes);
            if handle == 0 {
                return Err(BackendError::Unsupported(
                    "wasm host rejected the re-emitted trace module".into(),
                ));
            }
            if handle != old_handle {
                compiled.func_handle.set(handle);
            }
            handle
        };
        #[cfg(target_arch = "wasm32")]
        if install_handle == old_handle
            && old_handle != 0
            && compiled
                .resume_entry
                .slot
                .load(std::sync::atomic::Ordering::Relaxed)
                == old_handle
            && inputs.frame.frame_bytes == compiled.frame.frame_bytes
        {
            compiled
                .resume_entry
                .generation
                .store(resume_generation, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (wasm_bytes, labels_grew, resume_generation);
            return Err(BackendError::Unsupported(
                "wasm backend: no host replacement binding".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        let install_handle = old_handle;
        patch_pending_failure_recoveries(&guard_exits);

        // The host has accepted the replacement, so its newly encoded global
        // indices can now be made visible in the registry and local metadata.
        // Both instances remain resident, so account for the replacement block
        // in the same lifetime ledger as an ordinary compiled module.
        let block = self.asm_memory_stats.record_block(code_size, code_size);
        self.asm_memory_blocks.borrow_mut().push(block);
        if install_handle != 0 && install_handle != old_handle {
            asm_resources.table_slots.push(install_handle);
        }
        publish_exit_slots(
            &mut asm_resources,
            &guard_exits,
            &descrs,
            &inputs.ca.attached,
        );
        // Keep still-standalone bridge descriptors after the rebuilt merged
        // prefix. Adding regions grows that prefix, so every old positional
        // range moves by exactly the difference in guard-cell counts.
        let old_guard_count = compiled.num_guard_cells.get();
        let chained_descrs = compiled.fail_descrs.borrow()[old_guard_count..].to_vec();
        let mut replacement_descrs = descrs.clone();
        replacement_descrs.extend(chained_descrs);
        *compiled.fail_descrs.borrow_mut() = replacement_descrs;
        Self::register_meta_descrs(token, &descrs);
        let guard_growth = guard_exits.len().saturating_sub(old_guard_count);
        if guard_growth != 0 {
            for (_, _, start, _) in compiled.bridge_descr_ranges.borrow_mut().iter_mut() {
                *start += guard_growth;
            }
        }
        let old_homes = compiled.num_ref_homes.get();
        let old_labels = compiled.used_label_homes.get();
        let widened = merged_ref_homes.max(old_homes);
        let widened_labels = merged_labels.max(old_labels);
        // Leave retired bridge slots pointing at the old module. Callers
        // baked `return_call_indirect` with that bridge's LABEL key; the
        // replacement owner's `br_table` interprets the same key against
        // its own resumable-label prefix. `assembler.py` `patch_jump_for_descr`
        // rewrites the jump in place; wasm cannot, so the old module stays
        // the destination. The source-guard cell is already zero, so the
        // inlined region is not also dispatched.
        // Keep already-compiled bridges when only ordinary homes grew:
        // those new slots are nulled at keyed entry, and publication is
        // monotonic in the marked-bit set. Dropping them forced
        // `trace_eagerness` (200) extra guard failures.
        // The rebuilt module loads each guard's existing cell, so an
        // attached bridge stays reachable. The inlined region's cell was
        // already zeroed by the installer.
        compiled.module_bytes.set(code_size as u32);
        compiled.num_guard_cells.set(guard_exits.len());
        compiled.num_ref_homes.set(widened);
        compiled.used_label_homes.set(widened_labels);
        compiled.home_gcmap_ptr.set(leak_home_gcmap(
            &mut asm_resources,
            inputs.frame,
            widened,
            widened_labels,
        ));
        if let Some(clt) = token.compiled_loop_token() {
            let baseofs = (majit_gc::header::GcHeader::SIZE
                + majit_backend::jitframe::FIRST_ITEM_OFFSET) as i64;
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, inputs.frame.ca_frame_depth() as i64);
        }
        *compiled.reemit.borrow_mut() = Some(inputs.clone());

        // LABEL targets bake only the stable table slot, so restamp them for
        // this build. CA dispatch additionally carries the new finish index.
        let (_, _published_labels) = stamp_and_publish_label_targets(
            &mut asm_resources,
            install_handle,
            compiled.frame,
            &inputs.inputargs,
            &inputs.ops,
            inputs.bridge_entry_arity,
            token.identity_key(),
        );
        release::push_resources(token, asm_resources);
        if let Some(mut target) = target_from_token(token) {
            target.func_handle = install_handle;
            target.compiled_ptr = compiled as *const CompiledWasmLoop as usize as u64;
            target.marked_ordinary = compiled.num_ref_homes.get() as u32;
            target.marked_labels = compiled.used_label_homes.get() as u32;
            target.label_ref_slots = compiled.frame.label_ref_slots as u32;
            // Never clear a flag an out-of-line bridge already published:
            // re-emission can omit that bridge's ops while the attached
            // module still finishes through it.
            target.has_guard_not_forced_2 |=
                module_has_guard_not_forced_2(&inputs.ops, &inputs.inlined_bridges);
            publish_token_target(token, &target);
        }
        Ok(())
    }
}

// `Backend: Send` (`model.py` AbstractCPU is stored on MetaInterp).
// The TLS box is not in this struct; its destructor forgets the MiniMark
// so a thread hop cannot free the nursery on TLS teardown.
unsafe impl Send for WasmBackend {}

/// Report why a trace cannot be compiled by the wasm backend, or `None` if it
/// can. Declined traces fall back to the interpreter (correct, unaccelerated)
/// instead of producing an invalid trace module. `allow_ca` (set when every
/// CALL_ASSEMBLER target is admitted) lifts the CALL_ASSEMBLER decline so the
/// CA arm (guest→guest `call_indirect`) lowers it instead.
///
/// A JUMP whose target token is not defined by a local LABEL — the cross-loop
/// terminal jump — is not judged here: it is lowered to
/// `return_call_indirect(external_jump_slot)` and both callers resolve that
/// slot through [`resolve_cross_loop_jump_target`].

/// Stamp a position onto every non-Void-result op left unpositioned by the
/// optimizer, so no operand resolves to `OpRef::NONE` during codegen.
///
/// The optimizer's force path emits materialized allocation/store ops (e.g. a
/// virtualized list's `NewArray` backing block and its `SetfieldGc` /
/// `SetarrayitemGc` stores) with `Op::new`, and only assigns a position to ops
/// whose `result_type() != Void` — a Void-result store keeps `pos == NONE`.
/// A later op that consumes such a producer's result reads its `pos` through
/// `Operand::Op`, and an unpositioned producer yields `OpRef::NONE`
/// (`raw() == u32::MAX`), which `emit_resolve` would use to index `value_types`
/// out of bounds. The native backends normalize positions before codegen
/// (dynasm `prepare_ops_for_compile`, cranelift `normalize_ops_for_codegen_simple`);
/// the wasm backend does the same here.
fn normalize_ops_for_codegen(inputargs: &[InputArgRc], ops: &[OpRc]) -> Vec<Op> {
    let num_inputs = inputargs.len() as u32;
    ops.iter()
        .enumerate()
        .map(|(op_idx, op)| {
            let normalized = (**op).clone();
            let rt = normalized.result_type();
            if rt != majit_ir::Type::Void && normalized.pos().get().is_none() {
                normalized
                    .pos()
                    .set(majit_ir::OpRef::op_typed(num_inputs + op_idx as u32, rt));
            }
            normalized
        })
        .collect()
}

fn wasm_unsupported_trace_reason(ops: &[Op], allow_ca: bool) -> Option<String> {
    for op in ops {
        if op.opcode.is_call_assembler() && !allow_ca {
            // CALL_ASSEMBLER inlines a loop-bearing callee by jumping into another
            // trace's compiled token. `general_call_assembler_target` resolves
            // that token to a published guest function the CA arm reaches with a
            // `call_indirect`, so reaching here means some target did not resolve
            // — an unpublished token, an invalid signature, or a missing
            // deopt-helper slot — and there is nothing to call.
            return Some(format!(
                "wasm backend: {:?} (loop-callee inline)",
                op.opcode
            ));
        }
    }
    None
}

/// Whether a trace has a `JUMP` whose target token is not defined by one of
/// this compilation's LABELs, so codegen lowers it to a tail call into another
/// module rather than a `br`. This is the token test from
/// `x86/assembler.py:2463`. Testing only whether the trace had any LABEL was
/// wrong: a trace may define LABELs and still close onto a token from another
/// compilation, as a retrace attached as a bridge does.
fn has_cross_loop_terminal_jump(ops: &[Op]) -> bool {
    let has_jump = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Jump);
    has_jump && codegen::find_loop_label_index(ops).is_none()
}

/// Resolve the re-entry target of a cross-loop terminal JUMP off the descr
/// the JUMP holds (`LoopTargetDescr::ll_loop_code`). The JUMP and its
/// target LABEL share that descr, and every compiled loop published its
/// enterable labels there. The stamped `label_block_id` ordinal is NOT
/// identity: a retraced loop has several sibling specializations whose start
/// labels all carry ordinal 0, and a trace legitimately closes into a SIBLING
/// (jump-to-existing-trace) — the descr names the owning module's table slot
/// and resume key, so the tail call chains into the RIGHT loop.
///
/// Decline (`None`, after tallying which question answered) when the target is
/// unpublished (descr stripped, or its loop declined/was dropped), the JUMP
/// arity differs from the label's arg count (the resume loader reads exactly
/// that many positional frame slots), or the label's args are not the complete
/// live set of the target trace's remainder (`resume_safe` — resuming there
/// would read a null local).
///
/// `source` is the frame a chained bridge already runs on, with the table slot
/// of the loop that owns it. `compile_bridge` supplies it. A target whose
/// backend capture slots were filled by its own fall-through
/// (`requires_own_frame`) is resumable only when it IS that source loop: those
/// captures sit at the owner's offsets, and this jumper does not have the
/// target's `LabelResumeData`. A different `FrameGeometry` is otherwise
/// accepted; the caller writes the target's spill and dispatch-key offsets
/// (`remap_frame_layout`) and sizes `_check_frame_depth` to the target.
/// `compile_loop` passes `None` for an entry bridge, which owns its module and
/// its frame: it has no source slot, so `requires_own_frame` always declines,
/// and instead of matching a geometry it ADOPTS the target's (the caller
/// checks its own slots fit, then compiles against `t.frame`).
fn resolve_cross_loop_jump_target(
    ops: &[Op],
    source: Option<(u32, codegen::FrameGeometry)>,
) -> Option<LabelTarget> {
    let closing_jump = ops
        .iter()
        .rev()
        .find(|op| op.opcode == majit_ir::OpCode::Jump);
    let target_descr = closing_jump.and_then(|j| j.getdescr());
    let target = target_descr.as_ref().and_then(label_target);
    let arity = closing_jump.map_or(0, |j| j.getarglist().len());
    match target {
        // Descr stripped, or the target label was never published.
        None => {
            diag_bump(8);
            diag_bump(if target_descr.is_none() { 17 } else { 18 });
            None
        }
        Some(t) if arity != t.num_args => {
            diag_bump(10); // arity mismatch
            None
        }
        Some(t) if !t.resume_safe => {
            diag_bump(9); // label args not the full live set
            None
        }
        Some(t) if t.requires_own_frame && Some(t.func_handle) != source.map(|(slot, _)| slot) => {
            // The target's high capture homes were populated by its own
            // fall-through path, not by this sibling source loop.
            diag_bump(9);
            None
        }
        Some(t) => Some(t),
    }
}

/// Resolve every distinct compiled target used by CALL_ASSEMBLER ops in this
/// trace.  PyPy's `compile_tmp_callback` always supplies a real compiled token
/// while the final loop is pending, so every target here must likewise be an
/// installed `CompiledWasmLoop`; there is no bodyless self-placeholder case.
fn general_call_assembler_target(ops: &[Op]) -> Option<Vec<(usize, CallAssemblerTarget)>> {
    let mut resolved = Vec::new();
    let mut saw_ca = false;
    for op in ops.iter().filter(|op| op.opcode.is_call_assembler()) {
        saw_ca = true;
        let Some(descr_ref) = op.getdescr() else {
            diag_bump(58);
            return None;
        };
        let Some(descr) = descr_ref.as_call_descr() else {
            diag_bump(58);
            return None;
        };
        let arg_types = descr.arg_types();
        // A JitFrame slot is an i64 bit carrier for every scalar kind. Float
        // inputs/results therefore need no distinct CALL_ASSEMBLER ABI: the
        // callee entry and caller result arm reinterpret at the local boundary.
        // Void has no result local but follows the same finish-index protocol.
        if !arg_types.iter().all(|&tp| {
            matches!(
                tp,
                majit_ir::Type::Int | majit_ir::Type::Ref | majit_ir::Type::Float
            )
        }) || descr.result_type() != op.opcode.result_type()
        {
            diag_bump(59);
            return None;
        }
        let Some(owned) = descr_ref
            .as_loop_token_descr()
            .and_then(|ltd| ltd.token_handle_any())
            .and_then(|any| any.downcast_ref::<std::sync::Arc<JitCellToken>>())
        else {
            diag_bump(58);
            return None;
        };
        let target_token = owned.identity_key();
        let Some(mut registered) = target_from_token(owned) else {
            diag_bump(60);
            return None;
        };
        // A straight-line function trace may have deferred host module
        // compilation. CALL_ASSEMBLER is its first real consumer, so
        // materialize it before publishing the stable dispatch entry.
        if registered.func_handle == 0 && registered.compiled_ptr != 0 {
            let Some(loop_) =
                (unsafe { (registered.compiled_ptr as *const CompiledWasmLoop).as_ref() })
            else {
                diag_bump(61);
                return None;
            };
            let Ok(handle) = loop_.materialize_func_handle() else {
                diag_bump(61);
                return None;
            };
            if handle == 0 {
                diag_bump(61);
                return None;
            }
            registered.func_handle = handle;
            publish_token_target(owned, &registered);
        }
        if registered.input_types.as_slice() != arg_types {
            diag_bump(62);
            return None;
        }
        if registered.callee_frame_bytes == 0 {
            diag_bump(62);
            return None;
        }
        if registered.compiled_ptr == 0 {
            diag_bump(62);
            return None;
        }
        // A successfully compiled loop is retained by its token while it is
        // registered. It can subsequently become terminally declined, so read
        // the live state before baking every CA entry.
        let live = unsafe {
            (registered.compiled_ptr as *const CompiledWasmLoop)
                .as_ref()
                .is_some_and(|loop_| !loop_.ca_terminal_declined.get())
        };
        if !live {
            diag_bump(63);
            return None;
        }
        // The same target may occur in several operations; each operation was
        // validated above, while the codegen map needs one geometry per token.
        if !resolved
            .iter()
            .any(|(known_token, _)| *known_token == target_token)
        {
            resolved.push((target_token, registered));
        }
    }
    (saw_ca && !resolved.is_empty()).then_some(resolved)
}

fn module_has_guard_not_forced_2(ops: &[Op], inlined_bridges: &[codegen::InlinedBridge]) -> u32 {
    let is_gnf2 = |op: &Op| op.opcode == majit_ir::OpCode::GuardNotForced2;
    (ops.iter().any(is_gnf2)
        || inlined_bridges
            .iter()
            .any(|bridge| bridge.ops.iter().any(is_gnf2))) as u32
}

fn bridge_call_assembler_target(ops: &[Op]) -> Option<Vec<(usize, CallAssemblerTarget)>> {
    general_call_assembler_target(ops)
}

fn ca_codegen_targets(
    targets: &[(usize, CallAssemblerTarget)],
) -> std::collections::HashMap<usize, codegen::CaTarget> {
    targets
        .iter()
        .map(|(token, target)| {
            (
                *token,
                codegen::CaTarget {
                    dispatch_entry: target.dispatch_entry,
                },
            )
        })
        .collect()
}

fn ca_max_frame_bytes(targets: &[(usize, CallAssemblerTarget)]) -> u32 {
    targets
        .iter()
        .map(|(_, target)| target.callee_frame_bytes)
        .max()
        .expect("admitted CALL_ASSEMBLER targets must be non-empty")
}

fn mark_call_assembler_target_active(
    target: &CallAssemblerTarget,
    caller_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    // `caller_flag` is the invalidation flag the calling artifact's
    // `GUARD_NOT_INVALIDATED` reads — the token flag for a loop, the
    // bridge-generation flag for a bridge — so a terminal decline of the
    // callee invalidates exactly the artifact embedding the CA edge.
    // The target metadata is removed by `CompiledWasmLoop::drop`; compilation
    // is single-threaded, and callers only retain the pointer while the token
    // remains compiled. This is the same lifetime used by the deopt helper.
    let force_terminal_decline = unsafe {
        if let Some(loop_) = (target.compiled_ptr as *const CompiledWasmLoop).as_ref() {
            loop_.ca_active.set(true);
            {
                let mut callers = loop_.ca_callers.borrow_mut();
                callers.retain(|known| known.strong_count() > 0);
                let incoming = std::sync::Arc::downgrade(&caller_flag);
                if !callers
                    .iter()
                    .any(|known| std::sync::Weak::ptr_eq(known, &incoming))
                {
                    callers.push(incoming);
                }
            }

            // Runtime-regression hook for the terminal-decline CA path.  It
            // is dormant unless explicitly selected, and runs only after this
            // caller has already admitted and compiled a CA edge.  `1` selects
            // the first such target; a decimal JitCellToken number selects a
            // particular target.  The caller's invalidation bit still makes
            // this a bounded window, exactly like a real terminal bridge
            // decline.
            let selector = FORCE_CA_TERMINAL_DECLINE.load(Ordering::Relaxed);
            if selector != 0 && (selector == 1 || selector as i64 == target.token_number) {
                // One forced target per guest run. A real terminal decline
                // also transitions its target just once.
                FORCE_CA_TERMINAL_DECLINE.store(0, Ordering::Relaxed);
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    if force_terminal_decline {
        // `mark_call_assembler_terminal_decline` reads `ca_callers`; release
        // the registration borrow above before invalidating those callers.
        mark_call_assembler_terminal_decline(target.compiled_ptr as usize);
        diag_bump(16);
    }
}

/// Move the movable-CA caller census from a redirected target to its
/// replacement. Existing callers retain the old dispatch entry, but terminal
/// decline of the replacement must still invalidate those callers.
fn transfer_call_assembler_target_activity(
    old_target: &CallAssemblerTarget,
    new_target: &CallAssemblerTarget,
) {
    unsafe {
        let Some(old_loop) = (old_target.compiled_ptr as *const CompiledWasmLoop).as_ref() else {
            return;
        };
        let Some(new_loop) = (new_target.compiled_ptr as *const CompiledWasmLoop).as_ref() else {
            return;
        };

        new_loop
            .ca_active
            .set(new_loop.ca_active.get() || old_loop.ca_active.get());
        let old_callers = old_loop.ca_callers.borrow().clone();
        let mut new_callers = new_loop.ca_callers.borrow_mut();
        new_callers.retain(|known| known.strong_count() > 0);
        for caller in old_callers {
            if caller.strong_count() == 0 {
                continue;
            }
            if !new_callers
                .iter()
                .any(|known| std::sync::Weak::ptr_eq(known, &caller))
            {
                new_callers.push(caller);
            }
        }
    }
}

/// Mark a CA target whose callee guard was structurally declined.
/// `compile_bridge` calls this on a terminal `Unsupported` so callers
/// that baked this loop as CALL_ASSEMBLER retrace without the CA edge.
pub fn mark_call_assembler_terminal_decline(compiled_ptr: usize) {
    unsafe {
        let Some(loop_) = (compiled_ptr as *const CompiledWasmLoop).as_ref() else {
            return;
        };
        if loop_.ca_terminal_declined.replace(true) {
            return;
        }
        let mut callers = loop_.ca_callers.borrow_mut();
        callers.retain(|known| known.strong_count() > 0);
        for caller in callers.iter() {
            if let Some(flag) = caller.upgrade() {
                flag.store(true, Ordering::Release);
            }
        }
    }
}

fn note_call_assembler_terminal_decline_for_token(original_token: &JitCellToken) {
    let Some(loop_) = original_token
        .compiled
        .get()
        .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
    else {
        return;
    };
    mark_call_assembler_terminal_decline(loop_ as *const CompiledWasmLoop as usize);
}

fn terminal_bridge_unsupported(
    original_token: &JitCellToken,
    reason: impl Into<String>,
) -> BackendError {
    note_call_assembler_terminal_decline_for_token(original_token);
    BackendError::Unsupported(reason.into())
}

/// Exit slots to decode out of a returned frame for `fail_descr`.
///
/// Its fail arguments, plus the GUARD_VALUE operand the exit spills one slot
/// past them when the guard does not carry it as a fail argument
/// (`codegen::counter_value_spill`). That trailing word is what
/// `resolve_guard_value_operand` reads back through `get_value_direct` for
/// `make_a_counter_per_value`; it is never a fail argument, so it stays out of
/// `fail_arg_types` and out of every typed exit decode.
#[allow(dead_code)] // codegen::counter_value_spill exit width
pub(crate) fn exit_slot_count(fail_descr: &failguard::WasmFailDescr) -> usize {
    let fail_args = fail_descr.fail_arg_types.len();
    fail_descr
        .meta_descr
        .as_ref()
        .and_then(|d| d.as_fail_descr())
        .and_then(majit_backend::guard_value_counter_slot)
        .map_or(fail_args, |slot| fail_args.max(slot + 1))
}

/// Decode through the locations saved on the descriptor, as
/// BaseAssembler.rebuild_faillocs_from_descr does. The frontend still sees
/// its logical resume numbering; a hole neither reads nor reserves memory.
fn exit_arg_word(frame_ptr: usize, fail_descr: &WasmFailDescr, index: usize) -> i64 {
    fail_descr.frame_slot(index).map_or(0, |slot| unsafe {
        *((frame_ptr + codegen::FRAME_SLOT_BASE as usize + slot * 8) as *const i64)
    })
}

/// Reconstruct a [`DeadFrame`] from a callee frame an in-guest `call_indirect`
/// already ran to a guard/finish exit (the self-recursive CALL_ASSEMBLER fast
/// path, `PYRE_WASM_CA`). This is the post-`glue::execute` tail of
/// [`WasmBackend::execute_token`] factored for a frame the host did not itself
/// enter: `jf_descr` holds the exit's descr cell, `jf_frame` the exit slots.
/// `grab_exc_value` reads `jf_guard_exc`, stored by the failing arm.
/// `pyre-jit`'s `call_jit::wasm_ca_resume_deopt` calls this, then drives the
/// resulting `DeadFrame` through the same `get_latest_descr_arc` /
/// `get_*_value` / `grab_exc_value` Backend path the host's outermost deopt
/// handling uses, so the in-guest deopt completes identically.
///
/// `jf_descr` is the failing guard's [`failguard::FailDescrCell`]. The exit
/// may belong to a bridge chained past the source loop; the pointer names
/// that descr. `_compiled_ptr` stays in the trace ABI and is not consulted.
pub fn dead_frame_from_ran_frame(_compiled_ptr: usize, frame_ptr: usize) -> DeadFrame {
    let jf = unsafe {
        majit_backend::jitframe::JitFrame::resolve(
            (frame_ptr - majit_backend::jitframe::FIRST_ITEM_OFFSET)
                as *mut majit_backend::jitframe::JitFrame,
        )
    };
    let fail_descr = descr_at(unsafe { (*jf).jf_descr })
        .expect("invalid jf_descr from in-guest CA callee frame");
    DeadFrame::Boxed(WasmFrameData::from_live_frame(
        jf, fail_descr, false, true, None,
    ))
}

fn force_arg_word(frame_ptr: usize, fail_descr: &WasmFailDescr, index: usize) -> i64 {
    let Some(slot) = fail_descr.frame_slot(index) else {
        return 0;
    };
    // `slot` is the physical value-item index (`spill_slot_index`). Prefix
    // indices are the identity, so `force_args_offset + slot * 8` is
    // `force_slot_ofs`. A tail index steps by three from `value_tail_index`;
    // `force_tail_base` is the first tail force word, three items apart as well.
    let offset =
        if fail_descr.force_tail_base != 0 && slot >= fail_descr.force_prefix_slots as usize {
            let delta = slot - fail_descr.value_tail_index as usize;
            fail_descr.force_tail_base as usize + delta * std::mem::size_of::<i64>()
        } else {
            fail_descr.force_args_offset as usize + slot * std::mem::size_of::<i64>()
        };
    unsafe { *((frame_ptr + offset) as *const i64) }
}

fn decoded_force_word(frame_ptr: usize, fail_descr: &WasmFailDescr, index: usize) -> i64 {
    let word = force_arg_word(frame_ptr, fail_descr, index);
    // `emit_force_arm` publishes a Ref as `home_offset * 2 + 1`
    // (bit 0) so the value is read out of the traced home a
    // collection inside the bracketed call forwards. A non-null
    // ConstPtr has no home; it is published as `table_addr | 3`
    // (bits 0 and 1) so this path reloads the forwarded GC-table
    // slot. A literal is even (Ref pointers are 8-aligned; a
    // null and a non-Ref argument are published as themselves).
    if fail_descr.fail_arg_types.get(index) == Some(&majit_ir::Type::Ref) && word & 1 == 1 {
        let addr = if word & 2 == 2 {
            (word & !3) as usize
        } else {
            frame_ptr + (word >> 1) as usize
        };
        if word & 2 == 2 && std::mem::size_of::<majit_ir::GcRef>() == 4 {
            unsafe { i64::from(*(addr as *const u32)) }
        } else {
            unsafe { *(addr as *const i64) }
        }
    } else {
        word
    }
}

/// Reconstruct a [`DeadFrame`] for a frame a FORCE interrupted while its call
/// is still on the stack. A force is not an exit, so this must not consume
/// the pending-exception cell. `jitframe_resolve` follows `jf_forward` when
/// `_check_frame_depth` grew the frame after the token was captured.
fn dead_frame_from_forced_frame(frame_ptr: usize, fail_descr: Arc<WasmFailDescr>) -> DeadFrame {
    let jf = unsafe {
        majit_backend::jitframe::JitFrame::resolve(
            (frame_ptr - majit_backend::jitframe::FIRST_ITEM_OFFSET)
                as *mut majit_backend::jitframe::JitFrame,
        )
    };
    // The compiled run still owns this frame (it is on the shadow stack).
    // Borrow it: freeing or dropping the GC root here would unroot a frame
    // the call is still using. `read_force` makes get_* decode the force spill.
    DeadFrame::Boxed(WasmFrameData::from_live_frame(
        jf, fail_descr, true, true, None,
    ))
}

fn publish_exit_slots(
    resources: &mut release::LoopAsmResources,
    guards: &[codegen::GuardExit],
    descrs: &[Arc<WasmFailDescr>],
    attached: &majit_backend::AttachedDescrPtrs,
) {
    for (index, (guard, descr)) in guards.iter().zip(descrs).enumerate() {
        fill_exit_cell(guard.descr_cell, Arc::clone(descr));
        let cell = if guard.is_finish {
            failguard::attached_finish_exit_index(attached, &guard.meta_descr)
                .map(|exit| failguard::finish_cell_ptr(attached, exit))
                .filter(|cell| *cell != 0)
                .unwrap_or(guard.descr_cell)
        } else {
            guard.descr_cell
        };
        resources.write_exit_slot(index, cell, guard.exit_gcmap_ptr);
    }
}

fn wasm_frame_data(frame: &DeadFrame) -> &WasmFrameData {
    frame
        .boxed_data()
        .and_then(|d| d.downcast_ref::<WasmFrameData>())
        .expect("not WasmFrameData")
}

/// Logical fail-arg `index` from the live jitframe, or from the synthetic
/// `raw_values` vector when no frame is attached.
fn wasm_frame_word(data: &WasmFrameData, index: usize) -> i64 {
    let Some(base) = data.items_base() else {
        return data.raw_values[index];
    };
    if data.read_force() {
        decoded_force_word(base, &data.fail_descr, index)
    } else {
        exit_arg_word(base, &data.fail_descr, index)
    }
}

impl majit_backend::Backend for WasmBackend {
    /// `force(token)` where the token is what `FORCE_TOKEN` parked in the
    /// virtualizable: the running frame's `JitFrame`, whose data region starts
    /// `FIRST_ITEM_OFFSET` in.
    ///
    /// A zero token means no frame is holding the virtualizable, which is the
    /// interpreter-only state the default answers `None` for.
    fn force(&self, force_token: GcRef) -> Option<DeadFrame> {
        if force_token.0 == 0 {
            return None;
        }
        // runner.rs `force` on the native backends: assert the frame carries the
        // bracket its call published, then mark it so the GUARD_NOT_FORCED
        // waiting past that call deopts instead of running on.
        // `jitframe_resolve`: a bridge may have replaced the frame
        // `FORCE_TOKEN` captured.
        let jf = unsafe {
            majit_backend::jitframe::JitFrame::resolve(
                force_token.0 as *mut majit_backend::jitframe::JitFrame,
            )
        };
        let cell = unsafe { (*jf).jf_force_descr };
        assert_ne!(cell, 0, "force: wasm frame carries no force descriptor");
        // assembler.py `force`: publish the armed descr into `jf_descr` so
        // GUARD_NOT_FORCED's `CMP [jf_descr], 0` fails.
        unsafe { (*jf).jf_descr = cell };
        let fail_descr = descr_at(cell).expect("invalid jf_force_descr on a forced wasm frame");
        let items_base = jf as usize + majit_backend::jitframe::FIRST_ITEM_OFFSET;
        Some(dead_frame_from_forced_frame(items_base, fail_descr))
    }

    fn is_force_token_armed(&self, force_token: GcRef) -> bool {
        force_token.0 != 0
            && unsafe {
                let jf = majit_backend::jitframe::JitFrame::resolve(
                    force_token.0 as *mut majit_backend::jitframe::JitFrame,
                );
                (*jf).jf_force_descr
            } != 0
    }

    fn supports_efficient_uint_mul_high(&self) -> bool {
        // WebAssembly has no high-half integer multiply.  The fallback in
        // codegen is multi-precision software, while i64.div/rem are native
        // Wasm operations.
        false
    }

    fn cpu_tracker(&self) -> &std::sync::Arc<majit_backend::CpuTotalTracker> {
        &self.cpu_tracker
    }

    fn assembler_memory_stats(&self) -> (usize, usize) {
        self.asm_memory_stats.get_stats()
    }

    fn backend_name(&self) -> &'static str {
        "wasm"
    }

    // ── Blackhole allocation (llmodel.py bh_new) ──
    //
    // The blackhole interpreter materializes virtuals (e.g. a virtualized
    // `W_IntObject` loop variable forced at loop exit) through these. Without
    // a real implementation `bhimpl_new*` returns 0 and the resumed frame
    // carries null operands. Mirrors `CraneliftBackend`'s overrides but routes
    // through the wasm thread-local GC; the old-generation allocator never
    // collects, so allocation inputs need no rooting here.

    /// llmodel.py bh_new(sizedescr).
    fn bh_new(&self, sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        wasm_bh_alloc_struct(sizedescr)
    }

    /// llmodel.py bh_new_with_vtable(sizedescr): allocate, then write
    /// the type pointer at `vtable_offset` and the class word at
    /// `w_class_offset`.
    fn bh_new_with_vtable(&self, sizedescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let vtable = sizedescr.get_vtable();
        let ptr = wasm_bh_alloc_struct(sizedescr);
        unsafe {
            majit_backend::write_new_with_vtable_header(
                ptr as *mut u8,
                vtable,
                self.vtable_offset,
                self.w_class_offset,
            );
        }
        ptr
    }

    fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    fn w_class_offset(&self) -> Option<usize> {
        self.w_class_offset
    }

    fn set_w_class_offset(&mut self, offset: Option<usize>) {
        self.w_class_offset = offset;
    }

    /// llmodel.py bh_new_array(length, arraydescr).
    fn bh_new_array(&self, length: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let len_offset = arraydescr
            .array_len_offset()
            .expect("bh_new_array requires ArrayDescr.lendescr");
        // descr.py `ArrayDescr.get_type_id(): assert self.tid` — allocation
        // requires a real GC type id; tid=0 means the descr never went through
        // `gc.py:548 set_type_id` and the GC tracer would lack the per-item
        // visit shape.  The dynasm and cranelift runners assert the same.
        let type_id = arraydescr.resolve_gc_tid();
        assert!(
            type_id != 0,
            "bh_new_array requires ArrayDescr.tid (descr.py:340) — got 0"
        );
        let Some(payload_size) = itemsize
            .checked_mul(length)
            .and_then(|items| base_size.checked_add(items))
        else {
            return 0;
        };
        let ptr = wasm_bh_alloc(type_id, payload_size);
        if ptr != 0 {
            unsafe {
                *((ptr as *mut u8).add(len_offset) as *mut usize) = length;
            }
        }
        ptr
    }

    /// llmodel.py bh_new_array_clear = bh_new_array (allocator zeroes).
    fn bh_new_array_clear(&self, length: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        self.bh_new_array(length, arraydescr)
    }

    /// `LLtypeMixin.bh_newstr` → `gc_ll_descr.gc_malloc_str`.
    fn bh_newstr(&self, length: i64) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let base_size = 2 * std::mem::size_of::<usize>() + 1;
        let Some(payload_size) = base_size.checked_add(length) else {
            return 0;
        };
        let ptr = wasm_bh_alloc(majit_gc::lowlevel_str_type_id(), payload_size);
        if ptr != 0 {
            unsafe {
                *((ptr as *mut u8).add(std::mem::size_of::<usize>()) as *mut usize) = length;
            }
        }
        ptr
    }

    /// `LLtypeMixin.bh_newunicode` → `gc_ll_descr.gc_malloc_unicode`.
    fn bh_newunicode(&self, length: i64) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let base_size = 2 * std::mem::size_of::<usize>();
        let Some(payload_size) = length
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|items| base_size.checked_add(items))
        else {
            return 0;
        };
        let ptr = wasm_bh_alloc(majit_gc::lowlevel_unicode_type_id(), payload_size);
        if ptr != 0 {
            unsafe {
                *((ptr as *mut u8).add(std::mem::size_of::<usize>()) as *mut usize) = length;
            }
        }
        ptr
    }

    /// llmodel.py bh_arraylen_gc: read the length prefix at
    /// `lendescr.offset`. Word-width (`*const usize`), matching the store
    /// `bh_new_array` makes at the same offset — a fixed 8-byte read would fold
    /// the first item into the high half on wasm32.
    ///
    /// Without this the trait stub answers `0` for every array length reached
    /// at trace time, so a spare-capacity test (`length < len(items)`) records
    /// its at-capacity arm on a list that has room. The compiled code reads the
    /// real length, so that guard then fails on nearly every iteration and the
    /// trace never stays in compiled code.
    fn bh_arraylen_gc(&self, array_ptr: i64, arraydescr: &majit_jitcode::jitcode::BhDescr) -> i64 {
        let ofs = arraydescr
            .array_len_offset()
            .expect("bh_arraylen_gc requires ArrayDescr.lendescr");
        unsafe { *((array_ptr as *const u8).add(ofs) as *const usize) as i64 }
    }

    fn compile_loop(
        &mut self,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
        token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        // `gctypelayout.py encode_type_shapes_now` closes `type_info_group`
        // at translation. Close before `collect_guard_gc_type_info` reads it.
        majit_gc::ensure_type_registry_closed();
        diag_bump(23);
        let _header_pc = std::mem::take(&mut self.next_header_pc);
        // `x86/assembler.py:514` parity — bump
        // `cpu.tracker.total_compiled_loops` at the same point PyPy
        // creates the `CompiledLoopToken`.
        if let Some(clt) = token.compiled_loop_token() {
            majit_backend::record_compiled_loop_token(&self.cpu_tracker, &clt);
        }
        let mut ops_owned: Vec<Op> = normalize_ops_for_codegen(inputargs, ops);
        codegen::materialize_unbound_label_args(inputargs, &mut ops_owned);
        // Locs are read by `rewrite_ops_for_gc`. An entry bridge's tail
        // offsets depend only on the target's prefix and `frame_bytes`, which
        // `extend` keeps even when the post-rewrite slot count is larger.
        let entry_bridge_target = if has_cross_loop_terminal_jump(&ops_owned) {
            let Some(target) = resolve_cross_loop_jump_target(&ops_owned, None) else {
                diag_bump(2); // declined: JUMP target not chainable
                return decline_compile_loop(BackendError::Unsupported(
                    "wasm backend: cross-loop terminal JUMP target is not a \
                     chainable published label"
                        .into(),
                ));
            };
            Some(target)
        } else {
            None
        };
        let publish_frame = match entry_bridge_target {
            Some(target) => target.frame.extend(
                inputargs.len().max(target.frame.value_slots),
                target.frame.addressable_ordinary_homes(),
            ),
            None => codegen::FrameGeometry::fixed(),
        };
        publish_ca_initial_locs(token, publish_frame, inputargs.len());
        if let Some(reason) = missing_call_assembler_locs(&ops_owned) {
            diag_bump(25);
            return decline_compile_loop(BackendError::Unsupported(reason));
        }
        let (ops_owned, gc_table) = self.rewrite_ops_for_gc(ops_owned);
        let gc_table_base = gc_table.as_ref().map_or(0, |t| t.base_addr() as u32);
        let ops: &[Op] = &ops_owned;
        // Freeze this token's generated frame layout before CA resolution.  A
        // self-recursive CALL_ASSEMBLER reaches this point while its token is
        // still a pending placeholder, so this is the one authoritative frame
        // geometry for both the loop and each nursery-allocated self callee.
        let raw_frame_value_slots = codegen::frame_value_slots(inputargs, ops);
        let raw_num_ref_homes = codegen::count_ref_homes(inputargs, ops);
        let label_ref_slots =
            codegen::label_ref_capture_slots(inputargs, ops).max(FROZEN_CHAIN_LABEL_REF_SLOTS);
        // An entry bridge (`compile.py ResumeFromInterpDescr`) is sent
        // to the backend through `compile_loop` like any loop, but it is not one:
        // it has no LABEL of its own and ends in a JUMP into an
        // already-compiled loop. The target resolved above is the one
        // `compile_bridge` would use, and this loop ADOPTs its frozen
        // geometry — `LabelTarget::frame` exists because a tail call reuses the
        // caller's frame, so the two layouts must agree offset for offset, not
        // merely in size. Compiling against the target's geometry makes them
        // agree by construction. `_check_frame_depth`: a trace that needs more
        // than that layout keeps it and appends the overflow.
        let frame = match entry_bridge_target {
            Some(target) => target
                .frame
                .extend(raw_frame_value_slots, raw_num_ref_homes),
            None => codegen::FrameGeometry::compact(
                frozen_slot_count(raw_frame_value_slots.max(FROZEN_LABEL_PARAM_ARITY + 1)),
                frozen_slot_count(raw_num_ref_homes) + label_ref_slots,
                label_ref_slots,
            ),
        };
        // `x86/assembler.py::assemble_loop` installs the generated frame
        // depth on the token's `CompiledLoopToken.frame_info`.  CALL_ASSEMBLER
        // redirect later propagates the replacement depth through that exact
        // object (`CompiledLoopToken.update_frame_info`).
        if let Some(clt) = token.compiled_loop_token() {
            let baseofs = (majit_gc::header::GcHeader::SIZE
                + majit_backend::jitframe::FIRST_ITEM_OFFSET) as i64;
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, frame.ca_frame_depth() as i64);
        }
        // A general CALL_ASSEMBLER enters the real compiled token selected by
        // the descr.  While this loop is pending, PyPy puts a separately
        // compiled tmp callback in that cell; the bodyless pending-token
        // shortcut previously used by wasm is intentionally absent.
        let ca_targets = general_call_assembler_target(ops);
        let allow_ca = ca_deopt_helper_slot() != 0 && ca_targets.is_some();

        // Decline traces the wasm backend cannot compile correctly, so the
        // metainterp falls back to the interpreter (correct, if unaccelerated)
        // rather than installing a structurally-invalid trace module. For
        // CALL_ASSEMBLER that is the unresolved-target case only, judged by
        // `allow_ca` above; see `wasm_unsupported_trace_reason`.
        if let Some(reason) = wasm_unsupported_trace_reason(ops, allow_ca) {
            diag_bump(25);
            return decline_compile_loop(BackendError::Unsupported(reason));
        }
        if allow_ca {
            diag_bump(14); // accepted general CALL_ASSEMBLER loop
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;
        let trace_entry_census =
            alloc_trace_entry_census(trace_id, codegen::entry_dispatch_key_count(ops));

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        // Allocation helpers reached from a compiled trace through the host
        // `jit_call` trampoline. `fn as usize` is the `__indirect_function_table`
        // index on wasm32; taking it here keeps the function in the table.
        let alloc = alloc_helpers();
        let wb = wasm_write_barrier_helpers();
        // Each exit stores its `FailDescrCell` in `jf_descr`. The cell lives
        // in this loop's `LoopAsmResources`, so a chained module names its
        // own descr without a global index.
        let guard_exit_count = codegen::guard_exit_count(inputargs, ops);
        let fail_index_base = 0u32;
        let (bridge_cells_base, bridge_cells_owner) = codegen::alloc_bridge_cells(guard_exit_count);
        let bridge_param_dispatch = bridge_param_dispatch_for(guard_exit_count);
        // assembler.py keeps `_finish_gcmap` with the compiled loop. The
        // module leaks the map from its own RefHomes / LABEL captures after
        // those stores, matching a safepoint write.
        let used_label_homes = codegen::label_ref_capture_slots(inputargs, ops);
        let mut module_inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.iter().cloned().collect(),
            // Keep these rewritten operations exactly as rewrite_ops_for_gc
            // produced them; their LoadFromGcTable immediates share this base.
            ops: ops_owned.clone(),
            inlined_bridges: Vec::new(),
            constants: self.constants.clone(),
            vtable_offset: self.vtable_offset,
            classptr_to_typeid: typeid_table,
            guard_gc_type_info,
            alloc,
            wb,
            nursery: nursery_alloc_params(ops),
            invalidated_flag_addr: Arc::as_ptr(&token.invalidated) as usize as u32,
            gc_table_base,
            gc_const_keys: gc_const_keys_of(gc_table.as_deref()),
            fail_index_base,
            bridge_cells_base,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity: None,
            bridge_param_dispatch,
            trace_entry_census,
            inline_trip: None,
            // A real loop's JUMP is a local back-edge `br`; an entry bridge
            // tail-calls its target loop and is deliberately not re-emittable.
            external_jump_slot: entry_bridge_target.map_or(0, |t| t.func_handle),
            external_jump_key: entry_bridge_target.map_or(0, |t| t.key),
            external_jump_wide_slot: entry_bridge_target.map_or(0, |t| t.wide_slot),
            frame,
            ca: ca_targets.as_ref().map_or_else(
                || codegen::CaParams {
                    ca_reload_fn_ptr: body_reload_fn_ptr(),
                    jf_top_addr: jf_top_addr(),
                    compute_home_gcmap: true,
                    ..codegen::CaParams::default()
                },
                |targets| codegen::CaParams {
                    emit_ca: true,
                    targets: ca_codegen_targets(targets),
                    deopt_helper_slot: ca_deopt_helper_slot(),
                    ca_push_fn_ptr: wasm_jit_ca_push_frame as *const () as usize as i64,
                    ca_pop_fn_ptr: wasm_jit_ca_pop_frame as *const () as usize as i64,
                    ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                    ca_reload_caller_fn_ptr: wasm_jit_ca_reload_caller_frame as *const () as usize
                        as i64,
                    inline: ca_inline_params(ca_max_frame_bytes(targets)),
                    jf_top_addr: jf_top_addr(),
                    compute_home_gcmap: true,
                    ..codegen::CaParams::default()
                },
            ),
        };
        // `assembler.py` `_check_frame_depth`: a loop can run on a JITFRAME
        // another trace allocated (CALL_ASSEMBLER, a JUMP, a re-emitted
        // replacement tail-called with the live frame). The compare is a
        // no-op when `jf_frame.length` already matches `signed_item_count`.
        // `fn as usize` is the table index on wasm32.
        module_inputs.ca.realloc_fn_ptr = wasm_realloc_frame as *const () as usize as i64;
        let mut asm_resources = release::LoopAsmResources::default();
        asm_resources.exit_cells = Some(std::sync::Arc::clone(&self.exit_cells));
        module_inputs.ca.exit_table_base = asm_resources.alloc_exit_table(guard_exit_count) as u32;
        module_inputs.ca.gcmap_sink = &mut asm_resources as *mut _ as usize;
        // `runner.rs` captures `AttachedDescrPtrs` at `compile_loop` entry.
        module_inputs.ca.attached = self.exit_cells.descr_ptrs();
        let resume_entry = Box::new(failguard::ResumeEntry::new());
        #[cfg(target_arch = "wasm32")]
        {
            module_inputs.ca.resume_entry_addr = resume_entry.addr();
            module_inputs.ca.resume_generation = 1;
        }
        let (wasm_bytes, guard_exits, num_ref_homes, _used_labels) =
            match codegen::build_wasm_module(&module_inputs) {
                Ok(built) => built,
                Err(err) => {
                    record_last_compile_err(&err);
                    record_compile_loop_error(&err);
                    diag_bump(66);
                    return Err(err);
                }
            };
        let home_gcmap_ptr =
            leak_home_gcmap(&mut asm_resources, frame, num_ref_homes, used_label_homes);

        // Build fail descriptors
        let fail_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    fail_locs: g.fail_locs.clone(),
                    is_finish: g.is_finish,
                    force_args_offset: frame.force_slot_base as u32,
                    force_tail_base: force_tail_fields(frame).0,
                    force_prefix_slots: force_tail_fields(frame).1,
                    value_tail_index: force_tail_fields(frame).2,
                    force_gcmap_ptr: g.exit_gcmap_ptr,
                    bridge_cell: g.bridge_cell,
                    fail_arg_advanced: Vec::new(),
                    trace_ref_homes: 0,
                    trace_label_homes: 0,
                    param_dispatch: false,
                    bridge_slot: std::sync::atomic::AtomicU32::new(0),
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();
        if let Some(table) = gc_table {
            Self::register_gc_table(token, table);
        }

        // `runner.rs` / `compiler.rs` parity: the entry path reads
        // this to size the live-value list it hands `execute_token`
        // (`jitdriver.rs extend_compiled_live_values` →
        // `warmstate.py:188 cell.loop_token`). Leaving it unset makes a trace
        // whose inputargs outnumber the portal's live values be entered with
        // the short list, so every frame slot past it reads as a zero the
        // prologue then loads as a null Ref.
        token.set_inputarg_types(inputargs.iter().map(|ia| ia.tp.get()).collect());

        let max_output_slots = guard_exits
            .iter()
            .map(|g| g.fail_arg_refs.len())
            .max()
            .unwrap_or(0)
            .max(inputargs.len());

        // Straight-line function-entry traces finish the current invocation
        // concretely.  Do not ask the host to compile their wasm module until
        // a later invocation actually enters the token: quasi-immutable
        // invalidation can retire such a token before it ever executes (the
        // module-global `except ... as e` stress case does exactly that).
        // Loop-bearing and CALL_ASSEMBLER traces stay eager because their
        // published label/CA targets need a live table slot immediately.
        let defer_host_compile = !ops.iter().any(|op| {
            op.opcode == majit_ir::OpCode::Label
                || op.opcode == majit_ir::OpCode::Jump
                || op.opcode.is_call_assembler()
        });
        // The encoded module length, which is what this target can measure:
        // the host runtime owns the compiled code, so there is no pyre-owned
        // executable mapping and no retained capacity to report. It is read
        // before any host compilation, and on native builds no host exists at
        // all, so it is a submitted-bytes figure rather than
        // `asmmemmgr.py:90`'s mapped arena.
        let code_size = wasm_bytes.len();

        // Instantiate via the host binding on wasm32, or store bytes for
        // testing on native (no wasm host available).
        #[cfg(target_arch = "wasm32")]
        let func_handle = if defer_host_compile {
            0
        } else {
            glue::compile_module_cached(&wasm_bytes)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let func_handle = 0u32; // Placeholder — no wasm host available

        // `jit_compile_wasm` returns 0 when the host runtime rejects the emitted
        // module (e.g. a function body exceeding the parser's size limit — a
        // trace within the metainterp `trace_limit` can still overflow it once
        // the optimizer peels/unrolls the loop). Storing a token whose handle is
        // this dead sentinel would let `execute_token` dispatch table slot 0 (not
        // a trace), leaving `frame[0]` unwritten and resolving a wrong exit descr.
        // Decline the compile so the metainterp keeps the interpreter fallback —
        // a backend capability limit, reported like any other unsupported shape.
        #[cfg(target_arch = "wasm32")]
        if !defer_host_compile && func_handle == 0 {
            diag_bump(26);
            return decline_compile_loop(BackendError::Unsupported(
                "wasm host rejected the compiled trace module (oversized function body \
                 or invalid module)"
                    .to_string(),
            ));
        }

        patch_pending_failure_recoveries(&guard_exits);

        // `asmmemmgr.py` counts the block a `materialize` handed out, which
        // here is the module the host has taken. Below the decline above, as
        // `compile_bridge` does: a rejected module was never instantiated, and
        // the token retained for it would charge its bytes for the backend's
        // whole life.
        let block = self.asm_memory_stats.record_block(code_size, code_size);
        self.asm_memory_blocks.borrow_mut().push(block);

        // A peeled loop carries real work before its (last) LABEL — the
        // unrolled first iteration. codegen emits the `loop` at that LABEL, so
        // the preamble runs once on entry and is NOT part of the iterating body.
        // A loop-closing bridge that re-enters through `func_handle` would
        // re-run this preamble; record the shape so `compile_bridge` can decline
        // such a bridge (see `has_preamble` doc on the struct).
        // A peeled loop (the resume-at-LABEL wrapper's shape) — real work before
        // the last LABEL. Computed through the same predicate codegen's wrapper
        // gates on, so the recorded field and the emitted wrapper cannot drift.
        let has_preamble = codegen::is_resumable_peeled(ops);
        let (label_descrs, published_labels) = stamp_and_publish_label_targets(
            &mut asm_resources,
            func_handle,
            frame,
            inputargs,
            ops,
            None,
            token.identity_key(),
        );
        if func_handle != 0 {
            asm_resources.table_slots.push(func_handle);
        }
        publish_exit_slots(
            &mut asm_resources,
            &guard_exits,
            &fail_descrs,
            &module_inputs.ca.attached,
        );
        if let Some(cells) = bridge_cells_owner {
            asm_resources.bridge_cells.push(cells);
        }
        module_inputs.ca.gcmap_sink = 0;
        release::push_resources(token, asm_resources);
        // Per-guard, per-fail-arg induction-advance flags for
        // `compile_bridge`'s livelock check (see `guard_fail_args_advanced`).
        let guard_fail_arg_advanced = guard_fail_args_advanced(ops, &guard_exits);
        // `assembler.py` keeps `_finish_gcmap` with the compiled loop.  A
        // GUARD_NOT_FORCED_2 frame can be reached through a virtualizable token
        // after the host deadframe wrapper has returned, so this map cannot be
        // scoped to one `execute_token` call.

        let compiled = CompiledWasmLoop {
            token_number: token.identity_key(),
            trace_id,
            input_types: inputargs.iter().map(|ia| ia.tp.get()).collect(),
            func_handle: std::cell::Cell::new(func_handle),
            resume_entry,
            pending_wasm_bytes: std::cell::RefCell::new(defer_host_compile.then_some(wasm_bytes)),
            compiled_loop_token: token.compiled_loop_token_expect(),
            descrs_registered: std::cell::Cell::new(false),
            fail_descrs: std::cell::RefCell::new(fail_descrs),
            num_inputs: inputargs.len(),
            max_output_slots,
            num_ref_homes: std::cell::Cell::new(num_ref_homes),
            used_label_homes: std::cell::Cell::new(used_label_homes),
            frame,
            home_gcmap_ptr: std::cell::Cell::new(home_gcmap_ptr),
            bridge_cells_base: std::cell::Cell::new(bridge_cells_base),
            retained_owner_cells_base: std::cell::Cell::new(0),
            module_bytes: std::cell::Cell::new(code_size as u32),
            num_guard_cells: std::cell::Cell::new(guard_exits.len()),
            has_preamble,
            label_descrs,
            published_label_descrs: published_labels,
            guard_fail_arg_advanced,
            guard_fail_arg_counts: guard_exits
                .iter()
                .map(|guard| {
                    crate::codegen::live_fail_arg_count(
                        guard.meta_descr.as_ref(),
                        guard.fail_arg_refs.len(),
                    )
                })
                .collect(),
            bridge_param_dispatch,
            bridge_descr_ranges: std::cell::RefCell::new(Vec::new()),
            // Retaining the snapshot costs long-lived heap for the token's
            // whole lifetime, which moves when the collector next runs and so
            // moves which iteration a back edge's eval-breaker guard bails on.
            // Keep it only for a loop a merge can rebuild. An entry bridge
            // tail-calls another loop and stores none.
            reemit: std::cell::RefCell::new(entry_bridge_target.is_none().then_some(module_inputs)),
            bridge_owned_label_targets: std::cell::RefCell::new(Vec::new()),
            ca_active: std::cell::Cell::new(false),
            ca_terminal_declined: std::cell::Cell::new(false),
            ca_callers: std::cell::RefCell::new(Vec::new()),
        };

        token.set_compiled(Box::new(compiled));
        let compiled = token
            .compiled
            .get()
            .and_then(|compiled| compiled.downcast_ref::<CompiledWasmLoop>())
            .expect("newly compiled wasm loop is missing");
        // Native builds accept the encoded module as a test artifact. On the
        // real wasm host, eager compilation crossed the acceptance check
        // above; a deferred trace registers from `materialize_func_handle`.
        if !defer_host_compile || cfg!(not(target_arch = "wasm32")) {
            compiled.register_descrs_once();
        }
        compiled
            .resume_entry
            .slot
            .store(func_handle, std::sync::atomic::Ordering::Relaxed);
        // For a pending self target this is the exact map already embedded in
        // the module's CA arm. Reuse it for the published metadata so the
        // loop and its self-callee have demonstrably identical geometry. A
        // non-self loop still owns a freshly built map for future callers.
        let callee_gcmap_ptr = ca_targets
            .as_ref()
            .and_then(|targets| {
                targets
                    .iter()
                    .find(|(target_token, _)| *target_token == token.identity_key())
                    .map(|(_, target)| target.callee_gcmap_ptr)
            })
            .unwrap_or(home_gcmap_ptr as i64);
        // The module has now acquired its host-appended shared-table slot and
        // its finish index. Publish those mutable pieces before exposing the
        // immutable geometry metadata: previously compiled CALL_ASSEMBLER
        // modules load this stable entry at runtime.
        let has_guard_not_forced_2 = module_has_guard_not_forced_2(ops, &[]);
        publish_token_target(
            token,
            &CallAssemblerTarget {
                token_number: token.number(),
                dispatch_entry: 0,
                func_handle: compiled.eager_func_handle(),
                input_types: compiled.input_types.clone(),
                dispatch_key_ofs: compiled.frame.dispatch_key_ofs,
                callee_frame_bytes: compiled.frame.frame_bytes,
                callee_gcmap_ptr,
                compiled_ptr: compiled as *const CompiledWasmLoop as usize as u64,
                home_slot_base: compiled.frame.home_slot_base as u32,
                home_slots: compiled.frame.home_slots as u32,
                has_guard_not_forced_2,
                marked_ordinary: num_ref_homes as u32,
                marked_labels: used_label_homes as u32,
                label_ref_slots: compiled.frame.label_ref_slots as u32,
            },
        );
        if let Some(targets) = ca_targets.as_ref() {
            for (_, target) in targets {
                mark_call_assembler_target_active(target, token.invalidation_flag());
            }
        }

        diag_bump(24);
        Ok(AsmInfo {
            code_addr: 0,
            code_size,
        })
    }

    fn set_constants_pool(&mut self, constants: majit_ir::ConstMap<majit_ir::Const>) {
        self.constants.clear();
        for (&k, c) in constants.iter() {
            self.constants.insert(k, c.as_raw_i64());
        }
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.trace_counter = trace_id;
    }

    // `make_and_attach_done_descrs` — the FINISH fast path
    // needs the singletons' identity, so this backend takes the attachment
    // instead of the trait's no-op default. The cell address is what a wasm
    // frame stores in `jf_descr` (`CpuExitCells`).
    fn set_done_with_this_frame_descr_void(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        self.exit_cells
            .attach_finish(failguard::FINISH_EXIT_INDEX_VOID, descr);
    }

    fn set_done_with_this_frame_descr_int(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        self.exit_cells
            .attach_finish(failguard::FINISH_EXIT_INDEX_INT, descr);
    }

    fn set_done_with_this_frame_descr_ref(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        self.exit_cells
            .attach_finish(failguard::FINISH_EXIT_INDEX_REF, descr);
    }

    fn set_done_with_this_frame_descr_float(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        self.exit_cells
            .attach_finish(failguard::FINISH_EXIT_INDEX_FLOAT, descr);
    }

    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        self.exit_cells
            .attach_finish(failguard::FINISH_EXIT_INDEX_EXC, descr);
    }

    fn set_propagate_exception_descr(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        self.exit_cells.attach_propagate(descr);
    }

    fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = header_pc;
    }

    fn bridge_decline_is_terminal(&self) -> bool {
        // A structural `Unsupported` from `compile_bridge` is a shape
        // this backend rejects again. Trace-specific declines
        // (unpublished `CALL_ASSEMBLER`, an unchainable closing `JUMP`,
        // a loop-closing bridge that does not advance state, a
        // `COND_CALL` without a direct residual signature, a residual
        // call with more arguments than the call area, a headerless
        // nursery allocation, a read of an unproduced value, a
        // `GuardFutureCondition`, a `VirtualRef` the optimizer did
        // not lower, a raw load or store that missed the GC rewrite,
        // `GuardAlwaysFails`, an interior-field or string op the GC
        // rewrite did not consume) return
        // `CompilationFailed` so a later trace of the same guard can
        // still compile. `MetaInterp::compile_bridge` records the
        // guard only when this returns true and the error is `Unsupported`.
        true
    }

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
        original_token: &JitCellToken,
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError> {
        // Same close as `compile_loop`: bridge codegen reads `type_info_group`.
        majit_gc::ensure_type_registry_closed();
        // A bridge is a fresh trace that continues from a source loop's guard
        // exit. Instead of returning that guard's index to the host and
        // round-tripping through the interpreter, the source loop's epilogue
        // `call_indirect`s the bridge in-module (see `codegen` epilogue). The
        // bridge runs in the SOURCE loop's reused frame: the guard spilled its
        // fail args positionally into `frame[1..]`. `build_function` reads the
        // positional slot `k`, independently of the bridge value id, so no
        // argument-recovery layout is needed — hence `caller_recovery_layout`
        // and `previous_tokens` are unused.
        let ops_owned: Vec<Op> = normalize_ops_for_codegen(inputargs, ops);
        if let Some(reason) = missing_call_assembler_locs(&ops_owned) {
            diag_bump(1);
            // This trace's callee locs are unpublished. A later trace of
            // the same guard may publish them or omit CALL_ASSEMBLER.
            return Err(BackendError::CompilationFailed(reason));
        }
        // A bridge gets its own table, like `compile_loop`'s.
        let (ops_owned, gc_table) = self.rewrite_ops_for_gc(ops_owned);
        let gc_table_base = gc_table.as_ref().map_or(0, |t| t.base_addr() as u32);
        let ops: &[Op] = &ops_owned;
        diag_bump(0); // compile_bridge entered

        // is_loop=false: a bridge's terminal JUMP with no LABEL is a loop-closing
        // bridge whose re-entry target is plumbed via `external_jump_slot`.
        // Lift the CALL_ASSEMBLER decline when every callee target has frozen,
        // directly-enterable geometry; the CA arm lowers each operation to its
        // own in-module `call_indirect` target.
        // The CA arm must be able to complete a callee deopt; without the
        // registered `wasm_ca_resume_deopt` slot it could not, so decline the
        // lift (the host round-trip path still handles the CALL_ASSEMBLER).
        let ca_targets = bridge_call_assembler_target(ops);
        let ca_candidate = ca_deopt_helper_slot() != 0 && ca_targets.is_some();
        // The source guard this bridge attaches to. `fail_index` is its index in
        // the source loop's `fail_descrs` / cell array; `trace_id` identifies the
        // owning trace.
        let source_trace_id = fail_descr.trace_id();
        let source_fail_index = fail_descr.fail_index();

        // Scalars read from the source loop up front, so the immutable borrow of
        // `original_token` is released before the `&mut self` codegen calls.
        let (
            source_guard,
            source_func_handle,
            source_has_preamble,
            source_frame,
            is_direct,
            source_used_homes,
        ) = {
            let source_loop = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm backend: bridge source token has no compiled loop".into(),
                    )
                })?;
            let is_direct = source_trace_id == source_loop.trace_id;
            let fail_ptr = fail_descr as *const dyn FailDescr as *const ();
            let descrs = source_loop.fail_descrs.borrow();
            let by_meta = descrs.iter().find_map(|descr| {
                descr
                    .meta_descr
                    .as_ref()
                    .is_some_and(|meta| std::sync::Arc::as_ptr(meta) as *const () == fail_ptr)
                    .then(|| Arc::clone(descr))
            });
            let by_key = descrs.iter().find_map(|descr| {
                (descr.trace_id == source_trace_id && descr.fail_index == source_fail_index)
                    .then(|| Arc::clone(descr))
            });
            let wasm_guard = by_meta.or(by_key);
            let source_used_homes = wasm_guard
                .as_ref()
                .filter(|_descr| !is_direct)
                .map(|descr| (descr.trace_ref_homes, descr.trace_label_homes))
                .unwrap_or((
                    source_loop.num_ref_homes.get(),
                    source_loop.used_label_homes.get(),
                ));
            let guard = if is_direct {
                Some((
                    source_loop
                        .guard_fail_arg_advanced
                        .get(source_fail_index as usize)
                        .cloned()
                        .unwrap_or_default(),
                    source_loop
                        .guard_fail_arg_counts
                        .get(source_fail_index as usize)
                        .copied(),
                    source_loop.bridge_param_dispatch,
                ))
            } else {
                wasm_guard.as_ref().map(|descr| {
                    // Same live-position count the guard's parameter tail
                    // call baked (`live_fail_arg_count` / `rd_locs`), as
                    // `guard_fail_arg_counts` records for a direct guard. The
                    // advance-flag vec has one entry per fail arg, dead
                    // positions included, so its length is a different arity.
                    let n = descr.fail_arg_types.len();
                    (
                        descr.fail_arg_advanced.clone(),
                        Some(codegen::live_fail_arg_count(descr.meta_descr.as_ref(), n)),
                        descr.param_dispatch,
                    )
                })
            };
            (
                guard,
                source_loop.materialize_func_handle()?,
                source_loop.has_preamble,
                source_loop.frame,
                is_direct,
                source_used_homes,
            )
        };
        // The bridge cell hangs off `fail_descr.adr_jump_offset`, including
        // when the guard was emitted inside a nested trace. A missing side
        // table row is not a decline.
        let (source_fail_arg_advanced, source_fail_arg_count, source_bridge_param_dispatch) =
            source_guard.unwrap_or((Vec::new(), None, false));
        let source_cells_base = {
            let stamped = fail_descr.adr_jump_offset() as u32;
            if stamped != 0 {
                stamped
            } else {
                let fail_ptr = fail_descr as *const dyn FailDescr as *const ();
                original_token
                    .compiled
                    .get()
                    .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                    .and_then(|loop_| {
                        loop_.fail_descrs.borrow().iter().find_map(|descr| {
                            let same_meta = descr.meta_descr.as_ref().is_some_and(|meta| {
                                std::sync::Arc::as_ptr(meta) as *const () == fail_ptr
                            });
                            let same_key = descr.trace_id == source_trace_id
                                && descr.fail_index == source_fail_index;
                            (same_meta || same_key)
                                .then_some(descr.bridge_cell)
                                .filter(|&addr| addr != 0)
                        })
                    })
                    .unwrap_or(0)
            }
        };
        let bridge_entry_arity = if source_bridge_param_dispatch {
            if source_fail_arg_count != Some(inputargs.len()) {
                diag_bump(46);
                return Err(terminal_bridge_unsupported(
                    original_token,
                    "wasm backend: guard and bridge input arities differ",
                ));
            }
            Some(inputargs.len())
        } else {
            // BaseAssembler.rebuild_faillocs_from_descr binds only live
            // positions. The source guard spills them in that same compact
            // order, so frame entry input k reads physical slot k.
            if !codegen::frame_entry_reads_live_positions(fail_descr, inputargs.len()) {
                diag_bump(46);
                return Err(terminal_bridge_unsupported(
                    original_token,
                    "wasm backend: frame-entry bridge cannot address the source guard's \
                     live fail-arg slots",
                ));
            }
            None
        };
        let allow_ca = ca_candidate;
        if let Some(reason) = wasm_unsupported_trace_reason(ops, allow_ca) {
            diag_bump(1); // declined: CALL_ASSEMBLER
            // The unresolved callee is a property of these ops, not of
            // every future trace from the source guard.
            return Err(BackendError::CompilationFailed(reason));
        }
        if allow_ca {
            diag_bump(14); // accepted CALL_ASSEMBLER bridge
        }

        // A chained bridge executes in the source token's frame. Offsets of
        // that geometry stay put; slots past it are a tail (`FrameGeometry::extend`)
        // and `_check_frame_depth` grows the live frame at entry.
        let bridge_value_slots = codegen::frame_value_slots(inputargs, ops);
        let bridge_ref_homes = codegen::count_ref_homes(inputargs, ops);
        let frozen_frame = source_frame;
        let source_frame = frozen_frame.extend(bridge_value_slots, bridge_ref_homes);

        // A loop-closing bridge (terminal JUMP, no local LABEL) re-enters the
        // source loop through `source_func_handle` — the function entry. For a
        // peeled source loop, entering at the function entry re-runs the preamble
        // (the unrolled first iteration) against the bridge's mid-loop state
        // instead of resuming at the LABEL, so the induction variable never
        // advances: an infinite loop (the wasm chaining hang on nbody / fannkuch).
        //
        // A peeled loop carries the resume-at-LABEL dispatch: the loop-closing
        // JUMP arm sets the frame dispatch key to `target label ordinal + 1`,
        // so re-entering through `source_func_handle` `br_table`s to that
        // label's resume loader — chaining stays in-module. The bridge is
        // accepted when its JUMP's target label is recoverable from the descr,
        // the arities match, and the label's args are the complete live set of
        // the trace remainder (`label_resume_safe`); otherwise decline — the
        // guard then falls back to blackhole resume and
        // the guard descriptor's terminal bit stops the metainterp re-tracing it.
        // Non-peeled loops (entry == LABEL) re-enter correctly and keep
        // chaining.
        let bridge_is_loop_closing = has_cross_loop_terminal_jump(ops);
        if bridge_is_loop_closing {
            diag_bump(6); // loop-closing shape
        }
        if source_has_preamble {
            diag_bump(7); // source loop has preamble
        }
        let mut external_jump_key: u32 = 0;
        let mut external_jump_slot: u32 = source_func_handle;
        let mut external_jump_wide_slot: u32 = 0;
        let mut resumes_at_loop_header = false;
        let mut jump_target_frame: Option<codegen::FrameGeometry> = None;
        if bridge_is_loop_closing {
            let target =
                resolve_cross_loop_jump_target(ops, Some((source_func_handle, frozen_frame)));
            if let Some(t) = target {
                external_jump_key = t.key;
                external_jump_slot = t.func_handle;
                external_jump_wide_slot = t.wide_slot;
                resumes_at_loop_header = t.is_last_label;
                jump_target_frame = Some(t.frame);
            }
            if target.is_none() {
                diag_bump(2); // declined: JUMP target not chainable
                // The closing JUMP's target depends on the ops this trace
                // recorded. A later fail value can take a chainable path.
                return Err(BackendError::CompilationFailed(
                    "wasm backend: loop-closing bridge JUMP target is not a \
                     chainable published label"
                        .into(),
                ));
            }
        }

        // A loop-closing bridge carries the source loop's loop-carried state in
        // its terminal JUMP args and tail-calls the loop to iterate again. If no
        // JUMP arg is the result of a loop-state-advancing arithmetic or load op
        // — i.e. every loop-carried value is a verbatim input reload, a fresh
        // allocation, or a baked constant — the bridge re-presents byte-identical
        // induction/guard state on every pass, so the loop's exit guard never
        // flips and the loop⇄bridge cycle spins forever (a control-flow
        // livelock at constant stack depth and heap state). Such a bridge is a
        // guard side-trace that omits the loop body's advancing arithmetic; it
        // has no correct in-module resume, so decline it — the guard falls back
        // to blackhole resume and the guard descriptor's terminal bit stops the metainterp
        // re-tracing it. A genuinely advancing loop-closing bridge (an `i += 1`
        // counter feeding a JUMP arg) passes and keeps chaining.
        //
        // The check only concerns a bridge that lands directly AT the loop
        // header (the target's last label, or the entry of a non-peeled
        // loop): only then can the guard re-fail on byte-identical state. A
        // resume at an EARLIER label executes the segment between that label
        // and the header — the peeled iteration — which advances the state
        // before the loop re-runs, so no advance is required of the bridge
        // itself.
        if bridge_is_loop_closing && resumes_at_loop_header {
            // Bridge input position `k` reads frame slot `k`, where the source
            // guard spilled its k-th fail arg — so an `InputArg` JUMP arg is a
            // verbatim reload of source fail arg `k`. The advance for such an
            // arg may have happened in the SOURCE loop's body before the guard
            // (an `i += 1` preceding the failing branch): the source recorded
            // per-fail-arg whether the value was produced by a loop-state-
            // advancing op within the failing iteration
            // (`guard_fail_arg_advanced`), so consult that alongside the
            // in-bridge producers.
            let input_pos: std::collections::HashMap<u32, usize> = inputargs
                .iter()
                .enumerate()
                .map(|(k, ia)| (ia.index, k))
                .collect();
            let advances = ops
                .iter()
                .rev()
                .find(|op| op.opcode == majit_ir::OpCode::Jump)
                .is_some_and(|jump| {
                    jump.getarglist().iter().any(|arg| {
                        if let Some(producer) = arg.bound_op() {
                            advances_loop_state(producer.opcode)
                        } else if let Some(ia) = arg.bound_inputarg() {
                            input_pos.get(&ia.index).is_some_and(|&k| {
                                source_fail_arg_advanced.get(k).copied().unwrap_or(false)
                            })
                        } else {
                            false
                        }
                    })
                });
            // The JUMP hands input `k` back at position `k` only when it
            // re-presents the state the guard failed on unchanged. One that
            // reorders those inputs starts the next pass from a different state
            // vector, which is the same reasoning the heap carve-out below
            // uses: the shield refuses PROVABLY static bridges, and a permuted
            // state is not the byte-identical one the livelock argument rests
            // on. An arg that is not an input reload at all (a baked constant,
            // a fresh allocation) is static by itself and does not make the
            // JUMP a permutation.
            //
            // Reordering is not by itself enough, though. Read the arg list as
            // a map from JUMP position to the input position it reloads: the
            // state one pass produces is `s'[j] = s[source[j]]`, so a second
            // pass leaves it unchanged exactly when every position a source
            // names reloads ITSELF. `JUMP(input0, input0)` is the smallest
            // case — it moves slot 0 into slot 1 once and is a fixed point from
            // then on, so it re-presents byte-identical state and is refused
            // here as any verbatim reload is. A source chain that is not
            // stationary after one pass (a swap, a rotation) is admitted: its
            // orbit does have a finite period, but the shield is a static
            // approximation of the bridge alone — it does not model the loop
            // body that runs between two passes, which is where such a bridge's
            // advance actually comes from. Refusing one is not local to the
            // bridge either: the decline registers the guard in
            // `declined_bridge_guards`, which sends every later failure of it
            // to blackhole resume.
            let permutes_inputs = ops
                .iter()
                .rev()
                .find(|op| op.opcode == majit_ir::OpCode::Jump)
                .is_some_and(|jump| {
                    let sources: Vec<Option<usize>> = jump
                        .getarglist()
                        .iter()
                        .map(|arg| {
                            arg.bound_inputarg()
                                .and_then(|ia| input_pos.get(&ia.index).copied())
                        })
                        .collect();
                    sources.iter().any(|source| {
                        source.is_some_and(|k| sources.get(k).copied().flatten() != Some(k))
                    })
                });
            // Loop state carried on the HEAP (a permutation array flipped via
            // setarrayitem, an object field bumped via setfield, a residual
            // call's arbitrary effects) advances the cycle without any JUMP
            // arg showing inductive arithmetic. The shield only exists to
            // refuse PROVABLY static bridges, so any state-mutating op counts
            // as an advance.
            let mutates_heap = ops.iter().any(|op| {
                use majit_ir::OpCode::*;
                op.opcode.is_call()
                    || matches!(
                        op.opcode,
                        SetfieldGc
                            | SetfieldRaw
                            | SetarrayitemGc
                            | SetarrayitemRaw
                            | GcStore
                            | GcStoreIndexed
                            | RawStore
                            | Strsetitem
                            | Unicodesetitem
                    )
            });
            if !advances && !permutes_inputs && !mutates_heap {
                diag_bump(11); // declined: loop-closing bridge advances no loop-carried value
                // Whether this bridge advances state depends on the traced
                // path. Another fail value can still compile.
                return Err(BackendError::CompilationFailed(
                    "wasm backend: loop-closing bridge advances no loop-carried value \
                     (guard side-trace would livelock the chained loop)"
                        .into(),
                ));
            }
        }

        // A closing JUMP that names a LABEL of ANOTHER module cannot become a
        // `br`, so a region carrying one keeps the cross-module tail call its
        // out-of-line bridge made and only the ENTRY side is merged: the source
        // guard branches to the region's block with its values in locals
        // instead of storing them for a bridge call to read back.
        let region_external =
            (external_jump_slot != source_func_handle).then_some(codegen::ExternalJump {
                slot: external_jump_slot,
                key: external_jump_key,
                frame: jump_target_frame.unwrap_or(source_frame),
            });
        // Set by the inline block below to the owner of a merge candidate whose
        // merge waits on `INLINE_TRIP_THRESHOLD` entries into this bridge.
        let mut defer_inline: Option<(Arc<JitCellToken>, u32, bool, Option<(u64, u32)>)> = None;
        {
            // `model.py`: a bridge compiled after `invalidate_loop`
            // starts valid, and only a later invalidation activates its
            // GUARD_NOT_INVALIDATED (`runner_test.py test_guard_not_invalidated`
            // steps 3-4). A merged region reads the owner's root flag, which is
            // already set here, so it would be dead on arrival. Decline, and let
            // the out-of-line path mint the fresh flag that keeps the contract.
            // `(slot, key)` names the crossing this decline leaves in place,
            // which is what the trace-entry census counts.
            // By value: the id this compile will take, which the eager arm
            // below consumes before the last `decline` call is out of scope.
            let bridge_trace_id = self.trace_counter;
            // `source_fail_index` is the SOURCE TRACE's own exit ordinal, and
            // stays that everywhere else in this function: the dispatch cell
            // and `bridge_descr_ranges` are keyed by it, and the metainterp
            // re-derives that key off the FailDescr. The
            // region handed to the owner is the one thing that indexes the
            // merged stream instead.
            let merged_source_fail_index = if is_direct {
                // The owner's own guards come first in the merged stream, so a
                // direct guard's two numberings coincide.
                Some(source_fail_index)
            } else {
                original_token
                    .compiled
                    .get()
                    .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                    .and_then(|loop_| {
                        let inputs = loop_.reemit.borrow();
                        inputs.as_ref().and_then(|inputs| {
                            merged_region_fail_index(inputs, source_trace_id, source_fail_index)
                        })
                    })
            };
            let decline = |reason: &str| {
                record_inline_decline(format!(
                    "bridge={bridge_trace_id} src={source_trace_id} fi={source_fail_index} \
                     slot={external_jump_slot} key={external_jump_key} reason={reason}"
                ));
            };
            if original_token.is_invalidated() {
                diag_bump(50);
                decline("owner_invalidated");
            } else if merged_source_fail_index.is_none() {
                // Still out of line: the guard belongs to a bridge module of
                // its own, so the owner's stream holds no exit for it. A guard
                // in a region already merged into the owner is a different
                // case — its code was emitted from the merged stream, so it is
                // physically in this module and reachable by `br`.
                diag_bump(33);
                decline("not_direct");
                // The source guard is on a standalone parent. Arm a trip
                // that does not touch the owner's cells; install remaps
                // the fail index once that parent is in the merged stream.
                // Without the callback the probe would `call_indirect` slot
                // 0, so this stays a permanent out-of-line decline.
                if inline_trip_helper_slot() != 0
                    && bridge_is_loop_closing
                    && let Some(owner) = original_token
                        .compiled_loop_token()
                        .and_then(|clt| clt.upgrade_loop_token())
                {
                    // A remap is only useful if the parent is itself waiting
                    // to join this owner. A parent declined as
                    // `not_loop_closing` never enters PENDING, so the child
                    // would re-register forever.
                    let parent_pending = owner_pending_slot_addrs(&owner).into_iter().any(|addr| {
                        // SAFETY: `addr` was just read out of this owner's `pending_inlines`.
                        unsafe { pending_slot_at(addr) }
                            .entry
                            .borrow()
                            .as_ref()
                            .is_some_and(|item| {
                                item.same_owner(&owner) && item.region.trace_id == source_trace_id
                            })
                    });
                    if parent_pending {
                        defer_inline = Some((
                            owner,
                            0,
                            !resumes_at_loop_header,
                            Some((source_trace_id, source_fail_index)),
                        ));
                    }
                }
            } else if !bridge_is_loop_closing {
                diag_bump(34);
                decline("not_loop_closing");
            } else if let Some(candidate) = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .and_then(|loop_| loop_.reemit.borrow().as_ref().cloned())
            {
                // A guard in the peeled preamble cannot reach the loop-body
                // region blocks: the `loop` holding them has not been entered
                // there. `build_function` gives that class blocks of its own
                // outside the loop, and a body past the loop's `end` that only
                // the entry dispatch re-enters — so the header question below
                // is not asked of it.
                let source_in_preamble =
                    codegen::source_guard_precedes_loop_label(&candidate.ops, source_fail_index);
                if candidate
                    .inlined_bridges
                    .iter()
                    .any(|r| Some(r.source_fail_index) == merged_source_fail_index)
                {
                    diag_bump(36);
                    decline("already_owned");
                } else if !codegen::merged_stream_has_loop_label(&candidate) {
                    diag_bump(39);
                    decline("no_loop_label");
                } else if inline_trip_helper_slot() == 0 {
                    // Nothing to defer to: without the callback published the
                    // count could never be acted on, so the bridge stays out of
                    // line.
                    diag_bump(53);
                    decline("no_trip_helper");
                } else if let Some(owner) = original_token
                    .compiled_loop_token()
                    .and_then(|clt| clt.upgrade_loop_token())
                {
                    // Merging is append-only, so once one region takes the
                    // outside-the-loop placement every later one must too: its
                    // ops are the tail of the merged stream, and splicing an
                    // inside-loop region ahead of them would renumber the exits
                    // their sub-bridges' dispatch cells are keyed by. A
                    // loop-body guard can branch out to that placement, so this
                    // costs the later region its `br` to the header and not the
                    // merge.
                    let outside_loop = source_in_preamble
                        || !resumes_at_loop_header
                        || candidate.inlined_bridges.iter().any(|r| r.outside_loop);
                    // The `is_none` arm above already declined, so this holds.
                    let Some(merged_fail_index) = merged_source_fail_index else {
                        return Err(terminal_bridge_unsupported(
                            original_token,
                            "wasm backend: inline candidate without a merged-stream exit \
                             ordinal",
                        ));
                    };
                    self.collect_constants_from_ops(ops);
                    let has_invalidation_guard = codegen::has_invalidation_guard(ops);
                    let outside_labels_initialized =
                        !outside_loop || region_external.is_some() || {
                            let mut owner_ops = candidate.ops.clone();
                            for region in &candidate.inlined_bridges {
                                owner_ops.extend(region.ops.iter().cloned());
                            }
                            codegen::outside_region_labels_initialized(
                                &owner_ops,
                                merged_fail_index,
                                ops,
                            )
                        };
                    if !outside_labels_initialized {
                        // The owner stream may not yet include a sibling
                        // peel that publishes the JUMP target. Arm the
                        // trip: `install_pending_inline` re-checks against
                        // the owner as it stands then, and restores the
                        // out-of-line cell if the merge is still doomed.
                        defer_inline = Some((owner.clone(), merged_fail_index, outside_loop, None));
                        diag_bump(48);
                        decline("uninitialized_label");
                    } else if !has_invalidation_guard
                        || outside_loop
                        || region_external.is_some()
                        || compiled_wasm_loop(&owner).is_some_and(|loop_| {
                            loop_.module_bytes.get() > DEFAULT_INLINE_EAGER_MAX_BYTES
                        })
                    {
                        // compile.py::record_loop_or_bridge registers quasi-
                        // immutable dependencies on the whole JitCellToken.
                        // LoopInvalidation activates the root and all attached
                        // bridge flags, as llgraph/runner.py::invalidate_loop
                        // activates all traces. A mutation before installation
                        // makes install_inline_region_batch reject the owner;
                        // a later mutation activates the merged root guard.
                        // No per-bridge dependency needs moving at the trip.
                        // Eligible, but not yet worth its owner re-emission:
                        // arm the bridge's entry counter and merge when it
                        // trips. This applies equally to a header-resuming
                        // region: INLINE_TRIP_THRESHOLD is calibrated from
                        // bridge entries, and eager header merging otherwise
                        // bypasses that cost decision entirely. Everything else
                        // about this compile is the ordinary out-of-line path
                        // below.
                        defer_inline = Some((owner.clone(), merged_fail_index, outside_loop, None));
                        diag_bump(54);
                        decline("deferred");
                    } else {
                        // A small header region retains the existing eager
                        // cost policy. Invalidation itself is token-owned and
                        // does not require merging before compile returns.
                        let region = codegen::InlinedBridge {
                            source_fail_index: merged_fail_index,
                            external_jump: region_external.clone(),
                            outside_loop,
                            trace_id: self.trace_counter,
                            inputargs: inputargs.iter().cloned().collect(),
                            ops: ops_owned.clone(),
                            gc_table_base,
                            gc_const_keys: gc_const_keys_of(gc_table.as_deref()),
                            constants: self.constants.clone(),
                        };
                        if self.install_inline_region(&owner, region) {
                            self.trace_counter += 1;
                            if let Some(table) = gc_table {
                                Self::register_gc_table(original_token, table);
                            }
                            // The region has no code of its own: it was
                            // installed by rebuilding the owner, so there is no
                            // address to report. `model.py compile_bridge`
                            // permits `None` here, and the consumers treat the
                            // result as debug data — `interp_resop.py:253-255`
                            // defaults `asmaddr`/`asmlen` to 0 when it is
                            // absent — so a zero-address artifact says exactly
                            // "installed, but not as a block of its own".
                            return Ok(AsmInfo {
                                code_addr: 0,
                                code_size: 0,
                            });
                        }
                    }
                } else {
                    // The merge outlives this call and has to keep the owner
                    // alive; a token with no compiled-loop token has no strong
                    // handle to take.
                    diag_bump(53);
                    decline("no_owner_handle");
                }
            } else {
                diag_bump(35);
                decline("not_reemittable");
            }
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;
        let trace_entry_census =
            alloc_trace_entry_census(trace_id, codegen::entry_dispatch_key_count(ops));

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        let alloc = alloc_helpers();
        let wb = wasm_write_barrier_helpers();

        // CALL_ASSEMBLER: the CA arm allocates a fresh callee using the target
        // token's frozen geometry.
        // A keyed tail-call back into the source skips that module's
        // fresh-entry publish, so this map must cover the source trace's
        // already-initialized homes (the root loop, or the parent
        // chained bridge when this is a nested sub-bridge).
        let ca_params = if let Some(targets) = ca_targets.as_ref().filter(|_| allow_ca) {
            codegen::CaParams {
                emit_ca: true,
                targets: ca_codegen_targets(targets),
                deopt_helper_slot: ca_deopt_helper_slot(),
                ca_push_fn_ptr: wasm_jit_ca_push_frame as *const () as usize as i64,
                ca_pop_fn_ptr: wasm_jit_ca_pop_frame as *const () as usize as i64,
                ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                ca_reload_caller_fn_ptr: wasm_jit_ca_reload_caller_frame as *const () as usize
                    as i64,
                // See compile_loop: one shared inline path must fit every
                // per-op callee frame in this trace.
                inline: ca_inline_params(ca_max_frame_bytes(targets)),
                jf_top_addr: jf_top_addr(),
                compute_home_gcmap: true,
                home_gcmap_has_prior: true,
                home_gcmap_min_ordinary: source_used_homes.0,
                home_gcmap_min_labels: source_used_homes.1,
                ..codegen::CaParams::default()
            }
        } else {
            codegen::CaParams {
                ca_reload_fn_ptr: body_reload_fn_ptr(),
                jf_top_addr: jf_top_addr(),
                compute_home_gcmap: true,
                home_gcmap_has_prior: true,
                home_gcmap_min_ordinary: source_used_homes.0,
                home_gcmap_min_labels: source_used_homes.1,
                ..codegen::CaParams::default()
            }
        };

        // The region carries the trace id of the bridge standing in for it, so
        // a guard of this bridge that fails later resolves to its merged region
        // (`merged_region_fail_index`) when the owner is finally rebuilt.
        // Built before the probe so a rejected module drops the slot with
        // these resources instead of publishing it (`push_resources` is only
        // reached once the host has accepted the bridge).
        let mut asm_resources = release::LoopAsmResources::default();
        asm_resources.exit_cells = Some(std::sync::Arc::clone(&self.exit_cells));
        let inline_trip = defer_inline.map(|(owner, merged_fail_index, outside_loop, remap)| {
            if region_external.is_some() {
                diag_bump(51);
            }
            let region = codegen::InlinedBridge {
                source_fail_index: merged_fail_index,
                external_jump: region_external.clone(),
                // This region's own placement (`!resumes_at_loop_header` /
                // preamble) cannot change later. Install still ORs in any
                // outside sibling that landed first.
                outside_loop,
                trace_id,
                inputargs: inputargs.iter().cloned().collect(),
                ops: ops_owned.clone(),
                gc_table_base,
                gc_const_keys: gc_const_keys_of(gc_table.as_deref()),
                constants: self.constants.clone(),
            };
            // The owner's size prices this merge alone.
            let owner_module_bytes =
                compiled_wasm_loop(&owner).map_or(0, |loop_| loop_.module_bytes.get());
            register_pending_inline(&mut asm_resources, owner, region, owner_module_bytes, remap)
        });

        let guard_exit_count = codegen::guard_exit_count(inputargs, ops);
        let base = 0u32;
        // `rpython/jit/backend/model.py invalidate_loop`: a bridge compiled after an
        // invalidation starts valid; only a later invalidation may kill its
        // `GUARD_NOT_INVALIDATED` operations.
        let bridge_flag = original_token.mint_bridge_invalidation_flag();
        let (bridge_cells_base, bridge_cells_owner) = codegen::alloc_bridge_cells(guard_exit_count);
        let bridge_param_dispatch = bridge_param_dispatch_for(guard_exit_count);
        let mut module_inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.iter().cloned().collect(),
            ops: ops_owned.clone(),
            inlined_bridges: Vec::new(),
            constants: self.constants.clone(),
            vtable_offset: self.vtable_offset,
            classptr_to_typeid: typeid_table,
            guard_gc_type_info,
            alloc,
            wb,
            nursery: nursery_alloc_params(ops),
            invalidated_flag_addr: Arc::as_ptr(&bridge_flag) as usize as u32,
            gc_table_base,
            gc_const_keys: gc_const_keys_of(gc_table.as_deref()),
            fail_index_base: base,
            bridge_cells_base,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity,
            bridge_param_dispatch,
            trace_entry_census,
            inline_trip,
            external_jump_slot,
            external_jump_key,
            external_jump_wide_slot,
            frame: source_frame,
            ca: ca_params,
        };
        // `assembler.py` `_assemble` takes `max(frame_depth, target jfi_frame_depth)`.
        // `assemble_bridge` always emits `_check_frame_depth` and
        // `update_frame_depth` on the source token. The bridge keeps
        // `frozen_frame.extend` for its own spills; the target's slot counts
        // are not appended. Host and CALL_ASSEMBLER entries allocate
        // `jfi_frame_depth` (`ca_frame_depth` = the full signed item count).
        let mut required_items = source_frame.signed_item_count();
        if let Some(target_frame) = jump_target_frame {
            module_inputs.ca.external_jump_frame = Some(target_frame);
            required_items = required_items.max(target_frame.signed_item_count());
        }
        module_inputs.ca.realloc_fn_ptr = wasm_realloc_frame as *const () as usize as i64;
        module_inputs.ca.frame_depth_items = required_items;
        module_inputs.ca.exit_table_base = asm_resources.alloc_exit_table(guard_exit_count) as u32;
        module_inputs.ca.gcmap_sink = &mut asm_resources as *mut _ as usize;
        // `runner.rs` captures `AttachedDescrPtrs` at `compile_bridge` entry.
        module_inputs.ca.attached = self.exit_cells.descr_ptrs();
        let (wasm_bytes, guard_exits, _num_ref_homes, _used_labels) =
            match codegen::build_wasm_module(&module_inputs) {
                Ok(built) => built,
                Err(err) => {
                    let err = bridge_codegen_error(err);
                    record_last_compile_err(&err);
                    if matches!(err, BackendError::Unsupported(_)) {
                        note_call_assembler_terminal_decline_for_token(original_token);
                    }
                    return Err(err);
                }
            };

        // Bridge exit descrs (fail_index already base-offset by build_wasm_module).
        let bridge_advanced = guard_fail_args_advanced(ops, &guard_exits);
        let bridge_ref_floor = bridge_ref_homes.max(source_used_homes.0);
        let bridge_label_floor =
            codegen::label_ref_capture_slots(inputargs, ops).max(source_used_homes.1);
        let bridge_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .enumerate()
            .map(|(index, g)| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    fail_locs: g.fail_locs.clone(),
                    is_finish: g.is_finish,
                    force_args_offset: source_frame.force_slot_base as u32,
                    force_tail_base: force_tail_fields(source_frame).0,
                    force_prefix_slots: force_tail_fields(source_frame).1,
                    value_tail_index: force_tail_fields(source_frame).2,
                    force_gcmap_ptr: g.exit_gcmap_ptr,
                    bridge_cell: g.bridge_cell,
                    fail_arg_advanced: bridge_advanced.get(index).cloned().unwrap_or_default(),
                    trace_ref_homes: bridge_ref_floor,
                    trace_label_homes: bridge_label_floor,
                    param_dispatch: bridge_param_dispatch,
                    bridge_slot: std::sync::atomic::AtomicU32::new(0),
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();
        // Register the bridge module into the shared table, then publish its
        // descrs and flip the source guard's cell. Order matters: the descrs
        // must be resolvable (appended) before the cell makes the guard dispatch
        // into the bridge.
        #[cfg(target_arch = "wasm32")]
        let bridge_slot = glue::compile_module(&wasm_bytes);
        #[cfg(not(target_arch = "wasm32"))]
        let bridge_slot = 0u32;
        // A 0 handle means the host rejected the bridge module (see the
        // `compile_loop` decline). Flipping the source guard's cell to dispatch
        // into slot 0 would tail-call a non-trace; decline instead so the guard
        // keeps its host round-trip (correct, unaccelerated).
        #[cfg(target_arch = "wasm32")]
        if bridge_slot == 0 {
            // The host rejected this trace's module. A later failure can
            // record a smaller bridge, so the guard stays retryable.
            return Err(BackendError::CompilationFailed(
                "wasm host rejected the compiled bridge module (oversized function body \
                 or invalid module)"
                    .to_string(),
            ));
        }
        patch_pending_failure_recoveries(&guard_exits);
        // `assemble_bridge`: `update_frame_depth` on the source token, so a
        // later entry allocates the depth this bridge and its JUMP target need.
        if let Some(clt) = original_token.compiled_loop_token() {
            let baseofs = (majit_gc::header::GcHeader::SIZE
                + majit_backend::jitframe::FIRST_ITEM_OFFSET) as i64;
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, required_items as i64);
        }
        // The host accepted the bridge. Only now publish its global exit
        // descriptors and attach their resume-data tracer to the source CLT;
        // a rejected module can never execute and must retain neither.
        publish_exit_slots(
            &mut asm_resources,
            &guard_exits,
            &bridge_descrs,
            &module_inputs.ca.attached,
        );
        Self::register_meta_descrs(original_token, &bridge_descrs);
        // Only a bridge that survived the decline above gets its reference
        // constants rooted. The table is attached to the long-lived original
        // loop token, so rooting a rejected bridge's table would keep its
        // constants alive permanently, once per rejected attempt.
        if let Some(table) = gc_table {
            Self::register_gc_table(original_token, table);
        }
        diag_bump(5); // bridge compiled — chained in-module
        if bridge_entry_arity.is_some() {
            diag_bump(44); // bridge compiled with a parameter entry
        }

        // x86/assembler.py:706 publishes the target tokens defined by an
        // accepted bridge. `codegen::is_resumable_peeled` and
        // `codegen::resumable_label_count` both use `find_loop_label_index`. A
        // retrace closing onto its OWN new target token resolves the terminal
        // JUMP among this trace's LABELs and is peeled: codegen emitted the
        // resume `br_table`, and every resumable label is published at key
        // ordinal + 1. A `jump_to_preamble` retrace closes onto the ORIGINAL
        // loop's start descr, so it is not peeled and its first op is not a
        // LABEL; the
        // existing `first_label_at_entry` / arity guard correctly leaves that
        // label unpublished, because key 0 would re-run the work before it.
        let (_, published_label_descrs) = stamp_and_publish_label_targets(
            &mut asm_resources,
            bridge_slot,
            source_frame,
            inputargs,
            ops,
            bridge_entry_arity,
            original_token.identity_key(),
        );
        if bridge_slot != 0 {
            asm_resources.table_slots.push(bridge_slot);
        }
        if let Some(cells) = bridge_cells_owner {
            asm_resources.bridge_cells.push(cells);
        }
        release::push_resources(original_token, asm_resources);

        {
            let source_loop = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .expect("source loop disappeared between borrows");
            // Append the bridge's exit descrs to the source loop's flat
            // `fail_descrs` and record the slice they occupy, keyed by the
            // source guard's `fail_index`. `compiled_bridge_fail_descr_layouts`
            // maps a source guard back to that range. `start` is captured inside the same `borrow_mut`
            // critical section as the `extend`, so the range stays in lockstep
            // with the vec.
            let count = bridge_descrs.len();
            {
                let mut descrs = source_loop.fail_descrs.borrow_mut();
                let start = descrs.len();
                descrs.extend(bridge_descrs);
                source_loop.bridge_descr_ranges.borrow_mut().push((
                    source_trace_id,
                    source_fail_index,
                    start,
                    count,
                ));
            }
            // The bridge module lives as long as this source loop, so hand its
            // own cell array (if any) to the loop, freed when the loop drops.
            source_loop.bridge_owned_label_targets.borrow_mut().extend(
                published_label_descrs
                    .into_iter()
                    .map(|descr| (descr, bridge_slot)),
            );
            if let Some(targets) = ca_targets.as_ref().filter(|_| allow_ca) {
                // Freeze this recursion to the CA mechanism: no further bridge
                // chains here (see the decline above the codegen call).
                for (_, target) in targets {
                    mark_call_assembler_target_active(target, bridge_flag.clone());
                }
            }
        }

        // CA dispatch diagnostics (guest `eprintln` is a no-op on wasm32, so
        // route through the BRIDGE_DIAG tallies the host surfaces): 12 = CA bridge
        // cell actually written (loop epilogue will tail into it); 13 = CA bridge
        // but the source loop reserved no bridge cells (cells_base 0) so the guard
        // never dispatches in-module — the recursion stays a host round-trip.
        if allow_ca {
            if source_cells_base != 0 && bridge_slot != 0 {
                diag_bump(12);
            } else {
                diag_bump(13);
            }
        }
        // The same question for every accepted bridge (slots 27/28): a bridge
        // whose source guard has no cell is compiled and then unreachable.
        if source_cells_base != 0 && bridge_slot != 0 {
            diag_bump(27);
        } else {
            diag_bump(28);
        }
        // Arm the GNF2 flag before the source guard cell becomes the
        // new bridge. An out-of-line GNF2 bridge runs in the source
        // loop's CA frame; a CALL_ASSEMBLER already inside the callee
        // can finish through this cell as soon as it is written.
        if ops
            .iter()
            .any(|op| op.opcode == majit_ir::OpCode::GuardNotForced2)
        {
            mark_gnf2_token(original_token);
            if let Some(mut target) = target_from_token(original_token) {
                if target.has_guard_not_forced_2 == 0 {
                    target.has_guard_not_forced_2 = 1;
                    publish_token_target(original_token, &target);
                }
            }
        }
        // `assembler.py` `patch_jump_for_descr`: write the bridge slot through
        // `faildescr.adr_jump_offset`, then clear the slot ("patched").
        let patch_cell = {
            let stamped = fail_descr.adr_jump_offset() as u32;
            if stamped != 0 {
                stamped
            } else {
                source_cells_base
            }
        };
        #[cfg(target_arch = "wasm32")]
        if patch_cell != 0 && bridge_slot != 0 {
            if unsafe { core::ptr::read(patch_cell as *const u32) } != 0 {
                diag_bump(29);
            }
            crate::failguard::write_guard_cell(patch_cell, bridge_slot);
            fail_descr.set_adr_jump_offset(0);
            if let Some(source_loop) = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            {
                if let Some(descr) = source_loop.fail_descrs.borrow().iter().find(|descr| {
                    descr.trace_id == source_trace_id && descr.fail_index == source_fail_index
                }) {
                    descr
                        .bridge_slot
                        .store(bridge_slot, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = (patch_cell, bridge_slot, source_cells_base);

        let code_size = wasm_bytes.len();
        // `asmmemmgr.py:37`, as in `compile_loop` above: a bridge's module is a
        // block of its own.
        let block = self.asm_memory_stats.record_block(code_size, code_size);
        self.asm_memory_blocks.borrow_mut().push(block);

        Ok(AsmInfo {
            code_addr: 0,
            code_size,
        })
    }

    /// `compile.py` store_hash relies on a per-guard fail-descr layout
    /// to know which exits are real guards (vs FINISH) and to count them.
    /// `assign_guard_hashes` fetches one jitcounter hash per non-finish guard
    /// from this list, so without it no guard ever gets a hash, `must_compile`
    /// never fires, and a hot guard exit round-trips to the host forever instead
    /// of triggering a bridge. Build one layout per exit from the metainterp
    /// `ResumeGuardDescr` the optimizer stamped on the guard (`meta_descr`); the
    /// wasm backend keeps no machine-code recovery metadata (resume runs through
    /// the frontend `WasmFrameData` path), so the recovery / rd_* / gc-slot
    /// fields stay empty — `merge_backend_exit_layouts` keeps the frontend's own
    /// entry (`or_insert_with`) and only consumes `is_finish` + `source_op_index`.
    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())?;
        let trace_id = compiled.trace_id;
        let descrs = compiled.fail_descrs.borrow();
        let layouts = descrs
            .iter()
            .enumerate()
            .map(|(position, wfd)| {
                let meta = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
                majit_backend::FailDescrLayout {
                    fail_index: position as u32,
                    source_op_index: meta.and_then(|fd| fd.source_op_index()),
                    trace_id,
                    trace_info: None,
                    fail_arg_types: wfd.fail_arg_types.clone(),
                    is_finish: wfd.is_finish,
                    is_exception_exit: meta
                        .map(|fd| fd.is_exit_frame_with_exception())
                        .unwrap_or(false),
                    recovery_layout: None,
                    frame_stack: None,
                    descr: wfd.meta_descr.clone(),
                }
            })
            .collect();
        Some(layouts)
    }

    /// `compile.py` store_hash: stamp the jitcounter hashes assigned by
    /// `compile_bridge` appends a bridge's exit descrs to the source loop's flat
    /// `fail_descrs` and records their `(source_fail_index, start, count)` slice
    /// in `bridge_descr_ranges`. Return one layout per descr in that slice.
    /// `fail_index` is the 0-based position within the bridge's own
    /// exit list (matching the bridge's frontend `exit_layouts` keying and the
    /// native backends' `compiled_bridge_fail_descr_layouts`); `trace_id` is the
    /// bridge's own id, stamped on each appended `WasmFailDescr`.
    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = original_token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())?;
        // The most recently chained bridge at this source guard (last range).
        let (start, count) = compiled
            .bridge_descr_ranges
            .borrow()
            .iter()
            .rev()
            .find(|r| r.0 == source_trace_id && r.1 == source_fail_index)
            .map(|&(_, _, start, count)| (start, count))?;
        let descrs = compiled.fail_descrs.borrow();
        let layouts = descrs
            .get(start..start + count)?
            .iter()
            .enumerate()
            .map(|(position, wfd)| {
                let meta = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
                majit_backend::FailDescrLayout {
                    fail_index: position as u32,
                    source_op_index: meta.and_then(|fd| fd.source_op_index()),
                    trace_id: wfd.trace_id,
                    trace_info: None,
                    fail_arg_types: wfd.fail_arg_types.clone(),
                    is_finish: wfd.is_finish,
                    is_exception_exit: meta
                        .map(|fd| fd.is_exit_frame_with_exception())
                        .unwrap_or(false),
                    recovery_layout: None,
                    frame_stack: None,
                    descr: wfd.meta_descr.clone(),
                }
            })
            .collect();
        Some(layouts)
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        // Key 0 is the peeled preamble. `execute_token_with_dispatch_key`
        // is the same entry with a caller-supplied LABEL selector.
        self.execute_token_with_dispatch_key(token, args, 0)
    }

    /// `Backend::execute_token_with_dispatch_key`. Key 0 runs the preamble.
    /// Key `label_block_id + 1` is LABEL n: `stamp_and_publish_label_targets`
    /// stores that ordinal in `LoopTargetDescr::label_block_id`, and
    /// `build_function`'s entry `br_table` lands key `j + 1` on label j's
    /// resume loader. The loader reads label arg i from `spill_slot_ofs(i)`,
    /// the same slots this entry already fills. An out-of-range key takes the
    /// `br_table` default, which is the preamble — `compiler.rs`
    /// `execute_with_inputs_at_dispatch_key` does the same with its jump table.
    fn execute_token_with_dispatch_key(
        &self,
        token: &JitCellToken,
        args: &[Value],
        dispatch_key: u32,
    ) -> DeadFrame {
        let compiled = token
            .compiled
            .get()
            .expect("no compiled code")
            .downcast_ref::<CompiledWasmLoop>()
            .expect("not CompiledWasmLoop");
        compiled
            .frame
            .debug_assert_dispatch_entry_slots(args.len() as u64);
        #[cfg(target_arch = "wasm32")]
        let func_handle = compiled
            .materialize_func_handle()
            .expect("wasm backend failed to materialize a runnable trace");

        // Host entry allocates the complete frozen geometry, including the
        // tail call area. CALL_ASSEMBLER uses the same `jfi_frame_depth`.
        let frame_size = entry_frame_bytes(token, compiled.frame.frame_bytes).div_ceil(8);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (frame_size, args, dispatch_key);
            panic!("wasm backend execute_token requires a wasm host");
        }
        #[cfg(target_arch = "wasm32")]
        {
            // `malloc_entry_jitframe` (`llmodel.py` `malloc_jitframe` /
            // `jitframe.py` `JITFRAME.allocate`). A collecting allocation
            // forwards Ref homes through `jf_gcmap` on the jitframe shadow
            // stack. The data region (fail_index at 0, inputs/outputs at
            // FRAME_SLOT_BASE, call area, dispatch key, Ref homes) lives in
            // the `jf_frame` items area; passing `jf + FIRST_ITEM_OFFSET` as
            // the wasm frame pointer keeps every local-0-relative codegen
            // access unchanged. (See `build_home_gcmap` for the wasm32
            // Signed-item layout.) After `glue::execute`, resolve the shadow
            // slot before the write barrier: the nursery frame may have moved.
            if wasm_jitframe_tid() != 0 {
                use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame};
                let sign = std::mem::size_of::<isize>();
                // Data region (frame_size i64 slots) expressed in Signed items.
                let depth = frame_size * 8 / sign;
                let alloc_size = JitFrame::alloc_size(depth);
                // `alloc_entry_jitframe` roots input refs across the nursery
                // malloc. Rust's stack is not traced; the stores below read
                // the forwarded values back from those roots.
                let ref_roots: Vec<majit_gc::shadow_stack::OwnerRootGuard> = args
                    .iter()
                    .filter_map(|arg| match arg {
                        Value::Ref(value) => {
                            Some(majit_gc::shadow_stack::OwnerRootGuard::new(*value))
                        }
                        _ => None,
                    })
                    .collect();
                let jf = with_wasm_active_gc_mut(|gc| {
                    majit_backend::jitframe::malloc_entry_jitframe(gc, alloc_size)
                })
                .expect("wasm JitFrame allocation failed");
                // Nursery and old-gen arenas both recycle bytes.
                // `malloc_entry_jitframe` clears one header byte; a home the
                // trace has not defined yet must still read as null.
                unsafe {
                    std::ptr::write_bytes(jf as *mut u8, 0, alloc_size);
                    JitFrame::init(jf, frame_info_ptr(token), depth);
                }

                // `jf_gcmap` stays null until the trace executes `push_gcmap`.
                // A virtualizable token that outlives the call keeps
                // `_finish_gcmap` via the guard's exit-table store.
                unsafe { (*jf).jf_gcmap = std::ptr::null() };

                let items_base = jf as usize + FIRST_ITEM_OFFSET;
                let mut ref_index = 0usize;
                for (i, arg) in args.iter().enumerate() {
                    let v = match arg {
                        Value::Int(v) => *v,
                        Value::Float(v) => v.to_bits() as i64,
                        Value::Ref(_) => {
                            let current = ref_roots[ref_index].get();
                            ref_index += 1;
                            current.0 as i64
                        }
                        Value::Void => 0,
                    };
                    let ofs = compiled.frame.spill_slot_ofs(i as u64) as usize;
                    unsafe { *((items_base + ofs) as *mut i64) = v };
                }
                drop(ref_roots);
                unsafe {
                    *((items_base + compiled.frame.dispatch_key_ofs as usize) as *mut i64) =
                        i64::from(dispatch_key);
                }
                // llmodel.py `execute_token`: `llop.gc_writebarrier(ll_frame)`
                // after the input stores. A frame born old (oversized, or a
                // refused young rawmalloc) now holds young refs.
                wasm_jit_write_barrier(jf as i64);

                let saved = majit_gc::shadow_stack::push_jf(majit_ir::GcRef(jf as usize));
                {
                    let _exec = ExecutingBackendGuard::enter(self);
                    glue::execute(func_handle, items_base as u32);
                }

                // The pre-call pointer may name from-space. Resolve the
                // shadow slot first (`_check_frame_depth` may also have
                // stored `jf_forward`). No barrier on the way out: the entry
                // barrier above and `_reload_frame_if_necessary`'s after each
                // collecting call already left an old frame remembered.
                let jf = unsafe {
                    majit_backend::jitframe::JitFrame::resolve(
                        majit_gc::shadow_stack::peek_jf(saved).0 as *mut JitFrame,
                    )
                };
                let fail_descr = descr_at(unsafe { (*jf).jf_descr })
                    .expect("invalid jf_descr from compiled wasm");
                let data = WasmFrameData::from_live_frame(jf, fail_descr, false, true, None);
                majit_gc::shadow_stack::pop_jf_to(saved);

                return DeadFrame::Boxed(data);
            }

            // Host-buffer frame path, for an embedder that registered no
            // `JitFrame` type id: fail_index at item[0], inputs/outputs at
            // item[1 + i]. A home slot only ever holds null (entry init) or a
            // valid GcRef (store-on-def), so forwarding is safe. No collection
            // moves this buffer; a body reload simply reads the same stack
            // root. The release below is straight-line and the wasm32 build is
            // `panic=abort`, so `glue::execute` cannot unwind past the pop.
            //
            // One root mechanism: the off-GC frame is published on the jitframe
            // shadow stack for the span of the call. Ref homes are roots only
            // while `push_gcmap` has published `jf_gcmap` (`assembler.py`).
            // `llmodel.py`
            // `execute_token` allocates through `malloc_jitframe` and
            // `jitframe.py` `jitframe_trace` walks that map; dynasm `runner.rs`
            // `execute_token` runs the same off-GC frame, pushed by the
            // prologue (`gen_shadowstack_header`). A minor
            // (`MiniMarkGC::minor_collection_body`) or major
            // (`walk_stack_shaped_roots`) collection traces a
            // `register_libc_jitframe` entry through the libc-jitframe tracer,
            // which applies `jf_gcmap`.
            let sign = std::mem::size_of::<isize>();
            let depth = frame_size * 8 / sign;
            let alloc_size = majit_backend::jitframe::JitFrame::alloc_size(depth);
            // Off-GC storage so a FINISH that returns the force token can
            // hand the same block to `LibcJitFrameDeadFrame::owning`. A
            // `Vec` on this stack would free the token's JitFrame.
            let jf = majit_backend::jitframe::alloc_off_gc_jitframe(alloc_size);
            assert!(!jf.is_null(), "wasm host-buffer JitFrame allocation failed");
            unsafe { majit_backend::jitframe::JitFrame::init(jf, frame_info_ptr(token), depth) };
            unsafe { (*jf).jf_gcmap = std::ptr::null() };
            let items_base = jf as usize + majit_backend::jitframe::FIRST_ITEM_OFFSET;
            for (i, arg) in args.iter().enumerate() {
                let v = match arg {
                    Value::Int(v) => *v,
                    Value::Float(v) => v.to_bits() as i64,
                    Value::Ref(r) => r.0 as i64,
                    Value::Void => 0,
                };
                let ofs = compiled.frame.spill_slot_ofs(i as u64) as usize;
                unsafe { *((items_base + ofs) as *mut i64) = v };
            }
            unsafe {
                *((items_base + compiled.frame.dispatch_key_ofs as usize) as *mut i64) =
                    i64::from(dispatch_key);
            }
            majit_gc::shadow_stack::register_libc_jitframe(jf as usize);
            let saved = majit_gc::shadow_stack::push_jf(GcRef(jf as usize));
            {
                let _bh_phase = majit_gc::BhProbePhase::enter("compiled");
                {
                    let _exec = ExecutingBackendGuard::enter(self);
                    glue::execute(func_handle, items_base as u32);
                }
            }
            let tip = unsafe { majit_backend::jitframe::JitFrame::resolve(jf) };
            let tip_depth =
                unsafe { majit_backend::jitframe::JitFrame::frame_length(tip) as usize };
            majit_gc::shadow_stack::pop_jf_to(saved);
            let fail_descr =
                descr_at(unsafe { (*tip).jf_descr }).expect("invalid jf_descr from compiled wasm");
            // FINISH(force_token) parks this JitFrame pointer in the frame.
            // Own the off-GC block so a later `force` does not dereference
            // a freed frame. Fail args stay in the frame.
            let owner = unsafe {
                majit_backend::libc_deadframe::LibcJitFrameDeadFrame::owning(
                    jf,
                    tip,
                    tip_depth,
                    majit_backend::deadframe::ExitDescr::owned(
                        fail_descr.clone() as majit_ir::DescrRef
                    ),
                    None,
                )
            };
            DeadFrame::Boxed(WasmFrameData::from_live_frame(
                tip,
                fail_descr,
                false,
                false,
                Some(owner),
            ))
        }
    }

    fn supports_dispatch_key_entry(&self) -> bool {
        true
    }

    /// `LabelTarget::requires_own_frame` is set when that LABEL's resume
    /// loader reads a `LabelResumeData` capture slot. A fresh host frame
    /// never stored those slots, so the key is refused. Key 0 is the
    /// preamble and reads no capture. A key with no LABEL loader published by
    /// this token is refused too: `build_function`'s `br_table` default is
    /// the preamble, which does not take compact LABEL arguments.
    fn supports_dispatch_key_entry_for(&self, token: &JitCellToken, dispatch_key: u32) -> bool {
        if !self.supports_dispatch_key_entry() || dispatch_key == 0 {
            return self.supports_dispatch_key_entry();
        }
        let Some(compiled) = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return false;
        };
        for descr in &compiled.published_label_descrs {
            let Some(target) = label_target(descr) else {
                continue;
            };
            if target.owner_token != compiled.token_number || target.key != dispatch_key {
                continue;
            }
            return !target.requires_own_frame;
        }
        false
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    fn get_latest_descr<'a>(&self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        // The same selection as `get_latest_descr_arc` below: the metainterp
        // descr when the optimizer stamped one, since that is the object
        // carrying `is_exit_frame_with_exception` / `get_status` /
        // `rd_loop_token_clt`, and the backend descr only for synthetic exits.
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        data.fail_descr
            .meta_descr
            .as_ref()
            .and_then(|meta| meta.as_fail_descr())
            .unwrap_or(data.fail_descr.as_ref())
    }

    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn majit_ir::Descr> {
        // `history.py:125` parity — when the optimizer stamped a
        // metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
        // `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
        // `op.descr`, the wasm backend snapshotted it into
        // `WasmFailDescr.meta_descr`.  Forward through that Arc so
        // identity (`Arc::ptr_eq`) matches dynasm/cranelift; otherwise
        // fall back to the backend Arc upcast (synthetic backend-only
        // descrs).
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        if let Some(meta) = data.fail_descr.meta_descr.as_ref() {
            return Arc::clone(meta);
        }
        Arc::clone(&data.fail_descr) as Arc<dyn majit_ir::Descr>
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        wasm_frame_word(wasm_frame_data(frame), index)
    }

    fn get_value_direct(&self, frame: &DeadFrame, slot: usize) -> i64 {
        // The counter stamp is a logical fail-arg index, the same space
        // `get_int_value` reads. A live frame decodes it through fail_locs.
        self.get_int_value(frame, slot)
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        f64::from_bits(self.get_int_value(frame, index) as u64)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        GcRef(self.get_int_value(frame, index) as usize)
    }

    /// `llmodel.py` `grab_exc_value`: `deadframe.jf_guard_exc`.
    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        let data = wasm_frame_data(frame);
        if let Some(base) = data.items_base() {
            let jf = (base - majit_backend::jitframe::FIRST_ITEM_OFFSET)
                as *const majit_backend::jitframe::JitFrame;
            return GcRef(unsafe { (*jf).jf_guard_exc });
        }
        GcRef(data.exc_value as usize)
    }

    fn set_savedata_ref(&self, frame: &mut DeadFrame, data: GcRef) {
        let wasm = frame
            .boxed_data_mut()
            .and_then(|d| d.downcast_mut::<WasmFrameData>())
            .expect("not WasmFrameData");
        wasm.set_savedata(data);
    }

    fn get_savedata_ref(&self, frame: &DeadFrame) -> Option<GcRef> {
        let data = wasm_frame_data(frame);
        let word = if let Some(base) = data.items_base() {
            let jf = (base - majit_backend::jitframe::FIRST_ITEM_OFFSET)
                as *const majit_backend::jitframe::JitFrame;
            unsafe { (*jf).jf_savedata }
        } else {
            data.savedata as usize
        };
        let r = GcRef(word);
        if r.is_null() { None } else { Some(r) }
    }

    fn clear_stored_exception(&self) {
        crate::jit_exc_clear();
    }

    fn free_loop(&mut self, token: &JitCellToken) {
        // `llmodel.py` `AbstractLLCPU.free_loop_and_bridges`. Dropping
        // `asmmemmgr_blocks` releases this loop's wasm resources, including
        // the CALL_ASSEMBLER cell whose address is `_ll_function_addr`.
        // Clear that stand-in first so a later lookup does not load it.
        token.set_ll_function_addr(0);
        if let Some(clt) = token.compiled_loop_token() {
            clt.free_loop_and_bridges();
        }
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        // A validated wasm module's code is immutable, so
        // GUARD_NOT_INVALIDATED loads a live flag instead of having its
        // instruction bytes patched in place — the same shape the llgraph
        // backend uses (`llgraph/runner.py invalidate_loop` sets `trace.invalid` across
        // `_llgraph_alltraces`). `model.py:145` covers the loop AND its
        // attached bridges, each of which reads its own generation flag, so
        // this must go through `invalidate` rather than store to the root
        // flag alone.
        token.invalidate();
    }

    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        // `x86/assembler.py` `redirect_call_assembler` copies frame info, then
        // patches the target at `oldlooptoken._ll_function_addr`. It does not
        // refuse. The wasm cell is that address; callers already baked it.
        if let (Some(new_clt), Some(old_clt)) =
            (new.compiled_loop_token(), old.compiled_loop_token())
        {
            let baseofs = (majit_gc::header::GcHeader::SIZE
                + majit_backend::jitframe::FIRST_ITEM_OFFSET) as i64;
            let old_weak = std::sync::Arc::downgrade(&old_clt);
            new_clt.update_frame_info(&old_clt, old_weak, baseofs);
        }
        if let Some(mut new_target) = target_from_token(new) {
            if new_target.func_handle == 0 && new_target.compiled_ptr != 0 {
                if let Some(loop_) =
                    unsafe { (new_target.compiled_ptr as *const CompiledWasmLoop).as_ref() }
                {
                    if let Ok(handle) = loop_.materialize_func_handle() {
                        new_target.func_handle = handle;
                        publish_token_target(new, &new_target);
                    }
                }
            }
            let old_target = target_from_token(old);
            publish_token_target(old, &new_target);
            if let Some(old_target) = old_target.as_ref() {
                transfer_call_assembler_target_activity(old_target, &new_target);
            }
        }
        majit_backend::redirect_assembler(old, new, new.number() as u64);
        Ok(())
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_rewrite_interior_field_decline_stays_retryable() {
        let reason = "wasm codegen: GetinteriorfieldGcI reached codegen without the GC rewrite";
        assert!(bridge_unsupported_is_retryable(reason));
        assert!(bridge_unsupported_is_retryable(
            "wasm codegen: Copyunicodecontent reached codegen without the GC rewrite"
        ));
        assert!(!bridge_unsupported_is_retryable(
            "wasm codegen: some other unsupported shape"
        ));
    }

    #[test]
    fn nonconstant_gc_memory_decline_kind_is_retryable() {
        let err = bridge_codegen_error(BackendError::RetryableUnsupported(
            "wasm codegen: GcLoadIndexedI scale is not constant".into(),
        ));
        assert!(
            matches!(err, BackendError::CompilationFailed(_)),
            "got {err}"
        );
        let err = bridge_codegen_error(BackendError::RetryableUnsupported(
            "wasm codegen: GcLoadI item size is not constant".into(),
        ));
        assert!(matches!(err, BackendError::CompilationFailed(_)));
    }

    use failguard::{ca_entry, ca_mark_entry, ca_publish, mark_cells_holding};
    use majit_backend::{Backend, JitCellToken};
    use majit_gc::collector::MiniMarkGC;
    use majit_gc::trace::TypeInfo;
    use majit_ir::InputArg;
    use majit_ir::forwarding::bound_operand_from_opref as rb;

    #[test]
    fn boxed_identity_hash_reads_through_the_raw_mirror() {
        use std::cell::Cell;

        thread_local! {
            static MUT_CALLS: Cell<u32> = const { Cell::new(0) };
            static RE_CALLS: Cell<u32> = const { Cell::new(0) };
        }

        struct Probe;
        impl GcAllocator for Probe {
            fn alloc_nursery(&mut self, _size: usize) -> majit_ir::GcRef {
                majit_ir::GcRef(0)
            }
            fn alloc_nursery_no_collect(&mut self, _size: usize) -> majit_ir::GcRef {
                majit_ir::GcRef(0)
            }
            fn alloc_varsize(
                &mut self,
                _base_size: usize,
                _item_size: usize,
                _length: usize,
            ) -> majit_ir::GcRef {
                majit_ir::GcRef(0)
            }
            fn alloc_varsize_no_collect(
                &mut self,
                _base_size: usize,
                _item_size: usize,
                _length: usize,
            ) -> majit_ir::GcRef {
                majit_ir::GcRef(0)
            }
            fn write_barrier(&mut self, _obj: majit_ir::GcRef) {}
            fn jit_remember_young_pointer_from_array(&mut self, _obj: majit_ir::GcRef) {}
            fn remember_young_pointer_from_array2(
                &mut self,
                _obj: majit_ir::GcRef,
                _index: usize,
                _card_page_shift: u32,
            ) {
            }
            fn collect_nursery(&mut self) {}
            fn collect_full(&mut self) {}
            fn nursery_free(&self) -> *mut u8 {
                std::ptr::null_mut()
            }
            fn nursery_free_addr(&self) -> usize {
                0
            }
            fn nursery_top(&self) -> *const u8 {
                std::ptr::null()
            }
            fn nursery_top_addr(&self) -> usize {
                0
            }
            fn max_nursery_object_size(&self) -> usize {
                0
            }
            fn id_or_identityhash(&mut self, addr: usize) -> usize {
                MUT_CALLS.with(|calls| calls.set(calls.get() + 1));
                addr.wrapping_add(1)
            }
            fn id_or_identityhash_reentrant(&self, addr: usize) -> usize {
                RE_CALLS.with(|calls| calls.set(calls.get() + 1));
                addr.wrapping_add(2)
            }
        }

        let _lock = failguard::lock_cpu();
        let _box = install_gc_box(Box::new(Probe));
        MUT_CALLS.with(|calls| calls.set(0));
        RE_CALLS.with(|calls| calls.set(0));

        assert_eq!(wasm_id_or_identityhash(10), 11);
        assert_eq!(MUT_CALLS.with(|calls| calls.get()), 1);
        assert_eq!(RE_CALLS.with(|calls| calls.get()), 0);

        gc_box::with_mut(|_| assert_eq!(wasm_id_or_identityhash(10), 12))
            .expect("the installed box answers");
        assert_eq!(MUT_CALLS.with(|calls| calls.get()), 1);
        assert_eq!(RE_CALLS.with(|calls| calls.get()), 1);
    }

    fn gcmap_marks(buf: &[usize], index: usize) -> bool {
        let bits = usize::BITS as usize;
        let word = 1 + index / bits;
        word < buf.len() && (buf[word] & (1usize << (index % bits))) != 0
    }

    /// Peeled loop: preamble value `kept` is used after the LABEL and is not
    /// a LABEL argument, so the resume loader reads a capture slot.
    fn compile_peeled_label(token_number: u64, capture: bool) -> (WasmBackend, JitCellToken) {
        let mut backend = WasmBackend::new();
        let token = JitCellToken::new(token_number as i64);
        let label = majit_ir::make_loop_target_descr(token_number, false);
        let inputargs = vec![InputArg::new_int_rc(0)];
        let kept = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        kept.pos().set(majit_ir::OpRef::int_op(1));
        let carried = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(2)),
            ],
        ));
        carried.pos().set(majit_ir::OpRef::int_op(2));
        let label_arg = if capture {
            majit_ir::OpRef::int_op(2)
        } else {
            majit_ir::OpRef::int_op(1)
        };
        let label_op = OpRc::new(majit_ir::Op::new(majit_ir::OpCode::Label, &[rb(label_arg)]));
        label_op.setdescr(label);
        // `kept` (int_op 1) is a LABEL arg only when `capture` is false.
        // The capture loop finishes that value without passing it, so the
        // resume loader reads its capture slot.
        let finish = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Finish,
            &[rb(majit_ir::OpRef::int_op(1))],
        ));
        let jump = OpRc::new(majit_ir::Op::new(majit_ir::OpCode::Jump, &[rb(label_arg)]));
        jump.setdescr(label_op.getdescr().expect("label descr"));
        backend
            .compile_loop(&inputargs, &[kept, carried, label_op, finish, jump], &token)
            .expect("peeled loop compiles");
        (backend, token)
    }

    #[test]
    fn dispatch_key_entry_refuses_a_label_that_reads_captures() {
        let _compile_guard = failguard::lock_cpu();
        let (capture_backend, capture_token) = compile_peeled_label(9_910_301, true);
        assert!(capture_backend.supports_dispatch_key_entry());
        assert!(
            !capture_backend.supports_dispatch_key_entry_for(&capture_token, 1),
            "a LABEL resume loader that reads a capture slot cannot take a fresh host frame"
        );
        let (plain_backend, plain_token) = compile_peeled_label(9_910_302, false);
        assert!(plain_backend.supports_dispatch_key_entry_for(&plain_token, 1));
    }

    /// A trip queued while the bridge was alive must not be dereferenced after
    /// `free_loop_and_bridges` drops that bridge's `LoopAsmResources`.
    #[test]
    fn a_freed_pending_slot_is_not_dereferenced_on_drain() {
        let token = std::sync::Arc::new(JitCellToken::new(9_910_077));
        let clt = std::sync::Arc::new(majit_backend::CompiledLoopToken::new(token.number()));
        clt.set_loop_token_wref(std::sync::Arc::downgrade(&token));
        token.set_compiled_loop_token(Some(clt));

        let mut resources = release::LoopAsmResources::default();
        let (slot, _counter) = resources.alloc_pending_inline(PendingInline {
            owner: std::sync::Arc::downgrade(&token),
            region: codegen::InlinedBridge {
                source_fail_index: 0,
                external_jump: None,
                outside_loop: false,
                trace_id: 1,
                inputargs: Vec::new(),
                ops: Vec::new(),
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: indexmap::IndexMap::new(),
            },
            remap: None,
            retry_on_sibling: false,
        });
        release::push_resources(&token, resources);
        let backend = WasmBackend::new();
        backend
            .tripped_inlines
            .borrow_mut()
            .push((std::sync::Arc::downgrade(&token), slot));
        token.compiled_loop_token_expect().free_loop_and_bridges();

        let before = bridge_diag(55);
        for tripped in backend.take_tripped_inlines() {
            backend.install_pending_inline(tripped);
        }
        assert_eq!(
            bridge_diag(55),
            before,
            "a trip whose block was freed must not install"
        );
    }

    /// `llmodel.py` `free_loop_and_bridges`: a loop and its bridge drop their
    /// table slot, bridge-cell reservation, label target and call-assembler
    /// entry. The next compile reuses the slot id instead of growing it.
    #[test]
    fn free_loop_releases_slot_cells_and_registries() {
        let _compile_guard = failguard::lock_cpu();
        let mut backend = WasmBackend::new();
        let token = JitCellToken::new(9_910_001);
        let label = majit_ir::make_loop_target_descr(70, false);
        let label_kept = label.clone();
        let inputargs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        let label_op = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Label,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        label_op.setdescr(label.clone());
        let advance = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        advance.pos().set(majit_ir::OpRef::int_op(2));
        let guard = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::GuardTrue,
            &[rb(majit_ir::OpRef::int_op(2))],
        ));
        guard.setfailargs(
            vec![
                rb(majit_ir::OpRef::int_op(2)),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ]
            .into(),
        );
        guard.set_fail_arg_types(vec![majit_ir::Type::Int, majit_ir::Type::Int]);
        let jump = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Jump,
            &[
                majit_ir::operand::Operand::from_bound_op(&advance),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        jump.setdescr(label.clone());
        let ops = vec![label_op, advance, guard, jump];
        backend
            .compile_loop(&inputargs, &ops, &token)
            .expect("loop compiles");
        assert!(target_from_token(&token).is_some());
        assert!(failguard::label_target(&label_kept).is_some());
        let fail = FreeFailDescr {
            fail_index: 0,
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Int],
        };
        let bridge_advance = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        bridge_advance.pos().set(majit_ir::OpRef::int_op(3));
        let bridge_jump = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Jump,
            &[
                majit_ir::operand::Operand::from_bound_op(&bridge_advance),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        bridge_jump.setdescr(label);
        let bridge_inputs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        backend
            .compile_bridge(
                &fail,
                &bridge_inputs,
                &[bridge_advance, bridge_jump],
                &token,
                None,
            )
            .expect("bridge compiles");
        let clt = token.compiled_loop_token_expect();
        let owned = clt.asmmemmgr_blocks.lock().len();
        assert!(owned >= 1, "loop resources sit on asmmemmgr_blocks");
        let gcmaps_owned = clt.asmmemmgr_blocks.lock().iter().any(|block| {
            block
                .downcast_ref::<release::LoopAsmResources>()
                .is_some_and(|resources| !resources.gcmaps.is_empty())
        });
        assert!(gcmaps_owned, "gcmap is owned by the token, not leaked");
        drop(clt);

        backend.free_loop(&token);
        assert!(token.ll_function_addr() == 0 || target_from_token(&token).is_none());
        assert!(failguard::label_target(&label_kept).is_none());
        assert!(
            token
                .compiled_loop_token_expect()
                .asmmemmgr_blocks
                .lock()
                .is_empty(),
            "free_loop_and_bridges dropped the asm blocks"
        );
        drop(token);

        let token2 = JitCellToken::new(9_910_002);
        let label2 = majit_ir::make_loop_target_descr(71, false);
        let label_op = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Label,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        label_op.setdescr(label2.clone());
        let advance = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        advance.pos().set(majit_ir::OpRef::int_op(2));
        let jump = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Jump,
            &[
                majit_ir::operand::Operand::from_bound_op(&advance),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        jump.setdescr(label2);
        backend
            .compile_loop(&inputargs, &[label_op, advance, jump], &token2)
            .expect("second loop compiles");
        assert!(failguard::label_target(&label_kept).is_none());
        assert!(
            token2
                .compiled_loop_token_expect()
                .asmmemmgr_blocks
                .lock()
                .iter()
                .any(|block| {
                    block
                        .downcast_ref::<release::LoopAsmResources>()
                        .is_some_and(|resources| !resources.gcmaps.is_empty())
                }),
            "second compile owns a fresh gcmap"
        );
    }

    /// A closing JUMP names the label descr of a loop compiled earlier.
    /// Freeing an unrelated loop must leave that descr's target in place
    /// (`assembler.py` `closing_jump` reads `TargetToken._ll_loop_code`
    /// off the JUMP, not a process-global table).
    #[test]
    fn closing_jump_resolves_after_unrelated_loop_is_freed() {
        let _compile_guard = failguard::lock_cpu();
        let mut backend = WasmBackend::new();
        let inputargs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        let token = JitCellToken::new(9_910_101);
        let label = majit_ir::make_loop_target_descr(80, false);
        let label_op = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Label,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        label_op.setdescr(label.clone());
        let advance = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        advance.pos().set(majit_ir::OpRef::int_op(2));
        let guard = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::GuardTrue,
            &[rb(majit_ir::OpRef::int_op(2))],
        ));
        guard.setfailargs(
            vec![
                rb(majit_ir::OpRef::int_op(2)),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ]
            .into(),
        );
        guard.set_fail_arg_types(vec![majit_ir::Type::Int, majit_ir::Type::Int]);
        let jump = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Jump,
            &[
                majit_ir::operand::Operand::from_bound_op(&advance),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        jump.setdescr(label.clone());
        backend
            .compile_loop(&inputargs, &[label_op, advance, guard, jump], &token)
            .expect("first loop compiles");
        let published = failguard::label_target(&label).expect("label published on its descr");

        let other = JitCellToken::new(9_910_102);
        let other_label = majit_ir::make_loop_target_descr(81, false);
        let other_label_op = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Label,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        other_label_op.setdescr(other_label.clone());
        let other_advance = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        other_advance.pos().set(majit_ir::OpRef::int_op(2));
        let other_jump = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Jump,
            &[
                majit_ir::operand::Operand::from_bound_op(&other_advance),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        other_jump.setdescr(other_label.clone());
        backend
            .compile_loop(
                &inputargs,
                &[other_label_op, other_advance, other_jump],
                &other,
            )
            .expect("unrelated loop compiles");
        backend.free_loop(&other);
        assert!(failguard::label_target(&other_label).is_none());
        let still = failguard::label_target(&label).expect("earlier label survives");
        assert_eq!(still.func_handle, published.func_handle);
        assert_eq!(still.key, published.key);

        let fail = FreeFailDescr {
            fail_index: 0,
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Int],
        };
        let bridge_advance = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[
                rb(majit_ir::OpRef::input_arg_int(0)),
                rb(majit_ir::OpRef::const_int(1)),
            ],
        ));
        bridge_advance.pos().set(majit_ir::OpRef::int_op(3));
        let bridge_jump = OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::Jump,
            &[
                majit_ir::operand::Operand::from_bound_op(&bridge_advance),
                rb(majit_ir::OpRef::input_arg_int(1)),
            ],
        ));
        bridge_jump.setdescr(label.clone());
        backend
            .compile_bridge(
                &fail,
                &inputargs,
                &[bridge_advance, bridge_jump],
                &token,
                None,
            )
            .expect("closing JUMP still resolves the earlier label");
        assert!(failguard::label_target(&label).is_some());
    }

    #[derive(Debug)]
    struct FreeFailDescr {
        fail_index: u32,
        arg_types: Vec<majit_ir::Type>,
    }

    impl majit_ir::Descr for FreeFailDescr {}

    impl majit_ir::descr::FailDescr for FreeFailDescr {
        fn fail_index(&self) -> u32 {
            self.fail_index
        }
        fn fail_arg_types(&self) -> &[majit_ir::Type] {
            &self.arg_types
        }
    }

    #[test]
    fn home_gcmap_marks_used_homes_and_label_captures_only() {
        let _compile_guard = failguard::lock_cpu();
        let sign = std::mem::size_of::<isize>();
        let frame = codegen::FrameGeometry::compact(16, 128 + 2, 2);
        let map = codegen::build_home_gcmap(frame, 5, 2);
        let idx = |h: usize| (frame.home_slot_base as usize + h * 8) / sign;
        for h in 0..5 {
            assert!(gcmap_marks(&map, idx(h)), "used ordinary home {h}");
        }
        for h in 5..128 {
            assert!(!gcmap_marks(&map, idx(h)), "reserved ordinary home {h}");
        }
        assert!(gcmap_marks(&map, idx(128)), "label capture 0");
        assert!(gcmap_marks(&map, idx(129)), "label capture 1");
        let narrow = codegen::build_home_gcmap(frame, 5, 1);
        assert!(gcmap_marks(&narrow, idx(128)), "actual label capture");
        assert!(
            !gcmap_marks(&narrow, idx(129)),
            "reserved unused label slot"
        );
    }

    #[test]
    fn union_gcmap_covers_incomparable_ordinary_and_label_maps() {
        let _compile_guard = failguard::lock_cpu();
        let frame = codegen::FrameGeometry::compact(16, 128 + 2, 2);
        let owner = codegen::build_home_gcmap(frame, 8, 0);
        let bridge = codegen::build_home_gcmap(frame, 3, 2);
        let union = wasm_jit_union_gcmap(
            owner.as_ptr() as usize as i64,
            bridge.as_ptr() as usize as i64,
        ) as usize as *const usize;
        assert!(!union.is_null());
        let n = unsafe { *union };
        let words = unsafe { std::slice::from_raw_parts(union, 1 + n) };
        let sign = std::mem::size_of::<isize>();
        let idx = |h: usize| (frame.home_slot_base as usize + h * 8) / sign;
        for h in 0..8 {
            assert!(gcmap_marks(words, idx(h)), "owner ordinary home {h}");
        }
        assert!(gcmap_marks(words, idx(128)), "bridge label 0");
        assert!(gcmap_marks(words, idx(129)), "bridge label 1");
        assert_eq!(
            wasm_jit_union_gcmap(union as usize as i64, owner.as_ptr() as usize as i64),
            union as usize as i64,
            "owner is a subset of the union"
        );
    }

    #[test]
    fn union_gcmap_keeps_bits_past_sixty_four_words() {
        let _compile_guard = failguard::lock_cpu();
        // value_slots=16, 4200 homes: last signed index is past 64 data words
        // on a 64-bit host (`build_home_gcmap` word count).
        let frame = codegen::FrameGeometry::compact(16, 4200, 2);
        let owner = codegen::build_home_gcmap(frame, 4100, 0);
        let bridge = codegen::build_home_gcmap(frame, 3, 2);
        assert!(owner[0] > 64, "fixture must exceed the old 64-word cap");
        let union = wasm_jit_union_gcmap(
            owner.as_ptr() as usize as i64,
            bridge.as_ptr() as usize as i64,
        ) as usize as *const usize;
        let n = unsafe { *union };
        let words = unsafe { std::slice::from_raw_parts(union, 1 + n) };
        let sign = std::mem::size_of::<isize>();
        let idx = |h: usize| (frame.home_slot_base as usize + h * 8) / sign;
        assert!(gcmap_marks(words, idx(4099)), "high ordinary home");
        assert!(gcmap_marks(words, idx(4198)), "bridge label 0");
        assert!(gcmap_marks(words, idx(4199)), "bridge label 1");
    }

    #[test]
    fn parameter_bridge_dispatch_is_bounded_by_guard_population() {
        let _compile_guard = failguard::lock_cpu();
        assert!(bridge_param_dispatch_profitable(MAX_BRIDGE_PARAM_GUARDS));
        assert!(!bridge_param_dispatch_profitable(
            MAX_BRIDGE_PARAM_GUARDS + 1
        ));
    }

    #[test]
    fn ca_pop_publishes_the_forwarded_shadow_stack_frame() {
        let _compile_guard = failguard::lock_cpu();
        use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame, jitframe_type_info};
        use majit_gc::GcAllocator;

        let mut gc = MiniMarkGC::new();
        let tid = gc.register_type(jitframe_type_info());
        let old = gc.alloc_oldgen_typed(tid, JitFrame::alloc_size(1));
        let forwarded = gc.alloc_oldgen_typed(tid, JitFrame::alloc_size(1));
        let map = [1_usize, 0];
        for value in [old, forwarded] {
            unsafe {
                let jf = value.0 as *mut JitFrame;
                JitFrame::init(jf, std::ptr::null(), 1);
                (*jf).jf_gcmap = map.as_ptr().cast();
            }
        }
        let saved = majit_gc::shadow_stack::push_jf(old);
        // Model the root-slot forwarding performed by a collection in deopt.
        majit_gc::shadow_stack::walk_jf_roots(|root| {
            if *root == old {
                *root = forwarded;
            }
        });
        wasm_jit_ca_pop_frame((old.0 + FIRST_ITEM_OFFSET) as i64);
        assert_eq!(majit_gc::shadow_stack::jf_depth(), saved);
        // The exit already stored the guard gcmap. Pop forwards the frame
        // and leaves that map in place (`genop_finish` / `push_gcmap`).
        unsafe {
            assert_eq!(
                (*(forwarded.0 as *const JitFrame)).jf_gcmap,
                map.as_ptr().cast()
            );
            assert_eq!((*(old.0 as *const JitFrame)).jf_gcmap, map.as_ptr().cast());
        }
    }

    fn gcmap_has_index(buf: &[usize], index: usize) -> bool {
        let bits = usize::BITS as usize;
        let word = 1 + index / bits;
        word < buf.len() && buf[word] & (1usize << (index % bits)) != 0
    }

    #[test]
    fn home_gcmap_marks_homes_not_overwritable_input_slots() {
        let _compile_guard = failguard::lock_cpu();
        // FRAME_SLOT_BASE is the value/fail-arg area. A static gcmap bit there
        // stays set after a guard spill overwrites the slot with an integer,
        // and `is_nursery_object_start` is only a nursery range check.
        let frame = codegen::FrameGeometry::compact(4, 2, 0);
        let buf = codegen::build_home_gcmap(frame, 2, 0);
        let sign = std::mem::size_of::<isize>();
        let input0 = codegen::FRAME_SLOT_BASE as usize / sign;
        let home0 = frame.home_slot_base as usize / sign;
        let home1 = (frame.home_slot_base as usize + 8) / sign;
        assert!(
            !gcmap_has_index(&buf, input0),
            "Ref input slot {input0} must not stay marked; fail-arg spills reuse it"
        );
        assert!(gcmap_has_index(&buf, home0), "home 0 (item {home0})");
        assert!(gcmap_has_index(&buf, home1), "home 1 (item {home1})");
    }

    #[test]
    fn headerless_helper_rejects_non_positive_size() {
        let _compile_guard = failguard::lock_cpu();
        let gc = MiniMarkGC::new();
        let _gc_box = install_gc_box(Box::new(gc));
        assert_eq!(wasm_jit_alloc_headerless(-1), 0);
        assert_eq!(wasm_jit_alloc_headerless(0), 0);
    }

    #[test]
    fn oldgen_array_overflow_does_not_allocate() {
        let _compile_guard = failguard::lock_cpu();
        let mut gc = MiniMarkGC::new();
        let tid = gc.register_type(TypeInfo::simple(8));
        let _gc_box = install_gc_box(Box::new(gc));
        // 8 * 2^61 wraps to 0; without the checked size this would allocate
        // an 8-byte object and stamp the original length into it.
        assert_eq!(wasm_jit_alloc_array_oldgen(tid as i64, 8, 8, 1 << 61, 0), 0);
    }

    #[test]
    fn gc_rewriter_registers_rewrite_abi_malloc_wrappers() {
        let rewriter = gc_rewriter();
        assert_eq!(
            rewriter.malloc_array_fn,
            wasm_malloc_array as *const () as i64
        );
        assert_eq!(
            rewriter.malloc_array_nonstandard_fn,
            wasm_malloc_array_nonstandard as *const () as i64
        );
        assert_eq!(
            rewriter.malloc_array_oldgen_fn,
            wasm_malloc_array_oldgen as *const () as i64
        );
        assert_eq!(
            rewriter.malloc_array_nonstandard_oldgen_fn,
            wasm_malloc_array_nonstandard_oldgen as *const () as i64
        );
        assert_eq!(rewriter.malloc_str_fn, wasm_malloc_str as *const () as i64);
        assert_eq!(
            rewriter.malloc_unicode_fn,
            wasm_malloc_unicode as *const () as i64
        );
        assert_eq!(
            rewriter.malloc_big_fixedsize_fn,
            wasm_malloc_big_fixedsize as *const () as i64
        );
        assert_eq!(
            rewriter.malloc_big_fixedsize_oldgen_fn,
            wasm_malloc_big_fixedsize_oldgen as *const () as i64
        );
    }

    #[test]
    fn typed_blackhole_allocation_never_falls_back_to_raw_memory() {
        let _compile_guard = failguard::lock_cpu();
        // No active wasm GC is installed on this test thread.  A typed descr
        // therefore has no legal allocator and must report NULL to
        // blackhole.py `_get_method`; the previous raw fallback returned a
        // headerless block that the collector could neither identify nor
        // trace.
        assert_eq!(wasm_bh_alloc(1, 32), 0);
    }

    #[test]
    fn blackhole_varsize_rejects_negative_lengths_without_panicking() {
        let _compile_guard = failguard::lock_cpu();
        let backend = WasmBackend::new();
        assert_eq!(backend.bh_newstr(-1), 0);
        assert_eq!(backend.bh_newunicode(-1), 0);
    }

    #[test]
    fn cross_loop_terminal_jump_uses_target_descr_identity() {
        let _compile_guard = failguard::lock_cpu();
        let local_descr = majit_ir::make_loop_target_descr(1, false);
        let foreign_descr = majit_ir::make_loop_target_descr(2, false);
        let label = Op::new(majit_ir::OpCode::Label, &[]);
        label.setdescr(local_descr.clone());
        let jump = Op::new(majit_ir::OpCode::Jump, &[]);
        jump.setdescr(foreign_descr);
        let ops = vec![label, jump];

        assert!(has_cross_loop_terminal_jump(&ops));
        ops[1].setdescr(local_descr);
        assert!(!has_cross_loop_terminal_jump(&ops));
    }

    #[test]
    fn straightline_trace_defers_host_module_until_execution() {
        let _compile_guard = failguard::lock_cpu();
        let mut backend = WasmBackend::new();
        let token = JitCellToken::new(1);
        let finish = Op::new(majit_ir::OpCode::Finish, &[]);
        finish.pos().set(majit_ir::OpRef::void_op(0));
        finish.set_fail_arg_types(Vec::new());
        finish.setfailargs(Vec::new().into());

        backend
            .compile_loop(&[], &[OpRc::new(finish)], &token)
            .expect("compile straight-line wasm trace");
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            .expect("compiled wasm metadata");

        assert_eq!(compiled.eager_func_handle(), 0);
        assert!(compiled.pending_wasm_bytes.borrow().is_some());

        // Retiring an unentered token must leave the module unmaterialized;
        // this is the exception/global-version invalidation-storm case.
        token.invalidate();
        assert_eq!(compiled.eager_func_handle(), 0);
        assert!(compiled.pending_wasm_bytes.borrow().is_some());
    }

    fn fresh_entry() -> Box<failguard::WasmCaDispatchEntry> {
        Box::new(failguard::WasmCaDispatchEntry::pending())
    }

    fn publish_entry(
        entry: &failguard::WasmCaDispatchEntry,
        handle: u32,
        compiled_ptr: u32,
        gnf2: u32,
    ) {
        ca_publish(
            entry,
            handle,
            compiled_ptr as u64,
            33,
            44,
            55,
            0,
            0,
            gnf2,
            0,
            0,
            0,
        );
    }

    #[test]
    fn identical_call_assembler_publication_reuses_the_runtime_snapshot() {
        let _compile_guard = failguard::lock_cpu();
        let entry = fresh_entry();
        let compiled_ptr = 1_000_022;
        publish_entry(&entry, 11, compiled_ptr, 0);
        publish_entry(&entry, 11, compiled_ptr, 0);
        {
            let targets = entry.targets.lock().unwrap();
            assert_eq!(targets.len(), 1);
            assert_eq!(targets[0].has_guard_not_forced_2, 0);
        }
        publish_entry(&entry, 11, compiled_ptr, 1);
        {
            let targets = entry.targets.lock().unwrap();
            assert_eq!(targets.len(), 2);
            assert_eq!(targets[1].has_guard_not_forced_2, 1);
        }
        assert_eq!(
            entry
                .has_guard_not_forced_2
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
        publish_entry(&entry, 11, compiled_ptr, 0);
        assert_eq!(
            entry
                .has_guard_not_forced_2
                .load(std::sync::atomic::Ordering::Acquire),
            1,
            "a later publish without GNF2 must not clear the cell flag"
        );
    }

    #[test]
    fn mark_gnf2_sets_the_cell_without_a_new_snapshot() {
        let _compile_guard = failguard::lock_cpu();
        let entry = fresh_entry();
        publish_entry(&entry, 11, 1_000_023, 0);
        ca_mark_entry(&entry);
        assert_eq!(
            entry
                .has_guard_not_forced_2
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
        let targets = entry.targets.lock().unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].has_guard_not_forced_2, 0);
    }

    #[test]
    fn mark_gnf2_raises_redirected_alias_cells() {
        let _compile_guard = failguard::lock_cpu();
        let old = fresh_entry();
        let new = fresh_entry();
        let new_ptr = 1_000_025;
        publish_entry(&old, 1, 1_000_024, 0);
        publish_entry(&new, 2, new_ptr, 0);
        publish_entry(&old, 2, new_ptr, 0);
        mark_cells_holding(&[old.as_ref(), new.as_ref()], new_ptr);
        for entry in [&old, &new] {
            assert_eq!(
                entry
                    .has_guard_not_forced_2
                    .load(std::sync::atomic::Ordering::Acquire),
                1
            );
        }
    }

    #[test]
    fn mark_gnf2_raises_cells_that_retain_a_historical_target() {
        let _compile_guard = failguard::lock_cpu();
        let alias = fresh_entry();
        let source = fresh_entry();
        let source_ptr = 1_000_026;
        publish_entry(&alias, 1, source_ptr, 0);
        publish_entry(&source, 2, source_ptr, 0);
        publish_entry(&alias, 3, 1_000_027, 0);
        mark_cells_holding(&[alias.as_ref(), source.as_ref()], source_ptr);
        {
            let targets = alias.targets.lock().unwrap();
            assert_eq!(targets.len(), 2);
            assert_eq!(targets[0].compiled_ptr, source_ptr);
            assert_eq!(targets[1].compiled_ptr, 1_000_027);
        }
        assert_eq!(
            alias
                .has_guard_not_forced_2
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
    }

    #[test]
    fn publish_after_mark_raises_a_new_alias_cell() {
        let _compile_guard = failguard::lock_cpu();
        let source = fresh_entry();
        let alias = fresh_entry();
        let compiled_ptr = 1_000_028;
        publish_entry(&source, 2, compiled_ptr, 0);
        ca_mark_entry(&source);
        let flag = source
            .has_guard_not_forced_2
            .load(std::sync::atomic::Ordering::Acquire);
        publish_entry(&alias, 2, compiled_ptr, flag);
        assert_eq!(
            alias
                .has_guard_not_forced_2
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
    }

    #[test]
    fn redirect_call_assembler_grows_tmp_callback_frame_info() {
        let _compile_guard = failguard::lock_cpu();
        fn compile_with_depth(backend: &mut WasmBackend, token: &JitCellToken, value_count: u32) {
            let inputargs = vec![InputArg::new_int_rc(0)];
            let mut previous = majit_ir::OpRef::input_arg_int(0);
            let mut ops = Vec::new();
            let mut values = Vec::new();
            for position in 1..=value_count {
                let op = Op::new(
                    majit_ir::OpCode::IntAdd,
                    &[rb(previous), rb(majit_ir::OpRef::const_int(1))],
                );
                op.pos().set(majit_ir::OpRef::int_op(position));
                previous = op.pos().get();
                values.push(previous);
                ops.push(OpRc::new(op));
            }
            if value_count > 1 {
                let guard = Op::new(
                    majit_ir::OpCode::GuardTrue,
                    &[rb(majit_ir::OpRef::const_int(1))],
                );
                guard.pos().set(majit_ir::OpRef::void_op(value_count + 1));
                guard.setfailargs(values.iter().copied().map(rb).collect::<Vec<_>>().into());
                guard.set_fail_arg_types(vec![majit_ir::Type::Int; values.len()]);
                ops.push(OpRc::new(guard));
            }
            let finish = Op::new(majit_ir::OpCode::Finish, &[rb(previous)]);
            finish.pos().set(majit_ir::OpRef::void_op(value_count + 2));
            finish.set_fail_arg_types(vec![majit_ir::Type::Int]);
            ops.push(OpRc::new(finish));
            backend
                .compile_loop(&inputargs, &ops, token)
                .expect("compile wasm redirect target");
        }

        let mut backend = WasmBackend::new();
        let tmp = JitCellToken::new(9_900_060);
        let real = JitCellToken::new(9_900_061);
        compile_with_depth(&mut backend, &tmp, 1);
        compile_with_depth(&mut backend, &real, 96);

        let tmp_clt = tmp.compiled_loop_token().expect("tmp callback CLT");
        let real_clt = real.compiled_loop_token().expect("real loop CLT");
        let tmp_depth = tmp_clt.frame_info.lock().depth();
        let real_depth = real_clt.frame_info.lock().depth();
        assert!(tmp_depth < real_depth);
        let tmp_target = target_from_token(&tmp).expect("tmp callback metadata");
        let real_target = target_from_token(&real).expect("real loop metadata");
        assert_ne!(tmp_target.dispatch_key_ofs, real_target.dispatch_key_ofs);

        backend
            .redirect_call_assembler(&tmp, &real)
            .expect("redirect tmp callback to deeper real loop");
        assert_eq!(tmp_clt.frame_info.lock().depth(), real_depth);

        let redirected = target_from_token(&tmp).expect("redirected target metadata");
        let installed = target_from_token(&real).expect("real target metadata");
        assert_eq!(redirected.callee_frame_bytes, installed.callee_frame_bytes);
        assert_eq!(redirected.dispatch_key_ofs, installed.dispatch_key_ofs);
        assert_eq!(redirected.callee_gcmap_ptr, installed.callee_gcmap_ptr);

        let entry = ca_entry(tmp.ll_function_addr()).expect("redirected dispatch entry");
        let targets = entry.targets.lock().unwrap();
        let target = targets.last().expect("published runtime target");
        assert_eq!(target.callee_frame_bytes, installed.callee_frame_bytes);
        assert_eq!(target.dispatch_key_ofs as u64, installed.dispatch_key_ofs);
        assert_eq!(target.callee_gcmap_ptr, installed.callee_gcmap_ptr);
    }

    #[test]
    fn compile_loop_publishes_full_signed_item_depth() {
        let _compile_guard = failguard::lock_cpu();
        let mut backend = WasmBackend::new();
        let token = JitCellToken::new(9_900_071);
        let inputargs = vec![InputArg::new_int_rc(0)];
        let finish = majit_ir::Op::new(
            majit_ir::OpCode::Finish,
            &[rb(majit_ir::OpRef::input_arg_int(0))],
        );
        finish.pos().set(majit_ir::OpRef::void_op(1));
        finish.set_fail_arg_types(vec![majit_ir::Type::Int]);
        backend
            .compile_loop(&inputargs, &[majit_ir::OpRc::new(finish)], &token)
            .expect("compile loop");
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            .expect("compiled wasm loop");
        let depth = token
            .compiled_loop_token()
            .expect("clt")
            .frame_info
            .lock()
            .depth() as usize;
        assert_eq!(
            compiled.frame.ca_frame_depth(),
            compiled.frame.signed_item_count()
        );
        assert_eq!(depth, compiled.frame.signed_item_count());
        assert!(
            depth > compiled.frame.ca_frame_bytes as usize / std::mem::size_of::<isize>(),
            "CALL_ASSEMBLER / execute_token must allocate the full assembled depth, \
             not the homes prefix"
        );
        let map = codegen::build_home_gcmap(
            compiled.frame,
            compiled.frame.ordinary_home_slots(),
            compiled.frame.label_ref_slots,
        );
        let bits = usize::BITS as usize;
        let mut max_bit = 0usize;
        for (i, &w) in map[1..].iter().enumerate() {
            if w != 0 {
                max_bit = max_bit.max(i * bits + (bits - 1 - w.leading_zeros() as usize));
            }
        }
        assert!(
            max_bit < depth,
            "home gcmap bit {max_bit} past published jfi_frame_depth {depth}"
        );
        let alloc = majit_backend::jitframe::JitFrame::alloc_size(depth);
        let mut storage = vec![0u8; alloc];
        let jf = storage.as_mut_ptr() as *mut majit_backend::jitframe::JitFrame;
        unsafe {
            majit_backend::jitframe::JitFrame::init(jf, std::ptr::null(), depth);
            (*jf).jf_gcmap = map.as_ptr().cast();
            majit_backend::jitframe::jitframe_trace_gcmap(jf, |_| {});
        }
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the wasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping.
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let _compile_guard = failguard::lock_cpu();
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x3333_4400;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    /// `_StrFinalizerQueue.finalizer_trigger` drains via `ll_next_dead` from
    /// inside `execute_finalizers`. Index lookup and `next_dead` operate on
    /// the boxed collector's table (`framework.py get_finalizer_queue_index`)
    /// and do not re-enter the boxed collector.
    #[test]
    fn boxed_collector_next_dead_from_trigger_does_not_panic() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _compile_guard = failguard::lock_cpu();
        static FQ: u8 = 0;
        static POPPED: AtomicUsize = AtomicUsize::new(0);

        fn trigger() {
            let tag = majit_gc::FinalizerQueueTag {
                fq: std::ptr::addr_of!(FQ) as usize,
                trigger,
            };
            let index = majit_gc::gc_get_finalizer_queue_index(tag);
            if majit_gc::gc_fq_next_dead(index).is_some() {
                POPPED.store(1, Ordering::Relaxed);
            }
        }

        POPPED.store(0, Ordering::Relaxed);
        let mut gc = MiniMarkGC::new();
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = majit_gc::GcAllocator::alloc_oldgen_typed(&mut gc, tid, 16);
        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let tag = majit_gc::FinalizerQueueTag {
            fq: std::ptr::addr_of!(FQ) as usize,
            trigger,
        };
        let fq_index = majit_gc::gc_get_finalizer_queue_index(tag);
        majit_gc::gc_register_finalizer(fq_index, obj);
        majit_gc::collect_oldgen_nonmoving();
        assert_eq!(POPPED.load(Ordering::Relaxed), 1);
        assert!(majit_gc::gc_fq_next_dead(fq_index).is_none());
    }

    /// `gcreftracer.py` `llop.gc_writebarrier(tr)`: `compile_loop` must
    /// remember the interned table on the active MiniMark, not only pin it
    /// on the CLT. Minor collection walks `pending_gc_tables`, so a nursery
    /// ConstPtr that is only in `LIVE_GC_TABLES` would stay unforwarded.
    #[test]
    fn compile_loop_remembers_gc_table_for_minor_collection() {
        let _compile_guard = failguard::lock_cpu();
        let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
            nursery_size: 65536,
            large_object_threshold: 1024,
            ..majit_gc::collector::GcConfig::default()
        });
        let type_id = gc.register_type(TypeInfo::simple(16));
        let root = gc.alloc_with_type(type_id, 16);
        unsafe { *(root.0 as *mut u64) = 0xA11C_E701 };
        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let constant = majit_ir::Op::new(
            majit_ir::OpCode::SameAsR,
            &[rb(majit_ir::OpRef::const_ptr(root))],
        );
        constant.pos().set(majit_ir::OpRef::ref_op(1));
        let finish = majit_ir::Op::new(majit_ir::OpCode::Finish, &[rb(majit_ir::OpRef::ref_op(1))]);
        finish.pos().set(majit_ir::OpRef::void_op(2));
        finish.set_fail_arg_types(vec![majit_ir::Type::Ref]);
        finish.setfailargs(vec![rb(majit_ir::OpRef::ref_op(1))].into());

        let token = JitCellToken::new(1_500_294);
        backend
            .compile_loop(&[], &[OpRc::new(constant), OpRc::new(finish)], &token)
            .expect("compile wasm loop with a reference constant");

        let clt = token.compiled_loop_token().expect("CLT");
        let table = {
            let tracers = clt.asmmemmgr_gcreftracers.lock();
            tracers
                .iter()
                .find_map(|tracer| {
                    std::sync::Arc::clone(tracer)
                        .downcast::<majit_gc::GcTable>()
                        .ok()
                })
                .expect("compile_loop must pin the interned GcTable on the CLT")
        };
        assert_eq!(table.slot(0), root);

        with_wasm_active_gc_mut(|gc| gc.collect_nursery()).expect("active wasm MiniMark");
        let moved = table.slot(0);
        assert_ne!(moved, root, "remembered table slot must be forwarded");
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xA11C_E701);
    }

    /// Post-rewrite `CallMallocNursery*` traces must get nursery free/top
    /// addresses. `nursery_alloc_params` is the private gatherer that feeds
    /// `compile_loop`.
    #[test]
    fn nursery_alloc_params_accepts_call_malloc_nursery() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        let gc = MiniMarkGC::new();
        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let finish = majit_ir::Op::new(majit_ir::OpCode::Finish, &[]);
        assert!(
            nursery_alloc_params(&[finish.clone()]).is_none(),
            "a trace with no allocation op must still return None"
        );

        let malloc = majit_ir::Op::new(
            majit_ir::OpCode::CallMallocNursery,
            &[rb(majit_ir::OpRef::const_int(32))],
        );
        malloc.pos().set(majit_ir::OpRef::ref_op(1));
        let incr = majit_ir::Op::new(
            majit_ir::OpCode::NurseryPtrIncrement,
            &[
                rb(majit_ir::OpRef::ref_op(1)),
                rb(majit_ir::OpRef::const_int(32)),
            ],
        );
        incr.pos().set(majit_ir::OpRef::ref_op(2));
        let malloc_params = nursery_alloc_params(&[malloc, incr])
            .expect("CallMallocNursery must qualify for the inline bump");
        assert_ne!(malloc_params.free_addr, 0);
        assert_ne!(malloc_params.top_addr, 0);

        for opcode in [
            majit_ir::OpCode::CallMallocNurseryHeaderless,
            majit_ir::OpCode::CallMallocNurseryVarsizeFrame,
        ] {
            let op = majit_ir::Op::new(opcode, &[rb(majit_ir::OpRef::const_int(32))]);
            op.pos().set(majit_ir::OpRef::ref_op(1));
            assert!(
                nursery_alloc_params(&[op]).is_some(),
                "{opcode:?} must qualify for the inline bump"
            );
        }
        let varsize = majit_ir::Op::new(
            majit_ir::OpCode::CallMallocNurseryVarsize,
            &[
                rb(majit_ir::OpRef::const_int(0)),
                rb(majit_ir::OpRef::const_int(8)),
                rb(majit_ir::OpRef::const_int(4)),
            ],
        );
        varsize.pos().set(majit_ir::OpRef::ref_op(1));
        assert!(
            nursery_alloc_params(&[varsize]).is_some(),
            "CallMallocNurseryVarsize must qualify for the inline bump"
        );
        let _ = backend;
    }

    fn u1_new_setfield_ops(type_id: u32) -> (Vec<InputArgRc>, Vec<OpRc>) {
        use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
        use std::sync::Arc;
        let pointer_field = Arc::new(SimpleFieldDescr::new(
            0,
            0,
            std::mem::size_of::<usize>(),
            majit_ir::Type::Ref,
            false,
        ));
        let size = Arc::new(SimpleSizeDescr::new(0, 16, type_id));
        let new1 = majit_ir::Op::new(majit_ir::OpCode::New, &[]);
        new1.setdescr(size.clone());
        new1.pos().set(majit_ir::OpRef::ref_op(1));
        let store1 = majit_ir::Op::new(
            majit_ir::OpCode::SetfieldGc,
            &[
                rb(majit_ir::OpRef::ref_op(1)),
                rb(majit_ir::OpRef::input_arg_ref(0)),
            ],
        );
        store1.setdescr(pointer_field.clone());
        let new2 = majit_ir::Op::new(majit_ir::OpCode::New, &[]);
        new2.setdescr(size);
        new2.pos().set(majit_ir::OpRef::ref_op(2));
        let store2 = majit_ir::Op::new(
            majit_ir::OpCode::SetfieldGc,
            &[
                rb(majit_ir::OpRef::ref_op(2)),
                rb(majit_ir::OpRef::input_arg_ref(0)),
            ],
        );
        store2.setdescr(pointer_field);
        let finish = majit_ir::Op::new(majit_ir::OpCode::Finish, &[]);
        let inputargs = vec![InputArg::from_type_rc(majit_ir::Type::Ref, 0)];
        let ops = vec![
            OpRc::new(new1),
            OpRc::new(store1),
            OpRc::new(new2),
            OpRc::new(store2),
            OpRc::new(finish),
        ];
        (inputargs, ops)
    }

    fn test_module_inputs(inputargs: Vec<InputArgRc>, ops: Vec<Op>) -> codegen::ModuleBuildInputs {
        codegen::ModuleBuildInputs {
            inputargs,
            ops,
            inlined_bridges: Vec::new(),
            constants: indexmap::IndexMap::new(),
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: alloc_helpers(),
            wb: wasm_write_barrier_helpers(),
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            fail_index_base: 0,
            bridge_cells_base: 0,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            inline_trip: None,
            external_jump_slot: 0,
            external_jump_wide_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::compact(5, 2, 0),
            ca: codegen::CaParams::default(),
        }
    }

    fn wasm_import_names(bytes: &[u8]) -> Vec<String> {
        let mut names = Vec::new();
        for payload in wasmparser::Parser::new(0).parse_all(bytes) {
            if let Ok(wasmparser::Payload::ImportSection(imports)) = payload {
                for import in imports {
                    if let Ok(import) = import {
                        names.push(import.name.to_string());
                    }
                }
            }
        }
        names
    }

    /// The normalize + GC rewrite half of [`WasmBackend::compile_loop`], with
    /// the codegen that follows it left off.
    fn prepare_ops_for_compile(
        backend: &mut WasmBackend,
        inputargs: &[InputArgRc],
        ops: &[OpRc],
    ) -> (Vec<Op>, Option<Arc<majit_gc::GcTable>>) {
        let mut ops_owned = normalize_ops_for_codegen(inputargs, ops);
        codegen::materialize_unbound_label_args(inputargs, &mut ops_owned);
        backend.rewrite_ops_for_gc(ops_owned)
    }

    /// Adjacent `New`s merge into `CallMallocNursery` + `NurseryPtrIncrement`.
    #[test]
    fn gc_rewrite_on_merges_adjacent_news() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        let mut gc = MiniMarkGC::new();
        let type_id = gc.register_type(TypeInfo::simple(16));
        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));
        let (inputargs, ops) = u1_new_setfield_ops(type_id);
        let (prepared, _) = prepare_ops_for_compile(&mut backend, &inputargs, &ops);
        assert!(
            prepared
                .iter()
                .any(|op| op.opcode == majit_ir::OpCode::CallMallocNursery),
            "rewritten list must contain CallMallocNursery: {:?}",
            prepared.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
        assert!(
            prepared
                .iter()
                .any(|op| op.opcode == majit_ir::OpCode::NurseryPtrIncrement),
            "adjacent mallocs must merge via NurseryPtrIncrement"
        );
        assert!(
            prepared.iter().all(|op| op.opcode != majit_ir::OpCode::New),
            "rewritten list must not contain New"
        );
    }

    /// A ConstPtr used as an operand and as a failarg shares one gc table,
    /// and the failarg is bound.
    #[test]
    fn gc_rewrite_on_one_table_for_operand_and_failarg_constptr() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
            nursery_size: 65536,
            large_object_threshold: 1024,
            ..majit_gc::collector::GcConfig::default()
        });
        let type_id = gc.register_type(TypeInfo::simple(16));
        let root = gc.alloc_with_type(type_id, 16);
        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));
        let same = majit_ir::Op::new(
            majit_ir::OpCode::SameAsR,
            &[rb(majit_ir::OpRef::const_ptr(root))],
        );
        same.pos().set(majit_ir::OpRef::ref_op(1));
        let finish = majit_ir::Op::new(majit_ir::OpCode::Finish, &[rb(majit_ir::OpRef::ref_op(1))]);
        finish.setfailargs(vec![rb(majit_ir::OpRef::const_ptr(root))].into());
        let (prepared, table) =
            prepare_ops_for_compile(&mut backend, &[], &[OpRc::new(same), OpRc::new(finish)]);
        let table = table.expect("operand ConstPtr must intern into a gc table");
        assert_eq!(table.slot(0), root);
        assert!(
            prepared
                .iter()
                .any(|op| op.opcode == majit_ir::OpCode::LoadFromGcTable),
            "operand ConstPtr must become LoadFromGcTable"
        );
        let finish = prepared
            .iter()
            .find(|op| op.opcode == majit_ir::OpCode::Finish)
            .expect("Finish");
        let fa = finish.getfailargs().expect("failargs");
        assert!(
            fa.iter().any(|a| a.to_opref().inline_const_bits().is_some()
                || matches!(a.const_value(), Some(majit_ir::Value::Ref(g)) if g == root)),
            "failarg ConstPtr stays a constant; table still roots it"
        );
    }

    #[test]
    fn reflected_i32_result_follows_descr_signedness() {
        let signed = encode_func_sig(&[FuncSigVal::I64], Some(FuncSigVal::I32));
        set_test_residual_target_sig(11, signed);
        let bits = 0xffff_ffffu32 as i64;
        assert_eq!(widen_reflected_result(11, bits, 'i', true, 4), -1);
        assert_eq!(widen_reflected_result(11, bits, 'i', false, 4), bits);
        assert_eq!(widen_reflected_result(11, bits, 'i', true, 8), bits);
        assert_eq!(widen_reflected_result(11, bits, 'r', true, 4), bits);
        assert_eq!(widen_reflected_result(11, 0x80, 'i', true, 1), -128);
        let float_bits = f32::from_bits(0x3f80_0000).to_bits() as i64;
        let f32_sig = encode_func_sig(&[], Some(FuncSigVal::F32));
        set_test_residual_target_sig(12, f32_sig);
        assert_eq!(
            widen_reflected_result(12, float_bits, 'f', false, 8),
            f64::to_bits(1.0) as i64
        );
        assert_eq!(
            widen_reflected_result(12, float_bits, 'S', false, 4),
            float_bits
        );
    }

    /// Rewritten NewArray slow path (CallR malloc helper) must not import
    /// jit_call.
    #[test]
    fn gc_rewrite_newarray_slow_path_has_no_jit_call() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        use majit_ir::descr::SimpleArrayDescr;
        use std::sync::Arc;
        let mut backend = WasmBackend::new();
        let arr = majit_ir::Op::new(
            majit_ir::OpCode::NewArray,
            &[rb(majit_ir::OpRef::input_arg_int(0))],
        );
        arr.setdescr(Arc::new(SimpleArrayDescr::new(
            1,
            16,
            8,
            1,
            majit_ir::Type::Int,
        )));
        arr.pos().set(majit_ir::OpRef::ref_op(1));
        let finish = majit_ir::Op::new(majit_ir::OpCode::Finish, &[]);
        let inputargs = vec![InputArg::from_type_rc(majit_ir::Type::Int, 0)];
        let (prepared, _) = prepare_ops_for_compile(
            &mut backend,
            &inputargs,
            &[OpRc::new(arr), OpRc::new(finish)],
        );
        assert!(
            prepared
                .iter()
                .any(|op| op.opcode == majit_ir::OpCode::CallR),
            "runtime-length NewArray without a nursery must take the CallR slow path: {:?}",
            prepared.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
        let mut inputs = test_module_inputs(inputargs, prepared);
        inputs.constants = backend.constants.clone();
        let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("slow-path module");
        let names = wasm_import_names(&bytes);
        assert!(
            names
                .iter()
                .all(|n| n != "jit_call" && n != "jit_call_compact"),
            "rewritten NewArray slow path must not use the jit_call trampoline: {names:?}"
        );
    }

    /// Rewritten `New` of a type with a ref field zero-inits that field at
    /// `size_of::<usize>()` (not a host-literal 8).
    #[test]
    fn gc_rewrite_new_ref_field_zero_store_uses_usize_width() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
        use std::sync::Arc;
        let mut gc = MiniMarkGC::new();
        let type_id = gc.register_type(TypeInfo::simple(16));
        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));
        let field = Arc::new(SimpleFieldDescr::new(
            0,
            0,
            std::mem::size_of::<usize>(),
            majit_ir::Type::Ref,
            false,
        ));
        let size = SimpleSizeDescr::new(0, 16, type_id).with_all_fielddescrs(vec![
            field as std::sync::Arc<dyn majit_ir::descr::FieldDescr>,
        ]);
        let new_op = majit_ir::Op::new(majit_ir::OpCode::New, &[]);
        new_op.setdescr(Arc::new(size));
        new_op.pos().set(majit_ir::OpRef::ref_op(1));
        let finish = majit_ir::Op::new(majit_ir::OpCode::Finish, &[]);
        let (prepared, _) =
            prepare_ops_for_compile(&mut backend, &[], &[OpRc::new(new_op), OpRc::new(finish)]);
        let word = std::mem::size_of::<usize>() as i64;
        let sized = prepared.iter().any(|op| {
            op.opcode == majit_ir::OpCode::GcStore
                && op.arg(3).to_opref().inline_const_bits() == Some(word)
        });
        assert!(
            sized,
            "GcStore size operand must be size_of::<usize>()={word}: {:?}",
            prepared
                .iter()
                .filter(|op| op.opcode == majit_ir::OpCode::GcStore)
                .map(|op| op.arg(3).to_opref())
                .collect::<Vec<_>>()
        );
    }

    /// Spike for the wasm-JITFRAME refactor: prove the shared
    /// `MiniMarkGC` forwards a JitFrame's interior Ref item through the
    /// `jf_gcmap` custom-trace when the frame is discovered via the jitframe
    /// shadow stack. This is the exact GC path the orthodox wasm loop would
    /// depend on — a non-moving old-gen JitFrame whose live Ref item slots are
    /// traced by `jf_gcmap` during a minor collection (`do_collect_nursery`
    /// Phase 1c → `trace_and_update_object` → `jitframe_custom_trace`). The
    /// wasm backend has none of the feeders yet; this confirms the collector
    /// side works so the feeders can be built.
    #[test]
    fn jitframe_oldgen_gcmap_minor_forwards_ref_item() {
        let _compile_guard = failguard::lock_cpu();
        use majit_backend::jitframe::{
            FIRST_ITEM_OFFSET, JF_FRAME_OFS, JF_GCMAP_OFS, JitFrame, jitframe_type_info,
        };
        use majit_gc::GcAllocator;

        let mut gc = MiniMarkGC::new();
        let jf_tid = gc.register_type(jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        let depth = 2usize;
        // Pin the frame in old-gen so only the young item moves. Production
        // `malloc_jitframe` follows `jitframe_allocate` (`lltype.malloc`)
        // and is born in the nursery; this test is the old→young gcmap walk.
        let frame = gc.alloc_oldgen_typed(jf_tid, JitFrame::alloc_size(depth));
        assert_ne!(frame.0, 0, "old-gen JitFrame alloc failed");
        let frame_ptr = frame.0 as *mut JitFrame;
        unsafe { JitFrame::init(frame_ptr, std::ptr::null(), depth) };

        // A fresh nursery object reachable ONLY through the frame's item slot 0.
        let young = gc.alloc_nursery_typed(payload_tid, 16);
        assert_ne!(young.0, 0, "nursery alloc failed");
        let young_before = young.0;
        unsafe {
            let item0 = (frame_ptr as *mut u8).add(FIRST_ITEM_OFFSET) as *mut usize;
            *item0 = young_before;
        }

        // Per-loop gcmap marking item slot 0 as a Ref: [data_word_count, bits].
        // jitframe_trace reads gcmap_lgt at +0, a data word at +GCMAPBASEOFS(8),
        // and maps bit i (of word 0) to jf_frame item i.
        let gcmap: [usize; 2] = [1, 0b1];
        unsafe {
            let gcmap_field = (frame_ptr as *mut u8).add(JF_GCMAP_OFS as usize) as *mut *const u8;
            *gcmap_field = gcmap.as_ptr() as *const u8;
        }

        // The store into an old frame is followed by the frame barrier, as
        // `llmodel.py execute_token` and `_reload_frame_if_necessary` do; the
        // minor collection then traces the remembered frame through its gcmap.
        gc.write_barrier(frame);
        let saved = majit_gc::shadow_stack::push_jf(frame);
        gc.do_collect_nursery();
        majit_gc::shadow_stack::pop_jf_to(saved);

        // The young object must have been forwarded out of the nursery and the
        // item slot rewritten to its new address — proving the gcmap bit was
        // honored. An untraced slot would still hold young_before (now dangling).
        let item0_after =
            unsafe { *((frame_ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize) };
        assert_ne!(item0_after, 0, "item0 cleared: frame interior not traced");
        assert_ne!(
            item0_after, young_before,
            "item0 not forwarded: gcmap bit was not honored by the collector"
        );
        assert!(
            gc.is_managed_heap_object(item0_after),
            "forwarded item0 is not a live managed object"
        );

        // The old-gen frame must NOT have moved: its length header stays intact
        // in place, so a wasm local holding frame_ptr would remain valid.
        let len_after = unsafe { *((frame_ptr as *const u8).add(JF_FRAME_OFS) as *const isize) };
        assert_eq!(len_after, depth as isize, "old-gen frame moved/corrupted");
    }

    /// `llmodel.py` `execute_token`: an old frame that receives young input
    /// refs is remembered by the entry `gc_writebarrier`, before the run.
    /// The host pops the frame after FINISH with no second barrier, and a
    /// minor collection after that pop still forwards the young home.
    #[test]
    fn oldgen_entry_frame_is_remembered_by_the_entry_barrier_across_the_pop() {
        let _compile_guard = failguard::lock_cpu();
        use majit_backend::jitframe::{
            FIRST_ITEM_OFFSET, JF_GCMAP_OFS, JitFrame, jitframe_type_info,
        };
        use majit_gc::GcAllocator;

        let mut gc = MiniMarkGC::new();
        let jf_tid = gc.register_type(jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let frame = gc.alloc_oldgen_typed(jf_tid, JitFrame::alloc_size(2));
        let frame_ptr = frame.0 as *mut JitFrame;
        unsafe { JitFrame::init(frame_ptr, std::ptr::null(), 2) };
        let young_before = gc.alloc_nursery_typed(payload_tid, 16).0;
        let gcmap: [usize; 2] = [1, 0b1];
        unsafe {
            *((frame_ptr as *mut u8).add(FIRST_ITEM_OFFSET) as *mut usize) = young_before;
            *((frame_ptr as *mut u8).add(JF_GCMAP_OFS as usize) as *mut *const u8) =
                gcmap.as_ptr() as *const u8;
        }
        let _gc_box = install_gc_box(Box::new(gc));
        wasm_jit_write_barrier(frame_ptr as i64);
        let saved = majit_gc::shadow_stack::push_jf(frame);
        majit_gc::shadow_stack::pop_jf_to(saved);
        with_wasm_active_gc_mut(|gc| gc.collect_nursery());
        let item0 = unsafe { *((frame_ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize) };
        assert_ne!(item0, 0, "young home cleared after host pop");
        assert_ne!(
            item0, young_before,
            "young home not forwarded: frame was not in the remembered set"
        );
    }
}
