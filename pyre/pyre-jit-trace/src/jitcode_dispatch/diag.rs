//! pc-map audit probes.
//!
//! **Parity:** pyre-specific — the `PYRE_PCMAP_*` audit probes have no
//! `rpython/jit/metainterp/` counterpart (PyPy's pc handling is
//! codewriter-side).
//!
//! Extracted verbatim from `jitcode_dispatch/mod.rs`: the
//! `skip_python_trivia_forward` boundary walker and the report-only
//! `PYRE_PCMAP_*` audit probes.

use super::*;

/// `PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT` is a report-only census for the
/// recipe resume-coordinate result-color reader and the multi-frame callee
/// diagnostic's inversion. The optional `_PROBE` receives a fire row followed
/// by its verdict, since `check.py` discards diagnostic stderr.
pub(crate) fn pcmap_recipe_resultcolor_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT").is_some())
}

/// `PYRE_PCMAP_CONTAINING_AUDIT`: assert the Slice-B floor-only depth twin
/// (`depth_containing_for_jitcode_pc`) equals the raw
/// `depth_at_py_pc[vstack_containing_py_pc(jit_pc)]` read at both consumer
/// seams. Off in production; the gated branch is the only added code.
pub(crate) fn pcmap_containing_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_CONTAINING_AUDIT").is_some())
}

/// `PYRE_VSTACK_EXACT_AUDIT`: report where the per-emission exact segmentation
/// (`py_exact_by_jit_pc`) disagrees with the floor tier the vstack mirror
/// currently resolves its boundary from. The mirror INFERS whether a Python
/// opcode retired; the exact table states it. Every disagreement is a case the
/// inference must special-case, so this is the gate that has to be quiet
/// before any consumer switches over. Off in production.
pub(crate) fn vstack_exact_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_VSTACK_EXACT_AUDIT").is_some())
}

/// `PYRE_PCMAP_AFTERRESIDUAL_AUDIT`: assert the Slice-C after-residual depth
/// twin (`depth_after_residual_for_jitcode_pc`) equals the raw
/// `depth_at_py_pc[semantic_fallthrough_pc(containing_py_pc_for_jitcode_pc(jit_pc))]`
/// read at each consumer seam. Off in production; the gated branch is the only
/// added code.
pub(crate) fn pcmap_afterresidual_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_AFTERRESIDUAL_AUDIT").is_some())
}

/// `PYRE_FBW_STRICT_DIAG`: name the opcode that rejected a callee from the
/// strict straight-line fast path.
///
/// Cached like the audit flags above, and for the reason `majit-backend-dynasm`
/// states at `majit_log_enabled`: `std::env::var_os` takes a global lock and
/// walks the env table on every call, and this one is read from inside
/// `callee_fast_path_inlinable`'s per-opcode decode loop.
pub(crate) fn fbw_strict_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_STRICT_DIAG").is_some())
}

/// `PYRE_FBW_INLINE_DIAG`: name the reason a call site declined to inline.
pub(crate) fn fbw_inline_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_INLINE_DIAG").is_some())
}

/// `PYRE_NO_BINOP_REWIND`: stop treating a `BINARY_OP` / `COMPARE_OP` dunder
/// entry as a boundary the abort can rewind to.
///
/// On by default; the variable only turns it off, so a bisection can name this
/// change in one command.  It stands on two rewinds a non-CALL entry did not
/// have before.  The record-time one is the trace cut the caller takes on a
/// `NotImplemented` result, under the same all-clear odometer reading the
/// un-lowered-helper rollback uses.  The runtime one converts the live
/// framestack through `convert_and_run_from_pyjitpl` (`blackhole.py`);
/// `caller_operand_slots` is where that entry's `[lhs, rhs]` operand image
/// comes from, and reading the image off a CALL residual's operand list
/// instead is what resumed one operand short at a `BINARY_OP`.
///
/// What it does NOT widen is the promise the entry makes about commits.  A
/// body that commits and then answers `NotImplemented` has no sound exit in
/// either direction, and no reading of the code object separates one from a
/// body that does not, so the admission does not try: the descent runs under
/// `BinopRewindInlineGuard` and `fbw_binop_rewind_refuse_commit` refuses the
/// first commit before it runs.
pub(crate) fn binop_rewind_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_NO_BINOP_REWIND").is_none())
}

/// `PYRE_FBW_MF_DIAG`: multi-frame callee window diagnostics.
pub(crate) fn fbw_mf_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_MF_DIAG").is_some())
}

/// `PYRE_P2_DIAG`: self-recursive CALL_ASSEMBLER fold diagnostics.
pub(crate) fn p2_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_P2_DIAG").is_some())
}

/// `PYRE_BRIDGE_LATCH_AUDIT`: report what the bridge sub-walk's abort latch
/// would have been refused for at the walk-level latch site.
///
/// `drive_bridge_frame_subwalk` latches on any `Err`, where the walk-level site
/// first requires the abort coordinate, `leaves_complete_image`, a carrier
/// exclusion and `!abort_blackhole_latched`. Two of those are observable from
/// the bridge site without inventing state: whether an image is *already*
/// latched — the innermost abort owns the handoff, so a second latch replaces
/// it with one rooted at the caller — and whether the error class leaves a
/// complete image.
///
/// Both must be sampled **before** latching. Reading `abort_blackhole_latched`
/// afterwards always reports `true` and witnesses nothing; the multi-frame arm
/// this site always takes also prints nothing on accept, which is why the
/// question has had no answer.
pub(crate) fn bridge_latch_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_BRIDGE_LATCH_AUDIT").is_some())
}

pub(crate) fn pcmap_recipe_resultcolor_audit_probe(site: &'static str, verdict: &'static str) {
    if let Some(path) = std::env::var_os("PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT_PROBE") {
        use std::io::Write;

        if let Ok(mut probe) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(probe, "recipe_resultcolor\t{site}\t{verdict}");
        }
    }
}

/// Resolve the authoritative paused-parent coordinate at a Python-native
/// consumer. A missing JitCode/code object is reported to the caller as the
/// same multi-frame snapshot decline used by nearby unavailable-coordinate
/// paths; it is never a panic.
pub(crate) fn resolve_parent_resume_py_pc(parent: &InlineParentFrame) -> Option<u32> {
    match parent.resume_coord {
        ParentResumeCoord::Backxlat(jitcode_pc) => {
            // #73: read the forward py_pc twin, a codewriter-built
            // trivia-normalized twin of the containing coordinate. The
            // containing lookup survives only for the empty-twin class.
            let twin = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
                .and_then(|pjc| pjc.forward_py_pc_for_jitcode_pc(jitcode_pc));
            Some(twin.unwrap_or_else(|| {
                crate::py_coord::trivia_normalized_py_pc_for_jitcode_pc(
                    parent.jitcode_index as i32,
                    jitcode_pc as i32,
                ) as u32
            }))
        }
        ParentResumeCoord::CallFallthrough(call_jit_pc) => {
            let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
            else {
                return None;
            };
            if pjc.code_ptr.is_null() {
                return None;
            }
            // #73 Slice 4: read the forward after-residual fallthrough twin.
            // The containing lookup survives only for the empty-twin class
            // (populated code with no Python map) and as the audit oracle.
            let legacy = || {
                let call_py_pc =
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, call_jit_pc)
                        as usize;
                let code = unsafe { &*pjc.code_ptr };
                crate::pyjitpl::semantic_fallthrough_pc(code, call_py_pc) as u32
            };
            let twin = pjc
                .after_residual_fallthrough_py_pc_populated()
                .then(|| pjc.after_residual_fallthrough_py_pc_for_jitcode_pc(call_jit_pc))
                .flatten();
            match twin {
                Some(ft) => {
                    if pcmap_afterresidual_audit_enabled() {
                        assert_eq!(
                            ft,
                            legacy(),
                            "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: parent-resume fallthrough-py twin diverged at jit_pc {call_jit_pc}"
                        );
                    }
                    Some(ft)
                }
                None => Some(legacy()),
            }
        }
    }
}

/// Resolve an in-flight body channel exactly where a stash match needs its
/// Python body pc. A missing JitCode entry is deliberately `None`: callers
/// treat it as no match and retain the legacy replay/delivery fallback.
pub(crate) fn inflight_foriter_body_pc(body: InflightForiterBody) -> Option<usize> {
    match body {
        InflightForiterBody::Py(body_pc) => Some(body_pc),
        InflightForiterBody::Jit {
            jitcode_index,
            op_pc,
        } => crate::state::pyjitcode_for_jitcode_index(jitcode_index).map(|jc| {
            crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.metadata, op_pc) as usize + 1
        }),
    }
}

/// Capture the native coordinates that identify a `for_iter_next` residual.
/// The Python continue-arm fallthrough is intentionally not derived here.
///
/// `op_pc` is an offset into the JitCode the walk is currently executing, so
/// the identity paired with it must be that JitCode's.  Inside an inline
/// sub-walk that is the callee's ([`InlineCalleeConsts::jitcode_index`], the
/// same resolution `build_multi_frame_miframe` applies to its innermost
/// frame); `fbw_mode.snapshot_sym` still names the outer portal.  Pairing the
/// portal with a callee offset invents a coordinate: `inflight_foriter_body_pc`
/// resolves it through the CALLER's pc tables and answers with a Python pc that
/// belongs to neither loop, so a callee loop's item is stashed under an
/// identity no resume coordinate can match, and a caller loop that happens to
/// resolve to the same pc has its own in-flight entry truncated away and
/// replaced by the callee's item.
pub(crate) fn fbw_foriter_body_from_op_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Option<InflightForiterBody> {
    if let Some(consts) = ctx.inline_callee_consts {
        return Some(InflightForiterBody::Jit {
            jitcode_index: consts.jitcode_index,
            op_pc,
        });
    }
    let snapshot_sym = ctx.fbw_mode.snapshot_sym;
    if snapshot_sym.is_null() {
        return None;
    }
    // SAFETY: the snapshot root stays live for the full-body walk. Only the
    // immutable JitCode identity is read here.
    let sym = unsafe { &*snapshot_sym };
    if sym.jitcode().is_null() {
        return None;
    }
    Some(InflightForiterBody::Jit {
        jitcode_index: unsafe { (*sym.jitcode()).index as i32 },
        op_pc,
    })
}

/// Forward-skip Python trivia (`Cache` / `ExtendedArg` / `Resume` / `Nop`
/// / `NotTaken`) from `py_pc` to the next executable opcode.  Mirrors the
/// forward trivia walk in [`crate::pyjitpl::semantic_fallthrough_pc`]
/// but starts AT `py_pc` (not `py_pc + 1`) so a coordinate that already
/// points at trivia is advanced.  A resume coordinate must be a real
/// opcode boundary; the resume reader's own backtrack walks trivia
/// BACKWARD, which is wrong for a `NOT_TAKEN` branch-target coordinate.
pub fn skip_python_trivia_forward(code: &pyre_interpreter::CodeObject, mut py_pc: usize) -> usize {
    use pyre_interpreter::bytecode::Instruction;
    loop {
        match pyre_interpreter::decode_instruction_at(code, py_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => py_pc += 1,
            _ => return py_pc,
        }
    }
}

/// Every hand-written trace-time specialization, its call-site file, and — for
/// a fold reached only from another fold — the outer fold whose `fired` count
/// already contains it.
///
/// `site` names every `jitcode_dispatch` file whose `spec_gate` call sites
/// drive this row, without the `.rs`; a row gated from more than one file
/// joins them with `,` (no space — the census prints the column as the value
/// of a `site=` field on a space-separated line).  It is a reader's pointer
/// to where the gate lives, not something the census parses: `check.py`'s
/// `SPEC_CENSUS_FOLD_RE` reads only `fold=` and `fired=`.
///
/// Declared beside the counters so that a fold added next to an existing gated
/// one is hard to leave unnamed.  It is NOT a complete census, and cannot
/// become one by adding rows: this table names a fold by its function, and
/// three shapes have no name to give.
///
/// * **Functionless replace** — the emit is inlined into a `match` arm, so
///   there is no function to name.
/// * **Registry** — one entry more means one fold more, with no new call site
///   and no new function (`try_fold_registered_symbolic_residual`).
/// * **Elision** — the arm recognises a shape and emits *nothing*, so a census
///   keyed on "what did this fold emit instead" has nothing to key on.  The
///   three `fbw_strict_fold_frame_reg` arms in `vable_ops.rs` are this: a
///   store to the current inline level's own unseeded portal frame is a
///   virtual-field write, folded away with no `SETFIELD_GC` recorded.  Their
///   recognisers (`fbw_strict_fold_frame_reg`,
///   `folded_store_is_observable_local`) are predicates — they cannot emit,
///   because the eliding IS the arm.
///
/// Treat a count taken from these rows as a lower bound on the fold
/// population, not a total.  `site` is `"none"` for a fold with no call site
/// at all.
///
/// `store_attr` occupies two rows because its single call site has two firing
/// outcomes that must not be summed: `Direct` folds the store away, while
/// `Residual` only replaces the generic setattr residual's descr, arguments and
/// effect and lets the rewritten call continue through the ordinary record
/// path.  Both are `Ok(Some(_))`, so a single row would read as fully foldable
/// when part of it is a descr swap.  The two rows share one call, so they carry
/// identical `consulted` and `suppressed` counts and only `fired` is split;
/// `parent` marks the second row as a split of the first so the reader does not
/// sum them.
///
/// Nothing about a row is written twice: `SPEC_FOLD_COUNT` reads `len()`, and
/// the `SpecFold` a call site names is generated from the same list, so adding
/// a row cannot leave a hand-written length or a stale label behind it.
/// Declare the fold table.  The rows are the source: the `SpecFold` variants a
/// call site names, the `SPEC_FOLD_ROWS` the census prints, and the width of
/// the counter arrays all come out of the same list, in the same order, so a
/// row and its call site cannot disagree.  A label that is not a row does not
/// compile, and a row no site names is an unconstructed variant, which
/// `dead_code` reports.
macro_rules! spec_folds {
    ($($variant:ident => ($label:literal, $site:literal, $parent:literal),)*) => {
        /// The row a gated call site names.  Its discriminant is that row's
        /// index into the counter arrays, which is why the two orders are one
        /// order.
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) enum SpecFold {
            $($variant,)*
        }

        impl SpecFold {
            /// This row's index into [`SPEC_FOLD_ROWS`] and the counter arrays.
            const fn index(self) -> usize {
                self as usize
            }
        }

        #[rustfmt::skip]
        pub const SPEC_FOLD_ROWS: &[(&str, &str, &str)] = &[
            // (label, site, parent)
            $(($label, $site, $parent),)*
        ];
    };
}

#[rustfmt::skip]
spec_folds! {
    // variant                    label                       site             parent
    BinaryOpDescent      => ("binary_op_descent",        "residual_call,inline_call", "-"),
    CompareOpDescent     => ("compare_op_descent",       "residual_call", "-"),
    Setslice             => ("setslice",                 "residual_call", "-"),
    GetIter              => ("get_iter",                 "residual_call", "-"),
    ForIterNext          => ("for_iter_next",            "residual_call", "-"),
    MakeFunction         => ("make_function",            "residual_call", "-"),
    Newtuple             => ("newtuple",                 "residual_call", "-"),
    NewtupleObject       => ("newtuple_object",          "residual_call", "-"),
    Newlist              => ("newlist",                  "residual_call", "-"),
    BuiltinLenDescent    => ("builtin_len_descent",      "inline_call",   "-"),
    BuiltinIsinstance    => ("builtin_isinstance",       "residual_call", "-"),
    BuiltinDictGet       => ("builtin_dict_get",         "residual_call", "-"),
    BuiltinTypeGetattr   => ("builtin_type_getattr",     "residual_call", "-"),
    BuiltinGetattr       => ("builtin_getattr",          "residual_call", "-"),
    BuiltinRange         => ("builtin_range",            "residual_call", "-"),
    BuiltinLocals        => ("builtin_locals",           "residual_call", "-"),
    SysGetframe          => ("sys_getframe",             "residual_call", "-"),
    SysExcInfo           => ("sys_exc_info",             "residual_call", "-"),
    MathFrexp            => ("math_frexp",               "residual_call", "-"),
    BuiltinDivmodDescent => ("builtin_divmod_descent",   "residual_call", "-"),
    ExceptionReduce      => ("exception_reduce",         "residual_call", "-"),
    SetAddMethod         => ("set_add_method",           "residual_call", "-"),
    StoreAttrDirect      => ("store_attr_direct",        "residual_call", "-"),
    LoadAttr             => ("load_attr",                "residual_call", "-"),
    LoadAttrPureRead     => ("load_attr_pure_read",      "specialize",    "load_attr"),
    LoadTypeAttr         => ("load_type_attr",           "residual_call", "-"),
    LoadMethodAttr       => ("load_method_attr",         "residual_call", "-"),
    LoadClassmethodAttr  => ("load_classmethod_attr",    "residual_call", "-"),
    LoadBoundMethodAttr  => ("load_bound_method_attr",   "residual_call", "-"),
    Subscr               => ("subscr",                   "residual_call,inline_call", "-"),
    BinaryOpIntZeroDiv   => ("binary_op_int_zero_div",   "residual_call,inline_call", "-"),
    Unpack               => ("unpack",                   "residual_call", "-"),
    SubscrTupleDescent   => ("subscr_tuple_descent",     "specialize",    "subscr"),
    SubscrTupleSlice2    => ("subscr_tuple_slice2",      "specialize",    "subscr"),
    ZipTwoTupleIters     => ("zip_two_tuple_iters",      "specialize",    "for_iter_next"),
    InstanceNext         => ("instance_next",            "residual_call", "-"),
    InstanceIter         => ("instance_iter",            "residual_call", "-"),
    GeneratorNext        => ("generator_next",           "residual_call", "-"),
    FrameLasti           => ("frame_lasti",              "specialize",    "load_attr"),
    LoadDeref            => ("load_deref",               "residual_call", "-"),
    FrameLineno          => ("frame_lineno",             "specialize",    "load_attr"),
    BareSuperCall        => ("bare_super_call",          "residual_call", "-"),
    BareSuperVirtual     => ("bare_super_virtual",       "residual_call", "-"),
    TwoArgSuperCall      => ("two_arg_super_call",       "residual_call", "-"),
    LoadAttrOnSuper      => ("load_attr_on_super",       "specialize",    "load_attr"),
    KwonlyDefaultsInline => ("kwonly_defaults_inline",   "inline_call",   "-"),
    LoadSuperAttr        => ("load_super_attr",          "residual_call", "-"),
    SuperAttrUnwrap      => ("super_attr_unwrap",        "residual_call", "load_super_attr"),
    LoadSpecialMethod    => ("load_special_method",      "residual_call", "-"),
    FormatWithSpecInt    => ("format_with_spec_int",     "residual_call", "-"),
    SubscrUserGetitem    => ("subscr_user_getitem",      "inline_call",   "subscr"),
    LenUserDunder        => ("len_user_dunder",          "inline_call",   "-"),
}

const SPEC_FOLD_COUNT: usize = SPEC_FOLD_ROWS.len();

static SPEC_CONSULTED: [std::sync::atomic::AtomicU64; SPEC_FOLD_COUNT] = {
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z; SPEC_FOLD_COUNT]
};
static SPEC_FIRED: [std::sync::atomic::AtomicU64; SPEC_FOLD_COUNT] = {
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z; SPEC_FOLD_COUNT]
};
/// How many times the kill switch skipped the call.  Reported per fold so a
/// suppression run states what it actually suppressed rather than what it was
/// asked to suppress.
static SPEC_SUPPRESSED: [std::sync::atomic::AtomicU64; SPEC_FOLD_COUNT] = {
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z; SPEC_FOLD_COUNT]
};
static INSTANCE_NEXT_FORITER_ROUTE_GUARDS_KEYED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static INSTANCE_NEXT_FORITER_CALLEE_GUARDS_CAPTURED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static INSTANCE_NEXT_FORITER_CALLEE_GUARDS_KEYED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Set by [`spec_census_enable`], for a host that has no environment to read
/// `PYRE_FBW_SPEC_CENSUS` from.
static SPEC_CENSUS_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Arm the specialization census without an environment variable.
///
/// The wasm guest reads no environment, so the switch has to arrive as a
/// call. It must precede the first gated dispatch:
/// [`fbw_spec_census_enabled`] latches on first read, and a census armed
/// after that read stays off — the same ordering contract
/// `pyre_jit_guard_census_enable` carries.
pub fn spec_census_enable() {
    SPEC_CENSUS_ARMED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// `PYRE_FBW_SPEC_CENSUS`: per-fold consulted/fired tallies for the
/// hand-written trace-time specializations.  Off by default; the gated branch
/// is the only added work on the dispatch path.
pub(crate) fn fbw_spec_census_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("PYRE_FBW_SPEC_CENSUS").is_some()
            || SPEC_CENSUS_ARMED.load(std::sync::atomic::Ordering::Relaxed)
    })
}

pub(crate) fn spec_census_record_instance_next_route_guard_keyed() {
    if fbw_spec_census_enabled() {
        INSTANCE_NEXT_FORITER_ROUTE_GUARDS_KEYED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn spec_census_record_instance_next_callee_guard(keyed: bool) {
    if !fbw_spec_census_enabled() {
        return;
    }
    let ordering = std::sync::atomic::Ordering::Relaxed;
    INSTANCE_NEXT_FORITER_CALLEE_GUARDS_CAPTURED.fetch_add(1, ordering);
    if keyed {
        INSTANCE_NEXT_FORITER_CALLEE_GUARDS_KEYED.fetch_add(1, ordering);
    }
}

const FBW_DEPTH_HIST_BUCKETS: usize = 32;

static FBW_DEPTH_ENTRIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FBW_DEPTH_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FBW_DEPTH_HIST: [std::sync::atomic::AtomicU64; FBW_DEPTH_HIST_BUCKETS] = {
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z; FBW_DEPTH_HIST_BUCKETS]
};
static FBW_DEPTH_DEEPEST: parking_lot::Mutex<Vec<usize>> = parking_lot::Mutex::new(Vec::new());

/// `PYRE_FBW_DEPTH_CENSUS`: process-wide inline-walker descent depths and the
/// `w_code` chain for the deepest entry. Off by default; the cached test is the
/// only added work on the descent path when disabled.
#[inline]
pub(crate) fn fbw_depth_census_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_DEPTH_CENSUS").is_some())
}

/// Record the post-push inline depth. The portal frame at `framestack[0]` is
/// not an inlined callee and is not counted. The histogram's last slot is the
/// `>=32` bucket; all earlier slots correspond to their one-based depth.
pub(crate) fn fbw_depth_census_record(framestack: &[InlineFrame]) {
    let ordering = std::sync::atomic::Ordering::Relaxed;
    let inline = || framestack.iter().filter(|frame| !frame.is_portal);
    let depth = inline().count();
    debug_assert!(depth > 0);
    FBW_DEPTH_ENTRIES.fetch_add(1, ordering);
    FBW_DEPTH_HIST[depth.saturating_sub(1).min(FBW_DEPTH_HIST_BUCKETS - 1)].fetch_add(1, ordering);

    if depth > FBW_DEPTH_MAX.load(ordering) {
        let mut deepest = FBW_DEPTH_DEEPEST.lock();
        if depth > FBW_DEPTH_MAX.load(ordering) {
            deepest.clear();
            deepest.extend(inline().map(|frame| frame.w_code));
            FBW_DEPTH_MAX.store(depth, ordering);
        }
    }
}

/// One process-exit line for the inline-walker descent census.
pub fn fbw_depth_census_summary() -> String {
    use std::fmt::Write;

    let ordering = std::sync::atomic::Ordering::Relaxed;
    let deepest = FBW_DEPTH_DEEPEST.lock();
    let mut hist = String::new();
    for (idx, count) in FBW_DEPTH_HIST.iter().enumerate() {
        if idx != 0 {
            hist.push(',');
        }
        if idx + 1 == FBW_DEPTH_HIST_BUCKETS {
            write!(hist, ">=32:{}", count.load(ordering)).unwrap();
        } else {
            write!(hist, "{}:{}", idx + 1, count.load(ordering)).unwrap();
        }
    }
    let deepest = deepest
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "[fbw-depth] entries={} max_depth={} hist={hist} deepest=[{deepest}]\n",
        FBW_DEPTH_ENTRIES.load(ordering),
        FBW_DEPTH_MAX.load(ordering),
    )
}

/// Parse `PYRE_FBW_NO_SPECIALIZE` once into table-index bits and unknown
/// selector tokens.  The reserved `all` token turns off every
/// `SPEC_FOLD_ROWS` row and nothing else: the `try_walker_fold_*` trio
/// and the `try_walker_inline_*` descent entry points all stay live.
fn spec_suppression() -> &'static (SpecMask, Vec<String>) {
    static SUPPRESSION: std::sync::OnceLock<(SpecMask, Vec<String>)> = std::sync::OnceLock::new();
    SUPPRESSION.get_or_init(|| {
        let mut mask = SpecMask::none();
        let mut unknown = Vec::new();
        if let Some(selectors) = std::env::var_os("PYRE_FBW_NO_SPECIALIZE") {
            for selector in selectors.to_string_lossy().split(',') {
                let selector = selector.trim();
                if selector.is_empty() {
                    continue;
                }
                if selector == "all" {
                    mask = SpecMask::all();
                } else if let Some(idx) = spec_row_index(selector) {
                    mask.set(idx);
                } else {
                    unknown.push(selector.to_owned());
                }
            }
        }
        (mask, unknown)
    })
}

/// One bit per [`SPEC_FOLD_ROWS`] entry.  A single `u64` held the whole table
/// until the row count reached its width; the shape is the same, spread over
/// as many words as the table needs.
#[derive(Clone, Copy)]
struct SpecMask([u64; SPEC_FOLD_COUNT.div_ceil(u64::BITS as usize)]);

impl SpecMask {
    fn none() -> Self {
        Self([0; SPEC_FOLD_COUNT.div_ceil(u64::BITS as usize)])
    }

    fn all() -> Self {
        let mut mask = Self::none();
        for idx in 0..SPEC_FOLD_COUNT {
            mask.set(idx);
        }
        mask
    }

    fn set(&mut self, idx: usize) {
        self.0[idx / u64::BITS as usize] |= 1u64 << (idx % u64::BITS as usize);
    }

    fn get(&self, idx: usize) -> bool {
        self.0[idx / u64::BITS as usize] & (1u64 << (idx % u64::BITS as usize)) != 0
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }
}

fn spec_suppressed_mask() -> SpecMask {
    spec_suppression().0
}

fn spec_suppress_unknown() -> &'static [String] {
    &spec_suppression().1
}

/// Either axis armed.  A gate's fast path reads only this, so a default run
/// pays one cached load and one predictable branch per consult.
fn spec_instrumented() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fbw_spec_census_enabled() || !spec_suppressed_mask().is_empty())
}

fn spec_row_index(name: &str) -> Option<usize> {
    SPEC_FOLD_ROWS
        .iter()
        .position(|(label, _, _)| *label == name)
}

/// Gate and observe one hand-written specialization.
///
/// When `name` is suppressed the fold is NOT CALLED and the site sees
/// `Ok(None)` — the decline every one of these folds already produces for a
/// shape it cannot handle, and the decline path is uniformly the generic
/// residual.  Otherwise the call runs and its outcome is passed back unchanged,
/// including `Err`, so `?` at the call site behaves exactly as it did before
/// the wrap.
///
/// `consulted` counts calls that were made — in an `&&` chain that means every
/// preceding guard held, so `consulted == 0` says the site's guards never held.
/// `fired` counts `Ok(Some(_))`.  `consulted - fired` covers both `Ok(None)`
/// declines and `Err` propagations; an `Err` aborts the walk, so it is rare, and
/// it is counted as consulted and never as fired.
#[inline]
pub(crate) fn spec_gate<T, E>(
    fold: SpecFold,
    call: impl FnOnce() -> Result<Option<T>, E>,
) -> Result<Option<T>, E> {
    // Neither axis armed: one cached load, one predictable branch, then the
    // original call.  This is what every default run executes.
    if !spec_instrumented() {
        return call();
    }
    let idx = fold.index();
    if spec_suppressed_mask().get(idx) {
        SPEC_SUPPRESSED[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(None);
    }
    let outcome = call();
    if fbw_spec_census_enabled() {
        SPEC_CONSULTED[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if matches!(&outcome, Ok(Some(_))) {
            SPEC_FIRED[idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    outcome
}

/// One line per fold, sorted heaviest first, INCLUDING every fold with zero
/// firings — a zero is the answer this census exists to produce, so it is
/// printed, never omitted.
///
/// Deliberately not under the `[jit-stats]` prefix: `check.py`'s
/// `_jit_stats_snapshot` merges every such line, so this census uses its own
/// prefix.
pub fn spec_census_summary() -> String {
    let ordering = std::sync::atomic::Ordering::Relaxed;
    let mask = spec_suppressed_mask();
    let mut rows: Vec<_> = SPEC_FOLD_ROWS
        .iter()
        .enumerate()
        .map(|(idx, &(label, site, parent))| {
            (
                label,
                site,
                parent,
                SPEC_CONSULTED[idx].load(ordering),
                SPEC_FIRED[idx].load(ordering),
                SPEC_SUPPRESSED[idx].load(ordering),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.4.cmp(&a.4).then(b.3.cmp(&a.3)).then(a.0.cmp(b.0)));

    let consulted_total: u64 = rows.iter().map(|row| row.3).sum();
    let fired_total: u64 = rows.iter().map(|row| row.4).sum();
    let suppressed_total: u64 = rows.iter().map(|row| row.5).sum();
    let suppressed_names = if mask.is_empty() {
        "-".to_owned()
    } else {
        SPEC_FOLD_ROWS
            .iter()
            .enumerate()
            .filter_map(|(idx, (label, _, _))| mask.get(idx).then_some(*label))
            .collect::<Vec<_>>()
            .join(",")
    };
    let suppress_unknown = if spec_suppress_unknown().is_empty() {
        "-".to_owned()
    } else {
        spec_suppress_unknown().join(",")
    };
    let (direct_resume, call_replay) = super::inline_call::subwalk_resume_counts();
    let heap_clones = super::inline_call::subwalk_heap_clone_count();
    let mut summary = format!(
        "[spec-census] folds={} consulted_total={consulted_total} fired_total={fired_total} suppressed_total={suppressed_total} suppressed_names={suppressed_names} suppress_unknown={suppress_unknown} subwalk_direct_resume={direct_resume} subwalk_call_replay={call_replay} subwalk_heap_clones={heap_clones}\n",
        rows.len(),
    );
    summary.push_str(&format!(
        "[spec-census] instance_next_foriter route_guards_keyed={} callee_guards_captured={} callee_guards_keyed={}\n",
        INSTANCE_NEXT_FORITER_ROUTE_GUARDS_KEYED.load(ordering),
        INSTANCE_NEXT_FORITER_CALLEE_GUARDS_CAPTURED.load(ordering),
        INSTANCE_NEXT_FORITER_CALLEE_GUARDS_KEYED.load(ordering),
    ));
    for (label, site, parent, consulted, fired, suppressed) in rows {
        summary.push_str(&format!(
            "[spec-census] fold={label} consulted={consulted} fired={fired} suppressed={suppressed} site={site} parent={parent}\n"
        ));
    }
    summary
}
