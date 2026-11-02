//! Single source of truth for the per-helper policy byte that the
//! `__majit_call_policy_<name>()` accessor returns at runtime.
//!
//! Helpers tagged with `#[elidable]`, `#[dont_look_inside]`,
//! `#[jit_may_force]`, etc. emit a `(policy_byte, ...)` tuple whose
//! first element identifies the (return-kind × extraeffect) shape.
//! The byte feeds two consumers:
//!
//! 1. **Inferred policy lowering** (`jitcode_lower.rs::CallPolicySpec::Infer`)
//!    where the macro generated `match __policy { … }` block dispatches
//!    to the correct `JitCodeBuilder::<call_kind>_canonical_via_target*`
//!    method.
//! 2. **Inferred slot resolution** — the macro lowerer's inline `match
//!    __policy { … }` blocks at the `inferred_*_policy_check` and
//!    `slot_expr` sites classify the byte as
//!    `EffectInfoSlot::{CanRaise, CannotRaise, ElidableCanRaise,
//!    ElidableCannotRaise, ElidableOrMemerror, LoopInvariant}`.
//!    Mapping is generated inline rather than via a single named
//!    function because the slot decision is interleaved with
//!    cond_call / record_known_result kind-mismatch panics.
//!
//! RPython upstream has a single `EffectInfo` discriminant
//! (`rpython/jit/codewriter/effectinfo.py`); pyre's bytes encode
//! `EffectInfo × return_kind` because the `cond_call` /
//! `record_known_result` lowerers need to know the return kind from
//! the byte alone (the explicit `CallPolicyKind` carries both, but the
//! inferred path only sees the byte coming back from the helper's
//! `__majit_call_policy_<name>()` accessor).
//!
//! The mapping (`call.py` extraeffect ↔ pyre byte) is per result kind:
//! - **Void** (`bytes 1, 9, 17, 28`; slot 13 is reserved)
//! - **Int** (`bytes 2-4, 10, 18-20, 29`; slot 14 is reserved)
//! - **Ref** (`bytes 21-27, 30`)
//! - **Float**: inferred path returns `0` (UNSUPPORTED) because static
//!   float result-kind cannot be recovered; explicit
//!   `*_float_wrapped` policies carry the typed lowering instead.
//!
//! Adding a new byte requires a coordinated update of:
//! - `lib.rs::helper_policy_tokens_for_fn` (emit-side tuple),
//! - `jitcode_lower.rs` inferred-policy `slot_expr` match arms,
//! - validation tables (`inferred_conditional_call_policy_check`,
//!   `inferred_conditional_call_value_policy_check`,
//!   `inferred_record_known_result_policy_check`),
//! - inferred dispatch arms (stmt-form void at `lower_call_stmt`,
//!   value-form int at `lower_call_value_int`),
//! - `inferred_policy_live_condition` callsites (the `-live-` marker
//!   includes can-raise codes),
//! - `replay_kind_for_policy` and `is_wrapped_policy` (when adding a
//!   new `CallPolicyKind` enum variant).

// ── Void-return helper policy bytes ─────────────────────────────────

/// `#[dont_look_inside]` void — non-elidable, may raise (`EF_CAN_RAISE`).
pub(crate) const VOID_DONT_LOOK_INSIDE: u8 = 1;

/// `#[jit_may_force]` void — `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`.
pub(crate) const VOID_MAY_FORCE: u8 = 9;

// Reserved policy-byte slots 13 (void) and 14 (int). They were the
// macro-DSL `#[jit_release_gil]` bytes (`VOID_RELEASE_GIL` /
// `INT_RELEASE_GIL`). Later bytes keep their numbers. `llexternal`
// fills `EffectInfo.call_release_gil_target` instead.

/// `#[jit_loop_invariant]` void — `EF_LOOPINVARIANT`.
pub(crate) const VOID_LOOP_INVARIANT: u8 = 17;

/// `#[dont_look_inside_cannot_raise]` void — `EF_CANNOT_RAISE`.
/// Skips the trailing `-live-` (`jtransform.py:1681 calldescr_canraise`).
pub(crate) const VOID_DONT_LOOK_INSIDE_CANNOT_RAISE: u8 = 28;

// ── Int-return helper policy bytes ──────────────────────────────────

/// `#[dont_look_inside]` int — non-elidable, may raise.
pub(crate) const INT_DONT_LOOK_INSIDE: u8 = 2;

/// `#[elidable]` int — `EF_ELIDABLE_CAN_RAISE` (`call.py:297`).
pub(crate) const INT_ELIDABLE: u8 = 3;

/// `#[jit_inline]` int — pyre-only `inline_call_*` opcode that always
/// emits a trailing `-live-` (`jtransform.py:480-482`).  Slot 4 is the
/// inline byte for int helpers; ref/float helpers also carry inline at
/// runtime but go through wrapped explicit policies.
pub(crate) const INT_INLINE: u8 = 4;

/// `#[jit_may_force]` int.
pub(crate) const INT_MAY_FORCE: u8 = 10;

/// `#[jit_loop_invariant]` int.
pub(crate) const INT_LOOP_INVARIANT: u8 = 18;

/// `#[elidable_cannot_raise]` int — `EF_ELIDABLE_CANNOT_RAISE`
/// (`call.py:299`).  Skips the trailing `GUARD_NO_EXCEPTION`.
pub(crate) const INT_ELIDABLE_CANNOT_RAISE: u8 = 19;

/// `#[elidable_or_memerror]` int — `EF_ELIDABLE_OR_MEMORYERROR`
/// (`call.py:295`).
pub(crate) const INT_ELIDABLE_OR_MEMERROR: u8 = 20;

/// `#[dont_look_inside_cannot_raise]` int — `EF_CANNOT_RAISE`.
pub(crate) const INT_DONT_LOOK_INSIDE_CANNOT_RAISE: u8 = 29;

// ── Ref-return helper policy bytes ──────────────────────────────────

/// `#[elidable]` ref.
pub(crate) const REF_ELIDABLE: u8 = 21;

/// `#[elidable_cannot_raise]` ref.
pub(crate) const REF_ELIDABLE_CANNOT_RAISE: u8 = 22;

/// `#[elidable_or_memerror]` ref.
pub(crate) const REF_ELIDABLE_OR_MEMERROR: u8 = 23;

/// `#[jit_loop_invariant]` ref.
pub(crate) const REF_LOOP_INVARIANT: u8 = 24;

/// `#[dont_look_inside]` ref — non-elidable, may raise.
pub(crate) const REF_DONT_LOOK_INSIDE: u8 = 25;

/// `#[jit_may_force]` ref.
pub(crate) const REF_MAY_FORCE: u8 = 26;

/// `#[dont_look_inside_cannot_raise]` ref — `EF_CANNOT_RAISE`.
pub(crate) const REF_DONT_LOOK_INSIDE_CANNOT_RAISE: u8 = 30;

// ── Sentinel ────────────────────────────────────────────────────────

/// Inferred path produces this for helpers whose result kind cannot be
/// recovered statically (all float-return inferred paths) or whose
/// attribute is not modelled as an inferred policy.  Reaching this
/// byte at runtime triggers the `unsupported` arm in the dispatch
/// match (the inferred-policy check upstream of the dispatch already
/// rejects most cases earlier).
pub(crate) const UNSUPPORTED: u8 = 0;
