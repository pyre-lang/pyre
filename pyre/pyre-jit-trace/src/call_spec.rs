//! Canonical call/effect specification for `PyFrame`.
//!
//! This module is intentionally data-only so `build.rs` can share the same
//! translator-facing call/effect contract without re-encoding helper policy in
//! the analyzer.

pub const PYFRAME_CALL_OWNER_ROOT: &str = "PyFrame";

#[derive(Clone, Copy)]
pub enum CallEffectKind {
    Elidable,
    Residual,
    /// A callee whose effects are stated here rather than derived from a
    /// graph (callee census).
    ///
    /// `Elidable` and `Residual` are shorthands for two whole rows; this arm
    /// carries the row itself, which is what a callee with **no graph** needs.
    /// It is upstream's `analyze_external_call` answer — the middle arm of
    /// `graphanalyze.py` — reached through the only channel this side
    /// has for it: a spelling-keyed table, not a source attribute.
    ///
    /// `#[expect(dead_code)]` because **no row uses this arm yet.** The rows
    /// are blocked on a callee census: a re-run on a fresh LLBC set names
    /// the callees and, decisively, their SEGMENTATION — and a
    /// `FunctionPath` override matches on the split, so a row written from a
    /// `::`-joined listing is a guess that fails silently. Delete the
    /// attribute along with the first row.
    #[expect(dead_code)]
    Declared(DeclaredEffects),
}

/// `effectinfo.py:17-24` `EF_*`, minus `EF_RANDOM_EFFECTS`.
///
/// A table entry states what a callee **does**. `EF_RANDOM_EFFECTS` is the
/// answer for a callee nobody can describe — it is the absence of a
/// declaration, so it has no member here.
///
/// Omitting it is load-bearing, not tidiness: it is what makes upstream's
///
/// ```text
/// assert not (elidable_function and random_effects_on_gcobjs)
/// ```
///
/// (`rpython/rtyper/lltypesystem/rffi.py`, inside `llexternal`) hold **by
/// construction** — the illegal pairing cannot be written. Upstream enforces
/// it at the declaration site for externals, which is exactly what this table
/// is, so the constraint belongs here and not downstream in an analyzer.
///
/// The converse must stay writable. `elidable` **plus a concrete benign
/// row** is legal upstream and is the whole point: `random_effects_on_gcobjs`
/// derives from `invoke_around_handlers or has_callback` (`rffi.py`),
/// i.e. "can release the GIL or can run a callback". A primitive like
/// `core::ptr::null` does neither. A type that forbade elidable-with-a-row
/// would delete the population this arm exists to serve, at compile time,
/// while looking like a hardening.
///
/// `#[expect(dead_code)]` for the same reason as `CallEffectKind::Declared`:
/// every member is *matched* by the translators, none is *constructed*, and
/// none will be until the first row lands. Delete it with that row.
#[expect(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeclaredExtraEffect {
    /// `EF_ELIDABLE_CANNOT_RAISE` — the row a pure primitive wants.
    ElidableCannotRaise,
    /// `EF_LOOPINVARIANT`
    LoopInvariant,
    /// `EF_CANNOT_RAISE`
    CannotRaise,
    /// `EF_ELIDABLE_OR_MEMORYERROR`
    ElidableOrMemoryError,
    /// `EF_ELIDABLE_CAN_RAISE`
    ElidableCanRaise,
    /// `EF_CAN_RAISE`
    CanRaise,
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    ForcesVirtualOrVirtualizable,
}

/// One declared effects row.
///
/// The six read/write descr sets are **not** expressible here and are left
/// at `EffectInfo`'s default (empty, i.e. "touches no field or array"). A
/// `const` table has no way to name a descr — the descr universe does not
/// exist until the codewriter has run. Empty is the correct row for the
/// primitives this arm targets (`core::ptr::null` reads and writes nothing),
/// and it is **wrong** for anything that touches memory, so an entry for such
/// a callee must not be written until that channel exists.
#[derive(Clone, Copy)]
pub struct DeclaredEffects {
    pub extra: DeclaredExtraEffect,
    /// `effectinfo.py:125 can_collect` — upstream defaults it to `True`.
    /// `false` states the callee cannot trigger a collection.
    pub can_collect: bool,
    /// Whether the callee can invalidate a quasi-immutable.
    pub can_invalidate: bool,
}

#[derive(Clone, Copy)]
pub enum CallTargetSpec {
    Method {
        name: &'static str,
        receiver_root: &'static str,
    },
    FunctionPath(&'static [&'static str]),
}

#[derive(Clone, Copy)]
pub struct CallEffectSpec {
    pub target: CallTargetSpec,
    pub effect: CallEffectKind,
}

/// Every row below is currently **inert** — the per-entry match counter reads
/// 0 for all 88. The `FunctionPath` rows carry a bare leaf name while the call
/// sites resolve to the module-qualified path (`["opcode_binary_op"]` vs
/// `["pyre_interpreter", "pyopcode", "opcode_binary_op"]`), and non-`Method`
/// patterns are compared by full structural equality.
///
/// Do **not** repair that by relaxing the comparison to the leaf name. An
/// unmatched row is inert and safe; a wrongly-matched row installs another
/// function's declared effects on the call, and matching on the leaf alone
/// would make an `RBigInt::sub` row also capture `time::Instant::sub` and
/// `core::ops::arith::<Impl>::sub`. Fix the spelling in the row, not the
/// predicate — see `CallEffectOverride::target` for how to read the spelling a
/// call site actually carries, and verify against the census match counter.
pub const PYFRAME_CALL_EFFECTS: &[CallEffectSpec] = &[
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "peek_at",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "push_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "pop_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "call_callable",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "call_function_ex",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "call_kw",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "truth_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "bool_value_from_truth",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "concrete_truth_as_bool",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "to_bool",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "swap_values",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "copy_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_global",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_global",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_global_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "delete_global",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_name",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_name_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_from_dict_or_globals",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_from_dict_or_deref",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_name",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_name_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "delete_name",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "iter_next",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "for_iter",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "end_for",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "pop_iter",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "return_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_list",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_tuple",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unpack_sequence",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unpack_ex",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_subscr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "list_append",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    // ── Trait handler methods (called by opcode_* free functions) ──
    // These have generic receivers (e.g. handler: &mut H) in the source,
    // but receiver matching handles lowercase names as wildcards.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_local_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_local_checked_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_local_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    // binary_value, compare_value, unary_negative_value, unary_invert_value
    // are generic object-space operations (space.add, space.neg, space.eq etc.).
    // Type specialization happens at JIT trace time, not at codewriter
    // classification time. These are NOT int-specific.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "binary_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "compare_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unary_negative_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unary_invert_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "set_next_instr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    // MAKE_FUNCTION creates a function object from code+defaults — it does
    // NOT call a function. Residual, not FunctionCall.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "make_function",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_attr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_attr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_map",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_set",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    // ensure_iter_value is GET_ITER semantics (iterable → iterator),
    // not next/branch semantics. Residual, not RangeIterNext.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "ensure_iter_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "iter_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    // ── opcode_* free functions (FunctionPath targets) ──
    // These are called by OpcodeStepExecutor default methods.
    // Adding them here lets the classifier match at the default method
    // graph level without needing to follow into the free function body.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_call"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_return_value"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_const"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_small_int"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_fast_checked"]),
        effect: CallEffectKind::Residual,
    },
    // Multi-local superinstructions: these are CPython 3.12+ fused opcodes
    // with no PyPy equivalent. Cannot be represented as a single LocalRead
    // or LocalWrite since they perform two distinct local operations.
    // Classified as Residual — the JIT decomposes them at trace time.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_fast_pair_checked"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_fast_load_fast"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_fast"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_fast_load_fast"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_fast_store_fast"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_name"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_global"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_name"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_global"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_pop_top"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_push_null"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_copy_value"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_swap"]),
        effect: CallEffectKind::Residual,
    },
    // Generic object-space operations — type specialization at trace time.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_binary_op"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_compare_op"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unary_negative"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unary_invert"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unary_not"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_jump_forward"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_jump_backward"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_pop_jump_if_false"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_pop_jump_if_true"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_build_list"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_build_tuple"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_build_map"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_subscr"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_list_append"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unpack_sequence"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_attr"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_attr"]),
        effect: CallEffectKind::Residual,
    },
    // GET_ITER is space.iter() — converts iterable to iterator.
    // NOT next/branch semantics. Residual.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_get_iter"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_for_iter"]),
        effect: CallEffectKind::Residual,
    },
    // MAKE_FUNCTION creates a function object from code+defaults,
    // it does NOT call a function. Residual, not FunctionCall.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_make_function"]),
        effect: CallEffectKind::Residual,
    },
];
