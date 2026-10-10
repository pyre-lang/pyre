//! Runtime call descriptor constructors.
//!
//! No `rpython/jit/metainterp/call_descr.py` file exists. This module is
//! the Rust runtime boundary for descriptors produced by
//! `rpython/jit/codewriter/call.py::getcalldescr` through
//! `cpu.calldescrof(...)` (`rpython/jit/backend/model.py`) and then
//! consumed by metainterp, blackhole, optimizer, and backend call paths.
//! Keeping the constructors here avoids a fake metainterp upstream file
//! while still making the call-descr surface explicit.

use std::sync::Arc;

use majit_backend::JitCellToken;
use majit_ir::effectinfo::EffectInfoCell;
use majit_ir::{
    CallDescr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, RuntimeHelperKind, Type,
};

/// Generic CallDescr for function call operations.
///
/// Stores per-call-site EffectInfo, matching RPython's
/// `effectinfo_from_writeanalyze` (call.py:320). The EI is wrapped in
/// `EffectInfoCell` so `compute_bitstrings` can install the compacted
/// bitstrings post-construction (see `EffectInfoCell` doc and
/// `Descr::set_effect_bitstrings` SAFETY note).
#[derive(Debug)]
struct MetaCallDescr {
    heapcache_index: u32,
    arg_types: Vec<Type>,
    /// Raw `descr.py` argument class string. `'S'` is an int-bank value
    /// passed as C `float`.
    arg_classes: String,
    result_type: Type,
    /// `descr.py get_result_type()` returns the RAW char.  Carrying it
    /// keeps `'S'` (singlefloat) and `'L'` distinguishable from the
    /// `'i'`/`'f'` they normalise to, which the resume path depends on when it
    /// rebuilds a `BhCallDescr` off this descr for `bh_call_i`.
    result_class: char,
    result_signed: bool,
    result_size: usize,
    effect_info: Arc<EffectInfoCell>,
}

/// `compile.py isinstance(descr, JitCellToken)` parity.
///
/// RPython's `op.getdescr()` for a `CALL_ASSEMBLER_*` op IS a `JitCellToken`
/// — `record_loop_or_bridge` reads `descr.number` directly and calls
/// `original.record_jump_to(descr)` without any indirection. majit cannot
/// inherit-from the trait, but it preserves the *identity* contract by
/// owning an `Arc<JitCellToken>` here. Callers (`direct_assembler_call`,
/// `compile_tmp_callback`) clone the same Arc that the warm cell /
/// `CompiledEntry::token` /` MemoryManager.alive_loops` already hold, so the
/// keepalive walker's downcast recovers the production token's strong
/// reference rather than a number-recovered side-table lookup.
#[derive(Debug)]
struct MetaCallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: Arc<JitCellToken>,
}

impl majit_ir::Descr for MetaCallDescr {
    fn index(&self) -> u32 {
        self.heapcache_index
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
    /// `effectinfo.py:537-538 setattr(ei, 'bitstring_*', …)` — invoked
    /// by `effectinfo::compute_bitstrings` after class assignment.
    /// Delegates to `EffectInfoCell::set_bitstrings` which encodes the
    /// single-writer setup-time mutation through `UnsafeCell` rather
    /// than the earlier raw-pointer cast (Rust aliasing model).
    fn set_effect_bitstrings(
        &self,
        readonly_descrs_fields: Option<Vec<u8>>,
        write_descrs_fields: Option<Vec<u8>>,
        readonly_descrs_arrays: Option<Vec<u8>>,
        write_descrs_arrays: Option<Vec<u8>>,
        readonly_descrs_interiorfields: Option<Vec<u8>>,
        write_descrs_interiorfields: Option<Vec<u8>>,
    ) {
        self.effect_info.set_bitstrings(
            readonly_descrs_fields,
            write_descrs_fields,
            readonly_descrs_arrays,
            write_descrs_arrays,
            readonly_descrs_interiorfields,
            write_descrs_interiorfields,
        );
    }
}

impl CallDescr for MetaCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn arg_classes(&self) -> String {
        if self.arg_classes.len() == self.arg_types.len() {
            self.arg_classes.clone()
        } else {
            majit_ir::descr::arg_classes_of(&self.arg_types)
        }
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    /// `descr.py get_result_type()` — the raw char, not the class the
    /// normalised `result_type` would derive.
    fn result_class(&self) -> char {
        self.result_class
    }
    fn result_size(&self) -> usize {
        self.result_size
    }
    /// `descr.py is_result_signed()` — `result_flag == FLAG_SIGNED`.
    ///
    /// `result_flag` is derived from the RAW result char, not from the
    /// normalised one: only `'i'` consults `result_signed`, and `'S'` takes
    /// FLAG_UNSIGNED while `'L'` takes FLAG_FLOAT even though they normalise
    /// to `Int` and `Float`.  The constructor's `result_signed` argument is
    /// `is_ffi_type_signed` at a dynamic call site, which answers `kind != 'u'`
    /// and so says `true` for a float; the derivation is what discards it.
    fn is_result_signed(&self) -> bool {
        self.result_class == 'i' && self.result_signed
    }
    fn get_extra_info(&self) -> &EffectInfo {
        self.effect_info.get()
    }
}

impl majit_ir::Descr for MetaCallAssemblerDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::descr::LoopTokenDescr> {
        Some(self)
    }
}

impl CallDescr for MetaCallAssemblerDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        8
    }
    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token.number)
    }
    fn call_virtualizable_index(&self) -> Option<usize> {
        self.target_token.virtualizable_arg_index()
    }
    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
}

impl majit_ir::descr::LoopTokenDescr for MetaCallAssemblerDescr {
    fn loop_token_number(&self) -> u64 {
        self.target_token.number
    }

    fn call_virtualizable_index(&self) -> Option<usize> {
        self.target_token.virtualizable_arg_index()
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.target_token)
    }
}

/// EffectInfo for a residual call whose callee has no heap-analyzer
/// output — `EffectInfo::MOST_GENERAL`.
///
/// `graphanalyze.py analyze` splits the direct-call case three
/// ways, and only ONE of them is the empty-set bottom:
///
/// * `:104-108` the callee is `external` → `analyze_external_call`,
///   whose `graphanalyze.py` default is `bottom_result()`;
/// * `:109-112` the callee has no `funcobj.graph` (nothing to analyze)
///   → `top_result()`;
/// * `:117-122 indirect_call` with `graphs is None` → `top_result()`.
///
/// `effectinfo.py` then turns a `top_set` result into
/// `EF_RANDOM_EFFECTS` with all six raw sets and all six bitstrings set
/// to `None`, which is exactly `EffectInfo.MOST_GENERAL`
/// (`effectinfo.py`). `call.py get_jitcode_calldescr`
/// uses the same constant for the calldescr it attaches to a JitCode.
///
/// A pyre residual callee reaching this function is the second case,
/// not the first: it is a Rust helper the codewriter pipeline never
/// analyzed, so its write set is unknown rather than empty. Handing out
/// the `bottom_result()` shape here would assert "writes nothing",
/// which `heap.rs force_from_effectinfo` would believe.
///
/// Callers that HAVE proved something about the callee must not use
/// this: `cannot_raise_effect_info()`,
/// `CANNOT_RAISE_NO_HEAP_EFFECT_INFO`, the `ELIDABLE_*` constants and
/// the analyzer-built EIs from `codewriter::call::getcalldescr` all
/// carry concrete (possibly empty) raw sets.
pub fn default_effect_info() -> EffectInfo {
    EffectInfo::MOST_GENERAL
}

/// `call.py elif self._canraise(op):` — `EF_CAN_RAISE` built
/// through `effectinfo_from_writeanalyze` with an empty analyzer result.
///
/// Shape: `extraeffect=CanRaise`, every `_*_descrs_*` raw set =
/// `Some(Vec::new())`, every `*_descrs_*` bitstring = `Some(Vec::new())`,
/// `can_collect=true` (`effectinfo.py` default).
/// `effectinfo.py`'s else-branch starts the raw sets at `[]` and
/// grows them with the analyzer's actual effects, so an analyzer that
/// reported nothing leaves them `[]` — distinct from `MOST_GENERAL`,
/// which the `effectinfo.py` invariant ties to raw = `None`.
///
/// This is the row for a callee the producer classified by hand: an
/// opaque leaf helper the author marked `#[dont_look_inside]`
/// (`rlib/jit.py`), whose upstream counterpart still has a graph for
/// the write analyzer to walk. It asserts "writes no field of any object
/// the trace already cached", which
/// [`default_effect_info`] deliberately does not.
pub fn can_raise_effect_info() -> EffectInfo {
    EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None)
}

/// Nursery-alloc residual: `EF_CAN_RAISE` with empty write sets —
/// an allocation publishes a fresh object and writes no field of any
/// object the trace already cached, so `graphanalyze.py analyze_external_call
/// analyze_external_call`'s `bottom_result()` is the honest answer.
/// Stamped with [`RuntimeHelperKind::NurseryAlloc`] so the dynasm CallR
/// genop emits an inline nursery bump; the tag does not change effect
/// analysis.
pub fn nursery_alloc_effect_info() -> EffectInfo {
    let mut ei = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
    ei.runtime_helper = RuntimeHelperKind::NurseryAlloc;
    ei
}

/// `EF_CANNOT_RAISE` analyzer-absent fallback — the `call.py
/// else:` row of `call.py getcalldescr` selected when
/// `self._canraise(op) == False`, fed through
/// `effectinfo_from_writeanalyze` with the
/// `graphanalyze.py analyze_external_call` default
/// (`bottom_result()` = empty set).
///
/// Shape: `extraeffect=CannotRaise`, every `_*_descrs_*` raw set =
/// `Some(Vec::new())`, every `*_descrs_*` bitstring = `Some(Vec::new())`,
/// `can_collect=true` (the writeanalyzer's
/// `effectinfo.py can_collect=True` default).
/// `effectinfo.py` else-branch builds empty raw sets when
/// the analyzer returns non-`top_set` effects.
///
/// Distinct from [`CANNOT_RAISE_NO_HEAP_EFFECT_INFO`] only by
/// `can_collect`: that const carries `can_collect=false` for helpers
/// the producer additionally asserts cannot trigger GC; the
/// analyzer-absent default here keeps the
/// `effectinfo_from_writeanalyze(can_collect=True)` PyPy default.
/// `check_can_raise()` (`effectinfo.py`) reads
/// `extraeffect > EF_CANNOT_RAISE` so the canonical walker omits
/// the trailing `GUARD_NO_EXCEPTION` for this slot.
pub fn cannot_raise_effect_info() -> EffectInfo {
    EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None)
}

/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` analyzer-absent fallback —
/// the `call.py if self.virtualizable_analyzer.analyze(op)`
/// row of `call.py getcalldescr`, fed through
/// `effectinfo_from_writeanalyze` with the
/// `graphanalyze.py analyze_external_call` default
/// (`bottom_result()` = empty set).
///
/// Shape: `extraeffect=ForcesVirtualOrVirtualizable`, every
/// `_*_descrs_*` raw set = `Some(Vec::new())`, every `*_descrs_*`
/// bitstring = `Some(Vec::new())`, `can_collect=true` (PyPy
/// `effectinfo.py` `if extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE:
/// can_collect = True`).
///
/// **Distinct from `MOST_GENERAL`**: `EF_RANDOM_EFFECTS` is reserved
/// for the `RandomEffectsAnalyzer` (`effectinfo.py
/// random_effects_on_gcobjs`) branch. `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
/// is the dedicated virtualizable-forcing slot — both pass
/// `check_forces_virtual_or_virtualizable()` via the `>=` test at
/// `effectinfo.py`, but only `RandomEffects` trips
/// `has_random_effects()` (`effectinfo.py`) and routes
/// `OptHeap` through `clean_caches`. Collapsing MayForce to
/// `MOST_GENERAL` over-invalidates the heap cache PyPy keeps live
/// for analyzer-empty virtualizable-forcing callees.
pub fn forces_virtual_or_virtualizable_effect_info() -> EffectInfo {
    EffectInfo::const_new(
        ExtraEffect::ForcesVirtualOrVirtualizable,
        OopSpecIndex::None,
    )
}

/// `EF_CANNOT_RAISE` for a callee that the producer statically knows
/// touches no heap state and cannot trigger GC — typically a flat TLS
/// read/write or a buffer-flush shim.  `call.py:320-324
/// effectinfo_from_writeanalyze` would compute empty
/// `readonly_descrs_*` / `write_descrs_*` bitsets and `can_collect =
/// False` from `read_analyzer` / `write_analyzer` / `collect_analyzer`
/// for such helpers.  Using [`cannot_raise_effect_info()`] for them is
/// the analyzer-absent conservative fallback, which over-reports the
/// callee as a heap mutator and inflates GC map / liveness work; this
/// constant is the matching analyzer-output for known-flat helpers.
pub const CANNOT_RAISE_NO_HEAP_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::CannotRaise,
    oopspecindex: OopSpecIndex::None,
    runtime_helper: RuntimeHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    descr_set_keys: Some(majit_ir::effectinfo::DescrSetKeysImage::Empty),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CANNOT_RAISE` for a callee that cannot trigger GC.
///
/// This is a representable analyzer output, not a special case:
/// `EffectInfo.__new__` takes `can_collect` as its own parameter, defaulting
/// to true, and `effectinfo_from_writeanalyze` forces it true only for
/// `extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`;
/// `EF_ELIDABLE_CANNOT_RAISE` is 0.  Elidable with `can_collect=false` is
/// therefore the analyzer output for an `@jit.elidable` leaf whose collect
/// analyzer clears it.
///
/// `EffectInfo.__new__` additionally empties `_write_descrs_*` for every
/// `EF_ELIDABLE_*`.  Those sets are already empty here, so no other field
/// differs from [`CANNOT_RAISE_NO_HEAP_EFFECT_INFO`].
///
/// This differs from [`ELIDABLE_CANNOT_RAISE_EFFECT_INFO`], which goes
/// through `const_new` and leaves `can_collect=true`, costing every backend
/// the gcmap bracket and reference-register spill these callees do not need.
pub const ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::None,
    runtime_helper: RuntimeHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    descr_set_keys: Some(majit_ir::effectinfo::DescrSetKeysImage::Empty),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CANNOT_RAISE` with `OS_INT_PY_DIV` oopspec — Python `//`
/// (floor division). RPython parity: jtransform.py:2046-2047
/// `_handle_int_special` classifies `int.py_div` as
/// `EF_ELIDABLE_CANNOT_RAISE`. Source-level zero/overflow wrappers
/// (`rint.py ll_int_py_div_zer`, `:429 ll_int_py_div_ovf_zer`)
/// are inlined into the calling graph before the JIT sees this
/// oopspec call; their checks become runtime guards in the trace,
/// not properties of this call descriptor. The optimizer's
/// `optimize_call_int_py_div` (rewrite.py:713-766) reads the
/// `OS_INT_PY_DIV` oopspec to specialize power-of-2 divisors to
/// `int_rshift`, constant 1 to identity, constant -1 to `int_neg`, etc.
/// Callee is pure: no heap touched, no GC trigger, no raise.
pub const INT_PY_DIV_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::IntPyDiv,
    runtime_helper: RuntimeHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    descr_set_keys: Some(majit_ir::effectinfo::DescrSetKeysImage::Empty),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// Counterpart of [`INT_PY_DIV_EFFECT_INFO`] for Python `%`. RPython
/// parity: jtransform.py classifies `int.py_mod` as
/// `EF_ELIDABLE_CANNOT_RAISE`; zero/overflow checks from the source
/// wrappers (`rint.py ll_int_py_mod_zer`, `:520
/// ll_int_py_mod_ovf_zer`) are inlined upstream of the JIT trace.
pub const INT_PY_MOD_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::IntPyMod,
    runtime_helper: RuntimeHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    descr_set_keys: Some(majit_ir::effectinfo::DescrSetKeysImage::Empty),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CANNOT_RAISE` with `OS_INT_UDIV` oopspec — unsigned `/`.
/// RPython parity: `rint.py ll_uint_py_div` carries
/// `@jit.oopspec("int.udiv(x, y)")`, and jtransform.py:2043-2047
/// `_handle_int_special` classifies every `int.*div`/`int.*mod` oopspec as
/// `EF_ELIDABLE_CANNOT_RAISE`.
///
/// Unsigned division deliberately has NO trace opcode. RPython removed
/// `UINT_FLOORDIV` from the resop set (2016-05-10, "Remove divisions and
/// modulos from regular JIT operations, uses oopspec calls") and routes it
/// through this residual call instead, exactly as the signed
/// [`INT_PY_DIV_EFFECT_INFO`] does — the oopspec index, not an opcode, is
/// what the optimizer matches on (`rewrite.rs` `optimize_call_int_udiv`).
/// Callee is pure: no heap touched, no GC trigger, no raise.
pub const UINT_PY_DIV_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::IntUdiv,
    runtime_helper: RuntimeHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    descr_set_keys: Some(majit_ir::effectinfo::DescrSetKeysImage::Empty),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// Counterpart of [`UINT_PY_DIV_EFFECT_INFO`] for unsigned `%`.
/// RPython parity: `rint.py ll_uint_py_mod` carries
/// `@jit.oopspec("int.umod(x, y)")`.
pub const UINT_PY_MOD_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::IntUmod,
    runtime_helper: RuntimeHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    descr_set_keys: Some(majit_ir::effectinfo::DescrSetKeysImage::Empty),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CANNOT_RAISE` (effectinfo.py:17). Selected by
/// `call.py getcalldescr` when `_canraise(op) == False` for an
/// elidable callee — `pyjitpl.py do_residual_call` records
/// `CALL_PURE_*` without the trailing `GUARD_NO_EXCEPTION` because
/// `effectinfo.check_can_raise()` (`effectinfo.py`) is false for
/// `extraeffect == 0`.
pub const ELIDABLE_CANNOT_RAISE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);

/// `EF_ELIDABLE_OR_MEMORYERROR` (effectinfo.py:20). Selected by
/// `call.py getcalldescr` when `_canraise(op) == "mem"` — i.e.
/// the elidable callee's only failure mode is `MemoryError`. Same
/// dispatch as `EF_ELIDABLE_CAN_RAISE` (`check_can_raise()` is true
/// for extraeffect ≥ 3) but distinguishes memory-only raises for the
/// optimizer.
pub const ELIDABLE_OR_MEMERROR_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableOrMemoryError, OopSpecIndex::None);

/// `rstr.py ll_strconcat` `@jit.oopspec('stroruni.concat(s1, s2)')` →
/// `effectinfo.py` `EF_ELIDABLE_OR_MEMORYERROR` + `OS_STR_CONCAT`.
/// Same row `ll_strconcat_effectinfo()` mints for the walker and for
/// `resume.py concat_strings`.
pub const STR_CONCAT_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableOrMemoryError, OopSpecIndex::StrConcat);

/// [`STR_CONCAT_EFFECT_INFO`] as a function, matching
/// `can_raise_effect_info` / `nursery_alloc_effect_info`.
pub fn str_concat_effect_info() -> EffectInfo {
    STR_CONCAT_EFFECT_INFO
}

/// `EF_ELIDABLE_CAN_RAISE` (effectinfo.py:21). Pure calls do not need
/// the conservative flush — `effectinfo_from_writeanalyze` (effectinfo.py:
/// 169-181) clears `_write_descrs_*` for elidable extraeffects. With
/// the bitsets at zero this becomes "no writes" inside
/// `force_from_effectinfo`, matching upstream.
pub const ELIDABLE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None);

/// `EF_LOOPINVARIANT` (effectinfo.py:18). Same write-mask treatment as
/// elidable; the trace optimizer recognises the opcode and skips cache
/// invalidation regardless of the bitsets.
pub const LOOPINVARIANT_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::LoopInvariant, OopSpecIndex::None);

/// Per-callee analyzer-result slot.  Mirrors `call.py getcalldescr`'s
/// `extraeffect` selection without the `raise_analyzer` /
/// `readwrite_analyzer` / `collect_analyzer` / `randomeffects_analyzer`
/// graph-based machinery (the analyzers operate on RPython low-level
/// graphs, which pyre does not have).  Producers that statically know
/// the callee's classification — typically because the helper carries
/// a `#[elidable]` / `#[elidable_cannot_raise]` / `#[dont_look_inside]`
/// attribute — pick the matching slot at registration time;
/// [`effect_info_for_slot`] resolves it to the corresponding
/// [`EffectInfo`] const at descr construction.
///
/// `MayForce` (`EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`) and `ReleaseGil`
/// (`EF_RANDOM_EFFECTS` + non-zero `call_release_gil_target`) are
/// deliberately omitted. `getcalldescr` fills `call_release_gil_target`
/// from `_call_aroundstate_target_`; a const slot cannot name that
/// funcptr.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EffectInfoSlot {
    /// `EF_CAN_RAISE` — `call.py elif self._canraise(op):`
    /// branch of `getcalldescr`, resolved through
    /// `can_raise_effect_info()`: `CanRaise` + `Some(empty)` raw sets
    /// and bitstrings.
    ///
    /// Picked by producers that classified the callee by hand — the
    /// `#[dont_look_inside]` policy bytes, whose upstream counterpart is
    /// a `@dont_look_inside` function the write analyzer still walks. It
    /// is not the "nothing was analyzed" answer: that one is
    /// [`EffectInfoSlot::Unanalyzed`]. No analyzer walks a macro helper's
    /// body, so `residual_call_*_canonical_via_target` promotes this slot
    /// to `MOST_GENERAL` (`JitCodeBuilder::residual_effect_info_for_target`);
    /// a helper that raises or writes nothing is declared with a precise
    /// policy (`residual_*_cannot_raise`, `residual_writes`) instead.
    #[default]
    CanRaise,
    /// `EF_RANDOM_EFFECTS` / [`EffectInfo::MOST_GENERAL`] — the
    /// `graphanalyze.py:109-112` "callee has no analyzable graph" →
    /// `top_result()` outcome, promoted by `effectinfo.py`.
    /// Resolved through [`default_effect_info()`].
    ///
    /// Picked by producers that classified nothing: a helper binding
    /// arbitrary interpreter execution, with no graph for a write
    /// analyzer to walk. `check_forces_virtual_or_virtualizable()` holds
    /// for this row (`effectinfo.py`, `7 >= 6`), so a callee
    /// registered with it must not be dispatched through `cond_call` /
    /// `record_known_result` (`jtransform.py:1677`,
    /// `pyjitpl.py do_conditional_call` assert the opposite).
    Unanalyzed,
    /// `EF_CANNOT_RAISE` — `call.py` `else` branch.
    CannotRaise,
    /// `EF_CANNOT_RAISE` + analyzer-confirmed empty heap. Maps to
    /// `CANNOT_RAISE_NO_HEAP_EFFECT_INFO` (`effectinfo.py:281-283`).
    CannotRaiseNoHeap,
    /// `EF_ELIDABLE_CAN_RAISE` — `call.py` `elif cr:` branch.
    ElidableCanRaise,
    /// `EF_ELIDABLE_CANNOT_RAISE` — `call.py:299` `else` branch under
    /// `elif elidable:`.
    ElidableCannotRaise,
    /// `EF_ELIDABLE_OR_MEMORYERROR` — `call.py` `if cr == "mem":`.
    ElidableOrMemerror,
    /// `EF_LOOPINVARIANT` — `call.py` `elif loopinvariant:`.
    LoopInvariant,
}

/// Resolve a [`EffectInfoSlot`] to its matching [`EffectInfo`] const.
///
/// `call.py:320 effectinfo_from_writeanalyze` constructs the final EI
/// from the `extraeffect` plus the analyzer outputs; pyre's per-slot
/// const captures the analyzer-absent fallback for that `extraeffect`.
pub fn effect_info_for_slot(slot: EffectInfoSlot) -> EffectInfo {
    match slot {
        EffectInfoSlot::CanRaise => can_raise_effect_info(),
        EffectInfoSlot::Unanalyzed => default_effect_info(),
        EffectInfoSlot::CannotRaise => cannot_raise_effect_info(),
        EffectInfoSlot::CannotRaiseNoHeap => CANNOT_RAISE_NO_HEAP_EFFECT_INFO.clone(),
        EffectInfoSlot::ElidableCanRaise => ELIDABLE_EFFECT_INFO,
        EffectInfoSlot::ElidableCannotRaise => ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        EffectInfoSlot::ElidableOrMemerror => ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        EffectInfoSlot::LoopInvariant => LOOPINVARIANT_EFFECT_INFO,
    }
}

/// Pick the upstream-equivalent default effect for an opcode whose
/// callee has not been write-analyzed.
///
/// `pyjitpl.py do_residual_or_indirect_call` selects between
/// CALL / CALL_PURE / CALL_LOOPINVARIANT / CALL_MAY_FORCE based on
/// `descr.get_extra_info().extraeffect`. Pyre baked the choice into the
/// opcode at codewriter time, so reverse the mapping here so the descr
/// the optimizer reads carries the matching effect class.
///
/// `CALL_MAY_FORCE` maps to [`default_effect_info()`]: the opcode alone
/// only proves `check_forces_virtual_or_virtualizable()` held
/// (`pyjitpl.py`), which `EF_RANDOM_EFFECTS` satisfies via the
/// `>=` at `effectinfo.py`. It does NOT prove the write analyzer
/// ran, so the reconstruction cannot invent an empty write set.
/// `CALL_RELEASE_GIL` cannot be reconstructed from the opcode alone —
/// upstream `effectinfo.py MOST_GENERAL` pairs `EF_RANDOM_EFFECTS`
/// with a `call_release_gil_target` funcptr that this helper does not
/// see, so the analyzer-absent default is fail-loud: any production
/// path that needs a release-GIL EI must build it explicitly via
/// [`make_call_descr_with_effect`] with the resolved target.
pub fn default_effect_for_opcode(opcode: majit_ir::OpCode) -> EffectInfo {
    if opcode.is_call_pure() {
        ELIDABLE_EFFECT_INFO
    } else if opcode.is_call_loopinvariant() {
        LOOPINVARIANT_EFFECT_INFO
    } else if opcode.is_call_may_force() {
        default_effect_info()
    } else if opcode.is_call_release_gil() {
        unreachable!(
            "default_effect_for_opcode: CALL_RELEASE_GIL (`{opcode:?}`) requires \
             call_release_gil_target funcptr; build the EffectInfo explicitly via \
             make_call_descr_with_effect (effectinfo.py:271-273 MOST_GENERAL)"
        );
    } else {
        default_effect_info()
    }
}

/// Create a CallDescr with the conservative
/// [`default_effect_info()`] (`EffectInfo::MOST_GENERAL`), the
/// analyzer-absent fallback described there.
///
/// Production producers should prefer one of the more specific factories
/// so the per-callee classification reaches the trace IR:
///
/// * [`make_call_descr_from_target_slot`] when a resolved
///   [`crate::jitcode::JitCallTarget`] is available — threads the
///   macro-time [`EffectInfoSlot`] (`call.py getcalldescr` parity).
/// * `make_call_descr_for_opcode` when only the call opcode family is
///   known (`pyjitpl.py do_residual_or_indirect_call`'s
///   `EF_LOOPINVARIANT` / `EF_ELIDABLE_*` reverse-mapping).
/// * [`make_call_descr_with_effect`] when an explicit `EffectInfo` has
///   been hand-built (release-gil targets, oopspec specializations).
///
/// Remaining direct callers of this fallback are restricted to
/// `#[cfg(test)]` fixtures (pyjitpl/optimizeopt/backend test stubs)
/// where the conservative descr is the test's intent — matching the
/// "no analyzer ran" path the production fallbacks above subsume.
pub fn make_call_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, default_effect_info())
}

/// Create a CallDescr whose effect info matches the call opcode family.
pub fn make_call_descr_for_opcode(
    opcode: majit_ir::OpCode,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, default_effect_for_opcode(opcode))
}

/// Create a CallDescr from a per-target [`EffectInfoSlot`] classification.
///
/// `call.py getcalldescr` selects `extraeffect` per callsite
/// from the analyzer chain; pyre's analyzer-absent equivalent is the
/// `JitCallTarget.effect_info_slot` macro-time classification.  This
/// factory is the per-target entry point — callers that have a
/// resolved [`crate::jitcode::JitCallTarget`] thread its slot through.
pub fn make_call_descr_from_target_slot(
    arg_types: &[Type],
    result_type: Type,
    slot: EffectInfoSlot,
) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, effect_info_for_slot(slot))
}

fn result_metadata(result_type: Type) -> (bool, usize) {
    let result_size = match result_type {
        Type::Int | Type::Ref | Type::Float => 8,
        Type::Void => 0,
    };
    (result_type == Type::Int, result_size)
}

/// call.py:320 `effectinfo_from_writeanalyze` parity. Create a
/// CallDescr with explicit per-call-site EffectInfo.
pub fn make_call_descr_with_effect(
    arg_types: &[Type],
    result_type: Type,
    effect_info: EffectInfo,
) -> DescrRef {
    let (result_signed, result_size) = result_metadata(result_type);
    make_call_descr_sized(
        &majit_ir::descr::arg_classes_of(arg_types),
        arg_types,
        result_type,
        majit_ir::descr::result_class_of(result_type),
        result_signed,
        result_size,
        effect_info,
    )
}

/// [`make_call_descr_with_effect`] variant for hand-written `extern "C"`
/// helpers recorded as void residuals whose C signature actually RETURNS a
/// dummy machine word (`fn(i64×n) -> i64`, value ignored). The descr keeps
/// `result_type = Void` (the recorded op is `CallN` with no result box) but
/// carries `result_size = 8` where a plain void descr carries 0, so a
/// backend that emits signature-exact direct calls (wasm `call_indirect`)
/// can select the `(i64×n) -> i64` type and drop the result. Reflective
/// dispatch paths never read `result_size` for a void result, and the field
/// participates in the interning key, so word-ABI descrs never collapse
/// with plain void descrs of the same shape.
pub fn make_call_descr_void_word_abi(arg_types: &[Type], effect_info: EffectInfo) -> DescrRef {
    make_call_descr_sized(
        &majit_ir::descr::arg_classes_of(arg_types),
        arg_types,
        Type::Void,
        'v',
        false,
        8,
        effect_info,
    )
}

/// Sized variant of [`make_call_descr_with_effect`] for deserialized
/// `descrs.bin` call descriptors whose result size/sign were fixed by
/// `descr.py:650-665 getCallDescrClass`.
pub fn make_call_descr_sized_with_effect(
    arg_types: &[Type],
    result_type: Type,
    result_signed: bool,
    result_size: usize,
    effect_info: EffectInfo,
) -> DescrRef {
    make_call_descr_sized(
        &majit_ir::descr::arg_classes_of(arg_types),
        arg_types,
        result_type,
        majit_ir::descr::result_class_of(result_type),
        result_signed,
        result_size,
        effect_info,
    )
}

/// Deserialized translated-image variant. `translated_effect_info_id` is the
/// identity of RPython's canonical EffectInfo object, carried explicitly
/// because pyre's build script and runtime do not share object addresses.
pub fn make_call_descr_sized_with_translated_effect(
    arg_classes: &str,
    arg_types: &[Type],
    result_type: Type,
    result_class: char,
    result_signed: bool,
    result_size: usize,
    translated_effect_info_id: u32,
) -> DescrRef {
    let effect_info = majit_ir::effectinfo::translated_effect_info(translated_effect_info_id)
        .unwrap_or_else(|| {
            panic!(
                "translated EffectInfo {translated_effect_info_id} was not published before its CallDescr"
            )
        });
    make_call_descr_sized_with_cell(
        arg_classes,
        arg_types,
        result_type,
        result_class,
        result_signed,
        result_size,
        effect_info,
    )
}

fn make_call_descr_sized(
    arg_classes: &str,
    arg_types: &[Type],
    result_type: Type,
    result_class: char,
    result_signed: bool,
    result_size: usize,
    effect_info: EffectInfo,
) -> DescrRef {
    // `effectinfo.py` invariant: no new `EffectInfo` may be
    // constructed after `compute_bitstrings` has run; PyPy enforces this
    // implicitly through codewriter lifecycle ordering, with `Ellipsis`
    // as a post-hoc bitcheck-time tripwire.  Pyre allows trace-time
    // mints, so the gate is at the construction site.  Trivial-raw
    // EIs (`raw=None` ⇒ random-effects, or `raw=Some(empty)` ⇒
    // concrete-empty) keep the invariant intact because
    // `compute_bitstrings` would map them to a `None`/empty bitstring
    // independent of any other EI's (eisetr, eisetw) class — adding
    // one post-setup never reshuffles existing class assignments.
    // Non-trivial raw sets (`Some(non-empty)`) would shift the
    // partition and silently invalidate every cached bitstring.
    if majit_ir::effectinfo::compute_bitstrings_has_run() && effect_info.has_non_trivial_raw_set() {
        panic!(
            "make_call_descr_with_effect: EffectInfo with non-trivial raw \
             descr set constructed after compute_bitstrings ran.  PyPy \
             effectinfo.py:182-184 forbids the same shape via the \
             Ellipsis sentinel + bitcheck panic.  Fix: ensure all call \
             descrs whose analyzer outputs concrete frozensets are \
             minted before `MetaInterpStaticData::finish_setup_descrs` \
             runs (codewriter setup phase).\n  effect_info: {effect_info:?}"
        );
    }
    // effectinfo.py:147-148 interns ordinary EffectInfos before
    // descr.py:665 puts that object in `_cache_call`'s tuple. Release-gil
    // targets bypass the interner with a fresh cell, matching the `object()`
    // cache breaker at effectinfo.py:144-146.
    let effect_info = majit_ir::effectinfo::intern_effect_info(effect_info);
    make_call_descr_sized_with_cell(
        arg_classes,
        arg_types,
        result_type,
        result_class,
        result_signed,
        result_size,
        effect_info,
    )
}

/// Resolve the serialized half of an Assembler.descrs call entry once.
/// pyjitpl.py do_residual_call records that same descriptor on every use.
pub(crate) fn call_descr_from_bh(bh: &majit_jitcode::jitcode::BhCallDescr) -> DescrRef {
    let type_of = |class| match class {
        'i' | 'S' => Type::Int,
        'r' => Type::Ref,
        // descr.py CallDescr.get_normalized_result_type / verify_types:
        // singlefloat bits use the integer bank; longlong uses the float bank.
        'f' | 'L' => Type::Float,
        'v' => Type::Void,
        _ => panic!("invalid call descriptor class {class:?}"),
    };
    let arg_types: Vec<_> = bh.arg_classes.chars().map(type_of).collect();
    let result_type = type_of(bh.result_type);
    let result_size = if bh.void_word_abi { 8 } else { bh.result_size };
    if let Some(id) = bh.translated_effect_info_id {
        make_call_descr_sized_with_translated_effect(
            &bh.arg_classes,
            &arg_types,
            result_type,
            bh.result_type,
            bh.result_signed,
            result_size,
            id,
        )
    } else {
        make_call_descr_sized(
            &bh.arg_classes,
            &arg_types,
            result_type,
            bh.result_type,
            bh.result_signed,
            result_size,
            bh.extra_info.clone(),
        )
    }
}

fn make_call_descr_sized_with_cell(
    arg_classes: &str,
    arg_types: &[Type],
    result_type: Type,
    result_class: char,
    result_signed: bool,
    result_size: usize,
    effect_info: Arc<majit_ir::effectinfo::EffectInfoCell>,
) -> DescrRef {
    let key = majit_ir::descr::LLType::func_key(
        arg_classes,
        result_type,
        result_class,
        result_signed,
        result_size,
        &effect_info,
    );
    let mut gc = majit_ir::descr::gc_cache().lock();
    gc.intern_call_descr_with(key, || {
        let descr: DescrRef = Arc::new(MetaCallDescr {
            heapcache_index: majit_ir::descr::next_call_descr_heapcache_index(),
            arg_types: arg_types.to_vec(),
            arg_classes: arg_classes.to_string(),
            result_type,
            result_class,
            result_signed,
            result_size,
            effect_info,
        });
        descr
    })
}

/// Create a CallDescr for CALL_MAY_FORCE_* operations.
///
/// The trait-dispatch leg records a residual through this descr; the walker
/// leg records the equivalent residual through a calldescr the codewriter
/// builds with `forces_virtual_or_virtualizable_effect_info()`
/// (`CallFlavor::MayForce`). Both must carry the same
/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` `EffectInfo` so the optimizer and
/// `do_residual_call` treat the two legs' may-force ops identically
/// (`MetaCallMayForceDescr::get_extra_info` mirrors that constructor).
pub fn make_call_may_force_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    #[derive(Debug)]
    struct MetaCallMayForceDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl majit_ir::Descr for MetaCallMayForceDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for MetaCallMayForceDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            0
        }
        fn get_extra_info(&self) -> &EffectInfo {
            // Byte-identical to `forces_virtual_or_virtualizable_effect_info()`
            // (the `CallFlavor::MayForce` row the codewriter stamps on the
            // walker-leg calldescr): `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
            // with analyzer-empty read/write bitsets. The two legs must
            // agree on the descr shape for the same residual.
            //
            // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` is the only extraeffect
            // consistent with the `CALL_MAY_FORCE_*` opcode this descr
            // accompanies: `check_forces_virtual_or_virtualizable()` reads
            // `extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, so the
            // earlier `EF_CAN_RAISE` (5 < 6) failed that test while still
            // riding a may-force op. `> EF_CANNOT_RAISE` keeps
            // `check_can_raise()` true so the may-force sequence still
            // records its trailing `GUARD_NO_EXCEPTION`.
            //
            // The empty bitsets are faithful, not a shortcut: the
            // analyzer-absent fallback (`effectinfo_from_writeanalyze` with
            // `bottom_result()`) produces empty `read/write_descrs_*`, so
            // `force_from_effectinfo` finds no descr bits set and leaves
            // cached heap state live across the call. Promoting instead to
            // `EF_RANDOM_EFFECTS` would trip `has_random_effects()` and
            // route OptHeap through `clean_caches`, over-invalidating heap
            // PyPy keeps live for analyzer-empty virtualizable-forcing
            // callees.
            static INFO: EffectInfo = EffectInfo::const_new(
                ExtraEffect::ForcesVirtualOrVirtualizable,
                OopSpecIndex::None,
            );
            &INFO
        }
    }

    Arc::new(MetaCallMayForceDescr {
        arg_types: arg_types.to_vec(),
        result_type,
    })
}

/// `pyjitpl.py compile_trace` records `descr=ptoken` on the tentative JUMP.
/// `unroll.py optimize_bridge` then does `cell_token = jump_op.getdescr()`
/// and `assert isinstance(cell_token, JitCellToken)`.
#[derive(Debug)]
struct JitCellTokenDescr {
    token: Arc<JitCellToken>,
}

impl majit_ir::Descr for JitCellTokenDescr {
    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::descr::LoopTokenDescr> {
        Some(self)
    }
}

impl majit_ir::descr::LoopTokenDescr for JitCellTokenDescr {
    fn loop_token_number(&self) -> u64 {
        self.token.number
    }

    fn call_virtualizable_index(&self) -> Option<usize> {
        self.token.virtualizable_arg_index()
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.token)
    }
}

/// Wrap a live `JitCellToken` as the JUMP descr `compile_trace` records.
pub fn jit_cell_token_as_descr(token: Arc<JitCellToken>) -> DescrRef {
    Arc::new(JitCellTokenDescr { token })
}

/// `compile.py isinstance(descr, JitCellToken)` parity factory.
///
/// Create a `CALL_ASSEMBLER_*` descr that owns the same `Arc<JitCellToken>`
/// as the production warm cell / `CompiledEntry::token` / `alive_loops`.
/// `direct_assembler_call` (`pyjitpl.py`) is the canonical caller —
/// it threads the cell's compiled token through, so `record_loop_or_bridge`'s
/// keepalive walker downcasts the descr and pushes that same Arc into
/// `original.keepalive_tokens`, matching `compile.py record_jump_to(descr)`.
pub fn make_call_assembler_descr(
    target_token: Arc<JitCellToken>,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
    })
}

/// Test-only number factory for fixtures that do not have an
/// `Arc<JitCellToken>`.
///
/// Synthesises a fresh stand-alone `Arc<JitCellToken>` with the requested
/// `target_number` so the descr keeps the same shape as the identity-preserving
/// path. Identity is not preserved, so production callers must use
/// `make_call_assembler_descr`.
#[cfg(test)]
pub fn make_call_assembler_descr_by_number(
    target_number: u64,
    arg_types: &[Type],
    result_type: Type,
    virtualizable_arg_index: Option<usize>,
) -> DescrRef {
    let tok = JitCellToken::new(target_number);
    tok.virtualizable_arg_index.set(virtualizable_arg_index);
    make_call_assembler_descr(Arc::new(tok), arg_types, result_type)
}

#[cfg(test)]
mod set_effect_bitstrings_tests {
    use super::*;
    use majit_ir::EffectInfo;

    /// `Descr::set_effect_bitstrings` writes through to the cached
    /// `MetaCallDescr.effect_info`, visible to subsequent
    /// `cd.get_extra_info()` reads.  Mirrors `effectinfo.py:537-538
    /// setattr(ei, 'bitstring_*', …)`.
    #[test]
    fn set_effect_bitstrings_publishes_to_get_extra_info() {
        use majit_ir::descr::SimpleFieldDescr;
        let f3: DescrRef = Arc::new(SimpleFieldDescr::new(3, 0, 8, Type::Int, false));
        let f7: DescrRef = Arc::new(SimpleFieldDescr::new(7, 0, 8, Type::Int, false));
        let ei = EffectInfo {
            _readonly_descrs_fields: Some(vec![f3, f7]),
            _write_descrs_fields: Some(vec![]),
            _readonly_descrs_arrays: Some(vec![]),
            _write_descrs_arrays: Some(vec![]),
            _readonly_descrs_interiorfields: Some(vec![]),
            _write_descrs_interiorfields: Some(vec![]),
            ..EffectInfo::default()
        };
        let descr = make_call_descr_with_effect(&[Type::Int], Type::Int, ei);
        let cd = descr.as_call_descr().unwrap();
        // Pre-set: bitstring_readonly_descrs_fields was seeded at
        // construction by make_call_descr_with_effect via Default.
        // After set_effect_bitstrings the new value wins.
        descr.set_effect_bitstrings(
            Some(vec![0x88]),
            Some(vec![0x00]),
            Some(vec![0x00]),
            Some(vec![0x00]),
            Some(vec![0x00]),
            Some(vec![0x00]),
        );
        let ei_after = cd.get_extra_info();
        assert_eq!(
            ei_after.readonly_descrs_fields.as_deref(),
            Some(&[0x88u8][..])
        );
        assert_eq!(ei_after.write_descrs_fields.as_deref(), Some(&[0x00u8][..]));
    }

    #[test]
    fn jit_cell_token_as_descr_exposes_the_same_arc() {
        let token = Arc::new(JitCellToken::new(42));
        let descr = jit_cell_token_as_descr(Arc::clone(&token));
        let recovered = descr
            .as_loop_token_descr()
            .and_then(|ltd| ltd.token_handle_any())
            .and_then(|any| any.downcast_ref::<Arc<JitCellToken>>())
            .expect("JUMP descr must be the JitCellToken");
        assert!(
            Arc::ptr_eq(recovered, &token),
            "compile_trace records descr=ptoken, the same Arc the cell holds"
        );
        assert_eq!(
            descr
                .as_loop_token_descr()
                .map(|ltd| ltd.loop_token_number()),
            Some(42)
        );
    }

    /// Default `Descr::set_effect_bitstrings` is a no-op for descrs
    /// without an `EffectInfo` (e.g. field/array/size/fail descrs).
    #[test]
    fn default_set_effect_bitstrings_is_noop_for_non_call_descrs() {
        // `SimpleFieldDescr` does not override `set_effect_bitstrings`,
        // so calling it should not panic and should not affect any
        // other descr state.
        let descr: DescrRef = Arc::new(majit_ir::descr::SimpleFieldDescr::new(
            42,
            0,
            8,
            Type::Int,
            false,
        ));
        descr.set_effect_bitstrings(
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
        );
        // Field descr's own getters still work normally.
        assert_eq!(descr.index(), 42);
    }

    /// End-to-end integration: after `compute_bitstrings` runs over a
    /// constructed all_descrs vector, calling
    /// `Descr::set_effect_bitstrings` for each call descr publishes
    /// bitstrings keyed by `descr.get_ei_index()`. Mirrors
    /// `effectinfo.py` write-back loop.
    #[test]
    fn compute_bitstrings_then_set_publishes_eiindex_keyed_bitstrings() {
        use majit_ir::descr::SimpleFieldDescr;
        // Two field descrs.
        let f1: DescrRef = Arc::new(SimpleFieldDescr::new(91_000_001, 0, 8, Type::Int, false));
        let f2: DescrRef = Arc::new(SimpleFieldDescr::new(91_000_002, 0, 8, Type::Int, false));

        // Two EIs that BOTH read `f1` (Arc identity).
        let ei_a = EffectInfo {
            _readonly_descrs_fields: Some(vec![f1.clone()]),
            _write_descrs_fields: Some(vec![]),
            _readonly_descrs_arrays: Some(vec![]),
            _write_descrs_arrays: Some(vec![]),
            _readonly_descrs_interiorfields: Some(vec![]),
            _write_descrs_interiorfields: Some(vec![]),
            ..EffectInfo::default()
        };

        let ei_b = EffectInfo {
            _readonly_descrs_fields: Some(vec![f1.clone()]),
            _write_descrs_fields: Some(vec![]),
            _readonly_descrs_arrays: Some(vec![]),
            _write_descrs_arrays: Some(vec![]),
            _readonly_descrs_interiorfields: Some(vec![]),
            _write_descrs_interiorfields: Some(vec![]),
            ..EffectInfo::default()
        };

        let cd_a = make_call_descr_with_effect(&[Type::Int], Type::Int, ei_a.clone());
        let cd_b = make_call_descr_with_effect(&[Type::Float], Type::Float, ei_b.clone());

        // Run compute_bitstrings the way `MetaInterpStaticData::finish_setup_descrs`
        // does: clone EIs, mutate clones, write back via the trait.
        let all_descrs: Vec<DescrRef> = vec![f1.clone(), f2.clone(), cd_a.clone(), cd_b.clone()];
        let mut owned_eis: Vec<EffectInfo> = vec![ei_a.clone(), ei_b.clone()];
        {
            let mut ei_refs: Vec<&mut EffectInfo> = owned_eis.iter_mut().collect();
            majit_ir::effectinfo::compute_bitstrings(&all_descrs, &mut ei_refs);
        }
        cd_a.set_effect_bitstrings(
            owned_eis[0].readonly_descrs_fields.clone(),
            owned_eis[0].write_descrs_fields.clone(),
            owned_eis[0].readonly_descrs_arrays.clone(),
            owned_eis[0].write_descrs_arrays.clone(),
            owned_eis[0].readonly_descrs_interiorfields.clone(),
            owned_eis[0].write_descrs_interiorfields.clone(),
        );
        cd_b.set_effect_bitstrings(
            owned_eis[1].readonly_descrs_fields.clone(),
            owned_eis[1].write_descrs_fields.clone(),
            owned_eis[1].readonly_descrs_arrays.clone(),
            owned_eis[1].write_descrs_arrays.clone(),
            owned_eis[1].readonly_descrs_interiorfields.clone(),
            owned_eis[1].write_descrs_interiorfields.clone(),
        );

        // f1 was assigned ei_index = 0 (first descr in the only class).
        // f2 was not in any EI so ei_index stays at u32::MAX.
        assert_eq!(f1.get_ei_index(), 0);
        assert_eq!(f2.get_ei_index(), u32::MAX);

        // Both EIs' bitstrings encode a bit at f1's ei_index = 0.
        let bs_a = cd_a
            .as_call_descr()
            .unwrap()
            .get_extra_info()
            .readonly_descrs_fields
            .clone()
            .expect("ei_a readonly bitstring");
        assert!(majit_ir::bitstring::bitcheck(&bs_a, f1.get_ei_index()));
        let bs_b = cd_b
            .as_call_descr()
            .unwrap()
            .get_extra_info()
            .readonly_descrs_fields
            .clone()
            .expect("ei_b readonly bitstring");
        assert!(majit_ir::bitstring::bitcheck(&bs_b, f1.get_ei_index()));
    }

    /// `GcCache._cache_call` returns entries minted by
    /// `make_call_descr_with_effect`.  Used by
    /// `MetaInterpStaticData::finish_setup_descrs` to walk the full
    /// EI population for `compute_bitstrings`.
    #[test]
    fn gc_cache_call_snapshot_returns_recent_entries() {
        use majit_ir::descr::SimpleFieldDescr;
        let f1: DescrRef = Arc::new(SimpleFieldDescr::new(1, 0, 8, Type::Int, false));
        let ei = EffectInfo {
            _readonly_descrs_fields: Some(vec![f1]),
            _write_descrs_fields: Some(vec![]),
            _readonly_descrs_arrays: Some(vec![]),
            _write_descrs_arrays: Some(vec![]),
            _readonly_descrs_interiorfields: Some(vec![]),
            _write_descrs_interiorfields: Some(vec![]),
            ..EffectInfo::default()
        };
        let descr = make_call_descr_with_effect(&[Type::Int, Type::Ref], Type::Float, ei);

        let cached = majit_ir::descr::gc_cache().lock().snapshot_calls();
        // The descr we just constructed is in the cache. We also
        // tolerate the cache holding entries from earlier tests in the
        // same process; we only assert membership of OUR descr.
        let my_idx = descr.index();
        let found = cached.iter().any(|d| d.index() == my_idx);
        assert!(
            found,
            "GcCache._cache_call snapshot must include the descr we just made"
        );
    }

    #[test]
    fn call_cache_keys_on_canonical_effect_info_identity() {
        let first = make_call_descr_with_effect(
            &[Type::Int, Type::Int, Type::Ref],
            Type::Float,
            EffectInfo::default(),
        );
        let second = make_call_descr_with_effect(
            &[Type::Int, Type::Int, Type::Ref],
            Type::Float,
            EffectInfo::default(),
        );
        assert!(Arc::ptr_eq(&first, &second));
        assert!(std::ptr::eq(
            first.as_call_descr().unwrap().get_extra_info(),
            second.as_call_descr().unwrap().get_extra_info(),
        ));
    }

    #[test]
    fn release_gil_call_cache_keeps_fresh_effect_info_identity() {
        let mut effect_info = EffectInfo::default();
        effect_info.call_release_gil_target = (0xfeed, 0);
        let first = make_call_descr_with_effect(
            &[Type::Int, Type::Ref, Type::Int],
            Type::Void,
            effect_info.clone(),
        );
        let second = make_call_descr_with_effect(
            &[Type::Int, Type::Ref, Type::Int],
            Type::Void,
            effect_info,
        );
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(!std::ptr::eq(
            first.as_call_descr().unwrap().get_extra_info(),
            second.as_call_descr().unwrap().get_extra_info(),
        ));
    }

    /// The trait-dispatch leg's `make_call_may_force_descr` and the walker
    /// leg's `forces_virtual_or_virtualizable_effect_info()` must classify a
    /// may-force residual identically: `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    /// (consistent with the `CALL_MAY_FORCE_*` opcode), can-raise (trailing
    /// `GUARD_NO_EXCEPTION` retained), no random effects (no `clean_caches`
    /// over-invalidation), and no oopspec.
    #[test]
    fn may_force_descr_matches_forces_virtual_effect_info() {
        let canonical = forces_virtual_or_virtualizable_effect_info();
        let descr = make_call_may_force_descr(&[Type::Ref], Type::Ref);
        let ei = descr.as_call_descr().unwrap().get_extra_info();

        assert_eq!(ei.extraeffect, canonical.extraeffect);
        assert_eq!(ei.extraeffect, ExtraEffect::ForcesVirtualOrVirtualizable);
        assert!(ei.check_forces_virtual_or_virtualizable());
        assert!(ei.check_can_raise(false));
        assert!(!ei.has_random_effects());
        assert!(!ei.has_oopspec());
        assert!(!ei.check_can_invalidate());
    }

    /// `effectinfo.py:149-162` ties `EF_RANDOM_EFFECTS` to raw sets =
    /// `None`, so the hand-classified `call.py:300-301` row and the
    /// `graphanalyze.py:109-112` no-graph row are not interchangeable.
    /// A slot resolving `CanRaise` to `MOST_GENERAL` would make every
    /// `cond_call` / `record_known_result` target force the
    /// virtualizable, which `jtransform.py:1677` and
    /// `pyjitpl.py do_conditional_call` assert can never happen.
    #[test]
    fn can_raise_slot_is_the_analyzed_row_and_unanalyzed_is_most_general() {
        let can_raise = effect_info_for_slot(EffectInfoSlot::CanRaise);
        assert_eq!(can_raise.extraeffect, ExtraEffect::CanRaise);
        assert!(!can_raise.has_random_effects());
        assert!(!can_raise.check_forces_virtual_or_virtualizable());
        assert!(can_raise._write_descrs_fields.is_some());

        let unanalyzed = effect_info_for_slot(EffectInfoSlot::Unanalyzed);
        assert_eq!(unanalyzed.extraeffect, ExtraEffect::RandomEffects);
        assert!(unanalyzed.has_random_effects());
        assert!(unanalyzed._write_descrs_fields.is_none());
        assert_eq!(unanalyzed, default_effect_info());
        assert_ne!(can_raise, unanalyzed);
    }
}

#[cfg(test)]
mod translated_result_class_tests {
    use super::*;
    use majit_ir::EffectInfo;

    /// `descr.py get_result_type()` returns the raw result char, and
    /// `descr.py` keys the call-descr cache on it.  A descr rehydrated
    /// from the translated image goes through
    /// `make_call_descr_sized_with_translated_effect`, which normalises `'S'`
    /// to `Type::Int` for the IR; the raw char has to survive alongside it,
    /// because the resume path rebuilds a `BhCallDescr` off this descr
    /// (`pyre-jit-trace`'s `state.rs`) and hands it to `bh_call_i`.
    #[test]
    fn a_translated_descr_keeps_its_raw_result_class() {
        // Two ids so this test cannot be perturbed by a sibling publishing
        // over the same dense slot.
        let cell_id = 4242u32;
        majit_ir::effectinfo::intern_translated_effect_info(cell_id, EffectInfo::default());

        let singlefloat = make_call_descr_sized_with_translated_effect(
            "i",
            &[Type::Int],
            Type::Int,
            'S',
            false,
            4,
            cell_id,
        );
        assert_eq!(singlefloat.as_call_descr().unwrap().result_class(), 'S');
        // The normalised view is unchanged — `'S'` is an int-shaped result.
        assert_eq!(
            singlefloat.as_call_descr().unwrap().result_type(),
            Type::Int
        );

        // Everything the pre-fix key compared is identical here — arg types,
        // normalised result type, signedness, size, and the EffectInfo cell —
        // so if the raw char were absent from the key this second mint would
        // return the descr above and report `'S'`.
        let plain_int = make_call_descr_sized_with_translated_effect(
            "i",
            &[Type::Int],
            Type::Int,
            'i',
            false,
            4,
            cell_id,
        );
        assert_eq!(plain_int.as_call_descr().unwrap().result_class(), 'i');
        assert!(!std::sync::Arc::ptr_eq(&singlefloat, &plain_int));

        // Interning still holds for a repeat of the same raw char.
        let again = make_call_descr_sized_with_translated_effect(
            "i",
            &[Type::Int],
            Type::Int,
            'S',
            false,
            4,
            cell_id,
        );
        assert_eq!(singlefloat.as_call_descr().unwrap().arg_classes(), "i");
        let single_arg = make_call_descr_sized_with_translated_effect(
            "S",
            &[Type::Int],
            Type::Int,
            'i',
            false,
            4,
            cell_id,
        );
        assert_eq!(single_arg.as_call_descr().unwrap().arg_classes(), "S");
        assert!(!std::sync::Arc::ptr_eq(&plain_int, &single_arg));
        assert!(std::sync::Arc::ptr_eq(&singlefloat, &again));
    }
}
