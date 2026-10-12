//! Synthetic CPU shim for pyre's runtime per-CodeObject jitcodes.
//!
//! RPython's `cpu.bh_call_*` (see
//! `rpython/jit/backend/model.py bh_call_i` and `rpython/jit/metainterp/blackhole.py bhimpl_residual_call_r_i`
//! for both `bhimpl_residual_call_*` and `bhimpl_inline_call_*`) receives a real
//! function address (`adr2int(jitcode.fnaddr)`) and dispatches to native code.
//!
//! pyre's runtime per-CodeObject sub-jitcodes have **no** real function
//! addresses — they are produced by the runtime codewriter for each user
//! `CodeObject` and never get translated to native function pointers.  To let
//! `bhimpl_residual_call_*`/`bhimpl_inline_call_*` reach those sub-jitcodes
//! through the same `cpu.bh_call_*` interface, this module reserves a
//! synthetic fnaddr space (the high bit of `i64`) whose payload is the
//! callee jitcode's index.  A `Backend` shim decodes the synthetic fnaddr
//! back to the jitcode index and runs a nested `BlackholeInterpreter`.
//!
//! Real function addresses on x86_64 / aarch64 user space occupy the lower 48
//! bits with the upper bits zero-extended, so they are always non-negative as
//! `i64`.  Setting bit 63 yields a negative `i64` value that cannot collide
//! with any real fnaddr.

/// Marker bit reserved on `BhDescr::JitCode.fnaddr` to flag pyre runtime
/// sub-jitcodes that route through [`SyntheticCpu`] instead of native code.
///
/// `SYNTHETIC_FNADDR_BASE | jitcode_index` produces a negative `i64` that the
/// shim recognises and decodes; real function pointers are non-negative
/// 48-bit user-space addresses and never set bit 63.
pub const SYNTHETIC_FNADDR_BASE: i64 = i64::MIN;

/// Encode a pyre runtime jitcode index as a synthetic fnaddr suitable for
/// storing in `BhDescr::JitCode.fnaddr`.
#[inline]
#[allow(dead_code)]
pub fn encode_synthetic(jitcode_index: usize) -> i64 {
    debug_assert!(
        (jitcode_index as u64) <= (i64::MAX as u64),
        "synthetic jitcode index out of range",
    );
    SYNTHETIC_FNADDR_BASE | (jitcode_index as i64)
}

/// Return whether `fnaddr` was produced by [`encode_synthetic`].
#[inline]
#[allow(dead_code)]
pub fn is_synthetic_fnaddr(fnaddr: i64) -> bool {
    fnaddr < 0
}

/// Decode a synthetic fnaddr back to its jitcode index.  Returns `None` for
/// real (non-negative) function addresses.
#[inline]
#[allow(dead_code)]
pub fn decode_synthetic(fnaddr: i64) -> Option<usize> {
    if is_synthetic_fnaddr(fnaddr) {
        Some((fnaddr & i64::MAX) as usize)
    } else {
        None
    }
}

/// Backend shim that will eventually dispatch synthetic fnaddrs to pyre
/// runtime sub-jitcodes via a nested `BlackholeInterpreter`.
///
/// Installs the [`crate::Backend`] trait skeleton so the type satisfies the
/// trait and can be plugged into `BlackholeInterpBuilder.cpu`.  All required
/// compilation/execution methods `unreachable!` because SyntheticCpu only
/// services `bh_call_*` dispatch, never native compilation or trace
/// execution.  The four `bh_call_*` overrides currently `unimplemented!` —
/// the actual decode-and-dispatch path lives in `majit-metainterp` (which
/// depends on `majit-backend`, not vice versa), so the cross-crate wiring
/// is deferred.
#[derive(Default)]
#[allow(dead_code)]
pub struct SyntheticCpu {
    /// `model.py self.tracker = CPUTotalTracker()` — synthetic
    /// backends own a private tracker so cross-test/cross-instance
    /// total counts stay isolated rather than aliasing through the
    /// process-wide fallback.
    cpu_tracker: std::sync::Arc<crate::CpuTotalTracker>,
}

impl SyntheticCpu {
    #[allow(dead_code)]
    pub fn new() -> Self {
        SyntheticCpu::default()
    }
}

impl crate::Backend for SyntheticCpu {
    fn backend_name(&self) -> &'static str {
        "synthetic"
    }

    fn cpu_tracker(&self) -> &std::sync::Arc<crate::CpuTotalTracker> {
        &self.cpu_tracker
    }

    /// `rpython/jit/backend/model.py` declares `compile_loop` on every
    /// `AbstractCPU`.  SyntheticCpu does not produce native code; this method
    /// must never be reached.
    fn compile_loop(
        &mut self,
        _inputargs: &[majit_ir::InputArgRc],
        _ops: &[majit_ir::OpRc],
        _token: &crate::JitCellToken,
    ) -> Result<crate::AsmInfo, crate::BackendError> {
        unreachable!("SyntheticCpu does not compile native code; only services bh_call_* dispatch")
    }

    fn compile_bridge(
        &mut self,
        _fail_descr: &dyn majit_ir::FailDescr,
        _inputargs: &[majit_ir::InputArgRc],
        _ops: &[majit_ir::OpRc],
        _original_token: &crate::JitCellToken,
        _caller_recovery_layout: Option<&crate::ExitRecoveryLayout>,
    ) -> Result<crate::AsmInfo, crate::BackendError> {
        unreachable!("SyntheticCpu does not compile bridges; only services bh_call_* dispatch")
    }

    fn execute_token(
        &self,
        _token: &crate::JitCellToken,
        _args: &[majit_ir::Value],
    ) -> crate::DeadFrame {
        unreachable!(
            "SyntheticCpu does not execute native traces; only services bh_call_* dispatch"
        )
    }

    fn get_latest_descr<'a>(&'a self, _frame: &'a crate::DeadFrame) -> &'a dyn majit_ir::FailDescr {
        unreachable!("SyntheticCpu does not produce DeadFrames; get_latest_descr unreachable")
    }

    fn get_latest_descr_arc(
        &self,
        _frame: &crate::DeadFrame,
    ) -> std::sync::Arc<dyn majit_ir::Descr> {
        unreachable!("SyntheticCpu does not produce DeadFrames; get_latest_descr_arc unreachable")
    }

    fn get_int_value(&self, _frame: &crate::DeadFrame, _index: usize) -> i64 {
        unreachable!("SyntheticCpu does not produce DeadFrames; get_int_value unreachable")
    }

    fn get_value_direct(&self, frame: &crate::DeadFrame, slot: usize) -> i64 {
        // SyntheticCpu's slot space is the dense fail-value vector.
        self.get_int_value(frame, slot)
    }

    fn get_float_value(&self, _frame: &crate::DeadFrame, _index: usize) -> f64 {
        unreachable!("SyntheticCpu does not produce DeadFrames; get_float_value unreachable")
    }

    fn get_ref_value(&self, _frame: &crate::DeadFrame, _index: usize) -> majit_ir::GcRef {
        unreachable!("SyntheticCpu does not produce DeadFrames; get_ref_value unreachable")
    }

    fn invalidate_loop(&self, _token: &crate::JitCellToken) {
        unreachable!("SyntheticCpu does not own loops; invalidate_loop unreachable")
    }

    /// `rpython/jit/backend/model.py bh_call_i` declares `bh_call_*` on every
    /// `AbstractCPU`; `rpython/jit/metainterp/blackhole.py bhimpl_residual_call_r_i` shows
    /// `bhimpl_residual_call_*` and `bhimpl_inline_call_*` both routing
    /// through `cpu.bh_call_*`.
    ///
    /// Pyre's synthetic dispatch must decode `func` via
    /// [`decode_synthetic`] and run a nested `BlackholeInterpreter`
    /// against the resulting jitcode index.  `BlackholeInterpreter` lives
    /// in `majit-metainterp`, which depends on `majit-backend`, so the
    /// dispatch cannot be implemented here without a circular dependency.
    /// The cross-crate wiring shape (dispatcher closure injection vs
    /// `SyntheticCpu` relocation vs new trait callback) is deferred.
    /// Until then, the override panics so an
    /// accidentally-active synthetic backend in production fails loudly
    /// rather than silently returning 0 / `NULL` / 0.0 / `()` from the
    /// trait defaults.
    fn bh_call_i(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> i64 {
        unimplemented!("synthetic bh_call_i dispatch awaits cross-crate wiring")
    }

    fn bh_call_r(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        unimplemented!("synthetic bh_call_r dispatch awaits cross-crate wiring")
    }

    fn bh_call_f(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) -> f64 {
        unimplemented!("synthetic bh_call_f dispatch awaits cross-crate wiring")
    }

    fn bh_call_v(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_jitcode::jitcode::BhCallDescr,
    ) {
        unimplemented!("synthetic bh_call_v dispatch awaits cross-crate wiring")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_encodes_and_decodes_index() {
        for idx in [0_usize, 1, 7, 255, 65_535, 1 << 30] {
            let enc = encode_synthetic(idx);
            assert!(is_synthetic_fnaddr(enc), "0x{enc:016x} should be synthetic");
            assert_eq!(decode_synthetic(enc), Some(idx));
        }
    }

    #[test]
    fn real_fnaddrs_decode_to_none() {
        for fnaddr in [0_i64, 1, 0x0000_5555_aaaa_0000, i64::MAX] {
            assert!(!is_synthetic_fnaddr(fnaddr));
            assert_eq!(decode_synthetic(fnaddr), None);
        }
    }

    #[test]
    fn synthetic_base_is_disjoint_from_user_addresses() {
        const { assert!(SYNTHETIC_FNADDR_BASE < 0) };
        assert!(!is_synthetic_fnaddr(0));
        assert!(!is_synthetic_fnaddr(i64::MAX));
        assert!(is_synthetic_fnaddr(SYNTHETIC_FNADDR_BASE));
    }
}
