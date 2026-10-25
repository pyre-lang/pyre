//! The parts of `rpython/jit/codewriter/call.py` the JIT runtime reads:
//! the info-handle traits `CallControl` carries across the crate boundary
//! and the symbolic fnaddr scheme.

use std::collections::HashMap;
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::parse::CallPath;

/// virtualizable.py `VirtualizableInfo.is_vtypeptr(TYPE)` —
/// identity check for the VTYPEPTR (struct-pointer type) the
/// virtualizable describes.
///
/// TODO: pyre has no `lltype` so VTYPEPTR identity is
/// expressed via a `usize` token (typically the SizeDescr identity from
/// `majit_ir::descr::descr_identity`).  Hosts attach their rich
/// `VirtualizableInfo` (defined in `majit-metainterp::virtualizable`) by
/// implementing this trait so codewriter, which sits below metainterp in
/// the crate graph, can still consult `jd.virtualizable_info` per
/// `call.py CallControl.get_vinfo`.
pub trait VirtualizableInfoHandle: std::fmt::Debug + Send + Sync {
    /// virtualizable.py `is_vtypeptr(TYPE) → TYPE == self.VTYPEPTR`.
    fn is_vtypeptr(&self, vtypeptr_id: usize) -> bool;
    /// warmspot.py `WarmRunnerDesc.finish` → `vinfo.finish()`.
    ///
    /// `virtualizable.py VirtualizableInfo.finish` stamps
    /// `clear_vable_ptr` / `clear_vable_descr`. The stamp happens at
    /// construction (`set_clear_vable` / `build_pyframe_virtualizable_info`).
    /// Residual `jit_force_virtualizable` rewrite lives on
    /// `CallControl::finish` (`replace_force_virtualizable_with_call`
    /// over the graphs this handle does not own). Default is a no-op.
    fn finish(&self) {}
    /// Codewriter-side VTYPE name (`red_types[index_of_virtualizable]`).
    fn vtype_name(&self) -> Option<&str> {
        None
    }
    /// `fname in vinfo.static_field_to_extra_box` (`jtransform.py
    /// is_virtualizable_getset`).
    fn has_static_field(&self, _name: &str) -> bool {
        false
    }
    /// `fname in vinfo.array_fields` (`jtransform.py
    /// is_virtualizable_getset`).
    fn has_array_field(&self, _name: &str) -> bool {
        false
    }
    /// `vinfo.static_field_to_extra_box[fieldname]` (`jtransform.py
    /// get_virtualizable_field_descr`).
    fn static_field_index(&self, _name: &str) -> Option<usize> {
        None
    }
}

/// greenfield.py `GreenFieldInfo.green_fields` membership test.
///
/// TODO: same crate-boundary reasoning as
/// `VirtualizableInfoHandle`.  Hosts implement this on their rich
/// `GreenFieldInfo` so `CallControl.could_be_green_field`
/// (call.py) can walk `jd.greenfield_info` without depending on
/// metainterp.
pub trait GreenFieldInfoHandle: std::fmt::Debug + Send + Sync {
    /// `(GTYPE, fieldname) in self.green_fields`.
    fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool;
}

/// `virtualref.py VirtualRefInfo` opaque carrier handle.
///
/// TODO: same crate-boundary reasoning as
/// `VirtualizableInfoHandle`.  `VirtualRefInfo` is defined in
/// `majit-metainterp::virtualref` (the JIT runtime side); codewriter
/// sits below metainterp in the crate graph and cannot import it.
/// Hosts implement this trait on `VirtualRefInfo` so
/// `CodeWriter.setup_vrefinfo` (`codewriter.py`) can store the
/// instance on `CallControl.virtualref_info`
/// (`call.py CallControl virtualref_info = None`) for later forwarding to
/// `metainterp_sd.virtualref_info = codewriter.callcontrol.virtualref_info`
/// (`pyjitpl.py:2267`).  The three accessors expose the three `u32`
/// descriptor indices the rebuilt `VirtualRefInfo` consumes on the
/// metainterp side — `descr_virtual_token` / `descr_forced` index the
/// `JitVirtualRef` field descriptors, `descr_size` indexes the struct
/// size descriptor.
pub trait VirtualRefInfoHandle: std::fmt::Debug + Send + Sync {
    /// `virtualref.py:48 jit_virtual_ref_vtable` ↔ pyre
    /// `VirtualRefInfo.descr_virtual_token` — field descr index for
    /// `JitVirtualRef.virtual_token`.
    fn descr_virtual_token(&self) -> u32;
    /// `virtualref.py:49 jit_virtual_ref_vtable` ↔ pyre
    /// `VirtualRefInfo.descr_forced` — field descr index for
    /// `JitVirtualRef.forced`.
    fn descr_forced(&self) -> u32;
    /// `virtualref.py:48-49` size token ↔ pyre
    /// `VirtualRefInfo.descr_size` — size descr index for the
    /// `JitVirtualRef` struct itself.
    fn descr_size(&self) -> u32;
}

/// High-16-bit tag stamped on every symbolic fnaddr hash so consumers can
/// discriminate a placeholder from a real code address by an exact bit
/// pattern instead of a range heuristic.  User-space addresses keep bits
/// 48..64 clear on every 64-bit target pyre builds for, and wasm32 addresses
/// are 32-bit, so no real funcptr can carry the tag.  (A bit-47 range test is
/// NOT enough: aarch64 Linux uses a 48-bit VA and maps PIE code and mmap
/// regions with bit 47 set — 0xaaab…/0xffff… — so every real funcptr there
/// would read as symbolic.)  Same scheme as
/// `assembler::STR_CONST_SENTINEL_BASE`.
///
/// Bit 63 stays CLEAR: `BhDescr::JitCode.fnaddr` also carries the synthetic
/// tag, whose whole space is "negative `i64`"
/// (`majit-backend::synthetic_cpu`: `SYNTHETIC_FNADDR_BASE = i64::MIN`,
/// `is_synthetic_fnaddr(x) = x < 0`).  A tag setting bit 63 would put every
/// symbolic hash inside that space, and `decode_synthetic` would answer with
/// a fabricated jitcode index.
pub const SYMBOLIC_FNADDR_HIGH_MASK: u64 = 0xFFFF_0000_0000_0000;
pub const SYMBOLIC_FNADDR_BASE: u64 = 0x7ADD_0000_0000_0000;

/// Whether `fnaddr` is a `symbolic_fnaddr_for_path` placeholder rather than a
/// callable code address.
#[inline]
pub fn is_symbolic_fnaddr(fnaddr: i64) -> bool {
    let bits = fnaddr as u64;
    let high = bits & SYMBOLIC_FNADDR_HIGH_MASK;
    if high == SYMBOLIC_FNADDR_BASE {
        return true;
    }
    // A raw host funcptr keeps bits 48..63 clear (bit 47 may be set on
    // aarch64 Linux). A word whose high half is set and whose sign bit is
    // clear is not code and not a synthetic fnaddr (`i64::MIN` space): it is
    // the bytes of a path that were stored where the address should be.
    // Jumping to it faults. Decline it the same way as a symbolic hash.
    high != 0 && bits & 0x8000_0000_0000_0000 == 0
}

/// Whether `fnaddr` can be called as a function pointer.
///
/// Stricter than `!is_symbolic_fnaddr`: `0` is `jitcode.py JitCode.__init__`
/// `fnaddr=None` and must not be called either. Residual and inline blackhole
/// handlers (`reject_unresolved_call`, `bhimpl_inline_call_*`) decline it
/// before a backend jump; otherwise a callee that never ran could fabricate
/// a return value.
#[inline]
pub fn is_callable_fnaddr(fnaddr: i64) -> bool {
    fnaddr != 0 && !is_symbolic_fnaddr(fnaddr)
}

pub fn stable_symbolic_fnaddr<T: std::hash::Hash>(value: &T) -> i64 {
    use std::hash::Hasher;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    ((hasher.finish() & !SYMBOLIC_FNADDR_HIGH_MASK) | SYMBOLIC_FNADDR_BASE) as i64
}

static SYMBOLIC_FNADDR_PATHS: OnceLock<Mutex<HashMap<i64, String>>> = OnceLock::new();

pub fn record_symbolic_fnaddr(value: i64, description: String) {
    let registry = SYMBOLIC_FNADDR_PATHS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut paths = registry.lock();
    paths
        .entry(value)
        .and_modify(|existing| {
            if description < *existing {
                existing.clone_from(&description);
            }
        })
        .or_insert(description);
}

pub fn symbolic_fnaddr_paths_snapshot() -> Vec<(i64, String)> {
    let registry = SYMBOLIC_FNADDR_PATHS.get_or_init(|| Mutex::new(HashMap::new()));
    let paths = registry.lock();
    let mut snapshot: Vec<_> = paths
        .iter()
        .map(|(&value, description)| (value, description.clone()))
        .collect();
    snapshot.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
    snapshot
}

pub fn symbolic_fnaddr_for_path(path: &CallPath) -> i64 {
    let symbolic = stable_symbolic_fnaddr(path);
    record_symbolic_fnaddr(symbolic, path.canonical_key());
    symbolic
}

/// Compute the symbolic function address for the same segmented path shape
/// used by the codewriter.
pub fn symbolic_fnaddr_for_segments<'a>(segments: impl IntoIterator<Item = &'a str>) -> i64 {
    let path = CallPath::from_segments(segments);
    symbolic_fnaddr_for_path(&path)
}
