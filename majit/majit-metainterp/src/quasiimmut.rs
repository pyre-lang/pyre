//! `rpython/jit/metainterp/quasiimmut.py` — the residual half of an
//! `_immutable_fields_` entry spelled with a `?`.
//!
//! Only `do_force_quasi_immutable` lives here.  The recording half
//! (`QuasiImmutDescr`, `get_current_qmut_instance`, `register_loop_token`)
//! is reached through [`majit_ir::QuasiImmutHandle`], which the host
//! implements; this is the other end, the one a `jit_force_quasi_immutable`
//! bytecode runs when the blackhole executes the write that revokes the
//! folded loops.
//!
//! Upstream's hidden `mutate_<name>` field holds a `GCREF` to a `QuasiImmut`
//! instance, so the force reads the pointer, nulls it and invalidates what it
//! points at.  A host may embed the instance's state in the owner instead, in
//! which case unlinking and flipping every recorded loop flag is one call at
//! the field's address.  Both shapes fit the same signature — an address
//! derived from the struct and the mutate field descr — so majit computes the
//! address and the host performs the pair.

/// `quasiimmut.py`'s
/// ```python
/// cpu.bh_setfield_gc_r(p, ConstPtr.value, mutatefielddescr)
/// qmut = cast_gcref_to_instance(QuasiImmut, qmut_ref)
/// qmut.invalidate(mutatefielddescr.repr_of_descr())
/// ```
/// as one host call taking the address of the hidden mutate field.
pub type ForceQuasiImmutable = extern "C" fn(field_addr: i64);

/// `quasiimmut.py`:
/// ```python
/// def do_force_quasi_immutable(cpu, p, mutatefielddescr):
///     qmut_ref = cpu.bh_getfield_gc_r(p, mutatefielddescr)
///     if qmut_ref:
///         cpu.bh_setfield_gc_r(p, ConstPtr.value, mutatefielddescr)
///         qmut = cast_gcref_to_instance(QuasiImmut, qmut_ref)
///         qmut.invalidate(mutatefielddescr.repr_of_descr())
/// ```
///
/// The `if qmut_ref` test is the host's: it owns the representation that
/// answers whether any loop ever folded a read of this field, and the whole
/// point of the test is that an object no loop watches pays nothing.
pub fn do_force_quasi_immutable(
    struct_ptr: i64,
    mutatefielddescr: &majit_jitcode::jitcode::BhDescr,
) {
    if struct_ptr == 0 {
        return;
    }
    let hook = crate::pyjitpl::host_hooks().force_quasi_immutable;
    // Reaching here means the codewriter emitted `jit_force_quasi_immutable`
    // and the blackhole executed it, so the frontend is one that needs the
    // hook.  Returning quietly would skip the invalidation upstream always
    // performs, leaving every loop that folded a read of this field compiled
    // against a value the store just changed — a wrong answer with nothing to
    // trace it back to.
    let hook = hook.expect(
        "jit_force_quasi_immutable executed with no host invalidation routine: \
         call `publish_host_hooks` before running a jitcode that emits it",
    );
    let field_addr = struct_ptr.wrapping_add(mutatefielddescr.as_offset() as i64);
    hook(field_addr);
}
