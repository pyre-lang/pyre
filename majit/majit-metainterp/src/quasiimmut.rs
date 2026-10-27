//! `rpython/jit/metainterp/quasiimmut.py` — the residual half of an
//! `_immutable_fields_` entry spelled with a `?`.
//!
//! `do_force_quasi_immutable` and `get_current_qmut_instance` live here.
//! The host owns the mutate-field representation; majit computes the field
//! address from the struct pointer and the mutate descr, then calls the
//! matching [`HostHooks`](crate::pyjitpl::HostHooks) entry.  The recording
//! half's `QuasiImmut` instance is reached through
//! [`majit_ir::QuasiImmutHandle`], which the host implements; the force
//! half is the one a `jit_force_quasi_immutable` bytecode runs when the
//! write revokes the folded loops.
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

/// `quasiimmut.py get_current_qmut_instance(cpu, gcref, mutatefielddescr)`
/// as one host call taking the address of the hidden mutate field.
///
/// Returns the field's current `QuasiImmut` instance, creating one when
/// the field is still null.  A Rust `fn` (not `extern "C"`) because the
/// result is an [`Arc`](std::sync::Arc) trait object.
pub type GetCurrentQmutInstance =
    fn(field_addr: i64) -> std::sync::Arc<dyn majit_ir::QuasiImmutHandle>;

/// `quasiimmut.py get_current_qmut_instance(cpu, gcref, mutatefielddescr)`.
///
/// Computes the mutate-field address the same way
/// [`do_force_quasi_immutable`] does and asks the host for the instance.
pub fn get_current_qmut_instance(
    struct_ptr: i64,
    mutatefielddescr: &majit_jitcode::jitcode::BhDescr,
) -> std::sync::Arc<dyn majit_ir::QuasiImmutHandle> {
    let hook = crate::pyjitpl::host_hooks().get_current_qmut_instance;
    // Reaching here means the codewriter emitted `record_quasiimmut_field`
    // and the tracer is walking it, so the frontend is one that needs the
    // hook.  Returning a dummy handle would mint a `QuasiImmutDescr` that
    // cannot later invalidate the loops that folded this read.
    let hook = hook.expect(
        "record_quasiimmut_field executed with no host qmut-instance routine: \
         call `publish_host_hooks` before running a jitcode that emits it",
    );
    let field_addr = struct_ptr.wrapping_add(mutatefielddescr.as_offset() as i64);
    hook(field_addr)
}

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
///
/// `p` is an address: a GC ref payload for `/rd`, a raw struct pointer
/// for `/id`. The mutate field is reached by adding the descr offset
/// (`quasiimmut.py do_force_quasi_immutable` `cpu.bh_getfield_gc_r(p,
/// mutatefielddescr)`), which is valid for a raw struct as well as a
/// GC one — the host hook then unlinks at that field address.
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
