//! Generic GC-managed boxes for host-owned storage containers.
//!
//! This is the storage-container generalization of the established raw
//! `BigInt` box in `longobject.rs`: [`crate::longobject::BIGINT_PAYLOAD_SIZE`],
//! [`crate::longobject::bigint_destructor`], and
//! [`crate::longobject::alloc_bigint_stable`].

/// Allocate `value` inside a GC-managed, non-moving old-gen box tagged `tid`.
///
/// The box layout is `[GcHeader | T]`, so the returned `*mut T` points at the
/// payload exactly as [`crate::lltype::malloc_raw`] would. This makes it a
/// drop-in replacement at storage-box construction sites. The GC sweep
/// reclaims the box and runs the registered drop-glue destructor (see
/// [`storage_box_destructor`]), so the caller must not use `Box::from_raw` for
/// a GC-managed result.
///
/// The payload size registered in the matching `TypeInfo` must be exactly
/// `size_of::<T>()`, mirroring `longobject.rs`'s `BIGINT_PAYLOAD_SIZE` invariant.
///
/// Allocation uses [`crate::gc_hook::try_gc_alloc_stable_raw`] because storage
/// containers are self-mutating: their methods re-derive `self` from a raw
/// pointer across allocating calls, so the box address must remain stable. This
/// is the same non-moving rule used by
/// [`crate::longobject::alloc_bigint_stable`] and mapdict storage.
///
/// When `tid == 0` or no GC hook is installed (unit tests and pre-init), this
/// falls back to [`crate::lltype::malloc_raw`]. In that case the caller's
/// existing manual-free path remains responsible for the allocation.
// `specialize:argtype(0)` (`specialize.py` `specialize_argtype`) gives each
// payload shape its own residual graph. The call stays residual.
#[majit_macros::specialize_argtype(0)]
#[majit_macros::dont_look_inside]
#[inline(never)]
pub fn gc_alloc_storage_box<T: 'static>(value: T, tid: u32) -> *mut T {
    if tid != 0 {
        let raw = crate::gc_hook::try_gc_alloc_stable_raw(tid, std::mem::size_of::<T>());
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut T, value);
            }
            // The payload write is a store into an old-gen box. During MARKING
            // `oldgen_birth_flags` allocates it VISITED (black), so a filled
            // table (dict `copy`, `switch_to_object_strategy` refill) can
            // already hold white children. `write_barrier` /
            // `remember_young_pointer` re-greys the box
            // (`_add_to_more_objects_to_trace_if_black`).
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            return raw as *mut T;
        }
    }
    crate::lltype::malloc_raw(value)
}

/// [`gc_alloc_storage_box`] born young: `malloc_fixed_or_varsize_nonmovable`
/// (`external_malloc(..., alloc_young=True)`).
///
/// The box keeps the non-moving address the self-mutating rule needs, but a
/// minor collection frees it once nothing reaches it, instead of the next
/// major.  A dict's `dicttable` and the type's name string are nursery
/// objects upstream, so a program that builds and drops a class or a dict
/// gives the whole graph back on the next minor.  The caller owes the box a
/// root until it is stored into a traced object, and the store takes the
/// holder's `setfield_gc` barrier when the holder is old.
///
/// `value` is built on the Rust stack before the birth, so the entry is the
/// no-collect twin of `external_malloc`: a collection run inside the malloc
/// would leave the payload's own references stale.  A refused young birth
/// (`MAJIT_GC_YOUNG_RAWMALLOC=0`) is born old instead, and the barrier after
/// the write keeps a filled table's children reachable in that case; it is a
/// no-op on a young header.
// `specialize:argtype(0)` (`specialize.py` `specialize_argtype`) gives each
// payload shape its own residual graph, matching [`gc_alloc_storage_box`].
// Charon extracts with `monomorphize: false`, so a generic body without
// trait clauses is one unspecialized graph; the hint is what splits it.
#[majit_macros::specialize_argtype(0)]
#[majit_macros::dont_look_inside]
#[inline(never)]
pub fn gc_alloc_young_storage_box<T: 'static>(value: T, tid: u32) -> *mut T {
    if tid != 0 {
        let raw = crate::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(
            tid,
            std::mem::size_of::<T>(),
        );
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut T, value);
            }
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            return raw as *mut T;
        }
    }
    crate::lltype::malloc_raw(value)
}

/// GC-sweep destructor for a storage box built by
/// [`gc_alloc_storage_box::<T>`].
///
/// Runs `T`'s drop glue in place, reclaiming the container's owned heap buffer.
/// This is the generic form of `longobject.rs`'s
/// [`crate::longobject::bigint_destructor`], registered through
/// `TypeInfo::with_destructor` (or composed with a custom trace).
///
/// # Safety
///
/// `addr` must point at a live `T` payload allocated by
/// [`gc_alloc_storage_box::<T>`]. The collector must invoke this exactly once.
pub unsafe fn storage_box_destructor<T: 'static>(addr: usize) {
    unsafe { std::ptr::drop_in_place(addr as *mut T) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::mem::MaybeUninit;

    thread_local! {
        static DROP_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    struct DropProbe(u32);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            DROP_COUNT.with(|count| count.set(count.get() + 1));
        }
    }

    #[test]
    fn raw_fallback_round_trips_and_destructor_runs_drop_glue_once() {
        DROP_COUNT.with(|count| count.set(0));

        let storage = gc_alloc_storage_box(DropProbe(42), 0);
        assert!(!storage.is_null());
        unsafe {
            assert_eq!((*storage).0, 42);
            storage_box_destructor::<DropProbe>(storage as usize);

            // `tid == 0` guarantees the `malloc_raw` fallback, so reclaim the
            // allocation without running the already-invoked drop glue again.
            drop(Box::from_raw(storage.cast::<MaybeUninit<DropProbe>>()));
        }

        DROP_COUNT.with(|count| assert_eq!(count.get(), 1));
    }
}
