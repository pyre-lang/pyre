//! GIL-disabled concurrent mutators. Own process: the mode is process-global
//! and one-way.

use majit_gc::collector::{GcConfig, MiniMarkGC};
use majit_gc::nursery::NURSERY_CHUNK_SIZE;
use majit_gc::shadow_stack::{self, MutatorExtraWalkFn};
use majit_gc::{GcAllocator, GcHandle, TypeInfo, gc_sync, gc_write_barrier_from_array, rgil};
use majit_ir::GcRef;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// The GIL-off mode is process-global; keep the tests in this binary off
/// each other's mutator registry and singleton.
static NOGIL_SERIAL: Mutex<()> = Mutex::new(());

fn lock_nogil() -> std::sync::MutexGuard<'static, ()> {
    NOGIL_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const NURSERY: usize = 256 * 1024;
const THREADS: usize = 4;
const ROOTS: usize = 64;

unsafe fn walk_roots(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
    let slots = unsafe { &mut *(data as *mut [GcRef; ROOTS]) };
    for slot in slots.iter_mut() {
        visitor(slot);
    }
}

fn install_gc() -> (u32, u32) {
    if gc_sync::is_initialized() {
        rgil::disable_gil();
        return (0, 1);
    }
    let mut gc = MiniMarkGC::with_config(GcConfig {
        nursery_size: NURSERY,
        large_object_threshold: 32 * 1024,
        ..GcConfig::default()
    });
    let simple = gc.register_type(TypeInfo::simple(16));
    let array = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));
    assert_eq!(simple, 0);
    gc_sync::store_singleton(Box::new(gc));
    rgil::disable_gil();
    (simple, array)
}

fn tls_nursery_ptrs() -> majit_gc::nursery::NurseryPtrs {
    unsafe { *majit_gc::threadlocal::threadlocal_nursery_ptrs() }
}

fn tls_nursery_is_empty() -> bool {
    let ptrs = tls_nursery_ptrs();
    ptrs.free.is_null() && ptrs.top.is_null()
}

/// Advance the singleton arena cursor so a later unregistered bump sits
/// away from the nursery start that `_minor_collection` resets to.
fn fill_nursery_past_first_chunk(type_id: u32) {
    let h = thread::spawn(move || {
        register_worker();
        loop {
            let remaining = gc_sync::gc_query(|gc| {
                (gc.nursery_top() as usize).saturating_sub(gc.nursery_free() as usize)
            });
            if remaining <= NURSERY_CHUNK_SIZE * 2 {
                break;
            }
            let obj = gc_sync::gc_op(|gc| gc.try_alloc_nursery_bump_typed(type_id, 16));
            if obj.is_null() {
                break;
            }
        }
        gc_sync::gc_thread_die();
    });
    h.join().unwrap();
}

/// An unregistered thread must bump the arena, not a thread chunk. After a
/// foreign minor, the next allocation must not land in the pre-reset range.
fn unregistered_thread_alloc_not_in_stale_chunk(type_id: u32) {
    fill_nursery_past_first_chunk(type_id);

    let start = Arc::new(std::sync::Barrier::new(2));
    let after_first = Arc::new(std::sync::Barrier::new(2));
    let after_minor = Arc::new(std::sync::Barrier::new(2));

    let unreg = {
        let start = start.clone();
        let after_first = after_first.clone();
        let after_minor = after_minor.clone();
        thread::spawn(move || {
            start.wait();
            let first = alloc_small(type_id);
            assert!(!first.is_null());
            assert!(
                tls_nursery_is_empty(),
                "unregistered alloc must not install a thread chunk"
            );
            let old_lo = first.0;
            let old_hi = first.0.saturating_add(NURSERY_CHUNK_SIZE);
            after_first.wait();
            after_minor.wait();
            let second = alloc_small(type_id);
            assert!(!second.is_null());
            assert!(
                tls_nursery_is_empty(),
                "unregistered pair stays empty across a foreign minor"
            );
            assert!(
                second.0 < old_lo || second.0 >= old_hi,
                "stale chunk reuse: first={:#x} second={:#x}",
                first.0,
                second.0
            );
        })
    };

    let collector = {
        let start = start.clone();
        let after_first = after_first.clone();
        let after_minor = after_minor.clone();
        thread::spawn(move || {
            register_worker();
            start.wait();
            after_first.wait();
            gc_sync::gc_op_stw(&mut [], |gc, _| gc.collect_nursery());
            after_minor.wait();
            gc_sync::gc_thread_die();
        })
    };

    unreg.join().unwrap();
    collector.join().unwrap();
}

/// Removing the mutator registry entry empties this thread's pair under
/// the same lock, so a pair never outlives the entry a minor would clear.
fn unregister_empties_tls_nursery_pair(type_id: u32) {
    let h = thread::spawn(move || {
        shadow_stack::register_mutator();
        let obj = alloc_small(type_id);
        assert!(!obj.is_null());
        assert!(
            !tls_nursery_is_empty(),
            "a registered mutator must receive a thread chunk"
        );
        shadow_stack::unregister_mutator();
        assert!(
            tls_nursery_is_empty(),
            "unregister empties the thread nursery pair"
        );
    });
    h.join().unwrap();
}

/// A non-singleton `MiniMarkGC` bumps `self.nursery`, like
/// `incminimark.py` `malloc_fixedsize` bumping `self.nursery_free`.
fn non_singleton_minimark_bumps_its_own_nursery() {
    let h = thread::spawn(|| {
        shadow_stack::register_mutator();
        let before = tls_nursery_ptrs();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: NURSERY,
            large_object_threshold: 32 * 1024,
            ..GcConfig::default()
        });
        let type_id = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(type_id, 16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
        let after = tls_nursery_ptrs();
        assert_eq!(
            (before.free, before.top),
            (after.free, after.top),
            "non-singleton must not install a thread chunk"
        );
        shadow_stack::unregister_mutator();
    });
    h.join().unwrap();
}

fn register_worker() {
    shadow_stack::register_mutator();
    gc_sync::register_thread();
}

fn alloc_small(type_id: u32) -> GcRef {
    gc_sync::safepoint_poll();
    let mut root = GcRef(0);
    let mut wb = false;
    unsafe { majit_gc::alloc_fast_nursery_collecting_typed_rooted(type_id, 16, &mut root, &mut wb) }
}

fn chunks_are_disjoint(type_id: u32) {
    let start = Arc::new(std::sync::Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|thread_index| {
            let start = start.clone();
            thread::spawn(move || {
                register_worker();
                let mut roots = [GcRef(0); ROOTS];
                unsafe {
                    shadow_stack::register_mutator_extra_area(
                        walk_roots as MutatorExtraWalkFn,
                        roots.as_mut_ptr() as *const (),
                        "nogil_chunks",
                    );
                }
                start.wait();
                for i in 0..200_000u64 {
                    let obj = alloc_small(type_id);
                    assert!(!obj.is_null());
                    let payload = 0xA11C_0000_0000_0000u64 | ((thread_index as u64) << 32) | i;
                    unsafe { *(obj.0 as *mut u64) = payload };
                    roots[(i as usize) % ROOTS] = obj;
                }
                for (slot_i, slot) in roots.iter().enumerate() {
                    let payload = unsafe { *(slot.0 as *const u64) };
                    let idx = (200_000u64 - ROOTS as u64) + slot_i as u64;
                    let expected = 0xA11C_0000_0000_0000u64 | ((thread_index as u64) << 32) | idx;
                    assert_eq!(payload, expected, "thread {thread_index} slot {slot_i}");
                }
                gc_sync::gc_thread_die();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

fn write_barrier_races_with_collection(simple: u32, array_tid: u32) {
    let start = Arc::new(std::sync::Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|thread_index| {
            let start = start.clone();
            thread::spawn(move || {
                register_worker();
                let mut roots = [GcRef(0); ROOTS];
                unsafe {
                    shadow_stack::register_mutator_extra_area(
                        walk_roots as MutatorExtraWalkFn,
                        roots.as_mut_ptr() as *const (),
                        "nogil_wb",
                    );
                }
                start.wait();
                let array = gc_sync::gc_op_alloc(
                    |gc| gc.try_alloc_varsize_typed_bump(array_tid, 8, 8, 1),
                    |gc| gc.alloc_varsize_typed(array_tid, 8, 8, 1),
                );
                assert!(!array.is_null());
                roots[0] = array;
                gc_sync::gc_op_stw(&mut [], |gc, _| gc.collect_nursery());
                let array = roots[0];
                for i in 0..50_000u64 {
                    let young = alloc_small(simple);
                    assert!(!young.is_null());
                    let payload = 0xBEEF_0000_0000_0000u64 | ((thread_index as u64) << 32) | i;
                    unsafe { *(young.0 as *mut u64) = payload };
                    gc_write_barrier_from_array(array, 0);
                    unsafe { *((array.0 + 8) as *mut GcRef) = young };
                    let _next = alloc_small(simple);
                    gc_sync::safepoint_poll();
                    let got = unsafe { *((array.0 + 8) as *const GcRef) };
                    assert_eq!(
                        unsafe { *(got.0 as *const u64) },
                        payload,
                        "thread {thread_index} round {i}"
                    );
                }
                gc_sync::gc_thread_die();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

fn concurrent_collect_requests(type_id: u32) {
    let start = Arc::new(std::sync::Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|thread_index| {
            let start = start.clone();
            thread::spawn(move || {
                register_worker();
                let mut roots = [GcRef(0); ROOTS];
                unsafe {
                    shadow_stack::register_mutator_extra_area(
                        walk_roots as MutatorExtraWalkFn,
                        roots.as_mut_ptr() as *const (),
                        "nogil_collect",
                    );
                }
                start.wait();
                for round in 0..200u64 {
                    gc_sync::safepoint_poll();
                    gc_sync::gc_op_stw(&mut [], |gc, _| gc.collect_full());
                    for k in 0..32u64 {
                        let obj = alloc_small(type_id);
                        assert!(!obj.is_null());
                        unsafe {
                            *(obj.0 as *mut u64) = 0xC011_0000_0000_0000u64
                                | ((thread_index as u64) << 32)
                                | (round << 8)
                                | k;
                        }
                        roots[(k as usize) % ROOTS] = obj;
                    }
                }
                gc_sync::gc_thread_die();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// Thread A holds a nursery object only in a local `GcRef` slot (not on
/// the shadow stack) and calls a rooted collecting allocation with that
/// slot while thread B loops minor collections. After A returns, the slot
/// must name the live forwarded object with its payload intact.
///
/// Without the GIL-off write-back in `gc_op_stw_slots`, B's collection
/// moves the object while A is parked and A's slot stays a from-space
/// address.
fn rooted_slot_survives_foreign_minor_during_collecting_alloc(type_id: u32) {
    const ROUNDS: u64 = 64;
    let start = Arc::new(std::sync::Barrier::new(2));
    let stop = Arc::new(AtomicBool::new(false));
    let collecting = Arc::new(AtomicBool::new(false));
    let collector = {
        let start = start.clone();
        let stop = stop.clone();
        let collecting = collecting.clone();
        thread::spawn(move || {
            register_worker();
            start.wait();
            while !stop.load(Ordering::Relaxed) {
                collecting.store(true, Ordering::Release);
                gc_sync::gc_op_stw(&mut [], |gc, _| gc.collect_nursery());
                collecting.store(false, Ordering::Release);
            }
            gc_sync::gc_thread_die();
        })
    };
    let mutator = {
        let start = start.clone();
        let collecting = collecting.clone();
        thread::spawn(move || {
            register_worker();
            start.wait();
            for round in 0..ROUNDS {
                while !collecting.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut slot = loop {
                    let obj = gc_sync::gc_op(|gc| gc.try_alloc_nursery_bump_typed(type_id, 16));
                    if !obj.is_null() {
                        break obj;
                    }
                    gc_sync::gc_op_stw(&mut [], |gc, _| gc.collect_nursery());
                };
                let payload = 0xD00D_0000_0000_0000u64 | round;
                unsafe { *(slot.0 as *mut u64) = payload };
                while !gc_sync::gc_op(|gc| gc.try_alloc_nursery_bump_typed(type_id, 16)).is_null() {
                }
                let mut wb = false;
                let fresh = unsafe {
                    majit_gc::standalone_alloc_nursery_collecting_typed_rooted(
                        type_id, 16, &mut slot, &mut wb,
                    )
                };
                assert!(!fresh.is_null(), "round {round}: collecting alloc failed");
                assert!(!slot.is_null(), "round {round}: slot cleared");
                let hdr = unsafe { majit_gc::header::header_of(slot.0) };
                assert!(
                    !unsafe { (*hdr).is_forwarded() },
                    "round {round}: slot still a forwarding stub at {:#x}",
                    slot.0
                );
                assert_eq!(
                    unsafe { *(slot.0 as *const u64) },
                    payload,
                    "round {round}: payload lost at {:#x}",
                    slot.0
                );
            }
            gc_sync::gc_thread_die();
        })
    };
    mutator.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    collector.join().unwrap();
}

fn concurrent_full_collections_all_run() {
    const PAIR: usize = 2;
    const EACH: usize = 8;
    let majors_before = gc_sync::gc_query(|gc| gc.collection_counts().1);
    let start = Arc::new(std::sync::Barrier::new(PAIR));
    let handles: Vec<_> = (0..PAIR)
        .map(|_| {
            let start = start.clone();
            thread::spawn(move || {
                register_worker();
                start.wait();
                for _ in 0..EACH {
                    gc_sync::gc_op_stw(&mut [], |gc, _| gc.collect_full());
                }
                gc_sync::gc_thread_die();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let majors_after = gc_sync::gc_query(|gc| gc.collection_counts().1);
    assert!(
        majors_after >= majors_before + PAIR * EACH,
        "each concurrent full collection must run: before={majors_before} after={majors_after} expected>={}",
        majors_before + PAIR * EACH
    );
}

/// The STW drain parks registered mutators only. An unregistered thread's
/// `gc_op` takes `gc_lock` alone, so the owner holds that lock for as long
/// as it borrows `&mut MiniMarkGC`.
fn stw_owner_excludes_unregistered_gc_op() {
    register_worker();
    struct Die;
    impl Drop for Die {
        fn drop(&mut self) {
            gc_sync::gc_thread_die();
        }
    }
    let _die = Die;
    let entered = Arc::new(AtomicBool::new(false));
    let about_to_enter = Arc::new(std::sync::Barrier::new(2));
    let unreg = {
        let entered = entered.clone();
        let about_to_enter = about_to_enter.clone();
        thread::spawn(move || {
            about_to_enter.wait();
            gc_sync::gc_op(|_gc| {
                entered.store(true, Ordering::SeqCst);
            });
        })
    };
    gc_sync::gc_op_stw(&mut [], |_gc, _| {
        about_to_enter.wait();
        thread::sleep(Duration::from_millis(100));
        assert!(
            !entered.load(Ordering::SeqCst),
            "unregistered gc_op entered while the STW owner held the collector"
        );
    });
    unreg.join().unwrap();
    assert!(
        entered.load(Ordering::SeqCst),
        "unregistered gc_op must run once the STW owner releases the collector"
    );
}

/// A GIL-off young nonmoving birth whose size would trip `threshold_reached`
/// takes the collecting path (`minor_collection_with_major_progress`) rather
/// than allocating and only arming a deferred major.
fn nonmoving_alloc_collects_at_threshold(type_id: u32) {
    let minor_before = gc_sync::gc_query(|gc| gc.collection_counts().0);
    gc_sync::gc_op(|gc| gc.set_next_major_collection_threshold(0.0));
    let mut handle = GcHandle;
    let obj = handle.alloc_young_nonmoving_typed(type_id, 16);
    assert!(
        !obj.is_null(),
        "young nonmoving alloc at threshold returned null"
    );
    let minor_after = gc_sync::gc_query(|gc| gc.collection_counts().0);
    assert!(
        minor_after > minor_before,
        "threshold_reached must collect before the nonmoving birth: before={minor_before} after={minor_after}"
    );
}

fn with_watchdog(seconds: u64, f: impl FnOnce()) {
    let done = Arc::new(AtomicBool::new(false));
    let watchdog = {
        let done = done.clone();
        thread::spawn(move || {
            let t0 = Instant::now();
            while !done.load(Ordering::Relaxed) {
                if t0.elapsed() > Duration::from_secs(seconds) {
                    // A panic here would only end this thread while `f` stays
                    // blocked; abort so a deadlock fails the test.
                    eprintln!("nogil watchdog: exceeded {seconds}s");
                    std::process::abort();
                }
                thread::sleep(Duration::from_millis(100));
            }
        })
    };
    f();
    done.store(true, Ordering::Relaxed);
    watchdog.join().unwrap();
}

#[test]
fn gil_disabled_unregistered_thread_chunk_does_not_survive_minor() {
    let _serial = lock_nogil();
    let (simple, _) = install_gc();
    unregistered_thread_alloc_not_in_stale_chunk(simple);
}

#[test]
fn gil_disabled_unregister_clears_thread_nursery_chunk() {
    let _serial = lock_nogil();
    let (simple, _) = install_gc();
    unregister_empties_tls_nursery_pair(simple);
}

#[test]
fn gil_disabled_non_singleton_bumps_own_nursery() {
    let _serial = lock_nogil();
    let _ = install_gc();
    non_singleton_minimark_bumps_its_own_nursery();
}

#[test]
fn gil_disabled_stw_owner_excludes_unregistered_gc_op() {
    let _serial = lock_nogil();
    let _ = install_gc();
    with_watchdog(15, stw_owner_excludes_unregistered_gc_op);
}

#[test]
fn gil_disabled_nonmoving_alloc_collects_at_threshold() {
    let _serial = lock_nogil();
    let (simple, _) = install_gc();
    with_watchdog(15, || nonmoving_alloc_collects_at_threshold(simple));
}

#[test]
fn gil_disabled_concurrent_mutators() {
    let _serial = lock_nogil();
    let (simple, array) = install_gc();
    chunks_are_disjoint(simple);
    write_barrier_races_with_collection(simple, array);
    with_watchdog(120, || {
        concurrent_collect_requests(simple);
        concurrent_full_collections_all_run();
        rooted_slot_survives_foreign_minor_during_collecting_alloc(simple);
    });
}
