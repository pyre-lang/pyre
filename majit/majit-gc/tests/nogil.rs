//! GIL-disabled concurrent mutators. Own process: the mode is process-global
//! and one-way.

use majit_gc::collector::{GcConfig, MiniMarkGC};
use majit_gc::shadow_stack::{self, MutatorExtraWalkFn};
use majit_gc::{GcAllocator, TypeInfo, gc_sync, gc_write_barrier_from_array, rgil};
use majit_ir::GcRef;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

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
                let array = gc_sync::gc_op(|gc| gc.alloc_varsize_typed(array_tid, 8, 8, 1));
                assert!(!array.is_null());
                roots[0] = array;
                gc_sync::gc_op(|gc| gc.collect_nursery());
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
                    gc_sync::gc_op(|gc| gc.collect_full());
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
fn gil_disabled_concurrent_mutators() {
    let (simple, array) = install_gc();
    chunks_are_disjoint(simple);
    write_barrier_races_with_collection(simple, array);
    with_watchdog(120, || concurrent_collect_requests(simple));
}
