//! Per-loop GC table for reference constants baked into compiled traces.
//!
//! Counterpart of `rpython/jit/backend/llsupport/gcreftracer.py`. A
//! compiled loop/bridge that references constant GC objects must keep
//! those references alive and up to date across moving collections.
//! Instead of baking the raw `GcRef` value as a machine-code immediate
//! (which a moving GC cannot find or update), the backend bakes the
//! address of a per-loop array of reference slots and emits a
//! `LoadFromGcTable(index)` load (`x86/assembler.py _addr_from_gc_table`
//! `genop_load_from_gc_table`). Each slot is a GC root: the collector
//! forwards it in place during a collection step, so the next load
//! observes the relocated object.
//!
//! Upstream models the array with a `GCREFTRACER` `GcStruct`
//! (`gcreftracer.py`) carrying `array_base_addr` + `array_length`,
//! registered with the GC via a custom trace hook
//! (`register_custom_trace_hook`, `gcreftracer.py`) and *reached* by
//! the collector through the loop token's `asmmemmgr_gcreftracers`, which
//! in RPython is a GC-managed list (`x86/assembler.py`,
//! `model.py:294`) so the tracer header sits in the live object graph.
//!
//! pyre has the custom-trace-hook facility — `TypeInfo::custom_trace`
//! (`trace.rs`, `register_custom_trace_hook` parity), already used by
//! `JITFRAME` (`jitframe.rs`), the structural twin of `GCREFTRACER` —
//! so a managed `GCREFTRACER` GcStruct with a slot-tracing hook is
//! portable *in isolation*. What is not yet portable is the
//! *reachability*: pyre's `CompiledLoopToken` is a Rust-owned struct and
//! its `asmmemmgr_gcreftracers` is a `Mutex<Vec<Arc<dyn Any>>>` keepalive
//! (`CompiledLoopToken` in `majit-backend/src/lib.rs`), not a GC edge — the
//! collector has no
//! object-graph path to a managed header. A GcStruct header would still
//! need a GC root to be reached, and that root would be a walker over the
//! Rust-owned keepalive — a header allocation, a hook, and a mark step
//! added without removing the walker.
//!
//! So pyre forwards the slots directly from an extra root walker
//! (`shadow_stack::register_extra_root_walker`, walked at
//! `collector.rs`'s `do_collect_nursery` and
//! `rescan_major_nonstack_roots_and_drain`) over
//! the live-table registry — the analog of RPython reaching the tracer
//! through the CLT keepalive. Convergence path: once
//! `asmmemmgr_gcreftracers` (or `CompiledLoopToken` itself) is GC-traced,
//! this collapses to a managed `GCREFTRACER` GcStruct + custom trace
//! hook, matching upstream exactly.
//!
//! One upstream duty does not carry over. `free_loop_and_bridges` calls
//! `clear_gcref_tracer`, which zeroes `array_length`, because upstream's
//! slot array is reserved inside the code block (`reserve_gcref_table`)
//! and is freed with it — a tracer outliving the block would otherwise
//! hand the collector freed memory. Here the slots are the table's own
//! `Box`, allocated and released with it, so the array cannot outlive its
//! storage and there is nothing to turn off.

use parking_lot::RwLock;
use std::cell::Cell;
use std::sync::{Arc, Weak};

use majit_ir::GcRef;

/// `llsupport/symbolic.py` `WORD`.
const WORD: usize = core::mem::size_of::<usize>();

/// A per-loop array of reference-constant slots.
///
/// `gcreftracer.py` `GCREFTRACER` stores `array_base_addr` +
/// `array_length`. The array itself sits at the start of the loop's
/// machine-code block (`assembler.py reserve_gcref_table`), where the
/// `LoadFromGcTable` genop reaches it PC-relative; a backend that keeps the
/// array on the Rust heap instead ([`from_gcrefs`](GcTable::from_gcrefs))
/// hands out the heap address as `array_base_addr` and bakes it as an
/// absolute immediate. Either address is stable for the table's whole life.
pub struct GcTable {
    array_base_addr: usize,
    array_length: usize,
    /// The heap-owned array, when the slots do not live in a code block.
    _owned: Option<Box<[Cell<GcRef>]>>,
    /// Compile-time ConstPtr identities. `trace` updates slots in place;
    /// leftover failarg ConstPtrs still name these originals.
    compile_keys: Box<[usize]>,
}

// SAFETY: slots are written only by `trace`, which runs inside a
// collection step (`do_collect_nursery` / major). That step is exclusive
// on the collecting MiniMark: `gc_op` holds the GIL, and
// `quiesce_mutators` pauses other registered mutators when
// `stw_required()`. Incremental major steps resume mutators in between;
// those mutators only read slots (`LoadFromGcTable`). Construction fills
// the slots before the `Arc` is shared. The `Cell` is the in-place
// forward; `Send`/`Sync` let the `Arc` live on
// `CompiledLoopToken.asmmemmgr_gcreftracers` and a `Weak` in the
// registry.
unsafe impl Send for GcTable {}
unsafe impl Sync for GcTable {}

/// Live per-loop tables, walked as GC roots. A strong reference is held by
/// `CompiledLoopToken.asmmemmgr_gcreftracers` (parity
/// `gcreftracers.append(tracer)`, `x86/assembler.py`); the registry keeps
/// only a `Weak`, and a `Weak` that stops upgrading is the whole of
/// deregistration. `JitCellToken::drop` calls
/// `CompiledLoopToken::free_loop_and_bridges` (`llmodel.py`), which
/// clears `asmmemmgr_gcreftracers`; `CompiledLoopToken::drop` then
/// bumps `total_freed_*`. The memory manager retiring a token from
/// `alive_loops` is what starts that drop.
///
/// That drop is not always the CLT's. On cranelift a bridge's table is
/// pinned a second time by every `BridgeData` that can dispatch to the
/// bridge, so it outlives the token it was registered against for as long
/// as a fail descr in another token holds one. The table dies with its
/// last strong holder, whichever that is; on dynasm the CLT is the only
/// one.
static LIVE_GC_TABLES: RwLock<Vec<Weak<GcTable>>> = RwLock::new(Vec::new());

/// Test-only lock serializing a registry drop/observe window against a
/// concurrent extra-root walk. `LIVE_GC_TABLES` is process-global, and the
/// harness runs MiniMark tests in parallel, so a walk can upgrade a `Weak`
/// a registry test is dropping. A registry test takes the write side;
/// each walk takes the read side. Compiled out in production: table drop
/// vs walk is serialized by the collection-step exclusion (`gc_op` / GIL),
/// not by a world-stop.
#[cfg(test)]
static GC_TABLE_WALK_LOCK: RwLock<()> = RwLock::new(());

impl GcTable {
    /// Build a per-loop table from the rewrite's gcref output list and
    /// register it for GC forwarding.
    ///
    /// `gcreftracer.py` `make_framework_tracer` warns that the
    /// tracer allocation can itself trigger a GC, so it writes the
    /// gcrefs into the raw array only afterwards. Here the
    /// `Box<[Cell<GcRef>]>` is a plain Rust heap allocation that does not
    /// go through the moving GC, so that hazard cannot arise; the slots
    /// are filled before the `Arc` exists, and registration happens last,
    /// so no collection can observe a half-filled, registered table.
    pub fn from_gcrefs(gcrefs: &[GcRef]) -> Arc<GcTable> {
        // Ensure the forwarding walker is installed before the table can
        // be observed by a collection. `register_extra_root_walker`
        // dedups by fn address, so this is idempotent across every
        // compiled loop; before the first table exists there is nothing
        // to walk, so installing lazily here is equivalent to a one-time
        // backend init without depending on a separate init call site.
        install_gc_table_walker();
        let slots: Box<[Cell<GcRef>]> = gcrefs.iter().map(|g| Cell::new(*g)).collect();
        let table = GcTable {
            array_base_addr: slots.as_ptr() as usize,
            array_length: slots.len(),
            _owned: Some(slots),
            compile_keys: gcrefs.iter().map(|g| g.0).collect(),
        };
        Self::register(table)
    }

    /// `gcreftracer.py` `make_framework_tracer`: write the gcrefs into the
    /// raw array reserved at `array_base_addr` inside a machine-code block
    /// (`assembler.py patch_gcref_table`) and register the tracer. The
    /// block is written under the assembler-writing bracket
    /// (`rmmap.enter_assembler_writing`).
    ///
    /// # Safety
    /// `array_base_addr` must address `gcrefs.len()` word-aligned words
    /// inside a JIT mapping that outlives the returned table.
    #[cfg(not(target_arch = "wasm32"))]
    pub unsafe fn in_code(array_base_addr: usize, gcrefs: &[GcRef]) -> Arc<GcTable> {
        install_gc_table_walker();
        {
            let _writing = crate::rmmap::AssemblerWriting::enter();
            for (i, g) in gcrefs.iter().enumerate() {
                unsafe { *((array_base_addr + i * WORD) as *mut GcRef) = *g };
            }
        }
        Self::register(GcTable {
            array_base_addr,
            array_length: gcrefs.len(),
            _owned: None,
            compile_keys: gcrefs.iter().map(|g| g.0).collect(),
        })
    }

    fn register(table: GcTable) -> Arc<GcTable> {
        let table = Arc::new(table);
        register_table(&table);
        table
    }

    /// Raw base address of the slot array, baked by the backend genop as
    /// the `LoadFromGcTable` base immediate. `gcreftracer.py:9`
    /// `array_base_addr`.
    pub fn base_addr(&self) -> usize {
        self.array_base_addr
    }

    /// Number of reference-constant slots. `gcreftracer.py:10`
    /// `array_length`.
    pub fn len(&self) -> usize {
        self.array_length
    }

    pub fn is_empty(&self) -> bool {
        self.array_length == 0
    }

    /// Compile-time address of slot `i`, used as the ConstPtr lookup key.
    ///
    /// `trace` rewrites [`Self::slot`] to the forwarded address; a leftover
    /// failarg still names this original.
    pub fn compile_key(&self, i: usize) -> usize {
        self.compile_keys[i]
    }

    /// Read slot `i`.
    pub fn slot(&self, i: usize) -> GcRef {
        assert!(i < self.array_length);
        // SAFETY: `array_base_addr + i*WORD` is a slot of a live array
        // (see the struct invariant); slots are read outside a collection
        // and written only by `trace` inside one.
        unsafe { *((self.array_base_addr + i * WORD) as *const GcRef) }
    }

    /// Current slot values, after any collection that forwarded them.
    ///
    /// `assembler.py` `_allgcrefs` is a GC list, so `patch_gcref_table`
    /// writes the addresses the collector already updated. A Rust
    /// `Vec<GcRef>` captured at rewrite is not a root; copy from here
    /// after assemble instead of from that Vec.
    pub fn slots(&self) -> Vec<GcRef> {
        (0..self.array_length).map(|i| self.slot(i)).collect()
    }

    /// Forward every slot in place. `gcreftracer.py gcrefs_trace`
    /// `gcrefs_trace`: each `array_base_addr + i*WORD` slot is handed to
    /// the GC as a root; writing back through the visitor forwards the
    /// constant if the moving GC relocated the referenced object.
    pub fn trace(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        // `gcrefs_trace` brackets the walk with
        // `rmmap.enter_assembler_writing()`: an in-code array is part of a
        // JIT mapping.
        #[cfg(not(target_arch = "wasm32"))]
        let _writing = self
            ._owned
            .is_none()
            .then(crate::rmmap::AssemblerWriting::enter);
        for i in 0..self.array_length {
            let p = (self.array_base_addr + i * WORD) as *mut GcRef;
            // SAFETY: see `slot`; this runs inside a collection step, the
            // only writer.
            unsafe {
                let mut r = *p;
                visitor(&mut r);
                *p = r;
            }
        }
    }
}

/// Append a live table to the registry.
///
/// `assembler.py setup` / `gcreftracer.make_gcref_tracer` appends one
/// tracer per compiled piece. Sweep dead `Weak`s only when the vec is
/// about to reallocate — walking every live table on each register is
/// extra work the append does not do.
fn register_table(table: &Arc<GcTable>) {
    let mut guard = LIVE_GC_TABLES.write();
    if guard.len() == guard.capacity() {
        guard.retain(|w| w.strong_count() > 0);
    }
    guard.push(Arc::downgrade(table));
}

/// Forward the slots of every live per-loop table. Registered once via
/// [`install_gc_table_walker`]; fires at both the minor
/// (`do_collect_nursery`) and major
/// (`rescan_major_nonstack_roots_and_drain`) collection
/// phases.
fn walk_all_gc_tables(visitor: &mut dyn FnMut(&mut GcRef)) {
    // In test builds, hold the read side of the walk lock so a registry
    // test's drop/observe window (which takes the write side) is never
    // interleaved with a walk. Compiles out in production.
    #[cfg(test)]
    let _walk = GC_TABLE_WALK_LOCK.read();
    walk_all_gc_tables_inner(visitor);
}

/// The walk itself, without the test-only lock, so a registry test that
/// already holds the write side observes the registry without re-entering
/// the lock.
fn walk_all_gc_tables_inner(visitor: &mut dyn FnMut(&mut GcRef)) {
    // A minor reaches only the tables that MiniMark remembered
    // (`pending_gc_tables`, the `old_objects_pointing_to_young` analogue).
    // That drain lives on the collecting MiniMark, not here.
    if crate::shadow_stack::extra_root_walk_kind() == crate::shadow_stack::ExtraRootWalkKind::Minor
    {
        return;
    }
    // Snapshot the live tables under a read guard, then release the lock
    // before tracing (same snapshot-then-iterate discipline as
    // `walk_extra_roots`, `shadow_stack.rs`). Dead `Weak`s are
    // filtered by `upgrade`.
    let live: Vec<Arc<GcTable>> = {
        let guard = LIVE_GC_TABLES.read();
        guard.iter().filter_map(|w| w.upgrade()).collect()
    };
    for table in &live {
        table.trace(visitor);
    }
}

/// Extra-root-walker entry point. A plain `fn` so it can be deduped by
/// address in `register_extra_root_walker`.
fn gc_table_extra_root_walker(visitor: &mut dyn FnMut(&mut GcRef)) {
    walk_all_gc_tables(visitor);
}

/// Install the per-loop gc_table forwarding walker. Idempotent —
/// `register_extra_root_walker` dedups by fn address
/// (`shadow_stack.rs`). Call once at backend init.
pub fn install_gc_table_walker() {
    crate::shadow_stack::register_extra_root_walker(gc_table_extra_root_walker, "gc_table");
}

#[cfg(test)]
mod tests {
    use super::*;

    // `LIVE_GC_TABLES` is process-global. Production mutates it at compile
    // time and reads it inside a collection step; those are exclusive via
    // `gc_op` / GIL. The harness runs MiniMark tests in parallel, so the
    // write side of [`GC_TABLE_WALK_LOCK`] also excludes a sibling
    // collection whose walk would otherwise upgrade a `Weak` this test is
    // dropping.

    #[test]
    fn trace_forwards_slots_in_place() {
        let _serialize = GC_TABLE_WALK_LOCK.write();
        let table = GcTable::from_gcrefs(&[GcRef(0x1000), GcRef(0x2000)]);
        // A moving collection relocates 0x1000 -> 0x9000.
        table.trace(&mut |r| {
            if r.0 == 0x1000 {
                r.0 = 0x9000;
            }
        });
        assert_eq!(table.slot(0), GcRef(0x9000));
        assert_eq!(table.slot(1), GcRef(0x2000));
        assert_eq!(
            table.compile_key(0),
            0x1000,
            "ConstPtr lookup keys stay at the compile-time address"
        );
        assert_eq!(table.compile_key(1), 0x2000);
        assert_eq!(
            table.base_addr(),
            table._owned.as_ref().unwrap().as_ptr() as usize
        );
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn baked_load_reads_forwarded_ref_after_move() {
        // End-to-end model of a ref constant surviving across compilation and
        // later execution. The backend bakes a `LoadFromGcTable` as a load of
        // `*(base + index*WORD)` (rewrite.py `remove_constptr`), with
        // `base` fixed at compile time. A moving collection forwards the slot
        // value in place (gcreftracer.py `trace`), so a later execution of the
        // baked load observes the relocated address — no stale immediate. This
        // is the shared dynasm/cranelift/wasm `LoadFromGcTable` contract.
        // wasm `compile_loop` / `compile_bridge` run `rewrite_ops_for_gc`
        // (`rewrite.py` `remove_constptr`) and bind the table via
        // `bind_failarg_const_table`.
        let _serialize = GC_TABLE_WALK_LOCK.write();
        let table = GcTable::from_gcrefs(&[GcRef(0x1000), GcRef(0x2000)]);
        // `base` is the value baked into the trace at compile time.
        let base = table.base_addr();
        // `GcRef` is `#[repr(transparent)]` over `usize` and `Cell<GcRef>`
        // shares that layout, so `base + index*WORD` addresses `slots[index]`
        // exactly as the emitted machine load does.
        let baked_load = |index: usize| -> usize {
            // SAFETY: `index < len`; the slot is a live `Cell<GcRef>` (usize).
            unsafe { *((base + index * core::mem::size_of::<usize>()) as *const usize) }
        };
        assert_eq!(
            baked_load(0),
            0x1000,
            "pre-move load reads the original ref"
        );
        assert_eq!(baked_load(1), 0x2000);
        // A moving collection relocates 0x1000 -> 0x9000.
        table.trace(&mut |r| {
            if r.0 == 0x1000 {
                r.0 = 0x9000;
            }
        });
        assert_eq!(
            table.base_addr(),
            base,
            "baked base address is stable across the collection"
        );
        assert_eq!(
            baked_load(0),
            0x9000,
            "post-move load via the baked base reads the forwarded ref"
        );
        assert_eq!(baked_load(1), 0x2000, "unmoved ref unchanged");
    }

    #[test]
    fn dropping_table_deregisters_from_walk() {
        let _serialize = GC_TABLE_WALK_LOCK.write();
        // A sentinel unlikely to collide with any other test's table.
        const SENTINEL: GcRef = GcRef(0x0000_DEAD_BEEF);
        // The write side is already held, so count through the unlocked
        // walk to avoid re-entering the lock.
        let count_sentinels = || {
            let mut n = 0usize;
            walk_all_gc_tables_inner(&mut |r| {
                if *r == SENTINEL {
                    n += 1;
                }
            });
            n
        };
        {
            let _table = GcTable::from_gcrefs(&[SENTINEL]);
            assert_eq!(count_sentinels(), 1, "live table must be walked");
        }
        // The Arc dropped; the registry's Weak no longer upgrades.
        assert_eq!(count_sentinels(), 0, "freed table must not be walked");
    }

    #[test]
    fn pending_gc_tables_stay_on_the_remembering_minimark() {
        let mut owner = crate::collector::MiniMarkGC::with_config(crate::collector::GcConfig {
            nursery_size: 65536,
            large_object_threshold: 1024,
            ..crate::collector::GcConfig::default()
        });
        let type_id = owner.register_type(crate::TypeInfo::simple(16));
        let root = owner.alloc_with_type(type_id, 16);
        unsafe { *(root.0 as *mut u64) = 0xA11C_E700 };
        let table = GcTable::from_gcrefs(&[root]);
        owner.remember_gc_table(&table);

        let mut sibling = crate::collector::MiniMarkGC::with_config(crate::collector::GcConfig {
            nursery_size: 65536,
            large_object_threshold: 1024,
            ..crate::collector::GcConfig::default()
        });
        sibling.do_collect_nursery();
        assert_eq!(
            table.slot(0),
            root,
            "a sibling MiniMark must not drain this collector's table"
        );

        owner.do_collect_nursery();
        let moved = table.slot(0);
        assert_ne!(moved, root, "the owning MiniMark forwards the slot");
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xA11C_E700);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn pinned_rewrite_gcrefs_survive_minor_before_in_code() {
        // `assembler.py` `_allgcrefs` is a GC list: a collection between
        // rewrite and `patch_gcref_table` forwards the entries, and
        // `make_framework_tracer` writes those forwarded addresses.
        // Parking the rewrite Vec in a heap table with a write barrier
        // is that list.
        let mut owner = crate::collector::MiniMarkGC::with_config(crate::collector::GcConfig {
            nursery_size: 65536,
            large_object_threshold: 1024,
            ..crate::collector::GcConfig::default()
        });
        let type_id = owner.register_type(crate::TypeInfo::simple(16));
        let root = owner.alloc_with_type(type_id, 16);
        unsafe { *(root.0 as *mut u64) = 0xA11C_E700 };
        let pinned = GcTable::from_gcrefs(&[root]);
        owner.remember_gc_table(&pinned);
        owner.do_collect_nursery();
        let forwarded = pinned.slots();
        assert_eq!(forwarded.len(), 1);
        assert_ne!(forwarded[0], root, "pin table observes the move");
        assert_eq!(unsafe { *(forwarded[0].0 as *const u64) }, 0xA11C_E700);

        let mut buf = vec![GcRef(0); 1];
        let _table = unsafe { GcTable::in_code(buf.as_mut_ptr() as usize, &forwarded) };
        assert_eq!(
            buf[0], forwarded[0],
            "in_code copies the forwarded address, not the rewrite-time one"
        );
    }
}
