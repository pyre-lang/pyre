//! x86-specific per-CPU assembler state held by `DynasmBackend`.
//!
//! PyPy stores `self.malloc_slowpath` / `self.propagate_exception_path`
//! on `Assembler386` (`rpython/jit/backend/x86/assembler.py`);
//! the assembler is one-per-CPU and lives for the CPU's lifetime, so
//! the trampolines built at `setup_once` (`llsupport/assembler.py`)
//! persist on it.
//!
//! Pyre's `Assembler386` is constructed per-`compile_loop`/`compile_bridge`
//! (`runner.rs::compile_loop`, `compile_bridge`), so the per-CPU stash
//! moves up one level to `DynasmBackend` via this struct.  Aarch64 has
//! its own equivalent (`aarch64::cpu_ext::Aarch64CpuExt`) which is
//! currently a no-op placeholder — aarch64 inlines the slowpath
//! sequences today and has no per-CPU trampoline to memoise.

use crate::codebuf::ArenaExecutableBuffer;
use crate::guard::CpuDescrHandle;
use crate::x86::assembler::MallocSlowpathKind;
use majit_backend::AsmMemoryManager;
use std::sync::Arc;

/// Lazily-materialised per-CPU x86 trampolines.
///
/// Both addresses are set once on first use and reused for every
/// subsequent `compile_loop` / `compile_bridge` on this CPU.  The
/// owning arena buffers are kept alongside so their ranges stay live for the
/// CPU and return to its reusable free list on drop, matching PyPy's
/// `asmmemmgr` ownership.
pub(crate) struct X86CpuExt {
    asm_memory_manager: Arc<AsmMemoryManager>,
    /// `assembler.py:63 self.malloc_slowpath` parity.  Entry address
    /// of the shared malloc slowpath trampoline built by
    /// `build_malloc_slowpath_fixed`.  PyPy's `malloc_cond` (line
    /// 2554) and `malloc_cond_varsize_frame` (line 2578) both route
    /// through this single helper; pyre follows that — both call
    /// sites reach this address via the same `JA slow_path` jump and
    /// the trampoline's `SUB rdx, rcx` recovers the byte count.
    /// `_buffer` is the matching RX mapping kept for the lifetime of
    /// this struct.
    malloc_slowpath_fixed: Option<usize>,
    _malloc_slowpath_fixed_buffer: Option<ArenaExecutableBuffer>,
    malloc_slowpath_headerless: Option<usize>,
    _malloc_slowpath_headerless_buffer: Option<ArenaExecutableBuffer>,
    /// `_build_malloc_slowpath(kind='var')`. `malloc_cond_varsize` calls
    /// this address. Built once per CPU, same as `malloc_slowpath_fixed`.
    malloc_slowpath_varsize: Option<usize>,
    _malloc_slowpath_varsize_buffer: Option<ArenaExecutableBuffer>,
    /// `_build_malloc_slowpath(kind='str')` / `(kind='unicode')`.
    malloc_slowpath_str: Option<usize>,
    _malloc_slowpath_str_buffer: Option<ArenaExecutableBuffer>,
    malloc_slowpath_unicode: Option<usize>,
    _malloc_slowpath_unicode_buffer: Option<ArenaExecutableBuffer>,
    /// `assembler.py:344 self.propagate_exception_path` parity.
    /// Standalone trampoline that the malloc slowpath JMPs to on OOM and
    /// the stack-check helper JMPs to on overflow.
    propagate_exception_path: Option<usize>,
    _propagate_exception_path_buffer: Option<ArenaExecutableBuffer>,
    /// `build_frame_realloc_slowpath`: once per CPU. The per-bridge
    /// `IncreaseStackSlowPath` calls this address.
    frame_realloc_slowpath: Option<usize>,
    _frame_realloc_slowpath_buffer: Option<ArenaExecutableBuffer>,
    /// `_build_stack_check_slowpath`. `None` until `stack_check_addresses` is registered and
    /// `propagate_exception_descr` is installed; a failed attempt is not
    /// cached, so the next `ensure_stack_check_slowpath` retries.
    stack_check_slowpath: Option<usize>,
    _stack_check_slowpath_buffer: Option<ArenaExecutableBuffer>,
    /// `assembler.py self.wb_slowpath` parity: entries `0..4` indexed by
    /// `withcards + 2 * withfloats`, entry `4` the `for_frame` helper, `0`
    /// where no helper is built yet.
    wb_slowpath: [usize; 5],
    _wb_slowpath_buffers: Vec<ArenaExecutableBuffer>,
    /// `Assembler386.cond_call_slowpath`: four `_build_cond_call_slowpath`
    /// entries, index `floats * 2 + callee_only`. `0` until
    /// `ensure_cond_call_slowpath`. Not a propagate-dependent cache: the
    /// helper does not bake `propagate_exception_descr`.
    cond_call_slowpath: [usize; 4],
    _cond_call_slowpath_buffers: Vec<ArenaExecutableBuffer>,
    /// `Assembler386.failure_recovery_code`: four `_build_failure_recovery`
    /// entries, index `exc + 2 * withfloats`. `0` until
    /// `ensure_failure_recovery`. The helper does not bake a descr pointer.
    failure_recovery_code: [usize; 4],
    _failure_recovery_buffers: Vec<ArenaExecutableBuffer>,
}

impl X86CpuExt {
    pub(crate) fn new(asm_memory_manager: Arc<AsmMemoryManager>) -> Self {
        Self {
            asm_memory_manager,
            malloc_slowpath_fixed: None,
            _malloc_slowpath_fixed_buffer: None,
            malloc_slowpath_headerless: None,
            _malloc_slowpath_headerless_buffer: None,
            malloc_slowpath_varsize: None,
            _malloc_slowpath_varsize_buffer: None,
            malloc_slowpath_str: None,
            _malloc_slowpath_str_buffer: None,
            malloc_slowpath_unicode: None,
            _malloc_slowpath_unicode_buffer: None,
            propagate_exception_path: None,
            _propagate_exception_path_buffer: None,
            frame_realloc_slowpath: None,
            _frame_realloc_slowpath_buffer: None,
            stack_check_slowpath: None,
            _stack_check_slowpath_buffer: None,
            wb_slowpath: [0; 5],
            _wb_slowpath_buffers: Vec::new(),
            cond_call_slowpath: [0; 4],
            _cond_call_slowpath_buffers: Vec::new(),
            failure_recovery_code: [0; 4],
            _failure_recovery_buffers: Vec::new(),
        }
    }

    /// `build_frame_realloc_slowpath`, memoised once per CPU.
    pub(crate) fn ensure_frame_realloc_slowpath(&mut self) -> usize {
        if let Some(addr) = self.frame_realloc_slowpath {
            return addr;
        }
        let (buffer, addr) =
            super::assembler::build_frame_realloc_slowpath(&self.asm_memory_manager);
        debug_assert!(
            addr != 0,
            "build_frame_realloc_slowpath returned a null entry"
        );
        self._frame_realloc_slowpath_buffer = Some(buffer);
        self.frame_realloc_slowpath = Some(addr);
        addr
    }

    /// `_build_stack_check_slowpath`. Returns 0 while the three
    /// `insert_stack_check` addresses or `propagate_exception_descr` are
    /// missing; a failed attempt is not cached, so the next compile retries.
    pub(crate) fn ensure_stack_check_slowpath(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if let Some(addr) = self.stack_check_slowpath {
            return addr;
        }
        let Some(addrs) = crate::stack_check_addresses() else {
            return 0;
        };
        if addrs.slowpath_addr == 0 {
            return 0;
        }
        if cpu_handle.read().descr_ptrs().propagate_exception_descr == 0 {
            return 0;
        }
        let propagate_exception_path = self.ensure_propagate_exception_path(cpu_handle);
        let (buffer, addr) = super::assembler::build_stack_check_slowpath(
            addrs.slowpath_addr,
            propagate_exception_path,
            &self.asm_memory_manager,
        );
        debug_assert!(
            addr != 0,
            "build_stack_check_slowpath returned a null entry"
        );
        self._stack_check_slowpath_buffer = Some(buffer);
        self.stack_check_slowpath = Some(addr);
        addr
    }

    /// `llsupport/assembler.py setup_once` parity: build every
    /// `_build_wb_slowpath` variant not built yet and memoise the entries, which
    /// `_write_barrier_fastpath` then `CALL`s as `wb_slowpath[helper_num]`.
    /// Runs from `setup_once` and from `set_gc_allocator`, not per compile.
    pub(crate) fn build_wb_slowpaths(&mut self) {
        // `_build_wb_slowpath(False)`, `(True)`, `(False, for_frame=True)`,
        // then the `withfloats=True` pair.
        for (withcards, withfloats, for_frame) in [
            (false, false, false),
            (true, false, false),
            (false, false, true),
            (false, true, false),
            (true, true, false),
        ] {
            let helper_num = if for_frame {
                4
            } else {
                usize::from(withcards) + 2 * usize::from(withfloats)
            };
            // An entry built by an earlier call is kept; one left at 0 (no
            // `withcards` helper for the descr seen then) is built when
            // `set_gc_allocator` installs a collector whose descr has one.
            if self.wb_slowpath[helper_num] != 0 {
                continue;
            }
            let Some((buffer, addr)) = super::assembler::build_wb_slowpath(
                withcards,
                withfloats,
                for_frame,
                &self.asm_memory_manager,
            ) else {
                continue;
            };
            self.wb_slowpath[helper_num] = addr;
            self._wb_slowpath_buffers.push(buffer);
        }
    }

    /// The entries [`Self::build_wb_slowpaths`] built, which every compile
    /// hands to its assembler.
    pub(crate) fn wb_slowpath(&self) -> [usize; 5] {
        self.wb_slowpath
    }

    /// `_build_cond_call_slowpath` for all four `(supports_floats, callee_only)`
    /// pairs, memoised once per CPU. `build_wb_slowpaths` runs first:
    /// `reload_frame_if_necessary` inside the helper calls `wb_slowpath[4]`.
    pub(crate) fn ensure_cond_call_slowpath(&mut self) -> [usize; 4] {
        if self.cond_call_slowpath[0] != 0 {
            return self.cond_call_slowpath;
        }
        self.build_wb_slowpaths();
        let wb_slowpath = self.wb_slowpath;
        let (buffers, addrs) =
            super::assembler::build_cond_call_slowpaths(wb_slowpath, &self.asm_memory_manager);
        assert!(
            addrs.iter().all(|addr| *addr != 0),
            "build_cond_call_slowpaths returned a null entry"
        );
        self._cond_call_slowpath_buffers = buffers;
        self.cond_call_slowpath = addrs;
        addrs
    }

    /// `_build_failure_recovery` for `(exc, withfloats)` in index order
    /// `exc + 2 * withfloats`, memoised once per CPU. No propagate or
    /// write-barrier dependency: the body only reads the process-wide
    /// exception cells.
    pub(crate) fn ensure_failure_recovery(&mut self) -> [usize; 4] {
        if self.failure_recovery_code[0] != 0 {
            return self.failure_recovery_code;
        }
        let (buffers, addrs) = super::assembler::build_failure_recoveries(&self.asm_memory_manager);
        assert!(
            addrs.iter().all(|addr| *addr != 0),
            "build_failure_recoveries returned a null entry"
        );
        self._failure_recovery_buffers = buffers;
        self.failure_recovery_code = addrs;
        addrs
    }

    /// `assembler.py:328 _build_propagate_exception_path` parity:
    /// materialise the standalone propagate trampoline that
    /// `_store_and_reset_exception`s, writes `jf_guard_exc` / `jf_descr`,
    /// and tail-calls `_call_footer`.  The malloc slowpath JMPs to this
    /// single entry.  Materialised lazily; the address is then memoised
    /// here so every slowpath built on this CPU shares the same propagate
    /// path (matches PyPy's `self.propagate_exception_path` attribute).
    pub(crate) fn ensure_propagate_exception_path(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if let Some(addr) = self.propagate_exception_path {
            return addr;
        }
        let (buffer, addr) =
            super::assembler::build_propagate_exception_path(cpu_handle, &self.asm_memory_manager);
        debug_assert!(
            addr != 0,
            "build_propagate_exception_path returned NULL entry address — \
             dynasm finalize is expected to yield a non-zero buffer_ptr"
        );
        self._propagate_exception_path_buffer = Some(buffer);
        self.propagate_exception_path = Some(addr);
        addr
    }

    /// `assembler.py:231 _build_malloc_slowpath` parity: materialise
    /// the fixed-size malloc slowpath helper on first use and stash
    /// its address here.  Subsequent `compile_loop` / `compile_bridge`
    /// invocations reuse the same helper, matching PyPy's
    /// `setup_once` semantics where the helper is built once per CPU
    /// and referenced as `self.malloc_slowpath` thereafter.
    ///
    /// Ensures the propagate trampoline exists first so the slowpath's
    /// OOM branch can `JMP` to it (matches PyPy's `setup_once` ordering:
    /// `_build_propagate_exception_path` then `_build_malloc_slowpath`).
    pub(crate) fn ensure_malloc_slowpath_fixed(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if self.malloc_slowpath_fixed.is_none() {
            let propagate_path = self.ensure_propagate_exception_path(cpu_handle);
            let (buffer, addr) = super::assembler::build_malloc_slowpath_fixed(
                cpu_handle,
                propagate_path,
                &self.asm_memory_manager,
            );
            debug_assert!(
                addr != 0,
                "build_malloc_slowpath_fixed returned NULL entry address — \
                 dynasm finalize is expected to yield a non-zero buffer_ptr"
            );
            self._malloc_slowpath_fixed_buffer = Some(buffer);
            self.malloc_slowpath_fixed = Some(addr);
        }
        self.malloc_slowpath_fixed
            .expect("malloc_slowpath_fixed was just ensured")
    }

    pub(crate) fn ensure_malloc_slowpath_headerless(
        &mut self,
        cpu_handle: &CpuDescrHandle,
    ) -> usize {
        if self.malloc_slowpath_headerless.is_none() {
            let propagate_path = self.ensure_propagate_exception_path(cpu_handle);
            let (buffer, addr) = super::assembler::build_malloc_slowpath_headerless(
                cpu_handle,
                propagate_path,
                &self.asm_memory_manager,
            );
            debug_assert!(
                addr != 0,
                "build_malloc_slowpath_headerless returned NULL entry address — \
                 dynasm finalize is expected to yield a non-zero buffer_ptr"
            );
            self._malloc_slowpath_headerless_buffer = Some(buffer);
            self.malloc_slowpath_headerless = Some(addr);
        }
        self.malloc_slowpath_headerless
            .expect("malloc_slowpath_headerless was just ensured")
    }

    /// `_build_malloc_slowpath(kind='var')`: one trampoline per CPU.
    /// `ensure_propagate_exception_path` runs first; the OOM tail JMPs there.
    pub(crate) fn ensure_malloc_slowpath_varsize(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        self.ensure_malloc_slowpath_kind(cpu_handle, MallocSlowpathKind::Var)
    }

    /// `_build_malloc_slowpath(kind='str')`.
    pub(crate) fn ensure_malloc_slowpath_str(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        self.ensure_malloc_slowpath_kind(cpu_handle, MallocSlowpathKind::Str)
    }

    /// `_build_malloc_slowpath(kind='unicode')`.
    pub(crate) fn ensure_malloc_slowpath_unicode(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        self.ensure_malloc_slowpath_kind(cpu_handle, MallocSlowpathKind::Unicode)
    }

    fn ensure_malloc_slowpath_kind(
        &mut self,
        cpu_handle: &CpuDescrHandle,
        kind: MallocSlowpathKind,
    ) -> usize {
        let cached = match kind {
            MallocSlowpathKind::Var => self.malloc_slowpath_varsize,
            MallocSlowpathKind::Str => self.malloc_slowpath_str,
            MallocSlowpathKind::Unicode => self.malloc_slowpath_unicode,
            MallocSlowpathKind::Fixed => unreachable!("'fixed' is ensure_malloc_slowpath_fixed"),
        };
        if let Some(addr) = cached {
            return addr;
        }
        let propagate_path = self.ensure_propagate_exception_path(cpu_handle);
        let (buffer, addr) = super::assembler::build_malloc_slowpath_varsize(
            cpu_handle,
            kind,
            propagate_path,
            &self.asm_memory_manager,
        );
        debug_assert!(
            addr != 0,
            "build_malloc_slowpath_varsize returned NULL entry address —              dynasm finalize is expected to yield a non-zero buffer_ptr"
        );
        let (slot, buffer_slot) = match kind {
            MallocSlowpathKind::Var => (
                &mut self.malloc_slowpath_varsize,
                &mut self._malloc_slowpath_varsize_buffer,
            ),
            MallocSlowpathKind::Str => (
                &mut self.malloc_slowpath_str,
                &mut self._malloc_slowpath_str_buffer,
            ),
            MallocSlowpathKind::Unicode => (
                &mut self.malloc_slowpath_unicode,
                &mut self._malloc_slowpath_unicode_buffer,
            ),
            MallocSlowpathKind::Fixed => unreachable!(),
        };
        *buffer_slot = Some(buffer);
        *slot = Some(addr);
        addr
    }

    /// Whether either trampoline that bakes `propagate_exception_descr`
    /// as an immediate has already been materialised.  Used by
    /// `DynasmBackend::set_propagate_exception_descr` to refuse a
    /// non-identical `Arc` swap after the bake: such a swap would
    /// leave previously-compiled loops/bridges (whose `JMP` immediates
    /// point at this buffer's RX pages and whose helpers carry the
    /// old descr pointer) referencing a now-orphaned descr.  PyPy
    /// attaches `propagate_exception_descr` once before
    /// `cpu.setup_once()` and never replaces it
    /// (`pyjitpl.py` precedes `pyjitpl.py _setup_once`); pyre
    /// upholds the same invariant by panicking instead of dropping
    /// the buffer.
    pub(crate) fn has_propagate_dependent_caches(&self) -> bool {
        self.malloc_slowpath_fixed.is_some()
            || self.malloc_slowpath_headerless.is_some()
            || self.malloc_slowpath_varsize.is_some()
            || self.malloc_slowpath_str.is_some()
            || self.malloc_slowpath_unicode.is_some()
            || self.propagate_exception_path.is_some()
            || self.stack_check_slowpath.is_some()
    }
}
