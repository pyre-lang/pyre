//! Literal port of `rpython/jit/metainterp/opencoder.py` — the byte-
//! stream trace recorder + iterator + snapshot chain reader used by
//! the meta-interpreter.
use indexmap::IndexMap;
use majit_ir::operand::Operand;
use majit_ir::{InputArg, OPCODE_COUNT, OpCode, OpRc, OpRef, Type, Value};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};

#[path = "trace_bufs.rs"]
mod trace_bufs;

/// `GcArray(Char)` length word. `opencoder.py` `FixedSizeListRepr` stores
/// the list as that one array; the length the collector copies is this word.
const TRACE_OPS_CHARS: usize = std::mem::size_of::<usize>();

/// Unregistered sentinel. A real `register_type` id is a table index.
/// These ids index the collector that last called
/// [`register_trace_ops_gc_type`]. Production is one `gc_sync` singleton;
/// tests install one box at a time. Two live `install_gc_box` collectors
/// at once is not a supported configuration: a second register overwrites
/// the process-global id and later allocs on the first collector would
/// use the wrong type table.
static TRACE_OPS_GC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);
static NUMBERING_GC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);

/// Register `Trace._ops`' `GcArray(Char)` (`TypeInfo::varsize`, item size 1,
/// no GC pointers, no destructor). Call once, beside the JITFRAME
/// registration, before the type table freezes.
pub fn register_trace_ops_gc_type(gc: &mut dyn majit_gc::GcAllocator) -> u32 {
    let id = gc.register_type(majit_gc::trace::TypeInfo::varsize(
        TRACE_OPS_CHARS,
        1,
        0,
        false,
        Vec::new(),
    ));
    TRACE_OPS_GC_TYPE_ID.store(id, Ordering::Release);
    // `resumecode.py` `NUMBERING`: length word plus `Array(UCHAR)`.
    let numb_id = gc.register_type(majit_gc::trace::TypeInfo::varsize(
        TRACE_OPS_CHARS,
        1,
        0,
        false,
        Vec::new(),
    ));
    NUMBERING_GC_TYPE_ID.store(numb_id, Ordering::Release);
    majit_ir::resumecode::set_numbering_alloc(Some(alloc_numbering_bytes));
    // List header, word arrays, and `Trace._refs`. Same registration pass.
    trace_bufs::register_trace_pool_gc_types(gc);
    id
}

fn numbering_gc_type_id() -> u32 {
    let id = NUMBERING_GC_TYPE_ID.load(Ordering::Acquire);
    assert!(
        id != u32::MAX,
        "NUMBERING GcArray is not registered; call register_trace_ops_gc_type"
    );
    id
}

/// `lltype.malloc(NUMBERING, len)` via the collecting nursery entry.
fn alloc_numbering_bytes(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let payload = TRACE_OPS_CHARS + len;
    let mut live = majit_ir::GcRef(0);
    let mut needs_write_barrier = false;
    majit_gc::stress_trace_pool_alloc(&mut live, 1);
    let fresh = unsafe {
        majit_gc::alloc_fast_nursery_collecting_typed_rooted(
            numbering_gc_type_id(),
            payload,
            &mut live,
            &mut needs_write_barrier,
        )
    };
    if fresh.0 == 0 {
        majit_gc::gc_alloc_failed(payload);
    }
    unsafe {
        *(fresh.0 as *mut usize) = len;
        if len > 0 {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (fresh.0 as *mut u8).add(TRACE_OPS_CHARS),
                len,
            );
        }
    }
    let _ = needs_write_barrier;
    let _ = live;
    fresh.0
}

fn trace_ops_gc_type_id() -> u32 {
    let id = TRACE_OPS_GC_TYPE_ID.load(Ordering::Acquire);
    assert!(
        id != u32::MAX,
        "Trace._ops GcArray is not registered; call register_trace_ops_gc_type"
    );
    id
}

/// Host holder for `opencoder.py` `Trace._ops`.
///
/// The array is one nursery `GcArray(Char)` rooted by `OwnerRootGuard`.
/// Every access re-reads that guard. A raw `*mut u8` is never kept across
/// an allocation. When this thread has no collector (`current_thread_gc_available`
/// is false — bare unit tests, or a thread that never installed a box /
/// singleton), the same length-word-plus-chars block is a process
/// allocation. A process-global alloc hook (`gc_allocator_installed`) is
/// not enough: the dynasm trampoline would then call `standalone_alloc_*`
/// and panic on an uninitialized singleton.
struct TraceOpsBuf {
    root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    /// Non-null only on the no-GC arm. Freed in `Drop`.
    host: *mut u8,
    /// Char capacity. `len(_ops)` in `opencoder.py`. Not `_pos`.
    len: usize,
}

impl TraceOpsBuf {
    fn new(len: usize) -> Self {
        if majit_gc::current_thread_gc_available() {
            Self::alloc_gc(len, majit_ir::GcRef(0), 0)
        } else {
            Self::alloc_host(len, std::ptr::null(), 0)
        }
    }

    fn alloc_host(len: usize, src: *const u8, src_len: usize) -> Self {
        let payload = TRACE_OPS_CHARS + len;
        let layout = std::alloc::Layout::from_size_align(payload, std::mem::align_of::<usize>())
            .unwrap_or_else(|_| std::alloc::Layout::new::<usize>());
        let host = unsafe { std::alloc::alloc_zeroed(layout) };
        if host.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe {
            *(host as *mut usize) = len;
            if !src.is_null() && src_len > 0 {
                std::ptr::copy_nonoverlapping(src, host.add(TRACE_OPS_CHARS), src_len);
            }
        }
        Self {
            root: None,
            host,
            len,
        }
    }

    fn alloc_gc(len: usize, mut live: majit_ir::GcRef, src_len: usize) -> Self {
        let payload = TRACE_OPS_CHARS + len;
        let mut needs_write_barrier = false;
        // `malloc_varsize`: the nursery arm for everything up to
        // `nonlarge_max`, and `external_malloc` only past it. That test lives
        // in `alloc_fast_nursery_collecting_typed_rooted`
        // (`alloc_with_type_rooted_body`), not at this caller.
        // `malloc_fast` + one live root. The public no-collect entry spills
        // instead of collecting; `Trace._double_ops` has to collect.
        majit_gc::stress_trace_pool_alloc(&mut live, 1);
        let fresh = unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                trace_ops_gc_type_id(),
                payload,
                &mut live,
                &mut needs_write_barrier,
            )
        };
        if fresh.0 == 0 {
            majit_gc::gc_alloc_failed(payload);
        }
        // The collector sizes a varsize object from this word. Write it
        // before the new array is rooted and before the next allocation.
        unsafe {
            *(fresh.0 as *mut usize) = len;
            let dst = (fresh.0 as *mut u8).add(TRACE_OPS_CHARS);
            if live.0 != 0 && src_len > 0 {
                let src = (live.0 as *const u8).add(TRACE_OPS_CHARS);
                std::ptr::copy_nonoverlapping(src, dst, src_len);
            }
            if len > src_len {
                std::ptr::write_bytes(dst.add(src_len), 0, len - src_len);
            }
        }
        let _ = needs_write_barrier;
        Self {
            root: Some(majit_gc::shadow_stack::OwnerRootGuard::new(fresh)),
            host: std::ptr::null_mut(),
            len,
        }
    }

    /// `opencoder.py` `Trace._double_ops`: a new array, not a resize.
    fn double(&mut self) {
        let new_len = self.len.max(1).saturating_mul(2);
        let src = self.chars();
        let src_len = self.len;
        let next = if let Some(guard) = &self.root {
            Self::alloc_gc(new_len, guard.get(), src_len)
        } else {
            debug_assert!(
                !majit_gc::current_thread_gc_available(),
                "host Trace._ops buffer used while this thread has a collector"
            );
            Self::alloc_host(new_len, src, src_len)
        };
        *self = next;
    }

    fn addr(&self) -> *mut u8 {
        if let Some(guard) = &self.root {
            guard.get().0 as *mut u8
        } else {
            self.host
        }
    }

    fn chars(&self) -> *mut u8 {
        unsafe { self.addr().add(TRACE_OPS_CHARS) }
    }

    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.chars(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.chars(), self.len) }
    }
}

impl Drop for TraceOpsBuf {
    fn drop(&mut self) {
        if self.host.is_null() {
            return;
        }
        let payload = TRACE_OPS_CHARS + self.len;
        let layout = std::alloc::Layout::from_size_align(payload, std::mem::align_of::<usize>())
            .unwrap_or_else(|_| std::alloc::Layout::new::<usize>());
        unsafe { std::alloc::dealloc(self.host, layout) };
    }
}

#[allow(dead_code)]
fn u16_to_opcode(v: u16) -> OpCode {
    assert!(
        (v as usize) < OPCODE_COUNT,
        "invalid opcode discriminant: {v}"
    );
    // SAFETY: OpCode is #[repr(u16)] and we checked the discriminant is in range.
    unsafe { std::mem::transmute(v) }
}

// ── Signed varint encoding (opencoder.py encode_varint_signed) ──
//
// Literal port of rpython/jit/metainterp/opencoder.py encode_varint_signed
// / decode_varint_signed. Either 2 bytes or 4 bytes, chosen by range:
//
//   2 bytes: value in [-2^14, 2^14-1]  (bit 7 of byte 0 == 0)
//     byte 0: low 7 bits of value; bit 7 = 0
//     byte 1: bits 7..14 of value; bit 7 of byte 1 acts as sign bit
//
//   4 bytes: value in [MIN_VALUE, MAX_VALUE] = [-2^30, 2^30-1]
//     byte 0: low 7 bits of value; bit 7 = 1 (flag)
//     byte 1: bits 7..14 of value
//     byte 2: bits 15..22 of value
//     byte 3: bits 23..30 of value; bit 7 of byte 3 is sign bit
//
// Decoding sign-extends from bit (shift) using the top bit of the last
// written byte.
//
// This replaces the pre-Phase-B zigzag LEB128 encoder so the pyre wire
// format binary-matches RPython.

/// opencoder.py encode_varint_signed. Two bytes, or four when bit 7 of byte 0 is set.
pub fn encode_varint_signed_array(value: i64) -> ([u8; 4], usize) {
    debug_assert!(
        (MIN_VALUE..=MAX_VALUE).contains(&value),
        "encode_varint_signed out of range: {value}"
    );
    let mut out = [0u8; 4];
    let mut v = value;
    let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
        0x80
    } else {
        0
    };
    out[0] = ((v & 0b0111_1111) as u8) | flag;
    v >>= 7;
    out[1] = (v & 0xff) as u8;
    if flag == 0 {
        return (out, 2);
    }
    v >>= 8;
    out[2] = (v & 0xff) as u8;
    v >>= 8;
    out[3] = (v & 0xff) as u8;
    (out, 4)
}

/// opencoder.py encode_varint_signed.
pub fn encode_varint_signed(buf: &mut Vec<u8>, value: i64) {
    let (bytes, n) = encode_varint_signed_array(value);
    buf.extend_from_slice(&bytes[..n]);
}

/// opencoder.py decode_varint_signed. Returns (value, bytes_consumed).
#[inline(always)] // opencoder.py decode_varint_signed: @always_inline.
pub fn decode_varint_signed(buf: &[u8]) -> (i64, usize) {
    let byte0 = buf[0];
    let byte1 = buf[1];
    let mut res: i64 = ((byte0 & 0b0111_1111) as i64) | ((byte1 as i64) << 7);
    let mut shift: u32 = 15;
    let mut index: usize = 2;
    let mut lastbyte = byte1;
    if byte0 & 0b1000_0000 != 0 {
        let byte2 = buf[index];
        let byte3 = buf[index + 1];
        res |= ((byte2 as i64) << 15) | ((byte3 as i64) << 23);
        shift = 31;
        index += 2;
        lastbyte = byte3;
    }
    // sign-extend: top bit of the last written byte is the sign bit.
    res |= -i64::from(lastbyte & 0b1000_0000 != 0) << shift;
    (res, index)
}

// ── Snapshot management (opencoder.py) ──

/// Tag constants for snapshot encoding.
/// opencoder.py: TAGINT, TAGCONSTPTR, TAGCONSTOTHER, TAGBOX
pub const TAGINT: u8 = 0; // small integer constant
pub const TAGCONSTPTR: u8 = 1; // GC pointer constant
pub const TAGCONSTOTHER: u8 = 2; // big int or float constant
pub const TAGBOX: u8 = 3; // reference to a traced value (OpRef)

/// Number of tag bits.
pub const TAG_SHIFT: u32 = 2;
/// Mask for tag extraction.
pub const TAG_MASK: u8 = (1 << TAG_SHIFT) - 1;
/// RPython compatibility constants.
pub const TAGMASK: u8 = TAG_MASK;
pub const TAGSHIFT: u32 = TAG_SHIFT;

/// Initial trace buffer size.
pub const INIT_SIZE: usize = 4096;

/// RPython compatibility numeric limits for signed varints in this model.
pub const MIN_VALUE: i64 = -(1 << 30);
pub const MAX_VALUE: i64 = (1 << 30) - 1;

/// RPython compatibility bounds for small int tagging.
pub const SMALL_INT_START: i64 = -(1 << 28);
pub const SMALL_INT_STOP: i64 = 1 << 28;

/// RPython compatibility sentinel values for snapshot linked lists.
pub const SNAPSHOT_PREV_NEEDS_PATCHING: i32 = -3;
pub const SNAPSHOT_PREV_NONE: i32 = -2;
pub const SNAPSHOT_PREV_COMES_NEXT: i32 = -1;

/// opencoder.py skip_varint_signed.
///
/// Each varint is 2 bytes if bit 7 of byte 0 is clear, 4 bytes if set.
/// `skip` is the number of varints to skip over; returns the new cursor.
#[inline]
pub fn skip_varint_signed(buf: &[u8], mut index: usize, skip: usize) -> usize {
    assert!(skip > 0);
    let mut remaining = skip;
    loop {
        let byte = buf[index];
        if byte & 0b1000_0000 != 0 {
            index += 2;
        }
        index += 2;
        remaining -= 1;
        if remaining == 0 {
            return index;
        }
    }
}

#[inline]
pub fn varint_only_decode(buf: &[u8], index: usize, skip: usize) -> i64 {
    let start = if skip > 0 {
        skip_varint_signed(buf, index, skip)
    } else {
        index
    };
    decode_varint_signed(&buf[start..]).0
}

#[inline]
pub fn combine_uint(index1: u32, index2: u32) -> u32 {
    (index1 << 16) | index2
}

#[inline]
pub fn unpack_uint(packed: u32) -> (u32, u32) {
    ((packed >> 16) & 0xffff, packed & 0xffff)
}

/// Encode a tagged value for snapshot storage.
/// opencoder.py: tag(kind, value)
pub fn tag(kind: u8, value: u32) -> u32 {
    (value << TAG_SHIFT) | kind as u32
}

/// Decode a tagged value from snapshot storage.
/// opencoder.py: untag(tagged) -> (kind, value)
pub fn untag(tagged: u32) -> (u8, u32) {
    let kind = (tagged & TAG_MASK as u32) as u8;
    let value = tagged >> TAG_SHIFT;
    (kind, value)
}

/// opencoder.py `class BaseTrace(object): pass`.
pub trait BaseTrace {}

// ── Trace Iterator ──
// opencoder.py TraceIterator(BaseTrace).
//
// Literal port of RPython's TraceIterator. Each call to `next()` produces
// a *fresh* operation whose `pos` is a freshly allocated OpRef (from the
// majit-specific `_fresh` counter) and whose args have been translated
// through `_cache` from raw trace positions to the per-iteration fresh
// OpRefs.
//
// In RPython every visited op is a freshly allocated `cls()` ResOperation
// Python object whose identity is `is`. Two iterators over the same trace
// therefore produce DIFFERENT box objects at the same `_index` slot — the
// `_index` field walks `_cache` in raw trace position coordinates; the
// freshness comes from `cls()`. majit has no separate identity from
// position, so we split the RPython `_index` into two fields:
//
//   * `_index`  — raw trace position coordinate, matching RPython's
//                 `self._cache[self._index] = res; self._index += 1`. Used
//                 by `replace_last_cached`/`kill_cache_at` which key on
//                 `_cache` slots.
//   * `_fresh`  — majit-specific fresh OpRef allocation counter, seeded
//                 from the `start_fresh` constructor argument. Phase 1
//                 starts at `num_inputs`; Phase 2 starts at Phase 1's
//                 `_fresh` high water so the two iterations produce
//                 disjoint OpRef ranges.
//
// RPython keeps these two roles fused because Python object identity
// makes `_index`'s dual use safe.

/// opencoder.py class TraceIterator(BaseTrace).
pub struct TraceIterator<'a, T = majit_ir::OpRc> {
    /// opencoder.py:252 self.trace
    pub trace: &'a [T],
    /// opencoder.py:255 self._cache: per-iterator map from raw trace
    /// position to the fresh box OBJECT materialized for this iteration.
    /// In RPython this is `[None] * trace._index` holding the fresh
    /// `cls()` ResOperation / `inputarg_from_tp` objects themselves;
    /// here `Vec<Option<Operand>>` carrying the bound producer `Rc`
    /// directly (`Operand` IS the producer object, matching RPython's
    /// stored `cls()` instance — the cache slot owns it outright).
    pub _cache: Vec<Option<Operand>>,
    /// opencoder.py:259-262 self.inputargs: fresh inputarg boxes for this
    /// iteration. Each is a freshly allocated `InputArg` object whose
    /// position comes from the iterator's `_fresh` counter at
    /// construction time (`rop.inputarg_from_tp(arg.type)`).
    pub inputargs: Vec<majit_ir::InputArgRc>,
    /// opencoder.py:268 self.start
    pub start: usize,
    /// opencoder.py:269 self.pos
    pub pos: usize,
    /// opencoder.py:270 self._count
    pub _count: u32,
    /// opencoder.py:271 self._index: raw trace position coordinate of the
    /// next `_cache` slot to write. RPython uses this as both the cache
    /// slot and the implicit fresh-ResOp position; majit uses it ONLY for
    /// the cache slot (see `_fresh` for the OpRef allocation counter).
    /// Advanced on every non-void op processed by `next()`.
    pub _index: u32,
    /// opencoder.py:272 self.start_index — raw trace position coordinate
    /// of the iteration start.
    pub start_index: u32,
    /// opencoder.py:273 self.end
    pub end: usize,
    /// majit-specific fresh OpRef allocation counter. Seeded from the
    /// `start_fresh` constructor argument. Advances by one per non-void
    /// op in `next()` and per inputarg in `new()`. Two iterations over
    /// the same trace produce disjoint OpRef ranges by passing different
    /// `start_fresh` values.
    pub _fresh: u32,
}

impl<T> BaseTrace for TraceIterator<'_, T> where T: std::borrow::Borrow<majit_ir::Op> {}

impl<'a, T> TraceIterator<'a, T>
where
    T: std::borrow::Borrow<majit_ir::Op>,
{
    /// opencoder.py TraceIterator.__init__.
    ///
    /// `force_inputargs` corresponds to the same RPython parameter
    /// (cut-trace path); when None, the iterator pre-seeds the cache with
    /// the trace's own inputargs at positions `[0, num_inputargs)`.
    /// `start_fresh` is majit-specific: it seeds the fresh OpRef counter
    /// so two iterations over the same trace can produce disjoint OpRef
    /// ranges. RPython does not need this because each iteration's
    /// `cls()` allocation produces distinct Python objects.
    pub fn new(
        trace: &'a [T],
        start: usize,
        end: usize,
        force_inputargs: Option<&[OpRef]>,
        inputarg_types: &[majit_ir::Type],
        start_fresh: u32,
    ) -> Self {
        let dense_positions: Vec<OpRef> = inputarg_types
            .iter()
            .enumerate()
            .map(|(position, &tp)| OpRef::input_arg_typed(position as u32, tp))
            .collect();
        Self::new_with_input_layout(
            trace,
            start,
            end,
            force_inputargs,
            inputarg_types,
            &dense_positions,
            start_fresh,
        )
    }

    /// `TraceIterator.__init__` with the recorded trace's actual input boxes.
    ///
    /// RPython seeds `_cache[trace.inputargs[i].get_position()]`, not
    /// `_cache[i]`.  Most majit traces have dense input positions, so `new`
    /// retains the compact type-only surface for those callers.  Guard-failure
    /// histories can contain holes, however: `Trace(max_num_inputargs)` keeps
    /// the original coordinate space while `History.set_inputargs` filters
    /// dead boxes.  This entry point preserves those sparse positions.
    pub fn new_with_inputargs(
        trace: &'a [T],
        start: usize,
        end: usize,
        force_inputargs: Option<&[OpRef]>,
        inputargs: &[impl AsRef<InputArg>],
        start_fresh: u32,
    ) -> Self {
        let inputarg_types: Vec<majit_ir::Type> =
            inputargs.iter().map(|arg| arg.as_ref().tp.get()).collect();
        let inputarg_positions: Vec<OpRef> =
            inputargs.iter().map(|arg| arg.as_ref().opref()).collect();
        Self::new_with_input_layout(
            trace,
            start,
            end,
            force_inputargs,
            &inputarg_types,
            &inputarg_positions,
            start_fresh,
        )
    }

    fn new_with_input_layout(
        trace: &'a [T],
        start: usize,
        end: usize,
        force_inputargs: Option<&[OpRef]>,
        inputarg_types: &[majit_ir::Type],
        inputarg_positions: &[OpRef],
        start_fresh: u32,
    ) -> Self {
        debug_assert_eq!(inputarg_types.len(), inputarg_positions.len());
        // self._cache = [None] * trace._index
        // The iterator's cache must be large enough to hold any raw trace
        // position we may encounter. RPython sizes it from `trace._index`
        // (the encoder's monotonic op-result counter); here the equivalent
        // is the maximum raw position seen in `trace[start..end]` plus one.
        let num_inputargs = inputarg_types.len();
        let mut max_pos = 0u32;
        let mut consider = |opref: majit_ir::OpRef| {
            if !opref.is_none() && !opref.is_constant() {
                max_pos = max_pos.max(opref.raw());
            }
        };
        for op in trace[start..end].iter().map(std::borrow::Borrow::borrow) {
            consider(op.pos().get());
            for a in op.args_slice().iter() {
                consider(a.to_opref());
            }
            op.visit_failarg_oprefs(&mut consider);
        }
        let cache_size = ((max_pos as usize) + 1).max(num_inputargs);
        let mut _cache: Vec<Option<Operand>> = vec![None; cache_size];
        let mut _fresh = start_fresh;
        let inputargs: Vec<majit_ir::InputArgRc>;
        if let Some(force) = force_inputargs {
            // opencoder.py:259-262 self.inputargs =
            //     [rop.inputarg_from_tp(arg.type) for arg in force_inputargs]
            //   for i, arg in enumerate(force_inputargs):
            //       self._cache[arg.get_position()] = self.inputargs[i]
            inputargs = force
                .iter()
                .map(|arg| {
                    // opencoder.py:259-262: rop.inputarg_from_tp(arg.type)
                    // — every force_inputargs entry has a known type tag.
                    let tp = arg.ty().unwrap_or_else(|| {
                        panic!(
                            "TraceIterator::new(force_inputargs): untyped OpRef ({:?}) — \
                             upstream `inputarg_from_tp(arg.type)` requires every entry to \
                             carry a type tag",
                            arg
                        )
                    });
                    let r = InputArg::from_type_rc(tp, _fresh);
                    _fresh += 1;
                    r
                })
                .collect();
            for (i, &arg) in force.iter().enumerate() {
                let p = arg.raw() as usize;
                if p >= _cache.len() {
                    _cache.resize(p + 1, None);
                }
                _cache[p] = Some(Operand::from_bound_inputarg(&inputargs[i]));
            }
        } else {
            // opencoder.py:264-267 self.inputargs =
            //     [rop.inputarg_from_tp(arg.type) for arg in self.trace.inputargs]
            //   for i, arg in enumerate(self.inputargs):
            //       self._cache[self.trace.inputargs[i].get_position()] = arg
            inputargs = inputarg_types
                .iter()
                .map(|&tp| {
                    let r = InputArg::from_type_rc(tp, _fresh);
                    _fresh += 1;
                    r
                })
                .collect();
            for (i, position) in inputarg_positions.iter().enumerate() {
                let p = position.raw() as usize;
                if p >= _cache.len() {
                    _cache.resize(p + 1, None);
                }
                _cache[p] = Some(Operand::from_bound_inputarg(&inputargs[i]));
            }
        }
        TraceIterator {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            _count: start as u32,
            _index: start as u32,
            start_index: start as u32,
            end,
            _fresh,
        }
    }

    /// opencoder.py _get(self, i).
    fn _get(&self, i: usize) -> Operand {
        match self._cache.get(i).cloned().flatten() {
            Some(res) => res,
            None => {
                let cur = self.trace.get(self.pos.saturating_sub(1)).map(|op| {
                    let op: &majit_ir::Op = std::borrow::Borrow::borrow(op);
                    let hits: Vec<String> = op
                        .args_slice()
                        .iter()
                        .enumerate()
                        .filter_map(|(ai, a)| {
                            let r = a.to_opref();
                            (!r.is_none() && !r.is_constant() && r.raw() as usize == i)
                                .then(|| format!("arg{ai}={r:?}"))
                        })
                        .collect();
                    format!("{:?} pos={:?} hits={hits:?}", op.opcode, op.pos().get())
                });
                let producers: Vec<String> = self.trace[self.start..self.end]
                    .iter()
                    .map(std::borrow::Borrow::borrow)
                    .filter(|op| {
                        let p = op.pos().get();
                        !p.is_none() && !p.is_constant() && p.raw() as usize == i
                    })
                    .map(|op| format!("{:?} pos={:?}", op.opcode, op.pos().get()))
                    .collect();
                panic!(
                    "TraceIterator._get cache miss at {i} (cache_len={}, pos={}, start={}, end={}, index={}, start_index={}, fresh={}, cur={}, producers_at_{i}={:?})",
                    self._cache.len(),
                    self.pos,
                    self.start,
                    self.end,
                    self._index,
                    self.start_index,
                    self._fresh,
                    cur.as_deref().unwrap_or("<none>"),
                    producers,
                );
            }
        }
    }

    /// opencoder.py done().
    pub fn done(&self) -> bool {
        self.pos >= self.end
    }

    /// opencoder.py _untag(tagged).
    ///
    /// In RPython this dispatches on the tag (TAGBOX/TAGINT/TAGCONSTPTR/
    /// TAGCONSTOTHER). majit's OpRef carries the tag implicitly: constants
    /// have `OpRef >= CONST_BASE`, NONE is `u32::MAX`. Both pass through
    /// unchanged (value-inline const box); only TAGBOX-equivalent OpRefs
    /// go through `_get`, which returns the cached fresh box object.
    fn _untag(&self, opref: OpRef) -> Operand {
        if opref.is_none() || opref.is_constant() {
            Operand::from_opref(opref)
        } else {
            self._get(opref.raw() as usize)
        }
    }

    /// opencoder.py kill_cache_at(pos).
    pub fn kill_cache_at(&mut self, pos: usize) {
        if pos != 0 {
            self._cache[pos] = None;
        }
    }

    /// opencoder.py replace_last_cached(oldbox, box).
    /// RPython writes `_cache[_index - 1]` where `_index` walks `_cache`
    /// densely in ops-count coordinates. In majit the key is the raw
    /// trace position (which is monotonic for non-void ops in a
    /// well-formed trace), so `_index - 1` lands on the same cache slot
    /// the previous `next()` call wrote.
    pub fn replace_last_cached(&mut self, oldbox: OpRef, new_box: Operand) {
        let last_idx = (self._index - 1) as usize;
        // opencoder.py `assert self._cache[self._index - 1] is oldbox`
        // — checked by encoding here since callers identify the old box
        // by its OpRef.
        debug_assert_eq!(
            self._cache[last_idx].as_ref().map(|b| b.to_opref()),
            Some(oldbox)
        );
        self._cache[last_idx] = Some(new_box);
    }

    /// opencoder.py next() — produce the next operation as a fresh
    /// box (`cls()` in RPython) with translated args. The returned `OpRc`
    /// IS the cached box object: a later arg referencing this op's raw
    /// trace position resolves (via `_untag` → `_get`) to a clone of the
    /// same `Rc` handle, matching RPython's `_cache[i] is res` identity.
    #[expect(
        clippy::should_implement_trait,
        reason = "This is the direct opencoder.py TraceIterator.next port; it returns RPython operation identity handles and is consumed by explicit parity control flow rather than Rust iterator adapters"
    )]
    pub fn next(&mut self) -> Option<majit_ir::OpRc> {
        if self.done() {
            return None;
        }
        let src: &majit_ir::Op = self.trace[self.pos].borrow();
        self.pos += 1;
        // opencoder.py `TraceIterator.next` `cls()` / `ResOperation(opnum, args)`
        // — a fresh object, not a clone of the recorded op. Cloning copied
        // `Box<OpKindExtra>` (32 B) on every guard; minting writes
        // remapped args/fail_args onto a new `Op` the way `cls()` does.
        let mut args: majit_ir::resoperation::OpArgVec = smallvec::SmallVec::new();
        for i in 0..src.num_args() {
            let a = src.arg(i);
            args.push(if a.is_constant() {
                a
            } else {
                self._untag(a.to_opref())
            });
        }
        // opencoder.py `TraceIterator.next`: `if rop.is_guard(opnum): descr = None`
        // and `res.rd_resume_position = descr_index`. A guard's
        // `ResumeGuardDescr` is invented at emit (`invent_fail_descr_for_op`).
        // Copying the recorded descr makes phase 2 call `resume.py`
        // `ResumeDataVirtualAdder.finish` on the descr phase 1 already
        // numbered. A foriter marker is not a resume payload: re-mint it on
        // a fresh descr so failure routing keeps the key.
        let res = if src.opcode.is_guard() {
            let op = majit_ir::Op::new(src.opcode, &args);
            if let Some(d) = src.getdescr() {
                if let Some(key) = d.range_foriter_green_key() {
                    op.setdescr(crate::compile::make_resume_guard_descr_range_foriter(key));
                } else if let Some(key) = d.instance_next_foriter_green_key() {
                    op.setdescr(
                        crate::compile::make_resume_guard_descr_instance_next_foriter(
                            Some(src.opcode),
                            key,
                        ),
                    );
                }
            }
            op
        } else {
            match src.getdescr() {
                Some(d) => majit_ir::Op::with_descr(src.opcode, &args, d),
                None => majit_ir::Op::new(src.opcode, &args),
            }
        };
        // history.py FrontendOp `_resint`/`_resfloat`/`_resref` ride on
        // the recorded box. The byte-stream iterator never sees them;
        // this structured walker must keep the stamp or fannkuch grows
        // two extra JUMP bridges (22 → 24, 5049 → 5450 guard_failures).
        if let Some(v) = src.get_value() {
            res.set_value(v);
        }
        if let Some(vecinfo) = src.get_vecinfo() {
            res.set_vecinfo(vecinfo);
        }
        if let Some(fa) = src.guard_fail_args() {
            let mapped: majit_ir::resoperation::OpArgVec = fa
                .iter()
                .map(|arg| {
                    if arg.is_constant() {
                        arg.clone()
                    } else {
                        self._untag(arg.to_opref())
                    }
                })
                .collect();
            res.setfailargs(mapped);
        }
        let resume = src.rd_resume_position();
        if resume >= 0 {
            res.set_rd_resume_position(resume);
        }
        if let Some(types) = src.get_fail_arg_types() {
            res.set_fail_arg_types(types);
        }
        // RPython opencoder.py:399-401:
        //     res = ResOperation(opnum, args, descr)   # fresh cls() object
        //     if res.type != 'v':
        //         self._cache[self._index] = res
        //         self._index += 1
        //
        // RPython allocates a fresh `cls()` ResOperation Python object on
        // EVERY visited op (void or non-void); only non-void results land
        // in `_cache`, but the freshness is unconditional. majit's
        // analogue of "fresh cls()" is the `_fresh` OpRef counter, so we
        // must advance `_fresh` for void ops too — otherwise a non-void
        // op processed after a void op gets a fresh OpRef that collides
        // with the void op's raw `pos`. With this in place,
        // `start_fresh = 0` over a recorder-emitted trace produces a
        // bit-identical sequence of OpRefs (every op `i` gets
        // `OpRef::from_raw(num_inputs + i)`, matching `recorder.record_op`'s
        // monotonic `op_count`).
        let is_void_result =
            src.pos().get().is_none() || src.opcode.result_type() == majit_ir::Type::Void;
        let mut cache_slot: Option<usize> = None;
        if !is_void_result {
            let orig = src.pos().get().raw() as usize;
            if orig >= self._cache.len() {
                self._cache.resize(orig + 1, None);
            }
            // opencoder.py:399-401 always allocates a fresh ResOperation
            // instance and then overwrites `_cache[self._index]` with it.
            // Reusing an existing cache entry collapses distinct box
            // identities when a raw op slot collides with an inputarg or
            // earlier phase-local box. The ResOp class (`opclasses[opnum]`
            // / cls()) is intrinsically typed via the IntOp/FloatOp/RefOp
            // mixin (resoperation.py); pyre uses
            // `opcode.result_type()`.
            let fresh = OpRef::op_typed(self._fresh, src.opcode.result_type());
            self._fresh += 1;
            res.pos().set(fresh);
            cache_slot = Some(orig);
            // RPython `_index` parity: advance past the cache slot we
            // just wrote. In RPython this happens via `_index += 1`
            // because `_index` == cache slot index; in majit the cache
            // key is the raw trace position, so we assign `orig + 1`
            // directly to keep `_cache[_index - 1]` (`replace_last_cached`)
            // pointing at the slot we just wrote.
            self._index = orig as u32 + 1;
        } else if !src.pos().get().is_none() {
            // Void ops are not `_index` boxes. Name them by the walk's
            // `_count` so `_fresh` stays aligned with value `_index`.
            res.pos().set(OpRef::void_op(self._count));
        } else {
            res.pos().set(src.pos().get());
        }
        let res = OpRc::new(res);
        // opencoder.py `self._cache[self._index] = res` — cache
        // the fresh op OBJECT itself so later references resolve to the
        // same identity.
        if let Some(orig) = cache_slot {
            self._cache[orig] = Some(Operand::from_bound_op(&res));
        }
        // self._count += 1
        self._count += 1;
        Some(res)
    }
}

// ── Byte-stream TraceIterator (M4) ─────────────────────────────────────
//
// `ByteTraceIter` is the structurally-correct port of RPython's
// `TraceIterator` (opencoder.py).  It walks
// `TraceRecordBuffer._ops` byte-by-byte, producing a fresh `Op` per
// iteration.  The existing `TraceIterator<'a>` above still exists for
// legacy consumers that hold a pre-materialized `&[OpRc]` slice (via
// `TreeLoop.ops`); both coexist during the M2–M7 migration and
// `ByteTraceIter` takes over once every consumer is migrated off the
// structured-IR slice.
//
// Current scope (M4 step 1): fixed-arity 0/1/2/3 opcodes with TAGBOX
// args only.  Constants (TAGINT/TAGCONSTPTR/TAGCONSTOTHER), variable-
// arity ops (arity == None), and descr-bearing ops panic with a
// dedicated message — those decode branches land in M4 step 2 alongside
// inline-constant decode + descr-table wiring.

/// opencoder.py `TraceIterator`.  Byte-stream walker over
/// `TraceRecordBuffer._ops`.
/// opencoder.py `class CutTrace(BaseTrace)` — thin view over
/// a `TraceRecordBuffer` that starts iteration at `start` (a byte
/// offset captured from `cut_point()`) and substitutes a fresh
/// inputarg list for the op stream between `start` and
/// `trace._pos`.
///
/// `CutTrace.cut_at` mutates the parent, so the view owns an exclusive
/// borrow. Each iterator reborrows it for its own lifetime; the parent
/// cannot be cut while a byte-stream reader is still using it.
pub struct CutTrace<'a> {
    pub trace: &'a mut TraceRecordBuffer,
    pub start: usize,
    pub count: u32,
    pub index: u32,
    /// Templates: each typed `OpRef` names an op position in the
    /// parent trace and carries that op's result type; `get_iter`
    /// seeds `_cache[opref.raw()]` with the fresh OpRef so downstream
    /// TAGBOX(p) decodes resolve to the new iterator-local inputarg.
    /// Mirrors RPython `cut_trace_from(cut, inputargs)` where
    /// `inputargs` is a list of `Box{Int,Float,Ref}` — each Box
    /// carries `Box.type` (`history.py:182 AbstractValue.type`) which
    /// pyre encodes as the typed `OpRef` variant tag.
    pub inputargs: Vec<OpRef>,
}

impl BaseTrace for CutTrace<'_> {}

impl<'a> CutTrace<'a> {
    /// opencoder.py `CutTrace.cut_at(cut)` — delegate to the
    /// parent trace after asserting the cut lies beyond this cut-trace's
    /// starting count.
    pub fn cut_at(&mut self, cut: crate::recorder::TracePosition) {
        assert!(cut._count > self.count, "cut must lie after CutTrace start");
        self.trace.cut_at(cut);
    }

    /// opencoder.py `CutTrace.get_iter` — build a
    /// `ByteTraceIter` that walks the parent trace from `start` to
    /// `trace._pos`, with `_cache` / `inputargs` seeded from the cut's
    /// inputarg templates.
    #[allow(dead_code)]
    pub(crate) fn get_iter(&self) -> ByteTraceIter<'_> {
        let trace = &*self.trace;
        ByteTraceIter::new_for_cut(
            trace,
            self.start,
            trace._pos,
            self.count,
            self.index,
            &self.inputargs,
        )
    }
}

pub(crate) struct ByteTraceIter<'a> {
    /// Enclosing trace buffer — source of `_ops`, `_refs`, `_bigints`,
    /// `_floats`, `_descrs`, `metainterp_sd.all_descrs`.
    pub trace: &'a TraceRecordBuffer,
    /// opencoder.py:255 `self._cache` — raw trace position → fresh
    /// per-iteration producer `Operand`, mirroring
    /// `TraceIterator::_cache`. A later TAGBOX arg resolves through
    /// `_get` to the same bound producer (`Operand::Op` /
    /// `Operand::InputArg`). The slot holds the producer `Rc` strongly,
    /// so it owns the cached op for the walk's lifetime — RPython's
    /// `self._cache[self._index] = res` stored the ResOperation OBJECT
    /// outright.
    pub _cache: Vec<Option<Operand>>,
    /// opencoder.py:259-263 `self.inputargs` — fresh iterator-local
    /// `InputArg` objects bound to the trace's inputargs.
    #[allow(dead_code)]
    pub inputargs: Vec<majit_ir::InputArgRc>,
    #[allow(dead_code)]
    pub start: usize,
    pub pos: usize,
    pub end: usize,
    pub _count: u32,
    pub _index: u32,
    #[allow(dead_code)]
    pub start_index: u32,
    /// Majit-specific fresh OpRef counter (same role as
    /// `TraceIterator::_fresh` on the legacy walker).
    pub _fresh: u32,
}

impl<'a> ByteTraceIter<'a> {
    /// opencoder.py `TraceIterator.__init__`.
    ///
    /// `start` / `end` are byte offsets into `trace._ops`.  Pass
    /// `trace._start as usize` / `trace._pos` for a full walk.
    /// `start_fresh` seeds the fresh-OpRef counter (pyre-only — enables
    /// disjoint OpRef namespaces across successive iterations).
    ///
    /// `_untag` mints inline-Const OpRef variants directly
    /// (history.py/268/314 `Const*.value` inline), so the decode path
    /// needs no external/legacy ConstantPool. It still reads the
    /// opencoder-local `_refs` / `_bigints` / `_floats` pools.
    pub(crate) fn new(
        trace: &'a TraceRecordBuffer,
        start: usize,
        end: usize,
        start_fresh: u32,
    ) -> Self {
        Self::new_with_inputargs(trace, start, end, &trace.inputargs, start_fresh)
    }

    /// `TraceIterator.__init__` with an explicit inputarg list.
    ///
    /// `History.set_inputargs` is hole-filtered and keeps each surviving
    /// box's original `get_position()`. Seeding `_cache` from
    /// `trace.inputargs` (the reserved prefix) types a later live Int as
    /// the dead Ref when those lists are zipped by compact index.
    pub(crate) fn new_with_inputargs<A: AsRef<InputArg>>(
        trace: &'a TraceRecordBuffer,
        start: usize,
        end: usize,
        live_inputargs: &[A],
        start_fresh: u32,
    ) -> Self {
        let cache_size = (trace._index as usize).max(trace.max_num_inputargs as usize);
        let mut _cache: Vec<Option<Operand>> = vec![None; cache_size];
        // `_cache` is keyed by original `_index` (`get_position()`). The
        // fresh InputArg objects are the iterator's `cls()` identity;
        // their own index is `_fresh` so the optimizer can tell them
        // apart from the parent loop's boxes.
        let mut _fresh = start_fresh;
        let inputargs: Vec<majit_ir::InputArgRc> = live_inputargs
            .iter()
            .map(|ia| {
                let ia = ia.as_ref();
                let r = InputArg::from_type_rc(ia.tp.get(), _fresh);
                _fresh += 1;
                if let Some(value) = ia.get_value() {
                    r.set_value(value);
                }
                r
            })
            .collect();
        for (i, ia) in live_inputargs.iter().enumerate() {
            let p = ia.as_ref().index as usize;
            if p >= _cache.len() {
                _cache.resize(p + 1, None);
            }
            _cache[p] = Some(Operand::from_bound_inputarg(&inputargs[i]));
        }
        ByteTraceIter {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            end,
            _count: start as u32,
            _index: start as u32,
            start_index: start as u32,
            _fresh,
        }
    }

    /// opencoder.py + opencoder.py `CutTrace.get_iter`
    /// — specialised `TraceIterator` constructor for `cut_trace_from`
    /// that seeds `_cache` from explicit typed-`OpRef` inputarg
    /// templates.  Each template names an op position in the uncut
    /// trace and carries that op's result type via its variant tag,
    /// matching RPython's `Box.type` (`history.py:182`).  The seeded
    /// fresh OpRefs reuse the template's type via
    /// `OpRef::input_arg_typed`, so downstream consumers preserve the
    /// `Box.type` invariant the cut sub-trace operates under.
    #[allow(dead_code)]
    pub(crate) fn new_for_cut(
        trace: &'a TraceRecordBuffer,
        start: usize,
        end: usize,
        count: u32,
        index: u32,
        inputarg_templates: &[OpRef],
    ) -> Self {
        let cache_size = (trace._index as usize).max(trace.max_num_inputargs as usize);
        let mut _cache: Vec<Option<Operand>> = vec![None; cache_size];
        let mut inputargs: Vec<majit_ir::InputArgRc> = Vec::with_capacity(inputarg_templates.len());
        let mut _fresh = 0u32;
        for &template in inputarg_templates {
            // Constant templates would imply the cut sub-trace treats a
            // literal as an inputarg — not a shape RPython's
            // `cut_trace_from` emits — so reject up front rather than
            // silently substitute.
            if template.is_constant() {
                panic!(
                    "ByteTraceIter::new_for_cut: constant inputarg template {:?} has \
                     no original trace position to map into `_cache`",
                    template
                );
            }
            let tp = template.ty().unwrap_or_else(|| {
                panic!(
                    "ByteTraceIter::new_for_cut: untyped inputarg template {:?} has no \
                     `Box.type` to carry into the fresh inputarg variant",
                    template
                )
            });
            let fresh_ia = InputArg::from_type_rc(tp, _fresh);
            _fresh += 1;
            let slot = template.raw() as usize;
            if slot >= _cache.len() {
                _cache.resize(slot + 1, None);
            }
            _cache[slot] = Some(Operand::from_bound_inputarg(&fresh_ia));
            inputargs.push(fresh_ia);
        }
        ByteTraceIter {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            end,
            _count: count,
            _index: index,
            start_index: index,
            _fresh,
        }
    }

    /// opencoder.py `done`.
    pub fn done(&self) -> bool {
        self.pos >= self.end
    }

    /// opencoder.py `_nextbyte`.
    fn _nextbyte(&mut self) -> u8 {
        let b = self.trace.ops_bytes()[self.pos];
        self.pos += 1;
        b
    }

    /// opencoder.py `_next` — signed varint decode from
    /// `trace._ops` at `self.pos`.  Shares the wire format with
    /// `TraceRecordBuffer::append_int`.
    fn _next(&mut self) -> i64 {
        let (v, consumed) = decode_varint_signed(&self.trace.ops_bytes()[self.pos..]);
        self.pos += consumed;
        v
    }

    /// opencoder.py `_get`.
    fn _get(&self, i: usize) -> Operand {
        match self._cache.get(i).cloned().flatten() {
            Some(res) => res,
            None => panic!(
                "ByteTraceIter._get: cache miss at {i} \
                 (cache_len={}, filled={}, pos={}, start={}, end={}, \
                 index={}, start_index={}, fresh={}, \
                 max_in={}, trb_index={}, trb_count={}, n_in={})",
                self._cache.len(),
                self._cache.iter().filter(|s| s.is_some()).count(),
                self.pos,
                self.start,
                self.end,
                self._index,
                self.start_index,
                self._fresh,
                self.trace.max_num_inputargs,
                self.trace._index,
                self.trace._count,
                self.trace.inputargs.len(),
            ),
        }
    }

    /// opencoder.py `_untag` — full dispatch.
    ///
    /// TAGBOX → `_cache` lookup.  TAGINT → inline `ConstInt(v)`
    /// (history.py `ConstInt.value` inline).  TAGCONSTPTR → inline
    /// `ConstPtr(trace._refs[v])` (history.py `ConstPtr.value` inline).
    /// TAGCONSTOTHER → inline `ConstFloat`/`ConstInt` from
    /// `trace._floats`/`trace._bigints` (history.py:268/227 inline). Every
    /// TAG* arm mints inline-Const directly, so the decode needs no
    /// external/legacy ConstantPool beyond these opencoder-local pools.
    ///
    /// `pub(crate)` because `SnapshotIterator::get` / `unpack_array`
    /// (opencoder.py) dispatch through `main_iter._untag`.
    pub(crate) fn _untag(&mut self, tagged: i64) -> Operand {
        // RPython opencoder.py _untag uses arithmetic shift on a
        // Python int; in Rust we preserve sign by going through i64
        // rather than u32 for the value.
        let tag = (tagged & TAG_MASK as i64) as u8;
        let v = tagged >> TAG_SHIFT;
        match tag {
            TAGBOX => {
                debug_assert!(v >= 0, "TAGBOX value must be non-negative, got {}", v);
                self._get(v as usize)
            }
            TAGINT => {
                // opencoder.py:326-327 ConstInt(v) — signed small int.
                // history.py ConstInt.value inline. A const OpRef
                // sheds to `Operand::Const`, never a position-only box.
                Operand::from_opref(OpRef::const_int(v))
            }
            TAGCONSTPTR => {
                // opencoder.py:328-329 ConstPtr(self.trace._refs[v]) —
                // history.py `ConstPtr.value` inline. The
                // op-graph walker forwards `OpRef::ConstPtr(GcRef)`
                // slots across minor collection.
                let addr = self.trace._refs[v as usize];
                Operand::from_opref(OpRef::const_ptr(majit_ir::GcRef(addr as usize)))
            }
            TAGCONSTOTHER => {
                // opencoder.py:330-334 bigint vs float split on bit 0.
                let pool_idx = (v >> 1) as usize;
                if v & 1 != 0 {
                    // history.py ConstFloat.value inline.
                    let bits = self.trace._floats[pool_idx];
                    Operand::from_opref(OpRef::const_float(f64::from_bits(bits)))
                } else {
                    // history.py ConstInt.value inline.
                    let val = self.trace._bigints[pool_idx];
                    Operand::from_opref(OpRef::const_int(val))
                }
            }
            other => unreachable!("ByteTraceIter: unknown tag {}", other),
        }
    }
}

impl<'a> Iterator for ByteTraceIter<'a> {
    type Item = majit_ir::OpRc;
    fn next(&mut self) -> Option<majit_ir::OpRc> {
        if self.done() {
            return None;
        }
        // opencoder.py `opnum = self._nextbyte()`.
        let opnum = self._nextbyte();
        let opcode = OpCode::from_u16(opnum as u16)
            .unwrap_or_else(|| panic!("ByteTraceIter: unknown opnum {}", opnum));
        // opencoder.py `argnum = oparity[opnum]; if argnum == -1: argnum = self._next()`.
        let arity = match opcode.arity() {
            Some(n) => n as usize,
            // opencoder.py variable-arity path: `argnum = self._next()`.
            None => self._next() as usize,
        };
        // opencoder.py — read `argnum` tagged args and untag.
        let mut args: majit_ir::resoperation::OpArgVec = smallvec::SmallVec::new();
        for _ in 0..arity {
            let tagged = self._next();
            args.push(self._untag(tagged));
        }
        // opencoder.py __init__ `opwithdescr` branch.  Guards thread the
        // snapshot index into `rd_resume_position` but emit `descr=None`
        // at the Op level; every other descr-bearing op resolves its
        // descr via `metainterp_sd.all_descrs` (global slot, index <
        // `all_descr_len + 1`) or `trace._descrs` (local slot, offset
        // by `all_descr_len + 1`).  `descr_index == 0` is the
        // placeholder used by `record_op` for no-descr writes.
        let (descr, rd_resume_position) = if opcode.has_descr() {
            let descr_index = self._next();
            let resume_pos = if opcode.is_guard() {
                descr_index as i32
            } else {
                -1
            };
            let resolved = if descr_index == 0 || opcode.is_guard() {
                None
            } else {
                // The boundary is the one `Trace::new` captured, not the
                // live length — see `Trace::all_descr_len`.
                let all_descr_len = self.trace.all_descr_len as i64;
                if descr_index < all_descr_len + 1 {
                    let all_descrs = self.trace.metainterp_sd.all_descrs().lock();
                    Some(all_descrs[(descr_index - 1) as usize].clone())
                } else {
                    Some(
                        self.trace._descrs[(descr_index - all_descr_len - 1) as usize]
                            .clone()
                            .expect("ByteTraceIter: trace._descrs slot was None"),
                    )
                }
            };
            (resolved, resume_pos)
        } else {
            (None, -1)
        };
        // RPython opencoder.py: `res = ResOperation(opnum, args)`
        // is a fresh Python object; the cache only receives non-void
        // results.  majit allocates an OpRef for every op (void or not)
        // so later non-void ops get contiguous positions. `args` are
        // already bound producers (`_untag` → `_get` bound `Operand::Op` /
        // `Operand::InputArg` / inline const), so building the op binds
        // each arg to its producer directly.
        let op = match descr {
            Some(d) => majit_ir::Op::with_descr(opcode, &args, d),
            None => majit_ir::Op::new(opcode, &args),
        };
        op.set_rd_resume_position(rd_resume_position);
        // opencoder.py:373-374 `cls = opclasses[opnum]; res = cls()` —
        // the ResOp class is intrinsically typed via the IntOp/FloatOp/
        // RefOp mixin (resoperation.py) or the AbstractResOp
        // default `'v'` (resoperation.py). Value ops use `_index` as
        // `FrontendOp.get_position()`; void ops are named by the
        // op-sequence count of this walk.
        let ty = opcode.result_type();
        let pos = if ty != Type::Void {
            let p = OpRef::op_typed(self._fresh, ty);
            self._fresh += 1;
            p
        } else {
            OpRef::void_op(self._count)
        };
        op.pos().set(pos);
        // opencoder.py `self._cache[self._index] = res` — the
        // fresh op IS the cached box object, so a later TAGBOX arg binds
        // to this exact `Rc` (`from_bound_op` → `Operand::Op`).
        let op: majit_ir::OpRc = OpRc::new(op);
        // opencoder.py combine_uint — cache non-void result at `_index`, bump.
        if ty != Type::Void {
            let slot = self._index as usize;
            if slot >= self._cache.len() {
                self._cache.resize(slot + 1, None);
            }
            // The slot owns the producer `Rc` strongly, keeping the
            // cached op alive for the rest of the walk.
            self._cache[slot] = Some(Operand::from_bound_op(&op));
            self._index += 1;
        }
        // opencoder.py:432 `self._count += 1`.
        self._count += 1;
        Some(op)
    }
}

/// opencoder.py `Trace.get_iter()` — byte-stream form.
impl Trace {
    #[allow(dead_code)]
    pub(crate) fn get_byte_iter(&self) -> ByteTraceIter<'_> {
        ByteTraceIter::new(
            self,
            self._start as usize,
            self._pos,
            self.max_num_inputargs,
        )
    }
}

/// opencoder.py `class BoxArrayIter` — iterator over the
/// encoded tagged box values stored in `_snapshot_array_data`.
///
/// Layout at offset `array_idx`:
/// `[length:varint][box0:varint][box1:varint]...`. `array_idx == 0`
/// always decodes to the empty iterator (opencoder.py
/// `BoxArrayIter.BOXARRAYITER0`; also opencoder.py
/// `new_array(0) -> 0`).
pub struct BoxArrayIter<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: i64,
    /// opencoder.py `self.total_length = self.length` — the
    /// decoded array length, captured at construction. `remaining`
    /// counts down as `next()` consumes items, while `total_length`
    /// keeps the original count so `SnapshotIterator.size`
    /// (opencoder.py) can read it back without re-decoding.
    pub total_length: i64,
}

impl<'a> BoxArrayIter<'a> {
    /// opencoder.py BoxArrayIter.make(index, data).
    pub fn new(data: &'a [u8], array_idx: i64) -> Self {
        if array_idx == 0 {
            return Self {
                data,
                pos: 0,
                remaining: 0,
                total_length: 0,
            };
        }
        let pos = array_idx as usize;
        let (length, consumed) = decode_varint_signed(&data[pos..]);
        Self {
            data,
            pos: pos + consumed,
            remaining: length,
            total_length: length,
        }
    }
}

impl<'a> Iterator for BoxArrayIter<'a> {
    type Item = i64;
    fn size_hint(&self) -> (usize, Option<usize>) {
        // opencoder.py BoxArrayIter.length already tracks the remaining
        // elements; expose it so snapshot consumers allocate only once.
        let remaining = self.remaining.max(0) as usize;
        (remaining, Some(remaining))
    }

    fn next(&mut self) -> Option<i64> {
        if self.remaining <= 0 {
            return None;
        }
        let (tagged, consumed) = decode_varint_signed(&self.data[self.pos..]);
        self.pos += consumed;
        self.remaining -= 1;
        Some(tagged)
    }
}

impl ExactSizeIterator for BoxArrayIter<'_> {}

/// opencoder.py `class TopDownSnapshotIterator` — walks the
/// snapshot chain encoded into `_snapshot_data` in outermost-to-
/// innermost order.
///
/// Yields byte offsets of each parent snapshot record (in the
/// `[jitcode][pc][array][prev]` layout written by `_encode_snapshot`).
/// The top snapshot's leading `vable_array_index` /
/// `vref_array_index` are extracted in `new()` (opencoder.py)
/// and exposed as struct fields.
pub struct TopDownSnapshotIterator<'a> {
    snapshot_data: &'a [u8],
    /// opencoder.py:150 `self.vable_array_index`.
    pub vable_array_index: i64,
    /// opencoder.py:151 `self.vref_array_index`.
    pub vref_array_index: i64,
    /// opencoder.py:152 `self.snapshot_index` — position of the current
    /// parent snapshot's `jitcode` varint.
    current: usize,
    /// opencoder.py `if res == SNAPSHOT_PREV_NONE: raise StopIteration`.
    /// Pyre signals termination via a sentinel field because `usize` can't
    /// carry the -2 value RPython stores in `snapshot_index`.
    done: bool,
}

impl<'a> TopDownSnapshotIterator<'a> {
    /// opencoder.py TopDownSnapshotIterator.__init__.
    pub fn new(snapshot_data: &'a [u8], snapshot_index: usize) -> Self {
        let mut p = snapshot_index;
        let (vable, c) = decode_varint_signed(&snapshot_data[p..]);
        p += c;
        let (vref, c) = decode_varint_signed(&snapshot_data[p..]);
        p += c;
        Self {
            snapshot_data,
            vable_array_index: vable,
            vref_array_index: vref,
            current: p,
            done: false,
        }
    }

    /// opencoder.py iter_array(snapshot_index).
    ///
    /// `snapshot_index` is the byte offset of a parent snapshot's
    /// `jitcode` varint. Returns a byte iterator over the tagged
    /// boxes in that frame's array. `snapshot_array_data` is supplied
    /// as a separate argument to keep the returned iterator free of
    /// a back-borrow on `self`.
    pub fn iter_array<'b>(
        &self,
        snapshot_index: usize,
        snapshot_array_data: &'b [u8],
    ) -> BoxArrayIter<'b> {
        let array_idx = varint_only_decode(self.snapshot_data, snapshot_index, 2);
        BoxArrayIter::new(snapshot_array_data, array_idx)
    }

    /// opencoder.py length(snapshot_index).
    ///
    /// Returns the length of the box array attached to the parent
    /// snapshot at `snapshot_index`. Reads the array offset from the
    /// third varint of the snapshot record, then decodes the length
    /// prefix at that offset in `_snapshot_array_data`.
    pub fn length(&self, snapshot_index: usize, snapshot_array_data: &[u8]) -> i64 {
        let array_idx = varint_only_decode(self.snapshot_data, snapshot_index, 2);
        varint_only_decode(snapshot_array_data, array_idx as usize, 0)
    }

    /// opencoder.py unpack_jitcode_pc(snapshot_index).
    ///
    /// Decodes the leading `(jitcode_index, pc)` pair of a parent
    /// snapshot record. `snapshot_index` is the byte offset of the
    /// `jitcode` varint (the same coordinate yielded by the iterator
    /// and passed through `iter_array`).
    pub fn unpack_jitcode_pc(&self, snapshot_index: usize) -> (i64, i64) {
        let mut p = snapshot_index;
        let (jitcode_index, c) = decode_varint_signed(&self.snapshot_data[p..]);
        p += c;
        let (pc, _) = decode_varint_signed(&self.snapshot_data[p..]);
        (jitcode_index, pc)
    }

    /// opencoder.py is_empty_snapshot(snapshot_index).
    ///
    /// Checks whether the parent snapshot at `snapshot_index` was
    /// written by `create_empty_top_snapshot` (jitcode_index == -1 and
    /// array == empty). Uses the array-index == -1 sentinel to mirror
    /// RPython's `varint_only_decode(..., skip=2) == -1` check.
    pub fn is_empty_snapshot(&self, snapshot_index: usize) -> bool {
        varint_only_decode(self.snapshot_data, snapshot_index, 2) == -1
    }
}

impl<'a> Iterator for TopDownSnapshotIterator<'a> {
    type Item = usize;
    /// opencoder.py TopDownSnapshotIterator.next.
    fn next(&mut self) -> Option<usize> {
        if self.done {
            return None;
        }
        let res = self.current;
        // opencoder.py prev(snapshot_index):
        //   self._index = skip_varint_signed(snapshot_data, snapshot_index, skip=3)
        //   prev = self.decode_snapshot_int()
        //   if prev == SNAPSHOT_PREV_COMES_NEXT: prev = self._index
        let after_array = skip_varint_signed(self.snapshot_data, res, 3);
        let (prev, consumed) = decode_varint_signed(&self.snapshot_data[after_array..]);
        debug_assert!(
            prev as i32 != SNAPSHOT_PREV_NEEDS_PATCHING,
            "snapshot chain contains unpatched NEEDS_PATCHING sentinel"
        );
        let next_cursor = match prev as i32 {
            SNAPSHOT_PREV_NONE => {
                self.done = true;
                res
            }
            SNAPSHOT_PREV_COMES_NEXT => after_array + consumed,
            _ => prev as usize,
        };
        self.current = next_cursor;
        Some(res)
    }
}

/// opencoder.py `class SnapshotIterator` — eager reversed
/// walker over a snapshot chain.
///
/// RPython's `SnapshotIterator` holds the enclosing `TraceIterator` as
/// `main_iter` so `get(tagged)` can dispatch through `_untag` and the
/// per-iteration fresh-box cache. Pyre's `TraceIterator` still walks
/// a `&[OpRc]` slice (pre-byte-stream), so this port exposes the
/// read-side surface without coupling to `TraceIterator`: callers that
/// need TAGBOX → fresh-box mapping use a separate `_untag` helper on
/// their `TraceIterator` instance. The non-TAGBOX surface
/// (iter_array, iter_vable_array, iter_vref_array, unpack_jitcode_pc,
/// size, framestack) ports line-by-line.
///
/// `framestack` matches RPython's bottom-up order: outermost frame
/// first, innermost frame last. Empty when the snapshot was written
/// via `create_empty_top_snapshot` (jitcode_index == -1).
pub struct SnapshotIterator<'a> {
    /// opencoder.py:207 self.topdown_snapshot_iter.
    pub topdown_snapshot_iter: TopDownSnapshotIterator<'a>,
    /// opencoder.py `self.vable_array = it.iter_vable_array()`.
    /// Stored as the `_snapshot_array_data` offset from which a fresh
    /// `BoxArrayIter` can be constructed via `iter_vable_array()`.
    pub vable_array_index: i64,
    /// opencoder.py `self.vref_array = it.iter_vref_array()`.
    pub vref_array_index: i64,
    /// opencoder.py:210 `self.size = vable.total_length +
    /// vref.total_length + 3 + sum(length(f) + 2 for f in framestack)`.
    /// Used by resume-data consumers to pre-size output arrays.
    pub size: i64,
    /// opencoder.py:211,214-217 self.framestack — snapshot byte
    /// offsets in bottom-up order (outermost frame first, innermost
    /// frame last), built by reversing the top-down iterator.
    /// Sixteen inline slots cover the inlined-callee chain so
    /// `SnapshotIterator.__init__` does not spill a 128 B heap per snapshot.
    /// Regex `and`/`or` inlines past eight (`number_from_parts`).
    pub framestack: SmallVec<[usize; 16]>,
    /// Back-reference to `_snapshot_array_data` so callers can
    /// construct fresh `BoxArrayIter` values without rethreading the
    /// buffer. Matches RPython's implicit `main_iter.trace._snapshot_array_data`
    /// access path.
    snapshot_array_data: &'a [u8],
}

impl<'a> SnapshotIterator<'a> {
    /// opencoder.py SnapshotIterator.__init__(main_iter,
    /// snapshot_index).
    pub fn new(
        snapshot_data: &'a [u8],
        snapshot_array_data: &'a [u8],
        snapshot_index: usize,
    ) -> Self {
        let mut it = TopDownSnapshotIterator::new(snapshot_data, snapshot_index);
        let vable_idx = it.vable_array_index;
        let vref_idx = it.vref_array_index;
        let vable_len = BoxArrayIter::new(snapshot_array_data, vable_idx).total_length;
        let vref_len = BoxArrayIter::new(snapshot_array_data, vref_idx).total_length;
        // opencoder.py:210 `size = vable.total_length + vref.total_length + 3`.
        let mut size = vable_len + vref_len + 3;
        let mut framestack = SmallVec::new();
        // opencoder.py:212-213 early return for empty top snapshot.
        if !it.is_empty_snapshot(snapshot_index) {
            // opencoder.py:214-216 `for snapshot_index in it: ...`
            while let Some(snap_idx) = it.next() {
                framestack.push(snap_idx);
                size += it.length(snap_idx, snapshot_array_data) + 2;
            }
            // opencoder.py `self.framestack.reverse()`.
            framestack.reverse();
        }
        SnapshotIterator {
            topdown_snapshot_iter: it,
            vable_array_index: vable_idx,
            vref_array_index: vref_idx,
            size,
            framestack,
            snapshot_array_data,
        }
    }

    /// opencoder.py `it.iter_vable_array()` accessor — returns a
    /// fresh `BoxArrayIter` rather than caching one iterator (Rust
    /// iterators are single-use).
    pub fn iter_vable_array(&self) -> BoxArrayIter<'a> {
        BoxArrayIter::new(self.snapshot_array_data, self.vable_array_index)
    }

    /// opencoder.py `it.iter_vref_array()` accessor.
    pub fn iter_vref_array(&self) -> BoxArrayIter<'a> {
        BoxArrayIter::new(self.snapshot_array_data, self.vref_array_index)
    }

    /// opencoder.py iter_array(snapshot_index).
    pub fn iter_array(&self, snapshot_index: usize) -> BoxArrayIter<'a> {
        self.topdown_snapshot_iter
            .iter_array(snapshot_index, self.snapshot_array_data)
    }

    /// opencoder.py unpack_jitcode_pc(snapshot_index).
    pub fn unpack_jitcode_pc(&self, snapshot_index: usize) -> (i64, i64) {
        self.topdown_snapshot_iter.unpack_jitcode_pc(snapshot_index)
    }

    /// opencoder.py `get(index)` — resolve a tagged value to
    /// the enclosing iterator's fresh box (OpRef in pyre).
    ///
    /// RPython stores `self.main_iter` on the SnapshotIterator at
    /// construction (opencoder.py); pyre's `SnapshotIterator`
    /// does not own a `main_iter` because the legacy
    /// `TraceIterator<'a>` (pre-byte-stream) cannot be constructed
    /// without the `&[OpRc]` slice it walks, and the byte-stream
    /// `ByteTraceIter` carries its own borrow lifetime over the
    /// recorder buffer it walks.
    /// Callers pass the iterator explicitly instead so this helper
    /// stays structurally equivalent to `main_iter._untag(index)`.
    #[allow(dead_code)]
    pub(crate) fn get(&self, tagged: i64, main_iter: &mut ByteTraceIter<'_>) -> OpRef {
        main_iter._untag(tagged).to_opref()
    }

    /// opencoder.py `unpack_array(arr)` — `[self.get(i) for
    /// i in arr]`.  The RPython comment marks it NOT_RPYTHON (tests
    /// only); pyre exposes the same surface so test fixtures and
    /// debug tooling can read a snapshot's frame array without
    /// re-implementing the decode loop.
    #[allow(dead_code)]
    pub(crate) fn unpack_array(
        &self,
        arr: BoxArrayIter<'_>,
        main_iter: &mut ByteTraceIter<'_>,
    ) -> Vec<OpRef> {
        arr.map(|tagged| main_iter._untag(tagged).to_opref())
            .collect()
    }
}

/// opencoder.py update_liveranges(snapshot_index, trace, index,
/// liveranges).
///
/// Walks the snapshot chain starting at the top snapshot at
/// `_snapshot_data[snapshot_index..]` and, for every TAGBOX tagged
/// value in the vable array, the vref array, or any frame's active-
/// box array, writes `liveranges[v] = index`.
///
/// Split into a free function because `get_live_ranges` borrows
/// `self._ops` while simultaneously invoking this; the free-function
/// signature takes the two byte buffers by shared reference so it can
/// coexist with the op-stream walk.
pub fn update_liveranges(
    snapshot_data: &[u8],
    snapshot_array_data: &[u8],
    snapshot_index: usize,
    index: usize,
    liveranges: &mut [usize],
) {
    let mut it = TopDownSnapshotIterator::new(snapshot_data, snapshot_index);
    let vable_idx = it.vable_array_index;
    let vref_idx = it.vref_array_index;
    for tagged in BoxArrayIter::new(snapshot_array_data, vable_idx) {
        update_live_from_tagged(tagged, index, liveranges);
    }
    for tagged in BoxArrayIter::new(snapshot_array_data, vref_idx) {
        update_live_from_tagged(tagged, index, liveranges);
    }
    while let Some(snap_idx) = it.next() {
        for tagged in it.iter_array(snap_idx, snapshot_array_data) {
            update_live_from_tagged(tagged, index, liveranges);
        }
    }
}

/// opencoder.py `_update_liverange(item, index, liveranges)`.
#[inline]
fn update_live_from_tagged(tagged: i64, index: usize, liveranges: &mut [usize]) {
    let (tag, v) = untag(tagged as u32);
    if tag == TAGBOX {
        let idx = v as usize;
        if idx < liveranges.len() {
            liveranges[idx] = index;
        }
    }
}

// ── Trace Recording API (opencoder.py: Trace class) ──

/// opencoder.py `_encode(box)` — caller-provided view of an arg
/// to `record_op*`. RPython discriminates via `isinstance(box,
/// Const|AbstractResOp)`; pyre needs an explicit tag because Rust
/// lacks inheritance dispatch.
///
/// Variants mirror RPython class names 1:1:
/// * `ConstInt` / `ConstFloat` / `ConstPtr` — `Const` subclasses
///   (`history.py/261/307`), payloads match `getint() /
///   getfloatstorage() / getref_base()`.
/// * `ResOp` — RPython `AbstractResOp.get_position()` value. Also
///   covers `AbstractInputArg` because both types expose
///   `get_position()` returning the same `_index` slot (pyre's
///   `OpRef::from_raw(u32)` unifies them).
///
/// Tagged values can also be pre-computed via the free `_encode_*`
/// helpers (`_encode_smallint`, `_encode_bigint`, `_encode_float`,
/// `_encode_ptr`, `_encode_box_position`); callers that already have
/// an `i64` tag (e.g. snapshot writers) use `append_int` directly
/// without going through `Box`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Box {
    /// `history.py` `class ConstInt(Const)`. `_encode` auto-
    /// selects the SMALL_INT tag (`SMALL_INT_START..SMALL_INT_STOP`)
    /// vs the big-int pool based on the value.
    ConstInt(i64),
    /// `history.py` `class ConstFloat(Const)` —
    /// `getfloatstorage()` raw bits; no dedup.
    ConstFloat(u64),
    /// `history.py` `class ConstPtr(Const)` — `getref_base()`
    /// address; `_cached_const_ptr` dedups via `_refs_dict`.
    ConstPtr(u64),
    /// `resoperation.py` `AbstractResOpOrInputArg.get_position()`
    /// — the `_index` slot into the trace's op/inputarg namespace.
    ResOp(u32),
}

impl Box {
    /// Convenience: `Box::ResOp(opref.raw())`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn of_op(op: majit_ir::OpRef) -> Self {
        Box::ResOp(op.raw())
    }
}

#[cfg(test)]
thread_local! {
    static AFTER_ENCODE_HOOK: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
}

/// Run once after the next `_encode`. The recorder test uses this to
/// publish a forwarded ConstPtr slot the way `walk_const_ptr_refs`
/// would during the pool growth inside that encode.
#[cfg(test)]
pub(crate) fn set_after_encode_hook_for_test(hook: Option<fn()>) {
    AFTER_ENCODE_HOOK.with(|cell| cell.set(hook));
}

#[cfg(test)]
fn fire_after_encode_hook() {
    let hook = AFTER_ENCODE_HOOK.with(|cell| cell.get());
    if let Some(hook) = hook {
        hook();
    }
}

/// opencoder.py: Trace — compact trace recording buffer.
///
/// Literal port of `rpython/jit/metainterp/opencoder.py::Trace`. The state
/// shape matches RPython 1:1 so the wire format, snapshot chain, and
/// liverange computations can all track RPython behavior. Field names are
/// preserved with a leading underscore where RPython uses one, to make
/// line-by-line porting unambiguous.
///
/// NOTE: methods `record_op*`, `capture_resumedata`, `create_top_snapshot`,
/// `_encode_descr`, `tag_overflow_imminent`, and `tracing_done` are
/// rebuilt across Phase B2–B11 in small sub-phase commits. B1 lands only
/// the state fields and the lowest-level `append_byte` / `append_int` /
/// `_double_ops` / `length` / `cut_point` / `cut_at` helpers that RPython
/// puts next to the struct definition (opencoder.py).
pub(crate) type TraceRecordBuffer = Trace;

pub struct Trace {
    /// opencoder.py `Trace._ops` — `['\x00'] * INIT_SIZE`, one
    /// `GcArray(Char)`. `_pos` walks forward over it; `_double_ops`
    /// allocates a new array when `_pos + 4 > len`. The encoded length
    /// is `_pos`, not the array length (`opencoder.py` `length`).
    _ops: TraceOpsBuf,
    /// opencoder.py self._pos — next write position into `_ops`.
    pub _pos: usize,
    /// opencoder.py:497 self._count — total count of ops recorded, seeded
    /// to `max_num_inputargs`. Advances for every op including voids.
    pub _count: u32,
    /// opencoder.py:498 self._index — position of resulting resops, seeded
    /// to `max_num_inputargs`. Advances only for non-void ops (the ones
    /// that produce a box-valued result).
    pub _index: u32,
    /// opencoder.py:499 self._start — frozen baseline equal to
    /// `max_num_inputargs`; iterator construction uses this as the start
    /// cursor (see `get_iter`).
    pub _start: u32,
    /// opencoder.py:496 self.max_num_inputargs — cap on the number of
    /// input-arg boxes. Inputarg positions must stay under this value.
    pub max_num_inputargs: u32,
    /// opencoder.py:491,504 self.inputargs — the actual inputarg boxes
    /// after `set_inputargs` is called. Empty until `set_inputargs` runs.
    pub inputargs: Vec<InputArg>,
    /// opencoder.py `Trace._descrs` — `_descrs[0]` is always None; new
    /// descrs append. Host `Vec` for this stage: `DescrRef` is
    /// `Arc<dyn Descr>`, not a GC item. The list becomes a `GcArray` of
    /// descr instances when those objects themselves are nursery GC.
    pub _descrs: Vec<Option<majit_ir::DescrRef>>,
    /// opencoder.py `Trace._refs`. `GcArray` of `GCREF`, one
    /// `OwnerRootGuard` on the array (`items_have_gc_ptrs`). Index 0 is
    /// nullptr. A minor rewrites the slots; `refresh_from_gc` rekeys
    /// `_refs_dict`, which stays a host `IndexMap` until the dict stage.
    pub _refs: trace_bufs::WordArray<usize>,
    /// opencoder.py `Trace._refs_dict`. Key is `id_or_identityhash`,
    /// value is the index into `_refs`. Cleared by `tracing_done`.
    pub _refs_dict: crate::FxIndexMap<u64, u32>,
    /// opencoder.py `Trace._bigints`. `GcArray` of plain words.
    pub _bigints: trace_bufs::WordArray<i64>,
    /// opencoder.py `Trace._bigints_dict`. Host until the dict stage.
    pub _bigints_dict: crate::FxIndexMap<i64, u32>,
    /// opencoder.py `Trace._floats`. `GcArray` of plain words.
    pub _floats: trace_bufs::WordArray<u64>,
    /// opencoder.py `Trace._snapshot_data`. `ListRepr`: header plus
    /// `GcArray(Char)`. `encode_varint_signed` appends.
    pub _snapshot_data: trace_bufs::CharList,
    /// opencoder.py `Trace._snapshot_array_data`. Same `ListRepr` shape.
    /// Index 0 is the empty array (`append_snapshot_array_data_int(0)`).
    pub _snapshot_array_data: trace_bufs::CharList,
    /// opencoder.py:478 self._total_snapshots — monotonic snapshot
    /// counter; bumped in `create_snapshot` / `create_top_snapshot`.
    pub _total_snapshots: u32,
    /// opencoder.py self.tag_overflow — set when `append_int` or
    /// `append_snapshot_*_int` receives a value outside [MIN_VALUE,
    /// MAX_VALUE]. Consulted by `tracing_done` to decide whether to
    /// raise `SwitchToBlackhole(ABORT_TOO_LONG)`.
    pub tag_overflow: bool,
    /// opencoder.py:476-477,479-480 instrumentation counters. `nodict`
    /// tracks `ConstPtrJitCode` entries that hit the opencoder-index
    /// fast path (dict-bypass).
    pub _consts_bigint: u32,
    pub _consts_float: u32,
    pub _consts_ptr: u32,
    pub _consts_ptr_nodict: u32,

    /// Dead-range memoization cache. opencoder.py:469 and :873-883 —
    /// `_deadranges = (-1, None)` means "not computed". Stored as
    /// `Option<(u32, Vec<usize>)>` where the u32 is the `_count` snapshot
    /// at cache time.
    pub _deadranges: Option<(u32, Vec<usize>)>,

    /// opencoder.py:472 `self.metainterp_sd = metainterp_sd`.
    ///
    /// Stores the full static data object so the decode path can read
    /// `metainterp_sd.all_descrs`' contents for a global descr index (the
    /// length both sides classify against is `all_descr_len`). Held as
    /// `Arc<MetaInterpStaticData>` so the Trace can outlive its
    /// creating MetaInterp frame without reshuffling the reference.
    pub metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,

    /// opencoder.py `TraceIterator.__init__` `self.all_descr_len =
    /// len(metainterp_sd.all_descrs)` — the boundary between global and
    /// local descr indices, captured once.
    ///
    /// Upstream reads the live list on both sides of the codec because
    /// `MetaInterpStaticData.finish_setup` builds `all_descrs` exactly once
    /// (`descr.py setup_descrs` walks the caches at translation time), so
    /// encoder and decoder always measure the same number. Pyre's list is
    /// process-global (`descr_registry.rs`) and grows at runtime whenever
    /// `pyjitpl.rs take_back_all_descrs` publishes a longer one, so a live
    /// read would let the decoder use a larger boundary than the encoder
    /// did and resolve a local index against the global table — a silently
    /// wrong descr. Capturing the boundary per trace restores, for that
    /// trace, the frozen-list property upstream relies on.
    ///
    /// The captured value never exceeds the live length: the published list
    /// only ever grows, as `take_back_all_descrs` documents.
    pub all_descr_len: u32,
}

/// One recorded op in `_ops` without a `cls()` object.
///
/// `TraceIterator.next` is the materialize walk; this is the skip
/// walk used to patch a named guard's descr slot or recover a
/// GetfieldGcR's args before that `cls()` exists.
pub(crate) struct EncodedOp {
    pub opcode: OpCode,
    pub args_pos: usize,
    pub arity: usize,
    pub descr_index: i64,
    /// Pre-bump `_index` of this op (`record_op` return). Void ops
    /// share it with the next value-producing op.
    pub box_index: u32,
}

impl BaseTrace for Trace {}

impl Trace {
    /// opencoder.py `Trace.__init__(self, max_num_inputargs,
    /// metainterp_sd)`.
    pub fn new(
        max_num_inputargs: u32,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) -> Self {
        // opencoder.py `self._pos = 0` initial, then :500 `self._pos
        // = max_num_inputargs` as the final assignment. The first
        // `max_num_inputargs` bytes of `_ops` are reserved as placeholder
        // territory that `TraceIterator.pos = start` walks past —
        // iteration and all write positions operate in a unified
        // [max_num_inputargs, _pos) range. Size the buffer to hold that
        // prefix: `INIT_SIZE` alone is 4096, and a virtualizable array
        // can reserve more inputargs than that
        // (`create_empty_history` after `initialize_virtualizable`).
        let all_descr_len = metainterp_sd.all_descrs().lock().len() as u32;
        let mut t = Trace {
            _ops: TraceOpsBuf::new(INIT_SIZE.max(max_num_inputargs as usize)),
            _pos: max_num_inputargs as usize,
            _count: max_num_inputargs,
            _index: max_num_inputargs,
            _start: max_num_inputargs,
            max_num_inputargs,
            inputargs: Vec::new(),
            // opencoder.py — `_descrs = [None]` so index 0 is reserved.
            // Stays host: items are `DescrRef`, not GC objects yet.
            _descrs: vec![None],
            // opencoder.py `Trace._refs = [nullptr]`. Items array starts at
            // 32 slots so the first ConstPtrs do not reallocate.
            _refs: trace_bufs::new_refs(),
            _refs_dict: crate::FxIndexMap::with_capacity_and_hasher(32, Default::default()),
            _bigints: trace_bufs::new_bigints(),
            _bigints_dict: crate::FxIndexMap::default(),
            _floats: trace_bufs::new_floats(),
            // `ListRepr` items reserved at 128 chars (`ll_newlist_hint`).
            _snapshot_data: trace_bufs::new_snapshot(),
            _snapshot_array_data: trace_bufs::new_snapshot(),
            _total_snapshots: 0,
            tag_overflow: false,
            _consts_bigint: 0,
            _consts_float: 0,
            _consts_ptr: 0,
            _consts_ptr_nodict: 0,
            _deadranges: None,
            metainterp_sd,
            all_descr_len,
        };
        // opencoder.py — `append_snapshot_array_data_int(0)` so all
        // zero-length arrays share index 0.
        t.append_snapshot_array_data_int(0);
        t
    }

    /// opencoder.py set_inputargs(inputargs).
    ///
    /// Records the actual inputarg boxes. Positions must be < max_num_inputargs
    /// (checked in debug mode only, mirroring RPython's `we_are_translated`
    /// guard on the check).
    pub fn set_inputargs(&mut self, inputargs: Vec<InputArg>) {
        debug_assert!(
            inputargs.iter().all(|ia| ia.index < self.max_num_inputargs),
            "inputarg position exceeds max_num_inputargs"
        );
        self.inputargs = inputargs;
    }

    /// Append a single inputarg, returning its positional OpRef.
    ///
    /// Adapter matching pyre `recorder::Trace::record_input_arg`. Required
    /// because pyre's `setup_tracing` / `start_retrace` build the inputarg
    /// list incrementally from the live-value enumeration, whereas RPython
    /// always hands `set_inputargs` a fully-constructed list built by
    /// `MetaInterp.create_empty_loop`.
    ///
    /// Panics if `max_num_inputargs` would be exceeded — the byte-stream
    /// `_start`/`_count`/`_index` fields are seeded against this cap at
    /// `Trace::new`, so callers must size the cap correctly.
    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        let index = self.inputargs.len() as u32;
        assert!(
            index < self.max_num_inputargs,
            "record_input_arg: inputarg count ({}) exceeds max_num_inputargs ({}) — \
             Trace::new's cap was too small",
            index + 1,
            self.max_num_inputargs
        );
        self.inputargs.push(InputArg::from_type(tp, index));
        OpRef::input_arg_typed(index, tp)
    }

    /// Number of inputargs registered so far.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Inputarg slice accessor — RPython parity with `Trace.inputargs`.
    pub fn inputargs(&self) -> &[InputArg] {
        &self.inputargs
    }

    /// Inputarg types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.inputargs.iter().map(|ia| ia.tp.get()).collect()
    }

    /// history.py `length`: number of non-inputarg ops recorded so far.
    /// = `_count - max_num_inputargs`. Compared against
    /// `warmstate.trace_limit` by `MetaInterp.blackhole_if_trace_too_long`
    /// (pyjitpl.py).
    pub fn num_ops(&self) -> usize {
        (self._count - self.max_num_inputargs) as usize
    }

    // ── read API via ByteTraceIter ─────────────────────────────────────
    // RPython has no direct equivalent of these methods: opencoder.py
    // exposes only `get_iter()` and callers iterate. Pyre adds thin
    // walk-to-collect helpers so consumers migrating off
    // `recorder::Trace.ops()` / `get_op_by_pos` / `last_op` can keep the
    // same call shapes; each helper is a single ByteTraceIter walk.

    /// Materialize every recorded op in order. O(n) byte walk.
    /// `opencoder.py TraceIterator.next` returns the same ResOperation
    /// installed in `_cache`: cloning the payload here would split a
    /// returned producer from the object its consumers' arguments reference.
    pub fn ops(&self) -> Vec<OpRc> {
        self.get_byte_iter().collect()
    }

    /// Return the first materialized op whose `.pos == pos`. O(n) byte
    /// walk; `None` if no op claims that position.
    pub fn get_op_by_pos(&self, pos: OpRef) -> Option<OpRc> {
        self.get_byte_iter().find(|op| op.pos().get() == pos)
    }

    /// Return the last recorded op, or `None` if the trace is empty.
    /// O(n) byte walk — opencoder's byte layout has no back-pointer.
    pub fn last_op(&self) -> Option<OpRc> {
        self.get_byte_iter().last()
    }

    /// opencoder.py `_double_ops` — a new array twice as long. The
    /// previous array dies once the root is replaced.
    fn _double_ops(&mut self) {
        self._ops.double();
    }

    /// Grow `_ops` until `extra` more bytes fit.
    ///
    /// `record_bytes` calls this before it snapshots a `ConstPtr`
    /// address, so the collection inside `_double_ops` runs while the
    /// recorder still holds those indexes.
    pub(crate) fn reserve_ops_bytes(&mut self, extra: usize) {
        while self._pos.saturating_add(extra) > self._ops.len {
            self._double_ops();
        }
    }

    /// Char items of `_ops`. Re-reads the root; do not hold the slice
    /// across an allocation.
    pub fn ops_bytes(&self) -> &[u8] {
        self._ops.bytes()
    }

    /// `TraceIterator._nextbyte` over `[0, _pos)`.
    pub fn bytes_read_by_iterator(&self) -> Vec<u8> {
        let mut it = ByteTraceIter::new(self, 0, self._pos, 0);
        let mut out = Vec::with_capacity(self._pos);
        while !it.done() {
            out.push(it._nextbyte());
        }
        out
    }

    /// opencoder.py append_byte(c) — write a single byte and
    /// advance `_pos`. Doubles the buffer if needed.
    pub fn append_byte(&mut self, c: u8) {
        while self._pos >= self._ops.len {
            self._double_ops();
        }
        let pos = self._pos;
        self._ops.bytes_mut()[pos] = c;
        self._pos += 1;
    }

    /// opencoder.py append_int(i). Writes a signed varint into
    /// `_ops` at `_pos` using the same 2-or-4 byte layout as
    /// `encode_varint_signed`, but inlined so the buffer-doubling check
    /// happens before the write (RPython checks `_pos + 4 > len(_ops)`
    /// then writes in place). Doubling loops until the reserved
    /// `_pos = max_num_inputargs` prefix fits. Out-of-range values trip
    /// `tag_overflow` and encode 0.
    pub fn append_int(&mut self, i: i64) {
        let mut v = i;
        if !(MIN_VALUE..=MAX_VALUE).contains(&v) {
            self.tag_overflow = true;
            v = 0;
        }
        while self._pos + 4 > self._ops.len {
            self._double_ops();
        }
        let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
            0x80
        } else {
            0
        };
        // Re-read after `_double_ops`. The writes below do not allocate.
        let buf = self._ops.bytes_mut();
        let mut pos = self._pos;
        buf[pos] = ((v & 0b0111_1111) as u8) | flag;
        pos += 1;
        v >>= 7;
        buf[pos] = (v & 0xff) as u8;
        pos += 1;
        if flag != 0 {
            v >>= 8;
            pos += 1;
            buf[pos - 1] = (v & 0xff) as u8;
            v >>= 8;
            buf[pos] = (v & 0xff) as u8;
            pos += 1;
        }
        self._pos = pos;
    }

    /// opencoder.py tag_overflow_imminent — returns true once
    /// the trace byte stream has grown past 80% of `MAX_VALUE`.
    ///
    /// Checked by the tracer so it can force-close the trace (via the
    /// segmented-trace path) BEFORE any `append_int` actually trips
    /// `tag_overflow`. `tracing_done()` converts `tag_overflow == true`
    /// into `AbortReason::TooLong`, which lands the caller in the
    /// blackhole — calling this `imminent` check gives a graceful exit
    /// ahead of that hard abort.
    pub fn tag_overflow_imminent(&self) -> bool {
        (self._pos as i64) > (MAX_VALUE as f64 * 0.8) as i64
    }

    /// opencoder.py length — the encoded trace length is `_pos`,
    /// NOT `_ops.len()` (the buffer is pre-allocated with padding).
    pub fn length(&self) -> usize {
        self._pos
    }

    /// opencoder.py `cut_point` — 5-tuple `(_pos, _count, _index,
    /// len(_snapshot_data), len(_snapshot_array_data))` returned as a
    /// `TracePosition` struct so `cut_at` can restore all five fields
    /// atomically (RPython uses a bare tuple; the Rust struct carries the
    /// same shape plus field names).
    pub fn cut_point(&self) -> crate::recorder::TracePosition {
        crate::recorder::TracePosition {
            _pos: self._pos,
            _count: self._count,
            _index: self._index,
            snapshot_data_len: self._snapshot_data.len(),
            snapshot_array_data_len: self._snapshot_array_data.len(),
        }
    }

    /// opencoder.py `cut_at(end)` — restore the first three
    /// fields of the cut_point tuple. The snapshot lengths are
    /// intentionally NOT truncated in RPython (they grow monotonically
    /// for a single trace; bridge compilation handles snapshot reuse
    /// separately).
    pub fn cut_at(&mut self, end: crate::recorder::TracePosition) {
        self._pos = end._pos;
        self._count = end._count;
        // opencoder.py `assert index >= 0` — guarded here by the
        // `u32` type of `_index`.
        self._index = end._index;
        // `end.snapshot_data_len` / `end.snapshot_array_data_len` are
        // ignored by `cut_at` in RPython — snapshot chains grow
        // monotonically for a single trace. Parity preserved.
    }

    /// opencoder.py `cut_trace_from((start, count, index, _, _),
    /// inputargs)` — produces a `CutTrace` view over this trace's byte
    /// stream that iterates from the cut_point onward, treating
    /// `inputargs` as the new iterator-local inputargs.
    ///
    /// Used by bridge compilation: when a guard fails at some cut_point,
    /// the tracer resumes a new trace from that point, with the
    /// guard's fail_args (which reference ops recorded before the cut)
    /// acting as fresh inputargs for the resumed trace.
    pub fn cut_trace_from(
        &mut self,
        cut: crate::recorder::TracePosition,
        inputargs: Vec<OpRef>,
    ) -> CutTrace<'_> {
        CutTrace {
            trace: self,
            start: cut._pos,
            count: cut._count,
            index: cut._index,
            inputargs,
        }
    }

    /// opencoder.py tracing_done() — finalize the trace.
    ///
    /// Returns `Err(AbortReason::TooLong)` if `tag_overflow` was set
    /// while recording — any `append_int` that saw a value outside
    /// `[MIN_VALUE, MAX_VALUE]` trips it (opencoder.py). This
    /// mirrors RPython's `raise SwitchToBlackhole(Counters.ABORT_TOO_LONG)`
    /// path (opencoder.py) and lets the caller route into the
    /// blackhole.
    ///
    /// On success, clears the `_bigints_dict` / `_refs_dict` dedup
    /// tables (opencoder.py) — once tracing is done, no further
    /// constants will be interned and the encode hot-path dictionaries
    /// can be reclaimed. The stored `_bigints` / `_refs` Vecs are kept
    /// so `TraceIterator` can still resolve encoded constants back to
    /// values. RPython also emits debug-log lines through
    /// `debug_start`/`debug_print`/`debug_stop`; pyre skips that —
    /// `MAJIT_LOG` covers equivalent telemetry from other hooks.
    pub fn tracing_done(&mut self) -> Result<(), crate::pyjitpl::AbortReason> {
        if self.tag_overflow {
            return Err(crate::pyjitpl::AbortReason::TooLong);
        }
        self._bigints_dict.clear();
        self._refs_dict.clear();
        Ok(())
    }

    // ── Box encoding (opencoder.py _encode + _cached_const_*) ──

    /// opencoder.py _encode for ConstInt that fits in SMALL_INT
    /// range — return `tag(TAGINT, value)`.
    ///
    /// Values outside [SMALL_INT_START, SMALL_INT_STOP) must route
    /// through `_encode_bigint` because the signed varint can't pack
    /// them in the TAGINT namespace.
    pub fn _encode_smallint(value: i64) -> i64 {
        debug_assert!(
            (SMALL_INT_START..SMALL_INT_STOP).contains(&value),
            "value {value} out of SMALL_INT range — use _encode_bigint"
        );
        // RPython `tag(TAGINT, value)` evaluates with Python's
        // arbitrary-precision signed ints — `(-1) << 2 | 0 == -4`.
        // The `u32`-based `tag()` helper works for snapshot array
        // positions where values are always non-negative, but
        // SMALL_INT values are signed and `value as u32` zero-extends
        // negatives into a 32-bit positive that exceeds MAX_VALUE on
        // `append_int`. Use i64 arithmetic so the sign survives the
        // encode/decode round trip — `_untag`'s `tagged >> TAG_SHIFT`
        // is already an i64 arithmetic shift.
        (value << TAG_SHIFT) | (TAGINT as i64)
    }

    /// opencoder.py _encode for ConstInt that does NOT fit in
    /// SMALL_INT range — interns into `_bigints` (dedup via
    /// `_bigints_dict`) and returns `tag(TAGCONSTOTHER, (idx << 1))`.
    /// Bit 0 of `idx << 1` is 0 → bigint; bit 0 == 1 → float.
    pub fn _encode_bigint(&mut self, value: i64) -> i64 {
        self._consts_bigint += 1;
        let v = if let Some(&idx) = self._bigints_dict.get(&value) {
            idx as u64
        } else {
            let idx = self._bigints.len() as u64;
            self._bigints.push(value);
            self._bigints_dict.insert(value, idx as u32);
            idx
        };
        tag(TAGCONSTOTHER, (v << 1) as u32) as i64
    }

    /// opencoder.py _encode for ConstFloat — no caching; appends
    /// to `_floats` and returns `tag(TAGCONSTOTHER, (idx << 1) | 1)`.
    /// Bit 0 == 1 flags the float path on decode.
    pub fn _encode_float(&mut self, raw: u64) -> i64 {
        self._consts_float += 1;
        let idx = self._floats.len() as u32;
        self._floats.push(raw);
        tag(TAGCONSTOTHER, (idx << 1) | 1) as i64
    }

    /// opencoder.py `_cached_const_ptr` + `_encode` for ConstPtr.
    /// Dedup via `_refs_dict` keyed by `id_or_identityhash`, push the
    /// address to `_refs`, return `tag(TAGCONSTPTR, idx)`. Index 0 is
    /// nullptr (seeded by the constructor).
    ///
    /// The array is one rooted `GcArray` of `GCREF`. The dict key is the
    /// identity hash, so a minor that forwards `_refs` does not move the
    /// key. `opencoder.py` `_refs_dict` is keyed by the ref itself, whose
    /// hash is `lltype.identityhash`.
    pub fn _encode_ptr(&mut self, addr: u64) -> i64 {
        self._consts_ptr += 1;
        if addr == 0 {
            return tag(TAGCONSTPTR, 0) as i64;
        }
        let key = majit_ir::gc_id_or_identityhash(addr as usize) as u64;
        let cached = self
            ._refs_dict
            .get(&key)
            .copied()
            .or_else(|| self._refs_dict.get(&(addr as u64)).copied())
            .filter(|&idx| self._refs.as_slice().get(idx as usize).copied() == Some(addr as usize));
        let v = if let Some(idx) = cached {
            idx
        } else {
            let stored = self._refs.push(addr as usize);
            let idx = (self._refs.len() - 1) as u32;
            self._refs_dict.insert(key, idx);
            if key != stored as u64 {
                self._refs_dict.insert(stored as u64, idx);
            }
            idx
        };
        tag(TAGCONSTPTR, v) as i64
    }

    /// opencoder.py _encode for AbstractResOp — boxes that live
    /// in the trace's own op stream. Position is the box's `_index`
    /// slot from record time.
    pub fn _encode_box_position(position: u32) -> i64 {
        tag(TAGBOX, position) as i64
    }

    /// opencoder.py `_encode(box)` — dispatch a `Box` to the
    /// appropriate tagging helper and return the encoded `i64` tag.
    ///
    /// Mirrors RPython's `isinstance(box, Const|AbstractResOp)` chain.
    /// `ConstInt` further splits on the SMALL_INT range
    /// (opencoder.py vs 609-622).
    pub(crate) fn _encode(&mut self, b: Box) -> i64 {
        let tagged = match b {
            // opencoder.py:605-608 ConstInt within SMALL_INT range.
            Box::ConstInt(v) if (SMALL_INT_START..SMALL_INT_STOP).contains(&v) => {
                Self::_encode_smallint(v)
            }
            // opencoder.py:609-622 ConstInt outside SMALL_INT range.
            Box::ConstInt(v) => self._encode_bigint(v),
            // opencoder.py:623-628 ConstFloat.
            Box::ConstFloat(raw) => self._encode_float(raw),
            // opencoder.py:629-632 ConstPtr.
            Box::ConstPtr(addr) => self._encode_ptr(addr),
            // opencoder.py:633-638 AbstractResOp.get_position().
            Box::ResOp(p) => Self::_encode_box_position(p),
        };
        // A pool push above can minor-collect. The test hook stands in
        // for `walk_const_ptr_refs` updating a later ConstPtr slot
        // before the next argument is resolved.
        #[cfg(test)]
        fire_after_encode_hook();
        tagged
    }

    // ── Snapshot writers (opencoder.py _list_of_boxes) ──

    /// opencoder.py new_array(lgt).
    ///
    /// Length-0 arrays always return index 0 (the pre-seeded empty
    /// entry from the constructor). Non-zero lengths write the length
    /// as the first varint at the tail of `_snapshot_array_data` and
    /// return the offset before the write; callers then write the
    /// array contents via `_add_box_to_storage`.
    pub fn new_array(&mut self, lgt: usize) -> i64 {
        if lgt == 0 {
            return 0;
        }
        let res = self._snapshot_array_data.len() as i64;
        self.append_snapshot_array_data_int(lgt as i64);
        res
    }

    /// opencoder.py _add_box_to_storage(tagged_box).
    ///
    /// Appends a pre-encoded tagged box value to `_snapshot_array_data`.
    /// The caller computes the tag via the `_encode_*` helpers above.
    pub fn _add_box_to_storage(&mut self, tagged: i64) {
        self.append_snapshot_array_data_int(tagged);
    }

    /// opencoder.py `_add_box_to_storage(box)` — RPython takes
    /// the Box object directly and runs `self._encode(box)` inline.
    /// `_add_box_to_storage` (above) is the pyre-adapter tagged-int
    /// path used by callers that already have a tagged value; this
    /// method matches the RPython call shape for callers that hold a
    /// `Box` (e.g. `MIFrame::get_list_of_active_boxes`).
    pub(crate) fn _add_box_to_storage_box(&mut self, b: Box) {
        let tagged = self._encode(b);
        self.append_snapshot_array_data_int(tagged);
    }

    /// opencoder.py _list_of_boxes(boxes). Writes an array
    /// header then each box, returns the array index (0 for empty).
    pub fn _list_of_boxes(&mut self, boxes: &[i64]) -> i64 {
        let res = self.new_array(boxes.len());
        for &b in boxes {
            self._add_box_to_storage(b);
        }
        res
    }

    /// opencoder.py _list_of_boxes_virtualizable(boxes).
    ///
    /// The virtualizable lives at the end of the locals_cells_stack in
    /// the tracer's view, but the snapshot encoding wants it at the
    /// FRONT so the resume reader can pull it out before the local
    /// frame slots. This reorders `[a, b, c, vable]` to `[vable, a, b, c]`.
    pub fn _list_of_boxes_virtualizable(&mut self, boxes: &[i64]) -> i64 {
        if boxes.is_empty() {
            return self.new_array(0);
        }
        let res = self.new_array(boxes.len());
        // boxes[-1] first (the virtualizable), then boxes[:-1].
        self._add_box_to_storage(*boxes.last().unwrap());
        for &b in &boxes[..boxes.len() - 1] {
            self._add_box_to_storage(b);
        }
        res
    }

    /// opencoder.py _encode_snapshot(index, pc, array, is_last).
    ///
    /// Writes jitcode_index + pc + array_index + prev-sentinel into
    /// `_snapshot_data`. `prev` is either `SNAPSHOT_PREV_NONE` for the
    /// last snapshot in a chain (end of framestack) or
    /// `SNAPSHOT_PREV_NEEDS_PATCHING` for internal snapshots — those
    /// get patched later via `snapshot_add_prev`.
    ///
    /// Returns the snapshot_index (offset into `_snapshot_data` before
    /// the write).
    pub(crate) fn _encode_snapshot(
        &mut self,
        index: i64,
        pc: i64,
        array: i64,
        is_last: bool,
    ) -> i64 {
        let res = self._snapshot_data.len() as i64;
        self.append_snapshot_data_int(index);
        self.append_snapshot_data_int(pc);
        self.append_snapshot_data_int(array);
        let prev = if is_last {
            SNAPSHOT_PREV_NONE as i64
        } else {
            SNAPSHOT_PREV_NEEDS_PATCHING as i64
        };
        self.append_snapshot_data_int(prev);
        res
    }

    /// opencoder.py snapshot_add_prev(prev).
    ///
    /// Overwrites the trailing prev-sentinel slot in `_snapshot_data`
    /// (written as `SNAPSHOT_PREV_NEEDS_PATCHING` by the previous
    /// `_encode_snapshot`) with the actual prev index. Called from
    /// `create_snapshot` (SNAPSHOT_PREV_COMES_NEXT) and
    /// `_ensure_parent_resumedata` (the real parent snapshot index).
    ///
    /// INVARIANT: the last 2 bytes of `_snapshot_data` must be the
    /// signed-varint encoding of `SNAPSHOT_PREV_NEEDS_PATCHING` (-3).
    /// `-3` fits in 2 bytes (RPython literally asserts the bytes
    /// `'}' '\xff'` at opencoder.py). We assert the same.
    #[allow(dead_code)]
    pub(crate) fn snapshot_add_prev(&mut self, prev: i32) {
        debug_assert!(self._snapshot_data.len() >= 2, "snapshot_data too short");
        let n = self._snapshot_data.len();
        debug_assert_eq!(
            self._snapshot_data[n - 2],
            b'}',
            "prev slot byte 0 not \\x7d (NEEDS_PATCHING marker)"
        );
        debug_assert_eq!(
            self._snapshot_data[n - 1],
            0xffu8,
            "prev slot byte 1 not \\xff (NEEDS_PATCHING marker)"
        );
        self._snapshot_data.pop();
        self._snapshot_data.pop();
        self.append_snapshot_data_int(prev as i64);
    }

    // ── Snapshot chain entry points (opencoder.py create_top_snapshot) ──

    /// opencoder.py create_top_snapshot(frame, vable_boxes,
    /// vref_boxes, after_residual_call, is_last).
    ///
    /// The caller has already computed the active-boxes array via
    /// `new_array` + `_add_box_to_storage` calls (that's the `array`
    /// parameter here — RPython does this inline through the frame's
    /// `get_list_of_active_boxes` callback). vable_boxes / vref_boxes
    /// are pre-encoded tagged values; this writes their arrays into
    /// `_snapshot_array_data`, then the top snapshot record into
    /// `_snapshot_data`, and finally patches the preceding guard's
    /// descr slot to the snapshot index.
    #[allow(dead_code)]
    pub(crate) fn create_top_snapshot(
        &mut self,
        jitcode_index: i64,
        pc: i64,
        array: i64,
        vable_boxes: &[i64],
        vref_boxes: &[i64],
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let vable_array = self._list_of_boxes_virtualizable(vable_boxes);
        let vref_array = self._list_of_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(jitcode_index, pc, array, is_last);
        // Patch the guard's trailing descr slot to the snapshot index.
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py create_empty_top_snapshot(vable_boxes, vref_boxes).
    ///
    /// No frame data — used when tracing starts at a guard before any
    /// frame has been entered. Writes a snapshot with jitcode_index=-1,
    /// pc=0, an empty box array, and SNAPSHOT_PREV_NONE.
    #[allow(dead_code)]
    pub(crate) fn create_empty_top_snapshot(
        &mut self,
        vable_boxes: &[i64],
        vref_boxes: &[i64],
    ) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let empty_array = self._list_of_boxes(&[]);
        let vable_array = self._list_of_boxes_virtualizable(vable_boxes);
        let vref_array = self._list_of_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(-1, 0, empty_array, true);
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py create_snapshot(frame, is_last).
    ///
    /// Called while walking the framestack toward the outermost frame
    /// to capture parent resumedata. Writes a `SNAPSHOT_PREV_COMES_NEXT`
    /// sentinel on the PREVIOUS snapshot (via `snapshot_add_prev`),
    /// then encodes this snapshot with `SNAPSHOT_PREV_NEEDS_PATCHING`
    /// (to be resolved on the NEXT iteration).
    #[allow(dead_code)]
    pub(crate) fn create_snapshot(
        &mut self,
        jitcode_index: i64,
        pc: i64,
        array: i64,
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        self.snapshot_add_prev(SNAPSHOT_PREV_COMES_NEXT);
        self._encode_snapshot(jitcode_index, pc, array, is_last)
    }

    /// opencoder.py `TraceIterator.get_snapshot_iter(index)`
    /// — pyre hosts this on the Trace itself since the pre-byte-stream
    /// `TraceIterator` does not yet hold a `&TraceRecordBuffer`.
    ///
    /// Returns an eager `SnapshotIterator` that has already reversed
    /// the snapshot chain rooted at `snapshot_index` and captured the
    /// vable / vref array offsets. `snapshot_index` is the byte offset
    /// returned by `capture_resumedata` / `create_top_snapshot`.
    #[allow(dead_code)]
    pub(crate) fn get_snapshot_iter(&self, snapshot_index: usize) -> SnapshotIterator<'_> {
        SnapshotIterator::new(
            &self._snapshot_data,
            &self._snapshot_array_data,
            snapshot_index,
        )
    }

    /// opencoder.py `create_top_snapshot(frame, vable_boxes,
    /// vref_boxes, after_residual_call, is_last)` — frame-based
    /// wrapper that calls `frame.get_list_of_active_boxes` to produce
    /// the innermost-frame box array, then emits the top snapshot
    /// record.
    ///
    /// Matches the RPython call shape 1:1: the caller provides the
    /// `frame` and virtualizable / virtualref box arrays; the trace
    /// takes care of writing the active-box array, vable / vref
    /// arrays, and the snapshot record header. `op_live` and
    /// `all_liveness` are threaded through from the caller's
    /// `MetaInterpStaticData` (RPython reads them off
    /// `self.metainterp_sd` inline; pyre passes explicitly to avoid
    /// an overlapping borrow on `metainterp_sd` during the write).
    ///
    /// `in_a_call` is forced to `false` at the topmost frame — the
    /// RPython invariant at opencoder.py.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    #[allow(dead_code)]
    pub(crate) fn create_top_snapshot_from_frame(
        &mut self,
        frame: &mut crate::pyjitpl::MIFrame,
        vable_boxes: impl ExactSizeIterator<Item = Box> + DoubleEndedIterator,
        vref_boxes: impl ExactSizeIterator<Item = Box>,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
        is_last: bool,
        patch_guard_descr: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        // opencoder.py:769 frame.get_list_of_active_boxes(False, ...).
        // The topmost frame always uses `in_a_call=false`, so the
        // register-clear path is never entered — pass false.
        let array = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            self,
            /* clear_result_register */ false,
            op_live,
            all_liveness,
            after_residual_call,
        );
        // opencoder.py:771-780 — write vable / vref arrays, then the
        // snapshot record. `s` captures the snapshot_data offset
        // before any writes so `patch_last_guard_descr_slot` can
        // patch the preceding guard's 0-placeholder slot.
        let s = self._snapshot_data.len() as i64;
        let vable_array = self._list_of_boxes_virtualizable_from_boxes(vable_boxes);
        let vref_array = self._list_of_boxes_from_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        let jitcode_index = frame.jitcode.try_index().map(|i| i as i64).unwrap_or(-1);
        let pc = frame.pc as i64;
        self._encode_snapshot(jitcode_index, pc, array, is_last);
        // `create_top_snapshot` writes the snapshot byte offset into the
        // guard's trailing descr slot (`opencoder.py` `_pos -= 2`).
        if patch_guard_descr {
            self.patch_last_guard_descr_slot(s);
        }
        s
    }

    /// opencoder.py `create_snapshot(frame, is_last)` —
    /// frame-based wrapper for parent frames in a chain. Writes
    /// `SNAPSHOT_PREV_COMES_NEXT` on the previous snapshot (via
    /// `snapshot_add_prev`), then emits this snapshot with
    /// `SNAPSHOT_PREV_NEEDS_PATCHING`.
    ///
    /// `in_a_call=true` is forced on parent frames (pyjitpl.py:808):
    /// the paired `get_list_of_active_boxes` clears the in-flight
    /// CALL's result register to a zero inline-Const, mirroring
    /// RPython's `self.registers_i[index] = history.CONST_FALSE`
    /// mutation line-by-line.  When `clear_result_register` is `false`
    /// (test fixtures) the fallback substitutes `Box::Const*(0)`
    /// directly into the snapshot bytes — identical encoded output but
    /// the register slot keeps its pre-call contents.
    #[allow(dead_code)]
    pub(crate) fn create_snapshot_from_frame(
        &mut self,
        frame: &mut crate::pyjitpl::MIFrame,
        clear_result_register: bool,
        op_live: u8,
        all_liveness: &[u8],
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        let array = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            self,
            clear_result_register,
            op_live,
            all_liveness,
            /* after_residual_call */ false,
        );
        self.snapshot_add_prev(SNAPSHOT_PREV_COMES_NEXT);
        let jitcode_index = frame.jitcode.try_index().map(|i| i as i64).unwrap_or(-1);
        let pc = frame.pc as i64;
        self._encode_snapshot(jitcode_index, pc, array, is_last)
    }

    /// opencoder.py `capture_resumedata(framestack,
    /// virtualizable_boxes, virtualref_boxes, after_residual_call)`.
    ///
    /// Top-level resumedata capture: emits the innermost frame's top
    /// snapshot, then walks parent frames via
    /// `_ensure_parent_resumedata`. Returns the snapshot_index of
    /// the top snapshot (matches RPython `return result`).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    #[allow(dead_code)]
    pub(crate) fn capture_resumedata(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        virtualizable_boxes: &[Box],
        virtualref_boxes: &[Box],
        clear_result_register: bool,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> i64 {
        self.capture_resumedata_mapped(
            framestack,
            virtualizable_boxes.iter().copied(),
            virtualref_boxes.iter().copied(),
            clear_result_register,
            op_live,
            all_liveness,
            after_residual_call,
            true,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub(crate) fn capture_resumedata_mapped(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        virtualizable_boxes: impl ExactSizeIterator<Item = Box> + DoubleEndedIterator,
        virtualref_boxes: impl ExactSizeIterator<Item = Box>,
        clear_result_register: bool,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
        patch_guard_descr: bool,
    ) -> i64 {
        // opencoder.py `n = len(framestack) - 1`.
        let framestack_len = framestack.len();
        if framestack_len >= 1 {
            let n = framestack_len - 1;
            // opencoder.py `top = framestack[n]`. Split the
            // framestack so the topmost frame can be borrowed
            // mutably while `_ensure_parent_resumedata` later walks
            // parents.
            let result = {
                let top = &mut framestack[n];
                self.create_top_snapshot_from_frame(
                    top,
                    virtualizable_boxes,
                    virtualref_boxes,
                    op_live,
                    all_liveness,
                    after_residual_call,
                    /* is_last */ n == 0,
                    patch_guard_descr,
                )
            };
            // opencoder.py self._ensure_parent_resumedata(framestack, n).
            // The parent walk uses `in_a_call=true`, so the clear flag is
            // threaded through to the per-frame register-clear path.
            self._ensure_parent_resumedata(
                framestack,
                n,
                clear_result_register,
                op_live,
                all_liveness,
            );
            result
        } else {
            // opencoder.py:829-831 — empty framestack → empty top
            // snapshot.
            let _ = clear_result_register;
            self.create_empty_top_snapshot_from_boxes(
                virtualizable_boxes,
                virtualref_boxes,
                patch_guard_descr,
            )
        }
    }

    /// opencoder.py `_ensure_parent_resumedata(framestack, n)`.
    ///
    /// Walks parent frames backwards from `n-1` down to `0`, emitting
    /// a `create_snapshot` for each. Shortcuts via
    /// `frame.parent_snapshot` memo: if a parent frame has already
    /// produced its snapshot (e.g. re-entrancy, or a prior
    /// `capture_resumedata` in the same trace), patch the previous
    /// snapshot's prev-pointer directly to `frame.parent_snapshot`
    /// and return. Otherwise emit a fresh snapshot and memo it on
    /// the frame.
    pub(crate) fn _ensure_parent_resumedata(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        n: usize,
        clear_result_register: bool,
        op_live: u8,
        all_liveness: &[u8],
    ) {
        let mut n = n;
        while n > 0 {
            // opencoder.py:836-840 memo short-circuit via target frame.
            let target_parent = framestack[n].parent_snapshot;
            if target_parent >= 0 {
                self.snapshot_add_prev(target_parent as i32);
                return;
            }
            // opencoder.py s = self.create_snapshot(back, is_last).
            let is_last = n == 1;
            let s = {
                let back = &mut framestack[n - 1];
                self.create_snapshot_from_frame(
                    back,
                    clear_result_register,
                    op_live,
                    all_liveness,
                    is_last,
                )
            };
            // opencoder.py `target.parent_snapshot = s`.
            framestack[n].parent_snapshot = s;
            n -= 1;
        }
    }

    /// opencoder.py `_list_of_boxes(boxes)` — Box-taking
    /// sibling of the tagged-int `_list_of_boxes`. Used by
    /// `capture_resumedata` and friends which hold `Box` values.
    fn _list_of_boxes_from_boxes(&mut self, boxes: impl ExactSizeIterator<Item = Box>) -> i64 {
        let res = self.new_array(boxes.len());
        for b in boxes {
            self._add_box_to_storage_box(b);
        }
        res
    }

    /// opencoder.py `_list_of_boxes_virtualizable(boxes)` —
    /// Box-taking sibling; reorders `[a, b, c, vable]` to
    /// `[vable, a, b, c]` at encode time.
    fn _list_of_boxes_virtualizable_from_boxes(
        &mut self,
        mut boxes: impl ExactSizeIterator<Item = Box> + DoubleEndedIterator,
    ) -> i64 {
        let res = self.new_array(boxes.len());
        if let Some(last) = boxes.next_back() {
            self._add_box_to_storage_box(last);
        }
        for b in boxes {
            self._add_box_to_storage_box(b);
        }
        res
    }

    /// opencoder.py `create_empty_top_snapshot(vable_boxes,
    /// vref_boxes)` — Box-taking sibling for `capture_resumedata`'s
    /// empty-framestack branch.
    #[allow(dead_code)]
    fn create_empty_top_snapshot_from_boxes(
        &mut self,
        vable_boxes: impl ExactSizeIterator<Item = Box> + DoubleEndedIterator,
        vref_boxes: impl ExactSizeIterator<Item = Box>,
        patch_guard_descr: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let empty_array = self._list_of_boxes_from_boxes(std::iter::empty());
        let vable_array = self._list_of_boxes_virtualizable_from_boxes(vable_boxes);
        let vref_array = self._list_of_boxes_from_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(-1, 0, empty_array, true);
        if patch_guard_descr {
            self.patch_last_guard_descr_slot(s);
        }
        s
    }

    /// opencoder.py `_pos -= 2; self.append_int(s)` — patch the
    /// guard's trailing descr slot to the snapshot index.
    ///
    /// INVARIANT: guards record their trailing descr via `append_int(0)`
    /// (see `_op_end` + opencoder.py). `0` always encodes as
    /// exactly 2 bytes in the RPython signed-varint layout (both bytes
    /// are `\x00`), so `capture_resumedata` / `create_top_snapshot` can
    /// safely rewind `_pos -= 2` and overwrite the slot with the actual
    /// snapshot_index via a fresh `append_int(s)`. The new value may
    /// occupy 2 *or* 4 bytes; `_ops` is pre-grown by `_double_ops` so
    /// both sizes fit.
    ///
    /// RPython inlines this sequence in `create_top_snapshot` and
    /// `create_snapshot`; pyre factors it out so Phase B7's snapshot
    /// code is shorter and the assertion that the slot was indeed a
    /// guard 0-placeholder is always executed.
    ///
    /// `create_top_snapshot` runs while the guard is still the last
    /// recorded op (`pyjitpl.py` `generate_guard` records, then
    /// `capture_resumedata`). A delayed restamp of that same last
    /// guard (the stream slot already holds a snapshot offset) uses
    /// `patch_descr_slot_at` on the named slot instead of rewinding
    /// `_pos`.
    pub(crate) fn last_descr_slot_is_placeholder(&self) -> bool {
        let ops = self.ops_bytes();
        self._pos >= 2 && ops[self._pos - 2] == 0 && ops[self._pos - 1] == 0
    }

    #[allow(dead_code)]
    pub(crate) fn patch_last_guard_descr_slot(&mut self, snapshot_index: i64) {
        debug_assert!(
            self._pos >= 2,
            "patch_last_guard_descr_slot called with _pos < 2"
        );
        debug_assert!(
            self.last_descr_slot_is_placeholder(),
            "guard descr placeholder was not \\x00\\x00"
        );
        self._pos -= 2;
        self.append_int(snapshot_index);
    }

    /// Visit each recorded op in `_ops`[`_start`, `_pos`).
    pub(crate) fn for_each_encoded_op(&self, mut f: impl FnMut(EncodedOp) -> bool) {
        let mut pos = self._start as usize;
        let end = self._pos;
        let mut index = self._start;
        let ops = self.ops_bytes();
        while pos < end {
            let opnum = ops[pos];
            pos += 1;
            let opcode = OpCode::from_u16(opnum as u16)
                .unwrap_or_else(|| panic!("encoded op: unknown opnum {opnum}"));
            let arity = match opcode.arity() {
                Some(n) => n as usize,
                None => {
                    let (v, n) = decode_varint_signed(&ops[pos..]);
                    pos += n;
                    v as usize
                }
            };
            let args_pos = pos;
            for _ in 0..arity {
                let (_, n) = decode_varint_signed(&ops[pos..]);
                pos += n;
            }
            let descr_index = if opcode.has_descr() {
                let (idx, n) = decode_varint_signed(&ops[pos..]);
                pos += n;
                idx
            } else {
                0
            };
            if !f(EncodedOp {
                opcode,
                args_pos,
                arity,
                descr_index,
                box_index: index,
            }) {
                return;
            }
            if opcode.result_type() != Type::Void {
                index += 1;
            }
        }
    }

    /// Tagged arg `i` of an encoded op, starting at `args_pos`.
    pub(crate) fn encoded_arg_at(ops: &[u8], args_pos: usize, i: usize) -> i64 {
        let mut pos = args_pos;
        for _ in 0..i {
            let (_, n) = decode_varint_signed(&ops[pos..]);
            pos += n;
        }
        decode_varint_signed(&ops[pos..]).0
    }

    /// Overwrite a descr varint in the middle of `_ops`. A 2-byte
    /// placeholder can grow to 4 bytes (`append_int`); later ops shift.
    pub(crate) fn patch_descr_slot_at(&mut self, descr_pos: usize, snapshot_index: i64) {
        let mut value = snapshot_index;
        if !(MIN_VALUE..=MAX_VALUE).contains(&value) {
            self.tag_overflow = true;
            value = 0;
        }
        let (_, old_len) = decode_varint_signed(&self.ops_bytes()[descr_pos..]);
        let mut encoded = Vec::with_capacity(4);
        encode_varint_signed(&mut encoded, value);
        let new_len = encoded.len();
        if new_len != old_len {
            let tail = descr_pos + old_len;
            if new_len > old_len {
                let d = new_len - old_len;
                while self._pos + d > self._ops.len {
                    self._double_ops();
                }
                self._ops.bytes_mut().copy_within(tail..self._pos, tail + d);
                self._pos += d;
            } else {
                let d = old_len - new_len;
                self._ops.bytes_mut().copy_within(tail..self._pos, tail - d);
                self._pos -= d;
            }
        }
        self._ops.bytes_mut()[descr_pos..descr_pos + new_len].copy_from_slice(&encoded);
    }

    /// Patch the named guard's trailing descr slot to `snapshot_index`.
    /// The last recorded op being that guard uses
    /// `patch_last_guard_descr_slot` (`create_top_snapshot`).
    /// `TraceIterator.next` descr lookup for a non-guard encoded index.
    pub(crate) fn resolve_descr_index(&self, descr_index: i64) -> Option<majit_ir::DescrRef> {
        if descr_index == 0 {
            return None;
        }
        let all_descr_len = self.all_descr_len as i64;
        if descr_index < all_descr_len + 1 {
            let all_descrs = self.metainterp_sd.all_descrs().lock();
            Some(all_descrs[(descr_index - 1) as usize].clone())
        } else {
            self._descrs[(descr_index - all_descr_len - 1) as usize].clone()
        }
    }

    pub(crate) fn walk_descr_const_ptr_refs(
        &mut self,
        visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
    ) {
        for d in self._descrs.iter().flatten() {
            if let Some(qd) = d.as_quasi_immut_descr() {
                qd.walk_const_ptr_refs(visitor);
            }
        }
    }

    /// opencoder.py `_encode_descr(descr)`.
    ///
    /// Two-tier descr numbering mirroring RPython:
    ///   * Global descrs — `descr.get_descr_index()` inside
    ///     `[0, all_descr_len)`. Encoded as `global_index + 1`, so the
    ///     encoded range is `[1, all_descr_len + 1)`.
    ///   * Local descrs — everything else. Appended to
    ///     `self._descrs` (which starts with a `None` sentinel at [0]
    ///     so index 0 means "no descr") and encoded as
    ///     `all_descr_len + len(_descrs) - 1 + 1`. The TraceIterator
    ///     decode path looks up such descrs via
    ///     `_descrs[descr_index - all_descr_len - 1]`.
    ///
    /// The boundary is `self.all_descr_len`, captured by `Trace::new`, and
    /// the decode path uses that same captured value — so the two stay
    /// exact inverses however the process-global list grows afterwards. A
    /// descr whose global index falls at or past the boundary (stamped
    /// after this trace started recording) is recorded locally: the decoder
    /// would otherwise read such an index as local and resolve it against
    /// `_descrs`.
    pub(crate) fn _encode_descr(&mut self, descr: &majit_ir::DescrRef) -> i64 {
        let all_descr_len = self.all_descr_len as i64;
        let descr_index = descr.get_descr_index() as i64;
        if descr_index >= 0 && descr_index < all_descr_len {
            return descr_index + 1;
        }
        self._descrs.push(Some(descr.clone()));
        // _descrs[0] is the sentinel None; new descrs append from
        // index 1. RPython returns `len(_descrs) - 1 + all_descrs_len
        // + 1`. `len - 1` because index 0 is reserved; we want the
        // local-only offset.
        let local_index = (self._descrs.len() as i64) - 1;
        local_index + all_descr_len + 1
    }

    /// opencoder.py _op_start(opnum, num_argboxes).
    ///
    /// Writes the opnum byte. For fixed-arity ops the decoder reads the
    /// arity from `OpCode::arity()`; for variadic ops (arity == None)
    /// the writer records the actual `num_argboxes` count inline as a
    /// signed varint. Returns the old `_pos` so `_op_end` can patch the
    /// trailing descr int if the opnum turns out to be a guard that gets
    /// its snapshot index filled in later.
    pub(crate) fn _op_start(&mut self, opcode: OpCode, num_argboxes: usize) -> usize {
        let old_pos = self._pos;
        let opnum = opcode.as_u16();
        debug_assert!(opnum <= 0xFF, "opnum {opnum} exceeds 1 byte");
        self.append_byte(opnum as u8);
        match opcode.arity() {
            None => {
                self.append_int(num_argboxes as i64);
            }
            Some(expected) => {
                debug_assert_eq!(
                    num_argboxes, expected as usize,
                    "fixed-arity mismatch for {:?}: expected {expected}, got {num_argboxes}",
                    opcode
                );
            }
        }
        old_pos
    }

    /// opencoder.py _op_end(opnum, descr, old_pos).
    ///
    /// Writes the trailing descr int if the opcode carries one
    /// (`has_descr()`). `descr_index` is the already-encoded descr slot
    /// from `_encode_descr` (Phase B4); guards pass 0 to get a
    /// placeholder that `capture_resumedata` patches to the snapshot
    /// index (Phase B5/B7). Bumps `_count` unconditionally and `_index`
    /// only for non-void ops (opencoder.py).
    pub(crate) fn _op_end(&mut self, opcode: OpCode, descr_index: i64, _old_pos: usize) {
        if opcode.has_descr() {
            self.append_int(descr_index);
        }
        self._count += 1;
        if opcode.result_type() != Type::Void {
            self._index += 1;
        }
    }

    /// opencoder.py `_op_end` (descr-object overload).
    ///
    /// Drops the RPython `descr=None` branch (caller decides: guards and
    /// no-descr ops use `None`, which encodes the 0-placeholder; all
    /// other descrs are routed through `_encode_descr` first). Keeps the
    /// `_count` / `_index` bump parity with `_op_end`.
    fn _op_end_descr(
        &mut self,
        opcode: OpCode,
        descr: Option<&majit_ir::DescrRef>,
        old_pos: usize,
    ) {
        let descr_index = match descr {
            Some(d) => self._encode_descr(d),
            None => 0,
        };
        self._op_end(opcode, descr_index, old_pos);
    }

    /// opencoder.py record_op(opnum, argboxes, descr=None).
    ///
    /// Returns the pre-bump `_index` — the box position that THIS op's
    /// result will occupy if it is non-void (opencoder.py `pos =
    /// self._index`). Each `Box` is passed through `_encode` to
    /// produce the wire-format tagged value. `descr=None` emits the
    /// 0-placeholder (guards and no-descr ops); `Some(&descr)` routes
    /// through `_encode_descr`.
    #[allow(dead_code)]
    pub(crate) fn record_op(
        &mut self,
        opcode: OpCode,
        argboxes: &[Box],
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        self.record_op_resolved(opcode, argboxes.len(), descr, |i| argboxes[i])
    }

    /// `record_op`, resolving one argument at a time.
    ///
    /// `Trace::record_bytes` reads each ConstPtr from `const_ptr_table`
    /// immediately before `_encode`. An earlier argument can grow
    /// `_refs`, `_bigints`, or `_floats` and minor-collect.
    /// `live_const_indexes` forwards the table; a `Box::ConstPtr` built
    /// before that collection still holds the from-space address.
    pub(crate) fn record_op_resolved(
        &mut self,
        opcode: OpCode,
        nargs: usize,
        descr: Option<&majit_ir::DescrRef>,
        mut resolve_arg: impl FnMut(usize) -> Box,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, nargs);
        for i in 0..nargs {
            let tagged = self._encode(resolve_arg(i));
            self.append_int(tagged);
        }
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py record_op0(opnum, descr=None).
    #[allow(dead_code)]
    pub(crate) fn record_op0(&mut self, opcode: OpCode, descr: Option<&majit_ir::DescrRef>) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 0);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py record_op1(opnum, argbox1, descr=None).
    #[allow(dead_code)]
    pub(crate) fn record_op1(
        &mut self,
        opcode: OpCode,
        arg0: Box,
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 1);
        let t0 = self._encode(arg0);
        self.append_int(t0);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py record_op2(opnum, argbox1, argbox2, descr=None).
    #[allow(dead_code)]
    pub(crate) fn record_op2(
        &mut self,
        opcode: OpCode,
        arg0: Box,
        arg1: Box,
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 2);
        let t0 = self._encode(arg0);
        self.append_int(t0);
        let t1 = self._encode(arg1);
        self.append_int(t1);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py record_op3(opnum, argbox1, argbox2, argbox3, descr=None).
    #[allow(dead_code)]
    pub(crate) fn record_op3(
        &mut self,
        opcode: OpCode,
        arg0: Box,
        arg1: Box,
        arg2: Box,
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 3);
        let t0 = self._encode(arg0);
        self.append_int(t0);
        let t1 = self._encode(arg1);
        self.append_int(t1);
        let t2 = self._encode(arg2);
        self.append_int(t2);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// Encode a pyre `OpRef` into the wire `i64` tag.
    ///
    /// Pyre inline-Const OpRefs carry their value directly; this helper
    /// resolves the constant (via `OpRef::inline_const_to_value`) and
    /// routes through the existing `_encode(Box)` path so the resulting
    /// bytes match RPython byte-for-byte. RPython opencoder expects
    /// `Const{Int,Float,Ptr}` / `AbstractResOp` boxes inline.
    ///
    /// Panics when a constant OpRef is not inline-resolvable — that is a
    /// genuine invariant break.
    pub(crate) fn _encode_opref(&mut self, opref: OpRef) -> i64 {
        let b = if opref.is_constant() {
            let value = opref.inline_const_to_value().unwrap_or_else(|| {
                panic!("_encode_opref: constant {:?} not inline-resolvable", opref)
            });
            match value {
                Value::Int(v) => Box::ConstInt(v),
                Value::Float(f) => Box::ConstFloat(f.to_bits()),
                Value::Ref(r) => Box::ConstPtr(r.as_usize() as u64),
                Value::Void => {
                    panic!("_encode_opref: constant {:?} has Void type", opref)
                }
            }
        } else {
            Box::of_op(opref)
        };
        self._encode(b)
    }

    /// OpRef-taking adapter over `record_op(&[Box], descr)`.
    ///
    /// Mirrors opencoder.py `record_op(opnum, argboxes, descr=None)`
    /// but accepts pyre's constant-tagged OpRefs. Inline constants resolve
    /// via `OpRef::inline_const_to_value`; non-constant OpRefs route through
    /// the TAGBOX path via `Box::of_op`. Wire bytes are identical to the
    /// equivalent `record_op(&[Box])` call.
    #[allow(dead_code)]
    pub(crate) fn record_op_oprefs(
        &mut self,
        opcode: OpCode,
        argrefs: &[OpRef],
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, argrefs.len());
        for &r in argrefs {
            let tagged = self._encode_opref(r);
            self.append_int(tagged);
        }
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    // ── close_loop / finish helpers ────────────────────────────────────
    //
    // Thin wrappers over `record_op_oprefs` that mirror the
    // `recorder::Trace` surface (`close_loop`, `close_loop_with_descr`,
    // `finish`).

    /// Close the loop: append a JUMP op with no descr.  Mirrors
    /// `recorder::Trace::close_loop`.
    #[allow(dead_code)]
    pub(crate) fn close_loop_oprefs(&mut self, jump_args: &[OpRef]) -> u32 {
        self.record_op_oprefs(OpCode::Jump, jump_args, None)
    }

    /// Close the loop with an explicit JUMP descriptor (tentative JUMP
    /// target token recorded before compile_trace, pyjitpl.py).
    #[allow(dead_code)]
    pub(crate) fn close_loop_oprefs_with_descr(
        &mut self,
        jump_args: &[OpRef],
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        self.record_op_oprefs(OpCode::Jump, jump_args, descr)
    }

    /// Finish the trace: append a FINISH op with its FailDescr.
    /// pyjitpl.py:1637 `history.record1(rop.FINISH, ..., descr=token)`.
    #[allow(dead_code)]
    pub(crate) fn finish_oprefs(
        &mut self,
        finish_args: &[OpRef],
        descr: &majit_ir::DescrRef,
    ) -> u32 {
        self.record_op_oprefs(OpCode::Finish, finish_args, Some(descr))
    }

    /// opencoder.py append_snapshot_array_data_int(i).
    /// Writes a signed varint into `_snapshot_array_data`; values outside
    /// [MIN_VALUE, MAX_VALUE] trip `tag_overflow` (and encode 0 to keep
    /// the stream parseable).
    pub(crate) fn append_snapshot_array_data_int(&mut self, i: i64) {
        if !(MIN_VALUE..=MAX_VALUE).contains(&i) {
            self.tag_overflow = true;
            self._snapshot_array_data.append_varint(0);
        } else {
            self._snapshot_array_data.append_varint(i);
        }
    }

    /// opencoder.py append_snapshot_data_int(i).
    #[allow(dead_code)]
    pub(crate) fn append_snapshot_data_int(&mut self, i: i64) {
        if !(MIN_VALUE..=MAX_VALUE).contains(&i) {
            self.tag_overflow = true;
            self._snapshot_data.append_varint(0);
        } else {
            self._snapshot_data.append_varint(i);
        }
    }

    /// opencoder.py Trace.get_live_ranges().
    ///
    /// Walks the encoded byte stream `_ops` + `_snapshot_data` +
    /// `_snapshot_array_data` and returns `liveranges[v] =
    /// last_use_index` for every raw trace position `v` (opencoder.py:
    /// 855 `index = t._count`; walk via `next_element_update_live_range`
    /// at opencoder.py). Guard fail-arg equivalents come from
    /// the attached snapshot chain through `update_liveranges`
    /// (opencoder.py).
    #[allow(dead_code)]
    pub(crate) fn get_live_ranges(&self) -> Vec<usize> {
        let mut liveranges = vec![0usize; self._index as usize];
        // opencoder.py `index = t._count` with `t._count = start` at
        // iterator construction time (opencoder.py:270). `start` is the
        // Trace's `_start` field (= `max_num_inputargs`).
        let mut index = self._start as usize;
        // opencoder.py `self.pos = start` — byte cursor starts at
        // `_start`, the same value as the op index counter, because
        // the first `max_num_inputargs` bytes of `_ops` are reserved
        // as the inputarg placeholder region.
        let mut p: usize = self._start as usize;
        let end = self._pos;
        while p < end {
            let opnum = self.ops_bytes()[p];
            p += 1;
            let opcode = u16_to_opcode(opnum as u16);
            // opencoder.py:644-649 — variadic ops store `num_argboxes`
            // inline as a signed varint; fixed-arity ops do not.
            let argnum = match opcode.arity() {
                Some(n) => n as usize,
                None => {
                    let (n, consumed) = decode_varint_signed(&self.ops_bytes()[p..]);
                    p += consumed;
                    n as usize
                }
            };
            // opencoder.py:346-350 — TAGBOX args set liveranges[v] = index.
            for _ in 0..argnum {
                let (tagged, consumed) = decode_varint_signed(&self.ops_bytes()[p..]);
                p += consumed;
                let (tag, v) = untag(tagged as u32);
                if tag == TAGBOX {
                    let idx = v as usize;
                    if idx < liveranges.len() {
                        liveranges[idx] = index;
                    }
                }
            }
            let result_type = opcode.result_type();
            // opencoder.py:351-352 — non-void self-defines its slot.
            if result_type != Type::Void && index < liveranges.len() {
                liveranges[index] = index;
            }
            // opencoder.py:353-357 — guards drive the snapshot walk.
            if opcode.has_descr() {
                let (descr_index, consumed) = decode_varint_signed(&self.ops_bytes()[p..]);
                p += consumed;
                if opcode.is_guard() {
                    update_liveranges(
                        &self._snapshot_data,
                        &self._snapshot_array_data,
                        descr_index as usize,
                        index,
                        &mut liveranges,
                    );
                }
            }
            // opencoder.py:358-360 — `_index` advances only for non-void.
            if result_type != Type::Void {
                index += 1;
            }
        }
        liveranges
    }

    /// opencoder.py Trace.get_dead_ranges().
    ///
    /// For each index `x`, `deadranges[x]` names a value that is
    /// *certainly* dead before step `x`. Collisions (multiple values
    /// dying at the same step) are resolved by linear probing forward
    /// through `deadranges` — the nested `insert(ranges, pos, v)`
    /// helper at opencoder.py.
    ///
    /// Result is memoized in `_deadranges = (self._count, deadranges)`
    /// and returned unchanged across calls until `_count` advances
    /// (opencoder.py, 883).
    #[allow(dead_code)]
    pub(crate) fn get_dead_ranges(&mut self) -> Vec<usize> {
        // opencoder.py:873-875 cache hit path.
        if let Some((cached_count, cached)) = &self._deadranges
            && *cached_count == self._count
        {
            return cached.clone();
        }
        // opencoder.py `liveranges = self.get_live_ranges()`.
        let liveranges = self.get_live_ranges();
        // opencoder.py `deadranges = [0] * (self._index + 2)`.
        let mut deadranges = vec![0usize; (self._index as usize) + 2];
        debug_assert_eq!(deadranges.len(), liveranges.len() + 2);
        // opencoder.py — skip inputargs (i < _start), insert
        // live end-points with linear probing.
        for (i, &elem) in liveranges.iter().enumerate().skip(self._start as usize) {
            if elem != 0 {
                // opencoder.py nested insert(ranges, pos, v).
                let mut pos = elem + 1;
                while pos < deadranges.len() && deadranges[pos] != 0 {
                    pos += 1;
                    if pos == deadranges.len() {
                        break;
                    }
                }
                if pos < deadranges.len() {
                    deadranges[pos] = i;
                }
            }
        }
        // opencoder.py:883 — memoize.
        self._deadranges = Some((self._count, deadranges.clone()));
        deadranges
    }

    /// `_refs[index]` at the address the GC holds now. `_refs` is the
    /// traced `GcArray` of `GCREF`, so its slot is the forwarded address.
    pub(crate) fn current_ref(&self, index: usize) -> u64 {
        self._refs.as_slice()[index] as u64
    }

    /// `history.py new_ref_dict` keys are object identities
    /// (`id_or_identityhash`). The ref `GcArray` slots are already the
    /// forwarded addresses; this rekeys the host `_refs_dict` by that
    /// hash. A dictionary cleared by `tracing_done` stays cleared.
    pub(crate) fn refresh_from_gc(&mut self) {
        self.rekey_refs();
    }

    fn rekey_refs(&mut self) {
        if self._refs_dict.is_empty() {
            return;
        }
        self._refs_dict.clear();
        for index in 1..self._refs.len() {
            let addr = self._refs[index];
            if addr == 0 {
                continue;
            }
            let key = majit_ir::gc_id_or_identityhash(addr) as u64;
            self._refs_dict.insert(key, index as u32);
            if key != addr as u64 {
                self._refs_dict.insert(addr as u64, index as u32);
            }
        }
    }

    /// Fill `_ops`, `_refs`, `_bigints`, `_floats`, and both snapshot
    /// lists past their initial allocation, with a minor between appends.
    /// Read back through `ByteTraceIter` and `SnapshotIterator`.
    pub fn fill_pools_past_reserve_for_test(&mut self, obj_tid: u32) {
        let total = INIT_SIZE + 64;
        for i in 0..total {
            if i == INIT_SIZE / 2 || i == INIT_SIZE + 8 {
                majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
            }
            self.append_byte((i & 0xff) as u8);
        }
        let expected: Vec<u8> = (0..total).map(|i| (i & 0xff) as u8).collect();
        assert_eq!(self.bytes_read_by_iterator(), expected);

        let mut ptr_payloads = Vec::new();
        for i in 0..40u64 {
            if i == 8 || i == 36 {
                majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
            }
            let obj = majit_gc::gc_sync::gc_op(|gc| gc.alloc_with_type(obj_tid, 8));
            unsafe { *(obj.0 as *mut u64) = i };
            ptr_payloads.push(i);
            let _ = self._encode_ptr(obj.0 as u64);
        }
        let bigint = (1i64 << 40) + 7;
        for i in 0..10i64 {
            if i == 3 {
                majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
            }
            let _ = self._encode_bigint(bigint + i);
        }
        let float_bits = 1.5f64.to_bits();
        for i in 0..6u64 {
            if i == 2 {
                majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
            }
            let _ = self._encode_float(float_bits.wrapping_add(i));
        }
        for i in 0..80i64 {
            if i == 20 {
                majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
            }
            self.append_snapshot_data_int(i.wrapping_mul(10));
            self.append_snapshot_array_data_int(i);
        }
        assert!(self._refs.len() > 32, "refs {}", self._refs.len());
        assert!(self._bigints.len() > 4, "bigints {}", self._bigints.len());
        assert!(self._floats.len() > 4, "floats {}", self._floats.len());
        assert!(self._snapshot_data.len() > 128);
        assert!(self._snapshot_array_data.len() > 128);

        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        self.refresh_from_gc();
        for (i, payload) in ptr_payloads.iter().enumerate() {
            let addr = self._refs[i + 1];
            assert_ne!(addr, 0);
            assert_eq!(unsafe { *(addr as *const u64) }, *payload);
        }
        for i in 0..10i64 {
            assert_eq!(self._bigints[i as usize], bigint + i);
        }
        for i in 0..6u64 {
            assert_eq!(self._floats[i as usize], float_bits.wrapping_add(i));
        }

        let ptr_for_op = self._refs[1];
        let ops_at = self._pos;
        let _ = self.record_op1(OpCode::SameAsR, Box::ConstPtr(ptr_for_op as u64), None);
        let _ = self.record_op1(OpCode::SameAsI, Box::ConstInt(bigint), None);
        let _ = self.record_op1(OpCode::SameAsF, Box::ConstFloat(float_bits), None);
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        self.refresh_from_gc();
        let moved_ptr = self._refs[1];
        let mut saw_ptr = false;
        let mut saw_int = false;
        let mut saw_float = false;
        // The leading `_ops` bytes are the raw fill, not opcodes. The
        // iterator starts where `record_op1` began (`TraceIterator`).
        let mut it = ByteTraceIter::new(self, ops_at, self._pos, 0);
        while let Some(op) = it.next() {
            match op.opcode {
                OpCode::SameAsR => {
                    assert_eq!(
                        op.arg(0).to_opref().as_const_ptr(),
                        Some(majit_ir::GcRef(moved_ptr as usize))
                    );
                    saw_ptr = true;
                }
                OpCode::SameAsI => {
                    assert_eq!(op.arg(0).to_opref().as_const_int(), Some(bigint));
                    saw_int = true;
                }
                OpCode::SameAsF => {
                    assert_eq!(
                        op.arg(0).to_opref().as_const_float(),
                        Some(f64::from_bits(float_bits))
                    );
                    saw_float = true;
                }
                _ => {}
            }
        }
        assert!(saw_ptr && saw_int && saw_float);

        self.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let array = self.new_array(1);
        self._add_box_to_storage(Self::_encode_smallint(7));
        let snap = self.create_top_snapshot(1, 2, array, &[], &[], true);
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        let iter = self.get_snapshot_iter(snap as usize);
        assert_eq!(iter.framestack.len(), 1);
        let frame = iter.framestack[0];
        let mut boxes = iter.iter_array(frame);
        assert_eq!(boxes.next(), Some(Self::_encode_smallint(7)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test fixture equivalent of RPython's `metainterp_sd` fixture used
    /// by `test/test_opencoder.py`. Produces a fresh empty static data
    /// object so `Trace.__init__` (opencoder.py) has something to
    /// hold in `self.metainterp_sd`.
    fn empty_sd() -> Arc<crate::MetaInterpStaticData> {
        Arc::new(crate::MetaInterpStaticData::new())
    }

    #[test]
    fn trace_new_reserves_constptr_and_snapshot_pools() {
        // opencoder.py Trace.__init__ starts `_refs` / snapshot lists
        // empty. A Vec that doubles through 8/16/32/64/128 B on every
        // regex bridge was the timed-row class (`_encode_ptr`,
        // `encode_varint_signed`). The first 16 unique ConstPtrs and the
        // first 128 snapshot bytes must stay inside the constructor
        // reserve.
        let mut trace = TraceRecordBuffer::new(0, empty_sd());
        assert!(
            trace._refs.capacity() >= 32,
            "_refs must absorb the first unique ConstPtrs without growing"
        );
        assert!(
            trace._snapshot_data.capacity() >= 128,
            "_snapshot_data must absorb the first snapshot without a 64 B grow"
        );
        assert!(
            trace._snapshot_array_data.capacity() >= 128,
            "_snapshot_array_data must absorb the first snapshot without a 64 B grow"
        );
        let refs_cap = trace._refs.capacity();
        let snap_cap = trace._snapshot_array_data.capacity();
        for i in 1..=16u64 {
            let _ = trace._encode_ptr(i * 0x1000);
        }
        assert_eq!(
            trace._refs.capacity(),
            refs_cap,
            "16 unique ConstPtrs grew _refs off the constructor reserve"
        );
        for i in 0..32 {
            trace.append_snapshot_array_data_int(i);
        }
        assert_eq!(
            trace._snapshot_array_data.capacity(),
            snap_cap,
            "32 snapshot varints grew _snapshot_array_data off the constructor reserve"
        );
    }

    #[test]
    fn trace_drop_preserves_later_lexical_roots() {
        let base = majit_gc::shadow_stack::depth();
        let mut trace = TraceRecordBuffer::new(0, empty_sd());
        trace._encode_ptr(0x1000);
        let lexical = majit_gc::shadow_stack::push(majit_ir::GcRef(0x2000));
        drop(trace);
        let remaining = (majit_gc::shadow_stack::depth() > lexical)
            .then(|| majit_gc::shadow_stack::get(lexical));
        majit_gc::shadow_stack::pop_to(base);
        assert_eq!(remaining, Some(majit_ir::GcRef(0x2000)));
    }

    #[test]
    fn trace_refresh_rekeys_moved_constants_without_changing_encoded_indexes() {
        let mut trace = TraceRecordBuffer::new(0, empty_sd());
        let first = trace._encode_ptr(0x1000);
        let second = trace._encode_ptr(0x2000);
        // The collector rewrites `GcArray` slots in place. No per-element guard.
        trace._refs.set(1, 0x2000);
        trace._refs.set(2, 0x3000usize);
        trace.refresh_from_gc();
        assert_eq!(&trace._refs[..], &[0usize, 0x2000, 0x3000]);
        assert_eq!(trace._encode_ptr(0x2000), first);
        assert_eq!(trace._encode_ptr(0x3000), second);
        assert_eq!(trace._refs.len(), 3);
        assert!(!trace._refs_dict.contains_key(&0x1000));
    }

    #[test]
    fn trace_refs_do_not_push_per_element_shadow_roots() {
        let base = majit_gc::shadow_stack::depth();
        majit_gc::shadow_stack::push(majit_ir::GcRef(0x1000));
        let mut first = TraceRecordBuffer::new(0, empty_sd());
        first._encode_ptr(0x2000);
        let mut second = TraceRecordBuffer::new(0, empty_sd());
        second._encode_ptr(0x3000);
        // No collector: the ref array is a host block, not an owner root.
        // Encoding must not push the shadow stack, and dropping one trace
        // must not pop the other trace's storage or the lexical root.
        assert_eq!(majit_gc::shadow_stack::depth(), base + 1);
        majit_gc::shadow_stack::pop_to(base);
        drop(first);
        assert_eq!(second._refs[1], 0x3000);
        drop(second);
        assert_eq!(majit_gc::shadow_stack::depth(), base);
    }

    #[test]
    fn trace_refresh_after_tracing_done_does_not_retain_a_lookup_dictionary() {
        let mut trace = TraceRecordBuffer::new(0, empty_sd());
        trace.record_op1(OpCode::SameAsR, Box::ConstPtr(0x1000), None);
        trace.tracing_done().unwrap();
        trace._refs.set(1, 0x2000usize);
        trace.refresh_from_gc();
        assert!(trace._refs_dict.is_empty());
        let op = trace.get_byte_iter().next().unwrap();
        assert_eq!(
            op.arg(0).to_opref().as_const_ptr(),
            Some(majit_ir::GcRef(0x2000))
        );
    }

    #[test]
    fn test_signed_varint_roundtrip() {
        // opencoder.py enforces MIN_VALUE=-2^30, MAX_VALUE=2^30-1. Test
        // the 2-byte path (values in [-2^14, 2^14-1]) and the 4-byte path
        // (outside it). `i64::MAX` / `i64::MIN` would trip the
        // debug_assert — they are out of the encoder's documented range.
        let values = [
            0i64,
            1,
            -1,
            127,
            -128,
            1000,
            -1000,
            // 2-byte boundaries (stay in ±2^14 range)
            (1 << 14) - 1,
            -(1 << 14),
            // 4-byte range
            (1 << 14),
            -(1 << 14) - 1,
            (1 << 20),
            -(1 << 20),
            MIN_VALUE,
            MAX_VALUE,
        ];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            let (decoded, _consumed) = decode_varint_signed(&buf);
            assert_eq!(decoded, val, "signed varint roundtrip failed for {val}");
        }
    }

    /// Phase B2: encoder must match RPython's 2-byte / 4-byte layout
    /// bit-exactly for snapshot and op stream compatibility. Verify
    /// small values stay 2 bytes and big values expand to 4 bytes.
    #[test]
    fn test_signed_varint_wire_format_b2() {
        // 2-byte values: flag bit (bit 7 of byte 0) MUST be clear.
        for &val in &[0i64, 1, -1, (1 << 14) - 1, -(1 << 14)] {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            assert_eq!(buf.len(), 2, "value {val} should be 2 bytes");
            assert_eq!(buf[0] & 0x80, 0, "flag bit set for 2-byte value {val}");
        }
        // 4-byte values: flag bit MUST be set.
        for &val in &[1i64 << 14, -(1i64 << 14) - 1, MAX_VALUE, MIN_VALUE] {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            assert_eq!(buf.len(), 4, "value {val} should be 4 bytes");
            assert_eq!(buf[0] & 0x80, 0x80, "flag bit clear for 4-byte value {val}");
        }
    }

    #[test]
    fn test_tag_untag_roundtrip() {
        for kind in [TAGINT, TAGCONSTPTR, TAGCONSTOTHER, TAGBOX] {
            for value in [0u32, 1, 42, 255, 1000] {
                let tagged = tag(kind, value);
                let (k, v) = untag(tagged);
                assert_eq!(k, kind, "tag kind mismatch");
                assert_eq!(v, value, "tag value mismatch");
            }
        }
    }

    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn farg(pos: u32) -> OpRef {
        OpRef::input_arg_float(pos)
    }

    fn rarg(pos: u32) -> OpRef {
        OpRef::input_arg_ref(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    #[allow(dead_code)]
    fn fop(pos: u32) -> OpRef {
        OpRef::float_op(pos)
    }

    fn rop(pos: u32) -> OpRef {
        OpRef::ref_op(pos)
    }

    #[allow(dead_code)]
    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    /// Bind a position-only `OpRef` arg to a producer-bound `Operand`,
    /// oparser-faithful (`rpython/jit/tool/oparser.py` `self.vars[arg]`):
    /// body / inputarg refs shed to `Operand::Op` / `Operand::InputArg` via
    /// the rooted drop-in (same `to_opref()`), constants shed to
    /// `Operand::Const`.
    fn box_arg_operand(opref: OpRef) -> majit_ir::operand::Operand {
        use crate::history::test_support::{rooted_inputarg_operand, rooted_resop_operand};
        use majit_ir::Type;
        match opref {
            OpRef::InputArgInt(x) => rooted_inputarg_operand(Type::Int, x),
            OpRef::InputArgFloat(x) => rooted_inputarg_operand(Type::Float, x),
            OpRef::InputArgRef(x) => rooted_inputarg_operand(Type::Ref, x),
            OpRef::IntOp(x) => rooted_resop_operand(Type::Int, x),
            OpRef::FloatOp(x) => rooted_resop_operand(Type::Float, x),
            OpRef::RefOp(x) => rooted_resop_operand(Type::Ref, x),
            OpRef::VoidOp(x) => rooted_resop_operand(Type::Void, x),
            // Constants / None shed without minting Operand::Box.
            _ => majit_ir::operand::Operand::from_opref(opref),
        }
    }

    /// Helper: build an Op with a specific trace position and the same
    /// typed result class RPython's `opclasses[opnum]` would instantiate.
    fn op_at(pos: u32, opcode: majit_ir::OpCode, args: &[OpRef]) -> majit_ir::Op {
        let op_args: Vec<majit_ir::operand::Operand> =
            args.iter().map(|a| box_arg_operand(*a)).collect();
        let mut op = majit_ir::Op::new(opcode, &op_args);
        op.pos().set(OpRef::op_typed(pos, opcode.result_type()));
        op
    }

    /// Helper: project the iterator's fresh inputarg objects onto their
    /// OpRef encodings for positional assertions.
    fn inputarg_oprefs(iter: &TraceIterator<'_>) -> Vec<OpRef> {
        iter.inputargs.iter().map(|ia| ia.opref()).collect()
    }

    /// Helper: project a `_cache` slot onto its OpRef encoding.
    fn cache_opref(iter: &TraceIterator<'_>, i: usize) -> Option<OpRef> {
        iter._cache[i].as_ref().map(|b| b.to_opref())
    }

    #[test]
    fn test_trace_iterator_next_basic() {
        // opencoder.py TraceIterator.next() parity:
        //
        // Phase-1 iteration over a 2-inputarg trace producing two IntAdd
        // results. Inputargs occupy positions [0, 2); the first non-void
        // result is allocated as BoxInt at position 2 (= start_fresh = num_inputargs).
        let ops = vec![
            op_at(2, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(1)]),
            op_at(3, majit_ir::OpCode::IntAdd, &[iop(2), iarg(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[box_arg_operand(iop(3))]),
        ];
        let inputarg_types = vec![majit_ir::Type::Int; 2];

        // Phase 1 / legacy layout: start_fresh = 0 → inputargs allocated
        // as InputArgInt(0..2), op results follow as BoxInt(2..).
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 0);
        assert_eq!(inputarg_oprefs(&iter), vec![iarg(0), iarg(1)]);
        assert_eq!(cache_opref(&iter, 0), Some(iarg(0)));
        assert_eq!(cache_opref(&iter, 1), Some(iarg(1)));

        // First IntAdd: result at fresh BoxInt(2), args translated via cache.
        let r1 = iter.next().unwrap();
        assert_eq!(r1.pos().get(), iop(2));
        assert_eq!(r1.arg(0).to_opref(), iarg(0));
        assert_eq!(r1.arg(1).to_opref(), iarg(1));

        // Second IntAdd: result at fresh BoxInt(3); first arg references the
        // previous result (raw pos 2 → cached BoxInt(2)).
        let r2 = iter.next().unwrap();
        assert_eq!(r2.pos().get(), iop(3));
        assert_eq!(r2.arg(0).to_opref(), iop(2));
        assert_eq!(r2.arg(1).to_opref(), iarg(0));

        // Finish is void: pos unchanged, arg cached → BoxInt(3).
        let finish = iter.next().unwrap();
        assert_eq!(finish.arg(0).to_opref(), iop(3));
        assert!(iter.done());
    }

    #[test]
    fn trace_iterator_keeps_an_already_decoded_const_operand() {
        let constant = majit_ir::operand::Operand::const_from_value(majit_ir::Value::Int(7));
        let ops = vec![majit_ir::Op::new(
            majit_ir::OpCode::IntAdd,
            &[constant.clone(), constant.clone()],
        )];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &[], 0);

        let decoded = iter.next().expect("one encoded operation");

        assert!(decoded.arg(0).same_box(&constant));
        assert!(decoded.arg(1).same_box(&constant));
    }

    #[test]
    fn trace_iterator_seeds_sparse_guard_history_positions() {
        // `Trace(max_num_inputargs=3)` reserved position 1, but
        // `History.set_inputargs([box0, box2])` filtered that dead box.
        // `TraceIterator.__init__` must seed `_cache` by each surviving
        // box's recorded position, not by its compact vector index.
        let ops = vec![OpRc::new(op_at(
            3,
            majit_ir::OpCode::IntAdd,
            &[iarg(0), iarg(2)],
        ))];
        let inputargs = vec![
            majit_ir::InputArg::new_int(0),
            majit_ir::InputArg::new_int(2),
        ];

        let mut iter = TraceIterator::new_with_inputargs(&ops, 0, ops.len(), None, &inputargs, 10);
        assert_eq!(cache_opref(&iter, 0), Some(iarg(10)));
        assert_eq!(cache_opref(&iter, 1), None);
        assert_eq!(cache_opref(&iter, 2), Some(iarg(11)));

        let add = iter.next().unwrap();
        assert_eq!(add.arg(0).to_opref(), iarg(10));
        assert_eq!(add.arg(1).to_opref(), iarg(11));
        assert_eq!(add.pos().get(), iop(12));
    }

    #[test]
    fn test_trace_iterator_phase1_phase2_disjoint() {
        // opencoder.py TraceIterator.next() identity parity:
        //
        // RPython distinguishes Phase 1 boxes from Phase 2 boxes by Python
        // `is`-identity even when the two iterators visit the same source
        // trace position. majit lacks separate identity from position, so
        // each phase passes a different `start_fresh` and the iterator
        // produces disjoint OpRefs by construction.
        let ops = vec![
            op_at(2, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(1)]),
            op_at(3, majit_ir::OpCode::IntAdd, &[iop(2), iarg(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[box_arg_operand(iop(3))]),
        ];
        let inputarg_types = vec![majit_ir::Type::Int; 2];

        // Phase 1: start_fresh = 0 reproduces the legacy positional layout
        // (inputargs at [0, num_inputargs), op results at [num_inputargs, …)).
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut p1 = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 0);
        let mut p1_ops = Vec::new();
        while let Some(op) = p1.next() {
            p1_ops.push(op);
        }
        let p1_high_water = p1._fresh;
        assert_eq!(p1_high_water, 4);
        // RPython `_index` walks the `_cache` slots monotonically; after
        // the 2 non-void ops it should point just past the last written
        // raw trace position (3 + 1 = 4).
        assert_eq!(p1._index, 4);

        // Phase 2: start_fresh = phase1 high water → disjoint namespace.
        let mut p2 = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, p1_high_water);
        let mut p2_ops = Vec::new();
        while let Some(op) = p2.next() {
            p2_ops.push(op);
        }
        assert_eq!(inputarg_oprefs(&p2), vec![iarg(4), iarg(5)]);
        assert_eq!(p2_ops[0].pos().get(), iop(6));
        assert_eq!(p2_ops[0].arg(0).to_opref(), iarg(4));
        assert_eq!(p2_ops[0].arg(1).to_opref(), iarg(5));
        assert_eq!(p2_ops[1].pos().get(), iop(7));
        assert_eq!(p2_ops[1].arg(0).to_opref(), iop(6));
        assert_eq!(p2_ops[1].arg(1).to_opref(), iarg(4));
        assert_eq!(p2_ops[2].arg(0).to_opref(), iop(7));

        // No Phase 1 OpRef equals any Phase 2 OpRef.
        let p1_positions: Vec<u32> = p1_ops
            .iter()
            .map(|op| op.pos().get().raw())
            .filter(|&p| p != u32::MAX)
            .collect();
        let p2_positions: Vec<u32> = p2_ops
            .iter()
            .map(|op| op.pos().get().raw())
            .filter(|&p| p != u32::MAX)
            .collect();
        assert!(p1_positions.iter().all(|p| !p2_positions.contains(p)));
    }

    #[test]
    fn test_trace_iterator_constants_passthrough() {
        // opencoder.py _untag(): TAGINT/TAGCONSTPTR/TAGCONSTOTHER
        // pass through unchanged. In majit, constant OpRefs (inline-Const
        // variants) must NOT be remapped through `_cache`.
        let const_ref = OpRef::const_int(5);
        let ops = vec![op_at(1, majit_ir::OpCode::IntAdd, &[iarg(0), const_ref])];
        let inputarg_types = vec![majit_ir::Type::Int; 1];
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 0);
        let r = iter.next().unwrap();
        assert_eq!(r.pos().get(), iop(1));
        assert_eq!(r.arg(0).to_opref(), iarg(0));
        assert_eq!(r.arg(1).to_opref(), const_ref);
    }

    #[test]
    fn test_trace_iterator_replace_last_cached_shifted_phase() {
        // opencoder.py replace_last_cached parity. RPython walks
        // `_cache` densely by ops-count (`_index`), so `_cache[_index - 1]`
        // always targets the slot just written by `next()`. majit's
        // `_index` is separated from the fresh OpRef counter — in a
        // shifted phase (Phase 2 with `start_fresh` >> 0) the cache key
        // is still the raw trace position, so `_index` must stay in raw
        // trace position coordinates.
        let ops = vec![
            op_at(1, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(0)]),
            op_at(2, majit_ir::OpCode::IntAdd, &[iop(1), iarg(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[box_arg_operand(iop(2))]),
        ];
        // Shifted phase: start_fresh = 100 so any confusion between
        // _index (raw trace position) and _fresh (fresh OpRef counter)
        // surfaces as an out-of-bounds panic or stale cache slot.
        let inputarg_types = vec![majit_ir::Type::Int; 1];
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 100);
        let r1 = iter.next().unwrap();
        // After writing raw pos 1, _index should be 2 (next slot).
        assert_eq!(iter._index, 2);
        assert_eq!(r1.pos().get(), iop(101));
        // replace_last_cached(oldbox=BoxInt(101), new_box=BoxInt(999))
        // must target _cache[_index - 1] = _cache[1], where the last
        // write placed BoxInt(101).
        iter.replace_last_cached(iop(101), box_arg_operand(iop(999)));
        assert_eq!(cache_opref(&iter, 1), Some(iop(999)));

        // The next op references raw pos 1 as its first arg, so the
        // replaced value should flow through _untag → _get.
        let r2 = iter.next().unwrap();
        assert_eq!(r2.arg(0).to_opref(), iop(999));
        assert_eq!(r2.arg(1).to_opref(), iarg(100));
        // After writing raw pos 2, _index advances to 3.
        assert_eq!(iter._index, 3);
    }

    #[test]
    fn test_trace_iterator_fail_args_remapped() {
        // opencoder.py next() routes guard fail_args through the
        // same _untag/_get path as regular args.
        let mut guard = op_at(2, majit_ir::OpCode::GuardTrue, &[iop(1)]);
        guard.setfailargs(vec![box_arg_operand(iarg(0)), box_arg_operand(iop(1))].into());
        let ops = vec![
            op_at(1, majit_ir::OpCode::IntEq, &[iarg(0), iarg(0)]),
            guard,
        ];
        let inputarg_types = vec![majit_ir::Type::Int; 1];
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 10);
        let r1 = iter.next().unwrap();
        assert_eq!(r1.pos().get(), iop(11));
        assert_eq!(r1.arg(0).to_opref(), iarg(10));
        assert_eq!(r1.arg(1).to_opref(), iarg(10));
        let r2 = iter.next().unwrap();
        assert_eq!(r2.arg(0).to_opref(), iop(11));
        let fa = r2.guard_fail_args().unwrap();
        assert_eq!(fa[0].to_opref(), iarg(10));
        assert_eq!(fa[1].to_opref(), iop(11));
    }

    #[test]
    fn test_trace_iterator_keeps_frontend_stamp() {
        // history.py FrontendOp `_resint` lives on the recorded box.
        // The structured walker must copy it onto the cls() result.
        let mut add = op_at(1, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(0)]);
        add.set_value(majit_ir::value::Value::Int(7));
        let ops = vec![add];
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &[majit_ir::Type::Int], 10);
        let r = iter.next().unwrap();
        assert_eq!(r.get_value(), Some(majit_ir::value::Value::Int(7)));
    }

    #[test]
    fn test_trace_iterator_overwrites_inputarg_cache_slot_with_fresh_result() {
        // RPython next() allocates a fresh result box even when the raw trace
        // position collides with an already-seeded inputarg cache slot.
        let ops = vec![
            op_at(3, majit_ir::OpCode::GetfieldRawI, &[rarg(0)]),
            op_at(1, majit_ir::OpCode::GetarrayitemGcR, &[iop(3), rarg(1)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[box_arg_operand(rop(1))]),
        ];
        let inputarg_types = vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ];
        let ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 100);

        assert_eq!(
            inputarg_oprefs(&iter),
            vec![rarg(100), rarg(101), iarg(102), iarg(103)]
        );
        assert_eq!(cache_opref(&iter, 1), Some(rarg(101)));

        let op0 = iter.next().unwrap();
        assert_eq!(op0.pos().get(), iop(104));
        assert_eq!(
            op0.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![rarg(100)]
        );

        let op1 = iter.next().unwrap();
        assert_eq!(
            op1.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![iop(104), rarg(101)]
        );
        assert_eq!(op1.pos().get(), rop(105));
        assert_eq!(cache_opref(&iter, 1), Some(rop(105)));

        let finish = iter.next().unwrap();
        assert_eq!(
            finish
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![rop(105)]
        );
    }

    // Phase B1 intentionally drops `test_trace_record_buffer` and
    // `test_trace_record_snapshot`: they exercised the pre-Phase-B
    // legacy record_op / capture_resumedata / snapshots API that
    // diverged from rpython/jit/metainterp/opencoder.py and has been
    // removed. Phase B3 (wire format), B5 (guard 0-placeholder), B7
    // (capture_resumedata + create_top_snapshot), and Phase C
    // (RPython test_opencoder.py ports) replace them with parity
    // tests that exercise the RPython-shaped API.

    #[test]
    fn test_trace_record_buffer_initial_state_b1() {
        // Phase B1 smoke test: `Trace::new(max_num_inputargs)` seeds the
        // counters per opencoder.py (all four — including
        // `_pos` — equal `max_num_inputargs`) and reserves
        // `snapshot_array_data` index 0 for the empty-length array
        // (opencoder.py + :728-733).
        let buf = TraceRecordBuffer::new(3, empty_sd());
        assert_eq!(buf.max_num_inputargs, 3);
        assert_eq!(buf._count, 3);
        assert_eq!(buf._index, 3);
        assert_eq!(buf._start, 3);
        assert_eq!(buf._pos, 3);
        // opencoder.py `length() = self._pos` — byte length of the
        // encoded stream INCLUDES the reserved inputarg placeholder
        // prefix.
        assert_eq!(buf.length(), 3);
        assert!(buf.inputargs.is_empty());
        assert!(!buf.tag_overflow);
        // _descrs starts with one None entry so index 0 means "no descr".
        assert_eq!(buf._descrs.len(), 1);
        assert!(buf._descrs[0].is_none());
        // _refs starts with nullptr so index 0 means "null".
        assert_eq!(&buf._refs[..], &[0usize][..]);
        // _snapshot_array_data has one signed-varint-encoded 0 recorded.
        assert!(!buf._snapshot_array_data.is_empty());
    }

    /// Phase B8: `get_live_ranges` walks `_ops` + `_snapshot_data`
    /// and `_snapshot_array_data`, producing RPython-parity
    /// `liveranges[v] = last_use_index` output, with guards driving
    /// the snapshot-chain walk via `update_liveranges`.
    #[test]
    fn test_get_live_ranges_linear_b8() {
        // Build: inputarg i0 at position 0. Op1 = IntAdd(i0, i0) at
        // position 1 (result index 1). Op2 = IntSub(Op1.result, i0)
        // at position 2. Linear chain — no guards, no snapshots.
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let p1 = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(0), None);
        let p2 = buf.record_op2(OpCode::IntSub, Box::ResOp(p1), Box::ResOp(0), None);
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);

        let lr = buf.get_live_ranges();
        // lr has length _index == 3 (0, 1, 2).
        assert_eq!(lr.len(), 3);
        // i0 last used at index 2 (IntSub). Op1.result last used at
        // index 2 (IntSub arg). Op2 self-defines at index 2.
        assert_eq!(lr[0], 2, "inputarg last-used at the IntSub step");
        assert_eq!(lr[1], 2, "IntAdd result last-used by IntSub");
        assert_eq!(lr[2], 2, "IntSub self-defines its own slot");
    }

    /// Phase B8: when a guard fires, `update_liveranges` walks the
    /// snapshot chain and marks the snapshot's tagged TAGBOX items
    /// at the guard's step index.
    #[test]
    fn test_get_live_ranges_guard_snapshot_b8() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        // Op0 = GuardTrue(p0) — records a 0-placeholder that the
        // snapshot attach patches.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Build a small active-box array containing TAGBOX(1)
        // (inputarg 1). create_top_snapshot then patches the guard's
        // descr slot to the snapshot byte offset.
        let array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(1));
        let _snap = buf.create_top_snapshot(0, 0, array, &[], &[], true);

        let lr = buf.get_live_ranges();
        // _index hasn't advanced (GuardTrue is void), so the liveranges
        // array still has length == _start (= 2 inputargs).
        assert_eq!(lr.len(), 2);
        // The guard's step index is _start (2). p0 is referenced as the
        // guard's direct arg — last-used at step 2. p1 is referenced via
        // the snapshot's active-box array — also last-used at step 2.
        assert_eq!(lr[0], 2, "GuardTrue arg → liveranges[0]");
        assert_eq!(lr[1], 2, "snapshot chain → liveranges[1]");
    }

    /// Phase B8 + RPython opencoder.py `get_dead_ranges()` —
    /// cache-hit path: first call populates `_deadranges = (_count,
    /// vec)`, second call with the same `_count` must return the
    /// cached vec unchanged.
    #[test]
    fn test_get_dead_ranges_memoization_b8() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        // A couple of non-void ops to populate liveranges.
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(0), None);
        let _ = buf.record_op2(OpCode::IntSub, Box::ResOp(1), Box::ResOp(0), None);
        let first = buf.get_dead_ranges();
        let cached_count_after_first = buf._deadranges.as_ref().map(|(c, _)| *c);
        assert_eq!(cached_count_after_first, Some(buf._count));

        // opencoder.py — `self._deadranges[0] == self._count` returns
        // the cached list without rewalking the trace.
        let second = buf.get_dead_ranges();
        assert_eq!(first, second);

        // Recording one more op advances `_count`, invalidating the
        // cache (opencoder.py:873-875 gate).
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(2), Box::ResOp(0), None);
        let third = buf.get_dead_ranges();
        // The third call sees a new `_count` and recomputes — the
        // resulting vec is one slot longer (`_index + 2` grew by one).
        assert_eq!(third.len(), second.len() + 1);
    }

    /// `record_input_arg` returns a positional OpRef and appends
    /// an `InputArg` whose `tp` field matches the passed type. `num_ops`
    /// stays at 0 because recording inputargs advances `_count` purely
    /// through `_start = max_num_inputargs`, not through ops.
    #[test]
    fn test_record_input_arg_2a() {
        let mut buf = TraceRecordBuffer::new(3, empty_sd());
        let a = buf.record_input_arg(Type::Int);
        let b = buf.record_input_arg(Type::Float);
        let c = buf.record_input_arg(Type::Ref);
        assert_eq!(a, iarg(0));
        assert_eq!(b, farg(1));
        assert_eq!(c, rarg(2));
        assert_eq!(buf.num_inputargs(), 3);
        assert_eq!(
            buf.inputarg_types(),
            vec![Type::Int, Type::Float, Type::Ref]
        );
        assert_eq!(buf.inputargs().len(), 3);
        // `_count` was seeded to max_num_inputargs; record_input_arg must
        // NOT advance it (see history.py length parity — the formula
        // stays `_count - max_num_inputargs`).
        assert_eq!(buf.num_ops(), 0);
    }

    /// `record_input_arg` panics when asked to exceed the
    /// `max_num_inputargs` cap supplied to `Trace::new`.
    #[test]
    #[should_panic(expected = "exceeds max_num_inputargs")]
    fn test_record_input_arg_overflow_panics_2a() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_input_arg(Type::Int);
        // Second call must panic — cap is 1.
        buf.record_input_arg(Type::Int);
    }

    // ── M4 · ByteTraceIter round-trip tests ─────────────────────────

    /// M4 step 1: empty trace iterates to nothing, inputargs get fresh
    /// OpRefs seeded from `max_num_inputargs`.
    #[test]
    fn test_byte_trace_iter_empty_m4() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        let mut it = buf.get_byte_iter();
        assert!(it.done());
        assert!(it.next().is_none());
        assert_eq!(it.inputargs.len(), 2);
        // start_fresh = max_num_inputargs = 2; inputargs get InputArgInt(2), InputArgInt(3).
        assert_eq!(it.inputargs[0].opref(), iarg(2));
        assert_eq!(it.inputargs[1].opref(), iarg(3));
    }

    /// M4 step 1: TAGBOX-only round-trip for 2-arg fixed-arity op.
    /// After `record_op2(IntAdd, box0, box1)` the byte stream must
    /// decode back to an Op whose args are the fresh-OpRef form of the
    /// two inputargs.
    #[test]
    fn test_byte_trace_iter_tagbox_round_trip_m4() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_input_arg(Type::Int);
        // `record_op2` encodes `Box::ResOp(0)` / `Box::ResOp(1)` as
        // TAGBOX tagged values.  The fresh iterator cache maps raw
        // position 0 / 1 to the iterator-local OpRefs `inputargs[0..=1]`.
        let pos = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        assert_eq!(pos, 2); // pre-bump `_index`, seeded to max_num_inputargs

        let mut it = buf.get_byte_iter();
        let fresh_i0 = it.inputargs[0].opref();
        let fresh_i1 = it.inputargs[1].opref();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::IntAdd);
        assert_eq!(op.num_args(), 2);
        assert_eq!(op.arg(0).to_opref(), fresh_i0);
        assert_eq!(op.arg(1).to_opref(), fresh_i1);
        assert!(it.done());
        assert!(it.next().is_none());
    }

    #[test]
    fn byte_trace_iter_seeds_sparse_live_inputargs() {
        // History.set_inputargs keeps sparse get_position() (0 and 2).
        // Reminting the reserved prefix would type the second live Int as
        // the dead Ref when zipped by compact index.
        let mut buf = TraceRecordBuffer::new(3, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_input_arg(Type::Ref);
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(2), None);
        let live = vec![InputArg::new_int(0), InputArg::new_int(2)];
        let mut it =
            ByteTraceIter::new_with_inputargs(&buf, buf._start as usize, buf._pos, &live, 1000);
        assert_eq!(it.inputargs[0].opref(), iarg(1000));
        assert_eq!(it.inputargs[1].opref(), iarg(1001));
        assert_eq!(it.inputargs[0].tp.get(), Type::Int);
        assert_eq!(it.inputargs[1].tp.get(), Type::Int);
        let add = it.next().expect("one op");
        assert_eq!(add.arg(0).to_opref(), iarg(1000));
        assert_eq!(add.arg(1).to_opref(), iarg(1001));
    }

    /// M4 step 1: chained ops — the second op references the first
    /// op's result.  `_cache` / `_index` bookkeeping must keep the
    /// per-iteration OpRefs coherent across multiple `next()` calls.
    #[test]
    fn test_byte_trace_iter_chained_m4() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        // r2 = Int(0) + Int(1);  r3 = r2 * Int(0)
        let r2 = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let _r3 = buf.record_op2(OpCode::IntMul, Box::ResOp(r2), Box::ResOp(0), None);

        let mut it = buf.get_byte_iter();
        let fresh_i0 = it.inputargs[0].opref();
        let fresh_i1 = it.inputargs[1].opref();
        let add = it.next().expect("IntAdd");
        assert_eq!(add.opcode, OpCode::IntAdd);
        assert_eq!(add.arg(0).to_opref(), fresh_i0);
        assert_eq!(add.arg(1).to_opref(), fresh_i1);
        let mul = it.next().expect("IntMul");
        assert_eq!(mul.opcode, OpCode::IntMul);
        // The first arg of IntMul referenced the first op's result via
        // TAGBOX.  The iterator's `_cache` must map raw position 2 →
        // `add.pos` (the fresh OpRef emitted one `next()` ago).
        assert_eq!(mul.arg(0).to_opref(), add.pos().get());
        assert_eq!(mul.arg(1).to_opref(), fresh_i0);
        assert!(it.done());
    }

    /// All `TAG{INT,CONSTPTR,CONSTOTHER}` args mint
    /// inline-Const OpRefs (`history.py/268/314 Const{Int,Float,Ptr}.value`
    /// are inline) so the byte iterator no longer requires a
    /// ConstantPool to decode constant arguments. The op-graph GC walker
    /// (`MetaInterp::walk_active_trace_refs` /
    /// `walk_partial_trace_refs`) forwards `ConstPtr(GcRef)`
    /// slots across minor collection.
    #[test]
    fn test_byte_trace_iter_decodes_tagconstptr_inline_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_input_arg(Type::Ref);
        let _ = buf.record_op1(OpCode::SameAsR, Box::ConstPtr(0xdead_beef), None);
        let mut it = buf.get_byte_iter();
        let op = it.next().expect("SameAsR");
        assert_eq!(op.opcode, OpCode::SameAsR);
        assert_eq!(
            op.arg(0).to_opref().as_const_ptr(),
            Some(majit_ir::GcRef(0xdead_beef))
        );
    }

    /// M4: a TAGINT arg round-trips to an inline-Const OpRef whose value
    /// matches the recorded `Box::ConstInt(v)`.
    #[test]
    fn test_byte_trace_iter_tagint_with_pool_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(42), None);

        let mut it = ByteTraceIter::new(&buf, buf._start as usize, buf._pos, buf.max_num_inputargs);
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::IntAdd);
        let const_arg = op.arg(1).to_opref();
        assert!(const_arg.is_constant());
        // The TAGINT arg must round-trip to inline 42 (Int).
        drop(it);
        assert_eq!(
            const_arg.inline_const_to_value(),
            Some(majit_ir::Value::Int(42))
        );
    }

    /// M4: same-value TAGINT args round-trip to the same Value
    /// inline. `history.py:220 ConstInt.__init__` is
    /// fresh-alloc per call so the two OpRefs may differ; the
    /// upstream value-equality predicate is `Const.same_constant`
    /// (history.py:204).
    #[test]
    fn test_byte_trace_iter_tagint_same_value_resolves_via_same_constant_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(7), None);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ConstInt(7), Box::ResOp(0), None);

        let mut it = ByteTraceIter::new(&buf, buf._start as usize, buf._pos, buf.max_num_inputargs);
        let first = it.next().unwrap();
        let second = it.next().unwrap();
        drop(it);
        use crate::heapcache::SameConstantOracle;
        assert!(
            crate::history::ConstOprefOracle
                .same_constant(first.arg(1).to_opref(), second.arg(0).to_opref())
        );
        assert_eq!(
            first.arg(1).to_opref().inline_const_to_value(),
            Some(majit_ir::Value::Int(7))
        );
        assert_eq!(
            second.arg(0).to_opref().inline_const_to_value(),
            Some(majit_ir::Value::Int(7))
        );
    }

    /// M4 step 3: guard opcode decode — the decoded Op's `descr` is
    /// `None` (guards resolve snapshot placeholders separately) but its
    /// `rd_resume_position` carries the encoded descr index.
    #[test]
    fn test_byte_trace_iter_guard_rd_resume_position_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        // `record_op1` encodes a guard with descr_index=0 placeholder.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Patch the placeholder to a concrete snapshot index.
        buf.patch_last_guard_descr_slot(7);

        let mut it = buf.get_byte_iter();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::GuardTrue);
        assert!(!op.has_descr()); // guards do NOT carry a resolved descr
        assert_eq!(op.rd_resume_position(), 7); // opencoder.py setup_resume_point parity
    }

    /// M4 step 3: non-guard descr-bearing opcode routes through the
    /// local `_descrs` pool (for descrs without a global
    /// `get_descr_index`).  Decoded `descr` must resolve back to the
    /// same DescrRef that `record_op_with_descr` interned.
    #[test]
    fn test_byte_trace_iter_local_descr_m4() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct LocalDescr {
            tag: u32,
        }
        impl majit_ir::Descr for LocalDescr {
            fn index(&self) -> u32 {
                self.tag
            }
        }

        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let descr: majit_ir::DescrRef = Arc::new(LocalDescr { tag: 99 });
        // `record_op1` with a `Some(&descr)` exercises the
        // `_encode_descr` → local `_descrs` append path.
        let _ = buf.record_op1(OpCode::GetfieldGcI, Box::ResOp(0), Some(&descr));

        let mut it = buf.get_byte_iter();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::GetfieldGcI);
        let resolved = op.getdescr().expect("descr must resolve");
        assert_eq!(resolved.index(), 99);
        assert_eq!(op.rd_resume_position(), -1); // non-guard sentinel
    }

    /// M4 step 3: non-guard descr-bearing opcode with `descr_index == 0`
    /// (no-descr placeholder) decodes back to `descr: None` without
    /// panicking in the local pool lookup.  Uses `record_op1(_, _, None)`
    /// on a has_descr opcode.
    #[test]
    fn test_byte_trace_iter_descr_placeholder_zero_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op1(OpCode::GetfieldGcI, Box::ResOp(0), None);

        let mut it = buf.get_byte_iter();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::GetfieldGcI);
        assert!(!op.has_descr());
        assert_eq!(op.rd_resume_position(), -1);
    }

    // ── SnapshotIterator::get / unpack_array parity tests ──

    /// opencoder.py `SnapshotIterator.get(index)` — resolve
    /// a tagged value via `main_iter._untag`.  Build a real empty
    /// top-snapshot so `SnapshotIterator::new` has a valid record
    /// header to parse, then verify TAGBOX / TAGINT resolution.
    #[test]
    fn test_snapshot_iterator_get_via_main_iter() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        // `create_empty_top_snapshot` invokes `patch_last_guard_descr_slot`,
        // which requires `_pos >= 2`.  Record a guard op first so the
        // byte stream has a patch target.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Write an empty top snapshot so snapshot_data has the valid
        // header SnapshotIterator::new expects.
        let snap_idx = buf.create_empty_top_snapshot(&[], &[]);

        let mut main_iter =
            ByteTraceIter::new(&buf, buf._start as usize, buf._pos, buf.max_num_inputargs);

        let snap_it = SnapshotIterator::new(
            &buf._snapshot_data,
            &buf._snapshot_array_data,
            snap_idx as usize,
        );
        // TAGBOX(0) → inputarg[0]'s fresh OpRef (pre-seeded in
        // `_cache` at ByteTraceIter::new).
        let tagbox = TraceRecordBuffer::_encode_box_position(0);
        let resolved_box = snap_it.get(tagbox, &mut main_iter);
        let fresh_i0 = main_iter.inputargs[0].opref();
        assert_eq!(resolved_box, fresh_i0);
        // TAGINT(42) → pool-allocated constant OpRef.
        let tagint = TraceRecordBuffer::_encode_smallint(42);
        let resolved_int = snap_it.get(tagint, &mut main_iter);
        assert!(resolved_int.is_constant());
    }

    /// opencoder.py `unpack_array(arr)` — `[self.get(i) for i
    /// in arr]`.  Build a 2-element tagged box array, iterate it
    /// through `unpack_array`, and verify each tagged slot resolves
    /// to the expected fresh-iterator OpRef.
    #[test]
    fn test_snapshot_iterator_unpack_array() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_input_arg(Type::Int);

        // Write a 2-element tagged box array into `_snapshot_array_data`.
        let array_idx = buf.new_array(2);
        buf._add_box_to_storage_box(Box::ResOp(0));
        buf._add_box_to_storage_box(Box::ResOp(1));

        // `create_empty_top_snapshot` patches the preceding guard; a
        // guard op needs to precede the snapshot write.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Write an empty top snapshot so SnapshotIterator::new has a
        // valid record header (parses vable/vref indices from it).
        let snap_idx = buf.create_empty_top_snapshot(&[], &[]);

        let mut main_iter =
            ByteTraceIter::new(&buf, buf._start as usize, buf._pos, buf.max_num_inputargs);
        let expected_0 = main_iter.inputargs[0].opref();
        let expected_1 = main_iter.inputargs[1].opref();

        let snap_it = SnapshotIterator::new(
            &buf._snapshot_data,
            &buf._snapshot_array_data,
            snap_idx as usize,
        );
        let arr = BoxArrayIter::new(&buf._snapshot_array_data, array_idx);
        let unpacked = snap_it.unpack_array(arr, &mut main_iter);
        assert_eq!(unpacked, vec![expected_0, expected_1]);
    }

    #[test]
    fn snapshot_box_array_reports_its_remaining_length() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let array_idx = buf.new_array(3);
        for i in 0..3 {
            buf._add_box_to_storage_box(Box::ResOp(i));
        }
        let mut array = BoxArrayIter::new(&buf._snapshot_array_data, array_idx);
        for remaining in (1..=3).rev() {
            assert_eq!(array.len(), remaining);
            assert_eq!(array.size_hint(), (remaining, Some(remaining)));
            assert!(array.next().is_some());
        }
        assert_eq!(array.len(), 0);
        assert_eq!(array.next(), None);
        assert_eq!(array.size_hint(), (0, Some(0)));
        assert_eq!(BoxArrayIter::new(&[], 0).len(), 0);
    }

    /// `_encode_smallint` must preserve the sign bit so TAGINT decode
    /// through `ByteTraceIter::_untag` round-trips negative values like
    /// Python's arithmetic shift in opencoder.py.
    #[test]
    fn test_encode_smallint_preserves_negative_sign() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(-7), None);

        let mut it = ByteTraceIter::new(&buf, buf._start as usize, buf._pos, buf.max_num_inputargs);
        let op = it.next().expect("one op");
        let const_arg = op.arg(1).to_opref();
        drop(it);
        assert_eq!(
            const_arg.inline_const_to_value(),
            Some(majit_ir::Value::Int(-7))
        );
    }

    /// Phase B11: `tracing_done()` returns `Err(AbortReason::TooLong)`
    /// iff `tag_overflow` was tripped while recording, and clears the
    /// dedup dictionaries on success (opencoder.py).
    #[test]
    fn test_tracing_done_clears_dicts_b11() {
        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        // Intern a couple of constants so the dedup dicts are non-empty.
        let _ = buf._encode_bigint(1 << 40);
        let _ = buf._encode_ptr(0xdead_beef);
        assert!(!buf._bigints_dict.is_empty());
        assert!(!buf._refs_dict.is_empty());
        // Preserve the interned values — they back the TraceIterator.
        let bigints_len = buf._bigints.len();
        let refs_len = buf._refs.len();

        assert_eq!(buf.tracing_done(), Ok(()));
        assert!(buf._bigints_dict.is_empty());
        assert!(buf._refs_dict.is_empty());
        assert_eq!(buf._bigints.len(), bigints_len);
        assert_eq!(buf._refs.len(), refs_len);
    }

    /// Phase B11: an `append_int` outside [MIN_VALUE, MAX_VALUE] trips
    /// `tag_overflow`, which `tracing_done()` reports as
    /// `AbortReason::TooLong` — mirroring
    /// `raise SwitchToBlackhole(Counters.ABORT_TOO_LONG)`.
    #[test]
    fn test_tracing_done_too_long_b11() {
        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        buf.append_int(MAX_VALUE + 1);
        assert!(
            buf.tag_overflow,
            "append_int out of range must trip tag_overflow"
        );
        assert_eq!(
            buf.tracing_done(),
            Err(crate::pyjitpl::AbortReason::TooLong)
        );
    }

    /// Phase B11: `tag_overflow_imminent` flips once `_pos > 0.8 *
    /// MAX_VALUE`, letting the tracer force-close the trace before
    /// the hard overflow is recorded.
    #[test]
    fn test_tag_overflow_imminent_threshold_b11() {
        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        assert!(!buf.tag_overflow_imminent());
        // Fake the cursor forward past 80% of MAX_VALUE.
        buf._pos = (MAX_VALUE as f64 * 0.8) as usize + 1;
        assert!(buf.tag_overflow_imminent());
    }

    /// Phase B9: cut_point/cut_at shape matches opencoder.py.
    /// `cut_point` returns the 5-tuple (_pos, _count, _index,
    /// len(_snapshot_data), len(_snapshot_array_data)). `cut_at`
    /// restores _pos, _count, _index; the snapshot-chain lengths in
    /// slots 3/4 are observed but NOT rewound (they grow monotonically
    /// in RPython; bridge compilation reuses earlier snapshots).
    #[test]
    fn test_cut_point_five_tuple_b9() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let before = buf.cut_point();
        assert_eq!(before._pos, buf._pos);
        assert_eq!(before._count, buf._count);
        assert_eq!(before._index, buf._index);
        assert_eq!(before.snapshot_data_len, buf._snapshot_data.len());
        assert_eq!(
            before.snapshot_array_data_len,
            buf._snapshot_array_data.len()
        );

        // Advance all five independently.
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(7));
        let _ = buf.create_top_snapshot(1, 10, array, &[], &[], true);
        let after = buf.cut_point();
        assert_ne!(after._pos, before._pos);
        assert_ne!(after._count, before._count); // _count bumped by guard.
        assert_eq!(after._index, before._index); // _index did NOT bump (guard is void).
        assert_ne!(after.snapshot_data_len, before.snapshot_data_len);
        assert_ne!(
            after.snapshot_array_data_len,
            before.snapshot_array_data_len
        );

        // cut_at restores only the first three fields.
        buf.cut_at(before);
        assert_eq!(buf._pos, before._pos);
        assert_eq!(buf._count, before._count);
        assert_eq!(buf._index, before._index);
    }

    /// Phase B7 smoke test: record a guard, then call
    /// `create_top_snapshot` with small box arrays. Verify:
    /// - The guard's trailing 0-placeholder is patched to the snapshot
    ///   index returned by `create_top_snapshot`.
    /// - `_snapshot_data` grew, `_snapshot_array_data` grew for
    ///   non-empty arrays, `_total_snapshots` bumped.
    /// - Length-0 arrays collapse to index 0 (the pre-seeded slot).
    #[test]
    fn test_create_top_snapshot_b7() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        // Record a guard so we have a 0-placeholder to patch.
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let pos_after_guard = buf._pos;
        let total_before = buf._total_snapshots;
        let sd_len_before = buf._snapshot_data.len();

        // Build the active-boxes array via new_array + _add_box_to_storage,
        // mimicking how the frame's get_list_of_active_boxes drives it.
        let array = buf.new_array(2);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(1));
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(2));

        // No vable / vref boxes.
        let s = buf.create_top_snapshot(0, 0, array, &[], &[], true);
        assert_eq!(s as usize, sd_len_before);
        assert_eq!(buf._total_snapshots, total_before + 1);
        assert!(buf._snapshot_data.len() > sd_len_before);

        // The guard's trailing 2 zero bytes should have been replaced
        // with `s` (which fits in 2 bytes since it's 0 here).
        assert_eq!(buf._pos, pos_after_guard);
        let (decoded, _) = decode_varint_signed(&buf.ops_bytes()[pos_after_guard - 2..]);
        assert_eq!(decoded, s, "guard descr slot not patched to snapshot index");
    }

    /// Phase B7 smoke test: create_snapshot patches the previous
    /// snapshot's NEEDS_PATCHING sentinel with SNAPSHOT_PREV_COMES_NEXT
    /// (-1) and then writes a new snapshot with its own
    /// NEEDS_PATCHING sentinel at the end.
    #[test]
    fn test_create_snapshot_prev_chain_b7() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);

        // First: a top snapshot with is_last=false so it leaves a
        // NEEDS_PATCHING sentinel for the next create_snapshot to chain
        // onto.
        let top_array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(7));
        let _top_idx = buf.create_top_snapshot(1, 10, top_array, &[], &[], false);
        // The snapshot_data now ends with NEEDS_PATCHING (bytes 0x7d 0xff).
        let n = buf._snapshot_data.len();
        assert_eq!(buf._snapshot_data[n - 2], 0x7d);
        assert_eq!(buf._snapshot_data[n - 1], 0xff);

        // Now append a parent snapshot: create_snapshot patches the
        // trailing sentinel to SNAPSHOT_PREV_COMES_NEXT and writes the
        // new snapshot record.
        let parent_array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(9));
        let _parent_idx = buf.create_snapshot(0, 5, parent_array, true);
        // The top snapshot's prev slot is now COMES_NEXT (-1 encodes
        // as 0x7f 0xff in the signed-varint 2-byte layout).
        // Decoding at the original offset should yield -1.
        let (decoded, _) = decode_varint_signed(&buf._snapshot_data[n - 2..n]);
        assert_eq!(decoded, SNAPSHOT_PREV_COMES_NEXT as i64);
    }

    /// Phase B6 parity: `_count` advances on EVERY recorded op, `_index`
    /// advances ONLY for ops whose `opclasses[opnum].type != 'v'`
    /// (opencoder.py:660-662). Interleave non-void (IntAdd) and void
    /// (GuardTrue) ops and verify the two counters progress according
    /// to the RPython rule.
    #[test]
    fn test_void_op_index_policy_b6() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        // Initial state: _start = _count = _index = max_num_inputargs = 2.
        assert_eq!(buf._count, 2);
        assert_eq!(buf._index, 2);

        // IntAdd (non-void Int result) — bumps both counters.
        let p1 = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        assert_eq!(p1, 2, "record_op returns pre-bump _index");
        assert_eq!(buf._count, 3);
        assert_eq!(buf._index, 3);

        // GuardTrue (void) — bumps _count only.
        let p2 = buf.record_op1(OpCode::GuardTrue, Box::ResOp(2), None);
        assert_eq!(
            p2, 3,
            "guard returns pre-bump _index (same as last non-void)"
        );
        assert_eq!(buf._count, 4);
        assert_eq!(buf._index, 3, "void must NOT bump _index");

        // IntSub (non-void) — bumps both again. Note _index picks up
        // from 3, not from _count=4, so the next non-void result
        // position is 3.
        let p3 = buf.record_op2(OpCode::IntSub, Box::ResOp(0), Box::ResOp(2), None);
        assert_eq!(p3, 3, "non-void after void uses the void's pre-bump _index");
        assert_eq!(buf._count, 5);
        assert_eq!(buf._index, 4);
    }

    /// Phase B5 smoke test: guard records trailing descr as 0 (two
    /// \\x00 bytes in the 2-byte signed-varint form).
    /// `patch_last_guard_descr_slot` rewinds 2 bytes and replaces them
    /// with the actual snapshot index via append_int — works for both
    /// 2-byte and 4-byte encodings of the patched value.
    #[test]
    fn test_guard_descr_placeholder_patch_b5() {
        // Phase B3's record_op1 writes opnum + arg + descr-int. For a
        // guard with has_descr + descr_index=0, the trailing 2 bytes
        // are the zero-placeholder.
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let pos_before_patch = buf._pos;
        assert_eq!(
            buf.ops_bytes()[pos_before_patch - 2],
            0u8,
            "guard descr placeholder missing"
        );
        assert_eq!(buf.ops_bytes()[pos_before_patch - 1], 0u8);

        // Patch to a small snapshot index (fits in 2 bytes).
        buf.patch_last_guard_descr_slot(42);
        assert_eq!(
            buf._pos, pos_before_patch,
            "2-byte patch should keep _pos at the same offset"
        );
        // Verify we can decode it back.
        let tail_start = pos_before_patch - 2;
        let (decoded, consumed) = decode_varint_signed(&buf.ops_bytes()[tail_start..]);
        assert_eq!(decoded, 42);
        assert_eq!(consumed, 2);

        // Patch to a large snapshot index (requires 4 bytes).
        let mut buf2 = TraceRecordBuffer::new(1, empty_sd());
        buf2.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let pos_before = buf2._pos;
        buf2.patch_last_guard_descr_slot(MAX_VALUE);
        assert_eq!(
            buf2._pos,
            pos_before + 2,
            "4-byte patch should grow _pos by 2"
        );
    }

    /// Phase B4 + S smoke test: `_encode_descr` returns `global_index
    /// + 1` for descrs inside the captured global range and appends the
    /// rest to `_descrs` (opencoder.py `_encode_descr`).
    #[test]
    fn test_encode_descr_reads_metainterp_sd() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct D {
            idx: i32,
        }
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                self.idx
            }
        }

        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        // `empty_sd()` captures whatever the process-global registry holds.
        // Pin the boundary so both tiers are exercised deterministically,
        // and so `idx: 3` below names a slot that is really inside it.
        buf.all_descr_len = 4;

        // Global descr returns `get_descr_index() + 1`.
        let d_global: majit_ir::DescrRef = Arc::new(D { idx: 3 });
        assert_eq!(buf._encode_descr(&d_global), 4);
        assert_eq!(buf._descrs.len(), 1, "global descr must not append");

        // A local descr's encoded index is offset past the boundary.
        let d_local: majit_ir::DescrRef = Arc::new(D { idx: -1 });
        assert_eq!(buf._encode_descr(&d_local), 1 + 4 + 1);
        assert_eq!(buf._descrs.len(), 2);
        assert!(Arc::ptr_eq(
            buf._descrs[1].as_ref().expect("local descr appended"),
            &d_local
        ));
    }

    /// Phase B3 smoke test: fixed-arity record_op* does NOT write a
    /// count varint (opencoder.py), only the opnum byte + args.
    /// Variadic record_op writes opnum + count + args. _count bumps
    /// once per call; _index bumps only for non-void results.
    #[test]
    fn test_record_op_wire_format_b3() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let start_index = buf._index;
        let start_count = buf._count;
        let start_pos = buf._pos;
        // IntAdd is fixed-arity 2, result_type Int → non-void.
        // Args passed as RPython-encoded tagged values.
        let pos = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        assert_eq!(pos, start_index, "record_op returns pre-bump _index");
        assert_eq!(buf._count, start_count + 1);
        assert_eq!(buf._index, start_index + 1, "non-void should bump _index");
        // opnum byte at `start_pos` (= max_num_inputargs = 2, opencoder.py:500),
        // then 2 varints for args. IntAdd has no descr.
        assert_eq!(buf.ops_bytes()[start_pos], OpCode::IntAdd.as_u16() as u8);
        // 2 args × 2-byte varint each = 4 bytes after opnum.
        assert_eq!(buf._pos, start_pos + 1 + 2 + 2, "opnum + 2 varint-2 args");

        // A void op must not bump _index but must bump _count.
        let void_start_index = buf._index;
        let void_start_count = buf._count;
        // GuardTrue is fixed-arity 1, result_type Void, has_descr.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        assert_eq!(buf._index, void_start_index, "void should NOT bump _index");
        assert_eq!(buf._count, void_start_count + 1);
    }

    // ── record_op_oprefs adapter tests ─────────────────────────────────

    /// `record_op_oprefs` with non-constant OpRef args must
    /// emit the same wire bytes as `record_op(&[Box::ResOp(...)])`.
    #[test]
    fn test_record_op_oprefs_resop_parity_2b() {
        let mut expected = TraceRecordBuffer::new(2, empty_sd());
        let mut actual = TraceRecordBuffer::new(2, empty_sd());
        let pos_e = expected.record_op(OpCode::IntAdd, &[Box::ResOp(0), Box::ResOp(1)], None);
        let pos_a = actual.record_op_oprefs(OpCode::IntAdd, &[iarg(0), iarg(1)], None);
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
        assert_eq!(expected._count, actual._count);
        assert_eq!(expected._index, actual._index);
    }

    /// Inline-Const OpRef (Int) must resolve to `Box::ConstInt`
    /// and produce the same bytes as passing `Box::ConstInt(v)` directly.
    #[test]
    fn test_record_op_oprefs_const_int_2b() {
        let c = OpRef::const_int(42);
        assert!(c.is_constant());
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(OpCode::IntAdd, &[Box::ResOp(0), Box::ConstInt(42)], None);
        let pos_a = actual.record_op_oprefs(OpCode::IntAdd, &[iarg(0), c], None);
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
    }

    /// Inline-Const OpRef (Float) must resolve to `Box::ConstFloat`.
    #[test]
    fn test_record_op_oprefs_const_float_2b() {
        let raw = (3.25_f64).to_bits() as i64;
        let c = OpRef::const_float(f64::from_bits(raw as u64));
        assert!(c.is_constant());
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(
            OpCode::FloatAdd,
            &[Box::ResOp(0), Box::ConstFloat(raw as u64)],
            None,
        );
        let pos_a = actual.record_op_oprefs(OpCode::FloatAdd, &[farg(0), c], None);
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
        // Both should have registered the float constant in the pool.
        assert_eq!(&expected._floats[..], &actual._floats[..]);
    }

    /// Inline-Const OpRef (Ref) must resolve to `Box::ConstPtr`.
    #[test]
    fn test_record_op_oprefs_const_ref_2b() {
        let addr = 0xdead_beef_u64;
        let c = OpRef::const_ptr(majit_ir::GcRef(addr as usize));
        assert!(c.is_constant());
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(OpCode::PtrEq, &[Box::ResOp(0), Box::ConstPtr(addr)], None);
        let pos_a = actual.record_op_oprefs(OpCode::PtrEq, &[rarg(0), c], None);
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
        assert_eq!(&expected._refs[..], &actual._refs[..]);
    }

    // ── ops / get_op_by_pos / last_op tests ────────────────────────────

    /// `ops()` walks the byte stream and materializes every
    /// recorded op. Opcodes come back in record order.
    #[test]
    fn test_ops_materializer_2c() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let _ = buf.record_op2(OpCode::IntMul, Box::ResOp(2), Box::ResOp(0), None);
        let ops = buf.ops();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].opcode, OpCode::IntMul);
    }

    #[test]
    fn ops_materializer_returns_the_decoder_producer_identity() {
        let mut trace = TraceRecordBuffer::new(1, empty_sd());
        trace.record_input_arg(Type::Int);
        let position = trace.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(1), None);
        trace.record_op2(OpCode::IntMul, Box::ResOp(position), Box::ConstInt(2), None);
        let ops = trace.ops();
        let producer = ops[1].arg(0).bound_op().unwrap();
        let first: &majit_ir::Op = std::borrow::Borrow::borrow(&ops[0]);
        assert!(std::ptr::eq(first, producer.as_ref()));
    }

    /// `get_op_by_pos` returns the Op whose `.pos` equals the
    /// requested OpRef. `ByteTraceIter` seeds `_fresh` at
    /// `max_num_inputargs` and then bumps it once per inputarg before
    /// any op, so with 2 inputargs the first op has `op.pos == IntOp(4)`.
    #[test]
    fn test_get_op_by_pos_2c() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let _ = buf.record_op2(OpCode::IntMul, Box::ResOp(2), Box::ResOp(0), None);
        let first = buf.get_op_by_pos(iop(4)).expect("first op present");
        assert_eq!(first.opcode, OpCode::IntAdd);
        assert_eq!(first.pos().get(), iop(4));
        let second = buf.get_op_by_pos(iop(5)).expect("second op present");
        assert_eq!(second.opcode, OpCode::IntMul);
        assert!(buf.get_op_by_pos(iop(99)).is_none());
    }

    /// `last_op` returns the final recorded op.  On an empty
    /// buffer the result is `None`.
    #[test]
    fn test_last_op_2c() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        assert!(buf.last_op().is_none());
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let last = buf.last_op().expect("one op recorded");
        assert_eq!(last.opcode, OpCode::IntAdd);
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let last = buf.last_op().expect("two ops recorded");
        assert_eq!(last.opcode, OpCode::GuardTrue);
    }

    // ── guard / close_loop / finish helpers tests ──────────────────────

    fn fixed_descr() -> majit_ir::DescrRef {
        use std::sync::Arc;
        #[derive(Debug)]
        struct D;
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                0
            }
        }
        Arc::new(D)
    }

    /// `close_loop_oprefs` records a JUMP with no descr.
    #[test]
    fn test_close_loop_oprefs_2d() {
        let mut expected = TraceRecordBuffer::new(2, empty_sd());
        let mut actual = TraceRecordBuffer::new(2, empty_sd());
        expected.record_input_arg(Type::Int);
        expected.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        let pos_e = expected.record_op(OpCode::Jump, &[Box::ResOp(0), Box::ResOp(1)], None);
        let pos_a = actual.close_loop_oprefs(&[OpRef::input_arg_int(0), OpRef::input_arg_int(1)]);
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
    }

    /// `close_loop_oprefs_with_descr` records a JUMP with the
    /// given tentative descr.  Matches
    /// `record_op(Jump, args, Some(&descr))`.
    #[test]
    fn test_close_loop_oprefs_with_descr_2d() {
        let descr = fixed_descr();
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        expected.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        let pos_e = expected.record_op(OpCode::Jump, &[Box::ResOp(0)], Some(&descr));
        let pos_a = actual.close_loop_oprefs_with_descr(&[OpRef::input_arg_int(0)], Some(&descr));
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
    }

    /// `finish_oprefs` records a FINISH op with its terminal
    /// FailDescr.
    #[test]
    fn test_finish_oprefs_2d() {
        let descr = fixed_descr();
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        expected.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        let pos_e = expected.record_op(OpCode::Finish, &[Box::ResOp(0)], Some(&descr));
        let pos_a = actual.finish_oprefs(&[OpRef::input_arg_int(0)], &descr);
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
    }

    /// `record_op_oprefs` with a descr must encode the descr
    /// exactly as `record_op(&[Box], Some(&descr))` would.
    #[test]
    fn test_record_op_oprefs_with_descr_2b() {
        let descr = fixed_descr();
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(OpCode::CallN, &[Box::ResOp(0)], Some(&descr));
        let pos_a =
            actual.record_op_oprefs(OpCode::CallN, &[OpRef::input_arg_int(0)], Some(&descr));
        assert_eq!(pos_e, pos_a);
        assert_eq!(
            expected.ops_bytes()[..expected._pos],
            actual.ops_bytes()[..actual._pos]
        );
        assert_eq!(expected._descrs.len(), actual._descrs.len());
    }

    /// A descr whose global `descr_index` is fixed at construction, so a
    /// test can place it inside or past a trace's captured `all_descr_len`.
    fn descr_with_index(index: i32) -> majit_ir::DescrRef {
        use std::sync::Arc;
        #[derive(Debug)]
        struct D(i32);
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                self.0
            }
        }
        Arc::new(D(index))
    }

    /// `_encode_descr` splits global from local against the boundary
    /// `Trace::new` captured, never against the live process-global list.
    #[test]
    fn encode_descr_classifies_against_the_captured_boundary() {
        let mut t = TraceRecordBuffer::new(1, empty_sd());
        t.all_descr_len = 3;

        // Inside the boundary: global, encoded as `descr_index + 1` and
        // not copied into `_descrs`.
        assert_eq!(t._encode_descr(&descr_with_index(2)), 3);
        assert_eq!(t._descrs.len(), 1);

        // At the boundary: local, even though it carries a global index.
        // Such a descr was stamped after this trace captured its boundary,
        // so the decoder would read `descr_index + 1` as a local slot.
        assert_eq!(t._encode_descr(&descr_with_index(3)), 1 + 3 + 1);
        assert_eq!(t._descrs.len(), 2);

        // Never stamped: local, as upstream.
        assert_eq!(t._encode_descr(&descr_with_index(-1)), 2 + 3 + 1);
        assert_eq!(t._descrs.len(), 3);
    }

    /// Round-trip for a descr carrying a global index at or past the
    /// trace's captured boundary. Encoding it as `descr_index + 1` puts it
    /// in the decoder's local range, which resolves against `_descrs` — a
    /// different descr, or an out-of-bounds slot.
    #[test]
    fn a_descr_stamped_past_the_captured_boundary_round_trips() {
        let mut t = TraceRecordBuffer::new(2, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let descr = descr_with_index(5);
        assert!(
            t.all_descr_len <= 5,
            "test needs a descr index at or past the captured boundary"
        );
        t.record_op(OpCode::CallN, &[Box::ResOp(i0.raw())], Some(&descr));

        let mut it = ByteTraceIter::new(&t, t._start as usize, t._pos, t.max_num_inputargs);
        let op = it.next().expect("recorded CallN");
        let decoded = op.getdescr().expect("CallN kept its descr");
        assert!(
            std::sync::Arc::ptr_eq(&decoded, &descr),
            "decoded descr is not the one that was recorded"
        );
    }

    /// The decoder offsets a local slot by the boundary the trace captured,
    /// not by the live length of the process-global list. Those differ as
    /// soon as `pyjitpl.rs take_back_all_descrs` publishes a longer list
    /// between a trace's encode and a later decode of it.
    #[test]
    fn decode_resolves_a_local_descr_against_the_captured_boundary() {
        let mut t = TraceRecordBuffer::new(2, empty_sd());
        // As if the list had held four descrs when this trace was created.
        t.all_descr_len = 4;
        let i0 = t.record_input_arg(Type::Int);
        let descr = descr_with_index(-1);
        t.record_op(OpCode::CallN, &[Box::ResOp(i0.raw())], Some(&descr));

        let mut it = ByteTraceIter::new(&t, t._start as usize, t._pos, t.max_num_inputargs);
        let op = it.next().expect("recorded CallN");
        let decoded = op.getdescr().expect("CallN kept its descr");
        assert!(
            std::sync::Arc::ptr_eq(&decoded, &descr),
            "decoded descr is not the one that was recorded"
        );
    }

    #[test]
    fn test_trace_cut_point_roundtrip_b1() {
        // opencoder.py cut_point / cut_at tuple shape:
        // (_pos, _count, _index, len(_snapshot_data), len(_snapshot_array_data)).
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let p0 = buf.cut_point();
        // append_byte advances _pos; append_snapshot_*_int extends the chains.
        buf.append_byte(0x42);
        buf.append_snapshot_data_int(123);
        buf.append_snapshot_array_data_int(42);
        let p1 = buf.cut_point();
        assert_ne!(p0, p1);
        buf.cut_at(p0);
        assert_eq!(buf._pos, p0._pos);
        assert_eq!(buf._count, p0._count);
        assert_eq!(buf._index, p0._index);
        // RPython's cut_at intentionally does NOT rewind the snapshot chains
        // (they grow monotonically). Only _pos/_count/_index come back.
    }

    /// `SnapshotIterator`: an empty top snapshot
    /// (`create_empty_top_snapshot`, jitcode_index == -1) has an empty
    /// framestack and `size == vable.total_length + vref.total_length + 3`.
    /// Mirrors opencoder.py early-return branch.
    #[test]
    fn test_snapshot_iterator_empty_top() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        // Need a preceding guard for patch_last_guard_descr_slot.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let vable = [TraceRecordBuffer::_encode_box_position(0)];
        let vref: [i64; 0] = [];
        let snap = buf.create_empty_top_snapshot(&vable, &vref);

        let it = buf.get_snapshot_iter(snap as usize);
        assert!(it.framestack.is_empty(), "empty top → empty framestack");
        // vable has 1 entry, vref has 0, so size = 1 + 0 + 3 = 4.
        assert_eq!(it.size, 4);
        assert_eq!(it.iter_vable_array().total_length, 1);
        assert_eq!(it.iter_vref_array().total_length, 0);
    }

    /// `SnapshotIterator`: `capture_resumedata` with a single
    /// frame produces a non-empty framestack in bottom-up order with
    /// outermost == innermost (length 1).
    #[test]
    fn test_snapshot_iterator_single_frame() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Build the innermost-frame active-box array.
        let arr = buf.new_array(2);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(0));
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(1));
        let vable: [i64; 0] = [];
        let vref: [i64; 0] = [];
        // jitcode_index=7, pc=42.
        let snap = buf.create_top_snapshot(7, 42, arr, &vable, &vref, true);

        let it = buf.get_snapshot_iter(snap as usize);
        assert_eq!(it.framestack.len(), 1, "one frame in framestack");
        let (jit_idx, pc) = it.unpack_jitcode_pc(it.framestack[0]);
        assert_eq!(jit_idx, 7);
        assert_eq!(pc, 42);
        let items: Vec<i64> = it.iter_array(it.framestack[0]).collect();
        assert_eq!(items.len(), 2);
    }

    /// `capture_resumedata` on a framestack with a single
    /// frame drives `create_top_snapshot_from_frame` and returns a
    /// snapshot_index consistent with `get_snapshot_iter`.
    /// Parent chain is empty (n == 0 ⇒ is_last = true), so no
    /// `create_snapshot_from_frame` calls fire.
    #[test]
    fn test_capture_resumedata_single_frame_c() {
        use std::sync::Arc;

        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);

        // Build a minimal jitcode with 1 int reg and a LIVE op at pc 0.
        let mut builder = crate::jitcode::JitCodeBuilder::new();
        let mut jc = builder.finish();
        const LIVE_OP: u8 = 0x42;
        {
            let body = jc.body_mut();
            body.c_num_regs_i = 1;
            body.c_num_regs_r = 0;
            body.c_num_regs_f = 0;
            body.code = vec![LIVE_OP, 0x00, 0x00];
        }
        jc.set_index(11);
        let jc = Arc::new(jc);

        // all_liveness: len_i=1 len_r=0 len_f=0, bitmask = 0b1.
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0001];

        let mut frame = crate::pyjitpl::MIFrame::new(jc, 0);
        frame.pc = 0;
        frame.int_regs[0] = Some(majit_ir::OpRef::int_op(1));

        let mut stack = vec![frame];
        let snap = buf.capture_resumedata(
            &mut stack,
            &[],
            &[],
            /* clear_result_register */ false,
            LIVE_OP,
            &all_liveness,
            /* after_residual_call */ true,
        );

        let it = buf.get_snapshot_iter(snap as usize);
        assert_eq!(it.framestack.len(), 1, "single-frame capture");
        let (j, p) = it.unpack_jitcode_pc(it.framestack[0]);
        assert_eq!(j, 11, "frame.jitcode.index encoded into snapshot");
        assert_eq!(p, 0, "frame.pc encoded into snapshot");
        let boxes: Vec<i64> = it.iter_array(it.framestack[0]).collect();
        assert_eq!(
            boxes,
            vec![TraceRecordBuffer::_encode_box_position(1)],
            "live int register emitted via get_list_of_active_boxes"
        );
    }

    /// `capture_resumedata` with an empty framestack takes
    /// the `create_empty_top_snapshot_from_boxes` branch. Snapshot's
    /// `jitcode_index == -1`, `framestack` empty.
    #[test]
    fn test_capture_resumedata_empty_framestack_c() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);

        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let snap = buf.capture_resumedata(&mut stack, &[], &[], false, 0, &[], false);
        let it = buf.get_snapshot_iter(snap as usize);
        assert!(
            it.framestack.is_empty(),
            "empty framestack → empty snapshot"
        );
    }

    /// `SnapshotIterator`: multi-frame chain produces a
    /// framestack ordered outermost-first. Simulates RPython's
    /// capture_resumedata + `_ensure_parent_resumedata` pattern by
    /// emitting one `create_top_snapshot` followed by
    /// `create_snapshot` calls for parent frames.
    #[test]
    fn test_snapshot_iterator_multi_frame_order() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // innermost frame (n=2, jit=3 pc=30)
        let inner = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(0));
        let vable: [i64; 0] = [];
        let vref: [i64; 0] = [];
        let snap = buf.create_top_snapshot(3, 30, inner, &vable, &vref, false);
        // middle frame (n=1, jit=2 pc=20)
        let middle = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(1));
        let _ = buf.create_snapshot(2, 20, middle, false);
        // outermost frame (n=0, jit=1 pc=10)
        let outer = buf.new_array(0);
        let _ = buf.create_snapshot(1, 10, outer, true);

        let it = buf.get_snapshot_iter(snap as usize);
        assert_eq!(it.framestack.len(), 3, "three frames in chain");
        // opencoder.py `framestack.reverse()` → bottom-up
        // (outermost first).
        let (j0, p0) = it.unpack_jitcode_pc(it.framestack[0]);
        let (j1, p1) = it.unpack_jitcode_pc(it.framestack[1]);
        let (j2, p2) = it.unpack_jitcode_pc(it.framestack[2]);
        assert_eq!((j0, p0), (1, 10), "framestack[0] == outermost");
        assert_eq!((j1, p1), (2, 20), "framestack[1] == middle");
        assert_eq!((j2, p2), (3, 30), "framestack[2] == innermost");
    }

    // ── Phase C: RPython test_opencoder.py parity ports ─────────────────
    // Line-by-line ports of the canonical tests in
    // `rpython/jit/metainterp/test/test_opencoder.py`. Each Rust test cites
    // the upstream line it mirrors so regressions stay traceable to the
    // Python fixture.

    /// test_opencoder.py `TestOpencoder.test_simple_iterator` —
    /// canonical two-op `IntAdd` chain with a small-int constant argument.
    /// Ensures `record_op2` + `ByteTraceIter::new` round-trip the op list
    /// with the expected arity, opcode, and arg identities.
    #[test]
    fn test_simple_iterator_c() {
        let mut t = TraceRecordBuffer::new(2, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        // add = t.record_op(rop.INT_ADD, [i0, i1])
        let add_pos = t.record_op2(
            OpCode::IntAdd,
            Box::ResOp(i0.raw()),
            Box::ResOp(i1.raw()),
            None,
        );
        // t.record_op(rop.INT_ADD, [add, ConstInt(1)])
        let _ = t.record_op2(OpCode::IntAdd, Box::ResOp(add_pos), Box::ConstInt(1), None);

        // `self.unpack(t)` equivalent — ConstInt(1) decodes inline.
        let mut it = ByteTraceIter::new(&t, t._start as usize, t._pos, t.max_num_inputargs);
        let fresh_i0 = it.inputargs[0].opref();
        let fresh_i1 = it.inputargs[1].opref();
        let l0 = it.next().expect("first IntAdd");
        let l1 = it.next().expect("second IntAdd");
        assert!(it.done(), "only two ops recorded");
        // assert l[0].opnum == rop.INT_ADD
        assert_eq!(l0.opcode, OpCode::IntAdd);
        // assert l[1].opnum == rop.INT_ADD
        assert_eq!(l1.opcode, OpCode::IntAdd);
        // assert l[0].getarg(0) is i0; getarg(1) is i1
        assert_eq!(l0.arg(0).to_opref(), fresh_i0);
        assert_eq!(l0.arg(1).to_opref(), fresh_i1);
        // assert l[1].getarg(0) is l[0]
        assert_eq!(l1.arg(0).to_opref(), l0.pos().get());
        // assert l[1].getarg(1).getint() == 1 — inline-resolved constant.
        drop(it);
        assert_eq!(
            l1.arg(1).to_opref().inline_const_to_value(),
            Some(majit_ir::Value::Int(1))
        );
    }

    /// test_opencoder.py `test_constint_small` —
    /// SMALL_INT values encode to 2 or 4 bytes depending on the varint
    /// boundary, leave `_consts_bigint` untouched (no bigint pool use),
    /// and round-trip to the original value via `ByteTraceIter::_untag`.
    ///
    /// RPython uses Hypothesis; pyre iterates a hand-picked representative
    /// sample that spans the 2-byte / 4-byte split (upstream's `-2**12`
    /// boundary) plus SMALL_INT_START / SMALL_INT_STOP extremes.
    #[test]
    fn test_constint_small_c() {
        // Representative sample matching RPython's Hypothesis strategy
        // `integers(SMALL_INT_START, SMALL_INT_STOP - 1)`.
        let sample: &[i64] = &[
            0,
            1,
            -1,
            127,
            -128,
            (1 << 12) - 1,
            -(1 << 12),
            (1 << 12),
            -(1 << 12) - 1,
            SMALL_INT_START,
            SMALL_INT_STOP - 1,
        ];
        for &num in sample {
            let mut t = TraceRecordBuffer::new(0, empty_sd());
            // t.append_int(t._encode(ConstInt(num))) — SMALL_INT path.
            t.append_int(TraceRecordBuffer::_encode_smallint(num));
            // assert t._consts_bigint == 0
            assert_eq!(
                t._consts_bigint, 0,
                "SMALL_INT must not hit bigint pool for {num}"
            );
            // _pos counts only the appended varint (no inputargs reserve
            // because `max_num_inputargs == 0`).
            let expected_len = if (-(1i64 << 12)..(1i64 << 12)).contains(&num) {
                2
            } else {
                4
            };
            assert_eq!(
                t._pos, expected_len,
                "SMALL_INT {num} should encode as {expected_len} bytes, got {}",
                t._pos,
            );
            // Round-trip: `it._next()` reads the tagged i64, `_untag`
            // returns an inline-Const OpRef that resolves to `Value::Int(num)`.
            let mut it = ByteTraceIter::new(&t, 0, t._pos, t.max_num_inputargs);
            let tagged = it._next();
            let opref = it._untag(tagged).to_opref();
            drop(it);
            assert_eq!(
                opref.inline_const_to_value(),
                Some(majit_ir::Value::Int(num)),
                "SMALL_INT {num} round-trip via ByteTraceIter",
            );
        }
    }

    /// test_opencoder.py `TestOpencoder.test_liveranges` —
    /// NEW_WITH_VTABLE + GUARD_TRUE + capture_resumedata with non-empty
    /// vable/vref arrays but an empty framestack. Drives the snapshot
    /// chain through `update_liveranges` and asserts every input +
    /// NEW_WITH_VTABLE slot is marked live at the guard's index (4).
    #[test]
    fn test_liveranges_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let i2 = t.record_input_arg(Type::Int);
        // p0 = t.record_op(rop.NEW_WITH_VTABLE, [], descr=SomeDescr())
        let some_descr: majit_ir::DescrRef = majit_ir::descr::make_size_descr(64);
        let p0 = t.record_op0(OpCode::NewWithVtable, Some(&some_descr));
        // t.record_op(rop.GUARD_TRUE, [i0])
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(i0.raw()), None);
        // t.capture_resumedata([], [i1, i2, p0], [p0, i1])
        let vable = [Box::ResOp(i1.raw()), Box::ResOp(i2.raw()), Box::ResOp(p0)];
        let vref = [Box::ResOp(p0), Box::ResOp(i1.raw())];
        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let _ = t.capture_resumedata(&mut stack, &vable, &vref, false, 0, &[], false);
        // assert t.get_live_ranges() == [4, 4, 4, 4]
        assert_eq!(t.get_live_ranges(), vec![4, 4, 4, 4]);
    }

    /// test_opencoder.py `TestOpencoder.test_cut_trace_from` —
    /// record a prefix, take a cut_point, continue recording, then call
    /// `cut_trace_from(cut_point, new_inputargs)` to produce a CutTrace
    /// that iterates only the post-cut ops with `new_inputargs` acting
    /// as iterator-local inputargs. The first op's args should resolve
    /// to the fresh inputarg OpRefs (not the original trace positions).
    #[test]
    fn test_cut_trace_from_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let _i2 = t.record_input_arg(Type::Int);
        // add1 = t.record_op(rop.INT_ADD, [i0, i1])
        let add1 = t.record_op2(
            OpCode::IntAdd,
            Box::ResOp(i0.raw()),
            Box::ResOp(i1.raw()),
            None,
        );
        // cut_point = t.cut_point()
        let cut_point = t.cut_point();
        // add2 = t.record_op(rop.INT_ADD, [add1, i1])
        let add2 = t.record_op2(OpCode::IntAdd, Box::ResOp(add1), Box::ResOp(i1.raw()), None);
        // t.record_op(rop.GUARD_TRUE, [add2])
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(add2), None);
        // t.record_op(rop.INT_SUB, [add2, add1])
        let _ = t.record_op2(OpCode::IntSub, Box::ResOp(add2), Box::ResOp(add1), None);
        // t2 = t.cut_trace_from(cut_point, [add1, i1])
        // RPython `cut_trace_from(cut, inputargs)` takes BoxInt/Float/Ref objects
        // carrying `Box.type` (history.py:182); pyre's typed-OpRef variants encode
        // the same type tag — `IntOp(p)` for IntAdd's result, the `i1` InputArgInt
        // verbatim.
        let cut = t.cut_trace_from(cut_point, vec![OpRef::int_op(add1), i1]);
        // (i0, i1), l, iter = self.unpack(t2)
        let mut it = cut.get_iter();
        let fresh_add1 = it.inputargs[0].opref();
        let fresh_i1 = it.inputargs[1].opref();
        let mut ops = Vec::new();
        for op in it {
            ops.push(op);
        }
        // assert len(l) == 3
        assert_eq!(ops.len(), 3, "cut sub-trace contains 3 ops");
        // assert l[0].getarglist() == [add1_fresh, i1_fresh]
        // (RPython writes this as `[i0, i1]` because the test rebinds
        // the outer `i0, i1` identifiers to the cut's unpacked
        // inputargs; the pyre port keeps the names distinct.)
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(
            ops[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![fresh_add1, fresh_i1]
        );
        // Second op is the guard; third is the INT_SUB tail.
        assert_eq!(ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(ops[2].opcode, OpCode::IntSub);
    }

    #[test]
    fn cut_trace_reborrows_for_each_walk_and_delegates_the_rewind() {
        let mut trace = TraceRecordBuffer::new(1, empty_sd());
        let input = trace.record_input_arg(Type::Int);
        let start = trace.cut_point();
        let first = trace.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(1), None);
        let end = trace.cut_point();
        trace.record_op2(OpCode::IntMul, Box::ResOp(first), Box::ConstInt(2), None);
        {
            let mut cut = trace.cut_trace_from(start, vec![input]);
            assert_eq!(cut.get_iter().count(), 2);
            // `opencoder.py CutTrace.cut_at` delegates to the same parent
            // that the next get_iter reads, after the old reader has ended.
            cut.cut_at(end);
            assert_eq!(cut.get_iter().count(), 1);
        }
        assert_eq!(trace.cut_point(), end);
        trace.record_op2(OpCode::IntSub, Box::ResOp(first), Box::ConstInt(3), None);
        assert_eq!(trace.ops().last().unwrap().opcode, OpCode::IntSub);
    }

    /// test_opencoder.py `TestOpencoder.test_virtualizable_virtualref` —
    /// assert the vable/vref arrays decode in the canonical order:
    /// `[p0, i1, i2]` (the virtualizable tail box moves to index 0) and
    /// `[p0, i1]` (vref verbatim).
    #[test]
    fn test_virtualizable_virtualref_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let _i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let i2 = t.record_input_arg(Type::Int);
        let some_descr: majit_ir::DescrRef = majit_ir::descr::make_size_descr(64);
        let p0 = t.record_op0(OpCode::NewWithVtable, Some(&some_descr));
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let vable = [Box::ResOp(i1.raw()), Box::ResOp(i2.raw()), Box::ResOp(p0)];
        let vref = [Box::ResOp(p0), Box::ResOp(i1.raw())];
        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let snap = t.capture_resumedata(&mut stack, &vable, &vref, false, 0, &[], false);
        let it = t.get_snapshot_iter(snap as usize);
        // Empty framestack — equivalent to `assert not l[1].framestack`.
        assert!(it.framestack.is_empty());
        let vable_decoded: Vec<i64> = it.iter_vable_array().collect();
        let vref_decoded: Vec<i64> = it.iter_vref_array().collect();
        let p0_tag = TraceRecordBuffer::_encode_box_position(p0);
        let i1_tag = TraceRecordBuffer::_encode_box_position(i1.raw());
        let i2_tag = TraceRecordBuffer::_encode_box_position(i2.raw());
        // `virtualizables == [p0, i1, i2]` after the last-first rotation.
        assert_eq!(vable_decoded, vec![p0_tag, i1_tag, i2_tag]);
        // `vref_boxes == [p0, i1]` verbatim.
        assert_eq!(vref_decoded, vec![p0_tag, i1_tag]);
    }

    /// test_opencoder.py `TestOpencoder.test_virtualizable_bug` —
    /// 128 virtualizable boxes (p0 + 127 × i1) exercise the array-length
    /// varint at the 1-byte / 2-byte threshold. The canonical assertion
    /// is that the snapshot decoder returns the exact same box list in
    /// order; i.e. `SnapshotIterator::unpack_array(vable_array)` must
    /// decode `[p0, i1, i1, ..., i1]` (128 entries) without truncation.
    #[test]
    fn test_virtualizable_bug_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let _i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let _i2 = t.record_input_arg(Type::Int);
        let some_descr: majit_ir::DescrRef = majit_ir::descr::make_size_descr(64);
        let p0 = t.record_op0(OpCode::NewWithVtable, Some(&some_descr));
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // capture_resumedata([], [i1] * 127 + [p0], [p0, i1])
        let mut vable: Vec<Box> = (0..127).map(|_| Box::ResOp(i1.raw())).collect();
        vable.push(Box::ResOp(p0));
        let vref = [Box::ResOp(p0), Box::ResOp(i1.raw())];
        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let snap = t.capture_resumedata(&mut stack, &vable, &vref, false, 0, &[], false);
        // Unpack the snapshot and decode the vable/vref arrays. The
        // 128-entry vable array stresses the varint length field
        // (threshold at 2**7 / 2**14 = 128). If the encoder framed the
        // length as a single byte, the decoder would truncate or spill.
        let it = t.get_snapshot_iter(snap as usize);
        let vable_decoded: Vec<i64> = it.iter_vable_array().collect();
        let vref_decoded: Vec<i64> = it.iter_vref_array().collect();
        assert_eq!(vable_decoded.len(), 128, "128 vable entries decoded");
        assert_eq!(vref_decoded.len(), 2, "2 vref entries decoded");
        // `_list_of_boxes_virtualizable_from_boxes` writes the last entry
        // first, then the remaining forward — matching RPython's
        // `_list_of_boxes(virt)` with an already-rotated list. The
        // decoded stream should therefore open with p0 and continue with
        // 127 × i1.
        let p0_tag = TraceRecordBuffer::_encode_box_position(p0);
        let i1_tag = TraceRecordBuffer::_encode_box_position(i1.raw());
        assert_eq!(vable_decoded[0], p0_tag);
        for (idx, entry) in vable_decoded.iter().enumerate().skip(1) {
            assert_eq!(*entry, i1_tag, "vable[{idx}] == i1");
        }
        assert_eq!(vref_decoded[0], p0_tag);
        assert_eq!(vref_decoded[1], i1_tag);
    }

    /// test_opencoder.py `test_varint_hypothesis` —
    /// `encode_varint_signed` / `decode_varint_signed` round-trip over
    /// the full `[MIN_VALUE, MAX_VALUE]` range, including the case where
    /// the encoded bytes sit behind an arbitrary prefix (decoder's
    /// `start` offset). Pyre samples the hypothesis surface with a
    /// hand-picked set of boundary values.
    #[test]
    fn test_varint_hypothesis_c() {
        let values: &[i64] = &[
            0,
            1,
            -1,
            (1 << 14) - 1,
            -(1 << 14),
            1 << 14,
            -(1 << 14) - 1,
            (1 << 20),
            -(1 << 20),
            MIN_VALUE,
            MAX_VALUE,
        ];
        let prefixes: &[&[u8]] = &[&[], &[0xAA], &[0x00, 0xFF, 0x42]];
        for &i in values {
            let mut encoded = Vec::new();
            encode_varint_signed(&mut encoded, i);
            // Bare decode.
            let (res, consumed) = decode_varint_signed(&encoded);
            assert_eq!(res, i, "bare decode roundtrip for {i}");
            assert_eq!(consumed, encoded.len(), "consumed full encoding for {i}");
            // Decode after an arbitrary prefix — `decode_varint_signed`
            // operates on a slice starting at the varint, so pyre mirrors
            // RPython's `decode_varint_signed(prefix + b, len(prefix))`
            // by slicing off the prefix before the call.
            for &prefix in prefixes {
                let mut combined = prefix.to_vec();
                combined.extend_from_slice(&encoded);
                let (res, consumed) = decode_varint_signed(&combined[prefix.len()..]);
                assert_eq!(res, i, "prefixed decode roundtrip for {i}");
                assert_eq!(consumed, encoded.len(), "prefixed consumed for {i}");
            }
        }
    }
}
