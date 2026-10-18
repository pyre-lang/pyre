//! Resume bytecode encoding/decoding.
//!
//! Direct port of rpython/jit/metainterp/resumecode.py.
//!
//! Encoding: variable-length integers with zigzag encoding.
//! - 7-bit:  0xxxxxxx
//! - 14-bit: 1xxxxxxx 0xxxxxxx
//! - 21-bit: 1xxxxxxx 1xxxxxxx xxxxxxxx

/// resumecode.py: append_numbering(lst, item)
pub fn encode_varint(buf: &mut Vec<u8>, value: i32) {
    let mut item = (value as i64) * 2;
    if item < 0 {
        item = -1 - item;
    }
    assert!(item >= 0);
    let item = item as u32;
    if item < (1 << 7) {
        buf.push(item as u8);
    } else if item < (1 << 14) {
        buf.push((item | 0x80) as u8);
        buf.push((item >> 7) as u8);
    } else {
        // `resumecode.py` `append_numbering` asserts `item < 2**16` because
        // `append_int` feeds it a SHORT. A jitcode pc is a `fix_labels`
        // target (`0 <= target <= 0xFFFF`); its zigzag is `target * 2` and
        // still fits this 3-byte form (`item >> 14` in one byte, so
        // `item < 2**22`).
        assert!(item < (1 << 22), "resumecode item too large: {item}");
        buf.push((item | 0x80) as u8);
        buf.push(((item >> 7) | 0x80) as u8);
        buf.push((item >> 14) as u8);
    }
}

/// resumecode.py: append_numbering(lst, item)
pub fn append_numbering(buf: &mut Vec<u8>, item: i32) {
    encode_varint(buf, item);
}

/// resumecode.py: numb_next_item(numb, index)
///
/// Safe wrapper around the translated `rffi` load. A truncated buffer
/// panics through slice indexing rather than reading off the end.
#[inline]
pub fn decode_varint(buf: &[u8], index: usize) -> (i32, usize) {
    let b0 = buf[index];
    let mut last = index;
    if b0 & (1 << 7) != 0 {
        last = index + 1;
        let b1 = buf[last];
        if b1 & (1 << 7) != 0 {
            last = index + 2;
        }
    }
    let _ = buf[last];
    // SAFETY: `last` is in bounds, so this item's bytes are in `buf`.
    unsafe { decode_varint_at(buf.as_ptr(), buf.len(), index) }
}

/// `resumecode.py` `numb_next_item`. `_always_inline_ = True`: the
/// one-, two-, and three-byte forms are one function.
///
/// # Safety
/// `index` is the start of a complete encoded item in `ptr[..len]`
/// (`Writer.create_numbering` / `NUMBERING.code`). A truncated buffer
/// is a producer bug. Debug panics through `varint_oob`.
#[inline(always)]
unsafe fn decode_varint_at(
    ptr: *const u8,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] len: usize,
    index: usize,
) -> (i32, usize) {
    // `numb_next_item` is `numb.code[index]`.
    #[cfg(debug_assertions)]
    if index >= len {
        varint_oob(ptr, len, index);
    }
    let b0 = unsafe { *ptr.add(index) } as i64;
    if b0 & (1 << 7) == 0 {
        let value = if b0 & 1 != 0 { -1 - b0 } else { b0 };
        return ((value >> 1) as i32, index + 1);
    }
    unsafe { decode_varint_rest(ptr, len, index, b0) }
}

/// # Safety
/// Same as [`decode_varint_at`]: `index` starts a two- or three-byte
/// item that is fully inside `ptr[..len]`.
#[inline(always)]
unsafe fn decode_varint_rest(
    ptr: *const u8,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] len: usize,
    mut index: usize,
    b0: i64,
) -> (i32, usize) {
    let mut value = b0;
    index += 1;
    #[cfg(debug_assertions)]
    if index >= len {
        varint_oob(ptr, len, index);
    }
    value &= (1 << 7) - 1;
    value |= (unsafe { *ptr.add(index) } as i64) << 7;
    index += 1;
    if value & (1 << 14) != 0 {
        #[cfg(debug_assertions)]
        if index >= len {
            varint_oob(ptr, len, index);
        }
        value &= (1 << 14) - 1;
        value |= (unsafe { *ptr.add(index) } as i64) << 14;
        index += 1;
    }
    if value & 1 != 0 {
        value = -1 - value;
    }
    value >>= 1;
    (value as i32, index)
}

#[cfg(debug_assertions)]
#[cold]
#[inline(never)]
fn varint_oob(ptr: *const u8, len: usize, index: usize) -> ! {
    let buf: &[u8] = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let _ = buf[index];
    unreachable!("slice index already panicked");
}

/// resumecode.py: numb_next_item(numb, index)
pub fn numb_next_item(buf: &[u8], index: usize) -> (i32, usize) {
    decode_varint(buf, index)
}

/// resumecode.py numb_next_n_items — skip `size` items without
/// returning values. Used by decoders that advance over a subsection.
pub fn numb_next_n_items(buf: &[u8], size: usize, mut index: usize) -> usize {
    for _ in 0..size {
        let (_, new_index) = decode_varint(buf, index);
        index = new_index;
    }
    index
}

/// resumecode.py create_numbering(l) — module-level helper:
/// build a single Writer from the list and return its encoded buffer.
pub fn create_numbering(items: &[i32]) -> NumberingRef {
    let mut w = Writer::new(items.len());
    for &item in items {
        w.append_int(item as i64);
    }
    w.create_numbering()
}

/// resumecode.py: unpack_numbering(numb)
pub fn unpack_all(buf: &[u8]) -> Vec<i32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let (next, new_i) = decode_varint(buf, i);
        result.push(next);
        i = new_i;
    }
    result
}

/// resumecode.py: unpack_numbering(numb)
pub fn unpack_numbering(buf: &[u8]) -> Vec<i32> {
    unpack_all(buf)
}

/// resumecode.py: Writer
pub struct Writer {
    pub current: Vec<i32>,
}

impl Writer {
    pub fn new(size_hint: usize) -> Self {
        Writer {
            current: Vec::with_capacity(size_hint),
        }
    }

    /// resumecode.py: append_short
    pub fn append_short(&mut self, item: i32) {
        self.current.push(item);
    }

    /// resumecode.py append_int — `short = rffi.cast(rffi.SHORT, item);
    /// assert rffi.cast(lltype.Signed, short) == item`. The upstream
    /// signature is "any int" (RPython `Signed` ≈ machine word); accepting
    /// `i64` here lets the range check apply to the *original* caller value
    /// rather than a silently-truncated copy.
    #[track_caller]
    pub fn append_int(&mut self, item: i64) {
        // `resumecode.py` `append_int` casts through `rffi.SHORT`. A jitcode
        // pc is also a `fix_labels` target (`assert 0 <= target <= 0xFFFF`),
        // so that unsigned range has to round-trip too.
        let fits_short = (item as i16) as i64 == item;
        let fits_label = (0..=u16::MAX as i64).contains(&item);
        assert!(
            fits_short || fits_label,
            "append_int: value {item} out of i16 range and fix_labels u16 range"
        );
        self.append_short(item as i32);
    }

    /// Byte list `Writer.create_numbering` builds before `lltype.malloc`.
    pub fn encode_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.current.len() * 3);
        for &item in &self.current {
            encode_varint(&mut buf, item);
        }
        buf
    }

    /// resumecode.py `Writer.create_numbering`: malloc `NUMBERING` and
    /// copy the encoded bytes into `numb.code`.
    pub fn create_numbering(&self) -> NumberingRef {
        NumberingRef::from_bytes(&self.encode_bytes())
    }

    /// Same object as `create_numbering`. Kept so callers that shared
    /// the old handle name keep compiling.
    pub fn create_numbering_arc(&self) -> NumberingRef {
        self.create_numbering()
    }

    /// resumecode.py: patch_current_size
    pub fn patch_current_size(&mut self, index: usize) {
        self.current[index] = self.current.len() as i32;
    }

    /// resumecode.py: patch
    pub fn patch(&mut self, index: usize, item: i32) {
        self.current[index] = item;
    }
}

/// resumecode.py: Reader
///
/// Holds the numbering slot and an index. Each item re-reads
/// `numb.code[index]` (`numb_next_item`): one plain load of the payload
/// cell. The byte pointer is not cached across an item. The consumer of
/// an item can allocate, and a minor forwards `compile.py`
/// `ResumeGuardDescr.rd_numb`.
pub struct Reader<'a> {
    code: &'a [u8],
    /// `NumberingRef`'s slot address, copied at bind time. Zero means
    /// reads use `code`.
    slot: usize,
    _tie: std::marker::PhantomData<&'a NumberingRef>,
    pub cur_pos: usize,
    pub items_read: usize,
    /// `resumecode.py` `Reader.code`: the numbering payload `numb_next_item`
    /// indexes. A moving collection rewrites `NumSlot.addr`; `stable` is
    /// set only when the rest of this reader's walk does not allocate, so
    /// the pointer cannot go stale (`resume.py` `blackhole_from_resumedata`
    /// with empty virtuals and pending fields).
    payload_ptr: *const u8,
    payload_len: usize,
    stable: bool,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        let (payload_ptr, payload_len) = Self::unbound_payload(code);
        Reader {
            code,
            slot: 0,
            _tie: std::marker::PhantomData,
            cur_pos: 0,
            items_read: 0,
            payload_ptr,
            payload_len,
            // The unbound slice is not a GC numbering; it does not move.
            stable: true,
        }
    }

    /// Subsequent reads follow `numb` across a minor.
    ///
    /// `resumecode.py` `Reader.next_item` calls `numb_next_item`, which
    /// loads `numb.code[index]` from the GC pointer. That pointer is
    /// `Reader.code`; it is not re-fetched from a side cell unless a
    /// collection can rewrite it. `mark_payload_stable` pins it when
    /// the remaining walk does not allocate.
    pub fn bind_numbering(&mut self, numb: &'a NumberingRef) {
        self.slot = numb.slot.as_ptr() as usize;
        self.refresh_from_slot();
        self.stable = false;
    }

    pub fn from_numbering(numb: &'a NumberingRef) -> Self {
        let mut reader = Reader {
            code: &[],
            slot: numb.slot.as_ptr() as usize,
            _tie: std::marker::PhantomData,
            cur_pos: 0,
            items_read: 0,
            payload_ptr: std::ptr::null(),
            payload_len: 0,
            stable: false,
        };
        reader.refresh_from_slot();
        reader
    }

    /// Pin `Reader.code` when the numbering cannot move.
    ///
    /// `numb_next_item` then indexes this pointer the way translated
    /// `Reader.next_item` indexes `self.code` after the GC map has
    /// already forwarded it. `incminimark.py` never moves old objects.
    /// After the first minor, `collect_young_numberings`
    /// (`collect_oldrefs_to_nursery`) has promoted
    /// `compile.py` `ResumeGuardDescr.rd_numb` and the payload is old.
    /// A still-young cell stays on `YOUNG_NUMBERINGS` and the next minor
    /// rewrites `NumSlot.addr`; those reads keep following the slot.
    /// Host blocks do not move.
    #[inline]
    pub fn mark_payload_stable(&mut self) {
        if self.slot != 0 {
            self.refresh_from_slot();
            let slot = unsafe { &*(self.slot as *const NumSlot) };
            self.stable = slot.host || slot.young_index.load(Ordering::Relaxed) == usize::MAX;
            #[cfg(debug_assertions)]
            if self.stable {
                let (ptr, len) = self.payload_from_slot();
                debug_assert_eq!(ptr, self.payload_ptr);
                debug_assert_eq!(len, self.payload_len);
            }
        } else {
            self.stable = true;
        }
    }

    #[inline]
    pub fn payload_is_stable(&self) -> bool {
        self.stable
    }

    #[inline(always)]
    fn refresh_from_slot(&mut self) {
        let (ptr, len) = self.payload_from_slot();
        self.payload_ptr = ptr;
        self.payload_len = len;
    }

    /// Payload `(bytes, len)`. Reloads the slot's address word unless
    /// `stable` holds `Reader.code`. A null payload is the empty numbering.
    #[inline(always)]
    fn payload(&self) -> (*const u8, usize) {
        if self.stable {
            return (self.payload_ptr, self.payload_len);
        }
        if self.slot != 0 {
            return self.payload_from_slot();
        }
        Self::unbound_payload(self.code)
    }

    /// `resumecode.py` `numb_next_item`: `numb.code` is the GC numbering.
    #[inline(always)]
    fn payload_from_slot(&self) -> (*const u8, usize) {
        let addr = unsafe {
            (*(self.slot as *const NumSlot))
                .addr
                .load(Ordering::Relaxed)
        };
        payload_parts(addr)
    }

    #[cold]
    #[inline(never)]
    fn unbound_payload(code: &[u8]) -> (*const u8, usize) {
        (code.as_ptr(), code.len())
    }

    #[inline(always)]
    fn take_item(&mut self, ptr: *const u8, len: usize) -> i32 {
        let index = self.cur_pos;
        // SAFETY: `payload` is a complete `NUMBERING.code`
        // (`resumecode.py` `Writer.create_numbering`; `numb_next_item`).
        let (result, new_pos) = unsafe { decode_varint_at(ptr, len, index) };
        self.cur_pos = new_pos;
        self.items_read += 1;
        result
    }

    /// resumecode.py: next_item / `numb_next_item` (always inline).
    #[inline(always)]
    pub fn next_item(&mut self) -> i32 {
        let (ptr, len) = self.payload();
        self.take_item(ptr, len)
    }

    /// `numb_next_item` after `bind_numbering` / `from_numbering`.
    /// The unbound slice arm stays on `next_item` for tests.
    #[inline(always)]
    pub fn next_item_bound(&mut self) -> i32 {
        #[cfg(debug_assertions)]
        if self.slot == 0 {
            return self.next_item();
        }
        let (ptr, len) = if self.stable {
            (self.payload_ptr, self.payload_len)
        } else {
            self.payload_from_slot()
        };
        self.take_item(ptr, len)
    }

    /// resumecode.py: peek
    #[inline]
    pub fn peek(&self) -> i32 {
        let (ptr, len) = self.payload();
        // SAFETY: `payload` is a complete `NUMBERING.code` (`numb_next_item`).
        let (result, _) = unsafe { decode_varint_at(ptr, len, self.cur_pos) };
        result
    }

    /// resumecode.py: jump — skip n items forward
    pub fn jump(&mut self, size: usize) {
        for _ in 0..size {
            let index = self.cur_pos;
            let (ptr, len) = self.payload();
            // SAFETY: `payload` is a complete `NUMBERING.code` (`numb_next_item`).
            let (_, new_pos) = unsafe { decode_varint_at(ptr, len, index) };
            self.cur_pos = new_pos;
        }
        self.items_read += size;
    }

    #[inline]
    pub fn has_more(&self) -> bool {
        let (_, len) = self.payload();
        self.cur_pos < len
    }
}

/// `NUMBERING` payload: one length word, then `code` bytes.
/// Installed by `register_trace_ops_gc_type` once a collector exists.
/// Absent in GC-less unit tests, which use the host block below.
static NUMBERING_ALLOC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `fn(&[u8]) -> usize` payload address of a fresh `NUMBERING`.
pub fn set_numbering_alloc(hook: Option<fn(&[u8]) -> usize>) {
    let bits = hook.map(|f| f as usize).unwrap_or(0);
    NUMBERING_ALLOC.store(bits, std::sync::atomic::Ordering::Release);
}

fn call_numbering_alloc(bytes: &[u8]) -> Option<usize> {
    let bits = NUMBERING_ALLOC.load(std::sync::atomic::Ordering::Acquire);
    if bits == 0 {
        return None;
    }
    let hook: fn(&[u8]) -> usize = unsafe { std::mem::transmute(bits) };
    Some(hook(bytes))
}

use std::sync::atomic::{AtomicUsize, Ordering};

/// One address cell. Clones share it so a minor rewrites every holder
/// when any of them is walked. Two cells with the same young address
/// make the second root trace an already-forwarded object.
struct NumSlot {
    /// Payload address (length word, then bytes). Relaxed: a collection
    /// is exclusive, and cross-thread readers must not race an `UnsafeCell`.
    addr: AtomicUsize,
    host: bool,
    /// `NumberingRef` clones plus an in-flight list walk.
    users: AtomicUsize,
    /// Index in [`LIVE_NUMBERINGS`], or `usize::MAX` when absent.
    live_index: AtomicUsize,
    /// Index in [`YOUNG_NUMBERINGS`], or `usize::MAX` when absent.
    young_index: AtomicUsize,
}

/// `NonNull` is `!Send`. The slot is shared across threads: the address
/// cell is atomic, and list edits plus the last free run under
/// [`LIVE_NUMBERINGS`] then [`YOUNG_NUMBERINGS`].
struct SlotList {
    slots: Vec<std::ptr::NonNull<NumSlot>>,
}

unsafe impl Send for SlotList {}

/// Young `NUMBERING` slots. `incminimark.py` `collect_oldrefs_to_nursery`
/// drains `old_objects_pointing_to_young` once per minor; this list is that
/// remembered set for `compile.py` `ResumeGuardDescr.rd_numb`. A minor
/// forwards each cell and clears the list. The payload is then old.
static YOUNG_NUMBERINGS: parking_lot::Mutex<SlotList> =
    parking_lot::Mutex::new(SlotList { slots: Vec::new() });

/// Live `NUMBERING` slots until the last [`NumberingRef`] drops.
///
/// `collect_oldrefs_to_nursery` clears the young list at the minor, and the
/// descr may not be in the holder graph yet (`ResumeGuardDescr.rd_numb` is
/// stored before the loop is published). A major in that window still has
/// to mark the payload. Holder walks mark the same object again once the
/// descr is reachable; `GCFLAG_VISITED` makes the second mark a no-op.
static LIVE_NUMBERINGS: parking_lot::Mutex<SlotList> =
    parking_lot::Mutex::new(SlotList { slots: Vec::new() });

/// One `NUMBERING` (`GcStruct` with inline `Array(UCHAR)`).
///
/// `addr` is the payload (length word, then bytes). A minor rewrites
/// that word when the young list hands the cell to the collector.
/// GC-less tests use a process block with the same layout; that arm
/// is not taken once `set_numbering_alloc` is installed.
pub struct NumberingRef {
    slot: std::ptr::NonNull<NumSlot>,
}

unsafe impl Send for NumberingRef {}
unsafe impl Sync for NumberingRef {}

fn numb_len_word() -> usize {
    std::mem::size_of::<usize>()
}

/// Length word, then bytes. `addr == 0` is an empty numbering.
#[inline(always)]
fn payload_parts(addr: usize) -> (*const u8, usize) {
    if addr != 0 {
        let len = unsafe { *(addr as *const usize) };
        let ptr = unsafe { (addr as *const u8).add(numb_len_word()) };
        return (ptr, len);
    }
    empty_payload()
}

#[cold]
#[inline(never)]
fn empty_payload() -> (*const u8, usize) {
    (std::ptr::NonNull::<u8>::dangling().as_ptr(), 0)
}

#[inline(always)]
fn slice_from_parts<'a>((ptr, len): (*const u8, usize)) -> &'a [u8] {
    if len == 0 {
        return &[];
    }
    // The caller names `'a` from the numbering or the `code` borrow that
    // owns this payload. A zero length is the empty numbering, not a
    // dangling pointer.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn fresh_slot(addr: usize, host: bool) -> *mut NumSlot {
    Box::into_raw(Box::new(NumSlot {
        addr: AtomicUsize::new(addr),
        host,
        users: AtomicUsize::new(1),
        live_index: AtomicUsize::new(usize::MAX),
        young_index: AtomicUsize::new(usize::MAX),
    }))
}

/// Temporary P92 NUMBERING diagnostic. `PYRE_DIAG_P92` records payload
/// address and first bytes at creation; a later read panics if the cell
/// was not forwarded or the bytes were overwritten.
static DIAG_NUMB_ON: AtomicUsize = AtomicUsize::new(0);
static DIAG_NUMB: parking_lot::Mutex<Vec<NumbDiag>> = parking_lot::Mutex::new(Vec::new());

struct NumbDiag {
    slot: usize,
    payload: usize,
    len: usize,
    first: [u8; 16],
    nfirst: usize,
}

fn diag_numb_on() -> bool {
    let v = DIAG_NUMB_ON.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
    let on = std::env::var_os("PYRE_DIAG_P92").is_some();
    DIAG_NUMB_ON.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

fn diag_numb_first_bytes(addr: usize) -> (usize, [u8; 16], usize) {
    if addr == 0 {
        return (0, [0; 16], 0);
    }
    let len = unsafe { *(addr as *const usize) };
    let n = len.min(16);
    let mut first = [0u8; 16];
    if n > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                (addr as *const u8).add(numb_len_word()),
                first.as_mut_ptr(),
                n,
            );
        }
    }
    (len, first, n)
}

fn diag_numb_record(slot: *mut NumSlot, addr: usize) {
    if !diag_numb_on() || addr == 0 {
        return;
    }
    if unsafe { (*slot).host } {
        return;
    }
    let (len, first, nfirst) = diag_numb_first_bytes(addr);
    DIAG_NUMB.lock().push(NumbDiag {
        slot: slot as usize,
        payload: addr,
        len,
        first,
        nfirst,
    });
}

fn diag_numb_forwarded(slot: *mut NumSlot, old: usize, new: usize) {
    if !diag_numb_on() || old == new {
        return;
    }
    let mut guard = DIAG_NUMB.lock();
    for row in guard.iter_mut() {
        if row.slot == slot as usize {
            row.payload = new;
            return;
        }
    }
}

fn diag_numb_check(slot: *mut NumSlot, addr: usize, site: &'static str) {
    if !diag_numb_on() {
        return;
    }
    if unsafe { (*slot).host } || addr == 0 {
        return;
    }
    let (len, first, nfirst) = diag_numb_first_bytes(addr);
    let guard = DIAG_NUMB.lock();
    let Some(row) = guard.iter().rev().find(|r| r.slot == slot as usize) else {
        return;
    };
    let payload_moved = row.payload != addr;
    let bytes_changed = nfirst != row.nfirst || first[..nfirst] != row.first[..row.nfirst];
    if !payload_moved && !bytes_changed {
        return;
    }
    let backtrace = std::backtrace::Backtrace::force_capture();
    panic!(
        "STALE_NUMBERING site={site} slot={:#x} create_payload={:#x} read_payload={:#x} \
         payload_moved={payload_moved} bytes_changed={bytes_changed} create_len={} read_len={len} \
         create_first={:02x?} read_first={:02x?}\n{backtrace}",
        slot as usize,
        row.payload,
        addr,
        row.len,
        &row.first[..row.nfirst],
        &first[..nfirst]
    );
}

/// Lock order is [`LIVE_NUMBERINGS`] then [`YOUNG_NUMBERINGS`].
fn register_gc_slot(slot: *mut NumSlot) {
    let nn = std::ptr::NonNull::new(slot).expect("numb slot");
    let mut live = LIVE_NUMBERINGS.lock();
    let mut young = YOUNG_NUMBERINGS.lock();
    let live_index = live.slots.len();
    unsafe { (*slot).live_index.store(live_index, Ordering::Relaxed) };
    live.slots.push(nn);
    let young_index = young.slots.len();
    unsafe { (*slot).young_index.store(young_index, Ordering::Relaxed) };
    young.slots.push(nn);
}

impl NumberingRef {
    fn from_gc_addr(addr: usize) -> Self {
        // `Box::new` follows the allocating hook. It is the system
        // allocator, not a collection point. The address store and the
        // two list pushes then run with no safepoint between them.
        let slot = fresh_slot(0, false);
        unsafe { (*slot).addr.store(addr, Ordering::Relaxed) };
        register_gc_slot(slot);
        diag_numb_record(slot, addr);
        Self {
            slot: std::ptr::NonNull::new(slot).expect("numb slot"),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        if let Some(addr) = call_numbering_alloc(bytes) {
            return Self::from_gc_addr(addr);
        }
        Self::host_from_bytes(bytes)
    }

    fn host_from_bytes(bytes: &[u8]) -> Self {
        let layout = host_layout(bytes.len());
        let block = unsafe { std::alloc::alloc_zeroed(layout) };
        if block.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe {
            let body = block.add(numb_len_word());
            *(body as *mut usize) = bytes.len();
            if !bytes.is_empty() {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    body.add(numb_len_word()),
                    bytes.len(),
                );
            }
            let slot = fresh_slot(body as usize, true);
            Self {
                slot: std::ptr::NonNull::new(slot).expect("numb slot"),
            }
        }
    }

    /// Address of the `NUMBERING` payload. One relaxed load.
    /// `incminimark.py` `collect_oldrefs_to_nursery` stores the
    /// forwarded address back into this cell.
    #[inline(always)]
    pub fn payload_addr(&self) -> usize {
        unsafe { self.slot.as_ref().addr.load(Ordering::Relaxed) }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        let addr = self.payload_addr();
        if diag_numb_on() {
            diag_numb_check(self.slot.as_ptr(), addr, "NumberingRef::as_slice");
        }
        slice_from_parts(payload_parts(addr))
    }

    /// Hand the payload address to a holder walk. Host blocks are not `GcRef`s.
    ///
    /// The walked cell is this slot. A minor between `create_numbering`
    /// and the descr joining the holder graph rewrites it from
    /// [`collect_young_numberings`].
    pub fn visit_gc(&self, visitor: &mut dyn FnMut(&mut crate::GcRef)) {
        forward_slot(self.slot, visitor);
    }

    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.payload_addr() == b.payload_addr() && a.payload_addr() != 0
    }

    /// `true` while this cell is on `YOUNG_NUMBERINGS`.
    /// After `collect_young_numberings` (`collect_oldrefs_to_nursery`)
    /// the payload is old and the old generation does not move it.
    #[inline]
    pub fn is_young(&self) -> bool {
        unsafe { self.slot.as_ref().young_index.load(Ordering::Relaxed) != usize::MAX }
    }
}

fn forward_slot(slot: std::ptr::NonNull<NumSlot>, visitor: &mut dyn FnMut(&mut crate::GcRef)) {
    let addr = unsafe { slot.as_ref().addr.load(Ordering::Relaxed) };
    if unsafe { slot.as_ref().host } || addr == 0 {
        return;
    }
    let mut gc = crate::GcRef(addr);
    visitor(&mut gc);
    unsafe { slot.as_ref().addr.store(gc.0, Ordering::Relaxed) };
    if addr != gc.0 {
        diag_numb_forwarded(slot.as_ptr(), addr, gc.0);
    }
}

/// Minor remembered-set drain. Forwards each young cell and clears the list.
/// The payload is old afterwards and the old generation does not move it.
pub fn collect_young_numberings(visitor: &mut dyn FnMut(&mut crate::GcRef)) {
    let taken = {
        let mut young = YOUNG_NUMBERINGS.lock();
        if young.slots.is_empty() {
            return;
        }
        let taken = std::mem::take(&mut young.slots);
        for &slot in &taken {
            unsafe {
                slot.as_ref().users.fetch_add(1, Ordering::Relaxed);
                slot.as_ref()
                    .young_index
                    .store(usize::MAX, Ordering::Relaxed);
            }
        }
        taken
    };
    visit_bumped(&taken, visitor);
}

/// Major root for every live `NUMBERING`, including one whose descr is
/// not yet reachable from a holder. Does not clear [`YOUNG_NUMBERINGS`].
pub fn trace_live_numberings(visitor: &mut dyn FnMut(&mut crate::GcRef)) {
    let taken = {
        let live = LIVE_NUMBERINGS.lock();
        if live.slots.is_empty() {
            return;
        }
        let taken = live.slots.clone();
        for &slot in &taken {
            unsafe { slot.as_ref().users.fetch_add(1, Ordering::Relaxed) };
        }
        taken
    };
    visit_bumped(&taken, visitor);
}

fn visit_bumped(slots: &[std::ptr::NonNull<NumSlot>], visitor: &mut dyn FnMut(&mut crate::GcRef)) {
    let _release = ReleaseAll(slots);
    for &slot in slots {
        forward_slot(slot, visitor);
    }
}

struct ReleaseAll<'a>(&'a [std::ptr::NonNull<NumSlot>]);

impl Drop for ReleaseAll<'_> {
    fn drop(&mut self) {
        for &slot in self.0 {
            release_user(slot);
        }
    }
}

fn host_layout(len: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(numb_len_word() * 2 + len, numb_len_word())
        .unwrap_or_else(|_| std::alloc::Layout::new::<usize>())
}

impl Clone for NumberingRef {
    fn clone(&self) -> Self {
        unsafe {
            self.slot.as_ref().users.fetch_add(1, Ordering::Relaxed);
        }
        Self { slot: self.slot }
    }
}

impl Drop for NumberingRef {
    fn drop(&mut self) {
        release_user(self.slot);
    }
}

fn release_user(slot: std::ptr::NonNull<NumSlot>) {
    let prev = unsafe { slot.as_ref().users.fetch_sub(1, Ordering::Release) };
    if prev != 1 {
        return;
    }
    std::sync::atomic::fence(Ordering::Acquire);
    if unsafe { slot.as_ref().host } {
        free_slot(slot);
        return;
    }
    {
        // A walker may have bumped `users` while this drop waited.
        let mut live = LIVE_NUMBERINGS.lock();
        let mut young = YOUNG_NUMBERINGS.lock();
        if unsafe { slot.as_ref().users.load(Ordering::Acquire) } != 0 {
            return;
        }
        detach(&mut live.slots, slot, true);
        detach(&mut young.slots, slot, false);
    }
    free_slot(slot);
}

fn detach(
    list: &mut Vec<std::ptr::NonNull<NumSlot>>,
    slot: std::ptr::NonNull<NumSlot>,
    live: bool,
) {
    let index_ptr = unsafe {
        if live {
            &(*slot.as_ptr()).live_index
        } else {
            &(*slot.as_ptr()).young_index
        }
    };
    let index = index_ptr.load(Ordering::Relaxed);
    let pos = if index < list.len() && list[index] == slot {
        Some(index)
    } else {
        list.iter().position(|entry| *entry == slot)
    };
    index_ptr.store(usize::MAX, Ordering::Relaxed);
    let Some(pos) = pos else {
        return;
    };
    let last = list.len() - 1;
    list.swap_remove(pos);
    if pos != last {
        let moved = list[pos];
        let moved_index = unsafe {
            if live {
                &(*moved.as_ptr()).live_index
            } else {
                &(*moved.as_ptr()).young_index
            }
        };
        moved_index.store(pos, Ordering::Relaxed);
    }
}

fn free_slot(slot: std::ptr::NonNull<NumSlot>) {
    unsafe {
        if slot.as_ref().host {
            let addr = slot.as_ref().addr.load(Ordering::Relaxed);
            if addr != 0 {
                let len = *(addr as *const usize);
                let block = (addr as *mut u8).sub(numb_len_word());
                std::alloc::dealloc(block, host_layout(len));
            }
        }
        drop(Box::from_raw(slot.as_ptr()));
    }
}

impl std::fmt::Debug for NumberingRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumberingRef")
            .field("len", &self.as_slice().len())
            .field("host", &unsafe { (*self.slot.as_ptr()).host })
            .finish()
    }
}

impl PartialEq for NumberingRef {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for NumberingRef {}

impl AsRef<[u8]> for NumberingRef {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::Deref for NumberingRef {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(case: &str, values: &[i32]) {
        let mut buf = Vec::new();
        for &v in values {
            encode_varint(&mut buf, v);
        }
        let mut index = 0;
        for &expected in values {
            let (got, next) = decode_varint(&buf, index);
            assert_eq!(got, expected, "case {case} decode {expected}");
            index = next;
        }
        assert_eq!(index, buf.len(), "case {case}");
    }

    #[test]
    fn decode_varint_roundtrips_item_widths() {
        // one byte: zigzag `item < 2**7` (0, ±1, ±63). Then two- and three-byte items.
        let cases: &[(&str, &[i32])] = &[
            ("one_byte", &[0, 1, -1, 63, -63]),
            (
                "two_and_three_byte",
                &[
                    64, -64, 127, -128, 8191, -8192, 16383, -16384, 32768, 40479, 65535,
                ],
            ),
        ];
        for (name, values) in cases {
            roundtrip(name, values);
        }
    }

    #[test]
    fn numbering_ref_shares_identity() {
        let mut w = Writer::new(8);
        for i in 0..20 {
            w.append_int(i);
        }
        let a = w.create_numbering();
        let b = a.clone();
        assert!(NumberingRef::ptr_eq(&a, &b));
        assert_eq!(a.as_slice(), b.as_slice());
        assert_eq!(a.as_slice(), w.encode_bytes().as_slice());
    }

    #[test]
    fn numbering_ref_large_payload_round_trips() {
        let bytes = vec![0x5a; 200];
        let a = NumberingRef::from_bytes(&bytes);
        assert_eq!(a.as_slice(), bytes.as_slice());
        let b = a.clone();
        assert!(NumberingRef::ptr_eq(&a, &b));
    }

    #[test]
    fn reader_reads_each_item_from_the_numbering() {
        // `resumecode.py` `numb_next_item` loads `numb.code[index]` per item.
        let numb = create_numbering(&[1, -2, 3, 64]);
        let mut reader = Reader::from_numbering(&numb);
        assert_eq!(reader.next_item(), 1);
        assert_eq!(reader.peek(), -2);
        assert_eq!(reader.next_item(), -2);
        assert_eq!(reader.next_item(), 3);
        reader.jump(1);
        assert!(!reader.has_more());
        assert_eq!(reader.items_read, 4);
    }

    #[test]
    fn reader_stable_payload_matches_per_item_reload() {
        // `Reader.code` pinned: `numb_next_item` indexes the same bytes
        // `next_item` reloads from `NumSlot.addr`.
        let numb = create_numbering(&[1, -2, 3, 64, -128, 255]);
        let mut moving = Reader::from_numbering(&numb);
        let mut stable = Reader::from_numbering(&numb);
        stable.mark_payload_stable();
        while moving.has_more() {
            assert_eq!(stable.next_item_bound(), moving.next_item());
        }
        assert_eq!(stable.items_read, moving.items_read);
        assert!(!stable.has_more());
    }
}
