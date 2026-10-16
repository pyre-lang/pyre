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
    let (bytes, n) = encode_varint_bytes(value);
    buf.extend_from_slice(&bytes[..n]);
}

fn encode_varint_bytes(value: i32) -> ([u8; 3], usize) {
    let mut item = (value as i64) * 2;
    if item < 0 {
        item = -1 - item;
    }
    assert!(item >= 0);
    let item = item as u32;

    if item < (1 << 7) {
        ([item as u8, 0, 0], 1)
    } else if item < (1 << 14) {
        ([(item | 0x80) as u8, (item >> 7) as u8, 0], 2)
    } else {
        // `resumecode.py` `append_numbering` asserts `item < 2**16` because
        // `append_int` feeds it a SHORT. A jitcode pc is a `fix_labels`
        // target (`0 <= target <= 0xFFFF`); its zigzag is `target * 2` and
        // still fits this 3-byte form (`item >> 14` in one byte, so
        // `item < 2**22`).
        assert!(item < (1 << 22), "resumecode item too large: {item}");
        (
            [
                (item | 0x80) as u8,
                ((item >> 7) | 0x80) as u8,
                (item >> 14) as u8,
            ],
            3,
        )
    }
}

/// resumecode.py: append_numbering(lst, item)
pub fn append_numbering(buf: &mut Vec<u8>, item: i32) {
    encode_varint(buf, item);
}

/// resumecode.py: numb_next_item(numb, index)
///
/// line-by-line port. Does not bounds-check: upstream contract requires
/// the buffer to contain a complete varint at `index`. A truncated
/// buffer is a bug in resume data generation and should panic loudly
/// via the standard slice indexing rather than silently returning 0.
#[inline]
pub fn decode_varint(buf: &[u8], index: usize) -> (i32, usize) {
    let b0 = buf[index] as i64;
    // resumecode.py `numb_next_item`: one-byte items are `item < 2**7`
    // after zigzag. Same decode as the multi-byte path, without the
    // continuation loads.
    if b0 & (1 << 7) == 0 {
        let value = if b0 & 1 != 0 { -1 - b0 } else { b0 };
        return ((value >> 1) as i32, index + 1);
    }

    let mut value = b0;
    let mut index = index + 1;
    value &= (1 << 7) - 1;
    value |= (buf[index] as i64) << 7;
    index += 1;
    if value & (1 << 14) != 0 {
        value &= (1 << 14) - 1;
        value |= (buf[index] as i64) << 14;
        index += 1;
    }

    if value & 1 != 0 {
        value = -1 - value;
    }
    value >>= 1;

    (value as i32, index)
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
/// Holds `numb` and an index. Each item re-reads `numb.code[index]`
/// (`numb_next_item`): one plain load of the payload cell. The byte
/// pointer is not cached across an item. The consumer of an item can
/// allocate, and a minor forwards `compile.py` `ResumeGuardDescr.rd_numb`.
pub struct Reader<'a> {
    code: &'a [u8],
    /// When set, reads come from this numbering's payload.
    numb: Option<&'a NumberingRef>,
    pub cur_pos: usize,
    pub items_read: usize,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Reader {
            code,
            numb: None,
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// Subsequent reads follow `numb` across a minor.
    ///
    /// `resumecode.py` `Reader.next_item` calls `numb_next_item`, which
    /// loads `numb.code[index]` from the GC pointer on every item.
    pub fn bind_numbering(&mut self, numb: &'a NumberingRef) {
        self.numb = Some(numb);
    }

    pub fn from_numbering(numb: &'a NumberingRef) -> Self {
        Reader {
            code: &[],
            numb: Some(numb),
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// One `numb_next_item` view. The slice dies with the call that
    /// decodes it; the next item loads the cell again.
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        if let Some(numb) = self.numb {
            numb.as_slice()
        } else {
            self.code
        }
    }

    /// resumecode.py: next_item / `numb_next_item` (always inline).
    #[inline(always)]
    pub fn next_item(&mut self) -> i32 {
        let pos = self.cur_pos;
        let (result, new_pos) = {
            let buf = self.bytes();
            decode_varint(buf, pos)
        };
        self.cur_pos = new_pos;
        self.items_read += 1;
        result
    }

    /// resumecode.py: peek
    #[inline]
    pub fn peek(&self) -> i32 {
        let (result, _) = decode_varint(self.bytes(), self.cur_pos);
        result
    }

    /// resumecode.py: jump — skip n items forward
    pub fn jump(&mut self, size: usize) {
        for _ in 0..size {
            let pos = self.cur_pos;
            let (_, new_pos) = decode_varint(self.bytes(), pos);
            self.cur_pos = new_pos;
        }
        self.items_read += size;
    }

    #[inline]
    pub fn has_more(&self) -> bool {
        self.cur_pos < self.bytes().len()
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
        if addr == 0 {
            return &[];
        }
        if diag_numb_on() {
            diag_numb_check(self.slot.as_ptr(), addr, "NumberingRef::as_slice");
        }
        let len = unsafe { *(addr as *const usize) };
        unsafe { std::slice::from_raw_parts((addr as *const u8).add(numb_len_word()), len) }
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
}
