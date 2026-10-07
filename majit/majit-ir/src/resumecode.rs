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
pub struct Reader<'a> {
    code: &'a [u8],
    /// When set, reads come from this numbering's payload.
    numb: Option<&'a NumberingRef>,
    /// Payload bytes of `numb` at `cached_epoch`. Null until the first load.
    cached_ptr: *const u8,
    cached_len: usize,
    /// [`numbering_payload_epoch`] observed when `cached_ptr` was loaded.
    /// `0` is never published, so a fresh reader always reloads.
    cached_epoch: u64,
    pub cur_pos: usize,
    pub items_read: usize,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Reader {
            code,
            numb: None,
            cached_ptr: std::ptr::null(),
            cached_len: 0,
            cached_epoch: 0,
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// Subsequent reads follow `numb` across a minor.
    ///
    /// `resumecode.py` `Reader.next_item` calls `numb_next_item`, which
    /// loads `numb.code[index]` from the GC pointer. The pointer is
    /// stable until a collection. Reload when
    /// [`numbering_payload_epoch`] changes — that is
    /// `bump_minor_epoch` — rather than on every item.
    pub fn bind_numbering(&mut self, numb: &'a NumberingRef) {
        self.numb = Some(numb);
        self.cached_ptr = std::ptr::null();
        self.cached_len = 0;
        self.cached_epoch = 0;
    }

    pub fn from_numbering(numb: &'a NumberingRef) -> Self {
        Reader {
            code: &[],
            numb: Some(numb),
            cached_ptr: std::ptr::null(),
            cached_len: 0,
            cached_epoch: 0,
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// Payload for one `numb_next_item`. A matching epoch means no minor
    /// has run since the pointer was loaded (`bump_minor_epoch` publishes
    /// the new epoch before the mutator resumes).
    #[inline]
    fn ensure_cache(&mut self) {
        if self.numb.is_none() {
            return;
        }
        let epoch = numbering_payload_epoch();
        if epoch == self.cached_epoch {
            return;
        }
        let addr = self.numb.unwrap().payload_addr();
        let (ptr, len) = if addr == 0 {
            (std::ptr::null(), 0)
        } else {
            // Same header `as_slice` reads: length word, then bytes.
            let len = unsafe { *(addr as *const usize) };
            let ptr = unsafe { (addr as *const u8).add(numb_len_word()) };
            (ptr, len)
        };
        self.cached_ptr = ptr;
        self.cached_len = len;
        self.cached_epoch = epoch;
    }

    #[inline]
    fn buf(&mut self) -> &[u8] {
        if self.numb.is_some() {
            self.ensure_cache();
            if self.cached_ptr.is_null() {
                return &[];
            }
            // SAFETY: `ensure_cache` loaded this pointer at the current
            // payload epoch. `next_item` / `jump` do not allocate, so a
            // minor cannot move the payload before this slice is dropped.
            unsafe { std::slice::from_raw_parts(self.cached_ptr, self.cached_len) }
        } else {
            self.code
        }
    }

    fn bytes(&self) -> &[u8] {
        if let Some(numb) = self.numb {
            if self.cached_epoch == numbering_payload_epoch() {
                if self.cached_ptr.is_null() {
                    return &[];
                }
                // SAFETY: same epoch contract as `buf`.
                return unsafe { std::slice::from_raw_parts(self.cached_ptr, self.cached_len) };
            }
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
            let buf = self.buf();
            decode_varint(buf, pos)
        };
        self.cur_pos = new_pos;
        self.items_read += 1;
        result
    }

    /// resumecode.py: peek
    pub fn peek(&self) -> i32 {
        let (result, _) = decode_varint(self.bytes(), self.cur_pos);
        result
    }

    /// resumecode.py: jump — skip n items forward
    pub fn jump(&mut self, size: usize) {
        for _ in 0..size {
            let pos = self.cur_pos;
            let (_, new_pos) = {
                let buf = self.buf();
                decode_varint(buf, pos)
            };
            self.cur_pos = new_pos;
        }
        self.items_read += size;
    }

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

/// Owner-root slot for a nursery `NUMBERING`. `majit-ir` cannot name the
/// guard type; `register_trace_ops_gc_type` installs these.
/// `pin(addr) -> slot`, `read(slot) -> addr`, `write(slot, addr)`, `unpin(slot)`.
static NUMBERING_PIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static NUMBERING_READ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static NUMBERING_WRITE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static NUMBERING_UNPIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// `fn() -> u64`. `register_trace_ops_gc_type` still publishes
/// `minor_epoch` here. Payload caching does not call it: the
/// generation is [`numbering_payload_epoch`], advanced from
/// `bump_minor_epoch` so a resume item is one relaxed load.
#[allow(dead_code)]
static NUMBERING_EPOCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Bumped with `minor_epoch`. `0` is reserved so a `Reader` whose
/// `cached_epoch` is still 0 always reloads. The counter starts at 1
/// and skips 0 if it wraps.
static NUMBERING_PAYLOAD_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn set_numbering_root_hooks(
    pin: fn(usize) -> usize,
    read: fn(usize) -> usize,
    write: fn(usize, usize),
    unpin: fn(usize),
) {
    NUMBERING_PIN.store(pin as usize, std::sync::atomic::Ordering::Release);
    NUMBERING_READ.store(read as usize, std::sync::atomic::Ordering::Release);
    NUMBERING_WRITE.store(write as usize, std::sync::atomic::Ordering::Release);
    NUMBERING_UNPIN.store(unpin as usize, std::sync::atomic::Ordering::Release);
}

pub fn set_numbering_epoch(hook: fn() -> u64) {
    NUMBERING_EPOCH.store(hook as usize, std::sync::atomic::Ordering::Release);
}

/// Current numbering-payload generation. Matches `minor_epoch` in
/// lockstep: `bump_minor_epoch` calls [`bump_numbering_payload_epoch`].
#[inline]
pub fn numbering_payload_epoch() -> u64 {
    NUMBERING_PAYLOAD_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Invalidate cached numbering payload pointers. One minor, one bump.
pub fn bump_numbering_payload_epoch() {
    let prev = NUMBERING_PAYLOAD_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if prev == u64::MAX - 1 {
        // Wrapped to 0, which `Reader` treats as "not loaded".
        NUMBERING_PAYLOAD_EPOCH.store(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `true` after `register_trace_ops_gc_type` installed the owner-root
/// pin. Each nursery `NUMBERING` then lives in that slot for its whole
/// `NumberingRef` lifetime, and `walk_roots` is the minor edge.
#[inline]
pub fn numbering_owner_rooted() -> bool {
    NUMBERING_PIN.load(std::sync::atomic::Ordering::Acquire) != 0
}

fn call_numbering_pin(addr: usize) -> Option<usize> {
    let bits = NUMBERING_PIN.load(std::sync::atomic::Ordering::Acquire);
    if bits == 0 {
        return None;
    }
    let hook: fn(usize) -> usize = unsafe { std::mem::transmute(bits) };
    Some(hook(addr))
}

fn call_numbering_read(slot: usize) -> usize {
    let bits = NUMBERING_READ.load(std::sync::atomic::Ordering::Acquire);
    let hook: fn(usize) -> usize = unsafe { std::mem::transmute(bits) };
    hook(slot)
}

fn call_numbering_write(slot: usize, addr: usize) {
    let bits = NUMBERING_WRITE.load(std::sync::atomic::Ordering::Acquire);
    let hook: fn(usize, usize) = unsafe { std::mem::transmute(bits) };
    hook(slot, addr);
}

fn call_numbering_unpin(slot: usize) {
    let bits = NUMBERING_UNPIN.load(std::sync::atomic::Ordering::Acquire);
    if bits == 0 {
        return;
    }
    let hook: fn(usize) = unsafe { std::mem::transmute(bits) };
    hook(slot);
}

fn call_numbering_alloc(bytes: &[u8]) -> Option<usize> {
    let bits = NUMBERING_ALLOC.load(std::sync::atomic::Ordering::Acquire);
    if bits == 0 {
        return None;
    }
    let hook: fn(&[u8]) -> usize = unsafe { std::mem::transmute(bits) };
    Some(hook(bytes))
}

/// One address cell. Clones share it so a minor rewrites every holder
/// when any of them is walked. Two cells with the same young address
/// make the second root trace an already-forwarded object.
struct NumSlot {
    /// Host block, or the mirror of `root` when no pin hook is installed.
    addr: std::cell::UnsafeCell<usize>,
    /// `minor_epoch` observed when `addr` was last filled from `root`.
    /// `u64::MAX` forces the first read through the owner-root slot:
    /// `Box::new` below can collect after `pin` and move the array.
    seen_epoch: std::sync::atomic::AtomicU64,
    /// `acquire_owner_root` index. `usize::MAX` for a host block.
    root: usize,
    host: bool,
    refs: std::sync::atomic::AtomicUsize,
}

/// One `NUMBERING` (`GcStruct` with inline `Array(UCHAR)`).
///
/// `addr` is the payload (length word, then bytes). A minor rewrites
/// that word when `visit_gc` hands the cell to the collector.
/// GC-less tests use a process block with the same layout and a
/// refcount in the word before the payload; that arm is not taken
/// once `set_numbering_alloc` is installed.
pub struct NumberingRef {
    slot: std::ptr::NonNull<NumSlot>,
}

unsafe impl Send for NumberingRef {}
unsafe impl Sync for NumberingRef {}

fn numb_len_word() -> usize {
    std::mem::size_of::<usize>()
}

impl NumberingRef {
    fn from_slot(addr: usize, host: bool) -> Self {
        // Pin before the `NumSlot` exists so a minor inside `Box::new`
        // cannot collect the array. The owner-root slot is the walked cell.
        let root = if host {
            usize::MAX
        } else {
            call_numbering_pin(addr).unwrap_or(usize::MAX)
        };
        let slot = Box::into_raw(Box::new(NumSlot {
            addr: std::cell::UnsafeCell::new(addr),
            seen_epoch: std::sync::atomic::AtomicU64::new(u64::MAX),
            root,
            host,
            refs: std::sync::atomic::AtomicUsize::new(1),
        }));
        Self {
            slot: std::ptr::NonNull::new(slot).expect("numb slot"),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        if let Some(addr) = call_numbering_alloc(bytes) {
            return Self::from_slot(addr, false);
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
            Self::from_slot(body as usize, true)
        }
    }

    pub fn payload_addr(&self) -> usize {
        let slot = unsafe { self.slot.as_ref() };
        if slot.root != usize::MAX {
            // The owner-root cell changes only in a minor. `Reader`
            // keeps the byte pointer across items; this cache is the
            // reload those readers share. `numbering_payload_epoch`
            // advances in `bump_minor_epoch`, so a match means the
            // address last published is still the object's address.
            let epoch = numbering_payload_epoch();
            if slot.seen_epoch.load(std::sync::atomic::Ordering::Relaxed) == epoch {
                return unsafe { *slot.addr.get() };
            }
            let addr = call_numbering_read(slot.root);
            unsafe { *slot.addr.get() = addr };
            slot.seen_epoch
                .store(epoch, std::sync::atomic::Ordering::Relaxed);
            return addr;
        }
        unsafe { *slot.addr.get() }
    }

    pub fn as_slice(&self) -> &[u8] {
        let addr = self.payload_addr();
        if addr == 0 {
            return &[];
        }
        let len = unsafe { *(addr as *const usize) };
        unsafe { std::slice::from_raw_parts((addr as *const u8).add(numb_len_word()), len) }
    }

    /// Hand the payload address to a minor walk. Host blocks are not `GcRef`s.
    ///
    /// The walked cell is the owner-root slot acquired at malloc
    /// (`acquire_owner_root`), not a copy. A minor between `create_numbering`
    /// and the descr joining the `rd_consts` area rewrites that slot.
    pub fn visit_gc(&self, visitor: &mut dyn FnMut(&mut crate::GcRef)) {
        let slot = unsafe { self.slot.as_ref() };
        if slot.host {
            return;
        }
        if slot.root != usize::MAX {
            let mut gc = crate::GcRef(call_numbering_read(slot.root));
            if gc.0 == 0 {
                return;
            }
            visitor(&mut gc);
            call_numbering_write(slot.root, gc.0);
            return;
        }
        let gc = unsafe { &mut *(slot.addr.get() as *mut crate::GcRef) };
        if gc.0 == 0 {
            return;
        }
        visitor(gc);
    }

    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.payload_addr() == b.payload_addr() && a.payload_addr() != 0
    }
}

fn host_layout(len: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(numb_len_word() * 2 + len, numb_len_word())
        .unwrap_or_else(|_| std::alloc::Layout::new::<usize>())
}

impl Clone for NumberingRef {
    fn clone(&self) -> Self {
        unsafe {
            (*self.slot.as_ptr())
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Self { slot: self.slot }
    }
}

impl Drop for NumberingRef {
    fn drop(&mut self) {
        let slot = self.slot.as_ptr();
        let prev = unsafe {
            (*slot)
                .refs
                .fetch_sub(1, std::sync::atomic::Ordering::Release)
        };
        if prev != 1 {
            return;
        }
        unsafe {
            let root = (*slot).root;
            if root != usize::MAX {
                call_numbering_unpin(root);
            }
            if (*slot).host {
                let addr = *(*slot).addr.get();
                if addr != 0 {
                    let len = *(addr as *const usize);
                    let block = (addr as *mut u8).sub(numb_len_word());
                    std::alloc::dealloc(block, host_layout(len));
                }
            }
            drop(Box::from_raw(slot));
        }
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
    fn reader_follows_numbering_across_a_payload_epoch() {
        // Host block: a minor does not move it. The reader must still
        // drop its cached pointer when the epoch bumps and read the
        // same bytes afterwards. `resumecode.py` `numb_next_item`.
        let numb = create_numbering(&[1, -2, 3, 64]);
        let mut reader = Reader::from_numbering(&numb);
        assert_eq!(reader.next_item(), 1);
        bump_numbering_payload_epoch();
        assert_eq!(reader.peek(), -2);
        assert_eq!(reader.next_item(), -2);
        assert_eq!(reader.next_item(), 3);
        reader.jump(1);
        assert!(!reader.has_more());
        assert_eq!(reader.items_read, 4);
    }
}
