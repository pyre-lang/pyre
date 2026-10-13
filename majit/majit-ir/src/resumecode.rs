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
pub fn create_numbering(items: &[i32]) -> Vec<u8> {
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

    /// resumecode.py: create_numbering
    pub fn create_numbering(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.current.len() * 3);
        for &item in &self.current {
            encode_varint(&mut buf, item);
        }
        buf
    }

    /// Encode onto the stack when the numbering fits, then one slab cell.
    /// `lltype.malloc(NUMBERING)` is one GC object; a per-guard `Arc<[u8]>`
    /// was a 96 B class. Cells come from a chunked slab so identity is
    /// still a pointer.
    pub fn create_numbering_arc(&self) -> NumberingRef {
        let mut buf = smallvec::SmallVec::<[u8; 128]>::new();
        for &item in &self.current {
            let (bytes, n) = encode_varint_bytes(item);
            buf.extend_from_slice(&bytes[..n]);
        }
        NumberingRef::from_bytes(&buf)
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
    pub cur_pos: usize,
    pub items_read: usize,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Reader {
            code,
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// resumecode.py: next_item
    #[inline]
    pub fn next_item(&mut self) -> i32 {
        let (result, new_pos) = decode_varint(self.code, self.cur_pos);
        self.cur_pos = new_pos;
        self.items_read += 1;
        result
    }

    /// resumecode.py: peek
    pub fn peek(&self) -> i32 {
        let (result, _) = decode_varint(self.code, self.cur_pos);
        result
    }

    /// resumecode.py: jump — skip n items forward
    pub fn jump(&mut self, size: usize) {
        for _ in 0..size {
            let (_, new_pos) = decode_varint(self.code, self.cur_pos);
            self.cur_pos = new_pos;
        }
        self.items_read += size;
    }

    pub fn has_more(&self) -> bool {
        self.cur_pos < self.code.len()
    }
}

/// One `NUMBERING` object. Small payloads live in a chunked slab so
/// `create_numbering` does not mint a 96 B `Arc<[u8]>` per guard.
const NUMB_DATA: usize = 80;
const NUMB_CHUNK: usize = 256;

struct NumbCell {
    refs: std::sync::atomic::AtomicUsize,
    len: u16,
    bytes: [u8; NUMB_DATA],
}

struct NumbHeap {
    chunks: Vec<(*mut NumbCell, usize)>,
    free: Vec<*mut NumbCell>,
}

unsafe impl Send for NumbHeap {}

impl NumbHeap {
    const fn new() -> Self {
        Self {
            chunks: Vec::new(),
            free: Vec::new(),
        }
    }

    fn alloc(&mut self, bytes: &[u8]) -> *mut NumbCell {
        debug_assert!(bytes.len() <= NUMB_DATA);
        let cell = if let Some(cell) = self.free.pop() {
            cell
        } else {
            self.fresh_cell()
        };
        unsafe {
            (*cell).refs.store(1, std::sync::atomic::Ordering::Relaxed);
            (*cell).len = bytes.len() as u16;
            (&mut (*cell).bytes)[..bytes.len()].copy_from_slice(bytes);
        }
        cell
    }

    fn fresh_cell(&mut self) -> *mut NumbCell {
        if let Some((ptr, used)) = self.chunks.last_mut()
            && *used < NUMB_CHUNK
        {
            let cell = unsafe { (*ptr).add(*used) };
            *used += 1;
            return cell;
        }
        let layout = std::alloc::Layout::array::<NumbCell>(NUMB_CHUNK).expect("numb slab");
        let ptr = unsafe { std::alloc::alloc(layout) as *mut NumbCell };
        assert!(!ptr.is_null(), "numb slab alloc");
        self.chunks.push((ptr, 1));
        ptr
    }

    fn release(&mut self, cell: *mut NumbCell) {
        self.free.push(cell);
    }
}

static NUMB_HEAP: parking_lot::Mutex<NumbHeap> = parking_lot::Mutex::new(NumbHeap::new());

/// Handle for a `NUMBERING` buffer. Clone is a refcount bump.
#[derive(Debug)]
pub struct NumberingRef {
    inner: NumberingInner,
}

#[derive(Debug)]
enum NumberingInner {
    Slab(std::ptr::NonNull<NumbCell>),
    Heap(std::sync::Arc<[u8]>),
}

unsafe impl Send for NumberingRef {}
unsafe impl Sync for NumberingRef {}

impl NumberingRef {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() <= NUMB_DATA {
            let cell = NUMB_HEAP.lock().alloc(bytes);
            Self {
                inner: NumberingInner::Slab(std::ptr::NonNull::new(cell).expect("numb cell")),
            }
        } else {
            Self {
                inner: NumberingInner::Heap(std::sync::Arc::from(bytes)),
            }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            NumberingInner::Slab(ptr) => unsafe {
                let cell = ptr.as_ref();
                &cell.bytes[..cell.len as usize]
            },
            NumberingInner::Heap(arc) => arc.as_ref(),
        }
    }

    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        match (&a.inner, &b.inner) {
            (NumberingInner::Slab(x), NumberingInner::Slab(y)) => x.as_ptr() == y.as_ptr(),
            (NumberingInner::Heap(x), NumberingInner::Heap(y)) => std::sync::Arc::ptr_eq(x, y),
            _ => false,
        }
    }
}

impl Clone for NumberingRef {
    fn clone(&self) -> Self {
        match &self.inner {
            NumberingInner::Slab(ptr) => {
                unsafe {
                    (*ptr.as_ptr())
                        .refs
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Self {
                    inner: NumberingInner::Slab(*ptr),
                }
            }
            NumberingInner::Heap(arc) => Self {
                inner: NumberingInner::Heap(std::sync::Arc::clone(arc)),
            },
        }
    }
}

impl Drop for NumberingRef {
    fn drop(&mut self) {
        if let NumberingInner::Slab(ptr) = self.inner {
            let prev = unsafe {
                (*ptr.as_ptr())
                    .refs
                    .fetch_sub(1, std::sync::atomic::Ordering::Release)
            };
            if prev == 1 {
                NUMB_HEAP.lock().release(ptr.as_ptr());
            }
        }
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
    fn numbering_ref_slab_shares_identity() {
        let mut w = Writer::new(8);
        for i in 0..20 {
            w.append_int(i);
        }
        let a = w.create_numbering_arc();
        let b = a.clone();
        assert!(NumberingRef::ptr_eq(&a, &b));
        assert_eq!(a.as_slice(), b.as_slice());
        assert_eq!(a.as_slice(), w.create_numbering());
    }

    #[test]
    fn numbering_ref_large_payload_round_trips() {
        let bytes = vec![0x5a; NUMB_DATA + 8];
        let a = NumberingRef::from_bytes(&bytes);
        assert_eq!(a.as_slice(), bytes.as_slice());
        let b = a.clone();
        assert!(NumberingRef::ptr_eq(&a, &b));
    }
}
