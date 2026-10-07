//! Nursery storage for `opencoder.py` `Trace` pools other than `_ops`.
//!
//! `ListRepr` is a list header (`length` + `items`) plus a `GcArray`.
//! `_snapshot_data` and `_snapshot_array_data` are that pair: `encode_varint_signed`
//! appends, so the list is resized (`_ll_list_resize_ge` / `_ll_list_resize_hint_really`),
//! not a fixed `GcArray(Char)`. `_bigints` and `_floats` are `GcArray`s of
//! plain words. `_refs` is a `GcArray` of `GCREF` (`items_have_gc_ptrs`),
//! rooted by one `OwnerRootGuard`. `_descrs` stays a host `Vec`: `DescrRef`
//! is `Arc<dyn Descr>` and is not a GC item until the descr objects themselves
//! move (later stage).

use std::marker::PhantomData;

use majit_ir::GcRef;

const WORD: usize = std::mem::size_of::<usize>();

fn gc_installed() -> bool {
    majit_gc::current_thread_gc_available()
}

fn host_layout(payload: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(payload, std::mem::align_of::<usize>())
        .unwrap_or_else(|_| std::alloc::Layout::new::<usize>())
}

fn host_alloc(payload: usize) -> *mut u8 {
    let layout = host_layout(payload);
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    ptr
}

fn host_free(ptr: *mut u8, payload: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe { std::alloc::dealloc(ptr, host_layout(payload)) };
}

fn list_overallocate(newsize: usize) -> usize {
    // `_ll_list_resize_hint_really` when `overallocate` is true.
    if newsize == 0 {
        return 0;
    }
    let some = if newsize < 9 { 3 } else { 6 };
    newsize + some + (newsize >> 3)
}

fn type_id(slot: &std::sync::atomic::AtomicU32, what: &str) -> u32 {
    let id = slot.load(std::sync::atomic::Ordering::Acquire);
    assert!(
        id != u32::MAX,
        "{what} is not registered; call register_trace_ops_gc_type"
    );
    id
}

static TRACE_LIST_HDR_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);
static TRACE_WORD_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);
static TRACE_REFS_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

/// Header, plain-word array, and ref array. Char items reuse `Trace._ops`'
/// `GcArray(Char)` type (`register_trace_ops_gc_type`).
pub(crate) fn register_trace_pool_gc_types(gc: &mut dyn majit_gc::GcAllocator) {
    let hdr = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
        std::mem::size_of::<ListHdr>(),
        vec![std::mem::size_of::<usize>()],
    ));
    TRACE_LIST_HDR_GC_TYPE_ID.store(hdr, std::sync::atomic::Ordering::Release);
    let word = gc.register_type(majit_gc::trace::TypeInfo::varsize(
        WORD,
        std::mem::size_of::<u64>(),
        0,
        false,
        Vec::new(),
    ));
    TRACE_WORD_GC_TYPE_ID.store(word, std::sync::atomic::Ordering::Release);
    let refs = gc.register_type(majit_gc::trace::TypeInfo::varsize(
        WORD,
        std::mem::size_of::<usize>(),
        0,
        true,
        Vec::new(),
    ));
    TRACE_REFS_GC_TYPE_ID.store(refs, std::sync::atomic::Ordering::Release);
}

fn list_hdr_type_id() -> u32 {
    type_id(&TRACE_LIST_HDR_GC_TYPE_ID, "Trace list header")
}

fn word_type_id() -> u32 {
    type_id(&TRACE_WORD_GC_TYPE_ID, "Trace word GcArray")
}

fn refs_type_id() -> u32 {
    type_id(&TRACE_REFS_GC_TYPE_ID, "Trace._refs GcArray")
}

/// `ListRepr` header. `items` is `Ptr(GcArray(Char))`.
#[repr(C)]
struct ListHdr {
    length: usize,
    items: usize,
}

/// `opencoder.py` `_snapshot_data` / `_snapshot_array_data`.
///
/// One `OwnerRootGuard` holds the header. `items` is walked from the header,
/// so a minor updates both. Every access re-reads the guard.
pub struct CharList {
    root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    host: *mut ListHdr,
}

impl CharList {
    pub fn with_capacity(cap: usize) -> Self {
        if !gc_installed() {
            return Self::host_with_capacity(cap);
        }
        let mut scratch = GcRef(0);
        let hdr = unsafe {
            let mut needs = false;
            let fresh = majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                list_hdr_type_id(),
                std::mem::size_of::<ListHdr>(),
                &mut scratch,
                &mut needs,
            );
            if fresh.0 == 0 {
                majit_gc::gc_alloc_failed(std::mem::size_of::<ListHdr>());
            }
            let h = fresh.0 as *mut ListHdr;
            (*h).length = 0;
            (*h).items = 0;
            let _ = needs;
            fresh
        };
        let root = majit_gc::shadow_stack::OwnerRootGuard::new(hdr);
        let mut list = Self {
            root: Some(root),
            host: std::ptr::null_mut(),
        };
        if cap > 0 {
            list.realloc_items(cap);
        }
        list
    }

    fn host_with_capacity(cap: usize) -> Self {
        let host = host_alloc(std::mem::size_of::<ListHdr>()) as *mut ListHdr;
        unsafe {
            (*host).length = 0;
            (*host).items = 0;
        }
        let mut list = Self { root: None, host };
        if cap > 0 {
            list.realloc_items(cap);
        }
        list
    }

    fn header_ref(&self) -> GcRef {
        if let Some(guard) = &self.root {
            guard.get()
        } else {
            GcRef(self.host as usize)
        }
    }

    fn header(&self) -> *mut ListHdr {
        self.header_ref().0 as *mut ListHdr
    }

    pub fn capacity(&self) -> usize {
        let items = unsafe { (*self.header()).items };
        if items == 0 {
            0
        } else {
            unsafe { *(items as *const usize) }
        }
    }

    fn chars(&self) -> *mut u8 {
        let items = unsafe { (*self.header()).items };
        if items == 0 {
            std::ptr::null_mut()
        } else {
            unsafe { (items as *mut u8).add(WORD) }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self
    }

    /// `encode_varint_signed` into this list (`_ll_list_resize_ge` on growth).
    pub fn append_varint(&mut self, value: i64) {
        let (bytes, n) = super::encode_varint_signed_array(value);
        self.append_bytes(&bytes[..n]);
    }

    pub fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let need = self.len() + bytes.len();
        if need > self.capacity() {
            self.realloc_items(list_overallocate(need));
        }
        let h = self.header();
        unsafe {
            let len = (*h).length;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.chars().add(len), bytes.len());
            (*h).length = len + bytes.len();
        }
    }

    pub fn pop(&mut self) -> u8 {
        let h = self.header();
        let b = unsafe {
            let len = (*h).length;
            assert!(len > 0, "snapshot list pop on empty");
            let b = *self.chars().add(len - 1);
            (*h).length = len - 1;
            b
        };
        // `_ll_list_resize_le`: shrink once length falls under half the allocation.
        let len = self.len();
        let cap = self.capacity();
        if cap > 0 && len < (cap >> 1).saturating_sub(5) {
            self.realloc_items(len);
        }
        b
    }

    fn realloc_items(&mut self, new_cap: usize) {
        let old_len = self.len();
        let copy_n = old_len.min(new_cap);
        let old_items = unsafe { (*self.header()).items };
        let fresh_items = if !gc_installed() {
            debug_assert!(
                self.root.is_none(),
                "host snapshot list while a GC is installed"
            );
            if new_cap == 0 {
                0
            } else {
                let payload = WORD + new_cap;
                let ptr = host_alloc(payload);
                unsafe { *(ptr as *mut usize) = new_cap };
                ptr as usize
            }
        } else if new_cap == 0 {
            0
        } else {
            let mut live = self.header_ref();
            let mut needs = false;
            let fresh = unsafe {
                majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                    super::trace_ops_gc_type_id(),
                    WORD + new_cap,
                    &mut live,
                    &mut needs,
                )
            };
            if fresh.0 == 0 {
                majit_gc::gc_alloc_failed(WORD + new_cap);
            }
            unsafe {
                *(fresh.0 as *mut usize) = new_cap;
                std::ptr::write_bytes((fresh.0 as *mut u8).add(WORD), 0, new_cap);
            }
            let _ = needs;
            fresh.0
        };
        // Re-read the header. The allocating call may have moved it.
        let h = self.header();
        let old_items = if gc_installed() {
            unsafe { (*h).items }
        } else {
            old_items
        };
        unsafe {
            if copy_n > 0 && old_items != 0 && fresh_items != 0 {
                std::ptr::copy_nonoverlapping(
                    (old_items as *const u8).add(WORD),
                    (fresh_items as *mut u8).add(WORD),
                    copy_n,
                );
            }
            if gc_installed() {
                majit_gc::gc_write_barrier(GcRef(h as usize));
            }
            (*h).items = fresh_items;
        }
        if self.root.is_none() && old_items != 0 {
            let old_cap = unsafe { *(old_items as *const usize) };
            host_free(old_items as *mut u8, WORD + old_cap);
        }
    }
}

impl std::ops::Deref for CharList {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        let n = unsafe { (*self.header()).length };
        if n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.chars(), n) }
    }
}

impl Drop for CharList {
    fn drop(&mut self) {
        if self.root.is_some() || self.host.is_null() {
            return;
        }
        unsafe {
            let items = (*self.host).items;
            if items != 0 {
                let cap = *(items as *const usize);
                host_free(items as *mut u8, WORD + cap);
            }
        }
        host_free(self.host as *mut u8, std::mem::size_of::<ListHdr>());
        self.host = std::ptr::null_mut();
    }
}

/// `GcArray` of fixed-size items. Logical length sits on the host holder
/// (`Trace` is still host). The array length word is the allocation, which
/// is what a minor copies. One `OwnerRootGuard` roots the array.
pub struct WordArray<T: Copy> {
    root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    host: *mut u8,
    len: usize,
    cap: usize,
    /// `Trace._refs`: items are `GCREF` (`items_have_gc_ptrs`).
    gc_ptrs: bool,
    type_id: fn() -> u32,
    _ty: PhantomData<T>,
}

impl<T: Copy> WordArray<T> {
    pub fn empty(gc_ptrs: bool, type_id: fn() -> u32) -> Self {
        Self {
            root: None,
            host: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            gc_ptrs,
            type_id,
            _ty: PhantomData,
        }
    }

    pub fn with_capacity(cap: usize, gc_ptrs: bool, type_id: fn() -> u32) -> Self {
        let mut arr = Self::empty(gc_ptrs, type_id);
        if cap > 0 {
            arr.realloc(cap);
        }
        arr
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn as_slice(&self) -> &[T] {
        self
    }

    pub fn set(&mut self, index: usize, value: T) {
        assert!(
            index < self.len,
            "word array index {index} len {}",
            self.len
        );
        unsafe { *self.item_ptr().add(index) = value };
    }

    fn addr(&self) -> *mut u8 {
        if let Some(guard) = &self.root {
            guard.get().0 as *mut u8
        } else {
            self.host
        }
    }

    fn item_ptr(&self) -> *mut T {
        unsafe { self.addr().add(WORD) as *mut T }
    }

    /// Append. Returns the stored value: a `GCREF` may have been forwarded
    /// when growth collected.
    pub fn push(&mut self, value: T) -> T {
        if self.len == self.cap {
            return self.grow_push(value);
        }
        if self.gc_ptrs
            && let Some(guard) = &self.root
        {
            majit_gc::gc_write_barrier(guard.get());
        }
        unsafe { *self.item_ptr().add(self.len) = value };
        self.len += 1;
        value
    }

    fn grow_push(&mut self, value: T) -> T {
        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap.saturating_mul(2)
        };
        let stored = self.realloc_keeping(new_cap, Some(value));
        unsafe { *self.item_ptr().add(self.len) = stored };
        self.len += 1;
        stored
    }

    fn realloc(&mut self, new_cap: usize) {
        let _ = self.realloc_keeping(new_cap, None);
    }

    fn realloc_keeping(&mut self, new_cap: usize, extra: Option<T>) -> T {
        let payload = WORD + new_cap * std::mem::size_of::<T>();
        let old_len = self.len;
        let placeholder = extra.unwrap_or_else(|| unsafe { std::mem::zeroed() });
        if !gc_installed() {
            debug_assert!(
                self.root.is_none(),
                "host trace word array used while a GC is installed"
            );
            let ptr = host_alloc(payload);
            unsafe {
                *(ptr as *mut usize) = new_cap;
                if old_len > 0 {
                    std::ptr::copy_nonoverlapping(
                        self.item_ptr(),
                        ptr.add(WORD) as *mut T,
                        old_len,
                    );
                }
            }
            if !self.host.is_null() {
                host_free(self.host, WORD + self.cap * std::mem::size_of::<T>());
            }
            self.host = ptr;
            self.cap = new_cap;
            return placeholder;
        }
        if self.gc_ptrs {
            assert_eq!(
                std::mem::size_of::<T>(),
                std::mem::size_of::<usize>(),
                "GCREF item must be pointer-sized"
            );
        }
        let old = if let Some(guard) = &self.root {
            guard.get()
        } else {
            GcRef(0)
        };
        let extra_addr = if self.gc_ptrs {
            extra.as_ref().map(gc_addr_bits).unwrap_or(0)
        } else {
            0
        };
        let mut roots = [old, GcRef(extra_addr)];
        let root_count = if self.gc_ptrs { 2 } else { 1 };
        let mut needs = false;
        let fresh = unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_roots(
                (self.type_id)(),
                payload,
                roots.as_mut_ptr(),
                root_count,
                &mut needs,
            )
        };
        if fresh.0 == 0 {
            majit_gc::gc_alloc_failed(payload);
        }
        let stored = if self.gc_ptrs && extra.is_some() {
            from_gc_addr(roots[1].0)
        } else {
            placeholder
        };
        unsafe {
            *(fresh.0 as *mut usize) = new_cap;
            let dst = (fresh.0 as *mut u8).add(WORD) as *mut T;
            if roots[0].0 != 0 && old_len > 0 {
                let src = (roots[0].0 as *const u8).add(WORD) as *const T;
                std::ptr::copy_nonoverlapping(src, dst, old_len);
            }
            if new_cap > old_len {
                std::ptr::write_bytes(dst.add(old_len), 0, new_cap - old_len);
            }
            if needs {
                majit_gc::gc_write_barrier(fresh);
            }
        }
        self.root = Some(majit_gc::shadow_stack::OwnerRootGuard::new(fresh));
        self.host = std::ptr::null_mut();
        self.cap = new_cap;
        stored
    }
}

impl<T: Copy> std::ops::Deref for WordArray<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.item_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for WordArray<T> {
    fn drop(&mut self) {
        if self.root.is_some() || self.host.is_null() {
            return;
        }
        host_free(self.host, WORD + self.cap * std::mem::size_of::<T>());
        self.host = std::ptr::null_mut();
    }
}

fn gc_addr_bits<T: Copy>(value: &T) -> usize {
    let mut bits = 0usize;
    unsafe {
        std::ptr::copy_nonoverlapping(
            (value as *const T).cast::<u8>(),
            (&mut bits as *mut usize).cast::<u8>(),
            std::mem::size_of::<usize>(),
        );
    }
    bits
}

fn from_gc_addr<T: Copy>(bits: usize) -> T {
    let mut value = unsafe { std::mem::zeroed() };
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&bits as *const usize).cast::<u8>(),
            (&mut value as *mut T).cast::<u8>(),
            std::mem::size_of::<usize>(),
        );
    }
    value
}

/// `Trace._refs`: one slot per `GCREF`. Item width is `usize`, which is
/// 4 bytes on wasm32 and 8 on the native backends.
pub fn new_refs() -> WordArray<usize> {
    let mut refs = WordArray::with_capacity(32, true, refs_type_id);
    refs.push(0);
    refs
}

pub fn new_bigints() -> WordArray<i64> {
    WordArray::empty(false, word_type_id)
}

pub fn new_floats() -> WordArray<u64> {
    WordArray::empty(false, word_type_id)
}

pub fn new_snapshot() -> CharList {
    CharList::with_capacity(128)
}
