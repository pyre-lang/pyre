//! `alloc::vec::Vec<T>` as the translator lowers it.
//!
//! A Rust `Vec<T>` is a raw three-word header — buffer pointer, length and
//! capacity — over a raw item buffer. The translator gives it the lltype
//! `Ptr(Struct(raw) {ptr, len, cap})` and lowers its methods to the `ll_vec_*`
//! helpers in `majit_rlib::lltypesystem::rvec`, the way `rtyper/rlist.py`
//! lowers an RPython list to the `ll_*` helpers of `lltypesystem/rlist.py`.
//!
//! A borrowed slice `&[T]` / `&mut [T]` of the same items is the Rust fat
//! pointer itself: two words, the item pointer and the length, carried as two
//! values. Borrowing a `Vec` as a slice reads its header's pointer and length
//! words; nothing is allocated.
//!
//! This module owns the facts both sides must agree on:
//!
//! * the word index of each header component (`VEC_{PTR,LEN,CAP}_WORD`); the
//!   byte offset is the index times the target word size, and
//! * the `(operation, item kind)` → helper-path tables, read by the rtyper's
//!   `RustVecRepr` and by the front end that emits the same calls.

/// Word index of the capacity in a `Vec<T>` header.
pub const VEC_CAP_WORD: usize = 0;
/// Word index of the buffer pointer in a `Vec<T>` header.
pub const VEC_PTR_WORD: usize = 1;
/// Word index of the length in a `Vec<T>` header.
pub const VEC_LEN_WORD: usize = 2;
/// Number of words in a `Vec<T>` header.
pub const VEC_HEADER_WORDS: usize = 3;

/// Byte offset of header word `index` on a target whose pointer is `word`
/// bytes.
pub const fn vec_word_offset(index: usize, word: usize) -> usize {
    index * word
}

/// Register kind of a `Vec` item, which selects the helper variant.
///
/// The helper's own parameter is `&mut Vec<W>` with `W` = `usize` (`Int`),
/// `GCREF` (`Ref`) or `f64` (`Float`); the item size is passed separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VecItemKind {
    Int,
    Ref,
    Float,
}

impl VecItemKind {
    pub const ALL: [VecItemKind; 3] = [VecItemKind::Int, VecItemKind::Ref, VecItemKind::Float];

    /// The jitcode register-kind letter (`i`, `r`, `f`).
    pub fn from_kind_char(kind: char) -> Option<Self> {
        match kind {
            'i' => Some(VecItemKind::Int),
            'r' => Some(VecItemKind::Ref),
            'f' => Some(VecItemKind::Float),
            _ => None,
        }
    }

    pub fn kind_char(self) -> char {
        match self {
            VecItemKind::Int => 'i',
            VecItemKind::Ref => 'r',
            VecItemKind::Float => 'f',
        }
    }

    /// `(size, alignment)` in bytes of one item on a target whose pointer is
    /// `word` bytes: a word for `Int` and `Ref`, eight bytes for `Float`.
    pub const fn size_align(self, word: usize) -> (usize, usize) {
        match self {
            VecItemKind::Int | VecItemKind::Ref => (word, word),
            VecItemKind::Float => (8, 8),
        }
    }

    fn column(self) -> usize {
        match self {
            VecItemKind::Int => 0,
            VecItemKind::Ref => 1,
            VecItemKind::Float => 2,
        }
    }
}

/// Item kind of a `Vec<item>` whose lowering is handled, from the item's Rust
/// spelling and the target word size.
///
/// Items are one word: `usize` / `isize` (and `u64` / `i64` on a 64-bit
/// target) are `Int`, `f64` is `Float`, and a raw pointer (`*mut T` /
/// `*const T`, the spelling of a managed object reference) is `Ref`. Every
/// other item type, including every type with drop glue, answers `None`.
pub fn vec_item_kind_for_spelling(item: &str, word: usize) -> Option<VecItemKind> {
    let item = item.trim();
    match item {
        "usize" | "isize" => return Some(VecItemKind::Int),
        "u64" | "i64" if word == 8 => return Some(VecItemKind::Int),
        "f64" => return Some(VecItemKind::Float),
        "GCREF" => return Some(VecItemKind::Ref),
        _ => {}
    }
    if item.ends_with("::GCREF") {
        return Some(VecItemKind::Ref);
    }
    if raw_pointer_pointee_is_fat(item) {
        return None;
    }
    (item.starts_with("*mut ") || item.starts_with("*const ")).then_some(VecItemKind::Ref)
}

/// Two-word raw pointers: `dyn Trait`, `[T]`, `str`, and the parenthesized
/// forms `*mut (dyn T + Send)` / `*const ([u8])`.
fn raw_pointer_pointee_is_fat(item: &str) -> bool {
    let item = item.trim();
    let Some(rest) = item
        .strip_prefix("*mut ")
        .or_else(|| item.strip_prefix("*const "))
    else {
        return false;
    };
    let rest = rest.trim();
    let inner = rest
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(rest)
        .trim();
    inner.starts_with("dyn ") || unsized_slice_spelling(inner) || inner == "str"
}

/// `*const [T]` / `*mut [T]` are two-word fat pointers. `*const [T; N]` is
/// one word: the pointee is a sized array.
fn unsized_slice_spelling(inner: &str) -> bool {
    let Some(body) = inner.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
        return false;
    };
    !body.contains(';')
}

/// The item spelling of a `Vec<item>` spelling (`Vec<T>`, `alloc::vec::Vec<T>`,
/// behind any `&` / `&mut ` borrow or `*mut ` / `*const ` raw pointer, all of
/// which are the header address).
pub fn vec_item_spelling(ty: &str) -> Option<&str> {
    let mut ty = ty.trim();
    loop {
        let peeled = ty
            .strip_prefix('&')
            .map(|rest| rest.trim_start().strip_prefix("mut ").unwrap_or(rest))
            .or_else(|| ty.strip_prefix("*mut "))
            .or_else(|| ty.strip_prefix("*const "));
        match peeled {
            Some(rest) => ty = rest.trim_start(),
            None => break,
        }
    }
    let rest = ty
        .strip_prefix("alloc::vec::Vec<")
        .or_else(|| ty.strip_prefix("Vec<"))?;
    let rest = rest.strip_suffix('>')?;
    // `Vec<T, A = Global>`: the allocator is a second type argument.
    // The item kind is T; A is not an item.
    Some(first_generic_arg(rest))
}

/// The first comma-separated generic argument, respecting nested `<>`.
fn first_generic_arg(args: &str) -> &str {
    let mut depth = 0usize;
    for (i, c) in args.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return args[..i].trim(),
            _ => {}
        }
    }
    args.trim()
}

/// Item kind of a `Vec` spelling (see [`vec_item_spelling`]) whose items are
/// one word, or `None`. The `PyObjectRef` alias names
/// `*mut pyobject::PyObject`, a reference. `GCREF` is `llmemory.GCREF`.
pub fn rust_vec_item_kind_for_spelling(ty: &str, word: usize) -> Option<VecItemKind> {
    let item = vec_item_spelling(ty)?.trim();
    if item == "PyObjectRef"
        || item.ends_with("::PyObjectRef")
        || item == "GCREF"
        || item.ends_with("::GCREF")
    {
        return Some(VecItemKind::Ref);
    }
    vec_item_kind_for_spelling(item, word)
}

/// The item spelling of a slice spelling `[item]`, behind any `&` / `&mut `
/// borrow or `*mut ` / `*const ` raw pointer, all of which are the same
/// `(ptr, len)` pair. A fixed-size array `[item; N]` is not a slice.
pub fn slice_item_spelling(ty: &str) -> Option<&str> {
    let mut ty = ty.trim();
    loop {
        let peeled = ty
            .strip_prefix('&')
            .map(|rest| rest.trim_start().strip_prefix("mut ").unwrap_or(rest))
            .or_else(|| ty.strip_prefix("*mut "))
            .or_else(|| ty.strip_prefix("*const "));
        match peeled {
            Some(rest) => ty = rest.trim_start(),
            None => break,
        }
    }
    let item = ty.strip_prefix('[')?.strip_suffix(']')?;
    let mut depth = 0usize;
    for c in item.chars() {
        match c {
            '[' | '<' | '(' => depth += 1,
            ']' | '>' | ')' => depth = depth.saturating_sub(1),
            ';' if depth == 0 => return None,
            _ => {}
        }
    }
    Some(item)
}

/// Item kind of a slice spelling (see [`slice_item_spelling`]) whose items
/// are one word, or `None`. `GCREF` is `llmemory.GCREF`.
pub fn rust_slice_item_kind_for_spelling(ty: &str, word: usize) -> Option<VecItemKind> {
    let item = slice_item_spelling(ty)?.trim();
    if item == "PyObjectRef"
        || item.ends_with("::PyObjectRef")
        || item == "GCREF"
        || item.ends_with("::GCREF")
    {
        return Some(VecItemKind::Ref);
    }
    vec_item_kind_for_spelling(item, word)
}

/// A `Vec` operation that lowers to one `ll_vec_*` helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VecOp {
    /// `Vec::new()` — `ll_newemptylist`.
    NewEmpty,
    /// `Vec::with_capacity(n)` — `ll_newlist_hint`.
    NewHint,
    /// `vec![item; count]` — `rlist.py ll_alloc_and_set`. Args are
    /// `(count, item)`.
    AllocAndSet,
    /// `v.len()` — `ll_length`.
    Length,
    /// `v[i]` read — `ll_getitem_fast`.
    GetItem,
    /// `v[i] = x` — `ll_setitem_fast`.
    SetItem,
    /// `v.push(x)` — `ll_append`.
    Append,
    /// `v.reverse()` — `ll_reverse`.
    Reverse,
    /// Drop of a `Vec` whose items have no drop glue.
    Free,
    /// `ll_items`: the item pointer of the header, the first word of the
    /// `(ptr, len)` pair `Vec::deref` / `as_slice` / `&v[..]` produce.
    Items,
    /// `v.extend_from_slice(s)` — `ll_extend` from the `(ptr, len)` pair.
    ExtendFromSlice,
    /// `v.extend_from_slice(s)` of a one-word object slice — `rlist.ll_extend`
    /// with `l2` the GcArray word (length at offset 0, items after it).
    Extend,
}

impl VecOp {
    pub const ALL: [VecOp; 12] = [
        VecOp::NewEmpty,
        VecOp::NewHint,
        VecOp::Length,
        VecOp::GetItem,
        VecOp::SetItem,
        VecOp::Append,
        VecOp::Reverse,
        VecOp::Free,
        VecOp::Items,
        VecOp::ExtendFromSlice,
        VecOp::AllocAndSet,
        VecOp::Extend,
    ];

    fn row(self) -> usize {
        match self {
            VecOp::NewEmpty => 0,
            VecOp::NewHint => 1,
            VecOp::Length => 2,
            VecOp::GetItem => 3,
            VecOp::SetItem => 4,
            VecOp::Append => 5,
            VecOp::Reverse => 6,
            VecOp::Free => 7,
            VecOp::Items => 8,
            VecOp::ExtendFromSlice => 9,
            VecOp::AllocAndSet => 10,
            VecOp::Extend => 11,
        }
    }
}

/// A slice operation that lowers to one `ll_slice_*` helper. The slice is
/// the `(ptr, len)` pair; its length is the second value itself, so reading
/// it needs no helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SliceOp {
    /// `s[i]` read — `ll_getitem_fast` on `(ptr, i)`.
    GetItem,
    /// `s[i] = x` — `ll_setitem_fast` on `(ptr, i, x)`.
    SetItem,
    /// `s.reverse()` — `ll_reverse` on `(ptr, len)`.
    Reverse,
    /// `s.contains(&x)` — `ll_listcontains` on `(ptr, len, x)`.
    Contains,
    /// The raw item buffer of an array borrowed as a slice: `len` items.
    BufferNew,
    /// `[x; n]` into such a buffer: `(ptr, len, x)`.
    BufferFill,
    /// `s.get(i)` / `s.first()` / `s.last()`: the address of item `i` of
    /// `(ptr, len)`, or 0 when `i` is out of bounds — the one-word
    /// `Option<&T>`.
    GetAddr,
    /// `&s[start..]`: the item pointer `start` items into `(ptr, len)`.
    Offset,
    /// `opt.copied().unwrap_or(default)` over a [`SliceOp::GetAddr`]
    /// address: the item there, or `default` for the null address.
    LoadOr,
    /// `s.to_vec()`: a new `Vec` holding the pair's items.
    ToVec,
    /// `dst.copy_from_slice(src)`: `ll_arraycopy` of `dst`'s length items
    /// from `src` at index 0 into `dst` at index 0.
    ArrayCopy,
    /// `<[T]>::rotate_left` on a pair slice — in-place, same structure as
    /// [`SliceOp::Reverse`].
    RotateLeft,
    /// `<[T]>::rotate_right` on a pair slice — in-place, same structure as
    /// [`SliceOp::Reverse`].
    RotateRight,
}

impl SliceOp {
    pub const ALL: [SliceOp; 13] = [
        SliceOp::GetItem,
        SliceOp::SetItem,
        SliceOp::Reverse,
        SliceOp::Contains,
        SliceOp::BufferNew,
        SliceOp::BufferFill,
        SliceOp::GetAddr,
        SliceOp::Offset,
        SliceOp::LoadOr,
        SliceOp::ToVec,
        SliceOp::ArrayCopy,
        SliceOp::RotateLeft,
        SliceOp::RotateRight,
    ];

    fn row(self) -> usize {
        match self {
            SliceOp::GetItem => 0,
            SliceOp::SetItem => 1,
            SliceOp::Reverse => 2,
            SliceOp::Contains => 3,
            SliceOp::BufferNew => 4,
            SliceOp::BufferFill => 5,
            SliceOp::GetAddr => 6,
            SliceOp::Offset => 7,
            SliceOp::LoadOr => 8,
            SliceOp::ToVec => 9,
            SliceOp::ArrayCopy => 10,
            SliceOp::RotateLeft => 11,
            SliceOp::RotateRight => 12,
        }
    }
}

/// Module that defines every helper in [`VEC_HELPERS`].
pub const RVEC_MODULE: &str = "majit_rlib::lltypesystem::rvec";

/// The raw allocator of a `Vec` header the lowering creates
/// (`OS_RAW_MALLOC_VARSIZE_CHAR`, so an unescaped header is a virtual raw
/// buffer).
pub const VEC_HEADER_MALLOC: &str = "majit_rlib::lltypesystem::rffi::raw_malloc_varsize_char";

/// The opaque item-buffer allocator: `(allocated, itemsize, align)` → buffer
/// address.
pub const VEC_BUF_ALLOC: &str = "majit_rlib::lltypesystem::rvec::vec_buf_alloc";

/// The zero-filling item-buffer allocator (`raw_malloc(..., zero=True)`):
/// `(allocated, itemsize, align)` → buffer address.
pub const VEC_BUF_ALLOC_CLEAR: &str = "majit_rlib::lltypesystem::rvec::vec_buf_alloc_clear";

/// `(operation, item kind)` → helper path, rows in [`VecOp::ALL`] order and
/// columns in [`VecItemKind::ALL`] order. Every path is spelled here once.
const VEC_HELPERS: [[&str; 3]; 12] = [
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_newemptylist_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_newemptylist_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_newemptylist_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_newlist_hint_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_newlist_hint_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_newlist_hint_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_length_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_length_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_length_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_getitem_fast_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_getitem_fast_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_getitem_fast_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_setitem_fast_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_setitem_fast_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_setitem_fast_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_append_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_append_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_append_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_reverse_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_reverse_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_reverse_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_free_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_free_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_free_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_items_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_items_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_items_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_extend_from_slice_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_extend_from_slice_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_extend_from_slice_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_alloc_and_set_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_alloc_and_set_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_alloc_and_set_f",
    ],
    // Only `_r`: a one-word GcArray source is an object slice
    // (`Ptr(GcArray(Ptr(PyObject)))`). Int/float slices stay `(ptr, len)`
    // pairs and use [`VecOp::ExtendFromSlice`]. The `_i`/`_f` slots keep the
    // table rectangular; the front never emits them.
    [
        "majit_rlib::lltypesystem::rvec::ll_vec_extend_i",
        "majit_rlib::lltypesystem::rvec::ll_vec_extend_r",
        "majit_rlib::lltypesystem::rvec::ll_vec_extend_f",
    ],
];

/// `(operation, item kind)` → slice helper path, rows in [`SliceOp::ALL`]
/// order and columns in [`VecItemKind::ALL`] order.
const SLICE_HELPERS: [[&str; 3]; 13] = [
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_getitem_fast_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_getitem_fast_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_getitem_fast_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_setitem_fast_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_setitem_fast_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_setitem_fast_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_reverse_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_reverse_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_reverse_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_contains_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_contains_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_contains_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_buffer_new_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_buffer_new_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_buffer_new_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_buffer_fill_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_buffer_fill_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_buffer_fill_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_get_addr_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_get_addr_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_get_addr_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_offset_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_offset_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_offset_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_load_or_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_load_or_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_load_or_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_to_vec_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_to_vec_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_to_vec_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_arraycopy_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_arraycopy_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_arraycopy_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_rotate_left_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_rotate_left_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_rotate_left_f",
    ],
    [
        "majit_rlib::lltypesystem::rvec::ll_slice_rotate_right_i",
        "majit_rlib::lltypesystem::rvec::ll_slice_rotate_right_r",
        "majit_rlib::lltypesystem::rvec::ll_slice_rotate_right_f",
    ],
];

/// `ll_slice_buffer_free`: frees an array's raw item buffer at the end of
/// the array's storage.
pub const SLICE_BUFFER_FREE: &str = "majit_rlib::lltypesystem::rvec::ll_slice_buffer_free";

/// `ll_slice_bounds_panic`: DLI residual for a pair-slice index or
/// `copy_from_slice` length mismatch. The call is not an implicit
/// `AssertionError`, so `remove_assertion_errors` leaves the fail arm.
pub const SLICE_BOUNDS_PANIC: &str = "majit_rlib::lltypesystem::rvec::ll_slice_bounds_panic";

/// `ll_slice_len_slot_new`: the caller allocates the one-word out-param slot
/// a slice-returning callee stores the length into.
pub const SLICE_LEN_SLOT_NEW: &str = "majit_rlib::lltypesystem::rvec::ll_slice_len_slot_new";
/// `ll_slice_len_slot_store`: the callee stores the returned length.
pub const SLICE_LEN_SLOT_STORE: &str = "majit_rlib::lltypesystem::rvec::ll_slice_len_slot_store";
/// `ll_slice_len_slot_take`: the caller reads the length and frees the slot.
pub const SLICE_LEN_SLOT_TAKE: &str = "majit_rlib::lltypesystem::rvec::ll_slice_len_slot_take";

/// Path of the helper that lowers `op` on a slice of `kind` items.
pub fn slice_helper_path(op: SliceOp, kind: VecItemKind) -> &'static str {
    SLICE_HELPERS[op.row()][kind.column()]
}

/// The `(operation, item kind)` a slice helper path lowers, if it is one of
/// [`SLICE_HELPERS`].
pub fn slice_helper_for_path(path: &str) -> Option<(SliceOp, VecItemKind)> {
    SliceOp::ALL.into_iter().find_map(|op| {
        VecItemKind::ALL
            .into_iter()
            .find(|&kind| slice_helper_path(op, kind) == path)
            .map(|kind| (op, kind))
    })
}

/// Path of the helper that lowers `op` on a `Vec` of `kind` items.
pub fn vec_helper_path(op: VecOp, kind: VecItemKind) -> &'static str {
    VEC_HELPERS[op.row()][kind.column()]
}

/// The `(operation, item kind)` a helper path lowers, if it is one of
/// [`VEC_HELPERS`].
pub fn vec_helper_for_path(path: &str) -> Option<(VecOp, VecItemKind)> {
    VecOp::ALL.into_iter().find_map(|op| {
        VecItemKind::ALL
            .into_iter()
            .find(|&kind| vec_helper_path(op, kind) == path)
            .map(|kind| (op, kind))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_words_are_three_distinct_indices() {
        let mut words = [VEC_CAP_WORD, VEC_PTR_WORD, VEC_LEN_WORD];
        words.sort_unstable();
        assert_eq!(words, [0, 1, 2]);
        assert_eq!(VEC_HEADER_WORDS, 3);
    }

    #[test]
    fn header_words_match_the_host_vec() {
        let mut sample = Vec::<u8>::with_capacity(4);
        sample.push(1);
        let words = unsafe {
            std::slice::from_raw_parts(
                (&sample as *const Vec<u8>).cast::<usize>(),
                VEC_HEADER_WORDS,
            )
        };
        assert_eq!(words[VEC_PTR_WORD], sample.as_ptr() as usize);
        assert_eq!(words[VEC_LEN_WORD], sample.len());
        assert_eq!(words[VEC_CAP_WORD], sample.capacity());
    }

    #[test]
    fn item_size_align_matches_the_host() {
        let word = std::mem::size_of::<usize>();
        assert_eq!(
            VecItemKind::Int.size_align(word),
            (std::mem::size_of::<usize>(), std::mem::align_of::<usize>())
        );
        assert_eq!(
            VecItemKind::Ref.size_align(word),
            (
                std::mem::size_of::<*mut u8>(),
                std::mem::align_of::<*mut u8>()
            )
        );
        assert_eq!(
            VecItemKind::Float.size_align(word),
            (std::mem::size_of::<f64>(), std::mem::align_of::<f64>())
        );
    }

    #[test]
    fn item_kinds_are_one_word() {
        assert_eq!(
            vec_item_kind_for_spelling("usize", 4),
            Some(VecItemKind::Int)
        );
        assert_eq!(vec_item_kind_for_spelling("i64", 8), Some(VecItemKind::Int));
        assert_eq!(vec_item_kind_for_spelling("i64", 4), None);
        assert_eq!(
            vec_item_kind_for_spelling("f64", 4),
            Some(VecItemKind::Float)
        );
        assert_eq!(
            vec_item_kind_for_spelling("*mut pyobject::PyObject", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            vec_item_kind_for_spelling("GCREF", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            vec_item_kind_for_spelling("majit_gc::GCREF", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            vec_item_kind_for_spelling("*mut GCREFOpaque", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            vec_item_kind_for_spelling("*mut majit_gc::header::GCREFOpaque", 8),
            Some(VecItemKind::Ref)
        );
        for other in [
            "u8",
            "u32",
            "bool",
            "String",
            "Box<Foo>",
            "Vec<usize>",
            "&Foo",
            "*mut dyn AsyncActionOps",
            "*const dyn Foo",
            "*mut (dyn AsyncActionOps)",
            "*const (dyn Foo + Send)",
            "*const [u8]",
            "*mut [usize]",
            "*const ([u8])",
            "*const str",
            "*mut str",
        ] {
            assert_eq!(vec_item_kind_for_spelling(other, 8), None, "{other}");
        }
        assert_eq!(
            vec_item_kind_for_spelling("*const [u8; 4]", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            vec_item_kind_for_spelling("*mut [usize; 2]", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            vec_item_kind_for_spelling("*const ([u8; 4])", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(vec_item_spelling("Vec<usize>"), Some("usize"));
        assert_eq!(vec_item_spelling("&mut alloc::vec::Vec<f64>"), Some("f64"));
        assert_eq!(vec_item_spelling("VecDeque<usize>"), None);
        assert_eq!(vec_item_spelling("*mut Vec<f64>"), Some("f64"));
        assert_eq!(vec_item_spelling("& &mut Vec<isize>"), Some("isize"));
        assert_eq!(
            vec_item_spelling("Vec<PyObjectRef, Global>"),
            Some("PyObjectRef")
        );
        assert_eq!(
            vec_item_spelling("alloc::vec::Vec<pyre_object::PyObjectRef,alloc::alloc::Global>"),
            Some("pyre_object::PyObjectRef")
        );
        assert_eq!(
            rust_vec_item_kind_for_spelling("Vec<PyObjectRef, Global>", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_vec_item_kind_for_spelling("*mut Vec<GCREF>", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_vec_item_kind_for_spelling("&mut Vec<PyObjectRef>", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_vec_item_kind_for_spelling("Vec<pyre_object::PyObjectRef>", 4),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_vec_item_kind_for_spelling("alloc::vec::Vec<usize>", 4),
            Some(VecItemKind::Int)
        );
        assert_eq!(rust_vec_item_kind_for_spelling("Vec<String>", 8), None);
        assert_eq!(rust_vec_item_kind_for_spelling("[usize]", 8), None);
    }

    #[test]
    fn slice_item_kinds_are_one_word() {
        assert_eq!(slice_item_spelling("&[usize]"), Some("usize"));
        assert_eq!(slice_item_spelling("&mut [f64]"), Some("f64"));
        assert_eq!(
            slice_item_spelling("[*mut PyObject]"),
            Some("*mut PyObject")
        );
        assert_eq!(slice_item_spelling("*const [usize]"), Some("usize"));
        assert_eq!(slice_item_spelling("[usize; 4]"), None);
        assert_eq!(slice_item_spelling("[[usize; 2]]"), Some("[usize; 2]"));
        assert_eq!(slice_item_spelling("Vec<usize>"), None);
        assert_eq!(
            rust_slice_item_kind_for_spelling("&[GCREF]", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_slice_item_kind_for_spelling("&[PyObjectRef]", 8),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_slice_item_kind_for_spelling("[pyre_object::PyObjectRef]", 4),
            Some(VecItemKind::Ref)
        );
        assert_eq!(
            rust_slice_item_kind_for_spelling("&mut [f64]", 4),
            Some(VecItemKind::Float)
        );
        assert_eq!(rust_slice_item_kind_for_spelling("&[u8]", 8), None);
        assert_eq!(rust_slice_item_kind_for_spelling("&[usize; 3]", 8), None);
    }

    #[test]
    fn every_slice_helper_lives_in_the_rvec_module_and_round_trips() {
        for op in SliceOp::ALL {
            for kind in VecItemKind::ALL {
                let path = slice_helper_path(op, kind);
                let leaf = path
                    .strip_prefix(RVEC_MODULE)
                    .and_then(|rest| rest.strip_prefix("::"))
                    .unwrap_or_else(|| panic!("{path} is outside {RVEC_MODULE}"));
                assert!(leaf.starts_with("ll_slice_"), "{path}");
                assert!(leaf.ends_with(&format!("_{}", kind.kind_char())), "{path}");
                assert_eq!(slice_helper_for_path(path), Some((op, kind)));
                assert_eq!(vec_helper_for_path(path), None);
            }
        }
    }

    #[test]
    fn every_helper_lives_in_the_rvec_module_and_round_trips() {
        for op in VecOp::ALL {
            for kind in VecItemKind::ALL {
                let path = vec_helper_path(op, kind);
                let leaf = path
                    .strip_prefix(RVEC_MODULE)
                    .and_then(|rest| rest.strip_prefix("::"))
                    .unwrap_or_else(|| panic!("{path} is outside {RVEC_MODULE}"));
                assert!(leaf.starts_with("ll_vec_"), "{path}");
                assert!(leaf.ends_with(&format!("_{}", kind.kind_char())), "{path}");
                assert_eq!(vec_helper_for_path(path), Some((op, kind)));
            }
        }
        assert_eq!(
            vec_helper_for_path("majit_rlib::lltypesystem::rvec::ll_vec_nope_i"),
            None
        );
    }

    #[test]
    fn alloc_and_set_row_names_ll_alloc_and_set() {
        for kind in VecItemKind::ALL {
            let path = vec_helper_path(VecOp::AllocAndSet, kind);
            assert!(
                path.ends_with(&format!("::ll_vec_alloc_and_set_{}", kind.kind_char())),
                "{path}"
            );
            assert_eq!(vec_helper_for_path(path), Some((VecOp::AllocAndSet, kind)));
        }
    }
}
