//! Charon fixture corpus: representative shapes from issue #97.
//!
//! 1. `straight_line_add` — straight-line interpreter-shaped function.
//! 2. `branch_loop_sum`   — branch + loop, like opcode dispatch fragments.
//! 3. `strategy_dispatch` — enum-as-strategy (dict-strategy stand-in).
//! 4. `desugar_mix`       — `?`, `match`, and iterator desugaring together.

#![allow(dead_code)]

use majit_gc::GCREF;

pub type PyResult<T> = Result<T, &'static str>;

// 1. Straight-line
/// `#[repr(C)]` holder: one word, then an inline `Vec<u8>`.
/// `clear_inline_tag` borrows the `Vec` (`s.tags[i]`) the way a frame
/// reads `operand_tags`.
#[repr(C)]
pub struct TagHolder {
    pub word: i64,
    pub tags: Vec<u8>,
}

#[inline(never)]
pub fn clear_inline_tag(s: &mut TagHolder, i: usize) -> u8 {
    const N: usize = 8;
    if i < N && s.tags[i] != 0 {
        let t = s.tags[i];
        s.tags[i] = 0;
        return t;
    }
    0
}

#[inline(never)]
pub fn straight_line_add(a: i64, b: i64, c: i64) -> i64 {
    let s = a + b;
    let t = s * 2;
    t + c
}

// 2. Branch and loop
#[inline(never)]
pub fn branch_loop_sum(slice: &[i64], threshold: i64) -> i64 {
    let mut acc: i64 = 0;
    for &v in slice {
        if v > threshold {
            acc += v;
        } else {
            acc -= v;
        }
    }
    acc
}

// 2b. Iterator element kinds.  `next()`'s payload carries one reference for
// a slice iterator (`core::slice::iter::Iter` yields `Option<&T>`) and none
// for a by-value one (`core::array::iter::IntoIter` yields `Option<T>`), so
// the two spell the same `Option<&i64>` payload for different reasons: here
// the element is `&i64` both times, and only the first has a reference the
// iterator added.  A frontend that peels unconditionally, or never, types one
// of the two into the wrong register bank.
#[inline(never)]
pub fn slice_of_refs_sum(slice: &[&i64]) -> i64 {
    let mut acc: i64 = 0;
    for r in slice {
        acc += **r;
    }
    acc
}

#[inline(never)]
pub fn array_of_refs_sum(refs: [&i64; 3]) -> i64 {
    let mut acc: i64 = 0;
    for r in refs {
        acc += *r;
    }
    acc
}

// 3. Strategy dispatch (dict-strategy stand-in)
pub enum Strategy {
    Empty,
    IntKeyed { len: usize },
    StrKeyed { len: usize, capacity: usize },
}

#[inline(never)]
pub fn strategy_len(s: &Strategy) -> usize {
    match s {
        Strategy::Empty => 0,
        Strategy::IntKeyed { len } => *len,
        Strategy::StrKeyed { len, capacity: _ } => *len,
    }
}

// 4. Desugaring mix: `?`, `match`, and iteration
pub enum Token {
    Add(i64),
    Sub(i64),
    Halt,
}

fn parse_one(raw: i64) -> PyResult<Token> {
    match raw {
        i64::MIN => Ok(Token::Halt),
        0 => Err("halt-zero forbidden"),
        v if v > 0 => Ok(Token::Add(v)),
        v => Ok(Token::Sub(-v)),
    }
}

#[inline(never)]
pub fn desugar_mix(input: &[i64]) -> PyResult<i64> {
    let mut acc: i64 = 0;
    for &raw in input.iter() {
        let tok = parse_one(raw)?;
        match tok {
            Token::Add(v) => acc += v,
            Token::Sub(v) => acc -= v,
            Token::Halt => break,
        }
    }
    Ok(acc)
}

// 5. Tuple round-trip: construct a tuple and read both fields
//
// Exercises `Rvalue::Aggregate` for a *non-Adt* (tuple) value paired
// with `Field` projection reads of that same local. The lowering must
// emit a `__pos_<idx>` `FieldRead` symmetric to the construction-side
// `FieldWrite` chain rather than collapsing every `.N` to the base.

#[inline(never)]
pub fn tuple_roundtrip(a: i64, b: i64) -> i64 {
    let pair = (a + b, a - b);
    pair.0 * pair.1
}

// 6. Closures
// `bool_then_closure` is the exact `core::bool::<Impl>::then` census shape:
// an opaque combinator taking a `FnOnce` closure that captures a value from
// the enclosing scope. Charon extracts the closure's `call_once` body as a
// transparent inherent method of the closure type.

#[inline(never)]
pub fn bool_then_closure(c: bool, x: i64) -> Option<i64> {
    c.then(|| x + 1)
}

// `then_some` is the eager sibling of `then`: it takes an already-evaluated
// value rather than a closure, so the diamond's `then` arm wraps it in `Some`
// directly (no `call_once`). Same Opaque-core-combinator residual shape.
#[inline(never)]
pub fn bool_then_some(c: bool, x: i64) -> Option<i64> {
    c.then_some(x + 1)
}

// 7. Option question mark
// Exercises `Try::branch` on `Option`: `Some(v)` continues with `v`, while
// `None` returns `None` normally from the enclosing Option-returning function.

#[inline(never)]
fn option_source(keep: bool, value: i64) -> Option<i64> {
    if keep {
        Some(value)
    } else {
        None
    }
}

#[inline(never)]
pub fn option_question_mark(keep: bool, value: i64, addend: i64) -> Option<i64> {
    let v = option_source(keep, value)?;
    Some(v + addend)
}

/// `Option<*mut c_void>` from an integer. A raw pointer has no niche, so
/// this is a tagged Option, not the nullable-pointer spelling of
/// `Option<&T>` / `Option<NonNull<T>>`.
#[inline(never)]
pub fn option_raw_c_void_from_int(addr: isize) -> Option<*mut core::ffi::c_void> {
    if addr == 0 {
        return None;
    }
    Some(addr as *mut core::ffi::c_void)
}

/// `rffi.VOIDP` is `Ptr(Array(Char, render_as_void))` (`rffi.py VOIDP`):
/// `getkind` `'int'`. The Rust spelling is `*mut c_void`. A trait-method
/// return of that pointer stamps FUNC.RESULT the same way
/// `CastInto::cast_into` does.
pub trait CastCcharpToVoidP {
    fn ccharp_to_voidp(self) -> *mut core::ffi::c_void;
}

impl CastCcharpToVoidP for *const u8 {
    #[inline(never)]
    fn ccharp_to_voidp(self) -> *mut core::ffi::c_void {
        self as *mut core::ffi::c_void
    }
}

/// Raw struct whose pointer is an address (`lltype.Signed`).
#[repr(C)]
pub struct SomeRawStruct {
    pub word: i64,
}

/// `Option<*mut SomeRawStruct>` from an integer. Same Option layout as
/// `option_raw_c_void_from_int`: a raw pointer may be null, so the Option
/// is tagged rather than a nullable pointer word.
#[inline(never)]
pub fn option_raw_struct_from_int(addr: isize) -> Option<*mut SomeRawStruct> {
    if addr == 0 {
        return None;
    }
    Some(addr as *mut SomeRawStruct)
}

// 8. Header-first object model
//
//   1. `(*w).ob_type` off a `*mut ObjectHeader` — a `FieldRead` preceded by
//      a `__cast_instance_intrinsic/<Root>` narrow, not a classdef-less read.
//   2. `lltype::malloc_typed(Leaf { ob_header: .., payload })` — one
//      by-value argument, header written before the call, so
//      `fuse_boxing_alloc` mints `NewWithVtable` with a real vtable.
//   3. `if ta == &CLS { concrete(..) }` — the arm body a `FunctionPath`
//      call, i.e. inlinable, rather than an indirect one.
//   4. an `_immutable_fields_<Struct>` marker const, so the payload read
//      can fold to a pure getfield.
//
// `TypeOnlyHeader` matches RPython's root `OBJECT`, whose only data field is
// `typeptr`. `ObjectHeader` additionally represents an object model with a
// per-instance class word. Both allocation shapes must fuse.
//
// These `majit_gc::GcType` impls declare the gc kind.

#[repr(C)]
pub struct ClassObject {
    pub name: &'static str,
    pub kind: u8,
}

#[repr(C)]
pub struct ObjectHeader {
    pub ob_type: *const ClassObject,
    pub w_class: *const ClassObject,
}

/// The one-word header: `ob_type` and nothing else.
///
/// `fuse_boxing_alloc`'s substitution asks whether the per-instance class
/// word agrees with `ob_type`; where the header declares no such word the
/// question has no subject, and `model.rs`'s `header_declares_no_class_word`
/// arm admits the cluster on the layout alone. That arm is unreachable from
/// any fixture whose header declares the field, so it gets its own.
#[repr(C)]
pub struct TypeOnlyHeader {
    pub ob_type: *const ClassObject,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_IntObject {
    pub ob_header: ObjectHeader,
    pub intval: i64,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_TypeOnlyIntObject {
    pub ob_header: TypeOnlyHeader,
    pub intval: i64,
}

// Separate modules: Charon names every trait impl `::<Impl>::type_id`, so
// three impls in one module are one path and `source_for_name_path` misses.
mod class_object_gc {
    impl majit_gc::GcType for super::ClassObject {
        fn type_id() -> u32 {
            1
        }
        const SIZE: usize = std::mem::size_of::<super::ClassObject>();
    }
}

mod object_header_gc {
    impl majit_gc::GcType for super::ObjectHeader {
        fn type_id() -> u32 {
            2
        }
        const SIZE: usize = std::mem::size_of::<super::ObjectHeader>();
    }
}

mod type_only_header_gc {
    impl majit_gc::GcType for super::TypeOnlyHeader {
        fn type_id() -> u32 {
            3
        }
        const SIZE: usize = std::mem::size_of::<super::TypeOnlyHeader>();
    }
}

pub static INT_CLASS: ClassObject = ClassObject {
    name: "int",
    kind: 1,
};
pub static DOUBLE_CLASS: ClassObject = ClassObject {
    name: "double",
    kind: 2,
};

/// The allocation entry point. `is_malloc_typed` keys on the trailing path
/// segments `lltype::malloc_typed`, and `fuse_boxing_alloc` requires the
/// single by-value argument — an alloc-then-init spelling matches nothing
/// and degrades in silence.
pub mod lltype {
    #[inline(never)]
    pub fn malloc_typed<T>(value: T) -> *mut T {
        Box::into_raw(Box::new(value))
    }
}

/// Minimal stand-in for the class-instantiation lookup whose result the
/// class word is stored from.
///
/// `model.rs`'s `get_instantiate_arg_addr` recognizes this helper by its
/// single-argument runtime contract.
pub mod runtime_object {
    pub mod object_model {
        use crate::ClassObject;

        #[inline(never)]
        pub fn get_instantiate(tp: &ClassObject) -> *const ClassObject {
            tp
        }
    }
}

/// Premise 1: the header read.
#[inline(never)]
pub fn w_object_type(w: *mut ObjectHeader) -> *const ClassObject {
    unsafe { (*w).ob_type }
}

/// Premise 2: the boxing cluster — header store, payload store, then one
/// by-value `malloc_typed`.
#[inline(never)]
pub fn w_new_int(x: i64) -> *mut W_IntObject {
    lltype::malloc_typed(W_IntObject {
        ob_header: ObjectHeader {
            ob_type: &INT_CLASS,
            w_class: runtime_object::object_model::get_instantiate(&INT_CLASS),
        },
        intval: x,
    })
}

/// Premise 2, second header shape: the same cluster over a header that
/// declares no class word, so the fuse has to admit it on the layout alone.
/// The `get_instantiate` call has no subject here and is absent.
#[inline(never)]
pub fn w_new_type_only_int(x: i64) -> *mut W_TypeOnlyIntObject {
    lltype::malloc_typed(W_TypeOnlyIntObject {
        ob_header: TypeOnlyHeader {
            ob_type: &INT_CLASS,
        },
        intval: x,
    })
}

#[inline(never)]
fn w_int_add(a: *mut W_IntObject, b: *mut W_IntObject) -> i64 {
    unsafe { (*a).intval + (*b).intval }
}

/// Premise 3: the narrowing chain. `descroperation.py` `binop_impl`
/// (`type(w_obj1) is type(w_obj2)`, then the per-class shortcut)
/// transliterated — the shape that lowers each arm to a direct call.
#[inline(never)]
pub fn w_number_add(a: *mut ObjectHeader, b: *mut ObjectHeader) -> i64 {
    let ta = unsafe { (*a).ob_type };
    let tb = unsafe { (*b).ob_type };
    if ta == tb && ta == (&INT_CLASS as *const ClassObject) {
        return w_int_add(a as *mut W_IntObject, b as *mut W_IntObject);
    }
    0
}

/// Premise 4: the immutability marker `harvest_immutable_fields_from_llbcs`
/// reads (`front/llbc_hints.rs:148` `_immutable_fields_` prefix). Written by
/// hand rather than through `#[jit_immutable_fields]` so the corpus keeps
/// its zero-dependency manifest.
#[allow(non_upper_case_globals)]
pub const _immutable_fields_W_IntObject: &str = "intval";

// A host-registered callback table.

/// The callback a host installs at run time. A bare `fn` pointer, so the set
/// of addresses that can reach a call through it is not recoverable from this
/// artifact — the shape used by host-settable callback hooks.
pub type HostCallback = fn(i64) -> i64;

pub struct HostRegistry {
    pub slot: HostCallback,
    pub maybe_slot: Option<HostCallback>,
}

/// Call through the registered callback. `front::mir` lowers this to
/// `OpKind::IndirectCall { graphs: None }` — `indirect_call` with an
/// unknown PBC family, which `guess_call_kind` answers `residual` for
/// (`call.py`, `jtransform.py rewrite_op_indirect_call`). The `__dyn_call`
/// placeholder it used to reach is an unregistered synthetic path with no
/// continuation.
#[inline(never)]
pub fn host_registry_dispatch(reg: &HostRegistry, x: i64) -> i64 {
    (reg.slot)(x)
}

/// The one-hop `Option<fn-ptr>` spelling of the same shape.
#[inline(never)]
pub fn host_registry_dispatch_optional(reg: &HostRegistry, x: i64) -> i64 {
    match reg.maybe_slot {
        Some(f) => f(x),
        None => 0,
    }
}

// 9. An array whose element is a by-value aggregate.
//
// `v[i]` on a `Vec<T>` with a scalar subscript resolves to
// `<Vec<T> as Index<usize>>::index`, which `front::mir`'s
// `is_vec_index_call` (`vec_index_regular_leaf`) intercepts: the call site
// lowers eagerly to an `ArrayRead` and records an `IndexElemAlias` for the
// paired write and for the projections off the element, leaving no residual
// call. That arm gates on the *index* type and never on the *element* type —
// `item_ty` is whatever `tyref_deref_value_type(call.dest.ty)` answers — so
// the element bank it produces for a multi-word by-value ADT is decided by
// `tyref_to_value_type`'s fallback rather than by a deliberate arm.
//
// `SlotValue` carries an integer variant, a raw-pointer variant and a
// two-field variant. None of the three leaves a niche free, so the enum is a
// genuine tag-plus-payload aggregate several words wide and not a wrapper the
// front end can collapse to its inner bank (`tyref_transparent_inner_value_type`,
// `tyref_is_fieldless_enum_free`).
//
// The fallback bank is `ValueType::Ref(None)`, the `Vec` leg ships no
// `array_type_id`, and `arraydescrof_concrete` therefore hands the read a
// one-word `item_size`. The index spelling is UNSOUND over this element type —
// see `an_aggregate_element_array_read_is_given_a_one_word_item_size` in
// `majit-translate/tests/test_mir_frontend.rs` for the full chain. These four
// functions are what witnesses it.
pub enum SlotValue {
    Int(i64),
    Object(*const ObjectHeader),
    Pair { lhs: i64, rhs: i64 },
}

/// The treatment: read one element by scalar index and match it. The
/// discriminant read and the per-variant payload reads all land on the
/// `ArrayRead` result, so this is the shape that says whether the alias
/// projections carry an aggregate element. They do not — this is the spelling
/// whose descr strides by one word, not a spelling to copy.
#[inline(never)]
pub fn aggregate_slot_index(v: &Vec<SlotValue>, i: usize) -> i64 {
    match &v[i] {
        SlotValue::Int(x) => *x,
        SlotValue::Object(h) => unsafe { (*(**h).ob_type).kind as i64 },
        SlotValue::Pair { lhs, rhs } => *lhs + *rhs,
    }
}

/// The control: the same body over `<[T]>::get`, reached by deref coercion
/// from the `Vec`. Its `Option<&SlotValue>` destination is what
/// `recognize_slice_get_site` accepts or declines, so the call either becomes
/// `front::slice_get`'s bounds-checked diamond or stays residual — either way
/// it is a different lowering from the index arm. If it is not, the pair
/// discriminates nothing and neither function grades the index arm. Given the
/// index arm's descr, the residual outcome is also the *safe* one: real Rust
/// keeps computing the element address at the real stride.
#[inline(never)]
pub fn aggregate_slot_get(v: &Vec<SlotValue>, i: usize) -> i64 {
    match v.get(i) {
        Some(SlotValue::Int(x)) => *x,
        Some(SlotValue::Object(h)) => unsafe { (*(**h).ob_type).kind as i64 },
        Some(SlotValue::Pair { lhs, rhs }) => *lhs + *rhs,
        None => 0,
    }
}

/// The same two spellings over an element bank the index arm is already
/// known to serve, so a difference between the two pairs is attributable to
/// the element kind and not to the fixture.
#[inline(never)]
pub fn scalar_slot_index(v: &Vec<i64>, i: usize) -> i64 {
    v[i]
}

#[inline(never)]
pub fn scalar_slot_get(v: &Vec<i64>, i: usize) -> i64 {
    match v.get(i) {
        Some(x) => *x,
        None => 0,
    }
}

/// An `Option<char>` joined with a `char` literal default, then compared.
/// Both links into the join carry a `char`, so both are int-kind.
#[inline(never)]
pub fn char_unwrap_or_join(align: Option<char>) -> i64 {
    let a = align.unwrap_or('>');
    if a == '^' {
        1
    } else {
        0
    }
}

/// The shape `bitflags!` expands to: a `repr(transparent)` public flag type
/// around a `repr(transparent)` inner type around the integer, each with a
/// `const fn` constructor and a `const fn` accessor, and the flag an
/// associated const built through the constructor.  The inner type's methods
/// are spelled apart from the outer ones so every local body keeps a unique
/// name path.
#[repr(transparent)]
pub struct InnerCodeFlags(u16);

impl InnerCodeFlags {
    #[inline]
    pub const fn inner_from_bits_retain(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn inner_bits(&self) -> u16 {
        self.0
    }
}

#[repr(transparent)]
pub struct CodeFlags(InnerCodeFlags);

impl CodeFlags {
    pub const FLAT: Self = Self::from_bits_retain(0x100);

    #[inline]
    pub const fn from_bits_retain(bits: u16) -> Self {
        Self(InnerCodeFlags::inner_from_bits_retain(bits))
    }

    #[inline]
    pub const fn bits(&self) -> u16 {
        self.0.inner_bits()
    }
}

/// A flag const read through `.bits()` inside an integer expression.
#[inline(never)]
pub fn code_flags_bits_or(nargs: usize) -> usize {
    nargs | CodeFlags::FLAT.bits() as usize
}

/// A `char` element read at a constant index and switched on. `char` is a
/// 4-byte unsigned item, so the index arm lowers it to an int-banked
/// `ArrayRead` and the match switches on that int.
#[inline(never)]
pub fn char_slot_index(v: &Vec<char>) -> i64 {
    if v.len() >= 2 && matches!(v[1], '<' | '>' | '=' | '^') {
        1
    } else {
        0
    }
}

// 10. Register bank of a borrowed primitive, by container.
//
// Three shapes one peel decision has to answer together.  The payload's own
// type cannot separate them: an enum variant's `&u8`, another enum's `&usize`
// and a struct's `&u8` field all serialize as the same shared-borrow node, so
// the decision reads the container rather than the payload.

/// `<[T]>::get` hands the element back by reference, so the `?` payload is a
/// `&u8` and the switch operand is the byte behind it.  Banking that borrow as
/// a GC ref reaches `codewriter/flatten.rs` as a ref-kinded `SwitchInt`
/// (`switch exitswitch must be int`).
#[inline(never)]
pub fn slice_get_tag_dispatch(code: &[u8], pc: usize) -> Option<i64> {
    let tag = *code.get(pc)?;
    Some(match tag {
        0 => 10,
        1 => 20,
        _ => 30,
    })
}

/// `RangeBounds::start_bound` yields `Bound<&usize>`, an enum no `?`
/// desugaring produces.  `Unbounded` supplies the merged value by value while
/// the other arms supply it through a borrow, so typing the payload as a
/// pointer leaves sibling arms in different banks and the merge reaches
/// `encode_regorconst_source` as a cross-bank move.
#[inline(never)]
pub fn range_start_index(range: core::ops::Range<usize>) -> usize {
    use core::ops::{Bound, RangeBounds};
    match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    }
}

/// A `&u8` a struct declares and stores is a pointer the program can compare,
/// so it keeps its own bank.  Peeling every borrowed primitive — rather than
/// only an enum variant's — answers this one with the byte's value instead.
pub struct BorrowedByte<'a> {
    pub p: &'a u8,
}

#[inline(never)]
pub fn borrowed_byte_fields_alias(a: &BorrowedByte<'_>, b: &BorrowedByte<'_>) -> bool {
    core::ptr::eq(a.p, b.p)
}

/// `u32::from(u16)` — a numeric `From` whose source and destination are both
/// integer-kinded, so the conversion is a no-op once both sit in the `Int`
/// register bank. The pair, so the fixture also pins that the *argument*
/// survives: an alias that dropped it would read as "no call" just as well.
#[inline(never)]
pub fn widening_int_from(narrow: u16) -> u32 {
    u32::from(narrow) + 1
}

/// The same shape across the float boundary, which must NOT alias: this one
/// is a real conversion and the fixture is what keeps the integer rule from
/// being widened into it.
#[inline(never)]
pub fn widening_float_from(narrow: i32) -> f64 {
    f64::from(narrow) + 1.0
}

/// `mem::replace` of a field. The old field is the result; the field then
/// holds `new`.
pub struct ReplaceField {
    pub slot: i64,
}

#[inline(never)]
pub fn replace_field(cell: &mut ReplaceField, new: i64) -> i64 {
    std::mem::replace(&mut cell.slot, new)
}

/// `mem::replace` of a slice element. Same read-then-store as the field.
#[inline(never)]
pub fn replace_elem(items: &mut [i64], index: usize, new: i64) -> i64 {
    std::mem::replace(&mut items[index], new)
}

/// `mem::replace(&mut *r, new)` where `r` reborrows a local, then `*r`.
/// The read must observe `new`, not the value bound into `r`.
#[inline(never)]
pub fn replace_reborrow_then_read(mut slot: i64, new: i64) -> i64 {
    let r = &mut slot;
    std::mem::replace(&mut *r, new);
    *r
}

/// Two machine words, inline. `mem::replace` / `swap` / `take` of the whole
/// value is one read and one store of each field, not a residual copy.
#[derive(Default)]
pub struct TwoWords {
    pub lo: i64,
    pub hi: i64,
}

#[inline(never)]
pub fn replace_two_words(slot: &mut TwoWords, new: TwoWords) -> TwoWords {
    std::mem::replace(slot, new)
}

#[inline(never)]
pub fn swap_two_words(a: &mut TwoWords, b: &mut TwoWords) {
    std::mem::swap(a, b);
}

#[inline(never)]
pub fn take_two_words(slot: &mut TwoWords) -> TwoWords {
    std::mem::take(slot)
}

/// Hand-written `Default` whose fields are not zero. `mem::take` calls
/// this impl and stores the fields it returns. The impl lives in its own
/// module so its name path is not `charon_corpus::<Impl>::default`, which
/// the derived `TwoWords` impl already uses.
pub mod odd_default {
    pub struct OddDefault {
        pub lo: i64,
        pub hi: i64,
    }

    impl Default for OddDefault {
        fn default() -> Self {
            Self { lo: 7, hi: 9 }
        }
    }
}

#[inline(never)]
pub fn take_odd_default(slot: &mut odd_default::OddDefault) -> odd_default::OddDefault {
    std::mem::take(slot)
}

/// 16-byte-shaped payload enum: an `i64` plus an `i32`, the same inline
/// triple shape a `Union` scalar uses. A whole-value exchange copies each
/// field the layout records.
#[derive(Clone, Copy)]
pub enum WordUnion {
    Unit(i32),
    Int(i64, i32),
}

#[inline(never)]
pub fn replace_word_union(slot: &mut WordUnion, new: WordUnion) -> WordUnion {
    std::mem::replace(slot, new)
}
/// A variant field wider than one word. A whole-value move cannot copy it
/// as a field span, so `move_plan` declines.
pub enum WidePayload {
    Big(u128),
}

#[inline(never)]
pub fn replace_wide_payload(slot: &mut WidePayload, new: WidePayload) -> WidePayload {
    std::mem::replace(slot, new)
}

/// An `i64` payload and a `Box` payload share one word. `*held = value`
/// writes the integer field; it is not a copy of the reference variant.
pub enum HeldUnion {
    Int(i64),
    Ref(Box<i64>),
}

#[inline(never)]
pub fn store_held_int(cell: &mut HeldUnion, value: i64) {
    let HeldUnion::Int(held) = cell else {
        return;
    };
    *held = value;
}

/// Whole-value move of the same enum. The live variant's fields are the
/// ones the copy reads.
#[inline(never)]
pub fn replace_held_union(slot: &mut HeldUnion, new: HeldUnion) -> HeldUnion {
    std::mem::replace(slot, new)
}

/// The enum is behind `Box`, the same inline `Dynamic` a frame stores in
/// `operand_refs`. `*slot` is that enum; the exchange is the variant stores.
#[inline(never)]
pub fn replace_boxed_held(slot: &mut Box<HeldUnion>, new: HeldUnion) -> HeldUnion {
    std::mem::replace(slot.as_mut(), new)
}

/// `operand_mut` is its own function. The caller's `mem::replace` argument
/// is that call's `&mut HeldUnion`, not a borrow assigned in the caller.
#[inline(never)]
pub fn held_as_mut(slot: &mut Box<HeldUnion>) -> &mut HeldUnion {
    slot.as_mut()
}

#[inline(never)]
pub fn replace_boxed_held_call(slot: &mut Box<HeldUnion>, new: HeldUnion) -> HeldUnion {
    std::mem::replace(held_as_mut(slot), new)
}

/// `truncate_cells`: `mem::replace(operand_mut(&mut self.operand_refs[slot]), unit)`.
/// The mutable enum is an indexed `Box`, then `as_mut`.
#[inline(never)]
pub fn replace_indexed_box(
    slots: &mut Vec<Box<HeldUnion>>,
    index: usize,
    new: HeldUnion,
) -> HeldUnion {
    std::mem::replace(slots[index].as_mut(), new)
}

#[inline(never)]
pub fn replace_indexed_box_call(
    slots: &mut Vec<Box<HeldUnion>>,
    index: usize,
    new: HeldUnion,
) -> HeldUnion {
    std::mem::replace(held_as_mut(&mut slots[index]), new)
}

/// `Dynamic` is `#[repr(transparent)]` over the payload enum. The exchange
/// is still that enum's variant stores.
#[repr(transparent)]
pub struct DynLike(pub HeldUnion);

#[inline(never)]
pub fn replace_boxed_dynlike(slot: &mut Box<DynLike>, new: DynLike) -> DynLike {
    std::mem::replace(slot.as_mut(), new)
}

/// The enum sits inline at field 0. Assigning a whole value must move the
/// live variant's fields into that slot. A store of the temporary's address
/// overwrites the discriminant byte.
pub struct HeldCell(pub HeldUnion);

#[inline(never)]
pub fn store_held_cell(slot: &mut HeldCell, value: i64) {
    slot.0 = HeldUnion::Int(value);
}

/// Reads two items of a borrowed slice.
#[inline(never)]
pub fn sum_two_items(items: &[i64]) -> i64 {
    items[0] + items[1]
}

/// An array literal borrowed as a slice argument. The borrow reaches the
/// call through a copied reference, and the array is the slice's storage.
#[inline(never)]
pub fn sum_of_array_literal(x: i64) -> i64 {
    let borrowed = &[x, x + 1];
    let moved = borrowed;
    sum_two_items(moved)
}

/// Reverses a slice of GCREF items, then reads its first item.
#[inline(never)]
pub fn reverse_then_first_ref(items: &mut [GCREF]) -> GCREF {
    items.reverse();
    items[0]
}

/// A range subslice of a pointer array local, passed on as a slice.
#[inline(never)]
pub fn first_of_array_prefix(a: GCREF, b: GCREF, n: usize) -> GCREF {
    let buf = [a, b];
    first_ref(&buf[..n])
}

#[inline(never)]
pub fn first_ref(items: &[GCREF]) -> GCREF {
    items[0]
}

/// Appends a slice of integer items to a `Vec`.
#[inline(never)]
pub fn extend_vec_from_slice(v: &mut Vec<usize>, items: &[usize]) {
    v.extend_from_slice(items);
}

/// Reads an item of a slice carried in a tuple, or null past its end.
#[inline(never)]
pub fn item_of_tupled_slice(t: (&[GCREF], usize)) -> GCREF {
    let items = t.0;
    if t.1 < items.len() {
        items[t.1]
    } else {
        core::ptr::null_mut()
    }
}

/// Builds the tuple [`item_of_tupled_slice`] reads.
#[inline(never)]
pub fn tuple_a_slice(items: &[GCREF], i: usize) -> GCREF {
    item_of_tupled_slice((items, i))
}

/// A closure that captures a slice by reference and returns it.
#[inline(never)]
pub fn slice_through_a_closure(items: &[GCREF]) -> GCREF {
    let get = || -> &[GCREF] { items };
    first_ref(get())
}

/// A closure indexing a slice it captures by reference.
#[inline(never)]
pub fn index_through_a_closure(items: &[GCREF], i: usize) -> GCREF {
    let get = |j: usize| items[j];
    get(i)
}

/// The item an `Option<&T>` names, or null.
#[inline(never)]
pub fn item_or_null(item: Option<&GCREF>) -> GCREF {
    match item {
        Some(p) => *p,
        None => core::ptr::null_mut(),
    }
}

/// `Option<&T>` of a pointer item passed on as a value.
#[inline(never)]
pub fn get_item_or_null(items: &[GCREF], i: usize) -> GCREF {
    item_or_null(items.get(i))
}

/// Copies a slice of pointer items into another. The lengths are equal;
/// that check is an assertion.
#[inline(never)]
pub fn copy_slice_from_slice(dst: &mut [GCREF], src: &[GCREF]) {
    dst.copy_from_slice(src);
}

/// `vec![item; count]` of one-word integer items.
#[inline(never)]
pub fn repeat_vec_i64(item: i64, count: usize) -> Vec<i64> {
    vec![item; count]
}

/// `vec![item; count]` of GCREF items.
#[inline(never)]
pub fn repeat_vec_ptr(item: GCREF, count: usize) -> Vec<GCREF> {
    vec![item; count]
}

/// The data pointer of a shared slice of pointer items.
#[inline(never)]
pub fn ptr_of_slice(items: &[GCREF]) -> *const GCREF {
    items.as_ptr()
}

/// The data pointer of a mutable slice of pointer items.
#[inline(never)]
pub fn mut_ptr_of_slice(items: &mut [GCREF]) -> *mut GCREF {
    items.as_mut_ptr()
}

/// A root-bracket guard closed by an explicit `drop` before the function ends.
pub mod gc_roots {
    pub struct RootScope {
        pub base: usize,
    }

    impl Drop for RootScope {
        fn drop(&mut self) {}
    }

    #[inline(never)]
    pub fn push_roots() -> RootScope {
        RootScope { base: 0 }
    }
}

/// Closes the bracket with `drop(roots)` rather than a scope-end drop.
#[inline(never)]
pub fn mem_drop_root_scope() -> usize {
    let roots = gc_roots::push_roots();
    let base = roots.base;
    drop(roots);
    base
}

/// `drop` of an integer, which has no destructor.
#[inline(never)]
pub fn mem_drop_i64(x: i64) {
    drop(x);
}

/// `drop` of a `usize`, which has no destructor.
#[inline(never)]
pub fn mem_drop_usize(x: usize) {
    drop(x);
}

/// `s.get(1..).unwrap_or(&[])` is the checked spelling of a tail slice.
/// The length of that tail is the length word of the resulting pair.
#[inline(never)]
pub fn tail_len(s: &[i64]) -> usize {
    s.get(1..).unwrap_or(&[]).len()
}
