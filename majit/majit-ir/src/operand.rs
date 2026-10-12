//! `Operand` — the operand-union carrier for `Op.args` /
//! `Op.fail_args` (#9).
//!
//! `resoperation.py` `N_aryOp._args` stores operands as the
//! `AbstractValue` objects themselves — a result op, an input arg, or a
//! constant — with no integer-position indirection. `Operand` is the Rust
//! shape of that: a strong-ref union carrying the producer directly, so
//! operand identity is `Rc::ptr_eq` and forwarding reads straight off the
//! carried producer's `_forwarded` slot — with no `find_producer_op`
//! position→producer registry, no `Op::box_cache` memoization, and no
//! position-only ref fabrication.
//!
//! Strong `Rc`, as in [`Forwarded`](crate::forwarding::Forwarded):
//! operands must keep their producers alive. The trace already holds the 1st
//! strong ref in `Trace.ops: Vec<OpRc>` (#103); an operand `Rc<Op>` is a 2nd
//! strong ref on the acyclic SSA use-before-def DAG (operands reference
//! predecessors only), so no `Rc` cycle can form.
//!
//! This module is the #9 foundation: `Op.args` / `Op.fail_args` carry
//! `Operand` directly, and the `from_bound_*` constructors bind each producer
//! identity at construction.

use crate::forwarding::{
    Forwarded, ForwardingHost, IntBoundBorrow, IntBoundBorrowMut, PackedForwarded, PtrInfoBorrow,
    PtrInfoBorrowMut, classify_packed_forwarded,
};
use crate::op_info::OpInfo;
use crate::ptr_info::PtrInfo;
use crate::resoperation::{OpRc, OpRef};
use crate::value::{Const, GcRef, InputArgRc, Type, Value};
use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

/// Allocation-free identity source for the small-`ConstInt` arm below.
///
/// RPython's opencoder writes small integers directly into the byte stream and
/// only mints the `ConstInt` object when an iterator decodes the operation. The
/// legacy structured recorder has to expose an `Operand` immediately, but it
/// need not ask the process allocator for the overwhelmingly common small-int
/// object. The high word is an object-identity token; the low word is the
/// sign-preserving `i32` payload. A clone keeps the token, while a fresh mint
/// gets a new one, preserving `AbstractValue` identity and hashing.
static NEXT_SMALL_INT_ID: AtomicU32 = AtomicU32::new(1);

#[inline]
pub(crate) fn fresh_small_int(value: i64) -> Option<u64> {
    let value = i32::try_from(value).ok()?;
    let id = NEXT_SMALL_INT_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .unwrap_or_else(|_| panic!("small ConstInt identity space exhausted"));
    Some((u64::from(id) << 32) | u64::from(value as u32))
}

#[inline]
pub(crate) fn small_int_value(encoded: u64) -> i64 {
    (encoded as u32 as i32) as i64
}

/// What a wide constant stores. `Ref` is a [`crate::const_ptr_table`]
/// index (`history.py` `ConstPtr`), not the referent address: the
/// collector writes the table, and [`wide_value`] reads it.
#[derive(Clone, Copy)]
enum WideStored {
    Int(i64),
    Float(f64),
    Ref(u32),
    Void,
}

/// Allocation-free identity for `ConstFloat` / `ConstPtr` / out-of-i32
/// `ConstInt`. The token is the slab index; a clone keeps it, a fresh mint
/// gets a new one — the same `is` identity `SmallInt` keeps, without an
/// `Rc<Cell<Value>>` (32 B). Not a value intern: two mints of the same
/// bits are unequal. A `Ref` payload is a table index, so a minor does
/// not stale the slot.
const WIDE_CHUNK: usize = 512;
const WIDE_MAX_CHUNKS: usize = 8192;

static NEXT_WIDE_ID: AtomicU32 = AtomicU32::new(1);
static WIDE_CHUNKS: [AtomicPtr<Cell<WideStored>>; WIDE_MAX_CHUNKS] =
    [const { AtomicPtr::new(ptr::null_mut()) }; WIDE_MAX_CHUNKS];

fn init_wide_chunk(chunk_i: usize) -> *mut Cell<WideStored> {
    let boxed: Box<[Cell<WideStored>]> = (0..WIDE_CHUNK)
        .map(|_| Cell::new(WideStored::Void))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let raw = Box::into_raw(boxed) as *mut Cell<WideStored>;
    match WIDE_CHUNKS[chunk_i].compare_exchange(
        ptr::null_mut(),
        raw,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => raw,
        Err(existing) => {
            unsafe {
                drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
                    raw, WIDE_CHUNK,
                )));
            }
            existing
        }
    }
}

fn wide_slot(id: u32) -> &'static Cell<WideStored> {
    let idx = id as usize;
    let chunk_i = idx / WIDE_CHUNK;
    assert!(
        chunk_i < WIDE_MAX_CHUNKS,
        "wide Const identity space exhausted"
    );
    let off = idx % WIDE_CHUNK;
    let p = WIDE_CHUNKS[chunk_i].load(Ordering::Acquire);
    let p = if p.is_null() {
        init_wide_chunk(chunk_i)
    } else {
        p
    };
    unsafe { &*p.add(off) }
}

fn store_wide(value: Value) -> WideStored {
    match value {
        Value::Int(v) => WideStored::Int(v),
        Value::Float(v) => WideStored::Float(v),
        Value::Ref(r) => WideStored::Ref(crate::const_ptr_table::intern(r)),
        Value::Void => WideStored::Void,
    }
}

pub(crate) fn fresh_wide(value: Value) -> u64 {
    let id = NEXT_WIDE_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .unwrap_or_else(|_| panic!("wide Const identity space exhausted"));
    wide_slot(id).set(store_wide(value));
    u64::from(id)
}

pub(crate) fn wide_value(id: u64) -> Value {
    match wide_slot(id as u32).get() {
        WideStored::Int(v) => Value::Int(v),
        WideStored::Float(v) => Value::Float(v),
        WideStored::Ref(index) => Value::Ref(crate::const_ptr_table::resolve(index)),
        WideStored::Void => Value::Void,
    }
}

/// Table index stored for a wide `ConstPtr`, if this slot is a ref.
pub(crate) fn wide_ref_index(id: u64) -> Option<u32> {
    match wide_slot(id as u32).get() {
        WideStored::Ref(index) => Some(index),
        _ => None,
    }
}

/// An operand stored in `Op.args` / `Op.fail_args`.
///
/// Mirror of `OpRef`'s four logical cases, but carrying the producer by
/// strong `Rc` instead of a flat position: `Op` ⇆ `OpRef::*Op`, `InputArg`
/// ⇆ `OpRef::InputArg*`, `Const` ⇆ the inline `OpRef::Const*`, and `None` ⇆
/// `OpRef::None` (an absent `fail_args` slot).
///
/// Packed to 8 B so four inline args are 32 B and `Rc<Op>` leaves the
/// 128-byte class; a 4-/6-failarg list is 32/48 B instead of 64/96 B.
#[repr(transparent)]
pub struct Operand {
    packed: u64,
}

const OP_TAG: u64 = 0b111;
const OP_OP: u64 = 0;
const OP_INPUTARG: u64 = 1;
const OP_CONST: u64 = 2;
const OP_SMALLINT: u64 = 3;
const OP_SMALLWIDE: u64 = 4;
const OP_NULLREF: u64 = 5;

/// Decoded view. Pointer variants borrow the operand's own reference without
/// taking a count; the view must not outlive the operand it was read from.
enum Opnd {
    None,
    Op(ManuallyDrop<OpRc>),
    InputArg(ManuallyDrop<InputArgRc>),
    SmallInt(u64),
    SmallWide(u64),
    NullRef,
    Const(ManuallyDrop<Rc<Cell<Value>>>),
}

impl Operand {
    /// Absent slot — the mirror of `OpRef::None`.
    #[allow(non_upper_case_globals)]
    pub const None: Operand = Operand { packed: 0 };

    /// `opencoder.py Trace._cached_const_ptr` null.
    #[allow(non_upper_case_globals)]
    pub const NullRef: Operand = Operand { packed: OP_NULLREF };

    #[allow(non_snake_case)]
    pub fn Op(op: OpRc) -> Operand {
        let p = OpRc::into_raw(op) as u64;
        debug_assert_eq!(p & OP_TAG, 0);
        Operand { packed: p }
    }

    #[allow(non_snake_case)]
    pub fn InputArg(ia: InputArgRc) -> Operand {
        let p = InputArgRc::into_raw(ia) as u64;
        debug_assert_eq!(p & OP_TAG, 0);
        Operand {
            packed: p | OP_INPUTARG,
        }
    }

    #[allow(non_snake_case)]
    pub fn SmallInt(enc: u64) -> Operand {
        let id = enc >> 32;
        let val = enc as u32 as u64;
        debug_assert!(id < (1 << 29), "SmallInt identity exceeds 29 bits");
        Operand {
            packed: OP_SMALLINT | (val << 3) | (id << 35),
        }
    }

    #[allow(non_snake_case)]
    pub fn SmallWide(id: u64) -> Operand {
        debug_assert!(id < (1 << 61));
        Operand {
            packed: OP_SMALLWIDE | (id << 3),
        }
    }

    #[allow(non_snake_case)]
    pub fn Const(cell: Rc<Cell<Value>>) -> Operand {
        let p = Rc::into_raw(cell) as u64;
        debug_assert_eq!(p & OP_TAG, 0);
        Operand {
            packed: p | OP_CONST,
        }
    }

    fn view(&self) -> Opnd {
        if self.packed == 0 {
            return Opnd::None;
        }
        match self.packed & OP_TAG {
            OP_OP => Opnd::Op(ManuallyDrop::new(unsafe {
                OpRc::from_raw(self.packed as *const crate::resoperation::Op)
            })),
            OP_INPUTARG => Opnd::InputArg(ManuallyDrop::new(unsafe {
                InputArgRc::from_raw((self.packed & !OP_TAG) as *const crate::value::InputArg)
            })),
            OP_CONST => Opnd::Const(ManuallyDrop::new(unsafe {
                Rc::from_raw((self.packed & !OP_TAG) as *const Cell<Value>)
            })),
            OP_SMALLINT => {
                let val = (self.packed >> 3) as u32 as u64;
                let id = self.packed >> 35;
                Opnd::SmallInt((id << 32) | val)
            }
            OP_SMALLWIDE => Opnd::SmallWide(self.packed >> 3),
            OP_NULLREF => Opnd::NullRef,
            _ => Opnd::None,
        }
    }

    pub fn is_small_int(&self) -> bool {
        self.packed != 0 && self.packed & OP_TAG == OP_SMALLINT
    }

    pub fn is_small_wide(&self) -> bool {
        self.packed != 0 && self.packed & OP_TAG == OP_SMALLWIDE
    }

    pub fn is_null_ref(&self) -> bool {
        self.packed == OP_NULLREF
    }

    /// Steal the packed word. The caller owns the refs.
    pub(crate) fn into_packed(self) -> u64 {
        let packed = self.packed;
        std::mem::forget(self);
        packed
    }

    /// Clone an operand from a packed word the caller still owns.
    #[inline]
    pub(crate) fn clone_from_packed(packed: u64) -> Operand {
        let view = Operand { packed };
        let out = view.clone();
        std::mem::forget(view);
        out
    }

    /// Drop the refs in a packed word the caller is releasing.
    pub(crate) fn drop_packed(packed: u64) {
        drop(Operand { packed });
    }
}

impl Clone for Operand {
    #[inline]
    fn clone(&self) -> Self {
        if self.packed == 0 || self.packed == OP_NULLREF {
            return Operand {
                packed: self.packed,
            };
        }
        match self.packed & OP_TAG {
            OP_OP => unsafe {
                OpRc::increment_strong_count(self.packed as *const crate::resoperation::Op);
            },
            OP_INPUTARG => unsafe {
                InputArgRc::increment_strong_count(
                    (self.packed & !OP_TAG) as *const crate::value::InputArg,
                );
            },
            OP_CONST => unsafe {
                Rc::<Cell<Value>>::increment_strong_count(
                    (self.packed & !OP_TAG) as *const Cell<Value>,
                );
            },
            _ => {}
        }
        Operand {
            packed: self.packed,
        }
    }
}

impl Drop for Operand {
    #[inline]
    fn drop(&mut self) {
        if self.packed == 0 || self.packed == OP_NULLREF {
            return;
        }
        match self.packed & OP_TAG {
            OP_OP => {
                drop(unsafe { OpRc::from_raw(self.packed as *const crate::resoperation::Op) });
            }
            OP_INPUTARG => {
                drop(unsafe {
                    InputArgRc::from_raw((self.packed & !OP_TAG) as *const crate::value::InputArg)
                });
            }
            OP_CONST => {
                drop(unsafe { Rc::from_raw((self.packed & !OP_TAG) as *const Cell<Value>) });
            }
            _ => {}
        }
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.view() {
            Opnd::None => f.write_str("None"),
            Opnd::Op(op) => f.debug_tuple("Op").field(&op.pos().get()).finish(),
            Opnd::InputArg(ia) => f.debug_tuple("InputArg").field(&ia.index).finish(),
            Opnd::SmallInt(enc) => f
                .debug_tuple("SmallInt")
                .field(&small_int_value(enc))
                .finish(),
            Opnd::SmallWide(id) => f.debug_tuple("SmallWide").field(&wide_value(id)).finish(),
            Opnd::NullRef => f.write_str("NullRef"),
            Opnd::Const(c) => f.debug_tuple("Const").field(&c.get()).finish(),
        }
    }
}

impl Operand {
    #[inline]
    fn fresh_const_value(value: Value) -> Operand {
        if value == Value::Ref(GcRef::NULL) {
            return Operand::NullRef;
        }
        if let Value::Int(value) = value
            && let Some(encoded) = fresh_small_int(value)
        {
            return Operand::SmallInt(encoded);
        }
        Operand::SmallWide(fresh_wide(value))
    }

    /// Wrap a bound op as `Operand::Op` (`Rc::clone`, cheap). The successor
    /// (`resoperation.py AbstractResOp`) — no `box_cache` memoization, the `Rc`
    /// itself IS the stable identity.
    pub fn from_bound_op(op: &OpRc) -> Operand {
        Operand::Op(op.clone())
    }

    /// Wrap a bound input arg as `Operand::InputArg` (`Rc::clone`). Successor
    /// (`resoperation.py AbstractInputArg`).
    pub fn from_bound_inputarg(ia: &InputArgRc) -> Operand {
        Operand::InputArg(ia.clone())
    }

    /// A constant operand — mints a fresh const box (`history.py:227`
    /// `ConstInt(value)` object construction; identity starts here and is
    /// shared by every read of the slot).
    pub fn const_(value: Const) -> Operand {
        Self::fresh_const_value(value.to_value())
    }

    /// A constant operand straight from a [`Value`] — the successor to
    /// a fresh `Rc<Cell<Value>>` const identity (`history.py` `ConstInt`).
    pub fn const_from_value(value: Value) -> Operand {
        Self::fresh_const_value(value)
    }

    /// The absent-slot sentinel.
    pub fn none() -> Operand {
        Operand::None
    }

    /// Build an operand from a flat `OpRef`, for the producer-resolution sites
    /// that pick between a bound producer and the absent/const cases off a
    /// position ref. `None` and the three `Const*` variants carry their value
    /// inline (the non-position `OpRef` arms); a
    /// position-only ref (a `*Op` / `InputArg*` with no producer `Rc`) has no
    /// `Operand` representation under the #9 union and panics, the same
    /// #9 invariant tripwire the operand-union relies on. Callers route bound
    /// positions through [`from_bound_op`](Self::from_bound_op) /
    /// [`from_bound_inputarg`](Self::from_bound_inputarg) and reach here only on
    /// `None` / `Const`.
    pub fn from_opref(r: OpRef) -> Operand {
        match r {
            OpRef::None => Operand::None,
            OpRef::ConstInt(v) => Self::fresh_const_value(Value::Int(v)),
            OpRef::ConstFloat(v) => Self::fresh_const_value(Value::Float(v)),
            OpRef::ConstPtr(0) => Operand::NullRef,
            OpRef::ConstPtr(v) => {
                Self::fresh_const_value(Value::Ref(crate::const_ptr_table::resolve(v)))
            }
            _ => panic!(
                "from_opref: position-only ref {r:?} has no producer to bind — \
                 every operand source must carry a bound producer or a const (#9)"
            ),
        }
    }

    /// Bind `r` to a producer-carrying operand whose [`to_opref`](Self::to_opref)
    /// equals `r` — the binding sibling of [`from_opref`](Self::from_opref). A
    /// `None`/`Const` ref sheds inline (identical to `from_opref`); a ResOp /
    /// InputArg position — which `from_opref` cannot represent and panics on —
    /// binds to a freshly-minted synthetic producer (`SameAs*` / `InputArg`)
    /// carrying the same `pos`. The returned `Operand::Op` / `Operand::InputArg`
    /// holds a strong `Rc`, so the synthetic producer stays alive for exactly as
    /// long as the operand is stored — no external root table.
    /// The vector optimizer's guard-strengthening / accumulation stitching uses
    /// this where its producer buffers hold `Op` values (not `OpRc`), so no real
    /// producer `Rc` is reachable to bind — `guard.py` emits fresh boxes,
    /// `renamer.py` carries box objects.
    pub fn bound_from_opref(r: OpRef) -> Operand {
        use crate::resoperation::Op;
        use crate::resoperation::OpCode;
        use crate::value::InputArg;
        if r.is_none() || r.is_constant() {
            return Operand::from_opref(r);
        }
        let ty = r.ty().unwrap_or(Type::Void);
        match r {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                let ia: InputArgRc = InputArgRc::new(InputArg::from_type(ty, r.raw()));
                Operand::from_bound_inputarg(&ia)
            }
            _ => {
                let opcode = match ty {
                    Type::Int => OpCode::SameAsI,
                    Type::Float => OpCode::SameAsF,
                    Type::Ref => OpCode::SameAsR,
                    Type::Void => OpCode::Jump,
                };
                let op: OpRc = OpRc::new(Op::new(opcode, &[]));
                op.pos().set(r);
                Operand::from_bound_op(&op)
            }
        }
    }

    /// Flat-`OpRef` view for the OpRef-keyed side tables, `op.pos`
    /// comparisons, and backend/gc encoding (`forwarding.rs` parity). This
    /// is the PERMANENT handoff boundary where the optimizer's operand
    /// identity converts to the backend's `OpRef` encoding; it is
    /// re-expressed, never retired. An `Op` reads its (post-compaction)
    /// position straight off `op.pos`; a `Const*` maps to the matching inline
    /// `OpRef` (`history.py:227/268/314`).
    pub fn to_opref(&self) -> OpRef {
        match self.view() {
            Opnd::None => OpRef::NONE,
            Opnd::Op(op) => op.pos().get(),
            Opnd::InputArg(ia) => OpRef::input_arg_typed(ia.index, ia.tp.get()),
            Opnd::SmallInt(encoded) => OpRef::const_int(small_int_value(encoded)),
            Opnd::SmallWide(id) => match wide_slot(id as u32).get() {
                WideStored::Int(v) => OpRef::const_int(v),
                WideStored::Float(v) => OpRef::const_float(v),
                WideStored::Ref(index) => OpRef::ConstPtr(index),
                WideStored::Void => OpRef::NONE,
            },
            Opnd::NullRef => OpRef::const_ptr(GcRef::NULL),
            Opnd::Const(cell) => match cell.get() {
                Value::Int(v) => OpRef::const_int(v),
                Value::Float(v) => OpRef::const_float(v),
                Value::Ref(v) => OpRef::const_ptr(v),
                Value::Void => OpRef::NONE,
            },
        }
    }

    /// `resoperation.py AbstractResOpOrInputArg _pos` accessor: the pool index for `Op` /
    /// `InputArg`; `Const` / `None` have no canonical position.
    pub fn position(&self) -> Option<u32> {
        match self.view() {
            Opnd::Op(op) => Some(op.pos().get().raw()),
            Opnd::InputArg(ia) => Some(ia.index),
            _ => None,
        }
    }

    /// The operand's `Type` (`Int` / `Float` / `Ref` / `Void`).
    pub fn type_(&self) -> Type {
        match self.view() {
            Opnd::Op(op) => op.pos().get().ty().unwrap_or(op.type_),
            Opnd::InputArg(ia) => ia.tp.get(),
            Opnd::SmallInt(_) => Type::Int,
            Opnd::SmallWide(id) => wide_value(id).get_type(),
            Opnd::NullRef => Type::Ref,
            Opnd::Const(cell) => cell.get().get_type(),
            Opnd::None => Type::Void,
        }
    }

    /// The inline constant value (`history.py` `Const.getint` family),
    /// `None` for non-`Const`.
    pub fn const_value(&self) -> Option<Value> {
        match self.view() {
            Opnd::SmallInt(encoded) => Some(Value::Int(small_int_value(encoded))),
            Opnd::SmallWide(id) => Some(wide_value(id)),
            Opnd::NullRef => Some(Value::Ref(GcRef::NULL)),
            Opnd::Const(cell) => Some(cell.get()),
            _ => None,
        }
    }

    /// `history.py IntFrontendOp(pos, intval)` parity — read the
    /// concrete intrinsic value off this operand. `Const` reads its inline
    /// cell; a bound `Op` / `InputArg` reads the producer's value carrier
    /// (`resoperation.py IntOp._resint`); `None` carries no value.
    /// `history.py` concrete-value read.
    pub fn get_value(&self) -> Option<Value> {
        match self.view() {
            Opnd::SmallInt(encoded) => Some(Value::Int(small_int_value(encoded))),
            Opnd::SmallWide(id) => Some(wide_value(id)),
            Opnd::NullRef => Some(Value::Ref(GcRef::NULL)),
            Opnd::Const(cell) => Some(cell.get()),
            Opnd::Op(op) => op.get_value(),
            Opnd::InputArg(ia) => ia.get_value(),
            Opnd::None => None,
        }
    }

    /// Raw `ConstInt` value with no `IntBound` synthesis (`forwarding.rs`
    /// parity).
    pub fn const_int(&self) -> Option<i64> {
        match self.view() {
            Opnd::SmallInt(encoded) => Some(small_int_value(encoded)),
            Opnd::SmallWide(id) => match wide_value(id) {
                Value::Int(v) => Some(v),
                _ => None,
            },
            Opnd::Const(cell) => match cell.get() {
                Value::Int(v) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    /// `resoperation.py is_constant`.
    pub fn is_constant(&self) -> bool {
        if self.packed == 0 {
            return false;
        }
        matches!(
            self.packed & OP_TAG,
            OP_SMALLINT | OP_SMALLWIDE | OP_NULLREF | OP_CONST
        ) || self.packed == OP_NULLREF
    }

    pub fn is_inputarg(&self) -> bool {
        self.packed != 0 && self.packed & OP_TAG == OP_INPUTARG
    }

    pub fn is_resop(&self) -> bool {
        self.packed != 0 && self.packed & OP_TAG == OP_OP
    }

    /// True for the absent-slot sentinel — the mirror of `OpRef::is_none`.
    pub fn is_none(&self) -> bool {
        self.packed == 0
    }

    /// Object-identity token (`AbstractValue` `is`). Distinct mints of
    /// the same bits get distinct keys; a clone keeps the token.
    pub fn identity_key(&self) -> u64 {
        self.packed
    }

    /// `resoperation.py AbstractValue.same_box`: pointer identity
    /// (`Rc::ptr_eq`) for `Op` / `InputArg`, value comparison for `Const`
    /// (`history.py Const.same_box` delegates to `same_constant`), and the
    /// `None` sentinel matches only itself. Native dispatch on the operand
    /// union: two operands carrying the same producer `Rc` are `ptr_eq`; two
    /// `Const` operands compare by value (`Value`'s `==` is bit-exact, so
    /// `0.0 != -0.0` and `NaN == NaN` — `history.py:251/292/338`); cross-kind
    /// is never the same box. Unlike `==` (uniform `Rc::ptr_eq`, so two equal
    /// fresh `Const`s differ), `same_box` is the value-aware predicate callers
    /// opt into exactly where RPython spells out `same_box(...)`. Equivalent to
    /// the former box-wrapper round-trip (`from_bound_*` memoizes one
    /// wrapper per producer, so its `Rc::ptr_eq` short-circuit and this
    /// producer-`Rc` `ptr_eq` agree), without re-minting a `Const` box.
    pub fn same_box(&self, other: &Operand) -> bool {
        if (self.is_resop() && other.is_resop()) || (self.is_inputarg() && other.is_inputarg()) {
            return self.packed == other.packed;
        }
        if self.is_constant() && other.is_constant() {
            return self.const_value() == other.const_value();
        }
        self.is_none() && other.is_none()
    }

    /// `resoperation.py get_box_replacement(not_const=False)`.
    ///
    /// Walk the `_forwarded` chain from this operand, returning the operand
    /// one step before the chain hits `None`, an `Info` instance, or (when
    /// `not_const`) a constant. Only `Op` / `InputArg` carry a `_forwarded`
    /// slot (`AbstractResOpOrInputArg`); `Const` / `None` are terminal
    /// (`resoperation.py while isinstance(op, AbstractResOpOrInputArg)`).
    /// This is the canonical walker; the former box-wrapper `get_box_replacement`
    /// delegates here.
    #[inline]
    pub fn get_box_replacement(&self, not_const: bool) -> Operand {
        Operand::clone_from_packed(self.replacement_packed(not_const))
    }

    /// Borrowing form of `get_box_replacement(false)` for read-only queries:
    /// `f` sees the chain terminal without the terminal handle's count being
    /// taken and released. `self`'s chain keeps the terminal alive for the
    /// call, so `f` must not rewrite `_forwarded` on that chain.
    #[inline]
    pub fn with_box_replacement<R>(&self, f: impl FnOnce(&Operand) -> R) -> R {
        let view = std::mem::ManuallyDrop::new(Operand {
            packed: self.replacement_packed(false),
        });
        f(&view)
    }

    /// Packed word of the `_forwarded` chain terminal, not counted: `self`
    /// or a hop's `_forwarded` slot keeps it alive.
    fn replacement_packed(&self, not_const: bool) -> u64 {
        if !self.is_resop() && !self.is_inputarg() {
            return self.packed;
        }
        // Packed-tag walk: `self` keeps the first producer alive, and each
        // hop is kept alive by the `_forwarded` slot that points at it.
        let mut cur = self.packed;
        loop {
            let packed_fwd = if cur & OP_TAG == OP_OP {
                // SAFETY: `self` or the previous hop's `_forwarded` slot keeps this Op alive.
                unsafe {
                    crate::resoperation::packed_forwarded_of_op(
                        cur as *const crate::resoperation::Op,
                    )
                }
            } else {
                // SAFETY: `self` or the previous hop's `_forwarded` slot keeps this InputArg alive.
                unsafe {
                    crate::resoperation::packed_forwarded_of_inputarg(
                        (cur & !OP_TAG) as *const crate::value::InputArg,
                    )
                }
            };
            match classify_packed_forwarded(packed_fwd) {
                PackedForwarded::None | PackedForwarded::Info => return cur,
                PackedForwarded::Op(p) => cur = p as u64,
                PackedForwarded::InputArg(p) => cur = (p as u64) | OP_INPUTARG,
                // `_forwarded` contains the Const object itself in RPython;
                // its identity is reused.
                PackedForwarded::Const(p) => {
                    return if not_const {
                        cur
                    } else {
                        (p as u64) | OP_CONST
                    };
                }
                PackedForwarded::SmallConst(enc) => {
                    return if not_const {
                        cur
                    } else {
                        std::mem::ManuallyDrop::new(Operand::SmallInt(enc)).packed
                    };
                }
                PackedForwarded::SmallWide(id) => {
                    return if not_const {
                        cur
                    } else {
                        std::mem::ManuallyDrop::new(Operand::SmallWide(id)).packed
                    };
                }
            }
        }
    }

    /// The bound producer `Op` (`Operand::Op` arm), or `None` for
    /// `InputArg` / `Const` / `None`. The operand IS the producer `Rc` — no
    /// indirection and no `box_cache`.
    pub fn bound_op(&self) -> Option<OpRc> {
        if !self.is_resop() {
            return None;
        }
        let ptr = self.packed as *const crate::resoperation::Op;
        unsafe {
            OpRc::increment_strong_count(ptr);
            Some(OpRc::from_raw(ptr))
        }
    }

    /// The bound `InputArg` (`Operand::InputArg` arm); `None` otherwise.
    /// The carried `InputArg` producer handle, if this is an `InputArg`.
    pub fn bound_inputarg(&self) -> Option<InputArgRc> {
        if !self.is_inputarg() {
            return None;
        }
        let ptr = (self.packed & !OP_TAG) as *const crate::value::InputArg;
        unsafe {
            InputArgRc::increment_strong_count(ptr);
            Some(InputArgRc::from_raw(ptr))
        }
    }

    /// Route a forwarding read to the carried `_forwarded` host
    /// ([`ForwardingHost`]): the bound `Op` / `InputArg`. `Const` / `None`
    /// have no `_forwarded` slot and take the default (mirror of
    /// the carried producer's forwarding host).
    fn read_forwarding_host<R>(&self, default: R, f: impl FnOnce(&dyn ForwardingHost) -> R) -> R {
        match self.view() {
            Opnd::Op(op) => f(&**op),
            Opnd::InputArg(ia) => f(&**ia),
            _ => default,
        }
    }

    /// Route a forwarding write to the carried `_forwarded` host. `Const` is
    /// rejected by the caller's assert first; `None` has no slot and panics
    /// (routes to the carried producer's forwarding host).
    fn with_forwarding_host(&self, what: &str, f: impl FnOnce(&dyn ForwardingHost)) {
        match self.view() {
            Opnd::Op(op) => f(&**op),
            Opnd::InputArg(ia) => f(&**ia),
            _ => panic!(
                "Operand::{what} on a non-producer operand — only a bound \
                 Op/InputArg carries a _forwarded slot (box identity precondition)"
            ),
        }
    }

    /// `resoperation.py get_forwarded`. Clone of the canonical
    /// `_forwarded` slot routed through the carried `Op` / `InputArg`; `Const`
    /// and `None` return `Forwarded::None`. Successor to
    /// `resoperation.py get_forwarded`.
    pub fn get_forwarded(&self) -> Forwarded {
        self.read_forwarding_host(Forwarded::None, |h| h.get_forwarded())
    }

    /// `optimizer.py:394 op.set_forwarded(newop)` — `Op` target. Routes to
    /// [`ForwardingHost::set_forwarded_op`], which carries the
    /// `resoperation.py` self-cycle assert. Const has no `_forwarded`
    /// slot (`AbstractValue` invariant).
    pub fn set_forwarded_op(&self, target: &OpRc) {
        assert!(
            !self.is_constant(),
            "set_forwarded_op on Const violates the AbstractValue invariant \
             (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_op", |h| h.set_forwarded_op(target));
    }

    /// `optimizer.py:394 op.set_forwarded(newop)` — `InputArg` target
    /// (compile.py:478, unroll.py:497). Routes to
    /// [`ForwardingHost::set_forwarded_inputarg`].
    pub fn set_forwarded_inputarg(&self, target: &InputArgRc) {
        assert!(
            !self.is_constant(),
            "set_forwarded_inputarg on Const violates the AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_inputarg", |h| {
            h.set_forwarded_inputarg(target)
        });
    }

    /// `optimizer.py make_constant(box, constbox)` — terminates the chain
    /// in a value-typed payload. Routes to
    /// [`ForwardingHost::set_forwarded_const`].
    pub fn set_forwarded_const(&self, value: Const) {
        assert!(
            !self.is_constant(),
            "set_forwarded_const on Const violates the AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_const", |h| h.set_forwarded_const(value));
    }

    /// `resoperation.py set_forwarded(forwarded_to)` — `Info` target.
    /// Routes to [`ForwardingHost::set_forwarded_info`].
    pub fn set_forwarded_info(&self, info: OpInfo) {
        assert!(
            !self.is_constant(),
            "set_forwarded_info on Const violates the AbstractValue invariant \
             (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_info", |h| h.set_forwarded_info(info));
    }

    /// `_forwarded = None`. No-op on `Const` (no slot); routes to
    /// [`ForwardingHost::clear_forwarded`] on a bound producer.
    pub fn clear_forwarded(&self) {
        if self.is_constant() {
            return;
        }
        self.with_forwarding_host("clear_forwarded", |h| h.clear_forwarded());
    }

    /// `optimizer.py getintbound getptrinfo` reader: the inner `PtrInfo` when
    /// `_forwarded` is `Info(OpInfo::Ptr(_))`, else `None`. Does not walk the
    /// chain (`optimizer.py getintbound getptrinfo`).
    pub fn ptr_info(&self) -> Option<PtrInfoBorrow> {
        self.read_forwarding_host(None, |h| h.ptr_info())
    }

    /// Live `Rc<RefCell<PtrInfo>>` handle for identity-preserving callers
    /// (`Rc::ptr_eq`-based `same_info`).
    pub fn ptr_info_handle(&self) -> Option<Rc<RefCell<PtrInfo>>> {
        self.read_forwarding_host(None, |h| h.ptr_info_handle())
    }

    /// Mutable `PtrInfo` guard for in-place mutation through the shared `Rc`.
    /// Mutable counterpart of [`Operand::ptr_info`].
    pub fn ptr_info_mut(&self) -> Option<PtrInfoBorrowMut> {
        self.read_forwarding_host(None, |h| h.ptr_info_mut())
    }

    /// `optimizer.py getintbound` reader: the inner `IntBound` when
    /// `_forwarded` is `Info(OpInfo::IntBound(_))`, else `None`. Mirror of
    /// `optimizer.py getintbound`.
    pub fn int_bound(&self) -> Option<IntBoundBorrow> {
        self.read_forwarding_host(None, |h| h.int_bound())
    }

    /// Live `Rc<RefCell<IntBound>>` handle. Mirror of
    /// Live `Rc<RefCell<IntBound>>` handle.
    pub fn int_bound_handle(&self) -> Option<crate::intbound::IntBoundRc> {
        self.read_forwarding_host(None, |h| h.int_bound_handle())
    }

    /// Mutable `IntBound` guard for in-place mutation. Mirror of
    /// Mutable counterpart of [`Operand::int_bound`].
    pub fn int_bound_mut(&self) -> Option<IntBoundBorrowMut> {
        self.read_forwarding_host(None, |h| h.int_bound_mut())
    }

    /// True for the live-tracking producer variants (`Op` / `InputArg`),
    /// whose `to_opref()` reads the producer's CURRENT `op.pos`. The
    /// position-remap passes use this to skip operands that auto-track a
    /// renumbered producer (no snapshot rewrite needed); `Const` / `None`
    /// carry no position to remap.
    pub fn is_bound(&self) -> bool {
        self.is_resop() || self.is_inputarg()
    }

    /// GC walk over any inline `ConstPtr` reachable from this operand
    /// (`resoperation.py` `walk_const_ptr_refs`). A wide `Ref` stores a
    /// [`crate::const_ptr_table`] index; tracing that index forwards
    /// `ConstPtr.value` while this operand is live. The legacy `OP_CONST`
    /// cell still carries a `Value::Ref` address and is forwarded here.
    pub fn walk_const_ptr_refs(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        if self.packed != 0 && self.packed & OP_TAG == OP_SMALLWIDE {
            let id = self.packed >> 3;
            if let Some(index) = wide_ref_index(id) {
                crate::const_ptr_table::trace_index(index, visitor);
            }
            return;
        }
        if self.packed == 0 || self.packed & OP_TAG != OP_CONST {
            return;
        }
        let cell = unsafe { &*((self.packed & !OP_TAG) as *const Cell<Value>) };
        let mut v = cell.get();
        if let Value::Ref(gcref) = &mut v {
            visitor(gcref);
            cell.set(v);
        }
    }
}

impl PartialEq for Operand {
    /// Object identity — pure `Rc::ptr_eq`
    /// (`forwarding.rs`): `AbstractValue` defines no `__eq__`
    /// (`resoperation.py`), so every plain box-keyed dict keys by `is`.
    /// `Op` / `InputArg` / `Const` each carry an `Rc`, so `==` is `ptr_eq` on
    /// that producer/const handle; two `none()` sentinels match (Python's
    /// singleton `None`). Equal-valued constants minted separately are NOT
    /// equal here — value equality is the opt-in [`same_box`](Self::same_box)
    /// (`history.py`), never `==`, so a `same_box`-deduping table must
    /// build an explicit value-keyed map, not key on `Operand`.
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed
    }
}

impl Eq for Operand {}

impl std::hash::Hash for Operand {
    /// Identity hashing consistent with [`eq`](Self::eq) — the
    /// `compute_identity_hash` default (`resoperation.py:33-35`). A
    /// per-variant tag keeps cross-variant collisions from aliasing, and the
    /// `Rc` address is the identity for `Op` / `InputArg` / `Const`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.packed.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resoperation::{Op, OpCode};
    use crate::value::{Const, InputArg, Type, Value};

    fn op_at(pos: u32, tp: Type) -> OpRc {
        let op = OpRc::new(Op::new(OpCode::SameAsI, &[]));
        op.pos().set(OpRef::op_typed(pos, tp));
        op
    }

    #[test]
    fn to_opref_round_trips_each_variant() {
        let op = op_at(3, Type::Int);
        assert_eq!(
            Operand::from_bound_op(&op).to_opref(),
            OpRef::op_typed(3, Type::Int)
        );

        let ia = InputArgRc::new(InputArg::from_type(Type::Ref, 2));
        assert_eq!(
            Operand::from_bound_inputarg(&ia).to_opref(),
            OpRef::input_arg_typed(2, Type::Ref),
        );

        assert_eq!(
            Operand::const_(Const::Int(7)).to_opref(),
            OpRef::const_int(7)
        );
        assert_eq!(Operand::none().to_opref(), OpRef::NONE);
    }

    #[test]
    fn accessors_match_variant() {
        let op = op_at(5, Type::Int);
        let o_op = Operand::from_bound_op(&op);
        assert!(o_op.is_resop());
        assert_eq!(o_op.position(), Some(5));
        assert_eq!(o_op.type_(), Type::Int);
        assert_eq!(o_op.const_value(), None);

        let ia = InputArgRc::new(InputArg::from_type(Type::Float, 1));
        let o_ia = Operand::from_bound_inputarg(&ia);
        assert!(o_ia.is_inputarg());
        assert_eq!(o_ia.position(), Some(1));
        assert_eq!(o_ia.type_(), Type::Float);

        let o_c = Operand::const_(Const::Int(9));
        assert!(o_c.is_constant());
        assert_eq!(o_c.position(), None);
        assert_eq!(o_c.type_(), Type::Int);
        assert_eq!(o_c.const_value(), Some(Value::Int(9)));
        assert_eq!(o_c.const_int(), Some(9));

        let o_n = Operand::none();
        assert!(o_n.is_none());
        assert_eq!(o_n.position(), None);
        assert_eq!(o_n.type_(), Type::Void);
    }

    #[test]
    fn same_box_is_pointer_identity_for_producers_value_for_const() {
        let op = op_at(0, Type::Int);
        // Same Rc -> same box.
        assert!(Operand::from_bound_op(&op).same_box(&Operand::from_bound_op(&op)));
        // Distinct ops at the same position -> distinct boxes.
        let op_other = op_at(0, Type::Int);
        assert!(!Operand::from_bound_op(&op).same_box(&Operand::from_bound_op(&op_other)));

        // Equal-valued constants -> same box (value identity).
        assert!(Operand::const_(Const::Int(4)).same_box(&Operand::const_(Const::Int(4))));
        assert!(!Operand::const_(Const::Int(4)).same_box(&Operand::const_(Const::Int(5))));

        // None matches only None.
        assert!(Operand::none().same_box(&Operand::none()));
        assert!(!Operand::none().same_box(&Operand::const_(Const::Int(0))));
    }

    /// Native same_box edge cases the round-trip version also met: the
    /// InputArg `Rc::ptr_eq` arm, the float bit-exact Const compare (hazard 3:
    /// `0.0 != -0.0`, `NaN == NaN`), and cross-kind always-false.
    #[test]
    fn same_box_inputarg_float_and_cross_kind() {
        let ia = InputArgRc::new(InputArg::from_type(Type::Int, 0));
        assert!(Operand::from_bound_inputarg(&ia).same_box(&Operand::from_bound_inputarg(&ia)));
        let ia_other = InputArgRc::new(InputArg::from_type(Type::Int, 0));
        assert!(
            !Operand::from_bound_inputarg(&ia).same_box(&Operand::from_bound_inputarg(&ia_other))
        );

        // Float Const compares bit-exact (Value::eq is to_bits-based).
        assert!(Operand::const_(Const::Float(1.5)).same_box(&Operand::const_(Const::Float(1.5))));
        assert!(!Operand::const_(Const::Float(0.0)).same_box(&Operand::const_(Const::Float(-0.0))));
        assert!(
            Operand::const_(Const::Float(f64::NAN))
                .same_box(&Operand::const_(Const::Float(f64::NAN)))
        );

        // Cross-kind is never the same box.
        let op = op_at(0, Type::Int);
        assert!(!Operand::from_bound_op(&op).same_box(&Operand::from_bound_inputarg(&ia)));
        assert!(!Operand::from_bound_op(&op).same_box(&Operand::const_(Const::Int(0))));
        assert!(!Operand::from_bound_inputarg(&ia).same_box(&Operand::none()));
    }

    /// `from_opref` builds the absent / inline-const arms natively (mirror of
    /// the non-position `OpRef` cases); a position-only ref has no
    /// operand representation and panics (#9 invariant tripwire).
    #[test]
    fn from_opref_none_and_const_arms() {
        assert!(Operand::from_opref(OpRef::None).is_none());
        assert_eq!(
            Operand::from_opref(OpRef::ConstInt(7)).const_value(),
            Some(Value::Int(7))
        );
        assert_eq!(
            Operand::from_opref(OpRef::ConstFloat(1.5)).const_value(),
            Some(Value::Float(1.5))
        );
    }

    #[test]
    #[should_panic(expected = "position-only")]
    fn from_opref_position_only_panics() {
        let _ = Operand::from_opref(OpRef::IntOp(3));
    }

    /// `Eq` is object identity (`Rc::ptr_eq`), the box-key behaviour the
    /// re-keyed side tables depend on: same `Rc` is equal, a fresh mint is
    /// not — including for constants (value equality is `same_box`, never
    /// `==`). A clone shares the `Rc`, so it stays equal and `HashSet`-stable.
    #[test]
    fn eq_and_hash_are_object_identity() {
        use std::collections::HashSet;

        let op = op_at(0, Type::Int);
        // Same producer Rc -> equal; a clone shares the Rc -> equal.
        let a = Operand::from_bound_op(&op);
        assert_eq!(a, a.clone());
        assert_eq!(Operand::from_bound_op(&op), Operand::from_bound_op(&op));
        // Distinct ops at the same position -> distinct identity.
        let op_other = op_at(0, Type::Int);
        assert_ne!(
            Operand::from_bound_op(&op),
            Operand::from_bound_op(&op_other)
        );

        // Equal-valued constants minted separately are NOT `==` (distinct Rc),
        // even though they are `same_box`-equal.
        let c1 = Operand::const_(Const::Int(4));
        let c2 = Operand::const_(Const::Int(4));
        assert_ne!(c1, c2);
        assert!(c1.same_box(&c2));
        // A clone shares the const Rc -> equal.
        assert_eq!(c1, c1.clone());

        // None is a singleton; cross-variant never matches.
        assert_eq!(Operand::none(), Operand::none());
        assert_ne!(Operand::none(), Operand::const_(Const::Int(0)));

        // Hash agrees with Eq: a clone resolves the same bucket/membership.
        #[expect(
            clippy::mutable_key_type,
            reason = "Operand hashing follows immutable OpRef/constant identity; interior mutation belongs to the referenced JIT box and is excluded from Eq and Hash"
        )]
        let mut set = HashSet::new();
        set.insert(Operand::from_bound_op(&op));
        assert!(set.contains(&Operand::from_bound_op(&op)));
        assert!(!set.contains(&Operand::from_bound_op(&op_other)));
        set.insert(c1.clone());
        assert!(set.contains(&c1));
        assert!(!set.contains(&c2));
    }

    /// Forwarding read/write/clear routes through the carried `Op` host,
    /// exercising `{get,set,clear}_forwarded` on the carried producer.
    #[test]
    fn forwarding_get_set_clear_on_op() {
        let a = Operand::from_bound_op(&op_at(0, Type::Int));
        let b = op_at(1, Type::Int);
        assert!(matches!(a.get_forwarded(), Forwarded::None));
        a.set_forwarded_op(&b);
        match a.get_forwarded() {
            Forwarded::Op(target) => assert!(OpRc::ptr_eq(&target, &b)),
            other => panic!("expected Forwarded::Op, got {other:?}"),
        }
        // The walker follows a -> b to the terminal.
        match a.get_box_replacement(false).bound_op() {
            Some(op) => assert!(OpRc::ptr_eq(&op, &b)),
            None => panic!(
                "expected Operand::Op(b), got {:?}",
                a.get_box_replacement(false)
            ),
        }
        a.clear_forwarded();
        assert!(matches!(a.get_forwarded(), Forwarded::None));
    }

    /// The `_forwarded` slot keeps its target alive on its own.
    ///
    /// RPython reaches a forwarding target two ways at once — the assignment
    /// in `resoperation.py set_forwarded` and the trace `operations` list — so
    /// `get_box_replacement` never meets a collected target. While this slot
    /// held a `Weak`, only the second of those existed on the pyre side, spread
    /// over `OptContext`'s `resop_refs` / `phase1_emit_ops` / `new_operations`;
    /// a target held by none of them was dropped, the walk stopped one hop
    /// early, and it handed back an operand that was still forwarding. Its
    /// callers assert that cannot happen — `getnullness` reached an
    /// `unreachable!` on an `int_or` of two `int_is_true` results.
    ///
    /// Dropping every other reference is the point of this test: it is what
    /// no registry-side keep-alive can be asked to prevent.
    #[test]
    fn a_forwarded_target_survives_every_other_reference_being_dropped() {
        let a = Operand::from_bound_op(&op_at(0, Type::Int));

        // Two hops, so the walk has to hold the middle alive to reach the end.
        let (b_ptr, c_ptr) = {
            let b = op_at(1, Type::Int);
            let c = op_at(2, Type::Int);
            b.set_forwarded_op(&c);
            a.set_forwarded_op(&b);
            (OpRc::as_ptr(&b), OpRc::as_ptr(&c))
        };

        match a.get_box_replacement(false).bound_op() {
            Some(op) => assert!(
                std::ptr::eq(OpRc::as_ptr(&op), c_ptr),
                "the walk stopped short of the chain terminal",
            ),
            None => panic!(
                "the walk returned {:?}; landing back on `a` is the \
                 dropped-target termination this test exists to refuse \
                 (middle was {b_ptr:?})",
                a.get_box_replacement(false),
            ),
        }

        // The InputArg twin: `compile.py` / `unroll.py` redirect inputargs the
        // same way in bridge import and retrace remap.
        let d = Operand::from_bound_op(&op_at(3, Type::Int));
        let e_ptr = {
            let e = InputArgRc::new(InputArg::from_type(Type::Int, 9));
            d.set_forwarded_inputarg(&e);
            InputArgRc::as_ptr(&e)
        };
        match d.get_box_replacement(false).bound_inputarg() {
            Some(ia) => assert!(std::ptr::eq(InputArgRc::as_ptr(&ia), e_ptr)),
            None => panic!(
                "the InputArg walk returned {:?}",
                d.get_box_replacement(false)
            ),
        }
    }

    /// Borrowed `_forwarded` walk: one owned Operand at the end, no
    /// intermediate producer refcount churn.
    #[test]
    fn get_box_replacement_borrowed_walk_keeps_intermediate_counts() {
        use crate::op_info::OpInfo;
        use crate::ptr_info::PtrInfo;

        // Op -> Op -> Const
        let a = op_at(0, Type::Int);
        let b = op_at(1, Type::Int);
        let const_cell = Rc::new(Cell::new(Value::Int(99)));
        a.set_forwarded_op(&b);
        b.store_forwarded(Forwarded::Const(Rc::clone(&const_cell)));
        let start = Operand::from_bound_op(&a);
        let before_a = OpRc::strong_count(&a);
        let before_b = OpRc::strong_count(&b);
        {
            let r_false = start.get_box_replacement(false);
            assert_eq!(r_false, Operand::Const(Rc::clone(&const_cell)));
            assert_eq!(r_false.const_value(), Some(Value::Int(99)));
            let r_true = start.get_box_replacement(true);
            assert!(r_true.bound_op().is_some_and(|op| OpRc::ptr_eq(&op, &b)));
        }
        assert_eq!(OpRc::strong_count(&a), before_a);
        assert_eq!(OpRc::strong_count(&b), before_b);

        // Op -> InputArg
        let c = op_at(2, Type::Int);
        let ia = InputArgRc::new(InputArg::from_type(Type::Int, 4));
        let start_ia = Operand::from_bound_op(&c);
        start_ia.set_forwarded_inputarg(&ia);
        let before_c = OpRc::strong_count(&c);
        let before_ia = InputArgRc::strong_count(&ia);
        {
            let r_false = start_ia.get_box_replacement(false);
            assert!(
                r_false
                    .bound_inputarg()
                    .is_some_and(|got| InputArgRc::ptr_eq(&got, &ia))
            );
            let r_true = start_ia.get_box_replacement(true);
            assert!(
                r_true
                    .bound_inputarg()
                    .is_some_and(|got| InputArgRc::ptr_eq(&got, &ia))
            );
        }
        assert_eq!(OpRc::strong_count(&c), before_c);
        assert_eq!(InputArgRc::strong_count(&ia), before_ia);

        // Op whose slot holds Info
        let d = op_at(3, Type::Ref);
        let start_info = Operand::from_bound_op(&d);
        start_info.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        let before_d = OpRc::strong_count(&d);
        {
            let r_false = start_info.get_box_replacement(false);
            let r_true = start_info.get_box_replacement(true);
            assert!(r_false.bound_op().is_some_and(|op| OpRc::ptr_eq(&op, &d)));
            assert!(r_true.bound_op().is_some_and(|op| OpRc::ptr_eq(&op, &d)));
        }
        assert_eq!(OpRc::strong_count(&d), before_d);
    }

    /// `bound_op` / `bound_inputarg` expose the carried producer `Rc` for the
    /// matching arm and `None` everywhere else.
    #[test]
    fn bound_op_and_bound_inputarg_arms() {
        let op = op_at(2, Type::Int);
        let o_op = Operand::from_bound_op(&op);
        assert!(o_op.bound_op().is_some_and(|o| OpRc::ptr_eq(&o, &op)));
        assert!(o_op.bound_inputarg().is_none());

        let ia = InputArgRc::new(InputArg::from_type(Type::Ref, 1));
        let o_ia = Operand::from_bound_inputarg(&ia);
        assert!(
            o_ia.bound_inputarg()
                .is_some_and(|i| InputArgRc::ptr_eq(&i, &ia))
        );
        assert!(o_ia.bound_op().is_none());

        let o_c = Operand::const_(Const::Int(3));
        assert!(o_c.bound_op().is_none() && o_c.bound_inputarg().is_none());
        assert!(Operand::none().bound_op().is_none());
    }

    /// `resoperation.py:241` self-cycle assert fires straight off the carried
    /// producer (the production-direct write path).
    #[test]
    #[should_panic(expected = "one-node chain cycle")]
    fn set_forwarded_op_to_self_panics() {
        let op = op_at(0, Type::Int);
        Operand::from_bound_op(&op).set_forwarded_op(&op);
    }

    /// Const has no `_forwarded` slot — a forwarding write is rejected before
    /// it can silently lose data (`AbstractValue` invariant).
    #[test]
    #[should_panic(expected = "AbstractValue invariant")]
    fn set_forwarded_on_const_panics() {
        let op = op_at(0, Type::Int);
        Operand::const_(Const::Int(0)).set_forwarded_op(&op);
    }

    /// The `None` sentinel carries no host, so a forwarding write panics
    /// rather than no-op away the write.
    #[test]
    #[should_panic(expected = "non-producer operand")]
    fn set_forwarded_on_none_panics() {
        let op = op_at(0, Type::Int);
        Operand::none().set_forwarded_op(&op);
    }

    /// `ptr_info` / `int_bound` read the inner `OpInfo` payload off the
    /// carried host; the `_mut` guard mutates it in place through the shared
    /// `Rc`. Const / unset operands read `None`.
    #[test]
    fn ptr_info_and_int_bound_readers() {
        use crate::intbound::IntBound;
        use crate::op_info::OpInfo;
        use crate::ptr_info::PtrInfo;

        let a = Operand::from_bound_op(&op_at(0, Type::Ref));
        assert!(a.ptr_info().is_none() && a.int_bound().is_none());
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        assert!(a.ptr_info().expect("ptr_info Some").is_nonnull());
        assert!(a.int_bound().is_none());

        let b = Operand::from_bound_op(&op_at(1, Type::Int));
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(42)));
        let ib = b.int_bound().expect("int_bound Some");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 42);

        // Const has no _forwarded slot -> readers return None (no panic).
        assert!(Operand::const_(Const::Int(0)).ptr_info().is_none());
        assert!(Operand::none().int_bound().is_none());
    }

    /// The structured-recorder adapter keeps RPython ConstInt object identity
    /// without allocating the small values that opencoder.py writes inline.
    #[test]
    fn small_int_is_inline_but_keeps_fresh_object_identity() {
        let first = Operand::const_(Const::Int(42));
        let second = Operand::const_(Const::Int(42));
        assert!(first.is_small_int());
        assert!(second.is_small_int());
        assert_ne!(
            first, second,
            "fresh ConstInt objects have distinct identity"
        );
        assert_eq!(first, first.clone(), "cloning preserves ConstInt identity");
        assert!(first.same_box(&second), "ConstInt.same_box compares values");

        let edge = Operand::const_(Const::Int(i32::MAX as i64));
        let outside = Operand::const_(Const::Int(i32::MAX as i64 + 1));
        assert!(edge.is_small_int());
        assert!(outside.is_small_wide());

        // Tagged-word packing: four inline Operand slots stay in 32 B.
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<Operand>(), 8);
    }

    #[test]
    fn wide_const_keeps_fresh_object_identity_without_rc() {
        let f1 = Operand::const_(Const::Float(1.5));
        let f2 = Operand::const_(Const::Float(1.5));
        assert!(f1.is_small_wide());
        assert!(f2.is_small_wide());
        assert_ne!(f1, f2, "fresh ConstFloat objects have distinct identity");
        assert_eq!(f1, f1.clone(), "cloning preserves ConstFloat identity");
        assert!(f1.same_box(&f2), "ConstFloat.same_box compares bits");

        let p1 = Operand::const_(Const::from_gcref(GcRef(0x1000)));
        let p2 = Operand::const_(Const::from_gcref(GcRef(0x1000)));
        assert!(p1.is_small_wide());
        assert_ne!(p1, p2, "fresh ConstPtr objects have distinct identity");
        assert_eq!(p1, p1.clone());
        assert!(p1.same_box(&p2));
        assert_eq!(p1.to_opref(), OpRef::const_ptr(GcRef(0x1000)));

        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<Operand>(), 8);
    }

    #[test]
    fn set_forwarded_const_wide_keeps_token_identity() {
        let host = Operand::from_bound_op(&op_at(0, Type::Ref));
        host.set_forwarded_const(Const::from_gcref(GcRef(0x1000)));
        let first = host.get_box_replacement(false);
        let second = host.get_box_replacement(false);
        assert!(first.is_small_wide());
        assert_eq!(first, second);
        assert_eq!(first.const_value(), Some(Value::Ref(GcRef(0x1000))));
        assert!(host.get_forwarded().is_const());
    }

    /// `opencoder.py Trace._cached_const_ptr` reserves ref-pool index zero for
    /// null, so recording and decoding it must not mint a forwarding cell.
    #[test]
    fn null_ref_is_inline_and_round_trips() {
        let null = Operand::from_opref(OpRef::const_ptr(GcRef::NULL));
        assert!(null.is_null_ref());
        assert!(null.is_constant());
        assert_eq!(null.type_(), Type::Ref);
        assert_eq!(null.const_value(), Some(Value::Ref(GcRef::NULL)));
        assert_eq!(null.to_opref(), OpRef::const_ptr(GcRef::NULL));
        assert!(null.same_box(&Operand::const_(Const::from_gcref(GcRef::NULL))));

        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<Operand>(), 8);
    }
}
