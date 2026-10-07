/// Value types and constants for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py.
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cell::Cell;
use std::sync::atomic::{AtomicPtr, Ordering};

/// The type of a value in the JIT IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    /// Machine-word signed integer (i64).
    Int,
    /// GC-managed reference (pointer).
    Ref,
    /// IEEE 754 double-precision float.
    Float,
    /// No value (void).
    Void,
}

impl Type {
    pub fn from_char(c: char) -> Self {
        match c {
            'i' => Type::Int,
            'r' | 'p' => Type::Ref,
            'f' => Type::Float,
            'v' | 'n' => Type::Void,
            _ => panic!("unknown type char: {c}"),
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Type::Int => 'i',
            Type::Ref => 'r',
            Type::Float => 'f',
            Type::Void => 'v',
        }
    }
}

/// An opaque GC-managed reference.
///
/// In the actual runtime this wraps a pointer to a GC-managed object.
/// During tracing/optimization it may be a tagged value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GcRef(pub usize);

impl GcRef {
    pub const NULL: GcRef = GcRef(0);

    /// Tracer sentinel for "no concrete runtime value is known for this
    /// OpRef" (distinct from `NULL`, which is a genuine null reference).
    ///
    /// The value must never alias a real `PyObjectRef`: a tagged small int is
    /// always odd (`(v << 1) | 1`) and a heap object is 8-aligned, so an even,
    /// non-8-aligned, non-null address is unrepresentable as either. `usize::MAX`
    /// itself is unsound once ints are tagged — it equals `tag_int(-1)` — so the
    /// sentinel clears the low bit to stay off every tagged value.
    pub const NO_CONCRETE: GcRef = GcRef(usize::MAX - 1);

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// A concrete runtime value.
///
/// PartialEq, Eq, and Hash all use f64::to_bits() for Float values,
/// matching RPython history.py where ConstFloat._get_hash_()
/// and same_constant() are both bitwise: 0.0 ≠ -0.0, NaN == NaN
/// (same bits).
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Ref(GcRef),
    Void,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            // history.py:292: longlong.extract_bits(self.value) == longlong.extract_bits(other.value)
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Ref(a), Value::Ref(b)) => a.0 == b.0,
            (Value::Void, Value::Void) => true,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Int(v) => v.hash(state),
            // history.py:283: longlong.gethash(self.value) — bitwise
            Value::Float(v) => v.to_bits().hash(state),
            Value::Ref(r) => r.0.hash(state),
            Value::Void => {}
        }
    }
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::Ref(_) => Type::Ref,
            Value::Void => Type::Void,
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(v) => *v,
            _ => panic!("expected Int, got {:?}", self),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(v) => *v,
            _ => panic!("expected Float, got {:?}", self),
        }
    }

    pub fn as_ref(&self) -> GcRef {
        match self {
            Value::Ref(v) => *v,
            _ => panic!("expected Ref, got {:?}", self),
        }
    }

    /// Machine-word projection: the integer for `Int`, the pointer bits for
    /// `Ref`, the bit pattern for `Float`, and zero for `Void` — the encoding
    /// a compiled exit's slots are read out of the frame in.
    ///
    /// The `Void` arm is why this is not [`Const::as_raw_i64`]: a `Const` has
    /// no void variant, and an exit slot does.
    pub fn as_raw_i64(&self) -> i64 {
        match self {
            Value::Int(v) => *v,
            Value::Ref(v) => v.as_usize() as i64,
            Value::Float(v) => v.to_bits() as i64,
            Value::Void => 0,
        }
    }

    /// Project a `Value` into a `Const`.  Mirrors RPython where
    /// `ConstInt`/`ConstFloat`/`ConstPtr` (history.py/261/307) are
    /// the only concrete constant classes — there is no `ConstVoid`,
    /// so `Value::Void` panics rather than fabricate one.
    pub fn to_const(&self) -> Const {
        match self {
            Value::Int(v) => Const::Int(*v),
            Value::Float(f) => Const::Float(*f),
            Value::Ref(r) => Const::from_gcref(*r),
            Value::Void => panic!(
                "Value::to_const: Void has no Const equivalent \
                 (history.py:220/261/307 — no ConstVoid upstream)"
            ),
        }
    }
}

// `HeapBox { opref, value }` retired — the (identity, value) pair is
// now carried by the frontend object itself: identity by the `OpRef`
// position, value by the `Op` / `InputArg` `value: Cell<Option<Value>>`
// field (PyPy `history.py IntFrontendOp(pos, intval) /
// FloatFrontendOp(pos, floatval) / RefFrontendOp(pos, gcref)` parity).
// Cache writes store the bare `OpRef`; sanity readers resolve the
// intrinsic value via `TraceCtx::box_value` (composing const pool,
// standard-virtualizable shadow, the frontend object's `value` field).  No
// external pair carrier needed.

/// A constant value known at trace time.
///
/// Mirrors rpython/jit/metainterp/resoperation.py Const* classes.
#[derive(Clone, Copy, Debug)]
pub enum Const {
    Int(i64),
    Float(f64),
    /// Index into [`crate::const_ptr_table`], not the referent address.
    /// `getref_base` reads the address at the use. `history.py` `ConstPtr`.
    Ref(u32),
}

impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Const::Int(a), Const::Int(b)) => a == b,
            // history.py ConstFloat.same_constant: bitwise (0.0 ≠ -0.0, NaN == NaN).
            // A derived `f64 ==` here would be IEEE (0.0 == -0.0, NaN != NaN),
            // collapsing distinct ±0.0 constants — diverging from Value/OpRef.
            (Const::Float(a), Const::Float(b)) => a.to_bits() == b.to_bits(),
            (Const::Ref(a), Const::Ref(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Const {}

/// Shared, GC-forwardable `ResumeGuardDescr.rd_consts` list.
///
/// RPython stores one mutable list object on the guard and reference-shares it
/// with copied descriptors and every resume-data reader.  Rust needs interior
/// mutability because the stop-the-world GC root walker forwards `Const::Ref`
/// entries after the pool has been shared.  The JIT and collector run under
/// the same single-threaded/GIL contract; mutator access resumes only after
/// collection has finished.
pub struct SharedConstPool {
    values: std::cell::UnsafeCell<Vec<Const>>,
    visited_generation: std::sync::atomic::AtomicU64,
    /// `incminimark.py:old_objects_pointing_to_young` for this off-GC
    /// equivalent of `ResumeGuardDescr.rd_consts`.  Upstream's list is a GC
    /// object, so assigning the freshly built pool takes the normal write
    /// barrier.  Our immutable `Arc` needs the same one-minor remembered bit:
    /// after that walk, every nursery referent has been promoted to oldgen.
    minor_scan_pending: std::sync::atomic::AtomicBool,
}

unsafe impl Send for SharedConstPool {}
unsafe impl Sync for SharedConstPool {}

impl SharedConstPool {
    pub fn new(values: Vec<Const>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            values: std::cell::UnsafeCell::new(values),
            visited_generation: std::sync::atomic::AtomicU64::new(0),
            minor_scan_pending: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// Consume the write-barrier state for a minor root walk.
    ///
    /// `SharedConstPool` is immutable after publication except for GC
    /// forwarding, so no later store can create another old-to-young edge.
    pub fn take_minor_scan_pending(&self) -> bool {
        self.minor_scan_pending
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn mark_visited_generation(&self, generation: u64) -> bool {
        self.visited_generation
            .swap(generation, std::sync::atomic::Ordering::Relaxed)
            != generation
    }

    pub fn as_slice(&self) -> &[Const] {
        unsafe { &*self.values.get() }
    }

    pub fn snapshot(&self) -> Vec<Const> {
        self.as_slice().to_vec()
    }

    /// Append while the owning optimizer is still constructing its trace.
    ///
    /// RPython's `ResumeDataLoopMemo.consts` is one mutable list shared by
    /// every guard storage in that optimizer.  The optimizer is single-
    /// threaded and no resume-data reader can run until backend compilation
    /// publishes the completed trace, so extending the list in that window is
    /// the Rust analogue of `self.consts.append(const)`.
    ///
    /// # Safety
    /// No slice returned by [`SharedConstPool::as_slice`] may be live across
    /// this call.
    pub unsafe fn push_during_optimization(&self, value: Const) {
        unsafe { &mut *self.values.get() }.push(value);
        self.minor_scan_pending
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// # Safety
    ///
    /// The caller must be the exclusive GC root walker; no resume-data reader
    /// may run concurrently.
    // The pool is `UnsafeCell`-backed precisely so the walker can mutate it
    // through a shared reference; `mut_from_ref` is the signature that escape
    // hatch has to have, and the contract above is what makes it sound.
    #[allow(clippy::mut_from_ref)]
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn as_mut_vec_for_gc(&self) -> &mut Vec<Const> {
        unsafe { &mut *self.values.get() }
    }
}

impl std::ops::Deref for SharedConstPool {
    type Target = [Const];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::fmt::Debug for SharedConstPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl Const {
    pub fn get_type(&self) -> Type {
        match self {
            Const::Int(_) => Type::Int,
            Const::Float(_) => Type::Float,
            Const::Ref(_) => Type::Ref,
        }
    }

    pub fn to_value(self) -> Value {
        match self {
            Const::Int(v) => Value::Int(v),
            Const::Float(v) => Value::Float(v),
            Const::Ref(v) => Value::Ref(crate::const_ptr_table::resolve(v)),
        }
    }

    /// Inverse of [`Const::to_value`]. Panics on `Value::Void` since
    /// PyPy has no void-typed `Const` (`history.py`).
    pub fn from_value(value: Value) -> Self {
        match value {
            Value::Int(v) => Const::Int(v),
            Value::Float(v) => Const::Float(v),
            Value::Ref(v) => Const::from_gcref(v),
            Value::Void => panic!("Const::from_value: Void has no Const counterpart"),
        }
    }

    /// history.py ConstInt.getint — unsigned/signed integer value.
    /// Method belongs to ConstInt upstream; Rust single-enum collapses
    /// all variants, so assert variant at call time instead of
    /// silently cross-casting.
    pub fn getint(&self) -> i64 {
        match self {
            Const::Int(v) => *v,
            other => panic!("Const::getint on non-Int variant: {other:?}"),
        }
    }

    /// history.py `ConstPtr(value)`.
    pub fn from_gcref(v: GcRef) -> Self {
        Const::Ref(crate::const_ptr_table::intern(v))
    }

    /// history.py ConstPtr.getref_base — current GC pointer value.
    pub fn getref_base(&self) -> GcRef {
        match self {
            Const::Ref(v) => crate::const_ptr_table::resolve(*v),
            other => panic!("Const::getref_base on non-Ref variant: {other:?}"),
        }
    }

    /// history.py ConstFloat.getfloatstorage — i64 bit-pattern.
    pub fn getfloatstorage(&self) -> i64 {
        match self {
            Const::Float(v) => v.to_bits() as i64,
            other => panic!("Const::getfloatstorage on non-Float variant: {other:?}"),
        }
    }

    /// Raw i64 projection. For Int it's the value, for Ref the pointer bits,
    /// for Float the bit-pattern of the `f64`. Matches the encoded
    /// `rd_consts[idx].0` layout.
    pub fn as_raw_i64(&self) -> i64 {
        match self {
            Const::Int(v) => *v,
            Const::Ref(v) => crate::const_ptr_table::resolve(*v).0 as i64,
            Const::Float(v) => v.to_bits() as i64,
        }
    }

    /// Reconstruct a Const from the encoded `(raw_i64, Type)` pair.
    pub fn from_raw_i64(raw: i64, tp: Type) -> Self {
        match tp {
            Type::Int => Const::Int(raw),
            Type::Ref => Const::from_gcref(GcRef(raw as usize)),
            Type::Float => Const::Float(f64::from_bits(raw as u64)),
            Type::Void => Const::Int(raw),
        }
    }
}

/// An input argument to a loop or bridge.
///
/// Mirrors rpython/jit/metainterp/resoperation.py AbstractInputArg
/// (`InputArgInt` / `InputArgFloat` / `InputArgRef` at lines 719/727/739).
///
/// The `_forwarded` slot (`resoperation.py`) is the `forwarded`
/// field below — the canonical per-identity forwarding host. A bound
/// operand routes `set_forwarded_*` / `get_forwarded` to this field;
/// there is no Box-side mirror.
#[derive(Debug)]
pub struct InputArg {
    pub tp: Cell<Type>,
    /// Index in the inputargs list.
    pub index: u32,
    /// `resoperation.py AbstractInputArg._forwarded` parity slot —
    /// the canonical forwarding host for a bound InputArg box.
    /// `Forwarded::None` until a writer sets it; `set_forwarded_*`
    /// on a bound box routes here. `ForwardedSlot` (same as `Op`)
    /// drops the `RefCell` borrow flags so `Rc<InputArg>` leaves the
    /// 64-byte class.
    pub forwarded: crate::resoperation::ForwardedSlot,
    /// `resoperation.py/727/739 InputArgInt/Float/Ref` carry the
    /// concrete runtime value on the frontend-arg object itself (the
    /// `_resint`/`_resfloat`/`_resref` mixin slot, `history.py:803-807`).
    /// The canonical per-identity concrete carrier for a bound InputArg
    /// box; the `get_value`/`set_value` accessors route here. `None` until a
    /// writer stamps it (trace-time `set_opref_concrete`).
    pub value: std::cell::Cell<Option<Value>>,
}

impl AsRef<InputArg> for InputArg {
    fn as_ref(&self) -> &InputArg {
        self
    }
}

impl PartialEq for InputArg {
    /// PyPy compares `AbstractInputArg`s by Python object identity
    /// (`AbstractValue.same_box` at `resoperation.py`). Pyre's
    /// value-typed `InputArg` stands in for that identity via
    /// `(tp, index)` tuple equality. `_forwarded` is mutable per-op
    /// state and excluded from identity comparison.
    fn eq(&self, other: &Self) -> bool {
        self.tp.get() == other.tp.get() && self.index == other.index
    }
}

impl InputArg {
    pub fn new_int(index: u32) -> Self {
        InputArg {
            tp: Cell::new(Type::Int),
            index,
            forwarded: crate::resoperation::ForwardedSlot::new(crate::forwarding::Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }

    pub fn new_ref(index: u32) -> Self {
        InputArg {
            tp: Cell::new(Type::Ref),
            index,
            forwarded: crate::resoperation::ForwardedSlot::new(crate::forwarding::Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }

    pub fn new_float(index: u32) -> Self {
        InputArg {
            tp: Cell::new(Type::Float),
            index,
            forwarded: crate::resoperation::ForwardedSlot::new(crate::forwarding::Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }

    pub fn from_type(tp: Type, index: u32) -> Self {
        // `InputArg*` has no Void encoding (resoperation.py InputArgInt/727/739
        // — only `InputArgInt`/`InputArgFloat`/`InputArgRef`); fail at
        // construction so `opref()` can't be reached with a Void slot.
        assert!(
            tp != Type::Void,
            "InputArg::from_type: Type::Void is not a valid input-arg type",
        );
        InputArg {
            tp: Cell::new(tp),
            index,
            forwarded: crate::resoperation::ForwardedSlot::new(crate::forwarding::Forwarded::None),
            value: std::cell::Cell::new(None),
        }
    }

    /// `Rc`-wrapped variants of the typed factories, matching PyPy's
    /// `InputArgInt(index)` / `InputArgFloat(index)` / `InputArgRef(index)`
    /// at `resoperation.py/727/739`. Every call yields a fresh
    /// identity (no interning), so two `new_*_rc` results compare equal
    /// only when both `tp` and `index` match — identity is shared only
    /// when callers `Rc::clone` the same handle.
    /// Read the stamped concrete runtime value (`_resint`/`_resfloat`/
    /// `_resref`, `history.py *FrontendOp.getint()`). `None` until a
    /// writer stamps it.
    pub fn get_value(&self) -> Option<Value> {
        self.value.get()
    }

    /// Stamp the concrete runtime value on this frontend-arg identity.
    pub fn set_value(&self, v: Value) {
        self.value.set(Some(v));
    }

    pub fn new_int_rc(index: u32) -> InputArgRc {
        InputArgRc::new(Self::new_int(index))
    }

    pub fn new_ref_rc(index: u32) -> InputArgRc {
        InputArgRc::new(Self::new_ref(index))
    }

    pub fn new_float_rc(index: u32) -> InputArgRc {
        InputArgRc::new(Self::new_float(index))
    }

    pub fn from_type_rc(tp: Type, index: u32) -> InputArgRc {
        InputArgRc::new(Self::from_type(tp, index))
    }

    /// Returns the OpRef referencing this input arg's slot.
    ///
    /// RPython's `InputArg*` Box object IS its own reference — there is
    /// no separate handle/slot distinction. Pyre encodes the inputarg
    /// position into the low bits of an `OpRef`; this helper centralises
    /// that construction so call sites do not reach for the raw `.index`
    /// field directly.
    pub fn opref(&self) -> crate::resoperation::OpRef {
        crate::resoperation::OpRef::input_arg_typed(self.index, self.tp.get())
    }
}

/// Shared `InputArg` identity (resoperation.py `AbstractInputArg`).
///
/// PyPy's `inputargs` list holds Python objects that are reachable
/// unchanged from `TreeLoop.inputargs`, the optimizer's exported state,
/// the short preamble, resume metadata, and the backend's regalloc
/// surface. The handle is a one-word refcount (no `Weak`) in reserved
/// chunks so `from_type_rc` leaves the 48-byte `RcBox<InputArg>` class.
pub struct InputArgRc {
    ptr: std::ptr::NonNull<InputArgInner>,
}

#[repr(C)]
struct InputArgInner {
    strong: std::cell::Cell<usize>,
    value: InputArg,
}

const INPUTARG_INNER_CHUNK: usize = 1024;

struct InputArgInnerHeap {
    chunks: Vec<(*mut InputArgInner, usize)>,
    free: Vec<std::ptr::NonNull<InputArgInner>>,
}

unsafe impl Send for InputArgInnerHeap {}
unsafe impl Sync for InputArgInnerHeap {}

static INPUTARG_INNER_HEAP: parking_lot::Mutex<InputArgInnerHeap> =
    parking_lot::Mutex::new(InputArgInnerHeap {
        chunks: Vec::new(),
        free: Vec::new(),
    });

fn alloc_inputarg_inner() -> std::ptr::NonNull<InputArgInner> {
    let mut heap = INPUTARG_INNER_HEAP.lock();
    if let Some(p) = heap.free.pop() {
        return p;
    }
    if let Some((base, used)) = heap.chunks.last_mut()
        && *used < INPUTARG_INNER_CHUNK
    {
        let p = unsafe { std::ptr::NonNull::new_unchecked((*base).add(*used)) };
        *used += 1;
        return p;
    }
    let layout = std::alloc::Layout::array::<InputArgInner>(INPUTARG_INNER_CHUNK)
        .expect("InputArgInner chunk");
    let base = unsafe { std::alloc::alloc(layout) as *mut InputArgInner };
    assert!(!base.is_null(), "InputArgInner chunk alloc failed");
    heap.chunks.push((base, 1));
    unsafe { std::ptr::NonNull::new_unchecked(base) }
}

fn free_inputarg_inner(p: std::ptr::NonNull<InputArgInner>) {
    INPUTARG_INNER_HEAP.lock().free.push(p);
}

impl InputArgRc {
    pub fn new(ia: InputArg) -> Self {
        let ptr = alloc_inputarg_inner();
        unsafe {
            ptr.as_ptr().write(InputArgInner {
                strong: std::cell::Cell::new(1),
                value: ia,
            });
        }
        InputArgRc { ptr }
    }

    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }

    /// Update the box type in place. Fail-arg reconciliation can retag a
    /// shared InputArg after unboxing; `tp` is otherwise construction-fixed.
    pub fn set_tp(&self, tp: Type) {
        assert!(
            tp != Type::Void,
            "InputArg::set_tp: Type::Void is not a valid input-arg type",
        );
        self.tp.set(tp);
    }

    pub fn as_ptr(this: &Self) -> *const InputArg {
        unsafe { std::ptr::addr_of!((*this.ptr.as_ptr()).value) }
    }

    pub fn strong_count(this: &Self) -> usize {
        unsafe { this.ptr.as_ref().strong.get() }
    }

    pub fn into_raw(this: Self) -> *const InputArg {
        let p = Self::as_ptr(&this);
        std::mem::forget(this);
        p
    }

    pub unsafe fn from_raw(value: *const InputArg) -> Self {
        let offset = std::mem::offset_of!(InputArgInner, value);
        let inner = (value as usize).wrapping_sub(offset) as *mut InputArgInner;
        InputArgRc {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(inner) },
        }
    }

    #[inline]
    pub unsafe fn increment_strong_count(value: *const InputArg) {
        let rc = unsafe { Self::from_raw(value) };
        let extra = rc.clone();
        std::mem::forget(rc);
        std::mem::forget(extra);
    }
}

impl Clone for InputArgRc {
    #[inline]
    fn clone(&self) -> Self {
        let inner = unsafe { self.ptr.as_ref() };
        inner.strong.set(inner.strong.get() + 1);
        InputArgRc { ptr: self.ptr }
    }
}

impl Drop for InputArgRc {
    #[inline]
    fn drop(&mut self) {
        let inner = unsafe { self.ptr.as_ref() };
        let n = inner.strong.get() - 1;
        if n == 0 {
            release_inputarg_inner(self.ptr);
        } else {
            inner.strong.set(n);
        }
    }
}

/// Last-reference teardown, kept out of line so the count decrement in
/// `InputArgRc::drop` inlines at every handle drop.
#[cold]
#[inline(never)]
fn release_inputarg_inner(ptr: std::ptr::NonNull<InputArgInner>) {
    unsafe {
        std::ptr::drop_in_place(&mut (*ptr.as_ptr()).value);
    }
    free_inputarg_inner(ptr);
}

impl std::ops::Deref for InputArgRc {
    type Target = InputArg;
    fn deref(&self) -> &InputArg {
        unsafe { &self.ptr.as_ref().value }
    }
}

impl PartialEq for InputArgRc {
    fn eq(&self, other: &Self) -> bool {
        InputArg::eq(self, other)
    }
}

impl AsRef<InputArg> for InputArgRc {
    fn as_ref(&self) -> &InputArg {
        self
    }
}

impl std::borrow::Borrow<InputArg> for InputArgRc {
    fn borrow(&self) -> &InputArg {
        self
    }
}

impl std::fmt::Debug for InputArgRc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

impl From<InputArg> for InputArgRc {
    fn from(ia: InputArg) -> Self {
        InputArgRc::new(ia)
    }
}

/// Limit on the number of fail arguments per guard.
///
/// From history.py: FAILARGS_LIMIT = 1000
pub const FAILARGS_LIMIT: usize = 1000;

/// Classification of a variable as green (compile-time constant) or red (runtime).
///
/// Mirrors RPython's `JitDriver(greens=[...], reds=[...])` distinction:
/// - **Green** variables identify the program point (loop header).
///   They are fixed for a given compiled trace and encoded as constants in the IR.
/// - **Red** variables carry runtime state across loop iterations.
///   They become the trace's `InputArg`s.
///
/// During tracing, `promote` (GUARD_VALUE) converts a red value to green
/// by asserting it equals a specific constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarKind {
    /// Compile-time constant: contributes to the green key identity.
    /// Changing a green variable means a different loop / compilation unit.
    Green,
    /// Runtime variable: carried as InputArg in the trace.
    Red,
}

/// A descriptor for a JitDriver variable (green or red).
///
/// Mirrors RPython's JitDriver parameter lists. Each variable has
/// a name (for debugging), a type, and a kind (green/red).
///
/// `warmspot.py:663 jd._green_args_spec = [TYPE for ARG, TYPE in
/// jd._JIT_ENTER_FUNCTYPE.ARGS[:jd.num_green_args]]` carries the full
/// lltype TYPE per green — including `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`.
/// The IR-level [`Type`] (i/r/f/v) collapses those to `Ref`, so greens
/// also carry an optional [`GreenType`] preserving the STR/UNICODE
/// subtype.  Reds keep `green_type = None` (no equivalent of
/// `equal_whatever` for runtime args).
#[derive(Clone, Debug, PartialEq)]
pub struct JitDriverVar {
    /// Variable name (e.g., "pc", "stack", "sp").
    pub name: String,
    /// Type of this variable.
    pub tp: Type,
    /// Whether this is a green (constant) or red (runtime) variable.
    pub kind: VarKind,
    /// Per-green lltype subtype (`STR` / `UNICODE` / canonical
    /// `Int`/`Float`/`Ref`/`Void`).  `None` for reds and for greens
    /// constructed via [`Self::green`] without an explicit subtype —
    /// the canonical fall-back is `GreenType::from(self.tp)`.
    pub green_type: Option<GreenType>,
}

impl JitDriverVar {
    /// Construct a green var with the canonical [`GreenType`] derived
    /// from `tp` (Int/Ref/Float/Void).  Use [`Self::green_with_type`]
    /// when the upstream lltype is `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
    /// so `green_args_spec` reports the correct
    /// `equal_whatever`/`hash_whatever` dispatch type.
    pub fn green(name: impl Into<String>, tp: Type) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Green,
            green_type: None,
        }
    }

    /// Construct a green var with an explicit [`GreenType`] subtype —
    /// `warmspot.py:663` parity for STR/UNICODE Ptr greens that
    /// `Type::Ref` cannot distinguish on its own.
    pub fn green_with_type(name: impl Into<String>, tp: Type, green_type: GreenType) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Green,
            green_type: Some(green_type),
        }
    }

    pub fn red(name: impl Into<String>, tp: Type) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Red,
            green_type: None,
        }
    }
}

/// warmstate.py equal_whatever / :115-128 hash_whatever take a
/// TYPE parameter that can be primitive, generic Ptr, or specifically a
/// Ptr to rstr.STR / rstr.UNICODE. The IR-level [`Type`] only carries the
/// kind (i/r/f/v); this enum extends it with the STR/UNICODE subtypes so
/// green key comparisons match RPython 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GreenType {
    /// lltype.Signed / Unsigned / Bool / Char / primitive.
    Int,
    /// lltype.Float.
    Float,
    /// lltype.Void.
    Void,
    /// Generic GC Ptr — identityhash / pointer equality.
    Ref,
    /// Ptr to rstr.STR — ll_streq / ll_strhash.
    Str,
    /// Ptr to rstr.UNICODE — ll_streq / ll_strhash.
    Unicode,
}

impl From<Type> for GreenType {
    fn from(t: Type) -> Self {
        match t {
            Type::Int => GreenType::Int,
            Type::Ref => GreenType::Ref,
            Type::Float => GreenType::Float,
            Type::Void => GreenType::Void,
        }
    }
}

/// Project a [`GreenType`] back onto the IR-level [`Type`].  STR /
/// UNICODE collapse to `Type::Ref` (their lltype is `lltype.Ptr` —
/// `warmstate.py:109` `isinstance(TYPE, lltype.Ptr)` precondition).
pub fn green_type_to_ir(g: GreenType) -> Type {
    match g {
        GreenType::Int => Type::Int,
        GreenType::Float => Type::Float,
        GreenType::Void => Type::Void,
        GreenType::Ref | GreenType::Str | GreenType::Unicode => Type::Ref,
    }
}

/// Resolver function pointer types for `GreenType::Str` /
/// `GreenType::Unicode` content equality and hashing.
///
/// Pluggable because each frontend has its own `rstr.STR` / `rstr.UNICODE`
/// layout — RPython holds gc-typed pointers, pyre holds `*const &'static str`
/// cast through `usize` (per-occurrence `Box::leak`).  The IR layer keeps
/// no assumption about that ABI; frontends register their decoders via
/// [`set_str_resolver`] / [`set_unicode_resolver`] at startup, mirroring
/// RPython's `warmstate.py equal_whatever` indirection through
/// `rstr.LLHelpers.ll_streq` / `ll_strhash` (which is itself a function
/// pointer the frontend produces for its own STR/UNICODE layout).
pub type StrEqFn = fn(i64, i64) -> bool;
pub type StrHashFn = fn(i64) -> u64;

static STR_EQ: std::sync::OnceLock<StrEqFn> = std::sync::OnceLock::new();
static STR_HASH: std::sync::OnceLock<StrHashFn> = std::sync::OnceLock::new();
static UNICODE_EQ: std::sync::OnceLock<StrEqFn> = std::sync::OnceLock::new();
static UNICODE_HASH: std::sync::OnceLock<StrHashFn> = std::sync::OnceLock::new();

/// Frontend-registered Str green resolver.  First call wins; subsequent
/// calls are silently ignored to mirror `OnceLock::set`'s init-once
/// contract.  Frontends register at JitDriver startup before any
/// trace records a Str green.
pub fn set_str_resolver(eq: StrEqFn, hash: StrHashFn) {
    let _ = STR_EQ.set(eq);
    let _ = STR_HASH.set(hash);
}

/// Frontend-registered Unicode green resolver.  Same init-once contract
/// as [`set_str_resolver`].
pub fn set_unicode_resolver(eq: StrEqFn, hash: StrHashFn) {
    let _ = UNICODE_EQ.set(eq);
    let _ = UNICODE_HASH.set(hash);
}

/// Frontend hooks that keep a `GreenType::Ref` green's referent alive for as
/// long as a cell stores it.
///
/// Unlike every other citation in this file, this pair ports an INVARIANT
/// and not a function.  There is no upstream symbol to point at: RPython's
/// `JitCell.__init__` (warmstate.py) does
///
/// ```python
/// for attrname, _ in green_args_name_spec:
///     setattr(self, attrname, greenargs[i])
/// ```
///
/// and a `setattr` of a gc pointer simply *is* an owning store — the GC is
/// universal, so ownership needs no code.  Do not "fix" this comment by
/// citing a `warmstate.py` helper; none exists, and a faithful citation over
/// unfaithful code is worse than no citation.
///
/// The invariant it restores: `equal_whatever(Ref, ..)` compares the stored
/// pointer with the live pointer, so **address equality is a correct identity
/// only while the stored referent is kept alive by its owner** — two distinct
/// *live* objects cannot share an address.  `hash_whatever(Ref, ..)` separately
/// uses the GC's stable identity hash.  Upstream satisfies both because the
/// cell that stores the key is the owner.  Without these hooks a freed
/// referent's address can be recycled by a different object, whose green key
/// is then byte-identical to the dead one's: a true collision in the identity
/// that no comparator can resolve.
///
/// Ownership belongs to the **stored** key, not to [`GreenKey`] as a type.
/// A key built to *look up* a cell is transient and the caller is executing
/// the very code it names, so it needs no retain; `pyjitpl`'s decision key is
/// a reused thread-local whose slots are overwritten in place, and giving
/// `GreenKey` a `Drop` would leak the old referent and fail to retain the new
/// one on every back edge.  Retain where the cell takes its copy, which is
/// also where upstream's `setattr` runs.
pub type RefRetainFn = fn(i64) -> usize;
pub type RefCurrentFn = fn(usize) -> i64;
pub type RefReleaseFn = fn(usize);

static REF_RETAIN: std::sync::OnceLock<RefRetainFn> = std::sync::OnceLock::new();
static REF_CURRENT: std::sync::OnceLock<RefCurrentFn> = std::sync::OnceLock::new();
static REF_RELEASE: std::sync::OnceLock<RefReleaseFn> = std::sync::OnceLock::new();

/// Frontend-registered `Ref` green ownership hooks.  Same init-once contract
/// as [`set_str_resolver`].  Register at JitDriver startup, before any cell
/// stores a typed green key.
pub fn set_ref_resolver(retain: RefRetainFn, current: RefCurrentFn, release: RefReleaseFn) {
    let _ = REF_RETAIN.set(retain);
    let _ = REF_CURRENT.set(current);
    let _ = REF_RELEASE.set(release);
}

/// `lltype.identityhash` for a generic GC pointer (`warmstate.py hash_whatever`).
///
/// The GC transformer turns that call into the collector's `identityhash`
/// (`boehm.py ll_identityhash`, `minimark.py identityhash`). Installable and
/// clearable. Unset means the address.
pub type GcIdOrIdentityHashFn = fn(usize) -> usize;

static GC_ID_OR_IDENTITYHASH: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Install or clear the process-global `identityhash` hook.
#[inline]
pub fn set_gc_id_or_identityhash(hook: Option<GcIdOrIdentityHashFn>) {
    let raw: *mut () = match hook {
        None => std::ptr::null_mut(),
        Some(f) => {
            // `f` is a bare fn pointer, pointer-sized. Store those bits.
            unsafe { std::mem::transmute_copy::<GcIdOrIdentityHashFn, *mut ()>(&f) }
        }
    };
    GC_ID_OR_IDENTITYHASH.store(raw, Ordering::Release);
}

/// The installed `identityhash`, or `addr` when the slot is clear.
#[inline]
pub fn gc_id_or_identityhash(addr: usize) -> usize {
    let p = GC_ID_OR_IDENTITYHASH.load(Ordering::Acquire);
    if p.is_null() {
        addr
    } else {
        let f: GcIdOrIdentityHashFn = unsafe { std::mem::transmute_copy(&p) };
        f(addr)
    }
}

/// Live address of a managed object (`gc_current_object_address`).
///
/// Follows a nursery forwarding stub without treating `addr` as a root.
/// Unset means the address. `majit-ir` has no collector dependency; the
/// backend installs the real query next to `set_gc_id_or_identityhash`.
pub type GcCurrentObjectAddressFn = fn(usize) -> usize;

static GC_CURRENT_OBJECT_ADDRESS: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Install or clear the process-global live-address hook.
#[inline]
pub fn set_gc_current_object_address(hook: Option<GcCurrentObjectAddressFn>) {
    let raw: *mut () = match hook {
        None => std::ptr::null_mut(),
        Some(f) => unsafe { std::mem::transmute_copy::<GcCurrentObjectAddressFn, *mut ()>(&f) },
    };
    GC_CURRENT_OBJECT_ADDRESS.store(raw, Ordering::Release);
}

/// The installed live address, or `addr` when the slot is clear.
#[inline]
pub fn gc_current_object_address(addr: usize) -> usize {
    let p = GC_CURRENT_OBJECT_ADDRESS.load(Ordering::Acquire);
    if p.is_null() {
        addr
    } else {
        let f: GcCurrentObjectAddressFn = unsafe { std::mem::transmute_copy(&p) };
        f(addr)
    }
}

/// OPEN POLICY FORK — the one place that decides what an unregistered
/// frontend gets.  Today: today's behaviour, i.e. no retain, which leaves the
/// address-recycling hazard exactly as it is rather than changing it.
///
/// This is deliberately NOT [`set_str_resolver`]'s panic-on-unregistered
/// contract.  That contract is right for STR because comparing a STR green
/// without the frontend's decoder yields a *wrong answer*, so refusing is
/// strictly better.  A `Ref` green compares correctly without a retainer; it
/// is only unsound once a referent is freed and its address reused.  Copying
/// the panic here would break every frontend in the tree on day one for a
/// hazard most of them may not have.
///
/// The alternative under consideration is to require an explicit declaration —
/// either [`set_ref_resolver`] or an explicit "unmanaged" opt-out — and panic
/// when a `Ref` green is stored with neither, converting a silent assumption
/// into a checkable one.  Until that is decided, registering is opt-in: a
/// frontend that registers gets the invariant, one that does not is no worse
/// off than before.
fn ref_hooks() -> Option<(RefRetainFn, RefCurrentFn, RefReleaseFn)> {
    match (REF_RETAIN.get(), REF_CURRENT.get(), REF_RELEASE.get()) {
        (Some(retain), Some(current), Some(release)) => Some((*retain, *current, *release)),
        _ => None,
    }
}

/// Owning handle over the `Ref`-typed greens of one stored [`GreenKey`].
///
/// Retains on construction and releases on drop, so a cell holding one keeps
/// its referents alive for exactly the cell's lifetime.  Deliberately not
/// `Clone`: a second owner would need a second retain, and every store site
/// should be visible as its own [`RetainedGreens::retain`] call.
///
/// `GreenType::Str` / `GreenType::Unicode` are the same hazard class — they
/// are pointers too, and `green_type_to_type` maps both to [`Type::Ref`] — but
/// they are not retained here because `majit-macros` refuses those tags at
/// parse time today.  If that refusal is lifted, this is the site that must
/// grow with it.
#[derive(Debug, Default)]
pub struct RetainedGreens {
    /// `(GreenKey.values index, frontend root handle)`.
    retained: Vec<(usize, usize)>,
}

impl RetainedGreens {
    /// Retain every non-null `Ref` green in `key`.  A null slot is skipped:
    /// `hash_whatever` already folds null to 0 and there is nothing to own.
    pub fn retain(key: &GreenKey) -> Self {
        let Some((retain, _, _)) = ref_hooks() else {
            return RetainedGreens::default();
        };
        let mut retained = Vec::new();
        for (i, &value) in key.values.iter().enumerate() {
            if key.types.get(i).copied().unwrap_or(GreenType::Int) == GreenType::Ref && value != 0 {
                retained.push((i, retain(value)));
            }
        }
        RetainedGreens { retained }
    }

    /// Number of referents this handle owns.  Feeds the pinned-population
    /// counter: the memory an unbounded cell table now holds live is exactly
    /// the sum of this over all cells, and it must be readable rather than
    /// showing up only as RSS.
    pub fn len(&self) -> usize {
        self.retained.len()
    }

    pub fn is_empty(&self) -> bool {
        self.retained.is_empty()
    }

    /// Current address of the retained Ref at `green_index`.
    ///
    /// RPython stores the green directly in the GC-managed `JitCell`, so the
    /// collector rewrites that field. Pyre's cell is Rust-owned; the frontend
    /// root slot is the corresponding owning field and this read supplies its
    /// forwarded value to `JitCell.comparekey`.
    pub fn current_value(&self, green_index: usize, fallback: i64) -> i64 {
        let Some((_, handle)) = self
            .retained
            .iter()
            .find(|(index, _)| *index == green_index)
        else {
            return fallback;
        };
        ref_hooks().map_or(fallback, |(_, current, _)| current(*handle))
    }
}

impl Drop for RetainedGreens {
    fn drop(&mut self) {
        let Some((_, _, release)) = ref_hooks() else {
            return;
        };
        for &(_, handle) in &self.retained {
            release(handle);
        }
    }
}

/// Pyre canonical `ll_streq` analog for `GreenType::Str` / `GreenType::Unicode`.
///
/// Public so consumers can register it at startup via
/// [`set_str_resolver`] / [`set_unicode_resolver`] — pyre's macro
/// (`majit-macros::jit_interp::emit_green_repr`) emits the `*const
/// &'static str` slot ABI (per-occurrence `Box::leak` of a `Box<&'static str>`,
/// the pyre analog to RPython's GC-allocated `rstr.STR*`).  Decoding
/// the slot to compare by content is the pyre half of the contract;
/// RPython's half is in `rstr.py ll_streq`.  Other frontends with
/// different STR layouts register their own resolver pair.
///
/// Stable-slot helper for STR/UNICODE green emission.
///
/// Materialises a `*const &'static str` slot at a stable address by
/// leaking a `Box<&'static str>` — pyre's per-occurrence analog of
/// RPython's GC-allocated `rstr.STR*` (`rstr.py`).  Each call
/// allocates a fresh slot; the GreenKey HashMap content-de-dupes via
/// `default_str_eq` / `default_str_hash` so semantically every
/// merge-point hit collapses to a single cache entry, but the leaked
/// slots themselves grow unboundedly with hit count for long-running
/// programs.
///
/// This shape is a Rust-side adaptation: `&str` literals don't have
/// stable backing-storage addresses by default, and RPython's
/// `rstr.STR*` keeps the data + len at a GC-allocated address that is
/// stable for the JitCell's lifetime.  RPython does NOT maintain a
/// global string-intern side table — `JitCell.greenargs[i]` holds the
/// rstr pointer and the GC frees it when the cell dies.  An earlier
/// pyre revision added a process-global `HashMap<Box<str>, usize>`
/// here to bound the leak, but a global intern is non-orthodox: it
/// has no RPython precedent and never frees, so the bound it provides
/// only delays the unbounded growth (distinct STR/UNICODE contents
/// still grow without bound over a process lifetime).  The structural
/// fix — reshape `GreenKey::values` from `Vec<i64>` to a typed enum
/// carrying `Box<str>` for str/unicode greens, with the macro
/// emitting a temporary that the JitCell cache promotes on insertion
/// — is a refactor and is intentionally deferred.
/// Functional behavior matches RPython (content-keyed compare/hash);
/// only the lifetime / allocation profile differs.
pub fn make_str_slot(s: &str) -> i64 {
    let owned: Box<str> = Box::from(s);
    let owned_static: &'static str = Box::leak(owned);
    let slot: &'static &'static str = Box::leak(Box::new(owned_static));
    slot as *const &'static str as usize as i64
}

pub fn default_str_eq(a: i64, b: i64) -> bool {
    if a == b {
        // `if s1 == s2: return True` (`rstr.py:604`) — handles both
        // pointer-equal and `(0, 0)` (both null).
        return true;
    }
    if a == 0 || b == 0 {
        // `if not s1 or not s2: return False` (`rstr.py:606`) — exactly
        // one side is null; null vs non-null can never match.  Without
        // this guard the deref below would dereference a null pointer.
        return false;
    }
    // SAFETY: pyre canonical green ABI — both `i64`s are non-null
    // `*const &'static str` pointing to leaked slots with `'static`
    // lifetime per `emit_green_repr`'s `Box::leak` path.
    let (sa, sb) = unsafe {
        (
            *(a as usize as *const &'static str),
            *(b as usize as *const &'static str),
        )
    };
    sa == sb
}

/// `rpython/rlib/objectmodel.py _hash_string` parity over a byte
/// stream — the modified Fowler-Noll-Vo (FNV) variant that
/// CPython 2.7 (and RPython by inheritance) uses for string hashes:
///
/// ```text
///     length = len(s)
///     if length == 0: return -1
///     x = ord(s[0]) << 7
///     for i in 0..length: x = intmask((1000003 * x) ^ ord(s[i]))
///     x ^= length
///     return intmask(x)
/// ```
///
/// `intmask` truncates to the machine word; on a 64-bit target this
/// matches `i64::wrapping_mul` / `i64::wrapping_xor` exactly, so the
/// value is bit-identical to a 64-bit RPython build.
fn rpython_hash_bytes(bytes: &[u8]) -> i64 {
    let length = bytes.len();
    if length == 0 {
        return -1;
    }
    let mut x: i64 = (bytes[0] as i64) << 7;
    for &b in bytes {
        x = 1000003i64.wrapping_mul(x) ^ (b as i64);
    }
    x ^= length as i64;
    x
}

/// `rpython/rlib/objectmodel.py _hash_string` parity over a
/// codepoint stream — companion to [`rpython_hash_bytes`] for
/// `rstr.UNICODE` (whose `chars` field is an array of codepoint
/// values, not bytes).  Iterates the `&str` codepoint-by-codepoint
/// so multi-byte UTF-8 sequences hash by their decoded value rather
/// than the byte representation.
fn rpython_hash_codepoints(s: &str) -> i64 {
    let length = s.chars().count();
    if length == 0 {
        return -1;
    }
    let first = s.chars().next().unwrap() as i64;
    let mut x: i64 = first << 7;
    for c in s.chars() {
        x = 1000003i64.wrapping_mul(x) ^ (c as i64);
    }
    x ^= length as i64;
    x
}

/// `rpython/rtyper/lltypesystem/rstr.py _ll_strhash` zero-substitute.
///
/// RPython treats `0` as the "hash not yet computed" sentinel for
/// `rstr.STR.hash` / `rstr.UNICODE.hash` and substitutes a fixed
/// non-zero replacement (`29872897`) when the FNV result is zero.
/// pyre mirrors that contract for both `Str` and `Unicode` greens —
/// stable bucket assignment requires a deterministic non-zero hash.
fn rpython_zero_substitute(x: i64) -> i64 {
    if x == 0 { 29872897 } else { x }
}

/// Pyre canonical `ll_strhash` analog for `GreenType::Str` (Py2 byte
/// strings).  Public so consumers can register it via
/// [`set_str_resolver`].  Decodes the slot ABI (same as
/// [`default_str_eq`]) and hashes the underlying `&str` byte-by-byte
/// through [`rpython_hash_bytes`] — `rstr.STR.chars` is a sequence of
/// single-byte values, so byte iteration matches the upstream
/// `ord(s[i])` loop bit-for-bit.
pub fn default_str_hash(a: i64) -> u64 {
    // `if not s: return 0` (`rstr.py:407`) — null slot hashes to 0,
    // matching RPython's `_ll_strhash` precondition guard.  Without
    // this the deref would dereference a null pointer for unset
    // green slots.
    if a == 0 {
        return 0;
    }
    // SAFETY: non-null canonical slot ABI as [`default_str_eq`].
    let s = unsafe { *(a as usize as *const &'static str) };
    rpython_zero_substitute(rpython_hash_bytes(s.as_bytes())) as u64
}

/// Pyre canonical `ll_strhash` analog for `GreenType::Unicode`
/// (Py2 unicode strings — codepoint-indexed).  Public so consumers
/// can register it via [`set_unicode_resolver`].  Decodes the slot
/// ABI and hashes codepoint-by-codepoint through
/// [`rpython_hash_codepoints`] — `rstr.UNICODE.chars` is a sequence
/// of decoded codepoint values, matching `&str.chars()` iteration.
pub fn default_unicode_hash(a: i64) -> u64 {
    // `if not s: return 0` (`rstr.py:407`) — null slot hashes to 0,
    // matching `_ll_strhash`'s precondition guard.
    if a == 0 {
        return 0;
    }
    // SAFETY: non-null canonical slot ABI as [`default_str_eq`].
    let s = unsafe { *(a as usize as *const &'static str) };
    rpython_zero_substitute(rpython_hash_codepoints(s)) as u64
}

/// warmstate.py equal_whatever(TYPE, x, y)
///
/// Port of RPython's lltype dispatch:
/// - Ptr to STR / UNICODE → rstr.LLHelpers.ll_streq
/// - everything else → `x == y` (with Float using bitwise f64 equality)
///
/// STR / UNICODE indirect through frontend-registered resolvers
/// ([`set_str_resolver`] / [`set_unicode_resolver`]).  Each frontend
/// owns its own `rstr.STR` / `rstr.UNICODE` decoder; PyPy compiles a
/// type-specialised `equal_whatever(STR, ..)` whose body is the
/// frontend's `ll_streq` (`@specialize.arg(0)` at `warmstate.py`).
/// Pyre's runtime equivalent: a registered resolver MUST exist before
/// any STR/UNICODE green key is compared.  An unregistered call
/// panics rather than silently falling back to bitwise equality —
/// PyPy never returns `x == y` for an STR green, and a silent
/// fallback masks frontend-init bugs (e.g., calling
/// `equal_whatever(GreenType::Str, ..)` before
/// `install_jit_call_bridge` runs).
///
/// `#[inline]` is `@specialize.arg(0)` (warmstate.py): `tp` is a
/// translation-time constant at every generated merge point, so the
/// Int/Ref/Void arm is a compare once this inlines. Resolver-missing
/// panics stay behind `#[cold] #[inline(never)]` so they do not inflate
/// that arm.
#[inline]
pub fn equal_whatever(tp: GreenType, x: i64, y: i64) -> bool {
    match tp {
        GreenType::Str => match STR_EQ.get() {
            Some(eq) => eq(x, y),
            None => equal_whatever_missing_str(),
        },
        GreenType::Unicode => match UNICODE_EQ.get() {
            Some(eq) => eq(x, y),
            None => equal_whatever_missing_unicode(),
        },
        GreenType::Float => {
            let a = f64::from_bits(x as u64);
            let b = f64::from_bits(y as u64);
            a == b
        }
        // Int, Ref, Void: x == y (integer / pointer equality)
        GreenType::Int | GreenType::Ref | GreenType::Void => x == y,
    }
}

#[cold]
#[inline(never)]
fn equal_whatever_missing_str() -> ! {
    panic!(
        "equal_whatever(GreenType::Str, ..): no Str resolver \
                 registered — frontend must call \
                 `set_str_resolver(eq, hash)` at startup before any \
                 STR green key is compared (warmstate.py:108-111 \
                 ll_streq parity)"
    );
}

#[cold]
#[inline(never)]
fn equal_whatever_missing_unicode() -> ! {
    panic!(
        "equal_whatever(GreenType::Unicode, ..): no Unicode \
                 resolver registered — frontend must call \
                 `set_unicode_resolver(eq, hash)` at startup before \
                 any UNICODE green key is compared (warmstate.py:\
                 108-111 ll_streq parity)"
    );
}

/// warmstate.py hash_whatever(TYPE, x)
///
/// - Ptr to STR / UNICODE → rstr.LLHelpers.ll_strhash
/// - generic GC Ptr → identityhash (or 0 for null)
/// - primitive → rffi.cast(Signed, x)
///
/// STR / UNICODE indirect through frontend-registered resolvers
/// ([`set_str_resolver`] / [`set_unicode_resolver`]).  PyPy compiles
/// a type-specialised `hash_whatever(STR, ..)` whose body is the
/// frontend's `ll_strhash` (`@specialize.arg(0)` at `warmstate.py`).
/// Pyre's runtime equivalent: a registered resolver MUST exist before
/// any STR/UNICODE green key is hashed.  An unregistered call panics
/// rather than silently returning the slot-bits-as-u64 — PyPy never
/// hashes an STR green by raw pointer bits, and a silent fallback
/// masks frontend-init bugs (e.g., calling
/// `hash_whatever(GreenType::Str, ..)` before
/// `install_jit_call_bridge` runs).
///
/// `#[inline]` is `@specialize.arg(0)` (warmstate.py): `tp` is a
/// translation-time constant at every generated merge point, so the
/// Int arm is a cast once this inlines into [`green_uhash_step`].
/// Resolver-missing panics stay behind `#[cold] #[inline(never)]` so
/// they do not inflate that arm.
#[inline]
pub fn hash_whatever(tp: GreenType, value: i64) -> u64 {
    match tp {
        GreenType::Str => match STR_HASH.get() {
            Some(hash) => hash(value),
            None => hash_whatever_missing_str(),
        },
        GreenType::Unicode => match UNICODE_HASH.get() {
            Some(hash) => hash(value),
            None => hash_whatever_missing_unicode(),
        },
        GreenType::Ref => {
            // warmstate.py `hash_whatever`: `lltype.identityhash(x)` or 0.
            // `minimark.py identityhash` applies `support.py mangle_hash`,
            // `i ^ (i >> 4)`, to the move-stable address.
            if value == 0 {
                0
            } else {
                let identity = gc_id_or_identityhash(value as usize) as u64;
                identity ^ (identity >> 4)
            }
        }
        GreenType::Float => {
            // rffi.cast(Signed, x) — truncate float to integer
            let float_val = f64::from_bits(value as u64);
            (float_val as i64) as u64
        }
        // Int, Void: rffi.cast(Signed, x) — the value itself
        GreenType::Int | GreenType::Void => value as u64,
    }
}

#[cold]
#[inline(never)]
fn hash_whatever_missing_str() -> ! {
    panic!(
        "hash_whatever(GreenType::Str, ..): no Str resolver \
                 registered — frontend must call \
                 `set_str_resolver(eq, hash)` at startup before any \
                 STR green key is hashed (warmstate.py:115-121 \
                 ll_strhash parity)"
    );
}

#[cold]
#[inline(never)]
fn hash_whatever_missing_unicode() -> ! {
    panic!(
        "hash_whatever(GreenType::Unicode, ..): no Unicode \
                 resolver registered — frontend must call \
                 `set_unicode_resolver(eq, hash)` at startup before \
                 any UNICODE green key is hashed (warmstate.py:\
                 115-121 ll_strhash parity)"
    );
}

/// Seed of `JitCell.get_uhash` — `x = r_uint(-1888132534)`
/// (warmstate.py).
pub const GREEN_UHASH_SEED: u64 = (-1888132534_i64) as u64;

/// Multiplier of `JitCell.get_uhash` — `x = (x ^ y) * r_uint(1405695061)`
/// (warmstate.py).
pub const GREEN_UHASH_MULT: u64 = 1405695061;

/// One `get_uhash` fold step over a single green.
///
/// Upstream never materialises the greens: `get_uhash(*greenargs)` unrolls
/// over `green_args_name_spec`, which is fixed per JitCell class at
/// translation time (warmstate.py). A caller that likewise knows its
/// greens statically folds with this instead of building a [`GreenKey`] —
/// [`pypyjit_greenkey_uhash`] for pyre's fixed portal tuple, and the
/// `#[jit_interp]` macro, whose green count and types are known at expansion
/// time.
///
/// Shared with [`GreenKey::get_uhash`] so an unrolled caller and the
/// vector-based key cannot drift apart: a divergence here would file a loop
/// under a key nothing enters, which reads as a performance result rather
/// than a defect.
#[inline(always)]
pub fn green_uhash_step(x: u64, tp: GreenType, value: i64) -> u64 {
    (x ^ hash_whatever(tp, value)).wrapping_mul(GREEN_UHASH_MULT)
}

/// Inline slots for [`GreenKey`] values/types.
///
/// Inline capacity for the declared greens of one jitdriver
/// (`original_boxes[:num_green_args]`). Eight covers a five-green portal
/// (`pc, stackok, is_queue, bm, program`) plus a couple of extra greens
/// without spilling.
pub const GREEN_INLINE: usize = 8;

/// Structured green key — represents the exact values and types of all
/// green variables at a particular program point.
///
/// warmstate.py:564-565 green_args_name_spec — pairs each green arg with
/// its TYPE. comparekey uses equal_whatever(TYPE, ...) and get_uhash uses
/// hash_whatever(TYPE, ...) per RPython.
#[derive(Clone, Debug, Default)]
pub struct GreenKey {
    /// Values of all green variables, in declaration order.
    pub values: SmallVec<[i64; GREEN_INLINE]>,
    /// warmstate.py — per-entry TYPE. Drives hash_whatever/equal_whatever.
    /// `GreenType` (not IR `Type`) so `Ptr to rstr.STR/UNICODE` stays distinct
    /// from generic Ref and is dispatched through ll_streq / ll_strhash.
    pub types: SmallVec<[GreenType; GREEN_INLINE]>,
}

impl PartialEq for GreenKey {
    /// warmstate.py JitCell.comparekey(*greenargs2)
    ///
    /// RPython's comparekey iterates green_args_name_spec (fixed per JitCell
    /// class), comparing each stored attr with the incoming greenarg using
    /// equal_whatever(TYPE, stored, incoming). Both the type spec and the
    /// values must match for equality.
    fn eq(&self, other: &Self) -> bool {
        if self.values.len() != other.values.len() {
            return false;
        }
        if self.types != other.types {
            return false;
        }
        for i in 0..self.values.len() {
            if !equal_whatever(self.types[i], self.values[i], other.values[i]) {
                return false;
            }
        }
        true
    }
}

impl Eq for GreenKey {}

impl std::fmt::Display for GreenKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GreenKey({:?})", self.values)
    }
}

impl std::hash::Hash for GreenKey {
    /// warmstate.py JitCell.get_uhash(*greenargs)
    ///
    /// Delegates to get_uhash() so that HashMap<GreenKey, _> uses the same
    /// hash as jitcounter bucket lookup.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.get_uhash());
    }
}

impl GreenKey {
    /// Create an all-Int green key (most common case: PC-based keys).
    pub fn new(values: impl Into<SmallVec<[i64; GREEN_INLINE]>>) -> Self {
        let values = values.into();
        let mut types = SmallVec::new();
        types.resize(values.len(), GreenType::Int);
        GreenKey { values, types }
    }

    /// warmstate.py:564-565 — typed green key. Accepts either IR-level
    /// [`Type`] (via `From<Type>`) or the richer [`GreenType`].
    pub fn with_types<T: Into<GreenType> + Copy>(
        values: impl Into<SmallVec<[i64; GREEN_INLINE]>>,
        types: impl IntoIterator<Item = T>,
    ) -> Self {
        let values = values.into();
        let types: SmallVec<[GreenType; GREEN_INLINE]> =
            types.into_iter().map(Into::into).collect();
        debug_assert_eq!(values.len(), types.len());
        GreenKey { values, types }
    }

    /// Single Int green key.
    pub fn single(value: i64) -> Self {
        let mut values = SmallVec::new();
        values.push(value);
        let mut types = SmallVec::new();
        types.push(GreenType::Int);
        GreenKey { values, types }
    }

    /// warmstate.py JitCell.get_uhash(*greenargs)
    ///
    /// Exact port of RPython's hash algorithm:
    ///     x = r_uint(-1888132534)
    ///     for _, TYPE in green_args_name_spec:
    ///         y = r_uint(hash_whatever(TYPE, item))
    ///         x = (x ^ y) * r_uint(1405695061)
    ///     return x
    #[inline]
    pub fn get_uhash(&self) -> u64 {
        let mut x: u64 = GREEN_UHASH_SEED;
        for i in 0..self.values.len() {
            let tp = self.types.get(i).copied().unwrap_or(GreenType::Int);
            x = green_uhash_step(x, tp, self.values[i]);
        }
        x
    }

    /// Alias for get_uhash.
    pub fn hash_u64(&self) -> u64 {
        self.get_uhash()
    }
}

/// Allocation-free `get_uhash` for the pypyjit portal green tuple
/// `[next_instr:Int, is_being_profiled:Int, pycode:Ref]`
/// (interp_jit.py:67-70). Equals
/// `GreenKey::with_types(vec![pc, profiled, code], vec![Int, Int, Ref]).get_uhash()`
/// without allocating the key's `values`/`types` vectors — the portal
/// merge-point hook hashes this per back-edge, so the hot path must not
/// allocate (warmstate.py `JitCell.get_uhash`).
pub fn pypyjit_greenkey_uhash(pc: usize, is_being_profiled: bool, code_ptr: u64) -> u64 {
    let mut x: u64 = GREEN_UHASH_SEED;
    x = green_uhash_step(x, GreenType::Int, pc as i64);
    x = green_uhash_step(x, GreenType::Int, is_being_profiled as i64);
    x = green_uhash_step(x, GreenType::Ref, code_ptr as i64);
    x
}

/// The typed pypyjit portal green tuple `[next_instr:Int,
/// is_being_profiled:Int, pycode:Ref]` (interp_jit.py:67-70) that
/// [`pypyjit_greenkey_uhash`] hashes.
///
/// The two are a pair and must stay one: the uhash form is the hot
/// back-edge path (it must not allocate), this form is what a cell read
/// compares against (`JitCell.comparekey`, warmstate.py).  Sites
/// that only need to *find a bucket* take the uhash; sites that need to
/// know *which cell in it* take this.  Building the tuple here rather
/// than open-coding `vec![pc, 0, code]` at each caller is what keeps the
/// slot order and the `Ref` tag on the pycode from drifting apart — a
/// swapped slot still hashes to something, just not to the same cell.
///
/// `is_being_profiled` is a real green, and pyre's callers pass a live value:
/// every green-key construction site reads it off the frame
/// (`call_jit.rs` and `eval.rs` hand `frame.get_is_being_profiled()`, which is
/// `debugdata.is_being_profiled`), so a profiled portal does not share cells
/// with the unprofiled one.
///
/// What that separation buys a `setprofile` armed on an ALREADY-COMPILED loop
/// is a re-warmup, and only that: the green flips, the cell the back edge
/// looks up holds no compiled loop, and execution is interpreted and reporting
/// until the new cell reaches `threshold` and compiles a loop of its own.
/// Measured, a profiler armed mid-loop over a 100 000-iteration tail reports
/// 1041 `call` events for a callee the walker INLINES and all 99 999 for one
/// it leaves residual — the same reading with and without the `debugdata`
/// guard, so the ceiling is the inlined callee's unrecorded `call_trace` /
/// `return_trace` (see `walker_ec_enter`, `jitcode_dispatch/inline_call.rs`)
/// and not the green.  It is also not what protects a profiler installed
/// BEFORE the frame is entered.  That used to be answered by the frame never
/// reaching a trace at all, because the JIT-eligibility gate sent every
/// profiled frame down `execute_frame_plain`; the gate now reads only the
/// per-frame `f_trace`, so a profiled frame does reach a trace — measured on
/// a call-bearing hot loop, `loops_compiled = 1`, `loops_aborted = 5`.  The green is
/// still live at a back edge even so: `setprofile` runs `force_all_frames(True)`,
/// which sets `is_being_profiled` on every frame ALREADY running, and one of
/// those can be sitting in a compiled loop.  Tracing has no green of its own
/// and so gets neither: the mid-loop case is answered instead by the
/// `debugdata` guard `pyopcode.py dispatch_bytecode`'s `jit.we_are_jitted()`
/// arm records.
pub fn pypyjit_greenkey(pc: usize, is_being_profiled: bool, code_ptr: u64) -> GreenKey {
    GreenKey::with_types(
        vec![pc as i64, is_being_profiled as i64, code_ptr as i64],
        vec![GreenType::Int, GreenType::Int, GreenType::Ref],
    )
}

/// Macro-emitted bridge used by `#[jit_interp]` to build a typed
/// `GreenKey` from heterogeneous green expressions.
///
/// Returns the `(i64 bit-representation, GreenType)` pair so the caller
/// emits both vectors in lockstep with a single move of the green value.
/// Integer / bool greens widen as Int (`warmstate.py hash_whatever`
/// equal_int / hash_int);  float greens carry their bit-pattern under
/// Float (equal_float / hash_float);  reference greens widen via raw
/// pointer bits under Ref (Ptr identity — `equal_ptr` compares the bit
/// pattern that `hash_ptr` keyed on).
pub trait GreenAsI64 {
    fn __green_repr(self) -> (i64, GreenType);
}

macro_rules! impl_green_as_i64_int {
    ($($ty:ty),*) => {
        $(
            impl GreenAsI64 for $ty {
                #[inline(always)]
                fn __green_repr(self) -> (i64, GreenType) {
                    (self as i64, GreenType::Int)
                }
            }
        )*
    };
}

impl_green_as_i64_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, bool);

impl GreenAsI64 for f32 {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        // Promote f32 → f64 before extracting the i64 bit pattern.
        // [`equal_whatever`] / [`hash_whatever`] interpret the stored
        // bits as an `f64` (`f64::from_bits(x as u64)`);
        // storing the bare `f32::to_bits()` (a u32 in the low 32 bits
        // of an i64) would round-trip to a subnormal f64 instead of
        // the float value the caller intended.  PyPy / RPython's
        // Float type is always f64 (`lltype.Float`); promoting f32
        // matches that single-width contract.
        ((self as f64).to_bits() as i64, GreenType::Float)
    }
}

impl GreenAsI64 for f64 {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        (self.to_bits() as i64, GreenType::Float)
    }
}

impl<T: ?Sized> GreenAsI64 for &T {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        (
            self as *const T as *const () as usize as i64,
            GreenType::Ref,
        )
    }
}

impl<T: ?Sized> GreenAsI64 for &mut T {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        (
            self as *const T as *const () as usize as i64,
            GreenType::Ref,
        )
    }
}

/// Widens a registered call target's result to the machine word a trace
/// register holds it in.
///
/// A target registered through `#[jit_interp]`'s `calls = { ... }` is
/// dispatched by `majit_backend::call_stub::bh_call_i_dispatch`, which
/// transmutes the recorded address to `extern "C" fn(..) -> i64` and reads the
/// whole return register. Neither the System V AMD64 nor the AAPCS procedure
/// call standard defines the bits above a result narrower than a word, and
/// compilers use that freedom: an x86-64 callee returning `u8` from an array
/// read compiles to `mov al, [rax + rcx]`, leaving the element's own address in
/// the rest of `rax`. Read as an `i64` the result is then an address wearing
/// the returned byte as its low bits, so the damage surfaces far from the call
/// as an out-of-range index rather than as a crash at the call itself.
/// AArch64's `ldrb` zero-extends, which is why the same build is correct
/// there.
///
/// So the macro does not read a callee's return register directly. It routes
/// every int-result target through a shim returning `i64`, and the register is
/// defined by construction. A result that is not one of the types below — a
/// `#[repr(transparent)]` wrapper over a word, say — has to implement this
/// before it can be registered; the widening it wants is not something the
/// macro can guess.
pub trait CallResultWord {
    fn into_call_word(self) -> i64;
}

macro_rules! impl_call_result_word_int {
    ($($ty:ty),*) => {
        $(
            impl CallResultWord for $ty {
                #[inline(always)]
                fn into_call_word(self) -> i64 {
                    self as i64
                }
            }
        )*
    };
}

// `as i64` per type, so each width extends the way the language says it does:
// signed types sign-extend, unsigned types zero-extend, and the word-width
// types are the identity.
impl_call_result_word_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, bool);

impl<T: ?Sized> CallResultWord for *const T {
    #[inline(always)]
    fn into_call_word(self) -> i64 {
        self as *const () as usize as i64
    }
}

impl<T: ?Sized> CallResultWord for *mut T {
    #[inline(always)]
    fn into_call_word(self) -> i64 {
        self as *const () as usize as i64
    }
}

/// Narrows a machine word back to the parameter type a registered call target
/// declares.
///
/// The inverse of [`CallResultWord`], and owed for the same reason at the other
/// end of the call. A compiled call passes every non-float argument as the word
/// a trace register holds, so the shim standing in for the target has to be
/// spelled in words to be the thing that call names. Where a pointer is as wide
/// as a word that is invisible: `usize` and `&T` arrive in a whole register
/// already, and a shim spelled with them is the same function. Where a pointer
/// is narrower — a 32-bit address in a 64-bit register — it is not, and the
/// mismatch is a type error at the call rather than a slow path, so the shim
/// takes words and narrows here.
///
/// A parameter type that is not one of the types below has to implement this
/// before the target can be registered. A struct passed by value cannot: it is
/// not a word, and no narrowing recovers one.
pub trait CallArgWord {
    /// Reconstitute this parameter from the word a compiled call passes for it.
    ///
    /// # Safety
    ///
    /// `word` must be what the call holds for this parameter. For a reference
    /// or a raw pointer that means an address of the right type which stays
    /// live for the call, exactly as the caller of the target itself owes.
    unsafe fn from_call_word(word: i64) -> Self;
}

macro_rules! impl_call_arg_word_int {
    ($($ty:ty),*) => {
        $(
            impl CallArgWord for $ty {
                #[inline(always)]
                unsafe fn from_call_word(word: i64) -> Self {
                    word as $ty
                }
            }
        )*
    };
}

// `as $ty` per type, the exact inverse of the widening `into_call_word` did:
// a narrower type keeps the low bits the widening preserved, and the
// word-width types are the identity.
impl_call_arg_word_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl CallArgWord for bool {
    #[inline(always)]
    unsafe fn from_call_word(word: i64) -> Self {
        // `into_call_word` produces 0 or 1, but a target may be reached with a
        // word another producer widened, so read the whole word rather than
        // its low bit.
        word != 0
    }
}

// Only sized pointees: a word carries one address, and a fat pointer's
// metadata is not in it.
impl<T> CallArgWord for *const T {
    #[inline(always)]
    unsafe fn from_call_word(word: i64) -> Self {
        word as usize as *const T
    }
}

impl<T> CallArgWord for *mut T {
    #[inline(always)]
    unsafe fn from_call_word(word: i64) -> Self {
        word as usize as *mut T
    }
}

impl<T> CallArgWord for &T {
    #[inline(always)]
    unsafe fn from_call_word(word: i64) -> Self {
        unsafe { &*(word as usize as *const T) }
    }
}

impl<T> CallArgWord for &mut T {
    #[inline(always)]
    unsafe fn from_call_word(word: i64) -> Self {
        unsafe { &mut *(word as usize as *mut T) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_const_pool_minor_scan_is_a_one_shot_write_barrier() {
        let pool = SharedConstPool::new(vec![Const::from_gcref(GcRef(0x1234))]);
        assert!(pool.take_minor_scan_pending());
        assert!(!pool.take_minor_scan_pending());

        // A replacement pool represents a new `rd_consts` assignment and
        // therefore starts with its own remembered young-edge state.
        let replacement = SharedConstPool::new(Vec::new());
        assert!(replacement.take_minor_scan_pending());
    }

    #[test]
    fn inputarg_rc_leaves_the_64_byte_class() {
        assert!(
            std::mem::size_of::<InputArg>() <= 32,
            "InputArg is {} bytes; keep the payload at 32 B",
            std::mem::size_of::<InputArg>()
        );
        assert!(
            std::mem::size_of::<InputArgInner>() <= 40,
            "InputArgInner is {} B; from_type_rc must stay out of the 48-byte class",
            std::mem::size_of::<InputArgInner>()
        );
    }

    #[test]
    fn test_type_roundtrip() {
        for tp in [Type::Int, Type::Ref, Type::Float, Type::Void] {
            assert_eq!(Type::from_char(tp.to_char()), tp);
        }
    }

    #[test]
    fn test_value_types() {
        assert_eq!(Value::Int(42).get_type(), Type::Int);
        assert_eq!(Value::Float(std::f64::consts::PI).get_type(), Type::Float);
        assert_eq!(Value::Ref(GcRef::NULL).get_type(), Type::Ref);
        assert_eq!(Value::Void.get_type(), Type::Void);
    }

    #[test]
    fn test_gcref_null() {
        assert!(GcRef::NULL.is_null());
        assert!(!GcRef(0x1234).is_null());
    }

    #[test]
    fn test_var_kind() {
        let green = JitDriverVar::green("pc", Type::Int);
        assert_eq!(green.kind, VarKind::Green);
        assert_eq!(green.name, "pc");

        let red = JitDriverVar::red("stack", Type::Ref);
        assert_eq!(red.kind, VarKind::Red);
        assert_eq!(red.name, "stack");
    }

    #[test]
    fn test_green_key_hash() {
        let k1 = GreenKey::single(42);
        let k2 = GreenKey::single(42);
        let k3 = GreenKey::single(43);

        assert_eq!(k1.hash_u64(), k2.hash_u64());
        assert_ne!(k1.hash_u64(), k3.hash_u64());
    }

    /// An unrolled [`green_uhash_step`] fold must equal
    /// [`GreenKey::get_uhash`] over the same greens, for every arity and
    /// type mix.
    ///
    /// This is the contract that lets a caller who knows its greens
    /// statically skip building the key's `values` / `types` vectors —
    /// upstream's shape, where `get_uhash(*greenargs)` unrolls over a
    /// per-JitCell-class spec (warmstate.py). The `#[jit_interp]`
    /// macro emits exactly this fold.
    ///
    /// It is a correctness test, not a performance one: the two spellings
    /// disagreeing would file a loop under a key nothing enters, so
    /// `has_compiled_loop` misses forever and every back edge re-arms
    /// tracing. That surfaces as "the JIT got slower", not as a failure —
    /// which is why the equality is pinned rather than assumed.
    ///
    /// `Str` / `Unicode` are excluded here because `hash_whatever` routes
    /// them through a resolver a frontend must register first; they are
    /// covered in `tests/green_key_str_content_equality.rs`, which does
    /// register one.
    #[test]
    fn unrolled_fold_equals_greenkey_get_uhash() {
        // Deterministic sweep — no RNG seed to drift, and every case is
        // reproducible by index. Boundary values first: `hash_whatever`
        // special-cases 0 for Ref, and Float reinterprets the bits.
        let sample_values: [i64; 9] = [
            0,
            1,
            -1,
            42,
            i64::MAX,
            i64::MIN,
            (1.0_f64).to_bits() as i64,
            (-0.0_f64).to_bits() as i64,
            (f64::NAN).to_bits() as i64,
        ];
        let sample_types = [
            GreenType::Int,
            GreenType::Float,
            GreenType::Ref,
            GreenType::Void,
        ];

        let mut checked = 0usize;
        // Arity 0 (the empty key) through 8, covering every type at every
        // position via an offset walk over both sample tables.
        for arity in 0..=8usize {
            for offset in 0..sample_values.len() * sample_types.len() {
                let mut values = Vec::with_capacity(arity);
                let mut types = Vec::with_capacity(arity);
                for i in 0..arity {
                    values.push(sample_values[(offset + i) % sample_values.len()]);
                    types.push(sample_types[(offset + 3 * i) % sample_types.len()]);
                }

                // The unrolled form: what a static caller emits.
                let mut unrolled = GREEN_UHASH_SEED;
                for i in 0..arity {
                    unrolled = green_uhash_step(unrolled, types[i], values[i]);
                }

                let via_key = GreenKey::with_types(values.clone(), types.clone()).get_uhash();
                assert_eq!(
                    unrolled, via_key,
                    "unrolled fold diverged from GreenKey::get_uhash at \
                     arity={arity} offset={offset} values={values:?} types={types:?}",
                );
                checked += 1;
            }
        }

        // Guard the guard: an empty sweep would assert nothing.
        assert_eq!(
            checked,
            9 * 36,
            "sweep did not run the expected number of cases",
        );
    }

    #[test]
    fn green_repr_returns_per_type_green_type() {
        assert_eq!(7i64.__green_repr(), (7i64, GreenType::Int));
        assert_eq!(7u32.__green_repr(), (7i64, GreenType::Int));
        assert_eq!(true.__green_repr(), (1i64, GreenType::Int));

        let (fv, ft) = std::f64::consts::PI.__green_repr();
        assert_eq!(fv, std::f64::consts::PI.to_bits() as i64);
        assert_eq!(ft, GreenType::Float);

        let s: &'static str = "abc";
        let (rv, rt) = s.__green_repr();
        assert_eq!(rv, s as *const str as *const () as usize as i64);
        assert_eq!(rt, GreenType::Ref);
    }

    #[test]
    fn green_repr_f32_promotes_to_f64_for_consistent_hash() {
        // `equal_whatever(Float, x, y)` interprets bits as `f64`
        // (the same `f64::from_bits` read).  An f32 green that stored
        // only its 32-bit
        // pattern would round-trip to a subnormal f64; promoting to
        // f64 first keeps the i64 representation consistent with the
        // f64 green carrying the same numeric value.
        let f32_val: f32 = 1.5;
        let f64_val: f64 = 1.5;
        let (f32_bits, _) = f32_val.__green_repr();
        let (f64_bits, _) = f64_val.__green_repr();
        assert_eq!(f32_bits, f64_bits);

        // Round-trip via `equal_whatever(Float, _, _)` confirms both
        // evaluate to the same f64 value (1.5 == 1.5).
        assert!(equal_whatever(GreenType::Float, f32_bits, f64_bits));
    }

    #[test]
    fn test_green_key_multi() {
        let k1 = GreenKey::new(vec![10, 20, 30]);
        let k2 = GreenKey::new(vec![10, 20, 30]);
        let k3 = GreenKey::new(vec![10, 20, 31]);

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
        assert_eq!(k1.hash_u64(), k2.hash_u64());
    }

    #[test]
    fn test_green_type_from_type() {
        assert_eq!(GreenType::from(Type::Int), GreenType::Int);
        assert_eq!(GreenType::from(Type::Ref), GreenType::Ref);
        assert_eq!(GreenType::from(Type::Float), GreenType::Float);
        assert_eq!(GreenType::from(Type::Void), GreenType::Void);
    }

    /// The allocation-free portal hash must be byte-identical to building
    /// the typed `GreenKey([pc, profiled, code], [Int, Int, Ref])` and
    /// calling `get_uhash` — that equivalence is what lets the legacy
    /// `make_green_key` u64 flow and the typed marker path agree on the
    /// same cell.
    #[test]
    fn test_pypyjit_greenkey_uhash_matches_get_uhash() {
        let cases: &[(usize, bool, u64)] = &[
            (0, false, 0),
            (0, false, 0x1000),
            (7, false, 0xdead_beef),
            (7, true, 0xdead_beef),
            (123456, false, 0x7fff_ffff_ffff_fff0),
            (1, true, 1),
            // Width boundaries: `pc` widens usize -> i64 and `code_ptr`
            // u64 -> i64, so a sign-extension slip shows up here and
            // nowhere in the cases above.
            (usize::MAX, false, u64::MAX),
            (usize::MAX, true, 0),
        ];
        for &(pc, profiled, code) in cases {
            let key = GreenKey::with_types(
                vec![pc as i64, profiled as i64, code as i64],
                vec![GreenType::Int, GreenType::Int, GreenType::Ref],
            );
            assert_eq!(
                pypyjit_greenkey_uhash(pc, profiled, code),
                key.get_uhash(),
                "uhash mismatch for (pc={pc}, profiled={profiled}, code={code:#x})"
            );
            // The constructor is checked against the hand-written literal
            // above, not against the uhash it is paired with: comparing the
            // two functions to each other would pass just as happily if both
            // agreed on a wrong slot order. `values` and `types` are asserted
            // separately because `GreenKey`'s `PartialEq` routes through
            // `equal_whatever`, which is exactly the layout-sensitive thing
            // under test.
            let built = pypyjit_greenkey(pc, profiled, code);
            assert_eq!(
                built.values, key.values,
                "greenkey values mismatch for (pc={pc}, profiled={profiled}, code={code:#x})"
            );
            assert_eq!(
                built.types, key.types,
                "greenkey types mismatch for (pc={pc}, profiled={profiled}, code={code:#x})"
            );
        }
    }

    /// Each width extends the way the language says it does, so a caller
    /// reading the whole register recovers the value the callee returned and
    /// not a sign-flipped or truncated one.
    #[test]
    fn call_result_words_extend_by_signedness() {
        assert_eq!((-1i8).into_call_word(), -1);
        assert_eq!(0xffu8.into_call_word(), 255);
        assert_eq!((-1i32).into_call_word(), -1);
        assert_eq!(u32::MAX.into_call_word(), 4_294_967_295);
        assert_eq!(i64::MIN.into_call_word(), i64::MIN);
        assert_eq!(usize::MAX.into_call_word(), -1);
        assert_eq!(true.into_call_word(), 1);
        assert_eq!(false.into_call_word(), 0);
    }

    #[test]
    fn call_result_words_carry_a_pointer_whole() {
        let x = 7u32;
        let p: *const u32 = &x;
        assert_eq!(p.into_call_word() as usize, p as usize);
    }
}
