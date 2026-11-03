//! W_LongObject -- arbitrary-precision integer backed by RPython `rbigint`.
//!
//! Used when i64 overflow is detected in `W_IntObject` arithmetic.
//! The JIT may specialize bigint operations by reading immutable `value`
//! payloads, calling pure raw-payload helpers, and boxing the resulting payload
//! with the same `W_LongObject` layout.

use majit_rlib::rbigint::RBigInt as BigInt;

use crate::pyobject::*;

/// Arbitrary-precision integer object.
///
/// Layout: `[ob_type: *const PyType | value: *mut RBigInt]`
/// The `value` pointer references an immutable `rbigint` payload, usually
/// GC-managed and occasionally leaked via `malloc_raw` before GC init.
#[repr(C)]
pub struct W_LongObject {
    pub ob_header: PyObject,
    pub value: *mut BigInt,
}

// Safety: RBigInt is Send+Sync and W_LongObject only stores a raw pointer
// that is effectively owned.
unsafe impl Send for W_LongObject {}
unsafe impl Sync for W_LongObject {}

/// `typedef.py` `_getusercls(W_LongObject)`. The base payload stays at
/// offset 0, including the immutable `value` pointer; `MapdictStorageMixin`
/// contributes `map` and `storage`.
#[repr(C)]
pub struct W_LongObjectUser {
    pub base: W_LongObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_LongObjectUser, storage)
            == std::mem::offset_of!(W_LongObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// User-subclass big-int layout. Appended after the closed ids, ahead of
/// the target-gated tail (`build_gc`).
pub const W_LONG_USER_GC_TYPE_ID: u32 = 200;
pub const W_LONG_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_LongObjectUser>();

impl crate::lltype::GcType for W_LongObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_LONG_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_LONG_USER_OBJECT_SIZE;
}

/// Field offset of `value` within `W_LongObject`, for potential JIT field access.
pub const LONG_VALUE_OFFSET: usize = std::mem::offset_of!(W_LongObject, value);

/// GC type id assigned to `W_LongObject` at JitDriver init time.
pub const W_LONG_GC_TYPE_ID: u32 = 35;

/// Fixed payload size (`framework.py:811`).
pub const W_LONG_OBJECT_SIZE: usize = std::mem::size_of::<W_LongObject>();

impl crate::lltype::GcType for W_LongObject {
    fn type_id() -> u32 {
        W_LONG_GC_TYPE_ID
    }
    const SIZE: usize = W_LONG_OBJECT_SIZE;
}

/// Payload size of a raw `rbigint` GC object.
pub const BIGINT_PAYLOAD_SIZE: usize = majit_rlib::rbigint::RBIGINT_PAYLOAD_SIZE;

/// Translated rbigint GC-reference result ABI.
///
/// Native LLBC extraction must retain the pointer result so the codewriter
/// models it as a GcRef. wasm32 direct residual calls use the JIT's uniform
/// i64 word family, so encode the same pointer in an i64 on that target.
#[cfg(not(target_arch = "wasm32"))]
pub type JitBigIntResult = *mut BigInt;
#[cfg(target_arch = "wasm32")]
pub type JitBigIntResult = i64;

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn encode_jit_bigint_result(value: *mut BigInt) -> JitBigIntResult {
    value
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn encode_jit_bigint_result(value: *mut BigInt) -> JitBigIntResult {
    value as usize as i64
}

/// Payload size and traced-field offsets of the `tuple2` of two rbigints.
pub const BIGINT_PAIR_SIZE: usize = majit_rlib::rbigint::RBIGINT_PAIR_SIZE;
pub const BIGINT_PAIR_ITEM0_OFFSET: usize = majit_rlib::rbigint::RBIGINT_PAIR_ITEM0_OFFSET;
pub const BIGINT_PAIR_ITEM1_OFFSET: usize = majit_rlib::rbigint::RBIGINT_PAIR_ITEM1_OFFSET;

/// Same GC-reference result ABI as [`JitBigIntResult`], for the `tuple2` a
/// two-result elidable returns.
#[cfg(not(target_arch = "wasm32"))]
pub type JitBigIntPairResult = *mut majit_rlib::rbigint::RBigIntPair;
#[cfg(target_arch = "wasm32")]
pub type JitBigIntPairResult = i64;

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn encode_jit_bigint_pair_result(
    value: *mut majit_rlib::rbigint::RBigIntPair,
) -> JitBigIntPairResult {
    value
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn encode_jit_bigint_pair_result(
    value: *mut majit_rlib::rbigint::RBigIntPair,
) -> JitBigIntPairResult {
    value as usize as i64
}

/// Record the GC type id registered for the `tuple2` of two rbigints (called
/// once from `pyre-jit::eval` after `gc.register_type`).
pub fn set_bigint_pair_gc_type_id(id: u32) {
    majit_rlib::rbigint::set_rbigint_pair_gc_type_id(id);
}

/// Allocate the `tuple2` owning both halves of a divmod result.
#[inline]
pub fn alloc_bigint_pair_nursery_collecting(
    item0: BigInt,
    item1: BigInt,
) -> *mut majit_rlib::rbigint::RBigIntPair {
    majit_rlib::rbigint::alloc_rbigint_pair_nursery_collecting(item0, item1)
}

// GC type id for the raw `rbigint` payload is published at JitDriver init by
// `set_bigint_gc_type_id`. It remains 0 until then, in which case the alloc
// helpers fall back to leaked raw allocations in bare tests/pre-init bootstrap.

/// Record the GC type id registered for the `rbigint` payload (called once from
/// `pyre-jit::eval` after `gc.register_type`).
pub fn set_bigint_gc_type_id(id: u32) {
    majit_rlib::rbigint::set_rbigint_gc_type_id(id);
}

/// Reads the runtime-assigned rbigint type id (set once at init by
/// [`set_bigint_gc_type_id`]); the value is not a build-time constant, so the
/// JIT residualises the read instead of tracing into it (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub fn bigint_gc_type_id() -> u32 {
    majit_rlib::rbigint::rbigint_gc_type_id()
}

/// Allocate `value` as a GC-managed `rbigint` in the nursery (no-collect host
/// path). The caller must immediately store the result into a traced owner:
/// `w_long_new` uses a non-collecting old-gen wrapper allocation plus a
/// creation barrier, while JIT `*_raw` helpers flow straight into the boxing
/// `NewWithVtable`. Falls back to leaked `malloc_raw` when no GC hook is
/// installed (bare unit tests, where the result is never traced).
#[inline]
pub fn alloc_bigint_nursery(value: BigInt) -> *mut BigInt {
    majit_rlib::rbigint::alloc_rbigint_nursery(value)
}

/// Allocate `value` as a GC-managed `rbigint` through the collecting nursery.
/// a minor collection fires when the nursery is full (reclaiming dead bigints)
/// instead of spilling to old-gen unbounded. Only for the elidable bigint
/// payload helpers (`jit_bigint_*`), which the walker emits as a residual
/// `CallR` whose gcmap roots the trace's live set, and which read both operand
/// payloads into a local sum before allocating — so nothing unrooted is held
/// across the embedded minor cycle. Falls back to the no-collect path when no
/// collecting hook is installed (other backends), then to `malloc_raw`.
/// The result's digits are an ordinary nursery `GcArray(Signed)`, so their bytes
/// already participate in nursery pressure; no foreign/external-memory charge
/// is needed.
#[inline]
pub fn alloc_bigint_nursery_collecting(value: BigInt) -> *mut BigInt {
    majit_rlib::rbigint::alloc_rbigint_nursery_collecting(value)
}

/// Allocate `value` as a GC-managed `rbigint` at a stable (old-gen, non-moving)
/// address for callers that must retain a raw payload pointer across a
/// potentially collecting operation. Falls back to the leaked `malloc_raw`
/// pre-init.
#[inline]
pub fn alloc_bigint_stable(value: BigInt) -> *mut BigInt {
    majit_rlib::rbigint::alloc_rbigint_stable(value)
}

/// Wrap an already heap-allocated `*mut BigInt` in a fresh W_LongObject
/// without copying the payload — the wrapper just stores `value`, it does not
/// take exclusive ownership. Pure-call CSE of the elidable `rbigint` helpers
/// can fold two ops to the same `*mut BigInt`, so one payload may back more
/// than one wrapper; that is sound because payloads are immutable after
/// initialization and every wrapper/trace op treats the payload as a GC ref.
pub fn w_long_from_raw(value: *mut BigInt) -> PyObjectRef {
    // W_LongObject shares the `int` type with W_IntObject — the two only
    // differ in their storage layout, not their Python-level identity
    // (PyPy does the same via W_AbstractIntObject's typedef). Wire
    // `w_class` to INT_TYPE.instantiate so `type(x) is int` and
    // `isinstance(x, int)` both hold for long integers.
    // PyPy's `W_LongObject.__init__` stores only `num`; the allocation itself
    // is introduced later by `rclass.py`/`rbuiltin.py`.  Keep the Rust host's
    // non-moving requirement inside the malloc implementation rather than
    // spelling its allocator branch and aggregate copy in this constructor.
    // `fuse_boxing_alloc` recognizes this GC-flavoured malloc boundary and
    // restores the translated `NewWithVtable` plus ordinary field writes.
    // The wrapper malloc is the single allocation after the payload is
    // already a GC object; `gct_fv_gc_malloc` would push `num` as a livevar
    // here, but a no-collect old-gen bump does not move it. A registry root
    // (`gc_add_root` / owner-root) on this path is not that livevar.
    crate::lltype::malloc_typed_stable(W_LongObject {
        ob_header: PyObject {
            ob_type: &LONG_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        value,
    }) as PyObjectRef
}

/// Subclass wrapper for a big `int` (`intobject.py` `_new_int` →
/// `space.allocate_instance(W_LongObject, w_inttype)`). The layout is
/// `typedef.py` `_getusercls`: `W_LongObjectUser` shares the immutable
/// rbigint pointer and adds mapdict storage. The instance is nursery-managed
/// so a subtype `__del__` can run; exact big ints stay on [`w_long_from_raw`].
pub fn w_long_subclass_from_raw(value: *mut BigInt) -> PyObjectRef {
    let obj = W_LongObjectUser {
        base: W_LongObject {
            ob_header: PyObject {
                ob_type: &LONG_USER_TYPE as *const PyType,
                w_class: get_instantiate(&INT_TYPE),
            },
            value,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_LONG_USER_GC_TYPE_ID,
        std::mem::size_of::<W_LongObjectUser>(),
    );
    if raw.is_null() {
        crate::lltype::malloc_typed(obj) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_LongObjectUser, obj);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    }
}

/// Allocate a new W_LongObject on the heap from a `BigInt` value. The bigint
/// payload is a fresh nursery object, matching RPython's ordinary GC
/// allocation. The wrapper itself is born in non-moving old-gen and its
/// creation barrier records the old→young `LONG_VALUE_OFFSET` edge. The
/// collecting payload malloc roots the handle's `_digits` slot
/// (`alloc_fast_nursery_collecting_typed_rooted`); the wrapper malloc does
/// not collect, so the returned payload pointer needs no second registry.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the body
/// delegates through `alloc_bigint_nursery_collecting -> *mut BigInt`, so
/// tracing into it leaks the raw `BigInt` pointee into the return model and
/// unifies against the `w_int_new` fast path as `PyObject ∪ BigInt`.
/// Residualising the whole box tail models it by signature — a plain
/// `PyObjectRef` GCREF with no discriminant to erase — the
/// `w_str_new`/`box_bigint_constant` twin.
#[majit_macros::dont_look_inside]
pub fn w_long_new(value: BigInt) -> PyObjectRef {
    w_long_from_raw(alloc_bigint_nursery_collecting(value))
}

/// Wrap a fresh rbigint handle without canonicalizing a shallow copy whose
/// digit array happens to belong to a prebuilt value.
#[majit_macros::dont_look_inside]
pub fn w_long_new_fresh_rbigint_handle(value: BigInt) -> PyObjectRef {
    // Host counterpart of `jit_bigint_clone`: rbigint.neg always constructs
    // a new handle, including for the prebuilt zero digit array. The clone
    // allocator roots `_digits` across its own malloc.
    w_long_from_raw(majit_rlib::rbigint::alloc_rbigint_clone_nursery_collecting(
        value,
    ))
}

/// Create a W_LongObject from an i64 value.
pub fn w_long_from_i64(v: i64) -> PyObjectRef {
    w_long_new(BigInt::from(v))
}

/// Box a bigint constant into a heap Python int object.
pub fn box_bigint_constant(value: &BigInt) -> PyObjectRef {
    w_long_new(value.clone())
}

/// `rbigint.fromint()` (`rpython/rlib/rbigint.py`,
/// `@jit.elidable`) with the translated one-GC-reference ABI.
///
/// Rust returns `RBigInt` by value, but RPython returns a GC reference.  The
/// MIR front retargets machine-word constructor calls here so generated JIT
/// code receives a rooted `*mut RBigInt`, encoded in the JIT's uniform i64
/// word ABI on wasm32 rather than returned as a native wasm pointer.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_from_i64(value: i64) -> JitBigIntResult {
    encode_jit_bigint_result(alloc_bigint_nursery_collecting(BigInt::fromint(value)))
}

/// Unsigned machine-word companion of [`jit_bigint_from_i64`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_from_u64(value: u64) -> JitBigIntResult {
    encode_jit_bigint_result(alloc_bigint_nursery_collecting(BigInt::from_u128(
        value as u128,
    )))
}

/// Rust handle clone for the translated one-GC-reference rbigint shape.
///
/// `RBigInt::clone` shares the immutable digit array but returns a new by-value
/// handle. Translation therefore needs a fresh GC payload containing that
/// shallow handle, rather than either descending into the classdef-less Rust
/// fields or collapsing the clone to reference identity.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_clone(value: *const BigInt) -> JitBigIntResult {
    unsafe {
        encode_jit_bigint_result(majit_rlib::rbigint::alloc_rbigint_clone_nursery_collecting(
            (*value).clone(),
        ))
    }
}

macro_rules! bigint_comparison_residual {
    ($name:ident, $method:ident) => {
        #[doc = "Bare-RBigInt comparison residual using the translated GC-reference ABI."]
        #[majit_macros::elidable_cannot_raise]
        pub extern "C" fn $name(a: *const BigInt, b: *const BigInt) -> i64 {
            unsafe { BigInt::$method(&*a, &*b) as i64 }
        }
    };
}

bigint_comparison_residual!(jit_bigint_eq, eq);
bigint_comparison_residual!(jit_bigint_ne, ne);
bigint_comparison_residual!(jit_bigint_lt, lt);
bigint_comparison_residual!(jit_bigint_le, le);
bigint_comparison_residual!(jit_bigint_gt, gt);
bigint_comparison_residual!(jit_bigint_ge, ge);

macro_rules! bigint_scalar_residual {
    ($name:ident, $body:expr) => {
        #[doc = "Bare-RBigInt scalar residual using the translated GC-reference ABI."]
        #[majit_macros::elidable_cannot_raise]
        pub extern "C" fn $name(value: *const BigInt) -> i64 {
            let value = unsafe { &*value };
            ($body)(value) as i64
        }
    };
}

bigint_scalar_residual!(jit_bigint_bits, |value: &BigInt| value.bits());
bigint_scalar_residual!(jit_bigint_is_zero, |value: &BigInt| value.is_zero());
bigint_scalar_residual!(jit_bigint_is_one, |value: &BigInt| value.is_one());
bigint_scalar_residual!(jit_bigint_tobool, |value: &BigInt| value.tobool());
bigint_scalar_residual!(jit_bigint_hash, |value: &BigInt| value.hash());

/// `W_LongObject._fits_int()` — longobject.py / rbigint.fits_int.
/// True if the value fits in a machine-word integer (i64 on 64-bit).
/// Used by `is_plain_int1` to accept long objects that are in the int range.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_long_fits_int(obj: PyObjectRef) -> bool {
    unsafe {
        let big = w_long_get_value(obj);
        jit_bigint_to_i64_fits(big) != 0
    }
}

/// True when the W_LongObject's BigInt is zero. Divisor guard for the
/// can-raise floordiv/mod fast path (a zero divisor makes the payload helper
/// publish ZeroDivisionError, which the trait path defers to the generic
/// residual rather than triggering during tracing).
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_is_zero(obj: PyObjectRef) -> bool {
    unsafe { w_long_get_value(obj).get_sign() == 0 }
}

/// Extract a reference to the BigInt value from a known W_LongObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_get_value(obj: PyObjectRef) -> &'static BigInt {
    unsafe {
        let long_obj = obj as *const W_LongObject;
        &*(*long_obj).value
    }
}

/// Return the translated rbigint GC reference stored in a long wrapper.
///
/// PyPy's `rbigint.abs` returns `self` for a nonnegative value and
/// `W_LongObject.descr_abs` then constructs only a new Python wrapper around
/// that same payload. Consumers implementing that path need the stored
/// reference itself, not a Rust `Clone` that translates to a fresh GC payload.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_long_get_raw_value(obj: PyObjectRef) -> *mut BigInt {
    unsafe { (*(obj as *const W_LongObject)).value }
}

/// `rbigint.fits_int()` (`rpython/rlib/rbigint.py`) — JIT-callable
/// wrapper. Returns 1 when the W_LongObject's BigInt fits in i64,
/// 0 otherwise. Used as the runtime fits_int guard before
/// `jit_w_long_toint`.
///
/// Unlike `rbigint.toint()`, upstream `fits_int()` is not marked
/// `@jit.elidable`, so keep this call cannot-raise but non-elidable.
pub extern "C" fn jit_w_long_fits_int(obj: PyObjectRef) -> i64 {
    unsafe { w_long_fits_int(obj) as i64 }
}

/// `rbigint.fits_int()` on a bare `*mut BigInt` — the demote guard for the
/// inline-NEW boxing of a `jit_w_long_*_raw` result. Returns 1 when the bigint
/// fits i64 (i.e. should demote to `W_IntObject`), 0 otherwise. The walker/trait
/// emit `GuardFalse(fits)` after the raw op so a result that does fit deopts to
/// the interpreter (which performs the demote); the common bigint case (does
/// not fit) passes the guard and falls through to `NewWithVtable(W_LONG)`.
/// Non-elidable, cannot-raise (mirrors [`jit_w_long_fits_int`]).
///
/// The pointer is a live GC bigint produced by a preceding raw op in the same trace.
pub extern "C" fn jit_bigint_fits_int(num: *const BigInt) -> i64 {
    unsafe { jit_bigint_to_i64_fits(&*num) }
}

/// `rbigint.fits_int()` (`rpython/rlib/rbigint.py`) on a borrowed
/// BigInt payload. Scalar half of the `BigInt::to_i64()` split used by
/// the two-phase rtyper so it never has to model an `Option<i64>` ABI.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_i64_fits(num: &BigInt) -> i64 {
    num.fits_int() as i64
}

/// `rbigint.toint()` (`rpython/rlib/rbigint.py`, `@jit.elidable`) on a
/// borrowed BigInt payload. Callers must first check
/// [`jit_bigint_to_i64_fits`]; overflow means that guard was violated.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_i64_value(num: &BigInt) -> i64 {
    num.toint().unwrap_or_else(|_| {
        panic!("jit_bigint_to_i64_value: BigInt out of i64 range - fits guard violated")
    })
}

/// `rbigint.toint()` payload for the runtime-fallible narrowing lowering:
/// the value is read only on the fits (Ok) path, so out-of-range returns 0
/// instead of panicking — the walker's discriminant switch discards it on
/// the Err path. Companion of [`jit_bigint_to_i64_fits`].
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_i64_value_or_zero(num: &BigInt) -> i64 {
    num.toint().unwrap_or(0)
}

/// `rbigint.touint()` on a borrowed BigInt payload. Scalar half of the
/// fallible conversion so the two-phase rtyper never has to model a
/// `Result<u64, RBigIntError>` ABI. Companion of [`jit_bigint_to_u64_value`].
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_u64_fits(num: &BigInt) -> i64 {
    num.touint().is_ok() as i64
}

/// `rbigint.touint()` on a borrowed BigInt payload. Callers must first check
/// [`jit_bigint_to_u64_fits`]; an error here means that guard was violated.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_u64_value(num: &BigInt) -> u64 {
    num.touint().unwrap_or_else(|_| {
        panic!("jit_bigint_to_u64_value: BigInt exceeds u64 range - fits guard violated")
    })
}

/// `rbigint.sign` / sign-digit use (`rpython/rlib/rbigint.py`) on a borrowed
/// RBigInt payload. Returns the scalar signum (-1, 0, +1) so the two-phase
/// rtyper does not need a synthetic Rust enum ABI at this boundary.
#[majit_macros::elidable_cannot_raise]
pub fn jit_bigint_sign_i64(num: &BigInt) -> i64 {
    num.get_sign()
}

/// `rbigint.tofloat()` (`rpython/rlib/rbigint.py`) on a borrowed BigInt
/// payload, with the caller's existing overflow sentinel folded into the
/// scalar return.
#[majit_macros::elidable_cannot_raise]
pub fn jit_bigint_to_f64_or_inf(num: &BigInt) -> f64 {
    num.tofloat().unwrap_or(f64::INFINITY)
}

/// `rbigint.tofloat()` (`rpython/rlib/rbigint.py`) on a borrowed BigInt
/// payload, preserving callers that intentionally collapse overflow to NaN.
#[majit_macros::elidable_cannot_raise]
pub fn jit_bigint_to_f64_or_nan(num: &BigInt) -> f64 {
    num.tofloat().unwrap_or(f64::NAN)
}

/// `W_LongObject.toint()` (`pypy/objspace/std/longobject.py`) →
/// `rbigint.toint()` (`rpython/rlib/rbigint.py`, `@jit.elidable`).
/// Extract an i64 from a W_LongObject. RPython `toint` raises
/// `OverflowError` when the BigInt does not fit; the elidable
/// trace-time site emits a `fits_int` GUARD_TRUE first
/// (`pypy/objspace/std/listobject.py is_plain_int1` parity), so
/// the OverflowError path is unreachable in production. Pyre encodes
/// that unreachability as a panic. There is no `_int_w_unsafe` upstream —
/// this is the elidable `toint` after a `fits_int` guard.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_toint(obj: PyObjectRef) -> i64 {
    unsafe {
        let big = w_long_get_value(obj);
        i64::try_from(big).unwrap_or_else(|_| {
            panic!("jit_w_long_toint: BigInt out of i64 range — fits_int guard violated")
        })
    }
}

/// `rbigint.add` (`rpython/rlib/rbigint.py`, `@jit.elidable`) — the
/// payload half of `W_LongObject._add` (`pypy/objspace/std/longobject.py`).
/// Both operands are guaranteed `W_LongObject` by a preceding
/// `GuardClass(LONG_TYPE)` on each, so the BigInt payloads are read
/// directly. Returns a freshly heap-allocated `*mut BigInt` — the
/// arithmetic only, with no Python-object wrapper. `add` allocates a new
/// bigint, so its only failure mode is MemoryError: `EF_ELIDABLE_OR_MEMORYERROR`
/// (`call.py`, `cr == "mem"`). The value is still a pure function of the
/// operand payloads, so the optimizer may fold/CSE it; a trailing
/// `GuardNoException` covers the allocation. The result is an internal bigint
/// never exposed to Python `is`, so sharing one payload for two equal-input
/// adds is unobservable.
/// Wrapper-level (`W_LongObject` operands) variants used for record-time
/// concrete evaluation and the trait path. The digit malloc inside
/// `rbigint.add`/`sub`/`mul`/`and_`/`or_`/`xor` collects, and those bodies root
/// the operand digit arrays across it (the `push_roots(livevars)` bracket
/// around `rbigint.py add`); the result wrapper comes from the no-collect
/// `alloc_bigint_nursery`. The
/// operand wrappers are not read after the digit malloc here, so a caller that
/// holds them natively across this call must root them itself.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_add_raw(a: PyObjectRef, b: PyObjectRef) -> *mut BigInt {
    unsafe {
        let av = w_long_get_raw_value(a);
        let bv = w_long_get_raw_value(b);
        if (*av).get_sign() == 0 {
            return bv;
        }
        if (*bv).get_sign() == 0 {
            return av;
        }
        // Digit malloc is the collecting call; `_x_add` / `_x_sub` root the
        // operand digit arrays across it (`push_roots(livevars)`).
        alloc_bigint_nursery(BigInt::add(&*av, &*bv))
    }
}

/// `rbigint.sub` over `W_LongObject` operands (collecting digit malloc). See
/// [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_sub_raw(a: PyObjectRef, b: PyObjectRef) -> *mut BigInt {
    unsafe {
        let av = w_long_get_raw_value(a);
        let bv = w_long_get_raw_value(b);
        if (*bv).get_sign() == 0 {
            return av;
        }
        alloc_bigint_nursery(BigInt::sub(&*av, &*bv))
    }
}

/// `rbigint.mul` over `W_LongObject` operands (collecting digit malloc). See
/// [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_mul_raw(a: PyObjectRef, b: PyObjectRef) -> *mut BigInt {
    unsafe {
        alloc_bigint_nursery(BigInt::mul(
            &*w_long_get_raw_value(a),
            &*w_long_get_raw_value(b),
        ))
    }
}

/// `rbigint.and_` over `W_LongObject` operands (collecting digit malloc). See
/// [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_and_raw(a: PyObjectRef, b: PyObjectRef) -> *mut BigInt {
    unsafe {
        alloc_bigint_nursery(BigInt::and_(
            &*w_long_get_raw_value(a),
            &*w_long_get_raw_value(b),
        ))
    }
}

/// `rbigint.or_` over `W_LongObject` operands (collecting digit malloc). See
/// [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_or_raw(a: PyObjectRef, b: PyObjectRef) -> *mut BigInt {
    unsafe {
        alloc_bigint_nursery(BigInt::or_(
            &*w_long_get_raw_value(a),
            &*w_long_get_raw_value(b),
        ))
    }
}

/// `rbigint.xor_` over `W_LongObject` operands (collecting digit malloc). See
/// [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_xor_raw(a: PyObjectRef, b: PyObjectRef) -> *mut BigInt {
    unsafe {
        alloc_bigint_nursery(BigInt::xor(
            &*w_long_get_raw_value(a),
            &*w_long_get_raw_value(b),
        ))
    }
}

/// `rbigint.add`/`sub`/`mul`/`and_`/`or_`/`xor_` (`rpython/rlib/rbigint.py`,
/// each `@jit.elidable`) on bare `*const BigInt` payloads — the elidable
/// arithmetic the walker emits after reading each operand's immutable `value`
/// via an immutable `GetfieldGc`. Taking the payloads (not the `W_LongObject` wrappers)
/// keeps the call's inputs the immutable bigints, so the optimizer forwards the
/// field read and never reorders this elidable call ahead of the boxing
/// `setfield_gc` that initializes the fresh result wrapper. Allocates the result
/// via the COLLECTING nursery. The residual `CallR`'s gcmap roots its operand
/// payloads in the caller, and `rbigint.add` roots and reloads the native
/// callee copies around `_x_add`'s digit-list allocation, matching RPython's
/// `push_roots(livevars)`. Thus dead bigints are reclaimed by minor
/// collections instead of accumulating in old-gen. Returns a freshly
/// heap-allocated `*mut BigInt`. Allocates → `EF_ELIDABLE_OR_MEMORYERROR`.
///
/// The pointers are live GC bigints (the operands' value fields) for the duration of the call.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_add(a: *const BigInt, b: *const BigInt) -> *mut BigInt {
    unsafe {
        // `rbigint.add` (rbigint.py) returns the *other operand itself*
        // when one side is zero; the raw-pointer ABI spells that identity as
        // returning the incoming payload instead of re-wrapping an alias.
        if (&*a).get_sign() == 0 {
            return b as *mut BigInt;
        }
        if (&*b).get_sign() == 0 {
            return a as *mut BigInt;
        }
        alloc_bigint_nursery_collecting(BigInt::add(&*a, &*b))
    }
}

/// `rbigint.sub` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub(a: *const BigInt, b: *const BigInt) -> *mut BigInt {
    unsafe {
        // `rbigint.sub` (rbigint.py) returns `self` itself for a zero
        // subtrahend; see [`jit_bigint_add`] for the raw-pointer spelling.
        if (&*b).get_sign() == 0 {
            return a as *mut BigInt;
        }
        alloc_bigint_nursery_collecting(BigInt::sub(&*a, &*b))
    }
}

/// `rbigint.mul` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul(a: *const BigInt, b: *const BigInt) -> *mut BigInt {
    unsafe { alloc_bigint_nursery_collecting(BigInt::mul(&*a, &*b)) }
}

/// `rbigint.and_` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_and(a: *const BigInt, b: *const BigInt) -> *mut BigInt {
    unsafe { alloc_bigint_nursery_collecting(BigInt::and_(&*a, &*b)) }
}

/// `rbigint.or_` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_or(a: *const BigInt, b: *const BigInt) -> *mut BigInt {
    unsafe { alloc_bigint_nursery_collecting(BigInt::or_(&*a, &*b)) }
}

/// `rbigint.xor_` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_xor(a: *const BigInt, b: *const BigInt) -> *mut BigInt {
    unsafe { alloc_bigint_nursery_collecting(BigInt::xor(&*a, &*b)) }
}

/// `rbigint` comparison — returns the sign of `a <=> b` as `-1` / `0` / `1`.
/// RPython exposes the comparison as six methods (`lt`/`le`/`eq`/`ne`/`gt`/`ge`,
/// the latter built as `other.lt(self)` wrappers, `rbigint.py/664`); Rust's
/// total `Ord::cmp` collapses them into one three-way result, and the caller
/// recovers each relation with a plain `int_<cmp>(sign, 0)` (e.g. `a < b` ⟺
/// `sign < 0`, `a == b` ⟺ `sign == 0`).  A comparison neither allocates nor
/// raises, so this is `EF_ELIDABLE_CANNOT_RAISE` and the fast path records
/// `CallPure*` with NO trailing guard.
///
/// The arguments are the bare payloads, not the `W_LongObject` boxes:
/// `_make_descr_cmp` (longobject.py) compares `self.num` against
/// `w_other.num`, so the two field reads belong in the trace and not inside the
/// callee.  With them spelled out, a `W_LongObject` the same trace has just
/// built is read back through the heap cache and can stay virtual instead of
/// being forced into a real allocation just to be handed to a comparison.
#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_bigint_cmp(a: *const BigInt, b: *const BigInt) -> i64 {
    use core::cmp::Ordering;
    unsafe {
        match (*a).cmp(&*b) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_create_and_read() {
        let obj = w_long_new(BigInt::from(42));
        unsafe {
            assert!(is_long(obj));
            assert!(!is_int(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(42));
        }
    }

    #[test]
    fn test_long_from_i64() {
        let obj = w_long_from_i64(i64::MAX);
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_long_large_value() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big.clone());
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), big);
        }
    }

    #[test]
    fn test_long_field_offset() {
        assert_eq!(LONG_VALUE_OFFSET, 16);
    }

    #[test]
    fn test_jit_w_long_fits_int_in_range() {
        let obj = w_long_from_i64(123);
        assert_eq!(jit_w_long_fits_int(obj), 1);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_fits_int(obj), 1);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_fits_int(obj), 1);
    }

    #[test]
    fn test_jit_w_long_fits_int_out_of_range() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj), 0);
        let big = BigInt::from(i64::MIN) - BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj), 0);
    }

    #[test]
    fn test_jit_w_long_toint_extracts_i64() {
        let obj = w_long_from_i64(42);
        assert_eq!(jit_w_long_toint(obj), 42);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_toint(obj), i64::MAX);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_toint(obj), i64::MIN);
    }

    #[test]
    fn test_jit_w_long_add_raw_payload() {
        // The elidable half returns a bare `*mut BigInt` carrying the sum,
        // with no Python-object wrapper.
        let a = w_long_new(BigInt::from(i64::MAX));
        let b = w_long_new(BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a, b);
        unsafe {
            assert_eq!(*raw, BigInt::from(i64::MAX) * 2);
        }
    }

    #[test]
    fn test_jit_w_long_binop_raw_payloads() {
        // sub/mul/and/or/xor raw helpers mirror jit_w_long_add_raw: bare
        // `*mut BigInt` carrying the arithmetic result, no Python wrapper.
        let x = BigInt::from(i64::MAX) + BigInt::from(7);
        let y = BigInt::from(i64::MAX) - BigInt::from(3);
        let a = w_long_new(x.clone());
        let b = w_long_new(y.clone());
        unsafe {
            let sub = jit_w_long_sub_raw(a, b);
            assert_eq!(*sub, &x - &y);
            let mul = jit_w_long_mul_raw(a, b);
            assert_eq!(*mul, &x * &y);
            let and = jit_w_long_and_raw(a, b);
            assert_eq!(*and, &x & &y);
            let or = jit_w_long_or_raw(a, b);
            assert_eq!(*or, &x | &y);
            let xor = jit_w_long_xor_raw(a, b);
            assert_eq!(*xor, &x ^ &y);
        }
    }

    #[test]
    fn test_jit_w_long_add_raw_keeps_payload_when_fits() {
        // The raw helper never demotes: a sum that fits i64 still yields a
        // `*mut BigInt` payload (the boxing NEW wraps it as a W_LongObject,
        // matching `newlong` which does not demote).
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_long_new(BigInt::from(-1) - BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a, b);
        unsafe {
            assert_eq!(*raw, BigInt::from(0));
        }
    }

    #[test]
    fn test_bigint_add_zero_shortcuts_preserve_payload_identity() {
        let zero_obj = w_long_new(BigInt::from(0));
        let value_obj = w_long_new(BigInt::from(i64::MAX) + BigInt::from(17));
        let zero_payload = unsafe { w_long_get_raw_value(zero_obj) };
        let value_payload = unsafe { w_long_get_raw_value(value_obj) };

        assert_eq!(jit_w_long_add_raw(zero_obj, value_obj), value_payload);
        assert_eq!(jit_w_long_add_raw(value_obj, zero_obj), value_payload);
        assert_eq!(jit_bigint_add(zero_payload, value_payload), value_payload);
        assert_eq!(jit_bigint_add(value_payload, zero_payload), value_payload);
        unsafe {
            assert_eq!(&*zero_payload, &BigInt::from(0));
            assert_eq!(
                &*value_payload,
                &(BigInt::from(i64::MAX) + BigInt::from(17))
            );
        }
    }

    #[test]
    fn test_bare_bigint_constructor_comparison_and_scalar_residuals() {
        let a = jit_bigint_from_i64(-42);
        let b = jit_bigint_from_u64(42);
        let cloned = jit_bigint_clone(a);
        let prebuilt_zero = jit_bigint_from_i64(0);
        let cloned_zero = jit_bigint_clone(prebuilt_zero);
        unsafe {
            assert_eq!(&*a, &BigInt::from(-42));
            assert_eq!(&*b, &BigInt::from(42));
            assert_eq!(&*cloned, &*a);
            assert_ne!(cloned, a);
            assert_eq!(&*cloned_zero, &*prebuilt_zero);
            assert_ne!(cloned_zero, prebuilt_zero);
        }
        assert_eq!(jit_bigint_eq(a, b), 0);
        assert_eq!(jit_bigint_lt(a, b), 1);
        assert_eq!(jit_bigint_ge(b, a), 1);
        assert_eq!(jit_bigint_sign_i64(unsafe { &*a }), -1);
        assert_eq!(jit_bigint_is_zero(a), 0);
        assert_eq!(jit_bigint_tobool(a), 1);
        assert_eq!(jit_bigint_bits(b), 6);
        assert_eq!(jit_bigint_hash(b), 42);
    }
}
