//! W_IntObject — Python `int` type backed by i64.
//!
//! Uses a fixed i64 representation. BigInt support (like PyPy's
//! `W_LongObject`) will be added later.

use std::sync::LazyLock;

use crate::pyobject::*;

/// Python integer object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | intval: i64]`
/// The JIT reads `intval` via `GetfieldGcI` at `INT_INTVAL_OFFSET`.
#[repr(C)]
pub struct W_IntObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// The translated user-subclass layout selected by `typedef.py _getusercls`.
/// `W_IntObject` remains the base payload; `MapdictStorageMixin` contributes
/// its fields only to the generated user class.
#[repr(C)]
pub struct W_IntObjectUser {
    pub base: W_IntObject,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_IntObjectUser, storage)
            == std::mem::offset_of!(W_IntObjectUser, map) + std::mem::size_of::<usize>()
    );
};

/// Field offset of `intval` within `W_IntObject`, for JIT field access.
pub const INT_INTVAL_OFFSET: usize = std::mem::offset_of!(W_IntObject, intval);

/// GC type id assigned to `W_IntObject` at JitDriver init time. Held
/// as a constant here (rather than runtime-queried) so the allocation
/// hook can reach it without a back-channel. `pyre-jit/src/eval.rs`
/// asserts the same id is returned by `gc.register_type(...)` so any
/// mismatch panics on startup instead of silently misclassifying the
/// type at collection time.
pub const W_INT_GC_TYPE_ID: u32 = 1;
/// User-subclass int layout (`typedef.py` `_getusercls`). Unconditional,
/// so its tid sits with the other closed ids (158) ahead of the
/// target-gated tail.
pub const W_INT_USER_GC_TYPE_ID: u32 = 158;
pub const W_INT_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_IntObjectUser>();

// ── Prebuilt-int cache ───────────────────────────────────────────────
//
// `pypy/config/pypyoption.py:206-213` parity:
//   withprebuiltint  default False — prebuild commonly used int objects
//   prebuiltintfrom  default -5    — lowest integer which is prebuilt
//   prebuiltintto    default 100   — highest integer (stop-exclusive)
//
// `pypy/objspace/std/intobject.py` `setup_prebuilt` / `wrapint`
// consult these three constants. Pyre keeps them as translation-time
// constants since pyre has no live `space.config` surface — the exact
// PyPy default (`WITHPREBUILTINT=false`) is used to keep `wrapint`'s
// allocation behaviour line-by-line with PyPy default.

pub const WITHPREBUILTINT: bool = false;
pub const PREBUILTINTFROM: i64 = -5;
pub const PREBUILTINTTO: i64 = 100;
pub const W_INT_OBJECT_SIZE: usize = std::mem::size_of::<W_IntObject>();

impl crate::lltype::GcType for W_IntObject {
    fn type_id() -> u32 {
        W_INT_GC_TYPE_ID
    }
    const SIZE: usize = W_INT_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_IntObjectUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_INT_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_INT_USER_OBJECT_SIZE;
}

/// `intobject.py setup_prebuilt`. Empty when
/// `WITHPREBUILTINT=false`; populated `[PREBUILTINTFROM, PREBUILTINTTO)`
/// (stop-exclusive) when `true`.
struct PrebuiltInts(Vec<W_IntObject>);

// `intobject.py setup_prebuilt` publishes exact, immutable ints for
// process lifetime. Wrapping the contiguous allocation as Sync preserves that
// shared upstream owner without making mutable subclass instances globally
// shareable.
unsafe impl Send for PrebuiltInts {}
unsafe impl Sync for PrebuiltInts {}

static SMALL_INTS: LazyLock<PrebuiltInts> = LazyLock::new(|| {
    if !WITHPREBUILTINT {
        return PrebuiltInts(Vec::new());
    }
    PrebuiltInts(
        (PREBUILTINTFROM..PREBUILTINTTO)
            .map(|v| W_IntObject {
                ob_header: PyObject {
                    ob_type: &INT_TYPE as *const PyType,
                    w_class: std::ptr::null_mut(),
                },
                intval: v,
            })
            .collect(),
    )
});

/// `pypy/objspace/std/intobject.py wrapint` parity.
///
/// `withprebuiltint=False` (PyPy default) → always allocate fresh,
/// matching upstream `return W_IntObject(x)`. With the flag enabled
/// and `value` inside `[PREBUILTINTFROM, PREBUILTINTTO)` the cache
/// returns the pre-allocated entry; outside the range we allocate
/// (`instantiate(W_IntObject)` upstream).
///
/// Upstream's two arms then join on a shared `w_res.intval = x`
/// (`intobject.py`) which rewrites the payload of a *prebuilt* entry
/// too. `PREBUILT[x - lower]` already holds `x`, so that store changes no
/// value; its own comment calls it "an obscure hack to help the CPU cache",
/// there to pull the field into cache before the caller reads it. The
/// prebuilt arm here returns without it, and cannot currently spell it: the
/// pointer is derived from `&SMALL_INTS.0[idx]`, so writing through it is
/// undefined rather than merely racy, and the `unsafe impl Sync for
/// PrebuiltInts` above rests on those entries staying immutable for process
/// lifetime, which a store would break for concurrent callers. Spelling it
/// faithfully needs the payload behind an `UnsafeCell` — until then the
/// omission costs a cache hint and no semantics, and the arm is unreachable
/// while `WITHPREBUILTINT` is false.
///
/// Traced, not residualised, and `#[inline]` — `wrapint` carries no
/// `@dont_look_inside` and its own comment reads "this whole function is
/// getting inlined into every caller so keeping the branching to a minimum
/// is a good idea" (intobject.py). The allocation upstream is
/// `instantiate(W_IntObject)` followed by `w_res.intval = x`
/// (intobject.py): alloc-then-init, which the rtyper lowers to the
/// `new_with_vtable` + payload `setfield_gc` pair the optimizer can
/// virtualize where the box does not escape. The stack-built
/// `malloc_typed(W_IntObject { .. })` spelling below is the Rust form of the
/// same thing: `fuse_boxing_alloc` (`majit-translate` `model.rs`) rewrites
/// that cluster into exactly that pair, and it does fire here — its walk
/// resolves the header pointers across the block boundary each call ends.
///
/// `bench/synth/list_pop_append` reads the same either way, and its negative
/// control (boundary removed with the fusion reverted) failed to reproduce
/// the regression that motivated the boundary, so that bench does not
/// discriminate this mechanism in either direction. What decides it is the
/// shape: a residual call can never be virtualized and `new_with_vtable`
/// can, and the boundary's own stated blocker — the fusion not resolving a
/// vtable here — is gone.
///
/// The allocation path goes through [`crate::lltype::malloc_typed`]
/// which carries `W_INT_GC_TYPE_ID` +
/// `W_INT_OBJECT_SIZE` via the [`crate::lltype::GcType`] impl above —
/// the Rust analog of `gct_fv_gc_malloc`'s compile-time `c_type_id`
/// / `c_size` (`rpython/memory/gctransform/framework.py`).
/// `malloc_typed` prepends a `GcHeader` (`alloc_with_gc_header`) but
/// allocates outside the collector's heap, so the box carries a readable
/// type id while staying off the sweep set; future GC integration
/// replaces only that body, this constructor stays unchanged.
#[inline]
pub fn w_int_new(value: i64) -> PyObjectRef {
    // `ll_int_box`: a value that fits the immediate range is returned as a
    // tagged pointer `(value << 1) | 1` with no allocation — the whole point
    // of the representation. Gated on `CAN_BE_TAGGED` (default false), so the
    // heap-box path below is the only live one until enablement. Subclass
    // instances never reach here (they use `w_int_new_unique`), so a tagged
    // result is always an exact builtin `int`.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::fits_tagged(value) {
        return crate::tagged_int::tag_int(value);
    }
    if WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&value) {
        let idx = (value - PREBUILTINTFROM) as usize;
        return (&SMALL_INTS.0[idx] as *const W_IntObject).cast_mut() as PyObjectRef;
    }
    if crate::gc_interp::enabled() {
        let boxed = w_int_gc_alloc(value);
        if !boxed.is_null() {
            return boxed;
        }
    }
    crate::lltype::malloc_typed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: value,
    }) as PyObjectRef
}

/// The collector-heap arm of [`w_int_new`]: the `gct_fv_gc_malloc` bracket
/// (`rpython/memory/gctransform/framework.py`) in its alloc-then-init form —
/// take the block, then write the header and payload into it, rather than
/// copying a stack-built struct over it. Returns null when no collector owns
/// the heap, which is the caller's signal to take the `malloc_typed` arm.
///
/// Residualised (`rlib/jit.py @dont_look_inside`) rather than traced,
/// which [`w_int_new`] is not — a deviation this arm alone carries, because
/// neither spelling of it reaches the trace intact. Writing the fields into
/// the collector's block, as below, lands the header's own `ob_type` slot at
/// offset 0, which the wasm backend does not lower faithfully. Building the
/// struct on the stack and copying it in instead is not the
/// `malloc_typed(%agg)` route `fuse_boxing_alloc` rewrites, so its
/// `SyntheticTransparentCtor` for the `PyObject` header survives lowering as
/// a call to a `symbolic_fnaddr` hash, which a descending sub-jitcode walk
/// cannot record and declines on. The boundary keeps both shapes out of the
/// trace. It is not free anywhere: `gc_interp::enabled()` reads
/// `PYRE_GC_INTERP` and answers true for every value except exactly `0`
/// (`gc_interp.rs` `enabled_from_env`), so it is on by default on every
/// target and this arm — not the `malloc_typed` one `fuse_boxing_alloc`
/// rewrites into a `NewWithVtable` — is the arm [`w_int_new`] takes. Drop the
/// boundary once the wasm backend lowers an offset-0 field store faithfully.
///
/// Spelled `*mut PyObject` rather than `PyObjectRef` so the attribute emits
/// its `extern "C"` call trampoline: the macro recognises raw pointers
/// syntactically and declines to emit one for an aliased return type.
#[majit_macros::dont_look_inside]
pub fn w_int_gc_alloc(value: i64) -> *mut PyObject {
    w_int_gc_alloc_in(value)
}

fn w_int_gc_alloc_in(value: i64) -> *mut PyObject {
    // `get_instantiate` is a plain load of the class word. The residual
    // return is a GCREF (`*mut PyObject`), so the bump is
    // `malloc_fixedsize` — same as `w_tuple_new`.
    let w_class = get_instantiate(&INT_TYPE);
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_INT_GC_TYPE_ID, W_INT_OBJECT_SIZE);
    if raw.is_null() {
        return crate::PY_NULL;
    }
    unsafe {
        let p = raw as *mut W_IntObject;
        (*p).ob_header.ob_type = &INT_TYPE as *const PyType;
        (*p).ob_header.w_class = w_class;
        (*p).intval = value;
    }
    crate::gc_hook::try_gc_write_barrier_managed(raw);
    raw as PyObjectRef
}

/// Create a W_IntObject bypassing the small-int cache.
///
/// Used for int subclass instances that need unique object identity
/// (so per-object attributes don't collide). This must NEVER return a
/// tagged immediate: tagged ints are value-identical (`1 is 1`), which is
/// exactly the identity a subclass instance must not have. It always
/// allocates a distinct heap box, independent of `CAN_BE_TAGGED`.
pub fn w_int_new_unique(value: i64) -> PyObjectRef {
    let obj = W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: value,
    };
    // `intobject.py:_as_subint` creates a subtype through
    // `space.allocate_instance(W_IntObject, w_inttype)`, so the fresh box is
    // an ordinary collector-owned instance (and can enter the finalizer
    // queue when the subtype defines `__del__`).  Unlike exact ephemeral int
    // boxes, a user instance must not use the non-moving old-gen fast path:
    // that path is intentionally long-lived and would prevent the subtype's
    // registered finalizer from observing that the instance became dead.
    crate::lltype::malloc_typed_managed(obj) as PyObjectRef
}

/// Allocate a `W_IntObject` for an `int` subclass instance, on the managed
/// heap so it can be reclaimed.
///
/// [`w_int_new_unique`] is shared with the JIT's tagged-int unboxing paths
/// and keeps their `malloc_typed` allocation; a subclass instance cannot
/// use it, because `register_finalizer` drops anything outside the managed
/// heap, so a boxed instance would never die and its `__del__` would never
/// run. Allocation is unconditional rather than gated on
/// [`crate::gc_interp::enabled`], matching the other subclass allocators
/// ([`crate::bytesobject::w_bytes_subclass_from_bytes`],
/// `w_str_subclass_from_wtf8`).
pub fn w_int_subclass_new(value: i64) -> PyObjectRef {
    let obj = W_IntObjectUser {
        base: W_IntObject {
            ob_header: PyObject {
                ob_type: &crate::pyobject::INT_USER_TYPE as *const PyType,
                w_class: get_instantiate(&INT_TYPE),
            },
            intval: value,
        },
        map: 0,
        storage: std::ptr::null_mut(),
    };
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(
        W_INT_USER_GC_TYPE_ID,
        std::mem::size_of::<W_IntObjectUser>(),
    );
    if raw.is_null() {
        crate::lltype::malloc_typed(obj) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_IntObjectUser, obj);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    }
}

/// Return the address of INT_TYPE for JIT type-id validation.
#[inline]
pub fn w_int_type_id() -> usize {
    &INT_TYPE as *const PyType as usize
}

/// Extract the i64 value from a known W_IntObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_IntObject`.
#[inline]
pub unsafe fn w_int_get_value(obj: PyObjectRef) -> i64 {
    // `ll_unboxed_to_int`: a tagged immediate carries its value in the
    // pointer bits (`>> 1`). The `crate::tagged_int::CAN_BE_TAGGED`
    // static (default `false`) short-circuits the check away, so the
    // heap-box read below is the only live path until enablement.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return crate::tagged_int::untag_int(obj);
    }
    unsafe { (*(obj as *const W_IntObject)).intval }
}

/// `intobject.py descr_repr` / `descr_str` (they share a body).
/// `res = str(self.intval)` is `ll_int2dec`; the wrap is `space.newutf8`.
///
/// Look-inside: the generated JIT records that split, so a walker
/// descent of this function is the same graph the rtyper would emit
/// from the interpreter source.  Graph-level `UnaryOp { op: "str" }`
/// over an Int is `ll_int2dec` (`jtransform` `int_str_helper`); the
/// wrap lives only here.
///
/// # Safety
/// `obj` must be a machine `int` (not bool, not long).
#[inline(never)]
pub unsafe fn descr_str(obj: PyObjectRef) -> PyObjectRef {
    let v = unsafe { w_int_get_value(obj) };
    let payload = crate::lowlevel_string::jit_ll_int2dec(v);
    if payload.is_null() {
        return crate::PY_NULL;
    }
    // `len(res)` (`descr_repr`): `LLHelpers.ll_strlen` on the `ll_int2dec`
    // STR.  Keep the payload a `Ptr(STR)` so the rtyper emits STRLEN
    // rather than a Signed residual of `bh_lowlevel_string_len`.
    let length = unsafe { (*payload).length };
    crate::unicodeobject::w_str_from_storage_and_length(payload, length)
}

/// `>` in `newformat.py` `_parse_spec` (default align). Padding is
/// `_fill_number`'s `n_lpadding`.
pub const FORMAT_INT_ALIGN_RIGHT: i64 = 0;
/// `<`. Padding is `n_rpadding`.
pub const FORMAT_INT_ALIGN_LEFT: i64 = 1;
/// `=`. Padding is `n_spadding`, between the sign and the digits.
pub const FORMAT_INT_ALIGN_SIGN: i64 = 2;

/// Decimal machine-int arm of `newformat.py` `format_int_or_long`.
///
/// `ll_str.py` `ll_int2dec` is `_int_to_base` for base 10. A negative
/// value splits that sign off (`result[0] == "-"`) and `_fill_number`
/// places `n_spadding` between the sign and the digits, so
/// `format(-42, "05d")` is `"-0042"`. `align` is
/// [`FORMAT_INT_ALIGN_RIGHT`], [`FORMAT_INT_ALIGN_LEFT`], or
/// [`FORMAT_INT_ALIGN_SIGN`]. `forced_sign` is the `"+"` / `" "` STR,
/// or null when `_calc_num_width` keeps minus-only. `'^'` and a
/// non-ASCII fill stay on the residual formatter.
///
/// # Safety
/// `fill` is a one-character STR when `width` exceeds the rendered
/// length. `forced_sign` is null or a one-character STR.
#[inline(never)]
pub unsafe fn format_int_decimal(
    value: i64,
    fill: *mut crate::unicodeobject::Utf8Str,
    width: i64,
    align: i64,
    forced_sign: *mut crate::unicodeobject::Utf8Str,
) -> PyObjectRef {
    let payload = crate::lowlevel_string::jit_ll_int2dec(value);
    if payload.is_null() {
        return crate::PY_NULL;
    }
    let length = unsafe { (*payload).length } as i64;
    let mut digits = payload;
    let mut n_digits = length;
    let mut sign: *mut crate::unicodeobject::Utf8Str = std::ptr::null_mut();
    let mut sign_len = 0i64;
    if value < 0 {
        sign = crate::lowlevel_string::ll_stringslice_startstop(payload, 0, 1);
        digits = crate::lowlevel_string::ll_stringslice_startstop(payload, 1, length);
        if sign.is_null() || digits.is_null() {
            return crate::PY_NULL;
        }
        n_digits = length - 1;
        sign_len = 1;
    } else if !forced_sign.is_null() {
        sign = forced_sign;
        sign_len = 1;
    }
    let mut storage = digits;
    let mut out_len = n_digits;
    let n_padding = width - (sign_len + n_digits);
    if n_padding > 0 {
        let pad = crate::lowlevel_string::jit_ll_str_mul(fill, n_padding);
        if pad.is_null() {
            return crate::PY_NULL;
        }
        if align == FORMAT_INT_ALIGN_LEFT {
            if sign_len > 0 {
                storage = crate::lowlevel_string::jit_ll_strconcat(sign, storage);
                if storage.is_null() {
                    return crate::PY_NULL;
                }
                out_len += sign_len;
            }
            storage = crate::lowlevel_string::jit_ll_strconcat(storage, pad);
            if storage.is_null() {
                return crate::PY_NULL;
            }
            out_len += n_padding;
        } else if align == FORMAT_INT_ALIGN_SIGN {
            storage = crate::lowlevel_string::jit_ll_strconcat(pad, storage);
            if storage.is_null() {
                return crate::PY_NULL;
            }
            out_len += n_padding;
            if sign_len > 0 {
                storage = crate::lowlevel_string::jit_ll_strconcat(sign, storage);
                if storage.is_null() {
                    return crate::PY_NULL;
                }
                out_len += sign_len;
            }
        } else if sign_len > 0 {
            storage = crate::lowlevel_string::jit_ll_strconcat(sign, storage);
            if storage.is_null() {
                return crate::PY_NULL;
            }
            out_len += sign_len;
            storage = crate::lowlevel_string::jit_ll_strconcat(pad, storage);
            if storage.is_null() {
                return crate::PY_NULL;
            }
            out_len += n_padding;
        } else {
            storage = crate::lowlevel_string::jit_ll_strconcat(pad, storage);
            if storage.is_null() {
                return crate::PY_NULL;
            }
            out_len += n_padding;
        }
    } else if sign_len > 0 {
        storage = crate::lowlevel_string::jit_ll_strconcat(sign, storage);
        if storage.is_null() {
            return crate::PY_NULL;
        }
        out_len += sign_len;
    }
    crate::unicodeobject::w_str_from_storage_and_length(storage, out_len as usize)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_int_new(value: i64) -> PyObjectRef {
    w_int_new(value)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_small_int_const(value: i64) -> PyObjectRef {
    w_small_int_const(value)
}

/// True iff `value` falls inside the prebuilt-int cache range AND
/// the cache is enabled. Mirrors `wrapint` in-range branch
/// (`intobject.py`).
#[inline]
pub fn w_int_small_cached(value: i64) -> bool {
    WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&value)
}

#[inline]
pub fn w_int_small_cache_base_ptr() -> PyObjectRef {
    SMALL_INTS.0.as_ptr().cast_mut() as PyObjectRef
}

/// Compact integer-literal range for `LOAD_SMALL_INT` (oparg 0..=255).
/// Values outside this range fall back to [`w_int_new`].
pub const SMALL_INT_CONST_FROM: i64 = 0;
pub const SMALL_INT_CONST_TO: i64 = 256;

static SMALL_INT_CONSTS: [std::sync::OnceLock<usize>; 256] =
    [const { std::sync::OnceLock::new() }; 256];

/// Interned box for a `LOAD_SMALL_INT` literal.
///
/// `getconstant_w` / `co_consts_w` keep one identity per `LOAD_CONST` slot
/// so a JUMP can carry `ConstPtr`. `LOAD_SMALL_INT` has no const index —
/// the oparg *is* the value — so this table is that slot. `wrapint` /
/// [`w_int_new`] still allocate a fresh box; this is not
/// [`WITHPREBUILTINT`].
///
/// `dont_look_inside`: looking inside the `OnceLock` init writes a
/// translator-local `&INT_TYPE` into a fresh box. The residual hits
/// the process table so `type(w1) is type(w2)` sees the same typeptr
/// as interpreter-allocated ints.
#[inline(never)]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn w_small_int_const(value: i64) -> PyObjectRef {
    if (SMALL_INT_CONST_FROM..SMALL_INT_CONST_TO).contains(&value) {
        let idx = (value - SMALL_INT_CONST_FROM) as usize;
        return *SMALL_INT_CONSTS[idx].get_or_init(|| {
            crate::lltype::malloc_typed_immortal(W_IntObject {
                ob_header: PyObject {
                    ob_type: &INT_TYPE as *const PyType,
                    w_class: get_instantiate(&INT_TYPE),
                },
                intval: value,
            }) as usize
        }) as PyObjectRef;
    }
    w_int_new(value)
}

/// CALL_PURE fold: the intern table is deterministic after startup.
#[doc(hidden)]
#[allow(non_upper_case_globals)]
pub const _elidable_function_w_small_int_const: bool = true;

/// `intobject.py _bit_count` parity — population count of an i64.
///
/// `@jit.elidable` (`rlib/jit.py`): deterministic, no allocation,
/// no raise → `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`).
///
/// Line-by-line port of the upstream RPython loop; `i64::MIN` is
/// special-cased because `-i64::MIN` overflows in two's complement,
/// matching `val == -sys.maxint - 1` in the upstream guard.
#[majit_macros::elidable_cannot_raise]
pub fn int_bit_count(val: i64) -> i64 {
    if val == i64::MIN {
        return 1;
    }
    let mut val = val;
    if val < 0 {
        val = -val;
    }
    let mut count: i64 = 0;
    while val != 0 {
        count += val & 1;
        val >>= 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn exact_builtin_layouts_do_not_include_mapdict_storage() {
        use crate::boolobject::W_BoolObject;
        use crate::tupleobject::W_TupleObject;
        use crate::unicodeobject::{UNICODE_VALUE_OFFSET, W_UnicodeObject};

        assert_eq!(std::mem::size_of::<W_IntObject>(), 24);
        assert_eq!(std::mem::size_of::<W_BoolObject>(), 24);
        assert_eq!(std::mem::size_of::<W_UnicodeObject>(), 56);
        assert_eq!(std::mem::size_of::<W_TupleObject>(), 32);
        assert_eq!(INT_INTVAL_OFFSET, 16);
        assert_eq!(UNICODE_VALUE_OFFSET, 16);
        assert_eq!(std::mem::offset_of!(W_TupleObject, hash), 16);
        assert_eq!(std::mem::offset_of!(W_TupleObject, wrappeditems), 24);
    }

    #[test]
    fn test_int_create_and_read() {
        let cases = [("positive", 42_i64), ("negative", -7)];
        for (name, value) in cases {
            let obj = w_int_new(value);
            unsafe {
                assert!(is_int(obj), "case {name}");
                assert!(!is_bool(obj), "case {name}");
                assert_eq!(w_int_get_value(obj), value, "case {name}");
            }
        }
    }

    #[test]
    fn test_int_field_offset() {
        assert_eq!(INT_INTVAL_OFFSET, 16);
    }

    #[test]
    fn small_int_const_reuses_identity_and_does_not_alias_wrapint() {
        let a = w_small_int_const(1);
        let b = w_small_int_const(1);
        assert_eq!(a, b, "LOAD_SMALL_INT 1 is one interned box");
        assert_ne!(
            a,
            w_int_new(1),
            "wrapint still allocates; intern is LOAD_SMALL_INT only"
        );
        unsafe {
            assert!(is_int(a));
            assert_eq!(w_int_get_value(a), 1);
        }
        let outside = w_small_int_const(SMALL_INT_CONST_TO);
        assert_ne!(outside, w_small_int_const(SMALL_INT_CONST_TO));
    }

    /// `intobject.py wrapint` parity: with `withprebuiltint=False`
    /// (PyPy default, mirrored by `WITHPREBUILTINT=false`) every call
    /// allocates a fresh `W_IntObject`, so identity is per-call.
    #[test]
    fn test_w_int_new_identity_matches_config() {
        let a = w_int_new(42);
        let b = w_int_new(42);
        if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::fits_tagged(42) {
            assert_eq!(a, b, "tagged immediates are value-identical (1 is 1)");
        } else if WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&42) {
            assert_eq!(a, b, "in-cache values share the prebuilt pointer");
        } else {
            assert_ne!(a, b, "no cache: each wrapint allocates fresh");
        }
        unsafe {
            assert_eq!(w_int_get_value(a), 42);
            assert_eq!(w_int_get_value(b), 42);
        }
    }

    /// `intobject.py wrapint` in-cache branch — only meaningful
    /// when `WITHPREBUILTINT=true`. Skipped otherwise (no prebuilt
    /// pool means there is no boundary to test).
    #[test]
    fn test_prebuilt_cache_boundary() {
        if !WITHPREBUILTINT {
            return;
        }
        let low = w_int_new(PREBUILTINTFROM);
        let high = w_int_new(PREBUILTINTTO - 1);
        unsafe {
            assert_eq!(w_int_get_value(low), PREBUILTINTFROM);
            assert_eq!(w_int_get_value(high), PREBUILTINTTO - 1);
        }
        assert_eq!(low, w_int_new(PREBUILTINTFROM));
        assert_eq!(high, w_int_new(PREBUILTINTTO - 1));
    }

    #[test]
    fn test_outside_cache_allocates_fresh() {
        let a = w_int_new(1000);
        let b = w_int_new(1000);
        if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::fits_tagged(1000) {
            assert_eq!(a, b, "tagged immediates are value-identical");
        } else {
            assert_ne!(a, b, "out-of-cache ints allocate fresh per wrapint");
        }
        unsafe {
            assert_eq!(w_int_get_value(a), 1000);
            assert_eq!(w_int_get_value(b), 1000);
        }
    }

    #[test]
    fn test_w_int_gc_type_id_matches_descr() {
        // Guard against drift between the constant colocated with
        // `W_IntObject` and the id that `pyre-jit/src/eval.rs`
        // asserts at JitDriver init. See `descr.rs` re-export.
        assert_eq!(W_INT_GC_TYPE_ID, 1);
        assert_eq!(W_INT_USER_GC_TYPE_ID, 158);
    }

    /// `intobject.py _bit_count` parity — verifies the popcount
    /// against `i64::count_ones`, including the `i64::MIN` sentinel
    /// which the upstream short-circuits to `1`.
    #[test]
    fn test_int_bit_count_matches_intrinsic_popcount() {
        for &v in &[
            0i64,
            1,
            -1,
            42,
            -42,
            i64::MAX,
            i64::MAX - 1,
            // i64::MIN is the upstream `-sys.maxint - 1` guard: the
            // bit_count of i64::MIN in two's complement is 1.
            i64::MIN,
        ] {
            let abs = if v == i64::MIN { i64::MAX } else { v.abs() };
            let expected = if v == i64::MIN {
                1
            } else {
                abs.count_ones() as i64
            };
            assert_eq!(
                int_bit_count(v),
                expected,
                "int_bit_count({v}) must match the upstream popcount",
            );
        }
    }

    /// `#[elidable_cannot_raise]` macro on a production helper must
    /// emit `INT_ELIDABLE_CANNOT_RAISE = 19` (`call_policy_byte.rs`)
    /// and a non-null trace_target / concrete_target trampoline pair.
    #[test]
    fn test_int_bit_count_advertises_int_elidable_cannot_raise_byte() {
        let (policy, _, trace_target, concrete_target, _, _) = __majit_call_policy_int_bit_count();
        assert_eq!(policy, 19u8);
        assert!(!trace_target.is_null());
        assert!(!concrete_target.is_null());
    }

    fn storage_of(text: &str) -> *mut crate::unicodeobject::Utf8Str {
        unsafe { crate::unicodeobject::w_str_storage(crate::unicodeobject::w_str_new(text)) }
    }

    /// `newformat.py` `_fill_number` for the decimal machine-int subset:
    /// sign split off, then `n_lpadding` / `n_spadding` / `n_rpadding`.
    #[test]
    fn format_int_decimal_matches_fill_number() {
        let zero = storage_of("0");
        let space = storage_of(" ");
        let plus = storage_of("+");
        let star = storage_of("*");
        let none = std::ptr::null_mut();
        let cases = [
            (42, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "00042"),
            (-42, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "-0042"),
            (1, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "00001"),
            (-1, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "-0001"),
            (0, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "00000"),
            (12345, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "12345"),
            (10000, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "10000"),
            (-123456, zero, 5, FORMAT_INT_ALIGN_SIGN, none, "-123456"),
            (42, space, 5, FORMAT_INT_ALIGN_RIGHT, none, "   42"),
            (-42, space, 5, FORMAT_INT_ALIGN_RIGHT, none, "  -42"),
            (1, space, 5, FORMAT_INT_ALIGN_RIGHT, none, "    1"),
            (-1, space, 5, FORMAT_INT_ALIGN_RIGHT, none, "   -1"),
            (12345, space, 3, FORMAT_INT_ALIGN_RIGHT, none, "12345"),
            (1, space, 3, FORMAT_INT_ALIGN_RIGHT, none, "  1"),
            (42, space, 0, FORMAT_INT_ALIGN_RIGHT, none, "42"),
            (1, space, 0, FORMAT_INT_ALIGN_RIGHT, plus, "+1"),
            (-1, space, 0, FORMAT_INT_ALIGN_RIGHT, plus, "-1"),
            (0, space, 0, FORMAT_INT_ALIGN_RIGHT, plus, "+0"),
            (1, space, 0, FORMAT_INT_ALIGN_RIGHT, space, " 1"),
            (-1, space, 0, FORMAT_INT_ALIGN_RIGHT, space, "-1"),
            (1, space, 5, FORMAT_INT_ALIGN_RIGHT, plus, "   +1"),
            (-1, space, 5, FORMAT_INT_ALIGN_RIGHT, plus, "   -1"),
            (12, space, 5, FORMAT_INT_ALIGN_RIGHT, plus, "  +12"),
            (0, space, 5, FORMAT_INT_ALIGN_RIGHT, plus, "   +0"),
            (1, zero, 5, FORMAT_INT_ALIGN_SIGN, plus, "+0001"),
            (-1, zero, 5, FORMAT_INT_ALIGN_SIGN, plus, "-0001"),
            (42, space, 5, FORMAT_INT_ALIGN_LEFT, none, "42   "),
            (-42, space, 5, FORMAT_INT_ALIGN_LEFT, none, "-42  "),
            (7, space, 5, FORMAT_INT_ALIGN_LEFT, none, "7    "),
            (0, space, 5, FORMAT_INT_ALIGN_LEFT, none, "0    "),
            (42, zero, 5, FORMAT_INT_ALIGN_LEFT, none, "42000"),
            (-42, zero, 5, FORMAT_INT_ALIGN_LEFT, none, "-4200"),
            (-1, zero, 5, FORMAT_INT_ALIGN_LEFT, none, "-1000"),
            (42, zero, 5, FORMAT_INT_ALIGN_RIGHT, none, "00042"),
            (-42, zero, 5, FORMAT_INT_ALIGN_RIGHT, none, "00-42"),
            (-1, zero, 5, FORMAT_INT_ALIGN_RIGHT, none, "000-1"),
            (-42, space, 5, FORMAT_INT_ALIGN_SIGN, none, "-  42"),
            (1, space, 5, FORMAT_INT_ALIGN_SIGN, none, "    1"),
            (42, star, 5, FORMAT_INT_ALIGN_RIGHT, none, "***42"),
            (-42, star, 5, FORMAT_INT_ALIGN_RIGHT, none, "**-42"),
            (1, plus, 5, FORMAT_INT_ALIGN_RIGHT, none, "++++1"),
            (-1, plus, 5, FORMAT_INT_ALIGN_RIGHT, none, "+++-1"),
            (
                i64::MIN,
                zero,
                5,
                FORMAT_INT_ALIGN_SIGN,
                none,
                "-9223372036854775808",
            ),
        ];
        for (value, fill, width, align, sign, expect) in cases {
            let rendered = unsafe { format_int_decimal(value, fill, width, align, sign) };
            let text = unsafe { crate::unicodeobject::w_str_get_value_opt(rendered) }
                .unwrap_or_else(|| panic!("format({value}, width={width}, align={align})"));
            assert_eq!(
                text,
                expect,
                "value={value} width={width} align={align} sign_null={}",
                sign.is_null()
            );
        }
    }
}
