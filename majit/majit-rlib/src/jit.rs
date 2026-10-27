//! `rpython/rlib/jit.py` — the interpreter-facing JIT hint surface.
//!
//! `isconstant` / `isvirtual` return false when not tracing.  The translator
//! rewrites calls marked `oopspec("jit.isvirtual")` / `jit.isconstant` into
//! the `ref_isvirtual` / `*_isconstant` ops; the bodies here are the
//! untranslated residual (`rlib/jit.py isconstant` / `isvirtual`).
//! `record_exact_value` is `rlib/jit.py record_exact_value`; jtransform
//! rewrites that helper to `record_exact_value_{i,r}` (`jtransform.py
//! rewrite_op_jit_record_exact_value`).
//!
//! `we_are_jitted` is `_we_are_jitted = CDefinedIntSymbolic('0 /* we are not
//! jitted here */', default=0)`: the translated interpreter reads a literal
//! `0`, and only the codewriter's view of a graph folds it to true.

/// `rlib/jit.py we_are_jitted`.
///
/// `inline(never)` so Charon keeps the call, and `dont_look_inside` so the
/// translator cannot fold the native body to const false. `jtransform.rs`
/// `fold_we_are_jitted_calls` rewrites the call to `_we_are_jitted`, then
/// jitcode folds that to const true (`jtransform.py`
/// `rewrite_op_int_is_true` of `_we_are_jitted`).  Natively the body is the
/// symbolic's C value, so a `look_inside_iff` dispatch in the running
/// interpreter never pays a thread-local read to choose between two arms
/// that both call the original function.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn we_are_jitted() -> bool {
    false
}

/// `rlib/jit.py isconstant`.
///
/// The literal goes through [`crate::nonconst::non_constant`] so the
/// annotator reads the result as a non-constant `SomeBool`.  A bare `false`
/// would let it fold every `if isconstant(x)` in the interpreter to the
/// not-constant arm and delete the looked-inside arm before `jtransform`
/// ever rewrites the call to `*_isconstant`.
#[majit_macros::specialize_call_location]
#[majit_macros::oopspec("jit.isconstant(value)")]
pub fn isconstant<T: ?Sized>(_value: &T) -> bool {
    crate::nonconst::non_constant(false)
}

/// `rlib/jit.py isvirtual`.
///
/// Non-constant for the same reason as [`isconstant`].
#[majit_macros::specialize_call_location]
#[majit_macros::oopspec("jit.isvirtual(value)")]
pub fn isvirtual<T: ?Sized>(_value: &T) -> bool {
    crate::nonconst::non_constant(false)
}

/// Assure the JIT that value is the same as const_value.
///
/// rlib/jit.py record_exact_value — `assert value == const_value; return const_value`
#[inline(always)]
pub fn record_exact_value<T: Copy + PartialEq + std::fmt::Debug>(value: T, const_value: T) -> T {
    debug_assert_eq!(
        value, const_value,
        "record_exact_value called with two different arguments"
    );
    const_value
}

/// `rlib/jit.py conditional_call` residual body (`if condition: function(*args)`).
///
/// Translated graphs rewrite a `oopspec("jit.conditional_call")` call to
/// `conditional_call_ir_v` (`jtransform.py rewrite_op_jit_conditional_call`).
/// Upstream is one `*args` function with `_always_inline_ = 'try'`; Rust
/// spells one arity per residual word-count (`jtransform.py` rejects more
/// than 4 function arguments).
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call0(condition: bool, function: unsafe fn()) {
    if condition {
        unsafe { function() }
    }
}

/// [`conditional_call0`] with one function argument.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call1<A>(condition: bool, function: unsafe fn(A), a: A) {
    if condition {
        unsafe { function(a) }
    }
}

/// [`conditional_call0`] with two function arguments.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call2<A, B>(condition: bool, function: unsafe fn(A, B), a: A, b: B) {
    if condition {
        unsafe { function(a, b) }
    }
}

/// [`conditional_call0`] with three function arguments.
///
/// `rlist.py _ll_list_resize_ge` uses this shape:
/// `conditional_call(cond, _ll_list_resize_hint_really, l, newsize, True)`.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call3<A, B, C>(condition: bool, function: unsafe fn(A, B, C), a: A, b: B, c: C) {
    if condition {
        unsafe { function(a, b, c) }
    }
}

/// [`conditional_call0`] with four function arguments.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call4<A, B, C, D>(
    condition: bool,
    function: unsafe fn(A, B, C, D),
    a: A,
    b: B,
    c: C,
    d: D,
) {
    if condition {
        unsafe { function(a, b, c, d) }
    }
}

/// The `value` of [`conditional_call_elidable0`]: `rlib/jit.py
/// conditional_call_elidable` recomputes when an integer cache is `0` or a
/// pointer cache is NULL.
pub trait ConditionalCallValue: Copy {
    /// `value == 0` for an integer, `not value` for a pointer.
    fn is_miss(self) -> bool;
}

impl<T> ConditionalCallValue for *mut T {
    fn is_miss(self) -> bool {
        self.is_null()
    }
}

impl<T> ConditionalCallValue for *const T {
    fn is_miss(self) -> bool {
        self.is_null()
    }
}

impl ConditionalCallValue for isize {
    fn is_miss(self) -> bool {
        self == 0
    }
}

impl ConditionalCallValue for usize {
    fn is_miss(self) -> bool {
        self == 0
    }
}

impl ConditionalCallValue for i64 {
    fn is_miss(self) -> bool {
        self == 0
    }
}

impl ConditionalCallValue for u64 {
    fn is_miss(self) -> bool {
        self == 0
    }
}

impl ConditionalCallValue for i32 {
    fn is_miss(self) -> bool {
        self == 0
    }
}

impl ConditionalCallValue for u32 {
    fn is_miss(self) -> bool {
        self == 0
    }
}

/// `rlib/jit.py conditional_call_elidable` residual body
/// (`if value is NULL: value = function(*args); return value`).
///
/// Translated graphs rewrite a `oopspec("jit.conditional_call_value")` call
/// to `conditional_call_value_ir_{i,r}` (`jtransform.py`
/// `Transformer.rewrite_op_jit_conditional_call_value` /
/// `_jit_conditional_call_value`). Argument order is
/// `(value, function, *args)`, the same as the void [`conditional_call0`]
/// family that reaches `Transformer.rewrite_op_jit_conditional_call`.
/// An integer `0` or a NULL pointer is a miss (`rlib/jit.py
/// conditional_call_elidable`). Upstream is one `*args` function with
/// `_always_inline_ = 'try'`; Rust spells one arity per residual
/// word-count (`Transformer._rewrite_op_cond_call` rejects more than 4
/// function arguments).
#[majit_macros::oopspec("jit.conditional_call_value")]
pub fn conditional_call_elidable0<V: ConditionalCallValue>(
    value: V,
    function: unsafe fn() -> V,
) -> V {
    if value.is_miss() {
        unsafe { function() }
    } else {
        value
    }
}

/// [`conditional_call_elidable0`] with one function argument.
#[majit_macros::oopspec("jit.conditional_call_value")]
pub fn conditional_call_elidable1<V: ConditionalCallValue, A>(
    value: V,
    function: unsafe fn(A) -> V,
    a: A,
) -> V {
    if value.is_miss() {
        unsafe { function(a) }
    } else {
        value
    }
}

/// [`conditional_call_elidable0`] with two function arguments.
#[majit_macros::oopspec("jit.conditional_call_value")]
pub fn conditional_call_elidable2<V: ConditionalCallValue, A, B>(
    value: V,
    function: unsafe fn(A, B) -> V,
    a: A,
    b: B,
) -> V {
    if value.is_miss() {
        unsafe { function(a, b) }
    } else {
        value
    }
}

/// [`conditional_call_elidable0`] with three function arguments.
#[majit_macros::oopspec("jit.conditional_call_value")]
pub fn conditional_call_elidable3<V: ConditionalCallValue, A, B, C>(
    value: V,
    function: unsafe fn(A, B, C) -> V,
    a: A,
    b: B,
    c: C,
) -> V {
    if value.is_miss() {
        unsafe { function(a, b, c) }
    } else {
        value
    }
}

/// [`conditional_call_elidable0`] with four function arguments.
#[majit_macros::oopspec("jit.conditional_call_value")]
pub fn conditional_call_elidable4<V: ConditionalCallValue, A, B, C, D>(
    value: V,
    function: unsafe fn(A, B, C, D) -> V,
    a: A,
    b: B,
    c: C,
    d: D,
) -> V {
    if value.is_miss() {
        unsafe { function(a, b, c, d) }
    } else {
        value
    }
}

/// `rlib/jit.py loop_unrolling_heuristic`.
///
/// `isvirtual(lst)` is often lying for a resizable list (it reports the
/// containing struct, not the whole list), so size must also be constant.
/// `@specialize.call_location()` keeps each call site's first argument
/// on its own graph so a tuple's `wrappeditems` (`SomeList`) and a
/// specialised tuple's `self` (`SomeInstance`) never share a phi.
#[majit_macros::specialize_call_location]
pub fn loop_unrolling_heuristic<T: ?Sized>(lst: &T, size: usize, cutoff: usize) -> bool {
    size == 0 || (isconstant(&size) && (isvirtual(lst) || size <= cutoff))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_exact_value_passthrough() {
        assert_eq!(record_exact_value(42, 42), 42);
    }

    #[test]
    fn loop_unrolling_heuristic_empty_unrolls() {
        // `rlib/jit.py loop_unrolling_heuristic`: size == 0 is always true.
        let lst: &[i32] = &[];
        assert!(loop_unrolling_heuristic(lst, 0, 2));
    }

    #[test]
    fn loop_unrolling_heuristic_needs_constant_size() {
        // Residual `isconstant` is false, so a non-empty list does not unroll.
        let lst = &[1, 2];
        assert!(!loop_unrolling_heuristic(lst, 2, 2));
    }

    #[test]
    fn conditional_call_runs_when_true() {
        // Residual `rlib/jit.py conditional_call`: `if condition: function(*args)`.
        fn mark(flag: &mut bool) {
            *flag = true;
        }
        let mut flag = false;
        conditional_call1(true, mark, &mut flag);
        assert!(flag);
        let mut flag = true;
        conditional_call1(false, mark, &mut flag);
        assert!(flag);
    }

    #[test]
    fn conditional_call_elidable1_computes_on_null() {
        // Residual `rlib/jit.py conditional_call_elidable`: null miss calls
        // the function; a later hit returns the cached pointer.
        fn compute(flag: &mut i32) -> *mut i32 {
            *flag = 1;
            flag
        }
        let mut slot = 0i32;
        let hit = conditional_call_elidable1(std::ptr::null_mut(), compute, &mut slot);
        assert!(std::ptr::eq(hit, &mut slot));
        assert_eq!(slot, 1);
        let again = conditional_call_elidable1(hit, compute, &mut slot);
        assert!(std::ptr::eq(again, hit));
        assert_eq!(slot, 1);
    }

    #[test]
    fn conditional_call_elidable0_computes_on_null() {
        fn compute() -> *mut i32 {
            std::ptr::NonNull::<i32>::dangling().as_ptr()
        }
        let hit = conditional_call_elidable0(std::ptr::null_mut(), compute);
        assert!(!hit.is_null());
        let again = conditional_call_elidable0(hit, compute);
        assert!(std::ptr::eq(again, hit));
    }

    #[test]
    fn conditional_call_elidable2_computes_on_null() {
        fn compute(flag: &mut i32, add: i32) -> *mut i32 {
            *flag += add;
            flag
        }
        let mut slot = 0i32;
        let hit = conditional_call_elidable2(std::ptr::null_mut(), compute, &mut slot, 2);
        assert!(std::ptr::eq(hit, &mut slot));
        assert_eq!(slot, 2);
        let again = conditional_call_elidable2(hit, compute, &mut slot, 2);
        assert!(std::ptr::eq(again, hit));
        assert_eq!(slot, 2);
    }

    #[test]
    fn conditional_call_elidable_int_cache_computes_on_zero() {
        // Residual `rlib/jit.py conditional_call_elidable`: integer 0 is a miss.
        unsafe fn compute(x: isize) -> isize {
            x * 2
        }
        assert_eq!(conditional_call_elidable1(0isize, compute, 21), 42);
        assert_eq!(conditional_call_elidable1(7isize, compute, 21), 7);
    }

    #[test]
    fn conditional_call_elidable_ptr_cache_computes_on_null() {
        // Residual `rlib/jit.py conditional_call_elidable`: NULL is a miss.
        unsafe fn compute(target: *mut i32) -> *mut i32 {
            target
        }
        let target = Box::leak(Box::new(1i32)) as *mut i32;
        let hit = conditional_call_elidable1(std::ptr::null_mut(), compute, target);
        assert!(std::ptr::eq(hit, target));
        let cached = Box::leak(Box::new(2i32)) as *mut i32;
        let again = conditional_call_elidable1(cached, compute, target);
        assert!(std::ptr::eq(again, cached));
    }

    #[test]
    fn conditional_call_elidable0_usize_cache() {
        unsafe fn compute() -> usize {
            42
        }
        assert_eq!(conditional_call_elidable0(0usize, compute), 42);
        assert_eq!(conditional_call_elidable0(7usize, compute), 7);
    }

    #[test]
    fn conditional_call_elidable4_usize_cache() {
        unsafe fn compute(a: usize, b: usize, c: usize, d: usize) -> usize {
            a + b + c + d
        }
        assert_eq!(conditional_call_elidable4(0usize, compute, 1, 2, 3, 4), 10);
        assert_eq!(conditional_call_elidable4(7usize, compute, 1, 2, 3, 4), 7);
    }
}
