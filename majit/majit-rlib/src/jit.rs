//! `rpython/rlib/jit.py` — the interpreter-facing JIT hint surface.
//!
//! `isconstant` / `isvirtual` return false when not tracing.  The translator
//! rewrites calls marked `oopspec("jit.isvirtual")` / `jit.isconstant` into
//! the `ref_isvirtual` / `*_isconstant` ops; the bodies here are the
//! untranslated residual (`rlib/jit.py isconstant` / `isvirtual`).
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

/// `rlib/jit.py conditional_call_elidable` residual body
/// (`if value is NULL: value = function(*args); return value`).
///
/// Translated graphs rewrite a `oopspec("jit.conditional_call_value")` call
/// to `conditional_call_value_ir_{i,r}` (`jtransform.py`
/// `rewrite_op_jit_conditional_call_value` / `_jit_conditional_call_value`).
/// Argument order is `(value, function, *args)`, the same as the void
/// [`conditional_call1`] helper that reaches `rewrite_op_jit_conditional_call`.
/// Null is a miss. One function argument is the arity unicode
/// `_get_index_storage` needs.
#[majit_macros::oopspec("jit.conditional_call_value")]
pub fn conditional_call_elidable1<T, A>(
    value: *mut T,
    function: unsafe fn(A) -> *mut T,
    a: A,
) -> *mut T {
    if value.is_null() {
        unsafe { function(a) }
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
}
