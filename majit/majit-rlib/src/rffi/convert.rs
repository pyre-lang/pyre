//! `rffi.cast` (`ll2ctypes.force_cast`), `ptradd`, `sizeof`, `size_and_sign`,
//! `offsetof`, `setintfield`, `getintfield`, `structcopy`.

use super::SIGNED;

mod seal {
    pub trait Seal {}
}

/// Implemented for the integer, float, and pointer types `cast` accepts.
/// Outside this crate the trait cannot be implemented (`seal::Seal` is private).
pub trait CastFrom<Src>: seal::Seal {
    fn cast_from(value: Src) -> Self;
}

/// Source of a [`cast`]. The blanket impl is the only one; `CastFrom` is sealed.
pub trait CastInto<Dst> {
    fn cast_into(self) -> Dst;
}

impl<Src, Dst> CastInto<Dst> for Src
where
    Dst: CastFrom<Src>,
{
    #[inline(always)]
    fn cast_into(self) -> Dst {
        Dst::cast_from(self)
    }
}

/// `rffi.cast` / `ll2ctypes.force_cast`.
///
/// `cast::<T>(value)` names the destination. Integer and float conversions are
/// Rust `as` casts: integers truncate or sign-extend, and a float-to-integer
/// cast truncates toward zero.
#[inline(always)]
pub fn cast<T>(value: impl CastInto<T>) -> T {
    value.cast_into()
}

/// `(size, unsigned)` from `rffi.size_and_sign`.
///
/// The `unsigned` flag is the signedness of the Rust primitive an alias names
/// (`CHAR` is `c_char`). Pointers, `FLOAT`, and `DOUBLE` are signed, matching
/// the non-number branch of `size_and_sign` for `Signed`, `Float`, `Double`,
/// and pointer types.
pub fn size_and_sign<T: HasSign>() -> (usize, bool) {
    (core::mem::size_of::<T>(), T::UNSIGNED)
}

/// `rffi.sizeof` for a concrete C type (`INT`, a struct, a pointer, ...).
pub fn sizeof<T>() -> usize {
    core::mem::size_of::<T>()
}

/// Signedness half of `size_and_sign`. Sealed the same way as [`CastFrom`].
pub trait HasSign: seal::Seal {
    const UNSIGNED: bool;
}

/// `ptr + n` in C (`force_ptradd`): `n` is a count of `T`, not bytes.
///
/// # Safety
/// `p` and `p + n` must follow the same rules as a C pointer addition.
pub unsafe fn ptradd<T>(p: *mut T, n: SIGNED) -> *mut T {
    unsafe { p.offset(n) }
}

macro_rules! seal_types {
    ($($t:ty),+ $(,)?) => {$(
        impl seal::Seal for $t {}
    )+};
}

macro_rules! cast_pairs {
    ($($t:ty),+ $(,)?) => {
        cast_pairs!(@row [$($t),+] $($t),+);
    };
    (@row [$($dst:ty),+] $src:ty $(, $rest:ty)*) => {
        $(
            impl CastFrom<$src> for $dst {
                #[inline(always)]
                fn cast_from(value: $src) -> Self {
                    value as Self
                }
            }
        )+
        cast_pairs!(@row [$($dst),+] $($rest),*);
    };
    (@row [$($dst:ty),+]) => {};
}

macro_rules! sign_flags {
    (signed: $($s:ty),+; unsigned: $($u:ty),+ $(,)?) => {
        $(impl HasSign for $s { const UNSIGNED: bool = false; })+
        $(impl HasSign for $u { const UNSIGNED: bool = true; })+
    };
}

// Primitive types behind the `rffi` integer and float aliases. Aliases are
// not distinct types, so the impls live on the primitives (`c_int` is `i32`).
seal_types!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
);
cast_pairs!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
);
sign_flags!(
    signed: i8, i16, i32, i64, i128, isize, f32, f64;
    unsigned: u8, u16, u32, u64, u128, usize
);

impl<T> seal::Seal for *mut T {}
impl<T> seal::Seal for *const T {}
impl<T> HasSign for *mut T {
    const UNSIGNED: bool = false;
}
impl<T> HasSign for *const T {
    const UNSIGNED: bool = false;
}

macro_rules! cast_ptr_ints {
    ($($int:ty),+ $(,)?) => {$(
        impl<T> CastFrom<$int> for *mut T {
            #[inline(always)]
            fn cast_from(value: $int) -> Self {
                value as *mut T
            }
        }
        impl<T> CastFrom<$int> for *const T {
            #[inline(always)]
            fn cast_from(value: $int) -> Self {
                value as *const T
            }
        }
        impl<T> CastFrom<*mut T> for $int {
            #[inline(always)]
            fn cast_from(value: *mut T) -> Self {
                value as $int
            }
        }
        impl<T> CastFrom<*const T> for $int {
            #[inline(always)]
            fn cast_from(value: *const T) -> Self {
                value as $int
            }
        }
    )+};
}

cast_ptr_ints!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

impl<T, U> CastFrom<*mut U> for *mut T {
    #[inline(always)]
    fn cast_from(value: *mut U) -> Self {
        value as *mut T
    }
}
impl<T, U> CastFrom<*const U> for *mut T {
    #[inline(always)]
    fn cast_from(value: *const U) -> Self {
        value as *mut T
    }
}
impl<T, U> CastFrom<*mut U> for *const T {
    #[inline(always)]
    fn cast_from(value: *mut U) -> Self {
        value as *const T
    }
}
impl<T, U> CastFrom<*const U> for *const T {
    #[inline(always)]
    fn cast_from(value: *const U) -> Self {
        value as *const T
    }
}

/// `rffi.offsetof`. Field lookup is `core::mem::offset_of`.
#[macro_export]
macro_rules! offsetof {
    ($ty:ty, $field:tt) => {
        ::core::mem::offset_of!($ty, $field)
    };
}

/// `rffi.setintfield`: store `value` into `field`, casting to the field's
/// integer type. `value` needs a concrete integer type (a bare literal will
/// not infer which source `cast` should truncate).
#[macro_export]
macro_rules! setintfield {
    ($pdst:expr, $field:ident, $value:expr) => {{
        let ptr = $pdst;
        let value = $value;
        (*ptr).$field = $crate::rffi::cast(value);
    }};
}

/// `rffi.getintfield`: read `field` and cast it to `SIGNED`.
#[macro_export]
macro_rules! getintfield {
    ($pdst:expr, $field:ident) => {
        $crate::rffi::cast::<$crate::rffi::SIGNED>((*$pdst).$field)
    };
}

/// `rffi.structcopy`: copy every field of `*psrc` into `*pdst`.
/// `_get_structcopy_fn` asserts `PDST == PSRC`; the shared `T` is that
/// assertion. For a `#[repr(C)]` struct the byte copy writes every field
/// `_get_structcopy_fn` copies, plus `_hints['padding']` bytes no reader
/// observes.
///
/// # Safety
/// Both pointers must be valid and aligned for one `T` (overlap allowed).
pub unsafe fn structcopy<T: Copy>(pdst: *mut T, psrc: *const T) {
    unsafe { core::ptr::copy(psrc, pdst, 1) }
}

#[cfg(test)]
mod tests {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct X2 {
        x: isize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct X1 {
        a: isize,
        x2: X2,
        p: *mut X2,
    }

    #[test]
    fn structcopy() {
        let mut p2 = X2 { x: 123 };
        let p2_ptr = &mut p2 as *mut X2;
        let p1 = X1 {
            a: 5,
            x2: X2 { x: 456 },
            p: p2_ptr,
        };
        let mut p1bis = X1 {
            a: 0,
            x2: X2 { x: 0 },
            p: core::ptr::null_mut(),
        };
        let mut p2bis = X2 { x: 0 };
        unsafe {
            crate::rffi::structcopy(&mut p1bis, &p1);
        }
        assert_eq!(p1bis.a, 5);
        assert_eq!(p1bis.x2.x, 456);
        assert_eq!(p1bis.p, p2_ptr);
        unsafe {
            crate::rffi::structcopy(&mut p2bis, &p2);
        }
        assert_eq!(p2bis.x, 123);
    }

    #[test]
    fn structcopy_self() {
        let mut p2 = X2 { x: 123 };
        unsafe {
            crate::rffi::structcopy(&raw mut p2, &raw const p2);
        }
        assert_eq!(p2.x, 123);
    }
}
