//! `rffi.py _make_wrapper_for`: the C-callable wrapper `llexternal` puts around a callback
//! argument.

use std::any::Any;
use std::io::Write;
use std::sync::OnceLock;

/// `lltype` `_defl()` of a callback result.
pub trait Defl {
    fn _defl() -> Self;
}

impl Defl for () {
    fn _defl() -> Self {}
}

macro_rules! impl_defl_int {
    ($($t:ty),+) => {
        $(
            impl Defl for $t {
                fn _defl() -> Self {
                    0
                }
            }
        )+
    };
}
impl_defl_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl Defl for f32 {
    fn _defl() -> Self {
        0.0
    }
}

impl Defl for f64 {
    fn _defl() -> Self {
        0.0
    }
}

impl Defl for bool {
    fn _defl() -> Self {
        false
    }
}

impl<T> Defl for *mut T {
    fn _defl() -> Self {
        std::ptr::null_mut()
    }
}

impl<T> Defl for *const T {
    fn _defl() -> Self {
        std::ptr::null()
    }
}

/// `rgil.py acquire_maybe_in_new_thread`. The runtime that embeds majit installs the entry that also
/// brings a thread it has never seen into its per-thread state (`rthread.get_or_make_ident`,
/// `rthread.gc_thread_run`). The embedding runtime must install it before handing any callback to C
/// (pyre does so in `enter_runtime_thread`).
static ACQUIRE_MAYBE_IN_NEW_THREAD: OnceLock<fn() -> majit_gc::gc_sync::CallbackGuard> =
    OnceLock::new();

pub fn set_acquire_maybe_in_new_thread(entry: fn() -> majit_gc::gc_sync::CallbackGuard) {
    let _ = ACQUIRE_MAYBE_IN_NEW_THREAD.set(entry);
}

fn acquire_maybe_in_new_thread() -> majit_gc::gc_sync::CallbackGuard {
    let entry = ACQUIRE_MAYBE_IN_NEW_THREAD.get().unwrap_or_else(|| {
        panic!("no runtime installed `rgil.acquire_maybe_in_new_thread` (`set_acquire_maybe_in_new_thread`)")
    });
    entry()
}

/// `rffi.py _make_wrapper_for`. Monomorphization over a zero-sized fn-item `Self` is the
/// `specialize:memo`.
pub trait WrapFor<TP, const USE_GIL: bool> {
    fn _make_wrapper_for(self) -> TP;
}

fn uncaught_exception_in_callback(name: &str, payload: &dyn Any) {
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("");
    let _ = write!(
        std::io::stderr(),
        "Warning: uncaught exception in callback: {name} {message}\n"
    );
}

macro_rules! wrap_for_arity {
    ($wrapper:ident $(, $A:ident, $a:ident)*) => {
        extern "C" fn $wrapper<F $(, $A)*, R, const USE_GIL: bool>($($a: $A),*) -> R
        where
            F: Fn($($A),*) -> R,
            F: Copy + 'static,
            R: Defl,
        {
            let _gil = if USE_GIL {
                Some(acquire_maybe_in_new_thread())
            } else {
                None
            };
            // SAFETY: `WrapFor` admits only a zero-sized `F`, a fn item, whose one value needs no bytes.
            let callable: F = unsafe { std::mem::zeroed() };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || callable($($a),*))) {
                Ok(result) => result,
                Err(payload) => {
                    uncaught_exception_in_callback(std::any::type_name::<F>(), &*payload);
                    R::_defl()
                }
            }
        }

        impl<F $(, $A)*, R, const USE_GIL: bool> WrapFor<unsafe extern "C" fn($($A),*) -> R, USE_GIL>
            for F
        where
            F: Fn($($A),*) -> R,
            F: Copy + 'static,
            R: Defl,
        {
            fn _make_wrapper_for(self) -> unsafe extern "C" fn($($A),*) -> R {
                const {
                    assert!(
                        std::mem::size_of::<F>() == 0,
                        "an llexternal callback must be a function item (`_make_wrapper_for` memoizes per callable)"
                    );
                }
                let _ = self;
                $wrapper::<F $(, $A)*, R, USE_GIL>
            }
        }

        // `_isllptr(arg)`: an existing C function pointer reaches C unchanged.
        impl<$($A,)* R, const USE_GIL: bool> WrapFor<unsafe extern "C" fn($($A),*) -> R, USE_GIL>
            for unsafe extern "C" fn($($A),*) -> R
        {
            fn _make_wrapper_for(self) -> unsafe extern "C" fn($($A),*) -> R {
                self
            }
        }
    };
}

wrap_for_arity!(wrapper0);
wrap_for_arity!(wrapper1, A0, a0);
wrap_for_arity!(wrapper2, A0, a0, A1, a1);
wrap_for_arity!(wrapper3, A0, a0, A1, a1, A2, a2);
wrap_for_arity!(wrapper4, A0, a0, A1, a1, A2, a2, A3, a3);
wrap_for_arity!(wrapper5, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4);
wrap_for_arity!(wrapper6, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5);
wrap_for_arity!(
    wrapper7, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6
);
wrap_for_arity!(
    wrapper8, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7
);
wrap_for_arity!(
    wrapper9, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8
);
wrap_for_arity!(
    wrapper10, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9
);
wrap_for_arity!(
    wrapper11, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9, A10,
    a10
);
wrap_for_arity!(
    wrapper12, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9, A10,
    a10, A11, a11
);
wrap_for_arity!(
    wrapper13, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9, A10,
    a10, A11, a11, A12, a12
);
wrap_for_arity!(
    wrapper14, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9, A10,
    a10, A11, a11, A12, a12, A13, a13
);
wrap_for_arity!(
    wrapper15, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9, A10,
    a10, A11, a11, A12, a12, A13, a13, A14, a14
);
wrap_for_arity!(
    wrapper16, A0, a0, A1, a1, A2, a2, A3, a3, A4, a4, A5, a5, A6, a6, A7, a7, A8, a8, A9, a9, A10,
    a10, A11, a11, A12, a12, A13, a13, A14, a14, A15, a15
);

/// `rffi.py _isfunctype`: `llexternal!` recognises a callback parameter by its spelled type. This rejects a
/// function-pointer type it cannot see (a type alias, an `Option`), so a callback never reaches C unwrapped.
pub trait NotFuncType<M> {}
impl<T: ?Sized> NotFuncType<()> for T {}
pub const fn _assert_not_functype<T: ?Sized + NotFuncType<M>, M>() {}

macro_rules! not_functype_arity {
    ($($A:ident),*) => {
        impl<$($A,)* R> NotFuncType<u8> for unsafe extern "C" fn($($A),*) -> R {}
        impl<$($A,)* R> NotFuncType<u8> for extern "C" fn($($A),*) -> R {}
        impl<$($A,)* R> NotFuncType<u8> for Option<unsafe extern "C" fn($($A),*) -> R> {}
        impl<$($A,)* R> NotFuncType<u8> for Option<extern "C" fn($($A),*) -> R> {}
    };
}

not_functype_arity!();
not_functype_arity!(A0);
not_functype_arity!(A0, A1);
not_functype_arity!(A0, A1, A2);
not_functype_arity!(A0, A1, A2, A3);
not_functype_arity!(A0, A1, A2, A3, A4);
not_functype_arity!(A0, A1, A2, A3, A4, A5);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7, A8);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12);
not_functype_arity!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13);
not_functype_arity!(
    A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14
);
not_functype_arity!(
    A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15
);
