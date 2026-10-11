//! C-ABI call stub dispatch shared between backends.
//!
//! `bh_call_i_dispatch` mirrors `rpython/jit/backend/llsupport/llmodel.py bh_call_i call_stub_i`:
//! it materializes a typed `extern "C" fn` from a raw funcptr and forwards
//! arguments in the calldescr declaration order preserved by `arg_classes`.
//! `bh_call_f_dispatch` and `bh_call_v_dispatch` are the float-returning and
//! void-returning parallels for `llmodel.py bh_call_f` and
//! `llmodel.py bh_call_v`.
//!
//! Upstream `descr.py:574` builds the generated call expression by walking
//! `self.arg_classes`, and `descr.py` builds `FuncType(ARGS, RESULT)`
//! from that same ordered class list. Matching that shape is required by the
//! Microsoft x64 positional-slot convention and is also correct under SysV and
//! AAPCS.

use majit_jitcode::codewriter::insns::MAX_HOST_CALL_ARITY;
use majit_jitcode::jitcode::{BhCallDescr, BhCallStub};

/// `descr.py TYPE()` collapsed to the C-ABI register classes the dispatch
/// table can express: `'i'`, `'r'` and `'L'` (`lltype.Signed`,
/// `llmemory.GCREF`, `lltype.SignedLongLong`) pass in an integer register;
/// `'f'` (`lltype.Float`) passes as `f64`; `'S'` (`lltype.SingleFloat`)
/// passes as `f32`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgClass {
    Int,
    Float,
    Single,
}

/// Class-sequence table shared by `bh_call_i_dispatch`, `bh_call_f_dispatch`,
/// and `bh_call_v_dispatch`. Each row is one monomorphic stub module; the
/// same list is the match in `lookup_stub_*` and the arms `dispatch_classes_body!`
/// used to inline. `$ret` plugs into both the function-pointer signature
/// and the dispatch function's return type.
///
/// `descr.py` / `descr.py CallDescr.create_call_stub` parity: the signature
/// is built in `arg_classes` declaration order, so `fn(f64, i64)` is dispatched
/// as `extern "C" fn(f64, i64)` rather than a class-blind
/// `extern "C" fn(i64, f64)`. That preserves SysV/AAPCS register-file order
/// and the Microsoft x64 positional argument slots alike.
///
/// Coverage: every ordered sequence up to 5 arguments, plus the all-`Int`
/// sequences on to `MAX_HOST_CALL_ARITY`. The float-carrying bound is mirrored
/// by `majit_jitcode::codewriter::jitcode::MAX_FLOAT_CARRYING_CALL_ARITY`,
/// which flags such a signature where the calldescr is built instead of at the
/// deopt that first runs it; widening the arms here means raising it there in
/// the same change (`majit-translate` cannot call into `majit-backend`, so the
/// bound is stated on both sides rather than shared).
macro_rules! invoke_ty {
    (Int) => {
        i64
    };
    (Float) => {
        f64
    };
    (Single) => {
        f32
    };
}

macro_rules! invoke_arg {
    (Int, $a:ident, $i:tt) => {
        $a[$i]
    };
    (Float, $a:ident, $i:tt) => {
        f64::from_bits($a[$i] as u64)
    };
    (Single, $a:ident, $i:tt) => {
        f32::from_bits($a[$i] as u32)
    };
}

macro_rules! invoke_with_idx {
    ($f:ident, $a:ident) => {{
        let _ = $a;
        $f()
    }};
    ($f:ident, $a:ident, $c0:ident) => {
        $f(invoke_arg!($c0, $a, 0))
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident) => {
        $f(invoke_arg!($c0, $a, 0), invoke_arg!($c1, $a, 1))
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
            invoke_arg!($c13, $a, 13),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
            invoke_arg!($c13, $a, 13),
            invoke_arg!($c14, $a, 14),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident, $c15:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
            invoke_arg!($c13, $a, 13),
            invoke_arg!($c14, $a, 14),
            invoke_arg!($c15, $a, 15),
        )
    };
}

macro_rules! invoke_stub {
    ($func:ident, $args:ident, $ret:ty $(, $class:ident)*) => {{
        let f: unsafe extern "C" fn($(invoke_ty!($class)),*) -> $ret =
            std::mem::transmute($func);
        invoke_with_idx!(f, $args $(, $class)*)
    }};
}

/// `longlong.singlefloat2int`: `rffi.cast(Signed, uint32)` via `intmask`.
fn singlefloat2int(value: f32) -> i64 {
    let bits = value.to_bits();
    if cfg!(target_pointer_width = "32") {
        bits as i32 as i64
    } else {
        bits as i64
    }
}

/// One module per ABI sequence: `call_i` / `call_f` / `call_v` are the
/// monomorphic stubs `descr.py CallDescr.create_call_stub` would emit.
/// `dispatch_classes_body!` is this table (via `lookup_stub_*`).
macro_rules! call_sig_table {
    ($apply:ident) => {
        $apply! {
        Z {  }
        I { Int }
        F { Float }
        II { Int, Int }
        IF { Int, Float }
        FI { Float, Int }
        FF { Float, Float }
        III { Int, Int, Int }
        IIF { Int, Int, Float }
        IFI { Int, Float, Int }
        IFF { Int, Float, Float }
        FII { Float, Int, Int }
        FIF { Float, Int, Float }
        FFI { Float, Float, Int }
        FFF { Float, Float, Float }
        IIII { Int, Int, Int, Int }
        IIIF { Int, Int, Int, Float }
        IIFI { Int, Int, Float, Int }
        IIFF { Int, Int, Float, Float }
        IFII { Int, Float, Int, Int }
        IFIF { Int, Float, Int, Float }
        IFFI { Int, Float, Float, Int }
        IFFF { Int, Float, Float, Float }
        FIII { Float, Int, Int, Int }
        FIIF { Float, Int, Int, Float }
        FIFI { Float, Int, Float, Int }
        FIFF { Float, Int, Float, Float }
        FFII { Float, Float, Int, Int }
        FFIF { Float, Float, Int, Float }
        FFFI { Float, Float, Float, Int }
        FFFF { Float, Float, Float, Float }
        IIIII { Int, Int, Int, Int, Int }
        IIIIF { Int, Int, Int, Int, Float }
        IIIFI { Int, Int, Int, Float, Int }
        IIIFF { Int, Int, Int, Float, Float }
        IIFII { Int, Int, Float, Int, Int }
        IIFIF { Int, Int, Float, Int, Float }
        IIFFI { Int, Int, Float, Float, Int }
        IIFFF { Int, Int, Float, Float, Float }
        IFIII { Int, Float, Int, Int, Int }
        IFIIF { Int, Float, Int, Int, Float }
        IFIFI { Int, Float, Int, Float, Int }
        IFIFF { Int, Float, Int, Float, Float }
        IFFII { Int, Float, Float, Int, Int }
        IFFIF { Int, Float, Float, Int, Float }
        IFFFI { Int, Float, Float, Float, Int }
        IFFFF { Int, Float, Float, Float, Float }
        FIIII { Float, Int, Int, Int, Int }
        FIIIF { Float, Int, Int, Int, Float }
        FIIFI { Float, Int, Int, Float, Int }
        FIIFF { Float, Int, Int, Float, Float }
        FIFII { Float, Int, Float, Int, Int }
        FIFIF { Float, Int, Float, Int, Float }
        FIFFI { Float, Int, Float, Float, Int }
        FIFFF { Float, Int, Float, Float, Float }
        FFIII { Float, Float, Int, Int, Int }
        FFIIF { Float, Float, Int, Int, Float }
        FFIFI { Float, Float, Int, Float, Int }
        FFIFF { Float, Float, Int, Float, Float }
        FFFII { Float, Float, Float, Int, Int }
        FFFIF { Float, Float, Float, Int, Float }
        FFFFI { Float, Float, Float, Float, Int }
        FFFFF { Float, Float, Float, Float, Float }
        I6 { Int, Int, Int, Int, Int, Int }
        I7 { Int, Int, Int, Int, Int, Int, Int }
        I8 { Int, Int, Int, Int, Int, Int, Int, Int }
        I9 { Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I10 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I11 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I12 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I13 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I14 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I15 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I16 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        }
    };
}

macro_rules! define_call_sig_stubs {
    ($($name:ident { $($class:ident),* })*) => {
        fn lookup_stub_i(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) -> i64 {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) -> i64 {
                            unsafe { invoke_stub!(func, args, i64 $(, $class)*) }
                        }
                        stub
                    }
                )*
                classes => lookup_single_i(classes),
            }
        }

        fn lookup_stub_f(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) -> f64 {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) -> f64 {
                            unsafe { invoke_stub!(func, args, f64 $(, $class)*) }
                        }
                        stub
                    }
                )*
                classes => lookup_single_f(classes),
            }
        }

        fn lookup_stub_v(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) {
                            unsafe { invoke_stub!(func, args, () $(, $class)*) }
                        }
                        stub
                    }
                )*
                classes => lookup_single_v(classes),
            }
        }

        fn lookup_stub_s(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) -> i64 {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) -> i64 {
                            let value: f32 =
                                unsafe { invoke_stub!(func, args, f32 $(, $class)*) };
                            singlefloat2int(value)
                        }
                        stub
                    }
                )*
                classes => lookup_single_s(classes),
            }
        }
    };
}

include!(concat!(env!("OUT_DIR"), "/single_stubs.rs"));

fn unsupported_call_sig(classes: &[ArgClass]) -> ! {
    // `descr.py CallDescr.create_call_stub` generates a
    // per-calldescr stub at translation time, so every class
    // sequence has a matching extern "C" signature. Rust has no
    // translation-time codegen equivalent here, so the dispatch
    // is a hand-rolled class-sequence table. Convergence path:
    // wire libffi (or an ABI adapter) so any sequence is
    // dispatchable; until then, callees outside the table panic
    // instead of silently corrupting registers.
    panic!(
        "bh_call dispatch: unsupported arg class sequence {classes:?}; \
         needs libffi for general dispatch"
    );
}

call_sig_table!(define_call_sig_stubs);

/// The static stub table has every `Int`/`Float` sequence through arity 5,
/// every sequence that contains `Single` through arity 5, and all-`Int`
/// through [`MAX_HOST_CALL_ARITY`].
pub fn call_stub_arm_exists(classes: &[ArgClass]) -> bool {
    let n = classes.len();
    let wide_float = classes
        .iter()
        .any(|class| matches!(class, ArgClass::Float | ArgClass::Single));
    n <= MAX_HOST_CALL_ARITY && (n <= 5 || !wide_float)
}

fn wasm_residual_host_call(
    func: usize,
    args: &[i64],
    classes: &[ArgClass],
    result: char,
    result_signed: bool,
    result_size: usize,
) -> Option<i64> {
    residual_host_call()
        .and_then(|hook| hook(func, args, classes, result, result_signed, result_size))
}

/// `llmodel.py bh_call_i` and `bh_call_r` share `lookup_stub_i`: both results
/// are a machine word. `result` is `'i'` or `'r'`, the class the host hook
/// compares with the table signature.
unsafe fn dispatch_word_stub(func: usize, classes: &[ArgClass], args: &[i64], result: char) -> i64 {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    if let Some(value) = wasm_residual_host_call(func, args, classes, result, false, 8) {
        return value;
    }
    unsafe { (lookup_stub_i(classes))(func, args) }
}

/// llmodel.py bh_call_i call_stub_i: ABI-correct dispatch in calldescr declaration
/// order.
///
/// Safety: `func` must be a valid function pointer matching `classes`, i.e. an
/// `extern "C" fn(...) -> i64` whose parameter list is the same ordered
/// Int/Float sequence and whose float slots are carried in `args` as
/// `f64::to_bits`.
///
/// The `-> i64` is a requirement on the target, not a convenience of the
/// transmute: this reads the whole return register, and a callee whose result
/// is narrower leaves the bits above it undefined. `#[jit_interp]` meets it by
/// registering a widening shim in place of such a target — see
/// `majit_ir::CallResultWord`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
/// wasm32 `call_indirect` type-checks the callee against the descr FUNC.
/// `descr.py` `CallDescr.create_call_stub` builds that type from the same
/// FUNC the calldescr came from, and every published target has that wasm
/// type. A callee with no single wasm function type, or a host import outside
/// the guest table, is handled by the host hook.
pub unsafe fn bh_call_i_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> i64 {
    unsafe { dispatch_word_stub(func, classes, args, 'i') }
}

/// `llmodel.py bh_call_r`. The target returns one pointer word; a missing
/// value is the null word. Shares `dispatch_word_stub` with `bh_call_i`.
/// A wasm32 table type that is a real `i32` is host-reflected.
///
/// # Safety
/// `func` must match `classes` and return one pointer word.
pub unsafe fn bh_call_r_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> i64 {
    unsafe { dispatch_word_stub(func, classes, args, 'r') }
}

/// llmodel.py bh_call_v: void-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching `classes`.
/// `descr.py create_call_stub` builds a real void-returning stub for
/// `RESULT == lltype.Void`; calling such a function through an `i64`-returning
/// transmute reads garbage from rax/x0, so the canonical
/// `BC_RESIDUAL_CALL_*_V` blackhole/trace path must use this dispatcher.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bh_call_v_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    if wasm_residual_host_call(func, args, classes, 'v', false, 8).is_some() {
        return;
    }
    unsafe { (lookup_stub_v(classes))(func, args) }
}

/// llmodel.py bh_call_f: f64-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching `classes`.
/// `descr.py create_call_stub` generates a real f64-returning stub for
/// `RESULT == lltype.Float`; the C ABI returns f64 in xmm0 / d0 rather than
/// rax / x0, so an `i64`-typed transmute would read uninitialized integer-bank
/// state.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bh_call_f_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> f64 {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    if let Some(bits) = wasm_residual_host_call(func, args, classes, 'f', false, 8) {
        return f64::from_bits(bits as u64);
    }
    unsafe { (lookup_stub_f(classes))(func, args) }
}

/// The ordered class sequence and matching positional argument list that
/// [`collect_call_args`] hands to `bh_call_*_dispatch`.
///
/// Fixed-size so a residual call allocates nothing. `MAX_HOST_CALL_ARITY` is
/// the same bound the dispatch table's all-`Int` arms stop at, so a signature
/// that does not fit here has no arm either.
pub struct CallArgs {
    classes: [ArgClass; MAX_HOST_CALL_ARITY],
    args: [i64; MAX_HOST_CALL_ARITY],
    len: usize,
}

impl CallArgs {
    fn with_arity(arity: usize) -> Self {
        assert!(
            arity <= MAX_HOST_CALL_ARITY,
            "bh_call dispatch: {arity} arguments exceeds MAX_HOST_CALL_ARITY \
             ({MAX_HOST_CALL_ARITY}); the dispatch table has no arm this wide"
        );
        Self {
            classes: [ArgClass::Int; MAX_HOST_CALL_ARITY],
            args: [0; MAX_HOST_CALL_ARITY],
            len: 0,
        }
    }

    fn push(&mut self, class: ArgClass, arg: i64) {
        self.classes[self.len] = class;
        self.args[self.len] = arg;
        self.len += 1;
    }

    pub fn classes(&self) -> &[ArgClass] {
        &self.classes[..self.len]
    }

    pub fn args(&self) -> &[i64] {
        &self.args[..self.len]
    }
}

/// `descr.py` `assert self.result_type in return_type` — the half of
/// `verify_types` that [`collect_call_args`] does not cover.
///
/// `accepted` is the caller's `return_type`, the literal each `bh_call_*`
/// passes in `llmodel.py/822/828/835`: `"iS"` (`history.INT + 'S'`),
/// `"r"`, `"fL"` (`history.FLOAT + 'L'`) and `"v"`.
///
/// This is not redundant with the argument-class check. Nothing on the call
/// path reads `result_type` to pick a dispatcher: the blackhole reaches one of
/// the four entry points because the codewriter emitted
/// `residual_call_*_i/_r/_f/_v`, so the two records can disagree. They
/// disagree silently — the dispatcher would read the callee's result out of
/// the register file the *opcode* named while the callee returned it in the
/// one `result_type` names, e.g. an f64 result taken from `rax`. Comparing
/// them here is what makes that a failure rather than a garbage value.
///
/// `debug_assert` mirrors upstream's `if not we_are_translated():` guard at
/// each call site — a translated JIT does not run this. `majit-backend` is not
/// one of the LLBC-extracted crates, so a release build really does drop it.
///
/// A NULL function pointer is rejected by every `bh_call_*` entry before this
/// check.  `llmodel.py::bh_call_i/r/f/v` has no fake-success path for a missing
/// callee; callers that carry `JitCode.fnaddr == None` must abort before backend
/// dispatch.
pub fn verify_result_type(result_type: char, accepted: &str) {
    debug_assert!(
        accepted.contains(result_type),
        "BhCallDescr.verify_types: result_type {result_type:?} is not one of {accepted:?}; \
         the emitted residual_call opcode and the calldescr disagree about the return register"
    );
}

/// `descr.py CallDescr.create_call_stub`: walk `arg_classes` once, pick the
/// monomorphic ABI stub, and record the bank mapping. `verify_result_type`
/// and the `verify_types` count of each class run here, not per call.
pub fn create_call_stub(arg_classes: &str, result_type: char) -> BhCallStub {
    let accepted = match result_type {
        'i' | 'S' => "iS",
        'r' => "r",
        'f' | 'L' => "fL",
        'v' => "v",
        _ => "",
    };
    verify_result_type(result_type, accepted);

    let mut slots = [0u8; MAX_HOST_CALL_ARITY];
    let mut classes_buf = [ArgClass::Int; MAX_HOST_CALL_ARITY];
    let mut arity = 0u8;
    let mut expect_i = 0u8;
    let mut expect_r = 0u8;
    let mut expect_f = 0u8;
    let mut ii = 0u8;
    let mut ri = 0u8;
    let mut fi = 0u8;

    for c in arg_classes.chars() {
        if arity as usize >= MAX_HOST_CALL_ARITY {
            panic!(
                "bh_call dispatch: {} arguments exceeds MAX_HOST_CALL_ARITY \
                 ({MAX_HOST_CALL_ARITY}); the dispatch table has no arm this wide",
                arity as usize + 1
            );
        }
        let (bank, class) = match c {
            'i' => (BhCallStub::BANK_I, ArgClass::Int),
            'r' => (BhCallStub::BANK_R, ArgClass::Int),
            'f' => (BhCallStub::BANK_F, ArgClass::Float),
            'L' => (BhCallStub::BANK_F, ArgClass::Int),
            // descr.py process('S'): storage bank = `args_i`, C type = float.
            'S' => (BhCallStub::BANK_I, ArgClass::Single),
            other => panic!(
                "BhCallDescr.collect_call_args: unsupported arg class {other:?} \
                 in arg_classes={arg_classes:?}"
            ),
        };
        let idx = match bank {
            BhCallStub::BANK_I => {
                let i = ii;
                ii += 1;
                expect_i += 1;
                i
            }
            BhCallStub::BANK_R => {
                let i = ri;
                ri += 1;
                expect_r += 1;
                i
            }
            _ => {
                let i = fi;
                fi += 1;
                expect_f += 1;
                i
            }
        };
        slots[arity as usize] = (bank << 6) | idx;
        classes_buf[arity as usize] = class;
        arity += 1;
    }

    let classes = &classes_buf[..arity as usize];
    // descr.py result 'S': the callee returns C `float`, then
    // `singlefloat2int` stores the bits in the int result.
    let call_i = if result_type == 'S' {
        lookup_stub_s(classes)
    } else {
        lookup_stub_i(classes)
    };
    BhCallStub::new(
        slots,
        arity,
        expect_i,
        expect_r,
        expect_f,
        call_i,
        lookup_stub_f(classes),
        lookup_stub_v(classes),
    )
}

fn reflected_result(calldescr: &BhCallDescr) -> (char, bool, usize) {
    if calldescr.void_word_abi {
        ('i', false, 8)
    } else {
        (
            calldescr.result_type,
            calldescr.result_signed,
            calldescr.result_size,
        )
    }
}

fn call_stub_for(calldescr: &BhCallDescr) -> &BhCallStub {
    calldescr
        .call_stub
        .get_or_init(|| create_call_stub(&calldescr.arg_classes, calldescr.result_type))
}

/// `llmodel.py AbstractLLCPU.bh_call_i`: read the per-descr stub and call.
///
/// # Safety
/// `func` must match the ABI [`create_call_stub`] derives from `arg_classes`.
pub unsafe fn bh_call_i_with_descr(
    func: usize,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
    calldescr: &BhCallDescr,
) -> i64 {
    if residual_host_call().is_some() {
        let collected = collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        let (result_type, result_signed, result_size) = reflected_result(calldescr);
        if let Some(result) = wasm_residual_host_call(
            func,
            collected.args(),
            collected.classes(),
            result_type,
            result_signed,
            result_size,
        ) {
            return result;
        }
    }
    unsafe { call_stub_for(calldescr).call_i(func, args_i, args_r, args_f) }
}

/// `llmodel.py AbstractLLCPU.bh_call_f` parallel of [`bh_call_i_with_descr`].
///
/// # Safety
/// See [`bh_call_i_with_descr`].
pub unsafe fn bh_call_f_with_descr(
    func: usize,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
    calldescr: &BhCallDescr,
) -> f64 {
    if residual_host_call().is_some() {
        let collected = collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        let (result_type, result_signed, result_size) = reflected_result(calldescr);
        if let Some(bits) = wasm_residual_host_call(
            func,
            collected.args(),
            collected.classes(),
            result_type,
            result_signed,
            result_size,
        ) {
            return f64::from_bits(bits as u64);
        }
    }
    unsafe { call_stub_for(calldescr).call_f(func, args_i, args_r, args_f) }
}

/// `llmodel.py AbstractLLCPU.bh_call_v` parallel of [`bh_call_i_with_descr`].
///
/// # Safety
/// See [`bh_call_i_with_descr`].
pub unsafe fn bh_call_v_with_descr(
    func: usize,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
    calldescr: &BhCallDescr,
) {
    if residual_host_call().is_some() {
        let collected = collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        let (result_type, result_signed, result_size) = if calldescr.void_word_abi {
            ('i', false, 8)
        } else {
            ('v', false, 0)
        };
        if wasm_residual_host_call(
            func,
            collected.args(),
            collected.classes(),
            result_type,
            result_signed,
            result_size,
        )
        .is_some()
        {
            return;
        }
        if calldescr.void_word_abi {
            let _ = unsafe { call_stub_for(calldescr).call_i(func, args_i, args_r, args_f) };
            return;
        }
    }
    unsafe { call_stub_for(calldescr).call_v(func, args_i, args_r, args_f) }
}

/// Build the C-ABI class sequence and positional argument list from
/// `args_i` / `args_r` / `args_f`, following `calldescr.arg_classes` order.
///
/// `arg_classes` is the per-argument class string from
/// `majit_jitcode::jitcode::BhCallDescr`. RPython
/// `rpython/jit/backend/llsupport/descr.py create_call_stub`'s
/// `process(c)` walks the class string in declaration order and pulls the next
/// item out of the corresponding storage bank.
///
/// | class | storage bank | C ABI type              | dispatch class |
/// |-------|--------------|-------------------------|----------------|
/// | `i`   | `args_i`     | `lltype.Signed`         | Int            |
/// | `r`   | `args_r`     | `llmemory.GCREF`        | Int            |
/// | `f`   | `args_f`     | `lltype.Float`          | Float          |
/// | `L`   | `args_f`     | `lltype.SignedLongLong` | Int            |
/// | `S`   | `args_i`     | `lltype.SingleFloat`    | Single         |
///
/// Note the asymmetry: `L` is stored in the float bank (PyPy `process('L')`
/// rewrites `c = 'f'` for the storage lookup) yet passed in an integer
/// register, while `S` is stored in the int bank (PyPy
/// `int2singlefloat(args_i[..])`) yet passed as C `float`.
///
/// Mirrors `rpython/jit/backend/llsupport/descr.py verify_types`:
/// the per-class counts in `arg_classes` must match the corresponding list
/// length, and any unknown class is a codegen bug.
///
/// Returns a stack buffer rather than two `Vec`s: this runs on every residual
/// call the blackhole makes, and upstream's generated stub reaches the callee
/// with no intermediate collection at all.
pub fn collect_call_args(
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> CallArgs {
    // descr.py verify_types parity: assert per-class counts.
    let count_i: usize = arg_classes
        .chars()
        .filter(|c| matches!(c, 'i' | 'S'))
        .count();
    let count_r: usize = arg_classes.chars().filter(|c| *c == 'r').count();
    let count_f: usize = arg_classes
        .chars()
        .filter(|c| matches!(c, 'f' | 'L'))
        .count();
    let len_i = args_i.map_or(0, <[i64]>::len);
    let len_r = args_r.map_or(0, <[i64]>::len);
    let len_f = args_f.map_or(0, <[i64]>::len);
    assert_eq!(
        count_i, len_i,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_i} int slots, args_i has {len_i}"
    );
    assert_eq!(
        count_r, len_r,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_r} ref slots, args_r has {len_r}"
    );
    assert_eq!(
        count_f, len_f,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_f} float slots, args_f has {len_f}"
    );

    let mut out = CallArgs::with_arity(count_i + count_r + count_f);
    let mut ii = 0usize;
    let mut ri = 0usize;
    let mut fi = 0usize;
    for c in arg_classes.chars() {
        match c {
            'i' => {
                out.push(
                    ArgClass::Int,
                    args_i.expect("BhCallDescr.collect_call_args: args_i missing")[ii],
                );
                ii += 1;
            }
            'r' => {
                out.push(
                    ArgClass::Int,
                    args_r.expect("BhCallDescr.collect_call_args: args_r missing")[ri],
                );
                ri += 1;
            }
            'f' => {
                out.push(
                    ArgClass::Float,
                    args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi],
                );
                fi += 1;
            }
            'L' => {
                // descr.py process('L'): storage bank = `args_f`
                // (PyPy rewrites `c = 'f'` for the lookup); FUNC parameter
                // type = `lltype.SignedLongLong` -> C `long long` ->
                // 8-byte int dispatched in an integer register.
                out.push(
                    ArgClass::Int,
                    args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi],
                );
                fi += 1;
            }
            'S' => {
                // descr.py process('S'): storage bank = `args_i`
                // (`int2singlefloat`); FUNC parameter type = C `float`.
                out.push(
                    ArgClass::Single,
                    args_i.expect("BhCallDescr.collect_call_args: args_i missing")[ii],
                );
                ii += 1;
            }
            other => panic!(
                "BhCallDescr.collect_call_args: unsupported arg class {other:?} \
                 in arg_classes={arg_classes:?}"
            ),
        }
    }
    out
}

/// Bucket `args_i` / `args_r` / `args_f` into a single positional list in
/// `arg_classes` order, floats carried as their raw 64-bit pattern.
///
/// Unlike [`collect_call_args`] this does NOT split the arguments into the
/// SysV/AAPCS integer + float register files — it preserves the original
/// declaration order so a positional ABI (the wasm `call_indirect`, where
/// arguments are stack-positional rather than register-file partitioned)
/// receives them as the callee declares them. Used by the residual-host-call
/// trampoline path (see [`residual_host_call`]).
pub fn collect_call_args_positional(
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(arg_classes.len());
    let mut ii = 0usize;
    let mut ri = 0usize;
    let mut fi = 0usize;
    for c in arg_classes.chars() {
        match c {
            'i' => {
                out.push(args_i.expect("collect_call_args_positional: args_i missing")[ii]);
                ii += 1;
            }
            'r' => {
                out.push(args_r.expect("collect_call_args_positional: args_r missing")[ri]);
                ri += 1;
            }
            // `f` (Float) and `L` (SignedLongLong) both live in the `args_f`
            // storage bank; their raw 64-bit slot value is forwarded verbatim
            // — the host trampoline coerces it to f64/i64 from the callee's
            // reflected parameter type.
            'f' | 'L' => {
                out.push(args_f.expect("collect_call_args_positional: args_f missing")[fi]);
                fi += 1;
            }
            other => panic!(
                "collect_call_args_positional: unsupported arg class {other:?} \
                 in arg_classes={arg_classes:?}"
            ),
        }
    }
    out
}

/// Dispatch a residual call described by `arg_classes`, picking the signature
/// strategy the active backend supports.
///
/// `bh_call_*_dispatch` transmutes the funcptr to an `extern "C" fn` built
/// from `arg_classes` declaration order, matching `descr.py create_call_stub`.
/// `'r'` shares the integer-register class with `'i'`.
///
/// # Safety
/// On the transmute path, `func` must match the ABI [`collect_call_args`]
/// derives from `arg_classes` — see [`bh_call_i_dispatch`].
pub unsafe fn bh_call_i_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> i64 {
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_i_dispatch(func, collected.classes(), collected.args()) }
}

/// Ref-result parallel of [`bh_call_i_by_classes`].
///
/// # Safety
/// See [`bh_call_i_by_classes`].
pub unsafe fn bh_call_ref_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> i64 {
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_r_dispatch(func, collected.classes(), collected.args()) }
}

/// f64-returning parallel of [`bh_call_i_by_classes`].
///
/// # Safety
/// See [`bh_call_i_by_classes`].
pub unsafe fn bh_call_f_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> f64 {
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_f_dispatch(func, collected.classes(), collected.args()) }
}

/// Result-discarding parallel of [`bh_call_i_by_classes`].
///
/// # Safety
/// See [`bh_call_i_by_classes`].
pub unsafe fn bh_call_v_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) {
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_v_dispatch(func, collected.classes(), collected.args()) }
}

/// A host-provided trampoline that performs a residual call by reflecting the
/// callee's real signature, rather than transmuting the raw funcptr to a
/// statically-guessed `extern "C" fn`.
///
/// `func_ptr` is the raw callee address (a table index on wasm32); `args` is
/// the positional argument list (floats as raw bits). The return value is the
/// callee result as a 64-bit pattern (Void callees return 0; Ref returns the
/// pointer; Float returns `f64::to_bits`).
///
/// Installed only where in-module static-signature dispatch is impossible:
/// the wasm32 backend, whose `call_indirect` type-checks every call and so
/// cannot reuse the uniform-`i64` transmute that the SysV/AAPCS C ABI tolerates
/// on native backends. `None` (the default on dynasm/cranelift) keeps the
/// direct transmute path.
/// `None` means the callee's wasm type matches the stub, so the caller uses it.
/// `Some` is the host result when the types differ.
pub type ResidualHostCallFn = fn(
    func_ptr: usize,
    args: &[i64],
    classes: &[ArgClass],
    result: char,
    result_signed: bool,
    result_size: usize,
) -> Option<i64>;

/// The CPU's residual-call strategy, process-global and installed once at startup.
///
/// `llmodel.py AbstractLLCPU.bh_call_i` is a method of the one CPU per process.
/// This hook is that strategy for the wasm32 backend and is read on every
/// residual call. The word holds a [`ResidualHostCallFn`] as `usize` bits, or
/// 0 when none is installed and dispatch uses the direct transmute.
static RESIDUAL_HOST_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

const _: () = assert!(std::mem::size_of::<ResidualHostCallFn>() == std::mem::size_of::<usize>());

/// Install the CPU's residual-call strategy. Pass `None` to clear it.
///
/// Process-global, installed once at startup. See
/// `llmodel.py AbstractLLCPU.bh_call_i`.
pub fn set_residual_host_call(hook: Option<ResidualHostCallFn>) {
    let bits = match hook {
        Some(hook) => hook as usize,
        None => 0,
    };
    RESIDUAL_HOST_CALL.store(bits, std::sync::atomic::Ordering::Release);
}

/// The CPU's residual-call strategy, or `None` for the direct transmute.
///
/// Process-global, installed once at startup. See
/// `llmodel.py AbstractLLCPU.bh_call_i`.
pub fn residual_host_call() -> Option<ResidualHostCallFn> {
    let bits = RESIDUAL_HOST_CALL.load(std::sync::atomic::Ordering::Acquire);
    if bits == 0 {
        None
    } else {
        // SAFETY: a non-zero word was stored from `ResidualHostCallFn as usize`.
        // Rust function pointers are never null, and the two types have equal size.
        Some(unsafe { std::mem::transmute::<usize, ResidualHostCallFn>(bits) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The residual-call strategy is process-global, so tests that install it
    /// and tests that call `bh_call_*` / `*_dispatch` must not overlap.
    static HOOK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn hold_hook_test_lock() -> std::sync::MutexGuard<'static, ()> {
        HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    extern "C" fn single_int(a: f32, b: i64) -> i64 {
        a.to_bits() as i64 + b
    }

    extern "C" fn single_ret(a: f32) -> f32 {
        a + 1.0
    }

    extern "C" fn f2(a: f64, b: *const i64) -> f64 {
        a + unsafe { *b } as f64
    }

    extern "C" fn int_float_int(a: i64, b: f64, c: i64) -> i64 {
        a + b as i64 * 10 + c * 100
    }

    extern "C" fn float_float_int(a: f64, b: f64, c: i64) -> f64 {
        a + b * 10.0 + c as f64 * 100.0
    }

    /// `descr.py process('S')`: the bits live in `args_i` and the callee
    /// receives a C `float`.
    #[test]
    fn call_stub_i_singlefloat_passes_f32() {
        let _hook_test = hold_hook_test_lock();
        let bits = 1.5f32.to_bits() as i64;
        let collected = collect_call_args("Si", Some(&[bits, 7]), None, None);
        assert_eq!(collected.classes(), &[ArgClass::Single, ArgClass::Int]);
        let result = unsafe {
            bh_call_i_dispatch(
                single_int as *const () as usize,
                collected.classes(),
                collected.args(),
            )
        };
        assert_eq!(result, bits + 7);
    }

    /// `descr.py` result `'S'`: `singlefloat2int` of the callee's `float`.
    #[test]
    fn call_stub_result_singlefloat_returns_f32_bits() {
        let stub = create_call_stub("S", 'S');
        let bits = 1.5f32.to_bits() as i64;
        let result =
            unsafe { stub.call_i(single_ret as *const () as usize, Some(&[bits]), None, None) };
        assert_eq!(result, singlefloat2int(2.5));
        let neg = singlefloat2int(-2.0);
        let neg_result =
            unsafe { stub.call_i(single_ret as *const () as usize, Some(&[neg]), None, None) };
        assert_eq!(neg_result, singlefloat2int(-1.0));
    }

    /// Rust port of
    /// `rpython/jit/backend/llsupport/test/test_descr.py::test_call_stubs_2`.
    #[test]
    fn call_stub_f_interleaved_float_ref_preserves_declaration_order() {
        let _hook_test = hold_hook_test_lock();
        let b = [1_i64];
        let result = unsafe {
            bh_call_f_dispatch(
                f2 as *const () as usize,
                &[ArgClass::Float, ArgClass::Int],
                &[3.5_f64.to_bits() as i64, b.as_ptr() as i64],
            )
        };
        assert_eq!(result, 4.5);
    }

    #[test]
    fn call_stub_i_interleaved_int_float_int_preserves_declaration_order() {
        let _hook_test = hold_hook_test_lock();
        let result = unsafe {
            bh_call_i_dispatch(
                int_float_int as *const () as usize,
                &[ArgClass::Int, ArgClass::Float, ArgClass::Int],
                &[1, 2.0_f64.to_bits() as i64, 3],
            )
        };
        assert_eq!(result, 321);
    }

    #[test]
    fn call_stub_f_interleaved_float_float_int_preserves_declaration_order() {
        let _hook_test = hold_hook_test_lock();
        let result = unsafe {
            bh_call_f_dispatch(
                float_float_int as *const () as usize,
                &[ArgClass::Float, ArgClass::Float, ArgClass::Int],
                &[1.0_f64.to_bits() as i64, 2.0_f64.to_bits() as i64, 3],
            )
        };
        assert_eq!(result, 321.0);
    }

    extern "C" fn four_ints_then_float(a: i64, b: i64, c: i64, d: i64, e: f64) -> i64 {
        a + b * 10 + c * 100 + d * 1000 + e as i64 * 10000
    }

    /// The widest float-carrying sequence the table covers, and the bound
    /// `majit_jitcode::codewriter::jitcode::MAX_FLOAT_CARRYING_CALL_ARITY`
    /// states on the descr-build side.
    #[test]
    fn call_stub_i_dispatches_a_float_in_the_last_covered_slot() {
        let _hook_test = hold_hook_test_lock();
        let result = unsafe {
            bh_call_i_dispatch(
                four_ints_then_float as *const () as usize,
                &[
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Float,
                ],
                &[1, 2, 3, 4, 5.0_f64.to_bits() as i64],
            )
        };
        assert_eq!(result, 54321);
    }

    /// `collect_call_args` fills a fixed buffer, so a signature wider than the
    /// dispatch table's widest arm is refused while collecting rather than
    /// overrunning it.
    #[test]
    #[should_panic(expected = "exceeds MAX_HOST_CALL_ARITY")]
    fn collect_call_args_refuses_more_arguments_than_the_table_covers() {
        let too_many = MAX_HOST_CALL_ARITY + 1;
        let args_i = vec![0_i64; too_many];
        let _ = collect_call_args(&"i".repeat(too_many), Some(&args_i), None, None);
    }

    /// One argument past that bound the table has no arm, so the call is
    /// refused instead of being placed against the wrong signature.
    #[test]
    #[should_panic(expected = "unsupported arg class sequence")]
    fn call_stub_i_refuses_a_float_past_the_covered_width() {
        let _hook_test = hold_hook_test_lock();
        unsafe {
            bh_call_i_dispatch(
                four_ints_then_float as *const () as usize,
                &[
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Float,
                ],
                &[1, 2, 3, 4, 5, 6.0_f64.to_bits() as i64],
            );
        }
    }

    /// The four `return_type` literals `llmodel.py bh_call_i/822/828/835` pass, each
    /// against every result class `type_to_argclass` can produce. `'L'` and
    /// `'S'` ride along with the float and int sets the way upstream spells
    /// them (`history.FLOAT + 'L'`, `history.INT + 'S'`).
    #[test]
    fn verify_result_type_accepts_exactly_its_caller_s_return_classes() {
        for (accepted, ok) in [("iS", "iS"), ("r", "r"), ("fL", "fL"), ("v", "v")] {
            for c in ok.chars() {
                verify_result_type(c, accepted);
            }
        }
    }

    /// The mis-route this exists to catch: a float-returning callee reached
    /// through the integer entry point would take its result from `rax`.
    ///
    /// Only meaningful where the assertion is compiled in — `verify_result_type`
    /// is a `debug_assert`, so a release test binary would not panic.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not one of \"iS\"")]
    fn verify_result_type_rejects_a_float_result_on_the_int_entry_point() {
        verify_result_type('f', "iS");
    }

    /// `BhCallDescr::default()` leaves `result_type` at `char::default()`.
    /// `jitdriver.rs` reaches that through `unwrap_or_default()`, so the
    /// sentinel has to be rejected rather than silently matching some class.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not one of \"r\"")]
    fn verify_result_type_rejects_the_default_descr_s_null_result_type() {
        verify_result_type('\0', "r");
    }

    /// Every sequence through arity 5, and all-`Int` through
    /// [`MAX_HOST_CALL_ARITY`]. A float past arity 5 has no arm.
    #[test]
    fn call_stub_arm_exists_covers_arity_5_and_all_int() {
        assert!(call_stub_arm_exists(&[]));
        assert!(call_stub_arm_exists(&[ArgClass::Int; 5]));
        assert!(call_stub_arm_exists(&[
            ArgClass::Float,
            ArgClass::Int,
            ArgClass::Float,
            ArgClass::Int,
            ArgClass::Float,
        ]));
        assert!(call_stub_arm_exists(&[ArgClass::Int; 8]));
        assert!(call_stub_arm_exists(&[ArgClass::Int; MAX_HOST_CALL_ARITY]));
        let collected =
            collect_call_args("iriririr", Some(&[1, 2, 3, 4]), Some(&[5, 6, 7, 8]), None);
        assert!(collected.classes().iter().all(|c| *c == ArgClass::Int));
        assert!(call_stub_arm_exists(collected.classes()));
        assert!(call_stub_arm_exists(&[ArgClass::Single, ArgClass::Int]));
        assert!(!call_stub_arm_exists(&[ArgClass::Single; 6]));
        assert!(!call_stub_arm_exists(&[ArgClass::Float; 6]));
        let mut past = [ArgClass::Int; 6];
        past[3] = ArgClass::Float;
        assert!(!call_stub_arm_exists(&past));
        assert!(!call_stub_arm_exists(
            &[ArgClass::Int; MAX_HOST_CALL_ARITY + 1]
        ));
    }

    /// The installed hook runs before stub lookup and its `Some` result is the call's result.
    #[test]
    fn installed_host_hook_runs_before_the_stub_table() {
        let _hook_test = hold_hook_test_lock();
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                set_residual_host_call(None);
            }
        }
        let _clear = Clear;
        set_residual_host_call(Some(|func, args, _classes, result, signed, size| {
            assert_eq!(func, 7);
            assert_eq!(args, &[1, 5, 2, 6, 3, 7, 4, 8]);
            assert_eq!(result, 'i');
            assert!(signed);
            assert_eq!(size, 8);
            Some(42)
        }));
        let descr = BhCallDescr::from_arg_classes(
            "iriririr".to_string(),
            'i',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let result = unsafe {
            bh_call_i_with_descr(7, Some(&[1, 2, 3, 4]), Some(&[5, 6, 7, 8]), None, &descr)
        };
        assert_eq!(result, 42);
    }

    /// `'L'` is an i64 return stored in the float bank. The host result is the
    /// raw bits; `bh_call_f_with_descr` reinterprets them.
    #[test]
    fn host_hook_longlong_result_is_stored_as_float_bits() {
        let _hook_test = hold_hook_test_lock();
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                set_residual_host_call(None);
            }
        }
        let _clear = Clear;
        set_residual_host_call(Some(|_func, _args, _classes, result, _signed, _size| {
            assert_eq!(result, 'L');
            Some(f64::to_bits(1.0) as i64)
        }));
        let descr = BhCallDescr::from_arg_classes(
            "i".to_string(),
            'L',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let value = unsafe { bh_call_f_with_descr(3, Some(&[9]), None, None, &descr) };
        assert_eq!(value, 1.0);
    }

    /// `descr.py CallDescr.create_call_stub` + `llmodel.py AbstractLLCPU.bh_call_f`
    /// for the interleaved float/ref case of `test_call_stubs_2`.
    #[test]
    fn create_call_stub_f_interleaved_float_ref_preserves_declaration_order() {
        let _hook_test = hold_hook_test_lock();
        let b = [1_i64];
        let descr = BhCallDescr::from_arg_classes(
            "fr".to_string(),
            'f',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let result = unsafe {
            bh_call_f_with_descr(
                f2 as *const () as usize,
                None,
                Some(&[b.as_ptr() as i64]),
                Some(&[3.5_f64.to_bits() as i64]),
                &descr,
            )
        };
        assert_eq!(result, 4.5);
    }

    /// `shift` (`arg_classes = "rii"`) is the regex residual that the stub
    /// must serve without walking the class string on the second call.
    #[test]
    fn create_call_stub_i_rii_places_ref_then_two_ints() {
        let _hook_test = hold_hook_test_lock();
        extern "C" fn rii(a: i64, b: i64, c: i64) -> i64 {
            a + b * 10 + c * 100
        }
        let descr = BhCallDescr::from_arg_classes(
            "rii".to_string(),
            'i',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let result = unsafe {
            bh_call_i_with_descr(
                rii as *const () as usize,
                Some(&[2, 3]),
                Some(&[1]),
                None,
                &descr,
            )
        };
        assert_eq!(result, 321);
        let again = unsafe {
            bh_call_i_with_descr(
                rii as *const () as usize,
                Some(&[2, 3]),
                Some(&[1]),
                None,
                &descr,
            )
        };
        assert_eq!(again, 321);
        assert!(descr.call_stub.get().is_some());
    }
}
