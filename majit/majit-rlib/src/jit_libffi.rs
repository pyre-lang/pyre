//! The `CIF_DESCRIPTION` block and the call shape a trace can specialize —
//! `rpython/rlib/jit_libffi.py`.
//!
//! For each C function, one `CIF_DESCRIPTION` block of raw memory is built
//! and every field but `cif` filled in; `atypes` points at an array of
//! `ffi_type *` that lives in the same block.  Following the four fields
//! `ffi_prep_cif` takes are the three this file adds:
//!
//! * `exchange_size` — how big a buffer a call must allocate.  The first
//!   `nargs` pointers of that buffer are the argument array
//!   [`jit_ffi_call_impl_any`] fills in.
//! * `exchange_result` — the offset in that buffer where the result lands.
//! * `exchange_args[nargs]` — the offset of each argument.
//!
//! Each argument and the result have room for at least `SIZE_OF_FFI_ARG`
//! bytes even when the value is smaller.
//!
//! # Why the call is split by result kind
//!
//! A trace turns the foreign call into `call_release_gil`.  Recording the
//! result inside the metainterp would leave the freshly built result box out
//! of the `guard_not_forced` fail arguments, so the call is a jitcode-level
//! `direct_call` of its own whose result the caller stores back into the
//! exchange buffer:
//!
//! ```text
//!     %i0 = direct_call(libffi_call, ...)
//!     -live-
//!     raw_store(exchange_result, %i0)
//! ```
//!
//! The `-live-` is what keeps the value across a failing `guard_not_forced`.
//! `jit_ffi_call_impl_int` and its siblings read the same word back out of
//! the buffer when they run interpreted, so the store is a no-op there and
//! the two modes agree.

#[cfg(any(unix, windows))]
mod imp {
    use crate::clibffi::{self, FFI_ABI, FFI_TYPE_P, FFI_TYPE_PP, ffi_arg, ffi_cif};
    use crate::lltypesystem::rffi as llrffi;
    use crate::rffi::{LONGLONG, UINT, VOIDP};

    /// `jit_libffi.py CIF_DESCRIPTION`.  An array of `nargs` exchange offsets
    /// follows the record; `atypes` and any struct `ffi_type` follow that, all
    /// inside the one block this describes.
    #[repr(C)]
    pub struct CifDescription {
        pub cif: ffi_cif,
        pub abi: FFI_ABI,
        pub nargs: usize,
        pub rtype: FFI_TYPE_P,
        pub atypes: FFI_TYPE_PP,
        pub exchange_size: usize,
        pub exchange_result: usize,
    }

    /// `jit_libffi.py SIZE_OF_FFI_ARG`.
    pub const SIZE_OF_FFI_ARG: usize = std::mem::size_of::<ffi_arg>();

    /// # Safety
    /// `cif` must be a `CIF_DESCRIPTION` block with `exchange_args[nargs]`
    /// following the record.
    pub unsafe fn header(cif: usize) -> &'static mut CifDescription {
        unsafe { &mut *(cif as *mut CifDescription) }
    }

    /// `cif_description.exchange_args`.
    ///
    /// # Safety
    /// As [`header`].
    pub unsafe fn exchange_args(cif: usize) -> *mut usize {
        (cif + std::mem::size_of::<CifDescription>()) as *mut usize
    }

    // The block carries `hints={'immutable': True}`: every field is filled in
    // before the first call through it and read-only afterwards.  Each reader
    // below is one elidable leaf, so a call through a promoted function type
    // reads them at trace time and the offsets it computes are constants.

    /// `cif_description.exchange_size`.
    ///
    /// # Safety
    /// As [`header`].
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn exchange_size(cif: usize) -> usize {
        unsafe { header(cif).exchange_size }
    }

    /// `cif_description.exchange_result`.
    ///
    /// # Safety
    /// As [`header`].
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn exchange_result(cif: usize) -> usize {
        unsafe { header(cif).exchange_result }
    }

    /// `cif_description.exchange_args[i]`.
    ///
    /// # Safety
    /// As [`header`]; `i` must be below `nargs`.
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn exchange_arg(cif: usize, i: usize) -> usize {
        unsafe { exchange_args(cif).add(i).read() }
    }

    /// `cif_description.nargs`.
    ///
    /// # Safety
    /// As [`header`].
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn nargs(cif: usize) -> usize {
        unsafe { header(cif).nargs }
    }

    /// `cif_description.rtype`.
    ///
    /// # Safety
    /// As [`header`].
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn rtype(cif: usize) -> usize {
        unsafe { header(cif).rtype as usize }
    }

    /// `cif_description.atypes[i]`.
    ///
    /// # Safety
    /// As [`header`]; `i` must be below `nargs`.
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn atype(cif: usize, i: usize) -> usize {
        unsafe { header(cif).atypes.add(i).read() as usize }
    }

    /// `cif_description.abi`.
    ///
    /// # Safety
    /// As [`header`].
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn abi(cif: usize) -> i64 {
        unsafe { header(cif).abi as i64 }
    }

    /// `jit_libffi.py jit_ffi_prep_cif` — minimal wrapper around
    /// `ffi_prep_cif()`.  Call this after `cif_description` is initialized,
    /// in order to fill the last field: `cif`.
    ///
    /// # Safety
    /// As [`header`].
    pub unsafe fn jit_ffi_prep_cif(cif_description: usize) -> i64 {
        let descr = unsafe { header(cif_description) };
        let res = unsafe {
            clibffi::c_ffi_prep_cif(
                &raw mut descr.cif,
                descr.abi,
                descr.nargs as UINT,
                descr.rtype,
                descr.atypes,
            )
        };
        res as i64
    }

    /// `jit_libffi.py jit_ffi_prep_cif_var` — same as [`jit_ffi_prep_cif`],
    /// but for a call to a variadic function.  `nfixedargs` is the number of
    /// arguments declared before the `...` in the C signature.  Some ABIs
    /// (notably arm64 on macOS/iOS) pass the variadic arguments differently
    /// from the fixed ones, so libffi must be told where the variadic part
    /// starts.
    ///
    /// # Safety
    /// As [`header`].
    pub unsafe fn jit_ffi_prep_cif_var(cif_description: usize, nfixedargs: usize) -> i64 {
        let descr = unsafe { header(cif_description) };
        let res = unsafe {
            clibffi::c_ffi_prep_cif_var(
                &raw mut descr.cif,
                descr.abi,
                nfixedargs as UINT,
                descr.nargs as UINT,
                descr.rtype,
                descr.atypes,
            )
        };
        res as i64
    }

    /// The `kind` letters [`types::getkind`] answers with.
    pub mod kind {
        /// `void`.
        pub const VOID: i64 = b'v' as i64;
        /// `double`.
        pub const FLOAT: i64 = b'f' as i64;
        /// A signed integer.
        pub const SIGNED: i64 = b'i' as i64;
        /// An unsigned integer.
        pub const UNSIGNED: i64 = b'u' as i64;
        /// `float`, which travels in an integer-sized slot.
        pub const SINGLEFLOAT: i64 = b'S' as i64;
        /// An integer wider than a machine word.
        pub const LONGLONG: i64 = b'L' as i64;
        /// A struct passed by value.
        pub const STRUCT: i64 = b'*' as i64;
        /// Anything else, `long double` among it.
        pub const OTHER: i64 = b'?' as i64;
    }

    /// `jit_libffi.py types` — the mapping the JIT needs from an `ffi_type`
    /// to a less strict kind letter.
    pub mod types {
        use super::kind;
        use crate::clibffi::{self, FFI_TYPE_STRUCT};

        /// `types.slong` — `cast_type_to_ffitype(rffi.LONG)`.
        fn slong() -> usize {
            clibffi::_signed_type_for(std::mem::size_of::<std::ffi::c_long>()) as usize
        }

        /// `types.ulong` — `cast_type_to_ffitype(rffi.ULONG)`.
        fn ulong() -> usize {
            clibffi::_unsigned_type_for(std::mem::size_of::<std::ffi::c_ulong>()) as usize
        }

        /// `types.signed` — `cast_type_to_ffitype(rffi.SIGNED)`.  Win64 is
        /// where this differs from [`slong`]: a machine word is eight bytes
        /// there while a `long` is four.
        fn signed() -> usize {
            clibffi::_signed_type_for(std::mem::size_of::<isize>()) as usize
        }

        /// `types.unsigned` — `cast_type_to_ffitype(rffi.UNSIGNED)`.
        fn unsigned() -> usize {
            clibffi::_unsigned_type_for(std::mem::size_of::<usize>()) as usize
        }

        /// `types.getkind(ffi_type)`.
        ///
        /// The `slong` / `ulong` / `signed` / `unsigned` records are asked
        /// before the explicit 64-bit ones, and they *are* the 64-bit records
        /// wherever a `long` or a machine word is eight bytes wide, so an
        /// eight-byte integer answers `i` / `u` there and only a 32-bit
        /// platform reaches `L`.
        ///
        /// # Safety
        /// `ffi_type` must be an `ffi_type` record, or zero.
        #[majit_macros::elidable_cannot_raise]
        pub unsafe fn getkind(ffi_type: usize) -> i64 {
            if ffi_type == 0 {
                return kind::OTHER;
            }
            if ffi_type == clibffi::ffi_type_void() as usize {
                return kind::VOID;
            }
            if ffi_type == clibffi::ffi_type_double() as usize {
                return kind::FLOAT;
            }
            if ffi_type == clibffi::ffi_type_float() as usize {
                return kind::SINGLEFLOAT;
            }
            if ffi_type == clibffi::ffi_type_pointer() as usize {
                return kind::UNSIGNED;
            }
            if ffi_type == clibffi::ffi_type_schar() as usize {
                return kind::SIGNED;
            }
            if ffi_type == clibffi::ffi_type_uchar() as usize {
                return kind::UNSIGNED;
            }
            if ffi_type == clibffi::ffi_type_sshort() as usize {
                return kind::SIGNED;
            }
            if ffi_type == clibffi::ffi_type_ushort() as usize {
                return kind::UNSIGNED;
            }
            if ffi_type == clibffi::ffi_type_sint() as usize {
                return kind::SIGNED;
            }
            if ffi_type == clibffi::ffi_type_uint() as usize {
                return kind::UNSIGNED;
            }
            if ffi_type == slong() || ffi_type == signed() {
                return kind::SIGNED;
            }
            if ffi_type == ulong() || ffi_type == unsigned() {
                return kind::UNSIGNED;
            }
            if ffi_type == clibffi::ffi_type_sint8() as usize
                || ffi_type == clibffi::ffi_type_sint16() as usize
                || ffi_type == clibffi::ffi_type_sint32() as usize
            {
                return kind::SIGNED;
            }
            if ffi_type == clibffi::ffi_type_uint8() as usize
                || ffi_type == clibffi::ffi_type_uint16() as usize
                || ffi_type == clibffi::ffi_type_uint32() as usize
            {
                return kind::UNSIGNED;
            }
            if ffi_type == clibffi::ffi_type_sint64() as usize
                || ffi_type == clibffi::ffi_type_uint64() as usize
            {
                return kind::LONGLONG;
            }
            if unsafe { is_struct(ffi_type) } {
                return kind::STRUCT;
            }
            kind::OTHER
        }

        /// `types.getsize(ffi_type)`.
        ///
        /// # Safety
        /// `ffi_type` must be an `ffi_type` record.
        #[majit_macros::elidable_cannot_raise]
        pub unsafe fn getsize(ffi_type: usize) -> usize {
            unsafe { (*(ffi_type as clibffi::FFI_TYPE_P)).size }
        }

        /// `types.is_struct(ffi_type)`.
        ///
        /// # Safety
        /// `ffi_type` must be an `ffi_type` record.
        #[majit_macros::elidable_cannot_raise]
        pub unsafe fn is_struct(ffi_type: usize) -> bool {
            unsafe { (*(ffi_type as clibffi::FFI_TYPE_P)).type_ == FFI_TYPE_STRUCT }
        }
    }

    /// `jit_libffi.py jit_ffi_call` — call the function `cif_description`
    /// describes, with the arguments already converted into
    /// `exchange_buffer`, and leave the result in that buffer at
    /// `exchange_result`.
    ///
    /// `cif_description` has to be a trace constant for a trace to specialize
    /// the call, which is why the caller promotes the function type before it
    /// allocates the buffer.
    ///
    /// # Safety
    /// `exchange_buffer` must be a buffer of `exchange_size(cif_description)`
    /// bytes holding `nargs` converted arguments, and `func_addr` must be the
    /// entry point the cif was prepared for.
    pub unsafe fn jit_ffi_call(cif_description: usize, func_addr: usize, exchange_buffer: usize) {
        let reskind = unsafe { types::getkind(rtype(cif_description)) };
        if reskind == kind::VOID {
            unsafe { jit_ffi_call_impl_void(cif_description, func_addr, exchange_buffer) };
        } else if reskind == kind::SIGNED {
            unsafe { do_ffi_call_sint(cif_description, func_addr, exchange_buffer) };
        } else if reskind == kind::UNSIGNED {
            unsafe { do_ffi_call_uint(cif_description, func_addr, exchange_buffer) };
        } else if reskind == kind::FLOAT {
            unsafe { do_ffi_call_float(cif_description, func_addr, exchange_buffer) };
        } else if reskind == kind::LONGLONG {
            // `L` is for longlongs, on 32bit.
            unsafe { do_ffi_call_longlong(cif_description, func_addr, exchange_buffer) };
        } else if reskind == kind::SINGLEFLOAT {
            unsafe { do_ffi_call_singlefloat(cif_description, func_addr, exchange_buffer) };
        } else {
            // The result kind is not one a trace can carry, so call
            // `jit_ffi_call_impl_any` directly and let no `libffi_call`
            // oopspec reach the JIT at all.  `call_release_gil` is not
            // generated, so there is nothing to store back either.
            unsafe { jit_ffi_call_impl_any(cif_description, func_addr, exchange_buffer) };
        }
    }

    /// `jit_libffi.py _do_ffi_call_sint`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    unsafe fn do_ffi_call_sint(cif_description: usize, func_addr: usize, exchange_buffer: usize) {
        let result = unsafe { jit_ffi_call_impl_int(cif_description, func_addr, exchange_buffer) };
        let target =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        match unsafe { types::getsize(rtype(cif_description)) } {
            1 => unsafe { (target as *mut i8).write(result as i8) },
            2 => unsafe { (target as *mut i16).write(result as i16) },
            4 => unsafe { (target as *mut i32).write(result as i32) },
            // The default case expects a full signed number.
            _ => unsafe { (target as *mut i64).write(result) },
        }
    }

    /// `jit_libffi.py _do_ffi_call_uint`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    unsafe fn do_ffi_call_uint(cif_description: usize, func_addr: usize, exchange_buffer: usize) {
        let result = unsafe { jit_ffi_call_impl_int(cif_description, func_addr, exchange_buffer) };
        let target =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        match unsafe { types::getsize(rtype(cif_description)) } {
            1 => unsafe { (target as *mut u8).write(result as u8) },
            2 => unsafe { (target as *mut u16).write(result as u16) },
            4 => unsafe { (target as *mut u32).write(result as u32) },
            // The default case expects a full unsigned number.
            _ => unsafe { (target as *mut u64).write(result as u64) },
        }
    }

    /// `jit_libffi.py _do_ffi_call_float`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    unsafe fn do_ffi_call_float(cif_description: usize, func_addr: usize, exchange_buffer: usize) {
        let result =
            unsafe { jit_ffi_call_impl_float(cif_description, func_addr, exchange_buffer) };
        let target =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        llrffi::raw_write_f64(target, result);
    }

    /// `jit_libffi.py _do_ffi_call_longlong`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    unsafe fn do_ffi_call_longlong(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) {
        let result =
            unsafe { jit_ffi_call_impl_longlong(cif_description, func_addr, exchange_buffer) };
        let target =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        unsafe { (target as *mut LONGLONG).write(result) };
    }

    /// `jit_libffi.py _do_ffi_call_singlefloat`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    unsafe fn do_ffi_call_singlefloat(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) {
        let result =
            unsafe { jit_ffi_call_impl_singlefloat(cif_description, func_addr, exchange_buffer) };
        let target =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        unsafe { (target as *mut f32).write(result as f32) };
    }

    /// `jit_libffi.py jit_ffi_call_impl_int` — reads back a complete
    /// `ffi_arg` word.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    #[majit_macros::oopspec("libffi_call(cif_description,func_addr,exchange_buffer)")]
    pub unsafe fn jit_ffi_call_impl_int(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) -> i64 {
        unsafe { jit_ffi_call_impl_any(cif_description, func_addr, exchange_buffer) };
        let resultdata =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        unsafe { (resultdata as *const ffi_arg).read() as i64 }
    }

    /// `jit_libffi.py jit_ffi_call_impl_float`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    #[majit_macros::oopspec("libffi_call(cif_description,func_addr,exchange_buffer)")]
    pub unsafe fn jit_ffi_call_impl_float(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) -> f64 {
        unsafe { jit_ffi_call_impl_any(cif_description, func_addr, exchange_buffer) };
        let resultdata =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        llrffi::raw_read_f64(resultdata)
    }

    /// `jit_libffi.py jit_ffi_call_impl_longlong`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    #[majit_macros::oopspec("libffi_call(cif_description,func_addr,exchange_buffer)")]
    pub unsafe fn jit_ffi_call_impl_longlong(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) -> LONGLONG {
        unsafe { jit_ffi_call_impl_any(cif_description, func_addr, exchange_buffer) };
        let resultdata =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        unsafe { (resultdata as *const LONGLONG).read() }
    }

    /// `jit_libffi.py jit_ffi_call_impl_singlefloat`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    #[majit_macros::oopspec("libffi_call(cif_description,func_addr,exchange_buffer)")]
    pub unsafe fn jit_ffi_call_impl_singlefloat(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) -> f64 {
        unsafe { jit_ffi_call_impl_any(cif_description, func_addr, exchange_buffer) };
        let resultdata =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) });
        unsafe { f64::from((resultdata as *const f32).read()) }
    }

    /// `jit_libffi.py jit_ffi_call_impl_void`.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    #[majit_macros::oopspec("libffi_call(cif_description,func_addr,exchange_buffer)")]
    pub unsafe fn jit_ffi_call_impl_void(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) {
        unsafe { jit_ffi_call_impl_any(cif_description, func_addr, exchange_buffer) };
    }

    /// `jit_libffi.py jit_ffi_call_impl_any` — the one that actually calls
    /// libffi.  Everything above it exists to hand the JIT a typed result.
    ///
    /// # Safety
    /// As [`jit_ffi_call`].
    pub unsafe fn jit_ffi_call_impl_any(
        cif_description: usize,
        func_addr: usize,
        exchange_buffer: usize,
    ) {
        let buffer_array = exchange_buffer as *mut *mut std::ffi::c_void;
        for i in 0..unsafe { nargs(cif_description) } {
            let data =
                llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_arg(cif_description, i) });
            unsafe { buffer_array.add(i).write(data as *mut std::ffi::c_void) };
        }
        let resultdata =
            llrffi::raw_ptradd(exchange_buffer, unsafe { exchange_result(cif_description) })
                as *mut std::ffi::c_void;
        let descr = unsafe { header(cif_description) };
        // `clibffi.py c_ffi_call` (`save_err=RFFI_ERR_ALL | RFFI_ALT_ERRNO`).
        // A trace records `CALL_RELEASE_GIL` with the same flags
        // (`pyjitpl.py direct_libffi_call`) and the backend does the swap.
        unsafe {
            clibffi::c_ffi_call(
                &raw mut descr.cif,
                func_addr as VOIDP,
                resultdata,
                buffer_array,
            );
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    /// No libffi on this target, so no foreign call can be described.
    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn exchange_size(_cif: usize) -> usize {
        0
    }

    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn exchange_result(_cif: usize) -> usize {
        0
    }

    #[majit_macros::elidable_cannot_raise]
    pub unsafe fn exchange_arg(_cif: usize, _i: usize) -> usize {
        0
    }

    pub unsafe fn jit_ffi_call(
        _cif_description: usize,
        _func_addr: usize,
        _exchange_buffer: usize,
    ) {
    }
}

pub use imp::*;

#[cfg(all(test, any(unix, windows)))]
mod tests {
    use super::*;
    use crate::clibffi::ffi_type_double;

    // `c_ffi_prep_cif` is an rffi extern; default tests use the sandbox stub.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    #[test]
    fn jit_ffi_prep_cif_void_no_args() {
        use crate::clibffi::{FFI_DEFAULT_ABI, FFI_OK, ffi_cif, ffi_type_void};
        let mut descr = CifDescription {
            cif: ffi_cif::default(),
            abi: FFI_DEFAULT_ABI,
            nargs: 0,
            rtype: ffi_type_void(),
            atypes: std::ptr::null_mut(),
            exchange_size: 0,
            exchange_result: 0,
        };
        let res = unsafe { jit_ffi_prep_cif(&raw mut descr as usize) };
        assert_eq!(res, FFI_OK as i64);
    }

    #[test]
    fn getkind_matches_types_namespace() {
        unsafe {
            assert_eq!(
                types::getkind(crate::clibffi::ffi_type_void() as usize),
                kind::VOID
            );
            assert_eq!(types::getkind(ffi_type_double() as usize), kind::FLOAT);
            assert_eq!(
                types::getkind(crate::clibffi::ffi_type_float() as usize),
                kind::SINGLEFLOAT
            );
            // On a 64-bit word, `types.signed` is the 64-bit record and is
            // asked before `sint64`, so `L` is a 32-bit-only kind.
            assert_eq!(
                types::getkind(crate::clibffi::ffi_type_sint64() as usize),
                if std::mem::size_of::<isize>() == 8 {
                    kind::SIGNED
                } else {
                    kind::LONGLONG
                }
            );
        }
    }
}
