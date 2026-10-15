//! Function pointers — PyPy: `pypy/module/_cffi_backend/ctypefunc.py`.
//!
//! A non-variadic function type owns one `CIF_DESCRIPTION` block, built when
//! the type is created and reused by every call through it
//! (`CifDescrBuilder.rawallocate`).  That block holds the `ffi_cif` itself,
//! the `ffi_type *` array libffi reads, the `ffi_type` records any by-value
//! struct argument needs, and the offsets of each argument inside the
//! exchange buffer a call fills.  A variadic function has no such block: its
//! signature is only known per call, so one is built and freed around the
//! call itself.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};

/// `ctypefunc.py set_mustfree_flag`'s values.  RPython's flags 3 to 6 name
/// the pinned and non-moving-buffer cases of an RPython string handed
/// straight to C; pyre copies such an argument instead, so only the first
/// three occur.
pub const MUSTFREE_NOTHING: u8 = 0;
/// The argument slot holds memory this call allocated and must free.
pub const MUSTFREE_FREE: u8 = 1;
/// The argument slot holds a `FILE *` the call borrowed.
pub const MUSTFREE_FILE: u8 = 2;

/// `ctypefunc.py get_mustfree_flag`.
///
/// # Safety
/// `data` must be an argument slot of an exchange buffer whose ctype is a
/// pointer, so the byte before it belongs to this argument.
#[majit_macros::dont_look_inside_cannot_raise]
pub unsafe fn get_mustfree_flag(data: usize) -> u8 {
    unsafe { (data as *const u8).sub(1).read() }
}

/// `ctypefunc.py set_mustfree_flag`.
///
/// # Safety
/// As [`get_mustfree_flag`].
pub unsafe fn set_mustfree_flag(data: *mut u8, flag: u8) {
    unsafe { data.sub(1).write(flag) }
}

/// `W_CTypeFunc._compute_extra_text`.
pub fn compute_extra_text(fargs: &[PyObjectRef], ellipsis: bool, abi: i64) -> (String, i64) {
    let mut xpos = 2i64;
    let mut out = if stdcall_abi() == Some(abi) {
        xpos += "__stdcall ".len() as i64;
        "(__stdcall *)(".to_string()
    } else {
        "(*)(".to_string()
    };
    for (i, &w_farg) in fargs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        if let Some(farg) = ctypeobj::ctype_at(w_farg) {
            out.push_str(farg.name());
        }
    }
    if ellipsis {
        if !fargs.is_empty() {
            out.push_str(", ");
        }
        out.push_str("...");
    }
    out.push(')');
    (out, xpos)
}

/// `moduledef.py has_stdcall` and `FFI_STDCALL` — a calling convention only
/// 32-bit Windows has.
#[cfg(all(windows, target_arch = "x86"))]
pub fn stdcall_abi() -> Option<i64> {
    Some(majit_rlib::clibffi::FFI_STDCALL as i64)
}

#[cfg(not(all(windows, target_arch = "x86")))]
pub fn stdcall_abi() -> Option<i64> {
    None
}

/// The argument ctypes of a function type, snapshotted before anything that
/// could allocate.  Every ctype is non-moving, so the addresses stay valid.
pub fn fargs_of(ct: &W_CType) -> Vec<PyObjectRef> {
    if ct.fargs.is_null() {
        return Vec::new();
    }
    unsafe { pyre_object::tupleobject::w_tuple_items_copy_as_vec(ct.fargs) }
}

/// `W_CTypeFunc.call` — the entry `W_CData.call` reaches.
///
/// The non-variadic arm is the one a trace follows: the function type is
/// promoted, so its argument count, cif block and result type are trace
/// constants and [`do_call`] unrolls against them.  A variadic call builds
/// its cif per call and stays opaque, as `call_varargs` does.
pub fn call(ct: &W_CType, funcaddr: usize, args_w: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // `self = jit.promote(self)`.
    let ct: &W_CType = unsafe { &*majit_metainterp::jit::promote(ct as *const W_CType) };
    if funcaddr == 0 {
        return Err(unsafe { PyError::from_exc_object(cannot_call_null(ct)) });
    }
    let nargs = fargs_len(ct.fargs);
    let got = args_w.len();
    if ct.cif_descr != 0 {
        if got != nargs {
            return Err(unsafe { PyError::from_exc_object(wrong_nargs(ct, nargs, got)) });
        }
        return do_call(ct, funcaddr, args_w);
    }
    // Variadic completion allocates; it stays inside `call_varargs`.
    call_varargs(ct, funcaddr, args_w)
}

/// `W_CTypeFunc.call` — `oefmt("cannot call null function pointer from cdata '%s'")`.
#[majit_macros::dont_look_inside]
pub(crate) fn cannot_call_null(ct: &W_CType) -> PyObjectRef {
    PyError::runtime_error(format!(
        "cannot call null function pointer from cdata '{}'",
        ct.name()
    ))
    .to_exc_object()
}

/// `W_CTypeFunc.call` — `oefmt("'%s' expects %d arguments, got %d")`.
#[majit_macros::dont_look_inside]
pub(crate) fn wrong_nargs(ct: &W_CType, expected: usize, got: usize) -> PyObjectRef {
    PyError::type_error(format!(
        "'{}' expects {} arguments, got {}",
        ct.name(),
        expected,
        got
    ))
    .to_exc_object()
}

/// `len(self.fargs)` — the declared argument count of a function type.
fn fargs_len(w_fargs: PyObjectRef) -> usize {
    if w_fargs.is_null() {
        return 0;
    }
    unsafe { pyre_object::tupleobject::w_tuple_len(w_fargs) }
}

/// `self.fargs[i]` — the declared ctype of argument `i`.
fn farg(w_fargs: PyObjectRef, i: usize) -> PyObjectRef {
    match unsafe { pyre_object::tupleobject::w_tuple_getitem(w_fargs, i as i64) } {
        Some(w_farg) => w_farg,
        None => pyre_object::PY_NULL,
    }
}

/// `W_CTypeFunc.call_varargs` — the cif depends on what was passed, so it is
/// built for this call and freed with it.
#[majit_macros::dont_look_inside]
fn call_varargs(
    ct: &W_CType,
    funcaddr: usize,
    args_w: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let fargs = fargs_of(ct);
    if args_w.len() < fargs.len() {
        return Err(PyError::type_error(format!(
            "'{}' expects at least {} arguments, got {}",
            ct.name(),
            fargs.len(),
            args_w.len()
        )));
    }
    // `args_w` is a native copy: completing the argument types can allocate
    // a promoted primitive type, and the completed tuple below is a fresh
    // allocation.  The completed types themselves are stable ctypes.
    let roots = pyre_object::gc_roots::push_roots();
    let args_base = roots.pin_roots(args_w);
    let fvarargs = complete_argtypes(&fargs, &roots, args_base, args_w.len())?;
    let cif = build_cif_descr(&fvarargs, ct.ctitem, ct.abi, Some(fargs.len()))?;
    // `new_ctypefunc_completing_argtypes` builds a fresh function type and
    // calls `_call` on it. The completed tuple is young and is not a field
    // of `ct`, so this opaque arm pins it for the conversions below.
    let fargs_slot = roots.pin_roots(&[pyre_object::tupleobject::w_tuple_new(fvarargs)]);
    let mut live_args = Vec::with_capacity(args_w.len());
    for i in 0..args_w.len() {
        live_args.push(roots.get(args_base + i));
    }
    let result = do_call_fargs(roots.get(fargs_slot), ct.ctitem, cif, funcaddr, &live_args);
    unsafe { free_cif_descr(cif) };
    result
}

/// `W_CTypeFunc.new_ctypefunc_completing_argtypes` — the declared argument
/// types followed by the promoted type of each variadic argument.
///
/// The arguments are read from the caller's slots `args_base..+nargs`: a
/// promoted primitive type is allocated on first use.
fn complete_argtypes(
    fargs: &[PyObjectRef],
    roots: &pyre_object::gc_roots::RootScope,
    args_base: usize,
    nargs: usize,
) -> Result<Vec<PyObjectRef>, PyError> {
    let mut fvarargs = Vec::with_capacity(nargs);
    fvarargs.extend_from_slice(fargs);
    for i in fargs.len()..nargs {
        let w_obj = roots.get(args_base + i);
        let Some(cdata) = W_CData::from_obj(w_obj) else {
            return Err(PyError::type_error(format!(
                "argument {} passed in the variadic part needs to be a cdata object (got {})",
                i + 1,
                pyre_interpreter::type_methods::arg_type_name(w_obj)
            )));
        };
        fvarargs.push(ctypeobj::get_vararg_type(cdata.ctype)?);
    }
    Ok(fvarargs)
}

/// `W_CTypeFunc._call` — fill the exchange buffer, call, read the result out.
///
/// Upstream reads `self.fargs[i]` off the promoted function type
/// (`ctypefunc.py` `_call`). The ctype is `allocate_stable`, so `self` does
/// not move and a collection rewrites the `fargs` field when the tuple does.
/// Re-reading the field is that load; a second root bracket around a copy of
/// the pointer was only there so the jitcode eraser could see a single pin.
///
/// `args_w` is still a native slice the collector does not rewrite, so the
/// Python arguments stay on the shadow stack across a conversion that can
/// collect. The `try`/`finally` of the original is a labelled body whose
/// outcome both arms of the closing `match` release. `@jit.unroll_safe`: the
/// argument loop runs `len(self.fargs)` times, a trace constant once the
/// function type is promoted.
#[majit_macros::unroll_safe]
fn do_call(ct: &W_CType, funcaddr: usize, args_w: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let fresult = ctypeobj::ctype_arg(ct.ctitem)?;
    let cif = ct.cif_descr;
    let n = args_w.len();
    // `keepalive_until_here(args_w)` in `ctypefunc.py` `_call` keeps every
    // argument alive across `jit_ffi_call`. `args_w` is a native slice the
    // collector does not rewrite, so one argument is pinned the same way as
    // two or more. The arms stay separate: `Option::as_ref` does not lower.
    let size = unsafe { exchange_size(cif) };
    let buffer = cdataobj::raw_malloc_varsize_char(size);
    if buffer == 0 {
        return Err(PyError::new(
            pyre_interpreter::PyErrorKind::MemoryError,
            "out of memory",
        ));
    }
    let mut mustfree_max_plus_1 = 0usize;
    let called = 'body: {
        if n > 1 {
            let args_roots = pyre_object::gc_roots::push_roots();
            let args_slot = args_roots.pin_roots(args_w);
            for i in 0..n {
                let data = cdataobj::raw_ptradd(buffer, unsafe { exchange_arg(cif, i) });
                let w_argtype = farg(ct.fargs, i);
                let argtype = match ctypeobj::ctype_arg(w_argtype) {
                    Ok(argtype) => argtype,
                    Err(e) => break 'body Err(e),
                };
                match unsafe {
                    ctypeobj::convert_argument_from_object(
                        argtype,
                        data,
                        args_roots.get(args_slot + i),
                    )
                } {
                    Ok(true) => mustfree_max_plus_1 = i + 1,
                    Ok(false) => {}
                    Err(e) => break 'body Err(e),
                }
            }
            // The pins stay live across the call: a callback can collect.
            unsafe { jit_ffi_call(cif, funcaddr, buffer) };
            let resultdata = cdataobj::raw_ptradd(buffer, unsafe { exchange_result(cif) });
            unsafe { ctypeobj::copy_and_convert_to_object(fresult, resultdata) }
        } else if n == 1 {
            let args_roots = pyre_object::gc_roots::push_roots();
            let args_slot = args_roots.base();
            let w_arg = args_w[0];
            let _ = args_roots.pin_root(w_arg);
            let data = cdataobj::raw_ptradd(buffer, unsafe { exchange_arg(cif, 0) });
            let w_argtype = farg(ct.fargs, 0);
            let argtype = match ctypeobj::ctype_arg(w_argtype) {
                Ok(argtype) => argtype,
                Err(e) => break 'body Err(e),
            };
            match unsafe {
                ctypeobj::convert_argument_from_object(argtype, data, args_roots.get(args_slot))
            } {
                Ok(true) => mustfree_max_plus_1 = 1,
                Ok(false) => {}
                Err(e) => break 'body Err(e),
            }
            unsafe { jit_ffi_call(cif, funcaddr, buffer) };
            let resultdata = cdataobj::raw_ptradd(buffer, unsafe { exchange_result(cif) });
            unsafe { ctypeobj::copy_and_convert_to_object(fresult, resultdata) }
        } else {
            unsafe { jit_ffi_call(cif, funcaddr, buffer) };
            let resultdata = cdataobj::raw_ptradd(buffer, unsafe { exchange_result(cif) });
            unsafe { ctypeobj::copy_and_convert_to_object(fresult, resultdata) }
        }
    };
    match called {
        Ok(w_res) => {
            release_arguments(ct.fargs, cif, buffer, mustfree_max_plus_1);
            Ok(w_res)
        }
        Err(e) => {
            release_arguments(ct.fargs, cif, buffer, mustfree_max_plus_1);
            Err(e)
        }
    }
}

/// Variadic `_call`: the completed `fargs` tuple is not a field of `ct`.
#[majit_macros::dont_look_inside]
fn do_call_fargs(
    w_fargs: PyObjectRef,
    w_fresult: PyObjectRef,
    cif: usize,
    funcaddr: usize,
    args_w: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let fresult = ctypeobj::ctype_arg(w_fresult)?;
    let roots = pyre_object::gc_roots::push_roots();
    let fargs_slot = roots.base();
    let _ = roots.pin_root(w_fargs);
    let args_slot = fargs_slot + 1;
    for &w_arg in args_w {
        let _ = roots.pin_root(w_arg);
    }
    let size = unsafe { exchange_size(cif) };
    let buffer = cdataobj::raw_malloc_varsize_char(size);
    if buffer == 0 {
        return Err(PyError::new(
            pyre_interpreter::PyErrorKind::MemoryError,
            "out of memory",
        ));
    }
    let mut mustfree_max_plus_1 = 0usize;
    let called = 'body: {
        for i in 0..args_w.len() {
            let data = cdataobj::raw_ptradd(buffer, unsafe { exchange_arg(cif, i) });
            let argtype = match ctypeobj::ctype_arg(farg(roots.get(fargs_slot), i)) {
                Ok(argtype) => argtype,
                Err(e) => break 'body Err(e),
            };
            match unsafe {
                ctypeobj::convert_argument_from_object(argtype, data, roots.get(args_slot + i))
            } {
                Ok(true) => mustfree_max_plus_1 = i + 1,
                Ok(false) => {}
                Err(e) => break 'body Err(e),
            }
        }
        unsafe { jit_ffi_call(cif, funcaddr, buffer) };
        let resultdata = cdataobj::raw_ptradd(buffer, unsafe { exchange_result(cif) });
        unsafe { ctypeobj::copy_and_convert_to_object(fresult, resultdata) }
    };
    match called {
        Ok(w_res) => {
            release_arguments(roots.get(fargs_slot), cif, buffer, mustfree_max_plus_1);
            Ok(w_res)
        }
        Err(e) => {
            release_arguments(roots.get(fargs_slot), cif, buffer, mustfree_max_plus_1);
            Err(e)
        }
    }
}

/// The `finally` of `W_CTypeFunc._call`: free what the converted pointer
/// arguments allocated, then the exchange buffer itself.
#[majit_macros::unroll_safe]
fn release_arguments(w_fargs: PyObjectRef, cif: usize, buffer: usize, mustfree_max_plus_1: usize) {
    for i in 0..mustfree_max_plus_1 {
        let Some(argtype) = ctypeobj::ctype_at(farg(w_fargs, i)) else {
            continue;
        };
        if argtype.kind != ctypeobj::KIND_POINTER {
            continue;
        }
        let data = cdataobj::raw_ptradd(buffer, unsafe { exchange_arg(cif, i) });
        if unsafe { get_mustfree_flag(data) } == MUSTFREE_FREE {
            let raw = cdataobj::raw_read_ptr(data);
            cdataobj::raw_free(raw);
        }
    }
    cdataobj::raw_free(buffer);
}

// ── the CIF_DESCRIPTION block ───────────────────────────────────────────

#[cfg(any(unix, windows))]
pub(crate) mod cif {
    use super::super::jit_libffi::{CifDescription, SIZE_OF_FFI_ARG, exchange_args, header};
    use super::{PyError, PyObjectRef, W_CType, ctypeobj};
    use majit_rlib::clibffi::{self, FFI_OK, FFI_TYPE_P, FFI_TYPE_STRUCT, ffi_type};
    use majit_rlib::rffi::UINT;

    /// `CifDescrBuilder` — the two-pass bump allocator that measures the block
    /// and then fills it.
    ///
    /// The argument and result types stay on the shadow stack across both
    /// passes.
    struct Builder {
        fargs_base: usize,
        nargs: usize,
        fresult_slot: usize,
        nb_bytes: usize,
        bufferp: *mut u8,
        atypes: *mut FFI_TYPE_P,
        rtype: FFI_TYPE_P,
    }

    impl Builder {
        /// `CifDescrBuilder.fb_alloc`.  Every request is a multiple of eight
        /// and the block itself comes from `malloc`, so each record inside it
        /// lands on its own alignment.
        fn alloc(&mut self, size: usize) -> *mut u8 {
            let size = size.next_multiple_of(8);
            if self.bufferp.is_null() {
                self.nb_bytes += size;
                return std::ptr::null_mut();
            }
            let result = self.bufferp;
            self.bufferp = unsafe { result.add(size) };
            result
        }

        /// `CifDescrBuilder.fb_build`.
        fn build(&mut self) -> Result<(), PyError> {
            let nargs = self.nargs;
            self.alloc(
                std::mem::size_of::<CifDescription>() + nargs * std::mem::size_of::<usize>(),
            );
            let atypes = self.alloc(nargs * std::mem::size_of::<FFI_TYPE_P>());
            self.atypes = atypes.cast::<FFI_TYPE_P>();
            self.rtype = self.fill_type(
                ctypeobj::ctype_arg(pyre_object::gc_roots::shadow_stack_get(self.fresult_slot))?,
                true,
            )?;
            for i in 0..nargs {
                let farg = ctypeobj::ctype_arg(pyre_object::gc_roots::shadow_stack_get(
                    self.fargs_base + i,
                ))?;
                let atype = self.fill_type(farg, false)?;
                if !self.atypes.is_null() {
                    unsafe { self.atypes.add(i).write(atype) };
                }
            }
            Ok(())
        }

        /// `CifDescrBuilder.fb_fill_type` and the `_get_ffi_type` each ctype
        /// class is given at the bottom of `ctypefunc.py`.
        fn fill_type(&mut self, ct: &W_CType, is_result: bool) -> Result<FFI_TYPE_P, PyError> {
            let by_size = |signed: bool| -> Option<FFI_TYPE_P> {
                let t: FFI_TYPE_P = match (ct.size, signed) {
                    (1, true) => clibffi::ffi_type_sint8(),
                    (2, true) => clibffi::ffi_type_sint16(),
                    (4, true) => clibffi::ffi_type_sint32(),
                    (8, true) => clibffi::ffi_type_sint64(),
                    (1, false) => clibffi::ffi_type_uint8(),
                    (2, false) => clibffi::ffi_type_uint16(),
                    (4, false) => clibffi::ffi_type_uint32(),
                    (8, false) => clibffi::ffi_type_uint64(),
                    _ => return None,
                };
                Some(t)
            };
            match ct.kind {
                ctypeobj::KIND_STRUCT if ct.size >= 0 => self.struct_ffi_type(ct, is_result),
                // Only for a better error message: a completed union still
                // cannot be passed by value.
                ctypeobj::KIND_UNION if ct.size >= 0 => Err(unsupported(
                    ct,
                    is_result,
                    "libffi",
                    "Unions",
                    String::new(),
                )),
                ctypeobj::KIND_PRIM_SIGNED => {
                    by_size(true).ok_or_else(|| missing_ffi_type(ct, is_result))
                }
                ctypeobj::KIND_PRIM_UNSIGNED
                | ctypeobj::KIND_PRIM_BOOL
                | ctypeobj::KIND_PRIM_CHAR
                | ctypeobj::KIND_PRIM_UNICHAR => {
                    by_size(false).ok_or_else(|| missing_ffi_type(ct, is_result))
                }
                ctypeobj::KIND_PRIM_FLOAT => match ct.size {
                    4 => Ok(clibffi::ffi_type_float()),
                    8 => Ok(clibffi::ffi_type_double()),
                    _ => Err(missing_ffi_type(ct, is_result)),
                },
                ctypeobj::KIND_PRIM_LONGDOUBLE => Ok(clibffi::ffi_type_longdouble()),
                ctypeobj::KIND_PRIM_COMPLEX => Err(PyError::not_implemented(format!(
                    "ctype '{}' (size {}) not supported as {} (the support for complex types inside libffi is mostly missing at this point, so CFFI only supports complex types as arguments or return value in API-mode functions)",
                    ct.name(),
                    ct.size,
                    place(is_result)
                ))),
                // `W_CTypePtrBase._get_ffi_type` — a pointer or a function
                // pointer.  A plain array is not one of them.
                ctypeobj::KIND_POINTER | ctypeobj::KIND_FUNC => Ok(clibffi::ffi_type_pointer()),
                ctypeobj::KIND_VOID if is_result => Ok(clibffi::ffi_type_void()),
                _ => Err(missing_ffi_type(ct, is_result)),
            }
        }

        /// `CifDescrBuilder.fb_struct_ffi_type`.
        fn struct_ffi_type(
            &mut self,
            ct: &W_CType,
            is_result: bool,
        ) -> Result<FFI_TYPE_P, PyError> {
            ct.force_lazy_struct()?;
            // A struct completed from an incomplete declaration may be laid
            // out differently from the one the C compiler saw, and the calling
            // convention depends on the fields; so may an anonymous nested
            // struct, whose origin is no longer recorded here.
            if ct.has(ctypeobj::CTypeFlags::CUSTOM_FIELD_POS) {
                return Err(unsupported(
                    ct,
                    is_result,
                    "",
                    "Such structs",
                    "It is a struct declared with \"...;\", but the C calling convention may depend on the missing fields; or, it contains anonymous struct/unions".to_string(),
                ));
            }
            if ct.has(ctypeobj::CTypeFlags::WITH_PACKED_CHANGE) {
                return Err(unsupported(
                    ct,
                    is_result,
                    "",
                    "Such structs",
                    "It is a 'packed' structure, with a different layout than expected by libffi"
                        .to_string(),
                ));
            }
            let fields = super::super::ctypestruct::fields_list_of(ct)?;
            let mut nflat = 0usize;
            for &w_field in &fields {
                nflat += flatten(ct, w_field, is_result)?.1;
            }
            let elements = self
                .alloc((nflat + 1) * std::mem::size_of::<FFI_TYPE_P>())
                .cast::<FFI_TYPE_P>();
            let mut written = 0usize;
            for &w_field in &fields {
                let (base, flat) = flatten(ct, w_field, is_result)?;
                let sub = self.fill_type(base, false)?;
                if !elements.is_null() {
                    for _ in 0..flat {
                        unsafe { elements.add(written).write(sub) };
                        written += 1;
                    }
                }
            }
            if !elements.is_null() {
                unsafe { elements.add(written).write(std::ptr::null_mut()) };
            }
            let ffistruct = self
                .alloc(std::mem::size_of::<ffi_type>())
                .cast::<ffi_type>();
            if !ffistruct.is_null() {
                unsafe {
                    ffistruct.write(ffi_type {
                        size: ct.size as usize,
                        alignment: ct.alignof()? as u16,
                        type_: FFI_TYPE_STRUCT,
                        elements,
                    })
                };
            }
            Ok(ffistruct)
        }
    }

    /// One struct field expanded to the scalar it repeats and how many times.
    fn flatten(
        ct: &W_CType,
        w_field: PyObjectRef,
        is_result: bool,
    ) -> Result<(&'static mut W_CType, usize), PyError> {
        let field = super::super::ctypestruct::W_CField::from_obj(w_field)
            .ok_or_else(|| PyError::system_error("struct field list holds a non-field"))?;
        if field.is_bitfield() {
            return Err(unsupported(
                ct,
                is_result,
                "",
                "Such structs",
                "It is a struct with bit fields, which libffi does not support".to_string(),
            ));
        }
        let mut flat: i64 = 1;
        let mut item = ctypeobj::ctype_arg(field.ctype)?;
        while item.kind == ctypeobj::KIND_ARRAY {
            flat *= item.length;
            item = ctypeobj::ctype_arg(item.ctitem)?;
        }
        if flat <= 0 {
            return Err(unsupported(
                ct,
                is_result,
                "",
                "Such structs",
                "It is a struct with a zero-length array, which libffi does not support"
                    .to_string(),
            ));
        }
        Ok((item, flat as usize))
    }

    fn place(is_result: bool) -> &'static str {
        if is_result {
            "return value"
        } else {
            "argument"
        }
    }

    /// `ctypefunc.py _missing_ffi_type` and `_notimplemented_ffi_type`.
    fn missing_ffi_type(ct: &W_CType, is_result: bool) -> PyError {
        if ct.size < 0 {
            return PyError::type_error(format!("ctype '{}' has incomplete type", ct.name()));
        }
        PyError::not_implemented(format!(
            "ctype '{}' (size {}) not supported as {}",
            ct.name(),
            ct.size,
            place(is_result)
        ))
    }

    /// `CifDescrBuilder.fb_unsupported` and `fb_union_ffi_type`, which share
    /// `_SUPPORTED_IN_API_MODE`'s tail.
    fn unsupported(
        ct: &W_CType,
        is_result: bool,
        by: &str,
        subject: &str,
        detail: String,
    ) -> PyError {
        let place = place(is_result);
        let head = if by.is_empty() {
            format!(
                "ctype '{}' not supported as {place}.  {detail}.  ",
                ct.name()
            )
        } else {
            format!("ctype '{}' not supported as {place} by {by}.  ", ct.name())
        };
        PyError::not_implemented(format!(
            "{head}{subject} are only supported as {place} if the function is 'API mode' and non-variadic (i.e. declared inside ffibuilder.cdef()+ffibuilder.set_source() and not taking a final '...' argument)"
        ))
    }

    fn align_arg(n: usize) -> usize {
        (n + 7) & !7
    }

    fn align_to(n: usize, t: *mut ffi_type) -> usize {
        let a = unsafe { (*t).alignment } as usize - 1;
        (n + a) & !a
    }

    /// `CifDescrBuilder.rawallocate` — measure, allocate, fill, prepare.
    /// `CifDescrBuilder.rawallocate`.  `nfixedargs` is the count of declared
    /// arguments a variadic call passes before its `...`, and `None` says the
    /// function is not variadic.
    pub fn build_cif_descr(
        fargs: &[PyObjectRef],
        w_fresult: PyObjectRef,
        abi: i64,
        nfixedargs: Option<usize>,
    ) -> Result<usize, PyError> {
        let nargs = fargs.len();
        let roots = pyre_object::gc_roots::push_roots();
        let fargs_base = roots.pin_roots(fargs);
        let fresult_slot = roots.pin_roots(&[w_fresult]);
        let mut builder = Builder {
            fargs_base,
            nargs,
            fresult_slot,
            nb_bytes: 0,
            bufferp: std::ptr::null_mut(),
            atypes: std::ptr::null_mut(),
            rtype: std::ptr::null_mut(),
        };
        builder.build()?;
        let rawmem = super::cdataobj::raw_alloc(builder.nb_bytes as i64, true)?;
        builder.bufferp = rawmem as *mut u8;
        builder.nb_bytes = 0;
        if let Err(e) = builder.build() {
            unsafe { libc::free(rawmem as *mut libc::c_void) };
            return Err(e);
        }

        // `CifDescrBuilder.fb_build_exchange`.
        let mut offset = std::mem::size_of::<*mut u8>() * nargs;
        offset = align_arg(align_to(offset, builder.rtype));
        let exchange_result = offset;
        offset += unsafe { (*builder.rtype).size }.max(SIZE_OF_FFI_ARG);
        for i in 0..nargs {
            // Room for the must-free flag the pointer conversion writes just
            // before the slot.
            if ctypeobj::ctype_arg(roots.get(fargs_base + i))?.kind == ctypeobj::KIND_POINTER {
                offset += 1;
            }
            let atype = unsafe { builder.atypes.add(i).read() };
            offset = align_arg(align_to(offset, atype));
            unsafe { exchange_args(rawmem as usize).add(i).write(offset) };
            offset += unsafe { (*atype).size };
        }

        // `CifDescrBuilder.fb_extra_fields`, then `jit_ffi_prep_cif`.
        let descr = unsafe { header(rawmem as usize) };
        descr.abi = abi as clibffi::FFI_ABI;
        descr.nargs = nargs;
        descr.rtype = builder.rtype;
        descr.atypes = builder.atypes;
        descr.exchange_result = exchange_result;
        descr.exchange_size = align_arg(offset);
        let (abi, rtype, atypes) = (descr.abi, descr.rtype, descr.atypes);
        // A target whose variadic arguments do not travel the way its fixed
        // ones do -- aarch64 Darwin passes them on the stack -- needs the
        // fixed count, which is what `clibffi.py AbstractFuncPtr.__init__`
        // hands `c_ffi_prep_cif_var` whenever `variadic_args > 0`.
        let cif = &raw mut descr.cif;
        let res = match nfixedargs {
            Some(nfixed) => unsafe {
                clibffi::c_ffi_prep_cif_var(cif, abi, nfixed as UINT, nargs as UINT, rtype, atypes)
            },
            None => unsafe { clibffi::c_ffi_prep_cif(cif, abi, nargs as UINT, rtype, atypes) },
        };
        if res != FFI_OK {
            unsafe { libc::free(rawmem as *mut libc::c_void) };
            return Err(PyError::system_error(
                "libffi failed to build this function type",
            ));
        }
        Ok(rawmem as usize)
    }

    /// `W_CTypeFunc.__del__` — the block is one allocation.
    ///
    /// # Safety
    /// `cif` must come from [`build_cif_descr`] and not be in use.
    pub unsafe fn free_cif_descr(cif: usize) {
        unsafe { libc::free(cif as *mut libc::c_void) };
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) mod cif {
    use super::{PyError, PyObjectRef};

    /// No libffi on this target, so no foreign call can be described.
    pub fn build_cif_descr(
        _fargs: &[PyObjectRef],
        _w_fresult: PyObjectRef,
        _abi: i64,
        _nfixedargs: Option<usize>,
    ) -> Result<usize, PyError> {
        Err(PyError::not_implemented(
            "this platform has no libffi, so a C function cannot be called",
        ))
    }

    pub unsafe fn free_cif_descr(_cif: usize) {}
}

use super::jit_libffi::{exchange_arg, exchange_result, exchange_size, jit_ffi_call};
pub use cif::{build_cif_descr, free_cif_descr};
