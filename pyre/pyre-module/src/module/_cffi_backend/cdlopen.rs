//! ABI-mode library loading — PyPy: `pypy/module/_cffi_backend/cdlopen.py`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::{ffi_obj, lib_obj, misc, parse_c_type};

pub const VERSION_MIN: i64 = 0x2601;
pub const VERSION_MAX: i64 = 0x28ff;

pub struct StringDecoder<'a> {
    w_ffi: PyObjectRef,
    string: &'a [u8],
    pos: usize,
}

impl<'a> StringDecoder<'a> {
    pub fn new(w_ffi: PyObjectRef, string: &'a [u8]) -> Self {
        Self {
            w_ffi,
            string,
            pos: 0,
        }
    }

    pub fn next_4bytes(&mut self) -> Result<i32, PyError> {
        let bytes = self
            .string
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| PyError::value_error("truncated CFFI packed string"))?;
        self.pos += 4;
        Ok(i32::from_be_bytes(
            bytes.try_into().expect("four-byte slice"),
        ))
    }

    pub fn next_opcode(&mut self) -> Result<parse_c_type::OpcodeT, PyError> {
        Ok(self.next_4bytes()? as isize as parse_c_type::OpcodeT)
    }

    pub fn next_name(&mut self) -> Result<*const core::ffi::c_char, PyError> {
        let start = self.pos;
        // A name with no terminator leaves the cursor one past the end, so the
        // next read starts outside the buffer; `find` answers that with -1
        // upstream, which is the same empty name this reports.
        let start = start.min(self.string.len());
        let end = self.string[start..]
            .iter()
            .position(|&byte| byte == 0)
            .map_or(self.string.len(), |offset| start + offset);
        self.pos = end.saturating_add(1);
        let name = &self.string[start..end];
        let ptr = ffi_obj::allocate_free_mem(self.w_ffi, name.len() + 1)?;
        unsafe { core::ptr::copy_nonoverlapping(name.as_ptr(), ptr, name.len()) };
        Ok(ptr.cast())
    }
}

/// `space.bytes_w` — every packed descriptor is read as raw bytes, so one
/// that is not `bytes` has to be refused before the decoder sees it.
fn packed_bytes(w_obj: PyObjectRef) -> Result<&'static [u8], PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes(w_obj) } {
        return Err(PyError::type_error(format!(
            "expected bytes, got {} object",
            pyre_interpreter::type_methods::arg_type_name(w_obj)
        )));
    }
    Ok(unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) })
}

fn allocate_array<T>(w_ffi: PyObjectRef, nitems: usize) -> Result<*mut T, PyError> {
    let nbytes = nitems
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| {
            PyError::new(
                pyre_interpreter::PyErrorKind::MemoryError,
                "FFI array is too large",
            )
        })?;
    ffi_obj::allocate_free_mem(w_ffi, nbytes).map(|ptr| ptr.cast())
}

fn is_none_or_null(value: PyObjectRef) -> bool {
    value.is_null() || unsafe { pyre_object::pyobject::is_none(value) }
}

fn integer_value(w_integer: PyObjectRef) -> Result<(u64, i32), PyError> {
    unsafe {
        if pyre_object::pyobject::is_bool(w_integer) {
            let value = u64::from(pyre_object::boolobject::w_bool_get_value(w_integer));
            return Ok((value, i32::from(value == 0)));
        }
        if pyre_object::pyobject::is_int(w_integer) {
            let value = pyre_object::intobject::w_int_get_value(w_integer);
            return Ok((value as u64, i32::from(value <= 0)));
        }
        if pyre_object::pyobject::is_long(w_integer) {
            let value = pyre_object::longobject::w_long_get_value(w_integer);
            return Ok((value.ulonglongmask(), i32::from(value.get_sign() <= 0)));
        }
    }
    Err(PyError::type_error(format!(
        "expected an integer, got '{}'",
        pyre_interpreter::type_methods::arg_type_name(w_integer)
    )))
}

/// `ffiobj_init`.
#[allow(clippy::too_many_arguments)]
pub fn ffiobj_init(
    w_ffi: PyObjectRef,
    module_name: &str,
    version: i64,
    types: &[u8],
    w_globals: PyObjectRef,
    w_struct_unions: PyObjectRef,
    w_enums: PyObjectRef,
    w_typenames: PyObjectRef,
    w_includes: PyObjectRef,
) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    for value in [
        w_ffi,
        w_globals,
        w_struct_unions,
        w_enums,
        w_typenames,
        w_includes,
    ] {
        let _ = roots.pin_root(value);
    }
    if version == -1 && types.is_empty() {
        return Ok(());
    }
    if !(VERSION_MIN..=VERSION_MAX).contains(&version) {
        return Err(PyError::new(
            pyre_interpreter::PyErrorKind::ImportError,
            format!(
                "cffi out-of-line Python module '{module_name}' has unknown version {version:#x}"
            ),
        ));
    }

    if !types.is_empty() {
        if !types.len().is_multiple_of(4) {
            return Err(PyError::value_error(
                "CFFI packed types string is not a multiple of four bytes",
            ));
        }
        let n = types.len() / 4;
        let ntypes = allocate_array::<parse_c_type::OpcodeT>(roots.get(ffi_slot), n)?;
        let mut decoder = StringDecoder::new(roots.get(ffi_slot), types);
        for i in 0..n {
            unsafe { ntypes.add(i).write(decoder.next_opcode()?) };
        }
        let ffi = ffi_obj::ffi_arg(roots.get(ffi_slot))?;
        unsafe { (*ffi.ctxobj).ctx.types = ntypes };
        unsafe { (*ffi.ctxobj).ctx.num_types = n as core::ffi::c_int };
        let cached =
            pyre_object::w_list_new((0..n).map(|_| pyre_object::w_none()).collect::<Vec<_>>());
        ffi_obj::ffi_arg(roots.get(ffi_slot))?.cached_types = cached;
        pyre_object::gc_hook::try_gc_write_barrier_managed(
            roots.get(ffi_slot) as pyre_object::gc_hook::GCREF
        );
    }

    if !is_none_or_null(roots.get(ffi_slot + 1)) {
        let globals = pyre_interpreter::baseobjspace::fixedview(roots.get(ffi_slot + 1), -1)?;
        if globals.len() % 2 != 0 {
            return Err(PyError::value_error(
                "CFFI packed globals must contain string/value pairs",
            ));
        }
        let base = pyre_object::gc_roots::shadow_stack_len();
        for &item in &globals {
            let _ = roots.pin_root(item);
        }
        let n = globals.len() / 2;
        #[repr(C)]
        struct CdlIntConst {
            value: u64,
            neg: core::ffi::c_int,
        }
        let globals_bytes = n * core::mem::size_of::<parse_c_type::GlobalS>();
        let intconst_bytes = n * core::mem::size_of::<CdlIntConst>();
        let block =
            ffi_obj::allocate_free_mem(roots.get(ffi_slot), globals_bytes + intconst_bytes)?;
        let nglobs = block.cast::<parse_c_type::GlobalS>();
        let nintconsts = unsafe { block.add(globals_bytes) }.cast::<CdlIntConst>();
        for i in 0..n {
            let packed = packed_bytes(roots.get(base + i * 2))?;
            let mut decoder = StringDecoder::new(roots.get(ffi_slot), packed);
            let target = unsafe { &mut *nglobs.add(i) };
            target.type_op = decoder.next_opcode()?;
            target.name = decoder.next_name()?;
            let op = parse_c_type::getop(target.type_op);
            if matches!(op, parse_c_type::OP_CONSTANT_INT | parse_c_type::OP_ENUM) {
                unsafe { parse_c_type::pypy_set_cdl_realize_global_int(target) };
                let (value, neg) = integer_value(roots.get(base + i * 2 + 1))?;
                unsafe { nintconsts.add(i).write(CdlIntConst { value, neg }) };
            }
        }
        let ffi = ffi_obj::ffi_arg(roots.get(ffi_slot))?;
        unsafe {
            (*ffi.ctxobj).ctx.globals = nglobs;
            (*ffi.ctxobj).ctx.num_globals = n as core::ffi::c_int;
        }
    }

    if !is_none_or_null(roots.get(ffi_slot + 2)) {
        let struct_unions = pyre_interpreter::baseobjspace::fixedview(roots.get(ffi_slot + 2), -1)?;
        let base = pyre_object::gc_roots::shadow_stack_len();
        for &item in &struct_unions {
            let _ = roots.pin_root(item);
        }
        let n = struct_unions.len();
        let mut nftot = 0usize;
        for i in 0..n {
            nftot += pyre_interpreter::baseobjspace::fixedview(roots.get(base + i), -1)?
                .len()
                .saturating_sub(1);
        }
        let nstructs = allocate_array::<parse_c_type::StructUnionS>(roots.get(ffi_slot), n)?;
        let nfields = allocate_array::<parse_c_type::FieldS>(roots.get(ffi_slot), nftot)?;
        let mut nf = 0usize;
        for i in 0..n {
            let desc = pyre_interpreter::baseobjspace::fixedview(roots.get(base + i), -1)?;
            let desc_base = pyre_object::gc_roots::shadow_stack_len();
            for &item in &desc {
                let _ = roots.pin_root(item);
            }
            let nf1 = desc.len().saturating_sub(1);
            let packed = packed_bytes(roots.get(desc_base))?;
            let mut decoder = StringDecoder::new(roots.get(ffi_slot), packed);
            let target = unsafe { &mut *nstructs.add(i) };
            target.type_index = decoder.next_4bytes()?;
            target.flags = decoder.next_4bytes()?;
            target.name = decoder.next_name()?;
            if target.flags
                & (parse_c_type::CffiTypeFlags::OPAQUE | parse_c_type::CffiTypeFlags::EXTERNAL)
                    .bits()
                != 0
            {
                target.size = usize::MAX;
                target.alignment = -1;
                target.first_field_index = -1;
                target.num_fields = 0;
                assert_eq!(nf1, 0);
            } else {
                target.size = (-2isize) as usize;
                target.alignment = -2;
                target.first_field_index = nf as core::ffi::c_int;
                target.num_fields = nf1 as core::ffi::c_int;
            }
            for j in 0..nf1 {
                let packed = packed_bytes(roots.get(desc_base + j + 1))?;
                let mut decoder = StringDecoder::new(roots.get(ffi_slot), packed);
                let field = unsafe { &mut *nfields.add(nf) };
                field.field_type_op = decoder.next_opcode()?;
                field.field_offset = usize::MAX;
                field.field_size =
                    if parse_c_type::getop(field.field_type_op) != parse_c_type::OP_NOOP {
                        decoder.next_4bytes()? as isize as usize
                    } else {
                        usize::MAX
                    };
                field.name = decoder.next_name()?;
                nf += 1;
            }
        }
        assert_eq!(nf, nftot);
        let ffi = ffi_obj::ffi_arg(roots.get(ffi_slot))?;
        unsafe {
            (*ffi.ctxobj).ctx.struct_unions = nstructs;
            (*ffi.ctxobj).ctx.fields = nfields;
            (*ffi.ctxobj).ctx.num_struct_unions = n as core::ffi::c_int;
        }
    }

    if !is_none_or_null(roots.get(ffi_slot + 3)) {
        let enums = pyre_interpreter::baseobjspace::fixedview(roots.get(ffi_slot + 3), -1)?;
        let base = pyre_object::gc_roots::shadow_stack_len();
        for &item in &enums {
            let _ = roots.pin_root(item);
        }
        let n = enums.len();
        let nenums = allocate_array::<parse_c_type::EnumS>(roots.get(ffi_slot), n)?;
        for i in 0..n {
            let packed = packed_bytes(roots.get(base + i))?;
            let mut decoder = StringDecoder::new(roots.get(ffi_slot), packed);
            let target = unsafe { &mut *nenums.add(i) };
            target.type_index = decoder.next_4bytes()?;
            target.type_prim = decoder.next_4bytes()?;
            target.name = decoder.next_name()?;
            target.enumerators = decoder.next_name()?;
        }
        let ffi = ffi_obj::ffi_arg(roots.get(ffi_slot))?;
        unsafe {
            (*ffi.ctxobj).ctx.enums = nenums;
            (*ffi.ctxobj).ctx.num_enums = n as core::ffi::c_int;
        }
    }

    if !is_none_or_null(roots.get(ffi_slot + 4)) {
        let typenames = pyre_interpreter::baseobjspace::fixedview(roots.get(ffi_slot + 4), -1)?;
        let base = pyre_object::gc_roots::shadow_stack_len();
        for &item in &typenames {
            let _ = roots.pin_root(item);
        }
        let n = typenames.len();
        let ntypenames = allocate_array::<parse_c_type::TypenameS>(roots.get(ffi_slot), n)?;
        for i in 0..n {
            let packed = packed_bytes(roots.get(base + i))?;
            let mut decoder = StringDecoder::new(roots.get(ffi_slot), packed);
            let target = unsafe { &mut *ntypenames.add(i) };
            target.type_index = decoder.next_4bytes()?;
            target.name = decoder.next_name()?;
        }
        let ffi = ffi_obj::ffi_arg(roots.get(ffi_slot))?;
        unsafe {
            (*ffi.ctxobj).ctx.typenames = ntypenames;
            (*ffi.ctxobj).ctx.num_typenames = n as core::ffi::c_int;
        }
    }

    if !is_none_or_null(roots.get(ffi_slot + 5)) {
        let includes = pyre_interpreter::baseobjspace::fixedview(roots.get(ffi_slot + 5), -1)?;
        let base = pyre_object::gc_roots::shadow_stack_len();
        for &item in &includes {
            let _ = roots.pin_root(item);
        }
        let list_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(ffi_obj::ffi_arg(roots.get(ffi_slot))?.included_ffis_libs);
        for i in 0..includes.len() {
            let _ = ffi_obj::ffi_arg(roots.get(base + i))?;
            let pair = pyre_object::w_tuple_new(vec![roots.get(base + i), pyre_object::w_none()]);
            unsafe { pyre_object::listobject::w_list_append(roots.get(list_slot), pair) };
        }
    }
    Ok(())
}

/// `W_DlOpenLibObject.__init__`.
pub fn dlopen(
    w_ffi: PyObjectRef,
    w_filename: PyObjectRef,
    flags: i64,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let filename_slot = ffi_slot + 1;
    let _ = roots.pin_root(w_filename);
    let (fname, handle, autoclose) = misc::dlopen_w(roots.get(filename_slot), flags)?;
    lib_obj::new_lib(
        roots.get(ffi_slot),
        &fname,
        lib_obj::FLAVOR_DLOPEN,
        handle,
        autoclose,
    )
}
