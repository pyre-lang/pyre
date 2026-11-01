//! Bindings to the C declaration parser — PyPy:
//! `pypy/module/_cffi_backend/parse_c_type.py`, kept with its C parser outside
//! the interpreter's Charon/LLBC extraction.
//!
//! The parser itself is `cffi_backend/src/parse_c_type.c`, taken verbatim from
//! the same place PyPy takes it (`pypy/module/_cffi_backend/src/`) and compiled
//! from `PARSE_C_TYPE_ECI` (`parse_c_type.py` `eci`).  It is the one piece of
//! `_cffi_backend` that is C on both interpreters: the opcode stream it writes
//! is the format a compiled cffi extension module embeds, so re-spelling the
//! grammar in another language would be a second implementation of a wire
//! format rather than a port.
//!
//! Every declaration here mirrors `src/parse_c_type.h`.  The layouts are
//! `#[repr(C)]` because the same structs are read straight out of a loaded
//! extension module's static data.

#![allow(dead_code)]

use core::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

/// `_cffi_opcode_t` — a tagged word: the low byte is the opcode, the rest is
/// its argument.
pub type OpcodeT = *mut c_void;

/// `_CFFI_OP(opcode, arg)`.
pub const fn op(opcode: u8, arg: isize) -> OpcodeT {
    ((opcode as usize) | ((arg as usize) << 8)) as OpcodeT
}

/// `_CFFI_GETOP(cffi_opcode)`.
pub fn getop(opcode: OpcodeT) -> u8 {
    opcode as usize as u8
}

/// `_CFFI_GETARG(cffi_opcode)`.
pub fn getarg(opcode: OpcodeT) -> isize {
    (opcode as isize) >> 8
}

pub const OP_PRIMITIVE: u8 = 1;
pub const OP_POINTER: u8 = 3;
pub const OP_ARRAY: u8 = 5;
pub const OP_OPEN_ARRAY: u8 = 7;
pub const OP_STRUCT_UNION: u8 = 9;
pub const OP_ENUM: u8 = 11;
pub const OP_FUNCTION: u8 = 13;
pub const OP_FUNCTION_END: u8 = 15;
pub const OP_NOOP: u8 = 17;
pub const OP_BITFIELD: u8 = 19;
pub const OP_TYPENAME: u8 = 21;
pub const OP_CPYTHON_BLTN_V: u8 = 23;
pub const OP_CPYTHON_BLTN_N: u8 = 25;
pub const OP_CPYTHON_BLTN_O: u8 = 27;
pub const OP_CONSTANT: u8 = 29;
pub const OP_CONSTANT_INT: u8 = 31;
pub const OP_GLOBAL_VAR: u8 = 33;
pub const OP_DLOPEN_FUNC: u8 = 35;
pub const OP_DLOPEN_CONST: u8 = 37;
pub const OP_GLOBAL_VAR_F: u8 = 39;
pub const OP_EXTERN_PYTHON: u8 = 41;

pub const PRIM_VOID: usize = 0;
pub const PRIM_BOOL: usize = 1;
pub const PRIM_CHAR: usize = 2;
pub const PRIM_SCHAR: usize = 3;
pub const PRIM_UCHAR: usize = 4;
pub const PRIM_SHORT: usize = 5;
pub const PRIM_USHORT: usize = 6;
pub const PRIM_INT: usize = 7;
pub const PRIM_UINT: usize = 8;
pub const PRIM_LONG: usize = 9;
pub const PRIM_ULONG: usize = 10;
pub const PRIM_LONGLONG: usize = 11;
pub const PRIM_ULONGLONG: usize = 12;
pub const PRIM_FLOAT: usize = 13;
pub const PRIM_DOUBLE: usize = 14;
pub const PRIM_LONGDOUBLE: usize = 15;
pub const PRIM_WCHAR: usize = 16;
pub const PRIM_INT8: usize = 17;
pub const PRIM_UINT8: usize = 18;
pub const PRIM_INT16: usize = 19;
pub const PRIM_UINT16: usize = 20;
pub const PRIM_INT32: usize = 21;
pub const PRIM_UINT32: usize = 22;
pub const PRIM_INT64: usize = 23;
pub const PRIM_UINT64: usize = 24;
pub const PRIM_INTPTR: usize = 25;
pub const PRIM_UINTPTR: usize = 26;
pub const PRIM_PTRDIFF: usize = 27;
pub const PRIM_SIZE: usize = 28;
pub const PRIM_SSIZE: usize = 29;
pub const PRIM_INT_LEAST8: usize = 30;
pub const PRIM_UINT_LEAST8: usize = 31;
pub const PRIM_INT_LEAST16: usize = 32;
pub const PRIM_UINT_LEAST16: usize = 33;
pub const PRIM_INT_LEAST32: usize = 34;
pub const PRIM_UINT_LEAST32: usize = 35;
pub const PRIM_INT_LEAST64: usize = 36;
pub const PRIM_UINT_LEAST64: usize = 37;
pub const PRIM_INT_FAST8: usize = 38;
pub const PRIM_UINT_FAST8: usize = 39;
pub const PRIM_INT_FAST16: usize = 40;
pub const PRIM_UINT_FAST16: usize = 41;
pub const PRIM_INT_FAST32: usize = 42;
pub const PRIM_UINT_FAST32: usize = 43;
pub const PRIM_INT_FAST64: usize = 44;
pub const PRIM_UINT_FAST64: usize = 45;
pub const PRIM_INTMAX: usize = 46;
pub const PRIM_UINTMAX: usize = 47;
pub const PRIM_FLOATCOMPLEX: usize = 48;
pub const PRIM_DOUBLECOMPLEX: usize = 49;
pub const PRIM_CHAR16: usize = 50;
pub const PRIM_CHAR32: usize = 51;
pub const NUM_PRIM: usize = 52;

pub const UNKNOWN_PRIM: isize = -1;
pub const UNKNOWN_FLOAT_PRIM: isize = -2;
pub const UNKNOWN_LONG_DOUBLE: isize = -3;

bitflags::bitflags! {
    /// `_cffi_backend` `F_*` bits on a realized type.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CffiTypeFlags: c_int {
        const UNION = 0x01;
        const CHECK_FIELDS = 0x02;
        const PACKED = 0x04;
        const EXTERNAL = 0x08;
        const OPAQUE = 0x10;
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlobalS {
    pub name: *const c_char,
    pub address: *mut c_void,
    pub type_op: OpcodeT,
    pub size_or_direct_fn: *mut c_void,
}

#[repr(C)]
pub struct GetConstS {
    pub value: u64,
    pub ctx: *const TypeContextS,
    pub gindex: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StructUnionS {
    pub name: *const c_char,
    pub type_index: c_int,
    pub flags: c_int,
    pub size: usize,
    pub alignment: c_int,
    pub first_field_index: c_int,
    pub num_fields: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FieldS {
    pub name: *const c_char,
    pub field_offset: usize,
    pub field_size: usize,
    pub field_type_op: OpcodeT,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EnumS {
    pub name: *const c_char,
    pub type_index: c_int,
    pub type_prim: c_int,
    pub enumerators: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TypenameS {
    pub name: *const c_char,
    pub type_index: c_int,
}

#[repr(C)]
pub struct TypeContextS {
    pub types: *mut OpcodeT,
    pub globals: *const GlobalS,
    pub fields: *const FieldS,
    pub struct_unions: *const StructUnionS,
    pub enums: *const EnumS,
    pub typenames: *const TypenameS,
    pub num_globals: c_int,
    pub num_struct_unions: c_int,
    pub num_enums: c_int,
    pub num_typenames: c_int,
    pub includes: *const *const c_char,
    pub num_types: c_int,
    pub flags: c_int,
}

#[repr(C)]
pub struct ParseInfoS {
    pub ctx: *const TypeContextS,
    pub output: *mut OpcodeT,
    pub output_size: core::ffi::c_uint,
    pub error_location: usize,
    pub error_message: *const c_char,
}

/// `parse_c_type.py CTXOBJ` — the copied declaration context and the parser
/// control record that points back to it.
#[repr(C)]
pub struct CtxObj {
    pub ctx: TypeContextS,
    pub info: ParseInfoS,
}

#[repr(C)]
pub struct ExternPyS {
    pub name: *const c_char,
    pub size_of_result: usize,
    pub reserved1: *mut c_void,
    pub reserved2: *mut c_void,
}

// `parse_c_type.py eci`: `includes`/`pre_include_bits` matter for a generated
// header; ll2ctypes does not compile them for a `separate_module_files` ECI.
// `cdir` is added by majit-eci. The parser reads `info->ctx` unconditionally —
// `search_in_typenames` asks for `ctx->num_typenames` before it can conclude
// there is nothing to search — so a caller hands it a zeroed context, never a
// null one.  That is what `parse_c_type.py allocate_ctxobj` builds when it has
// no source context.
majit_rlib::rffi::external_compilation_info! {
    const PARSE_C_TYPE_ECI = {
        pre_include_bits: ["#define _CFFI_INTERNAL"],
        includes: ["parse_c_type.h"],
        include_dirs: ["src/cffi_backend/src"],
        separate_module_files: ["src/cffi_backend/src/parse_c_type.c"],
    };
}

majit_rlib::rffi::llexternal!(
    pub pypy_parse_c_type = "pypy_parse_c_type",
    [*mut ParseInfoS, *const c_char],
    c_int,
    compilation_info = PARSE_C_TYPE_ECI,
    _nowrapper = true
);
majit_rlib::rffi::llexternal!(
    pub pypy_search_in_globals = "pypy_search_in_globals",
    [*const TypeContextS, *const c_char, usize],
    c_int,
    compilation_info = PARSE_C_TYPE_ECI,
    _nowrapper = true
);
majit_rlib::rffi::llexternal!(
    pub pypy_search_in_struct_unions = "pypy_search_in_struct_unions",
    [*const TypeContextS, *const c_char, usize],
    c_int,
    compilation_info = PARSE_C_TYPE_ECI,
    _nowrapper = true
);
majit_rlib::rffi::llexternal!(
    pub pypy_set_cdl_realize_global_int = "pypy_set_cdl_realize_global_int",
    [*mut GlobalS],
    (),
    compilation_info = PARSE_C_TYPE_ECI,
    _nowrapper = true
);
majit_rlib::rffi::llexternal!(
    pub pypy_enum_common_types = "pypy_enum_common_types",
    [c_int],
    *mut c_char,
    compilation_info = PARSE_C_TYPE_ECI,
    _nowrapper = true
);

/// `FFI_COMPLEXITY_OUTPUT` — the shared scratch the parser writes its opcode
/// stream into.  One buffer for the whole interpreter, as in
/// `parse_c_type.py:internal_output`; a parse is never re-entered because the
/// opcodes are copied out before anything else can run.
pub const FFI_COMPLEXITY_OUTPUT: usize = 1200;

/// `parse_c_type.py internal_output`, shared by all ABI-mode parses.
fn internal_output() -> *mut OpcodeT {
    static OUTPUT: OnceLock<usize> = OnceLock::new();
    *OUTPUT.get_or_init(|| {
        let bytes = FFI_COMPLEXITY_OUTPUT * core::mem::size_of::<OpcodeT>();
        let output = unsafe { libc::calloc(1, bytes) }.cast::<OpcodeT>();
        assert!(!output.is_null(), "failed to allocate CFFI parser output");
        output as usize
    }) as *mut OpcodeT
}

/// `parse_c_type.py allocate_ctxobj`.
pub fn allocate_ctxobj(src_ctx: *const TypeContextS) -> *mut CtxObj {
    let mut ctx: TypeContextS = unsafe { core::mem::zeroed() };
    if !src_ctx.is_null() {
        unsafe { core::ptr::copy_nonoverlapping(src_ctx, &mut ctx, 1) };
    }
    let mut obj = Box::new(CtxObj {
        ctx,
        info: unsafe { core::mem::zeroed() },
    });
    obj.info.ctx = &obj.ctx;
    obj.info.output = internal_output();
    obj.info.output_size = FFI_COMPLEXITY_OUTPUT as core::ffi::c_uint;
    Box::into_raw(obj)
}

/// `parse_c_type.py free_ctxobj`.
///
/// # Safety
/// `ctxobj` must be a live allocation returned by [`allocate_ctxobj`].
pub unsafe fn free_ctxobj(ctxobj: *mut CtxObj) {
    if !ctxobj.is_null() {
        drop(unsafe { Box::from_raw(ctxobj) });
    }
}

/// `parse_c_type.py get_num_types`.
pub unsafe fn get_num_types(src_ctx: *const TypeContextS) -> usize {
    unsafe { (*src_ctx).num_types.max(0) as usize }
}

/// `parse_c_type.py parse_c_type`.
pub fn parse_type(info: *mut ParseInfoS, input: &core::ffi::CStr) -> isize {
    unsafe { pypy_parse_c_type(info, input.as_ptr()) as isize }
}

/// `parse_c_type.py search_in_globals`.
pub fn search_in_globals(ctx: *const TypeContextS, name: &str) -> isize {
    unsafe { pypy_search_in_globals(ctx, name.as_ptr().cast(), name.len()) as isize }
}

/// `parse_c_type.py search_in_struct_unions`.
pub fn search_in_struct_unions(ctx: *const TypeContextS, name: &str) -> isize {
    unsafe { pypy_search_in_struct_unions(ctx, name.as_ptr().cast(), name.len()) as isize }
}

/// `pypy_enum_common_types(index)` decoded into its `(name, replacement)`
/// pair.  The C side stores the two as one `"name\0replacement"` block, so the
/// second string starts one past the first one's terminator.
pub fn enum_common_types(index: c_int) -> Option<(&'static str, &'static str)> {
    // SAFETY: the table is a `static const char *[]` in `commontypes.c`, so
    // every entry outlives the process and holds two NUL-terminated ASCII
    // strings back to back.
    unsafe {
        let entry = pypy_enum_common_types(index);
        if entry.is_null() {
            return None;
        }
        let key = core::ffi::CStr::from_ptr(entry);
        let value = core::ffi::CStr::from_ptr(entry.add(key.to_bytes().len() + 1));
        Some((
            core::str::from_utf8_unchecked(key.to_bytes()),
            core::str::from_utf8_unchecked(value.to_bytes()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `commontypes.c` must be reachable and its entries must decode as the
    /// `"name\0replacement"` pairs `_get_common_types` walks.
    #[test]
    fn common_types_table_decodes() {
        let mut seen = Vec::new();
        let mut index = 0;
        while let Some(pair) = enum_common_types(index) {
            seen.push(pair);
            index += 1;
        }
        assert!(seen.contains(&("bool", "_Bool")), "{seen:?}");
        assert!(seen.contains(&("FILE", "struct _IO_FILE")), "{seen:?}");
        // The table is documented as alphabetical and `test_commontypes.py`
        // enforces it; a mis-sorted table makes the binary search in
        // `get_common_type` miss entries.
        let names: Vec<&str> = seen.iter().map(|(name, _)| *name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    /// `pypy_parse_c_type` writes an opcode stream into the caller's buffer.
    #[test]
    fn parse_c_type_emits_an_opcode_stream() {
        // `allocate_ctxobj` hands the parser a zeroed context rather than a
        // null one: `search_in_typenames` reads `ctx->num_typenames` before it
        // can decide it has nothing to search.
        let ctx: TypeContextS = unsafe { core::mem::zeroed() };
        let mut output = vec![core::ptr::null_mut::<c_void>(); FFI_COMPLEXITY_OUTPUT];
        let mut info = ParseInfoS {
            ctx: &ctx,
            output: output.as_mut_ptr(),
            output_size: FFI_COMPLEXITY_OUTPUT as core::ffi::c_uint,
            error_location: 0,
            error_message: core::ptr::null(),
        };
        let input = c"int";
        let result = unsafe { pypy_parse_c_type(&mut info, input.as_ptr()) };
        assert!(result >= 0, "parse failed at {}", info.error_location);
        assert_eq!(getop(output[result as usize]), OP_PRIMITIVE);
        assert_eq!(getarg(output[result as usize]) as usize, PRIM_INT);

        let input = c"int *";
        let result = unsafe { pypy_parse_c_type(&mut info, input.as_ptr()) };
        assert!(result >= 0, "parse failed at {}", info.error_location);
        assert_eq!(getop(output[result as usize]), OP_POINTER);

        let input = c"not_a_type_at_all";
        let result = unsafe { pypy_parse_c_type(&mut info, input.as_ptr()) };
        assert!(result < 0);
        assert!(!info.error_message.is_null());
    }
}
