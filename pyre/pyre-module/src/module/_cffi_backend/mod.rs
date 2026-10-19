//! `_cffi_backend` — PyPy: `pypy/module/_cffi_backend/`.
//!
//! cffi's native half.  PyPy implements it at interpreter level rather than as
//! a C extension, which is what lets its JIT see through a foreign call; pyre
//! ports that same interpreter-level module, so `interp_cffi_backend` holds the
//! module surface and [`parse_c_type`] binds the one piece that stays C — the
//! declaration parser whose opcode stream is the format a compiled cffi
//! extension embeds.

pyre_interpreter::pyre_module_init!(interp_cffi_backend);

pub mod allocator;
pub mod cbuffer;
pub mod ccallback;
pub mod cdataobj;
pub mod cdlopen;
pub mod cerrno;
pub mod cffi1_module;
pub mod cglob;
pub mod ctypearray;
pub mod ctypeenum;
pub mod ctypefunc;
pub mod ctypeobj;
pub mod ctypeprim;
pub mod ctypeptr;
pub mod ctypestruct;
pub mod ffi_obj;
pub mod func;
pub mod handle;
pub mod hide_reveal;
pub mod lib_obj;
pub mod libraryobj;
pub mod misc;
pub mod newtype;
pub use pyre_native::cffi as parse_c_type;
pub mod realize_c_type;
pub mod wchar_helper;
pub mod wrapper;

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // A ctype, the array iterator, struct field, allocator, MiniBuffer and
    // offset carrier hold ordinary traced fields; a cdata additionally owns the
    // block that `newp` malloc'd for it and a library owns its loader handle,
    // which their sweep destructors release.
    types.push(ModuleGcType {
        descriptor: <ctypeobj::W_CType as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: Some(gc_destructor!(ctypeobj::w_ctype_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <ctypearray::W_CDataIter as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // Frees the block a `newp`-owned cdata malloc'd; a cdata that only borrows
    // someone else's memory frees nothing.
    types.push(ModuleGcType {
        descriptor: <cdataobj::W_CData as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: Some(std::mem::offset_of!(
                cdataobj::W_CData,
                special_memory_pressure
            )),
        },
        destructor: Some(gc_destructor!(cdataobj::w_cdata_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <ctypestruct::W_CField as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // `W_Library._finalize_` closes a library nothing names any more, unless
    // the handle was opened by someone else.
    types.push(ModuleGcType {
        descriptor: <libraryobj::W_Library as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: Some(gc_destructor!(libraryobj::w_library_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <allocator::W_Allocator as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    types.push(ModuleGcType {
        descriptor: <cbuffer::MiniBuffer as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    types.push(ModuleGcType {
        descriptor: <func::OffsetInBytes as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // `FreeCtxObj.__del__` releases an FFI object's copied parser context.
    // A subclass is `typedef.py` `_getusercls` and shares that destructor.
    let ffi_layout = ModuleGcLayout::PyreClass {
        memory_pressure_offset: None,
    };
    types.push(ModuleGcType {
        descriptor: <ffi_obj::W_FFIObject as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ffi_layout,
        destructor: Some(gc_destructor!(ffi_obj::w_ffi_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: &ffi_obj::W_FFIOBJECT_USER_PYRE_CLASS_DESCRIPTOR,
        layout: ffi_layout,
        destructor: Some(gc_destructor!(ffi_obj::w_ffi_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <realize_c_type::W_RawFuncType as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // `W_DlOpenLibObject._finalize_` closes an ABI library nothing names.
    types.push(ModuleGcType {
        descriptor: <lib_obj::W_LibObject as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: Some(gc_destructor!(lib_obj::w_lib_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <cglob::W_GlobSupport as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    types.push(ModuleGcType {
        descriptor: <wrapper::W_FunctionWrapper as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
