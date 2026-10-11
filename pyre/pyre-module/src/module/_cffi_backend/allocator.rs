//! CFFI allocation policy — PyPy:
//! `pypy/module/_cffi_backend/allocator.py`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj;

/// `allocator.py W_Allocator`.
#[pyre_interpreter::pyre_class("_cffi_backend.__FFIAllocator")]
#[derive(Default)]
pub struct W_Allocator {
    /// `W_Allocator.ffi`; null for the two module-internal allocators.
    pub w_ffi: PyObjectRef,
    /// `W_Allocator.w_alloc`; null selects the raw allocator.
    pub w_alloc: PyObjectRef,
    /// `W_Allocator.w_free`.
    pub w_free: PyObjectRef,
    /// `W_Allocator.should_clear_after_alloc`.
    pub should_clear_after_alloc: i64,
}

/// `allocator.py W_Allocator.allocate`.
pub fn allocate(
    allocator: Option<&mut W_Allocator>,
    datasize: i64,
    w_ctype: PyObjectRef,
    length: i64,
) -> Result<PyObjectRef, PyError> {
    let Some(allocator) = allocator else {
        return cdataobj::new_cdata_owning(w_ctype, datasize, length);
    };
    if allocator.w_alloc.is_null() {
        let ptr = cdataobj::raw_alloc(datasize, allocator.should_clear_after_alloc != 0)?;
        return Ok(cdataobj::new_cdata_full_for_allocator(
            ptr, w_ctype, length, datasize,
        ));
    }

    // `space.call_function(self.w_alloc, space.newint(datasize))` may collect,
    // so all allocator fields and the ctype are read through fixed root slots.
    let roots = pyre_object::gc_roots::push_roots();
    let alloc_slot = roots.base();
    let _ = roots.pin_root(allocator.w_alloc);
    let free_slot = alloc_slot + 1;
    let _ = roots.pin_root(allocator.w_free);
    let ctype_slot = free_slot + 1;
    let _ = roots.pin_root(w_ctype);
    let size_slot = ctype_slot + 1;
    let _ = roots.pin_root(pyre_object::w_int_new(datasize));
    let w_raw = pyre_interpreter::call::call_function_impl_result(
        roots.get(alloc_slot),
        &[roots.get(size_slot)],
    )?;
    let raw_slot = size_slot + 1;
    let _ = roots.pin_root(w_raw);
    let raw = W_CData::from_obj(roots.get(raw_slot)).ok_or_else(|| {
        PyError::type_error(format!(
            "alloc() must return a cdata object (got {})",
            pyre_interpreter::type_methods::arg_type_name(roots.get(raw_slot))
        ))
    })?;
    let raw_ct = ctypeobj::ctype_at(raw.ctype)
        .ok_or_else(|| PyError::system_error("allocator result without a ctype"))?;
    if !raw_ct.is_ptr_or_array() {
        return Err(PyError::type_error(format!(
            "alloc() must return a cdata pointer, not '{}'",
            raw_ct.name()
        )));
    }
    if raw.ptr == 0 {
        return Err(PyError::new(
            pyre_interpreter::PyErrorKind::MemoryError,
            "alloc() returned NULL",
        ));
    }
    if allocator.should_clear_after_alloc != 0 {
        unsafe { std::ptr::write_bytes(raw.ptr as *mut u8, 0, datasize.max(0) as usize) };
    }
    let result = cdataobj::new_cdata_nonstd(
        raw.ptr,
        roots.get(ctype_slot),
        length,
        roots.get(raw_slot),
        roots.get(free_slot),
    );
    cdataobj::add_memory_pressure(result, datasize);
    Ok(result)
}

/// `allocator.py new_allocator`.
pub fn new_allocator(
    w_ffi: PyObjectRef,
    w_alloc: PyObjectRef,
    w_free: PyObjectRef,
    should_clear_after_alloc: bool,
) -> Result<PyObjectRef, PyError> {
    let w_alloc = if unsafe { pyre_object::pyobject::is_none(w_alloc) } {
        pyre_object::PY_NULL
    } else {
        w_alloc
    };
    let w_free = if unsafe { pyre_object::pyobject::is_none(w_free) } {
        pyre_object::PY_NULL
    } else {
        w_free
    };
    if w_alloc.is_null() && !w_free.is_null() {
        return Err(PyError::type_error("cannot pass 'free' without 'alloc'"));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for w_obj in [w_ffi, w_alloc, w_free] {
        let _ = roots.pin_root(w_obj);
    }
    let obj = W_Allocator::allocate_stable(W_Allocator {
        should_clear_after_alloc: i64::from(should_clear_after_alloc),
        ..Default::default()
    });
    let allocator = W_Allocator::from_obj(obj).expect("allocate_stable hands back this layout");
    allocator.w_ffi = roots.get(base);
    allocator.w_alloc = roots.get(base + 1);
    allocator.w_free = roots.get(base + 2);
    // The allocator is born old-gen and its three policy objects may be
    // young, so the post-allocation writes need a remembered-set entry.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    Ok(obj)
}

static ALLOCATOR_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.__FFIAllocator`.
pub fn allocator_type() -> PyObjectRef {
    ALLOCATOR_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.__FFIAllocator",
            init_allocator_type,
            pyre_interpreter::typedef::w_object(),
            <W_Allocator as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Allocator as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp
    })
}

fn init_allocator_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__call__",
        pyre_interpreter::make_builtin_function("__call__", allocator_call)
    )
}

/// `W_Allocator.descr_call`.
fn allocator_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // `interp2app(W_Allocator.descr_call)` — `self`, a required `arg`, and an
    // `init` that `WrappedDefault(None)` supplies.  Binding here is what makes
    // `alloc()` a TypeError rather than an out-of-bounds read.
    let bound = super::func::bind_entry_point(
        args,
        "__FFIAllocator.__call__",
        &["self", "arg", "init"],
        2,
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for value in [
        bound[0],
        bound[1],
        if bound[2].is_null() {
            pyre_object::w_none()
        } else {
            bound[2]
        },
    ] {
        let _ = roots.pin_root(value);
    }
    let allocator = W_Allocator::from_obj(roots.get(base))
        .ok_or_else(|| PyError::type_error("expected an allocator object"))?;
    let w_ctype = if allocator.w_ffi.is_null() {
        let _ = ctypeobj::ctype_arg(roots.get(base + 1))?;
        roots.get(base + 1)
    } else {
        super::ffi_obj::ffi_type(
            allocator.w_ffi,
            roots.get(base + 1),
            super::ffi_obj::ACCEPT_STRING | super::ffi_obj::ACCEPT_CTYPE,
        )?
    };
    ctypeobj::newp_with_allocator(w_ctype, roots.get(base + 2), roots.get(base))
}
