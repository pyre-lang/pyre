//! `pypy/interpreter/special.py` singleton payloads.

use crate::pyobject::*;

/// Python NotImplemented singleton.
/// PyPy: pypy/interpreter/special.py NotImplemented
#[repr(C)]
pub struct NotImplemented {
    pub ob_header: PyObject,
}

pub const W_NOT_IMPLEMENTED_GC_TYPE_ID: u32 = 46;

impl crate::lltype::GcType for NotImplemented {
    fn type_id() -> u32 {
        W_NOT_IMPLEMENTED_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<NotImplemented>();
}

static NOT_IMPLEMENTED_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// `space.w_NotImplemented`, the prebuilt singleton. The body reads the
/// static, which the translator resolves to the registered prebuilt address,
/// so a traced call is that constant, the same shape as [`w_none`].
#[inline]
pub fn w_not_implemented() -> PyObjectRef {
    *NOT_IMPLEMENTED_SINGLETON.get_or_init(|| {
        crate::lltype::malloc_typed_immortal(NotImplemented {
            ob_header: PyObject {
                ob_type: &NOTIMPLEMENTED_TYPE as *const PyType,
                w_class: std::ptr::null_mut(),
            },
        }) as usize
    }) as PyObjectRef
}

/// Python Ellipsis singleton (`...`).
/// PyPy: pypy/interpreter/special.py Ellipsis
#[repr(C)]
pub struct Ellipsis {
    pub ob_header: PyObject,
}

pub const W_ELLIPSIS_GC_TYPE_ID: u32 = 47;

impl crate::lltype::GcType for Ellipsis {
    fn type_id() -> u32 {
        W_ELLIPSIS_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<Ellipsis>();
}

static ELLIPSIS_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Get the Ellipsis singleton.
pub fn w_ellipsis() -> PyObjectRef {
    *ELLIPSIS_SINGLETON.get_or_init(|| {
        crate::lltype::malloc_typed_immortal(Ellipsis {
            ob_header: PyObject {
                ob_type: &ELLIPSIS_TYPE as *const PyType,
                w_class: std::ptr::null_mut(),
            },
        }) as usize
    }) as PyObjectRef
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_singletons_are_header_bearing_prebuilt_objects() {
        for (obj, type_id) in [
            (w_not_implemented(), W_NOT_IMPLEMENTED_GC_TYPE_ID),
            (w_ellipsis(), W_ELLIPSIS_GC_TYPE_ID),
        ] {
            unsafe {
                let hdr = majit_gc::header::header_of(obj as usize);
                assert_eq!((*hdr).type_id(), type_id);
                assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS));
            }
        }
        let a = w_not_implemented();
        let b = w_not_implemented();
        assert!(std::ptr::eq(a, b));
        unsafe {
            assert!(is_not_implemented(a));
            assert!(!is_not_implemented(crate::noneobject::w_none()));
        }
    }
}
