//! `pypy/objspace/std/frame.py`: the opcode overrides `build_frame`
//! installs on the std objspace frame.

use pyre_object::{INT_TYPE, PyObjectRef, py_type_check, w_int_get_value};

use crate::objspace::descroperation::{_int_add, _int_sub, user_overridden_class};

/// `type(w_1) is W_IntObject and type(w_2) is W_IntObject` in
/// `_intshortcut`'s opimpl: two exact builtin ints, neither a `bool` nor a
/// user subclass (each is a different RPython class there).
#[inline]
pub(crate) unsafe fn both_exact_int(w_1: PyObjectRef, w_2: PyObjectRef) -> bool {
    unsafe {
        py_type_check(w_1, &INT_TYPE)
            && !user_overridden_class(w_1)
            && py_type_check(w_2, &INT_TYPE)
            && !user_overridden_class(w_2)
    }
}

/// `_intshortcut('add')` / `_intshortcut('inplace_add')`'s int arm:
/// `ovfcheck(w_1.intval + w_2.intval)` then `space.newint`, and
/// `W_IntObject.descr_add` on overflow, the split `_int_add` holds.
#[inline]
pub(crate) unsafe fn int_add_arm(w_1: PyObjectRef, w_2: PyObjectRef) -> crate::PyResult {
    unsafe { _int_add(w_int_get_value(w_1), w_int_get_value(w_2)) }
}

/// `_intshortcut('sub')` / `_intshortcut('inplace_sub')`'s int arm; see
/// [`int_add_arm`].
#[inline]
pub(crate) unsafe fn int_sub_arm(w_1: PyObjectRef, w_2: PyObjectRef) -> crate::PyResult {
    unsafe { _int_sub(w_int_get_value(w_1), w_int_get_value(w_2)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_object::{w_bool_from, w_int_new};

    #[test]
    fn both_exact_int_rejects_bool() {
        unsafe {
            assert!(both_exact_int(w_int_new(1), w_int_new(2)));
            assert!(!both_exact_int(w_bool_from(true), w_int_new(1)));
        }
    }
}
