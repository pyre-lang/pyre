//! select.kevent — PyPy: pypy/module/select/interp_kqueue.py W_Kevent.
//!
//! Each `#[pyre_class]` emits a module-scoped `type_object()`, so kevent
//! and kqueue live in separate files to avoid a name clash.

#![allow(dead_code)]

#[cfg(all(target_os = "macos", feature = "host_env"))]
use super::interp_select::filedescriptor_w;
#[cfg(all(target_os = "macos", feature = "host_env"))]
use pyre_object::PyObjectRef;

/// `select.kevent` object — PyPy: `interp_kqueue.py class W_Kevent`.
///
/// Mirrors the platform `struct kevent`: a 64-bit `ident`, signed
/// 16-bit `filter`, 16-bit `flags`, 32-bit `fflags`, signed 64-bit
/// `data`, and an opaque pointer-sized `udata`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
// CPython 3.14 Modules/selectmodule.c:select_exec creates
// kqueue_event_Type_spec as a mutable module heap type.
#[pyre_interpreter::pyre_class("select.kevent", cpython_mutable)]
#[derive(Default)]
pub struct W_Kevent {
    pub ident: u64,
    pub filter: i16,
    pub flags: u16,
    pub fflags: u32,
    pub data: i64,
    pub udata: u64,
}

/// `interp_kqueue.py` `_compare_all_fields`. Field-by-field, not a
/// packed-tuple `cmp`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
impl W_Kevent {
    fn compare_all_fields_raw(&self, other: &W_Kevent, op: &str) -> bool {
        let l_ident = self.ident;
        let r_ident = other.ident;
        let l_filter = self.filter as i64;
        let r_filter = other.filter as i64;
        let l_flags = self.flags as u32;
        let r_flags = other.flags as u32;
        let l_fflags = self.fflags;
        let r_fflags = other.fflags;
        let l_data = self.data;
        let r_data = other.data;
        let l_udata = self.udata;
        let r_udata = other.udata;
        match op {
            "eq" => {
                l_ident == r_ident
                    && l_filter == r_filter
                    && l_flags == r_flags
                    && l_fflags == r_fflags
                    && l_data == r_data
                    && l_udata == r_udata
            }
            "lt" => {
                l_ident < r_ident
                    || (l_ident == r_ident && l_filter < r_filter)
                    || (l_ident == r_ident && l_filter == r_filter && l_flags < r_flags)
                    || (l_ident == r_ident
                        && l_filter == r_filter
                        && l_flags == r_flags
                        && l_fflags < r_fflags)
                    || (l_ident == r_ident
                        && l_filter == r_filter
                        && l_flags == r_flags
                        && l_fflags == r_fflags
                        && l_data < r_data)
                    || (l_ident == r_ident
                        && l_filter == r_filter
                        && l_flags == r_flags
                        && l_fflags == r_fflags
                        && l_data == r_data
                        && l_udata < r_udata)
            }
            "gt" => {
                l_ident > r_ident
                    || (l_ident == r_ident && l_filter > r_filter)
                    || (l_ident == r_ident && l_filter == r_filter && l_flags > r_flags)
                    || (l_ident == r_ident
                        && l_filter == r_filter
                        && l_flags == r_flags
                        && l_fflags > r_fflags)
                    || (l_ident == r_ident
                        && l_filter == r_filter
                        && l_flags == r_flags
                        && l_fflags == r_fflags
                        && l_data > r_data)
                    || (l_ident == r_ident
                        && l_filter == r_filter
                        && l_flags == r_flags
                        && l_fflags == r_fflags
                        && l_data == r_data
                        && l_udata > r_udata)
            }
            _ => unreachable!("compare_all_fields_raw op"),
        }
    }

    /// `interp_kqueue.py` `compare_all_fields`.
    fn compare_all_fields(&self, other: &W_Kevent, mut op: &str) -> bool {
        let mut negate = false;
        if op == "ne" {
            negate = true;
            op = "eq";
        } else if op == "le" {
            negate = true;
            op = "gt";
        } else if op == "ge" {
            negate = true;
            op = "lt";
        }
        let r = self.compare_all_fields_raw(other, op);
        if negate { !r } else { r }
    }
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
#[pyre_interpreter::pyre_methods(
    doc = "kevent(ident, filter=KQ_FILTER_READ, flags=KQ_EV_ADD, fflags=0, data=0, udata=0)"
)]
impl W_Kevent {
    /// `interp_kqueue.py descr__init__`.  Mirrors
    /// `@unwrap_spec(filter=int, flags='c_uint', fflags='c_uint', data=int,
    /// udata=r_uint)`: `ident` is `uint_w` for an int else
    /// `c_filedescriptor_w`; `flags`/`fflags` are `c_uint` (reject
    /// negative / >0xffffffff); `udata` is `r_uint` (full unsigned word).
    fn __init__(
        &mut self,
        w_ident: PyObjectRef,
        #[default(pyre_object::w_int_new(libc::EVFILT_READ as i64))] mut w_filter: PyObjectRef,
        #[default(pyre_object::w_int_new(libc::EV_ADD as i64))] mut w_flags: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] mut w_fflags: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] mut w_data: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] mut w_udata: PyObjectRef,
    ) -> Result<(), pyre_interpreter::PyError> {
        let ident: u64 = if unsafe { pyre_object::is_int(w_ident) } {
            pyre_interpreter::baseobjspace::uint_w(w_ident)?
        } else {
            pyre_object::with_roots!(w_data, w_fflags, w_filter, w_flags, w_udata => filedescriptor_w(w_ident))?
                as u64
        };
        let filter = pyre_object::with_roots!(w_data, w_fflags, w_flags, w_udata => pyre_interpreter::baseobjspace::int_w(w_filter))?;
        let flags = pyre_interpreter::baseobjspace::c_uint_w(w_flags)?;
        let fflags = pyre_interpreter::baseobjspace::c_uint_w(w_fflags)?;
        let data =
            pyre_object::with_roots!(w_udata => pyre_interpreter::baseobjspace::int_w(w_data))?;
        let udata = pyre_interpreter::baseobjspace::uint_w(w_udata)?;
        self.ident = ident;
        self.filter = filter as i16;
        self.flags = flags as u16;
        self.fflags = fflags;
        self.data = data;
        self.udata = udata;
        Ok(())
    }

    #[getter]
    fn ident(&self) -> PyObjectRef {
        newint_from_u64(self.ident)
    }
    #[getter]
    fn filter(&self) -> i64 {
        self.filter as i64
    }
    #[getter]
    fn flags(&self) -> i64 {
        self.flags as i64
    }
    #[getter]
    fn fflags(&self) -> i64 {
        self.fflags as i64
    }
    #[getter]
    fn data(&self) -> i64 {
        self.data
    }
    #[getter]
    fn udata(&self) -> PyObjectRef {
        newint_from_u64(self.udata)
    }

    /// `kqueue_event_repr` — all six fields, `ident` unsigned decimal,
    /// `filter` signed decimal, `flags` / `fflags` / `data` hex, and `udata`
    /// through `%p`, whose leading `0x` `PyUnicode_FromFormat` guarantees.
    /// `data` is formatted as `long long`, so a negative one reads as its
    /// two's complement.  `interp_kqueue.py W_Kevent` registers no `__repr__`
    /// at all.
    fn __repr__(&self) -> String {
        format!(
            "<select.kevent ident={} filter={} flags=0x{:x} fflags=0x{:x} \
             data=0x{:x} udata=0x{:x}>",
            self.ident, self.filter, self.flags, self.fflags, self.data, self.udata,
        )
    }

    /// `interp_kqueue.py descr__eq__` and friends. A non-kevent other
    /// yields `NotImplemented`.
    fn __eq__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, "eq")
    }
    fn __ne__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, "ne")
    }
    fn __lt__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, "lt")
    }
    fn __le__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, "le")
    }
    fn __gt__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, "gt")
    }
    fn __ge__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, "ge")
    }
}

/// Wrap a 64-bit unsigned word as a Python int, mirroring `space.newint`
/// of a `UINTPTR_T`: values in `i64` range become a plain int, larger ones
/// become a positive long instead of wrapping to a negative via `as i64`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
fn newint_from_u64(v: u64) -> PyObjectRef {
    if v <= i64::MAX as u64 {
        pyre_object::w_int_new(v as i64)
    } else {
        pyre_object::w_long_new(majit_rlib::rbigint::RBigInt::from(v))
    }
}

/// Shared body for the kevent rich-comparison dunders.
#[cfg(all(target_os = "macos", feature = "host_env"))]
fn kevent_compare(this: &W_Kevent, w_other: PyObjectRef, op: &str) -> PyObjectRef {
    match W_Kevent::from_obj(w_other) {
        Some(other) => pyre_object::w_bool_from(this.compare_all_fields(other, op)),
        None => pyre_object::w_not_implemented(),
    }
}
