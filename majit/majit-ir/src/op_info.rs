//! `OpInfo` — abstract information attached to operations during optimization.
//!
//! info.py: `AbstractInfo` hierarchy. Each operation can have associated
//! analysis info (e.g. known integer bounds, pointer info, virtual object
//! state). Hosted in `majit-ir` so the `Forwarded` move that follows can
//! reference it without a `majit-metainterp → majit-ir` circular dep.
//!
//! Pure data only. The `Rc<RefCell<…>>` wrappers around `IntBound` /
//! `PtrInfo` provide shared-identity semantics matching PyPy's
//! `_forwarded` slot — see `info.py get*ptrinfo` "return fw".

use crate::intbound::{IntBound, IntBoundRc};
use crate::ptr_info::PtrInfo;

/// info.py `FloatConstInfo`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatConstInfo {
    pub _const: f64,
}

impl FloatConstInfo {
    pub fn new(value: f64) -> Self {
        Self { _const: value }
    }

    pub fn is_constant(&self) -> bool {
        true
    }

    pub fn getconst(&self) -> f64 {
        self._const
    }
}

/// shortpreamble.py `EmptyInfo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptyInfo;

impl EmptyInfo {
    pub fn make_guards(&self) {}
}

/// `info.py` `AbstractInfo` hierarchy collapsed into a Rust enum.
///
/// `Ptr` carries `Rc<RefCell<PtrInfo>>` so the underlying info object has
/// the same object-identity semantics as RPython's `_forwarded` slot:
/// when two `_forwarded` slots are set to the same `Ptr(rc.clone())`,
/// in-place mutations through one handle are observable through the
/// other. `IntBound` is similarly wrapped so `optimizer.py getintbound
/// getintbound` mutations propagate.
#[derive(Clone)]
pub enum OpInfo {
    /// No information known.
    Unknown,
    /// shortpreamble.py `empty_info` sentinel.
    EmptyInfo(EmptyInfo),
    /// Known integer bounds. `intutils.py` `IntBound`.
    /// `IntBound::from_constant(v)` is the canonical Int constant carrier.
    /// Chunked [`IntBoundRc`] so first mint leaves the 56-byte
    /// `RcBox<RefCell<IntBound>>` class.
    IntBound(IntBoundRc),
    /// Pointer info (non-null, known class, virtual, etc.).
    /// `PtrInfo::Constant(GcRef)` is the Ref constant carrier
    /// (info.py ConstPtrInfo).
    Ptr(std::rc::Rc<std::cell::RefCell<PtrInfo>>),
    /// Known constant float value.
    /// info.py FloatConstInfo — Float constant carrier.
    FloatConstInfo(FloatConstInfo),
    /// `regalloc.py Lifetime(AbstractInfo)` stored by `get_lifetime` /
    /// `LifetimeManager.__setitem__` (`box.set_forwarded(lt)`). Packed
    /// through `FW_INFO_OTHER`. Not optimizer information.
    Lifetime(std::rc::Rc<dyn std::any::Any>),
}

impl std::fmt::Debug for OpInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpInfo::Unknown => f.write_str("OpInfo::Unknown"),
            OpInfo::EmptyInfo(v) => f.debug_tuple("OpInfo::EmptyInfo").field(v).finish(),
            OpInfo::IntBound(ib) => f
                .debug_tuple("OpInfo::IntBound")
                .field(&*ib.borrow())
                .finish(),
            OpInfo::Ptr(p) => f.debug_tuple("OpInfo::Ptr").field(&*p.borrow()).finish(),
            OpInfo::FloatConstInfo(v) => f.debug_tuple("OpInfo::FloatConstInfo").field(v).finish(),
            OpInfo::Lifetime(_) => f.write_str("Lifetime"),
        }
    }
}

impl OpInfo {
    /// Helper for constructing `OpInfo::Ptr` from owned `PtrInfo` —
    /// wraps in a fresh `Rc<RefCell<>>` for the shared-identity storage.
    pub fn ptr(info: PtrInfo) -> Self {
        OpInfo::Ptr(std::rc::Rc::new(std::cell::RefCell::new(info)))
    }

    /// Helper for constructing `OpInfo::IntBound` from owned `IntBound`.
    pub fn int_bound(b: IntBound) -> Self {
        OpInfo::IntBound(IntBoundRc::new(b))
    }

    pub fn is_constant(&self) -> bool {
        match self {
            OpInfo::FloatConstInfo(v) => v.is_constant(),
            OpInfo::Ptr(p) => matches!(&*p.borrow(), PtrInfo::Constant(_)),
            OpInfo::IntBound(b) => b.borrow().is_constant(),
            OpInfo::Unknown | OpInfo::EmptyInfo(_) | OpInfo::Lifetime(_) => false,
        }
    }

    /// Get the constant float value if this is a FloatConst.
    pub fn get_constant_float(&self) -> Option<f64> {
        match self {
            OpInfo::FloatConstInfo(f) => Some(f.getconst()),
            _ => None,
        }
    }

    /// Returns the live slab handle to the `IntBound` for the `IntBound`
    /// variant. Mirrors RPython object identity: callers that retain the
    /// handle observe in-place mutations through other holders.
    pub fn get_int_bound(&self) -> Option<&IntBoundRc> {
        match self {
            OpInfo::IntBound(b) => Some(b),
            _ => None,
        }
    }

    /// Whether this info is known non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            OpInfo::Ptr(p) => p.borrow().is_nonnull(),
            _ => false,
        }
    }

    /// Whether this info represents a virtual (allocation-removed) object.
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(self, OpInfo::Ptr(p) if p.borrow().is_virtual())
    }

    /// Returns the live `Rc` handle to the `PtrInfo` for the `Ptr`
    /// variant. Mirrors RPython object identity: callers that retain the
    /// handle observe in-place mutations through other holders. Borrow
    /// the returned handle (`handle.borrow()` / `handle.borrow_mut()`)
    /// to access the inner `PtrInfo`.
    pub fn get_ptr_info(&self) -> Option<&std::rc::Rc<std::cell::RefCell<PtrInfo>>> {
        match self {
            OpInfo::Ptr(p) => Some(p),
            _ => None,
        }
    }
}
