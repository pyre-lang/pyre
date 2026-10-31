//! `pypy/interpreter/nestedscope.py` — Python `cell` type for closures.
//!
//! A cell holds a reference to a single value. Closures use cells to
//! share mutable bindings between an outer function and its nested
//! inner functions.

use crate::pyobject::*;
use crate::quasiimmut::QuasiImmutField;
use pyre_macros::pyre_class;
use std::sync::Arc;

/// nestedscope.py `class CellFamily` — one per cellvar name of a code
/// object (`pycode.py` `PyCode._initialize`).
///
/// Every cell a frame creates for that cellvar shares this object, so
/// `ever_mutated` accumulates across frame instantiations: a binding that is
/// only ever filled once leaves it false, and that is what lets a cell the
/// trace has as a constant fold its contents.
///
/// `repr(C)` pins field offsets to declaration order so the JIT descr
/// group's `offset_of!` matches the walker arg0 (the family itself).
/// `name: String` stays first. `nestedscope.py CellFamily._immutable_fields_
/// = ['ever_mutated?']`.
#[repr(C)]
#[majit_macros::jit_immutable_fields("ever_mutated?")]
pub struct CellFamily {
    /// nestedscope.py `CellFamily.__init__` `name` — the cellvar this family belongs to.
    pub name: String,
    /// nestedscope.py `CellFamily.__init__` `ever_mutated`, declared quasi-immutable by
    /// nestedscope.py `_immutable_fields_ = ['ever_mutated?']`.  Recorded
    /// on a bound-to-bound transition ([`w_cell_set`]) and by
    /// [`w_cell_delete`]; [`Self::set_ever_mutated`] invalidates watchers
    /// before changing it.
    pub ever_mutated: std::cell::Cell<bool>,
    /// Hidden `mutate_ever_mutated` field (`quasiimmut.py get_mutate_field_name`
    /// of pyre `ever_mutated`; `nestedscope.py CellFamily._immutable_fields_ =
    /// ['ever_mutated?']`). The owner is a `Box::into_raw` leak that is
    /// never freed, so [`QuasiImmutField`]'s `Drop` is unreachable and its
    /// inner box is reclaimed only through [`QuasiImmutField::invalidate`].
    pub mutate_ever_mutated: QuasiImmutField,
}

/// Host `CellFamily` is a `Box::into_raw`'d host allocation, never a GC object.
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _alloc_flavor_CellFamily: &str = "raw";

impl CellFamily {
    /// nestedscope.py `CellFamily.__init__`.
    pub fn new(name: String) -> Self {
        Self {
            name,
            ever_mutated: std::cell::Cell::new(false),
            mutate_ever_mutated: QuasiImmutField::new(),
        }
    }

    /// `rclass.py InstanceRepr.hook_setfield` `jit_force_quasi_immutable(vinst, 'mutate_' + name)`, spelled
    /// the way `warmspot.py rewrite_force_quasi_immutable` leaves it for the native build:
    /// `make_invalidation_function` — `if getattr(p, mutatefieldname): _invalidate_now(p)`. The translator
    /// keys on the `#[jit_llop]` marker to rewrite this call into the llop
    /// (`jtransform.py rewrite_op_jit_force_quasi_immutable`), so a trace never looks inside it.
    #[majit_macros::jit_llop("jit_force_quasi_immutable")]
    #[inline(never)]
    pub fn jit_force_quasi_immutable(&self, mutatefieldname: &'static str) {
        match mutatefieldname {
            "mutate_ever_mutated" => {
                if !self.mutate_ever_mutated.is_installed() {
                    return;
                }
                unsafe { crate::quasiimmut::sweep_quasi_immut_field(&self.mutate_ever_mutated) };
            }
            _ => panic!("unknown mutate field name {mutatefieldname:?}"),
        }
    }

    /// Change `CellFamily.__init__`'s `ever_mutated` (`nestedscope.py`) using
    /// the ordering, fast path, and repeated-value guard
    /// `PlainAttribute::set_ever_mutated` documents.
    pub fn set_ever_mutated(&self, v: bool) {
        if self.ever_mutated.get() == v {
            return;
        }
        self.jit_force_quasi_immutable("mutate_ever_mutated");
        self.ever_mutated.set(v);
    }

    pub fn current_ever_mutated_qmut(&self) -> Arc<crate::quasiimmut::QuasiImmut> {
        self.mutate_ever_mutated.get_current_qmut_instance()
    }

    pub fn ever_mutated_qmut_installed(&self) -> bool {
        self.mutate_ever_mutated.is_installed()
    }

    pub fn force_ever_mutated_qmut(&self) {
        self.mutate_ever_mutated.invalidate();
    }
}

/// nestedscope.py `DUMMY_FAMILY` — the family a cell built by hand
/// carries (`descr_new_cell`, and any cell whose code object has no family
/// table).  Its `ever_mutated` starts true, so such a cell never folds.
///
/// Held as a leaked address rather than a `OnceLock<CellFamily>` because the
/// `std::cell::Cell<bool>` inside makes the family non-`Sync`, and the
/// allocation must outlive every cell pointing at it anyway.
///
/// This one family is shared by every cell that has no table of its own, so it
/// must stay WRITE-FREE, and the `usize` the `OnceLock` stores hides its
/// interior mutability from the compiler — nothing here is checked for you.
/// The invariant holds today by arithmetic rather than by construction: cells
/// carrying this family do reach [`CellFamily::set_ever_mutated`] (the rebind
/// and delete paths below both call it), but every caller passes `true` and
/// this family is born `true`, so each such call takes that method's
/// unchanged-value early return and never reaches the `Cell` write or the
/// watcher sweep.  A caller that passed `false` would write through a shared
/// address and sweep the quasi-immut watchers of every unrelated cell that
/// landed on the dummy — a process-wide effect rather than a data race, since
/// the GIL (`majit-gc` `rgil`) already serialises mutators.
pub fn dummy_family() -> *const CellFamily {
    static DUMMY_FAMILY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DUMMY_FAMILY.get_or_init(|| {
        let family = CellFamily::new("<dummy>".to_string());
        family.ever_mutated.set(true);
        Box::into_raw(Box::new(family)) as usize
    }) as *const CellFamily
}

/// Python cell object.
///
/// Layout: `[ob_type: *const PyType | contents: PyObjectRef | family]`
/// `contents` is `PY_NULL` when the cell is empty.
#[pyre_class("cell", type_id = 15, static_name = "CELL")]
pub struct Cell {
    pub contents: PyObjectRef,
    /// nestedscope.py `Cell` `_immutable_fields_ = ['family']` — the
    /// [`CellFamily`] shared with every other cell of the same cellvar.  Never
    /// null.  The families are leaked, not managed, so this word is absent
    /// from the traced pointer offsets and outlives the code object that
    /// created it — a closure keeps its cells alive long after the enclosing
    /// code object is collected.
    pub family: *const CellFamily,
}

/// Allocate a new cell wrapping `value` (`nestedscope.py` `Cell.__init__`).
/// Pass `PY_NULL` for an empty cell, and [`dummy_family`] when no code object
/// owns the binding.
pub fn w_cell_new(value: PyObjectRef, family: *const CellFamily) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`): `value`
    // is a live GC pointer that must survive — and be relocated by — the
    // collection the GC malloc below may trigger. `pin_root` records it in
    // the shadow stack so the moving collector keeps it alive and rewrites
    // the slot; we read the relocated address back after the malloc.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::pin_roots(&[value]);
    let header = PyObject {
        ob_type: &CELL_TYPE as *const PyType,
        w_class: get_instantiate(&CELL_TYPE),
    };
    // Same nursery bump as `w_tuple_new` / `w_slice_new` (`malloc_fixedsize`).
    // A cell whose `contents` is reachable only through this cell (e.g. a
    // closure cellvar) must itself be GC-traced; a `malloc_typed`
    // (`std::alloc`) cell is invisible to `is_managed_heap_object`, so the
    // mark-sweep skips it and the only-reachable-via-cell value is swept.
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_CELL_GC_TYPE_ID, W_CELL_OBJECT_SIZE);
    let value = crate::gc_roots::shadow_stack_get(save_point);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut Cell,
                Cell {
                    ob: header,
                    contents: value,
                    family,
                },
            );
        }
        // Nursery-full spills this header to old-gen while `contents` may
        // still be young. `try_gc_write_barrier` is a no-op on a young
        // header and records the old-gen store when the bump spilled
        // (incminimark.py `write_barrier`).
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    Cell::allocate(Cell {
        ob: header,
        contents: value,
        family,
    })
}

/// Check if an object is a cell.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_cell(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CELL_TYPE) }
}

/// `nestedscope.py Cell._elidable_get` — `@jit.elidable` plain read of
/// `w_value`.
///
/// # Safety
/// `obj` must point to a valid `Cell`.
#[inline]
#[majit_macros::elidable]
pub unsafe fn w_cell_elidable_get(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Cell)).contents }
}

/// `nestedscope.py Cell.get`:
///
/// ```python
/// def get(self):
///     if jit.isconstant(self):
///         if not self.family.ever_mutated:
///             w_res = self._elidable_get()
///             if w_res is not None:
///                 return w_res
///     if self.w_value is None:
///         raise ValueError("get() from an empty cell")
///     return self.w_value
/// ```
///
/// The empty-cell `ValueError` stays with `LOAD_DEREF`; this returns
/// `PY_NULL` for an empty cell the way the previous body did. A null
/// family is treated as mutated.
///
/// # Safety
/// `obj` must point to a valid `Cell`.
#[inline]
pub unsafe fn w_cell_get(obj: PyObjectRef) -> PyObjectRef {
    if majit_rlib::jit::isconstant(&obj) {
        let family = unsafe { w_cell_family(obj) };
        if !family.is_null() && !unsafe { (*family).ever_mutated.get() } {
            let w_res = unsafe { w_cell_elidable_get(obj) };
            if !w_res.is_null() {
                return w_res;
            }
        }
    }
    unsafe { (*(obj as *const Cell)).contents }
}

/// The [`CellFamily`] shared by every cell of this cell's cellvar.
///
/// # Safety
/// `obj` must point to a valid `Cell`.
#[inline]
pub unsafe fn w_cell_family(obj: PyObjectRef) -> *const CellFamily {
    unsafe { (*(obj as *const Cell)).family }
}

/// Set the value stored in a cell (nestedscope.py `Cell.set`).
///
/// # Safety
/// `obj` must point to a valid `Cell`.
#[inline]
pub unsafe fn w_cell_set(obj: PyObjectRef, value: PyObjectRef) {
    // nestedscope.py `Cell.set`: only a bound-to-bound transition counts as a
    // mutation.  The first binding of a cellvar does not, which is what keeps
    // the ordinary write-once closure variable foldable.
    let cell = obj as *mut Cell;
    if !unsafe { (*cell).contents }.is_null() {
        let family = unsafe { (*cell).family };
        if !family.is_null() {
            unsafe { (*family).set_ever_mutated(true) };
        }
    }
    unsafe { (*cell).contents = value }
    // The cell is an old-gen (`try_gc_alloc_stable`) object; storing a
    // possibly-nursery `value` into it needs the incminimark write barrier
    // (incminimark.py write_barrier_from_array) so the next minor collection scans the cell and
    // relocates the young value held only by `contents`.
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// Clear a cell (nestedscope.py `Cell.delete`).  Records the mutation
/// whatever the cell held, then clears it; returns false instead of raising
/// upstream's `ValueError` when the cell was already empty.
///
/// # Safety
/// `obj` must point to a valid `Cell`.
#[inline]
pub unsafe fn w_cell_delete(obj: PyObjectRef) -> bool {
    let cell = obj as *mut Cell;
    let family = unsafe { (*cell).family };
    if !family.is_null() {
        unsafe { (*family).set_ever_mutated(true) };
    }
    if unsafe { (*cell).contents }.is_null() {
        return false;
    }
    unsafe { (*cell).contents = PY_NULL }
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A family that lives as long as the test, standing in for the one a code
    /// object owns.
    fn leak_family() -> *const CellFamily {
        Box::into_raw(Box::new(CellFamily::new("x".to_string()))) as *const CellFamily
    }

    #[test]
    fn test_cell_create() {
        let cases = [("empty", PY_NULL), ("with_value", 0xDEAD as PyObjectRef)];
        for (name, value) in cases {
            let cell = w_cell_new(value, leak_family());
            unsafe {
                assert!(is_cell(cell), "case {name}");
                assert_eq!(w_cell_get(cell), value, "case {name}");
            }
        }
    }

    #[test]
    fn test_cell_set() {
        let cell = w_cell_new(PY_NULL, leak_family());
        let value = 0xBEEF as PyObjectRef;
        unsafe {
            w_cell_set(cell, value);
            assert_eq!(w_cell_get(cell), value);
        }
    }

    /// nestedscope.py — the first binding of an empty cell is not a
    /// mutation; a second write to a bound cell is.
    #[test]
    fn first_binding_is_not_a_mutation() {
        let family = leak_family();
        let cell = w_cell_new(PY_NULL, family);
        unsafe {
            w_cell_set(cell, 0xBEEF as PyObjectRef);
            assert!(!(*family).ever_mutated.get());
            w_cell_set(cell, 0xCAFE as PyObjectRef);
            assert!((*family).ever_mutated.get());
        }
    }

    /// The family is shared, so one cell's rebinding closes folding for every
    /// other cell of the same cellvar — the accumulation across frame
    /// instantiations `pycode.py` `PyCode._initialize` exists for.
    #[test]
    fn mutation_is_recorded_on_the_shared_family() {
        let family = leak_family();
        let first = w_cell_new(PY_NULL, family);
        let second = w_cell_new(PY_NULL, family);
        unsafe {
            w_cell_set(first, 0xBEEF as PyObjectRef);
            w_cell_set(first, 0xCAFE as PyObjectRef);
            assert!((*w_cell_family(second)).ever_mutated.get());
        }
    }

    /// nestedscope.py — `delete` records the mutation whatever the cell
    /// held, unlike `set`, and reports the already-empty case instead of
    /// raising.
    #[test]
    fn delete_always_records_the_mutation() {
        let family = leak_family();
        let cell = w_cell_new(PY_NULL, family);
        unsafe {
            assert!(!w_cell_delete(cell));
            assert!((*family).ever_mutated.get());

            let bound = leak_family();
            let cell = w_cell_new(0xBEEF as PyObjectRef, bound);
            assert!(w_cell_delete(cell));
            assert!(w_cell_get(cell).is_null());
            assert!((*bound).ever_mutated.get());
        }
    }

    /// nestedscope.py `DUMMY_FAMILY` — a hand-built cell can never fold.
    #[test]
    fn dummy_family_starts_mutated() {
        unsafe {
            assert!((*dummy_family()).ever_mutated.get());
        }
        assert_eq!(dummy_family(), dummy_family());
    }

    /// `nestedscope.py Cell.get` falls through to `w_value` when the family
    /// is already mutated; `_elidable_get` is the same field.
    #[test]
    fn get_matches_elidable_get_on_a_dummy_family_cell() {
        let cell = w_cell_new(0xBEEF as PyObjectRef, dummy_family());
        unsafe {
            assert_eq!(w_cell_get(cell), w_cell_elidable_get(cell));
            assert_eq!(w_cell_get(cell), 0xBEEF as PyObjectRef);
        }
    }

    /// Guard against drift between the constant colocated with
    /// `Cell` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT/FUNCTION trip-wire
    /// tests.
    #[test]
    fn w_cell_gc_type_id_matches_descr() {
        assert_eq!(W_CELL_GC_TYPE_ID, 15);
        assert_eq!(
            <Cell as crate::lltype::GcType>::type_id(),
            W_CELL_GC_TYPE_ID
        );
        assert_eq!(<Cell as crate::lltype::GcType>::SIZE, W_CELL_OBJECT_SIZE);
    }
}
