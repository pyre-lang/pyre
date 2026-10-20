//! Core Python object model with `#[repr(C)]` layout for JIT compatibility.
//!
//! Every Python object starts with a `PyObject` header containing a type pointer.
//! Concrete types (W_IntObject, W_BoolObject, etc.) embed this header as their
//! first field, enabling safe pointer casts between `*mut PyObject` and typed pointers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};

/// Type descriptor for Python objects — corresponds to RPython's OBJECT_VTABLE
/// (rclass.py).
///
/// Each built-in type has a single static `PyType` instance.
/// The JIT uses `GuardClass` on the `ob_type` pointer to specialize code paths,
/// and `GuardSubclass` via `int_between(cls.min, subcls.min, cls.max)`
/// (rclass.py `ll_issubclass`).
///
/// Fields match OBJECT_VTABLE layout order:
///   subclassrange_min, subclassrange_max, (rtti omitted), name, (instantiate omitted)
///
/// `AtomicI64`/`AtomicPtr` provide interior mutability for static instances:
/// ranges and instantiate are assigned once at init time,
/// mirroring `assign_inheritance_ids` (normalizecalls.py).
/// The JIT backend reads them at raw offsets — atomics are layout-
/// compatible with their inner types (same size and alignment).
/// `rclass.py` gives OBJECT_VTABLE `hints={'immutable': True,
/// 'static_immutable': True}`, and `lltype.py _immutable_field`
/// answers `True` for *every* field of a struct carrying that hint, so
/// each vtable slot reaches `descr.py:231 is_pure = STRUCT.
/// _immutable_field(fieldname) != False` as a pure field.  Pyre's
/// per-field spelling of the same declaration: all four slots are
/// written once during startup (`assign_subclass_range` /
/// `set_instantiate`, both before any bytecode runs) and never again.
#[repr(C)]
#[majit_macros::jit_immutable_fields(
    "subclassrange_min",
    "subclassrange_max",
    "name",
    "instantiate",
    "mapdict_offset",
    "weakref_offset",
    "user_subclass",
    "user_base"
)]
pub struct PyType {
    pub subclassrange_min: AtomicI64,
    pub subclassrange_max: AtomicI64,
    pub name: &'static str,
    /// rclass.py `('instantiate', Ptr(FuncType([], OBJECTPTR)))`.
    ///
    /// RPython stores an instantiate function pointer; pyre caches
    /// the W_TypeObject pointer here instead. rclass.py
    /// `new_instance` sets `__class__` at allocation — pyre reads
    /// this cached pointer to set `w_class` at allocation time.
    /// Null until `init_typeobjects()` runs.
    pub instantiate: AtomicPtr<PyObject>,
    /// Byte offset from the object start of the `MapdictStorageMixin`
    /// `map` word (`mapdict.py` `import_from_mixin` /
    /// `typedef._getusercls`). `storage` is the next word, at
    /// `mapdict_offset + size_of::<usize>()`. `0` means the class has no
    /// mixin: offset 0 is the header's `ob_type`, so it cannot be a real
    /// mixin offset. The offset lives on the typeptr, the RPython class,
    /// not on a caller-side type whitelist.
    pub mapdict_offset: usize,
    /// Byte offset of the `make_weakref_descr` `_lifeline_` slot
    /// (`typedef.py`). `0` means the class has no lifeline field: offset 0
    /// is the header's `ob_type`, so it cannot be a real field offset.
    /// A `_getusercls` class of a weakrefable base inherits the base's
    /// offset (the user struct keeps the base as a prefix). A non-weakrefable
    /// base leaves this at 0 and `MapdictWeakrefSupport` uses the `"weakref"`
    /// SPECIAL slot instead.
    pub weakref_offset: usize,
    /// `typedef.py get_unique_interplevel_subclass(space, cls)` answered
    /// ahead of time: the class every user subclass instance of this
    /// builtin carries as its typeptr (`_unique_subclass_cache[cls]`).  Null
    /// for a class whose user subclasses do not have one yet.  Since every
    /// user subclass instance carries that typeptr, an object whose typeptr
    /// is this class is an exact instance.
    pub user_subclass: *const PyType,
    /// The builtin class a `_getusercls` class was made from: its instances
    /// share that class's typedef and payload layout.  Null for every other
    /// class.
    pub user_base: *const PyType,
}

/// Common header for all Python objects.
///
/// RPython rclass.py: OBJECT = GcStruct('object', ('typeptr', CLASSTYPE))
///
/// - `ob_type`: static dispatch tag (like RPython's typeptr for guard_class)
/// - `w_class`: Python class pointer (like RPython's gettypefor(typeptr) result)
///
/// `w_class` is set at allocation time when the type registry is available,
/// or populated lazily by `init_typeobjects()` for static singletons.
///
/// `ob_type` is declared immutable and `w_class` is not, which is the split
/// upstream draws between the two.  `typeptr` is stamped by `new` and the
/// codewriter refuses to let a trace observe it as a field at all:
/// `jtransform.py rewrite_op_setfield` drops a `typeptr` store
/// ("ignore the operation completely -- instead, it's done by 'new'"), and
/// `handle_getfield_typeptr` answers a constant container with a
/// `Constant` and any other with `guard_class`.  The Python-level class is a
/// separate word for the same reason `mapdict.py` keeps it in `map`:
/// `__class__ =` rewrites it (`baseobjspace.rs instance_setclass`), so it stays
/// mutable.
///
/// Descriptor retags finish construction before publication:
/// `stamp_method_owners` runs before `new_builtin_typeobject`, and
/// `stamp_new_descr_self` before `w_type_ready`. Module carriers likewise pass
/// through `demote_module_function_to_builtin` before module publication.
/// These writes cannot change an already-observed JIT typeptr.
/// `classmethod_retag_descriptor` changes only the mutable `w_class` word.
#[repr(C)]
#[majit_macros::jit_immutable_fields("ob_type")]
pub struct PyObject {
    pub ob_type: *const PyType,
    pub w_class: *mut PyObject,
}

impl Default for PyObject {
    /// Null header — `Self::allocate` rewrites both fields at malloc time.
    #[inline]
    fn default() -> Self {
        Self {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        }
    }
}

/// The universal Python object reference — a raw pointer to `PyObject`.
///
/// `pyre` currently passes this through the JIT as an integer-sized raw pointer.
/// Uses leaked Box allocations; a proper GC will replace this later.
// Safety: PyType instances are read-only static data, safe to share across threads.
unsafe impl Sync for PyType {}
unsafe impl Send for PyType {}

// Safety: PyObject's ob_type points to immutable static PyType instances.
unsafe impl Sync for PyObject {}
unsafe impl Send for PyObject {}

pub type PyObjectRef = *mut PyObject;

impl crate::rordereddict::EntryDummy for PyObjectRef {
    fn dummy() -> Self {
        std::ptr::null_mut()
    }
}

/// Null object reference, used as a sentinel for "no value".
pub const PY_NULL: PyObjectRef = std::ptr::null_mut();

// ── Type identity ─────────────────────────────────────────────────────

/// Construct a PyType with zeroed subclass ranges.
/// Ranges are assigned at init time by `assign_subclass_range()`.
pub const fn new_pytype(name: &'static str) -> PyType {
    new_pytype_kind(name, 0, 0)
}

/// [`new_pytype`] for a storage class that imported `MapdictStorageMixin`.
/// `mapdict_offset` is the byte offset of the mixin's `map` word;
/// `storage` follows at the next word.
pub const fn new_pytype_with_mapdict_mixin(name: &'static str, mapdict_offset: usize) -> PyType {
    assert!(mapdict_offset != 0);
    new_pytype_kind(name, mapdict_offset, 0)
}

/// [`new_pytype_with_mapdict_mixin`] whose class also carries a
/// `make_weakref_descr` `_lifeline_` field.
pub const fn new_pytype_with_mapdict_and_weakref(
    name: &'static str,
    mapdict_offset: usize,
    weakref_offset: usize,
) -> PyType {
    assert!(mapdict_offset != 0);
    assert!(weakref_offset != 0);
    new_pytype_kind(name, mapdict_offset, weakref_offset)
}

/// [`new_pytype`] for a class whose payload carries a `_lifeline_` field
/// and no `MapdictStorageMixin`.
pub const fn new_pytype_with_weakref(name: &'static str, weakref_offset: usize) -> PyType {
    assert!(weakref_offset != 0);
    new_pytype_kind(name, 0, weakref_offset)
}

const fn new_pytype_kind(
    name: &'static str,
    mapdict_offset: usize,
    weakref_offset: usize,
) -> PyType {
    PyType {
        subclassrange_min: AtomicI64::new(0),
        subclassrange_max: AtomicI64::new(0),
        name,
        instantiate: AtomicPtr::new(std::ptr::null_mut()),
        mapdict_offset,
        weakref_offset,
        user_subclass: std::ptr::null(),
        user_base: std::ptr::null(),
    }
}

/// [`new_pytype`] for a builtin whose user subclass instances carry
/// `user_subclass` as their typeptr (`typedef.py _getusercls`).
pub const fn new_pytype_with_user_subclass(
    name: &'static str,
    user_subclass: &'static PyType,
) -> PyType {
    let mut tp = new_pytype_kind(name, 0, 0);
    tp.user_subclass = user_subclass;
    tp
}

/// [`new_pytype_with_user_subclass`] whose exact instances also carry a
/// `_lifeline_` field. The user typeptr does not inherit that offset:
/// a non-weakrefable typedef keeps `MapdictWeakrefSupport`, and a
/// weakrefable one publishes the offset on the user typeptr itself.
pub const fn new_pytype_with_user_subclass_and_weakref(
    name: &'static str,
    user_subclass: &'static PyType,
    weakref_offset: usize,
) -> PyType {
    assert!(weakref_offset != 0);
    let mut tp = new_pytype_kind(name, 0, weakref_offset);
    tp.user_subclass = user_subclass;
    tp
}

/// The `_getusercls` class made from `base`: it imports
/// `MapdictStorageMixin` after `base`'s payload. `mapdict_offset` is the
/// byte offset of that mixin's `map` word.
pub const fn new_user_pytype(
    name: &'static str,
    base: &'static PyType,
    mapdict_offset: usize,
) -> PyType {
    assert!(mapdict_offset != 0);
    let mut tp = new_pytype_kind(name, mapdict_offset, 0);
    tp.user_base = base;
    tp
}

/// [`new_user_pytype`] for a weakrefable base. `weakref_offset` is the
/// base field's offset; the user struct keeps `base` as a prefix, so the
/// same number addresses the lifeline on a subclass instance.
pub const fn new_user_pytype_with_lifeline(
    name: &'static str,
    base: &'static PyType,
    mapdict_offset: usize,
    weakref_offset: usize,
) -> PyType {
    assert!(mapdict_offset != 0);
    assert!(weakref_offset != 0);
    let mut tp = new_pytype_kind(name, mapdict_offset, weakref_offset);
    tp.user_base = base;
    tp
}

/// Whether `obj`'s typeptr publishes a `_lifeline_` field offset.
///
/// # Safety
/// `obj` must be a live object with a live `ob_type`.
#[inline]
pub unsafe fn has_weakref_lifeline_field(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let tp = unsafe { (*obj).ob_type };
    !tp.is_null() && unsafe { (*tp).weakref_offset } != 0
}

/// Read the `make_weakref_descr` `_lifeline_` slot.
///
/// # Safety
/// `obj` must be a live object whose typeptr has a nonzero `weakref_offset`,
/// and that offset must address a `PyObjectRef` inside `obj`.
#[inline]
pub unsafe fn read_weakref_lifeline(obj: PyObjectRef) -> PyObjectRef {
    let off = unsafe { (*(*obj).ob_type).weakref_offset };
    debug_assert!(off != 0);
    unsafe { *((obj as *const u8).add(off) as *const PyObjectRef) }
}

/// Store `value` into the `_lifeline_` slot.
///
/// `framework.py` `transform_generic_set` emits the write barrier on the
/// receiver, then the bare store.
///
/// # Safety
/// Same as [`read_weakref_lifeline`].
#[inline]
pub unsafe fn write_weakref_lifeline(obj: PyObjectRef, value: PyObjectRef) {
    let off = unsafe { (*(*obj).ob_type).weakref_offset };
    debug_assert!(off != 0);
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    unsafe {
        *((obj as *mut u8).add(off) as *mut PyObjectRef) = value;
    }
}

/// The builtin class whose typedef and payload layout `tp` uses: `tp`'s
/// `user_base` for a `_getusercls` class, `tp` itself otherwise.
///
/// # Safety
/// `tp` must point at a live `PyType`.
#[inline]
pub unsafe fn layout_base(tp: *const PyType) -> *const PyType {
    let base = unsafe { (*tp).user_base };
    if base.is_null() { tp } else { base }
}

/// rclass.py:739-743 parity — cache the W_TypeObject on the PyType
/// so allocators can set `w_class` at allocation time.
///
/// Called by `init_typeobjects()` for each built-in type.
pub fn set_instantiate(tp: &PyType, w_typeobject: PyObjectRef) {
    tp.instantiate.store(w_typeobject, Ordering::Release);
}

/// Read the cached W_TypeObject from a PyType.
///
/// Returns the W_TypeObject (for `w_class`), or null if not yet initialized
/// (bootstrap phase before `init_typeobjects()`).
///
/// `rclass.py` OBJECT_VTABLE marks every field immutable; upstream fills
/// `instantiate` at translation time.  Pyre also writes it from module
/// initializers (`set_instantiate` in mmap/_cffi_backend/posix), which run
/// under the GIL like every reader, so the GIL is the synchronizer and a
/// Relaxed load suffices.  Relaxed keeps the read a plain word load in the
/// generated jitcode — `Acquire` would make the frontend decline the graph
/// and leave every caller with a residual call, keeping `is_plain_int1`
/// off IntegerListStrategy.
#[inline]
pub fn get_instantiate(tp: &PyType) -> PyObjectRef {
    tp.instantiate.load(Ordering::Relaxed)
}

/// Whether `obj`'s storage class imported `MapdictStorageMixin`.
///
/// # Safety
/// `obj` must be a live object with a valid `ob_type`.
#[inline]
pub unsafe fn pytype_has_mapdict_mixin(obj: PyObjectRef) -> bool {
    let tp = unsafe { (*obj).ob_type };
    !tp.is_null() && unsafe { (*tp).mapdict_offset != 0 }
}

/// True when `obj`'s Python class is exactly the builtin type for its
/// layout — i.e. NOT a user subclass.
///
/// A `#[pyre_class(..., user_layout)]` subclass carries that class as its
/// typeptr (`typedef.py` `_getusercls`), which alone decides exactness.
/// A builtin without `user_layout` keeps the builtin `ob_type` (and therefore
/// the builtin struct layout and the `is_list` / … layout predicates) and
/// retags `w_class` (`typedef::subclass_to_tag`).
/// The type-specific
/// fast paths in
/// `space.is_true` / `eq_w` / `len` / `getitem` / … assume the receiver's
/// Python class IS the builtin (no overridable special method); for a
/// subclass instance they would bypass an overridden `__bool__` / `__len__`
/// / `__eq__` / `__getitem__` / … .  Gate each fast path on this predicate
/// and let a subclass fall through to the MRO `lookup` path.
///
/// A fresh builtin carries `w_class == get_instantiate(ob_type)` (see
/// `w_int_new` etc.); the specialised arity-2 tuple layouts instead carry the
/// canonical `tuple` class, exactly as [`is_exact_tuple`] requires.  The
/// read-only singletons (`True` / `False` / `None` / `Ellipsis` /
/// `NotImplemented`) leave `w_class` null and are always exact.
///
/// # Safety
/// `obj` must be null or a valid `PyObjectRef`.
#[inline]
pub unsafe fn is_exact_builtin_instance(obj: PyObjectRef) -> bool {
    // A tagged immediate is an exact builtin `int` (subclasses stay boxed),
    // so it is always an exact builtin instance. Gated on `CAN_BE_TAGGED`
    // (default false), synthesized before the `w_class`/`ob_type` derefs.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return true;
    }
    if obj.is_null() {
        return false;
    }
    if unsafe { typeptr_is_exact_builtin(obj) } {
        return true;
    }
    unsafe { class_word_is_exact_builtin(obj, (*obj).w_class) }
}

/// [`is_exact_builtin_instance`] for a caller that holds a live object, as
/// every `W_Root` argument of the object space is: no null test is made, so
/// none is recorded when the JIT traces the caller.
///
/// # Safety
/// `obj` must be a valid non-null `PyObjectRef`.
#[inline]
pub unsafe fn is_exact_builtin_instance_nonnull(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return true;
    }
    if unsafe { typeptr_is_exact_builtin(obj) } {
        return true;
    }
    unsafe { class_word_is_exact_builtin(obj, (*obj).w_class) }
}

/// A builtin with a `_getusercls` class (`typedef.py`
/// `get_unique_interplevel_subclass`) stamps that class as the typeptr of
/// every user subclass instance, so its own typeptr proves exactness and
/// the class word need not be read.  `false` means "not decided here".
///
/// # Safety
/// `obj` must be a valid non-null, untagged `PyObjectRef`.
#[inline]
pub unsafe fn typeptr_is_exact_builtin(obj: PyObjectRef) -> bool {
    unsafe { !(*(*obj).ob_type).user_subclass.is_null() }
}

/// The tail of [`is_exact_builtin_instance`] for a non-null `obj` whose
/// class word the caller has already read (and may have promoted): a null
/// class word is a read-only singleton and exact; otherwise the class must
/// be the one the object's layout instantiates, with the specialised
/// arity-2 tuple payload types mapping to the canonical `tuple` class.
///
/// # Safety
/// `obj` must be a valid non-null `PyObjectRef` and `w_class` its class word.
#[inline]
pub unsafe fn class_word_is_exact_builtin(obj: PyObjectRef, w_class: PyObjectRef) -> bool {
    if w_class.is_null() {
        return true;
    }
    unsafe {
        let ob_type = (*obj).ob_type;
        use crate::specialisedtupleobject::{
            SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
        };
        let builtin_class = if std::ptr::eq(ob_type, &SPECIALISED_TUPLE_II_TYPE)
            || std::ptr::eq(ob_type, &SPECIALISED_TUPLE_FF_TYPE)
            || std::ptr::eq(ob_type, &SPECIALISED_TUPLE_OO_TYPE)
        {
            get_instantiate(&TUPLE_TYPE)
        } else {
            get_instantiate(&*ob_type)
        };
        std::ptr::eq(w_class, builtin_class)
    }
}

/// `type(obj) is <the builtin type object for `tp`>` — the exact-type test
/// used for pickle dispatch and the `tuple`/`str`/`float` constructors.
///
/// Unlike [`is_exact_builtin_instance`] this is correct for the specialised
/// arity-2 tuples: they carry a distinct `ob_type`
/// (`SPECIALISED_TUPLE_*_TYPE`) but a `w_class` of the canonical `tuple` type
/// object, so `is_exact_type(t, &TUPLE_TYPE)` is `true` for them.  A user
/// subclass retags `w_class` to its own type object and so is rejected.
///
/// # Safety
/// `obj` must be null or a valid `PyObjectRef`; `tp` must be a canonical
/// builtin layout type with `get_instantiate(tp)` initialized.
#[inline]
pub unsafe fn is_exact_type(obj: PyObjectRef, tp: &PyType) -> bool {
    // A tagged immediate is always an exact builtin `int` (never a
    // subclass — those stay boxed via `w_int_new_unique`), so it is the
    // exact `tp` iff `tp` is the `int` vtable. Gated on `CAN_BE_TAGGED`
    // (default false), synthesized before the `w_class`/`ob_type` derefs.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return std::ptr::eq(tp as *const PyType, &INT_TYPE as *const PyType);
    }
    if obj.is_null() {
        return false;
    }
    unsafe {
        // For a builtin with a `_getusercls` class, its own typeptr proves
        // exactness: every user subclass instance carries the user class.
        if !tp.user_subclass.is_null() && std::ptr::eq((*obj).ob_type, tp as *const PyType) {
            return true;
        }
        let w_class = (*obj).w_class;
        if w_class.is_null() {
            std::ptr::eq((*obj).ob_type, tp as *const PyType)
        } else {
            std::ptr::eq(w_class, get_instantiate(tp))
        }
    }
}

// Compile-time verification: AtomicI64/AtomicPtr are layout-compatible
// with i64/*mut T so the JIT can read PyType fields at raw offsets.
// Also verify OBJECT_VTABLE field order: subclassrange_min @ 0, max @ 8.
const _: () = {
    assert!(std::mem::size_of::<AtomicI64>() == std::mem::size_of::<i64>());
    assert!(std::mem::align_of::<AtomicI64>() == std::mem::align_of::<i64>());
    assert!(std::mem::size_of::<AtomicPtr<PyObject>>() == std::mem::size_of::<*mut PyObject>());
    assert!(std::mem::offset_of!(PyType, subclassrange_min) == 0);
    assert!(std::mem::offset_of!(PyType, subclassrange_max) == 8);
    // `instantiate` must sit immediately after `name` with no padding, so
    // its offset is exactly `offset_of(name) + size_of::<&str>()`: 32 on a
    // 64-bit JIT host (`&str` is a 16-byte fat pointer), 24 on 32-bit
    // targets such as wasm32 (`&str` is 8 bytes). The JIT reads this slot
    // at a raw offset via the exact Charon struct layout on the 64-bit
    // host; this assert pins the no-padding invariant on every target.
    assert!(
        std::mem::offset_of!(PyType, instantiate)
            == std::mem::offset_of!(PyType, name) + std::mem::size_of::<&'static str>()
    );
};

pub static INT_TYPE: PyType = new_pytype_with_user_subclass("int", &INT_USER_TYPE);
/// `W_IntObjectUser` (`typedef.py _getusercls(W_IntObject)`).
pub static INT_USER_TYPE: PyType = new_user_pytype(
    "int",
    &INT_TYPE,
    std::mem::offset_of!(crate::intobject::W_IntObjectUser, map),
);
pub static BOOL_TYPE: PyType = new_pytype("bool");
pub static FLOAT_TYPE: PyType = new_pytype_with_user_subclass("float", &FLOAT_USER_TYPE);
/// `W_FloatObjectUser` (`typedef.py _getusercls(W_FloatObject)`).
pub static FLOAT_USER_TYPE: PyType = new_user_pytype(
    "float",
    &FLOAT_TYPE,
    std::mem::offset_of!(crate::floatobject::W_FloatObjectUser, map),
);
pub static COMPLEX_TYPE: PyType = new_pytype_with_user_subclass("complex", &COMPLEX_USER_TYPE);
/// `W_ComplexObjectUser` (`typedef.py _getusercls(W_ComplexObject)`).
pub static COMPLEX_USER_TYPE: PyType = new_user_pytype(
    "complex",
    &COMPLEX_TYPE,
    std::mem::offset_of!(crate::complexobject::W_ComplexObjectUser, map),
);
pub static STR_TYPE: PyType = new_pytype_with_user_subclass("str", &STR_USER_TYPE);
/// `W_UnicodeObjectUser` (`typedef.py _getusercls(W_UnicodeObject)`).
pub static STR_USER_TYPE: PyType = new_user_pytype(
    "str",
    &STR_TYPE,
    std::mem::offset_of!(crate::unicodeobject::W_UnicodeObjectUser, map),
);
pub static LIST_TYPE: PyType = new_pytype_with_user_subclass("list", &LIST_USER_TYPE);
/// `W_ListObjectUser` (`typedef.py _getusercls(W_ListObject)`).
pub static LIST_USER_TYPE: PyType = new_user_pytype(
    "list",
    &LIST_TYPE,
    std::mem::offset_of!(crate::listobject::W_ListObjectUser, map),
);
pub static TUPLE_TYPE: PyType = new_pytype_with_user_subclass("tuple", &TUPLE_USER_TYPE);
/// `W_TupleObjectUser` (`typedef.py _getusercls(W_TupleObject)`).
pub static TUPLE_USER_TYPE: PyType = new_user_pytype(
    "tuple",
    &TUPLE_TYPE,
    std::mem::offset_of!(crate::tupleobject::W_TupleObjectUser, map),
);
pub static DICT_TYPE: PyType = new_pytype("dict");
pub static LONG_TYPE: PyType = new_pytype_with_user_subclass("int", &LONG_USER_TYPE);
/// `W_LongObjectUser` (`typedef.py` `_getusercls(W_LongObject)`).
pub static LONG_USER_TYPE: PyType = new_user_pytype(
    "int",
    &LONG_TYPE,
    std::mem::offset_of!(crate::longobject::W_LongObjectUser, map),
);
pub static NONE_TYPE: PyType = new_pytype("NoneType");
pub static NOTIMPLEMENTED_TYPE: PyType = new_pytype("NotImplementedType");
pub static ELLIPSIS_TYPE: PyType = new_pytype("ellipsis");
pub static MODULE_TYPE: PyType = new_pytype_with_user_subclass_and_weakref(
    "module",
    &MODULE_USER_TYPE,
    std::mem::offset_of!(crate::module::Module, lifeline),
);
/// `ModuleUser` (`typedef.py` `_getusercls(Module)`). The lifeline lives on
/// the `Module` prefix, so a subclass instance addresses the same word.
pub static MODULE_USER_TYPE: PyType = new_user_pytype_with_lifeline(
    "module",
    &MODULE_TYPE,
    std::mem::offset_of!(crate::module::ModuleUser, map),
    std::mem::offset_of!(crate::module::Module, lifeline),
);
pub static MAPPING_PROXY_TYPE: PyType = new_pytype("mappingproxy");
pub static TYPE_TYPE: PyType = new_pytype_with_weakref(
    "type",
    std::mem::offset_of!(crate::typeobject::W_TypeObject, lifeline),
);
/// `W_Root` (`baseobjspace.py`). The rclass root has no app-level type
/// and no instances: `instantiate` stays null, `user_subclass` stays
/// null, and nothing stamps this vtable into `ob_type`.
pub static W_ROOT_TYPE: PyType = new_pytype("W_Root");
/// `W_ObjectObject` (`objectobject.py`). Exact `object()`. One child of
/// `W_ROOT_TYPE`, not the rclass root. `user_subclass` is the
/// `_getusercls` class below; both keep the `map` word
/// (`MapdictStorageMixin`).
pub static INSTANCE_TYPE: PyType = {
    let mut tp = new_pytype_with_mapdict_mixin(
        "object",
        std::mem::offset_of!(crate::objectobject::W_ObjectObject, map),
    );
    tp.user_subclass = &INSTANCE_USER_TYPE;
    tp
};
/// `W_ObjectObjectUserDictWeakrefable` (`typedef.py`
/// `_getusercls(W_ObjectObject)`). Same `{header, map, storage}` payload
/// as `W_ObjectObject`; a carrier whose app-level class is not exactly
/// `object` stamps this vtable.
pub static INSTANCE_USER_TYPE: PyType = new_user_pytype(
    "object",
    &INSTANCE_TYPE,
    std::mem::offset_of!(crate::objectobject::W_ObjectObject, map),
);

/// Field offset of `ob_type` within PyObject, for JIT field access.
pub const OB_TYPE_OFFSET: usize = std::mem::offset_of!(PyObject, ob_type);

/// Field offset of `w_class` within PyObject, for JIT field access.
/// RPython: this corresponds to reading typeptr + gettypefor (fused into one field).
pub const W_CLASS_OFFSET: usize = std::mem::offset_of!(PyObject, w_class);

/// Watcher set for `__class__` assignment (`notify_w_class_mutated`).
///
/// Instance `w_class` is an ordinary mutable field (`mapdict.py`
/// `BaseUserClassMapdict.setclass`); the JIT descr is not `?`.
/// `walker_pin_type_attr_binding` still registers on this watcher:
/// `TypeAttrBinding::UnboundHeap` records `record_quasiimmut_field`
/// against `w_class_descr()`, which installs via `w_class_current_qmut`.
/// Process-wide layout owner for header quasi-immuts. One space, one
/// watcher set — a field of this prebuilt, not a naked module static.
struct ObjectLayout {
    w_class_watchers: crate::quasiimmut::QuasiImmutField,
}

static OBJECT_LAYOUT: ObjectLayout = ObjectLayout {
    w_class_watchers: crate::quasiimmut::QuasiImmutField::new(),
};

/// `quasiimmut.py get_current_qmut_instance` for `PyObject.w_class`.
pub fn w_class_current_qmut() -> std::sync::Arc<crate::quasiimmut::QuasiImmut> {
    OBJECT_LAYOUT.w_class_watchers.get_current_qmut_instance()
}

/// Invalidate loops that folded a `w_class` read. Call after a published
/// object's class changes (`descr_set___class__`, exception retag).
#[inline]
pub fn notify_w_class_mutated() {
    if OBJECT_LAYOUT.w_class_watchers.is_installed() {
        OBJECT_LAYOUT.w_class_watchers.invalidate();
    }
}

/// Unlink the `w_class` watcher and publish `store` under the same lock
/// so a tracer cannot fold the old class onto a freshly installed watcher.
#[inline]
pub fn notify_w_class_mutated_then(store: impl FnOnce()) {
    OBJECT_LAYOUT.w_class_watchers.invalidate_then_store(store);
}

/// Field offset of `subclassrange_min` within PyType (OBJECT_VTABLE).
/// rclass.py — first field in OBJECT_VTABLE.
pub const SUBCLASSRANGE_MIN_OFFSET: usize = std::mem::offset_of!(PyType, subclassrange_min);

/// Field offset of `subclassrange_max` within PyType (OBJECT_VTABLE).
/// rclass.py — second field in OBJECT_VTABLE.
pub const SUBCLASSRANGE_MAX_OFFSET: usize = std::mem::offset_of!(PyType, subclassrange_max);

/// Field offset of `instantiate` within PyType (OBJECT_VTABLE).
/// rclass.py — `('instantiate', Ptr(FuncType([], OBJECTPTR)))`.
/// 32 on a 64-bit host (`name` is a 16-byte fat pointer); 24 on 32-bit
/// targets where `&str` is 8 bytes.
pub const INSTANTIATE_OFFSET: usize = std::mem::offset_of!(PyType, instantiate);

/// rclass.py `ll_cast_to_object(obj)`.
///
/// In RPython this casts a typed pointer to `OBJECTPTR`. In pyre all
/// objects are already `PyObjectRef`, so this is an identity function
/// kept for structural parity.
#[inline]
pub fn ll_cast_to_object(obj: PyObjectRef) -> PyObjectRef {
    obj
}

/// rclass.py `ll_type(obj)`.
///
/// Extract the type pointer (CLASSTYPE) from an object.
///
/// # Safety
/// `obj` must be a valid non-null `PyObject`.
#[inline]
pub unsafe fn ll_type(obj: PyObjectRef) -> *const PyType {
    // `ll_unboxed_getclass`: a tagged immediate's class is the `int`
    // vtable, synthesized before the `ob_type` deref. Gated on the
    // `CAN_BE_TAGGED` static (default false), so the deref is the only
    // live path until enablement.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return &INT_TYPE as *const PyType;
    }
    unsafe { (*obj).ob_type }
}

/// rclass.py `ll_issubclass(subcls, cls)`.
///
/// O(1) subclass check via preorder numbering:
///   `int_between(cls.subclassrange_min, subcls.subclassrange_min, cls.subclassrange_max)`
/// The subclass ranges are stamped once during startup and never change afterwards,
/// as declared by `PyType`'s `jit_immutable_fields`, so the result depends only on
/// the arguments and the call cannot raise.
///
/// # Safety
/// Both pointers must be non-null pointers to static `PyType`s whose startup
/// publication has completed before the calling thread begins reading them.
#[majit_macros::elidable_cannot_raise]
#[inline]
pub unsafe fn ll_issubclass(subcls: *const PyType, cls: *const PyType) -> bool {
    let subcls = unsafe { &*subcls };
    let cls = unsafe { &*cls };
    // rclass.py ll_issubclass: startup has published the immutable vtables
    // before any reader runs. No runtime writer can restamp their numbering.
    let cls_min = cls.subclassrange_min.load(Ordering::Relaxed);
    let subcls_min = subcls.subclassrange_min.load(Ordering::Relaxed);
    let cls_max = cls.subclassrange_max.load(Ordering::Relaxed);
    // int_between(a, b, c) ≡ a <= b < c
    cls_min <= subcls_min && subcls_min < cls_max
}

/// rclass.py `ll_issubclass_const(subcls, minid, maxid)`.
///
/// Variant of `ll_issubclass` where the class bounds are already known
/// constants. Used by the JIT when the target class is constant-folded.
/// The caller must have completed vtable startup publication.
#[inline]
pub fn ll_issubclass_const(subcls: &PyType, minid: i64, maxid: i64) -> bool {
    let subcls_min = subcls.subclassrange_min.load(Ordering::Relaxed);
    // int_between(a, b, c) ≡ a <= b < c
    minid <= subcls_min && subcls_min < maxid
}

/// rclass.py `ll_isinstance(obj, cls)`.
///
/// RPython-level type check: reads `obj.typeptr` (= `ob_type`) and checks
/// subclass ranges. This checks the **RPython class** (`W_IntObject`,
/// `W_ListObject`, ...), NOT the Python-level class. The rclass root is
/// `W_ROOT_TYPE` (`W_Root`). Exact `object()` is `INSTANCE_TYPE`
/// (`W_ObjectObject`). Every other carrier of that struct is
/// `INSTANCE_USER_TYPE` (`typedef.py` `_getusercls(W_ObjectObject)`),
/// a child of that vtable.
///
/// For Python-level `isinstance()`, use `issubtype_w` (MRO walk on
/// `w_class`), not this function.
///
/// # Safety
/// `obj` must be a valid non-null `PyObject`.
#[inline]
pub unsafe fn ll_isinstance(obj: PyObjectRef, cls: &PyType) -> bool {
    // `ll_unboxed_isinstance`: a tagged immediate's RPython class is the
    // `int` vtable, checked against `cls`'s subclass range without the
    // `ob_type` deref. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return unsafe { ll_issubclass(&INT_TYPE, cls) };
    }
    if obj.is_null() {
        return false;
    }
    // A collected or not-yet-headered object can sit in a walker's
    // concrete shadow with a null `ob_type`. `py_type_check` compares
    // that pointer and returns false; this function used to form
    // `&*null` and SIGSEGV (`abs(x-y)` on complexes during
    // `try_walker_inline_exception_string_override`).
    let obj_cls_ptr = unsafe { (*obj).ob_type };
    if obj_cls_ptr.is_null() {
        return false;
    }
    let obj_cls = unsafe { &*obj_cls_ptr };
    unsafe { ll_issubclass(obj_cls, cls) }
}

/// rclass.py `ll_inst_type(obj)`.
///
/// Return the typeptr if obj is non-null, null otherwise.
///
/// # Safety
/// If non-null, `obj` must be a valid `PyObject`.
#[inline]
pub unsafe fn ll_inst_type(obj: PyObjectRef) -> *const PyType {
    // `ll_unboxed_getclass_canbenone`: a tagged immediate has the low
    // bit set and is therefore non-null, so the `int`-vtable synth
    // precedes the null check. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return &INT_TYPE as *const PyType;
    }
    unsafe {
        if !obj.is_null() {
            (*obj).ob_type
        } else {
            std::ptr::null()
        }
    }
}

/// Write subclass ranges to a `PyType` instance.
///
/// Mirrors `assign_inheritance_ids` (normalizecalls.py) which
/// assigns `classdef.minid` / `classdef.maxid` to each vtable entry.
///
/// Uses `Relaxed` ordering: ranges are written once at init time
/// before any concurrent reads.
fn assign_subclass_range(tp: &PyType, min: i64, max: i64) {
    tp.subclassrange_min.store(min, Ordering::Relaxed);
    tp.subclassrange_max.store(max, Ordering::Relaxed);
}

/// One static `PyType` alias for an `rclass.OBJECT` typeid.
///
/// Several PyTypes can share one GC typeid (for example `set` and
/// `frozenset`), while some GC typeids contribute an inheritance peer but
/// have no vtable alias. Keeping aliases separate from the hierarchy
/// preserves that exact GC registration shape.
#[derive(Clone, Copy)]
pub struct SubclassRangeAlias {
    pub type_id: u32,
    pub pytype: &'static PyType,
}

pub const fn subclass_range_alias(type_id: u32, pytype: &'static PyType) -> SubclassRangeAlias {
    SubclassRangeAlias { type_id, pytype }
}

/// Canonical `rclass.OBJECT` inheritance census in GC registration order.
///
/// Each entry is `(typeid, parent_typeid)`. This is the shared input for the
/// interpreter-side fallback and is checked against the GC's `TypeRegistry`
/// before `freeze_types`. Sparse non-object typeids are intentionally absent.
/// The order and parent links mirror the `TypeInfo::object{,_subclass}` calls
/// in `pyre-jit/src/eval.rs`.
pub const SUBCLASS_RANGE_HIERARCHY: &[(u32, Option<u32>)] = &[
    (0, None),
    (1, Some(0)),
    (2, Some(0)),
    (5, Some(1)),
    (6, Some(0)),
    (7, Some(0)),
    (8, Some(0)),
    (10, Some(0)),
    (11, Some(0)),
    (12, Some(0)),
    (13, Some(0)),
    (14, Some(0)),
    (15, Some(0)),
    (16, Some(0)),
    (17, Some(0)),
    (18, Some(0)),
    (19, Some(0)),
    (20, Some(0)),
    (21, Some(0)),
    (22, Some(0)),
    (23, Some(0)),
    (24, Some(0)),
    (25, Some(0)),
    (26, Some(0)),
    (27, Some(0)),
    (28, Some(0)),
    (29, Some(0)),
    (30, Some(0)),
    (31, Some(0)),
    (32, Some(0)),
    (33, Some(0)),
    (34, Some(0)),
    (35, Some(0)),
    (36, Some(0)),
    // `pypy/interpreter/pyframe.py:PyFrame(W_Root)`.
    (37, Some(0)),
    (38, Some(0)),
    (39, Some(0)),
    (40, Some(0)),
    (43, Some(0)),
    (44, Some(0)),
    (45, Some(0)),
    (46, Some(0)),
    (47, Some(0)),
    (48, Some(0)),
    (49, Some(0)),
    (50, Some(0)),
    (52, Some(0)),
    (53, Some(0)),
    (54, Some(0)),
    (56, Some(31)),
    (57, Some(31)),
    (58, Some(31)),
    (59, Some(56)),
    (60, Some(59)),
    (61, Some(59)),
    (62, Some(56)),
    (63, Some(56)),
    (64, Some(63)),
    (65, Some(64)),
    (66, Some(64)),
    (67, Some(64)),
    (68, Some(56)),
    (69, Some(56)),
    (70, Some(69)),
    (71, Some(69)),
    (72, Some(56)),
    (73, Some(56)),
    (74, Some(73)),
    (75, Some(73)),
    (76, Some(56)),
    (77, Some(56)),
    (78, Some(56)),
    (79, Some(56)),
    (80, Some(56)),
    (81, Some(80)),
    (82, Some(56)),
    (83, Some(56)),
    (84, Some(0)),
    (85, Some(0)),
    (86, Some(0)),
    (87, Some(0)),
    (88, Some(0)),
    (89, Some(0)),
    (90, Some(0)),
    (91, Some(0)),
    (92, Some(0)),
    (93, Some(0)),
    (94, Some(0)),
    (95, Some(0)),
    (96, Some(0)),
    (99, Some(0)),
    (100, Some(0)),
    (101, Some(0)),
    (102, Some(0)),
    (103, Some(0)),
    (104, Some(0)),
    (105, Some(0)),
    (106, Some(0)),
    (107, Some(0)),
    (108, Some(0)),
    (109, Some(0)),
    (110, Some(0)),
    (111, Some(0)),
    (112, Some(0)),
    (113, Some(0)),
    (114, Some(0)),
    (115, Some(0)),
    (116, Some(0)),
    (117, Some(0)),
    (118, Some(0)),
    (119, Some(0)),
    (120, Some(0)),
    (121, Some(0)),
    (122, Some(0)),
    (123, Some(0)),
    (124, Some(0)),
    (125, Some(0)),
    (126, Some(0)),
    (127, Some(0)),
    (128, Some(0)),
    (129, Some(0)),
    (130, Some(0)),
    (131, Some(0)),
    (132, Some(0)),
    (133, Some(0)),
    (134, Some(0)),
    (135, Some(0)),
    (136, Some(0)),
    (137, Some(0)),
    (138, Some(0)),
    (139, Some(0)),
    (140, Some(0)),
    (141, Some(0)),
    (142, Some(0)),
    (143, Some(0)),
    (145, Some(0)),
    // `_thread` lock / RLock / handle, registered for the header `w_class`
    // edge (`all_w_class_only_descriptors`).
    (146, Some(0)),
    (147, Some(0)),
    (148, Some(0)),
    // `__pypy__.Bufferable` is `allocate_stable` with no inline object
    // payload, so the header `w_class` is the only edge its marker forwards.
    (149, Some(0)),
    // `_io.BytesIO` follows the `rbigint` result pair, which holds 150 as a
    // bare `with_gc_ptrs` id and is not an rclass.OBJECT type.
    (151, Some(0)),
    // `_io.StringIO` follows `_io.BytesIO`.
    (152, Some(0)),
    // The three walks over a code object.
    (153, Some(0)),
    (154, Some(0)),
    (155, Some(0)),
    // The two `step == 1` range-iterator shapes, whose ids are explicit.
    (156, Some(0)),
    (157, Some(0)),
    // 158-167 and 169-195 are `typedef.py` `_getusercls` layouts
    // (int/str/tuple/float/complex/bytes/bytearray/list/set/array,
    // plus enumerate/map/filter/zip/reversed/super/property, the itertools
    // user layouts, and `__pypy__.Bufferable`).
    // 168 is `interp__weakref.py` `W_Weakref`, an object subclass.
    // 169-195 parent on the builtin (`typedef.py` `_getusercls` `class subcls(cls)`).
    // 196-213 append deque, Struct, GenericAlias, big-int, weakref,
    // staticmethod, classmethod, module, `_thread._local` and typed `_io`
    // user layouts without moving the closed block above.
    (158, Some(1)),
    (159, Some(34)),
    (160, Some(8)),
    (161, Some(2)),
    (162, Some(54)),
    (163, Some(27)),
    (164, Some(28)),
    (165, Some(7)),
    (166, Some(30)),
    (167, Some(94)),
    (168, Some(0)), // W_Weakref
    (169, Some(111)),
    (170, Some(91)),
    (171, Some(90)),
    (172, Some(92)),
    (173, Some(89)),
    (174, Some(18)),
    (175, Some(19)),
    (176, Some(24)),
    (177, Some(25)),
    (178, Some(114)),
    (179, Some(115)),
    (180, Some(116)),
    (181, Some(134)),
    (182, Some(135)),
    (183, Some(136)),
    (184, Some(137)),
    (185, Some(138)),
    (186, Some(139)),
    (187, Some(140)),
    (188, Some(103)),
    (189, Some(104)),
    (190, Some(105)),
    (191, Some(106)),
    (192, Some(117)),
    (193, Some(93)),
    (194, Some(95)),
    (195, Some(149)),
    // Appended `_getusercls` layouts. Fixed ids 0-195 stay put.
    (196, Some(110)), // collections.deque
    (197, Some(119)), // _struct.Struct
    (198, Some(87)),  // types.GenericAlias
    (199, Some(35)),  // W_LongObject
    (200, Some(168)), // W_WeakrefUser
    (201, Some(20)),  // StaticMethodUser
    (202, Some(21)),  // ClassMethodUser
    (203, Some(36)),  // ModuleUser
    (204, Some(133)), // W_LocalUser
    (205, Some(151)), // W_BytesIOUser
    (206, Some(152)), // W_StringIOUser
    (207, Some(128)), // W_BufferedReaderUser
    (208, Some(129)), // W_BufferedWriterUser
    (209, Some(130)), // W_BufferedRWPairUser
    (210, Some(131)), // W_BufferedRandomUser
    (211, Some(132)), // W_TextIOWrapperUser
    (212, Some(0)),   // W_FileIO
    (213, Some(212)), // W_FileIOUser
    // Extra-field exception payload (`W_ExceptionExtended`). No PyType alias:
    // exact realbases keep their class vtables, and `_getusercls` instances
    // use the user layouts below.
    (214, Some(31)),
    (215, Some(31)),  // W_BaseExceptionUser
    (216, Some(214)), // W_ExceptionExtendedUser
    // `W_ObjectObjectUserDictWeakrefable` (`typedef.py`
    // `_getusercls(W_ObjectObject)`). Same payload as
    // `W_OBJECT_OBJECT_GC_TYPE_ID`.
    (217, Some(53)),
    // `_sre.SRE_Template` — registered immediately before the cfg-gated
    // posix / console tail, after the `_getusercls` layouts (158-217).
    (218, Some(0)),
    // Native-only type IDs 219 and 220 represent `posix.DirEntry` and
    // `posix.ScandirIterator`, matching `build_gc`'s registration order.
    #[cfg(not(target_arch = "wasm32"))]
    (219, Some(0)),
    #[cfg(not(target_arch = "wasm32"))]
    (220, Some(0)),
    // PEP 528 `_io._WindowsConsoleIO` is a subclassable `_RawIOBase` payload
    // and closes the interpreter's classes. `pyre-interpreter` drops it where
    // it compiles the class out.
    #[cfg(windows)]
    (221, Some(0)),
    #[cfg(windows)]
    (222, Some(221)), // W_WinConsoleIOUser
                      // The classes `pyre-module` registers follow, numbered by `build_gc` in
                      // the order the module hooks list them; `pyre-interpreter` appends them.
];

/// Compute subclass IDs from the active hierarchy and write every
/// supplied PyType alias via `assign_subclass_range`.
///
/// This mirrors RPython `TotalOrderSymbolic.compute_fn`
/// (`normalizecalls.py`): build each reversed-MRO witness, add its
/// Min and `witness + [MAX]` peers, lexicographically sort all peers, then
/// assign their 0-based `enumerate()` positions. The root Min peer is 0.
///
/// Private writer called only inside the startup publication gate, before
/// any `is_exception` / `ll_isinstance` reader. Interpreter-only and GC-first
/// startup share that gate; GC verifies the result without writing it back.
fn compute_subclass_ranges_from_hierarchy(
    hierarchy: &[(u32, Option<u32>)],
    alias_chains: &[&[SubclassRangeAlias]],
) {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum WitnessElement {
        Cdef(u32),
        Max,
    }

    let slots = hierarchy
        .last()
        .map_or(0, |(type_id, _)| *type_id as usize + 1);
    let mut witnesses: Vec<Option<Vec<WitnessElement>>> = vec![None; slots];
    for &(type_id, parent) in hierarchy {
        let mut witness = match parent {
            Some(parent_id) => witnesses[parent_id as usize]
                .clone()
                .expect("subclass-range parent must precede its child"),
            None => Vec::new(),
        };
        witness.push(WitnessElement::Cdef(type_id));
        witnesses[type_id as usize] = Some(witness);
    }

    #[derive(Clone)]
    struct Peer {
        witness: Vec<WitnessElement>,
        owner: u32,
        is_max: bool,
    }

    let mut peers = Vec::with_capacity(hierarchy.len() * 2);
    for &(type_id, _) in hierarchy {
        let witness = witnesses[type_id as usize]
            .as_ref()
            .expect("every subclass-range typeid must have a witness");
        peers.push(Peer {
            witness: witness.clone(),
            owner: type_id,
            is_max: false,
        });
        let mut max_witness = witness.clone();
        max_witness.push(WitnessElement::Max);
        peers.push(Peer {
            witness: max_witness,
            owner: type_id,
            is_max: true,
        });
    }
    peers.sort_by(|a, b| a.witness.cmp(&b.witness));

    let mut ranges = vec![(0, 0); slots];
    for (value, peer) in peers.iter().enumerate() {
        let range = &mut ranges[peer.owner as usize];
        if peer.is_max {
            range.1 = value as i64;
        } else {
            range.0 = value as i64;
        }
    }

    // Only the publication OnceLock calls this writer. Readers start after
    // that gate completes; GC reconstruction never writes these fields.
    for aliases in alias_chains {
        for alias in *aliases {
            let range = ranges
                .get(alias.type_id as usize)
                .copied()
                .expect("subclass-range alias typeid must be in the hierarchy");
            assert!(
                witnesses[alias.type_id as usize].is_some(),
                "subclass-range alias typeid must name an rclass.OBJECT node"
            );
            assign_subclass_range(alias.pytype, range.0, range.1);
        }
    }
}

/// Publication gate shared by full interpreter startup and the standalone
/// object unit-test initializer. There is no object-only runtime fallback:
/// production readers follow `typedef::init_subclass_ranges`, which supplies
/// the active hierarchy and both crates' aliases. GC verifies that publication.
static SUBCLASS_RANGES_INIT: OnceLock<()> = OnceLock::new();

/// Standalone object tests have no interpreter startup. Seed their vtables
/// explicitly, outside `is_exception`, just as rclass.py `ll_isinstance`
/// reads values prebuilt by `ClassRepr.fill_vtable_root` without a lazy call.
#[cfg(test)]
pub(crate) fn ensure_object_subclass_ranges_initialized() {
    let aliases = all_subclass_range_aliases();
    initialize_subclass_ranges_from_hierarchy(SUBCLASS_RANGE_HIERARCHY, &[&aliases]);
}

/// Publish all vtable ranges once, before starting any readers. Every caller
/// in a process must supply the same complete hierarchy and alias census.
/// Standalone object unit tests have their own process and seed only that
/// binary's aliases; partial-to-full extension is deliberately unsupported.
///
/// Startup adaptation of rclass.py `ClassRepr.fill_vtable_root`: the OnceLock
/// publishes the entire batch, and repeated/concurrent startup calls wait for
/// it without ever restamping a live vtable. Raw writer helpers are private.
pub fn initialize_subclass_ranges_from_hierarchy(
    hierarchy: &[(u32, Option<u32>)],
    alias_chains: &[&[SubclassRangeAlias]],
) {
    SUBCLASS_RANGES_INIT.get_or_init(|| {
        compute_subclass_ranges_from_hierarchy(hierarchy, alias_chains);
    });
}

#[cfg(test)]
mod subclass_range_publication_tests {
    use super::*;

    const CHILD_MODE: &str = "PYRE_SUBCLASS_PUBLICATION_TEST_ORDER";

    #[test]
    fn initialization_orders_use_one_gate() {
        // Each binary has one complete census, never a partial-to-full update.
        for mode in ["full", "object", "concurrent"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "pyobject::subclass_range_publication_tests::publication_child",
                    "--nocapture",
                ])
                .env(CHILD_MODE, mode)
                .output()
                .expect("run isolated publication");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{mode}: {stdout}{}",
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    #[test]
    fn publication_child() {
        let Ok(mode) = std::env::var(CHILD_MODE) else {
            return; // Invoked by initialization_orders_use_one_gate.
        };
        assert!(SUBCLASS_RANGES_INIT.get().is_none());
        static EXTRA_ALIAS: PyType = new_pytype("extra_publication_alias");
        let object_aliases = all_subclass_range_aliases();
        let extra = [subclass_range_alias(0, &EXTRA_ALIAS)];
        if mode == "object" {
            ensure_object_subclass_ranges_initialized();
            ensure_object_subclass_ranges_initialized();
            assert!(unsafe { ll_issubclass(&BOOL_TYPE, &INT_TYPE) });
            assert!(unsafe { ll_issubclass(&INSTANCE_TYPE, &W_ROOT_TYPE) });
            assert!(unsafe { ll_issubclass(&INT_TYPE, &W_ROOT_TYPE) });
            assert!(unsafe { ll_issubclass(&LIST_TYPE, &W_ROOT_TYPE) });
            assert!(!unsafe { ll_issubclass(&INT_TYPE, &INSTANCE_TYPE) });
            assert!(!unsafe { ll_issubclass(&LIST_TYPE, &INSTANCE_TYPE) });
        } else {
            // Omit one interpreter-only class that nothing else parents on:
            // the last such id. That is the posix tail here, the Windows
            // console where it is compiled in, and `_sre.SRE_Template` on
            // wasm32 (the object-crate user layouts stay in the table).
            let omitted_index = SUBCLASS_RANGE_HIERARCHY
                .iter()
                .rposition(|(id, _)| {
                    object_aliases.iter().all(|alias| alias.type_id != *id)
                        && SUBCLASS_RANGE_HIERARCHY
                            .iter()
                            .all(|(_, parent)| *parent != Some(*id))
                })
                .expect("an interpreter-only class with no children");
            let omitted = SUBCLASS_RANGE_HIERARCHY[omitted_index].0;
            assert!(object_aliases.iter().all(|alias| alias.type_id != omitted));
            let hierarchy: Vec<(u32, Option<u32>)> = SUBCLASS_RANGE_HIERARCHY
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != omitted_index)
                .map(|(_, entry)| *entry)
                .collect();
            let initialize_and_read = || {
                initialize_subclass_ranges_from_hierarchy(&hierarchy, &[&object_aliases, &extra]);
                assert_eq!(
                    EXTRA_ALIAS.subclassrange_max.load(Ordering::Relaxed),
                    (hierarchy.len() * 2 - 1) as i64,
                );
                assert!(unsafe { ll_issubclass(&W_ROOT_TYPE, &EXTRA_ALIAS) });
                assert!(unsafe { ll_issubclass(&INSTANCE_TYPE, &W_ROOT_TYPE) });
                assert!(unsafe { ll_issubclass(&INT_TYPE, &W_ROOT_TYPE) });
                assert!(unsafe { ll_issubclass(&LIST_TYPE, &W_ROOT_TYPE) });
                assert!(!unsafe { ll_issubclass(&INT_TYPE, &INSTANCE_TYPE) });
                assert!(!unsafe { ll_issubclass(&LIST_TYPE, &INSTANCE_TYPE) });
                assert!(ll_issubclass_const(
                    &EXTRA_ALIAS,
                    0,
                    (hierarchy.len() * 2 - 1) as i64
                ));
            };
            if mode == "concurrent" {
                let start = std::sync::Barrier::new(4);
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        scope.spawn(|| {
                            start.wait();
                            initialize_and_read();
                        });
                    }
                });
            } else {
                assert_eq!(mode, "full");
                initialize_and_read();
            }
            initialize_and_read();
        }
        assert!(SUBCLASS_RANGES_INIT.get().is_some());
        let before = INSTANCE_TYPE.subclassrange_max.load(Ordering::Relaxed);
        // Deliberately invalid input after publication: if the private writer
        // runs again, the alias has no hierarchy entry and this call panics.
        // Unlike a proxy counter, this detects actual recomputation.
        initialize_subclass_ranges_from_hierarchy(&[], &[&extra]);
        assert_eq!(
            INSTANCE_TYPE.subclassrange_max.load(Ordering::Relaxed),
            before
        );
    }
}

/// Every built-in `PyType` static that represents a full `PyObject`
/// subtype (i.e. instances carry `ob_type` at offset 0, matching
/// `rclass.OBJECT` layout), paired with its parent class.
///
/// Modelled on RPython's `assign_inheritance_ids`
/// (normalizecalls.py) which walks `classdef.getmro()` to build
/// the reversed-MRO witness for each class. The JIT registers each
/// `(type, parent)` pair with the GC via `register_vtable_for_type`,
/// using the parent typeid as `TypeInfo::object_subclass`'s `parent`
/// argument so the resulting `subclassrange_{min,max}` faithfully
/// represents the `rclass.OBJECT` hierarchy. `GUARD_SUBCLASS` then
/// resolves to `int_between(cls.min, subcls.min, cls.max)` per
/// rclass.py `ll_issubclass`.
///
/// `W_ROOT_TYPE` (`W_Root`) is intentionally absent: it is registered
/// separately as the `rclass.OBJECT` root with no parent. Every parent
/// below that means "child of the rclass root" names `W_ROOT_TYPE`.
/// Nothing here is a child of `W_ObjectObject`. `INSTANCE_TYPE` is also
/// absent: its payload is `W_ObjectObject`, registered at
/// `W_OBJECT_OBJECT_GC_TYPE_ID`. `INT_TYPE` and `FLOAT_TYPE` are absent
/// for the same reason (`W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID`):
/// NewWithVtable needs the real payload size.
pub fn all_foreign_pytypes() -> &'static [(&'static PyType, &'static PyType)] {
    static PYTYPES: &[(&PyType, &PyType)] = &[
        // bool inherits from int (objectobject.py W_BoolObject.typedef).
        (&BOOL_TYPE, &INT_TYPE),
        (&STR_TYPE, &W_ROOT_TYPE),
        (&LIST_TYPE, &W_ROOT_TYPE),
        (&TUPLE_TYPE, &W_ROOT_TYPE),
        (&DICT_TYPE, &W_ROOT_TYPE),
        // longobject.py W_LongObject — Python 3 unifies long under int,
        // but pyre carries a separate static for the BigInt-backed flavour.
        (&LONG_TYPE, &W_ROOT_TYPE),
        (&NONE_TYPE, &W_ROOT_TYPE),
        (&NOTIMPLEMENTED_TYPE, &W_ROOT_TYPE),
        (&ELLIPSIS_TYPE, &W_ROOT_TYPE),
        (&MODULE_TYPE, &W_ROOT_TYPE),
        (&MAPPING_PROXY_TYPE, &W_ROOT_TYPE),
        (&TYPE_TYPE, &W_ROOT_TYPE),
        (&crate::descriptor::SUPER_TYPE, &W_ROOT_TYPE),
        (&crate::bytearrayobject::BYTEARRAY_TYPE, &W_ROOT_TYPE),
        (&crate::bytesobject::BYTES_TYPE, &W_ROOT_TYPE),
        (&crate::generator::GENERATOR_TYPE, &W_ROOT_TYPE),
        (&crate::_pypy_generic_alias::UNION_TYPE, &W_ROOT_TYPE),
        (&crate::functional::RANGE_ITER_TYPE, &W_ROOT_TYPE),
        (&crate::iterobject::SEQ_ITER_TYPE, &W_ROOT_TYPE),
        (&crate::nestedscope::CELL_TYPE, &W_ROOT_TYPE),
        (&crate::function::METHOD_TYPE, &W_ROOT_TYPE),
        (&crate::descriptor::PROPERTY_TYPE, &W_ROOT_TYPE),
        (&crate::function::STATICMETHOD_TYPE, &W_ROOT_TYPE),
        (&crate::function::CLASSMETHOD_TYPE, &W_ROOT_TYPE),
        // Exception hierarchy: per-kind PyType statics chain to
        // `EXCEPTION_TYPE` (the BaseException root) so backend
        // `GuardClass` at `OB_TYPE_OFFSET` discriminates subclasses.
        // Order is topological — parent must register before child for
        // the `all_foreign_pytypes` loop in `pyre-jit/src/eval.rs` that
        // looks up `parent_tid` via `pytype_to_tid`.
        (&crate::interp_exceptions::EXCEPTION_TYPE, &W_ROOT_TYPE),
        (
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
            &crate::interp_exceptions::EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_STOP_ASYNC_ITERATION_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_OVERFLOW_ERROR_TYPE,
            &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE,
            &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_TYPE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_VALUE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        // `W_SyntaxError` — direct subclass of Exception
        // (`compile`/`exec`/`eval`/`ast.parse` raise it).
        (
            &crate::interp_exceptions::EXC_SYNTAX_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        // UnicodeError is the intermediate parent of UnicodeDecodeError
        // and UnicodeEncodeError per `pypy/module/exceptions/
        // interp_exceptions.py W_UnicodeError = _new_exception(
        // 'UnicodeError', W_ValueError, ...)`.  Register before its
        // subclasses so the topological-order constraint of the
        // foreign-pytype loop in pyre-jit's eval init holds.
        (
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_VALUE_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
        ),
        // `pypy/module/exceptions/interp_exceptions.py W_UnicodeTranslateError
        // W_UnicodeTranslateError = _new_exception('UnicodeTranslateError',
        // W_UnicodeError, ...)`.
        (
            &crate::interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_NAME_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        // UnboundLocalError subclasses NameError; listed after it so the
        // topological-order constraint of the foreign-pytype loop holds.
        // Its GC tid is pre-registered to the shared `W_BaseException`
        // tid by the per-ExcKind loop in `pyre-jit/src/eval.rs`, so that
        // loop skips this entry; without the pre-registration it would
        // assign an undersized standalone `sizeof(PyObject)` tid and shift
        // every hardcoded post-loop GC tid.
        (
            &crate::interp_exceptions::EXC_UNBOUND_LOCAL_ERROR_TYPE,
            &crate::interp_exceptions::EXC_NAME_ERROR_TYPE,
        ),
        // LookupError is the intermediate parent of IndexError and
        // KeyError per `pypy/module/exceptions/interp_exceptions.py:474
        // W_LookupError = _new_exception('LookupError', W_Exception,
        // ...)`.  Register before its subclasses.
        (
            &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_INDEX_ERROR_TYPE,
            &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_KEY_ERROR_TYPE,
            &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE,
            &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_RECURSION_ERROR_TYPE,
            &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_STOP_ITERATION_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_IMPORT_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE,
            &crate::interp_exceptions::EXC_IMPORT_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ASSERTION_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_REFERENCE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_OS_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE,
            &crate::interp_exceptions::EXC_OS_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_MEMORY_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_SYSTEM_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_GENERATOR_EXIT_TYPE,
            &crate::interp_exceptions::EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_SYSTEM_EXIT_TYPE,
            &crate::interp_exceptions::EXCEPTION_TYPE,
        ),
        (&crate::sliceobject::SLICE_TYPE, &W_ROOT_TYPE),
        (&crate::setobject::SET_TYPE, &W_ROOT_TYPE),
        (&crate::setobject::FROZENSET_TYPE, &W_ROOT_TYPE),
        (&crate::typedef::MEMBER_TYPE, &W_ROOT_TYPE),
        // `pypy/objspace/std/dictmultiobject.py:449/459/469` —
        // dict_keys / dict_values / dict_items.  The three Python
        // visible types share the `W_DictViewObject` payload but each
        // gets a distinct W_TypeObject so `type(d.keys()) is
        // dict_keys` parity holds.
        (&crate::dictmultiobject::DICT_KEYS_TYPE, &W_ROOT_TYPE),
        (&crate::dictmultiobject::DICT_VALUES_TYPE, &W_ROOT_TYPE),
        (&crate::dictmultiobject::DICT_ITEMS_TYPE, &W_ROOT_TYPE),
        // `pypy/interpreter/typedef.py GetSetProperty.typedef`.
        // Registered in the foreign-pytype loop so the `instantiate`
        // back-pointer is set before the first GetSetProperty
        // allocation runs (typedef.rs::getset_descriptor_type forces
        // it for the W_TypeObject side, but the static PyType also
        // needs the foreign-loop entry to seed pytype_to_tid for the
        // GC vtable lookup).
        (&crate::typedef::GETSET_DESCRIPTOR_TYPE, &W_ROOT_TYPE),
        // Appended at the TAIL: inserting mid-list would shift the
        // positionally-assigned type ids of every following entry,
        // silently breaking GuardClass / pytype_to_tid lookups.  The
        // parent `EXC_EXCEPTION_TYPE` is registered far earlier, so the
        // topological constraint still holds at the end.
        (
            &crate::interp_exceptions::EXC_BUFFER_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_EOF_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
    ];
    PYTYPES
}

/// PyType aliases owned by `pyre-object`, keyed by the GC typeid whose
/// inheritance peer supplies their range. Interpreter-owned aliases are
/// appended by `pyre-interpreter::all_subclass_range_aliases`.
pub fn all_subclass_range_aliases() -> Vec<SubclassRangeAlias> {
    use crate::lltype::PyreClassPyTypeOf;

    fn typed<T: PyreClassPyTypeOf>() -> &'static PyType {
        // Every `#[pyre_class]` descriptor points at its macro-emitted static
        // PyType for the program lifetime.
        unsafe { &*T::PYTYPE }
    }

    vec![
        subclass_range_alias(0, &W_ROOT_TYPE),
        subclass_range_alias(1, &INT_TYPE),
        subclass_range_alias(2, &FLOAT_TYPE),
        subclass_range_alias(5, &BOOL_TYPE),
        subclass_range_alias(6, &crate::functional::RANGE_ITER_TYPE),
        subclass_range_alias(7, &LIST_TYPE),
        subclass_range_alias(8, &TUPLE_TYPE),
        // The three arity-2 specialisations each carry their own `ob_type`
        // (`specialisedtupleobject.py` `Cls_ii / Cls_ff / Cls_oo`), so they
        // need their own vtable binding: without one, `subclass_range` on a
        // specialised tuple answers "unknown" and every pure field fold on a
        // constant one — `f.__defaults__` pinned by an identity guard, say —
        // fails `protect_speculative_field` and invalidates the whole loop.
        subclass_range_alias(
            10,
            &crate::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE,
        ),
        subclass_range_alias(
            11,
            &crate::specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE,
        ),
        subclass_range_alias(
            12,
            &crate::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
        ),
        subclass_range_alias(15, &crate::nestedscope::CELL_TYPE),
        subclass_range_alias(16, &crate::function::METHOD_TYPE),
        subclass_range_alias(17, &crate::sliceobject::SLICE_TYPE),
        subclass_range_alias(18, &crate::descriptor::SUPER_TYPE),
        subclass_range_alias(19, &crate::descriptor::PROPERTY_TYPE),
        subclass_range_alias(20, &crate::function::STATICMETHOD_TYPE),
        subclass_range_alias(21, &crate::function::CLASSMETHOD_TYPE),
        subclass_range_alias(22, &crate::_pypy_generic_alias::UNION_TYPE),
        subclass_range_alias(23, &crate::iterobject::SEQ_ITER_TYPE),
        // The producer-specific str/bytes/bytearray/memoryview/array iterator
        // identities all carry the `W_SeqIterObject` payload, so they share
        // its GC type id the way the six dict view iterators share 109.
        subclass_range_alias(23, &crate::iterobject::STR_ASCII_ITER_TYPE),
        subclass_range_alias(23, &crate::iterobject::STR_ITER_TYPE),
        subclass_range_alias(23, &crate::iterobject::BYTES_ITER_TYPE),
        subclass_range_alias(23, &crate::iterobject::BYTEARRAY_ITER_TYPE),
        subclass_range_alias(23, &crate::iterobject::MEMORY_ITER_TYPE),
        subclass_range_alias(23, &crate::iterobject::ARRAY_ITER_TYPE),
        subclass_range_alias(24, typed::<crate::interp_itertools::W_Count>()),
        subclass_range_alias(25, typed::<crate::interp_itertools::W_Repeat>()),
        // `enumerate` W_Enumerate — auto-id `allocate_stable` registered at the
        // tail of the JIT `register_pyre_class` chain (after W_Deque = 110).
        subclass_range_alias(111, typed::<crate::functional::W_Enumerate>()),
        // Formerly-immortal iterators converted to `allocate_stable` (managed),
        // registered at the tail of the JIT `register_pyre_class` chain in this
        // exact order (after W_Enumerate = 111).
        subclass_range_alias(112, typed::<crate::functional::W_Range>()),
        subclass_range_alias(113, typed::<crate::functional::W_LongRangeIterator>()),
        subclass_range_alias(114, typed::<crate::interp_itertools::W_TakeWhile>()),
        subclass_range_alias(115, typed::<crate::interp_itertools::W_DropWhile>()),
        subclass_range_alias(116, typed::<crate::interp_itertools::W_FilterFalse>()),
        subclass_range_alias(117, typed::<crate::interp_itertools::W_Pairwise>()),
        subclass_range_alias(118, typed::<crate::operation::_CallableIterator>()),
        // The two `step == 1` range-iterator shapes do not follow W_Range: they
        // register behind the unconditional code-object walks, the last block
        // before `build_gc`'s target-gated tail, so their ids are the same on
        // every target.
        subclass_range_alias(156, typed::<crate::functional::W_IntRangeStepOneIterator>()),
        subclass_range_alias(157, typed::<crate::functional::W_IntRangeOneArgIterator>()),
        subclass_range_alias(158, &INT_USER_TYPE),
        subclass_range_alias(159, &STR_USER_TYPE),
        subclass_range_alias(160, &TUPLE_USER_TYPE),
        subclass_range_alias(161, &FLOAT_USER_TYPE),
        subclass_range_alias(162, &COMPLEX_USER_TYPE),
        subclass_range_alias(163, &crate::bytesobject::BYTES_USER_TYPE),
        subclass_range_alias(164, &crate::bytearrayobject::BYTEARRAY_USER_TYPE),
        subclass_range_alias(165, &LIST_USER_TYPE),
        subclass_range_alias(166, &crate::setobject::SET_USER_TYPE),
        subclass_range_alias(166, &crate::setobject::FROZENSET_USER_TYPE),
        subclass_range_alias(167, &crate::interp_array::ARRAY_USER_TYPE),
        subclass_range_alias(168, &crate::weakref::WEAKREF_LAYOUT_TYPE),
        subclass_range_alias(169, &crate::functional::ENUMERATE_USER_TYPE),
        subclass_range_alias(170, &crate::functional::MAP_USER_TYPE),
        subclass_range_alias(171, &crate::functional::FILTER_USER_TYPE),
        subclass_range_alias(172, &crate::functional::ZIP_USER_TYPE),
        subclass_range_alias(173, &crate::functional::REVERSED_USER_TYPE),
        subclass_range_alias(174, &crate::descriptor::SUPER_USER_TYPE),
        subclass_range_alias(175, &crate::descriptor::PROPERTY_USER_TYPE),
        subclass_range_alias(176, &crate::interp_itertools::COUNT_USER_TYPE),
        subclass_range_alias(177, &crate::interp_itertools::REPEAT_USER_TYPE),
        subclass_range_alias(178, &crate::interp_itertools::TAKEWHILE_USER_TYPE),
        subclass_range_alias(179, &crate::interp_itertools::DROPWHILE_USER_TYPE),
        subclass_range_alias(180, &crate::interp_itertools::FILTERFALSE_USER_TYPE),
        subclass_range_alias(181, &crate::interp_itertools::ISLICE_USER_TYPE),
        subclass_range_alias(182, &crate::interp_itertools::BATCHED_USER_TYPE),
        subclass_range_alias(183, &crate::interp_itertools::PRODUCT_USER_TYPE),
        subclass_range_alias(184, &crate::interp_itertools::COMBINATIONS_USER_TYPE),
        subclass_range_alias(
            185,
            &crate::interp_itertools::COMBINATIONS_WITH_REPLACEMENT_USER_TYPE,
        ),
        subclass_range_alias(186, &crate::interp_itertools::PERMUTATIONS_USER_TYPE),
        subclass_range_alias(187, &crate::interp_itertools::GROUPBY_USER_TYPE),
        subclass_range_alias(188, &crate::interp_itertools::COMPRESS_USER_TYPE),
        subclass_range_alias(189, &crate::interp_itertools::STARMAP_USER_TYPE),
        subclass_range_alias(190, &crate::interp_itertools::ACCUMULATE_USER_TYPE),
        subclass_range_alias(191, &crate::interp_itertools::ZIP_LONGEST_USER_TYPE),
        subclass_range_alias(192, &crate::interp_itertools::PAIRWISE_USER_TYPE),
        subclass_range_alias(193, &crate::interp_itertools::CYCLE_USER_TYPE),
        subclass_range_alias(194, &crate::interp_itertools::CHAIN_USER_TYPE),
        subclass_range_alias(198, &crate::_pypy_generic_alias::GENERIC_ALIAS_USER_TYPE),
        subclass_range_alias(199, &LONG_USER_TYPE),
        subclass_range_alias(200, &crate::weakref::WEAKREF_LAYOUT_USER_TYPE),
        subclass_range_alias(201, &crate::function::STATICMETHOD_USER_TYPE),
        subclass_range_alias(202, &crate::function::CLASSMETHOD_USER_TYPE),
        subclass_range_alias(203, &MODULE_USER_TYPE),
        subclass_range_alias(26, &crate::typedef::MEMBER_TYPE),
        subclass_range_alias(27, &crate::bytesobject::BYTES_TYPE),
        subclass_range_alias(28, &crate::bytearrayobject::BYTEARRAY_TYPE),
        subclass_range_alias(29, &DICT_TYPE),
        subclass_range_alias(30, &crate::setobject::SET_TYPE),
        subclass_range_alias(30, &crate::setobject::FROZENSET_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXCEPTION_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_SYNTAX_ERROR_TYPE),
        subclass_range_alias(
            31,
            &crate::interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE,
        ),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_UNBOUND_LOCAL_ERROR_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_BUFFER_ERROR_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_STOP_ASYNC_ITERATION_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_EOF_ERROR_TYPE),
        // `_getusercls` instance layouts. `W_ExceptionExtended` (214) has no
        // vtable alias; these two are the mapdict typeptrs `is_exception`
        // must accept.
        subclass_range_alias(215, &crate::interp_exceptions::BASE_EXCEPTION_USER_TYPE),
        subclass_range_alias(216, &crate::interp_exceptions::EXCEPTION_EXTENDED_USER_TYPE),
        subclass_range_alias(32, &crate::generator::GENERATOR_TYPE),
        subclass_range_alias(33, &TYPE_TYPE),
        subclass_range_alias(34, &STR_TYPE),
        subclass_range_alias(35, &LONG_TYPE),
        subclass_range_alias(36, &MODULE_TYPE),
        subclass_range_alias(38, &MAPPING_PROXY_TYPE),
        subclass_range_alias(39, &crate::dictmultiobject::DICT_KEYS_TYPE),
        subclass_range_alias(39, &crate::dictmultiobject::DICT_VALUES_TYPE),
        subclass_range_alias(39, &crate::dictmultiobject::DICT_ITEMS_TYPE),
        subclass_range_alias(40, &crate::typedef::GETSET_DESCRIPTOR_TYPE),
        subclass_range_alias(45, &NONE_TYPE),
        subclass_range_alias(46, &NOTIMPLEMENTED_TYPE),
        subclass_range_alias(47, &ELLIPSIS_TYPE),
        subclass_range_alias(48, &crate::dictmultiobject::MODULE_DICT_TYPE),
        subclass_range_alias(49, &crate::celldict::OBJECT_MUTABLE_CELL_TYPE),
        subclass_range_alias(50, &crate::celldict::INT_MUTABLE_CELL_TYPE),
        subclass_range_alias(52, &crate::weakref::GC_WEAKREF_BOX_TYPE),
        subclass_range_alias(53, &INSTANCE_TYPE),
        subclass_range_alias(217, &INSTANCE_USER_TYPE),
        subclass_range_alias(54, &COMPLEX_TYPE),
        subclass_range_alias(56, &crate::interp_exceptions::EXC_EXCEPTION_TYPE),
        subclass_range_alias(57, &crate::interp_exceptions::EXC_SYSTEM_EXIT_TYPE),
        subclass_range_alias(58, &crate::interp_exceptions::EXC_GENERATOR_EXIT_TYPE),
        subclass_range_alias(59, &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE),
        subclass_range_alias(60, &crate::interp_exceptions::EXC_OVERFLOW_ERROR_TYPE),
        subclass_range_alias(61, &crate::interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE),
        subclass_range_alias(62, &crate::interp_exceptions::EXC_TYPE_ERROR_TYPE),
        subclass_range_alias(63, &crate::interp_exceptions::EXC_VALUE_ERROR_TYPE),
        subclass_range_alias(64, &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE),
        subclass_range_alias(65, &crate::interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE),
        subclass_range_alias(66, &crate::interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE),
        subclass_range_alias(
            67,
            &crate::interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE,
        ),
        subclass_range_alias(68, &crate::interp_exceptions::EXC_NAME_ERROR_TYPE),
        subclass_range_alias(69, &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE),
        subclass_range_alias(70, &crate::interp_exceptions::EXC_INDEX_ERROR_TYPE),
        subclass_range_alias(71, &crate::interp_exceptions::EXC_KEY_ERROR_TYPE),
        subclass_range_alias(72, &crate::interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE),
        subclass_range_alias(73, &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE),
        subclass_range_alias(
            74,
            &crate::interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE,
        ),
        subclass_range_alias(75, &crate::interp_exceptions::EXC_RECURSION_ERROR_TYPE),
        subclass_range_alias(76, &crate::interp_exceptions::EXC_STOP_ITERATION_TYPE),
        subclass_range_alias(77, &crate::interp_exceptions::EXC_IMPORT_ERROR_TYPE),
        subclass_range_alias(78, &crate::interp_exceptions::EXC_ASSERTION_ERROR_TYPE),
        subclass_range_alias(79, &crate::interp_exceptions::EXC_REFERENCE_ERROR_TYPE),
        subclass_range_alias(80, &crate::interp_exceptions::EXC_OS_ERROR_TYPE),
        subclass_range_alias(81, &crate::interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE),
        subclass_range_alias(82, &crate::interp_exceptions::EXC_MEMORY_ERROR_TYPE),
        subclass_range_alias(83, &crate::interp_exceptions::EXC_SYSTEM_ERROR_TYPE),
        subclass_range_alias(84, typed::<crate::interp_sre::W_SRE_Pattern>()),
        subclass_range_alias(85, typed::<crate::interp_sre::W_SRE_Match>()),
        subclass_range_alias(86, typed::<crate::interp_sre::W_SRE_Scanner>()),
        subclass_range_alias(87, typed::<crate::_pypy_generic_alias::GenericAlias>()),
        subclass_range_alias(89, typed::<crate::functional::W_ReversedIterator>()),
        subclass_range_alias(90, typed::<crate::functional::W_Filter>()),
        subclass_range_alias(91, typed::<crate::functional::W_Map>()),
        subclass_range_alias(92, typed::<crate::functional::W_Zip>()),
        subclass_range_alias(93, typed::<crate::interp_itertools::W_Cycle>()),
        subclass_range_alias(94, typed::<crate::interp_array::W_Array>()),
        subclass_range_alias(95, typed::<crate::interp_itertools::W_Chain>()),
        subclass_range_alias(96, typed::<crate::memoryview::W_MemoryView>()),
        subclass_range_alias(99, typed::<crate::setobject::W_SetIterObject>()),
        subclass_range_alias(100, typed::<crate::iterobject::W_ListIterObject>()),
        subclass_range_alias(101, typed::<crate::iterobject::W_ListReverseIterObject>()),
        subclass_range_alias(102, typed::<crate::iterobject::W_TupleIterObject>()),
        subclass_range_alias(103, typed::<crate::interp_itertools::W_Compress>()),
        subclass_range_alias(104, typed::<crate::interp_itertools::W_StarMap>()),
        subclass_range_alias(105, typed::<crate::interp_itertools::W_Accumulate>()),
        subclass_range_alias(106, typed::<crate::interp_itertools::W_ZipLongest>()),
        subclass_range_alias(107, &crate::generator::COROUTINE_TYPE),
        subclass_range_alias(108, typed::<crate::generator::CoroutineWrapper>()),
        subclass_range_alias(109, &crate::dictmultiobject::DICT_KEYITERATOR_TYPE),
        subclass_range_alias(109, &crate::dictmultiobject::DICT_VALUEITERATOR_TYPE),
        subclass_range_alias(109, &crate::dictmultiobject::DICT_ITEMITERATOR_TYPE),
        subclass_range_alias(109, &crate::dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE),
        subclass_range_alias(109, &crate::dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE),
        subclass_range_alias(109, &crate::dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE),
        // Async-generator support is appended after the interpreter-owned
        // FrameLocalsProxy (125) in build_gc: the shared generator payload
        // gets 126, followed by the three pyre_class helper awaitables.
        subclass_range_alias(124, &crate::generator::ASYNC_GENERATOR_TYPE),
        subclass_range_alias(125, typed::<crate::generator::AsyncGenValueWrapper>()),
        subclass_range_alias(126, typed::<crate::generator::AsyncGenASend>()),
        subclass_range_alias(127, typed::<crate::generator::AsyncGenAThrow>()),
        // W_ISlice is appended after the interpreter-owned W_Local (135) in
        // build_gc so every pre-existing Python-visible AUTO-ID stays stable.
        subclass_range_alias(134, typed::<crate::interp_itertools::W_ISlice>()),
        // W_Batched follows W_ISlice in the same append-only registration
        // chain.
        subclass_range_alias(135, typed::<crate::interp_itertools::W_Batched>()),
        subclass_range_alias(136, typed::<crate::interp_itertools::W_Product>()),
        subclass_range_alias(137, typed::<crate::interp_itertools::W_Combinations>()),
        subclass_range_alias(
            138,
            typed::<crate::interp_itertools::W_CombinationsWithReplacement>(),
        ),
        subclass_range_alias(139, typed::<crate::interp_itertools::W_Permutations>()),
        subclass_range_alias(140, typed::<crate::interp_itertools::W_GroupBy>()),
        subclass_range_alias(141, typed::<crate::interp_itertools::W_GroupByIterator>()),
        subclass_range_alias(
            142,
            typed::<crate::interp_itertools::W_TeeChainedListNode>(),
        ),
        subclass_range_alias(143, typed::<crate::interp_itertools::W_TeeIterable>()),
        // `_buffer_wrapper` follows the deque's internal non-object Block
        // (146) at the append-only GC registration tail.
        subclass_range_alias(145, typed::<crate::memoryview::W_BufferWrapper>()),
    ]
}

// ── Type checks ───────────────────────────────────────────────────────

/// Type name of any object, tag-safe. A tagged immediate is an exact `int`;
/// name it without derefing its (non-pointer) tagged bits as `ob_type`.
/// Gated on `CAN_BE_TAGGED` (folds to the raw `ob_type` deref at flag-false →
/// byte-identical). The chokepoint for the "must be X, not <name>" error
/// messages that a tagged int reaches after the tag-safe type probes reject
/// it. The else arm keeps the RAW `(*(*obj).ob_type).name` (NOT `r#type`,
/// which returns the `w_class` subclass name).
///
/// # Safety
/// `obj` must be a valid pointer to a `PyObject` unless it is a tagged
/// immediate.
#[inline]
pub unsafe fn type_name_of(obj: PyObjectRef) -> &'static str {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        "int"
    } else {
        unsafe { (*(*obj).ob_type).name }
    }
}

/// Check if an object is of a given type (pointer identity comparison).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn py_type_check(obj: PyObjectRef, tp: &PyType) -> bool {
    // A tagged immediate's type is `int`, synthesized before the `ob_type`
    // deref: it matches iff `tp` is the `int` vtable. Gated on
    // `CAN_BE_TAGGED` (default false), so the deref below is the only live
    // path until enablement. This is the shared chokepoint for
    // `is_bool`/`is_float`/`is_long`/`is_complex`, which inherit the guard.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return std::ptr::eq(tp as *const PyType, &INT_TYPE as *const PyType);
    }
    !obj.is_null() && unsafe { std::ptr::eq((*obj).ob_type, tp as *const PyType) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_int(obj: PyObjectRef) -> bool {
    // A tagged immediate is a plain `int` (never a `bool`: bools are
    // even-aligned singletons). `is_int` reaches `ob_type` via
    // `py_type_check`, which derefs directly — so it carries its own
    // tag short-circuit rather than routing through `ll_type`. Gated on
    // the `CAN_BE_TAGGED` static (default false), inspecting only the
    // pointer bits, so the deref path below is the only live one.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return true;
    }
    unsafe {
        py_type_check(obj, &INT_TYPE)
            || py_type_check(obj, &BOOL_TYPE)
            || py_type_check(obj, &INT_USER_TYPE)
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_bool(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BOOL_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_float(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FLOAT_TYPE) || py_type_check(obj, &FLOAT_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_complex(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COMPLEX_TYPE) || py_type_check(obj, &COMPLEX_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_long(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LONG_TYPE) || py_type_check(obj, &LONG_USER_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_int_or_long(obj: PyObjectRef) -> bool {
    unsafe { is_int(obj) || is_long(obj) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_list(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LIST_TYPE) || py_type_check(obj, &LIST_USER_TYPE) }
}

/// Recognise any of the four tuple variants —
/// canonical `W_TupleObject` plus the three `W_SpecialisedTupleObject_*`
/// arity-2 specialisations from
/// `pypy/objspace/std/specialisedtupleobject.py`. All four share the
/// same Python `tuple` typedef in pypy; pyre encodes that by giving
/// each variant a distinct `ob_type` (RPython-vtable equivalent) while
/// `w_class` always resolves to the canonical `tuple` class object.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_tuple(obj: PyObjectRef) -> bool {
    use crate::specialisedtupleobject::{
        SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    };
    unsafe {
        py_type_check(obj, &TUPLE_TYPE)
            || py_type_check(obj, &TUPLE_USER_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_II_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_FF_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_OO_TYPE)
    }
}

/// `PyTuple_CheckExact` — an exact `tuple`, excluding tuple subclasses.
/// Covers the specialised arity-2 variants too: they all carry
/// `w_class == get_instantiate(&TUPLE_TYPE)`, so comparing the user-visible
/// class object (not `get_instantiate(ob_type)`) keeps them exact while a
/// subclass instance (retagged `w_class`) reads as non-exact.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_exact_tuple(obj: PyObjectRef) -> bool {
    unsafe { is_tuple(obj) && std::ptr::eq((*obj).w_class, get_instantiate(&TUPLE_TYPE)) }
}

/// `PyList_CheckExact` — an exact `list`, excluding list subclasses.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_exact_list(obj: PyObjectRef) -> bool {
    unsafe { is_list(obj) && std::ptr::eq((*obj).w_class, get_instantiate(&LIST_TYPE)) }
}

/// `pypy/objspace/std/dictmultiobject.py` makes both `W_DictObject` and
/// `W_ModuleDictObject` subclasses of `W_DictMultiObject`, so user-level
/// `isinstance(obj, dict)` is true for both.  Pyre exposes each layout
/// behind a distinct static `PyType` tag (so the Rust runtime can pick
/// the right cast), but `is_dict` reports the user-visible answer and
/// returns true for either.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_dict(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DICT_TYPE) || crate::dictmultiobject::is_module_dict(obj) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_none(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NONE_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_not_implemented(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NOTIMPLEMENTED_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_ellipsis(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ELLIPSIS_TYPE) }
}

/// `W_IntObject.is_w` / `W_AbstractIntObject.is_w` (intobject.py).
///
/// `ObjSpace.is_w` calls `w_two.is_w(w_one)`. A machine int compares
/// `intval` with `int_w` of the other operand; `OverflowError` from a long
/// that does not fit means they are not identical. A long has no override,
/// so it compares `bigint_w` of both. A machine int equals a long exactly
/// when `rbigint.int_eq` says so, and that comparison does not allocate
/// while the machine word is inside `int_in_valid_range`. `i64::MIN` is
/// the one value outside that range; the bigint built for it is rooted
/// across, and the long is reloaded before the compare.
///
/// Two machine ints never build a bigint. `W_LongObject` shares the `int`
/// `w_class`, so `is_exact_type(..., INT_TYPE)` is true for it; the long
/// is told apart by `is_long` (`LONG_TYPE`), not by that gate.
///
/// `inline(never)` keeps this graph off `W_IntObject.is_w`'s body: that
/// method does not call `bigint_w`.
#[inline(never)]
fn abstract_int_is_w(w_one: PyObjectRef, w_two: PyObjectRef) -> bool {
    unsafe {
        let one_long = int_operand_is_long(w_one);
        let two_long = int_operand_is_long(w_two);
        if !one_long && !two_long {
            return crate::intobject::w_int_get_value(w_one)
                == crate::intobject::w_int_get_value(w_two);
        }
        if one_long && two_long {
            return crate::longobject::w_long_get_value(w_one)
                .eq(crate::longobject::w_long_get_value(w_two));
        }
        let (long_obj, intval) = if one_long {
            (w_one, crate::intobject::w_int_get_value(w_two))
        } else {
            (w_two, crate::intobject::w_int_get_value(w_one))
        };
        if majit_rlib::rbigint::int_in_valid_range(intval) {
            return crate::longobject::w_long_get_value(long_obj).int_eq(intval);
        }
        let _roots = crate::gc_roots::push_roots();
        let long_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(long_obj);
        let built = majit_rlib::rbigint::RBigInt::from(intval);
        let long_obj = crate::gc_roots::shadow_stack_get(long_slot);
        crate::longobject::w_long_get_value(long_obj).eq(&built)
    }
}

/// A `W_LongObject` among the exact-`int` pair. A tagged immediate is a
/// machine int and must not be read as a header.
fn int_operand_is_long(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    unsafe { is_long(obj) }
}

/// `baseobjspace.py` `ObjSpace.is_w`. Dispatches the per-type `is_w`
/// overrides `W_AbstractIntObject.is_w`, `W_BoolObject.is_w`,
/// `W_FloatObject.is_w`, `W_AbstractTupleObject.is_w`,
/// `W_AbstractBytesObject.is_w`, `W_UnicodeObject.is_w`, and
/// `W_FrozensetObject.is_w`.
///
/// The dispatch is on `w_two`, as `w_two.is_w(space, w_one)` is: every gate
/// reads `w_two`'s type first, so `x is CONST` never reads `x`'s class.
///
/// Exact `int` splits the way `intobject.py` does: `W_IntObject.is_w`
/// compares `intval`; `W_AbstractIntObject.is_w` (`abstract_int_is_w`)
/// is the long/`i64::MIN` path that may build a bigint. Both return
/// `bool` — an `Option<bool>` merge is `int ∪ r_uint` at the return block.
pub fn is_w(w_one: PyObjectRef, w_two: PyObjectRef) -> bool {
    if std::ptr::eq(w_one, w_two) {
        return true;
    }
    // `W_AbstractIntObject.is_w` (intobject.py): two plain `int`s
    // — `W_IntObject` or the BigInt-backed `W_LongObject` — are
    // identical when their values are equal.  `bool`
    // (`W_BoolObject.is_w` is pure pointer identity, boolobject.py)
    // and `int` subclasses (`user_overridden_class`) keep pointer
    // identity — the exact-type gate excludes both (a `bool`'s
    // `w_class` is `bool`, a subclass instance's is the subclass), so
    // they fall through to the `ptr::eq` above.
    //
    // Tagged immediates need no special case: `is_exact_type` reports a
    // tagged int as an exact `int` (never a subclass — those stay boxed),
    // and `range_obj_to_bigint` reads its value through the tag-aware
    // `w_int_get_value`. Two equal-valued immediates already matched the
    // `ptr::eq` above (identical bit patterns); an immediate and a boxed
    // int of the same value fall here and compare equal by value.
    unsafe {
        if crate::pyobject::is_exact_type(w_two, &crate::pyobject::INT_TYPE)
            && crate::pyobject::is_exact_type(w_one, &crate::pyobject::INT_TYPE)
        {
            if int_operand_is_long(w_one) || int_operand_is_long(w_two) {
                return abstract_int_is_w(w_one, w_two);
            }
            return crate::intobject::w_int_get_value(w_one)
                == crate::intobject::w_int_get_value(w_two);
        }
        // `W_FloatObject.is_w` (floatobject.py): two plain
        // `float`s are identical when their bit patterns are equal
        // (`float2longlong`), so `0.0 is -0.0` is false. `float` subclasses
        // (`user_overridden_class`) keep pointer identity — the exact-type
        // gate excludes them. That method has no `@jit` hint; `_hash_float`
        // is the elidable.
        //
        // Distinct NaNs stay boxed and compare by pointer (`Py_Is`).
        // `cpython_differences.rst` "Object Identity of Primitive Values"
        // documents PyPy's bit identity for float, NaN included.
        // `pin_free_builtin_is` repeats this arm so `IS_OP` agrees.
        if crate::pyobject::is_exact_type(w_two, &crate::pyobject::FLOAT_TYPE)
            && crate::pyobject::is_exact_type(w_one, &crate::pyobject::FLOAT_TYPE)
        {
            let one = crate::floatobject::w_float_get_value(w_one);
            let two = crate::floatobject::w_float_get_value(w_two);
            if one.is_nan() || two.is_nan() {
                return false;
            }
            return one.to_bits() == two.to_bits();
        }
        // CPython 3.14 gives complex objects pointer identity, handled above.
        // `W_AbstractTupleObject.is_w` (tupleobject.py): a `tuple` is
        // identical to another only when both are the empty tuple — "empty
        // tuples are unique-ified". Non-empty tuples keep pointer identity
        // (the `ptr::eq` above handled `self is w_other`); `tuple`
        // subclasses keep pointer identity through the exact-type gate. The
        // specialised arity-2 tuples carry the canonical `tuple` w_class, so
        // they pass the gate but are never empty (length 2).
        if crate::pyobject::is_exact_type(w_two, &crate::pyobject::TUPLE_TYPE)
            && crate::pyobject::is_exact_type(w_one, &crate::pyobject::TUPLE_TYPE)
        {
            return crate::tupleobject::w_tuple_len(w_one) == 0
                && crate::tupleobject::w_tuple_len(w_two) == 0;
        }
        // `W_AbstractBytesObject.is_w` (bytesobject.py): for distinct
        // exact-`bytes` operands, `len(s2) > 1` returns `s1 is s2` (storage
        // identity); BytesListStrategy re-wraps the same erased rpython string,
        // so distinct wrappers may deliberately share that backing block.
        // `len(s2) == 0` returns `len(s1) == 0`; `len(s2) == 1`
        // (unique-ified) returns `len(s1) == 1 && s1[0] == s2[0]`.
        if crate::pyobject::is_exact_type(w_two, &crate::bytesobject::BYTES_TYPE)
            && crate::pyobject::is_exact_type(w_one, &crate::bytesobject::BYTES_TYPE)
        {
            let len1 = crate::bytesobject::w_bytes_len(w_one);
            let len2 = crate::bytesobject::w_bytes_len(w_two);
            if len2 > 1 {
                return crate::bytesobject::w_bytes_block(w_one)
                    == crate::bytesobject::w_bytes_block(w_two);
            }
            if len2 == 0 {
                return len1 == 0;
            }
            return len1 == 1
                && crate::bytesobject::w_bytes_getitem(w_one, 0)
                    == crate::bytesobject::w_bytes_getitem(w_two, 0);
        }
        // `W_UnicodeObject.is_w` (unicodeobject.py), called on `w_two` as
        // `baseobjspace.py ObjSpace.is_w` does. Its `isinstance` and
        // `user_overridden_class` checks are the exact-type gate: `str`
        // subclasses keep pointer identity.
        if crate::pyobject::is_exact_type(w_two, &crate::pyobject::STR_TYPE)
            && crate::pyobject::is_exact_type(w_one, &crate::pyobject::STR_TYPE)
        {
            let w_one = &*(w_one as *const crate::unicodeobject::W_UnicodeObject);
            let w_two = &*(w_two as *const crate::unicodeobject::W_UnicodeObject);
            return w_two.is_w(w_one);
        }
        // `W_FrozensetObject.is_w` (setobject.py): two `frozenset`s
        // are identical only when both are empty — "empty frozensets are
        // unique-ified". The mutable `set` carries a distinct type tag and
        // does not override `is_w`, so the `FROZENSET_TYPE` gate excludes
        // it; `frozenset` subclasses are excluded too.
        if crate::pyobject::is_exact_type(w_two, &crate::setobject::FROZENSET_TYPE)
            && crate::pyobject::is_exact_type(w_one, &crate::setobject::FROZENSET_TYPE)
        {
            return crate::setobject::w_set_len(w_one) == 0
                && crate::setobject::w_set_len(w_two) == 0;
        }
    }
    false
}
