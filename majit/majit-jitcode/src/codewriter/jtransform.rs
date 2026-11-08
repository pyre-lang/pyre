//! The ARRAY identities `rpython/jit/codewriter/jtransform.py` and the MIR
//! front end name, which the JIT runtime's own descrs publish under.

/// Identity carrier for the length-prefixed `Ptr(GcArray(PyObjectRef))`
/// block — the `ItemsBlock` behind list / tuple storage and the
/// `FixedObjectArray` behind frame locals and mro blocks, which share one
/// tid and one runtime descr singleton (`pyobject_gcarray_descr`).
///
/// `descr.py get_array_descr` keys `cache[ARRAY]` on the ARRAY
/// lltype's object identity; every lltype op carries its `concretetype`,
/// so upstream never meets an identity-less array. The arms below reach
/// these blocks through devirtualized accessor calls rather than a MIR
/// `Place` projection, so there is no type to read the identity off —
/// name it explicitly instead. Without a name `arraydescrof_concrete`
/// returns no descr-set key, `canonicalize_keyed_descrs` drops the whole
/// set, and the callee's `EffectInfo` degrades to `EF_RANDOM_EFFECTS`.
///
/// Runtime `pyobject_gcarray_descr` publishes under this same string
/// (`cpu.arraydescrof(ARRAY)` / `descr.py get_array_descr` cache[ARRAY])
/// so short-preamble `ArrayPtrInfo.make_guards` can resolve
/// `PY_OBJECT_ARRAY_GC_TYPE_ID` from `path_hash` of this identity.
pub const OBJECT_REF_GCARRAY_TYPE_ID: &str = "majit::object_ref_gcarray";

/// ARRAY identity of the list backing blocks, the key `cpu.arraydescrof`
/// caches on.  `rlist.py` has one `GcArray(ITEM)` lltype per item kind, so
/// every access to such a block names the same ArrayDescr as the runtime's
/// own descr for it.
pub const LIST_INT_ITEMS_ARRAY: &str = "GcArray<i64>";
pub const LIST_FLOAT_ITEMS_ARRAY: &str = "GcArray<f64>";
/// `Ptr(GcArray(GCREF))` — `get_itemarray_lowleveltype` after
/// `externalvsinternal(gcref=True)`. One ARRAY, distinct from
/// [`LIST_OBJ_ITEMS_ARRAY`] (`GcArray(OBJECTPTR)`).
pub const LIST_GCREF_ITEMS_ARRAY: &str = "GcArray<GCREF>";
/// The object block is the one the MIR front end already names for tuple
/// items, frame locals and mro blocks.
pub const LIST_OBJ_ITEMS_ARRAY: &str = OBJECT_REF_GCARRAY_TYPE_ID;

/// One ARRAY lltype, one descr key.
///
/// `[i64]` and `GcArray<i64>` (and the `f64` pair) are the same
/// `GcArray(Signed)` / `GcArray(Float)` that `cpu.arraydescrof` and
/// `get_array_descr` key once. `[str]` is that same join for
/// `GcArray(GCREF)` (`get_itemarray_lowleveltype` after
/// `externalvsinternal(gcref=True)`). `[*mut PyObject]` is the published
/// field layout of a fixed list of refs (`wrappeditems`); it is the
/// same `GcArray(OBJECTPTR)` as [`OBJECT_REF_GCARRAY_TYPE_ID`], and it
/// is not `GcArray(GCREF)`. Callers that turn a spelling into a
/// `_cache_array` key or an effectinfo array index run the spelling
/// through here first. Every other identity is unchanged.
pub fn canonical_array_type_id(array_type_id: &str) -> std::borrow::Cow<'_, str> {
    match array_type_id {
        "[i64]" => std::borrow::Cow::Borrowed("GcArray<i64>"),
        "[f64]" => std::borrow::Cow::Borrowed("GcArray<f64>"),
        "[str]" => std::borrow::Cow::Borrowed(LIST_GCREF_ITEMS_ARRAY),
        "[*mut PyObject]" => std::borrow::Cow::Borrowed(OBJECT_REF_GCARRAY_TYPE_ID),
        other => std::borrow::Cow::Borrowed(other),
    }
}
