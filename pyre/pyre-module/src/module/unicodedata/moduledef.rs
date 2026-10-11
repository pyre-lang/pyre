//! `pypy/module/unicodedata/moduledef.py` — `class Module(MixedModule)`.
//!
//! Bodies live in `interp_ucd`.

use pyre_object::*;

use super::interp_ucd;

pyre_interpreter::py_module! {
    "unicodedata",
    interpleveldefs: {
        "unidata_version" => w_str_new(&rustpython_unicode::unicode_version()),
    },
    functions: {
        "category"         / * = interp_ucd::category,
        "bidirectional"    / * = interp_ucd::bidirectional,
        "east_asian_width" / * = interp_ucd::east_asian_width,
        "combining"        / * = interp_ucd::combining,
        "mirrored"         / * = interp_ucd::mirrored,
        "decomposition"    / * = interp_ucd::decomposition,
        "digit"            / * = interp_ucd::digit,
        "decimal"          / * = interp_ucd::decimal,
        "numeric"          / * = interp_ucd::numeric,
        "name"             / * = interp_ucd::name,
        "lookup"           / * = interp_ucd::lookup,
        "normalize"        / * = interp_ucd::normalize,
        "is_normalized"    / * = interp_ucd::is_normalized,
    },
    extra_init: |ns| {
        // `unicodedata.ucd_3_2_0` — a `UCD` instance pinned to the Unicode
        // 3.2.0 database (used by `stringprep`).  Version-sensitive queries
        // the typed UCD instance selects `Ucd::new(false)` while
        // lookup/normalize/is_normalized share version-independent
        // implementations with the module callables.
        // Install the TypeDef before allocation so the generated allocator
        // can stamp the canonical Python class in `w_class`.
        let mut ns = ns;
        let ucd_type = pyre_object::with_roots!(ns => interp_ucd::type_object());
        // `interp_ucd.py UCD.typedef` declares no `__new__`, so the two
        // database instances the module exports are the only ones that exist;
        // reaching generic allocation would hand back a `UCD` with no
        // database at all.
        unsafe { pyre_object::w_type_set_disallow_instantiation(ucd_type) };
        // Installing the module attribute can allocate; keep the freshly
        // allocated instance rooted until the namespace owns it.
        let mut ucd = pyre_object::with_roots!(ns => interp_ucd::ucd_3_2_0());
        pyre_interpreter::__pyre_store!(ns, "ucd_3_2_0", ucd);
    },
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `unicodedata.UCD`: an `allocate_stable` type with no inline object
    // payload, so the header `w_class` — which a Python subclass instance points
    // at a managed heap type — is the only edge its marker forwards.
    types.push(ModuleGcType {
        descriptor: <interp_ucd::W_UCD as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
