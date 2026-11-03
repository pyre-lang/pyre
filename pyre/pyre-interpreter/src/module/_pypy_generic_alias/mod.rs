//! `_pypy_generic_alias` — PyPy's app-level GenericAlias/UnionType module.
//!
//! The line-by-line port lives in `pyre_interpreter::_pypy_generic_alias`; expose its
//! canonical type objects under the private module names that PyPy app-level
//! modules import.

pyre_interpreter::py_module! {
    "_pypy_generic_alias",
    interpleveldefs: {
        "GenericAlias" => pyre_interpreter::typedef::gettypeobject(
            &pyre_object::GENERIC_ALIAS_TYPE,
        ),
        "UnionType" => pyre_interpreter::typedef::gettypeobject(
            &pyre_object::UNION_TYPE,
        ),
    },
    functions: {
        "_create_union" / 2 = |args| {
            pyre_interpreter::_pypy_generic_alias::create_union(args[0], args[1])
        },
    },
}
