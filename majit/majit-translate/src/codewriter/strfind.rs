//! Skip-spine mint of `LLHelpers.ll_find` (`rstr.py`).
//!
//! `pairtype(AbstractStringRepr, AbstractStringRepr).rtype_contains`
//! (`rstr.py`) is `gendirectcall(ll_strlen)` then
//! `gendirectcall(ll_find, s1, s2, 0, end)` then `int_ne(vres, -1)`.
//! A graph that stays on the rich-`OpKind` spine never meets the rtyper, so
//! the front's `BinOp { op: "contains" }` would assemble as the unwired
//! `int_contains/rr>i`.
//!
//! This module gives that spine the helper `rtype_contains` would have
//! minted: [`ll_find_path`] registers [`build_ll_find_helper_graph`] as a
//! Spine-B candidate (`CallControl::register_opname_helper_graph`), matching
//! `codewriter::minmax` / `codewriter::getslice`.

use std::rc::Rc;

use crate::codewriter::call::CallControl;
use crate::parse::CallPath;
use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, build_ll_find_helper_graph};

/// `LLHelpers.ll_find` (`rstr.py`).
pub const LL_FIND: &str = "ll_find";

/// The helper path for `ll_find`, minting and registering the graph on first use.
pub fn ll_find_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_FIND]);
    if cc.jitcode_handle(&path).is_none() {
        let py = build_ll_find_helper_graph(LL_FIND, STRPTR.clone()).expect("ll_find helper graph");
        let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
        let flow = Rc::try_unwrap(graph)
            .expect("fresh ll_find helper is uniquely owned")
            .into_inner();
        cc.register_opname_helper_graph(path.clone(), flow);
    }
    path
}
