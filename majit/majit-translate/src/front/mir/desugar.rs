//! Hand lowerings of individual callees.
//!
//! Upstream gets these by flowing the callee's own body or from an rtyper
//! `rtype_*` method. The set is meant to shrink.

use super::*;

pub(in crate::front::mir) mod call;
pub(in crate::front::mir) mod callee;
pub(in crate::front::mir) mod enum_site;
pub(in crate::front::mir) mod num;
pub(in crate::front::mir) mod rewire;
pub(in crate::front::mir) mod std_call;

pub(in crate::front::mir) use self::callee::*;
pub(in crate::front::mir) use self::rewire::*;
pub(in crate::front::mir) use self::std_call::{
    atomic_pure_reborrow_source, is_atomic_from_ptr_identity, is_atomic_load, is_atomic_store,
    is_inlined_mem_exchange_path, whole_local_copy_source,
};
