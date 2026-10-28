//! Inherent-method metadata carrier shared across the front-end and
//! codewriter. `CallPath` lives in `majit-jitcode`.

pub use majit_jitcode::parse::{
    ACCESS_DIRECT_ALT_SUFFIX, CallPath, GraphSpecialization, canonical_leaf,
};

#[derive(Debug, Clone)]
pub struct InherentMethodInfo {
    pub for_type: String,
    pub self_ty_root: Option<String>,
    pub name: String,
    pub graph: crate::model::LazyGraph,
    /// RPython: op.result.concretetype — return type for array identity.
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    pub hints: Vec<String>,
}
