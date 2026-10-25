//! RPython `rpython/rtyper/rvirtualizable.py`.
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::{ConstValue, Constant, FunctionGraph, Hlvalue};
use crate::translator::rtyper::lltypesystem::lltype::{self, _ptr, LowLevelType};
use crate::translator::rtyper::rclass::const_truthy;

/// RPython `class VirtualizableInstanceRepr(InstanceRepr)`.
///
/// Pyre's `InstanceRepr` is a single Rust struct rather than a Python subclass
/// hierarchy. This carrier records the extra state introduced by
/// `rvirtualizable.py`; production redirected-field identity comes from
/// `pyre-jit-trace/src/virtualizable_spec.rs` and is installed in
/// `GraphTransformConfig` by the prepass.
///
/// The rest of the production path follows the same division as upstream:
/// the annotator stores `access_directly` / `fresh_virtualizable` on
/// `Variable.annotation` as `SomeInstance.flags`; the codewriter's
/// `Transformer.hook_access_field` performs the
/// `VirtualizableInstanceRepr.hook_access_field` step for every configured
/// redirected access; and `rewrite_op_jit_force_virtualizable` deletes the
/// residual force marker from looked-inside graphs. The compiled interpreter
/// retains that marker at the gateway, so residual execution forces while the
/// generated JIT path does not.
#[derive(Clone, Debug, Default)]
pub struct VirtualizableInstanceRepr {
    pub top_of_virtualizable_hierarchy: bool,
    /// Shared with `VTYPE._hints['virtualizable_accessor']` so
    /// `_parse_field_list` initializes the same object the hint names.
    pub accessor:
        std::sync::Arc<parking_lot::Mutex<crate::translator::rtyper::rclass::FieldListAccessor>>,
    pub my_redirected_fields: HashMap<String, bool>,
}

impl VirtualizableInstanceRepr {
    pub fn new(top_of_virtualizable_hierarchy: bool) -> Self {
        VirtualizableInstanceRepr {
            top_of_virtualizable_hierarchy,
            accessor: std::sync::Arc::new(parking_lot::Mutex::new(
                crate::translator::rtyper::rclass::FieldListAccessor::default(),
            )),
            my_redirected_fields: HashMap::new(),
        }
    }

    /// `VirtualizableInstanceRepr.__init__` after `InstanceRepr.__init__`.
    pub fn from_classdesc(
        classdesc: &crate::annotator::classdesc::ClassDesc,
    ) -> Result<Self, crate::translator::rtyper::error::TyperError> {
        if const_truthy(&classdesc.get_param("_virtualizable2_", None, true)) {
            return Err(crate::translator::rtyper::error::TyperError::message(
                "_virtualizable2_ is now called _virtualizable_, please rename".to_string(),
            ));
        }
        let own = classdesc.get_param("_virtualizable_", None, false);
        if const_truthy(&own) {
            let basedesc = classdesc.basedesc.clone();
            if let Some(base) = basedesc {
                let base_param = base.borrow().get_param("_virtualizable_", None, true);
                if !matches!(base_param, ConstValue::None) {
                    return Err(crate::translator::rtyper::error::TyperError::message(
                        "basedesc must not declare _virtualizable_".to_string(),
                    ));
                }
            }
            Ok(Self::new(true))
        } else {
            Ok(Self::new(false))
        }
    }

    /// RPython `_setup_repr_llfields`.
    pub fn setup_repr_llfields(&self) -> Vec<(String, LowLevelType)> {
        if self.top_of_virtualizable_hierarchy {
            vec![("vable_token".to_string(), lltype::GCREF.clone())]
        } else {
            vec![]
        }
    }

    /// RPython `hook_access_field`: force only redirected fields.
    pub fn should_force_field(&self, mangled_name: &str) -> bool {
        self.my_redirected_fields
            .get(mangled_name)
            .copied()
            .unwrap_or(false)
    }
}

/// RPython `replace_force_virtualizable_with_call(graphs, VTYPEPTR,
/// funcptr)`.
pub fn replace_force_virtualizable_with_call(
    graphs: &[Rc<std::cell::RefCell<FunctionGraph>>],
    VTYPEPTR: &LowLevelType,
    funcptr: &_ptr,
) -> usize {
    let c_funcptr = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(funcptr.clone())),
        LowLevelType::Ptr(Box::new(lltype::typeOf(funcptr))),
    ));
    let mut count = 0;
    for graph in graphs {
        for block in graph.borrow().iterblocks() {
            let mut block = block.borrow_mut();
            let mut newoplist = Vec::with_capacity(block.operations.len());
            for mut op in block.operations.drain(..) {
                if op.opname == "jit_force_virtualizable"
                    && op.args.first().and_then(hlvalue_concretetype).as_ref() == Some(VTYPEPTR)
                {
                    if op.args.last().is_some_and(flags_access_directly) {
                        // `continue` before the append: an `access_directly`
                        // read reaches the virtualizable without going through
                        // the JIT, so the op is dropped rather than rewritten.
                        // Nothing downstream understands the `jit_force_virtualizable`
                        // opname, so leaving it in the block is not a weaker
                        // rewrite but an unlowerable operation.
                        continue;
                    }
                    op.opname = "direct_call".to_string();
                    op.args = vec![c_funcptr.clone(), op.args[0].clone()];
                    count += 1;
                }
                newoplist.push(op);
            }
            block.operations = newoplist;
        }
    }
    count
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

fn flags_access_directly(value: &Hlvalue) -> bool {
    let Hlvalue::Constant(c) = value else {
        return false;
    };
    let ConstValue::Dict(items) = &c.value else {
        return false;
    };
    let key = ConstValue::byte_str("access_directly");
    matches!(items.get(&key), Some(ConstValue::Bool(true)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::Variable;
    use crate::flowspace::model::{Block, SpaceOperation};
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

    fn force_op(vable: &Variable, access_directly: bool) -> SpaceOperation {
        let mut flags = HashMap::new();
        if access_directly {
            flags.insert(
                ConstValue::byte_str("access_directly"),
                ConstValue::Bool(true),
            );
        }
        let c_name =
            Constant::with_concretetype(ConstValue::byte_str("inst_x"), LowLevelType::Void);
        let c_flags = Constant::with_concretetype(
            ConstValue::Dict(std::sync::Arc::new(flags)),
            LowLevelType::Void,
        );
        SpaceOperation::new(
            "jit_force_virtualizable",
            vec![
                Hlvalue::Variable(vable.clone()),
                Hlvalue::Constant(c_name),
                Hlvalue::Constant(c_flags),
            ],
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            )),
        )
    }

    fn graph_with_ops(ops: Vec<SpaceOperation>) -> Rc<std::cell::RefCell<FunctionGraph>> {
        let block = Block::shared(vec![]);
        block.borrow_mut().operations = ops;
        Rc::new(std::cell::RefCell::new(FunctionGraph::new("g", block)))
    }

    #[test]
    fn replace_force_virtualizable_rewrites_matching_non_direct_access() {
        let VTYPEPTR = lltype::GCREF.clone();
        let vable = Variable::named("vable");
        vable.set_concretetype(Some(VTYPEPTR.clone()));
        let graph = graph_with_ops(vec![force_op(&vable, false)]);
        let funcptr = functionptr(
            FuncType {
                args: vec![VTYPEPTR.clone()],
                result: LowLevelType::Void,
            },
            "force",
            None,
            Some("force".to_string()),
        );

        let count = replace_force_virtualizable_with_call(
            std::slice::from_ref(&graph),
            &VTYPEPTR,
            &funcptr,
        );

        assert_eq!(count, 1);
        let graph_ref = graph.borrow();
        let start = graph_ref.startblock.borrow();
        let op = &start.operations[0];
        assert_eq!(op.opname, "direct_call");
        assert_eq!(op.args.len(), 2);
        assert!(matches!(op.args[0], Hlvalue::Constant(_)));
        assert_eq!(op.args[1], Hlvalue::Variable(vable));
    }

    #[test]
    fn replace_force_virtualizable_drops_access_directly_ops() {
        let VTYPEPTR = lltype::GCREF.clone();
        let vable = Variable::named("vable");
        vable.set_concretetype(Some(VTYPEPTR.clone()));
        let graph = graph_with_ops(vec![force_op(&vable, true)]);
        let funcptr = functionptr(
            FuncType {
                args: vec![VTYPEPTR.clone()],
                result: LowLevelType::Void,
            },
            "force",
            None,
            Some("force".to_string()),
        );

        let count = replace_force_virtualizable_with_call(
            std::slice::from_ref(&graph),
            &VTYPEPTR,
            &funcptr,
        );

        assert_eq!(count, 0);
        assert!(graph.borrow().startblock.borrow().operations.is_empty());
    }

    #[test]
    fn setup_repr_llfields_adds_vable_token_only_at_hierarchy_root() {
        assert_eq!(
            VirtualizableInstanceRepr::new(true).setup_repr_llfields(),
            vec![("vable_token".to_string(), lltype::GCREF.clone())]
        );
        assert!(
            VirtualizableInstanceRepr::new(false)
                .setup_repr_llfields()
                .is_empty()
        );
    }

    #[test]
    fn from_classdesc_rejects_a_base_declaration_that_is_not_none() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::flowspace::model::HostObject;
        use std::rc::Rc;

        let bk = Rc::new(Bookkeeper::new());
        let mut base_members = indexmap::IndexMap::new();
        base_members.insert("_virtualizable_".into(), ConstValue::List(vec![]));
        let base = crate::annotator::classdesc::ClassDesc::new(
            &bk,
            HostObject::new_class_with_members("Base", vec![], base_members),
            Some("Base".into()),
            None,
            None,
        )
        .expect("base ClassDesc");
        let mut child_members = indexmap::IndexMap::new();
        child_members.insert(
            "_virtualizable_".into(),
            ConstValue::List(vec![ConstValue::byte_str("x")]),
        );
        let child = crate::annotator::classdesc::ClassDesc::new(
            &bk,
            HostObject::new_class_with_members("Child", vec![], child_members),
            Some("Child".into()),
            Some(base),
            None,
        )
        .expect("child ClassDesc");
        let err = VirtualizableInstanceRepr::from_classdesc(&child.borrow())
            .expect_err("empty-list base declaration is not None");
        assert!(
            err.to_string()
                .contains("basedesc must not declare _virtualizable_")
        );
    }
}
