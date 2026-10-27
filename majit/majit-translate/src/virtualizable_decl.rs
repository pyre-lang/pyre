//! The `_virtualizable_` class declaration, as a registered set of roots.
//!
//! RPython puts it on the class: `_virtualizable_ = ['x', 'y[*]']` is read
//! back through `classdesc.get_param('_virtualizable_')`, and
//! `rlib/jit.py`'s `hint` entry consults exactly that before it mints
//! `access_directly` on a `SomeInstance` — a value whose class does not
//! declare it has the flags deleted instead.
//!
//! Pyre's interpreter is hand-written Rust and no struct carries that
//! parameter, so the declaration is supplied out of band by the consumer, the
//! same way the codewriter's `GraphTransformConfig::vable_fields` is
//! (`rvirtualizable.rs` records why). This module is where it is read back:
//! the bookkeeper stamps it on the class host (`stamp_host_virtualizable`),
//! where the `hint` entry's class test finds it, and the front end asks it
//! which fields are arrays.
//!
//! Upstream the declaration is per-CLASS Bookkeeper state, so it is neither
//! thread- nor invocation-scoped: `get_param` asks the class every time.
//! Scoping it to an invocation is a pyre deviation forced by having no
//! `ClassDesc` at this point, and the way it is kept honest is that the
//! pipeline re-seeds this registry from its own `AnalyzeConfig` on every run
//! (`lib.rs analyze_pipeline_from_module_paths`), deriving the roots from the
//! `owner_root` the same config already puts on every
//! `VirtualizableFieldDescriptor`. One declaration channel, refreshed per
//! invocation — the shape `local_crates` has.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::codewriter::jtransform::{GraphTransformConfig, VirtualizableFieldDescriptor};
use crate::flowspace::model::{ConstValue, HostObject};
use crate::model::ValueType;
use crate::translator::rtyper::lltypesystem::lltype::{self, LowLevelType};
use crate::translator::rtyper::rvirtualizable::VirtualizableInstanceRepr;

thread_local! {
    /// Per-pipeline-invocation `_virtualizable_` roots, seeded by the
    /// consumer before it builds a program and read back by the front end
    /// and the bookkeeper during the SAME invocation.
    ///
    /// Thread-local for the reason `local_crates.rs` spells out: a translate
    /// pipeline runs start-to-finish on one thread, and a process-global
    /// would let a parallel `cargo test` pipeline overwrite this run's
    /// declaration between its own seed and read.
    static REGISTERED: RefCell<HashMap<String, Vec<String>>> = RefCell::new(HashMap::new());
}

/// Replace this thread's `_virtualizable_` declarations: each root with the
/// list its class declares (`name`, or `name[*]` for an array). A later
/// invocation on the same thread overwrites, matching the per-invocation
/// semantics of `local_crates::register_local_crate_roots`.
///
/// The production caller is the pipeline itself, which derives the set from
/// the `AnalyzeConfig` it was handed ([`declarations_from_config`]). Public for
/// a consumer that builds a program without going through
/// `analyze_pipeline_from_module_paths`; such a consumer owns the ordering,
/// since the read happens during the build.
pub fn register_virtualizable_declarations(
    declarations: impl IntoIterator<Item = (String, Vec<String>)>,
) {
    REGISTERED.with(|registered| {
        let mut registered = registered.borrow_mut();
        registered.clear();
        for (root, fields) in declarations {
            registered.entry(root).or_default().extend(fields);
        }
    });
}

/// The `_virtualizable_` list each owner root declares in `config`: static
/// fields, then arrays spelled `name[*]`, each in declared index order.
pub fn declarations_from_config(config: &GraphTransformConfig) -> Vec<(String, Vec<String>)> {
    let mut declarations: Vec<(String, Vec<String>)> = Vec::new();
    let mut add = |fields: &[VirtualizableFieldDescriptor], suffix: &str| {
        let mut fields: Vec<&VirtualizableFieldDescriptor> = fields.iter().collect();
        fields.sort_by_key(|field| field.index);
        for field in fields {
            let Some(root) = field.owner_root.as_ref() else {
                continue;
            };
            let name = format!("{}{suffix}", field.name);
            match declarations.iter_mut().find(|(known, _)| known == root) {
                Some((_, names)) => names.push(name),
                None => declarations.push((root.clone(), vec![name])),
            }
        }
    };
    add(&config.vable_fields, "");
    add(&config.vable_arrays, "[*]");
    declarations
}

/// `cls._virtualizable_ = [...]` (`interp_jit.py` assigns it on the frame
/// class) — stamp the class attribute on a newly interned host when this
/// invocation declared it.
pub fn stamp_host_virtualizable(host: &HostObject, class_key: &str) {
    let Some(fields) = field_names_for(class_key) else {
        return;
    };
    let items = fields
        .into_iter()
        .map(ConstValue::byte_str)
        .collect::<Vec<_>>();
    host.class_set("_virtualizable_", ConstValue::List(Arc::new(items)));
}

/// Whether `owner` is the class that declared `_virtualizable_`
/// (`VirtualizableInstanceRepr.top_of_virtualizable_hierarchy`).
pub fn is_virtualizable_hierarchy_root(owner: &str) -> bool {
    REGISTERED.with(|registered| lookup(&registered.borrow(), owner).is_some())
}

/// The extra llfield type `_setup_repr_llfields` assigns when `owner` is a
/// virtualizable hierarchy root and `field_name` is one of those fields.
/// Callers overlay this onto a host declaration so a virtualizable token is
/// `llmemory.GCREF` even when the Rust field is a machine word.
pub fn virtualizable_llfield_type(owner: &str, field_name: &str) -> Option<LowLevelType> {
    if !is_virtualizable_hierarchy_root(owner) {
        return None;
    }
    VirtualizableInstanceRepr::new(true)
        .setup_repr_llfields()
        .into_iter()
        .find(|(name, _)| name == field_name)
        .map(|(_, ty)| ty)
}

/// Overlay `_setup_repr_llfields` onto a host field's register class.
pub fn overlay_virtualizable_llfield_value_type(
    owner: &str,
    field_name: &str,
    host: ValueType,
) -> ValueType {
    match virtualizable_llfield_type(owner, field_name) {
        Some(ty) if ty == *lltype::GCREF => ValueType::Ref(None),
        _ => host,
    }
}

/// Overlay `_setup_repr_llfields` onto a host field's layout spelling.
/// `GCREF` is the lltype name `get_type_flag` banks as a GC pointer.
pub fn overlay_virtualizable_llfield_layout(owner: &str, field_name: &str, host: String) -> String {
    match virtualizable_llfield_type(owner, field_name) {
        Some(ty) if ty == *lltype::GCREF => "GCREF".to_string(),
        _ => host,
    }
}

/// Whether `owner`'s `_virtualizable_` list declares `field_name` as an array
/// (`name[*]`). The codewriter keys the same declaration; the front end asks
/// it so a safe `Index` of that field pairs with the array read.
pub fn is_declared_array_field(owner_root: Option<&str>, field_name: &str) -> bool {
    let Some(owner) = owner_root else {
        return false;
    };
    let needle = format!("{field_name}[*]");
    REGISTERED.with(|registered| {
        let registered = registered.borrow();
        lookup(&registered, owner).is_some_and(|fields| fields.iter().any(|field| field == &needle))
    })
}

fn field_names_for(class_key: &str) -> Option<Vec<String>> {
    let fields = REGISTERED.with(|registered| lookup(&registered.borrow(), class_key).cloned())?;
    if fields.is_empty() {
        return None;
    }
    Some(fields)
}

fn lookup<'a>(
    registered: &'a HashMap<String, Vec<String>>,
    class_key: &str,
) -> Option<&'a Vec<String>> {
    if let Some(fields) = registered.get(class_key) {
        return Some(fields);
    }
    // Directional, not `names_same_type`: a QUALIFIED lookup key may resolve a
    // shorter registered one, because the suffix is then the same type seen
    // without its path. The reverse is not true -- a bare `Frame` host key
    // suffix-matches a registered `a::Frame` that names a different class, and
    // resolving it would stamp `_virtualizable_` onto an unrelated one and
    // send it down the virtualizable lowering.
    let mut matches = registered.iter().filter(|(key, _)| {
        class_key
            .strip_suffix(key.as_str())
            .is_some_and(|prefix| prefix.ends_with("::"))
    });
    match (matches.next(), matches.next()) {
        (None, _) => None,
        (Some((_, fields)), None) => Some(fields),
        (Some((first, _)), Some((second, _))) => {
            panic!(
                "ambiguous _virtualizable_ for class key {class_key}: \
                 registered keys `{first}` and `{second}` both match"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare host key must not pick up a QUALIFIED registration: the two
    /// share a last segment and nothing else, and resolving it would stamp
    /// `_virtualizable_` onto a class that never declared it.
    #[test]
    fn a_bare_class_key_does_not_resolve_a_qualified_registration() {
        let mut registered = HashMap::new();
        registered.insert("a::Frame".to_string(), vec!["pc".to_string()]);
        assert!(lookup(&registered, "Frame").is_none());
        // The direction that IS a resolution: a qualified key naming the same
        // type as a registered shorter one.
        assert!(lookup(&registered, "x::a::Frame").is_some());
    }

    #[test]
    fn stamp_host_virtualizable_writes_the_declared_list() {
        register_virtualizable_declarations([(
            "Frame".to_string(),
            vec!["pc".to_string(), "stack[*]".to_string()],
        )]);
        let host = HostObject::new_class("Frame", vec![]);
        stamp_host_virtualizable(&host, "Frame");
        let ConstValue::List(items) = host
            .class_get("_virtualizable_")
            .expect("the declared class carries _virtualizable_")
        else {
            panic!("_virtualizable_ must be a list");
        };
        assert_eq!(
            items.as_slice(),
            [ConstValue::byte_str("pc"), ConstValue::byte_str("stack[*]")]
        );
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }

    #[test]
    fn declarations_from_config_lists_statics_then_arrays_per_root() {
        let config = GraphTransformConfig {
            vable_fields: vec![
                VirtualizableFieldDescriptor::new("b", Some("Frame".into()), 1),
                VirtualizableFieldDescriptor::new("a", Some("Frame".into()), 0),
                VirtualizableFieldDescriptor::new("loose", None, 2),
            ],
            vable_arrays: vec![VirtualizableFieldDescriptor::new(
                "items",
                Some("Frame".into()),
                0,
            )],
            ..GraphTransformConfig::default()
        };
        assert_eq!(
            declarations_from_config(&config),
            vec![(
                "Frame".to_string(),
                vec!["a".to_string(), "b".to_string(), "items[*]".to_string()]
            )]
        );
    }

    #[test]
    fn virtualizable_llfield_type_is_the_setup_repr_token_on_a_hierarchy_root() {
        register_virtualizable_declarations([("Frame".to_string(), vec!["pc".to_string()])]);
        let extra = VirtualizableInstanceRepr::new(true).setup_repr_llfields();
        assert_eq!(extra.len(), 1);
        let (token_name, token_ty) = &extra[0];
        assert_eq!(
            virtualizable_llfield_type("Frame", token_name).as_ref(),
            Some(token_ty)
        );
        assert_eq!(
            overlay_virtualizable_llfield_value_type("Frame", token_name, ValueType::Unsigned),
            ValueType::Ref(None)
        );
        assert_eq!(
            overlay_virtualizable_llfield_layout("Frame", token_name, "usize".into()),
            "GCREF"
        );
        assert!(virtualizable_llfield_type("Frame", "pc").is_none());
        assert!(virtualizable_llfield_type("Plain", token_name).is_none());
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }

    #[test]
    fn is_declared_array_field_matches_the_star_entry_only() {
        register_virtualizable_declarations([(
            "GrainFrame".to_string(),
            vec!["scope".to_string(), "operand_words[*]".to_string()],
        )]);
        assert!(is_declared_array_field(Some("GrainFrame"), "operand_words"));
        assert!(is_declared_array_field(
            Some("vm::GrainFrame"),
            "operand_words"
        ));
        assert!(!is_declared_array_field(Some("GrainFrame"), "scope"));
        assert!(!is_declared_array_field(Some("Other"), "operand_words"));
        assert!(!is_declared_array_field(None, "operand_words"));
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }

    #[test]
    fn stamp_host_virtualizable_skips_undeclared_classes() {
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
        let host = HostObject::new_class("Plain", vec![]);
        stamp_host_virtualizable(&host, "Plain");
        assert!(host.class_get("_virtualizable_").is_none());
    }

    #[test]
    fn stamp_host_virtualizable_skips_a_root_with_no_fields() {
        register_virtualizable_declarations([("OtherFrame".to_string(), Vec::new())]);
        let host = HostObject::new_class("OtherFrame", vec![]);
        stamp_host_virtualizable(&host, "OtherFrame");
        assert!(host.class_get("_virtualizable_").is_none());
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }

    #[test]
    fn lookup_does_not_share_a_qualified_declaration_across_leaf_names() {
        REGISTERED.with(|registered| {
            *registered.borrow_mut() = [("a::Frame".to_string(), vec!["x".to_string()])]
                .into_iter()
                .collect();
        });
        let host = HostObject::new_class("Frame", vec![]);
        stamp_host_virtualizable(&host, "b::Frame");
        assert!(host.class_get("_virtualizable_").is_none());
        stamp_host_virtualizable(&host, "mod::a::Frame");
        let ConstValue::List(items) = host
            .class_get("_virtualizable_")
            .expect("qualified suffix of the registered key stamps")
        else {
            panic!("_virtualizable_ must be a list");
        };
        assert_eq!(items.as_slice(), [ConstValue::byte_str("x")]);
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }

    #[test]
    #[should_panic(expected = "ambiguous _virtualizable_ for class key x::a::Frame")]
    fn lookup_panics_when_two_suffixes_match() {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
            }
        }
        let _restore = Restore;
        REGISTERED.with(|registered| {
            *registered.borrow_mut() = [
                ("Frame".to_string(), vec!["leaf".to_string()]),
                ("a::Frame".to_string(), vec!["nested".to_string()]),
            ]
            .into_iter()
            .collect();
        });
        let host = HostObject::new_class("Frame", vec![]);
        stamp_host_virtualizable(&host, "x::a::Frame");
    }
}
