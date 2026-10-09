//! `pypy/objspace/std/typeobject.py`: declaration-driven builtin TypeCache.

use crate::baseobjspace::{ObjSpace, SpaceCache, SpaceCacheBuild};
use majit_rlib::cache::CacheError;
use pyre_object::typedef::{TypeDef, TypeDefValue};
use pyre_object::*;

/// `space.newtext_or_none` (`objspace.py`) for a TypeDef.doc.
fn newtext_or_none(doc: Option<&str>) -> PyObjectRef {
    match doc {
        Some(text) => w_str_new(text),
        None => w_none(),
    }
}

pub struct TypeCache {
    base: SpaceCache<usize, usize, crate::baseobjspace::SpaceHandle>,
}

impl TypeCache {
    pub const fn new(space: crate::baseobjspace::SpaceHandle) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    /// # Safety
    /// The definition and its bases are immutable, live prebuilt TypeDefs.
    pub unsafe fn getorbuild(
        &self,
        definition: *const TypeDef,
    ) -> Result<PyObjectRef, CacheError<crate::PyError>> {
        self.base
            .getorbuild(definition as usize, self)
            .map(|value| value as PyObjectRef)
    }

    pub fn walk_roots(&self, forward: &mut dyn FnMut(&mut PyObjectRef)) {
        self.base.visit_values_mut(|value| {
            let mut w_value = *value as PyObjectRef;
            forward(&mut w_value);
            *value = w_value as usize;
        });
    }

    /// TypeCache.build: allocate identity, compute bases, wrap declarations,
    /// initialize, then qualify functions. Cache publication/ready is outside
    /// this method, in SpaceCache's inherited getorbuild lifecycle.
    #[majit_macros::not_rpython]
    unsafe fn build(
        &self,
        definition: *const TypeDef,
    ) -> Result<PyObjectRef, CacheError<crate::PyError>> {
        let space = self.base.space.get();
        let definition = unsafe { &*definition };
        let name = definition
            .name
            .as_deref()
            .expect("layout-only TypeDef has no declarations");
        let w_type = w_type_alloc_builtin();
        let _roots = gc_roots::push_roots();
        let w_type_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(w_type);
        let bases_start = gc_roots::shadow_stack_len();
        if definition.bases.is_empty() {
            // The root object's declaration still belongs to the legacy
            // bootstrap. Do not infer a declaration key from its Layout.
            let object = crate::typedef::w_object();
            assert!(
                !object.is_null(),
                "root TypeDef bootstrap must precede derived declarations"
            );
            let _ = gc_roots::pin_root(object);
        } else {
            for &base in &definition.bases {
                let w_base = unsafe { space.gettypeobject(base)? };
                let _ = gc_roots::pin_root(w_base);
            }
        }
        let bases: Vec<_> = (bases_start..gc_roots::shadow_stack_len())
            .map(gc_roots::shadow_stack_get)
            .collect();
        let bases_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(w_tuple_new(bases));
        let ns_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(w_dict_new());
        // typeobject.py TypeCache.build: overridetypedef is
        // `typedef.applevel_subclasses_base.typedef` or `typedef`.
        // Compute it before `ensure_common_attributes` so
        // `setup_builtin_type` can seed `w_doc` and
        // `dict_w.setdefault('__doc__', w_doc)` from the override.
        let overridetypedef = if definition.applevel_subclasses_base.is_null() {
            definition as *const TypeDef
        } else {
            definition.applevel_subclasses_base
        };
        let override_def = unsafe { &*overridetypedef };
        let w_override_doc = newtext_or_none(override_def.doc.as_deref());
        let w_override_doc = gc_roots::pin_root(w_override_doc);
        unsafe {
            w_type_set_w_doc(gc_roots::shadow_stack_get(w_type_slot), w_override_doc);
        }
        for (name, value) in &definition.rawdict {
            let value = match value {
                TypeDefValue::Text(text) => w_str_new(text),
                TypeDefValue::None => w_none(),
                TypeDefValue::Root(slot) => {
                    let value = unsafe { *slot.get() };
                    if unsafe { pyre_object::typedef::is_getset_property(value) } {
                        gc_roots::pin_root(crate::typedef::copy_for_type(
                            value,
                            gc_roots::shadow_stack_get(w_type_slot),
                        ))
                    } else if unsafe { pyre_object::gateway::is_interp2app(value) } {
                        crate::gateway::interp2app_spacebind(value, &space)
                    } else {
                        value
                    }
                }
            };
            unsafe {
                w_dict_setitem_str_no_proxy(gc_roots::shadow_stack_get(ns_slot), name, value)
            };
        }
        if let Some(owner) = crate::typedef::method_owner(name) {
            unsafe {
                crate::typedef::stamp_method_owners(gc_roots::shadow_stack_get(ns_slot), owner)
            };
        }
        crate::typedef::init_builtin_typeobject(
            gc_roots::shadow_stack_get(w_type_slot),
            name,
            gc_roots::shadow_stack_get(bases_slot),
            gc_roots::shadow_stack_get(ns_slot),
            crate::typedef::w_type(),
        );
        unsafe {
            // setup_builtin_type reads hasdict/weakrefable/heaptype off
            // instancetypedef (the override), not the derived declaration.
            w_type_set_hasdict(
                gc_roots::shadow_stack_get(w_type_slot),
                override_def.hasdict,
            );
            w_type_set_weakrefable(
                gc_roots::shadow_stack_get(w_type_slot),
                override_def.weakrefable,
            );
            w_type_set_flag_sequence_bug_compat(
                gc_roots::shadow_stack_get(w_type_slot),
                definition.flag_sequence_bug_compat,
            );
            w_type_set_heaptype(
                gc_roots::shadow_stack_get(w_type_slot),
                override_def.heaptype,
            );
            if let Some(signature) = &override_def.text_signature {
                // setup_builtin_type: `w_self.text_signature =
                // instancetypedef.text_signature`.
                w_type_set_text_signature(gc_roots::shadow_stack_get(w_type_slot), signature);
            }
            // `W_TypeObject.acceptable_as_base_class` reads
            // `self.layout.typedef.acceptable_as_base_class` — the override
            // declaration, not the derived one.
            w_type_set_acceptable_as_base_class(
                gc_roots::shadow_stack_get(w_type_slot),
                override_def.acceptable_as_base_class(),
            );
            crate::baseobjspace::compute_and_set_mro(gc_roots::shadow_stack_get(w_type_slot))
                .map_err(CacheError::Build)?;
            let best = crate::call::find_best_base(gc_roots::shadow_stack_get(bases_slot))
                .map_err(CacheError::Build)?;
            let parent_layout = if best.is_null() {
                std::ptr::null()
            } else {
                w_type_get_layout_ptr(best)
            };
            let layout = if !parent_layout.is_null()
                && std::ptr::eq((*parent_layout).typedef, overridetypedef)
            {
                parent_layout
            } else {
                typeobject::leak_layout(typeobject::Layout {
                    typedef: overridetypedef,
                    nslots: 0,
                    newslotnames: vec![],
                    base_layout: parent_layout,
                    dict_data_slot: typeobject::DICT_DATA_SLOT_UNRESOLVED,
                })
            };
            w_type_set_layout(gc_roots::shadow_stack_get(w_type_slot), layout);
            if std::ptr::eq(definition as *const TypeDef, overridetypedef) {
                // TypeCache.build's else arm: qualify member functions.
                crate::typedef::stamp_new_descr_self(
                    gc_roots::shadow_stack_get(ns_slot),
                    gc_roots::shadow_stack_get(w_type_slot),
                );
            } else {
                // `typedef is not overridetypedef`: skip the
                // qualname/objclass pass.  `setup_builtin_type` already
                // seeded `w_doc` and `dict_w.setdefault('__doc__', w_doc)`
                // from the override; overwrite only the type-level slot
                // (`typeobject.py TypeCache.build`).
                let w_derived_doc = newtext_or_none(definition.doc.as_deref());
                let w_derived_doc = gc_roots::pin_root(w_derived_doc);
                w_type_set_w_doc(gc_roots::shadow_stack_get(w_type_slot), w_derived_doc);
                crate::typedef::ensure_static_new(
                    gc_roots::shadow_stack_get(ns_slot),
                    gc_roots::shadow_stack_get(w_type_slot),
                );
            }
        }
        Ok(gc_roots::shadow_stack_get(w_type_slot))
    }
}

impl SpaceCacheBuild<usize, usize> for TypeCache {
    type Error = crate::PyError;

    fn build(&self, key: &usize) -> Result<usize, CacheError<Self::Error>> {
        unsafe { TypeCache::build(self, *key as *const TypeDef).map(|value| value as usize) }
    }

    fn ready(&self, value: &usize) -> Result<(), CacheError<Self::Error>> {
        unsafe { typeobject::w_type_ready(*value as PyObjectRef) };
        gc_roots::mark_prebuilt_roots_dirty();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn answer(_: &[PyObjectRef]) -> crate::PyResult {
        Ok(w_int_new(42))
    }

    #[test]
    fn declarations_keep_typedef_identity_and_bind_functions_per_space() {
        crate::typedef::init_typeobjects();
        let _roots = gc_roots::push_roots();
        let gateway =
            crate::gateway::interp2app(crate::gateway::builtin_code_new("original", answer));
        let gateway_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(gateway);
        unsafe {
            let definition = TypeDef::from_rawdict(
                "Declared",
                vec![],
                IndexMap::from([
                    (
                        "method".into(),
                        TypeDefValue::root(gc_roots::shadow_stack_get(gateway_slot)),
                    ),
                    (
                        "__doc__".into(),
                        TypeDefValue::Text("declaration doc".into()),
                    ),
                ]),
                &INSTANCE_TYPE,
            );
            let gateway = &*(gc_roots::shadow_stack_get(gateway_slot)
                as *const pyre_object::gateway::interp2app);
            assert_eq!(gateway.name, "method");
            assert!(gateway._is_type_method);
            let a = ObjSpace::new();
            let b = ObjSpace::new();
            let first = a.gettypeobject(definition).unwrap();
            assert_eq!(first, a.gettypeobject(definition).unwrap());
            let other = b.gettypeobject(definition).unwrap();
            assert_ne!(first, other);
            assert_eq!((*w_type_get_layout_ptr(first)).typedef, definition);
            assert_eq!((*w_type_get_layout_ptr(other)).typedef, definition);
            let first_ns = w_type_get_dict_ptr(first) as PyObjectRef;
            let other_ns = w_type_get_dict_ptr(other) as PyObjectRef;
            assert_ne!(
                w_dict_getitem_str(first_ns, "method"),
                w_dict_getitem_str(other_ns, "method")
            );
            assert_eq!(
                w_str_get_wtf8(w_dict_getitem_str(first_ns, "__doc__").unwrap()),
                "declaration doc"
            );
            assert!(!w_type_get_mro(first).is_null());
            let child =
                TypeDef::from_rawdict("Child", vec![definition], IndexMap::new(), &INSTANCE_TYPE);
            let child = a.gettypeobject(child).unwrap();
            assert_eq!(w_tuple_getitem(w_type_get_bases(child), 0), Some(first));
        }
    }

    #[test]
    fn sequence_bug_compat_is_a_typeobject_field_not_an_identity_table() {
        crate::typedef::init_typeobjects();
        unsafe {
            for (tp, expect) in [
                (&LIST_TYPE, true),
                (&TUPLE_TYPE, true),
                (&STR_TYPE, true),
                (&pyre_object::bytesobject::BYTES_TYPE, true),
                (&pyre_object::bytearrayobject::BYTEARRAY_TYPE, true),
                (&DICT_TYPE, false),
                (&INT_TYPE, false),
            ] {
                let w_type = crate::typedef::gettypefor(tp).unwrap().as_ptr();
                assert_eq!(
                    crate::baseobjspace::flag_sequence_bug_compat(w_type),
                    expect,
                    "{}",
                    (*tp).name
                );
            }
            let definition =
                TypeDef::from_rawdict("NotASequence", vec![], IndexMap::new(), &INSTANCE_TYPE);
            let w_type = ObjSpace::new().gettypeobject(definition).unwrap();
            assert!(!crate::baseobjspace::flag_sequence_bug_compat(w_type));
            let flagged =
                TypeDef::from_rawdict("CompatSequence", vec![], IndexMap::new(), &INSTANCE_TYPE)
                    as *mut TypeDef;
            (*flagged).flag_sequence_bug_compat = true;
            let flagged = ObjSpace::new().gettypeobject(flagged).unwrap();
            assert!(crate::baseobjspace::flag_sequence_bug_compat(flagged));
        }
    }

    #[test]
    fn typecache_build_copies_heaptype_and_acceptable_as_base_class() {
        crate::typedef::init_typeobjects();
        unsafe {
            let definition =
                TypeDef::from_rawdict("NoNew", vec![], IndexMap::new(), &INSTANCE_TYPE);
            let w_type = ObjSpace::new().gettypeobject(definition).unwrap();
            assert!(!w_type_is_heaptype(w_type));
            assert!(!(*definition).acceptable_as_base_class());
            assert!(!w_type_get_acceptable_as_base_class(w_type));

            let heap = TypeDef::from_rawdict("HeapDecl", vec![], IndexMap::new(), &INSTANCE_TYPE)
                as *mut TypeDef;
            (*heap).heaptype = true;
            (*heap).set_acceptable_as_base_class(true);
            let heap = ObjSpace::new().gettypeobject(heap).unwrap();
            assert!(w_type_is_heaptype(heap));
            assert!(w_type_get_acceptable_as_base_class(heap));
        }
    }

    #[test]
    fn typecache_build_reuses_applevel_subclasses_base_layout() {
        crate::typedef::init_typeobjects();
        unsafe {
            let space = ObjSpace::new();
            let base_def = TypeDef::from_rawdict("IOBase", vec![], IndexMap::new(), &INSTANCE_TYPE);
            let base = space.gettypeobject(base_def).unwrap();
            let derived_def =
                TypeDef::from_rawdict("RawIO", vec![base_def], IndexMap::new(), &INSTANCE_TYPE)
                    as *mut TypeDef;
            (*derived_def).applevel_subclasses_base = base_def;
            let derived = space.gettypeobject(derived_def).unwrap();
            assert!(std::ptr::eq(
                w_type_get_layout_ptr(derived),
                w_type_get_layout_ptr(base)
            ));
            assert_eq!((*w_type_get_layout_ptr(derived)).typedef, base_def);
        }
    }

    #[test]
    fn derived_declaration_keeps_override_doc_in_the_namespace() {
        crate::typedef::init_typeobjects();
        unsafe {
            let base_def = TypeDef::from_rawdict(
                "DocBase",
                vec![],
                IndexMap::from([("__doc__".into(), TypeDefValue::Text("override doc".into()))]),
                &INSTANCE_TYPE,
            ) as *mut TypeDef;
            let space = ObjSpace::new();
            let _base = space.gettypeobject(base_def).unwrap();
            let derived_def = TypeDef::from_rawdict(
                "DocDerived",
                vec![base_def],
                IndexMap::new(),
                &INSTANCE_TYPE,
            ) as *mut TypeDef;
            (*derived_def).applevel_subclasses_base = base_def;
            let derived = space.gettypeobject(derived_def).unwrap();
            // `TypeCache.build`: dict keeps the override via setdefault;
            // `w_doc` is the derived declaration (absent → None).
            let ns = w_type_get_dict_ptr(derived) as PyObjectRef;
            assert_eq!(
                w_str_get_wtf8(w_dict_getitem_str(ns, "__doc__").unwrap()),
                "override doc"
            );
            assert!(is_none(w_type_get_w_doc(derived)));
        }
    }

    #[test]
    fn derived_declaration_doc_overwrites_only_the_type_slot() {
        crate::typedef::init_typeobjects();
        unsafe {
            let base_def = TypeDef::from_rawdict(
                "DocBaseOwn",
                vec![],
                IndexMap::from([("__doc__".into(), TypeDefValue::Text("override doc".into()))]),
                &INSTANCE_TYPE,
            ) as *mut TypeDef;
            let space = ObjSpace::new();
            let _base = space.gettypeobject(base_def).unwrap();
            let derived_def = TypeDef::from_rawdict(
                "DocDerivedOwn",
                vec![base_def],
                IndexMap::from([("__doc__".into(), TypeDefValue::Text("derived doc".into()))]),
                &INSTANCE_TYPE,
            ) as *mut TypeDef;
            (*derived_def).applevel_subclasses_base = base_def;
            let derived = space.gettypeobject(derived_def).unwrap();
            let ns = w_type_get_dict_ptr(derived) as PyObjectRef;
            assert_eq!(
                w_str_get_wtf8(w_dict_getitem_str(ns, "__doc__").unwrap()),
                "derived doc"
            );
            assert_eq!(w_str_get_wtf8(w_type_get_w_doc(derived)), "derived doc");
        }
    }

    #[test]
    fn process_wide_space_is_the_prebuilt_static() {
        let a = crate::baseobjspace::object_space();
        let b = crate::baseobjspace::object_space();
        assert!(std::ptr::eq(a, b));
        assert!(std::ptr::eq(a, &crate::baseobjspace::OBJECT_SPACE));
    }

    #[test]
    fn production_iterator_uses_the_declared_typedef_cache_entry() {
        crate::typedef::init_typeobjects();
        let definition = crate::objspace::std::iterobject::typedef();
        unsafe {
            let cached = crate::baseobjspace::object_space()
                .gettypeobject(definition)
                .unwrap();
            let registered = crate::typedef::gettypefor(&pyre_object::iterobject::SEQ_ITER_TYPE)
                .unwrap()
                .as_ptr();
            assert_eq!(cached, registered);
            assert_eq!((*w_type_get_layout_ptr(cached)).typedef, definition);
            assert!(!(*definition).acceptable_as_base_class());
        }
    }

    #[test]
    fn reverse_iterator_is_built_from_its_live_declaration() {
        crate::typedef::init_typeobjects();
        unsafe {
            let registered = crate::typedef::gettypefor(&LIST_REVERSE_ITER_TYPE)
                .unwrap()
                .as_ptr();
            let definition = (*w_type_get_layout_ptr(registered)).typedef;
            assert_eq!((*definition).name.as_deref(), Some("list_reverseiterator"));
            assert_eq!((*definition).rawdict.len(), 5);
            assert!(!(*definition).acceptable_as_base_class());
            let global = crate::baseobjspace::object_space();
            assert_eq!(global.gettypeobject(definition).unwrap(), registered);
            let other = ObjSpace::new().gettypeobject(definition).unwrap();
            assert_ne!(other, registered);
            assert_eq!((*w_type_get_layout_ptr(other)).typedef, definition);
            for name in [
                "__iter__",
                "__next__",
                "__reduce__",
                "__setstate__",
                "__length_hint__",
            ] {
                assert!(matches!(
                    (*definition).rawdict.get(name),
                    Some(TypeDefValue::Root(_))
                ));
                let first =
                    w_dict_getitem_str(w_type_get_dict_ptr(registered).cast(), name).unwrap();
                let second = w_dict_getitem_str(w_type_get_dict_ptr(other).cast(), name).unwrap();
                assert_ne!(first, second, "{name} must bind per space");
            }
            // Invoke the actual type-dictionary Functions, not baseobjspace's
            // separate getattr shortcut. The migrated declaration must execute.
            let _roots = gc_roots::push_roots();
            // __reduce__ resolves the live builtin through the execution
            // context, just as W_ReverseSeqIterObject.descr_reduce does.
            let context = Box::new(crate::PyExecutionContext::new());
            struct RestoreContext(*const crate::PyExecutionContext);
            impl Drop for RestoreContext {
                fn drop(&mut self) {
                    crate::call::set_last_exec_ctx(self.0);
                }
            }
            let _context = RestoreContext(crate::call::getexecutioncontext());
            crate::call::set_last_exec_ctx(&*context);
            let iter_slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(w_list_reverse_iter_new(w_list_new_range(1, 1, 3), 2));
            let invoke = |name: &str, args: &[PyObjectRef]| {
                let function =
                    w_dict_getitem_str(w_type_get_dict_ptr(registered).cast(), name).unwrap();
                crate::call::call_function_impl_result(function, args).unwrap()
            };
            assert_eq!(
                invoke("__iter__", &[gc_roots::shadow_stack_get(iter_slot)]),
                gc_roots::shadow_stack_get(iter_slot)
            );
            assert_eq!(
                w_int_get_value(invoke(
                    "__length_hint__",
                    &[gc_roots::shadow_stack_get(iter_slot)]
                )),
                3
            );
            assert_eq!(
                w_int_get_value(invoke("__next__", &[gc_roots::shadow_stack_get(iter_slot)])),
                3
            );
            let reduction_slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(invoke(
                "__reduce__",
                &[gc_roots::shadow_stack_get(iter_slot)],
            ));
            let reduction = gc_roots::shadow_stack_get(reduction_slot);
            assert_eq!(w_tuple_len(reduction), 3);
            assert_eq!(w_int_get_value(w_tuple_getitem(reduction, 2).unwrap()), 1);
            let restored = crate::baseobjspace::call(
                w_tuple_getitem(reduction, 0).unwrap(),
                w_tuple_getitem(reduction, 1).unwrap(),
                None,
            );
            assert!(!restored.is_null(), "{:?}", crate::call::take_call_error());
            let restored_slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(restored);
            let state = w_tuple_getitem(gc_roots::shadow_stack_get(reduction_slot), 2).unwrap();
            invoke(
                "__setstate__",
                &[gc_roots::shadow_stack_get(restored_slot), state],
            );
            assert_eq!(
                w_int_get_value(invoke(
                    "__next__",
                    &[gc_roots::shadow_stack_get(restored_slot)]
                )),
                2
            );
            assert_eq!(
                w_int_get_value(invoke(
                    "__next__",
                    &[gc_roots::shadow_stack_get(restored_slot)]
                )),
                1
            );
        }
    }

    #[test]
    fn sequence_iterator_types_own_distinct_next_descriptors() {
        crate::typedef::init_typeobjects();
        unsafe {
            let types = [
                &STR_ASCII_ITER_TYPE,
                &STR_ITER_TYPE,
                &BYTES_ITER_TYPE,
                &BYTEARRAY_ITER_TYPE,
            ];
            let nexts: Vec<PyObjectRef> = types
                .iter()
                .map(|tp| {
                    let w_type = crate::typedef::gettypefor(*tp).unwrap().as_ptr();
                    w_dict_getitem_str(w_type_get_dict_ptr(w_type).cast(), "__next__").unwrap()
                })
                .collect();
            for i in 0..nexts.len() {
                for j in (i + 1)..nexts.len() {
                    assert_ne!(
                        nexts[i], nexts[j],
                        "__next__ of {} and {} must be distinct objects",
                        types[i].name, types[j].name
                    );
                }
            }
            let ascii_type = crate::typedef::gettypefor(&STR_ASCII_ITER_TYPE)
                .unwrap()
                .as_ptr();
            assert_eq!(
                crate::function::fget_func_objclass(nexts[0]).unwrap(),
                ascii_type
            );
            let _roots = gc_roots::push_roots();
            let seq = gc_roots::pin_root(w_str_new("ab"));
            let iter_slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(w_seq_iter_new(seq, 2));
            assert!(std::ptr::eq(
                (*gc_roots::shadow_stack_get(iter_slot)).ob_type,
                &STR_ASCII_ITER_TYPE
            ));
            let first = crate::call::call_function_impl_result(
                nexts[0],
                &[gc_roots::shadow_stack_get(iter_slot)],
            )
            .unwrap();
            assert_eq!(w_str_get_wtf8(first), "a");
        }
    }

    #[test]
    fn fixed_pycode_function_gets_typecache_metadata() {
        crate::typedef::init_typeobjects();
        let code = crate::compile::compile_source("42", crate::compile::Mode::Eval).unwrap();
        let _roots = gc_roots::push_roots();
        let code_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(crate::w_code_new(Box::into_raw(Box::new(code)).cast()));
        let globals = w_module_dict_new();
        let function = crate::function::function_new_with_fixed_code(
            gc_roots::shadow_stack_get(code_slot).cast(),
            "original".into(),
            globals,
        );
        let function_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(function);
        unsafe {
            assert!(crate::function::is_function_with_fixed_code(
                gc_roots::shadow_stack_get(function_slot)
            ));
            assert!(!crate::gateway::is_builtin_code(crate::function::getcode(
                gc_roots::shadow_stack_get(function_slot)
            )));
            let definition = TypeDef::from_rawdict(
                "FixedCodeOwner",
                vec![],
                IndexMap::from([(
                    "method".into(),
                    TypeDefValue::root(gc_roots::shadow_stack_get(function_slot)),
                )]),
                &INSTANCE_TYPE,
            );
            let w_type = ObjSpace::new().gettypeobject(definition).unwrap();
            let function =
                w_dict_getitem_str(w_type_get_dict_ptr(w_type).cast(), "method").unwrap();
            assert_eq!(
                crate::function::function_get_qualname(function)
                    .as_str()
                    .unwrap(),
                "FixedCodeOwner.method"
            );
            assert_eq!(
                crate::function::fget_func_objclass(function).unwrap(),
                w_type
            );
        }
    }

    #[test]
    fn derived_declaration_skips_the_qualname_pass_and_reads_override_flags() {
        crate::typedef::init_typeobjects();
        let code = crate::compile::compile_source("42", crate::compile::Mode::Eval).unwrap();
        let _roots = gc_roots::push_roots();
        let code_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(crate::w_code_new(Box::into_raw(Box::new(code)).cast()));
        let globals = w_module_dict_new();
        let function = crate::function::function_new_with_fixed_code(
            gc_roots::shadow_stack_get(code_slot).cast(),
            "original".into(),
            globals,
        );
        let function_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(function);
        unsafe {
            let base_def =
                TypeDef::from_rawdict("FlagBase", vec![], IndexMap::new(), &INSTANCE_TYPE)
                    as *mut TypeDef;
            (*base_def).set_acceptable_as_base_class(true);
            (*base_def).text_signature = Some("(x, /)".into());
            let space = ObjSpace::new();
            let base = space.gettypeobject(base_def).unwrap();
            assert!(!base.is_null());
            let derived_def = TypeDef::from_rawdict(
                "FlagDerived",
                vec![base_def],
                IndexMap::from([(
                    "method".into(),
                    TypeDefValue::root(gc_roots::shadow_stack_get(function_slot)),
                )]),
                &INSTANCE_TYPE,
            ) as *mut TypeDef;
            (*derived_def).applevel_subclasses_base = base_def;
            let w_type = space.gettypeobject(derived_def).unwrap();
            // `typedef is not overridetypedef`: the qualname/objclass pass is
            // skipped, so the function keeps its own name and objclass.
            let function =
                w_dict_getitem_str(w_type_get_dict_ptr(w_type).cast(), "method").unwrap();
            assert_eq!(
                crate::function::function_get_qualname(function)
                    .as_str()
                    .unwrap(),
                "original"
            );
            // `acceptable_as_base_class` and `text_signature` read the
            // override declaration (`layout.typedef` / `setup_builtin_type`).
            assert!(w_type_get_acceptable_as_base_class(w_type));
            assert_eq!(w_type_get_text_signature(w_type), Some("(x, /)"));
        }
    }

    #[test]
    fn getset_template_is_copied_for_each_cached_type_identity() {
        crate::typedef::init_typeobjects();
        let _roots = gc_roots::push_roots();
        let property = pyre_object::typedef::w_getset_property_new(
            PY_NULL,
            PY_NULL,
            PY_NULL,
            PY_NULL,
            PY_NULL,
            false,
            w_str_new("template"),
        );
        let property_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(property);
        unsafe {
            let definition = TypeDef::from_rawdict(
                "PropertyOwner",
                vec![],
                IndexMap::from([(
                    "field".into(),
                    TypeDefValue::root(gc_roots::shadow_stack_get(property_slot)),
                )]),
                &INSTANCE_TYPE,
            );
            let a = ObjSpace::new();
            let b = ObjSpace::new();
            for space in [&a, &b] {
                let w_type = space.gettypeobject(definition).unwrap();
                let ns = w_type_get_dict_ptr(w_type) as PyObjectRef;
                let bound = w_dict_getitem_str(ns, "field").unwrap();
                assert_ne!(bound, gc_roots::shadow_stack_get(property_slot));
                assert_eq!(w_getset_get_objclass(bound), w_type);
            }
            assert!(w_getset_get_objclass(gc_roots::shadow_stack_get(property_slot)).is_null());
        }
    }
}
