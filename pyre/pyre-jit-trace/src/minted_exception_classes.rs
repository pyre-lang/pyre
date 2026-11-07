//! One immortal `PyType` per minted RPython exception class.
//!
//! The build script writes `$OUT_DIR/minted_exception_classes.rs` from
//! the harvested hierarchies. Placeholder extraction
//! (`MAJIT_LLBC_EXTRACTION=1`) and a program that minted nothing both
//! emit an empty table, so every build compiles. The table is a plain
//! static array: wasm does not use `linkme` for these rows.

use majit_gc::collector::MiniMarkGC;

/// One row of [`MINTED_EXCEPTION_CLASSES`].
///
/// `parent` indexes the parent row. `None` is a hierarchy base, whose
/// RPython parent is `HOST_EXCEPTION_TYPE`.
pub struct MintedExceptionClass {
    pub name: &'static str,
    pub pytype: &'static pyre_object::PyType,
    pub parent: Option<usize>,
}

include!(concat!(env!("OUT_DIR"), "/minted_exception_classes.rs"));

/// Register `rows` in parent-before-child order.
///
/// Each row is `TypeInfo::object_subclass` of `size_of::<PyObject>()`
/// with no GC-pointer offsets, the same shape `build_gc` uses for
/// `HOST_EXCEPTION_TYPE`. The tid exists so `subclass_range` can resolve
/// the class word through `vtable_to_type_id`. Instance allocation keeps
/// stamping the header tid of the layout it mallocs; this registration
/// does not replace that tid.
///
/// `parent` `None` uses `host_exc_tid`. A `Some` index must name an
/// earlier row. Returns the tid of each row, in order.
pub fn register_minted_exception_classes(
    gc: &mut MiniMarkGC,
    rows: &[MintedExceptionClass],
    host_exc_tid: u32,
) -> Vec<u32> {
    let mut tids = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let parent_tid = match row.parent {
            None => host_exc_tid,
            Some(parent) => {
                assert!(
                    parent < index,
                    "minted exception class `{}` parent {parent} is not an earlier row",
                    row.name
                );
                tids[parent]
            }
        };
        let tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            parent_tid,
        ));
        let pytype = row.pytype as *const pyre_object::PyType as usize;
        majit_gc::GcAllocator::register_vtable_for_type(gc, pytype, tid);
        tids.push(tid);
    }
    tids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_three_class_hierarchy_parent_before_child() {
        use majit_translate::front::llbc_hints::{
            ExceptionClassHint, minted_exception_class_rows, render_minted_exception_classes_rs,
        };

        let hint = ExceptionClassHint::new("synth::E", "B(A)", vec!["A".into(), "B".into()]);
        let rows = minted_exception_class_rows(std::slice::from_ref(&hint));
        assert_eq!(
            rows.iter()
                .map(|row| (row.name.as_str(), row.parent))
                .collect::<Vec<_>>(),
            vec![
                ("synth::E", None),
                ("synth::E::A", Some(0)),
                ("synth::E::B", Some(1)),
            ]
        );
        let src = render_minted_exception_classes_rs(&rows);
        // Each non-alphanumeric character is one `_`, so `::` is `__`.
        let base_at = src.find("static MINTED_EXC_synth__E:").expect("base ident");
        let a_at = src.find("static MINTED_EXC_synth__E__A:").expect("A ident");
        let b_at = src.find("static MINTED_EXC_synth__E__B:").expect("B ident");
        assert!(base_at < a_at && a_at < b_at);
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E\", pytype: &MINTED_EXC_synth__E, parent: None }"
        ));
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E::A\", pytype: &MINTED_EXC_synth__E__A, parent: Some(0) }"
        ));
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E::B\", pytype: &MINTED_EXC_synth__E__B, parent: Some(1) }"
        ));
        assert!(src.contains("pyre_object::pyobject::new_pytype(\"synth::E\")"));
        assert!(src.contains("pyre_object::pyobject::new_pytype(\"synth::E::A\")"));
        assert!(src.contains("pyre_object::pyobject::new_pytype(\"synth::E::B\")"));
    }

    /// `default_issubclass`: `cls_min <= subcls_min && subcls_min < cls_max`,
    /// else pointer equality. The predicate is private on the metainterp,
    /// so this applies it to a fresh collector's `subclass_range`.
    fn range_issubclass(
        gc: &MiniMarkGC,
        sub: &pyre_object::PyType,
        cls: &pyre_object::PyType,
    ) -> bool {
        let sub_ptr = sub as *const pyre_object::PyType as usize;
        let cls_ptr = cls as *const pyre_object::PyType as usize;
        match (
            majit_gc::GcAllocator::subclass_range(gc, cls_ptr),
            majit_gc::GcAllocator::subclass_range(gc, sub_ptr),
        ) {
            (Some((cls_min, cls_max)), Some((sub_min, _))) => {
                cls_min <= sub_min && sub_min < cls_max
            }
            _ => sub_ptr == cls_ptr,
        }
    }

    #[test]
    fn register_minted_hierarchy_matches_subclass_ranges() {
        static TEST_BASE: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::E");
        static TEST_A: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::E::A");
        static TEST_B: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::E::B");

        let rows = [
            MintedExceptionClass {
                name: "test::E",
                pytype: &TEST_BASE,
                parent: None,
            },
            MintedExceptionClass {
                name: "test::E::A",
                pytype: &TEST_A,
                parent: Some(0),
            },
            MintedExceptionClass {
                name: "test::E::B",
                pytype: &TEST_B,
                parent: Some(1),
            },
        ];
        let mut gc = MiniMarkGC::new();
        let object_tid = gc.register_type(majit_gc::trace::TypeInfo::object(std::mem::size_of::<
            pyre_object::PyObject,
        >()));
        let host_tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            object_tid,
        ));
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_interpreter::error::HOST_EXCEPTION_TYPE as *const _ as usize,
            host_tid,
        );
        let pyerror_tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            host_tid,
        ));
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_interpreter::error::PYERROR_TYPE as *const _ as usize,
            pyerror_tid,
        );
        let tids = register_minted_exception_classes(&mut gc, &rows, host_tid);
        assert_eq!(tids.len(), 3);
        majit_gc::GcAllocator::assign_inheritance_ids_now(&mut gc);

        let host = &pyre_interpreter::error::HOST_EXCEPTION_TYPE;
        let pyerror = &pyre_interpreter::error::PYERROR_TYPE;
        assert!(range_issubclass(&gc, &TEST_B, &TEST_A));
        assert!(range_issubclass(&gc, &TEST_A, &TEST_BASE));
        assert!(range_issubclass(&gc, &TEST_BASE, host));
        assert!(range_issubclass(&gc, &TEST_B, host));
        assert!(!range_issubclass(&gc, &TEST_A, &TEST_B));
        assert!(!range_issubclass(&gc, &TEST_BASE, pyerror));
        assert!(!range_issubclass(&gc, pyerror, &TEST_BASE));
    }
}
