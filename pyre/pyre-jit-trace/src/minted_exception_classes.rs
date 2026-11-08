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
/// RPython parent is `HOST_EXCEPTION_TYPE`. `destructor` frees that
/// class's raw-owned attribute boxes. `None` means the class owns none.
pub struct MintedExceptionClass {
    pub name: &'static str,
    pub pytype: &'static pyre_object::PyType,
    pub parent: Option<usize>,
    pub destructor: Option<majit_gc::trace::DestructorFn>,
}

/// Drop one raw-owned attribute of a minted exception instance.
///
/// `offset` is the `*mut T` slot in the instance. A null slot is already
/// clear. The slot is nulled before `Box::from_raw`, so a second call on
/// the same address is a no-op. The box owns the moved value.
pub unsafe fn drop_raw_owned_attr<T>(obj_addr: usize, offset: usize) {
    let slot = obj_addr.wrapping_add(offset) as *mut *mut T;
    let ptr = unsafe { slot.read() };
    if ptr.is_null() {
        return;
    }
    unsafe {
        slot.write(std::ptr::null_mut());
    }
    drop(unsafe { Box::from_raw(ptr) });
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
/// earlier row. A row with a destructor attaches it before registration
/// (`young_objects_with_destructors` reads it at alloc) and again with
/// `set_destructor`. Returns the tid of each row, in order.
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
        let mut info = majit_gc::trace::TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            parent_tid,
        );
        if let Some(destructor) = row.destructor {
            info = info.with_destructor_fn(destructor);
        }
        let tid = gc.register_type(info);
        let pytype = row.pytype as *const pyre_object::PyType as usize;
        majit_gc::GcAllocator::register_vtable_for_type(gc, pytype, tid);
        if let Some(destructor) = row.destructor {
            gc.types.set_destructor(tid, destructor);
        }
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
            "MintedExceptionClass { name: \"synth::E\", pytype: &MINTED_EXC_synth__E, parent: None, destructor: None }"
        ));
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E::A\", pytype: &MINTED_EXC_synth__E__A, parent: Some(0), destructor: None }"
        ));
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E::B\", pytype: &MINTED_EXC_synth__E__B, parent: Some(1), destructor: None }"
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
                destructor: None,
            },
            MintedExceptionClass {
                name: "test::E::A",
                pytype: &TEST_A,
                parent: Some(0),
                destructor: None,
            },
            MintedExceptionClass {
                name: "test::E::B",
                pytype: &TEST_B,
                parent: Some(1),
                destructor: None,
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

    #[test]
    fn renders_raw_owned_destructor_and_leaves_a_class_without_one() {
        use majit_translate::front::llbc_hints::{
            MintedExceptionClassEmit, MintedRawOwnedDrop, render_minted_exception_classes_rs,
        };

        let rows = vec![
            MintedExceptionClassEmit {
                name: "Arg".into(),
                parent: None,
                raw_owned: Vec::new(),
            },
            MintedExceptionClassEmit {
                name: "Arg::TooMany".into(),
                parent: Some(0),
                raw_owned: vec![MintedRawOwnedDrop {
                    offset: 8,
                    type_path: "fixture::Signature".into(),
                }],
            },
        ];
        let src = render_minted_exception_classes_rs(&rows);
        assert!(src.contains("unsafe fn MINTED_EXC_Arg__TooMany_destructor"));
        assert!(src.contains("drop_raw_owned_attr::<fixture::Signature>(obj_addr, 8)"));
        assert!(src.contains("destructor: Some(MINTED_EXC_Arg__TooMany_destructor)"));
        assert!(src.contains(
            "MintedExceptionClass { name: \"Arg\", pytype: &MINTED_EXC_Arg, parent: None, destructor: None }"
        ));
        assert!(
            !src.contains("fn MINTED_EXC_Arg_destructor"),
            "a class with no raw-owned attr has no destructor; src={src}"
        );
    }

    static MINTED_RAW_DESTRUCTOR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static MINTED_RAW_DESTRUCTOR_RUNS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    unsafe fn minted_raw_counting_destructor(_obj_addr: usize) {
        MINTED_RAW_DESTRUCTOR_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn minted_class_destructor_runs_once_on_collection() {
        let _guard = MINTED_RAW_DESTRUCTOR_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        MINTED_RAW_DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        static WITH_DEL: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::WithDel");
        static NO_DEL: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::NoDel");
        let rows = [
            MintedExceptionClass {
                name: "test::WithDel",
                pytype: &WITH_DEL,
                parent: None,
                destructor: Some(minted_raw_counting_destructor),
            },
            MintedExceptionClass {
                name: "test::NoDel",
                pytype: &NO_DEL,
                parent: None,
                destructor: None,
            },
        ];
        let mut gc = MiniMarkGC::new();
        let object_size = std::mem::size_of::<pyre_object::PyObject>();
        let object_tid = gc.register_type(majit_gc::trace::TypeInfo::object(object_size));
        let host_tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            object_size,
            object_tid,
        ));
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_interpreter::error::HOST_EXCEPTION_TYPE as *const _ as usize,
            host_tid,
        );
        let tids = register_minted_exception_classes(&mut gc, &rows, host_tid);
        assert_eq!(tids.len(), 2);
        assert!(gc.types.get(tids[0]).destructor.is_some());
        assert!(gc.types.get(tids[1]).destructor.is_none());
        let _with_del = gc.alloc_with_type(tids[0], object_size);
        let _no_del = gc.alloc_with_type(tids[1], object_size);
        gc.do_collect_nursery();
        assert_eq!(
            MINTED_RAW_DESTRUCTOR_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn drop_raw_owned_attr_drops_once_and_clears_the_slot() {
        static HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        HITS.store(0, std::sync::atomic::Ordering::SeqCst);

        struct Probe {
            hits: &'static std::sync::atomic::AtomicUsize,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let boxed = Box::into_raw(Box::new(Probe { hits: &HITS }));
        let mut slot: *mut Probe = boxed;
        let obj_addr = &mut slot as *mut *mut Probe as usize;
        unsafe { drop_raw_owned_attr::<Probe>(obj_addr, 0) };
        assert_eq!(HITS.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(slot.is_null());
        unsafe { drop_raw_owned_attr::<Probe>(obj_addr, 0) };
        assert_eq!(HITS.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
