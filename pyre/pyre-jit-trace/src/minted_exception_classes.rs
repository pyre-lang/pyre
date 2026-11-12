//! One immortal `PyType` per minted RPython exception class.
//!
//! The build script writes `$OUT_DIR/minted_exception_classes.rs` from
//! the harvested hierarchies. Placeholder extraction
//! (`MAJIT_LLBC_EXTRACTION=1`) and a program that minted nothing both
//! emit an empty table, so every build compiles. The row type lives in
//! `pyre_interpreter::error`. This module registers that one table
//! into the interpreter's subclass-range census: a `linkme` element
//! off wasm32, a constructor on wasm32.

use majit_gc::collector::MiniMarkGC;
use pyre_interpreter::error::MintedExceptionClass;

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

/// One census element: the whole generated table, row order preserved.
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice(pyre_interpreter::error::MINTED_EXCEPTION_CLASS_TABLES)]
static MINTED_EXCEPTION_CLASS_TABLE: &'static [MintedExceptionClass] = MINTED_EXCEPTION_CLASSES;

/// wasm32 has no `linkme` slice. The constructor publishes the same table
/// before any exported function, so `init_subclass_ranges` sees it.
#[cfg(target_arch = "wasm32")]
#[::ctor::ctor(unsafe)]
fn register_generated_minted_exception_class_table() {
    pyre_interpreter::error::register_minted_exception_class_table(MINTED_EXCEPTION_CLASSES);
}

/// `support.py` `build_ll_0_alloc_with_del` for one minted row.
///
/// `index` selects the STRUCT. The tid is the one
/// [`register_minted_exception_classes`] assigns that row. `vtable` is
/// the row's `PyType`. Header stores go through
/// `majit_backend::alloc_with_del` / `write_new_with_vtable_header`
/// (`llmodel.py` `bh_new_with_vtable`), with the product's
/// `OB_TYPE_OFFSET` and `W_CLASS_OFFSET`.
fn alloc_with_del_for_index(index: usize, vtable: usize) -> i64 {
    let row = &MINTED_EXCEPTION_CLASSES[index];
    let tid = pyre_interpreter::pyopcode::minted_exception_first_gc_type_id() + index as u32;
    majit_backend::alloc_with_del(
        tid,
        row.size,
        vtable,
        Some(pyre_object::pyobject::OB_TYPE_OFFSET),
        Some(pyre_object::pyobject::W_CLASS_OFFSET),
    )
}

/// Register `rows` in parent-before-child order.
///
/// Each row is `TypeInfo::object_subclass_with_gc_ptrs` of the layout
/// size and the payload GC-pointer offsets that layout recorded.
/// `TypeRegistry::register` merges the parent's offsets. The object
/// root registers `w_class`, and `HOST_EXCEPTION_TYPE` leaves that
/// edge to the merge, so a minted row does too. Raw-owned pointer
/// words stay out of the offset list; the destructor frees them.
/// The tid exists so `subclass_range` can resolve the class word
/// through `vtable_to_type_id`. It is also the id
/// `GcLLDescr_framework.init_size_descr` stores on the SizeDescr:
/// `TypeLayoutBuilder.get_type_id` memoizes one id per TYPE
/// (`id_of_type`), and `encode_type_shape` records this row's
/// destructor on that same id. `new_with_vtable` stamps this tid
/// into the header.
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
    // Generated `build_ll_0_alloc_with_del` wrappers call
    // `alloc_with_del_for_index`. The extraction placeholder emits no
    // wrapper, so the registration path names the helper too.
    let _ = alloc_with_del_for_index as fn(usize, usize) -> i64;
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
        let mut info = majit_gc::trace::TypeInfo::object_subclass_with_gc_ptrs(
            row.size,
            parent_tid,
            row.gc_ptr_offsets.to_vec(),
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

/// Collector tid [`register_minted_exception_classes`] assigns to the row
/// named `owner`, or whose `PyType` address is `vtable`.
///
/// `TypeLayoutBuilder.get_type_id` memoizes one id per TYPE
/// (`id_of_type`). Row `i` is `minted_exception_first_gc_type_id` plus
/// `i`. An empty `owner` does not match an empty name. `vtable == 0`
/// is the analyzer sentinel, not a type pointer
/// (`register_gcstruct_vtable` ignores it).
pub(crate) fn minted_exception_collector_tid(owner: &str, vtable: u64) -> Option<u32> {
    if owner.is_empty() && vtable == 0 {
        return None;
    }
    let first = pyre_interpreter::pyopcode::minted_exception_first_gc_type_id();
    for (index, row) in MINTED_EXCEPTION_CLASSES.iter().enumerate() {
        let name_hit = !owner.is_empty() && owner == row.name;
        let vtable_hit = vtable != 0 && vtable == row.pytype as *const pyre_object::PyType as u64;
        if name_hit || vtable_hit {
            return Some(first + index as u32);
        }
    }
    None
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
            "MintedExceptionClass { name: \"synth::E\", pytype: &MINTED_EXC_synth__E, parent: None, size: 0, gc_ptr_offsets: &[], destructor: None, alloc_with_del: None }"
        ));
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E::A\", pytype: &MINTED_EXC_synth__E__A, parent: Some(0), size: 0, gc_ptr_offsets: &[], destructor: None, alloc_with_del: None }"
        ));
        assert!(src.contains(
            "MintedExceptionClass { name: \"synth::E::B\", pytype: &MINTED_EXC_synth__E__B, parent: Some(1), size: 0, gc_ptr_offsets: &[], destructor: None, alloc_with_del: None }"
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

        let object_size = std::mem::size_of::<pyre_object::PyObject>();
        let rows = [
            MintedExceptionClass {
                name: "test::E",
                pytype: &TEST_BASE,
                parent: None,
                size: object_size,
                gc_ptr_offsets: &[],
                destructor: None,
                alloc_with_del: None,
            },
            MintedExceptionClass {
                name: "test::E::A",
                pytype: &TEST_A,
                parent: Some(0),
                size: object_size,
                gc_ptr_offsets: &[],
                destructor: None,
                alloc_with_del: None,
            },
            MintedExceptionClass {
                name: "test::E::B",
                pytype: &TEST_B,
                parent: Some(1),
                size: object_size,
                gc_ptr_offsets: &[],
                destructor: None,
                alloc_with_del: None,
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
                size: 0,
                gc_ptr_offsets: Vec::new(),
                raw_owned: Vec::new(),
            },
            MintedExceptionClassEmit {
                name: "Arg::TooMany".into(),
                parent: Some(0),
                size: 0,
                gc_ptr_offsets: Vec::new(),
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
        assert!(src.contains("unsafe extern \"C\" fn MINTED_EXC_Arg__TooMany_alloc_with_del"));
        assert!(src.contains(
            "alloc_with_del_for_index(1, &MINTED_EXC_Arg__TooMany as *const _ as usize)"
        ));
        assert!(src.contains("build_ll_0_alloc_with_del::Arg::TooMany"));
        assert!(src.contains("#[linkme(crate = ::majit_ir::linkme)]"));
        assert!(src.contains("alloc_with_del: Some(MINTED_EXC_Arg__TooMany_alloc_with_del)"));
        assert!(src.contains(
            "MintedExceptionClass { name: \"Arg\", pytype: &MINTED_EXC_Arg, parent: None, size: 0, gc_ptr_offsets: &[], destructor: None, alloc_with_del: None }"
        ));
        assert!(
            !src.contains("fn MINTED_EXC_Arg_destructor"),
            "a class with no raw-owned attr has no destructor; src={src}"
        );
        assert!(
            !src.contains("fn MINTED_EXC_Arg_alloc_with_del"),
            "a class with no raw-owned attr has no alloc_with_del; src={src}"
        );
    }

    static MINTED_RAW_DESTRUCTOR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    #[cfg(not(target_arch = "wasm32"))]
    static MINTED_RAW_DESTRUCTOR_RUNS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    #[cfg(not(target_arch = "wasm32"))]
    unsafe fn minted_raw_counting_destructor(_obj_addr: usize) {
        MINTED_RAW_DESTRUCTOR_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    struct ProductionMintedSize {
        cache_key: u64,
        descr: std::sync::Arc<majit_ir::descr::SimpleSizeDescr>,
        tid: u32,
        size: usize,
        pytype: u64,
        name: &'static str,
    }

    fn payload_field(offset: usize) -> majit_jitcode::jitcode::BhFieldSpec {
        majit_jitcode::jitcode::BhFieldSpec {
            index: 0,
            field_key: "payload".to_string(),
            name: "payload".to_string(),
            offset,
            field_size: 8,
            field_type: majit_ir::Type::Int,
            field_flag: majit_ir::descr::ArrayFlag::Signed,
            is_field_signed: true,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            is_class_word: Some(false),
            load_is_acquire: false,
            store_is_release: false,
        }
    }

    fn payload_field_spec(offset: usize) -> majit_ir::descr::SimpleFieldDescrSpec {
        majit_ir::descr::SimpleFieldDescrSpec {
            index: 0,
            field_key: "payload".to_string(),
            name: "payload".to_string(),
            offset,
            field_size: 8,
            field_type: majit_ir::Type::Int,
            is_immutable: false,
            is_quasi_immutable: false,
            flag: majit_ir::descr::ArrayFlag::Signed,
            virtualizable: false,
            index_in_parent: 0,
            is_class_word: Some(false),
            load_is_acquire: false,
            store_is_release: false,
        }
    }

    /// SizeDescr minted the way `new_with_vtable` publishes a parent layout:
    /// `make_descr_from_bh` → `simple_descr_group_from_bh_size`, owner = the
    /// class key, vtable 0. The cache key is not `path_hash` of the class
    /// name, so the process-global `GcCache` slot does not collide with the
    /// translated STRUCT.
    fn production_minted_size() -> ProductionMintedSize {
        use majit_ir::descr::SizeDescr;

        assert!(
            !MINTED_EXCEPTION_CLASSES.is_empty(),
            "minted exception table is empty; extraction did not emit rows"
        );
        let index = MINTED_EXCEPTION_CLASSES
            .iter()
            .position(|row| row.destructor.is_some())
            .unwrap_or(0);
        let row = &MINTED_EXCEPTION_CLASSES[index];
        let tid = pyre_interpreter::pyopcode::minted_exception_first_gc_type_id() + index as u32;
        assert_eq!(minted_exception_collector_tid(row.name, 0), Some(tid));
        let pytype = row.pytype as *const pyre_object::PyType as u64;
        assert_eq!(minted_exception_collector_tid("", pytype), Some(tid));
        assert_eq!(minted_exception_collector_tid("", 0), None);

        let header = std::mem::size_of::<pyre_object::PyObject>();
        let field_off = header;
        let size = row.size.max(field_off + 8);
        let mut salt = 0u32;
        let cache_key = loop {
            let key = majit_ir::descr::path_hash(&format!(
                "__minted_exception_size_descr__::{salt}::{}",
                row.name
            ));
            if key > u32::MAX as u64 && key as u32 != tid {
                break key;
            }
            salt += 1;
            assert!(salt < 16, "could not pick a minted size cache key");
        };
        let bh = majit_jitcode::jitcode::BhDescr::Size {
            size,
            type_id: cache_key,
            vtable: 0,
            owner: row.name.to_string(),
            all_fielddescrs: vec![payload_field(field_off)],
            is_gc_managed: true,
        };
        let returned = crate::descr::make_descr_from_bh(&bh);
        let returned =
            majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(returned)
                .expect("production minted SizeDescr");
        assert_eq!(SizeDescr::type_id(returned.as_ref()), tid);
        assert!(returned.collector_tid_is_fixed());
        let cached = majit_ir::descr::gc_cache()
            .lock()
            ._cache_size
            .get(&majit_ir::descr::LLType::Struct(cache_key))
            .cloned()
            .expect("minted SizeDescr cache entry");
        let descr = majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(cached)
            .expect("cached minted SizeDescr");
        assert_eq!(SizeDescr::type_id(descr.as_ref()), tid);
        assert!(descr.collector_tid_is_fixed());
        assert_eq!(descr.size(), size);
        ProductionMintedSize {
            cache_key,
            descr,
            tid,
            size,
            pytype,
            name: row.name,
        }
    }

    /// Re-run one collecting test in its own process.
    ///
    /// `MiniMarkGC::do_collect_nursery` takes the process-global STW when
    /// `gc_sync::stw_required` reports another registered mutator. A parallel
    /// lib test that then acquires the GIL trips `rejoin_running_after_gil`.
    /// The child is `--exact --test-threads=1`. `PYRE_MINTED_GC_CHILD` names
    /// the test: the matching invocation runs `body`, and the parent asserts
    /// the child passed that one test. wasm32 has no process spawn.
    #[cfg(not(target_arch = "wasm32"))]
    fn run_in_child(test_name: &str, body: fn()) {
        if std::env::var("PYRE_MINTED_GC_CHILD").is_ok_and(|name| name == test_name) {
            body();
            return;
        }
        let exe = std::env::current_exe().expect("current_exe");
        let module = module_path!();
        let module = module
            .split_once("::")
            .map(|(_, rest)| rest)
            .unwrap_or(module);
        let filter = format!("{module}::{test_name}");
        let output = std::process::Command::new(&exe)
            .args([&filter, "--exact", "--test-threads=1"])
            .env("PYRE_MINTED_GC_CHILD", test_name)
            .output()
            .unwrap_or_else(|err| panic!("spawn {exe:?}: {err}"));
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains(&filter) && stdout.contains("1 passed"),
            "{test_name} child failed: status={} filter={filter}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn minted_class_destructor_runs_once_on_collection() {
        run_in_child(
            "minted_class_destructor_runs_once_on_collection",
            minted_class_destructor_runs_once_on_collection_body,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn minted_class_destructor_runs_once_on_collection_body() {
        use majit_ir::descr::SizeDescr;

        let _guard = MINTED_RAW_DESTRUCTOR_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        MINTED_RAW_DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let minted = production_minted_size();
        assert_eq!(SizeDescr::type_id(minted.descr.as_ref()), minted.tid);
        let object_size = std::mem::size_of::<pyre_object::PyObject>();
        let mut gc = MiniMarkGC::new();
        while gc.types.len() < minted.tid as usize {
            let id = gc.register_type(majit_gc::trace::TypeInfo::object(object_size));
            assert!(id < minted.tid);
        }
        let registered = gc.register_type(
            majit_gc::trace::TypeInfo::object(minted.size)
                .with_destructor_fn(minted_raw_counting_destructor),
        );
        assert_eq!(registered, minted.tid);
        assert!(gc.types.get(registered).destructor.is_some());
        let live = gc.alloc_with_type(SizeDescr::type_id(minted.descr.as_ref()), minted.size);
        assert_ne!(live.0, 0);
        let header = unsafe { majit_gc::header::header_of(live.0) };
        assert_eq!(unsafe { (*header).type_id() }, minted.tid);
        let _plain = gc.alloc_with_type(0, object_size);
        gc.do_collect_nursery();
        assert_eq!(
            MINTED_RAW_DESTRUCTOR_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    /// `support.py` `build_ll_0_alloc_with_del` allocates through
    /// `MiniMarkGC::alloc_with_type`, so `finish_nursery_object` runs
    /// `register_young_object_if_needed`. A minor collection then runs the
    /// destructor once. The header stores are `write_new_with_vtable_header`:
    /// `ob_type` is the minted `PyType`, and `w_class` is that type's
    /// `instantiate` pointer.
    ///
    /// The helper calls `gc_op`, which borrows the process-global collector.
    /// Installing that collector, or the `w_class` resolver, in this process
    /// races lib tests that assert the singleton is still unset. The parent
    /// only spawns a one-test child (`run_in_child`).
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn alloc_with_del_registers_young_destructor_once() {
        run_in_child(
            "alloc_with_del_registers_young_destructor_once",
            alloc_with_del_child_body,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn alloc_with_del_child_body() {
        static CLASS_WORD: pyre_object::PyObject = pyre_object::PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        };

        let _guard = MINTED_RAW_DESTRUCTOR_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        MINTED_RAW_DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let index = MINTED_EXCEPTION_CLASSES
            .iter()
            .enumerate()
            .filter(|(_, row)| row.alloc_with_del.is_some() && row.size > 0)
            .min_by_key(|(_, row)| row.size)
            .map(|(index, _)| index)
            .expect("generated minted table has no alloc_with_del row");
        let row = &MINTED_EXCEPTION_CLASSES[index];
        let tid = pyre_interpreter::pyopcode::minted_exception_first_gc_type_id() + index as u32;
        let word = std::mem::size_of::<usize>();
        assert!(row.size >= pyre_object::pyobject::W_CLASS_OFFSET + word);
        assert!(row.size >= pyre_object::pyobject::OB_TYPE_OFFSET + word);
        let vtable = row.pytype as *const pyre_object::PyType as usize;
        let class_word = &CLASS_WORD as *const pyre_object::PyObject as usize;

        majit_gc::rgil::acquire_maybe_in_new_thread();
        if !majit_gc::gc_sync::is_initialized() {
            majit_gc::gc_sync::store_singleton(Box::new(MiniMarkGC::new()));
        }
        pyre_object::pyobject::set_instantiate(row.pytype, class_word as pyre_object::PyObjectRef);
        let _ = majit_ir::descr::set_w_class_obj_resolver(crate::descr::w_class_obj_for_vtable);
        assert_eq!(
            majit_ir::descr::resolve_w_class_obj(vtable),
            Some(class_word as i64),
            "w_class resolver did not return the minted instantiate pointer"
        );

        majit_gc::gc_sync::gc_op(|gc| {
            let object_size = std::mem::size_of::<pyre_object::PyObject>();
            if gc.types.len() > tid as usize {
                let info = gc.types.get(tid);
                assert!(
                    info.old_style_finalizer.is_none(),
                    "minted tid {tid} already has an old-style finalizer; set_destructor cannot attach the counting destructor"
                );
                gc.types.set_destructor(tid, minted_raw_counting_destructor);
                return;
            }
            while gc.types.len() < tid as usize {
                let id = gc.register_type(majit_gc::trace::TypeInfo::object(object_size));
                assert!(id < tid);
            }
            let registered = gc.register_type(
                majit_gc::trace::TypeInfo::object(row.size)
                    .with_destructor_fn(minted_raw_counting_destructor),
            );
            assert_eq!(registered, tid);
            assert!(gc.types.get(registered).destructor.is_some());
        });

        let alloc = row.alloc_with_del.expect("alloc_with_del");
        let addr = unsafe { alloc() };
        assert_ne!(addr, 0, "build_ll_0_alloc_with_del returned NULL");
        let addr = addr as usize;
        let ob_type = unsafe { *((addr + pyre_object::pyobject::OB_TYPE_OFFSET) as *const usize) };
        let w_class = unsafe { *((addr + pyre_object::pyobject::W_CLASS_OFFSET) as *const usize) };
        assert_eq!(ob_type, vtable);
        assert_eq!(w_class, class_word);

        majit_gc::gc_sync::gc_op(|gc| {
            assert!(gc.is_in_nursery(addr), "alloc_with_del left the nursery");
            let header = unsafe { majit_gc::header::header_of(addr) };
            assert_eq!(unsafe { (*header).type_id() }, tid);
            gc.do_collect_nursery();
        });
        assert_eq!(
            MINTED_RAW_DESTRUCTOR_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        majit_gc::gc_sync::gc_op(|gc| gc.do_collect_nursery());
        assert_eq!(
            MINTED_RAW_DESTRUCTOR_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    /// `register_unresolved_struct_tids` / `replay_synthetic_struct_tids`
    /// must not mint a second `TypeInfo` for a minted SizeDescr, and the
    /// vtable-mismatch clone `size_descr_ref_from_bh` records keeps the
    /// same tid (`init_size_descr` / `get_type_id`).
    #[test]
    fn register_unresolved_leaves_minted_size_descr_tid_unchanged() {
        use majit_ir::descr::SizeDescr;

        let _guard = MINTED_RAW_DESTRUCTOR_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let minted = production_minted_size();
        let field_off = std::mem::size_of::<pyre_object::PyObject>();
        let spec = payload_field_spec(field_off);
        let group = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
            u32::MAX,
            minted.size,
            minted.cache_key as u32,
            minted.cache_key,
            minted.pytype as usize,
            true,
            false,
            std::slice::from_ref(&spec),
            &[],
        );
        assert_ne!(group.size_descr.vtable(), 0);
        assert_eq!(SizeDescr::type_id(group.size_descr.as_ref()), minted.tid);
        assert_eq!(SizeDescr::type_id(minted.descr.as_ref()), minted.tid);
        assert!(minted.descr.collector_tid_is_fixed());

        let cleared = majit_jitcode::jitcode::BhDescr::Size {
            size: minted.size,
            type_id: minted.cache_key,
            vtable: minted.pytype,
            owner: String::new(),
            all_fielddescrs: vec![payload_field(field_off)],
            is_gc_managed: true,
        };
        let cleared = crate::descr::make_descr_from_bh(&cleared);
        let cleared =
            majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleSizeDescr>(cleared)
                .expect("owner-cleared minted SizeDescr");
        assert_eq!(SizeDescr::type_id(cleared.as_ref()), minted.tid);

        let mut sentinel = majit_jitcode::jitcode::BhDescr::Size {
            size: minted.size,
            type_id: minted.cache_key,
            vtable: majit_jitcode::codewriter::assembler::type_static_const_sentinel(0) as u64,
            owner: minted.name.to_string(),
            all_fielddescrs: vec![payload_field(field_off)],
            is_gc_managed: true,
        };
        crate::runtime_fnaddr_patch::rebind_type_static_size_vtable(&mut sentinel);
        match sentinel {
            majit_jitcode::jitcode::BhDescr::Size { vtable, owner, .. } => {
                assert!(owner.is_empty());
                assert_eq!(vtable, minted.pytype);
            }
            _ => unreachable!(),
        }
        let shape = majit_ir::descr::gc_cache()
            .lock()
            ._cache_gcstruct2vtable
            .get(&minted.cache_key)
            .copied();
        assert_eq!(shape, Some((minted.pytype as usize, minted.tid)));

        let mut private = majit_ir::descr::GcCache::new();
        private.register_keyed_size(
            majit_ir::descr::LLType::Struct(minted.cache_key),
            minted.descr.clone(),
        );
        let registered = private.register_unresolved_struct_tids(|_, _| 1);
        assert_eq!(registered, 0);
        private.replay_synthetic_struct_tids(|_, _| 1);
        assert_eq!(SizeDescr::type_id(minted.descr.as_ref()), minted.tid);
        assert!(minted.descr.collector_tid_is_fixed());
    }

    /// The parent here is `TypeInfo::object`, which has no offsets.
    /// `TypeRegistry::register` therefore keeps the row's list. Production
    /// `OBJECT` lists `w_class` and `HOST_EXCEPTION_TYPE` leaves that edge
    /// to the merge, so a minted row does not repeat it.
    #[test]
    fn register_minted_class_records_size_and_gc_ptr_offsets() {
        static SIZED: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::Sized");
        let ptr_off = std::mem::size_of::<pyre_object::PyObject>();
        let inst_size = ptr_off + std::mem::size_of::<u64>() + std::mem::size_of::<u64>();
        let row_offsets: &'static [usize] = vec![ptr_off].leak();
        let rows = [MintedExceptionClass {
            name: "test::Sized",
            pytype: &SIZED,
            parent: None,
            size: inst_size,
            gc_ptr_offsets: row_offsets,
            destructor: None,
            alloc_with_del: None,
        }];
        let mut gc = MiniMarkGC::new();
        let object_tid = gc.register_type(majit_gc::trace::TypeInfo::object(std::mem::size_of::<
            pyre_object::PyObject,
        >()));
        let host_tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            object_tid,
        ));
        let tids = register_minted_exception_classes(&mut gc, &rows, host_tid);
        let info = gc.types.get(tids[0]);
        assert_eq!(info.size, inst_size);
        assert_eq!(info.gc_ptr_offsets, vec![ptr_off]);
        assert!(info.has_gc_ptrs);
    }

    /// A nursery instance keeps its scalar bytes across a minor collection,
    /// and a young object stored in a registered GC-pointer field is
    /// updated to the moved address. The collection runs in a one-test
    /// child ([`run_in_child`]).
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn minted_instance_young_pointer_and_scalar_survive_minor_collection() {
        run_in_child(
            "minted_instance_young_pointer_and_scalar_survive_minor_collection",
            minted_instance_young_pointer_and_scalar_survive_minor_collection_body,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn minted_instance_young_pointer_and_scalar_survive_minor_collection_body() {
        let _guard = MINTED_RAW_DESTRUCTOR_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        static HOLDER: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::Holder");
        let ptr_off = std::mem::size_of::<pyre_object::PyObject>();
        let scalar_off = ptr_off + std::mem::size_of::<u64>();
        let inst_size = scalar_off + std::mem::size_of::<u64>();
        let offsets: &'static [usize] = vec![ptr_off].leak();
        let rows = [MintedExceptionClass {
            name: "test::Holder",
            pytype: &HOLDER,
            parent: None,
            size: inst_size,
            gc_ptr_offsets: offsets,
            destructor: None,
            alloc_with_del: None,
        }];
        let mut gc = MiniMarkGC::new();
        let object_tid = gc.register_type(majit_gc::trace::TypeInfo::object(std::mem::size_of::<
            pyre_object::PyObject,
        >()));
        let host_tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            object_tid,
        ));
        let tids = register_minted_exception_classes(&mut gc, &rows, host_tid);
        let (registered_size, registered_offsets) = {
            let info = gc.types.get(tids[0]);
            (info.size, info.gc_ptr_offsets.clone())
        };
        assert_eq!(registered_size, inst_size);
        assert_eq!(registered_offsets, vec![ptr_off]);

        let young_size = std::mem::size_of::<u64>();
        let young_tid = gc.register_type(majit_gc::trace::TypeInfo::object(young_size));
        let young = gc.alloc_with_type(young_tid, young_size);
        unsafe {
            *(young.0 as *mut u64) = 0xC0FFEE;
        }
        let mut inst = gc.alloc_with_type(tids[0], inst_size);
        unsafe {
            *((inst.0 + ptr_off) as *mut usize) = young.0;
            *((inst.0 + scalar_off) as *mut u64) = 0x5CA1_A5;
            gc.roots.add(&mut inst);
        }
        let young_addr = young.0;
        assert!(gc.is_in_nursery(inst.0));
        assert!(gc.is_in_nursery(young_addr));
        gc.do_collect_nursery();
        let moved = unsafe { *((inst.0 + ptr_off) as *const usize) };
        assert_ne!(moved, young_addr);
        assert!(!gc.is_in_nursery(moved));
        assert_eq!(unsafe { *(moved as *const u64) }, 0xC0FFEE);
        assert_eq!(unsafe { *((inst.0 + scalar_off) as *const u64) }, 0x5CA1_A5);
        gc.roots.clear();
    }

    /// Nursery allocation does not zero (`malloc_zero_filled` is false;
    /// `Nursery::reset` leaves recycled bytes, and `gc_nursery_debug`
    /// fills them with the poison `assert_traced_slot_initialized`
    /// rejects). `rewrite.py handle_new` then `clear_gc_fields` stores
    /// NULL at every `gc_fielddescrs` offset, which for a minted class
    /// is the header class word plus the payload GC fields.
    /// `gen_initialize_w_class` does not run: `w_class_obj()` is 0.
    ///
    /// `llmodel.py bh_new_with_vtable` does not need that list. It calls
    /// `gc_malloc`, framework `_bh_malloc` is `do_malloc_fixedsize_clear`,
    /// and `alloc_oldgen_typed` `raw_memclear`s the payload before
    /// `write_new_with_vtable_header` writes a class word only when
    /// `resolve_w_class_obj` is nonzero.
    ///
    /// This applies those `gc_fielddescrs` stores onto a production
    /// nursery bump. The parent is the object root that traces
    /// `w_class`, so a minor collection visits the word. A scalar that
    /// the descr does not list stays poison, which a whole-object
    /// memclear would have wiped. The collections run in a one-test child
    /// ([`run_in_child`]).
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn minted_instance_clears_header_class_word_on_a_dirty_nursery() {
        run_in_child(
            "minted_instance_clears_header_class_word_on_a_dirty_nursery",
            minted_instance_clears_header_class_word_on_a_dirty_nursery_body,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn minted_instance_clears_header_class_word_on_a_dirty_nursery_body() {
        use majit_ir::descr::{FieldDescr, SizeDescr};

        let _guard = MINTED_RAW_DESTRUCTOR_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        let header = std::mem::size_of::<pyre_object::PyObject>();
        let word = std::mem::size_of::<usize>();
        let class_off = pyre_object::pyobject::W_CLASS_OFFSET;
        assert!(class_off + word <= header);
        let str_off = header;
        let scalar_off = str_off + word;
        let list_off = scalar_off + word;
        let raw_off = list_off + word;
        let inst_size = raw_off + word;
        // `collector.rs assert_traced_slot_initialized`.
        let poison = (usize::MAX / 0xff) * 0xaa;

        let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
            debug: 0,
            gc_nursery_debug: true,
            ..majit_gc::collector::GcConfig::default()
        });
        let object_tid = gc.register_type(
            majit_gc::trace::TypeInfo::object_with_gc_ptrs(header, vec![class_off])
                .with_app_level_inspector_hidden_edge(class_off),
        );
        let host_tid = gc.register_type(majit_gc::trace::TypeInfo::object_subclass(
            header, object_tid,
        ));
        static DIRTY: pyre_object::PyType = pyre_object::pyobject::new_pytype("test::DirtyNursery");
        let payload: &'static [usize] = vec![str_off, list_off].leak();
        let rows = [MintedExceptionClass {
            name: "test::DirtyNursery",
            pytype: &DIRTY,
            parent: None,
            size: inst_size,
            gc_ptr_offsets: payload,
            destructor: None,
            alloc_with_del: None,
        }];
        let tids = register_minted_exception_classes(&mut gc, &rows, host_tid);
        let registered = gc.types.get(tids[0]).gc_ptr_offsets.clone();
        assert!(registered.contains(&class_off), "{registered:?}");
        assert!(registered.contains(&str_off), "{registered:?}");
        assert!(registered.contains(&list_off), "{registered:?}");
        assert!(!registered.contains(&scalar_off), "{registered:?}");
        assert!(!registered.contains(&raw_off), "{registered:?}");

        let fields = [
            field_spec(0, "w_class", "PyObject.w_class", class_off, true),
            field_spec(1, "msg", "Minted.msg", str_off, true),
            field_spec(2, "count", "Minted.count", scalar_off, false),
            field_spec(3, "missing", "Minted.missing", list_off, true),
            field_spec(4, "signature", "Minted.signature", raw_off, false),
        ];
        let group = majit_ir::descr::make_simple_descr_group(0, inst_size, 0, 0, &fields);
        let size = group.size_descr.as_ref();
        assert_eq!(size.w_class_obj(), None);
        assert_eq!(
            size.class_word_field().map(|field| field.offset()),
            Some(class_off)
        );
        let gc_offsets: Vec<usize> = size
            .gc_fielddescrs()
            .iter()
            .map(|field| field.offset())
            .collect();
        assert_eq!(gc_offsets, vec![class_off, str_off, list_off]);

        // The fresh arena is zero. A collection with `gc_nursery_debug`
        // and no rotating ring poisons that same arena in place.
        gc.do_collect_nursery();
        let mut inst = gc.alloc_with_type(tids[0], inst_size);
        assert!(gc.is_in_nursery(inst.0));
        let word_at = |offset: usize| unsafe { *((inst.0 + offset) as *const usize) };
        for offset in [class_off, str_off, scalar_off, list_off, raw_off] {
            assert_eq!(
                word_at(offset),
                poison,
                "nursery bump left offset {offset} uncleaned"
            );
        }
        // The stores `clear_gc_fields` emits after `gen_malloc_nursery`.
        for offset in gc_offsets {
            unsafe {
                *((inst.0 + offset) as *mut usize) = 0;
            }
        }
        assert_eq!(word_at(class_off), 0);
        assert_eq!(word_at(str_off), 0);
        assert_eq!(word_at(list_off), 0);
        assert_eq!(word_at(scalar_off), poison);
        assert_eq!(word_at(raw_off), poison);

        unsafe {
            gc.roots.add(&mut inst);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(inst.0));
        assert_eq!(unsafe { *((inst.0 + class_off) as *const usize) }, 0);
        assert_eq!(unsafe { *((inst.0 + str_off) as *const usize) }, 0);
        assert_eq!(unsafe { *((inst.0 + list_off) as *const usize) }, 0);
        assert_eq!(unsafe { *((inst.0 + scalar_off) as *const usize) }, poison);
        gc.roots.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn field_spec(
        index: u32,
        key: &str,
        name: &str,
        offset: usize,
        pointer: bool,
    ) -> majit_ir::descr::SimpleFieldDescrSpec {
        let word = std::mem::size_of::<usize>();
        majit_ir::descr::SimpleFieldDescrSpec {
            index,
            field_key: key.to_string(),
            name: name.to_string(),
            offset,
            field_size: word,
            field_type: if pointer {
                majit_ir::value::Type::Ref
            } else {
                majit_ir::value::Type::Int
            },
            is_immutable: false,
            is_quasi_immutable: false,
            flag: if pointer {
                majit_ir::descr::ArrayFlag::Pointer
            } else {
                majit_ir::descr::ArrayFlag::Unsigned
            },
            virtualizable: false,
            is_class_word: None,
            index_in_parent: index as usize,
            load_is_acquire: false,
            store_is_release: false,
        }
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
