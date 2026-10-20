//! `executor.py` `do_getarrayitem_raw_i` on a misaligned item address.
//!
//! `opimpl.rs` has no harness that can drive a single `opimpl_*`. This
//! pins the loader `bh_getarrayitem_raw_i` uses (`read_unaligned`) so a
//! misaligned i32 item is the value the tracer records.

use majit_ir::Type;
use majit_jitcode::jitcode::BhDescr;
use majit_metainterp::executor::do_getarrayitem_raw_i;

#[cfg(all(feature = "cranelift", not(feature = "dynasm")))]
use majit_backend_cranelift::CraneliftBackend as TestBackend;
#[cfg(feature = "dynasm")]
use majit_backend_dynasm::runner::DynasmBackend as TestBackend;

fn raw_i32_array_descr() -> BhDescr {
    BhDescr::Array {
        base_size: 0,
        itemsize: 4,
        len_offset: None,
        type_id: 0,
        gc_type_id: 0,
        item_type: Type::Int,
        is_array_of_pointers: false,
        is_array_of_structs: false,
        is_item_signed: true,
        ei_index: u32::MAX,
        array_type_id: None,
        interior_fields: Vec::new(),
        is_gc_managed: false,
        load_is_acquire: false,
    }
}

#[test]
fn do_getarrayitem_raw_i_reads_misaligned_i32() {
    let cpu = TestBackend::new();
    let mut buf = vec![0u8; 16];
    let base = buf.as_mut_ptr() as usize;
    let offset = (1 + 4 - (base % 4)) % 4;
    let item = (base + offset) as *mut u8;
    assert_eq!(
        item as usize % core::mem::align_of::<i32>(),
        1,
        "item address must be 1-mod-4 for an i32 descr"
    );
    let stored: i32 = 0x1234_5678;
    unsafe {
        (item as *mut i32).write_unaligned(stored);
    }
    let expected = unsafe { (item as *const i32).read_unaligned() as i64 };
    let got = do_getarrayitem_raw_i(&cpu, (), item as i64, 0, &raw_i32_array_descr());
    assert_eq!(got, expected);
}
