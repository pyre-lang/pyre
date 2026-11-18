//! DirectPointer vable arrays resume through the PyPy CPU chain.
//!
//! `bhimpl_getarrayitem_vable_*` / `bhimpl_arraylen_vable` /
//! `bhimpl_setarrayitem_vable_*` do
//! `clear_vable_token` → `cpu.bh_getfield_gc_r` → `cpu.bh_getarrayitem_gc_*`
//! (or `bh_arraylen_gc` / `bh_setarrayitem_gc_*`). `VirtArray` is
//! `Ptr(GcArray)` after `make_sure_not_resized` (`virtualizable.py`), so
//! that chain must be the one that answers.

use majit_metainterp::blackhole::{BhJitDriverSd, BlackholeInterpBuilder, wire_bhimpl_handlers};
use majit_metainterp::jitcode::{self, JitCodeBuilder};
use majit_metainterp::virt_array::{VirtArray, register_virt_array_field};
use majit_metainterp::virtualizable::{VableArrayStorage, VirtualizableInfo};

#[cfg(all(feature = "cranelift", not(feature = "dynasm")))]
use majit_backend_cranelift::CraneliftBackend as TestBackend;
#[cfg(feature = "dynasm")]
use majit_backend_dynasm::runner::DynasmBackend as TestBackend;

struct State {
    regs: VirtArray<i64>,
}

fn build_vinfo() -> VirtualizableInfo {
    let mut info = VirtualizableInfo::without_vable_token();
    register_virt_array_field(
        &mut info,
        "regs",
        majit_ir::Type::Int,
        std::mem::size_of::<i64>(),
        std::mem::offset_of!(State, regs),
        |s: &State| &s.regs,
    );
    assert_eq!(
        info.array_fields[0].storage,
        VableArrayStorage::DirectPointer
    );
    info
}

fn bh_builder() -> BlackholeInterpBuilder {
    let mut entries: indexmap::IndexMap<String, u8> = jitcode::wellknown_bh_insns()
        .iter()
        .map(|(key, value)| ((*key).to_string(), *value))
        .collect();
    entries.extend(
        jitcode::extension_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value)),
    );
    let mut builder = BlackholeInterpBuilder::new();
    builder.setup_insns(&entries);
    wire_bhimpl_handlers(&mut builder);
    builder
}

#[test]
fn getarrayitem_vable_i_reads_a_block_through_the_cpu_chain() {
    let cpu = TestBackend::new();
    let mut builder = bh_builder();
    builder.set_cpu(&cpu);

    let mut b = JitCodeBuilder::new();
    b.vable_getarrayitem_int_with_base(2, 0, 0, 1);
    b.int_return(2);
    let jitcode = std::sync::Arc::new(b.finish());

    let info = build_vinfo();
    let mut state = State {
        regs: VirtArray::from_slice(&[10i64, 20, 30]),
    };

    let mut bh = builder.acquire_interp();
    bh.virtualizable_info = &info;
    bh.setposition(jitcode, 0);
    bh.registers_r[0] = &mut state as *mut State as i64;
    bh.registers_i[1] = 2;
    let _ = bh.run();

    assert_eq!(bh.tmpreg_i, 30);
}

#[test]
fn setarrayitem_vable_i_then_arraylen_vable_use_the_cpu_chain() {
    let cpu = TestBackend::new();
    let mut builder = bh_builder();
    builder.set_cpu(&cpu);

    let mut b = JitCodeBuilder::new();
    b.vable_setarrayitem_int_with_base(0, 0, 1, 2);
    b.vable_arraylen_with_base(3, 0, 0);
    b.int_return(3);
    let jitcode = std::sync::Arc::new(b.finish());

    let info = build_vinfo();
    let mut state = State {
        regs: VirtArray::from_slice(&[10i64, 20, 30]),
    };

    let mut bh = builder.acquire_interp();
    bh.virtualizable_info = &info;
    bh.setposition(jitcode, 0);
    bh.registers_r[0] = &mut state as *mut State as i64;
    bh.registers_i[1] = 1;
    bh.registers_i[2] = 99;
    let _ = bh.run();

    assert_eq!(state.regs.to_vec(), vec![10, 99, 30]);
    assert_eq!(bh.tmpreg_i, 3);
}

#[test]
fn setarrayitem_vable_uses_type_vinfo_when_driver_vinfo_is_null() {
    // blackhole.py `bhimpl_setarrayitem_vable_*` `fdescr.get_vinfo()`:
    // TYPE vinfo, not the current driver's. A novable driver (filter)
    // leaves `bh.virtualizable_info` unset; the jitcode still runs vable
    // ops of another type.
    let cpu = TestBackend::new();
    let mut builder = bh_builder();
    builder.set_cpu(&cpu);

    let info = build_vinfo();
    builder.setup_jitdrivers_sd(vec![BhJitDriverSd {
        virtualizable_info: &info,
        ..Default::default()
    }]);

    let mut b = JitCodeBuilder::new();
    b.vable_setarrayitem_int_with_base(0, 0, 1, 2);
    b.vable_arraylen_with_base(3, 0, 0);
    b.int_return(3);
    let jitcode = std::sync::Arc::new(b.finish());

    let mut state = State {
        regs: VirtArray::from_slice(&[10i64, 20, 30]),
    };

    let mut bh = builder.acquire_interp();
    assert!(
        bh.virtualizable_info.is_null(),
        "novable resume leaves the driver vinfo unset"
    );
    bh.setposition(jitcode, 0);
    bh.registers_r[0] = &mut state as *mut State as i64;
    bh.registers_i[1] = 1;
    bh.registers_i[2] = 99;
    let _ = bh.run();

    assert_eq!(state.regs.to_vec(), vec![10, 99, 30]);
    assert_eq!(bh.tmpreg_i, 3);
}
