//! A `[.. ; virt]` state field declared with a block-backed container.
//!
//! `virtualizable.py:57-58` describes each virtualizable array as a
//! `Ptr(GcArray)` — one pointer word in the virtualizable, aimed at a block
//! whose length and payload sit at fixed offsets. That is what lets
//! `compile.py:441-457` rebuild the array inside a compiled entry from the
//! virtualizable alone: `GETFIELD_GC_R` for the pointer, then one
//! `GETARRAYITEM_GC_*` per element.
//!
//! A `Vec` embedded by value cannot present that, because its data pointer is
//! one of three words in an unspecified order. This fixture declares the same
//! machine with `VirtArray` instead and checks two things: the interpretation is
//! unchanged, and the field registers as the array-pointer storage whose two
//! offsets the entry can be emitted from.

use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::virtualizable::VableArrayStorage;

pub type Bytecode = [u8];

const OP_ACC: u8 = 1; // fregs[0] += 1.25
const OP_INC: u8 = 2; // regs[0] += 1
const OP_JUMP_BACK: u8 = 3; // [OP_JUMP_BACK, target]: loop while regs[0] < regs[1]
const OP_RETURN: u8 = 4; // yield regs[0]

fn count_program() -> Vec<u8> {
    vec![OP_ACC, OP_INC, OP_JUMP_BACK, 0, OP_RETURN]
}

struct Machine {
    regs: VirtArray<i64>,
    fregs: VirtArray<f64>,
}

#[majit_macros::jit_interp(
    state = Machine,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
        fregs: [float; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Machine> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = Machine {
        regs: VirtArray::from_slice(&[0i64, limit]),
        fregs: VirtArray::filled(0.0f64, 1),
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            OP_ACC => state.fregs[0] += 1.25,
            OP_INC => state.regs[0] += 1,
            OP_JUMP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.regs[0] < state.regs[1] {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                }
            }
            OP_RETURN => return state.regs[0],
            _ => unreachable!(),
        }
    }
    state.regs[0]
}

/// The machine computes what it computed with a `Vec` backing, warm and cold.
#[test]
fn a_block_backed_state_interprets_the_same_program_to_the_same_answer() {
    let program = count_program();
    assert_eq!(mainloop(&program, 7, u32::MAX), 7, "never-warm run");
    assert_eq!(mainloop(&program, 200, 3), 200, "warm run");
}

/// The registration the generated `VirtualizableInfo` builder produces is the
/// array-pointer storage, with the block's length and payload offsets. Those
/// two offsets are exactly what `patch_new_loop_to_load_virtualizable_fields`
/// needs to emit the reload preamble.
#[test]
fn a_block_backed_field_registers_the_storage_a_compiled_entry_can_reload() {
    let info = <Machine as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("a state with virt arrays builds a vinfo");

    // Each field is graded against the offsets of the container ITS OWN item
    // type produces, so a divergence between the two item types is a failure
    // rather than a coincidence, and the name is checked so a reordering of
    // `array_fields` fails here instead of silently swapping the expectations.
    let expected = [
        (
            "regs",
            VirtArray::<i64>::LENGTH_OFFSET,
            VirtArray::<i64>::ITEMS_OFFSET,
            size_of::<i64>(),
        ),
        (
            "fregs",
            VirtArray::<f64>::LENGTH_OFFSET,
            VirtArray::<f64>::ITEMS_OFFSET,
            size_of::<f64>(),
        ),
    ];
    assert_eq!(info.array_fields.len(), expected.len());
    for (index, (field, (name, length_offset, items_offset, item_size))) in
        info.array_fields.iter().zip(expected).enumerate()
    {
        assert_eq!(field.name, name, "array field {index} is out of order");
        assert_eq!(
            field.storage,
            VableArrayStorage::DirectPointer,
            "array field {index} ({name}) must be reachable through its pointer",
        );
        assert_eq!(field.length_offset, length_offset, "{name} length offset");
        assert_eq!(field.items_offset, items_offset, "{name} items offset");
        assert!(field.can_read_length_from_heap());

        // The element descr addresses the payload straight off the pointer the
        // field holds, so nothing sits between the two loads the entry emits.
        let descr = info.array_item_descr(index);
        let descr = descr.as_array_descr().expect("an array descr");
        assert_eq!(descr.base_size(), items_offset, "{name} descr base size");
        assert_eq!(descr.item_size(), item_size, "{name} descr item size");
    }
}

/// The container the field is declared with is what the array reads and writes
/// through, so a live state's arrays are addressable from the state pointer by
/// those same offsets.
#[test]
fn the_live_arrays_are_reachable_from_the_state_pointer_by_the_registered_offsets() {
    let info = <Machine as majit_metainterp::JitState>::__build_virtualizable_info()
        .expect("a state with virt arrays builds a vinfo");
    let mut state = Machine {
        regs: VirtArray::from_slice(&[11i64, 22, 33]),
        fregs: VirtArray::filled(0.5f64, 2),
    };
    let vable = &mut state as *mut Machine as *mut u8;

    let regs = &info.array_fields[0];
    let block = unsafe { *(vable.add(regs.field_offset) as *const *const u8) };
    let length = unsafe { *(block.add(regs.length_offset) as *const usize) };
    assert_eq!(length, 3);
    for (index, expected) in [11i64, 22, 33].into_iter().enumerate() {
        let item = unsafe { *(block.add(regs.items_offset + index * 8) as *const i64) };
        assert_eq!(item, expected);
    }

    let fregs = &info.array_fields[1];
    let block = unsafe { *(vable.add(fregs.field_offset) as *const *const u8) };
    assert_eq!(
        unsafe { *(block.add(fregs.length_offset) as *const usize) },
        2
    );
    assert_eq!(
        unsafe { *(block.add(fregs.items_offset) as *const f64) },
        0.5
    );
}

/// Because the storage is reloadable, installing the machine ARMS the flat
/// entry contract: `warmspot.py:520-545` names the virtualizable on the
/// jitdriver static data, which is what makes `compile.py:508-511`'s reload
/// preamble run.
///
/// The width is the flat live-value prefix, not the red count — this state has
/// no scalars, so the whole prefix is the one virtualizable identity slot at
/// index 0, and `virtualizable_arg_index` answers in that model rather than
/// with a position among the reds.
#[test]
fn installing_a_block_backed_machine_arms_the_flat_entry_contract() {
    let program = count_program();
    let mut driver: majit_metainterp::JitDriver<Machine> =
        majit_metainterp::JitDriver::new(u32::MAX);
    let state = Machine {
        regs: VirtArray::from_slice(&[0i64, 1]),
        fregs: VirtArray::filled(0.0f64, 1),
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &program)
            .install_canonical_liveness(&mut driver);
    }

    let contract = driver
        .flat_entry_contract()
        .expect("a reloadable virtualizable arms the contract");
    assert_eq!(contract.len, 1);
    assert_eq!(contract.index_of_virtualizable, 0);
}
