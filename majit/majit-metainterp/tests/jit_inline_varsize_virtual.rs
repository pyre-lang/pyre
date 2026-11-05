//! A `jit_inline` header+items block stays virtual when it does not escape.
//!
//! `rewrite_op_malloc_varsize` emits `new_array_clear`. `optimize_NEW_ARRAY`
//! (`virtualize.py`) drops that op, and the following `setarrayitem` /
//! `arraylen_gc`, when the block is only read inside the loop.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const OP_FILL: u8 = 1;
const OP_TICK: u8 = 2;

struct Cell;

type Slot = *mut Cell;

#[repr(C)]
struct Items {
    capacity: i64,
    items: [Slot; 0],
}

impl majit_metainterp::HasGcTypeId for Items {
    const GC_TYPE_ID: u32 = 1;
}

#[repr(C)]
struct Holder {
    block: *mut Items,
}

fn alloc_items(capacity: i64, _items: [Slot; 0]) -> *mut Items {
    let count = capacity.max(0) as usize;
    let bytes = std::mem::size_of::<i64>() + count * std::mem::size_of::<*mut Cell>();
    let layout = std::alloc::Layout::from_size_align(bytes.max(8), 8).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Items;
    unsafe {
        std::ptr::write(ptr as *mut i64, capacity.max(0));
    }
    ptr
}

fn alloc_holder(block: *mut Items) -> *mut Holder {
    let layout =
        std::alloc::Layout::from_size_align(std::mem::size_of::<Holder>().max(8), 8).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Holder;
    unsafe {
        (*ptr).block = block;
    }
    ptr
}

/// Allocate a one-slot block, store a null item, return the length.
///
/// The length is `arraylen_gc` on the virtual array, so the compiled loop
/// is the constant `1` with no allocation and no call.
#[majit_macros::jit_inline(
    ref_fields = { Holder::block => Items },
    array_fields = { Holder::block => Slot in Items },
    int_fields = { Items::capacity => i64 },
    struct_allocs = {
        Items => alloc_items,
        Holder => alloc_holder,
    },
)]
fn fill_slot() -> i64 {
    let block = Items {
        capacity: 1i64,
        items: [] as [Slot; 0],
    };
    let holder = Holder { block };
    holder.block[0] = core::ptr::null_mut();
    block.capacity
}

struct FillState {
    acc: i64,
    ticks: i64,
}

struct Recorder {
    compiles: AtomicUsize,
    body: Mutex<Vec<OpCode>>,
}

impl Recorder {
    const fn new() -> Self {
        Self {
            compiles: AtomicUsize::new(0),
            body: Mutex::new(Vec::new()),
        }
    }
}

static FILL: Recorder = Recorder::new();

#[majit_macros::jit_interp(
    state = FillState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
    calls = { fill_slot => inline_int },
)]
#[allow(unused_variables)]
fn dispatch_fill(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<FillState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        FILL.compiles.fetch_add(1, Ordering::Relaxed);
        *FILL.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = FillState { acc: 0, ticks };
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
            OP_FILL => {
                state.acc = fill_slot();
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.acc
}

#[test]
fn varsize_block_stays_virtual_when_it_does_not_escape() {
    let program = [OP_FILL, OP_TICK];
    let acc = dispatch_fill(&program, 20, 80);
    assert_eq!(acc, 1, "arraylen of the one-slot block");
    assert!(
        FILL.compiles.load(Ordering::Relaxed) > 0,
        "the loop did not compile, so virtualization was not exercised"
    );
    let body = FILL.body.lock().clone();
    let names: Vec<String> = body.iter().map(|op| format!("{op:?}")).collect();
    assert!(
        names
            .iter()
            .all(|name| !name.contains("NewArray") && !name.contains("Call")),
        "virtual varsize block leaked into the loop: {names:?}"
    );
}
