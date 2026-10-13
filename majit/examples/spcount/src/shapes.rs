//! Scalar-only residual loop.
//!
//! `ScalarState` has only one scalar red. `TICK` calls a
//! `#[dont_look_inside]` residual and `LOOP` jumps back to the header while
//! the counter is live.

type Code = [u8];

const TICK: u8 = 1;
const LOOP: u8 = 2;

#[cfg(test)]
static SCALAR_TICKS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

#[majit_macros::dont_look_inside]
extern "C" fn scalar_tick(left: i64) {
    #[cfg(test)]
    SCALAR_TICKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let _ = left;
}

struct ScalarState {
    left: i64,
}

#[majit_macros::jit_interp(
    state = ScalarState,
    env = Code,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        left: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn scalar_mainloop(program: &Code, n: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<ScalarState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    majit_metainterp::embed::Census::install(&mut driver);
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = ScalarState { left: n };

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
            TICK => {
                scalar_tick(state.left);
                state.left -= 1;
            }
            LOOP => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                if state.left > 0 {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.left
}

/// `TICK; LOOP` back to pc 0.
///
///   0: TICK
///   1: LOOP
///   2: offset byte. target = 2 + (-3) + 1 = 0
fn tick_loop_program() -> Vec<u8> {
    vec![TICK, LOOP, 253]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;
    use majit_metainterp::embed::Census;

    fn scalar_probe(n: i64) -> (i64, u32, usize) {
        let census = Census::begin();
        SCALAR_TICKS.store(0, Ordering::Relaxed);
        let got = scalar_mainloop(&tick_loop_program(), n, 3);
        (
            got,
            SCALAR_TICKS.load(Ordering::Relaxed),
            census.counts().loops_compiled,
        )
    }

    #[test]
    fn scalar_only_residual_closes_loop() {
        for n in [50_i64, 500] {
            let (got, ticks, compiles) = scalar_probe(n);
            assert!(
                compiles >= 1,
                "scalar-only loop never compiled for n={n} (ticks={ticks}, left={got})"
            );
            assert_eq!(
                ticks, n as u32,
                "scalar residual fired {ticks}× for n={n}, compiles={compiles}"
            );
            assert_eq!(got, 0, "scalar counter did not reach 0 for n={n}");
        }
    }

    /// TICK stays a sub-JitCode because `scalar_tick` is Infer
    /// (`auto_calls`, `#[dont_look_inside]`). That callee used to emit
    /// `load_state_field` / `store_state_field` against the parent's
    /// identity slots. The state-as-vable gate now covers that arm shape,
    /// so `rewrite_op_getfield` / `rewrite_op_setfield` land on the live
    /// virtualizable and no callee JitCode reads identity slots.
    #[test]
    fn scalar_state_callee_jitcodes_do_not_read_identity_slots() {
        use majit_metainterp::Assembler;
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
        super::__prebuild_jitcode_liveness_scalar_mainloop(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch = super::__dispatch_jitcode_scalar_mainloop(&mut asm, 0i64)
            .expect("dispatch lower must succeed for ScalarState");
        let identity_callees: Vec<&str> = dispatch
            .exec
            .descrs
            .iter()
            .filter_map(majit_metainterp::RuntimeBhDescr::as_jitcode)
            .filter(|jc| jc.reads_identity_slots())
            .map(|jc| jc.name())
            .collect();
        assert!(
            identity_callees.is_empty(),
            "ScalarState TICK must use getfield_vable / setfield_vable, \
             not load_state_field / store_state_field; \
             callees with identity slots: {identity_callees:?}"
        );
    }
}

/// Ref and float stores, then a guard in the same arm.
///
/// `ADVANCE` writes `cur` and `acc` and then guards `left > 0`. The guard's
/// snapshot is taken from the arm frame's identity registers. After the guard
/// fails, resume reads those fields; the pair must match a run with no JIT.

#[repr(C)]
struct Link {
    tag: i64,
    next: *mut Link,
}

struct WalkState {
    left: i64,
    seen: i64,
    cur: usize,
    acc: f64,
}

const ADVANCE: u8 = 1;

fn link_chain(n: usize) -> Vec<Link> {
    let mut links = Vec::with_capacity(n + 1);
    for i in 0..=n {
        links.push(Link {
            tag: i as i64,
            next: std::ptr::null_mut(),
        });
    }
    for i in 0..n {
        let next = std::ptr::from_mut(&mut links[i + 1]);
        links[i].next = next;
    }
    links
}

#[majit_macros::jit_interp(
    state = WalkState,
    env = Code,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        left: int,
        seen: int,
        cur: ref(Link),
        acc: float,
    },
    ref_fields = { Link::next => Link },
    int_fields = { Link::tag => i64 },
)]
#[allow(unused_assignments, unused_variables)]
pub fn walk_mainloop(program: &Code, n: i64, threshold: u32) -> (i64, f64) {
    let links = link_chain(n as usize);
    let mut driver: majit_metainterp::JitDriver<WalkState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    majit_metainterp::embed::Census::install(&mut driver);
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = WalkState {
        left: n,
        seen: 0,
        cur: std::ptr::from_ref(&links[0]) as usize,
        acc: 0.0,
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
        match opcode {
            ADVANCE => {
                state.cur = state.cur.next;
                state.seen = state.cur.tag;
                state.acc = state.acc + 1.0;
                state.left -= 1;
                if state.left > 0 {
                    can_enter_jit!(driver, 0, &mut state, program, || {});
                    continue;
                }
                pc += 1;
            }
            _ => break,
        }
    }
    let tag = unsafe { (*(state.cur as *const Link)).tag };
    (tag, state.acc)
}

fn advance_program() -> Vec<u8> {
    vec![ADVANCE]
}

/// Same steps as `walk_mainloop`, with no JIT hooks.
fn walk_interp(n: i64) -> (i64, f64) {
    let links = link_chain(n as usize);
    let mut cur = std::ptr::from_ref(&links[0]) as usize;
    let mut seen = 0;
    let mut acc = 0.0;
    let mut left = n;
    while left > 0 {
        cur = unsafe { (*(cur as *const Link)).next as usize };
        seen = unsafe { (*(cur as *const Link)).tag };
        acc += 1.0;
        left -= 1;
    }
    (seen, acc)
}

#[cfg(test)]
mod walk_tests {
    use super::*;
    use majit_metainterp::embed::Census;

    #[test]
    fn ref_and_float_store_survives_guard_resume() {
        for n in [20_i64, 50] {
            let census = Census::begin();
            let jit = walk_mainloop(&advance_program(), n, 3);
            let plain = walk_interp(n);
            let compiles = census.counts().loops_compiled;
            assert!(
                compiles >= 1,
                "walk loop never compiled for n={n} (jit={jit:?})"
            );
            assert_eq!(
                jit, plain,
                "guard resume diverged from the interpreter for n={n} (compiles={compiles})"
            );
        }
    }
}
