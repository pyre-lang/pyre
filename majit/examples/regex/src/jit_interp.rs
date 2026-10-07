//! The JIT portal, `&`/`|` variant: `shift` traced with the tree held constant.
//!
//! "A JIT for Regular Expression Matching" makes the regex the JitDriver's
//! green and the input position a red. The tracer then sees the tree as
//! constant, inlines the whole `shift` recursion for one character, and folds
//! every structural read away — what is left is the marks and the character
//! comparisons, which is subset construction performed by the tracer.
//!
//! **Which of the two portals is the post's own spelling.** The post's source
//! writes `shift` with Python `and`/`or`, and part 2 then remarks that `&`/`|`
//! is the better spelling because it keeps short-circuit branches out of the
//! loop body. This module took the remark; `shortcircuit.rs` next door takes
//! the source. Running the post's matcher through RPython's own JIT
//! (`rpython_original/`, under `LLJitMixin`) settles which is which: RPython's
//! trace carries 27 guards, `shortcircuit` carries 26, and this module carries
//! 1. So `shortcircuit` is the faithful port and this module is the adapted
//! one — an A/B of the remark, not a second copy of the post. Every structural
//! count is the same on all three (0 `getfield_gc_r`, 24 `getfield_gc_i`, 93
//! `setfield_gc`, 2 `int_eq`); the census table is in `shortcircuit.rs`.
//!
//! Two things this file has to say in majit's vocabulary:
//!
//! * The portal is `marked.py`'s `while i < len(s)`: first character outside
//!   the loop, `can_enter_jit` at the header (what `warmspot.rewrite_can_enter_jit`
//!   inserts when the source has no tick of its own), then `jit_merge_point`
//!   and `shift`. `pc` and `program` stay as the portal ABI greens and are
//!   constant (`pc` is 0); the regex is the promote that the trace specializes
//!   on.
//! * The regex root is a red `ref(NodeRec)` promoted once per iteration. The
//!   promote is what makes the root a `ConstPtr`; the `_immutable_fields_`
//!   declaration on `NodeRec` is what makes every read below it fold, so the
//!   walk collapses into the marks alone.
//!
//! The result, measured on `(a|b)*a(a|b){20}a(a|b)*` — 93 nodes — and
//! identical on dynasm and cranelift. The peeled loop body, which is what runs
//! per input character:
//!
//! ```text
//! 194 ops: 0 getfield_gc_r, 24 getfield_gc_i, 93 setfield_gc,
//!          2 int_eq, 0 guard_value, 1 strgetitem
//! ```
//!
//! Not one pointer read: the 92 edges of the tree walk are gone. 93 stores is
//! one mark per node, so every node is in the trace. The promote and the
//! string pointer are loop invariant and sit in the preamble. And
//! `2 int_eq` for 46 `Char` nodes is the subset construction itself — a node
//! whose incoming mark the optimizer proved constant zero has nothing to
//! compare, so its comparison is not in the loop at all.
//!
//! Association is free, as it should be: the depth-26 left-associated tree
//! produces the same 194-op body as the depth-8 balanced one.
//!
//! The post also prints one trace in full — for `(a|b)*`, four nodes, small
//! enough to read line by line — and the test below reproduces it op for op,
//! same ops in the same order, on both backends. Ours is that listing minus
//! two reads: the repetition's mark, which the preamble reads once and the
//! back edge then carries in a register, and `len(s0)`, which `OptPure`
//! hoists when `s` is invariant. 15 ops per character, and
//! no `getfield_gc_i` at all.

use crate::regex::{KIND_ALTERNATIVE, KIND_CHAR, KIND_REPETITION, KIND_SEQUENCE, NodeRec};
use majit_metainterp::JitDriver;

/// Shift one mark through the tree for character `c`.
///
/// `interp::shift`, arm for arm — the kind tags are the same constants and
/// the arms are in the same order, so the two bodies diff cleanly.
#[majit_macros::jit_inline(
    ref_params = { n: ref(NodeRec) },
    ref_fields = {
        NodeRec::left => NodeRec,
        NodeRec::right => NodeRec,
    },
    int_fields = {
        NodeRec::kind => u8,
        NodeRec::ch => u8,
        NodeRec::empty => bool,
        NodeRec::marked => bool,
    },
)]
fn shift(n: usize, c: i64, mark: bool) -> bool {
    let k = n.kind;
    let m = match k {
        KIND_CHAR => {
            let ch = n.ch as i64;
            mark & (ch == c)
        }
        KIND_ALTERNATIVE => shift(n.left, c, mark) | shift(n.right, c, mark),
        KIND_REPETITION => shift(n.left, c, mark | n.marked),
        KIND_SEQUENCE => {
            // The left mark from the PREVIOUS character is what enters the
            // right side, so read it before `shift` overwrites it.
            let l = n.left;
            let r = n.right;
            let old_left = l.marked;
            let marked_left = shift(l, c, mark);
            let marked_right = shift(r, c, old_left | (mark & l.empty));
            (marked_left & r.empty) | marked_right
        }
        // Epsilon, and anything else: no mark ever comes out.
        _ => false,
    };
    n.marked = m;
    m
}

// ── the portal ─────────────────────────────────────────────────────────────

pub type Bytecode = [u8];

/// Dummy env the portal ABI still threads as a green. The loop does not
/// dispatch on it; `pc` stays 0 so the green key is the regex.
pub const PROGRAM: [u8; 1] = [0];

/// `marked.py` `match` reds: position, running result, the `rstr.STR`.
struct MatchState {
    pos: i64,
    result: i64,
    /// `rstr.STR` pointer (`ll_strlen` / `ll_strgetitem`).
    s: usize,
}

pub static COMPILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static LAST_OPS_AFTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static LAST_BODY: std::sync::Mutex<Vec<majit_ir::OpCode>> = std::sync::Mutex::new(Vec::new());
/// Bridges compiled, aborts, and guard failures.
///
/// This portal has the same four counters as `shortcircuit`'s for one reason:
/// the branching module's whole finding is a comparison against this one, and
/// a side that reports "not measured" cannot be the control. A control that
/// answers zero is a result; a control that answers `None` is an omission.
pub static BRIDGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ABORTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Guard failures, counted only while [`GUARD_FAILURE_PROBE`] is set.
pub static GUARD_FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Whether [`mainloop`] installs the guard-failure callback at all.
///
/// The other three fire on compile and abort events, a handful of times per
/// run and never on the path a character takes. This one fires on every deopt,
/// so leaving it installed would put an instrument inside the quantity the
/// timing rows measure. It is installed per call, and the timed rows run with
/// it off — the same discipline `shortcircuit` uses, which is what keeps the
/// two portals' timed rows comparable to each other.
pub static GUARD_FAILURE_PROBE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[majit_macros::jit_interp(
    state = MatchState,
    env = Bytecode,
    greens = [pc, program, root: ref],
    state_fields = {
        pos: int,
        result: int,
        s: str,
    },
    ref_fields = {
        NodeRec::left => NodeRec,
        NodeRec::right => NodeRec,
    },
    int_fields = {
        NodeRec::kind => u8,
        NodeRec::ch => u8,
        NodeRec::empty => bool,
        NodeRec::marked => bool,
    },
    calls = { shift => inline_int },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(
    program: &Bytecode,
    mut driver: &mut JitDriver<MatchState>,
    root: usize,
    s: usize,
    first: i64,
) -> i64 {
    let mut pc: usize = 0;
    let mut state = MatchState {
        pos: 1i64,
        result: first,
        s,
    };
    // marked.py: first char is outside; the header tick is what
    // `rewrite_can_enter_jit` inserts in front of `jit_merge_point`.
    // `while i < len(s)` records `Strlen`; `ord(s[i])` records `Strgetitem`.
    while state.pos < crate::rstr::ll_strlen(state.s) {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        let c = crate::rstr::ll_strgetitem(state.s, state.pos);
        state.result = shift(root, c, false) as i64;
        state.pos = state.pos + 1i64;
    }
    state.result
}

/// A portal instance with warmspot.py ownership: its JitDriver and compiled
/// loop cache survive across calls. The lowered regex is a ref green, matching
/// `marked.py`'s module-level `JitDriver(greens=['re'])`.
pub struct Matcher {
    root: *mut NodeRec,
    driver: JitDriver<MatchState>,
}

unsafe fn walk_match_roots(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let driver = unsafe { &mut *(data as *mut JitDriver<MatchState>) };
    driver.walk_rd_numb_refs(visitor);
}

impl Matcher {
    /// `rd_consts` area for this portal's driver. `NUMBERING` lives in
    /// the same walk. Call once the `Matcher` will not move.
    pub fn register_gc_roots(&self) {
        unsafe {
            majit_gc::shadow_stack::register_mutator_extra_area(
                walk_match_roots,
                &self.driver as *const JitDriver<MatchState> as *const (),
                "rd_consts",
            );
        }
    }

    pub fn new(root: *mut NodeRec, threshold: u32) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        let mut driver = JitDriver::new(threshold);
        majit_metainterp::install_jitframe_gc(&mut driver);
        driver.set_on_compile_loop(|_gk, _before, ops_after, opcodes| {
            COMPILES.fetch_add(1, Relaxed);
            LAST_OPS_AFTER.store(ops_after, Relaxed);
            *LAST_BODY.lock().unwrap() = opcodes.to_vec();
        });
        driver.set_on_compile_bridge(|_gk, _fail_index, _num_ops| {
            BRIDGES.fetch_add(1, Relaxed);
        });
        driver.set_on_trace_abort(|_gk, _permanent| {
            ABORTS.fetch_add(1, Relaxed);
        });
        if GUARD_FAILURE_PROBE.load(Relaxed) {
            driver.set_on_guard_failure(|_gk, _trace_id, _fail_index| {
                GUARD_FAILURES.fetch_add(1, Relaxed);
            });
        }
        {
            use majit_metainterp::JitState as _;
            MatchState {
                pos: 0,
                result: 0,
                s: 0,
            }
            .build_meta(0, &PROGRAM)
            .install_canonical_liveness(&mut driver);
        }
        Self { root, driver }
    }

    pub fn matches(&mut self, s: &[u8]) -> bool {
        if s.is_empty() {
            return unsafe { (*self.root).empty };
        }
        let first = crate::interp::shift(self.root, s[0] as i64, 1);
        let result = if s.len() == 1 {
            first
        } else {
            let input = crate::rstr::RpyStr::from_bytes(s);
            mainloop(
                &PROGRAM,
                &mut self.driver,
                self.root as usize,
                input.as_usize(),
                first,
            )
        };
        crate::interp::reset(self.root);
        result != 0
    }
}

/// `interp::matches`, with the per-character loop handed to the JIT.
///
/// The first character is shifted in outside the loop — it is the one that
/// carries a mark in from the left — so the loop body is uniform and the trace
/// has no first-iteration special case.
pub fn matches(root: *mut NodeRec, s: &[u8], threshold: u32) -> bool {
    Matcher::new(root, threshold).matches(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex::{bench_regex, count, lower, nonmatching, vectors};
    use majit_ir::OpCode;
    use std::sync::atomic::Ordering;

    /// `COMPILES` / `LAST_BODY` are process-wide and libtest runs these in
    /// parallel, so EVERY test that enters the JIT takes this — not only the
    /// ones that read the counters. A test that enters without it does not read
    /// the numbers, but it does move them.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn matches_locked(root: *mut NodeRec, s: &[u8], threshold: u32) -> bool {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        matches(root, s, threshold)
    }

    /// One measured run: `(matched, loops compiled, last compiled body)`.
    ///
    /// Takes [`PROBE_LOCK`] itself, so callers must not also hold it — the
    /// plain mutex would deadlock on one thread.
    fn measure(root: *mut NodeRec, s: &[u8]) -> (bool, usize, Vec<OpCode>) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        LAST_BODY.lock().unwrap().clear();
        let matched = matches(root, s, 3);
        let compiles = COMPILES.load(Ordering::Relaxed);
        let body = LAST_BODY.lock().unwrap().clone();
        (matched, compiles, body)
    }

    /// The peeled loop body — everything from the LAST `Label` on.
    ///
    /// A compiled loop is preamble plus peeled body, and only the peeled body
    /// runs per character. Grading the whole thing double-counts, and it also
    /// charges the body for the reads the preamble hoisted out of it.
    fn loop_body(body: &[OpCode]) -> &[OpCode] {
        match body.iter().rposition(|op| *op == OpCode::Label) {
            Some(i) => &body[i..],
            None => body,
        }
    }

    /// The op census a compiled body is graded on.
    fn census(body: &[OpCode]) -> String {
        let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
        format!(
            "{} ops: {} getfield_gc_r, {} getfield_gc_i, {} setfield_gc, {} int_eq, {} guard_value",
            body.len(),
            n(OpCode::GetfieldGcR),
            n(OpCode::GetfieldGcI),
            n(OpCode::SetfieldGc),
            n(OpCode::IntEq),
            n(OpCode::GuardValue),
        )
    }

    fn degraded() -> Vec<(&'static str, &'static str)> {
        majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "MatchState")
            .map(|a| (a.arm, a.reason))
            .collect()
    }

    #[test]
    fn the_jit_matcher_agrees_with_the_interpreter() {
        for (re, s, want) in vectors() {
            let root = lower(&re);
            assert_eq!(matches_locked(root, s.as_bytes(), 3), want, "input {s:?}");
        }
    }

    /// Cold and warm must agree on an input long enough to cross the
    /// threshold: a compiled loop that disagrees is the failure this catches.
    #[test]
    fn warm_agrees_with_cold_on_a_long_input() {
        let s = nonmatching(4096, 20, 42);
        let root = lower(&bench_regex(20));
        let cold = matches_locked(root, &s, u32::MAX);
        let warm = matches_locked(root, &s, 3);
        assert!(!cold, "the benchmark input is supposed NOT to match");
        assert_eq!(warm, cold);
    }

    /// The non-matching input is the benchmark's, and it is also the input on
    /// which most of the tree carries no mark — so a trace specialized on it
    /// would still pass the test above. This one matches.
    #[test]
    fn warm_agrees_with_cold_on_a_matching_input() {
        let mut s = nonmatching(4096, 20, 42);
        // Two `a`s exactly 21 apart is the whole regex.
        s[100] = b'a';
        s[121] = b'a';
        let root = lower(&bench_regex(20));
        let cold = matches_locked(root, &s, u32::MAX);
        let warm = matches_locked(root, &s, 3);
        assert!(cold, "the reinstated pair is supposed to match");
        assert_eq!(warm, cold);
    }

    #[test]
    fn matcher_reuses_the_same_compiled_cell_across_calls() {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        let root = lower(&bench_regex(20));
        let input = nonmatching(4096, 20, 42);
        let mut matcher = Matcher::new(root, 3);

        assert!(!matcher.matches(&input));
        let after_first = COMPILES.load(Ordering::Relaxed);
        assert!(after_first > 0, "the first call did not compile a loop");

        assert!(!matcher.matches(&input));
        assert_eq!(
            COMPILES.load(Ordering::Relaxed),
            after_first,
            "a second call rebuilt the regex's JitDriver cell instead of reusing it",
        );
    }

    /// Association changes the tree depth and nothing else, and the depth is
    /// what the tracer's inline ceiling counts. 26 levels against the balanced
    /// 8 — if the ceiling refuses the deep one, this is where it shows, and the
    /// answers must still agree either way.
    #[test]
    fn the_left_associated_tree_gives_the_same_answers() {
        use crate::regex::bench_regex_left;
        let s = nonmatching(2048, 20, 7);
        let left = lower(&bench_regex_left(20));
        let balanced = lower(&bench_regex(20));

        let (deep_matched, deep_compiles, deep_body) = measure(left, &s);
        let (flat_matched, _, _) = measure(balanced, &s);
        assert_eq!(deep_matched, flat_matched, "association changed the answer");
        println!(
            "[regex-jit] depth 26: {deep_compiles} compiled, per character: {}",
            census(loop_body(&deep_body)),
        );
        assert!(
            deep_compiles > 0,
            "the depth-26 tree compiled nothing: the tracer's inline ceiling \
             refused the walk that the depth-8 tree fits inside; degraded={:?}",
            degraded(),
        );

        let mut hit = s.clone();
        hit[500] = b'a';
        hit[521] = b'a';
        assert!(matches_locked(left, &hit, 3));
    }

    /// The experiment. One trace covers the whole input, and the 93-node tree
    /// walk inside it costs no pointer read at all: every `left` / `right` is
    /// declared immutable and its base is constant, so the walk is folded away
    /// and what survives is the marks.
    ///
    /// `GetfieldGcR` is the subject and zero is the claim. The denominator is
    /// printed beside it — a body of a handful of ops would also have no
    /// pointer reads, and would mean the walk never entered the trace.
    #[test]
    fn the_tree_walk_folds_to_marks_and_comparisons() {
        let s = nonmatching(4096, 20, 42);
        let root = lower(&bench_regex(20));
        let nodes = count(root);

        let (matched, compiles, full) = measure(root, &s);
        assert!(!matched, "the benchmark input is supposed NOT to match");
        assert!(
            compiles > 0,
            "nothing compiled, so every character ran through the \
             interpreter; degraded={:?}",
            degraded(),
        );
        let body = loop_body(&full);
        println!(
            "[regex-jit] {nodes} nodes, {compiles} compiled, per character: {}",
            census(body),
        );

        let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
        let stores = n(OpCode::SetfieldGc);
        let ptr_reads = n(OpCode::GetfieldGcR);

        assert!(
            stores >= nodes,
            "the walk stored {stores} marks for a {nodes}-node tree, so it did \
             not reach every node: the recursion was cut short rather than \
             traced through; body={body:?}"
        );
        assert_eq!(
            ptr_reads, 0,
            "the walk reloaded a `left`/`right` pointer {ptr_reads} times. \
             Those reads are what `_immutable_fields_` plus a promoted root are \
             supposed to fold; body={body:?}"
        );
        assert_eq!(
            n(OpCode::Strgetitem),
            1,
            "one character is read per pass and the string is loop \
             invariant, so exactly one strgetitem belongs here; body={body:?}"
        );
        assert_eq!(
            n(OpCode::GuardValue),
            0,
            "the root promote is loop invariant and belongs in the preamble; a \
             per-pass `guard_value` means it was not hoisted; body={body:?}"
        );
    }

    // ── the post's own worked example ──────────────────────────────────────

    /// The post's listing for `(a|b)*`, line by line, beside the op of ours
    /// that stands for it.
    ///
    /// This is the whole of that listing except the two lines in [`HOISTED`],
    /// in the order the post prints them, and it is both what the test asserts
    /// and what it prints — so the two can never disagree about what the post
    /// says. `label` stands for the argument header because that is what a
    /// `Label` carries: the loop's arguments.
    const POST_BODY: [(OpCode, &str); 15] = [
        (OpCode::Label, "[i0, result0, s0]  # arguments"),
        (OpCode::Strgetitem, "char = s0[i0]      # read character"),
        (OpCode::IntEq, "i7 = char == 'a'"),
        (OpCode::IntAnd, "i8 = i5 & i7"),
        (OpCode::IntEq, "i10 = char == 'b'"),
        (OpCode::IntAnd, "i11 = i5 & i10"),
        (OpCode::SetfieldGc, "ConstPtr(ptr_chara).marked = i8"),
        (OpCode::IntOr, "i13 = i8 | i11"),
        (OpCode::SetfieldGc, "ConstPtr(ptr_charb).marked = i11"),
        (OpCode::SetfieldGc, "ConstPtr(ptr_alternative).marked = i13"),
        (OpCode::IntAdd, "i17 = i0 + 1"),
        (OpCode::SetfieldGc, "ConstPtr(ptr_repetition).marked = i13"),
        (OpCode::IntLt, "i19 = i17 < i18"),
        (
            OpCode::GuardTrue,
            "if not i19: go back to normally running match",
        ),
        (OpCode::Jump, "jump(i17, i13, s0)"),
    ];

    /// The two lines of the post's listing that our peeled body does not have,
    /// and what carries them instead.
    ///
    /// Both are reads, and neither is a read we lost: the values are still
    /// there, they just arrive as loop arguments. That is why this body has
    /// zero `getfield_gc_i` where the post's has one.
    const HOISTED: [&str; 2] = [
        "i5 = ConstPtr(ptr_repetition).marked  -- the preamble reads it once \
         and passes it into the label, and the store at the bottom of each \
         pass forwards to the next one's read, so the mark rides the back \
         edge in a register instead of being reloaded",
        "i18 = len(s0)  -- strlen(s) is pure and s is invariant, so OptPure \
         hoists it and the length rides the back edge",
    ];

    /// The RPython spelling of an opcode: `SetfieldGc` prints as `setfield_gc`,
    /// which is the vocabulary the post's listing is written in.
    fn op_name(op: OpCode) -> String {
        let mut out = String::new();
        for (i, c) in op.name().chars().enumerate() {
            if c.is_ascii_uppercase() {
                if i > 0 {
                    out.push('_');
                }
                out.push(c.to_ascii_lowercase());
            } else {
                out.push(c);
            }
        }
        out
    }

    /// The body against [`POST_BODY`], one line per op, for a reader to lay
    /// beside the listing in the post itself. Prints what is actually there,
    /// so a body that stopped matching prints its own shape rather than the
    /// expected one.
    fn beside_the_post(body: &[OpCode]) -> String {
        let mut out = String::new();
        for i in 0..body.len().max(POST_BODY.len()) {
            let ours = body.get(i).map_or("--".to_string(), |op| op_name(*op));
            let post = POST_BODY.get(i).map_or("--", |(_, line)| line);
            out += &format!("  {i:>3}  {ours:<18}  {post}\n");
        }
        for line in HOISTED {
            out += &format!("       {:<18}  {line}\n", "(not in our body)");
        }
        out
    }

    /// The post's own worked example, which is the reason the post shows a
    /// trace at all: `(a|b)*`, four nodes, small enough to read line by line.
    ///
    /// The 93-node benchmark tree above says the walk folds; this says what it
    /// folds *to*. The claim is the strong one — not "the same counts" but the
    /// same ops in the same order — so [`POST_BODY`] is asserted as a sequence
    /// and the counts below are the individual shape questions the post's
    /// listing answers, each with its own message.
    ///
    /// Two of the post's lines are missing here and both are explained in
    /// [`HOISTED`]; nothing else differs.
    #[test]
    fn the_four_node_tree_reproduces_the_posts_listing() {
        use crate::regex::{alt, ch, rep};

        // Any `a`/`b` string long enough to cross the threshold will do —
        // every one of them matches `(a|b)*`, and the fixup `nonmatching`
        // performs is about the benchmark regex, not this one. It is borrowed
        // because it is the file's source of long `a`/`b` inputs.
        let s = nonmatching(4096, 20, 42);
        let root = lower(&rep(alt(ch(b'a'), ch(b'b'))));
        let nodes = count(root);
        assert_eq!(nodes, 4, "the post's example is a 4-node tree");

        let (matched, compiles, full) = measure(root, &s);
        assert!(matched, "every `a`/`b` string matches `(a|b)*`");
        assert!(
            compiles > 0,
            "nothing compiled, so there is no trace to compare with the \
             post's; degraded={:?}",
            degraded(),
        );
        let body = loop_body(&full);
        println!(
            "[regex-jit] (a|b)*: {nodes} nodes, {compiles} compiled, \
             per character: {}\n{}",
            census(body),
            beside_the_post(body),
        );

        let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
        assert_eq!(
            n(OpCode::Strgetitem),
            1,
            "the post reads the character once per pass and so should this; \
             more than one means the buffer base or the index stopped being \
             hoisted, none means the read left the loop and the body is not \
             doing the matching; body={body:?}"
        );
        assert_eq!(
            n(OpCode::IntEq),
            2,
            "one comparison per `Char` node is what the post's listing has, \
             and `(a|b)*` has two: neither mark is provably zero here. Fewer \
             would mean a comparison was folded that cannot be — unlike the \
             93-node tree, where 44 of the 46 `Char` nodes really do have a \
             constant-zero incoming mark and drop out; body={body:?}"
        );
        assert_eq!(
            n(OpCode::SetfieldGc),
            nodes,
            "one mark stored per node, and the tree has {nodes}. Fewer means \
             the recursion did not reach every node; more means a node was \
             stored to twice, so the walk ran more than once; body={body:?}"
        );
        assert_eq!(
            n(OpCode::GetfieldGcR),
            0,
            "a pointer read means an edge of the tree walk survived the fold, \
             which is what `_immutable_fields_` plus a promoted root exist to \
             prevent; the post's listing reaches all four nodes as `ConstPtr` \
             and so must this; body={body:?}"
        );
        assert_eq!(
            n(OpCode::GuardValue),
            0,
            "the root promote is loop invariant and belongs in the preamble; \
             a per-pass `guard_value` means it was not hoisted, and the \
             specialization would then cost a guard per character; \
             body={body:?}"
        );

        let ours: Vec<OpCode> = body.to_vec();
        let post: Vec<OpCode> = POST_BODY.iter().map(|(op, _)| *op).collect();
        assert_eq!(
            ours,
            post,
            "the body is no longer the post's listing op for op. The counts \
             above can all still pass while the order changes, so this is the \
             assertion that says we reproduced the trace rather than merely \
             its census:\n{}",
            beside_the_post(body),
        );
    }
}
