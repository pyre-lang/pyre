//! The post's matcher as its own source writes it: a `shift` that branches.
//!
//! This is the module RPython's JIT produces the same trace for — see the
//! cross-check below. `jit_interp` next door is the *adapted* portal the
//! post's short-circuit remark tells you to write.
//!
//! "A JIT for Regular Expression Matching" says, without showing the numbers:
//!
//! > the JIT-version of the code needs to be adapted to use the
//! > non-short-circuiting operators `&` and `|` ... If you don't change the
//! > "and" and "or" you get a lot of assembler code generated, and it's not
//! > particularly fast
//!
//! `interp.rs` and `jit_interp.rs` both take that advice, and this crate had
//! documented it without ever measuring it. This module is the unadapted side
//! of the A/B, and therefore the post's own source: `jit_interp`'s portal
//! copied whole — same state fields, same greens, same matchless
//! `while pos < len` portal, same promote, same `_immutable_fields_` declarations, same inline
//! policy — with `shift`'s four arms written the way part 1's Python writes
//! them, `and`/`or` that stop early. The state struct's NAME is the one other
//! difference, and only so a degraded arm stays attributable; the field list is
//! identical. So the two portals differ in exactly the thing the remark is
//! about, which is what makes the comparison below an A/B rather than two
//! programs.
//!
//! # This is the module RPython's own JIT agrees with
//!
//! `rpython_original/` carries the post's matcher written in RPython — the
//! class hierarchy, `_immutable_fields_`, and a JitDriver whose green is the
//! regex — and `runner.py` puts it through `LLJitMixin`, the harness RPython's
//! own JIT tests use: the real metainterpreter, the real optimizer, the LLGraph
//! backend, in process. The loop that comes out is the trace RPython would
//! compile. Both sides scan the same 4096 bytes, pinned by an FNV-1a digest
//! asserted on both (`NONMATCHING_4096_FNV1A`).
//!
//! Peeled body, at both tree sizes:
//!
//! ```text
//!   93 nodes        RPython   this module   jit_interp
//!   getfield_gc_r         0             0            0
//!   getfield_gc_i        24            24           24
//!   setfield_gc          93            93           93
//!   int_eq                2             2            2
//!   guards               27            27            1
//!   total               153           151          194
//!
//!   21 nodes        RPython   this module   jit_interp
//!   getfield_gc_r         0             0            0
//!   getfield_gc_i         6             6            6
//!   setfield_gc          21            21           21
//!   int_eq                2             2            2
//!   guards                9             9            1
//!   total                45            43           50
//! ```
//!
//! Every structural count is exact, at both sizes and on both portals: no
//! pointer read survives, one mark store per node, and two comparisons for the
//! `Char` nodes — the subset construction, performed by every tracer here. The
//! loop tails are the same ops in the same order (`int_add`, `setfield_gc`,
//! `int_lt`, `guard_true`, `jump`). At 21 nodes even the guard count agrees
//! exactly, 9 against 9.
//!
//! `the_peeled_body_matches_the_rpython_original` asserts that table and
//! prints it, so this is a gate rather than a paragraph. RPython's column is a
//! recorded constant — the Rust suite cannot call a Python 2 — with the
//! command to re-derive it beside the constant.
//!
//! ## The gate has been shown to fail
//!
//! A passing gate means nothing until it has been watched to fail. Three
//! breaks, each applied to this file and then reverted:
//!
//! | break | what the gate did |
//! |---|---|
//! | the portal's `root` ref green removed, so the tree stops being constant | `getfield_gc_r` 0 → **84**, caught by the structural assertion |
//! | `Sequence` reads `left.marked` *after* the recursive shift instead of before | `getfield_gc_i` 24 → **1**, caught — and by six other tests, since it also changes the answer |
//! | the `Char` arm masks (`mark & (ch == c)`) instead of branching — **same answers, same specialization** | guards 26 → **28**, total 150 → **154** |
//!
//! The third is the one that shaped this gate. It was written first with the
//! branching guard count asserted as a band around RPython's 27, and the band
//! let 28 straight through: the whole suite passed with a `Char` arm silently
//! switched to the spelling the *other* portal is supposed to own. The band is
//! now an exact recorded value for majit's own two numbers, and the third
//! break fails.
//!
//! ## Speed
//!
//! `the_jit_is_worth_several_times_the_matcher_with_no_jit` measures the post's
//! headline ratio — the JIT against the same algorithm with no JIT under it —
//! in one process, over 1,048,576 characters, median of 5. On this machine,
//! `--release`, dynasm:
//!
//! ```text
//!   majit JIT : 39,773,276 chars/s   (min 34,615,751, max 40,783,960)
//!   no JIT    :  6,302,331 chars/s   (min  5,512,454, max  6,937,078)
//!   ratio     : 6.3x        the post's own: 16,500,000 / 720,000 = 22.9x
//! ```
//!
//! Ours is the smaller ratio because the denominator is better: `interp.rs`
//! through `rustc -O`, against RPython's C backend in 2010. Only the ratios
//! are comparable — never the absolute rows, sixteen years and two instruction
//! sets apart. A debug-profile run inflates the ratio for the same reason in
//! reverse, and the test says so rather than letting the number be quoted.
//!
//! The backends agree op for op on every census above; their wall-clock spread
//! is backend compilation cost, especially on bridges, rather than a different
//! traced program. The floor is 1.5x so it clears both under contention while
//! staying far above the ~1x a lost finding would read.
//!
//! This is not a speed comparison against RPython. `rpython_original/` runs
//! under `meta_interp`, which executes traces in an interpreter; timing it
//! would measure the harness. The RPython comparison is the trace-shape one,
//! and it is exact.
//!
//! **And the guards say this module, not `jit_interp`, is the post's own
//! spelling.** The post's source is Python `and`/`or`, which really do
//! short-circuit, so RPython's tracer emits branches — 27 guards. This module
//! reproduces them with explicit `if`s and lands on 27. `jit_interp` masks with
//! `&`/`|` and has 1. So `jit_interp` is the *adapted* version the post's
//! remark tells you to write, and this module is what the post's code does
//! before you take that advice.
//!
//! # Why the totals differ by two
//!
//! 153 ops (RPython) against 151 (here). The operational bodies now agree:
//! RPython's listing contains two zero-cost `debug_merge_point` operations
//! that majit's IR does not represent. Those two entries account for the
//! entire total difference.
//!
//! RPython carries the marks as `Bool`. A `Bool` local *is* a branch
//! condition, so `flatten.py` emits a plain `goto_if_not` and
//! `pyjitpl.py opimpl_goto_if_not` hands that box straight to
//! `generate_guard` — no truth test. A `Bool` field is likewise stored as it
//! stands, with no mask.
//!
//! `NodeRec.empty`, `NodeRec.marked`, and `shift` now preserve that Bool
//! source type. They still use majit's integer register bank and one-byte
//! field descriptors, exactly as `lltype.Bool` does, but no longer introduce
//! the former 24 `IntIsTrue` and two narrowing `IntAnd` operations.
//!
//! `REGEX_LISTING=1` on the census test prints both peeled bodies op by op,
//! and `runner.py --listing` prints RPython's, for diffing.
//!
//! # The measurements
//!
//! Measured on this machine, dynasm, on the post's own
//! `(a|b)*a(a|b){20}a(a|b)*` — 93 nodes, balanced — over a non-matching random
//! `a`/`b` string. Two separate claims, because they have separate evidence.
//!
//! # 1. The post's claim, about code shape. It holds.
//!
//! The peeled loop body, which is what runs per input character:
//!
//! ```text
//!            ops guard_true guard_false gf_gc_r gf_gc_i setfield_gc int_eq guard_value
//! masking    194     1           0         0      24        93        2        0
//! branching  151     6          21         0      24        93        2        0
//! ```
//!
//! One guard against twenty-seven. The masking body's single guard is the loop
//! exit — the `pos < len` test — and the branching body carries that one plus
//! twenty-six more, one per `and`/`or` whose operand the optimizer could not
//! prove constant. That is exactly the mechanism the post names: a branch the
//! tracer records becomes a guard, and masking has no branch to record.
//!
//! The surprise is the op count. The branching body is SMALLER, not bigger —
//! 151 against 194 — because a guard replaces the `int_and` / `int_or` it
//! stands in for, and because the dead side of a branch takes its ops out of
//! the trace with it. The specialization is untouched too: both bodies fold
//! the whole 93-node tree walk to zero `getfield_gc_r`, both store one mark
//! per node, and both reduce 46 `Char` nodes to 2 `int_eq`. Both compile
//! exactly one loop, and `degraded_dispatch_arms()` is empty for both, so the
//! branching body lowers completely — this is NOT the "refuses to lower"
//! outcome.
//!
//! So the post's "a lot of assembler code generated" is not about the loop
//! body: that gets smaller. It is about what the twenty-six guards go on to
//! cost, which is section 2.
//!
//! Both backends agree op for op on every number in that table, and on the
//! runtime counts below: dynasm and cranelift were measured separately and
//! differ only in the throughput rows.
//!
//! # 2. What majit's runtime then does with those guards. This is the cost.
//!
//! Over a 4096-character non-matching input, the two portals record:
//!
//! ```text
//! branching  guard failures 4080   bridges compiled 10   traces aborted 0
//! masking    guard failures    1   bridges compiled  0   traces aborted 0
//! ```
//!
//! The masking portal is the control and it is the number to read first: over
//! the same 4096 characters it leaves its compiled loop **once**. The
//! branching portal leaves it 4080 times — one guard failure per input
//! character — and it stays one per character even though ten bridges get
//! built. That is the shape of the finding: bridging works here, and it does
//! not catch up.
//!
//! The ten bridges are real machine code and they are entered. Before them
//! every failure in the run reported `tid=1` — the one compiled loop. Now the
//! failures spread across trace ids 1 through 11, the loop and its ten
//! bridges, which is what a bridge failing its OWN guard looks like: the loop
//! covers one path through the mark pattern, a bridge covers the next, that
//! bridge exits at a guard of its own, and the next bridge grows off THAT.
//! `must_compile` fires 10 times and all 10 reach `compile_trace`; none abort.
//!
//! Ten is not a ceiling the bridge tree ran into; it is a rate, and eight
//! times the input is what tells those apart:
//!
//! ```text
//!  4,096 chars:  4,080 guard failures, 10 bridges
//! 32,768 chars: 32,676 guard failures, 84 bridges
//! ```
//!
//! 8.0x the failures, 8.4x the bridges, and the failures stay at 0.997 per
//! character at both lengths. `trace_eagerness` is 200 — `PARAMETERS` carries
//! `rlib/jit.py`'s own default — so one guard must fail 200 times before it is
//! worth a bridge, which caps a pass of `n` characters at about `n / 200` of
//! them no matter how many distinct mark patterns the tree has. Bridging here
//! is not converging on the pattern and stopping; it is trailing it at a fixed
//! rate that the deopt outruns.
//!
//! So the post's "you get a lot of assembler code generated" IS what happens
//! here, once the bridges can be built at all — and it does not pay for
//! itself: the interpreter round trip per character stays, at every length
//! measured.
//!
//! ## What it took to get there, and what it says about majit
//!
//! Until this module was written, NO bridge was built here at all: the same
//! run recorded `guard failures 4090   bridges compiled 0   traces aborted 19`,
//! every one of the 19 aborting on the log line after its tracing session
//! opened, with zero bytecode-body ops walked —
//!
//! ```text
//! [bridgeB] multi-frame resume (7 frames) — only the root frame is seedable,
//!           giving up on the bridge
//! ```
//!
//! `shift` is a recursive `#[jit_inline]` helper, so a guard that fails inside
//! it resumes into a STACK of frames — 4 to 7 of them, the recursion depth at
//! the failing guard — and `start_bridge_tracing` refused whenever a
//! state-field driver's resume data spanned more than one, because
//! `setup_bridge_sym` seeds the root frame only. That refusal would have
//! applied to any `#[jit_interp]` machine whose hot loop calls an inlined
//! recursive helper; the masking portal was not spared it either, it simply
//! has one guard, which does not fail.
//!
//! It is fixed, in `majit-metainterp`, by doing what `resume.py
//! rebuild_from_resumedata` does: push one `MIFrame` per encoded resume
//! section, outermost first, each `setup_resume_at_op(pc)` at its own position
//! with its own liveness-decoded registers, and resume in the innermost one.
//! Two things the resume stream does not carry had to be recovered from the
//! bytecode the way `pyjitpl.py make_result_of_lastop` recovers them: which
//! callee each suspended caller is waiting on, and which of its registers
//! takes the result. `JitCode::inline_call_ending_at` is that read.
//!
//! `MAJIT_BRIDGE_DEBUG=1` now names each rebuilt stack instead of the give-up.
//! One 4096-character run rebuilds 10 of them — one 7-frame, three 6, three 5
//! and three 4 —
//!
//! ```text
//! [bridgeB] rebuild 7 frame(s): [(0, 101, 8), (1, 269, 7), (1, 269, 7),
//!           (1, 269, 7), (1, 269, 7), (1, 214, 2), (1, 131, 4)]
//! ```
//!
//! — jitcode 0 is the mainloop and jitcode 1 is `shift`, so the stack reads
//! directly as "the mainloop called `shift`, which called itself five more
//! times". The distinct `pc`s in the `shift` frames are the distinct call
//! sites: 269 is the recursive descent, 214 and 131 the shallower ones.
//!
//! # Throughput
//!
//! Absolute chars/s on this machine swings by 2x with load — the same binary
//! read 56,208 and 203,740 on the branching row an hour apart — so every
//! number below is either a ratio taken inside one loop or an A/B against a
//! binary built from the same tree minutes apart. A standalone absolute row
//! would not survive being quoted.
//!
//! ## Against RPython's own translated binaries
//!
//! `rpython_original/target.py` and `target_masking.py` are the same two
//! spellings translated with `--opt=jit` and `--opt=2`. Three rounds, each
//! running every binary once in the same loop so a load spike moves all six
//! rows together; each row the median of five timed runs after one untimed
//! warm-up, 1048576 characters at `n = 20`; the table the median of the three:
//!
//! ```text
//!   1,048,576 chars, 93 nodes         majit    RPython jit   RPython C
//!   &/|   jit_interp / target_masking 42,548,721  42,729,349  6,319,695
//!   and/or shortcircuit / target         135,690     555,383  4,514,372
//! ```
//!
//! **On the spelling the post reports, the two JITs are the same speed.** The
//! post's numbers come from the adapted `&`/`|` matcher — part 2 says so — and
//! on that row majit reads 1.00x of RPython. The headline ratio matches too:
//! 42,548,721 / 4,514,372 = 9.4x against RPython's own 9.5x.
//!
//! **On the unadapted spelling both JITs lose to their own C, and this module
//! loses harder.** RPython's `and`/`or` JIT is 8.1x slower than RPython
//! translated to C. "Not particularly fast" is a property of the spelling, not
//! of majit — upstream pays it in the same direction and the same order of
//! magnitude. This module pays it 4.1x harder than upstream does, and that
//! 4.1x is the one number in the table that is majit's own.
//!
//! ## What moved this module's row, and by how much
//!
//! The branching row was 10,154 and 9,407 chars/s when this module was
//! written, at 1-minute load 3.4-50 and 34-83, with the masking row at
//! 17.3M and the no-JIT row at 3.8M in the same two processes — so
//! masking / branching read 1706x and 1848x, and no-JIT / branching 378x and
//! 405x. Profiling that row found two early majit defects, each fixed and each
//! measured against a binary built from the same tree:
//!
//! * **The blackhole byte-interpreted `shift`.** `bhimpl_inline_call_*`
//!   (`blackhole.py`) runs `cpu.bh_call_X(adr2int(jitcode.fnaddr),
//!   ...)`: upstream's blackhole calls a callee, it does not interpret it.
//!   `#[jit_inline]` staged its helper jitcode with `fnaddr = 0`, so the whole
//!   ported `bhimpl_inline_call_*` family was unreachable and every deopt
//!   walked `shift`'s 4-to-7-frame recursion opcode by opcode: 2,142 blackhole
//!   ops per character, 319,567 blackhole frames entered over 4096 characters.
//!   With the entry staged the way `call.py:167-169` builds one, 113 ops per
//!   character and 20,236 frames — `bh-setpos == bh-frame`, so not one frame
//!   is entered from `BC_INLINE_CALL` any more. A/B at 4096 characters:
//!   172,269 against 62,900 chars/s.
//!   `shift_carries_the_native_entry_the_blackhole_calls` is the gate.
//! * **`OptContext::inputarg_refs` grew by the parent trace's length on every
//!   bridge.** It holds the canonical `_forwarded` host per InputArg, and it
//!   was a `Vec` indexed by absolute trace position — but a bridge enters with
//!   `inputarg_base = parent_high_water`, so naming one of its inputargs grew
//!   the vector across the whole parent trace, one `Rc` allocation per padding
//!   slot, on every bridge the optimizer ran including the ones it declined.
//!   `sample` put that growth and its matching frees at 56% of the run. A/B at
//!   1M characters: 140,775 against 42,065 for one shared padding `Rc`, then
//!   166,203 against 109,959 — six interleaved pairs, medians — for dropping
//!   the positional `Vec` for a map so the growth term disappears rather than
//!   becoming cheap. `optimizer.py` forwards on the box object
//!   (`op.set_forwarded(newop)`) and keeps no positional side table at all.
//!
//! Neither fix touches how OFTEN the deopt happens, only what it costs: the
//! rate is still 0.997 guard failures per character at 4096 and at 32768. The
//! verdict survives both — the branching portal is still far slower than
//! running the same algorithm with no JIT under it at all, because a character
//! that misses every bridge still pays an interpreter round trip the plain
//! matcher never pays.
//!
//! This pass closes three structural gaps after those measurements: marks keep
//! their RPython `Bool` source type (176 → 150 peeled ops), `shift`'s helper
//! JitCode is colored before flattening (36 int / 26 ref working registers →
//! 4 / 3), and one portal-owned `JitDriver` survives all matches with the regex
//! itself as a ref green. The last change exposed a shared liveness-table slot:
//! constructing the masking portal after this one made this portal decode the
//! other JitCode at the same pc. The normal hot-counter trace entry now
//! republishes its own driver-owned registry, as the force and bridge entries
//! already did. `live_portals_republish_their_own_liveness_registry` is the
//! two-live-portal regression gate.
//!
//! A release run at 4096 characters now reads about 57.1M chars/s masking and
//! 155k branching. The remaining difference is the behaviour both tracers
//! show in the census: the masking body has one non-failing loop guard, while
//! this body deopts about once per character and keeps compiling bridges.
//!
//! # The spelling the post's advice suggests literally
//!
//! Swapping `&`/`|` for `&&`/`||` and leaving the shape alone is NOT the A/B
//! this module runs, for two reasons, both measured on this machine in the
//! same session as the tables above.
//!
//! First, in a lowered body `&&` and `||` are not short-circuit operators at
//! all. `opcode_for_binop` maps `BinOp::And` / `BinOp::Or` to `IntAnd` /
//! `IntOr`, so they lower to the same two ops `&` and `|` do, and
//! `lower_bool_if` collapses `if cond { 1 } else { 0 }` back to `cond` with no
//! branch emitted. Measured: with ONLY the `Char` arm rewritten that way, the
//! peeled body carries one guard — the loop exit — exactly as the masking
//! column does. A branch has to be written as a branch whose arms are not the
//! literals `1` and `0`, which is why the arms below hand back the operand,
//! the way Python's `and` / `or` actually evaluate.
//!
//! Second, writing that spelling here is what found a majit defect, now
//! fixed. `(mark != 0) as i64 | (n.marked as i64 != 0) as i64` in the
//! `Repetition` arm made the optimizer die after the trace recorded, on both
//! backends, in `OptContext::getnullness` under `optimize_int_is_true`. The
//! `int_or` of two `int_is_true` results asked about an operand that forwarded
//! to an op whose `Rc` had been dropped: `_forwarded` held a `Weak`, so
//! `get_box_replacement` stopped one hop early and handed back an operand that
//! was still forwarding, which `getnullness` asserts cannot happen. The slot
//! now owns its target the way RPython's `_forwarded` does
//! (`majit-ir/src/forwarding.rs`). Measured after the fix: that spelling in
//! the `Repetition` arm alone, in the `Sequence` arm alone, and in all four
//! arms at once each compiles and agrees with the interpreter on both
//! backends. `majit-ir`'s
//! `operand::tests::a_forwarded_target_survives_every_other_reference_being_dropped`
//! is the gate.

use crate::regex::{KIND_ALTERNATIVE, KIND_CHAR, KIND_REPETITION, KIND_SEQUENCE, NodeRec};
use majit_metainterp::JitDriver;

/// `jit_interp::shift`, arm for arm, with every `&`/`|` replaced by the
/// short-circuiting spelling part 1's Python uses.
///
/// The `and`/`or` are transcribed as Python evaluates them, not as booleans:
/// `x and y` is `if x { y } else { x }` and `x or y` is `if x { x } else { y }`,
/// which is why the else arms hand back the operand rather than a literal.
/// That matters twice over — it is what makes the field read on the far side of
/// the operator conditional (`mark and c == self.c` never touches `self.c` when
/// the mark is clear), and it is what keeps the macro from folding the whole
/// `if` away.
///
/// The recursive `shift` calls stay where the Python puts them: both children
/// of an `Alternative` and both sides of a `Sequence` are shifted before the
/// combinator runs, because each call stores a mark and skipping one would
/// change the answer, not just the trace.
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
        // `mark and c == self.c`
        KIND_CHAR => {
            if mark {
                let ch = n.ch as i64;
                ch == c
            } else {
                mark
            }
        }
        // `marked_left or marked_right`
        KIND_ALTERNATIVE => {
            let marked_left = shift(n.left, c, mark);
            let marked_right = shift(n.right, c, mark);
            if marked_left {
                marked_left
            } else {
                marked_right
            }
        }
        // `self.re._shift(c, mark or self.marked)`
        KIND_REPETITION => {
            let inner_mark = if mark { mark } else { n.marked };
            shift(n.left, c, inner_mark)
        }
        KIND_SEQUENCE => {
            // The left mark from the PREVIOUS character is what enters the
            // right side, so read it before `shift` overwrites it.
            let l = n.left;
            let r = n.right;
            let old_left = l.marked;
            let marked_left = shift(l, c, mark);
            // `old_marked_left or (mark and self.left.empty)`
            let into_right = if old_left {
                old_left
            } else if mark {
                l.empty
            } else {
                mark
            };
            let marked_right = shift(r, c, into_right);
            // `marked_left and self.right.empty or marked_right`
            let left_closes = if marked_left { r.empty } else { marked_left };
            if left_closes {
                left_closes
            } else {
                marked_right
            }
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

/// `jit_interp::MatchState`'s fields, under a different type name.
///
/// The name is the one deliberate deviation from "identical everything else":
/// `degraded_dispatch_arms()` reports an arm under its state type, and two
/// portals sharing the name would make a degraded arm unattributable — and
/// `jit_interp`'s own test filters that list on `interp == "MatchState"`.
struct ShortCircuitState {
    pos: i64,
    result: i64,
    /// `rstr.STR` pointer (`ll_strlen` / `ll_strgetitem`).
    s: usize,
}

pub static COMPILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Bridges compiled. A guard that keeps failing grows a bridge, and bridges are
/// the machine code the post's "a lot of assembler code generated" is about —
/// the loop counter cannot see them, because a guard that declines to bridge
/// still deopts through the blackhole and leaves the loop count at 1.
pub static BRIDGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Operations in every bridge this run compiled, summed. `BRIDGES` counts how
/// many the guards earned; this counts how much trace each one carried, so a
/// per-bridge cost can be divided by the work the bridge actually holds.
pub static BRIDGE_OPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Traces abandoned. `COMPILES > 0` does not rule this out: a bridge can abort
/// while the loop stands.
pub static ABORTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Guard failures, counted only while [`GUARD_FAILURE_PROBE`] is set.
pub static GUARD_FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Whether [`Matcher::new`] installs the guard-failure callback at all.
///
/// The other three callbacks fire on compile and abort events — a handful of
/// times per run, and never on the path a character takes. This one fires on
/// every deopt, so leaving it installed would put an instrument inside the very
/// quantity the timing row measures. It is therefore installed only when a
/// probe constructs its matcher, and the timed matchers leave it off —
/// `jit_interp` gates its own the same way, which keeps the rows comparable.
pub static GUARD_FAILURE_PROBE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// The last run's aborts split by `Counters.ABORT_*` reason, indexed as
/// `majit_metainterp::jitprof::ABORT_COUNTER_KINDS`.
///
/// Read off the driver once, after the loop, because that is the only place
/// the driver is in scope; `set_on_trace_abort` reports `permanent` and not the
/// reason. Written per `matches` call, never per character.
pub static ABORT_REASONS: [std::sync::atomic::AtomicUsize; 6] = {
    const Z: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    [Z; 6]
};
static LAST_BODY: std::sync::Mutex<Vec<majit_ir::OpCode>> = std::sync::Mutex::new(Vec::new());

/// The peeled loop body of the last loop this portal compiled — everything
/// from the last `Label` on, which is what runs per input character.
///
/// A compiled loop is preamble plus peeled body. Grading the whole thing
/// double-counts, and it also charges the body for the reads the preamble
/// hoisted out of it.
pub fn last_peeled_body() -> Vec<majit_ir::OpCode> {
    let body = LAST_BODY.lock().unwrap_or_else(|e| e.into_inner());
    match body.iter().rposition(|op| *op == majit_ir::OpCode::Label) {
        Some(i) => body[i..].to_vec(),
        None => body.clone(),
    }
}

#[majit_macros::jit_interp(
    state = ShortCircuitState,
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
    mut driver: &mut JitDriver<ShortCircuitState>,
    root: usize,
    s: usize,
    first: i64,
) -> i64 {
    let mut pc: usize = 0;
    let mut state = ShortCircuitState {
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
    for (i, slot) in ABORT_REASONS.iter().enumerate() {
        slot.store(
            driver.abort_diag(i) as usize,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    state.result
}

/// A portal instance with warmspot.py ownership: its JitDriver and compiled
/// bridges survive across calls. The lowered regex is a ref green, matching
/// `marked.py`'s module-level `JitDriver(greens=['re'])`.
pub struct Matcher {
    root: *mut NodeRec,
    driver: JitDriver<ShortCircuitState>,
}

unsafe fn walk_shortcircuit_roots(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let driver = unsafe { &mut *(data as *mut JitDriver<ShortCircuitState>) };
    driver.walk_rd_numb_refs(visitor);
}

impl Matcher {
    /// `rd_consts` area for this portal's driver. `NUMBERING` lives in
    /// the same walk. Call once the `Matcher` will not move.
    pub fn register_gc_roots(&self) {
        unsafe {
            majit_gc::shadow_stack::register_mutator_extra_area(
                walk_shortcircuit_roots,
                &self.driver as *const JitDriver<ShortCircuitState> as *const (),
                "rd_consts",
            );
        }
    }

    pub fn new(root: *mut NodeRec, threshold: u32) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        let mut driver = JitDriver::new(threshold);
        majit_metainterp::install_jitframe_gc(&mut driver);
        driver.set_on_compile_loop(|_gk, _before, _ops_after, opcodes| {
            COMPILES.fetch_add(1, Relaxed);
            *LAST_BODY.lock().unwrap() = opcodes.to_vec();
        });
        driver.set_on_compile_bridge(|_gk, _fail_index, num_ops| {
            BRIDGES.fetch_add(1, Relaxed);
            BRIDGE_OPS.fetch_add(num_ops, Relaxed);
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
            ShortCircuitState {
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
        for (i, slot) in ABORT_REASONS.iter().enumerate() {
            slot.store(
                self.driver.abort_diag(i) as usize,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
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

    /// `shift` must reach the blackhole as a CALL, not as bytes to interpret.
    ///
    /// `bhimpl_inline_call_*` (`blackhole.py:1278-1319`) runs
    /// `cpu.bh_call_X(adr2int(jitcode.fnaddr), ..., jitcode.calldescr)`, so the
    /// whole 93-node walk this helper performs is one call from the blackhole's
    /// point of view. That only happens if the helper's jitcode carries the
    /// entry `JitCode(name, fnaddr, calldescr)` (`call.py:167-169`) names, and
    /// nothing else in the crate would notice its absence: without it the
    /// answers are identical and only the cost moves, which is exactly the
    /// shape a benchmark reports as noise.
    ///
    /// The signature is pinned with it. `collect_call_args` walks
    /// `arg_classes` to put the per-kind lists back into declaration order, so
    /// a wrong string there does not fail to call — it calls with the
    /// arguments transposed.
    #[test]
    fn shift_carries_the_native_entry_the_blackhole_calls() {
        let mut asm = majit_metainterp::Assembler::default();
        let jitcode = super::__majit_inline_jitcode_shift_with_asm(&mut asm);
        assert_ne!(
            jitcode.fnaddr, 0,
            "`shift` has no native entry, so every blackhole resume byte-interprets \
             its whole recursive descent",
        );
        let body = jitcode.try_body().expect("a finished jitcode has a body");
        assert_eq!(
            (
                jitcode.c_num_regs_i,
                jitcode.c_num_regs_r,
                jitcode.c_num_regs_f,
            ),
            (4, 3, 0),
            "RPython's pre-flatten register coloring regressed: the former monotonic \
             numbering needed 36 int and 26 ref working registers",
        );
        assert_eq!(
            body.calldescr.arg_classes, "rii",
            "`shift(n: usize, c: i64, mark: bool)` is one Ref then two Ints, in that order",
        );
        assert_eq!(body.calldescr.result_type, 'i');
        assert!(
            majit_metainterp::blackhole::native_entry_args_intact(&jitcode, 0),
            "pc 0 is the helper entry"
        );
        assert!(
            majit_metainterp::blackhole::native_entry_args_intact(&jitcode, 25),
            "Char ch==c is before any child call or this node's store"
        );
        assert!(
            !majit_metainterp::blackhole::native_entry_args_intact(&jitcode, 29),
            "int_eq dest overwrites mark (i1)"
        );
        assert!(
            !majit_metainterp::blackhole::native_entry_args_intact(&jitcode, 168),
            "Sequence after left child has already called; restart is unsafe"
        );
        let starts = jitcode.startpoints.as_ref().expect("assembled shift");
        let mut leftover = std::collections::BTreeMap::<u8, usize>::new();
        for &pc in starts {
            let op = jitcode.code[pc];
            if !matches!(
                op,
                majit_metainterp::jitcode::insns::BC_LIVE
                    | majit_metainterp::jitcode::insns::BC_JUMP
                    | majit_metainterp::jitcode::insns::BC_GOTO_IF_NOT
                    | majit_metainterp::jitcode::insns::BC_GOTO_IF_NOT_INT_IS_TRUE
                    | majit_metainterp::jitcode::insns::BC_INT_IS_TRUE
                    | majit_metainterp::jitcode::insns::BC_MOVE_I
                    | majit_metainterp::jitcode::insns::BC_MOVE_I_C
                    | majit_metainterp::jitcode::insns::BC_GETFIELD_GC_I
                    | majit_metainterp::jitcode::insns::BC_GETFIELD_GC_I_PURE
                    | majit_metainterp::jitcode::insns::BC_GETFIELD_GC_R
                    | majit_metainterp::jitcode::insns::BC_GETFIELD_GC_R_PURE
                    | majit_metainterp::jitcode::insns::BC_SETFIELD_GC_I
                    | majit_metainterp::jitcode::insns::BC_SETFIELD_GC_I_C
                    | majit_metainterp::jitcode::insns::BC_INT_EQ
                    | majit_metainterp::jitcode::insns::BC_INT_NE
                    | majit_metainterp::jitcode::insns::BC_INT_LT
                    | majit_metainterp::jitcode::insns::BC_INT_ADD
                    | majit_metainterp::jitcode::insns::BC_GOTO_IF_NOT_INT_EQ
                    | majit_metainterp::jitcode::insns::BC_INT_RETURN
                    | majit_metainterp::jitcode::insns::BC_INT_RETURN_C
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_R_I
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_R_R
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_R_V
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IR_I
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IR_R
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IR_V
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IRF_I
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IRF_R
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IRF_F
                    | majit_metainterp::jitcode::insns::BC_INLINE_CALL_IRF_V
            ) {
                *leftover.entry(op).or_insert(0) += 1;
            }
        }
        assert!(
            leftover.is_empty(),
            "shift jitcode still has blackhole opcodes outside run_inner: {leftover:?}"
        );
    }
    use super::*;
    use crate::regex::{bench_regex, count, lower, nonmatching, vectors};
    use majit_ir::OpCode;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    /// The counters and `LAST_BODY` — this module's and `jit_interp`'s — are
    /// process-wide, and libtest runs these in parallel. EVERY test that enters
    /// either portal takes this, not only the ones that read the numbers: a
    /// test that enters without it does not read the counters, but it moves
    /// them.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn matches_locked(root: *mut NodeRec, s: &[u8], threshold: u32) -> bool {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        matches(root, s, threshold)
    }

    /// What one measured run of either portal answers.
    struct Run {
        matched: bool,
        compiles: usize,
        /// `None` for the masking portal, which installs no bridge callback.
        bridges: Option<usize>,
        aborts: Option<usize>,
        guard_failures: Option<usize>,
        body: Vec<OpCode>,
    }

    fn peeled(full: Vec<OpCode>) -> Vec<OpCode> {
        match full.iter().rposition(|op| *op == OpCode::Label) {
            Some(i) => full[i..].to_vec(),
            None => full,
        }
    }

    /// One measured run of the short-circuit portal, guard-failure probe on.
    fn measure(root: *mut NodeRec, s: &[u8]) -> Run {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        BRIDGES.store(0, Ordering::Relaxed);
        ABORTS.store(0, Ordering::Relaxed);
        GUARD_FAILURES.store(0, Ordering::Relaxed);
        GUARD_FAILURE_PROBE.store(true, Ordering::Relaxed);
        LAST_BODY.lock().unwrap().clear();
        let matched = matches(root, s, 3);
        GUARD_FAILURE_PROBE.store(false, Ordering::Relaxed);
        Run {
            matched,
            compiles: COMPILES.load(Ordering::Relaxed),
            bridges: Some(BRIDGES.load(Ordering::Relaxed)),
            aborts: Some(ABORTS.load(Ordering::Relaxed)),
            guard_failures: Some(GUARD_FAILURES.load(Ordering::Relaxed)),
            body: last_peeled_body(),
        }
    }

    /// The same, for the masking portal next door.
    ///
    /// It carries the same four counters, so this side is a control that
    /// answers rather than one that reports "not measured": whatever the
    /// branching portal's numbers mean, they mean it against these.
    fn measure_masking(root: *mut NodeRec, s: &[u8]) -> Run {
        use crate::jit_interp;
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        jit_interp::COMPILES.store(0, Ordering::Relaxed);
        jit_interp::BRIDGES.store(0, Ordering::Relaxed);
        jit_interp::ABORTS.store(0, Ordering::Relaxed);
        jit_interp::GUARD_FAILURES.store(0, Ordering::Relaxed);
        jit_interp::GUARD_FAILURE_PROBE.store(true, Ordering::Relaxed);
        jit_interp::LAST_BODY.lock().unwrap().clear();
        let matched = jit_interp::matches(root, s, 3);
        jit_interp::GUARD_FAILURE_PROBE.store(false, Ordering::Relaxed);
        Run {
            matched,
            compiles: jit_interp::COMPILES.load(Ordering::Relaxed),
            bridges: Some(jit_interp::BRIDGES.load(Ordering::Relaxed)),
            aborts: Some(jit_interp::ABORTS.load(Ordering::Relaxed)),
            guard_failures: Some(jit_interp::GUARD_FAILURES.load(Ordering::Relaxed)),
            body: peeled(jit_interp::LAST_BODY.lock().unwrap().clone()),
        }
    }

    fn n_of(body: &[OpCode], op: OpCode) -> usize {
        body.iter().filter(|o| **o == op).count()
    }

    /// One peeled body's shape, as RPython's own JIT produced it.
    ///
    /// A recorded constant, not a computed one: reproducing it needs a Python
    /// 2 and the `rpython/` checkout, which the Rust test suite has no way to
    /// call. Recording it is what makes the comparison a gate instead of a
    /// paragraph — if majit's trace drifts away from RPython's, the assertion
    /// below fails rather than the module doc quietly going stale.
    ///
    /// To re-derive either row:
    ///
    /// ```text
    /// pypy majit/examples/regex/rpython_original/runner.py 20 4096
    /// pypy majit/examples/regex/rpython_original/runner.py 2 512
    /// ```
    struct CrossCheck {
        nodes: usize,
        /// RPython's own peeled body.
        rpython_total: usize,
        rpython_guards: usize,
        /// The four structural counts, which every column must agree on.
        getfield_gc_r: usize,
        getfield_gc_i: usize,
        setfield_gc: usize,
        int_eq: usize,
        /// majit's branching body, recorded. Its two-op difference from
        /// RPython is the two upstream debug merge points.
        /// Pinned exactly rather than as a band around RPython's: a band wide
        /// enough to hold the real gap is wide enough to hold a *changed* gap, and
        /// this was measured letting a `Char` arm silently switch from
        /// branching to masking (guards 27 -> 29) straight through.
        branching_total: usize,
        branching_guards: usize,
        /// majit's masking body, recorded, for the same reason the branching
        /// column is. This column is the other half of the A/B and it is
        /// quoted in five places -- this module's doc, `jit_interp.rs`,
        /// `regex.rs`, and `rpython_original/README.md` -- and until it was
        /// pinned here the only thing asserted about it was `mk_guards * 4 <
        /// sc_guards`, a band wide enough to hold any drift that kept the
        /// guard count near 1.
        masking_total: usize,
        masking_guards: usize,
    }

    /// The post's own benchmark regex, `(a|b)*a(a|b){20}a(a|b)*`.
    const XCHECK_93: CrossCheck = CrossCheck {
        nodes: 93,
        // RPython: guard_true 6 + guard_false 21.
        rpython_total: 153,
        rpython_guards: 27,
        getfield_gc_r: 0,
        getfield_gc_i: 24,
        setfield_gc: 93,
        int_eq: 2,
        // Two remaining ops are RPython's `debug_merge_point`s. The
        // first and/or after the two Char `int_eq`s is now
        // `guard_false` at threshold 3 — the header tick matches
        // `rewrite_can_enter_jit`, so the guard count lands on 27.
        branching_total: 151,
        branching_guards: 27,
        // masking: one guard -- the loop-exit `IntLt` -- and the 18 ops the
        // branching column spends on its other 25 guards reappear as
        // straight-line `IntAnd`/`IntOr`, so the body is LONGER than the
        // branching one while guarding 26x less. That inversion is the A/B.
        masking_total: 194,
        masking_guards: 1,
    };

    /// The node-count control: `{2}` instead of `{20}`, so 21 nodes.
    ///
    /// At this size the guard counts land exactly on top of each other.
    const XCHECK_21: CrossCheck = CrossCheck {
        nodes: 21,
        // RPython: guard_true 6 + guard_false 3.
        rpython_total: 45,
        rpython_guards: 9,
        getfield_gc_r: 0,
        getfield_gc_i: 6,
        setfield_gc: 21,
        int_eq: 2,
        branching_total: 43,
        branching_guards: 9,
        // At 21 nodes the masking body is eight ops longer than the Bool-
        // preserving branching one (50 against 43) and still carries only the
        // single loop-exit guard.
        masking_total: 50,
        masking_guards: 1,
    };

    fn guards(body: &[OpCode]) -> usize {
        n_of(body, OpCode::GuardTrue) + n_of(body, OpCode::GuardFalse)
    }

    fn census(body: &[OpCode]) -> String {
        format!(
            "{} ops: {} guard_true, {} guard_false, {} getfield_gc_r, \
             {} getfield_gc_i, {} setfield_gc, {} int_eq, {} guard_value",
            body.len(),
            n_of(body, OpCode::GuardTrue),
            n_of(body, OpCode::GuardFalse),
            n_of(body, OpCode::GetfieldGcR),
            n_of(body, OpCode::GetfieldGcI),
            n_of(body, OpCode::SetfieldGc),
            n_of(body, OpCode::IntEq),
            n_of(body, OpCode::GuardValue),
        )
    }

    /// The last run's aborts, `Counters.ABORT_*` label to count, zeroes
    /// dropped.
    fn abort_reasons() -> Vec<(&'static str, usize)> {
        majit_metainterp::jitprof::ABORT_COUNTER_KINDS
            .iter()
            .enumerate()
            .map(|(i, (_, label))| (*label, ABORT_REASONS[i].load(Ordering::Relaxed)))
            .filter(|(_, n)| *n != 0)
            .collect()
    }

    fn degraded() -> Vec<(&'static str, &'static str)> {
        majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "ShortCircuitState")
            .map(|a| (a.arm, a.reason))
            .collect()
    }

    #[test]
    fn the_short_circuit_matcher_agrees_with_the_interpreter() {
        for (re, s, want) in vectors() {
            let root = lower(&re);
            assert_eq!(matches_locked(root, s.as_bytes(), 3), want, "input {s:?}");
        }
    }

    /// Cold and warm must agree on an input long enough to cross the threshold
    /// — on one that does not match, and on one that does. A branch the tracer
    /// recorded one way and a guard that fails to re-check it shows up here and
    /// nowhere else, and the branching body has twenty-six places to get that
    /// wrong where the masking body has one.
    #[test]
    fn warm_agrees_with_cold_and_with_the_interpreter_on_a_long_input() {
        let s = nonmatching(4096, 20, 42);
        let root = lower(&bench_regex(20));
        let cold = matches_locked(root, &s, u32::MAX);
        let warm = matches_locked(root, &s, 3);
        assert!(!cold, "the benchmark input is supposed NOT to match");
        assert_eq!(warm, cold);
        assert_eq!(warm, crate::interp::matches(root, &s));

        let mut hit = s.clone();
        // Two `a`s exactly 21 apart is the whole regex.
        hit[100] = b'a';
        hit[121] = b'a';
        let cold = matches_locked(root, &hit, u32::MAX);
        let warm = matches_locked(root, &hit, 3);
        assert!(cold, "the reinstated pair is supposed to match");
        assert_eq!(warm, cold);
        assert_eq!(warm, crate::interp::matches(root, &hit));
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

    /// The header `can_enter_jit` is what `warmspot.rewrite_can_enter_jit`
    /// inserts in front of `jit_merge_point`, so threshold N traces
    /// character N — RPython's polarity. The first and/or guard after the
    /// two `int_eq`s is the witness: threshold 3 emits `guard_false`
    /// (27 guards), matching RPython. Threshold 2 specializes a different
    /// character and flips that guard.
    #[test]
    fn header_threshold_selects_the_specialized_character() {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let input = nonmatching(4096, 20, 42);
        let mut rows = [(2u32, 0usize, OpCode::Label), (3, 0, OpCode::Label)];
        GUARD_FAILURE_PROBE.store(true, Ordering::Relaxed);
        for row in &mut rows {
            let root = lower(&bench_regex(20));
            LAST_BODY.lock().unwrap().clear();
            COMPILES.store(0, Ordering::Relaxed);
            BRIDGES.store(0, Ordering::Relaxed);
            GUARD_FAILURES.store(0, Ordering::Relaxed);
            assert!(!Matcher::new(root, row.0).matches(&input));
            let body = last_peeled_body();
            row.1 = guards(&body);
            // First guard after the two Char int_eq / setfield pairs.
            if let Some(op) = body
                .iter()
                .skip_while(|op| **op != OpCode::IntEq)
                .skip(1)
                .skip_while(|op| **op != OpCode::IntEq)
                .skip(1)
                .find(|op| **op == OpCode::GuardTrue || **op == OpCode::GuardFalse)
            {
                row.2 = *op;
            }
            println!(
                "[threshold] {} -> {} guards, first post-inteq guard {:?}, \
                 {} deopt(s), {} bridge(s)",
                row.0,
                row.1,
                row.2,
                GUARD_FAILURES.load(Ordering::Relaxed),
                BRIDGES.load(Ordering::Relaxed),
            );
        }
        GUARD_FAILURE_PROBE.store(false, Ordering::Relaxed);
        assert_eq!(
            rows[1].2,
            OpCode::GuardFalse,
            "threshold 3 should specialize s[3] like RPython and emit \
             guard_false; got {:?} with {} guards",
            rows[1].2,
            rows[1].1,
        );
        assert_eq!(
            rows[1].1, 27,
            "RPython's 27-guard body should appear at threshold 3"
        );
        assert_eq!(
            rows[0].2,
            OpCode::GuardTrue,
            "threshold 2 specializes a different character"
        );
    }

    /// Same Matcher, same input, several passes: the bridge counter after each
    /// one. `bridge_was_compiled` already refuses a (trace_id, fail_index)
    /// that has a bridge; later passes compile *new* fail sites at the
    /// `trace_eagerness` rate (a bridge's own guards), not the same site
    /// again. The count therefore keeps growing at about `n / 200` per pass.
    #[test]
    fn matcher_bridge_count_across_repeated_passes() {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        BRIDGES.store(0, Ordering::Relaxed);
        let root = lower(&bench_regex(20));
        let input = nonmatching(4096, 20, 42);
        let mut matcher = Matcher::new(root, 3);
        let mut after = [0usize; 6];
        for slot in &mut after {
            assert!(!matcher.matches(&input));
            *slot = BRIDGES.load(Ordering::Relaxed);
        }
        let deltas: [usize; 6] = std::array::from_fn(|i| {
            if i == 0 {
                after[0]
            } else {
                after[i] - after[i - 1]
            }
        });
        println!(
            "[bridges-per-pass] 4096 chars: cumulative {after:?} deltas {deltas:?} loops {}",
            COMPILES.load(Ordering::Relaxed),
        );
        assert_eq!(
            COMPILES.load(Ordering::Relaxed),
            1,
            "repeated passes compiled more than the one loop",
        );
        assert!(
            after[0] > 0,
            "the first pass compiled no bridges; the later deltas are unreadable",
        );
        assert!(
            after[5] > after[0],
            "later passes compiled no new bridges ({after:?}); the eagerness \
             cascade on this body is supposed to keep trailing the mark pattern",
        );
        // A full reset of the compile decision would grow ~`after[0]` every
        // pass *and* retrace the same (trace_id, fail_index). The cascade
        // grows at a similar rate from *new* sites; the unique-source check
        // lives in the dynasm `source=` diag, not in this total.
        let _ = deltas;
    }

    #[test]
    fn live_portals_republish_their_own_liveness_registry() {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let input = nonmatching(4096, 20, 42);
        let sc_root = lower(&bench_regex(20));
        let masking_root = lower(&bench_regex(20));
        let mut sc = Matcher::new(sc_root, 3);
        let mut masking = crate::jit_interp::Matcher::new(masking_root, 3);

        // Constructing the masking portal publishes its liveness registry
        // last. The branching portal must re-publish its own before compiling;
        // otherwise identical pc values index the other portal's table.
        assert!(!sc.matches(&input));
        assert!(!masking.matches(&input));
        assert!(!sc.matches(&input));
    }

    /// RPython's own bridge behaviour on this body, recorded, so majit's can
    /// be graded against it rather than against an intuition.
    ///
    /// The branching portal deopts about once per input character and grows
    /// bridges without ever settling, and for a long time that read as a majit
    /// defect — a bridge cascade that never closes. It is not. RPython's JIT
    /// does the same thing on the same body, to within a tenth of a percent on
    /// the deopt rate. What the numbers cost to obtain is the reason they are
    /// pinned here:
    ///
    /// * `stats.get_all_loops()` counts LOOPS. `send_loop_to_backend` calls
    ///   `add_new_loop` (compile.py); `send_bridge_to_backend` does not.
    ///   Read as "loops and bridges" it reports 1 for a run that compiled 9.
    /// * `warmspot.py` pins `set_param_trace_eagerness(2)` "for tests",
    ///   against the `rlib/jit.py` PARAMETERS default of 200 that a translated
    ///   PyPy and majit both use. At 2 the same run compiles 1407 bridges.
    ///
    /// Re-derive with (`runner.py` pins the eagerness itself):
    ///
    /// ```sh
    /// pypy majit/examples/regex/rpython_original/runner.py 20 4096
    /// ```
    const RPYTHON_BRIDGES: [(usize, usize); 2] = [(4096, 9), (16384, 41)];
    /// Guard failures reaching RPython's frontend at those same two lengths —
    /// 0.9990 and 0.9998 per input character.
    const RPYTHON_GUARD_FAILURES: [(usize, usize); 2] = [(4096, 4092), (16384, 16380)];

    /// The branching portal's deopt rate and bridge growth are RPython's, not a
    /// majit defect.
    ///
    /// Both sides are asked the same question — same regex, same input, same
    /// trace eagerness — and the assertion is that majit tracks RPython, not
    /// that either number is small. A tighter band than this would be measuring
    /// the counter's phase against the input length rather than the behaviour:
    /// a bridge is earned on the 200th failure of one guard, so which guards
    /// cross the line before the input ends is a boundary effect.
    #[test]
    fn the_branching_portal_bridges_like_the_rpython_original() {
        for (&(len, rp_bridges), &(_, rp_failures)) in
            RPYTHON_BRIDGES.iter().zip(RPYTHON_GUARD_FAILURES.iter())
        {
            let s = nonmatching(len, 20, 42);
            let r = measure(lower(&bench_regex(20)), &s);
            let bridges = r.bridges.expect("the probe was installed");
            let failures = r.guard_failures.expect("the probe was installed");
            println!(
                "[bridges] {len:6} chars: majit {bridges:3} bridge(s) \
                 {failures:6} deopt(s) ({:.4}/char) | rpython {rp_bridges:3} \
                 {rp_failures:6} ({:.4}/char)",
                failures as f64 / len as f64,
                rp_failures as f64 / len as f64,
            );
            assert_eq!(
                r.compiles, 1,
                "the branching portal compiled {} loops",
                r.compiles
            );
            // The deopt rate is the finding: both sides leave compiled code
            // once per input character. This is the number that says the
            // branching body is hostile to a trace, on RPython as on majit.
            let rate = failures as f64 / len as f64;
            assert!(
                (0.90..=1.01).contains(&rate),
                "the branching portal deopted {rate:.4} times per character \
                 over {len}; RPython deopts {:.4} times, and the whole point \
                 of this row is that the two agree",
                rp_failures as f64 / len as f64,
            );
            // Bridges grow with the input on both sides, at the same rate. A
            // half-to-double band is what a threshold-crossing count supports.
            assert!(
                bridges * 2 >= rp_bridges && bridges <= rp_bridges * 2,
                "majit grew {bridges} bridge(s) over {len} characters where \
                 RPython grows {rp_bridges}; a cascade that has stopped \
                 tracking the original is what this asserts against",
            );
        }
    }

    /// The experiment's first half: what the two bodies look like.
    ///
    /// The numbers pinned here are the ones the module doc's table reports, and
    /// they are pinned as ranges wide enough to survive an optimizer change
    /// that keeps the finding and narrow enough to fail if the finding goes
    /// away. The finding is NOT "the branching body is bigger" — it is smaller,
    /// which is the surprise — it is that the branching body pays in guards.
    #[test]
    fn the_branching_body_trades_ops_for_guards() {
        let s = nonmatching(4096, 20, 42);
        let nodes = count(lower(&bench_regex(20)));

        let sc = measure(lower(&bench_regex(20)), &s);
        let mk = measure_masking(lower(&bench_regex(20)), &s);
        assert_eq!(sc.matched, mk.matched, "the two portals disagreed");
        assert!(!sc.matched, "the benchmark input is supposed NOT to match");

        println!(
            "[shortcircuit] {nodes} nodes, branching: {} loop(s), {:?} bridge(s), \
             {:?} abort(s), {:?} guard failure(s), per character: {}",
            sc.compiles,
            sc.bridges,
            sc.aborts,
            sc.guard_failures,
            census(&sc.body),
        );
        // `REGEX_LISTING=1` prints both peeled bodies op by op, which is what
        // the RPython cross-check reads: `rpython_original/runner.py
        // --listing` prints the same thing for the trace RPython's own
        // optimizer produces, and the two lists are meant to be diffed. A
        // census alone cannot tell "same ops in a different order" from "the
        // same trace".
        if std::env::var("REGEX_LISTING").is_ok() {
            for (label, body) in [("branching", &sc.body), ("masking", &mk.body)] {
                for (i, op) in body.iter().enumerate() {
                    println!("[listing:{label}] {i:4}  {op:?}");
                }
            }
        }
        println!(
            "[shortcircuit] {nodes} nodes, masking  : {} loop(s), {:?} bridge(s), \
             {:?} abort(s), {:?} guard failure(s), per character: {}",
            mk.compiles,
            mk.bridges,
            mk.aborts,
            mk.guard_failures,
            census(&mk.body),
        );

        // Both portals still reach the JIT, and the branching one lowers: a
        // refusal would make every comparison below a comparison of nothing.
        assert!(
            sc.compiles > 0,
            "the branching portal compiled nothing, so every character ran \
             interpreted; degraded={:?}",
            degraded(),
        );
        assert!(mk.compiles > 0, "the masking portal compiled nothing");
        // The control, and the single number this whole comparison rests on:
        // the masking portal leaves its compiled loop ONE time over the same
        // 4096 characters, where the branching portal leaves it 4080. Without
        // this the branching row is just a slow number; with it, the two rows
        // differ by whether the compiled code is what runs.
        let mk_failures = mk.guard_failures.expect("the probe was installed");
        assert!(
            mk_failures * 100 < s.len(),
            "the masking portal deopted {mk_failures} times over {} characters, \
             so it is not the stays-in-compiled-code control the branching \
             portal is being measured against",
            s.len(),
        );
        assert_eq!(
            mk.bridges,
            Some(0),
            "the masking portal grew a bridge, so its guards fail too and it \
             is no longer the control",
        );
        assert_eq!(
            degraded(),
            Vec::new(),
            "the branching `shift` degraded a dispatch arm; with the arm \
             interpreted the body below is not the branching body at all",
        );
        // Not asserted to be zero, because it is not: the branching portal
        // aborts, and the abort is half the mechanism. See the test below.

        // Every node is still in the trace, and the tree walk still folds: the
        // operator moved the guards, not the specialization.
        assert!(
            n_of(&sc.body, OpCode::SetfieldGc) >= nodes,
            "the branching walk stored {} marks for a {nodes}-node tree",
            n_of(&sc.body, OpCode::SetfieldGc),
        );
        assert_eq!(
            n_of(&sc.body, OpCode::GetfieldGcR),
            0,
            "the branching walk reloaded a `left`/`right` pointer",
        );

        // The finding. The masking body carries exactly one guard — the loop
        // exit — and the branching body carries that one plus one per `and`/
        // `or` the optimizer could not prove constant.
        assert_eq!(
            guards(&mk.body),
            1,
            "the masking body is supposed to carry only the loop-exit guard",
        );
        assert!(
            guards(&sc.body) >= 20,
            "the branching body carried {} guards; the whole claim is that \
             short-circuit operators become guards, and under 20 it did not \
             happen",
            guards(&sc.body),
        );
        // And it is SMALLER, not bigger: the dead side of each branch takes
        // its ops with it.
        assert!(
            sc.body.len() < mk.body.len(),
            "the branching body ({} ops) was not smaller than the masking one \
             ({} ops); the doc table says it is",
            sc.body.len(),
            mk.body.len(),
        );
    }

    /// The reproduction claim itself: majit's peeled body against the one
    /// RPython's own JIT produces for the post's own source.
    ///
    /// This is the test to read first. Everything else in this module measures
    /// majit against majit; this measures majit against the original, and it
    /// prints the two side by side so a reader sees the comparison rather than
    /// a number to go look up.
    ///
    /// What is asserted exactly, and why those four: `getfield_gc_r`,
    /// `getfield_gc_i`, `setfield_gc` and `int_eq` are the *structural* claims
    /// of the post — the tree walk folded away, one mark stored per node, and
    /// 46 `Char` nodes reduced to two comparisons by the subset construction.
    /// If majit reproduced the post, those must agree with RPython digit for
    /// digit, at both tree sizes. They do.
    ///
    /// The guard counts are asserted as a relationship, not an equality,
    /// because they are the thing the two portals are *supposed* to differ on:
    /// RPython's source short-circuits, so it guards; the `&`/`|` variant next
    /// door does not. The branching portal lands on RPython's 27 (the header
    /// tick matches `rewrite_can_enter_jit`), and the masking portal
    /// must be far below both, or the A/B is not an A/B.
    #[test]
    fn the_peeled_body_matches_the_rpython_original() {
        for want in [&XCHECK_93, &XCHECK_21] {
            let n = if want.nodes == 93 { 20 } else { 2 };
            let s = nonmatching(4096, n, 42);
            let nodes = count(lower(&bench_regex(n)));
            assert_eq!(nodes, want.nodes, "the tree changed size");

            let sc = measure(lower(&bench_regex(n)), &s);
            let mk = measure_masking(lower(&bench_regex(n)), &s);
            assert!(sc.compiles > 0 && mk.compiles > 0, "nothing compiled");

            println!("[rpython-xcheck] {nodes} nodes, peeled body");
            println!(
                "[rpython-xcheck]   {:<16}{:>9}{:>13}{:>12}",
                "", "RPython", "branching", "masking"
            );
            let rows = [
                (
                    "getfield_gc_r",
                    want.getfield_gc_r,
                    n_of(&sc.body, OpCode::GetfieldGcR),
                    n_of(&mk.body, OpCode::GetfieldGcR),
                ),
                (
                    "getfield_gc_i",
                    want.getfield_gc_i,
                    n_of(&sc.body, OpCode::GetfieldGcI),
                    n_of(&mk.body, OpCode::GetfieldGcI),
                ),
                (
                    "setfield_gc",
                    want.setfield_gc,
                    n_of(&sc.body, OpCode::SetfieldGc),
                    n_of(&mk.body, OpCode::SetfieldGc),
                ),
                (
                    "int_eq",
                    want.int_eq,
                    n_of(&sc.body, OpCode::IntEq),
                    n_of(&mk.body, OpCode::IntEq),
                ),
                (
                    "guards",
                    want.rpython_guards,
                    guards(&sc.body),
                    guards(&mk.body),
                ),
                ("total", want.rpython_total, sc.body.len(), mk.body.len()),
            ];
            for (label, rpy, got_sc, got_mk) in rows {
                println!("[rpython-xcheck]   {label:<16}{rpy:>9}{got_sc:>13}{got_mk:>12}");
            }

            // The four structural counts, exactly, on BOTH portals. These are
            // the post's structural claims — the tree walk folded away, one
            // mark per node, the `Char` nodes reduced by the subset
            // construction — so a reproduction must match RPython digit for
            // digit. `pypy majit/examples/regex/rpython_original/runner.py`
            // re-derives RPython's column.
            for (label, rpy, got_sc, got_mk) in rows.iter().take(4) {
                assert_eq!(
                    *got_sc, *rpy,
                    "{nodes} nodes: the branching body has {got_sc} {label}, \
                     RPython's has {rpy}",
                );
                assert_eq!(
                    *got_mk, *rpy,
                    "{nodes} nodes: the masking body has {got_mk} {label}, \
                     RPython's has {rpy}. The `&`/`|` rewrite is only supposed \
                     to move guards, not specialization",
                );
            }

            // majit's own two numbers, pinned exactly rather than as a band
            // around RPython's. The gap is RPython's two debug merge points;
            // a *changed* gap is a finding, not noise. A +/-1 band here was
            // measured letting a `Char` arm switch from branching to masking
            // (27 guards -> 29, 151 ops -> 155)
            // straight through, which is precisely the difference the two
            // portals are supposed to be an A/B of.
            assert_eq!(
                guards(&sc.body),
                want.branching_guards,
                "{nodes} nodes: the branching body has {} guards, and {} were \
                 recorded (RPython's own is {}). Guards are what identify this \
                 module as the port of the post's `and`/`or` source, so this \
                 number does not drift quietly: if the change is deliberate, \
                 re-record it here and in the module doc's table",
                guards(&sc.body),
                want.branching_guards,
                want.rpython_guards,
            );
            assert_eq!(
                sc.body.len(),
                want.branching_total,
                "{nodes} nodes: the branching body is {} ops against the {} \
                 recorded. RPython's is {}, and the difference between them is \
                 accounted for op by op in this module's doc — a different \
                 total means that accounting is now wrong",
                sc.body.len(),
                want.branching_total,
                want.rpython_total,
            );
            assert_eq!(
                n_of(&sc.body, OpCode::IntIsTrue),
                0,
                "{nodes} nodes: a Bool branch was widened back through int_is_true",
            );
            assert_eq!(
                n_of(&sc.body, OpCode::IntAnd),
                0,
                "{nodes} nodes: a Bool mark store was narrowed through int_and",
            );

            // The masking portal stays the other side of the A/B. The
            // relationship first, because it is the claim the post's remark is
            // about and it survives both columns moving together.
            let (sc_guards, mk_guards) = (guards(&sc.body), guards(&mk.body));
            assert!(
                mk_guards * 4 < sc_guards,
                "{nodes} nodes: the masking body has {mk_guards} guards \
                 against the branching body's {sc_guards}; the two portals no \
                 longer differ in the thing the post's remark is about",
            );
            // Then the column itself, exactly, the same way the branching one
            // is pinned. The relationship above holds for any masking guard
            // count under a quarter of the branching one, so on its own it
            // would let this column drift while four documents keep quoting
            // the numbers it used to have.
            assert_eq!(
                mk_guards, want.masking_guards,
                "{nodes} nodes: the masking body has {mk_guards} guards \
                 against the {} recorded. This column is quoted in this \
                 module's doc, `jit_interp.rs`, `regex.rs` and \
                 `rpython_original/README.md`; if the change is deliberate, \
                 re-record it in all of them",
                want.masking_guards,
            );
            assert_eq!(
                mk.body.len(),
                want.masking_total,
                "{nodes} nodes: the masking body is {} ops against the {} \
                 recorded (RPython's is {}, the branching body's is {})",
                mk.body.len(),
                want.masking_total,
                want.rpython_total,
                want.branching_total,
            );
        }
    }

    /// The mechanism behind the timing row, and the half that is free to
    /// assert: the branching body deopts once per input character.
    ///
    /// 27 guards in the peeled body and ~4092 failures over 4096 characters is
    /// not "some guards sometimes fail" — it is compiled code leaving through a
    /// guard on essentially every pass. Ten of those failures now grow a
    /// bridge, and the rest still take a blackhole deopt, because a bridge
    /// covers one more path through the mark pattern and the pattern has far
    /// more paths than one 4096-character pass can bridge. That is a
    /// per-character cost measured in microseconds against a compiled body
    /// measured in nanoseconds, and it is why the timing row below is not a
    /// percentage.
    #[test]
    fn the_branching_body_deopts_once_per_character() {
        let s = nonmatching(4096, 20, 42);
        let sc = measure(lower(&bench_regex(20)), &s);
        let failures = sc.guard_failures.expect("the probe was installed");
        println!(
            "[shortcircuit] {} chars: {failures} guard failures, {:?} bridges, \
             {:?} aborts {:?}, {} loop(s)",
            s.len(),
            sc.bridges,
            sc.aborts,
            abort_reasons(),
            sc.compiles,
        );
        assert!(
            failures * 2 > s.len(),
            "only {failures} guard failures over {} characters, so the \
             compiled loop was actually running; the timing row's explanation \
             does not hold",
            s.len(),
        );
        // Bridges ARE built now — 10 of them in this run — and the deopt
        // survives them anyway. Both halves are asserted, because each is a
        // separate way for the finding to move: a run with no bridges at all
        // is the multi-frame resume regressing back to a give-up, and a run
        // whose failures collapse is the bridge tree finally covering the mark
        // pattern, which would make the timing row wrong rather than stale.
        // Why ten and not more is measured next door, in
        // `the_bridge_count_is_set_by_trace_eagerness_not_by_a_ceiling`.
        assert!(
            sc.bridges.unwrap_or(0) > 0,
            "no bridge was compiled. Every guard here fails inside the
             recursive `shift` helper, so its resume data spans 4 to 7 frames;
             a state-field driver that cannot rebuild them all declines the
             bridge, which is where this portal used to sit",
        );
        assert_eq!(
            sc.aborts,
            Some(0),
            "a bridge attempt was abandoned; `MAJIT_BRIDGE_DEBUG=1` names the \
             decline on the `[bridgeB] DECLINE` line",
        );
    }

    /// This portal reserves no identity-only register, which is why its
    /// bridges exist at all.
    ///
    /// A state that reserves identity slots cannot be bridged across inline
    /// frames: the snapshot trim blanks that range on every non-root frame and
    /// nothing re-derives it, so `bridge_from_guard_resume_position` refuses
    /// and the blackhole serves the guard. This portal fails its guards 4 to 7
    /// frames deep, so if it ever started reserving slots every bridge count
    /// next door would silently drop to zero and the finding would read as a
    /// regression in the fix rather than in the declaration. Assert the
    /// premise instead of inferring it from the counts.
    #[test]
    fn this_portal_reserves_no_identity_slots_so_the_decline_is_inert() {
        assert_eq!(
            <ShortCircuitState as majit_metainterp::JitState>::reserved_int_identity_range(),
            None,
            "the branching portal now reserves identity slots, so every \
             multi-frame bridge is refused and the bridge counts in this \
             module are measuring a different thing",
        );
    }

    /// How many bridges a pass grows, and what sets that number.
    ///
    /// The finding next door is that ten bridges do not move the deopt rate.
    /// The obvious reading — "the bridge tree saturates" — is wrong, and this
    /// is the measurement that says so: the count is not a ceiling, it is a
    /// rate. `trace_eagerness` is 200 (`rlib/jit.py`'s own default, carried in
    /// `PARAMETERS`), so ONE guard has to fail 200 times before it is worth a
    /// bridge, and a pass of `n` characters can therefore grow at most about
    /// `n / 200` of them however many distinct mark patterns the tree has.
    ///
    /// Eight times the input is what separates a rate from a ceiling, so that
    /// is what this runs. A saturating tree would answer the same count twice.
    #[test]
    fn the_bridge_count_is_set_by_trace_eagerness_not_by_a_ceiling() {
        let short = nonmatching(4096, 20, 42);
        let long = nonmatching(4096 * 8, 20, 42);
        let a = measure(lower(&bench_regex(20)), &short);
        let b = measure(lower(&bench_regex(20)), &long);
        let (ab, bb) = (a.bridges.unwrap_or(0), b.bridges.unwrap_or(0));
        println!(
            "[shortcircuit] bridges: {} chars -> {ab}, {} chars -> {bb}; \
             guard failures {:?} -> {:?}",
            short.len(),
            long.len(),
            a.guard_failures,
            b.guard_failures,
        );
        assert!(
            bb > ab,
            "8x the input grew no more bridges ({ab} -> {bb}), so the count is \
             a ceiling after all and the module doc's explanation is wrong",
        );
    }

    /// The experiment's second half: what the two bodies cost.
    ///
    /// Short input and a loose floor, because this one runs in the normal
    /// suite. The doc table's figure — median of 5 over 1048576 characters —
    /// comes from [`the_branching_body_is_timed_at_the_full_length`], which is
    /// `#[ignore]`d because at the branching portal's rate a single pass over
    /// 1M characters takes over a minute.
    ///
    /// Both matchers survive the warm-up and all three timed calls, matching
    /// `warmspot.py`'s portal-level JitDriver ownership. This keeps a short row
    /// from grading repeated loop compilation instead of the compiled scan.
    #[test]
    fn the_branching_body_is_far_slower_per_character() {
        let s = nonmatching(1 << 12, 20, 42);
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let sc_root = lower(&bench_regex(20));
        let mk_root = lower(&bench_regex(20));
        let mut sc_matcher = Matcher::new(sc_root, 3);
        let mut mk_matcher = crate::jit_interp::Matcher::new(mk_root, 3);
        let sc = time_rate(sc_root, &s, 3, |_r, s| sc_matcher.matches(s));
        let mk = time_rate(mk_root, &s, 3, |_r, s| mk_matcher.matches(s));
        let ratio = median(&mk) / median(&sc);
        println!(
            "[shortcircuit] {} chars, branching: {:.0} chars/s, masking: \
             {:.0} chars/s, masking / branching = {ratio:.0}x",
            s.len(),
            median(&sc),
            median(&mk),
        );
        // The persistent matchers intentionally keep their different guard
        // behaviour visible: masking's one loop guard does not fail, while the
        // branching portal deopts about once per character and grows more
        // bridges on later calls. The backends differ here and not in the op
        // census because cranelift compiles those bridges more slowly than
        // dynasm. The floor stays loose because this is wall-clock on a shared
        // machine; 3x is still far above the ~1x a lost finding would read.
        assert!(
            ratio > 3.0,
            "masking was only {ratio:.1}x the branching rate at {} characters. \
             The branching trace deopts about once per character; under 3x that \
             measured finding is gone rather than merely noisy",
            s.len(),
        );
    }

    /// The post's headline ratio, measured: the JIT against the same matcher
    /// with no JIT under it.
    ///
    /// "A JIT for Regular Expression Matching" reports 16,500,000 chars/s for
    /// RPython + JIT against 720,000 for RPython translated to C — **22.9x**,
    /// and that ratio is the one quantity in the post that travels off its
    /// 2010 hardware. Both sides of it are measured here in one process, in
    /// one run, so the ratio below is comparable to it even though neither
    /// absolute row is.
    ///
    /// Ours is smaller than 22.9x, and that is expected rather than a
    /// shortfall: the denominator is `interp.rs` through `rustc -O`, which is
    /// a considerably better ahead-of-time compiler than RPython's C backend
    /// was. A bigger denominator is a smaller ratio for the same JIT.
    ///
    /// This is NOT a comparison against RPython's *speed*. `rpython_original/`
    /// runs under `meta_interp`, which executes traces in an interpreter;
    /// timing it would measure the harness. The RPython comparison is the
    /// trace-shape one above, and it is exact.
    ///
    /// Cheap enough to run by default — the two rows here are the fast ones,
    /// about a second together. The ten-minute test below is ten minutes
    /// because of the *branching* portal, which is not in this row.
    ///
    /// The floor is deliberately far under the measured value: this row is
    /// wall-clock on a shared machine, and the finding is "the JIT is worth
    /// several times the matcher it replaced", not a specific multiple. Min
    /// and max are printed beside each median so a reader can see the spread
    /// rather than take the median on trust.
    #[test]
    fn the_jit_is_worth_several_times_the_matcher_with_no_jit() {
        let s = nonmatching(TIMED_CHARS, 20, 42);
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let jit_root = lower(&bench_regex(20));
        let mut jit_matcher = crate::jit_interp::Matcher::new(jit_root, 3);
        let jit = time_rate(jit_root, &s, TIMED_RUNS, |_r, s| jit_matcher.matches(s));
        let plain = time_rate(lower(&bench_regex(20)), &s, TIMED_RUNS, |r, s| {
            crate::interp::matches(r, s)
        });
        let ratio = median(&jit) / median(&plain);
        println!(
            "[perf] {TIMED_CHARS} chars, majit JIT : {:.0} chars/s (min {:.0}, max {:.0})",
            median(&jit),
            jit[0],
            jit[jit.len() - 1],
        );
        println!(
            "[perf] {TIMED_CHARS} chars, no JIT    : {:.0} chars/s (min {:.0}, max {:.0})",
            median(&plain),
            plain[0],
            plain[plain.len() - 1],
        );
        println!(
            "[perf] majit JIT / no JIT = {ratio:.1}x   \
             (the post's own, on its 2010 machine: 16,500,000 / 720,000 = 22.9x)"
        );
        // The denominator is Rust that the test profile did or did not
        // optimize, and that changes the ratio by more than the finding is
        // worth arguing over. The JIT's own compiled code is the same either
        // way, so a debug run inflates this row rather than deflating it —
        // which is the safe direction for the floor below, and the wrong
        // direction for quoting a number.
        if cfg!(debug_assertions) {
            println!(
                "[perf] DEBUG PROFILE — `interp.rs` is unoptimized here, so both the rows \
                 above and the ratio are inflated. Quote only `--release`:"
            );
            println!(
                "[perf]   cargo test -p regex --release --no-default-features --features \
                 dynasm -- --nocapture the_jit_is_worth_several_times"
            );
        }
        assert!(
            ratio > 1.5,
            "the JIT read only {ratio:.2}x the same matcher with no JIT under \
             it, over {TIMED_CHARS} characters. The post's comparable ratio is \
             22.9x and this crate has measured 5.1x; under 1.5x the JIT is not \
             paying for itself and the reproduction claim is about shape only",
        );
    }

    /// The doc table's timing row: median of [`TIMED_RUNS`] over
    /// [`TIMED_CHARS`] characters, one untimed warm-up first so the timed runs
    /// measure compiled code and not the trace it had to record.
    ///
    /// Median rather than mean because benchmark noise is one-sided: a
    /// preemption, an interrupt or a thermal step can only make a run slower,
    /// so a mean reports a run that never happened and a rerun does not
    /// reproduce it.
    ///
    /// `#[ignore]`d: the branching portal still makes six million-character
    /// passes counting the warm-up. With the persistent driver and current
    /// bridge path the allocation-census run measured 49s end to end; that is
    /// still too expensive for every ordinary test invocation.
    /// `cargo test -p regex --release --no-default-features --features dynasm \
    /// -- --ignored --nocapture the_branching_body_is_timed_at_the_full_length`
    #[test]
    #[ignore = "~1 minute: six million-character branching scans"]
    fn the_branching_body_is_timed_at_the_full_length() {
        let s = nonmatching(TIMED_CHARS, 20, 42);
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let sc_root = lower(&bench_regex(20));
        let mk_root = lower(&bench_regex(20));
        let mut sc_matcher = Matcher::new(sc_root, 3);
        let mut mk_matcher = crate::jit_interp::Matcher::new(mk_root, 3);
        let sc = time_rate(sc_root, &s, TIMED_RUNS, |_r, s| sc_matcher.matches(s));
        let mk = time_rate(mk_root, &s, TIMED_RUNS, |_r, s| mk_matcher.matches(s));
        // The third row is the one that turns "slower" into a verdict: the same
        // algorithm with no JIT under it at all.
        let plain = time_rate(lower(&bench_regex(20)), &s, TIMED_RUNS, |r, s| {
            crate::interp::matches(r, s)
        });
        println!(
            "[shortcircuit] {TIMED_CHARS} chars, branching: {:.0} chars/s \
             (min {:.0}, max {:.0})",
            median(&sc),
            sc[0],
            sc[sc.len() - 1],
        );
        println!(
            "[shortcircuit] {TIMED_CHARS} chars, masking  : {:.0} chars/s \
             (min {:.0}, max {:.0})",
            median(&mk),
            mk[0],
            mk[mk.len() - 1],
        );
        println!(
            "[shortcircuit] {TIMED_CHARS} chars, no JIT    : {:.0} chars/s \
             (min {:.0}, max {:.0})",
            median(&plain),
            plain[0],
            plain[plain.len() - 1],
        );
        println!(
            "[shortcircuit] masking / branching = {:.0}x, \
             no-JIT / branching = {:.0}x",
            median(&mk) / median(&sc),
            median(&plain) / median(&sc),
        );
    }

    const TIMED_CHARS: usize = 1 << 20;
    const TIMED_RUNS: usize = 5;

    /// chars/s per timed run, ascending. The untimed first pass takes the page
    /// faults and the cold branch predictors, which are not what a row is about.
    fn time_rate(
        root: *mut NodeRec,
        s: &[u8],
        runs: usize,
        mut run: impl FnMut(*mut NodeRec, &[u8]) -> bool,
    ) -> Vec<f64> {
        assert!(
            !run(root, s),
            "the benchmark input is supposed NOT to match"
        );
        let mut rates = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t0 = Instant::now();
            let hit = run(root, s);
            let secs = t0.elapsed().as_secs_f64();
            assert!(!hit, "the benchmark input is supposed NOT to match");
            rates.push(s.len() as f64 / secs);
        }
        rates.sort_by(f64::total_cmp);
        rates
    }

    fn median(rates: &[f64]) -> f64 {
        rates[rates.len() / 2]
    }
}
