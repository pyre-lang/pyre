//! Bytecode liveness analysis (codewriter/liveness.py parity).

use pyre_interpreter::bytecode::{CodeObject, Instruction};

/// Skip Cache pseudo-instructions starting at `pos`.
/// Returns the index of the first non-Cache instruction at or after `pos`.
fn skip_caches(code: &CodeObject, mut pos: usize) -> usize {
    while pos < code.instructions.len() {
        if let Some((Instruction::Cache, _)) = pyre_interpreter::decode_instruction_at(code, pos) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// codewriter/liveness.py parity: bytecode liveness analysis.
/// For each bytecode PC, tracks which locals are live (may be read
/// on some path before being reassigned) and the operand stack depth.
///
/// `is_local_live(pc, idx)` returns true only when the slot is live
/// on the backward pass AND has been stored on every forward path
/// reaching `pc` — matching `liveness.py:7-11`'s "written to before,
/// and read afterwards" definition.
///
/// TODO: port jtransform.py's `-live-` marker insertion (inserted
/// strictly post-store) so `defined_bits` becomes redundant and the
/// backward pass alone suffices, as in RPython.
pub struct LiveVars {
    /// Flat multi-word bitvector for local liveness.
    /// `live_bits[pc * words_per_pc + w]` = word w of bitvector at PC.
    /// No 64-slot cap: supports `words_per_pc * 64` locals.
    live_bits: Vec<u64>,
    /// Forward definedness bitvector (same layout as `live_bits`):
    /// bit i at PC p means local i has been stored on every path
    /// reaching p. Intersected with `live_bits` so uninitialised
    /// slots are never reported live.
    defined_bits: Vec<u64>,
    /// Number of u64 words per PC (ceil(nlocals / 64)).
    words_per_pc: usize,
    /// Stack depth at each PC (forward analysis).
    /// RPython JitCode treats stack as registers with liveness.
    /// For Python bytecodes, stack depth determines which slots are live.
    stack_depth_at: Vec<usize>,
    /// Memoized per-Python-PC `u16` depth table (`stack_depth_at`
    /// truncated to `n`, `usize::MAX` sentinel → 0, saturated to
    /// `u16::MAX`).  Built once on first `depth_at_py_pc` call: the
    /// resume/handback paths query it repeatedly per compiled code
    /// object, and `LiveVars` is immutable after construction and
    /// cached per code pointer by `liveness_for`.
    depth_u16: std::sync::OnceLock<Vec<u16>>,
}

impl LiveVars {
    /// Backward dataflow liveness analysis on Python bytecodes.
    /// Fixed-point iteration handles loops correctly.
    fn compute(code: &CodeObject) -> Self {
        use pyre_interpreter::bytecode::Instruction;
        let n = code.instructions.len();
        if n == 0 {
            return LiveVars {
                live_bits: Vec::new(),
                defined_bits: Vec::new(),
                words_per_pc: 0,
                stack_depth_at: Vec::new(),
                depth_u16: std::sync::OnceLock::new(),
            };
        }
        let nlocals = code.varnames.len().max(1);
        let words_per_pc = (nlocals + 63) / 64;
        let total = (n + 1) * words_per_pc;
        let mut live_bits = vec![0u64; total];

        // Temporary per-PC bitvector for the inner loop.
        let mut live = vec![0u64; words_per_pc];

        // Exception table: (start, end, target) ranges for implicit
        // control flow. Any instruction in [start, end) can jump to
        // target on exception — the target's live set must be unioned
        // into every PC in the range.
        // `decode_exceptiontable` yields byte offsets; liveness operates
        // in code-unit indices (offset/2).
        let exc_handlers: Vec<(usize, usize, usize, usize)> =
            pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable)
                .map(|e| {
                    // Handler entry stack depth: on raise the unwinder pops the
                    // value stack to `depth`, pushes lasti when flagged, then
                    // pushes the exception, so the handler's first instruction
                    // sees `depth + lasti + 1` values. Matches the
                    // `handler_depth_at` computation in the codewriter's
                    // `decode_exception_catch_sites`.
                    let handler_depth = e.depth as usize + usize::from(e.lasti) + 1;
                    (
                        e.start as usize / 2,
                        e.end as usize / 2,
                        e.target as usize / 2,
                        handler_depth,
                    )
                })
                .collect();

        // Fixed-point backward analysis for local liveness.
        let mut changed = true;
        while changed {
            changed = false;
            for pc in (0..n).rev() {
                let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc)
                else {
                    continue;
                };
                // Start with successor's live set.
                let next_base = (pc + 1) * words_per_pc;
                for w in 0..words_per_pc {
                    live[w] = live_bits[next_base + w];
                }
                // Branch targets: union with target's live set.
                // CPython 3.13+ jump deltas are relative to NEXT_INSTR
                // (past any cache slots). Must use skip_caches to match
                // the codewriter's target calculation.
                let next = skip_caches(code, pc + 1);
                let target: Option<usize> = match instr {
                    Instruction::JumpForward { delta } => Some(next + delta.get(op_arg).as_usize()),
                    Instruction::JumpBackward { delta }
                    | Instruction::JumpBackwardNoInterrupt { delta } => {
                        Some(next.saturating_sub(delta.get(op_arg).as_usize()))
                    }
                    Instruction::PopJumpIfTrue { delta }
                    | Instruction::PopJumpIfFalse { delta } => {
                        Some(next + delta.get(op_arg).as_usize())
                    }
                    Instruction::PopJumpIfNone { delta }
                    | Instruction::PopJumpIfNotNone { delta } => {
                        Some(next + delta.get(op_arg).as_usize())
                    }
                    Instruction::ForIter { delta } => Some(next + delta.get(op_arg).as_usize()),
                    Instruction::Send { delta } => Some(next + delta.get(op_arg).as_usize()),
                    _ => None,
                };
                if let Some(tgt) = target {
                    let tgt_base = tgt * words_per_pc;
                    if tgt_base + words_per_pc <= live_bits.len() {
                        for w in 0..words_per_pc {
                            live[w] |= live_bits[tgt_base + w];
                        }
                    }
                }
                // liveness.py:59-78 GEN/KILL for locals.
                // GEN: register is read (add to alive).
                // KILL: register is written (remove from alive).
                match instr {
                    Instruction::LoadFast { var_num }
                    | Instruction::LoadFastBorrow { var_num }
                    | Instruction::LoadFastCheck { var_num }
                    | Instruction::LoadFastAndClear { var_num } => {
                        let i = var_num.get(op_arg).as_usize();
                        let word = i / 64;
                        if word < words_per_pc {
                            live[word] |= 1u64 << (i % 64);
                        }
                    }
                    // CPython super-instructions decompose to two plain
                    // local reads before reaching RPython jitcode. 4106f6cf
                    // made pyre's portal codewriter follow that shape by
                    // lowering both halves through getarrayitem_vable_r, so
                    // liveness must GEN both locals here as well.
                    Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                    | Instruction::LoadFastLoadFast { var_nums } => {
                        let pair = var_nums.get(op_arg);
                        for i in [
                            u32::from(pair.idx_1()) as usize,
                            u32::from(pair.idx_2()) as usize,
                        ] {
                            let word = i / 64;
                            if word < words_per_pc {
                                live[word] |= 1u64 << (i % 64);
                            }
                        }
                    }
                    Instruction::StoreFast { var_num } | Instruction::DeleteFast { var_num } => {
                        let i = var_num.get(op_arg).as_usize();
                        let word = i / 64;
                        if word < words_per_pc {
                            live[word] &= !(1u64 << (i % 64));
                        }
                    }
                    // STORE_FAST_LOAD_FAST(a, b): store top-of-stack into
                    // local a (KILL a), then load local b (GEN b). Backward
                    // order is reversed: GEN(b) first, then KILL(a), so
                    // a == b leaves the slot live (the load reads a value
                    // defined elsewhere).
                    Instruction::StoreFastLoadFast { var_nums } => {
                        let pair = var_nums.get(op_arg);
                        let load_idx = u32::from(pair.idx_2()) as usize;
                        let store_idx = u32::from(pair.idx_1()) as usize;
                        let lw = load_idx / 64;
                        if lw < words_per_pc {
                            live[lw] |= 1u64 << (load_idx % 64);
                        }
                        let sw = store_idx / 64;
                        if sw < words_per_pc {
                            live[sw] &= !(1u64 << (store_idx % 64));
                        }
                    }
                    // STORE_FAST_STORE_FAST(a, b): two consecutive stores,
                    // no reads. Both locals are KILLed.
                    Instruction::StoreFastStoreFast { var_nums } => {
                        let pair = var_nums.get(op_arg);
                        for i in [
                            u32::from(pair.idx_1()) as usize,
                            u32::from(pair.idx_2()) as usize,
                        ] {
                            let word = i / 64;
                            if word < words_per_pc {
                                live[word] &= !(1u64 << (i % 64));
                            }
                        }
                    }
                    _ => {}
                }
                // Exception handlers: union with handler target's live set.
                // This handles implicit control flow edges from try blocks
                // to except handlers (not covered by explicit jump targets).
                for entry in &exc_handlers {
                    if pc >= entry.0 && pc < entry.1 {
                        let tgt = entry.2;
                        let tgt_base = tgt * words_per_pc;
                        if tgt_base + words_per_pc <= live_bits.len() {
                            for w in 0..words_per_pc {
                                live[w] |= live_bits[tgt_base + w];
                            }
                        }
                    }
                }
                let base = pc * words_per_pc;
                if (0..words_per_pc).any(|w| live_bits[base + w] != live[w]) {
                    for w in 0..words_per_pc {
                        live_bits[base + w] = live[w];
                    }
                    changed = true;
                }
            }
        }

        // Forward stack depth analysis.
        let mut stack_depth_at = vec![usize::MAX; n + 1];
        stack_depth_at[0] = 0;
        let mut sd_changed = true;
        while sd_changed {
            sd_changed = false;
            for pc in 0..n {
                let d = stack_depth_at[pc];
                if d == usize::MAX {
                    continue;
                }
                let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc)
                else {
                    // Unknown instruction: propagate same depth.
                    if stack_depth_at[pc + 1] == usize::MAX {
                        stack_depth_at[pc + 1] = d;
                        sd_changed = true;
                    }
                    continue;
                };
                let (ft_d, br_d) = stack_effects(&instr, op_arg, d);
                // Fall-through.
                if !is_unconditional_jump(&instr) && ft_d < stack_depth_at[pc + 1] {
                    stack_depth_at[pc + 1] = ft_d;
                    sd_changed = true;
                }
                // Branch target.
                if let Some(tgt) = target_pc(code, &instr, pc, op_arg) {
                    if tgt <= n && br_d < stack_depth_at[tgt] {
                        stack_depth_at[tgt] = br_d;
                        sd_changed = true;
                    }
                }
                // Exception-table edges: any instruction in [s, e) may, on
                // raise, transfer control to the handler at `tgt`, where the
                // value stack has been unwound to the fixed handler entry
                // depth. RPython flow graphs carry these as explicit exception
                // links; pyre re-derives them from the packed table. Without
                // this edge the handler PCs keep the `usize::MAX` sentinel and
                // are treated as unreachable, so the codewriter empties their
                // resume `-live-` markers (dropping the portal frame register
                // among others).
                for &(s, e, tgt, hdepth) in &exc_handlers {
                    if pc >= s && pc < e && tgt <= n && hdepth < stack_depth_at[tgt] {
                        stack_depth_at[tgt] = hdepth;
                        sd_changed = true;
                    }
                }
            }
        }

        // Forward must-analysis: a local is defined at PC `p` iff it
        // has been stored on every path reaching `p`. Entry set is
        // the function's argument slots; each join intersects (AND).
        let mut defined_bits = vec![0u64; total];
        let mut defined_known = vec![false; n + 1];

        // Entry-defined locals: positional + keyword-only + *args + **kwargs.
        // varnames layout matches CPython's localsplus: positional /
        // kwonly / vararg / varkwarg in order, matching `pack_varargs` in
        // `call.rs`.
        let entry_defined_count = {
            let mut c = code.arg_count as usize + code.kwonlyarg_count as usize;
            use pyre_interpreter::bytecode::CodeFlags;
            if code.flags.contains(CodeFlags::VARARGS) {
                c += 1;
            }
            if code.flags.contains(CodeFlags::VARKEYWORDS) {
                c += 1;
            }
            c.min(nlocals)
        };
        for i in 0..entry_defined_count {
            defined_bits[i / 64] |= 1u64 << (i % 64);
        }
        defined_known[0] = true;

        // Worklist propagation: each visited PC contributes its post-op
        // defined set to every reachable successor.
        let mut worklist: Vec<usize> = vec![0];
        let mut post = vec![0u64; words_per_pc];
        while let Some(pc) = worklist.pop() {
            if pc >= n {
                continue;
            }
            // post = defined[pc] after gen/kill for this instruction.
            let base = pc * words_per_pc;
            for w in 0..words_per_pc {
                post[w] = defined_bits[base + w];
            }
            let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc) else {
                // Unknown instruction: propagate unchanged to pc+1.
                propagate_defined(
                    &post,
                    pc + 1,
                    words_per_pc,
                    &mut defined_bits,
                    &mut defined_known,
                    &mut worklist,
                );
                continue;
            };
            match instr {
                Instruction::StoreFast { var_num } => {
                    let i = var_num.get(op_arg).as_usize();
                    let word = i / 64;
                    if word < words_per_pc {
                        post[word] |= 1u64 << (i % 64);
                    }
                }
                Instruction::StoreFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let store_idx = u32::from(pair.idx_1()) as usize;
                    let word = store_idx / 64;
                    if word < words_per_pc {
                        post[word] |= 1u64 << (store_idx % 64);
                    }
                    // idx_2 (load half) is NOT a define; reading an
                    // undefined slot via the super-inst is caught by the
                    // forward lattice staying 0.
                }
                Instruction::StoreFastStoreFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    for i in [
                        u32::from(pair.idx_1()) as usize,
                        u32::from(pair.idx_2()) as usize,
                    ] {
                        let word = i / 64;
                        if word < words_per_pc {
                            post[word] |= 1u64 << (i % 64);
                        }
                    }
                }
                Instruction::DeleteFast { var_num } | Instruction::LoadFastAndClear { var_num } => {
                    let i = var_num.get(op_arg).as_usize();
                    let word = i / 64;
                    if word < words_per_pc {
                        post[word] &= !(1u64 << (i % 64));
                    }
                }
                _ => {}
            }

            // Propagate to fallthrough (unless unconditional jump).
            if !is_unconditional_jump(&instr) {
                propagate_defined(
                    &post,
                    pc + 1,
                    words_per_pc,
                    &mut defined_bits,
                    &mut defined_known,
                    &mut worklist,
                );
            }
            // Branch target.
            if let Some(tgt) = target_pc(code, &instr, pc, op_arg) {
                if tgt <= n {
                    propagate_defined(
                        &post,
                        tgt,
                        words_per_pc,
                        &mut defined_bits,
                        &mut defined_known,
                        &mut worklist,
                    );
                }
            }
            // Exception handlers: on exception, control jumps to tgt.
            // The handler sees the caller's current `post` (locals
            // defined up to this point) — intersect into handler entry.
            for (s, e, tgt, _) in &exc_handlers {
                if pc >= *s && pc < *e && *tgt <= n {
                    propagate_defined(
                        &post,
                        *tgt,
                        words_per_pc,
                        &mut defined_bits,
                        &mut defined_known,
                        &mut worklist,
                    );
                }
            }
        }

        LiveVars {
            live_bits,
            defined_bits,
            words_per_pc,
            stack_depth_at,
            depth_u16: std::sync::OnceLock::new(),
        }
    }

    /// codewriter/liveness.py _live_vars(pc) parity — local registers.
    /// No slot-count cap: supports arbitrary number of locals.
    ///
    /// A slot counts as live only when it is (a) live on the backward
    /// pass AND (b) defined on every forward path reaching `pc`.
    pub fn is_local_live(&self, pc: usize, local_idx: usize) -> bool {
        let word = local_idx / 64;
        let bit = local_idx % 64;
        if word >= self.words_per_pc {
            return false; // index beyond tracked locals
        }
        let live = self
            .live_bits
            .get(pc * self.words_per_pc + word)
            .map_or(true, |w| (w >> bit) & 1 != 0);
        if !live {
            return false;
        }
        self.defined_bits
            .get(pc * self.words_per_pc + word)
            .map_or(false, |w| (w >> bit) & 1 != 0)
    }

    /// True iff the dataflow analysis reached `pc`. Unreachable pcs have
    /// their `stack_depth_at` slot left at the `usize::MAX` sentinel that
    /// `LiveVars::compute` seeds and must not be treated as having a real
    /// stack height. Used by the codewriter to skip liveness emission for
    /// unreachable bytecode (RPython parity: flatten_graph() emits only
    /// reachable blocks).
    pub fn is_reachable(&self, pc: usize) -> bool {
        self.stack_depth_at
            .get(pc)
            .map_or(false, |&d| d != usize::MAX)
    }

    /// Semantic operand-stack depth at `pc` (count of values above
    /// `stack_base()`, excluding locals/cells), or `None` for an
    /// unreachable pc (forward-pass `usize::MAX` sentinel). The liveness
    /// register file may keep dead temp slots live above this depth, so the
    /// resume reconstruction sizes a frame's `valuestackdepth` from this —
    /// the true logical stack top — not from the live-register count (the
    /// portal path reads the equivalent value from the encoded vable
    /// `valuestackdepth` scalar; inline frames have no such scalar).
    pub fn stack_depth_at(&self, pc: usize) -> Option<usize> {
        match self.stack_depth_at.get(pc).copied() {
            Some(d) if d != usize::MAX => Some(d),
            _ => None,
        }
    }

    /// G.4.2: per-PC stack depth in `u16`, sized to
    /// `code.instructions.len()`.
    ///
    /// Used by the codewriter and resume-coordinate derivations to populate
    /// `PyJitCodeMetadata.depth_at_py_pc`.
    ///
    /// Unreachable PCs (the forward-pass `usize::MAX` sentinel from
    /// `LiveVars::compute`) collapse to `0` since they cannot fire
    /// guards or be the resume target.  `usize` values exceeding
    /// `u16::MAX` saturate; CPython's max stack depth fits comfortably
    /// inside `u16` for realistic bytecode.
    pub fn depth_at_py_pc(&self) -> &[u16] {
        // Built once and memoized: `liveness_for` caches this `LiveVars`
        // per code pointer, and the resume/handback callers query the
        // table repeatedly per code object.
        self.depth_u16.get_or_init(|| {
            // `stack_depth_at` has length `n + 1` (entry + each instr's
            // post-state); the metadata table indexes per Python PC, so
            // truncate to `n`.
            let n = self.stack_depth_at.len().saturating_sub(1);
            self.stack_depth_at[..n]
                .iter()
                .map(|&d| {
                    if d == usize::MAX {
                        0
                    } else {
                        u16::try_from(d).unwrap_or(u16::MAX)
                    }
                })
                .collect()
        })
    }
}

/// Forward-definedness propagation helper.
///
/// Intersects `post` (defined set after the predecessor instruction)
/// into the target PC's defined set. First touch initialises with a
/// copy; subsequent touches AND, matching the must-analysis join rule.
/// Queues `target` if the target's bitset changed.
fn propagate_defined(
    post: &[u64],
    target: usize,
    words_per_pc: usize,
    defined_bits: &mut [u64],
    defined_known: &mut [bool],
    worklist: &mut Vec<usize>,
) {
    let base = target * words_per_pc;
    if base + words_per_pc > defined_bits.len() {
        return;
    }
    let mut changed = false;
    if !defined_known[target] {
        for w in 0..words_per_pc {
            defined_bits[base + w] = post[w];
        }
        defined_known[target] = true;
        changed = true;
    } else {
        for w in 0..words_per_pc {
            let merged = defined_bits[base + w] & post[w];
            if merged != defined_bits[base + w] {
                defined_bits[base + w] = merged;
                changed = true;
            }
        }
    }
    if changed {
        worklist.push(target);
    }
}

pub(crate) fn stack_effects(
    instr: &pyre_interpreter::bytecode::Instruction,
    op_arg: pyre_interpreter::OpArg,
    depth: usize,
) -> (usize, usize) {
    use pyre_interpreter::bytecode::Instruction;
    let d = depth as i32;
    let (ft, br) = match instr {
        // No-ops
        Instruction::Nop
        | Instruction::Resume { .. }
        | Instruction::Cache
        | Instruction::NotTaken
        | Instruction::ExtendedArg => (d, d),
        // SetupAnnotations/setup_annotations touches only the namespace: pop 0, push 0.
        Instruction::SetupAnnotations => (d, d),
        // MakeCell/make_cell replaces a localsplus slot: pop 0, push 0.
        Instruction::MakeCell { .. } => (d, d),
        // ExitInitCheck is a pyopcode.rs dispatch no-op in pyre: pop 0, push 0.
        Instruction::ExitInitCheck => (d, d),
        // Push one
        Instruction::LoadConst { .. }
        | Instruction::LoadSmallInt { .. }
        | Instruction::LoadFast { .. }
        | Instruction::LoadFastBorrow { .. }
        | Instruction::LoadFastCheck { .. }
        | Instruction::LoadFastAndClear { .. }
        | Instruction::LoadName { .. }
        | Instruction::LoadDeref { .. }
        | Instruction::LoadLocals
        | Instruction::LoadBuildClass
        | Instruction::PushNull
        | Instruction::Copy { .. }
        | Instruction::PushExcInfo => (d + 1, d + 1),
        // LoadCommonConstant/load_common_constant pops 0 and pushes the resolved constant.
        Instruction::LoadCommonConstant { .. } => (d + 1, d + 1),
        // ReturnGenerator/return_generator pops 0 and pushes the following POP_TOP placeholder.
        Instruction::ReturnGenerator => (d + 1, d + 1),
        // GetAnext's async-stub shape pops 1 and pushes 2, for net +1.
        Instruction::GetAnext => (d + 1, d + 1),
        // LOAD_GLOBAL pushes 1 or 2 values depending on the low bit of namei.
        // When namei & 1 == 1, a NULL sentinel is pushed before the result (+2).
        // When namei & 1 == 0, only the result is pushed (+1).
        Instruction::LoadGlobal { namei } => {
            let raw = namei.get(op_arg) as usize;
            if raw & 1 != 0 {
                (d + 2, d + 2)
            } else {
                (d + 1, d + 1)
            }
        }
        // Push two (super-instructions: load two locals)
        Instruction::LoadFastBorrowLoadFastBorrow { .. } | Instruction::LoadFastLoadFast { .. } => {
            (d + 2, d + 2)
        }
        // STORE_FAST_LOAD_FAST: pop one, push one (net 0)
        Instruction::StoreFastLoadFast { .. } => (d, d),
        // STORE_FAST_STORE_FAST: pop two, push none
        Instruction::StoreFastStoreFast { .. } => (d - 2, d - 2),
        // Pop one
        Instruction::PopTop
        | Instruction::StoreFast { .. }
        | Instruction::StoreName { .. }
        | Instruction::StoreGlobal { .. }
        | Instruction::StoreDeref { .. }
        | Instruction::EndSend
        // SET_FUNCTION_ATTRIBUTE pops the attribute value (TOS) and leaves the
        // function (TOS1) it was applied to.
        | Instruction::SetFunctionAttribute { .. }
        | Instruction::PopIter => (d - 1, d - 1),
        // YieldValue: sends TOS to caller, receives new value back. Net 0.
        Instruction::YieldValue { .. } => (d, d),
        // Send/send_value pops the sent value and pushes either next() or StopIteration.value.
        Instruction::Send { .. } => (d, d),
        // PopExcept: codewriter pops the saved previous exception object
        // from the value stack and restores TLS from it. Net -1.
        Instruction::PopExcept => (d - 1, d - 1),
        // No-op; pyre's `end_for()` in `pyopcode.rs` is a no-op. PopIter handles the pop.
        Instruction::EndFor => (d, d),
        // Pop 0, push 0 (identity stack effect)
        Instruction::DeleteFast { .. }
        | Instruction::DeleteName { .. }
        | Instruction::DeleteGlobal { .. }
        | Instruction::DeleteDeref { .. }
        | Instruction::Swap { .. }
        | Instruction::CopyFreeVars { .. } => (d, d),
        // Pop 1 (consume item, append/extend into collection at TOS[i])
        Instruction::ListAppend { .. }
        | Instruction::SetAdd { .. }
        | Instruction::ListExtend { .. }
        | Instruction::SetUpdate { .. }
        | Instruction::DictUpdate { .. }
        | Instruction::DictMerge { .. } => (d - 1, d - 1),
        // Pop 2 (key + value into collection at TOS[i])
        Instruction::MapAdd { .. } => (d - 2, d - 2),
        // Pop 1 push 1 (net 0)
        Instruction::UnaryNegative
        | Instruction::UnaryNot
        | Instruction::UnaryInvert
        | Instruction::GetIter
        | Instruction::GetYieldFromIter
        | Instruction::GetAiter
        | Instruction::GetAwaitable { .. } => (d, d),
        // ConvertValue/convert_value pops the input and pushes the converted value.
        Instruction::ConvertValue { .. } => (d, d),
        // FormatSimple/format_simple pops the value and pushes its formatted string.
        Instruction::FormatSimple => (d, d),
        // ToBool/to_bool pops the value and pushes its bool result.
        Instruction::ToBool => (d, d),
        // LoadFromDictOrGlobals/load_from_dict_or_globals pops the mapping and pushes the result.
        Instruction::LoadFromDictOrGlobals { .. } => (d, d),
        // LoadFromDictOrDeref/load_from_dict_or_deref pops the mapping and pushes the result.
        Instruction::LoadFromDictOrDeref { .. } => (d, d),
        // MakeFunction/opcode_make_function pops the code object and pushes the function.
        Instruction::MakeFunction => (d, d),
        // LOAD_ATTR pushes 1 (attr) for the plain form, or 2 (attr,
        // self_or_null) for the method form — mirror of `execute_load_attr`
        // `attr.is_method()` → `load_method` (which pushes `[attr, NULL]`).
        // Without the method-form +1, `depth_at_py_pc` at a following
        // method-call CALL undercounts the `null_or_self` slot, so a guard
        // that resumes AT the CALL pc to re-execute it (the list-append fold)
        // rebuilds the operand stack one slot short.  Mirrors the
        // `LoadGlobal` / `LoadSuperAttr` method-form handling above.
        Instruction::LoadAttr { namei } => {
            if namei.get(op_arg).is_method() {
                (d + 1, d + 1)
            } else {
                (d, d)
            }
        }
        // LOAD_SPECIAL pops the object and pushes [attr, self_or_null]
        // (net +1) — `execute_load_special` always routes through
        // `load_method`, so it has no plain form.  Same depth-undercount
        // class as the method-form LOAD_ATTR above (a guard resuming inside
        // the `[LoadSpecial .. consuming CALL]` window — e.g. a `with` block's
        // `__enter__` / `__exit__` call — would otherwise rebuild the operand
        // stack one slot short).
        Instruction::LoadSpecial { .. } => (d + 1, d + 1),
        // Push 1 onto TOS without popping (net +1)
        Instruction::GetLen
        | Instruction::MatchMapping
        | Instruction::MatchSequence
        | Instruction::MatchKeys
        | Instruction::ImportFrom { .. } => (d + 1, d + 1),
        // ImportName: pop 2 (fromlist + level), push 1 (module). Net -1.
        Instruction::ImportName { .. } => (d - 1, d - 1),
        // LoadSuperAttr: pop 3 (global_super, cls, self).
        // is_method=false → push 1 (result). Net -2.
        // is_method=true  → push 2 (func, self_or_null). Net -1.
        Instruction::LoadSuperAttr { .. } => {
            let is_method = (u32::from(op_arg) & 1) != 0;
            if is_method {
                (d - 1, d - 1)
            } else {
                (d - 2, d - 2)
            }
        }
        // Both exception checks have no net stack effect.
        Instruction::CheckExcMatch | Instruction::CheckEgMatch => (d, d),
        // Pop 2 push 1 (net -1)
        Instruction::BinaryOp { .. }
        | Instruction::CompareOp { .. }
        | Instruction::IsOp { .. }
        | Instruction::ContainsOp { .. } => (d - 1, d - 1),
        // Pop 2: obj + value
        Instruction::StoreAttr { .. } => (d - 2, d - 2),
        // Pop 1: obj only
        Instruction::DeleteAttr { .. } => (d - 1, d - 1),
        // Pop 3: value + obj + key. Codewriter pops 3 explicitly.
        Instruction::StoreSubscr => (d - 3, d - 3),
        // Pop 2: obj + key
        Instruction::DeleteSubscr => (d - 2, d - 2),
        // Pop 3: container + start + end, push 1 (container[start:end]). Net -2.
        Instruction::BinarySlice => (d - 2, d - 2),
        // Pop 4: value + obj + start + end
        Instruction::StoreSlice => (d - 4, d - 4),
        // Unconditional jumps
        Instruction::JumpForward { .. }
        | Instruction::JumpBackward { .. }
        | Instruction::JumpBackwardNoInterrupt { .. } => (d, d),
        // Conditional pop-and-jump: pop 1 on both paths
        Instruction::PopJumpIfTrue { .. }
        | Instruction::PopJumpIfFalse { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. } => (d - 1, d - 1),
        // ForIter: fallthrough pushes TOS_next (+1); the exhaustion branch
        // keeps the iterator on the value stack (net 0) — it stays at TOS
        // through END_FOR until the following POP_ITER pops it, so the
        // branch-target depth still counts the iterator slot.
        Instruction::ForIter { .. } => (d + 1, d),
        // WithExceptStart/with_except_start preserves five inputs and pushes the exit result.
        Instruction::WithExceptStart => (d + 1, d + 1),
        // Return
        Instruction::ReturnValue => (d - 1, d - 1),
        // Build collections: pop count, push 1
        Instruction::BuildTuple { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildList { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        // BUILD_LIST_FROM_ARG: pop iterator, push list, push iterator. Net +1.
        Instruction::BuildListFromArg => (d + 1, d + 1),
        Instruction::BuildSet { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildMap { count } => {
            let s = (count.get(op_arg) as usize * 2) as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildString { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        // BuildTemplate/build_template_op pops strings+interpolations and pushes the template.
        Instruction::BuildTemplate => (d - 1, d - 1),
        // BuildInterpolation/build_interpolation_op pops value+expression(+spec) and pushes one.
        Instruction::BuildInterpolation { format } => {
            if u32::from(format.get(op_arg)) & 1 != 0 {
                (d - 2, d - 2)
            } else {
                (d - 1, d - 1)
            }
        }
        // BuildSlice/build_slice pops two or three components and pushes one slice.
        Instruction::BuildSlice { argc } => {
            use pyre_interpreter::bytecode::BuildSliceArgCount;
            match argc.get(op_arg) {
                BuildSliceArgCount::Two => (d - 1, d - 1),
                BuildSliceArgCount::Three => (d - 2, d - 2),
            }
        }
        // CALL: pop callable + args, push result
        Instruction::Call { argc } => {
            let n = argc.get(op_arg) as usize as i32;
            (d - n - 1, d - n - 1)
        }
        // CallFunctionEx/call_function_ex pops four call operands and pushes one result.
        Instruction::CallFunctionEx => (d - 3, d - 3),
        // CallKw/call_kw pops argc+3 call operands and pushes one result.
        Instruction::CallKw { argc } => {
            let n = argc.get(op_arg) as usize as i32;
            (d - n - 2, d - n - 2)
        }
        // CallIntrinsic1/call_intrinsic_1 replaces one operand with one result.
        Instruction::CallIntrinsic1 { .. } => (d, d),
        // CallIntrinsic2/call_intrinsic_2 consumes one net slot for every intrinsic variant.
        Instruction::CallIntrinsic2 { .. } => (d - 1, d - 1),
        // Unpack
        Instruction::UnpackSequence { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - 1 + s, d - 1 + s)
        }
        // UnpackEx/unpack_ex pops one and pushes before+starred+after values.
        Instruction::UnpackEx { counts } => {
            let args = counts.get(op_arg);
            let s = args.before as i32 + 1 + args.after as i32;
            (d - 1 + s, d - 1 + s)
        }
        // FormatWithSpec/format_with_spec pops value+spec and pushes one string.
        Instruction::FormatWithSpec => (d - 1, d - 1),
        // EndAsyncFor's async-stub shape pops iterator+exception and pushes nothing.
        Instruction::EndAsyncFor => (d - 2, d - 2),
        // CleanupThrow: `(sub_iter, last_sent_val, exc_value -- none, value)`.
        Instruction::CleanupThrow => (d - 1, d - 1),
        // MATCH_CLASS: pops subject, type, names (−3) and pushes the attrs
        // tuple or None (+1). Net −2.
        Instruction::MatchClass { .. } => (d - 2, d - 2),
        // Reraise pops TOS (exception). Net -1.
        Instruction::Reraise { .. } => (d - 1, d - 1),
        // RaiseVarargs: pops argc values from stack.
        // Codewriter pops 1 when argc >= 1, 0 when argc == 0.
        Instruction::RaiseVarargs { argc } => {
            let n = argc.get(op_arg) as i32;
            (d - n, d - n)
        }
        // Default: conservative, keep same depth.
        _ => (d, d),
    };
    (ft.max(0) as usize, br.max(0) as usize)
}

/// Is the instruction an unconditional jump (no fallthrough)?
pub(crate) fn is_unconditional_jump(instr: &pyre_interpreter::bytecode::Instruction) -> bool {
    use pyre_interpreter::bytecode::Instruction;
    matches!(
        instr,
        Instruction::JumpForward { .. }
            | Instruction::JumpBackward { .. }
            | Instruction::JumpBackwardNoInterrupt { .. }
            | Instruction::ReturnValue
            | Instruction::Reraise { .. }
            | Instruction::RaiseVarargs { .. }
    )
}

/// Branch target PC for branching instructions.
/// Uses skip_caches to account for cache slots after the instruction,
/// matching the codewriter's target calculation.
pub(crate) fn target_pc(
    code: &CodeObject,
    instr: &pyre_interpreter::bytecode::Instruction,
    pc: usize,
    op_arg: pyre_interpreter::OpArg,
) -> Option<usize> {
    use pyre_interpreter::bytecode::Instruction;
    let next = skip_caches(code, pc + 1);
    match instr {
        Instruction::JumpForward { delta } => Some(next + delta.get(op_arg).as_usize()),
        Instruction::JumpBackward { delta } | Instruction::JumpBackwardNoInterrupt { delta } => {
            Some(next.saturating_sub(delta.get(op_arg).as_usize()))
        }
        Instruction::PopJumpIfTrue { delta }
        | Instruction::PopJumpIfFalse { delta }
        | Instruction::PopJumpIfNone { delta }
        | Instruction::PopJumpIfNotNone { delta } => Some(next + delta.get(op_arg).as_usize()),
        Instruction::ForIter { delta } => Some(next + delta.get(op_arg).as_usize()),
        Instruction::Send { delta } => Some(next + delta.get(op_arg).as_usize()),
        _ => None,
    }
}

/// Handler PC an exception raised at `pc` transfers to, or `None` when no
/// entry covers `pc`.
///
/// [`target_pc`] only reads a branch delta out of the instruction's own
/// operand, so it cannot see this edge: under the zero-cost exception tables
/// the target lives in `co_exceptiontable`, not in the raising opcode. The
/// edge is no less a control-flow successor for that — a caller enumerating
/// an opcode's successors needs both.
///
/// `lookup_exceptiontable` speaks byte offsets and the walk speaks code-unit
/// indices, so both ends are converted here, the same convention the
/// codewriter states at its own decode sites.
pub(crate) fn exception_target_pc(code: &CodeObject, pc: usize) -> Option<usize> {
    let offset = u32::try_from(pc * 2).ok()?;
    let (target, _depth, _lasti) =
        pyre_interpreter::pycode::lookup_exceptiontable(&code.exceptiontable, offset)?;
    Some(target as usize / 2)
}

/// Cache liveness info per CodeObject pointer.
/// codewriter/liveness.py parity: thread-local cache of computed liveness.
pub fn liveness_for(code: *const CodeObject) -> &'static LiveVars {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::hash::{BuildHasherDefault, Hasher};

    /// Pointer-keyed: `usize` identity, not SipHash. `depth_based_vsd_for_wcode`
    /// consults this cache on a back-edge, and SipHash of the `CodeObject`
    /// address was the digest paid after the JIT had already given up.
    #[derive(Default)]
    struct IdentityUsizeHasher(u64);
    impl Hasher for IdentityUsizeHasher {
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = self.0.rotate_left(8) ^ (b as u64);
            }
        }
        fn write_usize(&mut self, i: usize) {
            self.0 = i as u64;
        }
        fn finish(&self) -> u64 {
            self.0
        }
    }
    type PtrMap<V> = HashMap<usize, V, BuildHasherDefault<IdentityUsizeHasher>>;

    thread_local! {
        // SAFETY: LiveVars is computed once and never mutated.
        // The 'static lifetime is safe because CodeObject outlives the JIT.
        static CACHE: RefCell<PtrMap<Box<LiveVars>>> =
            RefCell::new(PtrMap::default());
    }
    let key = code as usize;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let entry = c.entry(key).or_insert_with(|| {
            let code_ref = unsafe { &*code };
            Box::new(LiveVars::compute(code_ref))
        });
        // SAFETY: the Box is never removed from the HashMap and LiveVars
        // is immutable, so extending the lifetime is safe within this thread.
        unsafe { &*(entry.as_ref() as *const LiveVars) }
    })
}
