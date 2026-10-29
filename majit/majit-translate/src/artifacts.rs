//! Build-process transport for `CodeWriter.make_jitcodes` and `Assembler` tables.
//!
//! RPython's translator retains these objects in process. Rust build scripts
//! serialize them for the final binary; consumers supply paths and runtime
//! symbol bindings, while this module owns the wire format and validation.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::jitcode::{BhDescr, JitCode};

pub use majit_jitcode::artifacts::{JitCodeIndex, ReachableSymbolicResidualTable};

/// Wire-format version of [`EmbeddedArtifacts`]. Version 2 added the
/// reachable-symbolic-residual table to [`JitCodeIndex`]; a version-1 index
/// has no such table and is rejected rather than decoded as garbage.
/// Version 3 moves `JitCode.calldescr` to the header (`jitcode.py`
/// `__init__`), so a version-2 body blob is a different bincode layout.
/// Version 4 freezes each jitcode's `CallPath` (segments + specialization)
/// instead of its display name.
const ARTIFACT_VERSION: u32 = 4;

/// Encoded bodies remain bytes until requested, so this envelope is Send+Sync
/// without publishing translation-time graphs across threads.
#[derive(Serialize, Deserialize)]
pub struct EmbeddedArtifacts {
    version: u32,
    pub index: JitCodeIndex,
    bodies: Vec<u8>,
    descrs: Vec<u8>,
    pub symbolic_fnaddrs: Vec<(i64, String)>,
    pub liveness: Vec<u8>,
}

impl EmbeddedArtifacts {
    pub fn from_pipeline(
        pipeline: &crate::pipeline::ProgramPipelineResult,
    ) -> bincode::Result<Self> {
        let mut paths = vec![None; pipeline.jitcodes.len()];
        for (path, jitcode) in &pipeline.jitcodes_by_path {
            paths[jitcode.index()] = Some(path.clone());
        }
        let reachable = reachable_symbolic_residual_table(&pipeline.jitcodes, &pipeline.descrs);
        let (index, bodies) = JitCodeIndex::encode(&pipeline.jitcodes, paths, reachable)?;
        Ok(Self {
            version: ARTIFACT_VERSION,
            index,
            bodies,
            descrs: bincode::serialize(&pipeline.descrs)?,
            symbolic_fnaddrs: pipeline.symbolic_fnaddr_paths.clone(),
            liveness: pipeline.all_liveness.clone(),
        })
    }

    pub fn encode(&self) -> bincode::Result<Vec<u8>> {
        bincode::serialize(self)
    }

    pub fn decode(bytes: &[u8]) -> bincode::Result<Self> {
        // `version` leads the envelope. Read it alone first, so an envelope of
        // another layout is rejected by its version rather than failing
        // partway through an index shaped differently.
        let version: u32 = bincode::deserialize(bytes)?;
        if version != ARTIFACT_VERSION {
            return Err(Box::new(bincode::ErrorKind::Custom(
                "unsupported JitCode artifact version".into(),
            )));
        }
        let artifacts: Self = bincode::deserialize(bytes)?;
        artifacts.index.validate(&artifacts.bodies)?;
        Ok(artifacts)
    }

    pub fn jitcodes(&self) -> bincode::Result<Vec<Arc<JitCode>>> {
        (0..self.index.names.len())
            .map(|i| self.index.load(&self.bodies, i))
            .collect()
    }

    pub fn descrs(&self) -> bincode::Result<Vec<BhDescr>> {
        bincode::deserialize(&self.descrs)
    }
}

/// Sorted distinct symbolic residual funcptrs reachable from each jitcode,
/// plus how many jitcodes the walk entered.
///
/// A residual-call opcode at a `startpoints` pc reads `code[pc + 1]` as the
/// funcptr register. The register is a constant when it is `>= num_regs_i`,
/// and the constant is `constants_i[reg - num_regs_i]`. It is recorded when
/// [`crate::call::is_symbolic_fnaddr`] is true. An inline-call opcode reads
/// `u16::from_le_bytes(code[pc + 1..pc + 3])` as a descr index and continues
/// through [`BhDescr::JitCode`]'s `jitcode_index`. A missing body or missing
/// `startpoints` contributes the jitcode to the visit count and no further
/// edges.
pub fn reachable_symbolic_residual_table(
    jitcodes: &[Arc<JitCode>],
    descrs: &[BhDescr],
) -> ReachableSymbolicResidualTable {
    let mut values = Vec::new();
    let mut offsets = Vec::with_capacity(jitcodes.len() + 1);
    let mut visited = Vec::with_capacity(jitcodes.len());
    offsets.push(0);
    for root in 0..jitcodes.len() {
        let (targets, seen) = reachable_symbolic_residuals_from(root, jitcodes, descrs);
        values.extend(targets);
        offsets.push(
            u32::try_from(values.len())
                .expect("reachable symbolic residual table exceeds u32 offsets"),
        );
        visited.push(
            u32::try_from(seen).expect("reachable symbolic residual visit count exceeds u32"),
        );
    }
    ReachableSymbolicResidualTable {
        values,
        offsets,
        visited,
    }
}

fn reachable_symbolic_residuals_from(
    root: usize,
    jitcodes: &[Arc<JitCode>],
    descrs: &[BhDescr],
) -> (Vec<i64>, usize) {
    let mut seen = std::collections::BTreeSet::new();
    let mut targets = std::collections::BTreeSet::new();
    let mut stack = Vec::new();
    if root < jitcodes.len() {
        seen.insert(root);
        stack.push(root);
    }
    while let Some(index) = stack.pop() {
        let Some(body) = jitcodes[index].try_body() else {
            continue;
        };
        let Some(starts) = body.startpoints.as_ref() else {
            continue;
        };
        let num_regs_i = body.c_num_regs_i as usize;
        for &pc in starts {
            let Some(&opcode) = body.code.get(pc) else {
                continue;
            };
            if is_residual_call_opcode(opcode) {
                let Some(&funcptr_reg) = body.code.get(pc + 1) else {
                    continue;
                };
                let funcptr_reg = funcptr_reg as usize;
                if funcptr_reg < num_regs_i {
                    continue;
                }
                let Some(&target) = body.constants_i.get(funcptr_reg - num_regs_i) else {
                    continue;
                };
                if crate::call::is_symbolic_fnaddr(target) {
                    targets.insert(target);
                }
                continue;
            }
            if !is_inline_call_opcode(opcode) {
                continue;
            }
            let Some((&lo, &hi)) = body.code.get(pc + 1).zip(body.code.get(pc + 2)) else {
                continue;
            };
            let descr_index = u16::from_le_bytes([lo, hi]) as usize;
            if let Some(BhDescr::JitCode { jitcode_index, .. }) = descrs.get(descr_index)
                && *jitcode_index < jitcodes.len()
                && seen.insert(*jitcode_index)
            {
                stack.push(*jitcode_index);
            }
        }
    }
    (targets.into_iter().collect(), seen.len())
}

fn is_residual_call_opcode(opcode: u8) -> bool {
    matches!(
        opcode,
        crate::insns::BC_RESIDUAL_CALL_R_V
            | crate::insns::BC_RESIDUAL_CALL_IR_V
            | crate::insns::BC_RESIDUAL_CALL_IRF_V
            | crate::insns::BC_RESIDUAL_CALL_R_I
            | crate::insns::BC_RESIDUAL_CALL_IR_I
            | crate::insns::BC_RESIDUAL_CALL_IRF_I
            | crate::insns::BC_RESIDUAL_CALL_R_R
            | crate::insns::BC_RESIDUAL_CALL_IR_R
            | crate::insns::BC_RESIDUAL_CALL_IRF_R
            | crate::insns::BC_RESIDUAL_CALL_IRF_F
    )
}

fn is_inline_call_opcode(opcode: u8) -> bool {
    matches!(
        opcode,
        crate::insns::BC_INLINE_CALL
            | crate::insns::BC_INLINE_CALL_R_I
            | crate::insns::BC_INLINE_CALL_R_R
            | crate::insns::BC_INLINE_CALL_R_V
            | crate::insns::BC_INLINE_CALL_IR_I
            | crate::insns::BC_INLINE_CALL_IR_R
            | crate::insns::BC_INLINE_CALL_IR_V
            | crate::insns::BC_INLINE_CALL_IRF_I
            | crate::insns::BC_INLINE_CALL_IRF_R
            | crate::insns::BC_INLINE_CALL_IRF_F
            | crate::insns::BC_INLINE_CALL_IRF_V
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip_preserves_bodies_and_side_tables() {
        let code = Arc::new(JitCode::new("helper"));
        code.set_index(0);
        code.set_body(crate::jitcode::JitCodeBody {
            constants_i: vec![17, -4],
            ..Default::default()
        });
        let reachable = reachable_symbolic_residual_table(std::slice::from_ref(&code), &[]);
        let (index, bodies) = JitCodeIndex::encode(
            &[code],
            vec![Some(crate::parse::CallPath::from_segments([
                "module", "helper",
            ]))],
            reachable,
        )
        .unwrap();
        let mut original = EmbeddedArtifacts {
            version: ARTIFACT_VERSION,
            index,
            bodies,
            descrs: bincode::serialize(&Vec::<BhDescr>::new()).unwrap(),
            symbolic_fnaddrs: vec![(17, "runtime::helper".into())],
            liveness: vec![1, 2, 3],
        };
        let loaded = EmbeddedArtifacts::decode(&original.encode().unwrap()).unwrap();
        let codes = loaded.jitcodes().unwrap();
        assert_eq!(codes[0].name, "helper");
        assert_eq!(codes[0].index(), 0);
        assert_eq!(codes[0].body().constants_i, [17, -4]);
        assert_eq!(
            loaded.index.paths,
            [Some(crate::parse::CallPath::from_segments([
                "module", "helper"
            ]))]
        );
        assert_eq!(loaded.index.reachable_symbolic_visited, [1]);
        assert!(loaded.index.reachable_symbolic_residuals.is_empty());
        assert_eq!(loaded.symbolic_fnaddrs, original.symbolic_fnaddrs);
        assert_eq!(loaded.liveness, original.liveness);
        assert!(loaded.descrs().unwrap().is_empty());
        assert!(loaded.index.load(&loaded.bodies, 1).is_err());
        assert!(loaded.index.load(&loaded.bodies, usize::MAX).is_err());
        original.version = ARTIFACT_VERSION + 1;
        assert!(EmbeddedArtifacts::decode(&original.encode().unwrap()).is_err());
    }

    fn symbolic(tag: u64) -> i64 {
        (crate::call::SYMBOLIC_FNADDR_BASE | tag) as i64
    }

    fn push_residual(code: &mut Vec<u8>, starts: &mut Vec<usize>, reg: u8) {
        starts.push(code.len());
        code.push(crate::insns::BC_RESIDUAL_CALL_R_V);
        code.push(reg);
    }

    fn push_inline(code: &mut Vec<u8>, starts: &mut Vec<usize>, descr: u16) {
        starts.push(code.len());
        code.push(crate::insns::BC_INLINE_CALL_R_V);
        code.push(descr as u8);
        code.push((descr >> 8) as u8);
    }

    fn assembled(
        name: &str,
        index: usize,
        num_regs_i: u8,
        code: Vec<u8>,
        constants_i: Vec<i64>,
        starts: Vec<usize>,
    ) -> Arc<JitCode> {
        let jitcode = Arc::new(JitCode::new(name));
        jitcode.set_index(index);
        jitcode.set_body(crate::jitcode::JitCodeBody {
            c_num_regs_i: num_regs_i,
            code,
            constants_i,
            startpoints: Some(starts.into_iter().collect()),
            ..Default::default()
        });
        jitcode
    }

    /// A diamond plus a cycle, a non-constant funcptr, a real address, a
    /// non-startpoint byte equal to a residual opcode, a non-jitcode descr,
    /// and an out-of-range descr. Only symbolic constants at real instruction
    /// starts are recorded, once per closure.
    #[test]
    fn reachable_symbolic_residual_table_follows_inline_calls() {
        let symbolic_a = symbolic(1);
        let symbolic_b = symbolic(2);
        let symbolic_c = symbolic(3);
        let symbolic_d = symbolic(4);
        let real = 0x1234_5678i64;
        // High half set, sign bit clear, but not `SYMBOLIC_FNADDR_BASE`.
        // A host funcptr keeps bits 48..63 clear, so this word is declined
        // the same way as a symbolic hash and is recorded at a real start.
        let wrong_tag = 0x7ADE_0000_0000_0001u64 as i64;
        assert!(crate::call::is_symbolic_fnaddr(symbolic_a));
        assert!(!crate::call::is_symbolic_fnaddr(real));
        assert!(crate::call::is_symbolic_fnaddr(wrong_tag));

        let mut code = Vec::new();
        let mut starts = Vec::new();
        push_residual(&mut code, &mut starts, 1);
        push_residual(&mut code, &mut starts, 0);
        push_residual(&mut code, &mut starts, 2);
        push_residual(&mut code, &mut starts, 3);
        push_inline(&mut code, &mut starts, 0);
        push_inline(&mut code, &mut starts, 1);
        push_inline(&mut code, &mut starts, 3);
        push_inline(&mut code, &mut starts, 5);
        push_inline(&mut code, &mut starts, 4);
        // Operand bytes, not an instruction start. A linear scan would read
        // the residual opcode here and pick up `symbolic_d`.
        code.push(crate::insns::BC_INLINE_CALL_R_V);
        code.push(crate::insns::BC_RESIDUAL_CALL_R_V);
        code.push(4);

        let root = assembled(
            "root",
            0,
            1,
            code,
            vec![symbolic_a, real, wrong_tag, symbolic_d],
            starts,
        );

        let mut mid_b = Vec::new();
        let mut mid_b_starts = Vec::new();
        push_residual(&mut mid_b, &mut mid_b_starts, 0);
        push_inline(&mut mid_b, &mut mid_b_starts, 2);
        let via_b = assembled("via_b", 1, 0, mid_b, vec![symbolic_b], mid_b_starts);

        let mut mid_c = Vec::new();
        let mut mid_c_starts = Vec::new();
        push_residual(&mut mid_c, &mut mid_c_starts, 0);
        push_inline(&mut mid_c, &mut mid_c_starts, 2);
        let via_c = assembled("via_c", 2, 0, mid_c, vec![symbolic_b], mid_c_starts);

        let mut leaf = Vec::new();
        let mut leaf_starts = Vec::new();
        push_residual(&mut leaf, &mut leaf_starts, 0);
        let leaf = assembled("leaf", 3, 0, leaf, vec![symbolic_c], leaf_starts);

        let shell = Arc::new(JitCode::new("shell"));
        shell.set_index(4);

        let unscanned = Arc::new(JitCode::new("no-starts"));
        unscanned.set_index(5);
        unscanned.set_body(crate::jitcode::JitCodeBody {
            c_num_regs_i: 0,
            code: vec![crate::insns::BC_RESIDUAL_CALL_R_V, 0],
            constants_i: vec![symbolic_d],
            startpoints: None,
            ..Default::default()
        });

        let jitcodes = vec![root, via_b, via_c, leaf, shell, unscanned];
        let descrs = vec![
            BhDescr::JitCode {
                jitcode_index: 1,
                fnaddr: 0,
                calldescr: Default::default(),
            },
            BhDescr::JitCode {
                jitcode_index: 2,
                fnaddr: 0,
                calldescr: Default::default(),
            },
            BhDescr::JitCode {
                jitcode_index: 3,
                fnaddr: 0,
                calldescr: Default::default(),
            },
            BhDescr::JitCode {
                jitcode_index: 0,
                fnaddr: 0,
                calldescr: Default::default(),
            },
            BhDescr::Call {
                calldescr: Default::default(),
            },
        ];

        let table = reachable_symbolic_residual_table(&jitcodes, &descrs);
        let row = |index: usize| -> (Vec<i64>, u32) {
            let start = table.offsets[index] as usize;
            let end = table.offsets[index + 1] as usize;
            (table.values[start..end].to_vec(), table.visited[index])
        };
        assert_eq!(
            row(0),
            (vec![symbolic_a, symbolic_b, symbolic_c, wrong_tag], 4)
        );
        assert_eq!(row(1), (vec![symbolic_b, symbolic_c], 2));
        assert_eq!(row(2), (vec![symbolic_b, symbolic_c], 2));
        assert_eq!(row(3), (vec![symbolic_c], 1));
        assert_eq!(row(4), (Vec::<i64>::new(), 1));
        assert_eq!(row(5), (Vec::<i64>::new(), 1));

        let paths = vec![None; jitcodes.len()];
        let (index, bodies) = JitCodeIndex::encode(&jitcodes, paths, table.clone()).unwrap();
        let decoded = JitCodeIndex::decode(&bincode::serialize(&index).unwrap(), &bodies).unwrap();
        assert_eq!(
            decoded.reachable_symbolic_residuals,
            index.reachable_symbolic_residuals
        );
        assert_eq!(
            decoded.reachable_symbolic_visited,
            index.reachable_symbolic_visited
        );
        assert_eq!(
            decoded.reachable_symbolic_residuals_at(0),
            Some((&[symbolic_a, symbolic_b, symbolic_c, wrong_tag][..], 4))
        );
    }
}
