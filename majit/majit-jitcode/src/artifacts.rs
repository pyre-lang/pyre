//! The build-process transport the JIT runtime decodes: the JitCode archive
//! index and each driver's compiled identity. `majit-translate` owns the
//! encoding side (`EmbeddedArtifacts`) and re-exports these.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::jitcode::JitCode;
use crate::parse::CallPath;

/// Dense allocation order, with a separate graph key for colliding leaf names.
///
/// Body bytes are sliced by `offsets`. Symbolic residual funcptr constants
/// are a second table: entry `i` is
/// `reachable_symbolic_residuals[reachable_symbolic_offsets[i]..reachable_symbolic_offsets[i + 1]]`,
/// sorted and distinct, including constants reached through `inline_call`
/// callees. `reachable_symbolic_visited[i]` is how many jitcodes that walk
/// entered, including `i`.
#[derive(Serialize, Deserialize)]
pub struct JitCodeIndex {
    pub names: Vec<String>,
    /// Graph identity: `call.py CallControl.get_jitcode` keys by graph,
    /// `description.py FunctionDesc.cachedgraph` by specialization.
    /// `None` is a jitcode minted without a graph key.
    pub paths: Vec<Option<CallPath>>,
    pub offsets: Vec<u32>,
    pub reachable_symbolic_residuals: Vec<i64>,
    pub reachable_symbolic_offsets: Vec<u32>,
    pub reachable_symbolic_visited: Vec<u32>,
}

/// Per-jitcode symbolic residual closure, flat so the index stays one bincode
/// record. `values[offsets[i] as usize..offsets[i + 1] as usize]` is sorted
/// and distinct. `visited[i]` counts jitcodes entered from `i`, including `i`.
#[derive(Clone, Debug)]
pub struct ReachableSymbolicResidualTable {
    pub values: Vec<i64>,
    pub offsets: Vec<u32>,
    pub visited: Vec<u32>,
}

impl JitCodeIndex {
    pub fn encode(
        jitcodes: &[Arc<JitCode>],
        paths: Vec<Option<CallPath>>,
        reachable: ReachableSymbolicResidualTable,
    ) -> bincode::Result<(Self, Vec<u8>)> {
        assert_eq!(paths.len(), jitcodes.len());
        let mut bodies = Vec::new();
        let mut index = Self {
            names: Vec::new(),
            paths,
            offsets: vec![0],
            reachable_symbolic_residuals: reachable.values,
            reachable_symbolic_offsets: reachable.offsets,
            reachable_symbolic_visited: reachable.visited,
        };
        for jitcode in jitcodes {
            index.names.push(jitcode.name.clone());
            bincode::serialize_into(&mut bodies, jitcode)?;
            index
                .offsets
                .push(u32::try_from(bodies.len()).expect("JitCode archive exceeds u32 offsets"));
        }
        index.validate(&bodies)?;
        Ok((index, bodies))
    }

    pub fn decode(bytes: &[u8], bodies: &[u8]) -> bincode::Result<Self> {
        let index: Self = bincode::deserialize(bytes)?;
        index.validate(bodies)?;
        Ok(index)
    }

    pub fn validate(&self, bodies: &[u8]) -> bincode::Result<()> {
        let residual_end = self.reachable_symbolic_offsets.last().map(|&n| n as usize);
        if self.paths.len() != self.names.len()
            || self.offsets.len() != self.names.len() + 1
            || self.offsets.first().copied() != Some(0)
            || self.offsets.last().map(|&n| n as usize) != Some(bodies.len())
            || !self.offsets.windows(2).all(|p| p[0] <= p[1])
            || self.reachable_symbolic_offsets.len() != self.offsets.len()
            || self.reachable_symbolic_visited.len() + 1 != self.reachable_symbolic_offsets.len()
            || self.reachable_symbolic_offsets.first().copied() != Some(0)
            || residual_end != Some(self.reachable_symbolic_residuals.len())
            || !self
                .reachable_symbolic_offsets
                .windows(2)
                .all(|p| p[0] <= p[1])
        {
            return Err(Box::new(bincode::ErrorKind::Custom(
                "invalid JitCode archive index".into(),
            )));
        }
        Ok(())
    }

    /// Symbolic residual constants and visited-jitcode count for `index`.
    ///
    /// `None` when `index` is outside the table or an offset points past
    /// `reachable_symbolic_residuals`. A table that passed [`Self::decode`]
    /// answers `Some` for every jitcode index.
    pub fn reachable_symbolic_residuals_at(&self, index: usize) -> Option<(&[i64], u32)> {
        let start = *self.reachable_symbolic_offsets.get(index)? as usize;
        let end = *self.reachable_symbolic_offsets.get(index + 1)? as usize;
        let values = self.reachable_symbolic_residuals.get(start..end)?;
        let visited = *self.reachable_symbolic_visited.get(index)?;
        Some((values, visited))
    }

    pub fn load(&self, bodies: &[u8], index: usize) -> bincode::Result<Arc<JitCode>> {
        let start = self.offsets.get(index).copied();
        let end = index
            .checked_add(1)
            .and_then(|i| self.offsets.get(i))
            .copied();
        let bytes = start
            .zip(end)
            .and_then(|(s, e)| bodies.get(s as usize..e as usize))
            .ok_or_else(|| {
                Box::new(bincode::ErrorKind::Custom(
                    "JitCode index out of bounds".into(),
                ))
            })?;
        bincode::deserialize(bytes)
    }
}

/// Compiled identity of one configured JIT driver.
///
/// RPython equivalent: `JitDriverStaticData.portal_graph` together with
/// `JitDriverStaticData.mainjitcode.index` after
/// `CallControl.grab_initial_jitcodes()` and codewriter draining.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledJitDriver {
    pub portal: CallPath,
    /// RPython: `jitdriver_sd.portal_runner_ptr`, preserved as source-level
    /// identity alongside the run-time `portal_runner_adr`.
    #[serde(default)]
    pub portal_runner: Option<CallPath>,
    pub main_jitcode_index: usize,
    /// RPython: `jitdriver.greens` — the green names, in declaration order.
    #[serde(default)]
    pub greens: Vec<String>,
    /// RPython: `jitdriver.reds`. Empty for an auto-red driver, whose reds were
    /// never declared; `red_args_types` describes them in that case.
    #[serde(default)]
    pub reds: Vec<String>,
    /// RPython: `jd._green_args_spec` (`warmspot.py
    /// make_args_specification`), parallel to `greens`.
    ///
    /// Upstream stores each marker operand's full `concretetype`, including
    /// `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`.  Pyre's graph-side
    /// `ConcreteType` has already projected every GC pointer to `GcRef` before
    /// `Transformer::marker_operand_kinds` reads it, so this artifact can only
    /// preserve the IR kind today.  Widening this field to `GreenType` would
    /// not recover information its producer cannot supply.
    #[serde(default)]
    pub green_args_spec: Vec<majit_ir::Type>,
    /// RPython: `jd.red_args_types` (`warmspot.py:664`).
    ///
    /// Read off the merge-point operands during codewriting, so a consumer
    /// building the run-time driver takes the kinds the graph actually has
    /// rather than re-typing the build's declaration. Without this the two
    /// accounts are independent and nothing makes them agree.
    #[serde(default)]
    pub red_args_types: Vec<majit_ir::Type>,
    /// RPython: `jitdriver.virtualizables` — the red names declared
    /// virtualizable. Empty is the common non-pyre case.
    #[serde(default)]
    pub virtualizables: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::GraphSpecialization;

    fn index_bytes(
        names: Vec<String>,
        paths: Vec<Option<CallPath>>,
        offsets: Vec<u32>,
        residuals: Vec<i64>,
        residual_offsets: Vec<u32>,
        visited: Vec<u32>,
    ) -> Vec<u8> {
        bincode::serialize(&(names, paths, offsets, residuals, residual_offsets, visited)).unwrap()
    }

    fn empty_reachable(len: usize) -> ReachableSymbolicResidualTable {
        ReachableSymbolicResidualTable {
            values: Vec::new(),
            offsets: vec![0; len + 1],
            visited: vec![0; len],
        }
    }

    #[test]
    fn indexed_format_preserves_existing_tuple_wire_format() {
        let paths = vec![
            Some(CallPath::from_segments(["a", "same"])),
            Some(CallPath::from_segments(["b", "same"])),
        ];
        let old = index_bytes(
            vec!["same".to_string(); 2],
            paths.clone(),
            vec![0_u32, 1, 2],
            vec![4, 9],
            vec![0, 2, 2],
            vec![3, 1],
        );
        let index = JitCodeIndex::decode(&old, &[0, 0]).unwrap();
        assert_eq!(index.paths, paths);
        assert_eq!(index.reachable_symbolic_residuals, [4, 9]);
        assert_eq!(index.reachable_symbolic_visited, [3, 1]);
        assert_eq!(
            index.reachable_symbolic_residuals_at(0),
            Some((&[4, 9][..], 3))
        );
        assert_eq!(index.reachable_symbolic_residuals_at(1), Some((&[][..], 1)));
        assert_eq!(bincode::serialize(&index).unwrap(), old);
    }

    #[test]
    fn encode_keeps_regular_and_access_direct_paths_distinct() {
        let regular = CallPath::from_segments(["owner", "f"]);
        let access_direct = CallPath {
            segments: regular.segments.clone(),
            specialization: GraphSpecialization::AccessDirect,
        };
        // Same segments, different FunctionDesc.cachedgraph keys. A Regular
        // graph whose last segment is already the alt_name spelling shares
        // display_name with the AccessDirect copy; the frozen column stores
        // the CallPath pair so they stay distinct.
        assert_eq!(
            access_direct.display_name(),
            CallPath::from_segments(["owner", "f__AccessDirect_None"]).display_name()
        );
        assert_ne!(regular, access_direct);

        let a = Arc::new(JitCode::new("f"));
        a.set_index(0);
        let b = Arc::new(JitCode::new("f__AccessDirect_None"));
        b.set_index(1);
        let (index, bodies) = JitCodeIndex::encode(
            &[a, b],
            vec![Some(regular.clone()), Some(access_direct.clone())],
            empty_reachable(2),
        )
        .unwrap();
        let decoded = JitCodeIndex::decode(&bincode::serialize(&index).unwrap(), &bodies).unwrap();
        assert_eq!(decoded.paths, [Some(regular), Some(access_direct)]);
        assert_ne!(decoded.paths[0], decoded.paths[1]);
    }

    #[test]
    fn stale_index_without_residual_table_is_rejected() {
        let old = bincode::serialize(&(
            vec!["same".to_string(); 2],
            vec![
                Some(CallPath::from_segments(["a", "same"])),
                Some(CallPath::from_segments(["b", "same"])),
            ],
            vec![0_u32, 1, 2],
        ))
        .unwrap();
        assert!(JitCodeIndex::decode(&old, &[0, 0]).is_err());
    }

    #[test]
    fn rejects_corrupt_offsets_and_lengths() {
        for offsets in [vec![], vec![1_u32, 2], vec![0, 3], vec![0, 2, 1]] {
            let bytes = index_bytes(
                vec!["one".to_string()],
                vec![None],
                offsets,
                Vec::new(),
                vec![0, 0],
                vec![0],
            );
            assert!(JitCodeIndex::decode(&bytes, &[0, 0]).is_err());
        }
        // Body span is empty and valid. The residual offset span claims one
        // constant while the flat table holds two.
        let mismatched_len = index_bytes(
            vec!["one".to_string()],
            vec![None],
            vec![0, 0],
            vec![7, 9],
            vec![0, 1],
            vec![1],
        );
        assert!(JitCodeIndex::decode(&mismatched_len, &[]).is_err());
        // `last == values.len()`, but the span is not monotonic.
        let non_monotonic = index_bytes(
            vec!["a".to_string(), "b".to_string()],
            vec![None, None],
            vec![0, 0, 0],
            vec![9],
            vec![0, 2, 1],
            vec![1, 1],
        );
        assert!(JitCodeIndex::decode(&non_monotonic, &[]).is_err());
        let bad_origin = index_bytes(
            vec!["one".to_string()],
            vec![None],
            vec![0, 0],
            vec![5],
            vec![1, 1],
            vec![1],
        );
        assert!(JitCodeIndex::decode(&bad_origin, &[]).is_err());
        let bad_visited = index_bytes(
            vec!["one".to_string()],
            vec![None],
            vec![0, 0],
            Vec::new(),
            vec![0, 0],
            Vec::new(),
        );
        assert!(JitCodeIndex::decode(&bad_visited, &[]).is_err());
    }
}
