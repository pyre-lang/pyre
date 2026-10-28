//! Call-path and inherent-method metadata carriers shared across the
//! front-end and codewriter.

use serde::{Deserialize, Serialize};

/// `description.py FunctionDesc.cachedgraph` alt_name postfix for
/// `specialize.py default_specialize`'s `(AccessDirect, None)` graph
/// (`nameof(AccessDirect)` + `_` + `nameof(None)`). Display-only: the
/// cache key is [`GraphSpecialization`], not this suffix.
pub const ACCESS_DIRECT_ALT_SUFFIX: &str = "__AccessDirect_None";

/// Which `FunctionDesc.cachedgraph` key a graph was built under
/// (`specialize.py default_specialize`): `AccessDirect` is the
/// `(AccessDirect, None)` key; `Regular` is every other graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphSpecialization {
    #[default]
    Regular,
    AccessDirect,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallPath {
    pub segments: Vec<String>,
    #[serde(default)]
    pub specialization: GraphSpecialization,
}

impl CallPath {
    pub fn from_segments<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
            specialization: GraphSpecialization::Regular,
        }
    }

    /// Same specialization, new segments (crate-strip / alias spellings).
    pub fn with_segments<I, S>(&self, segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
            specialization: self.specialization,
        }
    }

    /// Build the canonical CallPath for an inherent / trait-impl method.
    ///
    /// `impl_type_joined` may be a single segment (`"Foo"`) or a
    /// `::`-joined type path (`"a::Foo"`, `"mod::Outer::Inner"`). The
    /// impl_type is split into its individual segments and concatenated
    /// with the method name so that the resulting CallPath is uniform
    /// with free-fn paths (`["a", "b", "f"]`) — both the
    /// type-qualified prefix and the method name live at the same
    /// segment granularity. Previously impl methods were stored as
    /// 2-segment `[impl_type_joined, method]`, which diverged from
    /// free-fn shape and forced macro-side heuristics; this form
    /// restores uniformity (RPython parity: `getfunctionptr(graph)` is
    /// string-free and does not distinguish the two shapes
    /// `rpython/jit/codewriter/call.py get_jitcode_calldescr`).
    ///
    // Structural adaptation: Rust `::` ↔ PyPy `.` path separator.
    // `impl_type_joined` may arrive in either spelling — Rust extraction
    // emits `module::Type`, while `ClassDef.name` mirrors classdesc.py
    // `cls.__module__ + '.' + cls.__name__` (a `.`-joined `module.Class`).
    // Split on both so the segment granularity is independent of which
    // caller minted the string: callers in lib.rs / call.rs /
    // codewriter.rs do not all route through a `.`→`::` normalization
    // boundary, so accepting both keeps the invariant statically true.
    pub fn for_impl_method(impl_type_joined: &str, method: &str) -> Self {
        let mut segments: Vec<String> = impl_type_joined
            .split("::")
            .flat_map(|s| s.split('.'))
            .filter(|seg| !seg.is_empty())
            .map(|seg| seg.to_string())
            .collect();
        segments.push(method.to_string());
        Self {
            segments,
            specialization: GraphSpecialization::Regular,
        }
    }

    /// Build the direct-call identity for one concrete trait-impl method.
    ///
    /// RPython `CallControl.graphs_from` obtains the callee graph from the
    /// function pointer, so two `impl From<_> for Dynamic` methods never
    /// collide even though their owner and method names are equal. Charon
    /// gives each concrete impl a stable id within the LLBC; keep that id as
    /// its own synthetic path segment so name-keyed registration preserves
    /// the same distinction.
    pub fn for_trait_impl_method(impl_type_joined: &str, impl_id: u64, method: &str) -> Self {
        let mut path = Self::for_impl_method(impl_type_joined, method);
        let leaf = path.segments.pop().expect("method segment");
        path.segments.push(format!("<Impl#{impl_id}>"));
        path.segments.push(leaf);
        path
    }

    pub fn last_segment(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    /// Segment join only. Identity-free: does not encode
    /// [`GraphSpecialization`]. Two graphs of one funcobj that differ
    /// only by `specialize.py default_specialize` cache key share this
    /// string.
    pub fn canonical_key(&self) -> String {
        self.segments.join("::")
    }

    /// Display name: `canonical_key()` for Regular, and
    /// `canonical_key()` plus [`ACCESS_DIRECT_ALT_SUFFIX`] for
    /// AccessDirect (`description.py FunctionDesc.cachedgraph` alt_name).
    pub fn display_name(&self) -> String {
        match self.specialization {
            GraphSpecialization::Regular => self.canonical_key(),
            GraphSpecialization::AccessDirect => {
                format!("{}{ACCESS_DIRECT_ALT_SUFFIX}", self.canonical_key())
            }
        }
    }

    /// For a path built by `for_impl_method`, extract the impl type
    /// portion (all segments except the trailing method name).
    pub fn impl_type_prefix(&self) -> String {
        if self.segments.len() >= 2 {
            self.segments[..self.segments.len() - 1].join("::")
        } else {
            self.segments.join("::")
        }
    }
}

/// The `canonical_key` spelling, written straight into a formatter.
///
/// Lets a diagnostic name a path through `format_args!` without the
/// `canonical_key()` `String` allocation, which matters where the
/// diagnostic is disabled: `format_args!` evaluates its arguments
/// eagerly, so an allocating accessor there would cost on the path that
/// prints nothing. AccessDirect appends the alt_name suffix
/// (display-only).
impl std::fmt::Display for CallPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                f.write_str("::")?;
            }
            f.write_str(segment)?;
        }
        if self.specialization == GraphSpecialization::AccessDirect {
            f.write_str(ACCESS_DIRECT_ALT_SUFFIX)?;
        }
        Ok(())
    }
}

/// Display-only, mirroring `description.py FunctionDesc.cachedgraph` alt_name.
/// Prints the pre-field shape: `CallPath { segments: [...] }`, with the
/// last segment shown as its alt_name when AccessDirect.
impl std::fmt::Debug for CallPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallPath")
            .field("segments", &CallPathDebugSegments(self))
            .finish()
    }
}

struct CallPathDebugSegments<'a>(&'a CallPath);

impl std::fmt::Debug for CallPathDebugSegments<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        let last = self.0.segments.len().checked_sub(1);
        for (i, segment) in self.0.segments.iter().enumerate() {
            if self.0.specialization == GraphSpecialization::AccessDirect && Some(i) == last {
                list.entry(&format!("{segment}{ACCESS_DIRECT_ALT_SUFFIX}"));
            } else {
                list.entry(&segment.as_str());
            }
        }
        list.finish()
    }
}

/// Strip the module prefix and return the trailing identifier.
///
/// Accepts both spellings: a `::`-joined Rust path and the `.`-joined
/// `ClassDef.name` form (classdesc.py `cls.__module__ + '.' +
/// cls.__name__`). A plain `rsplit('.')` misses Rust-rooted values and a
/// plain `rsplit("::")` misses Python-rooted values, so strip the longer
/// `::` prefix first and then any residual `.` prefix — the final
/// identifier is recovered regardless of which separator the caller used.
pub fn canonical_leaf(name: &str) -> &str {
    let after_colon = name.rsplit("::").next().unwrap_or(name);
    after_colon.rsplit('.').next().unwrap_or(after_colon)
}

#[cfg(test)]
mod tests {
    use super::{ACCESS_DIRECT_ALT_SUFFIX, CallPath, GraphSpecialization};

    #[test]
    fn concrete_trait_impl_paths_keep_same_named_methods_distinct() {
        let bool_from = CallPath::for_trait_impl_method("types::dynamic::Dynamic", 1105, "from");
        let fnptr_from = CallPath::for_trait_impl_method("types::dynamic::Dynamic", 835, "from");

        assert_eq!(
            bool_from.segments,
            ["types", "dynamic", "Dynamic", "<Impl#1105>", "from"]
        );
        assert_ne!(bool_from, fnptr_from);
    }

    #[test]
    fn display_name_tags_access_direct_without_changing_segments() {
        let regular = CallPath::from_segments(["owner", "pop"]);
        assert_eq!(regular.canonical_key(), "owner::pop");
        assert_eq!(regular.display_name(), "owner::pop");
        assert_eq!(regular.specialization, GraphSpecialization::Regular);

        let ad = CallPath {
            segments: regular.segments.clone(),
            specialization: GraphSpecialization::AccessDirect,
        };
        assert_eq!(ad.segments, regular.segments);
        assert_eq!(ad.canonical_key(), "owner::pop");
        assert_eq!(
            ad.display_name(),
            format!("owner::pop{ACCESS_DIRECT_ALT_SUFFIX}")
        );
        assert_eq!(ad.to_string(), ad.display_name());
        assert_eq!(
            format!("{regular:?}"),
            r#"CallPath { segments: ["owner", "pop"] }"#
        );
        assert_eq!(
            format!("{ad:?}"),
            r#"CallPath { segments: ["owner", "pop__AccessDirect_None"] }"#
        );
        assert_ne!(regular, ad);
    }
}
