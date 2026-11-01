//! Port of `rpython/translator/tool/cbuild.py`.
//!
//! `ExternalCompilationInfo` is the owner of compilation/link metadata
//! consumed by `translator.platform` (`Platform.compile` /
//! `Platform.execute`) and `tool.gcc_cache`.

use std::fmt;
use std::path::PathBuf;

use crate::translator::tool::taskengine::TaskError;

/// Port of `cbuild.ExternalCompilationInfo`.
///
/// Field order matches `_ATTRIBUTES` then `_EXTRA_ATTRIBUTES`
/// (`pre_include_bits` … `testonly_libraries`, then `use_cpp_linker`
/// and `platform`). Rust stores the tuple attributes as owned vectors
/// and `platform` as an optional opaque key (`ExternalCompilationInfo.__init__`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExternalCompilationInfo {
    pub pre_include_bits: Vec<String>,
    pub includes: Vec<String>,
    pub include_dirs: Vec<PathBuf>,
    pub post_include_bits: Vec<String>,
    pub libraries: Vec<String>,
    pub library_dirs: Vec<PathBuf>,
    pub separate_module_sources: Vec<String>,
    pub separate_module_files: Vec<PathBuf>,
    pub compile_extra: Vec<String>,
    pub link_extra: Vec<String>,
    pub frameworks: Vec<String>,
    pub link_files: Vec<PathBuf>,
    pub testonly_libraries: Vec<String>,
    pub use_cpp_linker: bool,
    pub platform: Option<String>,
}

impl ExternalCompilationInfo {
    /// `ExternalCompilationInfo._value`: `_ATTRIBUTES + _EXTRA_ATTRIBUTES`
    /// in declaration order.
    pub fn _value(
        &self,
    ) -> (
        &[String],
        &[String],
        &[PathBuf],
        &[String],
        &[String],
        &[PathBuf],
        &[String],
        &[PathBuf],
        &[String],
        &[String],
        &[String],
        &[PathBuf],
        &[String],
        bool,
        Option<&str>,
    ) {
        (
            &self.pre_include_bits,
            &self.includes,
            &self.include_dirs,
            &self.post_include_bits,
            &self.libraries,
            &self.library_dirs,
            &self.separate_module_sources,
            &self.separate_module_files,
            &self.compile_extra,
            &self.link_extra,
            &self.frameworks,
            &self.link_files,
            &self.testonly_libraries,
            self.use_cpp_linker,
            self.platform.as_deref(),
        )
    }

    /// `ExternalCompilationInfo.merge(*others)`.
    ///
    /// Attributes are de-duplicated in first-occurrence order except
    /// `compile_extra` and `link_extra` (`_DUPLICATES_OK`).
    pub fn merge(&self, others: &[&ExternalCompilationInfo]) -> Result<Self, TaskError> {
        let mut unique_others: Vec<&ExternalCompilationInfo> = Vec::new();
        for other in others {
            if !unique_others.contains(other) {
                unique_others.push(*other);
            }
        }

        for other in &unique_others {
            if other.platform != self.platform {
                return Err(TaskError {
                    message: format!(
                        "ExternalCompilationInfo.merge: mixing platforms {:?} and {:?}",
                        other.platform, self.platform
                    ),
                });
            }
        }

        let mut result = ExternalCompilationInfo {
            platform: self.platform.clone(),
            ..ExternalCompilationInfo::default()
        };
        for one in std::iter::once(self).chain(unique_others) {
            append_unique(&mut result.pre_include_bits, &one.pre_include_bits);
            append_unique(&mut result.includes, &one.includes);
            append_unique(&mut result.include_dirs, &one.include_dirs);
            append_unique(&mut result.post_include_bits, &one.post_include_bits);
            append_unique(&mut result.libraries, &one.libraries);
            append_unique(&mut result.library_dirs, &one.library_dirs);
            append_unique(
                &mut result.separate_module_sources,
                &one.separate_module_sources,
            );
            append_unique(
                &mut result.separate_module_files,
                &one.separate_module_files,
            );
            result
                .compile_extra
                .extend(one.compile_extra.iter().cloned());
            result.link_extra.extend(one.link_extra.iter().cloned());
            append_unique(&mut result.frameworks, &one.frameworks);
            append_unique(&mut result.link_files, &one.link_files);
            append_unique(&mut result.testonly_libraries, &one.testonly_libraries);
            result.use_cpp_linker |= one.use_cpp_linker;
        }
        Ok(result)
    }
}

/// `ExternalCompilationInfo.__repr__`.
impl fmt::Display for ExternalCompilationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let attrs = [
            format!("pre_include_bits={:?}", self.pre_include_bits),
            format!("includes={:?}", self.includes),
            format!("include_dirs={:?}", self.include_dirs),
            format!("post_include_bits={:?}", self.post_include_bits),
            format!("libraries={:?}", self.libraries),
            format!("library_dirs={:?}", self.library_dirs),
            format!("separate_module_sources={:?}", self.separate_module_sources),
            format!("separate_module_files={:?}", self.separate_module_files),
            format!("compile_extra={:?}", self.compile_extra),
            format!("link_extra={:?}", self.link_extra),
            format!("frameworks={:?}", self.frameworks),
            format!("link_files={:?}", self.link_files),
            format!("testonly_libraries={:?}", self.testonly_libraries),
            format!("use_cpp_linker={:?}", self.use_cpp_linker),
            format!("platform={:?}", self.platform),
        ];
        write!(f, "<ExternalCompilationInfo ({})>", attrs.join(", "))
    }
}

fn append_unique<T: Clone + Eq>(dst: &mut Vec<T>, src: &[T]) {
    for item in src {
        if !dst.contains(item) {
            dst.push(item.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_dedupes_except_compile_and_link_extra() {
        let left = ExternalCompilationInfo {
            includes: vec!["a.h".to_string(), "a.h".to_string()],
            include_dirs: vec![PathBuf::from("inc"), PathBuf::from("inc")],
            compile_extra: vec!["-O2".to_string()],
            link_extra: vec!["-lm".to_string()],
            use_cpp_linker: false,
            platform: Some("host".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let right = ExternalCompilationInfo {
            includes: vec!["a.h".to_string(), "b.h".to_string()],
            include_dirs: vec![PathBuf::from("inc"), PathBuf::from("other")],
            compile_extra: vec!["-O2".to_string()],
            link_extra: vec!["-lm".to_string()],
            use_cpp_linker: true,
            platform: Some("host".to_string()),
            ..ExternalCompilationInfo::default()
        };

        let merged = left.merge(&[&right]).expect("matching platform");

        assert_eq!(merged.includes, vec!["a.h", "b.h"]);
        assert_eq!(
            merged.include_dirs,
            vec![PathBuf::from("inc"), PathBuf::from("other")]
        );
        assert_eq!(merged.compile_extra, vec!["-O2", "-O2"]);
        assert_eq!(merged.link_extra, vec!["-lm", "-lm"]);
        assert!(merged.use_cpp_linker);
    }

    #[test]
    fn merge_rejects_mixed_platforms() {
        let left = ExternalCompilationInfo {
            platform: Some("host".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let right = ExternalCompilationInfo {
            platform: Some("other".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let err = left.merge(&[&right]).expect_err("mixed platforms");
        assert!(err.message.contains("mixing platforms"), "{}", err.message);
    }

    #[test]
    fn repr_and_value_cover_every_attribute() {
        let eci = ExternalCompilationInfo {
            includes: vec!["stdio.h".to_string()],
            use_cpp_linker: true,
            platform: Some("host".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let rendered = eci.to_string();
        assert!(rendered.starts_with("<ExternalCompilationInfo ("));
        assert!(rendered.contains("includes="));
        assert!(rendered.contains("use_cpp_linker=true"));
        assert!(rendered.contains("platform=Some(\"host\")"));
        let value = eci._value();
        assert_eq!(value.1, &["stdio.h".to_string()][..]);
        assert!(value.13);
        assert_eq!(value.14, Some("host"));
    }
}
