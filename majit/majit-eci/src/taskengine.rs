//! Error type of `translator/tool/taskengine.py`.
//! The rest of taskengine stays in majit-translate.

/// Error surface of a task callable. Upstream raises Python
/// exceptions and `_execute` re-raises after the `_error(goal)` hook
/// fires. The Rust port replaces the raise-and-catch with `Result` +
/// an error struct carrying the upstream-observable message text.
#[derive(Debug, Clone)]
pub struct TaskError {
    pub message: String,
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TaskError {}

impl From<majit_config::config::ConfigError> for TaskError {
    fn from(value: majit_config::config::ConfigError) -> Self {
        TaskError {
            message: value.to_string(),
        }
    }
}
