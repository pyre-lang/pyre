//! RPython `rpython/translator/interactive.py` — convenience wrapper.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use crate::annotator::policy::AnnotatorPolicy;
use crate::annotator::signature::AnnotationSpec;
use crate::config::config::{ConfigValue, OptionValue};
use crate::flowspace::model::HostObject;
use crate::rlib::entrypoint::export_symbol;
use crate::translator::driver::{ProceedGoals, TranslationDriver};
use crate::translator::tool::taskengine::{TaskError, TaskOutput};
use crate::translator::translator::{FlowingFlags, TranslationConfig, TranslationContext};

/// Upstream `DEFAULTS` (interactive.py).
pub const DEFAULTS: [(&str, OptionValue); 3] = [
    ("translation.backend", OptionValue::None),
    ("translation.type_system", OptionValue::None),
    ("translation.verbose", OptionValue::Bool(true)),
];

fn default_overrides() -> HashMap<String, OptionValue> {
    DEFAULTS
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn task_error(message: impl Into<String>) -> TaskError {
    TaskError {
        message: message.into(),
    }
}

/// RPython `class Translation(object)`.
pub struct Translation {
    pub driver: Rc<TranslationDriver>,
    pub config: Rc<crate::config::config::Config>,
    pub entry_point: HostObject,
    pub context: Rc<TranslationContext>,
    pub ann_argtypes: Option<Vec<AnnotationSpec>>,
    pub ann_policy: Option<AnnotatorPolicy>,
}

impl Translation {
    /// RPython `Translation.__init__(entry_point, argtypes=None, **kwds)`.
    pub fn new(
        entry_point: HostObject,
        argtypes: Option<Vec<AnnotationSpec>>,
        kwds: Vec<(String, OptionValue)>,
    ) -> Result<Self, TaskError> {
        Self::with_policy(entry_point, argtypes, None, kwds)
    }

    pub fn with_policy(
        entry_point: HostObject,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
        kwds: Vec<(String, OptionValue)>,
    ) -> Result<Self, TaskError> {
        let driver = TranslationDriver::new(
            None,
            None,
            Vec::new(),
            None,
            None,
            None,
            Some(default_overrides()),
        )?;
        let config = Rc::clone(&driver.config);
        let entry_point = export_symbol(entry_point);
        let context = Rc::new(TranslationContext::with_config_and_flowing_flags(
            Some(TranslationConfig::from_rc_config(&config)?),
            FlowingFlags::default(),
        ));

        let mut translation = Translation {
            driver,
            config,
            entry_point,
            context,
            ann_argtypes: None,
            ann_policy: None,
        };
        translation.update_options(kwds)?;
        translation.ensure_setup(argtypes, policy)?;

        // Upstream builds and stores a prebuilt graph so `view()` works
        // immediately after construction.
        let graph = translation
            .context
            .buildflowgraph(translation.entry_point.clone(), false)
            .map_err(task_error)?;
        translation
            .context
            ._prebuilt_graphs
            .borrow_mut()
            .insert(translation.entry_point.clone(), graph);
        Ok(translation)
    }

    /// RPython `view(self)`.
    pub fn view(&self) -> Result<(), TaskError> {
        Err(task_error(
            "interactive.py:25 Translation.view — graph viewer is not ported",
        ))
    }

    /// RPython `viewcg(self)`.
    pub fn viewcg(&self) -> Result<(), TaskError> {
        Err(task_error(
            "interactive.py:28 Translation.viewcg — callgraph viewer is not ported",
        ))
    }

    /// RPython `ensure_setup(self, argtypes=None, policy=None)`.
    pub fn ensure_setup(
        &mut self,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
    ) -> Result<(), TaskError> {
        self.driver.setup(
            Some(self.entry_point.clone()),
            argtypes.clone(),
            policy.clone(),
            HashMap::new(),
            Some(Rc::clone(&self.context)),
        )?;
        self.ann_argtypes = argtypes;
        self.ann_policy = policy;
        Ok(())
    }

    /// RPython `update_options(self, kwds)`.
    pub fn update_options(&self, mut kwds: Vec<(String, OptionValue)>) -> Result<(), TaskError> {
        for (name, _) in &mut kwds {
            if name == "gc" {
                *name = "translation.gc".to_string();
            }
        }
        self.config.set(kwds)?;
        Ok(())
    }

    /// RPython `ensure_opt(self, name, value=None, fallback=None)`.
    pub fn ensure_opt(
        &self,
        name: &str,
        value: Option<OptionValue>,
        fallback: Option<OptionValue>,
    ) -> Result<OptionValue, TaskError> {
        if let Some(value) = value {
            self.update_options(vec![(name.to_string(), value.clone())])?;
            return Ok(value);
        }
        let path = format!("translation.{name}");
        let val = match self.config.get(&path)? {
            ConfigValue::Value(value) => value,
            ConfigValue::SubConfig(_) => {
                return Err(task_error(format!(
                    "interactive.py ensure_opt: {name:?} names a config subgroup"
                )));
            }
        };
        if !matches!(val, OptionValue::None) {
            return Ok(val);
        }
        if let Some(fallback) = fallback {
            self.update_options(vec![(name.to_string(), fallback.clone())])?;
            return Ok(fallback);
        }
        Err(task_error(format!(
            "the {name:?} option should have been specified at this point"
        )))
    }

    /// RPython `ensure_type_system(self, type_system=None)`.
    pub fn ensure_type_system(&self, type_system: Option<String>) -> Result<String, TaskError> {
        let backend = match self.config.get("translation.backend")? {
            ConfigValue::Value(OptionValue::None) => None,
            ConfigValue::Value(OptionValue::Str(s))
            | ConfigValue::Value(OptionValue::Choice(s)) => Some(s),
            ConfigValue::Value(other) => {
                return Err(task_error(format!(
                    "translation.backend expected string/None, got {other:?}"
                )));
            }
            ConfigValue::SubConfig(_) => return Err(task_error("translation.backend subgroup")),
        };
        let value = if backend.is_some() {
            self.ensure_opt("type_system", None, None)?
        } else {
            self.ensure_opt(
                "type_system",
                type_system.map(OptionValue::Choice),
                Some(OptionValue::Choice("lltype".to_string())),
            )?
        };
        option_string("type_system", value)
    }

    /// RPython `ensure_backend(self, backend=None)`.
    pub fn ensure_backend(&self, backend: Option<String>) -> Result<String, TaskError> {
        let backend = option_string(
            "backend",
            self.ensure_opt("backend", backend.map(OptionValue::Choice), None)?,
        )?;
        self.ensure_type_system(None)?;
        Ok(backend)
    }

    /// RPython `disable(self, to_disable)`.
    pub fn disable(&self, to_disable: Vec<String>) {
        self.driver.disable(to_disable);
    }

    /// RPython `set_backend_extra_options(self, **extra_options)`.
    pub fn set_backend_extra_options(
        &self,
        extra_options: HashMap<String, OptionValue>,
    ) -> Result<(), TaskError> {
        for name in extra_options.keys() {
            if let Some((backend, _option)) = name.split_once('_') {
                self.ensure_backend(Some(backend.to_string()))?;
            }
        }
        self.driver.set_backend_extra_options(extra_options);
        Ok(())
    }

    /// RPython `annotate(self, **kwds)`.
    pub fn annotate(&self, kwds: Vec<(String, OptionValue)>) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds)?;
        self.driver
            .proceed(ProceedGoals::One("annotate".to_string()))
    }

    /// RPython `rtype(self, **kwds)`.
    pub fn rtype(&self, kwds: Vec<(String, OptionValue)>) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds)?;
        let ts = self.ensure_type_system(None)?;
        self.driver
            .proceed(ProceedGoals::One(format!("rtype_{ts}")))
    }

    /// RPython `backendopt(self, **kwds)`.
    pub fn backendopt(&self, kwds: Vec<(String, OptionValue)>) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds)?;
        let ts = self.ensure_type_system(Some("lltype".to_string()))?;
        self.driver
            .proceed(ProceedGoals::One(format!("backendopt_{ts}")))
    }

    /// RPython `source(self, **kwds)`.
    pub fn source(&self, kwds: Vec<(String, OptionValue)>) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds)?;
        let backend = self.ensure_backend(None)?;
        self.driver
            .proceed(ProceedGoals::One(format!("source_{backend}")))
    }

    /// RPython `source_c(self, **kwds)`.
    pub fn source_c(&self, kwds: Vec<(String, OptionValue)>) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds)?;
        self.ensure_backend(Some("c".to_string()))?;
        self.driver
            .proceed(ProceedGoals::One("source_c".to_string()))
    }

    /// RPython `compile(self, **kwds)`.
    pub fn compile(&self, kwds: Vec<(String, OptionValue)>) -> Result<Option<PathBuf>, TaskError> {
        self.update_options(kwds)?;
        let backend = self.ensure_backend(None)?;
        self.driver
            .proceed(ProceedGoals::One(format!("compile_{backend}")))?;
        Ok(self.driver.c_entryp.borrow().clone())
    }

    /// RPython `compile_c(self, **kwds)`.
    pub fn compile_c(
        &self,
        kwds: Vec<(String, OptionValue)>,
    ) -> Result<Option<PathBuf>, TaskError> {
        self.update_options(kwds)?;
        self.ensure_backend(Some("c".to_string()))?;
        self.driver
            .proceed(ProceedGoals::One("compile_c".to_string()))?;
        Ok(self.driver.c_entryp.borrow().clone())
    }
}

fn option_string(name: &str, value: OptionValue) -> Result<String, TaskError> {
    match value {
        OptionValue::Str(s) | OptionValue::Choice(s) => Ok(s),
        other => Err(task_error(format!("{name} expected string, got {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::{ConstValue, Constant, GraphFunc};
    use rustpython_compiler::{Mode, compile as rp_compile};
    use rustpython_compiler_core::bytecode::ConstantData;

    fn sample_entrypoint() -> HostObject {
        let module = rp_compile(
            "def main():\n    return 1\n",
            Mode::Exec,
            "<interactive-test>",
            Default::default(),
        )
        .expect("compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("function code");
        let host = HostCode::from_code(&code);
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        HostObject::new_user_function(GraphFunc::from_host_code(host, globals, vec![]))
    }

    #[test]
    fn defaults_match_interactive_py() {
        let defaults = default_overrides();
        assert!(matches!(
            defaults.get("translation.backend"),
            Some(OptionValue::None)
        ));
        assert!(matches!(
            defaults.get("translation.type_system"),
            Some(OptionValue::None)
        ));
        assert!(matches!(
            defaults.get("translation.verbose"),
            Some(OptionValue::Bool(true))
        ));
    }

    #[test]
    fn new_exports_entrypoint_and_prebuilds_graph() {
        let translation = Translation::new(sample_entrypoint(), None, vec![]).expect("translation");
        assert!(
            translation
                .entry_point
                .user_function()
                .expect("user function")
                .exported_symbol
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        assert!(
            translation
                .context
                ._prebuilt_graphs
                .borrow()
                .contains_key(&translation.entry_point)
        );
    }

    #[test]
    fn ensure_type_system_defaults_to_lltype_without_backend() {
        let translation = Translation::new(sample_entrypoint(), None, vec![]).expect("translation");
        assert_eq!(translation.ensure_type_system(None).unwrap(), "lltype");
    }
}
