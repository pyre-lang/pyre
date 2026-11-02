//! Base annotation policy for specialization.
//!
//! RPython upstream: `rpython/annotator/policy.py` (100 LOC).
//!
//! Ports `AnnotatorPolicy` and the `get_specializer(directive)`
//! dispatch loop that maps `specialize:arg(N)` / `specialize:memo` /
//! `specialize:argtype(N)` / … directive strings to the per-policy
//! specializer methods.
//!
//! ## Dependency-blocked paths
//!
//! * `specialize.memo` (specialize.py) evaluates `value = func(*args)`
//!   at annotation time through the function's
//!   [`crate::flowspace::model::HostCall`] hook. The memo tables and
//!   dispatch graphs are implemented in [`super::specialize::memo`].
//!   A function without a registered hook is still rejected; importing
//!   its graph alone does not make it callable at annotation time.
//! * `specialize__ll` / `specialize__ll_and_arg` — forward to
//!   [`LowLevelAnnotatorPolicy`] in
//!   [`crate::translator::rtyper::annlowlevel`], wired through the
//!   `Specializer::Ll` / `Specializer::LlAndArg` arms in
//!   [`crate::annotator::description::FunctionDesc::pycall`].
//! * `no_more_blocks_to_annotate(annotator)` (policy.py) —
//!   fully structurally ported. Drains `bk.pending_specializations`
//!   and iterates `added_blocks` / `annotated`, then rewrites call
//!   sites whose callee annotation carries `needs_sandboxing`
//!   (`SomeBuiltin::needs_sandboxing: Option<SandboxingPayload>`).
//!
//!   **Deferred**: upstream's `make_sandbox_trampoline` (rsandbox.py:
//!   see `rpython/translator/sandbox/rsandbox.py`) produces a real
//!   Python function carrying `_signature_ = ([SomeTuple(items=params_s
//!   )], s_result)` and a `functionptr` lltype stub. The rtyper
//!   primitives this depends on (`lltype.FuncType`, `functionptr`,
//!   `ll_ptrtype`) are now ported in
//!   [`crate::translator::rtyper::lltypesystem::lltype`]; what
//!   remains is the `rsandbox` body itself, which has no Rust
//!   counterpart yet. The Rust port substitutes a
//!   `HostObject::new_user_function` stand-in so the
//!   `emulated_pbc_calls` bookkeeping + instr rewrite are upstream-
//!   faithful even though the synthesized trampoline itself is an
//!   empty placeholder. When `rsandbox` lands, the
//!   `trampoline_name`/`GraphFunc` construction inside
//!   `no_more_blocks_to_annotate` is the single point to wire into
//!   real `make_sandbox_trampoline(name, params_s, s_result)`.

use std::fmt::Debug;
use std::rc::Rc;

use crate::annotator::bookkeeper::Bookkeeper;

/// Enumerates the specializer entry points referenced by upstream
/// `AnnotatorPolicy` class-level attributes (policy.py).
///
/// Each variant names one function in
/// `rpython/annotator/specialize.py` (or
/// `rpython/rtyper/annlowlevel.py` for the `Ll` / `LlAndArg`
/// variants). Bodies are executed by
/// [`super::description::FunctionDesc::specialize`]. `Memo` dispatches
/// to [`super::specialize::memo`]. `Ll` / `LlAndArg` dispatch to
/// `LowLevelAnnotatorPolicy`; `ArgLltype` / `GenConst` /
/// `LowLevelDefault` / `MixLevelDefault` dispatch to
/// `annlowlevel.py`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Specializer {
    /// RPython `default_specialize` (specialize.py) — the no-op
    /// fallback used when `directive` is `None`.
    Default,
    /// RPython `LowLevelAnnotatorPolicy.default_specialize`
    /// (annlowlevel.py) reached when the active policy is
    /// `LowLevelAnnotatorPolicy` and the callee has no explicit
    /// `_annspecialcase_` tag.
    LowLevelDefault,
    /// RPython `MixLevelAnnotatorPolicy.default_specialize`
    /// (annlowlevel.py) reached when the active policy is
    /// `MixLevelAnnotatorPolicy` and the callee has no explicit
    /// `_annspecialcase_` tag.
    MixLevelDefault,
    /// `specialize:memo` → `specialize.memo`.
    Memo,
    /// `specialize:arg(N)` → `specialize.specialize_argvalue`. The
    /// `parms` tuple carries the argument indices.
    Arg { parms: Vec<usize> },
    /// `specialize:arg_or_var(N)` → `specialize.specialize_arg_or_var`.
    ArgOrVar { parms: Vec<usize> },
    /// `specialize:argtype(N)` → `specialize.specialize_argtype`.
    Argtype { parms: Vec<usize> },
    /// `specialize:arglistitemtype(N)` →
    /// `specialize.specialize_arglistitemtype`.
    Arglistitemtype { parms: Vec<usize> },
    /// `specialize:call_location` →
    /// `specialize.specialize_call_location`.
    CallLocation,
    /// `specialize:ll` → `LowLevelAnnotatorPolicy.default_specialize`.
    Ll { parms: Vec<String> },
    /// `specialize:ll_and_arg` →
    /// `LowLevelAnnotatorPolicy.specialize__ll_and_arg`.
    LlAndArg { parms: Vec<String> },
    /// `specialize:arglltype` →
    /// `MixLevelAnnotatorPolicy.specialize__arglltype`.
    ArgLltype { parms: Vec<String> },
    /// `specialize:genconst` →
    /// `MixLevelAnnotatorPolicy.specialize__genconst`.
    GenConst { parms: Vec<String> },
}

pub trait PolicyOps: Debug {
    fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError>;
    fn no_more_blocks_to_annotate(
        &self,
        ann: &super::annrpython::RPythonAnnotator,
    ) -> Result<(), super::model::AnnotatorError>;
}

#[derive(Clone)]
pub struct PolicyHandle(Rc<dyn PolicyOps>);

impl PolicyHandle {
    pub fn new<P>(policy: P) -> Self
    where
        P: PolicyOps + 'static,
    {
        PolicyHandle(Rc::new(policy))
    }

    pub fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        self.0.get_specializer(directive)
    }

    pub fn no_more_blocks_to_annotate(
        &self,
        ann: &super::annrpython::RPythonAnnotator,
    ) -> Result<(), super::model::AnnotatorError> {
        self.0.no_more_blocks_to_annotate(ann)
    }
}

impl std::fmt::Debug for PolicyHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

/// RPython `class AnnotatorPolicy` (policy.py).
///
/// The upstream class is subclass-hook-heavy: `event`,
/// `get_specializer`, and `no_more_blocks_to_annotate` all accept a
/// `pol` (policy) instance as their first argument, letting subclasses
/// override individual pieces. The Rust port currently provides only
/// the default implementation; subclass behaviour lands as traits when
/// real policy overrides (e.g. `rpython/rlib/jit.py:JitPolicy`) are
/// ported.
#[derive(Clone, Debug, Default)]
pub struct AnnotatorPolicy;

impl AnnotatorPolicy {
    pub fn new() -> Self {
        AnnotatorPolicy
    }

    /// RPython `AnnotatorPolicy.event(pol, bookkeeper, what, *args)`
    /// (policy.py:17-18). Base implementation is a no-op; subclasses
    /// (e.g. `translator.test.snippet.StrictAnnotatorPolicy`) use
    /// it as a hook.
    pub fn event(&self, _bookkeeper: &Bookkeeper, _what: &str, _args: &[&str]) {}

    /// RPython `AnnotatorPolicy.get_specializer(pol, directive)`
    /// (policy.py:20-49).
    ///
    /// Parses a `specialize:...` directive string produced by
    /// `@specialize(...)` decorators (see
    /// `rpython/rlib/objectmodel.py:specialize`). Upstream splits the
    /// directive at `(`, `eval`s the parameter tuple, then dispatches
    /// to `pol.specialize__<name>`. The Rust port mirrors the parse,
    /// splitting the parameter string on commas instead of using
    /// `eval` — specialize directives carry only `int` / `str` parm
    /// literals (see `specialize:arg(0, 1)`,
    /// `specialize:argtype(0)`), so a comma split covers upstream's
    /// inputs without a full Python expression parser.
    pub fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        let Some(directive) = directive else {
            // upstream: `if directive is None: return pol.default_specialize`.
            return Ok(Specializer::Default);
        };
        let (normalized, parms) = parse_specializer_directive(directive)?;
        specializer_from_normalized(&normalized, parms)
    }

    /// RPython `AnnotatorPolicy.no_more_blocks_to_annotate(pol, annotator)`
    /// (policy.py:69-100).
    ///
    /// Drains pending specialization callbacks, then walks every
    /// `op.simple_call` / `op.call_args` instruction in the annotator's
    /// added-or-annotated block set. Call sites whose callee annotation
    /// carries `needs_sandboxing` are rewritten to invoke a sandbox
    /// trampoline instead.
    ///
    /// ```python
    /// key = ('sandboxing', s_func.const)
    /// if key not in bk.emulated_pbc_calls:
    ///     params_s = s_func.args_s
    ///     s_result = s_func.s_result
    ///     sandbox_trampoline = make_sandbox_trampoline(s_func.name, params_s, s_result)
    ///     sandbox_trampoline._signature_ = [SomeTuple(items=params_s)], s_result
    ///     bk.emulate_pbc_call(key, bk.immutablevalue(sandbox_trampoline), params_s)
    /// else:
    ///     s_trampoline = bk.emulated_pbc_calls[key][0]
    ///     sandbox_trampoline = s_trampoline.const
    /// new = instr.replace({instr.args[0]: Constant(sandbox_trampoline)})
    /// block.operations[i] = new
    /// ```
    ///
    /// Rust port: `needs_sandboxing` lives as an `Option<SandboxingPayload>`
    /// on `SomeBuiltin` (matches upstream's `setattr` on
    /// SomeExternalFunction). `make_sandbox_trampoline` is deferred —
    /// its body lives in `rpython/translator/sandbox/rsandbox.py` and
    /// depends on rtyper infrastructure — so the Rust branch emits a
    /// synthesized `HostObject::new_builtin_callable` placeholder and
    /// performs the `emulate_pbc_call` bookkeeping + instruction
    /// rewrite. Sandbox-mode translation is not yet end-to-end usable
    /// regardless, so the placeholder trampoline does not reach a
    /// backend.
    pub fn no_more_blocks_to_annotate(
        &self,
        ann: &super::annrpython::RPythonAnnotator,
    ) -> Result<(), super::model::AnnotatorError> {
        use super::bookkeeper::EmulatedPbcCallKey;
        use super::model::{AnnotatorError, SomePBC, SomeValue};
        use crate::flowspace::model::{BlockKey, ConstValue, Constant, Hlvalue, HostObject};
        use crate::flowspace::operation::OpKind;

        let bk = &ann.bookkeeper;
        // upstream: for callback in bk.pending_specializations: callback()
        let callbacks: Vec<Box<dyn Fn() -> Result<(), AnnotatorError>>> =
            bk.pending_specializations.borrow_mut().drain(..).collect();
        for callback in &callbacks {
            // upstream lets a callback exception propagate; surface the
            // first `Err` to the annotator's `complete()`.
            callback()?;
        }
        // upstream: del bk.pending_specializations[:]  (already drained)

        // upstream:
        //   if annotator.added_blocks is not None:
        //       all_blocks = annotator.added_blocks
        //   else:
        //       all_blocks = annotator.annotated
        let all_blocks: Vec<BlockKey> = {
            let added = ann.added_blocks.borrow();
            match added.as_ref() {
                Some(added_set) => added_set.keys().cloned().collect(),
                None => ann.annotated.borrow().keys().cloned().collect(),
            }
        };

        // upstream: for block in list(all_blocks):
        for block_key in all_blocks {
            let block_ref = {
                let all_blocks_index = ann.all_blocks.borrow();
                match all_blocks_index.get(&block_key).cloned() {
                    Some(b) => b,
                    None => continue,
                }
            };

            // Snapshot length / reify HLOperations so we can mutate the
            // block.operations slice in place without borrow conflicts.
            let op_count = block_ref.borrow().operations.len();
            for i in 0..op_count {
                let sp = block_ref.borrow().operations[i].clone();
                //   if not isinstance(instr, (op.simple_call, op.call_args)):
                //       continue
                let kind = OpKind::from_opname(&sp.opname);
                if !matches!(kind, Some(OpKind::SimpleCall | OpKind::CallArgs)) {
                    continue;
                }
                //   v_func = instr.args[0]
                let Some(v_func) = sp.args.first().cloned() else {
                    continue;
                };
                //   s_func = annotator.annotation(v_func)
                let Some(s_func) = ann.annotation(&v_func) else {
                    continue;
                };
                //   if not hasattr(s_func, 'needs_sandboxing'): continue
                let SomeValue::Builtin(ref builtin) = s_func else {
                    continue;
                };
                let Some(payload) = builtin.needs_sandboxing.clone() else {
                    continue;
                };
                let Some(func_const) = s_func.const_().cloned() else {
                    continue;
                };

                // upstream: key = ('sandboxing', s_func.const)
                let key = EmulatedPbcCallKey::sandboxing(&func_const);
                // upstream: if key not in bk.emulated_pbc_calls: ... else: ...
                let trampoline_host: HostObject = {
                    let existing = bk
                        .emulated_pbc_calls
                        .borrow()
                        .get(&key)
                        .map(|(pbc, _)| pbc.clone());
                    match existing {
                        Some(pbc) => {
                            //   s_trampoline = bk.emulated_pbc_calls[key][0]
                            //   sandbox_trampoline = s_trampoline.const
                            match pbc.base.const_box.as_ref().map(|c| c.value.clone()) {
                                Some(ConstValue::HostObject(obj)) => obj,
                                _ => continue,
                            }
                        }
                        None => {
                            //   sandbox_trampoline = make_sandbox_trampoline(...)
                            //   bk.emulate_pbc_call(key, bk.immutablevalue(sandbox_trampoline), params_s)
                            //
                            // `make_sandbox_trampoline` (rsandbox.py) produces
                            // a real Python function; deferred — we materialize
                            // a `HostObject::new_user_function` stand-in so
                            // `bk.immutablevalue` routes it through the
                            // user-function path that builds a real SomePBC,
                            // matching the shape `emulate_pbc_call` expects.
                            use crate::flowspace::model::GraphFunc;
                            let trampoline_name = format!("sandbox_trampoline({})", payload.name);
                            let globals = Constant::new(ConstValue::Dict(Default::default()));
                            let trampoline = HostObject::new_user_function(GraphFunc::new(
                                trampoline_name.clone(),
                                globals,
                            ));
                            let pbc_value = match bk
                                .immutablevalue(&ConstValue::HostObject(trampoline.clone()))
                            {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            let _ =
                                bk.emulate_pbc_call(key, &pbc_value, &payload.args_s, &[], None);
                            trampoline
                        }
                    }
                };

                // upstream:
                //   new = instr.replace({instr.args[0]: Constant(sandbox_trampoline)})
                //   block.operations[i] = new
                let new_arg =
                    Hlvalue::Constant(Constant::new(ConstValue::HostObject(trampoline_host)));
                let mut blk_mut = block_ref.borrow_mut();
                if let Some(op_mut) = blk_mut.operations.get_mut(i)
                    && !op_mut.args.is_empty()
                {
                    op_mut.args[0] = new_arg;
                }

                // Suppress unused-warning when the SomePBC import above
                // is structurally expected by upstream but not used on
                // this path.
                let _ = std::marker::PhantomData::<SomePBC>;
            }
        }
        Ok(())
    }
}

impl PolicyOps for AnnotatorPolicy {
    fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        AnnotatorPolicy::get_specializer(self, directive)
    }

    fn no_more_blocks_to_annotate(
        &self,
        ann: &super::annrpython::RPythonAnnotator,
    ) -> Result<(), super::model::AnnotatorError> {
        AnnotatorPolicy::no_more_blocks_to_annotate(self, ann)
    }
}

impl From<AnnotatorPolicy> for PolicyHandle {
    fn from(value: AnnotatorPolicy) -> Self {
        PolicyHandle::new(value)
    }
}

pub(crate) fn parse_specializer_directive(
    directive: &str,
) -> Result<(String, Vec<String>), PolicyError> {
    let (name, parms_src) = match directive.split_once('(') {
        None => (directive.to_string(), None),
        Some((n, rest)) => {
            let parms = rest.strip_suffix(')').ok_or_else(|| {
                PolicyError(format!("broken specialize directive parms: {directive}"))
            })?;
            (n.to_string(), Some(parms.to_string()))
        }
    };
    let parms: Vec<String> = match parms_src {
        None => Vec::new(),
        Some(s) if s.trim().is_empty() => Vec::new(),
        Some(s) => s.split(',').map(|p| p.trim().to_string()).collect(),
    };
    Ok((name.replace(':', "__"), parms))
}

pub(crate) fn specializer_from_normalized(
    normalized: &str,
    parms: Vec<String>,
) -> Result<Specializer, PolicyError> {
    match normalized {
        "default_specialize" => Ok(Specializer::Default),
        "specialize__memo" => Ok(Specializer::Memo),
        "specialize__arg" => Ok(Specializer::Arg {
            parms: parse_integer_argindices("arg", parms)?,
        }),
        "specialize__arg_or_var" => Ok(Specializer::ArgOrVar {
            parms: parse_integer_argindices("arg_or_var", parms)?,
        }),
        "specialize__argtype" => Ok(Specializer::Argtype {
            parms: parse_integer_argindices("argtype", parms)?,
        }),
        "specialize__arglistitemtype" => Ok(Specializer::Arglistitemtype {
            parms: parse_integer_argindices("arglistitemtype", parms)?,
        }),
        "specialize__call_location" => Ok(Specializer::CallLocation),
        "specialize__ll" => Ok(Specializer::Ll { parms }),
        "specialize__ll_and_arg" => Ok(Specializer::LlAndArg { parms }),
        other => Err(PolicyError(format!(
            "{other:?} specialize tag not defined in annotation policy AnnotatorPolicy"
        ))),
    }
}

/// `policy.py get_specializer` `eval`s specialize directive parms to
/// ints before calling `specialize_argvalue` / `arg_or_var` /
/// `argtype` / `arglistitemtype`.
fn parse_integer_argindices(kind: &str, parms: Vec<String>) -> Result<Vec<usize>, PolicyError> {
    parms
        .into_iter()
        .map(|parm| {
            parm.parse::<usize>().map_err(|_| {
                PolicyError(format!(
                    "specialize:{kind} expects integer indices, got {parm:?}"
                ))
            })
        })
        .collect()
}

/// RPython `raise AttributeError(...)` / `raise Exception(...)` from
/// `AnnotatorPolicy.get_specializer` (policy.py, :41-42).
///
/// The upstream class hierarchy mixes `AttributeError` and a bare
/// `Exception` on the same code path; Rust collapses both into a
/// single typed error with the upstream message verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_when_no_directive() {
        let pol = AnnotatorPolicy::new();
        assert_eq!(pol.get_specializer(None).unwrap(), Specializer::Default);
    }

    #[test]
    fn specialize_arg_parses_parm_lists() {
        let cases = [
            ("single", "specialize:arg(0)", vec![0]),
            ("multi", "specialize:arg(0, 1)", vec![0, 1]),
        ];
        let pol = AnnotatorPolicy::new();
        for (name, spec, parms) in cases {
            let sp = pol.get_specializer(Some(spec)).unwrap();
            assert_eq!(sp, Specializer::Arg { parms }, "case {name}");
        }
    }

    #[test]
    fn specialize_memo_no_parm() {
        let pol = AnnotatorPolicy::new();
        assert_eq!(
            pol.get_specializer(Some("specialize:memo")).unwrap(),
            Specializer::Memo
        );
    }

    #[test]
    fn specialize_argtype_call_location_argor_var() {
        let pol = AnnotatorPolicy::new();
        assert!(matches!(
            pol.get_specializer(Some("specialize:argtype(0)")).unwrap(),
            Specializer::Argtype { .. }
        ));
        assert_eq!(
            pol.get_specializer(Some("specialize:call_location"))
                .unwrap(),
            Specializer::CallLocation
        );
        assert!(matches!(
            pol.get_specializer(Some("specialize:arg_or_var(0)"))
                .unwrap(),
            Specializer::ArgOrVar { .. }
        ));
    }

    #[test]
    fn specialize_ll_variants() {
        let pol = AnnotatorPolicy::new();
        assert!(matches!(
            pol.get_specializer(Some("specialize:ll")).unwrap(),
            Specializer::Ll { .. }
        ));
        assert!(matches!(
            pol.get_specializer(Some("specialize:ll_and_arg")).unwrap(),
            Specializer::LlAndArg { .. }
        ));
    }

    #[test]
    fn unknown_directive_errors() {
        let pol = AnnotatorPolicy::new();
        let err = pol
            .get_specializer(Some("specialize:nonexistent"))
            .expect_err("unknown directive must error");
        assert!(err.0.contains("specialize tag not defined"));
    }

    #[test]
    fn broken_parms_errors() {
        let pol = AnnotatorPolicy::new();
        let err = pol
            .get_specializer(Some("specialize:arg(0"))
            .expect_err("missing closing paren must error");
        assert!(err.0.contains("broken specialize directive parms"));
    }

    #[test]
    fn empty_parms_list() {
        // "specialize:arg()" — parses as Arg with empty parms.
        let pol = AnnotatorPolicy::new();
        let sp = pol.get_specializer(Some("specialize:arg()")).unwrap();
        assert_eq!(sp, Specializer::Arg { parms: Vec::new() });
    }

    #[test]
    fn event_is_noop() {
        let pol = AnnotatorPolicy::new();
        let bk = Bookkeeper::new();
        pol.event(&bk, "some_event", &["arg1"]);
    }

    #[test]
    fn no_more_blocks_to_annotate_rewrites_sandboxed_call_site() {
        // Stand up a minimal annotator + graph + block containing a
        // `simple_call` whose callee annotation is a SomeBuiltin with
        // `needs_sandboxing = Some(payload)`. The policy walker must
        // rewrite `instr.args[0]` to a Constant(HostObject) carrying
        // the synthesized sandbox trampoline.
        use super::super::annrpython::RPythonAnnotator;
        use super::super::model::{SandboxingPayload, SomeBuiltin, SomeInteger, SomeValue};
        use crate::flowspace::model::{
            Block, BlockKey, ConstValue, Constant, FunctionGraph, Hlvalue, HostObject,
            SpaceOperation, Variable,
        };
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);

        // Build a callable HostObject and bind it on a variable as a
        // sandboxed SomeBuiltin.
        let callable = HostObject::new_builtin_callable("pkg.extfn");
        let mut builtin = SomeBuiltin::new("pkg.extfn", None, None);
        builtin.needs_sandboxing = Some(SandboxingPayload {
            name: "pkg.extfn".into(),
            args_s: vec![SomeValue::Integer(SomeInteger::default())],
            s_result: Box::new(SomeValue::Integer(SomeInteger::default())),
        });
        builtin.base.const_box = Some(Constant::new(ConstValue::HostObject(callable.clone())));
        let s_func = SomeValue::Builtin(builtin);

        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, s_func);
        let v_arg = Variable::named("x");
        let result_v = Variable::named("r");

        // Build a block with one simple_call op.
        let block = Rc::new(RefCell::new(Block::new(vec![])));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "simple_call",
            vec![Hlvalue::Variable(v_func), Hlvalue::Variable(v_arg)],
            Hlvalue::Variable(result_v),
        ));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("caller", block.clone())));
        let bkey = BlockKey::of(&block);
        ann.annotated.borrow_mut().insert(bkey.clone(), None);
        ann.all_blocks.borrow_mut().insert(bkey, block.clone());
        let _ = graph; // keep the Rc alive for the duration of the test

        // Run the policy walker.
        ann.policy
            .borrow()
            .no_more_blocks_to_annotate(&ann)
            .expect("no_more_blocks_to_annotate");

        // Verify the callee arg was rewritten to a Constant(HostObject).
        let args0 = block.borrow().operations[0].args[0].clone();
        match args0 {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => {
                    assert!(h.qualname().starts_with("sandbox_trampoline("));
                }
                other => panic!("expected HostObject, got {other:?}"),
            },
            other => panic!("expected Hlvalue::Constant, got {other:?}"),
        }

        // Verify the sandbox entry landed in emulated_pbc_calls.
        let key = super::super::bookkeeper::EmulatedPbcCallKey::sandboxing(
            &ConstValue::HostObject(callable),
        );
        assert!(
            ann.bookkeeper
                .emulated_pbc_calls
                .borrow()
                .contains_key(&key)
        );
    }

    #[test]
    fn no_more_blocks_to_annotate_drains_pending_specializations() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ann = super::super::annrpython::RPythonAnnotator::new(None, None, None, false);
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let c = Arc::clone(&counter);
            ann.bookkeeper
                .pending_specializations
                .borrow_mut()
                .push(Box::new(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }));
        }
        {
            let c = Arc::clone(&counter);
            ann.bookkeeper
                .pending_specializations
                .borrow_mut()
                .push(Box::new(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }));
        }
        ann.policy
            .borrow()
            .no_more_blocks_to_annotate(&ann)
            .expect("drain");
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert!(ann.bookkeeper.pending_specializations.borrow().is_empty());
    }

    #[test]
    fn no_more_blocks_to_annotate_propagates_callback_error() {
        use super::super::model::AnnotatorError;
        let ann = super::super::annrpython::RPythonAnnotator::new(None, None, None, false);
        ann.bookkeeper
            .pending_specializations
            .borrow_mut()
            .push(Box::new(|| Err(AnnotatorError::new("boom from finish"))));
        let err = ann
            .policy
            .borrow()
            .no_more_blocks_to_annotate(&ann)
            .expect_err("callback Err must surface, not panic");
        assert!(err.to_string().contains("boom from finish"));
    }
}
