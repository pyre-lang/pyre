//! Test-only smoke coverage for the `#[pyre_class]` / `#[pyre_methods]` /
//! `#[pyre_function]` / `py_module!` macro arms.
//!
//! These probes previously lived in the `_random` module proper, where
//! they leaked non-PyPy public API (`_unwrap_alias_probe`, `_PROBE_CONST`,
//! `_ProbeError`, `__reduce__`, `raw_state`, …).  Relocated here under
//! `#[cfg(test)]` so the macro codegen stays exercised end-to-end while
//! `_random`'s import-time surface matches `pypy/module/_random` exactly.
//!
//! Compiling the test build forces every macro arm below to expand, which
//! is the codegen smoke check; the `#[test]` functions additionally run
//! the wrappers that need no module-init bootstrap.

use pyre_object::*;

/// `#[pyre_class]` typed payload exercising getter/setter/deleter,
/// `__reduce__`, and the declarative `base = <expr>` arm.
#[pyre_interpreter::pyre_class("_pyre_smoke.Demo", weakrefable)]
#[derive(Default)]
pub struct Demo {
    pub state: u64,
}

#[pyre_interpreter::pyre_methods(
    doc = "Demo() -> smoke-test typed payload.",
    weakrefable,
    // `base = <expr>` arm — `object` is the implicit default, so this is
    // behaviorally identical while exercising the declarative-base plumbing.
    base = pyre_interpreter::typedef::w_object()
)]
impl Demo {
    fn __init__(&mut self, #[default(0i64)] seed: i64) {
        self.state = seed as u64;
    }
    fn getstate(&self) -> PyObjectRef {
        pyre_interpreter::pytuple![self.state as i64]
    }
    // `__reduce__` pickling-hook arm.
    fn __reduce__(&self) -> PyObjectRef {
        pyre_interpreter::pytuple![type_object(), pyre_interpreter::pytuple![], self.getstate()]
    }
    // A positional-or-keyword parameter plus a keyword-only default: the
    // instance-method arm must build a `Signature` (`self` posonly, then
    // `factor`, then a `marker_kwonly()` tail with `bias`) so the call path
    // binds keywords by name and the wrapper preamble needs no marker-dict
    // strip.
    fn combine(
        &self,
        factor: i64,
        #[kwonly]
        #[default(0i64)]
        bias: i64,
    ) -> i64 {
        self.state as i64 * factor + bias
    }
    // `#[getter]` / `#[setter]` / `#[deleter]` GetSetProperty quad.
    #[getter(doc = "raw 64-bit state as a signed int")]
    fn raw_state(&self) -> i64 {
        self.state as i64
    }
    #[setter]
    fn set_raw_state(&mut self, v: i64) {
        self.state = v as u64;
    }
    #[deleter("raw_state")]
    fn del_raw_state(&mut self) {
        self.state = 0;
    }
    // `set_traceback` pairs with getter `__traceback__` as one getset.
    #[getter]
    fn __traceback__(&self) -> PyObjectRef {
        w_none()
    }
    #[setter]
    fn set_traceback(&mut self, _value: PyObjectRef) {}
}

/// `PyPath` typed-receiver alias.
#[pyre_interpreter::pyre_function]
fn _seed_from_path(path: PyPath) -> i64 {
    path.into_iter().map(|b| b as i64).sum()
}

/// `Vec<i64>` auto return-wrap.
#[pyre_interpreter::pyre_function]
fn _path_bytes(path: PyPath) -> Vec<i64> {
    path.into_iter().map(|b| b as i64).collect()
}

/// One parameter per text / int unwrap alias, so the generated unwrap +
/// binding-type expansion is exercised for each.
#[pyre_interpreter::pyre_function]
fn _unwrap_alias_probe(
    u: PyUnicode,
    u8s: PyUtf8,
    ton: PyTextOrNone,
    t0n: PyText0OrNone,
    buf: PyBufferStr,
    cnn: PyCNonNegInt,
) -> i64 {
    let mut acc = u.len() as i64 + u8s.len() as i64 + buf.len() as i64 + cnn as i64;
    acc += ton.map(|s| s.len() as i64).unwrap_or(-1);
    acc += t0n.map(|s| s.len() as i64).unwrap_or(-1);
    acc
}

/// A Signature-bound scope includes keyword-only slots in its flat argument
/// array.  The typed wrapper must not recount those slots as positionals.
#[pyre_interpreter::pyre_function]
fn _kwonly_bound_probe(
    value: i64,
    #[kwonly]
    #[default(0i64)]
    adjustment: i64,
) -> i64 {
    value + adjustment
}

/// A `#[posonly]` marker on the first non-positional-only parameter ends the
/// positional-only run before it: `base` is positional-only, so the derived
/// `Signature` carries `posonlyargcount == 1` and a keyword named `base` is
/// rejected by `raise_if_posonly_kwds`.
#[pyre_interpreter::pyre_function]
fn _posonly_bound_probe(
    base: i64,
    #[posonly]
    #[default(0i64)]
    offset: i64,
) -> i64 {
    base + offset
}

pyre_interpreter::py_module! {
    "_pyre_smoke",
    interpleveldefs: {
        "Demo" => type_object(),
        "_seed_from_path" => pyre_interpreter::make_builtin_function("_seed_from_path", _seed_from_path),
        "_path_bytes" => pyre_interpreter::make_builtin_function("_path_bytes", _path_bytes),
        "_unwrap_alias_probe" =>
            pyre_interpreter::make_builtin_function("_unwrap_alias_probe", _unwrap_alias_probe),
    },
    int_constants: {
        // `int_constants:` arm — a plain integer module constant.
        "_PROBE_CONST" => 42,
    },
    exceptions: {
        // `exceptions:` arm — a module-local exception class.
        "_ProbeError" => pyre_interpreter::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before smoke init"),
    },
    appleveldefs: {
        // `appleveldefs:` arm — a pure-Python helper sharing the namespace.
        "app_smoke.py" => ["_ascii_seed"],
    },
    inline_app: {
        // `inline_app:` arm — an inline Python snippet.
        "def _is_even(n):\n    return n % 2 == 0\n" => ["_is_even"],
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `#[pyre_function]` unwrap-alias wrappers convert their typed
    /// arguments and return a wrapped int.  Exercises `PyUnicode` /
    /// `PyUtf8` / `PyTextOrNone` / `PyText0OrNone` / `PyBufferStr` /
    /// `PyCNonNegInt` through the space-level converters.
    #[test]
    fn unwrap_alias_probe_runs() {
        pyre_interpreter::typedef::init_typeobjects();
        let args = [
            w_str_new("ab"),
            w_str_new("cde"),
            w_none(),
            w_str_new("fghi"),
            pyre_object::bytesobject::w_bytes_from_bytes(b"jk"),
            w_int_new(7),
        ];
        let result = _unwrap_alias_probe(&args).expect("probe should succeed");
        // 2 + 3 + 2 + 7 (u + u8s + buf + cnn) - 1 (ton None) + 4 (t0n len)
        assert_eq!(unsafe { w_int_get_value(result) }, 2 + 3 + 2 + 7 - 1 + 4);
    }

    /// A missing required argument raises `TypeError` instead of panicking.
    #[test]
    fn missing_required_arg_is_type_error() {
        pyre_interpreter::typedef::init_typeobjects();
        let err = _seed_from_path(&[]).expect_err("missing path should error");
        assert_eq!(err.kind, pyre_interpreter::PyErrorKind::TypeError);
    }

    #[test]
    fn keyword_only_bound_slot_is_not_counted_as_positional() {
        pyre_interpreter::typedef::init_typeobjects();
        let signature = _kwonly_bound_probe_pyre_sig().expect("derived signature");
        let bound = pyre_interpreter::call::parse_builtin_signature(
            &signature,
            "_kwonly_bound_probe",
            &[w_int_new(40)],
            &[w_str_new("adjustment")],
            &[w_int_new(2)],
            None,
            pyre_object::PY_NULL,
        )
        .expect("signature binding");
        let result = _kwonly_bound_probe(&bound).expect("bound keyword-only scope");
        assert_eq!(unsafe { w_int_get_value(result) }, 42);
    }

    #[test]
    fn unmatched_keywords_pack_from_parallel_rooted_lists() {
        pyre_interpreter::typedef::init_typeobjects();
        let signature = pyre_interpreter::gateway::Signature::new(
            vec!["head"],
            None,
            Some("kwargs"),
            /*kwonlyargcount*/ 0,
            /*posonlyargcount*/ 0,
        );
        let bound = pyre_interpreter::call::parse_builtin_signature(
            &signature,
            "probe",
            &[w_int_new(1)],
            &[w_str_new("alpha"), w_str_new("beta")],
            &[w_int_new(2), w_int_new(3)],
            None,
            pyre_object::PY_NULL,
        )
        .expect("unmatched keywords should pack into **kwargs");
        assert_eq!(bound.len(), 2);
        let w_kwargs = bound[1];
        let w_alpha = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(w_kwargs, "alpha")
                .expect("alpha in **kwargs")
        };
        let w_beta = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(w_kwargs, "beta")
                .expect("beta in **kwargs")
        };
        assert_eq!(unsafe { w_int_get_value(w_alpha) }, 2);
        assert_eq!(unsafe { w_int_get_value(w_beta) }, 3);
    }

    #[test]
    fn posonly_marker_makes_leading_param_positional_only() {
        pyre_interpreter::typedef::init_typeobjects();
        let signature = _posonly_bound_probe_pyre_sig().expect("derived signature");
        assert_eq!(signature.posonlyargcount, 1);
        assert_eq!(signature.argnames, vec!["base", "offset"]);

        // The positional-only `base` binds fine by position, and `offset`
        // still binds by keyword.
        let bound = pyre_interpreter::call::parse_builtin_signature(
            &signature,
            "_posonly_bound_probe",
            &[w_int_new(40)],
            &[w_str_new("offset")],
            &[w_int_new(2)],
            None,
            pyre_object::PY_NULL,
        )
        .expect("signature binding");
        let result = _posonly_bound_probe(&bound).expect("bound positional-only scope");
        assert_eq!(unsafe { w_int_get_value(result) }, 42);

        // Passing the positional-only `base` as a keyword is a TypeError.
        let err = pyre_interpreter::call::parse_builtin_signature(
            &signature,
            "_posonly_bound_probe",
            &[],
            &[w_str_new("base")],
            &[w_int_new(40)],
            None,
            pyre_object::PY_NULL,
        )
        .expect_err("positional-only name as keyword must error");
        assert_eq!(err.kind, pyre_interpreter::PyErrorKind::TypeError);
    }

    /// `Vec<i64>` return auto-wraps to a list.
    #[test]
    fn path_bytes_returns_list() {
        pyre_interpreter::typedef::init_typeobjects();
        let result = _path_bytes(&[w_str_new("AB")]).expect("path bytes");
        assert!(unsafe { is_list(result) });
        assert_eq!(unsafe { w_list_len(result) }, 2);
    }

    /// PyPy's `generic_new_descr` accepts the constructor arguments and
    /// leaves them for `__init__`; the synthesized allocator must do the same.
    #[test]
    fn synthesized_new_accepts_init_arguments() {
        pyre_interpreter::typedef::init_typeobjects();
        let cls = type_object();
        let seed = w_int_new(37);
        let obj = __majit_wrap___new__(&[cls, seed]).expect("synthesized __new__");
        let demo = Demo::from_obj(obj).expect("Demo allocation");
        assert_eq!(demo.state, 0);
        __majit_wrap___init__(&[obj, seed]).expect("Demo.__init__");
        assert_eq!(Demo::from_obj(obj).expect("initialized Demo").state, 37);
    }

    /// A `#[pyre_methods]` instance method with a positional-or-keyword and a
    /// keyword-only parameter binds identically whether the keyword arrives
    /// positionally or by name through `parse_builtin_signature` — the same
    /// invariant the caller relies on to hand the wrapper a marker-free,
    /// PY_NULL-padded scope.
    #[test]
    fn instance_method_binds_keyword_only_through_signature() {
        pyre_interpreter::typedef::init_typeobjects();
        let cls = type_object();
        let obj = __majit_wrap___new__(&[cls]).expect("synthesized __new__");
        __majit_wrap___init__(&[obj, w_int_new(7)]).expect("Demo.__init__");

        // The `Signature` the instance-method arm derives for
        // `combine(&self, factor, #[kwonly] bias)`: `self` positional-only,
        // then `factor`, then a keyword-only `bias`.
        let signature = pyre_interpreter::gateway::Signature::new(
            vec!["self", "factor", "bias"],
            None,
            None,
            /*kwonlyargcount*/ 1,
            /*posonlyargcount*/ 1,
        );
        let bound = pyre_interpreter::call::parse_builtin_signature(
            &signature,
            "combine",
            &[obj, w_int_new(3)],
            &[w_str_new("bias")],
            &[w_int_new(5)],
            None,
            pyre_object::PY_NULL,
        )
        .expect("signature binding");
        // 7 * 3 + 5 through the bound (marker-free, PY_NULL-padded) scope.
        let via_keyword = __majit_wrap_combine(&bound).expect("keyword-bound combine");
        assert_eq!(unsafe { w_int_get_value(via_keyword) }, 26);

        // The all-positional call omits `bias`, so its `#[default(0)]` applies.
        let via_positional =
            __majit_wrap_combine(&[obj, w_int_new(3)]).expect("positional combine");
        assert_eq!(unsafe { w_int_get_value(via_positional) }, 21);
    }

    /// TypeDef ownership is process-global: another OS thread must observe
    /// the exact same Python type object, not a fresh TLS allocation.
    #[test]
    fn generated_type_object_is_process_global() {
        pyre_interpreter::typedef::init_typeobjects();
        let expected = type_object() as usize;
        let observed = std::thread::spawn(|| type_object() as usize)
            .join()
            .expect("type lookup thread");
        assert_eq!(observed, expected);
    }

    /// `#[pyre_class]` derives the mapdict-mixin bit from the `map` + `storage`
    /// field pair. `W_Random` no longer carries that pair: `typedef.py`
    /// `_getusercls` puts it on `RANDOM_USER_TYPE`.
    #[test]
    fn storage_class_mapdict_mixin_bit_follows_the_field_pair() {
        use pyre_object::lltype::PyreClassPyTypeOf;
        assert!(!crate::module::_random::W_Random::HAS_MAPDICT_MIXIN);
        assert!(!super::Demo::HAS_MAPDICT_MIXIN);
        assert_eq!(crate::module::_random::RANDOM_TYPE.mapdict_offset, 0);
        assert_ne!(crate::module::_random::RANDOM_USER_TYPE.mapdict_offset, 0);
    }

    /// `typedef.py` `_getusercls`: a `_random.Random` subclass instance is
    /// `W_RandomUser` (`RANDOM_USER_TYPE`) and carries mapdict storage. An
    /// exact instance is the base layout and does not.
    #[test]
    fn random_subclass_instance_carries_user_typeptr() {
        let w_subtype =
            pyre_object::w_type_new("RandomSub", pyre_object::PY_NULL, std::ptr::null_mut());
        let obj = crate::module::_random::W_Random::allocate_instance(
            crate::module::_random::W_Random::default(),
            w_subtype,
        );
        unsafe {
            assert!(std::ptr::eq(
                (*obj).ob_type,
                &crate::module::_random::RANDOM_USER_TYPE
            ));
            assert!(pyre_interpreter::objspace::std::mapdict::has_mapdict_layout(obj));
        }
        let exact = crate::module::_random::W_Random::allocate_instance(
            crate::module::_random::W_Random::default(),
            pyre_object::PY_NULL,
        );
        unsafe {
            assert!(std::ptr::eq(
                (*exact).ob_type,
                &crate::module::_random::RANDOM_TYPE
            ));
            assert!(!pyre_interpreter::objspace::std::mapdict::has_mapdict_layout(exact));
        }
    }
}
