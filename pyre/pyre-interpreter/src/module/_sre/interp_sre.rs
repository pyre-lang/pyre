//! _sre module — SRE regex engine bridge.
//!
//! Uses sre-engine crate (RustPython's SRE bytecode interpreter) in
//! place of `rpython/rlib/rsre`; the object model follows
//! `pypy/module/_sre/interp_sre.py` (`W_SRE_Pattern` / `W_SRE_Match`
//! typed fields, `pyre_object::interp_sre`).

use crate::{
    make_builtin_function, make_builtin_function_with_arity, make_module_builtin_function,
    make_module_builtin_function_with_arity,
};
use pyre_object::interp_sre::{
    W_SRE_Match, W_SRE_Pattern, W_SRE_Scanner, W_SRE_Template, is_sre_match, is_sre_pattern,
    is_sre_scanner, is_sre_template, w_sre_match_get_span, w_sre_match_new, w_sre_pattern_new,
    w_sre_scanner_new, w_sre_template_new,
};
use pyre_object::*;
use rustpython_wtf8::{Wtf8, Wtf8Buf};
use sre_engine::engine::{Request, SearchIter, State};
use sre_engine::string::{StrDrive, StringCursor};

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `W_SRE_Pattern.typedef` and the sibling classes are module TypeDefs.
    // Empty startup does not import `_sre`, so `init_typeobjects` does not
    // build them. Pin `ns` first: this build collects and can drop the GIL.
    crate::typedef::ensure_sre_typeobjects();
    // Must equal `re/_constants.py:MAGIC` (the bundled stdlib) — `_compiler.py`
    // asserts `_sre.MAGIC == MAGIC` at import time.
    crate::__pyre_put_new!(ns_slot, "MAGIC", w_int_new(20230612)); // SRE magic number
    crate::__pyre_put_new!(ns_slot, "CODESIZE", w_int_new(sre_engine::CODESIZE as i64));
    crate::__pyre_put_new!(
        ns_slot,
        "MAXREPEAT",
        w_int_new(sre_engine::MAXREPEAT as i64)
    );
    // `sre.h SRE_MAXGROUPS` halves `INT32_MAX`; the engine crate halves its own
    // `MAXREPEAT`, which is unsigned, and lands one bit wide. `re/_parser.py`
    // bounds a pattern's group count and two group references on this number,
    // so it is the one a caller can reach.
    crate::__pyre_put_new!(ns_slot, "MAXGROUPS", w_int_new(i32::MAX as i64 / 2));
    // _sre module-level functions: PyPy mixedmodule.py:111-116 wraps these
    // as BuiltinFunction so storing them on a user class does not bind self.
    crate::__pyre_put_new!(
        ns_slot,
        "compile",
        make_module_builtin_function("compile", sre_compile)
    );
    // `_sre.template` (`sre.c` `_sre_template_impl`). `re._compile_template`
    // caches its result; expand/sub consume that object instead of reparsing.
    crate::__pyre_put_new!(
        ns_slot,
        "template",
        make_module_builtin_function_with_arity("template", sre_template, 2)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "ascii_iscased",
        make_module_builtin_function_with_arity(
            "ascii_iscased",
            |args| {
                if args.is_empty() {
                    return Ok(w_bool_from(false));
                }
                let ch = unsafe { w_int_get_value(args[0]) } as u32;
                Ok(w_bool_from(ch < 128 && (ch as u8).is_ascii_alphabetic()))
            },
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "unicode_iscased",
        make_module_builtin_function_with_arity(
            "unicode_iscased",
            |args| {
                if args.is_empty() {
                    return Ok(w_bool_from(false));
                }
                let ch = char::from_u32(unsafe { w_int_get_value(args[0]) } as u32).unwrap_or('\0');
                Ok(w_bool_from(ch.is_alphabetic()))
            },
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "ascii_tolower",
        make_module_builtin_function_with_arity(
            "ascii_tolower",
            |args| {
                if args.is_empty() {
                    return Ok(w_int_new(0));
                }
                let ch = unsafe { w_int_get_value(args[0]) };
                Ok(w_int_new(if (0..128).contains(&ch) {
                    (ch as u8).to_ascii_lowercase() as i64
                } else {
                    ch
                }))
            },
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "unicode_tolower",
        make_module_builtin_function_with_arity(
            "unicode_tolower",
            |args| {
                if args.is_empty() {
                    return Ok(w_int_new(0));
                }
                let c = char::from_u32(unsafe { w_int_get_value(args[0]) } as u32).unwrap_or('\0');
                Ok(w_int_new(c.to_lowercase().next().unwrap_or(c) as i64))
            },
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "getcodesize",
        make_module_builtin_function_with_arity(
            "getcodesize",
            |_| Ok(w_int_new(sre_engine::CODESIZE as i64)),
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "getlower",
        make_module_builtin_function_with_arity(
            "getlower",
            |args| {
                if args.is_empty() {
                    return Ok(w_int_new(0));
                }
                Ok(w_int_new(sre_engine::string::lower_unicode(
                    unsafe { w_int_get_value(args[0]) } as u32
                ) as i64))
            },
            2,
        )
    );
    // `W_SRE_Pattern.typedef` / `W_SRE_Match.typedef` (`interp_sre.py`)
    // are published by `ensure_sre_typeobjects` above.  Instances carry
    // `pyre_object::interp_sre` typed payloads.
    Ok(())
}

/// `args[1]` as the typed pattern receiver for getsets registered on
/// the 're.Pattern' typedef (`args[0]` is the descriptor).
fn sre_pattern_receiver(args: &[PyObjectRef]) -> Result<*const W_SRE_Pattern, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if unsafe { is_sre_pattern(self_) } {
        Ok(self_ as *const W_SRE_Pattern)
    } else {
        Err(crate::PyError::type_error("descriptor is for 're.Pattern'"))
    }
}

/// `args[1]` as the typed match receiver for getsets registered on
/// the 're.Match' typedef.
fn sre_match_receiver(args: &[PyObjectRef]) -> Result<*const W_SRE_Match, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if unsafe { is_sre_match(self_) } {
        Ok(self_ as *const W_SRE_Match)
    } else {
        Err(crate::PyError::type_error("descriptor is for 're.Match'"))
    }
}

/// W_SRE_Pattern.typedef (interp_sre.py): instance methods are
/// registered on the type so `pat.match(s)` binds `pat` as `self`,
/// plus the `flags` / `groupindex` / `groups` / `pattern` attribute
/// properties (interp_sre.py).
pub(crate) fn init_sre_pattern_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // interp_sre.py `__new__ = interp2app(SRE_Pattern__new__)`.
    {
        let func = make_builtin_function("__new__", sre_pattern_new);
        let func_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            pyre_object::function::w_staticmethod_new(pyre_object::gc_roots::shadow_stack_get(
                func_slot
            ))
        );
    }
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "match",
            make_builtin_function("match", sre_pattern_match)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "fullmatch",
            make_builtin_function("fullmatch", sre_pattern_fullmatch)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "search",
            make_builtin_function("search", sre_pattern_search)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "findall",
            make_builtin_function("findall", sre_pattern_findall)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "finditer",
            make_builtin_function("finditer", sre_pattern_finditer)
        )
    };
    // interp_sre.py `scanner = interp2app(W_SRE_Pattern.finditer_w)`
    // — CPython/PyPy expose the same iterator constructor under both names.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "scanner",
            make_builtin_function("scanner", sre_pattern_finditer)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "sub",
            make_builtin_function("sub", sre_pattern_sub)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "subn",
            make_builtin_function("subn", sre_pattern_subn)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "split",
            make_builtin_function("split", sre_pattern_split)
        )
    };
    // interp_sre.py:651-653 `__repr__`/`__copy__`/`__deepcopy__`
    // (copy_identity_w returns self — compiled patterns are immutable).
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            make_builtin_function("__repr__", sre_pattern_repr)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__copy__",
            make_builtin_function("__copy__", sre_pattern_copy)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__deepcopy__",
            make_builtin_function("__deepcopy__", sre_pattern_copy)
        )
    };
    // interp_sre.py:655-657 value equality / hash.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__eq__",
            make_builtin_function("__eq__", sre_pattern_eq)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__hash__",
            make_builtin_function("__hash__", sre_pattern_hash)
        )
    };
    unsafe { crate::__pyre_put_new!(ns_slot, "__weakref__", crate::typedef::weakref_descr()) };
    // interp_sre.py:667-668 `generic_alias_class_getitem` as classmethod.
    {
        let func = make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        );
        let func_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        crate::__pyre_put_new!(
            ns_slot,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(pyre_object::gc_roots::shadow_stack_get(
                func_slot
            ))
        );
    }
    // interp_sre.py:662-663 `flags = interp_attrproperty('flags', ...,
    // wrapfn="newint")`.
    {
        let getter = make_builtin_function_with_arity(
            "flags",
            |args| Ok(w_int_new(unsafe { (*sre_pattern_receiver(args)?).flags })),
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "flags",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "flags"
            )
        );
    }
    // interp_sre.py `groupindex = GetSetProperty(fget_groupindex)`
    // (:202-206 — a dict groupindex is exposed through a dictproxy).
    {
        let getter = make_builtin_function_with_arity(
            "groupindex",
            |args| {
                let w_groupindex = unsafe { (*sre_pattern_receiver(args)?).w_groupindex };
                if unsafe { is_dict(w_groupindex) } {
                    return Ok(pyre_object::dictproxyobject::w_dict_proxy_new(w_groupindex));
                }
                Ok(w_groupindex)
            },
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "groupindex",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "groupindex"
            )
        );
    }
    // interp_sre.py:665-666 `groups = interp_attrproperty('num_groups',
    // ..., wrapfn="newint")`.
    {
        let getter = make_builtin_function_with_arity(
            "groups",
            |args| {
                Ok(w_int_new(unsafe {
                    (*sre_pattern_receiver(args)?).num_groups
                }))
            },
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "groups",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "groups"
            )
        );
    }
    // interp_sre.py:667 `pattern = interp_attrproperty_w('w_pattern', ...)`.
    {
        let getter = make_builtin_function_with_arity(
            "pattern",
            |args| Ok(unsafe { (*sre_pattern_receiver(args)?).w_pattern }),
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "pattern",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "pattern"
            )
        );
    }
}

/// W_SRE_Match.typedef (interp_sre.py): methods + the `re` /
/// `string` / `pos` / `endpos` / `lastgroup` / `lastindex` attribute
/// properties.
pub(crate) fn init_sre_match_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "group",
            make_builtin_function("group", sre_match_group)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "groups",
            make_builtin_function("groups", sre_match_groups)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "start",
            make_builtin_function("start", sre_match_start)
        )
    };
    unsafe { crate::__pyre_put_new!(ns_slot, "end", make_builtin_function("end", sre_match_end)) };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "span",
            make_builtin_function("span", sre_match_span)
        )
    };
    // interp_sre.py `groupdict = interp2app(W_SRE_Match.groupdict_w)`.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "groupdict",
            make_builtin_function("groupdict", sre_match_groupdict)
        )
    };
    // interp_sre.py `__getitem__ = interp2app(W_SRE_Match.descr_getitem)`.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__getitem__",
            make_builtin_function("__getitem__", sre_match_getitem)
        )
    };
    // interp_sre.py `expand = interp2app(W_SRE_Match.expand_w)`.
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "expand",
            make_builtin_function("expand", sre_match_expand)
        )
    };
    // interp_sre.py fget_endpos `__copy__`/`__deepcopy__`/`__repr__`
    // (copy_identity_w returns self — match results are immutable).
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            make_builtin_function("__repr__", sre_match_repr)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__copy__",
            make_builtin_function("__copy__", sre_match_copy)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__deepcopy__",
            make_builtin_function("__deepcopy__", sre_match_copy)
        )
    };
    // interp_sre.py:887 `re = interp_attrproperty_w('srepat', ...)`.
    {
        let getter = make_builtin_function_with_arity(
            "re",
            |args| Ok(unsafe { (*sre_match_receiver(args)?).w_srepat }),
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "re",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "re"
            )
        );
    }
    // interp_sre.py `string = GetSetProperty(fget_string)` (:866-867).
    {
        let getter = make_builtin_function_with_arity(
            "string",
            |args| Ok(unsafe { (*sre_match_receiver(args)?).w_string }),
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "string",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "string"
            )
        );
    }
    // interp_sre.py `pos = GetSetProperty(fget_pos)` (:851-852).
    {
        let getter = make_builtin_function_with_arity(
            "pos",
            |args| Ok(w_int_new(unsafe { (*sre_match_receiver(args)?).pos })),
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "pos",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "pos"
            )
        );
    }
    // interp_sre.py `endpos = GetSetProperty(fget_endpos)` (:854-855).
    {
        let getter = make_builtin_function_with_arity(
            "endpos",
            |args| Ok(w_int_new(unsafe { (*sre_match_receiver(args)?).endpos })),
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "endpos",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "endpos"
            )
        );
    }
    // interp_sre.py `lastgroup = GetSetProperty(fget_lastgroup)`
    // (:831-839 — the group name from `w_indexgroup[lastindex]`).
    {
        let getter = make_builtin_function_with_arity(
            "lastgroup",
            |args| {
                let m = sre_match_receiver(args)?;
                let lastindex = unsafe { (*m).lastindex };
                if lastindex < 0 {
                    return Ok(w_none());
                }
                let w_indexgroup = unsafe { (*(*m).w_srepat.cast::<W_SRE_Pattern>()).w_indexgroup };
                let found = unsafe {
                    if is_list(w_indexgroup) {
                        w_list_getitem(w_indexgroup, lastindex)
                    } else if is_tuple(w_indexgroup) {
                        w_tuple_getitem(w_indexgroup, lastindex)
                    } else {
                        None
                    }
                };
                Ok(found.unwrap_or_else(w_none))
            },
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "lastgroup",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "lastgroup"
            )
        );
    }
    // interp_sre.py `lastindex = GetSetProperty(fget_lastindex)`
    // (:841-845).
    {
        let getter = make_builtin_function_with_arity(
            "lastindex",
            |args| {
                let lastindex = unsafe { (*sre_match_receiver(args)?).lastindex };
                if lastindex >= 0 {
                    Ok(w_int_new(lastindex))
                } else {
                    Ok(w_none())
                }
            },
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "lastindex",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "lastindex"
            )
        );
    }
    // interp_sre.py `regs = GetSetProperty(W_SRE_Match.fget_regs)`.
    {
        let getter = make_builtin_function_with_arity("regs", sre_match_regs, 2);
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "regs",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "regs"
            )
        );
    }
    // interp_sre.py:894-895 `generic_alias_class_getitem` as classmethod.
    {
        let func = make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        );
        let func_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        crate::__pyre_put_new!(
            ns_slot,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(pyre_object::gc_roots::shadow_stack_get(
                func_slot
            ))
        );
    }
}

/// `_sre.SRE_Template` has no methods (`sre.c template_slots` is dealloc /
/// traverse / clear only).
pub(crate) fn init_sre_template_type(_ns: PyObjectRef) {}

/// W_SRE_Scanner.typedef (interp_sre.py): the finditer/scanner
/// iterator — `__iter__`/`__next__` plus the undocumented `match`/`search`
/// methods and the `pattern` attribute property.
pub(crate) fn init_sre_scanner_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__iter__",
            make_builtin_function("__iter__", sre_scanner_iter)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__next__",
            make_builtin_function("__next__", sre_scanner_next_w)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "match",
            make_builtin_function("match", sre_scanner_match)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "search",
            make_builtin_function("search", sre_scanner_search)
        )
    };
    // interp_sre.py:955 `pattern = interp_attrproperty_w('srepat', ...)`.
    {
        let getter = make_builtin_function_with_arity(
            "pattern",
            |args| {
                let self_ = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                if unsafe { is_sre_scanner(self_) } {
                    Ok(unsafe { (*(self_ as *const W_SRE_Scanner)).w_srepat })
                } else {
                    Err(crate::PyError::type_error(
                        "descriptor is for '_sre.SRE_Scanner'",
                    ))
                }
            },
            2,
        );
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        crate::__pyre_put_new!(
            ns_slot,
            "pattern",
            crate::typedef::make_getset_descriptor_named(
                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                "pattern"
            )
        );
    }
}

/// _sre.compile(pattern, flags, code, groups, groupindex, indexgroup)
/// — `SRE_Pattern__new__` (interp_sre.py).
fn sre_compile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "_sre.compile() requires at least 3 arguments",
        ));
    }
    let flags = unsafe { w_int_get_value(args[1]) };
    let groups = if args.len() > 3 {
        unsafe { w_int_get_value(args[3]) }
    } else {
        0
    };
    // `extract_code` allocates.  Publish every already-live operand first,
    // then mint the omitted defaults — a sequential pin of a fresh object
    // after a published live set is safe; pinning an unpublished live
    // second slice is not.
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = vec![args[0], args[2]];
    if args.len() > 4 {
        live.push(args[4]);
    }
    if args.len() > 5 {
        live.push(args[5]);
    }
    let base = roots.pin_roots(&live);
    let pattern_slot = base;
    let code_list_slot = base + 1;
    let groupindex_slot = if args.len() > 4 {
        base + 2
    } else {
        let _ = roots.pin_root(w_dict_new());
        pyre_object::gc_roots::shadow_stack_len() - 1
    };
    let indexgroup_slot = if args.len() > 5 {
        base + 3
    } else {
        let _ = roots.pin_root(w_tuple_new(vec![]));
        pyre_object::gc_roots::shadow_stack_len() - 1
    };

    let pattern = roots.get(pattern_slot);
    if !unsafe { is_none(pattern) }
        && !unsafe { is_str(pattern) }
        && !unsafe { pyre_object::bytesobject::is_bytes_like(pattern) }
    {
        return Err(crate::PyError::type_error(
            "first argument must be string, bytes-like object or None",
        ));
    }

    let code_vec = extract_code(roots.get(code_list_slot))?;
    let code_box: &'static [u32] = Box::leak(code_vec.into_boxed_slice());

    Ok(w_sre_pattern_new(
        roots.get(pattern_slot),
        flags,
        code_box,
        groups,
        roots.get(groupindex_slot),
        roots.get(indexgroup_slot),
    ))
}

fn sre_pattern_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 4 {
        return Err(crate::PyError::type_error(
            "SRE_Pattern.__new__() requires subtype, pattern, flags and code",
        ));
    }
    sre_compile(&args[1..])
}

fn extract_code(obj: PyObjectRef) -> Result<Vec<u32>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let n = crate::baseobjspace::len_w(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;
    let idx_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    let mut code = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        pyre_object::gc_roots::shadow_stack_set(idx_slot, w_int_new(i));
        let w_item = crate::baseobjspace::getitem(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            pyre_object::gc_roots::shadow_stack_get(idx_slot),
        )?;
        code.push(crate::baseobjspace::uint_w(w_item)? as u32);
    }
    Ok(code)
}

/// `srepat.code` reader — typecheck guards the raw cast (the methods
/// are reachable through `Pattern.match(non_pattern, ...)`).
fn get_code(pat: PyObjectRef) -> Option<&'static [u32]> {
    if !unsafe { is_sre_pattern(pat) } {
        return None;
    }
    let pat = pat as *const W_SRE_Pattern;
    unsafe {
        if (*pat).code.is_null() {
            return None;
        }
        Some(std::slice::from_raw_parts((*pat).code, (*pat).code_len))
    }
}

/// `_index_to_byte` (`unicodeobject.py`) — the byte offset of a character
/// index, read through the object's cached index storage.  An index at or past
/// the end is the end of the payload, which is the bound `make_ctx` applies to
/// `endpos` (interp_sre.py).
#[inline]
fn char_to_byte(obj: PyObjectRef, char_pos: usize) -> usize {
    let s = unsafe { w_str_get_wtf8(obj) };
    if char_pos >= unsafe { w_str_len(obj) } {
        return s.len();
    }
    unsafe { w_str_index_to_byte(obj, char_pos) }
}

fn byte_slice(b: &'static [u8], start: i64, end: i64) -> Option<&'static [u8]> {
    if start < 0 || end < start {
        return None;
    }
    // `BufMatchContext` slices clamp old match spans to the exporter's current
    // length (`interp_sre.py:slice_w`).  This matters for a one-shot match on
    // a bytearray resized after matching (issue 29444).
    let start = (start as usize).min(b.len());
    let end = (end as usize).min(b.len());
    Some(&b[start..end])
}

/// Clamp `pos`/`endpos` into `[0, len]` with `endpos >= pos` (make_ctx's
/// position fixup, interp_sre.py/272-275).  `len` is the subject
/// length in the engine's position units (characters for `str`, bytes for
/// a bytes-like subject).
fn normalize_bounds(len: usize, pos: i64, endpos: i64) -> (usize, usize) {
    let pos = (pos.max(0) as usize).min(len);
    let mut endpos = (endpos.max(0) as usize).min(len);
    if endpos < pos {
        endpos = pos;
    }
    (pos, endpos)
}

/// The subject of a match: a unicode `str` (the sre-engine driver reports
/// character positions and slices back to `str`) or a bytes-like buffer
/// (byte positions, slices back to `bytes`).  Mirrors make_ctx's
/// Utf8/Str/BufMatchContext split (interp_sre.py).
///
/// A `str` subject is driven over its `Wtf8` backing rather than a `&str`
/// view: the engine walks code points, and a lone surrogate is a code point
/// with no `&str` spelling, so `re.split('\\.', '\ud800')` matches on it the
/// same way it matches on any other character.
#[derive(Clone, Copy)]
enum Subject {
    /// `UnicodeAsciiMatchContext` (interp_sre.py), selected by `is_ascii()`
    /// at interp_sre.py — one byte per code point, so a character position
    /// *is* a byte offset and the engine drives the payload as bytes.
    /// `StrDrive` carries no semantics of its own (it is `count` and cursor
    /// arithmetic; every unicode decision keys on the compiled pattern's
    /// opcode), which is the same property that lets upstream spell this
    /// context as a bare `StrMatchContext` subclass.
    AsciiStr(PyObjectRef),
    /// `Utf8MatchContext` carrying `ctx.w_unicode_obj` (interp_sre.py:248-250)
    /// — the object is kept rather than just its payload because the length
    /// and the character-to-byte conversion are read off it, by [`Utf8Drive`].
    Str(PyObjectRef),
    Bytes(PyObjectRef),
}

impl Subject {
    /// Whether the subject slices back to `str` rather than to `bytes`.
    fn is_unicode(self) -> bool {
        !matches!(self, Subject::Bytes(_))
    }

    /// The subject's length in the engine's position units.
    fn len(self) -> usize {
        match self {
            Subject::AsciiStr(obj) => unsafe { w_str_get_wtf8(obj) }.as_bytes().len(),
            Subject::Bytes(obj) => unsafe { pyre_object::bytesobject::bytes_like_data(obj) }.len(),
            // `_len()` (interp_sre.py) — the stored code point count.
            Subject::Str(obj) => unsafe { w_str_len(obj) },
        }
    }
}

/// The subject for a `str` (or `str` subclass) — `make_ctx`'s `is_ascii()`
/// split (interp_sre.py).
///
/// # Safety
/// `string` must point to a valid `W_UnicodeObject`.
unsafe fn str_subject(string: PyObjectRef) -> Subject {
    if unsafe { w_str_is_ascii(string) } {
        Subject::AsciiStr(string)
    } else {
        Subject::Str(string)
    }
}

/// The drive behind [`Subject::Str`]: a non-ASCII `str` whose positions are
/// code point indices, resolved through the object's stored length and index
/// storage — `_len()` and `_index_to_byte` (unicodeobject.py) —
/// rather than by walking the payload.
///
/// This is the point of carrying `w_unicode_obj` on the context at all.
/// `impl StrDrive for &Wtf8` (`sre_engine` `string.rs:153`) has only the
/// payload, so it answers `count` by counting every code point and
/// `create_cursor(n)` by stepping over the first `n` of them; both run once
/// per match, which makes a scan that restarts at successive positions
/// quadratic in the subject. The index storage answers the same two questions
/// without touching the payload.
///
/// Stepping is left to that impl: it is pure cursor arithmetic, identical for
/// either way of arriving at the cursor.
#[derive(Clone, Copy)]
struct Utf8Drive {
    obj: PyObjectRef,
}

/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
unsafe fn utf8_drive(obj: PyObjectRef) -> Utf8Drive {
    Utf8Drive { obj }
}

impl StrDrive for Utf8Drive {
    #[inline]
    fn count(&self) -> usize {
        unsafe { w_str_len(self.obj) }
    }

    #[inline]
    fn create_cursor(&self, n: usize) -> StringCursor {
        // `n == count()` is the end-of-subject cursor, which has no character
        // to index; every other caller keeps `n` inside the subject.
        // Re-read the payload from the rooted object. Holding the `Wtf8`
        // pointer on the drive would dangle if the string moved.
        let wtf8 = unsafe { w_str_get_wtf8(self.obj) };
        let byte = if n >= unsafe { w_str_len(self.obj) } {
            wtf8.as_bytes().len()
        } else {
            unsafe { w_str_index_to_byte(self.obj, n) }
        };
        // `StringCursor.ptr` is crate-private, so the cursor is minted at the
        // head of the suffix — an O(1) reslice — and carries the code point
        // index that the engine actually compares against.
        let suffix: &Wtf8 = &wtf8[byte..];
        let mut cursor = suffix.create_cursor(0);
        cursor.position = n;
        cursor
    }

    #[inline]
    fn adjust_cursor(&self, cursor: &mut StringCursor, n: usize) {
        // The `&Wtf8` impl only rebuilds when it has to move backwards,
        // because seeking forward costs it a step per code point. Seeking is
        // a lookup here, so a rebuild is never the slower branch, and it also
        // avoids reading the crate-private `ptr` to test it for null.
        *cursor = self.create_cursor(n);
    }

    #[inline]
    fn advance(cursor: &mut StringCursor) -> u32 {
        <&Wtf8 as StrDrive>::advance(cursor)
    }

    #[inline]
    fn peek(cursor: &StringCursor) -> u32 {
        <&Wtf8 as StrDrive>::peek(cursor)
    }

    #[inline]
    fn skip(cursor: &mut StringCursor, n: usize) {
        <&Wtf8 as StrDrive>::skip(cursor, n)
    }

    #[inline]
    fn back_advance(cursor: &mut StringCursor) -> u32 {
        <&Wtf8 as StrDrive>::back_advance(cursor)
    }

    #[inline]
    fn back_peek(cursor: &StringCursor) -> u32 {
        <&Wtf8 as StrDrive>::back_peek(cursor)
    }

    #[inline]
    fn back_skip(cursor: &mut StringCursor, n: usize) {
        <&Wtf8 as StrDrive>::back_skip(cursor, n)
    }
}

/// One translated live GCREF in the shadow stack.  RPython's GC transform
/// inserts the equivalent push/reload around allocations automatically;
/// Rust collections otherwise retain the pre-move pointer while `_sre`
/// builds tuples, lists, and dictionaries of freshly sliced strings.
#[derive(Clone, Copy)]
struct RootedObject(usize);

impl RootedObject {
    fn pin(obj: PyObjectRef) -> Self {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        Self(slot)
    }

    fn get(self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.0)
    }
}

/// `is_known_bytes` (interp_sre.py) — the pattern was compiled from
/// a bytes-like object (a `None` pattern is unknown, accepting either).
fn pattern_is_known_bytes(pat: PyObjectRef) -> bool {
    let w_pattern = unsafe { (*(pat as *const W_SRE_Pattern)).w_pattern };
    !unsafe { is_none(w_pattern) } && !unsafe { is_str(w_pattern) }
}

/// `is_known_unicode` (interp_sre.py).
fn pattern_is_known_unicode(pat: PyObjectRef) -> bool {
    let w_pattern = unsafe { (*(pat as *const W_SRE_Pattern)).w_pattern };
    !unsafe { is_none(w_pattern) } && unsafe { is_str(w_pattern) }
}

/// `readbuf_w` (interp_sre.py:283) — the buffer-providing object backing a
/// read-buffer subject that is not itself `bytes`/`bytearray`.  pyre's only
/// such producer is `memoryview`; its live byte backing plays the captured
/// `BufMatchContext._buffer` (kept alive by the match it is stored into).
/// `PY_NULL` if `obj` is not a memoryview.
unsafe fn readbuf_obj(obj: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::memoryview::is_w_memoryview(obj) } {
        // The subject is the view window (offset/itemsize/length/strides), not
        // the whole backing — materialize the gathered slice bytes.  Any live
        // backing (bytes/bytearray/array) exports a readable byte buffer, so
        // the regex subject is gathered regardless of the backing type.
        let gathered = unsafe { crate::builtins::memoryview_gather_bytes(obj) };
        return pyre_object::bytesobject::w_bytes_from_bytes(&gathered);
    }
    // str / bytes / bytearray subjects are re-resolved directly from `string`
    // (`subject_of`), so they need no captured `_buffer`.
    if unsafe { is_str(obj) } || unsafe { pyre_object::bytesobject::is_bytes_like(obj) } {
        return pyre_object::PY_NULL;
    }
    // `readbuf_w` — any other readable-buffer exporter (e.g. `array.array`):
    // capture its bytes as `ctx._buffer` so slicing re-reads them.
    match unsafe { crate::typedef::buffer_as_bytes_like(obj) } {
        Ok(Some(b)) => pyre_object::bytesobject::w_bytes_from_bytes(unsafe {
            pyre_object::bytesobject::bytes_like_data(b)
        }),
        _ => pyre_object::PY_NULL,
    }
}

/// `make_ctx` (interp_sre.py) — resolve the subject and reject a
/// pattern/subject type mismatch (a bytes pattern on a str, or a str
/// pattern on a bytes-like object).  A `str` (or subclass) becomes a
/// character subject; `bytes`/`bytearray` (or subclass) a byte subject; any
/// other read-buffer object (a `memoryview`) a `BufMatchContext`-style byte
/// subject.
///
/// Returned with the `_buffer` object the subject borrows: a `memoryview` or
/// other exporter is gathered into a fresh `bytes`, and the `Subject::Bytes`
/// slice points into it, so it is the caller — not a second `readbuf_obj`
/// call, which would materialize a *different* object — that roots the one the
/// slice actually reads.  `PY_NULL` where the subject is its own storage.
fn make_subject(
    pat: PyObjectRef,
    string: PyObjectRef,
) -> Result<(Subject, PyObjectRef), crate::PyError> {
    if unsafe { is_str(string) } {
        if pattern_is_known_bytes(pat) {
            return Err(crate::PyError::type_error(
                "cannot use a bytes pattern on a string-like object",
            ));
        }
        Ok((unsafe { str_subject(string) }, pyre_object::PY_NULL))
    } else if unsafe { pyre_object::bytesobject::is_bytes_like(string) } {
        if pattern_is_known_unicode(pat) {
            return Err(crate::PyError::type_error(
                "cannot use a string pattern on a bytes-like object",
            ));
        }
        Ok((Subject::Bytes(string), pyre_object::PY_NULL))
    } else {
        // `make_ctx` acquires any other readable-buffer subject through
        // `readbuf_w` → `buffer_w` (a `memoryview`, or any buffer exporter such
        // as `array.array`).  A released memoryview is rejected before its
        // bytes are read.
        if unsafe { pyre_object::memoryview::is_w_memoryview(string) } {
            unsafe { crate::builtins::memoryview_check_released(string) }?;
        }
        // `is_known_unicode`: prove the subject is buffer-like via `readbuf_w`
        // *before* the pattern/subject-type check — a non-buffer object raises
        // "expected string or bytes-like object", a real buffer (memoryview /
        // array) raises the "string pattern on a bytes-like object" mismatch.
        let w_buffer = unsafe { readbuf_obj(string) };
        if w_buffer.is_null() {
            return Err(crate::PyError::type_error(format!(
                "expected string or bytes-like object, got '{}'",
                crate::baseobjspace::object_functionstr_type_name(string)
            )));
        }
        if pattern_is_known_unicode(pat) {
            return Err(crate::PyError::type_error(
                "cannot use a string pattern on a bytes-like object",
            ));
        }
        Ok((Subject::Bytes(w_buffer), w_buffer))
    }
}

/// Re-resolve the subject of a produced match/scanner for slicing or
/// re-driving — its type was validated by [`make_subject`].  A buffer
/// subject's validated buffer is captured in `w_buffer` (`ctx._buffer`,
/// interp_sre.py:61-64), so slicing reads it directly instead of re-reading
/// the original object; a `str`/`bytes`/`bytearray` subject is the `string`
/// itself (`w_buffer` is `PY_NULL` and unused).
unsafe fn subject_of(string: PyObjectRef, w_buffer: PyObjectRef) -> Subject {
    if unsafe { is_str(string) } {
        unsafe { str_subject(string) }
    } else if unsafe { pyre_object::bytesobject::is_bytes_like(string) } {
        Subject::Bytes(string)
    } else {
        Subject::Bytes(w_buffer)
    }
}

/// `slice_w` (interp_sre.py) — the span sliced out of the subject
/// (`str` → `str`, bytes-like → `bytes`), or `w_default` for an unmatched
/// group (span `(-1, -1)` or otherwise out of range).
fn slice_subject(subj: Subject, span: (i64, i64), w_default: PyObjectRef) -> PyObjectRef {
    let Some(bytes) = subject_span_bytes(subj, span) else {
        return w_default;
    };
    if subj.is_unicode() {
        // interp_sre.py:68/76 uses `space.newutf8`: a captured slice is an
        // ordinary runtime string and must participate in the GC.
        w_str_from_wtf8_managed(unsafe { Wtf8::from_bytes_unchecked(bytes) }.to_owned())
    } else {
        pyre_object::bytesobject::w_bytes_from_bytes(bytes)
    }
}

/// The empty string of the subject's kind — `w_emptystr` for unmatched
/// groups (findall, interp_sre.py:344-347) and empty replacement output.
fn empty_subject(subj: Subject) -> PyObjectRef {
    if subj.is_unicode() {
        w_str_new("")
    } else {
        pyre_object::bytesobject::w_bytes_from_bytes(b"")
    }
}

/// The raw bytes of a span's slice (the WTF-8 encoding for a `str`
/// subject), for building `sub`/`expand` replacement output.
fn subject_span_bytes(subj: Subject, span: (i64, i64)) -> Option<&'static [u8]> {
    let (start, end) = span;
    let (payload, w_unicode_obj) = match subj {
        Subject::Bytes(obj) => {
            return byte_slice(
                unsafe { pyre_object::bytesobject::bytes_like_data(obj) },
                start,
                end,
            );
        }
        // An ASCII span is a byte span already (interp_sre.py:66-68), so the
        // conversion is the identity.
        Subject::AsciiStr(obj) => (unsafe { w_str_get_wtf8(obj) }.as_bytes(), None),
        Subject::Str(obj) => (unsafe { w_str_get_wtf8(obj) }.as_bytes(), Some(obj)),
    };
    // A `str` span out of range is no slice at all, where a buffer span clamps
    // (see [`byte_slice`]).
    if start < 0 || end < start || end as usize > subj.len() {
        return None;
    }
    let to_byte =
        |pos: i64| w_unicode_obj.map_or(pos as usize, |obj| char_to_byte(obj, pos as usize));
    let lo = to_byte(start);
    let hi = to_byte(end);
    Some(&payload[lo..hi])
}

/// Wrap accumulated replacement bytes as the subject's kind — `str` from
/// the (valid UTF-8) builder, or `bytes` (subx result, interp_sre.py).
fn finish_output(subj: Subject, out: Vec<u8>) -> PyObjectRef {
    if subj.is_unicode() {
        // interp_sre.py:567 returns `space.newutf8(...)`: substitution and
        // expansion results are ordinary runtime strings, not bootstrap
        // structural strings that may live outside the managed heap.
        w_str_from_wtf8_managed(Wtf8Buf::from_bytes(out).unwrap_or_default())
    } else {
        pyre_object::bytesobject::w_bytes_from_bytes(&out)
    }
}

/// Drive the engine once over a subject of a concrete [`StrDrive`] type.
fn drive_match<S: StrDrive>(
    drive: S,
    pos: usize,
    endpos: usize,
    code: &[u32],
    search: bool,
    match_all: bool,
) -> (bool, State) {
    let req = Request::new(drive, pos, endpos, code, match_all);
    let mut state = State::default();
    let matched = if search {
        state.search(req)
    } else {
        state.py_match(&req)
    };
    (matched, state)
}

/// Advance a scanner one step over a subject of a concrete [`StrDrive`]
/// type, threading `must_advance` for zero-width matches.
fn drive_scanner_step<S: StrDrive>(
    drive: S,
    pos: usize,
    endpos: usize,
    code: &[u32],
    must_advance: bool,
    anchored: bool,
) -> (bool, State) {
    let mut req = Request::new(drive, pos, endpos, code, false);
    req.must_advance = must_advance;
    let mut state = State::default();
    let found = if anchored {
        state.py_match(&req)
    } else {
        state.search(req)
    };
    (found, state)
}

/// A snapshot of one match for deferred slicing or Match construction — the
/// flattened span table plus `_last_index` (interp_sre.py).
struct MatchSnapshot {
    lastindex: i64,
    spans: Vec<(i64, i64)>,
}

/// Collect a snapshot of every non-overlapping match of a subject of a
/// concrete [`StrDrive`] type — the `SearchIter` walk shared by
/// `findall`/`split`/`sub`.
fn collect_matches<S: StrDrive>(
    drive: S,
    pos: usize,
    endpos: usize,
    code: &[u32],
    pat: PyObjectRef,
) -> Vec<MatchSnapshot> {
    let req = Request::new(drive, pos, endpos, code, false);
    let mut iter = SearchIter {
        req,
        state: State::default(),
    };
    let mut out = Vec::new();
    while iter.next().is_some() {
        let li = iter.state.marks.last_index();
        out.push(MatchSnapshot {
            lastindex: if li >= 0 { li as i64 } else { -1 },
            spans: flatten_spans(pat, &iter.state),
        });
    }
    out
}

/// Drive one `SearchIter` for the whole scan, the way `subx` / `split_w` /
/// `finditer_w` keep a single context from `make_ctx`. `on_match` runs for
/// each non-overlapping match. `cap == 0` is unlimited; a negative `cap`
/// processes none. Returns how many matches were processed.
///
/// `AbstractMatchContext.reset` changes only the start and the empty-match
/// flag. The restarted `Request` keeps the scan's `end`: `pattern_subx` and
/// `pattern_split` hold the subject's buffer export from `state_init` until
/// `state_fini`, so the length cannot change under the scan. `drive` still
/// re-reads the subject after the callback, because a collection can move
/// the string object.
fn stream_matches<S: StrDrive>(
    drive: impl Fn() -> S,
    pos: usize,
    endpos: usize,
    code: &[u32],
    pat: PyObjectRef,
    cap: i64,
    mut on_match: impl FnMut(&MatchSnapshot) -> Result<(), crate::PyError>,
) -> Result<i64, crate::PyError> {
    let mut req = Request::new(drive(), pos, endpos, code, false);
    let mut state = State::default();
    let mut n: i64 = 0;
    while cap == 0 || n < cap {
        let mut iter = SearchIter { req, state };
        if iter.next().is_none() {
            break;
        }
        let li = iter.state.marks.last_index();
        let snap = MatchSnapshot {
            lastindex: if li >= 0 { li as i64 } else { -1 },
            spans: flatten_spans(pat, &iter.state),
        };
        on_match(&snap)?;
        n += 1;
        let SearchIter {
            req: searched,
            state: kept,
        } = iter;
        let string = drive();
        let mut restarted = Request::new(
            string,
            searched.start,
            searched.end,
            searched.pattern_codes,
            searched.match_all,
        );
        // `Request::new` clears `must_advance`. `reset` keeps the flag from
        // the match that just finished.
        restarted.must_advance = searched.must_advance;
        req = restarted;
        state = kept;
    }
    Ok(n)
}

/// The payload [`stream_matches`] drives for an ASCII `str` or a bytes-like
/// subject. A re-read subject keeps its kind: its object is immutable, or is
/// the buffer [`make_subject`] validated.
fn subject_ascii_bytes(subj: Subject) -> &'static [u8] {
    match subj {
        Subject::AsciiStr(obj) => unsafe { w_str_get_wtf8(obj) }.as_bytes(),
        Subject::Bytes(_) | Subject::Str(_) => {
            unreachable!("an ASCII str subject is driven from its string object")
        }
    }
}

fn subject_raw_bytes(subj: Subject) -> &'static [u8] {
    match subj {
        Subject::Bytes(obj) => unsafe { pyre_object::bytesobject::bytes_like_data(obj) },
        Subject::AsciiStr(_) | Subject::Str(_) => {
            unreachable!("a bytes-like subject is driven from its buffer")
        }
    }
}

/// The `W_UnicodeObject` a non-ASCII `str` subject's [`Utf8Drive`] reads.
fn subject_str(subj: Subject) -> PyObjectRef {
    match subj {
        Subject::Str(obj) => obj,
        Subject::AsciiStr(_) | Subject::Bytes(_) => {
            unreachable!("an ASCII str or bytes-like subject is driven as bytes")
        }
    }
}

/// `W_SRE_Match(self, ctx)` (e.g. interp_sre.py) from a collected
/// [`MatchSnapshot`].
fn make_match_from_snapshot(
    pat: PyObjectRef,
    string: PyObjectRef,
    w_buffer: PyObjectRef,
    snap: &MatchSnapshot,
    pos: i64,
    endpos: i64,
) -> PyObjectRef {
    w_sre_match_new(
        pat,
        string,
        w_buffer,
        pos,
        endpos,
        snap.lastindex,
        &snap.spans,
    )
}

fn do_match(
    args: &[PyObjectRef],
    search: bool,
    match_all: bool,
    name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["string", "pos", "endpos"], name)?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "string", "string", args.get(1).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "pos", "pos", args.get(2).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "endpos", "endpos", args.get(3).is_some())?;
    if args.len() > 4 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes at most 3 arguments ({} given)",
            args.len() - 1
        )));
    }
    let Some(string) = args
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "string"))
    else {
        return Err(crate::PyError::type_error("requires self and string"));
    };
    let pat = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{name} requires self and string")))?;
    // A buffer subject is gathered into fresh bytes nothing else holds
    // (`readbuf_obj`), and `subj` borrows that payload, so it has to survive the
    // bound conversions below — `pos` and `endpos` run `__index__`, which is
    // user code, and a sweep in there would free the gathered bytes underneath
    // both the slice and the match this returns.
    //
    // `args` is the gateway stack copy (`index_bounds_not_none`): the first
    // bound's `__index__` collects, so the pattern, the subject, the second
    // bound, and kwargs have to be shadow-stack slots rather than that copy.
    let _roots = pyre_object::gc_roots::push_roots();
    let arg_base = pyre_object::gc_roots::pin_roots(args);
    let args_len = args.len();
    let kw_slot = kwargs.map(|k| {
        let _ = pyre_object::gc_roots::pin_root(k);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let string_slot = pyre_object::gc_roots::pin_roots(&[string]);
    let pat = || pyre_object::gc_roots::shadow_stack_get(arg_base);
    let string = || pyre_object::gc_roots::shadow_stack_get(string_slot);
    let kwargs_now = || kw_slot.map(pyre_object::gc_roots::shadow_stack_get);
    get_code(pat()).ok_or_else(|| crate::PyError::type_error("no compiled code"))?;
    let (subj, w_buffer) = make_subject(pat(), string())?;
    let buffer_slot = pyre_object::gc_roots::pin_roots(&[w_buffer]);
    let w_buffer = || pyre_object::gc_roots::shadow_stack_get(buffer_slot);

    let (pos, endpos) = normalize_bounds(
        subj.len(),
        arg_int_kw_rooted(arg_base, args_len, 2, kwargs_now(), "pos", 0)?,
        arg_int_kw_rooted(arg_base, args_len, 3, kwargs_now(), "endpos", i64::MAX)?,
    );
    // `pos`/`endpos` ran `__index__`. Reload the subject from the
    // rooted objects so a moving collection during that conversion
    // cannot leave `subj` pointing into a reclaimed gathered buffer.
    let subj = unsafe { subject_of(string(), w_buffer()) };
    let code = get_code(pat()).ok_or_else(|| crate::PyError::type_error("no compiled code"))?;

    let (matched, state) = match subj {
        Subject::AsciiStr(obj) => drive_match(
            unsafe { w_str_get_wtf8(obj) }.as_bytes(),
            pos,
            endpos,
            code,
            search,
            match_all,
        ),
        Subject::Bytes(obj) => drive_match(
            unsafe { pyre_object::bytesobject::bytes_like_data(obj) },
            pos,
            endpos,
            code,
            search,
            match_all,
        ),
        Subject::Str(obj) => {
            let s = unsafe { utf8_drive(obj) };
            drive_match(s, pos, endpos, code, search, match_all)
        }
    };

    if matched {
        Ok(make_match(
            pat(),
            string(),
            w_buffer(),
            &state,
            pos as i64,
            endpos as i64,
        ))
    } else {
        Ok(w_none())
    }
}

/// Flatten the engine marks into the span table `do_flatten_marks`
/// (interp_sre.py) would produce, with group 0 (the whole match)
/// prepended.  The table is sized by the pattern's `num_groups` (filled
/// with `(-1, -1)` before copying the marks); the engine only
/// materialises marks up to the last touched group.  Positions are
/// character offsets for a `str` subject and byte offsets for a bytes-like
/// subject — the sre-engine driver's units, which is the external index
/// convention pyre exposes (PyPy stores byte positions internally and
/// converts on exposure for utf8).
fn flatten_spans(pat: PyObjectRef, state: &State) -> Vec<(i64, i64)> {
    let start = state.start;
    let end = state.cursor.position;
    let num_groups = unsafe { (*(pat as *const W_SRE_Pattern)).num_groups }.max(0) as usize;
    let marked_groups = state.marks.raw().len() / 2;
    let mut spans: Vec<(i64, i64)> = vec![(start as i64, end as i64)];
    for gi in 0..num_groups {
        if gi >= marked_groups {
            spans.push((-1, -1));
            continue;
        }
        let (gs, ge) = state.marks.get(gi);
        spans.push(match (gs.into_option(), ge.into_option()) {
            (Some(a), Some(b)) => (a as i64, b as i64),
            _ => (-1, -1),
        });
    }
    spans
}

/// Build the W_SRE_Match for a successful engine run — the
/// `W_SRE_Match(self, ctx)` constructions (e.g. interp_sre.py)
/// with the span table flattened eagerly (`flatten_marks`, :793-797).
fn make_match(
    pat: PyObjectRef,
    string: PyObjectRef,
    w_buffer: PyObjectRef,
    state: &State,
    pos: i64,
    endpos: i64,
) -> PyObjectRef {
    // `_last_index` (interp_sre.py); -1 plays None.
    let lastindex = {
        let li = state.marks.last_index();
        if li >= 0 { li as i64 } else { -1 }
    };
    let spans = flatten_spans(pat, state);
    w_sre_match_new(pat, string, w_buffer, pos, endpos, lastindex, &spans)
}

fn sre_pattern_match(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    do_match(args, false, false, "match")
}
fn sre_pattern_fullmatch(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    do_match(args, false, true, "fullmatch")
}
fn sre_pattern_search(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    do_match(args, true, false, "search")
}

/// Read an optional positional int argument (pos/endpos/count), using
/// `default` only when the argument is absent.  A supplied argument is
/// converted with `__index__` (`@unwrap_spec(pos=int, ...)` →
/// `space.index` + machine-int unwrapping); a non-integer value raises
/// TypeError and an out-of-range value raises OverflowError.  PyPy's
/// `@unwrap_spec(pos=int)` uses `int_w`, not the clamping
/// `getindex_w(..., w_exception=None)` variant.
fn sre_index_int(w: PyObjectRef) -> Result<i64, crate::PyError> {
    let indexed = crate::baseobjspace::space_index(w)?;
    crate::baseobjspace::int_w(indexed)
}

fn arg_int(args: &[PyObjectRef], idx: usize, default: i64) -> Result<i64, crate::PyError> {
    match args.get(idx) {
        Some(&w) if !w.is_null() => sre_index_int(w),
        _ => Ok(default),
    }
}

/// Resolve an optional int argument (`pos`/`endpos`/`count`) that may be
/// supplied positionally or by keyword — the unwrap_spec binding the
/// gateway performs for these builtins (e.g. `match(w_string, pos=0,
/// endpos=sys.maxint)`, interp_sre.py:262).  `pos_args` must already have
/// the trailing `__pyre_kw__` dict stripped ([`split_builtin_kwargs`]).
fn arg_int_kw(
    pos_args: &[PyObjectRef],
    idx: usize,
    kwargs: Option<PyObjectRef>,
    name: &str,
    default: i64,
) -> Result<i64, crate::PyError> {
    if let Some(w) = crate::builtins::kwarg_get(kwargs, name) {
        return sre_index_int(w);
    }
    arg_int(pos_args, idx, default)
}

/// `arg_int_kw` over a rooted gateway copy.  The first bound's `__index__`
/// collects, so the second bound is read back from its shadow-stack slot
/// (`index_bounds_not_none` / `slice_unpack`).
fn arg_int_kw_rooted(
    arg_base: usize,
    args_len: usize,
    idx: usize,
    kwargs: Option<PyObjectRef>,
    name: &str,
    default: i64,
) -> Result<i64, crate::PyError> {
    if let Some(w) = crate::builtins::kwarg_get(kwargs, name) {
        return sre_index_int(w);
    }
    if idx < args_len {
        let w = pyre_object::gc_roots::shadow_stack_get(arg_base + idx);
        if !w.is_null() {
            return sre_index_int(w);
        }
    }
    Ok(default)
}

fn required_arg_kw(
    pos_args: &[PyObjectRef],
    idx: usize,
    kwargs: Option<PyObjectRef>,
    name: &str,
    function: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let positional = pos_args.get(idx).copied();
    let keyword = crate::builtins::kwarg_get(kwargs, name);
    if positional.is_some() && keyword.is_some() {
        return Err(crate::PyError::type_error(format!(
            "{function}() got multiple values for argument '{name}'"
        )));
    }
    positional
        .or(keyword)
        .ok_or_else(|| crate::PyError::type_error(format!("{function} requires self and {name}")))
}

/// `findall_w` (interp_sre.py) — non-overlapping matches.  With no
/// groups the whole match is collected; with one group that group's text;
/// with two or more a tuple of the groups.  Unmatched groups become the
/// empty string (`w_emptystr`, :344-347).
fn sre_pattern_findall(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let pat = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("findall requires self and string"))?;
    let string = required_arg_kw(args, 1, kwargs, "string", "findall")?;
    let arg_base = pyre_object::gc_roots::pin_roots(args);
    let args_len = args.len();
    let kw_slot = kwargs.map(|k| {
        let _ = pyre_object::gc_roots::pin_root(k);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let string_slot = pyre_object::gc_roots::pin_roots(&[string]);
    let pat = || pyre_object::gc_roots::shadow_stack_get(arg_base);
    let string = || pyre_object::gc_roots::shadow_stack_get(string_slot);
    let kwargs_now = || kw_slot.map(pyre_object::gc_roots::shadow_stack_get);
    get_code(pat()).ok_or_else(|| crate::PyError::type_error("no code"))?;
    let (subj, w_buffer) = make_subject(pat(), string())?;
    let buffer_slot = pyre_object::gc_roots::pin_roots(&[w_buffer]);
    let (pos, endpos) = normalize_bounds(
        subj.len(),
        arg_int_kw_rooted(arg_base, args_len, 2, kwargs_now(), "pos", 0)?,
        arg_int_kw_rooted(arg_base, args_len, 3, kwargs_now(), "endpos", i64::MAX)?,
    );
    let pat = pat();
    let subj = unsafe {
        subject_of(
            string(),
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
        )
    };
    let code = get_code(pat).ok_or_else(|| crate::PyError::type_error("no code"))?;
    let num_groups = unsafe { (*(pat as *const W_SRE_Pattern)).num_groups }.max(0) as usize;
    let w_empty = RootedObject::pin(empty_subject(subj));
    // Collecting the matches allocates nothing; each slice below does, so
    // the payload is re-read through the pinned subject objects per slice.
    let subject_now = || unsafe {
        subject_of(
            string(),
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
        )
    };

    let matches = match subject_now() {
        Subject::AsciiStr(obj) => collect_matches(
            unsafe { w_str_get_wtf8(obj) }.as_bytes(),
            pos,
            endpos,
            code,
            pat,
        ),
        Subject::Bytes(obj) => collect_matches(
            unsafe { pyre_object::bytesobject::bytes_like_data(obj) },
            pos,
            endpos,
            code,
            pat,
        ),
        Subject::Str(obj) => {
            let s = unsafe { utf8_drive(obj) };
            collect_matches(s, pos, endpos, code, pat)
        }
    };
    // `matchlist_w = []` in interp_sre.py is a GC-managed list.  Build
    // that same shape here instead of retaining every result in an off-heap
    // Rust Vec (which would require an O(number of matches) shadow stack).
    let results = RootedObject::pin(w_list_new_empty());
    for snap in &matches {
        // Per-match bracket: once `w_list_append` makes the item reachable
        // from `results`, the slots can drop. A function-wide pin of every
        // match is O(matches × groups) and is not `findall_w`.
        let _match_roots = pyre_object::gc_roots::push_roots();
        let spans = &snap.spans;
        let w_item = if num_groups == 0 {
            RootedObject::pin(slice_subject(subject_now(), spans[0], w_empty.get()))
        } else if num_groups == 1 {
            RootedObject::pin(slice_subject(subject_now(), spans[1], w_empty.get()))
        } else {
            let grps: Vec<RootedObject> = (1..=num_groups)
                .map(|g| RootedObject::pin(slice_subject(subject_now(), spans[g], w_empty.get())))
                .collect();
            RootedObject::pin(w_tuple_new(grps.iter().map(|grp| grp.get()).collect()))
        };
        unsafe { w_list_append(results.get(), w_item.get()) };
    }
    Ok(results.get())
}

/// `finditer_w` (interp_sre.py) — returns the lazy
/// `W_SRE_Scanner` that yields a `W_SRE_Match` per non-overlapping match.
fn sre_pattern_finditer(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let pat = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("finditer requires self and string"))?;
    let string = required_arg_kw(args, 1, kwargs, "string", "finditer")?;
    if !unsafe { is_sre_pattern(pat) } {
        return Err(crate::PyError::type_error(
            "descriptor 'finditer' for 're.Pattern'",
        ));
    }
    // Validate the compiled code is present (matches do_match's guard).
    get_code(pat).ok_or_else(|| crate::PyError::type_error("no compiled code"))?;
    // Same window as `do_match`: the gathered bytes are held only here across
    // the `__index__` calls the bound conversions make, and the scanner keeps
    // them as its `_buffer`.  The gateway `args` copy is pinned too — the
    // first bound's `__index__` would otherwise leave the second bound as a
    // from-space word.
    let _roots = pyre_object::gc_roots::push_roots();
    let arg_base = pyre_object::gc_roots::pin_roots(args);
    let args_len = args.len();
    let kw_slot = kwargs.map(|k| {
        let _ = pyre_object::gc_roots::pin_root(k);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let string_slot = pyre_object::gc_roots::pin_roots(&[string]);
    let pat = || pyre_object::gc_roots::shadow_stack_get(arg_base);
    let string = || pyre_object::gc_roots::shadow_stack_get(string_slot);
    let kwargs_now = || kw_slot.map(pyre_object::gc_roots::shadow_stack_get);
    let (subj, w_buffer) = make_subject(pat(), string())?;
    let buffer_slot = pyre_object::gc_roots::pin_roots(&[w_buffer]);
    let w_buffer = || pyre_object::gc_roots::shadow_stack_get(buffer_slot);
    let (pos, endpos) = normalize_bounds(
        subj.len(),
        arg_int_kw_rooted(arg_base, args_len, 2, kwargs_now(), "pos", 0)?,
        arg_int_kw_rooted(arg_base, args_len, 3, kwargs_now(), "endpos", i64::MAX)?,
    );
    let export_active = unsafe { crate::builtins::buffer_export_incref(string()) };
    let scanner = w_sre_scanner_new(
        pat(),
        string(),
        w_buffer(),
        pos as i64,
        endpos as i64,
        export_active,
    );
    if export_active {
        crate::executioncontext::register_finalizer(scanner);
    }
    Ok(scanner)
}

/// `sub_w` (interp_sre.py).
fn sre_pattern_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (w_item, _n) = subx(args)?;
    Ok(w_item)
}

/// `subn_w` (interp_sre.py) — returns `(new_string, count)`.
fn sre_pattern_subn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let (w_item, n) = subx(args)?;
    let w_item = RootedObject::pin(w_item);
    let w_n = RootedObject::pin(w_int_new(n));
    Ok(w_tuple_new(vec![w_item.get(), w_n.get()]))
}

/// One buffer export held across a `subx` or `split_w` scan.
///
/// [3.14-spec] ours: the export is taken before the scan and released when
/// `subx` / `split_w` return, as `pattern_subx` / `pattern_split` hold the
/// buffer from `state_init` (`getstring`) to `state_fini`, including when the
/// replacement callable raises ↔ pypy: `W_SRE_Pattern.make_ctx` acquires the
/// buffer only to reach the live `BytearrayBuffer` and releases the export at
/// once; only `finditer_w` keeps a `buf_view` on its scanner.
/// Observable: a `re.sub` callback that resizes the subject bytearray raises
/// `BufferError` instead of letting the scan read past the shrunk buffer.
/// Evidence: `pyre/extra_tests/parity_tests/sre_sub_shrinks_bytearray.py`
/// (the pypy3 oracle prints `b'aaYY'` there), and lib-python
/// `test_re.test_keep_buffer`, which asserts the same held export for a
/// scan in progress.  The negative hint search over `make_ctx` / `subx` /
/// `split_w` in `pypy/module/_sre/interp_sre.py` finds no `@jit.*`,
/// `_immutable_*` or `rgc.*` hint governing the export's lifetime.
///
/// `slot` is the shadow-stack index of the exporter: a collection can move
/// it, so the release re-reads the slot. `bytes` and `str` acquire nothing
/// (`active` stays false).
struct HeldBufferExport {
    slot: usize,
    active: bool,
}

impl Drop for HeldBufferExport {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            unsafe {
                crate::builtins::buffer_export_decref(pyre_object::gc_roots::shadow_stack_get(
                    self.slot,
                ));
            }
        }
    }
}

/// `subx` (interp_sre.py) — the shared sub/subn body.  `repl` is a
/// callable (invoked per match), a literal template string with backslash
/// references expanded ([`parse_template`]), or a plain literal; `count`
/// caps the number of substitutions (0 = unlimited).
fn subx(args: &[PyObjectRef]) -> Result<(PyObjectRef, i64), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "sub requires self, repl, string",
        ));
    }
    let pat = args[0];
    // The pattern, the replacement, and the subject are published together.
    // Everything below runs Python -- an `__index__` on `count`, the template
    // parse, and the filter once per match -- and a match object stamps the
    // subject into traced fields.  `base` stays the replacement slot so the
    // held export below is the subject at `base + 1`.
    let pat_slot = pyre_object::gc_roots::publish_roots(&[pat]);
    let base = pyre_object::gc_roots::publish_roots(&[args[1], args[2]]);
    pyre_object::gc_roots::normalize_roots(pat_slot, 3);
    let w_repl = || pyre_object::gc_roots::shadow_stack_get(base);
    let string = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let code = get_code(pyre_object::gc_roots::shadow_stack_get(pat_slot))
        .ok_or_else(|| crate::PyError::type_error("no compiled code"))?;
    let (subj, w_buffer_obj) =
        make_subject(pyre_object::gc_roots::shadow_stack_get(pat_slot), string())?;
    let buffer_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_buffer_obj);
    let w_buffer = || pyre_object::gc_roots::shadow_stack_get(buffer_slot);
    // `subj` borrows the payload; the template parse and every match
    // allocate. Re-read it through the pinned subject objects after each.
    let subject_now = || unsafe { subject_of(string(), w_buffer()) };
    let count = arg_int_kw(args, 3, kwargs, "count", 0)?;

    // interp_sre.py:437-472 — a callable filter is applied per match; a
    // literal (no backslash) is inserted verbatim; otherwise the template
    // is compiled into a literal/group reference list.  The replacement's
    // type must match the subject's (str↔str, bytes↔bytes).
    let filter_is_callable = crate::baseobjspace::callable_w(w_repl());
    let template = if filter_is_callable {
        None
    } else {
        let (repl_bytes, is_bytes) = match subj {
            Subject::AsciiStr(_) | Subject::Str(_) => {
                if !unsafe { is_str(w_repl()) } {
                    return Err(crate::PyError::type_error(
                        "sub: replacement must be str or callable",
                    ));
                }
                (unsafe { w_str_get_wtf8(w_repl()) }.as_bytes(), false)
            }
            Subject::Bytes(_) => {
                let Some(w_bytes) = crate::typedef::buffer_as_bytes_like(w_repl())? else {
                    return Err(crate::PyError::type_error(
                        "sub: replacement must be bytes-like or callable",
                    ));
                };
                let w_bytes = pyre_object::gc_roots::pin_root(w_bytes);
                (
                    unsafe { pyre_object::bytesobject::bytes_like_data(w_bytes) },
                    true,
                )
            }
        };
        Some(if repl_bytes.contains(&b'\\') {
            parse_replacement_template(
                w_repl(),
                repl_bytes,
                pyre_object::gc_roots::shadow_stack_get(pat_slot),
                is_bytes,
            )?
        } else {
            // `subx` literal arm: no backslash, so the replacement is copied
            // verbatim and the template compiler is not invoked.
            vec![TemplateItem::Literal(repl_bytes.to_vec())]
        })
    };

    let endpos = subject_now().len();
    let mut out: Vec<u8> = Vec::new();
    let mut last = 0i64;
    // interp_sre.py `while not count or n < count` — 0 is unlimited, a
    // negative count performs no substitutions; the matching itself streams so
    // a callable filter's side effects interleave with `search`.
    let on_match = |snap: &MatchSnapshot| -> Result<(), crate::PyError> {
        let (mstart, mend) = snap.spans[0];
        // interp_sre.py:499-502 — copy the gap before this match.
        if let Some(gap) = subject_span_bytes(subject_now(), (last, mstart)) {
            out.extend_from_slice(gap);
        }
        last = mend;
        if let Some(items) = &template {
            let m = make_match_from_snapshot(
                pyre_object::gc_roots::shadow_stack_get(pat_slot),
                string(),
                w_buffer(),
                snap,
                0,
                endpos as i64,
            );
            expand_into(&mut out, items, m as *const W_SRE_Match, subject_now());
        } else {
            // interp_sre.py:505-513 — callable filter; None means "no
            // piece" (treated as empty), otherwise the returned string.
            let m = make_match_from_snapshot(
                pyre_object::gc_roots::shadow_stack_get(pat_slot),
                string(),
                w_buffer(),
                snap,
                0,
                endpos as i64,
            );
            let w_piece = crate::call::call_function_impl_result(w_repl(), &[m])?;
            if !unsafe { is_none(w_piece) } {
                match subj {
                    Subject::AsciiStr(_) | Subject::Str(_) => {
                        if !unsafe { is_str(w_piece) } {
                            return Err(crate::PyError::type_error(
                                "sub callable must return a string",
                            ));
                        }
                        out.extend_from_slice(unsafe { w_str_get_wtf8(w_piece) }.as_bytes());
                    }
                    Subject::Bytes(_) => {
                        if !unsafe { pyre_object::bytesobject::is_bytes_like(w_piece) } {
                            return Err(crate::PyError::type_error(
                                "sub callable must return bytes",
                            ));
                        }
                        out.extend_from_slice(unsafe {
                            pyre_object::bytesobject::bytes_like_data(w_piece)
                        });
                    }
                }
            }
        }
        Ok(())
    };
    // `state_init` holds the export across the callback loop. The guard
    // releases it on every return, including a callable that raises.
    let export_active = unsafe { crate::builtins::buffer_export_incref(string()) };
    let _held_export = HeldBufferExport {
        slot: base + 1,
        active: export_active,
    };
    let n = match subj {
        Subject::AsciiStr(_) => stream_matches(
            || subject_ascii_bytes(subject_now()),
            0,
            endpos,
            code,
            pyre_object::gc_roots::shadow_stack_get(pat_slot),
            count,
            on_match,
        )?,
        Subject::Bytes(_) => stream_matches(
            || subject_raw_bytes(subject_now()),
            0,
            endpos,
            code,
            pyre_object::gc_roots::shadow_stack_get(pat_slot),
            count,
            on_match,
        )?,
        Subject::Str(_) => stream_matches(
            || unsafe { utf8_drive(subject_str(subject_now())) },
            0,
            endpos,
            code,
            pyre_object::gc_roots::shadow_stack_get(pat_slot),
            count,
            on_match,
        )?,
    };
    // interp_sre.py:478-484 — no substitution was made (no occurrence, or a
    // non-positive `count`): the result is the subject itself.  An exact
    // `str`/`bytes` is returned verbatim (identity preserved); any other type
    // (`str`/`bytes` subclass or buffer input) is normalized to a base-type
    // slice of the whole subject.
    if n == 0 {
        let w_result = if is_exact_str_or_bytes(string()) {
            string()
        } else {
            let w_empty = RootedObject::pin(empty_subject(subj));
            slice_subject(subject_now(), (0, endpos as i64), w_empty.get())
        };
        return Ok((w_result, 0));
    }
    // `subx` appends the gap from the last match to `ctx.end`, the end
    // fixed when the scan started. The held export keeps that length.
    if let Some(tail) = subject_span_bytes(subject_now(), (last, endpos as i64)) {
        out.extend_from_slice(tail);
    }
    Ok((finish_output(subj, out), n))
}

/// `type(w_string) is unicode` / `is bytes` — the exact-type gate of the
/// `subx` n==0 shortcut (interp_sre.py).  A `str`/`bytes` subclass or
/// a buffer object fails this and is normalized to a base-type slice.
fn is_exact_str_or_bytes(w: PyObjectRef) -> bool {
    match crate::typedef::r#type(w) {
        Some(t) => unsafe {
            std::ptr::eq(
                t.as_ptr(),
                pyre_object::get_instantiate(&pyre_object::STR_TYPE),
            ) || std::ptr::eq(
                t.as_ptr(),
                pyre_object::get_instantiate(&pyre_object::BYTES_TYPE),
            )
        },
        None => false,
    }
}

/// `split_w` (interp_sre.py) — split `string` by the
/// non-overlapping matches of the pattern.  The text between matches is
/// emitted as list items; when the pattern has capturing groups, every
/// group's captured text is interleaved (an unmatched group contributes
/// `None`).  Empty matches are split points (3.7+ semantics).  `maxsplit`
/// (0 = unlimited) caps the number of splits; the unsplit remainder is the
/// final item.
fn sre_pattern_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let pat = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("split requires self and string"))?;
    let string = required_arg_kw(args, 1, kwargs, "string", "split")?;
    // The pattern and the subject are published together: the count conversion
    // and the result list both allocate, and the pattern is read again after
    // each of them.
    let pat_slot = pyre_object::gc_roots::publish_roots(&[pat]);
    let string_slot = pyre_object::gc_roots::publish_roots(&[string]);
    pyre_object::gc_roots::normalize_roots(pat_slot, 2);
    let code = get_code(pyre_object::gc_roots::shadow_stack_get(pat_slot))
        .ok_or_else(|| crate::PyError::type_error("no compiled code"))?;
    let (subj, w_buffer) = make_subject(
        pyre_object::gc_roots::shadow_stack_get(pat_slot),
        pyre_object::gc_roots::shadow_stack_get(string_slot),
    )?;
    let buffer_slot = pyre_object::gc_roots::pin_roots(&[w_buffer]);
    // `subj` borrows the payload, and everything below allocates. Re-read it
    // through the pinned subject objects after each allocation.
    let subject_now = || unsafe {
        subject_of(
            pyre_object::gc_roots::shadow_stack_get(string_slot),
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
        )
    };
    let maxsplit = arg_int_kw(args, 2, kwargs, "maxsplit", 0)?;
    let num_groups = unsafe {
        (*(pyre_object::gc_roots::shadow_stack_get(pat_slot) as *const W_SRE_Pattern)).num_groups
    }
    .max(0) as usize;
    let w_empty = RootedObject::pin(empty_subject(subj));

    let endpos = subject_now().len();
    // `splitlist = []` in interp_sre.py is itself the long-lived root;
    // keep only the item currently being appended on the shadow stack.
    let results = RootedObject::pin(w_list_new_empty());
    let mut last = 0i64;
    let append_slice = |span, w_default| {
        let _item_roots = pyre_object::gc_roots::push_roots();
        let w_item = RootedObject::pin(slice_subject(subject_now(), span, w_default));
        unsafe { w_list_append(results.get(), w_item.get()) };
    };
    // interp_sre.py `while not maxsplit or n < maxsplit` — 0 is unlimited,
    // a negative cap performs no splits; matching streams so `maxsplit` bounds
    // the scan rather than discarding already-found matches.
    let on_match = |snap: &MatchSnapshot| -> Result<(), crate::PyError> {
        let (mstart, mend) = snap.spans[0];
        // interp_sre.py:393 — the slice preceding this match.
        append_slice((last, mstart), w_empty.get());
        // interp_sre.py split_w — interleave each group's capture; an
        // unmatched group span `(-1, -1)` becomes None via slice_subject.
        for g in 1..=num_groups {
            append_slice(snap.spans[g], w_none());
        }
        last = mend;
        Ok(())
    };
    let export_active = unsafe {
        crate::builtins::buffer_export_incref(pyre_object::gc_roots::shadow_stack_get(string_slot))
    };
    let _held_export = HeldBufferExport {
        slot: string_slot,
        active: export_active,
    };
    match subj {
        Subject::AsciiStr(_) => stream_matches(
            || subject_ascii_bytes(subject_now()),
            0,
            endpos,
            code,
            pyre_object::gc_roots::shadow_stack_get(pat_slot),
            maxsplit,
            on_match,
        )?,
        Subject::Bytes(_) => stream_matches(
            || subject_raw_bytes(subject_now()),
            0,
            endpos,
            code,
            pyre_object::gc_roots::shadow_stack_get(pat_slot),
            maxsplit,
            on_match,
        )?,
        Subject::Str(_) => stream_matches(
            || unsafe { utf8_drive(subject_str(subject_now())) },
            0,
            endpos,
            code,
            pyre_object::gc_roots::shadow_stack_get(pat_slot),
            maxsplit,
            on_match,
        )?,
    };
    // `split_w` appends `slice_w(ctx, last, ctx.end)`. `ctx.end` is the
    // end captured for this scan; the held export keeps that length.
    append_slice((last, endpos as i64), w_empty.get());
    Ok(results.get())
}

// ── Replacement-template parser (`re._parser.parse_template`) ──────────

/// A parsed replacement-template element: either a literal run or a
/// reference to group `n` (0 = whole match) — `parse_template`'s result
/// list (`re/_parser.py`).
enum TemplateItem {
    /// A literal run, stored as raw bytes (UTF-8 for a `str` template).
    Literal(Vec<u8>),
    Group(usize),
}

/// Convert `re._parser.parse_template`'s flat result list — a sequence of
/// literal `str`/`bytes` runs interleaved with integer group references — into
/// the [`TemplateItem`] stream `expand_into` consumes.
fn template_items_from_list(w_result: PyObjectRef) -> Result<Vec<TemplateItem>, crate::PyError> {
    // `parse_template` is reached by name through `re._parser`, so what comes
    // back is whatever the program left there. `w_list_len` reads the list
    // header without checking, which a tuple answers with a wild length.
    // `_sre.template`'s own argument check is where this is refused upstream,
    // and its wording is what a program that reaches here sees.
    if !unsafe { pyre_object::is_list(w_result) } {
        return Err(crate::PyError::type_error(format!(
            "template() argument 2 must be list, not {}",
            crate::type_methods::clinic_arg_type_name(w_result)
        )));
    }
    let n = unsafe { pyre_object::w_list_len(w_result) };
    let mut items: Vec<TemplateItem> = Vec::with_capacity(n);
    for i in 0..n {
        let Some(elem) = (unsafe { pyre_object::w_list_getitem(w_result, i as i64) }) else {
            continue;
        };
        if unsafe { is_int(elem) } {
            items.push(TemplateItem::Group(
                unsafe { pyre_object::w_int_get_value(elem) } as usize,
            ));
        } else if unsafe { is_str(elem) } {
            items.push(TemplateItem::Literal(
                unsafe { w_str_get_wtf8(elem) }.as_bytes().to_vec(),
            ));
        } else if unsafe { pyre_object::bytesobject::is_bytes_like(elem) } {
            items.push(TemplateItem::Literal(
                unsafe { pyre_object::bytesobject::bytes_like_data(elem) }.to_vec(),
            ));
        }
    }
    Ok(items)
}

/// `PyLong_AsSsize_t`: an int (including a subclass and a bool) or a long
/// that fits in `ssize_t`. Anything else is `TypeError`, and a value that
/// does not fit is `OverflowError`.
fn template_group_index(elem: PyObjectRef) -> Result<i64, crate::PyError> {
    if elem.is_null() || !unsafe { pyre_object::is_int_or_long(elem) } {
        return Err(crate::PyError::type_error("an integer is required"));
    }
    if unsafe { pyre_object::is_bool(elem) } {
        return Ok(i64::from(unsafe {
            pyre_object::boolobject::w_bool_get_value(elem)
        }));
    }
    if unsafe { pyre_object::is_long(elem) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(elem) };
        if pyre_object::longobject::jit_bigint_to_i64_fits(big) == 0 {
            return Err(crate::PyError::overflow_error(
                "Python int too large to convert to C ssize_t",
            ));
        }
        return Ok(pyre_object::longobject::jit_bigint_to_i64_value(big));
    }
    Ok(unsafe { pyre_object::w_int_get_value(elem) })
}

/// Empty `str` / `bytes` literals are dropped from a group slot
/// (`_sre_template_impl`). The leading literal is kept either way.
fn template_literal_is_empty(elem: PyObjectRef) -> bool {
    unsafe {
        if is_str(elem) {
            return pyre_object::unicodeobject::w_str_len(elem) == 0;
        }
        if pyre_object::is_bytes(elem) {
            return pyre_object::bytesobject::w_bytes_len(elem) == 0;
        }
        false
    }
}

/// `sre.c` `_sre_template_impl` — turn `parse_template`'s flat list into a
/// `_sre.SRE_Template`. The list is interleaved literal / group-index /
/// literal, odd length. A list subclass is accepted (`PyList_Check`).
fn sre_template(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_list = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let _roots = pyre_object::gc_roots::push_roots();
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_list);
    let w_list = || pyre_object::gc_roots::shadow_stack_get(list_slot);
    if !unsafe { crate::baseobjspace::isinstance_list_w(w_list()) } {
        return Err(crate::PyError::type_error(format!(
            "template() argument 2 must be list, not {}",
            crate::baseobjspace::object_functionstr_type_name(w_list())
        )));
    }
    let n = unsafe { pyre_object::w_list_len(w_list()) };
    if n < 1 || n % 2 == 0 {
        return Err(crate::PyError::type_error("invalid template"));
    }
    let n_pairs = n / 2;
    let Some(literal) = (unsafe { pyre_object::w_list_getitem(w_list(), 0) }) else {
        return Err(crate::PyError::type_error("invalid template"));
    };
    let literal_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(literal);
    // `self->chunks = 1 + 2*n`, then one less for each skipped empty literal.
    let mut chunks = 1 + 2 * n_pairs as i64;
    let mut index_slots: Vec<usize> = Vec::with_capacity(n_pairs);
    let mut literal_slots: Vec<usize> = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        let Some(index_obj) =
            (unsafe { pyre_object::w_list_getitem(w_list(), (2 * i + 1) as i64) })
        else {
            return Err(crate::PyError::type_error("invalid template"));
        };
        let index = template_group_index(index_obj)?;
        if index < 0 {
            return Err(crate::PyError::type_error("invalid template"));
        }
        let Some(lit) = (unsafe { pyre_object::w_list_getitem(w_list(), (2 * i + 2) as i64) })
        else {
            return Err(crate::PyError::type_error("invalid template"));
        };
        let stored = if template_literal_is_empty(lit) {
            chunks -= 1;
            w_none()
        } else {
            lit
        };
        // The list root keeps the literal alive, but this copy is what the
        // template stores. Pin it before `w_int_new` can collect.
        let literal_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(stored);
        literal_slots.push(literal_slot);
        let index_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(index));
        index_slots.push(index_slot);
    }
    let indexes = w_tuple_new(
        index_slots
            .iter()
            .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot))
            .collect(),
    );
    let indexes_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(indexes);
    let literals = w_tuple_new(
        literal_slots
            .iter()
            .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot))
            .collect(),
    );
    let literals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(literals);
    Ok(w_sre_template_new(
        pyre_object::gc_roots::shadow_stack_get(literal_slot),
        chunks,
        pyre_object::gc_roots::shadow_stack_get(indexes_slot),
        pyre_object::gc_roots::shadow_stack_get(literals_slot),
    ))
}

/// Parse a replacement template into [`TemplateItem`]s by calling
/// `re._compile_template` (`sre.c` `compile_template`).  The stdlib function
/// is `lru_cache`d, so a repeated `(pattern, repl)` pays for `parse_template`
/// once and then reuses the `_sre.template` object.
fn parse_replacement_template(
    w_template: PyObjectRef,
    template_bytes: &[u8],
    pat: PyObjectRef,
    is_bytes: bool,
) -> Result<Vec<TemplateItem>, crate::PyError> {
    // The import and the `getattr` below collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_template = RootedObject::pin(w_template);
    let pat = RootedObject::pin(pat);
    let w_re = match crate::importing::get_sys_module("re") {
        Some(w_re) => w_re,
        None => crate::importing::importhook(
            rustpython_wtf8::Wtf8::new("re"),
            pyre_object::w_none(),
            pyre_object::w_none(),
            0,
            crate::call::getexecutioncontext(),
        )?,
    };
    let w_compile = crate::baseobjspace::getattr_str(w_re, "_compile_template")?;
    // A buffer template (e.g. `bytearray`) is not a cache key; compile the
    // exact `bytes` copy.  `_compile_template` still parses through
    // `_parser.parse_template` on a miss.
    let w_source = if is_bytes && !unsafe { pyre_object::is_bytes(w_template.get()) } {
        pyre_object::bytesobject::w_bytes_from_bytes(template_bytes)
    } else {
        w_template.get()
    };
    let w_result = crate::call::call_function_impl_result(w_compile, &[pat.get(), w_source])?;
    template_items_from_compiled(w_result)
}

/// Decode the `_sre.SRE_Template` `_sre.template` returned, or a list left
/// behind by a replaced compiler. Any other type is the `compile_template`
/// refusal.
fn template_items_from_compiled(
    w_result: PyObjectRef,
) -> Result<Vec<TemplateItem>, crate::PyError> {
    if unsafe { is_sre_template(w_result) } {
        return template_items_from_sre_template(w_result);
    }
    if unsafe { pyre_object::is_list(w_result) } {
        return template_items_from_list(w_result);
    }
    let name = crate::baseobjspace::object_functionstr_type_name(w_result);
    Err(crate::PyError::runtime_error(format!(
        "the result of compiling a replacement string is {name}"
    )))
}

fn template_items_from_sre_template(
    w_result: PyObjectRef,
) -> Result<Vec<TemplateItem>, crate::PyError> {
    let template = w_result as *const W_SRE_Template;
    let mut items = Vec::new();
    push_template_elem(&mut items, unsafe { (*template).literal });
    let indexes = unsafe { (*template).indexes };
    let literals = unsafe { (*template).literals };
    let n = unsafe { pyre_object::w_tuple_len(indexes) };
    for i in 0..n {
        let Some(index) = (unsafe { pyre_object::w_tuple_getitem(indexes, i as i64) }) else {
            continue;
        };
        push_template_elem(&mut items, index);
        let Some(literal) = (unsafe { pyre_object::w_tuple_getitem(literals, i as i64) }) else {
            continue;
        };
        if !unsafe { is_none(literal) } {
            push_template_elem(&mut items, literal);
        }
    }
    Ok(items)
}

fn push_template_elem(items: &mut Vec<TemplateItem>, elem: PyObjectRef) {
    if unsafe { is_int(elem) } {
        items.push(TemplateItem::Group(
            unsafe { pyre_object::w_int_get_value(elem) } as usize,
        ));
    } else if unsafe { is_str(elem) } {
        items.push(TemplateItem::Literal(
            unsafe { w_str_get_wtf8(elem) }.as_bytes().to_vec(),
        ));
    } else if unsafe { pyre_object::bytesobject::is_bytes_like(elem) } {
        items.push(TemplateItem::Literal(
            unsafe { pyre_object::bytesobject::bytes_like_data(elem) }.to_vec(),
        ));
    }
}

/// Expand the parsed template against a match, appending into `out` — the
/// `_sre.template().expand()` substitution.  An unmatched group reference
/// contributes the empty string (`g(group) or empty`).
fn expand_into(out: &mut Vec<u8>, items: &[TemplateItem], m: *const W_SRE_Match, subj: Subject) {
    for item in items {
        match item {
            TemplateItem::Literal(lit) => out.extend_from_slice(lit),
            TemplateItem::Group(idx) => {
                let span =
                    unsafe { w_sre_match_get_span(m as PyObjectRef, *idx) }.unwrap_or((-1, -1));
                if let Some(piece) = subject_span_bytes(subj, span) {
                    out.extend_from_slice(piece);
                }
            }
        }
    }
}

/// `args[0]` as the typed match receiver for the methods above (bound
/// through the typedef so `args[0]` is the match itself).
fn sre_match_self(args: &[PyObjectRef]) -> Result<*const W_SRE_Match, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if unsafe { is_sre_match(self_) } {
        Ok(self_ as *const W_SRE_Match)
    } else {
        Err(crate::PyError::type_error("descriptor is for 're.Match'"))
    }
}

/// `do_span` (interp_sre.py): resolve a group argument — an
/// index, or a name looked up in `srepat.w_groupindex` — to its span.
/// Unknown groups raise IndexError("no such group").
fn do_span(
    m: *const W_SRE_Match,
    w_arg: Option<PyObjectRef>,
) -> Result<(i64, i64), crate::PyError> {
    let groupnum: i64 = match w_arg {
        None => 0,
        Some(w_arg) => {
            // `match_getindex` — `PyIndex_Check(index)` gate: an operand whose
            // type defines `__index__` (any int subclass, or a duck-typed
            // integer) is the group number, and any error from that conversion
            // propagates.  Only an operand without an `__index__` slot is
            // treated as a group *name*, looked up in `srepat.w_groupindex`
            // (a miss → IndexError "no such group").
            let has_index = unsafe { pyre_object::pyobject::is_int_or_long(w_arg) }
                || unsafe { crate::baseobjspace::lookup(w_arg, "__index__") }.is_some();
            if has_index {
                crate::baseobjspace::getindex_w(w_arg)?
            } else {
                let w_groupindex = unsafe { (*(*m).w_srepat.cast::<W_SRE_Pattern>()).w_groupindex };
                let found = if unsafe { is_dict(w_groupindex) } {
                    unsafe { pyre_object::w_dict_lookup(w_groupindex, w_arg) }
                } else {
                    None
                };
                match found {
                    Some(w_groupnum) => unsafe { w_int_get_value(w_groupnum) },
                    None => return Err(crate::PyError::index_error("no such group")),
                }
            }
        }
    };
    if groupnum < 0 {
        return Err(crate::PyError::index_error("no such group"));
    }
    unsafe { w_sre_match_get_span(m as PyObjectRef, groupnum as usize) }
        .ok_or_else(|| crate::PyError::index_error("no such group"))
}

/// `slice_w` (interp_sre.py): the span sliced out of the subject
/// string, or `w_default` for an unmatched group.
unsafe fn slice_w(m: *const W_SRE_Match, span: (i64, i64), w_default: PyObjectRef) -> PyObjectRef {
    let subj = unsafe { subject_of((*m).w_string, (*m).w_buffer) };
    slice_subject(subj, span, w_default)
}

/// `group_w` (interp_sre.py).
fn sre_match_group(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    let group_args = &args[1..];
    if group_args.len() <= 1 {
        let span = do_span(m, group_args.first().copied())?;
        return Ok(unsafe { slice_w(m, span, w_none()) });
    }
    let _roots = pyre_object::gc_roots::push_roots();
    // Publish the match and every selector as one live set before performing
    // any forwarding query.  Besides matching RPython's `args_w` liveness,
    // this avoids a foreign collection entering between sequential pins while
    // a later dynamically-created group name is still unpublished.
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let m = RootedObject(args_base);
    // RPython's GC transform keeps every entry in `args_w` live across each
    // `slice_w` allocation.  The gateway's native argument copy is not a GC
    // root, so read selectors back from that live set after every allocation.
    let mut results: Vec<RootedObject> = Vec::with_capacity(group_args.len());
    for i in 0..group_args.len() {
        let w_arg = pyre_object::gc_roots::shadow_stack_get(args_base + 1 + i);
        let span = do_span(m.get() as *const W_SRE_Match, Some(w_arg))?;
        results.push(RootedObject::pin(unsafe {
            slice_w(m.get() as *const W_SRE_Match, span, w_none())
        }));
    }
    Ok(w_tuple_new(
        results.into_iter().map(RootedObject::get).collect(),
    ))
}

/// `groups_w` (interp_sre.py) — pyre reads the flattened span
/// table directly; unmatched groups (span `(-1, -1)`) take the optional
/// `default` argument.
fn sre_match_groups(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let m = RootedObject::pin(sre_match_self(args)? as PyObjectRef);
    let w_default = RootedObject::pin(args.get(1).copied().unwrap_or_else(w_none));
    let n = unsafe { (*(m.get() as *const W_SRE_Match)).spans_len };
    let mut groups: Vec<RootedObject> = Vec::new();
    for gi in 1..n {
        let span = unsafe { w_sre_match_get_span(m.get(), gi) }.unwrap_or((-1, -1));
        groups.push(RootedObject::pin(unsafe {
            slice_w(m.get() as *const W_SRE_Match, span, w_default.get())
        }));
    }
    Ok(w_tuple_new(
        groups.into_iter().map(RootedObject::get).collect(),
    ))
}

/// `groupdict_w` (interp_sre.py) — name→group-text map built by
/// iterating `srepat.w_groupindex`; unmatched groups take `default`.
fn sre_match_groupdict(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let m = RootedObject::pin(sre_match_self(args)? as PyObjectRef);
    let w_default = RootedObject::pin(args.get(1).copied().unwrap_or_else(w_none));
    let w_groupindex = RootedObject::pin(unsafe {
        (*(*(m.get() as *const W_SRE_Match))
            .w_srepat
            .cast::<W_SRE_Pattern>())
        .w_groupindex
    });
    let w_dict = RootedObject::pin(w_dict_new());
    // interp_sre.py groupdict_w — walk `w_groupindex` through the
    // object-space iterator / item protocol, resolving each value with
    // `do_span` so a duck-typed group number works.
    let w_iterator = RootedObject::pin(crate::baseobjspace::iter(w_groupindex.get())?);
    loop {
        let _item_roots = pyre_object::gc_roots::push_roots();
        let w_key = match crate::baseobjspace::next(w_iterator.get()) {
            Ok(k) => RootedObject::pin(k),
            Err(e) => {
                let (stop, e) = e.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(e);
            }
        };
        let w_value = RootedObject::pin(crate::baseobjspace::getitem(
            w_groupindex.get(),
            w_key.get(),
        )?);
        let span = do_span(m.get() as *const W_SRE_Match, Some(w_value.get()))?;
        let w_grp = RootedObject::pin(unsafe {
            slice_w(m.get() as *const W_SRE_Match, span, w_default.get())
        });
        crate::baseobjspace::setitem(w_dict.get(), w_key.get(), w_grp.get())?;
    }
    Ok(w_dict.get())
}

/// `new_charindex_tuple` (interp_sre.py) — a `(start, end)` pair.
/// Spans stored on the match are already character indexes.
fn new_charindex_tuple(start: i64, end: i64) -> PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_int_new(start));
    fields.push(w_int_new(end));
    w_tuple_new(fields.take())
}

/// `fget_regs` (interp_sre.py) — `((start, end), ...)` for group
/// 0..num_groups, matching what `span(i)` reports; unmatched is `(-1, -1)`.
fn sre_match_regs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_receiver(args)?;
    let n = unsafe { (*m).spans_len };
    let mut result_w = pyre_object::gc_roots::RootedItems::new();
    for gi in 0..n {
        let (start, end) =
            unsafe { w_sre_match_get_span(m as PyObjectRef, gi) }.unwrap_or((-1, -1));
        result_w.push(new_charindex_tuple(start, end));
    }
    Ok(w_tuple_new(result_w.take()))
}

/// `start_w` (interp_sre.py).
fn sre_match_start(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    let (start, _) = do_span(m, args.get(1).copied())?;
    Ok(w_int_new(start))
}

/// `end_w` (interp_sre.py).
fn sre_match_end(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    let (_, end) = do_span(m, args.get(1).copied())?;
    Ok(w_int_new(end))
}

/// `span_w` (interp_sre.py).
fn sre_match_span(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    let (start, end) = do_span(m, args.get(1).copied())?;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_int_new(start));
    fields.push(w_int_new(end));
    Ok(w_tuple_new(fields.take()))
}

/// `descr_getitem` (interp_sre.py) — `m[index]` resolves the
/// single group's span and slices the subject string.
fn sre_match_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    let span = do_span(m, args.get(1).copied())?;
    Ok(unsafe { slice_w(m, span, w_none()) })
}

/// `expand_w` (interp_sre.py) — substitute the parsed template
/// against this match.  Upstream delegates to `re._expand`; pyre expands
/// natively since it owns the template parser.
fn sre_match_expand(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let m = RootedObject::pin(sre_match_self(args)? as PyObjectRef);
    let w_template = args.get(1).copied().unwrap_or_else(w_none);
    let subj = unsafe {
        let mp = m.get() as *const W_SRE_Match;
        subject_of((*mp).w_string, (*mp).w_buffer)
    };
    let (template, is_bytes): (&[u8], bool) = match subj {
        Subject::AsciiStr(_) | Subject::Str(_) => {
            if !unsafe { is_str(w_template) } {
                return Err(crate::PyError::type_error("expand: template must be str"));
            }
            (unsafe { w_str_get_wtf8(w_template) }.as_bytes(), false)
        }
        Subject::Bytes(_) => {
            if !unsafe { pyre_object::bytesobject::is_bytes_like(w_template) } {
                return Err(crate::PyError::type_error("expand: template must be bytes"));
            }
            (
                unsafe { pyre_object::bytesobject::bytes_like_data(w_template) },
                true,
            )
        }
    };
    let w_pat = unsafe { (*(m.get() as *const W_SRE_Match)).w_srepat };
    let items = parse_replacement_template(w_template, template, w_pat, is_bytes)?;
    // The parse runs Python: re-read the match and its subject.
    let mp = m.get() as *const W_SRE_Match;
    let subj = unsafe { subject_of((*mp).w_string, (*mp).w_buffer) };
    let mut out: Vec<u8> = Vec::new();
    expand_into(&mut out, &items, mp, subj);
    Ok(finish_output(subj, out))
}

/// `repr_w` (interp_sre.py) — `<re.Match object; span=(s, e),
/// match=R>` with `R` the repr of the whole match truncated to 50
/// characters.  Positions are character offsets for a `str` subject and
/// byte offsets for a bytes-like subject (the sre-engine driver's units).
pub(crate) fn sre_match_repr_str(
    m: PyObjectRef,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let m = RootedObject::pin(m);
    let mp = m.get() as *const W_SRE_Match;
    let span = unsafe { w_sre_match_get_span(m.get(), 0) }.unwrap_or((-1, -1));
    let (start, end) = span;
    let subj = unsafe { subject_of((*mp).w_string, (*mp).w_buffer) };
    let w_match_str = RootedObject::pin(slice_subject(subj, span, w_none()));
    let matchrepr = truncate_code_points(
        unsafe { crate::display::py_repr_wtf8(w_match_str.get()) }?,
        50,
    );
    Ok(crate::display::wtf8_format!(
        format!("<re.Match object; span=({start}, {end}), match="),
        matchrepr,
        ">",
    ))
}

/// The first `limit` code points of `text`, for the repr truncations that
/// `str[:limit]` performs on a subject that may hold a lone surrogate
/// (`interp_sre.py W_SRE_Pattern.repr_w` `space.getslice(..., 0, 200)`).
fn truncate_code_points(text: rustpython_wtf8::Wtf8Buf, limit: usize) -> rustpython_wtf8::Wtf8Buf {
    let src: &rustpython_wtf8::Wtf8 = &text;
    let nbytes = src.as_bytes().len();
    let mut pos = 0usize;
    let mut n = 0usize;
    while n < limit && pos < nbytes {
        pos = pyre_object::rutf8::next_codepoint_pos(src, pos);
        n += 1;
    }
    if pos >= nbytes {
        return text;
    }
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_wtf8(unsafe { rustpython_wtf8::Wtf8::from_bytes_unchecked(&src.as_bytes()[..pos]) });
    out
}

fn sre_match_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    Ok(pyre_object::w_str_from_wtf8_managed(sre_match_repr_str(
        m as PyObjectRef,
    )?))
}

/// `copy_identity_w` (interp_sre.py) — match results are
/// immutable, so `__copy__`/`__deepcopy__` return self.
fn sre_match_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let m = sre_match_self(args)?;
    Ok(m as PyObjectRef)
}

/// `args[0]` as the typed pattern receiver for pattern methods bound
/// through the typedef (so `args[0]` is the pattern itself).
fn sre_pattern_self(args: &[PyObjectRef]) -> Result<*const W_SRE_Pattern, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if unsafe { is_sre_pattern(self_) } {
        Ok(self_ as *const W_SRE_Pattern)
    } else {
        Err(crate::PyError::type_error("descriptor is for 're.Pattern'"))
    }
}

// SRE flag bits (`sre_constants.py`) consulted by the pattern repr.
const SRE_FLAG_LOCALE: i64 = 4;
const SRE_FLAG_UNICODE: i64 = 32;
const SRE_FLAG_ASCII: i64 = 256;

/// interp_sre.py `FLAG_NAMES`.
const SRE_FLAG_NAMES: [&str; 9] = [
    "re.TEMPLATE",
    "re.IGNORECASE",
    "re.LOCALE",
    "re.MULTILINE",
    "re.DOTALL",
    "re.UNICODE",
    "re.VERBOSE",
    "re.DEBUG",
    "re.ASCII",
];

/// `repr_w` (interp_sre.py) — `re.compile(<pattern repr>, <flags>)`
/// with the pattern repr truncated to 200 characters and the flag bits
/// decoded into their `re.*` names (the implicit `re.UNICODE` on a known
/// unicode pattern is suppressed).
pub(crate) fn sre_pattern_repr_str(
    mut pat: PyObjectRef,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    let mut w_pattern = unsafe { (*(pat as *const W_SRE_Pattern)).w_pattern };
    let u = truncate_code_points(
        pyre_object::with_roots!(pat, w_pattern => unsafe { crate::display::py_repr_wtf8(w_pattern) })?,
        200,
    );

    let flags = unsafe { (*(pat as *const W_SRE_Pattern)).flags };
    let is_known_unicode = unsafe { is_str(w_pattern) };
    let tail = sre_pattern_flag_tail(flags, is_known_unicode);
    Ok(crate::display::wtf8_format!("re.compile(", u, tail))
}

/// interp_sre.py `W_SRE_Pattern.repr_w` flag-name join.
fn sre_pattern_flag_tail(mut flags: i64, is_known_unicode: bool) -> String {
    if is_known_unicode
        && (flags & (SRE_FLAG_LOCALE | SRE_FLAG_UNICODE | SRE_FLAG_ASCII)) == SRE_FLAG_UNICODE
    {
        flags &= !SRE_FLAG_UNICODE;
    }
    let mut flag_items: Vec<String> = Vec::new();
    // interp_sre.py `W_SRE_Pattern.repr_w`: `for i, name in enumerate(FLAG_NAMES)`.
    for i in 0..SRE_FLAG_NAMES.len() {
        let name = SRE_FLAG_NAMES[i];
        if flags & (1 << i) != 0 {
            flags -= 1 << i;
            flag_items.push(name.to_string());
        }
    }
    if flags != 0 {
        flag_items.push(format!("0x{flags:x}"));
    }
    if flag_items.is_empty() {
        ")".to_owned()
    } else {
        format!(", {})", flag_items.join("|"))
    }
}

fn sre_pattern_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let pat = sre_pattern_self(args)?;
    Ok(pyre_object::w_str_from_wtf8_managed(sre_pattern_repr_str(
        pat as PyObjectRef,
    )?))
}

/// `descr_eq` (interp_sre.py): compare flags, compiled code, and
/// original pattern; groupindex/indexgroup are derived from the pattern.
fn sre_pattern_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let pat = sre_pattern_self(args)?;
    let other = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if !unsafe { is_sre_pattern(other) } {
        return Ok(pyre_object::w_not_implemented());
    }
    let p = unsafe { &*pat };
    let q = unsafe { &*(other as *const W_SRE_Pattern) };
    let p_code = get_code(pat as PyObjectRef).unwrap_or(&[]);
    let q_code = get_code(other).unwrap_or(&[]);
    Ok(w_bool_from(
        p.flags == q.flags
            && p_code == q_code
            && crate::baseobjspace::eq_w(p.w_pattern, q.w_pattern)?,
    ))
}

/// `descr_hash` (interp_sre.py): hash the compiled code, flags,
/// and original pattern in the same structural order as PyPy.
fn sre_pattern_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let pat = sre_pattern_self(args)?;
    let p = unsafe { &*pat };
    let mut x: i64 = 0x345678;
    for &c in get_code(pat as PyObjectRef).unwrap_or(&[]) {
        x = x.wrapping_mul(1_000_003) ^ c as i64;
    }
    x = x.wrapping_mul(1_000_003) ^ p.flags;
    x = x.wrapping_mul(1_000_003) ^ crate::baseobjspace::hash_w_strict(p.w_pattern)?;
    Ok(w_int_new(x))
}

/// `copy_identity_w` (interp_sre.py) — compiled patterns are
/// immutable, so `__copy__`/`__deepcopy__` return self.
fn sre_pattern_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let pat = sre_pattern_self(args)?;
    Ok(pat as PyObjectRef)
}

// ── SRE_Scanner (finditer iterator) ───────────────────────────────────

/// `args[0]` as the typed scanner receiver.
fn sre_scanner_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if unsafe { is_sre_scanner(self_) } {
        Ok(self_)
    } else {
        Err(crate::PyError::type_error(
            "descriptor is for '_sre.SRE_Scanner'",
        ))
    }
}

/// `W_SRE_Scanner.getmatch` (interp_sre.py) — advance the scanner
/// one step and build the match.  `anchored` selects `py_match` (the
/// `match` method) over `search` (`__next__` / `search`).  Returns
/// `Ok(None)` and marks the scanner exhausted (`self.ctx = None`) when no
/// further match is found.
fn sre_scanner_step(
    obj: PyObjectRef,
    anchored: bool,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let sc = obj as *mut W_SRE_Scanner;
    let pos = unsafe { (*sc).pos };
    if pos < 0 {
        return Ok(None); // self.ctx is None
    }
    let w_srepat = unsafe { (*sc).w_srepat };
    let string = unsafe { (*sc).w_string };
    let w_buffer = unsafe { (*sc).w_buffer };
    let original_pos = unsafe { (*sc).original_pos };
    let endpos = unsafe { (*sc).endpos }.max(0) as usize;
    let subj = unsafe { subject_of(string, w_buffer) };
    let code = get_code(w_srepat).ok_or_else(|| crate::PyError::type_error("no compiled code"))?;
    let must_advance = unsafe { (*sc).must_advance } != 0;

    let (found, state) = match subj {
        Subject::AsciiStr(obj) => drive_scanner_step(
            unsafe { w_str_get_wtf8(obj) }.as_bytes(),
            pos as usize,
            endpos,
            code,
            must_advance,
            anchored,
        ),
        Subject::Bytes(obj) => drive_scanner_step(
            unsafe { pyre_object::bytesobject::bytes_like_data(obj) },
            pos as usize,
            endpos,
            code,
            must_advance,
            anchored,
        ),
        Subject::Str(obj) => {
            let s = unsafe { utf8_drive(obj) };
            drive_scanner_step(s, pos as usize, endpos, code, must_advance, anchored)
        }
    };
    if !found {
        unsafe { (*sc).pos = -1 }; // self.ctx = None
        unsafe { sre_scanner_release_export(obj) };
        return Ok(None);
    }
    // engine.rs:255-256 — thread (start, must_advance) for the next step.
    let new_must_advance = state.cursor.position == state.start;
    unsafe {
        (*sc).must_advance = new_must_advance as i64;
        (*sc).pos = state.cursor.position as i64;
    }
    Ok(Some(make_match(
        w_srepat,
        string,
        w_buffer,
        &state,
        original_pos,
        endpos as i64,
    )))
}

/// Release the buffer export owned by a lazy finditer/scanner.
///
/// PyPy's `W_SRE_Scanner._finalize_` and normal-exhaustion path share this
/// idempotent ownership-bit check.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn sre_scanner_release_export(obj: PyObjectRef) {
    let sc = obj as *mut W_SRE_Scanner;
    unsafe {
        if (*sc).export_active {
            (*sc).export_active = false;
            crate::builtins::buffer_export_decref((*sc).w_string);
        }
    }
}

/// `next_w` (interp_sre.py) — also the host `space.next` step for
/// `for m in pat.finditer(...)`; raises StopIteration when exhausted.
pub fn sre_scanner_next(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    match sre_scanner_step(obj, false)? {
        Some(m) => Ok(m),
        None => Err(crate::PyError::stop_iteration()),
    }
}

/// `iter_w` (interp_sre.py) — returns self.
fn sre_scanner_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    sre_scanner_self(args)
}

fn sre_scanner_next_w(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = sre_scanner_self(args)?;
    sre_scanner_next(self_)
}

/// `match_w` (interp_sre.py) — anchored match at the current
/// position; returns None when exhausted instead of raising.
fn sre_scanner_match(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = sre_scanner_self(args)?;
    Ok(sre_scanner_step(self_, true)?.unwrap_or_else(w_none))
}

/// `search_w` (interp_sre.py) — search from the current position;
/// returns None when exhausted instead of raising.
fn sre_scanner_search(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = sre_scanner_self(args)?;
    Ok(sre_scanner_step(self_, false)?.unwrap_or_else(w_none))
}
