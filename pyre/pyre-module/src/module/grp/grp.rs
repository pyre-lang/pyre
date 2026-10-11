//! grp implementation — `lib_pypy/grp.py`.
//!
//! Verbatim move of the inline block previously in importing.rs.

#[cfg(unix)]
/// `lib_pypy/grp.py class struct_group(metaclass=structseqtype)`
/// — process-wide cached subclass-of-tuple type so every getgrgid /
/// getgrnam / getgrall call materialises into the same structseq.
static STRUCT_GROUP_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

#[cfg(unix)]
fn struct_group_type() -> pyre_object::PyObjectRef {
    STRUCT_GROUP_TYPE.get_or_init(|| {
        pyre_interpreter::_structseq::make_struct_seq(
            "grp.struct_group",
            &["gr_name", "gr_passwd", "gr_gid", "gr_mem"],
        )
    })
}

/// `_pwdgrp_build.py` includes and `lib_pypy/grp.py` libc calls:
/// `includes=['sys/types.h', 'grp.h']`, `releasegil=False`, no `save_err`.
mod ll {
    use majit_rlib::rffi::CCHARP;

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/types.h", "grp.h"],
        };
    }

    macro_rules! external {
        ($($t:tt)*) => {
            majit_rlib::rffi::llexternal!($($t)*, compilation_info = ECI, releasegil = false);
        };
    }

    external!(
        pub(super) c_getgrgid = "getgrgid",
        [libc::gid_t],
        *mut libc::group
    );
    external!(
        pub(super) c_getgrnam = "getgrnam",
        [CCHARP],
        *mut libc::group
    );
    external!(
        pub(super) c_setgrent = "setgrent",
        [],
        ()
    );
    external!(
        pub(super) c_getgrent = "getgrent",
        [],
        *mut libc::group
    );
    external!(
        pub(super) c_endgrent = "endgrent",
        [],
        ()
    );
}

/// `lib_pypy/grp.py _group_from_gstruct`. String fields are copied with
/// `charp2str` immediately: `getgrent` (and `getgrgid` / `getgrnam`) may
/// return a pointer into a static buffer. `gr_mem` is a NULL-terminated
/// `char**` walked into a list of strings. Each C string is `os.fsdecode`
/// (`_group_from_gstruct`), via `fsdecode_filename_bytes`.
fn make_struct_group(g: *mut libc::group) -> pyre_object::PyObjectRef {
    let name = unsafe { majit_rlib::rffi::charp2str((*g).gr_name.cast()) };
    let passwd = unsafe { majit_rlib::rffi::charp2str((*g).gr_passwd.cast()) };
    let gid = unsafe { (*g).gr_gid } as i64;
    let mut member_bytes = Vec::new();
    unsafe {
        let mut p = (*g).gr_mem;
        loop {
            let member = *p;
            if member.is_null() {
                break;
            }
            member_bytes.push(majit_rlib::rffi::charp2str(member.cast()));
            p = p.add(1);
        }
    }
    let holder = pyre_object::gc_roots::push_roots();
    let mem_list = {
        let mut mem = pyre_object::gc_roots::RootedItems::new();
        for s in &member_bytes {
            mem.push(pyre_interpreter::gateway::fsdecode_filename_bytes(s));
        }
        pyre_object::w_list_new(mem.take())
    };
    let mem_slot = holder.pin_roots(&[mem_list]);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_interpreter::gateway::fsdecode_filename_bytes(&name));
    fields.push(pyre_interpreter::gateway::fsdecode_filename_bytes(&passwd));
    fields.push(pyre_object::w_int_new(gid));
    fields.push(holder.get(mem_slot));
    pyre_interpreter::_structseq::new_instance(struct_group_type(), fields.take())
}

/// `getgrall`'s `try`/`finally`: `c_endgrent` runs on every exit.
struct EndgrentOnDrop;

impl Drop for EndgrentOnDrop {
    fn drop(&mut self) {
        unsafe { ll::c_endgrent() };
    }
}

/// grp module — `lib_pypy/grp.py` (PyPy keeps it app-level via
/// `_pwdgrp_cffi`).
///
/// getgrgid / getgrnam / getgrall return a `grp.struct_group`
/// structseq (subclass of tuple) with named fields `gr_name`,
/// `gr_passwd`, `gr_gid`, `gr_mem` per `lib_pypy/grp.py`.
///
/// Calls `c_getgrgid` / `c_getgrnam` / `c_setgrent` / `c_getgrent` /
/// `c_endgrent`.
#[cfg(unix)]
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `lib_pypy/grp.py class struct_group` — exposed as
    // `grp.struct_group`; every result type uses this same class.
    pyre_interpreter::__pyre_put_new!(ns_slot, "struct_group", struct_group_type());
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getgrgid",
        pyre_interpreter::make_builtin_function_with_arity(
            "getgrgid",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getgrgid() missing argument",
                    ));
                }
                // `Modules/grpmodule.c grp_getgrgid` routes through
                // `_Py_Gid_Converter`, which permits `-1` as a sentinel
                // and rejects other out-of-range values rather than
                // silently truncating.  Mirror that here so a Python
                // bigint that doesn't fit in `gid_t` raises OverflowError.
                let mut w_gid = args[0];
                let val = pyre_object::with_roots!(w_gid => pyre_interpreter::baseobjspace::int_w(w_gid))?;
                let gid_min = libc::gid_t::MIN as i64;
                let gid_max = libc::gid_t::MAX as i64;
                let gid = if val == -1 {
                    libc::gid_t::MAX
                } else if (gid_min..=gid_max).contains(&val) {
                    val as libc::gid_t
                } else {
                    return Err(pyre_interpreter::PyError::overflow_error(
                        "getgrgid: gid is out of range",
                    ));
                };
                let g = pyre_object::with_roots!(w_gid => unsafe { ll::c_getgrgid(gid) });
                if g.is_null() {
                    Err(pyre_interpreter::PyError::key_error(format!(
                        "getgrgid(): gid not found: {}",
                        gid
                    )))
                } else {
                    Ok(make_struct_group(g))
                }
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getgrnam",
        pyre_interpreter::make_builtin_function_with_arity(
            "getgrnam",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getgrnam() missing argument",
                    ));
                }
                // `lib_pypy/grp.py getgrnam`: `isinstance(name, str)`, then
                // `os.fsencode(name)`, then `if b'\0' in name_b`.
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getgrnam(): name should be a string",
                    ));
                }
                let mut w_name = args[0];
                let name_b = pyre_object::with_roots!(w_name => {
                    pyre_interpreter::gateway::fsencode(w_name)
                })?;
                if name_b.contains(&0) {
                    return Err(pyre_interpreter::PyError::value_error("embedded null byte"));
                }
                let g = pyre_object::with_roots!(w_name => {
                    let ll_name =
                        majit_rlib::rffi::scoped_str2charp::new(Some(&name_b));
                    unsafe { ll::c_getgrnam(ll_name.buf) }
                });
                if g.is_null() {
                    Err(pyre_interpreter::PyError::key_error(format!(
                        "getgrnam(): name not found: {}",
                        String::from_utf8_lossy(&name_b)
                    )))
                } else {
                    Ok(make_struct_group(g))
                }
            },
            1,
        )
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getgrall",
        pyre_interpreter::make_builtin_function_with_arity(
            "getgrall",
            |_| {
                // Each struct_group is freshly allocated and building the
                // next one allocates again, so they are pinned as they
                // arrive.
                unsafe { ll::c_setgrent() };
                let _endgrent = EndgrentOnDrop;
                let mut items = pyre_object::gc_roots::RootedItems::new();
                loop {
                    let g = unsafe { ll::c_getgrent() };
                    if g.is_null() {
                        break;
                    }
                    items.push(make_struct_group(g));
                }
                Ok(pyre_object::w_list_new(items.take()))
            },
            0,
        )
    );
    Ok(())
}
