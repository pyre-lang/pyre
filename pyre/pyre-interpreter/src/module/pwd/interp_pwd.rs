//! pwd implementation — PyPy: pypy/module/pwd/interp_pwd.py
//!
//! Verbatim move of the inline block previously in importing.rs.

#[cfg(unix)]
/// `app_pwd.py class struct_passwd(metaclass=structseqtype)`.
/// Process-wide cached subclass-of-tuple type so every getpwuid /
/// getpwnam / getpwall result materialises into the same structseq.
static STRUCT_PASSWD_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

#[cfg(unix)]
fn struct_passwd_type() -> pyre_object::PyObjectRef {
    STRUCT_PASSWD_TYPE.get_or_init(|| {
        pyre_interpreter::_structseq::make_struct_seq(
            "pwd.struct_passwd",
            &[
                "pw_name",
                "pw_passwd",
                "pw_uid",
                "pw_gid",
                "pw_gecos",
                "pw_dir",
                "pw_shell",
            ],
        )
    })
}

/// `interp_pwd.py uid_converter` — narrow a python int to `uid_t`.
///
/// `-1` is the "current uid" sentinel and passes through unchanged
/// (cast to `uid_t` it becomes the max value, matching the C convention
/// most BSDs use).  Other negative inputs raise OverflowError "user id
/// is less than minimum"; values that don't fit in `uid_t` raise
/// OverflowError "user id is greater than maximum".  Floats / non-int
/// inputs raise TypeError via `int_w`.
#[cfg(unix)]
fn pwd_uid_converter(
    mut w_uid: pyre_object::PyObjectRef,
) -> Result<libc::uid_t, pyre_interpreter::PyError> {
    let val = match pyre_object::with_roots!(w_uid => pyre_interpreter::baseobjspace::int_w(w_uid))
    {
        Ok(v) => v,
        Err(e) if matches!(e.kind, pyre_interpreter::PyErrorKind::OverflowError) => {
            return Err(pyre_interpreter::PyError::overflow_error(
                "user id is greater than maximum",
            ));
        }
        Err(e) => return Err(e),
    };
    if val == -1 {
        return Ok((-1i64) as libc::uid_t);
    }
    if val < 0 {
        return Err(pyre_interpreter::PyError::overflow_error(
            "user id is less than minimum",
        ));
    }
    let uid = val as libc::uid_t;
    if uid as i64 != val {
        return Err(pyre_interpreter::PyError::overflow_error(
            "user id is greater than maximum",
        ));
    }
    Ok(uid)
}

/// `interp_pwd.py` `eci` and `external()`: `includes=['pwd.h']`,
/// `releasegil=False`, no `save_err`.
mod ll {
    use majit_rlib::rffi::CCHARP;

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["pwd.h"],
        };
    }

    macro_rules! external {
        ($($t:tt)*) => {
            majit_rlib::rffi::llexternal!($($t)*, compilation_info = ECI, releasegil = false);
        };
    }

    external!(
        pub(super) c_getpwuid = "getpwuid",
        [libc::uid_t],
        *mut libc::passwd
    );
    external!(
        pub(super) c_getpwnam = "getpwnam",
        [CCHARP],
        *mut libc::passwd
    );
    external!(
        pub(super) c_setpwent = "setpwent",
        [],
        ()
    );
    external!(
        pub(super) c_getpwent = "getpwent",
        [],
        *mut libc::passwd
    );
    external!(
        pub(super) c_endpwent = "endpwent",
        [],
        ()
    );
}

/// `interp_pwd.py make_struct_passwd`. String fields are copied with
/// `charp2str` immediately: `getpwent` (and `getpwuid` / `getpwnam`) may
/// return a pointer into a static buffer.
fn make_struct_passwd(pw: *mut libc::passwd) -> pyre_object::PyObjectRef {
    let name = unsafe { majit_rlib::rffi::charp2str((*pw).pw_name.cast()) };
    let passwd = unsafe { majit_rlib::rffi::charp2str((*pw).pw_passwd.cast()) };
    let gecos = unsafe { majit_rlib::rffi::charp2str((*pw).pw_gecos.cast()) };
    let dir = unsafe { majit_rlib::rffi::charp2str((*pw).pw_dir.cast()) };
    let shell = unsafe { majit_rlib::rffi::charp2str((*pw).pw_shell.cast()) };
    let uid = unsafe { (*pw).pw_uid } as i64;
    let gid = unsafe { (*pw).pw_gid } as i64;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
        &name,
    )));
    fields.push(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
        &passwd,
    )));
    fields.push(pyre_object::w_int_new(uid));
    fields.push(pyre_object::w_int_new(gid));
    fields.push(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
        &gecos,
    )));
    fields.push(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
        &dir,
    )));
    fields.push(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
        &shell,
    )));
    pyre_interpreter::_structseq::new_instance(struct_passwd_type(), fields.take())
}

/// `getpwall`'s `try`/`finally`: `c_endpwent` runs on every exit.
struct EndpwentOnDrop;

impl Drop for EndpwentOnDrop {
    fn drop(&mut self) {
        unsafe { ll::c_endpwent() };
    }
}

/// pwd module — `pypy/module/pwd/interp_pwd.py`.
///
/// getpwuid / getpwnam / getpwall return 7-tuples with the
/// `(pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell)`
/// layout.  `struct_passwd` / `struct_pwent` are exposed as the same
/// builtin type so `isinstance(pwd.struct_passwd, type)` succeeds and
/// `pwd.struct_passwd` is identity-equal to `pwd.struct_pwent`
/// (`app_pwd.py`).  Full structseq instance materialisation
/// (so `pw_entry.pw_name` returns a string) is a framework prereq
/// tracked separately.
///
/// Calls `c_getpwuid` / `c_getpwnam` / `c_setpwent` / `c_getpwent` /
/// `c_endpwent`.
#[cfg(unix)]
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `app_pwd.py class struct_passwd(metaclass=structseqtype)`.
    crate::__pyre_put_new!(ns_slot, "struct_passwd", struct_passwd_type());
    crate::__pyre_put_new!(ns_slot, "struct_pwent", struct_passwd_type());
    crate::__pyre_put_new!(
        ns_slot,
        "getpwuid",
        pyre_interpreter::make_builtin_function_with_arity(
            "getpwuid",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getpwuid() missing argument",
                    ));
                }
                // `interp_pwd.py uid_converter`: -1 sentinel passes
                // through; negative-other → OverflowError "less than
                // minimum"; positive-too-big → OverflowError "greater
                // than maximum".  `interp_pwd.py getpwuid` catches
                // OverflowError and converts it to KeyError "uid not
                // found".
                let mut w_uid = args[0];
                let uid = match pyre_object::with_roots!(w_uid => pwd_uid_converter(w_uid)) {
                    Ok(u) => u,
                    Err(e) if matches!(e.kind, pyre_interpreter::PyErrorKind::OverflowError) => {
                        return Err(pyre_interpreter::PyError::key_error(
                            "getpwuid(): uid not found",
                        ));
                    }
                    Err(e) => return Err(e),
                };
                let pw = pyre_object::with_roots!(w_uid => unsafe { ll::c_getpwuid(uid) });
                if pw.is_null() {
                    Err(pyre_interpreter::PyError::key_error(format!(
                        "getpwuid(): uid not found: {}",
                        uid as i64
                    )))
                } else {
                    Ok(make_struct_passwd(pw))
                }
            },
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "getpwnam",
        pyre_interpreter::make_builtin_function_with_arity(
            "getpwnam",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "getpwnam() missing argument",
                    ));
                }
                // `interp_pwd.py @unwrap_spec(name='text0')` → `text0_w`.
                let mut w_name = args[0];
                let name = pyre_object::with_roots!(w_name => {
                    pyre_interpreter::baseobjspace::text0_w(w_name)
                })?;
                let pw = pyre_object::with_roots!(w_name => {
                    let ll_name =
                        majit_rlib::rffi::scoped_str2charp::new(Some(name.as_bytes()));
                    unsafe { ll::c_getpwnam(ll_name.buf) }
                });
                if pw.is_null() {
                    Err(pyre_interpreter::PyError::key_error(format!(
                        "getpwnam(): name not found: {}",
                        name
                    )))
                } else {
                    Ok(make_struct_passwd(pw))
                }
            },
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "getpwall",
        pyre_interpreter::make_builtin_function_with_arity(
            "getpwall",
            |_| {
                // Every entry is freshly allocated and the next one allocates
                // again, so they are pinned as they arrive (`build_list_storage`).
                unsafe { ll::c_setpwent() };
                let _endpwent = EndpwentOnDrop;
                let mut items = pyre_object::gc_roots::RootedItems::new();
                loop {
                    let pw = unsafe { ll::c_getpwent() };
                    if pw.is_null() {
                        break;
                    }
                    items.push(make_struct_passwd(pw));
                }
                Ok(pyre_object::w_list_new(items.take()))
            },
            0,
        )
    );
    Ok(())
}
