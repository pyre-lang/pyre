//! winsound module — the Windows sound calls.
//!
//! `PlaySound`, `Beep` and `MessageBeep` are the whole surface, over
//! `PlaySoundW` (winmm), `Beep` (kernel32) and `MessageBeep` (user32).  PyPy
//! has no counterpart, so the shapes below follow `PC/winsound.c`.
//!
//! `sound` takes its own converter rather than the shared path one: this call
//! wants the name as wide characters, and it tells a `bytes` argument apart
//! from a `__fspath__` that answered with `bytes` — the filesystem-encoded
//! bytes the shared converter produces have forgotten which it was.

use pyre_object::PyObjectRef;
use rustpython_host_env::winsound as host_winsound;
use rustpython_wtf8::Wtf8;

/// Play from a buffer rather than from a name.
const SND_MEMORY: i32 = host_winsound::SND_MEMORY as i32;
/// Return before the sound finishes.
const SND_ASYNC: i32 = host_winsound::SND_ASYNC as i32;

enum PreparedSound {
    Memory(Vec<u16>),
    Name(Vec<u16>),
}

/// The wide, NUL-terminated name `PlaySoundW` takes.
///
/// `bytes` is turned away even though it names a file elsewhere: the argument
/// is a buffer under `SND_MEMORY`, and accepting it here as a name too would
/// make the same value mean two things.
fn sound_name(sound: PyObjectRef) -> Result<Vec<u16>, pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let sound_slot = roots.base();
    let sound = roots.pin_root(sound);
    let w_name = if unsafe { pyre_object::is_str(sound) } {
        roots.get(sound_slot)
    } else if unsafe { pyre_object::bytesobject::is_bytes(sound) } {
        return Err(pyre_interpreter::PyError::type_error(
            "'sound' must be str, os.PathLike, or None, not bytes",
        ));
    } else {
        // `type(sound).__fspath__(sound)` — the descriptor read off the type is
        // unbound, so the object is supplied as the sole argument.
        let Some(fspath_fn) = pyre_interpreter::typedef::r#type(sound).and_then(|pt| unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(
                pt.as_ptr(),
                pyre_object::unicodeobject::box_str_constant(Wtf8::new("__fspath__")),
            )
        }) else {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "expected str, bytes or os.PathLike object, not {}",
                pyre_interpreter::gateway::short_type_name(sound)
            )));
        };
        let fspath_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(fspath_fn);
        let resolved = pyre_interpreter::call::call_function_impl_result(
            roots.get(fspath_slot),
            &[roots.get(sound_slot)],
        )?;
        let resolved_slot = pyre_object::gc_roots::shadow_stack_len();
        let resolved = roots.pin_root(resolved);
        if !unsafe { pyre_object::is_str(resolved) } {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "'sound' must resolve to str, not {}",
                pyre_interpreter::gateway::short_type_name(resolved)
            )));
        }
        resolved
    };

    // The name goes to the host as spelled, unpaired surrogates included: it
    // takes UTF-16, and a lossy re-decode would address a different file.
    let mut units: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(w_name) }
        .encode_wide()
        .collect();
    if units.contains(&0) {
        return Err(pyre_interpreter::PyError::value_error(
            "embedded null character",
        ));
    }
    units.push(0);
    Ok(units)
}

/// `PyErr_SetExcFromWindowsErr(PyExc_RuntimeError, 0)` — a `RuntimeError`
/// carrying the arguments the OSError constructor takes: errno 0, the system's
/// own text for the code, no filename, the code as a signed int, no second
/// filename.  `RuntimeError` reads none of them and keeps the five as `args`.
fn win32_runtime_error(code: u32) -> pyre_interpreter::PyError {
    let message = pyre_interpreter::PyError::win32_strerror(code as i32);
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class("RuntimeError") else {
        return pyre_interpreter::PyError::runtime_error(message);
    };
    // Building an argument can collect, so each one is pinned as it is made:
    // a Rust array of raw references is not something the collector updates.
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.base();
    let _ = roots.pin_root(cls);
    let _ = roots.pin_root(pyre_object::w_int_new(0));
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&message));
    let _ = roots.pin_root(pyre_object::w_none());
    let _ = roots.pin_root(pyre_object::w_int_new(i64::from(code as i32)));
    let _ = roots.pin_root(pyre_object::w_none());
    let args = [
        roots.get(cls_slot + 1),
        roots.get(cls_slot + 2),
        roots.get(cls_slot + 3),
        roots.get(cls_slot + 4),
        roots.get(cls_slot + 5),
    ];
    match pyre_interpreter::call::call_function_impl_result(roots.get(cls_slot), &args) {
        Ok(exc) => unsafe { pyre_interpreter::PyError::from_exc_object(exc) },
        Err(error) => error,
    }
}

pyre_interpreter::py_module! {
    "winsound",
    int_constants: {
        // `PlaySound` flags (mmsystem.h).  SND_SYNC is the absence of
        // SND_ASYNC rather than a bit of its own.
        "SND_SYNC" => host_winsound::SND_SYNC as i32,
        "SND_ASYNC" => SND_ASYNC,
        "SND_NODEFAULT" => host_winsound::SND_NODEFAULT as i32,
        "SND_MEMORY" => SND_MEMORY,
        "SND_LOOP" => host_winsound::SND_LOOP as i32,
        "SND_NOSTOP" => host_winsound::SND_NOSTOP as i32,
        "SND_PURGE" => host_winsound::SND_PURGE as i32,
        "SND_APPLICATION" => host_winsound::SND_APPLICATION as i32,
        "SND_NOWAIT" => host_winsound::SND_NOWAIT as i32,
        "SND_ALIAS" => host_winsound::SND_ALIAS as i32,
        "SND_FILENAME" => host_winsound::SND_FILENAME as i32,
        "SND_SENTRY" => host_winsound::SND_SENTRY as i32,
        "SND_SYSTEM" => host_winsound::SND_SYSTEM as i32,
        // `MessageBeep` sound ids (winuser.h).  Several are aliases: HAND,
        // ERROR and STOP are one sound, as are EXCLAMATION and WARNING, and
        // ASTERISK and INFORMATION.
        "MB_OK" => host_winsound::MB_OK as i32,
        "MB_ICONHAND" => host_winsound::MB_ICONHAND as i32,
        "MB_ICONERROR" => host_winsound::MB_ICONERROR as i32,
        "MB_ICONSTOP" => host_winsound::MB_ICONSTOP as i32,
        "MB_ICONQUESTION" => host_winsound::MB_ICONQUESTION as i32,
        "MB_ICONEXCLAMATION" => host_winsound::MB_ICONEXCLAMATION as i32,
        "MB_ICONWARNING" => host_winsound::MB_ICONWARNING as i32,
        "MB_ICONASTERISK" => host_winsound::MB_ICONASTERISK as i32,
        "MB_ICONINFORMATION" => host_winsound::MB_ICONINFORMATION as i32,
    },
    inline_functions: {
        fn PlaySound(sound: PyObjectRef, flags: PyIndexCInt) -> Result<(), pyre_interpreter::PyError> {
            // `None` answers first, so it silences the device whatever else
            // the flags asked for.
            let source = if unsafe { pyre_object::is_none(sound) } {
                None
            } else if flags & SND_MEMORY != 0 {
                if flags & SND_ASYNC != 0 {
                    // The buffer would have to outlive the call, and nothing
                    // here can keep it alive that long.
                    return Err(pyre_interpreter::PyError::runtime_error(
                        "Cannot play asynchronously from memory",
                    ));
                }
                let data = pyre_interpreter::baseobjspace::simple_buffer_bytes(sound)?.ok_or_else(|| {
                    pyre_interpreter::PyError::type_error(format!(
                        "a bytes-like object is required, not '{}'",
                        pyre_interpreter::gateway::short_type_name(sound)
                    ))
                })?;
                let bytes = data.as_bytes().to_vec();
                data.release();
                let mut units = vec![0u16; bytes.len().div_ceil(2).max(1)];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        units.as_mut_ptr().cast::<u8>(),
                        bytes.len(),
                    );
                }
                Some(PreparedSound::Memory(units))
            } else {
                Some(PreparedSound::Name(sound_name(sound)?))
            };
            let result = {
                // A synchronous play runs to the end of the sound, so the
                // thread that will stop it has to be able to run.
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                match source.as_ref() {
                    None => host_winsound::play_sound(
                        host_winsound::PlaySoundSource::Stop,
                        (flags & !SND_MEMORY) as u32,
                    ),
                    Some(PreparedSound::Memory(units)) => {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                units.as_ptr().cast::<u8>(),
                                units.len() * std::mem::size_of::<u16>(),
                            )
                        };
                        host_winsound::play_sound(
                            host_winsound::PlaySoundSource::Memory(bytes),
                            flags as u32,
                        )
                    }
                    Some(PreparedSound::Name(name)) => {
                        let name = widestring::WideCStr::from_slice(name)
                            .expect("sound_name returns one trailing NUL");
                        host_winsound::play_sound(
                            host_winsound::PlaySoundSource::Name(name),
                            flags as u32,
                        )
                    }
                }
            };
            if result.is_err() {
                return Err(pyre_interpreter::PyError::runtime_error("Failed to play sound"));
            }
            Ok(())
        }
        fn Beep(frequency: PyIndexCInt, duration: PyIndexCInt) -> Result<(), pyre_interpreter::PyError> {
            // The range `Beep` itself accepts; naming one outside it is a
            // ValueError rather than a failed call.
            if !(37..=32767).contains(&frequency) {
                return Err(pyre_interpreter::PyError::value_error(
                    "frequency must be in 37 thru 32767",
                ));
            }
            let beeped = {
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                host_winsound::beep(frequency as u32, duration as u32)
            };
            if !beeped {
                return Err(pyre_interpreter::PyError::runtime_error("Failed to beep"));
            }
            Ok(())
        }
        fn MessageBeep(#[default(0i32)] type_: PyIndexCInt) -> Result<(), pyre_interpreter::PyError> {
            // Read inside the released region, as
            // `save_err=rffi.RFFI_SAVE_LASTERROR` does: reacquiring goes
            // through `mutex2_lock_timeout`, whose timed wait leaves
            // `ERROR_TIMEOUT` behind on a poll that expired.
            let result = {
                let _blocked = pyre_interpreter::module::thread::before_external_block();
                host_winsound::message_beep(type_ as u32)
            };
            if let Err(error) = result {
                return Err(win32_runtime_error(error.raw_os_error().unwrap_or(0) as u32));
            }
            Ok(())
        }
    }
}
