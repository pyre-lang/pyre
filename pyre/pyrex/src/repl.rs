use std::path::PathBuf;
use std::rc::Rc;

use rustpython_compiler::{
    CompileError, Mode, ParseError,
    parser::{InterpolatedStringErrorType, LexicalErrorType, ParseErrorType},
};

use pyre_interpreter::call::register_build_class;
use pyre_interpreter::importing;
use pyre_interpreter::{CompileOpts, PyDisplay, PyError, PyExecutionContext};
use pyre_jit::eval::eval_with_jit;

use crate::repl_readline::{Readline, ReadlineResult};

const DEFAULT_PRIMARY_PROMPT: &str = ">>> ";
const DEFAULT_SECONDARY_PROMPT: &str = "... ";

// This value is short-lived at the REPL compile boundary. Boxing CodeObject
// would add an allocation to every successfully compiled input solely to make
// the uncommon control variants smaller.
#[allow(clippy::large_enum_variant)]
enum ShellCompileAction {
    Execute(pyre_interpreter::CodeObject),
    Ignore,
    ContinueBlock,
    ContinueLine,
    CompileErr(CompileError),
}

enum ShellExecResult {
    Ok,
    ContinueBlock,
    ContinueLine,
    RuntimeErr(PyError),
    CompileErr(PyError),
}

struct ReplRuntime {
    ctx_ptr: *const PyExecutionContext,
    w_globals: pyre_object::PyObjectRef,
    sys_module: pyre_object::PyObjectRef,
}

/// The interpreter identification `app_main.py print_banner` heads both the
/// prompt and a verbose non-interactive run with.
pub(crate) const BANNER: &str = "pyre 0.0.1 (Rust meta-tracing JIT)";

/// `print_banner`'s second line, carried by the run that has no prompt to
/// suggest anything else. The prompt writes an exit hint there instead, so this
/// one is not shared, but the names it offers are: they are `site`'s, which is
/// why upstream gates the line on `not no_site`.
pub(crate) const BANNER_COPYRIGHT: &str =
    "Type \"help\", \"copyright\", \"credits\" or \"license\" for more information.";

/// `resume` carries the interpreter a program has just finished running in, for
/// the `-i` that follows it.  `pymain_run_python` reaches the prompt without
/// starting anything over -- the program's `__main__` is the prompt's, and
/// startup ran once -- so a resumed prompt skips the whole block below and
/// takes only the parts that are the prompt's own: `sys.ps1`/`ps2`, the
/// interactive flags, and `sys.__interactivehook__`.  The PYTHONSTARTUP file
/// is the one step it does not take; that belongs to `pymain_run_stdin`.
pub fn run_repl(quiet: bool, no_site: bool, resume: Option<crate::MainSession>) {
    let mut repl = Readline::new();
    let history_path = repl_history_path();

    if let Err(err) = repl.load_history(&history_path) {
        eprintln!("pyre: could not load REPL history: {err}");
    }

    let _root = pyre_object::gc_roots::push_roots();
    // A resumed prompt is `pymain_repl`, the one that follows a program; a fresh
    // one is `pymain_run_stdin`.  Only the second runs the startup file.
    let resumed = resume.is_some();
    let (execution_context, canonical, main_module) = match resume {
        Some(session) => {
            // The root scope the program pinned these in is gone with the
            // frame that ran it, and the prompt holds them for the rest of the
            // process, so pin them here and read the pins back.
            let canonical = pyre_object::gc_roots::pin_root(session.canonical);
            let main_module = pyre_object::gc_roots::pin_root(session.main_module);
            (session.execution_context, canonical, main_module)
        }
        None => {
            let execution_context = Rc::new(PyExecutionContext::default());
            register_build_class();
            // `app_main.py run_command_line` — install SIGINT →
            // default_int_handler so Ctrl-C at the REPL raises
            // KeyboardInterrupt rather than killing the process, and register
            // the periodic signal-check action.
            unsafe {
                let ec_ptr = Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
                pyre_interpreter::call::set_last_exec_ctx(ec_ptr);
                (*ec_ptr).install_user_del_action();
                pyre_interpreter::module::signal::interp_signal::install_signal_handling(
                    &mut *ec_ptr,
                );
            }

            let w_globals = execution_context.fresh_module_globals();
            let w_globals = pyre_object::gc_roots::pin_root(w_globals);
            unsafe {
                pyre_object::w_dict_setitem_str(
                    w_globals,
                    "__name__",
                    pyre_object::w_str_new("__main__"),
                )
            };

            // PyPy `module.py Module.getdict()` parity: use the canonical
            // W_DictObject so REPL STORE_NAME writes, `globals()`,
            // `f.__globals__`, and `__main__.__dict__` all share one identity.
            let canonical = w_globals;
            // PyPy `module.py Module` is an ordinary GC object. The managed twin
            // matches that, so detaching `sys.modules` can make the module/dict
            // cycle unreachable. The allocation can collect: the dict is already
            // on `_root`, and the result is pinned before `set_sys_module`.
            let main_module = pyre_object::module::w_module_new_aliasing_dict_managed(
                pyre_object::w_str_new("__main__"),
                canonical,
            );
            let main_module = pyre_object::gc_roots::pin_root(main_module);
            importing::set_sys_module("__main__", main_module);

            // pylifecycle.c init_importlib before site — see run_source.
            if let Err(e) =
                importing::init_importlib_bootstrap(canonical, Rc::as_ptr(&execution_context))
            {
                eprintln!("pyre: importlib bootstrap failed: {}", e.message_text());
            }

            pyre_interpreter::app_main::seed_main_loader(
                canonical,
                None,
                false,
                Rc::as_ptr(&execution_context),
            );
            crate::import_site(no_site, canonical, Rc::as_ptr(&execution_context));
            (execution_context, canonical, main_module)
        }
    };

    // `import_site` and the importlib bootstrap ran Python since the fresh arm
    // above minted its `__main__`, and the resumed one has a whole program
    // behind it, so the module is read back off a pin rather than trusted as a
    // local -- the same reason the resumed arm pins what the session carried.
    let main_module = pyre_object::gc_roots::pin_root(main_module);

    // The prompt reads `sys.ps1`/`ps2` off the interpreter's own `sys`, the way
    // `_PySys_GetOptionalAttr` reads `PyInterpreterState.sysdict`.  Importing
    // the name instead would follow whatever the program left in
    // `sys.modules["sys"]`, and a `None` there is the sentinel that makes an
    // import raise -- so a program that blocked the name for its own downstream
    // code would take the prompt with it.
    let Some(sys_module) = importing::get_interpreter_sys_module() else {
        eprintln!("pyre: the interpreter has no sys module");
        return;
    };
    configure_sys_for_repl(sys_module);

    let runtime = ReplRuntime {
        ctx_ptr: Rc::into_raw(Rc::clone(&execution_context)),
        w_globals: canonical,
        sys_module,
    };

    // `pymain_header` runs before either step below, and `pymain_run_stdin`
    // takes the startup file first and the hook second.
    if !quiet {
        // `pymain_header` writes both of its lines with `fprintf(stderr, ...)`,
        // and `print_banner` reaches the same descriptor through `sys.stderr`,
        // so the prompt's greeting is not part of its stdout.  A session whose
        // stdout is captured -- a pipe, a doctest harness, an editor's REPL
        // pane -- gets the results and not the banner.
        eprintln!("{BANNER}");
        eprintln!("Type \"exit()\" or Ctrl-D to exit.");
    }

    if !resumed {
        run_startup_file(&runtime, canonical, main_module);
    }
    run_interactive_hook(&runtime, canonical);

    let mut full_input = String::new();
    let mut continuing_block = false;
    let mut continuing_line = false;
    let mut pending_exit: Option<i32> = None;

    loop {
        let prompt = if continuing_block || continuing_line {
            load_prompt(&runtime, "ps2", DEFAULT_SECONDARY_PROMPT)
        } else {
            load_prompt(&runtime, "ps1", DEFAULT_PRIMARY_PROMPT)
        };
        continuing_line = false;

        let mut at_eof = false;
        let line = match repl.readline(&prompt) {
            ReadlineResult::Line(line) => line,
            ReadlineResult::Partial(line) => {
                // End of input inside a line: the tokenizer asks for the
                // rest of it under `ps2`, meets the end there, and only then
                // runs what it has.
                full_input.push_str(&line);
                full_input.push('\n');
                continuing_line = true;
                continue;
            }
            ReadlineResult::Interrupt => {
                eprintln!("KeyboardInterrupt");
                full_input.clear();
                continuing_block = false;
                continuing_line = false;
                continue;
            }
            ReadlineResult::Eof => {
                // The line the prompt left open ends on the prompt's stream.
                eprintln!();
                if full_input.is_empty() {
                    break;
                }
                // End of input ends the pending statement, block included, and
                // it runs before the prompt comes back for the next read.
                at_eof = true;
                String::new()
            }
            ReadlineResult::Io(err) => {
                eprintln!("pyre: REPL I/O error: {err}");
                break;
            }
            ReadlineResult::Other(err) => {
                eprintln!("pyre: REPL error: {err}");
                break;
            }
        };

        if !at_eof && let Err(err) = repl.add_history_entry(line.trim_end()) {
            eprintln!("pyre: could not record REPL history: {err}");
        }

        let empty_line_given = line.is_empty();
        if full_input.is_empty() {
            full_input = line;
        } else {
            full_input.push_str(&line);
        }
        full_input.push('\n');

        match shell_exec(&runtime, &full_input, empty_line_given, continuing_block) {
            ShellExecResult::Ok => {
                if continuing_block {
                    if empty_line_given {
                        continuing_block = false;
                        full_input.clear();
                    }
                } else {
                    full_input.clear();
                }
            }
            ShellExecResult::ContinueLine => {
                continuing_line = true;
            }
            ShellExecResult::ContinueBlock => {
                continuing_block = true;
            }
            ShellExecResult::CompileErr(err) => {
                print_syntax_error(err);
                full_input.clear();
                continuing_block = false;
                continuing_line = false;
            }
            ShellExecResult::RuntimeErr(err) => {
                if err.kind == pyre_interpreter::PyErrorKind::SystemExit {
                    // exit()/quit() raise SystemExit; terminate the REPL with
                    // its code instead of printing a traceback and continuing.
                    pending_exit = Some(pyre_interpreter::system_exit_code(&err));
                    break;
                }
                crate::note_unhandled_keyboard_interrupt(&err);
                pyre_interpreter::eprint_exception(&err, true);
                full_input.clear();
                continuing_block = false;
                continuing_line = false;
            }
        }
        if at_eof {
            // Whatever the statement left pending, there is no more input for it.
            full_input.clear();
            continuing_block = false;
            continuing_line = false;
        }
    }

    if let Err(err) = repl.save_history(&history_path) {
        eprintln!("pyre: could not save REPL history: {err}");
    }

    // Drop the launcher pin on `canonical` before `finalize_modules`
    // collects. `sys.modules` still owns `__main__` until
    // `release_sys_modules_for_shutdown`; `finalize_runtime` re-pins
    // `canonical` only until that detach.
    drop(_root);

    // `Py_FinalizeEx` runs after the prompt, not before it, and it is the same
    // one the program would have reached had no `-i` followed: `atexit`
    // callbacks, the module teardown and the stream flush all belong here.
    // `exit()` at the prompt takes it too -- the status it asked for is
    // delivered by the exit below, after the callbacks have run.
    crate::finish_main(canonical, Rc::as_ptr(&execution_context));

    // `Py_RunMain` consults `unhandled_keyboard_interrupt` after the prompt
    // and after `Py_FinalizeEx`, and overwrites any `exitcode` the prompt
    // produced — `exit(0)` / `exit(3)` included. A prompt that simply ended
    // still owes the interrupt the same way.
    if crate::had_unhandled_keyboard_interrupt() {
        crate::terminate_by_sigint();
    }
    if let Some(code) = pending_exit {
        std::process::exit(code);
    }
}

/// Report a failure one of the prompt's own startup steps raised.
///
/// `pymain_err_print` takes `SystemExit` first, so a startup file or an
/// interactive hook that raises one ends the process instead of reaching the
/// prompt; everything else is printed through `sys.excepthook` and the prompt
/// opens over it.
fn report_or_exit_before_prompt(
    error: PyError,
    canonical: pyre_object::PyObjectRef,
    ctx_ptr: *const PyExecutionContext,
) {
    if error.kind == pyre_interpreter::PyErrorKind::SystemExit {
        crate::finalize_system_exit(error, canonical, ctx_ptr);
    }
    crate::report_main_error_for_inspect(error);
}

/// Run `source` in the prompt's `__main__` namespace.
fn exec_in_main(runtime: &ReplRuntime, w_code: pyre_object::PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_code);
    let code_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut frame = pyre_interpreter::pyframe::createframe_obj(
        pyre_object::gc_roots::shadow_stack_get(code_slot) as *const (),
        runtime.w_globals,
        runtime.ctx_ptr,
        None,
    )?;
    eval_with_jit(&mut frame, None).map(|_| ())
}

/// `pymain_run_startup` — the PYTHONSTARTUP file, run in `__main__` ahead of
/// the prompt that reads stdin.
///
/// Only that prompt runs it.  `pymain_repl`, the prompt that follows `-c`, `-m`
/// or a script, does not, so a program never has one run after it.
///
/// A file that will not open is reported and skipped: the two lines are the
/// fixed `Could not open PYTHONSTARTUP` and the `OSError` built from the errno,
/// printed the way every other failure here is.
fn run_startup_file(
    runtime: &ReplRuntime,
    canonical: pyre_object::PyObjectRef,
    main_module: pyre_object::PyObjectRef,
) {
    if importing::ignore_environment_flag() {
        return;
    }
    let Some(path) = std::env::var_os("PYTHONSTARTUP") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let path = std::path::PathBuf::from(path);
    // `PYTHONSTARTUP` is read with `var_os`, so it is the one launcher path a
    // filename with no UTF-8 spelling reaches -- `parse_args` refuses a
    // non-Unicode argv outright.  Every name this file is reported under is
    // derived from its filesystem bytes, so the script can open its own
    // `__file__` and its tracebacks name the file that ran.
    let filename_bytes = pyre_interpreter::gateway::fsencode_os_str(path.as_os_str());
    let display = pyre_interpreter::gateway::fsdecode_filename_wtf8(&filename_bytes);
    let bytes = match importing::read_source_bytes(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Could not open PYTHONSTARTUP");
            let w_filename = pyre_interpreter::gateway::fsdecode_filename_bytes(&filename_bytes);
            report_or_exit_before_prompt(
                PyError::os_error_syscall(e.raw_os_error().unwrap_or(0), w_filename),
                canonical,
                runtime.ctx_ptr,
            );
            return;
        }
    };
    // `pymain_run_file_obj` is a tokenizer file boundary just like the normal
    // script path: an embedded NUL keeps its filename, line and source text.
    let source = match pyre_interpreter::decode_file_source_bytes(&bytes, &display, false) {
        Ok(source) => source,
        Err(error) => {
            report_or_exit_before_prompt(error, canonical, runtime.ctx_ptr);
            return;
        }
    };

    // `_PyRun_SimpleFileObject` binds the two module-identity attributes only
    // when `__file__` is not already there, and removes exactly what it bound.
    // The prompt's `__main__` has neither, so the startup file sees its own
    // path and the prompt that follows sees no `__file__` at all.
    let bind_file_name =
        pyre_interpreter::baseobjspace::getattr_str(main_module, "__file__").is_err();
    if bind_file_name {
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            main_module,
            "__file__",
            pyre_interpreter::gateway::fsdecode_filename_bytes(&filename_bytes),
        );
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            main_module,
            "__cached__",
            pyre_object::w_none(),
        );
    }
    // The script and `-c` paths compile through `compile_with_codegen_warnings`
    // so an invalid escape is a SyntaxWarning. The warning takes the fsdecoded
    // spelling; the compiler's `source_path` stays a `str` and the code object
    // is renamed afterwards the way `compile_source_named_by_bytes` does.
    let (source_path, filename_bytes) =
        pyre_interpreter::split_code_filename_bytes(filename_bytes, None);
    let compiled = pyre_interpreter::syntax_warnings::compile_with_codegen_warnings(
        &source,
        Mode::Exec,
        &display,
        CompileOpts {
            optimize: importing::optimize_flag(),
            debug_ranges: importing::code_debug_ranges_flag(),
            int_max_str_digits: pyre_interpreter::module::sys::state::int_max_str_digits().max(0)
                as usize,
            ..Default::default()
        },
    );
    let outcome = match compiled {
        Ok(code) => {
            let w_code = pyre_interpreter::box_code_object(code);
            unsafe {
                pyre_interpreter::set_compilation_unit_filename_bytes(w_code, filename_bytes);
            }
            exec_in_main(runtime, w_code)
        }
        Err(pyre_interpreter::syntax_warnings::SourceCompileError::Compile(error)) => {
            let mut err = pyre_interpreter::compile_err_to_syntax_error(error, &source, Mode::Exec);
            err.replace_syntax_error_filename(pyre_interpreter::gateway::fsdecode_filename_bytes(
                filename_bytes.as_deref().unwrap_or(source_path.as_bytes()),
            ));
            Err(err)
        }
        Err(pyre_interpreter::syntax_warnings::SourceCompileError::Warning(error)) => Err(error),
    };
    if bind_file_name {
        let _ = pyre_interpreter::baseobjspace::delattr_str(main_module, "__file__");
        let _ = pyre_interpreter::baseobjspace::delattr_str(main_module, "__cached__");
    }
    if let Err(error) = outcome {
        report_or_exit_before_prompt(error, canonical, runtime.ctx_ptr);
    }
}

/// `pymain_run_interactive_hook` — `sys.__interactivehook__()`, which
/// `site.enablerlcompleter` leaves behind for whichever prompt opens.
///
/// Every prompt calls it, the one that follows a program included.  The name is
/// read off the interpreter's own `sys` for the reason `ps1` is; a missing one
/// is not a failure, which is what an `-S` run relies on.
fn run_interactive_hook(runtime: &ReplRuntime, canonical: pyre_object::PyObjectRef) {
    // `findattr`, which is `PyObject_GetOptionalAttr`: only a missing name reads
    // as "no hook".  A `sys` whose lookup raises -- a module `__getattr__`, a
    // descriptor -- has failed to produce the hook, and that is reported rather
    // than passed off as an `-S` run's absence.
    let hook =
        match pyre_interpreter::baseobjspace::findattr(runtime.sys_module, "__interactivehook__") {
            Ok(Some(hook)) => hook,
            Ok(None) => return,
            Err(error) => {
                eprintln!("Failed calling sys.__interactivehook__");
                report_or_exit_before_prompt(error, canonical, runtime.ctx_ptr);
                return;
            }
        };
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(hook);
    let hook_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let called = pyre_interpreter::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(hook_slot),
        &[],
    );
    if let Err(error) = called {
        eprintln!("Failed calling sys.__interactivehook__");
        report_or_exit_before_prompt(error, canonical, runtime.ctx_ptr);
    }
}

fn configure_sys_for_repl(sys_module: pyre_object::PyObjectRef) {
    ensure_sys_prompt(sys_module, "ps1", DEFAULT_PRIMARY_PROMPT);
    ensure_sys_prompt(sys_module, "ps2", DEFAULT_SECONDARY_PROMPT);

    if let Ok(flags) = pyre_interpreter::baseobjspace::getattr_str(sys_module, "flags") {
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            flags,
            "interactive",
            pyre_object::w_int_new(1),
        );
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            flags,
            "inspect",
            pyre_object::w_int_new(1),
        );
    }
}

fn ensure_sys_prompt(sys_module: pyre_object::PyObjectRef, name: &str, fallback: &str) {
    if pyre_interpreter::baseobjspace::getattr_str(sys_module, name).is_err() {
        let _roots = pyre_object::gc_roots::push_roots();
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(sys_module);
        let prompt_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(fallback));
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(prompt_slot),
        );
    }
}

fn load_prompt(runtime: &ReplRuntime, name: &str, fallback: &str) -> String {
    read_prompt(runtime.sys_module, name).unwrap_or_else(|| fallback.to_string())
}

fn read_prompt(sys_module: pyre_object::PyObjectRef, name: &str) -> Option<String> {
    let prompt = pyre_interpreter::baseobjspace::getattr_str(sys_module, name).ok()?;
    if prompt.is_null() || unsafe { pyre_object::is_none(prompt) } {
        return None;
    }
    // A `sys.ps1` whose `__str__` raises leaves the prompt unread, so
    // `load_prompt`'s own default stands in; a placeholder would become the
    // prompt itself.
    unsafe { pyre_interpreter::display::py_str_display_result(prompt) }.ok()
}

/// Register a compiled tree with `linecache._interactive_cache` under
/// `filename`.
///
/// `inspect.findsource()` keys on the code object's
/// `(filename, qualname, firstlineno)` for synthetic filenames such as
/// `<stdin>` and `<string>`, which is how the 3.14 interactive runner and the
/// `-c` command path keep their sources retrievable.  `_register_code` walks
/// `co_consts`, so a function defined in the registered tree is published under
/// the same entry.  Callers treat a failure as non-fatal.
pub(crate) fn register_interactive_code(
    w_globals: pyre_object::PyObjectRef,
    ctx_ptr: *const PyExecutionContext,
    w_code: pyre_object::PyObjectRef,
    source: &str,
    filename: &str,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_code);
    let code_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Through the `builtins.__import__` binding -- see the note on
    // `import_site`; `interp_warnings` reaches `linecache` through the binding
    // too, though it calls it with the name alone.  This one stands in for the
    // `import linecache` the interactive runner performs in Python, so it
    // carries the arguments an import statement carries.  The native stand-in
    // behind the binding searches the filesystem itself, so it consults no
    // `sys.path_hooks` entry and parses the source instead of reading the
    // module's bytecode cache.
    let linecache = importing::call_dunder_import(
        "linecache",
        w_globals,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        0,
        ctx_ptr,
    )?;
    let _ = pyre_object::gc_roots::pin_root(linecache);
    let linecache_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let register = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(linecache_slot),
        "_register_code",
    )?;
    let _ = pyre_object::gc_roots::pin_root(register);
    let register_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Each `w_str_new` can collect, so the earlier string is rooted before
    // the next one is built rather than left in a temporary.
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(source));
    let source_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(filename));
    let filename_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_interpreter::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(register_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(code_slot),
            pyre_object::gc_roots::shadow_stack_get(source_slot),
            pyre_object::gc_roots::shadow_stack_get(filename_slot),
        ],
    )?;
    Ok(())
}

fn shell_exec(
    runtime: &ReplRuntime,
    source: &str,
    empty_line_given: bool,
    continuing_block: bool,
) -> ShellExecResult {
    match compile_repl_input(source, empty_line_given, continuing_block) {
        ShellCompileAction::Execute(code) => {
            let w_code = pyre_interpreter::pycode::box_code_object(code);
            let _roots = pyre_object::gc_roots::push_roots();
            let w_code = pyre_object::gc_roots::pin_root(w_code);
            let code_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // Source registration is cosmetic — it only feeds later traceback
            // and `inspect` lookups — so a failure must not discard the
            // statement the user just typed.
            let _ = register_interactive_code(
                runtime.w_globals,
                runtime.ctx_ptr,
                w_code,
                source,
                "<stdin>",
            );
            let mut frame = match pyre_interpreter::pyframe::createframe_obj(
                pyre_object::gc_roots::shadow_stack_get(code_slot) as *const (),
                runtime.w_globals,
                runtime.ctx_ptr,
                None,
            ) {
                Ok(frame) => frame,
                Err(err) => return ShellExecResult::RuntimeErr(err),
            };
            match eval_with_jit(&mut frame, None) {
                Ok(result) => {
                    if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                        println!("{}", PyDisplay(result));
                    }
                    ShellExecResult::Ok
                }
                Err(err) => ShellExecResult::RuntimeErr(err),
            }
        }
        ShellCompileAction::Ignore => ShellExecResult::Ok,
        ShellCompileAction::ContinueBlock => ShellExecResult::ContinueBlock,
        ShellCompileAction::ContinueLine => ShellExecResult::ContinueLine,
        ShellCompileAction::CompileErr(err) => {
            let source = pyre_interpreter::universal_newline(source);
            ShellExecResult::CompileErr(pyre_interpreter::compile_err_to_syntax_error(
                err,
                &source,
                Mode::Single,
            ))
        }
    }
}

/// A statement that does not compile is reported the way `PyErr_Print`
/// reports any other error at the prompt: through `sys.excepthook`.
fn print_syntax_error(mut err: PyError) {
    if !pyre_interpreter::error::print_exception_via_excepthook(&mut err) {
        pyre_interpreter::eprint_syntax_error(&err);
    }
}

fn compile_repl_input(
    source: &str,
    empty_line_given: bool,
    continuing_block: bool,
) -> ShellCompileAction {
    // `generate_tokens` normalizes every line it takes, on every platform and
    // for a lone `\r` as well (`pytokenizer.py`), so the shell hands the
    // compiler the same shape a script would rather than a windows-only
    // `\r\n` rewrite.
    let source = &*pyre_interpreter::universal_newline(source);

    if !empty_line_given && !continuing_block && starts_compound_statement(source) {
        return ShellCompileAction::ContinueBlock;
    }

    match pyre_interpreter::rp_compile(source, Mode::Single, "<stdin>", Default::default()) {
        Ok(code) => {
            if empty_line_given || !continuing_block {
                ShellCompileAction::Execute(code)
            } else {
                ShellCompileAction::Ignore
            }
        }
        Err(CompileError::Parse(ParseError {
            error: ParseErrorType::Lexical(LexicalErrorType::Eof),
            ..
        })) => ShellCompileAction::ContinueLine,
        Err(CompileError::Parse(ParseError {
            error:
                ParseErrorType::Lexical(LexicalErrorType::FStringError(
                    InterpolatedStringErrorType::UnterminatedTripleQuotedString { .. },
                )),
            ..
        })) => ShellCompileAction::ContinueLine,
        Err(CompileError::Parse(ParseError {
            is_unclosed_bracket: true,
            ..
        })) => ShellCompileAction::ContinueLine,
        Err(err) => {
            if let CompileError::Parse(ParseError {
                error: ParseErrorType::Lexical(LexicalErrorType::UnclosedStringError { .. }),
                raw_location,
                ..
            }) = &err
            {
                let loc = raw_location.start().to_usize();
                let mut iter = source.chars();
                if let Some(quote) = iter.nth(loc)
                    && iter.next() == Some(quote)
                    && iter.next() == Some(quote)
                {
                    return ShellCompileAction::ContinueLine;
                }
            }

            let bad_error = match &err {
                CompileError::Parse(parse_err) => match &parse_err.error {
                    ParseErrorType::Lexical(LexicalErrorType::IndentationError) => continuing_block,
                    // An incomplete block at the REPL continues rather than
                    // erroring, whether the missing suite arrives structured
                    // or as the message form below.
                    ParseErrorType::ExpectedIndentedBlock { .. } => false,
                    ParseErrorType::OtherError(msg) => {
                        // The compiler reports the missing suite as the CPython
                        // form "expected an indented block after <clause> on
                        // line N"; an incomplete block at the REPL continues
                        // rather than erroring.
                        !msg.to_ascii_lowercase()
                            .starts_with("expected an indented block")
                    }
                    _ => true,
                },
                _ => true,
            };

            if empty_line_given || bad_error {
                ShellCompileAction::CompileErr(err)
            } else {
                ShellCompileAction::ContinueBlock
            }
        }
    }
}

/// The interactive grammar closes a compound statement only at a blank line,
/// so one written on a single line is still incomplete
/// (`codeop.compile_command("if 1: pass")` answers `None`) and waits under
/// `ps2` however it would compile.
fn starts_compound_statement(source: &str) -> bool {
    use rustpython_compiler::ast::Stmt;
    let Ok(parsed) = rustpython_compiler::parser::parse_module(source) else {
        return false;
    };
    parsed.syntax().body.first().is_some_and(|stmt| {
        matches!(
            stmt,
            Stmt::FunctionDef(_)
                | Stmt::ClassDef(_)
                | Stmt::If(_)
                | Stmt::For(_)
                | Stmt::While(_)
                | Stmt::With(_)
                | Stmt::Try(_)
                | Stmt::Match(_)
        )
    })
}

fn repl_history_path() -> PathBuf {
    match dirs::config_dir() {
        Some(mut path) => {
            path.push("pyre");
            path.push("repl_history.txt");
            path
        }
        None => PathBuf::from(".pyre_repl_history.txt"),
    }
}

#[cfg(test)]
mod tests {
    use super::{ShellCompileAction, compile_repl_input, read_prompt};

    #[test]
    fn incomplete_block_continues() {
        let result = compile_repl_input("if True:\n", false, false);
        assert!(matches!(result, ShellCompileAction::ContinueBlock));
    }

    #[test]
    fn incomplete_line_continues() {
        let result = compile_repl_input("x = (\n", false, false);
        assert!(matches!(result, ShellCompileAction::ContinueLine));
    }

    #[test]
    fn one_line_compound_statement_waits_for_blank_line() {
        let result = compile_repl_input("def f(x, x): ...\n", false, false);
        assert!(matches!(result, ShellCompileAction::ContinueBlock));
        let result = compile_repl_input("def f(x, x): ...\n\n", true, true);
        assert!(matches!(result, ShellCompileAction::CompileErr(_)));
    }

    #[test]
    fn completed_block_waits_for_blank_line() {
        let result = compile_repl_input("if True:\n    1\n", false, true);
        assert!(matches!(result, ShellCompileAction::Ignore));
    }

    #[test]
    fn reads_prompt_from_sys_module() {
        pyre_interpreter::test_hooks::install_hash_hook();
        let canonical = pyre_object::w_module_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(canonical, "ps1", pyre_object::w_str_new("py> "));
        }
        let sys_module = pyre_object::module::w_module_new_aliasing_dict(
            pyre_object::w_str_new("sys"),
            canonical,
        );

        assert_eq!(read_prompt(sys_module, "ps1").as_deref(), Some("py> "));
    }
}
