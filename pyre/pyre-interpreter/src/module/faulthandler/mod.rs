//! faulthandler module — PyPy: pypy/module/faulthandler/
//!
//! The fatal-signal handlers, the watchdog thread and (off Windows) the
//! user-signal handlers live in `cintf` (`cintf.py` + `faulthandler.c`); the
//! traceback they write comes from `dumper` (`dumper.py`).

#[cfg(all(any(unix, windows), feature = "host_env"))]
mod cintf;
#[cfg(all(any(unix, windows), feature = "host_env"))]
mod dumper;

pyre_interpreter::pyre_module_init!(handler);
