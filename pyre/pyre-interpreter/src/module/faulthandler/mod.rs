//! faulthandler module — PyPy: pypy/module/faulthandler/
//!
//! Unix fatal-signal and user-signal handling lives in `cintf` (`cintf.py` +
//! `faulthandler.c`). Traceback dump from a signal handler needs
//! `vmprof_get_traceback`, which is not ported.

#[cfg(all(unix, feature = "host_env"))]
mod cintf;

pyre_interpreter::pyre_module_init!(handler);
