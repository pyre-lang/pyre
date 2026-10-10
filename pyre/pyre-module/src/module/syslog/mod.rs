//! syslog module — PyPy: `lib_pypy/syslog.py`.
//!
//! openlog / syslog / closelog / setlogmask call libc `llexternal!`
//! declarations.  Unix-only.

pyre_interpreter::pyre_module_init!(syslog);
