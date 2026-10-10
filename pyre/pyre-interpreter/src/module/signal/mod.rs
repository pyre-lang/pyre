//! _signal module — PyPy: pypy/module/signal/
//!
//! `signal()` / `getsignal()` register real handlers via `signalstate`
//! (sigaction + a pending-signal flag) and deliver them through
//! `CheckSignalAction` at the next interpreter checkpoint.  alarm / pause
//! / raise_signal / strsignal / valid_signals call `rsignal` llexternals.
//! `set_wakeup_fd` records the fd (and `warn_on_full_buffer`); the handler
//! writes the signal-number byte to it and stashes any write errno for the
//! next checkpoint to report.

pub mod signalstate;

crate::pyre_module_init!(interp_signal);
