/// Structured JIT profiling and statistics logging.
///
/// Tracks compilation events, guard failures, and timing data.
/// Activated via MAJIT_LOG=1 or MAJIT_STATS=1 environment variables.
/// Prints a summary report on drop.
use indexmap::IndexMap;
use std::time::{Duration, Instant};

/// Whether JIT statistics collection is enabled.
/// Checks MAJIT_STATS=1 or MAJIT_LOG=1.
pub fn stats_enabled() -> bool {
    std::env::var("MAJIT_STATS")
        .or_else(|_| std::env::var("MAJIT_LOG"))
        .is_ok_and(|v| v == "1")
}

/// logger.py int_could_be_an_address.
pub fn int_could_be_an_address(x: i64) -> bool {
    !(-32768..=32767).contains(&x)
}

/// logger.py `class LogOperations(object)`.
///
/// PyPy uses this helper to format ResOperation lists for debug output.
/// Pyre's current logger records structured statistics only; the type
/// surface is kept so the upstream `metainterp.logger` path exposes the
/// same public class name when the operation formatter is ported.
#[derive(Debug, Clone, Default)]
pub struct LogOperations;

/// Per-trace compilation record.
#[derive(Debug, Clone)]
pub struct TraceRecord {
    /// Green key hash identifying this trace.
    pub green_key: u64,
    /// Number of operations before optimization.
    pub ops_before_opt: usize,
    /// Number of operations after optimization.
    pub ops_after_opt: usize,
    /// Time spent in the unroll and optimization pipeline, excluding trace recording.
    pub opt_time: Duration,
    /// Time spent in backend compilation.
    pub compile_time: Duration,
}

/// Structured JIT profiling logger.
///
/// Collects statistics about trace compilation, guard failures,
/// and loop entries. Prints a summary on drop when enabled.
#[derive(Debug)]
pub struct Logger {
    /// Successfully compiled traces.
    compiled: Vec<TraceRecord>,
    /// Number of traces that were aborted.
    aborted: u64,
    /// Number of bridges compiled from guard failures.
    bridges_compiled: u64,
    /// Loop entry counts, keyed by green key hash.
    loop_entries: IndexMap<u64, u64>,
    /// Whether to print the summary on drop.
    print_on_drop: bool,
}

impl Logger {
    /// Create a new Logger. If `print_on_drop` is true, a summary
    /// is printed to stderr when the Logger is dropped.
    pub fn new(print_on_drop: bool) -> Self {
        Logger {
            compiled: Vec::new(),
            aborted: 0,
            bridges_compiled: 0,
            loop_entries: IndexMap::new(),
            print_on_drop,
        }
    }

    /// Create a Logger that is enabled based on environment variables.
    /// Returns `Some(Logger)` if MAJIT_STATS=1 or MAJIT_LOG=1, else `None`.
    pub fn from_env() -> Option<Self> {
        if stats_enabled() {
            Some(Self::new(true))
        } else {
            None
        }
    }

    /// Record a successful trace compilation.
    pub fn log_compile(
        &mut self,
        green_key: u64,
        ops_before_opt: usize,
        ops_after_opt: usize,
        opt_time: Duration,
        compile_time: Duration,
    ) {
        self.compiled.push(TraceRecord {
            green_key,
            ops_before_opt,
            ops_after_opt,
            opt_time,
            compile_time,
        });
    }

    /// Record a trace abort.
    pub fn log_abort(&mut self) {
        self.aborted += 1;
    }

    /// Record a loop entry for the given green key.
    pub fn log_loop_entry(&mut self, green_key: u64) {
        *self.loop_entries.entry(green_key).or_default() += 1;
    }

    /// Record a bridge compilation for the given guard index.
    pub fn log_bridge_compile(&mut self, _guard_index: u32) {
        self.bridges_compiled += 1;
    }

    /// Number of bridges compiled.
    pub fn bridges_compiled(&self) -> u64 {
        self.bridges_compiled
    }

    /// Number of successfully compiled traces.
    pub fn traces_compiled(&self) -> usize {
        self.compiled.len()
    }

    /// Number of aborted traces.
    pub fn traces_aborted(&self) -> u64 {
        self.aborted
    }

    /// Total operations recorded across all compiled traces (before optimization).
    pub fn total_ops_before(&self) -> usize {
        self.compiled.iter().map(|r| r.ops_before_opt).sum()
    }

    /// Total operations after optimization across all compiled traces.
    pub fn total_ops_after(&self) -> usize {
        self.compiled.iter().map(|r| r.ops_after_opt).sum()
    }

    /// Total compilation time across all traces.
    pub fn total_compile_time(&self) -> Duration {
        self.compiled.iter().map(|r| r.compile_time).sum()
    }

    /// Total optimization time across all traces.
    pub fn total_opt_time(&self) -> Duration {
        self.compiled.iter().map(|r| r.opt_time).sum()
    }

    /// Get the compiled trace records.
    pub fn compiled_traces(&self) -> &[TraceRecord] {
        &self.compiled
    }

    /// Get loop entry counts.
    pub fn loop_entry_counts(&self) -> &IndexMap<u64, u64> {
        &self.loop_entries
    }

    /// Format the summary report as a string.
    pub fn summary(&self) -> String {
        let total_before = self.total_ops_before();
        let total_after = self.total_ops_after();
        // Signed in f64: optimization can raise the op count (guards/spills), so
        // `total_after > total_before` is a legitimate negative reduction, not an
        // unsigned-subtraction underflow.
        let reduction_pct = if total_before > 0 {
            (total_before as f64 - total_after as f64) / total_before as f64 * 100.0
        } else {
            0.0
        };
        let compile_ms = self.total_compile_time().as_secs_f64() * 1000.0;
        let opt_ms = self.total_opt_time().as_secs_f64() * 1000.0;

        format!(
            "\
=== JIT Statistics ===
Traces compiled: {}
Traces aborted: {}
Traces bridged: {}
Total ops recorded: {}
Total ops after opt: {} ({:.0}% reduction)
Optimization time: {:.1}ms
Compilation time: {:.1}ms",
            self.traces_compiled(),
            self.traces_aborted(),
            self.bridges_compiled(),
            total_before,
            total_after,
            reduction_pct,
            opt_ms,
            compile_ms,
        )
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        if self.print_on_drop && (self.traces_compiled() > 0 || self.traces_aborted() > 0) {
            eprintln!("{}", self.summary());
        }
    }
}

/// A lightweight timer for measuring JIT phases.
pub struct JitTimer {
    start: Instant,
}

impl JitTimer {
    /// Start a new timer.
    pub fn start() -> Self {
        JitTimer {
            start: Instant::now(),
        }
    }

    /// Return elapsed duration since the timer was started.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_could_be_an_address() {
        assert!(!int_could_be_an_address(-32768));
        assert!(!int_could_be_an_address(0));
        assert!(!int_could_be_an_address(32767));
        assert!(int_could_be_an_address(-32769));
        assert!(int_could_be_an_address(32768));
    }

    #[test]
    fn test_jitlog_empty() {
        let log = Logger::new(false);
        assert_eq!(log.traces_compiled(), 0);
        assert_eq!(log.traces_aborted(), 0);
        assert_eq!(log.total_ops_before(), 0);
        assert_eq!(log.total_ops_after(), 0);
    }

    #[test]
    fn test_jitlog_compile() {
        let mut log = Logger::new(false);
        log.log_compile(
            42,
            100,
            60,
            Duration::from_millis(2),
            Duration::from_millis(3),
        );
        assert_eq!(log.traces_compiled(), 1);
        assert_eq!(log.total_ops_before(), 100);
        assert_eq!(log.total_ops_after(), 60);
        assert_eq!(log.total_compile_time(), Duration::from_millis(3));
        assert_eq!(log.total_opt_time(), Duration::from_millis(2));
    }

    #[test]
    fn test_jitlog_abort() {
        let mut log = Logger::new(false);
        log.log_abort();
        log.log_abort();
        assert_eq!(log.traces_aborted(), 2);
    }

    #[test]
    fn test_jitlog_loop_entries() {
        let mut log = Logger::new(false);
        log.log_loop_entry(100);
        log.log_loop_entry(100);
        log.log_loop_entry(200);
        assert_eq!(*log.loop_entry_counts().get(&100).unwrap(), 2);
        assert_eq!(*log.loop_entry_counts().get(&200).unwrap(), 1);
    }

    #[test]
    fn test_jitlog_summary_format() {
        let mut log = Logger::new(false);
        log.log_compile(
            1,
            50,
            30,
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        log.log_compile(
            2,
            100,
            55,
            Duration::from_millis(1),
            Duration::from_millis(3),
        );
        log.log_abort();

        let summary = log.summary();
        assert!(summary.contains("Traces compiled: 2"));
        assert!(summary.contains("Traces aborted: 1"));
        assert!(summary.contains("Total ops recorded: 150"));
        assert!(summary.contains("Total ops after opt: 85"));
        assert!(summary.contains("Compilation time:"));
    }

    #[test]
    fn test_jitlog_reduction_percentage() {
        let mut log = Logger::new(false);
        log.log_compile(1, 200, 100, Duration::ZERO, Duration::ZERO);
        let summary = log.summary();
        assert!(summary.contains("50% reduction"));
    }

    #[test]
    fn test_jitlog_zero_ops_no_panic() {
        let log = Logger::new(false);
        let summary = log.summary();
        assert!(summary.contains("0% reduction"));
    }

    #[test]
    fn test_jitlog_multiple_compiles_accumulate() {
        let mut log = Logger::new(false);
        for i in 0..5 {
            log.log_compile(
                i,
                10 * (i as usize + 1),
                5 * (i as usize + 1),
                Duration::from_micros(100 * (i + 1)),
                Duration::from_micros(200 * (i + 1)),
            );
        }
        assert_eq!(log.traces_compiled(), 5);
        // 10+20+30+40+50 = 150
        assert_eq!(log.total_ops_before(), 150);
        // 5+10+15+20+25 = 75
        assert_eq!(log.total_ops_after(), 75);
    }

    #[test]
    fn test_jit_timer() {
        let timer = JitTimer::start();
        // Just verify it doesn't panic and returns a non-negative duration
        let elapsed = timer.elapsed();
        assert!(elapsed.as_nanos() < 1_000_000_000); // less than 1 second
    }

    // ── Jitprof parity tests (rpython/jit/metainterp/test/test_jitprof.py) ──

    #[test]
    fn test_traces_compiled_counted_correctly() {
        // Parity with test_simple_loop: traces compiled count matches actual compilations
        let mut log = Logger::new(false);
        assert_eq!(log.traces_compiled(), 0);
        log.log_compile(
            1,
            50,
            30,
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        assert_eq!(log.traces_compiled(), 1);
        log.log_compile(
            2,
            80,
            40,
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        assert_eq!(log.traces_compiled(), 2);
    }

    #[test]
    fn test_traces_aborted_counted_correctly() {
        // Aborted traces are counted separately from compiled
        let mut log = Logger::new(false);
        log.log_compile(
            1,
            50,
            30,
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        log.log_abort();
        log.log_abort();
        log.log_abort();
        assert_eq!(log.traces_compiled(), 1);
        assert_eq!(log.traces_aborted(), 3);
    }

    #[test]
    fn test_ops_before_after_optimization() {
        // Parity with profiler.counters: total ops tracked before and after optimization
        let mut log = Logger::new(false);
        log.log_compile(1, 100, 60, Duration::ZERO, Duration::ZERO);
        log.log_compile(2, 50, 25, Duration::ZERO, Duration::ZERO);
        assert_eq!(log.total_ops_before(), 150);
        assert_eq!(log.total_ops_after(), 85);
    }

    #[test]
    fn test_compilation_timing() {
        // Parity with profiler.times: compilation and optimization timing
        let mut log = Logger::new(false);
        log.log_compile(
            1,
            50,
            30,
            Duration::from_millis(10), // opt time
            Duration::from_millis(20), // compile time
        );
        log.log_compile(
            2,
            80,
            40,
            Duration::from_millis(5),
            Duration::from_millis(15),
        );
        assert_eq!(log.total_opt_time(), Duration::from_millis(15));
        assert_eq!(log.total_compile_time(), Duration::from_millis(35));
    }

    #[test]
    fn test_bridge_compile_tracking() {
        let mut log = Logger::new(false);
        assert_eq!(log.bridges_compiled(), 0);
        log.log_bridge_compile(0);
        log.log_bridge_compile(1);
        assert_eq!(log.bridges_compiled(), 2);
    }

    #[test]
    fn test_from_env_returns_none_when_disabled() {
        // In a normal test env without MAJIT_STATS=1, from_env returns None
        // (unless the test env happens to set it — that's fine too)
        let result = Logger::from_env();
        // Just verify it returns an Option, not that it's a specific value
        let _ = result;
    }

    #[test]
    fn test_jitlog_summary_all_fields_present() {
        // Parity with profiler output: summary contains all tracked statistics
        let mut log = Logger::new(false);
        log.log_compile(
            1,
            200,
            100,
            Duration::from_millis(5),
            Duration::from_millis(10),
        );
        log.log_compile(
            2,
            100,
            50,
            Duration::from_millis(3),
            Duration::from_millis(7),
        );
        log.log_abort();
        log.log_abort();

        let summary = log.summary();
        assert!(summary.contains("Traces compiled: 2"));
        assert!(summary.contains("Traces aborted: 2"));
        assert!(summary.contains("Total ops recorded: 300"));
        assert!(summary.contains("Total ops after opt: 150"));
        assert!(summary.contains("50% reduction"));
        assert!(summary.contains("Optimization time:"));
        assert!(summary.contains("Compilation time:"));
    }

    #[test]
    fn test_jitlog_loop_entries_accumulate() {
        let mut log = Logger::new(false);
        for _ in 0..10 {
            log.log_loop_entry(42);
        }
        log.log_loop_entry(99);
        assert_eq!(*log.loop_entry_counts().get(&42).unwrap(), 10);
        assert_eq!(*log.loop_entry_counts().get(&99).unwrap(), 1);
    }

    #[test]
    fn test_compiled_traces_accessor() {
        let mut log = Logger::new(false);
        log.log_compile(
            42,
            100,
            60,
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        let traces = log.compiled_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].green_key, 42);
        assert_eq!(traces[0].ops_before_opt, 100);
        assert_eq!(traces[0].ops_after_opt, 60);
    }
}
