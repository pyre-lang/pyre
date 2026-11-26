//! Internal `PYRE_GIL=0` switch: `sys._is_gil_enabled`, `sys.flags.gil`,
//! threaded allocation, and the JIT-off eval path.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// The dynasm-backend `pyre` binary cargo built for this test.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const GIL_STATUS: &str = "import sys; print(sys._is_gil_enabled(), sys.flags.gil)\n";

const THREADED_ALLOC: &str = r#"
import threading

def work(tid):
    acc = 0
    items = []
    for i in range(20000):
        s = str(i)
        t = (tid, i)
        items.append((i, s, t))
        acc += i + len(s) + t[0] + t[1]
    return acc

expected = 0
for i in range(4):
    expected += work(i)
res = [None] * 4

def run(i):
    res[i] = work(i)

threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(sum(res), expected)
"#;

// A function-entry door and a back-edge door (jd0), and the `tuple.hash`
// driver's loop over a 40-item tuple.
const JIT_LOOP: &str = r#"
def f(n):
    return n + 1

s = 0
for _ in range(3000):
    s = f(s)
t = tuple(range(40))
h = 0
for _ in range(3000):
    h ^= hash(t)
print(s)
"#;

fn pyre_command() -> Command {
    let mut command = Command::new(PYRE);
    for flag in [
        "MAJIT_STATS",
        "MAJIT_LOG",
        "PYRE_MC_DIAG",
        "PYRE_GC_DIAG",
        "PYRE_FBW_SPEC_CENSUS",
        "PYRE_CELL_CENSUS",
        "PYRE_FBW_DEPTH_CENSUS",
        "PYRE_GIL",
    ] {
        command.env_remove(flag);
    }
    command
}

fn stdout_of(args: &[&str], env: &[(&str, &str)]) -> String {
    let mut command = pyre_command();
    command.args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let out = command.output().expect("spawn pyre-dynasm");
    report_success(&out)
}

fn stdout_of_source_timeout(name: &str, source: &str, env: &[(&str, &str)], secs: u64) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(name);
    std::fs::write(&path, source).expect("write the program under test");
    let mut command = pyre_command();
    command.arg(&path);
    for (key, value) in env {
        command.env(key, value);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn pyre-dynasm");
    let deadline = Instant::now() + Duration::from_secs(secs);
    while child.try_wait().expect("query child status").is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap hung child");
            panic!(
                "timed out after {secs}s\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let out = child.wait_with_output().expect("reap child");
    report_success(&out)
}

fn report_success(out: &Output) -> String {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let report = format!(
        "exited {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        out.status,
        stderr.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
    );
    assert!(!stderr.contains("panicked at"), "{report}");
    assert!(out.status.success(), "{report}");
    stdout.trim().to_string()
}

#[test]
fn default_env_reports_gil_enabled() {
    assert_eq!(stdout_of(&["-c", GIL_STATUS], &[]), "True 1");
}

#[test]
fn pyre_gil_zero_reports_gil_disabled() {
    assert_eq!(
        stdout_of(&["-c", GIL_STATUS], &[("PYRE_GIL", "0")]),
        "False 0"
    );
}

#[test]
fn pyre_gil_zero_threaded_alloc_checksum_matches_serial() {
    let out = stdout_of_source_timeout(
        "gil_env_switch_threaded.py",
        THREADED_ALLOC,
        &[("PYRE_GIL", "0")],
        120,
    );
    let mut parts = out.split_whitespace();
    let threaded = parts.next().expect("threaded sum");
    let serial = parts.next().expect("serial sum");
    assert_eq!(threaded, serial, "{out}");
    assert!(parts.next().is_none(), "{out}");
}

#[cfg(unix)]
#[test]
fn pyre_gil_zero_serializes_signal_actions_with_worker_tickers() {
    let source = r#"
import signal
import threading
import time

handled = 0
results = [None] * 4

def handle_usr1(signum, frame):
    global handled
    handled += 1

def run(tid):
    x = 0
    for _ in range(2_000_000):
        x += 1
    results[tid] = x

workers = [threading.Thread(target=run, args=(tid,)) for tid in range(4)]
for worker in workers:
    worker.start()
signal.signal(signal.SIGUSR1, handle_usr1)
for _ in range(200):
    signal.raise_signal(signal.SIGUSR1)
    time.sleep(0.001)
for worker in workers:
    worker.join()
print(handled, sum(results))
"#;
    assert_eq!(
        stdout_of_source_timeout(
            "gil_env_switch_signal_actions.py",
            source,
            &[("PYRE_GIL", "0")],
            120,
        ),
        "200 8000000"
    );
}

#[cfg(unix)]
#[test]
fn pyre_gil_zero_signal_handler_does_not_block_worker_ticker() {
    let source = r#"
import os
import signal
import threading

started = False
done = False

def handle_usr1(signum, frame):
    global started
    started = True
    while not done:
        pass

def run():
    global done
    while not started:
        pass
    x = 0
    for _ in range(300_000):
        x += 1
    done = True

signal.signal(signal.SIGUSR1, handle_usr1)
worker = threading.Thread(target=run)
worker.start()
os.kill(os.getpid(), signal.SIGUSR1)
worker.join()
print("ok")
"#;
    assert_eq!(
        stdout_of_source_timeout(
            "gil_env_switch_signal_handler_worker_ticker.py",
            source,
            &[("PYRE_GIL", "0")],
            60,
        ),
        "ok"
    );
}

#[cfg(unix)]
#[test]
fn pyre_gil_zero_signal_action_runs_while_worker_finalizer_blocks() {
    let source = r#"
import gc
import os
import signal
import threading
import time

ev = threading.Event()
finalizer_started = False
finalizer_on_worker = False
wait_result = None
worker = None
worker_ready = False

class C:
    def __del__(self):
        global finalizer_started, finalizer_on_worker, wait_result
        finalizer_on_worker = threading.current_thread() is worker
        finalizer_started = True
        wait_result = ev.wait(30)

def handle_usr1(signum, frame):
    ev.set()

def run():
    global worker_ready
    gc.collect()
    worker_ready = True
    time.sleep(0.1)
    while not finalizer_started:
        obj = C()
        obj.cycle = obj
        del obj
        gc.collect_step()

signal.signal(signal.SIGUSR1, handle_usr1)
worker = threading.Thread(target=run)
worker.start()
while not worker_ready:
    time.sleep(0.001)
time.sleep(2)
while not finalizer_started:
    time.sleep(0.001)
os.kill(os.getpid(), signal.SIGUSR1)
worker.join()
print(wait_result if finalizer_on_worker else "wrong-thread")
"#;
    assert_eq!(
        stdout_of_source_timeout(
            "gil_env_switch_signal_during_worker_finalizer.py",
            source,
            &[("PYRE_GIL", "0")],
            60,
        ),
        "True"
    );
}

#[test]
fn pyre_gil_zero_workers_run_finalizers_concurrently() {
    // Plain flags polled with `time.sleep`, not `threading.Event`: a finalizer
    // may be dispatched on a thread that is inside `Event.wait` holding the
    // condition's lock, and an `Event.set` in that finalizer would then wait
    // on its own thread.
    let source = r#"
import gc
import threading
import time

state = {"started": False, "released": False, "waiter_dropped": False,
         "setter_dropped": False}
results = []

def wait_for(key, seconds=50):
    deadline = time.monotonic() + seconds
    while not state[key]:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True

class Waiter:
    def __del__(self):
        state["started"] = True
        results.append(wait_for("released"))

class Setter:
    def __del__(self):
        state["released"] = True

def wait_in_finalizer():
    obj = Waiter()
    obj.cycle = obj
    del obj
    state["waiter_dropped"] = True
    if not wait_for("setter_dropped"):
        return
    gc.collect()

def release_other_finalizer():
    if not wait_for("waiter_dropped"):
        return
    obj = Setter()
    obj.cycle = obj
    del obj
    state["setter_dropped"] = True
    if not wait_for("started"):
        return
    gc.collect()

workers = [
    threading.Thread(target=wait_in_finalizer),
    threading.Thread(target=release_other_finalizer),
]
for worker in workers:
    worker.start()
for worker in workers:
    worker.join()
if results == [True] and state["released"]:
    print("ok")
"#;
    assert_eq!(
        stdout_of_source_timeout(
            "gil_env_switch_concurrent_finalizers.py",
            source,
            &[("PYRE_GIL", "0")],
            60,
        ),
        "ok"
    );
}

/// `loops_compiled=` from the `MAJIT_STATS=1` summary on stderr.
fn loops_compiled_running(source_name: &str, source: &str, env: &[(&str, &str)]) -> u64 {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(source_name);
    std::fs::write(&path, source).expect("write the program under test");
    let mut command = pyre_command();
    command.arg(&path).env("MAJIT_STATS", "1");
    for (key, value) in env {
        command.env(key, value);
    }
    let out = command.output().expect("spawn pyre-dynasm");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(report_success(&out), "3000");
    stderr
        .split_whitespace()
        .find_map(|word| word.strip_prefix("loops_compiled="))
        .unwrap_or_else(|| panic!("no loops_compiled= in stderr:\n{stderr}"))
        .parse()
        .expect("loops_compiled count")
}

#[test]
fn pyre_gil_zero_compiles_no_loop() {
    // The same program compiles with the GIL on, so a zero below is the
    // off switch and not a program too cold to trace.
    assert!(loops_compiled_running("gil_env_switch_jit_loop_on.py", JIT_LOOP, &[]) > 0);
    assert_eq!(
        loops_compiled_running(
            "gil_env_switch_jit_loop_off.py",
            JIT_LOOP,
            &[("PYRE_GIL", "0")]
        ),
        0
    );
}
