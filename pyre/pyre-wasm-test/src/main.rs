use pyre_interpreter::*;
use std::cell::RefCell;

thread_local! {
    static TEST_OUTPUT: RefCell<String> = const { RefCell::new(String::new()) };
}

fn capture_print(bytes: &[u8]) {
    TEST_OUTPUT.with(|buf| buf.borrow_mut().push_str(&String::from_utf8_lossy(bytes)));
}

fn run_test(name: &str, source: &str, expected: &str) {
    TEST_OUTPUT.with(|buf| buf.borrow_mut().clear());
    pyre_interpreter::set_print_hook(capture_print);
    pyre_module::register();
    pyre_interpreter::importing::install_builtin_modules();

    let code = match compile_source(source, Mode::Exec) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("[FAIL] {name}: SyntaxError: {e}");
            return;
        }
    };

    let execution_context = pyre_interpreter::alloc_execution_context();
    // `threadlocals.py enter_thread` — the ExecutionContext slot belongs to the
    // OS-thread locals, and each launcher installs it before running anything.
    // Left null, `eval_frame_plain_with_resume` takes its context-free arm: no
    // `enter`/`leave`, no `call_trace`, so `sys.settrace` is silently inert.
    pyre_interpreter::call::set_last_exec_ctx(execution_context.as_ptr());
    let mut frame = match pyframe::PyFrame::new_with_context(code, execution_context) {
        Ok(frame) => frame,
        Err(e) => {
            eprintln!("[FAIL] {name}: Error: {e}");
            return;
        }
    };

    // Use plain interpreter on WASI (no JS runtime for JIT execute).
    // The JIT compile path is tested separately.
    match frame.execute_frame(None) {
        Ok(_) => {
            let actual = TEST_OUTPUT.with(|buf| buf.borrow().trim_end().to_string());
            if actual == expected {
                eprintln!("[PASS] {name}");
            } else {
                eprintln!("[FAIL] {name}: expected '{expected}', got '{actual}'");
            }
        }
        Err(e) => {
            eprintln!("[FAIL] {name}: Error: {e}");
        }
    }
}

fn main() {
    eprintln!("=== pyre wasm interpreter test suite ===\n");

    // Basic arithmetic
    run_test("int_add", "print(1 + 2)", "3");
    run_test("int_sub", "print(10 - 3)", "7");
    run_test("int_mul", "print(6 * 7)", "42");
    run_test("int_div", "print(15 // 4)", "3");
    run_test("int_mod", "print(17 % 5)", "2");
    run_test("int_neg", "print(-42)", "-42");
    run_test("int_expr", "print(2 + 3 * 4 - 1)", "13");

    // Comparisons
    run_test("cmp_lt", "print(1 < 2)", "True");
    run_test("cmp_ge", "print(5 >= 5)", "True");
    run_test("cmp_eq", "print(3 == 4)", "False");

    // Boolean
    run_test("bool_and", "print(True and False)", "False");
    run_test("bool_or", "print(False or True)", "True");

    // While loop
    run_test(
        "while_sum",
        "s = 0\ni = 0\nwhile i < 100:\n    s = s + i\n    i = i + 1\nprint(s)",
        "4950",
    );

    // fib_loop
    run_test(
        "fib_loop_10",
        "def fib(n):\n    a, b = 0, 1\n    i = 0\n    while i < n:\n        a, b = b, a + b\n        i = i + 1\n    return a\nprint(fib(10))",
        "55",
    );

    run_test(
        "fib_loop_30",
        "def fib(n):\n    a, b = 0, 1\n    i = 0\n    while i < n:\n        a, b = b, a + b\n        i = i + 1\n    return a\nprint(fib(30))",
        "832040",
    );

    // Nested loops
    run_test(
        "nested_loop",
        "s = 0\ni = 0\nwhile i < 10:\n    j = 0\n    while j < 10:\n        s = s + 1\n        j = j + 1\n    i = i + 1\nprint(s)",
        "100",
    );

    // Float arithmetic
    run_test("float_add", "print(1.5 + 2.5)", "4.0");
    run_test("float_mul", "print(3.0 * 2.5)", "7.5");

    // String basics
    run_test(
        "str_concat",
        "print('hello' + ' ' + 'world')",
        "hello world",
    );
    run_test("str_len", "print(len('abc'))", "3");

    // List basics
    run_test("list_basic", "x = [1, 2, 3]\nprint(len(x))", "3");

    // Function calls
    run_test(
        "func_call",
        "def add(a, b):\n    return a + b\nprint(add(3, 4))",
        "7",
    );

    // Multiple prints
    run_test("multi_print", "print(1)\nprint(2)\nprint(3)", "1\n2\n3");

    // Countdown
    run_test(
        "countdown",
        "n = 10\nwhile n > 0:\n    n = n - 1\nprint(n)",
        "0",
    );

    // fib_recursive (interpreter only, no JIT)
    run_test(
        "fib_recursive_10",
        "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nprint(fib(10))",
        "55",
    );

    eprintln!("\n=== done ===");
}
