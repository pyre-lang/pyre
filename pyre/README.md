# pyre

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/2fAUZ49JX3)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/youknowone/pyre?utm_source=badge)

**Experimental** - This project is still in development, and not ready for prime time.

**Py**thon **Re**written — a no-GIL Python implementation in Rust, with a meta-tracing JIT compiler ported from PyPy.

## Why pyre?

PyPy proved that a meta-tracing JIT can make Python fast. pyre takes that proven architecture and rebuilds it in Rust — gaining memory safety, no GIL, and a modern toolchain, while keeping the same optimization pipeline that makes PyPy fast.

The key insight: pyre's JIT framework [MaJIT](majit/) handles tracing, optimization, and native code generation. This means the pyre interpreter itself can stay close to a straightforward Rust program that executes Python bytecodes, while MaJIT provides the tracing JIT machinery around it. In the same way that PyPy is "just a Python interpreter" that RPython makes fast, pyre is "just a Rust interpreter" that MaJIT makes fast.

The deeper goal is to **reproduce RPython in Rust**. RPython's real value was never one specific interpreter but the framework that turns an ordinary interpreter into a fast VM — and MaJIT is that reproduction. PyPy is the most complete language ever built on RPython, so porting PyPy is how we prove and complete MaJIT's reproduction of RPython. pyre is the vehicle; a faithful RPython-in-Rust is the destination.

## Status

pyre is under active development. Loop tracing and function inlining work, and the JIT fires on integer-, float-, and exception-heavy loops alike. On the benchmark set below the native JIT is faster than CPython on all ten programs and no more than 2.0x slower than PyPy; `float_loop` is already faster than PyPy. Many Python features are not yet implemented.

## Benchmarks

The latest full comparison includes both raw user CPU times and relative speed; lower execution time is better.

| Benchmark | CPython 3.14 | PyPy 7.3 | pyre (native) | native JIT vs CPython | native JIT vs PyPy | pyre (wasm) | wasm vs native JIT |
|-----------|--------------|----------|---------------|-----------------------|--------------------|-------------|--------------------|
| int_loop | 8.90s | 0.25s | 0.32s | ~27.8x faster | 1.2x slower | 0.32s | about the same |
| float_loop | 3.79s | 0.25s | 0.22s | ~17.2x faster | 1.25x faster | 0.19s | ~1.16x faster |
| fib_loop | 0.21s | 0.11s | 0.17s | ~1.24x faster | 1.2x slower | 0.19s | ~1.12x slower |
| inline_helper | 5.06s | 0.16s | 0.21s | ~24.1x faster | 1.2x slower | 0.32s | ~1.52x slower |
| fib_recursive | 0.67s | 0.24s | 0.33s | ~2.03x faster | 1.3x slower | 0.69s | ~2.09x slower |
| nested_loop | 8.47s | 0.20s | 0.30s | ~28.2x faster | 1.4x slower | 0.45s | ~1.50x slower |
| raise_catch | 16.81s | 0.51s | 0.59s | ~28.5x faster | 1.1x slower | 0.68s | ~1.15x slower |
| spectral_norm | 3.07s | 0.08s | 0.15s | ~20.5x faster | 1.5x slower | 0.18s | ~1.20x slower |
| nbody | 1.28s | 0.12s | 0.24s | ~5.33x faster | 1.8x slower | 0.37s | ~1.54x slower |
| fannkuch | 1.01s | 0.19s | 0.41s | ~2.46x faster | 2.0x slower | 0.91s | ~2.22x slower |

`dynasm` is the default backend of the `pyre` binary; `cranelift` and `wasm` are the other two MaJIT code generators. Cranelift was not measured in this comparison.

The CPython and wasm/native comparisons are approximate quotients of the displayed, rounded times. The PyPy comparisons use `check.py`'s execution-only ratios, which subtract pypy's measured startup from both sides; a reported 0.8x execution time is therefore written as 1.25x faster, while a reported 1.2x is written as 1.2x slower. Absolute results are machine-dependent, so read a single run as indicative rather than reproducible to the digit.

The native JIT is approximately 1.24x–28.5x faster than CPython across all ten benchmarks. Against PyPy it is 1.25x faster on `float_loop` and 1.1x–2.0x slower on the other nine. Wasm is about even with the native JIT on `int_loop`, approximately 1.16x faster on `float_loop`, and 1.12x–2.22x slower on the other eight. These are small synthetic benchmarks selected to exercise specific JIT paths, not a representative sample of general Python workloads; do not read these ratios as overall application-performance claims.

Run `python3 pyre/check.py --full` to reproduce all benchmarks with CPython / PyPy / pyre comparison on your machine. If the release backend binaries are already built, pass `--build=no` to skip the Cargo build phase; it still checks that `target/llbc/` describes the current tree, because the field offsets the benchmarks measure come from there and a build is what normally asks.

## Installation

### Homebrew

```sh
brew install youknowone/tap/pyrex
```

The formula lives in the [youknowone/homebrew-tap](https://github.com/youknowone/homebrew-tap/tree/main/Formula).

### Prebuilt binaries

Download a prebuilt binary from the [GitHub releases page](https://github.com/youknowone/pyre/releases).

### Cargo

```sh
cargo install pyrex
```

## Building from source

```sh
python3 scripts/install-charon.py
python3 scripts/extract-llbc.py
cargo build --release -p pyrex
./target/release/pyre script.py
```

`install-charon.py` installs the pinned Charon version used to produce the
LLBC artifacts; `extract-llbc.py` generates the artifacts under `target/llbc`
that MaJIT needs during its build.

## How it works

pyre follows PyPy's meta-tracing approach:

1. The **interpreter** (`pyre-interpreter`) executes Python bytecodes normally.
2. When a loop or function becomes hot, **MaJIT** records the interpreter's execution as a linear trace of IR operations.
3. The trace passes through an **8-pass optimizer** — the same pipeline as PyPy: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap.
4. The optimized IR is compiled by one of three MaJIT backends: `dynasm` (the default; x86-64 and aarch64) and Cranelift emit **native machine code**; the third emits **WebAssembly**, which `pyre-wasm-runner` executes under wasmtime by default, or under wasmi.
5. Subsequent executions of that path run the compiled code directly. Guard failures fall back to the interpreter.

### Function inlining

During loop tracing, pyre traces *through* function call boundaries. A call to `add(a, b)` in the loop body becomes `IntAddOvf(a_raw, b_raw)` in the compiled trace — no function call overhead, no frame allocation.

### no-GIL

pyre has no Global Interpreter Lock. RPython/PyPy features that depend on the GIL have no equivalent trigger path in pyre. The API surfaces are kept for naming parity with the original codebase but have no production call sites.

## Crate structure

```
pyre/
├── pyre-object      # Python object types (W_IntObject, W_FloatObject, W_ListObject, ...)
├── pyre-macros      # Proc macros for builtin module declarations (@unwrap_spec equivalent)
├── pyre-native      # Native library backends, kept outside the LLBC extraction
├── pyre-interpreter # Object space, interpreter frame, eval loop, opcode dispatch, bytecode
├── pyre-module      # Optional builtin modules
├── pyre-sandbox     # RPython-style sandbox protocol, virtual filesystem and controller
├── pyre-jit         # JIT compiler integration for the interpreter
├── pyre-jit-trace   # Trace-time JIT — MIFrame and tracing logic
├── pyre-wasm        # The interpreter compiled to WebAssembly
├── pyre-wasm-runner # Native wasmtime host satisfying the JIT host-import contract
├── pyre-wasm-test   # Interpreter-only smoke binary run inside the wasm sandbox
└── pyrex            # Executable entry point (builds the `pyre` binary)
```

## Relationship to PyPy

pyre is a structural port of PyPy's interpreter (`pypy/interpreter/` and `pypy/objspace/`). Every module, type, and function in the original Python codebase exists in the Rust port under the same name at the same relative location — only snake_case conversion is applied to method names. This naming parity makes it possible to read the PyPy source alongside pyre and see exactly what each piece corresponds to.

## Key differences from PyPy

- **Rust-source translation through MaJIT.** RPython translates a live Python
  program image through flowspace, annotator, rtyper, and codewriter. Pyre
  extracts Rust crates to Charon `.ullbc` artifacts and runs their graphs
  through MaJIT's corresponding
  `front → flowspace → annotator → rtyper → codewriter` pipeline at
  `cargo build` time. The explicit `eval::eval_loop_jit` portal seeds the
  ordinary graph closure that becomes assembled JitCodes.
- **Proc macros instead of decorators.** `@jit.elidable` becomes `#[elidable]`, `driver.jit_merge_point(...)` becomes `jit_merge_point!`. Same semantics, Rust syntax.
- **No GIL.** pyre is free-threaded from day one. GIL-dependent code paths in PyPy (heapcache resets on GIL release, `release_gil` effect info, etc.) simply don't exist.
- **Python 3.14, not 2.7.** PyPy's main branch targets Python 2.7/3.10. pyre targets CPython 3.14 bytecodes directly, using RustPython's compiler frontend.

## Relationship to MaJIT

[MaJIT](majit/) (**M**eta-tr**A**cing **JIT**) is a standalone Rust port of RPython's JIT infrastructure. It is a general-purpose framework for Rust bytecode interpreters that integrate with its tracing interface. pyre is MaJIT's primary consumer, but MaJIT has no dependency on pyre.

## Roadmap

What's next, roughly in priority order:

- **Trace exit cost** — `fannkuch` at 2.0x of PyPy is the largest remaining native-JIT benchmark gap; most of what is left is paid on guard failure, transferring state into bridges, rather than inside the compiled loop.
- **Cranelift backend parity** — bring the cranelift backend up to the default dynasm backend's performance and restore it to the full comparison.
- **Broader JIT coverage** — float and exception JIT now fire on the hot-loop benchmarks; extend that coverage to more of the language.
- **More Python built-ins** — str methods, dict operations, list comprehensions, generators.
- **Multi-threaded execution** — the no-GIL foundation is there; actual parallel thread scheduling is not.
- **CPython C extension compatibility** — long-term goal, likely via HPy or similar ABI layer.

## Name

**pyrex** = **pyre** e**x**ecutable. The `pyrex` crate builds the `pyre` command-line binary.

## License

MIT — same as [PyPy](https://github.com/pypy/pypy/blob/main/LICENSE).
