# Eager IR compilation

`majit_backend::eager::CompiledIr` (also `majit::eager::CompiledIr`) submits
backend-ready IR immediately, without a recorder, sample execution, hot counter,
or `MetaInterp`. Backend selection stays with the embedding: the same entry point
accepts `&mut dyn Backend` for dynasm, Cranelift, or another implementation.

This is an in-process compilation API, not an object-file writer, CEL compiler,
generic CFG compiler, or automatic substitute for the tracing frontend. A
frontend still owns language lowering, optimization if wanted, typed operands,
call descriptors, guards, and control-flow targets. Backend support for an IR
operation is unchanged; backend compilation errors propagate to the caller.

## Lifecycle

1. Configure the backend's existing runtime, GC and completion/exception
   descriptors, then call its ordinary `setup_once`. A backend already configured
   by the embedding needs no second setup. `CompiledIr` never installs a different
   collector, reattaches descriptors, or calls setup/teardown behind the caller.
2. Supply a fresh `Arc<JitCellToken>` from the embedding's existing unique token
   namespace, input arguments in call order, and backend-ready operations.
   `unsafe CompiledIr::compile(...)` compiles immediately. A caller
   supplying raw machine-level IR is responsible for its validity; the API is
   not a verifier for untrusted IR.
3. Reuse `unsafe compiled.execute(&values)` for different inputs. Arity and types
   are checked before entry. `RawExecResult` preserves finish/guard/exception
   classification and exit values. A guard is not silently resumed, retraced or
   reported as a successful language evaluation. Pointer inputs and returned
   references remain subject to the embedding's GC/rooting contract.
4. The handle keeps the token alive and exclusively borrows the backend; it
   cannot move to another execution thread. Drop it before reconfiguring or
   using that backend directly. Retain an `Arc` clone of the token to use the
   existing `compile_bridge`, `execute_token`, or invalidation APIs afterward.
   Backend and runtime teardown remain the embedding's responsibility. A
   thread-confined boxed collector installed through
   `CraneliftBackend::set_gc_allocator` / `DynasmBackend::set_gc_allocator`
   is released with that backend's `clear_gc_allocator`; dropping the
   backend value does not do it.

Do not interleave a partially staged lower-level compilation with this API.
A backend may partially populate a token before returning an error; retry
with a fresh token, not the failed token.

Standalone embeddings can share the metainterpreter's descriptor factory:

```rust,ignore
majit_backend::make_and_attach_done_descrs(&mut [cpu as &mut dyn Backend]);
cpu.set_propagate_exception_descr(Arc::new(majit_backend::PropagateExceptionDescr::new()));
cpu.setup_once();
```

Use this only when initially configuring a CPU, not before each compilation.
The factory creates the five completion/exception descriptors and shares their
identities across its targets; it does not install a collector, replace the
propagate-exception descriptor, or run setup. The metainterpreter's original
`compile::make_and_attach_done_descrs` signature remains as a forwarding wrapper.

See the module's compiled rustdoc example for an integer `add_one` unit. Native
conformance tests use directly constructed `Op`/`Operand` graphs, not even a
test recorder, and cover changing inputs, guard exits, mixed argument banks,
scalar result banks, void results, rejected arguments, host calls, and reuse
through the pre-existing backend API.

## Cost and scope

The execution path still uses `execute_token_raw`: the ordinary JITFRAME entry
and result decoding remain. Eager compilation removes warmup and runtime trace
discovery; this API does **not** claim to remove the measured per-call entry
cost, binding cost, GC overhead, or actual host-function invocation cost. A
dedicated low-overhead scalar ABI is separate work.

The `majit` example `eager_vs_jit` probes that boundary using the same compiled
token through `CompiledIr::execute` and `Backend::execute_token_raw`:

```sh
cargo run --release -p majit --no-default-features --features dynasm --example eager_vs_jit
```

Despite its historical name, it does not run a tracing frontend or compare AOT
against JIT code generation. It reports one cold compilation, the first checked
call, and seven-sample steady-state wall-clock minima for checked and raw entry.
Inputs vary and integer overflow results are checked outside the timed loops.
The panels run sequentially, so drift and timer noise prevent treating their
difference as an exact cost decomposition. Both retain ordinary JITFRAME entry.
Use `EAGER_BENCH_ITERS` to reduce iterations for a smoke run; use external power
and the existing thread-CPU benchmark board for performance conclusions.

For a CEL frontend, short-circuiting, errors, dynamic types and host side effects
must survive lowering. Compiling an example input and treating the observed path
as the whole program is not a valid implementation of this API's frontend.

## Compatibility boundary / parity review

This is a user-authorized embedding extension with no new counterpart module in
RPython. The underlying contract is `rpython/jit/backend/model.py`
`AbstractCPU.compile_loop` / `execute_token`; upstream's
`rpython/jit/backend/test/runner_test.py` `Runner.execute_operations` likewise
compiles supplied operations without a metainterp. This is not limited to
upstream tests: `rpython/jit/metainterp/compile.py` `compile_tmp_callback` directly
assembles a call/exception-guard/finish sequence through `cpu.compile_loop` too.

The descriptor factory and its `DescrContainer` contract now live beside the
existing descriptor classes in `majit-backend::finish_descrs`; their upstream
owner remains `compile.py` `make_and_attach_done_descrs`. This crate-placement
adaptation allows standalone embeddings to reuse them without depending on the
metainterpreter. Dynasm's existing test initializer also uses this factory.

Existing
`Backend` methods and concrete code generators, PyPy interpreter generation,
optimizer passes, GC layout, token identity, guard descriptors and resume
machinery are unchanged by this API addition. Pre-existing worktree changes in
the Cranelift entry allocator and JitDriver fixture are not part of this addition.

Native conformance command:

```sh
cargo test -p majit-backend-dynasm -p majit-backend-cranelift --test eager --no-default-features
cargo test -p majit-backend --doc --no-default-features
```

The shared API has no native-backend feature gate. This test command executes
the host's native backends; it does not establish wasm guest execution support
for a new frontend. Wasm retains its existing host-import/browser runtime setup
and operation support contract.
