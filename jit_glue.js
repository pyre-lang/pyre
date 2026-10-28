// JIT glue layer for majit-backend-wasm.
//
// Called from Rust (via wasm-bindgen) to compile and execute
// dynamically generated wasm modules inside the browser.

let funcTable = {};
let nextFuncId = 1;
let mainMemory = null;
let mainTable = null;

// Call area offsets (must match codegen.rs constants).
const CALL_RESULT_OFS = 2000;
const CALL_FUNC_OFS = 2008;
const CALL_NARGS_OFS = 2016;
const CALL_ARGS_OFS = 2024;

export function jit_set_memory(mem) {
  mainMemory = mem;
}

export function jit_set_table(table) {
  mainTable = table;
}

// Call trampoline — invoked by generated wasm when it encounters a CALL op.
// Reads func_ptr + args from the frame's call area, performs the call via
// the main module's indirect function table, writes result back.
function jitCallTrampoline(framePtr, callAreaOfs = CALL_RESULT_OFS) {
  const view = new DataView(mainMemory.buffer);
  const funcPtrLo = view.getUint32(framePtr + callAreaOfs + 8, true);
  const numArgs = Number(view.getBigInt64(framePtr + callAreaOfs + 16, true));

  // Descr word ABI: each argument slot is an i64 word.
  const args = [];
  for (let i = 0; i < numArgs; i++) {
    args.push(view.getBigInt64(framePtr + callAreaOfs + 24 + i * 8, true));
  }

  // Call via the main module's function table
  let result;
  if (mainTable) {
    try {
      const fn = mainTable.get(funcPtrLo);
      result = fn(...args);
    } catch (e) {
      console.error('[jit_call] trampoline error:', e);
      result = 0n;
    }
  }

  // Write result to the call-area i64 slot.
  if (typeof result === "bigint") {
    view.setBigInt64(framePtr + callAreaOfs, result, true);
  } else if (typeof result === "number") {
    view.setInt32(framePtr + callAreaOfs, result, true);
    view.setInt32(framePtr + callAreaOfs + 4, 0, true);
  } else {
    view.setBigInt64(framePtr + callAreaOfs, 0n, true);
  }
}

// JS cannot reflect a table entry's type portably.
export function jit_func_sig(_slot) { return 0n; }
function jitFuncSig(_slot) { return 0n; }

export function jit_compile_wasm(bytesPtr, bytesLen) {
  const entries = instantiateTrace(bytesPtr, bytesLen);
  return registerTrace(entries);
}

function instantiateTrace(bytesPtr, bytesLen) {
  if (!mainMemory) {
    throw new Error("jit_set_memory() must be called before jit_compile_wasm()");
  }
  const bytes = new Uint8Array(mainMemory.buffer, bytesPtr, bytesLen).slice();
  const module = new WebAssembly.Module(bytes);
  // Check if the module needs jit_call import
  // (wasm-encoder adds it when trace has CALL ops)
  try {
    // `__indirect_function_table` is reserved for inter-trace call_indirect
    // chaining; the module imports it only when it has CALL ops. Extra
    // entries in the import object are ignored when not declared.
    const instance = new WebAssembly.Instance(module, {
      env: { memory: mainMemory, jit_call: jitCallTrampoline, jit_call_compact: jitCallTrampoline, jit_func_sig: jitFuncSig, __indirect_function_table: mainTable }
    });
    return traceEntries(instance);
  } catch (e) {
    // Retry without jit_call (for traces without CALL ops)
    const instance = new WebAssembly.Instance(module, {
      env: { memory: mainMemory }
    });
    return traceEntries(instance);
  }
}

// A resumable peeled loop exports a fixed-arity `trace_wide` beside the narrow
// `trace` shim, and the backend publishes its table slot as `handle + 1`. Both
// hosts must therefore install the pair adjacently or that published slot names
// an unrelated trace.
function traceEntries(instance) {
  return { trace: instance.exports.trace, wide: instance.exports.trace_wide };
}

// Compile and instantiate a trace, then overwrite an existing shared-table
// slot. A caller already running the old function retains that invocation;
// later indirect calls use the replacement.
export function jit_replace_wasm(funcId, bytesPtr, bytesLen) {
  try {
    if (!funcTable[funcId]) {
      return 0;
    }
    const { trace, wide } = instantiateTrace(bytesPtr, bytesLen);
    // Modules emitted while this slot was wide carry `call_indirect funcId + 1`
    // baked in. A narrow replacement cannot retract those, so accepting one
    // would leave the pair straddling two compiles: `funcId` on the new trace
    // and `funcId + 1` still on the old. `funcTable` records the wide entry
    // only when one existed, so it is the discriminator the shared table
    // cannot be — its spare slot holds the narrow function either way. Reject
    // the shape change before touching either table; narrow-to-wide stays
    // allowed, mirroring the wasmtime host.
    if (!wide && funcTable[funcId + 1] !== undefined) {
      console.error('[jit_replace_wasm] refused: id', funcId, 'has a published wide entry the replacement does not');
      return 0;
    }
    if (mainTable) {
      mainTable.set(funcId, trace);
      if (wide) {
        mainTable.set(funcId + 1, wide);
      }
    }
    funcTable[funcId] = trace;
    if (wide) {
      funcTable[funcId + 1] = wide;
    }
    return funcId;
  } catch (e) {
    console.error('[jit_replace_wasm] failed:', e);
    return 0;
  }
}

// Append a compiled trace to the shared indirect function table and use its
// table slot as the id, mirroring the wasmtime host. The slot is both the
// jit_execute_wasm handle and the index an in-module call_indirect targets.
// Falls back to a private counter when no table is available.
function registerTrace(entries) {
  const { trace, wide } = entries;
  let id;
  // The pair is appended even for a narrow module, mirroring the wasmtime
  // host. An emitted module names its wide entry `id + 1`, and a later
  // `jit_replace_wasm` may install one where this compile had none; without
  // the reservation that write would land on the next trace's own entry.
  if (mainTable) {
    id = mainTable.grow(2, trace);
    mainTable.set(id, trace);
    if (wide) {
      mainTable.set(id + 1, wide);
    }
  } else {
    id = nextFuncId;
    nextFuncId += 2;
  }
  funcTable[id] = trace;
  if (wide) {
    funcTable[id + 1] = wide;
  }
  return id;
}

export function jit_execute_wasm(funcId, framePtr) {
  const func = funcTable[funcId];
  if (!func) {
    throw new Error(`jit_execute_wasm: unknown funcId ${funcId}`);
  }
  return func(framePtr);
}

export function jit_free_wasm(funcId) {
  delete funcTable[funcId];
}
