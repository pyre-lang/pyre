//! Native half of the wasm CPU's execution/call transport.
//!
//! Corresponds to `llmodel.py AbstractLLCPU`'s code publication, execution
//! and call boundary. Wasmtime module/table operations are target-specific;
//! interpreters own their imports, I/O and diagnostics, not this ABI.

use std::collections::BTreeSet;

pub use majit_backend_wasm::codegen::{
    CALL_ARGS_OFS, CALL_FUNC_OFS, CALL_NARGS_OFS, CALL_RESULT_OFS, CALL_RESULT_SIZE_OFS,
    CALL_SRET_OFS, MAX_CALL_ARGS, MAX_SRET_BYTES,
};
pub use majit_backend_wasm::{FuncSigVal, WasmSig, decode_func_sig, encode_func_sig};
use wasmtime::error::Context;
use wasmtime::{
    Caller, Error, Extern, Func, Instance, Memory, Module, Ref, Result, Table, Val, ValType,
};

#[derive(Default)]
pub struct TraceState {
    pub memory: Option<Memory>,
    pub table: Option<Table>,
    pub trace_base: u64,
    // Pair-base slots whose spare half is a published `trace_wide`.
    // The table stores a copy of the narrow function in that half when no
    // wide entry exists, so membership here is what forbids a later replace
    // from dropping a published wide target. Keys are pair bases, not wide
    // indices; replace/free must reject a wide index before they rewrite
    // `slot` and `slot + 1`.
    wide_slots: BTreeSet<u32>,
    /// Pair bases `free` cleared. `publish` reuses one before growing.
    free_slots: Vec<u32>,
}

pub trait HostState {
    fn traces(&self) -> &TraceState;
    fn traces_mut(&mut self) -> &mut TraceState;
}

impl HostState for TraceState {
    fn traces(&self) -> &TraceState {
        self
    }
    fn traces_mut(&mut self) -> &mut TraceState {
        self
    }
}

pub fn memory<T: HostState>(caller: &Caller<'_, T>) -> Result<Memory> {
    caller
        .data()
        .traces()
        .memory
        .context("guest memory not initialized")
}

pub fn table<T: HostState>(caller: &Caller<'_, T>) -> Result<Table> {
    caller
        .data()
        .traces()
        .table
        .context("guest function table not initialized")
}

/// Instantiate a compiled trace against the guest's memory and table.
pub fn instantiate<T: HostState + 'static>(
    caller: &mut Caller<'_, T>,
    module: &Module,
    call: fn(&mut Caller<'_, T>, u32, u32) -> Result<()>,
) -> Result<(Func, Option<Func>, Instance)> {
    let memory = memory(caller)?;
    let table = table(caller)?;
    let jit_call = Func::wrap(
        &mut *caller,
        move |mut caller: Caller<'_, T>, frame: i32| {
            if let Err(error) = call(&mut caller, frame as u32, CALL_RESULT_OFS as u32) {
                eprintln!("[jit_call] {error:?}");
            }
        },
    );
    let jit_call_compact = Func::wrap(
        &mut *caller,
        move |mut caller: Caller<'_, T>, frame: i32, offset: i32| {
            if let Err(error) = call(&mut caller, frame as u32, offset as u32) {
                eprintln!("[jit_call_compact] {error:?}");
            }
        },
    );
    let jit_func_sig = Func::wrap(
        &mut *caller,
        |mut caller: Caller<'_, T>, slot: i32| -> i64 { jit_func_sig_of_slot(&mut caller, slot) },
    );
    let mut imports = Vec::new();
    for import in module.imports() {
        imports.push(match (import.module(), import.name()) {
            ("env", "memory") => Extern::Memory(memory),
            ("env", "__indirect_function_table") => Extern::Table(table),
            ("env", "jit_call") => Extern::Func(jit_call),
            ("env", "jit_call_compact") => Extern::Func(jit_call_compact),
            ("env", "jit_func_sig") => Extern::Func(jit_func_sig),
            (m, n) => return Err(Error::msg(format!("unexpected trace import {m}.{n}"))),
        });
    }
    let instance = Instance::new(&mut *caller, module, &imports)?;
    let trace = instance
        .get_func(&mut *caller, "trace")
        .context("trace export missing")?;
    let wide = instance.get_func(&mut *caller, "trace_wide");
    Ok((trace, wide, instance))
}

/// Every trace reserves a pair, including a narrow trace that may become wide.
pub fn publish<T: HostState>(
    caller: &mut Caller<'_, T>,
    trace: Func,
    wide: Option<Func>,
) -> Result<u32> {
    if let Some(slot) = caller.data_mut().traces_mut().free_slots.pop() {
        let table = table(caller)?;
        table.set(&mut *caller, slot as u64, Ref::Func(Some(trace)))?;
        if let Some(wide) = wide {
            table.set(&mut *caller, slot as u64 + 1, Ref::Func(Some(wide)))?;
            caller.data_mut().traces_mut().wide_slots.insert(slot);
        }
        return Ok(slot);
    }
    let table = table(caller)?;
    let slot = u32::try_from(table.grow(&mut *caller, 2, Ref::Func(Some(trace)))?)?;
    if let Some(wide) = wide {
        table.set(&mut *caller, slot as u64 + 1, Ref::Func(Some(wide)))?;
        caller.data_mut().traces_mut().wide_slots.insert(slot);
    }
    Ok(slot)
}

pub fn replace<T: HostState>(
    caller: &mut Caller<'_, T>,
    slot: u32,
    trace: Func,
    wide: Option<Func>,
) -> Result<u32> {
    let table = table(caller)?;
    pair_base(caller, slot)?;
    live_trace(caller, slot)?;
    if wide.is_none() && caller.data().traces().wide_slots.contains(&slot) {
        return Err(Error::msg("replacement would drop a published wide entry"));
    }
    // Store keeps an old instance alive while an active guest frame uses it.
    table.set(&mut *caller, slot as u64, Ref::Func(Some(trace)))?;
    if let Some(wide) = wide {
        table.set(&mut *caller, slot as u64 + 1, Ref::Func(Some(wide)))?;
        caller.data_mut().traces_mut().wide_slots.insert(slot);
    }
    Ok(slot)
}

/// The published handle is the even index of a reserved pair.
///
/// `publish` grows the table by two. `execute` may call `slot + 1` as the
/// wide entry, so that index is a live function, but `replace`/`free` rewrite
/// or clear both halves. Treating a wide index as a handle would write past
/// the last pair or clear the next pair's narrow function. `live_trace`
/// cannot tell those apart. `execute` does not use this: calling a wide
/// entry is the point of the spare slot.
fn pair_base<T: HostState>(caller: &Caller<'_, T>, slot: u32) -> Result<u32> {
    let traces = caller.data().traces();
    if (slot as u64) < traces.trace_base {
        return Err(Error::msg(format!(
            "slot {slot} belongs to the guest, not a trace"
        )));
    }
    let size = table(caller)?.size(caller);
    if (slot as u64) + 1 >= size || (slot as u64 - traces.trace_base) % 2 != 0 {
        return Err(Error::msg(format!("slot {slot} is not a trace pair base")));
    }
    Ok(slot)
}

fn live_trace<T: HostState>(caller: &mut Caller<'_, T>, slot: u32) -> Result<Func> {
    if (slot as u64) < caller.data().traces().trace_base {
        return Err(Error::msg(format!(
            "slot {slot} belongs to the guest, not a trace"
        )));
    }
    match table(caller)?.get(&mut *caller, slot as u64) {
        Some(Ref::Func(Some(trace))) => Ok(trace),
        _ => Err(Error::msg(format!("trace {slot} is not live"))),
    }
}

pub fn execute<T: HostState>(caller: &mut Caller<'_, T>, slot: u32, frame: u32) -> Result<u32> {
    let trace = live_trace(caller, slot)?;
    let mut result = [Val::I32(0)];
    trace.call(&mut *caller, &[Val::I32(frame as i32)], &mut result)?;
    match result[0] {
        Val::I32(value) => Ok(value as u32),
        _ => Err(Error::msg("trace returned a non-i32 result")),
    }
}

pub fn free<T: HostState>(caller: &mut Caller<'_, T>, slot: u32) -> Result<()> {
    if (slot as u64) < caller.data().traces().trace_base {
        return Ok(());
    }
    pair_base(caller, slot)?;
    let table = table(caller)?;
    table.set(&mut *caller, slot as u64, Ref::Func(None))?;
    table.set(&mut *caller, slot as u64 + 1, Ref::Func(None))?;
    let traces = caller.data_mut().traces_mut();
    traces.wide_slots.remove(&slot);
    traces.free_slots.push(slot);
    Ok(())
}

/// Caller records a histogram or a missing-slot diagnostic without owning ABI decoding.
pub fn residual_call<T: HostState>(
    caller: &mut Caller<'_, T>,
    frame: u32,
    offset: u32,
    mut observe: impl FnMut(u32, bool),
) -> Result<()> {
    let memory = memory(caller)?;
    let table = table(caller)?;
    let area = frame as usize + offset as usize;
    let mut word = [0; 8];
    let mut slot_bytes = [0; 4];
    memory.read(
        &*caller,
        area + (CALL_FUNC_OFS - CALL_RESULT_OFS) as usize,
        &mut slot_bytes,
    )?;
    let slot = u32::from_le_bytes(slot_bytes);
    let func = match table.get(&mut *caller, slot as u64) {
        Some(Ref::Func(Some(func))) if slot != 0 => Some(func),
        _ => None,
    };
    observe(slot, func.is_some());
    let Some(func) = func else {
        memory.write(&mut *caller, area, &0_i64.to_le_bytes())?;
        return Ok(());
    };
    let ty = func.ty(&*caller);
    memory.read(
        &*caller,
        area + (CALL_NARGS_OFS - CALL_RESULT_OFS) as usize,
        &mut word,
    )?;
    let nargs = i64::from_le_bytes(word).max(0) as usize;
    memory.read(
        &*caller,
        area + (CALL_RESULT_SIZE_OFS - CALL_RESULT_OFS) as usize,
        &mut word,
    )?;
    let result_size = i64::from_le_bytes(word).max(0) as usize;
    let nparams = ty.params().len();
    // `llmodel.py AbstractLLCPU.bh_call_*` takes the result from the call
    // descr (`result_size`), not from the callee's machine signature. A
    // void wasm type with a non-zero descr size is an sret of that size.
    let sret = ty.results().len() == 0 && result_size > 0;
    let logical = if sret { nargs } else { nparams };
    if logical > MAX_CALL_ARGS {
        return Err(Error::msg("residual argument area overflow"));
    }
    if sret {
        if result_size > MAX_SRET_BYTES {
            return Err(Error::msg(format!(
                "residual sret result_size {result_size} is an aggregate the call descr cannot describe"
            )));
        }
        if nparams != nargs + 1 {
            return Err(Error::msg(format!(
                "residual sret param count {nparams} != nargs {nargs} + 1 \
                 (slot {slot}, result_size {result_size}, params {:?}, results {:?})",
                ty.params().collect::<Vec<_>>(),
                ty.results().collect::<Vec<_>>(),
            )));
        }
    } else if nargs != 0 && nargs != nparams {
        return Err(Error::msg(format!(
            "residual param count {nparams} != nargs {nargs}"
        )));
    }
    let sret_addr = area + (CALL_SRET_OFS - CALL_RESULT_OFS) as usize;
    if sret {
        let sret_zero = [0u8; MAX_SRET_BYTES];
        memory.write(&mut *caller, sret_addr, &sret_zero[..result_size])?;
    }
    let mut args = Vec::with_capacity(nparams);
    if sret {
        args.push(Val::I32(sret_addr as i32));
    }
    for i in 0..logical {
        let param_ty = ty.params().nth(if sret { i + 1 } else { i }).unwrap();
        memory.read(
            &*caller,
            area + (CALL_ARGS_OFS - CALL_RESULT_OFS) as usize + i * 8,
            &mut word,
        )?;
        let raw = i64::from_le_bytes(word);
        args.push(match param_ty {
            ValType::I32 => Val::I32(raw as i32),
            ValType::I64 => Val::I64(raw),
            ValType::F32 => Val::F32(raw as u32),
            ValType::F64 => Val::F64(raw as u64),
            other => return Err(Error::msg(format!("unsupported residual param {other:?}"))),
        });
    }
    if args.len() != nparams {
        return Err(Error::msg(format!(
            "residual assembled {} args, wasm type has {nparams}",
            args.len()
        )));
    }
    let mut results: Vec<Val> = ty
        .results()
        .map(|ty| match ty {
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(0),
            ValType::F64 => Val::F64(0),
            _ => Val::I32(0),
        })
        .collect();
    // Existing wasm/browser transport contract: report a trap and return zero.
    let result = match func.call(&mut *caller, &args, &mut results) {
        Err(error) => {
            eprintln!("[jit_call] residual target trapped: {error:?}");
            0
        }
        Ok(()) if sret => {
            let mut ret = [0u8; 8];
            memory.read(&*caller, sret_addr, &mut ret[..result_size])?;
            i64::from_le_bytes(ret)
        }
        Ok(()) => match results.first() {
            Some(Val::I32(v)) => (*v as u32) as i64,
            Some(Val::I64(v)) => *v,
            Some(Val::F32(v)) => *v as i64,
            Some(Val::F64(v)) => *v as i64,
            _ => 0,
        },
    };
    memory.write(&mut *caller, area, &result.to_le_bytes())?;
    Ok(())
}

/// `env.jit_func_sig(slot) -> i64` — the callee's packed wasm type, or 0.
pub fn jit_func_sig_of_slot<T: HostState>(caller: &mut Caller<'_, T>, slot: i32) -> i64 {
    if slot == 0 {
        return 0;
    }
    let Ok(table) = table(caller) else {
        return 0;
    };
    let func = match table.get(&mut *caller, slot as u64) {
        Some(Ref::Func(Some(func))) => func,
        _ => return 0,
    };
    encode_functype(&func.ty(&*caller))
}

fn encode_functype(ty: &wasmtime::FuncType) -> i64 {
    if ty.params().len() > MAX_CALL_ARGS || ty.results().len() > 1 {
        return 0;
    }
    let mut params = Vec::with_capacity(ty.params().len());
    for param in ty.params() {
        params.push(match param {
            ValType::I32 => FuncSigVal::I32,
            ValType::I64 => FuncSigVal::I64,
            ValType::F32 => FuncSigVal::F32,
            ValType::F64 => FuncSigVal::F64,
            _ => return 0,
        });
    }
    let result = match ty.results().next() {
        None => None,
        Some(ValType::I32) => Some(FuncSigVal::I32),
        Some(ValType::I64) => Some(FuncSigVal::I64),
        Some(ValType::F32) => Some(FuncSigVal::F32),
        Some(ValType::F64) => Some(FuncSigVal::F64),
        _ => return 0,
    };
    encode_func_sig(&params, result)
}
