/* tslint:disable */
/* eslint-disable */

/**
 * Browser / JS entry point: marshalled by wasm-bindgen.
 *
 * The browser has one output channel, so the fd-2 text (traceback, warnings)
 * is appended to the returned string rather than handed over separately the
 * way the C-ABI surface does it.
 */
export function run_python(source: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __indirect_function_table: WebAssembly.Table;
    readonly majit_jf_shadow_stack_get_top_jf_ptr: () => bigint;
    readonly majit_jf_shadow_stack_pop_to: (a: bigint) => void;
    readonly majit_jf_shadow_stack_push: (a: bigint) => bigint;
    readonly majit_shadow_stack_pop_and_get: (a: bigint, b: bigint) => bigint;
    readonly majit_shadow_stack_push: (a: bigint) => bigint;
    readonly majit_shadow_stack_set_depth: (a: bigint) => void;
    readonly PyGILState_Ensure: () => number;
    readonly PyGILState_Release: (a: number) => void;
    readonly PyThreadState_SetAsyncExc: (a: number, b: number) => number;
    readonly pyre_stack_check_for_jit_prologue: () => bigint;
    readonly pyre_stack_check_slowpath_for_backend: (a: number) => number;
    readonly pyre_stack_criticalcode_start: () => void;
    readonly pyre_stack_criticalcode_stop: () => void;
    readonly pyre_stack_get_end: () => number;
    readonly pyre_stack_get_end_adr: () => number;
    readonly pyre_stack_get_length: () => number;
    readonly pyre_stack_get_length_adr: () => number;
    readonly pyre_stack_set_length_fraction: (a: number) => void;
    readonly pyre_stack_too_big_slowpath: (a: number) => number;
    readonly run_python: (a: number, b: number) => [number, number];
    readonly lzma_alloc: (a: number, b: number) => number;
    readonly lzma_alloc_zero: (a: number, b: number) => number;
    readonly lzma_alone_decoder: (a: number, b: bigint) => number;
    readonly lzma_alone_encoder: (a: number, b: number) => number;
    readonly lzma_auto_decoder: (a: number, b: bigint, c: number) => number;
    readonly lzma_bcj_arm64_decode: (a: number, b: number, c: number) => number;
    readonly lzma_bcj_arm64_encode: (a: number, b: number, c: number) => number;
    readonly lzma_bcj_riscv_decode: (a: number, b: number, c: number) => number;
    readonly lzma_bcj_riscv_encode: (a: number, b: number, c: number) => number;
    readonly lzma_bcj_x86_decode: (a: number, b: number, c: number) => number;
    readonly lzma_bcj_x86_encode: (a: number, b: number, c: number) => number;
    readonly lzma_block_buffer_bound: (a: number) => number;
    readonly lzma_block_buffer_decode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly lzma_block_buffer_encode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly lzma_block_compressed_size: (a: number, b: bigint) => number;
    readonly lzma_block_decoder: (a: number, b: number) => number;
    readonly lzma_block_encoder: (a: number, b: number) => number;
    readonly lzma_block_header_decode: (a: number, b: number, c: number) => number;
    readonly lzma_block_header_encode: (a: number, b: number) => number;
    readonly lzma_block_header_size: (a: number) => number;
    readonly lzma_block_total_size: (a: number) => bigint;
    readonly lzma_block_uncomp_encode: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly lzma_block_unpadded_size: (a: number) => bigint;
    readonly lzma_check_is_supported: (a: number) => number;
    readonly lzma_check_size: (a: number) => number;
    readonly lzma_code: (a: number, b: number) => number;
    readonly lzma_cputhreads: () => number;
    readonly lzma_crc32: (a: number, b: number, c: number) => number;
    readonly lzma_crc64: (a: number, b: number, c: bigint) => bigint;
    readonly lzma_easy_buffer_encode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly lzma_easy_decoder_memusage: (a: number) => bigint;
    readonly lzma_easy_encoder: (a: number, b: number, c: number) => number;
    readonly lzma_easy_encoder_memusage: (a: number) => bigint;
    readonly lzma_end: (a: number) => void;
    readonly lzma_file_info_decoder: (a: number, b: number, c: bigint, d: bigint) => number;
    readonly lzma_filter_decoder_is_supported: (a: bigint) => number;
    readonly lzma_filter_encoder_is_supported: (a: bigint) => number;
    readonly lzma_filter_flags_decode: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly lzma_filter_flags_encode: (a: number, b: number, c: number, d: number) => number;
    readonly lzma_filter_flags_size: (a: number, b: number) => number;
    readonly lzma_filters_copy: (a: number, b: number, c: number) => number;
    readonly lzma_filters_free: (a: number, b: number) => void;
    readonly lzma_filters_update: (a: number, b: number) => number;
    readonly lzma_free: (a: number, b: number) => void;
    readonly lzma_get_check: (a: number) => number;
    readonly lzma_get_progress: (a: number, b: number, c: number) => void;
    readonly lzma_index_append: (a: number, b: number, c: bigint, d: bigint) => number;
    readonly lzma_index_block_count: (a: number) => bigint;
    readonly lzma_index_buffer_decode: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly lzma_index_buffer_encode: (a: number, b: number, c: number, d: number) => number;
    readonly lzma_index_cat: (a: number, b: number, c: number) => number;
    readonly lzma_index_checks: (a: number) => number;
    readonly lzma_index_decoder: (a: number, b: number, c: bigint) => number;
    readonly lzma_index_dup: (a: number, b: number) => number;
    readonly lzma_index_encoder: (a: number, b: number) => number;
    readonly lzma_index_end: (a: number, b: number) => void;
    readonly lzma_index_file_size: (a: number) => bigint;
    readonly lzma_index_hash_append: (a: number, b: bigint, c: bigint) => number;
    readonly lzma_index_hash_decode: (a: number, b: number, c: number, d: number) => number;
    readonly lzma_index_hash_init: (a: number, b: number) => number;
    readonly lzma_index_hash_size: (a: number) => bigint;
    readonly lzma_index_init: (a: number) => number;
    readonly lzma_index_iter_init: (a: number, b: number) => void;
    readonly lzma_index_iter_locate: (a: number, b: bigint) => number;
    readonly lzma_index_iter_next: (a: number, b: number) => number;
    readonly lzma_index_iter_rewind: (a: number) => void;
    readonly lzma_index_memusage: (a: bigint, b: bigint) => bigint;
    readonly lzma_index_memused: (a: number) => bigint;
    readonly lzma_index_size: (a: number) => bigint;
    readonly lzma_index_stream_count: (a: number) => bigint;
    readonly lzma_index_stream_flags: (a: number, b: number) => number;
    readonly lzma_index_stream_padding: (a: number, b: bigint) => number;
    readonly lzma_index_stream_size: (a: number) => bigint;
    readonly lzma_index_total_size: (a: number) => bigint;
    readonly lzma_index_uncompressed_size: (a: number) => bigint;
    readonly lzma_lzip_decoder: (a: number, b: bigint, c: number) => number;
    readonly lzma_lzma_preset: (a: number, b: number) => number;
    readonly lzma_memlimit_get: (a: number) => bigint;
    readonly lzma_memlimit_set: (a: number, b: bigint) => number;
    readonly lzma_memusage: (a: number) => bigint;
    readonly lzma_mf_is_supported: (a: number) => number;
    readonly lzma_microlzma_decoder: (a: number, b: bigint, c: bigint, d: number, e: number) => number;
    readonly lzma_microlzma_encoder: (a: number, b: number) => number;
    readonly lzma_mode_is_supported: (a: number) => number;
    readonly lzma_mt_block_size: (a: number) => bigint;
    readonly lzma_physmem: () => bigint;
    readonly lzma_properties_decode: (a: number, b: number, c: number, d: number) => number;
    readonly lzma_properties_encode: (a: number, b: number) => number;
    readonly lzma_properties_size: (a: number, b: number) => number;
    readonly lzma_raw_buffer_decode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly lzma_raw_buffer_encode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly lzma_raw_decoder: (a: number, b: number) => number;
    readonly lzma_raw_decoder_memusage: (a: number) => bigint;
    readonly lzma_raw_encoder: (a: number, b: number) => number;
    readonly lzma_raw_encoder_memusage: (a: number) => bigint;
    readonly lzma_str_from_filters: (a: number, b: number, c: number, d: number) => number;
    readonly lzma_str_list_filters: (a: number, b: bigint, c: number, d: number) => number;
    readonly lzma_str_to_filters: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly lzma_stream_buffer_bound: (a: number) => number;
    readonly lzma_stream_buffer_decode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
    readonly lzma_stream_buffer_encode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly lzma_stream_decoder: (a: number, b: bigint, c: number) => number;
    readonly lzma_stream_encoder: (a: number, b: number, c: number) => number;
    readonly lzma_stream_flags_compare: (a: number, b: number) => number;
    readonly lzma_stream_footer_decode: (a: number, b: number) => number;
    readonly lzma_stream_footer_encode: (a: number, b: number) => number;
    readonly lzma_stream_header_decode: (a: number, b: number) => number;
    readonly lzma_stream_header_encode: (a: number, b: number) => number;
    readonly lzma_version_number: () => number;
    readonly lzma_version_string: () => number;
    readonly lzma_vli_decode: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly lzma_vli_encode: (a: bigint, b: number, c: number, d: number, e: number) => number;
    readonly lzma_vli_size: (a: bigint) => number;
    readonly lzma_index_hash_end: (a: number, b: number) => void;
    readonly __wbindgen_exn_store_command_export: (a: number) => void;
    readonly __externref_table_alloc_command_export: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc_command_export: (a: number, b: number) => number;
    readonly __wbindgen_realloc_command_export: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free_command_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
