#!/usr/bin/env bash
#
# Build the browser (web / wasm-bindgen) flavour of pyre-wasm and stage the
# artefacts that www/index.html loads.
#
# Both the `web` and `wasm-host` builds of this single crate emit to the shared
# target/wasm32-unknown-unknown/release/pyre_wasm.wasm path, so a later build
# of the other flavour (or check.py, which builds wasm-host) overwrites it. This
# script snapshots the web output to a feature-distinct pyre_wasm.web.wasm
# immediately and feeds that snapshot to wasm-bindgen, so the deployed module
# never depends on the shared path surviving. (check.py does the mirror image
# for wasm-host -> pyre_wasm.wasm-host.wasm.)
#
# Requires: the wasm32-unknown-unknown rustup target and wasm-bindgen-cli
# (`cargo install wasm-bindgen-cli`, matching the crate's wasm-bindgen version).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

target_dir="target/wasm32-unknown-unknown/release"
raw_wasm="$target_dir/pyre_wasm.wasm"
web_wasm="$target_dir/pyre_wasm.web.wasm"
www_dir="pyre/pyre-wasm/www"
glue_src="majit/majit-backend-wasm/js/jit_glue.js"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
    echo "error: wasm-bindgen not found; run 'cargo install wasm-bindgen-cli'" >&2
    exit 1
fi

# `web` selects the wasm-bindgen entry point, getrandom's wasm_js backend, and
# the embedded stdlib VFS (no host filesystem in the browser). `--export-table`
# exposes __indirect_function_table for the JIT glue (jit_set_table) and
# `--growable-table` drops its fixed maximum so the glue can append compiled
# trace functions; unlike the wasm-host build, web does not use getrandom's
# `custom` backend.
RUSTFLAGS='-C link-arg=--export-table -C link-arg=--growable-table -C link-arg=-zstack-size=8388608' \
    cargo build --release -p pyre-wasm \
    --target wasm32-unknown-unknown \
    --no-default-features --features web

# Snapshot before anything else can clobber the shared output path, then drive
# wasm-bindgen from the snapshot. `--out-name pyre_wasm` keeps the emitted
# pyre_wasm.js / pyre_wasm_bg.wasm names index.html imports, regardless of the
# snapshot's filename.
cp "$raw_wasm" "$web_wasm"
wasm-bindgen --target web --out-dir "$www_dir" --out-name pyre_wasm "$web_wasm"

# The JIT glue module the page imports alongside the bindgen output.
cp "$glue_src" "$www_dir/jit_glue.js"

echo "web build staged in $www_dir (snapshot: $web_wasm)"
