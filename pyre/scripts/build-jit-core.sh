#!/bin/sh
# Fast core for JIT-quality loops. Leaves `pyre-module` out of the binary and
# out of the extraction; the other artefacts are the product `target/llbc` set.
#
#   pyre/scripts/build-jit-core.sh
#   pyre/scripts/build-jit-core.sh --check   # cargo check, no codegen
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODE=release
if [ "${1:-}" = "--check" ]; then
    MODE=check
fi
python3 pyre/scripts/extract-llbc.py majit-rlib pyre-object pyre-interpreter pyre-jit
# Without `pyre-module`, `pyre-jit-trace`'s prepass neither links nor hashes
# `pyre-module` and does not read its artefact.
if [ "$MODE" = check ]; then
    exec cargo check -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,mimalloc
fi
# The `jit-core` profile writes `target/jit-core/pyre-dynasm`; measure it with
# `python3 pyre/check.py --build no --backend dynasm target/jit-core/pyre-dynasm`.
exec cargo build --profile jit-core -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,mimalloc
