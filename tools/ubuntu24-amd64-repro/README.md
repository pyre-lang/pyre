# Ubuntu 24.04 amd64 repro image

This image fixes the Linux userspace used to reproduce Ubuntu-only
`pyre-cranelift` failures. It intentionally does not copy the repository into
the image; mount the worktree at runtime so local edits and build artifacts are
visible.

The image includes the CI oracles: CPython 3.15.2 and PyPy 8.0.0, plus
CPython 3.15.2t for free-threaded spec checks. `PYRE_CHECK_PYTHON3` and
`PYRE_CHECK_PYPY3` name the pinned installations; Ubuntu's system Python stays
unchanged. The same Dockerfile supports `--platform linux/arm64` for native
Apple-silicon reproduction; use a separate image tag and omit `--rosetta`.

Build with Apple `container`:

```bash
container build --platform linux/amd64 -m 8G -c 4 --progress plain \
  -t pyre-ubuntu24-amd64-repro \
  tools/ubuntu24-amd64-repro
```

Run from the repository root:

```bash
mkdir -p "$(dirname "$PWD")/.pyre-build/llbc-container/pyre-linux-repro"
MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
container run --rm --platform linux/amd64 --rosetta -m 20G -c 4 \
  --mount type=bind,source="$(pwd)",target=/workspace/pyre \
  --mount type=bind,source="$(dirname "$PWD")/.pyre-build",target=/workspace/.pyre-build \
  --mount type=bind,source="$MAIN",target="$MAIN" \
  --mount type=bind,source="$(dirname "$PWD")/.pyre-build/llbc-container/pyre-linux-repro",target=/workspace/pyre/target/llbc \
  pyre-ubuntu24-amd64-repro
```

The second mount keeps the shared Charon cache outside the worktree and reuses
it across sibling worktrees and container runs. The third is needed when the
repository is a `git worktree`: its `.git` is a file naming an absolute path
into the main checkout, and `extract-llbc.py` fingerprints the tree with
`git ls-files`, so without that path — at the *same* location inside — every
git call fails with `fatal: not a git repository`. The fourth mount keeps
Linux LLBC and target layouts out of the host's native `target/llbc`; using
`CARGO_TARGET_DIR` alone does not separate these extraction outputs. Both trees also need
`git config --global --add safe.directory <path>` inside the container.

Inside the container. `CARGO_TARGET_DIR` keeps the container's Linux objects out
of the host's `target/`, which the bind mount otherwise shares; and
`extract-llbc.py` takes no crate arguments, because naming a subset mixes fresh
and stale artefacts. Give `CARGO_TARGET_DIR` an absolute path — the extract runs
charon from each crate's own directory, so a relative value drops a separate
multi-gigabyte `target-linux/` into every crate instead of sharing one.

```bash
set -e
python3 scripts/install-charon.py
export CARGO_TARGET_DIR=/workspace/pyre/target-linux   # absolute: see below
export CHARON_TARGET_DIR=/workspace/pyre/target-linux/charon   # per worktree, next to CARGO_TARGET_DIR
CARGO_INCREMENTAL=0 python3 scripts/extract-llbc.py
CARGO_INCREMENTAL=0 cargo build --release -p pyrex --bin pyre-cranelift \
  --no-default-features --features cranelift,mimalloc
python3 scripts/extract-llbc.py --check
RUST_BACKTRACE=1 ./target-linux/release/pyre-cranelift pyre/bench/synth/bool_compare.py
python3 pyre/check.py --build no --backend cranelift ./target-linux/release/pyre-cranelift
```

The Charon target lives under the worktree mount so containers from different
worktrees do not share one cargo target.

The explicit binary path matters: `check.py` otherwise expects the host-shared
`target/release`, while this build writes to `CARGO_TARGET_DIR`. The freshness
check above verifies the extracted inputs before running that external binary.
Container timing ratios are diagnostic; the native CI runner remains the
performance reference.

For an existing Linux container built before the oracles were included, use
the same installer and container-local paths as the image:

```bash
bash tools/ubuntu24-amd64-repro/install-oracles.sh
export PYRE_CHECK_PYTHON3=/opt/pyre-oracles/bin/python3.15
export PYRE_CHECK_PYPY3=/opt/pyre-oracles/bin/pypy3
```

Update the installer pins alongside `pyre-ci.yml`; changing the PyPy version
changes the denominator of every performance ratio.
