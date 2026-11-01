---
name: apple-container
description: Reproduce Linux-only pyre failures on macOS inside Apple's `container` CLI, using the Ubuntu 24.04 repro image in `tools/`. Invoked via `/apple-container`. Use this skill whenever a failure is visible on a Linux CI leg but not locally — a `CPython suite (gate)` row, an ubuntu-only `pyre/check.py` failure, a crash that only the linux runner sees — or when the user says "linux에서 재현해줘", "run this in the container", "ubuntu에서 확인", "apple container로 돌려줘". Also use it before claiming a Linux-only red is inherited or unreproducible.
---

# Reproduce Linux-only failures with Apple `container`

Only the CI legs on `ubuntu-24.04` run some checks, and darwin/linux divergence
is a recurring source of reds that cannot be judged from a macOS worktree. This
skill runs the same code under a Linux userspace on the same machine.

**Use Apple's `container` CLI and nothing else. Never Docker, never Podman,
never `colima`.** If `container` is missing, stop and say so — do not substitute
another runtime.

## What this can and cannot reproduce

Verified 2026-10-31 against the image on this machine. Read this table before
promising a repro:

| target | works? | why |
|---|---|---|
| A crash / wrong answer from a bench or script | **yes** | run the binary directly |
| `pyre/cpython_tests/run.py` — the `CPython suite (gate)` job | **yes** | the runner needs no oracle |
| `pyre/check.py` ratio + jitstats gates | **yes** | the image pins CI's CPython 3.15.2 and PyPy 8.0.0; container timings remain diagnostic |
| `pyre/extra_tests/parity_tests/run.py` | **yes** | the same pinned oracles are available |

The image sets `PYRE_CHECK_PYTHON3` and `PYRE_CHECK_PYPY3` without replacing
Ubuntu's system Python. It also installs CPython 3.15.2t for free-threaded
spec checks. Older containers can use the same reproducible installer:
`bash tools/ubuntu24-amd64-repro/install-oracles.sh`, then export those
variables to `/opt/pyre-oracles/bin/python3.15` and
`/opt/pyre-oracles/bin/pypy3`, matching the image's container-local paths.
Never substitute Ubuntu's Python 3.12 for the required oracle.

## Images

`tools/ubuntu24-amd64-repro/` holds the Dockerfile. Two variants matter:

- **arm64** (`pyre-ubuntu24-arm64-repro`) — native on Apple silicon, so builds
  and runs at full speed. Use it to answer *"is this linux-vs-macos?"*. Its
  charon lives at `$PYRE_SHARED_BUILD/charon/linux-aarch64`.
- **amd64** (`pyre-ubuntu24-amd64-repro`) — matches the GitHub `ubuntu-24.04`
  runner's architecture, and runs under `--rosetta`, so it is much slower. Use
  it only when the arm64 answer disagrees with CI, or when the failure is
  plausibly x86-specific (codegen, alignment, float formatting).

Build (per `tools/ubuntu24-amd64-repro/README.md`):

```shell
container build --platform linux/amd64 -m 8G -c 4 --progress plain \
  -t pyre-ubuntu24-amd64-repro tools/ubuntu24-amd64-repro
```

## Workflow

1. Check whether the container is already up — do not start a second one:

   ```shell
   container list -a 2>/dev/null | rg pyre-linux
   ```

   A `stopped` row is not a missing container — `container start pyre-linux`
   and keep the toolchain it already has.

2. Start it if absent, from the repository root. **Four** mounts are required:
   the worktree; the shared build cache **outside** the worktree so sibling
   worktrees and container runs reuse one charon install; the main checkout
   at its own host path, because these worktrees are `git worktree`s (see the
   trap below); and a per-container directory over `/workspace/pyre/target/llbc`
   so extraction never writes into the host's artefacts.

   ```shell
   mkdir -p "$(dirname "$PWD")/.pyre-build" \
            "$(dirname "$PWD")/.pyre-build/llbc-container/pyre-linux"
   MAIN=$(sed -n 's|^gitdir: \(.*\)/\.git/worktrees/.*|\1|p' .git)   # empty in a normal clone
   container run -d --name pyre-linux --platform linux/arm64 -m 12G -c 4 \
     --mount type=bind,source="$(pwd)",target=/workspace/pyre \
     --mount type=bind,source="$(dirname "$PWD")/.pyre-build",target=/workspace/.pyre-build \
     --mount type=bind,source="$MAIN",target="$MAIN" \
     --mount type=bind,source="$(dirname "$PWD")/.pyre-build/llbc-container/pyre-linux",target=/workspace/pyre/target/llbc \
     pyre-ubuntu24-arm64-repro sleep infinity
   ```

   `-m` is the memory cap — the container is where the "always cap memory when
   running pyre" rule is enforced, so never drop it.

   Then mark both trees safe, or every `git` call inside refuses on ownership:

   ```shell
   container exec pyre-linux sh -c "git config --global --add safe.directory /workspace/pyre
                                    git config --global --add safe.directory $MAIN"
   ```

3. One-time per cache, inside the container:

   ```shell
   container exec pyre-linux sh -c 'cd /workspace/pyre && python3 scripts/install-charon.py'
   container exec pyre-linux sh -c 'cd /workspace/pyre && charon toolchain-path'
   ```

   The first downloads a prebuilt charon into the shared cache, so it is nearly
   free after the first worktree does it. The second triggers the pinned
   nightly's install, and that lands in the container's **own** filesystem —
   `rustc-dev` alone is several hundred MB and takes many minutes. It survives
   `container stop`/`start` but **not** `container rm`, so do not delete the
   container to "reset" something; stop it instead.

4. **Set `CARGO_TARGET_DIR` to an absolute path for every build and run.** The
   worktree is bind-mounted, so an unqualified Linux `cargo build` writes into
   the same `target/release/` the macOS binaries live in and silently replaces
   them. It must be **absolute**: `scripts/extract-llbc.py` runs charon once per
   crate with that crate's directory as the cwd, so a relative value scatters a
   separate `target-linux/` into every crate — 4.8 GB across four crates,
   untracked and outside `.gitignore`'s anchored `/target-linux` rule.

   ```shell
   container exec pyre-linux sh -c 'cd /workspace/pyre \
     && export CARGO_TARGET_DIR=/workspace/pyre/target-linux CARGO_INCREMENTAL=0 \
     && python3 scripts/extract-llbc.py \
     && cargo build --release -p pyrex --bin pyre-cranelift \
          --no-default-features --features cranelift'
   ```

   Extraction and the Linux `pyre-jit-trace` build both read and write
   `/workspace/pyre/target/llbc` as usual; that is the step-2 mount.

   `scripts/extract-llbc.py` takes **no crate arguments** — the bare form
   extracts the whole set. Naming a subset mixes fresh and stale artefacts.
   Budget for it: on this machine the extract alone ran ~35 min (four dev-profile
   charon passes) and the release build another ~20.

5. Run the thing that is red on Linux:

   ```shell
   # a bench or script
   container exec pyre-linux sh -c 'cd /workspace/pyre \
     && RUST_BACKTRACE=1 ./target-linux/release/pyre-cranelift pyre/bench/synth/<name>.py'

   # the CPython suite gate job
   container exec pyre-linux sh -c 'cd /workspace/pyre \
     && python3 pyre/cpython_tests/run.py --backend cranelift \
          --binary target-linux/release/pyre-cranelift --filter <module>'
   ```

6. Report by comparing the two hosts explicitly: what macOS does, what Linux
   does, and whether that matches the CI leg. A repro that only reproduces on
   one architecture is itself the finding — say which.

7. Leave the container running. Remove it only when asked:

   ```shell
   container rm -f pyre-linux
   ```

## Traps

- **A `git worktree` breaks every git call inside the container**, and
  `scripts/extract-llbc.py` runs `git ls-files` for its fingerprint, so the
  extract dies before it starts:

  ```
  fatal: not a git repository: /Users/youknowone/Projects/pyre/.git/worktrees/pyre-6
  ```

  A worktree's `.git` is a *file* holding `gitdir: <absolute host path>` into the
  main checkout's `.git/worktrees/`. Mounting only the worktree leaves that path
  unresolvable. The mount in step 2 maps the main checkout to the **identical**
  path inside the container, which is what makes the pointer resolve — a
  different target does not work, the path is baked into the `.git` file. This
  is not a `safe.directory` problem; that error is a different one and needs the
  `git config` calls above as well.
- **A Linux red is not automatically the branch's.** Check main's own CI at
  your merge-base first, and diff your branch's row list against main's before
  attributing anything. The suite gate rides `cargo-test-linux` on
  `ubuntu-24.04`, which is also where the baseline's verdicts are enforced, so
  a row the container disagrees about is a finding rather than a known skew.
- **`target-linux/` must not be committed.** Confirm it is ignored before
  building, or the worktree fills with Linux objects.
- **Timing measured in the container is not a measurement.** It is a VM sharing
  the host's cores with whatever else is running. Use the container for
  crashes, assertions, and wrong answers — never to judge a perf ratio.
- **The container sees your uncommitted edits**, because the worktree is
  bind-mounted. That is the point, but it also means a rebase or a sibling
  session's edit changes what the container is building mid-run. It cuts the
  other way too: `source=` is a hash of the tree, so editing anything after an
  extraction marks the artefacts stale — do the extract last and leave the tree
  alone until whatever you are measuring has finished.
- **A container lacking the `target/llbc` mount must not extract.** The
  driver's `out_dir` is a fixed `<repo-root>/target/llbc` with no override,
  and the worktree is bind-mounted, so an extract without the nested mount
  replaces the artefacts your macOS build reads. The step-2 mount is now
  mandatory; recreate a container that lacks it (`container rm` then
  `container run`; the pinned nightly toolchain is reinstalled, see step 3)
  rather than running `extract-llbc.py` in it. Every field the freshness
  check compares is computed from the tracked tree and so agrees across
  hosts; `platform=` is the one that does not, and `fail_if_llbc_stale` now
  refuses on it rather than letting the build fail somewhere downstream.
  Leave `LLBC_DEST` unset inside the container. `scripts/llbc_extract.py`
  (`llbc_dest_path`) takes it as an output override and resolves a relative
  value against the repo root, i.e. the bind-mounted host worktree, so a set
  `LLBC_DEST` bypasses the step-2 mount; the Linux `pyre-jit-trace` build
  reads the fixed `target/llbc` regardless.
