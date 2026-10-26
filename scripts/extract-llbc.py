#!/usr/bin/env python3
"""Forwarding shim: extraction drivers now live per-repo.

The pyre crate table moved to `pyre/scripts/extract-llbc.py` (external consumer
repos carry their own drivers). This shim preserves the historical
`scripts/extract-llbc.py` entry point that CI, `build.rs`, and `check.py`
invoke, forwarding argv verbatim.
"""

import subprocess
import sys
from pathlib import Path

driver = Path(__file__).resolve().parents[1] / "pyre" / "scripts" / "extract-llbc.py"
# `subprocess.run`, not `os.execv`: on Windows `os.execv` does not replace the
# process image — it spawns the child and the parent returns immediately with
# exit 0, so the invoking shell step finishes before the extraction driver has
# produced any `target/llbc/*.ullbc`, and the following upload-artifact step
# fails with `if-no-files-found: error`. Run the driver as a child and forward
# its exit code so every platform blocks until extraction completes.
sys.exit(subprocess.run([sys.executable, str(driver), *sys.argv[1:]]).returncode)
