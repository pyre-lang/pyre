# pyre-check: max-pypy-ratio=1.8
# pyre-check: requires-modules=math
# pyre-check: skip-cpython
# This fixture carried a wasm allowance of 13, fitted to a darwin-arm64
# reading of 8.1-9.0x.  Most of that was the host crossing: `isclose`, `frexp`
# and `ldexp` lower to helpers with mixed `f64`/word signatures, which the
# backend would not lower in-module without a caller vouching for them, so
# 3,998,958 calls left the trace module.  Vouched, the fixture reads 1.5x on
# the same host and carries no allowance.
# What remains of the gap is structural rather than a regression: `pymath`
# reaches the platform libm on native and its pure-Rust `libm` fallback in the
# guest, and the guest also pays an errno-classifying wrapper the native build
# folds away.  Measured on the same fold machinery and the same loop, wasm
# runs 2M folded `log` (which lowers to `x.ln()`) in 0.09s and 2M folded `exp`
# (which goes through `pymath`) in 0.24s.  Both backends fold; only the
# operation underneath differs.
# The ratio is the detector here: losing a fold changes no jit-stats counter,
# because the residual it falls back to compiles the same loop.  Measured on
# an idle darwin-arm64 box against pypy 0.33s — every fold 0.63s, the generic
# float/isclose folds suppressed 2.71s, all folds suppressed 33.4s.  The gate
# sits between the first two.
# Every `math` primitive the walker folds, one hot loop per fold shape.  A
# residual `bh_call_fn(builtin, NULL, x)` costs an argument tuple, a
# `W_FloatObject` allocation and a full builtin dispatch per iteration; each
# specialization instead unboxes the operands, emits the raw operation, and
# leaves the result box virtualizable.
#
#   sqrt         `__majit_wrap_math_sqrt` descent — `x >= 0` and `isfinite(x)`
#                pin the two `ll_math_sqrt` branches, then `_float_sqrt`
#                (`sqrt_nonneg`) with no result guard.
#   log/cos/sin  `__majit_wrap_math_{log,cos,sin}` descent — one domain guard
#                each (`x > 0` for `log`, `isfinite` for `cos`/`sin`), then
#                `_float_log` / `_float_cos` / `_float_sin`.
#   fabs         `__majit_wrap_math_fabs` descent — `_float_abs` with no
#                domain guard. `fabs` raises for no input.
#   floor/ceil/  `__majit_wrap_math_{floor,ceil,trunc}` — an exact float inside
#   trunc        the signed machine range, then `_int_from_{floor,ceil,trunc}`.
#   isclose      `__majit_wrap_math_isclose` — two finite operands and the
#                default tolerances, then `_float_isclose`.
#   the rest     `__majit_wrap_math_{exp,pow,...}` — the raw leaf, and a
#                non-finite result (overflow, `pow(0.0, -2.0)`) stays in the
#                original body.
#
# A rebound callable, a numeric subclass, or an operand outside the folded
# domain keeps the residual in every case.
#
# Every loop below derives its operand from the counter, so what the ratio
# reads is the per-iteration cost of a fold and not what the optimizer can do
# with one.  Recording these calls as elidable lets the pure pass serve a
# second identical call from the first, which pays only where the operand does
# not change.  Measured on darwin-arm64 against the previous binary,
# interleaved over seven rounds, an invariant operand reads 6.2x on
# `pow`/`atan2`, 3.5x on `tan`/`exp`, 3.3x on `log`/`cos`/`sin`, 2.1x on
# `frexp`, 1.9x on `ldexp`, 1.4x on `floor`/`ceil`, 1.2x on `isqrt` and 1.05x
# on `sqrt` -- each ahead in 7 of 7 rounds -- while the shapes below stay
# within noise of where they were, their peeled bodies unchanged op for op.
# `isclose` is the one fold that does not move even on an invariant operand:
# its answer is consumed entirely by the branch's own guard and never reaches
# the `Jump`, so nothing pulls the cached value across the loop boundary.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and declines the baseline as too small.
SQRT_N = 8000000
LOG_TRIG_N = 1600000
FABS_N = 10000000
ROUND_N = 10000000
UNARY_N = 1500000
BINARY_N = 2000000
ISCLOSE_N = 4000000


def run_sqrt():
    total = 0.0
    for i in range(SQRT_N):
        total += math.sqrt(float(i))
    return total


def run_log_trig():
    total = 0.0
    for i in range(LOG_TRIG_N):
        x = 1.0 + float(i % 97) / 97.0
        total += math.log(x) + math.cos(x) + math.sin(x)
    return total


def run_fabs():
    total = 0.0
    for i in range(FABS_N):
        total += math.fabs(float(i) - 6000000.0)
    return total


def run_round_to_int():
    total = 0
    for i in range(ROUND_N):
        x = float(i) * 0.5 - 1000.0
        total += math.floor(x) + math.ceil(x) + math.trunc(x)
    return total


def run_unary():
    total = 0.0
    for i in range(UNARY_N):
        x = float(i % 71) / 128.0
        total += math.exp(x) + math.tan(x) + math.atan(x)
        total += math.tanh(x) + math.log1p(x) + math.degrees(x)
    return total


def run_binary():
    total = 0.0
    for i in range(BINARY_N):
        x = 1.0 + float(i % 53) / 53.0
        y = 1.0 + float(i % 31) / 31.0
        total += math.pow(x, y) + math.fmod(x, y)
        total += math.copysign(x, y) + math.atan2(x, y)
    return total


def run_isclose():
    # The result has to decide a branch and nothing else: a bool that escapes
    # keeps the residual, because pinning it would bail on every re-entry with
    # the other truth.
    # The perturbation stays well inside the default `rel_tol=1e-09`, so the
    # branch resolves the same way every iteration and the trace measures the
    # fold rather than a bridge.
    hits = 0
    for i in range(ISCLOSE_N):
        x = 1.0 + float(i % 97) / 1e12
        if math.isclose(x, 1.0):
            hits += 1
    return hits


print(round(run_sqrt(), 6))
print(round(run_log_trig(), 6))
print(round(run_fabs(), 6))
print(run_round_to_int())
print(round(run_unary(), 6))
print(round(run_binary(), 6))
print(run_isclose())
