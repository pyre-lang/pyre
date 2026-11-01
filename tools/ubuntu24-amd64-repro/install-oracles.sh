#!/usr/bin/env bash
# Match pyre-ci.yml without replacing Ubuntu's system Python.
set -euo pipefail

oracle_root=/opt/pyre-oracles
if [ "$#" -gt 0 ]; then
    oracle_root=$1
fi
cpython_version=3.15.2
pypy_version=8.0.0
uv_version=0.13.6
machine=$(uname -m)
case "$machine" in
    aarch64) pypy_platform=aarch64 ;;
    x86_64) pypy_platform=linux64 ;;
    *) echo "unsupported Linux architecture: $machine" >&2; exit 1 ;;
esac

mkdir -p "$oracle_root/bin"
oracle_root=$(cd "$oracle_root" && pwd)
oracle_tmp=$(mktemp -d)
trap 'rm -rf "$oracle_tmp"' EXIT
uv_archive="uv-$machine-unknown-linux-gnu"
curl --proto '=https' --tlsv1.2 -fLsS --retry 2 \
    "https://github.com/astral-sh/uv/releases/download/$uv_version/$uv_archive.tar.gz" \
    -o "$oracle_tmp/uv.tar.gz"
tar -xzf "$oracle_tmp/uv.tar.gz" -C "$oracle_tmp"
uv_bin="$oracle_tmp/$uv_archive/uv"
export UV_PYTHON_INSTALL_DIR="$oracle_root/cpython"
"$uv_bin" python install "$cpython_version" "$cpython_version"t --no-bin
cpython=$("$uv_bin" python find --no-project --managed-python "$cpython_version")
cpython_t=$("$uv_bin" python find --no-project --managed-python "$cpython_version"t)
ln -sfn "$cpython" "$oracle_root/bin/python3.15"
ln -sfn "$cpython_t" "$oracle_root/bin/python3.15t"

pypy_archive="pypy3.11-v$pypy_version-$pypy_platform"
if [ ! -x "$oracle_root/$pypy_archive/bin/pypy3" ]; then
    curl --proto '=https' --tlsv1.2 -fLsS --retry 2 \
        "https://downloads.python.org/pypy/$pypy_archive.tar.gz" \
        -o "$oracle_tmp/pypy.tar.gz"
    tar -xzf "$oracle_tmp/pypy.tar.gz" -C "$oracle_root"
fi
ln -sfn "$oracle_root/$pypy_archive/bin/pypy3" "$oracle_root/bin/pypy3"

"$oracle_root/bin/python3.15" -c \
    'import sys; assert sys.version_info[:3] == (3, 15, 2); print(sys.version)'
"$oracle_root/bin/python3.15t" -c \
    'import sys; assert sys.version_info[:3] == (3, 15, 2); assert not sys._is_gil_enabled(); print(sys.version)'
"$oracle_root/bin/pypy3" -c \
    'import sys; assert sys.pypy_version_info[:3] == (8, 0, 0); assert sys.version_info[:2] == (3, 11); print(sys.version)'
