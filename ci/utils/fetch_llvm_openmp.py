#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fetch a recent LLVM OpenMP runtime (libomp) from conda-forge for wheel builds.

Rocky Linux 8's llvm-toolset libomp lags conda-forge's, so this resolves and
fetches the newest llvm-openmp build satisfying the same floor the conda
packages use, directly from conda-forge's repodata; no conda/mamba CLI needed,
since .conda packages are just a zip containing a zstd-compressed tar.
See https://github.com/NVIDIA/cuopt/issues/1219
"""

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import zstandard

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES_YAML = REPO_ROOT / "dependencies.yaml"

ARCH_SUBDIRS = {"x86_64": "linux-64", "aarch64": "linux-aarch64"}

# Newest GOMP ABI entry point cuOpt imports, plus OpenMP 5.0 detached tasks.
REQUIRED_SYMBOLS = ("GOMP_taskwait_depend@GOMP_5.0", "omp_fulfill_event")


def min_llvm_openmp_major():
    """Read the llvm-openmp version floor from dependencies.yaml.

    Also used for conda/recipes/libcuopt/recipe.yaml, read from
    there so this can't drift from the version resolved below.
    """
    match = re.search(r"- llvm-openmp >=(\d+)", DEPENDENCIES_YAML.read_text())
    if not match:
        sys.exit(
            f"Could not find an 'llvm-openmp >=N' floor in {DEPENDENCIES_YAML}"
        )
    return int(match.group(1))


def curl(url, dest):
    """Download url to dest via curl.

    Not urllib: this pyenv-built Python's default SSL context fails
    TLS verification against conda.anaconda.org in the Rocky 8
    wheel-build image, even though curl (using the system's own
    trust store) resolves the same URL fine there.
    """
    print(f"Fetching {url}")
    subprocess.run(["curl", "-fsSL", "-o", str(dest), url], check=True)


def release_version(version):
    """Version as an int tuple, or None for pre-releases."""
    parts = version.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(map(int, parts))


def resolve_build(subdir, min_major, workdir):
    """Find the newest llvm-openmp build satisfying the floor.

    Reads conda-forge's own repodata, the same host the package
    itself is downloaded from below.
    """
    repodata_path = workdir / "current_repodata.json"
    curl(
        f"https://conda.anaconda.org/conda-forge/{subdir}/current_repodata.json",
        repodata_path,
    )
    with open(repodata_path) as f:
        data = json.load(f)

    candidates = []
    for fn, v in data["packages.conda"].items():
        if v["name"] != "llvm-openmp":
            continue
        version = release_version(v["version"])
        if version is not None and version[0] >= min_major:
            candidates.append((version, v["timestamp"], fn, v["sha256"]))
    if not candidates:
        sys.exit(f"No llvm-openmp >={min_major} build found for {subdir}")

    _, _, fn, sha256 = max(candidates)
    return fn, sha256


def download(subdir, pkg_file, sha256, dest):
    """Download pkg_file and verify it against conda-forge's own sha256.

    Verifying against the digest from repodata (not just the
    download itself) is a real integrity check (CWE-494).
    """
    curl(f"https://conda.anaconda.org/conda-forge/{subdir}/{pkg_file}", dest)

    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != sha256:
        sys.exit(
            f"sha256 mismatch for {pkg_file}: expected {sha256}, got {digest}"
        )


def extract_libomp(pkg_path, out_dir):
    """Extract lib/libomp.so out of a .conda package."""
    zf = zipfile.ZipFile(pkg_path)
    pkg_name = next(n for n in zf.namelist() if n.startswith("pkg-"))
    tar_bytes = zstandard.ZstdDecompressor().decompress(
        zf.read(pkg_name), max_output_size=200 * 1024 * 1024
    )
    tarfile.open(fileobj=io.BytesIO(tar_bytes)).extractall(out_dir)

    libomp_so = Path(out_dir) / "lib" / "libomp.so"
    if not libomp_so.is_file() or libomp_so.is_symlink():
        sys.exit(f"Could not find lib/libomp.so in {pkg_path}")
    return libomp_so


def verify_symbols(libomp_so):
    """Verify it exports what we need rather than trusting the pin alone."""
    result = subprocess.run(
        ["nm", "-D", "--defined-only", str(libomp_so)],
        capture_output=True,
        text=True,
        check=True,
    )
    exported = {line.split()[-1] for line in result.stdout.splitlines()}
    for symbol in REQUIRED_SYMBOLS:
        if not any(
            s == symbol or s.startswith(symbol + "@") for s in exported
        ):
            sys.exit(
                f"Fetched libomp does not export {symbol}: "
                "wrong package or bad extraction"
            )


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <dest_dir>")
    dest_dir = Path(sys.argv[1])
    dest_dir.mkdir(parents=True, exist_ok=True)

    machine = platform.machine()
    subdir = ARCH_SUBDIRS.get(machine)
    if subdir is None:
        sys.exit(f"Unsupported architecture for llvm-openmp fetch: {machine}")

    # Ephemeral CI container, no need to clean this up ourselves.
    workdir = Path(tempfile.mkdtemp())

    min_major = min_llvm_openmp_major()
    pkg_file, sha256 = resolve_build(subdir, min_major, workdir)

    pkg_path = workdir / pkg_file
    download(subdir, pkg_file, sha256, pkg_path)

    libomp_so = extract_libomp(pkg_path, workdir / "extracted")
    verify_symbols(libomp_so)

    # libomp's SONAME is the unversioned libomp.so, so one file is enough.
    dest_file = dest_dir / "libomp.so"
    dest_file.write_bytes(libomp_so.read_bytes())

    print(f"LLVM OpenMP runtime ready at {dest_file} (from {pkg_file})")


if __name__ == "__main__":
    main()
