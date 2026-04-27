#!/usr/bin/env python3
"""
Generate a binary file of test numbers for vkRadix benchmarking.

Output is a raw little-endian array (no header). Read it from C++ with:

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> data(bytes / sizeof(uint32_t));   // or float
    f.read(reinterpret_cast<char*>(data.data()), bytes);

Filename convention encodes the type so you don't mix them up:
    data_<dist>_<dtype>_<count>.bin   e.g. data_uniform_u32_100M.bin

Usage examples:
    ./gen_data.py 100M                              # uniform u32, default
    ./gen_data.py 1G --dtype f32 --dist gaussian
    ./gen_data.py 10M --dist almost-sorted --seed 42
"""
import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np


# Default output directory: <repo_root>/assets/test/, computed from this script's location
# so it works regardless of cwd.
DEFAULT_OUT_DIR = (Path(__file__).resolve().parent.parent / "assets" / "test").as_posix()


SUFFIXES = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000, "T": 1_000_000_000}


def parse_count(s: str) -> int:
    m = re.fullmatch(r"(\d+)([KMGBT]?)", s.upper())
    if not m:
        raise argparse.ArgumentTypeError(f"bad count: {s!r}")
    return int(m.group(1)) * SUFFIXES.get(m.group(2), 1)


def gen(dist: str, n: int, dtype: str, rng: np.random.Generator) -> np.ndarray:
    if dtype == "u32":
        np_dtype = np.uint32
        if dist == "uniform":
            # Full range; chunked to avoid building one giant Python int range.
            return rng.integers(0, 2**32, size=n, dtype=np.uint32, endpoint=False)
        if dist == "gaussian":
            # Centered at 2^31, sigma ~ 2^29; clip into u32.
            x = rng.normal(loc=2**31, scale=2**29, size=n)
            return np.clip(x, 0, 2**32 - 1).astype(np.uint32)
        if dist == "sorted":
            return np.arange(n, dtype=np.uint32)
        if dist == "reverse":
            return np.arange(n, 0, -1, dtype=np.uint32)
        if dist == "almost-sorted":
            base = np.arange(n, dtype=np.int64)
            noise = rng.integers(-32, 33, size=n)
            return np.clip(base + noise, 0, 2**32 - 1).astype(np.uint32)
        if dist == "few-unique":
            # 256 distinct values — stresses the histogram path.
            vals = rng.integers(0, 2**32, size=256, dtype=np.uint32)
            return rng.choice(vals, size=n)
        if dist == "constant":
            return np.full(n, 0xDEADBEEF, dtype=np.uint32)

    elif dtype == "f32":
        if dist == "uniform":
            # Uniform over a wide range that exercises both signs.
            return rng.uniform(-1e6, 1e6, size=n).astype(np.float32)
        if dist == "gaussian":
            return rng.normal(loc=0.0, scale=1e3, size=n).astype(np.float32)
        if dist == "sorted":
            return np.linspace(-1e6, 1e6, n, dtype=np.float32)
        if dist == "reverse":
            return np.linspace(1e6, -1e6, n, dtype=np.float32)
        if dist == "almost-sorted":
            base = np.linspace(-1e6, 1e6, n, dtype=np.float32)
            noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)
            return base + noise
        if dist == "few-unique":
            vals = rng.uniform(-1e6, 1e6, size=256).astype(np.float32)
            return rng.choice(vals, size=n)
        if dist == "constant":
            return np.full(n, 3.14159, dtype=np.float32)

    raise ValueError(f"unsupported (dist, dtype): ({dist}, {dtype})")


def fmt_count(n: int) -> str:
    for suf, base in (("G", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if n >= base and n % base == 0:
            return f"{n // base}{suf}"
    return str(n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("count", type=parse_count, help="number of elements (e.g. 1000, 1M, 100M, 1G)")
    p.add_argument("--dtype", choices=["u32", "f32"], default="u32")
    p.add_argument("--dist", choices=["uniform", "gaussian", "sorted", "reverse",
                                      "almost-sorted", "few-unique", "constant"],
                   default="uniform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", help="output path (default: data_<dist>_<dtype>_<count>.bin)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"directory for default-named output (default: {DEFAULT_OUT_DIR})")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    arr = gen(args.dist, args.count, args.dtype, rng)

    if args.output is None:
        fname = f"data_{args.dist}_{args.dtype}_{fmt_count(args.count)}.bin"
        path = os.path.join(args.out_dir, fname)
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        path = args.output
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    arr.tofile(path)
    bytes_written = arr.nbytes
    print(f"wrote {path}  ({args.count:,} elems, {bytes_written / (1<<20):.1f} MiB, dtype={args.dtype}, dist={args.dist}, seed={args.seed})")


if __name__ == "__main__":
    sys.exit(main())
