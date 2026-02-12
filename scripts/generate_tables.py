#!/usr/bin/env python
"""
Generate lookup tables used by MCPower.

Currently used tables (loaded by mcpower.tables.lookup):
  - norm_cdf.npz  — Normal CDF lookup for fast p-value interpolation
  - t3_ppf.npz    — t(3) PPF lookup for heavy-tailed data transforms

Usage:
    python scripts/generate_tables.py [--output-dir PATH] [--resolution N]
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.stats import t as t_dist, norm


def generate_norm_tables(resolution: int = 4096) -> tuple:
    """
    Generate normal distribution tables.

    Args:
        resolution: Number of points in tables

    Returns:
        Tuple of (cdf_table, ppf_table, x_range, percentile_range)
    """
    print(f"Generating normal distribution tables (resolution {resolution})...")

    # CDF table
    x_range = np.linspace(-6, 6, resolution)
    cdf_table = norm.cdf(x_range).astype(np.float64)

    # t(3) PPF table for heavy-tailed transforms
    sqrt3 = np.sqrt(3)
    percentile_range = np.linspace(0.001, 0.999, resolution)
    t3_ppf_table = (t_dist.ppf(percentile_range, 3) / sqrt3).astype(np.float64)

    return cdf_table, t3_ppf_table, x_range, percentile_range


def main():
    parser = argparse.ArgumentParser(
        description="Generate lookup tables for MCPower"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "mcpower" / "tables" / "data",
        help="Output directory for tables"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4096,
        help="Number of points in lookup tables (default: 4096)"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    norm_cdf, t3_ppf, x_range, perc_range = generate_norm_tables(
        resolution=args.resolution
    )

    np.savez_compressed(
        args.output_dir / "norm_cdf.npz",
        norm_cdf=norm_cdf,
        x_range=x_range
    )
    np.savez_compressed(
        args.output_dir / "t3_ppf.npz",
        t3_ppf=t3_ppf,
        percentile_range=perc_range
    )
    print(f"  Saved norm_cdf.npz, t3_ppf.npz")

    print(f"\nAll tables generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
