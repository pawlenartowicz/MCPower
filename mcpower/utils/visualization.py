"""
Visualization utilities for Monte Carlo Power Analysis.

This module provides plotting functions for power analysis results.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

__all__ = []


def _create_power_plot(
    sample_sizes: List[int],
    powers_by_test: Dict[str, List[float]],
    first_achieved: Dict[str, int],
    target_tests: List[str],
    target_power: float,
    title: str,
):
    """Create sample size vs power plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.get_cmap("Set1")(np.linspace(0, 1, len(target_tests)))

    for i, test in enumerate(target_tests):
        powers = powers_by_test[test]
        ax.plot(
            sample_sizes,
            powers,
            "o-",
            color=colors[i],
            label=test,
            linewidth=2,
            markersize=4,
        )

        # Mark achievement point
        if first_achieved[test] > 0:
            achieved_idx = sample_sizes.index(first_achieved[test])
            achieved_power = powers[achieved_idx]
            ax.plot(
                first_achieved[test],
                achieved_power,
                "s",
                color=colors[i],
                markersize=10,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=colors[i],
            )

            # Annotation
            ax.annotate(
                f"N={first_achieved[test]}",
                xy=(first_achieved[test], achieved_power),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3),
                arrowprops=dict(arrowstyle="->", color=colors[i]),
            )

    # Target power line
    ax.axhline(
        y=target_power,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target Power ({target_power}%)",
    )

    # Configure axes
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("Power (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 105)

    # X-axis ticks
    tick_interval = max(10, (max(sample_sizes) - min(sample_sizes)) // 10)
    ax.set_xticks(range(min(sample_sizes), max(sample_sizes) + 1, tick_interval))

    plt.tight_layout()
    plt.show()
