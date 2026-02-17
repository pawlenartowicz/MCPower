"""
Visualization utilities for Monte Carlo Power Analysis.

This module provides plotting functions for power analysis results.
"""

from typing import Dict, List

import numpy as np

__all__ = []


def _create_power_plot(
    sample_sizes: List[int],
    powers_by_test: Dict[str, List[float]],
    first_achieved: Dict[str, int],
    target_tests: List[str],
    target_power: float,
    title: str,
):
    """Create a sample-size vs. power line plot with achievement markers.

    Draws one line per target test, a horizontal dashed line at the
    target power, and annotates the first sample size that reaches
    the target.

    Args:
        sample_sizes: X-axis values.
        powers_by_test: Mapping of test name to list of power percentages.
        first_achieved: Mapping of test name to the first sample size
            that achieved target power (``-1`` if not achieved).
        target_tests: Ordered list of test names to plot.
        target_power: Target power percentage (drawn as reference line).
        title: Plot title.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting: pip install mcpower[plot]") from None

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
                bbox={"boxstyle": "round,pad=0.3", "facecolor": colors[i], "alpha": 0.3},
                arrowprops={"arrowstyle": "->", "color": colors[i]},
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

    fig.text(
        0.5,
        0.01,
        "made in MCPower \u2014 simple Monte Carlo power analysis for complex models",
        ha="center",
        fontsize=9,
        color="#888888",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()
