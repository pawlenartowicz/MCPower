"""Simple weekly update checker for the MCPower package.

Queries PyPI at most once per week (cached on disk) and issues a
``UserWarning`` when a newer version is available.
"""

import json
import os
import sys
import urllib.request
import warnings
from datetime import datetime, timedelta
from pathlib import Path


def _check_for_updates(current_version):
    """Check PyPI weekly for a newer MCPower version and warn if found.

    Uses a JSON cache file to avoid repeated network requests. Skipped
    silently in worker processes (detected via environment variable)
    and in frozen (PyInstaller) bundles where pip is unavailable.
    """

    # Skip in frozen bundles (PyInstaller) — the GUI has its own update checker
    if getattr(sys, "frozen", False):
        return

    # Skip in worker processes (loky/joblib inherit env vars from parent)
    if os.environ.get("_MCPOWER_UPDATE_CHECKED"):
        return
    os.environ["_MCPOWER_UPDATE_CHECKED"] = "1"

    cache_path = Path(__file__).parent.parent / ".mcpower_cache.json"
    cache_path.parent.mkdir(exist_ok=True)

    # Load cache
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    # Check weekly
    last_check = cache.get("last_check")
    if not last_check or datetime.now() - datetime.fromisoformat(last_check) > timedelta(days=7):
        latest = _get_latest_version()
        cache = {
            "last_check": datetime.now().isoformat(),
            "latest_version": latest,
            "current_version": current_version,
        }
        try:
            cache_path.write_text(json.dumps(cache))
        except Exception:
            pass  # Read-only install; skip cache write

    # Show update message only when PyPI version is strictly newer
    latest = cache.get("latest_version")
    current = cache.get("current_version")
    if latest and current and _is_newer(latest, current):
        msg = f"\nNEW MCPower VERSION AVAILABLE: {latest} (you have {current})\nUpdate now: pip install --upgrade MCPower\n"
        warnings.warn(msg, stacklevel=3)


def _is_newer(latest, current):
    """Return ``True`` if *latest* is strictly newer than *current* (semver comparison)."""
    try:
        latest_parts = tuple(int(x) for x in latest.split("."))
        current_parts = tuple(int(x) for x in current.split("."))
        return latest_parts > current_parts
    except (ValueError, AttributeError):
        return False


def _get_latest_version():
    """Fetch the latest MCPower version string from the PyPI JSON API."""
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/MCPower/json", timeout=5) as response:
            data = json.loads(response.read())
            return data["info"]["version"]
    except Exception:
        return None
