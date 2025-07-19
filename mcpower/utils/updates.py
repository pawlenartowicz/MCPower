"""
Simple update checker for MCPower package.
"""

import json
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


def _check_for_updates(current_version):
    """Check PyPI weekly for updates and show message if available."""

    cache_path = Path(__file__).parent.parent / ".mcpower_cache.json"
    cache_path.parent.mkdir(exist_ok=True)

    # Load cache
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    # Check weekly
    last_check = cache.get("last_check")
    if not last_check or datetime.now() - datetime.fromisoformat(
        last_check
    ) > timedelta(days=7):
        latest = _get_latest_version()
        cache = {
            "last_check": datetime.now().isoformat(),
            "latest_version": latest,
            "current_version": current_version,
        }
        cache_path.write_text(json.dumps(cache))

    # Show update message
    if cache.get("latest_version") and cache["latest_version"] != cache.get(
        "current_version"
    ):
        print("=" * 60)
        print(
            f"NEW MCPower VERSION AVAILABLE: {cache['latest_version']} (you have {cache.get('current_version')})"
        )
        print("Update now: pip install --upgrade MCPower")
        print("=" * 60)


def _get_latest_version():
    """Get latest version from PyPI."""
    with urllib.request.urlopen(
        "https://pypi.org/pypi/MCPower/json", timeout=5
    ) as response:
        data = json.loads(response.read())
        return data["info"]["version"]
