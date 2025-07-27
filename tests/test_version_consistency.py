"""
Simple tests for version consistency and update checker.
"""

import pytest
import re
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


def test_version_consistency():
    """Test that __init__.py and setup.py have same version."""
    package_root = Path(__file__).parent.parent

    # Get version from __init__.py
    init_file = package_root / "mcpower" / "__init__.py"
    with open(init_file) as f:
        init_content = f.read()

    init_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_content)
    assert init_match, "No __version__ found in __init__.py"
    init_version = init_match.group(1)

    # Get version from setup.py
    setup_file = package_root / "setup.py"
    assert setup_file.exists(), "setup.py not found"

    with open(setup_file) as f:
        setup_content = f.read()

    setup_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', setup_content)
    assert setup_match, "No version= found in setup.py"
    setup_version = setup_match.group(1)

    # Check they match
    assert (
        init_version == setup_version
    ), f"Version mismatch: __init__.py={init_version}, setup.py={setup_version}"


def test_update_checker_works():
    """Test update checker shows message when newer version available."""
    from mcpower.utils.updates import _check_for_updates

    # Mock PyPI to return newer version
    with patch("mcpower.utils.updates._get_latest_version", return_value="9.9.9"):
        # Mock file operations to avoid cache files
        with patch("mcpower.utils.updates.Path") as mock_path:
            mock_path.return_value.parent.parent = Path("/tmp")
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.write_text = MagicMock()

            # Capture print output
            with patch("builtins.print") as mock_print:
                _check_for_updates("1.0.0")

                # Check update message was printed
                calls = [str(call) for call in mock_print.call_args_list]
                update_found = any("NEW MCPower VERSION" in call for call in calls)
                assert update_found, "Update message should be shown"


def test_pypi_api_call():
    """Test PyPI API call returns version."""
    from mcpower.utils.updates import _get_latest_version

    # Mock successful API response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"info": {"version": "1.2.3"}}
    ).encode()
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_response):
        version = _get_latest_version()
        assert version == "1.2.3"


if __name__ == "__main__":
    test_version_consistency()
    test_update_checker_works()
    test_pypi_api_call()
