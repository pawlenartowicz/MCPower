#!/usr/bin/env python
"""
Setup script for MCPower with AOT compilation support.
Forces wheel to be platform-specific by declaring extensions.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext


class BuildWithAOT(build_py):
    """Custom build command that runs AOT compilation."""

    def run(self):
        # Run AOT compilation before regular build
        print("Running AOT compilation...")

        # Set environment variable
        os.environ["NUMBA_AOT_BUILD"] = "1"

        try:
            # Run AOT compilation
            result = subprocess.run(
                [sys.executable, "build_aot.py"], capture_output=True, text=True
            )

            if result.returncode == 0:
                print("✓ AOT compilation successful")
                print(result.stdout)
            else:
                print("✗ AOT compilation failed:")
                print(result.stdout)
                print(result.stderr)
                # Don't fail the build - fall back to JIT

        except Exception as e:
            print(f"AOT compilation error: {e}")
            # Don't fail the build - fall back to JIT

        finally:
            # Clean up environment
            if "NUMBA_AOT_BUILD" in os.environ:
                del os.environ["NUMBA_AOT_BUILD"]

        # Continue with regular build
        super().run()


class CustomBuildExt(build_ext):
    """Custom build_ext that doesn't actually build anything but signals non-pure wheel."""

    def build_extensions(self):
        # Don't actually build anything - AOT files are already created
        pass

    def run(self):
        # Just run the base, but don't build extensions
        pass


def get_compiled_extensions():
    """Find compiled extensions to include in package data."""
    utils_dir = Path("mcpower/utils")
    if not utils_dir.exists():
        return []

    extensions = []
    for pattern in ["*.so", "*.pyd", "*.dll", "*_compiled*"]:
        extensions.extend(utils_dir.glob(pattern))

    return [str(f.relative_to(utils_dir)) for f in extensions]


# Read long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Monte Carlo Power Analysis for Statistical Models"

# Create dummy extension to force platform-specific wheel
# This gets replaced by numba-compiled files
dummy_extension = Extension(
    name="mcpower.utils._dummy",
    sources=[],  # No sources - we'll use AOT compiled files
    optional=True,
)

setup(
    name="MCPower",
    version="0.3.3",
    description="Monte Carlo Power Analysis for Statistical Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paweł Lenartowicz",
    author_email="pawellenartowicz@europe.com",
    url="https://github.com/pawlenartowicz/MCPower",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "JIT": ["numba>=0.61.0"],
        "parallel": ["joblib>=1.3.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "build>=0.10.0", "twine>=4.0.0"],
        "all": ["joblib>=1.3.0", "pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    package_data={
        "mcpower.utils": ["*.so", "*.pyd", "*.dll", "*_compiled*", "*.c", "*.h"],
    },
    include_package_data=True,
    zip_safe=False,
    # This is the key: declare extensions to force platform-specific wheel
    ext_modules=[dummy_extension],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords=["power analysis", "statistics", "monte carlo", "linear regression"],
    cmdclass={
        "build_py": BuildWithAOT,
        "build_ext": CustomBuildExt,
    },
)
