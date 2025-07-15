#!/usr/bin/env python
"""
AOT compilation script for MCPower package.
Compiles Numba functions to native code for distribution.
"""

import os
import sys
import shutil
import glob
from pathlib import Path
from setuptools.command.build_py import build_py

MODULES_TO_COMPILE = [
    'ols',
    'data_generation',
]

def main():
    """Compile AOT modules."""    
    # Setup
    os.environ['NUMBA_AOT_BUILD'] = '1'
    script_dir = Path(__file__).parent
    
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))
    
    # Compile modules
    try:
        for module_name in MODULES_TO_COMPILE:
            __import__(f'mcpower.utils.{module_name}')
    except Exception as e:
        return 0
    
    # Clear environment variable
    del os.environ['NUMBA_AOT_BUILD']
    
    return 0


class BuildPyWithAOT(build_py):
    def run(self):
        main()
        super().run()


if __name__ == "__main__":
    sys.exit(main())