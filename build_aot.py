#!/usr/bin/env python
"""
AOT compilation script for MCPower package.
Compiles Numba functions to native code for distribution.
"""

import os
import sys
from pathlib import Path

def main():
    """Compile AOT modules."""
    # Set AOT build flag
    os.environ['NUMBA_AOT_BUILD'] = '1'
    
    # Change to script directory and add to path
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))
    
    try:
        # Import modules to trigger AOT compilation
        import mcpower.utils.ols
        import mcpower.utils.data_generation            
    except Exception as e:
        print(f"AOT compilation failed: {e}")
        return 1
    finally:
        # Clean up environment
        if 'NUMBA_AOT_BUILD' in os.environ:
            del os.environ['NUMBA_AOT_BUILD']
    
    return 0

if __name__ == "__main__":
    sys.exit(main())