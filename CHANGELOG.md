# Changelog

All notable changes to this project will be documented in this file.
## [0.3.3] - 2025-07-27
### Major changes
- Wheels for pyhon 3.13

## [0.3.2] - 2025-07-27

### Major changes
- Factors are now supported
- The workflow requires that the set_variable_types be set before the correlation and effects

## [0.3.1] - 2025-07-17

### Major changes
- Added weekly update checker

### Minor changes
- Small changes in documentation


## [0.3.0] - 2025-07-16

### Major changes
- AOT Compiled files added
- PyPI release

### Minor changes
- Small changes in documentation, added fallback to compilation


## [0.2.4] - 2025-07-16
- Test PyPI release

## [0.2.3] - 2025-07-16
- Test PyPI release

## [0.2.2] - 2025-07-11

### Major changes
- Moved seed setting from method arguments to `model.set_seed()` for better control
- Added public method to customize scenario configurations
- Improved lookup table performance with extended range and faster fallbacks
- Added support for 3-way and n-way interactions in formulas
- Fixed import error when uploading user data
- Added comprehensive test suite

### Minor changes
- Improved code documentation with clearer comments and global variables
- Clarified correlation behavior during variable transformations
- Fixed minor cache creation error
- Adjusted vulnerability thresholds from 30% to 20% power drop
- Added warning for inflated Type I error risk

## [0.2.1] - 2025-06-26

### Major changes
- Fixed import statements to use relative imports instead of absolute imports

### Minor changes
- Added changelog
- Extended docstrings

## [0.2.0] - 2025-06-26
- Initial release

## [0.1.0] - 2025-06-07
- Proof of concept release