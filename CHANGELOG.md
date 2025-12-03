# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Automatic cycle slip detection and correction in TEC calculation
- Add an extra feature 'custom-alloc' that can improve performance but uses more memory, which is disabled by default. Users have to enable it manually when building from source by adding `--features custom-alloc` flag to `maturin build` command.
- Add `station` parameter to `read_rinex_obs` function to allow custom station name assignment in case the RINEX header station name is not desired.

### Changed

- Change column names to lowercase for convenience
- Symplify function signatures. All functions now return LazyFrames, users can call `.collect()` to get DataFrames when needed.

### Fixed

- Improve memory efficiency in TEC calculation by avoiding unnecessary intermediate columns
- Improve performance of RINEX file reading

## [0.2.0](https://github.com/Eureka-0/pygnss-tec/releases/tag/v0.2.0) - 2025-11-28

### Added

- `calc_tec` function to calculate TEC from RINEX observation and navigation files
- Support using single layer model (SLM) to map slant TEC to vertical TEC
- Support DCB bias correction using external bias files

### Changed

- Add `pivot` parameter to `read_rinex_obs` function

## [0.1.0](https://github.com/Eureka-0/pygnss-tec/releases/tag/v0.1.0) - 2025-11-24

### Added

- Initial release of pygnss-tec
- `read_rinex_obs` function that supports reading observation RINEX files and automatically calculating azimuth and elevation angles when navigation RINEX files are provided
