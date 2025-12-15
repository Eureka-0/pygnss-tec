# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]



## [0.4.0](https://github.com/Eureka-0/pygnss-tec/releases/tag/v0.4.0) - 2025-12-15

### Added

- Add modified single layer model (MSLM) as an additional mapping function option
- Support calculating TEC from RINEX v2 files

## [0.3.1](https://github.com/Eureka-0/pygnss-tec/releases/tag/v0.3.1) - 2025-12-10

### Fixed

- Update `rinex` crate to main branch to fix RINEX v3 Hatanaka compressed file reading issue ([#397](https://github.com/nav-solutions/rinex/issues/397))

## [0.3.0](https://github.com/Eureka-0/pygnss-tec/releases/tag/v0.3.0) - 2025-12-09

### Added

- Automatic cycle slip detection and correction in TEC calculation
- Add an extra feature 'custom-alloc' that can improve performance but uses more memory, which is disabled by default. Users have to enable it manually when building from source by adding `--features custom-alloc` flag to `maturin build` command.
- Add `station` parameter to `read_rinex_obs` function to allow custom station name assignment in case the RINEX header station name is not desired
- Add two receiver bias estimation methods: 'mstd' (minimum standard deviation) and 'lsq' (least squares)

### Changed

- Change column names to lowercase for convenience
- Symplify function signatures. All functions now return LazyFrames, users can call `.collect()` to get DataFrames when needed.
- All configurations are now handled by a single `TECConfig` dataclass
- Remove `pandas` dependency

### Fixed

- Improve performance and memory efficiency in TEC calculation
- Improve performance of RINEX file reading
- Fix observable columns missing when reading RINEX 2 files using `pivot=True`

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
