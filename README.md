# PyGNSS-TEC

[![PyPI - Version](https://img.shields.io/pypi/v/pygnss-tec)](https://pypi.org/project/pygnss-tec/)
![Supported Python Versions](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
[![Test](https://github.com/Eureka-0/pygnss-tec/actions/workflows/test.yml/badge.svg)](https://github.com/Eureka-0/pygnss-tec/actions/workflows/test.yml)

PyGNSS-TEC is a high-performance Python package leveraging Rust acceleration, designed for processing and analyzing Total Electron Content (TEC) data derived from Global Navigation Satellite System (GNSS) observations. The package provides tools for RINEX file reading, TEC calculation, and DCB correction to support ionospheric studies.

> **Warning**: This package is under active development and may undergo significant changes. It is not recommended for production use until it reaches a stable release (v1.0.0).

## Features

- **RINEX File Reading**: Efficient reading and parsing of RINEX GNSS observation files using [rinex crate](https://crates.io/crates/rinex) (see [benchmarks](#benchmarks) for details).

- **Multiple File Formats**: Support for RINEX versions 2.x and 3.x., as well as Hatanaka compressed files (e.g., .Z, .crx, .crx.gz).

- **TEC Calculation**: Efficiently compute TEC from dual-frequency GNSS observations using [polars](https://pola.rs/) DataFrames and lazy evaluation (see [benchmarks](#benchmarks) for details).

- **Multi-GNSS Support**: Process observations from multiple GNSS constellations (see [Overview](#overview) for constellation support).

- **Open-Source**: Fully open-source under the MIT License, encouraging community contributions and collaboration.

## Installation

### Via pip

You can install PyGNSS-TEC via pip:

```bash
pip install pygnss-tec
```

### Via uv (recommended)

[uv](https://docs.astral.sh/uv/) is a modern Python package and project manager written in Rust. You can add PyGNSS-TEC to your uv project with:

```bash
uv add pygnss-tec
```

### From Source

Building from source requires Rust and Cargo to be installed. Once you have both, run:

```bash
git clone https://github.com/Eureka-0/pygnss-tec.git
cd pygnss-tec
uv run maturin build --release

# Or enable custom memory allocator feature, which can improve performance in some scenarios (~10%) but may increase memory usage
uv run maturin build --release --features custom-alloc
```

The built package will be available in the `target/wheels` directory. You can then install it to your Python environment or uv project with:

```bash
# Using pip
pip install target/wheels/pygnss_tec-*.whl

# Or using uv
uv pip install target/wheels/pygnss_tec-*.whl
```

## Usage

### Overview

The following table summarizes the support for different GNSS constellations in PyGNSS-TEC:

| Constellation  | RINEX Reading  | TEC Calculation |
| -------------- | -------------- | --------------- |
| GPS (G)        | Yes            | Yes             |
| Beidou (C)     | Yes            | Yes             |
| Galileo (E)    | Yes            | No              |
| GLONASS (R)    | Yes            | No              |
| QZSS (J)       | Yes            | No              |
| IRNSS (I)      | Yes            | No              |
| SBAS (S)       | Yes            | No              |

### RINEX file reading

Read a RINEX observation file (supports RINEX v2.x, v3.x, and Hatanaka compressed files):

```python
import gnss_tec as gt

header, lf = gt.read_rinex_obs("./data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz")

# You can read multiple files from the same station by passing a list of file paths
# header, lf = gt.read_rinex_obs(["./data/file1.crx.gz", "./data/file2.crx.gz"])

# header is a dataclass containing RINEX file header information
print(header)
# RinexObsHeader(
#     version='3.04',
#     constellation='MIXED',
#     marker_name='CIBG',
#     marker_type='GEODETIC',
#     rx_ecef=(-1837003.1909, 6065631.1631, -716184.055),
#     rx_geodetic=(-6.490367937958374, 106.84916836419953, 173.0000212144293),
#     sampling_interval=30,
#     leap_seconds=18,
# )

# lf is a polars LazyFrame, you can collect it to get a DataFrame.
# By default, time is in UTC timezone. You can keep it in GPS time by passing `utc=False` to `read_rinex_obs`.
print(lf.collect())
# shape: (180_015, 71)
# ┌─────────────────────────┬─────────┬─────┬──────────┬──────┬──────────┬──────────┬──────┬───┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
# │ time                    ┆ station ┆ prn ┆ C1C      ┆ C1P  ┆ C1X      ┆ C1Z      ┆ C2C  ┆ … ┆ S5X  ┆ S6I  ┆ S6X  ┆ S7D  ┆ S7I  ┆ S7X  ┆ S8X  │
# │ ---                     ┆ ---     ┆ --- ┆ ---      ┆ ---  ┆ ---      ┆ ---      ┆ ---  ┆   ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
# │ datetime[ms, UTC]       ┆ cat     ┆ cat ┆ f64      ┆ f64  ┆ f64      ┆ f64      ┆ f64  ┆   ┆ f64  ┆ f64  ┆ f64  ┆ f64  ┆ f64  ┆ f64  ┆ f64  │
# ╞═════════════════════════╪═════════╪═════╪══════════╪══════╪══════════╪══════════╪══════╪═══╪══════╪══════╪══════╪══════╪══════╪══════╪══════╡
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C01 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ 42.9 ┆ null ┆ null ┆ 44.1 ┆ null ┆ null │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C02 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ 43.0 ┆ null ┆ null ┆ 46.2 ┆ null ┆ null │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C03 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ 44.5 ┆ null ┆ null ┆ 46.3 ┆ null ┆ null │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C04 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ 39.9 ┆ null ┆ null ┆ 42.1 ┆ null ┆ null │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C05 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ 41.1 ┆ null ┆ null ┆ 41.3 ┆ null ┆ null │
# │ …                       ┆ …       ┆ …   ┆ …        ┆ …    ┆ …        ┆ …        ┆ …    ┆ … ┆ …    ┆ …    ┆ …    ┆ …    ┆ …    ┆ …    ┆ …    │
# │ 2024-01-10 23:59:12 UTC ┆ CIBG    ┆ I06 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
# │ 2024-01-10 23:59:12 UTC ┆ CIBG    ┆ I09 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
# │ 2024-01-10 23:59:12 UTC ┆ CIBG    ┆ I10 ┆ null     ┆ null ┆ null     ┆ null     ┆ null ┆ … ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
# │ 2024-01-10 23:59:12 UTC ┆ CIBG    ┆ J02 ┆ 3.7775e7 ┆ null ┆ 3.7775e7 ┆ 3.7775e7 ┆ null ┆ … ┆ 47.5 ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
# │ 2024-01-10 23:59:12 UTC ┆ CIBG    ┆ J03 ┆ 3.4200e7 ┆ null ┆ 3.4200e7 ┆ 3.4200e7 ┆ null ┆ … ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
# └─────────────────────────┴─────────┴─────┴──────────┴──────┴──────────┴──────────┴──────┴───┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

If both observation and navigation files are provided, satellite azimuth and elevation angles will be calculated and included in the returned LazyFrame:

```python
header, lf = gt.read_rinex_obs(
    "./data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz",
    "./data/rinex_nav_v3/BRDC00IGS_R_20240100000_01D_MN.rnx.gz"
)
```

### TEC calculation

#### From RINEX files

Directly calculate from RINEX files using `calc_tec_from_rinex` function:

```python
tec_lf = gt.calc_tec_from_rinex(
    "./data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz",
    "./data/rinex_nav_v3/BRDC00IGS_R_20240100000_01D_MN.rnx.gz",
    "./data/bias/CAS0OPSRAP_20240100000_01D_01D_DCB.BIA.gz",  # Optional DCB file, can be omitted if DCB correction is not needed
)

print(tec_lf.collect())
# shape: (25_357, 12)
# ┌─────────────────────────┬─────────┬─────┬───────────┬────────────┬─────────┬─────────┬────────────┬────────────┬─────────────┬────────────────────┬───────────┐
# │ time                    ┆ station ┆ prn ┆ rx_lat    ┆ rx_lon     ┆ C1_code ┆ C2_code ┆ ipp_lat    ┆ ipp_lon    ┆ stec        ┆ stec_dcb_corrected ┆ vtec      │
# │ ---                     ┆ ---     ┆ --- ┆ ---       ┆ ---        ┆ ---     ┆ ---     ┆ ---        ┆ ---        ┆ ---         ┆ ---                ┆ ---       │
# │ datetime[ms, UTC]       ┆ cat     ┆ cat ┆ f32       ┆ f32        ┆ cat     ┆ cat     ┆ f32        ┆ f32        ┆ f64         ┆ f64                ┆ f64       │
# ╞═════════════════════════╪═════════╪═════╪═══════════╪════════════╪═════════╪═════════╪════════════╪════════════╪═════════════╪════════════════════╪═══════════╡
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C01 ┆ -6.490368 ┆ 106.849167 ┆ C2I     ┆ C6I     ┆ -6.021448  ┆ 110.041397 ┆ -64.574271  ┆ 37.972972          ┆ 28.641702 │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C02 ┆ -6.490368 ┆ 106.849167 ┆ C2I     ┆ C6I     ┆ -5.874807  ┆ 105.12574  ┆ -92.516869  ┆ 30.728694          ┆ 27.456595 │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C03 ┆ -6.490368 ┆ 106.849167 ┆ C2I     ┆ C6I     ┆ -5.987422  ┆ 107.10218  ┆ -100.530353 ┆ 27.26433           ┆ 26.936984 │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C05 ┆ -6.490368 ┆ 106.849167 ┆ C2I     ┆ C6I     ┆ -5.77573   ┆ 102.127449 ┆ -80.226673  ┆ 37.913022          ┆ 23.666211 │
# │ 2024-01-09 23:59:42 UTC ┆ CIBG    ┆ C06 ┆ -6.490368 ┆ 106.849167 ┆ C2I     ┆ C6I     ┆ -2.625823  ┆ 105.790565 ┆ -117.679878 ┆ 30.499514          ┆ 20.802594 │
# │ …                       ┆ …       ┆ …   ┆ …         ┆ …          ┆ …       ┆ …       ┆ …          ┆ …          ┆ …           ┆ …                  ┆ …         │
# │ 2024-01-10 12:33:42 UTC ┆ CIBG    ┆ G07 ┆ -6.490368 ┆ 106.849167 ┆ C1C     ┆ C2W     ┆ -11.208825 ┆ 108.190903 ┆ 112.260893  ┆ 67.006327          ┆ 41.107371 │
# │ 2024-01-10 12:33:42 UTC ┆ CIBG    ┆ G11 ┆ -6.490368 ┆ 106.849167 ┆ C1C     ┆ C2W     ┆ -7.12187   ┆ 102.291412 ┆ 113.764545  ┆ 62.884908          ┆ 40.079394 │
# │ 2024-01-10 12:33:42 UTC ┆ CIBG    ┆ G14 ┆ -6.490368 ┆ 106.849167 ┆ C1C     ┆ C2W     ┆ -4.726127  ┆ 107.7995   ┆ 92.510418   ┆ 39.972655          ┆ 35.000992 │
# │ 2024-01-10 12:33:42 UTC ┆ CIBG    ┆ G22 ┆ -6.490368 ┆ 106.849167 ┆ C1C     ┆ C2W     ┆ -2.877814  ┆ 106.910889 ┆ 87.070455   ┆ 44.615582          ┆ 31.961073 │
# │ 2024-01-10 12:33:42 UTC ┆ CIBG    ┆ G30 ┆ -6.490368 ┆ 106.849167 ┆ C1C     ┆ C2W     ┆ -8.699801  ┆ 105.905724 ┆ 125.373327  ┆ 55.229748          ┆ 46.131525 │
# └─────────────────────────┴─────────┴─────┴───────────┴────────────┴─────────┴─────────┴────────────┴────────────┴─────────────┴────────────────────┴───────────┘
```

#### From DataFrame or LazyFrame

If you wish to calculate TEC from an existing polars DataFrame or LazyFrame (e.g., after some custom preprocessing), you can use the `calc_tec_from_df` function:

```python
header, lf = gt.read_rinex_obs(
    "./data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz",
    "./data/rinex_nav_v3/BRDC00IGS_R_20240100000_01D_MN.rnx.gz"
)

# ...
# Perform any custom preprocessing on lf if needed
# ...

tec_lf = gt.calc_tec_from_df(lf, header, "./data/bias/CAS0OPSRAP_20240100000_01D_01D_DCB.BIA.gz")
```

#### From parquet file

Reading RINEX files are time-consuming, accounting for at least 80% of the total processing time. Thus, if you need to perform TEC calculation multiple times on the same RINEX files (e.g., when tuning configuration), it is recommended to save the parsed LazyFrame to a parquet file after the first read, and then use `calc_tec_from_parquet` for subsequent TEC calculations:

```python
header, lf = gt.read_rinex_obs(
    "./data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz",
    "./data/rinex_nav_v3/BRDC00IGS_R_20240100000_01D_MN.rnx.gz"
)

# ...
# Perform any custom preprocessing on lf if needed
# ...

# Note: Make sure to include header information when saving to parquet
lf.sink_parquet("./data/cibg_obs_2024010.parquet", metadata=header.to_metadata())

tec_lf = gt.calc_tec_from_parquet(
    "./data/cibg_obs_2024010.parquet",
    "./data/bias/CAS0OPSRAP_20240100000_01D_01D_DCB.BIA.gz"
)
```

#### Configuration

You can customize the TEC calculation process using the `TecConfig` dataclass:

```python
# To see the default configuration
print(gt.TecConfig())
# TECConfig(
#     constellations='CG',
#     ipp_height=400,
#     min_elevation=30.0,
#     min_snr=30.0,
#     c1_codes={
#         'C': ['C2I', 'C2D', 'C2X', 'C1I', 'C1D', 'C1X', 'C2W', 'C1C'],
#         'G': ['C1W', 'C1C', 'C1X']
#     },
#     c2_codes={
#         'C': ['C6I', 'C6D', 'C6X', 'C7I', 'C7D', 'C7X', 'C5I', 'C5D', 'C5X'],
#         'G': ['C2W', 'C2C', 'C2X', 'C5W', 'C5C', 'C5X']
#     },
#     rx_bias='external',
#     retain_intermediate=None
# )
```

The meaning of each parameter is as follows:
- `constellations`: A string specifying which GNSS constellations to consider for TEC calculation. 'C' for Beidou, 'G' for GPS.
- `ipp_height`: The assumed height of the ionospheric pierce point (IPP) in kilometers.
- `min_elevation`: The minimum satellite elevation angle (in degrees) for observations to be considered in the TEC calculation.
- `min_snr`: The minimum signal-to-noise ratio (in dB-Hz) for observations to be considered in the TEC calculation.
- `c1_codes`: A dictionary specifying the preferred observation codes for the first frequency (C1) for each constellation. The codes are prioritized in the order they are listed, with the first available code being used.
- `c2_codes`: A dictionary specifying the preferred observation codes for the second frequency (C2) for each constellation. The codes are prioritized in the order they are listed, with the first available code being used.
- `rx_bias`: Specifies how to handle receiver bias. It can be set to 'external' to use an external DCB file for correction, 'mstd' to use the minimum standard deviation method for estimation, 'lsq' to use least squares estimation, or `None` to skip receiver bias correction.
- `retain_intermediate`: Names of intermediate columns to retain in the output DataFrame. It can be set to `None` to discard all intermediate columns, 'all' to retain all intermediate columns, or a list of column names to keep specific ones.

## Benchmarks
